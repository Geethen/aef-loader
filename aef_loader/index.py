"""
AEF Index management - download and filter the tile index.

Uses obstore for efficient GCS and S3 access. Only the columns needed for
searching are downloaded (ranged reads of the remote parquet); the footprint
geometry is fetched separately and only for ``search(..., exact=True)``.
Supports both Google Cloud Storage (GCS) and Source Cooperative (S3) backends.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import obstore as obs
import pandas as pd
from obstore.store import GCSStore, S3Store

from aef_loader.constants import (
    GCS_BUCKET,
    GCS_INDEX_BLOB,
    SOURCE_COOP_BUCKET,
    SOURCE_COOP_INDEX_BLOB,
    SOURCE_COOP_REGION,
    DataSource,
)
from aef_loader.types import (
    AEFTileInfo,
    BoundingBox,
    DateRange,
)

if TYPE_CHECKING:
    import pyarrow as pa

logger = logging.getLogger(__name__)

_YEAR_RE = re.compile(r"^\d{4}(-\d{2}-\d{2})?$")

# Columns needed to search and build AEFTileInfo (the ``geom`` column is ~80% of
# the file and is only fetched for ``exact=True``).
_REQUIRED_COLUMNS = (
    "crs",
    "path",
    "year",
    "wgs84_west",
    "wgs84_south",
    "wgs84_east",
    "wgs84_north",
)
_OPTIONAL_COLUMNS = (
    "fid",
    "utm_zone",
    "utm_west",
    "utm_south",
    "utm_east",
    "utm_north",
)


def _import_shapely():
    """Import shapely for ``exact=True``, with an actionable error if missing."""
    try:
        import shapely
    except ImportError as exc:
        raise ImportError(
            "exact=True requires shapely; install it with "
            'pip install "aef-loader-plus[exact]"'
        ) from exc
    return shapely


def _default_cache_dir() -> Path:
    """Per-user cache directory (platformdirs if installed, else ``~/.cache``)."""
    try:
        from platformdirs import user_cache_dir

        return Path(user_cache_dir("aef-loader"))
    except ImportError:
        return Path.home() / ".cache" / "aef-loader"


def _parse_epsg(crs: object) -> int:
    """EPSG code from ``"EPSG:32633"``, ``"epsg:32633"`` or bare ``"32633"``."""
    text = str(crs).strip()
    code = text.split(":", 1)[1] if text.upper().startswith("EPSG:") else text
    if not code.isdigit():
        raise ValueError(
            f"unrecognised CRS {crs!r} in index; expected 'EPSG:<code>' or '<code>'"
        )
    return int(code)


class _ObstoreRangeFile(io.RawIOBase):
    """Read-only seekable file over an obstore object, using ranged reads.

    Lets ``pyarrow.parquet.ParquetFile`` read the footer and individual column
    chunks without downloading the whole file. The size comes from ``obs.head``;
    every ``read`` issues one ``obs.get_range``.
    """

    def __init__(self, store, path: str):
        super().__init__()
        self._store = store
        self._path = path
        self._size = int(obs.head(store, path)["size"])
        self._pos = 0

    @property
    def size(self) -> int:
        return self._size

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self._size + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        self._pos = max(self._pos, 0)
        return self._pos

    def readinto(self, buffer) -> int:
        start = self._pos
        end = min(start + len(buffer), self._size)
        if end <= start:
            return 0
        data = bytes(obs.get_range(self._store, self._path, start=start, end=end))
        buffer[: len(data)] = data
        self._pos = start + len(data)
        return len(data)


def _geom_column_name(schema: pa.Schema) -> str:
    """Name of the geometry column (GeoParquet metadata, else ``geometry``/``geom``)."""
    meta = (schema.metadata or {}).get(b"geo")
    if meta:
        primary = json.loads(meta).get("primary_column")
        if primary in schema.names:
            return primary
    for name in ("geometry", "geom"):
        if name in schema.names:
            return name
    raise ValueError("index parquet has no geometry column")


def _write_table_atomic(table: pa.Table, path: Path) -> None:
    """Write ``table`` to ``path`` via a temp file so readers never see a partial file."""
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        pq.write_table(table, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _as_year(value: Any) -> int:
    """``2024``, ``np.int64(2024)``, ``"2024"`` or ``"2024-05-10"`` -> ``2024``."""
    if isinstance(value, str):
        if not _YEAR_RE.match(value.strip()):
            raise ValueError(
                f"invalid year string: {value!r} (expected 'YYYY' or 'YYYY-MM-DD')"
            )
        return int(value.strip()[:4])
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"invalid year: {value!r}")
    return int(value)

def selected_years(years: Any) -> list[int]:
    """Years selected by ``years``, using the same rules as the extract API.

    A single year (int, numpy integer or ``"YYYY[-MM-DD]"`` string); a 2-tuple
    ``(start, end)`` for an inclusive range; or any other iterable as an
    explicit list of years.
    """
    if isinstance(years, (int, np.integer, str)):
        return [_as_year(years)]
    if isinstance(years, tuple):
        if len(years) != 2:
            raise ValueError(
                f"years tuple must be (start, end), got length {len(years)}; "
                "pass a list for an explicit set of years"
            )
        start, end = _as_year(years[0]), _as_year(years[1])
        if end < start:
            raise ValueError(f"years range {years!r} ends before it starts")
        return list(range(start, end + 1))
    try:
        selected = sorted({_as_year(y) for y in years})
    except TypeError:
        raise ValueError(f"invalid years: {years!r}") from None
    if not selected:
        raise ValueError("years must not be empty")
    return selected


class AEFIndex:
    """
    Manages the AEF tile index for efficient spatial/temporal queries.

    The index contains metadata about all AEF tiles including their
    bounding boxes and paths. It is loaded as a plain pandas DataFrame and
    filtered by WGS84 bbox overlap; ``exact=True`` refines with the true
    footprint geometry (requires shapely: ``pip install "aef-loader-plus[exact]"``).

    Supports both GCS (Google Cloud Storage) and Source Cooperative (AWS S3) backends.

    Example:
        GCS (requires GCP project for requester-pays):

        ```python
        index = AEFIndex(source=DataSource.GCS, gcp_project="my-project")
        await index.download()
        tiles = index.search(bbox=(-122.5, 37.5, -122.0, 38.0), years=(2020, 2023))
        ```

        Source Cooperative (public, no auth required):

        ```python
        index = AEFIndex(source=DataSource.SOURCE_COOP)
        await index.download()
        tiles = index.search(bbox=(-122.5, 37.5, -122.0, 38.0), years=(2020, 2023))
        ```
    """

    def __init__(
        self,
        source: DataSource | str = DataSource.SOURCE_COOP,
        gcp_project: str | None = None,
        cache_dir: Path | str | None = None,
    ):
        """
        Initialize AEF index manager.

        Args:
            source: Data source: ``DataSource.SOURCE_COOP`` (default, public) or
                ``DataSource.GCS`` (requester pays); the strings
                ``"source_coop"``/``"gcs"`` are accepted case-insensitively
            gcp_project: GCP project ID for requester-pays bucket access (GCS only)
            cache_dir: Directory for caching the index (default: the per-user
                cache directory, ``platformdirs.user_cache_dir("aef-loader")``
                when available, else ``~/.cache/aef-loader``)
        """
        self.source = DataSource(source)
        self.gcp_project = gcp_project
        self.cache_dir = Path(cache_dir) if cache_dir else _default_cache_dir()
        self._df: pd.DataFrame | None = None
        self._geoms = None  # shapely geometry array, fetched lazily for exact=True
        self._index_path: Path | None = None

    @property
    def _gdf(self) -> pd.DataFrame | None:
        """Alias of the loaded table (kept so a GeoDataFrame can be assigned)."""
        return self._df

    @_gdf.setter
    def _gdf(self, value: pd.DataFrame | None) -> None:
        self._df = value
        self._geoms = None

    @property
    def _cache_filename(self) -> str:
        """Get cache filename based on data source (``v2`` = column-slim index)."""
        if self.source == DataSource.SOURCE_COOP:
            return "aef_index_source_coop.v2.parquet"
        return "aef_index_gcs.v2.parquet"

    @property
    def _geom_cache_filename(self) -> str:
        """Cache filename of the separately fetched footprint geometry column."""
        return self._cache_filename.replace(".v2.parquet", ".v2.geom.parquet")

    @property
    def _bucket(self) -> str:
        """Get bucket name based on data source."""
        if self.source == DataSource.SOURCE_COOP:
            return SOURCE_COOP_BUCKET
        return GCS_BUCKET

    @property
    def _index_blob(self) -> str:
        """Get index blob path based on data source."""
        if self.source == DataSource.SOURCE_COOP:
            return SOURCE_COOP_INDEX_BLOB
        return GCS_INDEX_BLOB

    def _make_store(self):
        """Build the obstore store for the configured source."""
        if self.source == DataSource.SOURCE_COOP:
            return S3Store(
                bucket=self._bucket,
                region=SOURCE_COOP_REGION,
                skip_signature=True,  # Public bucket, no auth needed
            )
        # GCS - requires project for requester-pays
        if not self.gcp_project:
            raise ValueError(
                "gcp_project is required for downloading from GCS requester-pays bucket"
            )
        return GCSStore(
            bucket=self._bucket,
            client_options={"default_headers": {"x-goog-user-project": self.gcp_project}},
        )

    def _read_remote_columns(self, columns: list[str] | None = None) -> pa.Table:
        """Read columns of the remote index with ranged reads.

        ``columns=None`` selects the slim search columns present in the file;
        otherwise exactly the given columns are read.
        """
        import pyarrow.parquet as pq

        store = self._make_store()
        source = _ObstoreRangeFile(store, self._index_blob)
        parquet_file = pq.ParquetFile(source)
        if columns is None:
            names = parquet_file.schema_arrow.names
            missing = [c for c in _REQUIRED_COLUMNS if c not in names]
            if missing:
                raise ValueError(f"index parquet is missing required columns: {missing}")
            columns = [c for c in (*_OPTIONAL_COLUMNS, *_REQUIRED_COLUMNS) if c in names]
        return parquet_file.read(columns=columns)

    async def download(
        self,
        force: bool = False,
        local_path: Path | None = None,
    ) -> Path:
        """
        Download the slim AEF index (search columns only) using ranged reads.

        The result is written atomically to ``aef_index_<source>.v2.parquet``.

        Args:
            force: Force re-download even if cached (also clears the loaded table)
            local_path: Custom path for the index file

        Returns:
            Path to the downloaded index file
        """
        if local_path is None:
            local_path = self.cache_dir / self._cache_filename

        if force:
            self._df = None
            self._geoms = None

        if local_path.exists() and not force:
            logger.info(f"Using cached AEF index at {local_path}")
            self._index_path = local_path
            return local_path

        logger.info(f"Downloading AEF index from {self._bucket}/{self._index_blob}")
        table = await asyncio.to_thread(self._read_remote_columns)
        await asyncio.to_thread(_write_table_atomic, table, local_path)
        logger.info(f"Downloaded AEF index to {local_path}")

        self._index_path = local_path
        return local_path

    def load(self, path: Path | None = None) -> pd.DataFrame:
        """
        Load the index into memory as a plain pandas DataFrame (no geometry).

        Args:
            path: Path to index file (uses cached path if not provided)

        Returns:
            DataFrame with AEF tile metadata
        """
        if path is None:
            path = self._index_path
        if path is None:
            path = self.cache_dir / self._cache_filename

        if not path.exists():
            raise FileNotFoundError(
                f"Index not found at {path}. Call download() first."
            )

        logger.info(f"Loading AEF index from {path}")
        self._df = pd.read_parquet(path)
        self._geoms = None
        logger.info(f"Loaded {len(self._df)} tiles from AEF index")
        return self._df

    def _load_geoms(self):
        """Footprint geometries in index row order, fetched once and cached on disk."""
        if self._geoms is not None:
            return self._geoms
        shapely = _import_shapely()

        df = self._df
        if df is not None and "geometry" in df.columns:  # e.g. an assigned GeoDataFrame
            self._geoms = np.asarray(df["geometry"].values)
            return self._geoms

        import pyarrow.parquet as pq

        geom_path = self.cache_dir / self._geom_cache_filename
        if not geom_path.exists():
            store = self._make_store()
            remote = pq.ParquetFile(_ObstoreRangeFile(store, self._index_blob))
            column = _geom_column_name(remote.schema_arrow)
            logger.info(f"Fetching index geometry column '{column}' (large download)")
            _write_table_atomic(remote.read(columns=[column]), geom_path)
        table = pq.read_table(geom_path)
        wkb = table.column(0).to_numpy(zero_copy_only=False)
        self._geoms = shapely.from_wkb(wkb)
        return self._geoms

    def _selected_years(self, years: Any) -> list[int]:
        return selected_years(years)

    def _get_start_and_end_year(self, years: Any) -> tuple[int, int]:
        """Inclusive (first, last) year selected by ``years``."""
        selected = self._selected_years(years)
        return selected[0], selected[-1]

    @staticmethod
    def _bbox_to_wgs84(bbox: BoundingBox, bbox_crs: str | int) -> BoundingBox:
        """Reproject ``bbox`` from ``bbox_crs`` to WGS84 with edge densification.

        Returns the bbox unchanged when ``bbox_crs`` is already WGS84. Otherwise
        uses ``pyproj.Transformer.transform_bounds(densify_pts=21)`` so the
        WGS84 envelope covers the projected rectangle's outward-bowed edges (see
        ``search`` note) rather than just its four corners.
        """
        from pyproj import CRS, Transformer

        if CRS.from_user_input(bbox_crs) == CRS.from_epsg(4326):
            return bbox
        transformer = Transformer.from_crs(bbox_crs, "EPSG:4326", always_xy=True)
        minx, miny, maxx, maxy = transformer.transform_bounds(
            bbox[0], bbox[1], bbox[2], bbox[3], densify_pts=21
        )
        return (minx, miny, maxx, maxy)

    async def query(
        self,
        bbox: BoundingBox | None = None,
        years: int | str | DateRange | list[int] | None = None,
        limit: int | None = None,
        bbox_crs: str | int = "EPSG:4326",
        exact: bool = False,
    ) -> list[AEFTileInfo]:
        """Async wrapper around :meth:`search` (kept for upstream compatibility)."""
        return await asyncio.to_thread(self.search, bbox, years, limit, bbox_crs, exact)

    def search(
        self,
        bbox: BoundingBox | None = None,
        years: int | str | DateRange | list[int] | None = None,
        limit: int | None = None,
        bbox_crs: str | int = "EPSG:4326",
        exact: bool = False,
    ) -> list[AEFTileInfo]:
        """
        Search the index for tiles matching the given criteria.

        Args:
            bbox: Bounding box filter (minx, miny, maxx, maxy) in ``bbox_crs``.
            years: Single year or (start_year, end_year) tuple
            limit: Maximum number of tiles to return (``None`` or >= 0)
            bbox_crs: CRS of ``bbox``. Defaults to WGS84. Pass a projected CRS
                (e.g. ``"EPSG:32633"``) to supply the AOI in projected
                coordinates; the bbox is reprojected to WGS84 with edge
                densification before filtering the index (see note).
            exact: By default tiles are selected by overlap of their WGS84 bbox
                with the AOI, which may include a few edge tiles whose footprint
                does not touch it (the reader crops those to nothing). With
                ``exact=True`` the footprint geometry is fetched once (large,
                cached) and candidates are refined with a true intersection;
                requires shapely.

        Returns:
            List of AEFTileInfo objects matching the query, in index row order

        Note:
            When ``bbox_crs`` is projected, the bbox edges are densified
            (``Transformer.transform_bounds(densify_pts=21)``) before reprojecting
            to WGS84. Transforming only the four corners of a projected rectangle
            under-covers the true footprint: at high latitudes the reprojected
            edges bow outward, so a corner-only envelope can miss boundary tiles.
            Densifying samples along each edge and takes the outer envelope.
        """
        if limit is not None and limit < 0:
            raise ValueError("limit must be None or >= 0")
        if self._df is None:
            self.load()

        assert self._df is not None, "Index not loaded"
        df = self._df
        # Boolean masks over numpy arrays; the frame is never copied or mutated.
        keep = np.ones(len(df), dtype=bool)

        if bbox:
            minx, miny, maxx, maxy = self._bbox_to_wgs84(bbox, bbox_crs)
            keep &= df["wgs84_west"].to_numpy() <= maxx
            keep &= df["wgs84_east"].to_numpy() >= minx
            keep &= df["wgs84_south"].to_numpy() <= maxy
            keep &= df["wgs84_north"].to_numpy() >= miny
            logger.info(f"After bbox filter: {int(keep.sum())} tiles")

        if years is not None:
            year = df["year"].to_numpy()
            keep &= np.isin(year, self._selected_years(years))
            logger.info(f"After year filter: {int(keep.sum())} tiles")

        positions = np.flatnonzero(keep)  # ascending == file row order

        if exact and bbox and len(positions):
            shapely = _import_shapely()

            geoms = self._load_geoms()
            hit = shapely.intersects(
                geoms[positions], shapely.box(minx, miny, maxx, maxy)
            )
            positions = positions[hit]
            logger.info(f"After exact filter: {len(positions)} tiles")

        if limit is not None:
            positions = positions[:limit]

        if len(positions) == 0:
            return []

        gdf = df.iloc[positions]

        # itertuples avoids constructing a pandas Series for every result.
        # Build a column-position map so optional columns remain optional and
        # pandas tuple field-renaming rules do not matter.
        columns = list(gdf.columns)
        column_pos = {name: pos + 1 for pos, name in enumerate(columns)}
        has_fid = "fid" in column_pos
        has_utm_zone = "utm_zone" in column_pos
        has_utm_bounds = "utm_west" in column_pos

        def value(row: tuple, name: str):
            return row[column_pos[name]]

        tiles: list[AEFTileInfo] = []
        for row in gdf.itertuples(index=True, name=None):
            tiles.append(
                AEFTileInfo(
                    id=str(value(row, "fid") if has_fid else row[0]),
                    path=value(row, "path"),
                    year=value(row, "year"),
                    bbox=(
                        value(row, "wgs84_west"),
                        value(row, "wgs84_south"),
                        value(row, "wgs84_east"),
                        value(row, "wgs84_north"),
                    ),
                    crs_epsg=_parse_epsg(value(row, "crs")),
                    utm_zone=value(row, "utm_zone") if has_utm_zone else None,
                    utm_bounds=(
                        (
                            value(row, "utm_west"),
                            value(row, "utm_south"),
                            value(row, "utm_east"),
                            value(row, "utm_north"),
                        )
                        if has_utm_bounds
                        else None
                    ),
                    source=self.source,
                )
            )

        return tiles
