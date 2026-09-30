"""
Point and zonal extraction of AEF embeddings.

Both functions work on each tile's NATIVE grid: nothing is reprojected or
mosaicked, so the values are exactly the stored int8 codes (dequantized only
afterwards, and always dequantized BEFORE any averaging, because the code ->
value mapping is nonlinear).

Conventions
-----------
* Pixels are half-open cells. A point exactly on the shared edge of two pixels
  belongs to the pixel to its right / below it (``floor`` of the fractional
  pixel coordinate, after snapping to 1e-6 of a pixel to absorb float noise).
* A tile covers ``west <= x < east`` and ``south < y <= north`` in its native
  CRS, so abutting tiles never both claim a point.
* Where tiles of the same year overlap (adjacent UTM zones), the tile in which
  the location is deepest inside its native bounds wins (ties: first in the
  index order). The same rule de-duplicates zonal pixels so a pixel covered by
  two zones is counted once.

The core functions take already opened tile datasets (``embeddings`` with
``band, y, x`` dims and optionally a length-1 ``time``), so they can be tested
without any network access; the public functions add the index search and the
lazy tile opening.
"""

from __future__ import annotations

import asyncio
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import dask
import numpy as np
import pandas as pd
import xarray as xr
from affine import Affine

from aef_loader.api import ensure_index_loaded, get_shared_index, run_sync
from aef_loader.collection import AEF
from aef_loader.constants import AEF_NODATA_VALUE, DataSource
from aef_loader.index import AEFIndex
from aef_loader.reader import VirtualTiffReader
from aef_loader.types import AEFTileInfo
from aef_loader.utils import dequantize_aef

STATS = ("mean", "median", "std", "min", "max", "count")
_PIXEL_SNAP = 6  # decimals of a pixel used to snap coordinates onto pixel edges
_OPEN_CONCURRENCY = 16


# ---------------------------------------------------------------------------
# input normalisation
# ---------------------------------------------------------------------------


def _normalise_years(years: Any) -> list[int]:
    """``2024``, ``(2020, 2022)`` (inclusive range) or ``[2020, 2022]`` (explicit list)."""

    def as_year(value: Any) -> int:
        return int(str(value)[:4]) if isinstance(value, str) else int(value)

    if isinstance(years, (int, np.integer, str)):
        return [as_year(years)]
    if isinstance(years, tuple) and len(years) == 2:
        start, end = as_year(years[0]), as_year(years[1])
        if end < start:
            raise ValueError(f"years range {years!r} ends before it starts")
        return list(range(start, end + 1))
    return sorted({as_year(y) for y in years})


def _import_geopandas_type():
    try:
        import geopandas as gpd
    except ImportError:
        return None
    return gpd.GeoDataFrame


def _normalise_points(points: Any, crs: str, point_id: Sequence | None):
    """-> (x, y, ids, crs) with float64 x/y arrays."""
    gdf_type = _import_geopandas_type() if hasattr(points, "geometry") else None
    if gdf_type is not None and isinstance(points, gdf_type):
        geom = points.geometry
        if not (geom.geom_type == "Point").all():
            raise ValueError("points GeoDataFrame must contain only Point geometries")
        x = geom.x.to_numpy(dtype="float64")
        y = geom.y.to_numpy(dtype="float64")
        if points.crs is not None:
            crs = points.crs.to_string()
        ids = list(points.index) if point_id is None else list(point_id)
    else:
        arr = np.asarray(points, dtype="float64")
        if arr.size == 0:
            arr = arr.reshape(0, 2)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("points must be a sequence of (x, y) pairs or a GeoDataFrame")
        x, y = arr[:, 0], arr[:, 1]
        ids = list(range(len(x))) if point_id is None else list(point_id)
    if len(ids) != len(x):
        raise ValueError(f"point_id has {len(ids)} entries for {len(x)} points")
    return x, y, ids, crs


def _import_shapely():
    try:
        import shapely
    except ImportError as exc:
        raise ImportError(
            "extract_zonal requires shapely; install it with "
            'pip install "aef-loader-plus[exact]"'
        ) from exc
    return shapely


def _normalise_polygons(polygons: Any, crs: str, polygon_id: Sequence | None):
    """-> (geometries, ids, crs) with shapely geometries."""
    shapely = _import_shapely()
    gdf_type = _import_geopandas_type() if hasattr(polygons, "geometry") else None
    if gdf_type is not None and isinstance(polygons, gdf_type):
        geoms = list(polygons.geometry.values)
        if polygons.crs is not None:
            crs = polygons.crs.to_string()
        ids = list(polygons.index) if polygon_id is None else list(polygon_id)
    else:
        if hasattr(polygons, "geom_type"):  # a single geometry
            polygons = [polygons]
        geoms = [
            g if hasattr(g, "geom_type") else shapely.geometry.shape(g) for g in polygons
        ]
        ids = list(range(len(geoms))) if polygon_id is None else list(polygon_id)
    if len(ids) != len(geoms):
        raise ValueError(f"polygon_id has {len(ids)} entries for {len(geoms)} polygons")
    for geom in geoms:
        if geom.geom_type not in ("Polygon", "MultiPolygon"):
            raise ValueError(f"expected Polygon or MultiPolygon, got {geom.geom_type}")
    return geoms, ids, crs


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------


_TRANSFORMERS: dict[tuple[str, int], Any] = {}


def _transformer(src_crs: str, dst_epsg: int):
    from pyproj import CRS, Transformer

    key = (str(src_crs), dst_epsg)
    if key not in _TRANSFORMERS:
        src, dst = CRS.from_user_input(src_crs), CRS.from_epsg(dst_epsg)
        _TRANSFORMERS[key] = None if src == dst else Transformer.from_crs(src, dst, always_xy=True)
    return _TRANSFORMERS[key]


def _project_xy(x: np.ndarray, y: np.ndarray, src_crs: str, dst_epsg: int):
    tr = _transformer(src_crs, dst_epsg)
    if tr is None:
        return x, y
    with np.errstate(all="ignore"):
        px, py = tr.transform(x, y)
    return np.asarray(px, dtype="float64"), np.asarray(py, dtype="float64")


def _project_geometry(geom, src_crs: str, dst_epsg: int):
    """Reproject a shapely geometry, densifying edges first so they follow the curve."""
    shapely = _import_shapely()
    tr = _transformer(src_crs, dst_epsg)
    if tr is None:
        return geom
    minx, miny, maxx, maxy = geom.bounds
    extent = max(maxx - minx, maxy - miny)
    if extent > 0:
        geom = shapely.segmentize(geom, extent / 100)

    def func(coords: np.ndarray) -> np.ndarray:
        px, py = tr.transform(coords[:, 0], coords[:, 1])
        return np.column_stack([px, py])

    return shapely.transform(geom, func)


def _inside_and_score(px, py, bounds) -> tuple[np.ndarray, np.ndarray]:
    """Half-open containment in native tile bounds and the distance to the nearest edge."""
    west, south, east, north = bounds
    with np.errstate(invalid="ignore"):
        inside = (px >= west) & (px < east) & (py > south) & (py <= north)
        score = np.minimum.reduce([px - west, east - px, py - south, north - py])
    return inside, score


@dataclass(frozen=True)
class _Grid:
    """Native pixel grid of an opened tile (cell edges, not centres)."""

    x0: float  # left edge of column 0
    y0: float  # edge of row 0 (top edge for north-up tiles)
    dx: float
    dy: float  # negative for north-up tiles
    width: int
    height: int

    @classmethod
    def from_dataset(cls, ds: xr.Dataset) -> _Grid:
        x = ds["x"].values
        y = ds["y"].values
        if len(x) < 2 or len(y) < 2:
            raise ValueError("tile datasets need at least 2 pixels along x and y")
        dx = float(x[1] - x[0])
        dy = float(y[1] - y[0])
        return cls(
            float(x[0]) - dx / 2, float(y[0]) - dy / 2, dx, dy, len(x), len(y)
        )

    @property
    def transform(self) -> Affine:
        return Affine(self.dx, 0.0, self.x0, 0.0, self.dy, self.y0)

    def to_pixel(self, px: np.ndarray, py: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Integer (col, row) of each coordinate (inverse affine, floor; NaN -> -1)."""
        with np.errstate(invalid="ignore"):
            col = np.floor(np.round((px - self.x0) / self.dx, _PIXEL_SNAP))
            row = np.floor(np.round((py - self.y0) / self.dy, _PIXEL_SNAP))
        col = np.where(np.isfinite(col), col, -1)
        row = np.where(np.isfinite(row), row, -1)
        return col.astype("int64"), row.astype("int64")


def _embeddings(ds: xr.Dataset) -> xr.DataArray:
    emb = ds["embeddings"]
    if "time" in emb.dims:
        emb = emb.isel(time=0)
    return emb


def _band_names(datasets: dict[str, xr.Dataset]) -> list[str]:
    for ds in datasets.values():
        emb = ds["embeddings"]
        if "band" in emb.coords:
            return [str(b) for b in emb["band"].values]
        return AEF.band_names(emb.sizes["band"])
    return AEF.band_names(64)


# ---------------------------------------------------------------------------
# tile opening
# ---------------------------------------------------------------------------


async def _open_tiles(
    tiles: Sequence[AEFTileInfo],
    reader: VirtualTiffReader | None,
    gcp_project: str | None,
) -> dict[str, xr.Dataset]:
    """Open each tile once, lazily, at native chunks and without a bbox crop.

    Returns ``{tile.path: Dataset}``; the dataset has ``embeddings(time, band, y, x)``.
    """
    semaphore = asyncio.Semaphore(_OPEN_CONCURRENCY)

    async def open_one(active: VirtualTiffReader, tile: AEFTileInfo):
        async with semaphore:
            tree = await active.open_tiles_by_zone([tile], chunks="native")
        return tile.path, tree[next(iter(tree.children))].to_dataset()

    if not tiles:
        return {}
    if reader is not None:
        return dict(await asyncio.gather(*[open_one(reader, t) for t in tiles]))
    async with VirtualTiffReader(gcp_project=gcp_project) as active:
        return dict(await asyncio.gather(*[open_one(active, t) for t in tiles]))


async def _search_tiles(
    index: AEFIndex | None,
    source: DataSource | str,
    bounds_wgs84: tuple[float, float, float, float] | None,
    years: list[int],
) -> tuple[list[AEFTileInfo], AEFIndex]:
    if index is None:
        index = get_shared_index(source)
    await ensure_index_loaded(index)
    if bounds_wgs84 is None:
        return [], index
    tiles = await asyncio.to_thread(
        index.search, bounds_wgs84, (min(years), max(years)), None, "EPSG:4326"
    )
    return [t for t in tiles if t.year in set(years)], index


def _tiles_need_bounds(tiles: Sequence[AEFTileInfo]) -> None:
    missing = [t.id for t in tiles if t.utm_bounds is None]
    if missing:
        raise ValueError(
            f"index has no utm_west/south/east/north columns (tiles {missing[:3]}...); "
            "extraction needs the native UTM bounds"
        )


# ---------------------------------------------------------------------------
# points
# ---------------------------------------------------------------------------


def _plan_points(
    x: np.ndarray, y: np.ndarray, crs: str, years: list[int], tiles: Sequence[AEFTileInfo]
) -> np.ndarray:
    """Choose the tile for every (point, year): int array ``(n_points, n_years)``, -1 = none.

    Vectorised over points; each tile's native bounds are tested against the
    points projected once per tile CRS. Overlapping tiles: the one with the
    largest distance to its native edges wins.
    """
    _tiles_need_bounds(tiles)
    by_year: dict[int, list[int]] = defaultdict(list)
    for i, tile in enumerate(tiles):
        by_year[tile.year].append(i)

    projected: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    chosen = np.full((len(x), len(years)), -1, dtype="int64")
    for yi, year in enumerate(years):
        best = np.full(len(x), -np.inf)
        for ti in by_year.get(year, []):
            tile = tiles[ti]
            if tile.crs_epsg not in projected:
                projected[tile.crs_epsg] = _project_xy(x, y, crs, tile.crs_epsg)
            inside, score = _inside_and_score(*projected[tile.crs_epsg], tile.utm_bounds)
            better = inside & (score > best)
            chosen[better, yi] = ti
            best[better] = score[better]
    return chosen


def _sample_points(
    chosen: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    crs: str,
    years: list[int],
    tiles: Sequence[AEFTileInfo],
    datasets: dict[str, xr.Dataset],
    *,
    dequantize: bool,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Pointwise-select codes from the tiles in ONE dask compute.

    Returns ``(values, tile_index, band_names)``: values ``(n_points, n_years, n_bands)``
    (float32 NaN or int8 -128 where there is no data), and the tile index actually
    used per (point, year) (-1 when none).
    """
    bands = _band_names(datasets)
    n, ny = chosen.shape
    used = np.full((n, ny), -1, dtype="int64")
    jobs: list[tuple[np.ndarray, np.ndarray]] = []  # (flat output positions, ...)
    lazies = []
    for ti in np.unique(chosen[chosen >= 0]):
        tile = tiles[ti]
        ds = datasets[tile.path]
        grid = _Grid.from_dataset(ds)
        pi, yi = np.nonzero(chosen == ti)
        px, py = _project_xy(x[pi], y[pi], crs, tile.crs_epsg)
        col, row = grid.to_pixel(px, py)
        ok = (col >= 0) & (col < grid.width) & (row >= 0) & (row < grid.height)
        if not ok.any():
            continue
        pi, yi, col, row = pi[ok], yi[ok], col[ok], row[ok]
        used[pi, yi] = ti
        sel = _embeddings(ds).isel(
            y=xr.DataArray(row, dims="pt"), x=xr.DataArray(col, dims="pt")
        )
        lazies.append(sel.transpose("pt", "band").data)
        jobs.append((pi, yi))

    codes = np.full((n, ny, len(bands)), AEF_NODATA_VALUE, dtype="int8")
    if lazies:
        for (pi, yi), block in zip(jobs, dask.compute(*lazies), strict=True):
            codes[pi, yi] = np.asarray(block)
    if not dequantize:
        return codes, used, bands
    return dequantize_aef(codes), used, bands


def _points_frame(
    ids, x, y, years, tiles, used, values, bands
) -> pd.DataFrame:
    n, ny = used.shape
    flat_used = used.reshape(-1)
    tile_id = np.array(
        [tiles[i].id if i >= 0 else None for i in flat_used], dtype=object
    )
    zone = np.array(
        [tiles[i].utm_zone if i >= 0 else None for i in flat_used], dtype=object
    )
    head = pd.DataFrame(
        {
            "point_id": np.repeat(np.array(ids, dtype=object), ny) if n else [],
            "year": np.tile(np.array(years, dtype="int64"), n),
            "x": np.repeat(x, ny),
            "y": np.repeat(y, ny),
            "tile_id": pd.Series(tile_id, dtype=object),
            "utm_zone": pd.Series(zone, dtype=object),
        }
    )
    body = pd.DataFrame(values.reshape(n * ny, len(bands)), columns=bands)
    return pd.concat([head, body], axis=1)


def _extract_points_from_datasets(
    x, y, ids, years, crs, tiles, datasets, *, dequantize=True
) -> pd.DataFrame:
    """Offline core: plan + sample given every candidate tile's opened dataset."""
    chosen = _plan_points(x, y, crs, years, tiles)
    values, used, bands = _sample_points(
        chosen, x, y, crs, years, tiles, datasets, dequantize=dequantize
    )
    return _points_frame(ids, x, y, years, tiles, used, values, bands)


def _points_wgs84_bounds(x, y, crs) -> tuple[float, float, float, float] | None:
    px, py = _project_xy(x, y, crs, 4326)
    finite = np.isfinite(px) & np.isfinite(py)
    if not finite.any():
        return None
    pad = 1e-6
    return (
        float(px[finite].min()) - pad,
        float(py[finite].min()) - pad,
        float(px[finite].max()) + pad,
        float(py[finite].max()) + pad,
    )


async def aextract_points(
    points: Any,
    years: Any,
    *,
    crs: str = "EPSG:4326",
    index: AEFIndex | None = None,
    source: DataSource | str = DataSource.SOURCE_COOP,
    dequantize: bool = True,
    point_id: Sequence | None = None,
    reader: VirtualTiffReader | None = None,
) -> pd.DataFrame:
    """Extract the 64 AEF embedding values at points (async).

    Args:
        points: Sequence of ``(x, y)`` in ``crs``, or a GeoDataFrame of Points (its
            own CRS, if set, takes precedence over ``crs``).
        years: A year, an inclusive ``(start, end)`` tuple, or a list of years.
        crs: CRS of ``points``.
        index: ``AEFIndex`` to search (default: shared module-level index).
        source: Data source when ``index`` is not given.
        dequantize: ``True`` gives float32 values (NaN for nodata); ``False`` the
            raw int8 codes (-128 for nodata).
        point_id: Ids for the ``point_id`` column (default: positions, or the
            GeoDataFrame index).
        reader: ``VirtualTiffReader`` to reuse for opening tiles.

    Returns:
        One row per (point, year): ``point_id, year, x, y, tile_id, utm_zone`` then the
        band columns ``A00..A63``. Points outside every tile (or years without a tile)
        give NaN / -128 rows with ``tile_id`` and ``utm_zone`` None.

    Each tile is opened once, lazily at native chunks, pixels are picked with one
    vectorised pointwise selection per tile and everything is computed in a single
    ``dask.compute``. No reprojection or mosaicking is involved.
    """
    x, y, ids, crs = _normalise_points(points, crs, point_id)
    year_list = _normalise_years(years)
    bounds = _points_wgs84_bounds(x, y, crs)
    tiles, index = await _search_tiles(index, source, bounds, year_list)

    chosen = _plan_points(x, y, crs, year_list, tiles)
    needed = [tiles[i] for i in np.unique(chosen[chosen >= 0])]
    datasets = await _open_tiles(needed, reader, index.gcp_project)
    values, used, bands = await asyncio.to_thread(
        _sample_points, chosen, x, y, crs, year_list, tiles, datasets, dequantize=dequantize
    )
    return _points_frame(ids, x, y, year_list, tiles, used, values, bands)


def extract_points(points: Any, years: Any, **kwargs: Any) -> pd.DataFrame:
    """Synchronous, notebook-safe twin of :func:`aextract_points` (same arguments)."""
    return run_sync(aextract_points(points, years, **kwargs))


# ---------------------------------------------------------------------------
# zonal statistics
# ---------------------------------------------------------------------------


@dataclass
class _Item:
    """One polygon's footprint on one tile: a window of the tile and its pixel mask."""

    ti: int
    row0: int
    row1: int
    col0: int
    col1: int
    mask: np.ndarray  # (rows, cols) bool


def _plan_zonal(
    geoms: Sequence[Any],
    crs: str,
    years: list[int],
    tiles: Sequence[AEFTileInfo],
    datasets: dict[str, xr.Dataset],
    all_touched: bool,
) -> dict[tuple[int, int], list[_Item]]:
    """Per (polygon, year) the windows and masks on each overlapping tile's native grid.

    Pixels claimed by several tiles (overlapping UTM zones) are kept only on the tile
    where their centre is deepest inside its bounds, so nothing is counted twice.
    """
    from rasterio.features import geometry_mask

    _tiles_need_bounds(tiles)
    by_year: dict[int, list[int]] = defaultdict(list)
    for i, tile in enumerate(tiles):
        by_year[tile.year].append(i)

    native: dict[tuple[int, int], Any] = {}

    def geom_in(pi: int, epsg: int):
        if (pi, epsg) not in native:
            native[pi, epsg] = _project_geometry(geoms[pi], crs, epsg)
        return native[pi, epsg]

    grids: dict[int, _Grid] = {}
    plan: dict[tuple[int, int], list[_Item]] = {}
    for pi in range(len(geoms)):
        for year in years:
            cands = []
            for ti in by_year.get(year, []):
                tile = tiles[ti]
                gminx, gminy, gmaxx, gmaxy = geom_in(pi, tile.crs_epsg).bounds
                west, south, east, north = tile.utm_bounds
                if gminx > east or gmaxx < west or gminy > north or gmaxy < south:
                    continue
                if tile.path not in datasets:
                    continue
                if ti not in grids:
                    grids[ti] = _Grid.from_dataset(datasets[tile.path])
                grid = grids[ti]
                c_lo, c_hi = sorted(((gminx - grid.x0) / grid.dx, (gmaxx - grid.x0) / grid.dx))
                r_lo, r_hi = sorted(((gminy - grid.y0) / grid.dy, (gmaxy - grid.y0) / grid.dy))
                col0, col1 = max(math.floor(c_lo), 0), min(math.ceil(c_hi), grid.width)
                row0, row1 = max(math.floor(r_lo), 0), min(math.ceil(r_hi), grid.height)
                if col1 <= col0 or row1 <= row0:
                    continue
                window = grid.transform @ Affine.translation(col0, row0)
                mask = geometry_mask(
                    [geom_in(pi, tile.crs_epsg)],
                    out_shape=(row1 - row0, col1 - col0),
                    transform=window,
                    all_touched=all_touched,
                    invert=True,
                )
                if mask.any():
                    cands.append(_Item(ti, row0, row1, col0, col1, mask))
            if len(cands) > 1:
                _dedupe_overlaps(cands, tiles, grids)
            plan[pi, year] = [c for c in cands if c.mask.any()]
    return plan


def _dedupe_overlaps(
    cands: list[_Item],
    tiles: Sequence[AEFTileInfo],
    grids: dict[int, _Grid],
) -> None:
    """Drop mask pixels that another candidate tile claims more deeply (in place)."""
    centres = []
    for item in cands:
        tile, grid = tiles[item.ti], grids[item.ti]
        rows, cols = np.nonzero(item.mask)
        px = grid.x0 + (item.col0 + cols + 0.5) * grid.dx
        py = grid.y0 + (item.row0 + rows + 0.5) * grid.dy
        centres.append((rows, cols, px, py, tile))
    for a, item in enumerate(cands):
        rows, cols, px, py, tile = centres[a]
        _, score = _inside_and_score(px, py, tile.utm_bounds)
        drop = np.zeros(len(rows), dtype=bool)
        for b in range(len(cands)):
            if a == b:
                continue
            otile = centres[b][4]
            ox, oy = _project_xy(px, py, f"EPSG:{tile.crs_epsg}", otile.crs_epsg)
            inside, oscore = _inside_and_score(ox, oy, otile.utm_bounds)
            drop |= inside & ((oscore > score) | ((oscore == score) & (b < a)))
        if drop.any():
            item.mask[rows[drop], cols[drop]] = False


def _aggregate(chunks: list[np.ndarray], stat: str, n_bands: int) -> tuple[int, np.ndarray]:
    """Combine per-tile pixel blocks ``(n_bands, n_pixels)`` (float32, NaN = nodata).

    Returns ``(n_pixels, values[n_bands])``. mean/std/min/max/count combine per-tile
    partial sums, counts and second moments (Chan et al.), never means of means;
    the median needs the pixels, so those are concatenated. std is the population
    standard deviation (ddof=0).
    """
    n_pixels = int(sum(np.any(np.isfinite(c), axis=0).sum() for c in chunks))
    if stat == "median":
        pixels = np.concatenate(chunks, axis=1) if chunks else np.empty((n_bands, 0))
        with np.errstate(all="ignore"):
            valid = np.isfinite(pixels).any(axis=1)
            out = np.full(n_bands, np.nan)
            if valid.any():
                out[valid] = np.nanmedian(pixels[valid].astype("float64"), axis=1)
        return n_pixels, out

    count = np.zeros(n_bands)
    mean = np.zeros(n_bands)
    m2 = np.zeros(n_bands)
    lo = np.full(n_bands, np.inf)
    hi = np.full(n_bands, -np.inf)
    for c in chunks:
        c = c.astype("float64")
        finite = np.isfinite(c)
        n_c = finite.sum(axis=1)
        if not n_c.any():
            continue
        safe = np.where(finite, c, 0.0)
        mean_c = np.where(n_c > 0, safe.sum(axis=1) / np.maximum(n_c, 1), 0.0)
        m2_c = np.where(finite, (c - mean_c[:, None]) ** 2, 0.0).sum(axis=1)
        total = count + n_c
        delta = mean_c - mean
        share = np.where(total > 0, n_c / np.maximum(total, 1), 0.0)
        mean = mean + delta * share
        m2 = m2 + m2_c + delta**2 * count * share
        count = total
        lo = np.minimum(lo, np.where(finite, c, np.inf).min(axis=1))
        hi = np.maximum(hi, np.where(finite, c, -np.inf).max(axis=1))
    has = count > 0
    if stat == "count":
        return n_pixels, count
    out = np.full(n_bands, np.nan)
    if stat == "mean":
        out[has] = mean[has]
    elif stat == "std":
        out[has] = np.sqrt(m2[has] / count[has])
    elif stat == "min":
        out[has] = lo[has]
    elif stat == "max":
        out[has] = hi[has]
    return n_pixels, out


def _zonal_from_plan(
    plan: dict[tuple[int, int], list[_Item]],
    ids: Sequence,
    years: list[int],
    tiles: Sequence[AEFTileInfo],
    datasets: dict[str, xr.Dataset],
    stat: str,
) -> pd.DataFrame:
    bands = _band_names(datasets)
    keys = [(pi, year) for pi in range(len(ids)) for year in years]
    lazies, slots = [], []
    for key in keys:
        for item in plan[key]:
            emb = _embeddings(datasets[tiles[item.ti].path])
            window = emb.isel(
                y=slice(item.row0, item.row1), x=slice(item.col0, item.col1)
            ).transpose("band", "y", "x")
            lazies.append(window.data)
            slots.append(key)
    blocks: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    if lazies:
        computed = dask.compute(*lazies)
        items = [it for key in keys for it in plan[key]]
        for key, item, codes in zip(slots, items, computed, strict=True):
            values = dequantize_aef(np.asarray(codes)[:, item.mask])  # (bands, n_masked)
            blocks[key].append(values)

    rows_n, rows_v = [], []
    for key in keys:
        n_pixels, out = _aggregate(blocks.get(key, []), stat, len(bands))
        rows_n.append(n_pixels)
        rows_v.append(out)
    n = len(keys)
    head = pd.DataFrame(
        {
            "polygon_id": np.repeat(np.array(ids, dtype=object), len(years)) if n else [],
            "year": np.tile(np.array(years, dtype="int64"), len(ids)),
            "n_pixels": np.array(rows_n, dtype="int64"),
        }
    )
    body = pd.DataFrame(
        np.array(rows_v, dtype="float32").reshape(n, len(bands)), columns=bands
    )
    return pd.concat([head, body], axis=1)


def _extract_zonal_from_datasets(
    geoms, ids, years, crs, tiles, datasets, *, stat="mean", all_touched=False
) -> pd.DataFrame:
    """Offline core: masks + aggregation given every candidate tile's opened dataset."""
    plan = _plan_zonal(geoms, crs, years, tiles, datasets, all_touched)
    return _zonal_from_plan(plan, ids, years, tiles, datasets, stat)


def _check_stat(stat: str) -> None:
    if stat not in STATS:
        raise ValueError(f"stat must be one of {STATS}, got {stat!r}")


def _geoms_wgs84_bounds(geoms, crs) -> tuple[float, float, float, float] | None:
    from pyproj import CRS, Transformer

    if not geoms:
        return None
    bounds = np.array([g.bounds for g in geoms], dtype="float64")
    minx, miny = bounds[:, 0].min(), bounds[:, 1].min()
    maxx, maxy = bounds[:, 2].max(), bounds[:, 3].max()
    if CRS.from_user_input(crs) != CRS.from_epsg(4326):
        tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        minx, miny, maxx, maxy = tr.transform_bounds(minx, miny, maxx, maxy, densify_pts=21)
    return (float(minx), float(miny), float(maxx), float(maxy))


async def aextract_zonal(
    polygons: Any,
    years: Any,
    *,
    crs: str = "EPSG:4326",
    stat: str = "mean",
    index: AEFIndex | None = None,
    source: DataSource | str = DataSource.SOURCE_COOP,
    all_touched: bool = False,
    polygon_id: Sequence | None = None,
    reader: VirtualTiffReader | None = None,
) -> pd.DataFrame:
    """Zonal statistics of the 64 AEF bands over polygons (async).

    Args:
        polygons: Shapely Polygon/MultiPolygon geometries (or GeoJSON-like dicts), or a
            GeoDataFrame (its own CRS, if set, takes precedence over ``crs``).
        years: A year, an inclusive ``(start, end)`` tuple, or a list of years.
        crs: CRS of the polygons.
        stat: ``"mean"``, ``"median"``, ``"std"`` (population), ``"min"``, ``"max"`` or
            ``"count"`` (valid pixels per band).
        all_touched: Include every pixel the polygon touches, not just those whose
            centre it contains (rasterio semantics).
        polygon_id: Ids for the ``polygon_id`` column (default: positions / GeoDataFrame index).

    Returns:
        One row per (polygon, year): ``polygon_id, year, n_pixels`` (pixels with data)
        then the band columns. Membership is decided on each tile's native grid,
        values are dequantized before aggregating, and polygons spanning tiles or UTM
        zones are combined from per-tile sums/counts (or pixels, for the median).
    """
    _check_stat(stat)
    geoms, ids, crs = _normalise_polygons(polygons, crs, polygon_id)
    year_list = _normalise_years(years)
    tiles, index = await _search_tiles(
        index, source, _geoms_wgs84_bounds(geoms, crs), year_list
    )

    needed = _tiles_touching(geoms, crs, year_list, tiles)
    datasets = await _open_tiles([tiles[i] for i in needed], reader, index.gcp_project)

    def run() -> pd.DataFrame:
        return _extract_zonal_from_datasets(
            geoms, ids, year_list, crs, tiles, datasets, stat=stat, all_touched=all_touched
        )

    return await asyncio.to_thread(run)


def _tiles_touching(geoms, crs, years, tiles) -> list[int]:
    """Indices of tiles whose native bounds intersect any polygon of a requested year."""
    _tiles_need_bounds(tiles)
    hit = set()
    for ti, tile in enumerate(tiles):
        if tile.year not in years:
            continue
        west, south, east, north = tile.utm_bounds
        for geom in geoms:
            gminx, gminy, gmaxx, gmaxy = _project_geometry(geom, crs, tile.crs_epsg).bounds
            if not (gminx > east or gmaxx < west or gminy > north or gmaxy < south):
                hit.add(ti)
                break
    return sorted(hit)


def extract_zonal(polygons: Any, years: Any, **kwargs: Any) -> pd.DataFrame:
    """Synchronous, notebook-safe twin of :func:`aextract_zonal` (same arguments)."""
    return run_sync(aextract_zonal(polygons, years, **kwargs))
