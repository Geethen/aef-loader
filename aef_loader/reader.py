"""
COG reader using virtual-tiff library.

Provides efficient COG reading by virtualizing TIFFs as Zarr stores.
Supports both GCS (Google Cloud Storage) and S3 (Source Cooperative) backends.

The primary access pattern is loading tiles organized by UTM zone using
`open_tiles_by_zone()`. For combining data across zones, use
`reproject_datatree()` from the utils module.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import math
from collections import OrderedDict, defaultdict
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Literal

import xarray as xr
from affine import Affine
from obstore.store import GCSStore, S3Store
from odc.geo.geobox import GeoBox
from odc.geo.xr import assign_crs, xr_coords
from virtual_tiff import VirtualTIFF
from xarray import DataTree

try:
    from obspec_utils.registry import ObjectStoreRegistry
except ImportError:  # Compatibility with older VirtualiZarr releases
    from virtualizarr.registry import ObjectStoreRegistry

from aef_loader.cache import load_cached_manifest, save_manifest
from aef_loader.constants import AEF_NODATA_VALUE, SOURCE_COOP_REGION
from aef_loader.utils import set_aef_nodata

if TYPE_CHECKING:
    from aef_loader.types import AEFTileInfo

logger = logging.getLogger(__name__)

PathProtocol = Literal["gs", "s3"]
ChunkSpec = int | dict | Literal["auto", "native", "balanced", "all-bands"] | None


def _parse_gcs_path(path: str) -> tuple[str, str]:
    """Parse gs://bucket/key into (bucket, key)."""
    path = path.removeprefix("gs://")
    parts = path.split("/", 1)
    return parts[0], parts[1] if len(parts) > 1 else ""


def _parse_s3_path(path: str) -> tuple[str, str]:
    """Parse s3://bucket/key into (bucket, key)."""
    path = path.removeprefix("s3://")
    parts = path.split("/", 1)
    return parts[0], parts[1] if len(parts) > 1 else ""


def _detect_protocol(path: str) -> PathProtocol:
    """Detect the cloud protocol from a path."""
    if path.startswith("gs://"):
        return "gs"
    elif path.startswith("s3://"):
        return "s3"
    else:
        raise ValueError(f"Unknown protocol for path: {path}. Expected gs:// or s3://")


def _parse_cloud_path(path: str) -> tuple[PathProtocol, str, str]:
    """Parse cloud path into (protocol, bucket, key)."""
    protocol = _detect_protocol(path)
    if protocol == "gs":
        bucket, key = _parse_gcs_path(path)
    else:
        bucket, key = _parse_s3_path(path)
    return protocol, bucket, key


def _get_affine_from_model_pixel_scale_and_tiepoint(
    pixel_scale: tuple[float, float, float],
    tiepoint: tuple[float, float, float, float, float, float],
) -> Affine:
    """Creates an affine transform from the pixel scale and tiepoint.

    Args:
        pixel_scale: the ModelPixelScale tag - 3 values representing scale factor of x, y, and z
        tiepoint: the ModelTiepointTag - 6 values representing each of the tiepoints

    Returns:
        An affine transform calculated from these values.
    """
    sx, sy, _ = pixel_scale
    x, y = tiepoint[3], tiepoint[4]

    # TODO: validate the positive sy is correct, as I believe all the AEF images are bottom up
    return Affine(sx, 0, x, 0, sy, y)


def _get_affine_from_model_transform(model_transform: tuple[float, ...]) -> Affine:
    """Creates an affine transform from the model transform.

    Args:
        model_transform: The ModelTransformTag - 4x4 homogeneous transformation matrix in row-major order

    Returns:
        An affine transform calculated from these values
    """
    return Affine(
        model_transform[0],
        model_transform[1],
        model_transform[3],
        model_transform[4],
        model_transform[5],
        model_transform[7],
    )


def _get_geobox_from_dataset(ds: xr.Dataset, crs: str) -> GeoBox:
    """Extract GeoBox from dataset using the model_transformation or model_pixel_scale attribute if available.

    The model_transformation is a 4x4 matrix from the GeoTIFF that defines the
    affine transformation from pixel coordinates to CRS coordinates. This properly
    handles images stored bottom-up (positive y scale).

    Args:
        ds: Dataset with model_transformation in data variable attrs
        crs: CRS string (e.g., "EPSG:32610")

    Returns:
        GeoBox with correct affine transformation
    """
    height = ds.sizes["y"]
    width = ds.sizes["x"]

    for var in ds.data_vars:
        attrs = ds[var].attrs
        if ("model_pixel_scale" in attrs) or "model_transformation" in attrs:
            break
    else:
        raise ValueError(
            "Dataset missing model_pixel_scale or model_transformation attribute"
        )

    if "model_pixel_scale" in attrs:
        affine = _get_affine_from_model_pixel_scale_and_tiepoint(
            attrs["model_pixel_scale"], attrs.get("model_tiepoint", [0, 0, 0, 0, 0, 0])
        )
    else:
        affine = _get_affine_from_model_transform(attrs["model_transformation"])

    return GeoBox(shape=(height, width), affine=affine, crs=crs)


def _nodata_fill_for(ds: xr.Dataset):
    """Join/concat fill value in the dataset's own dtype (no dtype promotion).

    For integer (raw/quantized) data, returns the AEF nodata sentinel (-128) cast
    to that integer dtype, so an outer-join gap-fill stays int8 instead of being
    promoted to float64 by a NaN fill. For float (dequantized) data, returns NaN,
    the natural nodata for floats. Mixed/absent vars fall back to NaN.
    """
    import numpy as np

    for var in ds.data_vars:
        dtype = ds[var].dtype
        if np.issubdtype(dtype, np.integer):
            return dtype.type(AEF_NODATA_VALUE)
        return np.nan
    return np.nan


def _concat_time_slices(time_slices: list[xr.Dataset]) -> xr.Dataset:
    """Concatenate years without promoting integer data at coverage gaps."""
    return xr.concat(
        time_slices,
        dim="time",
        coords="minimal",
        compat="override",
        combine_attrs="drop_conflicts",
        join="outer",
        fill_value=_nodata_fill_for(time_slices[0]),
    )


def _native_block_chunks(manifest_store) -> dict[str, int]:
    """Chunk sizes matching the COG's stored block grid, keyed by dim name.

    Reads the first array's Zarr chunk shape from the manifest (e.g. (1, 1024,
    1024) for AEF: one band, one 1024x1024 block) and maps it onto the dataset's
    dimension names. Opening with these chunks gives one dask block per stored
    COG block, so ``expand_dims``/``concat`` stay graph-only and a windowed read
    fetches just the overlapping blocks. Falls back to ``{}`` (let open_zarr
    choose) if the grid can't be read, so this never breaks opening.
    """
    try:
        group = manifest_store._group
        _, marr = next(iter(group.arrays.items()))
        chunk_shape = marr.metadata.chunk_grid.chunk_shape
        dim_names = marr.metadata.dimension_names
        if dim_names is None or len(dim_names) != len(chunk_shape):
            # AEF arrays are (band, y, x); use that when names are absent.
            dim_names = ("band", "y", "x")[: len(chunk_shape)]
        return {dim: int(size) for dim, size in zip(dim_names, chunk_shape)}
    except Exception:  # noqa: BLE001 — chunk hinting is best-effort
        return {}


def _resolve_chunks(manifest_store, chunks: ChunkSpec):
    """Resolve named performance profiles to storage-aligned Dask chunks.

    Native retains one Dask chunk per COG block. Balanced groups up to
    16 stored band blocks per task, and all-bands groups every band. Spatial
    chunks always remain aligned with the COG block grid, so values and decoded
    source bytes are unchanged.
    """
    if chunks is None or chunks == "native":
        return _native_block_chunks(manifest_store)
    if chunks not in ("balanced", "all-bands"):
        return chunks

    resolved = _native_block_chunks(manifest_store)
    try:
        group = manifest_store._group
        _, marr = next(iter(group.arrays.items()))
        shape = marr.metadata.shape
        dim_names = marr.metadata.dimension_names
        if dim_names is None or len(dim_names) != len(shape):
            dim_names = ("band", "y", "x")[: len(shape)]
        sizes = dict(zip(dim_names, shape))
        if "band" in resolved and "band" in sizes:
            resolved["band"] = (
                int(sizes["band"])
                if chunks == "all-bands"
                else min(16, int(sizes["band"]))
            )
    except Exception as exc:  # noqa: BLE001 - profiles are best-effort hints
        logger.debug("could not resolve chunk profile %s: %s", chunks, exc)
    return resolved


def _is_empty_crop(ds: xr.Dataset) -> bool:
    """True when a cropped dataset has zero extent along x or y."""
    return ds.sizes.get("x", 1) == 0 or ds.sizes.get("y", 1) == 0


def _crop_to_bbox(
    ds: xr.Dataset,
    geobox: GeoBox,
    bbox: tuple[float, float, float, float],
    bbox_crs: str,
    buffer_pixels: int,
) -> xr.Dataset:
    """Select the source pixels whose cells cover a requested AOI.

    Bounds are densified during reprojection (when ``bbox_crs`` differs from the
    tile CRS) and converted to a half-open integer pixel window through the
    geobox's inverse affine. A pixel is included iff its cell intersects the AOI
    interior with positive area, so cells that merely touch the AOI edge are
    excluded. An AOI smaller than one pixel still selects its covering cell.
    ``buffer_pixels`` are then added on every side (useful for downstream
    interpolating kernels) and the window is clamped to the array. The result may
    be empty (0 along x and/or y) when the AOI does not overlap the tile.
    """
    from pyproj import CRS, Transformer

    minx, miny, maxx, maxy = bbox
    if CRS.from_user_input(bbox_crs) != geobox.crs:
        transformer = Transformer.from_crs(bbox_crs, str(geobox.crs), always_xy=True)
        minx, miny, maxx, maxy = transformer.transform_bounds(
            minx, miny, maxx, maxy, densify_pts=21
        )

    inverse = ~geobox.transform
    corners = [inverse @ (x, y) for x in (minx, maxx) for y in (miny, maxy)]
    cols = [c for c, _ in corners]
    rows = [r for _, r in corners]

    # Tolerance (in pixels) so float noise on an exact cell edge does not pull in
    # a neighbouring pixel.
    eps = 1e-6

    def window(low: float, high: float, size: int) -> slice:
        start = math.floor(low + eps)
        stop = math.ceil(high - eps)
        if stop <= start:  # sub-pixel (or zero-width) AOI: keep its covering cell
            stop = start + 1
        start = min(max(start - buffer_pixels, 0), size)
        stop = min(max(stop + buffer_pixels, start), size)
        return slice(start, stop)

    return ds.isel(
        x=window(min(cols), max(cols), ds.sizes["x"]),
        y=window(min(rows), max(rows), ds.sizes["y"]),
    )


class VirtualTiffReader:
    """
    COG reader using virtual-tiff to create virtual zarr stores.

    This provides efficient COG access by:
    - Creating virtual zarr stores from COGs without data duplication
    - Using async I/O via obstore for cloud access (GCS and S3)
    - Organizing tiles by UTM zone for proper CRS handling
    - Integrating directly with xarray for data loading

    The primary method is `open_tiles_by_zone()` which loads tiles organized
    by their native UTM zone. To combine data across zones, use
    `reproject_datatree()` from the utils module.

    Example:
        ```python
        from aef_loader import AEFIndex, VirtualTiffReader, DataSource
        from aef_loader.utils import reproject_datatree
        from odc.geo.geobox import GeoBox

        # Query tiles
        index = AEFIndex(source=DataSource.SOURCE_COOP)
        await index.download()
        index.load()
        tiles = index.search(bbox=(-122.5, 37.5, -121.5, 38.5), years=(2020, 2022))

        # Load by UTM zone
        async with VirtualTiffReader() as reader:
            tree = await reader.open_tiles_by_zone(tiles)

        # Reproject to common CRS if needed
        target = GeoBox.from_bbox(bbox=(-122.5, 37.5, -121.5, 38.5), crs="EPSG:4326", resolution=0.0001)
        combined = reproject_datatree(tree, target)
        result = combined.compute()
        ```
    """

    def __init__(
        self,
        gcp_project: str | None = None,
        manifest_cache_dir: str | Path | None = None,
        memory_manifest_cache_size: int = 128,
    ):
        """
        Initialize the virtual TIFF reader.

        Args:
            gcp_project: GCP project ID for requester-pays buckets (GCS only)
            manifest_cache_dir: Directory in which to cache parsed COG manifests.
                When set, each tile's virtual-tiff header parse (~1.6 s/tile) is
                serialised to disk on first open and reloaded (~15 ms) on later
                opens, so repeated runs over the same tiles skip the parse. Only
                header-derived metadata is cached — no pixel data — and reads
                return bit-identical results. When None (default), no on-disk
                cache is used and every open reparses (upstream behaviour).
            memory_manifest_cache_size: Maximum number of reconstructed manifests
                retained by this reader. Set to 0 to disable in-memory reuse.
        """
        self.gcp_project = gcp_project
        self.manifest_cache_dir = (
            Path(manifest_cache_dir) if manifest_cache_dir is not None else None
        )
        self._stores: dict[str, object] = {}  # Cache stores by (protocol, bucket)
        self._registry = None
        self.memory_manifest_cache_size = max(0, memory_manifest_cache_size)
        self._manifest_cache: OrderedDict[tuple[str, int], object] = OrderedDict()
        self._stats: dict[str, int | float] = {
            "open_calls": 0,
            "tiles_opened": 0,
            "memory_manifest_hits": 0,
            "disk_manifest_hits": 0,
            "manifest_parses": 0,
            "tile_open_seconds": 0.0,
            "last_open_wall_seconds": 0.0,
        }

    @property
    def stats(self) -> dict[str, int | float]:
        """Snapshot of reader activity for repeatable performance measurements."""
        return self._stats.copy()

    def _remember_manifest(self, key: tuple[str, int], manifest_store) -> None:
        if self.memory_manifest_cache_size == 0:
            return
        self._manifest_cache[key] = manifest_store
        self._manifest_cache.move_to_end(key)
        while len(self._manifest_cache) > self.memory_manifest_cache_size:
            self._manifest_cache.popitem(last=False)

    async def __aenter__(self) -> VirtualTiffReader:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._stores.clear()
        self._registry = None
        self._manifest_cache.clear()

    def _get_gcs_store(self, bucket: str) -> GCSStore:
        """Get or create an obstore GCSStore for a bucket."""
        if not self.gcp_project:
            raise ValueError(
                "gcp_project is required for reading from GCS requester-pays bucket"
            )
        return GCSStore(
            bucket=bucket,
            client_options={
                "default_headers": {"x-goog-user-project": self.gcp_project}
            },
        )

    def _get_s3_store(self, bucket: str) -> S3Store:
        """Get or create an obstore S3Store for a bucket (Source Cooperative)."""
        return S3Store(
            bucket=bucket,
            region=SOURCE_COOP_REGION,
            skip_signature=True,  # Source Coop is public, no auth needed
        )

    def _get_store(self, protocol: PathProtocol, bucket: str):
        """Get or create an obstore for a bucket based on protocol."""
        store_key = f"{protocol}://{bucket}"
        if store_key not in self._stores:
            if protocol == "gs":
                self._stores[store_key] = self._get_gcs_store(bucket)
            elif protocol == "s3":
                self._stores[store_key] = self._get_s3_store(bucket)
            else:
                raise ValueError(f"Unsupported protocol: {protocol}")
        return self._stores[store_key]

    def _get_registry(self, protocol: PathProtocol, bucket: str):
        """Get or create an ObjectStoreRegistry for a bucket."""
        if self._registry is None:
            self._registry = ObjectStoreRegistry()

        bucket_url = f"{protocol}://{bucket}/"
        store = self._get_store(protocol, bucket)
        self._registry.register(bucket_url, store)

        return self._registry

    async def open_tiles_by_zone(
        self,
        tiles: list[AEFTileInfo],
        ifd: int = 0,
        chunks: ChunkSpec = "auto",
        *,
        bbox: tuple[float, float, float, float] | None = None,
        bbox_crs: str = "EPSG:4326",
        buffer_pixels: int = 0,
        max_concurrency: int | None = 16,
    ) -> DataTree:
        """
        Open tiles and organize them by UTM zone in a DataTree.

        Each UTM zone becomes a group in the DataTree, containing a Dataset
        with a single 'embeddings' variable with a band dimension (A00–A63).
        Both ``nodata`` and ``_FillValue`` attrs are set to ``-128`` on each
        embeddings variable so that downstream tools (odc-geo ``xr_reproject``,
        xarray) correctly identify the AEF nodata sentinel.

        This is the primary method for loading AEF data. It keeps each zone's
        data in its native CRS for accurate spatial operations. To combine
        data across zones, use `reproject_datatree()` from the utils module.

        Args:
            tiles: List of AEFTileInfo objects from AEFIndex.search()
            ifd: Image File Directory index (0 for full resolution)
            chunks: The chunks parameter to pass to open_zarr, defaults to auto,
                with additional storage-aligned profiles: native, balanced
                (16 bands per task), and all-bands. None is an alias for native.
            bbox: Optional AOI to crop each tile before mosaicking. Pixel cells
                covering the AOI are retained; pixel values are never resampled.
            bbox_crs: CRS of bbox.
            buffer_pixels: Extra source pixels retained around bbox.
            max_concurrency: Shared limit for tile header/open work across every
                zone. Set to None for unbounded concurrency.

        Returns:
            DataTree with structure:
                ├── 10N/  → Dataset with embeddings(time, band, y, x) in EPSG:32610
                ├── 10S/  → Dataset with embeddings(time, band, y, x) in EPSG:32710
                ├── 11N/  → Dataset with embeddings(time, band, y, x) in EPSG:32611
                ...

        Example:
            ```python
            tiles = index.search(bbox=bbox, years=(2020, 2022))
            async with VirtualTiffReader() as reader:
                tree = await reader.open_tiles_by_zone(tiles)
            for zone in tree.children:
                ds = tree[zone].ds
                print(f"{zone}: {ds.odc.crs}, {dict(ds.sizes)}")
            ```
        """
        if not tiles:
            raise ValueError("No tiles provided")
        if buffer_pixels < 0:
            raise ValueError("buffer_pixels must be non-negative")
        if max_concurrency is not None and max_concurrency < 1:
            raise ValueError("max_concurrency must be positive or None")
        call_started = perf_counter()
        self._stats["open_calls"] += 1

        # Group tiles by UTM zone
        tiles_by_zone: dict[str, list[AEFTileInfo]] = defaultdict(list)
        for tile in tiles:
            zone = tile.utm_zone or "unknown"
            tiles_by_zone[zone].append(tile)

        logger.info(
            f"Loading {len(tiles)} tiles across {len(tiles_by_zone)} UTM zones: "
            f"{list(tiles_by_zone.keys())}"
        )

        semaphore = (
            asyncio.Semaphore(max_concurrency) if max_concurrency is not None else None
        )

        async def process_zone(
            zone: str, zone_tiles: list[AEFTileInfo]
        ) -> tuple[str, xr.Dataset | None]:
            logger.info(f"Processing zone {zone}: {len(zone_tiles)} tiles")

            ds = await self._combine_tiles_single_zone(
                zone_tiles,
                ifd,
                chunks=chunks,
                bbox=bbox,
                bbox_crs=bbox_crs,
                buffer_pixels=buffer_pixels,
                semaphore=semaphore,
            )
            if ds is None:
                logger.info(f"Zone {zone}: AOI does not overlap any tile, skipping")
                return zone, None

            # Add CRS metadata using odc-geo
            crs = f"EPSG:{zone_tiles[0].crs_epsg}"
            ds = assign_crs(ds, crs)

            # Add zone metadata
            ds.attrs["utm_zone"] = zone
            ds.attrs["num_tiles"] = len(zone_tiles)

            return zone, ds

        # Zone opens share the same bounded tile semaphore, keeping all available
        # I/O slots busy without the previous per-zone serial bottleneck.
        zone_results = await asyncio.gather(
            *[
                process_zone(zone, zone_tiles)
                for zone, zone_tiles in tiles_by_zone.items()
            ]
        )
        zone_datasets = {zone: ds for zone, ds in zone_results if ds is not None}
        if not zone_datasets:
            raise ValueError(
                f"bbox {bbox} (bbox_crs={bbox_crs!r}) does not overlap any of the "
                f"{len(tiles)} tiles"
            )

        # Build DataTree
        tree_dict = {f"/{zone}": ds for zone, ds in zone_datasets.items()}
        tree = DataTree.from_dict(tree_dict)

        # Add root attributes
        tree.attrs["total_tiles"] = len(tiles)
        tree.attrs["zones"] = list(zone_datasets.keys())
        self._stats["last_open_wall_seconds"] = perf_counter() - call_started

        return tree

    async def _combine_tiles_single_zone(
        self,
        tiles: list[AEFTileInfo],
        ifd: int = 0,
        chunks: ChunkSpec = "auto",
        *,
        bbox: tuple[float, float, float, float] | None = None,
        bbox_crs: str = "EPSG:4326",
        buffer_pixels: int = 0,
        semaphore: asyncio.Semaphore | None = None,
    ) -> xr.Dataset | None:
        """
        Combine tiles within a single UTM zone.

        Tiles whose crop to ``bbox`` is empty are skipped; returns None when every
        tile in the zone is empty. All tiles must be in the same CRS. Combines spatially and temporally,
        keeping bands as a single 'embeddings' variable with a band dimension.
        Sets both ``nodata`` and ``_FillValue`` to ``-128`` on the output via
        ``set_aef_nodata``.
        """
        parser = VirtualTIFF(ifd=ifd)

        cache_dir = self.manifest_cache_dir

        async def process_tile(tile: AEFTileInfo) -> xr.Dataset | None:
            if semaphore is not None:
                async with semaphore:
                    return await process_tile_unlocked(tile)
            return await process_tile_unlocked(tile)

        async def process_tile_unlocked(tile: AEFTileInfo) -> xr.Dataset | None:
            started = perf_counter()
            protocol, bucket, key = _parse_cloud_path(tile.path)
            file_url = f"{protocol}://{bucket}/{key}"
            registry = self._get_registry(protocol, bucket)

            # Reuse a cached manifest when available; otherwise parse the COG
            # header and write it back so the next run skips the parse. Both the
            # load and save are best-effort (never raise) — a cache miss or error
            # just falls through to a normal parse, so behaviour is unchanged when
            # no cache_dir is set or the cache is cold/corrupt.
            manifest_key = (file_url, ifd)
            manifest_store = self._manifest_cache.get(manifest_key)
            if manifest_store is not None:
                self._manifest_cache.move_to_end(manifest_key)
                self._stats["memory_manifest_hits"] += 1
            elif cache_dir is not None:
                manifest_store = await asyncio.to_thread(
                    load_cached_manifest, cache_dir, file_url, ifd, registry
                )
                if manifest_store is not None:
                    self._stats["disk_manifest_hits"] += 1
            if manifest_store is None:
                manifest_store = await asyncio.to_thread(
                    parser, url=file_url, registry=registry
                )
                self._stats["manifest_parses"] += 1
                if cache_dir is not None:
                    await asyncio.to_thread(
                        save_manifest, cache_dir, file_url, ifd, manifest_store
                    )
            self._remember_manifest(manifest_key, manifest_store)

            # Resolve the chunks argument. chunks=None asks open_zarr for numpy-
            # backed (non-dask) arrays; that is a footgun here because the very
            # next step, expand_dims(time=...), then force-reads the whole tile
            # (~100 s for 8192^2 x 64), as do concat and any lazy windowing. To
            # keep those operations lazy while still honouring the intent of
            # chunks=None ("don't explode the dask graph"), open at the COG's
            # native block size — one dask chunk per stored block, so a windowed
            # read still only fetches the blocks it overlaps. An explicit chunks=
            # value (int/dict/"auto") is passed straight through unchanged.
            resolved_chunks = _resolve_chunks(manifest_store, chunks)
            ds: xr.Dataset = await asyncio.to_thread(
                xr.open_zarr,
                manifest_store,
                zarr_format=3,
                consolidated=False,
                chunks=resolved_chunks,
                mask_and_scale=False,
            )

            # Extract GeoBox from the model_transformation in the TIFF
            # This correctly handles bottom-up images (positive y scale)
            crs = f"EPSG:{tile.crs_epsg}"
            geobox = _get_geobox_from_dataset(ds, crs)
            coords = xr_coords(geobox)

            # Assign spatial coordinates from the actual TIFF affine
            ds = ds.assign_coords(x=coords["x"].values, y=coords["y"].values)
            if bbox is not None:
                ds = _crop_to_bbox(ds, geobox, bbox, bbox_crs, buffer_pixels)
                if _is_empty_crop(ds):
                    return None  # AOI does not overlap this tile

            # Expand time as a dimension. This is free on dask-backed arrays (a
            # graph-only op) but forces a full materialising read on numpy-backed
            # ones (~100 s for an 8192^2 x 64 int8 tile). open_tiles_by_zone opens
            # with dask chunks (see the chunks handling there) precisely so this,
            # concat, and windowed reads all stay lazy.
            ds = ds.expand_dims(time=[tile.as_datetime])

            ds.attrs["_source_url"] = file_url
            ds.attrs["_tile_id"] = tile.id
            self._stats["tiles_opened"] += 1
            self._stats["tile_open_seconds"] += perf_counter() - started

            return ds

        datasets = [
            ds
            for ds in await asyncio.gather(*[process_tile(tile) for tile in tiles])
            if ds is not None
        ]
        if not datasets:
            return None

        # Group by time
        datasets_by_time: dict[dt.datetime, list[xr.Dataset]] = defaultdict(list)
        for ds in datasets:
            time_val = ds.coords["time"].values[0]
            time_key = dt.datetime.fromtimestamp(
                time_val.astype("datetime64[s]").astype("int"), tz=dt.UTC
            )
            datasets_by_time[time_key].append(ds)

        # Combine spatially within each time, then temporally
        time_slices = []
        for time_val in sorted(datasets_by_time.keys()):
            time_datasets = datasets_by_time[time_val]

            if len(time_datasets) == 1:
                spatial_combined = time_datasets[0]
            else:
                # fill_value pins the join gap-fill to the AEF nodata sentinel in
                # the data's own dtype. Without it, an outer join over tiles that
                # don't fully tile the union rectangle fills gaps with float NaN,
                # which promotes int8 -> float64: an 8x memory/transfer blow-up of
                # the whole zone in the dask graph, and it discards the -128
                # sentinel. Using the sentinel keeps the result int8 and nodata-
                # correct. (For already-dequantized float input, callers pass
                # float data and NaN promotion is a no-op; -128 as float is still
                # a valid distinct fill there.)
                fill = _nodata_fill_for(time_datasets[0])
                spatial_combined = xr.combine_by_coords(
                    time_datasets,
                    coords="minimal",
                    compat="override",
                    combine_attrs="drop_conflicts",
                    join="outer",
                    fill_value=fill,
                )
            time_slices.append(spatial_combined)

        if len(time_slices) == 1:
            combined = time_slices[0]
        else:
            combined = _concat_time_slices(time_slices)

        # Keep bands as a single variable with string band coordinates
        if "band" in combined.dims:
            data_var = list(combined.data_vars)[0]
            da = combined[data_var]
            # Assign string band coordinate labels (A00, A01, ..., A63)
            band_names = [f"A{i:02d}" for i in range(da.sizes["band"])]
            da = da.assign_coords(band=band_names)
            da.name = "embeddings"
            da = set_aef_nodata(da)
            combined = da.to_dataset()

        return combined
