"""
Loader for TESSERA embeddings stored as Zarr on Source Cooperative.

TESSERA (Cambridge, https://geotessera.org) provides yearly 128-dimensional,
10 m embeddings from Sentinel-1/2. The Source Cooperative Zarr v3 store has one
group per UTM zone (``utm01`` ... ``utm60``) holding:

- ``embeddings``: int8, dims ``(time, band, y, x)``
- ``scales``: float32, dims ``(time, y, x)``; real value = ``embeddings * scales``.
  NaN marks water, +inf marks a pixel that was never written.

Each zone uses the northern-hemisphere CRS ``EPSG:326zz`` with a continuous
(negative in the south) northing, so southern sites use the same group.

``open_tessera`` returns a DataTree with one child per UTM zone, matching the
layout of ``VirtualTiffReader.open_tiles_by_zone``, so ``reproject_datatree``
works on it unchanged.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable

import numpy as np
import xarray as xr
from xarray import DataTree

TESSERA_V1_1_DCLIMATE_URL = (
    "https://data.source.coop/tessera/tessera/zarr/v1.1-dclimate"
)

_QUALITY_VARS = (
    "s1_asc_obs_count",
    "s1_desc_obs_count",
    "s2_obs_count",
    "s1_asc_month_covered",
    "s1_desc_month_covered",
    "s2_month_covered",
)


def _open_root(url: str):
    import obstore
    import zarr
    from zarr.storage import ObjectStore

    store = obstore.store.from_url(url)
    return zarr.open_group(ObjectStore(store, read_only=True), mode="r")


def _candidate_zones(bbox_wgs84: tuple[float, float, float, float]) -> list[int]:
    west, _, east, _ = bbox_wgs84
    first = int(math.floor((west + 180) / 6)) + 1
    last = int(math.floor((min(east, 179.999999) + 180) / 6)) + 1
    return [z for z in range(max(first, 1), min(last, 60) + 1)]


def _to_wgs84(bbox, bbox_crs: str) -> tuple[float, float, float, float]:
    if str(bbox_crs).upper() == "EPSG:4326":
        return tuple(bbox)  # type: ignore[return-value]
    from pyproj import Transformer

    t = Transformer.from_crs(bbox_crs, "EPSG:4326", always_xy=True)
    return t.transform_bounds(*bbox, densify_pts=21)


def _normalise_years(years: int | Iterable[int] | tuple[int, int]) -> list[int]:
    """Sorted list of years: int -> [int]; 2-tuple -> inclusive range; else list."""
    if isinstance(years, int):
        return [int(years)]
    if isinstance(years, tuple) and len(years) == 2:
        return list(range(int(years[0]), int(years[1]) + 1))
    return sorted({int(y) for y in years})


def _select_years(ds: xr.Dataset, sel: list[int], group_name: str) -> xr.Dataset:
    """Select ``sel`` years from ``ds``; raise if none exist, warn if some are missing."""
    available = set(ds["time"].values.tolist())
    present = [y for y in sel if y in available]
    missing = [y for y in sel if y not in available]
    if not present:
        raise ValueError(
            f"None of the requested years {sel} are present in {group_name}; "
            f"the store has years {sorted(available)}"
        )
    if missing:
        warnings.warn(
            f"Years {missing} are not present in {group_name} and were skipped",
            stacklevel=3,
        )
    return ds.sel(time=present)


def open_tessera(
    bbox: tuple[float, float, float, float],
    bbox_crs: str = "EPSG:4326",
    years: int | Iterable[int] | tuple[int, int] | None = None,
    *,
    zones: Iterable[int] | None = None,
    dequantize: bool = False,
    include_quality: bool = False,
    chunks: dict | None = None,
    url: str = TESSERA_V1_1_DCLIMATE_URL,
) -> DataTree:
    """
    Lazily open TESSERA embeddings for an area of interest, organised by UTM zone.

    Nothing is downloaded until the result is computed. The AOI is cropped by
    source pixel index with no resampling.

    Args:
        bbox: ``(xmin, ymin, xmax, ymax)`` in ``bbox_crs``.
        bbox_crs: CRS of ``bbox`` (default ``EPSG:4326``).
        years: A single year, an inclusive ``(start, end)`` tuple, or any other
            iterable of years. ``None`` selects every year in the store.
        zones: UTM zone numbers to read. Default: zones the bbox touches.
        dequantize: If True, return float32 ``embeddings * scales`` with
            water / never-written pixels as NaN. Default keeps int8 codes plus
            the ``scales`` variable.
        include_quality: Also return the Sentinel observation-count and
            month-coverage variables.
        chunks: Dask chunks. Default ``{"time": 1, "band": -1, "y": 1024, "x": 1024}``.
        url: Root URL of the Zarr store.

    Returns:
        DataTree with one child per zone (``"31N"`` style), each a Dataset with
        ``embeddings`` (and ``scales`` unless dequantized) in the zone's CRS.
    """
    from odc.geo.xr import assign_crs

    root = _open_root(url)
    wgs = _to_wgs84(bbox, bbox_crs)
    zone_list = sorted(zones) if zones is not None else _candidate_zones(wgs)
    chunks = chunks or {"time": 1, "band": -1, "y": 1024, "x": 1024}
    # Normalise once: ``years`` may be a one-shot iterator that every zone reads.
    sel = _normalise_years(years) if years is not None else None

    from pyproj import Transformer

    zone_datasets: dict[str, xr.Dataset] = {}
    for zone in zone_list:
        group_name = f"utm{zone:02d}"
        if group_name not in root:
            continue
        crs = f"EPSG:326{zone:02d}"
        attrs = dict(root[group_name].attrs)
        a, _, c, _, e, f = attrs["spatial:transform"]
        nrows, ncols = attrs["spatial:shape"]

        if str(bbox_crs).upper() == crs:
            xmin, ymin, xmax, ymax = bbox
        else:
            xmin, ymin, xmax, ymax = Transformer.from_crs(
                "EPSG:4326", crs, always_xy=True
            ).transform_bounds(*wgs, densify_pts=21)
        col0 = max(int(math.floor((xmin - c) / a)), 0)
        col1 = min(int(math.ceil((xmax - c) / a)), ncols)
        row0 = max(int(math.floor((f - ymax) / -e)), 0)
        row1 = min(int(math.ceil((f - ymin) / -e)), nrows)
        if col1 <= col0 or row1 <= row0:
            continue

        ds = xr.open_zarr(
            root.store,
            group=group_name,
            consolidated=True,
            zarr_format=3,
            chunks=chunks,
            drop_variables=["x", "y", "time_bnds"],
        )
        ds = ds.isel(y=slice(row0, row1), x=slice(col0, col1))
        ds = ds.assign_coords(
            x=c + a * (np.arange(col0, col1) + 0.5),
            y=f + e * (np.arange(row0, row1) + 0.5),
        )
        if sel is not None:
            ds = _select_years(ds, sel, group_name)

        keep = ["embeddings", "scales"] + (list(_QUALITY_VARS) if include_quality else [])
        ds = ds[[v for v in keep if v in ds.data_vars]]

        if dequantize:
            finite = np.isfinite(ds["scales"])
            emb = ds["embeddings"].astype("float32") * ds["scales"].where(finite)
            emb.attrs["dequantized"] = True
            ds = ds.drop_vars("scales").assign(embeddings=emb)

        ds = assign_crs(ds, crs)
        ds.attrs.update(utm_zone=f"{zone}N", source="TESSERA", source_url=url)
        zone_datasets[f"{zone}N"] = ds

    if not zone_datasets:
        raise ValueError("No TESSERA data found for the requested bbox and zones")

    tree = DataTree.from_dict({f"/{k}": v for k, v in zone_datasets.items()})
    tree.attrs["zones"] = list(zone_datasets)
    return tree
