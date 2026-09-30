"""
Chip (patch) reads: fixed-size windows on a tile's native grid.

``read_chip`` returns one ``size`` x ``size`` window of raw codes centred on a
point; ``read_chips`` does the same for many points while fetching each stored
COG block only once. Nothing is resampled: a chip is a slice of the tile's own
10 m pixel lattice, in the tile's UTM CRS, north-up.

Chip geometry
-------------
The chip contains the pixel that holds the requested point (its *centre pixel*).
That pixel sits at index ``size // 2`` (counting from the top-left, 0-based) on
both axes. For odd sizes it is the exact middle pixel; for even sizes there is no
middle pixel, so the chip extends ``size // 2`` pixels to the west/north of the
centre pixel and ``size // 2 - 1`` to the east/south. A point exactly on a pixel
edge belongs to the pixel to its east/south (``floor``).

Chips that cross a tile edge inside one UTM zone are mosaicked from the
neighbouring tiles. Where the point's zone has no data (across a zone boundary, or
outside coverage) the chip is filled with the collection's nodata value (AEF:
-128); the neighbouring zone's tiles are never reprojected in.

Block sharing
-------------
``read_chips`` groups chips that share tiles, opens each tile once with native
chunks (one dask chunk per stored block), builds every window lazily and calls
``dask.compute`` once per group, so dask fetches a block shared by several chips a
single time. To share blocks across calls, give the reader
``VirtualTiffReader(block_cache_bytes=...)``.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import numpy as np
from affine import Affine

from aef_loader.constants import DataSource
from aef_loader.reader import VirtualTiffReader

if TYPE_CHECKING:
    import xarray as xr

    from aef_loader.index import AEFIndex
    from aef_loader.types import AEFTileInfo

_NOMINAL_RES = 10.0  # AEF pixel size in metres, used only to pre-select tiles
_ALIGN_TOL = 1e-3  # allowed lattice misalignment, in pixels
_DEFAULT_NODATA = -128


@dataclass(frozen=True, eq=False)
class Chip:
    """A ``size`` x ``size`` window of raw codes on a tile's native grid.

    Attributes:
        data: Array with dims ``(time, band, y, x)``, raw stored codes (AEF int8,
            nodata -128), rows north to south.
        crs: The tile's UTM CRS (e.g. ``"EPSG:32631"``).
        transform: Affine of the chip in ``crs`` (north-up; ``transform * (0, 0)``
            is the top-left corner of the top-left pixel).
        years: Year of each entry along ``time``.
        band_names: Label of each entry along ``band``.
        tile_ids: Ids of the tiles that overlap the chip.
    """

    data: np.ndarray
    crs: str
    transform: Affine
    years: tuple[int, ...]
    band_names: tuple[str, ...]
    tile_ids: tuple[str, ...]


# --------------------------------------------------------------------------- #
# Pure geometry                                                                #
# --------------------------------------------------------------------------- #


def chip_axes(
    px: float, py: float, size: int, x0: float, dx: float, y0: float, dy: float
) -> tuple[np.ndarray, np.ndarray, Affine]:
    """Pixel-centre coordinates and affine of the chip around ``(px, py)``.

    ``x0``/``y0`` are the centre of any pixel of the lattice and ``dx``/``dy`` its
    signed step (``dy < 0`` for a north-up grid). The chip is always returned
    north-up (``ys`` decreasing) whatever the lattice orientation.
    """
    rx, ry = abs(dx), abs(dy)
    cx = x0 + math.floor((px - x0) / dx + 0.5) * dx
    cy = y0 + math.floor((py - y0) / dy + 0.5) * dy
    k = np.arange(size) - size // 2
    xs = cx + k * rx
    ys = cy - k * ry
    transform = Affine(rx, 0.0, xs[0] - rx / 2, 0.0, -ry, ys[0] + ry / 2)
    return xs, ys, transform


@dataclass(frozen=True)
class _AxisPlan:
    src: slice  # slice into the mosaic axis
    dst: slice  # where the slice lands in the chip axis
    flip: bool  # the mosaic runs opposite to the chip along this axis


def _axis_plan(centres: np.ndarray, first: float, step: float, n: int) -> _AxisPlan | None:
    """Map chip pixel centres onto mosaic indices; None when there is no overlap."""
    frac = (centres - first) / step
    idx = np.rint(frac).astype(np.int64)
    if np.abs(frac - idx).max() > _ALIGN_TOL:
        raise ValueError("chip lattice is not aligned with the tile pixel grid")
    valid = np.flatnonzero((idx >= 0) & (idx < n))
    if valid.size == 0:
        return None
    lo, hi = idx[valid].min(), idx[valid].max()
    return _AxisPlan(
        src=slice(int(lo), int(hi) + 1),
        dst=slice(int(valid[0]), int(valid[-1]) + 1),
        flip=bool(idx.size > 1 and idx[1] < idx[0]),
    )


def _paste(
    shape: tuple[int, ...], dtype, nodata, plan: tuple[_AxisPlan, _AxisPlan] | None, sub
) -> np.ndarray:
    """Place the computed overlap ``sub`` into a nodata-filled chip array."""
    out = np.full(shape, nodata, dtype=dtype)
    if plan is None:
        return out
    yplan, xplan = plan
    if yplan.flip:
        sub = sub[..., ::-1, :]
    if xplan.flip:
        sub = sub[..., :, ::-1]
    out[..., yplan.dst, xplan.dst] = sub
    return out


@dataclass(frozen=True)
class _Request:
    px: float  # point in the mosaic's CRS
    py: float
    size: int
    years: tuple[int, ...]


def read_windows(
    da: xr.DataArray, requests: Sequence[_Request], nodata=_DEFAULT_NODATA
) -> list[tuple[np.ndarray, Affine]]:
    """Cut every requested chip out of one lazy mosaic with a single compute.

    ``da`` has dims ``(time, band, y, x)`` with pixel-centre ``x``/``y`` coords and
    is normally dask-backed with one chunk per stored block. All windows are
    combined into one ``dask.compute``, so a chunk needed by several chips is
    loaded once. Returns ``(data, transform)`` per request, in order.
    """
    import dask
    import pandas as pd

    x = np.asarray(da["x"].values, dtype=np.float64)
    y = np.asarray(da["y"].values, dtype=np.float64)
    if x.size < 2 or y.size < 2:
        raise ValueError("mosaic must span at least two pixels on each axis")
    x0, dx, y0, dy = x[0], x[1] - x[0], y[0], y[1] - y[0]
    tile_years = pd.DatetimeIndex(da["time"].values).year.to_numpy()
    da = da.transpose("time", "band", "y", "x")

    layouts = []
    lazy = []
    for req in requests:
        xs, ys, transform = chip_axes(req.px, req.py, req.size, x0, dx, y0, dy)
        tidx = np.flatnonzero(np.isin(tile_years, req.years))
        shape = (tidx.size, da.sizes["band"], req.size, req.size)
        xplan = _axis_plan(xs, x0, dx, x.size)
        yplan = _axis_plan(ys, y0, dy, y.size)
        plan = (yplan, xplan) if xplan is not None and yplan is not None else None
        if plan is not None and tidx.size:
            lazy.append(
                da.isel(time=tidx, y=plan[0].src, x=plan[1].src).data
            )
        else:
            plan = None
            lazy.append(None)
        layouts.append((shape, plan, transform))

    pending = [a for a in lazy if a is not None]
    computed = iter(dask.compute(*pending) if pending else ())
    out = []
    for arr, (shape, plan, transform) in zip(lazy, layouts):
        sub = next(computed) if arr is not None else None
        out.append((_paste(shape, da.dtype, nodata, plan, sub), transform))
    return out


# --------------------------------------------------------------------------- #
# Point -> tiles                                                               #
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=32)
def _transformer(src: str, dst: str):
    from pyproj import Transformer

    return Transformer.from_crs(src, dst, always_xy=True)


def _bounds_distance(tile: AEFTileInfo, px: float, py: float) -> float:
    west, south, east, north = tile.utm_bounds
    return math.hypot(max(west - px, 0, px - east), max(south - py, 0, py - north))


@dataclass(frozen=True)
class _Resolved:
    epsg: int
    px: float
    py: float
    tiles: tuple[AEFTileInfo, ...]


def _resolve_point(
    x: float, y: float, crs: str, source: AEFIndex | Sequence[AEFTileInfo], years, size: int
) -> _Resolved:
    """Pick the point's CRS and the tiles that overlap its chip window."""
    lon, lat = _transformer(crs, "EPSG:4326").transform(x, y)
    from aef_loader.index import selected_years

    wanted = set(selected_years(years))

    if hasattr(source, "search"):
        candidates = source.search(bbox=(lon, lat, lon, lat), years=years)
    else:
        candidates = [
            t
            for t in source
            if t.year in wanted
            and t.bbox[0] <= lon <= t.bbox[2]
            and t.bbox[1] <= lat <= t.bbox[3]
        ]
    if not candidates:
        raise ValueError(f"no tile covers point ({x}, {y}) in {crs} for years {years}")

    # The zone containing the point: the CRS whose tile bounds hold (or are
    # nearest to) the projected point, else the one closest to its central meridian.
    scored: dict[int, float] = {}
    for tile in candidates:
        px, py = _transformer(crs, f"EPSG:{tile.crs_epsg}").transform(x, y)
        score = (
            _bounds_distance(tile, px, py)
            if tile.utm_bounds is not None
            else abs(px - 500_000.0)
        )
        scored[tile.crs_epsg] = min(score, scored.get(tile.crs_epsg, math.inf))
    epsg = min(scored, key=lambda e: (scored[e], e))
    px, py = _transformer(crs, f"EPSG:{epsg}").transform(x, y)

    # Chip window on the tile lattice (tile corners are lattice-aligned), used
    # only to avoid opening tiles that the chip does not touch.
    anchor = next(
        (t for t in candidates if t.crs_epsg == epsg and t.utm_bounds is not None), None
    )
    res = _NOMINAL_RES
    if anchor is not None:
        ox, oy = anchor.utm_bounds[0], anchor.utm_bounds[3]
        left = ox + (math.floor((px - ox) / res) - size // 2) * res
        top = oy - (math.floor((oy - py) / res) - size // 2) * res
        window = (left, top - size * res, left + size * res, top)
    else:
        pad = (size // 2 + 1) * res
        window = (px - pad, py - pad, px + pad, py + pad)

    if hasattr(source, "search"):
        near = source.search(bbox=window, bbox_crs=f"EPSG:{epsg}", years=years)
    else:
        near = [t for t in source if t.year in wanted]
        w, s, e, n = _transformer(f"EPSG:{epsg}", "EPSG:4326").transform_bounds(
            *window, densify_pts=21
        )
        near = [
            t
            for t in near
            if t.bbox[0] <= e and t.bbox[2] >= w and t.bbox[1] <= n and t.bbox[3] >= s
        ]
    chosen = []
    for t in near:
        if t.crs_epsg != epsg:
            continue
        if t.utm_bounds is not None and not (
            t.utm_bounds[0] < window[2]
            and t.utm_bounds[2] > window[0]
            and t.utm_bounds[1] < window[3]
            and t.utm_bounds[3] > window[1]
        ):
            continue
        chosen.append(t)
    if not chosen:
        raise ValueError(f"no tile overlaps the chip window at ({x}, {y}) in {crs}")
    return _Resolved(epsg=epsg, px=px, py=py, tiles=tuple(chosen))


def _normalise_points(points: Any, crs: str) -> tuple[list[tuple[float, float]], str]:
    """Accept a sequence of (x, y) or a GeoDataFrame/GeoSeries (duck-typed)."""
    if hasattr(points, "geometry") or hasattr(points, "representative_point"):
        geom = getattr(points, "geometry", points)
        if getattr(geom, "crs", None) is not None:
            crs = geom.crs.to_string()
        reps = geom.representative_point()
        return [(float(px), float(py)) for px, py in zip(reps.x, reps.y)], crs
    return [(float(px), float(py)) for px, py in points], crs


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #


async def read_chips(
    points,
    *,
    size: int = 256,
    years,
    index: AEFIndex | Sequence[AEFTileInfo] | None = None,
    reader: VirtualTiffReader | None = None,
    crs: str = "EPSG:4326",
    max_concurrency: int = 16,
    source: DataSource | str = DataSource.SOURCE_COOP,
) -> list[Chip]:
    """Read one ``size`` x ``size`` native-grid chip per point.

    ``index`` is an ``AEFIndex`` (downloaded and loaded on demand), a list of
    ``AEFTileInfo``, or None to use the shared index for ``source``.

    Chips that share tiles are read together: each tile is opened once (native
    chunks), every window is built lazily, and the group is computed in one
    ``dask.compute``, so a stored block shared by several chips is fetched once.
    Output order matches ``points``. See the module docstring for chip geometry.

    Args:
        points: Sequence of ``(x, y)`` in ``crs``, or a GeoDataFrame/GeoSeries
            (its own CRS wins; non-point geometries use a representative point).
        size: Chip edge in native pixels.
        years: A year, ``(start, end)`` years (inclusive), or a list of years, as for
            ``AEFIndex.search``; the chip's time axis has one entry per year found.
        index: An ``AEFIndex`` (searched per chip) or a list of ``AEFTileInfo``.
        reader: Reader to use. Pass one built with ``block_cache_bytes`` to reuse
            blocks across calls. When None a temporary reader is used.
        crs: CRS of the point coordinates.
        max_concurrency: Bound on concurrent tile opens across all groups; group
            computes run at most ``min(max_concurrency, 4)`` at a time.
    """
    if size < 1:
        raise ValueError("size must be a positive integer")
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be positive")
    xy, crs = _normalise_points(points, crs)
    if not xy:
        return []

    from aef_loader.api import ensure_index_loaded, get_shared_index

    if index is None:
        index = get_shared_index(source)
    if hasattr(index, "search"):
        index = await ensure_index_loaded(index)

    resolved = [
        await asyncio.to_thread(_resolve_point, x, y, crs, index, years, size)
        for x, y in xy
    ]

    # Union chips that share any tile, so a shared block is computed once.
    parent = list(range(len(resolved)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owner: dict[str, int] = {}
    for i, res in enumerate(resolved):
        for tile in res.tiles:
            j = owner.setdefault(tile.path, i)
            parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(resolved)):
        groups.setdefault(find(i), []).append(i)

    own_reader = reader is None
    reader = reader or VirtualTiffReader()
    nodata = reader.collection.nodata
    nodata = _DEFAULT_NODATA if nodata is None else nodata
    compute_slots = asyncio.Semaphore(min(max_concurrency, 4))
    # One limit on tile opens across every group, not one per group.
    open_slots = asyncio.Semaphore(max_concurrency)
    chips: list[Chip | None] = [None] * len(resolved)

    async def run_group(members: list[int]) -> None:
        tiles = list({t.path: t for i in members for t in resolved[i].tiles}.values())
        tree = await reader.open_tiles_by_zone(
            tiles, chunks="native", semaphore=open_slots
        )
        (zone,) = tree.children
        da = tree[zone].ds["embeddings"]
        requests = [
            _Request(
                resolved[i].px,
                resolved[i].py,
                size,
                tuple(sorted({t.year for t in resolved[i].tiles})),
            )
            for i in members
        ]
        async with compute_slots:
            windows = await asyncio.to_thread(read_windows, da, requests, nodata)
        band_names = tuple(str(b) for b in da["band"].values)
        for i, req, (data, transform) in zip(members, requests, windows):
            chips[i] = Chip(
                data=data,
                crs=f"EPSG:{resolved[i].epsg}",
                transform=transform,
                years=req.years,
                band_names=band_names,
                tile_ids=tuple(dict.fromkeys(t.id for t in resolved[i].tiles)),
            )

    async def run_all() -> None:
        await asyncio.gather(*(run_group(m) for m in groups.values()))

    if own_reader:
        async with reader:
            await run_all()
    else:
        await run_all()
    return [c for c in chips if c is not None]


async def read_chip(
    tiles_or_index: AEFIndex | Sequence[AEFTileInfo] | None = None,
    *,
    x: float,
    y: float,
    crs: str = "EPSG:4326",
    size: int = 256,
    years,
    reader: VirtualTiffReader | None = None,
    source: DataSource | str = DataSource.SOURCE_COOP,
) -> Chip:
    """Read one ``size`` x ``size`` native-grid chip centred on ``(x, y)``.

    ``tiles_or_index`` is an ``AEFIndex`` (tiles found with ``index.search``) or a
    list of ``AEFTileInfo``. A chip crossing a tile edge in the same UTM zone is
    mosaicked from the neighbouring tiles; pixels outside the zone's coverage are
    nodata. With ``tiles_or_index=None`` the shared index for ``source`` is
    used (downloaded on first use). Geometry and return value: see the module docstring and
    :class:`Chip`.
    """
    (chip,) = await read_chips(
        [(x, y)],
        size=size,
        years=years,
        index=tiles_or_index,
        reader=reader,
        crs=crs,
        source=source,
    )
    return chip
