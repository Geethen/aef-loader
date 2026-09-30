"""
One-line entrypoints: ``open_aef`` / ``aopen_aef``.

``aopen_aef`` wraps the index download, the tile search and
``VirtualTiffReader.open_tiles_by_zone(bbox=...)``. ``open_aef`` is a synchronous
twin that is safe to call from Jupyter (where an event loop is already running).
A module-level ``AEFIndex`` is shared per ``(source, gcp_project, cache_dir)`` so
repeated calls do not re-read the index.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from xarray import DataTree

from aef_loader.constants import DataSource
from aef_loader.index import AEFIndex
from aef_loader.reader import ChunkSpec, VirtualTiffReader
from aef_loader.types import BoundingBox, DateRange

# Keyword arguments of open_tiles_by_zone; anything else in **reader_kwargs goes to
# the VirtualTiffReader constructor.
_OPEN_KWARGS = frozenset({"ifd", "buffer_pixels", "max_concurrency"})

_INDEXES: dict[tuple, AEFIndex] = {}
_INDEXES_LOCK = threading.Lock()


def get_shared_index(
    source: DataSource | str = DataSource.SOURCE_COOP,
    gcp_project: str | None = None,
    cache_dir: Path | str | None = None,
) -> AEFIndex:
    """The module-level ``AEFIndex`` for ``(source, gcp_project, cache_dir)``."""
    candidate = AEFIndex(source=source, gcp_project=gcp_project, cache_dir=cache_dir)
    key = (candidate.source, gcp_project, candidate.cache_dir)
    with _INDEXES_LOCK:
        return _INDEXES.setdefault(key, candidate)


async def ensure_index_loaded(index: AEFIndex) -> AEFIndex:
    """Download (if not cached on disk) and load ``index`` unless it is already loaded."""
    if index._df is None:
        await index.download()
        await asyncio.to_thread(index.load)
    return index


def run_sync[T](coro: Coroutine[Any, Any, T]) -> T:
    """Run ``coro`` to completion from synchronous code, even inside a running loop.

    Without a running event loop this is ``asyncio.run``. With one (Jupyter), the
    coroutine runs on a fresh event loop in a worker thread, so
    "asyncio.run() cannot be called from a running event loop" is never raised.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


async def aopen_aef(
    bbox: BoundingBox,
    years: int | DateRange,
    *,
    bbox_crs: str = "EPSG:4326",
    source: DataSource | str = DataSource.SOURCE_COOP,
    gcp_project: str | None = None,
    chunks: ChunkSpec = "balanced",
    index: AEFIndex | None = None,
    reader: VirtualTiffReader | None = None,
    **reader_kwargs: Any,
) -> DataTree:
    """Open the AEF embeddings covering ``bbox`` as a lazy DataTree (one group per UTM zone).

    Args:
        bbox: ``(minx, miny, maxx, maxy)`` in ``bbox_crs``.
        years: A year, or an inclusive ``(start_year, end_year)`` tuple.
        bbox_crs: CRS of ``bbox`` (default WGS84).
        source: ``DataSource.SOURCE_COOP`` (public, default) or ``DataSource.GCS``.
        gcp_project: GCP project for the requester-pays GCS bucket.
        chunks: Chunk profile passed to ``open_tiles_by_zone``.
        index: An ``AEFIndex`` to use instead of the shared module-level one.
        reader: A ``VirtualTiffReader`` to reuse (it is not closed here).
        **reader_kwargs: ``ifd``, ``buffer_pixels`` and ``max_concurrency`` go to
            ``open_tiles_by_zone``; anything else (e.g. ``manifest_cache_dir``) goes to
            the ``VirtualTiffReader`` constructor.

    Returns:
        DataTree from ``VirtualTiffReader.open_tiles_by_zone``; the bbox crop is lazy,
        nothing is read until ``.compute()``.

    Example:
        ```python
        tree = await aopen_aef((5.55, 58.65, 5.65, 58.75), 2024)
        ```
    """
    open_kwargs = {k: v for k, v in reader_kwargs.items() if k in _OPEN_KWARGS}
    ctor_kwargs = {k: v for k, v in reader_kwargs.items() if k not in _OPEN_KWARGS}
    if reader is not None and ctor_kwargs:
        raise TypeError(
            f"{sorted(ctor_kwargs)} configure a new VirtualTiffReader and cannot be "
            "combined with reader="
        )

    if index is None:
        index = get_shared_index(source, gcp_project)
    await ensure_index_loaded(index)
    tiles = await asyncio.to_thread(index.search, bbox, years, None, bbox_crs)

    async def _open(active: VirtualTiffReader) -> DataTree:
        return await active.open_tiles_by_zone(
            tiles, chunks=chunks, bbox=bbox, bbox_crs=bbox_crs, **open_kwargs
        )

    if reader is not None:
        return await _open(reader)
    async with VirtualTiffReader(gcp_project=gcp_project, **ctor_kwargs) as active:
        return await _open(active)


def open_aef(
    bbox: BoundingBox,
    years: int | DateRange,
    *,
    bbox_crs: str = "EPSG:4326",
    source: DataSource | str = DataSource.SOURCE_COOP,
    gcp_project: str | None = None,
    chunks: ChunkSpec = "balanced",
    index: AEFIndex | None = None,
    reader: VirtualTiffReader | None = None,
    **reader_kwargs: Any,
) -> DataTree:
    """Synchronous, notebook-safe twin of :func:`aopen_aef` (same arguments)."""
    return run_sync(
        aopen_aef(
            bbox,
            years,
            bbox_crs=bbox_crs,
            source=source,
            gcp_project=gcp_project,
            chunks=chunks,
            index=index,
            reader=reader,
            **reader_kwargs,
        )
    )
