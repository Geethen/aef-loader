"""Thread- and event-loop-safe limits for physical object-store reads."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from typing import Any


class ReadStats:
    """Shared read counters for the stores owned by one reader."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests = 0
        self._bytes = 0
        self._inflight = 0
        self._peak_inflight = 0

    def started(self) -> None:
        with self._lock:
            self._requests += 1
            self._inflight += 1
            self._peak_inflight = max(self._peak_inflight, self._inflight)

    def finished(self, value: Any = None) -> None:
        with self._lock:
            self._bytes += _byte_length(value)
            self._inflight -= 1

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "read_requests": self._requests,
                "read_bytes": self._bytes,
                "read_peak_inflight": self._peak_inflight,
            }

    def reset(self) -> None:
        """Clear counters between independent measurements when no reads run."""
        with self._lock:
            self._requests = 0
            self._bytes = 0
            self._inflight = 0
            self._peak_inflight = 0


def _byte_length(value: Any) -> int:
    """Return a result's byte count without consuming a streaming result."""
    if value is None:
        return 0
    if isinstance(value, (list, tuple)):
        return sum(_byte_length(item) for item in value)
    try:
        return len(value)
    except TypeError:
        bytes_value = getattr(value, "bytes", None)
        if bytes_value is not None and not callable(bytes_value):
            try:
                return len(bytes_value)
            except TypeError:
                pass
    return 0


class ReadLimitedStore:
    """Delegate an object store while globally limiting physical read calls.

    ``threading.BoundedSemaphore`` is deliberately used instead of an
    ``asyncio.Semaphore``: Zarr's I/O may run on several event loops in Dask
    worker threads, while an asyncio semaphore belongs to exactly one loop.
    """

    def __init__(
        self, store: Any, max_concurrency: int | None, stats: ReadStats
    ) -> None:
        if max_concurrency is not None and max_concurrency < 1:
            raise ValueError("max_read_concurrency must be positive or None")
        self._store = store
        self._semaphore = (
            threading.BoundedSemaphore(max_concurrency)
            if max_concurrency is not None
            else None
        )
        self._stats = stats

    @property
    def stats(self) -> dict[str, int]:
        """Snapshot of reads made through this wrapper's shared reader stats."""
        return self._stats.snapshot()

    def __getattr__(self, name: str) -> Any:
        """Expose the complete underlying obstore surface unchanged."""
        return getattr(self._store, name)

    async def _acquire_async(self) -> None:
        if self._semaphore is None:
            return
        acquire_task = asyncio.create_task(asyncio.to_thread(self._semaphore.acquire))
        try:
            await asyncio.shield(acquire_task)
        except BaseException:
            # A cancelled waiter can still acquire in the executor. Return that
            # permit when it does, rather than leaking capacity permanently.
            acquire_task.add_done_callback(self._release_if_acquired)
            raise

    def _release_if_acquired(self, task: asyncio.Task[bool]) -> None:
        try:
            task.result()
        except BaseException:
            return
        assert self._semaphore is not None
        self._semaphore.release()

    async def _call_async(self, method: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        await self._acquire_async()
        self._stats.started()
        try:
            result = await method(*args, **kwargs)
        except BaseException:
            self._stats.finished()
            raise
        else:
            self._stats.finished(result)
            return result
        finally:
            if self._semaphore is not None:
                self._semaphore.release()

    def _call_sync(self, method: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        if self._semaphore is not None:
            self._semaphore.acquire()
        self._stats.started()
        try:
            result = method(*args, **kwargs)
        except BaseException:
            self._stats.finished()
            raise
        else:
            self._stats.finished(result)
            return result
        finally:
            if self._semaphore is not None:
                self._semaphore.release()

    async def get_range_async(self, *args: Any, **kwargs: Any) -> Any:
        return await self._call_async(self._store.get_range_async, *args, **kwargs)

    async def get_ranges_async(self, *args: Any, **kwargs: Any) -> Any:
        return await self._call_async(self._store.get_ranges_async, *args, **kwargs)

    async def get_async(self, *args: Any, **kwargs: Any) -> Any:
        return await self._call_async(self._store.get_async, *args, **kwargs)

    def get_range(self, *args: Any, **kwargs: Any) -> Any:
        return self._call_sync(self._store.get_range, *args, **kwargs)
