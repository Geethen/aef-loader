"""Thread- and event-loop-safe limits for physical object-store reads."""

from __future__ import annotations

import asyncio
import threading
from collections import deque
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


class _Waiter:
    """One blocked acquirer: an asyncio future or a thread event."""

    __slots__ = ("event", "future", "granted", "loop")

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop | None = None,
        future: asyncio.Future[None] | None = None,
        event: threading.Event | None = None,
    ) -> None:
        self.loop = loop
        self.future = future
        self.event = event
        self.granted = False


def _grant(future: asyncio.Future[None]) -> None:
    if not future.done():
        future.set_result(None)


class ReadLimiter:
    """Counting limiter shared by threads and any number of event loops.

    Waiters queue in FIFO order and a released permit is handed straight to the
    next one (no executor threads sit blocked on a semaphore). A permit granted to
    a waiter that was abandoned in the meantime (task cancelled, loop closed) is
    passed on rather than lost.
    """

    def __init__(self, max_concurrency: int) -> None:
        if max_concurrency < 1:
            raise ValueError("max_read_concurrency must be positive or None")
        self.max_concurrency = max_concurrency
        self._lock = threading.Lock()
        self._available = max_concurrency
        self._waiters: deque[_Waiter] = deque()

    def try_acquire(self) -> bool:
        with self._lock:
            if self._available > 0 and not self._waiters:
                self._available -= 1
                return True
            return False

    async def acquire(self) -> None:
        loop = asyncio.get_running_loop()
        with self._lock:
            if self._available > 0 and not self._waiters:
                self._available -= 1
                return
            waiter = _Waiter(loop=loop, future=loop.create_future())
            self._waiters.append(waiter)
        try:
            await waiter.future
        except BaseException:
            self._abandon(waiter)
            raise

    def acquire_sync(self) -> None:
        with self._lock:
            if self._available > 0 and not self._waiters:
                self._available -= 1
                return
            waiter = _Waiter(event=threading.Event())
            self._waiters.append(waiter)
        try:
            waiter.event.wait()
        except BaseException:
            self._abandon(waiter)
            raise

    def _abandon(self, waiter: _Waiter) -> None:
        """Undo a waiter that will never use its permit."""
        with self._lock:
            granted = waiter.granted
            if not granted:
                try:
                    self._waiters.remove(waiter)
                except ValueError:
                    pass
        if granted:
            self.release()

    def release(self) -> None:
        with self._lock:
            while self._waiters:
                waiter = self._waiters.popleft()
                waiter.granted = True
                if waiter.event is not None:
                    waiter.event.set()
                    return
                try:
                    waiter.loop.call_soon_threadsafe(_grant, waiter.future)
                    return
                except RuntimeError:  # its loop is closed: offer it to the next
                    continue
            if self._available >= self.max_concurrency:
                raise ValueError("ReadLimiter released too many times")
            self._available += 1


class ReadLimitedStore:
    """Delegate an object store while globally limiting physical read calls.

    A :class:`ReadLimiter` is used instead of an ``asyncio.Semaphore``: Zarr's I/O
    may run on several event loops in Dask worker threads, while an asyncio
    semaphore belongs to exactly one loop.
    """

    def __init__(
        self, store: Any, max_concurrency: int | None, stats: ReadStats
    ) -> None:
        if max_concurrency is not None and max_concurrency < 1:
            raise ValueError("max_read_concurrency must be positive or None")
        self._store = store
        self._limiter = (
            ReadLimiter(max_concurrency) if max_concurrency is not None else None
        )
        self._stats = stats

    @property
    def stats(self) -> dict[str, int]:
        """Snapshot of reads made through this wrapper's shared reader stats."""
        return self._stats.snapshot()

    def __getattr__(self, name: str) -> Any:
        """Expose the complete underlying obstore surface unchanged."""
        return getattr(self._store, name)

    async def _call_async(self, method: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        if self._limiter is not None:
            await self._limiter.acquire()
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
            if self._limiter is not None:
                self._limiter.release()

    def _call_sync(self, method: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        if self._limiter is not None:
            self._limiter.acquire_sync()
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
            if self._limiter is not None:
                self._limiter.release()

    async def get_range_async(self, *args: Any, **kwargs: Any) -> Any:
        return await self._call_async(self._store.get_range_async, *args, **kwargs)

    async def get_ranges_async(self, *args: Any, **kwargs: Any) -> Any:
        return await self._call_async(self._store.get_ranges_async, *args, **kwargs)

    async def get_async(self, *args: Any, **kwargs: Any) -> Any:
        return await self._call_async(self._store.get_async, *args, **kwargs)

    def get_range(self, *args: Any, **kwargs: Any) -> Any:
        return self._call_sync(self._store.get_range, *args, **kwargs)
