"""Regression tests for read-limiter permit handling."""

from __future__ import annotations

import asyncio
import threading
import time

from aef_loader._limits import ReadLimitedStore, ReadStats


class _Inner:
    def __init__(self) -> None:
        self.release_first = threading.Event()
        self.entered = threading.Event()

    async def get_range_async(self, path, start, end=None, length=None):
        await asyncio.sleep(0.01)
        return b"x" * 10

    def get_range(self, path, start, end=None, length=None):
        self.entered.set()
        self.release_first.wait(10)
        return b"x" * 10


def test_permit_not_leaked_when_loop_closes_while_waiting():
    """A waiter abandoned by asyncio.run shutdown must not swallow or block a permit."""
    inner = _Inner()
    store = ReadLimitedStore(inner, 1, ReadStats())

    holder = threading.Thread(
        target=store.get_range, args=("p",), kwargs={"start": 0, "end": 10}
    )
    holder.start()
    assert inner.entered.wait(5)
    # Free the held permit only after asyncio.run has had to shut down.
    timer = threading.Timer(1.5, inner.release_first.set)
    timer.start()

    async def main():
        try:
            await asyncio.wait_for(store.get_range_async("p", start=0, end=10), 0.2)
        except TimeoutError:
            pass

    started = time.perf_counter()
    asyncio.run(main())
    assert time.perf_counter() - started < 1.0, "asyncio.run blocked on a waiting reader"

    holder.join(5)
    timer.join()
    # The permit must be available again: a fresh read completes promptly.
    result: list[bytes] = []
    inner.release_first.set()
    reader = threading.Thread(
        target=lambda: result.append(store.get_range("p", start=0, end=10))
    )
    reader.start()
    reader.join(3)
    assert not reader.is_alive(), "permit was leaked"
    assert result == [b"x" * 10]


def test_permit_released_when_waiter_cancelled_after_grant():
    inner = _Inner()
    store = ReadLimitedStore(inner, 1, ReadStats())

    async def main():
        await store._limiter.acquire()
        waiter = asyncio.create_task(store.get_range_async("p", start=0, end=10))
        await asyncio.sleep(0.05)
        store._limiter.release()  # hands the permit to the waiter
        waiter.cancel()
        try:
            await waiter
        except asyncio.CancelledError:
            pass
        assert store._limiter.try_acquire()

    asyncio.run(main())
