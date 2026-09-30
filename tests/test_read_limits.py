from __future__ import annotations

import asyncio
import threading

from aef_loader.reader import VirtualTiffReader


class FakeAsyncStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.peak = 0

    async def get_range_async(self, path, *, start, end, **kwargs):
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(0.02)
            return b"x" * (end - start)
        finally:
            with self._lock:
                self.active -= 1


def test_read_limit_is_shared_by_event_loops_in_two_threads(monkeypatch):
    store = FakeAsyncStore()
    reader = VirtualTiffReader(max_read_concurrency=3)
    monkeypatch.setattr(reader, "_get_s3_store", lambda bucket: store)
    limited = reader._get_store("s3", "bucket")
    errors: list[BaseException] = []

    def run_reads() -> None:
        try:
            async def reads() -> None:
                await asyncio.gather(
                    *(limited.get_range_async("tile", start=0, end=7) for _ in range(20))
                )

            asyncio.run(reads())
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=run_reads) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert store.peak == 3
    assert {
        name: reader.stats[name]
        for name in ("read_requests", "read_bytes", "read_peak_inflight")
    } == {
        "read_requests": 40,
        "read_bytes": 280,
        "read_peak_inflight": 3,
    }


def test_no_limit_leaves_reader_store_unwrapped(monkeypatch):
    raw_store = object()
    reader = VirtualTiffReader(max_read_concurrency=None)
    monkeypatch.setattr(reader, "_get_s3_store", lambda bucket: raw_store)

    assert reader._get_store("s3", "bucket") is raw_store
