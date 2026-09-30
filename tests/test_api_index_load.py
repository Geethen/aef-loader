"""Offline tests: the first index load is single-flight across callers, loops and threads."""

import asyncio
import threading

import pytest

from aef_loader.api import ensure_index_loaded


class Index:
    def __init__(self, fail_first=False):
        self._df = None
        self.downloads = 0
        self.fail_first = fail_first
        self.started = threading.Event()

    async def download(self):
        self.downloads += 1
        self.started.set()
        await asyncio.sleep(0.05)  # every concurrent caller sees _df is None
        if self.fail_first and self.downloads == 1:
            raise OSError("download failed")

    def load(self):
        self._df = object()


async def test_concurrent_callers_on_one_loop_download_once():
    index = Index()
    results = await asyncio.gather(*(ensure_index_loaded(index) for _ in range(20)))
    assert index.downloads == 1
    assert all(r is index for r in results)
    await ensure_index_loaded(index)  # already loaded: no further download
    assert index.downloads == 1


def test_callers_on_different_loops_and_threads_download_once():
    index = Index()
    errors = []

    def call():
        try:
            asyncio.run(ensure_index_loaded(index))
        except BaseException as exc:  # noqa: BLE001 - surfaced by the assertion below
            errors.append(exc)

    first = threading.Thread(target=call)
    first.start()
    assert index.started.wait(5)  # thread 1 is mid-download when thread 2 starts
    second = threading.Thread(target=call)
    second.start()
    first.join(10)
    second.join(10)
    assert not errors
    assert index.downloads == 1
    assert index._df is not None


async def test_failed_first_load_raises_in_all_waiters_and_retries():
    index = Index(fail_first=True)
    results = await asyncio.gather(
        *(ensure_index_loaded(index) for _ in range(5)), return_exceptions=True
    )
    assert all(isinstance(r, OSError) for r in results)
    assert index.downloads == 1
    assert index._df is None
    assert await ensure_index_loaded(index) is index  # a later call retries
    assert index.downloads == 2
    assert index._df is not None


async def test_cancelled_loader_lets_a_waiter_take_over():
    index = Index()
    loader = asyncio.ensure_future(ensure_index_loaded(index))
    await asyncio.sleep(0.01)
    waiter = asyncio.ensure_future(ensure_index_loaded(index))
    await asyncio.sleep(0.01)
    loader.cancel()
    with pytest.raises(asyncio.CancelledError):
        await loader
    assert await waiter is index
    assert index.downloads == 2
