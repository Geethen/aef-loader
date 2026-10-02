"""Offline tests for the compressed-block LRU cache."""

import asyncio
import threading

import pytest

from aef_loader.blockcache import BlockCache, CachingStore
from aef_loader.reader import VirtualTiffReader


class FakeStore:
    """Stand-in for an obstore store that counts ranged reads."""

    prefix = "pre"

    def __init__(self, blob: bytes = bytes(range(256)) * 16):
        self.blob = blob
        self.calls: list[tuple[str, int, int | None]] = []

    async def get_range_async(self, path, start, end=None, length=None):
        if end is None:
            end = start + length
        self.calls.append((path, start, end))
        return self.blob[start:end]

    def get_range(self, path, start, end=None, length=None):
        if end is None:
            end = start + length
        self.calls.append((path, start, end))
        return self.blob[start:end]


def test_hit_miss_and_identical_bytes():
    store, cache = FakeStore(), BlockCache(10_000)
    cs = CachingStore(store, cache, "s3://bucket")

    first = asyncio.run(cs.get_range_async("a.tif", start=10, end=110))
    second = asyncio.run(cs.get_range_async("a.tif", start=10, end=110))
    by_length = asyncio.run(cs.get_range_async("a.tif", start=10, length=100))

    assert bytes(first) == bytes(second) == bytes(by_length) == store.blob[10:110]
    assert len(store.calls) == 1
    stats = cache.stats
    assert stats["block_cache_hits"] == 2
    assert stats["block_cache_misses"] == 1
    assert stats["block_cache_bytes"] == 100


def test_key_separates_path_and_range_and_sync_path_shares_cache():
    store, cache = FakeStore(), BlockCache(10_000)
    cs = CachingStore(store, cache, "s3://bucket/")
    cs.get_range("a.tif", start=0, end=8)
    cs.get_range("b.tif", start=0, end=8)
    cs.get_range("a.tif", start=0, end=9)
    cs.get_range("a.tif", start=0, end=8)
    assert len(store.calls) == 3
    assert cache.stats["block_cache_hits"] == 1


def test_unbounded_read_is_not_cached_and_attrs_delegate():
    store, cache = FakeStore(), BlockCache(1000)
    cs = CachingStore(store, cache, "s3://bucket")
    assert cs.prefix == "pre"
    with pytest.raises(TypeError):  # length/end both missing: passed straight through
        asyncio.run(cs.get_range_async("a", start=0))
    assert cache.stats["block_cache_misses"] == 0


def test_byte_bound_and_lru_eviction():
    cache = BlockCache(10)
    for name in "abc":
        cache.put((name, 0, 4), b"xxxx")
    assert cache.stats["block_cache_bytes"] == 8  # a was evicted, b and c remain
    assert cache.get(("a", 0, 4)) is None
    assert cache.get(("b", 0, 4)) == b"xxxx"  # touch b: c is now the LRU
    cache.put(("d", 0, 4), b"yyyy")
    assert cache.get(("c", 0, 4)) is None
    assert cache.get(("b", 0, 4)) is not None
    assert cache.stats["block_cache_bytes"] <= 10
    assert cache.stats["block_cache_evictions"] == 2


def test_oversized_entry_is_skipped():
    cache = BlockCache(4)
    cache.put(("big", 0, 9), b"123456789")
    assert cache.stats["block_cache_bytes"] == 0
    assert cache.stats["block_cache_entries"] == 0


def test_thread_safety_keeps_bound():
    cache = BlockCache(500)

    def work(seed):
        for i in range(500):
            key = ("f", (seed * 7 + i) % 40, 0)
            if cache.get(key) is None:
                cache.put(key, b"z" * 50)

    threads = [threading.Thread(target=work, args=(s,)) for s in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stats = cache.stats
    assert stats["block_cache_bytes"] <= 500
    assert stats["block_cache_hits"] + stats["block_cache_misses"] == 8 * 500


def test_reader_wraps_stores_and_reports_stats(monkeypatch):
    fake = FakeStore()
    monkeypatch.setattr(VirtualTiffReader, "_get_s3_store", lambda self, bucket: fake)
    reader = VirtualTiffReader(block_cache_bytes=1000)
    store = reader._get_store("s3", "bucket")
    assert isinstance(store, CachingStore)
    assert reader._get_store("s3", "bucket") is store
    asyncio.run(store.get_range_async("k", start=0, end=4))
    asyncio.run(store.get_range_async("k", start=0, end=4))
    stats = reader.stats
    assert (stats["block_cache_hits"], stats["block_cache_misses"]) == (1, 1)
    assert stats["block_cache_bytes"] == 4


def test_default_reader_has_no_cache(monkeypatch):
    fake = FakeStore()
    monkeypatch.setattr(VirtualTiffReader, "_get_s3_store", lambda self, bucket: fake)
    reader = VirtualTiffReader()
    assert reader._get_store("s3", "bucket") is fake
    assert "block_cache_hits" not in reader.stats
