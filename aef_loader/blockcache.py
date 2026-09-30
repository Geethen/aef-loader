"""
Byte-bounded LRU cache of compressed chunk bytes, in front of an object store.

A COG chunk read by zarr is one ranged GET of a whole compressed block (about
435 KB for AEF), and reading a small chip costs one block per band. When several
chips (or several calls) touch the same blocks, this cache lets the later reads
skip the network.

The seam is the obstore store registered in the ``ObjectStoreRegistry``:
``ManifestStore.get`` resolves the store for a chunk's URL and awaits exactly
``store.get_range_async(path, start=, end=)``. :class:`CachingStore` wraps that
store, so the cache holds the compressed bytes exactly as fetched (decoding still
happens in zarr) and is shared by every tile opened through the same reader.
"""

from __future__ import annotations

import threading
from collections import OrderedDict


class BlockCache:
    """Thread-safe LRU of ``(url, start, end, identity) -> bytes`` bounded by total bytes.

    ``identity`` is the object's known version (ETag and size), so bytes read from
    one version of an object are never served for another. The reader records it
    with :meth:`set_identity`; keys are opaque to the cache except that the first
    element is the URL, which :meth:`evict_url` uses.

    zarr runs its I/O on its own event-loop thread while dask workers and the
    caller's thread read the statistics, so every access is under a lock. An entry
    larger than the whole budget is never stored.
    """

    def __init__(self, max_bytes: int):
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self.max_bytes = int(max_bytes)
        self._items: OrderedDict[tuple, bytes] = OrderedDict()
        self._identities: dict[str, object] = {}
        self._lock = threading.Lock()
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: tuple) -> bytes | None:
        with self._lock:
            value = self._items.get(key)
            if value is None:
                self._misses += 1
                return None
            self._items.move_to_end(key)
            self._hits += 1
            return value

    def put(self, key: tuple, value: bytes) -> None:
        size = len(value)
        if size > self.max_bytes:
            return
        with self._lock:
            old = self._items.pop(key, None)
            if old is not None:
                self._bytes -= len(old)
            self._items[key] = value
            self._bytes += size
            while self._bytes > self.max_bytes:
                _, evicted = self._items.popitem(last=False)
                self._bytes -= len(evicted)
                self._evictions += 1

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._identities.clear()
            self._bytes = 0

    def _evict_url_locked(self, url: str) -> int:
        doomed = [key for key in self._items if key[0] == url]
        for key in doomed:
            self._bytes -= len(self._items.pop(key))
        self._evictions += len(doomed)
        return len(doomed)

    def evict_url(self, url: str) -> int:
        """Drop every cached block of ``url``; return how many were removed."""
        with self._lock:
            return self._evict_url_locked(url)

    def identity(self, url: str) -> object:
        """The identity recorded for ``url`` (None when unknown)."""
        with self._lock:
            return self._identities.get(url)

    def set_identity(self, url: str, identity: object) -> None:
        """Record ``url``'s current identity; drop its blocks if that changed."""
        with self._lock:
            if self._identities.get(url) != identity:
                self._evict_url_locked(url)
                self._identities[url] = identity

    @property
    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "block_cache_hits": self._hits,
                "block_cache_misses": self._misses,
                "block_cache_bytes": self._bytes,
                "block_cache_entries": len(self._items),
                "block_cache_evictions": self._evictions,
            }


class CachingStore:
    """Wrap an obstore store, caching ``get_range[_async]`` results in a BlockCache.

    Only bounded ranges are cached (``end`` or ``length`` given); every other
    attribute (``prefix``, ``url``, ``get_ranges``, ...) is delegated unchanged,
    so the wrapper is a drop-in for the registry and for the TIFF header parser.
    ``base_url`` (e.g. ``s3://bucket``) keys entries so that one cache can serve
    several buckets. Entries are also keyed by the object identity the cache knows
    for that URL, so a replaced object never hits blocks of its previous version.
    """

    def __init__(self, store, cache: BlockCache, base_url: str):
        self._store = store
        self._cache = cache
        self._base_url = base_url.rstrip("/")

    def __getattr__(self, name: str):
        return getattr(self._store, name)

    def _key(self, path: str, start: int, end: int | None, length: int | None):
        if end is None and length is not None:
            end = start + length
        if end is None:
            return None
        url = f"{self._base_url}/{path}"
        return (url, int(start), int(end), self._cache.identity(url))

    async def get_range_async(self, path, start, end=None, length=None):
        key = self._key(path, start, end, length)
        if key is None:
            return await self._store.get_range_async(
                path, start=start, end=end, length=length
            )
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        result = await self._store.get_range_async(
            path, start=start, end=end, length=length
        )
        self._cache.put(key, bytes(result))
        return result

    def get_range(self, path, start, end=None, length=None):
        key = self._key(path, start, end, length)
        if key is None:
            return self._store.get_range(path, start=start, end=end, length=length)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        result = self._store.get_range(path, start=start, end=end, length=length)
        self._cache.put(key, bytes(result))
        return result
