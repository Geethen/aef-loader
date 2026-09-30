"""Regression tests: replaced objects must never be served from stale caches."""

from __future__ import annotations

import numpy as np
from _cog_harness import local_reader, pattern, read_tile, write_cog

from aef_loader.blockcache import BlockCache


def test_block_cache_evict_url_and_identity():
    cache = BlockCache(1000)
    cache.put(("a", 0, 4, None), b"xxxx")
    cache.put(("a", 4, 8, None), b"yyyy")
    cache.put(("b", 0, 4, None), b"zzzz")
    assert cache.evict_url("a") == 2
    assert cache.get(("a", 0, 4, None)) is None
    assert cache.get(("b", 0, 4, None)) == b"zzzz"
    assert cache.stats["block_cache_bytes"] == 4

    cache.set_identity("b", ("etag-1", 4))
    assert cache.identity("b") == ("etag-1", 4)
    assert cache.get(("b", 0, 4, None)) is None  # a new identity drops old blocks


async def test_replaced_object_with_disk_manifest_and_block_cache(tmp_path):
    """Header and pixels of a replaced object must come from the new object."""
    tile = write_cog(tmp_path, "s.tif")
    old = pattern(2020)
    new = (old + 1).astype("int8")
    reader = local_reader(
        tmp_path,
        manifest_cache_dir=tmp_path / "manifests",
        manifest_validation="head",
        memory_manifest_cache_size=0,
        block_cache_bytes=10**8,
    )
    assert (await read_tile(reader, tile) == old).all()

    write_cog(tmp_path, "s.tif", data=new, compress="none")
    result = await read_tile(reader, tile)

    assert reader.stats["manifest_stale"] == 1
    assert (result == new).all()


async def test_replaced_object_with_block_cache_only(tmp_path):
    """Without manifest caches every open reparses; the header must not be cached."""
    tile = write_cog(tmp_path, "s.tif")
    old = pattern(2020)
    new = np.flip(old, axis=2).copy()
    reader = local_reader(tmp_path, memory_manifest_cache_size=0, block_cache_bytes=10**8)
    assert (await read_tile(reader, tile) == old).all()

    write_cog(tmp_path, "s.tif", data=new, compress="none")
    assert (await read_tile(reader, tile) == new).all()
