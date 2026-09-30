"""Regression tests for read_chips concurrency limits."""

from __future__ import annotations

import threading
import time

from _cog_harness import NORTH, SIZE, WEST, local_reader, write_cog

import aef_loader.reader as reader_module
from aef_loader import read_chips


async def test_max_concurrency_bounds_tile_opens_across_groups(tmp_path, monkeypatch):
    # Disjoint tiles: every chip lands in its own group.
    tiles = [write_cog(tmp_path, f"t{i}.tif", col_off=i * SIZE * 2) for i in range(6)]
    points = [(WEST + i * SIZE * 20 + 1000, NORTH - 1000) for i in range(6)]

    original_call = reader_module.VirtualTIFF.__call__
    lock = threading.Lock()
    state = {"active": 0, "peak": 0}

    def counting_call(self, *args, **kwargs):
        with lock:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        try:
            time.sleep(0.2)
            return original_call(self, *args, **kwargs)
        finally:
            with lock:
                state["active"] -= 1

    monkeypatch.setattr(reader_module.VirtualTIFF, "__call__", counting_call)

    chips = await read_chips(
        points,
        size=4,
        years=2020,
        index=tiles,
        reader=local_reader(tmp_path),
        crs="EPSG:32631",
        max_concurrency=1,
    )

    assert len(chips) == 6
    assert state["peak"] <= 1
