"""Offline microbenchmarks for optimisations that do not require cloud access."""

from __future__ import annotations

import asyncio
from statistics import median
from time import perf_counter

import geopandas as gpd
import numpy as np
import xarray as xr
from shapely.geometry import box

from aef_loader import AEFIndex, AEFTileInfo, DataSource
from aef_loader.reader import _concat_time_slices
from aef_loader.utils import _DEQUANT_LUT, dequantize_aef


def timed(function, repeats=7):
    samples = []
    for _ in range(repeats):
        started = perf_counter()
        function()
        samples.append(perf_counter() - started)
    return median(samples)


def benchmark_dequantize():
    raw = np.random.default_rng(2026).integers(
        -128, 128, size=(64, 512, 512), dtype=np.int8
    )

    def previous():
        return _DEQUANT_LUT[raw.astype(np.int16) + 128]

    old = timed(previous)
    new = timed(lambda: dequantize_aef(raw))
    print(f"dequantize: {old / new:.2f}x ({old:.4f}s -> {new:.4f}s)")


def benchmark_index():
    size = 100_000
    side = 1000
    x = np.arange(size) % side
    y = np.arange(size) // side
    geometry = [box(a, b, a + 0.9, b + 0.9) for a, b in zip(x, y)]
    gdf = gpd.GeoDataFrame(
        {
            "fid": np.arange(size),
            "path": [f"s3://bucket/{i}.tif" for i in range(size)],
            "year": 2020 + np.arange(size) % 5,
            "wgs84_west": x,
            "wgs84_south": y,
            "wgs84_east": x + 0.9,
            "wgs84_north": y + 0.9,
            "crs": "EPSG:32632",
            "utm_zone": "32N",
            "utm_west": x * 100,
            "utm_south": y * 100,
            "utm_east": (x + 1) * 100,
            "utm_north": (y + 1) * 100,
        },
        geometry=geometry,
        crs="EPSG:4326",
    )
    query_box = box(400.2, 40.2, 420.2, 60.2)
    # Build the spatial index outside timing, as it is retained across queries.
    _ = gdf.sindex

    def previous():
        working = gdf.copy()
        working = working[working.geometry.intersects(query_box)]
        working = working[(working["year"] >= 2020) & (working["year"] <= 2024)]
        tiles = []
        for _, row in working.iterrows():
            crs = str(row["crs"])
            tiles.append(
                AEFTileInfo(
                    id=str(row["fid"]),
                    path=row["path"],
                    year=row["year"],
                    bbox=(
                        row["wgs84_west"],
                        row["wgs84_south"],
                        row["wgs84_east"],
                        row["wgs84_north"],
                    ),
                    crs_epsg=int(crs.split(":", 1)[1]),
                    utm_zone=row["utm_zone"],
                    utm_bounds=(
                        row["utm_west"],
                        row["utm_south"],
                        row["utm_east"],
                        row["utm_north"],
                    ),
                    source=DataSource.SOURCE_COOP,
                )
            )
        return tiles

    index = AEFIndex(source=DataSource.SOURCE_COOP)
    index._gdf = gdf

    def optimized():
        return asyncio.run(index.query(bbox=query_box.bounds, years=(2020, 2024)))

    old = timed(previous)
    new = timed(optimized)
    print(f"index query: {old / new:.2f}x ({old:.4f}s -> {new:.4f}s)")


def report_graph_and_memory_reductions():
    first = xr.Dataset(
        {"v": (("time", "x"), np.ones((1, 2048), dtype=np.int8))},
        coords={"time": [2020], "x": np.arange(2048)},
    )
    second = xr.Dataset(
        {"v": (("time", "x"), np.ones((1, 2048), dtype=np.int8))},
        coords={"time": [2021], "x": np.arange(1024, 3072)},
    )
    previous = xr.concat([first, second], dim="time", join="outer")
    optimized = _concat_time_slices([first, second])
    print(
        "temporal concat memory: "
        f"{previous.v.nbytes / optimized.v.nbytes:.2f}x "
        f"({previous.v.dtype} -> {optimized.v.dtype})"
    )
    print(
        "8192x8192x64 chunk count: "
        "native=4096, balanced=256 (16.00x fewer), "
        "all-bands=64 (64.00x fewer)"
    )


if __name__ == "__main__":
    benchmark_dequantize()
    benchmark_index()
    report_graph_and_memory_reductions()
