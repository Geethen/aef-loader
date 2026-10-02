"""Small live Source Cooperative integrity and timing check."""

from __future__ import annotations

import asyncio
import hashlib
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np
import xarray as xr

from aef_loader import AEFIndex, DataSource, VirtualTiffReader

# Roughly 1.1 by 1.1 km near Stavanger. The loader expands this to complete
# source pixel cells, without resampling them.
AOI = (5.72, 58.96, 5.74, 58.97)
YEAR = 2024
BANDS = slice(0, 4)


def digest(data: xr.DataArray) -> str:
    contiguous = np.ascontiguousarray(data.values)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


async def open_timed(reader, tiles, *, crop):
    started = perf_counter()
    tree = await reader.open_tiles_by_zone(
        tiles,
        chunks="native",
        bbox=AOI if crop else None,
        bbox_crs="EPSG:4326",
        max_concurrency=4,
    )
    return tree, perf_counter() - started


async def main():
    with tempfile.TemporaryDirectory(prefix="aef-loader-demo-") as temporary:
        cache_dir = Path(temporary)
        index = AEFIndex(
            source=DataSource.SOURCE_COOP,
            cache_dir=cache_dir / "index",
        )

        started = perf_counter()
        index_path = await index.download()
        index_download_seconds = perf_counter() - started
        index.load(index_path)

        started = perf_counter()
        tiles = await index.query(bbox=AOI, years=YEAR)
        query_seconds = perf_counter() - started
        if not tiles:
            raise RuntimeError("Demo AOI returned no AEF tiles")

        manifest_dir = cache_dir / "manifests"
        async with VirtualTiffReader(
            manifest_cache_dir=manifest_dir,
            memory_manifest_cache_size=16,
        ) as reader:
            reference_tree, cold_open_seconds = await open_timed(
                reader, tiles, crop=False
            )
            cropped_tree, memory_open_seconds = await open_timed(
                reader, tiles, crop=True
            )
            memory_stats = reader.stats

            zone = next(iter(cropped_tree.children))
            cropped = cropped_tree[zone].ds["embeddings"].isel(time=0, band=BANDS)
            reference = (
                reference_tree[zone]
                .ds["embeddings"]
                .sel(x=cropped.x.values, y=cropped.y.values)
                .isel(time=0, band=BANDS)
            )

            started = perf_counter()
            reference_values = reference.compute()
            reference_compute_seconds = perf_counter() - started
            started = perf_counter()
            cropped_values = cropped.compute()
            cropped_compute_seconds = perf_counter() - started

        # A fresh reader exercises reconstruction from the on-disk manifest.
        async with VirtualTiffReader(
            manifest_cache_dir=manifest_dir,
            memory_manifest_cache_size=16,
        ) as disk_reader:
            disk_tree, disk_open_seconds = await open_timed(
                disk_reader, tiles, crop=True
            )
            disk_stats = disk_reader.stats
            disk_values = (
                disk_tree[zone].ds["embeddings"].isel(time=0, band=BANDS).compute()
            )

        xr.testing.assert_identical(reference_values, cropped_values)
        xr.testing.assert_identical(reference_values, disk_values)
        reference_hash = digest(reference_values)
        assert reference_hash == digest(cropped_values) == digest(disk_values)

        full_sizes = dict(reference_tree[zone].ds.sizes)
        cropped_sizes = dict(cropped_tree[zone].ds.sizes)
        full_pixels = full_sizes["x"] * full_sizes["y"]
        cropped_pixels = cropped_sizes["x"] * cropped_sizes["y"]

        print(f"tiles={len(tiles)} zone={zone} year={YEAR}")
        print(f"index_download_seconds={index_download_seconds:.3f}")
        print(f"query_seconds={query_seconds:.4f}")
        print(f"cold_open_seconds={cold_open_seconds:.3f}")
        print(f"memory_open_seconds={memory_open_seconds:.3f}")
        print(f"disk_open_seconds={disk_open_seconds:.3f}")
        print(f"reference_compute_seconds={reference_compute_seconds:.3f}")
        print(f"cropped_compute_seconds={cropped_compute_seconds:.3f}")
        print(f"full_sizes={full_sizes}")
        print(f"cropped_sizes={cropped_sizes}")
        print(f"source_pixel_reduction={full_pixels / cropped_pixels:.1f}x")
        print(f"sha256={reference_hash}")
        print(f"memory_reader_stats={memory_stats}")
        print(f"disk_reader_stats={disk_stats}")
        print("integrity=PASS (coordinates, attrs, dtype and bytes identical)")


if __name__ == "__main__":
    asyncio.run(main())
