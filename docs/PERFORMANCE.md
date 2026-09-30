# Performance options

The default reader output remains unchanged. The performance controls only
change query execution, task grouping, caching, scheduling, or optional source
pixel subsetting; none of them resample or modify downloaded embedding values.

## Recommended use

    async with VirtualTiffReader(
        manifest_cache_dir="/local-ssd/aef-manifests",
        memory_manifest_cache_size=128,
    ) as reader:
        tree = await reader.open_tiles_by_zone(
            tiles,
            chunks="balanced",
            bbox=aoi_bbox,
            bbox_crs="EPSG:4326",
            max_concurrency=16,
        )
        print(reader.stats)

Use chunks="native" for small windows or small band selections. Use
chunks="balanced" when most or all bands are processed. chunks="all-bands"
minimises graph size but each 1024 by 1024 block expands to 256 MiB when
dequantized to float32, so it requires ample worker memory.

The bbox option crops by source pixel coordinates before the spatial and
temporal mosaics. It includes every pixel cell covering the AOI and performs no
resampling. Set buffer_pixels=1 or more when a later operation needs a kernel
halo.

max_concurrency is shared across all UTM zones. Values between 8 and 16 are a
reasonable starting point; use reader.stats["last_open_wall_seconds"] to tune
for the network and machine. The remaining counters distinguish in-memory
manifest reuse, disk-cache reuse, and remote header parsing.

## Offline benchmark

Run:

    python -m benchmarks.benchmark_optimizations

Observed median ranges across repeated seven-run sessions on the development
Windows machine:

| Optimisation | Before | After | Gain |
|---|---:|---:|---:|
| Dequantize 64 by 512 by 512 int8 array | 77-81 ms | 48-55 ms | 1.4-1.7x |
| Query a 100,000-row synthetic index | 19-32 ms | 3.2-4.7 ms | 5.8-6.9x |
| Temporal outer-join storage | float32 | int8 | 4x less memory |
| Full-tile Dask chunks, balanced profile | 4,096 | 256 | 16x fewer |
| Full-tile Dask chunks, all-bands profile | 4,096 | 64 | 64x fewer |

The index benchmark includes result conversion to AEFTileInfo in both paths.
Chunk counts describe scheduler graph size; cloud request count and elapsed time
depend on the object store and selected window. AOI, manifest-cache and
cross-zone concurrency gains should therefore be measured on the target
deployment using reader.stats.

## Integrity checks

tests/test_optimizations.py verifies:

- all 256 int8 codes against the original float32 formula;
- eager and Dask-backed dequantization equality and chunk preservation;
- int8 nodata and values across mismatched yearly coverage;
- spatial-index results, ordering and limits against the previous full scan;
- AOI output as an unchanged view of source pixels;
- storage-aligned chunk profiles;
- bounded LRU manifest reuse; and
- the shared cross-zone concurrency limit.

## Live Source Cooperative demo

Run:

    python -m benchmarks.demo_download

The 9 September 2026 demo used a roughly 1.1 by 1.1 km AOI near Stavanger,
one 2024 zone-31N tile, and four embedding bands. Results:

| Measurement | Result |
|---|---:|
| Index download | 14.927 s |
| Cold tile open and manifest creation | 3.907 s |
| In-memory manifest reopen | 0.013 s, about 298x faster |
| Disk manifest reopen in a fresh reader | 0.028 s, about 139x faster |
| Full virtual source size | 8192 by 8192 pixels |
| AOI virtual source size | 117 by 121 pixels |
| Pixels excluded before mosaicking | 4740.3x |
| Reference compute | 1.565 s |
| AOI compute | 1.717 s |

The compute timings are effectively equivalent because both comparisons select
the same AOI and the native COG block is the minimum remote read. Cropping still
shrinks the mosaic and downstream graph dramatically for larger multi-tile
jobs.

The reference, in-memory-cache, and disk-cache results had identical
coordinates, attributes, int8 dtype, and bytes. Their shared SHA-256 was:

    1152f5bccf90857bb6c62c3cf378731d489caeda6c32d0c1cc0b6ba320a563fc
