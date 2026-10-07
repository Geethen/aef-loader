# Repo review — 2026-09-30 (GPT-6 Astra)

Second, independent review of `aef_loader_plus` at commit `658436c`, produced by **GPT-6 Astra**
via the Codex CLI (`codex exec --model gpt-6-astra`, reasoning effort high, read-only sandbox).
It was given the same brief as [REVIEW-2026-09-30.md](REVIEW-2026-09-30.md) — usability,
extensibility, cloud-native speed, correctness — and worked independently, reading the package,
running the offline test suite and reproducing failures in memory. It made no edits.

Where it says **Verified**, it reproduced the behaviour by execution. Its cloud footer probe was
blocked by the sandbox, so the index-layout measurements in the companion review
(single row group, per-column byte sizes) are that review's evidence, not this one's.

Two of its findings overturn or refine the companion review; those are noted in
[REVIEW-2026-09-30.md](REVIEW-2026-09-30.md) under "Cross-check".

Text below is verbatim.

---

## 1. Usability

1. **P0 — Provide a short, notebook-safe AEF entrypoint.** AEF requires download → load → asynchronous query → asynchronous reader, while TESSERA exposes one synchronous function. `AEFIndex.query()` contains no `await`; it blocks the event loop while filtering. Add `open_aef(...)` plus an explicitly named asynchronous equivalent, retaining compatibility with existing methods. Show `await main()` for notebooks: the README’s `asyncio.run(main())` fails inside an already-running event loop.  
   **Locations:** `aef_loader/index.py:222`, `aef_loader/reader.py:422`, `aef_loader/tessera.py:68`, `README.md:59`, `README.md:78`.

2. **P1 — Align defaults with the recommended host.** `AEFIndex()` defaults to requester-pays GCS and subsequently requires `gcp_project`, although the README recommends public Source Cooperative. Passing `"source_coop"` instead of its enum also silently takes the GCS branch. Default to Source Cooperative; normalize supported strings or reject them immediately.  
   **Locations:** `aef_loader/index.py:64`, `aef_loader/index.py:85`, `aef_loader/index.py:126`, `aef_loader/index.py:137`, `README.md:107`.

3. **P1 — Make cache behavior discoverable and portable.** The index uses `Path("/tmp")`; the manifest cache accepts strings and normalizes them, but the index cache does not. Use a platform-specific user cache, accept `str | Path` consistently, and expose cache location, refresh, clear, size limits, and offline mode. Explain that manifest caching saves headers, **not pixels**.  
   **Locations:** `aef_loader/index.py:66`, `aef_loader/index.py:78`, `aef_loader/index.py:119`, `aef_loader/reader.py:321`, `aef_loader/reader.py:340`, `aef_loader/cache.py:12`.

4. **P1 — Document chunk profiles as workload choices.** The actual default is `"auto"`; `None` means native Dask chunks rather than xarray’s usual non-Dask behavior. Users need a compact table covering few-band reads, full-vector chips, reprojection, and worker memory. Include `.isel(band=...)` versus AEF’s string `.sel(band=["A00", ...])`.  
   **Locations:** `aef_loader/reader.py:219`, `aef_loader/reader.py:426`, `aef_loader/reader.py:449`, `aef_loader/reader.py:701`, `docs/PERFORMANCE.md:22`.

5. **P1 — Define one selection and empty-result policy.** AEF accepts a year or range; TESSERA additionally accepts arbitrary iterables and silently drops unavailable years. AEF returns `[]`, the reader raises `"No tiles provided"`, and TESSERA can return zones containing zero time slices. Expose available years and distinguish unavailable years, no spatial coverage, and invalid input. Add bbox/year/zone context to errors.  
   **Locations:** `aef_loader/index.py:188`, `aef_loader/index.py:279`, `aef_loader/reader.py:476`, `aef_loader/tessera.py:148`, `aef_loader/tessera.py:170`.

6. **P1 — Explain the output contract and materialization boundary.** The AEF quick start finishes with a lazy object and neither returns nor computes it. TESSERA’s “nothing downloaded yet” claim overlooks synchronous remote metadata opening. Document metadata I/O, deferred pixel I/O, `time` coordinate types, native grid orientation, missing-data semantics, and how to obtain a bounded NumPy result.  
   **Locations:** `README.md:75`, `README.md:92`, `aef_loader/tessera.py:43`, `aef_loader/tessera.py:105`, `aef_loader/tessera.py:135`, `aef_loader/reader.py:643`.

7. **P1 — Correct stale documentation.** `reproject_datatree()` describes `combine_first`, but implements `xr.where`; the changelog says the Dask dequantization path is unchanged, but it now uses `map_blocks`. The recorded diff likewise predates that change. The `aoi_geobox` explanation incorrectly says current `GeoBox.from_bbox()` does not snap by default; see section 4.  
   **Locations:** `aef_loader/utils.py:327`, `aef_loader/utils.py:438`, `docs/CHANGES.md:40`, `aef_loader/utils.py:117`, `docs/upstream.diff:319`, `aef_loader/utils.py:460`.

8. **P2 — Separate lightweight discovery from the complete geospatial stack.** Importing `open_tessera` also imports the index, reader, GeoPandas, and VirtualiZarr through package initialization. Consider lazy public exports and optional backend extras. Rasterio is relevant to the exposed ODC reprojection path; removing it merely because there is no direct `import rasterio` would be inappropriate. Document GCS credentials as well as billing-project configuration.  
   **Locations:** `aef_loader/__init__.py:14`, `aef_loader/index.py:13`, `aef_loader/reader.py:27`, `aef_loader/utils.py:12`, `aef_loader/utils.py:397`, `pyproject.toml:18`, `README.md:108`.

9. **P2 — Make the default test instructions genuinely offline.** README says `pytest`, but marking a test `slow` does not exclude it. Consequently, the documented command includes the remote TESSERA smoke test; CI explicitly excludes it. Document both commands.  
   **Locations:** `README.md:45`, `README.md:115`, `tests/test_tessera.py:15`, `pyproject.toml:66`, `.github/workflows/tests.yml:12`.

## 2. Extensibility / future features

1. **P1 — Separate dataset semantics, discovery, storage, and assembly.** The current reader combines AEF naming, nodata, annual timestamps, cloud access, TIFF parsing, and mosaicking. TESSERA independently implements another selection and grid path. Introduce small contracts:
   - `DatasetSpec`: identity/version, dimensions, band names, time representation, decoding, validity mask.
   - `TileIndex.query(...)`: discovery independent of encoding.
   - `AssetOpener`: COG or Zarr → a native-grid dataset.
   - Assembly: grouping, selection, reprojection, and overlap policy.

   Keep AEF’s nonlinear decoder and TESSERA’s codes-plus-scales decoder separate. This directly addresses the cross-dataset correctness problems below.  
   **Hooks:** `aef_loader/types.py:18`, `aef_loader/reader.py:157`, `aef_loader/reader.py:546`, `aef_loader/reader.py:696`, `aef_loader/tessera.py:113`, `aef_loader/tessera.py:160`.

2. **P1 — Inject storage configuration instead of adding protocol branches.** Every S3 bucket currently receives Source Cooperative’s region and anonymous access; every GCS bucket requires a billing project. HTTP, local COGs, Azure, private S3, and alternative endpoints require changes to multiple modules. Accept an object-store registry or factory plus per-host options; keep the existing hosts as presets. Include endpoint/authentication identity in store reuse rules.  
   **Hooks:** `aef_loader/constants.py:6`, `aef_loader/index.py:130`, `aef_loader/index.py:145`, `aef_loader/reader.py:62`, `aef_loader/reader.py:378`, `aef_loader/reader.py:391`, `aef_loader/reader.py:399`.

3. **P1 — Add point/time-series extraction and zonal statistics before a large plugin framework.** These are natural downstream tasks for annual embeddings. Group points and polygons by tile and physical block; select native pixels, deduplicate reads, and return `(feature, time, band)` plus coverage/validity. For polygon means, decode before aggregation: averaging AEF codes and then decoding is not equivalent. Define area weighting, overlap ownership, and whether output vectors are normalized.  
   **Hooks:** `aef_loader/index.py:222`, `aef_loader/reader.py:244`, `aef_loader/utils.py:78`, `aef_loader/utils.py:309`. A new extraction module should compose these operations rather than require a full-AOI mosaic.

4. **P1 — Promote chip extraction into a supported API.** The benchmark already implements bounds, cropping, band ordering, orientation normalization, and NumPy materialization. Extract a `read_chip`/`iter_chips` interface with explicit size, grid, halo, edge padding, and validity mask. For training, group patches by source block, reuse decoded blocks, bound prefetching, and expose reproducible sampling. Make PyTorch integration optional.  
   **Hooks:** `benchmarks/compare_chip_download.py:64`, `benchmarks/compare_chip_download.py:144`, `benchmarks/compare_chip_download.py:153`, `aef_loader/reader.py:244`, `aef_loader/reader.py:362`.

5. **P1 — Preserve a virtual-reference branch before converting everything to Dask.** Immediately calling `xr.open_zarr()` converts the parsed manifest into the execution-facing representation. Publishing reusable virtual collections should branch at the `ManifestStore` stage; writing the returned Dask dataset with `.to_zarr()` materializes pixels. Offer reference export separately from materialized Zarr/COG/NumPy export. Validate CRS, time, nodata, and band metadata at each writer boundary.  
   **Hooks:** `aef_loader/reader.py:598`, `aef_loader/reader.py:618`, `aef_loader/cache.py:61`, `aef_loader/utils.py:448`.

6. **P1 — Add xarray/ODC entrypoints with explicit multi-CRS behavior.** A convenience `open_aef()` is inexpensive. An xarray backend should return one zone, require a target grid, or expose a DataTree entrypoint; silently representing multiple native CRSs as one Dataset is unacceptable. Retain `GeoBox` as the target-grid contract and add an existing-dataset/“like” convenience.  
   **Hooks:** `aef_loader/__init__.py:30`, `aef_loader/reader.py:485`, `aef_loader/reader.py:536`, `aef_loader/utils.py:309`, `pyproject.toml:53`.

7. **P2 — Add STAC as an adapter over discovery.** Convert index records into items with actual footprint, annual interval, asset URL, projection metadata, and raster band/nodata information; accept external items through the same asset-opening interface. This enables existing STAC/ODC workflows without making STAC a prerequisite for direct loading. Do not derive precise footprints solely from the tile’s WGS84 bounding box.  
   **Hooks:** `aef_loader/index.py:184`, `aef_loader/index.py:293`, `aef_loader/types.py:21`, `aef_loader/types.py:39`.

8. **P2 — Start a registry with two explicit built-ins; add entry points later.** `DataSource` identifies hosts, while TESSERA is selected through a separate function and hardcoded versioned URL. A registry should distinguish dataset/version from host and advertise available years, bands, decoder, and license. A CLI can then compose `datasets`, `query`, `chip`, and `cache` operations.  
   **Hooks:** `aef_loader/constants.py:6`, `aef_loader/tessera.py:29`, `aef_loader/tessera.py:77`, `aef_loader/__init__.py:30`, `pyproject.toml:48`.

## 3. Speed / cloud-native

**Estimates below are engineering estimates, not new benchmark measurements.** Effort assumes one engineer familiar with the stack. The repository’s measured source tiles are **8192×8192**, with examples of **1×1024×1024** storage chunks; they do not establish performance for ~2000×2000 source tiles. A 2048²×64 tile is 256 MiB raw or 1 GiB float32; many-tile memory planning remains important.  
**Evidence:** `docs/PERFORMANCE.md:88`, `tests/test_optimizations.py:78`, `tests/test_optimizations.py:80`, `aef_loader/reader.py:191`.

1. **P0 — Fix measurement before attributing speedups.** The benchmark records open/read/wall time, but no HTTP count, transferred bytes, decode time, scheduler time, or peak RSS. Its “64× more requests” claim follows from task grouping, not measured requests. Array `nbytes` is output size, not compressed network traffic. Add those measurements, plus multi-tile/multi-zone and repeated-chip cases.  
   The raw results support approximately **1.6×** and **2.3×** improvements for warm `all-bands` versus warm `native` at 2048 and 4096 pixels, respectively, but the all-band runs were a separate batch. Those are observations, not controlled evidence for a particular I/O mechanism.  
   **Win:** prevents incorrect optimization decisions. **Effort:** 1–3 days.  
   **Locations:** `benchmarks/compare_chip_download.py:11`, `benchmarks/compare_chip_download.py:79`, `benchmarks/compare_chip_download.py:159`, `docs/chip-benchmarks.md:96`, `docs/chip-benchmarks.md:106`, `benchmarks/chip_results_by_size_2026-09-30.json:2031`.

2. **P1 — Tune band grouping while retaining spatial storage alignment.** The existing `native`, `balanced`, and `all-bands` profiles are the right kind of controls. For full-vector reads, larger band groups reduce scheduling overhead and expose more I/O parallelism inside each task. They do **not** rechunk the TIFF or necessarily reduce GET count. For few-band reads, grouping unused bands can increase work.  
   For hypothetical 2048² tiles with `(1,1024,1024)` storage chunks, the profiles produce **256 / 16 / 4** source Dask chunks per tile. An all-band spatial block expands to **256 MiB** in float32 before warp temporaries.  
   **Win:** observed 1.6–2.3× in the cited chip cases; potentially negligible for bandwidth-bound reads. **Effort:** ½–2 days for a workload-aware policy and benchmarks.  
   **Locations:** `aef_loader/reader.py:211`, `aef_loader/reader.py:234`, `docs/PERFORMANCE.md:22`, `docs/chip-benchmarks.md:96`.

3. **P1 — Separate metadata concurrency from pixel-read concurrency.** `max_concurrency` limits tile opening; it does not govern subsequent Dask/Zarr chunk reads. Budget Dask threads, Zarr asynchronous operations, decode threads, and bytes in flight together. Reuse the existing per-bucket stores; expose timeout/retry/client configuration instead of constructing a new client per request. Avoid materializing all pending tile coroutines for very large queries.  
   Zarr’s guidance explicitly discusses interaction with Dask concurrency. [Zarr performance guidance](https://zarr.readthedocs.io/en/latest/user-guide/performance/).  
   **Win:** potentially 1.2–2× for a misconfigured deployment, or simply fewer throttles/OOMs; zero guaranteed gain. **Effort:** 2–4 days.  
   **Locations:** `aef_loader/reader.py:399`, `aef_loader/reader.py:496`, `aef_loader/reader.py:570`, `aef_loader/reader.py:618`, `aef_loader/reader.py:652`.

4. **P1 — Investigate batching byte ranges at the manifest-store boundary.** This package passes individual manifests to Zarr; it does not implement range coalescing. The inspected cached VirtualiZarr 2.7.3 implementation fetches individual ranges and leaves `get_partial_values` unimplemented. A useful upstream optimization is batching ranges by object, merging adjacent ranges subject to a maximum gap/read-amplification budget, then splitting returned buffers.  
   `object_store.get_ranges` can coalesce adjacent ranges, but **S3 does not support multiple disjoint ranges in one GET**. Coalescing and HTTP multipart-range requests are different mechanisms. [object_store vectored reads](https://docs.rs/object_store/latest/object_store/), [S3 GetObject](https://docs.aws.amazon.com/AmazonS3/latest/API/API_GetObject.html).  
   **Win:** potentially several-fold fewer requests where bytes are adjacent; planning estimate **0–30% end-to-end**, highly layout-dependent. Fetching intervening bands can make it worse. **Effort:** 3–10 days, likely upstream work.  
   **Hooks:** `aef_loader/cache.py:80`, `aef_loader/reader.py:598`, `aef_loader/reader.py:618`.

5. **P1 — Optimize index acquisition before replacing the in-memory STRtree.** The current path downloads the entire index and constructs all geometry objects. Its warm query already uses a spatial index and avoids copying the full frame. Add projected-column reading and an Arrow/SQL discovery backend; use scalar bbox overlap for candidate filtering and exact geometry only when required. Preserve exact-intersection and `limit` semantics, or expose a clearly named coarse mode.  
   DuckDB/Overture-style remote SQL is useful **only when the file layout supports pruning**. Inspect row-group statistics and spatial ordering first; installing DuckDB cannot create a missing index. Column projection can still help when row pruning cannot. The repository’s separate review reports a single-row-group index, but my cloud footer probe was blocked, so I did not independently verify that report. [DuckDB Parquet pushdown](https://duckdb.org/docs/data/parquet), [Overture queries](https://docs.overturemaps.org/getting-data/duckdb/).  
   **Win:** potentially large cold-start byte/RAM reduction; do not promise a warm-query improvement over the recorded 3–5 ms synthetic result. **Effort:** 2–5 days; publishing a spatially partitioned index requires upstream cooperation.  
   **Locations:** `aef_loader/index.py:154`, `aef_loader/index.py:184`, `aef_loader/index.py:266`, `aef_loader/index.py:276`, `docs/PERFORMANCE.md:49`, `docs/REVIEW-2026-09-30.md:245`.

6. **P1 — Reuse decoded blocks for repeated chips.** The LRU stores manifests, not pixels. Overlapping patch requests can therefore repeat the same transfer and decode. Batch chips by `(object version, time, storage block)`, fetch once, and extract multiple patches locally. Make any decoded-block cache byte-bounded.  
   **Win:** for sixteen aligned 256² patches occupying one 1024² block, theoretical source-read work falls by up to **16×** versus independent uncached reads. Actual gain depends on overlap and existing lower-level caches. **Effort:** 3–7 days.  
   **Locations:** `aef_loader/reader.py:346`, `aef_loader/reader.py:362`, `aef_loader/cache.py:12`, `benchmarks/compare_chip_download.py:120`.

7. **P1 — Keep the manifest cache, but distinguish local cache optimization from reusable collection publication.** Local JSON already removes header parsing on repeat opens. Dictionary-encoding repeated paths and binary offset/length arrays could reduce cache footprint; timing improvements will be modest when a complete warm open is already tens of milliseconds. Prioritize validation and concurrency safety before another serializer.  
   **Win:** recorded warm-open savings range from roughly **1–2 seconds per chip process** to much larger ratios in the small demo; not a corresponding pixel-read speedup. Binary serialization is plausibly a several-fold size reduction, unmeasured here. **Effort:** 1–3 days.  
   **Locations:** `aef_loader/cache.py:80`, `aef_loader/cache.py:137`, `aef_loader/cache.py:162`, `docs/PERFORMANCE.md:85`, `docs/chip-benchmarks.md:108`.

8. **P1 for shared services; P2 for individual users — Publish virtual collections using Icechunk or Kerchunk Parquet.** The custom JSON is a private per-tile cache, not an interoperable collection format. VirtualiZarr supports Kerchunk JSON/Parquet and Icechunk; Parquet avoids loading a giant JSON reference document, while Icechunk adds transactional snapshots and source-change checks. Preserve native grids as separate groups.  
   Icechunk is attractive for centrally maintained zone/year catalogs, but it cannot shrink existing TIFF compression blocks. Its Rust implementation also does not imply a major speedup over this already-obstore-backed reader. [VirtualiZarr format comparison](https://virtualizarr.readthedocs.io/en/stable/faq.html), [reference serialization](https://virtualizarr.readthedocs.io/en/stable/how_to/usage.html).  
   **Win:** removes repeated per-tile parsing/assembly for consumers; potentially substantial cold-open savings, **no inherent pixel-byte reduction**. **Effort:** 1–3 weeks including publishing, versioning, and compatibility checks.  
   **Hooks:** `aef_loader/cache.py:90`, `aef_loader/reader.py:598`, `aef_loader/reader.py:681`.

9. **P2 — Zarr v3 sharding is an output-layout decision.** AEF is already exposed through Zarr v3 metadata; TESSERA explicitly opens v3. A version upgrade alone will not improve COG reads. Repacking a frequently sampled subset into smaller spatial chunks inside larger shards could reduce read amplification without creating millions of tiny objects. Choose band/time layout from actual sampling patterns. [Zarr sharding guidance](https://zarr.readthedocs.io/en/latest/user-guide/performance/).  
   **Win:** changing independent spatial units from 1024² to 256² gives up to **16× less decoded data** for an aligned small patch; fewer wall-clock seconds are not guaranteed. No such gain without rewriting pixels. **Effort:** 1–2 weeks plus conversion/storage cost.  
   **Locations:** `aef_loader/reader.py:621`, `aef_loader/tessera.py:139`, `aef_loader/reader.py:191`.

10. **P2 — Track GeoZarr conventions; do not claim a finalized OGC GeoZarr standard.** The current project describes a future specification assembled from mature conventions, with projection, spatial, and multiscale conventions developed separately. I did not find evidence establishing final OGC adoption. Apply explicit, versioned metadata conventions to future exports and validate them independently of chunk layout. [Current GeoZarr project status](https://github.com/zarr-developers/geozarr-spec).  
    **Win:** interoperability, approximately **0% direct I/O acceleration**. **Effort:** 2–5 days for a defined export profile; ongoing compatibility maintenance.  
    **Hooks:** `aef_loader/tessera.py:119`, `aef_loader/tessera.py:166`, `aef_loader/reader.py:630`, `aef_loader/utils.py:446`.

11. **P2 — Keep obstore; no aiohttp/fsspec migration is justified.** AEF index download, AEF pixel access, and TESSERA already use object-store-backed paths. A custom HTTP stack would duplicate authentication, retries, pooling, and range handling. Fsspec remains useful for compatibility, but obstore itself recommends using its direct APIs when possible. [obstore guidance](https://developmentseed.org/obstore/latest/api/fsspec/).  
    **Win:** approximately **zero expected gain from a transport rewrite**; focus on request shape and client reuse. **Effort:** no migration needed.  
    **Locations:** `aef_loader/index.py:154`, `aef_loader/reader.py:384`, `aef_loader/reader.py:393`, `aef_loader/tessera.py:48`.

12. **P2 — S3 Express, CRT, and Mountpoint are conditional infrastructure choices.** Express One Zone requires a directory bucket and colocated compute; it cannot accelerate reads from the existing Source Cooperative bucket without copying data. CRT may help sustained high-throughput transfers, but there is no measured advantage here over obstore. Mountpoint adds a filesystem interface that this reader does not accept. First benchmark compute near the existing `us-west-2` source. [S3 Express requirements](https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-express-optimizing-performance-design-patterns.html), [CRT](https://aws.amazon.com/blogs/storage/improving-amazon-s3-throughput-for-the-aws-cli-and-boto3-with-the-aws-common-runtime/), [Mountpoint](https://github.com/awslabs/mountpoint-s3).  
    **Win:** no direct Express/Mountpoint gain for the current URLs; colocation can substantially reduce latency but is unmeasured. **Effort:** hours for a regional benchmark; days–weeks plus recurring costs for staging/transport changes.  
    **Locations:** `aef_loader/constants.py:19`, `aef_loader/reader.py:62`, `aef_loader/reader.py:391`.

13. **P2 — Keep Dask for mosaics/reprojection; evaluate alternatives narrowly.** The current path deliberately requires Dask-backed arrays to avoid eager materialization during assembly. A direct asynchronous window API can avoid graph construction for simple chips if it selects pixels before assembly. Cubed offers bounded-memory execution, but is not a drop-in replacement for this package’s `map_blocks` and ODC warp pipeline. [Cubed](https://github.com/cubed-dev/cubed).  
    **Win:** direct-chip execution may remove scheduler overhead, likely modest when transfer dominates; Cubed’s primary potential gain is predictable memory rather than faster reads. **Effort:** 3–7 days for a chip prototype; weeks for another supported execution backend.  
    **Locations:** `aef_loader/reader.py:608`, `aef_loader/reader.py:638`, `aef_loader/utils.py:117`, `aef_loader/utils.py:397`.

14. **P2 — GPU decode is low priority until profiling demonstrates a CPU bottleneck.** The decoder explicitly converts through NumPy, and the warp path is ODC-based. Enabling GPU-backed Zarr buffers does not automatically move TIFF decompression or this decoder to the GPU; current Zarr documentation still describes host-side encoding/decoding. A bespoke GPU pipeline must support the actual TIFF codecs/predictors and keep downstream processing on-device. [Zarr GPU documentation](https://zarr.readthedocs.io/en/stable/user-guide/gpu/).  
    **Win:** likely negligible for latency/network-bound chips; if decode accounts for 20% of runtime, even eliminating it entirely caps acceleration at **1.25×**. **Effort:** weeks, not a configuration toggle.  
    **Locations:** `aef_loader/utils.py:67`, `aef_loader/utils.py:120`, `aef_loader/utils.py:397`, `benchmarks/compare_chip_download.py:160`.

## 4. Correctness / bug risks

1. **P0 — Quantization converts missing data into valid embeddings. Verified.** `quantize_aef()` casts NaN directly to `int8`. In the inspected environment, `quantize_aef([NaN, 0])` returns `[0, 0]` with a warning. Consequently, the advertised dequantize/quantize round trip loses nodata. Mask nonfinite values before conversion, map missing values explicitly to `-128`, and define behavior for infinities.  
   **Locations:** `aef_loader/utils.py:151`, `aef_loader/utils.py:167`, `aef_loader/utils.py:173`.

2. **P0 — Raw TESSERA merging can combine codes and scales from different zones. Verified with synthetic data and the actual warp/merge path.** TESSERA validity is represented by `scales`, but the merger chooses each variable independently. Integer embeddings without a nodata attribute never satisfy `.isnull()`. A first zone containing code `0`, scale `NaN`, followed by a valid zone containing code `10`, scale `2`, produced **code `0`, scale `2`, value `0` instead of `20`**.  
   Use one pixel-validity/source-selection mask for embeddings, scales, and associated quality variables, or decode and mask before merging. Do not assign AEF’s `-128` convention to TESSERA without verifying its encoding.  
   **Locations:** `aef_loader/tessera.py:8`, `aef_loader/tessera.py:157`, `aef_loader/tessera.py:160`, `aef_loader/utils.py:419`, `aef_loader/utils.py:429`, `aef_loader/utils.py:438`.

3. **P1 — Multi-zone reprojection fails when temporal coverage differs. Verified.** A common target GeoBox guarantees spatial alignment, not identical time/band coordinates. Different numbers of years trigger the explicit shape error; equal lengths with different years trigger xarray’s exact-alignment error. Align nonspatial dimensions explicitly with dtype-safe missing values before merging. Variables appearing only in later zones are also silently omitted.  
   **Locations:** `aef_loader/utils.py:415`, `aef_loader/utils.py:419`, `aef_loader/utils.py:422`, `aef_loader/utils.py:438`.

4. **P1 — Pixel-scale/tiepoint georeferencing is wrong for standard north-up TIFFs.** The helper uses `+sy` and ignores the tiepoint’s raster indices. Under standard GeoTIFF semantics the transform is `Affine(sx, 0, X-I*sx, 0, -sy, Y+J*sy)`. Missing tiepoints are silently replaced with zeros. Correct this and test against an independent GeoTIFF reader; existing files using `ModelTransformation` take another branch. [OGC GeoTIFF requirements](https://docs.ogc.org/is/19-008r4/19-008r4.html?swpmtx=3cf3f7ecd79a56c635037aa32ecfeb07&swpmtxnonce=1e04dc1fc1).  
   **Locations:** `aef_loader/reader.py:95`, `aef_loader/reader.py:99`, `aef_loader/reader.py:147`, `aef_loader/reader.py:149`.

5. **P1 — CRS grouping can silently mix incompatible grids.** All missing UTM-zone values become `"unknown"`; the completed group is assigned the first tile’s CRS without validation. Two tiles with numerically overlapping coordinates in different CRSs can therefore be combined incorrectly. Group by a canonical CRS/grid identity and validate any zone label. Separately, a bare index CRS such as `"32633"` silently becomes EPSG:4326.  
   **Locations:** `aef_loader/reader.py:488`, `aef_loader/reader.py:516`, `aef_loader/reader.py:681`, `aef_loader/index.py:307`.

6. **P1 — Manifest freshness is assumed, not enforced.** Cache identity contains URL, IFD, and the package’s serialization version, but no object version, ETag, size, or modification time. Replacing a COG at the same path leaves stale byte offsets/codecs usable indefinitely. This is a conditional silent-corruption risk, not evidence that the present upstream files changed. Offer immutable-version references or a validation/refresh policy.  
   **Locations:** `aef_loader/cache.py:45`, `aef_loader/cache.py:52`, `aef_loader/cache.py:138`, `aef_loader/reader.py:586`.

7. **P1 — Both disk-cache paths have concurrency/recovery problems.** Manifest writers use the same `.json.tmp` name; concurrent processes or duplicate tile opens can truncate, rename, or remove one another’s temporary file. The index is written directly to its final filename, and any existing file is accepted on subsequent downloads. A crash can leave a permanently reused truncated index. Use unique temporary files plus replacement, cache validation, and per-key in-flight deduplication.  
   **Locations:** `aef_loader/cache.py:161`, `aef_loader/cache.py:163`, `aef_loader/index.py:121`, `aef_loader/index.py:157`, `aef_loader/reader.py:587`.

8. **P1 — Antimeridian bboxes are mishandled.** For `(west=179, east=-179)`, AEF constructs a box spanning the intervening longitudes, while TESSERA’s zone range is empty. Projected bounds can also transform into an antimeridian-crossing envelope. Split these into two geographic queries/crops or reject them explicitly; validate finite coordinates and latitude bounds.  
   **Locations:** `aef_loader/index.py:217`, `aef_loader/index.py:262`, `aef_loader/index.py:263`, `aef_loader/tessera.py:52`, `aef_loader/tessera.py:65`.

9. **P1 — Exact-grid cropping adds unintended border pixels. Verified.** Expanding bounds by half a pixel and applying inclusive label slices includes cells that merely touch an AOI edge. On a 10 m grid, `(20,40,50,80)` returned **5×6** pixels rather than **3×4**. The benchmark compensates by shrinking its AOI by one metre. Use inverse-affine integer windows with an explicit edge convention; test exact dimensions. Skip empty cropped tiles before mosaicking.  
   **Locations:** `aef_loader/reader.py:266`, `aef_loader/reader.py:275`, `aef_loader/reader.py:635`, `tests/test_optimizations.py:111`, `benchmarks/compare_chip_download.py:144`.

10. **P1 — TESSERA consumes year generators once per zone. Verified.** `years` accepts `Iterable[int]`, but converts it to a list inside the zone loop. `iter([2024])` selects 2024 in the first zone and zero years in the second. Normalize selection once before the loop and report unavailable years.  
    **Locations:** `aef_loader/tessera.py:71`, `aef_loader/tessera.py:113`, `aef_loader/tessera.py:148`, `aef_loader/tessera.py:154`.

11. **P1 — The resampling guard confuses dtype with encoding. Verified.** A dequantized TESSERA dataset with integer observation counts is rejected as raw AEF. Conversely, `int8_to_float32()` preserves quantized codes but changes dtype, thereby bypassing the guard without performing nonlinear dequantization. Track encoding state on the embedding variable; choose resampling per variable, keeping categorical/count-quality semantics explicit.  
    **Locations:** `aef_loader/utils.py:215`, `aef_loader/utils.py:303`, `aef_loader/utils.py:374`, `aef_loader/tessera.py:157`, `aef_loader/tessera.py:164`.

12. **P1 — Unsupported decoder inputs can silently produce plausible wrong values. Verified for double decoding.** The wider-type fallback casts to `int16` before lookup. Passing already-decoded `64 → approximately 0.252` through `dequantize_aef()` again yields zero; out-of-range negative integers can index from the LUT’s end. Reject floating/nonintegral and out-of-range inputs rather than silently truncating them. Dataset-wide decoding should target declared embedding variables.  
    **Locations:** `aef_loader/utils.py:75`, `aef_loader/utils.py:112`, `aef_loader/utils.py:135`.

13. **P1 — Important contracts lack regression coverage.** The current tests validate the LUT, one temporal join, chunk hints, a subset crop, an index query, LRU ordering, and mocked zone concurrency. They do not exercise the failure cases above. The live TESSERA smoke test never computes pixels. Add focused cases for nodata round trips, cross-zone code/scale consistency, unequal years, TIFF transforms, exact windows, and manifest round-trip/replacement behavior.  
    **Locations:** `tests/test_optimizations.py:35`, `tests/test_optimizations.py:59`, `tests/test_optimizations.py:99`, `tests/test_optimizations.py:169`, `tests/test_optimizations.py:183`, `tests/test_tessera.py:16`.

14. **P2 — `snap=False` does not disable snapping. Verified with odc-geo 0.5.3.** Both branches invoke `GeoBox.from_bbox()` with its snapping defaults. `snap=False` produced the same snapped transform as `snap=True` for off-grid bounds. Pass the appropriate `tight`/anchor options and correct the documentation. [Current ODC behavior](https://odc-geo.readthedocs.io/en/latest/_api/odc.geo.geobox.GeoBox.from_bbox.html).  
    **Locations:** `aef_loader/utils.py:460`, `aef_loader/utils.py:474`, `aef_loader/utils.py:492`.

15. **P2 — Index refresh and limit semantics are inconsistent.** `download(force=True)` changes the disk file but leaves an already-loaded `_gdf` active; subsequent queries use stale data until `load()` is called explicitly. `limit=0` means unlimited, and negative limits inherit pandas’ “all but the last N” behavior. Invalidate memory on refresh; validate limits and reversed year ranges.  
    **Locations:** `aef_loader/index.py:160`, `aef_loader/index.py:193`, `aef_loader/index.py:252`, `aef_loader/index.py:276`.

16. **P2 — Failed opens do not have structured cancellation.** Nested `asyncio.gather()` propagates the first exception while siblings can continue; `to_thread` work can also outlive coroutine cancellation. Context exit clears caches while those operations may still be finishing. Use structured task lifetime management and drain outstanding work before teardown. This is a lifecycle/race risk, not a demonstrated permanent socket leak.  
    **Locations:** `aef_loader/reader.py:373`, `aef_loader/reader.py:527`, `aef_loader/reader.py:598`, `aef_loader/reader.py:652`.

17. **P2 — Private dependency APIs make compatibility fragile.** Serialization accesses `_group`, `_paths`, `_offsets`, and `_lengths`, and reconstructs Zarr metadata directly. Declared NumPy/xarray minima also understate APIs used directly; present transitive dependencies may supply stronger constraints. Test an explicit supported dependency matrix and a lower-bound environment rather than relying on unconstrained latest installs.  
    **Locations:** `aef_loader/cache.py:74`, `aef_loader/cache.py:80`, `aef_loader/cache.py:107`, `aef_loader/cache.py:112`, `aef_loader/reader.py:28`, `pyproject.toml:20`, `pyproject.toml:23`, `.github/workflows/tests.yml:10`.

18. **P2 — Benchmark integrity and cleanup need tightening.** `--methods` allows subsets, but integrity comparison unconditionally expects upstream and fork reference files. All-band timed outputs are not included in that comparison. Earth Engine comparison checks only jointly finite values, so differing nodata masks can pass; a zero-overlap comparison reports zero error. The driver’s `mkdtemp()` workspace is never cleaned up.  
    **Locations:** `benchmarks/compare_chip_download.py:231`, `benchmarks/compare_chip_download.py:241`, `benchmarks/compare_chip_download.py:248`, `benchmarks/compare_chip_download.py:275`, `benchmarks/compare_chip_download.py:295`, `benchmarks/compare_chip_download.py:328`.

Validation: **12 offline tests passed; 1 live test excluded.** Tests and additional reproductions ran without repository edits, using existing cached dependencies and disabled bytecode/cache writes. Cloud pixel reads were not revalidated; the direct cloud footer probe was blocked.
