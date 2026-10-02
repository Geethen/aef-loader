# Range coalescing at the AEF COG boundary

Research date: 2026-09-30. This is an implementation-path and layout check, not
a benchmark. Installed by `uv sync --extra dev` and located with the requested
command: VirtualiZarr 2.7.3, obstore 0.11.1, Zarr 3.4.0, and obspec-utils 0.9.0
are under `.venv/Lib/site-packages/`.

## 1. Current chunk-read path

For an ordinary (unsharded) ManifestStore-backed chunk, the precise path is:

```
Zarr AsyncArray selection
  -> codec pipeline, one StorePath per selected Zarr chunk
  -> StorePath.get() -> ManifestStore.get()
  -> ObjectStoreRegistry.resolve(source URL)
  -> obstore S3Store.get_range_async(source key, start, end)
  -> Rust binding obs.get_range_async -> S3 ranged GET
```

Evidence:

- Zarr turns every indexed chunk into `store_path / encode_chunk_key(...)` and
  passes that list to its codec pipeline at
  `.venv/Lib/site-packages/zarr/core/array.py:5621-5648`.
- The normal fallback calls `byte_getter.get(prototype)` once per chunk; it uses
  concurrent tasks but does not combine the requests
  (`zarr/core/codec_pipeline.py:321-360`). `StorePath.get` delegates to
  `store.get(path, ...)` (`zarr/storage/_common.py:141-163`).
- `ManifestStore.get` is `async`, looks up one manifest entry, resolves its
  object store, translates a possible Zarr byte request, then awaits exactly
  one `store.get_range_async(...)` (`virtualizarr/manifests/store.py:143-220`).
  Its `get_partial_values` raises `NotImplementedError`
  (`virtualizarr/manifests/store.py:222-229`).
- `S3Store.get_range_async` delegates to `obs.get_range_async`
  (`obstore/store.py:222-240`; `obstore/store.py:7-8` imports that binding).

Thus the network call is asynchronous, and Zarr may have several independent
chunk reads in flight (bounded by its `async.concurrency` configuration), but
there is no cross-chunk range batch on this normal path. Dask can further split
native chunks into separate tasks, so task concurrency is not a request batch.

## 2. What Zarr 3.4.0 can batch

`Store.get_partial_values(prototype, key_ranges)` is part of the Store ABI
(`zarr/abc/store.py:221-240), and the built-in obstore store implements it
(`zarr/storage/_obstore.py:142-148). However, in this installed Zarr tree its
only non-test occurrences are definitions/delegations: ordinary array reads do
not call it. Consequently implementing it in a ManifestStore wrapper alone
does not change normal AEF reads.

There is a different batched API: `Store.get_ranges(key, byte_ranges, ...)`.
Its default implementation coalesces ranges for *one Zarr storage key*, with
10 concurrent merged reads, a 1 MiB maximum gap, and a 16 MiB maximum merged
span (`zarr/abc/store.py:414-472`; merge rules at
`zarr/core/_coalesce.py:102-133`). Zarr invokes it for a **partial read of a
sharded Zarr v3 chunk**: after reading that shard's index, `ShardingCodec`
requests several inner-byte ranges from the same shard key
(`zarr/codecs/sharding.py:1664-1725). It is not a general multiple-Zarr-key
API. AEF's VirtualTIFF manifest exposes ordinary Zarr chunks, not Zarr shards,
so this route is not reached.

## 3. obstore vectored reads

Yes. `ObjectStore` exposes `get_ranges` and `get_ranges_async`, accepting many
starts/ends for one object and a `coalesce` parameter, defaulting to 1 MiB
(`obstore/store.py:242-284`). The installed binding's API contract says it
combines ranges less than `coalesce` bytes apart into one underlying request
(default 1 MiB), runs up to 10 fetches in parallel, and returns one `Bytes`
result per requested range (`obstore/_get.pyi:364-405`). Thus adjacent ranges
(zero gap) coalesce; `coalesce=0` disables it. The actual Rust implementation
is compiled, not present as Rust source in this wheel; this behavior is
verified from the installed Python binding contract, not by source inspection
of the Rust crate.

Coalescing means fetch the intervening bytes and slice locally. It is not an
HTTP multipart-range request: [S3 GetObject explicitly says it cannot retrieve
multiple ranges in one GET](https://docs.aws.amazon.com/AmazonS3/latest/API/API_GetObject.html).

## 4. Measured AEF COG layout

The supplied object path 404ed. I listed
`tge-labs/aef/v1/annual/2024/31N/` with the public anonymous S3 store and then
parsed the TIFF header/IFD only (no pixel array was computed) for:

```
s3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/2024/31N/x05g5rs56l3cq3pex-0000000000-0000000000.tiff
```

The listed object was 2,121,286,026 bytes (ETag
`"4ea8121cd2f7978d53992f647ff432ab-16"`). `VirtualTIFF(ifd=0)` reported shape
`(64, 8192, 8192)`, chunk shape `(1, 1024, 1024)`, and manifest grid
`(64, 8, 8)`. These are offsets and compressed lengths in bytes for the 2x2
neighbourhood `(band, block-y, block-x)`:

| Band | block | offset | length |
| ---: | :---: | ---: | ---: |
| 0 | (0, 0) | 592,559,958 | 412,065 |
| 0 | (0, 1) | 592,972,031 | 446,849 |
| 0 | (1, 0) | 596,576,611 | 532,107 |
| 0 | (1, 1) | 597,108,726 | 573,123 |
| 1 | (0, 0) | 626,777,248 | 484,609 |
| 1 | (0, 1) | 627,261,865 | 456,991 |
| 1 | (1, 0) | 630,399,775 | 385,344 |
| 1 | (1, 1) | 630,785,127 | 449,965 |

Horizontally neighbouring blocks have an 8-byte gap in both rows. Vertically
neighbouring blocks in this 2x2 have gaps of 3,604,588 bytes (band 0) and
3,137,918 bytes (band 1), because the remaining six blocks of the preceding
row lie between them. Band 0 `(7,7)` ends at 626,777,240 and band 1 `(0,0)`
starts at 626,777,248: planes themselves are consecutive with an 8-byte gap.
Therefore it is **band-major / planar, with row-major tiles within a band**,
not tile-major interleaving. Corresponding block `(0,0)` in bands 0 and 1 is
34,217,290 bytes apart.

This measurement uses the manifest entries produced by VirtualTIFF; their
per-chunk path/offset/length are precisely what ManifestStore consumes
(`virtualizarr/manifests/store.py:187-219). It is one real 2024 tile, so do not
assume exact compressed sizes for every COG without sampling them.

## 5. Realistic benefit and implementation point

Assume an all-64-band request and native `(1,1024,1024)` chunks, as this
reader selects (`aef_loader/reader.py:637-653`). Request-count figures below
are best-case planning figures; native Dask tasks may not present all requests
to a coalescer at the same time.

| selection | current chunk GETs | coalescing opportunity | realistic result |
| --- | ---: | --- | --- |
| 256 px chip inside one block | 64 | One block per band; matching blocks across bands are ~34 MB apart | Essentially none (0% request reduction) |
| 1024 px chip straddling 2x2 blocks | 256 | The two horizontal blocks per band are 8 bytes apart; rows are >3 MiB apart | At most 256 -> 128 GETs with a 1 MiB gap, only if requests are batched together; bytes barely fall |
| full 8192 px tile | 4,096 | 64 contiguous blocks per band; all band planes are sequential too | Cap ranges deliberately: roughly 64 ~34 MB plane GETs is plausible, but payload remains ~2.12 GB; uncapped merging could make a disastrous ~2.12 GB GET |

For the full tile, 64 GETs is an illustrative safe band-plane cap, not a
guarantee of speedup. Zarr's own sharding coalescer would cap at 16 MiB, while
obstore's `get_ranges` API has a gap control but no maximum merged-span control
in its exposed signature. A local implementation must add that cap.

No obstore change is needed. The cleanest durable home is (1) Zarr gaining a
normal-read multi-key batch call, then (2) VirtualiZarr grouping manifest keys
by physical object and using obstore vectored/ranged reads. VirtualiZarr merely
implementing `get_partial_values` is insufficient until Zarr calls it. A repo-
local wrapper/subclass is feasible today, but it must micro-batch concurrent
`get` calls and reuse VirtualiZarr private manifest details; it is necessarily
sensitive to scheduler timing and upstream internals.

Illustrative local design (29 lines; preserve the exact byte-range transform
from `ManifestStore.get`, including inline/missing chunks):

```python
class CoalescingManifestStore(ManifestStore):
    def __init__(self, *args, gap=1 << 20, cap=16 << 20, **kw):
        super().__init__(*args, **kw); self.gap, self.cap = gap, cap
        self.pending, self.flush_scheduled = [], False

    async def get(self, key, prototype, byte_range=None):
        ref = self._physical_reference(key, byte_range)  # copy VZ 2.7.3 logic
        if ref.inline_or_missing: return ref.as_buffer(prototype)
        loop = asyncio.get_running_loop(); future = loop.create_future()
        self.pending.append((ref, prototype, future))
        if not self.flush_scheduled:
            self.flush_scheduled = True; loop.call_soon(self._start_flush)
        return await future

    def _start_flush(self):
        batch, self.pending = self.pending, []
        self.flush_scheduled = False
        asyncio.create_task(self._flush(batch))

    async def _flush(self, batch):
        for (store, path), requests in group_by_source(batch).items():
            for group in merge_by_offset(requests, self.gap, self.cap):
                lo, hi = group[0].start, max(r.end for r in group)
                blob = await store.get_range_async(path, start=lo, end=hi)
                for ref, prototype, future in group:
                    future.set_result(prototype.buffer.from_bytes(
                        blob[ref.start - lo : ref.end - lo]))
```

It needs cancellation/error propagation, a lock for cross-thread Dask use,
memory/back-pressure limits, and tests against the base store before it is
production quality. A better local alternative for a known chip workload is
to plan `(source object, band, block-row)` reads before handing work to Dask;
that provides deterministic batches rather than timing-dependent ones.

## 6. Recommendation

**Measure first; do not implement a general wrapper yet.** For the dominant
small-chip case, this layout gives no range-coalescing win across bands. For a
2x2 boundary case it could halve GET count but not payload, and only when the
Dask/Zarr execution shape exposes a batch. First add read tracing (physical
range, bytes, concurrency, and task grouping) and benchmark 256 px, straddled
1024 px, and full-tile reads with warm manifests.

Estimate: 0.5-1 day for tracing plus a reproducible benchmark; 2-4 days for a
bounded local experimental wrapper and regression tests; upstream-quality
Zarr/VirtualiZarr multi-key batching is a separate, multi-week coordinated
change. Prioritize decoded-block reuse for repeated 256 px chips instead: it
avoids both transfer and decode, whereas coalescing only reduces request
overhead.

## Addendum (2026-09-30, after review)

- The read tracing recommended above now exists: `benchmarks/compare_chip_download.py`
  records `obstore_requests` / `obstore_bytes` per run. Its first live run agrees with
  this note: 64 range requests for a 256 px chip (one block per band) and 256 for a
  1024 px chip over four blocks, with blocks averaging ~435 KB compressed.
- The upstream Zarr/VirtualiZarr batching mentioned in section 5 is recorded for
  understanding only. This project does not file issues or PRs against other
  repositories; any work here stays in this repo.
- Next step, per section 6: decoded-block reuse for repeated chips (D4 / Astra section 3.6)
  ranks above a coalescing wrapper.
