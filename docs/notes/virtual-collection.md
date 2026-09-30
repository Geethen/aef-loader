# Design Notes: AEF Virtual Collection

## Goal
Publish ONE pre-built virtual dataset per (UTM zone, year) covering all AlphaEarth Foundations tiles. This eliminates per-tile header parsing over the network, allowing readers to jump directly to pixel decoding.

## Size Estimate
- **Tiles in Index:** ~302,000
- **Chunk References per Tile:** 8x8 blocks $\times$ 64 bands = 4096 refs
- **Total References:** ~1.23 billion globally.
- **Per Zone/Year:** ~5,000 tiles per zone/year $\rightarrow$ ~20 million chunk references.

## VirtualiZarr Serialisation Options

We can use `virtualizarr.VirtualiZarrDatasetAccessor` to export the xarray dataset.

### 1. Kerchunk JSON
- **API:** `ds.virtualize.to_kerchunk(format="json")` (`virtualizarr/accessor.py:171`).
- **Pros:** Human-readable, ubiquitous ecosystem support (fsspec `ReferenceFileSystem`).
- **Cons:** ~20M refs per zone is huge as a JSON (likely 1-2 GB). Loading this single monolithic JSON into memory for a single zone takes time and high memory, defying the fast-start goal.

### 2. Kerchunk Parquet
- **API:** `ds.virtualize.to_kerchunk(format="parquet")` (`virtualizarr/accessor.py:171`).
- **Pros:** Columnar compression is much more efficient. Can partition into multiple files using `record_size`. Supports `categorical_threshold` for URLs.
- **Cons:** Still a static file format. Not all Zarr readers support Kerchunk Parquet natively without `fsspec` wrappers.

### 3. Icechunk
- **API:** `ds.virtualize.to_icechunk()` (`virtualizarr/accessor.py:72`).
- **Pros:** Native transactional virtual store. `virtualizarr` can write directly to an `IcechunkStore` and it handles append efficiently.
- **Cons:** Requires `icechunk` library.

## Layout: Zone/Year vs Global Store
- **Global:** 1.23B chunk refs in a single Kerchunk JSON/Parquet is unmanageable. Icechunk could handle a global store, but mixing CRSs (UTM zones) in a single root is anti-pattern in Xarray/Zarr.
- **Per Zone/Year:** Fits cleanly into Xarray's `DataTree` model. Each dataset acts as a cohesive STAC item or Zarr group, naturally mapping to our existing `VirtualTiffReader.open_tiles_by_zone` API.

## Versioning and Staleness
VirtualiZarr refs point to the byte ranges in Source Cooperative COGs (no pixel copying). If an upstream COG is updated in-place, the byte offsets become invalid. 
- Icechunk explicitly supports a `last_updated_at` checksum argument to protect against stale chunks (`virtualizarr/accessor.py:101`). We could tie this to the object's `last_modified` metadata from TASK 1. ETag could be used to detect changes upstream.

## Hosting & Cost
- **Storage:** These are just references, so the entire index (Parquet or Icechunk) will be small (< 100 GB). 
- **Hosting:** Host on an S3 bucket (e.g., Cloudflare R2 or AWS S3).
- **Cost:** Virtually free for storage. Requests cost is minimised because it's a few large reads per zone instead of one HEAD/GET per tile. Self-hosting avoids third-party API dependencies.

## API Sketch
```python
class VirtualCollectionReader:
    async def open_tiles_by_zone(self, tiles, ...):
        # Instead of parsing virtual_tiff over network, we load the Parquet/Icechunk index.
        # Tree hierarchy: /10N/embeddings(time, band, y, x)
        import xarray as xr
        
        # Kerchunk Parquet example:
        mapper = fsspec.get_mapper("reference://", fo="s3://my-aef-index/2024_10N.parquet")
        ds = xr.open_zarr(mapper, consolidated=False)
        
        # Icechunk example:
        store = icechunk.IcechunkStore.open("s3://my-aef-index/2024_10N")
        ds = xr.open_zarr(store, zarr_format=3)
        return ds.sel(time=...)
```

## Recommendation
**Use Kerchunk Parquet** partitioned by (UTM zone, year). It requires minimal infrastructure (no database, just static files on S3) and provides massive compression over JSON for our 20M refs per zone. Effort is medium (batch script to run `to_kerchunk` once).
