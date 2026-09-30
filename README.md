# aef-loader-plus

[VirtualiZarr](https://github.com/zarr-developers/VirtualiZarr) access to
[AlphaEarth Foundations](https://developers.google.com/earth-engine/guides/aef_on_gcs_readme) (AEF)
embeddings as an analysis-ready xarray/dask cube, with fast index queries against
Source Cooperative and Google Cloud Storage.

This is a fork of **[jakenotjay/aef-loader](https://github.com/jakenotjay/aef-loader)** by Jake Wilkins
(Apache-2.0). All credit for the original design and implementation goes to the upstream
project; see [NOTICE](NOTICE) and [docs/UPSTREAM_README.md](docs/UPSTREAM_README.md).

## What this fork adds

- `chunks=None` no longer materialises whole tiles, plus `chunks="native" | "balanced" | "all-bands"` options
- On-disk and in-memory manifest (COG header) caching
- Faster LUT-based `dequantize_aef`
- `AEFIndex.query(bbox_crs=...)` with densified bbox reprojection
- `aoi_geobox` for lattice-snapped, mosaic-safe grids
- `reproject_datatree` refuses lossy resampling on int8 data unless `allow_lossy_resampling=True`
- `combine_by_coords` keeps int8 and its `-128` nodata sentinel

Details and measurements: [docs/CHANGES.md](docs/CHANGES.md),
[docs/PERFORMANCE.md](docs/PERFORMANCE.md), [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Installation

Requires Python 3.12+.

```bash
pip install git+https://github.com/Geethen/aef_loader_plus.git
```

or with uv:

```bash
uv add git+https://github.com/Geethen/aef_loader_plus.git
```

For development:

```bash
git clone https://github.com/Geethen/aef_loader_plus.git
cd aef_loader_plus
uv sync --extra dev
uv run python -m pytest -m "not slow"  # offline tests
uv run python -m pytest -m slow        # live tests (hit Source Cooperative)
```

> **Note:** the import name is still `aef_loader`, so this package conflicts with the
> upstream `aef-loader` distribution. Install one or the other in an environment, not both.

## Quick start

```python
import asyncio
from aef_loader import AEFIndex, VirtualTiffReader, DataSource
from aef_loader.utils import reproject_datatree
from odc.geo.geobox import GeoBox

async def main():
    # Source Cooperative is free and needs no auth
    index = AEFIndex(source=DataSource.SOURCE_COOP)
    await index.download()
    index.load()

    bbox = (-122.5, 37.5, -122.0, 38.0)
    tiles = await index.query(bbox=bbox, years=(2020, 2023))

    async with VirtualTiffReader(manifest_cache_dir="aef-manifests") as reader:
        tree = await reader.open_tiles_by_zone(tiles, chunks="balanced", bbox=bbox, bbox_crs="EPSG:4326")

    for zone in tree.children:
        ds = tree[zone].ds
        print(f"{zone}: {ds.odc.crs}, {dict(ds.sizes)}")

    target = GeoBox.from_bbox(bbox=bbox, crs="EPSG:4326", resolution=0.0001)
    combined = reproject_datatree(tree, target)

asyncio.run(main())
```

## TESSERA embeddings

Besides AEF, `open_tessera` reads [TESSERA](https://geotessera.org) embeddings (128 bands, 10 m, yearly 2017–2025)
from the [Source Cooperative Zarr store](https://source.coop/tessera/tessera/zarr/v1.1-dclimate). It returns the same
DataTree-by-UTM-zone layout as the AEF reader, so `reproject_datatree` works on it.

```python
from aef_loader import open_tessera, reproject_datatree
from odc.geo.geobox import GeoBox

bbox = (5.6, 58.7, 5.63, 58.72)
tree = open_tessera(bbox, years=(2023, 2024), dequantize=True)   # lazy, nothing downloaded yet
target = GeoBox.from_bbox(bbox, crs="EPSG:4326", resolution=0.0002)
ds = reproject_datatree(tree, target, resampling="bilinear").compute()
```

- `dequantize=False` (default) returns int8 `embeddings` plus per-pixel `scales` (value = `embeddings * scales`);
  `dequantize=True` returns float32 with water and never-written pixels as NaN.
- `years` is a year, an inclusive `(start, end)` tuple, or a list; `zones=[31, 32]` forces UTM zones;
  `include_quality=True` adds the Sentinel observation-count variables.
- Bboxes may be in any CRS (`bbox_crs=`). Southern-hemisphere sites use the same `utmNN` groups (north-referenced, negative northings).
- TESSERA is produced by the University of Cambridge; check its [licence terms](https://geotessera.org) before redistributing.
## Hosts

| Host | Access | Notes |
|---|---|---|
| [Source Cooperative](https://source.coop/tge-labs/aef) | Free, AWS S3 | Full range 2017–2025. **Recommended** |
| [Google Cloud Storage](https://developers.google.com/earth-engine/guides/aef_on_gcs_readme) | Requester pays (needs `gcp_project`) | Maintained by the Earth Engine team |

## Repository layout

| Path | Contents |
|---|---|
| `aef_loader/` | The package |
| `tests/` | Offline pytest suite |
| `benchmarks/` | Benchmark scripts and raw results (`pip install -e ".[benchmark]"`) |
| `docs/` | Change log, performance notes, benchmarks, upstream README and diff |

## Acknowledgements and licence

- Fork of [jakenotjay/aef-loader](https://github.com/jakenotjay/aef-loader) by Jake Wilkins, forked at
  commit `feff13e`. Thanks also to Max Jones ([virtual-tiff](https://github.com/virtual-zarr/virtual-tiff))
  and [VirtualiZarr](https://github.com/zarr-developers/VirtualiZarr).
- Code is licensed under [Apache-2.0](LICENSE).
- Dataset attribution (CC-BY 4.0): "The AlphaEarth Foundations Satellite Embedding dataset is produced by Google and Google DeepMind."
