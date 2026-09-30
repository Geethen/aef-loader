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
pip install -e ".[dev]"
pytest
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
