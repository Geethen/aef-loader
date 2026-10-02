"""Offline synthetic-COG helpers for reader regression tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from obstore.store import LocalStore
from pyproj import Transformer
from rasterio.transform import Affine as RasterAffine

from aef_loader.reader import VirtualTiffReader
from aef_loader.types import AEFTileInfo

SIZE = 256
BANDS = 4
BLOCK = 64
EPSG = 32631
WEST, NORTH = 500_000.0, 6_500_000.0


def pattern(year: int, col_off: int = 0, row_off: int = 0) -> np.ndarray:
    """Deterministic (band, y, x) int8 data."""
    r = np.arange(SIZE)[None, :, None] + row_off
    c = np.arange(SIZE)[None, None, :] + col_off
    b = np.arange(BANDS)[:, None, None]
    return (((r * 7 + c * 3 + b * 11 + (year - 2020) * 5) % 100) - 50).astype("int8")


def write_cog(
    root: Path,
    name: str,
    year: int = 2020,
    col_off: int = 0,
    row_off: int = 0,
    data: np.ndarray | None = None,
    compress: str = "deflate",
) -> AEFTileInfo:
    """Write (or overwrite) a tiled int8 GeoTIFF under ``root``; return its tile info."""
    west = WEST + col_off * 10
    north = NORTH - row_off * 10
    values = pattern(year, col_off, row_off) if data is None else data
    with rasterio.open(
        root / name,
        "w",
        driver="GTiff",
        width=SIZE,
        height=SIZE,
        count=BANDS,
        dtype="int8",
        crs=f"EPSG:{EPSG}",
        transform=RasterAffine(10, 0, west, 0, -10, north),
        tiled=True,
        blockxsize=BLOCK,
        blockysize=BLOCK,
        compress=compress,
        interleave="band",
        nodata=-128,
    ) as ds:
        ds.write(values)
    bounds = (west, north - SIZE * 10, west + SIZE * 10, north)
    w, s, e, n = Transformer.from_crs(EPSG, 4326, always_xy=True).transform_bounds(
        *bounds
    )
    return AEFTileInfo(
        id=name,
        path=f"s3://bucket/{name}",
        year=year,
        bbox=(w, s, e, n),
        crs_epsg=EPSG,
        utm_zone="31N",
        utm_bounds=bounds,
    )


def local_reader(root: Path, **kwargs) -> VirtualTiffReader:
    """A reader whose s3 buckets are all served from the local directory ``root``."""
    reader = VirtualTiffReader(**kwargs)
    reader._get_s3_store = lambda bucket: LocalStore(str(root))
    return reader


async def read_tile(reader: VirtualTiffReader, tile: AEFTileInfo) -> np.ndarray:
    tree = await reader.open_tiles_by_zone([tile], chunks="native")
    return tree["31N"].ds.embeddings.isel(time=0).compute().values
