"""
AEF Loader - Efficient loader for Alpha Earth Foundations embeddings.

Uses virtual-tiff to create virtual zarr stores from COGs,
enabling lazy xarray/dask operations without data duplication.

Supports both Google Cloud Storage (GCS) and Source Cooperative (AWS S3) backends,
and TESSERA embeddings (Zarr on Source Cooperative) via open_tessera().

The primary access pattern is loading tiles by UTM zone using VirtualTiffReader.open_tiles_by_zone().
For combining data across zones, use reproject_datatree() from the utils module.
"""

from importlib.metadata import PackageNotFoundError, version

from aef_loader.api import aopen_aef, open_aef
from aef_loader.chips import Chip, read_chip, read_chips
from aef_loader.constants import DataSource
from aef_loader.extract import (
    aextract_points,
    aextract_zonal,
    extract_points,
    extract_zonal,
)
from aef_loader.index import AEFIndex
from aef_loader.reader import VirtualTiffReader
from aef_loader.tessera import open_tessera
from aef_loader.types import AEFTileInfo
from aef_loader.utils import (
    aoi_geobox,
    dequantize_aef,
    int8_to_float32,
    mask_nodata,
    quantize_aef,
    reproject_datatree,
    set_aef_nodata,
    split_bands,
)

__all__ = [
    # Core classes
    "AEFIndex",
    "VirtualTiffReader",
    # Types
    "AEFTileInfo",
    "Chip",
    "DataSource",
    # Utility functions
    "aextract_points",
    "aextract_zonal",
    "aoi_geobox",
    "aopen_aef",
    "dequantize_aef",
    "extract_points",
    "extract_zonal",
    "int8_to_float32",
    "mask_nodata",
    "open_aef",
    "open_tessera",
    "read_chip",
    "read_chips",
    "quantize_aef",
    "reproject_datatree",
    "set_aef_nodata",
    "split_bands",
]

try:
    __version__ = version("aef-loader-plus")
except PackageNotFoundError:  # not installed (e.g. run from a source tree)
    __version__ = "0+unknown"
