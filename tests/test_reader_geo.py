"""Offline tests for GeoTIFF georeferencing helpers and zone grouping in the reader."""

import pytest
import xarray as xr
from affine import Affine

from aef_loader.reader import (
    _get_affine_from_model_pixel_scale_and_tiepoint,
    _get_geobox_from_dataset,
)


def _ds(**attrs):
    return xr.Dataset({"embeddings": (("y", "x"), [[0, 0], [0, 0]], attrs)})


def test_pixel_scale_tiepoint_origin_is_north_up():
    aff = _get_affine_from_model_pixel_scale_and_tiepoint(
        (10.0, 10.0, 0.0), (0, 0, 0, 500000.0, 4000000.0, 0)
    )
    assert aff == Affine(10, 0, 500000, 0, -10, 4000000)


def test_pixel_scale_tiepoint_honours_raster_indices():
    aff = _get_affine_from_model_pixel_scale_and_tiepoint(
        (10.0, 10.0, 0.0), (2, 3, 0, 500020.0, 3999970.0, 0)
    )
    assert aff == Affine(10, 0, 500000, 0, -10, 4000000)


def test_missing_tiepoint_raises():
    with pytest.raises(ValueError, match="model_tiepoint"):
        _get_affine_from_model_pixel_scale_and_tiepoint((10.0, 10.0, 0.0), None)
    with pytest.raises(ValueError, match="model_tiepoint"):
        _get_geobox_from_dataset(_ds(model_pixel_scale=(10.0, 10.0, 0.0)), "EPSG:32633")


def test_model_transformation_preferred_over_pixel_scale():
    ds = _ds(
        model_pixel_scale=(1.0, 1.0, 0.0),
        model_tiepoint=(0, 0, 0, 0.0, 0.0, 0),
        model_transformation=(10, 0, 0, 500000, 0, -10, 0, 4000000, 0, 0, 0, 0, 0, 0, 0, 1),
    )
    gb = _get_geobox_from_dataset(ds, "EPSG:32633")
    assert gb.affine == Affine(10, 0, 500000, 0, -10, 4000000)


# --- zone grouping -----------------------------------------------------------

import asyncio  # noqa: E402
from types import MethodType  # noqa: E402

import numpy as np  # noqa: E402

import aef_loader.reader as reader_module  # noqa: E402
from aef_loader import AEFTileInfo  # noqa: E402
from aef_loader.reader import VirtualTiffReader  # noqa: E402


def _tile(i, zone, epsg):
    return AEFTileInfo(
        id=str(i),
        path=f"s3://bucket/{i}.tif",
        year=2024,
        bbox=(0, 0, 1, 1),
        crs_epsg=epsg,
        utm_zone=zone,
    )


def _open(tiles, monkeypatch):
    seen = []

    async def fake_combine(self, zone_tiles, ifd=0, **kwargs):
        seen.append([t.id for t in zone_tiles])
        return xr.Dataset({"v": ("x", np.array([1], dtype=np.int8))})

    reader = VirtualTiffReader()
    reader._combine_tiles_single_zone = MethodType(fake_combine, reader)
    monkeypatch.setattr(reader_module, "assign_crs", lambda ds, crs: ds)
    tree = asyncio.run(reader.open_tiles_by_zone(tiles))
    return tree, seen


def test_tiles_without_zone_are_grouped_by_crs(monkeypatch):
    tiles = [_tile(1, None, 32633), _tile(2, None, 32634), _tile(3, None, 32633)]
    tree, seen = _open(tiles, monkeypatch)
    assert sorted(tree.children) == ["EPSG32633", "EPSG32634"]
    assert sorted(seen) == [["1", "3"], ["2"]]


def test_zone_label_kept_when_single_crs(monkeypatch):
    tree, _ = _open([_tile(1, "33N", 32633), _tile(2, "34N", 32634)], monkeypatch)
    assert sorted(tree.children) == ["33N", "34N"]


def test_one_zone_label_with_multiple_crs_raises(monkeypatch):
    with pytest.raises(ValueError, match="33N"):
        _open([_tile(1, "33N", 32633), _tile(2, "33N", 32733)], monkeypatch)


def test_no_tiles_error_says_what_was_searched():
    reader = VirtualTiffReader()
    with pytest.raises(ValueError, match=r"bbox=\(1, 2, 3, 4\).*EPSG:4326"):
        asyncio.run(reader.open_tiles_by_zone([], bbox=(1, 2, 3, 4)))


def test_reproject_empty_tree_error_lists_zones():
    from odc.geo.geobox import GeoBox
    from xarray import DataTree

    from aef_loader.utils import reproject_datatree

    tree = DataTree.from_dict({"/33N": xr.Dataset()})
    target = GeoBox.from_bbox((0, 0, 100, 100), crs="EPSG:32633", resolution=10)
    with pytest.raises(ValueError, match=r"33N.*EPSG:32633"):
        reproject_datatree(tree, target)
