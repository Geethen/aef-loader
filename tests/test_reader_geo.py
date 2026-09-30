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
