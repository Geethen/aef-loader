"""Regression tests for quantize_aef / dequantize_aef edge cases."""

import warnings

import numpy as np
import pytest
import xarray as xr

from aef_loader.utils import dequantize_aef, quantize_aef


def test_quantize_nan_and_inf_become_nodata_without_warning():
    data = np.array([0.0, 0.5, np.nan, np.inf, -np.inf, -1.0], dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = quantize_aef(data)
    assert out.dtype == np.int8
    assert out[2] == out[3] == out[4] == -128
    assert out[0] == 0 and out[5] == -127


def test_quantize_roundtrip_keeps_nodata_dataarray_dask():
    raw = np.array([[-128, 0], [64, 127]], dtype=np.int8)
    da = xr.DataArray(raw, dims=("y", "x")).chunk({"y": 1})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        back = quantize_aef(dequantize_aef(da)).compute()
    np.testing.assert_array_equal(back.values, raw)
    assert back.attrs["nodata"] == -128


def test_dequantize_rejects_float_input():
    with pytest.raises(TypeError, match="already dequantized"):
        dequantize_aef(np.array([0.5, -0.25], dtype=np.float32))
    with pytest.raises(TypeError):
        dequantize_aef(xr.DataArray(np.array([0.5]), dims=("x",)))


def test_dequantize_wide_int_range_checked():
    with pytest.raises(ValueError, match="-128, 127"):
        dequantize_aef(np.array([-200, 0], dtype=np.int16))
    with pytest.raises(ValueError):
        dequantize_aef(np.array([200], dtype=np.int16))
    lazy = xr.DataArray(np.array([-200, 0], dtype=np.int16), dims=("x",)).chunk(1)
    out = dequantize_aef(lazy)  # stays lazy
    with pytest.raises(ValueError):
        out.compute()


def test_dequantize_wide_int_in_range_matches_int8():
    raw = np.arange(-128, 128, dtype=np.int8)
    np.testing.assert_array_equal(
        dequantize_aef(raw.astype(np.int16)), dequantize_aef(raw)
    )


def test_dequantize_rejects_dequantized_float_dask_input_at_compute():
    import dask.array as da
    import xarray as xr

    lazy = xr.DataArray(da.from_array(np.array([0.25, 0.5], dtype=np.float32), chunks=1), dims="x")
    with pytest.raises(TypeError, match="already dequantized"):
        dequantize_aef(lazy).compute()  # float is validated per block


def test_dequantize_rejects_bool_dask_input_at_call_time():
    import dask.array as da
    import xarray as xr

    lazy = xr.DataArray(da.from_array(np.array([True, False]), chunks=1), dims="x")
    with pytest.raises(TypeError, match="already dequantized"):
        dequantize_aef(lazy)  # must not wait until .compute()


def test_aoi_geobox_snap_true_uses_lattice_and_snap_false_is_aoi_anchored():
    from aef_loader.utils import aoi_geobox

    bbox = (100.3, 200.7, 150.9, 260.4)
    snapped = aoi_geobox(bbox, "EPSG:32633", 10, snap=True)
    assert (snapped.affine.c, snapped.affine.f) == (100.0, 270.0)

    anchored = aoi_geobox(bbox, "EPSG:32633", 10, snap=False)
    assert anchored.affine.c == 100.3
    assert anchored.affine.f == 260.4
