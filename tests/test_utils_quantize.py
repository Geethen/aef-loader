"""Regression tests for quantize_aef / dequantize_aef edge cases."""

import warnings

import numpy as np
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
