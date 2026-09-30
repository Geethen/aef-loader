"""Offline tests for the one-source-zone-per-pixel merge in reproject_datatree."""

import numpy as np
import pytest
import xarray as xr

from aef_loader.utils import _merge_zone_datasets, _pixel_validity

Y, X = 2, 2


def _zone(name, times, emb, scales=None, nodata=-128, chunks=None, extra=None):
    """Synthetic reprojected zone on a shared grid; ``emb`` is (time, band, y, x)."""
    emb = np.asarray(emb)
    data = {
        "embeddings": xr.DataArray(
            emb,
            dims=("time", "band", "y", "x"),
            coords={"time": times, "band": ["A00", "A01"]},
            attrs={"nodata": nodata} if nodata is not None else {},
        )
    }
    if scales is not None:
        data["scales"] = xr.DataArray(
            np.asarray(scales, dtype=np.float32), dims=("time", "y", "x")
        )
    if extra is not None:
        data["extra"] = extra
    ds = xr.Dataset(data)
    ds.attrs["source_zone"] = name
    return ds.chunk(chunks) if chunks else ds


def test_astra_case_codes_and_scales_come_from_same_zone():
    z1 = _zone(
        "31N", [2024], np.zeros((1, 2, Y, X), np.int8), np.full((1, Y, X), np.nan), None
    )
    z2 = _zone(
        "32N", [2024], np.full((1, 2, Y, X), 10, np.int8), np.full((1, Y, X), 2), None
    )
    out = _merge_zone_datasets([z1, z2])
    assert (out.embeddings.values == 10).all()
    assert (out.scales.values == 2).all()
    assert (out.embeddings * out.scales).values.max() == 20


def test_zones_with_different_years_merge():
    z1 = _zone("31N", [2023], np.full((1, 2, Y, X), 5, np.int8))
    z2 = _zone("32N", [2024], np.full((1, 2, Y, X), 7, np.int8))
    out = _merge_zone_datasets([z1, z2])
    assert list(out.time.values.astype(int)) == [2023, 2024]
    assert (out.embeddings.sel(time=2023).values == 5).all()
    assert (out.embeddings.sel(time=2024).values == 7).all()


def test_int_zone_without_nodata_attr_raises():
    z1 = _zone("31N", [2024], np.zeros((1, 2, Y, X), np.int8), nodata=None)
    with pytest.raises(ValueError, match="31N"):
        _pixel_validity(z1, "31N")
    z2 = _zone("32N", [2024], np.zeros((1, 2, Y, X), np.int8))
    with pytest.raises(ValueError, match="31N"):
        _merge_zone_datasets([z1, z2])


def test_var_present_only_in_second_zone_survives():
    e1 = np.zeros((1, 2, Y, X), np.int8)
    e1[..., 0, 0] = -128  # zone 1 has no data here
    z1 = _zone("31N", [2024], e1)
    extra = xr.DataArray(
        np.full((1, Y, X), 3.0, np.float32),
        dims=("time", "y", "x"),
        coords={"time": [2024]},
    )
    z2 = _zone("32N", [2024], np.full((1, 2, Y, X), 10, np.int8), extra=extra)
    out = _merge_zone_datasets([z1, z2])
    assert "extra" in out.data_vars
    assert out.extra.values[0, 0, 0] == 3.0
    assert np.isnan(out.extra.values[0, 1, 1])  # zone 1 owns this pixel


def test_first_zone_wins_where_both_valid_and_dask_band_chunks_kept():
    e1 = np.full((1, 2, Y, X), 5, np.int8)
    e1[..., 0, 0] = -128  # invalid in zone 1 at one pixel
    e2 = np.full((1, 2, Y, X), 9, np.int8)
    z1 = _zone("31N", [2024], e1, chunks={"band": 1})
    z2 = _zone("32N", [2024], e2, chunks={"band": 1})
    out = _merge_zone_datasets([z1, z2])
    assert out.embeddings.chunks[1] == (1, 1)
    vals = out.embeddings.compute().values
    assert (vals[..., 0, 0] == 9).all()
    assert (vals[..., 1, 1] == 5).all()
    assert out.embeddings.dtype == np.int8


def _early_only_zones(chunks=None):
    """Zone 1 is invalid at (0, 0) and the only zone with ``extra``; zone 2 takes (0, 0)."""
    e1 = np.full((1, 2, Y, X), 5, np.int8)
    e1[..., 0, 0] = -128
    extra = xr.DataArray(
        np.full((1, Y, X), 99, np.int16),
        dims=("time", "y", "x"),
        coords={"time": [2024]},
        attrs={"nodata": -1},
    )
    z1 = _zone("31N", [2024], e1, chunks=chunks, extra=extra)
    z2 = _zone("32N", [2024], np.full((1, 2, Y, X), 10, np.int8), chunks=chunks)
    return z1, z2


def test_var_present_only_in_first_zone_is_filled_where_second_zone_takes():
    out = _merge_zone_datasets(list(_early_only_zones()))
    assert (out.embeddings.values[0, :, 0, 0] == 10).all()
    # the pixel came from zone 2, so zone 1's value must not survive next to it
    assert out.extra.values[0, 0, 0] == -1
    assert out.extra.values[0, 1, 1] == 99  # zone 1 still owns this pixel
    assert out.extra.dtype == np.int16
    assert out.extra.attrs == {"nodata": -1}


def test_var_present_only_in_first_zone_keeps_dask_chunks():
    out = _merge_zone_datasets(list(_early_only_zones(chunks={"band": 1, "y": 1})))
    assert out.extra.chunks == ((1,), (1, 1), (2,))
    assert out.extra.dims == ("time", "y", "x")
    vals = out.extra.compute().values
    assert vals[0, 0, 0] == -1 and vals[0, 1, 1] == 99
