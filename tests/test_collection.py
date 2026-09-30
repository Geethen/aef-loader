import numpy as np
import pytest
import xarray as xr

from aef_loader.collection import (
    _REGISTRY,
    AEF,
    ENCODING_ATTR,
    TESSERA,
    Collection,
    get_collection,
    is_raw_encoded,
    list_collections,
    register_collection,
)


def test_zone_crs_per_collection():
    assert AEF.zone_crs("31N") == "EPSG:32631"
    assert AEF.zone_crs("35S") == "EPSG:32735"
    assert AEF.zone_crs("1n") == "EPSG:32601"
    # TESSERA keeps the northern CRS with negative northings in the south.
    assert TESSERA.zone_crs("35S") == "EPSG:32635"
    with pytest.raises(ValueError, match="UTM zone"):
        AEF.zone_crs("61N")
    with pytest.raises(ValueError, match="UTM zone"):
        AEF.zone_crs("unknown")


def test_band_names():
    assert AEF.band_names(64)[0] == "A00" and AEF.band_names(64)[-1] == "A63"
    assert TESSERA.band_names is None


def test_registry_lookup_and_duplicate_guard():
    assert {"aef-v1", "tessera-v1.1"} <= set(list_collections())
    assert get_collection("aef-v1") is AEF
    with pytest.raises(KeyError, match="registered"):
        get_collection("nope")
    with pytest.raises(ValueError, match="already registered"):
        register_collection(AEF)


def _aef_ds():
    raw = np.array([[[[127, -128]]]], dtype=np.int8)  # (time, band, y, x)
    return xr.Dataset({"embeddings": (("time", "band", "y", "x"), raw)})


def test_tag_and_aef_dequantize():
    ds = AEF.tag(_aef_ds())
    assert ds.embeddings.attrs[ENCODING_ATTR] == "aef-sqrt"
    assert ds.embeddings.attrs["nodata"] == -128
    assert is_raw_encoded(ds)

    out = AEF.dequantize(ds)
    assert out.embeddings.dtype == np.float32
    assert out.embeddings.attrs[ENCODING_ATTR] == "float"
    assert np.isnan(out.embeddings.values[0, 0, 0, 1])
    assert not is_raw_encoded(out)


def _tessera_ds():
    emb = np.array([[[[10, 3]]]], dtype=np.int8)
    scales = np.array([[[2.0, np.inf]]], dtype=np.float32)  # inf = never written
    counts = np.array([[[4, 0]]], dtype=np.int16)
    return TESSERA.tag(
        xr.Dataset(
            {
                "embeddings": (("time", "band", "y", "x"), emb),
                "scales": (("time", "y", "x"), scales),
                "s2_obs_count": (("time", "y", "x"), counts),
            }
        )
    )


def test_tessera_dequantize_masks_non_finite_scales():
    out = TESSERA.dequantize(_tessera_ds())
    assert "scales" not in out
    np.testing.assert_array_equal(out.embeddings.values[0, 0, 0], [20.0, np.nan])
    assert np.isnan(out.embeddings.attrs["nodata"])


def test_raw_check_uses_encoding_not_quality_var_dtype():
    raw = _tessera_ds()
    assert is_raw_encoded(raw)
    # Dequantized data with integer quality counts is NOT raw (Astra §4.11).
    assert not is_raw_encoded(TESSERA.dequantize(raw))


def test_raw_check_catches_float_cast_codes():
    # int8_to_float32-style output: codes cast to float, still the AEF encoding.
    ds = AEF.tag(_aef_ds())
    cast = ds.assign(embeddings=ds.embeddings.astype("float32"))
    assert is_raw_encoded(cast)


def test_untagged_falls_back_to_embeddings_dtype_only():
    assert is_raw_encoded(_aef_ds())
    untagged_float = xr.Dataset(
        {
            "embeddings": (("band", "y", "x"), np.zeros((1, 1, 1), np.float32)),
            "s2_obs_count": (("y", "x"), np.zeros((1, 1), np.int16)),
        }
    )
    assert not is_raw_encoded(untagged_float)


def test_custom_collection_registration():
    custom = Collection(
        name="test-custom",
        encoding="float",
        nodata=None,
        band_names=None,
        zone_crs=AEF.zone_crs,
        dequantize=lambda ds: ds,
    )
    register_collection(custom)
    try:
        assert get_collection("test-custom") is custom
    finally:
        _REGISTRY.pop("test-custom", None)
