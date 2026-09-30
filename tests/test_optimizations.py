from __future__ import annotations

import asyncio
from types import MethodType, SimpleNamespace

import dask.array as da
import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from affine import Affine
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_coords
from shapely.geometry import box

import aef_loader.reader as reader_module
from aef_loader import AEFIndex, AEFTileInfo, DataSource
from aef_loader.reader import (
    VirtualTiffReader,
    _concat_time_slices,
    _crop_to_bbox,
    _resolve_chunks,
)
from aef_loader.utils import dequantize_aef


def _reference_dequantize(raw, divisor=127.5, nodata=-128):
    raw = np.asarray(raw)
    normalized = raw.astype(np.float32) / divisor
    result = (normalized**2) * np.sign(raw)
    result[raw == nodata] = np.nan
    return result


def test_dequantize_all_codes_is_bit_identical():
    raw = np.arange(-128, 128, dtype=np.int16).astype(np.int8)
    actual = dequantize_aef(raw)
    expected = _reference_dequantize(raw)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float32

    # Wider arrays retain the established fallback path.
    np.testing.assert_array_equal(dequantize_aef(raw.astype(np.int16)), expected)


def test_dequantize_eager_and_dask_dataarrays_match():
    raw = np.arange(-128, 128, dtype=np.int16).astype(np.int8).reshape(16, 16)
    eager = xr.DataArray(raw, dims=("y", "x"), attrs={"source": "test"})
    lazy = xr.DataArray(da.from_array(raw, chunks=(4, 8)), dims=("y", "x"))

    eager_result = dequantize_aef(eager)
    lazy_result = dequantize_aef(lazy)
    np.testing.assert_array_equal(eager_result.values, lazy_result.compute().values)
    np.testing.assert_array_equal(eager_result.values, _reference_dequantize(raw))
    assert lazy_result.dtype == np.float32
    assert lazy_result.data.chunks == lazy.data.chunks


def test_temporal_outer_join_stays_int8_and_uses_nodata():
    first = xr.Dataset(
        {"v": (("time", "x"), np.array([[1, 2]], dtype=np.int8))},
        coords={"time": [2020], "x": [0, 1]},
    )
    second = xr.Dataset(
        {"v": (("time", "x"), np.array([[3, 4]], dtype=np.int8))},
        coords={"time": [2021], "x": [1, 2]},
    )
    result = _concat_time_slices([first, second])
    assert result.v.dtype == np.int8
    np.testing.assert_array_equal(
        result.v.values,
        np.array([[1, 2, -128], [-128, 3, 4]], dtype=np.int8),
    )


def _fake_manifest():
    metadata = SimpleNamespace(
        shape=(64, 8192, 8192),
        dimension_names=("band", "y", "x"),
        chunk_grid=SimpleNamespace(chunk_shape=(1, 1024, 1024)),
    )
    array = SimpleNamespace(metadata=metadata)
    return SimpleNamespace(_group=SimpleNamespace(arrays={"0": array}))


@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        ("native", {"band": 1, "y": 1024, "x": 1024}),
        ("balanced", {"band": 16, "y": 1024, "x": 1024}),
        ("all-bands", {"band": 64, "y": 1024, "x": 1024}),
        (None, {"band": 1, "y": 1024, "x": 1024}),
    ],
)
def test_chunk_profiles_are_storage_aligned(profile, expected):
    assert _resolve_chunks(_fake_manifest(), profile) == expected


def _crop_fixture():
    geobox = GeoBox(
        shape=(10, 10),
        affine=Affine(10, 0, 0, 0, -10, 100),
        crs="EPSG:32632",
    )
    coords = xr_coords(geobox)
    raw = np.arange(100, dtype=np.int8).reshape(10, 10)
    ds = xr.Dataset(
        {"v": (("y", "x"), raw)},
        coords={"x": coords["x"].values, "y": coords["y"].values},
    )
    return ds, geobox


def test_aoi_crop_is_an_unmodified_source_subset():
    ds, geobox = _crop_fixture()
    cropped = _crop_to_bbox(ds, geobox, (20, 40, 50, 80), "EPSG:32632", buffer_pixels=0)
    # Cells merely touching the AOI edge are excluded: x 20-50, y 40-80 only.
    assert cropped.sizes["x"] == 3
    assert cropped.sizes["y"] == 4
    expected = ds.sel(x=cropped.x, y=cropped.y)
    xr.testing.assert_identical(cropped, expected)
    assert np.shares_memory(cropped.v.values, ds.v.values)


def test_aoi_crop_subpixel_aoi_selects_covering_cell():
    ds, geobox = _crop_fixture()
    cropped = _crop_to_bbox(ds, geobox, (23, 43, 27, 47), "EPSG:32632", buffer_pixels=0)
    assert (cropped.sizes["y"], cropped.sizes["x"]) == (1, 1)
    assert float(cropped.x[0]) == 25.0 and float(cropped.y[0]) == 45.0


def test_aoi_crop_buffer_pixels_expand_each_side():
    ds, geobox = _crop_fixture()
    cropped = _crop_to_bbox(ds, geobox, (20, 40, 50, 80), "EPSG:32632", buffer_pixels=1)
    assert (cropped.sizes["y"], cropped.sizes["x"]) == (6, 5)


def test_aoi_crop_outside_tile_is_empty():
    ds, geobox = _crop_fixture()
    cropped = _crop_to_bbox(ds, geobox, (500, 500, 600, 600), "EPSG:32632", 0)
    assert cropped.sizes["x"] == 0 or cropped.sizes["y"] == 0


def _index_frame():
    records = []
    geometries = []
    for i in range(200):
        x = i % 20
        y = i // 20
        year = 2020 + i % 3
        records.append(
            {
                "fid": i,
                "path": f"s3://bucket/{i}.tif",
                "year": year,
                "wgs84_west": x,
                "wgs84_south": y,
                "wgs84_east": x + 0.9,
                "wgs84_north": y + 0.9,
                "crs": "EPSG:32632",
                "utm_zone": "32N",
                "utm_west": x * 100,
                "utm_south": y * 100,
                "utm_east": (x + 1) * 100,
                "utm_north": (y + 1) * 100,
            }
        )
        geometries.append(box(x, y, x + 0.9, y + 0.9))
    return gpd.GeoDataFrame(records, geometry=geometries, crs="EPSG:4326")


def test_spatial_index_query_matches_legacy_scan_and_order():
    gdf = _index_frame()
    index = AEFIndex(source=DataSource.SOURCE_COOP)
    index._gdf = gdf
    bbox = (3.2, 2.2, 13.2, 7.2)
    years = (2020, 2021)
    legacy = gdf[gdf.geometry.intersects(box(*bbox))]
    legacy = legacy[(legacy.year >= years[0]) & (legacy.year <= years[1])].head(12)

    actual = asyncio.run(index.query(bbox=bbox, years=years, limit=12))
    assert [tile.id for tile in actual] == [str(v) for v in legacy.fid]
    assert [tile.path for tile in actual] == list(legacy.path)
    assert [tile.bbox for tile in actual] == list(
        zip(
            legacy.wgs84_west,
            legacy.wgs84_south,
            legacy.wgs84_east,
            legacy.wgs84_north,
        )
    )


def test_manifest_memory_cache_is_bounded_lru():
    reader = VirtualTiffReader(memory_manifest_cache_size=2)
    reader._remember_manifest(("a", 0), object())
    reader._remember_manifest(("b", 0), object())
    reader._manifest_cache[("a", 0)]
    reader._manifest_cache.move_to_end(("a", 0))
    reader._remember_manifest(("c", 0), object())
    assert list(reader._manifest_cache) == [("a", 0), ("c", 0)]


def test_zone_processing_obeys_shared_concurrency_limit(monkeypatch):
    active = 0
    peak = 0

    async def fake_combine(self, tiles, ifd=0, **kwargs):
        nonlocal active, peak
        semaphore = kwargs["semaphore"]
        async with semaphore:
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
        return xr.Dataset({"v": ("x", np.array([1], dtype=np.int8))})

    reader = VirtualTiffReader()
    reader._combine_tiles_single_zone = MethodType(fake_combine, reader)
    monkeypatch.setattr(reader_module, "assign_crs", lambda ds, crs: ds)
    tiles = [
        AEFTileInfo(
            id=str(i),
            path=f"s3://bucket/{i}.tif",
            year=2024,
            bbox=(0, 0, 1, 1),
            crs_epsg=32632,
            utm_zone=f"{i}N",
        )
        for i in range(1, 5)
    ]
    tree = asyncio.run(reader.open_tiles_by_zone(tiles, max_concurrency=2))
    assert len(tree.children) == 4
    assert peak == 2
    assert reader.stats["open_calls"] == 1
    assert reader.stats["last_open_wall_seconds"] > 0


def _zone_tiles(zones):
    return [
        AEFTileInfo(
            id=zone,
            path=f"s3://bucket/{zone}.tif",
            year=2024,
            bbox=(0, 0, 1, 1),
            crs_epsg=32632,
            utm_zone=zone,
        )
        for zone in zones
    ]


def test_empty_crop_is_detected():
    ds, geobox = _crop_fixture()
    outside = _crop_to_bbox(ds, geobox, (500, 500, 600, 600), "EPSG:32632", 0)
    assert reader_module._is_empty_crop(outside)
    assert not reader_module._is_empty_crop(ds)


def test_zones_with_no_overlap_are_dropped_and_all_empty_raises(monkeypatch):
    async def fake_combine(self, tiles, ifd=0, **kwargs):
        if tiles[0].utm_zone == "2N":
            return None  # every tile in this zone had an empty crop
        return xr.Dataset({"v": ("x", np.array([1], dtype=np.int8))})

    reader = VirtualTiffReader()
    reader._combine_tiles_single_zone = MethodType(fake_combine, reader)
    monkeypatch.setattr(reader_module, "assign_crs", lambda ds, crs: ds)

    tree = asyncio.run(reader.open_tiles_by_zone(_zone_tiles(["1N", "2N"])))
    assert list(tree.children) == ["1N"]
    assert tree.attrs["zones"] == ["1N"]

    with pytest.raises(ValueError, match=r"bbox .*EPSG:4326"):
        asyncio.run(
            reader.open_tiles_by_zone(_zone_tiles(["2N"]), bbox=(1, 2, 3, 4))
        )
