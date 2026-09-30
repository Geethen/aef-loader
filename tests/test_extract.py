"""Offline tests for point/zonal extraction and the open_aef entrypoints."""

import asyncio

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from odc.geo.xr import assign_crs
from pyproj import Transformer
from shapely.geometry import box

import aef_loader.api as api
import aef_loader.extract as extract
from aef_loader import AEFIndex, dequantize_aef
from aef_loader.collection import AEF
from aef_loader.types import AEFTileInfo

BANDS = ["A00", "A01", "A02"]
RES = 10.0


def make_tile(name, epsg, west, north, codes, year=2024):
    """Synthetic AEF tile: ``codes`` is an int8 (band, y, x) array on a 10 m north-up grid."""
    codes = np.asarray(codes, dtype="int8")
    _, ny, nx = codes.shape
    xs = west + RES * (np.arange(nx) + 0.5)
    ys = north - RES * (np.arange(ny) + 0.5)
    emb = xr.DataArray(
        codes[None],
        dims=("time", "band", "y", "x"),
        coords={"time": [np.datetime64(f"{year}-01-01")], "band": BANDS, "y": ys, "x": xs},
        name="embeddings",
    )
    ds = AEF.tag(emb.to_dataset())
    ds = assign_crs(ds, f"EPSG:{epsg}")
    tile = AEFTileInfo(
        id=name,
        path=f"s3://bucket/{name}.tif",
        year=year,
        bbox=(0, 0, 1, 1),
        crs_epsg=epsg,
        utm_zone=f"{epsg - 32600}N",
        utm_bounds=(west, north - RES * ny, west + RES * nx, north),
        source=None,
    )
    return tile, ds


def unique_codes(nx=4, ny=4, offset=0):
    """Distinct code per (band, row, col): 10*band + 4*row + col + offset, all in int8 range."""
    b, r, c = np.meshgrid(np.arange(3), np.arange(ny), np.arange(nx), indexing="ij")
    return (10 * b + nx * r + c + offset).astype("int8")


@pytest.fixture
def two_tiles():
    a = make_tile("A", 32631, 500000, 6500000, unique_codes())
    b = make_tile("B", 32631, 500040, 6500000, unique_codes(offset=50))
    tiles = [a[0], b[0]]
    return tiles, {a[0].path: a[1], b[0].path: b[1]}


def run_points(pts, tiles, datasets, years=(2024,), **kw):
    x = np.array([p[0] for p in pts], dtype="float64")
    y = np.array([p[1] for p in pts], dtype="float64")
    return extract._extract_points_from_datasets(
        x, y, list(range(len(pts))), list(years), "EPSG:32631", tiles, datasets, **kw
    )


def test_points_exact_pixel_selection(two_tiles):
    tiles, datasets = two_tiles
    # centre of (row 1, col 2) in A, and (row 3, col 1) in B
    df = run_points([(500025, 6499985), (500045 + 10, 6499965)], tiles, datasets, dequantize=False)
    assert list(df.columns[:6]) == ["point_id", "year", "x", "y", "tile_id", "utm_zone"]
    assert df.tile_id.tolist() == ["A", "B"]
    assert df.utm_zone.tolist() == ["31N", "31N"]
    assert df.loc[0, BANDS].tolist() == [4 + 2, 10 + 4 + 2, 20 + 4 + 2]
    assert df.loc[1, BANDS].tolist() == [50 + 12 + 1, 50 + 10 + 13, 50 + 20 + 13]
    assert all(df[BANDS].dtypes == np.int8)


def test_point_on_pixel_edge_belongs_to_right_and_lower_pixel(two_tiles):
    tiles, datasets = two_tiles
    # x = 500010 is the edge between cols 0/1; y = 6499990 the edge between rows 0/1.
    df = run_points([(500010, 6499990)], tiles, datasets, dequantize=False)
    assert df.loc[0, "A00"] == 4 * 1 + 1  # row 1, col 1
    # A tile's east edge belongs to the neighbouring tile (half-open bounds).
    df = run_points([(500040, 6499995)], tiles, datasets, dequantize=False)
    assert df.tile_id.tolist() == ["B"]
    assert df.loc[0, "A00"] == 50  # B row 0 col 0
    # The tile's north edge belongs to the tile (row 0), its south edge does not.
    df = run_points([(500005, 6500000), (500005, 6499960)], tiles, datasets, dequantize=False)
    assert df.tile_id.tolist() == ["A", None]
    assert df.loc[0, "A00"] == 0


def test_dequantized_values_and_missing_rows(two_tiles):
    tiles, datasets = two_tiles
    df = run_points([(500005, 6499995), (1.0, 1.0)], tiles, datasets, years=(2023, 2024))
    assert len(df) == 4  # point-major, one row per year
    assert df.year.tolist() == [2023, 2024, 2023, 2024]
    assert all(df[BANDS].dtypes == np.float32)
    np.testing.assert_array_equal(
        df.loc[1, BANDS].to_numpy(dtype="float32"),
        dequantize_aef(np.array([0, 10, 20], dtype="int8")),
    )
    assert df.loc[[0, 2, 3], BANDS].isna().all().all()  # no 2023 tile / outside every tile
    assert df.tile_id.tolist() == [None, "A", None, None]
    raw = run_points([(1.0, 1.0)], tiles, datasets, dequantize=False)
    assert (raw[BANDS] == -128).all().all()


def test_nodata_pixel_is_nan(two_tiles):
    tiles, datasets = two_tiles
    ds = datasets[tiles[0].path]
    codes = ds["embeddings"].values.copy()
    codes[0, :, 0, 0] = -128
    datasets[tiles[0].path] = ds.assign(embeddings=ds["embeddings"].copy(data=codes))
    df = run_points([(500005, 6499995), (500015, 6499995)], tiles, datasets)
    assert df.loc[0, BANDS].isna().all()
    assert df.loc[1, BANDS].notna().all()


def test_points_from_other_crs_and_dask_backed(two_tiles):
    tiles, datasets = two_tiles
    datasets = {k: v.chunk({"band": 1, "y": 2, "x": 2}) for k, v in datasets.items()}
    to_ll = Transformer.from_crs(32631, 4326, always_xy=True)
    lon, lat = to_ll.transform(500025.0, 6499985.0)
    df = extract._extract_points_from_datasets(
        np.array([lon]), np.array([lat]), ["p"], [2024], "EPSG:4326", tiles, datasets,
        dequantize=False,
    )
    assert df.point_id.tolist() == ["p"]
    assert df.loc[0, "A00"] == 6


def test_zonal_mean_matches_hand_computed(two_tiles):
    tiles, datasets = two_tiles
    poly = box(500010, 6499970, 500030, 6499990)  # centres of rows 1-2, cols 1-2 of A
    df = extract._extract_zonal_from_datasets(
        [poly], ["z"], [2024], "EPSG:32631", tiles, datasets
    )
    codes = datasets[tiles[0].path]["embeddings"].values[0][:, 1:3, 1:3]
    expected = dequantize_aef(codes).reshape(3, -1).mean(axis=1)
    assert df.polygon_id.tolist() == ["z"]
    assert df.n_pixels.tolist() == [4]
    np.testing.assert_allclose(df.loc[0, BANDS].to_numpy(dtype="float64"), expected, rtol=1e-6)
    # dequantize-then-average, not average-then-dequantize
    naive = dequantize_aef(codes.reshape(3, -1).mean(axis=1).round().astype("int8"))
    assert not np.allclose(expected, naive, rtol=1e-3)


def test_zonal_multi_tile_uses_sums_and_counts(two_tiles):
    tiles, datasets = two_tiles
    # cols 2-3 of A (8 px) + col 0 of B (4 px): unequal counts per tile.
    poly = box(500020, 6499960, 500050, 6500000)
    all_codes = [
        datasets[tiles[0].path]["embeddings"].values[0][:, :, 2:4],
        datasets[tiles[1].path]["embeddings"].values[0][:, :, 0:1],
    ]
    pixels = np.concatenate([dequantize_aef(c).reshape(3, -1) for c in all_codes], axis=1)
    assert pixels.shape[1] == 12

    def zonal(stat):
        return extract._extract_zonal_from_datasets(
            [poly], [0], [2024], "EPSG:32631", tiles, datasets, stat=stat
        )

    mean = zonal("mean")
    assert mean.n_pixels.tolist() == [12]
    got = mean.loc[0, BANDS].to_numpy(dtype="float64")
    np.testing.assert_allclose(got, pixels.mean(axis=1), rtol=1e-6)
    mean_of_means = np.mean([dequantize_aef(c).reshape(3, -1).mean(axis=1) for c in all_codes], axis=0)
    assert not np.allclose(got, mean_of_means, rtol=1e-3)
    for stat, fn in [("median", np.median), ("std", np.std), ("min", np.min), ("max", np.max)]:
        np.testing.assert_allclose(
            zonal(stat).loc[0, BANDS].to_numpy(dtype="float64"),
            fn(pixels, axis=1),
            rtol=1e-5,
            atol=1e-7,
            err_msg=stat,
        )
    assert zonal("count").loc[0, BANDS].tolist() == [12, 12, 12]


def test_zonal_ignores_nodata_and_counts_valid_pixels(two_tiles):
    tiles, datasets = two_tiles
    ds = datasets[tiles[0].path]
    codes = ds["embeddings"].values.copy()
    codes[0, :, 1, 1] = -128
    datasets[tiles[0].path] = ds.assign(embeddings=ds["embeddings"].copy(data=codes))
    poly = box(500010, 6499970, 500030, 6499990)
    df = extract._extract_zonal_from_datasets(
        [poly], [0], [2024], "EPSG:32631", tiles, datasets
    )
    assert df.n_pixels.tolist() == [3]
    expected = dequantize_aef(codes[0][:, 1:3, 1:3]).reshape(3, -1)
    np.testing.assert_allclose(
        df.loc[0, BANDS].to_numpy(dtype="float64"), np.nanmean(expected, axis=1), rtol=1e-6
    )


def test_zonal_overlapping_tiles_count_each_pixel_once():
    a = make_tile("A", 32631, 500000, 6500000, unique_codes())
    b = make_tile("B", 32631, 500020, 6500000, unique_codes(offset=50))  # overlaps cols 2-3 of A
    tiles, datasets = [a[0], b[0]], {a[0].path: a[1], b[0].path: b[1]}
    poly = box(500000, 6499960, 500060, 6500000)
    df = extract._extract_zonal_from_datasets([poly], [0], [2024], "EPSG:32631", tiles, datasets)
    assert df.n_pixels.tolist() == [24]  # cols 0..5 x 4 rows; the 8 shared pixels count once
    df_a = extract._extract_zonal_from_datasets(
        [box(500000, 6499960, 500040, 6500000)], [0], [2024], "EPSG:32631", tiles[:1],
        {a[0].path: a[1]},
    )
    assert df_a.n_pixels.tolist() == [16]


def test_zonal_across_utm_zones_counts_once():
    to_utm = {e: Transformer.from_crs(4326, e, always_xy=True) for e in (32631, 32632)}
    lon, lat = 6.0, 50.0  # on the 31N/32N boundary
    tiles, datasets = [], {}
    for epsg in (32631, 32632):
        x, y = to_utm[epsg].transform(lon, lat)
        west, north = np.floor(x / 10) * 10 - 200, np.floor(y / 10) * 10 + 200
        rng = np.random.default_rng(epsg)
        tile, ds = make_tile(f"T{epsg}", epsg, west, north, rng.integers(-120, 120, (3, 40, 40)))
        tiles.append(tile)
        datasets[tile.path] = ds
    poly = box(lon - 0.0003, lat - 0.0002, lon + 0.0003, lat + 0.0002)
    both = extract._extract_zonal_from_datasets([poly], [0], [2024], "EPSG:4326", tiles, datasets)
    singles = [
        extract._extract_zonal_from_datasets(
            [poly], [0], [2024], "EPSG:4326", [t], {t.path: datasets[t.path]}
        ).n_pixels[0]
        for t in tiles
    ]
    assert min(singles) > 20
    assert abs(both.n_pixels[0] - max(singles)) <= 0.1 * max(singles)  # not the sum


def test_zonal_stat_validation():
    with pytest.raises(ValueError, match="stat"):
        extract._check_stat("mode")


# ---------------------------------------------------------------------------
# public async/sync entry points against a fake index (no network)
# ---------------------------------------------------------------------------


def fake_index(tiles):
    rows = [
        {
            "fid": t.id,
            "crs": f"EPSG:{t.crs_epsg}",
            "path": t.path,
            "year": t.year,
            "utm_zone": t.utm_zone,
            "utm_west": t.utm_bounds[0],
            "utm_south": t.utm_bounds[1],
            "utm_east": t.utm_bounds[2],
            "utm_north": t.utm_bounds[3],
            "wgs84_west": -10.0,
            "wgs84_south": 0.0,
            "wgs84_east": 20.0,
            "wgs84_north": 80.0,
        }
        for t in tiles
    ]
    index = AEFIndex(cache_dir="unused")
    index._df = pd.DataFrame(rows)
    return index


def test_public_extract_points_and_zonal_with_fake_index(two_tiles, monkeypatch):
    tiles, datasets = two_tiles
    index = fake_index(tiles)
    opened = []

    async def fake_open(needed, reader, gcp_project):
        opened.extend(t.id for t in needed)
        return {t.path: datasets[t.path] for t in needed}

    monkeypatch.setattr(extract, "_open_tiles", fake_open)
    to_ll = Transformer.from_crs(32631, 4326, always_xy=True)
    lon, lat = to_ll.transform(500025.0, 6499985.0)
    # the fake index reports one broad WGS84 bbox, so tiles are chosen by native bounds
    df = extract.extract_points([(lon, lat)], 2024, index=index, dequantize=False)
    assert df.loc[0, "A00"] == 6 and df.tile_id.tolist() == ["A"]
    assert opened == ["A"]  # only the tile that holds a point is opened
    poly = box(lon - 1e-5, lat - 1e-5, lon + 1e-5, lat + 1e-5)
    z = extract.extract_zonal([poly], 2024, index=index)
    assert z.n_pixels.tolist() == [1]


# ---------------------------------------------------------------------------
# open_aef
# ---------------------------------------------------------------------------


class FakeReader:
    instances = 0

    def __init__(self, gcp_project=None, **kwargs):
        FakeReader.instances += 1
        self.kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def open_tiles_by_zone(self, tiles, **kwargs):
        await asyncio.sleep(0)
        return {"tiles": [t.id for t in tiles], **kwargs}


@pytest.fixture
def patched(two_tiles, monkeypatch):
    monkeypatch.setattr(api, "VirtualTiffReader", FakeReader)
    return fake_index(two_tiles[0])


def test_open_aef_without_running_loop(patched):
    out = api.open_aef((5.5, 51.0, 5.6, 51.1), 2024, index=patched, buffer_pixels=2)
    assert out["tiles"] == ["A", "B"]
    assert out["bbox"] == (5.5, 51.0, 5.6, 51.1)
    assert out["chunks"] == "balanced" and out["buffer_pixels"] == 2


async def test_open_aef_inside_running_loop(patched):
    asyncio.get_running_loop()  # we are in a loop, as in Jupyter
    out = api.open_aef((5.5, 51.0, 5.6, 51.1), 2024, index=patched)
    assert out["tiles"] == ["A", "B"]
    assert (await api.aopen_aef((5.5, 51.0, 5.6, 51.1), 2024, index=patched))["tiles"] == ["A", "B"]


def test_reader_kwargs_routing(patched):
    reader = FakeReader()
    with pytest.raises(TypeError, match="manifest_cache_dir"):
        api.open_aef((5.5, 51.0, 5.6, 51.1), 2024, index=patched, reader=reader,
                     manifest_cache_dir="x")


def test_shared_index_is_reused(tmp_path):
    one = api.get_shared_index("source_coop", None, tmp_path)
    assert api.get_shared_index("source_coop", None, tmp_path) is one
    assert api.get_shared_index("source_coop", None, tmp_path / "other") is not one
    assert api.get_shared_index("gcs", "proj", tmp_path) is not one
