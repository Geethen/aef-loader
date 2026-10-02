"""Offline tests for chip geometry, tile-edge mosaics and shared block reads."""

import asyncio
import datetime as dt

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from affine import Affine
from pyproj import Transformer
from xarray import DataTree

from aef_loader import Chip, read_chip, read_chips
from aef_loader.chips import chip_axes
from aef_loader.collection import AEF
from aef_loader.types import AEFTileInfo

EPSG = 32631
N = 16  # tile edge in pixels
BANDS = 2
BLOCK = 8
WEST, NORTH = 500_000.0, 6_500_160.0


def value(year, band, row, col):
    """Deterministic code for a global (row, col) of the test lattice."""
    return (row * 7 + col * 3 + band * 11 + (year - 2020) * 5) % 100


class CountingSource:
    """Array-like backing one tile; records every chunk read."""

    def __init__(self, year, west, north, loads, tag, south_up):
        self.year, self.west, self.north = year, west, north
        self.loads, self.tag, self.south_up = loads, tag, south_up
        self.shape, self.dtype, self.ndim = (BANDS, N, N), np.dtype("int8"), 3

    def __getitem__(self, key):
        bs, ys, xs = key
        self.loads.append((self.tag, bs.start, ys.start, xs.start))
        b = np.arange(BANDS)[bs][:, None, None]
        r = np.arange(N)[ys][None, :, None]
        c = np.arange(N)[xs][None, None, :]
        if self.south_up:
            r = N - 1 - r
        grow = round((NORTH - self.north) / 10) + r
        gcol = round((self.west - WEST) / 10) + c
        return value(self.year, b, grow, gcol).astype("int8")


def make_tile(year, col_off, tag="t"):
    west = WEST + col_off * 10
    bounds = (west, NORTH - N * 10, west + N * 10, NORTH)
    w, s, e, n = Transformer.from_crs(EPSG, 4326, always_xy=True).transform_bounds(*bounds)
    return AEFTileInfo(
        id=f"{tag}{year}-{col_off}",
        path=f"s3://bucket/{tag}{year}-{col_off}.tiff",
        year=year,
        bbox=(w, s, e, n),
        crs_epsg=EPSG,
        utm_zone="31N",
        utm_bounds=bounds,
    )


class FakeReader:
    """Builds a lazy zone mosaic from synthetic tiles, like open_tiles_by_zone."""

    collection = AEF

    def __init__(self, south_up=False):
        self.loads: list[tuple] = []
        self.opens = 0
        self.south_up = south_up

    async def open_tiles_by_zone(self, tiles, chunks=None, max_concurrency=None, **_):
        assert chunks == "native"
        self.opens += 1
        seen = len(self.loads)
        parts = []
        for t in tiles:
            west, _, _, north = t.utm_bounds
            src = CountingSource(t.year, west, north, self.loads, t.id, self.south_up)
            arr = da.from_array(src, chunks=(1, BLOCK, BLOCK), name=f"src-{t.id}")
            ys = north - 5 - 10 * np.arange(N)
            if self.south_up:
                ys = ys[::-1]
            arr = xr.DataArray(
                arr,
                dims=("band", "y", "x"),
                coords={
                    "band": [f"A{i:02d}" for i in range(BANDS)],
                    "y": ys,
                    "x": west + 5 + 10 * np.arange(N),
                },
            ).expand_dims(time=[dt.datetime(t.year, 1, 1)])
            parts.append(arr.to_dataset(name="embeddings"))
        ds = xr.combine_by_coords(parts, join="outer", fill_value=np.int8(-128))
        del self.loads[seen:]  # xarray peeks at one chunk while combining; not a real read
        return DataTree.from_dict({"/31N": ds})


def expected(year, row0, col0, size, cols_valid=range(2 * N)):
    out = np.full((BANDS, size, size), -128, dtype=np.int8)
    for b in range(BANDS):
        for i in range(size):
            for j in range(size):
                r, c = row0 + i, col0 + j
                if 0 <= r < N and c in cols_valid:
                    out[b, i, j] = value(year, b, r, c)
    return out


def point(col, row, fx=0.5, fy=0.5):
    return WEST + (col + fx) * 10, NORTH - (row + fy) * 10


def chip_at(tiles, x, y, reader, **kw):
    kw.setdefault("years", 2024)
    return asyncio.run(
        read_chip(tiles, x=x, y=y, crs=f"EPSG:{EPSG}", reader=reader, **kw)
    )


def test_chip_axes_even_and_odd_centring():
    # pixel centres at 5, 15, ...; the point 47 lies in the pixel centred 45
    xs, ys, tr = chip_axes(47, 47, 4, 5.0, 10.0, 155.0, -10.0)
    np.testing.assert_allclose(xs, [25, 35, 45, 55])  # centre pixel is index 2
    np.testing.assert_allclose(ys, [65, 55, 45, 35])  # north-up
    assert tr == Affine(10, 0, 20, 0, -10, 70)
    xs, _, _ = chip_axes(47, 47, 5, 5.0, 10.0, 155.0, -10.0)
    np.testing.assert_allclose(xs, [25, 35, 45, 55, 65])
    # a point on a pixel edge belongs to the pixel east of it
    xs, _, _ = chip_axes(50.0, 45, 1, 5.0, 10.0, 155.0, -10.0)
    assert xs[0] == 55


def test_even_chip_matches_expected_window():
    tile = make_tile(2024, 0)
    x, y = point(5, 5)
    chip = chip_at([tile], x, y, FakeReader(), size=4)
    assert isinstance(chip, Chip)
    assert chip.data.dtype == np.int8 and chip.data.shape == (1, BANDS, 4, 4)
    np.testing.assert_array_equal(chip.data[0], expected(2024, 3, 3, 4))
    assert chip.crs == f"EPSG:{EPSG}"
    assert chip.transform == Affine(10, 0, WEST + 30, 0, -10, NORTH - 30)
    assert chip.years == (2024,) and chip.band_names == ("A00", "A01")
    assert chip.tile_ids == (tile.id,)


def test_odd_chip_is_centred_and_geographic_input_agrees():
    tile = make_tile(2024, 0)
    x, y = point(8, 9)
    chip = chip_at([tile], x, y, FakeReader(), size=5)
    np.testing.assert_array_equal(chip.data[0], expected(2024, 7, 6, 5))
    lon, lat = Transformer.from_crs(EPSG, 4326, always_xy=True).transform(x, y)
    again = asyncio.run(read_chip([tile], x=lon, y=lat, size=5, years=2024, reader=FakeReader()))
    np.testing.assert_array_equal(again.data, chip.data)
    assert again.transform == chip.transform


@pytest.mark.parametrize("south_up", [False, True])
def test_chip_mosaics_across_tile_edge(south_up):
    tiles = [make_tile(2024, 0), make_tile(2024, N)]
    x, y = point(15, 4)
    chip = chip_at(tiles, x, y, FakeReader(south_up), size=6)
    assert chip.data.shape == (1, BANDS, 6, 6)
    np.testing.assert_array_equal(chip.data[0], expected(2024, 1, 12, 6))
    assert set(chip.tile_ids) == {t.id for t in tiles}


def test_missing_neighbour_is_nodata_and_unneeded_tile_not_opened():
    tiles = [make_tile(2024, 0), make_tile(2024, 3 * N)]  # the far tile is irrelevant
    x, y = point(15, 4)
    reader = FakeReader()
    chip = chip_at(tiles, x, y, reader, size=6)
    np.testing.assert_array_equal(chip.data[0], expected(2024, 1, 12, 6, cols_valid=range(N)))
    assert (chip.data[0, :, :, 4:] == -128).all()
    assert chip.tile_ids == (tiles[0].id,)
    assert {load[0] for load in reader.loads} == {tiles[0].id}


def test_multiple_years_stack_on_time():
    tiles = [make_tile(2023, 0), make_tile(2024, 0)]
    x, y = point(5, 5)
    chip = chip_at(tiles, x, y, FakeReader(), size=4, years=(2023, 2024))
    assert chip.years == (2023, 2024) and chip.data.shape[0] == 2
    for k, year in enumerate(chip.years):
        np.testing.assert_array_equal(chip.data[k], expected(year, 3, 3, 4))


def test_no_covering_tile_raises():
    with pytest.raises(ValueError, match="no tile covers"):
        asyncio.run(read_chip([make_tile(2024, 0)], x=0.0, y=0.0, size=4, years=2024))


def test_shared_blocks_are_computed_once_per_call():
    tiles = [make_tile(2024, 0)]
    pts = [point(5, 5), point(6, 6), point(4, 5)]  # all inside the first 8x8 block
    kw = {"size": 4, "years": 2024, "crs": f"EPSG:{EPSG}"}

    together = FakeReader()
    chips = asyncio.run(read_chips(pts, index=tiles, reader=together, **kw))
    assert together.opens == 1  # one open for the whole group
    assert len(together.loads) == BANDS  # one block per band, fetched once
    assert len(set(together.loads)) == len(together.loads)

    apart = FakeReader()
    for x, y in pts:
        asyncio.run(read_chip(tiles, x=x, y=y, reader=apart, **kw))
    assert len(apart.loads) == BANDS * len(pts)

    for chip, (x, y) in zip(chips, pts):
        col, row = int((x - WEST) // 10), int((NORTH - y) // 10)
        np.testing.assert_array_equal(chip.data[0], expected(2024, row - 2, col - 2, 4))


def test_read_chips_preserves_input_order_across_groups():
    tiles = [make_tile(2024, 0), make_tile(2024, 3 * N)]
    pts = [point(3 * N + 8, 10), point(6, 6), point(3 * N + 5, 8), point(9, 12)]
    reader = FakeReader()
    chips = asyncio.run(
        read_chips(pts, size=4, years=2024, index=tiles, crs=f"EPSG:{EPSG}", reader=reader)
    )
    assert reader.opens == 2  # two disjoint tile groups
    for chip, (x, _y) in zip(chips, pts):
        col = int((x - WEST) // 10)
        assert chip.transform.c == WEST + (col - 2) * 10
        assert chip.tile_ids == (f"t2024-{0 if col < N else 3 * N}",)


def test_read_chips_accepts_geodataframe():
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Point

    tiles = [make_tile(2024, 0)]
    x, y = point(5, 5)
    gdf = gpd.GeoDataFrame(geometry=[Point(x, y)], crs=f"EPSG:{EPSG}")
    (chip,) = asyncio.run(read_chips(gdf, size=4, years=2024, index=tiles, reader=FakeReader()))
    np.testing.assert_array_equal(chip.data[0], expected(2024, 3, 3, 4))


def test_year_selection_matches_index_rules():
    # read_chips(years=[2024]) used to crash: lists were unpacked as (start, end).
    from aef_loader.chips import _resolve_point

    tiles = [
        AEFTileInfo(id=str(y), path=f"s3://b/{y}.tif", year=y, bbox=(4.5, 58.0, 6.5, 59.5),
                    crs_epsg=32631, utm_zone="31N", utm_bounds=(600000, 6450000, 700000, 6550000))
        for y in (2022, 2023, 2024)
    ]
    for years, expected in [([2024], {2024}), ([2022, 2024], {2022, 2024}),
                            ((2022, 2023), {2022, 2023}), (2023, {2023}), ("2024", {2024})]:
        resolved = _resolve_point(5.6, 58.7, "EPSG:4326", tiles, years, 16)
        assert {t.year for t in resolved.tiles} == expected, years
