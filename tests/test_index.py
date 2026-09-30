"""Offline tests for AEFIndex: search semantics and the ranged column download."""

import asyncio
import io

import geopandas as gpd
import numpy as np
import obstore as obs
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import shapely
from obstore.store import MemoryStore
from shapely.geometry import Polygon

import aef_loader.index as index_module
from aef_loader import AEFIndex, DataSource
from aef_loader.index import _ObstoreRangeFile


def _records(n=100):
    rows = []
    geoms = []
    for i in range(n):
        x, y = i % 10, i // 10
        rows.append(
            {
                "fid": i,
                "crs": "EPSG:32632",
                "path": f"s3://bucket/{i}.tif",
                "year": 2020 + i % 3,
                "utm_zone": "32N",
                "utm_west": x * 100.0,
                "utm_south": y * 100.0,
                "utm_east": (x + 1) * 100.0,
                "utm_north": (y + 1) * 100.0,
                "wgs84_west": float(x),
                "wgs84_south": float(y),
                "wgs84_east": x + 1.0,
                "wgs84_north": y + 1.0,
            }
        )
        # Lower-left triangle: its bbox is the full cell but the upper-right
        # corner of the cell is not covered.
        geoms.append(Polygon([(x, y), (x + 1, y), (x, y + 1)]))
    return pd.DataFrame(rows), geoms


def test_search_is_bbox_superset_and_exact_matches_true_intersects():
    df, geoms = _records()
    index = AEFIndex(source=DataSource.SOURCE_COOP)
    index._gdf = gpd.GeoDataFrame(df, geometry=geoms, crs="EPSG:4326")
    aoi = (2.8, 2.8, 3.0, 3.0)  # in the cell (2, 2) upper-right corner
    truth = [
        str(i) for i, g in zip(df.fid, geoms) if g.intersects(shapely.box(*aoi))
    ]
    coarse = [t.id for t in index.search(bbox=aoi)]
    exact = [t.id for t in index.search(bbox=aoi, exact=True)]
    assert set(truth) <= set(coarse)
    assert coarse != truth  # the corner cell is a bbox-only candidate
    assert exact == truth
    assert coarse == sorted(coarse, key=int)  # file row order


def test_search_limit_validation_and_zero():
    df, _ = _records()
    index = AEFIndex(source=DataSource.SOURCE_COOP)
    index._df = df
    assert index.search(limit=0) == []
    assert len(index.search(limit=5)) == 5
    assert asyncio.run(index.query(bbox=(0, 0, 3, 3), limit=2))[0].id == "0"
    with pytest.raises(ValueError, match="limit"):
        index.search(limit=-1)


def test_default_cache_dir_is_not_tmp():
    index = AEFIndex(source=DataSource.SOURCE_COOP)
    assert "aef-loader" in str(index.cache_dir)
    assert index._cache_filename == "aef_index_source_coop.v2.parquet"


def _remote_parquet():
    """Two-row-group parquet with a large incompressible geometry column first."""
    df, geoms = _records(100)
    rng = np.random.default_rng(0)
    blobs = [rng.bytes(2000) for _ in range(len(df))]
    pad = [rng.bytes(2000) for _ in range(len(df))]  # keeps the footer tail read off geom
    table = pa.Table.from_pandas(df).add_column(0, "geom", pa.array(blobs))
    table = table.append_column("pad", pa.array(pad))
    sink = io.BytesIO()
    pq.write_table(table, sink, row_group_size=50, compression="none")
    return sink.getvalue(), df


def _geom_ranges(data: bytes) -> list[tuple[int, int]]:
    meta = pq.ParquetFile(io.BytesIO(data)).metadata
    ranges = []
    for rg in range(meta.num_row_groups):
        for col in range(meta.num_columns):
            chunk = meta.row_group(rg).column(col)
            if chunk.path_in_schema == "geom":
                start = chunk.data_page_offset
                if chunk.has_dictionary_page:
                    start = min(start, chunk.dictionary_page_offset)
                ranges.append((start, start + chunk.total_compressed_size))
    return ranges


@pytest.fixture
def remote(monkeypatch, tmp_path):
    data, df = _remote_parquet()
    store = MemoryStore()
    obs.put(store, "index.parquet", data)
    requested: list[tuple[int, int]] = []
    real_get_range = obs.get_range

    def recording(store_, path, *, start, end=None, length=None):
        requested.append((start, end))
        return real_get_range(store_, path, start=start, end=end, length=length)

    monkeypatch.setattr(index_module.obs, "get_range", recording)
    monkeypatch.setattr(AEFIndex, "_make_store", lambda self: store)
    monkeypatch.setattr(AEFIndex, "_index_blob", "index.parquet")
    index = AEFIndex(source=DataSource.SOURCE_COOP, cache_dir=tmp_path)
    return index, requested, _geom_ranges(data), df


def _overlaps(requested, geom_ranges):
    return any(
        s < ge and e > gs for s, e in requested for gs, ge in geom_ranges
    )


def test_range_file_reads_only_requested_bytes():
    store = MemoryStore()
    obs.put(store, "f", bytes(range(100)))
    f = _ObstoreRangeFile(store, "f")
    assert f.size == 100
    f.seek(10)
    assert f.read(5) == bytes(range(10, 15))
    f.seek(-3, io.SEEK_END)
    assert f.read() == bytes([97, 98, 99])


def test_download_never_requests_geom_bytes_unless_exact(remote):
    index, requested, geom_ranges, df = remote
    assert geom_ranges  # sanity: the geom column exists and has byte ranges

    path = asyncio.run(index.download())
    assert path.name == "aef_index_source_coop.v2.parquet"
    assert not list(path.parent.glob("*.tmp"))
    assert "geom" not in pq.read_schema(path).names
    assert requested and not _overlaps(requested, geom_ranges)

    index.load()
    tiles = index.search(bbox=(2.5, 2.5, 3.5, 3.5), years=2020)
    assert {t.id for t in tiles} <= {str(v) for v in df.fid}
    assert not _overlaps(requested, geom_ranges)

    # exact=True fetches the geometry column (from the store) exactly once.
    del requested[:]
    index._geoms = None
    with pytest.raises(shapely.errors.GEOSException):  # random bytes: not WKB
        index.search(bbox=(2.5, 2.5, 3.5, 3.5), exact=True)
    assert _overlaps(requested, geom_ranges)


def test_force_clears_loaded_table(remote):
    index, _, _, _ = remote
    asyncio.run(index.download())
    index.load()
    assert index._df is not None
    asyncio.run(index.download(force=True))
    assert index._df is None


def test_exact_fetches_geom_column_once_and_caches(monkeypatch, tmp_path):
    df, geoms = _records()
    table = pa.Table.from_pandas(df).append_column(
        "geom", pa.array(list(shapely.to_wkb(np.array(geoms, dtype=object))))
    )
    sink = io.BytesIO()
    pq.write_table(table, sink)
    store = MemoryStore()
    obs.put(store, "index.parquet", sink.getvalue())
    monkeypatch.setattr(AEFIndex, "_make_store", lambda self: store)
    monkeypatch.setattr(AEFIndex, "_index_blob", "index.parquet")
    index = AEFIndex(source=DataSource.SOURCE_COOP, cache_dir=tmp_path)
    asyncio.run(index.download())
    index.load()

    aoi = (2.8, 2.8, 3.0, 3.0)
    truth = [str(i) for i, g in zip(df.fid, geoms) if g.intersects(shapely.box(*aoi))]
    assert [t.id for t in index.search(bbox=aoi, exact=True)] == truth
    cache = tmp_path / "aef_index_source_coop.v2.geom.parquet"
    assert cache.exists()

    def boom(*args, **kwargs):
        raise AssertionError("geometry should come from the local cache")

    monkeypatch.setattr(index_module.obs, "get_range", boom)
    fresh = AEFIndex(source=DataSource.SOURCE_COOP, cache_dir=tmp_path)
    fresh.load()
    assert [t.id for t in fresh.search(bbox=aoi, exact=True)] == truth


@pytest.mark.parametrize(
    "text,code", [("EPSG:32633", 32633), ("epsg:32633", 32633), ("32633", 32633)]
)
def test_parse_epsg_accepts_common_forms(text, code):
    assert index_module._parse_epsg(text) == code


@pytest.mark.parametrize("text", ["WGS84", "EPSG:", "utm33", ""])
def test_parse_epsg_rejects_unknown(text):
    with pytest.raises(ValueError, match="CRS"):
        index_module._parse_epsg(text)


def test_search_raises_on_unparseable_crs():
    df, _ = _records(3)
    df["crs"] = "WGS84"
    index = AEFIndex(source=DataSource.SOURCE_COOP)
    index._df = df
    with pytest.raises(ValueError, match="CRS"):
        index.search()


def test_default_source_is_source_coop():
    assert AEFIndex().source is DataSource.SOURCE_COOP


@pytest.mark.parametrize(
    "value,expected",
    [
        ("source_coop", DataSource.SOURCE_COOP),
        ("SOURCE_COOP", DataSource.SOURCE_COOP),
        ("gcs", DataSource.GCS),
        ("GCS", DataSource.GCS),
        (DataSource.GCS, DataSource.GCS),
    ],
)
def test_source_strings_are_normalised(value, expected):
    assert AEFIndex(source=value).source is expected


def test_unknown_source_string_raises():
    with pytest.raises(ValueError):
        AEFIndex(source="s3")


def test_cache_dir_accepts_str_and_path(tmp_path):
    assert AEFIndex(cache_dir=str(tmp_path)).cache_dir == tmp_path
    assert AEFIndex(cache_dir=tmp_path).cache_dir == tmp_path


def test_exact_without_shapely_raises_helpful_importerror(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "shapely" or name.startswith("shapely."):
            raise ImportError("no shapely")
        return real_import(name, *args, **kwargs)

    df, _ = _records(5)
    index = AEFIndex(source=DataSource.SOURCE_COOP)
    index._df = df
    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"aef-loader-plus\[exact\]"):
        index.search(bbox=(0, 0, 3, 3), exact=True)


def test_version_comes_from_metadata_with_fallback(monkeypatch):
    import importlib
    import importlib.metadata as md

    import aef_loader

    assert aef_loader.__version__ == md.version("aef-loader-plus")

    def missing(name):
        raise md.PackageNotFoundError(name)

    monkeypatch.setattr(md, "version", missing)
    try:
        assert importlib.reload(aef_loader).__version__ == "0+unknown"
    finally:
        monkeypatch.undo()
        importlib.reload(aef_loader)

def test_search_years_semantics():
    df, _ = _records(5)
    # The records have years 2020, 2021, 2022, 2020, 2021.
    index = AEFIndex(source=DataSource.SOURCE_COOP)
    index._df = df
    
    # int
    assert len(index.search(years=2021)) == 2
    # string
    assert len(index.search(years="2021")) == 2
    assert len(index.search(years="2021-05-10")) == 2
    # tuple
    assert len(index.search(years=(2020, 2021))) == 4
    assert len(index.search(years=("2020", "2021-12"))) == 4
    
    with pytest.raises(ValueError, match="invalid year string"):
        index.search(years="abcd")
    with pytest.raises(ValueError, match="invalid year string"):
        index.search(years=("2020", "xyz"))
    with pytest.raises(ValueError, match="must have length 2"):
        index.search(years=(2020,))
    with pytest.raises(ValueError, match="ends before it starts"):
        index.search(years=(2022, 2021))
    with pytest.raises(ValueError, match="invalid years type"):
        index.search(years=[2020, 2021])
