import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from aef_loader.cache import CACHE_FORMAT_VERSION, cache_path_for
from aef_loader.reader import VirtualTiffReader
from aef_loader.types import AEFTileInfo


@pytest.mark.asyncio
async def test_manifest_validation_offline(tmp_path, monkeypatch):
    tile = AEFTileInfo(
        id="test_tile",
        utm_zone="10N",
        path="s3://b/t.tif",
        crs_epsg=32610,
        year=2024,
        bbox=(0, 0, 1, 1)
    )

    mock_head = AsyncMock()
    mock_head.return_value = {"e_tag": '"etag-1"', "size": 100}
    
    mock_store_obj = MagicMock()
    mock_store_obj.head_async = mock_head

    reader = VirtualTiffReader(manifest_cache_dir=tmp_path)
    reader._get_store = MagicMock(return_value=mock_store_obj)
    reader._get_registry = MagicMock(return_value=MagicMock())
    
    mock_manifest = MagicMock()
    
    # We also need to mock VirtualTIFF and open_zarr and _get_geobox_from_dataset
    # so we don't actually do any real logic. Or we can just test `_combine_tiles_single_zone` up to the cache.
    # Actually, the simplest is to mock `process_tile_unlocked` internals except the caching block, 
    # but `reader.py` combines them tightly.
    # Let's mock `virtual_tiff.VirtualTIFF.__call__` and `xr.open_zarr`.

    def mock_parser_call(*args, **kwargs):
        return mock_manifest
    
    monkeypatch.setattr("aef_loader.reader.VirtualTIFF.__call__", mock_parser_call)
    
    # Mock open_zarr to raise an exception to abort early or just return an empty dataset?
    # If we return an empty dataset, it might fail in crop. 
    import xarray as xr
    def mock_open_zarr(*args, **kwargs):
        ds = xr.Dataset({"embeddings": (["y", "x"], [[1, 2], [3, 4]])})
        ds["embeddings"].attrs["model_pixel_scale"] = (10.0, 10.0, 0.0)
        ds["embeddings"].attrs["model_tiepoint"] = (0.0, 0.0, 0.0, 500000.0, 4600000.0, 0.0)
        return ds
        
    monkeypatch.setattr("aef_loader.reader.xr.open_zarr", MagicMock(side_effect=mock_open_zarr))
    
    # Mock jsonable serialization so we don't need a real manifest store
    def mock_jsonable(store, object_meta=None):
        return {
            "format_version": CACHE_FORMAT_VERSION,
            "object_meta": object_meta,
            "arrays": {}
        }
    monkeypatch.setattr("aef_loader.cache._manifest_to_jsonable", mock_jsonable)
    monkeypatch.setattr("aef_loader.cache._jsonable_to_manifest", MagicMock(return_value=mock_manifest))

    # Disable memory cache to ensure we hit the disk cache
    reader.memory_manifest_cache_size = 0
    
    # 1. Save records identity (ETag, size)
    await reader._combine_tiles_single_zone([tile])
    assert reader.stats["manifest_parses"] == 1
    
    path = cache_path_for(tmp_path, "s3://b/t.tif", 0)
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["format_version"] == CACHE_FORMAT_VERSION
    assert data["object_meta"]["e_tag"] == '"etag-1"'
    assert data["object_meta"]["size"] == 100
    
    # 2. "none" never calls head
    reader.manifest_validation = "none"
    mock_head.reset_mock()
    await reader._combine_tiles_single_zone([tile])
    assert reader.stats["disk_manifest_hits"] == 1
    mock_head.assert_not_called()
    
    # 3. "head" with matching ETag -> hit
    reader.manifest_validation = "head"
    mock_head.reset_mock()
    reader._stats["disk_manifest_hits"] = 0
    await reader._combine_tiles_single_zone([tile])
    assert reader.stats["disk_manifest_hits"] == 1
    assert "manifest_stale" not in reader.stats
    mock_head.assert_called_once()
    
    # 4. mismatched ETag -> miss + stats
    mock_head.return_value = {"e_tag": '"etag-2"', "size": 100}
    reader._stats["disk_manifest_hits"] = 0
    reader._stats["manifest_parses"] = 0
    await reader._combine_tiles_single_zone([tile])
    assert reader.stats["disk_manifest_hits"] == 0
    assert reader.stats["manifest_parses"] == 1
    assert reader.stats["manifest_stale"] == 1
    
    # 5. v1 entry ignored
    # Write a v1 entry
    data["format_version"] = 1
    path.write_text(json.dumps(data))
    
    reader._stats["disk_manifest_hits"] = 0
    reader._stats["manifest_parses"] = 0
    await reader._combine_tiles_single_zone([tile])
    assert reader.stats["disk_manifest_hits"] == 0
    assert reader.stats["manifest_parses"] == 1
