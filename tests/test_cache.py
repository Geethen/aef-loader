"""Offline tests for the manifest cache writer."""

import json
import os
import threading

import aef_loader.cache as cache


def _patch_payload(monkeypatch, payload):
    monkeypatch.setattr(cache, "_manifest_to_jsonable", lambda store, object_meta=None: payload)


def test_concurrent_saves_of_same_key_use_unique_temp_files(monkeypatch, tmp_path):
    payload = {"format_version": cache.CACHE_FORMAT_VERSION, "arrays": {}}
    _patch_payload(monkeypatch, payload)
    sources = []
    real_replace = os.replace

    def spy_replace(src, dst):
        sources.append(str(src))
        return real_replace(src, dst)

    monkeypatch.setattr(cache.os, "replace", spy_replace)

    threads = [
        threading.Thread(
            target=cache.save_manifest, args=(tmp_path, "s3://b/t.tif", 0, None)
        )
        for _ in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(sources) == 8 and len(set(sources)) == 8
    path = cache.cache_path_for(tmp_path, "s3://b/t.tif", 0)
    assert json.loads(path.read_text()) == payload
    assert [p.name for p in tmp_path.iterdir()] == [path.name]  # no .tmp left


def test_failed_save_leaves_no_temp_file(monkeypatch, tmp_path):
    _patch_payload(monkeypatch, {"bad": object()})  # not JSON-serialisable
    cache.save_manifest(tmp_path, "s3://b/t.tif", 0, None)  # best-effort: no raise
    assert list(tmp_path.iterdir()) == []


def test_failed_replace_cleans_up_temp_file(monkeypatch, tmp_path):
    _patch_payload(monkeypatch, {"format_version": cache.CACHE_FORMAT_VERSION})

    def boom(src, dst):
        raise OSError("disk says no")

    monkeypatch.setattr(cache.os, "replace", boom)
    cache.save_manifest(tmp_path, "s3://b/t.tif", 0, None)
    assert list(tmp_path.iterdir()) == []
