"""Tests for the TESSERA loader (offline helpers + one live smoke test)."""

import numpy as np
import pytest

from aef_loader.tessera import _candidate_zones, open_tessera


def test_candidate_zones_single_and_multi():
    assert _candidate_zones((5.0, 58.0, 5.5, 59.0)) == [31]
    assert _candidate_zones((5.0, 58.0, 7.0, 59.0)) == [31, 32]
    assert _candidate_zones((27.6, -28.0, 27.7, -27.9)) == [35]


@pytest.mark.slow
def test_open_tessera_live_small_chip():
    bbox = (648000, 6526000 - 640, 648000 + 640, 6526000)
    tree = open_tessera(bbox, "EPSG:32631", years=2024)
    ds = tree["31N"].ds
    assert ds.embeddings.dtype == np.int8
    assert dict(ds.sizes) == {"time": 1, "band": 128, "y": 64, "x": 64}
    deq = open_tessera(bbox, "EPSG:32631", years=2024, dequantize=True)["31N"].ds
    assert deq.embeddings.dtype == np.float32
    assert "scales" not in deq
