"""Tests for the TESSERA loader (offline helpers + one live smoke test)."""

import warnings

import numpy as np
import pytest
import xarray as xr

from aef_loader.tessera import (
    _candidate_zones,
    _normalise_years,
    _select_years,
    open_tessera,
)


def test_candidate_zones_single_and_multi():
    assert _candidate_zones((5.0, 58.0, 5.5, 59.0)) == [31]
    assert _candidate_zones((5.0, 58.0, 7.0, 59.0)) == [31, 32]
    assert _candidate_zones((27.6, -28.0, 27.7, -27.9)) == [35]


def test_normalise_years_forms_and_generator():
    assert _normalise_years(2024) == [2024]
    assert _normalise_years((2020, 2023)) == [2020, 2021, 2022, 2023]
    assert _normalise_years([2023, 2021, 2023]) == [2021, 2023]
    gen = iter([2024, 2022])
    sel = _normalise_years(gen)
    assert sel == [2022, 2024]
    # A normalised list can be reused for every zone; the generator could not.
    assert sel == [2022, 2024]


def _years_ds(years):
    return xr.Dataset({"v": ("time", list(range(len(years))))}, coords={"time": years})


def test_select_years_partial_warns_and_subsets():
    ds = _years_ds([2020, 2021, 2022])
    with pytest.warns(UserWarning, match=r"\[2025\]"):
        out, _ = _select_years(ds, [2021, 2025], "utm31")
    assert out["time"].values.tolist() == [2021]


def test_select_years_all_present_is_silent():
    ds = _years_ds([2020, 2021])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out, _ = _select_years(ds, [2020, 2021], "utm31")
    assert out.sizes["time"] == 2


def test_select_years_none_present_skips_zone_and_reports_available():
    # A zone without the years is skipped, not fatal: coverage can differ by
    # zone, and open_tessera raises only when no zone has any requested year.
    ds = _years_ds([2020, 2021])
    out, available = _select_years(ds, [2030], "utm31")
    assert out is None
    assert available == [2020, 2021]


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
