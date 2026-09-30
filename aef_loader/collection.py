"""
Dataset descriptions for the embedding collections this package reads.

Each supported collection (AlphaEarth Foundations, TESSERA) has its own band
naming, nodata convention, encoding, CRS per UTM zone and decoder. The reader and
``reproject_datatree`` used to hard-code AEF's values (``A00..A63``, ``-128``)
and guessed the encoding from the dtype. A ``Collection`` gathers those facts in
one place, and every embeddings variable records its encoding in the
``aef:encoding`` attribute, so downstream code can ask what the data *is*
instead of inferring it from the dtype.

See docs/DESIGN-2026-09-30.md, section 3.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import xarray as xr

from aef_loader.constants import AEF_NODATA_VALUE

Encoding = Literal["aef-sqrt", "codes-times-scales", "float"]

ENCODING_ATTR = "aef:encoding"
"""Attribute on an embeddings variable naming its encoding (see ``Encoding``)."""

RAW_ENCODINGS: frozenset[str] = frozenset({"aef-sqrt", "codes-times-scales"})
"""Encodings whose values are codes that interpolating resamplers would corrupt."""

_ZONE_RE = re.compile(r"^(\d{1,2})([NS])$", re.IGNORECASE)


def _parse_zone(zone: str) -> tuple[int, str]:
    match = _ZONE_RE.match(zone.strip())
    if not match or not 1 <= int(match.group(1)) <= 60:
        raise ValueError(f"not a UTM zone label: {zone!r} (expected e.g. '31N')")
    return int(match.group(1)), match.group(2).upper()


def _utm_crs(zone: str) -> str:
    """``"31N"`` -> ``"EPSG:32631"``, ``"31S"`` -> ``"EPSG:32731"``."""
    number, hemisphere = _parse_zone(zone)
    return f"EPSG:{(32600 if hemisphere == 'N' else 32700) + number}"


def _north_utm_crs(zone: str) -> str:
    """Northern-referenced UTM CRS for either hemisphere (TESSERA's convention)."""
    number, _ = _parse_zone(zone)
    return f"EPSG:{32600 + number}"


def _aef_band_names(n: int) -> list[str]:
    return [f"A{i:02d}" for i in range(n)]


def _aef_dequantize(ds: xr.Dataset) -> xr.Dataset:
    from aef_loader.utils import dequantize_aef

    emb = dequantize_aef(ds["embeddings"])
    emb.attrs[ENCODING_ATTR] = "float"
    return ds.assign(embeddings=emb)


def _tessera_dequantize(ds: xr.Dataset) -> xr.Dataset:
    scales = ds["scales"]
    emb = ds["embeddings"].astype("float32") * scales.where(np.isfinite(scales))
    emb.attrs.update(
        {ENCODING_ATTR: "float", "dequantized": True, "nodata": np.nan, "_FillValue": np.nan}
    )
    return ds.drop_vars("scales").assign(embeddings=emb)


@dataclass(frozen=True)
class Collection:
    """Static facts about one embedding collection.

    Attributes:
        name: Registry key, e.g. ``"aef-v1"``.
        encoding: Encoding of the raw ``embeddings`` variable as stored.
        nodata: Raw-encoding nodata sentinel, or None when validity comes from
            another variable (TESSERA's ``scales``).
        band_names: Maps a band count to band labels, or None to keep the
            store's own band coordinate.
        zone_crs: Maps a UTM zone label (``"31N"``) to that zone's CRS.
        dequantize: Turns a raw dataset into float32 ``embeddings``, dropping any
            helper variables it consumed.
        quality_vars: Optional per-pixel quality variables the collection offers.
    """

    name: str
    encoding: Encoding
    nodata: int | float | None
    band_names: Callable[[int], list[str]] | None
    zone_crs: Callable[[str], str]
    dequantize: Callable[[xr.Dataset], xr.Dataset]
    quality_vars: tuple[str, ...] = field(default=())

    def tag(self, ds: xr.Dataset) -> xr.Dataset:
        """Stamp the raw encoding (and nodata, if any) on ``ds.embeddings``."""
        attrs = {ENCODING_ATTR: self.encoding}
        if self.nodata is not None:
            attrs.update(nodata=self.nodata, _FillValue=self.nodata)
        return ds.assign(embeddings=ds["embeddings"].assign_attrs(attrs))


AEF = Collection(
    name="aef-v1",
    encoding="aef-sqrt",
    nodata=AEF_NODATA_VALUE,
    band_names=_aef_band_names,
    zone_crs=_utm_crs,
    dequantize=_aef_dequantize,
)

TESSERA = Collection(
    name="tessera-v1.1",
    encoding="codes-times-scales",
    nodata=None,
    band_names=None,
    zone_crs=_north_utm_crs,
    dequantize=_tessera_dequantize,
    quality_vars=(
        "s1_asc_obs_count",
        "s1_desc_obs_count",
        "s2_obs_count",
        "s1_asc_month_covered",
        "s1_desc_month_covered",
        "s2_month_covered",
    ),
)

_REGISTRY: dict[str, Collection] = {c.name: c for c in (AEF, TESSERA)}


def register_collection(collection: Collection, *, replace: bool = False) -> None:
    """Add a collection to the registry (``replace=True`` to overwrite a name)."""
    if collection.name in _REGISTRY and not replace:
        raise ValueError(f"collection {collection.name!r} is already registered")
    _REGISTRY[collection.name] = collection


def get_collection(name: str) -> Collection:
    """Look up a registered collection by name."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown collection {name!r}; registered: {sorted(_REGISTRY)}"
        ) from None


def list_collections() -> list[str]:
    """Names of all registered collections."""
    return sorted(_REGISTRY)


def encoding_of(da: xr.DataArray) -> str | None:
    """The variable's recorded encoding, or None if it was never tagged."""
    return da.attrs.get(ENCODING_ATTR)


def is_raw_encoded(ds: xr.Dataset) -> bool:
    """True if ``ds.embeddings`` holds codes that must not be interpolated.

    Uses the ``aef:encoding`` attribute when present. Untagged data (e.g. built
    by hand, or written by an older version) falls back to the embeddings
    variable's dtype only, never to other variables such as integer quality
    counts.
    """
    if "embeddings" not in ds.data_vars:
        return False
    emb = ds["embeddings"]
    encoding = encoding_of(emb)
    if encoding is not None:
        return encoding in RAW_ENCODINGS
    return bool(np.issubdtype(emb.dtype, np.integer))
