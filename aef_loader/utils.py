"""
Utility functions for AEF data processing.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, overload

import numpy as np
import xarray as xr
from odc.geo.xr import xr_reproject
from xarray import DataTree

from aef_loader.constants import (
    AEF_DEQUANT_DIVISOR,
    AEF_NODATA_VALUE,
)

if TYPE_CHECKING:
    from odc.geo.geobox import GeoBox


def _build_dequant_lut(
    divisor: float = AEF_DEQUANT_DIVISOR,
    nodata_value: int = AEF_NODATA_VALUE,
) -> np.ndarray:
    """256-entry int8->float32 dequantization lookup table indexed by ``raw + 128``.

    AEF stores embeddings as int8 in ``[-128, 127]``, so every possible value is
    one of 256 outcomes. Precomputing them once and gathering (``lut[raw + 128]``)
    is measurably faster than recomputing ``(v/127.5)**2 * sign(v)`` per pixel and
    fuses the ``-128 -> NaN`` nodata mapping into the same gather, avoiding a
    separate boolean mask and its temporaries. At ~4.3e9 elements per 8192^2 x 64
    tile this dominates the cost of ``dequantize_aef`` on the numpy path.
    """
    idx = np.arange(-128, 128, dtype=np.float32)
    lut = (idx / divisor) ** 2 * np.sign(idx)
    lut[nodata_value + 128] = np.nan  # -128 -> NaN, honoured by resamplers as nodata
    return lut


# Default LUT for the common (divisor=127.5, nodata=-128) case. A non-default
# call rebuilds a table on the fly, so correctness never depends on this cache.
_DEQUANT_LUT = _build_dequant_lut()
# Reordering maps the raw int8 bit pattern directly to its result. Indexing this
# with raw.view(uint8) avoids allocating an int16 array plus a shifted copy.
_DEQUANT_LUT_UINT8 = np.concatenate((_DEQUANT_LUT[128:], _DEQUANT_LUT[:128]))


def _dequantize_lut(
    raw: np.ndarray,
    divisor: float = AEF_DEQUANT_DIVISOR,
    nodata_value: int = AEF_NODATA_VALUE,
) -> np.ndarray:
    """Dequantize a raw int8/int16 array via the LUT; ``nodata_value`` -> NaN.

    Returns a fresh float32 array. ``raw`` may be int8 or a wider int type (the
    warped-read path yields int16 with the -128 sentinel preserved); values are
    shifted into ``[0, 255]`` to index the table.
    """
    lut = (
        _DEQUANT_LUT
        if divisor == AEF_DEQUANT_DIVISOR and nodata_value == AEF_NODATA_VALUE
        else _build_dequant_lut(divisor, nodata_value)
    )
    raw_array = np.asarray(raw)
    if raw_array.dtype == np.int8:
        uint8_lut = (
            _DEQUANT_LUT_UINT8
            if lut is _DEQUANT_LUT
            else np.concatenate((lut[128:], lut[:128]))
        )
        return uint8_lut[raw_array.view(np.uint8)]
    if raw_array.dtype.kind not in "iu":
        raise TypeError(
            f"dequantize_aef expects integer-quantized data, got {raw_array.dtype}; "
            "float/bool input looks already dequantized."
        )
    if raw_array.size and (raw_array.min() < -128 or raw_array.max() > 127):
        raise ValueError(
            "quantized values must be within [-128, 127]; got range "
            f"[{raw_array.min()}, {raw_array.max()}]"
        )
    return lut[raw_array.astype(np.int16) + 128]


def dequantize_aef(
    data: np.ndarray | xr.DataArray | xr.Dataset,
    divisor: float = AEF_DEQUANT_DIVISOR,
    nodata_value: int = AEF_NODATA_VALUE,
) -> np.ndarray | xr.DataArray | xr.Dataset:
    """
    Dequantize AEF embeddings from int8 to float32.

    AEF embeddings are stored as quantized int8 values [-127, 127].
    This function converts them back to float32 [-1, 1] for use in ML pipelines.

    The formula is: ((value / 127.5) ** 2) * sign(value)

    NoData values (-128) are automatically converted to NaN. For DataArray
    inputs, both ``nodata`` and ``_FillValue`` attrs are set to ``NaN`` on
    the output so that downstream tools (odc-geo, xarray) recognise the
    new fill value. All other existing attrs are preserved.

    Args:
        data: Quantized embedding data (int8)
        divisor: Dequantization divisor (default: 127.5)
        nodata_value: Value to treat as nodata (default: -128)

    Returns:
        Dequantized float32 data in range [-1, 1], with NaN for nodata

    Example:
        ```python
        import numpy as np
        quantized = np.array([127, -127, 0, -128], dtype=np.int8)
        dequantized = dequantize_aef(quantized)
        print(dequantized)  # [~1.0, ~-1.0, 0.0, nan]
        ```
    """
    if isinstance(data, xr.Dataset):
        return data.map(lambda x: dequantize_aef(x, divisor, nodata_value))

    if isinstance(data, xr.DataArray):
        array = data.data
        # Check the dtype up front: for dask input the per-block check in
        # _dequantize_lut would otherwise only fire at compute time.
        if np.dtype(array.dtype).kind not in "iu":
            raise TypeError(
                f"dequantize_aef expects integer-quantized data, got {array.dtype}; "
                "float/bool input looks already dequantized."
            )
        if hasattr(array, "map_blocks"):
            # One task per existing block, with no intermediate mask, cast,
            # division, square and sign arrays in the Dask graph.
            dequantized = array.map_blocks(
                _dequantize_lut,
                divisor=divisor,
                nodata_value=nodata_value,
                dtype=np.float32,
            )
        else:
            dequantized = _dequantize_lut(array, divisor, nodata_value)
        result = xr.DataArray(
            dequantized,
            dims=data.dims,
            coords=data.coords,
            attrs=data.attrs.copy(),
        )
        result.attrs["units"] = "embedding"
        result.attrs["dequantized"] = True
        return set_aef_nodata(result, nodata=np.nan)

    # Plain ndarray: gather through the precomputed LUT (see _dequantize_lut).
    # Bit-identical to the elementwise formula, ~1.3x faster, and it folds the
    # -128 -> NaN nodata step into the same gather (no extra mask/temporaries).
    return _dequantize_lut(data, divisor, nodata_value)


def quantize_aef(
    data: np.ndarray | xr.DataArray,
    divisor: float = AEF_DEQUANT_DIVISOR,
) -> np.ndarray | xr.DataArray:
    """
    Quantize float32 embeddings to int8 for storage.

    This is the inverse of dequantize_aef().
    Dequantization: ((v / 127.5) ** 2) * sign(v)
    Quantization (inverse): sign(v) * sqrt(|v|) * 127.5

    For DataArray inputs, both ``nodata`` and ``_FillValue`` attrs are set
    to ``-128`` (AEF_NODATA_VALUE) on the output so that downstream tools
    (odc-geo, xarray) recognise the int8 nodata sentinel. All other
    existing attrs are preserved.

    Args:
        data: Float32 embedding data in range [-1, 1]
        divisor: Quantization divisor (default: 127.5)

    Non-finite inputs (NaN and +/-inf) are treated as nodata and written as
    ``-128`` (AEF_NODATA_VALUE), so a dequantize/quantize round trip preserves
    nodata instead of turning it into a valid zero code.

    Returns:
        Quantized int8 data in range [-127, 127], with -128 for nodata
    """
    finite = np.isfinite(data)
    sign = np.sign(data)
    magnitude = np.sqrt(np.abs(data))
    quantized = np.round(sign * magnitude * divisor)

    # Clamp to valid range [-127, 127] BEFORE casting to int8
    # This prevents overflow (128 -> -128 in int8)
    quantized = np.clip(quantized, -127, 127)
    # Replace non-finite values before the cast: NaN -> int8 is undefined.
    if isinstance(data, xr.DataArray):
        quantized = xr.where(finite, quantized, AEF_NODATA_VALUE)
    else:
        quantized = np.where(finite, quantized, AEF_NODATA_VALUE)
    quantized = quantized.astype(np.int8)

    if isinstance(data, xr.DataArray):
        result = xr.DataArray(
            quantized,
            dims=data.dims,
            coords=data.coords,
            attrs=data.attrs.copy(),
        )
        result.attrs["quantized"] = True
        return set_aef_nodata(result, nodata=AEF_NODATA_VALUE)

    return quantized


def mask_nodata(
    data: np.ndarray | xr.DataArray,
    nodata_value: int = AEF_NODATA_VALUE,
) -> np.ndarray | xr.DataArray:
    """
    Mask NoData values (-128) in AEF embeddings.

    NoData pixels have -128 in all channels. This function replaces
    NoData values with NaN for proper handling in analysis.

    Args:
        data: AEF embedding data (int8)
        nodata_value: Value to mask (default: -128)

    Returns:
        Data with NoData values replaced by NaN
    """
    if isinstance(data, xr.DataArray):
        return data.where(data != nodata_value)
    return np.where(data == nodata_value, np.nan, data.astype(np.float32))


def int8_to_float32(
    data: np.ndarray | xr.DataArray | xr.Dataset,
    nodata_value: int = AEF_NODATA_VALUE,
) -> np.ndarray | xr.DataArray | xr.Dataset:
    """
    Cast int8 AEF embeddings to float32 without dequantization.

    Unlike dequantize_aef(), this performs a simple type cast: int8 values
    become their float32 equivalents (e.g. 64 -> 64.0, not 0.252).
    NoData values (-128) are replaced with NaN.

    For DataArray inputs, both ``nodata`` and ``_FillValue`` attrs are set
    to ``NaN`` on the output. All other existing attrs are preserved.

    Args:
        data: Quantized embedding data (int8)
        nodata_value: Value to treat as nodata (default: -128)

    Returns:
        Float32 data with raw int8 values preserved, NaN for nodata
    """
    if isinstance(data, xr.Dataset):
        return data.map(lambda x: int8_to_float32(x, nodata_value))

    nodata_mask = data == nodata_value

    if isinstance(data, xr.DataArray):
        result = data.astype(np.float32).where(~nodata_mask)
        return set_aef_nodata(result, nodata=np.nan)

    return np.where(nodata_mask, np.nan, data.astype(np.float32))


@overload
def set_aef_nodata(data: xr.DataArray, nodata: float = ...) -> xr.DataArray: ...


@overload
def set_aef_nodata(data: xr.Dataset, nodata: float = ...) -> xr.Dataset: ...


def set_aef_nodata(
    data: xr.DataArray | xr.Dataset,
    nodata: float = AEF_NODATA_VALUE,
) -> xr.DataArray | xr.Dataset:
    """Return a copy with the nodata and _FillValue attributes set explicitly.

    Args:
        data: Input DataArray or Dataset.
        nodata: The nodata sentinel to stamp. Use AEF_NODATA_VALUE (-128) for
                raw/quantized embeddings, or np.nan for dequantized float data.

    The input is not modified; a shallow copy (shared data, new attrs) is returned.
    """
    if isinstance(data, xr.Dataset):
        new_vars = {var: set_aef_nodata(data[var], nodata) for var in data.data_vars}
        return data.assign(new_vars)

    return data.assign_attrs({"nodata": nodata, "_FillValue": nodata})


def split_bands(ds: xr.Dataset, var: str = "embeddings") -> xr.Dataset:
    """
    Split a single multi-band DataArray into separate named variables (A00–A63).

    This is the inverse of the compact band representation used by
    VirtualTiffReader.open_tiles_by_zone(). Use this when downstream code
    expects individual A00–A63 data variables.

    Args:
        ds: Dataset containing a variable with a 'band' dimension
        var: Name of the variable to split (default: "embeddings")

    Returns:
        Dataset with one variable per band (A00, A01, ..., A63)
    """
    da = ds[var]
    split = da.to_dataset(dim="band")
    split.attrs = ds.attrs.copy()
    return split


def _is_quantized(tree: DataTree) -> bool:
    """True if any zone's data variables are an integer dtype (raw/quantized int8).

    Dequantized embeddings are float; raw AEF COGs opened by
    ``VirtualTiffReader.open_tiles_by_zone`` are int8. Interpolating resamplers
    must not run on the integer form (see ``reproject_datatree``).
    """
    for zone_name in tree.children:
        ds = tree[zone_name].ds
        if ds is None:
            continue
        for var in ds.data_vars:
            if np.issubdtype(ds[var].dtype, np.integer):
                return True
    return False


def _nodata_attr(da: xr.DataArray):
    """The variable's ``nodata`` / ``_FillValue`` attr, or None."""
    nodata = da.attrs.get("nodata", da.attrs.get("_FillValue"))
    return None if nodata is None else nodata


def _pixel_validity(ds: xr.Dataset, zone: str | None = None) -> xr.DataArray:
    """Boolean ``(time, y, x)`` mask of pixels that hold data, with no ``band`` dim.

    Raw TESSERA (has ``scales``): finite scales. Otherwise the ``embeddings``
    variable decides: integer data needs a ``nodata``/``_FillValue`` attr
    (``!= nodata`` in any band); float data uses non-null in any band (and, when
    a finite nodata attr is present, ``!= nodata``). Integer data with no nodata
    attr raises, since nothing distinguishes empty pixels from real codes.
    """
    if "scales" in ds.data_vars:
        return np.isfinite(ds["scales"])
    if "embeddings" not in ds.data_vars:
        raise ValueError(f"zone {zone!r} has no 'embeddings' or 'scales' variable")
    emb = ds["embeddings"]
    nodata = _nodata_attr(emb)
    if np.issubdtype(emb.dtype, np.integer):
        if nodata is None:
            raise ValueError(
                f"zone {zone!r}: integer variable 'embeddings' has no nodata/"
                "_FillValue attr, so pixel validity cannot be determined"
            )
        valid = emb != nodata
    else:
        valid = emb.notnull()
        if nodata is not None and not math.isnan(nodata):
            valid = valid & (emb != nodata)
    return valid.any("band") if "band" in valid.dims else valid


def _align_fill(da: xr.DataArray):
    """Fill value for a variable's gaps: its nodata for ints, NaN for floats."""
    if np.issubdtype(da.dtype, np.integer):
        nodata = _nodata_attr(da)
        return 0 if nodata is None else int(nodata)
    return np.nan


def _merge_zone_datasets(datasets: list[xr.Dataset]) -> xr.Dataset:
    """Merge zone datasets on a common grid, one source zone per pixel.

    ``take = ~valid_so_far & valid_next`` is computed once per zone from
    ``_pixel_validity`` and applied to every variable, so e.g. TESSERA codes and
    scales of one pixel never come from different zones. ``time`` (and ``band``)
    are outer-aligned first, filling each variable with its own nodata. Merging
    is elementwise ``xr.where`` (not ``combine_first``, which reindexes ``band``
    and fragments its chunks), so chunk structure is preserved.
    """
    names = [ds.attrs.get("source_zone") for ds in datasets]
    valids = [_pixel_validity(ds, n) for ds, n in zip(datasets, names)]

    for dim in ("time", "band"):
        if not all(dim in ds.dims for ds in datasets):
            continue
        indexes = [ds.indexes[dim] for ds in datasets]
        if all(idx.equals(indexes[0]) for idx in indexes[1:]):
            continue
        union = indexes[0]
        for idx in indexes[1:]:
            union = union.union(idx, sort=None)
        aligned = []
        for ds in datasets:
            fills = {var: _align_fill(ds[var]) for var in ds.data_vars}
            attrs = {var: ds[var].attrs for var in ds.data_vars}
            new = ds.reindex({dim: union}, fill_value=fills)
            for var, var_attrs in attrs.items():
                new[var].attrs = var_attrs
            aligned.append(new)
        datasets = aligned
        if dim == "time":
            valids = [v.reindex(time=union, fill_value=False) for v in valids]

    combined = datasets[0]
    valid_combined = valids[0]
    for ds, valid_next in zip(datasets[1:], valids[1:]):
        take = ~valid_combined & valid_next
        for var in ds.data_vars:
            if var not in combined.data_vars:
                # Only this zone has the variable: earlier-zone pixels get nodata.
                fill = _align_fill(ds[var])
                base = ds[var].where(~valid_combined, fill).astype(ds[var].dtype)
                base.attrs = ds[var].attrs
                combined[var] = base
                continue
            attrs = combined[var].attrs
            dims = combined[var].dims
            merged = xr.where(take, ds[var], combined[var], keep_attrs=True)
            # xr.where broadcasts ``take`` first; restore the original dim order.
            combined[var] = merged.transpose(*dims)
            combined[var].attrs = attrs
        valid_combined = valid_combined | valid_next
    return combined


def reproject_datatree(
    tree: DataTree,
    target_geobox: GeoBox,
    resampling: str = "nearest",
    dst_nodata: float | None = None,
    allow_lossy_resampling: bool = False,
) -> xr.Dataset:
    """
    Reproject all zones in a DataTree to a common target GeoBox.

    This function takes a DataTree with multiple UTM zones and reprojects each
    zone's dataset to a common coordinate system defined by the target GeoBox.
    The reprojected datasets are then combined into a single dataset.

    The reprojection is lazy - it builds a dask computation graph that only
    executes when .compute() is called. Chunks are loaded and reprojected
    on-demand.

    For combining zones, each pixel is taken from exactly one zone for all of its
    variables (see ``_pixel_validity``):
    - Uses values from earlier zones where the pixel is valid
    - Fills invalid pixels with values from subsequent zones
    - In true overlapping regions (both have valid data), earlier zones take precedence
    - Zones with different years (or bands) are outer-aligned first

    Since overlapping regions contain reprojections of the same underlying data,
    values should be identical regardless of which zone they come from.

    Args:
        tree: DataTree with zone datasets as children (from open_tiles_by_zone)
        target_geobox: Target GeoBox defining the output CRS, resolution, and extent.
                       Can be created with GeoBox.from_bbox() or from an existing dataset.
        resampling: Resampling method - "nearest", "bilinear", "cubic", etc.
                    Default is "nearest" which preserves original int8 values.
        dst_nodata: Nodata value for the output. When None (default), xr_reproject
                    reads the value from the source DataArray's nodata/_FillValue attrs.
                    When set, the value is passed to xr_reproject and both ``nodata``
                    and ``_FillValue`` attrs are stamped on each output data variable
                    after reprojection and after the zone merge.

    Returns:
        Combined xr.Dataset with all zones reprojected to the target GeoBox.
        Data variables remain as dask arrays until .compute() is called.

    Example:
        ```python
        from odc.geo.geobox import GeoBox

        # Create target geobox (e.g., 100m resolution in EPSG:4326)
        target = GeoBox.from_bbox(
            bbox=(-122.5, 37.5, -121.5, 38.5),
            crs="EPSG:4326",
            resolution=0.001,  # ~100m at this latitude
        )

        # Reproject all zones to target
        combined = reproject_datatree(tree, target)
        result = combined.compute()  # triggers actual reprojection
        ```
    """
    # Guard: interpolating resamplers corrupt raw int8 embeddings. They blend the
    # -128 nodata sentinel into neighbouring valid pixels AND interpolate along
    # AEF's nonlinear quantization curve (a weighted mean of quantized codes does
    # not dequantize to the mean of the underlying values). Only "nearest" (which
    # copies source codes verbatim) is exact on quantized data. To use bilinear/
    # cubic/etc., dequantize to float first (dequantize_aef, -128 -> NaN) so the
    # resampler operates on real embedding values with NaN-aware nodata.
    if resampling != "nearest" and not allow_lossy_resampling and _is_quantized(tree):
        raise ValueError(
            f"resampling={resampling!r} would corrupt raw int8 (quantized) AEF "
            "embeddings: it interpolates across the -128 nodata sentinel and the "
            "nonlinear quantization curve. Dequantize first (dequantize_aef, "
            "which maps -128 -> NaN) and reproject the float data, or pass "
            "allow_lossy_resampling=True to override. 'nearest' is always safe."
        )

    reproject_kwargs: dict = {"resampling": resampling}
    if dst_nodata is not None:
        reproject_kwargs["dst_nodata"] = dst_nodata

    reprojected_datasets: list[xr.Dataset] = []

    for zone_name in tree.children:
        zone_ds = tree[zone_name].ds

        # Skip empty datasets
        if zone_ds is None or len(zone_ds.data_vars) == 0:
            continue

        # Reproject to target geobox (lazy operation with dask)
        reprojected = xr_reproject(zone_ds, target_geobox, **reproject_kwargs)

        if dst_nodata is not None:
            reprojected = set_aef_nodata(reprojected, nodata=dst_nodata)

        # Add source zone as attribute
        reprojected.attrs["source_zone"] = zone_name
        reprojected_datasets.append(reprojected)

    if len(reprojected_datasets) == 0:
        raise ValueError("No datasets to reproject")

    if len(reprojected_datasets) == 1:
        return reprojected_datasets[0]

    # Merge each pixel from exactly one zone (see _merge_zone_datasets).
    combined = _merge_zone_datasets(reprojected_datasets)

    # Re-stamp nodata on the final merged dataset to ensure consistency.
    if dst_nodata is not None:
        combined = set_aef_nodata(combined, nodata=dst_nodata)
    combined.attrs["source_zones"] = [
        tree[z].ds.attrs.get("utm_zone", z) for z in tree.children
    ]
    combined.attrs["target_crs"] = str(target_geobox.crs)

    return combined


def aoi_geobox(
    bbox: tuple[float, float, float, float],
    crs: str,
    resolution: float,
    bbox_crs: str | None = None,
    snap: bool = True,
) -> GeoBox:
    """Build a target ``GeoBox`` for an AOI, snapped to a global pixel lattice.

    ``GeoBox.from_bbox`` anchors the grid at the AOI's own corner, so two AOIs
    reprojected independently (e.g. adjacent tiles, or the same area at different
    times) land on *different* pixel grids and cannot be mosaicked without
    resampling. Snapping the origin to integer multiples of ``resolution``
    (anchored at 0, 0) makes every AOI at a given resolution/CRS share one grid,
    so outputs align exactly and can be merged losslessly. This is the odc-geo
    analogue of the lattice snap ``prep_aef_tiles.py`` applies to warped tiles.

    Args:
        bbox: AOI bounds ``(minx, miny, maxx, maxy)``.
        crs: Target CRS for the GeoBox (e.g. ``"EPSG:32633"``).
        resolution: Pixel size in target CRS units (metres for UTM).
        bbox_crs: CRS of ``bbox`` if different from ``crs``; the bbox is
            reprojected (densified) to ``crs`` first. Defaults to ``crs``.
        snap: When True (default), snap the origin to the ``resolution`` lattice
            and grow the extent outward to fully cover ``bbox``. When False,
            behaves like ``GeoBox.from_bbox`` (AOI-anchored grid).

    Returns:
        A north-up ``GeoBox`` in ``crs`` at ``resolution`` covering ``bbox``.
    """
    from odc.geo.geobox import GeoBox

    minx, miny, maxx, maxy = bbox
    if bbox_crs is not None and str(bbox_crs) != str(crs):
        from pyproj import Transformer

        transformer = Transformer.from_crs(bbox_crs, crs, always_xy=True)
        minx, miny, maxx, maxy = transformer.transform_bounds(
            minx, miny, maxx, maxy, densify_pts=21
        )

    if not snap:
        return GeoBox.from_bbox(
            (minx, miny, maxx, maxy), crs=crs, resolution=resolution
        )

    # Snap outward to the global lattice: floor the min edges, ceil the max edges.
    snapped_minx = math.floor(minx / resolution) * resolution
    snapped_miny = math.floor(miny / resolution) * resolution
    snapped_maxx = math.ceil(maxx / resolution) * resolution
    snapped_maxy = math.ceil(maxy / resolution) * resolution
    return GeoBox.from_bbox(
        (snapped_minx, snapped_miny, snapped_maxx, snapped_maxy),
        crs=crs,
        resolution=resolution,
    )
