"""Time downloading one AEF embedding chip three ways.

Methods (each run in a fresh Python process, so nothing is shared between runs
except the on-disk Source Cooperative index parquet and, for ``plus_warm``, the
on-disk manifest cache):

* ``geedim``    Earth Engine ``GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`` via geedim.
* ``upstream``  Source Cooperative COGs via PyPI ``aef-loader`` (jakenotjay).
* ``plus``      Source Cooperative COGs via this fork, cold manifest cache.
* ``plus_warm`` As ``plus``, but with the manifest cache populated beforehand.
* ``plus_warm_allbands`` As ``plus_warm`` but ``chunks="all-bands"`` (one Dask
  task per 1024 px block holding all 64 bands, like upstream) instead of
  ``"native"`` (one task per band per block, 64x more requests and tasks).

All methods fetch the same chip: ``size`` by ``size`` native 10 m pixels, all 64
bands, one year, on the native UTM grid (no resampling). The Source Cooperative
methods return the stored int8 codes; Earth Engine serves dequantized floats, so
geedim is requested as float32 (4x the payload of int8).

Usage (driver, runs every method/size/repeat and writes JSON results):

    python benchmarks/compare_chip_download.py --sizes 256 512 1024 2048 4096 --out results.json

Set CHIP_SITE=stavanger to reproduce the first (256/1024 px, Norway) run.

The driver calls ``--worker`` subprocesses; ``plus`` workers get this package
on PYTHONPATH, the other workers must see the PyPI ``aef-loader``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

EE_PROJECT = os.environ.get("EE_PROJECT", "")
EE_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
YEAR = 2024
# Upper-left chip corners on the native 10 m lattice. Each lies inside a single
# AEF tile, off the 1024-pixel COG block grid (a 256 px chip sits in one block).
SITES = {
    # Jaeren/Klepp, S of Stavanger, Norway; tile 31N/xbishs9u127yarggl
    # (UTM 581920-663840 E, 6471680-6553600 N). Fits chips up to 1584 px.
    "stavanger": ("EPSG:32631", 648_000.0, 6_526_000.0),
    # Free State farmland, South Africa; tile 35S/xs4n5bjzz10rqe9hk
    # (UTM 418080-500000 E, 6805120-6887040 N). All land; fits up to 6892 px.
    "freestate": ("EPSG:32735", 431_000.0, 6_875_000.0),
}
SITE = os.environ.get("CHIP_SITE", "freestate")
CRS, X0, Y0 = SITES[SITE]
RES = 10.0


def chip_bounds(size: int) -> tuple[float, float, float, float]:
    return X0, Y0 - size * RES, X0 + size * RES, Y0


def checksum(values: np.ndarray) -> dict:
    # Band by band, so a 4096 px float32 chip does not need an 8 GB float64 copy.
    finite, total, count = 0, 0.0, 0
    for band in values:
        ok = np.isfinite(band) if band.dtype.kind == "f" else np.ones(band.shape, bool)
        finite += int(ok.sum())
        total += float(band[ok].sum(dtype=np.float64))
        count += band.size
    return {
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "nbytes": int(values.nbytes),
        "finite_fraction": finite / count,
        "mean": total / max(finite, 1),
    }


# --------------------------------------------------------------------------- #
# Workers                                                                      #
# --------------------------------------------------------------------------- #


def run_geedim(size: int) -> dict:
    t0 = perf_counter()
    import ee
    import geedim  # noqa: F401  (registers the .gd accessor)

    ee.Initialize(
        project=EE_PROJECT or None, opt_url="https://earthengine-highvolume.googleapis.com"
    )
    t_init = perf_counter() - t0

    minx, miny, maxx, maxy = chip_bounds(size)
    image = (
        ee.ImageCollection(EE_COLLECTION)
        .filterDate(f"{YEAR}-01-01", f"{YEAR + 1}-01-01")
        .filterBounds(ee.Geometry.Point([(minx + maxx) / 2, (miny + maxy) / 2], CRS))
        .first()
    )
    t1 = perf_counter()
    values = image.gd.prepareForExport(
        crs=CRS,
        crs_transform=(RES, 0, minx, 0, -RES, maxy),
        shape=(size, size),
        dtype="float32",
    ).gd.toNumPy()
    t_fetch = perf_counter() - t1
    # geedim returns (row, col, band); make it (band, row, col) like the COGs.
    values = np.moveaxis(np.asarray(values), -1, 0)
    return {"init_s": t_init, "fetch_s": t_fetch, "values": values}


async def run_aef(size: int, *, plus: bool, manifest_dir: str | None,
                  chunks: str = "native") -> dict:
    t0 = perf_counter()
    import aef_loader
    from aef_loader import AEFIndex, DataSource, VirtualTiffReader

    is_plus = hasattr(aef_loader, "aoi_geobox")
    assert is_plus == plus, f"wrong aef_loader imported: {aef_loader.__file__}"

    index_dir = Path(os.environ["AEF_INDEX_DIR"])
    index = AEFIndex(source=DataSource.SOURCE_COOP, cache_dir=index_dir)
    await index.download()  # already cached on disk by the driver
    index.load()
    minx, miny, maxx, maxy = chip_bounds(size)
    tiles = await index.query(bbox=(minx, miny, maxx, maxy), years=YEAR, bbox_crs=CRS)
    t_init = perf_counter() - t0

    t1 = perf_counter()
    if plus:
        reader_ctx = VirtualTiffReader(manifest_cache_dir=manifest_dir)
    else:
        reader_ctx = VirtualTiffReader()
    async with reader_ctx as reader:
        if plus:
            # The reader's bbox crop is exact (selects exactly the covered cells),
            # so no shrink is needed.
            inner = (minx, miny, maxx, maxy)
            tree = await reader.open_tiles_by_zone(
                tiles, chunks=chunks, bbox=inner, bbox_crs=CRS
            )
        else:
            tree = await reader.open_tiles_by_zone(tiles)
        t_open = perf_counter() - t1
        (zone,) = tree.children
        da = tree[zone].ds["embeddings"].isel(time=0)
        half = RES / 2
        y = da["y"].values
        yslice = slice(maxy - half, miny + half) if y[0] > y[-1] else slice(miny + half, maxy - half)
        # AEF COGs are stored south-up; flip to north-up to match Earth Engine.
        da = da.sel(x=slice(minx + half, maxx - half), y=yslice).sortby("y", ascending=False)
        t2 = perf_counter()
        t_build = t2 - (t1 + t_open)
        values = np.asarray(da.transpose("band", "y", "x").values)
        t_read = perf_counter() - t2
    return {
        "init_s": t_init,
        "open_s": t_open,
        "build_s": t_build,
        "read_s": t_read,
        "fetch_s": perf_counter() - t1,
        "tiles": len(tiles),
        "values": values,
        "module": aef_loader.__file__,
    }


_OBSTORE_REQS = 0
_OBSTORE_BYTES = 0

def patch_obstore():
    """
    Monkeypatch obstore to count network requests and bytes received.
    This hooks the Python-level get/get_range/get_ranges functions.
    Misses any requests made directly within Rust (e.g., if a Rust function
    calls another Rust function directly without crossing the Python boundary),
    but catches the primary fetch calls from virtualizarr / fsspec.
    """
    try:
        import obstore
    except ImportError:
        return
    
    orig_get = obstore.get
    orig_get_async = obstore.get_async
    orig_get_range = obstore.get_range
    orig_get_range_async = obstore.get_range_async
    orig_get_ranges = obstore.get_ranges
    orig_get_ranges_async = obstore.get_ranges_async

    def _record(size):
        global _OBSTORE_REQS, _OBSTORE_BYTES
        _OBSTORE_REQS += 1
        _OBSTORE_BYTES += size

    def my_get(*args, **kwargs):
        res = orig_get(*args, **kwargs)
        _record(res.meta["size"])
        return res
        
    async def my_get_async(*args, **kwargs):
        res = await orig_get_async(*args, **kwargs)
        _record(res.meta["size"])
        return res
        
    def my_get_range(*args, **kwargs):
        res = orig_get_range(*args, **kwargs)
        _record(len(res))
        return res
        
    async def my_get_range_async(*args, **kwargs):
        res = await orig_get_range_async(*args, **kwargs)
        _record(len(res))
        return res
        
    def my_get_ranges(*args, **kwargs):
        res = orig_get_ranges(*args, **kwargs)
        _record(sum(len(b) for b in res))
        return res
        
    async def my_get_ranges_async(*args, **kwargs):
        res = await orig_get_ranges_async(*args, **kwargs)
        _record(sum(len(b) for b in res))
        return res
        
    obstore.get = my_get
    obstore.get_async = my_get_async
    obstore.get_range = my_get_range
    obstore.get_range_async = my_get_range_async
    obstore.get_ranges = my_get_ranges
    obstore.get_ranges_async = my_get_ranges_async

def get_peak_rss_bytes() -> int:
    import psutil, os, sys
    try:
        p = psutil.Process(os.getpid())
        if sys.platform == "win32":
            return p.memory_info().peak_wset
        else:
            import resource
            ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return ru * 1024 if sys.platform.startswith("linux") else ru
    except Exception:
        return 0

def worker(method: str, size: int, manifest_dir: str | None, save: str | None) -> None:
    patch_obstore()
    if method == "geedim":
        result = run_geedim(size)
        result["obstore_requests"] = None
        result["obstore_bytes"] = None
    else:
        result = asyncio.run(
            run_aef(size, plus=method.startswith("plus"), manifest_dir=manifest_dir,
                    chunks="all-bands" if method.endswith("allbands") else "native")
        )
        result["obstore_requests"] = _OBSTORE_REQS
        result["obstore_bytes"] = _OBSTORE_BYTES

    result["peak_rss_bytes"] = get_peak_rss_bytes()
    values = result.pop("values")
    result["check"] = checksum(values)
    if save:
        np.save(save, values)
    print("RESULT " + json.dumps(result))


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #


def spawn(method, size, env, manifest_dir=None, save=None) -> dict:
    # Run by file path from a neutral cwd so only PYTHONPATH decides which
    # aef_loader is imported (``-m`` from PACKAGE_ROOT would always pick the fork).
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker",
           method, "--size", str(size)]
    if manifest_dir:
        cmd += ["--manifest-dir", manifest_dir]
    if save:
        cmd += ["--save", save]
    started = perf_counter()
    proc = subprocess.run(cmd, env=env, cwd=env["AEF_INDEX_DIR"], capture_output=True, text=True)
    wall = perf_counter() - started
    line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT ")), None)
    if proc.returncode or line is None:
        raise RuntimeError(f"{method} {size} failed:\n{proc.stdout}\n{proc.stderr}")
    result = json.loads(line[len("RESULT "):])
    result.update(method=method, size=size, process_wall_s=wall)
    return result


def dequantize(codes: np.ndarray) -> np.ndarray:
    v = codes.astype(np.float32)
    out = (v / 127.5) ** 2 * np.sign(v)
    out[codes == -128] = np.nan
    return out


def arrays_equal(left_path: str, right_path: str) -> bool:
    """Compare memory-mapped arrays one band at a time."""
    left = np.load(left_path, mmap_mode="r")
    right = np.load(right_path, mmap_mode="r")
    if left.shape != right.shape:
        return False
    return all(np.array_equal(left[band], right[band]) for band in range(left.shape[0]))


def compare(upstream_path: str | None, plus_path: str | None,
            gee_path: str | None, plus_allbands_path: str | None = None,
            plus_native_path: str | None = None) -> dict:
    """Run available integrity checks on memory-mapped reference arrays.

    ``plus_path`` is the warm/native reference.  Other Source Cooperative
    variants must be byte-identical to it; Earth Engine is compared to it when
    available, otherwise to upstream.
    """
    paths = {
        "upstream": upstream_path,
        "plus": plus_native_path,
        "plus_warm": plus_path,
        "geedim": gee_path,
        "plus_warm_allbands": plus_allbands_path,
    }
    result = {"shapes": {}, "integrity_skipped": [], "integrity_failures": []}
    for method, path in paths.items():
        if path:
            result["shapes"][method] = list(np.load(path, mmap_mode="r").shape)

    if upstream_path and plus_path:
        source_equal = arrays_equal(upstream_path, plus_path)
        # Retain this established field name: the plus reference is plus_warm.
        result["upstream_equals_plus"] = source_equal
        if not source_equal:
            result["integrity_failures"].append(
                "upstream and plus_warm are not bit-identical"
            )
    else:
        result["integrity_skipped"].append(
            "upstream_vs_plus_warm (requires upstream and plus_warm)"
        )

    if plus_native_path and plus_path:
        native_equal = arrays_equal(plus_native_path, plus_path)
        result["plus_equals_plus_warm"] = native_equal
        if not native_equal:
            result["integrity_failures"].append(
                "plus and plus_warm are not bit-identical"
            )
    else:
        result["integrity_skipped"].append(
            "plus_vs_plus_warm (requires plus and plus_warm)"
        )

    if plus_allbands_path and plus_path:
        allbands_equal = arrays_equal(
            plus_allbands_path, plus_path
        )
        result["plus_warm_allbands_equals_plus_warm"] = allbands_equal
        if not allbands_equal:
            result["integrity_failures"].append(
                "plus_warm_allbands and plus_warm are not bit-identical"
            )
    else:
        result["integrity_skipped"].append(
            "plus_warm_allbands_vs_plus_warm (requires plus_warm_allbands and plus_warm)"
        )

    gee_reference_path = plus_path or upstream_path
    if gee_path and gee_reference_path:
        source = np.load(gee_reference_path, mmap_mode="r")
        gee = np.load(gee_path, mmap_mode="r")
        result["gee_reference_method"] = "plus_warm" if plus_path else "upstream"
        if source.shape != gee.shape:
            result["integrity_failures"].append(
                "geedim and the Source Cooperative reference have different shapes"
            )
        else:
            max_diff, diff_sum, finite_overlap, nodata_mismatch = 0.0, 0.0, 0, 0
            for band in range(source.shape[0]):
                deq = dequantize(np.asarray(source[band]))
                g = np.asarray(gee[band])
                source_finite = np.isfinite(deq)
                gee_finite = np.isfinite(g)
                nodata_mismatch += int(np.logical_xor(source_finite, gee_finite).sum())
                both = source_finite & gee_finite
                d = np.abs(deq[both] - g[both])
                if d.size:
                    max_diff = max(max_diff, float(d.max()))
                    diff_sum += float(d.sum(dtype=np.float64))
                    finite_overlap += d.size
            result["gee_vs_dequantized_finite_overlap_pixels"] = finite_overlap
            result["gee_vs_dequantized_nodata_mask_mismatch_pixels"] = nodata_mismatch
            result["gee_vs_dequantized_finite_overlap_empty"] = finite_overlap == 0
            if nodata_mismatch:
                result["integrity_failures"].append(
                    "Earth Engine and the Source Cooperative reference have different nodata masks"
                )
            if finite_overlap:
                result["gee_vs_dequantized_max_abs_diff"] = max_diff
                result["gee_vs_dequantized_mean_abs_diff"] = diff_sum / finite_overlap
            else:
                result["gee_vs_dequantized_max_abs_diff"] = None
                result["gee_vs_dequantized_mean_abs_diff"] = None
                result["integrity_failures"].append(
                    "Earth Engine and the Source Cooperative reference have no finite overlap"
                )
    else:
        result["integrity_skipped"].append(
            "geedim_vs_dequantized (requires geedim and a Source Cooperative reference)"
        )

    result["integrity_status"] = "FAILED" if result["integrity_failures"] else "OK"
    return result


def driver(sizes, repeats, repeats_large, out_path, methods, keep_workdir: bool) -> None:
    import shutil

    work = Path(tempfile.mkdtemp(prefix="aef-chip-bench-"))
    try:
        base_env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        base_env["AEF_INDEX_DIR"] = str(work / "index")
        base_env["CHIP_SITE"] = SITE
        (work / "index").mkdir(parents=True)
        # Import the fork from a local copy: importing it from a network drive adds
        # ~8 s per process, which is a property of the drive, not the package.
        shutil.copytree(PACKAGE_ROOT / "aef_loader", work / "fork" / "aef_loader")
        plus_env = dict(base_env, PYTHONPATH=str(work / "fork"))

        # Download the shared index once (not timed per run).
        t = perf_counter()
        subprocess.run(
            [sys.executable, "-c",
             "import asyncio,os;from pathlib import Path;"
             "from aef_loader import AEFIndex,DataSource;"
             "i=AEFIndex(source=DataSource.SOURCE_COOP,cache_dir=Path(os.environ['AEF_INDEX_DIR']));"
             "asyncio.run(i.download())"],
            env=plus_env, cwd=base_env["AEF_INDEX_DIR"], check=True,
        )
        index_download_s = perf_counter() - t

        results = []
        for size in sizes:
            warm_dir = str(work / f"manifests-warm-{size}")
            # Save every selected method as a reference.  A warm/native plus
            # reference is also needed to validate selected plus variants.
            reference_methods = set(methods)
            if any(method.startswith("plus") for method in methods):
                reference_methods.add("plus_warm")
            ref = {}
            for method in ("plus_warm", "plus_warm_allbands", "plus", "upstream", "geedim"):
                if method not in reference_methods:
                    continue
                env = plus_env if method.startswith("plus") else base_env
                path = str(work / f"{method}-{size}.npy")
                spawn(method, size, env,
                      manifest_dir=warm_dir if method.startswith("plus_warm") else None,
                      save=path)
                ref[method] = path
            for rep in range(repeats if size < 2048 else repeats_large):
                # Rotate the order so no method always goes first.
                order = [m for m in ("geedim", "upstream", "plus", "plus_warm", "plus_warm_allbands") if m in methods]
                order = order[rep % len(order):] + order[:rep % len(order)]
                for method in order:
                    env = plus_env if method.startswith("plus") else base_env
                    mdir = (warm_dir if method.startswith("plus_warm")
                            else str(work / f"manifests-cold-{size}-{rep}") if method == "plus"
                            else None)
                    r = spawn(method, size, env, manifest_dir=mdir)
                    r["rep"] = rep
                    print(json.dumps({k: r[k] for k in ("method", "size", "rep", "fetch_s", "process_wall_s", "obstore_requests", "obstore_bytes", "peak_rss_bytes") if k in r}), flush=True)
                    results.append(r)

            results.append({
                "method": "integrity",
                "size": size,
                **compare(ref.get("upstream"), ref.get("plus_warm"), ref.get("geedim"),
                          ref.get("plus_warm_allbands"), ref.get("plus")),
            })
            print(json.dumps(results[-1]), flush=True)
            for path in ref.values():
                Path(path).unlink()

            # Write after every size so a late failure keeps the earlier results.
            payload = {"index_download_s": index_download_s, "year": YEAR, "site": SITE,
                       "crs": CRS, "origin": [X0, Y0], "results": results}
            Path(out_path).write_text(json.dumps(payload, indent=2))
            print(f"wrote {out_path}", flush=True)
    finally:
        if keep_workdir:
            print(f"kept benchmark workdir: {work}", flush=True)
        else:
            shutil.rmtree(work, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=["geedim", "upstream", "plus", "plus_warm", "plus_warm_allbands"])
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--manifest-dir")
    parser.add_argument("--save")
    parser.add_argument("--sizes", type=int, nargs="+", default=[256, 1024])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--repeats-large", type=int, default=3,
                        help="repeats for chips of 2048 px and larger")
    parser.add_argument("--methods", nargs="+", default=["geedim", "upstream", "plus", "plus_warm"],
                        choices=["geedim", "upstream", "plus", "plus_warm", "plus_warm_allbands"],
                        help="methods to time (default: all four)")
    parser.add_argument("--out", default="chip_benchmark_results.json")
    parser.add_argument("--keep-workdir", action="store_true",
                        help="keep the temporary benchmark workspace for inspection")
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.size, args.manifest_dir, args.save)
    else:
        driver(args.sizes, args.repeats, args.repeats_large, args.out, args.methods,
               args.keep_workdir)


if __name__ == "__main__":
    main()
