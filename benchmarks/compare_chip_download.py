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
            # Shrink by 1 m so the half-pixel margin does not add an edge row.
            inner = (minx + 1, miny + 1, maxx - 1, maxy - 1)
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
        values = np.asarray(da.transpose("band", "y", "x").values)
        t_read = perf_counter() - t2
    return {
        "init_s": t_init,
        "open_s": t_open,
        "read_s": t_read,
        "fetch_s": perf_counter() - t1,
        "tiles": len(tiles),
        "values": values,
        "module": aef_loader.__file__,
    }


def worker(method: str, size: int, manifest_dir: str | None, save: str | None) -> None:
    if method == "geedim":
        result = run_geedim(size)
    else:
        result = asyncio.run(
            run_aef(size, plus=method.startswith("plus"), manifest_dir=manifest_dir,
                    chunks="all-bands" if method.endswith("allbands") else "native")
        )
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


def compare(upstream_path, plus_path, gee_path) -> dict:
    """Band-by-band integrity check on memory-mapped reference arrays."""
    upstream = np.load(upstream_path, mmap_mode="r")
    plus = np.load(plus_path, mmap_mode="r")
    gee = np.load(gee_path, mmap_mode="r")
    equal = upstream.shape == plus.shape
    max_diff, diff_sum, n = 0.0, 0.0, 0
    for b in range(upstream.shape[0]):
        u = np.asarray(upstream[b])
        equal = equal and np.array_equal(u, plus[b])
        deq, g = dequantize(u), np.asarray(gee[b])
        both = np.isfinite(deq) & np.isfinite(g)
        d = np.abs(deq[both] - g[both])
        if d.size:
            max_diff = max(max_diff, float(d.max()))
            diff_sum += float(d.sum(dtype=np.float64))
            n += d.size
    return {
        "upstream_equals_plus": bool(equal),
        "shapes": [list(upstream.shape), list(plus.shape), list(gee.shape)],
        "gee_vs_dequantized_max_abs_diff": max_diff,
        "gee_vs_dequantized_mean_abs_diff": diff_sum / max(n, 1),
    }


def driver(sizes, repeats, repeats_large, out_path, methods) -> None:
    import shutil

    work = Path(tempfile.mkdtemp(prefix="aef-chip-bench-"))
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
        # Populate the warm manifest cache (untimed) and save reference arrays.
        ref = {}
        refs = [(m, e) for m, e in (("plus_warm", plus_env), ("upstream", base_env), ("geedim", base_env))
                if m in methods or m == "plus_warm" and any(x.startswith("plus") for x in methods)]
        for method, env in refs:
            path = str(work / f"{method}-{size}.npy")
            spawn(method, size, env, manifest_dir=warm_dir if method == "plus_warm" else None, save=path)
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
                print(json.dumps({k: r[k] for k in ("method", "size", "rep", "fetch_s", "process_wall_s")}), flush=True)
                results.append(r)

        if "geedim" in ref:
            results.append({"method": "integrity", "size": size,
                            **compare(ref["upstream"], ref["plus_warm"], ref["geedim"])})
        else:  # Earth Engine skipped: only the two Source Coop routes can be compared
            a = np.load(ref["upstream"], mmap_mode="r")
            b = np.load(ref["plus_warm"], mmap_mode="r")
            results.append({"method": "integrity", "size": size,
                            "upstream_equals_plus": bool(a.shape == b.shape and all(
                                np.array_equal(a[i], b[i]) for i in range(a.shape[0]))),
                            "shapes": [list(a.shape), list(b.shape)]})
            del a, b
        print(json.dumps(results[-1]), flush=True)
        a = b = None  # release memory maps first, or Windows refuses the unlink
        for path in ref.values():
            Path(path).unlink()

        # Write after every size so a late failure keeps the earlier results.
        payload = {"index_download_s": index_download_s, "year": YEAR, "site": SITE,
                   "crs": CRS, "origin": [X0, Y0], "results": results}
        Path(out_path).write_text(json.dumps(payload, indent=2))
        print(f"wrote {out_path}", flush=True)


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
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.size, args.manifest_dir, args.save)
    else:
        driver(args.sizes, args.repeats, args.repeats_large, args.out, args.methods)


if __name__ == "__main__":
    main()
