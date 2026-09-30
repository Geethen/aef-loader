"""Plot results of compare_chip_download.py.

Usage: python plot_chip_benchmark.py results.json --out chip_benchmark.png
"""
import argparse
import json
import statistics as st
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BANDS = 64
# method -> (legend label, colour, marker, bytes per value)
METHODS = {
    "geedim": ("geedim (Earth Engine, float32)", "#2a78d6", "o", 4),
    "upstream": ("aef-loader 0.3.0 (Source Coop)", "#eb6834", "s", 1),
    "plus": ("aef_loader_plus, cold cache (native chunks)", "#1baf7a", "^", 1),
    "plus_warm": ("aef_loader_plus, warm cache (native chunks)", "#4a3aa7", "D", 1),
    "plus_warm_allbands": ("aef_loader_plus, warm cache (all-bands chunks)", "#c2337a", "P", 1),
}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def load(path):
    with open(path) as f:
        data = json.load(f)
    stats = defaultdict(dict)  # method -> size -> (median, min, max)
    vals = defaultdict(lambda: defaultdict(list))
    for r in data["results"]:
        m = r.get("method")
        if m not in METHODS or "fetch_s" not in r or "size" not in r:
            continue
        vals[m][int(r["size"])].append(float(r["fetch_s"]))
    for m, d in vals.items():
        for s, v in d.items():
            stats[m][s] = (st.median(v), min(v), max(v))
    return data, stats


def size_label(s):
    return f"{s}²"


def style(ax, sizes):
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{size_label(s)}\n{s * s / 1e6:.2f} MP" for s in sizes])
    ax.minorticks_off()
    ax.set_xlabel("Chip size (pixels per side)", color=INK2)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#b9b8b2")
    ax.tick_params(colors=INK2, labelsize=8.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--out", default="chip_benchmark.png")
    a = ap.parse_args()
    data, stats = load(a.results)
    sizes = sorted({s for m in stats.values() for s in m})
    if not sizes:
        raise SystemExit("no timing rows found")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for m, (label, col, mk, _) in METHODS.items():
        if m not in stats:
            continue
        xs = sorted(stats[m])
        med = [stats[m][s][0] for s in xs]
        lo = [stats[m][s][1] for s in xs]
        hi = [stats[m][s][2] for s in xs]
        ax1.fill_between(xs, lo, hi, color=col, alpha=0.15, lw=0)
        ax1.plot(xs, med, color=col, marker=mk, ms=6, lw=2, label=label,
                 markeredgecolor="#fcfcfb", markeredgewidth=1)
        ax2.plot(xs, [s * s / 1e6 / t for s, t in zip(xs, med)], color=col,
                 marker=mk, ms=6, lw=2, label=label,
                 markeredgecolor="#fcfcfb", markeredgewidth=1)

    ax1.set_yscale("log")
    ax2.set_yscale("log")
    ax1.set_ylabel("Fetch time (s), median with min–max band", color=INK2)
    ax2.set_ylabel("Throughput (megapixels / s)", color=INK2)
    ax1.set_title("Fetch time", loc="left", fontsize=11, color=INK)
    ax2.set_title("Throughput", loc="left", fontsize=11, color=INK)
    style(ax1, sizes)
    style(ax2, sizes)
    ax1.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK)
    fig.suptitle(f"Chip download benchmark: {BANDS} bands, 10 m, {data.get('year', 2024)}, "
                 f"site: {data.get('site', '?')}", x=0.01, ha="left",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(a.out, dpi=150, facecolor="#fcfcfb")

    # markdown table
    order = [m for m in METHODS if m in stats]
    head = ["size"] + [f"{METHODS[m][0]} fetch_s median (min–max)" for m in order] \
        + [f"{METHODS[m][0]} MB" for m in order]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for s in sizes:
        cells = [size_label(s)]
        for m in order:
            if s in stats[m]:
                md, lo, hi = stats[m][s]
                cells.append(f"{md:.2f} ({lo:.2f}–{hi:.2f})")
            else:
                cells.append("–")
        cells += [f"{s * s * BANDS * METHODS[m][3] / 1e6:.1f}" for m in order]
        print("| " + " | ".join(cells) + " |")
    print(f"\nSaved {a.out}")


if __name__ == "__main__":
    main()
