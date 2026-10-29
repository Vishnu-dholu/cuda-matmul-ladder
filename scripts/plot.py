#!/usr/bin/env python3
"""Make the README charts from results/summary.csv.

Usage (from the project root):
    python3 scripts/plot.py                           # 7546 GFLOPS peak (boost clock)
    python3 scripts/plot.py --peak-gflops 6100        # another clock/GPU
Outputs:
    results/gflops_vs_n.png   every kernel across all sizes
    results/ladder.png        the optimization ladder at one size
    results/roofline.png      roofline with each kernel's modelled arithmetic intensity
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # draw to files, no window needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

ORDER = ["v1_naive", "v2_coalesced", "v3_tiled_t16", "v3_tiled_t32", "v4_reg_64", "v4_reg_128", "cublas"]
# modelled FLOP per DRAM byte (see notes, Part D3): naive 0.25, tile T -> T/4, block BMxBN -> BM*BN/(2(BM+BN))
AI = {"v1_naive": 0.25, "v2_coalesced": 0.25, "v3_tiled_t16": 4.0, "v3_tiled_t32": 8.0,
      "v4_reg_64": 16.0, "v4_reg_128": 32.0}
# label offsets in points, so neighbouring points do not overlap on the roofline chart
LABEL_OFFSET = {"v1_naive": (6, -12), "v2_coalesced": (6, 4), "v3_tiled_t16": (-78, 6), "v3_tiled_t32": (6, -12),
                "v4_reg_64": (-62, 6), "v4_reg_128": (6, -12)}


def gflops_vs_n(df, kernels):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for k in kernels:
        d = df[df.kernel == k].sort_values("N")
        ax.errorbar(d.N, d.gflops, yerr=[d.gflops - d.gflops_min, d.gflops_max - d.gflops],
                    marker="o", capsize=3, label=k, linestyle="--" if k == "cublas" else "-")
    sizes = sorted(df.N.unique())
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(sizes)
    ax.xaxis.set_major_formatter(mticker.FixedFormatter([str(s) for s in sizes]))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel("Matrix size N (N x N)")
    ax.set_ylabel("GFLOPS (log scale, median of runs)")
    ax.set_title("FP32 matrix multiply: every kernel across sizes")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / "gflops_vs_n.png", dpi=150)


def ladder(df, kernels, n):
    d = df[df.N == n].set_index("kernel").reindex(kernels)
    ceiling = d.loc["cublas", "gflops"] if "cublas" in d.index else d.gflops.max()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = ["#999999" if k == "cublas" else "#76b900" for k in kernels]   # NVIDIA green
    bars = ax.barh(kernels[::-1], d.gflops.values[::-1], color=colors[::-1])
    labels = [f"{g:,.0f} GFLOPS  ({100 * g / ceiling:.0f}% of cuBLAS)" for g in d.gflops.values[::-1]]
    ax.bar_label(bars, labels=labels, padding=4, fontsize=8)
    ax.set_xlim(0, d.gflops.max() * 1.45)
    ax.set_xlabel("GFLOPS (median)")
    ax.set_title(f"Optimization ladder at N = {n}")
    fig.tight_layout()
    fig.savefig(RES / "ladder.png", dpi=150)


def roofline(df, kernels, n, peak, bw):
    d = df[df.N == n].set_index("kernel")
    x = np.logspace(-1, 3, 400)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.loglog(x, np.minimum(peak, x * bw), "k-", lw=2, label=f"roofline ({peak:.0f} GFLOPS, {bw:.0f} GB/s)")
    ridge = peak / bw
    ax.axvline(ridge, color="k", ls=":", alpha=0.4)
    ax.text(ridge * 1.08, peak * 0.03, f"ridge {ridge:.1f} FLOP/B", fontsize=8, alpha=0.7)
    for k in kernels:
        if k in AI and k in d.index:
            ax.scatter(AI[k], d.loc[k, "gflops"], s=50, zorder=3)
            ax.annotate(k, (AI[k], d.loc[k, "gflops"]), textcoords="offset points", xytext=LABEL_OFFSET.get(k, (6, 4)), fontsize=8)
    if "cublas" in d.index:
        ax.axhline(d.loc["cublas", "gflops"], color="gray", ls="--",
                   label=f"cuBLAS measured ({d.loc['cublas', 'gflops']:.0f} GFLOPS)")
    ax.text(0.12, peak * 1.6, "Points above the roof: caches serve reuse that this DRAM-only model ignores",
            fontsize=8, alpha=0.8)
    ax.set_xlim(0.1, 1000)
    ax.set_ylim(10, peak * 3)
    ax.set_xlabel("Modelled arithmetic intensity (FLOP per DRAM byte)")
    ax.set_ylabel("GFLOPS")
    ax.set_title(f"Roofline at N = {n}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(RES / "roofline.png", dpi=150)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--peak-gflops", type=float, default=7546.0,
                   help="FP32 peak: 2 x 1920 cores x boost clock (1.965 GHz seen in the clock log = 7546; device_info reports only 1.2 GHz)")
    p.add_argument("--peak-bw", type=float, default=264.0, help="DRAM bandwidth in GB/s (device_info)")
    p.add_argument("--n", type=int, default=2048, help="matrix size for the ladder and roofline charts")
    args = p.parse_args()

    df = pd.read_csv(RES / "summary.csv")
    kernels = [k for k in ORDER if k in set(df.kernel)]
    if args.n not in set(df.N):
        raise SystemExit(f"N={args.n} not in summary.csv (have {sorted(df.N.unique())})")

    gflops_vs_n(df, kernels)
    ladder(df, kernels, args.n)
    roofline(df, kernels, args.n, args.peak_gflops, args.peak_bw)
    print("saved results/gflops_vs_n.png, results/ladder.png, results/roofline.png")


if __name__ == "__main__":
    main()
