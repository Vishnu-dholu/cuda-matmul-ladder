#!/usr/bin/env python3
"""Print Markdown results tables from results/summary.csv (paste them into the README).

Usage (from the project root, inside the venv):
    python3 scripts/make_table.py            # N = 2048 and 4096
    python3 scripts/make_table.py 1024 2048  # chosen sizes
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ORDER = ["v1_naive", "v2_coalesced", "v3_tiled_t16", "v3_tiled_t32", "v4_reg_64", "v4_reg_128", "cublas"]
PREV = {"v2_coalesced": "v1_naive", "v3_tiled_t16": "v2_coalesced", "v3_tiled_t32": "v2_coalesced",
        "v4_reg_64": "v3_tiled_t32", "v4_reg_128": "v3_tiled_t32", "cublas": "v4_reg_128"}


def table(df, n):
    d = df[df.N == n].set_index("kernel")
    kernels = [k for k in ORDER if k in d.index]
    ceiling, base = d.loc["cublas", "gflops"], d.loc["v1_naive", "gflops"]
    out = [f"**N = {n}**", "",
           "| Kernel | GFLOPS | % of cuBLAS | vs v1 | vs previous step |",
           "|---|---:|---:|---:|---:|"]
    for k in kernels:
        g = d.loc[k, "gflops"]
        step = f"{g / d.loc[PREV[k], 'gflops']:.2f}x" if k in PREV and PREV[k] in d.index else "-"
        out.append(f"| {k} | {g:,.0f} | {100 * g / ceiling:.1f}% | {g / base:.1f}x | {step} |")
    return "\n".join(out)


if __name__ == "__main__":
    df = pd.read_csv(ROOT / "results" / "summary.csv")
    sizes = [int(a) for a in sys.argv[1:]] or [2048, 4096]
    print("\n\n".join(table(df, n) for n in sizes))
