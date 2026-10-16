#!/usr/bin/env python3
"""Run the cache simulator on the traces, print two tables and save results/cache_sim.csv.

Usage (from the project root, after 'make cachesim'):
    python3 cache_sim/run_sim.py
"""
import csv
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SIM = ROOT / "build" / "cache_sim"
TRACES = ROOT / "cache_sim" / "traces"
OUT = ROOT / "results"
CACHE_BYTES, LINE_BYTES = 32 * 1024, 64          # a typical 32 KB L1 data cache with 64-byte lines


def simulate(trace, ways):
    out = subprocess.run([str(SIM), str(CACHE_BYTES), str(LINE_BYTES), str(ways), str(TRACES / f"{trace}.trace")],
                         capture_output=True, text=True, check=True).stdout
    m = re.search(r"sets=(\d+).*accesses=(\d+) hits=(\d+) misses=(\d+).*hit rate=([\d.]+)%", out)
    return {"sets": int(m[1]), "accesses": int(m[2]), "hits": int(m[3]), "misses": int(m[4]), "hit_rate": float(m[5])}


def bar(pct):
    return "#" * int(pct / 5 + 0.5)


def main():
    if not SIM.exists():
        sys.exit(f"{SIM} not found: run 'make cachesim' first")
    if not (TRACES / "row.trace").exists():
        subprocess.run([sys.executable, str(ROOT / "cache_sim" / "gen_traces.py")], check=True)
    OUT.mkdir(exist_ok=True)
    rows = []

    print(f"\nCache: {CACHE_BYTES // 1024} KB, {LINE_BYTES} B lines, LRU. Matrix: 512 x 512 ints.\n")
    print("A) Traversal order, 8-way cache")
    print(f"  {'Trace':<10} {'Accesses':>9} {'Hits':>8} {'Misses':>8} {'Hit rate':>9}  Hit-rate bar")
    print("  " + "-" * 62)
    for trace in ("row", "col", "blocked"):
        r = simulate(trace, 8)
        rows.append({"experiment": "traversal", "trace": trace, "ways": 8, **r})
        print(f"  {trace:<10} {r['accesses']:>9} {r['hits']:>8} {r['misses']:>8} {r['hit_rate']:>8.2f}%  {bar(r['hit_rate'])}")

    print("\nB) Column traversal, different associativity (same 32 KB)")
    print(f"  {'Ways':<22} {'Sets':>5} {'Hit rate':>9}  Hit-rate bar")
    print("  " + "-" * 56)
    for ways in (1, 2, 4, 8, 16, 64, CACHE_BYTES // LINE_BYTES):
        r = simulate("col", ways)
        label = f"{ways}" + (" (fully associative)" if r["sets"] == 1 else "")
        rows.append({"experiment": "associativity", "trace": "col", "ways": ways, **r})
        print(f"  {label:<22} {r['sets']:>5} {r['hit_rate']:>8.2f}%  {bar(r['hit_rate'])}")

    with open(OUT / "cache_sim.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["experiment", "trace", "ways", "sets", "accesses", "hits", "misses", "hit_rate"])
        w.writeheader()
        w.writerows(rows)
    print("\nsaved results/cache_sim.csv\n")


if __name__ == "__main__":
    main()
