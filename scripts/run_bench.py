#!/usr/bin/env python3
"""Run ./build/matmul for several sizes, repeat each run, save raw + median results.

Usage (from the project root):
    python3 scripts/run_bench.py                      # default sizes, 3 repeats
    python3 scripts/run_bench.py --sizes 1024 2048    # just some sizes
Outputs:
    results/raw.csv      every run
    results/summary.csv  median per (N, kernel), plus min/max
    results/run_info.txt GPU, driver and date, for the README
"""
import argparse
import csv
import datetime
import io
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "build" / "matmul"
OUT = ROOT / "results"


def run_once(n, reps):
    """Run the benchmark once in CSV mode and return its rows as dicts."""
    proc = subprocess.run([str(BIN), str(n), str(reps), "--csv"],
                          capture_output=True, text=True, check=True)
    return list(csv.DictReader(io.StringIO(proc.stdout)))


def shell(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        return "n/a"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sizes", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096])
    p.add_argument("--repeats", type=int, default=3, help="independent runs per size (median is reported)")
    p.add_argument("--cooldown", type=float, default=5.0, help="seconds of rest between runs (laptop heat)")
    args = p.parse_args()

    if not BIN.exists():
        sys.exit(f"{BIN} not found: run 'make matmul' first")
    OUT.mkdir(exist_ok=True)

    raw = []
    for n in args.sizes:
        reps = 3 if n >= 4096 else 10          # the naive kernel is slow at 4096
        for r in range(1, args.repeats + 1):
            print(f"N={n:5d}  run {r}/{args.repeats}", flush=True)
            for row in run_once(n, reps):
                row["run"] = r
                raw.append(row)
            time.sleep(args.cooldown)

    with open(OUT / "raw.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["kernel", "N", "run", "ms", "gflops", "correct"])
        w.writeheader()
        for row in raw:
            w.writerow({k: row[k] for k in w.fieldnames})

    groups = defaultdict(list)
    for row in raw:
        groups[(int(row["N"]), row["kernel"])].append(row)

    summary, bad = [], []
    for (n, kernel), rows in groups.items():
        g = [float(r["gflops"]) for r in rows]
        ms = [float(r["ms"]) for r in rows]
        ok = all(r["correct"] == "1" for r in rows)
        if not ok:
            bad.append(f"{kernel} at N={n}")
        summary.append({"N": n, "kernel": kernel,
                        "ms": round(statistics.median(ms), 4),
                        "gflops": round(statistics.median(g), 1),
                        "gflops_min": round(min(g), 1), "gflops_max": round(max(g), 1),
                        "correct": int(ok)})

    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    info = [f"date: {datetime.datetime.now():%Y-%m-%d %H:%M}",
            f"gpu, driver: {shell(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'])}",
            f"nvcc: {shell(['nvcc', '--version']).splitlines()[-1] if shell(['nvcc', '--version']) else 'n/a'}",
            f"sizes: {args.sizes}  repeats: {args.repeats}  cooldown: {args.cooldown}s"]
    (OUT / "run_info.txt").write_text("\n".join(info) + "\n")

    print(f"\nsaved {len(raw)} rows to results/raw.csv and {len(summary)} to results/summary.csv")
    if bad:
        print("WARNING: incorrect results for: " + ", ".join(bad))


if __name__ == "__main__":
    main()
