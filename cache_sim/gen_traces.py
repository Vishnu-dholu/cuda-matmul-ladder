#!/usr/bin/env python3
"""Write address traces for an N x N int matrix into cache_sim/traces/.

row.trace     visit row by row      (walks memory in order)
col.trace     visit column by column (jumps N*4 bytes every step)
blocked.trace column order, but inside small B x B blocks (cache blocking)
One hexadecimal byte address per line.
"""
from pathlib import Path

N, ELEM, B = 512, 4, 8                       # 512 x 512 ints, 4 bytes each, 8 x 8 blocks
OUT = Path(__file__).resolve().parent / "traces"


def addr(i, j):
    return (i * N + j) * ELEM


def write(name, order):
    with open(OUT / f"{name}.trace", "w") as f:
        for i, j in order:
            f.write(f"{addr(i, j):x}\n")


def row_order():
    return ((i, j) for i in range(N) for j in range(N))


def col_order():
    return ((i, j) for j in range(N) for i in range(N))


def blocked_order():
    for i0 in range(0, N, B):
        for j0 in range(0, N, B):
            for j in range(j0, j0 + B):
                for i in range(i0, i0 + B):
                    yield i, j


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    write("row", row_order())
    write("col", col_order())
    write("blocked", blocked_order())
    print(f"wrote 3 traces of {N * N} accesses to {OUT}")
