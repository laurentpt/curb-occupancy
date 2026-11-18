"""Intersections and per-block counts in a reference pass's along-street coordinates.

With a perpendicular view, a cross street is a stretch where the horizon row looks straight
down a street (depth >> house setbacks). Both opposite-direction passes see the same cross
streets at the same along-street positions, so we accept a stretch if both (aligned) passes
see it deep, or one pass sees it very deep. Validated by eye on 2026-10-22 (10/10 accepted
candidates were cross streets; the 2 single-pass candidates at lower depth were a driveway and
a car gap).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter


def _runs(mask: np.ndarray, grid: np.ndarray, min_len: float):
    out, i, n = [], 0, len(mask)
    step = grid[1] - grid[0]
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if (j - i) * step >= min_len:
                out.append((float(grid[i]), float(grid[j - 1])))
            i = j
        else:
            i += 1
    return out


def depth_on_grid(trace: pd.DataFrame, mapf, grid: np.ndarray) -> np.ndarray:
    xa = mapf(trace.X_m.values)
    o = np.argsort(xa)
    Z = np.nan_to_num(trace.Z_m.values, nan=100, posinf=100).clip(0, 100)[o]
    return median_filter(np.interp(grid, xa[o], Z, left=np.nan, right=np.nan), size=9)


def find_intersections(depths: list[np.ndarray], grid: np.ndarray, both_m: float = 20.0,
                       single_m: float = 30.0, min_len: float = 5.0, merge_m: float = 15.0,
                       groups: list | None = None, cars_ref: list | None = None,
                       veto_passes: int = 1):
    """Deep stretches: every riding direction (`groups`, one label per pass; default: each
    pass its own group) sees > both_m in at least one of its passes, or one pass sees
    > single_m. Voting is per direction because passes in the same direction look at the
    same side of the street: an open yard on that side is 'deep' in all of them, while a
    cross street is deep from both sides (one occluded pass does not erase it). Open yards/fields behind the parked row pass this test too; with
    `cars_ref` (per pass, reference-axis centres of parked-row vehicles) runs containing
    parked vehicles in >= veto_passes passes are dropped *before* merging, so a field next
    to a cross street does not take the cross street with it. Returns (kept, dropped)."""
    D = np.vstack(depths)
    groups = np.asarray(groups if groups is not None else range(len(D)))
    valid = np.isfinite(D)
    deep = np.nan_to_num(D, nan=0) > both_m
    G = np.unique(groups)
    g_valid = np.array([valid[groups == g].any(0) for g in G])
    g_deep = np.array([deep[groups == g].any(0) for g in G])
    both = g_valid.all(0) & g_deep.all(0) if len(G) > 1 else (valid.all(0) & deep.all(0))
    single = (np.nan_to_num(D, nan=0) > single_m).any(0)
    runs = _runs(both, grid, min_len) + _runs(single, grid, min_len + 1)
    runs.sort()
    dropped = pd.DataFrame(columns=["x0_m", "x1_m", "centre_m", "passes_with_cars"])
    if cars_ref is not None and runs:
        r = pd.DataFrame([{"x0_m": a, "x1_m": b, "centre_m": (a + b) / 2} for a, b in runs])
        r, dropped = drop_occupied(r, cars_ref, veto_passes)
        runs = list(zip(r.x0_m, r.x1_m))
    merged = []
    for a, b in runs:
        if merged and a - merged[-1][1] < merge_m:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    out = pd.DataFrame([{"x0_m": a, "x1_m": b, "centre_m": (a + b) / 2} for a, b in merged],
                       columns=["x0_m", "x1_m", "centre_m"])
    return (out, dropped) if cars_ref is not None else out


def drop_occupied(ints: pd.DataFrame, cars_ref: list[np.ndarray], min_passes: int = 1,
                  margin_m: float = 1.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove candidates with parked-row vehicle centres inside them in >= min_passes passes
    (or in every pass when fewer): a cross street on the parked side has no parked cars in
    its roadway. `cars_ref`: per pass, centres (reference axis) of static kept vehicles of
    the active plane. Returns (kept, dropped)."""
    if not len(ints):
        return ints, ints
    need = min(min_passes, len(cars_ref))
    hits = np.array([sum(bool(np.any((c > r.x0_m + margin_m) & (c < r.x1_m - margin_m)))
                         for c in cars_ref) for r in ints.itertuples()])
    bad = hits >= max(need, 1)
    return (ints[~bad].reset_index(drop=True),
            ints[bad].assign(passes_with_cars=hits[bad]).reset_index(drop=True))


def blocks_from_intersections(ints: pd.DataFrame, x_max: float, blank_m: float = 6.0,
                              x_min: float = 0.0):
    """Blocks between consecutive intersections; a margin of `blank_m` beyond each
    intersection stretch is excluded (corner parking is mostly illegal and the main FP risk).
    Blocks are clipped to [x_min, x_max] (intersections may lie just outside); a block is
    partial when it runs to an end of the span or loses more than `blank_m` to the clipping."""
    edges = [(x_min, x_min)] + [(r.x0_m - blank_m, r.x1_m + blank_m) for r in ints.itertuples()] \
        + [(x_max, x_max)]
    names = [""] + (list(ints["name"].fillna("")) if "name" in ints else [""] * len(ints)) + [""]
    rows = []
    for k in range(len(edges) - 1):
        a0, b0 = edges[k][1], edges[k + 1][0]
        a, b = max(a0, x_min), min(b0, x_max)
        if b - a > 10:
            partial = (k == 0 or k == len(edges) - 2 or a - a0 > blank_m or b0 - b > blank_m)
            rows.append({"block_id": f"B{k:02d}", "x0_m": round(a, 1), "x1_m": round(b, 1),
                         "length_m": round(b - a, 1), "partial": partial,
                         "from": names[k], "to": names[k + 1]})
    return pd.DataFrame(rows)


def assign_blocks(x: np.ndarray, blocks: pd.DataFrame) -> np.ndarray:
    out = np.full(len(x), "", dtype=object)
    for r in blocks.itertuples():
        out[(x >= r.x0_m) & (x < r.x1_m)] = r.block_id
    return out
