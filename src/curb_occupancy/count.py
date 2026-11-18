"""Merge per-plane detections into one list of parked vehicles per clip; occupancy profiles."""
from __future__ import annotations

import numpy as np
import pandas as pd


def merge_planes(dets: dict[str, pd.DataFrame], min_overlap: float = 0.5) -> pd.DataFrame:
    """Kept detections from all planes; a vehicle accepted in two planes is kept once
    (highest confidence). Overlap = x-overlap / shorter length."""
    rows = []
    for plane, d in dets.items():
        k = d[d.kept].copy()
        k["plane"] = plane
        rows.append(k)
    if not rows:
        return pd.DataFrame()
    allk = pd.concat(rows, ignore_index=True).sort_values("conf", ascending=False)
    keep = []
    for r in allk.itertuples():
        dup = False
        for q in keep:
            ov = min(r.x_end_m, q.x_end_m) - max(r.x_start_m, q.x_start_m)
            if ov > min_overlap * min(r.length_m, q.length_m):
                dup = True
                break
        if not dup:
            keep.append(r)
    out = pd.DataFrame(keep).drop(columns="Index", errors="ignore")
    return out.sort_values("x_start_m").reset_index(drop=True)


def union_length(starts, ends) -> float:
    iv = sorted(zip(starts, ends))
    tot, cur0, cur1 = 0.0, None, None
    for a, b in iv:
        if cur1 is None or a > cur1:
            if cur1 is not None:
                tot += cur1 - cur0
            cur0, cur1 = a, b
        else:
            cur1 = max(cur1, b)
    if cur1 is not None:
        tot += cur1 - cur0
    return tot


def occupancy_profile(cars: pd.DataFrame, x_max: float, bin_m: float = 25.0) -> pd.DataFrame:
    """Per along-street bin: number of vehicle centres and occupied length, per plane."""
    edges = np.arange(0, x_max + bin_m, bin_m)
    out = pd.DataFrame({"x0_m": edges[:-1], "x1_m": edges[1:]})
    for plane in sorted(cars.plane.unique()) if len(cars) else []:
        c = cars[cars.plane == plane]
        mid = (c.x_start_m + c.x_end_m) / 2
        out[f"n_{plane}"] = np.histogram(mid, edges)[0]
        occ = []
        for a, b in zip(edges[:-1], edges[1:]):
            s = c.x_start_m.clip(a, b)
            e = c.x_end_m.clip(a, b)
            m = e > s
            occ.append(union_length(s[m], e[m]))
        out[f"occ_m_{plane}"] = occ
    return out


def depth_occupancy(X_m: np.ndarray, Z_m: np.ndarray, band_m, step_m: float = 0.5) -> tuple:
    """Detection-free occupancy: fraction of along-street samples whose horizon-row depth
    falls in `band_m`. Returns (x grid, occupied bool) resampled on a regular grid."""
    grid = np.arange(X_m[0], X_m[-1], step_m)
    idx = np.searchsorted(X_m, grid).clip(0, len(Z_m) - 1)
    z = Z_m[idx]
    return grid, (z > band_m[0]) & (z < band_m[1])


def apply_layout(dets: dict[str, pd.DataFrame], lay, clip: str, review_min_conf: float = 0.4,
                 travel_lane_m: dict | None = None):
    """Gate detections by the street layout.

    Adds `active` (the box centre lies where this plane images a present parking row) and
    turns kept boxes of inactive planes into rejects ('travel_lane' if inside the plane's
    travel-lane depth band, `count.travel_lane_m`, else 'other_side'). Returns (gated dets,
    review list): those plus confident vehicle-sized rejects of active planes that are not
    mere geometry noise (too near/far = stopped or double-parked in the travel lane) and
    flagged kept boxes.
    """
    travel_lane_m = travel_lane_m or {}
    out, review = {}, []
    for plane, d in dets.items():
        d = d.copy()
        mid = (d.x_start_m + d.x_end_m) / 2
        d["active"] = lay.plane_active(clip, plane, mid.values) if len(d) else []
        band = travel_lane_m.get(plane)
        lane = d.depth_m.between(*band) if band else pd.Series(False, index=d.index)
        wrong = d.kept & ~d.active
        d.loc[wrong, "reject"] = np.where(lane[wrong], "travel_lane", "other_side")
        d.loc[wrong, "kept"] = False
        rv = wrong.copy()
        geo = d.reject.str.contains("too_near|too_far") & ~d.reject.str.contains(
            "clipped|too_narrow|low_conf")
        rv |= d.active & geo & (d.conf >= review_min_conf)
        rv |= d.kept & ~d.flag.isin(["", "clipped"])  # clipped: near row, depth bounded
        r = d[rv].copy()
        r["plane"] = plane
        r["reason"] = np.where(r.kept, r.flag, r.reject)
        review.append(r)
        out[plane] = d
    cols = ["plane", "reason", "x_start_m", "x_end_m", "t", "cls_name", "conf", "length_m",
            "height_m", "depth_m", "flow_ratio", "active"]
    rev = pd.concat(review, ignore_index=True)[cols] if review else pd.DataFrame(columns=cols)
    return out, rev.sort_values("t").reset_index(drop=True)


def block_counts(cars: pd.DataFrame, lay, clip: str) -> pd.DataFrame:
    """Parked vehicles and occupied length per block (blocks on the street axis)."""
    from .blocks import assign_blocks
    if not len(lay.blocks):
        return pd.DataFrame()
    m = lay.maps[clip]
    rows = []
    if len(cars):
        s, e = m.interval(cars.x_start_m.values, cars.x_end_m.values)
        blk = assign_blocks((s + e) / 2, lay.blocks)
    for b in lay.blocks.itertuples():
        r = {"block_id": b.block_id, "x0_ref_m": b.x0_m, "x1_ref_m": b.x1_m,
             "length_m": b.length_m, "partial": b.partial, "n_parked": 0,
             "occupied_length_m": 0.0, "n_flagged": 0}
        if len(cars):
            k = blk == b.block_id
            r["n_parked"] = int(k.sum())
            r["occupied_length_m"] = round(union_length(s[k], e[k]), 1) if k.any() else 0.0
            r["n_flagged"] = int((cars.flag[k] != "").sum())
        rows.append(r)
    return pd.DataFrame(rows)
