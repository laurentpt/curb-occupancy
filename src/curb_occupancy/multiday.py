"""Multi-day fusion on the street axis: curb map, per-block occupancy per pass, cross-pass
vehicle matching, each per parking row.

Every pass is counted layout-gated, so its `cars.csv` holds the present parking rows whatever
the direction (each car tagged with its row); mapped onto the street axis (the layout's maps)
passes become directly comparable, block by block, without A/B pairs.

* Curb map (1 m bins x present rows): is the bin covered by the pass and occupied by a car?
  Over days, bins that are never occupied are driveways, hydrants, bus/no-parking zones; the
  ever-occupied stretches (small gaps closed) are the parking *supply*, the denominator for
  occupancy that needs no city inventory. It only grows with more days: with few passes it
  under-estimates supply (and over-estimates the occupancy rate).
* Per block x pass: coverage fraction, parked count, occupied length, rate vs supply.
* Pass pairs: order-preserving sequence match (position, length, colour) inside the common
  coverage. A match means "this spot is occupied in both"; a strict subset (small offset,
  length and colour difference) estimates "same vehicle" (residents' cars). The median offset
  of matches is the residual registration error between the two passes.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .blocks import assign_blocks
from .axismap import sequence_match
from .count import union_length

DEFAULTS = {"bin_m": 1.0, "close_gap_m": 2.5, "min_coverage": 0.9,
            "same": {"dx_m": 3.0, "dlen_m": 0.5, "dE": 15.0}}


def _closing(mask: np.ndarray, n: int) -> np.ndarray:
    """Fill False runs shorter than n bins between True bins."""
    out = mask.copy()
    idx = np.where(mask)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= n + 1:
            out[a:b] = True
    return out


def _runs(mask: np.ndarray, grid: np.ndarray, step: float):
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return [(float(grid[a]), float(grid[b - 1] + step)) for a, b in
            zip(np.where(d == 1)[0], np.where(d == -1)[0])]


def _clip_meta(root: Path, c: str) -> dict:
    v = json.loads((root / c / "video.json").read_text())
    return {"date": v["creation_time"][:10], "start": v["creation_time"][11:16]}


def _block_geo(lay, blocks: pd.DataFrame) -> pd.DataFrame:
    """Block ends as lat/lon on the street axis (none without geography)."""
    if lay.axis is None or not len(blocks):
        return pd.DataFrame()
    la0, lo0 = lay.axis.latlon(blocks.x0_m.values)
    la1, lo1 = lay.axis.latlon(blocks.x1_m.values)
    return pd.DataFrame({"block_id": blocks.block_id, "lat0": np.round(la0, 6),
                         "lon0": np.round(lo0, 6), "lat1": np.round(la1, 6),
                         "lon1": np.round(lo1, 6)})


def multiday(cfg: dict) -> dict | None:
    from .config import _merge
    from .layout import Layout
    mc = _merge(DEFAULTS, cfg.get("multiday") or {})
    root = Path(cfg["out_dir"])
    lay = Layout.load(root / "layout")
    if lay is None:
        return None
    clips = [c["id"] for c in cfg["clips"]
             if c["id"] in lay.maps and (root / c["id"] / "cars.csv").exists()]
    if not clips:
        return None
    out = root / "multiday"
    out.mkdir(exist_ok=True)
    meta = {c: _clip_meta(root, c) for c in clips}
    step = mc["bin_m"]
    grid = np.arange(lay.extent_m[0], lay.extent_m[1], step)
    mid = grid + step / 2
    blk_of_bin = assign_blocks(mid, lay.blocks) if len(lay.blocks) else np.full(len(grid), "")

    rows = list(lay.rows)
    present = {r: lay.present(r, mid) for r in rows}
    cars, ext, cov = {}, {}, {}
    for c in clips:
        d = pd.read_csv(root / c / "cars.csv", keep_default_na=False, na_values=[""])
        L = float(pd.read_csv(root / c / "trace.csv").X_m.iloc[-1])
        m = lay.maps[c]
        ext[c] = tuple(float(v) for v in m.interval(0.0, L))
        if len(d):
            d["xr_start_m"], d["xr_end_m"] = m.interval(d.x_start_m.values, d.x_end_m.values)
            d["xr_mid_m"] = (d.xr_start_m + d.xr_end_m) / 2
            d["block_id"] = assign_blocks(d.xr_mid_m.values, lay.blocks) if len(lay.blocks) else ""
            d["row"] = [lay.row_of(c, p) for p in d.plane]
        d = d.assign(clip=c, **meta[c]).sort_values("xr_start_m").reset_index(drop=True) \
            if len(d) else d
        cars[c] = d
        cov[c] = (mid >= ext[c][0]) & (mid < ext[c][1])
    allcars = pd.concat([cars[c] for c in clips if len(cars[c])], ignore_index=True)
    allcars.to_csv(out / "cars_ref.csv", index=False, float_format="%.3f")

    def of_row(d, r):
        return d[d["row"] == r] if len(d) else d

    # ---- curb map + supply, per row (bins where the row is present)
    in_block = blk_of_bin != ""
    curb_parts, sup_parts, ever = [], [], {}
    for r in rows:
        occ = {}
        for c in clips:
            o = np.zeros(len(grid), bool)
            k = of_row(cars[c], r)
            for a, b in zip(k.get("xr_start_m", []), k.get("xr_end_m", [])):
                o |= (mid >= a) & (mid < b)
            occ[c] = o & cov[c]
        n_cov = np.sum([cov[c] for c in clips], 0)
        n_occ = np.sum([occ[c] for c in clips], 0)
        ever[r] = _closing(n_occ > 0, int(round(mc["close_gap_m"] / step))) & in_block & present[r]
        pm = present[r]
        curb_parts.append(pd.DataFrame({
            "x_ref_m": grid[pm], "row": r, "block_id": blk_of_bin[pm], "n_covered": n_cov[pm],
            "n_occupied": n_occ[pm], "supply": ever[r][pm],
            **{f"occ_{c}": np.where(cov[c], occ[c].astype(int), -1)[pm] for c in clips}}))
        sup_parts += [{"row": r, "x0_ref_m": a, "x1_ref_m": b, "length_m": round(b - a, 1),
                       "block_id": blk_of_bin[int((a - grid[0]) / step)]}
                      for a, b in _runs(ever[r], grid, step)]
    curb = pd.concat(curb_parts, ignore_index=True).sort_values(["x_ref_m", "row"])
    curb.to_csv(out / "curb.csv", index=False, float_format="%.1f")
    supply = pd.DataFrame(sup_parts, columns=["row", "x0_ref_m", "x1_ref_m", "length_m",
                                              "block_id"]).sort_values("x0_ref_m")
    supply.to_csv(out / "supply.csv", index=False, float_format="%.1f")

    # ---- per block x row x pass (rows present somewhere in the block)
    out_rows = []
    for b in lay.blocks.itertuples():
        inb = blk_of_bin == b.block_id
        for r in rows:
            inr = inb & present[r]
            if not inr.any():
                continue
            sup_m = float(ever[r][inb].sum() * step)
            for c in clips:
                covf = float(cov[c][inr].mean())
                k = of_row(cars[c], r)
                k = k[k.block_id == b.block_id] if len(k) else k
                ok = covf >= mc["min_coverage"]
                occ_m = round(union_length(k.xr_start_m, k.xr_end_m), 1) if len(k) else 0.0
                out_rows.append({"block_id": b.block_id, "row": r, "x0_ref_m": b.x0_m,
                                 "x1_ref_m": b.x1_m, "length_m": b.length_m,
                                 "row_length_m": round(float(inr.sum() * step), 1),
                                 "supply_m": round(sup_m, 1), "clip": c, **meta[c],
                                 "coverage": round(covf, 2), "complete": ok,
                                 "n_parked": int(len(k)) if covf > 0 else None,
                                 "occupied_m": occ_m if covf > 0 else None,
                                 "occ_rate": round(occ_m / sup_m, 3) if ok and sup_m > 0 else None,
                                 "n_flagged": int((k.flag.fillna("") != "").sum()) if len(k)
                                 else 0})
    long = pd.DataFrame(out_rows)
    long.to_csv(out / "blocks_by_pass.csv", index=False)
    wide = None
    if len(long):
        key = ["block_id", "row", "x0_ref_m", "x1_ref_m", "length_m", "row_length_m", "supply_m"]
        long["n_show"] = np.where(long.complete, long.n_parked, np.nan)
        idx = ["block_id", "row"]
        n = long.pivot(index=idx, columns="clip", values="n_show")
        rate = long.pivot(index=idx, columns="clip", values="occ_rate")
        wide = (long[key].drop_duplicates(idx).set_index(idx)
                .join(n.reindex(columns=clips).add_prefix("n_"))
                .join(rate.reindex(columns=clips).add_prefix("rate_")).reset_index())
        geo = _block_geo(lay, lay.blocks)
        if len(geo):
            wide = wide.merge(geo, on="block_id", how="left")
        wide.to_csv(out / "counts_by_block.csv", index=False)

    # ---- pairs (sequence match within each row)
    pair_res, sheets = {}, []
    for a, b in itertools.combinations(clips, 2):
        lo, hi = max(ext[a][0], ext[b][0]), min(ext[a][1], ext[b][1])
        A = cars[a][cars[a].xr_mid_m.between(lo, hi)] if len(cars[a]) else cars[a]
        B = cars[b][cars[b].xr_mid_m.between(lo, hi)] if len(cars[b]) else cars[b]
        mr = []
        for r in rows:
            Ar, Br = of_row(A, r), of_row(B, r)
            pm = sequence_match(Ar, Br, Ar.xr_mid_m.values, Br.xr_mid_m.values) \
                if len(Ar) and len(Br) else []
            for i, j, dx in pm:
                ra, rb = Ar.iloc[i], Br.iloc[j]
                dE = float(np.linalg.norm(ra[["lab_L", "lab_a", "lab_b"]].values.astype(float)
                                          - rb[["lab_L", "lab_a", "lab_b"]].values.astype(float)))
                mr.append({"a_idx": int(Ar.index[i]), "b_idx": int(Br.index[j]), "row": r,
                           "xr_m": round(float(ra.xr_mid_m), 1), "dx_m": round(dx, 2),
                           "len_a": ra.length_m, "len_b": rb.length_m,
                           "dlen_m": round(float(ra.length_m - rb.length_m), 2),
                           "dE": round(dE, 1), "block_id": ra.block_id})
        M = pd.DataFrame(mr)
        if len(M):
            M = M.sort_values("xr_m").reset_index(drop=True)
            # residual registration: remove the local (block) median offset before the test
            M["dx_res_m"] = M.dx_m - M.groupby(["block_id", "row"]).dx_m.transform("median")
            s = mc["same"]
            M["same_vehicle"] = ((M.dx_res_m.abs() <= s["dx_m"]) & (M.dlen_m.abs() <= s["dlen_m"])
                                 & (M.dE <= s["dE"]))
        M.to_csv(out / f"matches_{a}_{b}.csv", index=False)
        both = long[(long["clip"].isin([a, b])) & long.complete].groupby(["block_id", "row"]) \
            .filter(lambda g: len(g) == 2) if len(long) else long
        diffs = (both.pivot(index=["block_id", "row"], columns="clip", values="n_parked")
                 .pipe(lambda w: (w[a] - w[b]).abs()) if len(both) else pd.Series(dtype=float))
        pair_res[f"{a}_{b}"] = {
            "days": [meta[a]["date"], meta[b]["date"]], "common_m": [round(lo, 1), round(hi, 1)],
            "n_a": int(len(A)), "n_b": int(len(B)), "matched": int(len(M)),
            "only_a": int(len(A) - len(M)), "only_b": int(len(B) - len(M)),
            "same_vehicle": int(M.same_vehicle.sum()) if len(M) else 0,
            "median_dx_m": round(float(M.dx_m.median()), 2) if len(M) else None,
            "blocks_compared": int(len(diffs)),
            "block_abs_diff_mean": round(float(diffs.mean()), 2) if len(diffs) else None,
            "blocks_within_1": int((diffs <= 1).sum()) if len(diffs) else 0}
        sheets.append((a, b, A, B, M))

    res = {"reference": lay.reference, "clips": clips, "meta": meta,
           "extent_ref_m": {c: [round(v, 1) for v in ext[c]] for c in clips},
           "supply_m": round(float(sum(v.sum() for v in ever.values()) * step), 1),
           "supply_m_by_row": {r: round(float(ever[r].sum() * step), 1) for r in rows},
           "pairs": pair_res}
    (out / "multiday.json").write_text(json.dumps(res, indent=2))
    _plot(lay, clips, cars, ext, curb, supply, meta, out / "occupancy.png")
    for a, b, A, B, M in sheets:
        _sheet(cfg, lay, a, b, A, B, M, out / f"pairs_{a}_{b}.jpg")
    return res


def _plot(lay, clips, cars, ext, curb, supply, meta, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(clips)
    fig, ax = plt.subplots(figsize=(22, 1.0 + 0.7 * (n + 2)))
    for r in lay.intersections.itertuples():
        ax.axvspan(r.x0_m, r.x1_m, color="0.85", zorder=-2)
    for s in lay.switches_m:
        ax.axvline(s, color="k", lw=1.2, ls="--")
    for b in lay.blocks.itertuples():
        ax.text((b.x0_m + b.x1_m) / 2, n + 1.15, b.block_id, ha="center", fontsize=7)
    ax.broken_barh(list(zip(supply.x0_ref_m, supply.length_m)) if len(supply) else [],
                   (n + 0.1, 0.5), color="#2f855a")
    ax.text(lay.extent_m[0] - 5, n + 0.35, "supply (ever occupied)", ha="right", va="center",
            fontsize=8)
    for k, c in enumerate(clips[::-1]):
        y = k
        ax.broken_barh([(ext[c][0], ext[c][1] - ext[c][0])], (y, 0.7), color="0.93", zorder=-1)
        d = cars[c]
        if len(d):
            fl = d.flag.fillna("") != ""
            ax.broken_barh(list(zip(d.xr_start_m[~fl], (d.xr_end_m - d.xr_start_m)[~fl])),
                           (y + 0.1, 0.5), color="#2b6cb0")
            ax.broken_barh(list(zip(d.xr_start_m[fl], (d.xr_end_m - d.xr_start_m)[fl])),
                           (y + 0.1, 0.5), color="#dd6b20")
        ax.text(lay.extent_m[0] - 5, y + 0.35, f"{c} {meta[c]['date']} {meta[c]['start']} "
                f"(n={len(d)})", ha="right", va="center", fontsize=8)
    ax.set_xlim(lay.extent_m[0] - 160, lay.extent_m[1] + 5)
    ax.set_ylim(-0.2, n + 1.5)
    ax.set_yticks([])
    ax.set_xlabel(f"street axis m (reference {lay.reference}); grey = intersections, dashed = "
                  "change of parked rows, orange = flagged")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def _sheet(cfg, lay, a, b, A, B, M, path, thumb_h: int = 100):
    """Matched pairs side by side (each pass flipped to the reference direction), green
    border = same vehicle; then unmatched cars (blue = only a, orange = only b)."""
    import cv2
    cache, tiles = {}, []
    fa, fb = lay.reversed(a), lay.reversed(b)
    rows = M.to_dict("records") if len(M) else []
    for r in rows:
        ta = car_crop(cfg, a, A.loc[r["a_idx"]], cache, fa, thumb_h)
        tb = car_crop(cfg, b, B.loc[r["b_idx"]], cache, fb, thumb_h)
        t = np.hstack([ta, np.full((thumb_h, 4, 3), 255, np.uint8), tb])
        col = (40, 200, 40) if r["same_vehicle"] else (200, 200, 200)
        t = cv2.copyMakeBorder(t, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=col)
        cv2.putText(t, f"{r['xr_m']:.0f}m dE{r['dE']:.0f}", (5, 15), 0, 0.42, (255, 255, 0), 1)
        tiles.append(cv2.resize(t, (t.shape[1], thumb_h)))
    ua = set(A.index) - {r["a_idx"] for r in rows}
    ub = set(B.index) - {r["b_idx"] for r in rows}
    for clip, D, used, flip, col in ((a, A, ua, fa, (60, 60, 255)), (b, B, ub, fb, (255, 140, 0))):
        for i in sorted(used, key=lambda i: D.loc[i, "xr_mid_m"]):
            r = D.loc[i]
            t = car_crop(cfg, clip, r, cache, flip, thumb_h)
            t = cv2.copyMakeBorder(t, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=col)
            cv2.putText(t, f"only {clip[-2:]} {r.xr_mid_m:.0f}m", (5, 15), 0, 0.42,
                        (255, 255, 0), 1)
            tiles.append(cv2.resize(t, (t.shape[1], thumb_h)))
    if tiles:
        cv2.imwrite(str(path), cv2.cvtColor(pack_rows(tiles, thumb_h), cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])

def car_crop(cfg: dict, clip: str, r, cache: dict, flip: bool = False, thumb_h: int = 110):
    """Mosaic crop (+15 % margin) of one vehicle row (x_start_m/x_end_m in clip metres)."""
    import cv2
    from .config import height
    from .mosaic import Mosaic
    key = (clip, r.plane)
    if key not in cache:
        cache[key] = Mosaic.load(Path(cfg["out_dir"]) / clip / f"mosaic_{r.plane}")
    m = cache[key]
    h = height(cfg, clip)
    x0 = int((r.x_start_m / h - m.x0) * m.px_per_m)
    x1 = int((r.x_end_m / h - m.x0) * m.px_per_m)
    pad = int(0.15 * (x1 - x0))
    c = m.image[:, max(0, x0 - pad):x1 + pad]
    c = c[:, ::-1] if flip else c
    s = thumb_h / c.shape[0]
    return cv2.resize(np.ascontiguousarray(c), (max(1, int(c.shape[1] * s)), thumb_h))


def pack_rows(tiles, thumb_h: int, width: int = 2400) -> np.ndarray:
    """Pack thumbnails left to right into rows of ~`width` px."""
    rows, cur, wcur = [], [], 0
    for t in tiles:
        if wcur + t.shape[1] > width and cur:
            rows.append(cur)
            cur, wcur = [], 0
        cur.append(t)
        wcur += t.shape[1] + 8
    if cur:
        rows.append(cur)
    lines = []
    for r in rows:
        line = np.hstack(sum([[t, np.zeros((thumb_h, 8, 3), np.uint8)] for t in r], []))
        lines.append(np.pad(line, ((0, 8), (0, max(0, width + 160 - line.shape[1])),
                                   (0, 0)))[:, :width + 160])
    return np.vstack(lines) if lines else np.zeros((thumb_h, width, 3), np.uint8)
