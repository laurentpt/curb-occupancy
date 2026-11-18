"""Pass-to-pass registration of mosaic strips on their background (car independent).

Two passes riding the same way image the same curb in the same plane from the same side:
house fronts, trees, poles and the curb itself are static between evenings while the
parked cars change. A strip of each pass is resampled onto the reference axis through a
rough map (the occupancy / GPS map of `layout`), and windows of one are located in the
other by normalised cross-correlation of log gradient magnitude (robust to exposure
differences between evenings) with vertical slack (pitch and height differ between
passes). The confident windows give (clip x, reference x) correspondences, from which the
pass's map is refitted: robust affine + rolling-median residual.

Why not the vehicle-occupancy correlation alone: parked cars are quasi-periodic (one every
~5.5-6 m), so windowed occupancy correlation has near-equal peaks one or two cars apart and
locks onto the wrong one where cars changed between evenings (5-14 m local excursions on
the 2026-10 study passes). Passes riding the other way see the curb from the other side,
at another plane depth: their strips do not register (background parallax differs); they
are registered among themselves and their group is aligned to the reference by the pooled
occupancy of all its passes (`register_passes`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .axismap import AxisMap, MapChain, local_shifts

PXM = 60.0   # strip px per metre (registration runs at half of it)
SC = 0.5


def resample_strip(img: np.ndarray, col_of_x, xs: np.ndarray, rows_src: np.ndarray,
                   valid: np.ndarray) -> np.ndarray:
    """Mosaic columns at fractional positions col_of_x(xs), rows at rows_src; invalid
    columns black. cv2.remap works on a slice (its coordinates are limited to int16)."""
    import cv2
    cx = col_of_x(xs)
    out = np.zeros((len(rows_src), len(xs), 3), np.uint8)
    ok = valid & (cx > 0) & (cx < img.shape[1] - 1)
    if not ok.any():
        return out
    a = int(max(0, np.floor(cx[ok].min()) - 2))
    b = int(min(img.shape[1], np.ceil(cx[ok].max()) + 3))
    mx = np.tile((cx - a).astype(np.float32), (len(rows_src), 1))
    my = np.tile(rows_src.astype(np.float32)[:, None], (1, len(xs)))
    r = cv2.remap(np.ascontiguousarray(img[:, a:b]), mx, my, cv2.INTER_LINEAR,
                  borderMode=cv2.BORDER_CONSTANT)
    out[:, ok] = r[:, ok]
    return out


class Strip:
    """Prepared strip on reference-axis metres: log gradient magnitude at PXM * SC px/m
    (`g`), column validity at PXM (`valid`), first column at x0."""

    def __init__(self, g: np.ndarray, valid: np.ndarray, x0: float):
        self.g, self.valid, self.x0 = g, valid, float(x0)

    @property
    def x1(self) -> float:
        return self.x0 + len(self.valid) / PXM


def prep(img: np.ndarray) -> np.ndarray:
    import cv2
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    g = cv2.resize(g, None, fx=SC, fy=SC, interpolation=cv2.INTER_AREA)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    return np.log1p(np.sqrt(gx ** 2 + gy ** 2))


def render(mosaic, h: float, f, L: float, y_m=(-1.0, 2.6)) -> Strip | None:
    """The mosaic of one pass on reference coordinates u = f(x_clip) (any monotone map),
    heights y_m (m above the road at the plane), PXM px/m; the pass's own extent only."""
    xc = np.arange(0.0, L, 0.05)
    u = f(xc)
    o = np.argsort(u)
    u, xc = u[o], xc[o]
    u0, u1 = np.ceil(u[0]), np.floor(u[-1])
    if u1 - u0 < 10:
        return None
    xs = u0 + (np.arange(int((u1 - u0) * PXM)) + 0.5) / PXM
    ppm = mosaic.px_per_m / h
    Y = y_m[1] - (np.arange(int(round((y_m[1] - y_m[0]) * PXM))) + 0.5) / PXM
    rows_src = (mosaic.y_top * h - Y) * ppm - 0.5
    gs, valid = [], []
    for a in range(0, len(xs), 6000):  # remap slices; prepared slices are concatenated
        sl = xs[a:a + 6000]
        img = resample_strip(mosaic.image, lambda x: (np.interp(x, u, xc) / h - mosaic.x0)
                             * mosaic.px_per_m - 0.5, sl, rows_src, np.ones(len(sl), bool))
        valid.append(img.max(axis=(0, 2)) > 8)
        gs.append(prep(img))
    return Strip(np.concatenate(gs, axis=1), np.concatenate(valid), u0)


def register(anchor: Strip, cur: Strip, win_m: float = 30.0, step_m: float = 5.0,
             search_m: float = 20.0, band_m=(0.6, 2.3), y_top_m: float = 2.6) -> pd.DataFrame:
    """Per anchor window [a, a + win_m] (template rows = heights band_m: the background and
    curb above the road; the search has the full height and is cut at the ends of `cur`):
    best position in `cur`. shift = x_anchor - x_cur (m); peak2 = best score >= 1 m away."""
    import cv2
    A, C = anchor.g, cur.g
    p = PXM * SC
    rows = (int(round((y_top_m - band_m[1]) * p)), int(round((y_top_m - band_m[0]) * p)))
    lo, hi = max(anchor.x0, cur.x0), min(anchor.x1, cur.x1)
    out = []
    for a in np.arange(np.ceil(lo), hi - win_m, step_m):
        ia, wa = int(round((a - anchor.x0) * p)), int(round(win_m * p))
        if anchor.valid[int((a - anchor.x0) * PXM):int((a + win_m - anchor.x0) * PXM)].mean() < 0.98:
            continue
        c0, c1 = max(a - search_m, cur.x0), min(a + win_m + search_m, cur.x1)
        if c1 - c0 < win_m + 2.0:
            continue
        ic, wc = int(round((c0 - cur.x0) * p)), int(round((c1 - c0) * p))
        if cur.valid[int((c0 - cur.x0) * PXM):int((c1 - cur.x0) * PXM)].mean() < 0.98:
            continue
        r = cv2.matchTemplate(C[:, ic:ic + wc], A[rows[0]:rows[1], ia:ia + wa],
                              cv2.TM_CCOEFF_NORMED)
        ky, kx = np.unravel_index(np.argmax(r), r.shape)
        row = r[ky]
        d = 0.0
        if 0 < kx < len(row) - 1:
            den = row[kx - 1] - 2 * row[kx] + row[kx + 1]
            d = 0.5 * (row[kx - 1] - row[kx + 1]) / den if den < 0 else 0.0
        far = r.copy()
        far[:, max(0, kx - int(p)):kx + int(p) + 1] = -1
        x_cur = c0 + (kx + d) / p
        out.append({"x_anchor": a + win_m / 2, "x_cur": x_cur + win_m / 2, "shift": a - x_cur,
                    "dy_px": int(ky) - rows[0], "peak": float(r.max()),
                    "peak2": float(far.max())})
    return pd.DataFrame(out, columns=["x_anchor", "x_cur", "shift", "dy_px", "peak", "peak2"])


def smooth(df: pd.DataFrame, min_peak=0.2, min_margin=0.04, tol_m=0.75, k=7) -> pd.DataFrame:
    """ok = distinct peak (peak >= min_peak, >= min_margin above the best score >= 1 m away),
    consistent within tol_m with the running median of the confident windows (k windows)."""
    d = df.sort_values("x_cur").reset_index(drop=True).copy()
    conf = (d.peak >= min_peak) & ((d.peak - d.peak2) >= min_margin)
    d["med"] = d["shift"].where(conf).rolling(k, center=True, min_periods=3).median()
    d["ok"] = conf & ((d["shift"] - d.med).abs() <= tol_m)
    return d


def fit_map(x_clip: np.ndarray, x_ref: np.ndarray) -> AxisMap:
    """Correspondences -> AxisMap: robust affine + rolling median (5 windows) of the
    residual as knots in reference coordinates (held constant beyond the ends)."""
    from .street import _robust_line
    o = np.argsort(x_clip)
    x_clip, x_ref = np.asarray(x_clip, float)[o], np.asarray(x_ref, float)[o]
    a, b, _ = _robust_line(x_clip, x_ref)
    y = a * x_clip + b
    res = pd.Series(x_ref - y).rolling(5, center=True, min_periods=2).median().values
    k = np.argsort(y)
    return AxisMap(scale=abs(a), offset=b, reverse=a < 0, knots=tuple(y[k]),
                   shifts=tuple(res[k]))


def register_passes(cfg: dict, root, clips: list, ref: str, rev: dict, maps: dict,
                    L: dict, rows: dict, kept: dict, scores: dict) -> tuple[dict, dict]:
    """Refit the clip -> reference maps by strip registration (`layout.register`).

    Passes riding the same way form a group; within it a spanning tree grown from the
    reference (other groups: from their best occupancy-aligned pass, which keeps its map) adds,
    one at a time, the pass registering best (most confident windows, >= min_ok) onto a pass
    already placed, rendered through its final map. Every plane imaging a parked row in
    both passes contributes windows. Passes that register to nothing keep their map (and start
    a component of their own). A component without the reference is then moved as a whole
    by local occupancy shifts (+- pooled_max_shift_m) of all its passes' vehicles against all
    of the reference component's (several evenings pooled: fewer periodic ambiguities).
    Returns (maps, diagnostics)."""
    from .config import height
    from .mosaic import Mosaic
    from .street import plane_rows
    rc = cfg["layout"]["register"]
    planes = [p for p in cfg["mosaic"]["planes"] if p in set(rc.get("planes") or
                                                             cfg["mosaic"]["planes"])]
    mos = {}

    def mosaic(c, p):
        if (c, p) not in mos:
            stem = root / c / f"mosaic_{p}"
            mos[(c, p)] = Mosaic.load(stem) if stem.with_suffix(".png").exists() else None
        return mos[(c, p)]

    def imaged(c):  # planes imaging a parked row
        pr = plane_rows(rows, rev[c])
        return [p for p in planes if p in pr]

    strips = {}

    def strip(c, p, f, tag):
        key = (c, p, tag)
        if key not in strips:
            m = mosaic(c, p)
            strips[key] = render(m, height(cfg, c), f, L[c], rc["y_m"]) if m is not None else None
        return strips[key]

    def corr(c, v):
        """Registration of pass c (initial map) onto placed pass v (final map)."""
        out = []
        for p in set(imaged(c)) & set(imaged(v)):
            a, b = strip(v, p, final[v], "final"), strip(c, p, maps[c], "init")
            if a is None or b is None:
                continue
            d = register(a, b, rc["win_m"], rc["step_m"], rc["search_m"], rc["band_m"],
                         rc["y_m"][1])
            if not len(d):
                continue
            d = smooth(d, rc["min_peak"])
            xc = np.arange(0.0, L[c], 0.05)
            u = maps[c](xc)
            o = np.argsort(u)
            d["x_clip"] = np.interp(d.x_cur.values, u[o], xc[o])
            out.append(d.assign(plane=p))
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

    final, diag, comp = {}, {}, {}
    cache = {}
    for way in sorted(set(rev[c] for c in clips), key=lambda w: w != rev[ref]):
        todo = [c for c in clips if rev[c] == way]
        while todo:
            root_c = ref if ref in todo else max(todo, key=lambda c: scores.get(c, -1.0))
            final[root_c], comp[root_c] = maps[root_c], root_c
            diag[root_c] = {"registration": "anchor" if root_c != ref else "reference"}
            todo.remove(root_c)
            placed = [root_c]
            while todo:
                best = None
                for c in todo:
                    for v in placed:
                        if (c, v) not in cache:
                            cache[(c, v)] = corr(c, v)
                        d = cache[(c, v)]
                        n = int(d.ok.sum()) if len(d) else 0
                        if n >= rc["min_ok"] and (best is None or n > best[0]):
                            best = (n, c, v)
                if best is None:
                    break
                n, c, v = best
                d = cache[(c, v)]
                ok = d[d.ok]
                final[c], comp[c] = fit_map(ok.x_clip.values, ok.x_anchor.values), comp[v]
                res = ok.x_anchor.values - final[c](ok.x_clip.values)
                diag[c] = {"registration": "strips", "registered_to": v, "windows": len(d),
                           "ok": n, "planes": sorted(ok.plane.unique().tolist()),
                           "residual_mad_m": round(float(np.median(np.abs(res))), 2),
                           "change_vs_previous_max_m": round(float(np.abs(
                               final[c](ok.x_clip.values) - maps[c](ok.x_clip.values)).max()), 1)}
                todo.remove(c)
                placed.append(c)
    for c in clips:
        if c not in diag:
            diag[c] = {"registration": "none"}
    # components without the reference: pooled occupancy against the reference component
    ref_comp = comp[ref]
    for r0 in sorted(set(comp.values()) - {ref_comp}):
        members = [c for c in clips if comp[c] == r0]
        A = _pooled(kept, final, [c for c in clips if comp[c] == ref_comp], rows, rev)
        B = _pooled(kept, final, members, rows, rev)
        if not len(A) or not len(B):
            continue
        o = min(A.xa_start_m.min(), B.xa_start_m.min())
        La = max(A.xa_end_m.max(), B.xa_end_m.max()) - o
        a = A.assign(x_start_m=A.xa_start_m - o, x_end_m=A.xa_end_m - o)
        b = B.assign(xa_start_m=B.xa_start_m - o, xa_end_m=B.xa_end_m - o)
        pairs = [(r, r) for r in rows]
        cen, sh = local_shifts(a, b, La, pairs, max_shift=rc["pooled_max_shift_m"])
        ok = np.isfinite(sh)
        if ok.sum() < 2:
            continue
        from scipy.ndimage import median_filter
        corr_map = AxisMap(knots=tuple(cen[ok] + o),
                           shifts=tuple(median_filter(sh[ok], size=3, mode="mirror")))
        for c in members:
            final[c] = MapChain((final[c], corr_map))
            diag[c]["pooled_with"] = members
            diag[c]["pooled_shift_m"] = [round(float(np.min(corr_map.shifts)), 1),
                                         round(float(np.max(corr_map.shifts)), 1)]
    return final, diag


def _pooled(kept: dict, maps: dict, clips: list, rows: dict, rev: dict) -> pd.DataFrame:
    """Vehicles of several passes on the reference axis, `plane` = the row they stand in."""
    from .street import plane_rows
    out = []
    for c in clips:
        k = kept[c]
        pr = plane_rows(rows, rev[c])
        k = k[k.plane.isin(list(pr))]
        if not len(k):
            continue
        a, b = maps[c].interval(k.x_start_m.values, k.x_end_m.values)
        out.append(pd.DataFrame({"plane": k.plane.map(pr).values, "xa_start_m": a,
                                 "xa_end_m": b}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
