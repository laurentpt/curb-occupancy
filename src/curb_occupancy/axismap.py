"""Along-street maps between passes, aligned without GPS.

A parked row is imaged by one mosaic plane of each pass (`street.plane_pairs`: the near plane
of one pass can be the far plane of a pass riding the other way), possibly in reverse order.
Clip b's along-street axis maps onto clip a's as x_a = offset +- scale*x_b (`AxisMap`, with
local shifts between knots); (offset, scale) maximises the agreement of the vehicle
occupancy signals of all rows jointly (`align`, `fit_axis_map`). `sequence_match` pairs the
vehicles of two passes in order along the street (used by `multiday`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _occ(cars: pd.DataFrame, plane: str, grid: np.ndarray, s0="x_start_m", e0="x_end_m"):
    o = np.zeros(len(grid))
    c = cars[cars.plane == plane]
    for a, b in zip(c[s0], c[e0]):
        o[(grid >= a) & (grid < b)] = 1.0
    return o


@dataclass
class AxisMap:
    """Along-street position of clip b mapped onto clip a's axis:
    x_a = offset -/+ scale * x_b (minus when the passes run in opposite directions), plus a
    piecewise-linear shift (knots in a-coordinates) absorbing the few-% drift of visual odometry.
    """
    scale: float = 1.0
    offset: float = 0.0
    reverse: bool = False
    knots: tuple = ()
    shifts: tuple = ()

    def __call__(self, x_b):
        x = np.asarray(x_b, dtype=float)
        y = self.offset - self.scale * x if self.reverse else self.offset + self.scale * x
        if len(self.knots) >= 2:
            y = y + np.interp(y, self.knots, self.shifts)
        return y

    def interval(self, s_b, e_b):
        """Map [start, end] intervals; returns (start_a, end_a) with start_a <= end_a."""
        a, b = self(s_b), self(e_b)
        return np.minimum(a, b), np.maximum(a, b)

    def to_dict(self):
        return {"scale": self.scale, "offset": self.offset, "reverse": self.reverse,
                "knots": list(map(float, self.knots)), "shifts": list(map(float, self.shifts))}

    @classmethod
    def from_dict(cls, d):
        return cls(d["scale"], d["offset"], d["reverse"], tuple(d.get("knots", ())),
                   tuple(d.get("shifts", ())))


@dataclass
class MapChain:
    """Composition of axis maps: x -> maps[-1](... maps[0](x))."""
    maps: tuple

    def __call__(self, x):
        y = np.asarray(x, dtype=float)
        for m in self.maps:
            y = m(y)
        return y

    def interval(self, s_b, e_b):
        a, b = self(s_b), self(e_b)
        return np.minimum(a, b), np.maximum(a, b)

    def to_dict(self):
        return {"chain": [m.to_dict() for m in self.maps]}


def map_from_dict(d):
    return MapChain(tuple(map_from_dict(v) for v in d["chain"])) if "chain" in d \
        else AxisMap.from_dict(d)


def align(ca: pd.DataFrame, cb: pd.DataFrame, La: float, Lb: float, reverse: bool,
          pairs, step: float = 0.25, scales=np.linspace(0.92, 1.08, 33)) -> dict:
    """Global (scale, offset) maximising the overlap of vehicle occupancy signals of the two
    passes, summed over the rows both image. `pairs`: (plane of a, plane of b[, row]) per row
    (`street.plane_pairs`)."""
    ga = np.arange(0, La, step)
    best = None
    pairs = [(p[0], p[1]) for p in pairs]
    sig_a = {pa: _occ(ca, pa, ga) for pa, _ in pairs}
    for s in scales:
        gb = np.arange(0, Lb * s, step)  # b axis rescaled
        cbs = cb.copy()
        cbs["x_start_m"], cbs["x_end_m"] = cb.x_start_m * s, cb.x_end_m * s
        sig_b = {pb: _occ(cbs, pb, gb) for _, pb in pairs}
        if reverse:
            sig_b = {p: v[::-1] for p, v in sig_b.items()}
        score = None
        for pa, pb in pairs:
            cc = np.correlate(np.pad(sig_a[pa], (len(gb), len(gb))), sig_b[pb], mode="valid")
            score = cc if score is None else score + cc
        k = int(np.argmax(score))
        lag = (k - len(gb)) * step  # a-position of the first sample of (possibly reversed) b
        val = float(score[k])
        if best is None or val > best[0]:
            best = (val, s, lag, len(gb) * step)
    val, s, lag, Lbs = best
    # reversed: b index j maps to a = lag + j*step and b position x_b*s = Lbs - j*step
    offset = lag + Lbs if reverse else lag
    return {"scale": float(s), "offset_m": float(offset), "score_m": float(val * step)}


def local_shifts(ca, cb_mapped, La, pairs, win=120.0, hop=30.0, max_shift=15.0, step=0.25):
    """Windowed refinement of the global map: per window (in a-coords) the shift of b that
    maximises occupancy overlap for both rows. `cb_mapped` has xa_start_m/xa_end_m. Returns
    (centres, shifts) with NaN where a window has too little occupancy to decide."""
    grid = np.arange(-max_shift, La + max_shift, step)
    pairs = [(p[0], p[1]) for p in pairs]
    sig_a = {pa: _occ(ca, pa, grid) for pa, _ in pairs}
    sig_b = {pb: _occ(cb_mapped, pb, grid, "xa_start_m", "xa_end_m") for _, pb in pairs}
    K = int(max_shift / step)
    centres, shifts = [], []
    for c in np.arange(win / 2, La - win / 2 + 1e-6, hop):
        m = (grid >= c - win / 2) & (grid < c + win / 2)
        idx = np.where(m)[0]
        best, bk = 0.0, None
        for k in range(-K, K + 1):
            sc = 0.0
            for pa, pb in pairs:
                j = (idx - k).clip(0, len(grid) - 1)
                sc += float((sig_a[pa][idx] * sig_b[pb][j]).sum())
            if sc > best:
                best, bk = sc, k
        enough = best * step > 8.0  # at least ~2 car lengths of agreeing occupancy
        centres.append(c)
        shifts.append(bk * step if (enough and bk is not None) else np.nan)
    return np.array(centres), np.array(shifts)


def fit_axis_map(ca: pd.DataFrame, cb: pd.DataFrame, La: float, Lb: float,
                 reverse: bool, pairs) -> tuple[AxisMap, dict]:
    """Global alignment + median-smoothed local shifts -> AxisMap (b -> a), diagnostics."""
    from scipy.ndimage import median_filter

    al = align(ca, cb, La, Lb, reverse, pairs)
    am = AxisMap(al["scale"], al["offset_m"], reverse)
    cbm = cb.assign(xa_start_m=0.0, xa_end_m=0.0)
    cbm["xa_start_m"], cbm["xa_end_m"] = am.interval(cb.x_start_m, cb.x_end_m)
    cen, sh = local_shifts(ca, cbm, La, pairs)
    ok = np.isfinite(sh)
    if ok.sum() >= 2:
        am.knots = tuple(cen[ok])
        am.shifts = tuple(median_filter(sh[ok], size=3, mode="mirror"))  # nearest keeps an edge outlier
    return am, {**al, "local_centres_m": cen.tolist(), "local_shifts_m": sh.tolist()}


def sequence_match(A: pd.DataFrame, B: pd.DataFrame, xa: np.ndarray, xb: np.ndarray,
                   max_dx: float = 8.0, gap: float = 1.6):
    """Order-preserving alignment (Needleman-Wunsch) of two along-street vehicle sequences.

    Match cost: |dx|/3 m + |dlength|/0.8 m + dE(Lab)/30; pairs further than max_dx apart are
    not allowed. A gap (cost `gap`) is a vehicle present in only one pass (arrived/left, or a
    miss). Returns list of (i, j, dx) index pairs into A, B (positional).
    """
    n, m = len(A), len(B)
    la, lb = A.length_m.values, B.length_m.values
    ca = A[["lab_L", "lab_a", "lab_b"]].values if "lab_L" in A else np.zeros((n, 3))
    cbb = B[["lab_L", "lab_a", "lab_b"]].values if "lab_L" in B else np.zeros((m, 3))
    D = np.full((n + 1, m + 1), np.inf)
    D[0, :] = gap * np.arange(m + 1)
    D[:, 0] = gap * np.arange(n + 1)
    bt = np.zeros((n + 1, m + 1), np.int8)  # 0 diag, 1 up (gap in B), 2 left (gap in A)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dx = abs(xa[i - 1] - xb[j - 1])
            c = np.inf
            if dx < max_dx:
                dE = np.nan_to_num(np.linalg.norm(ca[i - 1] - cbb[j - 1]), nan=30.0)
                c = dx / 3 + abs(la[i - 1] - lb[j - 1]) / 0.8 + dE / 30
            opts = (D[i - 1, j - 1] + c, D[i - 1, j] + gap, D[i, j - 1] + gap)
            k = int(np.argmin(opts))
            D[i, j], bt[i, j] = opts[k], k
    i, j, pairs = n, m, []
    while i > 0 and j > 0:
        k = bt[i, j]
        if k == 0:
            pairs.append((i - 1, j - 1, float(xa[i - 1] - xb[j - 1])))
            i, j = i - 1, j - 1
        elif k == 1:
            i -= 1
        else:
            j -= 1
    return pairs[::-1]
