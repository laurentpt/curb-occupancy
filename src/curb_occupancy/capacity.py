"""Area capacity, block face by block face: measured where a street run covered the block,
estimated elsewhere from the measured spaces per metre of curb.

The guideline capacity of a block face needs its driveways, hydrants and signs, i.e. a pass
along it. For the faces nobody rode, the area stage extrapolates: rate = measured spaces /
measured curb length over the sampled faces (the faces of the `blocks` list that a run
covered), estimate = rate x the face's OpenStreetMap length (centreline, intersection to
intersection: the same span as the runs' `row_length_m`). The area total carries a
prediction interval from the ratio model (residual variance proportional to length, faces
as units, t with n - 1 df); it treats the sampled faces as a random draw, which a chosen
sample is not.

    # area yaml
    capacity:
      osm_json: workdir/external/osm/<extract>.json
      grid_north_deg: -58        # side names on the city grid (NDG 'north' = bearing -58)
      blocks:                    # both curbs of every block of `street` between two cross
        - [Avenue Coronation, Avenue Somerled, Avenue de Monkland]   # streets, split at each
        - [Avenue Belmore, Avenue Somerled, Rue de Terrebonne]       # named crossing between
      interval: 0.90
      note: {en: ..., fr: ...}   # optional: scope choices, shown in the map's method text
      restrictions:              # optional: large signed restrictions on estimated faces
        - {street: Avenue Benny, from: Rue de Terrebonne, to: Avenue de Monkland,
           side: west side, restricted_m: 165}

Streets of the area that are not in `blocks` (e.g. the main street) count as measured as
they are (every covered block of their runs); runs' blocks of listed streets that match no
listed face (stubs past the area's edge) stay out.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import osm as O
from .ride import side_label
from .street import local_xy

DRAW_OFFSET_M = 5.0  # estimated faces: drawn this far from the centreline ...
TRIM_M = 9.0         # ... and this far short of each intersection


def _fold(s) -> str:
    return str(s or "").casefold()


def osm_faces(osm: dict, blocks: list, grid_north: float) -> list[dict]:
    """Block faces of the listed street spans: street, from, to, side, length_m, lat/lon
    centreline (from -> to) and the drawn curb line."""
    lat0 = float(np.mean([g["lat"] for e in O._ways(osm) for g in e["geometry"]]))
    lon0 = float(np.mean([g["lon"] for e in O._ways(osm) for g in e["geometry"]]))
    out = []
    for street, a, b in blocks:
        ways, name = O.street_ways(osm, street)
        span = None
        for comp in O.components(ways):
            path = O.chain(comp)
            if path is None:
                continue
            xs = [c for c in O.crossings(osm, comp, path)
                  if c["kind"] == "intersection" and c["name"] and _fold(c["name"]) != _fold(name)]
            ia = [c["i"] for c in xs if _fold(c["name"]) == _fold(a)]
            ib = [c["i"] for c in xs if _fold(c["name"]) == _fold(b)]
            if ia and ib:
                i0, i1 = min(((p, q) for p in ia for q in ib), key=lambda t: abs(t[0] - t[1]))
                span = (comp, path, xs, i0, i1)
                break
        if span is None:
            raise ValueError(f"capacity.blocks: no {street} span between {a!r} and {b!r} in OSM")
        comp, path, xs, i0, i1 = span
        step = 1 if i1 > i0 else -1
        # split at each named crossing between the ends (first node of each name)
        cuts, seen = [(i0, a)], {_fold(a), _fold(b)}
        for c in sorted(xs, key=lambda c: c["i"] * step):
            if (c["i"] - i0) * step > 0 and (i1 - c["i"]) * step > 0 and _fold(c["name"]) not in seen:
                cuts.append((c["i"], c["name"]))
                seen.add(_fold(c["name"]))
        cuts.append((i1, b))
        coords = O._node_coords(comp)
        for (j0, n0), (j1, n1) in zip(cuts[:-1], cuts[1:]):
            ll = np.array([coords[path[k]] for k in range(j0, j1 + step, step)])
            P = local_xy(ll[:, 0], ll[:, 1], lat0, lon0)
            L = float(np.hypot(*np.diff(P, axis=0).T).sum())
            d = P[-1] - P[0]
            bearing = float(np.degrees(np.arctan2(d[0], d[1])) % 360)
            for hand, sgn in (("left", 1), ("right", -1)):
                out.append({"street": name, "from": n0, "to": n1,
                            "side": side_label(bearing, hand, grid_north), "length_m": round(L, 1),
                            "line": _to_ll(_offset(P, sgn * DRAW_OFFSET_M, TRIM_M), lat0, lon0)})
    return out


def _offset(P: np.ndarray, d: float, trim: float) -> np.ndarray:
    """Polyline shifted d metres to its left (d < 0: right), trimmed at both ends."""
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    if s[-1] <= 2 * trim + 1:
        trim = max(0.0, (s[-1] - 1) / 2)
    t = np.r_[trim, s[(s > trim) & (s < s[-1] - trim)], s[-1] - trim]
    Q = np.stack([np.interp(t, s, P[:, 0]), np.interp(t, s, P[:, 1])], -1)
    tan = np.gradient(Q, axis=0)
    tan /= np.maximum(np.hypot(*tan.T), 1e-9)[:, None]
    return Q + d * np.stack([-tan[:, 1], tan[:, 0]], -1)


def _to_ll(P: np.ndarray, lat0: float, lon0: float) -> list:
    R = 6371000.0
    lat = lat0 + np.degrees(P[:, 1] / R)
    lon = lon0 + np.degrees(P[:, 0] / (R * np.cos(np.radians(lat0))))
    return [[round(float(x), 7), round(float(y), 7)] for x, y in zip(lon, lat)]


def _restrict(faces: list, items: list) -> None:
    """`restricted_m` per face: curb of an estimated face under a large signed restriction
    in force at the survey time (school zone, sports-centre drop-off...), read from the sign
    data by hand; the rate applies to the rest. Each item must name one estimated face."""
    for f in faces:
        f["restricted_m"] = 0.0
    for it in items:
        hit = [f for f in faces if _fold(f["street"]) == _fold(it["street"])
               and _fold(f["side"]) == _fold(it["side"])
               and {_fold(f["from"]), _fold(f["to"])} == {_fold(it["from"]), _fold(it["to"])}]
        if len(hit) != 1:
            raise ValueError(f"capacity.restrictions: {len(hit)} faces match {it}")
        f = hit[0]
        if "measured" in f:
            raise ValueError(f"capacity.restrictions: {it} is a measured face (its run has the signs)")
        if not 0 < it["restricted_m"] <= f["length_m"]:
            raise ValueError(f"capacity.restrictions: restricted_m out of range in {it}")
        f["restricted_m"] = float(it["restricted_m"])


def measured_blocks(B: pd.DataFrame) -> pd.DataFrame:
    """One row per street x block x row that some pass covered: capacity (in the guideline
    window when an in-window pass covered it, else any covering pass; the most common value),
    measured curb length."""
    c = B[B.covered.astype(bool)].copy()
    c["w"] = c.in_window.astype(bool)
    rows = []
    for (key, blk, row), g in c.groupby(["key", "block_id", "row"], sort=False):
        g = g[g.w] if g.w.any() else g
        rows.append({"key": key, "street": g.street.iloc[0], "side": g.side.iloc[0],
                     "block_id": blk, "row": row, "from": g["from"].iloc[0], "to": g["to"].iloc[0],
                     "capacity": int(g.capacity.mode().iloc[0]),
                     "length_m": float(g.row_length_m.median()),
                     "n_passes": int(len(g)), "in_window": bool(g.w.any())})
    return pd.DataFrame(rows)


def build(cfg: dict, B: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Faces table (measured + estimated) and the summary. B = the area's blocks_by_pass with
    `key` and `row_length_m`."""
    from scipy import stats
    M = measured_blocks(B)
    faces = osm_faces(json.loads(Path(cfg["osm_json"]).read_text()), cfg.get("blocks") or [],
                      float(cfg.get("grid_north_deg", 0.0)))
    listed = {_fold(f["street"]) for f in faces}
    M["face"] = None
    for i, f in enumerate(faces):
        hit = M[(M.street.map(_fold) == _fold(f["street"])) & (M.side == f["side"])
                & (M[["from", "to"]].apply(lambda r: {_fold(r["from"]), _fold(r["to"])}, axis=1)
                   == {_fold(f["from"]), _fold(f["to"])})]
        if len(hit) > 1:
            raise ValueError(f"capacity: several measured blocks on one face {f}")
        if len(hit):
            M.loc[hit.index, "face"] = i
            f["measured"] = hit.index[0]
    # measured faces of the listed streets (the sample) and the area's other streets
    keep = M.face.notna() | ~M.street.map(_fold).isin(listed)
    M = M[keep]
    S = M[M.face.notna()]
    if len(S) < 2:
        raise ValueError("capacity: fewer than 2 measured faces among the listed blocks")
    R = S.capacity.sum() / S.length_m.sum()
    n = len(S)
    sig2 = float((((S.capacity - R * S.length_m) ** 2) / S.length_m).sum() / (n - 1))
    est = [f for f in faces if "measured" not in f]
    _restrict(faces, cfg.get("restrictions") or [])
    # unmeasured parkable-candidate length: block length minus large signed restrictions
    Lu, Ls = sum(f["length_m"] - f["restricted_m"] for f in est), float(S.length_m.sum())
    level = float(cfg.get("interval", 0.90))
    t = float(stats.t.ppf(0.5 + level / 2, n - 1))
    se = float(np.sqrt(sig2 * Lu * (Lu + Ls) / Ls))
    meas_total = int(M.capacity.sum())
    total = meas_total + R * Lu

    rows = []
    for r in M.to_dict("records"):
        rows.append({"street": r["street"], "side": r["side"], "from": r["from"], "to": r["to"],
                     "length_m": r["length_m"], "spaces": r["capacity"], "kind": "measured",
                     "key": r["key"], "block_id": r["block_id"], "row": r["row"],
                     "sample": not pd.isna(r["face"])})
    for f in est:
        rows.append({"street": f["street"], "side": f["side"], "from": f["from"], "to": f["to"],
                     "length_m": f["length_m"], "restricted_m": f["restricted_m"],
                     "spaces": int(round(R * (f["length_m"] - f["restricted_m"]))),
                     "kind": "estimated", "key": None, "block_id": None, "row": None,
                     "sample": False, "line": f["line"]})
    F = pd.DataFrame(rows)
    F["restricted_m"] = F.restricted_m.fillna(0.0)
    by_street = (F.assign(m=np.where(F.kind == "measured", F.spaces, 0),
                          e=np.where(F.kind == "estimated", R * (F.length_m - F.restricted_m), 0))
                 .groupby("street", sort=False)
                 .agg(measured=("m", "sum"), estimated=("e", "sum"),
                      faces_measured=("kind", lambda k: int((k == "measured").sum())),
                      faces_estimated=("kind", lambda k: int((k == "estimated").sum())))
                 .reset_index())
    summary = {
        "rate_per_km": round(1000 * R, 1), "m_per_space": round(1 / R, 2),
        "sample": {"faces": n, "length_m": round(Ls), "spaces": int(S.capacity.sum()),
                   "rate_cv": round(float(np.std(S.capacity / S.length_m, ddof=1)
                                          / np.mean(S.capacity / S.length_m)), 3)},
        "measured": {"faces": int(len(M)), "spaces": meas_total},
        "estimated": {"faces": len(est), "length_m": round(sum(f["length_m"] for f in est)),
                      "restricted_m": round(sum(f["restricted_m"] for f in est)),
                      "faces_restricted": sum(f["restricted_m"] > 0 for f in est),
                      "spaces": round(R * Lu, 1)},
        # whole spaces, half up (the page and the report show the same figures)
        "total": int(np.floor(total + 0.5)), "interval": level,
        "low": int(np.floor(total - t * se + 0.5)), "high": int(np.floor(total + t * se + 0.5)),
        "streets": [{k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()}
                    for r in by_street.to_dict("records")],
    }
    return F, summary
