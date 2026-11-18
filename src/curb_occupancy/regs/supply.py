"""Supply stage: regulations on the street axis -> curb status per parking row -> guideline
capacity per block and row at each pass's date/time -> occupancy.

City-independent. Regulations come from the sources listed in `regulations.sources` (city
adapters, manual zones; see `regs/__init__.py`), in the common schema of `regs/rules.py`.
Capacity per block and row = floor(length / stall_m) summed over the stretches that are
parkable at the pass's date/time: not inside an active no-stopping / no-parking / reserved
zone, an intersection or point clearance (hydrants), nor a driveway; time-limited and permit
stretches count per the guideline profile (`regs/guideline.py`).

Observations check the regulations: a no-stopping zone with a car mostly inside it on
>= `dispute_passes` dates is `disputed` (open data has errors). With `trust_observations`
disputed zones are treated as legal; `capacity_if_signs_right` gives the other reading.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .guideline import guideline, occupancy_class, slot_of
from .rules import PARKABLE, POINT_COLS, PRIORITY, ZONE_COLS, Regs

DEFAULTS = {"sources": [], "clearances_m": {}, "max_lateral_m": 16.0, "max_dist_m": 30.0,
            "dispute_passes": 2, "trust_observations": True}


# ---------------------------------------------------------------------------- context

@dataclass
class Context:
    """What a source needs: the street axis (projection of lat/lon), the layout (rows present,
    intersections, extent) and the observed parked cars (for offset fits)."""
    cfg: dict
    lay: object
    rc: dict
    cars: pd.DataFrame | None
    _P: np.ndarray            # street-axis polyline (local metres)
    _X: np.ndarray            # its street-axis positions
    origin: tuple             # (lat0, lon0) of the local frame
    bbox: tuple               # (lat_min, lat_max, lon_min, lon_max) of the street
    info: dict = field(default_factory=dict)

    def project(self, lat, lon) -> np.ndarray:
        """(N,3): distance to the street axis, axis position, lateral (+ = left)."""
        from ..street import local_xy, project_polyline
        pts = local_xy(lat, lon, *self.origin)
        return project_polyline(pts, self._P, self._X) if len(pts) else np.zeros((0, 3))

    def near(self, df: pd.DataFrame, latc: str, lonc: str, pad: float = 4e-4) -> pd.DataFrame:
        """Rows of df within the street's bounding box (+ pad degrees), projected: adds dist_m,
        x_ref, lat_m and keeps those within max_dist_m of the axis."""
        la0, la1, lo0, lo1 = self.bbox
        d = df[df[latc].between(la0 - pad, la1 + pad) & df[lonc].between(lo0 - pad, lo1 + pad)]
        d = d.copy()
        pr = self.project(d[latc].values, d[lonc].values)
        d["dist_m"], d["x_ref"], d["lat_m"] = pr.T if len(d) else ([], [], [])
        return d[d.dist_m < self.rc["max_dist_m"]].reset_index(drop=True)

    def curb_row(self, x, lat) -> np.ndarray:
        """Parking row whose curb a sidewalk point (axis position x, lateral offset lat) is on:
        a row present at x whose curb is on that side (right: lat < 0), within max_lateral_m.
        None where there is none."""
        x, lat = np.asarray(x, float), np.asarray(lat, float)
        out = np.full(len(x), None, dtype=object)
        ok = np.abs(lat) < self.rc["max_lateral_m"]
        for r, rc in self.lay.rows.items():
            side = lat < 0 if rc["curb"] == "right" else lat > 0
            m = ok & side & self.lay.present(r, x) & (out == None)  # noqa: E711
            out[m] = r
        return out


def _context(cfg: dict, lay, rc: dict, cars) -> Context:
    """Regulations need geography: the layout's street axis (OSM centreline or the reference
    pass's GPS track)."""
    ax = lay.axis
    if ax is None:
        raise RuntimeError(f"the layout has no street axis (reference pass {lay.reference} has "
                           "no GPS fix): regulations cannot be placed")
    return Context(cfg, lay, rc, cars, ax.P, ax.X, ax.origin, ax.bbox(*lay.extent_m, 30.0),
                   info={"axis": ax.kind, **{k: ax.meta[k] for k in ("name", "gps_lag_s", "fit")
                                              if k in ax.meta}})


# ---------------------------------------------------------------------------- curb status

def curb_status(grid: np.ndarray, zones: pd.DataFrame, rules: dict, ints: pd.DataFrame,
                points: pd.DataFrame, when: datetime, clearances_m: dict,
                trust_observations: bool = True) -> np.ndarray:
    """Status of each curb bin of one row at `when`: 'legal', 'limited', 'permit' or the
    reason it is not parkable (intersection, no_stopping, no_parking, reserved, hydrant,
    driveway, ...). `zones`/`points` of that row only."""
    st = np.full(len(grid), "legal", dtype=object)

    def put(a, b, s):
        m = (grid >= a) & (grid < b)
        cur = np.array([PRIORITY[v] for v in st[m]])
        st[m] = np.where(cur < PRIORITY[s], s, st[m])
    for z in zones.itertuples():
        if getattr(z, "disputed", False) and trust_observations:
            continue
        if z.kind in PRIORITY and rules[z.rule].active(when):
            put(z.x0, z.x1, z.kind)
    c = clearances_m.get("intersection", 0.0)
    for r in ints.itertuples():
        put(r.x0_m - c, r.x1_m + c, "intersection")
    for r in points.itertuples():
        c = clearances_m.get(r.kind, 0.0)
        if r.kind in PRIORITY:
            put(r.x_ref - c, r.x_ref + c, r.kind)
    return st


def runs(mask: np.ndarray, grid: np.ndarray, step: float):
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return [(float(grid[a]), float(grid[b - 1] + step))
            for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0])]


def parkable(st: np.ndarray, g: dict) -> np.ndarray:
    return (st == "legal") | ((st == "limited") & g["count_limited"]) | \
        ((st == "permit") & g["count_permit"])


def capacity(mask: np.ndarray, grid: np.ndarray, step: float, stall_m: float) -> tuple:
    """(spaces, legal length): floor(length / stall) per contiguous parkable stretch."""
    legal = runs(mask, grid, step)
    return int(sum(np.floor((e - s) / stall_m) for s, e in legal)), sum(e - s for s, e in legal)


def guideline_blocks(ints: pd.DataFrame, extent) -> pd.DataFrame:
    """Blocks between consecutive cross streets, split at the intersection centres so they
    partition the street: an intersection's own curb is excluded by its clearance only for
    the rows whose curb it joins (`cuts`), and a curb that runs past a T-junction on the
    other side still belongs to a block."""
    ints = ints.sort_values("centre_m")
    c = list(ints.centre_m)
    names = list(ints["name"].fillna("")) if "name" in ints else [""] * len(ints)
    # a cross street just outside the passes (they started/stopped short of the corner) ends
    # the first/last block instead of the passes' extent
    i0 = max([i for i, v in enumerate(c) if v <= extent[0]], default=None)
    i1 = min([i for i, v in enumerate(c) if v >= extent[1]], default=None)
    inner = [i for i, v in enumerate(c) if extent[0] < v < extent[1]]
    e = ([(c[i0], names[i0])] if i0 is not None else [(extent[0], None)]) \
        + [(c[i], names[i]) for i in inner] \
        + ([(c[i1], names[i1])] if i1 is not None else [(extent[1], None)])
    rows = []
    for k in range(len(e) - 1):  # G00 = up to the first cross street inside the passes
        (a, na), (b, nb) = e[k], e[k + 1]
        if b - a > 10:
            rows.append({"block_id": f"G{k:02d}", "x0_m": round(a, 1), "x1_m": round(b, 1),
                         "partial": na is None or nb is None,
                         "from": na or "", "to": nb or ""})
    return pd.DataFrame(rows)


def cuts(ints: pd.DataFrame, curb: str) -> pd.DataFrame:
    """Intersections that interrupt a row on `curb`: those joining from that side (all when
    the side is not known, e.g. intersections seen by the cameras only)."""
    if "sides" not in ints or not len(ints):
        return ints
    sd = ints.sides.fillna("unknown").astype(str)
    return ints[sd.str.contains(curb) | sd.eq("unknown")]


def in_roadway(cars: pd.DataFrame, ints: pd.DataFrame, rows: dict) -> np.ndarray:
    """Vehicles whose centre lies on a cross street's roadway, on a curb that street cuts:
    not parked (turning or crossing traffic, or a detection of something else). Never counted;
    the intersection clearance beyond the roadway edges is curb, where vehicles count."""
    m = np.zeros(len(cars), dtype=bool)
    if not len(cars) or not len(ints):
        return m
    x, row = cars.xr_mid_m.values, cars["row"].values
    for r, spec in rows.items():
        for i in cuts(ints, spec["curb"]).itertuples():
            m |= (row == r) & (x >= i.x0_m) & (x <= i.x1_m)
    return m


# ---------------------------------------------------------------------------- observations

def overlap_frac(cars: pd.DataFrame, x0: float, x1: float) -> np.ndarray:
    ins = (np.minimum(cars.xr_end_m, x1) - np.maximum(cars.xr_start_m, x0)).clip(lower=0)
    return (ins / (cars.xr_end_m - cars.xr_start_m)).values


def cars_inside(cars: pd.DataFrame, x0: float, x1: float, min_m: float = 2.0) -> pd.DataFrame:
    """Vehicles mostly (> 50 % and >= min_m of their length) inside [x0, x1]: a 1-2 m
    overlap is within the sign-position error."""
    ins = (np.minimum(cars.xr_end_m, x1) - np.maximum(cars.xr_start_m, x0)).clip(lower=0)
    return cars[(overlap_frac(cars, x0, x1) > 0.5) & (ins >= min_m).values]


def disputes(zones: pd.DataFrame, cars: pd.DataFrame | None, min_dates: int) -> np.ndarray:
    """No-stopping zones contradicted by observation: on >= min_dates different dates a
    vehicle of the zone's row lies mostly inside (passes the same evening see the same
    cars)."""
    out = np.zeros(len(zones), bool)
    if cars is None or not len(cars):
        return out
    by = "date" if "date" in cars else "clip"
    for i, z in enumerate(zones.itertuples()):
        if z.kind != "no_stopping":
            continue
        k = cars[cars["row"] == z.row]
        out[i] = cars_inside(k, z.x0, z.x1)[by].nunique() >= min_dates
    return out


def fit_offset(build, cars: pd.DataFrame, search: float, step: float = 1.0,
               tol_m: float = 2.0, tol_frac: float = 0.05):
    """Residual along-axis offset of geolocated regulations: the dx minimising parked-car
    length inside no-stopping zones, `build(dx)` -> zones (with `row`) of the shifted sources.
    No-stopping zones are anchored on cross streets, so the minimum is sharp on the outside
    but often flat over a few metres: the centre of the contiguous plateau within
    max(tol_m, tol_frac x minimum) of the minimum is returned, not its arbitrary argmin.
    Returns (dx, fraction of parked length inside no-stopping zones at dx)."""
    tot = float((cars.xr_end_m - cars.xr_start_m).sum())
    dxs = np.arange(-search, search + 0.01, step)
    ovs = []
    for dx in dxs:
        z = build(dx)
        z = z[z.kind == "no_stopping"]
        ov = 0.0
        for r in cars.itertuples():
            zr = z[z.row == r.row]
            ov += min(r.xr_end_m - r.xr_start_m,
                      float(np.clip(np.minimum(r.xr_end_m, zr.x1.values)
                                    - np.maximum(r.xr_start_m, zr.x0.values), 0, None).sum()))
        ovs.append(ov)
    ovs = np.array(ovs)
    k = int(np.argmin(ovs))
    ok = ovs <= ovs[k] + max(tol_m, tol_frac * ovs[k])
    lo, hi = k, k
    while lo > 0 and ok[lo - 1]:
        lo -= 1
    while hi < len(ok) - 1 and ok[hi + 1]:
        hi += 1
    dx = float(np.round((dxs[lo] + dxs[hi]) / 2 / step) * step)
    return dx, float(ovs[np.argmin(np.abs(dxs - dx))]) / max(tot, 1e-6)


def pass_time(root: Path, clip: str) -> datetime:
    """Local wall-clock start time of a pass. GoPro writes local time in creation_time
    despite the 'Z' suffix."""
    v = json.loads((root / clip / "video.json").read_text())
    return datetime.fromisoformat(v["creation_time"][:19])


# ---------------------------------------------------------------------------- stage

# regulation source types: `regulations.sources[].type` -> module with `load(sc, ctx) -> Regs`;
# optional: `fetch_source(sc, force)` (`curb-occupancy fetch`), `ATTRIBUTION` {en, fr} (map pages),
# `POLES = True` (sign poles: site checks of set-aside panels on the review map)
SOURCES = {"montreal": "curb_occupancy.adapters.montreal",
           "manual": "curb_occupancy.regs.manual",
           "osm": "curb_occupancy.regs.osm"}


def source_module(kind: str):
    import importlib
    if kind not in SOURCES:
        raise ValueError(f"regulation source type {kind!r}: known {sorted(SOURCES)}")
    return importlib.import_module(SOURCES[kind])


def source_name(sc: dict) -> str:
    return sc.get("name") or sc["type"]


def attributions(cfg: dict) -> list[dict]:
    """Data credits of a config's regulation sources ({en, fr} each, once per type)."""
    out = []
    for kind in dict.fromkeys(sc.get("type") for sc in (cfg.get("regulations") or {})
                              .get("sources", [])):
        a = getattr(source_module(kind), "ATTRIBUTION", None)
        if a and a not in out:
            out.append(a)
    return out


def supply(cfg: dict) -> dict | None:
    """Run-level stage. Writes <out_dir>/supply/."""
    from ..config import _merge
    from ..layout import Layout
    rc = _merge(DEFAULTS, cfg.get("regulations") or {})
    if not rc["sources"]:
        return None
    root = Path(cfg["out_dir"])
    lay = Layout.load(root / "layout")
    if lay is None:
        return None
    g = guideline(cfg)
    out = root / "supply"
    out.mkdir(exist_ok=True)
    f = root / "multiday" / "cars_ref.csv"
    cars = pd.read_csv(f) if f.exists() else None
    ctx = _context(cfg, lay, rc, cars)
    regs: list[tuple[str, Regs]] = []
    for sc in rc["sources"]:
        regs.append((source_name(sc), source_module(sc.get("type")).load(sc, ctx)))
    zones = pd.concat([r.zones.assign(source=n) for n, r in regs if len(r.zones)],
                      ignore_index=True) if any(len(r.zones) for _, r in regs) \
        else pd.DataFrame(columns=ZONE_COLS)
    points = pd.concat([r.points.assign(source=n) for n, r in regs if len(r.points)],
                       ignore_index=True) if any(len(r.points) for _, r in regs) \
        else pd.DataFrame(columns=POINT_COLS)
    rules = {k: v for _, r in regs for k, v in r.rules.items()}
    clear = {}
    for _, r in regs:
        clear = {**r.clearances_m, **clear}  # earlier sources win
    clear = {**clear, **rc["clearances_m"]}
    zones["disputed"] = disputes(zones, cars, rc["dispute_passes"])
    for _, r in regs:
        for name, t in r.tables.items():
            t.to_csv(out / f"{name}.csv", index=False)  # full precision: lat/lon columns
    zones.to_csv(out / "zones.csv", index=False, float_format="%.1f")
    points.to_csv(out / "points.csv", index=False, float_format="%.1f")

    ints = lay.intersections
    blocks = guideline_blocks(ints, lay.extent_m)
    blocks.to_csv(out / "blocks.csv", index=False)
    step = 1.0
    grid = np.arange(lay.extent_m[0], float(lay.extent_m[1]), step)
    mid = grid + step / 2
    present = {r: lay.present(r, mid) for r in lay.rows}
    rows, status = [], {}
    ext = json.loads((root / "multiday" / "multiday.json").read_text())["extent_ref_m"]
    for c in [c["id"] for c in cfg["clips"] if c["id"] in lay.maps]:
        when = pass_time(root, c)
        for r in lay.rows:
            zr, pr = zones[zones["row"] == r], points[points["row"] == r]
            ir = cuts(ints, lay.rows[r]["curb"])
            st = curb_status(grid, zr, rules, ir, pr, when, clear, rc["trust_observations"])
            st[~present[r]] = "absent"
            status[(c, r)] = st
            strict = curb_status(grid, zr, rules, ir, pr, when, clear, False) \
                if zones.disputed.any() else None
            for b in blocks.itertuples():
                m = (grid >= b.x0_m) & (grid < b.x1_m) & present[r]
                if not m.any():
                    continue
                n, leg_m = capacity(m & parkable(st, g), grid, step, g["stall_m"])
                # curb of the block the pass did not see (it started/stopped inside the block,
                # e.g. at the corner): legal metres there under the signs at the pass time
                e0, e1 = (float(v) for v in ext[c])
                un = np.r_[np.arange(b.x0_m, min(e0, b.x1_m), step),
                           np.arange(max(e1, b.x0_m), b.x1_m, step)]
                # only where the row is present (the layout's edge state continues beyond its
                # extent): a stretch without this row (e.g. a turn lane) is no unseen curb
                un = un[lay.present(r, un + step / 2)] if len(un) else un
                unseen = (float(parkable(curb_status(un, zr, rules, ir, pr, when, clear,
                                                     rc["trust_observations"]), g).sum() * step)
                          if len(un) else 0.0)
                rows.append({"clip": c, "when": when.isoformat(timespec="minutes"),
                             "weekday": when.strftime("%a"), "block_id": b.block_id, "row": r,
                             "x0_ref_m": b.x0_m, "x1_ref_m": b.x1_m, "partial": b.partial,
                             "unseen_legal_m": round(unseen, 1),
                             "row_length_m": int(m.sum() * step), "legal_m": round(leg_m, 1),
                             "capacity": n, "capacity_len": round(leg_m / g["stall_m"], 2),
                             "capacity_if_signs_right":
                                 capacity(m & parkable(strict, g), grid, step, g["stall_m"])[0]
                                 if strict is not None else -1,
                             **{f"{k}_m": int(((st == k) & m).sum() * step) for k in
                                ("limited", "permit", "no_stopping", "no_parking", "reserved",
                                 "hydrant", "driveway")}})
    cap = pd.DataFrame(rows)
    cap.to_csv(out / "capacity_by_pass.csv", index=False)
    parts = []
    for r in lay.rows:
        if present[r].any():
            parts.append(pd.DataFrame({"x_ref_m": grid[present[r]], "row": r,
                                       **{f"status_{c}": v[present[r]]
                                          for (c, rr), v in status.items() if rr == r}}))
    pd.concat(parts, ignore_index=True).sort_values(["x_ref_m", "row"]).to_csv(
        out / "curb_status.csv", index=False, float_format="%.1f")
    occ, summ = occupancy(root, cap, grid, status, g, ints, lay.rows)
    st, _, sev = stretches(root, blocks, occ, g, cfg["street"].get("stretches"))
    if summ.get("evening"):
        summ["evening"]["stretches_by_class"] = sev["class"].value_counts().to_dict()
    sl = out / "occupancy_stretches_slots.csv"
    if g.get("slots"):
        summ["slots"] = slot_totals(pd.read_csv(sl)) if sl.stat().st_size > 1 else {}
    elif sl.exists():
        sl.unlink()                                  # slots removed from the config
    res = {"guideline": g, "clearances_m": clear, "axis": ctx.info,
           "trust_observations": rc["trust_observations"],
           "sources": {n: r.info for n, r in regs},
           "disputed_zones": zones[zones.disputed][["row", "x0", "x1", "text", "ref", "source"]]
           .round(1).to_dict("records"),
           "occupancy": summ,
           "rules": {k: v.to_dict() for k, v in rules.items()}}
    (out / "supply.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str))
    return res


def occupancy(root: Path, cap: pd.DataFrame, grid, status: dict, g: dict,
              ints: pd.DataFrame | None = None, rows: dict | None = None):
    """Guideline occupancy per block, row and pass: parked vehicles (all, legal or not) over
    capacity, only for passes covering the whole block (`unseen_legal_m`: legal curb of the
    block outside the pass, under one space); then per block and row over the
    passes in the guideline window: mean occupancy and class (guideline `classes`).
    Vehicles on a cross street's roadway are not parked (`in_roadway`): listed in
    supply/not_parked.csv, not counted."""
    cars = pd.read_csv(root / "multiday" / "cars_ref.csv")
    if ints is not None and rows:
        off = in_roadway(cars, ints, rows)
        cars[off].to_csv(root / "supply" / "not_parked.csv", index=False)
        cars = cars[~off]
    lo_t, hi_t = g["window"]
    rows = []
    for r in cap.itertuples():
        st = status[(r.clip, r.row)]
        covered = r.unseen_legal_m < g["stall_m"]
        k = cars[(cars["clip"] == r.clip) & (cars["row"] == r.row)
                 & cars.xr_mid_m.between(r.x0_ref_m, r.x1_ref_m)]
        illegal = 0
        for c in k.itertuples():
            m = (grid >= c.xr_start_m) & (grid < c.xr_end_m)
            if m.any() and np.mean(~np.isin(st[m], PARKABLE)) > 0.5:
                illegal += 1
        h = datetime.fromisoformat(r.when)
        rows.append({"clip": r.clip, "when": r.when, "date": r.when[:10],
                     "block_id": r.block_id, "row": r.row,
                     "x0_ref_m": r.x0_ref_m, "x1_ref_m": r.x1_ref_m, "covered": covered,
                     "in_window": lo_t <= h.hour + h.minute / 60 < hi_t,
                     "capacity": r.capacity, "n_parked": int(len(k)) if covered else None,
                     "n_not_legal_spot": illegal if covered else None,
                     "occupancy": round(len(k) / r.capacity, 3)
                     if covered and r.capacity > 0 else None})
    occ = pd.DataFrame(rows)
    occ.to_csv(root / "supply" / "occupancy_by_pass.csv", index=False)
    out = {}
    for sel, name in ((occ.in_window, "evening"), (occ.index == occ.index, "all_passes")):
        q = occ[sel & occ.covered & occ.occupancy.notna()]
        if not len(q):
            out[name] = None
            continue
        blk = q.groupby(["block_id", "row"]).agg(n_passes=("clip", "nunique"),
                                                 capacity_mean=("capacity", "mean"),
                                                 parked_mean=("n_parked", "mean"),
                                                 occupancy_mean=("occupancy", "mean")).reset_index()
        tot_cap, tot_n = q.groupby("clip").capacity.sum(), q.groupby("clip").n_parked.sum()
        # the same with capacity = parkable length / stall_m per block (no rounding down per
        # parkable stretch): the rounding rule is a method choice worth ~15 % of capacity on a
        # street with many driveways
        cl = cap.set_index(["clip", "block_id", "row"]).capacity_len
        tot_len = q.set_index(["clip", "block_id", "row"]).index.map(cl).to_series(
            index=q.index).groupby(q["clip"]).sum()
        blk["class"] = blk.occupancy_mean.map(lambda o: occupancy_class(o, g["classes"]))
        blk.round(3).to_csv(root / "supply" / f"occupancy_{name}.csv", index=False)
        n_dates = int(q.date.nunique())
        out[name] = {"passes": sorted(q["clip"].unique()), "n_dates": n_dates,
                     "meets_n_evenings": n_dates >= g["n_evenings"] if name == "evening" else None,
                     "street_occupancy_per_pass": (tot_n / tot_cap).round(3).to_dict(),
                     "street_occupancy_per_pass_length_rule": (tot_n / tot_len).round(3).to_dict(),
                     "capacity_per_pass": tot_cap.astype(int).to_dict(),
                     "capacity_length_rule_per_pass": tot_len.round(1).to_dict(),
                     "blocks_by_class": blk["class"].value_counts().to_dict()}
    return occ, out


def partition(w: np.ndarray, target: float) -> list[tuple[int, int]]:
    """Contiguous groups [i, j) of the weights w whose sums are closest to `target` (least
    squares over the groups; the number of groups is free)."""
    n = len(w)
    cs = np.r_[0.0, np.cumsum(w)]
    best = np.full(n + 1, np.inf)
    best[0] = 0.0
    prev = np.zeros(n + 1, int)
    for j in range(1, n + 1):
        c = best[:j] + (cs[j] - cs[:j] - target) ** 2
        prev[j] = int(np.argmin(c))
        best[j] = c[prev[j]]
    out, j = [], n
    while j:
        out.append((int(prev[j]), j))
        j = prev[j]
    return out[::-1]


def fixed_groups(b: pd.DataFrame, spec: list) -> list[tuple[int, int]]:
    """Blocks [i, j) of each stretch given as [from, to] cross streets (block `from` / `to`
    names, consecutive along the axis; None = the open end). Blocks before the first and
    after the last stretch are in none (partial blocks beyond the study's cross streets)."""
    def name(v):
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else v
    fr, to = [name(v) for v in b["from"]], [name(v) for v in b["to"]]
    out = []
    for n, (a, z) in enumerate(spec):
        if n == 0:
            i = 0 if a is None else next((k for k in range(len(b)) if fr[k] == a), None)
        else:
            i = out[-1][1] if out[-1][1] < len(b) and fr[out[-1][1]] == a else None
        if i is None:
            raise ValueError(f"street.stretches: [{a}, {z}] does not start "
                             + ("at any block" if n == 0 else f"where [{spec[n - 1][0]}, "
                                f"{spec[n - 1][1]}] ends"))
        j = next((k + 1 for k in range(i, len(b)) if to[k] == z), None)
        if j is None:
            raise ValueError(f"street.stretches: no block after {a} ends at {z}")
        out.append((i, j))
    return out


def stretches(root: Path, blocks: pd.DataFrame, occ: pd.DataFrame, g: dict,
              fixed: list | None = None):
    """Reporting unit: consecutive blocks (all rows) grouped to ~`stretch_spaces` spaces, so
    a 3-space block does not read 0 % or 100 % on its own. Capacity per block = mean over the
    guideline-window passes (all passes if none), summed over the rows. Per pass, a stretch
    sums parked and capacity over its covered block x rows with capacity (as the street
    total), and counts when they hold at least half its capacity. Evening = mean over the
    counted window passes of the block x rows covered by all of them (one stretch of curb:
    end blocks reached by only some passes would otherwise average two curb lengths).
    `fixed` ([from, to] cross streets per stretch) freezes the grouping: the partition moves
    when one block gains or loses a space."""
    out = root / "supply"
    q = occ[occ.in_window] if occ.in_window.any() else occ
    w = q.groupby(["clip", "block_id"]).capacity.sum().groupby("block_id").mean()
    b = blocks.sort_values("x0_m").reset_index(drop=True)
    b = b[b.block_id.isin(w.index)].reset_index(drop=True)
    wb = w.reindex(b.block_id).fillna(0).values
    target = g["stretch_spaces"]
    groups = (fixed_groups(b, fixed) if fixed else partition(wb, target) if target > 0
              else [(i, i + 1) for i in range(len(b))])
    st = pd.DataFrame([{"stretch_id": f"S{k + 1:02d}",
                        "blocks": " ".join(b.block_id.iloc[i:j]),
                        # an open-ended partial block at a stretch end has no cross street:
                        # name the stretch by the nearest one inside it
                        "from": next((v for v in b["from"].iloc[i:j] if isinstance(v, str) and v), ""),
                        "to": next((v for v in b["to"].iloc[i:j][::-1] if isinstance(v, str) and v), ""),
                        "x0_m": b.x0_m.iloc[i], "x1_m": b.x1_m.iloc[j - 1],
                        "capacity_ref": round(float(wb[i:j].sum()), 1)}
                       for k, (i, j) in enumerate(groups)])
    st.to_csv(out / "stretches.csv", index=False)
    of = {bid: s.stretch_id for s in st.itertuples() for bid in s.blocks.split()}
    o = occ.assign(stretch_id=occ.block_id.map(of)).dropna(subset=["stretch_id"])
    full = o.groupby(["clip", "stretch_id"]).capacity.sum()
    cov = o[o.covered & o.n_parked.notna()]
    u = cov[cov.capacity > 0]
    sp = u.groupby(["clip", "stretch_id"]).agg(when=("when", "first"), date=("date", "first"),
                                                in_window=("in_window", "first"),
                                                capacity=("capacity", "sum"),
                                                n_parked=("n_parked", "sum")).reset_index()
    sp["capacity_full"] = sp.set_index(["clip", "stretch_id"]).index.map(full).values
    sp["counted"] = sp.capacity >= 0.5 * sp.capacity_full
    sp["occupancy"] = (sp.n_parked / sp.capacity).round(3)
    sp.to_csv(out / "occupancy_stretches_by_pass.csv", index=False)
    e = sp[sp.counted & (sp.in_window if sp.in_window.any() else True)]
    rows = []
    for sid, ge in e.groupby("stretch_id"):
        cs = cov[(cov.stretch_id == sid) & cov["clip"].isin(ge["clip"])]
        n = cs.groupby(["block_id", "row"])["clip"].nunique()
        common = set(n[n == ge["clip"].nunique()].index)
        cs = cs[[k in common for k in zip(cs.block_id, cs["row"])]]
        per = cs.groupby("clip").agg(capacity=("capacity", "sum"), n_parked=("n_parked", "sum"))
        per = per[per.capacity > 0]
        if len(per):
            rows.append({"stretch_id": sid, "n_passes": len(per),
                         "capacity_mean": per.capacity.mean(), "parked_mean": per.n_parked.mean(),
                         "occupancy_mean": (per.n_parked / per.capacity).mean(),
                         "capacity_full": float(sp[sp.stretch_id == sid].capacity_full.mean())})
    ev = pd.DataFrame(rows, columns=["stretch_id", "n_passes", "capacity_mean", "parked_mean",
                                     "occupancy_mean", "capacity_full"])
    ev["class"] = ev.occupancy_mean.map(lambda v: occupancy_class(v, g["classes"]))
    ev.round(3).to_csv(out / "occupancy_stretches_evening.csv", index=False)
    if g.get("slots"):
        slot_means(e, cov, sp, g).round(3).to_csv(out / "occupancy_stretches_slots.csv",
                                                  index=False)
    return st, sp, ev


SLOT_COLS = ["stretch_id", "agg", "slots", "n_passes", "n_dates", "capacity_mean",
             "parked_mean", "occupancy_mean", "capacity_full", "class"]


def slot_means(e: pd.DataFrame, cov: pd.DataFrame, sp: pd.DataFrame, g: dict) -> pd.DataFrame:
    """Per stretch, one row per view (`agg`): `slot:<key>` = mean over the slot's evenings
    (dates; the passes of one evening averaged first) and `avg` = the time-weighted average
    over the slots that saw the stretch (weights = slot hours): parked and capacity are each
    weighted, occupancy = their ratio (occupied / available space-hours; capacity moves with
    the hour's rules). Each view counts the block x rows covered by all of its passes (as the
    evening mean). e: counted window passes per stretch (`stretches`), cov: covered blocks."""
    slots = g["slots"]
    w = {s["key"]: s["weight"] for s in slots}
    order = [s["key"] for s in slots]
    e = e.assign(slot=[slot_of(v, slots) for v in e["when"]]).dropna(subset=["slot"])
    rows = []
    for sid, ge in e.groupby("stretch_id"):
        have = [k for k in order if (ge.slot == k).any()]
        views = [("avg", have)] + [(f"slot:{k}", [k]) for k in have] if have else []
        for agg, keys in views:
            clips = ge.loc[ge.slot.isin(keys), "clip"].unique()
            cs = cov[(cov.stretch_id == sid) & cov["clip"].isin(clips)]
            n = cs.groupby(["block_id", "row"])["clip"].nunique()
            common = set(n[n == len(clips)].index)
            cs = cs[[k in common for k in zip(cs.block_id, cs["row"])]]
            per = cs.groupby("clip").agg(capacity=("capacity", "sum"),
                                         n_parked=("n_parked", "sum"), date=("date", "first"))
            per = per[per.capacity > 0]
            if not len(per):
                continue
            per["slot"] = per.index.map(dict(zip(ge["clip"], ge.slot)))
            d = (per.groupby(["slot", "date"])[["capacity", "n_parked"]].mean()
                 .groupby("slot").agg(capacity=("capacity", "mean"), n_parked=("n_parked", "mean"),
                                      n_dates=("capacity", "size")))
            d = d.reindex([k for k in order if k in d.index])
            ws = np.array([w[k] for k in d.index])
            cap = float((d.capacity * ws).sum() / ws.sum())
            par = float((d.n_parked * ws).sum() / ws.sum())
            rows.append({"stretch_id": sid, "agg": agg, "slots": "+".join(d.index),
                         "n_passes": int(len(per)), "n_dates": int(d.n_dates.sum()),
                         "capacity_mean": cap, "parked_mean": par,
                         "occupancy_mean": par / cap if cap else np.nan,
                         "capacity_full": float(sp[sp.stretch_id == sid].capacity_full.mean())})
    out = pd.DataFrame(rows, columns=SLOT_COLS[:-1])
    out["class"] = out.occupancy_mean.map(lambda v: occupancy_class(v, g["classes"]))
    return out


def slot_totals(df: pd.DataFrame) -> dict:
    """Street (or area) totals per view over the stretches it has: sum parked / sum capacity;
    `one_slot` = stretches whose `avg` rests on a single slot."""
    out = {}
    for agg, q in df.groupby("agg"):
        c, n = float(q.capacity_mean.sum()), float(q.parked_mean.sum())
        out[agg] = {"parked": round(n, 2), "capacity": round(c, 2),
                    "occupancy": round(n / c, 3) if c else None, "stretches": int(len(q)),
                    "one_slot": int((~q.slots.str.contains("+", regex=False)).sum())
                    if agg == "avg" else None}
    return out
