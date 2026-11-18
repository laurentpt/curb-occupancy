"""Map stage: the supply view on an OpenStreetMap basemap (Leaflet, one self-contained HTML
page per mode, data inlined) + the same layers as GeoJSON.

Everything the supply stage computes is in street-axis metres per parking row; it is drawn
offset from the street axis toward the row's curb. The curb offset comes from the OSM
sidewalks (`footway=sidewalk`, centreline minus `sidewalk_inset_m`) where they are mapped,
else `curb_offset_m`; the parking lane is drawn `lane_inset_m` inside the curb.

* `review.html` (private, under workdir): occupancy and regulation views, regulation zones,
  sign poles, hydrants, curb cuts, parked cars per pass, GPS tracks, and one site-check card
  per sign pole (a zone with cars parked inside, or a panel the prior did not apply) with
  mosaic crops (every pass), raw frames of the pole (passes that see it) and a Street View
  link. Images contain plates: never publish this page.
* `public.html` + `layers.geojson`: occupancy only: parkable curb coloured by the block's
  occupancy class, non-parkable curb, parked-vehicle footprints (no class, length or flags,
  no cross-pass identity), block numbers.
* `curb.html` (private): the curb labeller (`labeller.py`).
* `ride_<clip>.jpg` (private): one picture per pass, parked row only, block by block
  (`rideview.py`).

    map:
      curb_offset_m: 6.0     # axis -> curb where no sidewalk is mapped
      sidewalk_inset_m: 1.0  # sidewalk centreline -> curb
      lane_inset_m: 2.5      # curb -> parked-row centre (drawing only)
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C

DEFAULTS = {"curb_offset_m": 6.0, "sidewalk_inset_m": 1.0, "lane_inset_m": 2.5,
            "crop_pad_m": 10.0, "frames": True}
R_EARTH = 6371000.0


# ---------------------------------------------------------------------------- geometry

def densify(P: np.ndarray, step: float) -> np.ndarray:
    out = [P[:1]]
    for a, b in zip(P[:-1], P[1:]):
        n = max(1, int(np.ceil(np.hypot(*(b - a)) / step)))
        out.append(a + (b - a) * (np.arange(1, n + 1)[:, None] / n))
    return np.vstack(out)


def curb_offsets(ax, osm: dict | None, x0: float, x1: float, default: float,
                 inset: float, step: float = 5.0, win: int = 7) -> dict:
    """Lateral distance of each curb from the street axis, sampled every `step` m:
    {'left'|'right': (grid, offsets)}. From the OSM sidewalks where mapped (nearest sidewalk
    point per bin, running median over `win` bins), interpolated across gaps, else default."""
    from ..street import local_xy, project_polyline
    grid = np.arange(x0 - step, x1 + 2 * step, step)
    if osm is None:
        return {s: (grid, np.full(len(grid), default)) for s in ("left", "right")}
    la0, la1, lo0, lo1 = ax.bbox(x0, x1, 40.0)
    pts = []
    for w in osm["elements"]:
        if w.get("type") != "way" or w.get("tags", {}).get("footway") != "sidewalk":
            continue
        g = np.array([(p["lat"], p["lon"]) for p in w["geometry"]])
        if g[:, 0].max() < la0 or g[:, 0].min() > la1 or g[:, 1].max() < lo0 or g[:, 1].min() > lo1:
            continue
        pts.append(densify(local_xy(g[:, 0], g[:, 1], *ax.origin), 2.0))
    if not pts:
        return {s: (grid, np.full(len(grid), default)) for s in ("left", "right")}
    pr = project_polyline(np.vstack(pts), ax.P, ax.X)
    out = {}
    for side, sgn in (("left", 1.0), ("right", -1.0)):
        lat = pr[:, 2] * sgn
        ok = (lat > 2.5) & (lat < 20.0) & (np.abs(pr[:, 0] - lat) < 1.0)
        k = np.digitize(pr[ok, 1], grid) - 1
        v = np.full(len(grid), np.nan)
        for i in np.unique(k[(k >= 0) & (k < len(grid))]):
            v[i] = lat[ok][k == i].min()
        h = win // 2
        med = np.array([np.nanmedian(v[max(0, i - h):i + h + 1])
                        if np.isfinite(v[max(0, i - h):i + h + 1]).any() else np.nan
                        for i in range(len(v))])
        f = np.isfinite(med)
        off = np.interp(grid, grid[f], med[f]) - inset if f.sum() >= 2 else \
            np.full(len(grid), default)
        out[side] = (grid, off)
    return out


class Geo:
    """Street-axis metres + lateral offsets -> lat/lon. `lat_m` + = left of increasing X."""

    def __init__(self, ax, offsets: dict, lane_inset: float):
        o = np.argsort(ax.X)
        self.ax, self.X, self.P = ax, ax.X[o], ax.P[o]
        self.offsets, self.lane_inset = offsets, lane_inset

    def _pt(self, x):
        x = np.atleast_1d(np.asarray(x, float))
        p = np.stack([np.interp(x, self.X, self.P[:, k]) for k in (0, 1)], -1)
        t = np.stack([np.interp(x + 1.0, self.X, self.P[:, k])
                      - np.interp(x - 1.0, self.X, self.P[:, k]) for k in (0, 1)], -1)
        t /= np.maximum(np.hypot(*t.T), 1e-9)[:, None]
        return p, np.stack([-t[:, 1], t[:, 0]], -1)   # point, left normal

    def to_ll(self, xy) -> np.ndarray:
        lat0, lon0 = self.ax.origin
        xy = np.asarray(xy, float)
        lat = lat0 + np.degrees(xy[:, 1] / R_EARTH)
        lon = lon0 + np.degrees(xy[:, 0] / (R_EARTH * np.cos(np.radians(lat0))))
        return np.stack([lat, lon], -1)

    def curb(self, side: str, x) -> np.ndarray:
        g, v = self.offsets[side]
        return np.interp(x, g, v)

    def line(self, x0: float, x1: float, side: str, extra: float = 0.0, step: float = 2.0):
        """[[lat, lon], ...] along [x0, x1] at the curb of `side` moved `extra` m outward
        (negative = toward the axis)."""
        n = max(2, int(np.ceil((x1 - x0) / step)) + 1)
        x = np.linspace(x0, x1, n)
        p, nrm = self._pt(x)
        sgn = 1.0 if side == "left" else -1.0
        q = p + nrm * (sgn * (self.curb(side, x) + extra))[:, None]
        return self.to_ll(q).round(7).tolist()

    def lane(self, x0, x1, side, across: float = 0.0):
        return self.line(x0, x1, side, -self.lane_inset + across)

    def centre(self, x0: float, x1: float, step: float = 2.0):
        """[[lat, lon], ...] along the street axis over [x0, x1]."""
        x = np.linspace(x0, x1, max(2, int(np.ceil((x1 - x0) / step)) + 1))
        return self.to_ll(self._pt(x)[0]).round(7).tolist()

    def point(self, x: float, lat_m: float):
        p, nrm = self._pt(x)
        return self.to_ll(p + nrm * lat_m)[0].round(7).tolist()


def _s(v) -> str:
    """'' for NaN / None."""
    return "" if v is None or v != v else str(v)


def _feat(coords, gtype: str, /, **props) -> dict:
    """GeoJSON feature from [lat, lon] (Point) or [[lat, lon], ...] (Line, Polygon)."""
    if gtype == "Point":
        geom = {"type": "Point", "coordinates": [coords[1], coords[0]]}
    elif gtype == "Polygon":
        geom = {"type": "Polygon", "coordinates": [[[c[1], c[0]] for c in coords]]}
    else:
        geom = {"type": "LineString", "coordinates": [[c[1], c[0]] for c in coords]}
    return {"type": "Feature", "geometry": geom, "properties": props}


def _runs(values: np.ndarray, x: np.ndarray, step: float):
    """(value, x0, x1) of runs of equal consecutive values."""
    cut = np.r_[0, np.where(values[1:] != values[:-1])[0] + 1, len(values)]
    return [(values[a], float(x[a]), float(x[b - 1] + step)) for a, b in zip(cut[:-1], cut[1:])]


def _stack(z: pd.DataFrame) -> np.ndarray:
    """Greedy track per interval so overlapping zones of a row are drawn side by side."""
    ends, out = [], np.zeros(len(z), int)
    for i, (a, b) in enumerate(zip(z.x0.values, z.x1.values)):
        k = next((k for k, e in enumerate(ends) if e <= a), None)
        if k is None:
            k = len(ends)
            ends.append(b)
        ends[k] = b
        out[i] = k
    return out


# ---------------------------------------------------------------------------- inputs

def _rule(d: dict):
    from ..regs.rules import Rule
    return Rule("", d["kind"], d.get("minutes"), [tuple(h) for h in d.get("hours") or []],
                set(d.get("days", range(7))),
                tuple(map(tuple, d["season"])) if d.get("season") else None)


def _passes(cfg: dict, root: Path, lay) -> list[dict]:
    from ..regs.supply import pass_time
    out = []
    for c in cfg["clips"]:
        if c["id"] in lay.maps:
            w = pass_time(root, c["id"])
            out.append({"id": c["id"], "when": w.isoformat(timespec="minutes"),
                        "label": f"{c['id']} {w:%a %d %b %H:%M}",
                        "dir": "with axis" if not lay.reversed(c["id"]) else "against axis"})
    return out


class ClipPos:
    """Street-axis x -> clip x (metres) and video time, from the clip's trace."""

    def __init__(self, root: Path, lay, clip: str):
        tr = pd.read_csv(root / clip / "trace.csv").dropna(subset=["X_m"])
        xs = lay.maps[clip](tr.X_m.values)
        o = np.argsort(xs, kind="stable")
        self.xs, self.xc, self.t = xs[o], tr.X_m.values[o], tr.t.values[o]

    def clip_x(self, x):
        return np.interp(x, self.xs, self.xc)

    def time(self, x):
        return np.interp(x, self.xs, self.t)

    def covers(self, x) -> bool:
        return self.xs[0] <= x <= self.xs[-1]


# ---------------------------------------------------------------------------- images

def zone_crop(cfg, lay, clip: str, pos: ClipPos, row: str, x0: float, x1: float, pad: float,
              cache: dict, marks=(), max_w: int = 12000) -> np.ndarray | None:
    """Mosaic of the plane imaging `row` over [x0 - pad, x1 + pad] (street x), with a ruler
    in street metres and the zone ends (`marks`, default x0 and x1) in red."""
    import cv2
    from ..config import height
    from ..mosaic import Mosaic
    plane = next((p for p in cfg["mosaic"]["planes"] if lay.row_of(clip, p) == row), None)
    if plane is None:
        return None
    key = (clip, plane)
    if key not in cache:
        f = Path(cfg["out_dir"]) / clip / f"mosaic_{plane}"
        cache[key] = Mosaic.load(f) if f.with_suffix(".png").exists() else None
    m = cache[key]
    if m is None:
        return None
    h = height(cfg, clip)

    def col(x):
        return (pos.clip_x(x) / h - m.x0) * m.px_per_m

    a, b = sorted((col(x0 - pad), col(x1 + pad)))
    a, b = int(max(0, a)), int(min(m.image.shape[1], b))
    if b - a < 20:
        return None
    img = np.pad(m.image[:, a:b], ((18, 0), (0, 0), (0, 0))).copy()
    for xs in np.arange(np.ceil((x0 - pad) / 5) * 5, x1 + pad, 5.0):
        c = int(col(xs) - a)
        cv2.line(img, (c, 0), (c, 8 if xs % 25 else 14), (255, 255, 255), 1)
        if xs % 25 == 0 or xs == np.ceil((x0 - pad) / 5) * 5:
            cv2.putText(img, f"{xs:.0f}", (c + 3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1)
    for xs in sorted(set(marks or (x0, x1))):
        c = int(col(xs) - a)
        cv2.line(img, (c, 18), (c, img.shape[0] - 1), (232, 73, 72), 2)
    if img.shape[1] > max_w:
        img = cv2.resize(img, (max_w, int(img.shape[0] * max_w / img.shape[1])),
                         interpolation=cv2.INTER_AREA)
    return img


def pole_frames(cfg, lay, clip: str, pos: ClipPos, x_pole: float, lat_pole: float,
                geo: Geo) -> list[tuple[float, float]]:
    """(street x of the bike, video time) where the raw frame should show the pole: only
    passes with the pole on the rider's left (the camera looks ~65 deg left); bike-pole
    distance D from the curb offsets; the pole is near the optical axis D / tan(yaw) ahead,
    and seen more frontally ~1.5 D ahead."""
    fwd = not lay.reversed(clip)
    if (lat_pole > 0) != fwd:
        return []
    other = "right" if lat_pole > 0 else "left"
    D = abs(lat_pole) + float(geo.curb(other, x_pole)) - 1.0
    s = 1.0 if fwd else -1.0
    out = []
    yaw = C.clip(cfg, clip)["camera"].get("mount_yaw_deg") or 90.0
    for a in (D / np.tan(np.radians(yaw)), 1.5 * D):
        xb = x_pole - s * a
        if pos.covers(xb):
            out.append((float(xb), float(pos.time(xb))))
    return out


def _save_jpg(img: np.ndarray, path: Path, q: int = 85):
    import cv2
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])


# ---------------------------------------------------------------------------- layers

def build(cfg: dict, force: bool = False) -> dict | None:
    """Run-level stage. Writes <out_dir>/map/{review,public,curb}.html, layers.geojson, img/,
    strips/."""
    from ..config import _merge
    from ..layout import Layout
    from ..regs.supply import cars_inside, parkable
    mc = _merge(DEFAULTS, cfg.get("map") or {})
    root = Path(cfg["out_dir"])
    lay = Layout.load(root / "layout")
    sup = root / "supply"
    if lay is None or lay.axis is None or not (sup / "curb_status.csv").exists():
        return None
    out = root / "map"
    (out / "img").mkdir(parents=True, exist_ok=True)
    osm_json = cfg["street"].get("osm_json")
    osm = json.loads(Path(osm_json).read_text()) if osm_json and Path(osm_json).exists() else None
    x_lo, x_hi = lay.extent_m
    geo = Geo(lay.axis, curb_offsets(lay.axis, osm, x_lo, x_hi, mc["curb_offset_m"],
                                     mc["sidewalk_inset_m"]), mc["lane_inset_m"])
    side = {r: rc["curb"] for r, rc in lay.rows.items()}
    passes = _passes(cfg, root, lay)
    pids = [p["id"] for p in passes]
    sj = json.loads((sup / "supply.json").read_text())
    g = sj["guideline"]
    rules = {k: _rule(v) for k, v in sj.get("rules", {}).items()}
    when = {p["id"]: datetime.fromisoformat(p["when"]) for p in passes}
    ev_ids = [c for c in pids
              if g["window"][0] <= when[c].hour + when[c].minute / 60 < g["window"][1]]
    aggs = aggregates(g, {c: when[c] for c in ev_ids})
    L: dict[str, list] = {k: [] for k in ("status", "zones", "signs", "hydrants", "cuts",
                                          "stretches", "cars", "tracks")}
    blk = pd.read_csv(sup / "blocks.csv").set_index("block_id")
    edges = np.r_[blk.x0_m.values, blk.x1_m.values[-1:]]
    sts = pd.read_csv(sup / "stretches.csv").rename(columns={"from": "frm"})
    stretch_of = {b: q.stretch_id for q in sts.itertuples() for b in q.blocks.split()}

    # curb status per pass (+ 'evening': the majority over the guideline-window passes), runs
    # at the curb line split at block ends; `parkable` = counts as capacity (guideline)
    cs = pd.read_csv(sup / "curb_status.csv")
    for r, gr in cs.groupby("row"):
        gr = gr.sort_values("x_ref_m")
        x = gr.x_ref_m.values
        cols = {c: gr[f"status_{c}"].values for c in pids if f"status_{c}" in gr}
        if ev_ids:
            cols["evening"] = _majority(np.stack([cols[c] for c in ev_ids if c in cols]), g)
        for a in aggs:  # slot views: majority weighted like the view's numbers
            ks = [c for c in a["passes"] if c in cols]
            if ks:
                cols[a["id"]] = _majority(np.stack([cols[c] for c in ks]), g,
                                          np.array([a["pass_weight"][c] for c in ks]))
        b_of = np.searchsorted(edges, x + 0.5, side="right") - 1
        brk = np.r_[0, np.where((np.diff(x) > 1.5) | (np.diff(b_of) != 0))[0] + 1, len(x)]
        for c, st in cols.items():
            for a, b in zip(brk[:-1], brk[1:]):
                bid = blk.index[b_of[a]] if 0 <= b_of[a] < len(blk) else ""
                for s, s0, s1 in _runs(st[a:b], x[a:b], 1.0):
                    if s != "absent":
                        L["status"].append(_feat(
                            geo.line(s0, s1, side[r]), "Line", clip=c, row=r, status=s,
                            block=bid, stretch=stretch_of.get(bid, ""),
                            parkable=bool(parkable(np.array([s]), g)[0]),
                            x0=round(s0), x1=round(s1)))

    # regulation zones, stacked outward from the curb
    z = pd.read_csv(sup / "zones.csv")
    z = z.drop_duplicates(["row", "x0", "x1", "kind", "ref", "source"]).reset_index(drop=True)
    z["ref"] = z["ref"].map(lambda v: str(int(v)) if isinstance(v, float) and v == v else str(v))
    z["track"] = 0
    for r, gz in z.sort_values("x0").groupby("row"):
        z.loc[gz.index, "track"] = _stack(gz)
    cars = pd.read_csv(root / "multiday" / "cars_ref.csv")
    from ..regs.supply import in_roadway
    cars = cars[~in_roadway(cars, lay.intersections, lay.rows)]   # not parked: as supply
    checks: dict[str, dict] = {}   # site checks, one per sign pole

    def check(ref, row, text, source, x0, x1, why, zone=None, marks=()):
        c = checks.setdefault(ref, {"ref": ref, "row": row, "text": text, "source": source,
                                    "x0": x0, "x1": x1, "why": [], "zones": [], "marks": []})
        c["x0"], c["x1"] = min(c["x0"], x0), max(c["x1"], x1)
        c["why"].append(why)
        c["marks"] += list(marks)
        if zone is not None:
            c["zones"].append(zone)
    for _, q in z.iterrows():
        if q.row not in side:
            continue
        x0, x1 = max(q.x0, x_lo), min(q.x1, x_hi)
        if x1 - x0 < 0.5:
            continue
        active = {c: bool(rules[q.rule].active(when[c])) if q.rule in rules else None
                  for c in pids}
        k = cars[cars["row"] == q.row]
        inside = {c: int(len(cars_inside(k[k["clip"] == c], q.x0, q.x1))) for c in pids}
        props = dict(row=q.row, kind=q.kind, text=str(q.text), ref=q.ref, source=q.source,
                     inference=_s(q.get("inference")), x0=round(q.x0, 1), x1=round(q.x1, 1),
                     disputed=bool(q.disputed), active=active, cars_inside=inside,
                     track=int(q.track))
        if q.kind in ("no_stopping", "reserved") and any(
                inside[c] and active[c] is not False for c in pids):
            props["check"] = q.ref
            check(q.ref, q.row, str(q.text), q.source, q.x0, q.x1, "cars parked inside",
                  {**props, "length_m": round(q.x1 - q.x0, 1)}, (q.x0, q.x1))
        L["zones"].append(_feat(geo.line(x0, x1, side[q.row], 1.2 + 1.0 * q.track), "Line",
                                **props))
    ig = sup / "signs_ignored.csv"
    from ..regs.supply import attributions, source_module, source_name
    pole_src = next((source_name(sc) for sc in (cfg.get("regulations") or {}).get("sources", [])
                     if getattr(source_module(sc.get("type")), "POLES", False)), None)
    if pole_src and ig.exists() and ig.stat().st_size > 1:
        for q in pd.read_csv(ig).itertuples():
            if q.row in side and x_lo <= q.x_ref <= x_hi:
                check(str(int(q.pole)), q.row, q.rpa, pole_src, q.x_ref - 20, q.x_ref + 20,
                      f"not applied: {q.why}" + ("" if str(q.why).startswith("dropped")
                                                 else " (prior)"), marks=(q.x_ref,))

    # sign poles at their surveyed position
    signs = pd.read_csv(sup / "signs_ref.csv") if (sup / "signs_ref.csv").exists() else None
    poles = {}
    if signs is not None and len(signs):
        signs["pole"] = signs.POTEAU_ID_POT.astype(int).astype(str)
        for pole, gp in signs.groupby("pole"):
            r0 = gp.iloc[0]
            poles[pole] = {"lat": float(r0.Latitude), "lon": float(r0.Longitude),
                           "x": float(r0.x_ref_fit), "lat_m": float(r0.lat_m)}
            L["signs"].append(_feat([r0.Latitude, r0.Longitude], "Point", pole=pole,
                                    x=round(float(r0.x_ref_fit), 1),
                                    lat_m=round(float(r0.lat_m), 1),
                                    row=_s(r0.row),
                                    parked_curb=bool(r0.parked_curb),
                                    check=pole in checks,
                                    panels=[{"text": p.DESCRIPTION_RPA, "code": p.CODE_RPA,
                                             "arrow": int(p.FLECHE_PAN), "kind": p.kind,
                                             "installed": str(p.DATE_EXECUTION_ACT)}
                                            for p in gp.itertuples()]))
    hy = sup / "hydrants_ref.csv"
    if hy.exists():
        for h in pd.read_csv(hy).itertuples():
            L["hydrants"].append(_feat([h.LATITUDE, h.LONGITUDE], "Point", ref=str(h.ID_BI),
                                       address=str(h.ADRESSE), x=round(h.x_ref, 1),
                                       row=_s(h.row)))
    cr = lay.crossings
    if len(cr):
        for c in cr[(cr.kind == "curb_cut") & cr.x_m.between(x_lo, x_hi)].itertuples():
            for sd in str(c.sides).split("+"):
                if sd in ("left", "right"):
                    L["cuts"].append(_feat(geo.point(c.x_m, (1 if sd == "left" else -1)
                                                      * float(geo.curb(sd, c.x_m))), "Point",
                                           service=str(c.service), way=str(c.osm_way),
                                           x=round(c.x_m, 1), side=sd))

    # stretch occupancy (consecutive blocks grouped to ~guideline.stretch_spaces spaces, all
    # rows; per pass + evening mean): label anchor on the axis + numbers
    sp = pd.read_csv(sup / "occupancy_stretches_by_pass.csv")
    ev = sup / "occupancy_stretches_evening.csv"
    ev = pd.read_csv(ev).set_index("stretch_id") if ev.exists() and ev.stat().st_size > 1 \
        else pd.DataFrame()
    sl = sup / "occupancy_stretches_slots.csv"
    sl = pd.read_csv(sl) if aggs and sl.exists() and sl.stat().st_size > 1 else pd.DataFrame(
        columns=["stretch_id", "agg"])
    for q in sts.itertuples():
        x0, x1 = max(q.x0_m, x_lo), min(q.x1_m, x_hi)
        if x1 - x0 < 5:
            continue
        per = {r.clip: {"capacity": int(r.capacity), "covered": bool(r.counted),
                        "parked": int(r.n_parked), "occupancy": float(r.occupancy)}
               for r in sp[sp.stretch_id == q.stretch_id].itertuples()}
        e = ev.loc[q.stretch_id] if q.stretch_id in ev.index else None
        L["stretches"].append(_feat(geo.centre(x0, x1), "Line", stretch=q.stretch_id,
                                    name=f"{_s(q.frm) or 'start'} – {_s(q.to) or 'end'}",
                                    blocks=q.blocks, x0=round(q.x0_m), x1=round(q.x1_m),
                                    per_pass=per,
                                    evening=None if e is None else {
                                        "n_passes": int(e.n_passes),
                                        "occupancy": float(e.occupancy_mean),
                                        "parked": float(e.parked_mean),
                                        "capacity": float(e.capacity_mean),
                                        "class": e["class"]},
                                    agg={r["agg"]: {"n_passes": int(r["n_passes"]),
                                                    "n_dates": int(r["n_dates"]),
                                                    "slots": r["slots"],
                                                    "occupancy": float(r["occupancy_mean"]),
                                                    "parked": float(r["parked_mean"]),
                                                    "capacity": float(r["capacity_mean"]),
                                                    "class": r["class"]}
                                         for r in sl[sl.stretch_id == q.stretch_id]
                                         .to_dict("records")}))

    # parked vehicles per pass (public: footprint only), GPS tracks (private). n_cov = evening
    # passes covering the vehicle's block x row: the evening overlay draws each footprint at
    # 1 / n_cov of full opacity, so a spot parked whenever it was seen is darkest
    ob = pd.read_csv(root / "supply" / "occupancy_by_pass.csv")
    ob = ob[ob.covered & ob["clip"].isin(ev_ids)]

    def n_cov(row, x):  # guideline blocks (supply), not the multiday blocks of cars_ref
        return int(ob.loc[(ob["row"] == row) & (ob.x0_ref_m <= x) & (x < ob.x1_ref_m), "clip"].nunique())

    def opacity(row, x, clip):
        """Footprint opacity per slot view (`agg` id): the car's share of the view's
        weight among the passes covering its block (slot weight / its covering passes,
        renormalised over the slots that cover it): a spot parked whenever seen is darkest."""
        here = set(ob.loc[(ob["row"] == row) & (ob.x0_ref_m <= x) & (x < ob.x1_ref_m), "clip"])
        out = {}
        for a in aggs:
            if clip not in a["passes"] or clip not in here:
                continue
            ks = [c for c in a["passes"] if c in here]
            out[a["id"]] = round(a["pass_weight"][clip] / sum(a["pass_weight"][c] for c in ks), 3)
        return out
    for c in cars.itertuples():
        if c.row not in side or c.clip not in pids:
            continue
        a = geo.lane(c.xr_start_m, c.xr_end_m, side[c.row], 0.85)
        b = geo.lane(c.xr_start_m, c.xr_end_m, side[c.row], -0.85)
        L["cars"].append(_feat(a + b[::-1] + a[:1], "Polygon", clip=c.clip, row=c.row,
                               n_cov=n_cov(c.row, c.xr_mid_m), op=opacity(c.row, c.xr_mid_m, c.clip),
                               cls=c.cls_name, length_m=round(c.length_m, 1),
                               x0=round(c.xr_start_m, 1), x1=round(c.xr_end_m, 1),
                               flag=_s(c.flag)))
    from ..config import telemetry
    for c in pids:
        f = root / c / "telemetry.npz"
        if f.exists():
            t = telemetry(cfg, c)
            ll = t.gps[t.gps_fix >= 3, :2][::18]
            if len(ll) > 1:
                L["tracks"].append(_feat(ll.round(7).tolist(), "Line", clip=c))

    # site-check cards, one per sign pole: mosaic crops per pass, raw frames of the pole;
    # image names carry their content
    cache, pos = {}, {c: ClipPos(root, lay, c) for c in pids}
    cards = []
    for i, (ref, d) in enumerate(sorted(checks.items(), key=lambda kv: kv[1]["x0"])):
        x0, x1 = d["x0"], d["x1"]
        k = cars[cars["row"] == d["row"]]
        d.update({"check": f"C{i + 1:02d}", "length_m": round(x1 - x0, 1),
                  "why": sorted(set(d["why"])), "crops": [], "frames": [],
                  "cars_inside": {c: int(len(cars_inside(k[k["clip"] == c], x0, x1)))
                                  for c in pids}})
        cards.append(d)
        for c in pids:
            f = out / "img" / f"zone_{d['row']}_{x0:.0f}-{x1:.0f}_{c}.jpg"
            if force or not f.exists():
                img = zone_crop(cfg, lay, c, pos[c], d["row"], x0, x1, mc["crop_pad_m"], cache,
                                d["marks"])
                if img is None:
                    continue
                _save_jpg(img, f)
            d["crops"].append({"pass": c, "src": f"img/{f.name}"})
        p = poles.get(ref)
        if p is None:
            continue
        d["pole"] = p
        sv_from = geo.point(p["x"], 0.0)
        d["streetview"] = ("https://www.google.com/maps/@?api=1&map_action=pano&viewpoint="
                           f"{sv_from[0]},{sv_from[1]}&heading={_bearing(sv_from, p):.0f}")
        if not mc["frames"]:
            continue
        for c in pids:
            for j, (xb, t) in enumerate(pole_frames(cfg, lay, c, pos[c], p["x"], p["lat_m"], geo)):
                f = out / "img" / f"pole{ref}_{p['x']:.0f}_{c}_frame{j}.jpg"
                if force or not f.exists():
                    from ..config import source
                    from ..capture.video import grab
                    _save_jpg(grab(source(cfg, c), t), f)
                d["frames"].append({"pass": c, "src": f"img/{f.name}", "t": round(t, 2),
                                    "bike_x": round(xb, 1)})

    fc = {k: {"type": "FeatureCollection", "features": v} for k, v in L.items()}
    # public: vehicle footprints without class, length or flags; no tracks, no regulation
    # detail beyond curb status
    pub_cars = [{**f, "properties": {k: f["properties"][k] for k in ("clip", "row", "n_cov", "op")}}
                for f in L["cars"]]
    pub = {"status": fc["status"], "stretches": fc["stretches"],
           "cars": {"type": "FeatureCollection", "features": pub_cars}}
    sources = sj.get("sources", {})
    meta = {"name": cfg["name"], "street": cfg["street"].get("name"), "passes": passes,
            "evening_passes": ev_ids,
            "aggregates": [{k: v for k, v in a.items() if k != "pass_weight"} for a in aggs],
            "extent_m": [round(x_lo, 1), round(x_hi, 1)], "guideline": g,
            "occupancy": sj.get("occupancy"), "axis": sj.get("axis"),
            "driveways_labelled": any((sources.get(source_name(sc)) or {}).get("n_zones")
                                      for sc in (cfg.get("regulations") or {}).get("sources", [])
                                      if sc.get("type") == "manual"),
            "prior": next((i["prior"] for i in sources.values() if i.get("prior")), None),
            "regs": regs_summary([(cfg["street"].get("name") or cfg["name"], sources)]),
            "attribution": attributions(cfg),
            "generated": datetime.now().isoformat(timespec="minutes"),
            "center": geo.point((x_lo + x_hi) / 2, 0.0)}
    (out / "layers.geojson").write_text(json.dumps(
        {"type": "FeatureCollection",
         "features": [{**f, "properties": {**f["properties"], "layer": k}}
                      for k, v in pub.items() for f in v["features"]]}))
    tpl = (Path(__file__).parent / "webmap.html").read_text()
    for mode in ("review", "public"):
        data = {"meta": {**meta, "mode": mode}, "layers": fc if mode == "review" else pub,
                "checks": cards if mode == "review" else []}
        (out / f"{mode}.html").write_text(
            tpl.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False, default=str)))
    from . import labeller
    lab = labeller.build(cfg, lay, geo, passes, out, force)
    from . import rideview
    ride = rideview.build(cfg, lay, passes, out)
    return {"layers": {k: len(v) for k, v in L.items()}, "checks": len(cards),
            "labeller": lab, "ride": ride, "out": str(out)}


def regs_summary(per_street: list[tuple[str, dict]]) -> dict | None:
    """Sign handling for the page's method, from the regulation sources' info of each street
    ([(street name, supply.json sources)]): streets read as signed vs with inferred arrows,
    and the corrections of the recorded data (unique poles). None without a signage source."""
    conv, inf, arrows, dropped, moved = [], [], set(), set(), set()
    for name, sources in per_street:
        for i in sources.values():
            if i.get("arrows") not in ("convention", "inferred") or i.get("prior"):
                continue
            (conv if i["arrows"] == "convention" else inf).append(name)
            arrows |= set(i.get("poles_arrows") or [])
            dropped |= set(i.get("poles_dropped") or [])
            moved |= {int(p) for p in (i.get("poles_moved") or {})}
    if not conv and not inf:
        return None
    return {"convention": sorted(set(conv)), "inferred": sorted(set(inf)),
            "arrows": len(arrows), "dropped": len(dropped), "moved": len(moved)}


def _majority(st: np.ndarray, g: dict, w: np.ndarray | None = None) -> np.ndarray:
    """Per curb bin over passes (rows of st, weights w, default equal): the most common status
    among the parkable ones when most passes (by weight) find it parkable, else the most
    common non-parkable one."""
    from ..regs.supply import parkable
    w = np.ones(st.shape[0]) if w is None else np.asarray(w, float)
    out = np.empty(st.shape[1], dtype=object)
    pk = parkable(st, g)
    for j in range(st.shape[1]):
        m = pk[:, j] if (w * pk[:, j]).sum() >= 0.5 * w.sum() else ~pk[:, j]
        v, inv = np.unique(st[m, j].astype(str), return_inverse=True)
        out[j] = v[np.argmax(np.bincount(inv, weights=w[m]))]
    return out


def aggregates(g: dict, when: dict) -> list[dict]:
    """Slot views of the guideline-window passes (`when`: pass -> start time), when
    `guideline.slots` is set: `avg` (time-weighted average of the slots with passes) then
    one view per such slot. `pass_weight`: a pass's weight in the view (slot weight / the
    slot's evenings / the passes of its evening), for status majorities and car opacity.
    The numbers themselves come from supply (`occupancy_stretches_slots.csv`)."""
    from ..regs.guideline import slot_of
    slots = g.get("slots") or []
    by = {s["key"]: [c for c, t in when.items() if slot_of(t, slots) == s["key"]] for s in slots}
    have = [s for s in slots if by[s["key"]]]
    if not have:
        return []

    def weights(keys):
        pw, tot = {}, sum(s["weight"] for s in have if s["key"] in keys)
        for s in have:
            if s["key"] not in keys:
                continue
            days = {}
            for c in by[s["key"]]:
                days.setdefault(when[c].date(), []).append(c)
            for cs in days.values():
                for c in cs:
                    pw[c] = s["weight"] / tot / len(days) / len(cs)
        return pw

    def dates(cs):
        return sorted({when[c].date().isoformat() for c in cs})
    out = [{"id": "avg", "kind": "avg", "slots": [s["key"] for s in have],
            "passes": [c for s in have for c in by[s["key"]]],
            "dates": dates([c for s in have for c in by[s["key"]]]),
            "pass_weight": weights([s["key"] for s in have])}]
    for s in have:
        out.append({"id": f"slot:{s['key']}", "kind": "slot", "slot": s["key"],
                    "hours": s["hours"], "weight": s["weight"], "label": s.get("label"),
                    "passes": by[s["key"]], "dates": dates(by[s["key"]]),
                    "pass_weight": weights([s["key"]])})
    return out


def _bearing(a, p: dict) -> float:
    la1, lo1, la2, lo2 = map(np.radians, (a[0], a[1], p["lat"], p["lon"]))
    y = np.sin(lo2 - lo1) * np.cos(la2)
    x = np.cos(la1) * np.sin(la2) - np.sin(la1) * np.cos(la2) * np.cos(lo2 - lo1)
    return float(np.degrees(np.arctan2(y, x)) % 360)
