"""Street model: the street axis and the parking rows of the cross-section.

Street axis (`build_axis`): the along-street coordinate every run-level stage works in. With
GPS on the reference pass and an OpenStreetMap extract (`street.osm_json`), the mapped
centreline in map metres, oriented with the reference pass and starting where it starts; the
reference pass's visual axis maps onto it through its GPS track (`fit_track_map`). Cross
streets and curb cuts come from the map (`merge_osm_intersections`, `osm.py`). Without the
map, the reference GPS track; without GPS, the reference pass's visual axis.

Rows (`street.rows`): a row is a physical parking lane, governed by the signs of one curb
('right'/'left' of the street-axis direction). A pass riding forward (same way as the
reference) or backward images each row in one of its mosaic planes, or not at all (a row on
the side the camera does not look at). Everything street-specific about which plane is which
row lives here; the rest of the pipeline asks `plane_rows`/`row_plane`.
"""
from __future__ import annotations


def rows_cfg(cfg: dict) -> dict:
    return cfg["street"]["rows"]


def _key(backward: bool) -> str:
    return "backward" if backward else "forward"


def row_plane(rows: dict, row: str, backward: bool) -> str | None:
    """Mosaic plane imaging `row` for a pass riding forward/backward (None: not imaged)."""
    return (rows[row].get("plane") or {}).get(_key(backward))


def plane_rows(rows: dict, backward: bool) -> dict:
    """plane -> row for a pass riding forward/backward (planes imaging no row are absent)."""
    out = {}
    for r in rows:
        p = row_plane(rows, r, backward)
        if p is not None:
            if p in out:
                raise ValueError(f"rows {out[p]} and {r} are both imaged by plane {p!r} "
                                 f"riding {_key(backward)}: one plane per row")
            out[p] = r
    return out


def plane_pairs(rows: dict, back_a: bool, back_b: bool) -> list[tuple[str, str, str]]:
    """(plane of pass a, plane of pass b, row) for every row both passes image."""
    out = []
    for r in rows:
        pa, pb = row_plane(rows, r, back_a), row_plane(rows, r, back_b)
        if pa is not None and pb is not None:
            out.append((pa, pb, r))
    return out


def validate(cfg: dict) -> None:
    rows, planes = rows_cfg(cfg), set(cfg["mosaic"]["planes"])
    for k, v in (("street.rows", rows), ("street.states", cfg["street"]["states"]),
                 ("mosaic.planes", planes)):
        if not v:
            raise ValueError(f"{k} is required (the street's cross-section; see the study configs)")
    for r, rc in rows.items():
        if rc.get("curb") not in ("right", "left"):
            raise ValueError(f"street.rows.{r}.curb must be 'right' or 'left'")
        for k, p in (rc.get("plane") or {}).items():
            if k not in ("forward", "backward") or p not in planes:
                raise ValueError(f"street.rows.{r}.plane.{k} = {p!r}: unknown plane")
    for b in (False, True):
        plane_rows(rows, b)
    for s in cfg["street"]["states"]:
        if not set(s) <= set(rows):
            raise ValueError(f"street.states: {s} names an unknown row")


def state_label(rows_present) -> str:
    return "+".join(rows_present) if len(rows_present) else "none"


# ---------------------------------------------------------------------------- geometry

def local_xy(lat, lon, lat0: float, lon0: float):
    """Equirectangular local metres (x east, y north) around (lat0, lon0)."""
    import numpy as np
    R = 6371000.0
    return np.stack([np.radians(np.asarray(lon, float) - lon0) * R * np.cos(np.radians(lat0)),
                     np.radians(np.asarray(lat, float) - lat0) * R], -1)


def project_polyline(points, P, X):
    """(N,2) local points onto a polyline P (M,2) whose vertices sit at axis positions X (M,):
    (N,3) = distance to the polyline, axis position, signed lateral offset (+ = left of the
    direction of increasing X)."""
    import numpy as np
    a, b = P[:-1], P[1:]
    d = b - a
    L2 = (d ** 2).sum(1)
    ok = L2 > 0.09
    a, d, L2, X0, X1 = a[ok], d[ok], L2[ok], X[:-1][ok], X[1:][ok]
    out = np.empty((len(points), 3))
    for i, q in enumerate(np.asarray(points, float)):
        u = np.clip(((q - a) * d).sum(1) / L2, 0, 1)
        p = a + u[:, None] * d
        dist = np.hypot(*(q - p).T)
        k = int(np.argmin(dist))
        lat = (d[k, 0] * (q - a[k])[1] - d[k, 1] * (q - a[k])[0]) / np.sqrt(L2[k])
        if X1[k] < X0[k]:  # vertex order runs against the axis: left of +X is right of d
            lat = -lat
        out[i] = (dist[k], X0[k] + u[k] * (X1[k] - X0[k]), lat)
    return out


# ---------------------------------------------------------------------------- street axis

class StreetAxis:
    """Along-street coordinate: a lat/lon polyline whose vertices sit at axis positions X
    (metres, increasing in the reference pass's riding direction). kind 'osm' = the mapped
    centreline (map metres); 'track' = the reference pass's GPS track at its visual-odometry
    positions (street axis = reference axis)."""

    def __init__(self, kind: str, lat, lon, X, meta: dict | None = None):
        import numpy as np
        self.kind = kind
        self.lat, self.lon, self.X = (np.asarray(v, float) for v in (lat, lon, X))
        self.origin = (float(self.lat.mean()), float(self.lon.mean()))
        self.P = local_xy(self.lat, self.lon, *self.origin)
        self.meta = meta or {}

    def project(self, lat, lon):
        """(N,3): distance to the axis, axis position, lateral offset (+ = left)."""
        import numpy as np
        pts = local_xy(lat, lon, *self.origin)
        return project_polyline(pts, self.P, self.X) if len(pts) else np.zeros((0, 3))

    def latlon(self, x):
        import numpy as np
        o = np.argsort(self.X)
        return (np.interp(x, self.X[o], self.lat[o]), np.interp(x, self.X[o], self.lon[o]))

    def bbox(self, x0: float, x1: float, pad_m: float = 0.0) -> tuple:
        """(lat_min, lat_max, lon_min, lon_max) of the axis between x0 and x1."""
        import numpy as np
        m = (self.X >= x0) & (self.X <= x1)
        ea, eo = self.latlon(np.array([x0, x1]))  # ends (a short pass may hold no vertex)
        la, lo = np.r_[self.lat[m], ea], np.r_[self.lon[m], eo]
        d = pad_m / 111_000
        return (float(la.min() - d), float(la.max() + d),
                float(lo.min() - d / np.cos(np.radians(la.mean()))),
                float(lo.max() + d / np.cos(np.radians(la.mean()))))

    def to_dict(self) -> dict:
        return {"kind": self.kind, "lat": self.lat.round(7).tolist(),
                "lon": self.lon.round(7).tolist(), "X": self.X.round(2).tolist(),
                "meta": self.meta}

    @classmethod
    def from_dict(cls, d: dict) -> "StreetAxis":
        return cls(d["kind"], d["lat"], d["lon"], d["X"], d.get("meta"))


def _robust_line(x, y, it: int = 4):
    """y ~ a x + b, iteratively dropping |residual| > 3 robust sigma (>= 2 m)."""
    import numpy as np
    m = np.ones(len(x), bool)
    for _ in range(it):
        a, b = np.polyfit(x[m], y[m], 1)
        r = y - (a * x + b)
        s = max(2.0, 3 * 1.4826 * np.median(np.abs(r[m])))
        m = np.abs(r) <= s
    return float(a), float(b), m


def fit_track_map(x_ref, x_street, knot_m: float = 100.0, max_shift_m: float = 15.0,
                  min_samples: int = 20):
    """Reference axis -> street axis from matched samples (visual position of the reference
    pass, GPS position projected on the street axis): robust affine fit + piecewise shifts
    every `knot_m` (median residual within +-knot_m/2: GPS errors are low-frequency, so the
    knots are wide). Returns (AxisMap, diagnostics)."""
    import numpy as np
    from .axismap import AxisMap
    x_ref, x_street = np.asarray(x_ref, float), np.asarray(x_street, float)
    a, b, ok = _robust_line(x_ref, x_street)
    y = a * x_ref + b
    r = x_street - y
    knots, shifts = [], []
    for k in np.arange(y[ok].min(), y[ok].max() + 1e-6, knot_m):
        w = ok & (np.abs(y - k) <= knot_m / 2)
        if w.sum() >= min_samples:
            knots.append(float(k))
            shifts.append(float(np.clip(np.median(r[w]), -max_shift_m, max_shift_m)))
    m = AxisMap(scale=a, offset=b)
    if len(knots) >= 3:  # each knot is already a median of a few hundred samples; no further
        m.knots = tuple(knots)  # filtering (a median filter flattens real offsets at the ends)
        m.shifts = tuple(shifts)
    res = x_street[ok] - m(x_ref[ok])
    return m, {"scale": round(a, 5), "offset_m": round(b, 2), "n_samples": int(len(x_ref)),
               "n_inliers": int(ok.sum()),
               "residual_mad_m": round(float(np.median(np.abs(res))), 2),
               "residual_p95_m": round(float(np.percentile(np.abs(res), 95)), 2)}


def gps_pass_map(cfg: dict, clip: str, axis, to_street, min_fixes: int = 50):
    """Pass -> reference axis from the pass's own GPS projected on the (OSM) street axis and
    the inverse of the reference->street map: (AxisMap, diagnostics), or None without enough
    fixes on the axis. Absolute, unlike the occupancy alignment of two passes (which locks
    onto wrong offsets on short streets whose parked cars differ between evenings)."""
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from .config import telemetry
    from .capture.telemetry import gps_lag
    tel = telemetry(cfg, clip)
    tr = pd.read_csv(Path(cfg["out_dir"]) / clip / "trace.csv")
    g = tel.gps_fix >= 3
    if g.sum() < min_fixes:
        return None
    lag = float(gps_lag(tel, tr))
    x_vis = np.interp(tel.gps_t[g] - lag, tr.t.values, tr.X_m.values)
    pr = axis.project(tel.gps[g, 0], tel.gps[g, 1])
    on = (pr[:, 0] < 15) & (tel.gps[g, 3] > 1.0)
    if on.sum() < min_fixes:
        return None
    s = pr[on, 1]
    grid = np.arange(s.min() - 200, s.max() + 200, 0.25)  # reference axis (scale ~1)
    x_ref = np.interp(s, to_street(grid), grid)
    m, fit = fit_track_map(x_vis[on], x_ref)
    if m.scale < 0:  # riding against the reference pass: y = offset - scale * x
        m.scale, m.reverse = -m.scale, True
    return m, {**fit, "gps_lag_s": round(lag, 2)}


def _no_backtracks(P, max_turn_deg: float = 90.0):
    """Vertices to keep so that the polyline never turns by more than max_turn_deg at a vertex
    (offset intersections / nodes mapped off the line make it double back, which adds arc
    length that is not street)."""
    import numpy as np
    keep = list(range(len(P)))
    changed = True
    while changed and len(keep) > 2:
        changed = False
        for j in range(1, len(keep) - 1):
            a, b, c = P[keep[j - 1]], P[keep[j]], P[keep[j + 1]]
            u, v = b - a, c - b
            nu, nv = np.hypot(*u), np.hypot(*v)
            if nu < 1e-6 or nv < 1e-6 or (u @ v) / (nu * nv) < np.cos(np.radians(max_turn_deg)):
                del keep[j]
                changed = True
                break
    return np.array(keep)


def build_axis(cfg: dict, ref: str):
    """Street axis for a run: (axis | None, reference->street map | None, crossings DataFrame,
    diagnostics). Needs the reference pass's GPS (none: the reference axis, no geography).
    street.axis: 'osm' (mapped centreline, `street.osm_json`), 'track' (reference GPS track),
    'auto' (osm when the extract exists, else track)."""
    import json
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from .capture.telemetry import gps_lag
    sc = cfg["street"]
    root = Path(cfg["out_dir"])
    from .config import telemetry
    tel = telemetry(cfg, ref)
    tr = pd.read_csv(root / ref / "trace.csv")
    g = tel.gps_fix >= 3
    none = pd.DataFrame()
    if g.sum() < 50:
        return None, None, none, {"axis": "reference", "why": f"{ref} has no GPS fix"}
    lag = float(gps_lag(tel, tr))
    t, ll = tel.gps_t[g], tel.gps[g, :2]
    x_vis = np.interp(t - lag, tr.t.values, tr.X_m.values)  # GPS time t = video time t - lag
    mode = sc.get("axis", "auto")
    osm_path = sc.get("osm_json")
    if mode == "auto":
        mode = "osm" if osm_path and Path(osm_path).exists() else "track"
    diag = {"axis": mode, "gps_lag_s": round(lag, 2)}
    if mode == "track":
        from scipy.ndimage import uniform_filter1d
        lat0, lon0 = float(ll[:, 0].mean()), float(ll[:, 1].mean())
        P = uniform_filter1d(local_xy(ll[:, 0], ll[:, 1], lat0, lon0), size=20, axis=0,
                             mode="nearest")
        # back to lat/lon (inverse of the equirectangular local frame)
        R = 6371000.0
        lat = lat0 + np.degrees(P[:, 1] / R)
        lon = lon0 + np.degrees(P[:, 0] / (R * np.cos(np.radians(lat0))))
        return StreetAxis("track", lat[::5], lon[::5], x_vis[::5], diag), None, none, diag
    if mode != "osm":
        raise ValueError(f"street.axis {mode!r}: osm, track or auto")
    from . import osm as O
    data = json.loads(Path(osm_path).read_text())
    pl, path, cr, name, chained = O.street_polyline(data, sc.get("name"), ll)
    lat0, lon0 = float(ll[:, 0].mean()), float(ll[:, 1].mean())
    P = local_xy(pl[:, 0], pl[:, 1], lat0, lon0)
    keep = _no_backtracks(P)
    pl, P = pl[keep], P[keep]
    S = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    pr = project_polyline(local_xy(ll[:, 0], ll[:, 1], lat0, lon0), P, S)
    on = (pr[:, 0] < 15) & (tel.gps[g, 3] > 1.0)
    if on.sum() < 50:
        raise RuntimeError(f"reference pass {ref} is not on {name!r} (OSM): {on.sum()} samples")
    # orient with the reference pass, origin at its start
    a, b, _ = _robust_line(x_vis[on], pr[on, 1])
    sign = 1.0 if a > 0 else -1.0
    X = sign * (S - b)
    o = np.argsort(X)  # store vertices in axis order
    axis = StreetAxis("osm", pl[o, 0], pl[o, 1], X[o],
                      {**diag, "name": name, "osm_nodes": [int(n) for n in np.asarray(path)[keep][o]],
                       "chained": chained, "osm_timestamp": data.get("osm3s", {})
                       .get("timestamp_osm_base"), "source": "(c) OpenStreetMap contributors, ODbL"})
    to_street, fit = fit_track_map(x_vis[on], sign * (pr[on, 1] - b))
    axis.meta["fit"] = fit
    rows = []
    node_ll = O.node_coords(data)
    for c in cr:
        x = float(axis.project([node_ll[c["node"]][0]], [node_ll[c["node"]][1]])[0, 1])
        sides = set()
        for nb in c["neighbours"]:
            sides.add("left" if axis.project([nb[0]], [nb[1]])[0, 2] > 0 else "right")
        rows.append({"x_m": round(x, 1), "kind": c["kind"], "name": c["name"],
                     "highway": c["highway"], "service": c["service"], "osm_node": c["node"],
                     "osm_way": c["way"], "sides": "+".join(sorted(sides)) or "unknown"})
    cross = pd.DataFrame(rows)
    if len(cross):  # one row per node and kind; sides merged over the ways meeting there
        # the street's own name under another spelling (OSM "Prince-of-Wales" past a
        # crossing) is not a cross street
        def own(v):
            return " ".join(str(v).lower().replace("-", " ").split()) == \
                " ".join(str(name).lower().replace("-", " ").split())
        cross = (cross.groupby(["osm_node", "kind"], as_index=False)
                 .agg(x_m=("x_m", "first"),
                      name=("name", lambda v: " / ".join(sorted(set(v) - {""} - {u for u in v if own(u)}))),
                      highway=("highway", "first"), service=("service", "first"),
                      osm_way=("osm_way", "first"),
                      sides=("sides", lambda v: "+".join(sorted({s for x in v for s in x.split("+")})))))
        cross = cross.sort_values("x_m").reset_index(drop=True)
    return axis, to_street, cross, {**diag, "name": name, "fit": fit}


def merge_osm_intersections(det: pd.DataFrame, cross: pd.DataFrame, extent, snap_m: float = 15.0,
                            width_m: float = 10.0, reach_m: float = 40.0):
    """Intersections from the map, extents from the cameras: each OSM cross street inside the
    extent (or within `reach_m` of its ends: a pass starting or stopping short of a corner
    still has the block's end; the supply stage then checks the curb it did not see) takes the extent of a detected deep stretch within `snap_m`, else +-width_m/2
    around its node (e.g. T-junctions on the curb behind a camera looking the other way).
    Detected stretches without a mapped cross street are dropped (open yards, lots).
    Returns (intersections, dropped)."""
    import numpy as np
    import pandas as pd
    cols = ["x0_m", "x1_m", "centre_m", "name", "sides", "source", "osm_node"]
    c = (cross[(cross.kind == "intersection")
               & cross.x_m.between(extent[0] - reach_m, extent[1] + reach_m)]
         if len(cross) else cross)
    used, rows = set(), []
    for r in c.itertuples():
        near = det[(det.x0_m - snap_m <= r.x_m) & (det.x1_m + snap_m >= r.x_m)] if len(det) else det
        if len(near):
            k = int(np.argmin(np.abs(near.centre_m.values - r.x_m)))
            d = near.iloc[k]
            used.add(near.index[k])
            rows.append((float(d.x0_m), float(d.x1_m), r.name, r.sides, "osm+detected", r.osm_node))
        else:
            rows.append((r.x_m - width_m / 2, r.x_m + width_m / 2, r.name, r.sides, "osm",
                         r.osm_node))
    # several nodes snapped to one stretch (a boulevard's two carriageways): one row
    out = pd.DataFrame(rows, columns=["x0_m", "x1_m", "name", "sides", "source", "osm_node"])
    if len(out):
        out = (out.groupby(["x0_m", "x1_m"], as_index=False)
               .agg(name=("name", lambda v: " / ".join(dict.fromkeys(v))),
                    sides=("sides", lambda v: "+".join(sorted({s for x in v for s in x.split("+")}))),
                    source=("source", "first"), osm_node=("osm_node", "first")))
        out["centre_m"] = (out.x0_m + out.x1_m) / 2
        out = out.sort_values("x0_m").reset_index(drop=True)[cols]
    dropped = det.drop(index=list(used)).assign(reason="not_in_osm") if len(det) else det
    return (out if len(out) else pd.DataFrame(columns=cols)), dropped
