"""Street layout, fused over all passes of one street.

The street has a few parking rows (config `street.rows`), each imaged by one mosaic plane of a
pass depending on its riding direction, and the set of rows actually present changes only
rarely along the street (`street.states` lists the allowed sets). Default, rue de Terrebonne:
curb | bike lane | parking | travel lane | bike lane | curb with the parking lane switching
sides, so a pass sees the parked row either in its near plane (riding beside it, ~1.5 m) or in
its far plane (across the travel lane, ~5 m), piecewise constant with very few switches.

That one fact removes most false positives: where a plane's row is absent, what it sees is the
travel lane or the other side (far-parking segment: traffic at 1.5-3.5 m in the near band;
near-parking segment: traffic behind the row at 3.5-5 m in the far band). Gating detections by
row presence drops moving/stopped traffic (school buses) and near cars re-detected in the far
mosaic in one rule, and the present-row frames give an uncontaminated estimate of each plane's
depth.

The layout lives on the street axis. Every clip is first mapped onto the reference clip's
axis with an AxisMap (affine + piecewise shift; reversed for the opposite direction, fitted on
vehicle occupancy), then onto the street axis (identity when the street axis is the reference
axis). Per-row evidence (kept, static detections, each attributed to the row its plane images)
is summed there, and a Viterbi over the allowed row sets with a large switch cost segments it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .axismap import AxisMap, MapChain, fit_axis_map, local_shifts, map_from_dict
from .street import plane_pairs, plane_rows, rows_cfg, state_label


def load_dets(clip_dir: Path, planes) -> dict[str, pd.DataFrame]:
    out = {}
    for p in planes:
        d = pd.read_csv(clip_dir / f"det_{p}.csv", keep_default_na=False, na_values=[""])
        d = d.fillna({"reject": "", "flag": ""})
        d["kept"] = d["kept"].astype(str).str.lower() == "true"
        out[p] = d
    return out


def kept_all(dets: dict[str, pd.DataFrame], static_only: bool = False) -> pd.DataFrame:
    rows = [d[d.kept & ~(static_only & (d.flag == "moving"))].assign(plane=p)
            for p, d in dets.items()]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


@dataclass
class Layout:
    reference: str
    directions: dict                 # clip -> direction label
    maps: dict                       # clip -> map (clip x -> street-axis x)
    switches_m: list                 # street-axis positions where the set of present rows changes
    states: list                     # rows present in each segment (tuples), len = switches+1
    extent_m: tuple                  # street-axis span covered by the passes
    intersections: pd.DataFrame
    blocks: pd.DataFrame
    rows: dict = field(default_factory=dict)          # street.rows (curb, planes per direction)
    plane_depth_h: dict = field(default_factory=dict)  # clip -> plane -> depth (h) or None
    ref_maps: dict = field(default_factory=dict)       # clip -> AxisMap onto the reference axis
    to_street: AxisMap | None = None                   # reference axis -> street axis (None: same)
    axis: object = None                                # street.StreetAxis (None: no geography)
    crossings: pd.DataFrame = field(default_factory=pd.DataFrame)  # mapped cross streets, cuts

    # ---- queries
    @property
    def axis_is_reference(self) -> bool:
        return self.to_street is None

    def _seg(self, x_ref) -> np.ndarray:
        return np.searchsorted(np.asarray(self.switches_m), np.asarray(x_ref, float))

    def present(self, row: str, x_ref) -> np.ndarray:
        """Is `row` present at street-axis positions `x_ref`?"""
        return np.array([row in s for s in self.states], bool)[self._seg(x_ref)]

    def state_at(self, x_ref) -> np.ndarray:
        """Label of the present rows ('A', 'A+B', 'none') at street-axis positions."""
        return np.array([state_label(s) for s in self.states], dtype=object)[self._seg(x_ref)]

    def reversed(self, clip: str) -> bool:
        return self.directions[clip] != self.directions[self.reference]

    def row_of(self, clip: str, plane: str) -> str | None:
        """Row imaged by `plane` of `clip` (None: the plane images no row)."""
        return plane_rows(self.rows, self.reversed(clip)).get(plane)

    def plane_active(self, clip: str, plane: str, x_clip) -> np.ndarray:
        """Does `plane` of `clip` image a present row at clip-axis positions `x_clip`?"""
        r = self.row_of(clip, plane)
        x = np.asarray(x_clip, float)
        if r is None:
            return np.zeros(x.shape, bool)
        return self.present(r, self.maps[clip](x))

    def segments(self, clip: str | None = None) -> pd.DataFrame:
        e = [self.extent_m[0], *self.switches_m, self.extent_m[1]]
        rows = []
        for k, st in enumerate(self.states):
            r = {"x0_ref_m": round(e[k], 1), "x1_ref_m": round(e[k + 1], 1),
                 "rows": state_label(st)}
            if clip is not None:
                pr = {v: p for p, v in plane_rows(self.rows, self.reversed(clip)).items()}
                r["planes"] = state_label([pr[v] for v in st if v in pr])
            rows.append(r)
        return pd.DataFrame(rows)

    # ---- io
    def save(self, out: Path):
        out.mkdir(parents=True, exist_ok=True)
        meta = {"reference": self.reference, "directions": self.directions,
                "maps": {c: m.to_dict() for c, m in self.ref_maps.items()},
                "to_street": self.to_street.to_dict() if self.to_street is not None else None,
                "switches_m": [float(v) for v in self.switches_m],
                "states": [list(v) for v in self.states], "rows": self.rows,
                "extent_m": [float(v) for v in self.extent_m],
                "plane_depth_h": self.plane_depth_h}
        (out / "layout.json").write_text(json.dumps(meta, indent=2))
        if self.axis is not None:
            (out / "axis.json").write_text(json.dumps(self.axis.to_dict()))
        self.crossings.to_csv(out / "osm_crossings.csv", index=False)
        self.intersections.to_csv(out / "intersections.csv", index=False, float_format="%.1f")
        self.blocks.to_csv(out / "blocks.csv", index=False)
        self.segments().to_csv(out / "segments.csv", index=False)

    @classmethod
    def load(cls, out: Path) -> "Layout | None":
        p = Path(out) / "layout.json"
        if not p.exists():
            return None
        m = json.loads(p.read_text())
        if "states" not in m:
            raise RuntimeError(f"{p} predates street rows: rerun `curb-occupancy layout`")

        def csv(name):
            f = Path(out) / name
            return pd.read_csv(f) if f.stat().st_size > 1 else pd.DataFrame()
        ref_maps = {c: map_from_dict(d) for c, d in m["maps"].items()}
        to_street = map_from_dict(m["to_street"]) if m.get("to_street") else None
        from .street import StreetAxis
        ax = Path(out) / "axis.json"
        axis = StreetAxis.from_dict(json.loads(ax.read_text())) if ax.exists() else None
        cr = Path(out) / "osm_crossings.csv"
        crossings = pd.read_csv(cr) if cr.exists() and cr.stat().st_size > 1 else pd.DataFrame()
        return cls(reference=m["reference"], directions=m["directions"],
                   maps=compose_maps(ref_maps, to_street), switches_m=m["switches_m"],
                   states=[tuple(v) for v in m["states"]], extent_m=tuple(m["extent_m"]),
                   intersections=csv("intersections.csv"), blocks=csv("blocks.csv"),
                   rows=m["rows"], plane_depth_h=m.get("plane_depth_h", {}),
                   ref_maps=ref_maps, to_street=to_street, axis=axis, crossings=crossings)


def compose_maps(ref_maps: dict, to_street) -> dict:
    return dict(ref_maps) if to_street is None else \
        {c: MapChain((m, to_street)) for c, m in ref_maps.items()}


# ------------------------------------------------------------------------------- fitting

def row_evidence(kept: pd.DataFrame, amap, rows_of_plane: dict, rows, grid: np.ndarray):
    """Per-bin evidence (metres of box coverage x confidence) for each row; boxes of planes
    that image no row for this pass are ignored."""
    step = grid[1] - grid[0]
    ev = {r: np.zeros(len(grid)) for r in rows}
    if not len(kept):
        return ev
    a, b = amap.interval(kept.x_start_m.values, kept.x_end_m.values)
    for x0, x1, plane, c in zip(a, b, kept.plane.values, kept.conf.values):
        r = rows_of_plane.get(plane)
        if r is not None:
            ev[r][(grid >= x0) & (grid < x1)] += c * step
    return ev


def viterbi_states(E: np.ndarray, M: np.ndarray, switch_cost: float,
                   row_cost: float = 0.0, force: np.ndarray | None = None,
                   n_free: int | None = None) -> np.ndarray:
    """Segmentation into allowed row sets. E (n bins, R rows): evidence per row; M (K states,
    R): rows present in each state. Per-bin cost of a state = evidence for rows it lacks +
    `row_cost` per row it has; changing state costs `switch_cost` (units: metres of confident
    box coverage). `force` (n,): state index imposed on a bin, -1 = free; states from index
    `n_free` on exist only where forced. Returns the state index per bin."""
    n, K = len(E), len(M)
    unary = E @ (~M).T.astype(float) + row_cost * M.sum(1)[None, :]
    if force is not None:
        f = force >= 0
        unary[~f, n_free if n_free is not None else K:] = np.inf
        unary[f] = np.inf
        unary[f, force[f]] = 0.0
    D = np.zeros((n, K))
    bt = np.zeros((n, K), np.int16)
    D[0] = unary[0]
    for i in range(1, n):
        for s in range(K):
            other = np.where(np.arange(K) != s, D[i - 1] + switch_cost, np.inf)
            j = int(np.argmin(other))
            stay = D[i - 1, s]
            bt[i, s] = s if stay <= other[j] else j
            D[i, s] = min(stay, other[j]) + unary[i, s]
    st = np.zeros(n, np.int16)
    st[-1] = int(np.argmin(D[-1]))
    for i in range(n - 1, 0, -1):
        st[i - 1] = bt[i, st[i]]
    return st


def forced_segments(segs: list, cross: pd.DataFrame, grid: np.ndarray, rnames: list,
                    M: np.ndarray):
    """Config `street.segments` (site knowledge: [{between: [cross street, cross street] |
    x_m: [a, b], rows: [A, B]}]) -> (M with any new row set appended, state forced per bin
    (-1 = free), forced boundaries in street metres). `between` uses the OSM cross-street
    names (centres)."""
    force = np.full(len(grid), -1, int)
    bounds = []
    names = {}
    if len(cross) and "name" in cross:
        ii = cross[cross.kind == "intersection"]
        names = dict(zip(ii["name"].fillna(""), ii.x_m))
    for sg in segs or []:
        if "between" in sg:
            missing = [n for n in sg["between"] if n not in names]
            if missing:
                raise ValueError(f"street.segments between {missing}: cross streets {sorted(names)}")
            a, b = sorted(float(names[n]) for n in sg["between"])
        else:
            a, b = sorted(float(v) for v in sg["x_m"])
        v = np.array([r in sg["rows"] for r in rnames])
        k = next((i for i, m in enumerate(M) if (m == v).all()), None)
        if k is None:
            M = np.vstack([M, v])
            k = len(M) - 1
        force[(grid + 0.5 * (grid[1] - grid[0]) >= a) & (grid < b)] = k
        bounds += [a, b]
    return M, force, bounds


def _place_switches(states, grid, e_tot, intersections: pd.DataFrame):
    """Viterbi puts a switch at an arbitrary bin of the evidence-free gap between the two
    states' last/first vehicles; move it to an intersection inside that gap (parking usually
    changes side at a cross street) or else the gap's midpoint."""
    sw = []
    for i in np.where(np.diff(states) != 0)[0]:
        lo = i
        while lo > 0 and e_tot[lo] == 0:
            lo -= 1
        hi = i + 1
        while hi < len(grid) - 1 and e_tot[hi] == 0:
            hi += 1
        g0, g1 = grid[lo], grid[hi]
        x = (g0 + g1) / 2
        if len(intersections):
            c = intersections.centre_m.values
            inside = c[(c > g0) & (c < g1)]
            if len(inside):
                x = float(inside[np.argmin(np.abs(inside - x))])
        sw.append(float(x))
    return sw


AUTO = "auto"


def _resolve_directions(labels: dict, rev: dict, ref: str) -> dict:
    """Direction label per clip from its fitted reversal relative to the reference clip.
    Configured labels are kept; 'auto' clips borrow the label of a configured clip riding
    the same way (inventing '<label>_rev' when there is none)."""
    known = {c: l for c, l in labels.items() if l != AUTO}
    same = [l for c, l in known.items() if not rev[c]]
    opp = [l for c, l in known.items() if rev[c]]
    r_lab = same[0] if same else ((opp[0] + "_rev") if opp else "d1")
    o_lab = opp[0] if opp else next((l for l in known.values() if l != r_lab), r_lab + "_rev")
    return {c: labels[c] if labels[c] != AUTO else (o_lab if rev[c] else r_lab) for c in rev}


def pin_intersections(ints: pd.DataFrame, pins: list[dict]) -> pd.DataFrame:
    """Config `street.intersections`: roadway edges checked on the imagery replace the
    detected ones ({name, x0_m?, x1_m?}); source becomes 'config'."""
    if not pins:
        return ints
    ints = ints.copy()
    if "source" not in ints:
        ints["source"] = ""
    for p in pins:
        m = (ints["name"] == p["name"]) if "name" in ints else np.zeros(len(ints), bool)
        if m.sum() != 1:
            raise ValueError(f"street.intersections: {p['name']!r} matches {int(m.sum())} intersections")
        for k in ("x0_m", "x1_m"):
            if p.get(k) is not None:
                ints.loc[m, k] = float(p[k])
        if ints.loc[m, "x0_m"].item() >= ints.loc[m, "x1_m"].item():
            raise ValueError(f"street.intersections: {p['name']!r} edges cross")
        ints.loc[m, "source"] = "config"
    ints["centre_m"] = (ints.x0_m + ints.x1_m) / 2
    return ints.sort_values("x0_m").reset_index(drop=True)


def dark_clip(clip_dir: Path, max_exposure: float | None) -> bool:
    """Too dark for the horizon depth: median ISO x exposure time (s) of the clip above
    `max_exposure` (HERO8 study clips: daylight 0.5-2, dusk 5-16, night 53-107). False
    without exposure telemetry or threshold."""
    f = clip_dir / "telemetry.npz"
    if max_exposure is None or not f.exists():
        return False
    z = np.load(f)
    s, iso = z["shutter_s"], z["iso"]
    n = min(len(s), len(iso))
    if not n:
        return False
    e = s[:n] * iso[:n]
    e = e[e > 0]
    return bool(len(e)) and float(np.median(e)) > max_exposure


def fit_layout(cfg: dict) -> Layout:
    from .blocks import blocks_from_intersections, depth_on_grid, find_intersections
    from .mosaic import parked_depth

    lc = cfg["layout"]
    root = Path(cfg["out_dir"])
    planes = list(cfg["mosaic"]["planes"])
    rows = rows_cfg(cfg)
    rnames = list(rows)
    M = np.array([[r in s for r in rnames] for s in cfg["street"]["states"]], bool)
    clips = [c["id"] for c in cfg["clips"]]
    ref = lc.get("reference") or clips[0]
    labels = {c["id"]: c.get("direction", "d1") for c in cfg["clips"]}
    dets = {c: load_dets(root / c, planes) for c in clips}
    traces = {c: pd.read_csv(root / c / "trace.csv") for c in clips}
    L = {c: float(traces[c].X_m.iloc[-1]) for c in clips}
    kept = {c: kept_all(dets[c]) for c in clips}  # occupancy for alignment
    static = {c: kept_all(dets[c], static_only=True) for c in clips}  # row evidence

    from scipy.ndimage import median_filter

    from .street import build_axis, gps_pass_map, merge_osm_intersections
    axis, to_street, cross, axis_diag = build_axis(cfg, ref)
    if axis is not None and axis.kind == "track":
        to_street = None  # the track sits on the reference axis

    maps, diag, rev = {ref: AxisMap()}, {}, {ref: False}
    for c in clips:
        if c == ref:
            continue
        # occupancy alignment with the reference pass; unknown direction: both ways, keep the
        # better one (a way in which the two passes image no common row is out)
        ways = (labels[c] != labels[ref],) if AUTO not in (labels[c], labels[ref]) \
            else (False, True)
        fits = {r: fit_axis_map(kept[ref], kept[c], L[ref], L[c], r, pp)
                for r in ways if (pp := plane_pairs(rows, False, r))}
        if fits:
            rev[c] = max(fits, key=lambda r: fits[r][1]["score_m"])
            maps[c], diag[c] = fits[rev[c]]
            if len(ways) == 2:
                diag[c]["direction_scores_m"] = {("reverse" if r else "same"): f[1]["score_m"]
                                                 for r, f in fits.items()}
        # checked against the pass's own GPS on the mapped street: occupancy is the finer map
        # when the passes share many parked cars (Terrebonne), but can lock onto a wrong
        # offset on short streets whose cars differ between evenings (side streets: 25-65 m)
        gm = gps_pass_map(cfg, c, axis, to_street) if to_street is not None else None
        if gm is None or (len(ways) == 1 and gm[0].reverse != ways[0]):
            if gm is not None:
                print(f"  warning: {c} rides the other way by GPS than its direction label")
            if not fits:
                raise ValueError(f"{c} and {ref} image no common row")
            continue
        if fits and rev[c] == gm[0].reverse:
            xs = np.linspace(0.0, L[c], 50)
            dev = float(np.abs(maps[c](xs) - gm[0](xs)).max())
            diag[c]["gps_max_dev_m"] = round(dev, 1)
            if dev <= lc["gps_tolerance_m"]:
                continue
        # GPS global map, refined locally by occupancy within the GPS along-street error
        rev[c] = gm[0].reverse
        maps[c] = gm[0]
        diag[c] = {"global": "gps", "gps_fit": gm[1], "occupancy_fit": diag.get(c)}
        pp = plane_pairs(rows, False, rev[c])
        if pp:
            cbm = kept[c].assign(xa_start_m=0.0, xa_end_m=0.0)
            cbm["xa_start_m"], cbm["xa_end_m"] = gm[0].interval(kept[c].x_start_m,
                                                                kept[c].x_end_m)
            cen, sh = local_shifts(kept[ref], cbm, L[ref], pp, max_shift=5.0)
            ok = np.isfinite(sh)
            if ok.sum() >= 2:
                maps[c] = MapChain((gm[0], AxisMap(knots=tuple(cen[ok]), shifts=tuple(
                    median_filter(sh[ok], size=3, mode="mirror")))))
            diag[c].update(local_centres_m=cen.tolist(), local_shifts_m=sh.tolist())
        dev = (diag[c]["occupancy_fit"] or {}).get("gps_max_dev_m")
        print(f"  {c}: occupancy alignment {dev} m off its GPS (or other way): GPS map")
    if lc["register"]["enabled"] and len(clips) > 1:
        from .register import register_passes
        scores = {c: float((d.get("score_m") if "score_m" in d else
                            (d.get("occupancy_fit") or {}).get("score_m", -1.0)) or -1.0)
                  - (1e6 if d.get("global") == "gps" else 0.0) for c, d in diag.items()}
        maps, reg = register_passes(cfg, root, clips, ref, rev, maps, L, rows, kept, scores)
        for c, r in reg.items():
            diag.setdefault(c, {})["register"] = r
            print(f"  {c}: {r}")
    dirs = _resolve_directions(labels, rev, ref)
    ref_maps = maps
    maps = compose_maps(ref_maps, to_street)
    ends = np.array([maps[c].interval(0.0, L[c]) for c in clips], dtype=float).reshape(-1, 2)
    extent = (float(ends[:, 0].min()), float(ends[:, 1].max()))
    grid = np.arange(extent[0], extent[1], lc["bin_m"])

    ev = {r: np.zeros(len(grid)) for r in rnames}
    per_clip, pr = {}, {}
    for c in clips:
        pr[c] = plane_rows(rows, rev[c])
        e = row_evidence(static[c], maps[c], pr[c], rnames, grid)
        per_clip[c] = e
        for r in rnames:
            ev[r] += e[r]
    E = np.stack([ev[r] for r in rnames], 1)
    n_free = len(M)
    M, force, fbounds = forced_segments(cfg["street"].get("segments"), cross, grid, rnames, M)
    st = viterbi_states(E, M, lc["switch_cost_m"], lc["row_cost"], force, n_free)

    # cross streets: deep horizon views seen by the passes at the same street position;
    # parked-row vehicles (static, on a row present there) veto deep stretches that are open
    # yards/lots behind the row rather than cross streets
    present_bin = M[st]
    centres = []
    for c in clips:
        k = static[c]
        if not len(k):
            centres.append(np.array([]))
            continue
        a, b = maps[c].interval(k.x_start_m.values, k.x_end_m.values)
        x = (a + b) / 2
        ri = np.array([rnames.index(pr[c][p]) if p in pr[c] else -1 for p in k.plane.values])
        bi = np.clip(np.searchsorted(grid, x), 0, len(grid) - 1)
        on_row = (ri >= 0) & present_bin[bi, np.maximum(ri, 0)]
        centres.append(x[on_row])
    # dark passes do not vote: at night the horizon row sees past the unlit parked row (25-34 m
    # on curb that dusk reads 13-20 m) and widens cross streets; a run without a lit pass
    # keeps them all
    lit = [i for i, c in enumerate(clips) if not dark_clip(root / c, lc["dark_exposure"])]
    vote = lit if lit else list(range(len(clips)))
    diag["intersection_votes"] = [clips[i] for i in vote]
    ints, dropped = find_intersections([depth_on_grid(traces[clips[i]], maps[clips[i]], grid)
                                        for i in vote], grid,
                                       groups=[dirs[clips[i]] for i in vote],
                                       cars_ref=[centres[i] for i in vote])
    dropped = dropped.assign(reason="parked_cars")
    if axis is not None and axis.kind == "osm":
        sc = cfg["street"]
        ints, not_mapped = merge_osm_intersections(ints, cross, extent, sc["osm_snap_m"],
                                                   sc["cross_street_m"])
        dropped = pd.concat([dropped, not_mapped], ignore_index=True)
    ints = pin_intersections(ints, cfg["street"].get("intersections") or [])
    blocks = blocks_from_intersections(ints, extent[1], x_min=extent[0])
    switches = _place_switches(st, grid, E.sum(1), ints)
    for b in fbounds:   # a forced segment ends exactly where configured
        if switches:
            j = int(np.argmin(np.abs(np.array(switches) - b)))
            if abs(switches[j] - b) <= 2 * lc["bin_m"]:
                switches[j] = b
    seq = [int(st[0])] + [int(st[i + 1]) for i in np.where(np.diff(st) != 0)[0]]
    states = [tuple(r for r, m in zip(rnames, M[k]) if m) for k in seq]
    lay = Layout(ref, dirs, maps, switches, states, extent, ints, blocks, rows=rows,
                 ref_maps=ref_maps, to_street=to_street, axis=axis, crossings=cross)

    # plane depth per clip from its horizon-depth samples where the plane images a present row
    from .config import height, plane_band_h
    for c in clips:
        tr = traces[c]
        lay.plane_depth_h[c] = {}
        for p in cfg["mosaic"]["planes"]:
            lo, hi = plane_band_h(cfg, p, height(cfg, c))
            m = lay.plane_active(c, p, tr.X_m.values) & (tr.speed_kmh.values > 3)
            Z = tr.Z_h.values[m]
            n_ok = int(np.sum((Z > lo) & (Z < hi)))
            lay.plane_depth_h[c][p] = (parked_depth(Z, lo, hi)
                                       if n_ok >= lc["min_depth_samples"] else None)
    out = root / "layout"
    lay.save(out)
    (out / "alignment.json").write_text(json.dumps({**diag, "street_axis": axis_diag}, indent=2))
    dropped.to_csv(out / "intersections_dropped.csv", index=False, float_format="%.1f")
    _plot(lay, grid, ev, per_clip, clips, out / "layout.png")
    return lay


def _plot(lay: Layout, grid, ev, per_clip, clips, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import uniform_filter1d

    fig, ax = plt.subplots(len(clips) + 1, 1, figsize=(20, 2.2 * (len(clips) + 1)), sharex=True)
    rows = [("fused", ev)] + [(c, per_clip[c]) for c in clips]
    w = max(1, int(10 / (grid[1] - grid[0])))
    colors = ["#2b6cb0", "#c05621", "#2f855a", "#6b46c1"]
    for a, (name, e) in zip(ax, rows):
        for k, r in enumerate(lay.rows):
            sign = 1 if k % 2 == 0 else -1  # alternate rows above/below the axis
            a.plot(grid, sign * uniform_filter1d(e[r], w), color=colors[k % len(colors)], lw=1,
                   label=f"row {r} ({lay.rows[r]['curb']} curb)")
        a.axhline(0, color="0.5", lw=0.5)
        a.set_ylabel(name, fontsize=8)
        for r in lay.intersections.itertuples():
            a.axvspan(r.x0_m, r.x1_m, color="0.85", zorder=-1)
        for s in lay.switches_m:
            a.axvline(s, color="k", lw=1.5, ls="--")
    seg = lay.segments()
    for r in seg.itertuples():
        ax[0].text((r.x0_ref_m + r.x1_ref_m) / 2, ax[0].get_ylim()[1] * 0.8,
                   f"parked: {r.rows}", ha="center", fontsize=9)
    ax[0].legend(loc="lower left", fontsize=8)
    ax[-1].set_xlabel(f"street axis m (reference {lay.reference}); grey = intersections, "
                      f"dashed = change of parked rows")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)
