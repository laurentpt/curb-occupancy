"""Command-line pipeline. Every stage caches its outputs under <out_dir>/<clip_id>/ and is
skipped when they exist (use --force to recompute).

    curb-occupancy fetch <study>/streets/<street>.yaml                     # external data (city open data, map)
    curb-occupancy run <study>/streets/<street>.yaml                       # all stages, all clips
    curb-occupancy <stage> <study>/streets/<street>.yaml [--clip ID] [--force]
    curb-occupancy serve <study>/streets/<street|area>.yaml [--port 8000]  # map pages over http (OSM tiles)
    curb-occupancy area <study>/streets/<area>.yaml                        # merge street runs (tables, public map)
    curb-occupancy segment <study>/rides/<ride>.yaml [--dry-run]           # long recording -> street clip windows

Stages (in order): telemetry heading orient profiles trace mosaic detect [layout] count
multiday supply map  (run-level, over all clips: layout multiday supply map)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

STAGES = ["telemetry", "heading", "orient", "profiles", "trace", "mosaic", "detect", "count"]
RUN_STAGES = ["layout", "multiday", "supply", "map"]
SETUP = ["fetch"]  # external data (city open data, map)
TOOLS = ["serve", "area", "segment"]  # local http server for the map pages; merge of street
# runs; one long recording over several streets -> clip windows in the street configs
# `run` fits the street layout after `detect` (all clips), redoes mosaic/detect for planes
# whose layout-based depth moved, refits, then counts.
PRE_LAYOUT = STAGES[:STAGES.index("count")]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class Ctx:
    def __init__(self, cfg: dict, clip: dict, force: bool):
        self.cfg, self.clip, self.force = cfg, clip, force
        self.video = C.source(cfg, clip["id"])  # chapters cut to t_range (video.Source)
        self.dir = Path(cfg["out_dir"]) / clip["id"]
        self.cam = clip["camera"]            # street camera defaults + per-clip overrides
        self.dir.mkdir(parents=True, exist_ok=True)
        self._cache = {}
        self._check_source()

    def _check_source(self):
        """The cached outputs belong to one window of the recording (source.json). Outputs
        from before windows (no source.json) are whole-file ones: kept when the window is the
        whole file. A changed window invalidates every per-clip output: refuse to mix."""
        f = self.p("source.json")
        d = self.video.describe()
        if f.exists():
            old = json.loads(f.read_text())
            if old == d:
                return
            why = f"was {old['files']} frames {old['frames']}"
        elif not any(self.dir.iterdir()):
            why = None
        elif self.video.i0 == 0 and self.video.i1 == sum(c.n_frames
                                                         for c in self.video.chapters):
            why = None
        else:
            why = "cached outputs of the whole file"
        if why:
            raise SystemExit(
                f"{self.dir}: clip window changed ({d['files']} frames {d['frames']}; {why}). "
                f"Move or delete this directory and rerun.")
        f.write_text(json.dumps(d, indent=2))

    @property
    def h(self) -> float:
        """Effective camera height (m): GPS-calibrated once the trace stage has run."""
        return C.height(self.cfg, self.clip["id"])

    def p(self, name):
        return self.dir / name

    def done(self, *names):
        return not self.force and all(self.p(n).exists() for n in names)

    # lazily loaded shared objects
    @property
    def info(self):
        from .capture.video import probe
        if "info" not in self._cache:
            self._cache["info"] = probe(self.video)
        return self._cache["info"]

    @property
    def lens(self):
        from .capture.lens import FisheyeLens
        if "lens" not in self._cache:
            self._cache["lens"] = FisheyeLens.from_gyroflow(
                self.cam["lens_profile"], self.info.width, self.info.height)
        return self._cache["lens"]

    @property
    def tel(self):
        if "tel" not in self._cache:
            self._cache["tel"] = C.telemetry(self.cfg, self.clip["id"])
        return self._cache["tel"]

    @property
    def trace(self) -> pd.DataFrame:
        if "trace" not in self._cache:
            self._cache["trace"] = pd.read_csv(self.p("trace.csv"))
        return self._cache["trace"]


# ---------------------------------------------------------------------------------- stages

def _video_json(x: Ctx) -> None:
    """video.json = the window's video info; `creation_time` = local start time of the window,
    from the GPS clock when `camera.timezone` is set and the recording has a 3D fix (the camera
    clock as set by hand: no DST change, drift), else the camera's (`clock` says which)."""
    from .capture import telemetry as T
    d = x.info.to_dict()
    tz = x.cam.get("timezone")
    d.update(creation_time_camera=d["creation_time"], clock="camera", clock_zone=tz)
    u0 = T.gps_utc0(x.video.files) if tz else None
    if u0 is not None:
        d["creation_time"] = T.local_start(u0, x.video.i0 / x.video.fps, tz)
        d["clock"] = "gps"
        log(f"  start {d['creation_time']} {tz} (GPS clock; camera clock "
            f"{d['creation_time_camera'][:19]})")
    x.p("video.json").write_text(json.dumps(d, indent=2))


def st_telemetry(x: Ctx):
    from .capture import telemetry as T
    if x.done("telemetry.npz"):
        v = json.loads(x.p("video.json").read_text()) if x.p("video.json").exists() else {}
        if v.get("clock_zone", "-") != x.cam.get("timezone"):   # older run / zone changed
            _video_json(x)
        return
    tel = T.extract_source(x.video, x.cam["imu_to_cv"])
    tel.save(x.p("telemetry.npz"))
    _video_json(x)
    s = tel.summary()
    log(f"  GPS fix fraction {s['gps_fix_fraction']:.2f}, HiLights {len(s['hilights_s'])}, "
        f"shutter 1/{s['shutter_1_over']}, ISO {s['iso_range']}")


def st_heading(x: Ctx):
    from .capture.motion import estimate_heading
    if x.done("heading.npz"):
        return
    h = x.cfg["heading"]
    hd = estimate_heading(x.video, x.lens, step=h["step"], every=h["every"])
    hd.save(x.p("heading.npz"))


def st_orient(x: Ctx):
    from .capture import motion as M
    if x.done("rotations.npy", "orientation.csv"):
        return
    hd = M.HeadingTrack.load(x.p("heading.npz"))
    t = np.arange(x.info.n_frames) / x.info.fps
    fwd, down, diag = M.street_frames(t, x.tel, hd, smooth_s=x.cfg["heading"]["smooth_s"])
    side = x.cam.get("view_side", "left")
    if side not in ("left", "right"):
        raise ValueError(f"camera.view_side must be 'left' or 'right', not {side!r}")
    R = M.virtual_rotations(fwd, down, 90.0, 0.0, side=side)
    np.save(x.p("rotations.npy"), R.astype(np.float32))
    pd.DataFrame({"t": t, **diag}).iloc[::6].to_csv(x.p("orientation.csv"), index=False,
                                                    float_format="%.3f")


def st_profiles(x: Ctx):
    from .capture.lens import PinholeView
    from .mosaic import flow_profiles
    if x.done("profiles.npz"):
        return
    pc = x.cfg["profiles"]
    view = PinholeView.from_hfov(pc["width"], pc["height"], pc["hfov_deg"])
    R = np.load(x.p("rotations.npy"))
    fp = flow_profiles(x.video, x.lens, R, view, decode_scale=pc["decode_scale"],
                       device=x.cfg["device"])
    fp.save(x.p("profiles.npz"))


def st_trace(x: Ctx):
    """Per-frame speed, along-street position and horizon-row depth (metric via height_m).
    With a GPS fix, also the camera height implied by GPS speed (scale_check.json).
    `camera.speed: gps` takes the motion from the GPS (Doppler) speed instead of the road
    flow wherever the bike moves (> 1 m/s) with a 3D fix: at dusk the road flow of dark,
    low-texture asphalt reads low (10-20 %, with drop-outs to zero; 2026-10-26 side streets).
    The horizon depth stays visual (a ratio of two flows; their bias cancels)."""
    from .mosaic import FlowProfiles, ground_slope, horizon_depth
    from .capture.telemetry import gps_scale_check
    if x.done("trace.csv"):
        return
    fp = FlowProfiles.load(x.p("profiles.npz"))
    s = ground_slope(fp)
    Z = horizon_depth(fp, s)
    X = np.concatenate([[0.0], np.cumsum(s[1:])])
    h0 = float(x.cam["height_m"])
    chk = gps_scale_check(x.tel, pd.DataFrame({"t": fp.t, "s_h": s, "X_m": X * h0}),
                          x.info.fps, h0)
    chk["usable"] = C.gps_scale_ok(chk)
    x.p("scale_check.json").write_text(json.dumps(chk, indent=2))
    if chk.get("h_gps_m"):
        log(f"  GPS: height implied by speed {chk['h_gps_m']:.3f} m (IQR {chk['h_gps_iqr_m']}, "
            f"{'used' if chk['usable'] and x.cam.get('scale') == 'gps' else 'not used'}; "
            f"nominal {h0} m), GPS path {chk['gps_path_m']:.0f} m vs visual "
            f"{chk['visual_path_m']:.0f} m at nominal")
    if x.cam.get("speed", "visual") == "gps":
        from .capture.telemetry import gps_speed
        v, lag = gps_speed(x.tel, fp.t, vis_speed=s)
        use = np.isfinite(v) & (v > 1.0)
        s = np.where(use, np.nan_to_num(v) / (x.info.fps * x.h), s)
        log(f"  speed from GPS on {use.mean():.0%} of frames (lag {lag:.2f} s)")
        X = np.concatenate([[0.0], np.cumsum(s[1:])])
    h = x.h
    df = pd.DataFrame({"frame": np.arange(len(s)), "t": fp.t, "s_h": s, "X_h": X, "Z_h": Z,
                       "speed_kmh": s * x.info.fps * h * 3.6, "X_m": X * h, "Z_m": Z * h})
    df.to_csv(x.p("trace.csv"), index=False, float_format="%.5g")
    log(f"  distance {X[-1] * h:.0f} m, median speed {np.median(df.speed_kmh[df.speed_kmh > 3]):.1f}"
        f" km/h (scale assumes camera height {h:.3f} m)")


def plane_depth(x: Ctx, name: str) -> tuple[float, str]:
    """Plane depth (h units): from the street layout's active segments for this clip when
    available (traffic in the other configuration's segments otherwise pulls the mode), else
    the mode of the horizon depth over the whole clip."""
    from .layout import Layout
    from .mosaic import parked_depth
    lay = Layout.load(Path(x.cfg["out_dir"]) / "layout")
    D = (lay.plane_depth_h.get(x.clip["id"], {}).get(name) if lay else None)
    if D is not None:
        return float(D), "layout"
    return parked_depth(x.trace.Z_h.values, *C.plane_band_h(x.cfg, name, x.h)), "clip"


def mosaic_params(x: Ctx) -> tuple[float, tuple]:
    """Metric mosaic settings in this clip's camera-height units: (px per h, y range in h)."""
    mc = x.cfg["mosaic"]
    return mc["px_per_m"] * x.h, tuple(v / x.h for v in mc["y_range_m"])


def st_mosaic(x: Ctx):
    from .mosaic import build_mosaic
    mc = x.cfg["mosaic"]
    R = None
    tr = x.trace
    P, yr = mosaic_params(x)
    for name, pl in mc["planes"].items():
        stem = x.p(f"mosaic_{name}")
        D, src = plane_depth(x, name)
        if x.done(f"mosaic_{name}.png") and not _stale(x, f"mosaic_{name}.png", "trace.csv"):
            old = json.loads(x.p(f"mosaic_{name}.json").read_text())
            moved = abs(np.log(D / old["depth"])) >= mc["rebuild_depth_tol"]
            rescaled = (abs(np.log(P / old["px_per_m"])) >= mc["rebuild_scale_tol"]
                        or abs(np.log(yr[1] / old["y_top"])) >= mc["rebuild_scale_tol"])
            if not (moved or rescaled):
                continue
            log(f"  plane {name}: depth {old['depth']:.2f} h -> {D:.2f} h ({src}), "
                f"{old['px_per_m']:.1f} -> {P:.1f} px/h, rebuilding")
        if R is None:
            R = np.load(x.p("rotations.npy"))
        log(f"  plane {name}: depth {D:.2f} h = {D * x.h:.2f} m ({src})")
        m = build_mosaic(x.video, x.lens, R, tr.s_h.values, D, 0.0, float(tr.t.iloc[-1]),
                         px_per_m=P, y_range=yr, decode_scale=pl["decode_scale"],
                         rs_readout_s=x.cam["rs_readout_s"], device=x.cfg["device"])
        m.save(stem)
        # quick-look JPEG (downscaled) for humans
        import cv2
        small = cv2.resize(m.image, (m.image.shape[1] // 4, m.image.shape[0] // 4),
                           interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(x.p(f"mosaic_{name}_quick.jpg")), cv2.cvtColor(small, cv2.COLOR_RGB2BGR))


def _stale(x: Ctx, out: str, *inputs: str) -> bool:
    o = x.p(out)
    return not o.exists() or any(not x.p(i).exists() or x.p(i).stat().st_mtime > o.stat().st_mtime
                                 for i in inputs)


def acceptance(cfg: dict, plane: str, dark: bool = False):
    from .detect import Acceptance
    dc = cfg["detect"]
    min_conf = dc["min_conf_dark"] if dark and dc.get("min_conf_dark") is not None \
        else dc["min_conf"]
    return Acceptance(min_height_m=dc["min_height_m"], max_height_m=dc["max_height_m"],
                      min_length_m=dc["min_length_m"], max_length_m=dc["max_length_m"],
                      min_conf=min_conf, static_ratio=tuple(dc["static_ratio"]),
                      flag_ratio=tuple(dc["flag_ratio"]), clip_px=dc["clip_px"],
                      lateral_m=tuple(cfg["mosaic"]["planes"][plane]["lateral_m"]))


def st_detect(x: Ctx):
    """Detector boxes are cached in raw_<plane>.csv, with the detector that made them in
    raw_<plane>.json (redone when the mosaic is newer or the detector changed); the cheap
    geometric acceptance is redone when det_<plane>.csv is missing/stale or --force."""
    from . import detect as De
    from .mosaic import Mosaic
    dc = x.cfg["detect"]
    model = None
    h = x.h
    want = De.detector_id(dc)
    from .layout import dark_clip
    dark = dark_clip(x.dir, x.cfg["layout"].get("dark_exposure"))
    if dark and dc.get("min_conf_dark") is not None:
        log(f"  dark pass: min_conf {dc['min_conf_dark']}")
    for name in x.cfg["mosaic"]["planes"]:
        mos = f"mosaic_{name}.png"
        raw_f, det_f, id_f = f"raw_{name}.csv", f"det_{name}.csv", f"raw_{name}.json"
        raw_stale = (_stale(x, raw_f, mos) or not x.p(id_f).exists()
                     or json.loads(x.p(id_f).read_text()) != want)
        if not x.force and not raw_stale and not _stale(x, det_f, mos, raw_f):
            continue
        m = Mosaic.load(x.p(f"mosaic_{name}"))
        if raw_stale:
            model = model or De.load_detector(dc, x.cfg["device"])
            raw = De.tile_detect(m.image, model, dc["tile_w"], dc["stride"])
            raw.to_csv(x.p(raw_f), index=False, float_format="%.2f")
            x.p(id_f).write_text(json.dumps(want, indent=2))
        raw = pd.read_csv(x.p(raw_f))
        if dc.get("merge_splits", True):
            raw = De.merge_splits(raw, m.px_per_m / h, dc["min_length_m"])
        if "fp" not in x._cache:
            from .mosaic import FlowProfiles
            x._cache["fp"] = FlowProfiles.load(x.p("profiles.npz"))
        det = De.accept(raw, m, h, acceptance(x.cfg, name, dark), profiles=x._cache["fp"],
                        slope=x.trace.s_h.values)
        det = De.color_features(m.image, det)
        det["t"] = det.frame / x.info.fps
        det.to_csv(x.p(det_f), index=False, float_format="%.3f")
        log(f"  {name}: {len(det)} detections, {int(det.kept.sum())} kept; rejects: "
            f"{det[~det.kept].reject.value_counts().to_dict()}")


def write_chunks(img, out_dir: Path, px_per_m: float, x0_m: float, chunk_m: float = 100.0,
                 rows_per_file: int = 4):
    """Review JPEGs: `rows_per_file` stacked strips of `chunk_m` metres each, with a metre ruler."""
    import cv2
    out_dir.mkdir(exist_ok=True)
    W = int(chunk_m * px_per_m)
    strips = []
    for i, c0 in enumerate(range(0, img.shape[1], W)):
        s = img[:, c0:c0 + W].copy()
        s = np.pad(s, ((14, 0), (0, W - s.shape[1]), (0, 0)))
        for m in range(0, int(chunk_m) + 1, 5):
            xx = int(m * px_per_m)
            cv2.line(s, (xx, 0), (xx, 8 if m % 25 else 13), (255, 255, 255), 1)
            if m % 25 == 0:
                cv2.putText(s, f"{x0_m + c0 / px_per_m + m:.0f} m", (xx + 3, 11),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        strips.append(s)
    for f in range(0, len(strips), rows_per_file):
        cv2.imwrite(str(out_dir / f"{f // rows_per_file:03d}.jpg"),
                    np.vstack(strips[f:f + rows_per_file]), [cv2.IMWRITE_JPEG_QUALITY, 85])


def st_count(x: Ctx):
    """Parked vehicles = kept detections of the plane the street layout says is the parked
    row at that position; kept detections of the other plane are travel-lane / other-side
    objects and go to the review list with the active plane's geometric rejects."""
    import cv2
    from . import detect as De
    from .count import apply_layout, block_counts, merge_planes, occupancy_profile, union_length
    from .layout import Layout, load_dets
    from .mosaic import Mosaic
    planes = list(x.cfg["mosaic"]["planes"])
    h = x.h
    dets = load_dets(x.dir, planes)
    lay = Layout.load(Path(x.cfg["out_dir"]) / "layout")
    if lay is not None and x.clip["id"] not in lay.maps:
        log("  clip not in the street layout (run `curb-occupancy layout`); counting without it")
        lay = None
    if lay is not None:
        dets, review = apply_layout(dets, lay, x.clip["id"], x.cfg["count"]["review_min_conf"],
                                    x.cfg["count"]["travel_lane_m"])
        review.to_csv(x.p("review.csv"), index=False, float_format="%.3f")
    cars = merge_planes(dets, x.cfg["count"]["merge_overlap"])
    if lay is not None and len(cars):
        cars["row"] = [lay.row_of(x.clip["id"], p) for p in cars.plane]
    cars.to_csv(x.p("cars.csv"), index=False, float_format="%.3f")
    xmax = float(x.trace.X_m.iloc[-1])
    occupancy_profile(cars, xmax).to_csv(x.p("occupancy_25m.csv"), index=False,
                                         float_format="%.2f")
    summ = {"clip": x.clip["id"], "distance_m": xmax, "n_parked": int(len(cars)),
            "layout": lay is not None}
    for n in dets:
        c = cars[cars.plane == n] if len(cars) else cars
        summ[f"n_{n}"] = int(len(c))
        summ[f"occupied_m_{n}"] = round(union_length(c.x_start_m, c.x_end_m), 1) if len(c) else 0
    summ["n_flagged"] = int((cars.flag != "").sum()) if len(cars) else 0
    if lay is not None:
        summ["n_review"] = int(len(review))
        summ["review_reasons"] = review.reason.value_counts().to_dict() if len(review) else {}
        block_counts(cars, lay, x.clip["id"]).to_csv(x.p("counts_by_block.csv"), index=False)
    x.p("count.json").write_text(json.dumps(summ, indent=2))
    log("  " + json.dumps(summ))
    for n, d in dets.items():
        m = Mosaic.load(x.p(f"mosaic_{n}"))
        inactive = None
        if lay is not None:
            colx = (m.x0 + (np.arange(m.image.shape[1]) + 0.5) / m.px_per_m) * h
            inactive = ~lay.plane_active(x.clip["id"], n, colx)
        ov = De.overlay(m.image, d, inactive_cols=inactive)
        cv2.imwrite(str(x.p(f"overlay_{n}.png")), ov)  # JPEG is capped at 65535 px wide
        write_chunks(ov, x.p(f"overlay_{n}"), m.px_per_m / h, m.x0 * h)


def run_layout(cfg: dict, force: bool):
    from .layout import fit_layout
    lay = fit_layout(cfg)
    seg = lay.segments()
    log("  layout: " + "; ".join(f"{r.x0_ref_m:.0f}-{r.x1_ref_m:.0f} m rows {r.rows}"
                                 for r in seg.itertuples())
        + f" (ref {lay.reference}); {len(lay.intersections)} intersections")
    for c, d in lay.plane_depth_h.items():
        log(f"  {c} plane depths (h): " + ", ".join(
            f"{p} {v:.2f}" if v is not None else f"{p} n/a" for p, v in d.items()))


def run_multiday(cfg: dict, force: bool):
    from .multiday import multiday
    res = multiday(cfg)
    if res:
        log("  multiday: " + json.dumps(res["pairs"]))


def run_supply(cfg: dict, force: bool):
    from .regs.supply import supply
    res = supply(cfg)
    if res is None:
        log("  supply: no `regulations.sources` in the config (or no layout); skipped")
        return
    log(f"  supply: axis {json.dumps(res['axis'])}, guideline {res['guideline']['profile']}")
    for name, info in res["sources"].items():
        log(f"  source {name}: {json.dumps(info, ensure_ascii=False)}")
    log("  occupancy: " + json.dumps(res["occupancy"]))


def run_fetch(cfg: dict, force: bool):
    """Download the external data the config names (missing files, or all with --force):
    the OpenStreetMap extract around the reference pass's GPS track (`street.osm_json`, needs
    its telemetry stage) and the regulation sources' files."""
    from .regs.supply import source_module
    osm_json = cfg["street"].get("osm_json")
    if osm_json and (force or not Path(osm_json).exists()):
        from . import osm
        ref = cfg["layout"].get("reference") or cfg["clips"][0]["id"]
        f = Path(cfg["out_dir"]) / ref / "telemetry.npz"
        if not f.exists():
            log(f"  fetch: run `curb-occupancy telemetry` first (the OSM extract is cut around {ref}'s GPS)")
        else:
            tel = C.telemetry(cfg, ref)
            ll = tel.gps[tel.gps_fix >= 3, :2]
            if not len(ll):
                log(f"  fetch: {ref} has no GPS fix; no OSM extract")
            else:
                pad = 0.004  # ~300-400 m: cross streets and the street's ways beyond the ends
                bbox = (ll[:, 0].min() - pad, ll[:, 1].min() - pad, ll[:, 0].max() + pad,
                        ll[:, 1].max() + pad)
                d = osm.fetch(tuple(float(v) for v in bbox), osm_json)
                log(f"  fetch: {len(d['elements'])} OSM ways -> {osm_json}")
    for sc in (cfg.get("regulations") or {}).get("sources", []):
        m = source_module(sc.get("type"))
        if hasattr(m, "fetch_source"):
            m.fetch_source(sc, force)


def run_serve(cfg: dict, port: int):
    """Serve <out_dir>/map on localhost: OSM's tile servers refuse pages without an http(s)
    Referer, which a file:// page cannot send."""
    import functools
    import http.server

    from .publish.labeller import labels_file, read_labels, write_labels
    is_area = "streets" in cfg   # area config (`area.load`): one public page, no labeller
    lf = None if is_area else labels_file(cfg)

    class Quiet(http.server.SimpleHTTPRequestHandler):
        """Static files + /api/labels: GET -> the curb labels, POST {zones} -> saved to the
        manual source's `file` (curb.html)."""
        def log_message(self, *a):
            pass

        def end_headers(self):
            # pages are rebuilt in place: without this the browser may show a stale copy
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def _json(self, code: int, obj):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path.split("?")[0] == "/api/labels":
                return self._json(200, {"file": str(lf) if lf else None,
                                        "zones": read_labels(lf)})
            return super().do_GET()

        def do_POST(self):
            if self.path.split("?")[0] != "/api/labels":
                return self._json(404, {"error": "not found"})
            if lf is None:
                return self._json(409, {"error": "no `file` on a manual regulation source"})
            n = int(self.headers.get("Content-Length", 0))
            zones = json.loads(self.rfile.read(n))["zones"]
            write_labels(lf, zones, cfg["name"])
            log(f"  serve: {len(zones)} curb labels -> {lf}")
            return self._json(200, {"file": str(lf), "n": len(zones)})
    d = Path(cfg["out_dir"]) if is_area else Path(cfg["out_dir"]) / "map"
    if not (d / ("public.html" if is_area else "review.html")).exists():
        log(f"  serve: no map yet, run `curb-occupancy {'area' if is_area else 'map'}` first")
        return
    h = functools.partial(Quiet, directory=str(d))
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), h) as srv:
        log(f"  serving {d}: http://127.0.0.1:{port}/public.html  (Ctrl-C to stop)" if is_area else
            f"  serving {d}: http://127.0.0.1:{port}/review.html (private), "
            f"http://127.0.0.1:{port}/public.html, http://127.0.0.1:{port}/curb.html "
            f"(curb labeller -> {lf})  (Ctrl-C to stop)")
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass


def run_map(cfg: dict, force: bool):
    from .publish.webmap import build
    res = build(cfg, force)
    if res is None:
        log("  map: needs the layout's street axis and the supply stage; skipped")
        return
    log(f"  map: {json.dumps(res['layers'])}, {res['checks']} site checks, labeller "
        f"{json.dumps(res['labeller'])}, ride views {sorted(res['ride'].values())} -> "
        f"{res['out']}/review.html, curb.html (private) + public.html")


STAGE_FN = {s: globals()[f"st_{s}"] for s in STAGES}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="curb-occupancy", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["run"] + SETUP + STAGES + RUN_STAGES + TOOLS)
    ap.add_argument("config")
    ap.add_argument("--clip", action="append", help="restrict to clip id(s)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--until", choices=STAGES, help="with run: stop after this stage")
    ap.add_argument("--port", type=int, default=8000, help="with serve")
    ap.add_argument("--dry-run", action="store_true", help="with segment: write nothing")
    a = ap.parse_args(argv)
    from .publish import area
    from . import ride
    if a.stage == "segment" or ride.is_ride(a.config):
        if a.stage != "segment":
            ap.error(f"{a.config} is a ride config: only `curb-occupancy segment` applies")
        r = ride.load(a.config)
        seg = ride.write(r, *ride.plan(r), dry_run=a.dry_run)
        with pd.option_context("display.width", 200, "display.max_colwidth", 60):
            print(seg[["road", "side", "t0", "t1", "length_m", "speed_kmh", "action",
                       "config"]].to_string(index=False, float_format="%.0f"))
        return
    if a.stage == "area" or area.is_area(a.config):
        if a.stage == "serve":
            return run_serve(area.load(a.config), a.port)
        if a.stage != "area":
            ap.error(f"{a.config} is an area config: only `curb-occupancy area` and `curb-occupancy serve` apply")
        res = area.build(area.load(a.config))
        log(f"  area: {json.dumps(res)}")
        return
    cfg = C.load(a.config)
    Path(cfg["out_dir"]).mkdir(parents=True, exist_ok=True)
    clips = [c for c in cfg["clips"] if not a.clip or c["id"] in a.clip]

    def per_clip(stages, force):
        for clip in clips:
            x = Ctx(cfg, clip, force)
            for s in stages:
                t0 = time.time()
                log(f"[{clip['id']}] {s}")
                STAGE_FN[s](x)
                log(f"[{clip['id']}] {s} done in {time.time() - t0:.0f}s")

    if a.stage == "run":
        stages = STAGES[: STAGES.index(a.until) + 1] if a.until else STAGES
        pre = [s for s in stages if s in PRE_LAYOUT]
        per_clip(pre, a.force)
        if a.until and a.until in PRE_LAYOUT:
            return
        run_fetch(cfg, False)  # only what is missing
        log("[run] layout")
        run_layout(cfg, a.force)
        # second pass: only planes whose layout-based depth moved are rebuilt/re-detected
        def stamp():
            return {p.name: p.stat().st_mtime
                    for c in cfg["clips"]
                    for p in (Path(cfg["out_dir"]) / c["id"]).glob("det_*.csv")}
        before = stamp()
        per_clip(["mosaic", "detect"], False)
        if stamp() != before:
            log("[run] layout (refit)")
            run_layout(cfg, False)
        per_clip([s for s in stages if s not in PRE_LAYOUT], a.force)
        if not a.until:
            run_multiday(cfg, True)
            run_supply(cfg, True)
            run_map(cfg, a.force)
    elif a.stage in STAGES:
        per_clip([a.stage], a.force)
    elif a.stage == "layout":
        run_layout(cfg, a.force)
    elif a.stage == "multiday":
        run_multiday(cfg, True)
    elif a.stage == "supply":
        run_supply(cfg, True)
    elif a.stage == "serve":
        run_serve(cfg, a.port)
    elif a.stage == "map":
        run_map(cfg, a.force)
    else:
        run_fetch(cfg, a.force)


if __name__ == "__main__":
    main()
