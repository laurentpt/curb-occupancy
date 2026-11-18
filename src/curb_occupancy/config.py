"""Per-street YAML config with defaults. Values are plain dicts; see the study configs (ndg-terrebonne-parking-2026: config/streets/*.yaml)."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULTS = {
    "name": None,
    "data_dir": "workdir/input_data",
    "out_dir": None,  # default: workdir/runs/<name>
    "device": "cuda",
    "camera": {
        # rig profile (package `rigs/<name>/rig.yaml`): lens_profile, rs_readout_s, imu_to_cv,
        # merged under this section. Without a rig, give those three here.
        "rig": None,
        "height_m": None,          # lens above road (nominal/measured, required); see `scale`
        # 'gps': effective height = GPS speed / visual speed (trace stage, scale_check.json)
        # when the clip has a good fix, else height_m. 'nominal': always height_m.
        "scale": "gps",
        # motion source of the trace: 'visual' (road flow) or 'gps' (Doppler speed while
        # moving with a 3D fix; for dusk/night, where the road flow reads low)
        "speed": "visual",
        # side of the direction of travel the virtual view looks at. 'right' renders a
        # mirrored view so every downstream stage sees the same geometry as 'left'.
        "view_side": "left",
        # mount yaw: optical axis this many degrees from forward, toward `view_side` (only to
        # pick the frames that show a sign pole on the review map; None: 90)
        "mount_yaw_deg": None,
        # IANA time zone of the captures (e.g. America/Montreal). When set, a clip's start time
        # (video.json creation_time: passes in the guideline window, time-of-day rules) comes
        # from the GPS UTC clock in this zone; else, or without a 3D fix, from the camera clock
        # (GoPro: local time as set, no DST change, drifts)
        "timezone": None,
    },
    # Street cross-section. Parking rows, each governed by the signs of one curb (right/left
    # of the street-axis direction = the reference pass's riding direction) and imaged by one
    # mosaic plane when riding forward (same way as the reference pass) or backward. `states`:
    # the sets of rows that can be present at a point of the street; the layout segments the
    # street into them. Required (no default): see the study configs for examples (two
    # rows on alternating curbs; one row seen across a travel lane).
    "street": {
        "name": None,           # OSM name of the street (None: the road the GPS track follows)
        "axis": "auto",         # 'osm' | 'track' | 'auto' (osm when osm_json exists); see street.py
        "osm_json": None,       # OpenStreetMap extract (`curb-occupancy fetch` downloads it)
        "osm_snap_m": 15.0,     # a mapped cross street takes a detected deep stretch this close
        "cross_street_m": 10.0,  # roadway width of a mapped cross street nobody saw (T-junction)
        "rows": {},             # {<row>: {curb: right|left, label, plane: {forward, backward}}}
        "states": [],           # e.g. [[A], [B]] (one row at a time) or [[A, B]]
        # site knowledge overriding the fused layout on a stretch:
        # [{between: [<cross street>, <cross street>] | x_m: [a, b], rows: [A, B], note: ..}]
        "segments": [],
        # cross-street roadway edges checked on the imagery, overriding the detected extent:
        # [{name: <cross street as in layout/intersections.csv>, x0_m: a, x1_m: b, note: ..}]
        # (either edge may be omitted)
        "intersections": [],
        # reporting stretches frozen by cross streets ([from, to] per stretch, along the axis;
        # null = the street's open end); empty: grouped to ~guideline.stretch_spaces per run
        "stretches": [],
    },
    "heading": {"step": 3, "every": 6, "smooth_s": 2.0},
    "profiles": {"width": 480, "height": 432, "hfov_deg": 90, "decode_scale": 0.5},
    # Metric; converted to each clip's camera-height units (mosaics are built in h units).
    # Values = the 2026-10-22 h-unit settings (100 px/h, [-0.9, 2.4] h, ...) at h = 1.18 m,
    # so cars have the same pixel size in every clip whatever the mount height.
    "mosaic": {
        "px_per_m": 85,              # mosaic scale at the plane, px per metre
        "y_range_m": [-1.06, 2.83],  # heights above road covered by the rows
        # name -> {depth_m: search band of the plane depth, decode_scale, lateral_m: accepted
        # ground-contact band}, one per parked row distance the camera sees (street-specific)
        "planes": {},
        "rebuild_depth_tol": 0.15,  # |log| change of the (layout) plane depth that triggers
        "rebuild_scale_tol": 0.02,  # a mosaic rebuild; same for px/h and the top row height
    },
    "detect": {
        # 'hf': a transformers COCO detector (hub id or directory); 'yolo': ultralytics .pt
        "backend": "hf", "model": "ustc-community/dfine-xlarge-obj2coco",
        "tile_w": 1280, "stride": 640, "conf": 0.15,
        "min_height_m": 1.15, "max_height_m": 3.2, "min_length_m": 2.4, "max_length_m": 9.0,
        # score scales differ by detector: D-FINE-X letterboxed is flat over 0.3-0.5 on the
        # study mosaics (0.35); the YOLO11-L interim results used 0.25
        "min_conf": 0.35, "clip_px": 12,
        # dark passes (layout.dark_exposure): night boxes under ~0.5 were fragments of parked
        # cars and non-vehicles (hand count 2026-11-17); None: min_conf everywhere
        "min_conf_dark": 0.5,
        "merge_splits": True,         # re-join cars split by a thin occluder (detect.merge_splits)
        "static_ratio": [0.75, 2.0],  # flow depth / contact depth outside -> moving
        "flag_ratio": [0.85, 1.6],
    },
    "layout": {
        "reference": None,         # clip id whose along-street axis is the street axis
        "bin_m": 1.0,
        "switch_cost_m": 30.0,     # evidence (m of confident boxes) needed to change state
        # per bin and present row, so states with more rows must be earned. Car evidence cannot
        # tell an empty row from an absent one: a row is dropped only after switch_cost /
        # row_cost metres without cars (600 m here). Irrelevant for exclusive states;
        # for a street whose rows always exist, use street.states [[A, B]].
        "row_cost": 0.05,
        "min_depth_samples": 200,  # horizon samples to re-estimate a plane depth per clip
        # cross streets are voted by lit passes only: a clip whose median ISO x exposure time
        # (s) exceeds this is dark (night); None: every pass votes
        "dark_exposure": 30.0,
        # a pass aligned to the reference by occupancy must stay within this of its own GPS
        # on the mapped (OSM) street, else its GPS gives the global map. Evening GPS on
        # Terrebonne (DOP ~6) is 8-12 m off correct occupancy alignments (the same-evening
        # return pass, 94 cars matched); wrong side-street alignments were 26-52 m off
        "gps_tolerance_m": 20.0,
        # then each pass's map is refitted on the background of its strips, registered onto
        # the passes riding the same way (`register.py`; car independent). Passes riding the
        # other way form their own group, aligned by the pooled occupancy of its passes
        "register": {
            "enabled": True,
            "planes": None,          # mosaic planes to register (None: all imaging a row)
            "y_m": [-1.0, 2.6],      # strip heights (m above the road at the plane)
            "band_m": [0.6, 2.3],    # template heights (background and curb above the road)
            "win_m": 30.0, "step_m": 5.0,
            "search_m": 20.0,        # around the occupancy / GPS map
            "min_peak": 0.2,         # NCC of a confident window
            "min_ok": 10,            # confident windows to accept a registration
            "pooled_max_shift_m": 15.0,
        },
    },
    "count": {"merge_overlap": 0.5, "review_min_conf": 0.4,
              # depth band (m) of the travel lane in each plane's mosaic, only to label
              # gated-out boxes ('travel_lane' vs 'other_side') in the review list
              "travel_lane_m": {}},  # {<plane>: [lo, hi]}
    # list of {id, file, direction (label or 'auto'), camera (overrides), t_range ([t0, t1] s:
    # the part on the street, e.g. without a turn at the end), note}. `file`: one file or the
    # list of chapter files of one recording (GX01nnnn.MP4, GX02nnnn.MP4, ...); with
    # `t_range`, the clip is that window only (see `source`). id default: the first file's stem.
    "clips": [],
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if v is None and isinstance(out.get(k), dict):  # `name: null` drops a default entry
            del out[k]                                    # (e.g. a mosaic plane)
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _raw(path: Path) -> dict:
    """YAML of a config; `extends: <path>` (relative to this file) merges it over a shared
    base, e.g. the settings common to a set of side streets."""
    raw = yaml.safe_load(path.read_text()) or {}
    base = raw.pop("extends", None)
    return _merge(_raw((path.parent / base).resolve()), raw) if base else raw


RIGS = Path(__file__).parent / "rigs"
RIG_KEYS = ("lens_profile", "rs_readout_s", "imu_to_cv")


def rig(name: str) -> dict:
    """Rig profile `rigs/<name>/rig.yaml`; its lens_profile path made absolute."""
    f = RIGS / name / "rig.yaml"
    if not f.exists():
        known = sorted(d.name for d in RIGS.iterdir() if (d / "rig.yaml").exists())
        raise ValueError(f"camera.rig {name!r}: unknown rig (known: {known})")
    r = yaml.safe_load(f.read_text())
    cam = r.setdefault("camera", {})
    if cam.get("lens_profile"):
        cam["lens_profile"] = str((f.parent / cam["lens_profile"]).resolve())
    return r


def _camera(raw_cam: dict | None) -> dict:
    """Camera section: defaults < rig profile < the config's own values."""
    raw_cam = raw_cam or {}
    name = raw_cam.get("rig")
    base = _merge(DEFAULTS["camera"], rig(name)["camera"]) if name else DEFAULTS["camera"]
    cam = _merge(base, raw_cam)
    missing = [k for k in RIG_KEYS + ("height_m",) if cam.get(k) is None]
    if missing:
        raise ValueError(f"camera: {missing} not set (set camera.rig or give them)")
    return cam


def load(path: str | Path) -> dict:
    path = Path(path)
    raw = _raw(path)
    if isinstance(raw.get("street"), str):  # old configs: `street: <name>`
        raw["street"] = {"name": raw["street"]}
    cfg = _merge(DEFAULTS, raw)
    cfg["camera"] = _camera(raw.get("camera"))
    cfg["name"] = cfg["name"] or path.stem
    cfg["out_dir"] = cfg["out_dir"] or f"workdir/runs/{cfg['name']}"
    cfg["_path"] = str(path)
    for c in cfg["clips"]:
        c.setdefault("id", Path(files(c)[0]).stem)
        c["camera"] = _merge(cfg["camera"], c.get("camera") or {})
    from .street import validate
    validate(cfg)
    return cfg


def height(cfg: dict, clip_id: str) -> float:
    """Effective camera height (m) of one clip, i.e. the metric scale of its visual odometry:
    GPS-calibrated when `camera.scale` is 'gps' and the trace stage found a usable fix,
    else the (per-clip or street) nominal `camera.height_m`."""
    import json
    cam = clip(cfg, clip_id)["camera"]
    if cam.get("scale", "gps") == "gps":
        f = Path(cfg["out_dir"]) / clip_id / "scale_check.json"
        if f.exists():
            chk = json.loads(f.read_text())
            if gps_scale_ok(chk):
                return float(chk["h_gps_m"])
    return float(cam["height_m"])


def gps_scale_ok(chk: dict, min_samples: int = 600, max_rel_iqr: float = 0.15) -> bool:
    """GPS height usable: enough moving 3D-fix samples (~1 min at 10 Hz) and a tight ratio."""
    if not chk.get("h_gps_m") or chk.get("n_samples", 0) < min_samples:
        return False
    lo, hi = chk["h_gps_iqr_m"]
    return (hi - lo) / chk["h_gps_m"] <= max_rel_iqr


def plane_band_h(cfg: dict, plane: str, h: float) -> tuple[float, float]:
    lo, hi = cfg["mosaic"]["planes"][plane]["depth_m"]
    return lo / h, hi / h


def files(clip: dict) -> list[str]:
    """Chapter files of a clip (`file`: one name or a list)."""
    f = clip["file"]
    return [f] if isinstance(f, str) else list(f)


def t_range(cfg: dict, clip_id: str) -> tuple[float, float | None]:
    """Part of the recording on the street (s): clip `t_range: [t0, t1]` (t1 null = to the
    end). Default: the whole recording."""
    tr = clip(cfg, clip_id).get("t_range") or [0.0, None]
    return float(tr[0] or 0.0), None if tr[1] is None else float(tr[1])


def source(cfg: dict, clip_id: str):
    """The clip as a `video.Source`: its chapters, cut to `t_range`. Every per-clip stage
    decodes only this window, and clip times / frame indices start at its beginning (a long
    recording with several streets costs ~1x in total over the streets' runs)."""
    from .capture.video import Source
    return Source.open([Path(cfg["data_dir"]) / f for f in files(clip(cfg, clip_id))],
                       t_range(cfg, clip_id))


def telemetry(cfg: dict, clip_id: str):
    """Telemetry of a clip (already cut to its window by the telemetry stage)."""
    from .capture.telemetry import Telemetry
    return Telemetry.load(Path(cfg["out_dir"]) / clip_id / "telemetry.npz")


def clip(cfg: dict, clip_id: str) -> dict:
    return next(c for c in cfg["clips"] if c["id"] == clip_id)
