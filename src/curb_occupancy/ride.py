"""Long recordings: one ride over several streets -> one clip window per street.

    curb-occupancy segment <study>/rides/<ride>.yaml [--dry-run]

The ride's GPS track (every chapter of the recording) is matched to the named car roads of an
OpenStreetMap extract: a fix takes the nearest road within `max_dist_m` (30 m: evening GPS
runs 10-25 m beside the road, avenues are ~90 m apart) whose direction agrees
with the riding direction (within `max_angle_deg`; crossing a street at an intersection does not
match it), stopped fixes keep the previous road, and short interruptions between two runs on the
same road are absorbed. Every run on one road of at least `min_length_m` is a street pass; its
window (video time = GPS time - `gps_lag_s`, +- `pad_s`) becomes a clip `t_range`.

Each pass is added as a clip to the street config of its street and side (the curb imaged:
the base config's single row, on the camera's side of the riding direction), found in
`streets_dir` by `street.name` and `street.rows.<row>.label`, or written there as a new config
that `extends` the base. New configs are appended to the `area` config. Existing files are only
appended to (a clip already listed is left alone), so the configs stay plain, commented YAML.
Every street run then decodes only its own window (`video.Source`).

    # config/rides/2026-10-29_GX014499.yaml (study repository)
    video: GX014499.MP4          # first chapter; GX02nnnn, GX03nnnn, ... are found next to it
    osm_json: inputs/osm/terrebonne_full_2026-10-23.json
    streets_dir: ../streets/side # relative to this file
    base: _base.yaml             # relative to streets_dir; `extends` of new street configs
    area: _area.yaml             # relative to streets_dir; optional
    prefix: side_                # run name of new street configs: <prefix><slug>
    skip: [Rue de Terrebonne]    # streets not to write
    grid_north_deg: -58          # bearing of the city grid's "north" (side names)

Writes workdir/rides/<name>/segments.csv (every run, with what was done with it).
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import config as C

DEFAULTS = {"data_dir": "workdir/input_data", "max_dist_m": 30.0, "max_angle_deg": 35.0,
            "min_speed_ms": 1.5, "absorb_s": 4.0, "min_length_m": 60.0, "gps_lag_s": 1.08,
            "pad_s": 1.0, "grid_north_deg": 0.0, "prefix": "", "skip": [], "row": None,
            "area": None}
SIDES = ["north", "east", "south", "west"]


def load(path: str | Path) -> dict:
    path = Path(path)
    r = {**DEFAULTS, **(yaml.safe_load(path.read_text()) or {})}
    r["name"] = r.get("name") or path.stem
    r["streets_dir"] = (path.parent / r["streets_dir"]).resolve()
    r["_path"] = str(path)
    return r


def is_ride(path: str | Path) -> bool:
    d = yaml.safe_load(Path(path).read_text()) or {}
    return "video" in d and "clips" not in d


def chapters(data_dir: str | Path, video) -> list[str]:
    """Chapter file names of a recording: a list as given, or the GoPro chapters of a first
    chapter G?01nnnn.MP4 that exist in data_dir (G?02nnnn, G?03nnnn, ...)."""
    if not isinstance(video, str):
        return list(video)
    m = re.fullmatch(r"(G[HX])01(\d{4})(\.MP4)", video, re.I)
    out = [video]
    while m:
        nxt = f"{m.group(1)}{len(out) + 1:02d}{m.group(2)}{m.group(3)}"
        if not (Path(data_dir) / nxt).exists():
            break
        out.append(nxt)
    return out


# ------------------------------------------------------------------------------ matching

def _roads(osm: dict, lat0: float, lon0: float):
    from .osm import CAR_ROADS, _ways
    from .street import local_xy
    A, B, names = [], [], []
    for w in _ways(osm):
        t = w.get("tags", {})
        if t.get("highway") not in CAR_ROADS or not t.get("name"):
            continue
        g = local_xy([p["lat"] for p in w["geometry"]], [p["lon"] for p in w["geometry"]],
                     lat0, lon0)
        A.append(g[:-1])
        B.append(g[1:])
        names += [t["name"]] * (len(g) - 1)
    return np.concatenate(A), np.concatenate(B), np.array(names, dtype=object)


def match(tel, osm: dict, r: dict, rate_hz: float = 5.0) -> pd.DataFrame:
    """GPS fixes (3D, decimated to ~rate_hz) -> road name per fix ('' = none). Columns t (GPS
    time in the recording), x, y (local m), speed, road."""
    from scipy.ndimage import uniform_filter1d

    from .street import local_xy
    g = tel.gps_fix >= 3
    t, ll, v = tel.gps_t[g], tel.gps[g, :2], tel.gps[g, 3]
    if len(t) < 10:
        raise ValueError("no GPS fix in the recording")
    hz = len(t) / max(t[-1] - t[0], 1e-6)
    lat0, lon0 = float(ll[:, 0].mean()), float(ll[:, 1].mean())
    xy = uniform_filter1d(local_xy(ll[:, 0], ll[:, 1], lat0, lon0),
                          max(1, int(round(hz))), axis=0, mode="nearest")
    k = max(1, int(round(hz)))  # heading over +-1 s
    hd = np.roll(xy, -k, 0) - np.roll(xy, k, 0)
    hd[:k], hd[-k:] = hd[k], hd[-k - 1]
    idx = np.arange(0, len(t), max(1, int(round(hz / rate_hz))))
    t, xy, v, hd = t[idx], xy[idx], v[idx], hd[idx]
    A, B, names = _roads(osm, lat0, lon0)
    d = B - A
    L2 = np.maximum((d * d).sum(1), 1e-9)
    du = d / np.sqrt(L2)[:, None]
    cos_max = np.cos(np.radians(r["max_angle_deg"]))
    road = np.full(len(t), "", dtype=object)
    moving = v >= r["min_speed_ms"]
    for i in range(len(t)):
        s = np.clip(((xy[i] - A) * d).sum(1) / L2, 0, 1)
        dist = np.linalg.norm(A + s[:, None] * d - xy[i], axis=1)
        ok = dist < r["max_dist_m"]
        if moving[i]:
            h = hd[i] / max(np.linalg.norm(hd[i]), 1e-9)
            ok &= np.abs(du @ h) >= cos_max
        if ok.any():
            road[i] = names[np.flatnonzero(ok)[dist[ok].argmin()]]
    # stopped (stop signs, lights): the road it was riding on
    for i in np.flatnonzero(~moving):
        if i > 0:
            road[i] = road[i - 1]
    return pd.DataFrame({"t": t, "x": xy[:, 0], "y": xy[:, 1], "speed": v, "road": road})


def runs(m: pd.DataFrame, absorb_s: float) -> pd.DataFrame:
    """Run-length segments of `road`; a run shorter than absorb_s between two runs of the
    same road is absorbed into them (repeat until stable)."""
    lab = m.road.to_numpy(object).copy()
    while True:
        b = np.flatnonzero(np.r_[True, lab[1:] != lab[:-1]])
        e = np.r_[b[1:], len(lab)] - 1
        changed = False
        for j in range(1, len(b) - 1):
            if (lab[b[j - 1]] == lab[b[j + 1]] and lab[b[j]] != lab[b[j - 1]]
                    and m.t.iloc[e[j]] - m.t.iloc[b[j]] < absorb_s):
                lab[b[j]:e[j] + 1] = lab[b[j - 1]]
                changed = True
                break
        if not changed:
            break
    out = []
    for i0, i1 in zip(b, e):
        s = m.iloc[i0:i1 + 1]
        dx, dy = s.x.iloc[-1] - s.x.iloc[0], s.y.iloc[-1] - s.y.iloc[0]
        out.append({"road": lab[i0], "gps_t0": s.t.iloc[0], "gps_t1": s.t.iloc[-1],
                    "length_m": float(np.hypot(np.diff(s.x), np.diff(s.y)).sum()),
                    "bearing_deg": float(np.degrees(np.arctan2(dx, dy)) % 360),
                    "speed_kmh": float(s.speed.median() * 3.6)})
    return pd.DataFrame(out)


def side_label(bearing: float, view_side: str, grid_north: float) -> str:
    """Name of the curb the camera images when riding along `bearing` (true, deg): the
    city-grid compass side (grid 'north' at bearing `grid_north`)."""
    b = bearing + (-90 if view_side == "left" else 90) - grid_north
    return f"{SIDES[int(np.round((b % 360) / 90)) % 4]} side"


# ------------------------------------------------------------------------------ configs

def slug(name: str) -> str:
    """'Avenue de Hampton' -> 'hampton' (config file stem)."""
    from .publish.area import _short
    s = unicodedata.normalize("NFKD", _short(name)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _flow(v) -> str:
    s = yaml.safe_dump(v, default_flow_style=True, width=1000, allow_unicode=True).strip()
    return s[:-3].strip() if s.endswith("\n...") else s  # scalars end with a document marker


def _append_item(path: Path, key: str, item: str, expect) -> None:
    """Append a block-list item (YAML text, indented) to top-level list `key`, which must be
    the file's last top-level key; check that the list gained exactly `expect`."""
    text = path.read_text()
    before = yaml.safe_load(text)[key] or []
    lines = text.rstrip("\n").split("\n")
    at = [i for i, ln in enumerate(lines) if re.match(rf"{re.escape(key)}:\s*(#.*)?$", ln)]
    later = [ln for ln in lines[(at or [0])[-1] + 1:] if re.match(r"[A-Za-z_]", ln)]
    if not at or later:
        raise ValueError(f"{path}: `{key}:` must be the last top-level key, as a block list")
    new = text.rstrip("\n") + "\n" + item.rstrip("\n") + "\n"
    after = yaml.safe_load(new)[key]
    if after != before + [expect]:
        raise ValueError(f"{path}: appending to `{key}` did not give the expected list")
    path.write_text(new)


def _clip_yaml(c: dict) -> str:
    out = [f"  - file: {_flow(c['file'])}"]
    for k in ("id", "direction", "t_range", "note"):
        if k in c:
            out.append(f"    {k}: {_flow(c[k])}")
    return "\n".join(out)


def _street_configs(d: Path, row: str) -> list[tuple[Path, str, str]]:
    """(path, street name, side label) of the street configs in d."""
    out = []
    for p in sorted(d.glob("*.yaml")):
        if p.name.startswith("_"):
            continue
        raw = C._raw(p)
        st = raw.get("street") or {}
        lab = ((st.get("rows") or {}).get(row) or {}).get("label")
        if st.get("name"):
            out.append((p, st["name"], lab))
    return out


def config_path(d: Path, existing: list, planned: dict, road: str, side: str) -> Path:
    """Street config of a pass on `road` imaging `side`: an existing one (by street name and
    row label), one already planned in this ride, or a new `<slug>.yaml` (`<slug>_<side>`
    when the slug is taken by the street's other side)."""
    hit = [p for p, n, lab in existing if n.casefold() == road.casefold() and lab == side]
    hit += [p for p, items in planned.items()
            if any(rd.casefold() == road.casefold() and sd == side for _, _, rd, sd in items)]
    if hit:
        return hit[0]
    stem = slug(road)
    if any(p.stem == stem for p, _, _ in existing) or d / f"{stem}.yaml" in planned:
        stem += "_" + side.split()[0]
    return d / f"{stem}.yaml"


def plan(r: dict) -> tuple[pd.DataFrame, dict]:
    """Segments of the ride, each with the street config it goes to and the clip entry."""
    import json
    from datetime import datetime, timedelta

    from .capture.telemetry import Telemetry, extract, gps_utc0, local_start
    from .capture.video import Source
    files = chapters(r["data_dir"], r["video"])
    src = Source.open([Path(r["data_dir"]) / f for f in files])
    tel = Telemetry.concat([extract(f) for f in src.files], [o / src.fps for o in src.offsets])
    osm = json.loads(Path(r["osm_json"]).read_text())
    bb = osm.get("query_bbox")
    ll = tel.gps[tel.gps_fix >= 3, :2]
    if bb and len(ll) and not (bb[0] <= ll[:, 0].min() and ll[:, 0].max() <= bb[2]
                               and bb[1] <= ll[:, 1].min() and ll[:, 1].max() <= bb[3]):
        print(f"warning: the ride leaves the OSM extract {r['osm_json']} (bbox {bb})")
    m = match(tel, osm, r)
    seg = runs(m, r["absorb_s"])
    base = C._raw(r["streets_dir"] / r["base"])
    rows = [k for k, v in (base.get("street") or {}).get("rows", {}).items() if v]
    row = r["row"] = r["row"] or (rows[0] if len(rows) == 1 else None)
    if row is None:
        raise ValueError(f"base config has rows {rows}: set `row:` in the ride config")
    view = (base.get("camera") or {}).get("view_side", "left")
    info = src.info
    dur = info.n_frames / src.fps
    # pass times in the notes: GPS clock in the study's zone (camera.timezone), as video.json
    tz = (base.get("camera") or {}).get("timezone")
    u0 = gps_utc0(src.files) if tz else None
    existing = _street_configs(r["streets_dir"], row)
    seg["action"], seg["config"], seg["side"] = "", "", ""
    seg["t0"] = (seg.gps_t0 - r["gps_lag_s"] - r["pad_s"]).clip(0, dur).round(1)
    seg["t1"] = (seg.gps_t1 - r["gps_lag_s"] + r["pad_s"]).clip(0, dur).round(1)
    clips = {}
    for i, s in seg.iterrows():
        if not s.road:
            seg.loc[i, "action"] = "off road"
            continue
        side = side_label(s.bearing_deg, view, r["grid_north_deg"])
        seg.loc[i, "side"] = side
        if s.road in r["skip"]:
            seg.loc[i, "action"] = "skipped"
            continue
        if s.length_m < r["min_length_m"]:
            seg.loc[i, "action"] = "short"
            continue
        path = config_path(r["streets_dir"], existing, clips, s.road, side)
        prev = next((x for x in seg.road[:i][::-1] if x and x != s.road), "?")
        nxt = next((x for x in seg.road[i + 1:] if x and x != s.road), "?")
        clock = ""
        if u0 is not None:
            clock = local_start(u0, float(s.t0), tz)[:16].replace("T", " ")
        elif info.creation_time:
            clock = (datetime.fromisoformat(info.creation_time.rstrip("Z"))
                     + timedelta(seconds=float(s.t0))).strftime("%Y-%m-%d %H:%M")
        compass = SIDES[int(np.round(((s.bearing_deg - r["grid_north_deg"]) % 360) / 90)) % 4]
        clip = {"file": files[0] if len(files) == 1 else files, "direction": "auto",
                "t_range": [float(s.t0), float(s.t1)],
                "note": f"{clock}, ride {r['name']}: {prev} -> {nxt} (riding {compass}, "
                        f"{s.length_m:.0f} m by GPS)"}
        clips.setdefault(path, []).append((i, clip, s.road, side))
        seg.loc[i, "config"] = str(path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd())
                                   else path)
    return seg, clips


def write(r: dict, seg: pd.DataFrame, clips: dict, dry_run: bool = False) -> pd.DataFrame:
    """Apply the plan: new street configs, clips appended to existing ones, area list."""
    from datetime import date
    if r["area"] and not (r["streets_dir"] / r["area"]).exists():
        raise FileNotFoundError(f"area config {r['streets_dir'] / r['area']} does not exist")
    new_files = []
    for path, items in clips.items():
        stem0 = Path(C.files(items[0][1])[0]).stem
        have = (yaml.safe_load(path.read_text()) or {}).get("clips") or [] if path.exists() else []
        ids = {c.get("id") or Path(C.files(c)[0]).stem for c in have}
        same = {(tuple(C.files(c)), tuple(c.get("t_range") or ())) for c in have}
        todo = []
        for i, clip, road, side in items:
            key = (tuple(C.files(clip)), tuple(clip["t_range"]))
            if key in same:
                seg.loc[i, "action"] = "already listed"
                continue
            cid = stem0 if stem0 not in ids else f"{stem0}_{int(clip['t_range'][0]):04d}"
            if cid != stem0:
                clip = {"file": clip["file"], "id": cid, **{k: v for k, v in clip.items()
                                                           if k != "file"}}
            ids.add(cid)
            todo.append((i, clip, road, side))
        if not todo:
            continue
        if not path.exists():
            i, clip, road, side = todo[0]
            ref = clip.get("id") or stem0
            text = (f"# {road}, {side}: written by `curb-occupancy segment {r['_path']}` ({date.today()})\n"
                    f"extends: {r['base']}\nname: {r['prefix']}{path.stem}\nstreet:\n"
                    f"  name: {_flow(road)}\n  rows:\n    {r['row']}: "
                    f"{{label: {_flow(side)}}}\nlayout:\n  reference: {ref}\nclips:\n")
            text += "\n".join(_clip_yaml(c) for _, c, _, _ in todo) + "\n"
            for i, *_ in todo:
                seg.loc[i, "action"] = "new config"
            if not dry_run:
                path.write_text(text)
            new_files.append(path)
            continue
        for i, clip, _, _ in todo:
            seg.loc[i, "action"] = "clip added"
            if not dry_run:
                _append_item(path, "clips", _clip_yaml(clip), clip)
    if r["area"] and new_files:
        ap = r["streets_dir"] / r["area"]
        listed = (yaml.safe_load(ap.read_text()) or {}).get("streets") or []
        for p in new_files:
            if p.name not in listed and not dry_run:
                _append_item(ap, "streets", f"  - {p.name}", p.name)
    if not dry_run:
        out = Path("workdir/rides") / r["name"]
        out.mkdir(parents=True, exist_ok=True)
        seg.to_csv(out / "segments.csv", index=False, float_format="%.1f")
    return seg

