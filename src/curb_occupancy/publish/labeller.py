"""Curb labeller: every pass's mosaic of a parking row resampled onto the street axis and
stacked (near views: curb-cut markings on the ground in front of the row; far views from the
opposite curb: the curb cuts, driveways and garages behind the row), so curb features that no
data source has (driveways) can be drawn as street-axis intervals in a browser.

`curb-occupancy map` writes `<out_dir>/map/curb.html` + `map/strips/` (private: vehicle imagery);
`curb-occupancy serve <config>` serves it and saves the labels to the `file` of the first `manual`
regulation source (YAML, positions as lat/lon at the curb so they survive axis changes and
are shared by every run of the street):

    regulations:
      sources:
        - type: manual
          name: curb_labels
          file: config/streets/terrebonne_curb.yaml
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..register import resample_strip

DEFAULTS = {"px_per_m": 60, "chunk_m": 100.0, "y_m": [-1.0, 2.6], "jpeg_q": 82,
            "gap_m": [2.5, 12.0]}


def labels_file(cfg: dict) -> Path | None:
    """The labels YAML: `file` of the first manual regulation source."""
    for sc in (cfg.get("regulations") or {}).get("sources") or []:
        if sc.get("type") == "manual" and sc.get("file"):
            return Path(sc["file"])
    return None


def read_labels(path: Path | None) -> list[dict]:
    import yaml
    if path is None or not path.exists():
        return []
    d = yaml.safe_load(path.read_text()) or {}
    return list(d.get("zones") or [])


def write_labels(path: Path, zones: list[dict], run: str = ""):
    """One zone per line (JSON flow mappings are valid YAML), sorted along the street."""
    from datetime import datetime
    keep = ("kind", "curb", "latlon", "axis_m", "note")
    zs = sorted(({k: z[k] for k in keep if z.get(k) not in (None, "")} for z in zones),
                key=lambda z: (z.get("axis_m") or [0])[0])
    lines = ["# Curb labels drawn with the curb labeller (`curb-occupancy serve <config>` -> curb.html).",
             "# latlon: zone ends at the curb (authoritative); axis_m: street-axis metres of "
             f"run {run!r} (information only).",
             f"# Saved {datetime.now().isoformat(timespec='seconds')}.", "zones:"]
    lines += [f"  - {json.dumps(z, ensure_ascii=False)}" for z in zs]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def strips(cfg: dict, lay, root: Path, clip: str, row: str, x0: float, x1: float,
           out_dir: Path, lc: dict, force: bool = False) -> dict | None:
    """JPEG chunks of the mosaic imaging `row` in pass `clip`, on the street axis at
    lc.px_per_m over heights lc.y_m (metres above the road at the plane)."""
    from ..config import height
    from ..mosaic import Mosaic
    from .webmap import ClipPos, _save_jpg
    plane = next((p for p in cfg["mosaic"]["planes"] if lay.row_of(clip, p) == row), None)
    stem = root / clip / f"mosaic_{plane}"
    if plane is None or not stem.with_suffix(".png").exists():
        return None
    S, cm = float(lc["px_per_m"]), float(lc["chunk_m"])
    h = height(cfg, clip)
    pos = ClipPos(root, lay, clip)
    m = None
    ylo, yhi = lc["y_m"]
    chunks = []
    for a in np.arange(x0, x1, cm):
        f = out_dir / f"{row}_{clip}_{plane}_{a:.0f}_{S:.0f}.jpg"
        chunks.append({"src": f"strips/{f.name}", "x0": round(float(a), 2)})
        if f.exists() and not force:
            continue
        if m is None:
            m = Mosaic.load(stem)
            ppm = m.px_per_m / h
            Y = yhi - (np.arange(int(round((yhi - ylo) * S))) + 0.5) / S
            rows_src = (m.y_top * h - Y) * ppm - 0.5
        xs = a + (np.arange(int(round(cm * S))) + 0.5) / S
        valid = (xs >= pos.xs[0]) & (xs <= pos.xs[-1])
        img = resample_strip(m.image, lambda x: (pos.clip_x(x) / h - m.x0) * m.px_per_m - 0.5,
                             xs, rows_src, valid)
        _save_jpg(img, f, int(lc["jpeg_q"]))
    return {"clip": clip, "plane": plane, "chunks": chunks}


def gap_candidates(cars: pd.DataFrame, present: np.ndarray, parkable: np.ndarray,
                   grid: np.ndarray, gap_m) -> list[list[float]]:
    """Curb never occupied on any pass, between two parked cars, gap_m[0]..gap_m[1] long and
    parkable: the classic driveway signature (long empty stretches are just low demand)."""
    occ = np.zeros(len(grid), bool)
    for c in cars.itertuples():
        occ |= (grid + 0.5 >= c.xr_start_m) & (grid + 0.5 < c.xr_end_m)
    out = []
    free = ~occ & present
    d = np.diff(np.r_[0, free.astype(int), 0])
    for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
        if s == 0 or e >= len(grid) or not (occ[s - 1] and occ[e]):
            continue
        L = grid[e] - grid[s]
        if gap_m[0] <= L <= gap_m[1] and parkable[s:e].mean() > 0.5:
            out.append([round(float(grid[s]), 1), round(float(grid[e]), 1)])
    return out


def build(cfg: dict, lay, geo, passes: list[dict], out: Path, force: bool = False) -> dict:
    """Writes out/curb.html and out/strips/."""
    from ..config import _merge
    from ..regs.rules import PARKABLE
    lc = _merge(DEFAULTS, (cfg.get("map") or {}).get("labeller") or {})
    root = Path(cfg["out_dir"])
    sd = out / "strips"
    sd.mkdir(parents=True, exist_ok=True)
    cars = pd.read_csv(root / "multiday" / "cars_ref.csv")
    cs = pd.read_csv(root / "supply" / "curb_status.csv")
    ints = lay.intersections
    x_lo, x_hi = lay.extent_m
    grid = np.arange(np.floor(x_lo), x_hi, 1.0)
    rows = {}
    for r, rc in lay.rows.items():
        pres = lay.present(r, grid + 0.5)
        if not pres.any():
            continue
        x0 = max(x_lo, float(grid[pres].min()) - 15)
        x1 = min(x_hi, float(grid[pres].max()) + 16)
        g = cs[cs["row"] == r]
        stcols = [c for c in g.columns if c.startswith("status_")]
        # status bins are 1 m wide, centred at x_ref_m; grid value k = the bin [k, k+1)
        st = g.set_index(np.round(g.x_ref_m - 0.5).astype(int))[stcols]
        st = st[~st.index.duplicated()]
        stx = st.reindex(grid.astype(int))
        park = stx.isin(PARKABLE).all(axis=1).values & pres
        blocked = []
        for s0, s1 in _runs_of(~park & pres, grid):
            k = stx.iloc[int(s0 - grid[0])].dropna()
            blocked.append([s0, s1, str(k.iloc[0]) if len(k) else ""])
        strip_list = []
        for p in passes:
            s = strips(cfg, lay, root, p["id"], r, x0, x1, sd, lc, force)
            if s is None:
                continue
            k = cars[(cars["clip"] == p["id"]) & (cars["row"] == r)]
            strip_list.append({**s, "label": p["label"], "dir": p["dir"],
                               "cars": k[["xr_start_m", "xr_end_m"]].round(1).values.tolist()})
        strip_list.sort(key=lambda s: (s["plane"] != "near", s["clip"]))
        kr = cars[cars["row"] == r]
        side = rc["curb"]
        xs = np.arange(x0, x1 + 1, 1.0)
        rows[r] = {
            "curb": side, "x0": x0, "x1": x1, "strips": strip_list,
            "present": [list(v) for v in _runs_of(pres, grid)],
            "blocked": blocked,
            "candidates": gap_candidates(kr, pres, park, grid, lc["gap_m"]),
            "intersections": [[round(q.x0_m, 1), round(q.x1_m, 1), str(q.name)]
                              for q in ints.itertuples()
                              if q.x1_m > x0 and q.x0_m < x1],
            "curb_ll": {"x0": float(xs[0]), "step": 1.0,
                        "ll": geo.line(float(xs[0]), float(xs[-1]), side, 0.0, 1.0)},
        }
    lf = labels_file(cfg)
    data = {"run": cfg["name"], "street": cfg["street"].get("name"),
            "px_per_m": lc["px_per_m"], "chunk_m": lc["chunk_m"], "y_m": lc["y_m"], "rows": rows,
            "labels_file": str(lf) if lf else None, "labels": read_labels(lf)}
    tpl = (Path(__file__).parent / "labeller.html").read_text()
    (out / "curb.html").write_text(tpl.replace("/*__DATA__*/null",
                                               json.dumps(data, ensure_ascii=False)))
    return {"rows": {r: len(v["strips"]) for r, v in rows.items()},
            "candidates": sum(len(v["candidates"]) for v in rows.values())}


def _runs_of(mask: np.ndarray, grid: np.ndarray):
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return [(float(grid[a]), float(grid[b - 1] + 1.0))
            for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0])]
