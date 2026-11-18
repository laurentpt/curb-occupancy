"""Manual regulation source: zones and points written in the street config, for cities
without open data, for driveways (no open data anywhere), and for corrections checked on
site.

    - type: manual
      name: site_survey_2026-11          # optional; default 'manual'
      zones:
        - {x_m: [412, 417], row: A, kind: driveway, note: "curb cut"}
        - {latlon: [[45.4652, -73.6361], [45.4655, -73.6358]], curb: right, kind: no_parking,
           rule: {hours: [[7, 9]], days: [mon, tue, wed, thu, fri]}, note: "photo 0412"}
      points:
        - {x_m: 640, row: A, kind: hydrant}

`file: <path>`: more zones/points from a YAML file with the same `zones:` / `points:` keys
(the curb labeller, `labeller.py`, writes it).

Position: `x_m` (street-axis metres; tied to the current axis) or `latlon` (projected on the
axis; preferred, survives axis changes). Row: `row`, or `curb` (right/left of the axis
direction: the row present there on that curb), or neither (every row present at the
zone's middle). `rule` (rules.rule_from_dict) defaults to always.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .rules import KINDS, POINT_COLS, ZONE_COLS, Regs, rule_from_dict


def _span(item: dict, ctx) -> tuple[float, float, float | None]:
    """(x0, x1, lateral of the middle or None)."""
    if "x_m" in item:
        v = np.atleast_1d(np.asarray(item["x_m"], float))
        return float(v.min()), float(v.max()), None
    ll = np.asarray(item["latlon"], float).reshape(-1, 2)
    pr = ctx.project(ll[:, 0], ll[:, 1])
    return float(pr[:, 1].min()), float(pr[:, 1].max()), float(pr[:, 2].mean())


def _rows(item: dict, ctx, x: float, lat: float | None) -> list[str]:
    lay = ctx.lay
    if item.get("row"):
        return [item["row"]]
    here = [r for r in lay.rows if lay.present(r, np.array([x]))[0]]
    curb = item.get("curb") or (None if lat is None else ("right" if lat < 0 else "left"))
    return [r for r in here if curb is None or lay.rows[r]["curb"] == curb]


def load(sc: dict, ctx) -> Regs:
    from ..publish.labeller import read_labels
    name = sc.get("name") or "manual"
    zones, points, rules = [], [], {}
    extra = read_labels(Path(sc["file"])) if sc.get("file") else []
    for i, z in enumerate((sc.get("zones") or []) + extra):
        if z.get("kind") not in KINDS:
            raise ValueError(f"manual zone {i}: kind {z.get('kind')!r} not in {KINDS}")
        x0, x1, lat = _span(z, ctx)
        key = f"{name}:{i}"
        rules[key] = rule_from_dict(z.get("rule"), z["kind"], z.get("note", ""))
        for r in _rows(z, ctx, (x0 + x1) / 2, lat):
            zones.append({"row": r, "x0": x0, "x1": x1, "kind": z["kind"], "rule": key,
                          "source": name, "ref": z.get("note", ""), "text": z.get("note", "")})
    for i, p in enumerate(sc.get("points") or []):
        x0, x1, lat = _span(p, ctx)
        for r in _rows(p, ctx, x0, lat):
            points.append({"row": r, "x_ref": x0, "kind": p.get("kind", "hydrant"),
                           "source": name, "ref": p.get("note", "")})
    return Regs(zones=pd.DataFrame(zones, columns=ZONE_COLS),
                points=pd.DataFrame(points, columns=POINT_COLS), rules=rules,
                info={"n_zones": len(zones), "n_points": len(points)})
