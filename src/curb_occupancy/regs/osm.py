"""OpenStreetMap curb cuts as a regulation source: `highway=service` ways (alleys, driveways,
parking aisles) joining the street are driveways on the curb they join (the layout's
`osm_crossings.csv`, street axis 'osm'). OSM maps few residential driveways; this only
catches alleys and lot entrances.

    - type: osm
      curb_cut_m: 6.0      # curb length lost per cut (no width in the data)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .rules import POINT_COLS, ZONE_COLS, Regs, rule_from_dict


def load(sc: dict, ctx) -> Regs:
    lay, w = ctx.lay, float(sc.get("curb_cut_m", 6.0))
    cr = lay.crossings
    cuts = cr[(cr.kind == "curb_cut") & cr.x_m.between(*lay.extent_m)] if len(cr) else cr
    zones = []
    for c in cuts.itertuples():
        sides = set(str(c.sides).split("+"))
        for r, rc in lay.rows.items():
            if rc["curb"] in sides and lay.present(r, np.array([c.x_m]))[0]:
                zones.append({"row": r, "x0": c.x_m - w / 2, "x1": c.x_m + w / 2,
                              "kind": "driveway", "rule": "osm:curb_cut", "source": "osm",
                              "ref": f"way/{c.osm_way}", "text": f"service={c.service or 'yes'}"})
    return Regs(zones=pd.DataFrame(zones, columns=ZONE_COLS),
                points=pd.DataFrame(columns=POINT_COLS),
                rules={"osm:curb_cut": rule_from_dict(None, "driveway", "OSM curb cut")},
                info={"n_curb_cuts": len(zones), "axis": lay.axis.kind if lay.axis else None})
