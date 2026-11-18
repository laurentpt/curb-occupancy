import json

import numpy as np
import pandas as pd

from curb_occupancy import capacity as K

LAT0, LON0 = 45.47, -73.62
KY, KX = 111195.0, 111195.0 * np.cos(np.radians(LAT0))


def _ll(x, y):
    return {"lat": LAT0 + y / KY, "lon": LON0 + x / KX}


def _osm():
    """N-S 'Avenue Test' (x = 0, y -10..610) crossing E-W streets at y = 0, 250, 300, 600."""
    els, nid = [], [100]

    def node():
        nid[0] += 1
        return nid[0]
    ys = [-10, 0, 120, 250, 300, 450, 600, 610]
    ids = {y: node() for y in ys}
    els.append({"type": "way", "id": 1, "tags": {"highway": "residential", "name": "Avenue Test"},
                "nodes": [ids[y] for y in ys], "geometry": [_ll(0, y) for y in ys]})
    for k, (y, name) in enumerate([(0, "South St"), (250, "Mid St"), (300, "Lane St"),
                                   (600, "North St")]):
        a, b = node(), node()
        els.append({"type": "way", "id": 10 + k, "tags": {"highway": "residential", "name": name},
                    "nodes": [a, ids[y], b], "geometry": [_ll(-80, y), _ll(0, y), _ll(80, y)]})
    return {"elements": els}


def test_osm_faces_split_and_sides():
    F = K.osm_faces(_osm(), [["Avenue Test", "South St", "North St"]], grid_north=0.0)
    spans = [(f["from"], f["to"], f["length_m"]) for f in F[::2]]
    assert spans == [("South St", "Mid St", 250.0), ("Mid St", "Lane St", 50.0),
                     ("Lane St", "North St", 300.0)]
    # riding north: left = west side, right = east side
    assert [f["side"] for f in F[:2]] == ["west side", "east side"]
    west = np.array(F[0]["line"])
    assert (west[:, 0] < LON0).all()  # drawn west of the centreline


def test_build_measured_and_estimated(tmp_path):
    osm = tmp_path / "osm.json"
    osm.write_text(json.dumps(_osm()))
    # one measured face (west side, South -> Mid) + another street that is not listed
    B = pd.DataFrame([
        {"key": "test", "street": "Avenue Test", "side": "west side", "clip": "c1", "block_id": "G01",
         "row": "L", "from": "South St", "to": "Mid St", "covered": True, "in_window": True,
         "capacity": 25, "row_length_m": 250.0},
        {"key": "test2", "street": "Avenue Test", "side": "east side", "clip": "c1", "block_id": "G02",
         "row": "L", "from": "Mid St", "to": "Lane St", "covered": True, "in_window": True,
         "capacity": 6, "row_length_m": 50.0},
        {"key": "main", "street": "Main St", "side": "south side", "clip": "c2", "block_id": "G01",
         "row": "A", "from": "X", "to": "Y", "covered": True, "in_window": True,
         "capacity": 40, "row_length_m": 300.0},
        {"key": "main", "street": "Main St", "side": "south side", "clip": "c2", "block_id": "G02",
         "row": "A", "from": "Y", "to": "Z", "covered": False, "in_window": True,
         "capacity": 99, "row_length_m": 300.0},
    ])
    F, s = K.build({"osm_json": str(osm), "grid_north_deg": 0,
                    "blocks": [["Avenue Test", "South St", "North St"]]}, B)
    assert s["rate_per_km"] == round(1000 * 31 / 300, 1)
    assert s["measured"] == {"faces": 3, "spaces": 25 + 6 + 40}  # uncovered block left out
    assert s["estimated"]["faces"] == 4 and s["estimated"]["length_m"] == 250 + 50 + 300 + 300
    est = F[F.kind == "estimated"]
    assert est.spaces.tolist() == [round(31 / 300 * L) for L in est.length_m]
    assert s["low"] < s["total"] < s["high"]


def test_restrictions(tmp_path):
    import pytest
    osm = tmp_path / "osm.json"
    osm.write_text(json.dumps(_osm()))
    B = pd.DataFrame([{"key": "test", "street": "Avenue Test", "side": "west side", "clip": "c1",
                       "block_id": "G01", "row": "L", "from": "South St", "to": "Mid St",
                       "covered": True, "in_window": True, "capacity": 25, "row_length_m": 250.0},
                      {"key": "test", "street": "Avenue Test", "side": "west side", "clip": "c1",
                       "block_id": "G03", "row": "L", "from": "Lane St", "to": "North St",
                       "covered": True, "in_window": True, "capacity": 30, "row_length_m": 300.0}])
    cfg = {"osm_json": str(osm), "blocks": [["Avenue Test", "South St", "North St"]],
           "restrictions": [{"street": "Avenue Test", "from": "North St", "to": "Lane St",
                             "side": "east side", "restricted_m": 100}]}
    F, s = K.build(cfg, B)
    r = F[(F.side == "east side") & (F["from"] == "Lane St")].iloc[0]
    assert r.restricted_m == 100 and r.spaces == round(0.1 * 200)
    assert s["estimated"]["restricted_m"] == 100 and s["estimated"]["faces_restricted"] == 1
    cfg["restrictions"][0]["side"] = "west side"  # a measured face: its run has the signs
    with pytest.raises(ValueError):
        K.build(cfg, B)
