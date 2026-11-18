from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from curb_occupancy.adapters.montreal import build_zones, parse_rule
from curb_occupancy.regs.supply import curb_status


def test_parse_rule_time_windows():
    r = parse_rule("P 15 MIN 07h-09h30 ET 15h-18h LUN A VEN")
    assert r.kind == "limited" and r.minutes == 15 and r.days == {0, 1, 2, 3, 4}
    assert r.active(datetime(2026, 10, 23, 17, 30)) and not r.active(datetime(2026, 10, 23, 18, 30))
    assert not r.active(datetime(2026, 10, 24, 17, 30))           # Saturday
    c = parse_rule("\\P 12h30-14h30 MERCREDI 1 AVRIL AU 1 DEC")
    assert c.kind == "no_parking" and c.active(datetime(2026, 10, 21, 13, 0))
    assert not c.active(datetime(2026, 12, 9, 13, 0))             # out of season
    assert parse_rule("\\P EXCEPTE HANDICAPES (PICTO)").kind == "reserved"
    assert parse_rule("\\P EXCEPTE S3R 9h-23h").kind == "permit"
    assert parse_rule("\\A EN TOUT TEMPS").active(datetime(2026, 1, 1, 3, 0))


def _signs(rows):
    return pd.DataFrame(rows, columns=["POTEAU_ID_POT", "x_ref", "FLECHE_PAN", "DESCRIPTION_RPA",
                                       "kind"])


def test_zone_rules():
    ints = pd.DataFrame({"x0_m": [0.0, 200.0], "x1_m": [10.0, 210.0]})
    A, P15 = "\\A EN TOUT TEMPS", "P 15 MIN 15h-18h LUN A VEN"
    S = _signs([(1, 25.0, 3, A, "no_stopping"),             # lone: ends the approach at 10 m
                (2, 80.0, 2, P15, "limited"), (2, 80.0, 3, A, "no_stopping"),
                (3, 120.0, 3, P15, "limited"), (3, 120.0, 2, A, "no_stopping"),
                (4, 150.0, 0, A, "no_stopping")])           # no arrow: both ways
    z = build_zones(S, ints, 300.0)
    get = lambda pole, kind: z[(z.pole == pole) & (z.kind == kind)][["x0", "x1"]].values.tolist()
    assert get(1, "no_stopping") == [[10.0, 25.0]]
    assert get(2, "limited") == [[80.0, 120.0]] and get(3, "limited") == [[80.0, 120.0]]
    assert get(2, "no_stopping") == [[10.0, 80.0]] or get(2, "no_stopping") == [[25.0, 80.0]]
    assert sorted(get(4, "no_stopping")) == [[120.0, 150.0], [150.0, 200.0]]
    rules = {t: parse_rule(t) for t in (A, P15)}
    grid = np.arange(0, 300, 1.0)
    z = z.assign(rule=z.rpa, disputed=False)
    clear = {"intersection": 5.0, "hydrant": 5.0}
    st17 = curb_status(grid, z, rules, ints, pd.DataFrame(), datetime(2026, 10, 23, 17), clear)
    st19 = curb_status(grid, z, rules, ints, pd.DataFrame(), datetime(2026, 10, 23, 19), clear)
    assert st17[100] == "limited" and st19[100] == "legal"
    assert st17[5] == "intersection" and st17[212] == "intersection" and st17[260] == "legal"


def test_manual_rules_and_points():
    from curb_occupancy.regs.rules import rule_from_dict
    r = rule_from_dict({"hours": [[7, 9]], "days": ["mon", "fri"]}, "no_parking")
    assert r.active(datetime(2026, 10, 23, 8)) and not r.active(datetime(2026, 10, 22, 8))
    assert rule_from_dict(None, "driveway").active(datetime(2026, 1, 1, 3))
    grid = np.arange(0, 100, 1.0)
    zones = pd.DataFrame({"x0": [40.0], "x1": [45.0], "kind": ["driveway"], "rule": ["d"],
                          "disputed": [False]})
    pts = pd.DataFrame({"x_ref": [70.0], "kind": ["hydrant"]})
    st = curb_status(grid, zones, {"d": rule_from_dict(None, "driveway")}, pd.DataFrame(), pts,
                     datetime(2026, 10, 23, 18), {"hydrant": 5.0})
    assert (st[40:45] == "driveway").all() and st[39] == st[45] == "legal"
    assert (st[65:75] == "hydrant").all() and st[64] == st[75] == "legal"


def test_prior_zones_corners_keep_and_ignored():
    from curb_occupancy.adapters.montreal import build_zones_prior, convention_dir
    assert convention_dir(2, "right") == 1 and convention_dir(3, "right") == -1
    assert convention_dir(2, "left") == -1 and convention_dir(0, "left") == 0
    ints = pd.DataFrame({"x0_m": [0.0, 200.0], "x1_m": [10.0, 210.0]})
    A, H = "\\A EN TOUT TEMPS", "\\P EXCEPTE HANDICAPES (PICTO)"
    S = _signs([(1, 25.0, 2, A, "no_stopping"),     # corner panel, arrow 'wrong': still to 10
                (2, 185.0, 3, A, "no_stopping"),    # corner panel at the far end
                (3, 100.0, 0, A, "no_stopping"),    # mid-block, no arrow: ignored
                (4, 120.0, 2, A, "no_stopping"),    # mid-block, arrowed, kept: +X to pole 2
                (5, 60.0, 2, H, "reserved")])       # lone reserved panel: one space, +X
    prior = {"corner_m": 45.0, "reserved_m": 6.5}
    z, ig = build_zones_prior(S, ints, 300.0, "right", prior, keep_x=[(110.0, 130.0)])
    get = lambda pole: z[z.pole == pole][["x0", "x1"]].values.tolist()
    assert get(1) == [[10.0, 25.0]] and get(2) == [[185.0, 200.0]]
    assert get(4) == [[120.0, 185.0]]
    assert get(5) == [[60.0, 66.5]]
    assert ig.pole.tolist() == [3]
    z, ig = build_zones_prior(S, ints, 300.0, "right", prior)       # nothing kept
    assert sorted(ig.pole.tolist()) == [3, 4]


def test_occupancy_classes():
    from curb_occupancy.regs.guideline import PROFILES, occupancy_class
    c = PROFILES["montreal"]["classes"]
    assert [occupancy_class(v, c) for v in (0.2, 0.5, 0.85, 0.9, 1.3)] == \
        ["very_low", "low", "optimal", "full", "full"]
    assert occupancy_class(float("nan"), c) is None


def test_partition_groups_consecutive_blocks_near_target():
    from curb_occupancy.regs.supply import partition
    w = np.array([5, 9, 10, 8, 7, 9, 4, 3, 3, 8, 17], float)
    g = partition(w, 25)
    assert g[0][0] == 0 and g[-1][1] == len(w)
    assert all(a[1] == b[0] for a, b in zip(g[:-1], g[1:]))      # contiguous, no gaps
    assert all(abs(w[i:j].sum() - 25) <= 6 for i, j in g)
    assert partition(np.array([3.0]), 25) == [(0, 1)]


def test_fixed_groups_by_cross_streets():
    from curb_occupancy.regs.supply import fixed_groups
    b = pd.DataFrame({"from": [np.nan, "A", "A", "B", "C"], "to": ["A", "A", "B", "C", np.nan]})
    assert fixed_groups(b, [[None, "A"], ["A", "C"], ["C", None]]) == [(0, 1), (1, 4), (4, 5)]
    assert fixed_groups(b, [["A", "B"], ["B", "C"]]) == [(1, 3), (3, 4)]   # end blocks out
    with pytest.raises(ValueError):
        fixed_groups(b, [[None, "A"], ["B", "C"]])   # gap between stretches
    with pytest.raises(ValueError):
        fixed_groups(b, [["D", "C"]])                # no such cross street
