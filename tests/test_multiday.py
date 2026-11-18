import numpy as np
import pandas as pd
import pytest

from curb_occupancy.layout import _resolve_directions
from curb_occupancy.multiday import _closing, _runs
from curb_occupancy.capture.telemetry import Telemetry, gps_scale_check


def test_resolve_directions_auto_reference():
    labels = {"r": "auto", "a": "d1", "b": "d2", "c": "auto"}
    rev = {"r": False, "a": True, "b": False, "c": True}
    assert _resolve_directions(labels, rev, "r") == {"r": "d2", "a": "d1", "b": "d2", "c": "d1"}
    # no configured clip rides the other way: invent a label
    assert _resolve_directions({"r": "d1", "x": "auto"}, {"r": False, "x": True}, "r") == \
        {"r": "d1", "x": "d1_rev"}


def test_gps_scale_check_recovers_height():
    fps, h_true = 60.0, 0.93
    t = np.arange(0, 120, 1 / fps)
    v = 5.5 + 1.0 * np.sin(t / 7)                      # m/s
    s = v / fps / h_true                               # visual slope (h units / frame)
    gt = np.arange(0, 120, 0.1)
    rng = np.random.default_rng(0)
    vg = np.interp(gt, t, v) + rng.normal(0, 0.3, len(gt))
    lat = 45.46 + np.cumsum(np.interp(gt, t, v)) * 0.1 / 111_000
    gps = np.stack([lat, np.full_like(gt, -73.64), np.zeros_like(gt), vg, vg], 1)
    tel = Telemetry(imu_t=t, gyro=None, accel=None, gps_t=gt, gps=gps,
                    gps_fix=np.full(len(gt), 3), gps_dop=np.full(len(gt), 250),
                    exp_t=t, shutter_s=t, iso=t)
    tr = pd.DataFrame({"t": t, "s_h": s, "X_m": np.cumsum(s) * 1.0})
    chk = gps_scale_check(tel, tr, fps, h=1.0)
    assert chk["h_gps_m"] == pytest.approx(h_true, rel=0.01)
    assert chk["gps_path_m"] == pytest.approx(np.cumsum(v)[-1] / fps, rel=0.02)


def test_curb_closing_and_runs():
    m = np.array([0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0], bool)
    c = _closing(m, 2)
    assert c.tolist() == [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0]
    grid = np.arange(len(m)) * 1.0
    assert _runs(c, grid, 1.0) == [(1.0, 6.0), (10.0, 12.0)]


def test_intersections_vote_per_direction_and_car_veto():
    from curb_occupancy.blocks import find_intersections
    grid = np.arange(0, 200, 1.0)
    base = np.full(len(grid), 5.0)
    d1a, d1b, d2 = base.copy(), base.copy(), base.copy()
    for d in (d1a, d1b, d2):
        d[50:62] = 40.0                 # cross street: deep from both directions
    d1a[120:140] = d1b[120:140] = 25.0  # open yard on the d1 side only
    d1a[160:175] = 35.0                 # very deep yard seen by one pass, but cars park there
    cars = [np.array([167.0]), np.array([]), np.array([])]
    ints, dropped = find_intersections([d1a, d1b, d2], grid, groups=["d1", "d1", "d2"],
                                       cars_ref=cars)
    assert len(ints) == 1 and 50 <= ints.centre_m[0] <= 62
    assert len(dropped) == 1 and dropped.x0_m[0] == 160


def test_project_polyline_lateral_is_left_of_increasing_x():
    from curb_occupancy.street import project_polyline
    P = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])   # east
    q = np.array([[5.0, 3.0]])                             # north of the line
    fwd = project_polyline(q, P, np.array([0.0, 10.0, 20.0]))[0]
    rev = project_polyline(q, P[::-1], np.array([0.0, 10.0, 20.0]))[0]  # axis runs west
    assert fwd[1] == 5.0 and fwd[2] == 3.0       # left of an eastward axis
    assert rev[1] == 15.0 and rev[2] == -3.0     # right of a westward axis
    # same westward axis, vertices stored against it
    back = project_polyline(q, P, np.array([20.0, 10.0, 0.0]))[0]
    assert back[1] == 15.0 and back[2] == -3.0
