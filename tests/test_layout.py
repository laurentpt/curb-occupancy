import numpy as np
import pandas as pd
import pytest

from curb_occupancy.axismap import AxisMap, align
from curb_occupancy.count import apply_layout
from curb_occupancy.layout import Layout, viterbi_states
from curb_occupancy.street import plane_pairs

# two rows on alternating curbs (rue de Terrebonne): A seen by the near plane riding forward
ROWS = {"A": {"curb": "right", "plane": {"forward": "near", "backward": "far"}},
        "B": {"curb": "left", "plane": {"forward": "far", "backward": "near"}}}
from curb_occupancy.mosaic import FlowProfiles, ground_slope


def _row(starts, plane, length=4.6):
    s = np.asarray(starts, float)
    return pd.DataFrame({"x_start_m": s, "x_end_m": s + length, "plane": plane})


def test_axis_map_reverse_and_interval():
    m = AxisMap(scale=0.99, offset=1200.0, reverse=True)
    assert m(0.0) == pytest.approx(1200.0)
    a, b = m.interval(np.array([10.0]), np.array([14.0]))
    assert a[0] < b[0] and b[0] - a[0] == pytest.approx(3.96)
    assert AxisMap.from_dict(m.to_dict()) == m


@pytest.mark.parametrize("reverse", [False, True])
def test_align_recovers_offset(reverse):
    rng = np.random.default_rng(3)
    near = np.cumsum(rng.uniform(5.2, 14, 40))          # parked row of clip a (near plane)
    ca = _row(near, "near")
    La, off = float(near[-1] + 20), 37.0
    if reverse:  # b rides the other way: same cars, reversed axis, far plane
        xb = (La + off) - (near + 4.6)
        cb = _row(xb, "far")
    else:
        cb = _row(near - off, "near")
    al = align(ca, cb, La, La + off, reverse, plane_pairs(ROWS, False, reverse), scales=[1.0])
    m = AxisMap(al["scale"], al["offset_m"], reverse)
    mid_b = (cb.x_start_m + cb.x_end_m) / 2
    assert np.abs(m(mid_b) - (ca.x_start_m + ca.x_end_m) / 2).max() < 0.5


def test_viterbi_ignores_short_contrary_evidence():
    n = 300
    e_near, e_far = np.zeros(n), np.zeros(n)
    e_near[:180] = 0.8            # long near-parking stretch
    e_far[60:90] = 0.8            # 30 m of travel-lane traffic in the far band
    e_far[200:] = 0.8             # real switch at ~190 m
    E = np.stack([e_near, e_far], 1)
    st = viterbi_states(E, np.array([[1, 0], [0, 1]], bool), switch_cost=30.0)
    assert (st[:180] == 0).all() and (st[200:] == 1).all()
    assert np.count_nonzero(np.diff(st)) == 1


def test_viterbi_independent_rows():
    """Rows may coexist: states none / A / B / A+B. Car evidence cannot tell an empty row from
    an absent one, so a row switches off only after switch_cost / row_cost metres without
    cars: a 15 m gap stays 'present'; 200 m without cars switches row B off at row_cost 0.15
    (threshold 200 m) but not at 0.05 (600 m)."""
    n = 400
    E = np.zeros((n, 2))
    E[:, 0] = np.where(np.arange(n) % 12 < 5, 0.8, 0.0)   # row A: cars along the whole street
    E[:200, 1] = np.where(np.arange(200) % 12 < 5, 0.8, 0.0)  # row B only in the first 200 m
    E[60:75, 1] = 0.0                                        # 15 m gap in row B: kept present
    M = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], bool)
    st = viterbi_states(E, M, switch_cost=30.0, row_cost=0.15)
    assert (st[:195] == 3).all() and (st[210:] == 1).all()
    assert (viterbi_states(E, M, switch_cost=30.0, row_cost=0.05) == 3).all()


def test_apply_layout_gates_by_side():
    maps = {"a": AxisMap(), "b": AxisMap(1.0, 100.0, True)}
    lay = Layout(reference="a", directions={"a": "d1", "b": "d2"}, maps=maps,
                 switches_m=[50.0], states=[("A",), ("B",)], extent_m=(0.0, 100.0),
                 intersections=pd.DataFrame(), blocks=pd.DataFrame(), rows=ROWS, ref_maps=maps)

    def dets(starts, depth):
        d = _row(starts, None).drop(columns="plane")
        return d.assign(kept=True, reject="", flag="", conf=0.9, t=0.0, cls_name="car",
                        length_m=4.6, height_m=1.5, depth_m=depth, flow_ratio=1.0)

    # clip a: near row active before 50 m. Inactive-plane boxes: the near one at 1.5 m is in
    # the travel lane (far-parking segment); the far one at 6 m is across the street
    g, rev = apply_layout({"near": dets([10, 60], 1.5), "far": dets([10, 70], 6.0)}, lay, "a",
                          travel_lane_m={"near": [0.8, 3.8], "far": [2.8, 5.0]})
    assert g["near"].kept.tolist() == [True, False]
    assert g["far"].kept.tolist() == [False, True]
    assert set(rev.reason) == {"travel_lane", "other_side"}
    # clip b rides the other way: its x=10 is reference 90 (far side for ref) -> b's near plane
    g, _ = apply_layout({"near": dets([8], 1.5), "far": dets([8], 5.0)}, lay, "b")
    assert g["near"].kept.tolist() == [True] and g["far"].kept.tolist() == [False]


def test_ground_slope_removes_upward_spikes_keeps_braking():
    n, H, f, cy = 1200, 100, 100.0, 0.0
    v = np.full(n, 0.1)
    v[600:] = np.linspace(0.1, 0.05, 600)                # braking ramp
    dy = np.arange(H) - cy
    u = -(v[:, None] * dy[None, :])                      # road flow, linear in the row
    u[300:400] *= 2.0                                    # close object covers all road rows
    fp = FlowProfiles(t=np.arange(n) / 60, u=u.astype(np.float32), v=np.zeros_like(u),
                      spread=np.zeros_like(u), f=f, cy=cy)
    s = ground_slope(fp)
    assert np.abs(s[300:400] / v[300:400] - 1).max() < 0.02
    assert np.abs(s[700:1100] / v[700:1100] - 1).max() < 0.02


def test_view_side_right_is_mirrored_left():
    from curb_occupancy.capture.motion import virtual_rotations
    fwd = np.array([[0.0, 0.0, 1.0]])
    down = np.array([[0.0, 1.0, 0.0]])
    right = np.cross(down, fwd)[0]
    L = virtual_rotations(fwd, down, side="left")[0]
    R = virtual_rotations(fwd, down, side="right")[0]
    assert np.allclose(L[:, 2], -right) and np.allclose(R[:, 2], right)   # optical axis
    assert np.allclose(L[:, 0], fwd[0]) and np.allclose(R[:, 0], fwd[0])  # image x = forward
    assert np.allclose(L[:, 1], down[0]) and np.allclose(R[:, 1], down[0])


def test_forced_segment_only_where_configured():
    from curb_occupancy.layout import forced_segments, viterbi_states
    grid = np.arange(0.0, 300.0)
    M = np.array([[True, False], [False, True]])               # states [A], [B]
    cross = pd.DataFrame({"kind": ["intersection"] * 2, "x_m": [100.0, 190.0],
                          "name": ["Royal", "Draper"]})
    M2, force, bounds = forced_segments([{"between": ["Draper", "Royal"], "rows": ["A", "B"]}],
                                        cross, grid, ["A", "B"], M)
    assert M2.shape == (3, 2) and bounds == [100.0, 190.0]
    assert (force[100:190] == 2).all() and (force[:100] == -1).all() and (force[190:] == -1).all()
    E = np.zeros((300, 2))
    E[:, 0] = 1.0                                              # evidence for A everywhere
    E[:, 1] = 1.0                                              # ... and B: A+B would be cheapest
    st = viterbi_states(E, M2, 30.0, 0.05, force, n_free=2)
    assert set(st[100:190]) == {2} and 2 not in set(st[:100]) | set(st[190:])
