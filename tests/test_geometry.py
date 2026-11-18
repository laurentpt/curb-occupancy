import numpy as np
import pandas as pd
import pytest

from curb_occupancy import config as C
from curb_occupancy.capture.lens import FisheyeLens, PinholeView
from curb_occupancy.capture import motion as M
from curb_occupancy.count import union_length
from curb_occupancy.axismap import sequence_match

LENS = C.rig("gopro_hero8_wide_hseis")["camera"]["lens_profile"]


@pytest.fixture(scope="module")
def lens():
    return FisheyeLens.from_gyroflow(LENS, 1920, 1440)


def test_lens_scaled_from_4k(lens):
    assert lens.K[0, 0] == pytest.approx(1915.597 * 0.48, rel=1e-4)
    assert lens.K[0, 2] == pytest.approx(2014.951 * 0.48, rel=1e-4)


def test_project_unproject_roundtrip(lens):
    rng = np.random.default_rng(0)
    uv = np.stack([rng.uniform(200, 1700, 200), rng.uniform(150, 1300, 200)], 1)
    uv2 = lens.project(lens.unproject(uv))
    assert np.abs(uv2 - uv).max() < 0.05


def test_torch_projection_matches_numpy(lens):
    import torch
    rays = np.random.default_rng(1).normal(size=(500, 3))
    rays[:, 2] = np.abs(rays[:, 2]) + 0.5
    a = lens.project(rays)
    b = lens.project_torch(torch.tensor(rays, dtype=torch.float64)).numpy()
    assert np.allclose(a, b, atol=1e-6)


def test_virtual_rotation_axes():
    # camera yawed 75 deg left of travel, rolled 20 deg, level otherwise
    up_cam = np.array([[np.sin(np.radians(20)), -np.cos(np.radians(20)), 0.0]])
    down = -up_cam
    # travel direction: 75 deg to the camera's right in the levelled plane
    z_h = np.array([[0, 0, 1.0]])
    right = np.cross(down, z_h)
    fwd = np.cos(np.radians(75)) * z_h + np.sin(np.radians(75)) * right
    R = M.virtual_rotations(fwd, down, 90.0, 0.0)[0]
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-9)
    assert np.linalg.det(R) == pytest.approx(1.0)
    # virtual x = forward, y = down, z = left (= -right of travel)
    assert np.allclose(R[:, 0], fwd[0])
    assert np.allclose(R[:, 1], down[0])
    assert np.allclose(R[:, 2], -np.cross(down[0], fwd[0]))


def test_pinhole_from_hfov():
    v = PinholeView.from_hfov(480, 432, 90)
    assert v.f == pytest.approx(240.0)


def test_union_length():
    assert union_length([0, 2, 10], [3, 5, 12]) == pytest.approx(7.0)
    assert union_length([], []) == 0.0


def test_sequence_match_handles_gaps():
    A = pd.DataFrame({"length_m": [4.5, 4.6, 5.0, 4.4], "lab_L": [50, 80, 20, 60],
                      "lab_a": [0] * 4, "lab_b": [0] * 4})
    B = pd.DataFrame({"length_m": [4.5, 5.0, 4.4], "lab_L": [50, 20, 60],
                      "lab_a": [0] * 3, "lab_b": [0] * 3})
    xa = np.array([0.0, 5.5, 11.0, 16.5])
    xb = np.array([0.8, 11.6, 17.0])  # second car of A left between passes
    pairs = sequence_match(A, B, xa, xb)
    assert [(i, j) for i, j, _ in pairs] == [(0, 0), (2, 1), (3, 2)]


def test_clipped_contact_bounds_depth():
    """A box cut by the visible bottom (mosaic edge or black out-of-FOV rows) has only an upper
    bound on depth: kept + flagged when that bound reaches the plane's band, else rejected."""
    import pandas as pd
    from curb_occupancy.detect import Acceptance, accept
    from curb_occupancy.mosaic import Mosaic
    h, P = 1.0, 100.0                       # h units: 100 px per h
    img = np.full((300, 2000, 3), 128, np.uint8)
    img[260:, 1000:] = 0                    # right half: FOV ends 40 px above the mosaic bottom
    m = Mosaic(image=img, px_per_m=P, depth=2.0, y_top=2.0, x0=0.0,
               col_frame=np.zeros(2000, np.int32), t0=0.0, t1=1.0)
    acc = Acceptance(lateral_m=(0.8, 3.0))
    box = dict(conf=0.9, cls=2, cls_name="car")
    df = pd.DataFrame([{**box, "x0": 100, "x1": 550, "y0": 150, "y1": 300},   # cut by mosaic bottom
                       {**box, "x0": 1100, "x1": 1550, "y0": 110, "y1": 258},  # cut by FOV edge
                       {**box, "x0": 1100, "x1": 1550, "y0": 60, "y1": 200}])  # contact visible
    out = accept(df, m, h, acc)
    # bottom row 300 -> Y = -1 h -> depth bound 2 / (1 - -1) = 1.0 h: inside the band -> kept
    assert out.clipped.tolist() == [True, True, False]
    assert out.kept[0] and out.flag[0] == "clipped"
    # FOV edge at row 260 -> Y = -0.6 -> bound 1.25 h: kept too
    assert out.kept[1] and out.flag[1] == "clipped"
    # the same cut in a far plane (band starts at 3.8 h): nearer than the parked row
    far = accept(df, Mosaic(image=img, px_per_m=P, depth=6.0, y_top=2.0, x0=0.0,
                            col_frame=np.zeros(2000, np.int32), t0=0.0, t1=1.0), h,
                 Acceptance(lateral_m=(3.8, 8.0)))
    assert far.reject[0] == "clipped_bottom" and not far.kept[0]


def test_merge_splits_rejoins_bollard_split_only():
    from curb_occupancy.detect import merge_splits
    P = 85.0
    m = lambda a, b, y1=240.0: (a * P, 150.0, b * P, y1, 0.8, 2, "car")
    df = pd.DataFrame([m(10.0, 12.1), m(12.2, 14.1),          # 4.1 m car split by a bollard
                       m(20.0, 22.7), m(23.2, 25.9),          # two short real cars (>= 2.4 m)
                       m(30.0, 34.2), m(34.3, 34.7, 215.0)],  # car + bollard sliver (higher)
                      columns=["x0", "y0", "x1", "y1", "conf", "cls", "cls_name"])
    out = merge_splits(df, P)
    L = ((out.x1 - out.x0) / P).round(1).tolist()
    assert L == [4.1, 2.7, 2.7, 4.2, 0.4] and out.merged.tolist() == [2, 1, 1, 1, 1]
