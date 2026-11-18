"""Ego-motion without GPS: heading from two-view geometry, orientation from gravity.

Stage 1 (heading): KLT tracks between frames i and i+step on the raw (fisheye) video,
undistorted to normalized rays, essential matrix with RANSAC -> translation direction in the
camera frame. Combined with the gravity 'up' vector this gives the street-aligned frame
(forward, down, left) per frame, i.e. the rotation of the virtual camera.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

from .lens import FisheyeLens, rot_x, rot_y
from .telemetry import Telemetry
from .video import iter_frames, probe


@dataclass
class HeadingTrack:
    t: np.ndarray        # s, time of first frame of each pair
    tdir: np.ndarray     # (N,3) unit translation direction, camera frame
    n_inliers: np.ndarray
    rot_deg: np.ndarray  # rotation angle between the two frames (from E), deg
    parallax: np.ndarray # median inlier displacement (normalized units)

    def save(self, path):
        np.savez_compressed(path, **self.__dict__)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(**{k: z[k] for k in z.files})


def estimate_heading(video: str, lens: FisheyeLens, step: int = 3, scale: float = 0.5,
                     every: int = 6, max_corners: int = 1500, progress=None) -> HeadingTrack:
    """Pairs (i, i+step) for i in range(0, N, every)."""
    info = probe(video)
    w, h = int(info.width * scale), int(info.height * scale)
    L = lens.scaled(w, h)
    # read frames at stride gcd(step, every)
    stride = np.gcd(step, every)
    prev = {}
    out_t, out_d, out_n, out_r, out_p = [], [], [], [], []
    lk = dict(winSize=(21, 21), maxLevel=4,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    # mask out the extreme fisheye corners (black / heavily stretched)
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.hypot((xx - L.K[0, 2]) / L.K[0, 0], (yy - L.K[1, 1] * 0 - L.K[1, 2]) / L.K[1, 1])
    mask = (rr < 1.15).astype(np.uint8) * 255
    for idx, img in iter_frames(video, w, h, gray=True, step=stride):
        if progress and idx % 600 == 0:
            progress(idx, info.n_frames)
        if idx % every == 0:
            prev[idx] = img.copy()
        i0 = idx - step
        if i0 in prev:
            a = prev.pop(i0)
            p0 = cv2.goodFeaturesToTrack(a, max_corners, 0.01, 8, mask=mask, blockSize=7)
            if p0 is None or len(p0) < 30:
                continue
            p1, st, _ = cv2.calcOpticalFlowPyrLK(a, img, p0, None, **lk)
            pb, stb, _ = cv2.calcOpticalFlowPyrLK(img, a, p1, None, **lk)
            ok = (st[:, 0] == 1) & (stb[:, 0] == 1) & (np.linalg.norm(pb - p0, axis=2)[:, 0] < 1.0)
            if ok.sum() < 30:
                continue
            r0 = L.unproject(p0[ok, 0])
            r1 = L.unproject(p1[ok, 0])
            fwd = (r0[:, 2] > 0.1) & (r1[:, 2] > 0.1)
            n0 = r0[fwd, :2] / r0[fwd, 2:]
            n1 = r1[fwd, :2] / r1[fwd, 2:]
            if len(n0) < 30:
                continue
            E, inl = cv2.findEssentialMat(n0, n1, np.eye(3), cv2.RANSAC, 0.999, 1.5e-3)
            if E is None or E.shape != (3, 3):
                continue
            n_in, R, tvec, inl2 = cv2.recoverPose(E, n0, n1, np.eye(3), mask=inl)
            # recoverPose: x1 = R x0 + t  ->  camera moves by -R^T t in frame-0 coordinates
            c = (-R.T @ tvec).ravel()
            c /= np.linalg.norm(c)
            m = inl2.ravel() > 0
            par = np.median(np.linalg.norm(n1[m] - n0[m], axis=1)) if m.any() else 0
            out_t.append(i0 / info.fps)
            out_d.append(c)
            out_n.append(int(n_in))
            out_r.append(np.degrees(np.linalg.norm(cv2.Rodrigues(R)[0])))
            out_p.append(par)
        # drop stale
        for k in [k for k in prev if k < idx - step]:
            prev.pop(k)
    return HeadingTrack(np.array(out_t), np.array(out_d).reshape(-1, 3), np.array(out_n),
                        np.array(out_r), np.array(out_p))


def street_frames(t: np.ndarray, tel: Telemetry, head: HeadingTrack, smooth_s: float = 2.0,
                  min_inliers: int = 60, min_parallax: float = 4e-3):
    """Per-time camera-frame axes of the street frame: (fwd, down), each (N,3), plus diagnostics.

    fwd  = robustly smoothed visual translation direction (unit, camera frame).
    down = minus the low-passed accelerometer 'up', orthogonalised against fwd.

    The orthogonalisation makes the travel direction exactly horizontal in the virtual view.
    It removes the accelerometer's main failure mode on a bike (braking/accelerating tilts the
    specific-force vector along the travel direction) and levels to the road surface rather
    than to gravity, which is what the mosaic geometry wants. Lateral accelerations (turns)
    are not corrected; they only matter at intersections.
    """
    good = (head.n_inliers >= min_inliers) & (head.parallax >= min_parallax)
    if good.sum() < 5:
        raise RuntimeError("not enough good heading samples")
    tg, dg = head.t[good], head.tdir[good]
    dt = np.median(np.diff(tg)) if len(tg) > 1 else 0.1
    win = max(5, int(smooth_s / dt) | 1)
    ds = median_filter(dg, size=(9, 1), mode="nearest")
    if len(ds) > win:
        ds = savgol_filter(ds, win, 2, axis=0)
    fwd = np.stack([np.interp(t, tg, ds[:, i]) for i in range(3)], 1)
    fwd /= np.linalg.norm(fwd, axis=1, keepdims=True)
    up_acc = tel.up_vector(t)
    up = up_acc - (up_acc * fwd).sum(1, keepdims=True) * fwd
    up /= np.linalg.norm(up, axis=1, keepdims=True)
    down = -up
    diag = {
        # angle of travel direction right of the optical axis, in the levelled plane
        "yaw_deg": np.degrees(np.arctan2((fwd * np.cross(-up, _horiz(up))).sum(1),
                                         (fwd * _horiz(up)).sum(1))),
        # camera roll about its optical axis w.r.t. the levelled frame (+ = image needs CCW fix)
        "roll_deg": np.degrees(np.arctan2(-up[:, 0], -up[:, 1])),
        "roll_acc_deg": np.degrees(np.arctan2(-up_acc[:, 0], -up_acc[:, 1])),
        # camera pitch: + = optical axis below the horizon (tilted down; checked 2026-10-23
        # against the FOV bottom edge: GX014482 +1.9 sees 45.5 deg down, GX014484 -3.6 40.4)
        "pitch_deg": np.degrees(np.arcsin(np.clip(-up[:, 2], -1, 1))),
        # how far the accelerometer was from the vision-consistent up (deg)
        "acc_correction_deg": np.degrees(np.arccos(np.clip((up * up_acc).sum(1), -1, 1))),
    }
    return fwd, down, diag


def _horiz(up: np.ndarray) -> np.ndarray:
    """Camera optical axis projected on the plane orthogonal to `up`, unit."""
    z = np.array([0, 0, 1.0]) - up[:, 2:3] * up
    return z / np.linalg.norm(z, axis=1, keepdims=True)


def virtual_rotations(fwd: np.ndarray, down: np.ndarray, view_yaw_deg: float = 90.0,
                      view_pitch_deg: float = 0.0, side: str = "left") -> np.ndarray:
    """R_vc (N,3,3) for a virtual camera looking `view_yaw_deg` to the `side` of travel
    (90 = perpendicular), pitched `view_pitch_deg` (+ = down), level horizon.

    Virtual axes at yaw 0: z_v = fwd, y_v = down, x_v = down x fwd = right. Looking left,
    image x points forward along the street. Looking right, the view is mirrored (x_v -> -x_v,
    a reflection, fine for ray lookup) so image x still points forward and every later stage
    (flow signs, mosaics, depth) sees the same geometry as a left view of a mirrored street.
    """
    right = np.cross(down, fwd)
    base = np.stack([right, down, fwd], axis=2)  # columns: camera coords of (x_v,y_v,z_v)
    # yaw left by a: rotate virtual frame about its y (down) axis by -a
    a = view_yaw_deg if side == "left" else -view_yaw_deg
    Ry = rot_y(-np.deg2rad(a))
    Rx = rot_x(-np.deg2rad(view_pitch_deg))
    R = base @ (Ry @ Rx)
    if side == "right":
        R = R * np.array([-1.0, 1.0, 1.0])  # negate the x_v column
    return R
