"""Fisheye lens model (OpenCV fisheye / Kannala-Brandt, as used by Gyroflow profiles) and
virtual pinhole cameras.

Conventions: OpenCV camera frame (x right, y down, z forward). A *virtual* camera is a pinhole
camera sharing the optical centre of the physical one, rotated by R_vc (maps virtual-camera
rays to physical-camera rays: ray_cam = R_vc @ ray_virt).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class FisheyeLens:
    K: np.ndarray  # 3x3, at the working resolution
    D: np.ndarray  # k1..k4
    width: int
    height: int
    name: str = ""

    @classmethod
    def from_gyroflow(cls, path: str | Path, width: int, height: int) -> "FisheyeLens":
        prof = json.loads(Path(path).read_text())
        fp = prof["fisheye_params"]
        cw, ch = prof["calib_dimension"]["w"], prof["calib_dimension"]["h"]
        sx, sy = width / cw, height / ch
        if abs(sx - sy) > 1e-3:
            raise ValueError(f"aspect mismatch: profile {cw}x{ch} vs video {width}x{height}")
        K = np.array(fp["camera_matrix"], dtype=np.float64)
        K[0] *= sx
        K[1] *= sy
        return cls(K=K, D=np.array(fp["distortion_coeffs"], dtype=np.float64), width=width,
                   height=height, name=prof.get("name", ""))

    def scaled(self, width: int, height: int) -> "FisheyeLens":
        K = self.K.copy()
        K[0] *= width / self.width
        K[1] *= height / self.height
        return FisheyeLens(K=K, D=self.D.copy(), width=width, height=height, name=self.name)

    # --- numpy -------------------------------------------------------------------------
    def project(self, rays: np.ndarray) -> np.ndarray:
        """rays (...,3) in camera frame -> pixel coords (...,2). Rays behind camera -> nan."""
        x, y, z = rays[..., 0], rays[..., 1], rays[..., 2]
        r = np.hypot(x, y)
        th = np.arctan2(r, z)
        k1, k2, k3, k4 = self.D
        th2 = th * th
        thd = th * (1 + th2 * (k1 + th2 * (k2 + th2 * (k3 + th2 * k4))))
        scale = np.where(r > 1e-12, thd / np.maximum(r, 1e-12), 1.0)
        u = self.K[0, 0] * x * scale + self.K[0, 2]
        v = self.K[1, 1] * y * scale + self.K[1, 2]
        out = np.stack([u, v], -1)
        out[th > np.deg2rad(100)] = np.nan
        return out

    def unproject(self, uv: np.ndarray) -> np.ndarray:
        """pixels (...,2) -> unit rays (...,3)."""
        import cv2

        pts = np.asarray(uv, dtype=np.float64).reshape(-1, 1, 2)
        und = cv2.fisheye.undistortPoints(pts, self.K, self.D).reshape(-1, 2)
        rays = np.concatenate([und, np.ones((len(und), 1))], 1)
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        return rays.reshape(*np.shape(uv)[:-1], 3)

    # --- torch (GPU remap) ---------------------------------------------------------------
    def project_torch(self, rays: torch.Tensor) -> torch.Tensor:
        x, y, z = rays[..., 0], rays[..., 1], rays[..., 2]
        r = torch.sqrt(x * x + y * y).clamp_min(1e-12)
        th = torch.atan2(r, z)
        k1, k2, k3, k4 = [float(k) for k in self.D]
        th2 = th * th
        thd = th * (1 + th2 * (k1 + th2 * (k2 + th2 * (k3 + th2 * k4))))
        s = thd / r
        u = self.K[0, 0] * x * s + self.K[0, 2]
        v = self.K[1, 1] * y * s + self.K[1, 2]
        bad = th > np.deg2rad(100)
        u = torch.where(bad, torch.full_like(u, -1e4), u)
        return torch.stack([u, v], -1)


@dataclass
class PinholeView:
    """Virtual pinhole camera intrinsics."""
    width: int
    height: int
    f: float
    cx: float
    cy: float

    @classmethod
    def from_hfov(cls, width: int, height: int, hfov_deg: float, cy_frac: float = 0.5):
        f = (width / 2) / np.tan(np.deg2rad(hfov_deg) / 2)
        return cls(width, height, f, (width - 1) / 2, (height - 1) * cy_frac)

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.f, 0, self.cx], [0, self.f, self.cy], [0, 0, 1.0]])

    def rays_torch(self, device) -> torch.Tensor:
        ys, xs = torch.meshgrid(torch.arange(self.height, device=device, dtype=torch.float32),
                                torch.arange(self.width, device=device, dtype=torch.float32),
                                indexing="ij")
        return torch.stack([(xs - self.cx) / self.f, (ys - self.cy) / self.f,
                            torch.ones_like(xs)], -1)


class Reprojector:
    """GPU fisheye -> virtual pinhole resampling for batches of frames with per-frame rotations."""

    def __init__(self, lens: FisheyeLens, view: PinholeView, device: str = "cuda"):
        self.lens, self.view, self.device = lens, view, device
        self.rays = view.rays_torch(device)  # H,W,3

    def grids(self, R_vc: np.ndarray) -> torch.Tensor:
        """R_vc: (B,3,3) virtual->camera rotations. Returns normalized grids (B,H,W,2)."""
        R = torch.as_tensor(R_vc, dtype=torch.float32, device=self.device)
        rays = torch.einsum("bij,hwj->bhwi", R, self.rays)
        uv = self.lens.project_torch(rays)
        gx = uv[..., 0] / (self.lens.width - 1) * 2 - 1
        gy = uv[..., 1] / (self.lens.height - 1) * 2 - 1
        return torch.stack([gx, gy], -1)

    def __call__(self, frames: torch.Tensor, R_vc: np.ndarray) -> torch.Tensor:
        """frames: (B,C,H,W) float on device (physical images). Returns (B,C,h,w)."""
        g = self.grids(R_vc)
        return F.grid_sample(frames, g, mode="bilinear", padding_mode="zeros", align_corners=True)


def rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
