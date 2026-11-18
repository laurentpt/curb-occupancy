"""Pushbroom (strip) mosaic of the parked row from a gravity-levelled perpendicular virtual view.

Geometry of the virtual view (pinhole, focal f, principal point (cx, cy), x_v = travel
direction, z_v = left, y_v = down; camera height h above the road):

* a static point at lateral depth Z moves by  dx = -f * v*dt / Z  between frames;
* the road at row y (> cy) is at depth Z = f*h/(y-cy), so its flow is linear in the row:
      dx_ground(y) = -(v*dt/h) * (y - cy)            -> slope s = v*dt/h  [1/frame]
* the parked row (depth D) is a plateau  dx_park = -f*v*dt/D = -s * f*h/D.

So strip widths at the parked depth follow from the road slope times a per-pass constant
k = f*h/D, which we calibrate from frames where both the road and parked cars are visible.
That keeps the horizontal scale uniform (metric up to h) through gaps, driveways and
intersections, where the brief's 'band flow' would collapse to the far background.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from scipy.ndimage import median_filter

from .capture.lens import FisheyeLens, PinholeView, Reprojector
from .capture.video import iter_frames, probe


@dataclass
class FlowProfiles:
    t: np.ndarray        # (N,) time of frame i (flow from frame i-1 to i)
    u: np.ndarray        # (N,H) per-row median horizontal flow, px/frame in the small view
    v: np.ndarray        # (N,H) per-row median vertical flow
    spread: np.ndarray   # (N,H) per-row IQR of u (texture/consistency)
    f: float             # focal length of the small view
    cy: float

    def save(self, path):
        np.savez_compressed(path, **{k: getattr(self, k) for k in self.__dataclass_fields__})

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(**{k: (z[k] if z[k].ndim else z[k].item()) for k in z.files})


def flow_profiles(video: str, lens: FisheyeLens, R_vc: np.ndarray, view: PinholeView,
                  decode_scale: float = 0.5, col_frac: float = 0.5, device: str = "cuda",
                  batch: int = 32, progress=None) -> FlowProfiles:
    """Dense DIS flow between consecutive virtual views; per-row robust statistics over the
    central `col_frac` of the columns (the region the strip is cut from)."""
    info = probe(video)
    w, h = int(info.width * decode_scale), int(info.height * decode_scale)
    rp = Reprojector(lens.scaled(w, h), view, device)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    c0 = int(view.width * (0.5 - col_frac / 2))
    c1 = int(view.width * (0.5 + col_frac / 2))
    N = min(len(R_vc), info.n_frames)
    U = np.full((N, view.height), np.nan, np.float32)
    Vv = np.full_like(U, np.nan)
    S = np.full_like(U, np.nan)
    prev = None
    buf_i, buf_f = [], []

    def flush():
        nonlocal prev
        x = torch.from_numpy(np.stack(buf_f)).to(device)[:, None].float()
        out = rp(x, R_vc[buf_i]).clamp(0, 255).byte()[:, 0].cpu().numpy()
        valid = rp(torch.ones_like(x[:1]), R_vc[buf_i[:1]])[0, 0].cpu().numpy() > 0.99
        for i, img in zip(buf_i, out):
            if prev is not None:
                fl = torch.from_numpy(dis.calc(prev, img, None)[:, c0:c1]).to(device)
                vm = torch.from_numpy(valid[:, c0:c1]).to(device)
                fu = torch.where(vm, fl[..., 0], torch.nan)
                fv = torch.where(vm, fl[..., 1], torch.nan)
                q = torch.nanquantile(fu, torch.tensor([0.25, 0.5, 0.75], device=device), dim=1)
                U[i], S[i] = q[1].cpu().numpy(), (q[2] - q[0]).cpu().numpy()
                Vv[i] = torch.nanmedian(fv, dim=1).values.cpu().numpy()
            prev = img
        buf_i.clear()
        buf_f.clear()

    for idx, img in iter_frames(video, w, h, gray=True):
        if idx >= N:
            break
        buf_i.append(idx)
        buf_f.append(img)
        if len(buf_i) == batch:
            flush()
            if progress:
                progress(idx, N)
    if buf_i:
        flush()
    return FlowProfiles(t=np.arange(N) / info.fps, u=U, v=Vv, spread=S, f=view.f, cy=view.cy)


# ----------------------------------------------------------------------------------------
# speed and depth from the profiles

def ground_slope(fp: FlowProfiles, rows=(0.55, 0.8), pct: float = 30, smooth: int = 31,
                 close: int = 31, open_: int = 181, spike_window: int = 121,
                 spike_ratio: float = 1.25):
    """Per-frame road-flow slope s = v*dt/h (camera-height units per frame).

    Uses rows whose road depth is ~1.25-1.8 h. A low percentile across rows rejects rows
    covered by nearer objects (which only ever *raise* the ratio). Then a temporal median.

    When an object covers *all* those rows (cars parked ~1.1 m away, a bus in the travel lane)
    the estimate jumps up for as long as it takes to pass it (seen: 22 -> 46 km/h for 1.7 s,
    stretching both cars 1.6x in the mosaic). Errors are one-sided, so a grey opening
    (window `open_` frames, ~3 s at 60 fps) removes upward excursions shorter than the window
    while keeping ramps and valleys; a short closing first stops brief downward glitches
    (untextured road) from being widened by the opening.
    """
    from scipy.ndimage import grey_closing, grey_opening

    y = np.arange(fp.u.shape[1])
    dy = y - fp.cy
    sel = (dy > rows[0] * fp.f) & (dy < rows[1] * fp.f)
    with np.errstate(all="ignore"):
        r = np.nanpercentile(-fp.u[:, sel] / dy[sel], pct, axis=1)
    fill = np.nanmedian(r)
    r = np.where(np.isfinite(r), r, fill)
    r = median_filter(r, size=smooth, mode="nearest")
    r = grey_opening(grey_closing(r, size=close, mode="nearest"), size=open_, mode="nearest")
    # Large near objects (buses, vans) can cover all road rows for ~1 s and read as a speed
    # spike; a bicycle cannot accelerate like that, so clamp to a slow rolling median.
    slow = median_filter(r, size=int(spike_window) | 1, mode="nearest")
    r = np.minimum(r, spike_ratio * slow)
    return np.clip(r, 0, None)


def horizon_depth(fp: FlowProfiles, s: np.ndarray, band: float = 0.08) -> np.ndarray:
    """Depth (camera-height units) of whatever is at the camera-height row, per frame."""
    y = np.arange(fp.u.shape[1])
    sel = np.abs(y - fp.cy) < band * fp.f
    with np.errstate(all="ignore"):
        uh = np.nanmedian(-fp.u[:, sel], axis=1)
    return fp.f * s / np.maximum(uh, 1e-3)


def parked_depth(Z: np.ndarray, lo: float, hi: float) -> float:
    """Mode of log-depth within [lo, hi] (the parked-row plane for this pass)."""
    z = Z[np.isfinite(Z) & (Z > lo) & (Z < hi)]
    if len(z) < 20:
        return float(np.sqrt(lo * hi))
    hist, e = np.histogram(np.log(z), bins=30)
    k = np.convolve(hist, np.ones(3) / 3, mode="same").argmax()
    return float(np.exp((e[k] + e[k + 1]) / 2))


# ----------------------------------------------------------------------------------------
# mosaic

@dataclass
class Mosaic:
    image: np.ndarray      # (rows, cols, 3) uint8, RGB
    px_per_m: float        # horizontal and vertical scale at the parked plane (h units)
    depth: float           # parked-plane lateral distance (h units)
    y_top: float           # height above road of row 0 (h units)
    x0: float              # along-street position of column 0 (h units, from clip start)
    col_frame: np.ndarray  # (cols,) source frame index per column
    t0: float
    t1: float

    def row_of_height(self, Y):
        return (self.y_top - np.asarray(Y)) * self.px_per_m

    def save(self, stem):
        import json
        from pathlib import Path

        stem = Path(stem)
        cv2.imwrite(str(stem.with_suffix(".png")), cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR))
        np.save(stem.with_name(stem.name + "_colframe.npy"), self.col_frame)
        meta = {k: getattr(self, k) for k in ("px_per_m", "depth", "y_top", "x0", "t0", "t1")}
        meta["shape"] = list(self.image.shape)
        stem.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, stem):
        import json
        from pathlib import Path

        stem = Path(stem)
        img = cv2.cvtColor(cv2.imread(str(stem.with_suffix(".png"))), cv2.COLOR_BGR2RGB)
        meta = json.loads(stem.with_suffix(".json").read_text())
        meta.pop("shape", None)
        return cls(image=img, col_frame=np.load(stem.with_name(stem.name + "_colframe.npy")),
                   **meta)


def build_mosaic(video: str, lens: FisheyeLens, R_vc: np.ndarray, s: np.ndarray, depth: float,
                 t0: float, t1: float, px_per_m: float = 100.0, y_range=(-0.6, 2.2),
                 h: float = 1.0, decode_scale: float = 0.5, supersample: int = 2,
                 rs_readout_s: float = 0.011, device: str = "cuda", batch: int = 64,
                 progress=None) -> Mosaic:
    """Orthographic sampling of the vertical plane at lateral distance `depth` (h units).

    For mosaic column at along-street position X and height Y (road = 0), the point in the
    virtual frame of camera k (x fwd, y down, z left) is (X - X_k, h - Y, depth); its ray is
    rotated into the physical camera and projected with the fisheye model. Each frame
    contributes the columns between the midpoints to its neighbours (pushbroom), so static
    objects at `depth` keep their true aspect and stops add nothing.

    Rolling shutter: a sensor row v is exposed at (v/H - 0.5)*readout relative to the frame
    time; the camera position is advanced accordingly (one fixed-point iteration).
    `y_range` = (Ymin, Ymax): heights above the road (h units) covered by the mosaic rows.
    """
    info = probe(video)
    fps = info.fps
    w, hh = int(info.width * decode_scale), int(info.height * decode_scale)
    L = lens.scaled(w, hh)
    P = px_per_m * supersample
    Ymin, Ymax = y_range
    rows = int(round((Ymax - Ymin) * P))
    Yrow = Ymax - (np.arange(rows) + 0.5) / P

    i0, i1 = int(np.floor(t0 * fps)), min(int(np.ceil(t1 * fps)), len(s), len(R_vc))
    X = np.concatenate([[0.0], np.cumsum(s[1:])]) * h  # camera position per frame
    vel = s * fps * h                                     # h-units per second
    Xa = X[i0]
    mids = np.concatenate([[X[i0]], (X[i0:i1 - 1] + X[i0 + 1:i1]) / 2, [X[i1 - 1]]])
    total = int(np.ceil((X[i1 - 1] - Xa) * P)) + 1
    out = np.zeros((rows, total, 3), np.uint8)
    col_frame = np.full(total, -1, np.int32)

    Yt = torch.as_tensor(Yrow, dtype=torch.float32, device=device)
    pending = []

    def render(items):
        frames = torch.from_numpy(np.stack([f for _, f, _ in items])).to(device)
        frames = frames.permute(0, 3, 1, 2).float()
        for b, (k, _, cols) in enumerate(items):
            Xc = torch.as_tensor(Xa + (cols + 0.5) / P, dtype=torch.float32, device=device)
            R = torch.as_tensor(R_vc[k], dtype=torch.float32, device=device)
            pos = torch.tensor(X[k], dtype=torch.float32, device=device)
            uv = None
            for it in range(2 if rs_readout_s else 1):
                if uv is None:
                    dxk = torch.zeros(rows, len(cols), device=device)
                else:
                    dtr = (uv[..., 1] / (hh - 1) - 0.5) * rs_readout_s
                    dxk = dtr * float(vel[k])
                px = Xc[None, :] - pos - dxk
                py = (h - Yt)[:, None].expand_as(px)
                pz = torch.full_like(px, depth)
                ray = torch.stack([px, py, pz], -1) @ R.T
                uv = L.project_torch(ray)
            g = torch.stack([uv[..., 0] / (w - 1) * 2 - 1, uv[..., 1] / (hh - 1) * 2 - 1], -1)
            strip = torch.nn.functional.grid_sample(frames[b:b + 1], g[None], mode="bilinear",
                                                    padding_mode="zeros", align_corners=True)
            out[:, cols] = strip[0].permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()
            col_frame[cols] = k

    for idx, img in iter_frames(video, w, hh, start_s=i0 / fps, duration_s=(i1 - i0) / fps):
        k = idx
        if k >= i1:
            break
        j = k - i0
        c0 = int(np.ceil((mids[j] - Xa) * P))
        c1 = int(np.ceil((mids[j + 1] - Xa) * P))
        if c1 > c0:
            pending.append((k, img, np.arange(c0, min(c1, total))))
        if len(pending) >= batch:
            render(pending)
            pending = []
            if progress:
                progress(k, i1)
    if pending:
        render(pending)
    if supersample > 1:
        out = cv2.resize(out, (total // supersample, rows // supersample),
                         interpolation=cv2.INTER_AREA)
        col_frame = col_frame[::supersample][: out.shape[1]]
    return Mosaic(image=out, px_per_m=px_per_m, depth=depth, y_top=Ymax, x0=Xa,
                  col_frame=col_frame, t0=t0, t1=t1)
