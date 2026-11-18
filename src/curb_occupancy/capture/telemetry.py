"""GoPro GPMF telemetry: IMU, GPS (with fix quality), exposure/ISO, settings, HiLights.

Parsing uses Gyroflow's telemetry-parser. IMU vectors are converted to the OpenCV camera frame
(x right, y down, z forward) of the *recorded image* with the rig's `camera.imu_to_cv` matrix
(telemetry-parser normalized IMU frame -> camera). For the HERO8 (`rigs/gopro_hero8_wide_hseis`)
it is v_cv = (-y_imu, -x_imu, -z_imu), established by checking that gravity-levelled
reprojection produces vertical building edges (2026-10-22).
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt

# GPMF settings keys found in the first payload ("Default" group, raw FourCC as hex)
SETTING_NAMES = {
    "EISA": "eis_mode", "EISE": "eis_enabled", "VFOV": "lens_fov", "ZFOV": "diag_fov_deg",
    "PRTN": "protune", "PTCL": "color_profile", "PTWB": "white_balance", "PTSH": "sharpness",
    "PTEV": "ev_comp", "PIMN": "iso_min", "PIMX": "iso_max", "EXPT": "exposure_type",
    "SROT": "rolling_shutter_ms", "RATE": "photo_rate", "OREN": "orientation", "AUDO": "audio",
    "SMTR": "spot_meter", "DZOM": "digital_zoom", "FMWR": "firmware", "CASN": "camera_serial",
    "LINF": "lens_serial", "MINF": "model",
}



def _fourcc(hexkey: str) -> str:
    try:
        return int(hexkey, 16).to_bytes(4, "big").decode("latin1")
    except (ValueError, OverflowError):
        return hexkey


@dataclass
class Telemetry:
    imu_t: np.ndarray          # s
    gyro: np.ndarray           # rad/s, camera (cv) frame
    accel: np.ndarray          # m/s^2, camera (cv) frame (specific force: +up at rest)
    gps_t: np.ndarray          # s
    gps: np.ndarray            # lat, lon, alt, speed2d, speed3d
    gps_fix: np.ndarray        # per GPS sample: 0 none, 2 2D, 3 3D
    gps_dop: np.ndarray        # per GPS sample, x100
    exp_t: np.ndarray
    shutter_s: np.ndarray
    iso: np.ndarray
    settings: dict = field(default_factory=dict)
    hilights_s: list = field(default_factory=list)

    # --- derived ---------------------------------------------------------------------
    def up_vector(self, t: np.ndarray, cutoff_hz: float = 0.3) -> np.ndarray:
        """Unit 'up' direction in camera frame at times t, from low-passed accelerometer.

        The low-pass removes vibration and pedalling sway; on a bicycle the residual includes
        centripetal acceleration in turns (bike-frame 'up'), which is what we want for levelling
        within a straight street anyway.
        """
        fs = 1.0 / np.median(np.diff(self.imu_t))
        b, a = butter(2, cutoff_hz / (fs / 2))
        acc = filtfilt(b, a, self.accel, axis=0)
        up = np.stack([np.interp(t, self.imu_t, acc[:, i]) for i in range(3)], 1)
        return up / np.linalg.norm(up, axis=1, keepdims=True)

    def summary(self) -> dict:
        good = self.gps_fix >= 2
        return {
            "imu_rate_hz": float(1 / np.median(np.diff(self.imu_t))) if len(self.imu_t) > 1 else 0,
            "gps_samples": int(len(self.gps_t)),
            "gps_fix_fraction": float(good.mean()) if len(good) else 0.0,
            "gps_min_dop": float(self.gps_dop.min() / 100) if len(self.gps_dop) else None,
            "shutter_1_over": sorted({int(round(1 / s)) for s in self.shutter_s if s > 0}),
            "iso_range": [float(self.iso.min()), float(self.iso.max())] if len(self.iso) else None,
            "hilights_s": self.hilights_s,
            "settings": self.settings,
        }

    def cut(self, t0: float, t1: float) -> "Telemetry":
        """The part in [t0, t1] (s), every stream shifted so that t0 becomes 0 (a clip window)."""
        import copy
        out = copy.copy(self)
        for tk, keys in (("imu_t", ("gyro", "accel")), ("gps_t", ("gps", "gps_fix", "gps_dop")),
                         ("exp_t", ("shutter_s", "iso"))):
            t = getattr(self, tk)
            m = (t >= t0) & (t <= t1)
            setattr(out, tk, t[m] - t0)
            for k in keys:
                setattr(out, k, getattr(self, k)[m])
        out.hilights_s = [h - t0 for h in self.hilights_s if t0 <= h <= t1]
        return out

    @classmethod
    def concat(cls, parts: list["Telemetry"], offsets_s: list[float]) -> "Telemetry":
        """Chapters of one recording: each part's times shifted by its chapter's start.

        GoPro chapters mix two clocks: telemetry-parser's IMU restarts at 0 in every file, but
        the GPMF GPS / exposure timestamps run on the recording's clock (GX024503 starts at
        531.6 s, its chapter offset 531.5 s). A stream that already starts at or past half its
        chapter's offset is taken as being on the recording clock and is not shifted."""
        import copy
        out = copy.copy(parts[0])

        def shifted(t, o):
            return t if (o > 0 and len(t) and t[0] >= 0.5 * o) else t + o

        for tk, keys in (("imu_t", ("gyro", "accel")), ("gps_t", ("gps", "gps_fix", "gps_dop")),
                         ("exp_t", ("shutter_s", "iso"))):
            setattr(out, tk, np.concatenate([shifted(getattr(p, tk), o)
                                             for p, o in zip(parts, offsets_s)]))
            for k in keys:
                setattr(out, k, np.concatenate([getattr(p, k) for p in parts]))
        out.hilights_s = [h + o for p, o in zip(parts, offsets_s) for h in p.hilights_s]
        return out

    def save(self, path: str | Path):
        path = Path(path)
        np.savez_compressed(path, **{k: v for k, v in self.__dict__.items()
                                     if isinstance(v, np.ndarray)})
        path.with_suffix(".json").write_text(json.dumps(self.summary(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Telemetry":
        path = Path(path)
        z = np.load(path)
        meta = json.loads(path.with_suffix(".json").read_text())
        return cls(**{k: z[k] for k in z.files}, settings=meta.get("settings", {}),
                   hilights_s=meta.get("hilights_s", []))


def gps_utc0(files: list) -> float | None:
    """UTC (epoch s) of recording time 0, from the GPS clock: median over 3D-fix packets of
    GPSU - packet time. GPSU and the GPS packet times run on the recording's clock in every
    chapter, so the first chapter with a fix gives it. None without a 3D fix. (The GPS lags the
    video ~1 s: irrelevant for a wall-clock start time.)"""
    import telemetry_parser
    for f in files:
        d = []
        for g in telemetry_parser.Parser(str(f)).telemetry():
            G = g.get("GPS") or {}
            u = G.get("0x47505355")   # GPSU: UTC in ms since epoch (telemetry-parser)
            if G.get("Data") and int(G.get("0x47505346", 0)) >= 3 and isinstance(u, (int, float)):
                d.append(u / 1000 - G["TimestampUs"] / 1e6)
        if d:
            return float(np.median(d))
    return None


def local_start(utc0: float, t_s: float, tz: str) -> str:
    """Wall-clock local time (ISO, no zone suffix) of recording time t_s."""
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo
    t = datetime.fromtimestamp(utc0 + t_s, timezone.utc).astimezone(ZoneInfo(tz))
    return t.replace(tzinfo=None).isoformat(timespec="seconds")


def read_hilights(mp4: str | Path) -> list[float]:
    """HiLight tags (seconds) from the moov/udta/HMMT atom."""
    def walk(f, start, end):
        pos = start
        while pos < end - 8:
            f.seek(pos)
            sz, typ = struct.unpack(">I4s", f.read(8))
            hdr = 8
            if sz == 1:
                sz, hdr = struct.unpack(">Q", f.read(8))[0], 16
            if sz == 0:
                sz = end - pos
            if typ in (b"moov", b"udta"):
                r = walk(f, pos + hdr, pos + sz)
                if r is not None:
                    return r
            elif typ == b"HMMT":
                f.seek(pos + hdr)
                n = struct.unpack(">I", f.read(4))[0]
                return [x / 1000 for x in struct.unpack(f">{n}I", f.read(4 * n))] if n else []
            pos += sz
        return None

    with open(mp4, "rb") as f:
        f.seek(0, 2)
        return walk(f, 0, f.tell()) or []


def extract_source(src, imu_to_cv=None) -> Telemetry:
    """Telemetry of a clip (`video.Source`): every chapter parsed, joined, cut to the window
    (times relative to the window start, like the clip's frames)."""
    parts = [extract(f, imu_to_cv) for f in src.files]
    tel = Telemetry.concat(parts, [o / src.fps for o in src.offsets]) if len(parts) > 1 \
        else parts[0]
    return tel.cut(src.i0 / src.fps, src.i1 / src.fps)


def extract(mp4: str | Path, imu_to_cv=None) -> Telemetry:
    """`imu_to_cv` (3x3, the rig's `camera.imu_to_cv`): telemetry-parser's normalized IMU axes
    -> OpenCV camera frame. None keeps the normalized axes (GPS-only uses, e.g. `ride`)."""
    import telemetry_parser

    M = np.eye(3) if imu_to_cv is None else np.asarray(imu_to_cv, dtype=np.float64)
    p = telemetry_parser.Parser(str(mp4))
    imu = p.normalized_imu()
    imu_t = np.array([d["timestamp_ms"] for d in imu]) / 1000
    gyro = np.deg2rad(np.array([d["gyro"] for d in imu], dtype=np.float64)) @ M.T
    accel = np.array([d["accl"] for d in imu], dtype=np.float64) @ M.T

    groups = p.telemetry()
    settings = {}
    for k, v in groups[0].get("Default", {}).items():
        cc = _fourcc(k) if k.startswith("0x") else k
        settings[SETTING_NAMES.get(cc, cc)] = v

    gps_t, gps, fix, dop, exp_t, shut, iso = [], [], [], [], [], [], []
    ts = [g.get("GPS", {}).get("TimestampUs") for g in groups]
    for i, g in enumerate(groups):
        G = g.get("GPS")
        if G and G.get("Data"):
            d = np.array(G["Data"], dtype=np.float64) / np.array(G["Scale"], dtype=np.float64)
            t0 = G["TimestampUs"] / 1e6
            nxt = next((t for t in ts[i + 1:] if t is not None), None)
            t1 = nxt / 1e6 if nxt else t0 + 1.0
            gps_t.extend(t0 + (t1 - t0) * np.arange(len(d)) / len(d))
            gps.extend(d)
            fix.extend([int(G.get("0x47505346", 0))] * len(d))   # GPSF
            dop.extend([int(G.get("0x47505350", 9999))] * len(d))  # GPSP
        E = g.get("Exposure")
        if E and E.get("Data") is not None:
            # a payload with a single sample comes back as a scalar (seen on GX014492)
            e_d = np.atleast_1d(np.asarray(E["Data"], dtype=np.float64))
            n = len(e_d)
            t0 = E["TimestampUs"] / 1e6
            exp_t.extend(t0 + np.arange(n) / n)
            shut.extend(e_d)
            iso_d = np.atleast_1d(np.asarray(g.get("SensorISO", {}).get("Data", []),
                                             dtype=np.float64))
            iso.extend(np.interp(np.arange(n), np.linspace(0, n - 1, len(iso_d)), iso_d)
                       if len(iso_d) else [np.nan] * n)
    return Telemetry(
        imu_t=imu_t, gyro=gyro, accel=accel,
        gps_t=np.array(gps_t), gps=np.array(gps).reshape(-1, 5), gps_fix=np.array(fix),
        gps_dop=np.array(dop), exp_t=np.array(exp_t), shutter_s=np.array(shut),
        iso=np.array(iso), settings=settings, hilights_s=read_hilights(mp4),
    )


def gps_scale_check(tel: Telemetry, trace, fps: float, h: float, min_speed: float = 2.0,
                    max_dop: float = 10.0, smooth_s: float = 2.0) -> dict:
    """Camera height implied by GPS: visual odometry measures speed in camera-height units
    (s = v*dt/h), so h_gps = v_gps / (s * fps). Both signals are box-smoothed over `smooth_s`
    (GPS speed is Doppler-based, noisy per sample but unbiased); only samples with a 3D fix,
    DOP <= max_dop and v_gps >= min_speed m/s count. Also compares path lengths. The DOP gate
    is loose: evening passes (2026-10-23, DOP ~6) give the same h as a DOP ~2.7 morning pass
    on the same mount (0.926 vs 0.925, same IQR); `gps_scale_ok` checks the spread."""
    good = (tel.gps_fix >= 3) & (tel.gps_dop / 100 <= max_dop) if len(tel.gps_t) else []
    out = {"gps_fix_fraction": float(np.mean(tel.gps_fix >= 2)) if len(tel.gps_fix) else 0.0,
           "h_assumed_m": h, "visual_path_m": float(trace.X_m.iloc[-1])}
    if not np.any(good) or np.mean(good) < 0.5:
        return out
    gt, v = tel.gps_t[good], tel.gps[good, 3]
    k = max(1, int(round(smooth_s * len(gt) / max(gt[-1] - gt[0], 1e-6))))
    v_s = np.convolve(v, np.ones(k) / k, mode="same")
    w = max(1, int(round(smooth_s * fps)))
    s_sm = np.convolve(trace.s_h.values, np.ones(w) / w, mode="same") * fps
    s_vis = np.interp(gt, trace.t.values, s_sm)                       # h units / s
    m = (v_s >= min_speed) & (s_vis > 0.1)
    r = v_s[m] / s_vis[m]
    lat, lon = np.radians(tel.gps[good, 0]), np.radians(tel.gps[good, 1])
    R = 6371000.0
    xy = np.stack([(lon - lon[0]) * R * np.cos(lat[0]), (lat - lat[0]) * R], 1)
    # GPS positions jitter at ~1 m: path length over 1-s decimated samples
    step = max(1, int(round(len(gt) / max(gt[-1] - gt[0], 1e-6))))
    path = float(np.linalg.norm(np.diff(xy[::step], axis=0), axis=1).sum())
    q = np.percentile(r, [25, 50, 75]) if len(r) else [np.nan] * 3
    out.update({"h_gps_m": round(float(q[1]), 4),
                "h_gps_iqr_m": [round(float(q[0]), 3), round(float(q[2]), 3)],
                "n_samples": int(m.sum()), "gps_path_m": path,
                "gps_speed_median_kmh": float(np.median(v_s[m]) * 3.6) if m.any() else None,
                "start_latlon": [float(tel.gps[good, 0][0]), float(tel.gps[good, 1][0])],
                "end_latlon": [float(tel.gps[good, 0][-1]), float(tel.gps[good, 1][-1])]})
    return out


def gps_speed(tel: Telemetry, t: np.ndarray, vis_speed: np.ndarray | None = None,
              lag_s: float = 1.08, smooth_s: float = 1.0, max_lag_s: float = 2.5):
    """GPS (Doppler, 2D) speed in m/s at video times t, NaN where there is no 3D fix nearby.
    GPS lags the video: sample stamped t' = video time t' - lag. With `vis_speed` (visual
    speed at t, any scale), the lag is the one maximising their correlation over moving
    frames (the default `lag_s` when the peak is weak). Returns (speed, lag)."""
    from scipy.ndimage import uniform_filter1d
    g = tel.gps_fix >= 3
    if g.sum() < 20:
        return np.full(len(t), np.nan), lag_s
    gt, v = tel.gps_t[g], tel.gps[g, 3]
    k = max(1, int(round(smooth_s * len(gt) / max(gt[-1] - gt[0], 1e-6))))
    v = uniform_filter1d(v, k, mode="nearest")
    lag = lag_s
    if vis_speed is not None:
        best = (0.0, lag_s)
        for L in np.arange(-0.5, max_lag_s + 1e-9, 0.05):
            vg = np.interp(t, gt - L, v)
            m = np.isfinite(vis_speed) & (vg > 1.0)
            if m.sum() > 300:
                c = np.corrcoef(vg[m], vis_speed[m])[0, 1]
                best = max(best, (c, L))
        if best[0] >= 0.8:
            lag = float(best[1])
    out = np.interp(t, gt - lag, v)
    near = np.interp(t, gt - lag, np.ones(len(gt)), left=0, right=0) > 0  # inside the fix span
    return np.where(near, out, np.nan), lag


def gps_lag(tel: Telemetry, trace, max_lag_s: float = 4.0, step_s: float = 0.05) -> float:
    """Latency (s) of GPS positions w.r.t. the video: GPS sample stamped t was the position
    at video time t - lag. Path length along the (smoothed) GPS track vs visual odometry
    distance, scale-matched; median of the time differences at equal distance (the first
    and last 50 m excluded). ~1.06 s on GX014482 (HERO8)."""
    from scipy.ndimage import uniform_filter1d
    g = tel.gps_fix >= 3
    if g.sum() < 100:
        return 0.0
    t = tel.gps_t[g]
    lat, lon = np.radians(tel.gps[g, 0]), np.radians(tel.gps[g, 1])
    R = 6371000.0
    xy = np.stack([(lon - lon[0]) * R * np.cos(lat[0]), (lat - lat[0]) * R], 1)
    xy = uniform_filter1d(xy, size=20, axis=0, mode="nearest")
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    X = trace.X_m.values
    s = s * X[-1] / max(s[-1], 1e-6)
    tv = np.interp(s, X, trace.t.values)
    m = (s > 50) & (s < s[-1] - 50)
    lag = float(np.median((t - tv)[m])) if m.any() else 0.0
    return float(np.clip(lag, -max_lag_s, max_lag_s))
