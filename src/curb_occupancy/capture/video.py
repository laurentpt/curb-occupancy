"""Video probing and frame streaming through ffmpeg (NVDEC when available).

A clip is a `Source`: a window of a recording that may span several chapter files (the HERO8
splits a recording at 4 GB, ~12 min at 1440p60, into GX01nnnn, GX02nnnn, ... with no gap).
Its frame 0 / time 0 is the window start; `probe`, `iter_frames` and `grab` accept a Source
wherever they take a path."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from fractions import Fraction
from pathlib import Path
from typing import Iterator

import numpy as np


def _bin(name: str) -> str:
    # prefer the binary shipped in the same env as the interpreter (works without activation)
    local = Path(sys.executable).parent / name
    p = str(local) if local.exists() else shutil.which(name)
    if p is None:
        raise RuntimeError(f"{name} not found on PATH (activate the curb-occupancy env)")
    return p


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    n_frames: int
    duration: float
    codec: str
    creation_time: str
    firmware: str

    def to_dict(self):
        return self.__dict__.copy()


def probe(path) -> VideoInfo:
    if isinstance(path, Source):
        return path.info
    out = subprocess.run([_bin("ffprobe"), "-v", "error", "-print_format", "json", "-show_format",
                          "-show_streams", str(path)], capture_output=True, check=True, text=True)
    d = json.loads(out.stdout)
    v = next(s for s in d["streams"] if s["codec_type"] == "video")
    fmt = d["format"]
    return VideoInfo(
        path=str(path), width=int(v["width"]), height=int(v["height"]),
        fps=float(Fraction(v["r_frame_rate"])), n_frames=int(v.get("nb_frames", 0)),
        duration=float(v["duration"]), codec=v["codec_name"],
        creation_time=fmt.get("tags", {}).get("creation_time", ""),
        firmware=fmt.get("tags", {}).get("firmware", ""),
    )


def iter_frames(path, width: int | None = None, height: int | None = None,
                gray: bool = False, start_s: float = 0.0, duration_s: float | None = None,
                step: int = 1, hwaccel: bool = True) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_index, image) with image HxW (gray) or HxWx3 (RGB, uint8).

    frame_index counts frames of the source stream (so step>1 skips indices). Seeking with
    -ss before -i is frame-accurate for re-encoded output in modern ffmpeg; the first frame is
    round(start_s * fps) (the seek goes half a frame earlier: at exactly k/fps, printed
    rounding gave frame k+1 half of the time).
    """
    if isinstance(path, Source):
        yield from path.iter_frames(width, height, gray, start_s, duration_s, step, hwaccel)
        return
    yield from _iter_file(path, width, height, gray, start_s, duration_s, step, hwaccel)


def _iter_file(path, width=None, height=None, gray=False, start_s=0.0, duration_s=None, step=1,
               hwaccel=True, n_max: int | None = None, phase: int = 0):
    """`iter_frames` of one file; `n_max` caps the decoded frames, `phase` shifts the
    step selection (frame n is kept when (n + phase) % step == 0; indices still count from
    the first frame)."""
    info = probe(path)
    w, h = width or info.width, height or info.height
    vf = []
    if step > 1:
        vf.append(f"select='not(mod(n+{phase}\\,{step}))'")
    if (w, h) != (info.width, info.height):
        vf.append(f"scale={w}:{h}:flags=area")
    cmd = [_bin("ffmpeg"), "-hide_banner", "-loglevel", "error", "-nostdin"]
    if hwaccel:
        cmd += ["-hwaccel", "cuda"]
    first = int(round(start_s * info.fps))
    if first > 0:
        cmd += ["-ss", f"{(first - 0.5) / info.fps:.6f}"]
    cmd += ["-i", str(path)]
    if n_max is not None:
        cmd += ["-frames:v", str(int(n_max))]
    elif duration_s is not None:
        cmd += ["-t", f"{duration_s:.4f}"]
    if vf:
        cmd += ["-vf", ",".join(vf)]
    cmd += ["-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "gray" if gray else "rgb24", "-"]
    nbytes = w * h * (1 if gray else 3)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=nbytes * 4)
    try:
        k = 0
        while True:
            buf = proc.stdout.read(nbytes)
            if len(buf) < nbytes:
                break
            img = np.frombuffer(buf, np.uint8).reshape((h, w) if gray else (h, w, 3))
            yield first + (-phase) % step + k * step, img
            k += 1
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


def grab(path, t_s: float, width: int | None = None, height: int | None = None) -> np.ndarray:
    """Single RGB frame at time t_s."""
    if isinstance(path, Source):
        path, t_s = path.locate(t_s)
    for _, img in iter_frames(path, width, height, start_s=t_s, duration_s=0.05, hwaccel=False):
        return img.copy()
    raise ValueError(f"no frame at {t_s}s")


@dataclass
class Source:
    """Frames [i0, i1) of the concatenation of `files` (chapters of one recording, in order).
    Times and frame indices are relative to frame i0. Chapters are joined by frame count
    (constant frame rate), so chapter k starts at the sum of the earlier chapters' frames."""
    files: list
    i0: int
    i1: int
    chapters: list  # VideoInfo per file

    @classmethod
    def open(cls, files, t_range=(0.0, None)) -> "Source":
        """`files`: one path or a list of chapter paths; `t_range` = [t0, t1] s in the
        recording (t1 None = to the end)."""
        files = [str(f) for f in ([files] if isinstance(files, (str, Path)) else files)]
        chapters = [probe(f) for f in files]
        fps, total = chapters[0].fps, sum(c.n_frames for c in chapters)
        for c in chapters[1:]:
            if abs(c.fps - fps) > 1e-6 or (c.width, c.height) != (chapters[0].width,
                                                                  chapters[0].height):
                raise ValueError(f"chapter {c.path} differs from {files[0]} (fps/size)")
        t0, t1 = t_range
        i0 = max(0, int(round(float(t0 or 0.0) * fps)))
        i1 = total if t1 is None else min(total, int(round(float(t1) * fps)))
        if i1 <= i0:
            raise ValueError(f"empty window {t_range} of {files} ({total / fps:.1f} s)")
        return cls(files, i0, i1, chapters)

    @property
    def fps(self) -> float:
        return self.chapters[0].fps

    @property
    def offsets(self) -> list[int]:
        """First frame of each chapter in the recording."""
        return [int(v) for v in np.cumsum([0] + [c.n_frames for c in self.chapters[:-1]])]

    @property
    def info(self) -> VideoInfo:
        """The window as one video; creation_time moved to the window start (GoPro writes
        local time with a 'Z' suffix; kept as is)."""
        c = self.chapters[0]
        n = self.i1 - self.i0
        ct = c.creation_time
        if ct and self.i0:
            try:
                d = datetime.fromisoformat(ct.rstrip("Z")) + timedelta(seconds=self.i0 / self.fps)
                ct = d.strftime("%Y-%m-%dT%H:%M:%S.%f") + ("Z" if ct.endswith("Z") else "")
            except ValueError:
                pass
        return replace(c, n_frames=n, duration=n / self.fps, creation_time=ct)

    def describe(self) -> dict:
        """Identity of the window (cache key of a clip's outputs)."""
        return {"files": [Path(f).name for f in self.files], "frames": [self.i0, self.i1],
                "fps": self.fps}

    def locate(self, t_s: float) -> tuple[str, float]:
        """(chapter file, time in that file) of window time t_s."""
        g = self.i0 + t_s * self.fps
        k = max(0, int(np.searchsorted(self.offsets, g, side="right")) - 1)
        return self.files[k], (g - self.offsets[k]) / self.fps

    def iter_frames(self, width=None, height=None, gray=False, start_s=0.0, duration_s=None,
                    step=1, hwaccel=True):
        a = self.i0 + max(0, int(round(start_s * self.fps)))
        b = self.i1 if duration_s is None else min(self.i1, a + int(round(duration_s * self.fps)))
        for f, info, off in zip(self.files, self.chapters, self.offsets):
            lo, hi = max(a, off), min(b, off + info.n_frames)
            if hi <= lo:
                continue
            phase = (lo - self.i0) % step  # keep window frames whose index is a multiple of step
            n = len(range((-phase) % step, hi - lo, step))
            if n == 0:
                continue
            for k, img in _iter_file(f, width, height, gray, (lo - off) / self.fps, None, step,
                                     hwaccel, n_max=n, phase=phase):
                yield k - (lo - off) + lo - self.i0, img
