import numpy as np
import pandas as pd
import pytest

from curb_occupancy import ride
from curb_occupancy.capture import video as V


def _src(n=(10, 7), i0=0, i1=None):
    ch = [V.VideoInfo(f"c{k}.MP4", 8, 6, 60.0, m, m / 60, "hevc", "2026-10-26T18:00:00.000000Z",
                      "") for k, m in enumerate(n)]
    return V.Source([c.path for c in ch], i0, sum(n) if i1 is None else i1, ch)


def test_source_frames_across_chapters(monkeypatch):
    """Window [8, 15) of chapters of 10 + 7 frames, step 3: window frames 0, 3, 6 = recording
    frames 8 (chapter 0), 11, 14 (chapter 1, its frames 1 and 4)."""
    calls = []

    def fake(path, w, h, gray, start_s, dur, step, hw, n_max, phase):
        first = int(round(start_s * 60))
        calls.append((path, first, n_max, phase))
        for k in range(n_max):
            n = (-phase) % step + k * step
            yield first + n, np.full((6, 8), int(path[1]) * 100 + first + n, np.uint8)
    monkeypatch.setattr(V, "_iter_file", fake)
    s = _src(i0=8, i1=15)
    got = [(i, int(im[0, 0])) for i, im in s.iter_frames(step=3)]
    assert got == [(0, 8), (3, 101), (6, 104)]
    assert calls == [("c0.MP4", 8, 1, 0), ("c1.MP4", 0, 2, 2)]
    assert s.info.n_frames == 7
    assert s.info.creation_time == "2026-10-26T18:00:00.133333Z"
    assert s.locate(4 / 60) == ("c1.MP4", pytest.approx(2 / 60))


def test_runs_absorb_and_side():
    road = ["A"] * 20 + ["B"] * 3 + ["A"] * 20 + [""] * 5 + ["C"] * 30
    t = np.arange(len(road)) * 0.2
    m = pd.DataFrame({"t": t, "x": np.zeros(len(t)), "y": t * 5, "speed": 5.0, "road": road})
    r = ride.runs(m, absorb_s=4.0)
    assert r.road.tolist() == ["A", "", "C"]
    assert r.length_m.iloc[0] == pytest.approx(42 * 0.2 * 5)
    # riding grid-north (true 302 deg in NDG) with the camera on the left: west curb
    assert ride.side_label(302, "left", -58) == "west side"
    assert ride.side_label(122, "left", -58) == "east side"
    assert ride.side_label(122, "right", -58) == "west side"
    assert ride.slug("Avenue de Hampton") == "hampton"


def test_append_item(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text("name: x  # keep\nclips:\n  - file: A.MP4\n    note: a\n")
    clip = {"file": ["B.MP4", "C.MP4"], "t_range": [1.5, 9.0], "note": "b: c"}
    ride._append_item(p, "clips", ride._clip_yaml(clip), clip)
    assert "# keep" in p.read_text()
    p.write_text("clips:\n  - file: A.MP4\nname: x\n")
    with pytest.raises(ValueError):
        ride._append_item(p, "clips", ride._clip_yaml(clip), clip)


def test_chapters(tmp_path):
    for f in ("GX014496.MP4", "GX024496.MP4", "GX034496.MP4", "GX014497.MP4"):
        (tmp_path / f).touch()
    assert ride.chapters(tmp_path, "GX014496.MP4") == ["GX014496.MP4", "GX024496.MP4",
                                                       "GX034496.MP4"]
    assert ride.chapters(tmp_path, "GX014497.MP4") == ["GX014497.MP4"]


def test_config_path(tmp_path):
    from curb_occupancy.ride import config_path
    d = tmp_path
    existing = [(d / "draper.yaml", "Avenue Draper", "east side")]
    assert config_path(d, existing, {}, "avenue draper", "east side") == d / "draper.yaml"
    assert config_path(d, existing, {}, "Avenue Draper", "west side") == d / "draper_west.yaml"
    # a second pass the same way on a street new in this ride joins the first one's config
    planned = {d / "somerled.yaml": [(0, {}, "Avenue Somerled", "south side")]}
    assert config_path(d, existing, planned, "Avenue Somerled", "south side") == d / "somerled.yaml"
    assert config_path(d, existing, planned, "Avenue Somerled", "north side") == \
        d / "somerled_north.yaml"
