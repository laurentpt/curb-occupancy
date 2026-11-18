from pathlib import Path

import numpy as np
import pytest

from curb_occupancy import config as C
from curb_occupancy.capture.telemetry import Telemetry

ONE_ROW = ("camera: {rig: gopro_hero8_wide_hseis, height_m: 0.9}\n"
           "street:\n  rows: {L: {curb: left, plane: {forward: mid}}}\n  states: [[L]]\n"
           "mosaic:\n  planes: {mid: {depth_m: [2, 5], decode_scale: 0.5, lateral_m: [2, 5]}}\n")


def test_extends_and_null_drop(tmp_path):
    (tmp_path / "_base.yaml").write_text(
        ONE_ROW + "detect: {min_height_m: 0.9}\n")
    (tmp_path / "s.yaml").write_text(
        "extends: _base.yaml\nstreet: {rows: {L: {label: east side}}}\n"
        "mosaic: {planes: {extra: {depth_m: [1, 2]}}}\ndetect: {min_height_m: null}\n"
        "clips: [{file: X.MP4, t_range: [3, null], camera: {height_m: 0.8}}]\n")
    (tmp_path / "s2.yaml").write_text(
        "extends: s.yaml\nmosaic: {planes: {extra: null}}\n")
    cfg = C.load(tmp_path / "s2.yaml")
    assert list(cfg["street"]["rows"]) == ["L"]
    assert cfg["street"]["rows"]["L"] == {"curb": "left", "plane": {"forward": "mid"},
                                          "label": "east side"}
    assert list(cfg["mosaic"]["planes"]) == ["mid"]   # `extra: null` dropped the plane
    assert cfg["name"] == "s2"
    assert C.t_range(cfg, "X") == (3.0, None)
    cam = cfg["clips"][0]["camera"]                   # defaults < rig < street < clip
    assert cam["height_m"] == 0.8 and cfg["camera"]["height_m"] == 0.9
    assert cam["rs_readout_s"] == 0.011 and Path(cam["lens_profile"]).exists()


def test_cross_section_and_camera_required(tmp_path):
    (tmp_path / "s.yaml").write_text("camera: {rig: gopro_hero8_wide_hseis, height_m: 1}\n")
    with pytest.raises(ValueError, match="street.rows is required"):
        C.load(tmp_path / "s.yaml")
    (tmp_path / "s.yaml").write_text(ONE_ROW.replace("rig: gopro_hero8_wide_hseis, ", ""))
    with pytest.raises(ValueError, match="lens_profile"):
        C.load(tmp_path / "s.yaml")


def test_chapter_list_id(tmp_path):
    (tmp_path / "s.yaml").write_text(
        ONE_ROW + "clips: [{file: [GX014496.MP4, GX024496.MP4], t_range: [600, 700]}]\n")
    cfg = C.load(tmp_path / "s.yaml")
    assert cfg["clips"][0]["id"] == "GX014496"
    assert C.files(cfg["clips"][0]) == ["GX014496.MP4", "GX024496.MP4"]


def _tel(n=10):
    t = np.arange(float(n))
    return Telemetry(imu_t=t, gyro=np.zeros((n, 3)), accel=np.zeros((n, 3)), gps_t=t,
                     gps=np.zeros((n, 5)), gps_fix=np.full(n, 3), gps_dop=np.full(n, 200),
                     exp_t=t, shutter_s=t, iso=t, hilights_s=[1.0, 4.0])


def test_telemetry_cut_concat():
    tel = _tel()
    w = tel.cut(2.0, 5.0)
    assert w.gps_t.tolist() == [0, 1, 2, 3] and w.gps.shape == (4, 5)
    assert w.imu_t.tolist() == [0, 1, 2, 3] and w.accel.shape == (4, 3)
    assert w.iso.tolist() == [2, 3, 4, 5] and w.hilights_s == [2.0]
    assert len(tel.gps_t) == 10
    c = Telemetry.concat([tel, _tel(5)], [0.0, 10.0])
    assert c.imu_t.tolist() == list(range(15)) and c.gps.shape == (15, 5)
    assert c.hilights_s == [1.0, 4.0, 11.0, 14.0]
    # GoPro chapter: IMU restarts at 0, GPS / exposure already on the recording clock
    ch = _tel(5)
    ch.gps_t, ch.exp_t = ch.gps_t + 10.1, ch.exp_t + 10.1
    c = Telemetry.concat([tel, ch], [0.0, 10.0])
    assert c.imu_t.tolist() == list(range(15))
    assert np.allclose(c.gps_t[10:], np.arange(5) + 10.1)
    assert np.allclose(c.exp_t[10:], np.arange(5) + 10.1)


def test_local_start_dst():
    from curb_occupancy.capture.telemetry import local_start
    # GX014504: GPS UTC 2026-11-11 01:17:23 = 20:17:23 EST (camera clock still on EDT: 21:19)
    assert local_start(1794359843.139 - 0.118, 0.118, "America/Montreal") == "2026-11-10T20:17:23"
    # GX014498 (before the change, EDT): 21:21:48 UTC = 17:21:48
    assert local_start(1793308908.625, 0.0, "America/Montreal") == "2026-10-29T17:21:48"
