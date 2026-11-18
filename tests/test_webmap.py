import numpy as np
import pandas as pd

from curb_occupancy.street import StreetAxis
from curb_occupancy.publish.webmap import Geo, _runs, _stack, curb_offsets


def _east_axis(length=200.0, lat0=45.47, lon0=-73.63):
    """Straight street axis heading east (X increasing eastward)."""
    X = np.arange(0.0, length + 1, 10.0)
    lon = lon0 + np.degrees(X / (6371000.0 * np.cos(np.radians(lat0))))
    return StreetAxis("osm", np.full(len(X), lat0), lon, X)


def _sidewalk(ax, lat_m, x0=0.0, x1=200.0):
    lat = ax.lat[0] + np.degrees(lat_m / 6371000.0)
    lon = np.interp([x0, x1], ax.X, ax.lon)
    return {"type": "way", "tags": {"highway": "footway", "footway": "sidewalk"},
            "geometry": [{"lat": lat, "lon": float(v)} for v in lon]}


def test_curb_offsets_from_sidewalks_and_lateral_sign():
    ax = _east_axis()
    osm = {"elements": [_sidewalk(ax, 8.0), _sidewalk(ax, -7.0, 0, 120)]}  # north = left
    off = curb_offsets(ax, osm, 0.0, 200.0, default=6.0, inset=1.0)
    g, left = off["left"]
    assert np.allclose(left[(g > 20) & (g < 180)], 7.0, atol=0.2)
    g, right = off["right"]
    assert np.allclose(right[(g > 20) & (g < 100)], 6.0, atol=0.2)
    assert np.allclose(right[g > 150], 6.0, atol=0.2)        # unmapped: held from the last bin
    geo = Geo(ax, off, lane_inset=2.5)
    lat_l, _ = geo.line(50, 60, "left")[0]
    lat_r, _ = geo.line(50, 60, "right")[0]
    assert lat_l > ax.lat[0] > lat_r                          # left of an eastward axis = north
    lane = geo.lane(50, 60, "left")[0][0]
    assert ax.lat[0] < lane < lat_l                           # parked row inside the curb


def test_runs_and_zone_stacking():
    v = np.array(["legal", "legal", "hydrant", "legal"], dtype=object)
    assert _runs(v, np.arange(4.0), 1.0) == [("legal", 0.0, 2.0), ("hydrant", 2.0, 3.0),
                                             ("legal", 3.0, 4.0)]
    z = pd.DataFrame({"x0": [0.0, 5.0, 12.0, 30.0], "x1": [10.0, 20.0, 25.0, 40.0]})
    assert list(_stack(z)) == [0, 1, 0, 0]


def test_labels_roundtrip(tmp_path):
    from curb_occupancy.publish.labeller import read_labels, write_labels
    f = tmp_path / "curb.yaml"
    z = [{"kind": "driveway", "curb": "right", "latlon": [[45.1, -73.6], [45.10005, -73.6]],
          "axis_m": [12.0, 18.0], "note": "4500 'x'"},
         {"kind": "driveway", "curb": "left", "latlon": [[45.2, -73.6], [45.2, -73.6]],
          "axis_m": [5.0, 9.0], "extra": 1}]
    write_labels(f, z, "run")
    back = read_labels(f)
    assert [b["axis_m"][0] for b in back] == [5.0, 12.0]            # sorted along the axis
    assert back[1]["note"] == "4500 'x'" and "extra" not in back[0]


def test_gap_candidates_between_cars_only():
    from curb_occupancy.publish.labeller import gap_candidates
    grid = np.arange(0.0, 100.0)
    cars = pd.DataFrame({"xr_start_m": [10.0, 21.0, 60.0], "xr_end_m": [15.0, 26.0, 65.0]})
    ok = np.ones(len(grid), bool)
    # 15-21 (6 m) between cars: candidate; 26-60 (34 m) too long; edges never
    assert gap_candidates(cars, ok, ok, grid, (2.5, 12.0)) == [[15.0, 21.0]]


def test_resample_strip_flips_and_masks():
    from curb_occupancy.register import resample_strip
    img = np.zeros((4, 50, 3), np.uint8)
    img[:, :, 0] = np.arange(50)[None, :]
    xs = np.arange(10.0)
    out = resample_strip(img, lambda x: 40.0 - x, xs, np.arange(4.0), xs < 8)
    assert out[0, 0, 0] == 40 and out[0, 7, 0] == 33 and (out[:, 8:] == 0).all()


def test_ride_curb_classes_and_compass_side():
    from curb_occupancy.publish.rideview import compass_side, curb_class
    g = {"count_limited": False, "count_permit": True}
    st = np.array(["legal", "permit", "limited", "driveway", "hydrant", "no_stopping",
                   "intersection", None], dtype=object)
    assert list(curb_class(st, g)) == ["allowed", "allowed", "short", "cut", "cut",
                                       "forbidden", "", ""]
    ax = _east_axis()
    assert compass_side(ax, 0, 100, "right") == "south side"
    assert compass_side(ax, 0, 100, "left") == "north side"
