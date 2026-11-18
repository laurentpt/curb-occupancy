import pandas as pd

from curb_occupancy.cli import acceptance
from curb_occupancy.config import DEFAULTS
from curb_occupancy.regs.supply import in_roadway


def test_in_roadway_only_on_cut_curb():
    ints = pd.DataFrame({"x0_m": [100.0], "x1_m": [110.0], "sides": ["left"], "name": ["X"]})
    rows = {"A": {"curb": "right"}, "B": {"curb": "left"}}
    cars = pd.DataFrame({"row": ["B", "B", "A", "B"], "xr_mid_m": [105.0, 112.0, 105.0, 99.0]})
    # B on the cut curb inside the roadway; B in the clearance beyond the edge counts;
    # A: the T does not cut the right curb
    assert in_roadway(cars, ints, rows).tolist() == [True, False, False, False]


def test_dark_min_conf():
    cfg = {"detect": {**DEFAULTS["detect"]}, "mosaic": {"planes": {"near": {"lateral_m": [0, 1]}}}}
    assert acceptance(cfg, "near").min_conf == DEFAULTS["detect"]["min_conf"]
    assert acceptance(cfg, "near", dark=True).min_conf == DEFAULTS["detect"]["min_conf_dark"]
