import numpy as np
import pandas as pd
import pytest

from curb_occupancy.adapters.montreal import set_arrows
from curb_occupancy.layout import dark_clip


def _signs():
    return pd.DataFrame({"POTEAU_ID_POT": [1, 2, 2, 2],
                         "DESCRIPTION_RPA": ["\\A EN TOUT TEMPS", "\\A EN TOUT TEMPS",
                                             "\\P EXCEPTE HANDICAPES (PICTO)", "\\P 10h-12h LUNDI"],
                         "FLECHE_PAN": [2, 3, 2, 0]})


def test_set_arrows_single_and_rpa():
    S = set_arrows(_signs(), [{"pole": 1, "arrow": 3},
                              {"pole": 2, "arrow": 2, "rpa": "\\A EN TOUT TEMPS"}])
    assert S.FLECHE_PAN.tolist() == [3, 2, 2, 0]


def test_set_arrows_ambiguous_pole_needs_rpa():
    with pytest.raises(ValueError):
        set_arrows(_signs(), [{"pole": 2, "arrow": 3}])
    with pytest.raises(ValueError):
        set_arrows(_signs(), [{"pole": 9, "arrow": 3}])


def test_dark_clip(tmp_path):
    np.savez(tmp_path / "telemetry.npz", shutter_s=np.full(10, 1 / 60), iso=np.full(10, 6400.0))
    assert dark_clip(tmp_path, 30.0)
    assert not dark_clip(tmp_path, None)
    np.savez(tmp_path / "telemetry.npz", shutter_s=np.full(10, 1 / 240), iso=np.full(10, 3800.0))
    assert not dark_clip(tmp_path, 30.0)
    assert not dark_clip(tmp_path / "missing", 30.0)


def test_pin_intersections():
    from curb_occupancy.layout import pin_intersections
    ints = pd.DataFrame({"x0_m": [10.0, 50.0], "x1_m": [30.0, 60.0], "centre_m": [20.0, 55.0],
                         "name": ["A St", "B St"], "source": ["osm+detected", "osm"]})
    out = pin_intersections(ints, [{"name": "A St", "x1_m": 22.0}])
    r = out[out["name"] == "A St"].iloc[0]
    assert (r.x0_m, r.x1_m, r.centre_m, r.source) == (10.0, 22.0, 16.0, "config")
    with pytest.raises(ValueError):
        pin_intersections(ints, [{"name": "C St", "x0_m": 1.0}])
    with pytest.raises(ValueError):
        pin_intersections(ints, [{"name": "B St", "x0_m": 70.0}])
