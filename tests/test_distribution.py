import pandas as pd
import pytest

from curb_occupancy.publish.area import distribution, stretch_evenings

G = {"slots": [{"key": "early", "hours": [17, 19], "weight": 0.5},
               {"key": "late", "hours": [19, 21], "weight": 0.5}],
     "classes": [[0.5, "very_low", "Very low"], [0.8, "low", "Low"], [0.9, "optimal", "Optimal"],
                 [None, "full", "Full"]]}


def test_stretch_evenings_averages_passes_of_one_evening():
    sp = pd.DataFrame({"street": "T", "stretch_id": "t:S01", "clip": ["a", "b", "c"],
                       "when": ["2026-10-23T17:17", "2026-10-23T17:25", "2026-11-05T20:00"],
                       "date": ["2026-10-23", "2026-10-23", "2026-11-05"], "in_window": True,
                       "counted": True, "capacity": [10, 10, 10], "n_parked": [4, 6, 9]})
    U = stretch_evenings(sp, G)
    assert len(U) == 2
    e = U[U.date == "2026-10-23"].iloc[0]
    assert (e.n_passes, e.parked, e.slot, e["class"]) == (2, 5.0, "early", "low")


def test_distribution_time_weighted_and_slot_views():
    U = pd.DataFrame({"street": ["S"] * 4, "stretch_id": ["s:1", "s:1", "s:1", "s:2"],
                      "date": ["d1", "d2", "d3", "d3"], "slot": ["early", "early", "late", "late"],
                      "class": ["full", "low", "very_low", "low"]})
    b = distribution(U, {"bars": [{"view": "avg"}, {"view": "slot:early", "example": True}]}, G)
    # s:1: early evenings 0.25 each, late evening 0.5; s:2 (late only): 1 -> total 2
    assert b[0]["shares"] == {"very_low": 0.25, "low": 0.625, "optimal": 0.0, "full": 0.125}
    assert b[0]["n_units"] == 4 and b[0]["weighted"]
    assert b[1]["shares"]["full"] == 0.5 and b[1]["n_units"] == 2
    assert [x["example"] for x in b] == [False, True]
    with pytest.raises(ValueError):
        distribution(U, {"bars": [{"view": "evening"}]}, G)
