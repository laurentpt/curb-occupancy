"""Time slots (guideline.slots): validation, pass -> slot, per-stretch slot means and the
time-weighted (space-hours) average."""
import numpy as np
import pandas as pd
import pytest

from curb_occupancy.publish.webmap import _majority, aggregates
from curb_occupancy.regs.guideline import check_slots, guideline, slot_of
from curb_occupancy.regs.supply import slot_means, slot_totals

SLOTS = [{"key": "late", "hours": [19, 21]}, {"key": "early", "hours": [17, 19]}]


def g_slots(slots=SLOTS):
    return guideline({"guideline": {"profile": "montreal", "slots": slots}})


def test_check_slots_sorts_and_weights():
    s = check_slots(SLOTS, [17, 21])
    assert [x["key"] for x in s] == ["early", "late"]
    assert [x["weight"] for x in s] == [0.5, 0.5]
    s = check_slots([{"key": "a", "hours": [17, 18]}, {"key": "b", "hours": [18, 21]}], [17, 21])
    assert [x["weight"] for x in s] == [0.25, 0.75]


@pytest.mark.parametrize("bad", [
    [{"key": "a", "hours": [17, 19.5]}, {"key": "b", "hours": [19, 21]}],   # overlap
    [{"key": "a", "hours": [16, 19]}],                                     # outside window
    [{"key": "a", "hours": [17, 19]}, {"key": "a", "hours": [19, 21]}],    # duplicate key
])
def test_check_slots_rejects(bad):
    with pytest.raises(ValueError):
        check_slots(bad, [17, 21])


def test_no_slots_by_default():
    assert "slots" not in guideline({"guideline": {"profile": "montreal"}})


def test_slot_of():
    s = g_slots()["slots"]
    assert slot_of("2026-11-12T17:17", s) == "early"
    assert slot_of("2026-11-12T18:59", s) == "early"
    assert slot_of("2026-11-12T19:00", s) == "late"
    assert slot_of("2026-11-12T21:00", s) is None


def _tables(passes):
    """passes: [(clip, when, [(block, capacity, parked), ...])] for stretch S01."""
    cov = pd.DataFrame([{"clip": c, "when": w, "date": w[:10], "stretch_id": "S01",
                         "block_id": b, "row": "L", "capacity": k, "n_parked": n}
                        for c, w, bl in passes for b, k, n in bl])
    e = cov.groupby("clip").agg(when=("when", "first"), date=("date", "first")).reset_index()
    e["stretch_id"] = "S01"
    sp = pd.DataFrame({"stretch_id": ["S01"], "capacity_full": [30.0]})
    return e, cov, sp


def test_slot_means_space_hours():
    # early: two passes the same evening (averaged first) + one other evening; late: one
    # pass with a larger capacity (rules lifted at night)
    e, cov, sp = _tables([
        ("a", "2026-10-23T17:17", [("G1", 20, 10)]),
        ("b", "2026-10-23T17:25", [("G1", 20, 14)]),
        ("c", "2026-10-26T17:39", [("G1", 20, 8)]),
        ("d", "2026-11-10T20:10", [("G1", 30, 24)])])
    r = slot_means(e, cov, sp, g_slots()).set_index("agg")
    # early: evening 10-23 = 12, 10-26 = 8 -> 10 / 20
    assert r.loc["slot:early", "parked_mean"] == pytest.approx(10)
    assert r.loc["slot:early", "n_dates"] == 2 and r.loc["slot:early", "n_passes"] == 3
    assert r.loc["slot:late", "occupancy_mean"] == pytest.approx(0.8)
    # avg: (0.5 * 10 + 0.5 * 24) / (0.5 * 20 + 0.5 * 30) = 17 / 25, not mean(0.5, 0.8)
    assert r.loc["avg", "occupancy_mean"] == pytest.approx(17 / 25)
    assert r.loc["avg", "slots"] == "early+late"
    t = slot_totals(r.reset_index())
    assert t["avg"]["one_slot"] == 0 and t["avg"]["occupancy"] == pytest.approx(0.68)


def test_slot_means_one_slot_and_common_blocks():
    # one slot only: avg = that slot; blocks not covered by every pass of a view are dropped
    e, cov, sp = _tables([
        ("a", "2026-11-10T20:10", [("G1", 10, 5), ("G2", 10, 9)]),
        ("b", "2026-11-11T20:20", [("G1", 10, 7)])])
    r = slot_means(e, cov, sp, g_slots()).set_index("agg")
    assert list(r.index) == ["avg", "slot:late"]
    assert r.loc["avg", "slots"] == "late"
    assert r.loc["avg", "capacity_mean"] == pytest.approx(10)       # G1 only
    assert r.loc["avg", "occupancy_mean"] == pytest.approx(0.6)
    assert slot_totals(r.reset_index())["avg"]["one_slot"] == 1


def test_aggregates_pass_weights():
    from datetime import datetime as D
    when = {"a": D(2026, 10, 23, 17, 17), "b": D(2026, 10, 23, 17, 25),
            "c": D(2026, 10, 26, 17, 39), "d": D(2026, 11, 10, 20, 10)}
    A = {a["id"]: a for a in aggregates(g_slots(), when)}
    assert list(A) == ["avg", "slot:early", "slot:late"]
    w = A["avg"]["pass_weight"]
    assert w["d"] == pytest.approx(0.5) and w["c"] == pytest.approx(0.25)
    assert w["a"] == pytest.approx(0.125) and sum(w.values()) == pytest.approx(1)
    assert A["slot:early"]["dates"] == ["2026-10-23", "2026-10-26"]
    assert aggregates(guideline({"guideline": {"profile": "montreal"}}), when) == []


def test_weighted_majority():
    g = g_slots()
    st = np.array([["legal"], ["legal"], ["no_parking"]], dtype=object)
    assert _majority(st, g)[0] == "legal"
    assert _majority(st, g, np.array([0.1, 0.1, 0.8]))[0] == "no_parking"
