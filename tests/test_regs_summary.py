from curb_occupancy.publish.webmap import regs_summary


def test_regs_summary_unique_poles_and_modes():
    t = {"montreal": {"arrows": "convention", "prior": None, "poles_arrows": [1, 2],
                      "poles_dropped": [5, 6], "poles_moved": {"7": 10.0}}, "osm": {"axis": "osm"}}
    m = {"montreal": {"arrows": "inferred", "prior": None, "poles_dropped": [6],
                      "poles_moved": {}}}
    r = regs_summary([("T", t), ("M", m), ("M2", {"site": {"n_zones": 1}})])
    assert r == {"convention": ["T"], "inferred": ["M"], "arrows": 2, "dropped": 2, "moved": 1}


def test_regs_summary_none_without_signage_or_with_prior():
    assert regs_summary([("X", {"curb_labels": {"n_zones": 3}})]) is None
    assert regs_summary([("X", {"montreal": {"arrows": "convention", "prior": {"corner_m": 9}}})]) is None
