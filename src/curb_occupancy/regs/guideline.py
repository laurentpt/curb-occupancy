"""Occupancy-study guideline profiles: how capacity is counted and occupancy judged.

`guideline: {profile: montreal, ...overrides}` in the street config. Values:
stall_m (curb length per space), window (decimal hours, passes counted), n_evenings (passes
on distinct dates the method asks for), thresholds (the method's own verdict: >= optimal is
optimal, < low is low), classes (display bins: [upper bound, key, label], last bound null =
open), count_limited (count
time-limited stretches as capacity while their limit is in force), count_permit (count
resident-permit stretches), stretch_spaces (reporting unit, not the city's: consecutive
blocks grouped to about this many spaces for the map and stretch tables; 0 = per block).

`slots` (study design, optional, not in any profile): the window split into time slots,
`[{key, hours: [a, b], label: {en, fr}}]`, contiguous inside `window`. Passes are grouped by
the slot their start time falls in; a slot's value per stretch is the mean over its evenings
(dates; passes of one evening averaged first) and the time-weighted average weighs each slot
by its hours: occupied space-hours / available space-hours, i.e. sum(w * parked) / sum(w *
capacity) (capacity moves with the hour's rules). A stretch seen in one slot only takes
that slot's value (flagged).
"""
from __future__ import annotations

PROFILES = {
    # Ville de Montreal, as in the REV Bellechasse monitoring plan (2021): capacity per street
    # segment at 5.5 m per space minus driveways, hydrants and no-parking-at-all-times curb;
    # occupancy 17-21 h over three weekday evenings; 85 % generally optimal, < 80 % low, maps
    # with 80-90 % optimal and > 90 % high. `reference` = the document followed (page text).
    "montreal": {"stall_m": 5.5, "window": [17.0, 21.0], "n_evenings": 3,
                 "thresholds": {"optimal": 0.85, "low": 0.80},
                 "classes": [[0.5, "very_low", "Very low"], [0.8, "low", "Low"],
                             [0.9, "optimal", "Optimal"], [None, "full", "Full"]],
                 "count_limited": False, "count_permit": True, "stretch_spaces": 25,
                 "reference": {
                     "title": "Ville de Montréal, Plan de suivi, REV dans l'axe Bellechasse (2021)",
                     "url": "https://github.com/OpenMobilityData/Terrebonne-access-requests/blob/"
                            "main/2025-12847/raw/Annexe_1.12_Plan_de_suivi_REV_Bellechasse_"
                            "2022-02-18.pdf"}},
}


def guideline(cfg: dict) -> dict:
    from ..config import _merge
    g = dict(cfg.get("guideline") or {})
    name = g.pop("profile", None)
    if name not in PROFILES:
        raise ValueError(f"guideline.profile {name!r}: known profiles {sorted(PROFILES)}")
    out = {"profile": name, **_merge(PROFILES[name], g)}
    if out.get("slots"):
        out["slots"] = check_slots(out["slots"], out["window"])
    return out


def check_slots(slots: list, window) -> list[dict]:
    """Slots sorted by start, each with `weight` = its hours / the hours of all slots; they
    must lie inside the window, without overlap, with distinct keys."""
    out = sorted(({**s, "hours": [float(s["hours"][0]), float(s["hours"][1])]} for s in slots),
                 key=lambda s: s["hours"][0])
    keys = [s["key"] for s in out]
    if len(set(keys)) != len(keys):
        raise ValueError(f"guideline.slots: duplicate keys {keys}")
    for a, b in zip(out, out[1:]):
        if b["hours"][0] < a["hours"][1]:
            raise ValueError(f"guideline.slots: {a['key']} and {b['key']} overlap")
    for s in out:
        a, b = s["hours"]
        if not (window[0] <= a < b <= window[1]):
            raise ValueError(f"guideline.slots: {s['key']} {s['hours']} outside the window {window}")
    tot = sum(s["hours"][1] - s["hours"][0] for s in out)
    return [{**s, "weight": round((s["hours"][1] - s["hours"][0]) / tot, 6)} for s in out]


def slot_of(when, slots: list) -> str | None:
    """Key of the slot a pass starting at `when` (datetime or ISO string) belongs to."""
    from datetime import datetime
    h = datetime.fromisoformat(when) if isinstance(when, str) else when
    v = h.hour + h.minute / 60
    return next((s["key"] for s in slots or [] if s["hours"][0] <= v < s["hours"][1]), None)


def occupancy_class(o, classes: list):
    """Key of the first class whose upper bound exceeds o (bins [lo, hi))."""
    import numpy as np
    if o is None or not np.isfinite(o):
        return None
    return next(k for hi, k, _ in classes if hi is None or o < hi)
