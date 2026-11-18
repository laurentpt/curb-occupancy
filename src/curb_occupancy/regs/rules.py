"""City-independent curb regulation schema.

A regulation source (a city adapter, manual YAML zones, ...) produces, on the street axis:

* **zones**: one row per regulated curb stretch, columns
  `row` (parking row it applies to), `x0`, `x1` (street-axis metres), `kind` (below), `rule`
  (key into the rules dict: when the zone is in force), `source`, `ref` (source id, e.g. a
  sign pole), `text` (human-readable regulation), `disputed` (bool, set by the supply stage);
* **points**: curb features with a legal clearance, columns `row`, `x_ref`, `kind`
  (`hydrant`, ...), `source`, `ref`; the clearance per kind comes from `clearances_m`;
* **rules**: `{rule key: Rule}`, a time pattern (days, hours, season) plus the kind.

Kinds, by priority when zones overlap (higher wins): `intersection` (clearance, generated) >
`no_stopping` > `no_parking` > `reserved` (disabled, delivery, drop-off, taxi, bus) >
`hydrant` = `driveway` (curb cuts) > `limited` (time-limited parking) > `permit` (resident
permit zones) > `legal`. A zone only counts while its rule is active.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

KINDS = ("no_stopping", "no_parking", "reserved", "driveway", "limited", "permit", "unknown")
PRIORITY = {"legal": 0, "permit": 1, "limited": 2, "driveway": 3, "hydrant": 3, "reserved": 4,
            "no_parking": 5, "no_stopping": 6, "intersection": 7}
PARKABLE = ("legal", "limited", "permit")      # a parked car there is not illegally parked
ZONE_COLS = ["row", "x0", "x1", "kind", "rule", "source", "ref", "text"]
POINT_COLS = ["row", "x_ref", "kind", "source", "ref"]


@dataclass
class Rule:
    """When a regulation applies. kind: one of KINDS. Active when day-of-week in `days`,
    (month, day) in `season` and the time in one of `hours` (empty = always)."""
    text: str
    kind: str
    minutes: int | None = None
    hours: list = field(default_factory=list)      # [(h0, h1)] decimal hours, may wrap midnight
    days: set = field(default_factory=lambda: set(range(7)))  # 0 = Monday
    season: tuple | None = None                    # ((m0, d0), (m1, d1)) inclusive, may wrap

    def active(self, when: datetime) -> bool:
        if when.weekday() not in self.days:
            return False
        if self.season is not None:
            (m0, d0), (m1, d1) = self.season
            md = (when.month, when.day)
            inside = (m0, d0) <= md <= (m1, d1) if (m0, d0) <= (m1, d1) else \
                (md >= (m0, d0) or md <= (m1, d1))
            if not inside:
                return False
        if not self.hours:
            return True
        h = when.hour + when.minute / 60
        return any((h0 <= h < h1) if h0 < h1 else (h >= h0 or h < h1) for h0, h1 in self.hours)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "minutes": self.minutes, "hours": self.hours,
                "days": sorted(self.days), "season": self.season}


_DAY = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}


def rule_from_dict(d: dict | None, kind: str, text: str = "") -> Rule:
    """Rule from config: {hours: [[17, 19]], days: [mon, ..] or [0, ..], season: [[4, 1],
    [12, 1]], minutes: 15}; missing fields = always."""
    d = d or {}
    r = Rule(text=text or kind, kind=kind, minutes=d.get("minutes"))
    r.hours = [tuple(map(float, h)) for h in d.get("hours", [])]
    if "days" in d:
        r.days = {_DAY[v[:3].lower()] if isinstance(v, str) else int(v) for v in d["days"]}
    if d.get("season"):
        (m0, d0), (m1, d1) = d["season"]
        r.season = ((int(m0), int(d0)), (int(m1), int(d1)))
    return r


def empty_zones() -> pd.DataFrame:
    return pd.DataFrame(columns=ZONE_COLS)


def empty_points() -> pd.DataFrame:
    return pd.DataFrame(columns=POINT_COLS)


@dataclass
class Regs:
    """What a source returns: zones and points on the street axis, rules, diagnostics, and
    extra tables to write to supply/ (name -> DataFrame)."""
    zones: pd.DataFrame
    points: pd.DataFrame
    rules: dict
    info: dict = field(default_factory=dict)
    tables: dict = field(default_factory=dict)
    clearances_m: dict = field(default_factory=dict)  # jurisdiction defaults, e.g. {hydrant: 5}
