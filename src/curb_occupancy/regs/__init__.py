"""Parking regulations -> curb supply -> guideline occupancy.

* `rules`     city-independent schema: zones, points, time rules (Rule), kinds and priorities
* `guideline` occupancy-study profiles (stall length, time window, thresholds)
* `supply`    the stage: sources -> curb status per row -> capacity per block -> occupancy
* sources (`regulations.sources[].type`, registry `supply.SOURCES`): `manual` (zones/points
  written in the config: driveways, site corrections, any city), `osm` (curb cuts of alleys /
  driveways mapped in OpenStreetMap; needs the OSM street axis), `montreal`
  (`adapters/montreal.py`: city open data, signage + hydrants)

A new city = an adapter module with `load(source_cfg, ctx) -> Regs` (optional `fetch_source`,
`ATTRIBUTION`, `POLES`; see `adapters/montreal.py`), one line in `supply.SOURCES`, and a
guideline profile if the city has its own method.
"""
from .rules import Regs, Rule, rule_from_dict  # noqa: F401
