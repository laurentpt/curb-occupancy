"""City of Montreal adapter: on-street parking signage + fire hydrants from the open data.

Data (CC-BY 4.0, donnees.montreal.ca): "Signalisation (stationnement sur rue)", one row per
sign panel with pole id (POTEAU_ID_POT), regulation text (DESCRIPTION_RPA), arrow code
(FLECHE_PAN) and lon/lat, updated daily; "Bornes d'incendie" (hydrant points).
`python -m curb_occupancy.adapters.montreal fetch <dir>` (or `curb-occupancy fetch <config>`) downloads both.

Source config (`regulations.sources`):
    - type: montreal
      signs_csv: <path>        hydrants_csv: <path>       offset_search_m: 8
      offset_m: -1.0           # pin the residual along-axis offset (default: fitted)
      drop_poles:              # panels a site check showed misplaced / gone: not applied
        - {pole: 257482, note: "at the corner, within the intersection clearance"}
      pole_positions:          # poles placed where the imagery shows them (street-axis metres
        - {pole: 93836, x_m: 1272.5, note: "seen in GX014484"}   # of the reference pass)
      pole_arrows:             # arrow codes recorded wrong (FLECHE_PAN 0 / 2 / 3); `rpa` = the
        - {pole: 299368, arrow: 3, note: "..."}   # panel's DESCRIPTION_RPA, needed when the
                                                  # pole carries several arrowed panels

Geometry. Panels and hydrants are projected onto the street axis (signed lateral offset, + =
left of the axis direction) and assigned to the parking row whose curb they stand on (a row
present at that position, curb on that side; `Context.curb_row`). Sign coordinates are a few
metres off: a residual along-axis offset is fitted by minimising parked-car length inside
no-stopping zones (`supply.fit_offset`).

Zones (rules checked against three passes on rue de Terrebonne, 2026-10-23). Arrow codes are
relative to the panel's facing, which the data does not record, so directions are inferred:
1. two arrowed panels with the same non-'\\A' regulation in a block point at each other; the
   zone runs between them (pairs taken in order along the block);
2. other arrowed panels on a pole that carries such a pair end point the opposite way;
3. remaining arrowed panels end an intersection approach: they point to the nearest cross
   street;
4. no arrow: both ways.
A zone runs from its pole to the next pole on the curb carrying the same regulation, or to
the block end. Code de la securite routiere (Quebec): no parking within 5 m of an
intersection or a hydrant (`CLEARANCES_M`).

`arrows: convention` reads every no-stopping panel as signed with the Quebec convention below
(no prior; zone from the pole in the arrow's direction to the next pole with the same text or
the block end, both ways without an arrow); `pole_arrows` corrects recorded arrow codes.

`prior:` replaces the inference for no-stopping and reserved panels with site knowledge
(`build_zones_prior`): parking is allowed everywhere except intersection approaches and
listed exceptions. Arrows then follow the Quebec convention (panels face traffic; toward the
roadway = the zone starts at the pole downstream), which along the street axis depends only
on the curb side: right curb 2 -> +X, 3 -> -X; left curb mirrored.

    prior:
      corner_m: 45          # arrowed no-stopping panel this close to a cross street cutting
                            # the row: intersection approach, pole -> corner (arrow ignored)
      reserved_m: 6.5       # lone reserved panel (no partner): one space
      keep:                 # mid-block regulations applied as signed
        - {between: [Boulevard Cavendish, Avenue Benny], row: A, note: school}
        - {pole: 93864}
Other mid-block no-stopping panels are not applied; they are listed (`signs_ignored.csv`)
for a site check.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ..regs.rules import POINT_COLS, ZONE_COLS, Regs, Rule

SIGNS_URL = ("https://donnees.montreal.ca/dataset/8ac6dd33-b0d3-4eab-a334-5a6283eb7940/resource/"
             "7f1d4ae9-1a12-46d7-953e-6b9c18c78680/download/signalisation_stationnement.csv")
HYDRANTS_URL = ("https://donnees.montreal.ca/dataset/cb4de65e-138b-4936-9d5c-2d9a0bc9b4ce/"
                "resource/75d2f1e2-b88d-4784-87df-5e7fc1e491a9/download/aqu_borneincendie_p.csv")
CLEARANCES_M = {"intersection": 5.0, "hydrant": 5.0}   # Code de la securite routiere
POLES = True
ATTRIBUTION = {"en": "Regulations and hydrants: Ville de Montr&eacute;al open data (CC-BY 4.0).",
               "fr": "R&eacute;glementation et bornes d&rsquo;incendie : donn&eacute;es ouvertes de "
                     "la Ville de Montr&eacute;al (CC-BY 4.0)."}
DEFAULTS = {"signs_csv": None, "hydrants_csv": None, "offset_search_m": 8.0, "offset_m": None,
            "drop_poles": [], "pole_positions": [], "pole_arrows": [], "arrows": "inferred"}

# ---------------------------------------------------------------------------- text parsing

DAYS = {"LUN": 0, "MAR": 1, "MER": 2, "JEU": 3, "VEN": 4, "SAM": 5, "DIM": 6}
MONTHS = {"JAN": 1, "FEV": 2, "MAR": 3, "AVR": 4, "MAI": 5, "JUIN": 6, "JUIL": 7, "AOU": 8,
          "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}
_MONTH_RE = r"(JANV?|FEVR?|MARS|AVR(?:IL)?|MAI|JUIN|JUIL(?:LET)?|AOUT?|SEPT?|OCT|NOV|DEC)\.?"


def _month(tok: str) -> int:
    return MONTHS[tok[:4]] if tok[:4] in MONTHS else MONTHS[tok[:3]]


def parse_rule(text: str) -> Rule:
    """DESCRIPTION_RPA text -> Rule. '\\A' = no stopping, '\\P' = no parking (EXCEPTE ...:
    reserved or permit), 'P' = parking (time-limited, reserved)."""
    t = " ".join(str(text).upper().replace("É", "E").replace("È", "E").split())
    if t.startswith("\\A"):
        kind = "no_stopping"
    elif t.startswith("\\P"):
        kind = "no_parking"
        if "EXCEPT" in t:
            kind = "reserved" if re.search(r"HANDICAP|DEBARC|LIVRAISON|TAXI|AUTOBUS", t) \
                else "permit"
    elif re.match(r"^P\b", t):
        kind = ("reserved" if re.search(r"RESERV|SEULEMENT|HANDICAP", t) else
                "limited" if "MIN" in t else "unknown")
    else:
        kind = "unknown"
    r = Rule(text=str(text), kind=kind)
    m = re.search(r"P\s+(\d+)\s*MIN", t)
    if m:
        r.minutes = int(m.group(1))
    for h0, m0, h1, m1 in re.findall(r"(\d{1,2})\s*H\s*(\d{2})?\s*(?:-|A)\s*(\d{1,2})\s*H\s*(\d{2})?",
                                     t):
        r.hours.append((int(h0) + int(m0 or 0) / 60, int(h1) + int(m1 or 0) / 60))
    # days: ranges "LUN A VEN", "LUN.AU SAM.", lists "MARDI JEUDI"
    rng = re.search(r"\b(LUN|MAR|MER|JEU|VEN|SAM|DIM)\w*\.?\s*(?:A|AU)\s*(LUN|MAR|MER|JEU|VEN|SAM|DIM)",
                    t)
    if rng:
        a, b = DAYS[rng.group(1)], DAYS[rng.group(2)]
        r.days = set(range(a, b + 1)) if a <= b else set(range(a, 7)) | set(range(0, b + 1))
    else:
        named = re.findall(r"\b(LUNDI|MARDI|MERCREDI|JEUDI|VENDREDI|SAMEDI|DIMANCHE)\b", t)
        if named:
            r.days = {DAYS[n[:3]] for n in named}
    if "JOURS D'ECOLE" in t or "JOURS D ECOLE" in t:
        r.days = set(range(5))
        r.season = r.season or ((8, 25), (6, 25))
    sm = re.search(r"(\d{1,2})?(?:ER)?\s*" + _MONTH_RE + r"\s*(?:A|AU)\s*(\d{1,2})?(?:ER)?\s*"
                   + _MONTH_RE, t)
    if sm:
        d0, mo0, d1, mo1 = sm.groups()
        m0, m1 = _month(mo0), _month(mo1)
        r.season = ((m0, int(d0) if d0 else 1), (m1, int(d1) if d1 else 31))
    return r


# ---------------------------------------------------------------------------- zones

def build_zones(S: pd.DataFrame, ints: pd.DataFrame, x_max: float,
                reserved_m: float = 6.5) -> pd.DataFrame:
    """Zones (x0, x1, rpa, kind, pole, rule) from the panels of one curb (see module doc).
    A reserved panel (disabled, ...) with no other pole of the same text in its block marks
    one space (`reserved_m`) in its arrow direction, centred without arrow: a lone panel
    run to the block end made a whole block a disabled zone (Hampton, 2026-10-26)."""
    i0, i1 = np.sort(ints.x0_m.values), np.sort(ints.x1_m.values)

    def block_of(x):
        lo = i1[i1 <= x].max() if np.any(i1 <= x) else 0.0
        hi = i0[i0 > x].min() if np.any(i0 > x) else x_max
        return lo, hi
    S = S.sort_values("x_ref").reset_index(drop=True)
    S["block"] = [block_of(x) for x in S.x_ref]
    S["dir"] = 0
    S["rule"] = ""
    arrowed = S.FLECHE_PAN.isin([2, 3])
    paired_poles = {}
    for (blk, rpa), g in S[arrowed & ~S.kind.eq("no_stopping")].groupby(["block", "DESCRIPTION_RPA"]):
        idx = list(g.drop_duplicates("POTEAU_ID_POT").index)
        for a, b in zip(idx[0::2], idx[1::2]):
            S.loc[a, ["dir", "rule"]] = (1, "pair")
            S.loc[b, ["dir", "rule"]] = (-1, "pair")
            paired_poles[S.loc[a, "POTEAU_ID_POT"]] = 1
            paired_poles[S.loc[b, "POTEAU_ID_POT"]] = -1
    for i in S.index[arrowed & (S.rule == "")]:
        pole = S.loc[i, "POTEAU_ID_POT"]
        if pole in paired_poles:
            S.loc[i, ["dir", "rule"]] = (-paired_poles[pole], "opposite_of_pair")
        else:
            lo, hi = S.loc[i, "block"]
            x = S.loc[i, "x_ref"]
            S.loc[i, ["dir", "rule"]] = (1 if hi - x < x - lo else -1, "to_cross_street")
    S.loc[~arrowed, "rule"] = "both_ways"
    rows = []
    for r in S.itertuples():
        lo, hi = r.block
        same = S[(S.DESCRIPTION_RPA == r.DESCRIPTION_RPA) & (S.POTEAU_ID_POT != r.POTEAU_ID_POT)]
        if r.kind == "reserved" and not same.x_ref.between(lo, hi).any():
            a, b = ((r.x_ref, r.x_ref + r.dir * reserved_m) if r.dir
                    else (r.x_ref - reserved_m / 2, r.x_ref + reserved_m / 2))
            rows.append({"x0": min(a, b), "x1": max(a, b), "rpa": r.DESCRIPTION_RPA,
                         "kind": r.kind, "pole": r.POTEAU_ID_POT, "rule": "one_space"})
            continue
        for d in ([r.dir] if r.dir else [1, -1]):
            end = hi if d > 0 else lo
            nxt = same.x_ref[(same.x_ref - r.x_ref) * d > 0]
            if len(nxt):
                end = min(end, nxt.min()) if d > 0 else max(end, nxt.max())
            rows.append({"x0": min(r.x_ref, end), "x1": max(r.x_ref, end),
                         "rpa": r.DESCRIPTION_RPA, "kind": r.kind, "pole": r.POTEAU_ID_POT,
                         "rule": r.rule})
    return pd.DataFrame(rows, columns=["x0", "x1", "rpa", "kind", "pole", "rule"])


PRIOR_DEFAULTS = {"corner_m": 45.0, "reserved_m": 6.5, "keep": []}


def convention_dir(arrow: int, curb: str) -> int:
    """Along-axis direction of the zone from its pole (Quebec convention, panels facing
    traffic): right curb 2 -> +1, 3 -> -1; left curb mirrored; 0 without an arrow."""
    if arrow not in (2, 3):
        return 0
    return (1 if arrow == 2 else -1) * (1 if curb == "right" else -1)


def build_zones_prior(S: pd.DataFrame, ints: pd.DataFrame, x_max: float, curb: str,
                      prior: dict, keep_x=(), keep_poles=()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Zones of one curb under the prior 'parking is allowed except intersection approaches
    and listed exceptions' (module doc). Returns (zones, ignored panels). No-stopping panels:
    inside a keep range / pole -> as signed (convention, to the next pole with the same text
    or the block end; no arrow: both ways); else arrowed within corner_m of a block end ->
    from the pole to that end; else ignored. Reserved panels: consecutive panels with the
    same text in a block bound a zone; a lone one is reserved_m in its arrow direction
    (centred without an arrow). Other kinds: `build_zones`."""
    S = S.sort_values("x_ref").reset_index(drop=True)
    i0, i1 = np.sort(ints.x0_m.values), np.sort(ints.x1_m.values)
    cols = ["x0", "x1", "rpa", "kind", "pole", "rule"]
    other = S[~S.kind.isin(["no_stopping", "reserved"])]
    rows = build_zones(other, ints, x_max).to_dict("records") if len(other) else []
    ignored = []

    def block_of(x):
        lo = i1[i1 <= x].max() if np.any(i1 <= x) else 0.0
        hi = i0[i0 > x].min() if np.any(i0 > x) else x_max
        return lo, hi

    def inside_int(x):
        return bool(np.any((ints.x0_m.values <= x) & (x <= ints.x1_m.values)))

    def kept(r):
        return r.POTEAU_ID_POT in keep_poles or any(a <= r.x_ref <= b for a, b in keep_x)
    for r in S[S.kind.eq("no_stopping")].itertuples():
        if inside_int(r.x_ref):
            continue   # at the corner itself: the intersection clearance covers it
        lo, hi = block_of(r.x_ref)
        d = convention_dir(int(r.FLECHE_PAN), curb)
        if kept(r):
            same = S[(S.DESCRIPTION_RPA == r.DESCRIPTION_RPA) & (S.POTEAU_ID_POT != r.POTEAU_ID_POT)]
            for dd in ([d] if d else [1, -1]):
                end = hi if dd > 0 else lo
                nxt = same.x_ref[(same.x_ref - r.x_ref) * dd > 0]
                if len(nxt):
                    end = min(end, nxt.min()) if dd > 0 else max(end, nxt.max())
                rows.append({"x0": min(r.x_ref, end), "x1": max(r.x_ref, end), "rpa": r.DESCRIPTION_RPA,
                             "kind": r.kind, "pole": r.POTEAU_ID_POT,
                             "rule": "kept_as_signed" if d else "kept_both_ways"})
            continue
        real_lo, real_hi = np.any(i1 <= r.x_ref), np.any(i0 > r.x_ref)
        d_lo = r.x_ref - lo if real_lo else np.inf
        d_hi = hi - r.x_ref if real_hi else np.inf
        if d and min(d_lo, d_hi) <= prior["corner_m"]:
            end = lo if d_lo < d_hi else hi
            rows.append({"x0": min(r.x_ref, end), "x1": max(r.x_ref, end), "rpa": r.DESCRIPTION_RPA,
                         "kind": r.kind, "pole": r.POTEAU_ID_POT, "rule": "corner"})
        else:
            ignored.append({"pole": r.POTEAU_ID_POT, "x_ref": r.x_ref, "rpa": r.DESCRIPTION_RPA,
                            "arrow": int(r.FLECHE_PAN),
                            "why": "no arrow, mid-block" if not d else "mid-block"})
    res = S[S.kind.eq("reserved")]
    for (blk, rpa), g in res.groupby([res.x_ref.map(block_of), "DESCRIPTION_RPA"]):
        xs = g.drop_duplicates("POTEAU_ID_POT")
        if len(xs) >= 2:
            for a, b in zip(xs.itertuples(), list(xs.itertuples())[1:]):
                rows.append({"x0": a.x_ref, "x1": b.x_ref, "rpa": rpa, "kind": "reserved",
                             "pole": a.POTEAU_ID_POT, "rule": "pair"})
            continue
        r = xs.iloc[0]
        d, w = convention_dir(int(r.FLECHE_PAN), curb), prior["reserved_m"]
        a, b = (r.x_ref, r.x_ref + d * w) if d else (r.x_ref - w / 2, r.x_ref + w / 2)
        rows.append({"x0": min(a, b), "x1": max(a, b), "rpa": rpa, "kind": "reserved",
                     "pole": r.POTEAU_ID_POT, "rule": "one_space"})
    return pd.DataFrame(rows, columns=cols), pd.DataFrame(
        ignored, columns=["pole", "x_ref", "rpa", "arrow", "why"])


def _zones_all_rows(S: pd.DataFrame, ints, x_max, rows: dict, prior: dict | None = None,
                    ignored: list | None = None) -> pd.DataFrame:
    """Common-schema zones from panels already assigned to rows (column `row`); a row's
    blocks end at the intersections that cut its curb. With `prior` (resolved keep ranges),
    ignored panels are appended to `ignored` (DataFrames with `row`)."""
    from ..regs.supply import cuts
    parts = []
    for r, g in S.groupby("row"):
        ic = cuts(ints, rows[r]["curb"])
        if prior is not None:
            z, ig = build_zones_prior(g, ic, x_max, rows[r]["curb"], prior,
                                      [k["x"] for k in prior["keep_r"]
                                       if k["x"] is not None and k["row"] in (None, r)],
                                      {k["pole"] for k in prior["keep_r"] if k.get("pole")})
            if ignored is not None and len(ig):
                ignored.append(ig.assign(row=r))
        else:
            z = build_zones(g, ic, x_max)
        parts.append(pd.DataFrame({"row": r, "x0": z.x0, "x1": z.x1, "kind": z.kind,
                                   "rule": z.rpa, "source": "montreal", "ref": z.pole,
                                   "text": z.rpa, "inference": z.rule}))
    return pd.concat(parts, ignore_index=True) if parts else \
        pd.DataFrame(columns=ZONE_COLS + ["inference"])


# ---------------------------------------------------------------------------- adapter

def load(sc: dict, ctx) -> Regs:
    """Signs and hydrants near the street, on the street axis and assigned to rows."""
    from ..regs.supply import fit_offset
    sc = {**DEFAULTS, **sc}
    lay = ctx.lay
    S = ctx.near(pd.read_csv(sc["signs_csv"], low_memory=False), "Latitude", "Longitude")
    S["FLECHE_PAN_DATA"] = S.FLECHE_PAN
    S = set_arrows(S, sc["pole_arrows"] or [])
    rules = {t: parse_rule(t) for t in S.DESCRIPTION_RPA.unique()}
    S["kind"] = S.DESCRIPTION_RPA.map(lambda t: rules[t].kind)
    S["row"] = ctx.curb_row(S.x_ref.values, S.lat_m.values)
    drop = {int(d["pole"]): d.get("note", "") for d in sc["drop_poles"] or []}
    dropped = S["row"].notna() & S.POTEAU_ID_POT.isin(drop)
    Sc = S[S["row"].notna() & ~dropped]
    H = pd.DataFrame(columns=["x_ref", "row"])
    if sc.get("hydrants_csv"):
        H = ctx.near(pd.read_csv(sc["hydrants_csv"], encoding="utf-8-sig", low_memory=False),
                     "LATITUDE", "LONGITUDE")
        H = H[H.STATUT_ACTIF.astype(str).str.upper().str.startswith("EXIST")]
        H["row"] = ctx.curb_row(H.x_ref.values, H.lat_m.values)
        H = H[H["row"].notna()]
    ints, x_max = lay.intersections, float(lay.extent_m[1])
    prior = _prior(sc.get("prior"), ints)
    if sc["arrows"] not in ("inferred", "convention"):
        raise ValueError(f"montreal source: arrows {sc['arrows']!r} (inferred | convention)")
    as_signed = prior is None and sc["arrows"] == "convention"
    if as_signed:   # every panel as signed, Quebec convention: the prior machinery, all kept
        prior = {**PRIOR_DEFAULTS, "keep_r": [{"row": None, "note": "as signed", "x": (-np.inf, np.inf)}]}
    dx, frac = 0.0, None
    if ctx.cars is not None and len(ctx.cars):
        # a pinned offset keeps a correction of one sign from shifting every other sign
        d0 = sc["offset_m"] or 0.0
        dx, frac = fit_offset(lambda d: _zones_all_rows(Sc.assign(x_ref=Sc.x_ref + d0 + d), ints,
                                                        x_max, lay.rows, prior),
                              ctx.cars, 0.0 if sc["offset_m"] is not None else sc["offset_search_m"])
        dx += d0
    elif sc["offset_m"] is not None:
        dx = float(sc["offset_m"])
    Sc = Sc.assign(x_ref=Sc.x_ref + dx)
    moved = {int(d["pole"]): float(d["x_m"]) for d in sc["pole_positions"] or []}
    Sc.loc[Sc.POTEAU_ID_POT.isin(moved), "x_ref"] = Sc.POTEAU_ID_POT.map(moved)
    H = H.assign(x_ref=H.x_ref + dx)
    ign: list = []
    zones = _zones_all_rows(Sc, ints, x_max, lay.rows, prior, ign)
    D = S[dropped]
    ign.append(pd.DataFrame({"pole": D.POTEAU_ID_POT, "x_ref": D.x_ref + dx,
                             "rpa": D.DESCRIPTION_RPA, "arrow": D.FLECHE_PAN,
                             "why": [f"dropped (site check): {drop[p]}" for p in D.POTEAU_ID_POT],
                             "row": D["row"]}))
    ign = pd.concat([g for g in ign if len(g)], ignore_index=True) if any(len(g) for g in ign) \
        else pd.DataFrame(columns=["pole", "x_ref", "rpa", "arrow", "why", "row"])
    points = pd.DataFrame({"row": H["row"].values, "x_ref": H.x_ref.values, "kind": "hydrant",
                           "source": "montreal",
                           "ref": H["ID_BI"].values if "ID_BI" in H else ""},
                          columns=POINT_COLS)
    info = {"residual_offset_m": dx,
            "car_length_in_no_stopping": None if frac is None else round(frac, 3),
            "n_panels": int(len(S)), "n_panels_on_parked_curbs": int(len(Sc)),
            "n_hydrants_on_parked_curbs": int(len(H)),
            "poles_moved": moved,
            # corrections of the recorded data (site checks), reported on the public map
            "poles_dropped": sorted({int(p) for p in D.POTEAU_ID_POT}),
            "poles_arrows": sorted({int(f["pole"]) for f in sc["pole_arrows"] or []}),
            "unparsed": sorted(t for t, r in rules.items() if r.kind == "unknown"),
            "arrows": "convention" if as_signed or prior is not None else "inferred",
            "prior": None if prior is None or as_signed else {
                "corner_m": prior["corner_m"], "reserved_m": prior["reserved_m"],
                "keep": prior["keep_r"], "n_ignored": int(len(ign))}}
    signs = S.assign(x_ref_fit=S.x_ref + dx, parked_curb=S["row"].notna())
    signs.loc[signs.POTEAU_ID_POT.isin(moved), "x_ref_fit"] = signs.POTEAU_ID_POT.map(moved)
    return Regs(zones=zones, points=points, rules=rules, info=info,
                tables={"signs_ref": signs, "hydrants_ref": H, "signs_ignored": ign},
                clearances_m=dict(CLEARANCES_M))


def set_arrows(S: pd.DataFrame, fixes: list[dict]) -> pd.DataFrame:
    """Arrow codes corrected by site checks: {pole, arrow, rpa (optional)}. Without `rpa` the
    pole must carry exactly one arrowed panel (or one panel)."""
    S = S.copy()
    for f in fixes:
        m = S.POTEAU_ID_POT == int(f["pole"])
        if f.get("rpa") is not None:
            m &= S.DESCRIPTION_RPA == f["rpa"]
        elif (m & S.FLECHE_PAN.isin([2, 3])).sum() > 1:
            raise ValueError(f"pole_arrows: pole {f['pole']} has several arrowed panels, give `rpa`")
        elif (m & S.FLECHE_PAN.isin([2, 3])).any():
            m &= S.FLECHE_PAN.isin([2, 3])
        if not m.any():
            raise ValueError(f"pole_arrows: no panel {f.get('rpa') or ''} on pole {f['pole']} near the street")
        if m.sum() > 1:
            raise ValueError(f"pole_arrows: pole {f['pole']}: {int(m.sum())} panels match, give `rpa`")
        S.loc[m, "FLECHE_PAN"] = int(f["arrow"])
    return S


def _prior(pc: dict | None, ints: pd.DataFrame) -> dict | None:
    """Prior config with keep entries resolved to street-axis ranges: `between` two cross
    streets (OSM names, their centres), `x_m`, or a `pole` id; `row` optional."""
    if pc is None:
        return None
    p = {**PRIOR_DEFAULTS, **pc}
    names = dict(zip(ints["name"].fillna(""), ints.centre_m)) if "name" in ints else {}
    keep = []
    for k in p["keep"] or []:
        e = {"row": k.get("row"), "note": k.get("note", ""), "x": None}
        if "between" in k:
            missing = [n for n in k["between"] if n not in names]
            if missing:
                raise ValueError(f"prior.keep between {missing}: cross streets are {sorted(names)}")
            a, b = sorted(names[n] for n in k["between"])
            e["x"] = (float(a), float(b))
        elif "x_m" in k:
            e["x"] = tuple(float(v) for v in sorted(k["x_m"]))
        if "pole" in k:
            e["pole"] = int(k["pole"])
        keep.append(e)
    p["keep_r"] = keep
    return p


def fetch_source(sc: dict, force: bool = False):
    """`curb-occupancy fetch`: download the source's CSVs when missing (all with force)."""
    paths = [Path(sc[k]) for k in ("signs_csv", "hydrants_csv") if sc.get(k)]
    if force or not all(p.exists() for p in paths):
        fetch(paths[0].parent, sc)


def fetch(out_dir: str | Path, sc: dict | None = None):
    """Download the signage and hydrant CSVs (follows the portal's signed-URL redirect). With
    a source config, to its signs_csv / hydrants_csv paths."""
    import urllib.request
    out = Path(out_dir)
    targets = [(SIGNS_URL, out / "signalisation_stationnement.csv"),
               (HYDRANTS_URL, out / "bornes_incendie.csv")]
    if sc:
        targets = [(u, Path(sc[k])) for u, k in ((SIGNS_URL, "signs_csv"),
                                                 (HYDRANTS_URL, "hydrants_csv")) if sc.get(k)]
    for url, path in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "curb-occupancy"})
        with urllib.request.urlopen(req, timeout=300) as r, open(path, "wb") as f:
            f.write(r.read())
        print(path)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "fetch":
        fetch(sys.argv[2])
    else:
        print("usage: python -m curb_occupancy.adapters.montreal fetch <dir>")
