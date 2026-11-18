"""Area stage: several street runs merged into one set of tables and one public map.

Each street keeps its own run (own axis, layout, supply); an area config lists the street
configs and this stage reads their outputs (`supply/`, `multiday/cars_ref.csv`,
`map/layers.geojson`), so nothing is recomputed. Stretch and pass ids are prefixed with the
street key (config file stem) to stay unique.

    # config/streets/side/_area.yaml
    name: side_streets
    title: Side streets around rue de Terrebonne
    streets: [wilson.yaml, draper.yaml, ...]   # relative to this file

Writes <out_dir>/ (default workdir/runs/<name>):
* `blocks_by_pass.csv`: every block x row x pass of every street (supply's
  occupancy_by_pass + street, side; uncovered blocks get `n_parked_seen`, the vehicles seen
  on the part the pass covered: a lower bound, never in the totals);
* `stretches.csv`: guideline stretches (evening means) per street;
* with `guideline.slots` (all streets): `stretches_slots.csv` (per stretch and view: `avg` =
  time-weighted average, `slot:<key>` = slot means) and the map's views over them;
* `summary.json`, `report.md`: area totals over the covered stretches, per street;
* with a `distribution:` section (needs `guideline.slots`): `stretch_evenings.csv` (one row per
  stretch and evening: passes of one evening on one street averaged) and the class shares of
  the configured bars (`distribution()`), drawn as stacked bars on the map page (above the
  stretch table);
* `public.html` + `layers.geojson`: the public map (`webmap.html`) over all streets;
* with a `capacity:` section (`capacity.py`): `capacity_faces.csv` (every block face of the
  listed streets + the other streets' covered blocks, measured or estimated) and the map's
  capacity view (first view), area total with a prediction interval in `summary.json`.

`title` may be {en: ..., fr: ...}: the map shows the viewer's language (EN / FR toggle).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .. import config as C


def load(path: str | Path) -> dict:
    path = Path(path)
    a = yaml.safe_load(path.read_text()) or {}
    a["name"] = a.get("name") or path.stem
    a["out_dir"] = a.get("out_dir") or f"workdir/runs/{a['name']}"
    a["streets"] = [str((path.parent / s).resolve()) for s in a.get("streets") or []]
    a["_path"] = str(path)
    return a


def is_area(path: str | Path) -> bool:
    return "streets" in (yaml.safe_load(Path(path).read_text()) or {})


def _street(path: str) -> dict | None:
    cfg = C.load(path)
    root = Path(cfg["out_dir"])
    need = [root / "supply" / f for f in ("supply.json", "occupancy_by_pass.csv",
                                          "stretches.csv", "occupancy_stretches_evening.csv")]
    need += [root / "map" / "layers.geojson", root / "multiday" / "cars_ref.csv"]
    missing = [str(f) for f in need if not f.exists()]
    rows = cfg["street"]["rows"]
    return {"key": Path(path).stem, "cfg": cfg, "root": root, "missing": missing,
            "name": cfg["street"].get("name") or cfg["name"],
            "sides": {r: (d or {}).get("label") or r for r, d in rows.items() if d}}


def _attributions(streets: list[dict]) -> list[dict]:
    from ..regs.supply import attributions
    out = []
    for s in streets:
        out += [a for a in attributions(s["cfg"]) if a not in out]
    return out


def _short(name: str) -> str:
    """'Avenue de Hampton' -> 'Hampton' (map labels; the full name stays in the tables)."""
    for p in ("Avenue de ", "Avenue d’", "Avenue d'", "Avenue ", "Rue de ", "Rue ", "Boulevard "):
        if name.startswith(p):
            return name[len(p):]
    return name


def build(area: dict) -> dict:
    out = Path(area["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    streets, skipped = [], []
    for p in area["streets"]:
        s = _street(p)
        (skipped if s["missing"] else streets).append(s)
    if not streets:
        return {"streets": 0, "skipped": [(s["key"], s["missing"]) for s in skipped]}

    blocks, stretches, feats, passes, ev_ids, guides, evs = [], [], [], [], [], [], []
    slot_rows, sp_rows, reg_sources = [], [], []
    for order, s in enumerate(streets):
        root, key = s["root"], s["key"]
        sj = json.loads((root / "supply" / "supply.json").read_text())
        guides.append(sj["guideline"])
        reg_sources.append((s["name"], sj.get("sources") or {}))
        evs.append((sj.get("occupancy") or {}).get("evening") or {})
        pid = lambda c: f"{key}/{c}"  # noqa: E731

        # blocks x row x pass; vehicles seen on uncovered blocks (lower bound)
        o = pd.read_csv(root / "supply" / "occupancy_by_pass.csv")
        cars = pd.read_csv(root / "multiday" / "cars_ref.csv")
        o["n_parked_seen"] = [
            np.nan if r.covered else int(((cars["clip"] == r.clip) & (cars["row"] == r.row)
                                          & cars.xr_mid_m.between(r.x0_ref_m, r.x1_ref_m)).sum())
            for r in o.itertuples()]
        bk = pd.read_csv(root / "supply" / "blocks.csv")[["block_id", "from", "to"]]
        o = o.merge(bk, on="block_id", how="left")
        cl = pd.read_csv(root / "supply" / "capacity_by_pass.csv")
        o = o.merge(cl[["clip", "block_id", "row", "row_length_m"]], how="left",
                    on=["clip", "block_id", "row"])
        o["key"] = key
        o.insert(0, "side", o.row.map(s["sides"]))
        o.insert(0, "street", s["name"])
        blocks.append(o)

        # stretches: evening means where the evening passes cover them
        st = pd.read_csv(root / "supply" / "stretches.csv")
        se = pd.read_csv(root / "supply" / "occupancy_stretches_evening.csv")
        st = st.merge(se, on="stretch_id", how="left")
        st.insert(0, "side", ", ".join(s["sides"].values()))
        st.insert(0, "street", s["name"])
        st["stretch_id"] = key + ":" + st.stretch_id
        stretches.append(st)
        sp = pd.read_csv(root / "supply" / "occupancy_stretches_by_pass.csv")
        sp.insert(0, "street", s["name"])
        sp["stretch_id"] = key + ":" + sp.stretch_id
        sp_rows.append(sp)
        sl = root / "supply" / "occupancy_stretches_slots.csv"
        if sl.exists() and sl.stat().st_size > 1:
            q = pd.read_csv(sl)
            q.insert(0, "street", s["name"])
            q["stretch_id"] = key + ":" + q.stretch_id
            slot_rows.append(q)

        # map layers: prefix stretch and pass ids ('evening' stays: one evening view)
        g = json.loads((root / "map" / "layers.geojson").read_text())
        for f in g["features"]:
            q = dict(f["properties"])
            q["street"] = key
            if q.get("stretch"):
                q["stretch"] = f"{key}:{q['stretch']}"
            if q.get("clip") and not is_view(q["clip"]):
                q["clip"] = pid(q["clip"])
            if q["layer"] == "stretches":
                ends = [_short(q["name"].split(" – ")[k]) for k in (0, -1)]
                q["name"] = f"{_short(s['name'])} ({', '.join(s['sides'].values())}): " \
                            f"{ends[0]} – {ends[1]}"
                # parts of the name, for the page to translate the side names
                q["parts"] = {"street": _short(s["name"]), "sides": list(s["sides"].values()),
                              "from": ends[0], "to": ends[1]}
                q["per_pass"] = {pid(c): v for c, v in (q.get("per_pass") or {}).items()}
                q["order"] = order * 100000 + q["x0"]  # table: by street, then along it
            feats.append({**f, "properties": q})
        for c in sorted(o["clip"].unique()):
            w = datetime.fromisoformat(o.loc[o["clip"] == c, "when"].iloc[0])
            passes.append({"id": pid(c), "when": w.isoformat(timespec="minutes"),
                           "street": _short(s["name"]),
                           "label": f"{_short(s['name'])} {w:%a %d %b %H:%M}"})
            if o.loc[o["clip"] == c, "in_window"].any():
                ev_ids.append(pid(c))

    if any(g != guides[0] for g in guides):
        raise ValueError("area: the streets use different guideline settings")
    g = guides[0]
    B = pd.concat(blocks, ignore_index=True)
    S = pd.concat(stretches, ignore_index=True)
    B.to_csv(out / "blocks_by_pass.csv", index=False)
    S.to_csv(out / "stretches.csv", index=False)

    # totals over the covered stretches (evening means), per street and for the area
    cov = S[S.capacity_mean.notna()]
    per_street = []
    for s in streets:
        q, b = cov[cov.street == s["name"]], B[(B.street == s["name"]) & B.in_window]
        per_street.append({
            "street": s["name"], "key": s["key"], "sides": list(s["sides"].values()),
            "parked": float(q.parked_mean.sum()), "capacity": float(q.capacity_mean.sum()),
            "occupancy": round(q.parked_mean.sum() / q.capacity_mean.sum(), 3)
            if q.capacity_mean.sum() else None,
            "stretches": int(len(S[S.street == s["name"]])), "stretches_covered": int(len(q)),
            "blocks_uncovered": b.loc[~b.covered, "block_id"].tolist()})
    cap, n = float(cov.capacity_mean.sum()), float(cov.parked_mean.sum())
    dates = sorted(B.loc[B.in_window, "date"].unique())
    from ..regs.supply import slot_totals
    from ..publish.webmap import aggregates, regs_summary
    SL = pd.concat(slot_rows, ignore_index=True) if slot_rows else pd.DataFrame()
    aggs = []
    if g.get("slots") and len(SL):
        SL.to_csv(out / "stretches_slots.csv", index=False)
        when = {p["id"]: datetime.fromisoformat(p["when"]) for p in passes if p["id"] in ev_ids}
        aggs = [{k: v for k, v in a.items() if k != "pass_weight"} for a in aggregates(g, when)]
        for s in per_street:
            s["slots"] = slot_totals(SL[SL.street == s["street"]])
    summary = {
        "name": area["name"], "title": _title(area.get("title")), "streets": per_street,
        "skipped": [{"key": s["key"], "missing": s["missing"]} for s in skipped],
        "evening": {"parked": n, "capacity": cap,
                    "occupancy": round(n / cap, 3) if cap else None,
                    "stretches_covered": int(len(cov)), "stretches": int(len(S)),
                    "stretches_by_class": cov["class"].value_counts().to_dict(),
                    "dates": dates,
                    "n_dates_min": min((e.get("n_dates") or 0) for e in evs),
                    "meets_n_evenings": all(e.get("meets_n_evenings") for e in evs)},
        "slots": slot_totals(SL) if aggs else None,
        "driveways_labelled": [s["key"] for s in streets
                               if any(t.get("type") == "manual"
                                      for t in s["cfg"].get("regulations", {}).get("sources", []))],
        "guideline": g,
    }
    # class distribution of stretch-evenings (`distribution` section)
    dist = None
    if area.get("distribution"):
        if not g.get("slots"):
            raise ValueError("area distribution: needs guideline.slots")
        U = stretch_evenings(pd.concat(sp_rows, ignore_index=True), g)
        U.to_csv(out / "stretch_evenings.csv", index=False, float_format="%.3f")
        dist = distribution(U, area["distribution"], g)
        summary["distribution"] = dist
    # capacity block face by block face: measured + estimated (`capacity` section)
    cap_feats = []
    if area.get("capacity"):
        from .. import capacity as K
        Fc, csum = K.build(area["capacity"], B)
        Fc.drop(columns=["line"], errors="ignore").to_csv(out / "capacity_faces.csv", index=False)
        summary["capacity"] = csum
        cap_feats = _capacity_features(Fc, feats, streets)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out / "report.md").write_text(_report(summary, S, B))

    # public map: the street page's template over all streets
    pub = [f for f in feats if f["properties"]["layer"] in ("status", "stretches", "cars")]
    pub += cap_feats
    (out / "layers.geojson").write_text(json.dumps({"type": "FeatureCollection",
                                                    "features": pub}))
    ll = np.array([c for f in pub if f["geometry"]["type"] == "LineString"
                   for c in f["geometry"]["coordinates"]])
    layers = {k: {"type": "FeatureCollection",
                  "features": [f for f in pub if f["properties"]["layer"] == k]}
              for k in ("status", "stretches", "cars", "capacity")}
    # the map's pass selector lists evenings (one recording each), not street passes
    evenings = []
    for d in sorted({p["when"][:10] for p in passes}):
        ps = sorted((p for p in passes if p["when"][:10] == d), key=lambda p: p["when"])
        t0, t1 = (datetime.fromisoformat(ps[k]["when"]) for k in (0, -1))
        span = f"{t0:%H:%M}" if t0 == t1 else f"{t0:%H:%M}–{t1:%H:%M}"
        n = f"{len(ps)} streets" if len(ps) > 1 else _short(ps[0]["label"].rsplit(" ", 4)[0])
        evenings.append({"id": f"ev:{d}", "passes": [p["id"] for p in ps],
                         "label": f"{t0:%a %d %b} {span} ({n})", "date": d,
                         "t0": f"{t0:%H:%M}", "t1": f"{t1:%H:%M}",
                         "streets": sorted({p["street"] for p in ps})})
    meta = {"name": area["name"], "street": area.get("title") or area["name"],
            "capacity": summary.get("capacity") and {
                **{k: v for k, v in summary["capacity"].items() if k != "streets"},
                "note": area["capacity"].get("note")},
            "mode": "public", "passes": passes, "evening_passes": ev_ids, "evenings": evenings,
            "aggregates": aggs, "distribution": dist,
            "guideline": g,
            "occupancy": {"evening": {"n_dates": summary["evening"]["n_dates_min"],
                                      "meets_n_evenings": summary["evening"]["meets_n_evenings"]}},
            "driveways_labelled": len(summary["driveways_labelled"]) == len(streets),
            "prior": None, "regs": regs_summary(reg_sources),
            "attribution": _attributions(streets),
            "generated": datetime.now().isoformat(timespec="minutes"),
            "center": [float(ll[:, 1].mean()), float(ll[:, 0].mean())],
            "area": {"streets": [{"street": s["name"], "sides": list(s["sides"].values())}
                                 for s in streets]}}
    tpl = (Path(__file__).parent / "webmap.html").read_text()
    (out / "public.html").write_text(tpl.replace(
        "/*__DATA__*/null", json.dumps({"meta": meta, "layers": layers, "checks": []},
                                       ensure_ascii=False, default=str)))
    return {"streets": len(streets), "skipped": [s["key"] for s in skipped],
            "occupancy": summary["evening"]["occupancy"], "out": str(out)}


def stretch_evenings(sp: pd.DataFrame, g: dict) -> pd.DataFrame:
    """One row per stretch and evening (date) in the window: the passes of one evening on one
    street averaged (parked, capacity), as the slot means do; only passes covering the
    stretch (`counted`) with capacity. Columns street, stretch_id, date, slot, n_passes,
    parked, capacity, occupancy, class."""
    from ..regs.guideline import occupancy_class, slot_of
    d = sp[sp.in_window & sp.counted & (sp.capacity > 0)].copy()
    d["slot"] = [slot_of(w, g["slots"]) for w in d["when"]]
    d = d[d.slot.notna()]
    U = (d.groupby(["street", "stretch_id", "date", "slot"], as_index=False)
         .agg(n_passes=("clip", "size"), parked=("n_parked", "mean"), capacity=("capacity", "mean")))
    U["occupancy"] = U.parked / U.capacity
    U["class"] = [occupancy_class(o, g["classes"]) for o in U.occupancy]
    return U


def distribution(U: pd.DataFrame, dc: dict, g: dict) -> list[dict]:
    """Class shares of stretch-evenings per configured bar: {streets | exclude: [street names],
    view: 'slot:<key>' | 'avg', label, example}. `example: true` = the bar the map page's note
    takes its worked example from. A slot view counts each stretch-evening once; 'avg'
    gives each stretch weight 1, split over its slots by the slot weights (renormalised over
    the slots it was seen in, as the time-weighted average) and within a slot equally over
    its evenings."""
    W = {s["key"]: s["weight"] for s in g["slots"]}
    keys = [k for _, k, _ in g["classes"]]
    out = []
    for b in dc.get("bars") or []:
        u = U[U.street.isin(b["streets"])] if b.get("streets") else U
        u = u[~u.street.isin(b.get("exclude") or [])]
        v = b.get("view", "avg")
        if v.startswith("slot:"):
            u = u[u.slot == v[5:]].assign(w=1.0)
        elif v == "avg":
            n = u.groupby(["stretch_id", "slot"]).date.transform("size")
            tot = u.drop_duplicates(["stretch_id", "slot"]).assign(ws=lambda q: q.slot.map(W)) \
                .groupby("stretch_id").ws.sum()
            u = u.assign(w=u.slot.map(W) / n / u.stretch_id.map(tot))
        else:
            raise ValueError(f"distribution bar view {v!r}: 'avg' or 'slot:<key>'")
        if b.get("streets") and not len(u):
            raise ValueError(f"distribution bar {b.get('label')}: no stretch-evening")
        sw = u.groupby("class").w.sum()
        out.append({"label": b.get("label"), "view": v,
                    "shares": {k: round(float(sw.get(k, 0.0) / u.w.sum()), 4) if len(u) else 0.0
                               for k in keys},
                    "units": {k: int((u["class"] == k).sum()) for k in keys},
                    "n_units": int(len(u)), "n_stretches": int(u.stretch_id.nunique()),
                    "n_dates": int(u.date.nunique()), "n_streets": int(u.street.nunique()),
                    "weighted": v == "avg", "example": bool(b.get("example"))})
    return out


def is_view(clip: str) -> bool:
    """Pseudo-pass ids of the aggregate views (not prefixed with the street key)."""
    return clip in ("evening", "avg") or clip.startswith("slot:")


def _title(t):
    """Area title: a string, or {en, fr, ...} (the map page picks the viewer's language)."""
    return (t.get("en") or next(iter(t.values()), None)) if isinstance(t, dict) else t


def _capacity_features(F: pd.DataFrame, feats: list, streets: list) -> list:
    """Map features of the capacity faces. Measured: the block's curb as drawn in the
    evening view (else its first pass), whose status pieces get the face id (`face`);
    estimated: the OSM face line."""
    order = {s["name"]: i for i, s in enumerate(streets)}
    status = [f for f in feats if f["properties"]["layer"] == "status"]
    clip_of, rank = {}, {"avg": 0, "evening": 1}
    for f in status:  # per street: the time-weighted view, else 'evening', else its first pass
        k, c = f["properties"]["street"], f["properties"]["clip"]
        if k not in clip_of or rank.get(c, 9) < rank.get(clip_of[k], 9):
            clip_of[k] = c
    out = []
    for i, r in enumerate(F.to_dict("records")):
        fid = f"F{i:03d}"
        props = {"layer": "capacity", "face": fid, "kind": r["kind"], "street": _short(r["street"]),
                 "side": r["side"], "from": _short(r["from"]), "to": _short(r["to"]),
                 "spaces": int(r["spaces"]), "length_m": round(float(r["length_m"])),
                 "restricted_m": round(float(r.get("restricted_m") or 0)),
                 "order": order.get(r["street"], len(order)) * 1000 + i}
        if r["kind"] == "measured":
            pieces = sorted((f for f in status if f["properties"]["street"] == r["key"]
                             and f["properties"]["clip"] == clip_of.get(r["key"])
                             and f["properties"]["block"] == r["block_id"]
                             and f["properties"]["row"] == r["row"]),
                            key=lambda f: f["properties"]["x0"])
            if not pieces:
                continue
            for f in pieces:
                f["properties"]["face"] = fid
            line = [c for f in pieces for c in f["geometry"]["coordinates"]]
        else:
            line = r["line"]
        out.append({"type": "Feature", "geometry": {"type": "LineString", "coordinates": line},
                    "properties": props})
    return out


def _pct(v) -> str:
    return "–" if v is None or pd.isna(v) else f"{100 * v:.0f} %"


def _report(sm: dict, S: pd.DataFrame, B: pd.DataFrame) -> str:
    e = sm["evening"]
    L = [f"# {sm['title'] or sm['name']}", "",
         f"Evening occupancy over the covered stretches: **{_pct(e['occupancy'])}** "
         f"({e['parked']:g} parked / {e['capacity']:g} spaces, {e['stretches_covered']} of "
         f"{e['stretches']} stretches; dates {', '.join(e['dates'])}).", ""]
    if len(sm["driveways_labelled"]) < len(sm["streets"]):
        L += ["Driveways not deducted on: " + ", ".join(
            s["street"] for s in sm["streets"] if s["key"] not in sm["driveways_labelled"])
            + " (capacity high, occupancy low).", ""]
    if sm["skipped"]:
        L += ["Skipped (run outputs missing): " + ", ".join(s["key"] for s in sm["skipped"]), ""]
    if sm.get("slots"):
        sl = {x["key"]: x for x in sm["guideline"]["slots"]}
        head = ["avg"] + [f"slot:{k}" for k in sl if f"slot:{k}" in sm["slots"]]
        name = {"avg": "time-weighted", **{f"slot:{k}": f"{k} {v['hours'][0]:g}-{v['hours'][1]:g} h"
                                           for k, v in sl.items()}}
        a = sm["slots"]["avg"]
        L += [f"Time-weighted occupancy (slots weighted by their hours: "
              + ", ".join(f"{k} {100 * v['weight']:.0f} %" for k, v in sl.items())
              + f"): **{_pct(a['occupancy'])}** ({a['parked']:g} / {a['capacity']:g} space-hour "
              f"weighted spaces; {a['one_slot']} of {a['stretches']} stretches seen in one slot "
              "only, which then stands for the whole window).", "",
              "| street | " + " | ".join(name[h] for h in head) + " |",
              "|---|" + "---|" * len(head)]
        for s in sm["streets"]:
            v = s.get("slots") or {}
            L.append(f"| {s['street']} | " + " | ".join(
                f"{_pct(v[h]['occupancy'])} ({v[h]['parked']:g} / {v[h]['capacity']:g})"
                if h in v else "–" for h in head) + " |")
        L.append("| **area** | " + " | ".join(
            f"**{_pct(sm['slots'][h]['occupancy'])}**" if h in sm["slots"] else "–" for h in head) + " |")
        L.append("")
    L += ["## Per street", "", "| street | side | parked / spaces | occupancy | stretches covered |",
          "|---|---|---|---|---|"]
    for s in sm["streets"]:
        L.append(f"| {s['street']} | {', '.join(s['sides'])} | {s['parked']:g} / "
                 f"{s['capacity']:g} | {_pct(s['occupancy'])} | {s['stretches_covered']} / "
                 f"{s['stretches']} |")
    k = sm.get("capacity")
    if k:
        L += ["", "## Capacity (block faces, measured + estimated)", "",
              f"Area capacity **{k['total']}** spaces ({100 * k['interval']:.0f} % prediction "
              f"interval {k['low']}–{k['high']}): {k['measured']['spaces']} measured on "
              f"{k['measured']['faces']} block faces + {k['estimated']['spaces']:.0f} estimated on "
              f"{k['estimated']['faces']} faces ({k['estimated']['length_m']} m of curb, of which "
              f"{k['estimated'].get('restricted_m', 0)} m on {k['estimated'].get('faces_restricted', 0)} "
              "faces under large signed restrictions, left out) at "
              f"{k['rate_per_km']} spaces / km ({k['m_per_space']} m / space; sample "
              f"{k['sample']['faces']} faces, {k['sample']['spaces']} spaces / "
              f"{k['sample']['length_m']} m, CV {k['sample']['rate_cv']}). The interval treats "
              "the sampled faces as a random draw.", "",
              "| street | measured | estimated | faces measured / estimated |", "|---|---|---|---|"]
        for r in k["streets"]:
            L.append(f"| {r['street']} | {r['measured']:g} | {r['estimated']:.0f} | "
                     f"{r['faces_measured']} / {r['faces_estimated']} |")
    L += ["", "## Blocks (evening passes)", "",
          "| street | side | block | pass | covered | parked / spaces | occupancy |",
          "|---|---|---|---|---|---|---|"]
    # partial end blocks have no cross street on their open side
    ends = B[B.in_window].rename(columns={"from": "frm"}).fillna({"frm": "(end)", "to": "(end)"})
    for r in ends.itertuples():
        n = (f"{r.n_parked:g} / {r.capacity}" if r.covered
             else f"({r.n_parked_seen:g} seen) / {r.capacity}")
        L.append(f"| {r.street} | {r.side} | {r.frm} – {r.to} | {r.when[5:].replace('T', ' ')} | "
                 f"{'yes' if r.covered else 'no'} | {n} | {_pct(r.occupancy)} |")
    if sm.get("distribution"):
        cls = [(k, lab) for _, k, lab in sm["guideline"]["classes"]]
        L += ["", "## Stretch-evenings by class", "",
              "One unit = one stretch on one evening (passes of the same evening on a street "
              "averaged). Time-weighted: each stretch counts once, split over its slots by the slot "
              "weights.", "",
              "| bar | " + " | ".join(lab for _, lab in cls) + " | stretch-evenings | stretches | evenings |",
              "|---|" + "---:|" * (len(cls) + 3)]
        for b in sm["distribution"]:
            L.append(f"| {_title(b['label'])} | " + " | ".join(
                f"{_pct(b['shares'][k])} ({b['units'][k]})" for k, _ in cls)
                + f" | {b['n_units']} | {b['n_stretches']} | {b['n_dates']} |")
    return "\n".join(L) + "\n"
