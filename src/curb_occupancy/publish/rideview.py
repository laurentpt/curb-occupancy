"""Ride view: one picture per pass for people who do not read mosaics. The street is cut into
blocks (cross street to cross street) in riding order; each block x parking row is one strip
showing only the parked row, from the mosaic plane that images it on that pass (near view on
the rider's side, far view across the street), on a common metric scale so both views splice
at the same car size. Cross-street names at the strip ends (intersections shaded), parked
vehicles boxed and numbered, the block's count against its guideline capacity, and a curb bar
under the strip: where parking is allowed, short-term only, forbidden, or a driveway/hydrant.

`curb-occupancy map` writes `<out_dir>/map/ride_<clip>.jpg` (private: vehicle imagery, not anonymised).

    map:
      ride:
        px_per_m: 50       # image scale (both axes)
        y_m: [-0.6, 2.4]   # heights above the road shown (at the parked row)
        pad_m: 4.0         # street shown past each block end (cross-street centre)
    street:
      rows:
        A: {curb: right, label: south side, ...}   # side names (default: compass side)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULTS = {"px_per_m": 50, "y_m": [-0.6, 2.4], "pad_m": 4.0, "jpeg_q": 85,
            "min_present_m": 5.0}

# RGB
CAR = (255, 212, 0)
CAR_OTHER = (200, 200, 200)
CURB = {"allowed": (46, 160, 67), "short": (240, 170, 0), "forbidden": (214, 39, 40),
        "cut": (120, 120, 120)}
CURB_TEXT = {"allowed": "parking allowed (counted as spaces)", "short": "short-term only",
             "forbidden": "no parking / no stopping", "cut": "driveway or hydrant"}
BG, INK, MUTED = (255, 255, 255), (30, 30, 30), (110, 110, 110)
SIDE_H, BAR_H, GAP = 34, 12, 22


def curb_class(st: np.ndarray, g: dict) -> np.ndarray:
    """Curb status -> display class ('' = intersection / not on the street)."""
    from ..regs.supply import parkable
    out = np.full(st.shape, "forbidden", dtype=object)
    out[st == "limited"] = "short"
    out[np.isin(st, ["driveway", "hydrant"])] = "cut"
    out[parkable(st, g)] = "allowed"
    out[np.isin(st, ["intersection", "absent"]) | pd.isna(st)] = ""
    return out


def compass_side(ax, x0: float, x1: float, curb: str) -> str:
    """'north side' etc. of the row's curb: the outward normal of the street axis there."""
    la, lo = ax.latlon(np.array([x0, x1], float))
    b = np.degrees(np.arctan2((lo[1] - lo[0]) * np.cos(np.radians(la[0])), la[1] - la[0]))
    n = (b + (90 if curb == "right" else -90)) % 360
    names = ["north", "north-east", "east", "south-east", "south", "south-west", "west",
             "north-west"]
    return names[int((n + 22.5) // 45) % 8] + " side"


def _fonts():
    from PIL import ImageFont
    try:
        from matplotlib import font_manager as fm
        reg = fm.findfont("DejaVu Sans")
        bold = fm.findfont(fm.FontProperties(family="DejaVu Sans", weight="bold"))
        return {k: ImageFont.truetype(f, s) for k, f, s in
                (("title", bold, 26), ("b", bold, 17), ("r", reg, 16), ("s", reg, 13))}
    except Exception:  # noqa: BLE001 - any font problem: Pillow's default
        return {k: ImageFont.load_default() for k in ("title", "b", "r", "s")}


def _travel_name(name) -> str:
    return "" if name is None or (isinstance(name, float) and name != name) else str(name)


class _PassView:
    """Mosaics of one pass resampled onto street-axis stretches in riding order."""

    def __init__(self, cfg: dict, lay, root: Path, clip: str, S: float, y_m):
        from ..config import height
        from .webmap import ClipPos
        self.cfg, self.lay, self.root, self.clip, self.S = cfg, lay, root, clip, S
        self.h = height(cfg, clip)
        self.pos = ClipPos(root, lay, clip)
        self.sgn = -1.0 if lay.reversed(clip) else 1.0
        self.ylo, self.yhi = y_m
        self.n_rows = int(round((self.yhi - self.ylo) * S))
        self.mosaics: dict = {}

    def mosaic(self, plane: str):
        from ..mosaic import Mosaic
        if plane not in self.mosaics:
            stem = self.root / self.clip / f"mosaic_{plane}"
            self.mosaics[plane] = Mosaic.load(stem) if stem.with_suffix(".png").exists() \
                else None
        return self.mosaics[plane]

    def xs(self, xa: float, n: int) -> np.ndarray:
        """Street x of the column centres of a strip starting at xa (riding order)."""
        return xa + self.sgn * (np.arange(n) + 0.5) / self.S

    def col(self, xa: float, x) -> np.ndarray:
        return self.sgn * (np.asarray(x, float) - xa) * self.S

    def strip(self, plane: str, xs: np.ndarray) -> np.ndarray | None:
        from ..register import resample_strip
        m = self.mosaic(plane)
        if m is None:
            return None
        ppm = m.px_per_m / self.h
        Y = self.yhi - (np.arange(self.n_rows) + 0.5) / self.S
        rows_src = (m.y_top * self.h - Y) * ppm - 0.5
        valid = (xs >= self.pos.xs[0]) & (xs <= self.pos.xs[-1])
        return resample_strip(m.image, lambda x: (self.pos.clip_x(x) / self.h - m.x0)
                              * m.px_per_m - 0.5, xs, rows_src, valid)

    def box_rows(self, plane: str, y0, y1) -> tuple:
        """Mosaic pixel rows (box edges) -> strip rows."""
        m = self.mosaic(plane)
        ppm = m.px_per_m / self.h
        f = lambda y: (self.yhi - (m.y_top * self.h - np.asarray(y, float) / ppm)) * self.S  # noqa: E731
        return f(y0), f(y1)


def render_pass(cfg: dict, lay, root: Path, p: dict, rc: dict, where: list | None = None):
    """RGB picture of one pass (see module doc). `where` (a list) receives one dict per
    strip: block, row, image top / height (rows) and left / width (columns) on the canvas."""
    from PIL import Image, ImageDraw
    sup = root / "supply"
    sj = pd.read_json(sup / "supply.json", typ="series")
    g = sj["guideline"]
    blocks = pd.read_csv(sup / "blocks.csv").rename(columns={"from": "frm"})
    occ = pd.read_csv(sup / "occupancy_by_pass.csv")
    occ = occ[occ["clip"] == p["id"]].set_index(["block_id", "row"])
    cs = pd.read_csv(sup / "curb_status.csv")
    cars = pd.read_csv(root / "multiday" / "cars_ref.csv")
    cars = cars[cars["clip"] == p["id"]]
    from ..regs.supply import in_roadway
    cars = cars[~in_roadway(cars, lay.intersections, lay.rows)]   # not parked: as supply
    ints = lay.intersections
    pv = _PassView(cfg, lay, root, p["id"], float(rc["px_per_m"]), rc["y_m"])
    S, pad = pv.S, float(rc["pad_m"])
    rev = pv.sgn < 0
    planes = cfg["mosaic"]["planes"]
    pr = {lay.row_of(p["id"], pl): pl for pl in planes if lay.row_of(p["id"], pl)}
    depth = {pl: (lay.plane_depth_h.get(p["id"]) or {}).get(pl) for pl in planes}
    near_first = sorted(pr, key=lambda r: list(planes).index(pr[r]))
    F = _fonts()

    strips = []
    for b in (blocks.iloc[::-1] if rev else blocks).itertuples():
        x0, x1 = b.x0_m - pad, b.x1_m + pad
        xa = x1 if rev else x0
        n = int(round((x1 - x0) * S))
        xs = pv.xs(xa, n)
        for r in near_first:
            pres = lay.present(r, xs) & (xs >= pv.pos.xs[0]) & (xs <= pv.pos.xs[-1])
            inside = (xs >= b.x0_m) & (xs <= b.x1_m)
            if (pres & inside).sum() / S < rc["min_present_m"]:
                continue
            img = pv.strip(pr[r], xs)
            if img is None:
                continue
            img = img.astype(np.float32)
            img[:, ~pres] *= 0.3
            # intersections: washed out
            for q in ints.itertuples():
                c = np.sort(pv.col(xa, [q.x0_m, q.x1_m]))
                a, e = int(max(0, c[0])), int(min(n, c[1]))
                if e > a:
                    img[:, a:e] = 0.45 * img[:, a:e] + 0.55 * 235
            # curb bar
            # 1 m status bins centred at x_ref_m; key k = the bin [k, k+1)
            gr = cs[cs["row"] == r].set_index(np.round(cs.loc[cs["row"] == r, "x_ref_m"] - 0.5)
                                               .astype(int))
            gr = gr[~gr.index.duplicated()]
            col = f"status_{p['id']}"
            st = gr[col].reindex(np.floor(xs).astype(int)).values if col in gr else \
                np.full(n, np.nan, object)
            cls = curb_class(np.asarray(st, dtype=object), g)
            bar = np.full((BAR_H, n, 3), BG, np.float32)
            for k, rgb in CURB.items():
                bar[:, cls == k] = rgb
            bar[:, ~pres] = BG
            strips.append({"b": b, "row": r, "xa": xa, "n": n, "img": img.astype(np.uint8),
                           "bar": bar.astype(np.uint8), "plane": pr[r],
                           "far": depth[pr[r]] is not None and depth[pr[r]] * pv.h > 3.0})
    if not strips:
        return None

    W = max(s["n"] for s in strips) + 2 * 24
    H_strip = SIDE_H + pv.n_rows + BAR_H + GAP
    TITLE_H = 112
    canvas = Image.new("RGB", (W, TITLE_H + H_strip * len(strips)), BG)
    D = ImageDraw.Draw(canvas)
    first, last = strips[0]["b"], strips[-1]["b"]
    street = cfg["street"].get("name") or cfg["name"]
    start = _travel_name(first.to if rev else first.frm)
    end = _travel_name(last.frm if rev else last.to)
    D.text((24, 14), f"{street} — {p['label']}", font=F["title"], fill=INK)
    route = f"riding from {start or 'the start'} to {end or 'the end'}; " \
        "each strip = one block, one side of the street, in riding order"
    D.text((24, 50), route, font=F["r"], fill=MUTED)
    lx = 24
    yl = 80
    D.rectangle((lx, yl + 2, lx + 18, yl + 16), outline=CAR, width=3)
    D.text((lx + 24, yl), "parked vehicle (counted)", font=F["s"], fill=INK)
    lx += 24 + D.textlength("parked vehicle (counted)", font=F["s"]) + 28
    for k, rgb in CURB.items():
        D.rectangle((lx, yl + 4, lx + 28, yl + 2 + BAR_H), fill=rgb)
        D.text((lx + 34, yl), CURB_TEXT[k], font=F["s"], fill=INK)
        lx += 34 + D.textlength(CURB_TEXT[k], font=F["s"]) + 28
    D.text((lx, yl), "shaded = cross street, dark = parking lane on the other side",
           font=F["s"], fill=MUTED)

    x_off = 24
    for i, s in enumerate(strips):
        b, r, xa, n = s["b"], s["row"], s["xa"], s["n"]
        y = TITLE_H + i * H_strip
        frm, to = _travel_name(b.frm), _travel_name(b.to)
        left, right = (to, frm) if rev else (frm, to)
        side = (cfg["street"]["rows"].get(r) or {}).get("label") or \
            (compass_side(lay.axis, b.x0_m, b.x1_m, lay.rows[r]["curb"])
             if lay.axis is not None else f"row {r}")
        o = occ.loc[(b.block_id, r)] if (b.block_id, r) in occ.index else None
        if o is not None and o["covered"] and o["capacity"] == o["capacity"]:
            k, cap = int(o["n_parked"] or 0), int(o["capacity"])
            summ = f"{k} parked · {cap} space{'s' if cap != 1 else ''}"
            if cap:
                summ += f" ({100 * k / cap:.0f} %)"
            if (o["n_not_legal_spot"] or 0) > 0:
                summ += f", {int(o['n_not_legal_spot'])} outside a legal space"
        else:
            summ = "block not fully covered"
        mid = f"{side}{', seen across the street' if s['far'] else ''}  ·  {summ}"
        # header: cross streets at the strip ends, block summary in the middle
        D.text((x_off, y + 8), f"◀ {left}" if left else "", font=F["b"], fill=INK)
        if right:
            t = f"{right} ▶"
            D.text((x_off + n - D.textlength(t, font=F["b"]), y + 8), t, font=F["b"], fill=INK)
        tw = D.textlength(mid, font=F["r"])
        D.text((x_off + (n - tw) / 2, y + 9), mid, font=F["r"], fill=INK)
        yi = y + SIDE_H
        if where is not None:
            where.append({"block_id": b.block_id, "row": r, "y": yi, "h": pv.n_rows + BAR_H,
                          "x": x_off, "w": n, "img_h": pv.n_rows})
        canvas.paste(Image.fromarray(s["img"]), (x_off, yi))
        canvas.paste(Image.fromarray(s["bar"]), (x_off, yi + pv.n_rows))
        # cars of this block numbered in riding order; neighbours' cars outlined grey
        k = cars[(cars["row"] == r)]
        c0, c1 = pv.col(xa, k.xr_start_m.values), pv.col(xa, k.xr_end_m.values)
        cl, cr = np.minimum(c0, c1), np.maximum(c0, c1)
        keep = (cr > 0) & (cl < n)
        k, cl, cr = k[keep], cl[keep], cr[keep]
        order = np.argsort(cl)
        num = 0
        for j in order:
            car = k.iloc[j]
            ra, rb = pv.box_rows(car["plane"], car["y0"], car["y1"])
            ra, rb = max(0.0, float(ra)), min(float(pv.n_rows - 1), float(rb))
            box = (x_off + max(0, cl[j]), yi + ra, x_off + min(n - 1, cr[j]), yi + rb)
            mine = b.x0_m <= car["xr_mid_m"] < b.x1_m
            D.rectangle(box, outline=CAR if mine else CAR_OTHER, width=3 if mine else 1)
            if mine:
                num += 1
                t = str(num)
                tw = D.textlength(t, font=F["s"])
                D.rectangle((box[0], box[1], box[0] + tw + 8, box[1] + 18), fill=CAR)
                D.text((box[0] + 4, box[1] + 1), t, font=F["s"], fill=INK)
        # cross-street names inside the shaded areas
        for q in ints.itertuples():
            c = np.sort(pv.col(xa, [q.x0_m, q.x1_m]))
            a, e = max(0.0, c[0]), min(float(n), c[1])
            if e - a < 20:
                continue
            t = _travel_name(q.name)
            tw = D.textlength(t, font=F["s"])
            if tw + 8 > e - a:
                t = t.split(" ", 1)[-1]
                tw = D.textlength(t, font=F["s"])
            tx = x_off + (a + e - tw) / 2
            D.text((tx, yi + 6), t, font=F["s"], fill=INK)
    return np.asarray(canvas)


def build(cfg: dict, lay, passes: list[dict], out: Path) -> dict:
    """Writes out/ride_<clip>.jpg per pass."""
    from ..config import _merge
    from .webmap import _save_jpg
    rc = _merge(DEFAULTS, (cfg.get("map") or {}).get("ride") or {})
    root = Path(cfg["out_dir"])
    res = {}
    for p in passes:
        img = render_pass(cfg, lay, root, p, rc)
        if img is None:
            continue
        f = out / f"ride_{p['id']}.jpg"
        _save_jpg(img, f, int(rc["jpeg_q"]))
        res[p["id"]] = f.name
    return res
