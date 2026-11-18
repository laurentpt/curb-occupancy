"""Example figure: one block of one pass from the ride view, without street names or date.

    python tools/example_block.py <street config> <clip> <block_id> [<block_id> ...] --out DIR

Crops the curb image + curb-status bar of each block (numbered boxes = counted vehicles),
trimming the shaded cross-street ends (they carry the street names). Check every output for
readable plates and faces before publishing it.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("clip")
    ap.add_argument("blocks", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lines", type=int, default=1, help="wrap the strip onto this many lines")
    a = ap.parse_args()
    from PIL import Image

    from curb_occupancy import config as C
    from curb_occupancy.config import _merge
    from curb_occupancy.layout import Layout
    from curb_occupancy.publish import rideview as RV
    cfg = C.load(a.config)
    root = Path(cfg["out_dir"])
    lay = Layout.load(root / "layout")
    rc = _merge(RV.DEFAULTS, (cfg.get("map") or {}).get("ride") or {})
    where: list = []
    img = RV.render_pass(cfg, lay, root, {"id": a.clip, "label": ""}, rc, where)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for w in where:
        if w["block_id"] not in a.blocks:
            continue
        crop = img[w["y"]:w["y"] + w["h"], w["x"]:w["x"] + w["w"]]
        # shaded cross streets: washed-out columns (0.45 img + 0.55 * 235) at the ends
        body = crop[: w["img_h"]].astype(float)
        # (10th percentile: the street name printed there is dark)
        washed = np.percentile(body.min(axis=2), 10, axis=0) > 0.55 * 235 - 6
        cols = np.where(~washed)[0]
        crop = crop[:, cols[0]:cols[-1] + 1] if len(cols) else crop
        if a.lines > 1:   # cut in gaps between counted boxes, near equal widths
            box = np.all(np.abs(crop[: w["img_h"]].astype(int) - RV.CAR) < 40, axis=2).any(axis=0)
            gaps = np.where(~box)[0]
            cuts = [0] + [int(gaps[np.argmin(np.abs(gaps - crop.shape[1] * k / a.lines))])
                          for k in range(1, a.lines)] + [crop.shape[1]]
            parts = [crop[:, c0:c1] for c0, c1 in zip(cuts[:-1], cuts[1:])]
            wmax, gap = max(p.shape[1] for p in parts), 16
            canvas = np.full((len(parts) * (crop.shape[0] + gap) - gap, wmax, 3), 255, np.uint8)
            for i, p in enumerate(parts):
                canvas[i * (crop.shape[0] + gap): i * (crop.shape[0] + gap) + crop.shape[0],
                       : p.shape[1]] = p
            crop = canvas
        f = out / f"example_{w['block_id']}_{w['row']}.png"
        Image.fromarray(crop).save(f)
        print(f, crop.shape)


if __name__ == "__main__":
    main()
