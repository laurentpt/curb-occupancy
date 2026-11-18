"""Static figure of the stretch-evening class distribution (the map page's stacked bars).

    python tools/distribution_figure.py <area run or study data dir>/summary.json --out FIG.svg

Reads `distribution` and `guideline.classes` from an area run's summary.json and draws the
same bars as the map page (same class ramp, same labels), as a standalone SVG for a README
(the page's note goes in the caption; it is the SVG's alt text).
Text follows the viewer's light / dark scheme; the bars keep the map's colours.
"""
from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path

RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]   # webmap.html RAMP (ordinal blue ramp)
TEXT = {"en": {"units": "{n} stretch-evenings · {s} stretches · {d} evenings",
               "note": "Share of stretch-evenings in each class: one unit is one stretch (about {ss} "
                       "spaces) on one evening, passes of the same evening averaged."},
        "fr": {"units": "{n} sections-soirs · {s} sections · {d} soirs",
               "note": "Part des sections-soirs dans chaque classe : une unité est une section "
                       "(environ {ss} places) un soir donné, passages d’un même soir moyennés."}}
CLS_FR = {"very_low": "Très faible", "low": "Faible", "optimal": "Optimal", "full": "Complet"}


def pct(v: float, unit: bool = True) -> str:
    return f"{round(v * 100)}{' %' if unit else ''}"


def classes(g: dict, lang: str) -> list[dict]:
    out, lo = [], 0.0
    for i, (hi, key, label) in enumerate(g["classes"]):
        name = CLS_FR.get(key, label) if lang == "fr" else label
        rng = (f"≥ {pct(lo)}" if hi is None else f"{pct(lo, False)}–{pct(hi)}" if lo
               else f"< {pct(hi)}")
        out.append({"key": key, "label": f"{name} ({rng})", "color": RAMP[min(i, len(RAMP) - 1)]})
        lo = hi
    return out


def svg(summary: dict, lang: str = "en", width: int = 640) -> str:
    g, bars = summary["guideline"], summary["distribution"]
    cls, T = classes(g, lang), TEXT[lang]
    pad, bar_h, gap, row_h = 8, 20, 2, 50
    lab = lambda b: b["label"][lang] if isinstance(b["label"], dict) else b["label"]
    el, rects, y = [], [], pad      # rects: bar segments, clipped to rounded bar outlines
    for b in bars:
        el.append(f'<text x="{pad}" y="{y + 12}" class="ink">{escape(lab(b))}</text>')
        el.append(f'<text x="{width - pad}" y="{y + 12}" class="muted" text-anchor="end">'
                  f'{escape(T["units"].format(n=b["n_units"], s=b["n_stretches"], d=b["n_dates"]))}</text>')
        segs = [c for c in cls if b["shares"].get(c["key"], 0) > 0]
        span = width - 2 * pad - gap * (len(segs) - 1)
        x, by = float(pad), y + 18
        for j, c in enumerate(segs):
            v = b["shares"][c["key"]]
            w = span * v / sum(b["shares"][s["key"]] for s in segs)
            rects.append(f'<rect x="{x:.1f}" y="{by}" width="{w:.1f}" height="{bar_h}" '
                         f'fill="{c["color"]}"><title>{escape(c["label"])}: {pct(v)} '
                         f'({b["units"].get(c["key"], 0)} / {b["n_units"]})</title></rect>')
            if v >= 0.08:
                ink = "#0b0b0b" if c["color"] == RAMP[0] else "#fff"
                el.append(f'<text x="{x + w / 2:.1f}" y="{by + 14}" text-anchor="middle" '
                          f'font-size="11" fill="{ink}">{pct(v)}</text>')
            x += w + gap
        y += row_h
    kx = pad
    for c in cls:
        el.append(f'<rect x="{kx}" y="{y + 1}" width="10" height="10" rx="2" fill="{c["color"]}"/>')
        el.append(f'<text x="{kx + 14}" y="{y + 10}" class="ink2" font-size="11">{escape(c["label"])}</text>')
        kx += 14 + 6.2 * len(c["label"]) + 14
    h = y + 12 + pad
    note = T["note"].format(ss=g.get("stretch_spaces", "?"))   # alt text; the caption explains it
    clip = "".join(f'<rect x="{pad}" y="{pad + 18 + i * row_h}" width="{width - 2 * pad}" '
                   f'height="{bar_h}" rx="4"/>' for i in range(len(bars)))
    style = ("text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;"
             "font-size:12px;font-variant-numeric:tabular-nums}"
             ".ink{fill:#0b0b0b}.ink2{fill:#52514e}.muted{fill:#6f6d68}"
             "@media (prefers-color-scheme:dark){.ink{fill:#f0efec}.ink2{fill:#c3c2b7}.muted{fill:#a3a29b}}")
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{h}" '
            f'viewBox="0 0 {width} {h}" role="img" aria-label="{escape(note)}">\n'
            f'<style>{style}</style>\n<defs><clipPath id="bars">{clip}</clipPath></defs>\n'
            f'<g clip-path="url(#bars)">\n' + "\n".join(rects) + '\n</g>\n' + "\n".join(el) + '\n</svg>\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary", help="summary.json of an area run (or the study's data/summary.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lang", choices=sorted(TEXT), default="en")
    ap.add_argument("--width", type=int, default=640)
    a = ap.parse_args()
    s = json.loads(Path(a.summary).read_text(encoding="utf-8"))
    if not s.get("distribution"):
        raise SystemExit(f"{a.summary}: no distribution (area config needs a `distribution:` section)")
    Path(a.out).write_text(svg(s, a.lang, a.width), encoding="utf-8")
    print(a.out)


if __name__ == "__main__":
    main()
