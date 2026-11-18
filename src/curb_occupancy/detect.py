"""Vehicle detection on pushbroom mosaics + geometric acceptance tests.

A mosaic built at plane depth D is an orthographic elevation of that plane: column = along-
street position, row = height above the road, both at `px_per_m` per camera-height unit.
A static object at depth D' appears with its true length, but its heights are scaled by D/D'
and its ground contact sits at height h*(1 - D/D') in the mosaic. Hence:

* parked car at the mosaic's plane: bottom ~ road row, height 1.4-2.0 m, length 3-6 m;
* nearer objects (bollards, overtaking cyclists): too tall, bottom below the road row;
* farther objects (cross-street cars, the other side of the street): squat, floating;
* moving vehicles: length scaled by v_bike/|v_rel| (mangled), usually fail length/aspect.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


@dataclass
class Acceptance:
    """Thresholds in metres (converted with the camera height)."""
    min_height_m: float = 1.15     # true height after depth correction
    max_height_m: float = 3.2
    min_length_m: float = 2.4
    max_length_m: float = 9.0
    lateral_m: tuple = (0.8, 3.0)  # accepted lateral distance band of the parked row
    min_conf: float = 0.25
    # flow depth / geometric depth: ~0.9-1.5 for parked cars (2026-10-22 probe). Outside
    # `static_ratio` -> flagged 'moving', outside `flag_ratio` -> 'flow_depth_mismatch'.
    # Flags, not rejects: at <1.2 m (bollards in front, ~20 px/frame flow) a parked car read
    # 0.66, and the street layout already removes travel-lane traffic.
    static_ratio: tuple = (0.75, 2.0)
    flag_ratio: tuple = (0.85, 1.6)
    clip_px: int = 12              # box bottom this close to the mosaic bottom -> contact unknown


class YoloDetector:
    """Ultralytics YOLO (AGPL-3.0; optional, `pip install ultralytics`), as used up to the
    interim results of 2026-10-31. `model`: a .pt weights file."""

    def __init__(self, model: str, conf: float, device: str):
        from ultralytics import YOLO
        self.model, self.conf, self.device = YOLO(model), conf, device

    def __call__(self, tiles: list[np.ndarray]):
        H, W = tiles[0].shape[:2]
        res = self.model.predict([np.ascontiguousarray(t[:, :, ::-1]) for t in tiles],  # BGR
                                 imgsz=(int(np.ceil(H / 32) * 32), W), conf=self.conf,
                                 classes=list(VEHICLE_CLASSES), device=self.device,
                                 verbose=False, rect=True)
        return [(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                 r.boxes.cls.cpu().numpy().astype(int)) for r in res]


class HFDetector:
    """A Hugging Face `transformers` object detector with COCO labels (D-FINE, RT-DETR: Apache-
    2.0 code and weights). `model`: hub id or local directory.

    Each tile is letterboxed into the model's square training input (640 x 640: a 1280 x 330
    mosaic tile is scaled by 0.5 and padded with grey), so vehicles appear at the scale and
    aspect ratio the model was trained on. Feeding the strip at its native resolution (these
    models build anchors from the input size, so it runs) recognised fewer cars: on the 2026
    study mosaics D-FINE-X matched 1171 of YOLO11-L's 1258 accepted vehicles at native
    resolution and 1253 letterboxed (2026-11-01 detector A/B)."""

    def __init__(self, model: str, conf: float, device: str):
        import torch
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
        self.proc = AutoImageProcessor.from_pretrained(model)
        self.model = AutoModelForObjectDetection.from_pretrained(model).to(device).eval()
        size = self.proc.size
        self.side = int(size["height"] if "height" in size else size["shortest_edge"])
        self.conf, self.device, self.torch = conf, device, torch
        names = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        alias = {"motorcycle": ("motorcycle", "motorbike")}
        self.cls = {}   # model label id -> COCO-80 index of VEHICLE_CLASSES
        for coco, name in VEHICLE_CLASSES.items():
            for n in alias.get(name, (name,)):
                if n in names:
                    self.cls[names[n]] = coco

    def __call__(self, tiles: list[np.ndarray]):
        import cv2
        torch = self.torch
        H, W = tiles[0].shape[:2]
        s = self.side / max(H, W)
        h, w = int(round(H * s)), int(round(W * s))
        x = np.full((len(tiles), self.side, self.side, 3), 114, np.uint8)
        for i, t in enumerate(tiles):
            x[i, :h, :w] = cv2.resize(t, (w, h), interpolation=cv2.INTER_AREA)
        px = torch.from_numpy(x).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        with torch.no_grad():
            out = self.model(pixel_values=px)
        res = self.proc.post_process_object_detection(
            out, threshold=self.conf, target_sizes=[(self.side, self.side)] * len(tiles))
        dets = []
        for r in res:
            lab = r["labels"].cpu().numpy()
            keep = np.isin(lab, list(self.cls))
            bx = r["boxes"].cpu().numpy()[keep] / s
            bx[:, [0, 2]] = bx[:, [0, 2]].clip(0, W)
            bx[:, [1, 3]] = bx[:, [1, 3]].clip(0, H)
            dets.append((bx, r["scores"].cpu().numpy()[keep],
                         np.array([self.cls[k] for k in lab[keep]], int)))
        return dets


def load_detector(dc: dict, device: str = "cuda"):
    """`detect.backend`: 'hf' (transformers, default) or 'yolo' (ultralytics)."""
    kind = dc.get("backend", "hf")
    if kind == "hf":
        return HFDetector(dc["model"], dc["conf"], device)
    if kind == "yolo":
        return YoloDetector(dc["model"], dc["conf"], device)
    raise ValueError(f"detect.backend {kind!r}: 'hf' or 'yolo'")


def detector_id(dc: dict) -> dict:
    """What the cached raw boxes depend on (raw_<plane>.json next to raw_<plane>.csv)."""
    return {k: dc.get(k) for k in ("backend", "model", "tile_w", "stride", "conf")}


def tile_detect(image: np.ndarray, detector, tile_w: int = 1280, stride: int = 640,
                iou_merge: float = 0.5, batch: int = 16) -> pd.DataFrame:
    """Run a detector (`load_detector`) on horizontal tiles of a (short, very wide) RGB mosaic.

    Returns boxes in mosaic pixel coordinates (x0,y0,x1,y1,conf,cls). Boxes cut by an interior
    tile edge are discarded (the overlap guarantees the car is whole in a neighbouring tile
    as long as it is shorter than tile_w - stride).
    """
    import torch
    from torchvision.ops import nms

    H, W = image.shape[:2]
    xs = list(range(0, max(W - tile_w, 0) + 1, stride))
    if xs[-1] + tile_w < W:
        xs.append(W - tile_w)
    rows = []
    for b in range(0, len(xs), batch):
        chunk = xs[b:b + batch]
        res = detector([image[:, x:x + tile_w] for x in chunk])
        for x, (bx, cf, cl) in zip(chunk, res):
            for (x0, y0, x1, y1), c, k in zip(bx, cf, cl):
                left_cut = x0 < 4 and x > 0
                right_cut = x1 > tile_w - 4 and x + tile_w < W
                if left_cut or right_cut:
                    continue
                rows.append((x0 + x, y0, x1 + x, y1, float(c), int(k)))
    df = pd.DataFrame(rows, columns=["x0", "y0", "x1", "y1", "conf", "cls"])
    if len(df):
        keep = nms(torch.tensor(df[["x0", "y0", "x1", "y1"]].values, dtype=torch.float32),
                   torch.tensor(df["conf"].values, dtype=torch.float32), iou_merge).numpy()
        df = df.iloc[np.sort(keep)].reset_index(drop=True)
        df = _suppress_contained(df)
    df["cls_name"] = df["cls"].map(VEHICLE_CLASSES)
    return df


def _suppress_contained(df: pd.DataFrame, frac: float = 0.8) -> pd.DataFrame:
    """Drop boxes mostly contained in a higher-confidence box (part-of-car detections)."""
    b = df[["x0", "y0", "x1", "y1"]].values
    order = np.argsort(-df["conf"].values)
    keep = np.ones(len(df), bool)
    for ii, i in enumerate(order):
        if not keep[i]:
            continue
        for j in order[ii + 1:]:
            if not keep[j]:
                continue
            ix = max(0, min(b[i, 2], b[j, 2]) - max(b[i, 0], b[j, 0]))
            iy = max(0, min(b[i, 3], b[j, 3]) - max(b[i, 1], b[j, 1]))
            aj = (b[j, 2] - b[j, 0]) * (b[j, 3] - b[j, 1])
            if aj > 0 and ix * iy / aj > frac:
                keep[j] = False
    return df[keep].reset_index(drop=True)


def merge_splits(df: pd.DataFrame, px_per_m: float, min_part_m: float = 2.4,
                 max_gap_m: float = 0.7, max_len_m: float = 6.5, bottom_tol_m: float = 0.25
                 ) -> pd.DataFrame:
    """Re-join vehicles split in two by a thin occluder (bike-lane bollards in front of the
    far row cut a car into two boxes ~0.1 m apart; both halves can fall under the minimum
    length and the car is lost). Adjacent boxes (x order) merge, smallest gap first, when
    the gap is <= max_gap_m, their bottoms (ground contact) agree within bottom_tol_m, one
    part is shorter than min_part_m and the union is <= max_len_m. Column `merged` = number
    of boxes joined (1 = untouched). `px_per_m` at the mosaic plane."""
    if df.empty:
        return df.assign(merged=np.ones(0, int))
    b = df.sort_values("x0").reset_index(drop=True).copy()
    b["merged"] = 1
    P = px_per_m
    while True:
        L = (b.x1 - b.x0) / P
        gap = (b.x0.values[1:] - b.x1.values[:-1]) / P
        ok = ((gap >= -0.3) & (gap <= max_gap_m)
              & (np.abs(b.y1.values[1:] - b.y1.values[:-1]) <= bottom_tol_m * P)
              & ((L.values[1:] < min_part_m) | (L.values[:-1] < min_part_m))
              & ((np.maximum(b.x1.values[1:], b.x1.values[:-1]) - b.x0.values[:-1]) / P
                 <= max_len_m))
        if not ok.any():
            return b
        i = int(np.argmin(np.where(ok, gap, np.inf)))
        a, c = b.iloc[i], b.iloc[i + 1]
        big = a if (a.x1 - a.x0) >= (c.x1 - c.x0) else c
        row = big.copy()
        row["x0"], row["x1"] = min(a.x0, c.x0), max(a.x1, c.x1)
        row["y0"], row["y1"] = min(a.y0, c.y0), max(a.y1, c.y1)
        row["conf"] = max(a.conf, c.conf)
        row["merged"] = a.merged + c.merged
        b = pd.concat([b.iloc[:i], row.to_frame().T, b.iloc[i + 2:]], ignore_index=True)
        b = b.astype({k: df[k].dtype for k in df.columns if k in b}).astype({"merged": int})


def accept(df: pd.DataFrame, mosaic, height_m: float, acc: Acceptance = Acceptance(),
           profiles=None, slope: np.ndarray | None = None) -> pd.DataFrame:
    """Annotate detections with metric geometry and an accept/reject reason.

    Geometric depth: the box bottom (tyre contact) appears at height Y_b = h(1 - D/D') in a
    mosaic built at plane depth D, so D' = D / (1 - Y_b/h) and the true height is
    H' * D'/D (length is depth-independent for static objects). Flow depth: median of the
    flow at the rows that see the box's lower body panels, over the frames the box spans;
    for a static object it must agree with D' (moving vehicles break this).
    """
    h = height_m
    P = mosaic.px_per_m / h                 # px per metre at the plane
    out = df.copy()
    out["length_m"] = (out.x1 - out.x0) / P
    out["bottom_m"] = (mosaic.y_top * h) - out.y1 / P   # apparent height of bottom above road
    ratio = 1 - out["bottom_m"] / h                     # = D / D'
    out["depth_m"] = np.where(ratio > 0.05, mosaic.depth * h / ratio.clip(lower=0.05), np.inf)
    out["height_m"] = (out.y1 - out.y0) / P / ratio.clip(lower=0.05)
    # contact cut by the mosaic bottom or the physical FOV edge (black rows below the box)
    out["clipped"] = out.y1 >= _lowest_valid_row(mosaic.image, out) - acc.clip_px
    out["x_start_m"] = (mosaic.x0 + out.x0 / mosaic.px_per_m) * h
    out["x_end_m"] = (mosaic.x0 + out.x1 / mosaic.px_per_m) * h
    cf = mosaic.col_frame
    out["frame"] = [int(cf[min(int((a + b) / 2), len(cf) - 1)]) for a, b in zip(out.x0, out.x1)]
    if profiles is not None and slope is not None:
        out["flow_depth_m"] = _body_flow_depth(out, mosaic, profiles, slope) * h
    else:
        out["flow_depth_m"] = np.nan
    # a vehicle moving along the street at v_o shifts the flow by v_o but not the ground
    # contact: ratio = v_bike / (v_bike + v_o) (<1 oncoming, >1 same direction, slower)
    out["flow_ratio"] = out["flow_depth_m"] / out["depth_m"]
    reasons = []
    lo, hi = acc.lateral_m
    for r in out.itertuples():
        why = []
        if r.conf < acc.min_conf:
            why.append("low_conf")
        if r.clipped:
            # the contact is at or below the visible bottom, so depth_m is an upper bound:
            # below the band -> nearer than the parked row (near-row cars in the far mosaic);
            # else nothing but a parked car is that close (closer than ~1.1 m on 2026-10-23
            # GX014484): kept, flagged, true depth unknown
            if r.depth_m < lo:
                why.append("clipped_bottom")
            elif r.height_m > acc.max_height_m:
                why.append("too_tall")
        elif r.depth_m < lo:
            why.append("too_near")
        elif r.depth_m > hi:
            why.append("too_far")
        else:
            if r.height_m < acc.min_height_m:
                why.append("too_short")
            if r.height_m > acc.max_height_m:
                why.append("too_tall")
        if r.length_m < acc.min_length_m:
            why.append("too_narrow")
        if r.length_m > acc.max_length_m:
            why.append("too_long")
        reasons.append(",".join(why))
    out["reject"] = reasons
    fr = out["flow_ratio"]  # NaN (no flow measured) compares False -> no flag
    out["flag"] = np.select(
        [(fr < acc.static_ratio[0]) | (fr > acc.static_ratio[1]),
         (fr < acc.flag_ratio[0]) | (fr > acc.flag_ratio[1])],
        ["moving", "flow_depth_mismatch"], "")
    out.loc[out.clipped & (out.flag == ""), "flag"] = "clipped"
    if "merged" in out:
        out.loc[(out.merged > 1) & (out.flag == ""), "flag"] = "merged"
    out["kept"] = out["reject"] == ""
    return out


def _lowest_valid_row(img: np.ndarray, det: pd.DataFrame, half_w: int = 20) -> np.ndarray:
    """Per box: index + 1 of the lowest mosaic row with image content under the box centre
    (rows outside the physical field of view are black)."""
    rows = img.shape[0]
    out = np.full(len(det), rows, float)
    for i, (a, b) in enumerate(zip(det.x0.values, det.x1.values)):
        c = int((a + b) / 2)
        col = img[:, max(c - half_w, 0):c + half_w]
        valid = (col.max(axis=2) > 0).mean(1) > 0.5
        if valid.any():
            out[i] = np.where(valid)[0].max() + 1
    return out


def _body_flow_depth(det: pd.DataFrame, mosaic, fp, slope, band=(0.35, 0.75)) -> np.ndarray:
    """Median flow depth (h units) at the rows imaging the box's lower body (fraction `band`
    of the box height from the top), over the box's source frames.

    A mosaic row at plane height Y (h units) is the ray (., 1 - Y, D) of the virtual camera,
    i.e. small-view row cy + f (1 - Y)/D; depth there is f * s / (-u).
    """
    cf = mosaic.col_frame
    P = mosaic.px_per_m
    res = []
    for r in det.itertuples():
        fr = cf[max(int(r.x0), 0):min(int(r.x1), len(cf))]
        fr = np.unique(fr[fr >= 0])
        if not len(fr):
            res.append(np.nan)
            continue
        ys = r.y0 + np.array(band) * (r.y1 - r.y0)
        Y = mosaic.y_top - ys / P
        rows = (fp.cy + fp.f * (1 - Y) / mosaic.depth).astype(int).clip(0, fp.u.shape[1] - 1)
        u = -fp.u[np.ix_(fr, np.arange(rows[0], rows[1] + 1))]
        with np.errstate(all="ignore"):
            z = fp.f * slope[fr, None] / np.where(u > 0.05, u, np.nan)
            res.append(float(np.nanmedian(z)))
    return np.array(res)


def overlay(mosaic_img: np.ndarray, det: pd.DataFrame, blur_plates: bool = False,
            inactive_cols: np.ndarray | None = None) -> np.ndarray:
    """BGR overlay: kept boxes green, rejected red with reason, flagged orange; with a street
    layout, columns where this plane is not the parked row are dimmed and boxes rejected for
    being on the travel lane / other side are magenta."""
    import cv2

    img = cv2.cvtColor(mosaic_img, cv2.COLOR_RGB2BGR).copy()
    if blur_plates:
        img = privacy_blur(img)
    if inactive_cols is not None:
        img[:, inactive_cols] = (img[:, inactive_cols] * 0.45).astype(np.uint8)
    for r in det.itertuples():
        col = (0, 200, 0) if r.kept else (0, 0, 230)
        if r.kept and r.flag:
            col = (0, 140, 255)
        if r.reject in ("travel_lane", "other_side"):
            col = (230, 0, 230)
        p0, p1 = (int(r.x0), int(r.y0)), (int(r.x1), int(r.y1))
        cv2.rectangle(img, p0, p1, col, 2)
        label = f"{r.cls_name} {r.conf:.2f} L{r.length_m:.1f} H{r.height_m:.1f} D{r.depth_m:.1f}"
        if not r.kept:
            label += " " + r.reject
        cv2.putText(img, label, (p0[0] + 2, max(p0[1] - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    col, 1, cv2.LINE_AA)
    return img


def privacy_blur(img: np.ndarray, band=(0.45, 0.95)) -> np.ndarray:
    """Coarse anonymisation for sharing: pixelate the lower body band where plates sit.

    Placeholder until a plate/face detector is wired in; aggressive on purpose.
    """
    import cv2

    h = img.shape[0]
    y0, y1 = int(h * band[0]), int(h * band[1])
    roi = img[y0:y1]
    small = cv2.resize(roi, (max(1, roi.shape[1] // 6), max(1, roi.shape[0] // 6)),
                       interpolation=cv2.INTER_AREA)
    img[y0:y1] = cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
    return img


def color_features(image_rgb: np.ndarray, det: pd.DataFrame) -> pd.DataFrame:
    """Median CIELAB of the door panel (rows 32-52 %, cols 25-75 % of the box): below the
    glass, above the wheels/black cladding/shadow (rows 50-80 % read near-black for most
    cars, 2026-10-23), and the same on both sides of a car (matching feature)."""
    import cv2

    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    feats = []
    for r in det.itertuples():
        x0, x1 = int(r.x0 + 0.25 * (r.x1 - r.x0)), int(r.x0 + 0.75 * (r.x1 - r.x0))
        y0, y1 = int(r.y0 + 0.32 * (r.y1 - r.y0)), int(r.y0 + 0.52 * (r.y1 - r.y0))
        patch = lab[max(y0, 0):max(y1, y0 + 1), max(x0, 0):max(x1, x0 + 1)].reshape(-1, 3)
        feats.append(np.median(patch, 0) if len(patch) else [np.nan] * 3)
    f = np.array(feats, dtype=np.float32).reshape(-1, 3)
    out = det.copy()
    # OpenCV 8-bit Lab: L*255/100, a+128, b+128
    out["lab_L"], out["lab_a"], out["lab_b"] = f[:, 0] * 100 / 255, f[:, 1] - 128, f[:, 2] - 128
    return out
