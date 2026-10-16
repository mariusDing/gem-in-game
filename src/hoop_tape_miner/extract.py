"""Pass 1: per-frame detection + court homography + jersey colour features.

Writes one parquet of detections and one npz of homographies. Resumable in chunks.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from .camera import CourtCameraTracker
from .court import project
from .device import device, half_dtype

PERSON_COCO_ID = 1
BALL_HF_LABEL = 0
HF_BALL_MODEL = "koppolusameer/rfdetr-medium-basketball-playerexpanded-referee-detection"


def torso_feature(bgr, box):
    """Hue/sat/value histogram of the upper body (jersey) region -> 24-d vector."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx1, cx2 = int(x1 + 0.25 * w), int(x2 - 0.25 * w)
    cy1, cy2 = int(y1 + 0.18 * h), int(y1 + 0.5 * h)
    crop = bgr[max(cy1, 0):max(cy2, 1), max(cx1, 0):max(cx2, 1)]
    if crop.size == 0:
        return np.zeros(24, np.float32)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    hh = np.histogram(hsv[:, 0], bins=12, range=(0, 180))[0]
    sh = np.histogram(hsv[:, 1], bins=6, range=(0, 256))[0]
    vh = np.histogram(hsv[:, 2], bins=6, range=(0, 256))[0]
    f = np.concatenate([hh, sh, vh]).astype(np.float32)
    return f / max(len(hsv), 1)


class BallDetector:
    def __init__(self):
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
        self.proc = AutoImageProcessor.from_pretrained(HF_BALL_MODEL)
        self.model = AutoModelForObjectDetection.from_pretrained(HF_BALL_MODEL).to(device(), half_dtype()).eval()

    @torch.no_grad()
    def __call__(self, rgb, thr=0.2):
        inp = self.proc(images=rgb, return_tensors="pt").to(device())
        inp["pixel_values"] = inp["pixel_values"].to(half_dtype())
        out = self.model(**inp)
        r = self.proc.post_process_object_detection(out, threshold=thr, target_sizes=[rgb.shape[:2]])[0]
        res = []
        for score, label, box in zip(r["scores"], r["labels"], r["boxes"]):
            box = box.float().tolist()
            if label.item() == BALL_HF_LABEL and (box[2] - box[0]) < 40 and (box[3] - box[1]) < 40:
                res.append((*box, score.item()))
        return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib", required=True, help="json: [{image, H}] calibrated keyframes")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=1e9)
    args = ap.parse_args(argv)

    from rfdetr import RFDETRLarge
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from .calibrate import load_bank
    bank = load_bank(args.calib)  # image paths relative to keyframes.json, or absolute
    tracker = CourtCameraTracker(bank)
    det = RFDETRLarge()
    ball = BallDetector()

    cap = cv2.VideoCapture(args.video)
    vfps = cap.get(cv2.CAP_PROP_FPS)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(vfps / args.fps))
    f0 = int(args.start * vfps)
    f_end = min(n_total, int(args.end * vfps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)

    rows, Hs, frames_meta = [], [], []
    t0 = time.time()
    fi = f0
    while fi < f_end:
        if (fi - f0) % step:
            if not cap.grab():
                break
            fi += 1
            continue
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = bgr[:, :, ::-1].copy()
        d = det.predict(rgb, threshold=0.3)
        pm = d.class_id == PERSON_COCO_ID
        pboxes, pconf = d.xyxy[pm], d.confidence[pm]
        H, q, how = tracker.update(bgr, pboxes[pconf > 0.45])
        t = fi / vfps
        Hs.append(H if H is not None else np.full((3, 3), np.nan))
        frames_meta.append((fi, t, q, how))
        if H is not None:
            Hinv = np.linalg.inv(H)
        for b, c in zip(pboxes, pconf):
            foot = np.array([[(b[0] + b[2]) / 2, b[3]]])
            cx, cy = project(Hinv, foot)[0] if H is not None else (np.nan, np.nan)
            rows.append(dict(frame=fi, t=t, kind="person", x1=b[0], y1=b[1], x2=b[2], y2=b[3], conf=c,
                             cx=cx, cy=cy, feat=torso_feature(bgr, b).tolist()))
        for x1, y1, x2, y2, c in ball(rgb):
            rows.append(dict(frame=fi, t=t, kind="ball", x1=x1, y1=y1, x2=x2, y2=y2, conf=c,
                             cx=np.nan, cy=np.nan, feat=None))
        n = len(frames_meta)
        if n % 500 == 0:
            el = time.time() - t0
            done = (fi - f0) / max(f_end - f0, 1)
            print(f"[{t/60:5.1f} min] {n} frames, {n/el:.1f} fps, eta {el/done*(1-done)/60 if done else 0:.0f} min, q={q:.1f} {how}", flush=True)
        if n % 1000 == 0:
            _save(out, rows, Hs, frames_meta)
        fi += 1
    _save(out, rows, Hs, frames_meta)
    print("done", len(frames_meta), "frames in", time.time() - t0, "s")


def _save(out, rows, Hs, meta):
    pd.DataFrame(rows).to_pickle(out / "detections.pkl")
    m = np.array(meta, dtype=object)
    np.savez(out / "camera.npz", H=np.array(Hs), frame=m[:, 0].astype(int), t=m[:, 1].astype(float),
             quality=m[:, 2].astype(float), how=m[:, 3].astype(str))


if __name__ == "__main__":
    main()
