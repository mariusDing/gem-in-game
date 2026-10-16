"""SigLIP appearance embeddings for on-court person detections (team classification).

Same recipe as roboflow/sports TeamClassifier: SigLIP vision embeddings -> UMAP -> KMeans.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from .device import device, half_dtype
from .tracks import on_court

SIGLIP_ID = "google/siglip-base-patch16-224"


class Embedder:
    def __init__(self):
        from transformers import AutoImageProcessor, SiglipVisionModel
        self.proc = AutoImageProcessor.from_pretrained(SIGLIP_ID)
        self.model = SiglipVisionModel.from_pretrained(SIGLIP_ID, torch_dtype=half_dtype()).to(device()).eval()

    @torch.no_grad()
    def __call__(self, crops_rgb: list[np.ndarray]) -> np.ndarray:
        inp = self.proc(images=crops_rgb, return_tensors="pt")
        out = self.model(pixel_values=inp["pixel_values"].to(device(), half_dtype()))
        return out.pooler_output.float().cpu().numpy()


def central_crop(bgr, r):
    """Torso-focused crop: middle 70% width, 10-65% height (jersey + shorts top)."""
    w, h = r.x2 - r.x1, r.y2 - r.y1
    x1, x2 = int(r.x1 + 0.15 * w), int(r.x2 - 0.15 * w)
    y1, y2 = int(r.y1 + 0.1 * h), int(r.y1 + 0.65 * h)
    c = bgr[max(y1, 0):max(y2, 1), max(x1, 0):max(x2, 1)]
    return np.ascontiguousarray(c[:, :, ::-1]) if c.size else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args(argv)
    run = Path(args.run)
    det = pd.read_pickle(run / "detections.pkl")
    cam = np.load(run / "camera.npz")
    good = set(cam["frame"][cam["quality"] < 60])
    p = det[(det.kind == "person") & det.frame.isin(good)]
    p = p[on_court(p) & ((p.y2 - p.y1) > 35)]
    prev_idx, prev_emb = np.array([], int), None
    if (run / "embeddings.npz").exists():  # incremental: only embed new detections
        e = np.load(run / "embeddings.npz")
        prev_idx, prev_emb = e["idx"], e["emb"]
        p = p[~p.index.isin(prev_idx)]
    print(f"{len(p)} detections to embed", flush=True)
    if p.empty:
        return
    by_frame = {f: g for f, g in p.groupby("frame")}
    emb = Embedder()
    cap = cv2.VideoCapture(args.video)
    frames = sorted(by_frame)
    out_idx, out_emb, buf_idx, buf = [], [], [], []
    fi, k = 0, 0
    last = frames[-1]
    while fi <= last:
        if fi not in by_frame:
            cap.grab()
            fi += 1
            continue
        ok, bgr = cap.read()
        if not ok:
            break
        for idx, r in by_frame[fi].iterrows():
            c = central_crop(bgr, r)
            if c is not None:
                buf.append(c)
                buf_idx.append(idx)
        if len(buf) >= args.batch:
            out_emb.append(emb(buf))
            out_idx += buf_idx
            buf, buf_idx = [], []
        fi += 1
        k += 1
        if k % 2000 == 0:
            print(f"{fi}/{last}", flush=True)
    if buf:
        out_emb.append(emb(buf))
        out_idx += buf_idx
    idx = np.concatenate([prev_idx, np.array(out_idx)])
    emb = np.concatenate(([prev_emb] if prev_emb is not None else []) + out_emb).astype(np.float16)
    np.savez(run / "embeddings.npz", idx=idx, emb=emb)
    print("embedded", len(out_idx))


if __name__ == "__main__":
    main()
