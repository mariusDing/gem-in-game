"""Re-run only the court homography (reusing pass-1 detections) and recompute court coords."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .calibrate import load_bank
from .camera import CourtCameraTracker
from .court import project


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=1e9)
    ap.add_argument("--dry", action="store_true", help="do not overwrite outputs")
    args = ap.parse_args(argv)
    run = Path(args.run)
    det = pd.read_pickle(run / "detections.pkl")
    cam = dict(np.load(run / "camera.npz"))
    persons = det[det.kind == "person"]
    boxes = {f: g[["x1", "y1", "x2", "y2"]].to_numpy()[g.conf.to_numpy() > 0.45] for f, g in persons.groupby("frame")}

    trk = CourtCameraTracker(load_bank(args.calib))  # image paths relative to keyframes.json, or absolute
    cap = cv2.VideoCapture(args.video)
    sel = (cam["t"] >= args.start) & (cam["t"] <= args.end)
    want = {int(f): i for i, f in enumerate(cam["frame"]) if sel[i]}
    last = max(want)
    fi = min(want)
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    H_all, q_all, how_all = cam["H"].copy(), cam["quality"].copy(), cam["how"].astype(object).copy()
    t0 = time.time()
    while fi <= last:
        if fi not in want:
            cap.grab()
            fi += 1
            continue
        ok, bgr = cap.read()
        if not ok:
            break
        H, q, how = trk.update(bgr, boxes.get(fi, []))
        i = want[fi]
        H_all[i] = H if H is not None else np.nan
        q_all[i], how_all[i] = q, how
        if i % 1000 == 0:
            print(f"{i}/{len(want)} {(time.time() - t0) / 60:.1f} min, refined so far {(q_all[:i + 1] < 3).mean():.3f}, bank {len(trk.bank.frames)}", flush=True)
        fi += 1
    if args.dry:
        q = q_all[sel]
        print("dry run: refined", (q < 3).mean().round(3), "pred-only", (q == 50).mean().round(3))
        np.savez(run / "camera_dry.npz", H=H_all, frame=cam["frame"], t=cam["t"], quality=q_all, how=how_all.astype(str))
        return
    np.savez(run / "camera.npz", H=H_all, frame=cam["frame"], t=cam["t"], quality=q_all, how=how_all.astype(str))
    # recompute court coordinates of every person detection
    Hinv = {int(f): np.linalg.inv(H) for f, H in zip(cam["frame"], H_all) if np.isfinite(H).all()}
    cx, cy = np.full(len(det), np.nan), np.full(len(det), np.nan)
    for f, g in det[det.kind == "person"].groupby("frame"):
        if f in Hinv:
            foot = np.c_[(g.x1 + g.x2) / 2, g.y2]
            c = project(Hinv[f], foot)
            pos = det.index.get_indexer(g.index)
            cx[pos], cy[pos] = c[:, 0], c[:, 1]
    det["cx"], det["cy"] = cx, cy
    det.to_pickle(run / "detections.pkl")
    print("done; refined", (q_all < 3).mean().round(3), "pred-only", (q_all == 50).mean().round(3))


if __name__ == "__main__":
    main()
