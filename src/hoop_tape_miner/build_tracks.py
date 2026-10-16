"""CLI: detections.pkl + camera.npz -> tracks.pkl (team + court-space track ids)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .game_config import load_game
from .tracks import on_court, smooth_tracks, track_team

SPECIAL = ("ref", "unk")


def team_map(labels, us: str | None) -> dict[str, str]:
    """cluster label -> "us" / "them" / "ref".

    cluster_names.json may use team labels directly ("us", "them", "ref", "unk"), or jersey colour words
    (e.g. "navy", "teal"): then `us` names our colour and the single other non-ref label is them.
    """
    teams = sorted({label for label in labels if label not in SPECIAL})
    if set(teams) <= {"us", "them"}:
        return {"us": "us", "them": "them", "ref": "ref"}
    if us is None:
        raise SystemExit(f"cluster labels {teams} are colours: pass --us <colour> or --game with us.colour, "
                         'or label the clusters "us" / "them" in cluster_names.json')
    others = [label for label in teams if label != us]
    if us not in teams or len(others) != 1:
        raise SystemExit(f"--us {us!r} must be one of exactly two team labels in cluster_names.json, got {teams}")
    return {us: "us", others[0]: "them", "ref": "ref"}


def write_cluster_montage(p: pd.DataFrame, path: Path, video: str, n=16):
    import cv2
    cap = cv2.VideoCapture(video)
    rows = []
    for c in sorted(p.clu.unique()):
        s = p[(p.clu == c) & ((p.y2 - p.y1) > 60)]
        s = s.sample(min(n, len(s)), random_state=1).sort_values("frame")
        cells = []
        for _, r in s.iterrows():
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(r.frame))
            ok, f = cap.read()
            cells.append(cv2.resize(f[int(r.y1):int(r.y2), int(r.x1):int(r.x2)], (60, 120)))
        cells += [np.zeros((120, 60, 3), np.uint8)] * (n - len(cells))
        row = np.hstack(cells)
        cv2.putText(row, str(c), (2, 16), 0, 0.6, (0, 0, 255), 2)
        rows.append(row)
    cv2.imwrite(str(path), np.vstack(rows))


def classify_colours(p: pd.DataFrame, run: Path, video: str, seed=0) -> pd.DataFrame:
    """SigLIP -> UMAP -> KMeans(6); a human names each cluster in cluster_names.json (us/them/ref/unk,
    or jersey colour words such as navy/teal), then a logistic regression on the confident samples
    labels every detection (low-confidence -> 'unk')."""
    import umap
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LogisticRegression

    e = np.load(run / "embeddings.npz")
    p = p.loc[p.index.intersection(e["idx"])].copy()
    pos = pd.Series(np.arange(len(e["idx"])), index=e["idx"])
    X = e["emb"][pos.loc[p.index].to_numpy()].astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(X), min(20000, len(X)), replace=False)
    red = umap.UMAP(n_components=3, random_state=seed).fit(X[sub])
    Z = red.transform(X)
    k = 6
    km = KMeans(k, n_init=10, random_state=seed).fit(Z)
    lab_file = run / "cluster_names.json"
    if not lab_file.exists():
        write_cluster_montage(p.assign(clu=km.labels_), run / "cluster_montage.jpg", video)
        raise SystemExit(f"Inspect {run / 'cluster_montage.jpg'} and write {lab_file} as "
                         '{"0": "us", "1": "them", "2": "ref", "3": "unk", ...} (jersey colour words such as '
                         '"navy" / "teal" work too, with --us <colour>), then re-run.')
    names = {int(c): n for c, n in json.loads(lab_file.read_text()).items()}
    y = np.array([names[label] for label in km.labels_])
    conf = y != "unk"
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(X[conf], y[conf])
    prob = clf.predict_proba(X)
    p["colour"] = np.where(prob.max(1) > 0.7, clf.classes_[prob.argmax(1)], "unk")
    return p


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--us", default=None, help="our cluster label when cluster_names.json uses colour words "
                    "(default: us.colour from --game; not needed for us/them labels)")
    ap.add_argument("--game", default=None, help="game.json (us.colour)")
    ap.add_argument("--fps", type=float, default=10.0)
    args = ap.parse_args(argv)
    run = Path(args.run)

    det = pd.read_pickle(run / "detections.pkl")
    cam = np.load(run / "camera.npz")
    good_frames = set(cam["frame"][cam["quality"] < 60])
    p = det[(det.kind == "person") & det.frame.isin(good_frames)].copy()
    p = p[on_court(p) & ((p.y2 - p.y1) > 35)]
    us = args.us or (load_game(args.game).get("us") or {}).get("colour")
    p = classify_colours(p, run, args.video)
    print(p.colour.value_counts())
    teams = team_map(set(p.colour), us)

    p["track"] = -1
    order = sorted(label for label in teams if label != "ref") + ["ref"]  # navy < teal: track ids as before
    for i, col in enumerate(order):
        off = i * 100000
        m = p.colour == col
        ids = track_team(p[m], args.fps)
        p.loc[m, "track"] = np.where(ids >= 0, ids + off, -1)
    p = p[p.track >= 0]
    # a tracklet's colour = majority vote of its detections
    maj = p.groupby("track").colour.agg(lambda s: s.value_counts().index[0])
    p["colour"] = p.track.map(maj)
    p["team"] = p.colour.map(teams)
    p = smooth_tracks(p)
    p.drop(columns=["feat"]).to_pickle(run / "tracks.pkl")
    tl = p.groupby("track").size()
    print(f"{p.track.nunique()} tracklets, median len {tl.median():.0f} frames, >=50 frames: {(tl >= 50).sum()}")


if __name__ == "__main__":
    main()
