"""CLI: tracks + jersey reads + game config -> player identities, possessions, metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .metrics import possession_metrics
from .possessions import frame_table, offence_by_distance, segment_possessions


def identities(tr: pd.DataFrame, jersey: dict, rosters: dict) -> dict[int, str]:
    """track -> jersey number, keeping only numbers on the team's roster (if given).

    A tracklet whose vote is "none" is still accepted when it has exactly one distinct non-none read
    that is on the team's roster (the vote needs 2 agreeing reads, which drops many one-read tracklets).
    """
    team_of = tr.groupby("track").team.first()
    out = {}
    for tid, j in jersey.items():
        n = j["number"]
        tid = int(tid)
        if tid not in team_of.index:
            continue
        roster = rosters.get(team_of[tid])
        if n == "none":
            reads = {r for r in j.get("reads", []) if r != "none"}
            if len(reads) != 1 or not roster:
                continue
            n = reads.pop()
        if roster and n not in roster:
            continue
        out[tid] = n
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--game", required=True, help="game.json with periods / rosters")
    ap.add_argument("--fps", type=float, default=10.0)
    args = ap.parse_args(argv)
    run = Path(args.run)
    game = json.loads(Path(args.game).read_text(encoding="utf8"))
    tr = pd.read_pickle(run / "tracks.pkl")
    jersey = json.loads((run / "jersey.json").read_text()) if (run / "jersey.json").exists() else {}
    nums = identities(tr, jersey, game.get("rosters", {}))
    (run / "player_numbers.json").write_text(json.dumps(nums, indent=0))
    tr["number"] = tr.track.map(nums)

    ft = frame_table(tr)
    ft["period"] = 0
    for i, (a, b) in enumerate(game["periods"], 1):
        ft.loc[ft.t.between(a, b), "period"] = i
    live = (ft.period > 0) & (ft.n_us + ft.n_them >= 6)

    # attack direction per period from the "offence stands further from the basket" signal
    attack = {}
    for per in sorted(ft.period.unique()):
        if per == 0:
            continue
        f = ft[(ft.period == per) & live]
        sig = {s: offence_by_distance(tr, f.index[(f.x_med > 14) == (s == "right")], s) for s in ("left", "right")}
        us_right = sig["right"] - sig["left"] > 0
        attack[(per, "us")] = "right" if us_right else "left"
        attack[(per, "them")] = "left" if us_right else "right"
        print(f"period {per}: us attacks {attack[(per, 'us')]}  (signal R {sig['right']:+.2f}, L {sig['left']:+.2f})")
    poss = segment_possessions(tr, ft, live, args.fps, attack)
    poss.to_pickle(run / "possessions.pkl")
    pm = possession_metrics(tr, poss)
    pm.to_pickle(run / "possession_metrics.pkl")
    tr.to_pickle(run / "tracks.pkl")
    print(f"{len(poss)} possessions; by offence:", poss.offence.value_counts().to_dict())
    print(pm.groupby("offence")[["dur", "spacing", "off_hull", "def_hull", "guard_gap", "paint_share",
                                  "three_share", "cross_time"]].mean().round(2).T)


if __name__ == "__main__":
    main()
