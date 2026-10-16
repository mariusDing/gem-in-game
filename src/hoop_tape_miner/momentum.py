"""Momentum figure: score margin over our-video time, with scoring runs, foul/FT markers,
and (optionally) on-court bands for selected players (from CV jersey identification).

Everything game-specific comes from the data: the score flow from official.json, period breaks
from game.json `periods`, team colours from game.json `us.hex` / `them.hex`. Scoring runs are found
automatically (largest margin swings) unless given with --stretches.

usage: hoop-tape-miner momentum --official official.json --out momentum.png [--game game.json]
           [--lineup lineup_intervals.json --players us#4,us#7,them#11] [--lang en|zh]
           [--stretches "Start:0-450,Their run:450-1272,..."] [--title "..."]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .game_config import find_up, load_game
from .plots import GRID, INK2, MUTED, TEAM, use_game

TEXT = {
    "zh": dict(legend="▼ 对手罚球命中（= 我们犯规送分）  ▲ 我方罚球", ylabel="分差（我方 − 对手）",
               title="比分走势与 momentum", title_run="比分走势与 momentum：{who} {a}–{b}（{t0}–{t1}）",
               us="我 #", them="对 #", xlabel="录像时间（分钟）", half="中场", brk="节间",
               start="开局", finish="收官", run_us="我方 {a}–{b}", run_them="对手 {a}–{b}", plain="{a}–{b}",
               who_us="我方", who_them="对手",
               lanes="在场时段（CV 识别到号码的时间，为下限；空白 ≠ 一定不在场）"),
    "en": dict(legend="▼ their made FT (= points from our fouls)  ▲ our FT", ylabel="Margin (us − them)",
               title="Score margin", title_run="Score margin: {who} {a}–{b} run ({t0}–{t1})",
               us="Us #", them="Them #", xlabel="Video time (min)", half="Half", brk="Break",
               start="Start", finish="Finish", run_us="Our {a}–{b} run", run_them="Their {a}–{b} run", plain="{a}–{b}",
               who_us="our", who_them="their",
               lanes="On-court time (when CV read the jersey number; a lower bound, gaps ≠ off court)"),
}


def _mmss(s: float) -> str:
    return f"{int(s // 60):02d}:{int(s % 60):02d}"


def find_runs(flow: list[dict], min_swing: int = 8, max_runs: int = 3) -> list[tuple[int, int]]:
    """Scoring runs as (i, j) index pairs into `flow`: the run covers events i+1..j (i = -1: from tip-off).

    Greedy: take the window with the largest margin swing, then look for further runs in the
    unclaimed stretches before/after it, as long as the swing is at least `min_swing` points.
    """
    margin = [0] + [e["score_us"] - e["score_them"] for e in flow]  # margin[k + 1] = after event k

    def best(lo, hi):  # largest swing inside margin[lo..hi]; ties -> the shortest window
        top, key = (0, None), (0, 0)
        for a in range(lo, hi + 1):
            for b in range(a + 1, hi + 1):
                k = (abs(margin[b] - margin[a]), a - b)
                if k > key:
                    key, top = k, (k[0], (a, b))
        return top

    runs, gaps = [], [(0, len(margin) - 1)]
    while gaps and len(runs) < max_runs:
        cands = [(best(lo, hi), (lo, hi)) for lo, hi in gaps]
        (sw, win), gap = max(cands, key=lambda c: c[0][0])
        if win is None or sw < min_swing:
            break
        runs.append(win)
        gaps.remove(gap)
        gaps += [g for g in ((gap[0], win[0]), (win[1], gap[1])) if g[1] > g[0]]
    return sorted((a - 1, b - 1) for a, b in runs)


def auto_stretches(flow: list[dict], tx: dict, end_t: float, min_swing: int = 8):
    """[(label, t0, t1)] alternating plain stretches and detected runs, labelled with their partial score."""
    if not flow:
        return [], None
    t = [0.0] + [e["our_video_t"] for e in flow]
    us = [0] + [e["score_us"] for e in flow]
    th = [0] + [e["score_them"] for e in flow]
    runs = find_runs(flow, min_swing)
    cuts, kinds = [0], []
    for i, j in runs:
        a, b = i + 1, j + 1  # indices into t/us/th
        if a > cuts[-1]:
            cuts.append(a)
            kinds.append("plain")
        cuts.append(b)
        kinds.append("run")
    if cuts[-1] < len(t) - 1:
        cuts.append(len(t) - 1)
        kinds.append("plain")
    out, title = [], None
    for n, (a, b, kind) in enumerate(zip(cuts[:-1], cuts[1:], kinds)):
        du, dt = us[b] - us[a], th[b] - th[a]
        t0, t1 = t[a], (end_t if b == len(t) - 1 else t[b])
        if kind == "run":
            mine = du > dt
            lab = tx["run_us" if mine else "run_them"].format(a=max(du, dt), b=min(du, dt))
            sw = abs((du - dt))
            if title is None or sw > title[0]:
                title = (sw, tx["title_run"].format(who=tx["who_us" if mine else "who_them"], a=max(du, dt),
                                                    b=min(du, dt), t0=_mmss(t0), t1=_mmss(t[b])))
        else:
            pre = tx["start"] + " " if n == 0 else tx["finish"] + " " if b == len(t) - 1 else ""
            lab = pre + tx["plain"].format(a=du, b=dt)
        out.append((lab, t0, t1))
    return out, (title[1] if title else None)


def parse_stretches(spec: str) -> list[tuple[str, float, float]]:
    """--stretches "Label:t0-t1,..." (our-video seconds) or a JSON file [[label, t0, t1], ...]."""
    if spec.endswith(".json") and Path(spec).exists():
        return [(str(label), float(a), float(b)) for label, a, b in json.loads(Path(spec).read_text(encoding="utf8"))]
    out = []
    for s in filter(None, (x.strip() for x in spec.split(","))):
        label, rng = s.rsplit(":", 1)
        a, b = rng.split("-")
        out.append((label, float(a), float(b)))
    return out


def breaks(periods) -> list[tuple[float, float, bool]]:
    """Gaps between periods as (t0, t1, is_halftime)."""
    ps = sorted(periods or [])
    gaps = [(a[1], b[0]) for a, b in zip(ps[:-1], ps[1:]) if b[0] > a[1]]
    mid = len(ps) // 2 - 1
    return [(a, b, i == mid) for i, (a, b) in enumerate(gaps)]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="hoop-tape-miner momentum", description=__doc__.splitlines()[0])
    ap.add_argument("--official", required=True)
    ap.add_argument("--game", help="game.json (periods, team colours); default: next to --official or its parent")
    ap.add_argument("--lineup", help="lineup_intervals.json {\"us#4\": [[t0, t1], ...]} for on-court lanes")
    ap.add_argument("--players", help="comma list like us#4,them#11 (default: everyone in --lineup)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lang", choices=sorted(TEXT), default="en")
    ap.add_argument("--stretches", help='"Label:t0-t1,..." or a JSON file; default: detected scoring runs')
    ap.add_argument("--min-run", type=int, default=8, help="smallest margin swing that counts as a run")
    ap.add_argument("--title", help="figure title (default: derived from the biggest run)")
    args = ap.parse_args(argv)
    off = json.loads(Path(args.official).read_text(encoding="utf8"))
    game = load_game(args.game or find_up(Path(args.official).parent))
    use_game(game)
    flow = [e for e in off["score_flow"] if e["our_video_t"] is not None]
    tx = TEXT[args.lang]
    periods = game.get("periods") or []
    end_t = max([p[1] for p in periods] + [e["our_video_t"] for e in flow] + [0])

    if args.stretches:
        stretches, run_title = parse_stretches(args.stretches), None
    else:
        stretches, run_title = auto_stretches(flow, tx, end_t, args.min_run)
    title = args.title or run_title or tx["title"]

    lu = json.loads(Path(args.lineup).read_text(encoding="utf8")) if args.lineup else {}
    if args.players:
        players = [p.strip() for p in args.players.split(",") if p.strip()]
    else:  # everyone with on-court intervals, us first, most time first
        tot = {k: sum(b - a for a, b in v) for k, v in lu.items()}
        players = sorted(lu, key=lambda k: (not k.startswith("us#"), -tot[k]))

    if players:
        fig, (ax, axl) = plt.subplots(2, 1, figsize=(13, 7.2), sharex=True,
                                      gridspec_kw=dict(height_ratios=[2.2, max(len(players), 1) * 0.32]))
    else:
        fig, ax = plt.subplots(figsize=(13, 4.6))
        axl = None
    t = [0] + [e["our_video_t"] / 60 for e in flow]
    margin = [0] + [e["score_us"] - e["score_them"] for e in flow]
    lo, hi = min(margin) - 3, max(margin) + 3
    ax.step(t, margin, where="post", color="#0b0b0b", lw=2)
    ax.axhline(0, color=MUTED, lw=1)
    ax.fill_between(t, margin, 0, step="post", where=np.array(margin) >= 0, color=TEAM["us"], alpha=0.25)
    ax.fill_between(t, margin, 0, step="post", where=np.array(margin) < 0, color=TEAM["them"], alpha=0.25)
    for i, (label, a, b) in enumerate(stretches):
        ax.axvspan(a / 60, b / 60, color=GRID, alpha=0.35 if i % 2 else 0.0, lw=0)
        ax.text((a + b) / 120, hi - 2.4, label, ha="center", va="bottom", fontsize=9, color=INK2)
    for a, b, half in breaks(periods):
        ax.axvspan(a / 60, b / 60, color="#d9d8d3", alpha=0.5, lw=0)
        ax.text((a + b) / 120, lo + 0.6, tx["half"] if half else tx["brk"], ha="center", fontsize=8, color=INK2)
    for e in flow:
        if e["points"] == 1:
            ax.plot(e["our_video_t"] / 60, e["score_us"] - e["score_them"], marker="v" if e["team"] == "them" else "^",
                    color=TEAM[e["team"]], ms=7, zorder=5)
    ax.text(0.3, lo + 2, tx["legend"], color=INK2, fontsize=9)
    ax.set_ylim(lo, hi)
    ax.set_ylabel(tx["ylabel"])
    ax.set_title(title, loc="left")
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    if axl is None:
        ax.set_xlabel(tx["xlabel"])
    else:
        for i, pl in enumerate(players):
            team = pl.split("#")[0]
            for a, b in lu.get(pl, []):
                axl.barh(i, (b - a) / 60, left=a / 60, height=0.6, color=TEAM.get(team, MUTED), alpha=0.9)
        axl.set_yticks(range(len(players)), [p.replace("us#", tx["us"]).replace("them#", tx["them"]) for p in players])
        axl.invert_yaxis()
        axl.set_xlabel(tx["xlabel"])
        axl.set_title(tx["lanes"], loc="left", fontsize=10)
        axl.xaxis.grid(True, color=GRID)
        axl.set_axisbelow(True)
        for i, (_, a, b) in enumerate(stretches):
            axl.axvspan(a / 60, b / 60, color=GRID, alpha=0.35 if i % 2 else 0.0, lw=0)
        for a, b, _ in breaks(periods):
            axl.axvspan(a / 60, b / 60, color="#d9d8d3", alpha=0.5, lw=0)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
