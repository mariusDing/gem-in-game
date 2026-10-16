"""CLI: build report figures + stats json from analysed run (+ optional reviewed events)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .events import event_summary, resolved_events, reviewed_events
from .metrics import player_summary
from .game_config import load_game
from .plots import INK2, GRID, TEAM, TEAM_NAME, bar_compare, heatmap, to_attack_right, use_game

# Figure labels. "en" = English only; "zh" = the bilingual labels (English + Chinese) used so far.
STR = {
    "en": dict(role="CV {role} (estimate)",
               zones_share=["Paint", "Mid-range", "Beyond arc", "Backcourt"],
               zones_shot=["Paint", "Mid-range", "Three", "Zone unclear"]),
    "zh": dict(role="CV {role} ({zh}估计)",
               zones_share=["Paint 禁区", "Mid-range 中距离", "Beyond arc 三分线外", "Backcourt 后场"],
               zones_shot=["Paint 禁区", "Mid-range 中距离", "Three 三分", "Zone unclear 位置不明"]),
}
L = dict(STR["en"])


def role_label(role):
    return L["role"].format(role=role, zh="进攻" if role == "offence" else "防守")


def team_positions(tr, poss, team, role):
    """All positions of `team` while on offence/defence, mirrored to attack right."""
    xs, ys = [], []
    sel = poss[(poss.offence == team) if role == "offence" else (poss.offence != team)]
    fr = tr[tr.team == team].set_index("frame")
    for _, p in sel.iterrows():
        g = fr.loc[fr.index.intersection(p.frames)]
        x, y = to_attack_right(g.cx_s.to_numpy(), g.cy_s.to_numpy(), p.side)
        # the *attacked* basket for the defence is the same basket
        xs.append(x)
        ys.append(y)
    return (np.concatenate(xs), np.concatenate(ys)) if xs else (np.array([]), np.array([]))


def fig_heatmaps(tr, poss, out):
    fig, axs = plt.subplots(2, 2, figsize=(11, 8.6))
    for i, team in enumerate(("us", "them")):
        for j, role in enumerate(("offence", "defence")):
            x, y = team_positions(tr, poss, team, role)
            heatmap(axs[i, j], x, y, team, f"{TEAM_NAME[team]} — {role_label(role)}")
    fig.text(0.5, 0.02, "CV roles inferred from court half; dead balls and role errors remain. Basket on right; density = tracked samples.",
             ha="center", color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_compare(pm, out):
    g = pm.groupby("offence")
    m = {
        "Avg CV half-court segment length (s)": g.dur.mean(),
        "Offensive spacing: avg nearest teammate (m)": g.spacing.mean(),
        "Offensive floor area (m²)": g.off_hull.mean(),
        "Defensive shell area (m²)": pm.groupby(pm.offence.map({"us": "them", "them": "us"})).def_hull.mean(),
        "Avg distance attacker→closest defender (m)": g.guard_gap.mean(),
        "From segment start until 4 attackers detected over halfcourt (s)": g.cross_time.mean(),
    }
    fig, axs = plt.subplots(len(m), 1, figsize=(9, 1.25 * len(m) + 0.6))
    for ax, (lab, s) in zip(axs, m.items()):
        bar_compare(ax, [""], [s.get("us", np.nan)], [s.get("them", np.nan)], title=lab)
        ax.legend().set_visible(ax is axs[0])
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_zone_share(pm, out):
    g = pm.groupby("offence")[["paint_share", "mid_share", "three_share"]].mean()
    # Shares include all tracked attackers, including players still in backcourt.
    g["backcourt_share"] = 1 - g.sum(axis=1)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    bar_compare(ax, L["zones_share"],
                [g.loc["us", c] * 100 if "us" in g.index else np.nan for c in g.columns],
                [g.loc["them", c] * 100 if "them" in g.index else np.nan for c in g.columns],
                fmt="{:.0f}%", title="Mean per-segment position shares (CV roles; equal segment weights)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_player(tr, poss, track_ids, label, out):
    t = tr[tr.track.isin(track_ids)]
    # Simultaneous tracklets with the same OCR number are ambiguous identities.
    t = t[~t.duplicated("frame", keep=False)]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
    for ax, role in zip(axs[:2], ("offence", "defence")):
        xs, ys = [], []
        sel = poss[(poss.offence == "us") if role == "offence" else (poss.offence == "them")]
        tt = t.set_index("frame")
        for _, p in sel.iterrows():
            g = tt.loc[tt.index.intersection(p.frames)]
            x, y = to_attack_right(g.cx_s.to_numpy(), g.cy_s.to_numpy(), p.side)
            xs.append(x)
            ys.append(y)
        x = np.concatenate(xs) if xs else np.array([])
        y = np.concatenate(ys) if ys else np.array([])
        heatmap(ax, x, y, "us", f"{label} — {role_label(role)}")
    ax = axs[2]
    mins = np.sort(t.t.unique()) / 60
    ax.eventplot([mins], colors=[TEAM["us"]], lineoffsets=0.5, linelengths=0.8)
    ax.set_xlim(0, tr.t.max() / 60)
    ax.set_yticks([])
    ax.set_xlabel("game video minute")
    ax.set_title(f"{label}: unambiguous OCR tracking samples", loc="left")
    ax.xaxis.grid(True, color=GRID)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_timeline(poss, events, out):
    fig, ax = plt.subplots(figsize=(13, 2.8))
    for _, p in poss.iterrows():
        ax.barh(0 if p.offence == "us" else 1, p.dur / 60, left=p.start / 60, height=0.7,
                color=TEAM[p.offence], edgecolor="#fcfcfb", linewidth=1)
    if events is not None and len(events):
        for _, e in events.iterrows():
            pts = e.get("points", 0) or 0
            if pts > 0:
                yy = 0 if e.offence == "us" else 1
                ax.text(e.t / 60, yy - 0.5, f"+{int(pts)}", ha="center", va="bottom", fontsize=7, color=INK2)
    ax.set_yticks([0, 1], [TEAM_NAME["us"], TEAM_NAME["them"]])
    ax.invert_yaxis()
    ax.set_xlabel("game video minute")
    ax.set_title("CV segments (bar = half-court segment; +N = reviewed points)", loc="left")
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_score_flow(events, out):
    e = events[events.event.isin(["shot", "free_throws"]) & events.points.notna()].sort_values("t")
    if e.empty or e.points.sum() <= 0:
        Path(out).unlink(missing_ok=True)
        return
    us = (e.points.fillna(0) * (e.offence == "us")).cumsum()
    th = (e.points.fillna(0) * (e.offence == "them")).cumsum()
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.step(e.t / 60, us, where="post", color=TEAM["us"], lw=2)
    ax.step(e.t / 60, th, where="post", color=TEAM["them"], lw=2)
    ax.text(e.t.iloc[-1] / 60, us.iloc[-1], f"  {TEAM_NAME['us']} {int(us.iloc[-1])}", va="center", color=INK2)
    ax.text(e.t.iloc[-1] / 60, th.iloc[-1], f"  {TEAM_NAME['them']} {int(th.iloc[-1])}", va="center", color=INK2)
    ax.set_xlabel("game video minute")
    ax.set_title("Reviewed points only (partial; uncertain outcomes excluded)", loc="left")
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_shot_zones(events, out):
    """Display unresolved outcomes separately, rather than counting them as misses."""
    zones = ["paint", "midrange", "three", None]
    zl = L["zones_shot"]
    sh = events[events.event == "shot"]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    y = np.arange(len(zones))
    for off, team in ((-0.2, "us"), (0.2, "them")):
        s = sh[sh.offence == team]
        groups = [(s.zone == z) if z is not None else s.zone.isna() for z in zones]
        mk = np.array([(g & (s.result == "made")).sum() for g in groups])
        miss = np.array([(g & (s.result == "missed")).sum() for g in groups])
        unknown = np.array([(g & s.result.isna()).sum() for g in groups])
        ax.barh(y + off, mk, height=0.36, color=TEAM[team], label=f"{TEAM_NAME[team]} made")
        ax.barh(y + off, miss, left=mk, height=0.36, color=TEAM[team], alpha=0.3,
                label=f"{TEAM_NAME[team]} missed")
        ax.barh(y + off, unknown, left=mk + miss, height=0.36, color="none", edgecolor=TEAM[team],
                hatch="///", label=f"{TEAM_NAME[team]} unknown")
        for yi, m_, n_, u_ in zip(y + off, mk, miss, unknown):
            ax.text(m_ + n_ + u_, yi, f"  {m_} made / {n_} missed / {u_} ?", va="center", color=INK2, fontsize=8)
    ax.set_yticks(y, zl)
    ax.invert_yaxis()
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.margins(x=0.45)
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    ax.set_title("Reviewed shot actions by zone (unknown outcomes shown separately)", loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--focus", default=None,
                    help="our player's jersey number for a player figure (default: us.focus_player in --game, else none)")
    ap.add_argument("--game", default=None, help="game.json: team names + colours, focus player")
    ap.add_argument("--lang", choices=sorted(STR), default="en", help="figure labels: en, or zh = English + Chinese")
    ap.add_argument("--events", default=None, help="reviewed events json")
    ap.add_argument("--coverage", default=None, help="review coverage JSON list, one record per segment")
    args = ap.parse_args(argv)
    run, out = Path(args.run), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    game = load_game(args.game)
    use_game(game)
    L.update(STR[args.lang])
    focus = args.focus or (game.get("us") or {}).get("focus_player")
    tr = pd.read_pickle(run / "tracks.pkl")
    poss = pd.read_pickle(run / "possessions.pkl")
    pm = pd.read_pickle(run / "possession_metrics.pkl")
    events = None
    if args.events:
        events = reviewed_events(json.loads(Path(args.events).read_text(encoding="utf8")))
        if not events.pid.isin(poss.index).all():
            raise ValueError("Reviewed events reference unknown possession ids")
    accepted = resolved_events(events) if events is not None else None

    fig_heatmaps(tr, poss, out / "heatmaps_team.png")
    fig_compare(pm, out / "team_compare.png")
    fig_zone_share(pm, out / "zone_share.png")
    fig_timeline(poss, accepted, out / "possession_timeline.png")
    if focus:
        focus_tracks = tr[(tr.team == "us") & (tr.number == str(focus))].track.unique()
        fig_player(tr, poss, focus_tracks, f"#{focus}", out / f"player_{focus}.png")

    stats = {"possessions": poss.offence.value_counts().to_dict(),
             "method_notes": ["Possessions are CV half-court segments, not official ball possessions.",
                              "CV offence roles are inferred from court half; visual review found errors and dead-ball segments.",
                              "Tracking coverage is the union of recognised frames, not minutes played.",
                              "Movement omits simultaneous conflicting jersey identities; OCR errors remain possible.",
                              "Position shares describe tracked player samples, not shot attempts.",
                              "Team comparison metrics weight each CV segment equally, including short and dead-ball segments.",
                              "Cross time starts at the CV segment boundary, not a confirmed change of possession."],
             "possession_metrics_mean": pm.groupby("offence").mean(numeric_only=True).round(3).to_dict()}
    if focus:
        stats["focus_player"] = player_summary(tr, focus_tracks)
    players = {}
    for team in ("us", "them"):
        for n, g in tr[(tr.team == team) & tr.number.notna()].groupby("number"):
            players[f"{team}#{n}"] = player_summary(tr, g.track.unique())
    stats["players"] = players
    if events is not None:
        stats["reviewed_events"] = event_summary(events)
        fig_shot_zones(accepted, out / "shot_zones.png")
        fig_score_flow(accepted, out / "score_flow.png")
    if args.coverage:
        coverage = json.loads(Path(args.coverage).read_text(encoding="utf8"))
        ids = [c["pid"] for c in coverage]
        if len(set(ids)) != len(ids) or not set(ids).issubset(set(poss.index)):
            raise ValueError("Review coverage contains duplicate or unknown segment ids")
        stats["review_coverage"] = dict(total_segments=len(poss), reviewed_segments=len(ids),
                                        reviewed_sheets=sum(len(c["reviewed_sheets"]) for c in coverage))
    def finite_json(value):
        if isinstance(value, dict):
            return {k: finite_json(v) for k, v in value.items()}
        if isinstance(value, list):
            return [finite_json(v) for v in value]
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return None
        return value

    (out / "stats.json").write_text(json.dumps(finite_json(stats), indent=1, default=float, allow_nan=False))
    if focus:
        print(json.dumps(stats["focus_player"], indent=1))
    print("figures ->", out)


if __name__ == "__main__":
    main()
