"""Static figures for the report (matplotlib, PNG).

Default palette (validated with the dataviz validator, light surface #fcfcfb):
  us   = violet #6250d6
  them = aqua   #1baf7a
Override per game with us.hex / them.hex in game.json (call `use_game`). Aqua is < 3:1 on the light
surface, so every chart direct-labels its marks.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .court import LENGTH, WIDTH, court_polylines, hoop_xy
from .game_config import DEFAULT_HEX, DEFAULT_NAME, team, team_hex, team_name

SURFACE = "#fcfcfb"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
TEAM = dict(DEFAULT_HEX)
TEAM_NAME = dict(DEFAULT_NAME)


def use_game(game: dict):
    """Set chart team colours / legend names from game.json (in place, so `from .plots import TEAM` sees it)."""
    for side, tag in (("us", "us"), ("them", "opp")):
        TEAM[side] = team_hex(game, side)
        TEAM_NAME[side] = f"{team_name(game, side)} ({tag})" if team(game, side).get("team") else DEFAULT_NAME[side]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "font.sans-serif": ["Microsoft YaHei", "Noto Sans CJK SC", "PingFang SC", "DejaVu Sans"],
    "axes.unicode_minus": False,
})


def team_cmap(team):
    return LinearSegmentedColormap.from_list(team, [SURFACE, TEAM[team]])


def draw_court(ax, half=False, color=MUTED, lw=1.0):
    for pl in court_polylines():
        if half:
            pl = pl[pl[:, 0] >= LENGTH / 2 - 1e-6]
            if len(pl) < 2:
                continue
        ax.plot(pl[:, 0], pl[:, 1], color=color, lw=lw, zorder=3)
    for side in (("right",) if half else ("left", "right")):
        hx, hy = hoop_xy(side)
        ax.add_patch(plt.Circle((hx, hy), 0.23, fill=False, color=INK2, lw=1.4, zorder=4))
    ax.set_xlim((LENGTH / 2 - 0.5) if half else -0.5, LENGTH + 0.5)
    ax.set_ylim(-0.5, WIDTH + 0.5)
    ax.set_aspect("equal")
    ax.axis("off")


def to_attack_right(x, y, side):
    """Mirror coordinates so the attacked basket is always on the right."""
    x, y = np.asarray(x), np.asarray(y)
    return (x, y) if side == "right" else (LENGTH - x, WIDTH - y)


def heatmap(ax, x, y, team, title, bins=(28, 15), half=True):
    rng = [[LENGTH / 2 if half else 0, LENGTH], [0, WIDTH]]
    h, xe, ye = np.histogram2d(x, y, bins=(bins[0] // (2 if half else 1), bins[1]), range=rng)
    h = h / max(h.sum(), 1)
    ax.imshow(h.T, origin="lower", extent=[xe[0], xe[-1], ye[0], ye[-1]], cmap=team_cmap(team),
              interpolation="bilinear", vmin=0, vmax=np.quantile(h[h > 0], 0.97) if (h > 0).any() else 1, zorder=1)
    draw_court(ax, half=half)
    ax.set_title(title, loc="left")


def bar_compare(ax, labels, us_vals, them_vals, fmt="{:.1f}", title=""):
    """Grouped horizontal bars, us vs them, direct-labelled."""
    y = np.arange(len(labels))
    h = 0.36
    for off, vals, side in ((-h / 2 - 0.02, us_vals, "us"), (h / 2 + 0.02, them_vals, "them")):
        ax.barh(y + off, vals, height=h, color=TEAM[side], label=TEAM_NAME[side])
        for yi, v in zip(y + off, vals):
            if np.isfinite(v):
                ax.text(v, yi, " " + fmt.format(v), va="center", ha="left", color=INK2, fontsize=9)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title(title, loc="left")
