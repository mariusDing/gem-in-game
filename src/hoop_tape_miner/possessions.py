"""Pass 4: game phases, half-court possessions and attack direction from player positions.

A broadcast camera follows the ball, so the half of the court the ten players occupy
is a strong signal of where the ball is.  A possession is a contiguous stretch of
play in one half; the offence is the team attacking that basket in that period.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .court import LENGTH, hoop_xy


def frame_table(tracks: pd.DataFrame) -> pd.DataFrame:
    """One row per frame: player counts per team and centroid of play."""
    g = tracks[tracks.team.isin(["us", "them"])].groupby("frame")
    ft = pd.DataFrame({
        "t": g.t.first(),
        "n_us": g.apply(lambda d: (d.team == "us").sum()),
        "n_them": g.apply(lambda d: (d.team == "them").sum()),
        "x_med": g.cx.median(),
    })
    return ft


def side_series(ft: pd.DataFrame, fps: float, hyst=2.0, min_dur=2.0) -> pd.Series:
    """Left/right half per frame with hysteresis around midcourt and short-segment removal."""
    side, cur = [], None
    for x in ft.x_med.rolling(int(fps), center=True, min_periods=1).median():
        if cur is None:
            cur = "left" if x < LENGTH / 2 else "right"
        elif cur == "left" and x > LENGTH / 2 + hyst:
            cur = "right"
        elif cur == "right" and x < LENGTH / 2 - hyst:
            cur = "left"
        side.append(cur)
    s = pd.Series(side, index=ft.index)
    # remove segments shorter than min_dur
    seg = (s != s.shift()).cumsum()
    lens = s.groupby(seg).transform("size") / fps
    s[lens < min_dur] = np.nan
    return s.ffill().bfill()


def offence_by_distance(tracks: pd.DataFrame, frames: pd.Index, side: str) -> float:
    """Positive => 'us' is further from the basket than 'them' (i.e. 'us' attacking)."""
    hx, hy = hoop_xy(side)
    d = tracks[tracks.frame.isin(frames) & tracks.team.isin(["us", "them"])]
    d = d.assign(dist=np.hypot(d.cx - hx, d.cy - hy))
    m = d.groupby("team").dist.median()
    return float(m.get("us", np.nan) - m.get("them", np.nan))


def segment_possessions(tracks: pd.DataFrame, ft: pd.DataFrame, live: pd.Series, fps: float,
                        attack: dict) -> pd.DataFrame:
    """attack: {(period, team): side_attacked}. Returns one row per possession."""
    ft = ft[live.reindex(ft.index).fillna(False)]
    side = side_series(ft, fps)
    seg = (side != side.shift()).cumsum()
    # also split when the gap in time is large (dead ball / cut)
    gap = ft.t.diff().fillna(0) > 3
    seg = (seg.diff().fillna(0).ne(0) | gap).cumsum()
    rows = []
    for sid, idx in ft.groupby(seg).groups.items():
        f = ft.loc[idx]
        if f.t.iloc[-1] - f.t.iloc[0] < 3:
            continue
        s = side.loc[idx].iloc[0]
        period = f.period.iloc[0]
        off = "us" if attack.get((period, "us")) == s else "them"
        rows.append(dict(start=f.t.iloc[0], end=f.t.iloc[-1], dur=f.t.iloc[-1] - f.t.iloc[0], side=s,
                         period=period, offence=off, frames=list(idx),
                         dist_signal=offence_by_distance(tracks, idx, s)))
    p = pd.DataFrame(rows)
    p.index.name = "pid"
    return p
