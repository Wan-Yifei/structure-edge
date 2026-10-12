"""Did the entry actually happen where the write-up says it did?

The journal labels most trades with a profile level -- POC, VAH, VAL. That
label is a claim about price, and the statement knows the price that was
actually paid, so the claim is testable. This rebuilds the volume profile as it
stood at the moment of entry and measures how far the fill was from the level
it was supposed to be at.

Two things make the test honest rather than decorative:

**The profile is the one IVB actually uses.** It anchors at the cash open, so a
regular-session trade is scored against the profile developing from 09:30 up to
the fill -- what was on screen, not what the session turned out to be. Outside
the cash session there is no developing profile to anchor to, so overnight and
pre-market trades carry the previous regular session forward, and after-hours
trades use the session that has just closed. In both cases that is the most
recent *completed* 09:30-16:00 profile before the fill.

**Levels live on SOXL.** The notes say "SOXL POC" even for the SOXS trades,
because SOXS is just how a short gets expressed. So the profile is always
SOXL's, and a SOXS fill is compared using SOXL's price at that minute rather
than the SOXS price paid.

Distance is reported both as a percentage of price and in profile bins, because
neither alone is meaningful: 0.3% is tight on a quiet morning and loose on a
10% day, while a bin is only as wide as the session's range divided by 100.
"""

from __future__ import annotations

import pathlib
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from strategy.journal.reconcile import session_of
from strategy.session_vp.profile import compute_value_area
from strategy.smc.fvg import compute_volume_profile

LEVELS = ("poc", "vah", "val")

RTH_OPEN, RTH_CLOSE = 9.5, 16.0

# Minimum bars before the profile means anything. A trade three minutes into a
# session has no profile to be near or far from, and scoring it would invent a
# verdict out of two candles.
MIN_BARS = 15


def profile_window(ts: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp, str]:
    """(start, end, which) of the profile IVB would have been reading.

    In the cash session the profile is still forming, so it is cut at the fill.
    Everywhere else the last completed cash session is carried forward -- the
    same profile the chart would still be showing.
    """
    day_open = ts.normalize() + timedelta(hours=RTH_OPEN)
    day_close = ts.normalize() + timedelta(hours=RTH_CLOSE)
    if day_open <= ts < day_close:
        return day_open, ts, "developing"
    if ts >= day_close:                  # after-hours: today's session is done
        return day_open, day_close, "prior RTH"
    # Before the open -- overnight or pre-market. Step back to the last
    # weekday; the fetcher simply returns no bars for a holiday, which the
    # bar-count guard then rejects.
    d = ts.normalize() - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d + timedelta(hours=RTH_OPEN), d + timedelta(hours=RTH_CLOSE), "prior RTH"


def check(positions: pd.DataFrame, ledger: pd.DataFrame,
          klines: pd.DataFrame, n_bins: int = 100) -> pd.DataFrame:
    """One row per journal trade that names a level and has a known fill.

    `klines` is SOXL 1-minute OHLCV with a `time_key` column.
    """
    k = klines.copy()
    k["t"] = pd.to_datetime(k.time_key)
    k = k.sort_values("t").reset_index(drop=True)

    j = ledger[ledger.Source == "journal+account"].copy()
    j = j[j.Entry.isin(LEVELS and [l.upper() for l in LEVELS])]
    if j.empty:
        return pd.DataFrame()

    rows = []
    for _, r in j.iterrows():
        try:
            ts = pd.Timestamp(f"{r.date} {r.Time}")
        except Exception:
            continue
        # The journal's date is the trading day; an evening fill belongs to the
        # previous calendar date.
        if ts.hour >= 20:
            ts -= timedelta(days=1)
        start, end, which = profile_window(ts)
        win = k[(k.t >= start) & (k.t <= end)]
        if len(win) < MIN_BARS:
            rows.append({"date": r.date, "Time": r.Time, "Entry": r.Entry,
                         "bars": len(win), "profile": which,
                         "verdict": "not enough bars for a profile"})
            continue
        edges, vols = compute_volume_profile(win, n_bins=n_bins)
        if edges is None:
            continue
        va = compute_value_area(edges, vols)
        if not va["poc"]:
            continue
        binw = float(edges[1] - edges[0])

        # SOXL's price at the fill: the fill itself for a SOXL trade, the
        # minute's close for a SOXS one.
        if r.Symbol == "SOXL":
            px = float(r.Notional) / float(r.Qty)
        else:
            near = k[k.t <= ts]
            if near.empty:
                continue
            px = float(near.iloc[-1].close)

        d = {lv: px - va[lv] for lv in LEVELS}
        claimed = r.Entry.lower()
        nearest = min(LEVELS, key=lambda lv: abs(d[lv]))
        rows.append({
            "date": r.date, "Time": r.Time, "Symbol": r.Symbol,
            "Session": r.Session, "Entry": r.Entry, "W/L": r["W/L"],
            "profile": which, "bars": len(win), "px": round(px, 3),
            "poc": round(va["poc"], 3), "vah": round(va["vah"], 3),
            "val": round(va["val"], 3), "bin": round(binw, 4),
            "dist_pct": round(100 * d[claimed] / px, 3),
            "dist_bins": round(d[claimed] / binw, 2) if binw else np.nan,
            # Distance alone misreads a breakout: price that has broken
            # through VAL is a long way from it by construction, and that is
            # the trade rather than a mislabelled entry. The side says which.
            "side": ("above" if d[claimed] > binw / 2 else
                     "below" if d[claimed] < -binw / 2 else "at"),
            "nearest": nearest.upper(),
            "matches": nearest.upper() == r.Entry,
            "PnL$": r["PnL$"],
        })
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> str:
    if df.empty:
        return "   no matched trade names a level yet"
    ok = df[df.verdict.isna()] if "verdict" in df else df
    ok = ok[ok.dist_bins.notna()] if "dist_bins" in ok else ok
    if ok.empty:
        return "   nothing with enough session history to score"
    out = [f"   {len(ok)} trades scored against the profile as it stood at entry",
           f"   label matches the nearest level: {int(ok.matches.sum())}"
           f"/{len(ok)} ({100*ok.matches.mean():.0f}%)",
           f"   |distance| to the claimed level: median "
           f"{ok.dist_bins.abs().median():.1f} bins "
           f"({ok.dist_pct.abs().median():.2f}% of price)",
           "   side of the claimed level: "
           + ", ".join(f"{k} {v}" for k, v in ok.side.value_counts().items())]
    for lv in ("POC", "VAH", "VAL"):
        x = ok[ok.Entry == lv]
        if len(x):
            out.append(f"     {lv}: n={len(x):>2}  matches {int(x.matches.sum()):>2}"
                       f"  median |dist| {x.dist_bins.abs().median():.1f} bins")
    return "\n".join(out)
