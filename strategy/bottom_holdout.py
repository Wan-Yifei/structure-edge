"""A holdout for the bottom study that grows one session at a time.

The 526-event sample ends 2026-10-12 and its test half has been scored. Every
later session is data the study never saw, and the only honest way to keep it
that way is to write it down before anyone looks: this module extracts a
session's events and their features, appends them, and reports NOTHING about
outcomes. Checking as you go is how a clean holdout stops being one.

Sizing, so the wait is not a surprise. Separating 46% from 33% at the usual
thresholds needs about 174 events a side; at 6.25 events a session with
roughly a third carrying delta_r >= 0, that is near 80 sessions. Four months,
not an afternoon. The first two sessions (2026-10-13, 2026-10-14) gave 15
events between them and settle nothing -- the positive side ran 2/4 against
the study's 46%, with a 95% interval of [7%, 93%].

APPEND ONLY, and a day already written is never rebuilt. A later change to
the selection or the features would otherwise rewrite history to agree with
whatever the code does now, which is the same silent rewrite `feeds/depth_bars`
and `feeds/book_ladder` exist to prevent. Changing the rules means starting a
new file, not refreshing this one.

Selection and labelling are copied from the study unchanged:
  at a low      the bar's low is the lowest of the last 30 bars, itself
                included, so backward-looking only
  real decline  the high over those 30 bars is at least 1 ATR above it
  spacing       at least 15 bars since the previous accepted event
  label         +2 ATR against -1 ATR, 30 bars, first touched decides
  guards        at least 300 signed shares on the marked bar and the one
                before it, and at least 25 bars into the session
"""

from __future__ import annotations

import pathlib
from datetime import date

import numpy as np
import pandas as pd

_ROOT = pathlib.Path(__file__).parent.parent
_CACHE = _ROOT / "db" / "cache" / "bottom_holdout"

# The study's sample ends here, so the holdout starts the session after.
# Without this a routine "--days 3" sweep walks back into scored data and the
# file stops being a holdout -- which it did on the first run, quietly adding
# seven events from 2026-10-12.
HOLDOUT_START = "2026-10-13"

RTH = (9.5, 16.0)
LOOK, MIN_DROP_ATR = 30, 1.0
UP_ATR, DN_ATR = 2.0, 1.0
FWD, GAP, BEFORE, AFTER = 30, 15, 45, 35
MIN_SIGNED, MIN_BARS_IN = 300.0, 25

COLUMNS = ["day", "tm", "low", "close", "atr", "lab",
           "delta_r", "flip", "sell_drop", "signed", "neutral"]


def _cache_path(code: str) -> pathlib.Path:
    return _CACHE / f"{code.replace('.', '_')}_events.parquet"


def _atr(h, l, c, n=14):
    tr = np.full(len(h), np.nan)
    for i in range(1, len(h)):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    a = np.full(len(h), np.nan)
    if len(h) > n:
        a[n] = np.nanmean(tr[1:n + 1])
        for i in range(n + 1, len(h)):
            a[i] = (a[i - 1] * (n - 1) + tr[i]) / n
    return a


def build_day(code: str, day: date | str) -> pd.DataFrame:
    """Events for one session. Empty frame when the day cannot be scored.

    A session is skipped rather than half-scored: fewer than 380 regular-hours
    bars, or delta coverage under 300 minutes, means the features would come
    from a different population than the rest of the file.
    """
    from feeds import delta_bars
    from feeds.fetcher import fetch_klines

    d0 = pd.Timestamp(day).normalize()
    lo = (d0 - pd.Timedelta(days=4)).strftime("%Y-%m-%d")
    hi = (d0 + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    k = fetch_klines(code, "1m", lo, hi, force_refresh=True)
    if k is None or k.empty:
        return pd.DataFrame(columns=COLUMNS)
    k["t"] = pd.to_datetime(k.time_key)
    k = k.sort_values("t").drop_duplicates("t").set_index("t")

    dl = delta_bars.load(code, lo, hi)
    cov = delta_bars.coverage(dl)
    if d0.date() not in set(cov[cov.bars >= 300].index):
        return pd.DataFrame(columns=COLUMNS)
    full = k.join(dl[["buy", "sell", "neutral", "n"]], how="left")
    full[["buy", "sell", "neutral", "n"]] = \
        full[["buy", "sell", "neutral", "n"]].fillna(0.0)

    g = full[full.index.date == d0.date()]
    hh = g.index.hour + g.index.minute / 60
    reg = g[(hh > RTH[0]) & (hh <= RTH[1])].reset_index()
    if len(reg) < 380:
        return pd.DataFrame(columns=COLUMNS)

    H, L, C = (reg.high.to_numpy(float), reg.low.to_numpy(float),
               reg.close.to_numpy(float))
    buy, sell = reg.buy.to_numpy(float), reg.sell.to_numpy(float)
    neu = reg.neutral.to_numpy(float)
    A = _atr(H, L, C)

    out, last = [], -99
    for i in range(BEFORE, len(reg) - FWD - AFTER):
        a = A[i]
        if np.isnan(a) or a <= 0 or i - last < GAP:
            continue
        if L[i] != L[i - LOOK:i + 1].min():
            continue
        if (H[i - LOOK:i + 1].max() - L[i]) < MIN_DROP_ATR * a:
            continue
        c0 = C[i]
        up, dn = c0 + UP_ATR * a, c0 - DN_ATR * a
        lab = None
        for j in range(i + 1, i + 1 + FWD):
            tu, td = H[j] >= up, L[j] <= dn
            if tu and td:
                lab = 0
                break
            if tu:
                lab = 1
                break
            if td:
                lab = 0
                break
        if lab is None:
            continue
        last = i                      # spacing counts accepted events only
        if i < MIN_BARS_IN:
            continue
        dirv, prev = buy[i] + sell[i], buy[i - 1] + sell[i - 1]
        if dirv < MIN_SIGNED or prev < MIN_SIGNED:
            continue
        out.append({
            "day": str(d0.date()), "tm": f"{reg['t'].iloc[i]:%H:%M}",
            "low": float(L[i]), "close": float(c0), "atr": float(a),
            "lab": int(lab),
            "delta_r": (buy[i] - sell[i]) / dirv,
            "flip": (buy[i] - sell[i]) / dirv
                    - (buy[i - 1] - sell[i - 1]) / prev,
            "sell_drop": sell[i] / max(sell[i - 1], 1.0),
            "signed": float(dirv), "neutral": float(neu[i]),
        })
    return pd.DataFrame(out, columns=COLUMNS)


def append_day(code: str, day: date | str,
               cache_path: pathlib.Path | None = None) -> int:
    """Add one session. A day already written is left exactly as it is.

    Returns rows added -- 0 when the day was already there, when it produced
    no events, and when it predates HOLDOUT_START. Three different facts, none
    of them an error.
    """
    p = cache_path or _cache_path(code)
    d = str(pd.Timestamp(day).date())
    if d < HOLDOUT_START:
        return 0                      # the study already scored this session
    old = pd.read_parquet(p) if p.exists() else None
    if old is not None and d in set(old.day):
        return 0
    fresh = build_day(code, day)
    if fresh.empty:
        return 0
    p.parent.mkdir(parents=True, exist_ok=True)
    merged = fresh if old is None else pd.concat([old, fresh])
    merged = merged.sort_values(["day", "tm"]).reset_index(drop=True)
    merged.to_parquet(p, index=False)
    return len(fresh)


def load(code: str, cache_path: pathlib.Path | None = None) -> pd.DataFrame:
    """Read the holdout. Never rebuilt -- see the module docstring."""
    p = cache_path or _cache_path(code)
    if not p.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_parquet(p)


def coverage(code: str, cache_path: pathlib.Path | None = None) -> pd.DataFrame:
    """Sessions and event counts. Deliberately says nothing about outcomes."""
    d = load(code, cache_path=cache_path)
    if d.empty:
        return pd.DataFrame(columns=["events"])
    return d.groupby("day").agg(events=("lab", "size"))


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--codes", nargs="+", default=["US.SOXL"])
    ap.add_argument("--days", type=int, default=2,
                    help="how many days back to sweep, today included")
    ap.add_argument("--coverage", action="store_true")
    args = ap.parse_args()

    for code in args.codes:
        if args.coverage:
            cov = coverage(code)
            n = int(cov.events.sum()) if len(cov) else 0
            print(f"{code}: {len(cov)} sessions, {n} events")
            if len(cov):
                print(cov.tail(10).to_string())
            continue
        today = pd.Timestamp.now().normalize()
        for k in range(args.days):
            day = (today - pd.Timedelta(days=k)).date()
            print(f"  {code} {day}: +{append_day(code, day)} events",
                  flush=True)
        cov = coverage(code)
        n = int(cov.events.sum()) if len(cov) else 0
        print(f"{code}: holdout now {len(cov)} sessions / {n} events",
              flush=True)


if __name__ == "__main__":
    main()
