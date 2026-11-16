"""Replication of Zarattini & Aziz (SSRN 4416622) 5-minute ORB, on QQQ.

The paper is a first-hand source -- its Table 1 is reproduced below from the
PDF, not from a summary of it. That matters here: the two earlier rule sets
tested in this repo came from third-party summaries, and one of them got the
time exit wrong by an hour.

    Underlying     QQQ or TQQQ
    Entry          open of the SECOND 5-minute candle ("no slippage in fills")
    Direction      first 5-minute candle up -> long, down -> short,
                   doji (open == close) -> no trade
    Stop           low of the first candle (long) / high (short); that
                   distance IS R
    Target         10R or end of day, whichever comes first
    Risk           1% of account      Max leverage  4x
    Capital        $25,000            Commission    $0.0005 per share
    Shares         int[ min( A*0.01/R , 4*A/P ) ],  P = second candle's open
    Period         2016-01-01 .. 2023-02-17
    Reported       676% total, 33%/yr, Sharpe 1.13, MDD 22%, 1,795 trades,
                   51% long / 49% short, win rate 24%, mean +0.13 R

This is NOT the strategy the other two rounds tested. There is no breakout
of a range: the direction comes from the sign of the first candle and the
entry is unconditional at 09:35. A 10R target is in practice no target at
all -- 10R is 2.37% of price at the median and is reached on 5.4% of days,
so 93-95% of trades are closed at the bell. Structurally this is "stop, then
hold to the close", which is exactly the bucket this repo has already shown
carries QQQ's intraday drift. Hence the two direction controls below.

THREE NUMBERS, NOT ONE
----------------------
The 4x cap binds on 53.5% of trades, and it binds hardest where R is
smallest. Those trades carry full weight in an unweighted mean R and almost
none in the dollar curve, so the paper's +0.13 R and its +676% are weighted
differently and cannot both be checked against one statistic:

    A1  unweighted mean R      -> against the paper's +0.13 R
    A2  exposure-weighted R    -> the bridge; what the strategy actually earned
    A3  dollar curve           -> against Table 2's 676%

Pre-registration: backtest/results/20261115_zarattini_qqq/PREREG.md, written
before any P&L was computed.

    uv run backtest/orb_zarattini.py                       # paper's own costs
    uv run backtest/orb_zarattini.py --spread-cents 1      # plus a spread
    uv run backtest/orb_zarattini.py --controls            # C1/C2 direction
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backtest.orb import load_1m, sessions, hours        # noqa: E402

COMMISSION_PER_SHARE = 0.0005        # one way, as the paper states
RISK_FRAC = 0.01
MAX_LEV = 4.0
EQUITY0 = 25_000.0
TARGET_R = 10.0
OVERLAP_END = "2023-02-17"           # the paper's last day


def first_candle(g: pd.DataFrame):
    """The first 5-minute candle, built from 1m bars 09:31..09:35.

    Its OPEN is the 09:31 bar's open and its CLOSE is the 09:35 bar's close,
    which is what decides the direction; its high/low over those five bars
    are what the stop is placed at.
    """
    gh = hours(g.index)
    c1 = g[(gh > 9.5) & (gh <= 9.5 + 5 / 60)]
    if len(c1) != 5:
        return None
    return dict(open=float(c1.open.iloc[0]), close=float(c1.close.iloc[-1]),
                high=float(c1.high.max()), low=float(c1.low.min()))


def trade(g: pd.DataFrame, direction: str | None = None) -> dict | None:
    """One session's trade, or None.

    `direction` overrides the rule's own signal -- that is how the controls
    are run through identical geometry, so that the only thing that differs
    between the rule and a control is which way the position points.
    """
    c1 = first_candle(g)
    if c1 is None:
        return None
    if direction is None:
        if c1["close"] > c1["open"]:
            direction = "long"
        elif c1["close"] < c1["open"]:
            direction = "short"
        else:
            return None                     # doji: the paper stands aside

    after = g[hours(g.index) > 9.5 + 5 / 60]
    if after.empty:
        return None
    entry = float(after.open.iloc[0])        # second 5m candle's open
    stop = c1["low"] if direction == "long" else c1["high"]
    risk = (entry - stop) if direction == "long" else (stop - entry)
    if risk <= 0:
        return None
    target = entry + TARGET_R * risk if direction == "long" \
        else entry - TARGET_R * risk

    if direction == "long":
        sl = np.flatnonzero((after.low <= stop).to_numpy())
        tp = np.flatnonzero((after.high >= target).to_numpy())
        mfe = (float(after.high.max()) - entry) / risk
        mae = (entry - float(after.low.min())) / risk
    else:
        sl = np.flatnonzero((after.high >= stop).to_numpy())
        tp = np.flatnonzero((after.low <= target).to_numpy())
        mfe = (entry - float(after.low.min())) / risk
        mae = (float(after.high.max()) - entry) / risk

    first_sl = sl[0] if sl.size else len(after)
    first_tp = tp[0] if tp.size else len(after)
    gap = False
    if first_sl == len(after) and first_tp == len(after):
        exit_px, result = float(after.close.iloc[-1]), "eod"
    elif first_sl <= first_tp:
        # Did the bar that triggered the stop OPEN beyond it? At a median R
        # of $0.84 this is a real question, unlike the wide stops of the
        # earlier rounds where it was measured at 0.0%.
        bar = after.iloc[first_sl]
        gap = (bar.open < stop) if direction == "long" else (bar.open > stop)
        exit_px, result = (float(bar.open) if gap else stop), "stop"
    else:
        exit_px, result = target, "target"

    r = (exit_px - entry) / risk if direction == "long" \
        else (entry - exit_px) / risk
    return dict(direction=direction, entry=entry, stop=stop, risk=risk,
                exit_px=exit_px, result=result, r=r, mae_r=mae, mfe_r=mfe,
                gap=gap)


def run(rth: pd.DataFrame, spread_cents: float = 0.0,
        direction_mode: str = "rule", seed: int = 0) -> pd.DataFrame:
    """Walk every session. `direction_mode` is rule | long | random."""
    rng = np.random.default_rng(seed)
    cost_per_share = 2 * COMMISSION_PER_SHARE + spread_cents / 100.0
    rows, equity = [], EQUITY0
    for day, g in sessions(rth):
        if direction_mode == "rule":
            d = None
        elif direction_mode in ("long", "short"):
            d = direction_mode
        else:
            d = "long" if rng.random() < 0.5 else "short"
        t = trade(g, d)
        if t is None:
            continue
        want = (RISK_FRAC * equity) / t["risk"]          # intended shares
        cap = (MAX_LEV * equity) / t["entry"]
        shares = int(min(want, cap))
        # Cost in R is size-free: (shares*c)/(shares*risk) = c/risk.
        r_net = t["r"] - cost_per_share / t["risk"]
        pnl = 0.0
        if shares > 0:
            gross = shares * (t["exit_px"] - t["entry"])
            if t["direction"] == "short":
                gross = -gross
            pnl = gross - shares * cost_per_share
            equity += pnl
        rows.append(dict(
            day=day.date(), direction=t["direction"],
            entry=round(t["entry"], 4), stop=round(t["stop"], 4),
            risk=round(t["risk"], 4),
            risk_pct=round(100 * t["risk"] / t["entry"], 4),
            result=t["result"], gap=t["gap"],
            r=round(t["r"], 4), r_net=round(r_net, 4),
            mae_r=round(t["mae_r"], 3), mfe_r=round(t["mfe_r"], 3),
            shares=shares, want=round(want, 1),
            # exposure actually taken, as a share of what 1% risk asked for
            w=round(min(1.0, cap / want), 4) if want > 0 else 0.0,
            capped=want > cap,
            cost_r=round(cost_per_share / t["risk"], 5),
            pnl=round(pnl, 2), equity=round(equity, 2)))
    return pd.DataFrame(rows)


def boot(x: np.ndarray, w: np.ndarray | None = None, reps: int = 10000,
         seed: int = 7, block: int = 1) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    if n == 0:
        return (float("nan"), float("nan"))
    if block <= 1:
        idx = rng.integers(0, n, size=(reps, n))
    else:
        nb = int(np.ceil(n / block))
        st = rng.integers(0, max(1, n - block + 1), size=(reps, nb))
        idx = (st[:, :, None] + np.arange(block)[None, None, :]
               ).reshape(reps, -1)[:, :n] % n
    if w is None:
        m = x[idx].mean(axis=1)
    else:
        xs, ws = x[idx], w[idx]
        m = (xs * ws).sum(axis=1) / ws.sum(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def describe(d: pd.DataFrame, label: str, bh: float = float("nan")) -> None:
    if d.empty:
        print(f"{label}: no trades")
        return
    r = d.r_net.to_numpy()
    w = d.w.to_numpy()
    lo, hi = boot(r)
    blo, bhi = boot(r, block=5)
    wlo, whi = boot(r, w)
    # A slice must be measured from ITS OWN opening equity, not from the
    # account's original $25,000 -- otherwise every slice that ends the run
    # reports the whole run's return.
    eq0 = float(d.equity.iloc[0] - d.pnl.iloc[0])
    eq = np.concatenate([[eq0], d.equity.to_numpy()])
    peak = np.maximum.accumulate(eq)
    yrs = (pd.to_datetime(d.day.iloc[-1]) - pd.to_datetime(d.day.iloc[0])).days / 365.25
    tot = eq[-1] / eq0 - 1
    cagr = (eq[-1] / eq0) ** (1 / yrs) - 1 if yrs > 0 else float("nan")
    daily = np.diff(eq) / eq[:-1]
    sharpe = (daily.mean() / daily.std(ddof=1) * np.sqrt(252)
              if daily.std(ddof=1) > 0 else float("nan"))

    print(f"\n=== {label} ===")
    print(f"trades {len(d)}   long {int((d.direction=='long').sum())} "
          f"({100*(d.direction=='long').mean():.1f}%)   "
          f"short {int((d.direction=='short').sum())}")
    print(f"A1 unweighted mean R  {r.mean():+.4f}   95% [{lo:+.4f}, {hi:+.4f}]"
          f"   block-5 [{blo:+.4f}, {bhi:+.4f}]        paper: +0.13")
    print(f"A2 exposure-weighted  {(r*w).sum()/w.sum():+.4f}"
          f"   95% [{wlo:+.4f}, {whi:+.4f}]")
    print(f"A3 total {100*tot:+.1f}%   CAGR {100*cagr:+.1f}%   "
          f"Sharpe {sharpe:.2f}   MDD {100*((eq-peak)/peak).min():.1f}% "
          f"(single path)        paper: 676% / 33% / 1.13 / 22%")
    if not np.isnan(bh):
        print(f"   buy & hold QQQ over the same days {bh:+.1f}%")
    wins = (d.r_net > 0).mean()
    print(f"win rate {100*wins:.1f}%   paper: 24%")
    print(f"outcome: {d.result.value_counts(normalize=True).round(4).to_dict()}")
    print(f"4x cap bound on {100*d.capped.mean():.1f}% of trades   "
          f"mean exposure taken {100*w.mean():.1f}% of intended   "
          f"paper: 60% / -40%")
    print(f"cost_R: mean {100*d.cost_r.mean():.3f}%  p95 "
          f"{100*d.cost_r.quantile(.95):.3f}%  p99 "
          f"{100*d.cost_r.quantile(.99):.3f}%  max {100*d.cost_r.max():.3f}%")
    stops = d[d.result == "stop"]
    if len(stops):
        print(f"stop gap-through: {int(stops.gap.sum())} of {len(stops)} "
              f"= {100*stops.gap.mean():.2f}% of stopped trades")
    for side in ("long", "short"):
        s = d[d.direction == side]
        if len(s):
            print(f"  {side:<5} n={len(s):4d}  mean R {s.r_net.mean():+.4f}  "
                  f"win {100*(s.r_net>0).mean():.1f}%")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--code", default="US.QQQ")
    ap.add_argument("--spread-cents", type=float, default=0.0,
                    help="round-trip spread in cents, on top of the paper's "
                         "commission (0 = the paper's own assumption)")
    ap.add_argument("--controls", action="store_true",
                    help="also run C1 (random direction) and C2 (always long)")
    ap.add_argument("--draws", type=int, default=200,
                    help="C1 coin-flip draws to average over")
    ap.add_argument("--csv")
    args = ap.parse_args()

    rth = load_1m(args.code)
    ss = list(sessions(rth))
    bh = 100 * (float(ss[-1][1].close.iloc[-1]) / float(ss[0][1].open.iloc[0]) - 1)
    print(f"{args.code}: {len(ss)} tradeable sessions, "
          f"{ss[0][0].date()} .. {ss[-1][0].date()}, buy & hold {bh:+.1f}%")
    print(f"costs: commission ${COMMISSION_PER_SHARE}/share each way"
          + (f" + {args.spread_cents:g}c round-trip spread"
             if args.spread_cents else "  (the paper's own assumption)"))

    d = run(rth, args.spread_cents)
    cut = pd.to_datetime(OVERLAP_END).date()
    describe(d[pd.to_datetime(d.day).dt.date <= cut],
             f"OVERLAP 2019-01..{OVERLAP_END}  (replication check)")
    describe(d[pd.to_datetime(d.day).dt.date > cut],
             f"OUT OF SAMPLE  {OVERLAP_END}..  (power ~50%: "
             f"'unproven' is the likely outcome, not a refutation)")
    describe(d, "FULL SAMPLE 2019..2026", bh)

    if args.controls:
        print()
        print("=== direction controls (identical geometry, "
              "only the side differs) ===")
        cl = run(rth, args.spread_cents, direction_mode="long")
        cs = run(rth, args.spread_cents, direction_mode="short")
        both = cl.merge(cs, on="day", suffixes=("_l", "_s"))
        pair = (both.r_net_l.to_numpy() + both.r_net_s.to_numpy()) / 2.0
        rng = np.random.default_rng(11)
        draws = [np.where(rng.random(len(both)) < 0.5,
                          both.r_net_l.to_numpy(),
                          both.r_net_s.to_numpy()).mean()
                 for _ in range(args.draws)]
        lo, hi = boot(pair)
        print(f"C1 random direction   n={len(both)}  mean R {pair.mean():+.4f}"
              f"  95% [{lo:+.4f}, {hi:+.4f}]")
        print(f"   (exact: a fair coin's expectation is the average of the "
              f"two sides;")
        print(f"    {args.draws} simulated coins give sd {np.std(draws):.4f} "
              f"across draws)")
        print(f"C2 always long        n={len(cl):4d}  mean R "
              f"{cl.r_net.mean():+.4f}")
        print(f"C3 always short       n={len(cs):4d}  mean R "
              f"{cs.r_net.mean():+.4f}")

        # rule - C1 on the days both exist, paired, so the difference is
        # not two means over two different day sets
        rr = d.set_index("day").r_net
        idx = both.day
        rsub = rr.reindex(idx).to_numpy()
        ok = ~np.isnan(rsub)
        diff = rsub[ok] - pair[ok]
        dlo, dhi = boot(diff)
        print(f"rule - C1 = {diff.mean():+.4f}  95% [{dlo:+.4f}, {dhi:+.4f}]"
              f"   n={int(ok.sum())}")
        print("   ^ what the FIRST CANDLE'S SIGN is worth, with the drift "
              "that both legs")
        print("     inherit removed. MQL5's CFD replication got +0.124 "
              "(t=2.1) on NQ.")
        c2sub = cl.set_index("day").r_net.reindex(idx).to_numpy()
        d2 = rsub[ok] - c2sub[ok]
        d2lo, d2hi = boot(d2)
        print(f"rule - C2 = {d2.mean():+.4f}  95% [{d2lo:+.4f}, {d2hi:+.4f}]"
              f"   (long-only reference; a bull sample flatters C2)")

    if args.csv:
        d.to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
