"""Opening-range breakout on SOXL, long only, with fixed-fractional sizing.

WHAT THIS IS, AND IS NOT
------------------------
A reconstruction of an opening-range breakout rule described in a YouTube
video ("How to IMPROVE FaberVaale Strategy", 2026-11). The rules were NOT
taken from the video -- no transcript is published -- but from a
third-party AI summary that contradicts itself in two places (846 vs 878
trades, $181k vs $170k base profit). The primary configuration here uses a
FIVE-minute range, not the thirty described there. So: never report this
under the author's name, and prefer "5-minute range breakout" to "opening
range breakout". Nothing here reproduces anyone's published numbers.

The video's instrument is NQ futures. On SOXL the video's own 30-minute
window leaves 74.3% of trades touching neither the stop nor the target by
the 14:00 flat -- its "1:1" is a time exit wearing a stop's clothes.
Shortening the range to 5 minutes brings that to 31.6%.

WHY THE CONTROL IS THE RESULT
-----------------------------
With a 5-minute range the rule is in the market on 74.2% of days. On an
instrument that went from $8.40 to $298 inside the sample, a long-only
rule that trades three days in four is mostly measuring the drift. So the
decisive statistic is not the rule's own mean R, but the rule MINUS a
matched control that enters every single day at the same time with the
same stop and target and never looks at the breakout condition. The
control is fully tradeable and contains no future information.

Analysis tool only: no database writes, no trade IDs, no algo tag -- the
same footing as backtest/open_bias.py, deliberately not that of
backtest/session_vp_engine.py.

    uv run backtest/orb.py                      # the pre-registered primary
    uv run backtest/orb.py --sweep              # every pre-declared cell
    uv run backtest/orb.py --csv trades.csv
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import duckdb
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

KLINES_DB = ROOT / "db" / "backtest_klines.duckdb"
CODE = "US.SOXL"
RTH_BARS = 390          # a full regular session, asserted not assumed

# An early-close session (July 3, the day after Thanksgiving, Christmas
# Eve) ends at 13:00, but the pull used extended_time/session=ALL, so those
# days still carry 390 stamps -- the last 180 are AFTER-HOURS bars, not
# regular ones. The bar count therefore cannot detect them: on 2019-12-24,
# 118 of those 180 have zero volume. They have to go, because the rule
# exits at 14:00 and on those days 14:00 is an hour into a nearly empty
# book, where the cost assumption does not hold.
#
# The afternoon/morning volume ratio separates them cleanly: the 14 lowest
# sessions in 2019-2026 are all recognisable early closes, running from
# 0.017 to 0.186, and the next session up is 0.378. Anything under this
# threshold is one of them.
HALF_DAY_VOL_RATIO = 0.20

# Pre-registered cells. Declared here so the count is in the source rather
# than in whoever is reading the output: 5 windows x 2 flat times x 3 cost
# levels, with one primary configuration named below.
WINDOWS = (5, 10, 15, 20, 30)
FLATS = (14.0, 16.0)
COSTS = (1.0, 3.0, 5.0)
PRIMARY = dict(or_min=5, flat=14.0, cost_cents=3.0, entry_tf="5min")


# ── data ──────────────────────────────────────────────────────────────────

def load_1m(code: str) -> pd.DataFrame:
    """RTH 1-minute bars indexed on the bar's END stamp.

    moomoo's time_key is the END of the bar (CHANGELOG v0.92.0 is the
    forward-leak that came from forgetting it), so an L-minute range from
    09:30 is the bars stamped 09:31 .. 09:30+L.
    """
    con = duckdb.connect(str(KLINES_DB), read_only=True)
    try:
        k = con.execute(
            "SELECT time_key, open, high, low, close, volume FROM klines "
            "WHERE code=? AND ktype='1m' ORDER BY time_key", [code]).fetchdf()
    finally:
        con.close()
    k["t"] = pd.to_datetime(k.time_key)
    k = k.drop_duplicates("t").set_index("t").sort_index()
    hh = k.index.hour + k.index.minute / 60
    return k[(hh > 9.5) & (hh <= 16.0)]


def hours(idx: pd.DatetimeIndex) -> np.ndarray:
    return idx.hour + idx.minute / 60


def sessions(rth: pd.DataFrame):
    """Whole regular sessions only. Half days are excluded, never patched."""
    for day, g in rth.groupby(rth.index.normalize()):
        if len(g) != RTH_BARS:
            continue
        gh = hours(g.index)
        am = float(g.volume[gh <= 13.0].sum())
        pm = float(g.volume[gh > 13.0].sum())
        if am <= 0 or pm / am < HALF_DAY_VOL_RATIO:
            continue
        yield day, g


# ── one session ───────────────────────────────────────────────────────────

def fill_after(day_bars: pd.DataFrame, trig_ts: pd.Timestamp):
    """Where a signal stamped `trig_ts` can actually be filled.

    A bar's close is not a tradeable price: by the time it is known, it is
    gone. The fill is the OPEN of the next 1-minute bar, and that bar's own
    high/low are live for the exit from the moment we are in.
    """
    nxt = day_bars[day_bars.index > trig_ts]
    if nxt.empty:
        return None, None
    return nxt.index[0], float(nxt.open.iloc[0])


def simulate(day_bars: pd.DataFrame, entry_ts, entry, or_lo, flat_hh) -> dict:
    """Walk 1-minute bars from the fill; the first edge touched wins.

    Ties go to the stop, as backtest/engine.py's _find_exit() does. The
    bars are sliced to THIS session and to the flat time, so there is no
    bar-count horizon that could reach into the next day.
    """
    risk = entry - or_lo
    target = entry + risk
    live = day_bars[(day_bars.index >= entry_ts) &
                    (hours(day_bars.index) <= flat_hh)]
    if live.empty:
        return {}

    sl = np.flatnonzero((live.low <= or_lo).to_numpy())
    tp = np.flatnonzero((live.high >= target).to_numpy())
    first_sl = sl[0] if sl.size else len(live)
    first_tp = tp[0] if tp.size else len(live)

    out = dict(mae_r=float((entry - live.low.min()) / risk),
               mfe_r=float((live.high.max() - entry) / risk),
               both=bool(sl.size and tp.size))
    if first_sl == len(live) and first_tp == len(live):
        px = float(live.close.iloc[-1])
        out.update(result="timeout", exit_px=px, r=(px - entry) / risk,
                   t_out=live.index[-1])
    elif first_sl <= first_tp:
        out.update(result="loss", exit_px=or_lo, r=-1.0,
                   t_out=live.index[first_sl])
    else:
        out.update(result="win", exit_px=target, r=1.0,
                   t_out=live.index[first_tp])
    return out


def day_legs(day_bars: pd.DataFrame, or_min: int, entry_tf: str,
             flat_hh: float, risk_mode: str = "range_low") -> dict:
    """Both legs for one session: the rule, and the every-day control.

    They share the opening range, the 1R target and the flat time. The only
    difference is that the control does not wait for a breakout -- it
    enters on the first entry-timeframe bar after the range, every day.
    That makes the difference between them attributable to the breakout
    filter and nothing else.

    `risk_mode` picks the risk unit, and it is the whole reason the first
    attempt at this comparison had to be thrown away:

    "range_low"  the rule as specified: stop AT the range low. Correct for
                 the rule, WRONG for the control, whose fixed-time entry
                 can land a hair above that low. Measured on SOXL: the
                 control's risk unit reaches 0.010% of price, which puts
                 cost_R at 25 and asks for 4863% of the account. The
                 difference then measures how unlike the two denominators
                 are, not how good the filter is -- it came out exactly
                 proportional to the cost constant.
    "or_width"   both legs stop one opening-range WIDTH below their own
                 entry, with a symmetric target, and R is denominated in
                 that width: a per-day constant, independent of entry
                 price, bounded below. The cost term then cancels exactly
                 in the same-day difference, so the statistic is immune to
                 the cost constant -- precisely the defect that killed the
                 first one.
    """
    gh = hours(day_bars.index)
    orw = day_bars[gh <= 9.5 + or_min / 60]
    if len(orw) != or_min:
        return {}
    rest = day_bars[gh > 9.5 + or_min / 60]
    if rest.empty:
        return {}

    or_hi, or_lo = float(orw.high.max()), float(orw.low.min())
    or_w = or_hi - or_lo
    if or_w <= 0:
        return {}
    bars = rest.resample(entry_tf, label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    if bars.empty:
        return {}
    bh = hours(bars.index)
    out = dict(or_hi=or_hi, or_lo=or_lo, or_w=or_w)

    def leg(trig_ts):
        ets, px = fill_after(day_bars, trig_ts)
        if ets is None:
            return None
        stop = or_lo if risk_mode == "range_low" else px - or_w
        if px <= stop:
            return None
        sim = simulate(day_bars, ets, px, stop, flat_hh)
        return dict(entry=px, entry_ts=ets, stop=stop, **sim) if sim else None

    # control: the first entry-timeframe bar after the range, unconditionally
    if bh[0] <= flat_hh:
        c = leg(bars.index[0])
        if c:
            out["ctrl"] = c

    # rule: the first such bar whose CLOSE is above the range high
    trig = bars[(bars.close > or_hi) & (bh <= flat_hh)]
    if not trig.empty:
        r = leg(trig.index[0])
        if r:
            out["rule"] = r
    return out


# ── a configuration end to end ────────────────────────────────────────────

def run(rth: pd.DataFrame, or_min: int, entry_tf: str, flat_hh: float,
        equity0: float, risk_frac: float, cost_cents: float,
        compound: bool, max_leverage: float = 1.0,
        risk_mode: str = "range_low") -> pd.DataFrame:
    """One cell. Returns one row per session, with both legs on it.

    Sizing is fixed-fractional, which is what makes a return series
    interpretable here: shares = risk_dollars / stop_distance, so
    per-trade return is exactly risk_frac x R and the price level -- $7.72
    to $265 inside this sample -- cancels out of the arithmetic.

    Position VALUE is asserted against `max_leverage`, never silently
    capped: a silent cap is undeclared resizing, and it hides exactly the
    thing worth knowing. On SOXL the stop is wide enough that 0.5% of
    equity is a 19% position and 1x never binds. On QQQ the same 0.5%
    wants 103% of equity, because an unleveraged index ETF's opening range
    is a fifth as wide -- so the cap is a real parameter there, not a
    safety net. How often it binds is reported.
    """
    rows, equity = [], equity0
    for day, g in sessions(rth):
        legs = day_legs(g, or_min, entry_tf, flat_hh, risk_mode)
        if not legs:
            continue
        rec = dict(day=day.date(), or_hi=legs["or_hi"], or_lo=legs["or_lo"])

        for name in ("rule", "ctrl"):
            leg = legs.get(name)
            if leg is None:
                continue
            risk = leg["entry"] - leg["stop"]
            # Cost in R does not depend on size: (shares x c) / (shares x
            # risk) = c / risk. So both legs get a net R without being
            # sized, and only the leg that actually trades the account
            # needs shares at all.
            r_net = leg["r"] - (cost_cents / 100.0) / risk
            base = equity if compound else equity0
            shares = int((risk_frac * base) / risk)
            value = shares * leg["entry"]
            rec.update({
                f"{name}_entry": round(leg["entry"], 4),
                f"{name}_in": leg["entry_ts"].strftime("%H:%M"),
                f"{name}_out": leg["t_out"].strftime("%H:%M"),
                f"{name}_risk_pct": round(100 * risk / leg["entry"], 3),
                f"{name}_shares": shares,
                f"{name}_pos_pct": round(100 * value / equity, 1),
                f"{name}_result": leg["result"],
                f"{name}_r": round(leg["r"], 4),
                f"{name}_r_net": round(r_net, 4),
                f"{name}_mae_r": round(leg["mae_r"], 3),
                f"{name}_mfe_r": round(leg["mfe_r"], 3),
                f"{name}_both": leg["both"],
            })
            if name == "rule":
                # Inference and presentation are separated here. A and B'
                # are R-denominated and therefore invariant to size, so
                # they are computed with no cap at all. The cap exists only
                # for the equity curve, where buying power is real; the
                # trades it binds on are flagged rather than silently
                # resized, and the leverage distribution is reported,
                # because it is the one thing a return curve hides: at
                # 0.5% risk this rule wants a median 1.28x on QQQ and 0.19x
                # on SOXL, which the two curves would otherwise present as
                # comparable.
                rec["rule_lev"] = round(value / equity, 3)
                cap_shares = int((max_leverage * equity) / leg["entry"])
                rec["rule_capped"] = shares > cap_shares
                used = min(shares, cap_shares)
                if used <= 0:
                    continue
                rec["rule_shares_used"] = used
                rec["rule_pnl"] = round(
                    used * (leg["exit_px"] - leg["entry"])
                    - used * (cost_cents / 100.0), 2)
        # only the rule's P&L compounds the account being reported
        if "rule_pnl" in rec:
            equity += rec["rule_pnl"]
        rec["equity"] = round(equity, 2)
        rows.append(rec)
    return pd.DataFrame(rows)


# ── inference ─────────────────────────────────────────────────────────────

def boot_diff(d: pd.DataFrame, reps: int = 10000, seed: int = 7,
              block: int = 1) -> dict:
    """Resample SESSIONS, recomputing both legs on each resample.

    One trade per day, so resampling trades is resampling days. Both legs
    are recomputed on the SAME resampled days, which is what makes the
    difference a paired quantity rather than two independent means
    subtracted. A block length > 1 keeps day-to-day dependence that an iid
    resample destroys; both are reported because they can disagree.
    """
    rng = np.random.default_rng(seed)
    rule = d.rule_r_net.to_numpy(dtype=float)
    ctrl = d.ctrl_r_net.to_numpy(dtype=float)
    n = len(d)

    def means(idx):
        a, b = rule[idx], ctrl[idx]
        a, b = a[~np.isnan(a)], b[~np.isnan(b)]
        return (a.mean() if a.size else np.nan,
                b.mean() if b.size else np.nan)

    if block <= 1:
        idxs = rng.integers(0, n, size=(reps, n))
    else:
        nb = int(np.ceil(n / block))
        st = rng.integers(0, max(1, n - block + 1), size=(reps, nb))
        idxs = (st[:, :, None] + np.arange(block)[None, None, :]
                ).reshape(reps, -1)[:, :n] % n

    out = np.array([means(i) for i in idxs])
    diff = out[:, 0] - out[:, 1]
    diff = diff[~np.isnan(diff)]
    q = lambda a, p: float(np.percentile(a, p))
    return dict(
        rule_lo=q(out[~np.isnan(out[:, 0]), 0], 2.5),
        rule_hi=q(out[~np.isnan(out[:, 0]), 0], 97.5),
        diff_lo=q(diff, 2.5), diff_hi=q(diff, 97.5),
    )


def buy_hold(rth: pd.DataFrame) -> float:
    """Return, in percent, of holding from the first session's open to the
    last session's close -- over the SAME sessions the rule can trade."""
    ss = list(sessions(rth))
    if not ss:
        return float("nan")
    first = float(ss[0][1].open.iloc[0])
    last = float(ss[-1][1].close.iloc[-1])
    return 100 * (last / first - 1)


def summarise(d: pd.DataFrame, equity0: float, bh: float = float("nan")) -> dict:
    """bh: buy-and-hold return over the same sessions, in percent.

    A strategy return without it is unreadable on an instrument that rose
    5.3x over the sample: a long-only rule in the market three days in four
    inherits most of that, and the question is never "did it make money"
    but "did it beat holding".
    """
    r = d.rule_r_net.dropna().to_numpy()
    c = d.ctrl_r_net.dropna().to_numpy()
    ci = boot_diff(d)
    cib = boot_diff(d, block=5)
    eq = np.concatenate([[equity0], d.equity.to_numpy()])
    peak = np.maximum.accumulate(eq)
    lev = d.rule_lev.dropna()
    return dict(
        sessions=len(d), trades=len(r), ctrl_n=len(c),
        rule_mean=r.mean(), ctrl_mean=c.mean(),
        diff=r.mean() - c.mean(),
        diff_lo=ci["diff_lo"], diff_hi=ci["diff_hi"],
        diff_lo_b5=cib["diff_lo"], diff_hi_b5=cib["diff_hi"],
        rule_lo=ci["rule_lo"], rule_hi=ci["rule_hi"],
        ret_pct=100 * (eq[-1] / equity0 - 1),
        maxdd_pct=100 * float(((eq - peak) / peak).min()),
        bh_pct=bh,
        lev_med=float(lev.median()) if len(lev) else float("nan"),
        lev_p90=float(lev.quantile(.9)) if len(lev) else float("nan"),
        lev_max=float(lev.max()) if len(lev) else float("nan"),
        lev_over1=100 * float((lev > 1).mean()) if len(lev) else float("nan"),
        capped=int(d.rule_capped.sum()) if "rule_capped" in d else 0,
        # share of TRADES that timed out, not of sessions: the rule is flat
        # on a quarter of sessions and those are not timeouts.
        timeout_pct=100 * (d.rule_result.dropna() == "timeout").mean(),
    )


def report(d: pd.DataFrame, equity0: float, label: str,
           bh: float = float("nan")) -> None:
    s = summarise(d, equity0, bh)
    print(f"\n=== {label} ===")
    print(f"sessions {s['sessions']}   rule trades {s['trades']}"
          f"   control trades {s['ctrl_n']}")
    print(f"A  rule mean R (net)    {s['rule_mean']:+.4f}"
          f"   95% [{s['rule_lo']:+.4f}, {s['rule_hi']:+.4f}]")
    print(f"   control mean R (net) {s['ctrl_mean']:+.4f}")
    print(f"B  rule - control       {s['diff']:+.4f}"
          f"   95% [{s['diff_lo']:+.4f}, {s['diff_hi']:+.4f}]"
          f"   block-5 [{s['diff_lo_b5']:+.4f}, {s['diff_hi_b5']:+.4f}]"
          f"   <- decisive")
    print(f"   compounded return {s['ret_pct']:+.1f}%   "
          f"max drawdown {s['maxdd_pct']:.1f}%  (single realised path)")
    print(f"   buy & hold over the same sessions {s['bh_pct']:+.1f}%")
    print(f"   leverage wanted at this risk: median {s['lev_med']:.2f}x   "
          f"p90 {s['lev_p90']:.2f}x   max {s['lev_max']:.2f}x   "
          f">1x on {s['lev_over1']:.1f}% of trades   "
          f"capped {s['capped']}")
    sub = d.dropna(subset=["rule_r_net"])
    print("\n   outcome buckets (different mechanisms -- the pooled number "
          "describes neither):")
    print(sub.groupby("rule_result").agg(
        n=("rule_r_net", "size"), mean_r=("rule_r_net", "mean"),
        pnl=("rule_pnl", "sum")).round(4).to_string())
    print(f"\n   skew separation (MAE-MFE, R): rule "
          f"{(sub.rule_mae_r - sub.rule_mfe_r).mean():+.3f}   control "
          f"{(d.ctrl_mae_r - d.ctrl_mfe_r).mean():+.3f}")
    for name, half in (("2025", d[pd.to_datetime(d.day).dt.year == 2025]),
                       ("2026", d[pd.to_datetime(d.day).dt.year == 2026])):
        if half.empty:
            continue
        hr = half.rule_r_net.dropna()
        hc = half.ctrl_r_net.dropna()
        print(f"   {name}: n={len(hr)}  rule {hr.mean():+.4f}  "
              f"ctrl {hc.mean():+.4f}  diff {hr.mean()-hc.mean():+.4f}"
              f"   (power ~43-45%, a negative half is not a refutation)")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--code", default=CODE)
    ap.add_argument("--or-min", type=int, default=PRIMARY["or_min"])
    ap.add_argument("--entry-tf", default=PRIMARY["entry_tf"])
    ap.add_argument("--flat", type=float, default=PRIMARY["flat"])
    ap.add_argument("--equity", type=float, default=100_000.0)
    ap.add_argument("--risk", type=float, default=0.005)
    ap.add_argument("--cost-cents", type=float, default=PRIMARY["cost_cents"])
    ap.add_argument("--bprime", action="store_true",
                    help="the repaired filter statistic: both legs on "
                         "opening-range-width geometry, difference "
                         "resampled over the full set of sessions")
    ap.add_argument("--by-year", action="store_true",
                    help="descriptive per-year table (noisy: ~185 trades a "
                         "year on QQQ, half-width ~0.135 R -- never a basis "
                         "for a conclusion)")
    ap.add_argument("--no-compound", action="store_true")
    ap.add_argument("--max-leverage", type=float, default=1.0,
                    help="cap on position value as a multiple of equity; "
                         "4.0 is a PDT margin account's intraday buying "
                         "power, and this rule is flat by the close")
    ap.add_argument("--sweep", action="store_true",
                    help="every pre-declared cell: windows x flats x costs")
    ap.add_argument("--csv")
    args = ap.parse_args()

    rth = load_1m(args.code)
    bh = buy_hold(rth)
    print(f"{args.code}: {len(rth):,} RTH 1m bars, "
          f"{sum(1 for _ in sessions(rth))} tradeable sessions, "
          f"buy & hold {bh:+.1f}%")

    if args.bprime:
        d = run(rth, args.or_min, args.entry_tf, args.flat, args.equity,
                args.risk, args.cost_cents, not args.no_compound,
                args.max_leverage, risk_mode="or_width")
        r = d.rule_r_net.dropna().to_numpy()
        c = d.ctrl_r_net.dropna().to_numpy()
        ci, cib = boot_diff(d), boot_diff(d, block=5)
        print()
        print(f"=== B' (repaired): both legs on opening-range-width "
              f"geometry, OR {args.or_min}min flat {args.flat:g} ===")
        print(f"sessions {len(d)}   rule {len(r)}   control {len(c)}")
        print(f"  rule    {r.mean():+.4f}  95% [{ci['rule_lo']:+.4f}, "
              f"{ci['rule_hi']:+.4f}]")
        print(f"  control {c.mean():+.4f}")
        print(f"  B'      {r.mean() - c.mean():+.4f}  95% "
              f"[{ci['diff_lo']:+.4f}, {ci['diff_hi']:+.4f}]   block-5 "
              f"[{cib['diff_lo']:+.4f}, {cib['diff_hi']:+.4f}]")
        print("  cost cancels in the same-day difference, so this is immune")
        print("  to the cost constant: re-run with another --cost-cents and")
        print("  B' must not move.")
        return 0

    if not args.sweep:
        d = run(rth, args.or_min, args.entry_tf, args.flat, args.equity,
                args.risk, args.cost_cents, not args.no_compound,
                args.max_leverage)
        report(d, args.equity,
               f"PRIMARY  OR {args.or_min}min  flat {args.flat:g}  "
               f"cost {args.cost_cents:g}c", bh)
        if args.csv:
            d.to_csv(args.csv, index=False)
            print(f"\nwrote {args.csv}")
        return 0

    grid = []
    for L in WINDOWS:
        for flat in FLATS:
            d = run(rth, L, args.entry_tf, flat, args.equity, args.risk,
                    args.cost_cents, not args.no_compound,
                    args.max_leverage)
            s = summarise(d, args.equity, bh)
            s.update(or_min=L, flat=flat, cost=args.cost_cents)
            grid.append(s)
    for cost in COSTS:
        if cost == args.cost_cents:
            continue        # already in the grid above, at the primary cell
        d = run(rth, PRIMARY["or_min"], args.entry_tf, PRIMARY["flat"],
                args.equity, args.risk, cost, not args.no_compound,
                args.max_leverage)
        s = summarise(d, args.equity, bh)
        s.update(or_min=PRIMARY["or_min"], flat=PRIMARY["flat"], cost=cost)
        grid.append(s)

    g = pd.DataFrame(grid)[["or_min", "flat", "cost", "trades", "rule_mean",
                            "rule_lo", "rule_hi", "timeout_pct", "ret_pct",
                            "bh_pct", "maxdd_pct", "lev_med", "capped"]]
    # +0.09 R is the pre-registered floor for an operable edge (PF 1.2).
    # A cell whose upper bound sits below it has that edge RULED OUT in
    # this sample; a cell whose bound is above it is merely unproven, and
    # the two must not be written the same way.
    g["excl_PF1.2"] = g.rule_hi < 0.09
    g["excl_PF1.33"] = g.rule_hi < 0.143
    print(f"\n=== every pre-declared cell ({len(g)} of them; all reported, "
          f"none selected) ===")
    print(g.round(4).to_string(index=False))
    n_sess = sum(1 for _ in sessions(rth))
    print()
    print("These cells are NOT independent confirmations: the windows are "
          "nested, the")
    print(f"same {n_sess} sessions are behind every row, and the cost "
          "levels re-price one")
    print("trade set. Treat the table as one observation with internal "
          "consistency.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
