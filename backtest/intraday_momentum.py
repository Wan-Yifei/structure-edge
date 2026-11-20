"""Intraday momentum on QQQ: first half hour's sign, traded in the last.

    uv run backtest/intraday_momentum.py --stage prereg   # locks the cells
    uv run backtest/intraday_momentum.py --stage main
    uv run backtest/intraday_momentum.py --stage all

WHY THIS ONE IS BUILT DIFFERENTLY FROM backtest/intraday_bias.py
----------------------------------------------------------------
The hour study chose its winner from eight candidates, so most of its
machinery existed to pay for that choice: a max-of-k critical value, a
null-winner line, a walk-forward to prove the choice survived out of
sample. None of that applies here, because nothing is chosen. The rule
comes from a 2018 paper whose sample ends in 2013; this sample starts in
2019 and shares not one day with it, and not one number in the rule is
ours. k = 1, and the critical t is 1.69 instead of 2.49.

That is a large advantage and an extremely fragile one. ONE added knob
-- a different signal window, a magnitude filter, a volume condition --
takes k to 3 or 27, and worse, destroys the out-of-sample property
entirely: the moment we choose something, the sample we chose on stops
being a clean test of the choice. The grid is therefore declared and NOT
RUN; its critical values are computed and recorded so a later session
can see what skipping it was worth.

THREE PREMISES THAT DID NOT SURVIVE THE DESIGN REVIEW
-----------------------------------------------------
1. "3x leverage means a stronger effect." SOXL's returns are 3x the
   index and so is its volatility, so t is unchanged. Comparing bps
   across instruments of different volatility is a measurement of the
   volatility. Only IR can be compared, and on IR the leverage is
   exactly neutral.
2. "The leveraged-ETF rebalancing mechanism supports this rule." The
   rebalancing trade is proportional to the WHOLE DAY's return, not the
   first half hour; the two agree in sign on 68% of days. The mechanism
   and the rule are two different rules, so both are registered and the
   mechanism's own signal is a named secondary test.
3. "SOXL and SOXS give a cross-sectional test." Their signals are
   opposite on 97.9% of days and their traded returns correlate -0.96.
   One mirror is not a second observation.

Analysis tool only: no DB writes, no trade IDs, no algo tag.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from strategy.intraday_bias import panel as P            # noqa: E402
from strategy.intraday_momentum import rule as R         # noqa: E402

BPS = 1e4
BLOCK_MIN = 30

# ── pre-registered cells (PLAN section 11) ───────────────────────────────
PRIMARY = "US.QQQ"                       # the only leg with power
DESCRIBE = ("US.SOXL", "US.SOXS")        # declared unable to decide, in advance
SIGNAL_BLOCK = 19                        # 09:30-10:00, stamps 09:31..10:00
TRADE_BLOCK = 31                         # 15:30-16:00, stamps 15:31..16:00
MECH_END_BLOCK = 30                      # 09:30->15:30, the rebalancing trigger
BARS_EXACT = 30                          # both traded blocks, no fragments
WINSOR = (0.0, 0.005)                    # both reported
SEED = 20261116
REPS_NULL = 20000
REPS_BOOT = 10000
DAY_BLOCK = 5

# Costs in bps, round trip. The last tier is Roll, kept only as a
# declared upper bound: measured against the order book on 2026-11-16 it
# overstates QQQ's spread by 8.0x and SOXL's by 1.5x, because it reads
# mean reversion as spread. Quoting it as a spread would be wrong.
COSTS = {"US.QQQ": (0.00, 0.15, 0.30, 0.61, 2.42),
         "US.SOXL": (0.00, 1.00, 2.05, 3.11),
         "US.SOXS": (0.00, 13.5, 17.0)}
COST_DECIDE = {"US.QQQ": 0.61, "US.SOXL": 3.11, "US.SOXS": 17.0}
COST_MID = {"US.QQQ": 0.30, "US.SOXL": 2.05, "US.SOXS": 17.0}

DAILY_BPS = 2.5          # his operable threshold, set for a rule traded daily
T_CRIT = 1.691           # k=1, day-block permutation; computed, not assumed
JACK_KEEP = 0.50         # condition 5

# Declared and NOT run. Recorded so the cost of skipping it is on file.
GRID_NOT_RUN = dict(signal_windows=3, hold_windows=3, magnitude_filters=3,
                    k=27, t_crit=2.584)


# ── data ─────────────────────────────────────────────────────────────────

def load(code: str):
    bars = P.load_1m(code)
    days = P.tradeable_days(bars)
    b = P.prepare(bars, days, BLOCK_MIN)
    # The WHOLE day is panelled, all 48 blocks, and the two traded ones
    # are taken from it. Building only those two would quietly disable
    # the audit's telescoping check -- the hour study shipped a
    # reconciliation that could not fail, and that is how.
    pan = P.build_panel(b, block_min=BLOCK_MIN)
    return bars, days, b, pan


def frame(pan: pd.DataFrame) -> pd.DataFrame:
    """One row per session: the signal return and the traded return."""
    def grab(blk):
        g = pan[pan.hour == blk].set_index("session")
        return g[["ret", "n_bars", "first_open", "last_close", "volume"]]

    sig, trd, mid = grab(SIGNAL_BLOCK), grab(TRADE_BLOCK), grab(MECH_END_BLOCK)
    d = pd.DataFrame({
        "sig": sig.ret, "out": trd.ret,
        "sig_bars": sig.n_bars, "out_bars": trd.n_bars,
        "open0": sig.first_open, "mid_close": mid.last_close,
        "out_vol": trd.volume,
    }).dropna(subset=["sig", "out"])
    # the mechanism's real trigger: the whole move into 15:30
    d["mech"] = np.log(d.mid_close / d.open0)
    d["year"] = pd.DatetimeIndex(d.index).year
    return d


def exact(d: pd.DataFrame) -> pd.DataFrame:
    """Both traded blocks must be complete. 30 bars, not 'at least 15'."""
    ok = (d.sig_bars == BARS_EXACT) & (d.out_bars == BARS_EXACT)
    if not ok.all():
        print(f"  dropped {int((~ok).sum())} sessions with an incomplete "
              f"signal or traded block")
    return d[ok]


def winsor(x: pd.Series, frac: float) -> np.ndarray:
    if frac <= 0:
        return x.to_numpy(dtype=float)
    return x.clip(x.quantile(frac), x.quantile(1 - frac)).to_numpy(dtype=float)


# ── stages ───────────────────────────────────────────────────────────────

def stage_prereg(code, bars, days, b, pan, d) -> None:
    print(f"\n{'='*74}\nPRE-REGISTRATION DATA  {code}\n{'='*74}")
    rec = P.reconcile(b, pan, days, BLOCK_MIN)
    print(f"1m bars {len(bars):,}   tradeable days {len(days)}   "
          f"panel cells {rec['cells']:,} holding {rec['bars']:,} bars")
    print(f"audit: blocks in time order, longest spans "
          f"{rec['max_span']:.0f} min; blocks {rec['blocks']:+.4f} + gaps "
          f"{rec['gaps']:+.4f}\n  = {rec['bh']:+.4f} log = "
          f"{100*(np.exp(rec['bh'])-1):+.1f}%, read straight off the bars "
          f"at {rec['t0']} .. {rec['t1']}")

    s = pan[pan.hour == SIGNAL_BLOCK]
    t = pan[pan.hour == TRADE_BLOCK]
    print(f"\nthe two traded blocks, before any return is looked at:")
    for name, g in (("09:30-10:00", s), ("15:30-16:00", t)):
        full = int((g.n_bars == BARS_EXACT).sum())
        print(f"  {name}  n={len(g)}  exactly {BARS_EXACT} bars on "
              f"{full} ({100*full/len(g):.1f}%)  first stamp "
              f"{g.t_first.iloc[0].time()}  last {g.t_last.iloc[0].time()}")
    print(f"  sessions with both complete: {len(d)}")

    # second moments and coverage only -- no conditional mean is touched
    sd = float(d.out.std(ddof=1))
    n_up = int((d.sig > 0).sum())
    p = float(n_up / (d.sig != 0).sum())
    thr = DAILY_BPS / p
    # L is a mean over the TRIGGER days only, so its standard error is
    # sd/sqrt(n_up), not sd/sqrt(n). Using the full count here would
    # understate the detectable effect by about 30% and make the study
    # look more powerful than it is.
    se = sd / np.sqrt(n_up)
    seD = sd * np.sqrt(1 / n_up + 1 / (len(d) - n_up))
    print(f"\ntraded window dispersion: sd {sd*BPS:.1f} bps over "
          f"{len(d)} days")
    print(f"  k=1 critical t {T_CRIT} (day-block permutation, seed {SEED})")
    print(f"  se(L) {se*BPS:.3f} bps on {n_up} trigger days; "
          f"se(D) {seD*BPS:.3f} bps")
    print(f"  detectable effect on L: {T_CRIT*se*BPS:+.2f} bps at 50% power, "
          f"{(T_CRIT+0.8416)*se*BPS:+.2f} at 80%")
    print(f"  detectable effect on D: {T_CRIT*seD*BPS:+.2f} / "
          f"{(T_CRIT+0.8416)*seD*BPS:+.2f} bps")
    print(f"  signal is positive on {100*p:.1f}% of days "
          f"({int((d.sig==0).sum())} exactly flat, not traded)")
    print(f"  operable threshold {DAILY_BPS} bps/DAY becomes "
          f"{thr:.2f} bps/TRADE at that trigger rate")
    print(f"  threshold must exceed the 80% MDE: {thr:.2f} vs "
          f"{(T_CRIT+0.8416)*se*BPS:.2f} -> "
          f"{'ok' if thr > (T_CRIT+0.8416)*se*BPS else 'NOT OK'}")
    print(f"\n  variance of the traded window by year (why condition 5 is "
          f"a jackknife):")
    v = d.groupby("year").out.apply(lambda x: float((x ** 2).sum()))
    print("   " + "  ".join(f"{y}:{100*s/v.sum():.1f}%"
                            for y, s in v.items()))
    print(f"\n  grid NOT run: {GRID_NOT_RUN['k']} cells would need t > "
          f"{GRID_NOT_RUN['t_crit']}, and choosing anything at all would "
          f"end\n  the out-of-sample property that makes k=1 legitimate.")


def stage_main(code, d, args) -> dict:
    p_thr = float((d.sig > 0).sum() / (d.sig != 0).sum())
    thr = DAILY_BPS / p_thr
    cd = COST_DECIDE[code]
    print(f"\n{'='*74}\nPRIMARY TEST  {code}  "
          f"09:30-10:00 sign -> buy 15:30-16:00  n={len(d)}\n{'='*74}")

    out = {}
    for w in WINSOR:
        sig = winsor(d.sig, w)
        y = winsor(d.out, w)
        out[w] = R.legs(sig, y)
    r = out[0.005]
    raw = out[0.0]

    print(f"trigger: {r['n_up']} long days, {r['n_dn']} not taken, "
          f"{r['n_flat']} exactly flat (not traded); p = {r['p']:.3f}")
    print(f"\n  L = U + (1-p)D   -- never quote L on its own")
    print(f"  L  long-leg mean          {r['L']*BPS:+7.2f} bps  "
          f"(se {r['seL']*BPS:.2f})")
    print(f"  U  unconditional hold     {r['U']*BPS:+7.2f} bps  "
          f"(se {r['seU']*BPS:.2f})   <- the drift in every number above")
    print(f"  D  long minus short leg   {r['D']*BPS:+7.2f} bps  "
          f"(se {r['seD']*BPS:.2f})   <- the signal itself")
    print(f"  (1-p)D  what the signal adds to L  {r['edge']*BPS:+7.2f} bps  "
          f"(se {r['se_edge']*BPS:.2f})")
    print(f"  M  the leg he cannot trade {r['M']*BPS:+7.2f} bps  "
          f"[short only -- NOT EXECUTABLE, excluded from every total]")

    bs = R.boot(winsor(d.sig, 0.005), winsor(d.out, 0.005),
                reps=REPS_BOOT, seed=SEED, block=DAY_BLOCK)
    nl = R.perm_null(winsor(d.sig, 0.005), winsor(d.out, 0.005),
                     reps=REPS_NULL, seed=SEED, block=DAY_BLOCK)
    tL = r["L"] / r["seL"]
    # Two critical values for L, and they test different things. The
    # permutation one is lower than 1.645 for a real reason: under "the
    # signal is uninformative" the long leg is a subsample WITHOUT
    # replacement of a fixed set of days, so its mean varies less than an
    # independent draw would (finite-population correction), and it is
    # centred on U rather than on zero. That makes it a test of "better
    # than an arbitrary split", which is condition 4's question. The
    # pre-registered 1.691 tests "L > 0" outright and is the stricter of
    # the two, so the verdict uses it and both are printed.
    tcL = max(T_CRIT, nl["t_crit_L"])
    print(f"\n  t(L) {tL:+.3f}   pre-registered critical {T_CRIT:.3f} "
          f"(used); permutation gives {nl['t_crit_L']:.3f}")
    print(f"    they differ because the permutation null keeps the drift "
          f"and the finite-population\n    correction, so it asks "
          f"condition 4's question, not 'is L above zero'")
    print(f"  t(D) {r['D']/r['seD']:+.3f} vs permutation "
          f"{nl['t_crit_D']:.3f} (block {DAY_BLOCK}, {nl['reps']} reps)")
    print(f"  L 95% [{bs['L_lo']*BPS:+.2f}, {bs['L_hi']*BPS:+.2f}]   "
          f"L-U 90% [{bs['gap_lo']*BPS:+.2f}, {bs['gap_hi']*BPS:+.2f}]   "
          f"D 90% [{bs['D_lo']*BPS:+.2f}, {bs['D_hi']*BPS:+.2f}]")

    print(f"\ncosts (round trip, bps) and what L is worth net:")
    for c in COSTS[code]:
        tag = ""
        if c == cd:
            tag = "  <- used for the verdict"
        if c == 2.42:
            tag = "  <- Roll, measured to overstate the spread 8x; an " \
                  "upper bound, not a spread"
        print(f"  {c:5.2f} -> {r['L']*BPS - c:+7.2f} bps/trade"
              f"   = {(r['L']*BPS - c)*p_thr:+6.2f} bps/day{tag}")

    print(f"\nrobustness: winsorised {r['L']*BPS:+.2f} bps, "
          f"raw {raw['L']*BPS:+.2f} bps  "
          f"(if these disagree the finding is a print)")
    lev = float((d.out - d.out.mean()).abs().max() / len(d) / r["seU"])
    print(f"  largest single day's push on the mean: {lev:.3f} se")

    jk = R.jackknife_years(winsor(d.sig, 0.005), winsor(d.out, 0.005),
                           d.year.to_numpy())
    print(f"\ncondition 5 is a jackknife, not a walk-forward: with the rule "
          f"fully\n  specified in advance there is nothing to select, so "
          f"'fit on the past and\n  trade forward' has no meaning here. "
          f"What can still go wrong is one year\n  carrying everything.")
    print(jk.round(3).to_string())
    worst = float(jk.kept.min())
    print(f"  worst year removed keeps {worst:.2f} of the full estimate "
          f"(needs >= {JACK_KEEP})")

    cond = [
        (f"1 t(L) > {tcL:.3f}", tL > tcL, f"t {tL:+.3f}"),
        ("2 (1-p)D lower bound > 0", bs["D_lo"] * (1 - r["p"]) > 0,
         f"{bs['D_lo']*(1-r['p'])*BPS:+.2f} bps"),
        ("3 L - worst cost lower bound > 0",
         bs["L_lo95"] * BPS - cd > 0,
         f"{bs['L_lo95']*BPS - cd:+.2f} bps"),
        ("4 L - U lower bound > 0", bs["gap_lo"] > 0,
         f"{bs['gap_lo']*BPS:+.2f} bps"),
        (f"5 year jackknife keeps >= {JACK_KEEP:.0%}", worst >= JACK_KEEP,
         f"worst keeps {worst:.2f}"),
        (f"6 net >= {thr:.2f} bps/trade", r["L"] * BPS - cd >= thr,
         f"{r['L']*BPS - cd:+.2f} bps"),
    ]
    print(f"\n{'='*74}\nVERDICT  (conditions 1-6; 7 is the mechanism probe)"
          f"\n{'='*74}")
    for name, ok, det in cond:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name:36s} {det}")
    npass = sum(ok for _, ok, _ in cond)
    hi = bs["L_hi95"] * BPS - cd
    if npass == 6:
        verdict = "PASSES 1-6, pending the mechanism probe"
    elif hi < thr:
        verdict = "RULED OUT at the operable size"
    else:
        verdict = "NEITHER A PASS NOR A RULING-OUT"
    print(f"\n  {npass}/6 -> {verdict}")
    print(f"  one-sided 95% upper bound on net L {hi:+.2f} bps vs threshold "
          f"{thr:.2f} bps")
    print(f"  for the SIGNAL to carry that threshold on its own, D would "
          f"have to be\n  >= {thr/(1-r['p']):.2f} bps, and 80% power "
          f"reaches {(T_CRIT+0.8416)*r['seD']*BPS:.2f} bps "
          f"-- so a pass here is loud or it is drift")
    return dict(d=d, r=r, bs=bs, nl=nl, thr=thr, npass=npass,
                verdict=verdict, p=p_thr)


def stage_mechanism(d) -> None:
    """Two named probes. Neither can rescue a failed primary test."""
    print(f"\n{'='*74}\nMECHANISM\n{'='*74}")
    sig = winsor(d.sig, 0.005)
    mech = winsor(d.mech, 0.005)
    y = winsor(d.out, 0.005)
    a = R.legs(sig, y)
    bm = R.legs(mech, y)
    agree = float(((sig > 0) == (mech > 0)).mean())
    print(f"the rebalancing trade is proportional to the WHOLE day's move, "
          f"not the\nfirst half hour; the two agree in sign on "
          f"{100*agree:.1f}% of days. So this is a\nsecond rule, registered "
          f"in advance, judged at max-of-2 (t > 1.923).")
    print(f"  Gao signal   r(09:30->10:00): L {a['L']*BPS:+.2f} bps  "
          f"t {a['L']/a['seL']:+.3f}   D {a['D']*BPS:+.2f}")
    print(f"  mech signal  r(09:30->15:30): L {bm['L']*BPS:+.2f} bps  "
          f"t {bm['L']/bm['seL']:+.3f}   D {bm['D']*BPS:+.2f}")
    print(f"  (p = {a['p']:.3f} vs {bm['p']:.3f})")


def stage_describe(code, thr_src) -> None:
    """Legs declared in advance to be unable to decide anything.

    Printed because leaving them out would invite someone to run them
    later and read the result as evidence. The power line is the point.
    """
    bars, days, b, pan = load(code)
    d = exact(frame(pan))
    sig, y = winsor(d.sig, 0.005), winsor(d.out, 0.005)
    r = R.legs(sig, y)
    se = r["seL"]
    cd = COST_DECIDE[code]
    thr = DAILY_BPS / r["p"]
    print(f"\n{'='*74}\nDESCRIPTIVE ONLY  {code}  (declared unable to "
          f"decide, before the run)\n{'='*74}")
    print(f"  n {r['n']}   L {r['L']*BPS:+.2f} bps   t "
          f"{r['L']/se:+.3f}   D {r['D']*BPS:+.2f} bps")
    print(f"  cost {cd:.2f} -> net {r['L']*BPS - cd:+.2f} bps/trade, "
          f"threshold {thr:.2f}")
    print(f"  POWER: 80% reaches {(T_CRIT+0.8416)*se*BPS:.2f} bps, the "
          f"threshold is {thr:.2f} bps.")
    pw = 1 - _norm_cdf(T_CRIT - thr / (se * BPS))
    print(f"  If the effect were exactly at the threshold, this leg would "
          f"detect it {100*pw:.0f}% of the time.")
    print(f"  Its traded-window sd is {d.out.std(ddof=1)*BPS:.1f} bps vs "
          f"QQQ's {thr_src:.1f} -- comparing bps across these two "
          f"instruments\n  measures the volatility, not the effect. "
          f"Only IR can be compared, and on IR\n  the leverage is exactly "
          f"neutral.")


def _norm_cdf(x: float) -> float:
    from scipy import stats as sps
    return float(sps.norm.cdf(x))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--code", default=PRIMARY)
    ap.add_argument("--stage", default="all",
                    choices=["prereg", "main", "mechanism", "describe",
                             "all"])
    args = ap.parse_args()

    bars, days, b, pan = load(args.code)
    d = exact(frame(pan))

    if args.stage in ("prereg", "all"):
        stage_prereg(args.code, bars, days, b, pan, d)
    if args.stage == "prereg":
        return 0
    res = None
    if args.stage in ("main", "all"):
        res = stage_main(args.code, d, args)
    if args.stage in ("mechanism", "all"):
        stage_mechanism(d)
    if args.stage in ("describe", "all"):
        sd = float(d.out.std(ddof=1)) * BPS
        for c in DESCRIBE:
            try:
                stage_describe(c, sd)
            except Exception as exc:              # noqa: BLE001
                print(f"\n  {c}: not available ({exc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
