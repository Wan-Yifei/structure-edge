"""Narrow opening range -> breakout; wide opening range -> reversion.

    uv run backtest/initial_balance.py --stage prereg
    uv run backtest/initial_balance.py --stage all

WHAT THE DESIGN REVIEW FOUND BEFORE ANY RETURN WAS COMPUTED
-----------------------------------------------------------
1. "A wide opening range means the day's move is already spent" is
   false. Wide days have MORE absolute range left after 10:00 than
   narrow days (QQQ 135 vs 106 bps). What is true is proportional:
   narrow days leave 2.29x their own range, wide days 1.64x -- and that
   is also what a noisy 30-minute estimate of volatility would produce
   on its own, so it is not evidence of a mechanism.
2. It is barely a trigger. Narrow days break out 72.8% of the time,
   wide days 65.7%, and on 99.4% of days price takes out one side or
   the other. This is closer to "buy blind, late" than to a state.
3. The mechanism was tested and came back zero. If compression stores
   energy behind resting stops, realised volatility should ACCELERATE
   after a breakout and more so on narrow days. Across 800 breakouts it
   DECELERATES 17-19% on both, and the narrow-minus-wide difference is
   t = +0.84 against a probe that resolves an 8% difference.

Point 3 is why condition 7 is written as it is. This project has been
here before: the absorption marker was reported as "very accurate" twice
from screenshots and died in a chronological holdout. A one-in-twenty
hit with no mechanism is not worth trading.

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

from strategy.intraday_bias import panel as P           # noqa: E402
from strategy.initial_balance import rule as R          # noqa: E402

BPS = 1e4
BLOCK_MIN = 30
IB_BLOCK = 19                 # 09:30-10:00, stamps 09:31..10:00

# ── pre-registered cells (PREREG section 3) ──────────────────────────────
PRIMARY = "US.QQQ"
DESCRIBE = ("US.SOXL", "US.SOXS")
LOOKBACK = 40                 # the longest window keeping every calendar
                              # year's narrow share inside 50% +/- 10%
TICK = {"US.QQQ": 0.01, "US.SOXL": 0.01, "US.SOXS": 0.01}
COST_TICKS = (0, 1, 2, 4)     # charged at the day's own price
JUDGED = (("narrow", "up"), ("wide", "dn"))     # the two the user named
DIAGNOSTIC = (("narrow", "dn"), ("wide", "up"))  # reported, never judged
SEED = 20261117
REPS_NULL = 20000
REPS_BOOT = 10000
DAY_BLOCK = 5
DAILY_BPS = 2.5
JACK_KEEP = 0.50

T_CRIT = {1: 1.637, 2: 1.980, 4: 2.264}   # measured, not assumed


def load(code: str):
    bars = P.load_1m(code)
    days = P.tradeable_days(bars)
    b = P.prepare(bars, days, BLOCK_MIN)
    pan = P.build_panel(b, block_min=BLOCK_MIN)
    return bars, days, b, pan


def tick_bps(d: pd.DataFrame, ticks: int, tick: float) -> float:
    """A tick in bps at the day's OWN price, averaged over the days used.

    QQQ's tick runs from 0.56 bps in 2019 to 0.14 bps in 2026 -- a 4x
    trend. A full-sample median price would put a made-up time trend
    into the cost series.
    """
    return float((ticks * tick / d.o10).mean()) * BPS


def stage_prereg(code, bars, days, b, pan, d, cls) -> None:
    print(f"\n{'='*74}\nPRE-REGISTRATION DATA  {code}\n{'='*74}")
    rec = P.reconcile(b, pan, days, BLOCK_MIN)
    print(f"1m bars {len(bars):,}   tradeable days {len(days)}   "
          f"panel cells {rec['cells']:,} holding {rec['bars']:,} bars")
    print(f"audit: longest block spans {rec['max_span']:.0f} min; blocks "
          f"{rec['blocks']:+.4f} + gaps {rec['gaps']:+.4f} = "
          f"{rec['bh']:+.4f} log\n  = {100*(np.exp(rec['bh'])-1):+.1f}%, "
          f"read straight off the bars")
    print(f"  sessions with a complete opening range and a tradeable "
          f"afternoon: {len(d)}")
    print(f"  classified (needs {LOOKBACK} prior sessions): "
          f"{int(cls.notna().sum())}")

    print(f"\nthe opening range itself -- geometry before returns:")
    for name, m in (("narrow", cls == True), ("wide", cls == False)):
        g = d[m.fillna(False)]
        print(f"  {name:6s} n={len(g)}  range median "
              f"{g.ib_range.median()*BPS:6.1f} bps   "
              f"breakout fires {100*g.r_up.notna().mean():.1f}%   "
              f"reversion fires {100*g.r_dn.notna().mean():.1f}%")
        print(f"          dist_up mean {g.dist_up.mean()*BPS:5.1f} bps   "
              f"dist_dn mean {g.dist_dn.mean()*BPS:5.1f} bps   "
              f"<- the head start each leg is born with")

    print(f"\n  narrow share by calendar year (the regime check that "
          f"killed the absolute ruler):")
    yr = cls.dropna().groupby(pd.DatetimeIndex(cls.dropna().index).year).mean()
    print("   " + "  ".join(f"{y}:{v:.2f}" for y, v in yr.items()))

    print(f"\n  cost, charged at each day's own price:")
    for t in COST_TICKS:
        print(f"    {t} tick -> {tick_bps(d, t, TICK[code]):.3f} bps")

    # power, from second moments only -- no conditional mean is touched
    print(f"\n  power (second moments only, no conditional mean used):")
    for name, m, which in (("narrow x breakout", cls == True, "up"),
                           ("wide x reversion", cls == False, "dn")):
        mm = m.fillna(False).to_numpy()
        g = d[mm]
        r = (g.r_up if which == "up" else g.r_dn).to_numpy(dtype=float)
        a = np.where(~np.isnan(r), np.nan_to_num(r), 0.0) - g.ctrl.to_numpy()
        se = float(a.std(ddof=1) / np.sqrt(len(a))) * BPS
        thr = DAILY_BPS / float(mm.mean())
        mde80 = (T_CRIT[2] + 0.8416) * se
        print(f"    {name:18s} n={len(g)}  se(A) {se:.2f} bps  "
              f"MDE80 {mde80:.2f}  economic threshold {thr:.2f}")
        print(f"      -> decision level max(threshold, MDE80) = "
              f"{max(thr, mde80):.2f} bps per class day")


def stage_main(code, d, cls, args) -> dict:
    print(f"\n{'='*74}\nPRIMARY TEST  {code}  opening range "
          f"09:30-10:00, narrow = below the median of the last "
          f"{LOOKBACK}\n{'='*74}")
    masks = {"narrow": (cls == True).fillna(False).to_numpy(),
             "wide": (cls == False).fillna(False).to_numpy()}

    # The difference A is FLATTERED by cost, because the control trades
    # every class day and the rule does not. So A is judged at ZERO cost
    # and the rule's own standalone return at the WORST tier -- each
    # statistic at its own conservative end.
    c_zero = 0.0
    c_worst = tick_bps(d, max(COST_TICKS), TICK[code])
    print(f"cost tiers {[round(tick_bps(d, t, TICK[code]), 3) for t in COST_TICKS]}"
          f" bps; A judged at 0.000 (cost FLATTERS A: the control trades "
          f"every\n  class day and the rule only {'~70'}% of them), the "
          f"rule's own return judged at {c_worst:.3f}")

    cells, out = {}, {}
    for cl, which in JUDGED + DIAGNOSTIC:
        cells[(cl, which)] = R.leg(d, masks[cl], which, c_zero)
        out[(cl, which)] = R.leg(d, masks[cl], which, c_worst)

    tcrit = R.signflip_crit([cells[k]["a"] for k in JUDGED],
                            reps=REPS_NULL, seed=SEED, block=DAY_BLOCK)
    print(f"\nmax-of-{len(JUDGED)} critical t, measured: {tcrit:.3f} "
          f"(pre-registered {T_CRIT[2]:.3f}; k=1 would be "
          f"{T_CRIT[1]:.3f}, k=4 {T_CRIT[4]:.3f})")
    use = max(tcrit, T_CRIT[2])

    print(f"\n  rule   = what the leg pays per CLASS DAY (flat days book 0)")
    print(f"  ctrl   = buy 10:00, sell 16:00, same days, same ruler")
    print(f"  A      = rule - ctrl, the only comparable number")
    rows = []
    for key in JUDGED + DIAGNOSTIC:
        c = cells[key]
        w = out[key]
        rows.append(dict(cell=f"{key[0]}x{key[1]}", n=c["n"],
                         fires=f"{100*c['rate']:.1f}%",
                         rule_bps=c["rule"] * BPS, ctrl_bps=c["ctrl"] * BPS,
                         A_bps=c["A"] * BPS, t_A=c["t_A"],
                         per_trade=w["per_trade"] * BPS,
                         dist=c["dist"] * BPS, lever=c["lever"],
                         judged=key in JUDGED))
    print(pd.DataFrame(rows).set_index("cell").round(3).to_string())
    print(f"\n  'dist' is the mechanical head start: a breakout leg is born "
          f"behind by it,\n  a reversion leg ahead by it. Comparing the two "
          f"legs' raw returns compares those\n  two constants, which is why "
          f"nothing below uses raw.")

    verdicts = {}
    for key in JUDGED:
        cl, which = key
        c, w = cells[key], out[key]
        thr_econ = DAILY_BPS / float(masks[cl].mean())
        se = c["se_A"] * BPS
        mde80 = (use + 0.8416) * se
        level = max(thr_econ, mde80)
        lo, hi = R.boot(c["a"], reps=REPS_BOOT, seed=SEED, block=DAY_BLOCK)
        rlo, _ = R.boot(w["rule_v"], reps=REPS_BOOT, seed=SEED,
                        block=DAY_BLOCK)
        jk = R.jackknife(d, masks[cl], which, c_zero)
        worst = float(jk.kept.min())
        wins = R.leg(d, masks[cl], which, c_zero)
        wv = pd.Series(c["a"])
        wz = wv.clip(wv.quantile(0.005), wv.quantile(0.995))
        same = (np.sign(wz.mean()) == np.sign(wv.mean())
                and abs(wz.mean() - wv.mean()) < 0.5 * abs(wv.mean()))

        print(f"\n{'='*74}\nVERDICT  {cl} x {which}\n{'='*74}")
        print(f"  n {c['n']} class days, fires {100*c['rate']:.1f}%   "
              f"A {c['A']*BPS:+.2f} bps/class day  90% "
              f"[{lo*BPS:+.2f}, {hi*BPS:+.2f}]")
        print(f"  economic threshold {thr_econ:.2f}, MDE80 {mde80:.2f} "
              f"-> decision level {level:.2f} bps")
        print(f"  jackknife by year: worst drop keeps {worst:.2f}")
        print(jk.round(3).to_string())
        cond = [
            (f"1 t(A) > {use:.3f}", c["t_A"] > use, f"t {c['t_A']:+.3f}"),
            ("2 A lower bound > 0", lo > 0, f"{lo*BPS:+.2f} bps"),
            (f"3 A >= decision level {level:.2f}", c["A"] * BPS >= level,
             f"{c['A']*BPS:+.2f} bps"),
            ("4 rule's own return, worst cost, lower bound > 0", rlo > 0,
             f"{rlo*BPS:+.2f} bps"),
            (f"5 year jackknife keeps >= {JACK_KEEP:.0%}",
             worst >= JACK_KEEP, f"keeps {worst:.2f}"),
            ("6 winsorised agrees with raw", bool(same),
             f"{wz.mean()*BPS:+.2f} vs {wv.mean()*BPS:+.2f} bps"),
            ("7 an untested causal chain, stated in advance", False,
             "none written down; the stop-cascade one measured zero"),
        ]
        for name, ok, det in cond:
            print(f"  [{'PASS' if ok else 'FAIL'}]  {name:48s} {det}")
        npass = sum(ok for _, ok, _ in cond)
        if npass == 7:
            v = "PASS"
        elif hi * BPS < level:
            v = "RULED OUT at the operable size"
        else:
            v = "NEITHER A PASS NOR A RULING-OUT"
        print(f"\n  {npass}/7 -> {v}")
        print(f"  one-sided 95% upper bound on A {hi*BPS:+.2f} bps vs "
              f"level {level:.2f} bps")
        verdicts[key] = (npass, v)
    return dict(cells=cells, verdicts=verdicts, tcrit=tcrit)


def stage_fill(d, cls) -> None:
    """How much of the reversion leg survives a stricter fill assumption.

    The leg buys with a resting bid at the opening range's low, and the
    run above assumes it fills whenever price trades one tick through
    it. That is the generous end. The opening low is a level everybody
    can see, so the queue in front of that bid is long, and the fills
    that DO happen are the ones where the level keeps giving way --
    adverse selection, not luck. Requiring a deeper penetration is a
    crude stand-in for queue position: if the edge needs a one-tick
    fill, it is a queue artefact rather than a return.

    Diagnostic, declared as such. It is not a judged cell and nothing
    here can turn a failed condition into a passed one.
    """
    print(f"\n{'='*74}\nFILL ASSUMPTION  (diagnostic, not a gate)\n{'='*74}")
    m = (cls == False).fillna(False).to_numpy()
    g = d[m]
    base = None
    print(f"  penetration   fills   A bps   per trade")
    for k in (1, 2, 3, 5, 10):
        tick = TICK[PRIMARY]
        fill = g.min_low.to_numpy() <= g.ib_lo.to_numpy() - k * tick
        r = np.where(fill, np.log(g.close / g.ib_lo), 0.0)
        a = r - g.ctrl.to_numpy()
        if base is None:
            base = a.mean()
        print(f"  {k:>6} tick  {100*fill.mean():5.1f}%  "
              f"{a.mean()*BPS:+7.2f}   "
              f"{np.log(g.close/g.ib_lo).to_numpy()[fill].mean()*BPS:+7.2f}"
              f"   {a.mean()/base:5.2f} of the 1-tick figure")


def stage_interaction(d, cls, res) -> None:
    """Is it the width doing the work, or does the trigger pay the same
    everywhere? Named in advance; it cannot rescue a failed primary."""
    print(f"\n{'='*74}\nINTERACTION  (named in advance, not a gate)\n{'='*74}")
    for which in ("up", "dn"):
        a = res["cells"][("narrow", which)]["A"]
        bq = res["cells"][("wide", which)]["A"]
        print(f"  {which}: A(narrow) {a*BPS:+.2f} - A(wide) {bq*BPS:+.2f} "
              f"= {(a-bq)*BPS:+.2f} bps")
    print("  a 2x2 whose reverse pairings look as good as the judged ones "
          "is noise,\n  whatever the judged cells' own t says.")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--code", default=PRIMARY)
    ap.add_argument("--stage", default="all",
                    choices=["prereg", "main", "all"])
    args = ap.parse_args()

    bars, days, b, pan = load(args.code)
    d = R.day_table(b, IB_BLOCK, TICK[args.code])
    cls = R.classify(d, LOOKBACK)
    d = d[cls.notna()]
    cls = cls[cls.notna()].astype(bool)

    if args.stage in ("prereg", "all"):
        stage_prereg(args.code, bars, days, b, pan, d, cls)
    if args.stage == "prereg":
        return 0
    res = stage_main(args.code, d, cls, args)
    stage_fill(d, cls)
    stage_interaction(d, cls, res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
