"""Hour-of-day bias: does any one hour of the day pay, after costs?

WHAT THIS REPLICATES, AND WHAT IT DOES NOT
------------------------------------------
The method comes from a YouTube video (WUh7nIIRo9g) demonstrated on ES
futures: sweep the 24 one-hour windows, rank them by net PnL, trade the
winners. Its own description says the results are GROSS OF COSTS. That
single line splits the question in two, and this study answers only the
second half: not "does it show up gross" but "does it survive costs on
the instruments this account can actually trade". The full method and
the reasons for every choice are in strategy/intraday_bias/PLAN.md.

THE THREE THINGS THAT DECIDE THE ANSWER
---------------------------------------
1. alpha sums to zero across the hour set every single day, so an hour
   with positive excess ALWAYS exists -- in this market, in a random
   one, in any of them. Finding one is zero information. Only beating
   the max-of-k critical value is information, and only the decomposition
   mu_h = drift + alpha_h says which of the two you found.

2. The best of k hours on n days is positive under a true null: on
   SOXL's 466 days that is +41% a year out of pure noise. Every winner
   here is printed next to the number noise would have produced in the
   same shape.

3. Costs are a threshold, not a nuisance. An hour whose round-trip cost
   exceeds the smallest effect this sample could detect is declared
   untradeable BEFORE any return is computed, and is not counted in k.
   On QQQ this rules out the overnight hours -- which is where the
   video's headline finding lives. That exclusion is honest: the reason
   is cost and stale quotes, not performance.

HOW TO RUN IT
-------------
    uv run backtest/intraday_bias.py --stage prereg   # locks the cells
    uv run backtest/intraday_bias.py --stage main     # the primary test
    uv run backtest/intraday_bias.py --stage all

`prereg` prints coverage, costs, critical values and null winners and
deliberately prints NO mean return, so the cell list can be fixed and
written down before anyone has seen which hour looks good.

Analysis tool only: no DB writes, no trade IDs, no algo tag -- the same
footing as backtest/orb.py.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from strategy.intraday_bias import panel as P          # noqa: E402
from strategy.intraday_bias import stats as S          # noqa: E402

BPS = 1e4

# ── pre-registered constants (PLAN sections 4, 6, 7, 10) ─────────────────
# The cell count lives in the source, not in whoever reads the output.
SELECT_CODE = "US.QQQ"      # selects the hour
CONFIRM_CODE = "US.SOXL"    # confirms it, never selects (466 days is far
                            # too few to choose from 24 cells: the best
                            # one is +41%/yr under a true null)

COVER_MIN = 0.99            # an hour must exist on ~every tradeable day
STALE_ZERO_VOL = 0.05       # above this share of zero-volume minutes the
                            # Roll estimate is not believable: carried
                            # prices erase the bid-ask bounce it reads
STALE_COST_BPS = 5.0        # the declared stand-in for those hours
FLOOR_CENTS = 1.0           # a marketable limit at the inside
HIGH_MULT = 2.0             # the worst of the three cost tiers

SEED = 20261116
REPS_NULL = 20000
REPS_BOOT = 10000
BLOCK = 5                   # day-block bootstrap length, reported beside
                            # the iid one because they can disagree

MIN_EDGE_BPS = 2.5          # condition 6: an operable size, and above the
                            # MDE so that "passed" is not "not refuted"
WARMUP = 252                # condition 5: days before the first live pick
ROLLING_L = (5, 15, 30, 60, 120, 250)   # shown as null demonstrations only


# ── data ─────────────────────────────────────────────────────────────────

def load(code: str):
    bars = P.load_1m(code)
    days = P.tradeable_days(bars)
    b = P.prepare(bars, days)
    pan = P.build_panel(b)
    return bars, days, b, pan


def hour_table(b: pd.DataFrame, days: pd.DatetimeIndex,
               pan: pd.DataFrame) -> pd.DataFrame:
    """Coverage, staleness and cost for all 24 hours. No returns."""
    n_days = len(days)
    rows = []
    for h in range(24):
        g = pan[pan.hour == h]
        if g.empty:
            rows.append(dict(hour=h, n=0, cover=0.0))
            continue
        steps = b[b.hour == h].step
        roll = S.roll_spread(steps) * BPS
        px = float(g.first_open.median())
        cent = (FLOOR_CENTS / 100.0) / px * BPS
        zero = float(g.zero_vol.mean())
        stale = zero > STALE_ZERO_VOL
        if stale:
            low = mid = STALE_COST_BPS
        else:
            low = cent
            mid = max(roll, cent) if np.isfinite(roll) else cent
        rows.append(dict(
            hour=h, n=len(g), cover=len(g) / n_days,
            med_bars=float(g.n_bars.median()), px=px,
            zero_vol=zero, stale=stale, bad=int(g.bad_print.sum()),
            roll_bps=roll, cent_bps=cent,
            cost_low=low, cost_mid=mid, cost_high=HIGH_MULT * mid,
        ))
    return pd.DataFrame(rows).set_index("hour")


def lock_hours(ht: pd.DataFrame, w_all: pd.DataFrame) -> dict:
    """Apply the pre-registered eligibility rule and report its derivation.

    The rule is circular on its face -- the detectable effect depends on
    k, and k depends on which cells the cost rule removes -- so it is
    resolved in one fixed direction, declared in advance: the MDE used
    for the cost test is the one of the CANDIDATE set (every hour with
    full coverage), and k is then recomputed on the survivors for the
    actual critical value. Doing it the other way round would let the
    exclusions widen themselves.
    """
    cand = [h for h in ht.index if ht.cover.get(h, 0) >= COVER_MIN]
    w0 = w_all[cand].dropna(how="any")
    null0 = S.signflip_null(w0, reps=4000, seed=SEED)
    mde0 = null0["mde_bps"]
    keep = [h for h in cand if ht.cost_mid[h] < mde0]
    return dict(candidates=cand, mde0=mde0, null0=null0, hours=tuple(keep))


# ── reporting helpers ────────────────────────────────────────────────────

def line(ch="-", n=74):
    print(ch * n)


def cost_vec(ht: pd.DataFrame, hours, tier: str) -> np.ndarray:
    return ht.loc[list(hours), f"cost_{tier}"].to_numpy(dtype=float)


def per_hour_report(w: pd.DataFrame, a: pd.DataFrame, ht: pd.DataFrame,
                    hours) -> pd.DataFrame:
    cs = S.cell_stats(w)
    ca = S.cell_stats(a)
    ch = cost_vec(ht, hours, "high")
    rows = []
    for i, h in enumerate(hours):
        lo, hi = S.boot_mean(w[h].to_numpy(), reps=REPS_BOOT, seed=SEED,
                             block=BLOCK)
        alo, ahi = S.boot_mean(a[h].to_numpy(), reps=REPS_BOOT, seed=SEED,
                               block=BLOCK)
        rows.append(dict(
            hour=h, n=int(cs.n[h]),
            mu=cs.mean_bps[h], mu_lo=lo * BPS, mu_hi=hi * BPS, t=cs.t[h],
            alpha=ca.mean_bps[h], a_lo=alo * BPS, a_hi=ahi * BPS,
            lever=cs.max_lever_se[h],
            cost=ch[i], net=cs.mean_bps[h] - ch[i],
        ))
    return pd.DataFrame(rows).set_index("hour")


# ── stages ───────────────────────────────────────────────────────────────

def stage_prereg(code: str, bars, days, b, pan, ht, lock) -> None:
    print(f"\n{'='*74}\nPRE-REGISTRATION DATA  {code}\n{'='*74}")
    rec = P.reconcile(b, pan, days)
    print(f"1m bars {len(bars):,}   tradeable days {len(days)}   "
          f"panel cells {len(pan):,}")
    print(f"audit: {rec['cells']:,} cells hold {rec['bars']:,} bars, one "
          f"cell each; blocks are in time order;\n  longest block spans "
          f"{rec['max_span']:.0f} min; blocks {rec['blocks']:+.4f} + gaps "
          f"{rec['gaps']:+.4f} = {rec['bh']:+.4f} log, which is "
          f"log(close {rec['t1']} / open {rec['t0']})\n  read straight off "
          f"the bars ({100*(np.exp(rec['bh'])-1):+.1f}%)")
    print(f"  of that, {100*(np.exp(rec['inside'])-1):+.1f}% is inside the "
          f"sessions -- the rest is in the gaps between them, which no "
          f"hour-of-day rule can hold")

    print("\nhour coverage, staleness and cost  -- NO return is shown here")
    show = ht[["n", "cover", "med_bars", "zero_vol", "bad", "roll_bps",
               "cent_bps", "cost_mid"]].copy()
    show["cover"] = (100 * show.cover).round(1)
    show["zero_vol"] = (100 * show.zero_vol).round(1)
    print(show.round(3).to_string())
    print("  cover/zero_vol are percent; roll NaN = autocovariance was "
          "positive\n  (momentum swamped the bounce), cost falls back to "
          "the 1-cent floor.")
    print(f"  stale rule: zero-volume minutes > {100*STALE_ZERO_VOL:.0f}% "
          f"-> Roll discarded, cost set to {STALE_COST_BPS} bps.")

    c = lock["candidates"]
    print(f"\ncandidates (coverage >= {100*COVER_MIN:.0f}%): "
          f"{c[0]:02d}-{c[-1]:02d}, k0={len(c)}   "
          f"t_crit {lock['null0']['t_crit']:.2f}   "
          f"MDE {lock['mde0']:+.2f} bps")
    out = [h for h in c if h not in lock["hours"]]
    for h in out:
        print(f"  h{h:02d} excluded: cost {ht.cost_mid[h]:.2f} bps >= MDE "
              f"{lock['mde0']:.2f} bps"
              + ("  (stale quotes)" if ht.stale[h] else ""))
    print(f"  -> locked hour set {lock['hours']}  k={len(lock['hours'])}")


def stage_main(code: str, ht, lock, w_all, args) -> dict:
    hours = lock["hours"]
    w_raw = w_all[list(hours)].dropna(how="any")
    w = P.winsorise(w_raw)
    a = P.alpha(w)
    n = len(w)

    null = S.signflip_null(w, reps=REPS_NULL, seed=SEED)
    rbar = S.mean_pairwise_corr(w)

    print(f"\n{'='*74}\nPRIMARY TEST  {code}  hours {hours}  k={len(hours)}"
          f"  n={n} days\n{'='*74}")
    print(f"max-of-{len(hours)} one-sided 95% critical t = "
          f"{null['t_crit']:.3f}   (Bonferroni would be "
          f"{_bonf(len(hours)):.3f}; mean pairwise correlation between "
          f"hours {rbar:+.3f}, so there is almost no discount)")
    # "MDE" on its own means 50% power -- the size at which the point
    # estimate lands exactly on the critical value, so it clears it half
    # the time. The 80% figure is the one that says what this sample can
    # actually be relied on to see, and it is ABOVE the operable
    # threshold this study set for itself.
    se_med = float(np.median(w.std(ddof=1) / np.sqrt(n)))
    print(f"detectable effect: {null['mde_bps']:+.2f} bps/day at 50% power,"
          f" {(null['t_crit']+0.8416)*se_med*BPS:+.2f} at 80% power"
          f"  (the operable threshold is {MIN_EDGE_BPS:+.2f})")
    print(f"NULL WINNER LINE: with no hour effect at all, the best of "
          f"{len(hours)} cells on {n} days averages "
          f"{null['null_winner_bps']:+.2f} bps/day "
          f"(95th pct {null['null_winner_p95_bps']:+.2f})")

    rep = per_hour_report(w, a, ht, hours)
    mubar = float(w.to_numpy().mean())
    mlo, mhi = S.boot_mean(w.mean(axis=1).to_numpy(), reps=REPS_BOOT,
                           seed=SEED, block=BLOCK)
    print(f"\ndrift: holding ANY one of these hours at random is worth "
          f"{mubar*BPS:+.2f} bps/day "
          f"[{mlo*BPS:+.2f}, {mhi*BPS:+.2f}] -- this is in every cell below")
    print("\n  mu = what the hour pays   alpha = what is the hour's OWN "
          "(mu - drift)\n  net = mu - worst-tier round-trip cost   "
          "lever = largest single day's push on the mean, in se")
    print(rep.round(3).to_string())

    win = int(rep.t.idxmax())
    print(f"\nwinner: h{win:02d}  mu {rep.mu[win]:+.2f} bps  t "
          f"{rep.t[win]:+.3f}  alpha {rep.alpha[win]:+.2f} bps")

    # -- the three controls -----------------------------------------------
    print("\ncontrols (all paired on the same resampled days):")
    wv = w[win].to_numpy()
    cw = ht.cost_high[win]

    a_lo, a_hi = rep.a_lo[win], rep.a_hi[win]
    print(f"  B'1 alpha (hour identity, NOT tradeable)   "
          f"{rep.alpha[win]:+.2f} bps  95% [{a_lo:+.2f}, {a_hi:+.2f}]")

    ch = cost_vec(ht, hours, "high")
    w_net = w - ch / BPS
    # one set of draws for both legs: the control's level and its
    # difference from the winner have to come off the same resampled days,
    # or the difference picks up variance the two legs actually share.
    rng = np.random.default_rng(SEED)
    idx = S._day_index(n, REPS_BOOT, rng, BLOCK)
    pick = rng.integers(0, len(hours), size=idx.shape)
    wn = w_net.to_numpy()
    b2 = wn[idx, pick].mean(axis=1)
    d2 = wn[idx, win_col(hours, win)].mean(axis=1) - b2
    print(f"  B'2 random hour, same costs, same round trips  "
          f"{b2.mean()*BPS:+.2f} bps")
    print(f"      winner - B'2  {np.mean(d2)*BPS:+.2f} bps  95% "
          f"[{np.percentile(d2,2.5)*BPS:+.2f}, "
          f"{np.percentile(d2,97.5)*BPS:+.2f}]   <- the decisive one")
    # Part of that difference is not the hour at all: a random pick has to
    # pay the average cost of the set, and the winner pays its own. Both
    # halves are real money, but only one of them is an hour effect.
    dcost = float(ch.mean()) - cw
    print(f"      of which {dcost:+.2f} bps is only that the winner is a "
          f"CHEAPER hour than the average\n      of the set "
          f"({ch.mean():.2f} vs {cw:.2f} bps), leaving "
          f"{np.mean(d2)*BPS - dcost:+.2f} bps that is the hour itself")

    b3 = long_hold(args.pan, hours, w.index)
    c3 = (ht.cost_high[hours[0]] + ht.cost_high[hours[-1]]) / 2
    b3n = b3.to_numpy() - c3 / BPS
    d3lo, d3hi = S.boot_paired(wv - cw / BPS, b3n, reps=REPS_BOOT,
                               seed=SEED, block=BLOCK)
    print(f"  B'3 hold h{hours[0]:02d}-h{hours[-1]:02d} straight through, "
          f"one round trip  {b3n.mean()*BPS:+.2f} bps/day "
          f"({100*(np.exp(b3n.sum())-1):+.1f}% over the sample)")
    print(f"      winner - B'3  {(wv.mean()-cw/BPS-b3n.mean())*BPS:+.2f} "
          f"bps  95% [{d3lo*BPS:+.2f}, {d3hi*BPS:+.2f}]")

    # -- out of sample -----------------------------------------------------
    # NOT "the winner, measured on the second half". The winner was chosen
    # on all 1965 days, and the only reason h11 has the highest full-sample
    # t is that its second half is strong (+0.93 bps first, +2.36 second).
    # Scoring it there asks the data a question it already answered. The
    # honest version chooses with history alone and trades forward.
    print(f"\nout of sample (expanding window, {WARMUP}-day warmup, the "
          f"hour chosen from history only):")
    wf = {}
    for freq, name in (("Y", "refit yearly"), ("Q", "refit quarterly"),
                       ("M", "refit monthly")):
        r, picks = walk_forward(w, ch, WARMUP, freq)
        lo, hi = S.boot_mean(r, reps=REPS_BOOT, seed=SEED, block=BLOCK)
        wf[freq] = (r, lo, hi, picks)
        top = pd.Series(picks).value_counts().head(3)
        print(f"  {name:18s} n={len(r)}  net {r.mean()*BPS:+.2f} bps/day  "
              f"95% [{lo*BPS:+.2f}, {hi*BPS:+.2f}]   picked "
              + ", ".join(f"h{h:02d}x{c}" for h, c in top.items()))
    r_ref = wf["Y"][0]
    ref_days = w.index[-len(r_ref):]
    print(f"  {'knowing it is h11':18s} n={len(r_ref)}  net "
          f"{w[win].loc[ref_days].mean()*BPS - cw:+.2f} bps/day   "
          f"<- only available with hindsight")
    rnd = S.boot_random_hour(w_net.loc[ref_days], reps=2000, seed=SEED,
                             block=BLOCK)
    print(f"  {'random hour':18s} n={len(r_ref)}  net "
          f"{rnd.mean()*BPS:+.2f} bps/day")
    f_net = s_net = float("nan")

    # -- robustness --------------------------------------------------------
    print("\nrobustness of the winner:")
    print(f"  winsorised (primary) {w[win].mean()*BPS:+.2f} bps   "
          f"raw {w_raw[win].mean()*BPS:+.2f} bps")
    bad = args.pan[(args.pan.hour == win) & args.pan.bad_print]
    drop = w_raw[win].drop(index=pd.DatetimeIndex(bad.session), errors="ignore")
    print(f"  bad-print blocks in this hour: {len(bad)}; dropping them "
          f"{drop.mean()*BPS:+.2f} bps")
    for tier in ("low", "mid", "high"):
        print(f"  cost tier {tier:>4}: {ht[f'cost_{tier}'][win]:.2f} bps "
              f"-> net {rep.mu[win] - ht[f'cost_{tier}'][win]:+.2f} bps")
    print(f"  cost tier zero: 0.00 bps -> net {rep.mu[win]:+.2f} bps "
          f"(the extrapolation; a conclusion that needs this is a "
          f"conclusion about the cost constant)")

    # -- the six conditions ------------------------------------------------
    cond = [
        ("1 t > max-of-k critical", rep.t[win] > null["t_crit"],
         f"t {rep.t[win]:+.3f} vs {null['t_crit']:.3f}"),
        ("2 alpha lower bound > 0", a_lo > 0,
         f"alpha 95% lo {a_lo:+.2f} bps"),
        ("3 (mu - worst cost) lower bound > 0",
         rep.mu_lo[win] - cw > 0,
         f"{rep.mu_lo[win] - cw:+.2f} bps"),
        ("4 winner - B'2 lower bound > 0",
         np.percentile(d2, 2.5) > 0,
         f"{np.percentile(d2,2.5)*BPS:+.2f} bps net, "
         f"{np.percentile(d2,2.5)*BPS - dcost:+.2f} gross"),
        ("5 walk-forward net > 0, lower bound > 0",
         wf["Y"][0].mean() > 0 and wf["Y"][1] > 0,
         f"{wf['Y'][0].mean()*BPS:+.2f} bps, lower bound "
         f"{wf['Y'][1]*BPS:+.2f}"),
        (f"6 operable size: net >= {MIN_EDGE_BPS} bps",
         rep.mu[win] - cw >= MIN_EDGE_BPS,
         f"{rep.mu[win] - cw:+.2f} bps"),
    ]
    print(f"\n{'='*74}\nVERDICT  (all six, or it is not a pass)\n{'='*74}")
    for name, ok, detail in cond:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name:42s} {detail}")
    npass = sum(ok for _, ok, _ in cond)
    # One-sided, because every one of the six conditions above is
    # one-sided and the rule is long only. The first version used the
    # two-sided 97.5th percentile here and nowhere else, and that single
    # inconsistency was worth 0.27 bps -- exactly enough to carry the
    # upper bound over the threshold and turn a ruling-out into a shrug.
    _, hi95 = S.boot_mean(wv, reps=REPS_BOOT, seed=SEED, block=BLOCK,
                          lo=5, hi=95)
    hi_mu = hi95 * BPS - cw
    if npass == 6:
        verdict = "PASS"
    elif hi_mu < MIN_EDGE_BPS:
        verdict = "RULED OUT at the operable size"
    else:
        verdict = "NEITHER A PASS NOR A RULING-OUT"
    print(f"\n  {npass}/6 -> {verdict}")
    print(f"  one-sided 95% upper bound on the winner's net "
          f"{hi_mu:+.2f} bps vs threshold {MIN_EDGE_BPS:+.2f} bps "
          f"(two-sided would be {rep.mu_hi[win]-cw:+.2f})")
    nw, nw95 = null["null_winner_bps"], null["null_winner_p95_bps"]
    pct = 100 * (rep.mu[win] / nw) if nw else float("nan")
    print(f"\n  against the null winner: the best of {len(hours)} cells "
          f"with NO hour effect\n  averages {nw:+.2f} bps and clears "
          f"{nw95:+.2f} bps one time in twenty. This winner pays "
          f"{rep.mu[win]:+.2f} bps\n  -- {pct:.0f}% of what noise alone "
          f"produces, and "
          f"{'below' if rep.mu[win] < nw95 else 'above'} the 1-in-20 line.")
    return dict(hours=hours, winner=win, w=w, rep=rep, null=null,
                verdict=verdict, npass=npass)


def win_col(hours, win):
    return list(hours).index(win)


def walk_forward(w: pd.DataFrame, cost: np.ndarray, warmup: int,
                 freq: str) -> tuple[np.ndarray, list]:
    """Pick the hour from history alone, trade it forward, repeat.

    This is the only out-of-sample statement the study can make. Choosing
    on the whole sample and then reading off a late slice is not one: the
    slice helped choose. Here the hour traded in any period was selected
    using strictly earlier days, so the sequence could have been traded
    as it stands.
    """
    idx = w.index
    x = w.to_numpy(dtype=float)
    per = pd.PeriodIndex(idx, freq=freq)
    rets, picks = [], []
    for p in per.unique():
        m = np.asarray(per == p)
        start = int(np.argmax(m))
        if start < warmup:
            continue
        hist = x[:start]
        t = hist.mean(0) / (hist.std(0, ddof=1) / np.sqrt(len(hist)))
        j = int(np.argmax(t))
        rets.append(x[m, j] - cost[j] / BPS)
        picks.extend([int(w.columns[j])] * int(m.sum()))
    return np.concatenate(rets), picks


def long_hold(pan: pd.DataFrame, hours, index) -> pd.Series:
    """B'3: one entry at the first locked hour's open, one exit at the
    last locked hour's close. Includes whatever happens between the
    blocks, which is the point -- the alternative to timing is not timing.
    """
    lo, hi = hours[0], hours[-1]
    o = pan[pan.hour == lo].set_index("session").first_open
    c = pan[pan.hour == hi].set_index("session").last_close
    r = np.log(c / o).reindex(index)
    return r.dropna()


def _bonf(k: int, alpha: float = 0.05) -> float:
    from scipy import stats as sps
    return float(sps.norm.ppf(1 - alpha / k))


def stage_rolling(w: pd.DataFrame) -> None:
    """What a short lookback produces when there is nothing to find.

    Not a warning, a measurement: the panel's own days with their signs
    randomised, selected exactly the way a rolling-window rule would
    select, and the answer is an in-sample annualised return that looks
    like a strategy.
    """
    print(f"\n{'='*74}\nROLLING WINDOWS UNDER A TRUE NULL  "
          f"(k={w.shape[1]})\n{'='*74}")
    x = w.to_numpy(dtype=float)
    n, k = x.shape
    rng = np.random.default_rng(SEED)
    print(f"{'L':>5} {'best cell in-sample':>22} {'95th pct':>12} "
          f"{'annualised':>12}")
    for L in ROLLING_L:
        if L > n:
            continue
        best = []
        for _ in range(400):
            s = rng.choice([-1.0, 1.0], size=(n, 1))
            y = x * s
            st = rng.integers(0, n - L, size=20)
            for a0 in st:
                best.append(y[a0:a0 + L].mean(axis=0).max())
        best = np.array(best)
        ann = 100 * (np.exp(best.mean() * 252) - 1)
        print(f"{L:>5} {best.mean()*BPS:>+19.2f} bps "
              f"{np.percentile(best,95)*BPS:>+9.2f} {ann:>+11.1f}%")
    print("  Every number above comes from a market with NO hour effect.\n"
          "  A rolling-window rule does not risk finding this; it finds it.")


def stage_persist(w: pd.DataFrame) -> None:
    """Is the hour pattern stable enough for a window length to matter?

    Splitting the sample and correlating the two alpha vectors answers
    the "regimes change" worry without having to guess a lookback. High
    correlation means the full sample was right all along; low means it
    moves faster than any window could estimate it, and the answer is the
    same either way.
    """
    a = P.alpha(w)
    n = len(a)
    h1, h2 = a.iloc[:n // 2].mean(), a.iloc[n // 2:].mean()
    r = float(np.corrcoef(h1, h2)[0, 1])
    print(f"\n{'='*74}\nDOES THE HOUR PATTERN PERSIST?\n{'='*74}")
    print(pd.DataFrame({"alpha_1st_half_bps": h1 * BPS,
                        "alpha_2nd_half_bps": h2 * BPS}).round(2).to_string())
    k = a.shape[1]
    print(f"\n  cross-period correlation of the alpha vector r = {r:+.3f}"
          f"   (null sd ~ {1/np.sqrt(k-1):.3f} with k={k})")
    print("  high -> use the full sample;  low -> it moves faster than any\n"
          "  window can estimate it. Both answers say: do not rotate hours.")
    sh = S.shrinkage(S.cell_stats(w))
    if not np.isfinite(sh["L0"]):
        print(f"\n  shrinkage: tau^2 estimated at {sh['tau2']:.3e} <= 0 -- "
              f"there is no measurable\n  hour effect to shrink towards, so "
              f"every finite lookback earns weight 0.")
    else:
        print(f"\n  shrinkage: tau = {sh['tau']:.2f} bps, L0 = "
              f"{sh['L0']:.0f} days; a window of")
        print("   " + "  ".join(f"L={L}:{100*sh[f'w{L}']:.0f}%"
                                for L in (5, 15, 30, 60, 120, 250)))
        print("  earns that much of the say, the rest going to the full "
              "sample.")


def stage_blocks(w: pd.DataFrame, ht: pd.DataFrame, hours) -> None:
    """Contiguous blocks, not single hours.

    The video's own finding was a block (22:00-06:00), and if an effect
    is spread over m adjacent hours a block has sqrt(m) times the signal
    of any one of them. The 36 blocks available here overlap heavily, so
    Bonferroni-36 would be far too harsh -- the critical value comes from
    the same sign-flip null, applied to the blocks themselves.
    """
    x = w.to_numpy(dtype=float)
    n, k = x.shape
    specs = [(i, j) for i in range(k) for j in range(i, k)]
    cost = cost_vec(ht, hours, "high")
    B = np.column_stack([x[:, i:j + 1].sum(axis=1) for i, j in specs])
    # one round trip per block: half-spread in at hour i, out at hour j
    bc = np.array([(cost[i] + cost[j]) / 2 for i, j in specs])
    rng = np.random.default_rng(SEED + 1)
    mx = []
    for _ in range(4000):
        s = rng.choice([-1.0, 1.0], size=(n, 1))
        y = B * s
        mx.append((y.mean(axis=0) / (y.std(axis=0, ddof=1) / np.sqrt(n))).max())
    tcrit = float(np.percentile(mx, 95))
    t = B.mean(axis=0) / (B.std(axis=0, ddof=1) / np.sqrt(n))
    net = B.mean(axis=0) * BPS - bc
    o = np.argsort(-t)[:6]
    print(f"\n{'='*74}\nCONTIGUOUS BLOCKS  ({len(specs)} overlapping, "
          f"sign-flip critical t = {tcrit:.3f})\n{'='*74}")
    print(f"{'block':>12} {'hours':>5} {'gross bps':>11} {'cost':>7} "
          f"{'net bps':>9} {'t':>8}")
    for i in o:
        a0, b0 = specs[i]
        print(f"  h{hours[a0]:02d}-h{hours[b0]:02d}{'':>4} "
              f"{b0-a0+1:>5} {B[:, i].mean()*BPS:>+11.2f} {bc[i]:>7.2f} "
              f"{net[i]:>+9.2f} {t[i]:>+8.3f}")
    best = o[0]
    print(f"\n  best block t {t[best]:+.3f} vs critical {tcrit:.3f} -> "
          f"{'clears' if t[best] > tcrit else 'does not clear'}")


def stage_confirm(win_hour: int, ir_sel: float) -> None:
    """SOXL checks an hour chosen elsewhere. It never chooses one.

    466 days is nowhere near enough to pick from 24 -- under a true null
    SOXL's best cell is worth +41% a year. The harder question, asked
    here with `ir_sel`, is whether 466 days is enough to CONFIRM one
    either. If it is not, "does not confirm" is not evidence of anything
    and must not be quoted as if it were.
    """
    code = CONFIRM_CODE
    bars, days, b, pan = load(code)
    ht = hour_table(b, days, pan)
    w_all = pan.pivot(index="session", columns="hour", values="ret")
    print(f"\n{'='*74}\nCONFIRMATION  {code}  h{win_hour:02d} only, chosen "
          f"on {SELECT_CODE}\n{'='*74}")
    if win_hour not in w_all.columns:
        print("  that hour does not exist in this instrument's panel")
        return
    x = w_all[win_hour].dropna()
    c = ht.cost_high[win_hour]
    lo, hi = S.boot_mean(x.to_numpy(), reps=REPS_BOOT, seed=SEED, block=BLOCK)
    t = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    print(f"  n {len(x)} days   mu {x.mean()*BPS:+.2f} bps  95% "
          f"[{lo*BPS:+.2f}, {hi*BPS:+.2f}]   t {t:+.3f}")
    print(f"  worst-tier cost {c:.2f} bps -> net {x.mean()*BPS - c:+.2f} bps")
    print(f"  one-sided 95% critical t for a SINGLE pre-specified hour is "
          f"1.645 (no multiplicity: the hour was not chosen here)")
    ok = t > 1.645 and x.mean() * BPS - c > 0
    print(f"  -> {'confirms' if ok else 'does not confirm'}")

    # The power line, without which "does not confirm" will be read by a
    # later session as evidence against the hour.
    from scipy import stats as sps
    se = float(x.std(ddof=1) / np.sqrt(len(x)))
    pw = float(1 - sps.norm.cdf(1.645 - ir_sel * np.sqrt(len(x))))
    need = int(np.ceil(((1.645 + 0.8416) / ir_sel) ** 2))
    print(f"  POWER: the smallest effect this leg could see at all is "
          f"{1.645*se*BPS:+.2f} bps/day, and the\n  effect being checked "
          f"is {ir_sel*x.std(ddof=1)*BPS:+.2f} bps/day on this "
          f"instrument's own volatility. If the hour is\n  real, this "
          f"test had a {100*pw:.0f}% chance of saying so; confirming it "
          f"at 80% needs {need:,} days and\n  there are {len(x)}. "
          f"\"Does not confirm\" here means \"could not have\".")


# ── main ─────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--code", default=SELECT_CODE)
    ap.add_argument("--stage", default="all",
                    choices=["prereg", "main", "rolling", "persist",
                             "blocks", "confirm", "all"])
    args = ap.parse_args()

    bars, days, b, pan = load(args.code)
    args.pan = pan
    ht = hour_table(b, days, pan)
    w_all = pan.pivot(index="session", columns="hour", values="ret")
    lock = lock_hours(ht, w_all)

    if args.stage in ("prereg", "all"):
        stage_prereg(args.code, bars, days, b, pan, ht, lock)
    if args.stage == "prereg":
        return 0

    res = None
    if args.stage in ("main", "all"):
        res = stage_main(args.code, ht, lock, w_all, args)
    w = (res["w"] if res is not None
         else P.winsorise(w_all[list(lock["hours"])].dropna(how="any")))
    if args.stage in ("rolling", "all"):
        stage_rolling(w)
    if args.stage in ("persist", "all"):
        stage_persist(w)
    if args.stage in ("blocks", "all"):
        stage_blocks(w, ht, lock["hours"])
    if args.stage in ("confirm", "all") and res is not None:
        wcol = res["w"][res["winner"]]
        stage_confirm(res["winner"],
                      float(wcol.mean() / wcol.std(ddof=1)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
