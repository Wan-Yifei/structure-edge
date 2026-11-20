"""The two rules, the control they must beat, and the mechanical identity.

THE IDENTITY THAT DECIDES HOW THIS IS SCORED
--------------------------------------------
Both legs exit on the same closing print, so with `O10` the 10:00 price,
`E` the breakout fill and `Lo` the initial balance's low::

    breakout       log(C/E)  = log(C/O10) - dist_up      dist_up >= 0
    mean reversion log(C/Lo) = log(C/O10) + dist_dn      dist_dn >= 0

Measured on QQQ, dist_up is 16-34 bps and dist_dn is 20-38 bps, so the
two legs start 36 to 72 bps apart BEFORE either of them is right about
anything. A raw comparison of "breakout versus reversion" is therefore a
comparison of two constants. Everything judged here is the difference
against a control that enters at 10:00 on the same days, which is the
only way the two legs are measured with one ruler.

A breakout leg can only win by AVOIDING the days it does not trigger.
It cannot win by "breakouts carrying on", because the identity has
already charged it the distance to the breakout.

WHY THE CONSERVATIVE COST TIER IS ZERO, NOT THE HIGHEST
-------------------------------------------------------
The rule is out of the market on roughly 30% of its class days and the
control is in on all of them, so the control pays more cost than the
rule does. Raising the cost assumption therefore IMPROVES the
difference. The honest conservative direction for the difference is zero
cost, and the honest conservative direction for the rule's own
standalone return is the highest tier. Both are used, each where it
bites. The ORB round shipped a statistic proportional to the cost
constant; this is the same trap with its sign reversed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BPS = 1e4


def day_table(b: pd.DataFrame, ib_block: int, tick: float) -> pd.DataFrame:
    """One row per session: the opening range, and where each leg fills.

    `b` is the prepared bar frame (session and block already tagged).
    No information from after a fill is used to decide that fill: the
    breakout is the first one-minute CLOSE above the range, and it is
    taken at the NEXT bar's open, because a close is not a tradeable
    price by the time it is known.
    """
    rows = []
    for sess, g in b.groupby("session", sort=True):
        ib = g[g.hour == ib_block]
        post = g[(g.hour > ib_block) & (g.index.hour < 16)
                 | ((g.index.hour == 16) & (g.index.minute == 0))]
        post = post[post.hour > ib_block]
        if len(ib) != 30 or post.empty:
            continue
        hi, lo = float(ib.high.max()), float(ib.low.min())
        o10 = float(post.open.iloc[0])
        close = float(post.close.iloc[-1])

        # breakout: first close above the range, filled at the next open
        c = post.close.to_numpy(dtype=float)
        up = np.flatnonzero(c > hi)
        e_up = t_up = np.nan
        if up.size and up[0] + 1 < len(post):
            e_up = float(post.open.iloc[up[0] + 1])
            t_up = up[0] + 1
        # reversion: a resting bid at the range low fills only if price
        # actually trades through it
        low = post.low.to_numpy(dtype=float)
        dn = np.flatnonzero(low <= lo - tick)
        e_dn = float(lo) if dn.size else np.nan
        rows.append(dict(session=sess, ib_hi=hi, ib_lo=lo,
                         min_low=float(low.min()),
                         ib_range=np.log(hi / lo), o10=o10, close=close,
                         e_up=e_up, e_dn=e_dn, bar_up=t_up,
                         n_post=len(post)))
    d = pd.DataFrame(rows).set_index("session")
    d["ctrl"] = np.log(d.close / d.o10)
    d["r_up"] = np.log(d.close / d.e_up)
    d["r_dn"] = np.log(d.close / d.e_dn)
    d["dist_up"] = np.log(d.e_up / d.o10)
    d["dist_dn"] = np.log(d.o10 / d.e_dn)
    d["year"] = pd.DatetimeIndex(d.index).year
    return d


def classify(d: pd.DataFrame, lookback: int) -> pd.Series:
    """Narrow = this session's range is below the median of the previous
    `lookback` sessions. Strictly past, so the label is known at 10:00.

    This replaces "narrow in absolute terms" with "quieter than the last
    forty days", and that substitution has to be stated rather than
    assumed: on an absolute ruler QQQ's narrow share runs from 9.6% in
    2022 to 71.9% in 2024, so "narrow versus wide" would be mostly
    "2024 versus 2022" -- a regime comparison wearing a range label.
    """
    med = d.ib_range.rolling(lookback, closed="left").median()
    return (d.ib_range < med).where(med.notna())


def leg(d: pd.DataFrame, mask: np.ndarray, which: str,
        cost_bps: float) -> dict:
    """One judged cell: the rule, the control, and their difference.

    `mask` picks the class days. On a day the rule does not trigger it
    books zero and pays nothing -- it is flat, which is a real outcome
    and not a missing value.
    """
    g = d[mask]
    r = (g.r_up if which == "up" else g.r_dn).to_numpy(dtype=float)
    fired = ~np.isnan(r)
    rule = np.where(fired, r - cost_bps / BPS, 0.0)
    ctrl = g.ctrl.to_numpy(dtype=float) - cost_bps / BPS
    a = rule - ctrl
    n = len(g)
    return dict(
        n=n, fired=int(fired.sum()), rate=float(fired.mean()),
        rule=float(rule.mean()), se_rule=float(rule.std(ddof=1)/np.sqrt(n)),
        ctrl=float(ctrl.mean()),
        A=float(a.mean()), se_A=float(a.std(ddof=1) / np.sqrt(n)),
        t_A=float(a.mean() / (a.std(ddof=1) / np.sqrt(n))),
        per_trade=float(r[fired].mean() - cost_bps / BPS) if fired.any()
        else np.nan,
        dist=float((g.dist_up if which == "up"
                    else g.dist_dn).dropna().mean()),
        lever=float(np.abs(a - a.mean()).max() / n
                    / (a.std(ddof=1) / np.sqrt(n))),
        a=a, rule_v=rule,
    )


def signflip_crit(cols: list[np.ndarray], reps: int = 20000,
                  seed: int = 20261117, block: int = 5) -> float:
    """max-of-k critical t, flipping whole blocks of days together.

    The cells overlap heavily here -- the same sessions appear in
    several of them -- so the flips are shared across cells, which is
    what preserves that overlap in the null. Measured on 27 cells the
    overlap buys almost nothing (mean pairwise correlation +0.79, and
    the critical value only falls from Bonferroni's 2.78 to 2.64), so
    the saving has to come from locking the knobs, not from the null.
    """
    rng = np.random.default_rng(seed)
    n = max(len(c) for c in cols)
    out = []
    for _ in range(reps):
        nb = int(np.ceil(n / block))
        s = np.repeat(rng.choice([-1.0, 1.0], size=nb), block)[:n]
        best = -np.inf
        for c in cols:
            y = c * s[:len(c)]
            best = max(best, y.mean() / (y.std(ddof=1) / np.sqrt(len(y))))
        out.append(best)
    return float(np.percentile(out, 95))


def boot(x: np.ndarray, reps: int = 10000, seed: int = 20261117,
         block: int = 5) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    nb = int(np.ceil(n / block))
    st = rng.integers(0, max(1, n - block + 1), size=(reps, nb))
    idx = (st[:, :, None] + np.arange(block)[None, None, :]
           ).reshape(reps, -1)[:, :n] % n
    m = x[idx].mean(axis=1)
    return float(np.percentile(m, 5)), float(np.percentile(m, 95))


def jackknife(d: pd.DataFrame, mask: np.ndarray, which: str,
              cost_bps: float) -> pd.DataFrame:
    full = leg(d, mask, which, cost_bps)["A"]
    years = d[mask].year.to_numpy()
    rows = []
    for y in sorted(set(years)):
        m = mask.copy()
        m[np.flatnonzero(mask)[years == y]] = False
        r = leg(d, m, which, cost_bps)
        rows.append(dict(dropped=int(y), n=r["n"], A_bps=r["A"] * BPS,
                         kept=r["A"] / full if full else np.nan))
    return pd.DataFrame(rows).set_index("dropped")
