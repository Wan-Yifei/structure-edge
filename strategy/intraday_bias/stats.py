"""The statistics this study is allowed to use, and the nulls they need.

Three ideas carry the whole thing.

1. THE NULL WINNER. With k hours and n days, the best-looking hour in a
   market with NO hour effect at all is still positive, and on 466 days
   it is worth about +41% a year. Every reported winner is printed beside
   the number pure noise would have produced in the same shape. A winner
   that does not clear it is not a weak finding, it is the expected
   output of the procedure.

2. THE NULL IS SIGN-FLIPPED BY DAY, NOT BY CELL. Flipping whole rows
   keeps the cross-hour correlation of a day intact, so the max-of-k
   critical value is the one that applies to THIS panel. Measured, the
   average pairwise correlation between hours is about +0.01, so the
   discount over Bonferroni is nearly nothing -- unlike the ORB study's
   15 nested cells, where not correcting was the right call.

3. RESAMPLING IS BY TRADING DAY. A day is the independent unit; an hour
   is not. Block resampling is offered beside the iid version because
   they can disagree, and when they do the block answer is the one to
   quote.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BPS = 1e4


# -- descriptive ---------------------------------------------------------

def cell_stats(w: pd.DataFrame) -> pd.DataFrame:
    """Per hour: n, mean in bps, standard error, t, and the leverage of
    the single most extreme observation on the mean, measured in standard
    errors. That last column is the one that says whether an hour is a
    finding or a print.
    """
    n = w.notna().sum()
    mu = w.mean()
    sd = w.std(ddof=1)
    se = sd / np.sqrt(n)
    dev = w.sub(mu, axis=1).abs()
    lever = dev.max() / n / se          # how far one point moves the mean
    return pd.DataFrame({
        "n": n, "mean_bps": mu * BPS, "se_bps": se * BPS,
        "t": mu / se, "max_lever_se": lever,
    })


# -- max-of-k null -------------------------------------------------------

def _t_cols(x: np.ndarray) -> np.ndarray:
    n = x.shape[-2]
    mu = x.mean(axis=-2)
    sd = x.std(axis=-2, ddof=1)
    return mu / (sd / np.sqrt(n))


def signflip_null(w: pd.DataFrame, reps: int = 20000, seed: int = 20261116,
                  chunk: int = 500) -> dict:
    """Distribution of max_h t_h when every hour's true mean is zero.

    Each replicate multiplies a whole DAY by +1 or -1, so a day's shape
    across hours survives and only its direction is randomised. Under the
    null of a symmetric zero-mean day this leaves the joint distribution
    alone, which is exactly the assumption a mean test already makes.

    Returns the one-sided 95% critical t, the implied minimum detectable
    effect in bps, and the mean and 95th percentile of the best hour's
    MEAN -- the null-winner line that goes beside every reported winner.
    """
    x = w.to_numpy(dtype=float)
    n, k = x.shape
    rng = np.random.default_rng(seed)
    maxt, maxmu = [], []
    done = 0
    while done < reps:
        m = min(chunk, reps - done)
        s = rng.choice([-1.0, 1.0], size=(m, n, 1))
        y = x[None, :, :] * s
        maxt.append(_t_cols(y).max(axis=1))
        maxmu.append(y.mean(axis=1).max(axis=1))
        done += m
    maxt = np.concatenate(maxt)
    maxmu = np.concatenate(maxmu)
    se = x.std(axis=0, ddof=1) / np.sqrt(n)
    tcrit = float(np.percentile(maxt, 95))
    return dict(
        k=k, n=n, reps=reps, seed=seed,
        t_crit=tcrit,
        mde_bps=tcrit * float(np.median(se)) * BPS,
        null_winner_bps=float(maxmu.mean()) * BPS,
        null_winner_p95_bps=float(np.percentile(maxmu, 95)) * BPS,
        t_crit_p99=float(np.percentile(maxt, 99)),
    )


def mean_pairwise_corr(w: pd.DataFrame) -> float:
    c = w.corr().to_numpy()
    iu = np.triu_indices_from(c, k=1)
    return float(np.nanmean(c[iu]))


# -- bootstrap -----------------------------------------------------------

def _day_index(n: int, reps: int, rng, block: int) -> np.ndarray:
    if block <= 1:
        return rng.integers(0, n, size=(reps, n))
    nb = int(np.ceil(n / block))
    st = rng.integers(0, max(1, n - block + 1), size=(reps, nb))
    return (st[:, :, None] + np.arange(block)[None, None, :]
            ).reshape(reps, -1)[:, :n] % n


def boot_mean(x: np.ndarray, reps: int = 10000, seed: int = 20261116,
              block: int = 1, lo: float = 2.5, hi: float = 97.5) -> tuple:
    rng = np.random.default_rng(seed)
    idx = _day_index(len(x), reps, rng, block)
    m = x[idx].mean(axis=1)
    return float(np.percentile(m, lo)), float(np.percentile(m, hi))


def boot_paired(a: np.ndarray, b: np.ndarray, reps: int = 10000,
                seed: int = 20261116, block: int = 1,
                lo: float = 2.5, hi: float = 97.5) -> tuple:
    """CI for mean(a) - mean(b) with both legs recomputed on the SAME
    resampled days. Subtracting two independently resampled means instead
    would inflate the interval by the part the two legs share, which on a
    common-drift pair is most of it.
    """
    rng = np.random.default_rng(seed)
    idx = _day_index(len(a), reps, rng, block)
    d = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return float(np.percentile(d, lo)), float(np.percentile(d, hi))


def boot_random_hour(w: pd.DataFrame, reps: int = 10000,
                     seed: int = 20261116, block: int = 1) -> np.ndarray:
    """One replicate of control B'2: resample days, then on each day pick
    one of the locked hours uniformly at random and hold it for the hour.

    Tradeable, unlike alpha, and it pays the same number of round trips
    as the rule it is the control for. Its expectation is the cross-hour
    mean, so "winner minus B'2" is alpha with the extra noise that
    actually choosing at random would cost you.
    """
    x = w.to_numpy(dtype=float)
    n, k = x.shape
    rng = np.random.default_rng(seed)
    idx = _day_index(n, reps, rng, block)
    pick = rng.integers(0, k, size=idx.shape)
    return x[idx, pick].mean(axis=1)


# -- cost ----------------------------------------------------------------

def roll_spread(steps: pd.Series) -> float:
    """Roll (1984) effective spread from the first-order autocovariance of
    1-minute log returns: s = 2*sqrt(-cov). NaN when the autocovariance
    is positive, which means momentum has swamped the bid-ask bounce and
    the estimator has nothing to say -- not that the spread is zero.
    """
    s = steps.dropna().to_numpy(dtype=float)
    if len(s) < 100:
        return float("nan")
    cov = float(np.cov(s[1:], s[:-1], ddof=1)[0, 1])
    return 2 * np.sqrt(-cov) if cov < 0 else float("nan")


# -- shrinkage -----------------------------------------------------------

def shrinkage(cell: pd.DataFrame) -> dict:
    """How much say a short lookback has actually earned.

    tau^2 is the variance of the TRUE hour effects, estimated by taking
    the observed spread of the hour means and removing the part pure
    sampling noise explains. L0 = sigma^2 / tau^2, and a window of L days
    gets weight w = L/(L+L0).

    When the estimate comes out at or below zero -- which is the usual
    outcome -- there is no measurable hour effect to shrink towards and
    the formula answers by itself: every finite window gets weight zero.
    """
    mu = cell.mean_bps.to_numpy() / BPS
    se = cell.se_bps.to_numpy() / BPS
    n = cell.n.to_numpy()
    sigma2 = float(np.mean(se ** 2 * n))        # per-day variance
    tau2 = float(np.var(mu, ddof=1) - np.mean(se ** 2))
    out = dict(sigma2=sigma2, tau2=tau2, tau=np.sqrt(max(tau2, 0.0)) * BPS)
    out["L0"] = sigma2 / tau2 if tau2 > 0 else float("inf")
    for L in (5, 15, 30, 60, 120, 250):
        out[f"w{L}"] = L / (L + out["L0"]) if np.isfinite(out["L0"]) else 0.0
    return out
