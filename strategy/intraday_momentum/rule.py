"""The rule, its decomposition, and the null that belongs to it.

THE DECOMPOSITION IS THE POINT
------------------------------
Four quantities, and an identity that is exact rather than approximate::

    p     share of days the signal is positive
    L     mean last-half-hour return on those days      <- what he gets
    M     mean last-half-hour return on the others
    U     mean last-half-hour return, unconditionally   <- the control
    D     L - M                                         <- the signal

    L = U + (1 - p) * D          exactly, every sample

This is the counterpart of `mu_h = drift + alpha_h` in the hour study,
and it is a better one. Alpha summed to zero across hours, so "a
positive alpha exists" was worth nothing. D is not constrained to sum to
anything: D > 0 is a real claim that can be false.

Reporting rule, which the driver enforces: never quote L alone. L
without U is a drift measurement wearing a signal's name, and if L is
significant while (1-p)*D is not, the right product is "buy the last
half hour every day" -- not this rule.

THE NULL IS A PERMUTATION OF THE SIGNAL, NOT A SIGN FLIP
--------------------------------------------------------
Shuffling which days carry the signal leaves the last half hour's own
distribution completely intact -- its fat tails, its volatility
clustering, its drift -- and destroys only the pairing. That is exactly
the null being tested ("the first half hour says nothing about the
last"), and it is exact in finite samples rather than asymptotic. Blocks
of five days are used because the signal is not independent across days:
QQQ's sign series has a real negative autocorrelation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BPS = 1e4


def legs(sig: np.ndarray, out: np.ndarray) -> dict:
    """p, L, M, U, D and their standard errors, plus the identity check.

    `sig` is the signal return and `out` the traded return, both in log
    units, aligned by day. A day whose signal is exactly zero is not
    traded -- it is a real case (QQQ has six) and rounding it into the
    long leg would be a silent choice.
    """
    live = sig != 0
    up = live & (sig > 0)
    dn = live & (sig < 0)
    yl, ym, yu = out[up], out[dn], out[live]
    p = up.sum() / live.sum()

    def ms(x):
        return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))

    (L, seL), (M, seM), (U, seU) = ms(yl), ms(ym), ms(yu)
    D = L - M
    seD = float(np.sqrt(seL ** 2 + seM ** 2))
    err = abs(L - (U + (1 - p) * D))
    if err > 1e-12:
        raise AssertionError(
            f"L = U + (1-p)D is an identity, off by {err:.3e}")
    return dict(n=int(live.sum()), n_up=int(up.sum()), n_dn=int(dn.sum()),
                n_flat=int((~live).sum()), p=p,
                L=L, seL=seL, M=M, seM=seM, U=U, seU=seU, D=D, seD=seD,
                # the part of L that the signal, not the drift, supplies
                edge=(1 - p) * D, se_edge=(1 - p) * seD)


def perm_null(sig: np.ndarray, out: np.ndarray, reps: int = 20000,
              seed: int = 20261116, block: int = 5) -> dict:
    """Critical t, and what the best-looking result is worth under the null.

    Whole blocks of consecutive days are moved together, so a run of
    same-signed signals stays a run. With block=1 this is the ordinary
    permutation test; the block version is the one to quote, and it comes
    out slightly more conservative here.
    """
    rng = np.random.default_rng(seed)
    n = len(sig)
    nb = int(np.ceil(n / block))
    tL, tD = [], []
    for _ in range(reps):
        order = rng.permutation(nb)
        idx = (order[:, None] * block + np.arange(block)).ravel()
        idx = idx[idx < n][:n]
        s = sig[idx]
        live = s != 0
        up, dn = live & (s > 0), live & (s < 0)
        if up.sum() < 2 or dn.sum() < 2:
            continue
        a, b = out[up], out[dn]
        ma, mb = a.mean(), b.mean()
        sa = a.std(ddof=1) / np.sqrt(len(a))
        sb = b.std(ddof=1) / np.sqrt(len(b))
        tL.append(ma / sa)
        tD.append((ma - mb) / np.sqrt(sa ** 2 + sb ** 2))
    tL, tD = np.array(tL), np.array(tD)
    return dict(reps=len(tL), seed=seed, block=block,
                t_crit_L=float(np.percentile(tL, 95)),
                t_crit_D=float(np.percentile(tD, 95)),
                null_best_L_bps=float(np.percentile(tL, 95)
                                      * out.std(ddof=1)
                                      / np.sqrt(len(out)) * BPS))


def boot(sig: np.ndarray, out: np.ndarray, reps: int = 10000,
         seed: int = 20261116, block: int = 5) -> dict:
    """Day-block bootstrap of L, of the control gap L-U, and of D.

    All three are recomputed on the SAME resampled days. L and U overlap
    by construction -- the long-leg days are a subset of all days -- so
    resampling them separately would inflate the gap's interval by the
    part they share, which is most of it.
    """
    rng = np.random.default_rng(seed)
    n = len(sig)
    nb = int(np.ceil(n / block))
    Ls, gaps, Ds = [], [], []
    for _ in range(reps):
        st = rng.integers(0, max(1, n - block + 1), size=nb)
        idx = (st[:, None] + np.arange(block)).ravel()[:n] % n
        s, o = sig[idx], out[idx]
        live = s != 0
        up, dn = live & (s > 0), live & (s < 0)
        if up.sum() < 2 or dn.sum() < 2:
            continue
        a, b, u = o[up], o[dn], o[live]
        Ls.append(a.mean())
        gaps.append(a.mean() - u.mean())
        Ds.append(a.mean() - b.mean())
    q = lambda v, k: float(np.percentile(v, k))
    return dict(
        L_lo=q(Ls, 2.5), L_hi=q(Ls, 97.5), L_hi95=q(Ls, 95), L_lo95=q(Ls, 5),
        gap_lo=q(gaps, 5), gap_hi=q(gaps, 95),
        D_lo=q(Ds, 5), D_hi=q(Ds, 95))


def jackknife_years(sig: np.ndarray, out: np.ndarray,
                    years: np.ndarray) -> pd.DataFrame:
    """Drop one calendar year at a time and recompute the long leg.

    Chosen over a split-sample holdout because there is nothing to
    select: with the rule fully specified in advance there is no "fit on
    the past, trade the future" to run. What can still go wrong is that
    one year carries everything, and on QQQ one year carries half the
    variance of the traded window, so this is the gate that matters.
    """
    rows = []
    full = legs(sig, out)["L"]
    for y in sorted(set(years)):
        m = years != y
        r = legs(sig[m], out[m])
        rows.append(dict(dropped=int(y), n=r["n"], L_bps=r["L"] * BPS,
                         kept=r["L"] / full if full else np.nan))
    return pd.DataFrame(rows).set_index("dropped")
