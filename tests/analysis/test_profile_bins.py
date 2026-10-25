"""_compute_profile_bins must give the same answer after being vectorised.

It used to call np.digitize once per price level -- 90,000 scalar calls
a render -- and the rewrite moves the binning out of the loop. A speedup
that changes a single bin is not a speedup, so the test is equality
against a reference implementation written the slow, obvious way, kept
here rather than in the module so the two cannot drift into agreement.

The reference is the original code, including the details that are easy
to lose: prices outside [low, high] are skipped, a price level whose
three directions sum to zero does not count as tick coverage, and a bar
with no usable ticks falls back to spreading its OHLCV volume over the
bins its range spans.
"""
import pathlib
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from core.time_utils import candle_start  # noqa: E402
from analysis.trade_viewer_qt import _compute_profile_bins  # noqa: E402


def reference(kl, ticks, candle_mins, n_bins=60):
    """The pre-vectorisation implementation, verbatim in behaviour."""
    lo = float(kl["low"].min())
    hi = float(kl["high"].max())
    empty = {"total": 0.0, "buy": 0.0, "sell": 0.0, "neutral": 0.0,
             "medium": 0.0}
    if hi <= lo:
        return np.array([]), np.array([]), False, empty
    bins = np.linspace(lo, hi, n_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    volumes = np.zeros(n_bins, dtype=float)
    stats = dict(empty)
    used = False
    klo = kl["low"].values.astype(float)
    khi = kl["high"].values.astype(float)
    kvol = kl["volume"].fillna(0).values.astype(float)
    kt = kl["time_key"].values
    for j in range(len(kl)):
        bar_used = False
        if ticks:
            try:
                be = datetime.strptime(str(kt[j])[:16], "%Y-%m-%d %H:%M")
                bk = candle_start(be - timedelta(minutes=candle_mins),
                                  candle_mins)
            except ValueError:
                bk = None
            if bk is not None:
                pd_ = ticks.get(bk)
                if pd_:
                    for price, c in pd_.items():
                        buy = c.get("buy", 0)
                        sell = c.get("sell", 0)
                        neu = c.get("neutral", 0)
                        tot = buy + sell + neu
                        if tot > 0:
                            p = float(price)
                            if lo <= p <= hi:
                                bi = int(np.clip(np.digitize(p, bins) - 1,
                                                 0, n_bins - 1))
                                volumes[bi] += tot
                                bar_used = True
                                stats["total"] += tot
                                stats["buy"] += buy
                                stats["sell"] += sell
                                stats["neutral"] += neu
                                stats["medium"] += (c.get("buy_m", 0)
                                                    + c.get("sell_m", 0))
                    if bar_used:
                        used = True
        if not bar_used:
            mask = (centers >= klo[j]) & (centers <= khi[j])
            n_hit = int(mask.sum())
            if n_hit:
                volumes[mask] += kvol[j] / n_hit
    return centers, volumes, used, stats


def make(n_bars=90, seed=3, with_ticks=True, gap_every=0):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2026-10-23 09:31", periods=n_bars, freq="1min")
    base = 100 + np.cumsum(rng.normal(0, 0.05, n_bars))
    kl = pd.DataFrame({
        "time_key": t.strftime("%Y-%m-%d %H:%M:%S"),
        "low": base - 0.12, "high": base + 0.12,
        "open": base, "close": base,
        "volume": rng.integers(500, 5000, n_bars).astype(float)})
    ticks = {}
    if with_ticks:
        for j in range(n_bars):
            if gap_every and j % gap_every == 0:
                continue                      # a bar with no tick coverage
            bk = candle_start(t[j].to_pydatetime() - timedelta(minutes=0), 1)
            lvl = {}
            for p in np.round(np.linspace(base[j] - 0.12, base[j] + 0.12, 9),
                              2):
                lvl[float(p)] = {
                    "buy": float(rng.integers(0, 400)),
                    "sell": float(rng.integers(0, 400)),
                    "neutral": float(rng.integers(0, 900)),
                    "buy_m": float(rng.integers(0, 120)),
                    "sell_m": float(rng.integers(0, 120)),
                }
            ticks[bk] = lvl
    return kl, ticks


def same(a, b):
    ca, va, ua, sa = a
    cb, vb, ub, sb = b
    assert np.allclose(ca, cb), "bin centres differ"
    assert np.allclose(va, vb, rtol=0, atol=1e-9), "bin volumes differ"
    assert ua == ub, "used_ticks differs"
    for k in sa:
        assert abs(sa[k] - sb[k]) < 1e-6, f"stats[{k}] differs"


class TestMatchesReference:
    def test_full_tick_coverage(self):
        kl, tk = make()
        same(_compute_profile_bins(kl, tk, 1), reference(kl, tk, 1))

    def test_some_bars_have_no_ticks(self):
        """Those fall back to spreading OHLCV volume over the bins their
        range covers, and must not be counted as tick coverage."""
        kl, tk = make(gap_every=4)
        same(_compute_profile_bins(kl, tk, 1), reference(kl, tk, 1))

    def test_no_ticks_at_all(self):
        kl, _ = make(with_ticks=False)
        got = _compute_profile_bins(kl, None, 1)
        same(got, reference(kl, None, 1))
        assert got[2] is False

    def test_prices_outside_the_range_are_skipped(self):
        """A level above the window's high or below its low is dropped,
        not clamped into an end bin -- clamping would pile foreign volume
        onto a real level."""
        kl, tk = make(n_bars=20)
        before = _compute_profile_bins(kl, tk, 1)
        bk = next(iter(tk))
        hi = float(kl["high"].max())
        tk[bk][hi + 5.0] = {"buy": 9999.0, "sell": 0.0, "neutral": 0.0}
        after = _compute_profile_bins(kl, tk, 1)
        same(after, reference(kl, tk, 1))
        # Adding a level outside the window must change nothing at all --
        # comparing totals against the rogue level's own size would not
        # catch clamping, since the in-range total is larger anyway.
        same(after, before)

    def test_a_zero_volume_level_is_not_coverage(self):
        """All three directions zero: the level contributes nothing and
        must not mark the bar as tick-covered, or the bar loses its OHLCV
        fallback and its volume vanishes."""
        kl, _ = make(n_bars=6, with_ticks=False)
        t0 = datetime.strptime(str(kl["time_key"].iloc[0])[:16],
                               "%Y-%m-%d %H:%M")
        bk = candle_start(t0 - timedelta(minutes=1), 1)
        tk = {bk: {float(kl["low"].iloc[0]):
                   {"buy": 0.0, "sell": 0.0, "neutral": 0.0}}}
        got = _compute_profile_bins(kl, tk, 1)
        same(got, reference(kl, tk, 1))
        assert got[2] is False
        assert got[1].sum() > 0, "该 bar 的 OHLCV 回退被吞掉了"

    def test_degenerate_range(self):
        kl, tk = make(n_bars=5)
        kl["low"] = 100.0
        kl["high"] = 100.0
        c, v, u, _s = _compute_profile_bins(kl, tk, 1)
        assert c.size == 0 and v.size == 0 and u is False


class TestItIsActuallyFaster:
    def test_no_scalar_digitize_in_the_loop(self):
        """The regression that matters is someone putting np.digitize back
        inside the per-price loop. Counting calls catches it; a timing
        assertion would be flaky on a loaded machine."""
        import unittest.mock as m
        import analysis.trade_viewer_qt as tv
        kl, tk = make(n_bars=60)
        real = np.digitize
        calls = []
        with m.patch.object(tv.np, "digitize",
                            lambda *a, **k: (calls.append(1),
                                             real(*a, **k))[1]):
            tv._compute_profile_bins(kl, tk, 1)
        assert len(calls) <= 2, f"digitize 被调用 {len(calls)} 次，应为 1"
