"""The per-bucket array cache must not change what the profile shows.

_compute_profile_bins used to read every price level of every bucket
into python lists on every render. It now splits a bucket into numpy
arrays once and keeps them, keyed by bucket key and validated by
holding the bucket dict and comparing identity.

tests/analysis/test_profile_bins.py already pins the function against a
slow reference implementation; what is left to pin here is the cache
itself: that passing one changes nothing, that a replaced bucket is
re-read, and -- the part worth stating out loud -- that a bucket mutated
in place is NOT re-read.

That last one is a contract, not a bug: the tick loader replaces whole
buckets (load_live_ticks does `buckets.update(fresh)`) and never edits
one in place, which is exactly why identity is a sound check. A test
that pretended otherwise would be pinning a guarantee the cache does not
make.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from core.time_utils import candle_start                      # noqa: E402
from analysis.trade_viewer_qt import (_bucket_arrays,         # noqa: E402
                                      _compute_profile_bins)


def make(n_bars=40, seed=5):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2026-10-23 09:31", periods=n_bars, freq="1min")
    base = 100 + np.cumsum(rng.normal(0, 0.05, n_bars))
    kl = pd.DataFrame({
        "time_key": t.strftime("%Y-%m-%d %H:%M:%S"),
        "low": base - 0.12, "high": base + 0.12,
        "open": base, "close": base,
        "volume": rng.integers(500, 5000, n_bars).astype(float)})
    ticks = {}
    for j in range(n_bars):
        bk = candle_start(t[j].to_pydatetime(), 1)
        ticks[bk] = {
            float(round(p, 2)): {"buy": float(rng.integers(0, 400)),
                                 "sell": float(rng.integers(0, 400)),
                                 "neutral": float(rng.integers(0, 900)),
                                 "buy_m": float(rng.integers(0, 120)),
                                 "sell_m": float(rng.integers(0, 120))}
            for p in np.linspace(base[j] - 0.12, base[j] + 0.12, 9)}
    return kl, ticks


def same(a, b):
    assert np.allclose(a[0], b[0]), "bin 中心不同"
    assert np.allclose(a[1], b[1], rtol=0, atol=1e-9), "bin 成交量不同"
    assert a[2] == b[2], "used_ticks 不同"
    for k in a[3]:
        assert abs(a[3][k] - b[3][k]) < 1e-6, f"stats[{k}] 不同"


class TestTheCacheIsInvisible:
    def test_one_call_with_and_without(self):
        kl, tk = make()
        cache = {}
        same(_compute_profile_bins(kl, tk, 1, cache=cache),
             _compute_profile_bins(kl, tk, 1))

    def test_a_warm_cache_gives_the_same_answer(self):
        kl, tk = make()
        cache = {}
        _compute_profile_bins(kl, tk, 1, cache=cache)
        same(_compute_profile_bins(kl, tk, 1, cache=cache),
             _compute_profile_bins(kl, tk, 1))

    def test_a_replaced_bucket_is_re_read(self):
        """What the live loader does: a whole new dict for that bucket."""
        kl, tk = make()
        cache = {}
        _compute_profile_bins(kl, tk, 1, cache=cache)

        bk = sorted(tk)[3]
        tk = dict(tk)
        tk[bk] = {**{p: dict(c) for p, c in tk[bk].items()},
                  float(kl["low"].iloc[3]): {"buy": 50000.0, "sell": 0.0,
                                             "neutral": 0.0}}
        same(_compute_profile_bins(kl, tk, 1, cache=cache),
             _compute_profile_bins(kl, tk, 1))

    def test_a_shorter_window_reuses_the_same_buckets(self):
        """Different bars, different lo/hi, same buckets underneath."""
        kl, tk = make()
        cache = {}
        _compute_profile_bins(kl, tk, 1, cache=cache)
        part = kl.iloc[5:25].reset_index(drop=True)
        same(_compute_profile_bins(part, tk, 1, cache=cache),
             _compute_profile_bins(part, tk, 1))

    def test_no_ticks_at_all(self):
        kl, _ = make(12)
        cache = {}
        same(_compute_profile_bins(kl, None, 1, cache=cache),
             _compute_profile_bins(kl, None, 1))


class TestTheContractTheCacheRelies_on:
    def test_a_bucket_mutated_in_place_is_not_re_read(self):
        """Documented, not accidental: the loader replaces whole buckets
        and never edits one, which is what makes identity sound. If that
        ever changes, this test fails and says where to look."""
        kl, tk = make()
        cache = {}
        before = _compute_profile_bins(kl, tk, 1, cache=cache)
        bk = sorted(tk)[3]
        tk[bk][float(kl["low"].iloc[3])] = {"buy": 50000.0, "sell": 0.0,
                                            "neutral": 0.0}
        after = _compute_profile_bins(kl, tk, 1, cache=cache)
        same(after, before)
        # And without the cache the same mutation is of course visible,
        # so the equality above is the cache's doing, not the data's.
        fresh = _compute_profile_bins(kl, tk, 1)
        assert not np.allclose(fresh[1], before[1])

    def test_the_cache_holds_the_bucket_alive(self):
        """Identity would be unsafe if the entry kept only an id()."""
        kl, tk = make(8)
        cache = {}
        _compute_profile_bins(kl, tk, 1, cache=cache)
        for key, entry in cache.items():
            assert entry[0] is tk[key]


class TestBucketArrays:
    def test_zero_sum_levels_are_dropped(self):
        pd_ = {10.0: {"buy": 0, "sell": 0, "neutral": 0},
               10.1: {"buy": 3, "sell": 1, "neutral": 0}}
        price, tot, buy, sell, neu, med = _bucket_arrays(pd_)
        assert price.tolist() == [10.1]
        assert tot.tolist() == [4.0]
        assert buy.tolist() == [3.0] and sell.tolist() == [1.0]
        assert neu.tolist() == [0.0] and med.tolist() == [0.0]

    def test_medium_is_buy_m_plus_sell_m(self):
        pd_ = {10.0: {"buy": 5, "sell": 2, "neutral": 1,
                      "buy_m": 3, "sell_m": 1}}
        assert _bucket_arrays(pd_)[5].tolist() == [4.0]

    def test_an_empty_bucket_gives_empty_arrays(self):
        out = _bucket_arrays({})
        assert all(a.size == 0 for a in out)

    def test_the_cache_returns_the_very_same_arrays(self):
        pd_ = {10.0: {"buy": 1, "sell": 1, "neutral": 0}}
        cache = {}
        a = _bucket_arrays(pd_, cache, "k")
        b = _bucket_arrays(pd_, cache, "k")
        assert all(x is y for x, y in zip(a, b))
