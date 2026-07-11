"""Unit tests for strategy/chandelier_exit/ — ATR, trailing-stop simulator, entries."""

import numpy as np
import pandas as pd
import pytest

from strategy.chandelier_exit.atr import wilder_atr
from strategy.chandelier_exit.chandelier import rolling_extremes, simulate_chandelier_exit
from strategy.chandelier_exit.entries import collect_entries


# ── atr.wilder_atr ───────────────────────────────────────────────────────────

class TestWilderAtr:
    def test_hand_computed_reference(self):
        highs  = np.array([10.0, 12.0, 11.0, 13.0, 14.0])
        lows   = np.array([8.0,  9.0,  9.0,  10.0, 11.0])
        closes = np.array([9.0,  11.0, 10.0, 12.0, 13.0])

        atr = wilder_atr(highs, lows, closes, period=3)

        assert np.isnan(atr[0]) and np.isnan(atr[1])
        # TR = [2, 3, 2, 3, 3]; seed ATR[2] = mean(2,3,2) = 2.33333
        assert atr[2] == pytest.approx(2.333333, abs=1e-4)
        # ATR[3] = (2.33333*2 + 3) / 3 = 2.555556
        assert atr[3] == pytest.approx(2.555556, abs=1e-4)
        # ATR[4] = (2.555556*2 + 3) / 3 = 2.703704
        assert atr[4] == pytest.approx(2.703704, abs=1e-4)

    def test_too_short_series_all_nan(self):
        highs = lows = closes = np.array([1.0, 2.0])
        atr = wilder_atr(highs, lows, closes, period=5)
        assert np.all(np.isnan(atr))

    def test_empty_series(self):
        atr = wilder_atr(np.array([]), np.array([]), np.array([]), period=3)
        assert len(atr) == 0


# ── chandelier.simulate_chandelier_exit ──────────────────────────────────────

class TestChandelierSimulate:
    def test_no_self_stop_and_ratchet_catches_up_next_bar(self):
        """A bar's own new extreme can't stop it out; the stop only reflects it
        starting the NEXT bar, and the exit fill is the stop price, not the
        (possibly much lower, gapped-through) bar low."""
        highs  = np.array([100.0, 110.0, 106.0])
        lows   = np.array([99.6,   99.6, 105.0])
        closes = np.array([99.8,  105.0, 105.5])
        times  = np.array(["t0", "t1", "t2"])
        atr    = np.array([0.5, 0.5, 0.5])
        hh, ll = highs.copy(), lows.copy()   # period=1 stand-in: each bar is its own extreme

        res = simulate_chandelier_exit(
            highs, lows, closes, times, atr, hh, ll,
            entry_idx=0, entry_price=99.8, direction="bull",
            period=1, multiplier=1.0, risk_unit=1.0, max_bars=200,
        )

        assert res is not None
        # bar1's own new high (110) must NOT stop bar1 out on itself.
        assert res.exit_bar == 2
        # Fill is exactly the stop level (109.5 = hh[1] - atr[1]), not bar2's
        # low (105) which gapped through it.
        assert res.exit_price == pytest.approx(109.5)
        assert res.cause == "stopped"
        assert res.stop_series[0] == pytest.approx(99.5)
        assert res.stop_series[1] == pytest.approx(99.5)   # unchanged at bar1
        assert res.stop_series[2] == pytest.approx(109.5)  # catches up at bar2

    def test_ratchet_never_loosens_bull(self):
        highs  = np.array([100, 105, 103, 108, 104, 110], dtype=float)
        lows   = np.array([98,  102, 100, 104, 101, 106], dtype=float)
        closes = np.array([99,  104, 101, 106, 102, 109], dtype=float)
        times  = np.array([f"t{i}" for i in range(6)])
        atr    = np.full(6, 1.0)
        hh, ll = rolling_extremes(highs, lows, period=2)

        res = simulate_chandelier_exit(
            highs, lows, closes, times, atr, hh, ll,
            entry_idx=2, entry_price=101.0, direction="bull",
            period=2, multiplier=1.5, risk_unit=1.0, max_bars=200,
        )
        assert res is not None
        diffs = np.diff(res.stop_series)
        assert np.all(diffs >= -1e-9), "bull stop must never move down"

    def test_ratchet_never_loosens_bear(self):
        highs  = np.array([110, 106, 108, 103, 105, 100], dtype=float)
        lows   = np.array([106, 101, 104, 99,  102, 96],  dtype=float)
        closes = np.array([108, 103, 105, 101, 103, 98],  dtype=float)
        times  = np.array([f"t{i}" for i in range(6)])
        atr    = np.full(6, 1.0)
        hh, ll = rolling_extremes(highs, lows, period=2)

        res = simulate_chandelier_exit(
            highs, lows, closes, times, atr, hh, ll,
            entry_idx=2, entry_price=105.0, direction="bear",
            period=2, multiplier=1.5, risk_unit=1.0, max_bars=200,
        )
        assert res is not None
        diffs = np.diff(res.stop_series)
        assert np.all(diffs <= 1e-9), "bear stop must never move up"

    def test_timeout_when_never_stopped(self):
        n = 10
        closes = np.array([100.0 + 3.0 * i for i in range(n)])  # steady +3/bar trend
        highs  = closes + 0.5   # narrow 1.0-point bar range, much smaller than ATR offset
        lows   = closes - 0.5
        times  = np.array([f"t{i}" for i in range(n)])
        atr    = np.full(n, 2.0)   # offset (atr*mult=2.0) > bar range (1.0): no self-stop
        hh, ll = rolling_extremes(highs, lows, period=2)

        # entry_idx=1 (not 0): period=2 needs 2 bars of warmup, so hh[0]/ll[0]
        # are NaN by design -- entering at bar 0 would correctly return None.
        res = simulate_chandelier_exit(
            highs, lows, closes, times, atr, hh, ll,
            entry_idx=1, entry_price=closes[1], direction="bull",
            period=2, multiplier=1.0, risk_unit=1.0, max_bars=5,
        )
        assert res is not None
        assert res.cause == "timeout"
        assert res.exit_bar == 6
        assert res.exit_price == pytest.approx(closes[6])

    def test_insufficient_warmup_returns_none(self):
        highs = lows = closes = np.array([1.0, 2.0, 3.0])
        times = np.array(["t0", "t1", "t2"])
        atr, hh, ll = (np.full(3, np.nan) for _ in range(3))
        res = simulate_chandelier_exit(
            highs, lows, closes, times, atr, hh, ll,
            entry_idx=0, entry_price=1.0, direction="bull",
            period=5, multiplier=2.0, risk_unit=1.0,
        )
        assert res is None

    def test_entry_stop_override_covers_entry_and_next_bar(self):
        """entry_stop_override must replace BOTH the entry bar's candidate
        AND the very next bar's (both read hh[entry_idx]/atr[entry_idx] under
        the shift convention -- overriding only position 0 would let bar 1's
        cummax immediately revert to the un-overridden value one bar later,
        defeating the point). The ratchet must still trail via HH/LL from the
        bar after that onward, exactly as without the override -- it's a
        starting-point fix, not a different trailing algorithm."""
        highs  = np.array([100.0, 110.0, 106.0])
        lows   = np.array([99.6,   99.6, 105.0])
        closes = np.array([99.8,  105.0, 105.5])
        times  = np.array(["t0", "t1", "t2"])
        atr    = np.array([0.5, 0.5, 0.5])
        hh, ll = highs.copy(), lows.copy()   # period=1 stand-in, as in the earlier test

        # Without override: cand[0] = hh[0] - atr[0] = 100 - 0.5 = 99.5 (see the
        # "no self-stop" test above). Override it to something further below.
        res = simulate_chandelier_exit(
            highs, lows, closes, times, atr, hh, ll,
            entry_idx=0, entry_price=99.8, direction="bull",
            period=1, multiplier=1.0, risk_unit=1.0, max_bars=200,
            entry_stop_override=90.0,
        )
        assert res is not None
        assert res.stop_series[0] == pytest.approx(90.0)     # entry bar uses the override
        assert res.stop_series[1] == pytest.approx(90.0)     # still below bar1's cand (99.5) -> ratchet holds at 90
        assert res.stop_series[2] == pytest.approx(109.5)    # bar2 still ratchets up via HH exactly as before
        assert res.cause == "stopped"
        assert res.exit_price == pytest.approx(109.5)

    def test_no_override_matches_default_hh_anchored_behavior(self):
        """entry_stop_override=None (the default) must reproduce the exact
        same result as omitting it entirely -- existing callers (grid_search,
        calculator.py) are unaffected."""
        highs  = np.array([100.0, 110.0, 106.0])
        lows   = np.array([99.6,   99.6, 105.0])
        closes = np.array([99.8,  105.0, 105.5])
        times  = np.array(["t0", "t1", "t2"])
        atr    = np.array([0.5, 0.5, 0.5])
        hh, ll = highs.copy(), lows.copy()

        kwargs = dict(
            highs=highs, lows=lows, closes=closes, times=times, atr=atr, hh=hh, ll=ll,
            entry_idx=0, entry_price=99.8, direction="bull",
            period=1, multiplier=1.0, risk_unit=1.0, max_bars=200,
        )
        baseline = simulate_chandelier_exit(**kwargs)
        explicit_none = simulate_chandelier_exit(**kwargs, entry_stop_override=None)
        assert baseline.exit_bar == explicit_none.exit_bar
        assert baseline.exit_price == pytest.approx(explicit_none.exit_price)
        assert np.array_equal(baseline.stop_series, explicit_none.stop_series)


# ── entries.collect_entries ───────────────────────────────────────────────────

def _trending_klines(n: int, start: float, step: float, freq: str) -> pd.DataFrame:
    closes  = [start + i * step for i in range(n)]
    highs   = [c + abs(step) * 0.6 for c in closes]
    lows    = [c - abs(step) * 0.6 for c in closes]
    volumes = [10_000] * n
    times   = pd.date_range("2025-01-02 09:30", periods=n, freq=freq)
    return pd.DataFrame({
        "time_key": times.strftime("%Y-%m-%d %H:%M:%S"),
        "open": closes, "high": highs, "low": lows,
        "close": closes, "volume": volumes,
    })


class TestCollectEntries:
    def test_smoke_and_entry_ltf_bar_indexing(self, monkeypatch):
        htf_df = _trending_klines(60, start=100.0, step=0.5, freq="60min")
        ltf_df = _trending_klines(240, start=100.0, step=0.1, freq="15min")

        def _fake_fetch_klines(code, ktype, start, end, **kw):
            return htf_df if ktype == "60m" else ltf_df

        monkeypatch.setattr(
            "strategy.chandelier_exit.entries.fetch_klines", _fake_fetch_klines
        )

        from backtest.engine import BacktestParams
        params = BacktestParams(trend_tf="60m", entry_tf="15m")

        entries, result, returned_ltf = collect_entries(
            "US.TEST", ("60m", "15m"), "2025-01-01", "2025-01-10", params,
        )

        assert returned_ltf is ltf_df   # same object, never re-fetched
        assert result.n_trades == len(entries)
        for e in entries:
            assert 0 <= e.entry_ltf_bar < len(returned_ltf)
            assert e.direction in ("bull", "bear")
