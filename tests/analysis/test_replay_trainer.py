"""Unit tests for analysis/replay_trainer.py's pure settlement logic.

These test the module-level functions (check_settlement, _compute_r_multiple,
_compute_pnl_usd) directly -- no QApplication/Qt widgets needed, since that
logic was deliberately extracted out of the bound ReplayTrainerWindow methods
to keep it testable without GUI machinery.
"""

import pandas as pd
import pytest

from analysis.replay_trainer import (
    check_settlement, check_limit_fill, _compute_pnl_usd, _compute_r_multiple,
    validate_chandelier_stop,
)


def _klines(rows: list[tuple]) -> pd.DataFrame:
    """rows: list of (time_key, open, high, low, close)."""
    return pd.DataFrame(rows, columns=["time_key", "open", "high", "low", "close"])


class TestComputePnlAndR:
    def test_pnl_long_profit(self):
        assert _compute_pnl_usd("bull", 100, 100.0, 105.0) == pytest.approx(500.0)

    def test_pnl_long_loss(self):
        assert _compute_pnl_usd("bull", 100, 100.0, 97.0) == pytest.approx(-300.0)

    def test_pnl_short_profit(self):
        assert _compute_pnl_usd("bear", 50, 100.0, 90.0) == pytest.approx(500.0)

    def test_pnl_short_loss(self):
        assert _compute_pnl_usd("bear", 50, 100.0, 110.0) == pytest.approx(-500.0)

    def test_r_multiple_long(self):
        assert _compute_r_multiple("bull", 100.0, 110.0, 5.0) == pytest.approx(2.0)

    def test_r_multiple_short(self):
        assert _compute_r_multiple("bear", 100.0, 95.0, 5.0) == pytest.approx(1.0)


class TestCheckSettlementFixed:
    def test_no_touch_yet_returns_none(self):
        kl = _klines([
            ("t0", 100, 101, 99, 100),
            ("t1", 100, 105, 99, 104),
            ("t2", 104, 111, 103, 110),
        ])
        trade = {"exit_mode": "fixed", "entry_idx": 0, "entry_price": 100.0,
                  "direction": "bull", "sl": 95.0, "tp": 110.0}
        # Entry bar itself: fixed mode never checks the entry bar.
        assert check_settlement(kl, trade, replay_idx=0, max_bars_in_trade=5) is None
        # One bar revealed, no touch yet.
        assert check_settlement(kl, trade, replay_idx=1, max_bars_in_trade=5) is None

    def test_tp_hit_long(self):
        kl = _klines([
            ("t0", 100, 101, 99, 100),
            ("t1", 100, 105, 99, 104),
            ("t2", 104, 111, 103, 110),
        ])
        trade = {"exit_mode": "fixed", "entry_idx": 0, "entry_price": 100.0,
                  "direction": "bull", "sl": 95.0, "tp": 110.0}
        r = check_settlement(kl, trade, replay_idx=2, max_bars_in_trade=5)
        assert r is not None
        assert r["cause"] == "tp"
        assert r["result"] == "win"
        assert r["exit_price"] == pytest.approx(110.0)
        assert r["r_multiple"] == pytest.approx(2.0)   # (110-100)/(100-95)

    def test_sl_hit_short(self):
        kl = _klines([
            ("t0", 100, 101, 99, 100),
            ("t1", 100, 104, 96, 102),
            ("t2", 102, 106, 94, 105),
        ])
        trade = {"exit_mode": "fixed", "entry_idx": 0, "entry_price": 100.0,
                  "direction": "bear", "sl": 105.0, "tp": 90.0}
        r = check_settlement(kl, trade, replay_idx=2, max_bars_in_trade=5)
        assert r is not None
        assert r["cause"] == "sl"
        assert r["result"] == "loss"
        assert r["exit_price"] == pytest.approx(105.0)
        assert r["r_multiple"] == pytest.approx(-1.0)

    def test_genuine_timeout_only_at_cap(self):
        # Price never reaches SL or TP within the 2-bar cap.
        kl = _klines([
            ("t0", 100, 101, 99, 100),
            ("t1", 100, 102, 99, 101),
            ("t2", 101, 102, 100, 101),
        ])
        trade = {"exit_mode": "fixed", "entry_idx": 0, "entry_price": 100.0,
                  "direction": "bull", "sl": 90.0, "tp": 120.0}
        # window(1) < cap(2) -- not a real timeout yet.
        assert check_settlement(kl, trade, replay_idx=1, max_bars_in_trade=2) is None
        # window(2) == cap(2) -- genuine timeout now.
        r = check_settlement(kl, trade, replay_idx=2, max_bars_in_trade=2)
        assert r is not None
        assert r["cause"] == "timeout"


class TestCheckSettlementChandelier:
    def test_entry_bar_self_stop(self):
        """Chandelier (unlike fixed SL/TP) CAN settle on the entry bar itself."""
        kl = _klines([
            ("t0", 99, 101, 97, 100),
        ])
        trade = {
            "exit_mode": "chandelier", "entry_idx": 0, "entry_price": 100.0,
            "direction": "bull", "chandelier_period": 1, "chandelier_multiplier": 0.5,
            "risk_unit": 1.0,
        }
        r = check_settlement(kl, trade, replay_idx=0, max_bars_in_trade=5)
        assert r is not None
        assert r["cause"] == "chandelier"
        # stop = hh(101) - atr(TR=101-97=4)*0.5 = 101 - 2 = 99; low(97) <= 99 -> stopped
        assert r["exit_price"] == pytest.approx(99.0)

    def test_genuine_timeout_only_at_cap(self):
        kl = _klines([
            ("t0", 100, 101, 99, 100),
            ("t1", 100, 102, 99, 101),
            ("t2", 101, 102, 100, 101),
        ])
        trade = {
            "exit_mode": "chandelier", "entry_idx": 0, "entry_price": 100.0,
            "direction": "bull", "chandelier_period": 1, "chandelier_multiplier": 50.0,
            "risk_unit": 1.0,
        }
        # Huge multiplier -> stop far below price -> never touched within the cap.
        assert check_settlement(kl, trade, replay_idx=1, max_bars_in_trade=2) is None
        r = check_settlement(kl, trade, replay_idx=2, max_bars_in_trade=2)
        assert r is not None
        assert r["cause"] == "timeout"

    def test_none_trade_or_klines(self):
        assert check_settlement(None, {"exit_mode": "fixed"}, 0) is None
        assert check_settlement(_klines([("t0", 1, 1, 1, 1)]), None, 0) is None


class TestCheckLimitFill:
    def test_no_fill_on_placement_bar(self):
        # Even if the placement bar's own low already satisfies the limit,
        # it must not count -- matches _find_exit's from_bar+1 convention.
        kl = _klines([
            ("t0", 100, 101, 90, 100),   # low already <= limit=95, but this is placed_idx itself
        ])
        order = {"direction": "bull", "limit_price": 95.0, "placed_idx": 0}
        assert check_limit_fill(kl, order, replay_idx=0) is None

    def test_buy_limit_fills_when_price_dips_to_it(self):
        kl = _klines([
            ("t0", 100, 101, 99, 100),
            ("t1", 100, 102, 98, 101),   # low=98, still above limit=95
            ("t2", 101, 103, 94, 96),    # low=94, crosses limit=95
        ])
        order = {"direction": "bull", "limit_price": 95.0, "placed_idx": 0}
        assert check_limit_fill(kl, order, replay_idx=1) is None
        r = check_limit_fill(kl, order, replay_idx=2)
        assert r == (2, 95.0)

    def test_sell_limit_fills_when_price_rallies_to_it(self):
        kl = _klines([
            ("t0", 100, 101, 99, 100),
            ("t1", 100, 104, 99, 103),   # high=104, still below limit=105
            ("t2", 103, 106, 102, 105),  # high=106, crosses limit=105
        ])
        order = {"direction": "bear", "limit_price": 105.0, "placed_idx": 0}
        assert check_limit_fill(kl, order, replay_idx=1) is None
        r = check_limit_fill(kl, order, replay_idx=2)
        assert r == (2, 105.0)

    def test_never_fills_returns_none(self):
        kl = _klines([
            ("t0", 100, 101, 99, 100),
            ("t1", 100, 102, 99, 101),
            ("t2", 101, 103, 100, 102),
        ])
        order = {"direction": "bull", "limit_price": 50.0, "placed_idx": 0}
        assert check_limit_fill(kl, order, replay_idx=2) is None

    def test_none_order_or_klines(self):
        assert check_limit_fill(None, {"placed_idx": 0}, 5) is None


class TestValidateChandelierStop:
    def test_bull_valid_stop_below_entry(self):
        assert validate_chandelier_stop("bull", ref_price=100.0, init_stop=95.0) is None

    def test_bull_invalid_stop_at_entry(self):
        assert validate_chandelier_stop("bull", ref_price=100.0, init_stop=100.0) is not None

    def test_bull_invalid_stop_above_entry(self):
        # Price has pulled back below HH far enough that HH - ATR*mult >= entry --
        # the exact "entry immediately hits stop" scenario this guards against.
        msg = validate_chandelier_stop("bull", ref_price=100.0, init_stop=101.0)
        assert msg is not None
        assert "101.0000" in msg and "100.0000" in msg

    def test_bear_valid_stop_above_entry(self):
        assert validate_chandelier_stop("bear", ref_price=100.0, init_stop=105.0) is None

    def test_bear_invalid_stop_at_or_below_entry(self):
        assert validate_chandelier_stop("bear", ref_price=100.0, init_stop=100.0) is not None
        assert validate_chandelier_stop("bear", ref_price=100.0, init_stop=99.0) is not None
        assert check_limit_fill(_klines([("t0", 1, 1, 1, 1)]), None, 5) is None
