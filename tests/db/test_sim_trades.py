"""Unit tests for db/sim_trades.py's SimTradesDB."""

import pathlib

import pytest

from db.sim_trades import SimTradesDB


def _trade(trade_id="t1", result="win", r_multiple=1.0, pnl_usd=100.0):
    return {
        "trade_id": trade_id, "symbol": "US.SOXL", "direction": "bull",
        "entry_time": "2026-01-01 09:30:00", "exit_time": "2026-01-01 10:00:00",
        "entry_price": 100.0, "exit_price": 105.0,
        "sl_price": 95.0, "tp_price": 110.0, "exit_cause": "tp",
        "result": result, "r_multiple": r_multiple, "shares": 100, "pnl_usd": pnl_usd,
        "chandelier_period": None, "chandelier_multiplier": None,
    }


@pytest.fixture()
def db(tmp_path: pathlib.Path):
    d = SimTradesDB(tmp_path / "sim_trades_test.duckdb")
    yield d
    d.close()


class TestSimTradesDB:
    def test_insert_and_fetch(self, db):
        db.insert_trade(_trade())
        rows = db.fetch_recent(10)
        assert len(rows) == 1
        assert rows[0]["trade_id"] == "t1"
        assert rows[0]["result"] == "win"

    def test_insert_duplicate_trade_id_ignored(self, db):
        db.insert_trade(_trade("dup"))
        db.insert_trade(_trade("dup", result="loss", r_multiple=-1.0))
        rows = db.fetch_recent(10)
        assert len(rows) == 1
        assert rows[0]["result"] == "win"   # first insert wins, duplicate ignored

    def test_session_stats_empty(self, db):
        stats = db.session_stats()
        assert stats == {"n_trades": 0, "win_rate": 0.0, "total_r": 0.0, "total_pnl_usd": 0.0}

    def test_session_stats_aggregate(self, db):
        db.insert_trade(_trade("a", result="win",  r_multiple=2.0, pnl_usd=200.0))
        db.insert_trade(_trade("b", result="loss", r_multiple=-1.0, pnl_usd=-100.0))
        stats = db.session_stats()
        assert stats["n_trades"] == 2
        assert stats["win_rate"] == pytest.approx(0.5)
        assert stats["total_r"] == pytest.approx(1.0)
        assert stats["total_pnl_usd"] == pytest.approx(100.0)

    def test_optional_sl_tp_none_for_chandelier(self, db):
        t = _trade("c")
        t["sl_price"] = None
        t["tp_price"] = None
        t["chandelier_period"] = 20
        t["chandelier_multiplier"] = 2.0
        db.insert_trade(t)
        row = db.fetch_recent(1)[0]
        assert row["sl_price"] is None
        assert row["chandelier_period"] == 20
