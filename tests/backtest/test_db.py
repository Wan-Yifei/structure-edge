"""Unit tests for backtest/db.py — BacktestDB schema and CRUD operations."""

import json
import pathlib
import tempfile
import uuid

import pandas as pd
import pytest

from backtest.db import BacktestDB
from backtest.engine import BacktestParams, BacktestResult, Trade


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def db(tmp_path):
    """Fresh in-memory-equivalent DB in a temp directory."""
    instance = BacktestDB(tmp_path / "test.duckdb")
    yield instance
    instance.close()


def _make_params(**kw) -> BacktestParams:
    return BacktestParams(trend_tf="4h", entry_tf="15m", **kw)


def _make_trades(n: int = 3) -> list[Trade]:
    trades = []
    for i in range(n):
        r = 2.0 if i % 2 == 0 else -1.0
        trades.append(Trade(
            direction="bull", entry_price=100.0, sl=98.0, tp=104.0,
            planned_rr=2.0, entry_time=f"2025-03-{i+1:02d} 10:00",
            exit_time=f"2025-03-{i+1:02d} 14:00",
            exit_price=104.0 if r > 0 else 98.0,
            result="win" if r > 0 else "loss",
            r_multiple=r,
        ))
    return trades


def _make_result(n_trades: int = 3) -> BacktestResult:
    return BacktestResult(params=_make_params(), trades=_make_trades(n_trades))


# ── DDL / connection ──────────────────────────────────────────────────────────

class TestSchemaCreation:
    def test_tables_exist(self, db):
        tables = {r[0] for r in db._conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema='main'"
        ).fetchall()}
        assert {"runs", "trades", "run_stats", "live_trades"} <= tables

    def test_context_manager(self, tmp_path):
        with BacktestDB(tmp_path / "ctx.duckdb") as db2:
            count = db2._conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        assert count == 0


# ── runs ──────────────────────────────────────────────────────────────────────

class TestRuns:
    def test_get_or_create_new_run(self, db):
        run_id, needs_write = db.get_or_create_run(
            "abc123", {"k": 1}, "US.TEST", "4h", "15m", "2025-01-01", "2025-12-31"
        )
        assert isinstance(run_id, str)
        assert len(run_id) == 36          # UUID format
        assert needs_write is True

    def test_existing_done_run_skipped(self, db):
        run_id, _ = db.get_or_create_run(
            "abc123", {}, "US.TEST", "4h", "15m", "2025-01-01", "2025-12-31"
        )
        db.mark_done(run_id)
        run_id2, needs_write = db.get_or_create_run(
            "abc123", {}, "US.TEST", "4h", "15m", "2025-01-01", "2025-12-31"
        )
        assert run_id2 == run_id
        assert needs_write is False

    def test_crashed_run_reset_needs_write(self, db):
        run_id, _ = db.get_or_create_run(
            "abc123", {}, "US.TEST", "4h", "15m", "2025-01-01", "2025-12-31"
        )
        db.mark_running(run_id)
        # Simulate crash: status is 'running' (not 'done')
        run_id2, needs_write = db.get_or_create_run(
            "abc123", {}, "US.TEST", "4h", "15m", "2025-01-01", "2025-12-31"
        )
        assert run_id2 == run_id
        assert needs_write is True

    def test_different_symbols_get_different_runs(self, db):
        id1, _ = db.get_or_create_run("h", {}, "US.AAPL", "4h", "15m", "2025-01-01", "2025-12-31")
        id2, _ = db.get_or_create_run("h", {}, "US.NVDA", "4h", "15m", "2025-01-01", "2025-12-31")
        assert id1 != id2

    def test_mark_failed_sets_status(self, db):
        run_id, _ = db.get_or_create_run("h", {}, "US.TEST", "4h", "15m", "2025-01-01", "2025-12-31")
        db.mark_failed(run_id, "something broke")
        status = db._conn.execute(
            "SELECT status FROM runs WHERE run_id=?", [run_id]
        ).fetchone()[0]
        assert status == "failed"


# ── trades ────────────────────────────────────────────────────────────────────

class TestTrades:
    def _create_run(self, db) -> str:
        run_id, _ = db.get_or_create_run(
            str(uuid.uuid4())[:8], {}, "US.TEST", "4h", "15m", "2025-01-01", "2025-12-31"
        )
        return run_id

    def test_write_and_read_trades(self, db):
        run_id = self._create_run(db)
        trades = _make_trades(5)
        db.write_trades(run_id, "US.TEST", trades)
        df = db.get_trades(run_id)
        assert len(df) == 5

    def test_empty_trades_no_op(self, db):
        run_id = self._create_run(db)
        db.write_trades(run_id, "US.TEST", [])
        df = db.get_trades(run_id)
        assert len(df) == 0

    def test_write_trades_is_idempotent(self, db):
        run_id = self._create_run(db)
        trades = _make_trades(3)
        db.write_trades(run_id, "US.TEST", trades)
        db.write_trades(run_id, "US.TEST", trades)   # second write replaces
        df = db.get_trades(run_id)
        assert len(df) == 3                          # not 6

    def test_trade_fields_persisted(self, db):
        run_id = self._create_run(db)
        t = Trade(
            direction="bear", entry_price=55.5, sl=57.0, tp=52.0,
            planned_rr=2.33, entry_time="2025-06-01 09:30",
            exit_time="2025-06-01 11:00", exit_price=52.0,
            result="win", r_multiple=2.33,
        )
        db.write_trades(run_id, "US.TEST", [t])
        df = db.get_trades(run_id)
        row = df.iloc[0]
        assert row["direction"]   == "bear"
        assert row["entry_price"] == pytest.approx(55.5)
        assert row["result"]      == "win"
        assert row["r_multiple"]  == pytest.approx(2.33)

    def test_fetch_trade_by_id(self, db):
        run_id = self._create_run(db)
        db.write_trades(run_id, "US.TEST", _make_trades(2))
        all_ids = db.get_trades(run_id)["trade_id"].tolist()
        record = db.fetch_trade(all_ids[0])
        assert record is not None
        assert "config_json" in record

    def test_fetch_trade_not_found(self, db):
        assert db.fetch_trade("nonexistent-id") is None


# ── run_stats ─────────────────────────────────────────────────────────────────

class TestRunStats:
    def test_write_and_query_stats(self, db):
        run_id, _ = db.get_or_create_run(
            "s1", {}, "US.TEST", "4h", "15m", "2025-01-01", "2025-12-31"
        )
        bt = _make_result(4)
        db.write_stats(run_id, bt)
        db.mark_done(run_id)

        df = db.get_run_stats(top_n=10)
        assert len(df) == 1
        assert df.iloc[0]["n_trades"] == 4

    def test_stats_columns_present(self, db):
        run_id, _ = db.get_or_create_run(
            "s2", {}, "US.TEST", "4h", "15m", "2025-01-01", "2025-12-31"
        )
        db.write_stats(run_id, _make_result())
        db.mark_done(run_id)
        df = db.get_run_stats()
        for col in ("n_trades", "win_rate", "profit_factor", "total_r", "max_drawdown_r"):
            assert col in df.columns

    def test_top_n_ordering(self, db):
        for i, pf_expected in enumerate([3.0, 1.5, 5.0]):
            run_id, _ = db.get_or_create_run(
                f"hash{i}", {}, "US.TEST", "4h", "15m", "2025-01-01", "2025-12-31"
            )
            # Craft results with different profit factors via trade lists
            wins  = int(pf_expected)
            trades = [Trade("bull", 100, 98, 104, 2, f"2025-0{i+1}-01 09:30",
                            f"2025-0{i+1}-01 10:00", 104, "win", 2.0)
                      for _ in range(wins)] + \
                     [Trade("bull", 100, 98, 104, 2, f"2025-0{i+1}-02 09:30",
                            f"2025-0{i+1}-02 10:00", 98, "loss", -1.0)]
            bt = BacktestResult(params=_make_params(), trades=trades)
            db.write_stats(run_id, bt)
            db.mark_done(run_id)

        df = db.get_run_stats(top_n=3)
        assert df.iloc[0]["profit_factor"] >= df.iloc[1]["profit_factor"]


# ── live_trades ───────────────────────────────────────────────────────────────

class TestLiveTrades:
    def _base(self, account_type="PAPER") -> dict:
        return {
            "account_type": account_type,
            "symbol": "US.SNDK",
            "direction": "LONG",
            "entry_time": "2026-05-21 10:00:00",
            "entry_price": 42.0,
            "qty": 100,
            "sl_price": 40.5,
            "tp_price": 45.0,
            "result": "open",
        }

    def test_insert_paper_trade(self, db):
        tid = db.insert_live_trade(self._base("PAPER"))
        assert len(tid) == 36

    def test_insert_live_trade(self, db):
        tid = db.insert_live_trade(self._base("LIVE"))
        assert db.fetch_live_trade(tid) is not None

    def test_invalid_account_type_rejected(self, db):
        with pytest.raises(ValueError, match="account_type"):
            db.insert_live_trade({**self._base(), "account_type": "BACKTEST"})

    def test_fetch_live_trade(self, db):
        tid = db.insert_live_trade(self._base())
        row = db.fetch_live_trade(tid)
        assert row["symbol"]       == "US.SNDK"
        assert row["account_type"] == "PAPER"
        assert row["result"]       == "open"

    def test_fetch_nonexistent_returns_none(self, db):
        assert db.fetch_live_trade("does-not-exist") is None

    def test_update_live_trade_exit(self, db):
        tid = db.insert_live_trade(self._base())
        db.update_live_trade(tid, {
            "exit_price": 45.0, "exit_time": "2026-05-21 14:00:00",
            "result": "win", "pnl_gross": 300.0,
            "commission": 2.0, "pnl_net": 298.0, "r_multiple": 2.0,
        })
        row = db.fetch_live_trade(tid)
        assert row["result"]    == "win"
        assert row["pnl_net"]   == pytest.approx(298.0)
        assert row["r_multiple"] == pytest.approx(2.0)

    def test_update_ignores_unknown_fields(self, db):
        tid = db.insert_live_trade(self._base())
        db.update_live_trade(tid, {"bogus_field": "x", "result": "loss"})
        row = db.fetch_live_trade(tid)
        assert row["result"] == "loss"    # valid field applied
        assert "bogus_field" not in row   # invalid field silently dropped

    def test_get_live_trades_filter_by_type(self, db):
        db.insert_live_trade(self._base("PAPER"))
        db.insert_live_trade(self._base("PAPER"))
        db.insert_live_trade(self._base("LIVE"))
        paper = db.get_live_trades(account_type="PAPER")
        live  = db.get_live_trades(account_type="LIVE")
        assert len(paper) == 2
        assert len(live)  == 1

    def test_get_open_live_trades(self, db):
        db.insert_live_trade(self._base())                            # open
        tid2 = db.insert_live_trade(self._base())
        db.update_live_trade(tid2, {"result": "win"})                 # closed
        open_trades = db.get_open_live_trades()
        assert len(open_trades) == 1
        assert open_trades.iloc[0]["result"] == "open"

    def test_live_paper_separation(self, db):
        db.insert_live_trade(self._base("PAPER"))
        db.insert_live_trade(self._base("LIVE"))
        paper_open = db.get_open_live_trades(account_type="PAPER")
        live_open  = db.get_open_live_trades(account_type="LIVE")
        assert len(paper_open) == 1
        assert len(live_open)  == 1

    def test_signal_params_json_roundtrip(self, db):
        params = {"fvg_depth": 0.3, "min_rr": 2.0, "entry_tf": "15m"}
        tid = db.insert_live_trade({**self._base(), "signal_params": params})
        row = db.fetch_live_trade(tid)
        stored = json.loads(row["signal_params"]) if isinstance(row["signal_params"], str) else row["signal_params"]
        assert stored["entry_tf"] == "15m"
