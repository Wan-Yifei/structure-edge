"""Unit tests for db/signals.py — SignalsDB CRUD and query operations."""

import json
import pathlib
import uuid

import pytest

from db.signals import SignalsDB


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_sig(**kw) -> dict:
    base = {
        "signal_id":          str(uuid.uuid4()),
        "symbol":             "US.AAPL",
        "direction":          "bull",
        "signal_time":        "2026-06-07 10:00:00",
        "trend_tf":           "1h",
        "entry_tf":           "15m",
        "entry_zone_top":     185.0,
        "entry_zone_bottom":  182.0,
        "sl_price":           180.0,
        "tp_price":           192.0,
        "rr_ratio":           2.33,
        "bos_price":          183.5,
        "strategy":           "smc",
        "params_json":        json.dumps({"trend_tf": "1h", "entry_tf": "15m"}),
        "algo_version":       "smc_v2.4",
        "source":             "auto",
        "status":             "open",
        "closed_at":          None,
        "created_at":         "2026-06-07 10:00:01",
    }
    base.update(kw)
    return base


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def db(tmp_path):
    instance = SignalsDB(tmp_path / "test_signals.db")
    yield instance
    instance.close()


# ── insert & read-back ────────────────────────────────────────────────────────

def test_insert_and_get_open(db):
    sig = _make_sig()
    returned_id = db.insert_signal(sig)
    assert returned_id == sig["signal_id"]

    open_sigs = db.get_open_signals("US.AAPL")
    assert len(open_sigs) == 1
    row = open_sigs[0]
    assert row["symbol"] == "US.AAPL"
    assert row["direction"] == "bull"
    assert abs(row["entry_zone_top"] - 185.0) < 1e-6
    assert row["status"] == "open"


def test_insert_or_ignore_duplicate(db):
    sig = _make_sig()
    db.insert_signal(sig)
    db.insert_signal(sig)  # same signal_id — should silently skip
    assert len(db.get_open_signals("US.AAPL")) == 1


def test_insert_without_signal_id_generates_uuid(db):
    sig = _make_sig()
    del sig["signal_id"]
    returned_id = db.insert_signal(sig)
    assert returned_id  # non-empty string
    assert len(db.get_open_signals("US.AAPL")) == 1


# ── update_status ─────────────────────────────────────────────────────────────

def test_update_status_hit_tp(db):
    sig = _make_sig()
    sid = db.insert_signal(sig)
    db.update_status(sid, "hit_tp", "2026-06-07 11:00:00")

    open_sigs = db.get_open_signals("US.AAPL")
    assert len(open_sigs) == 0  # no longer open

    all_sigs = db.query_signals("US.AAPL", "2026-01-01")
    assert len(all_sigs) == 1
    assert all_sigs[0]["status"] == "hit_tp"
    assert all_sigs[0]["closed_at"] == "2026-06-07 11:00:00"


def test_update_status_sets_closed_at_if_omitted(db):
    sig = _make_sig()
    sid = db.insert_signal(sig)
    db.update_status(sid, "expired")

    row = db.query_signals("US.AAPL", "2026-01-01")[0]
    assert row["status"] == "expired"
    assert row["closed_at"] is not None  # auto-filled


# ── query_signals ─────────────────────────────────────────────────────────────

def test_query_signals_since_filter(db):
    db.insert_signal(_make_sig(signal_id=str(uuid.uuid4()), signal_time="2026-06-01 09:00:00"))
    db.insert_signal(_make_sig(signal_id=str(uuid.uuid4()), signal_time="2026-06-07 10:00:00"))

    recent = db.query_signals("US.AAPL", "2026-06-05")
    assert len(recent) == 1
    assert recent[0]["signal_time"] == "2026-06-07 10:00:00"


def test_query_signals_status_filter(db):
    s1 = _make_sig(signal_id=str(uuid.uuid4()))
    s2 = _make_sig(signal_id=str(uuid.uuid4()))
    sid1 = db.insert_signal(s1)
    db.insert_signal(s2)
    db.update_status(sid1, "hit_tp")

    open_only = db.query_signals("US.AAPL", "2026-01-01", status="open")
    assert len(open_only) == 1
    assert open_only[0]["signal_id"] == s2["signal_id"]

    hit_tp = db.query_signals("US.AAPL", "2026-01-01", status="hit_tp")
    assert len(hit_tp) == 1


# ── get_all_open_signals ──────────────────────────────────────────────────────

def test_get_all_open_signals_multi_symbol(db):
    db.insert_signal(_make_sig(symbol="US.AAPL", signal_id=str(uuid.uuid4())))
    db.insert_signal(_make_sig(symbol="US.TSLA", signal_id=str(uuid.uuid4())))
    db.insert_signal(_make_sig(symbol="US.NVDA", signal_id=str(uuid.uuid4())))

    all_open = db.get_all_open_signals()
    symbols = {r["symbol"] for r in all_open}
    assert symbols == {"US.AAPL", "US.TSLA", "US.NVDA"}


# ── context manager ───────────────────────────────────────────────────────────

def test_context_manager(tmp_path):
    sig = _make_sig()
    db_path = tmp_path / "ctx.db"
    with SignalsDB(db_path) as db:
        db.insert_signal(sig)
    # After close, reopen and verify persistence
    with SignalsDB(db_path) as db:
        rows = db.get_open_signals("US.AAPL")
        assert len(rows) == 1


# ── fvg_watch_signals (lightweight "FVG formed" alerts) ─────────────────────

def _make_fvg_watch(**kw) -> dict:
    base = {
        "signal_id":   str(uuid.uuid4()),
        "symbol":      "US.SOXL",
        "tf":          "15m",
        "direction":   "bull",
        "zone_top":    102.5,
        "zone_bottom": 102.0,
        "formed_time": "2026-06-20 14:30:00",
        "filled":      False,
        "params_json": json.dumps({"min_gap_pct": 0.001}),
        "status":      "open",
        "created_at":  "2026-06-20 14:30:01",
    }
    base.update(kw)
    return base


def test_insert_and_get_open_fvg_watch(db):
    sig = _make_fvg_watch()
    returned_id = db.insert_fvg_watch(sig)
    assert returned_id == sig["signal_id"]

    open_sigs = db.get_open_fvg_watch("US.SOXL", "15m")
    assert len(open_sigs) == 1
    row = open_sigs[0]
    assert row["symbol"] == "US.SOXL"
    assert row["tf"] == "15m"
    assert abs(row["zone_top"] - 102.5) < 1e-6
    assert row["status"] == "open"


def test_get_open_fvg_watch_filters_by_tf(db):
    db.insert_fvg_watch(_make_fvg_watch(tf="15m"))
    db.insert_fvg_watch(_make_fvg_watch(signal_id=str(uuid.uuid4()), tf="30m"))

    assert len(db.get_open_fvg_watch("US.SOXL", "15m")) == 1
    assert len(db.get_open_fvg_watch("US.SOXL", "30m")) == 1
    assert len(db.get_open_fvg_watch("US.SOXL", "60m")) == 0


def test_insert_fvg_watch_or_ignore_duplicate(db):
    sig = _make_fvg_watch()
    db.insert_fvg_watch(sig)
    db.insert_fvg_watch(sig)  # same signal_id — should silently skip
    assert len(db.get_open_fvg_watch("US.SOXL", "15m")) == 1


def test_query_fvg_watch_since_and_status_filter(db):
    sid1 = db.insert_fvg_watch(_make_fvg_watch(formed_time="2026-06-01 09:00:00"))
    db.insert_fvg_watch(_make_fvg_watch(signal_id=str(uuid.uuid4()), formed_time="2026-06-07 10:00:00"))

    recent = db.query_fvg_watch("US.SOXL", "2026-06-05")
    assert len(recent) == 1
    assert recent[0]["formed_time"] == "2026-06-07 10:00:00"

    all_sigs = db.query_fvg_watch("US.SOXL", "2026-01-01", status="open")
    assert len(all_sigs) == 2

    none_filled = db.query_fvg_watch("US.SOXL", "2026-01-01", status="filled")
    assert len(none_filled) == 0
