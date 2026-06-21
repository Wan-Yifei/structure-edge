"""Unit tests for analysis/fvg_watcher.py — config loading and scan_symbol_tf."""

import json
from unittest.mock import patch

import pandas as pd
import pytest

from analysis.fvg_watcher import load_fvg_watch_config, scan_symbol_tf


# ── helpers ───────────────────────────────────────────────────────────────────

def _klines(closes, highs=None, lows=None, opens=None, volumes=None):
    n = len(closes)
    closes = list(closes)
    highs  = highs  if highs  is not None else [c + 1.0 for c in closes]
    lows   = lows   if lows   is not None else [c - 1.0 for c in closes]
    opens  = opens  if opens  is not None else closes
    volumes = volumes if volumes is not None else [1000] * n
    return pd.DataFrame({
        "time_key": [f"2026-06-01 {9 + i:02d}:00:00" for i in range(n)],
        "open":     opens,
        "high":     highs,
        "low":      lows,
        "close":    closes,
        "volume":   volumes,
    })


def _bull_fvg_klines() -> pd.DataFrame:
    # candle[0] high=10, candle[2] low=12 -> bull gap 10-12, third candle idx=2
    return _klines(closes=[10, 11, 12], highs=[10, 11, 13], lows=[9, 10, 12])


# ── load_fvg_watch_config ──────────────────────────────────────────────────────

class TestLoadFvgWatchConfig:
    def test_loads_symbol_entries(self, tmp_path):
        path = tmp_path / "fvg_watch_params.json"
        path.write_text(json.dumps({
            "_note": "ignored",
            "US.SOXL": [{"tf": "15m", "min_gap_pct": 0.001}],
        }))
        cfg = load_fvg_watch_config(path)
        assert "_note" not in cfg
        assert cfg["US.SOXL"] == [{"tf": "15m", "min_gap_pct": 0.001}]

    def test_missing_file_returns_empty_dict(self, tmp_path):
        cfg = load_fvg_watch_config(tmp_path / "does_not_exist.json")
        assert cfg == {}


# ── scan_symbol_tf ────────────────────────────────────────────────────────────

class TestScanSymbolTf:
    def test_empty_klines_returns_empty_list(self):
        with patch("analysis.fvg_watcher.fetch_klines", return_value=pd.DataFrame()):
            hits = scan_symbol_tf("US.SOXL", "15m", {"min_gap_pct": 0.0}, "2026-01-01", "2026-01-02")
        assert hits == []

    def test_detects_gap_and_enriches_fields(self):
        klines = _bull_fvg_klines()
        params = {"min_gap_pct": 0.0, "require_displacement": False}
        with patch("analysis.fvg_watcher.fetch_klines", return_value=klines) as mock_fetch:
            hits = scan_symbol_tf("US.SOXL", "15m", params, "2026-01-01", "2026-01-02", force_refresh=True)

        mock_fetch.assert_called_once_with("US.SOXL", "15m", "2026-01-01", "2026-01-02", force_refresh=True)
        assert len(hits) == 1
        hit = hits[0]
        assert hit["symbol"]      == "US.SOXL"
        assert hit["tf"]          == "15m"
        assert hit["direction"]   == "bull"
        assert hit["zone_top"]    == pytest.approx(12.0)
        assert hit["zone_bottom"] == pytest.approx(10.0)
        assert hit["formed_time"] == "2026-06-01 11:00:00"  # third candle's time_key
        assert hit["filled"] is False
        assert hit["width_pct"]   == pytest.approx(2 / 11)
        assert json.loads(hit["params_json"]) == params

    def test_require_lvn_overlap_builds_profiles_and_filters(self):
        # Reuse the displacement-style klines but require an LVN overlap that
        # can never be satisfied (degenerate single-day volume profile) to
        # confirm the LVN path is wired in without needing to re-prove
        # gaps_for_combo's own LVN logic (already covered in test_smc.py).
        klines = _bull_fvg_klines()
        params = {
            "min_gap_pct": 0.0, "require_displacement": False,
            "require_lvn_overlap": True, "lvn_threshold": 0.30,
        }
        with patch("analysis.fvg_watcher.fetch_klines", return_value=klines):
            hits = scan_symbol_tf("US.SOXL", "15m", params, "2026-01-01", "2026-01-02")
        # Single-day frame -> build_daily_lvn_profiles has no preceding day,
        # so no profile is available and the LVN filter drops every gap.
        assert hits == []
