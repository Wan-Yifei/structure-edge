"""Unit tests for analysis/fvg_backscan.py — pair resolution and backscan glue."""

from unittest.mock import patch

import pandas as pd

from analysis.fvg_backscan import _resolve_pairs, run_backscan


# ── _resolve_pairs ────────────────────────────────────────────────────────────

class TestResolvePairs:
    def _config(self):
        return {
            "US.SOXL": [{"tf": "15m", "min_gap_pct": 0.001}, {"tf": "30m", "min_gap_pct": 0.0005}],
            "US.SOXS": [{"tf": "15m", "min_gap_pct": 0.002}],
        }

    def test_no_filters_returns_every_pair(self):
        pairs = _resolve_pairs(self._config(), symbol=None, tf=None)
        assert len(pairs) == 3

    def test_symbol_filter(self):
        pairs = _resolve_pairs(self._config(), symbol="US.SOXL", tf=None)
        assert [sym for sym, _ in pairs] == ["US.SOXL", "US.SOXL"]

    def test_tf_filter(self):
        pairs = _resolve_pairs(self._config(), symbol=None, tf="15m")
        assert len(pairs) == 2
        assert all(entry["tf"] == "15m" for _, entry in pairs)

    def test_symbol_and_tf_filter(self):
        pairs = _resolve_pairs(self._config(), symbol="US.SOXL", tf="30m")
        assert len(pairs) == 1
        assert pairs[0] == ("US.SOXL", {"tf": "30m", "min_gap_pct": 0.0005})

    def test_unknown_symbol_returns_empty(self):
        pairs = _resolve_pairs(self._config(), symbol="US.NVDA", tf=None)
        assert pairs == []


# ── run_backscan ──────────────────────────────────────────────────────────────

class TestRunBackscan:
    def test_aggregates_hits_across_pairs(self, tmp_path):
        config_path = tmp_path / "fvg_watch_params.json"
        config_path.write_text(
            '{"US.SOXL": [{"tf": "15m", "min_gap_pct": 0.001}, {"tf": "30m", "min_gap_pct": 0.0005}]}'
        )

        def fake_scan(symbol, tf, params, start, end, force_refresh=False):
            return [{
                "symbol": symbol, "tf": tf, "direction": "bull",
                "zone_top": 102.5, "zone_bottom": 102.0,
                "formed_time": "2026-06-20 14:30:00", "filled": False,
                "width_pct": 0.005, "params_json": "{}",
            }]

        with patch("analysis.fvg_backscan.scan_symbol_tf", side_effect=fake_scan):
            df = run_backscan(config_path, symbol="US.SOXL", tf=None, start="2026-06-01", end="2026-06-20")

        assert len(df) == 2
        assert set(df["tf"]) == {"15m", "30m"}

    def test_no_matching_pairs_returns_empty_frame(self, tmp_path):
        config_path = tmp_path / "fvg_watch_params.json"
        config_path.write_text('{"US.SOXL": [{"tf": "15m", "min_gap_pct": 0.001}]}')

        df = run_backscan(config_path, symbol="US.NVDA", tf=None, start="2026-06-01", end="2026-06-20")
        assert isinstance(df, pd.DataFrame)
        assert df.empty
