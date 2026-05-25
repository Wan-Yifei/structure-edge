"""SMC backtest runner with grid search.

Usage:
    uv run backtest/run.py --codes US.SNDK --start 2025-02-13 --end 2025-12-31
    uv run backtest/run.py --codes US.SNDK US.NVDA US.AAPL   # multi-stock sweep
    uv run backtest/run.py --fast                             # smoke test
    uv run backtest/run.py --no-resume                        # ignore checkpoint, run fresh

Fetches klines (or loads from cache), runs every (TF pair × parameter) combination
for each stock code, prints a ranked results table, and saves:
  backtest/results/<timestamp>/backtest_results.csv   — all results with a 'code' column
  backtest/results/<timestamp>/backtest_viz.png       — visualisation (single-code only)

Checkpoint / resume
-------------------
Results are checkpointed every --save-every completions (default 500) under
  backtest/results/checkpoints/<hash>.pkl
On the next run with identical config, completed combos are skipped automatically.
Use --no-resume to ignore the checkpoint and rerun everything from scratch.

Flags:
  --codes       One or more moomoo stock codes (default: US.SNDK)
  --fast        Smoke test — 2 TF pairs, minimal params
  --force       Re-fetch klines from API even if cached
  --no-viz      Skip the matplotlib chart
  --show-chart  Open chart interactively (blocks until closed)
  --top N       Number of top runs to print / show in equity panel
  --workers N   Override parallel worker count (default: from config)
  --no-resume   Ignore existing checkpoint; rerun all combos from scratch
  --save-every  Save checkpoint every N completions (default: 500)

Timeframes
----------
Native (fetched from moomoo):  1m  3m  5m  15m  30m  60m  1d
Synthetic (resampled from 60m): 2h  3h  4h
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import multiprocessing
import os
import pathlib
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional

from tqdm import tqdm

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from feeds.fetcher import fetch_klines
from backtest.engine  import ALGO_VERSION, BacktestParams, BacktestResult, run_backtest


def _git_commit_hash() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=pathlib.Path(__file__).parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


_COMMIT_HASH = _git_commit_hash()
from backtest.viz     import plot_backtest_results, plot_from_csv
from backtest.report  import generate_report
from backtest.logger  import make_listener, worker_init, get_logger
from backtest.db      import BacktestDB

_RESULTS_DIR    = pathlib.Path(__file__).parent / "results"
_CHECKPOINT_DIR = pathlib.Path(__file__).parent / "results" / "checkpoints"
_DEFAULT_CONFIG = pathlib.Path(__file__).parent.parent / "config" / "backtest.json"


def _load_json_config(path: pathlib.Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _resolve_workers(w: int) -> int:
    """≤ 0 → auto (cpu_count - 1, min 1)."""
    if w <= 0:
        return max(1, (os.cpu_count() or 4) - 1)
    return w


# ── Run-level configuration ───────────────────────────────────────────────────

@dataclass
class BacktestConfig:
    """Controls how the grid search is executed (not the strategy parameters)."""
    codes:         list[str] = field(default_factory=lambda: ["US.SNDK"])
    start:         str       = "2025-02-13"
    end:           str       = "2025-12-31"
    workers:       int       = field(default_factory=lambda: max(1, (os.cpu_count() or 4) - 1))
    top_n:         int       = 20
    fast:          bool      = False
    force_refresh: bool      = False
    no_viz:        bool      = False
    show_chart:    bool      = False


def _config_from_json(path: pathlib.Path) -> tuple[BacktestConfig, list, list]:
    """Load BacktestConfig + TF_PAIRS + TF_PAIRS_FAST from a JSON file."""
    raw = _load_json_config(path)
    cfg = BacktestConfig(
        codes=raw.get("codes", ["US.SNDK"]),
        start=raw.get("start", "2025-02-13"),
        end=raw.get("end",   "2025-12-31"),
        workers=_resolve_workers(raw.get("workers", -1)),
        top_n=raw.get("top_n", 20),
    )
    pairs      = [tuple(p) for p in raw.get("tf_pairs",      [])]
    pairs_fast = [tuple(p) for p in raw.get("tf_pairs_fast", [])]
    return cfg, pairs, pairs_fast


# ── Timeframe pairs — loaded from config.json at startup ─────────────────────

_TF_PAIRS_DEFAULT: list[tuple[str, str]] = [
    ("60m", "5m"),
    ("4h",  "5m"),
    ("60m", "15m"),
    ("2h",  "15m"),
    ("4h",  "15m"),
    ("1d",  "15m"),
    ("2h",  "30m"),
    ("4h",  "30m"),
    ("1d",  "30m"),
    ("4h",  "60m"),
    ("1d",  "60m"),
    ("1d",  "2h"),
    ("1d",  "4h"),
]

_TF_PAIRS_FAST_DEFAULT: list[tuple[str, str]] = [
    ("60m", "15m"),
    ("1d",  "60m"),
]

if _DEFAULT_CONFIG.exists():
    _json_cfg, TF_PAIRS, TF_PAIRS_FAST = _config_from_json(_DEFAULT_CONFIG)
else:
    _json_cfg      = BacktestConfig()
    TF_PAIRS       = _TF_PAIRS_DEFAULT
    TF_PAIRS_FAST  = _TF_PAIRS_FAST_DEFAULT

# ── Parameter grid ────────────────────────────────────────────────────────────

PARAM_GRID: dict[str, list] = {
    # Focused grid: derived from random-search results (2026-05-22, 50 samples, 15m/1m SNDK).
    # bos_count=2 and min_rr=1.5 dominated top performers; displacement_required added
    # (was absent from random sample due to grid default); rr=3.0 dropped (too few trades).
    "htf_window_bars":            [20, 30, 50],   # ~5 h / ~8 h / ~13 h at 15 m (min ~20 for reliable swing detection)
    "swing_lookback":             [2, 3],
    "bos_count":                  [2],
    "fvg_min_width_pct":          [0.001, 0.003, 0.005],
    "fvg_entry_depth_pct":        [0.05, 0.20, 0.50, 0.70],
    "require_ltf_confirmation":   [True, False],
    "displacement_required":      [True, False],
    "sl_buffer_pct":              [0.001, 0.003, 0.005],
    "max_sl_pct":                 [0.005, 0.010, 0.020, 0.030],
    "min_rr":                     [1.5, 2.0],
    "kd_sl_fallback":             [True, False],
}

PARAM_GRID_KD: dict[str, list] = {
    "htf_trend_methods": [("kd",)],
    "htf_trend_params": [
        {"kd_fast": 15, "kd_slow": 60, "kd_window":  5, "kd_flat_threshold": 0.0},
        {"kd_fast": 15, "kd_slow": 60, "kd_window": 10, "kd_flat_threshold": 0.0},
        {"kd_fast": 15, "kd_slow": 90, "kd_window":  5, "kd_flat_threshold": 0.0},
        {"kd_fast": 15, "kd_slow": 90, "kd_window": 10, "kd_flat_threshold": 0.0},
        {"kd_fast": 25, "kd_slow": 60, "kd_window":  5, "kd_flat_threshold": 0.0},
        {"kd_fast": 25, "kd_slow": 90, "kd_window": 10, "kd_flat_threshold": 0.0},
        {"kd_fast": 25, "kd_slow": 90, "kd_window": 20, "kd_flat_threshold": 0.0},
        {"kd_fast": 25, "kd_slow": 90, "kd_window": 10, "kd_flat_threshold": 0.05},
    ],
    "htf_window_bars":           [50],
    "swing_lookback":            [2],
    "bos_count":                 [2],
    "fvg_min_width_pct":         [0.001, 0.003, 0.005],
    "fvg_entry_depth_pct":       [0.05, 0.10, 0.20, 0.50],
    "require_ltf_confirmation":  [False],
    "displacement_required":     [False],
    "sl_buffer_pct":             [0.003, 0.005],
    "max_sl_pct":                [0.010, 0.020],
    "min_rr":                    [1.5, 2.0],
}

PARAM_GRID_COMBINED: dict[str, list] = {
    "htf_trend_methods": [("bos_choch", "kd")],
    "htf_trend_params": [
        {"kd_fast": 15, "kd_slow": 60, "kd_window":  5, "kd_flat_threshold": 0.0},
        {"kd_fast": 15, "kd_slow": 60, "kd_window": 10, "kd_flat_threshold": 0.0},
        {"kd_fast": 15, "kd_slow": 90, "kd_window":  5, "kd_flat_threshold": 0.0},
        {"kd_fast": 15, "kd_slow": 90, "kd_window": 10, "kd_flat_threshold": 0.0},
        {"kd_fast": 25, "kd_slow": 60, "kd_window":  5, "kd_flat_threshold": 0.0},
        {"kd_fast": 25, "kd_slow": 90, "kd_window": 10, "kd_flat_threshold": 0.0},
        {"kd_fast": 25, "kd_slow": 90, "kd_window": 20, "kd_flat_threshold": 0.0},
        {"kd_fast": 25, "kd_slow": 90, "kd_window": 10, "kd_flat_threshold": 0.05},
    ],
    "htf_window_bars":           [50],
    "swing_lookback":            [2],
    "bos_count":                 [2],
    "fvg_min_width_pct":         [0.001, 0.003, 0.005],
    "fvg_entry_depth_pct":       [0.05, 0.10, 0.20, 0.50],
    "require_ltf_confirmation":  [False],
    "displacement_required":     [False],
    "sl_buffer_pct":             [0.003, 0.005],
    "max_sl_pct":                [0.010, 0.020],
    "min_rr":                    [1.5, 2.0],
}

PARAM_GRID_FOCUSED: dict[str, list] = {
    # Adaptive segment mode: zero-crossing segmentation + conditional lag comp.
    # kd_fast=15 dominated in prior random search; slow=60/90 both included.
    # smooth=3 pre-smoothing; min_bars=3 prevents noise micro-segments.
    # atr_threshold=0.0/0.036: without vs with flat-segment filter (0.036 = p25, data-driven).
    # 8 KD configs × strategy grid = 1536 combos across 15m/1m + 15m/3m.
    "htf_trend_methods": [("bos_choch", "kd")],
    "htf_trend_params": [
        # no flat filter
        {"kd_fast": 15, "kd_slow": 60, "kd_smooth": 3, "kd_min_bars": 3, "kd_atr_threshold": 0.0},
        {"kd_fast": 15, "kd_slow": 60, "kd_smooth": 3, "kd_min_bars": 5, "kd_atr_threshold": 0.0},
        {"kd_fast": 15, "kd_slow": 90, "kd_smooth": 3, "kd_min_bars": 3, "kd_atr_threshold": 0.0},
        {"kd_fast": 15, "kd_slow": 90, "kd_smooth": 3, "kd_min_bars": 5, "kd_atr_threshold": 0.0},
        # atr_threshold=0.05: flat-segment filter
        {"kd_fast": 15, "kd_slow": 60, "kd_smooth": 3, "kd_min_bars": 3, "kd_atr_threshold": 0.05},
        {"kd_fast": 15, "kd_slow": 60, "kd_smooth": 3, "kd_min_bars": 5, "kd_atr_threshold": 0.05},
        {"kd_fast": 15, "kd_slow": 90, "kd_smooth": 3, "kd_min_bars": 3, "kd_atr_threshold": 0.05},
        {"kd_fast": 15, "kd_slow": 90, "kd_smooth": 3, "kd_min_bars": 5, "kd_atr_threshold": 0.05},
    ],
    "htf_window_bars":           [50],
    "swing_lookback":            [2],
    "bos_count":                 [2],
    "fvg_min_width_pct":         [0.001, 0.003, 0.005],
    "fvg_entry_depth_pct":       [0.05, 0.10, 0.20, 0.50],
    "require_ltf_confirmation":  [False],
    "displacement_required":     [False],
    "sl_buffer_pct":             [0.003, 0.005],
    "max_sl_pct":                [0.010, 0.020],
    "min_rr":                    [1.5, 2.0],
}

PARAM_GRID_MU: dict[str, list] = {
    # Focused grid derived from MU random-100 results (2026-05-25).
    # Core finding: direction_mismatch (80% of events) is the primary blocker.
    # fvg_max_age_bars added as a new lever — shorter values expire stale
    # counter-trend FVGs faster, directly attacking the mismatch problem.
    # Fixed: 15m/3m only, htf_window=50, lb=2, fvg_min=0.001, bos=2, ltf=False.
    "htf_window_bars":          [50],
    "swing_lookback":           [2],
    "bos_count":                [2],
    "fvg_min_width_pct":        [0.001],
    "fvg_entry_depth_pct":      [0.05, 0.10, 0.20, 0.50],
    "fvg_max_age_bars":         [15, 25, 50],
    "require_ltf_confirmation": [False],
    "displacement_required":    [True, False],
    "sl_buffer_pct":            [0.003, 0.005],
    "max_sl_pct":               [0.020, 0.030],
    "min_rr":                   [1.5, 2.0],
    "kd_sl_fallback":           [True, False],
}

PARAM_GRID_FAST: dict[str, list] = {
    "htf_window_bars":            [20],
    "swing_lookback":             [2],
    "bos_count":                  [1],
    "fvg_min_width_pct":          [0.002],
    "fvg_entry_depth_pct":        [0.20, 0.50],
    "require_ltf_confirmation":   [True, False],
    "displacement_required":      [False],
    "sl_buffer_pct":              [0.001],
    "max_sl_pct":                 [0.010],
    "min_rr":                     [1.5],
}

_KD_LEGACY = {"kd_fast": 15, "kd_slow": 60, "kd_smooth": 0}

# Top-10 combos from 20260523 SNDK focused-grid run (smc_v2), re-run with
# kd_smooth=0 (legacy mode) to isolate the effect of the over-refill guard.
# fvg_min_width_pct values read directly from that CSV (not the dataclass default).
LEGACY_TOP_PARAMS: list[BacktestParams] = [
    BacktestParams(trend_tf="15m", entry_tf="3m", swing_lookback=2, bos_count=2,
                   htf_window_bars=50, fvg_min_width_pct=0.005, fvg_entry_depth_pct=0.10,
                   sl_buffer_pct=0.003, max_sl_pct=0.02, min_rr=2.0,
                   htf_trend_methods=("bos_choch", "kd"),
                   htf_trend_params={**_KD_LEGACY, "kd_window": 10, "kd_flat_threshold": 0.0}),
    BacktestParams(trend_tf="15m", entry_tf="3m", swing_lookback=2, bos_count=2,
                   htf_window_bars=50, fvg_min_width_pct=0.005, fvg_entry_depth_pct=0.20,
                   sl_buffer_pct=0.003, max_sl_pct=0.02, min_rr=2.0,
                   htf_trend_methods=("bos_choch", "kd"),
                   htf_trend_params={**_KD_LEGACY, "kd_window": 10, "kd_flat_threshold": 0.0}),
    BacktestParams(trend_tf="15m", entry_tf="3m", swing_lookback=2, bos_count=2,
                   htf_window_bars=50, fvg_min_width_pct=0.005, fvg_entry_depth_pct=0.50,
                   sl_buffer_pct=0.003, max_sl_pct=0.02, min_rr=2.0,
                   htf_trend_methods=("bos_choch", "kd"),
                   htf_trend_params={**_KD_LEGACY, "kd_window": 10, "kd_flat_threshold": 0.0}),
    BacktestParams(trend_tf="15m", entry_tf="3m", swing_lookback=2, bos_count=2,
                   htf_window_bars=50, fvg_min_width_pct=0.001, fvg_entry_depth_pct=0.10,
                   sl_buffer_pct=0.005, max_sl_pct=0.02, min_rr=2.0,
                   htf_trend_methods=("bos_choch", "kd"),
                   htf_trend_params={**_KD_LEGACY, "kd_window": 5, "kd_flat_threshold": 0.0}),
    BacktestParams(trend_tf="15m", entry_tf="3m", swing_lookback=2, bos_count=2,
                   htf_window_bars=50, fvg_min_width_pct=0.001, fvg_entry_depth_pct=0.20,
                   sl_buffer_pct=0.005, max_sl_pct=0.02, min_rr=2.0,
                   htf_trend_methods=("bos_choch", "kd"),
                   htf_trend_params={**_KD_LEGACY, "kd_window": 5, "kd_flat_threshold": 0.0}),
    BacktestParams(trend_tf="15m", entry_tf="3m", swing_lookback=2, bos_count=2,
                   htf_window_bars=50, fvg_min_width_pct=0.001, fvg_entry_depth_pct=0.50,
                   sl_buffer_pct=0.005, max_sl_pct=0.02, min_rr=2.0,
                   htf_trend_methods=("bos_choch", "kd"),
                   htf_trend_params={**_KD_LEGACY, "kd_window": 5, "kd_flat_threshold": 0.0}),
    BacktestParams(trend_tf="15m", entry_tf="3m", swing_lookback=2, bos_count=2,
                   htf_window_bars=50, fvg_min_width_pct=0.001, fvg_entry_depth_pct=0.10,
                   sl_buffer_pct=0.005, max_sl_pct=0.01, min_rr=1.5,
                   htf_trend_methods=("bos_choch", "kd"),
                   htf_trend_params={**_KD_LEGACY, "kd_window": 10, "kd_flat_threshold": 0.1}),
    BacktestParams(trend_tf="15m", entry_tf="3m", swing_lookback=2, bos_count=2,
                   htf_window_bars=50, fvg_min_width_pct=0.001, fvg_entry_depth_pct=0.20,
                   sl_buffer_pct=0.005, max_sl_pct=0.01, min_rr=1.5,
                   htf_trend_methods=("bos_choch", "kd"),
                   htf_trend_params={**_KD_LEGACY, "kd_window": 10, "kd_flat_threshold": 0.1}),
    BacktestParams(trend_tf="15m", entry_tf="3m", swing_lookback=2, bos_count=2,
                   htf_window_bars=50, fvg_min_width_pct=0.001, fvg_entry_depth_pct=0.50,
                   sl_buffer_pct=0.005, max_sl_pct=0.01, min_rr=1.5,
                   htf_trend_methods=("bos_choch", "kd"),
                   htf_trend_params={**_KD_LEGACY, "kd_window": 10, "kd_flat_threshold": 0.1}),
    BacktestParams(trend_tf="15m", entry_tf="1m", swing_lookback=2, bos_count=2,
                   htf_window_bars=50, fvg_min_width_pct=0.005, fvg_entry_depth_pct=0.05,
                   sl_buffer_pct=0.005, max_sl_pct=0.02, min_rr=2.0,
                   htf_trend_methods=("bos_choch", "kd"),
                   htf_trend_params={**_KD_LEGACY, "kd_window": 10, "kd_flat_threshold": 0.0}),
]

_KD_ADAPTIVE = {"kd_fast": 15, "kd_slow": 60, "kd_smooth": 3, "kd_min_bars": 3}

# Same 7 strategy configs as above (flat_threshold=0.0 only — those are the valid
# smc_v2 baselines), re-run in adaptive-segment mode with two atr_threshold levels.
ADAPTIVE_TOP_PARAMS: list[BacktestParams] = [
    p
    for atr in (0.0, 0.036)
    for p in [
        BacktestParams(trend_tf="15m", entry_tf="3m", swing_lookback=2, bos_count=2,
                       htf_window_bars=50, fvg_min_width_pct=0.005, fvg_entry_depth_pct=0.10,
                       sl_buffer_pct=0.003, max_sl_pct=0.02, min_rr=2.0,
                       htf_trend_methods=("bos_choch", "kd"),
                       htf_trend_params={**_KD_ADAPTIVE, "kd_atr_threshold": atr}),
        BacktestParams(trend_tf="15m", entry_tf="3m", swing_lookback=2, bos_count=2,
                       htf_window_bars=50, fvg_min_width_pct=0.005, fvg_entry_depth_pct=0.20,
                       sl_buffer_pct=0.003, max_sl_pct=0.02, min_rr=2.0,
                       htf_trend_methods=("bos_choch", "kd"),
                       htf_trend_params={**_KD_ADAPTIVE, "kd_atr_threshold": atr}),
        BacktestParams(trend_tf="15m", entry_tf="3m", swing_lookback=2, bos_count=2,
                       htf_window_bars=50, fvg_min_width_pct=0.005, fvg_entry_depth_pct=0.50,
                       sl_buffer_pct=0.003, max_sl_pct=0.02, min_rr=2.0,
                       htf_trend_methods=("bos_choch", "kd"),
                       htf_trend_params={**_KD_ADAPTIVE, "kd_atr_threshold": atr}),
        BacktestParams(trend_tf="15m", entry_tf="3m", swing_lookback=2, bos_count=2,
                       htf_window_bars=50, fvg_min_width_pct=0.001, fvg_entry_depth_pct=0.10,
                       sl_buffer_pct=0.005, max_sl_pct=0.02, min_rr=2.0,
                       htf_trend_methods=("bos_choch", "kd"),
                       htf_trend_params={**_KD_ADAPTIVE, "kd_atr_threshold": atr}),
        BacktestParams(trend_tf="15m", entry_tf="3m", swing_lookback=2, bos_count=2,
                       htf_window_bars=50, fvg_min_width_pct=0.001, fvg_entry_depth_pct=0.20,
                       sl_buffer_pct=0.005, max_sl_pct=0.02, min_rr=2.0,
                       htf_trend_methods=("bos_choch", "kd"),
                       htf_trend_params={**_KD_ADAPTIVE, "kd_atr_threshold": atr}),
        BacktestParams(trend_tf="15m", entry_tf="3m", swing_lookback=2, bos_count=2,
                       htf_window_bars=50, fvg_min_width_pct=0.001, fvg_entry_depth_pct=0.50,
                       sl_buffer_pct=0.005, max_sl_pct=0.02, min_rr=2.0,
                       htf_trend_methods=("bos_choch", "kd"),
                       htf_trend_params={**_KD_ADAPTIVE, "kd_atr_threshold": atr}),
        BacktestParams(trend_tf="15m", entry_tf="1m", swing_lookback=2, bos_count=2,
                       htf_window_bars=50, fvg_min_width_pct=0.005, fvg_entry_depth_pct=0.05,
                       sl_buffer_pct=0.005, max_sl_pct=0.02, min_rr=2.0,
                       htf_trend_methods=("bos_choch", "kd"),
                       htf_trend_params={**_KD_ADAPTIVE, "kd_atr_threshold": atr}),
    ]
]


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _checkpoint_key(
    code: str, start: str, end: str, pairs: list, grid: dict,
    random_n: int = 0, random_seed: Optional[int] = None,
) -> str:
    """Stable hash identifying a unique (code, date, tf_pairs, param_grid[, random]) run."""
    payload: dict = {
        "code":  code,
        "start": start,
        "end":   end,
        "pairs": sorted([list(p) for p in pairs]),
        "grid":  {k: sorted(str(x) for x in v) for k, v in sorted(grid.items())},
    }
    if random_n > 0:
        payload["random_n"]    = random_n
        payload["random_seed"] = random_seed
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def _params_hash(params: BacktestParams) -> str:
    """Hash strategy params (excluding trend_tf/entry_tf) for DB run deduplication."""
    d = params.to_dict()
    d.pop("trend_tf")
    d.pop("entry_tf")
    return hashlib.md5(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _ckpt_path(key: str) -> pathlib.Path:
    return _CHECKPOINT_DIR / f"{key}.pkl"


# ── Date-range reuse helpers ──────────────────────────────────────────────────

_REUSE_WARMUP_DAYS = 14  # calendar days prepended to each gap for HTF warmup

def _merge_segments(segs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Merge overlapping / adjacent date-range segments, return sorted list."""
    if not segs:
        return []
    segs = sorted(segs)
    merged = [segs[0]]
    for s, e in segs[1:]:
        if s <= merged[-1][1]:          # overlap or adjacent
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _coverage_gaps(
    req_start: str, req_end: str,
    covered: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Return sub-ranges of [req_start, req_end] not covered by any segment.

    Args:
        req_start / req_end: the full requested date range (YYYY-MM-DD strings)
        covered: list of (start, end) already computed in the DB

    Returns:
        List of (gap_start, gap_end) that need to be run.
    """
    relevant = [
        (max(s, req_start), min(e, req_end))
        for s, e in covered
        if s <= req_end and e >= req_start
    ]
    merged = _merge_segments(relevant)

    gaps: list[tuple[str, str]] = []
    cursor = req_start
    for seg_start, seg_end in merged:
        if cursor < seg_start:
            gaps.append((cursor, _prev_day(seg_start)))
        cursor = _next_day(seg_end)
    if cursor <= req_end:
        gaps.append((cursor, req_end))
    return gaps


def _next_day(d: str) -> str:
    return (date.fromisoformat(d) + timedelta(days=1)).isoformat()

def _prev_day(d: str) -> str:
    return (date.fromisoformat(d) - timedelta(days=1)).isoformat()

def _warmup_start(gap_start: str) -> str:
    """Extend gap_start backward by _REUSE_WARMUP_DAYS for HTF context."""
    return (date.fromisoformat(gap_start) - timedelta(days=_REUSE_WARMUP_DAYS)).isoformat()


def _load_checkpoint(key: str) -> dict[int, BacktestResult]:
    p = _ckpt_path(key)
    if not p.exists():
        return {}
    try:
        with open(p, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[checkpoint] Cannot load {p.name}: {e} — starting fresh")
        return {}


def _save_checkpoint(key: str, completed: dict[int, BacktestResult]) -> None:
    _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    with open(_ckpt_path(key), "wb") as f:
        pickle.dump(completed, f, protocol=pickle.HIGHEST_PROTOCOL)


# ── Grid helpers ──────────────────────────────────────────────────────────────

def build_param_list(
    pairs: list[tuple[str, str]],
    grid: dict,
) -> list[BacktestParams]:
    keys   = list(grid.keys())
    values = list(grid.values())
    result: list[BacktestParams] = []
    for trend_tf, entry_tf in pairs:
        for combo in itertools.product(*values):
            result.append(BacktestParams(
                trend_tf=trend_tf,
                entry_tf=entry_tf,
                **dict(zip(keys, combo)),
            ))
    return result


def build_param_list_random(
    pairs: list[tuple[str, str]],
    grid: dict,
    n_samples: int = 300,
    seed: int = 42,
) -> list[BacktestParams]:
    """Random search: sample n_samples combinations per TF pair (with replacement).

    Much faster than exhaustive grid for large parameter spaces.
    seed ensures reproducibility; use different seeds for independent runs.
    """
    import random as _random
    rng = _random.Random(seed)
    result: list[BacktestParams] = []
    for trend_tf, entry_tf in pairs:
        for _ in range(n_samples):
            combo = {k: rng.choice(v) for k, v in grid.items()}
            result.append(BacktestParams(trend_tf=trend_tf, entry_tf=entry_tf, **combo))
    return result


def _write_review_trades(code: str, params: BacktestParams, trades: list) -> None:
    """Write trades to review_trades.duckdb so trade_viewer can look them up by ID."""
    if not trades:
        return
    try:
        from backtest.db import ReviewTradesDB
        with ReviewTradesDB() as rdb:
            rdb.insert_trades(code, params.to_dict(), trades)
    except Exception as exc:
        get_logger("main").warning("review DB write failed: %s", exc)


def _worker(args: tuple) -> tuple[int, BacktestResult]:
    """Top-level so ProcessPoolExecutor can pickle it on Windows (spawn mode)."""
    idx, params, htf, ltf = args
    log = get_logger(f"W{idx:04d} {params.trend_tf}/{params.entry_tf} lb{params.swing_lookback}")
    log.debug("Starting combo %d: %s", idx, params.label())
    result = run_backtest(htf, ltf, params)
    log.debug("Done: %d trades", result.n_trades)
    return idx, result


def _fmt_row(d: dict) -> str:
    return (
        f"  trades={d['n_trades']:3d}  wr={d['win_rate']:.1%}  "
        f"R={d['total_r']:+.1f}  avgR={d['avg_r']:+.3f}  "
        f"PF={d['profit_factor']:.2f}  DD={d['max_drawdown_r']:.2f}  "
        f"maxL={d['max_loss_r']:.2f}"
    )


def run_grid(
    code: str,
    klines: dict[str, pd.DataFrame],
    params_list: list[BacktestParams],
    workers: int | None = None,
    checkpoint_key: str | None = None,
    no_resume: bool = False,
    save_every: int = 500,
    log_path: Optional[pathlib.Path] = None,
    db: Optional[BacktestDB] = None,
    start_date: str = "",
    end_date: str = "",
    no_reuse: bool = False,
) -> list[BacktestResult]:
    n = len(params_list)
    log = get_logger("main")

    # Load checkpoint
    done: dict[int, BacktestResult] = {}
    if checkpoint_key and not no_resume:
        done = _load_checkpoint(checkpoint_key)
        if done:
            log.info("Resuming from checkpoint: %d/%d combos already done", len(done), n)
            print(f"  Resuming from checkpoint: {len(done)}/{n} combos already done")

    # Build remaining tasks, skipping completed and missing data.
    # For combos not in the checkpoint, check DB for partial date coverage (--no-reuse bypasses).
    tasks: list[tuple] = []
    db_preloaded: dict[int, BacktestResult] = {}  # combos fully satisfied from DB

    for idx, params in enumerate(params_list, 1):
        if idx in done:
            continue
        htf = klines.get(params.trend_tf)
        ltf = klines.get(params.entry_tf)
        if htf is None or ltf is None:
            log.warning("SKIP combo %d — missing data for %s/%s", idx, params.trend_tf, params.entry_tf)
            print(f"  [{idx}/{n}] SKIP — missing data for {params.trend_tf}/{params.entry_tf}")
            continue

        # ── DB date-range reuse ────────────────────────────────────────────
        if db is not None and not no_reuse and start_date and end_date:
            phash   = _params_hash(params)
            covered = db.covered_segments(phash, code, params.trend_tf, params.entry_tf, ALGO_VERSION)
            gaps    = _coverage_gaps(start_date, end_date, covered)

            if not gaps:
                # Fully covered — load trades from DB, skip engine entirely
                cached = db.load_trades_in_range(
                    phash, code, params.trend_tf, params.entry_tf,
                    start_date, end_date, ALGO_VERSION,
                )
                bt = BacktestResult(params=params)
                bt.trades = cached
                db_preloaded[idx] = bt
                log.info("DB reuse combo %d — %d trades loaded (fully covered)", idx, len(cached))
                continue

            elif covered:  # partially covered — run only the gaps
                # Partially covered — run only the gap segments with warmup extension,
                # then merge with cached trades from covered portions.
                gap_trades: list = []
                for gap_start, gap_end in gaps:
                    ws  = _warmup_start(gap_start)
                    # Slice klines to warmup_start → gap_end
                    htf_g = htf[htf["time_key"].astype(str) >= ws].reset_index(drop=True)
                    ltf_g = ltf[ltf["time_key"].astype(str) >= ws].reset_index(drop=True)
                    htf_g = htf_g[htf_g["time_key"].astype(str) <= gap_end].reset_index(drop=True)
                    ltf_g = ltf_g[ltf_g["time_key"].astype(str) <= gap_end].reset_index(drop=True)
                    if htf_g.empty or ltf_g.empty:
                        continue
                    r = run_backtest(htf_g, ltf_g, params)
                    # Keep only trades that entered in the actual gap (not the warmup zone)
                    gap_trades.extend(
                        t for t in r.trades
                        if str(t.entry_time)[:10] >= gap_start
                    )

                # Load cached trades from covered portions
                cached = db.load_trades_in_range(
                    phash, code, params.trend_tf, params.entry_tf,
                    start_date, end_date, ALGO_VERSION,
                )
                all_trades = sorted(
                    cached + gap_trades, key=lambda t: str(t.entry_time)
                )
                bt = BacktestResult(params=params)
                bt.trades = all_trades
                # Write new gap trades to DB under a new run_id per gap
                for gap_start, gap_end in gaps:
                    new_trades = [
                        t for t in gap_trades
                        if gap_start <= str(t.entry_time)[:10] <= gap_end
                    ]
                    if new_trades:
                        run_id = db.get_or_create_run(
                            phash, params.to_dict(), code,
                            params.trend_tf, params.entry_tf, gap_start, gap_end,
                            ALGO_VERSION, _COMMIT_HASH,
                        )[0]
                        db.mark_running(run_id)
                        db.write_trades(run_id, code, new_trades)
                        _write_review_trades(code, params, new_trades)
                        db.write_stats(run_id, bt)
                        db.mark_done(run_id)
                db_preloaded[idx] = bt
                log.info(
                    "DB reuse combo %d — %d cached + %d new trades (%d gaps filled)",
                    idx, len(cached), len(gap_trades), len(gaps),
                )
                continue

        tasks.append((idx, params, htf, ltf))

    bt_results: dict[int, BacktestResult] = {**done, **db_preloaded}
    save_counter = 0

    # One shared queue — listener in main process, workers send via QueueHandler
    import logging as _logging
    log_q: multiprocessing.Queue = multiprocessing.Queue(-1)
    listener = None
    if log_path is not None:
        import logging.handlers as _lh
        fh = _lh.RotatingFileHandler(log_path, encoding="utf-8", maxBytes=0)
        fh.setFormatter(_logging.Formatter(
            "%(asctime)s [%(tag)-26s] %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
        ))
        listener = _lh.QueueListener(log_q, fh, respect_handler_level=True)
        listener.start()

    log_level = _logging.DEBUG if log_path else _logging.WARNING

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=worker_init,
        initargs=(log_q, log_level),
    ) as ex:
        futures = {ex.submit(_worker, t): t[0] for t in tasks}
        bar = tqdm(
            total=n,
            initial=len(done) + len(db_preloaded),
            ncols=90,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )
        for fut in as_completed(futures):
            orig_idx, bt = fut.result()
            bt_results[orig_idx] = bt
            save_counter += 1

            if db is not None:
                try:
                    p = params_list[orig_idx - 1]
                    run_id, needs_write = db.get_or_create_run(
                        _params_hash(p), p.to_dict(), code,
                        p.trend_tf, p.entry_tf, start_date, end_date,
                        ALGO_VERSION, _COMMIT_HASH,
                    )
                    if needs_write:
                        db.mark_running(run_id)
                        db.write_trades(run_id, code, bt.trades)
                        _write_review_trades(code, p, bt.trades)
                        db.write_stats(run_id, bt)
                        db.mark_done(run_id)
                except Exception as exc:
                    log.warning("DB write failed for combo %d: %s", orig_idx, exc)

            d = bt.summary_dict()
            bar.set_postfix_str(
                f"{bt.params.trend_tf}/{bt.params.entry_tf}"
                f"  T={d['n_trades']} WR={d['win_rate']:.0%}"
                f"  R={d['total_r']:+.1f} PF={d['profit_factor']:.2f}",
                refresh=False,
            )
            bar.update(1)

            if checkpoint_key and save_counter % save_every == 0:
                _save_checkpoint(checkpoint_key, bt_results)
        bar.close()

    if listener is not None:
        listener.stop()

    # Final checkpoint save
    if checkpoint_key:
        _save_checkpoint(checkpoint_key, bt_results)
        print(f"  Checkpoint saved ({len(bt_results)} combos) → {_ckpt_path(checkpoint_key).name}")

    return [bt_results[i] for i in sorted(bt_results) if bt_results.get(i) is not None]


def main() -> None:
    ap = argparse.ArgumentParser(description="SMC backtest grid search")
    ap.add_argument("--config",  default=None, metavar="PATH",
                    help=f"JSON config file (default: {_DEFAULT_CONFIG})")
    ap.add_argument("--codes", nargs="+", default=None,
                    help="One or more moomoo stock codes (overrides config)")
    ap.add_argument("--start", default=None, help="YYYY-MM-DD (overrides config)")
    ap.add_argument("--end",   default=None, help="YYYY-MM-DD (overrides config)")
    ap.add_argument("--fast",       action="store_true", help="Smoke test — 2 TF pairs, minimal params")
    ap.add_argument("--legacy-top",   action="store_true", help="Run LEGACY_TOP_PARAMS (top-10 smc_v2 combos, kd_smooth=0)")
    ap.add_argument("--adaptive-top", action="store_true", help="Run ADAPTIVE_TOP_PARAMS (top-7 smc_v2 combos in adaptive KD mode)")
    ap.add_argument("--kd",         action="store_true", help="Use KD trend grid (PARAM_GRID_KD)")
    ap.add_argument("--combined",   action="store_true", help="Use bos_choch+kd consensus grid (PARAM_GRID_COMBINED)")
    ap.add_argument("--grid",       action="store_true", help="Use focused exhaustive grid derived from random search (PARAM_GRID_FOCUSED)")
    ap.add_argument("--force",      action="store_true", help="Re-fetch klines from API")
    ap.add_argument("--no-viz",     action="store_true", help="Skip the matplotlib visualisation")
    ap.add_argument("--show-chart", action="store_true", help="Open chart interactively (blocks)")
    ap.add_argument("--top",        type=int, default=None,
                    help="Print top N results ranked by profit factor (overrides config)")
    ap.add_argument("--workers",    type=int, default=None,
                    help="Parallel workers; 0 or negative = auto (overrides config)")
    ap.add_argument("--no-resume",  action="store_true",
                    help="Ignore existing checkpoint; rerun all combos from scratch")
    ap.add_argument("--no-reuse",   action="store_true",
                    help="Ignore DB date-range cache; always run engine for full date range")
    ap.add_argument("--save-every", type=int, default=500, metavar="N",
                    help="Save checkpoint every N completions (default: 500)")
    ap.add_argument("--random",     type=int, default=0, metavar="N",
                    help="Random search: N samples per TF pair (0 = exhaustive grid)")
    ap.add_argument("--seed",       type=int, default=42,
                    help="Random seed for --random (default: 42)")
    ap.add_argument("--min-trades", type=int, default=10, metavar="N",
                    help="Exclude combos with fewer than N trades from top-N ranking (default: 10)")
    ap.add_argument("--no-report",  action="store_true",
                    help="Skip HTML report generation")
    ap.add_argument("--mu",         action="store_true",
                    help="Use MU-focused grid (PARAM_GRID_MU), forces 15m/3m only")
    ap.add_argument("--from-csv",   metavar="PATH",
                    help="Regenerate chart/report from an existing CSV (skips backtest)")
    args = ap.parse_args()

    if args.from_csv:
        csv_in = pathlib.Path(args.from_csv)
        if not csv_in.exists():
            print(f"ERROR: CSV not found: {csv_in}")
            sys.exit(1)
        cfg_top = args.top or BacktestConfig().top_n
        if not args.no_report:
            generate_report(csv_in, top_n=cfg_top, open_browser=args.show_chart)
        else:
            plot_from_csv(csv_path=csv_in, show=args.show_chart, top_n=cfg_top)
        return

    # ── Load JSON config (CLI flag > default path > hardcoded fallback) ───
    config_path = pathlib.Path(args.config) if args.config else _DEFAULT_CONFIG
    if config_path.exists():
        cfg, pairs_normal, pairs_fast = _config_from_json(config_path)
    else:
        cfg          = BacktestConfig()
        pairs_normal = _TF_PAIRS_DEFAULT
        pairs_fast   = _TF_PAIRS_FAST_DEFAULT

    # ── CLI overrides ─────────────────────────────────────────────────────
    if args.codes:   cfg.codes         = args.codes
    if args.start:   cfg.start         = args.start
    if args.end:     cfg.end           = args.end
    if args.top:     cfg.top_n         = args.top
    if args.fast:    cfg.fast          = True
    if args.force:   cfg.force_refresh = True
    if args.no_viz:  cfg.no_viz        = True
    if args.show_chart: cfg.show_chart = True
    if args.workers is not None:
        cfg.workers = _resolve_workers(args.workers)

    if args.legacy_top:
        params = LEGACY_TOP_PARAMS
        pairs  = list({(p.trend_tf, p.entry_tf) for p in params})
    elif args.adaptive_top:
        params = ADAPTIVE_TOP_PARAMS
        pairs  = list({(p.trend_tf, p.entry_tf) for p in params})
    else:
        pairs  = pairs_fast if cfg.fast else pairs_normal
        if cfg.fast:
            grid = PARAM_GRID_FAST
        elif args.mu:
            grid  = PARAM_GRID_MU
            pairs = [("15m", "3m")]   # MU grid is 15m/3m only
        elif args.grid:
            grid = PARAM_GRID_FOCUSED
        elif args.combined:
            grid = PARAM_GRID_COMBINED
        elif args.kd:
            grid = PARAM_GRID_KD
        else:
            grid = PARAM_GRID
        if args.random > 0:
            params = build_param_list_random(pairs, grid, n_samples=args.random, seed=args.seed)
        else:
            params = build_param_list(pairs, grid)

    needed_tfs = {tf for trend_tf, entry_tf in pairs for tf in (trend_tf, entry_tf)}
    tf_order   = ["1m", "3m", "5m", "15m", "30m", "60m", "2h", "3h", "4h", "1d"]
    sorted_tfs = sorted(needed_tfs, key=lambda t: tf_order.index(t) if t in tf_order else 99)

    search_mode = f"random(n={args.random}, seed={args.seed})" if args.random > 0 else "exhaustive"
    print(f"Codes:       {cfg.codes}")
    print(f"Date range:  {cfg.start} → {cfg.end}")
    print(f"TF pairs:    {len(pairs)}")
    print(f"Search:      {search_mode}")
    print(f"Combos/code: {len(params)}")
    print(f"Workers:     {cfg.workers}")
    print(f"Resume:      {'disabled (--no-resume)' if args.no_resume else 'enabled'}")
    print(f"Total runs:  {len(params) * len(cfg.codes)}")

    if args.mu:
        grid_tag = "mu_"
    elif args.grid:
        grid_tag = "focused_"
    elif args.combined:
        grid_tag = "combined_"
    elif args.kd:
        grid_tag = "kd_"
    else:
        grid_tag = ""
    if cfg.fast:
        mode_tag = "smoke"
    elif args.legacy_top:
        mode_tag = "legacy_top10"
    elif args.adaptive_top:
        mode_tag = "adaptive_top14"
    elif args.random > 0:
        mode_tag = f"{grid_tag}random_{args.random}"
    else:
        mode_tag = f"{grid_tag}grid"
    run_tag     = f"{datetime.now().strftime('%Y%m%d_%H%M')}_{ALGO_VERSION}_{mode_tag}"
    results_dir = _RESULTS_DIR / run_tag
    csv_path    = results_dir / "backtest_results.csv"

    all_frames: list[pd.DataFrame] = []
    results_dir.mkdir(parents=True, exist_ok=True)

    db = BacktestDB()
    for code in cfg.codes:
        print(f"\n── Fetching klines: {code} ───────────────────────────────────────────\n")
        klines: dict[str, pd.DataFrame] = {}
        for tf in sorted_tfs:
            df = fetch_klines(
                code=code, ktype=tf,
                start=cfg.start, end=cfg.end,
                force_refresh=cfg.force_refresh,
            )
            bar_range = (
                f"{df['time_key'].iloc[0]} … {df['time_key'].iloc[-1]}"
                if len(df) else "—"
            )
            print(f"  {tf}: {len(df)} bars  ({bar_range})")
            klines[tf] = df

        ck_key = _checkpoint_key(
            code, cfg.start, cfg.end, pairs,
            grid if not (args.legacy_top or args.adaptive_top) else {"_explicit": [p.label() for p in params]},
            random_n=args.random, random_seed=args.seed if args.random > 0 else None,
        )
        print(f"\n── Running grid for {code} ({cfg.workers} workers) ─────────────────────\n")
        print(f"  Checkpoint key: {ck_key}")
        bt_results = run_grid(
            code, klines, params,
            workers=cfg.workers,
            checkpoint_key=ck_key,
            no_resume=args.no_resume,
            save_every=args.save_every,
            log_path=results_dir / f"run_{code.replace('.','_')}.log",
            db=db,
            start_date=cfg.start,
            end_date=cfg.end,
            no_reuse=args.no_reuse,
        )

        if not bt_results:
            print(f"No results for {code}.")
            continue

        df_code = pd.DataFrame([r.summary_dict() for r in bt_results])
        df_code.insert(0, "code", code)
        all_frames.append(df_code)

        # ── Per-code CSV ──────────────────────────────────────────────────
        code_slug = code.replace(".", "_")
        code_csv  = results_dir / f"results_{code_slug}.csv"
        code_viz  = results_dir / f"viz_{code_slug}.png"
        df_code.to_csv(code_csv, index=False)
        print(f"  Saved {len(df_code)} results → {code_csv}")

        # ── Per-code top N ────────────────────────────────────────────────
        min_trades = args.min_trades
        df_ranked_code = (
            df_code[df_code["n_trades"] >= min_trades]
            .sort_values(["profit_factor", "total_r"], ascending=[False, False])
            .head(cfg.top_n)
        )
        n_filtered = len(df_code) - len(df_code[df_code["n_trades"] >= min_trades])
        print(f"\n── Top {cfg.top_n} [{code}] by profit factor  (min_trades≥{min_trades}, {n_filtered} excluded) ──\n")
        for _, row in df_ranked_code.iterrows():
            p = BacktestParams.from_dict(row.to_dict())
            print(f"  {p.label()}")
            print(_fmt_row(row.to_dict()))
            print()

        # ── Per-code HTML report (default) ───────────────────────────────
        if not cfg.no_viz and not args.no_report:
            generate_report(
                code_csv,
                output_path=code_csv.parent / f"report_{code_slug}.html",
                top_n=cfg.top_n,
                open_browser=cfg.show_chart,
            )

        # ── Per-code matplotlib PNG (opt-in via --show-chart or legacy) ──
        if cfg.show_chart and not cfg.no_viz:
            ranked = sorted(bt_results, key=lambda r: r.profit_factor, reverse=True)
            plot_backtest_results(
                ranked, df_code,
                top_n=min(cfg.top_n, 5),
                save_path=code_viz,
                show=cfg.show_chart,
            )
            print(f"  Chart → {code_viz}")

    db.close()

    if not all_frames:
        print("No results.")
        return

    # ── Combined CSV (all codes) ──────────────────────────────────────────
    if len(all_frames) > 1:
        df_out = pd.concat(all_frames, ignore_index=True)
        df_out.to_csv(csv_path, index=False)
        print(f"\nCombined {len(df_out)} results ({len(cfg.codes)} codes) → {csv_path}")

        df_ranked = (
            df_out[df_out["n_trades"] >= min_trades]
            .sort_values(["profit_factor", "total_r"], ascending=[False, False])
            .head(cfg.top_n)
        )
        print(f"\n── Top {cfg.top_n} across all codes  (min_trades≥{min_trades}) ───────────────\n")
        for _, row in df_ranked.iterrows():
            p = BacktestParams.from_dict(row.to_dict())
            print(f"[{row['code']}]  {p.label()}")
            print(_fmt_row(row.to_dict()))
            print()


if __name__ == "__main__":
    main()
