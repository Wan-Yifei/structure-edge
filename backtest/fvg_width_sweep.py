"""FVG width/count parameter sweep.

Finds the strategy/smc/fvg.py detect_fvg() parameters that maximize FVG width
and FVG count jointly, per stock and per timeframe. Optionally also scores
each FVG against the prior trading day's volume profile (LVN overlap). This
is NOT a trading strategy backtest: it does not simulate trades and is
independent of BacktestParams / run_backtest / the smc_v algo version scheme
— it only tunes the FVG detector's own parameters.

Usage:
    uv run backtest/fvg_width_sweep.py --config config/backtest/fvg_width_default.json
    uv run backtest/fvg_width_sweep.py --codes US.SOXL --tfs 5m 15m --top 10
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # prevent garbled output on Windows cp1252 consoles

from feeds.fetcher import fetch_klines
from strategy.smc.fvg import (
    detect_fvg, is_displacement_candle, compute_volume_profile, fvg_overlaps_lvn,
)

_RESULTS_DIR    = pathlib.Path(__file__).parent / "results"
_DEFAULT_CONFIG = pathlib.Path(__file__).parent.parent / "config" / "backtest" / "fvg_width_default.json"

# Sweep param name -> equivalent BacktestParams field name (backtest/engine.py),
# for transplanting a winning combo into a real trading-backtest config.
_BACKTEST_PARAM_NAMES = {
    "min_gap_pct":           "fvg_min_width_pct",
    "require_displacement":  "displacement_required",
    "atr_mult":              "displacement_atr_mult",
    "body_ratio_min":        "displacement_body_ratio",
    "lookback":              "displacement_lookback",
    "require_lvn_overlap":   "require_lvn_overlap",
    "lvn_threshold":         "lvn_threshold",
}

_DISPLACEMENT_PARAMS = ("atr_mult", "body_ratio_min", "lookback")
_LVN_PARAMS          = ("lvn_threshold",)
_LVN_N_BINS          = 100  # bins for compute_volume_profile, matches engine.py's default


# ── Run-level configuration ───────────────────────────────────────────────────

@dataclass
class SweepConfig:
    codes: list[str] = field(default_factory=lambda: ["US.SOXL"])
    tfs:   list[str] = field(default_factory=lambda: ["5m", "15m", "60m"])
    start: str = "2025-05-22"
    end:   str = "2026-05-22"
    top_n: int = 15


def _load_json_config(path: pathlib.Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _config_from_json(path: pathlib.Path) -> tuple[SweepConfig, dict]:
    raw = _load_json_config(path)
    cfg = SweepConfig(
        codes=raw.get("codes", ["US.SOXL"]),
        tfs=raw.get("tfs",     ["5m", "15m", "60m"]),
        start=raw.get("start", "2025-05-22"),
        end=raw.get("end",     "2026-05-22"),
        top_n=raw.get("top_n", 15),
    )
    grid = raw.get("param_grid")
    if grid is None:
        raise ValueError(f"{path} missing 'param_grid'")
    return cfg, grid


# ── Grid helpers ──────────────────────────────────────────────────────────────

def build_combo_list(grid: dict) -> list[dict]:
    """Expand a parameter grid into a deduplicated list of combo dicts.

    When a combo's require_displacement / require_lvn_overlap is False, the
    corresponding param-only fields have no effect on the result, so all
    their variants are collapsed into a single combo instead of emitting one
    per value (same dedup idiom as run.py:build_param_list). The two dedup
    rules are independent of each other.
    """
    keys   = list(grid.keys())
    values = list(grid.values())
    seen:   set[str] = set()
    combos: list[dict] = []

    for combo in itertools.product(*values):
        d = dict(zip(keys, combo))
        if not d.get("require_displacement", False):
            for k in _DISPLACEMENT_PARAMS:
                d.pop(k, None)
        if not d.get("require_lvn_overlap", False):
            for k in _LVN_PARAMS:
                d.pop(k, None)
        key = json.dumps(d, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        combos.append(d)
    return combos


# ── LVN (volume profile) ──────────────────────────────────────────────────────

def build_daily_lvn_profiles(klines: pd.DataFrame, n_bins: int = _LVN_N_BINS) -> dict[str, tuple]:
    """Map each trading date -> volume profile of the PRECEDING trading date.

    One profile per calendar day, shared by every gap detected that day —
    much cheaper than recomputing a rolling window per gap, and avoids mixing
    together price levels from days the price has since drifted away from.
    The first date in the frame has no preceding day and is left unmapped.
    """
    dates = klines["time_key"].astype(str).str.slice(0, 10)
    unique_dates = dates.unique()
    profiles: dict[str, tuple] = {}
    for i in range(1, len(unique_dates)):
        prev_date, cur_date = unique_dates[i - 1], unique_dates[i]
        day_bars = klines[dates.values == prev_date]
        profiles[cur_date] = compute_volume_profile(day_bars, n_bins=n_bins)
    return profiles


def gap_overlaps_lvn(
    klines: pd.DataFrame,
    gap: dict,
    lvn_profiles: dict | None,
    lvn_threshold: float,
) -> bool:
    """Whether gap sits in a low-volume node of its day's preceding-day profile."""
    if not lvn_profiles:
        return False
    date = str(klines.iloc[gap["idx"]]["time_key"])[:10]
    edges, bin_vols = lvn_profiles.get(date, (None, None))
    return fvg_overlaps_lvn(gap, edges, bin_vols, lvn_threshold)


# ── Scoring ───────────────────────────────────────────────────────────────────

def gap_width_pct(gap: dict) -> float:
    """Gap width normalized by its own midpoint price."""
    top, bottom = gap["top"], gap["bottom"]
    mid = (top + bottom) / 2
    return (top - bottom) / mid if mid > 0 else 0.0


def _filter_gaps(
    klines: pd.DataFrame,
    gaps: list[dict],
    combo: dict,
    lvn_profiles: dict | None,
) -> list[dict]:
    """Apply the displacement and LVN-overlap filters to an already-detected gap list."""
    if combo.get("require_displacement", False):
        gaps = [
            g for g in gaps
            if is_displacement_candle(
                klines, g["idx"],
                combo["atr_mult"], combo["body_ratio_min"], combo["lookback"],
            )
        ]
    if combo.get("require_lvn_overlap", False):
        gaps = [g for g in gaps if gap_overlaps_lvn(klines, g, lvn_profiles, combo["lvn_threshold"])]
    return gaps


def gaps_for_combo(
    klines: pd.DataFrame,
    combo: dict,
    lvn_profiles: dict | None = None,
    raw_gaps_cache: dict | None = None,
) -> list[dict]:
    """Detect FVGs for one parameter combo.

    detect_fvg()'s own require_displacement branch hardcodes
    is_displacement_candle()'s defaults, so the displacement filter is
    applied separately here to allow custom atr_mult/body_ratio_min/lookback
    — mirrors the pattern backtest/engine.py already uses (engine.py:712-714).

    raw_gaps_cache (keyed by min_gap_pct) lets callers sweeping many combos
    against the same klines skip re-running detect_fvg for combos that only
    differ in displacement/LVN params, which are applied as a cheap filter
    on top of the same raw gap list.
    """
    mgp = combo["min_gap_pct"]
    if raw_gaps_cache is not None and mgp in raw_gaps_cache:
        gaps = raw_gaps_cache[mgp]
    else:
        gaps = detect_fvg(klines, min_gap_pct=mgp, require_displacement=False)
        if raw_gaps_cache is not None:
            raw_gaps_cache[mgp] = gaps
    return _filter_gaps(klines, gaps, combo, lvn_profiles)


def score_combo(
    klines: pd.DataFrame,
    combo: dict,
    lvn_profiles: dict | None = None,
    raw_gaps_cache: dict | None = None,
) -> dict:
    """Width/count metrics (in % of price) for one parameter combo.

    total_width_pct (sum of all gap widths) is the joint width x count
    objective: it only increases when gaps are wider AND/OR more numerous,
    so ranking by it alone avoids picking params that win on just one axis.
    """
    widths = [gap_width_pct(g) for g in gaps_for_combo(klines, combo, lvn_profiles, raw_gaps_cache)]
    n = len(widths)
    if n == 0:
        return {"n_gaps": 0, "total_width_pct": 0.0, "mean_width_pct": 0.0, "median_width_pct": 0.0}
    return {
        "n_gaps":           n,
        "total_width_pct":  float(np.sum(widths)),
        "mean_width_pct":   float(np.mean(widths)),
        "median_width_pct": float(np.median(widths)),
    }


# ── Run / report ──────────────────────────────────────────────────────────────

def run_sweep(
    codes: list[str],
    tfs: list[str],
    start: str,
    end: str,
    combos: list[dict],
) -> pd.DataFrame:
    """Score every combo against every (code, tf) and return a flat results table."""
    rows: list[dict] = []
    for code in codes:
        for tf in tfs:
            klines       = fetch_klines(code=code, ktype=tf, start=start, end=end)
            lvn_profiles = build_daily_lvn_profiles(klines)
            raw_cache: dict[float, list[dict]] = {}
            for combo in tqdm(combos, desc=f"{code} {tf}", ncols=80):
                metrics = score_combo(klines, combo, lvn_profiles, raw_cache)
                rows.append({"code": code, "tf": tf, **combo, **metrics})
    return pd.DataFrame(rows)


def print_top_n(df: pd.DataFrame, top_n: int) -> None:
    param_cols = [c for c in _BACKTEST_PARAM_NAMES if c in df.columns]
    for (code, tf), group in df.groupby(["code", "tf"]):
        ranked = group.sort_values("total_width_pct", ascending=False).head(top_n)
        print(f"\n-- Top {top_n} [{code} {tf}]  (ranked by total_width_pct) --\n")
        for _, row in ranked.iterrows():
            params = {k: row[k] for k in param_cols}
            print(
                f"  n_gaps={row['n_gaps']:3.0f}  total={row['total_width_pct']:.4f}  "
                f"mean={row['mean_width_pct']:.4f}  median={row['median_width_pct']:.4f}  "
                f"params={params}"
            )


def print_translation_table() -> None:
    print("\nsweep param -> BacktestParams field (use this to transplant a winning combo "
          "into a real backtest config):")
    for k, v in _BACKTEST_PARAM_NAMES.items():
        print(f"  {k:<22} -> {v}")


def main() -> None:
    """CLI entry point for the FVG width/count parameter sweep."""
    ap = argparse.ArgumentParser(description="FVG width/count parameter sweep")
    ap.add_argument("--config", default=None, metavar="PATH",
                     help=f"JSON config file (default: {_DEFAULT_CONFIG})")
    ap.add_argument("--codes", nargs="+", default=None, help="Stock codes (overrides config)")
    ap.add_argument("--tfs",   nargs="+", default=None, help="Timeframes (overrides config)")
    ap.add_argument("--start", default=None, help="YYYY-MM-DD (overrides config)")
    ap.add_argument("--end",   default=None, help="YYYY-MM-DD (overrides config)")
    ap.add_argument("--top",   type=int, default=None, help="Top N per (code, tf) to print (overrides config)")
    args = ap.parse_args()

    config_path = pathlib.Path(args.config) if args.config else _DEFAULT_CONFIG
    cfg, grid = _config_from_json(config_path)

    if args.codes: cfg.codes = args.codes
    if args.tfs:   cfg.tfs   = args.tfs
    if args.start: cfg.start = args.start
    if args.end:   cfg.end   = args.end
    if args.top:   cfg.top_n = args.top

    combos = build_combo_list(grid)
    print(f"Codes:   {cfg.codes}")
    print(f"TFs:     {cfg.tfs}")
    print(f"Range:   {cfg.start} -> {cfg.end}")
    print(f"Combos:  {len(combos)} per (code, tf)")

    df = run_sweep(cfg.codes, cfg.tfs, cfg.start, cfg.end, combos)

    run_tag  = f"{datetime.now().strftime('%Y%m%d_%H%M')}_fvg_width_sweep"
    out_dir  = _RESULTS_DIR / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df)} rows -> {csv_path}")

    print_top_n(df, cfg.top_n)
    print_translation_table()


if __name__ == "__main__":
    main()
