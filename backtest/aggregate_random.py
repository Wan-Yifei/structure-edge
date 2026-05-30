"""Cross-stock random search aggregator.

Reads all results_*.csv files from a backtest run directory, joins them by
parameter combination (requires all stocks were run with the same random seed
so each gets the same combos), computes cross-stock aggregate metrics, and
produces:
  - HTML report: ranked combos, per-stock breakdown, parameter frequency table
  - Narrowed param_grid JSON written to config/backtest/<out_config>

Usage:
    uv run backtest/aggregate_random.py --run-dir backtest/results/<timestamp>/
    uv run backtest/aggregate_random.py --run-dir <dir> --top-n 30 --min-freq 0.25
    uv run backtest/aggregate_random.py --run-dir <dir> --out-config cross_stock_grid_v2.json
"""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import sys
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent

# Parameters that define a unique combo (order matters for groupby key tuple)
_PARAM_COLS = [
    "trend_tf", "entry_tf",
    "htf_window_bars", "swing_lookback", "bos_count",
    "fvg_min_width_pct", "fvg_entry_depth_pct", "fvg_max_age_bars",
    "displacement_required", "displacement_atr_mult", "displacement_body_ratio",
    "displacement_lookback",
    "require_ltf_confirmation", "require_ltf_trend_bar",
    "require_lvn_overlap", "lvn_threshold",
    "sl_buffer_pct", "max_sl_pct", "min_rr",
    "allow_short", "intraday_only", "kd_sl_fallback",
    "htf_trend_methods", "htf_trend_params",
]

# Subset of params that are candidates for grid narrowing
# htf_trend_methods is included so the frequency table shows which trend methods dominate;
# htf_trend_params (dicts) is handled separately in write_narrowed_config.
_GRID_PARAMS = [
    "htf_trend_methods",
    "htf_window_bars", "swing_lookback", "bos_count",
    "fvg_min_width_pct", "fvg_entry_depth_pct",
    "displacement_required", "require_ltf_confirmation", "require_ltf_trend_bar",
    "sl_buffer_pct", "max_sl_pct", "min_rr", "kd_sl_fallback",
]

# Primary sort metric and secondary
_SORT_PRIMARY = "avg_sharpe"
_SORT_SECONDARY = "avg_total_r"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_run_dir(run_dir: pathlib.Path) -> tuple[pd.DataFrame, list[str]]:
    """Load all results CSVs from run_dir into a single DataFrame.

    Args:
        run_dir: Directory containing results_*.csv files.

    Returns:
        Tuple of (combined DataFrame with _code column, list of stock codes).
    """
    csvs = sorted(run_dir.glob("results_*.csv"))
    if not csvs:
        # Fall back to per-stock subdirectories (*/results_*.csv)
        csvs = sorted(run_dir.glob("*/results_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No results_*.csv files found in {run_dir} or its subdirs")

    frames, codes = [], []
    for csv in csvs:
        df = pd.read_csv(csv, dtype={"htf_trend_methods": str, "htf_trend_params": str})
        code = str(df["code"].iloc[0]) if "code" in df.columns else csv.stem.replace("results_", "")
        df["_code"] = code
        frames.append(df)
        codes.append(code)

    print(f"Loaded {len(codes)} stocks: {codes}")
    total = sum(len(f) for f in frames)
    print(f"Total rows: {total}  ({total // len(codes)} combos per stock)")
    return pd.concat(frames, ignore_index=True), codes


# ── Aggregation ───────────────────────────────────────────────────────────────

def _remove_kd_keys(params_str: str, keys_to_remove: set) -> str:
    """Remove specified keys from a JSON-encoded htf_trend_params string."""
    try:
        d = ast.literal_eval(params_str) if params_str else {}
        if not isinstance(d, dict):
            return params_str
        d = {k: v for k, v in d.items() if k not in keys_to_remove}
        return json.dumps(d, sort_keys=True)
    except Exception:
        return params_str


def normalize_params(df: pd.DataFrame) -> pd.DataFrame:
    """Remove irrelevant parameter variation before groupby aggregation.

    Rules applied (each only when the relevant column exists):

    1. bos_choch-only: htf_trend_params → '{}' (KD params have no effect).
    2. kd-only in adaptive mode (kd_smooth absent or >0): remove kd_window
       from htf_trend_params (window is unused in segment-based mode).
    3. kd-only: bos_count → 1 (BOS counter unused when method has no bos_choch).

    After normalisation, duplicate (code, params) keys are collapsed, keeping
    the row with the most trades.
    """
    if "htf_trend_methods" not in df.columns or "htf_trend_params" not in df.columns:
        return df
    df = df.copy()
    methods = df["htf_trend_methods"].astype(str).str.strip()
    bos_only = methods == '["bos_choch"]'
    kd_only  = methods == '["kd"]'

    # Rule 1: bos_choch-only → clear KD params
    df.loc[bos_only, "htf_trend_params"] = "{}"

    # Rule 2: kd-only adaptive → strip kd_window (unused in adaptive/segment mode)
    def _is_adaptive(p: str) -> bool:
        try:
            d = ast.literal_eval(p) if p else {}
            return isinstance(d, dict) and int(d.get("kd_smooth", 3)) > 0
        except Exception:
            return True  # default kd_smooth=3 is adaptive

    kd_adaptive = kd_only & df["htf_trend_params"].apply(_is_adaptive)
    df.loc[kd_adaptive, "htf_trend_params"] = df.loc[kd_adaptive, "htf_trend_params"].apply(
        lambda p: _remove_kd_keys(p, {"kd_window"})
    )

    # Rule 3: kd-only → bos_count irrelevant, canonicalise to 1
    if "bos_count" in df.columns:
        df.loc[kd_only, "bos_count"] = 1

    param_cols = [c for c in _PARAM_COLS if c in df.columns]
    key_cols = (["_code"] if "_code" in df.columns else []) + param_cols
    df = (df.sort_values("n_trades", ascending=False)
            .drop_duplicates(subset=key_cols)
            .sort_index())
    return df


def aggregate(df: pd.DataFrame, codes: list[str], min_trades: int) -> pd.DataFrame:
    """Join combos across stocks and compute cross-stock aggregate metrics.

    Args:
        df:          Combined DataFrame from load_run_dir().
        codes:       Stock code list (same order as loaded).
        min_trades:  Drop combos where any stock has fewer than this many trades.

    Returns:
        DataFrame with one row per unique combo, sorted by avg_sharpe descending.
    """
    param_cols = [c for c in _PARAM_COLS if c in df.columns]
    short_codes = [c.replace("US.", "") for c in codes]

    rows = []
    for key, grp in df.groupby(param_cols, sort=False, dropna=False):
        if len(grp) < len(codes):
            continue  # combo missing for some stock (shouldn't happen with fixed seed)
        if (grp["n_trades"] < min_trades).any():
            continue

        row: dict = {}
        # Param values
        if isinstance(key, tuple):
            for col, val in zip(param_cols, key):
                row[col] = val
        else:
            row[param_cols[0]] = key

        # Per-stock metrics
        for _, srow in grp.iterrows():
            sc = srow["_code"].replace("US.", "")
            row[f"sharpe_{sc}"]  = round(float(srow["sharpe"]),  3)
            row[f"total_r_{sc}"] = round(float(srow["total_r"]), 2)
            row[f"n_{sc}"]       = int(srow["n_trades"])

        # Aggregate metrics
        row["avg_sharpe"]        = round(float(grp["sharpe"].mean()),         3)
        row["min_sharpe"]        = round(float(grp["sharpe"].min()),          3)
        row["avg_total_r"]       = round(float(grp["total_r"].mean()),        2)
        row["avg_sortino"]       = round(float(grp["sortino"].mean()),        3)
        row["avg_profit_factor"] = round(float(grp["profit_factor"].mean()),  3)
        row["avg_win_rate"]      = round(float(grp["win_rate"].mean()),       3)
        row["max_dd"]            = round(float(grp["max_drawdown_r"].max()),  2)
        row["total_trades"]      = int(grp["n_trades"].sum())
        rows.append(row)

    if not rows:
        print("WARNING: no combos passed the min_trades filter across all stocks.")
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    result = result.sort_values(
        [_SORT_PRIMARY, _SORT_SECONDARY], ascending=False
    ).reset_index(drop=True)
    result.index += 1  # 1-based rank
    print(f"Valid combos (all stocks ≥ {min_trades} trades): {len(result)}")
    return result


# ── Parameter frequency analysis ─────────────────────────────────────────────

def _to_native(val) -> object:
    """Convert numpy scalar to Python native type for JSON serialisation."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if isinstance(val, float) and val == int(val):
        return int(val) if abs(val) < 1e6 else val
    return val


def analyze_param_freq(
    agg_df: pd.DataFrame,
    top_n: int,
    min_freq: float,
) -> tuple[pd.DataFrame, dict]:
    """Compute parameter value frequencies in top-N combos and suggest narrowed grid.

    Args:
        agg_df:   Aggregated DataFrame from aggregate().
        top_n:    Number of top combos to analyse.
        min_freq: Minimum frequency (0–1) for a value to be kept in narrowed grid.

    Returns:
        Tuple of (frequency DataFrame for report, narrowed param_grid dict).
    """
    top = agg_df.head(top_n)
    freq_rows = []
    narrowed: dict = {}

    for param in _GRID_PARAMS:
        if param not in top.columns:
            continue
        counts = top[param].value_counts().sort_index()
        total  = len(top)
        kept   = []
        for val, cnt in counts.items():
            frac = cnt / total
            freq_rows.append({
                "parameter": param,
                "value":     str(val),
                "count":     int(cnt),
                "freq_%":    round(frac * 100, 1),
                "keep":      frac >= min_freq,
            })
            if frac >= min_freq:
                kept.append(_to_native(val))

        if kept:
            # Keep original type ordering (numeric sort)
            try:
                kept_sorted = sorted(kept)
            except TypeError:
                kept_sorted = sorted(kept, key=str)
            narrowed[param] = kept_sorted

    freq_df = pd.DataFrame(freq_rows)
    return freq_df, narrowed


# ── HTML report ───────────────────────────────────────────────────────────────

_CSS = """
<style>
  body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 13px; margin: 32px; color: #222; }
  h1   { font-size: 20px; color: #1a1a2e; }
  h2   { font-size: 15px; color: #16213e; margin-top: 32px; border-bottom: 1px solid #ddd; padding-bottom: 4px; }
  p    { color: #444; }
  table { border-collapse: collapse; width: 100%; margin-top: 8px; }
  th   { background: #1a1a2e; color: #fff; padding: 5px 8px; text-align: left; font-size: 12px; }
  td   { padding: 4px 8px; border-bottom: 1px solid #eee; }
  tr:hover td { background: #f5f7ff; }
  .keep-yes { color: #1a7a1a; font-weight: bold; }
  .keep-no  { color: #aaa; }
  pre  { background: #f4f4f4; padding: 12px; border-radius: 4px; font-size: 12px; }
  .meta { color: #666; font-size: 12px; margin-bottom: 16px; }
</style>
"""


def _df_to_html(df: pd.DataFrame, float_fmt: str = "{:.3f}") -> str:
    return df.to_html(border=0, classes="", index=True, float_format=lambda x: f"{x:.3f}")


def _freq_table_html(freq_df: pd.DataFrame) -> str:
    """Render frequency table with colour coding."""
    rows_html = ""
    for _, r in freq_df.iterrows():
        cls = "keep-yes" if r["keep"] else "keep-no"
        rows_html += (
            f"<tr><td>{r['parameter']}</td><td>{r['value']}</td>"
            f"<td>{r['count']}</td>"
            f"<td class='{cls}'>{r['freq_%']:.1f}%</td>"
            f"<td class='{cls}'>{'✓' if r['keep'] else '✗'}</td></tr>\n"
        )
    return (
        "<table><thead><tr>"
        "<th>Parameter</th><th>Value</th><th>Count in top-N</th>"
        "<th>Frequency</th><th>Keep</th>"
        "</tr></thead><tbody>\n"
        + rows_html
        + "</tbody></table>"
    )


def generate_html(
    agg_df:       pd.DataFrame,
    freq_df:      pd.DataFrame,
    narrowed:     dict,
    codes:        list[str],
    run_dir:      pathlib.Path,
    top_n:        int,
    min_freq:     float,
    min_trades:   int,
    out_config:   str,
) -> str:
    """Render the full HTML report as a string."""
    short_codes = [c.replace("US.", "") for c in codes]
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Build top-N display table (selected columns)
    display_cols = (
        ["avg_sharpe", "min_sharpe", "avg_total_r", "avg_sortino",
         "avg_profit_factor", "avg_win_rate", "max_dd", "total_trades"]
        + [f"sharpe_{sc}" for sc in short_codes]
        + [f"total_r_{sc}" for sc in short_codes]
        + [p for p in _GRID_PARAMS if p in agg_df.columns]
    )
    display_cols = [c for c in display_cols if c in agg_df.columns]
    top_html = _df_to_html(agg_df.head(top_n)[display_cols])

    freq_html  = _freq_table_html(freq_df)
    config_json = json.dumps({"param_grid": narrowed}, indent=4, ensure_ascii=False)

    reduction = ""
    if not agg_df.empty:
        n_valid = len(agg_df)
        reduction = f"<p class='meta'>{n_valid} combos passed filter (all stocks ≥ {min_trades} trades).</p>"

    return f"""<!DOCTYPE html>
<html lang="zh">
<head><meta charset="utf-8"><title>Cross-Stock Random Aggregation</title>{_CSS}</head>
<body>
<h1>Cross-Stock Random Search Aggregation</h1>
<p class="meta">
  Run: <code>{run_dir.name}</code> &nbsp;|&nbsp;
  Stocks: {', '.join(codes)} &nbsp;|&nbsp;
  Generated: {ts}
</p>
{reduction}

<h2>Top-{top_n} Combos (sorted by avg_sharpe)</h2>
<p class="meta">
  All stocks must have ≥ {min_trades} trades to be included.
  Per-stock sharpe and total_r shown alongside aggregate metrics.
</p>
{top_html}

<h2>Parameter Frequency in Top-{top_n} (min_freq = {min_freq:.0%})</h2>
<p class="meta">
  Values with frequency ≥ {min_freq:.0%} are kept in the narrowed grid (✓).
  Values appearing rarely are dropped (✗).
</p>
{freq_html}

<h2>Suggested Narrowed <code>param_grid</code> → <code>{out_config}</code></h2>
<p class="meta">
  Only parameters with ≥ 2 values are shown (single-value params stay fixed).
</p>
<pre>{config_json}</pre>
</body>
</html>"""


# ── Config writer ─────────────────────────────────────────────────────────────

def _parse_methods_str(s: str) -> list[str]:
    """Convert a CSV methods string like \"('bos_choch', 'kd')\" to [\"bos_choch\", \"kd\"]."""
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (tuple, list)):
            return list(parsed)
    except Exception:
        pass
    return [s]


def _parse_params_str(s: str) -> dict:
    """Convert a CSV params string like \"{'kd_fast': 15}\" to a dict."""
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {}


def write_narrowed_config(
    run_dir:    pathlib.Path,
    narrowed:   dict,
    out_config: str,
    src_config: pathlib.Path | None,
    top_df:     "pd.DataFrame | None" = None,
) -> pathlib.Path:
    """Write the narrowed param_grid into a new config JSON file.

    Copies top-level fields (codes, start, end, workers, top_n, tf_pairs,
    tf_pairs_fast) from src_config if provided, then replaces param_grid.
    Converts htf_trend_methods strings back to list-of-lists, and extracts
    unique htf_trend_params dicts from top_df rows that use KD methods.

    Args:
        run_dir:    Run directory (used to locate source config if not given).
        narrowed:   Narrowed param_grid dict from analyze_param_freq().
        out_config: Output filename (basename only, placed in config/backtest/).
        src_config: Source config to copy top-level fields from.
        top_df:     Top-N aggregated rows; used to extract htf_trend_params.

    Returns:
        Path to the written config file.
    """
    base: dict = {}
    if src_config and src_config.exists():
        with open(src_config) as f:
            base = json.load(f)

    grid = dict(narrowed)

    # Convert htf_trend_methods from repr strings back to list-of-lists
    if "htf_trend_methods" in grid:
        raw_methods = grid["htf_trend_methods"]
        grid["htf_trend_methods"] = [_parse_methods_str(m) for m in raw_methods]

    # Extract unique htf_trend_params dicts from top_df rows that need KD
    if top_df is not None and "htf_trend_params" in top_df.columns:
        kd_methods = [m for m in grid.get("htf_trend_methods", [])
                      if "kd" in m]
        if kd_methods and "htf_trend_methods" in top_df.columns:
            kd_mask = top_df["htf_trend_methods"].apply(
                lambda s: "kd" in str(s)
            )
            kd_rows = top_df[kd_mask]
            seen, unique_params = set(), []
            for raw in kd_rows["htf_trend_params"].dropna():
                d = _parse_params_str(str(raw))
                key = json.dumps(d, sort_keys=True)
                if key not in seen and d:
                    seen.add(key)
                    unique_params.append(d)
            if unique_params:
                grid["htf_trend_params"] = unique_params

    base["param_grid"] = grid
    out_path = ROOT / "config" / "backtest" / out_config
    with open(out_path, "w") as f:
        json.dump(base, f, indent=4, ensure_ascii=False)
        f.write("\n")
    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """Parse args, load results, aggregate, write report and narrowed config."""
    ap = argparse.ArgumentParser(description="Aggregate cross-stock random search results.")
    ap.add_argument("--run-dir",    required=True, type=pathlib.Path,
                    help="Directory containing results_*.csv files")
    ap.add_argument("--top-n",      type=int,   default=30,
                    help="Number of top combos to analyse for narrowing (default: 30)")
    ap.add_argument("--min-trades", type=int,   default=5,
                    help="Exclude combos where any stock has fewer than N trades (default: 5)")
    ap.add_argument("--min-freq",   type=float, default=0.25,
                    help="Min frequency (0–1) for a param value to be kept (default: 0.25)")
    ap.add_argument("--out-config", type=str,   default="cross_stock_grid_v2.json",
                    help="Output config filename in config/backtest/ (default: cross_stock_grid_v2.json)")
    ap.add_argument("--src-config", type=pathlib.Path, default=None,
                    help="Source config to copy top-level fields from (codes/start/end/workers/…)")
    ap.add_argument("--out",        type=pathlib.Path, default=None,
                    help="Output HTML report path (default: <run-dir>/agg_report.html)")
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        print(f"ERROR: {run_dir} is not a directory")
        sys.exit(1)

    out_html = args.out or run_dir / "agg_report.html"

    # Load
    df, codes = load_run_dir(run_dir)

    # Normalize bos_choch-only rows (htf_trend_params irrelevant → collapse duplicates)
    df = normalize_params(df)

    # Aggregate
    agg_df = aggregate(df, codes, min_trades=args.min_trades)
    if agg_df.empty:
        print("No valid combos found. Try lowering --min-trades.")
        sys.exit(1)

    # Frequency analysis
    freq_df, narrowed = analyze_param_freq(agg_df, top_n=args.top_n, min_freq=args.min_freq)

    # Print summary
    print(f"\nTop-5 combos by avg_sharpe:")
    show_cols = ["avg_sharpe", "min_sharpe", "avg_total_r", "total_trades"] + \
                [f"sharpe_{c.replace('US.','')}" for c in codes]
    show_cols = [c for c in show_cols if c in agg_df.columns]
    print(agg_df.head(5)[show_cols].to_string())

    print(f"\nNarrowed param_grid ({args.top_n} top combos, min_freq={args.min_freq:.0%}):")
    for k, v in narrowed.items():
        print(f"  {k}: {v}")

    # Write HTML report
    html = generate_html(
        agg_df, freq_df, narrowed, codes, run_dir,
        top_n=args.top_n, min_freq=args.min_freq, min_trades=args.min_trades,
        out_config=args.out_config,
    )
    out_html.write_text(html, encoding="utf-8")
    print(f"\nReport written: {out_html}")

    # Write narrowed config (pass top_df so htf_trend_params can be extracted)
    top_df  = agg_df.head(args.top_n)
    out_cfg = write_narrowed_config(run_dir, narrowed, args.out_config, args.src_config, top_df)
    print(f"Config written: {out_cfg}")


if __name__ == "__main__":
    main()
