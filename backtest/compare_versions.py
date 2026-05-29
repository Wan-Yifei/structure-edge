"""Version comparison: re-run top-N combos from an old CSV under the current engine.

Takes a result CSV produced by a previous backtest run (any algo version),
selects the top-N parameter combos by Sharpe (with a minimum trade count filter),
re-runs them against the same stock + date range under the current engine, and
produces a side-by-side comparison table showing metric deltas.

Usage:
    uv run backtest/compare_versions.py \\
        --csv backtest/results/20260528_0202_…/results_US_CSCO.csv \\
        --config config/backtest/cross_stock_grid_v2.json

    # Custom filters
    uv run backtest/compare_versions.py \\
        --csv …/results_US_CSCO.csv \\
        --config …/cross_stock_grid_v2.json \\
        --top-n 50 --min-trades 15 --workers 6

Output:
    backtest/results/compare_<old_tag>_vs_<new_tag>_<code>.html
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import shutil
import tempfile

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from backtest.engine import ALGO_VERSION, BacktestParams, run_backtest
from feeds.fetcher import fetch_klines

_RESULTS_DIR = pathlib.Path(__file__).parent / "results"

# Metrics shown in the comparison table (in order)
_METRICS = ["n_trades", "win_rate", "profit_factor", "sharpe", "sortino",
            "total_r", "avg_r", "max_drawdown_r"]

# Params columns to reconstruct BacktestParams from a CSV row
_PARAM_COLS = [
    "trend_tf", "entry_tf", "swing_lookback", "bos_count",
    "fvg_min_width_pct", "fvg_entry_depth_pct", "fvg_max_age_bars",
    "displacement_required", "displacement_atr_mult", "displacement_body_ratio",
    "displacement_lookback", "require_ltf_confirmation",
    "require_lvn_overlap", "lvn_threshold",
    "sl_buffer_pct", "max_sl_pct", "min_rr",
    "htf_window_bars", "allow_short", "intraday_only", "kd_sl_fallback",
    "htf_trend_methods", "htf_trend_params",
]

# Direction: +1 = higher is better, -1 = lower is better
_METRIC_DIRECTION = {
    "n_trades":       +1,
    "win_rate":       +1,
    "profit_factor":  +1,
    "sharpe":         +1,
    "sortino":        +1,
    "total_r":        +1,
    "avg_r":          +1,
    "max_drawdown_r": -1,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _row_to_params(row: pd.Series) -> BacktestParams:
    """Reconstruct a BacktestParams from a CSV row."""
    d: dict = {}
    for col in _PARAM_COLS:
        if col not in row.index:
            continue
        val = row[col]
        if col == "htf_trend_methods":
            val = tuple(json.loads(val)) if isinstance(val, str) else val
        elif col == "htf_trend_params":
            val = json.loads(val) if isinstance(val, str) else val
        elif isinstance(val, float) and np.isnan(val):
            continue
        d[col] = val
    return BacktestParams(**d)


# ── Worker-process globals (initialised once per worker, never pickled) ───────

_W_KLINES: dict = {}   # populated by _worker_init

def _worker_init(parquet_paths: dict) -> None:
    """Load klines from parquet files into the worker process once at startup.

    The main process writes klines to temp parquet files before spawning workers.
    Workers read those files (concurrent reads are safe; no DuckDB lock contention).
    Only the file-path dict (~tiny) is pickled for initargs — not the DataFrames.
    """
    global _W_KLINES
    _W_KLINES = {tf: pd.read_pickle(p) for tf, p in parquet_paths.items()}


def _run_one(params: BacktestParams) -> dict:
    """Re-run one combo using worker-local klines (no pickle of DataFrames)."""
    htf = _W_KLINES[params.trend_tf]
    ltf = _W_KLINES[params.entry_tf]
    result = run_backtest(htf, ltf, params)
    s = result.summary_dict()
    return {m: s.get(m, 0.0) for m in _METRICS}


# ── HTML report ───────────────────────────────────────────────────────────────

def _colour(delta: float, metric: str) -> str:
    """Return CSS colour for a delta cell."""
    if abs(delta) < 1e-6:
        return "#546e7a"
    direction = _METRIC_DIRECTION.get(metric, +1)
    good = (delta * direction) > 0
    return "#26a69a" if good else "#ef5350"


def _fmt(v, metric: str) -> str:
    if metric in ("n_trades",):
        return str(int(round(v)))
    if metric in ("win_rate",):
        return f"{v:.1%}"
    if metric in ("profit_factor", "sharpe", "sortino"):
        return f"{v:.3f}"
    return f"{v:.2f}"


def _fmt_delta(d, metric: str) -> str:
    if metric in ("win_rate",):
        return f"{d:+.1%}"
    if metric in ("n_trades",):
        return f"{int(round(d)):+d}"
    return f"{d:+.3f}"


def _generate_html(old_tag: str, new_tag: str, code: str,
                   old_df: pd.DataFrame, new_rows: list[dict],
                   top_df: pd.DataFrame) -> str:
    """Build a self-contained HTML comparison report."""

    new_df = pd.DataFrame(new_rows)

    # Summary stats
    summary_rows = []
    for m in _METRICS:
        if m not in old_df.columns or m not in new_df.columns:
            continue
        o_vals = old_df[m].values
        n_vals = new_df[m].values
        deltas = n_vals - o_vals
        d = _METRIC_DIRECTION.get(m, +1)
        improved = int((deltas * d > 0).sum())
        worsened = int((deltas * d < 0).sum())
        unchanged = len(deltas) - improved - worsened
        avg_delta = float(np.mean(deltas))
        summary_rows.append({
            "metric": m,
            "old_mean": float(np.mean(o_vals)),
            "new_mean": float(np.mean(n_vals)),
            "avg_delta": avg_delta,
            "improved": improved,
            "worsened": worsened,
            "unchanged": unchanged,
        })

    summary_html = "<table><thead><tr>"
    for h in ["Metric", f"{old_tag} mean", f"{new_tag} mean", "Avg Δ",
              "▲ Better", "▼ Worse", "= Same"]:
        summary_html += f"<th>{h}</th>"
    summary_html += "</tr></thead><tbody>"
    for r in summary_rows:
        m = r["metric"]
        col = _colour(r["avg_delta"], m)
        summary_html += "<tr>"
        summary_html += f'<td class="param">{m}</td>'
        summary_html += f'<td class="num">{_fmt(r["old_mean"], m)}</td>'
        summary_html += f'<td class="num">{_fmt(r["new_mean"], m)}</td>'
        summary_html += (
            f'<td class="num" style="color:{col};font-weight:bold">'
            f'{_fmt_delta(r["avg_delta"], m)}</td>'
        )
        summary_html += f'<td class="num" style="color:#26a69a">{r["improved"]}</td>'
        summary_html += f'<td class="num" style="color:#ef5350">{r["worsened"]}</td>'
        summary_html += f'<td class="num" style="color:#546e7a">{r["unchanged"]}</td>'
        summary_html += "</tr>"
    summary_html += "</tbody></table>"

    # Per-combo detail table
    param_labels = [c for c in top_df.columns if c in _PARAM_COLS]
    detail_html = "<table><thead><tr>"
    for h in param_labels:
        detail_html += f"<th>{h}</th>"
    for m in _METRICS:
        if m in old_df.columns:
            detail_html += f"<th>{old_tag}<br>{m}</th><th>{new_tag}<br>{m}</th><th>Δ</th>"
    detail_html += "</tr></thead><tbody>"
    for i, (_, old_row) in enumerate(old_df.iterrows()):
        if i >= len(new_rows):
            break
        new_row_d = new_rows[i]
        src_row = top_df.iloc[i]
        detail_html += "<tr>"
        for col in param_labels:
            v = src_row.get(col, "")
            detail_html += f'<td class="param">{v}</td>'
        for m in _METRICS:
            if m not in old_df.columns:
                continue
            ov = float(old_row[m])
            nv = float(new_row_d.get(m, 0))
            delta = nv - ov
            col = _colour(delta, m)
            detail_html += f'<td class="num">{_fmt(ov, m)}</td>'
            detail_html += f'<td class="num">{_fmt(nv, m)}</td>'
            detail_html += (
                f'<td class="num" style="color:{col}">{_fmt_delta(delta, m)}</td>'
            )
        detail_html += "</tr>"
    detail_html += "</tbody></table>"

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Version Comparison: {old_tag} → {new_tag} ({code})</title>
<style>
  body  {{background:#1a1a2e;color:#e0e0e0;font-family:monospace;font-size:13px;padding:20px}}
  h1,h2 {{color:#ffa726;margin-top:1.5em}}
  table {{border-collapse:collapse;width:100%;margin-bottom:2em;overflow-x:auto;display:block}}
  th    {{background:#16213e;color:#b0bec5;padding:6px 10px;text-align:left;border:1px solid #263238;white-space:nowrap}}
  td    {{padding:5px 10px;border:1px solid #263238;white-space:nowrap}}
  .num  {{text-align:right}}
  .param{{color:#42a5f5}}
  tr:hover td {{background:#0f3460}}
  .meta {{color:#546e7a;font-size:12px;margin-bottom:1em}}
</style>
</head><body>
<h1>Version Comparison: {old_tag} → {new_tag}</h1>
<p class="meta">Code: {code} &nbsp;|&nbsp; Generated: {ts} &nbsp;|&nbsp;
Combos compared: {len(new_rows)}</p>

<h2>Summary (per-metric averages)</h2>
{summary_html}

<h2>Per-combo detail</h2>
{detail_html}

</body></html>"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Compare backtest results across algo versions")
    ap.add_argument("--csv",        required=True,
                    help="Path to old results CSV (any algo version)")
    ap.add_argument("--config",     required=True,
                    help="Config JSON used for the original run (supplies dates)")
    ap.add_argument("--top-n",      type=int, default=30,
                    help="Number of top combos to re-run (default 30)")
    ap.add_argument("--min-trades", type=int, default=20,
                    help="Minimum trade count filter (default 20)")
    ap.add_argument("--workers",    type=int, default=6,
                    help="Worker process count (default 6; each worker loads "
                         "klines from cache once — no per-task pickle overhead)")
    ap.add_argument("--out",        default=None,
                    help="Output HTML path (default: auto-named in results/)")
    ap.add_argument("--inspect",    action="store_true",
                    help="After comparison, run fvg_inspect on the #1 combo "
                         "to show per-event rejection reasons under the new version")
    args = ap.parse_args()

    # ── Load old results ──────────────────────────────────────────────────────
    csv_path = pathlib.Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path}")
    old_df_all = pd.read_csv(csv_path)
    code = str(old_df_all["code"].iloc[0])

    # Infer old algo version from the run directory name.
    # The directory looks like "20260528_0202_smc_v2.2_cross_stock_grid_v2_grid".
    # Rejoin all underscore-split tokens and search for the smc_vX.Y pattern.
    import re as _re
    dir_name = csv_path.parent.name
    m = _re.search(r"smc_v[\d.]+", dir_name)
    old_tag = m.group(0) if m else "old"

    # ── Load config for date range ────────────────────────────────────────────
    cfg_path = pathlib.Path(args.config)
    if not cfg_path.exists():
        sys.exit(f"Config not found: {cfg_path}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    start_date = cfg.get("start") or cfg.get("start_date")
    end_date   = cfg.get("end")   or cfg.get("end_date")
    if not start_date or not end_date:
        sys.exit("Config must contain 'start' and 'end' date fields.")

    # ── Select top-N combos ───────────────────────────────────────────────────
    filtered = old_df_all[old_df_all["n_trades"] >= args.min_trades].copy()
    if filtered.empty:
        sys.exit(f"No combos with n_trades >= {args.min_trades} found in CSV.")
    top_df = (filtered
              .sort_values("sharpe", ascending=False)
              .drop_duplicates(subset=_PARAM_COLS, keep="first")
              .head(args.top_n)
              .reset_index(drop=True))
    print(f"[compare] {code} | old={old_tag} new={ALGO_VERSION} | "
          f"combos={len(top_df)} | {start_date}→{end_date}")

    # ── Fetch klines (uses local cache) ──────────────────────────────────────
    all_tfs = set()
    for _, row in top_df.iterrows():
        all_tfs.add(str(row["trend_tf"]))
        all_tfs.add(str(row["entry_tf"]))
    print(f"[compare] Fetching klines for TFs: {sorted(all_tfs)}")
    klines_dict: dict = {}
    for tf in sorted(all_tfs):
        klines_dict[tf] = fetch_klines(code, tf, start_date, end_date)
        print(f"  {tf}: {len(klines_dict[tf])} bars")

    # ── Serialise klines to temp parquet (avoids DuckDB multi-process locking) ─
    # DuckDB rejects concurrent read-write opens from multiple processes.
    # Writing parquet once in the main process lets workers read concurrently
    # without touching DuckDB at all.  Only file paths (strings) are in initargs.
    tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="cmp_klines_"))
    parquet_paths: dict[str, str] = {}
    for tf, df in klines_dict.items():
        p = tmpdir / f"{tf}.pkl"
        df.to_pickle(p)
        parquet_paths[tf] = str(p)
    print(f"[compare] Klines cached to {tmpdir}")

    # ── Re-run top-N combos under current engine ──────────────────────────────
    params_list = [_row_to_params(row) for _, row in top_df.iterrows()]
    new_rows: list[dict | None] = [None] * len(params_list)
    n_workers = min(args.workers, len(params_list))

    ctx = mp.get_context("spawn")
    try:
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx,
                                 initializer=_worker_init,
                                 initargs=(parquet_paths,)) as pool:
            fut_to_idx = {pool.submit(_run_one, p): i
                          for i, p in enumerate(params_list)}
            for fut in tqdm(as_completed(fut_to_idx), total=len(params_list),
                            desc=f"Re-running ({ALGO_VERSION})"):
                idx = fut_to_idx[fut]
                try:
                    new_rows[idx] = fut.result()
                except Exception as e:
                    print(f"  [warn] combo {idx} failed: {e}")
                    new_rows[idx] = {m: 0.0 for m in _METRICS}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # ── Compare & report ──────────────────────────────────────────────────────
    old_metric_df = top_df[_METRICS].copy()

    print("\n── Summary ──────────────────────────────────────────────────────────")
    print(f"{'Metric':<20} {old_tag:>12} {ALGO_VERSION:>12} {'Avg Δ':>10}  "
          f"{'▲':>5} {'▼':>5} {'=':>5}")
    print("-" * 70)
    for m in _METRICS:
        if m not in old_metric_df.columns:
            continue
        o_vals = old_metric_df[m].values.astype(float)
        n_vals = np.array([r[m] for r in new_rows], dtype=float)
        deltas  = n_vals - o_vals
        d = _METRIC_DIRECTION.get(m, +1)
        improved = (deltas * d > 0).sum()
        worsened = (deltas * d < 0).sum()
        unchanged = len(deltas) - improved - worsened
        sym = "▲" if float(np.mean(deltas)) * d > 0 else ("▼" if float(np.mean(deltas)) * d < 0 else "=")
        print(f"{m:<20} {np.mean(o_vals):>12.3f} {np.mean(n_vals):>12.3f} "
              f"{np.mean(deltas):>+10.3f}  "
              f"{improved:>5} {worsened:>5} {unchanged:>5}  {sym}")

    # ── Save HTML ─────────────────────────────────────────────────────────────
    if args.out:
        out_path = pathlib.Path(args.out)
    else:
        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        code_slug = code.replace(".", "_")
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        out_path = _RESULTS_DIR / f"compare_{old_tag}_vs_{ALGO_VERSION}_{code_slug}_{ts}.html"

    html = _generate_html(old_tag, ALGO_VERSION, code,
                          old_metric_df, new_rows, top_df)
    out_path.write_text(html, encoding="utf-8")
    print(f"\n[compare] Report saved → {out_path}")

    # ── Optional: fvg_inspect on the #1 combo under the new version ──────────
    if args.inspect:
        _run_inspect(top_df.iloc[0], code, start_date, end_date, out_path)


def _run_inspect(best_row: "pd.Series", code: str,
                 start_date: str, end_date: str,
                 compare_out: pathlib.Path) -> None:
    """Run fvg_inspect on the best combo and save alongside the comparison report."""
    from backtest.fvg_inspect import run_inspect

    params = _row_to_params(best_row)
    inspect_path = compare_out.parent / (
        compare_out.stem + f"_inspect_{params.trend_tf}_{params.entry_tf}.html"
    )
    print(f"\n[inspect] Running fvg_inspect for best combo "
          f"({params.trend_tf}/{params.entry_tf} lb={params.swing_lookback} "
          f"bos={params.bos_count}) …")
    run_inspect(
        code=code,
        params=params,
        start=start_date,
        end=end_date,
        inspect_start=start_date,
        inspect_end=end_date,
        out_path=inspect_path,
    )
    print(f"[inspect] Saved → {inspect_path}")


if __name__ == "__main__":
    main()
