"""Single-stock random-search aggregator for the session_vp strategy.

Counterpart to backtest/aggregate_random.py, which is hardcoded to the SMC
strategy's parameter names (_PARAM_COLS/_GRID_PARAMS reference
htf_trend_methods, fvg_*, etc.) and joins results across multiple stock
codes -- confirmed by reading it directly, not assumed. Neither applies
here: session_vp is SOXL-only by construction (can't be shorted, so there
is no other symbol to cross-join against), so this aggregator ranks a
single stock's random-search results directly instead of joining per-combo
rows across codes.

Usage:
    uv run backtest/aggregate_random_session_vp.py --run-dir backtest/results/<ts>_svp_random/
    uv run backtest/aggregate_random_session_vp.py --run-dir <dir> --top-n 30 --out-config session_vp_grid_v1.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent

# SessionVPParams fields that are candidates for grid narrowing.
# tradeable_sessions is deliberately excluded: it's a segmentation dimension
# (one session per combo, per the approved plan), not something to narrow.
_GRID_PARAMS = [
    "warmup_minutes", "va_pct", "n_bins",
    "rsi_period", "rsi_threshold",
    "max_bars", "min_val_poc_dist_pct",
]

_SORT_PRIMARY   = "profit_factor"
_SORT_SECONDARY = "total_r"


def load_run_dir(run_dir: pathlib.Path) -> pd.DataFrame:
    csvs = sorted(run_dir.glob("results_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No results_*.csv files found in {run_dir}")
    frames = [pd.read_csv(csv) for csv in csvs]
    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(df)} combos from {len(csvs)} file(s)")
    return df


def rank(df: pd.DataFrame, min_trades: int) -> pd.DataFrame:
    """Filter by min_trades and sort by profit_factor / total_r, per-session."""
    ranked = (
        df[df["n_trades"] >= min_trades]
        .sort_values([_SORT_PRIMARY, _SORT_SECONDARY], ascending=[False, False])
        .reset_index(drop=True)
    )
    ranked.index += 1
    n_excl = len(df) - len(ranked)
    print(f"Valid combos (n_trades >= {min_trades}): {len(ranked)}  ({n_excl} excluded)")
    return ranked


def _to_native(val) -> object:
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if isinstance(val, float) and val == int(val):
        return int(val)
    return val


def analyze_param_freq(
    ranked: pd.DataFrame, top_n: int, min_freq: float,
) -> tuple[pd.DataFrame, dict]:
    """Compute parameter value frequencies in the top-N combos, PER SESSION.

    Pooling all 4 sessions' top-N together before narrowing would let
    whichever session happens to produce the highest profit factor dominate
    the frequency counts, silently overriding the other 3 sessions' own
    optima -- the opposite of the approved plan's "backtested/reported
    separately, not pooled" requirement. So each tradeable_sessions group is
    ranked and frequency-analysed independently (its own top-N, its own
    min_freq cut), and a value is kept in the final narrowed grid if ANY
    session's group wants it -- the union, not the intersection, since grid
    search still explores every (param x session) combination afterward and
    can freely reject a param value for sessions where it doesn't apply.
    """
    freq_rows, narrowed = [], {}

    if "tradeable_sessions" not in ranked.columns:
        groups = [("(all)", ranked)]
    else:
        groups = list(ranked.groupby("tradeable_sessions", sort=False))

    for session_label, grp in groups:
        top = grp.sort_values([_SORT_PRIMARY, _SORT_SECONDARY], ascending=[False, False]).head(top_n)
        total = len(top)
        if total == 0:
            continue
        for param in _GRID_PARAMS:
            if param not in top.columns:
                continue
            counts = top[param].value_counts().sort_index()
            for val, cnt in counts.items():
                frac = cnt / total
                freq_rows.append({
                    "session": session_label, "parameter": param, "value": str(val),
                    "count": int(cnt), "freq_%": round(frac * 100, 1),
                    "keep": frac >= min_freq,
                })
                if frac >= min_freq:
                    narrowed.setdefault(param, set()).add(_to_native(val))

    narrowed_sorted = {}
    for param, vals in narrowed.items():
        try:
            narrowed_sorted[param] = sorted(vals)
        except TypeError:
            narrowed_sorted[param] = sorted(vals, key=str)

    return pd.DataFrame(freq_rows), narrowed_sorted


_CSS = """
<style>
  body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 13px; margin: 32px; color: #222; }
  h1   { font-size: 20px; color: #1a1a2e; }
  h2   { font-size: 15px; color: #16213e; margin-top: 32px; border-bottom: 1px solid #ddd; padding-bottom: 4px; }
  table { border-collapse: collapse; width: 100%; margin-top: 8px; }
  th   { background: #1a1a2e; color: #fff; padding: 5px 8px; text-align: left; font-size: 12px; }
  td   { padding: 4px 8px; border-bottom: 1px solid #eee; }
  tr:hover td { background: #f5f7ff; }
  .keep-yes { color: #1a7a1a; font-weight: bold; }
  .keep-no  { color: #aaa; }
</style>
"""


def _freq_table_html(freq_df: pd.DataFrame) -> str:
    rows_html = ""
    for _, r in freq_df.iterrows():
        cls = "keep-yes" if r["keep"] else "keep-no"
        rows_html += (
            f"<tr><td>{r['session']}</td><td>{r['parameter']}</td><td>{r['value']}</td>"
            f"<td>{r['count']}</td><td class='{cls}'>{r['freq_%']:.1f}%</td>"
            f"<td class='{cls}'>{'Y' if r['keep'] else 'n'}</td></tr>\n"
        )
    return (
        "<table><thead><tr><th>Session</th><th>Parameter</th><th>Value</th>"
        "<th>Count in top-N</th><th>Frequency</th><th>Keep</th></tr></thead>"
        f"<tbody>\n{rows_html}</tbody></table>"
    )


def generate_html(ranked: pd.DataFrame, freq_df: pd.DataFrame, narrowed: dict,
                   run_dir: pathlib.Path, top_n: int, min_freq: float,
                   min_trades: int, out_config: str) -> str:
    display_cols = [c for c in (
        ["tradeable_sessions"] + _GRID_PARAMS +
        ["n_trades", "win_rate", "total_r", "avg_r", "profit_factor", "max_drawdown_r"]
    ) if c in ranked.columns]

    # Per-session top-N tables -- the plan requires the 4 sessions be
    # reported separately, not pooled into one ranking (see analyze_param_freq).
    per_session_html = ""
    if "tradeable_sessions" in ranked.columns:
        for session_label, grp in ranked.groupby("tradeable_sessions", sort=False):
            top = grp.sort_values([_SORT_PRIMARY, _SORT_SECONDARY], ascending=[False, False]).head(top_n)
            per_session_html += f"<h2>Top {top_n} -- session {session_label}</h2>\n"
            per_session_html += top[display_cols].to_html(border=0, index=True)
    else:
        per_session_html = f"<h2>Top {top_n} combos</h2>\n" + ranked.head(top_n)[display_cols].to_html(border=0, index=True)

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>session_vp random search aggregation</title>{_CSS}</head><body>
<h1>Session Value-Area Reversal -- Random Search Aggregation</h1>
<p>Source: {run_dir}</p>
<p>min_trades&gt;={min_trades}, top_n={top_n} (per session), min_freq={min_freq}</p>
{per_session_html}
<h2>Parameter frequency by session (narrowed grid candidates -- union across sessions)</h2>
{_freq_table_html(freq_df)}
<h2>Narrowed param_grid -&gt; {out_config}</h2>
<pre>{json.dumps(narrowed, indent=2)}</pre>
</body></html>"""


def write_narrowed_config(narrowed: dict, out_config: str, src_config: pathlib.Path | None) -> pathlib.Path:
    base: dict = {}
    if src_config and src_config.exists():
        with open(src_config, encoding="utf-8") as f:
            base = json.load(f)
    grid = dict(narrowed)
    # tradeable_sessions is always swept over all 4 sessions in the refined grid
    # too -- narrowing never collapses it, see analyze_param_freq's docstring.
    grid["tradeable_sessions"] = [["premarket"], ["regular"], ["afterhours"], ["overnight"]]
    base["param_grid"] = grid
    out_path = ROOT / "config" / "backtest" / out_config
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(base, f, indent=4, ensure_ascii=False)
        f.write("\n")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate session_vp random search results (single-stock).")
    ap.add_argument("--run-dir", required=True, type=pathlib.Path)
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--min-trades", type=int, default=5)
    ap.add_argument("--min-freq", type=float, default=0.25)
    ap.add_argument("--out-config", type=str, default="session_vp_grid_v1.json")
    ap.add_argument("--src-config", type=pathlib.Path, default=None)
    ap.add_argument("--out", type=pathlib.Path, default=None)
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        print(f"ERROR: {run_dir} is not a directory")
        sys.exit(1)
    out_html = args.out or run_dir / "agg_report.html"

    df = load_run_dir(run_dir)
    ranked = rank(df, min_trades=args.min_trades)
    if ranked.empty:
        print("No valid combos found. Try lowering --min-trades.")
        sys.exit(1)

    freq_df, narrowed = analyze_param_freq(ranked, top_n=args.top_n, min_freq=args.min_freq)
    html = generate_html(ranked, freq_df, narrowed, run_dir, args.top_n, args.min_freq, args.min_trades, args.out_config)
    out_html.write_text(html, encoding="utf-8")
    print(f"Report -> {out_html}")

    out_path = write_narrowed_config(narrowed, args.out_config, args.src_config)
    print(f"Narrowed config -> {out_path}")


if __name__ == "__main__":
    main()
