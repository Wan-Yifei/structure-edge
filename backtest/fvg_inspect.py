"""FVG rejection inspector.

Shows every FVG touch event in a date window and why it was (or was not)
traded — useful for comparing the engine's decisions against manual chart
reading.

Usage examples:

  # Best combo from a results CSV, inspect one week
  uv run backtest/inspect.py \\
      --from-csv backtest/results/.../results_US_SNDK.csv \\
      --code US.SNDK --start 2025-05-22 --end 2026-05-22 \\
      --inspect-start 2025-11-03 --inspect-end 2025-11-07

  # Manual params
  uv run backtest/inspect.py \\
      --code US.SNDK --start 2025-10-01 --end 2026-05-22 \\
      --inspect-start 2025-11-03 --inspect-end 2025-11-07 \\
      --trend-tf 15m --entry-tf 3m \\
      --htf-trend-methods bos_choch kd \\
      --kd-fast 15 --kd-slow 60 --kd-window 10

Output: HTML table saved next to the results CSV (or to --out-dir).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from datetime import datetime

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import pandas as pd

from feeds.fetcher import fetch_klines
from backtest.engine import ALGO_VERSION, BacktestParams, run_backtest


# ── Outcome display config ────────────────────────────────────────────────────

_OUTCOME_META = {
    "entered":             {"label": "Entered",              "color": "#2ecc71"},
    "depth_never_reached": {"label": "Depth not reached",    "color": "#95a5a6"},
    "ltf_confirmation":    {"label": "LTF confirm failed",   "color": "#e67e22"},
    "lvn_filter":          {"label": "LVN filter",           "color": "#9b59b6"},
    "displacement_filter": {"label": "Displacement filter",  "color": "#8e44ad"},
    "no_sl_tp":            {"label": "No SL/TP swing",       "color": "#e74c3c"},
    "max_sl_pct":          {"label": "SL too wide",          "color": "#c0392b"},
    "min_rr":              {"label": "RR too low",           "color": "#e74c3c"},
}


def _build_params_from_args(args: argparse.Namespace) -> BacktestParams:
    methods = tuple(args.htf_trend_methods) if args.htf_trend_methods else ("bos_choch",)
    trend_params: dict = {}
    if "kd" in methods:
        trend_params = {
            "kd_fast":           args.kd_fast,
            "kd_slow":           args.kd_slow,
            "kd_window":         args.kd_window,
            "kd_flat_threshold": args.kd_flat_threshold,
        }
    return BacktestParams(
        trend_tf                = args.trend_tf,
        entry_tf                = args.entry_tf,
        htf_window_bars         = args.htf_window_bars,
        swing_lookback          = args.swing_lookback,
        bos_count               = args.bos_count,
        fvg_min_width_pct       = args.fvg_min_width_pct,
        fvg_entry_depth_pct     = args.fvg_entry_depth_pct,
        fvg_max_age_bars        = args.fvg_max_age_bars,
        displacement_required   = args.displacement_required,
        displacement_atr_mult   = args.displacement_atr_mult,
        displacement_body_ratio = args.displacement_body_ratio,
        require_ltf_confirmation= args.require_ltf_confirmation,
        require_lvn_overlap     = args.require_lvn_overlap,
        lvn_threshold           = args.lvn_threshold,
        sl_buffer_pct           = args.sl_buffer_pct,
        max_sl_pct              = args.max_sl_pct,
        min_rr                  = args.min_rr,
        htf_trend_methods       = methods,
        htf_trend_params        = trend_params,
    )


def _render_html(
    events: list[dict],
    code: str,
    params: BacktestParams,
    inspect_start: str,
    inspect_end: str,
    out_path: pathlib.Path,
) -> None:
    total   = len(events)
    entered = sum(1 for e in events if e["outcome"] == "entered")

    rows_html = ""
    for e in events:
        meta    = _OUTCOME_META.get(e["outcome"], {"label": e["outcome"], "color": "#bdc3c7"})
        depth   = f"{e['depth']:.3f}" if e.get("depth") is not None else "—"
        detail  = e.get("detail", "")
        tid     = e.get("trade_id", "")
        tid_html = f'<code style="font-size:11px">{tid}</code>' if tid else ""
        rows_html += f"""
        <tr>
          <td>{e['touch_time']}</td>
          <td>{e['fvg_bottom']:.4f} – {e['fvg_top']:.4f}</td>
          <td>{'▲' if e['direction'] == 'bull' else '▼'} {e['direction']}</td>
          <td>{e.get('depth_time', '—')}</td>
          <td>{depth}</td>
          <td style="color:{meta['color']};font-weight:600">{meta['label']}</td>
          <td style="font-size:11px;color:#888">{detail}</td>
          <td>{tid_html}</td>
        </tr>"""

    counts: dict[str, int] = {}
    for e in events:
        counts[e["outcome"]] = counts.get(e["outcome"], 0) + 1
    sorted_counts = sorted(counts.items(), key=lambda x: -x[1])
    max_count = max(counts.values()) if counts else 1

    # Horizontal bar chart rows
    bar_rows = ""
    for outcome, count in sorted_counts:
        meta  = _OUTCOME_META.get(outcome, {"label": outcome, "color": "#bdc3c7"})
        pct   = count / total * 100 if total else 0
        width = count / max_count * 100
        bar_rows += f"""
        <tr>
          <td style="width:160px;color:{meta['color']};font-weight:600;white-space:nowrap">
            {meta['label']}
          </td>
          <td style="width:100%">
            <div style="background:{meta['color']};height:16px;border-radius:3px;
                        width:{width:.1f}%;min-width:2px"></div>
          </td>
          <td style="width:40px;text-align:right;font-variant-numeric:tabular-nums">
            {count}
          </td>
          <td style="width:48px;text-align:right;color:#666;font-size:12px">
            {pct:.0f}%
          </td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>FVG Inspect — {code} {inspect_start} → {inspect_end}</title>
<style>
  body  {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          background:#111; color:#ddd; margin:0; padding:20px; }}
  h1    {{ color:#f0f0f0; font-size:18px; margin-bottom:4px; }}
  h2    {{ color:#aaa; font-size:13px; font-weight:600; margin:24px 0 10px; text-transform:uppercase;
           letter-spacing:.06em; }}
  .sub  {{ color:#888; font-size:13px; margin-bottom:20px; }}
  table {{ border-collapse:collapse; width:100%; font-size:13px; }}
  .bar-table td {{ padding:4px 8px; border:none; }}
  .bar-table tr:hover td {{ background:transparent; }}
  th    {{ background:#1e1e1e; color:#aaa; text-align:left; padding:8px 10px;
           position:sticky; top:0; border-bottom:1px solid #333; }}
  td    {{ padding:7px 10px; border-bottom:1px solid #222; vertical-align:top; }}
  tr:hover td {{ background:#1a1a1a; }}
  code  {{ background:#222; padding:2px 5px; border-radius:3px; color:#80b0ff; }}
</style>
</head>
<body>
<h1>FVG Rejection Log — {code}</h1>
<div class="sub">
  {inspect_start} → {inspect_end} &nbsp;|&nbsp;
  params: <code>{params.label()}</code> &nbsp;|&nbsp;
  {total} events &nbsp;|&nbsp; {entered} entered &nbsp;|&nbsp;
  {total - entered} filtered
</div>
<h2>Outcome breakdown</h2>
<table class="bar-table" style="max-width:520px;margin-bottom:28px">
  <tbody>{bar_rows}</tbody>
</table>
<table>
  <thead>
    <tr>
      <th>Touch time</th><th>FVG zone</th><th>Direction</th>
      <th>Depth reached</th><th>Depth</th>
      <th>Outcome</th><th>Detail</th><th>Trade ID</th>
    </tr>
  </thead>
  <tbody>{rows_html}</tbody>
</table>
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")
    print(f"[inspect] Report → {out_path}")


def run_inspect(
    code: str,
    params: BacktestParams,
    start: str,
    end: str,
    inspect_start: str,
    inspect_end: str,
    out_path: pathlib.Path,
) -> None:
    print(f"[inspect] Fetching klines: {code}  {params.trend_tf} + {params.entry_tf} …")
    htf = fetch_klines(code, params.trend_tf, start, end)
    ltf = fetch_klines(code, params.entry_tf,  start, end)
    if htf is None or htf.empty or ltf is None or ltf.empty:
        raise RuntimeError(f"No kline data for {code}")

    print(f"[inspect] Running backtest with rejection log ({inspect_start} → {inspect_end}) …")
    log: list[dict] = []
    run_backtest(
        htf, ltf, params,
        rejection_log=log,
        inspect_window=(inspect_start, inspect_end),
    )

    print(f"[inspect] {len(log)} events captured.")
    _render_html(log, code, params, inspect_start, inspect_end, out_path)
    print(f"\nOpen: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="FVG rejection inspector")

    ap.add_argument("--from-csv",       metavar="PATH",
                    help="Pick params from a results CSV (sorted by profit_factor desc)")
    ap.add_argument("--rank",           type=int, default=1)
    ap.add_argument("--min-trades",     type=int, default=5)
    ap.add_argument("--code",           default=None)
    ap.add_argument("--start",          default=None, help="Backtest start (warmup)")
    ap.add_argument("--end",            default=None)
    ap.add_argument("--inspect-start",  required=True,  help="YYYY-MM-DD window start")
    ap.add_argument("--inspect-end",    required=True,  help="YYYY-MM-DD window end")
    ap.add_argument("--out-dir",        default=None)

    # Manual strategy params (used when --from-csv is not provided)
    ap.add_argument("--trend-tf",                 default="15m")
    ap.add_argument("--entry-tf",                 default="3m")
    ap.add_argument("--htf-window-bars",          type=int,   default=20)
    ap.add_argument("--swing-lookback",           type=int,   default=2)
    ap.add_argument("--bos-count",                type=int,   default=1)
    ap.add_argument("--fvg-min-width-pct",        type=float, default=0.001)
    ap.add_argument("--fvg-entry-depth-pct",      type=float, default=0.10)
    ap.add_argument("--fvg-max-age-bars",         type=int,   default=50)
    ap.add_argument("--displacement-required",    action="store_true")
    ap.set_defaults(displacement_required=False)
    ap.add_argument("--displacement-atr-mult",    type=float, default=1.5)
    ap.add_argument("--displacement-body-ratio",  type=float, default=0.5)
    ap.add_argument("--require-ltf-confirmation", action="store_true")
    ap.set_defaults(require_ltf_confirmation=False)
    ap.add_argument("--require-lvn-overlap",      action="store_true")
    ap.set_defaults(require_lvn_overlap=False)
    ap.add_argument("--lvn-threshold",            type=float, default=0.30)
    ap.add_argument("--sl-buffer-pct",            type=float, default=0.003)
    ap.add_argument("--max-sl-pct",               type=float, default=0.010)
    ap.add_argument("--min-rr",                   type=float, default=2.0)
    ap.add_argument("--htf-trend-methods",        nargs="+", default=["bos_choch"])
    ap.add_argument("--kd-fast",                  type=int,   default=15)
    ap.add_argument("--kd-slow",                  type=int,   default=60)
    ap.add_argument("--kd-window",                type=int,   default=10)
    ap.add_argument("--kd-flat-threshold",        type=float, default=0.0)

    args = ap.parse_args()

    if args.from_csv:
        csv_path = pathlib.Path(args.from_csv)
        if not csv_path.exists():
            print(f"ERROR: CSV not found: {csv_path}"); sys.exit(1)
        df = pd.read_csv(csv_path)
        df_ranked = (
            df[df["n_trades"] >= args.min_trades]
            .sort_values(["profit_factor", "total_r"], ascending=[False, False])
            .reset_index(drop=True)
        )
        if args.rank > len(df_ranked):
            print(f"ERROR: only {len(df_ranked)} combos with ≥{args.min_trades} trades")
            sys.exit(1)
        row    = df_ranked.iloc[args.rank - 1]
        params = BacktestParams.from_dict(row.to_dict())
        code   = args.code or str(row.get("code", ""))
        if not code:
            print("ERROR: --code required when CSV has no 'code' column"); sys.exit(1)
        if not args.start or not args.end:
            print("ERROR: --start and --end required with --from-csv"); sys.exit(1)
        start, end = args.start, args.end
        out_dir = pathlib.Path(args.out_dir) if args.out_dir else csv_path.parent
        print(f"[inspect] Rank #{args.rank}: {params.label()}")
    else:
        if not args.code or not args.start or not args.end:
            ap.error("--code, --start, --end are required without --from-csv")
        params  = _build_params_from_args(args)
        code, start, end = args.code, args.start, args.end
        out_dir = pathlib.Path(args.out_dir) if args.out_dir else pathlib.Path("backtest/results")

    slug     = code.replace(".", "_")
    tf_slug  = f"{params.trend_tf}_{params.entry_tf}"
    filename = f"inspect_{slug}_{tf_slug}_{args.inspect_start}_{args.inspect_end}.html"
    out_path = out_dir / filename

    run_inspect(code, params, start, end, args.inspect_start, args.inspect_end, out_path)
