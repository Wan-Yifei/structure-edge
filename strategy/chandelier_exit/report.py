"""Write grid_results.csv + a short REPORT.md comparing the chandelier exit
grid against the static SL/TP baseline.

This is intentionally NOT the full 8-section REVIEW.md template CLAUDE.md
describes for full grid backtests -- that template is for the smc_v grid
search workflow. This is a smaller, standalone comparison tool.
"""

from __future__ import annotations

import pathlib

import pandas as pd

from backtest.engine import BacktestResult

_METRIC_COLS = [
    "n_trades", "win_rate", "total_r", "avg_r", "profit_factor",
    "max_drawdown_r", "noise_stopout_rate",
]


def _fmt_row(label: str, d: dict) -> str:
    vals = " | ".join(str(d.get(col, "-")) for col in _METRIC_COLS)
    return f"| {label} | {vals} |"


def write_outputs(
    grid_df: pd.DataFrame, baseline: BacktestResult, run_meta: dict,
    out_dir: pathlib.Path, rank_by: str, top_n: int = 5,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    if grid_df.empty:
        (out_dir / "REPORT.md").write_text(
            "# Chandelier Exit — no results\n\n"
            "The grid produced zero valid combos (every entry skipped, likely "
            "insufficient ATR warmup for all tested periods given the date range).\n",
            encoding="utf-8",
        )
        return

    sorted_df = grid_df.sort_values(rank_by, ascending=False).reset_index(drop=True)
    sorted_df.to_csv(out_dir / "grid_results.csv", index=False)

    base = baseline.summary_dict()
    base["noise_stopout_rate"] = "n/a"   # baseline has no noise-stopout concept
    best = sorted_df.iloc[0].to_dict()

    lines: list[str] = []
    lines.append("# Chandelier Exit vs. Static SL/TP\n")
    lines.append("## Run")
    lines.append(f"- code: {run_meta['code']}")
    lines.append(f"- tf_pair: {run_meta['tf_pair']}")
    lines.append(f"- range: {run_meta['start']} .. {run_meta['end']}")
    lines.append(f"- entry params source: {run_meta['entry_params_source']}")
    lines.append(f"- n_entries (from engine): {run_meta['n_entries']}")
    lines.append(f"- grid: atr_periods={run_meta['atr_periods']}  multipliers={run_meta['multipliers']}")
    lines.append(f"- rank_by: {rank_by}\n")

    lines.append("## Baseline vs. best chandelier combo")
    lines.append("| variant | " + " | ".join(_METRIC_COLS) + " |")
    lines.append("|---|" + "---|" * len(_METRIC_COLS))
    lines.append(_fmt_row("static SL/TP (baseline)", base))
    lines.append(_fmt_row(f"chandelier period={best['atr_period']} mult={best['multiplier']}", best))
    lines.append("")

    lines.append(f"## Top {top_n} chandelier combos (by {rank_by})")
    lines.append("| period | mult | " + " | ".join(_METRIC_COLS) + " |")
    lines.append("|---|---|" + "---|" * len(_METRIC_COLS))
    for _, r in sorted_df.head(top_n).iterrows():
        vals = " | ".join(str(r.get(c, "-")) for c in _METRIC_COLS)
        lines.append(f"| {r['atr_period']} | {r['multiplier']} | {vals} |")
    lines.append("")

    lines.append("## Noise stop-out rate — caution")
    lines.append(
        "`noise_stopout_rate` is informational only, NOT used for ranking: the "
        "fraction of trades where the chandelier stop exited at less than the "
        "original TP would have paid, but price still reached that TP afterward. "
        "A high rate on the WINNING combo may simply mean its stop is looser / "
        "less active rather than genuinely better -- cross-check against "
        "`max_drawdown_r` before trusting a wide-multiplier winner.\n"
    )

    lines.append("## Caveats")
    lines.append("- Single stock / single date range / single entry-params combo per run.")
    lines.append("- Exit fills at the stop price exactly (no gap-through modeling), matching "
                  "backtest/engine.py's own fixed-SL convention -- optimistic vs. real fills.")
    lines.append("- R-multiple denominator is the ORIGINAL engine trade's SL distance for both "
                  "baseline and chandelier rows, so total_r/avg_r are directly comparable.")
    lines.append(f"- n_skipped_warmup per combo is in grid_results.csv -- entries before a "
                  f"period's ATR warmup are excluded from that combo's n_trades.")
    lines.append("- Small sample sizes (few dozen trades) make total_r differences between "
                  "top combos easy to overfit to; treat this as a starting hypothesis, not proof.")

    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
