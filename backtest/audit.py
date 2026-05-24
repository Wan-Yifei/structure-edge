"""Trade Audit Report Generator.

Produces a self-contained HTML file for a single BacktestParams combo.
Provides a human-inspection window into trade logic:
  · Combo parameters + full statistics (WR, PF, Sharpe, DD, …)
  · Mini equity-curve chart
  · All-trades table with trade IDs
  · Longest consecutive win / loss streaks (with per-trade charts)
  · Top-N largest losses (with per-trade charts)
  · Top-N largest wins  (with per-trade charts)

Usage:
    uv run backtest/audit.py --code US.SNDK --start 2025-05-22 --end 2026-05-22

Or import and call generate_audit() directly.
"""

from __future__ import annotations

import argparse
import base64
import io
import math
import pathlib
import sys
from dataclasses import asdict
from datetime import datetime
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from backtest.engine import BacktestParams, BacktestResult, Trade, run_backtest
from backtest.stats  import sharpe_ratio, sortino_ratio
from core.chart      import BG_BAR, FG, GREEN, RED, GOLD, GRID, UP, DOWN
from core.draw       import draw_candles, draw_fvg, draw_bos_choch
from feeds.fetcher   import fetch_klines
from strategy.smc   import (
    find_swings, detect_bos_choch, detect_fvg, determine_trend,
)
from strategy.smc.kd_trend import compute_kd, kd_trend as _kd_trend

_HTF_CHART_BARS   = 80   # total HTF bars shown, centered on entry bar
_HTF_HALF         = _HTF_CHART_BARS // 2
_LTF_PRE_BARS     = 40   # LTF bars shown before entry
_LTF_POST_BARS    = 20   # LTF bars shown after exit
_TOP_N_LOSSES     = 5
_TOP_N_WINS       = 3
_RESULTS_DIR      = pathlib.Path(__file__).parent / "results"

# ── Streak detection ──────────────────────────────────────────────────────────

def _find_streaks(trades: list[Trade]) -> dict:
    """Return longest win/loss streaks as lists of Trade objects."""
    best_win: list[Trade]  = []
    best_loss: list[Trade] = []
    cur_win:  list[Trade]  = []
    cur_loss: list[Trade]  = []

    for t in trades:
        if t.result == "win":
            cur_win.append(t)
            cur_loss = []
        elif t.result == "loss":
            cur_loss.append(t)
            cur_win = []
        else:  # timeout — resets both streaks
            cur_win  = []
            cur_loss = []
        if len(cur_win)  > len(best_win):
            best_win  = list(cur_win)
        if len(cur_loss) > len(best_loss):
            best_loss = list(cur_loss)

    return {"win": best_win, "loss": best_loss}


# ── Chart helpers ─────────────────────────────────────────────────────────────

def _entry_fvg(
    fvgs: list[dict],
    entry_price: float,
    direction: str,
    entry_bar: int,
) -> list[dict]:
    """Return the single FVG most directly responsible for this trade entry.

    Selects the most recent FVG that (a) matches trade direction, (b) formed at
    or before entry_bar, and (c) contains the entry price. Falls back to the
    most recent same-direction FVG if none contain the price.
    """
    candidates = [
        f for f in fvgs
        if f["direction"] == direction and f["idx"] <= entry_bar
    ]
    if not candidates:
        return []
    containing = [
        f for f in candidates
        if f["bottom"] <= entry_price <= f["top"]
    ]
    if containing:
        return [max(containing, key=lambda f: f["idx"])]
    return [max(candidates, key=lambda f: f["idx"])]


def _style_ax(ax):
    ax.set_facecolor(BG_BAR)
    ax.tick_params(colors=FG, labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#444466")
    ax.grid(axis="y", color=GRID, linewidth=0.4)


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor=BG_BAR, edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _equity_curve_b64(trades: list[Trade]) -> str:
    rs = [t.r_multiple for t in trades]
    cum = np.cumsum(rs)
    xs  = np.arange(1, len(rs) + 1)

    fig, ax = plt.subplots(figsize=(9, 2.5))
    fig.patch.set_facecolor(BG_BAR)
    _style_ax(ax)

    colors = [GREEN if r >= 0 else RED for r in rs]
    ax.bar(xs, rs, color=colors, width=0.6, alpha=0.7, zorder=2)
    ax.plot(xs, cum, color=GOLD, lw=1.5, zorder=3, label="Cumulative R")
    ax.axhline(0, color=GRID, lw=0.8)
    ax.set_xlabel("Trade #", color=FG, fontsize=8)
    ax.set_ylabel("R", color=FG, fontsize=8)
    ax.legend(fontsize=7, facecolor=BG_BAR, labelcolor=FG)
    fig.suptitle("Equity Curve (per-trade R + cumulative)", color=FG, fontsize=9)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _trade_chart_b64(
    trade: Trade,
    htf: pd.DataFrame,
    ltf: pd.DataFrame,
    params: BacktestParams,
) -> str:
    """Two-panel chart: HTF context (left) + LTF entry zoom (right)."""
    htf_times = htf["time_key"].values.astype(str)
    ltf_times = ltf["time_key"].values.astype(str)

    # ── locate bars ──────────────────────────────────────────────────────
    entry_str = str(trade.entry_time)
    exit_str  = str(trade.exit_time)

    # HTF: center on entry bar so you can see context before and after
    htf_pos   = int(np.searchsorted(htf_times, entry_str, side="right")) - 1
    htf_pos   = max(0, min(htf_pos, len(htf) - 1))
    htf_start = max(0, htf_pos - _HTF_HALF)
    htf_end   = min(len(htf), htf_pos + _HTF_HALF + 1)
    htf_slice = htf.iloc[htf_start:htf_end].reset_index(drop=True)
    rel_entry_htf = htf_pos - htf_start  # entry bar position within the slice

    # LTF: use stored bar index
    entry_bar = trade.entry_ltf_bar
    exit_bar  = int(np.searchsorted(ltf_times, exit_str, side="right")) - 1
    exit_bar  = max(entry_bar, min(exit_bar, len(ltf) - 1))
    ltf_start = max(0, entry_bar - _LTF_PRE_BARS)
    ltf_end   = min(len(ltf), exit_bar + _LTF_POST_BARS + 1)
    ltf_slice = ltf.iloc[ltf_start:ltf_end].reset_index(drop=True)
    # local indices relative to the slice
    rel_entry = entry_bar - ltf_start
    rel_exit  = exit_bar  - ltf_start

    # ── reconstruct SMC signals matching the engine's backward window ─────
    # The engine sees 200 HTF bars ending at the current bar; the chart only
    # shows 80 bars centred on entry.  Re-running detect_bos_choch on the
    # narrow chart slice gives different (and often wrong) signals because the
    # trend-setting CHoCH might be 50-150 bars back — outside the visible window.
    # Solution: run BOS detection on the engine's full backward window, then
    # remap signal indices into the chart coordinate system.
    eng_start   = max(0, htf_pos + 1 - params.htf_window_bars)
    htf_eng     = htf.iloc[eng_start : htf_pos + 1].reset_index(drop=True)
    htf_bos_eng = detect_bos_choch(htf_eng, params.swing_lookback)

    # Remap: engine bar k → absolute htf index = eng_start + k
    #        chart  bar m → absolute htf index = htf_start + m
    #        so chart_m = k + (eng_start - htf_start)
    idx_offset = eng_start - htf_start
    n_chart = len(htf_slice)
    htf_bos: list[dict] = []
    for sig in htf_bos_eng:
        s = dict(sig)
        s["idx"]      = sig["idx"] + idx_offset
        s["from_idx"] = sig.get("from_idx", max(0, sig["idx"] - 1)) + idx_offset
        # Only include signals with at least one endpoint visible in chart
        if s["from_idx"] < n_chart and s["idx"] >= 0:
            htf_bos.append(s)

    all_htf_fvgs = detect_fvg(htf_slice, params.fvg_min_width_pct)
    # Show only the FVG directly responsible for this trade entry (not all FVGs)
    htf_fvgs   = _entry_fvg(all_htf_fvgs, trade.entry_price, trade.direction, rel_entry_htf)

    # ── reconstruct LTF BOS/CHoCH (only if ltf_confirmation was used) ────
    ltf_bos: list[dict] = []
    if params.require_ltf_confirmation:
        ltf_bos = detect_bos_choch(ltf_slice, lookback=1)

    # ── draw ──────────────────────────────────────────────────────────────
    fig, (ax_h, ax_l) = plt.subplots(
        1, 2, figsize=(14, 4),
        gridspec_kw={"width_ratios": [1.4, 1]},
    )
    fig.patch.set_facecolor(BG_BAR)
    _style_ax(ax_h)
    _style_ax(ax_l)

    # HTF panel
    labels_h = draw_candles(ax_h, htf_slice)
    draw_fvg(ax_h, htf_slice, htf_fvgs, max_bars=_HTF_CHART_BARS)
    draw_bos_choch(ax_h, htf_slice, htf_bos)
    n_h = len(htf_slice)
    step_h = max(1, n_h // 8)
    ax_h.set_xticks(range(0, n_h, step_h))
    ax_h.set_xticklabels([labels_h[i] for i in range(0, n_h, step_h)],
                         rotation=30, fontsize=6, color=FG)

    # vertical line at entry bar (centered in window)
    dir_label   = "LONG"  if trade.direction == "bull" else "SHORT"
    trend_color = UP if trade.direction == "bull" else DOWN
    ax_h.axvline(rel_entry_htf, color=GOLD, lw=1, linestyle="--", alpha=0.7)

    # entry price + SL/TP horizontal reference lines on HTF
    ax_h.axhline(trade.entry_price, color=GOLD,  lw=0.9, linestyle=":",  alpha=0.7,
                 label=f"Entry {trade.entry_price:.2f}")
    ax_h.axhline(trade.sl,          color=RED,   lw=0.9, linestyle="--", alpha=0.6,
                 label=f"SL {trade.sl:.2f}")
    ax_h.axhline(trade.tp,          color=GREEN, lw=0.9, linestyle="--", alpha=0.6,
                 label=f"TP {trade.tp:.2f}")

    # Engine's trend from the full 200-bar window (same calculation the engine did)
    eng_trend = determine_trend(htf_bos_eng, params.bos_count)

    # Find the last CHoCH that established the current trend direction
    trend_choch_info = ""
    if htf_bos_eng:
        last_choch = None
        for sig in htf_bos_eng:
            if sig["type"] == "CHoCH":
                last_choch = sig
        if last_choch is not None:
            bars_ago = len(htf_eng) - 1 - last_choch["idx"]
            abs_choch = eng_start + last_choch["idx"]
            choch_dir = "bull" if last_choch["direction"] == "bull" else "bear"
            choch_chart = abs_choch - htf_start  # chart coordinate
            if 0 <= choch_chart < n_chart:
                # CHoCH is visible — draw a vertical dashed line
                ax_h.axvline(choch_chart, color=trend_color, lw=0.8, linestyle=":", alpha=0.5)
            trend_choch_info = (
                f"  CHoCH {choch_dir} {bars_ago}bars ago"
                if bars_ago > 0 else ""
            )

    # trend direction badge (top-left)
    trend_arrow = "▲" if trade.direction == "bull" else "▼"
    ax_h.text(0.02, 0.97, f"{trend_arrow} {dir_label}{trend_choch_info}",
              transform=ax_h.transAxes, color=trend_color, fontsize=7,
              fontweight="bold", va="top", ha="left", zorder=10,
              bbox=dict(fc=BG_BAR, ec=trend_color, alpha=0.85, pad=3, boxstyle="round"))

    ax_h.set_title(
        f"HTF {params.trend_tf}  — {dir_label} setup  [engine window: last {min(htf_pos+1, params.htf_window_bars)} bars]",
        color=FG, fontsize=8)
    ax_h.legend(fontsize=6, facecolor=BG_BAR, labelcolor=FG)

    # LTF panel
    labels_l = draw_candles(ax_l, ltf_slice)
    if ltf_bos:
        draw_bos_choch(ax_l, ltf_slice, ltf_bos)
    n_l = len(ltf_slice)

    # SL / TP lines
    ax_l.axhline(trade.sl, color=RED,  lw=1.2, linestyle="--", alpha=0.85,
                 label=f"SL {trade.sl:.2f}")
    ax_l.axhline(trade.tp, color=GREEN, lw=1.2, linestyle="--", alpha=0.85,
                 label=f"TP {trade.tp:.2f}")

    # Entry arrow
    entry_y = trade.entry_price
    arrow_dy = (trade.tp - trade.entry_price) * 0.15
    ax_l.annotate(
        f"  {dir_label} {trade.entry_price:.2f}",
        xy=(rel_entry, entry_y),
        xytext=(rel_entry, entry_y - arrow_dy),
        arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.5),
        color=GOLD, fontsize=7, va="center",
    )

    # Exit marker
    exit_color = GREEN if trade.result == "win" else (RED if trade.result == "loss" else GOLD)
    r_label = f"{trade.r_multiple:+.2f}R ({trade.result})"
    ax_l.scatter([rel_exit], [trade.exit_price], marker="X", s=80,
                 color=exit_color, zorder=5, label=r_label)

    step_l = max(1, n_l // 8)
    ax_l.set_xticks(range(0, n_l, step_l))
    ax_l.set_xticklabels([labels_l[i] for i in range(0, n_l, step_l)],
                         rotation=30, fontsize=6, color=FG)
    ax_l.set_title(
        f"LTF {params.entry_tf}  — {trade.entry_time[:16]}  →  {trade.exit_time[:16]}",
        color=FG, fontsize=8,
    )
    ax_l.legend(fontsize=6, facecolor=BG_BAR, labelcolor=FG)

    fig.suptitle(
        f"Trade {trade.trade_id}  |  {dir_label}  |  {trade.result.upper()}  {trade.r_multiple:+.2f}R",
        color=FG, fontsize=9, fontweight="bold",
    )
    fig.tight_layout()
    return _fig_to_b64(fig)


# ── HTML assembly ─────────────────────────────────────────────────────────────

_CSS = """
body { background:#0d0d1a; color:#c8c8e8; font-family:Consolas,monospace;
       font-size:13px; margin:0; padding:16px; }
h1   { color:#FFA005; border-bottom:1px solid #333366; padding-bottom:6px; }
h2   { color:#8888cc; margin-top:28px; }
h3   { color:#aaaadd; margin-top:16px; }
table{ border-collapse:collapse; width:100%; margin-top:8px; }
th   { background:#1a1a2e; color:#8888cc; padding:5px 8px;
       border:1px solid #333366; text-align:left; }
td   { padding:4px 8px; border:1px solid #222244; }
tr:hover td { background:#1a1a2e; }
tr.win  td  { color:#4caf50; }
tr.loss td  { color:#ef5350; }
tr.timeout td { color:#888899; }
tr.highlight td { outline:1px solid #FFA005; }
.kpi-grid { display:flex; flex-wrap:wrap; gap:12px; margin-top:12px; }
.kpi { background:#12122a; border:1px solid #333366; border-radius:6px;
       padding:10px 18px; min-width:120px; text-align:center; }
.kpi .val { font-size:1.5em; font-weight:bold; color:#FFA005; }
.kpi .lbl { font-size:0.75em; color:#8888cc; margin-top:2px; }
.kpi.good .val { color:#4caf50; }
.kpi.bad  .val { color:#ef5350; }
.kpi.neutral .val { color:#8888ff; }
.trade-card { background:#0e0e22; border:1px solid #333366; border-radius:6px;
              margin:12px 0; padding:12px; }
.trade-card h4 { margin:0 0 8px 0; color:#aaaadd; }
img { max-width:100%; border-radius:4px; }
.param-grid { display:flex; flex-wrap:wrap; gap:6px; margin-top:8px; }
.param-badge { background:#1a1a2e; border:1px solid #333366; border-radius:4px;
               padding:3px 10px; font-size:0.85em; }
.param-badge .k { color:#8888cc; }
.param-badge .v { color:#FFA005; }
.streak-label { display:inline-block; background:#1a1a2e; border:1px solid #FFA005;
                border-radius:12px; padding:2px 10px; font-size:0.85em;
                color:#FFA005; margin-bottom:8px; }
"""


def _kpi(label: str, val: str, kind: str = "neutral") -> str:
    return (f'<div class="kpi {kind}">'
            f'<div class="val">{val}</div>'
            f'<div class="lbl">{label}</div></div>')


def _params_html(params: BacktestParams) -> str:
    skip = {"trend_tf", "entry_tf"}
    badges = "".join(
        f'<span class="param-badge"><span class="k">{k}</span> '
        f'<span class="v">{v}</span></span>'
        for k, v in asdict(params).items() if k not in skip
    )
    return f'<div class="param-grid">{badges}</div>'


def _trades_table(trades: list[Trade], highlight_ids: set[str]) -> str:
    rows = []
    for i, t in enumerate(trades, 1):
        cls = t.result
        hl  = " highlight" if t.trade_id in highlight_ids else ""
        rows.append(
            f'<tr class="{cls}{hl}">'
            f"<td>{i}</td>"
            f"<td><code>{t.trade_id}</code></td>"
            f"<td>{t.entry_time[:16]}</td>"
            f"<td>{'L' if t.direction == 'bull' else 'S'}</td>"
            f"<td>{t.entry_price:.2f}</td>"
            f"<td>{t.sl:.2f}</td>"
            f"<td>{t.tp:.2f}</td>"
            f"<td>{t.exit_price:.2f}</td>"
            f"<td>{t.r_multiple:+.2f}</td>"
            f"<td>{t.result}</td>"
            f"</tr>"
        )
    header = (
        "<tr>"
        "<th>#</th><th>trade_id</th><th>Entry time</th><th>Dir</th>"
        "<th>Entry</th><th>SL</th><th>TP</th><th>Exit</th>"
        "<th>R</th><th>Result</th>"
        "</tr>"
    )
    return f"<table>{header}{''.join(rows)}</table>"


def _kd_at_entry(trade: Trade, htf: pd.DataFrame, params: BacktestParams) -> tuple[float, str | None]:
    """Return (avg_width, trend) from KD indicator at trade entry time."""
    tp     = params.htf_trend_params
    fast   = tp.get("kd_fast", 25)
    slow   = tp.get("kd_slow", 90)
    window = tp.get("kd_window", 10)
    flat_t = tp.get("kd_flat_threshold", 0.0)
    htf_times = htf["time_key"].values.astype(str)
    htf_pos   = int(np.searchsorted(htf_times, str(trade.entry_time), side="right")) - 1
    htf_pos   = max(0, min(htf_pos, len(htf) - 1))
    htf_full  = htf.iloc[: htf_pos + 1].reset_index(drop=True)
    kd        = compute_kd(htf_full, fast=fast, slow=slow)
    avg_w     = float(kd["width"].iloc[-window:].mean()) if len(kd) >= window else float("nan")
    trend     = _kd_trend(htf_full, fast=fast, slow=slow, window=window, flat_threshold=flat_t)
    return avg_w, trend


def _trade_card(trade: Trade, htf: pd.DataFrame, ltf: pd.DataFrame,
                params: BacktestParams, idx: int | None = None,
                kd_cache: dict | None = None) -> str:
    b64 = _trade_chart_b64(trade, htf, ltf, params)
    label = f"#{idx}  " if idx is not None else ""
    r_color = "color:#4caf50" if trade.r_multiple >= 0 else "color:#ef5350"
    kd_html = ""
    if kd_cache is not None:
        meta = kd_cache.get(trade.trade_id)
        if meta:
            avg_w, trend = meta
            if not math.isnan(avg_w):
                trend_label = trend if trend else "flat"
                trend_color = "#4caf50" if trend == "bull" else ("#ef5350" if trend == "bear" else "#888888")
                kd_html = (
                    f'  ·  <span style="color:#aaaadd">KDW&nbsp;{avg_w:+.4f}</span>'
                    f'  <span style="color:{trend_color}">▸&nbsp;{trend_label}</span>'
                )
    return (
        f'<div class="trade-card">'
        f"<h4>{label}<code>{trade.trade_id}</code>  ·  "
        f"{trade.entry_time[:16]}  ·  {trade.direction.upper()}  ·  "
        f'<span style="{r_color}">{trade.r_multiple:+.2f}R  ({trade.result})</span>'
        f"{kd_html}</h4>"
        f'<img src="data:image/png;base64,{b64}" />'
        f"</div>"
    )


# ── Main report generator ─────────────────────────────────────────────────────

def generate_audit(
    code:       str,
    params:     BacktestParams,
    start:      str,
    end:        str,
    out_dir:    pathlib.Path | None = None,
    top_losses: int = _TOP_N_LOSSES,
    top_wins:   int = _TOP_N_WINS,
) -> pathlib.Path:
    """Run the backtest for one combo and write an HTML review report.

    Returns the path to the generated HTML file.
    """
    print(f"[review] Fetching klines: {code}  {params.trend_tf} + {params.entry_tf} …")
    htf = fetch_klines(code, params.trend_tf, start, end)
    ltf = fetch_klines(code, params.entry_tf, start, end)
    if htf is None or ltf is None or htf.empty or ltf.empty:
        raise RuntimeError(f"No kline data for {code}")

    print(f"[review] Running backtest ({params.label()}) …")
    result: BacktestResult = run_backtest(htf, ltf, params)
    trades = result.trades
    if not trades:
        raise RuntimeError("No trades produced — try different parameters or date range")
    print(f"[review] {len(trades)} trades found. Generating charts …")

    # Persist trades so trade_viewer can look them up by ID without manual date entry.
    try:
        from backtest.db import BacktestDB
        with BacktestDB() as _db:
            _db.insert_review_trades(code, params.to_dict(), trades)
    except Exception:
        pass  # DB unavailable (locked, missing) — not fatal for HTML report

    # ── Statistics ─────────────────────────────────────────────────────────
    rs        = [t.r_multiple for t in trades]
    n_trades  = len(trades)
    n_wins    = sum(1 for t in trades if t.result == "win")
    n_losses  = sum(1 for t in trades if t.result == "loss")
    n_timeout = sum(1 for t in trades if t.result == "timeout")
    wr        = n_wins / n_trades
    total_r   = sum(rs)
    avg_r     = total_r / n_trades
    wins_r    = sum(r for r in rs if r > 0)
    loss_r    = sum(-r for r in rs if r < 0)
    pf        = wins_r / loss_r if loss_r else float("inf")
    sharpe    = sharpe_ratio(rs)
    sortino   = sortino_ratio(rs)
    max_dd    = result.max_drawdown_r
    max_loss  = result.max_loss_r

    # ── Special-case trade sets ────────────────────────────────────────────
    streaks       = _find_streaks(trades)
    loss_sorted   = sorted(
        [t for t in trades if t.result == "loss"],
        key=lambda t: t.r_multiple,
    )[:top_losses]
    win_sorted    = sorted(
        [t for t in trades if t.result == "win"],
        key=lambda t: -t.r_multiple,
    )[:top_wins]

    highlight_ids = (
        {t.trade_id for t in streaks["win"]}
        | {t.trade_id for t in streaks["loss"]}
        | {t.trade_id for t in loss_sorted}
        | {t.trade_id for t in win_sorted}
    )

    # ── Equity curve ───────────────────────────────────────────────────────
    eq_b64 = _equity_curve_b64(trades)

    # ── KPI boxes ──────────────────────────────────────────────────────────
    pf_kind      = "good" if pf >= 1.5 else ("bad" if pf < 1.0 else "neutral")
    sharpe_kind  = "good" if sharpe >= 1.0 else ("bad" if sharpe < 0 else "neutral")
    dd_kind      = "good" if max_dd <= 5 else ("bad" if max_dd > 15 else "neutral")
    kpis = "".join([
        _kpi("Trades",        str(n_trades),           "neutral"),
        _kpi("Win Rate",      f"{wr:.1%}",              "good" if wr >= 0.4 else "bad"),
        _kpi("W / L / T",    f"{n_wins}/{n_losses}/{n_timeout}", "neutral"),
        _kpi("Total R",       f"{total_r:+.2f}",        "good" if total_r > 0 else "bad"),
        _kpi("Avg R",         f"{avg_r:+.3f}",          "good" if avg_r > 0 else "bad"),
        _kpi("Profit Factor", f"{pf:.2f}",              pf_kind),
        _kpi("Sharpe",        f"{sharpe:.3f}",          sharpe_kind),
        _kpi("Sortino",       f"{sortino:.3f}",         sharpe_kind),
        _kpi("Max DD",        f"{max_dd:.2f}R",         dd_kind),
        _kpi("Max Loss",      f"{max_loss:.2f}R",       "bad" if max_loss > 3 else "neutral"),
    ])

    # ── Section: streaks ───────────────────────────────────────────────────
    def streak_section(label: str, streak_trades: list[Trade], color: str) -> str:
        if not streak_trades:
            return f"<p style='color:#666688'>No {label.lower()} streak found.</p>"
        ids = ", ".join(f"<code>{t.trade_id}</code>" for t in streak_trades)
        cards = "".join(
            _trade_card(t, htf, ltf, params, i + 1, kd_cache or None)
            for i, t in enumerate(streak_trades)
        )
        return (
            f'<span class="streak-label" style="border-color:{color};color:{color}">'
            f"{len(streak_trades)}-trade streak</span>  {ids}"
            f"{cards}"
        )

    # ── KD cache for featured trades ───────────────────────────────────────
    kd_cache: dict[str, tuple] = {}
    if "kd" in params.htf_trend_methods:
        featured = streaks["win"] + streaks["loss"] + loss_sorted + win_sorted
        for t in featured:
            if t.trade_id not in kd_cache:
                kd_cache[t.trade_id] = _kd_at_entry(t, htf, params)

    # ── Section: top loss / win cards ──────────────────────────────────────
    def ranked_cards(trade_list: list[Trade], offset: int = 0) -> str:
        return "".join(
            _trade_card(t, htf, ltf, params, i + 1 + offset, kd_cache or None)
            for i, t in enumerate(trade_list)
        )

    # ── Assemble HTML ──────────────────────────────────────────────────────
    tf_label = f"{params.trend_tf}/{params.entry_tf}"
    now_str  = datetime.now().strftime("%Y-%m-%d %H:%M")

    body = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>Trade Review — {code}  {tf_label}</title>
<style>{_CSS}</style>
</head>
<body>
<h1>Trade Review — {code} &nbsp; {tf_label} &nbsp;
    <small style="font-size:0.55em;color:#666688">{start} → {end} &nbsp;|&nbsp; {now_str}</small>
</h1>

<!-- ── Parameters ─────────────────────────────────────────────────────── -->
<h2>Parameters</h2>
{_params_html(params)}

<!-- ── Statistics ─────────────────────────────────────────────────────── -->
<h2>Statistics</h2>
<div class="kpi-grid">{kpis}</div>
<br>
<img src="data:image/png;base64,{eq_b64}" style="width:100%;max-width:900px"/>

<!-- ── All Trades ─────────────────────────────────────────────────────── -->
<h2>All Trades &nbsp; <small style="font-size:0.65em;color:#666688">
  (highlighted = featured in sections below)</small></h2>
{_trades_table(trades, highlight_ids)}

<!-- ── Streaks ────────────────────────────────────────────────────────── -->
<h2>Consecutive Streaks</h2>
<h3>Longest Win Streak</h3>
{streak_section("Win", streaks["win"], "#4caf50")}
<h3>Longest Loss Streak</h3>
{streak_section("Loss", streaks["loss"], "#ef5350")}

<!-- ── Top Losses ─────────────────────────────────────────────────────── -->
<h2>Largest Losses (Top {top_losses})</h2>
{ranked_cards(loss_sorted)}

<!-- ── Top Wins ───────────────────────────────────────────────────────── -->
<h2>Best Wins (Top {top_wins})</h2>
{ranked_cards(win_sorted)}

</body>
</html>"""

    # ── Write file ─────────────────────────────────────────────────────────
    if out_dir is None:
        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = _RESULTS_DIR / f"review_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = f"{code.replace('.','_')}_{params.trend_tf}_{params.entry_tf}"
    out_path = out_dir / f"audit_{slug}.html"
    out_path.write_text(body, encoding="utf-8")
    print(f"[review] Report written → {out_path}")
    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_params_from_args(args) -> BacktestParams:
    return BacktestParams(
        trend_tf                 = args.trend_tf,
        entry_tf                 = args.entry_tf,
        htf_window_bars          = args.htf_window_bars,
        swing_lookback           = args.swing_lookback,
        bos_count                = args.bos_count,
        fvg_min_width_pct        = args.fvg_min_width_pct,
        fvg_entry_depth_pct      = args.fvg_entry_depth_pct,
        displacement_required    = args.displacement_required,
        require_ltf_confirmation = args.require_ltf_confirmation,
        sl_buffer_pct            = args.sl_buffer_pct,
        max_sl_pct               = args.max_sl_pct,
        min_rr                   = args.min_rr,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate a trade audit HTML report")
    ap.add_argument("--from-csv", metavar="PATH",
                    help="Pick params from a results CSV (sorted by profit_factor desc)")
    ap.add_argument("--rank",  type=int, default=1,
                    help="Which result to audit when using --from-csv (1 = best, default: 1)")
    ap.add_argument("--min-trades", type=int, default=10,
                    help="Minimum trades filter when ranking (default: 10)")
    ap.add_argument("--code",  default=None, help="e.g. US.SNDK (required without --from-csv)")
    ap.add_argument("--start", default=None, help="YYYY-MM-DD (required without --from-csv)")
    ap.add_argument("--end",   default=None, help="YYYY-MM-DD (required without --from-csv)")
    # Strategy params — best-cluster defaults from SNDK grid search
    ap.add_argument("--trend-tf",                 default="15m")
    ap.add_argument("--entry-tf",                 default="1m")
    ap.add_argument("--htf-window-bars",          type=int,   default=20,
                    help="HTF bars for trend window (20 × 15 m ≈ 5 h, min ~20 for swing detection)")
    ap.add_argument("--swing-lookback",           type=int,   default=2)
    ap.add_argument("--bos-count",                type=int,   default=1)
    ap.add_argument("--fvg-min-width-pct",        type=float, default=0.001)
    ap.add_argument("--fvg-entry-depth-pct",      type=float, default=0.20)
    ap.add_argument("--displacement-required",    action="store_true")
    ap.add_argument("--no-displacement",          dest="displacement_required",
                                                  action="store_false")
    ap.set_defaults(displacement_required=False)
    ap.add_argument("--require-ltf-confirmation", action="store_true")
    ap.add_argument("--no-ltf-confirmation",      dest="require_ltf_confirmation",
                                                  action="store_false")
    ap.set_defaults(require_ltf_confirmation=False)
    ap.add_argument("--sl-buffer-pct",            type=float, default=0.003)
    ap.add_argument("--max-sl-pct",               type=float, default=0.010)
    ap.add_argument("--min-rr",                   type=float, default=2.0)
    ap.add_argument("--top-losses",               type=int,   default=_TOP_N_LOSSES)
    ap.add_argument("--top-wins",                 type=int,   default=_TOP_N_WINS)
    ap.add_argument("--out-dir",                  default=None,
                    help="Output directory (default: backtest/results/review_<timestamp>/)")

    args = ap.parse_args()
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else None

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
            print(f"ERROR: only {len(df_ranked)} combos with ≥{args.min_trades} trades"); sys.exit(1)
        row    = df_ranked.iloc[args.rank - 1]
        params = BacktestParams.from_dict(row.to_dict())
        code   = args.code or str(row.get("code", ""))
        start  = args.start or str(csv_path.stem).split("_")[0]  # fallback
        end    = args.end   or ""
        if not code:
            print("ERROR: --code required when CSV has no 'code' column"); sys.exit(1)
        # derive date range from kline cache if not provided
        if not start or not end:
            print("ERROR: --start and --end required with --from-csv"); sys.exit(1)
        print(f"[audit] Rank #{args.rank}: {params.label()}")
    else:
        if not args.code or not args.start or not args.end:
            ap.error("--code, --start, and --end are required without --from-csv")
        params = _build_params_from_args(args)
        code, start, end = args.code, args.start, args.end

    out_path = generate_audit(
        code       = code,
        params     = params,
        start      = start,
        end        = end,
        out_dir    = out_dir,
        top_losses = args.top_losses,
        top_wins   = args.top_wins,
    )
    print(f"\nOpen: {out_path}")
