"""Backtest result visualisation.

Four-panel figure:
  1. Equity curves        — cumulative R for the top-N parameter sets
  2. R-multiple histogram — win/loss distribution for the best set
  3. Grid scatter         — every combination: win-rate × profit-factor
  4. Sensitivity heatmap  — mean profit-factor across fvg_min_width vs min_rr
                            (or the two highest-variance parameters)
"""

from __future__ import annotations

import pathlib
from itertools import cycle

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from backtest.engine import BacktestResult

# ── Colour palette (matches core/chart.py dark theme) ─────────────────────────
_BG     = "#0b1120"
_BG2    = "#131f30"
_FG     = "#cdd6f4"
_GRID   = "#1e2d42"
_GREEN  = "#26a69a"
_RED    = "#ef5350"
_GOLD   = "#f9a825"
_BLUE   = "#5c9cf5"
_PURPLE = "#ce93d8"
_CYAN   = "#80cbc4"
_ACCENT = [_GREEN, _BLUE, _GOLD, _PURPLE, _CYAN, "#ff8a65", "#a5d6a7"]

_SAVE_PATH = pathlib.Path(__file__).parent / "results" / "backtest_viz.png"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _style_ax(ax, title: str = "") -> None:
    ax.set_facecolor(_BG2)
    ax.tick_params(colors=_FG, labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID)
    ax.yaxis.label.set_color(_FG)
    ax.xaxis.label.set_color(_FG)
    ax.grid(color=_GRID, linewidth=0.5, linestyle="--", alpha=0.6)
    if title:
        ax.set_title(title, color=_FG, fontsize=9, fontweight="bold", pad=6)


def _equity_curve(trades) -> np.ndarray:
    r = np.array([t.r_multiple for t in trades], dtype=float)
    return np.concatenate([[0.0], np.cumsum(r)])


# ── Panel 1: equity curves ─────────────────────────────────────────────────────

def _plot_equity_curves(ax, results: list[BacktestResult], top_n: int) -> None:
    _style_ax(ax, f"Equity Curves — top {top_n} by profit factor")
    ax.axhline(0, color=_GRID, linewidth=0.8)

    for color, res in zip(cycle(_ACCENT), results[:top_n]):
        if not res.trades:
            continue
        eq   = _equity_curve(res.trades)
        x    = np.arange(len(eq))
        p    = res.params
        lbl  = (f"{p.trend_tf}/{p.entry_tf}  "
                f"bos{p.bos_count}  w{p.fvg_min_width_pct:.3f}  "
                f"rr{p.min_rr:.1f}  "
                f"PF={res.profit_factor:.2f}")
        ax.plot(x, eq, color=color, linewidth=1.2, label=lbl)
        ax.fill_between(x, 0, eq, color=color, alpha=0.06)

    ax.set_xlabel("Trade #", fontsize=7)
    ax.set_ylabel("Cumulative R", fontsize=7)
    leg = ax.legend(fontsize=6, framealpha=0.3, labelcolor=_FG,
                    loc="upper left", handlelength=1.5)
    leg.get_frame().set_facecolor(_BG2)
    leg.get_frame().set_edgecolor(_GRID)


# ── Panel 2: R-multiple histogram ─────────────────────────────────────────────

def _plot_r_histogram(ax, best: BacktestResult) -> None:
    _style_ax(ax, "R-Multiple Distribution (best run)")
    if not best.trades:
        ax.text(0.5, 0.5, "No trades", color=_FG, ha="center", va="center",
                transform=ax.transAxes, fontsize=9)
        return

    rs = np.array([t.r_multiple for t in best.trades])

    bins = np.linspace(rs.min() - 0.1, rs.max() + 0.1, 30)
    wins  = rs[rs >= 0]
    loses = rs[rs <  0]

    if len(wins):
        ax.hist(wins,  bins=bins, color=_GREEN, alpha=0.75, label=f"Win  ({len(wins)})")
    if len(loses):
        ax.hist(loses, bins=bins, color=_RED,   alpha=0.75, label=f"Loss ({len(loses)})")

    ax.axvline(0,        color=_FG,   linewidth=0.8, linestyle="--")
    ax.axvline(rs.mean(), color=_GOLD, linewidth=1.0, linestyle=":",
               label=f"Mean {rs.mean():+.2f}R")

    ax.set_xlabel("R-multiple", fontsize=7)
    ax.set_ylabel("Count",      fontsize=7)
    leg = ax.legend(fontsize=6, framealpha=0.3, labelcolor=_FG)
    leg.get_frame().set_facecolor(_BG2)
    leg.get_frame().set_edgecolor(_GRID)

    p = best.params
    ax.set_title(
        f"R-Multiple Distribution — {p.trend_tf}/{p.entry_tf}  "
        f"PF={best.profit_factor:.2f}  WR={best.win_rate:.1%}",
        color=_FG, fontsize=9, fontweight="bold", pad=6,
    )


# ── Panel 3: grid scatter — win-rate × profit-factor ──────────────────────────

def _plot_grid_scatter(ax, df: pd.DataFrame) -> None:
    _style_ax(ax, "All Combinations: Win Rate × Profit Factor")

    if df.empty:
        return

    # colour by min_rr, size by n_trades
    rr_vals  = sorted(df["min_rr"].unique())
    cmap     = {r: c for r, c in zip(rr_vals, cycle(_ACCENT))}

    for rr, grp in df.groupby("min_rr"):
        color = cmap[rr]
        sz    = np.clip(grp["n_trades"].values * 6, 20, 200).astype(float)
        ax.scatter(grp["win_rate"], grp["profit_factor"],
                   s=sz, color=color, alpha=0.65, edgecolors=_BG, linewidths=0.4,
                   label=f"RR≥{rr:.1f}")

    # annotate top 5
    top = df.nlargest(5, "profit_factor")
    for _, row in top.iterrows():
        ax.annotate(
            f"PF={row['profit_factor']:.1f}",
            (row["win_rate"], row["profit_factor"]),
            xytext=(4, 4), textcoords="offset points",
            color=_GOLD, fontsize=5.5,
        )

    ax.axhline(1.0, color=_GRID, linewidth=0.8, linestyle="--")
    ax.set_xlabel("Win Rate",      fontsize=7)
    ax.set_ylabel("Profit Factor", fontsize=7)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))

    leg = ax.legend(fontsize=6, framealpha=0.3, labelcolor=_FG,
                    markerscale=0.7, title="min R/R", title_fontsize=6)
    leg.get_frame().set_facecolor(_BG2)
    leg.get_frame().set_edgecolor(_GRID)


# ── Panel 4: sensitivity heatmap ──────────────────────────────────────────────

def _pick_heatmap_axes(df: pd.DataFrame) -> tuple[str, str]:
    """Return the two parameters with the highest variance in profit_factor."""
    candidates = [
        "fvg_min_width_pct", "min_rr", "bos_count",
        "fvg_entry_depth_pct", "sl_buffer_pct", "swing_lookback",
    ]
    variances = {}
    for col in candidates:
        if col not in df.columns:
            continue
        if df[col].nunique() < 2:
            continue
        grp_pf = df.groupby(col)["profit_factor"].mean()
        variances[col] = grp_pf.var()

    top2 = sorted(variances, key=variances.get, reverse=True)[:2]  # type: ignore[arg-type]
    if len(top2) < 2:
        top2 = (candidates[:2] if len(candidates) >= 2 else top2 * 2)
    return top2[0], top2[1]


def _plot_sensitivity_heatmap(ax, df: pd.DataFrame) -> None:
    if df.empty or df["profit_factor"].isna().all():
        _style_ax(ax, "Sensitivity Heatmap")
        return

    row_col, col_col = _pick_heatmap_axes(df)
    pivot = df.pivot_table(
        values="profit_factor",
        index=row_col,
        columns=col_col,
        aggfunc="mean",
    )

    mat  = pivot.values
    vmin = max(0, np.nanmin(mat))
    vmax = np.nanmax(mat)

    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=vmax,
                   origin="lower")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels([f"{v:.3g}" for v in pivot.columns], fontsize=6, color=_FG)
    ax.set_yticklabels([f"{v:.3g}" for v in pivot.index],   fontsize=6, color=_FG)
    ax.set_xlabel(col_col, fontsize=7, color=_FG)
    ax.set_ylabel(row_col, fontsize=7, color=_FG)

    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID)

    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            v = mat[r, c]
            if np.isnan(v):
                continue
            txt_color = "black" if v > (vmin + (vmax - vmin) * 0.6) else _FG
            ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                    fontsize=6.5, color=txt_color)

    cbar = ax.get_figure().colorbar(im, ax=ax, pad=0.02, fraction=0.046)
    cbar.ax.tick_params(colors=_FG, labelsize=6)
    cbar.set_label("Mean Profit Factor", color=_FG, fontsize=7)

    ax.set_title(
        f"Sensitivity: {row_col}  ×  {col_col}\n"
        f"(mean profit factor across all other params)",
        color=_FG, fontsize=8, fontweight="bold", pad=6,
    )


# ── CSV-only panels ───────────────────────────────────────────────────────────

def _plot_top_r_bars(ax, df: pd.DataFrame, top_n: int) -> None:
    """Panel 1 substitute: horizontal bar chart of top-N combos by total_r."""
    _style_ax(ax, f"Top {top_n} Combinations — Total R")

    top = (
        df[df["n_trades"] > 0]
        .sort_values(["profit_factor", "total_r"], ascending=[False, False])
        .head(top_n)
        .iloc[::-1]   # reverse so best is at top
    )
    if top.empty:
        ax.text(0.5, 0.5, "No trades", color=_FG, ha="center", va="center",
                transform=ax.transAxes, fontsize=9)
        return

    labels = [
        f"{row['trend_tf']}/{row['entry_tf']}  "
        f"w{row['fvg_min_width_pct']:.3f}  dp{row['fvg_entry_depth_pct']:.2f}  "
        f"rr{row['min_rr']:.1f}"
        for _, row in top.iterrows()
    ]
    colors = [_GREEN if r >= 0 else _RED for r in top["total_r"]]

    bars = ax.barh(range(len(top)), top["total_r"], color=colors, alpha=0.8,
                   edgecolor=_BG, linewidth=0.4)

    for i, (bar, (_, row)) in enumerate(zip(bars, top.iterrows())):
        x = bar.get_width()
        ax.text(
            x + (0.05 if x >= 0 else -0.05), i,
            f"  T={int(row['n_trades'])}  WR={row['win_rate']:.0%}  PF={row['profit_factor']:.2f}",
            va="center", ha="left" if x >= 0 else "right",
            color=_FG, fontsize=6,
        )

    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.axvline(0, color=_FG, linewidth=0.8)
    ax.set_xlabel("Total R", fontsize=7)


def _plot_tf_winrate_scatter(ax, df: pd.DataFrame) -> None:
    """Panel 2 substitute: n_trades vs total_r scatter coloured by TF pair."""
    _style_ax(ax, "TF Pairs — Trades vs Total R")

    active = df[df["n_trades"] > 0].copy()
    if active.empty:
        ax.text(0.5, 0.5, "No trades", color=_FG, ha="center", va="center",
                transform=ax.transAxes, fontsize=9)
        return

    active["tf_pair"] = active["trend_tf"] + "/" + active["entry_tf"]
    pairs  = sorted(active["tf_pair"].unique())
    colors = {p: c for p, c in zip(pairs, cycle(_ACCENT))}

    for pair, grp in active.groupby("tf_pair"):
        ax.scatter(
            grp["n_trades"], grp["total_r"],
            color=colors[pair], alpha=0.7, s=30,
            edgecolors=_BG, linewidths=0.3,
            label=pair,
        )

    ax.axhline(0, color=_GRID, linewidth=0.8, linestyle="--")
    ax.set_xlabel("# Trades", fontsize=7)
    ax.set_ylabel("Total R",  fontsize=7)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    leg = ax.legend(fontsize=6, framealpha=0.3, labelcolor=_FG,
                    ncol=2, title="TF pair", title_fontsize=6)
    leg.get_frame().set_facecolor(_BG2)
    leg.get_frame().set_edgecolor(_GRID)


# ── Public API ─────────────────────────────────────────────────────────────────

def plot_from_csv(
    csv_path: str | pathlib.Path,
    save_path: str | pathlib.Path | None = None,
    show: bool = False,
    top_n: int = 20,
) -> None:
    """Regenerate a four-panel figure from a saved backtest CSV.

    Panels 1/2 are CSV-compatible replacements (no trade-level data needed).
    Panels 3/4 are identical to the live run.
    """
    csv_path = pathlib.Path(csv_path)
    df = pd.read_csv(csv_path)

    if save_path is None:
        save_path = csv_path.parent / "backtest_viz.png"

    plt.style.use("dark_background")

    fig = plt.figure(figsize=(18, 11), facecolor=_BG)
    fig.suptitle(
        f"SMC Backtest — {csv_path.parent.name}  [{csv_path.name}]",
        color=_FG, fontsize=13, fontweight="bold", y=0.98,
    )

    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35,
                          left=0.06, right=0.97, top=0.93, bottom=0.07)

    _plot_top_r_bars(fig.add_subplot(gs[0, 0]), df, top_n)
    _plot_tf_winrate_scatter(fig.add_subplot(gs[0, 1]), df)
    _plot_grid_scatter(fig.add_subplot(gs[1, 0]), df)
    _plot_sensitivity_heatmap(fig.add_subplot(gs[1, 1]), df)

    if save_path:
        out = pathlib.Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=_BG)
        print(f"Chart saved → {out}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_backtest_results(
    results: list[BacktestResult],
    df_summary: pd.DataFrame,
    top_n: int = 5,
    save_path: str | pathlib.Path | None = _SAVE_PATH,
    show: bool = True,
) -> None:
    """Generate and display / save the four-panel backtest figure.

    Args:
        results:    List of BacktestResult sorted by profit_factor descending.
        df_summary: DataFrame from [r.summary_dict() for r in results].
        top_n:      Number of top runs shown in the equity-curve panel.
        save_path:  Path to save the PNG (None to skip saving).
        show:       Whether to call plt.show().
    """
    plt.style.use("dark_background")

    fig = plt.figure(figsize=(18, 11), facecolor=_BG)
    fig.suptitle(
        "SMC Backtest — Grid Search Results",
        color=_FG, fontsize=13, fontweight="bold", y=0.98,
    )

    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35,
                          left=0.06, right=0.97, top=0.93, bottom=0.07)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # best run first
    ranked = sorted(results, key=lambda r: r.profit_factor, reverse=True)

    _plot_equity_curves(ax1, ranked, top_n)
    _plot_r_histogram(ax2, ranked[0] if ranked else BacktestResult(params=results[0].params))  # type: ignore[arg-type]
    _plot_grid_scatter(ax3, df_summary)
    _plot_sensitivity_heatmap(ax4, df_summary)

    if save_path:
        out = pathlib.Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=_BG)
        print(f"Chart saved → {out}")

    if show:
        plt.show()
    else:
        plt.close(fig)
