"""FVG width/count sweep visualization.

Reads a results.csv produced by backtest/fvg_width_sweep.py and plots the
n_gaps vs mean_width_pct tradeoff per timeframe, colored by which filter
(displacement / LVN overlap) was active for that combo — the quantity-vs-
quality tradeoff explored interactively during the sweep.

Usage:
    uv run backtest/fvg_width_viz.py backtest/results/<run>/results.csv
    uv run backtest/fvg_width_viz.py backtest/results/<run>/results.csv --show
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib.pyplot as plt
import pandas as pd

# ── Colour palette (matches viz.py / report.py dark theme) ───────────────────
_BG     = "#0b1120"
_BG2    = "#131f30"
_FG     = "#cdd6f4"
_GRID   = "#1e2d42"
_GREEN  = "#26a69a"
_GOLD   = "#f9a825"
_BLUE   = "#5c9cf5"
_PURPLE = "#ce93d8"

_CATEGORY_COLORS = {
    "raw":              _GREEN,   # no filters
    "displacement":     _GOLD,    # displacement filter only
    "lvn":              _BLUE,    # LVN overlap filter only
    "displacement+lvn": _PURPLE,  # both filters
}

_TF_ORDER = ["1m", "3m", "5m", "15m", "30m", "60m", "2h", "3h", "4h", "1d"]


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


def _category(row: pd.Series) -> str:
    disp = bool(row.get("require_displacement", False))
    lvn  = bool(row.get("require_lvn_overlap", False))
    if disp and lvn:
        return "displacement+lvn"
    if disp:
        return "displacement"
    if lvn:
        return "lvn"
    return "raw"


def _plot_tf_panel(ax, df_tf: pd.DataFrame, tf: str) -> None:
    for cat, color in _CATEGORY_COLORS.items():
        sub = df_tf[df_tf["_category"] == cat]
        if sub.empty:
            continue
        ax.scatter(sub["n_gaps"], sub["mean_width_pct"], s=14, alpha=0.6,
                   color=color, label=cat, edgecolors="none")
    ax.set_xscale("log")
    ax.set_xlabel("n_gaps (log)", fontsize=7)
    ax.set_ylabel("mean_width_pct", fontsize=7)
    _style_ax(ax, title=tf)


# ── Public API ─────────────────────────────────────────────────────────────────

def plot_from_csv(
    csv_path: str | pathlib.Path,
    save_path: str | pathlib.Path | None = None,
    show: bool = False,
) -> None:
    """Render the n_gaps vs mean_width_pct tradeoff, one panel per timeframe."""
    csv_path = pathlib.Path(csv_path)
    df = pd.read_csv(csv_path)
    df["_category"] = df.apply(_category, axis=1)

    if save_path is None:
        save_path = csv_path.parent / "fvg_width_viz.png"

    tfs = sorted(df["tf"].unique(), key=lambda t: _TF_ORDER.index(t) if t in _TF_ORDER else 99)
    ncols = min(3, len(tfs))
    nrows = -(-len(tfs) // ncols)  # ceil division

    plt.style.use("dark_background")
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), facecolor=_BG)
    axes = list(axes.flat) if len(tfs) > 1 else [axes]

    for ax, tf in zip(axes, tfs):
        _plot_tf_panel(ax, df[df["tf"] == tf], tf)
    for ax in axes[len(tfs):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="upper center", ncol=len(_CATEGORY_COLORS),
                      fontsize=8, labelcolor=_FG, framealpha=0.3, bbox_to_anchor=(0.5, 1.04))
    leg.get_frame().set_facecolor(_BG2)
    leg.get_frame().set_edgecolor(_GRID)

    fig.suptitle(f"FVG width/count tradeoff — {csv_path.parent.name}",
                 color=_FG, fontsize=12, fontweight="bold", y=1.08)
    fig.tight_layout()

    out = pathlib.Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=_BG)
    print(f"Chart saved -> {out}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize an FVG width/count sweep results CSV")
    ap.add_argument("csv", help="Path to results.csv from fvg_width_sweep.py")
    ap.add_argument("--out", default=None, metavar="PATH",
                     help="Output PNG path (default: alongside the CSV)")
    ap.add_argument("--show", action="store_true", help="Open the chart interactively")
    args = ap.parse_args()
    plot_from_csv(args.csv, save_path=args.out, show=args.show)


if __name__ == "__main__":
    main()
