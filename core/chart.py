"""Shared chart utilities — colour palette, profile computation, annotation helpers.

No moomoo SDK imports, no Tkinter.  Safe to import from tests and backtest.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

# ── Colour palette ────────────────────────────────────────────────────────────
BG_DARK = "#1a1a2e"
BG_BAR  = "#2a2a3e"
BG_EDIT = "#333355"
BG_TIP  = "#0d1b2a"
FG      = "white"
GREEN   = "#26a69a"
RED     = "#ef5350"
GREY    = "#888888"
GRID    = "#334455"
GOLD    = "#ffd700"
CROSS   = "#aaaacc"

# Asian market convention: red = up/bullish, green = down/bearish
UP   = RED    # "#ef5350"
DOWN = GREEN  # "#26a69a"

PROFILE_BINS = 30


def build_ohlcv_profile(
    klines: pd.DataFrame,
    n_bins: int = PROFILE_BINS,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Approximate volume-at-price from OHLCV data.

    Distributes each candle's volume uniformly across its [low, high] range
    and accumulates across all candles into *n_bins* price buckets.

    Returns ``(bin_centers, volumes)`` or ``None`` when the price range is
    degenerate (all candles at the same price).
    """
    lo_min = klines["low"].min()
    hi_max = klines["high"].max()
    if hi_max - lo_min < 1e-9:
        return None

    bins    = np.linspace(lo_min, hi_max, n_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    volumes = np.zeros(n_bins)

    for _, row in klines.iterrows():
        mask = (centers >= row["low"]) & (centers <= row["high"])
        n = mask.sum()
        if n:
            volumes[mask] += row["volume"] / n

    return centers, volumes


def make_annot(ax):
    """Create an initially-invisible hover tooltip annotation on *ax*."""
    return ax.annotate(
        "", xy=(0, 0), xytext=(12, 12), textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.4", fc=BG_TIP, ec="#556688", alpha=0.92),
        color=FG, fontsize=8, visible=False, zorder=10,
    )


def make_float_tip(ax) -> object:
    """Create a floating tooltip that follows the cursor (positioned above it)."""
    return ax.annotate(
        "", xy=(0, 0), xytext=(0, 14), textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.4", fc=BG_TIP, ec="#556688", alpha=0.92),
        color=FG, fontsize=8, visible=False, zorder=10,
        ha="center", va="bottom",
    )
