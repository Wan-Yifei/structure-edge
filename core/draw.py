"""Pure matplotlib drawing helpers shared across chart windows.

All functions are side-effect-free on data: they take axes + data and return
artist objects (for later cleanup).  No GUI state, no tkinter, no moomoo SDK.
"""

from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle as MplRect, Patch
from matplotlib.collections import LineCollection

from core.chart import (
    BG_BAR, BG_TIP, FG, GREEN, RED, GREY, GRID, GOLD, UP, DOWN,
    build_ohlcv_profile,
)
from core.time_utils import candle_start


# ── Candles ───────────────────────────────────────────────────────────────────

def draw_candles(ax, klines: pd.DataFrame) -> list[str]:
    """Draw OHLCV candle bodies + wicks.  Returns x-axis tick label list."""
    opens  = klines["open"].values
    highs  = klines["high"].values
    lows   = klines["low"].values
    closes = klines["close"].values
    n      = len(klines)
    x      = np.arange(n, dtype=float)

    up_mask = closes >= opens
    heights = np.abs(closes - opens)
    bottoms = np.minimum(opens, closes)

    # two bar() calls (one per colour) → 2 BarContainers instead of N
    if up_mask.any():
        ax.bar(x[up_mask], heights[up_mask], bottom=bottoms[up_mask],
               color=UP, width=0.6, zorder=3)
    if (~up_mask).any():
        ax.bar(x[~up_mask], heights[~up_mask], bottom=bottoms[~up_mask],
               color=DOWN, width=0.6, zorder=3)

    # one LineCollection for all wicks instead of N Line2D objects
    segments     = [[(xi, lows[i]), (xi, highs[i])] for i, xi in enumerate(x)]
    wick_colors  = [UP if u else DOWN for u in up_mask]
    ax.add_collection(LineCollection(segments, colors=wick_colors, linewidths=1, zorder=2))

    return [
        (str(r["time_key"])[5:16] if len(str(r["time_key"])) >= 16 else str(r["time_key"]))
        for _, r in klines.iterrows()
    ]


# ── Tick profile bars ─────────────────────────────────────────────────────────

_MAX_PROFILE_BINS = 200


def _bin_profile(
    prices: list[float],
    buy_v: np.ndarray,
    sell_v: np.ndarray,
    neu_v: np.ndarray,
    max_bins: int = _MAX_PROFILE_BINS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collapse many tick prices into at most max_bins price buckets."""
    prices_arr = np.array(prices, dtype=float)
    if len(prices_arr) <= max_bins:
        return prices_arr, buy_v, sell_v, neu_v
    lo, hi = prices_arr[0], prices_arr[-1]
    edges   = np.linspace(lo, hi, max_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    idx     = np.clip(np.searchsorted(edges[1:], prices_arr, side="left"), 0, max_bins - 1)
    new_buy  = np.zeros(max_bins, dtype=float)
    new_sell = np.zeros(max_bins, dtype=float)
    new_neu  = np.zeros(max_bins, dtype=float)
    np.add.at(new_buy,  idx, buy_v)
    np.add.at(new_sell, idx, sell_v)
    np.add.at(new_neu,  idx, neu_v)
    return centers, new_buy, new_sell, new_neu


def draw_tick_profile_bars(
    ax_p,
    prices: list[float],
    buy_v: np.ndarray,
    sell_v: np.ndarray,
    neu_v: np.ndarray,
    title: str,
    ax_c=None,
    klines: pd.DataFrame | None = None,
    max_bins: int = _MAX_PROFILE_BINS,
) -> tuple[list, object, object]:
    """Draw stacked buy/sell/neutral horizontal bars on ax_p.

    Returns (patch_list, axvline, legend) for later surgical removal.
    Prices are binned to at most max_bins levels to keep artist count low.
    """
    y, buy_v, sell_v, neu_v = _bin_profile(prices, buy_v, sell_v, neu_v, max_bins=max_bins)
    h = float((y[1] - y[0]) * 0.8) if len(y) > 1 else 0.05

    bars1 = ax_p.barh(y, buy_v,             height=h, color=UP,   alpha=0.85)
    bars2 = ax_p.barh(y, neu_v, left=buy_v, height=h, color=GREY, alpha=0.70)
    bars3 = ax_p.barh(y, sell_v, left=buy_v + neu_v,
                      height=h, color=DOWN, alpha=0.85)
    vl  = ax_p.axvline(0, color=FG, linewidth=0.5, alpha=0.4)

    leg = ax_p.legend(
        handles=[Patch(color=UP, alpha=0.85, label="Buy"),
                 Patch(color=GREY, alpha=0.70, label="Neutral"),
                 Patch(color=DOWN, alpha=0.85, label="Sell")],
        loc="lower right", fontsize=7,
        facecolor=BG_BAR, labelcolor=FG, edgecolor="#444466")

    patches = [p for c in [bars1, bars2, bars3] for p in c.patches]

    ax_p.set_title(title, color=FG, fontsize=9)
    ax_p.set_xlabel("Volume", color=FG, fontsize=8)
    ax_p.tick_params(axis="x", colors=FG, labelsize=7)
    ax_p.grid(axis="x", color=GRID, linewidth=0.5)

    if ax_c is not None and klines is not None:
        # y-axis is determined solely by candle OHLC — tick prices must not extend it
        ymin = float(klines["low"].min())  * 0.999
        ymax = float(klines["high"].max()) * 1.001
        ax_c.set_ylim(ymin, ymax)

    return patches, vl, leg


# ── Bucket helpers ────────────────────────────────────────────────────────────

def aggregate_buckets(buckets: dict) -> dict[float, dict]:
    """Flatten candle-keyed buckets into price → {buy, sell, neutral}."""
    agg: dict[float, dict] = defaultdict(lambda: {"buy": 0, "sell": 0, "neutral": 0})
    for price_levels in buckets.values():
        for price, counts in price_levels.items():
            agg[price]["buy"]     += counts["buy"]
            agg[price]["sell"]    += counts["sell"]
            agg[price]["neutral"] += counts["neutral"]
    return dict(agg)


def bucket_coverage(buckets: dict, klines: pd.DataFrame, candle_mins: int) -> int:
    """Return % of kline candles that have tick data in buckets."""
    kline_candles: set[datetime] = set()
    for tk_str in klines["time_key"]:
        try:
            bar_end = datetime.strptime(str(tk_str)[:16], "%Y-%m-%d %H:%M")
            kline_candles.add(candle_start(bar_end - timedelta(minutes=candle_mins), candle_mins))
        except ValueError:
            pass
    if not kline_candles:
        return 0
    covered = kline_candles & set(buckets.keys())
    return int(100 * len(covered) / len(kline_candles))


def prices_arrays(
    price_dict: dict,
    lo: float | None = None,
    hi: float | None = None,
) -> tuple[list[float], np.ndarray, np.ndarray, np.ndarray]:
    """Unpack price_dict → (prices, buy_v, sell_v, neu_v) sorted by price.

    lo/hi: optional bounds to clip outlier tick prices before binning.
    """
    prices = sorted(
        p for p in price_dict
        if (lo is None or p >= lo) and (hi is None or p <= hi)
    )
    buy_v  = np.array([price_dict[p]["buy"]     for p in prices], dtype=float)
    sell_v = np.array([price_dict[p]["sell"]    for p in prices], dtype=float)
    neu_v  = np.array([price_dict[p]["neutral"] for p in prices], dtype=float)
    return prices, buy_v, sell_v, neu_v


# ── OHLCV profile ─────────────────────────────────────────────────────────────

def draw_ohlcv_profile(
    ax_p,
    ax_c,
    klines: pd.DataFrame,
    n_bins: int = 40,
    date_label: str = "",
) -> tuple[np.ndarray, np.ndarray] | None:
    """Draw OHLCV-estimated volume profile with POC.

    Returns (centers, volumes) or None if price range is degenerate.
    """
    result = build_ohlcv_profile(klines, n_bins=n_bins)
    if result is None:
        ax_p.text(0.5, 0.5, "Insufficient\nprice range",
                  ha="center", va="center", color="#aaaaaa", fontsize=9,
                  transform=ax_p.transAxes)
        ax_p.set_title("Vol Profile", color=FG, fontsize=10)
        return None

    centers, volumes = result
    bin_h   = (centers[1] - centers[0]) * 0.85
    max_vol = volumes.max() or 1
    colors  = [GREEN if v >= max_vol * 0.7 else (FG if v >= max_vol * 0.4 else GREY)
               for v in volumes]
    ax_p.barh(centers, volumes, height=bin_h, color=colors, alpha=0.85)

    poc_price = centers[int(np.argmax(volumes))]
    for ax in (ax_c, ax_p):
        ax.axhline(poc_price, color=GOLD, linewidth=0.9, linestyle="--", alpha=0.8, zorder=5)
    ax_c.text(len(klines) - 0.5, poc_price, f" POC {poc_price:.2f}",
              color=GOLD, fontsize=7, va="center", ha="left")

    ax_p.set_title(f"Vol Profile (OHLCV)\n{date_label}", color=FG, fontsize=9)
    ax_p.set_xlabel("Volume", color=FG, fontsize=8)
    ax_p.tick_params(axis="x", colors=FG, labelsize=7)
    ax_p.grid(axis="x", color=GRID, linewidth=0.5)
    ax_c.set_ylim(klines["low"].min() * 0.999, klines["high"].max() * 1.001)

    return centers, volumes


# ── Heatmap + Delta ───────────────────────────────────────────────────────────

_UP_RGB = np.array([0xef / 255, 0x53 / 255, 0x50 / 255], dtype=np.float32)
_DN_RGB = np.array([0x26 / 255, 0xa6 / 255, 0x9a / 255], dtype=np.float32)


def draw_candle_heatmap(
    ax_c,
    klines: pd.DataFrame,
    buckets: dict,
    candle_mins: int,
    n_price_bins: int = 120,
) -> None:
    """Overlay buy/sell heat as a single imshow image (replaces per-patch approach).

    Reduces matplotlib artist count from n_candles*n_bins to 1, giving a large
    speedup on charts with many candles or many tick price levels.
    """
    n    = len(klines)
    ylo  = float(klines["low"].min())  * 0.9998
    yhi  = float(klines["high"].max()) * 1.0002
    if yhi <= ylo:
        return

    img   = np.zeros((n_price_bins, n, 4), dtype=np.float32)
    edges = np.linspace(ylo, yhi, n_price_bins + 1)
    ctrs  = (edges[:-1] + edges[1:]) / 2.0

    for i, (_, row) in enumerate(klines.iterrows()):
        tk_str = str(row["time_key"])
        try:
            bar_end = datetime.strptime(tk_str[:16], "%Y-%m-%d %H:%M")
            bk = candle_start(bar_end - timedelta(minutes=candle_mins), candle_mins)
        except ValueError:
            continue
        pd_ = buckets.get(bk)
        if not pd_:
            continue
        lo, hi = float(row["low"]), float(row["high"])
        if hi <= lo:
            continue

        prices_arr = np.fromiter(pd_.keys(), dtype=float, count=len(pd_))
        buy_arr    = np.array([pd_[p]["buy"]  for p in pd_], dtype=np.float32)
        sell_arr   = np.array([pd_[p]["sell"] for p in pd_], dtype=np.float32)

        bin_idx = np.clip(
            np.searchsorted(edges[1:], prices_arr, side="left"),
            0, n_price_bins - 1,
        ).astype(int)
        buy_b  = np.zeros(n_price_bins, dtype=np.float32)
        sell_b = np.zeros(n_price_bins, dtype=np.float32)
        np.add.at(buy_b,  bin_idx, buy_arr)
        np.add.at(sell_b, bin_idx, sell_arr)

        in_candle = (ctrs >= lo) & (ctrs <= hi)
        total = buy_b + sell_b
        mx = float(total[in_candle].max()) if in_candle.any() else 0.0
        if mx == 0:
            continue

        bs    = buy_b + sell_b
        ratio = np.divide(buy_b - sell_b, bs,
                          out=np.zeros(n_price_bins, dtype=np.float32),
                          where=bs > 0)
        alpha = ((0.12 + 0.55 * total / mx) * (0.25 + 0.75 * np.abs(ratio))) * in_candle
        np.clip(alpha, 0, 0.85, out=alpha)

        img[:, i, :3] = np.where(ratio[:, None] >= 0, _UP_RGB, _DN_RGB)
        img[:, i, 3]  = alpha

    ax_c.imshow(
        img, aspect="auto", origin="lower",
        extent=(-0.5, n - 0.5, ylo, yhi),
        interpolation="nearest", zorder=1,
    )


def draw_candle_deltas(
    ax_c,
    klines: pd.DataFrame,
    buckets: dict,
    candle_mins: int,
) -> None:
    """Annotate each candle with net Δ (buy − sell) below the low."""
    price_range = max(float(klines["high"].max() - klines["low"].min()), 0.01)
    offset      = price_range * 0.004

    for i, (_, row) in enumerate(klines.iterrows()):
        tk_str = str(row["time_key"])
        try:
            bar_end = datetime.strptime(tk_str[:16], "%Y-%m-%d %H:%M")
            bk = candle_start(bar_end - timedelta(minutes=candle_mins), candle_mins)
        except ValueError:
            continue
        pd_ = buckets.get(bk)
        if not pd_:
            continue

        total_buy  = sum(pd_[p]["buy"]  for p in pd_)
        total_sell = sum(pd_[p]["sell"] for p in pd_)
        delta = total_buy - total_sell
        if delta == 0:
            continue

        sign = "+" if delta >= 0 else ""
        if abs(delta) >= 1_000_000:
            txt = f"{sign}{delta/1_000_000:.1f}M"
        elif abs(delta) >= 1_000:
            txt = f"{sign}{delta/1_000:.0f}K"
        else:
            txt = f"{sign}{delta}"

        ax_c.text(
            i, float(row["low"]) - offset, txt,
            ha="center", va="top", fontsize=6,
            color=UP if delta >= 0 else DOWN,
            fontweight="bold", zorder=6,
        )


# ── SMC: BOS / CHoCH ─────────────────────────────────────────────────────────

def draw_bos_choch(ax_c, klines: pd.DataFrame, signals: list[dict]) -> list:
    """Draw BOS / CHoCH horizontal lines + labels.

    Each signal dict: {type: 'BOS'|'CHoCH', direction: 'bull'|'bear',
                       idx: int, price: float, from_idx: int}
    Returns list of drawn artists for cleanup.
    """
    artists = []
    for sig in signals:
        color   = UP   if sig["direction"] == "bull" else DOWN
        ls      = "-"  if sig["type"] == "BOS" else "--"
        x_start = sig.get("from_idx", max(0, sig["idx"] - 3))
        x_end   = sig["idx"]          # stop at the confirmation candle
        line,   = ax_c.plot([x_start, x_end], [sig["price"], sig["price"]],
                            color=color, linewidth=1.5, linestyle=ls,
                            alpha=0.90, zorder=4)
        label = ax_c.text(
            x_end, sig["price"], f" {sig['type']}",
            color=color, fontsize=7, va="center", ha="left",
            fontweight="bold", zorder=5,
            bbox=dict(fc=BG_TIP, ec="none", alpha=0.70, pad=1.5),
        )
        artists.extend([line, label])
    return artists


# ── SMC: FVG ──────────────────────────────────────────────────────────────────

def draw_fvg(ax_c, klines: pd.DataFrame, gaps: list[dict]) -> list:
    """Draw Fair Value Gap rectangles.

    Each gap dict: {direction: 'bull'|'bear', top: float, bottom: float,
                    idx: int, filled: bool}
    Returns list of drawn artists.
    """
    artists = []
    n = len(klines)
    for gap in gaps:
        color   = UP   if gap["direction"] == "bull" else DOWN
        alpha   = 0.13 if gap.get("filled") else 0.28
        x_start = gap["idx"]
        rect = MplRect(
            (x_start - 0.5, gap["bottom"]),
            n - x_start,
            gap["top"] - gap["bottom"],
            facecolor=color, edgecolor=color,
            alpha=alpha, zorder=2, linewidth=0.6,
        )
        ax_c.add_patch(rect)
        lbl = ax_c.text(
            x_start, (gap["top"] + gap["bottom"]) / 2,
            "FVG", color=color, fontsize=6, va="center", ha="left",
            fontweight="bold", zorder=5,
            bbox=dict(fc=BG_TIP, ec="none", alpha=0.65, pad=1),
        )
        artists.extend([rect, lbl])
    return artists


# ── SMC: Order Blocks ─────────────────────────────────────────────────────────

_OB_COLORS = {
    "regular":    ("#4fc3f7", "#ef9a9a"),   # bull / bear light blue / light red
    "breaker":    ("#ff7043", "#ab47bc"),   # burnt orange / purple
    "mitigation": ("#ffca28", "#ffca28"),   # amber for both
}

def draw_order_blocks(ax_c, klines: pd.DataFrame, blocks: list[dict]) -> list:
    """Draw Order Block rectangles with subtype labels.

    Each block dict: {direction: 'bull'|'bear',
                      subtype: 'regular'|'breaker'|'mitigation',
                      top: float, bottom: float, idx: int}
    Returns list of drawn artists.
    """
    artists = []
    n = len(klines)
    for blk in blocks:
        bull = blk["direction"] == "bull"
        bull_c, bear_c = _OB_COLORS.get(blk.get("subtype", "regular"), ("#4fc3f7", "#ef9a9a"))
        color   = bull_c if bull else bear_c
        x_start = blk["idx"]
        rect = MplRect(
            (x_start - 0.5, blk["bottom"]),
            n - x_start,
            blk["top"] - blk["bottom"],
            edgecolor=color, facecolor=color,
            alpha=0.22, zorder=2, linewidth=1.2,
        )
        ax_c.add_patch(rect)
        subtype = blk.get("subtype", "OB").replace("regular", "OB").capitalize()
        lbl = ax_c.text(
            x_start, blk["top"], f" {subtype}",
            color=color, fontsize=6, va="bottom", ha="left",
            fontweight="bold", zorder=5,
            bbox=dict(fc=BG_TIP, ec="none", alpha=0.65, pad=1),
        )
        artists.extend([rect, lbl])
    return artists
