"""
Trade Viewer Qt — PyQtGraph-based chart viewer

Replaces Tkinter + Matplotlib with PyQt6 + PyQtGraph for significantly
faster rendering, native zoom/pan, and smooth crosshair interaction.

Features (Phase 1 + 2):
  - Live / Historical modes
  - Candlestick chart with per-bin tick heatmap colouring
  - Volume subplot
  - Delta Δ annotations per candle
  - BOS / CHoCH structure markers
  - FVG zone overlays
  - Order Block (OB) overlays with regular / mitigation / breaker subtypes
  - KD channel subplot (spread width momentum, bull/bear/flat coloured)
  - CVD subplot (cumulative buy-sell tick delta)
  - A/D Line subplot (ta.volume.AccDistIndexIndicator, pure OHLCV)
  - EMA overlays (20 / 50 / 200) on candle chart
  - Session Vol Profile panel (right)
  - Single-candle Tick Profile panel (centre-right, on hover)
  - Range Profile: user-selected region volume-by-price + Buy/Sell/Neutral/Medium stats
  - Crosshair + OHLCV tooltip (left-side, below price label)
  - Indicator toggle toolbar
  - Session filter checkboxes (Pre / Regular / Post / Night)
  - Profile date-range selector (1D / 3D / 1W, trading-day aware)
  - Pin-to-top toggle (WindowStaysOnTopHint)
  - Trade Review mode: enter a trade_id, see entry/exit/SL/TP markers
    plus HTF FVG + BOS context overlaid on the entry TF

Usage:
    uv run analysis/trade_viewer_qt.py
    uv run analysis/trade_viewer_qt.py --code US.SNDK --tf 5m
    uv run analysis/trade_viewer_qt.py --mode Historical --date 2026-05-20
    uv run analysis/trade_viewer_qt.py --code US.AAPL --tf 15m --mode Historical --date 2026-05-15
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import os
import pathlib
import sqlite3
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtCore import (
    Qt, QThread, QTimer, pyqtSignal, QRectF, QPointF, QMetaObject, Q_ARG,
)
from PyQt6.QtGui import (
    QColor, QPainter, QPicture, QPen, QBrush, QFont, QIcon, QPixmap,
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QToolBar, QLabel, QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox,
    QPushButton, QCheckBox, QButtonGroup, QRadioButton, QSizePolicy,
    QFrame, QStatusBar, QMessageBox, QDialog, QDialogButtonBox, QFormLayout,
    QSystemTrayIcon,
)

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from core.time_utils import candle_start
from feeds.fetcher import fetch_klines
from strategy.smc.market_structure import detect_bos_choch
from strategy.smc.fvg import detect_fvg
from strategy.smc.order_blocks import detect_order_blocks
from strategy.smc.kd_trend import compute_kd
from strategy.chandelier_exit.chandelier import current_stop
from strategy.option.gex import (
    get_near_expiry_dates, fetch_option_data as fetch_option_chain_data,
    compute_gex, build_gex_stats, top_gex_walls,
)
from ta.volume import AccDistIndexIndicator
from moomoo import (
    OpenQuoteContext, SubType, KLType, AuType, Session,
    TickerHandlerBase, StockQuoteHandlerBase, RET_OK,
)

# ── PyQtGraph global config ───────────────────────────────────────────────────

# useOpenGL/enableExperimental were on since this file's first commit and
# never revisited. PyQtGraph's experimental OpenGL path is documented
# upstream as unstable, particularly around widget teardown on Windows --
# live py-spy dumps of a genuinely hung process (reported: main viewer goes
# unresponsive after closing the Liquidity Heatmap window, worse the longer
# it ran) showed the main thread parked in app.exec() with *no* Python frame
# underneath it at all, i.e. stuck inside native Qt/GL C++ code that never
# returns control to the interpreter -- exactly this class of bug. The
# Liquidity Heatmap window (pg.ImageItem, per-tick setImage() on a
# N_PRICE x max_cols grid) is this process's heaviest GL-texture churn,
# consistent with "worse the longer the heatmap window ran."
pg.setConfigOptions(antialias=True)

# ── Colour palette (matches trade_viewer.py) ──────────────────────────────────

_BG       = "#1a1a2e"
_BG_BAR   = "#16213e"
_BG_TIP   = "#0f3460"
_FG       = "#e0e0e0"
_GREEN    = "#26a69a"
_RED      = "#ef5350"
_GREY     = "#546e7a"
_GRID     = "#263238"
_GOLD     = "#ffa726"
_CROSS    = "#b0bec5"
_UP       = "#ffa005"   # heatmap gold  (buyer-dominant)
_DOWN     = "#ab47bc"   # heatmap purple (seller-dominant)
_OB_BULL  = "#26a69a"   # bull OB teal
_OB_BEAR  = "#ef5350"   # bear OB red
_OB_BREAK = "#9e9e9e"   # breaker OB grey
_OB_MIT   = "#ffa726"   # mitigation OB amber
_EMA_COLS = ["#42a5f5", "#ab47bc", "#ffa726"]  # EMA 20/50/200
_AVWAP_COL = "#ffeb3b"  # anchored VWAP line/label color
_ZERO_GAMMA_COL = "#ff8c00"  # option Zero Gamma line -- matches gex.py's own matplotlib chart

def _qc(hex_str: str, alpha: int = 255) -> QColor:
    c = QColor(hex_str)
    c.setAlpha(alpha)
    return c


def _fmt_vol_short(v: float) -> str:
    """Abbreviate a volume value: 1_234_567 -> '1.2M', 250_000 -> '250K'."""
    v = abs(v)
    if v >= 1_000_000:
        return f"{v/1_000_000:.1f}M"
    if v >= 1_000:
        return f"{v/1_000:.0f}K"
    return f"{v:.0f}"


def _fmt_signed_short(v: float) -> str:
    """Abbreviate a signed value: -45_000 -> '-45K', 1_234_567 -> '1.2M'."""
    sign = "-" if v < 0 else ""
    v = abs(v)
    if v >= 1_000_000:
        return f"{sign}{v/1_000_000:.1f}M"
    if v >= 1_000:
        return f"{sign}{v/1_000:.0f}K"
    return f"{sign}{v:.0f}"


class _VolumeAxisItem(pg.AxisItem):
    """Axis with K/M-abbreviated tick labels, for the profile/CVD panels.

    Their volume/CVD axis carries raw values up to the hundreds of
    thousands; pyqtgraph's default tick formatting (and enableAutoSIPrefix,
    which doesn't reliably kick in once the ViewBox's range is set manually
    rather than left on auto-range) renders these as unabbreviated 6-digit
    numbers that either run together in a narrow panel or just clutter the
    axis unnecessarily.
    """

    def __init__(self, orientation: str = "bottom", signed: bool = False) -> None:
        super().__init__(orientation=orientation)
        self._fmt = _fmt_signed_short if signed else _fmt_vol_short

    def tickStrings(self, values, scale, spacing):
        return [self._fmt(v * scale) for v in values]

# ── Timeframe / BOS config (mirrors trade_viewer.py) ─────────────────────────

TIMEFRAME_MAP: dict[str, tuple[KLType, int]] = {
    "1m":  (KLType.K_1M,    1),
    "3m":  (KLType.K_3M,    3),
    "5m":  (KLType.K_5M,    5),
    "15m": (KLType.K_15M,  15),
    "30m": (KLType.K_30M,  30),
    "1h":  (KLType.K_60M,  60),
    "4h":  (KLType.K_240M, 240),
    "1d":  (KLType.K_DAY,  1440),
}

# SubType equivalents for each TF, used to subscribe K-line push so that
# request_history_kline for intraday bars (especially 1min) returns data.
# K_240M has no push SubType in moomoo SDK; omit to avoid subscribe error.
_TF_KLINE_SUBTYPE: dict[str, object] = {
    "1m":  SubType.K_1M,
    "3m":  SubType.K_3M,
    "5m":  SubType.K_5M,
    "15m": SubType.K_15M,
    "30m": SubType.K_30M,
    "1h":  SubType.K_60M,
    "1d":  SubType.K_DAY,
}

_DAY_CANDLES: dict[str, int] = {
    "1m": 390, "5m": 78, "15m": 26, "30m": 14, "1h": 7, "4h": 6, "1d": 1,
}

# BOS max span: max bars between a reference swing and its break bar.
# Viewer uses larger values than the backtest engine so that market structure
# spanning the full visible window is shown (not just nearby swings).
_BOS_MAX_SPAN: dict[str, int] = {
    "1m": 390, "5m": 120, "15m": 100, "30m": 60, "1h": 40, "4h": 30, "1d": 20,
}

# Trend window: backward-looking bar count used to determine local trend direction
# for BOS/CHoCH classification.  Larger than engine default for broader context.
_TREND_WINDOW: dict[str, int] = {
    "1m": 120, "5m": 60, "15m": 50, "30m": 40, "1h": 30, "4h": 20, "1d": 15,
}

_LIVE_LOOKBACK_DAYS: dict[str, int] = {
    "1m": 5, "3m": 5, "5m": 5, "15m": 7, "30m": 10, "1h": 14, "4h": 30, "1d": 730,
}

# Historical mode: calendar days before the selected date to fetch.
# Larger values for higher timeframes so the chart has enough candles.
_HIST_LOOKBACK_DAYS: dict[str, int] = {
    "1m": 3, "3m": 5, "5m": 10, "15m": 20, "30m": 30, "1h": 90, "4h": 500, "1d": 2000,
}

# EMA periods shown on the candle chart
_EMA_PERIODS = [20, 50, 200]

# KD default parameters (match backtest defaults)
_KD_FAST = 25
_KD_SLOW = 90

_VOL_MA = 20  # period for the volume moving-average curve in the MAVOL subplot

# ── chart.json config (BOS/CHoCH session-gap settings) ───────────────────────
_ROOT_DIR = pathlib.Path(__file__).parent.parent
_CHART_CFG: dict = {}
try:
    _CHART_CFG = json.loads((_ROOT_DIR / "config" / "chart.json").read_text())
except Exception:
    pass

# max_session_gap per TF from config (None = no restriction)
_BOS_SESSION_GAP: dict[str, int | None] = (
    _CHART_CFG.get("bos_choch", {}).get("max_session_gap", {})
)

# ── Shared tick-loading helper ────────────────────────────────────────────────

def load_local_ticks(code: str, date_str: str, tf: str) -> dict | None:
    """Load tick buckets from ticks.db for a given code and date.

    Each price bin contains:
        buy / sell / neutral  — totals (used by heatmap)
        buy_s / buy_m / buy_l — size-stratified buy volume
        sell_s / sell_m / sell_l — size-stratified sell volume

    S/M/L thresholds are computed from the day's volume distribution
    (p33 = small/medium boundary, p67 = medium/large boundary).
    """
    db_path = pathlib.Path(__file__).parent.parent / "db" / "ticks.db"
    if not db_path.exists():
        return None
    from feeds.tick_store import TickStore
    try:
        dt             = datetime.strptime(date_str, "%Y-%m-%d")
        _, candle_mins = TIMEFRAME_MAP[tf]
        tick_start     = dt - timedelta(days=1) + timedelta(hours=20)
        tick_end       = dt.replace(hour=23, minute=59, second=59)
        with TickStore(db_path, read_only=True) as store:
            rows = store.query_ticks(code, tick_start, tick_end)
        if not rows:
            return None

        # Compute per-day volume percentiles for adaptive S/M/L thresholds.
        vols = np.array([r["volume"] for r in rows if r["volume"] > 0],
                        dtype=float)
        if len(vols) >= 6:
            thresh_m = float(np.percentile(vols, 33))  # small  < p33
            thresh_l = float(np.percentile(vols, 67))  # medium p33..p67, large > p67
        else:
            thresh_m = thresh_l = float("inf")          # too few ticks → all small

        def _size(v: int) -> str:
            if v <= thresh_m:
                return "s"
            if v <= thresh_l:
                return "m"
            return "l"

        def _new_bin() -> dict:
            return {
                "buy": 0, "sell": 0, "neutral": 0,
                "buy_s": 0, "buy_m": 0, "buy_l": 0,
                "sell_s": 0, "sell_m": 0, "sell_l": 0,
                "neutral_s": 0, "neutral_m": 0, "neutral_l": 0,
            }

        buckets: dict = defaultdict(lambda: defaultdict(_new_bin))
        for r in rows:
            ts = (r["ts"] if isinstance(r["ts"], datetime)
                  else datetime.fromisoformat(str(r["ts"])))
            bucket = candle_start(ts, candle_mins)
            key = {"BUY": "buy", "SELL": "sell"}.get(
                r["direction"].upper(), "neutral")
            vol = r["volume"]
            buckets[bucket][r["price"]][key] += vol
            buckets[bucket][r["price"]][f"{key}_{_size(vol)}"] += vol
        return dict(buckets)
    except Exception:
        return None


def load_raw_ticks(code: str, date_str: str) -> list[dict]:
    """Load raw tick records from ticks.db for cross-referencing with order book data.

    Returns list of dicts: {ts, price, volume, direction}.
    Returns [] when ticks.db does not exist.
    """
    db_path = pathlib.Path(__file__).parent.parent / "db" / "ticks.db"
    if not db_path.exists():
        return []
    from feeds.tick_store import TickStore
    try:
        dt    = datetime.strptime(date_str, "%Y-%m-%d")
        start = (dt - timedelta(days=1)).replace(hour=20, minute=0, second=0)
        end   = dt.replace(hour=23, minute=59, second=59)
        with TickStore(db_path, read_only=True) as store:
            return store.query_ticks(code, start, end)
    except Exception:
        return []


def load_order_book_data(code: str, date_str: str) -> list[dict]:
    """Load order book snapshots from order_book.db for the given date.

    Returns list of dicts: {ts, side, price, volume}.
    Returns [] when the DB does not exist (collector not yet run).
    """
    db_path = pathlib.Path(__file__).parent.parent / "db" / "order_book.db"
    if not db_path.exists():
        return []
    from feeds.order_book_store import OrderBookStore
    try:
        dt    = datetime.strptime(date_str, "%Y-%m-%d")
        start = (dt - timedelta(days=1)).replace(hour=20, minute=0, second=0)
        end   = dt.replace(hour=23, minute=59, second=59)
        with OrderBookStore(db_path, read_only=True) as store:
            return store.query_snapshots(code, start, end)
    except Exception:
        return []


def load_order_book_window(code: str, start: datetime, end: datetime) -> list[dict]:
    """Load order book snapshots for an explicit time window.

    Used in Live mode so that after-hours / overnight / weekend snapshots
    (which fall outside any K-line bar) are included and can be mapped to the
    last visible bar on the chart.
    """
    db_path = pathlib.Path(__file__).parent.parent / "db" / "order_book.db"
    if not db_path.exists():
        return []
    from feeds.order_book_store import OrderBookStore
    try:
        with OrderBookStore(db_path, read_only=True) as store:
            return store.query_snapshots(code, start, end)
    except Exception:
        return []


def _trading_day(time_key, cm: int):
    """Trading-day date for a bar, end-of-bar `time_key`.

    A trading day runs 20:00 ET -> next 20:00 ET (overnight session start
    through the following after-hours close), matching the "night" session
    window used by _filter_sessions()/config/schedule.json and the same rule
    _draw_cvd() resets on. Returns None if `time_key` doesn't parse.
    """
    try:
        bar_end = datetime.strptime(str(time_key)[:16], "%Y-%m-%d %H:%M")
    except ValueError:
        return None
    bk = candle_start(bar_end - timedelta(minutes=cm), cm)
    return bk.date() + timedelta(days=1) if bk.hour >= 20 else bk.date()


def apply_profile_range(klines: pd.DataFrame, range_val: str, cm: int) -> pd.DataFrame:
    """Trim klines to N trading days ending at the last bar.

    Uses _trading_day() (20:00 ET boundary), not the calendar date -- the
    calendar date would split the continuous overnight session in half at
    midnight, so for a few minutes right after midnight "today" would only
    contain the handful of bars since 00:00 while the entire evening's
    overnight-session bars (actually still part of the same still-open
    trading day) get dropped, leaving the profile empty or near-empty.
    """
    if klines.empty:
        return klines
    n_days = {"1d": 1, "2d": 2, "3d": 3, "7d": 5}.get(range_val)
    if n_days is None:
        return klines
    days  = klines["time_key"].map(lambda tk: _trading_day(tk, cm))
    valid = days.notna()
    if not valid.any():
        return klines.iloc[0:0]
    anchor = days[valid].iloc[-1]
    uniq   = sorted(days[valid].unique())
    start  = uniq[-n_days] if len(uniq) >= n_days else uniq[0]
    mask   = valid & (days >= start) & (days <= anchor)
    return klines[mask]


def _compute_profile_bins(
    kl: pd.DataFrame,
    ticks: dict | None,
    candle_mins: int,
    n_bins: int = 60,
) -> tuple[np.ndarray, np.ndarray, bool, dict]:
    """Volume profile bins for the given klines rows.

    `kl` is whatever slice of bars the caller wants a profile over -- a
    contiguous chart-index range (Range Profile) or a session-filtered,
    possibly non-contiguous subset (Session Volume Profile). Either way
    this function just iterates the rows it's given; it no longer assumes
    they're a contiguous slice of some larger frame, which is what let the
    Session Volume Profile share this instead of keeping its own inlined
    copy of the same tick/OHLCV hybrid logic.

    Prefers tick data (exact price levels via np.digitize); falls back to OHLCV
    proportional distribution.  Returns (centers, volumes, used_ticks, stats).
    centers and volumes are empty arrays when the price range is degenerate.

    stats holds range-wide totals {"total", "buy", "sell", "neutral", "medium"},
    all derived from tick-covered bars only (OHLCV-fallback bars carry no
    direction/size info, so they're excluded here even though they still
    contribute to the visual `volumes` bins).  "total" = buy + sell + neutral,
    so it's a closed sum -- moomoo's ticker_direction tags a large share of
    prints NEUTRAL (zero-tick trades), so neutral is often the biggest of the
    three.  "medium" is a size breakdown *within* buy + sell (buy_m + sell_m),
    not a fourth additive bucket -- it will always be <= buy + sell.
    """
    lo = float(kl["low"].min())
    hi = float(kl["high"].max())
    empty_stats = {"total": 0.0, "buy": 0.0, "sell": 0.0, "neutral": 0.0, "medium": 0.0}
    if hi <= lo:
        return np.array([]), np.array([]), False, empty_stats

    bins    = np.linspace(lo, hi, n_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    volumes = np.zeros(n_bins, dtype=float)
    stats   = dict(empty_stats)

    # Hybrid: use tick data where available, OHLCV proportional fill for bars
    # that have no tick coverage.  This ensures bars outside the loaded tick
    # date range still contribute to the profile rather than being silently
    # dropped, which would bias the profile toward the most-recently-loaded days.
    used_ticks = False
    klo  = kl["low"].values.astype(float)
    khi  = kl["high"].values.astype(float)
    kvol = kl["volume"].fillna(0).values.astype(float)
    kt   = kl["time_key"].values
    for j in range(len(kl)):
        bar_ticks_used = False
        if ticks:
            try:
                bar_end = datetime.strptime(str(kt[j])[:16], "%Y-%m-%d %H:%M")
                bk = candle_start(
                    bar_end - timedelta(minutes=candle_mins), candle_mins)
            except ValueError:
                bk = None
            if bk is not None:
                pd_ = ticks.get(bk)
                if pd_:
                    for price, counts in pd_.items():
                        buy     = counts.get("buy", 0)
                        sell    = counts.get("sell", 0)
                        neutral = counts.get("neutral", 0)
                        total   = buy + sell + neutral
                        if total > 0:
                            p = float(price)
                            if lo <= p <= hi:
                                bi = int(np.clip(np.digitize(p, bins) - 1,
                                                 0, n_bins - 1))
                                volumes[bi] += total
                                bar_ticks_used = True
                                stats["total"]   += total
                                stats["buy"]     += buy
                                stats["sell"]    += sell
                                stats["neutral"] += neutral
                                stats["medium"]  += (counts.get("buy_m", 0)
                                                      + counts.get("sell_m", 0))
                    if bar_ticks_used:
                        used_ticks = True

        if not bar_ticks_used:
            mask  = (centers >= klo[j]) & (centers <= khi[j])
            n_hit = int(mask.sum())
            if n_hit:
                volumes[mask] += kvol[j] / n_hit

    return centers, volumes, used_ticks, stats


def _compute_poc_vah_val(
    centers: np.ndarray,
    volumes: np.ndarray,
    va_pct: float = 0.70,
) -> tuple[float, float, float]:
    """Return (poc_price, vah_price, val_price) from a volume profile.

    Uses the standard market-profile algorithm: start at POC and expand the
    Value Area outward (up or down, whichever adds more volume at each step)
    until va_pct of total volume is covered.  VAL ≤ POC ≤ VAH by construction.

    When all bins have equal volume (OHLCV flat distribution) argmax returns
    index 0 (bottom price), which would be misleading; in that case the median
    bin is used as POC instead.
    """
    if centers.size == 0:
        return 0.0, 0.0, 0.0
    poc_idx = int(np.argmax(volumes))
    # OHLCV flat distribution: all bins equal → use median bin as POC
    if np.all(volumes == volumes[poc_idx]):
        poc_idx = len(volumes) // 2
    poc   = float(centers[poc_idx])
    total = float(volumes.sum())
    if total <= 0:
        return poc, poc, poc

    # Expand VA outward from POC; at each step pick the side that adds more volume.
    # A zero-volume neighbor must still be stepped through (not treated as a
    # wall) -- with tick-placed volume the profile can be spiky/sparse (a
    # bin's ticks landing in one narrow bin with genuinely empty bins right
    # next to it), so stopping at the first zero-volume neighbor on both
    # sides gave up immediately at POC itself, leaving VAH == VAL == POC
    # even though non-zero bins existed further out. Use -1 (not 0) as the
    # "no bins left on this side" sentinel so a real zero bin still loses to
    # -1 and keeps expansion moving until bins are truly exhausted on both
    # sides or va_pct is reached.
    lo_idx = poc_idx
    hi_idx = poc_idx
    cumvol = float(volumes[poc_idx])
    while cumvol / total < va_pct and (hi_idx + 1 < len(volumes) or lo_idx - 1 >= 0):
        above = float(volumes[hi_idx + 1]) if hi_idx + 1 < len(volumes) else -1.0
        below = float(volumes[lo_idx - 1]) if lo_idx - 1 >= 0 else -1.0
        if above >= below:
            hi_idx += 1
            cumvol += float(volumes[hi_idx])
        else:
            lo_idx -= 1
            cumvol += float(volumes[lo_idx])

    return poc, float(centers[hi_idx]), float(centers[lo_idx])


def _load_trade_from_db(trade_id: str) -> tuple[dict | None, str]:
    """Load a trade record from BacktestDB or ReviewTradesDB.

    Returns (row_dict, source_label) or (None, "").
    """
    row    = None
    source = ""
    try:
        from backtest.db import BacktestDB
        with BacktestDB(read_only=True) as db:
            row = db.fetch_live_trade(trade_id)
            source = "live_trades"
            if row is None:
                row = db.fetch_trade(trade_id)
                source = "backtest"
    except Exception:
        pass

    if row is None:
        try:
            from backtest.db import ReviewTradesDB
            with ReviewTradesDB(read_only=True) as rdb:
                row = rdb.fetch_trade(trade_id)
                source = "review"
        except Exception:
            pass

    return row, source


# ── Custom GraphicsItems ──────────────────────────────────────────────────────

class CandlestickItem(pg.GraphicsObject):
    """Draws OHLCV candles with optional per-bin tick heatmap colouring."""

    HEATMAP_BINS = 20  # price bins per candle for heatmap

    def __init__(self):
        super().__init__()
        self._klines:      pd.DataFrame | None = None
        self._buckets:     dict | None         = None
        self._candle_mins: int                 = 1
        self._show_heatmap: bool               = True
        self._red_up:      bool                = False   # True = 红涨绿跌
        self._picture:     QPicture | None     = None
        self._rect:        QRectF              = QRectF()

    def set_data(self, klines: pd.DataFrame, buckets: dict | None,
                 candle_mins: int, show_heatmap: bool = True,
                 red_up: bool = False) -> None:
        self._klines       = klines
        self._buckets      = buckets
        self._candle_mins  = candle_mins
        self._show_heatmap = show_heatmap
        self._red_up       = red_up
        self._picture      = None
        if not klines.empty:
            self._rect = QRectF(
                -0.5,
                float(klines["low"].min()),
                float(len(klines)),
                float(klines["high"].max() - klines["low"].min()),
            )
        self.prepareGeometryChange()
        self.update()

    # ── build QPicture once, replay on every paint ────────────────────────────

    def _build(self) -> None:
        pic = QPicture()
        p   = QPainter(pic)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        klines      = self._klines
        buckets     = self._buckets or {}
        candle_mins = self._candle_mins

        # Color scheme: green-up/red-down (Western) OR red-up/green-down (Chinese)
        _bull_col = _RED   if self._red_up else _GREEN
        _bear_col = _GREEN if self._red_up else _RED

        # Pre-compute per-candle data so we can do a two-pass draw
        # (bodies + heatmap first, wicks on top last).
        candle_data: list[tuple] = []
        for i, (_, row) in enumerate(klines.iterrows()):
            o = float(row["open"])
            h = float(row["high"])
            l = float(row["low"])
            c = float(row["close"])
            body_lo = min(o, c)
            body_hi = max(o, c)
            if body_hi <= body_lo:
                body_hi = body_lo + 0.0001
            try:
                bar_end = datetime.strptime(
                    str(row["time_key"])[:16], "%Y-%m-%d %H:%M")
                bk = candle_start(
                    bar_end - timedelta(minutes=candle_mins), candle_mins)
            except ValueError:
                bk = None
            pd_ = buckets.get(bk) if bk else None
            candle_data.append((i, o, h, l, c, body_lo, body_hi, pd_))

        # Pass 1 — bodies + heatmap overlay
        p.setPen(Qt.PenStyle.NoPen)
        for i, o, h, l, c, body_lo, body_hi, pd_ in candle_data:
            is_bull  = c >= o
            if self._show_heatmap and pd_:
                # Draw faint base body first (direction reference only),
                # then overlay high-contrast buy/sell bins on top.
                base_col = _qc(_bull_col if is_bull else _bear_col, 50)
                p.setBrush(QBrush(base_col))
                p.drawRect(QRectF(i - 0.35, body_lo, 0.7, body_hi - body_lo))
                self._draw_heatmap_overlay(p, i, l, h, body_lo, body_hi, pd_)
            else:
                base_col = _qc(_bull_col if is_bull else _bear_col)
                p.setBrush(QBrush(base_col))
                p.drawRect(QRectF(i - 0.35, body_lo, 0.7, body_hi - body_lo))

        # Pass 2 — wicks drawn on top of everything so they are never covered.
        # width=0 → cosmetic pen: always 1 screen pixel regardless of zoom level.
        for i, o, h, l, c, body_lo, body_hi, _ in candle_data:
            is_bull    = c >= o
            wick_color = _qc(_bull_col if is_bull else _bear_col)
            p.setPen(QPen(wick_color, 0))
            p.setBrush(Qt.BrushStyle.NoBrush)
            # Upper wick
            if h > body_hi:
                p.drawLine(QPointF(i, body_hi), QPointF(i, h))
            # Lower wick
            if l < body_lo:
                p.drawLine(QPointF(i, l), QPointF(i, body_lo))

        p.end()
        self._picture = pic

    def _draw_heatmap_overlay(self, p: QPainter, i: int,
                               low: float, high: float,
                               body_lo: float, body_hi: float,
                               pd_: dict) -> None:
        """Draw buy/sell heatmap bins within the candle body only.

        Called AFTER a faint base body has already been painted (alpha=50).
        Each body-range bin is painted at high opacity so the buy/sell signal
        reads clearly over the faint base.

        Color convention (independent of red-up toggle):
            gold   (_UP)   = buyer-dominant bin  (#ffa005)
            purple (_DOWN) = seller-dominant bin (#ab47bc)
        Alpha = f(volume density, directional imbalance): 60 … 230.
        Wick-range bins are skipped entirely for a clean look.
        """
        price_range = high - low
        if price_range < 1e-9:
            return
        bins  = self.HEATMAP_BINS
        bin_h = price_range / bins

        buys   = np.zeros(bins)
        sells  = np.zeros(bins)
        totals = np.zeros(bins)
        for price, counts in pd_.items():
            b = max(0, min(bins - 1, int((price - low) / bin_h)))
            buys[b]   += counts.get("buy", 0)
            sells[b]  += counts.get("sell", 0)
            totals[b] += (counts.get("buy", 0) + counts.get("sell", 0)
                          + counts.get("neutral", 0))

        max_total = float(totals.max()) if totals.max() > 0 else 1.0
        p.setPen(Qt.PenStyle.NoPen)

        for b in range(bins):
            bin_lo = low + b * bin_h
            bin_hi = bin_lo + bin_h

            # Skip bins entirely outside the body (no wick heatmap)
            if bin_hi <= body_lo or bin_lo >= body_hi:
                continue
            draw_lo = max(bin_lo, body_lo)
            draw_hi = min(bin_hi, body_hi)
            if draw_hi <= draw_lo:
                continue

            vol_frac = min(totals[b] / max_total, 1.0)

            if totals[b] == 0:
                continue   # fully empty bin — leave faint base body visible

            total_dir = buys[b] + sells[b]
            if total_dir > 0:
                ratio = (buys[b] - sells[b]) / total_dir   # -1 (pure sell)..+1 (pure buy)
                col   = QColor(_UP if ratio >= 0 else _DOWN)   # gold / purple
                # Alpha: baseline 60, up to 230; scales with volume density AND
                # directional imbalance so pure-direction, high-volume bins are opaque.
                imbalance = abs(ratio)                      # 0..1
                alpha = int(min(230, 60 + 170 * vol_frac * (0.4 + 0.6 * imbalance)))
            else:
                # Only neutral volume — draw light grey at reduced opacity
                col   = QColor(_GREY)
                alpha = int(min(80, 30 + 50 * vol_frac))

            col.setAlpha(alpha)
            p.setBrush(QBrush(col))
            p.drawRect(QRectF(i - 0.35, draw_lo, 0.7, draw_hi - draw_lo))

    def paint(self, p: QPainter, *args) -> None:
        if self._klines is None or self._klines.empty:
            return
        if self._picture is None:
            self._build()
        self._picture.play(p)

    def boundingRect(self) -> QRectF:
        return self._rect


class FvgItem(pg.GraphicsObject):
    """Draws FVG (Fair Value Gap) zones as translucent rectangles."""

    def __init__(self):
        super().__init__()
        self._gaps:    list[dict] = []
        self._n:       int        = 0
        self._red_up:  bool       = False
        self._picture: QPicture | None = None
        self._rect     = QRectF()

    def set_data(self, fvg_gaps: list[dict], n_bars: int, red_up: bool = False) -> None:
        self._gaps    = fvg_gaps
        self._n       = n_bars
        self._red_up  = red_up
        self._picture = None
        self.prepareGeometryChange()
        self.update()

    def _build(self) -> None:
        pic = QPicture()
        p   = QPainter(pic)
        # Diagonal-stripe brushes make FVG zones visually distinct from OB boxes
        # (which use solid fills).  FDiag = forward-slash lines (bull ↑);
        # BDiag = back-slash lines (bear ↓).  Colors follow red_up toggle.
        bull_col   = _RED   if self._red_up else _GREEN
        bear_col   = _GREEN if self._red_up else _RED
        bull_brush = QBrush(_qc(bull_col, 180), Qt.BrushStyle.FDiagPattern)
        bear_brush = QBrush(_qc(bear_col, 180), Qt.BrushStyle.BDiagPattern)
        bull_pen   = QPen(_qc(bull_col, 140), 0)
        bear_pen   = QPen(_qc(bear_col, 140), 0)

        for g in self._gaps:
            if g.get("filled", False):
                continue   # skip — price already closed inside the zone

            x0   = float(g["idx"])
            top  = float(g["top"])
            bot  = float(g["bottom"])
            bull = g.get("direction", "bull") == "bull"

            p.setPen(bull_pen   if bull else bear_pen)
            p.setBrush(bull_brush if bull else bear_brush)
            width = self._n - x0
            if width > 0:
                p.drawRect(QRectF(x0, bot, width, top - bot))
        p.end()
        self._picture = pic

    def paint(self, p: QPainter, *args) -> None:
        if not self._gaps:
            return
        if self._picture is None:
            self._build()
        self._picture.play(p)

    def boundingRect(self) -> QRectF:
        if not self._gaps:
            return QRectF()
        tops = [g["top"]    for g in self._gaps]
        bots = [g["bottom"] for g in self._gaps]
        return QRectF(0, min(bots), self._n, max(tops) - min(bots))


class ObItem(pg.GraphicsObject):
    """Draws Order Block zones with colour-coded subtypes.

    Colours:
        regular    → bull=#26a69a / bear=#ef5350 (solid border, faint fill)
        mitigation → amber (#ffa726) fill
        breaker    → grey (#9e9e9e) fill, dashed border
    """

    def __init__(self):
        super().__init__()
        self._blocks:  list[dict] = []
        self._n:       int        = 0
        self._picture: QPicture | None = None
        self._rect     = QRectF()

    def set_data(self, blocks: list[dict], n_bars: int) -> None:
        self._blocks  = blocks
        self._n       = n_bars
        self._picture = None
        self.prepareGeometryChange()
        self.update()

    def _build(self) -> None:
        pic = QPicture()
        p   = QPainter(pic)

        for blk in self._blocks:
            bull    = blk["direction"] == "bull"
            subtype = blk.get("subtype", "regular")
            x0      = float(blk.get("idx", 0))
            top     = float(blk["top"])
            bot     = float(blk["bottom"])

            if subtype == "breaker":
                fill_color = _qc(_OB_BREAK, 50)
                pen_color  = _qc(_OB_BREAK, 160)
                pen_style  = Qt.PenStyle.DashLine
            elif subtype == "mitigation":
                fill_color = _qc(_OB_MIT, 55)
                pen_color  = _qc(_OB_MIT, 180)
                pen_style  = Qt.PenStyle.DotLine
            else:  # regular
                base       = _OB_BULL if bull else _OB_BEAR
                fill_color = _qc(base, 40)
                pen_color  = _qc(base, 200)
                pen_style  = Qt.PenStyle.SolidLine

            p.setPen(QPen(pen_color, 0, pen_style))
            p.setBrush(QBrush(fill_color))

            width = self._n - x0
            if width > 0:
                p.drawRect(QRectF(x0, bot, width, top - bot))

            pass  # labels removed — see OB legend in chart top-left

        p.end()
        self._picture = pic

    def paint(self, p: QPainter, *args) -> None:
        if not self._blocks:
            return
        if self._picture is None:
            self._build()
        self._picture.play(p)

    def boundingRect(self) -> QRectF:
        if not self._blocks:
            return QRectF()
        tops = [b["top"]    for b in self._blocks]
        bots = [b["bottom"] for b in self._blocks]
        return QRectF(0, min(bots), self._n, max(tops) - min(bots))


# ── Background data-fetch worker ──────────────────────────────────────────────

class DataFetcher(QThread):
    """Fetches klines + ticks + SMC signals in a background thread."""

    ready = pyqtSignal(object)   # emits a dict of results
    error = pyqtSignal(str)

    def __init__(self, ctx, params: dict):
        super().__init__()
        self._ctx    = ctx
        self._params = params

    def run(self) -> None:
        p = self._params
        try:
            code        = p["code"]
            tf          = p["tf"]
            historical  = p["historical"]
            date_str    = p["date_str"]
            candle_mins = p["candle_mins"]
            ind         = p["ind"]

            # Determine fetch window
            if historical:
                dt       = datetime.strptime(date_str, "%Y-%m-%d")
                lb       = _HIST_LOOKBACK_DAYS.get(tf, 8)
                end_dt   = dt + timedelta(days=3)
                start    = (dt - timedelta(days=lb)).strftime("%Y-%m-%d 20:00:00")
                end      = f"{end_dt.strftime('%Y-%m-%d')} 23:59:59"
            else:
                # Use a future end date so that bars whose time_key is in a
                # different calendar date than local time (e.g. US ET bars on
                # Monday morning showing as 06-01 while local clock is still
                # 05-31 Beijing) are never excluded by the end boundary.
                end   = (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d 23:59:59")
                lb    = _LIVE_LOOKBACK_DAYS.get(tf, 10)
                start = (datetime.now() - timedelta(days=lb)).strftime(
                    "%Y-%m-%d %H:%M:%S")

            # Paginate via page_req_key — a single max_count=2000 call silently
            # truncates to the first 2000 bars from `start` when the requested
            # range holds more (e.g. 1m over 5 days with near-24h sessions is
            # up to ~7200 bars), dropping the most recent days including
            # "today" entirely. Mirrors feeds/fetcher.py's paginated fetch.
            ktype, _ = TIMEFRAME_MAP[tf]
            frames: list[pd.DataFrame] = []
            page_req_key = None
            while True:
                if page_req_key is None:
                    ret, page_df, page_req_key = self._ctx.request_history_kline(
                        code, start=start, end=end, ktype=ktype,
                        autype=AuType.NONE, max_count=2000, extended_time=True,
                        session=Session.ALL,
                    )
                else:
                    ret, page_df, page_req_key = self._ctx.request_history_kline(
                        code, ktype=ktype, autype=AuType.NONE, max_count=2000,
                        extended_time=True, session=Session.ALL,
                        page_req_key=page_req_key,
                    )
                if ret != RET_OK:
                    self.error.emit(f"Kline fetch failed: {page_df}")
                    return
                if page_df is not None and not page_df.empty:
                    frames.append(page_df)
                if not page_req_key:
                    break
            df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            if df.empty:
                self.error.emit(
                    f"Kline fetch returned no data for {code} {tf} "
                    f"({start[:10]} → {end[:10]}). "
                    "Check moomoo data subscription or try a higher timeframe."
                )
                return

            # In Live mode, supplement with get_cur_kline so that bars from the
            # current session (which request_history_kline may not yet include)
            # are also visible.  Requires that the symbol is already subscribed.
            #
            # This call was observed silently failing (bare except: pass) for
            # long stretches -- the chart would then miss newly-completed
            # bars entirely until the user manually disconnected and
            # reconnected (which re-subscribes and appears to restore it).
            # Surface ret!=RET_OK / exceptions as a warning (not fatal --
            # request_history_kline's own data is still usable) so the next
            # occurrence is diagnosable from the log instead of just "chart
            # stopped updating, no idea why".
            if not historical:
                try:
                    r2, cur_df = self._ctx.get_cur_kline(
                        code, 200, ktype, AuType.NONE)
                    if r2 == RET_OK and cur_df is not None and not cur_df.empty:
                        hist_keys = set(df["time_key"].astype(str))
                        cur_new   = cur_df[
                            ~cur_df["time_key"].astype(str).isin(hist_keys)]
                        if not cur_new.empty:
                            cur_new = cur_new.reindex(columns=df.columns)
                            df = pd.concat([df, cur_new], ignore_index=True)
                            df = df.sort_values("time_key").reset_index(drop=True)
                    elif r2 != RET_OK:
                        self.error.emit(f"get_cur_kline warning: {cur_df}")
                except Exception as exc:
                    self.error.emit(f"get_cur_kline warning: {exc}")

            # K_DAY time_key arrives as "YYYY-MM-DD" (no time component).
            # Normalise to "YYYY-MM-DD 00:00:00" so all downstream [:16] slices
            # and strptime("%Y-%m-%d %H:%M") calls work identically for every TF.
            if tf == "1d":
                df = df.copy()
                df["time_key"] = df["time_key"].astype(str).apply(
                    lambda s: s + " 00:00:00" if len(s) == 10 else s
                )

            # SMC detection on last N warmup bars
            warmup_n = min(len(df), 400)
            warmup   = df.iloc[-warmup_n:].reset_index(drop=True)

            smc_signals: list[dict] = []
            if True:  # always compute — checkbox only controls display
                smc_signals = detect_bos_choch(
                    warmup,
                    max_span_bars=_BOS_MAX_SPAN.get(tf),
                    trend_window=_TREND_WINDOW.get(tf, 20),
                    filter_choch=False,          # viewer shows all CHoCH, no displacement filter
                    max_session_gap=_BOS_SESSION_GAP.get(tf),  # respect chart.json session gap
                )

            # Index offset: warmup = df[-warmup_n:] re-indexed from 0.
            # warmup index i maps to full-df index (i + disp_off).
            # ALL overlay items (FVG, BOS, OB) must add this offset so they
            # are drawn at the correct x position on the chart.
            disp_off = len(df) - warmup_n   # 0 when all bars fit in warmup

            # Always compute FVG/OB — checkboxes only control display visibility.
            fvg_gaps: list[dict] = []
            raw_fvgs = detect_fvg(
                warmup,
                require_displacement=False,
                min_gap_pct=p.get("fvg_min_gap_pct", 0.001),
            )
            for g in raw_fvgs:
                if g.get("filled", False):
                    continue
                r = dict(g)
                r["idx"] = max(0, g["idx"] + disp_off)
                fvg_gaps.append(r)

            # Apply disp_off to BOS/CHoCH signal indices so they render at
            # the correct x position relative to the full kline DataFrame.
            if disp_off > 0:
                adj_smc: list[dict] = []
                for s in smc_signals:
                    r = dict(s)
                    r["idx"]      = max(0, s["idx"]      + disp_off)
                    r["from_idx"] = max(0, s.get("from_idx", s["idx"]) + disp_off)
                    adj_smc.append(r)
                smc_signals = adj_smc

            ob_blocks: list[dict] = []
            if smc_signals:
                # detect_order_blocks uses warmup-relative indices; add disp_off.
                raw_obs = detect_order_blocks(warmup, [
                    dict(s, idx=s["idx"] - disp_off,
                         from_idx=s.get("from_idx", s["idx"]) - disp_off)
                    for s in smc_signals
                ] if disp_off > 0 else smc_signals,
                    max_count=p.get("ob_max_count", 4),
                )
                for b in raw_obs:
                    r = dict(b)
                    r["idx"]     = max(0, b["idx"]      + disp_off)
                    r["bos_idx"] = max(0, b.get("bos_idx", b["idx"]) + disp_off)
                    ob_blocks.append(r)

            # Tick data
            ticks: dict | None = None
            if historical:
                ticks = load_local_ticks(code, date_str, tf)
            else:
                # Live mode: load today's ticks from DB first (covers the whole
                # trading day even when the viewer is opened after hours), then
                # use the ticks accumulated in this session to fill any buckets
                # ticks.db doesn't have yet. ticks.db is kept fresh by an
                # external collector on the same feed, so a bucket already
                # present there is authoritative — adding the live snapshot on
                # top would double-count the same real trades.
                today = datetime.now().strftime("%Y-%m-%d")
                ticks = load_local_ticks(code, today, tf) or {}
                live_snap: dict = p.get("live_ticks") or {}
                for bk, pd_ in live_snap.items():
                    if bk not in ticks:
                        ticks[bk] = {price: dict(counts) for price, counts in pd_.items()}

            self.ready.emit({
                "klines":      df,
                "warmup":      warmup,
                "smc_signals": smc_signals,
                "fvg_gaps":    fvg_gaps,
                "ob_blocks":   ob_blocks,
                "ticks":       ticks,
                "candle_mins": candle_mins,
                "historical":  historical,
                "date_str":    date_str,
                "tf":          tf,
                "code":        code,
            })
        except Exception as exc:
            self.error.emit(str(exc))


# ── Background option-chain fetch worker ────────────────────────────────────────

class OptionChainFetcher(QThread):
    """Fetches near-expiry option chain data, computes GEX walls, and detects
    single-contract volume spikes since the last poll.

    prev_volumes is a read-only snapshot passed in by the caller -- diffing
    happens entirely inside this thread so there is no shared mutable state
    between this worker and the main-thread cache it will replace on ready.
    """

    ready = pyqtSignal(object)   # emits a dict of results
    error = pyqtSignal(str)

    def __init__(self, ctx, code: str, dte: int, threshold: int,
                 prev_volumes: dict[str, int], wall_count: int = 1):
        super().__init__()
        self._ctx          = ctx
        self._code         = code
        self._dte          = dte
        self._threshold    = threshold
        self._prev_volumes = prev_volumes
        self._wall_count   = wall_count

    def run(self) -> None:
        try:
            expiries = get_near_expiry_dates(self._ctx, self._code, self._dte)
            if not expiries:
                self.error.emit(
                    f"No option expiries within {self._dte} DTE for {self._code}.")
                return

            df = fetch_option_chain_data(self._ctx, self._code, expiries)

            ret, snap = self._ctx.get_market_snapshot([self._code])
            if ret != RET_OK or snap is None or snap.empty:
                self.error.emit(f"Failed to fetch spot price for {self._code}: {snap}")
                return
            spot = float(snap.iloc[0]["last_price"])

            df, by_strike = compute_gex(df, spot)
            # build_gex_stats() computes zero_gamma internally (it's part of
            # the returned stats dict) -- calling it separately here would
            # just redo the same BS-repricing grid search a second time.
            stats = build_gex_stats(df, by_strike, spot, 0.0, expiries)
            call_walls, put_walls = top_gex_walls(by_strike, self._wall_count)

            # Large-trade detection: diff this poll's per-contract volume
            # against the previous poll's. A contract missing from
            # prev_volumes (first poll after enabling/symbol change) has no
            # baseline -- never alert on it, that would just be "today's
            # entire volume so far", not a spike.
            large_trades: list[dict] = []
            volumes: dict[str, int] = {}
            if "volume" in df.columns:
                for _, row in df.iterrows():
                    code    = row["code"]
                    vol_raw = pd.to_numeric(row["volume"], errors="coerce")
                    vol     = int(vol_raw) if pd.notna(vol_raw) else 0
                    volumes[code] = vol
                    prev = self._prev_volumes.get(code)
                    if prev is None:
                        continue
                    delta = vol - prev
                    if delta >= self._threshold:
                        large_trades.append({
                            "code":         code,
                            "strike":       float(row["option_strike_price"]),
                            "option_type":  row["option_type"],
                            "volume_delta": delta,
                            "computed_at":  datetime.now(),
                        })

            self.ready.emit({
                "call_walls":   call_walls,   # [{"strike":, "gex":}, ...] strongest first
                "put_walls":    put_walls,
                "zero_gamma":   stats["zero_gamma"],
                "net_gex":      stats["net_gex"],
                "expiries":     expiries,
                "spot":         spot,
                "computed_at":  datetime.now(),
                "large_trades": large_trades,
                "volumes":      volumes,
            })
        except Exception as exc:
            self.error.emit(str(exc))


class ChandelierParamsDialog(QDialog):
    """ATR period / multiplier / direction for the chandelier-exit corner label.

    Values are only applied on OK; Cancel leaves the caller's current settings
    untouched. See strategy/chandelier_exit/README.md for the underlying formula.
    """

    def __init__(self, period: int, multiplier: float, direction: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Chandelier Exit Parameters")

        self._period_spin = QSpinBox()
        self._period_spin.setRange(2, 200)
        self._period_spin.setValue(period)

        self._mult_spin = QDoubleSpinBox()
        self._mult_spin.setRange(0.5, 10.0)
        self._mult_spin.setSingleStep(0.1)
        self._mult_spin.setDecimals(1)
        self._mult_spin.setValue(multiplier)

        self._dir_combo = QComboBox()
        self._dir_combo.addItems(["Long", "Short"])
        self._dir_combo.setCurrentIndex(0 if direction == "bull" else 1)

        form = QFormLayout()
        form.addRow("ATR period:", self._period_spin)
        form.addRow("ATR multiplier:", self._mult_spin)
        form.addRow("Direction:", self._dir_combo)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def values(self) -> tuple[int, float, str]:
        """Return (period, multiplier, direction) -- direction is "bull"/"bear"."""
        direction = "bull" if self._dir_combo.currentIndex() == 0 else "bear"
        return self._period_spin.value(), self._mult_spin.value(), direction


_HANG_LOG_PATH = pathlib.Path(__file__).parent.parent / "hang_watchdog.log"


def _close_with_hang_watchdog(window: QWidget, name: str, timeout: float = 1.0) -> None:
    """Close a child window, always logging how long close() took; if it
    doesn't return within `timeout` seconds, also dump every thread's
    Python stack to hang_watchdog.log.

    Reported symptom: the main viewer stutters/goes unresponsive after
    closing the Liquidity Heatmap window, more often the longer it had been
    running. closeEvent()/worker teardown there look non-blocking by design
    (see _retire_worker in liq_hm_window.py), but nothing here rules out a
    SQLite lock wait (e.g. contending with tick_collector.py's periodic WAL
    checkpoint/prune) or a slow Qt/pyqtgraph teardown actually blocking the
    GUI thread inside close(). A `threading.Timer` runs on its own OS thread
    independent of the (possibly frozen) Qt event loop, so it still fires and
    captures real stack frames even while the main thread is stuck.
    timeout is 1s (not e.g. 3s) because the reported symptom is a stutter,
    not necessarily a multi-second freeze -- always logging the actual
    elapsed time (below) means even a sub-threshold stutter leaves a number
    in the log instead of no evidence at all.
    """
    def _dump() -> None:
        try:
            with open(_HANG_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(f"\n=== {datetime.now().isoformat()} "
                        f"{name}.close() exceeded {timeout}s ===\n")
                faulthandler.dump_traceback(file=f, all_threads=True)
        except Exception:
            pass

    timer = threading.Timer(timeout, _dump)
    timer.start()
    t0 = time.monotonic()
    try:
        window.close()
    finally:
        timer.cancel()
        elapsed = time.monotonic() - t0
        try:
            with open(_HANG_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(f"{datetime.now().isoformat()} "
                        f"{name}.close() took {elapsed:.3f}s\n")
        except Exception:
            pass


# ── Main window ───────────────────────────────────────────────────────────────

class TradeViewerQt(QMainWindow):
    """PyQtGraph-based trade viewer window."""

    def __init__(self, args=None):
        super().__init__()
        self.setWindowTitle("Trade Viewer Qt")
        self.resize(1680, 950)

        # moomoo API context
        host = getattr(args, "host", "127.0.0.1")
        port = getattr(args, "port", 11111)
        self._ctx: OpenQuoteContext | None = None
        self._ctx_lock = threading.Lock()

        # State
        self._klines:       pd.DataFrame | None = None
        self._warmup:       pd.DataFrame | None = None
        self._ticks:        dict | None         = None
        self._live_ticks:   dict                = defaultdict(
            lambda: defaultdict(lambda: {"buy": 0, "sell": 0, "neutral": 0}))
        self._tick_lock      = threading.Lock()
        self._last_tick_price: float            = 0.0
        self._last_nbbo:      tuple[float, float] = (0.0, 0.0)
        self._fetcher:      DataFetcher | None  = None
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._trigger_fetch)
        self._candle_mins   = 1
        self._smc_signals:  list[dict]          = []
        self._fvg_gaps:     list[dict]          = []
        self._ob_blocks:    list[dict]          = []
        self._liq_hm_window: "LiqHmWindow | None" = None
        self._dom_window:  QWidget | None      = None  # DOM depth-of-market window
        self._trade_record: dict | None         = None  # active trade review
        self._live_code: str = ""        # code currently subscribed for tick push
        self._chandelier_period:     int   = 20
        self._chandelier_multiplier: float = 2.0
        self._chandelier_direction:  str   = "bull"   # "bull" | "bear"
        self._live_kl_sub: object = None  # kline SubType subscribed alongside TICKER/QUOTE
        # Track which (code, tf) was last auto-ranged; prevents live-refresh
        # from resetting the user's manual pan/zoom on every tick.
        self._last_chart_key: tuple             = ("", "")

        # Option Walls / Option Alerts: own OpenQuoteContext (avoids
        # concurrent-access uncertainty on self._ctx, which DataFetcher's
        # background thread also calls into) and own poll timer, decoupled
        # from the K-line refresh cycle since option-chain fetches are much
        # slower and only need to run every several seconds, not per tick.
        self._option_ctx: OpenQuoteContext | None = None
        self._option_fetcher: OptionChainFetcher | None = None
        self._option_timer = QTimer(self)
        self._option_timer.timeout.connect(self._trigger_option_fetch)
        self._option_wall_data: dict | None = None
        self._option_prev_volumes: dict[str, int] = {}
        self._option_code: str = ""   # symbol the cached option state belongs to
        self._tray: QSystemTrayIcon | None = None   # built lazily on first alert

        # Range Profile state
        self._range_region:          pg.LinearRegionItem | None = None
        self._range_profile_inline:  list                       = []
        self._range_last_indices:    tuple[int, int]            = (-1, -1)
        self._range_profile_timer    = QTimer(self)
        self._range_profile_timer.setSingleShot(True)
        self._range_profile_timer.timeout.connect(self._rebuild_range_profile)
        # Label-pinning connections on the profile ViewBox — cleared on each rebuild
        self._profile_pin_conns:     list                       = []

        # Build UI
        self._build_toolbar(args)
        self._build_central()
        self._build_statusbar()

        # Apply dark theme
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{ background: {_BG}; color: {_FG}; }}
            QToolBar {{ background: {_BG_BAR}; border: none; spacing: 4px; }}
            QLabel {{ color: {_FG}; }}
            QComboBox, QLineEdit, QSpinBox {{
                background: {_BG_BAR}; color: {_FG};
                border: 1px solid {_GREY}; padding: 2px 4px;
            }}
            QPushButton {{
                background: {_BG_BAR}; color: {_FG};
                border: 1px solid {_GREY}; padding: 3px 8px;
            }}
            QPushButton:hover {{ background: {_BG_TIP}; }}
            QPushButton:checked {{ background: {_BG_TIP}; border-color: {_GOLD}; }}
            QCheckBox {{ color: {_FG}; spacing: 4px; }}
            QRadioButton {{ color: {_FG}; spacing: 4px; }}
        """)

        # Auto-connect if args provided
        if args:
            self._connect_opend(host, port)

    # ── Toolbars (six rows) ───────────────────────────────────────────────────

    def _build_toolbar(self, args) -> None:
        def _lbl(text: str) -> QLabel:
            l = QLabel(text)
            l.setStyleSheet(f"color: {_FG}; font-size: 11px; padding: 0 2px;")
            return l

        toolbar_rows: list[QToolBar] = []

        # ── Row 1: core data controls ─────────────────────────────────────────
        tb1 = QToolBar("Controls", self)
        tb1.setMovable(False)
        tb1.setFloatable(False)
        self.addToolBar(tb1)
        toolbar_rows.append(tb1)

        # Code
        tb1.addWidget(_lbl("Code:"))
        _fb = _default_code()
        self._code_edit = QLineEdit(getattr(args, "code", _fb) or _fb)
        self._code_edit.setFixedWidth(90)
        self._code_edit.returnPressed.connect(self._trigger_fetch)
        # Event filter: catches Enter key before QToolBar can consume it on Windows.
        # returnPressed alone is unreliable inside a QToolBar on some Qt/Windows versions.
        self._code_edit.installEventFilter(self)
        tb1.addWidget(self._code_edit)

        tb1.addSeparator()

        # Timeframe
        tb1.addWidget(_lbl("TF:"))
        self._tf_combo = QComboBox()
        self._tf_combo.addItems(list(TIMEFRAME_MAP.keys()))
        self._tf_combo.setCurrentText(getattr(args, "tf", "5m") or "5m")
        self._tf_combo.currentTextChanged.connect(self._on_tf_changed)
        tb1.addWidget(self._tf_combo)

        tb1.addSeparator()

        # Mode
        tb1.addWidget(_lbl("Mode:"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Live", "Historical"])
        init_mode = getattr(args, "mode", "Live") or "Live"
        self._mode_combo.setCurrentText(init_mode)
        self._mode_combo.currentTextChanged.connect(self._on_mode_changed)
        tb1.addWidget(self._mode_combo)

        tb1.addSeparator()

        # Date (Historical)
        tb1.addWidget(_lbl("Date:"))
        init_date = getattr(args, "date", None) or datetime.now().strftime("%Y-%m-%d")
        self._date_edit = QLineEdit(init_date)
        self._date_edit.setFixedWidth(90)
        self._date_edit.setEnabled(init_mode == "Historical")
        self._date_edit.returnPressed.connect(self._trigger_fetch)
        tb1.addWidget(self._date_edit)

        tb1.addSeparator()

        # Load button
        load_btn = QPushButton("Load")
        load_btn.setToolTip("Fetch and render chart now")
        load_btn.clicked.connect(self._trigger_fetch)
        tb1.addWidget(load_btn)

        home_btn = QPushButton("⌂")
        home_btn.setFixedWidth(28)
        home_btn.setToolTip("Reset zoom/pan to last 150 bars  (Home key)")
        home_btn.clicked.connect(self._on_home)
        tb1.addWidget(home_btn)

        tb1.addSeparator()

        self._pin_btn = QPushButton("📌 Pin")
        self._pin_btn.setCheckable(True)
        self._pin_btn.setFixedWidth(60)
        self._pin_btn.setToolTip("Keep this window on top of all other windows")
        self._pin_btn.toggled.connect(self._on_pin_toggled)
        tb1.addWidget(self._pin_btn)

        tb1.addSeparator()

        # Refresh (Live)
        tb1.addWidget(_lbl("Refresh (s, min 5):"))
        self._refresh_spin = QSpinBox()
        self._refresh_spin.setRange(5, 300)
        self._refresh_spin.setValue(getattr(args, "refresh", 15) or 15)
        self._refresh_spin.setFixedWidth(55)
        self._refresh_spin.valueChanged.connect(self._on_refresh_changed)
        tb1.addWidget(self._refresh_spin)

        tb1.addSeparator()

        # Connect / Stop
        self._conn_btn = QPushButton("Connect")
        self._conn_btn.setCheckable(True)
        self._conn_btn.clicked.connect(self._on_connect_toggle)
        tb1.addWidget(self._conn_btn)

        self._ind_checks: dict[str, QCheckBox | QRadioButton] = {}
        _default_checked = ("heatmap", "delta", "bos_choch", "cvd")

        # ── Row 2: SMC structural indicators ────────────────────────────────
        self.addToolBarBreak()
        tb_smc = QToolBar("SMC", self)
        tb_smc.setMovable(False)
        tb_smc.setFloatable(False)
        self.addToolBar(tb_smc)
        toolbar_rows.append(tb_smc)

        tb_smc.addWidget(_lbl("SMC:"))
        for key, label in [
            ("bos_choch", "BOS/CHoCH"),
            ("fvg",       "FVG"),
            ("ob",        "OB"),
        ]:
            cb = QCheckBox(label)
            cb.setChecked(key in _default_checked)
            cb.stateChanged.connect(self._on_indicator_toggle)
            self._ind_checks[key] = cb
            tb_smc.addWidget(cb)
            if key == "fvg":
                self._fvg_min_pct = QDoubleSpinBox()
                self._fvg_min_pct.setRange(0.05, 5.0)
                self._fvg_min_pct.setSingleStep(0.05)
                self._fvg_min_pct.setValue(1.00)
                self._fvg_min_pct.setDecimals(2)
                self._fvg_min_pct.setSuffix("%")
                self._fvg_min_pct.setFixedWidth(72)
                self._fvg_min_pct.setEnabled(False)
                self._fvg_min_pct.setToolTip("FVG minimum gap size (% of price)")
                self._fvg_min_pct.valueChanged.connect(self._trigger_fetch)
                tb_smc.addWidget(self._fvg_min_pct)
            if key == "ob":
                self._ob_max_count = QSpinBox()
                self._ob_max_count.setRange(1, 20)
                self._ob_max_count.setValue(4)
                self._ob_max_count.setFixedWidth(50)
                self._ob_max_count.setEnabled(False)
                self._ob_max_count.setToolTip("Max number of recent OBs to display")
                self._ob_max_count.valueChanged.connect(self._trigger_fetch)
                tb_smc.addWidget(self._ob_max_count)

        # ── Row 3: other indicators / overlays ──────────────────────────────
        self.addToolBarBreak()
        tb_ind = QToolBar("Indicators", self)
        tb_ind.setMovable(False)
        tb_ind.setFloatable(False)
        self.addToolBar(tb_ind)
        toolbar_rows.append(tb_ind)

        tb_ind.addWidget(_lbl("Indicators:"))
        for key, label in [
            ("heatmap",   "Heatmap"),
            ("delta",     "Δ Delta"),
            ("kd_band",   "KD"),    # KD fast/slow midline ribbon on main chart
            ("kd",        "KDV"),   # KDV = KD spread-width subplot
            ("cvd",       "CVD"),   # Cumulative Volume Delta subplot
            ("adi",       "A/D"),   # Accumulation/Distribution Line subplot
            ("ema",       "EMA"),
            ("avwap",     "AVWAP"), # Anchored VWAP -- cumulative from a user-picked date
            ("vol",       "MAVOL"), # Volume subplot toggle
            ("chandelier", "Chandelier"), # ATR trailing-stop suggestion, bottom-right corner label
        ]:
            cb = QCheckBox(label)
            cb.setChecked(key in _default_checked)
            cb.stateChanged.connect(self._on_indicator_toggle)
            self._ind_checks[key] = cb
            tb_ind.addWidget(cb)
            if key == "avwap":
                self._avwap_anchor_edit = QLineEdit()
                self._avwap_anchor_edit.setPlaceholderText("YYYY-MM-DD")
                self._avwap_anchor_edit.setFixedWidth(80)
                self._avwap_anchor_edit.setToolTip(
                    "Anchor date for the VWAP (blank = start of the current trading\n"
                    "day, 20:00 ET -> next 20:00 ET, same as CVD's reset boundary).\n"
                    "VWAP accumulates from the first bar on/after this date.\n"
                    "Accepts YYYY-MM-DD, MM/DD/YYYY, YYYY/MM/DD, or MM-DD-YYYY -- "
                    "an unrecognised date shows a red warning instead of silently "
                    "using the whole chart.")
                self._avwap_anchor_edit.returnPressed.connect(
                    lambda: self._render(self._klines, self._ticks) if self._klines is not None else None)
                tb_ind.addWidget(self._avwap_anchor_edit)
                self._avwap_sigma_spin = QSpinBox()
                self._avwap_sigma_spin.setRange(0, 3)
                self._avwap_sigma_spin.setValue(0)
                self._avwap_sigma_spin.setFixedWidth(36)
                self._avwap_sigma_spin.setToolTip(
                    "Number of volume-weighted standard-deviation bands to draw\n"
                    "around the AVWAP (0 = none). Each band k is AVWAP +-k*stddev,\n"
                    "stddev computed the same way TradingView's VWAP bands are:\n"
                    "cumulative volume-weighted variance of typical price from the\n"
                    "anchor bar forward.\n"
                    "Under a normal distribution: +-1sigma ~= 68.3%, +-2sigma ~= 95.4%,\n"
                    "+-3sigma ~= 99.7% of observations fall inside that band --\n"
                    "approximate, since price returns aren't perfectly normal.")
                self._avwap_sigma_spin.valueChanged.connect(
                    lambda: self._render(self._klines, self._ticks) if self._klines is not None else None)
                tb_ind.addWidget(_lbl("sigma:"))
                tb_ind.addWidget(self._avwap_sigma_spin)
            if key == "chandelier":
                chandelier_settings_btn = QPushButton("⚙")
                chandelier_settings_btn.setFixedWidth(28)
                chandelier_settings_btn.setToolTip(
                    "Chandelier exit parameters (ATR period, multiplier, direction)")
                chandelier_settings_btn.clicked.connect(self._on_chandelier_settings)
                tb_ind.addWidget(chandelier_settings_btn)

        tb_ind.addSeparator()

        # Color scheme toggle
        cb_red_up = QCheckBox("Red Up")
        cb_red_up.setChecked(True)
        cb_red_up.setToolTip(
            "Checked = red rises, green falls (CN convention)\n"
            "Unchecked = green rises, red falls (Western convention)")
        cb_red_up.stateChanged.connect(self._on_indicator_toggle)
        self._ind_checks["red_up"] = cb_red_up
        tb_ind.addWidget(cb_red_up)

        # ── Row 4: session filters + profile date range ─────────────────────
        self.addToolBarBreak()
        tb_sess = QToolBar("Session", self)
        tb_sess.setMovable(False)
        tb_sess.setFloatable(False)
        self.addToolBar(tb_sess)
        toolbar_rows.append(tb_sess)

        tb_sess.addWidget(_lbl("Session:"))
        for key, label in [
            ("regular", "Regular"), ("pre", "Pre"),
            ("post", "Post"),       ("night", "Night"),
        ]:
            cb = QCheckBox(label)
            cb.setChecked(key == "regular")
            cb.stateChanged.connect(self._on_session_toggle)
            self._ind_checks[f"sess_{key}"] = cb
            tb_sess.addWidget(cb)

        tb_sess.addSeparator()

        # Profile date range
        tb_sess.addWidget(_lbl("Range:"))
        self._range_group = QButtonGroup(self)
        for val, label in [("1d", "1D"), ("2d", "2D"), ("3d", "3D"), ("7d", "1W")]:
            rb = QRadioButton(label)
            rb.setChecked(val == "1d")
            rb.toggled.connect(self._on_range_changed)
            self._ind_checks[f"range_{val}"] = rb
            self._range_group.addButton(rb)
            tb_sess.addWidget(rb)

        tb_sess.addSeparator()
        tb_sess.addWidget(_lbl("Bins:"))
        self._profile_bins_spin = QSpinBox()
        self._profile_bins_spin.setRange(20, 300)
        self._profile_bins_spin.setValue(60)
        self._profile_bins_spin.setFixedWidth(48)
        self._profile_bins_spin.setToolTip(
            "Number of price bins in the session volume profile (right panel) --\n"
            "higher = thinner bars, finer price resolution.")
        self._profile_bins_spin.valueChanged.connect(self._on_range_changed)
        tb_sess.addWidget(self._profile_bins_spin)

        # ── Row 5: order flow controls ────────────────────────────────────────
        self.addToolBarBreak()
        tb3 = QToolBar("Order Flow", self)
        tb3.setMovable(False)
        tb3.setFloatable(False)
        self.addToolBar(tb3)
        toolbar_rows.append(tb3)

        # Tick profile order-size filter
        tb3.addWidget(_lbl("Orders:"))
        for key, label, tip in [
            ("tick_s", "S", "Small orders  (volume < day p33)"),
            ("tick_m", "M", "Medium orders (volume p33–p67)"),
            ("tick_l", "L", "Large orders  (volume > day p67)"),
        ]:
            cb = QCheckBox(label)
            cb.setChecked(True)
            cb.setToolTip(tip)
            cb.stateChanged.connect(self._on_tick_size_toggle)
            self._ind_checks[key] = cb
            tb3.addWidget(cb)

        tb3.addSeparator()
        cb_net = QCheckBox("Net")
        cb_net.setChecked(False)
        cb_net.setToolTip(
            "Tick profile: show a single net (buy − sell) bar per price level\n"
            "instead of separate diverging buy/sell bars")
        cb_net.stateChanged.connect(self._on_tick_size_toggle)
        self._ind_checks["tick_net"] = cb_net
        tb3.addWidget(cb_net)

        tb3.addSeparator()
        tb3.addWidget(_lbl("Imb:"))
        self._tick_imb_ratio = QDoubleSpinBox()
        self._tick_imb_ratio.setRange(1.5, 20.0)
        self._tick_imb_ratio.setSingleStep(0.5)
        self._tick_imb_ratio.setValue(3.0)
        self._tick_imb_ratio.setDecimals(1)
        self._tick_imb_ratio.setFixedWidth(52)
        self._tick_imb_ratio.setToolTip(
            "Tick profile: highlight price levels where the dominant side's\n"
            "volume is at least this many times the opposite side's -- gold\n"
            "outline on the bar, plus the price printed at its tip.")
        self._tick_imb_ratio.valueChanged.connect(self._on_tick_size_toggle)
        tb3.addWidget(self._tick_imb_ratio)

        tb3.addSeparator()

        # Liquidity Heatmap floating window
        self._liq_hm_btn = QPushButton("Liquidity Heatmap")
        self._liq_hm_btn.setCheckable(True)
        self._liq_hm_btn.setChecked(False)
        self._liq_hm_btn.setToolTip(
            "Open Liquidity Heatmap window\n"
            "Real-time resting order book depth as price × time heatmap.\n"
            "Requires order_book_collector.py to be running.")
        self._liq_hm_btn.clicked.connect(self._on_liq_hm_toggle)
        tb3.addWidget(self._liq_hm_btn)

        tb3.addSeparator()

        # DOM window toggle
        self._dom_btn = QPushButton("DOM")
        self._dom_btn.setCheckable(True)
        self._dom_btn.setChecked(False)
        self._dom_btn.setToolTip(
            "Open Depth of Market window\n"
            "Shows resting order book bid/ask depth for the current symbol.\n"
            "Requires order_book_collector.py to be running.")
        self._dom_btn.clicked.connect(self._on_dom_toggle)
        tb3.addWidget(self._dom_btn)

        tb3.addSeparator()

        # Range Volume Profile
        self._range_profile_btn = QPushButton("Range Profile")
        self._range_profile_btn.setCheckable(True)
        self._range_profile_btn.setChecked(False)
        self._range_profile_btn.setToolTip(
            "Toggle Range Volume Profile mode\n"
            "Drag the shaded region on the chart to select a bar range.\n"
            "Volume profile for that range appears in the right panel and\n"
            "as a semi-transparent overlay on the main chart.\n"
            "Uses tick-level data when available; falls back to OHLCV.")
        self._range_profile_btn.clicked.connect(self._toggle_range_profile)
        tb3.addWidget(self._range_profile_btn)

        tb3.addSeparator()
        self._scanner_signals_btn = QPushButton("Scanner Signals")
        self._scanner_signals_btn.setCheckable(True)
        self._scanner_signals_btn.setChecked(False)
        self._scanner_signals_btn.setToolTip(
            "Overlay open scanner signals (entry zone, SL, TP) from db/signals.db.\n"
            "Signals are read from the local SQLite database written by the scanner."
        )
        self._scanner_signals_btn.clicked.connect(self._toggle_scanner_signals)
        tb3.addWidget(self._scanner_signals_btn)

        # ── Row: options (Call/Put Wall + Zero Gamma, large-trade alerts) ────
        self.addToolBarBreak()
        tb_opt = QToolBar("Options", self)
        tb_opt.setMovable(False)
        tb_opt.setFloatable(False)
        self.addToolBar(tb_opt)
        toolbar_rows.append(tb_opt)

        tb_opt.addWidget(_lbl("Options:"))
        for key, label in [
            ("option_walls",  "Option Walls"),
            ("option_alerts", "Option Alerts"),
        ]:
            cb = QCheckBox(label)
            cb.setChecked(False)
            cb.stateChanged.connect(self._on_option_indicator_toggle)
            self._ind_checks[key] = cb
            tb_opt.addWidget(cb)

        tb_opt.addSeparator()

        # Shared controls -- both indicators come from the same poll cycle.
        tb_opt.addWidget(_lbl("DTE:"))
        self._option_dte_spin = QSpinBox()
        self._option_dte_spin.setRange(0, 30)
        self._option_dte_spin.setValue(7)
        self._option_dte_spin.setFixedWidth(40)
        self._option_dte_spin.setToolTip("Max days-to-expiry for the option chain polled")
        tb_opt.addWidget(self._option_dte_spin)

        self._option_0dte_cb = QCheckBox("0DTE")
        self._option_0dte_cb.setToolTip(
            "Force DTE=0 (today's expiry only) -- ignores the DTE spinbox\n"
            "while checked. Useful for watching same-day option structure,\n"
            "which shifts much faster intraday than further-dated expiries.")
        self._option_0dte_cb.stateChanged.connect(self._on_option_0dte_toggle)
        tb_opt.addWidget(self._option_0dte_cb)

        tb_opt.addSeparator()
        tb_opt.addWidget(_lbl("Walls:"))
        self._option_wall_count_spin = QSpinBox()
        self._option_wall_count_spin.setRange(1, 10)
        self._option_wall_count_spin.setValue(1)
        self._option_wall_count_spin.setFixedWidth(36)
        self._option_wall_count_spin.setToolTip(
            "Number of Call Wall / Put Wall lines to draw per side, strongest\n"
            "GEX first (1 = just the single strongest wall each side, matching\n"
            "the original behavior).")
        self._option_wall_count_spin.valueChanged.connect(self._on_option_wall_count_changed)
        tb_opt.addWidget(self._option_wall_count_spin)

        tb_opt.addSeparator()
        tb_opt.addWidget(_lbl("Poll(s):"))
        self._option_interval_spin = QSpinBox()
        self._option_interval_spin.setRange(5, 300)
        self._option_interval_spin.setValue(15)
        self._option_interval_spin.setFixedWidth(45)
        self._option_interval_spin.setToolTip(
            "Seconds between option-chain polls -- drives both Option Walls\n"
            "refresh and Option Alerts detection latency.")
        self._option_interval_spin.valueChanged.connect(self._on_option_interval_changed)
        tb_opt.addWidget(self._option_interval_spin)

        tb_opt.addSeparator()
        tb_opt.addWidget(_lbl("Threshold:"))
        self._option_threshold_spin = QSpinBox()
        self._option_threshold_spin.setRange(1, 100_000)
        self._option_threshold_spin.setValue(500)
        self._option_threshold_spin.setFixedWidth(65)
        self._option_threshold_spin.setToolTip(
            "Option Alerts: minimum single-contract volume increase between\n"
            "two polls to fire an alert (contracts, not dollars).")
        tb_opt.addWidget(self._option_threshold_spin)

        tb_opt.addSeparator()
        option_refresh_btn = QPushButton("↻")
        option_refresh_btn.setFixedWidth(28)
        option_refresh_btn.setToolTip("Refresh option data now, without waiting for the next poll")
        option_refresh_btn.clicked.connect(self._trigger_option_fetch)
        tb_opt.addWidget(option_refresh_btn)

        self._option_status_lbl = QLabel("")
        self._option_status_lbl.setStyleSheet("color: #888;")
        tb_opt.addWidget(self._option_status_lbl)

        # ── Row 6: trade review ───────────────────────────────────────────────
        self.addToolBarBreak()
        tb3 = QToolBar("Trade Review", self)
        tb3.setMovable(False)
        tb3.setFloatable(False)
        self.addToolBar(tb3)
        toolbar_rows.append(tb3)

        tb3.addWidget(_lbl("Trade ID:"))
        self._trade_id_edit = QLineEdit()
        self._trade_id_edit.setPlaceholderText("trade UUID — enter or paste, then press Enter or Review")
        self._trade_id_edit.setMinimumWidth(340)
        self._trade_id_edit.setMaximumWidth(520)
        self._trade_id_edit.returnPressed.connect(self._load_trade_review)
        tb3.addWidget(self._trade_id_edit)
        tb3.addSeparator()
        review_btn = QPushButton("Review")
        review_btn.setToolTip("Load trade entry/exit markers onto the chart")
        review_btn.clicked.connect(self._load_trade_review)
        tb3.addWidget(review_btn)
        clear_btn = QPushButton("Clear")
        clear_btn.setToolTip("Remove all trade review markers")
        clear_btn.clicked.connect(self._clear_trade_review)
        tb3.addWidget(clear_btn)

        # QToolBar's overflow chevron does not reliably engage for rows built
        # from addWidget() (checkboxes/combos/line edits) stacked via
        # addToolBarBreak() -- shrinking the window can silently clip the
        # widest row's trailing controls off the edge instead of collapsing
        # them into a popup. Floor the window width at the widest row's
        # natural size so that state is never reachable.
        widest = max((tb.sizeHint().width() for tb in toolbar_rows), default=0)
        if widest:
            self.setMinimumWidth(widest + 16)

    # ── Central widget: chart + profiles ─────────────────────────────────────

    def _build_central(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        # ── Left: main chart (candles + volume + KD) ─────────────────────────
        self._chart_widget = pg.GraphicsLayoutWidget()
        self._chart_widget.setBackground(_BG)

        # Candle plot (row 0)
        self._plot_c: pg.PlotItem = self._chart_widget.addPlot(row=0, col=0)
        self._plot_c.showGrid(x=True, y=True, alpha=0.15)
        self._plot_c.setLabel("left", "", **{"color": _FG})
        self._plot_c.getAxis("left").setTextPen(_qc(_FG))
        self._plot_c.getAxis("bottom").setTextPen(_qc(_FG))
        self._plot_c.setMenuEnabled(False)

        # Volume plot (row 1)
        self._chart_widget.nextRow()
        self._plot_v: pg.PlotItem = self._chart_widget.addPlot(row=1, col=0)
        self._plot_v.showGrid(x=True, y=True, alpha=0.10)
        self._plot_v.setLabel("left", "MAVOL", **{"color": _FG})
        self._plot_v.getAxis("left").setTextPen(_qc(_FG))
        self._plot_v.getAxis("bottom").setTextPen(_qc(_FG))
        self._plot_v.setMenuEnabled(False)
        self._plot_v.setXLink(self._plot_c)

        # KD subplot (row 2) — hidden until KD indicator enabled
        self._chart_widget.nextRow()
        self._plot_kd: pg.PlotItem = self._chart_widget.addPlot(row=2, col=0)
        self._plot_kd.showGrid(x=True, y=True, alpha=0.10)
        self._plot_kd.setLabel("left", "KDV", **{"color": _FG})
        self._plot_kd.getAxis("left").setTextPen(_qc(_FG))
        self._plot_kd.getAxis("bottom").setTextPen(_qc(_FG))
        self._plot_kd.setMenuEnabled(False)
        self._plot_kd.setXLink(self._plot_c)
        # Add zero line reference
        self._plot_kd.addItem(pg.InfiniteLine(
            pos=0, angle=0, movable=False,
            pen=pg.mkPen(_GREY, width=1, style=Qt.PenStyle.DashLine),
        ))
        # visibility + row-collapse applied below via _set_subplot_row_visible

        # CVD subplot (row 3) — hidden until CVD indicator enabled
        self._chart_widget.nextRow()
        self._plot_cvd: pg.PlotItem = self._chart_widget.addPlot(
            row=3, col=0, axisItems={"left": _VolumeAxisItem(orientation="left", signed=True)})
        self._plot_cvd.showGrid(x=True, y=True, alpha=0.10)
        self._plot_cvd.setLabel("left", "CVD", **{"color": _FG})
        self._plot_cvd.getAxis("left").setTextPen(_qc(_FG))
        self._plot_cvd.getAxis("bottom").setTextPen(_qc(_FG))
        self._plot_cvd.setMenuEnabled(False)
        self._plot_cvd.setXLink(self._plot_c)
        # Add zero line reference
        self._plot_cvd.addItem(pg.InfiniteLine(
            pos=0, angle=0, movable=False,
            pen=pg.mkPen(_GREY, width=1, style=Qt.PenStyle.DashLine),
        ))
        # visibility + row-collapse applied below via _set_subplot_row_visible

        # A/D Line subplot (row 4) — hidden until A/D indicator enabled
        self._chart_widget.nextRow()
        self._plot_adi: pg.PlotItem = self._chart_widget.addPlot(row=4, col=0)
        self._plot_adi.showGrid(x=True, y=True, alpha=0.10)
        self._plot_adi.setLabel("left", "A/D", **{"color": _FG})
        self._plot_adi.getAxis("left").setTextPen(_qc(_FG))
        self._plot_adi.getAxis("bottom").setTextPen(_qc(_FG))
        self._plot_adi.setMenuEnabled(False)
        self._plot_adi.setXLink(self._plot_c)
        # Add zero line reference
        self._plot_adi.addItem(pg.InfiniteLine(
            pos=0, angle=0, movable=False,
            pen=pg.mkPen(_GREY, width=1, style=Qt.PenStyle.DashLine),
        ))
        # visibility + row-collapse applied below via _set_subplot_row_visible

        # Row stretch factors — main chart is fixed; subplot rows collapse to
        # zero height when their indicator is off instead of leaving a gap.
        self._chart_widget.ci.layout.setRowStretchFactor(0, 5)
        self._set_subplot_row_visible(self._plot_v,   1, False)
        self._set_subplot_row_visible(self._plot_kd,  2, False)
        self._set_subplot_row_visible(self._plot_cvd, 3, True)   # CVD on by default
        self._set_subplot_row_visible(self._plot_adi, 4, False)

        # ── Graphics items ────────────────────────────────────────────────────
        self._candle_item = CandlestickItem()
        self._plot_c.addItem(self._candle_item)

        self._fvg_item = FvgItem()
        self._plot_c.addItem(self._fvg_item)

        self._ob_item = ObItem()
        self._plot_c.addItem(self._ob_item)

        self._bos_items:     list = []  # PlotCurveItem + TextItem per signal
        self._delta_items:   list = []  # TextItem per candle
        self._ema_items:     list = []  # PlotCurveItem per EMA period
        self._avwap_items:   list = []  # anchored-VWAP curve + anchor line + label
        self._option_wall_items: list = []  # Call/Put Wall + Zero Gamma lines -- cleared+redrawn each update
        self._option_alert_items: list = []  # large-trade markers -- accumulate, not cleared each update
        self._kd_items:      list = []  # PlotCurveItem + fill for KD subplot
        self._kd_band_items: list = []  # PlotCurveItem + fill for KD band on main chart
        self._cvd_items:     list = []  # PlotCurveItem for CVD subplot
        self._adi_items:     list = []  # PlotCurveItem for A/D Line subplot
        self._trade_items:          list = []  # all trade review overlay items
        self._scanner_signal_items: list = []  # scanner signals overlay items

        # Volume bars + MA curve
        self._vol_item = pg.BarGraphItem(
            x=[], height=[], width=0.7,
            brush=_qc(_GREEN, 100), pen=pg.mkPen(None),
        )
        self._plot_v.addItem(self._vol_item)
        self._vol_ma_item = pg.PlotCurveItem(
            x=[], y=[],
            pen=pg.mkPen(_qc(_GOLD, 200), width=1.5),
        )
        self._plot_v.addItem(self._vol_ma_item)

        # Crosshair — one set per subplot so lines extend into every panel
        cross_pen = pg.mkPen(_CROSS, width=1, style=Qt.PenStyle.DashLine)

        # Main candle plot: vertical + horizontal lines
        self._vline    = pg.InfiniteLine(angle=90, movable=False, pen=cross_pen)
        self._hline    = pg.InfiniteLine(angle=0,  movable=False, pen=cross_pen)
        self._vline.setVisible(False)
        self._hline.setVisible(False)
        self._plot_c.addItem(self._vline, ignoreBounds=True)
        self._plot_c.addItem(self._hline, ignoreBounds=True)

        # Volume subplot: vertical line + value readout label
        self._vline_v  = pg.InfiniteLine(angle=90, movable=False, pen=cross_pen)
        self._vline_v.setVisible(False)
        self._plot_v.addItem(self._vline_v, ignoreBounds=True)

        self._vol_label = pg.TextItem(
            text="", color=_FG,
            fill=pg.mkBrush(_qc(_BG_TIP, 200)),
            anchor=(0.0, 0.5),
        )
        self._vol_label.setFont(QFont("Monospace", 8))
        self._vol_label.setZValue(100)
        self._vol_label.setVisible(False)
        self._plot_v.addItem(self._vol_label, ignoreBounds=True)

        # KD subplot: vertical line + value readout label
        self._vline_kd = pg.InfiniteLine(angle=90, movable=False, pen=cross_pen)
        self._vline_kd.setVisible(False)
        self._plot_kd.addItem(self._vline_kd, ignoreBounds=True)

        self._kd_label = pg.TextItem(
            text="", color=_FG,
            fill=pg.mkBrush(_qc(_BG_TIP, 200)),
            anchor=(0.0, 0.5),
        )
        self._kd_label.setFont(QFont("Monospace", 8))
        self._kd_label.setZValue(100)
        self._kd_label.setVisible(False)
        self._plot_kd.addItem(self._kd_label, ignoreBounds=True)

        # Stored KD width array for crosshair readout (populated by _draw_kd)
        self._kd_width_arr: np.ndarray | None = None

        # CVD subplot: vertical line + value readout label
        self._vline_cvd = pg.InfiniteLine(angle=90, movable=False, pen=cross_pen)
        self._vline_cvd.setVisible(False)
        self._plot_cvd.addItem(self._vline_cvd, ignoreBounds=True)

        self._cvd_label = pg.TextItem(
            text="", color=_FG,
            fill=pg.mkBrush(_qc(_BG_TIP, 200)),
            anchor=(0.0, 0.5),
        )
        self._cvd_label.setFont(QFont("Monospace", 8))
        self._cvd_label.setZValue(100)
        self._cvd_label.setVisible(False)
        self._plot_cvd.addItem(self._cvd_label, ignoreBounds=True)

        # Stored CVD series for crosshair readout (populated by _draw_cvd)
        self._cvd_arr: np.ndarray | None = None

        # A/D Line subplot: vertical line + value readout label
        self._vline_adi = pg.InfiniteLine(angle=90, movable=False, pen=cross_pen)
        self._vline_adi.setVisible(False)
        self._plot_adi.addItem(self._vline_adi, ignoreBounds=True)

        self._adi_label = pg.TextItem(
            text="", color=_FG,
            fill=pg.mkBrush(_qc(_BG_TIP, 200)),
            anchor=(0.0, 0.5),
        )
        self._adi_label.setFont(QFont("Monospace", 8))
        self._adi_label.setZValue(100)
        self._adi_label.setVisible(False)
        self._plot_adi.addItem(self._adi_label, ignoreBounds=True)

        # Stored A/D Line series for crosshair readout (populated by _draw_adi)
        self._adi_arr: np.ndarray | None = None

        # Last candle index shown in the tick profile (used to re-render on
        # order-size filter toggle without waiting for the next mouse move).
        self._last_hover_idx: int | None = None

        # Profile panel: horizontal line that follows main chart price (Y)
        self._profile_hline = pg.InfiniteLine(
            angle=0, movable=False,
            pen=pg.mkPen(_CROSS, width=1, style=Qt.PenStyle.DashLine),
        )
        self._profile_hline.setVisible(False)

        # Tick profile panel: same crosshair horizontal line
        self._tick_profile_hline = pg.InfiniteLine(
            angle=0, movable=False,
            pen=pg.mkPen(_CROSS, width=1, style=Qt.PenStyle.DashLine),
        )
        self._tick_profile_hline.setVisible(False)

        # Price label: yellow price tag that tracks cursor Y, left-aligned
        self._price_label = pg.TextItem(
            text="", color=_GOLD,
            fill=pg.mkBrush(_qc(_BG_TIP, 180)),
            anchor=(0.0, 0.5),   # left edge, vertically centered
        )
        self._price_label.setFont(QFont("Monospace", 7))
        self._price_label.setZValue(100)
        self._price_label.setVisible(False)
        self._plot_c.addItem(self._price_label, ignoreBounds=True)

        # OHLCV tooltip: floating multi-line label above cursor, with dark fill
        # anchor=(0.0, 1.0): BOTTOM-LEFT at position → text body extends upward
        self._ohlcv_label = pg.TextItem(
            text="", color="#ffffff",
            fill=pg.mkBrush(_qc(_BG_TIP, 220)),
            anchor=(0.0, 1.0),
        )
        self._ohlcv_label.setFont(QFont("Monospace", 8))
        self._ohlcv_label.setZValue(100)
        self._ohlcv_label.setVisible(False)
        self._plot_c.addItem(self._ohlcv_label, ignoreBounds=True)

        # Mouse tracking — direct connection (no SignalProxy buffer) for instant
        # crosshair and tooltip response on every mouse-move event.
        self._chart_widget.scene().sigMouseMoved.connect(self._on_mouse_move)

        splitter.addWidget(self._chart_widget)
        splitter.setStretchFactor(0, 5)

        # ── Right: tick profile + session profile ─────────────────────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        # Single-candle tick profile (top-right, shown on hover)
        self._tick_profile_widget = pg.PlotWidget(
            axisItems={"bottom": _VolumeAxisItem()})
        self._tick_profile_widget.setBackground(_BG)
        self._tick_profile_widget.setMinimumWidth(200)
        self._tick_profile_widget.setMaximumWidth(300)
        self._tick_profile_widget.getPlotItem().showGrid(x=True, y=False, alpha=0.15)
        self._tick_profile_widget.getPlotItem().setMenuEnabled(False)
        self._tick_profile_widget.getPlotItem().getAxis("left").setStyle(
            autoReduceTextSpace=True, tickTextOffset=2)
        # setLabel("top", ...) (used for the buy/sell/delta title) implicitly
        # shows the top axis -- with no style of its own it renders default,
        # unabbreviated tick numbers that visually run together in this
        # narrow (200-300px) panel. The top axis exists here only to host
        # that title text, so hide its tick values entirely (bottom already
        # shows the volume scale).
        self._tick_profile_widget.getPlotItem().getAxis("top").setStyle(showValues=False)
        # Add crosshair line (survives until next pw.clear() call; restored in _show_tick_profile)
        self._tick_profile_widget.addItem(self._tick_profile_hline)
        right_layout.addWidget(self._tick_profile_widget, 1)

        # Sync tick profile Y range whenever the main candle chart is panned/zoomed
        self._plot_c.vb.sigRangeChanged.connect(self._on_main_range_changed)

        # ── Heatmap legend (top-left of candle chart) ─────────────────────────
        # Two small colored squares + labels, pinned to the top-left corner.
        # Visibility is toggled by the Heatmap checkbox.
        self._heatmap_legend = pg.TextItem(
            html=(
                f'<span style="color:{_UP}; font-family:Monospace; font-size:9pt;">'
                f'&#9632; Buy</span>'
                f'<span style="color:{_FG}; font-family:Monospace; font-size:9pt;">'
                f'&nbsp;&nbsp;</span>'
                f'<span style="color:{_DOWN}; font-family:Monospace; font-size:9pt;">'
                f'&#9632; Sell</span>'
            ),
            anchor=(0.0, 0.0),   # top-left of text at position
        )
        self._heatmap_legend.setZValue(60)
        self._heatmap_legend.setVisible(False)   # shown when heatmap checkbox on
        self._plot_c.addItem(self._heatmap_legend, ignoreBounds=True)

        # Pin legend to top-left whenever view range changes
        self._plot_c.vb.sigRangeChanged.connect(self._pin_heatmap_legend)

        # ── OB legend (second row, top-left) ──────────────────────────────────
        self._ob_legend = pg.TextItem(
            html=(
                f'<span style="color:{_OB_BULL};  font-family:Monospace; font-size:8pt;">&#9632; Bull OB&nbsp;&nbsp;</span>'
                f'<span style="color:{_OB_BEAR};  font-family:Monospace; font-size:8pt;">&#9632; Bear OB&nbsp;&nbsp;</span>'
                f'<span style="color:{_OB_MIT};   font-family:Monospace; font-size:8pt;">&#9632; Mitig.&nbsp;&nbsp;</span>'
                f'<span style="color:{_OB_BREAK}; font-family:Monospace; font-size:8pt;">&#9632; Breaker</span>'
            ),
            anchor=(0.0, 0.0),
        )
        self._ob_legend.setZValue(60)
        self._ob_legend.setVisible(False)
        self._plot_c.addItem(self._ob_legend, ignoreBounds=True)
        self._plot_c.vb.sigRangeChanged.connect(self._pin_ob_legend)

        # ── Chandelier exit suggestion label (bottom-right) ─────────────────────
        self._chandelier_label = pg.TextItem(
            anchor=(1.0, 1.0),
            fill=pg.mkBrush(_qc(_BG_TIP, 200)),
        )
        self._chandelier_label.setFont(QFont("Monospace", 8))
        self._chandelier_label.setZValue(60)
        self._chandelier_label.setVisible(False)
        self._plot_c.addItem(self._chandelier_label, ignoreBounds=True)
        self._plot_c.vb.sigRangeChanged.connect(self._pin_chandelier_label)

        # Session vol profile (bottom-right)
        self._profile_widget = pg.PlotWidget(
            axisItems={"bottom": _VolumeAxisItem()})
        self._profile_widget.setBackground(_BG)
        self._profile_widget.setMinimumWidth(200)
        self._profile_widget.setMaximumWidth(300)
        self._profile_widget.getPlotItem().showGrid(x=True, y=False, alpha=0.15)
        self._profile_widget.getPlotItem().setMenuEnabled(False)
        self._profile_widget.getPlotItem().getAxis("left").setStyle(
            autoReduceTextSpace=True, tickTextOffset=2)
        self._profile_widget.getPlotItem().getAxis("bottom").setStyle(
            autoReduceTextSpace=True, tickTextOffset=2)
        # Disabled (not just left alone) so `scale` passed into
        # _VolumeAxisItem.tickStrings is always 1.0 -- _fmt_vol_short does
        # its own K/M abbreviation and would otherwise double up with
        # whatever prefix auto-SI-scaling decides to apply.
        self._profile_widget.getPlotItem().getAxis("bottom").enableAutoSIPrefix(False)
        # setLabel("top", ...) (used for the range-mode title, e.g. "1D")
        # implicitly shows the top axis -- with no style of its own it
        # renders default, unabbreviated tick numbers ("200000400000...")
        # that visually run together in this narrow (200-300px) panel. The
        # top axis exists here only to host that title text, so hide its
        # tick values entirely (bottom already shows the volume scale).
        self._profile_widget.getPlotItem().getAxis("top").setStyle(showValues=False)
        # Add the crosshair sync line here so it persists across pw.clear() calls
        self._profile_widget.addItem(self._profile_hline)
        right_layout.addWidget(self._profile_widget, 2)

        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)

        self.setCentralWidget(splitter)

    # ── Status bar ────────────────────────────────────────────────────────────

    def _build_statusbar(self) -> None:
        self._status = QStatusBar()
        self._status.setStyleSheet(
            f"QStatusBar {{ background: {_BG_BAR}; color: {_FG}; font-size: 11px; }}")
        self.setStatusBar(self._status)
        self._log("Ready.")

    def _log(self, msg: str) -> None:
        ts   = datetime.now().strftime("%H:%M:%S")
        line = f"{ts}  {msg}"
        self._status.showMessage(line)
        # The status bar only ever shows the latest message -- the next
        # periodic log call (e.g. "Fetching K-lines...", every refresh tick)
        # overwrites it within seconds, so an intermittent warning (like a
        # get_cur_kline failure) flashes and is gone before it can even be
        # read, let alone screenshotted. Also print to the console the app
        # was launched from so there's a scrollable, copyable record.
        print(line, flush=True)

    # ── OpenD connection ──────────────────────────────────────────────────────

    def _connect_opend(self, host: str = "127.0.0.1", port: int = 11111) -> None:
        try:
            with self._ctx_lock:
                if self._ctx:
                    self._ctx.close()
                self._ctx = OpenQuoteContext(host=host, port=port)
            self._last_chart_key = ("", "")  # force view reset on next render
            # Discard any stuck DataFetcher from the previous session so the
            # isRunning() guard in _trigger_fetch doesn't block new fetches.
            if self._fetcher is not None:
                try:
                    self._fetcher.ready.disconnect()
                    self._fetcher.error.disconnect()
                except Exception:
                    pass
                self._fetcher = None
            self._conn_btn.setText("Stop")
            self._conn_btn.setChecked(True)
            self._log(f"Connected to OpenD {host}:{port}")
            self._trigger_fetch()
            if self._mode_combo.currentText() == "Live":
                self._start_live()
        except Exception as exc:
            self._log(f"Connect failed: {exc}")
            self._conn_btn.setChecked(False)

    def _on_connect_toggle(self, checked: bool) -> None:
        if checked:
            self._connect_opend()
        else:
            self._stop_live()
            with self._ctx_lock:
                if self._ctx:
                    self._ctx.close()
                    self._ctx = None
            if self._fetcher is not None:
                try:
                    self._fetcher.ready.disconnect()
                    self._fetcher.error.disconnect()
                except Exception:
                    pass
                self._fetcher = None
            self._conn_btn.setText("Connect")
            self._log("Disconnected.")

    # ── Live mode tick subscription ───────────────────────────────────────────

    def _start_live(self) -> None:
        code = self._code_edit.text().strip()
        if not code or not self._ctx:
            return
        try:
            tf = self._tf_combo.currentText()
            kl_sub = _TF_KLINE_SUBTYPE.get(tf)
            sub_types = [SubType.TICKER, SubType.QUOTE]
            if kl_sub is not None:
                sub_types.append(kl_sub)
            self._ctx.subscribe(code, sub_types,
                                extended_time=True, session=Session.ALL)
            self._live_kl_sub = kl_sub
            _, candle_mins = TIMEFRAME_MAP[tf]

            viewer = self

            class _TickHandler(TickerHandlerBase):
                def on_recv_rsp(self, rsp_str):
                    ret, data = super().on_recv_rsp(rsp_str)
                    if ret != RET_OK or data is None or data.empty:
                        return
                    with viewer._tick_lock:
                        for _, row in data.iterrows():
                            raw = str(row["time"])
                            fmt = ("%Y-%m-%d %H:%M:%S.%f" if "." in raw
                                   else "%Y-%m-%d %H:%M:%S")
                            t      = datetime.strptime(raw, fmt)
                            bucket = candle_start(t, candle_mins)
                            price  = float(row["price"])
                            vol    = int(row["volume"])
                            d      = str(row["ticker_direction"]).upper()
                            key    = ("buy" if d == "BUY"
                                      else ("sell" if d == "SELL" else "neutral"))
                            viewer._live_ticks[bucket][price][key] += vol
                            viewer._last_tick_price = price

            class _QuoteHandler(StockQuoteHandlerBase):
                def on_recv_rsp(self, rsp_str):
                    ret, data = super().on_recv_rsp(rsp_str)
                    if ret != RET_OK or data is None or data.empty:
                        return
                    for _, row in data.iterrows():
                        bid = float(row.get("bid_price", 0) or 0)
                        ask = float(row.get("ask_price", 0) or 0)
                        if bid > 0 and ask > 0:
                            viewer._last_nbbo = (bid, ask)
                            if viewer._liq_hm_window is not None:
                                QMetaObject.invokeMethod(
                                    viewer._liq_hm_window, "update_live_price",
                                    Qt.ConnectionType.QueuedConnection,
                                    Q_ARG(float, bid), Q_ARG(float, ask),
                                )

            self._ctx.set_handler(_TickHandler())
            self._ctx.set_handler(_QuoteHandler())
            self._live_code = code
            interval = self._refresh_spin.value() * 1000
            self._refresh_timer.start(interval)
            self._log(f"Live: subscribed {code}, refreshing every "
                      f"{self._refresh_spin.value()}s")
        except Exception as exc:
            self._log(f"Live start error: {exc}")

    def _stop_live(self) -> None:
        self._refresh_timer.stop()
        try:
            if self._ctx and self._live_code:
                unsub = [SubType.TICKER, SubType.QUOTE]
                if self._live_kl_sub is not None:
                    unsub.append(self._live_kl_sub)
                self._ctx.unsubscribe(self._live_code, unsub)
        except Exception:
            pass
        self._live_code = ""
        self._live_kl_sub = None

    # ── Toolbar callbacks ─────────────────────────────────────────────────────

    def _on_tf_changed(self, tf: str) -> None:
        _, self._candle_mins = TIMEFRAME_MAP[tf]
        self._last_chart_key = ("", "")  # force zoom reset on next render
        # Live tick buckets are keyed by bar-start time at the OLD candle_mins
        # alignment; switching timeframe changes the alignment, so stale
        # buckets from before the switch must not leak into new bars whose
        # start time happens to coincide with an old bucket key.
        with self._tick_lock:
            self._live_ticks.clear()
        if self._mode_combo.currentText() == "Live" and self._live_code:
            # _start_live() subscribes to the K-line SubType matching the
            # *current* timeframe -- without re-subscribing here, switching
            # TF while live leaves the old timeframe's K-line subscription
            # active (or none) while get_cur_kline is now asked for the new
            # one, which moomoo rejects with "please subscribe to KL_*Min
            # data first" (visible as a get_cur_kline warning in the log)
            # and the newest bar for that TF never appears until a manual
            # disconnect/reconnect forces a fresh subscribe.
            try:
                if self._ctx:
                    unsub = [SubType.TICKER, SubType.QUOTE]
                    if self._live_kl_sub is not None:
                        unsub.append(self._live_kl_sub)
                    self._ctx.unsubscribe(self._live_code, unsub)
            except Exception:
                pass
            self._live_code = ""
            self._start_live()
        self._trigger_fetch()

    def _on_mode_changed(self, mode: str) -> None:
        self._date_edit.setEnabled(mode == "Historical")
        self._last_chart_key = ("", "")  # force zoom reset on next render
        if mode == "Live":
            self._start_live()
        else:
            self._stop_live()
        self._trigger_fetch()

    def _on_refresh_changed(self, value: int) -> None:
        if self._refresh_timer.isActive():
            self._refresh_timer.start(value * 1000)

    def _set_subplot_row_visible(self, plot_item: pg.PlotItem, row: int, visible: bool) -> None:
        """Show/hide a subplot row, collapsing its layout space to zero when hidden.

        PlotItem.hide() alone leaves the row's stretch-factor share of space
        empty in the GraphicsLayout grid; zeroing the row's stretch factor and
        max height gives that space back to the other (visible) rows.
        """
        layout = self._chart_widget.ci.layout
        if visible:
            plot_item.show()
            layout.setRowStretchFactor(row, 1)
            layout.setRowMaximumHeight(row, 16777215)  # Qt's QWIDGETSIZE_MAX
        else:
            plot_item.hide()
            layout.setRowStretchFactor(row, 0)
            layout.setRowMaximumHeight(row, 0)

    def _on_indicator_toggle(self) -> None:
        # Toggle KD subplot row visibility
        show_kd = self._ind("kd")
        if show_kd:
            code = self._code_edit.text().strip()
            self._plot_kd.setTitle(f"{code}  KD", color=_FG, size="8pt")
        self._set_subplot_row_visible(self._plot_kd, 2, show_kd)
        # Toggle Vol (MAVOL) subplot visibility
        show_vol = self._ind("vol")
        self._set_subplot_row_visible(self._plot_v, 1, show_vol)
        if not show_vol:
            self._vline_v.setVisible(False)
        # Toggle CVD subplot visibility
        show_cvd = self._ind("cvd")
        if show_cvd:
            code = self._code_edit.text().strip()
            self._plot_cvd.setTitle(f"{code}  CVD", color=_FG, size="8pt")
        self._set_subplot_row_visible(self._plot_cvd, 3, show_cvd)
        if not show_cvd:
            self._vline_cvd.setVisible(False)
        # Toggle A/D Line subplot visibility
        show_adi = self._ind("adi")
        if show_adi:
            code = self._code_edit.text().strip()
            self._plot_adi.setTitle(f"{code}  A/D", color=_FG, size="8pt")
        self._set_subplot_row_visible(self._plot_adi, 4, show_adi)
        if not show_adi:
            self._vline_adi.setVisible(False)
        # Toggle heatmap legend visibility
        show_hm = self._ind("heatmap")
        self._heatmap_legend.setVisible(show_hm)
        if show_hm:
            self._pin_heatmap_legend()
        show_ob = self._ind("ob")
        self._ob_legend.setVisible(show_ob)
        if show_ob:
            self._pin_ob_legend()
        self._ob_max_count.setEnabled(self._ind("ob"))
        self._fvg_min_pct.setEnabled(self._ind("fvg"))
        self._render(self._klines, self._ticks)
        if self._liq_hm_window is not None:
            self._liq_hm_window.set_red_up(self._ind("red_up"))

    def _on_tick_size_toggle(self) -> None:
        """Re-render the tick profile panel when S/M/L filter changes."""
        if self._range_region is not None:
            i0, i1 = self._range_last_indices
            if i0 >= 0:
                self._show_range_tick_profile(i0, i1)
        elif self._last_hover_idx is not None:
            self._show_tick_profile(self._last_hover_idx)

    def _on_session_toggle(self) -> None:
        self._rebuild_session_profile()

    def _on_range_changed(self) -> None:
        self._rebuild_session_profile()

    def _get_range_val(self) -> str:
        for val in ("1d", "2d", "3d", "7d"):
            rb = self._ind_checks.get(f"range_{val}")
            if rb and rb.isChecked():
                return val
        return "1d"

    def _ind(self, key: str) -> bool:
        cb = self._ind_checks.get(key)
        return bool(cb and cb.isChecked())

    def _active_sessions(self) -> set[str]:
        sessions = set()
        for key in ("regular", "pre", "post", "night"):
            if self._ind(f"sess_{key}"):
                sessions.add(key)
        return sessions

    # ── Data fetch ────────────────────────────────────────────────────────────

    def _trigger_fetch(self) -> None:
        with self._ctx_lock:
            ctx = self._ctx
        if ctx is None:
            self._log("Not connected.")
            return
        if self._fetcher and self._fetcher.isRunning():
            self._log("Fetch in progress, please wait…")
            return

        tf           = self._tf_combo.currentText()
        _, cm        = TIMEFRAME_MAP[tf]
        self._candle_mins = cm
        historical   = self._mode_combo.currentText() == "Historical"
        date_str     = self._date_edit.text().strip()
        ind          = {k: self._ind(k) for k in self._ind_checks}
        new_code     = self._code_edit.text().strip()

        if not historical and new_code != self._live_code and self._live_code:
            # Code changed while in Live mode — resubscribe and discard stale ticks.
            try:
                if self._ctx:
                    unsub = [SubType.TICKER, SubType.QUOTE]
                    if self._live_kl_sub is not None:
                        unsub.append(self._live_kl_sub)
                    self._ctx.unsubscribe(self._live_code, unsub)
            except Exception:
                pass
            with self._tick_lock:
                self._live_ticks.clear()
            self._live_code = ""
            self._start_live()
            self._log(f"Live: switched to {new_code}, waiting for first bar…")
            return  # let the refresh timer drive the first fetch for the new code

        with self._tick_lock:
            live_snap = {k: dict(v) for k, v in self._live_ticks.items()}

        params = {
            "code":            self._code_edit.text().strip(),
            "tf":              tf,
            "historical":      historical,
            "date_str":        date_str,
            "candle_mins":     cm,
            "ind":             ind,
            "live_ticks":      live_snap,
            "fvg_min_gap_pct": self._fvg_min_pct.value() / 100.0,
            "ob_max_count":    self._ob_max_count.value(),
        }
        self._log(f"Fetching K-lines ({tf}) ...")
        self._fetcher = DataFetcher(ctx, params)
        self._fetcher.ready.connect(self._on_data_ready)
        self._fetcher.error.connect(self._log)
        self._fetcher.start()

        # Push real-time NBBO to the heatmap so its spread lines reflect the
        # true best bid/ask rather than values derived from ORDER_BOOK depth
        # (which on LITE accounts does not start at the actual NBBO).
        if not historical and self._live_code:
            try:
                ret, df = ctx.get_market_snapshot([self._live_code])
                if ret == RET_OK and not df.empty:
                    row = df.iloc[0]
                    bid = float(row.get("bid_price", 0) or 0)
                    ask = float(row.get("ask_price", 0) or 0)
                    if bid > 0 and ask > 0:
                        self._last_nbbo = (bid, ask)
                        if self._liq_hm_window is not None:
                            self._liq_hm_window.update_quote(bid, ask)
            except Exception:
                pass

    def _on_data_ready(self, result: dict) -> None:
        prev_n = len(self._klines) if self._klines is not None else 0
        xlo, xhi = self._plot_c.vb.viewRange()[0] if prev_n > 0 else (0, 0)

        self._klines      = result["klines"]
        self._ticks       = result["ticks"]
        self._warmup      = result["warmup"]
        self._smc_signals = result["smc_signals"]
        self._fvg_gaps    = result["fvg_gaps"]
        self._ob_blocks   = result["ob_blocks"]
        self._render(self._klines, self._ticks)

        # In Live mode, auto-scroll right when new bars arrive and the view
        # was already at the right edge (user hasn't manually panned left).
        new_n = len(self._klines)
        if self._mode_combo.currentText() == "Live" and new_n > prev_n > 0:
            # _reset_view() only sets pan/zoom limits (xMax/maxXRange) on a
            # genuine code/tf change; without this, the limits set at the
            # last reset stay frozen while new_n keeps growing, and once
            # enough bars accumulate the stale xMax clamps setXRange below,
            # silently cutting off the newest candle(s) until Home is pressed.
            self._plot_c.vb.setLimits(xMax=new_n + 5, maxXRange=new_n + 10)
            if xhi >= prev_n - 2:
                visible = xhi - xlo
                # Candle bodies are drawn i±0.35 wide, so the last body's
                # right edge sits at new_n - 0.65. Leave a full bar of margin
                # past that (matching _reset_view's buffer) so it never clips
                # at padding=0.
                new_xhi = new_n + 0.5
                self._plot_c.setXRange(new_xhi - visible, new_xhi, padding=0)
                # Price (and volume) Y-range was only set once by _reset_view()
                # at load time and never revisited -- as price drifts away from
                # that frozen band the candles walk above/below the visible
                # window over time. Only re-fit while we're also re-centering
                # X (i.e. the user hasn't panned away to review history) so
                # this doesn't fight a deliberate manual vertical zoom.
                self._autorange_price_subplots(new_xhi - visible, new_xhi)
            # CVD / A-D Line Y-range was only set once by _reset_view() at
            # load time and never revisited -- refresh it to the current
            # visible window on every live tick so it doesn't go stale (see
            # _autorange_cumulative_subplots).
            xlo2, xhi2 = self._plot_c.vb.viewRange()[0]
            self._autorange_cumulative_subplots(xlo2, xhi2)
        # Keep DOM window in sync with active code, mode, and timeframe
        if self._dom_window is not None:
            code = self._code_edit.text().strip()
            live = self._mode_combo.currentText() == "Live"
            self._dom_window.set_code(code)
            self._dom_window.set_live(live)
            self._dom_window.set_timeframe(self._candle_mins)
        # Keep Liq HM window in sync
        if self._liq_hm_window is not None:
            code = self._code_edit.text().strip()
            live = self._mode_combo.currentText() == "Live"
            self._liq_hm_window.set_code(code)
            self._liq_hm_window.set_live(live)

    # ── Option Walls / Option Alerts ──────────────────────────────────────────
    # Shared poll: one OptionChainFetcher run serves both indicators. Uses its
    # own OpenQuoteContext (self._option_ctx), separate from self._ctx, so its
    # background-thread network calls never run concurrently against the same
    # context DataFetcher uses on the K-line refresh cycle.

    def _ensure_option_ctx(self) -> None:
        if self._option_ctx is not None:
            return
        try:
            self._option_ctx = OpenQuoteContext(host="127.0.0.1", port=11111)
        except Exception as exc:
            self._log(f"Option context connect failed: {exc}")
            self._option_ctx = None

    def _on_option_indicator_toggle(self) -> None:
        walls_on  = self._ind("option_walls")
        alerts_on = self._ind("option_alerts")
        if walls_on or alerts_on:
            self._ensure_option_ctx()
            if not self._option_timer.isActive():
                self._option_timer.start(self._option_interval_spin.value() * 1000)
            self._trigger_option_fetch()
        else:
            self._option_timer.stop()
        if not walls_on:
            self._clear_option_wall_items()
        if not alerts_on:
            self._clear_option_alert_items()

    def _on_option_0dte_toggle(self) -> None:
        self._option_dte_spin.setEnabled(not self._option_0dte_cb.isChecked())
        if self._ind("option_walls") or self._ind("option_alerts"):
            self._trigger_option_fetch()

    def _on_option_interval_changed(self, value: int) -> None:
        if self._option_timer.isActive():
            self._option_timer.start(value * 1000)

    def _on_option_wall_count_changed(self) -> None:
        # Cached self._option_wall_data only ever holds as many walls per
        # side as the *previous* fetch was asked for -- redrawing from cache
        # alone can't show more than that, so a count increase needs a fresh
        # fetch (not just a re-render) to actually have the extra strikes.
        if self._ind("option_walls") or self._ind("option_alerts"):
            self._trigger_option_fetch()

    def _trigger_option_fetch(self) -> None:
        if not (self._ind("option_walls") or self._ind("option_alerts")):
            return
        self._ensure_option_ctx()
        if self._option_ctx is None:
            return
        if self._option_fetcher is not None and self._option_fetcher.isRunning():
            return  # previous poll still in flight -- skip this tick
        code = self._code_edit.text().strip()
        if not code:
            return
        if code != self._option_code:
            # Symbol changed since the last poll -- no volume baseline for
            # the new symbol yet (would falsely compare against the old
            # symbol's last-known volumes), and any drawn walls/alerts
            # belong to the old symbol.
            self._option_prev_volumes = {}
            self._option_code = code
            self._clear_option_wall_items()
            self._clear_option_alert_items()
        dte = 0 if self._option_0dte_cb.isChecked() else self._option_dte_spin.value()
        threshold  = self._option_threshold_spin.value()
        wall_count = self._option_wall_count_spin.value()
        self._option_status_lbl.setText("options: fetching…")
        self._option_fetcher = OptionChainFetcher(
            self._option_ctx, code, dte, threshold, dict(self._option_prev_volumes), wall_count)
        self._option_fetcher.ready.connect(self._on_option_fetch_ready)
        self._option_fetcher.error.connect(self._on_option_fetch_error)
        self._option_fetcher.start()

    def _on_option_fetch_error(self, msg: str) -> None:
        self._log(f"Option fetch: {msg}")
        self._option_status_lbl.setText("options: no data")

    def _on_option_fetch_ready(self, result: dict) -> None:
        self._option_wall_data     = result
        self._option_prev_volumes  = result["volumes"]
        ts_str = result["computed_at"].strftime("%H:%M:%S")
        self._option_status_lbl.setText(f"options: updated {ts_str}")
        if self._ind("option_walls") and self._klines is not None:
            self._clear_option_wall_items()
            self._draw_option_walls(self._klines)
        if self._ind("option_alerts"):
            for entry in result["large_trades"]:
                self._fire_option_alert(entry)

    def _fire_option_alert(self, entry: dict) -> None:
        code   = self._code_edit.text().strip()
        title  = f"{code} large option trade"
        msg    = f"{entry['option_type']} ${entry['strike']:.0f}  +{entry['volume_delta']} contracts"
        self._show_option_tray_message(title, msg)
        self._add_option_alert_marker(entry)

    def _build_option_tray(self) -> None:
        if not QSystemTrayIcon.isSystemTrayAvailable():
            self._tray = None
            return
        pix = QPixmap(16, 16)
        pix.fill(QColor(_RED))
        self._tray = QSystemTrayIcon(QIcon(pix), self)
        self._tray.setToolTip("Trade Viewer — Option Alerts")
        self._tray.show()

    def _show_option_tray_message(self, title: str, msg: str) -> None:
        if self._tray is None:
            self._build_option_tray()
        if self._tray is not None:
            self._tray.showMessage(
                title, msg, QSystemTrayIcon.MessageIcon.Information, 8000)

    # ── Chart rendering ───────────────────────────────────────────────────────

    def _render(self, klines: pd.DataFrame | None, ticks: dict | None) -> None:
        if klines is None or klines.empty:
            return

        tf          = self._tf_combo.currentText()
        _, cm       = TIMEFRAME_MAP[tf]
        n           = len(klines)
        show_hm     = self._ind("heatmap")
        show_delta  = self._ind("delta")
        show_bos    = self._ind("bos_choch")
        show_fvg    = self._ind("fvg")
        show_ob     = self._ind("ob")
        show_kd     = self._ind("kd")
        show_kd_band = self._ind("kd_band")
        show_cvd    = self._ind("cvd")
        show_adi    = self._ind("adi")
        show_ema    = self._ind("ema")
        show_avwap  = self._ind("avwap")
        red_up      = self._ind("red_up")
        bull_col    = _RED   if red_up else _GREEN
        bear_col    = _GREEN if red_up else _RED

        # Compose live + historical ticks. ticks.db is kept fresh by an
        # external collector process subscribed to the same feed, so once a
        # bucket shows up there it is authoritative — _live_ticks (this
        # process's own tick subscription) must only fill buckets ticks.db
        # doesn't have yet, never add on top of one it already has, or the
        # same real trades get counted twice.
        buckets: dict = {}
        if ticks:
            buckets.update(ticks)
        if self._mode_combo.currentText() == "Live":
            with self._tick_lock:
                for bk, pd_ in self._live_ticks.items():
                    if bk in buckets:
                        continue
                    buckets[bk] = {price: dict(counts) for price, counts in pd_.items()}

        # Keep self._ticks in sync with the fully-merged buckets so that
        # _show_tick_profile (which reads self._ticks) also sees live data.
        self._ticks = buckets if buckets else self._ticks

        # Heatmap legend: show only when heatmap is active
        self._heatmap_legend.setVisible(show_hm)
        if show_hm:
            self._pin_heatmap_legend()

        # Candlesticks
        self._candle_item.set_data(klines, buckets if show_hm else None,
                                   cm, show_heatmap=show_hm, red_up=red_up)

        # FVG zones
        self._fvg_item.set_data(
            self._fvg_gaps if show_fvg else [], n, red_up=red_up)

        # Order Blocks
        self._ob_item.set_data(
            self._ob_blocks if show_ob else [], n)
        self._ob_legend.setVisible(show_ob)
        if show_ob:
            self._pin_ob_legend()

        # Volume bars
        x      = np.arange(n)
        vols   = klines["volume"].fillna(0).values.astype(float)
        opens  = klines["open"].values.astype(float)
        closes = klines["close"].values.astype(float)
        vol_colors = [
            pg.mkBrush(_qc(bull_col, 100)) if c >= o
            else pg.mkBrush(_qc(bear_col, 100))
            for o, c in zip(opens, closes)
        ]
        self._vol_item.setOpts(
            x=x, height=vols, width=0.7,
            brushes=vol_colors,
        )
        # Volume MA curve — rolling mean, NaN for the warm-up period
        vol_ma = pd.Series(vols).rolling(_VOL_MA, min_periods=1).mean().values
        self._vol_ma_item.setData(x=x, y=vol_ma.astype(float))

        # BOS / CHoCH
        self._clear_bos_items()
        if show_bos:
            self._draw_bos_choch(self._smc_signals, red_up=red_up)

        # Delta annotations
        self._clear_delta_items()
        if show_delta and buckets:
            self._draw_delta(klines, buckets, cm)

        # EMA overlays
        self._clear_ema_items()
        if show_ema:
            self._draw_ema(klines)

        # Anchored VWAP overlay
        self._clear_avwap_items()
        if show_avwap:
            self._draw_avwap(klines)

        # Option Walls (Call/Put Wall + Zero Gamma) -- re-renders from the
        # already-cached self._option_wall_data only; never triggers a new
        # option-chain fetch (that's solely _trigger_option_fetch's job, on
        # its own poll timer, decoupled from this per-tick chart redraw).
        self._clear_option_wall_items()
        if self._ind("option_walls"):
            self._draw_option_walls(klines)

        # KD band overlay on main chart (fast/slow midline ribbon)
        self._clear_kd_band_items()
        if show_kd_band:
            self._draw_kd_band(klines)

        # KD subplot
        self._clear_kd_items()
        if show_kd:
            self._draw_kd(klines)

        # CVD subplot
        self._clear_cvd_items()
        if show_cvd:
            self._draw_cvd(klines, buckets, cm)

        # A/D Line subplot
        self._clear_adi_items()
        if show_adi:
            self._draw_adi(klines)

        # Chandelier exit suggestion (bottom-right corner label)
        if self._ind("chandelier"):
            self._update_chandelier_label()
        else:
            self._chandelier_label.setVisible(False)

        # Trade Review overlay
        self._clear_trade_items()
        if self._trade_record is not None:
            self._draw_trade_review(klines)

        # Scanner Signals overlay
        self._clear_scanner_signal_items()
        if self._scanner_signals_btn.isChecked():
            self._load_scanner_signals(klines)

        # X-axis time labels
        self._set_xaxis_ticks(klines)

        # Session vol profile (or range profile panel if active)
        if self._range_region is not None:
            self._clear_range_inline()
            self._range_last_indices = (-1, -1)
            self._rebuild_range_profile()
        else:
            self._rebuild_session_profile()

        # Set view range only when Code or TF changes (i.e. a genuinely new
        # chart).  Deferred via singleShot so profile / overlay drawing cannot
        # override the range we set here.
        code = self._code_edit.text().strip()
        date_str  = self._date_edit.text().strip()
        chart_key = (code, tf, date_str)
        if chart_key != self._last_chart_key:
            self._last_chart_key = chart_key
            n_snap = n
            QTimer.singleShot(0, lambda: self._reset_view(n_snap))


        mode = self._mode_combo.currentText()
        self.setWindowTitle(
            f"Trade Viewer Qt  —  {code}  {tf}  {mode}")

        # Keep subplot titles in sync with the current symbol so multiple
        # open viewer windows are easy to distinguish at a glance.
        self._plot_v.setTitle(f"{code}  Vol", color=_FG, size="8pt")
        if self._plot_kd.isVisible():
            self._plot_kd.setTitle(f"{code}  KD", color=_FG, size="8pt")
        if self._plot_cvd.isVisible():
            self._plot_cvd.setTitle(f"{code}  CVD", color=_FG, size="8pt")
        if self._plot_adi.isVisible():
            self._plot_adi.setTitle(f"{code}  A/D", color=_FG, size="8pt")

        self._log(
            f"Chart rendered | {n} candles | "
            f"{'heatmap' if show_hm else 'plain'}"
        )

    # ── Overlay helpers ───────────────────────────────────────────────────────

    def _clear_bos_items(self) -> None:
        for item in self._bos_items:
            self._plot_c.removeItem(item)
        self._bos_items.clear()

    def _draw_bos_choch(self, signals: list[dict], red_up: bool = False) -> None:
        # Show all signals in the warmup window (no artificial cap).
        # The warmup is already limited to 400 bars so the list is bounded.
        recent   = signals
        bull_col = _RED   if red_up else _GREEN
        bear_col = _GREEN if red_up else _RED

        # Y offset for label: 0.5 % of candle price (scale-invariant).
        # Ensures label floats clearly above/below the wick regardless of zoom.
        klines = self._klines
        highs  = klines["high"].values.astype(float)  if klines is not None else None
        lows   = klines["low"].values.astype(float)   if klines is not None else None
        n      = len(highs) if highs is not None else 0

        for sig in recent:
            color   = bull_col if sig["direction"] == "bull" else bear_col
            label   = sig["type"]
            idx     = sig["idx"]
            price   = sig["price"]          # the broken swing level
            from_i  = sig.get("from_idx", max(0, idx - 5))
            is_bull = sig["direction"] == "bull"

            # Wick extremes at both endpoints (clamp to valid range)
            from_i_c = max(0, min(from_i, n - 1))
            idx_c    = max(0, min(idx,    n - 1))
            if highs is not None:
                wick_from = float(highs[from_i_c] if is_bull else lows[from_i_c])
                wick_idx  = float(highs[idx_c]    if is_bull else lows[idx_c])
            else:
                wick_from = price * (1.002 if is_bull else 0.998)
                wick_idx  = price * (1.003 if is_bull else 0.997)

            # Horizontal line is raised above BOTH candles.
            # offset = 0.5 % of the higher wick so scaling is price-invariant.
            extreme   = max(wick_from, wick_idx) if is_bull else min(wick_from, wick_idx)
            offset    = abs(extreme) * 0.005
            line_y    = extreme + offset if is_bull else extreme - offset

            # ── 1. Horizontal dotted line at raised level ────────────────────
            h_line = pg.PlotCurveItem(
                x=[from_i, idx], y=[line_y, line_y],
                pen=pg.mkPen(color, width=1, style=Qt.PenStyle.DotLine),
            )
            self._plot_c.addItem(h_line, ignoreBounds=True)
            self._bos_items.append(h_line)

            # ── 2a. Left vertical: from the left candle wick up to line_y ────
            # Start slightly beyond the wick so it doesn't overlap the wick tip.
            gap = offset * 0.4
            left_start  = wick_from + gap if is_bull else wick_from - gap
            v_left = pg.PlotCurveItem(
                x=[from_i, from_i], y=[left_start, line_y],
                pen=pg.mkPen(color, width=1, style=Qt.PenStyle.DashLine),
            )
            self._plot_c.addItem(v_left, ignoreBounds=True)
            self._bos_items.append(v_left)

            # ── 2b. Right vertical: from the break candle wick up to line_y ──
            right_start = wick_idx + gap if is_bull else wick_idx - gap
            v_right = pg.PlotCurveItem(
                x=[idx, idx], y=[right_start, line_y],
                pen=pg.mkPen(color, width=1, style=Qt.PenStyle.DashLine),
            )
            self._plot_c.addItem(v_right, ignoreBounds=True)
            self._bos_items.append(v_right)

            # ── 3. Label at the mid-point of the horizontal line ─────────────
            # anchor y=1.0: bottom of text at line_y → text grows upward (bull)
            # anchor y=0.0: top of text at line_y   → text grows downward (bear)
            # White text on colored background for maximum contrast.
            mid_x = (from_i + idx) / 2
            txt = pg.TextItem(
                text=label, color="#ffffff",
                fill=pg.mkBrush(_qc(color, 160)),
                anchor=(0.5, 1.0 if is_bull else 0.0),
            )
            txt.setFont(QFont("Monospace", 8))
            txt.setZValue(50)
            txt.setPos(mid_x, line_y)
            self._plot_c.addItem(txt, ignoreBounds=True)
            self._bos_items.append(txt)

    def _clear_delta_items(self) -> None:
        for item in self._delta_items:
            self._plot_c.removeItem(item)
        self._delta_items.clear()

    def _draw_delta(self, klines: pd.DataFrame, buckets: dict, cm: int) -> None:
        for i, (_, row) in enumerate(klines.iterrows()):
            try:
                bar_end = datetime.strptime(
                    str(row["time_key"])[:16], "%Y-%m-%d %H:%M")
                bk = candle_start(bar_end - timedelta(minutes=cm), cm)
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
                txt_str = f"{sign}{delta/1_000_000:.1f}M"
            elif abs(delta) >= 1_000:
                txt_str = f"{sign}{delta/1_000:.0f}K"
            else:
                txt_str = f"{sign}{delta}"

            # Use bright white with a coloured fill tag so text is always readable
            # against the dark background, regardless of heatmap overlay colour.
            color = _UP if delta >= 0 else _DOWN
            txt = pg.TextItem(
                text=txt_str, color="#ffffff",
                fill=pg.mkBrush(_qc(color, 180)),
                anchor=(0.5, 0.0),
            )
            txt.setFont(QFont("Monospace", 8))
            txt.setPos(i, float(row["low"]) * 0.9997)
            self._plot_c.addItem(txt, ignoreBounds=True)
            self._delta_items.append(txt)

    # ── DOM window ────────────────────────────────────────────────────────────

    def _on_dom_toggle(self, checked: bool) -> None:
        if checked:
            from analysis.dom_window import DomWindow
            code = self._code_edit.text().strip()
            live = self._mode_combo.currentText() == "Live"
            self._dom_window = DomWindow(code=code, live=live)
            self._dom_window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
            self._dom_window.destroyed.connect(self._on_dom_closed)
            self._dom_window.show()
        else:
            if self._dom_window is not None:
                self._dom_window.close()
                self._dom_window = None

    def _on_dom_closed(self) -> None:
        self._dom_window = None
        self._dom_btn.setChecked(False)

    def _dom_sync(self, ts: datetime) -> None:
        """Push crosshair bar time to the DOM window (historical mode)."""
        if self._dom_window is not None and not self._dom_window._live:
            self._dom_window.pin_timestamp(ts)

    # ── Liquidity Heatmap floating window ─────────────────────────────────────

    def _on_liq_hm_toggle(self, checked: bool) -> None:
        if checked:
            from analysis.liq_hm_window import LiqHmWindow
            code = self._code_edit.text().strip()
            live = self._mode_combo.currentText() == "Live"
            self._liq_hm_window = LiqHmWindow()
            self._liq_hm_window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
            self._liq_hm_window.destroyed.connect(self._on_liq_hm_closed)
            self._liq_hm_window.set_code(code)
            self._liq_hm_window.set_live(live)
            self._liq_hm_window.set_red_up(self._ind("red_up"))
            self._liq_hm_window.show()
        else:
            if self._liq_hm_window is not None:
                _close_with_hang_watchdog(self._liq_hm_window, "LiqHmWindow")
                self._liq_hm_window = None

    def _on_liq_hm_closed(self) -> None:
        self._liq_hm_window = None
        self._liq_hm_btn.setChecked(False)

    def _clear_ema_items(self) -> None:
        for item in self._ema_items:
            self._plot_c.removeItem(item)
        self._ema_items.clear()

    def _draw_ema(self, klines: pd.DataFrame) -> None:
        """Overlay EMA 20 / 50 / 200 on the candle chart."""
        closes = klines["close"].reset_index(drop=True)
        x      = np.arange(len(klines))
        for period, color in zip(_EMA_PERIODS, _EMA_COLS):
            if len(klines) < period:
                continue
            ema    = closes.ewm(span=period, adjust=False).mean().values
            curve  = pg.PlotCurveItem(
                x=x, y=ema,
                pen=pg.mkPen(color, width=1),
                name=f"EMA{period}",
            )
            self._plot_c.addItem(curve, ignoreBounds=True)
            self._ema_items.append(curve)

            # Label at right end
            lbl = pg.TextItem(
                text=f"EMA{period}", color=color, anchor=(0.0, 0.5),
            )
            lbl.setFont(QFont("Monospace", 6))
            lbl.setPos(len(klines) - 1, float(ema[-1]))
            self._plot_c.addItem(lbl, ignoreBounds=True)
            self._ema_items.append(lbl)

    def _clear_avwap_items(self) -> None:
        for item in self._avwap_items:
            self._plot_c.removeItem(item)
        self._avwap_items.clear()

    def _draw_avwap(self, klines: pd.DataFrame) -> None:
        """Overlay an anchored VWAP: cumulative volume-weighted typical price
        starting from a user-specified date (self._avwap_anchor_edit), or
        from the start of the current *trading* day if left blank (20:00 ET
        -> next 20:00 ET, the same overnight-session boundary CVD resets on
        -- see _trading_day() -- not the whole loaded chart and not the
        calendar date, which would cut the anchor mid-overnight-session).

        An unparseable date is NOT silently treated as "start of chart" --
        that previously made a mistyped date (e.g. "7/1/2026" instead of the
        placeholder's "YYYY-MM-DD") draw a plausible-looking but wrong line
        spanning the whole chart with no indication anything was off. Instead
        it's flagged with a visible warning label and nothing is drawn.
        """
        n = len(klines)
        if n == 0:
            return

        anchor_idx = 0
        anchor_text = self._avwap_anchor_edit.text().strip()
        if anchor_text:
            anchor_dt = None
            for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d", "%m-%d-%Y"):
                try:
                    anchor_dt = datetime.strptime(anchor_text, fmt)
                    break
                except ValueError:
                    continue
            if anchor_dt is None:
                lbl = pg.TextItem(
                    text=f"AVWAP: unrecognised date '{anchor_text}' (use YYYY-MM-DD)",
                    color=_RED, anchor=(0.0, 1.0),
                )
                lbl.setFont(QFont("Monospace", 7))
                lbl.setPos(0, float(klines["high"].max()))
                self._plot_c.addItem(lbl, ignoreBounds=True)
                self._avwap_items.append(lbl)
                return
            times = klines["time_key"].astype(str).values
            anchor_idx = int(np.searchsorted(times, anchor_dt.strftime("%Y-%m-%d"), side="left"))
            anchor_idx = max(0, min(anchor_idx, n - 1))
        else:
            cm = self._candle_mins
            last_day = _trading_day(klines["time_key"].iloc[-1], cm)
            if last_day is not None:
                days    = klines["time_key"].map(lambda tk: _trading_day(tk, cm))
                matches = np.where(days == last_day)[0]
                if len(matches):
                    anchor_idx = int(matches[0])

        sub     = klines.iloc[anchor_idx:]
        typical = (sub["high"] + sub["low"] + sub["close"]) / 3.0
        vol     = sub["volume"].fillna(0)
        cum_v   = vol.cumsum().replace(0, np.nan)   # avoid divide-by-zero before any volume accrues
        vwap    = ((typical * vol).cumsum() / cum_v).values

        x = np.arange(anchor_idx, n)
        curve = pg.PlotCurveItem(
            x=x, y=vwap, pen=pg.mkPen(_AVWAP_COL, width=1.5),
            name="AVWAP", connect="finite",
        )
        self._plot_c.addItem(curve, ignoreBounds=True)
        self._avwap_items.append(curve)

        anchor_line = pg.InfiniteLine(
            pos=anchor_idx, angle=90, movable=False,
            pen=pg.mkPen(_AVWAP_COL, width=1, style=Qt.PenStyle.DashLine),
        )
        self._plot_c.addItem(anchor_line, ignoreBounds=True)
        self._avwap_items.append(anchor_line)

        last_v = float(vwap[-1]) if not np.isnan(vwap[-1]) else float(vwap[np.isfinite(vwap)][-1])
        lbl = pg.TextItem(text=f"AVWAP {last_v:.2f}", color=_AVWAP_COL, anchor=(0.0, 0.5))
        lbl.setFont(QFont("Monospace", 8))
        lbl.setPos(n - 1, last_v)
        self._plot_c.addItem(lbl, ignoreBounds=True)
        self._avwap_items.append(lbl)

        n_sigma = self._avwap_sigma_spin.value()
        if n_sigma > 0:
            self._draw_avwap_bands(x, typical.values, vol.values, cum_v.values, vwap, n_sigma)

    _AVWAP_BAND_ALPHA = {1: 140, 2: 95, 3: 55}   # fainter for wider (less likely) bands

    def _draw_avwap_bands(self, x: np.ndarray, typical: np.ndarray, vol: np.ndarray,
                          cum_v: np.ndarray, vwap: np.ndarray, n_sigma: int) -> None:
        """Volume-weighted standard-deviation bands around the AVWAP.

        Same formula TradingView's VWAP indicator uses for its bands:
        variance = cumsum(vol * (typical_price - running_vwap)^2) / cumsum(vol),
        i.e. each bar's squared deviation from *that bar's own* running VWAP,
        volume-weighted and accumulated from the same anchor point as the
        VWAP itself -- not deviation from the final VWAP value.
        """
        dev2      = (typical - vwap) ** 2
        variance  = (vol * dev2).cumsum() / cum_v
        stddev    = np.sqrt(np.clip(np.nan_to_num(variance), 0, None))

        for k in range(1, n_sigma + 1):
            alpha = self._AVWAP_BAND_ALPHA.get(k, 40)
            pen   = pg.mkPen(_qc(_AVWAP_COL, alpha), width=1, style=Qt.PenStyle.DashLine)
            # Half-sigma reference line (k - 0.5) -- same opacity as the
            # integer band it precedes, no label (label-free is the only
            # thing that distinguishes it visually now).
            half_pen = pg.mkPen(_qc(_AVWAP_COL, alpha), width=1, style=Qt.PenStyle.DashLine)
            for sign in (1, -1):
                half_band  = vwap + sign * (k - 0.5) * stddev
                half_curve = pg.PlotCurveItem(x=x, y=half_band, pen=half_pen, connect="finite")
                self._plot_c.addItem(half_curve, ignoreBounds=True)
                self._avwap_items.append(half_curve)

            for sign, tag in ((1, "+"), (-1, "-")):
                band = vwap + sign * k * stddev
                curve = pg.PlotCurveItem(x=x, y=band, pen=pen, connect="finite")
                self._plot_c.addItem(curve, ignoreBounds=True)
                self._avwap_items.append(curve)
                finite_idx = np.where(np.isfinite(band))[0]
                if finite_idx.size == 0:
                    continue
                last_i = finite_idx[-1]
                # Label text stays fully opaque (with a background fill for
                # contrast against the candles) even though the band *line*
                # fades out for wider/less-likely bands -- a faint line is a
                # tasteful visual hierarchy, faint text is just unreadable.
                lbl = pg.TextItem(
                    text=f"{tag}{k}s {band[last_i]:.2f}", color=_AVWAP_COL,
                    fill=pg.mkBrush(_qc(_BG_TIP, 180)), anchor=(0.0, 0.5),
                )
                lbl.setFont(QFont("Monospace", 8))
                lbl.setPos(x[last_i], band[last_i])
                self._plot_c.addItem(lbl, ignoreBounds=True)
                self._avwap_items.append(lbl)

    def _clear_option_wall_items(self) -> None:
        for item in self._option_wall_items:
            self._plot_c.removeItem(item)
        self._option_wall_items.clear()

    def _draw_option_wall_line(self, price: float, color: str, label: str,
                               n: int, alpha: int = 255) -> None:
        line = pg.InfiniteLine(
            pos=price, angle=0, movable=False,
            pen=pg.mkPen(_qc(color, alpha), width=1, style=Qt.PenStyle.DashLine),
        )
        self._plot_c.addItem(line, ignoreBounds=True)
        self._option_wall_items.append(line)

        lbl = pg.TextItem(
            text=f"{label} {price:.2f}", color=_qc(color, alpha), anchor=(0.0, 1.0),
        )
        lbl.setFont(QFont("Monospace", 7))
        lbl.setPos(n - 1, price)
        self._plot_c.addItem(lbl, ignoreBounds=True)
        self._option_wall_items.append(lbl)

    def _draw_option_walls(self, klines: pd.DataFrame) -> None:
        """Overlay Call Wall(s) / Put Wall(s) / Zero Gamma as horizontal price
        levels from the latest cached option-chain poll
        (self._option_wall_data). Only re-renders already-fetched data --
        never triggers a new fetch itself (see _trigger_option_fetch).

        call_walls/put_walls are lists, strongest GEX first (see
        top_gex_walls() in strategy/option/gex.py) -- the "Walls" spinbox
        controls how many per side get requested/drawn. Weaker (later-rank)
        walls fade out, same visual-hierarchy idiom as the AVWAP sigma bands
        (_AVWAP_BAND_ALPHA): still visible, but clearly secondary to the
        single strongest wall on each side.
        """
        data = self._option_wall_data
        if not data:
            return
        n = len(klines)
        if n == 0:
            return

        for rank, wall in enumerate(data.get("call_walls") or []):
            alpha = max(60, 255 - rank * 40)
            label = "Call Wall" if rank == 0 else f"Call Wall #{rank + 1}"
            self._draw_option_wall_line(wall["strike"], _RED, label, n, alpha)

        for rank, wall in enumerate(data.get("put_walls") or []):
            alpha = max(60, 255 - rank * 40)
            label = "Put Wall" if rank == 0 else f"Put Wall #{rank + 1}"
            self._draw_option_wall_line(wall["strike"], _GREEN, label, n, alpha)

        zero_gamma = data.get("zero_gamma")
        if zero_gamma is not None:
            self._draw_option_wall_line(zero_gamma, _ZERO_GAMMA_COL, "Zero Gamma", n)

    def _clear_option_alert_items(self) -> None:
        for item in self._option_alert_items:
            self._plot_c.removeItem(item)
        self._option_alert_items.clear()

    def _add_option_alert_marker(self, entry: dict) -> None:
        """Add one large-trade marker at the latest bar -- this is a live
        "just happened" event, not a historical timestamp needing a
        searchsorted lookup like other overlays use. Appended to
        self._option_alert_items without clearing previous markers: each
        alert is a discrete past event that should stay visible, unlike the
        walls (current state, cleared+redrawn every poll)."""
        if self._klines is None or self._klines.empty:
            return
        x       = len(self._klines) - 1
        is_call = entry["option_type"] == "CALL"
        color   = _RED if is_call else _GREEN
        marker = pg.ScatterPlotItem(
            x=[x], y=[entry["strike"]], size=12,
            symbol="o" if is_call else "s",
            brush=pg.mkBrush(color), pen=pg.mkPen("#ffffff", width=0.5),
        )
        self._plot_c.addItem(marker, ignoreBounds=True)
        self._option_alert_items.append(marker)

        tag = "C" if is_call else "P"
        lbl = pg.TextItem(
            text=f"{tag}{entry['strike']:.0f} +{entry['volume_delta']}",
            color=color, anchor=(0.0, 0.5 if is_call else 1.0),
        )
        lbl.setFont(QFont("Monospace", 7))
        lbl.setPos(x, entry["strike"])
        self._plot_c.addItem(lbl, ignoreBounds=True)
        self._option_alert_items.append(lbl)

    def _clear_kd_items(self) -> None:
        for item in self._kd_items:
            self._plot_kd.removeItem(item)
        self._kd_items.clear()
        self._kd_width_arr = None

    def _draw_kd(self, klines: pd.DataFrame) -> None:
        """Draw KD channel width (momentum) in the KD subplot.

        Plots the spread-width series, coloured gold (bullish) / purple (bearish)
        with a fill vs zero to make direction instantly readable.
        """
        if len(klines) < _KD_SLOW + 5:
            return

        warmup = self._warmup if self._warmup is not None else klines
        kd     = compute_kd(warmup, fast=_KD_FAST, slow=_KD_SLOW)

        # Align to klines length; warmup may be shorter than klines in Live mode
        # (warmup capped at 400 bars, klines can be 1500+).
        # x must cover exactly the last len(width) bars of the chart.
        n      = len(klines)
        width  = kd["width"].values[-n:]   # at most min(400, n) values
        nw     = len(width)
        x      = np.arange(n - nw, n)

        # Store for crosshair readout in _on_mouse_move
        self._kd_width_arr = width

        # Split into bull (>0) and bear (<0) segments for colouring
        bull_y = np.where(width >= 0, width, 0.0)
        bear_y = np.where(width <  0, width, 0.0)

        bull_fill = pg.FillBetweenItem(
            pg.PlotCurveItem(x=x, y=bull_y),
            pg.PlotCurveItem(x=x, y=np.zeros(nw)),
            brush=_qc(_UP, 100),
        )
        bear_fill = pg.FillBetweenItem(
            pg.PlotCurveItem(x=x, y=bear_y),
            pg.PlotCurveItem(x=x, y=np.zeros(nw)),
            brush=_qc(_DOWN, 100),
        )
        self._plot_kd.addItem(bull_fill)
        self._plot_kd.addItem(bear_fill)
        self._kd_items.extend([bull_fill, bear_fill])

        # Main width line
        line = pg.PlotCurveItem(
            x=x, y=width,
            pen=pg.mkPen(_GOLD, width=1),
        )
        self._plot_kd.addItem(line)
        self._kd_items.append(line)

        # Spread midline (MID1 – MID2) as a secondary context line (dashed, dim)
        spread = kd["spread"].values[-n:][-nw:]
        spread_norm = spread / (np.abs(spread).max() or 1.0) * (np.abs(width).max() or 1.0)
        spread_line = pg.PlotCurveItem(
            x=x, y=spread_norm,
            pen=pg.mkPen(_GREY, width=1, style=Qt.PenStyle.DashLine),
        )
        self._plot_kd.addItem(spread_line)
        self._kd_items.append(spread_line)

    def _clear_cvd_items(self) -> None:
        for item in self._cvd_items:
            self._plot_cvd.removeItem(item)
        self._cvd_items.clear()
        self._cvd_arr = None

    def _draw_cvd(self, klines: pd.DataFrame, buckets: dict, cm: int) -> None:
        """Draw Cumulative Volume Delta in the CVD subplot.

        Per-bar delta = buy_vol - sell_vol from tick-tagged trades (same
        source as the Delta overlay); bars with no tick coverage contribute 0
        (the line holds flat rather than gapping). Neutral-tagged volume is
        excluded from the delta itself, same caveat as the Range Profile
        Buy/Sell stats -- moomoo's ticker_direction tags a large share of
        same-price prints NEUTRAL, so this only reflects a subset of volume.
        Resets to 0 at the start of each *trading* day (matches TradingView's
        default CVD session reset) instead of accumulating across the whole
        loaded range. A trading day here runs 20:00 ET -> next 20:00 ET
        (overnight session start through the following after-hours close,
        matching the "night" session window used by _filter_sessions() and
        config/schedule.json) -- resetting on the calendar-date boundary
        instead would split the continuous 20:00-04:00 overnight session in
        half at midnight, showing a spurious jump/reset in the middle of a
        session that never actually paused.
        """
        n = len(klines)
        delta = np.zeros(n, dtype=float)
        day = [None] * n
        for i, (_, row) in enumerate(klines.iterrows()):
            try:
                bar_end = datetime.strptime(
                    str(row["time_key"])[:16], "%Y-%m-%d %H:%M")
                bk = candle_start(bar_end - timedelta(minutes=cm), cm)
            except ValueError:
                continue
            day[i] = bk.date() + timedelta(days=1) if bk.hour >= 20 else bk.date()
            pd_ = buckets.get(bk)
            if not pd_:
                continue
            total_buy  = sum(pd_[p]["buy"]  for p in pd_)
            total_sell = sum(pd_[p]["sell"] for p in pd_)
            delta[i] = total_buy - total_sell

        cvd = np.zeros(n, dtype=float)
        running  = 0.0
        prev_day = None
        for i in range(n):
            if day[i] is not None and day[i] != prev_day:
                running  = 0.0
                prev_day = day[i]
            running += delta[i]
            cvd[i] = running
        self._cvd_arr = cvd

        x = np.arange(n)

        # Zero-referenced green/red fill (buy-/sell-pressure bias readable at
        # a glance) plus a crisp outline on top for the precise curve shape --
        # a plain single-color 1px line made the trend hard to read. Each
        # fill curve pins to 0 on the opposite side of zero from its own
        # sign, matching the pattern of a signed histogram/area chart rather
        # than drawing two overlapping flat-at-zero pens (which would leave
        # a visible seam line cutting across the other side's fill).
        cvd_pos = np.where(cvd >= 0, cvd, 0.0)
        cvd_neg = np.where(cvd < 0, cvd, 0.0)
        pos_fill = pg.PlotCurveItem(
            x=x, y=cvd_pos, pen=None,
            fillLevel=0.0, brush=_qc(_GREEN, 55),
        )
        neg_fill = pg.PlotCurveItem(
            x=x, y=cvd_neg, pen=None,
            fillLevel=0.0, brush=_qc(_RED, 55),
        )
        outline = pg.PlotCurveItem(
            x=x, y=cvd,
            pen=pg.mkPen("#42a5f5", width=1.3),
        )
        for item in (pos_fill, neg_fill, outline):
            self._plot_cvd.addItem(item)
            self._cvd_items.append(item)

    def _clear_adi_items(self) -> None:
        for item in self._adi_items:
            self._plot_adi.removeItem(item)
        self._adi_items.clear()
        self._adi_arr = None

    def _draw_adi(self, klines: pd.DataFrame) -> None:
        """Draw the Accumulation/Distribution Line in the A/D subplot.

        Pure OHLCV indicator (ta.volume.AccDistIndexIndicator) -- estimates
        buy/sell pressure from where each bar's close sits within its own
        high-low range, weighted by bar volume. Unlike CVD, it needs no tick
        data and so has no NEUTRAL-volume blind spot, but it's a price-action
        proxy for direction rather than actual tagged trade sides.
        """
        adi = AccDistIndexIndicator(
            high=klines["high"], low=klines["low"], close=klines["close"],
            volume=klines["volume"].fillna(0),
        ).acc_dist_index().values

        x = np.arange(len(klines))
        line = pg.PlotCurveItem(
            x=x, y=adi,
            pen=pg.mkPen("#66bb6a", width=1),
        )
        self._plot_adi.addItem(line)
        self._adi_items.append(line)
        self._adi_arr = adi

    def _clear_kd_band_items(self) -> None:
        for item in self._kd_band_items:
            self._plot_c.removeItem(item)
        self._kd_band_items.clear()

    def _draw_kd_band(self, klines: pd.DataFrame) -> None:
        """Draw KD fast/slow midline ribbon on the main candlestick chart.

        Visual layers (bottom-to-top):
          1. Filled ribbon between mid1 and mid2 — gold alpha-30 when fast is
             above slow (bullish spread), blue alpha-30 when fast is below.
          2. mid1 line (gold, fast channel midpoint).
          3. mid2 line (blue, slow channel midpoint).

        The fill is split at crossover points so each segment gets the correct
        directional color without bleed from the opposite side.
        """
        if len(klines) < _KD_SLOW + 5:
            return

        warmup = self._warmup if self._warmup is not None else klines
        kd     = compute_kd(warmup, fast=_KD_FAST, slow=_KD_SLOW)

        n    = len(klines)
        mid1 = kd["mid1"].values[-n:].astype(float)
        mid2 = kd["mid2"].values[-n:].astype(float)
        nw   = len(mid1)   # warmup may be shorter than klines in Live mode
        x    = np.arange(n - nw, n, dtype=float)

        # Find zero-crossings of (mid1 - mid2) for directional fill segments.
        diff = mid1 - mid2
        crosses: list[float] = []
        for k in range(1, nw):
            if diff[k - 1] * diff[k] < 0:
                # Linear interpolation of crossing x-position
                t = diff[k - 1] / (diff[k - 1] - diff[k])
                crosses.append(k - 1 + t)

        # Build per-segment fill items between crossover boundaries
        boundaries = [0.0] + crosses + [float(nw - 1)]
        for seg_idx in range(len(boundaries) - 1):
            x0_f = boundaries[seg_idx]
            x1_f = boundaries[seg_idx + 1]
            i0   = int(np.floor(x0_f))
            i1   = int(np.ceil(x1_f)) + 1
            i1   = min(i1, nw)
            if i1 <= i0 + 1:
                continue

            seg_x    = x[i0:i1]
            seg_mid1 = mid1[i0:i1]
            seg_mid2 = mid2[i0:i1]

            # Determine color from midpoint of this segment
            mid_i   = (i0 + i1) // 2
            is_bull = diff[min(mid_i, nw - 1)] >= 0
            color   = _UP if is_bull else _DOWN

            fill = pg.FillBetweenItem(
                pg.PlotCurveItem(x=seg_x, y=seg_mid1),
                pg.PlotCurveItem(x=seg_x, y=seg_mid2),
                brush=_qc(color, 35),
            )
            self._plot_c.addItem(fill, ignoreBounds=True)
            self._kd_band_items.append(fill)

        # Draw the two midlines on top of the fills
        line_mid1 = pg.PlotCurveItem(
            x=x, y=mid1,
            pen=pg.mkPen("#42a5f5", width=1),
        )
        line_mid2 = pg.PlotCurveItem(
            x=x, y=mid2,
            pen=pg.mkPen(_UP, width=1),
        )
        self._plot_c.addItem(line_mid1, ignoreBounds=True)
        self._plot_c.addItem(line_mid2, ignoreBounds=True)
        self._kd_band_items.extend([line_mid1, line_mid2])

    # ── Trade Review ─────────────────────────────────────────────────────────

    def _load_trade_review(self) -> None:
        """Load a trade by ID from DB and switch to review mode."""
        trade_id = self._trade_id_edit.text().strip()
        if not trade_id:
            return

        row, source = _load_trade_from_db(trade_id)
        if row is None:
            QMessageBox.warning(
                self, "Not found",
                f"Trade ID not found in any DB:\n{trade_id}\n\n"
                "Check backtest.duckdb or review_trades.duckdb.",
            )
            return

        # Parse config JSON
        config: dict = {}
        raw_cfg = row.get("config_json") or row.get("signal_params")
        if raw_cfg:
            config = json.loads(raw_cfg) if isinstance(raw_cfg, str) else raw_cfg

        # Resolve display TF (prefer HTF/trend_tf for context)
        tf_alias = {"60m": "1h"}
        trend_tf = config.get("trend_tf", "")
        tf       = tf_alias.get(trend_tf, trend_tf)
        if tf not in TIMEFRAME_MAP:
            entry_tf = config.get("entry_tf", "15m")
            tf       = tf_alias.get(entry_tf, entry_tf)
        if tf not in TIMEFRAME_MAP:
            tf = "15m"

        # Date from entry_time
        entry_str = str(row.get("entry_time") or "")
        date_str  = entry_str[:10] if len(entry_str) >= 10 else ""
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            date_str = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")

        # Populate toolbar
        self._code_edit.setText(row.get("symbol", ""))
        self._tf_combo.setCurrentText(tf)
        self._mode_combo.setCurrentText("Historical")
        self._date_edit.setText(date_str)
        self._date_edit.setEnabled(True)

        self._trade_record = {**row, "_source": source, "_config": config}
        self._log(
            f"Loaded trade {trade_id[:8]}… ({source}) → "
            f"{row.get('symbol')} {tf} {date_str}"
        )
        self._trigger_fetch()

    def _clear_trade_review(self) -> None:
        """Remove trade review overlay and reset state."""
        self._trade_record = None
        self._trade_id_edit.clear()
        self._clear_trade_items()
        self._log("Trade review cleared.")

    def _clear_trade_items(self) -> None:
        for item in self._trade_items:
            self._plot_c.removeItem(item)
        self._trade_items.clear()

    def _draw_trade_review(self, klines: pd.DataFrame) -> None:
        """Draw entry/exit arrows, SL/TP lines on the chart."""
        trade  = self._trade_record
        if trade is None:
            return

        direction   = str(trade.get("direction", "")).upper()
        is_bull     = direction in ("LONG", "BULL")
        entry_price = _safe_float(trade.get("entry_price"))
        exit_price  = _safe_float(trade.get("exit_price"))
        sl_price    = _safe_float(trade.get("sl_price"))
        tp_price    = _safe_float(trade.get("tp_price"))
        entry_time  = str(trade.get("entry_time") or "")
        exit_time   = str(trade.get("exit_time")  or "")
        result      = str(trade.get("result") or "")

        times = klines["time_key"].astype(str).values
        n     = len(klines)

        def _bar_idx(ts: str) -> int | None:
            if not ts:
                return None
            pos = int(np.searchsorted(times, ts[:16], side="left"))
            return min(pos, n - 1)

        entry_idx = _bar_idx(entry_time)
        exit_idx  = _bar_idx(exit_time)

        # SL line
        if sl_price:
            sl_line = pg.InfiniteLine(
                pos=sl_price, angle=0, movable=False,
                pen=pg.mkPen(_RED, width=1, style=Qt.PenStyle.DashLine),
                label=f"SL {sl_price:.2f}",
                labelOpts={"color": _RED, "position": 0.05},
            )
            self._plot_c.addItem(sl_line)
            self._trade_items.append(sl_line)

        # TP line
        if tp_price:
            tp_line = pg.InfiniteLine(
                pos=tp_price, angle=0, movable=False,
                pen=pg.mkPen(_GREEN, width=1, style=Qt.PenStyle.DashLine),
                label=f"TP {tp_price:.2f}",
                labelOpts={"color": _GREEN, "position": 0.05},
            )
            self._plot_c.addItem(tp_line)
            self._trade_items.append(tp_line)

        # Entry arrow: upward for long, downward for short
        if entry_idx is not None and entry_price:
            arrow_col = _GREEN if is_bull else _RED
            bar       = klines.iloc[entry_idx]
            arr       = pg.ArrowItem(
                pos=(entry_idx, entry_price),
                angle=90 if is_bull else -90,
                headLen=14,
                tipAngle=30,
                brush=_qc(arrow_col),
                pen=pg.mkPen(arrow_col, width=1.5),
            )
            self._plot_c.addItem(arr)
            self._trade_items.append(arr)

            lbl = pg.TextItem(
                text=f"{'▲' if is_bull else '▼'} {entry_price:.2f}",
                color=arrow_col, anchor=(0.0, 0.5),
            )
            lbl.setFont(QFont("Monospace", 7))
            lbl.setPos(entry_idx + 0.5, entry_price)
            self._plot_c.addItem(lbl)
            self._trade_items.append(lbl)

        # Exit marker: circle (win) / X (loss) / square (open)
        if exit_idx is not None and exit_price:
            is_win  = result == "win"
            exc_col = _GREEN if is_win else (_RED if result == "loss" else _GREY)
            sym     = "o" if is_win else ("x" if result == "loss" else "s")
            pt = pg.ScatterPlotItem(
                x=[exit_idx], y=[exit_price],
                symbol=sym, size=12,
                brush=_qc(exc_col, 200), pen=pg.mkPen(exc_col),
            )
            self._plot_c.addItem(pt)
            self._trade_items.append(pt)

            lbl = pg.TextItem(
                text=f"{'✓' if is_win else '✕'} {exit_price:.2f}",
                color=exc_col, anchor=(0.0, 0.5),
            )
            lbl.setFont(QFont("Monospace", 7))
            lbl.setPos(exit_idx + 0.5, exit_price)
            self._plot_c.addItem(lbl)
            self._trade_items.append(lbl)

        # Entry → exit shaded region
        if (entry_idx is not None and exit_idx is not None
                and entry_price and exit_price
                and entry_idx < exit_idx):
            region_col = _qc(_GREEN if is_bull else _RED, 20)
            region     = pg.LinearRegionItem(
                values=[entry_idx, exit_idx],
                orientation="vertical",
                brush=region_col,
                pen=pg.mkPen(None),
                movable=False,
            )
            self._plot_c.addItem(region)
            self._trade_items.append(region)

    # ── Scanner Signals overlay ───────────────────────────────────────────────

    def _toggle_scanner_signals(self, checked: bool) -> None:
        if checked:
            if self._klines is not None:
                self._load_scanner_signals(self._klines)
        else:
            self._clear_scanner_signal_items()

    def _clear_scanner_signal_items(self) -> None:
        for item in self._scanner_signal_items:
            self._plot_c.removeItem(item)
        self._scanner_signal_items.clear()

    def _load_scanner_signals(self, klines: pd.DataFrame) -> None:
        """Read open signals for the current symbol from db/signals.db and render them."""
        import pathlib as _pl
        db_path = _pl.Path(__file__).parent.parent / "db" / "signals.db"
        if not db_path.exists():
            return
        code = self._code_edit.text().strip()
        if not code:
            return
        try:
            from db.signals import SignalsDB
            with SignalsDB(db_path, read_only=False) as db:
                sigs = db.get_open_signals(code)
        except Exception as exc:
            self._log(f"Scanner signals load error: {exc}")
            return

        for sig in sigs:
            self._render_scanner_signal(sig, klines)

    def _render_scanner_signal(self, sig: dict, klines: pd.DataFrame) -> None:
        """Draw one scanner signal's entry zone, SL, TP, and label on the chart."""
        is_bull = sig.get("direction", "") == "bull"
        col     = _GREEN if is_bull else _RED
        qcol    = _qc(col)

        top    = _safe_float(sig.get("entry_zone_top"))
        bottom = _safe_float(sig.get("entry_zone_bottom"))
        sl     = _safe_float(sig.get("sl_price"))
        tp     = _safe_float(sig.get("tp_price"))
        rr     = sig.get("rr_ratio", 0)

        # Entry zone band (horizontal LinearRegionItem spanning full chart width)
        if top and bottom:
            zone = pg.LinearRegionItem(
                values=[bottom, top],
                orientation="horizontal",
                brush=_qc(col, 35),
                pen=pg.mkPen(col, width=0.5, style=Qt.PenStyle.DotLine),
                movable=False,
            )
            self._plot_c.addItem(zone)
            self._scanner_signal_items.append(zone)

        # SL line
        if sl:
            sl_line = pg.InfiniteLine(
                pos=sl, angle=0, movable=False,
                pen=pg.mkPen(_RED, width=1, style=Qt.PenStyle.DashLine),
                label=f"SL {sl:.2f}",
                labelOpts={"color": _RED, "position": 0.02},
            )
            self._plot_c.addItem(sl_line)
            self._scanner_signal_items.append(sl_line)

        # TP line
        if tp:
            tp_line = pg.InfiniteLine(
                pos=tp, angle=0, movable=False,
                pen=pg.mkPen(_GREEN, width=1, style=Qt.PenStyle.DashLine),
                label=f"TP {tp:.2f}",
                labelOpts={"color": _GREEN, "position": 0.02},
            )
            self._plot_c.addItem(tp_line)
            self._scanner_signal_items.append(tp_line)

        # Label near the right edge of the entry zone
        if top and bottom:
            times    = klines["time_key"].astype(str).values
            sig_time = str(sig.get("signal_time", ""))[:16]
            x_idx    = int(np.searchsorted(times, sig_time, side="left")) if sig_time else len(klines) - 1
            x_idx    = min(x_idx, len(klines) - 1)
            mid      = (top + bottom) / 2.0
            arrow    = "▲" if is_bull else "▼"
            lbl = pg.TextItem(
                text=f"{arrow} {sig.get('direction','').upper()}  RR {rr:.1f}",
                color=col,
                anchor=(0.0, 0.5),
            )
            lbl.setFont(QFont("Monospace", 7))
            lbl.setPos(x_idx + 0.5, mid)
            self._plot_c.addItem(lbl)
            self._scanner_signal_items.append(lbl)

    # ── Session vol profile ───────────────────────────────────────────────────

    def _disconnect_profile_pins(self) -> None:
        """Disconnect all accumulated label-pinning slots from the profile ViewBox."""
        vb = self._profile_widget.getViewBox()
        for conn in self._profile_pin_conns:
            try:
                vb.sigRangeChanged.disconnect(conn)
            except Exception:
                pass
        self._profile_pin_conns.clear()

    def _rebuild_session_profile(self) -> None:
        if self._klines is None:
            return
        self._disconnect_profile_pins()
        pw = self._profile_widget
        pw.clear()
        vb = pw.getViewBox()
        # Disable Y auto-range before adding items so the crosshair hline at
        # y=0 (default position) cannot pull the view range away from the data.
        vb.enableAutoRange(axis=pg.ViewBox.YAxis, enable=False)
        pw.addItem(self._profile_hline, ignoreBounds=True)

        range_val = self._get_range_val()
        klines    = apply_profile_range(self._klines, range_val, self._candle_mins)
        klines    = self._filter_sessions(klines)
        if klines.empty:
            return

        lo = float(klines["low"].min())
        hi = float(klines["high"].max())
        n_bins = self._profile_bins_spin.value()
        # Tick-first, OHLCV-fallback -- shared with the Range Volume Profile
        # via _compute_profile_bins() (used to be its own inlined copy of the
        # same hybrid logic; unified so the two panels can't quietly drift
        # apart). Bars with real tick coverage place volume at its actual
        # traded price via np.digitize; only bars with no tick data for their
        # bucket fall back to spreading the bar's volume evenly across every
        # bin its high-low range touches.
        centers, volumes, _used_ticks, _stats = _compute_profile_bins(
            klines, self._ticks, self._candle_mins, n_bins=n_bins)
        if centers.size == 0:
            return

        # Horizontal bars: x0=0 (left edge), x1=volume (right edge), y=price centre
        bar = pg.BarGraphItem(
            x0=np.zeros(n_bins), x1=volumes,
            y=centers, height=(centers[1] - centers[0]) * 0.9,
            brush=_qc(_GOLD, 80), pen=pg.mkPen(None),
        )
        pw.addItem(bar)
        pw.getPlotItem().setLabel("top", range_val.upper(),
                                  **{"color": _FG, "size": "8pt"})

        poc, vah, val = _compute_poc_vah_val(centers, volumes)

        poc_line = pg.InfiniteLine(
            pos=poc, angle=0, movable=False,
            pen=pg.mkPen(_RED, width=1),
        )
        poc_label = pg.TextItem(
            text=f"POC {poc:.2f}", color=_RED,
            fill=pg.mkBrush(_qc(_BG_TIP, 180)),
            anchor=(0.0, 1.0),
        )
        poc_label.setFont(QFont("Monospace", 7))
        pw.addItem(poc_line,  ignoreBounds=True)
        pw.addItem(poc_label, ignoreBounds=True)

        def _pin_poc_label() -> None:
            xlo = vb.viewRange()[0][0]
            poc_label.setPos(xlo, poc)

        conn = vb.sigRangeChanged.connect(lambda *_: _pin_poc_label())
        self._profile_pin_conns.append(conn)
        _pin_poc_label()

        for price, lbl in [(vah, "VAH"), (val, "VAL")]:
            va_line = pg.InfiniteLine(
                pos=price, angle=0, movable=False,
                pen=pg.mkPen(_GOLD, width=1, style=Qt.PenStyle.DashLine),
            )
            va_label = pg.TextItem(
                text=f"{lbl} {price:.2f}", color=_GOLD,
                fill=pg.mkBrush(_qc(_BG_TIP, 180)),
                anchor=(0.0, 0.0),
            )
            va_label.setFont(QFont("Monospace", 7))
            pw.addItem(va_line,  ignoreBounds=True)
            pw.addItem(va_label, ignoreBounds=True)

            def _pin_va(lbl_item=va_label, p=price) -> None:
                xlo = vb.viewRange()[0][0]
                lbl_item.setPos(xlo, p)

            conn2 = vb.sigRangeChanged.connect(lambda *_, f=_pin_va: f())
            self._profile_pin_conns.append(conn2)
            _pin_va()

        # Project the current (latest) price onto the profile -- solid line,
        # distinct from POC/VAH/VAL's dashed style, so it reads as "where
        # price is now" vs. "where the volume concentrated". Deliberately
        # read from self._klines (unfiltered), NOT the session-filtered
        # `klines` used for the histogram/POC/VAH/VAL -- if e.g. "Night" is
        # unchecked while price is currently in the overnight session, the
        # filtered frame's last row is a stale pre-session close, not the
        # true current price, and this line/label would silently go stale.
        #
        # In Live mode, prefer the tick stream's last trade price over the
        # kline close: self._klines only refreshes once per _trigger_fetch
        # cycle (the "Refresh(s)" interval, e.g. every 5-30s), while
        # _last_tick_price updates on every push from moomoo -- reading the
        # kline close here made this line visibly lag moomoo's own
        # real-time price between refreshes.
        last_price = float(self._klines["close"].iloc[-1])
        if self._mode_combo.currentText() == "Live" and self._last_tick_price > 0:
            last_price = self._last_tick_price
        price_line = pg.InfiniteLine(
            pos=last_price, angle=0, movable=False,
            pen=pg.mkPen("#42a5f5", width=1.5),
        )
        price_label = pg.TextItem(
            text=f"Last {last_price:.2f}", color="#42a5f5",
            fill=pg.mkBrush(_qc(_BG_TIP, 180)),
            anchor=(0.0, 1.0),
        )
        price_label.setFont(QFont("Monospace", 7))
        pw.addItem(price_line,  ignoreBounds=True)
        pw.addItem(price_label, ignoreBounds=True)

        def _pin_price_label() -> None:
            xlo = vb.viewRange()[0][0]
            price_label.setPos(xlo, last_price)

        conn3 = vb.sigRangeChanged.connect(lambda *_: _pin_price_label())
        self._profile_pin_conns.append(conn3)
        _pin_price_label()

        max_vol = float(volumes.max())
        vb.enableAutoRange(axis=pg.ViewBox.XAxis, enable=False)
        vb.setXRange(0, max_vol * 1.15, padding=0)
        # Extend the Y-range to cover last_price too -- it can sit outside
        # the session-filtered histogram's own [lo, hi] (e.g. price moved
        # further during a session that's excluded from the profile).
        y_lo = min(lo, last_price)
        y_hi = max(hi, last_price)
        y_pad = (y_hi - y_lo) * 0.10
        vb.setYRange(y_lo - y_pad, y_hi + y_pad, padding=0)

    def _filter_sessions(self, klines: pd.DataFrame) -> pd.DataFrame:
        """Keep only rows whose time falls in the active session windows."""
        active = self._active_sessions()
        if not active or klines.empty:
            return klines
        sessions = {
            "pre":     (4  * 60,      9  * 60 + 30),
            "regular": (9  * 60 + 30, 16 * 60),
            "post":    (16 * 60,      20 * 60),
            "night":   (20 * 60,      28 * 60),
        }
        def _in_session(ts: str) -> bool:
            try:
                t    = datetime.strptime(str(ts)[11:16], "%H:%M")
                mins = t.hour * 60 + t.minute
            except ValueError:
                return True
            for s in active:
                lo, hi = sessions.get(s, (0, 1440))
                # "night" is stored as (1200, 1680) -- 20:00 through the
                # next day's 04:00 -- so hi > 24*60 marks a window that
                # crosses midnight. mins is always a raw hour*60+minute
                # (0-1439, no next-day concept), so it can never reach a
                # hi of 1680 on its own: checking lo <= mins < hi directly
                # would only ever match 20:00-23:59 and silently exclude
                # every post-midnight timestamp (00:00-03:59) from "night"
                # -- reported as the session volume profile's POC/VAH/VAL
                # freezing at whatever they were right before midnight
                # instead of covering the rest of the still-ongoing
                # overnight session. The previous `if lo < hi / else` split
                # used lo >= hi as the "wraps" signal, which is never true
                # for (1200, 1680) even though it does wrap -- checking
                # hi > 24*60 instead is what actually identifies it.
                if hi > 24 * 60:
                    if mins >= lo or mins < (hi - 24 * 60):
                        return True
                elif lo <= mins < hi:
                    return True
            return False
        mask = klines["time_key"].astype(str).apply(_in_session)
        return klines[mask]

    # ── Tick profile (single candle on hover) ─────────────────────────────────

    def _draw_tick_profile(self, pd_: dict, header: str,
                           y_lo: float | None = None,
                           y_hi: float | None = None) -> None:
        """Render tick buy/sell/neutral bars into the tick profile panel.

        pd_ maps price → {buy, sell, neutral, buy_s, …} (same format as self._ticks values).
        header is placed in the top label of the panel.
        y_lo/y_hi: explicit Y range (use candle OHLC for single-candle mode so stray
        ticks outside the candle range don't distort the scale).  When omitted the
        range is derived from min/max of prices in pd_ (suitable for range accumulation).
        """
        show_s = self._ind("tick_s")
        show_m = self._ind("tick_m")
        show_l = self._ind("tick_l")
        prices = sorted(pd_.keys())
        # When candle bounds are provided, restrict to prices within [y_lo, y_hi]
        # so stray ticks from adjacent bars don't appear in the profile.
        if y_lo is not None and y_hi is not None and prices:
            eps = max((y_hi - y_lo) * 0.01, 0.001)
            prices = [p for p in prices if y_lo - eps <= p <= y_hi + eps]
        if not prices:
            pw = self._tick_profile_widget
            pw.clear()
            pw.addItem(self._tick_profile_hline)
            return
        has_breakdown = "buy_s" in next(iter(pd_.values()), {})
        if has_breakdown and (show_s or show_m or show_l):
            buys = [
                (pd_[p].get("buy_s", 0) if show_s else 0) +
                (pd_[p].get("buy_m", 0) if show_m else 0) +
                (pd_[p].get("buy_l", 0) if show_l else 0)
                for p in prices
            ]
            sells = [
                (pd_[p].get("sell_s", 0) if show_s else 0) +
                (pd_[p].get("sell_m", 0) if show_m else 0) +
                (pd_[p].get("sell_l", 0) if show_l else 0)
                for p in prices
            ]
            neutrals = [
                (pd_[p].get("neutral_s", 0) if show_s else 0) +
                (pd_[p].get("neutral_m", 0) if show_m else 0) +
                (pd_[p].get("neutral_l", 0) if show_l else 0)
                for p in prices
            ]
        else:
            buys     = [pd_[p]["buy"]            for p in prices]
            sells    = [pd_[p]["sell"]           for p in prices]
            neutrals = [pd_[p].get("neutral", 0) for p in prices]

        pw = self._tick_profile_widget
        pw.clear()
        pw.addItem(self._tick_profile_hline)

        bin_h = (max(prices) - min(prices)) / max(len(prices), 1) * 0.9 if prices else 0.01
        bin_h = max(bin_h, 0.001)

        buys_arr    = np.array(buys,     dtype=float)
        sells_arr   = np.array(sells,    dtype=float)
        neutral_arr = np.array(neutrals, dtype=float)
        buys_log    = np.log1p(buys_arr)
        sells_log   = np.log1p(sells_arr)
        neutral_log = np.log1p(neutral_arr)
        zeros       = np.zeros(len(prices))

        half_n = neutral_log / 2
        neutral_bar = pg.BarGraphItem(
            x0=-half_n, x1=half_n,
            y=prices, height=bin_h,
            brush=_qc(_GREY, 80), pen=pg.mkPen(None),
        )
        pw.addItem(neutral_bar)

        red_up   = self._ind("red_up")
        buy_col  = _RED   if red_up else _GREEN
        sell_col = _GREEN if red_up else _RED

        # Imbalance highlight: price levels where the dominant side's volume
        # is at least `ratio` times the opposite side's get a gold outline
        # (instead of no pen) and the price printed above the bar -- makes
        # the price levels worth noticing pop out of an otherwise dense,
        # narrow panel without labelling every single level. Capped to the
        # top 3 qualifying levels by dominant-side volume (not by ratio,
        # which favours a tiny order with an extreme but visually
        # insignificant ratio over a much longer bar that's "merely" very
        # imbalanced) so a busy tick set can't crowd the narrow panel.
        ratio_thresh = self._tick_imb_ratio.value()
        dominant     = np.maximum(buys_arr, sells_arr)
        minor        = np.minimum(buys_arr, sells_arr)
        ratio        = dominant / np.maximum(minor, 1.0)
        qualifies    = ratio >= ratio_thresh
        buy_dom      = buys_arr >= sells_arr
        pass_idx     = np.where(qualifies)[0]
        if len(pass_idx) > 3:
            pass_idx = pass_idx[np.argsort(-dominant[pass_idx])[:3]]
        imbalanced = np.zeros(len(prices), dtype=bool)
        imbalanced[pass_idx] = True
        no_pen       = pg.mkPen(None)
        hi_pen       = pg.mkPen(_GOLD, width=1.5)

        if self._ind("tick_net"):
            # Single net (buy - sell) bar per price level: one-sided, colored
            # by sign, instead of two diverging buy/sell bars.
            net       = buys_arr - sells_arr
            net_log   = np.log1p(np.abs(net))
            net_x0    = np.where(net >= 0, zeros, -net_log)
            net_x1    = np.where(net >= 0, net_log, zeros)
            net_brush = [_qc(buy_col, 180) if d >= 0 else _qc(sell_col, 180) for d in net]
            net_pens  = [hi_pen if imb else no_pen for imb in imbalanced]
            net_bar = pg.BarGraphItem(
                x0=net_x0, x1=net_x1,
                y=prices, height=bin_h,
                brushes=net_brush, pens=net_pens,
            )
            pw.addItem(net_bar)
        else:
            buy_pens  = [hi_pen if (imb and dom) else no_pen
                         for imb, dom in zip(imbalanced, buy_dom)]
            sell_pens = [hi_pen if (imb and not dom) else no_pen
                         for imb, dom in zip(imbalanced, buy_dom)]
            buy_bar = pg.BarGraphItem(
                x0=zeros, x1=buys_log,
                y=prices, height=bin_h,
                brush=_qc(buy_col, 180), pens=buy_pens,
            )
            sell_bar = pg.BarGraphItem(
                x0=-sells_log, x1=zeros,
                y=prices, height=bin_h,
                brush=_qc(sell_col, 180), pens=sell_pens,
            )
            pw.addItem(buy_bar)
            pw.addItem(sell_bar)

        # Label above each row's center (not at the bar's tip): a bar's tip
        # can sit anywhere from near-zero to the panel edge depending on its
        # own width, so tip-anchored text would either overlap the opposite
        # bar (wide bars, few rows) or float awkwardly far from a narrow one.
        # Above-center is stable regardless of bar width/row count.
        y_off = bin_h * 0.5 * 0.85
        for i, price in enumerate(prices):
            if not imbalanced[i]:
                continue
            dom_col = buy_col if buys_arr[i] >= sells_arr[i] else sell_col
            label = pg.TextItem(
                text=f"{price:.2f}", color=dom_col,
                fill=pg.mkBrush(_qc(_BG_TIP, 130)), anchor=(0.5, 1.0),
            )
            label.setFont(QFont("Monospace", 7))
            label.setPos(0.0, price + y_off)
            pw.addItem(label, ignoreBounds=True)

        total_buy  = sum(buys)
        total_sell = sum(sells)
        delta      = total_buy - total_sell
        sign       = "+" if delta >= 0 else ""
        delta_str  = (f"{sign}{delta/1000:.0f}K" if abs(delta) >= 1000 else f"{sign}{delta}")
        d_col      = buy_col if delta >= 0 else sell_col
        title_html = (
            f"<span style='color:{_FG}'>{header}</span>"
            f"&nbsp;&nbsp;<span style='color:{d_col}'>Δ {delta_str}</span>"
        )
        pw.getPlotItem().setLabel("bottom", "", **{})
        pw.getPlotItem().setLabel("top", title_html, **{"size": "7pt"})

        # Sync Y range to current main chart viewport so profile aligns spatially.
        # sigRangeChanged doesn't fire on hover, so we apply it once after drawing.
        _, (ylo, yhi) = self._plot_c.vb.viewRange()
        pw.setYRange(ylo, yhi, padding=0)

    def _show_tick_profile(self, candle_idx: int) -> None:
        if self._range_region is not None:
            return
        if self._klines is None or self._ticks is None:
            return
        if candle_idx < 0 or candle_idx >= len(self._klines):
            return

        # Always update so the cache check in _on_mouse_move stays consistent.
        self._last_hover_idx = candle_idx

        def _clear_panel():
            pw = self._tick_profile_widget
            pw.clear()
            pw.addItem(self._tick_profile_hline)

        row = self._klines.iloc[candle_idx]
        try:
            bar_end = datetime.strptime(
                str(row["time_key"])[:16], "%Y-%m-%d %H:%M")
            bk = candle_start(
                bar_end - timedelta(minutes=self._candle_mins), self._candle_mins)
        except ValueError:
            _clear_panel()
            return

        pd_ = self._ticks.get(bk)
        if not pd_:
            _clear_panel()
            return

        self._draw_tick_profile(
            pd_, str(row["time_key"])[:16],
            y_lo=float(row.get("low", min(pd_.keys()))),
            y_hi=float(row.get("high", max(pd_.keys()))),
        )

    def _show_range_tick_profile(self, i0: int, i1: int) -> None:
        """Accumulate tick data across candles i0..i1 and render in tick panel."""
        if self._klines is None or self._ticks is None:
            return
        cm = self._candle_mins
        merged: dict = {}
        for idx in range(i0, i1 + 1):
            row = self._klines.iloc[idx]
            try:
                bar_end = datetime.strptime(str(row["time_key"])[:16], "%Y-%m-%d %H:%M")
                bk = candle_start(bar_end - timedelta(minutes=cm), cm)
            except ValueError:
                continue
            pd_ = self._ticks.get(bk)
            if not pd_:
                continue
            for price, counts in pd_.items():
                bucket = merged.setdefault(price, {})
                for k, v in counts.items():
                    bucket[k] = bucket.get(k, 0) + v
        if not merged:
            pw = self._tick_profile_widget
            pw.clear()
            pw.addItem(self._tick_profile_hline)
            n = i1 - i0 + 1
            lbl = pg.TextItem(text=f"Range {n} bars\nNo tick data", color=_GREY, anchor=(0.5, 0.5))
            lbl.setFont(QFont("Monospace", 8))
            lbl.setPos(0.0, 0.0)
            pw.addItem(lbl)
            pw.getPlotItem().setLabel("top", f"Range  {n} bars", **{"color": _GREY, "size": "7pt"})
            return
        self._draw_tick_profile(merged, f"Range  {i1 - i0 + 1} bars")

    # ── Tick profile Y range sync ─────────────────────────────────────────────

    # ── Range Volume Profile ──────────────────────────────────────────────────

    def _toggle_range_profile(self, checked: bool) -> None:
        if checked:
            if self._klines is None or self._klines.empty:
                self._range_profile_btn.setChecked(False)
                return
            # Default span: middle 40 % of currently visible range
            xlo, xhi = self._plot_c.vb.viewRange()[0]
            span = xhi - xlo
            x0 = xlo + span * 0.30
            x1 = xlo + span * 0.70
            self._range_region = pg.LinearRegionItem(
                values=[x0, x1],
                orientation="vertical",
                movable=True,
                brush=pg.mkBrush(_qc("#1565c0", 40)),
                pen=pg.mkPen(_qc("#42a5f5", 180), width=1),
            )
            self._range_region.setZValue(10)
            self._plot_c.addItem(self._range_region)
            self._range_region.sigRegionChanged.connect(self._on_range_region_changed)
            self._range_last_indices = (-1, -1)
            self._rebuild_range_profile()
        else:
            self._range_profile_timer.stop()
            if self._range_region is not None:
                self._plot_c.removeItem(self._range_region)
                self._range_region = None
            self._clear_range_inline()
            self._range_last_indices = (-1, -1)
            pw = self._tick_profile_widget
            pw.clear()
            pw.addItem(self._tick_profile_hline)
            if self._last_hover_idx is not None:
                self._show_tick_profile(self._last_hover_idx)
            self._rebuild_session_profile()

    def _on_range_region_changed(self) -> None:
        self._range_profile_timer.start(150)

    def _clear_range_inline(self) -> None:
        for item in self._range_profile_inline:
            self._plot_c.removeItem(item)
        self._range_profile_inline.clear()

    def _compute_range_profile(
        self, i0: int, i1: int
    ) -> tuple[np.ndarray, np.ndarray, bool, dict]:
        return _compute_profile_bins(
            self._klines.iloc[i0 : i1 + 1], self._ticks, self._candle_mins)

    def _rebuild_range_profile(self) -> None:
        if self._range_region is None or self._klines is None:
            return
        n = len(self._klines)
        if n == 0:
            return

        rx0, rx1 = self._range_region.getRegion()
        i0 = max(0, min(n - 1, int(round(min(rx0, rx1)))))
        i1 = max(0, min(n - 1, int(round(max(rx0, rx1)))))
        if (i0, i1) == self._range_last_indices:
            return
        self._range_last_indices = (i0, i1)

        centers, volumes, used_ticks, stats = self._compute_range_profile(i0, i1)
        if centers.size == 0:
            return

        max_vol = float(volumes.max())
        if max_vol <= 0:
            return

        poc, vah, val = _compute_poc_vah_val(centers, volumes)

        bin_h    = float(centers[1] - centers[0]) if len(centers) > 1 else 1.0
        n_bars   = i1 - i0
        src_lbl  = "tick" if used_ticks else "OHLCV"

        def _fmt_vol(v: float) -> str:
            if v >= 1_000_000:
                return f"{v/1_000_000:.1f}M"
            if v >= 1_000:
                return f"{v/1_000:.0f}K"
            return f"{v:.0f}"

        # "total" = buy + sell + neutral from tick-covered bars only (a closed
        # sum -- shown here so the four numbers are self-checkable instead of
        # leaving a silent gap). moomoo's ticker_direction tags a lot of
        # same-price prints NEUTRAL, so it's often the largest of the three.
        # "medium" is a size breakdown *within* buy+sell, not a 5th additive
        # bucket, hence the "*" -- it will always be <= buy + sell.
        stats_lbl = (
            f"Tot {_fmt_vol(stats['total'])}  "
            f"Buy {_fmt_vol(stats['buy'])}  "
            f"Sell {_fmt_vol(stats['sell'])}  "
            f"Neu {_fmt_vol(stats['neutral'])}  "
            f"Mid* {_fmt_vol(stats['medium'])}"
        )
        stats_tip = (
            "Tot = Buy + Sell + Neutral, tick-covered bars only "
            "(bars with no local tick data aren't counted here, even though "
            "they still show up as bars in the profile).\n"
            "Neu = moomoo ticker_direction NEUTRAL prints (often same-price/"
            "zero-tick trades) -- frequently the largest bucket.\n"
            "Mid* = medium-lot volume already included inside Buy/Sell "
            "(buy_m + sell_m) -- not a separate bucket, so Buy+Sell+Mid != Tot."
        )

        # ── Right panel ───────────────────────────────────────────────────────
        self._disconnect_profile_pins()
        pw = self._profile_widget
        pw.clear()
        vb = pw.getViewBox()
        vb.enableAutoRange(axis=pg.ViewBox.YAxis, enable=False)
        pw.addItem(self._profile_hline, ignoreBounds=True)

        bar_item = pg.BarGraphItem(
            x0=np.zeros(len(centers)), x1=volumes,
            y=centers, height=bin_h * 0.9,
            brush=_qc("#42a5f5", 80), pen=pg.mkPen(None),
        )
        pw.addItem(bar_item)
        pw.getPlotItem().setLabel(
            "top", f"Range  {i1-i0+1}b  [{src_lbl}]",
            **{"color": "#42a5f5", "size": "7pt"},
        )

        def _add_panel_line(price: float, color: str,
                            style=Qt.PenStyle.SolidLine) -> pg.TextItem:
            line = pg.InfiniteLine(
                pos=price, angle=0, movable=False,
                pen=pg.mkPen(color, width=1, style=style),
            )
            lbl = pg.TextItem(
                text="", color=color,
                fill=pg.mkBrush(_qc(_BG_TIP, 180)),
                anchor=(0.0, 1.0),
            )
            lbl.setFont(QFont("Monospace", 7))
            pw.addItem(line,  ignoreBounds=True)
            pw.addItem(lbl,   ignoreBounds=True)
            return lbl

        poc_lbl = _add_panel_line(poc, _RED)
        vah_lbl = _add_panel_line(vah, _GOLD, Qt.PenStyle.DashLine)
        val_lbl = _add_panel_line(val, _GOLD, Qt.PenStyle.DashLine)
        # Project the current price (range's last bar) onto the profile --
        # solid accent line, distinct from POC/VAH/VAL's red/gold dashed style.
        last_price = float(self._klines["close"].iloc[i1])
        price_lbl  = _add_panel_line(last_price, "#42a5f5")

        stats_item = pg.TextItem(
            text=stats_lbl, color="#42a5f5",
            fill=pg.mkBrush(_qc(_BG_TIP, 180)),
            anchor=(0.0, 0.0),
        )
        stats_item.setFont(QFont("Monospace", 7))
        stats_item.setToolTip(stats_tip)
        pw.addItem(stats_item, ignoreBounds=True)

        def _pin_panel_labels() -> None:
            xlo = vb.viewRange()[0][0]
            yhi = vb.viewRange()[1][1]
            poc_lbl.setPos(xlo, poc);  poc_lbl.setText(f"POC {poc:.2f}")
            vah_lbl.setPos(xlo, vah);  vah_lbl.setText(f"VAH {vah:.2f}")
            val_lbl.setPos(xlo, val);  val_lbl.setText(f"VAL {val:.2f}")
            price_lbl.setPos(xlo, last_price); price_lbl.setText(f"Last {last_price:.2f}")
            stats_item.setPos(xlo, yhi)

        conn = vb.sigRangeChanged.connect(lambda *_: _pin_panel_labels())
        self._profile_pin_conns.append(conn)
        _pin_panel_labels()

        vb.enableAutoRange(axis=pg.ViewBox.XAxis, enable=False)
        vb.setXRange(0, max_vol * 1.15, padding=0)
        range_lo = float(centers[0])
        range_hi = float(centers[-1])
        y_pad = (range_hi - range_lo) * 0.10
        vb.setYRange(range_lo - y_pad, range_hi + y_pad, padding=0)

        # ── Inline overlay on main chart ──────────────────────────────────────
        self._clear_range_inline()

        # Bars grow rightward from i0; max bar fills 45 % of the range width.
        max_width  = max(n_bars * 0.45, 1.0)
        vol_norm   = volumes / max_vol * max_width
        x0_arr     = np.full(len(centers), float(i0))
        x1_arr     = x0_arr + vol_norm

        inline_bar = pg.BarGraphItem(
            x0=x0_arr, x1=x1_arr,
            y=centers, height=bin_h * 0.9,
            brush=_qc("#42a5f5", 35), pen=pg.mkPen(None),
        )
        inline_bar.setZValue(5)
        self._plot_c.addItem(inline_bar)
        self._range_profile_inline.append(inline_bar)

        # POC line (red solid) across the selected range
        poc_line = pg.PlotCurveItem(
            x=[float(i0), float(i1)], y=[poc, poc],
            pen=pg.mkPen(_RED, width=1),
        )
        poc_line.setZValue(6)
        self._plot_c.addItem(poc_line)
        self._range_profile_inline.append(poc_line)

        # VAH / VAL lines (gold dashed) across the selected range
        for price in (vah, val):
            va_line = pg.PlotCurveItem(
                x=[float(i0), float(i1)], y=[price, price],
                pen=pg.mkPen(_GOLD, width=1, style=Qt.PenStyle.DashLine),
            )
            va_line.setZValue(6)
            self._plot_c.addItem(va_line)
            self._range_profile_inline.append(va_line)

        # POC / VAH / VAL price labels pinned at the right edge of the selection
        label_x = float(i1) + 0.3
        for price, tag, color in [
            (poc, "POC", _RED),
            (vah, "VAH", _GOLD),
            (val, "VAL", _GOLD),
        ]:
            lbl = pg.TextItem(
                text=f"{tag} {price:.2f}", color=color,
                fill=pg.mkBrush(_qc(_BG_TIP, 160)),
                anchor=(0.0, 0.5),
            )
            lbl.setFont(QFont("Monospace", 7))
            lbl.setPos(label_x, price)
            lbl.setZValue(20)
            self._plot_c.addItem(lbl)
            self._range_profile_inline.append(lbl)

        self._log(
            f"Range profile | bars {i0}–{i1} ({i1-i0+1}) | "
            f"source={src_lbl} | POC={poc:.2f} VAH={vah:.2f} VAL={val:.2f} | "
            f"{stats_lbl}"
        )

        self._show_range_tick_profile(i0, i1)

    def _on_main_range_changed(self, _vb, ranges) -> None:
        """Sync tick profile Y axis with the main candle chart on every zoom/pan."""
        ylo, yhi = ranges[1]
        self._tick_profile_widget.setYRange(ylo, yhi, padding=0)

    def _pin_heatmap_legend(self, *_) -> None:
        """Re-anchor the heatmap legend to the top-left of the candle view."""
        if not self._heatmap_legend.isVisible():
            return
        xlo, xhi = self._plot_c.vb.viewRange()[0]
        ylo, yhi = self._plot_c.vb.viewRange()[1]
        x_pad    = (xhi - xlo) * 0.01
        y_pad    = (yhi - ylo) * 0.015
        self._heatmap_legend.setPos(xlo + x_pad, yhi - y_pad)

    def _pin_ob_legend(self, *_) -> None:
        """Re-anchor the OB legend just below the heatmap legend."""
        if not self._ob_legend.isVisible():
            return
        xlo, xhi = self._plot_c.vb.viewRange()[0]
        ylo, yhi = self._plot_c.vb.viewRange()[1]
        x_pad    = (xhi - xlo) * 0.01
        # Offset one row below the heatmap legend slot (3.5% of y range)
        y_top    = yhi - (yhi - ylo) * 0.015
        y_row    = (yhi - ylo) * 0.035
        self._ob_legend.setPos(xlo + x_pad, y_top - y_row)

    def _pin_chandelier_label(self, *_) -> None:
        """Re-anchor the chandelier suggestion label to the bottom-right of the candle view."""
        if not self._chandelier_label.isVisible():
            return
        xlo, xhi = self._plot_c.vb.viewRange()[0]
        ylo, yhi = self._plot_c.vb.viewRange()[1]
        x_pad    = (xhi - xlo) * 0.01
        y_pad    = (yhi - ylo) * 0.015
        self._chandelier_label.setPos(xhi - x_pad, ylo + y_pad)

    def _on_chandelier_settings(self) -> None:
        dlg = ChandelierParamsDialog(
            self._chandelier_period, self._chandelier_multiplier,
            self._chandelier_direction, self,
        )
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._chandelier_period, self._chandelier_multiplier, self._chandelier_direction = dlg.values()
            if self._ind("chandelier"):
                self._update_chandelier_label()

    def _update_chandelier_label(self) -> None:
        """Recompute the bottom-right chandelier-exit suggestion from self._klines.

        Point-in-time only (latest bar), same convention as
        strategy/chandelier_exit/calculator.py -- not a ratcheted stop tracked
        since a real entry date.
        """
        period = self._chandelier_period
        if self._klines is None or self._klines.empty or len(self._klines) < period:
            self._chandelier_label.setHtml(
                f"<span style='color:{_GREY}'>Chandelier: need &ge;{period} bars</span>")
            self._chandelier_label.setVisible(True)
            self._pin_chandelier_label()
            return

        highs  = self._klines["high"].to_numpy(dtype=float)
        lows   = self._klines["low"].to_numpy(dtype=float)
        closes = self._klines["close"].to_numpy(dtype=float)
        mult   = self._chandelier_multiplier
        bull   = self._chandelier_direction == "bull"
        r      = current_stop(highs, lows, closes, period, mult, self._chandelier_direction)

        if r is None:
            self._chandelier_label.setHtml(
                f"<span style='color:{_GREY}'>Chandelier: insufficient warmup</span>")
            self._chandelier_label.setVisible(True)
            self._pin_chandelier_label()
            return

        red_up    = self._ind("red_up")
        # direction color follows the same buy/sell convention as tick-profile bars
        col       = (_RED if red_up else _GREEN) if bull else (_GREEN if red_up else _RED)
        dir_label = "Long" if bull else "Short"

        offset     = r["atr"] * mult
        offset_pct = offset / r["price"] * 100 if r["price"] else 0.0
        html = (
            f"<span style='color:{_FG}'>Chandelier {dir_label}  (p{period} x{mult:g})</span><br>"
            f"<span style='color:{col}'>Stop: {r['stop']:.4f}</span><br>"
            f"<span style='color:{_GREY}'>Dist: {r['dist']:.4f}  ({r['dist_pct']:.2f}%)</span><br>"
            f"<span style='color:{_GREY}'>ATR&times;mult: {offset:.4f}  ({offset_pct:.2f}%)</span>"
        )
        self._chandelier_label.setHtml(html)
        self._chandelier_label.setVisible(True)
        self._pin_chandelier_label()

    # ── Crosshair + tooltip ───────────────────────────────────────────────────

    def _on_mouse_move(self, pos) -> None:
        # pos is QPointF emitted directly by scene.sigMouseMoved
        in_candle = self._plot_c.sceneBoundingRect().contains(pos)
        in_vol    = (self._plot_v.isVisible()
                     and self._plot_v.sceneBoundingRect().contains(pos))
        in_kd     = (self._plot_kd.isVisible()
                     and self._plot_kd.sceneBoundingRect().contains(pos))
        in_cvd    = (self._plot_cvd.isVisible()
                     and self._plot_cvd.sceneBoundingRect().contains(pos))
        in_adi    = (self._plot_adi.isVisible()
                     and self._plot_adi.sceneBoundingRect().contains(pos))
        in_any    = in_candle or in_vol or in_kd or in_cvd or in_adi

        if not in_any:
            for line in (self._vline, self._hline,
                         self._vline_v, self._vline_kd, self._vline_cvd,
                         self._vline_adi,
                         self._price_label, self._ohlcv_label,
                         self._vol_label, self._kd_label, self._cvd_label,
                         self._adi_label,
                         self._profile_hline, self._tick_profile_hline):
                line.setVisible(False)
            return

        # Map scene position to candle-plot data coordinates regardless of which
        # sub-plot the cursor is in (all share the same X axis via setXLink).
        if in_candle:
            mouse_pt = self._plot_c.vb.mapSceneToView(pos)
        elif in_vol:
            mouse_pt = self._plot_v.vb.mapSceneToView(pos)
        elif in_kd:
            mouse_pt = self._plot_kd.vb.mapSceneToView(pos)
        elif in_cvd:
            mouse_pt = self._plot_cvd.vb.mapSceneToView(pos)
        else:
            mouse_pt = self._plot_adi.vb.mapSceneToView(pos)

        x = mouse_pt.x()
        # Y price is only meaningful from the candle plot
        if in_candle:
            y = mouse_pt.y()
        else:
            # Convert cursor scene-Y to candle-plot data-Y for the price label
            candle_pt = self._plot_c.vb.mapSceneToView(pos)
            y = candle_pt.y()

        # Update all vertical lines (shared X axis)
        self._vline.setPos(x);    self._vline.setVisible(True)
        if self._plot_v.isVisible():
            self._vline_v.setPos(x);  self._vline_v.setVisible(True)
        if self._plot_kd.isVisible():
            self._vline_kd.setPos(x); self._vline_kd.setVisible(True)
        if self._plot_cvd.isVisible():
            self._vline_cvd.setPos(x); self._vline_cvd.setVisible(True)
        if self._plot_adi.isVisible():
            self._vline_adi.setPos(x); self._vline_adi.setVisible(True)

        # Horizontal line only in candle plot
        self._hline.setPos(y);  self._hline.setVisible(in_candle)

        # Profile panel sync lines (session profile + tick profile)
        self._profile_hline.setPos(y)
        self._profile_hline.setVisible(True)
        self._tick_profile_hline.setPos(y)
        self._tick_profile_hline.setVisible(True)

        xlo, xhi = self._plot_c.vb.viewRange()[0]
        ylo, yhi = self._plot_c.vb.viewRange()[1]
        y_span   = yhi - ylo
        label_x  = xlo + (xhi - xlo) * 0.01  # ~1% from left edge

        # Price tag: left edge, vertically centered on cursor price
        self._price_label.setPos(label_x, y)
        self._price_label.setText(f"{y:.2f}")
        self._price_label.setVisible(True)

        if self._klines is not None and not self._klines.empty:
            idx = int(round(x))
            idx = max(0, min(idx, len(self._klines) - 1))
            row = self._klines.iloc[idx]
            vol = int(row.get("volume", 0) or 0)

            # OHLCV tooltip: anchor bottom-left → body floats ABOVE position.
            # Clamp so the anchor is at most 2% below the view top (avoids going
            # off-screen when cursor is near the very top of the chart).
            tip_y = min(y + y_span * 0.14, yhi - y_span * 0.02)
            self._ohlcv_label.setPos(label_x, tip_y)
            self._ohlcv_label.setText(
                f"{str(row['time_key'])[:16]}\n"
                f"O {row['open']:.2f}  H {row['high']:.2f}\n"
                f"L {row['low']:.2f}  C {row['close']:.2f}\n"
                f"Vol {vol:,}"
            )
            self._ohlcv_label.setVisible(True)

            if idx != self._last_hover_idx:
                self._show_tick_profile(idx)

            # DOM / Liq HM crosshair sync (historical mode only)
            if self._dom_window is not None or self._liq_hm_window is not None:
                if self._mode_combo.currentText() == "Historical":
                    try:
                        bar_ts = datetime.strptime(str(row["time_key"])[:16], "%Y-%m-%d %H:%M")
                        self._dom_sync(bar_ts)
                        if self._liq_hm_window is not None:
                            self._liq_hm_window.pin_timestamp(bar_ts)
                    except Exception:
                        pass

            # ── Subplot readout labels ────────────────────────────────────────
            # MAVOL: show volume of the hovered bar as a floating label on the
            # left edge of the vol subplot, vertically at the bar's volume level.
            if self._plot_v.isVisible():
                vol_str = (f"{vol/1e6:.2f}M" if vol >= 1_000_000 else
                           f"{vol/1e3:.0f}K"  if vol >= 1_000     else str(vol))
                xlo_v, xhi_v = self._plot_v.vb.viewRange()[0]
                lx_v = xlo_v + (xhi_v - xlo_v) * 0.01
                self._vol_label.setPos(lx_v, float(vol))
                self._vol_label.setText(vol_str)
                self._vol_label.setVisible(True)
            else:
                self._vol_label.setVisible(False)

            # KDV: show spread-width value of the hovered bar.
            if self._plot_kd.isVisible() and self._kd_width_arr is not None:
                # _kd_width_arr covers the last nw bars of klines; subtract offset.
                kd_offset = (len(self._klines) - len(self._kd_width_arr)
                             if self._klines is not None else 0)
                kd_idx = max(0, min(idx - kd_offset, len(self._kd_width_arr) - 1))
                kd_val = float(self._kd_width_arr[kd_idx])
                kd_str = f"{kd_val:+.4f}"
                xlo_kd, xhi_kd = self._plot_kd.vb.viewRange()[0]
                lx_kd = xlo_kd + (xhi_kd - xlo_kd) * 0.01
                self._kd_label.setPos(lx_kd, kd_val)
                self._kd_label.setText(kd_str)
                self._kd_label.setVisible(True)
            else:
                self._kd_label.setVisible(False)

            # CVD: show cumulative delta value of the hovered bar.
            if self._plot_cvd.isVisible() and self._cvd_arr is not None:
                cvd_idx = max(0, min(idx, len(self._cvd_arr) - 1))
                cvd_val = float(self._cvd_arr[cvd_idx])
                sign    = "+" if cvd_val >= 0 else ""
                if abs(cvd_val) >= 1_000_000:
                    cvd_str = f"{sign}{cvd_val/1_000_000:.2f}M"
                elif abs(cvd_val) >= 1_000:
                    cvd_str = f"{sign}{cvd_val/1_000:.0f}K"
                else:
                    cvd_str = f"{sign}{cvd_val:.0f}"
                xlo_cvd, xhi_cvd = self._plot_cvd.vb.viewRange()[0]
                lx_cvd = xlo_cvd + (xhi_cvd - xlo_cvd) * 0.01
                self._cvd_label.setPos(lx_cvd, cvd_val)
                self._cvd_label.setText(cvd_str)
                self._cvd_label.setVisible(True)
            else:
                self._cvd_label.setVisible(False)

            # A/D: show accumulation/distribution line value of the hovered bar.
            if self._plot_adi.isVisible() and self._adi_arr is not None:
                adi_idx = max(0, min(idx, len(self._adi_arr) - 1))
                adi_val = float(self._adi_arr[adi_idx])
                sign    = "+" if adi_val >= 0 else ""
                if abs(adi_val) >= 1_000_000:
                    adi_str = f"{sign}{adi_val/1_000_000:.2f}M"
                elif abs(adi_val) >= 1_000:
                    adi_str = f"{sign}{adi_val/1_000:.0f}K"
                else:
                    adi_str = f"{sign}{adi_val:.0f}"
                xlo_adi, xhi_adi = self._plot_adi.vb.viewRange()[0]
                lx_adi = xlo_adi + (xhi_adi - xlo_adi) * 0.01
                self._adi_label.setPos(lx_adi, adi_val)
                self._adi_label.setText(adi_str)
                self._adi_label.setVisible(True)
            else:
                self._adi_label.setVisible(False)

    # ── X-axis tick labels ────────────────────────────────────────────────────

    def _on_home(self) -> None:
        """Reset zoom/pan to the initial view (last 150 bars)."""
        if self._klines is not None and not self._klines.empty:
            self._reset_view(len(self._klines))

    def _on_pin_toggled(self, checked: bool) -> None:
        flags = self.windowFlags()
        if checked:
            self.setWindowFlags(flags | Qt.WindowType.WindowStaysOnTopHint)
        else:
            self.setWindowFlags(flags & ~Qt.WindowType.WindowStaysOnTopHint)
        self.show()

    def _reset_view(self, n_bars: int, init_bars: int = 150) -> None:
        """Set the initial view to the last init_bars candles.

        Both X and Y are set explicitly — no autoRange() calls — so the
        view is always correct on first load without being overridden by
        the linked volume plot or a global bounding-rect auto-range.
        Live refreshes skip this method entirely so user zoom is preserved.
        """
        x_end   = n_bars - 1 + 3
        x_start = max(-1, n_bars - init_bars)

        # Constrain scrolling/zoom so the user cannot wander off-data into
        # empty space — prevents the "all candles squeezed into a sliver" issue
        # that occurs when PyQtGraph's linked-axis zoom is overdriven.
        self._plot_c.vb.setLimits(
            xMin=-5, xMax=n_bars + 5,
            minXRange=5, maxXRange=n_bars + 10,
        )

        self._plot_c.setXRange(x_start, x_end, padding=0)
        # Derive Y range (price + volume) from the actually visible bars
        vis_start = max(0, n_bars - init_bars)
        self._autorange_price_subplots(vis_start, n_bars)

        # CVD / A-D Line: both are unbounded running cumulative sums, so a
        # plain bounding-rect autorange fits the *entire* loaded history --
        # since the visible init_bars window is a small slice of that, its
        # real variation gets dwarfed and the curve looks like a flat line.
        # Derive Y range from the visible bars only, same fix as Volume above.
        self._autorange_cumulative_subplots(vis_start, n_bars)

    def _autorange_price_subplots(self, x_lo: float, x_hi: float) -> None:
        """Fit main-chart price Y-range (and Volume Y-range) to the given
        visible X window.

        _reset_view() calls this once at load/TF-change time, but Live mode
        keeps streaming new bars via _on_data_ready() without ever re-running
        _reset_view() -- without a second call from there too, this Y-range
        goes stale and price can drift entirely above/below the frozen
        window as it moves away from the level it was at on load, making the
        candles appear to walk off the top/bottom of the chart.
        """
        if self._klines is None or self._klines.empty:
            return
        i0 = max(0, int(x_lo))
        i1 = min(int(x_hi) + 1, len(self._klines))
        vis = self._klines.iloc[i0:i1]
        if vis.empty:
            return
        y_lo   = float(vis["low"].min())
        y_hi   = float(vis["high"].max())
        margin = (y_hi - y_lo) * 0.08 or max(abs(y_hi), 1.0) * 0.08
        self._plot_c.setYRange(y_lo - margin, y_hi + margin, padding=0)
        max_v = float(vis["volume"].max()) if not vis["volume"].isna().all() else 1.0
        self._plot_v.setYRange(0, max_v * 1.15, padding=0)

    def _autorange_cumulative_subplots(self, x_lo: float, x_hi: float) -> None:
        """Fit CVD / A-D Line Y-range to the given visible X window.

        Both are unbounded running cumulative sums, so PyQtGraph's default
        bounding-rect autorange (fit to the *entire* loaded series) dwarfs
        the visible window's real variation. _reset_view() calls this once
        at load/TF-change time, but Live mode keeps streaming new bars via
        _on_data_ready() without ever re-running _reset_view() -- without a
        second call from there too, this Y-range goes stale and the curve
        can drift entirely outside the frozen window (e.g. after a
        trading-day CVD reset), making the panel look empty.
        """
        i0 = max(0, int(x_lo))
        for arr, plot in ((self._cvd_arr, self._plot_cvd),
                          (self._adi_arr, self._plot_adi)):
            if arr is None or len(arr) == 0:
                continue
            i1 = min(int(x_hi) + 1, len(arr))
            vis = arr[i0:i1]
            if len(vis) == 0:
                continue
            lo, hi = float(vis.min()), float(vis.max())
            pad = (hi - lo) * 0.1 or max(abs(hi), 1.0) * 0.1
            plot.setYRange(lo - pad, hi + pad, padding=0)

    def _set_xaxis_ticks(self, klines: pd.DataFrame) -> None:
        """Map integer bar indices to time_key strings on x-axis.

        Only the main candle plot shows text labels; subplots suppress labels
        to prevent crowding — they are X-linked so positions already align.
        """
        n    = len(klines)
        step = max(1, n // 10)
        tf   = self._tf_combo.currentText()
        # Daily: show "YYYY-MM-DD"; intraday: show "MM-DD HH:MM"
        if tf == "1d":
            label_slice = slice(0, 10)
        else:
            label_slice = slice(5, 16)
        ticks = [
            (i, str(klines.iloc[i]["time_key"])[label_slice])
            for i in range(0, n, step)
        ]
        self._plot_c.getAxis("bottom").setTicks([ticks])
        # Subplots: pass tick positions so grid lines align, but hide text.
        pos_only = [(i, "") for i, _ in ticks]
        if self._ind("vol"):
            self._plot_v.getAxis("bottom").setTicks([pos_only])
        if self._ind("kd"):
            self._plot_kd.getAxis("bottom").setTicks([pos_only])
        if self._ind("cvd"):
            self._plot_cvd.getAxis("bottom").setTicks([pos_only])
        if self._ind("adi"):
            self._plot_adi.getAxis("bottom").setTicks([pos_only])

    # ── Event filter (Enter key in toolbar QLineEdit) ─────────────────────────

    def eventFilter(self, obj: object, event) -> bool:
        """Intercept Enter key on code_edit before QToolBar can consume it.

        On Windows, QToolBar sometimes swallows the Enter key for its own
        navigation, preventing QLineEdit.returnPressed from firing.  Catching
        the raw KeyPress here and forwarding to _trigger_fetch is reliable.
        """
        from PyQt6.QtCore import QEvent
        if (obj is self._code_edit
                and event.type() == QEvent.Type.KeyPress
                and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)):
            self._trigger_fetch()
            return True   # consume — prevents returnPressed double-fire
        return super().eventFilter(obj, event)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        """Shut down timers, moomoo connection, and background threads."""
        # Close child floating windows first so their timers/workers stop before
        # the moomoo context is torn down.  Without this the app event loop keeps
        # running (quitOnLastWindowClosed won't fire while they're still open).
        if self._liq_hm_window is not None:
            _close_with_hang_watchdog(self._liq_hm_window, "LiqHmWindow")
            self._liq_hm_window = None
        if self._dom_window is not None:
            self._dom_window.close()
            self._dom_window = None

        # Stop the live refresh timer and unsubscribe tickers.
        self._stop_live()

        # Stop the option-chain poll timer and its context/fetcher, mirroring
        # the main ctx/fetcher teardown below (own connection, own thread).
        self._option_timer.stop()
        if self._option_ctx is not None:
            try:
                self._option_ctx.close()
            except Exception:
                pass
            self._option_ctx = None
        if self._option_fetcher and self._option_fetcher.isRunning():
            self._option_fetcher.quit()
            if not self._option_fetcher.wait(2000):
                self._option_fetcher.terminate()
                self._option_fetcher.wait(500)

        # Close the moomoo context BEFORE asking the fetcher thread to stop.
        # Closing the socket unblocks any in-flight request_history_kline call
        # inside DataFetcher.run(), allowing the thread to exit on its own.
        with self._ctx_lock:
            if self._ctx:
                try:
                    self._ctx.close()
                except Exception:
                    pass
                self._ctx = None

        # Ask the fetcher thread to exit; give it 2 s, then force-terminate.
        # DataFetcher.run() has no event loop, so quit() alone is not enough —
        # we need terminate() as a fallback if the thread is still blocked.
        if self._fetcher and self._fetcher.isRunning():
            self._fetcher.quit()
            if not self._fetcher.wait(2000):
                self._fetcher.terminate()
                self._fetcher.wait(500)

        event.accept()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_float(v) -> float:
    """Return float(v) or 0.0 on error."""
    try:
        return float(v) if v is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


# ── Entry point ───────────────────────────────────────────────────────────────

def _default_code() -> str:
    cfg = pathlib.Path(__file__).parent.parent / "config" / "schedule.json"
    try:
        import json
        data = json.loads(cfg.read_text(encoding="utf-8"))
        return data.get("default_code") or (data.get("targets") or ["US.SOXL"])[0]
    except Exception:
        return "US.SOXL"


def _parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Trade Viewer Qt (PyQtGraph)")
    ap.add_argument("--code",    default=_default_code())
    ap.add_argument("--tf",      default="5m",
                    choices=list(TIMEFRAME_MAP.keys()))
    ap.add_argument("--mode",    default="Live",
                    choices=["Live", "Historical"])
    ap.add_argument("--date",    default=None)
    ap.add_argument("--host",    default="127.0.0.1")
    ap.add_argument("--port",    type=int, default=11111)
    ap.add_argument("--refresh", type=int, default=15)
    return ap.parse_args(argv)


def main(argv=None) -> None:
    """Entry point — accepts optional argv list for dispatch from main.py."""
    args = _parse_args(argv)
    app  = QApplication(sys.argv)
    app.setApplicationName("Trade Viewer Qt")
    win  = TradeViewerQt(args)
    win.show()
    ret = app.exec()
    # os._exit() bypasses Python's thread-join shutdown so the process exits
    # immediately even if the moomoo SDK left non-daemon threads running.
    os._exit(ret)


if __name__ == "__main__":
    main()
