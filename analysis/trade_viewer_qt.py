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
  - EMA overlays (20 / 50 / 200) on candle chart
  - Session Vol Profile panel (right)
  - Single-candle Tick Profile panel (centre-right, on hover)
  - Crosshair + OHLCV tooltip (left-side, below price label)
  - Indicator toggle toolbar
  - Session filter checkboxes (Pre / Regular / Post / Night)
  - Profile date-range selector (1D / 3D / 1W, trading-day aware)
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
import json
import pathlib
import sqlite3
import sys
import threading
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtCore import (
    Qt, QThread, QTimer, pyqtSignal, QRectF, QPointF,
)
from PyQt6.QtGui import (
    QColor, QPainter, QPicture, QPen, QBrush, QFont,
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QToolBar, QLabel, QComboBox, QLineEdit, QSpinBox,
    QPushButton, QCheckBox, QButtonGroup, QRadioButton, QSizePolicy,
    QFrame, QStatusBar, QMessageBox,
)

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from core.time_utils import candle_start
from feeds.fetcher import fetch_klines
from strategy.smc.market_structure import detect_bos_choch
from strategy.smc.fvg import detect_fvg
from strategy.smc.order_blocks import detect_order_blocks
from strategy.smc.kd_trend import compute_kd
from moomoo import (
    OpenQuoteContext, SubType, KLType, AuType,
    TickerHandlerBase, RET_OK,
)

# ── PyQtGraph global config ───────────────────────────────────────────────────

pg.setConfigOptions(antialias=True, useOpenGL=True, enableExperimental=True)

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

def _qc(hex_str: str, alpha: int = 255) -> QColor:
    c = QColor(hex_str)
    c.setAlpha(alpha)
    return c

# ── Timeframe / BOS config (mirrors trade_viewer.py) ─────────────────────────

TIMEFRAME_MAP: dict[str, tuple[KLType, int]] = {
    "1m":  (KLType.K_1M,    1),
    "5m":  (KLType.K_5M,    5),
    "15m": (KLType.K_15M,  15),
    "30m": (KLType.K_30M,  30),
    "1h":  (KLType.K_60M,  60),
    "4h":  (KLType.K_240M, 240),
}

_DAY_CANDLES: dict[str, int] = {
    "1m": 390, "5m": 78, "15m": 26, "30m": 14, "1h": 7, "4h": 6,
}

_BOS_MAX_SPAN: dict[str, int] = {
    "1m": 60, "5m": 12, "15m": 26, "30m": 13, "1h": 7, "4h": 8,
}

_TREND_WINDOW: dict[str, int] = {
    "1m": 60, "5m": 12, "15m": 26, "30m": 13, "1h": 7, "4h": 8,
}

_LIVE_LOOKBACK_DAYS: dict[str, int] = {
    "1m": 2, "5m": 5, "15m": 7, "30m": 10, "1h": 14, "4h": 30,
}

# EMA periods shown on the candle chart
_EMA_PERIODS = [20, 50, 200]

# KD default parameters (match backtest defaults)
_KD_FAST = 25
_KD_SLOW = 90

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
    """Load tick buckets from ticks.db for a given code and date."""
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
        buckets: dict = defaultdict(
            lambda: defaultdict(lambda: {"buy": 0, "sell": 0, "neutral": 0})
        )
        for r in rows:
            ts = (r["ts"] if isinstance(r["ts"], datetime)
                  else datetime.fromisoformat(str(r["ts"])))
            bucket = candle_start(ts, candle_mins)
            key = {"BUY": "buy", "SELL": "sell"}.get(
                r["direction"].upper(), "neutral")
            buckets[bucket][r["price"]][key] += r["volume"]
        return dict(buckets)
    except Exception:
        return None


def apply_profile_range(klines: pd.DataFrame, range_val: str) -> pd.DataFrame:
    """Trim klines to N trading days ending at the last bar."""
    if klines.empty:
        return klines
    n_days = {"1d": 1, "3d": 3, "7d": 5}.get(range_val)
    if n_days is None:
        return klines
    times = klines["time_key"].astype(str).str[:10]
    anchor = times.iloc[-1]
    dates  = sorted(times[times <= anchor].unique())
    start  = dates[-n_days] if len(dates) >= n_days else dates[0]
    return klines[(times >= start) & (times <= anchor)]


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
            base_col = _qc(_bull_col if is_bull else _bear_col)
            p.setBrush(QBrush(base_col))
            p.drawRect(QRectF(i - 0.35, body_lo, 0.7, body_hi - body_lo))
            if self._show_heatmap and pd_:
                self._draw_heatmap_overlay(p, i, l, h, body_lo, body_hi, pd_)

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
        """Overlay semi-transparent buy/sell heatmap bins on the candle body.

        Drawn AFTER the solid base body so the candle is always readable.
        Bins outside the body (wick area) are drawn at reduced opacity.
        Zero-volume bins are skipped entirely.
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
            if totals[b] == 0:
                continue   # skip empty bins entirely

            bin_lo = low  + b       * bin_h
            bin_hi = bin_lo + bin_h

            # Determine draw bounds: body region vs wick region
            in_body = not (bin_hi <= body_lo or bin_lo >= body_hi)
            if in_body:
                draw_lo = max(bin_lo, body_lo)
                draw_hi = min(bin_hi, body_hi)
                wick_mult = 1.0
            else:
                draw_lo, draw_hi = bin_lo, bin_hi
                wick_mult = 0.35   # fainter outside the body

            total = buys[b] + sells[b]
            vol_frac = min(totals[b] / max_total, 1.0)

            if total > 0:
                ratio = (buys[b] - sells[b]) / total  # -1..+1
                # alpha: floor 60, ceiling 200, scaled by volume intensity and delta magnitude
                delta_frac = 0.5 + 0.5 * abs(ratio)   # 0.5 (neutral) .. 1.0 (pure)
                alpha = int(min(200, max(60, 160 * vol_frac * delta_frac)) * wick_mult)
                col = QColor(_UP if ratio >= 0 else _DOWN)
            else:
                alpha = int(50 * wick_mult)
                col   = QColor(_GREY)

            col.setAlpha(alpha)
            p.setBrush(QBrush(col))
            if draw_hi > draw_lo:
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
        self._picture: QPicture | None = None
        self._rect     = QRectF()

    def set_data(self, fvg_gaps: list[dict], n_bars: int) -> None:
        self._gaps    = fvg_gaps
        self._n       = n_bars
        self._picture = None
        self.prepareGeometryChange()
        self.update()

    def _build(self) -> None:
        pic = QPicture()
        p   = QPainter(pic)
        bull_fill = _qc(_GREEN, 35)
        bear_fill = _qc(_RED,   35)
        bull_pen  = QPen(_qc(_GREEN, 120), 0)
        bear_pen  = QPen(_qc(_RED,   120), 0)

        for g in self._gaps:
            x0   = float(g["idx"])
            top  = float(g["top"])
            bot  = float(g["bottom"])
            bull = g.get("direction", "bull") == "bull"

            p.setPen(bull_pen  if bull else bear_pen)
            p.setBrush(QBrush(bull_fill if bull else bear_fill))
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

            # Subtype label at left edge of zone
            label_map = {"regular": "OB", "mitigation": "OB~", "breaker": "BRK"}
            lbl       = label_map.get(subtype, "OB")
            p.setPen(QPen(pen_color, 0))
            p.setFont(QFont("Monospace", 6))
            p.drawText(QPointF(x0, top), lbl)

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
                end_dt   = dt + timedelta(days=3)
                start    = (dt - timedelta(days=8)).strftime("%Y-%m-%d 20:00:00")
                end      = f"{end_dt.strftime('%Y-%m-%d')} 23:59:59"
            else:
                end   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                lb    = _LIVE_LOOKBACK_DAYS.get(tf, 10)
                start = (datetime.now() - timedelta(days=lb)).strftime(
                    "%Y-%m-%d %H:%M:%S")

            ktype, _ = TIMEFRAME_MAP[tf]
            ret, df, _ = self._ctx.request_history_kline(
                code, start=start, end=end, ktype=ktype,
                autype=AuType.NONE, max_count=2000, extended_time=True,
            )
            if ret != RET_OK or df is None or df.empty:
                self.error.emit(f"Kline fetch failed: ret={ret}")
                return

            # SMC detection on last N warmup bars
            warmup_n = min(len(df), 400)
            warmup   = df.iloc[-warmup_n:].reset_index(drop=True)

            smc_signals: list[dict] = []
            if ind.get("bos_choch") or ind.get("ob"):
                smc_signals = detect_bos_choch(
                    warmup,
                    max_span_bars=_BOS_MAX_SPAN.get(tf),
                    trend_window=_TREND_WINDOW.get(tf, 20),
                    filter_choch=False,          # viewer shows all CHoCH, no displacement filter
                    max_session_gap=_BOS_SESSION_GAP.get(tf),  # respect chart.json session gap
                )

            fvg_gaps: list[dict] = []
            if ind.get("fvg"):
                raw_fvgs = detect_fvg(warmup)
                disp_off = warmup_n - min(warmup_n, len(df))
                for g in raw_fvgs:
                    r = dict(g)
                    r["idx"] = max(0, g["idx"] - disp_off)
                    fvg_gaps.append(r)

            ob_blocks: list[dict] = []
            if ind.get("ob") and smc_signals:
                ob_blocks = detect_order_blocks(warmup, smc_signals)

            # Tick data
            ticks: dict | None = None
            if historical:
                ticks = load_local_ticks(code, date_str, tf)
            else:
                ticks = p.get("live_ticks") or {}

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
        self._tick_lock     = threading.Lock()
        self._fetcher:      DataFetcher | None  = None
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._trigger_fetch)
        self._candle_mins   = 1
        self._smc_signals:  list[dict]          = []
        self._fvg_gaps:     list[dict]          = []
        self._ob_blocks:    list[dict]          = []
        self._trade_record: dict | None         = None  # active trade review
        # Track which (code, tf) was last auto-ranged; prevents live-refresh
        # from resetting the user's manual pan/zoom on every tick.
        self._last_chart_key: tuple             = ("", "")

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

    # ── Toolbars (two rows) ───────────────────────────────────────────────────

    def _build_toolbar(self, args) -> None:
        def _lbl(text: str) -> QLabel:
            l = QLabel(text)
            l.setStyleSheet(f"color: {_FG}; font-size: 11px; padding: 0 2px;")
            return l

        # ── Row 1: core data controls ─────────────────────────────────────────
        tb1 = QToolBar("Controls", self)
        tb1.setMovable(False)
        tb1.setFloatable(False)
        self.addToolBar(tb1)

        # Code
        tb1.addWidget(_lbl("Code:"))
        self._code_edit = QLineEdit(getattr(args, "code", "US.SNDK") or "US.SNDK")
        self._code_edit.setFixedWidth(90)
        self._code_edit.returnPressed.connect(self._trigger_fetch)
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

        # ── Row 2: indicators / session / range / trade review ────────────────
        self.addToolBarBreak()
        tb2 = QToolBar("Indicators", self)
        tb2.setMovable(False)
        tb2.setFloatable(False)
        self.addToolBar(tb2)

        # Indicators
        tb2.addWidget(_lbl("Indicators:"))
        self._ind_checks: dict[str, QCheckBox | QRadioButton] = {}
        for key, label in [
            ("heatmap",   "Heatmap"),
            ("delta",     "Δ Delta"),
            ("bos_choch", "BOS/CHoCH"),
            ("fvg",       "FVG"),
            ("ob",        "OB"),
            ("kd",        "KD"),
            ("ema",       "EMA"),
        ]:
            cb = QCheckBox(label)
            cb.setChecked(key in ("heatmap", "delta", "bos_choch"))
            cb.stateChanged.connect(self._on_indicator_toggle)
            self._ind_checks[key] = cb
            tb2.addWidget(cb)

        tb2.addSeparator()

        # Session filters
        tb2.addWidget(_lbl("Session:"))
        for key, label in [
            ("regular", "Regular"), ("pre", "Pre"),
            ("post", "Post"),       ("night", "Night"),
        ]:
            cb = QCheckBox(label)
            cb.setChecked(key == "regular")
            cb.stateChanged.connect(self._on_session_toggle)
            self._ind_checks[f"sess_{key}"] = cb
            tb2.addWidget(cb)

        tb2.addSeparator()

        # Profile range
        tb2.addWidget(_lbl("Range:"))
        self._range_group = QButtonGroup(self)
        for val, label in [("1d", "1D"), ("3d", "3D"), ("7d", "1W")]:
            rb = QRadioButton(label)
            rb.setChecked(val == "1d")
            rb.toggled.connect(self._on_range_changed)
            self._ind_checks[f"range_{val}"] = rb
            self._range_group.addButton(rb)
            tb2.addWidget(rb)

        tb2.addSeparator()

        # Color scheme toggle
        cb_red_up = QCheckBox("Red Up")
        cb_red_up.setChecked(False)   # default: Western green-up / red-down
        cb_red_up.setToolTip("Checked = red rises, green falls (CN convention)\nUnchecked = green rises, red falls (Western convention)")
        cb_red_up.stateChanged.connect(self._on_indicator_toggle)
        self._ind_checks["red_up"] = cb_red_up
        tb2.addWidget(cb_red_up)

        tb2.addSeparator()

        # Trade Review input
        tb2.addWidget(_lbl("Trade ID:"))
        self._trade_id_edit = QLineEdit()
        self._trade_id_edit.setPlaceholderText("trade UUID…")
        self._trade_id_edit.setFixedWidth(200)
        self._trade_id_edit.returnPressed.connect(self._load_trade_review)
        tb2.addWidget(self._trade_id_edit)
        review_btn = QPushButton("Review")
        review_btn.clicked.connect(self._load_trade_review)
        tb2.addWidget(review_btn)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._clear_trade_review)
        tb2.addWidget(clear_btn)

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
        self._plot_v.setLabel("left", "Vol", **{"color": _FG})
        self._plot_v.getAxis("left").setTextPen(_qc(_FG))
        self._plot_v.getAxis("bottom").setTextPen(_qc(_FG))
        self._plot_v.setMenuEnabled(False)
        self._plot_v.setXLink(self._plot_c)

        # KD subplot (row 2) — hidden until KD indicator enabled
        self._chart_widget.nextRow()
        self._plot_kd: pg.PlotItem = self._chart_widget.addPlot(row=2, col=0)
        self._plot_kd.showGrid(x=True, y=True, alpha=0.10)
        self._plot_kd.setLabel("left", "KD", **{"color": _FG})
        self._plot_kd.getAxis("left").setTextPen(_qc(_FG))
        self._plot_kd.getAxis("bottom").setTextPen(_qc(_FG))
        self._plot_kd.setMenuEnabled(False)
        self._plot_kd.setXLink(self._plot_c)
        # Add zero line reference
        self._plot_kd.addItem(pg.InfiniteLine(
            pos=0, angle=0, movable=False,
            pen=pg.mkPen(_GREY, width=1, style=Qt.PenStyle.DashLine),
        ))
        self._plot_kd.hide()  # shown when KD checkbox enabled

        # Row stretch factors
        self._chart_widget.ci.layout.setRowStretchFactor(0, 5)
        self._chart_widget.ci.layout.setRowStretchFactor(1, 1)
        self._chart_widget.ci.layout.setRowStretchFactor(2, 1)

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
        self._kd_items:      list = []  # PlotCurveItem + fill for KD subplot
        self._trade_items:   list = []  # all trade review overlay items

        # Volume bars
        self._vol_item = pg.BarGraphItem(
            x=[], height=[], width=0.7,
            brush=_qc(_GREEN, 100), pen=pg.mkPen(None),
        )
        self._plot_v.addItem(self._vol_item)

        # Crosshair — one set per subplot so lines extend into every panel
        cross_pen = pg.mkPen(_CROSS, width=1, style=Qt.PenStyle.DashLine)

        # Main candle plot: vertical + horizontal lines
        self._vline    = pg.InfiniteLine(angle=90, movable=False, pen=cross_pen)
        self._hline    = pg.InfiniteLine(angle=0,  movable=False, pen=cross_pen)
        self._vline.setVisible(False)
        self._hline.setVisible(False)
        self._plot_c.addItem(self._vline, ignoreBounds=True)
        self._plot_c.addItem(self._hline, ignoreBounds=True)

        # Volume subplot: vertical line only (no meaningful horizontal)
        self._vline_v  = pg.InfiniteLine(angle=90, movable=False, pen=cross_pen)
        self._vline_v.setVisible(False)
        self._plot_v.addItem(self._vline_v, ignoreBounds=True)

        # KD subplot: vertical line only
        self._vline_kd = pg.InfiniteLine(angle=90, movable=False, pen=cross_pen)
        self._vline_kd.setVisible(False)
        self._plot_kd.addItem(self._vline_kd, ignoreBounds=True)

        # Profile panel: horizontal line that follows main chart price (Y)
        self._profile_hline = pg.InfiniteLine(
            angle=0, movable=False,
            pen=pg.mkPen(_CROSS, width=1, style=Qt.PenStyle.DashLine),
        )
        self._profile_hline.setVisible(False)

        # Price label (follows crosshair Y, anchored at left edge — text grows right)
        self._price_label = pg.TextItem(
            text="", color=_GOLD, anchor=(0.0, 0.5),
        )
        self._price_label.setFont(QFont("Monospace", 7))
        self._price_label.setVisible(False)
        self._plot_c.addItem(self._price_label, ignoreBounds=True)

        # OHLCV tooltip (fixed at top-left of chart — grows rightward and downward)
        self._ohlcv_label = pg.TextItem(
            text="", color=_FG, anchor=(0.0, 0.0),
        )
        self._ohlcv_label.setFont(QFont("Monospace", 8))
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
        self._tick_profile_widget = pg.PlotWidget()
        self._tick_profile_widget.setBackground(_BG)
        self._tick_profile_widget.setMinimumWidth(160)
        self._tick_profile_widget.setMaximumWidth(200)
        self._tick_profile_widget.getPlotItem().showGrid(x=True, y=False, alpha=0.15)
        self._tick_profile_widget.getPlotItem().setMenuEnabled(False)
        right_layout.addWidget(self._tick_profile_widget, 1)

        # Session vol profile (bottom-right)
        self._profile_widget = pg.PlotWidget()
        self._profile_widget.setBackground(_BG)
        self._profile_widget.setMinimumWidth(160)
        self._profile_widget.setMaximumWidth(200)
        self._profile_widget.getPlotItem().showGrid(x=True, y=False, alpha=0.15)
        self._profile_widget.getPlotItem().setMenuEnabled(False)
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
        ts = datetime.now().strftime("%H:%M:%S")
        self._status.showMessage(f"{ts}  {msg}")

    # ── OpenD connection ──────────────────────────────────────────────────────

    def _connect_opend(self, host: str = "127.0.0.1", port: int = 11111) -> None:
        try:
            with self._ctx_lock:
                if self._ctx:
                    self._ctx.close()
                self._ctx = OpenQuoteContext(host=host, port=port)
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
            self._conn_btn.setText("Connect")
            self._log("Disconnected.")

    # ── Live mode tick subscription ───────────────────────────────────────────

    def _start_live(self) -> None:
        code = self._code_edit.text().strip()
        if not code or not self._ctx:
            return
        try:
            self._ctx.subscribe(code, [SubType.TICKER])
            tf = self._tf_combo.currentText()
            _, candle_mins = TIMEFRAME_MAP[tf]

            viewer = self

            class _Handler(TickerHandlerBase):
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
                            d      = str(row["direction"]).upper()
                            key    = ("buy" if d == "BUY"
                                      else ("sell" if d == "SELL" else "neutral"))
                            viewer._live_ticks[bucket][price][key] += vol

            self._ctx.set_handler(_Handler())
            interval = self._refresh_spin.value() * 1000
            self._refresh_timer.start(interval)
            self._log(f"Live: subscribed {code}, refreshing every "
                      f"{self._refresh_spin.value()}s")
        except Exception as exc:
            self._log(f"Live start error: {exc}")

    def _stop_live(self) -> None:
        self._refresh_timer.stop()
        code = self._code_edit.text().strip()
        try:
            if self._ctx and code:
                self._ctx.unsubscribe(code, [SubType.TICKER])
        except Exception:
            pass

    # ── Toolbar callbacks ─────────────────────────────────────────────────────

    def _on_tf_changed(self, tf: str) -> None:
        _, self._candle_mins = TIMEFRAME_MAP[tf]
        self._trigger_fetch()

    def _on_mode_changed(self, mode: str) -> None:
        self._date_edit.setEnabled(mode == "Historical")
        if mode == "Live":
            self._start_live()
        else:
            self._stop_live()
        self._trigger_fetch()

    def _on_refresh_changed(self, value: int) -> None:
        if self._refresh_timer.isActive():
            self._refresh_timer.start(value * 1000)

    def _on_indicator_toggle(self) -> None:
        # Toggle KD subplot row visibility
        if self._ind("kd"):
            self._plot_kd.show()
        else:
            self._plot_kd.hide()
        self._render(self._klines, self._ticks)

    def _on_session_toggle(self) -> None:
        self._rebuild_session_profile()

    def _on_range_changed(self) -> None:
        self._rebuild_session_profile()

    def _get_range_val(self) -> str:
        for val in ("1d", "3d", "7d"):
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
            return  # previous fetch still in progress

        tf           = self._tf_combo.currentText()
        _, cm        = TIMEFRAME_MAP[tf]
        self._candle_mins = cm
        historical   = self._mode_combo.currentText() == "Historical"
        date_str     = self._date_edit.text().strip()
        ind          = {k: self._ind(k) for k in self._ind_checks}

        with self._tick_lock:
            live_snap = {k: dict(v) for k, v in self._live_ticks.items()}

        params = {
            "code":        self._code_edit.text().strip(),
            "tf":          tf,
            "historical":  historical,
            "date_str":    date_str,
            "candle_mins": cm,
            "ind":         ind,
            "live_ticks":  live_snap,
        }
        self._log(f"Fetching K-lines ({tf}) ...")
        self._fetcher = DataFetcher(ctx, params)
        self._fetcher.ready.connect(self._on_data_ready)
        self._fetcher.error.connect(self._log)
        self._fetcher.start()

    def _on_data_ready(self, result: dict) -> None:
        self._klines      = result["klines"]
        self._ticks       = result["ticks"]
        self._warmup      = result["warmup"]
        self._smc_signals = result["smc_signals"]
        self._fvg_gaps    = result["fvg_gaps"]
        self._ob_blocks   = result["ob_blocks"]
        self._render(self._klines, self._ticks)

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
        show_ema    = self._ind("ema")
        red_up      = self._ind("red_up")
        bull_col    = _RED   if red_up else _GREEN
        bear_col    = _GREEN if red_up else _RED

        # Compose live + historical ticks
        buckets: dict = {}
        if ticks:
            buckets.update(ticks)
        if self._mode_combo.currentText() == "Live":
            with self._tick_lock:
                for bk, pd_ in self._live_ticks.items():
                    if bk not in buckets:
                        buckets[bk] = {}
                    for price, counts in pd_.items():
                        if price not in buckets[bk]:
                            buckets[bk][price] = dict(counts)
                        else:
                            for k in counts:
                                buckets[bk][price][k] = (
                                    buckets[bk][price].get(k, 0) + counts[k])

        # Candlesticks
        self._candle_item.set_data(klines, buckets if show_hm else None,
                                   cm, show_heatmap=show_hm, red_up=red_up)

        # FVG zones
        self._fvg_item.set_data(
            self._fvg_gaps if show_fvg else [], n)

        # Order Blocks
        self._ob_item.set_data(
            self._ob_blocks if show_ob else [], n)

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

        # KD subplot
        self._clear_kd_items()
        if show_kd:
            self._draw_kd(klines)

        # Trade Review overlay
        self._clear_trade_items()
        if self._trade_record is not None:
            self._draw_trade_review(klines)

        # X-axis time labels
        self._set_xaxis_ticks(klines)

        # Set view range only when Code or TF changes (i.e. a genuinely new
        # chart).  On live refreshes we preserve the user's pan/zoom state.
        code = self._code_edit.text().strip()
        chart_key = (code, tf)
        if chart_key != self._last_chart_key:
            self._last_chart_key = chart_key
            self._reset_view(n)

        # Session vol profile
        self._rebuild_session_profile()

        mode = self._mode_combo.currentText()
        self.setWindowTitle(
            f"Trade Viewer Qt  —  {code}  {tf}  {mode}")
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
        # Show the most recent 8 signals to give context
        recent = signals[-8:] if len(signals) > 8 else signals
        bull_col = _RED   if red_up else _GREEN
        bear_col = _GREEN if red_up else _RED
        for sig in recent:
            color   = bull_col if sig["direction"] == "bull" else bear_col
            label   = sig["type"]
            idx     = sig["idx"]
            price   = sig["price"]
            from_i  = sig.get("from_idx", max(0, idx - 5))

            # Horizontal reference line from swing to break bar
            line = pg.PlotCurveItem(
                x=[from_i, idx], y=[price, price],
                pen=pg.mkPen(color, width=1, style=Qt.PenStyle.DotLine),
            )
            self._plot_c.addItem(line)
            self._bos_items.append(line)

            # Label at break point
            txt = pg.TextItem(
                text=label, color=color,
                fill=pg.mkBrush(_qc(color, 120)),
                anchor=(0.5, 1.0),
            )
            txt.setFont(QFont("Monospace", 8))
            yoff = price * 1.0003 if sig["direction"] == "bull" else price * 0.9997
            txt.setPos(idx, yoff)
            self._plot_c.addItem(txt)
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
            self._plot_c.addItem(txt)
            self._delta_items.append(txt)

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
            self._plot_c.addItem(curve)
            self._ema_items.append(curve)

            # Label at right end
            lbl = pg.TextItem(
                text=f"EMA{period}", color=color, anchor=(0.0, 0.5),
            )
            lbl.setFont(QFont("Monospace", 6))
            lbl.setPos(len(klines) - 1, float(ema[-1]))
            self._plot_c.addItem(lbl)
            self._ema_items.append(lbl)

    def _clear_kd_items(self) -> None:
        for item in self._kd_items:
            self._plot_kd.removeItem(item)
        self._kd_items.clear()

    def _draw_kd(self, klines: pd.DataFrame) -> None:
        """Draw KD channel width (momentum) in the KD subplot.

        Plots the spread-width series, coloured gold (bullish) / purple (bearish)
        with a fill vs zero to make direction instantly readable.
        """
        if len(klines) < _KD_SLOW + 5:
            return

        warmup = self._warmup if self._warmup is not None else klines
        kd     = compute_kd(warmup, fast=_KD_FAST, slow=_KD_SLOW)

        # Align to klines length (warmup may be longer)
        n      = len(klines)
        width  = kd["width"].values[-n:]
        x      = np.arange(n)

        # Split into bull (>0) and bear (<0) segments for colouring
        bull_y = np.where(width >= 0, width, 0.0)
        bear_y = np.where(width <  0, width, 0.0)

        bull_fill = pg.FillBetweenItem(
            pg.PlotCurveItem(x=x, y=bull_y),
            pg.PlotCurveItem(x=x, y=np.zeros(n)),
            brush=_qc(_UP, 100),
        )
        bear_fill = pg.FillBetweenItem(
            pg.PlotCurveItem(x=x, y=bear_y),
            pg.PlotCurveItem(x=x, y=np.zeros(n)),
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
        spread = kd["spread"].values[-n:]
        spread_norm = spread / (np.abs(spread).max() or 1.0) * (width.max() or 1.0)
        spread_line = pg.PlotCurveItem(
            x=x, y=spread_norm,
            pen=pg.mkPen(_GREY, width=1, style=Qt.PenStyle.DashLine),
        )
        self._plot_kd.addItem(spread_line)
        self._kd_items.append(spread_line)

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

    # ── Session vol profile ───────────────────────────────────────────────────

    def _rebuild_session_profile(self) -> None:
        if self._klines is None:
            return
        pw     = self._profile_widget
        pw.clear()
        # pw.clear() removes all items — restore the crosshair sync line
        pw.addItem(self._profile_hline)

        range_val = self._get_range_val()
        klines    = apply_profile_range(self._klines, range_val)
        klines    = self._filter_sessions(klines)
        if klines.empty:
            return

        lo   = float(klines["low"].min())
        hi   = float(klines["high"].max())
        if hi <= lo:
            return
        n_bins  = 60
        bins    = np.linspace(lo, hi, n_bins + 1)
        centers = (bins[:-1] + bins[1:]) / 2
        volumes = np.zeros(n_bins)
        for _, row in klines.iterrows():
            mask = (centers >= float(row["low"])) & (centers <= float(row["high"]))
            n = int(mask.sum())
            if n:
                volumes[mask] += float(row["volume"]) / n

        # Horizontal bars: x0=0 (left edge), x1=volume (right edge), y=price centre
        bar = pg.BarGraphItem(
            x0=np.zeros(n_bins), x1=volumes,
            y=centers, height=(bins[1] - bins[0]) * 0.9,
            brush=_qc(_GOLD, 80), pen=pg.mkPen(None),
        )
        pw.addItem(bar)
        pw.getPlotItem().setLabel("top", range_val.upper(),
                                  **{"color": _FG, "size": "8pt"})

        # POC — line + separate TextItem so the label never clips at panel edges
        poc_idx = int(np.argmax(volumes))
        poc     = float(centers[poc_idx])
        poc_line = pg.InfiniteLine(
            pos=poc, angle=0, movable=False,
            pen=pg.mkPen(_RED, width=1),
        )
        poc_label = pg.TextItem(
            text=f"POC {poc:.2f}", color=_RED,
            fill=pg.mkBrush(_qc(_BG_TIP, 180)),
            anchor=(0.0, 1.0),   # top-left: text grows right and upward from anchor
        )
        poc_label.setFont(QFont("Monospace", 7))
        pw.addItem(poc_line)
        pw.addItem(poc_label, ignoreBounds=True)

        vb = pw.getViewBox()

        def _pin_poc_label() -> None:
            """Re-position POC label at left edge of current view."""
            xlo = vb.viewRange()[0][0]
            poc_label.setPos(xlo, poc)

        # Pin once now; re-pin whenever the view is panned / zoomed
        vb.sigRangeChanged.connect(lambda *_: _pin_poc_label())
        _pin_poc_label()

        # VAH / VAL (70 % of volume)
        total_vol = float(volumes.sum())
        if total_vol > 0:
            sorted_idx = np.argsort(volumes)[::-1]
            cumvol     = 0.0
            va_indices = []
            for idx in sorted_idx:
                cumvol += volumes[idx]
                va_indices.append(idx)
                if cumvol / total_vol >= 0.70:
                    break
            if va_indices:
                vah = float(centers[max(va_indices)])
                val = float(centers[min(va_indices)])
                for price, lbl in [(vah, "VAH"), (val, "VAL")]:
                    va_line = pg.InfiniteLine(
                        pos=price, angle=0, movable=False,
                        pen=pg.mkPen(_GOLD, width=1, style=Qt.PenStyle.DashLine),
                    )
                    va_label = pg.TextItem(
                        text=f"{lbl} {price:.2f}", color=_GOLD,
                        fill=pg.mkBrush(_qc(_BG_TIP, 180)),
                        anchor=(0.0, 0.0),   # top-left; text grows right and downward
                    )
                    va_label.setFont(QFont("Monospace", 7))
                    pw.addItem(va_line)
                    pw.addItem(va_label, ignoreBounds=True)

                    def _pin_va(lbl_item=va_label, p=price) -> None:
                        xlo = vb.viewRange()[0][0]
                        lbl_item.setPos(xlo, p)

                    vb.sigRangeChanged.connect(lambda *_, f=_pin_va: f())
                    _pin_va()

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
                if lo < hi:
                    if lo <= mins < hi:
                        return True
                else:
                    if mins >= lo or mins < (hi - 24 * 60):
                        return True
            return False
        mask = klines["time_key"].astype(str).apply(_in_session)
        return klines[mask]

    # ── Tick profile (single candle on hover) ─────────────────────────────────

    def _show_tick_profile(self, candle_idx: int) -> None:
        if self._klines is None or self._ticks is None:
            return
        if candle_idx < 0 or candle_idx >= len(self._klines):
            return

        row = self._klines.iloc[candle_idx]
        try:
            bar_end = datetime.strptime(
                str(row["time_key"])[:16], "%Y-%m-%d %H:%M")
            bk = candle_start(
                bar_end - timedelta(minutes=self._candle_mins), self._candle_mins)
        except ValueError:
            return

        pd_ = self._ticks.get(bk)
        if not pd_:
            return

        prices = sorted(pd_.keys())
        buys   = [pd_[p]["buy"]  for p in prices]
        sells  = [pd_[p]["sell"] for p in prices]

        pw = self._tick_profile_widget
        pw.clear()

        bin_h = (max(prices) - min(prices)) / max(len(prices), 1) * 0.9 if prices else 0.01
        bin_h = max(bin_h, 0.001)

        buys_arr  = np.array(buys,  dtype=float)
        sells_arr = np.array(sells, dtype=float)
        zeros     = np.zeros(len(prices))

        # Buys extend rightward: x0=0 → x1=buy_volume
        buy_bar = pg.BarGraphItem(
            x0=zeros, x1=buys_arr,
            y=prices, height=bin_h,
            brush=_qc(_GREEN, 140), pen=pg.mkPen(None),
        )
        # Sells extend leftward: x0=-sell_volume → x1=0
        sell_bar = pg.BarGraphItem(
            x0=-sells_arr, x1=zeros,
            y=prices, height=bin_h,
            brush=_qc(_RED, 140), pen=pg.mkPen(None),
        )
        pw.addItem(buy_bar)
        pw.addItem(sell_bar)
        pw.getPlotItem().setLabel(
            "top", str(row["time_key"])[:16], **{"color": _FG, "size": "7pt"})

        # Delta total as a label
        total_buy  = sum(buys)
        total_sell = sum(sells)
        delta      = total_buy - total_sell
        sign       = "+" if delta >= 0 else ""
        delta_str  = (f"{sign}{delta/1000:.0f}K" if abs(delta) >= 1000
                      else f"{sign}{delta}")
        d_col      = _UP if delta >= 0 else _DOWN
        dlbl       = pg.TextItem(
            text=f"Δ {delta_str}", color=d_col, anchor=(0.5, 0.0),
        )
        dlbl.setFont(QFont("Monospace", 7))
        if prices:
            dlbl.setPos(float(buys_arr.max()) / 2 if buys_arr.size else 0.0,
                        float(max(prices)))
        pw.addItem(dlbl)

    # ── Crosshair + tooltip ───────────────────────────────────────────────────

    def _on_mouse_move(self, pos) -> None:
        # pos is QPointF emitted directly by scene.sigMouseMoved
        in_candle = self._plot_c.sceneBoundingRect().contains(pos)
        in_vol    = self._plot_v.sceneBoundingRect().contains(pos)
        in_kd     = self._plot_kd.sceneBoundingRect().contains(pos)
        in_any    = in_candle or in_vol or in_kd

        if not in_any:
            for line in (self._vline, self._hline,
                         self._vline_v, self._vline_kd,
                         self._price_label, self._ohlcv_label,
                         self._profile_hline):
                line.setVisible(False)
            return

        # Map scene position to candle-plot data coordinates regardless of which
        # sub-plot the cursor is in (all share the same X axis via setXLink).
        if in_candle:
            mouse_pt = self._plot_c.vb.mapSceneToView(pos)
        elif in_vol:
            mouse_pt = self._plot_v.vb.mapSceneToView(pos)
        else:
            mouse_pt = self._plot_kd.vb.mapSceneToView(pos)

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
        self._vline_v.setPos(x);  self._vline_v.setVisible(True)
        if self._plot_kd.isVisible():
            self._vline_kd.setPos(x); self._vline_kd.setVisible(True)

        # Horizontal line only in candle plot
        self._hline.setPos(y);  self._hline.setVisible(in_candle)

        # Profile panel sync line
        self._profile_hline.setPos(y)
        self._profile_hline.setVisible(True)

        xlo, xhi = self._plot_c.vb.viewRange()[0]
        ylo, yhi = self._plot_c.vb.viewRange()[1]
        label_x  = xlo + (xhi - xlo) * 0.01  # ~1% from left edge; text grows rightward
        # Price label tracks cursor Y, left-aligned so full text is visible
        self._price_label.setPos(label_x, y)
        self._price_label.setText(f"{y:.2f}")
        self._price_label.setVisible(True)

        if self._klines is not None and not self._klines.empty:
            idx = int(round(x))
            idx = max(0, min(idx, len(self._klines) - 1))
            row = self._klines.iloc[idx]
            vol = int(row.get("volume", 0) or 0)
            # OHLCV label is pinned to top-left corner of the chart (anchor 0,0)
            top_y = yhi - (yhi - ylo) * 0.01
            self._ohlcv_label.setPos(label_x, top_y)
            self._ohlcv_label.setText(
                f"{str(row['time_key'])[:16]}\n"
                f"O {row['open']:.2f}  H {row['high']:.2f}\n"
                f"L {row['low']:.2f}  C {row['close']:.2f}\n"
                f"Vol {vol:,}"
            )
            self._ohlcv_label.setVisible(True)

            self._show_tick_profile(idx)

    # ── X-axis tick labels ────────────────────────────────────────────────────

    def _reset_view(self, n_bars: int, init_bars: int = 150) -> None:
        """Set the initial view to the last init_bars candles.

        Both X and Y are set explicitly — no autoRange() calls — so the
        view is always correct on first load without being overridden by
        the linked volume plot or a global bounding-rect auto-range.
        Live refreshes skip this method entirely so user zoom is preserved.
        """
        x_end   = n_bars - 1 + 3
        x_start = max(-1, n_bars - init_bars)

        # Derive Y range from the actually visible bars
        if self._klines is not None and not self._klines.empty:
            vis    = self._klines.iloc[max(0, n_bars - init_bars):]
            y_lo   = float(vis["low"].min())
            y_hi   = float(vis["high"].max())
            margin = (y_hi - y_lo) * 0.08
            self._plot_c.setXRange(x_start, x_end, padding=0)
            self._plot_c.setYRange(y_lo - margin, y_hi + margin, padding=0)
        else:
            self._plot_c.setXRange(x_start, x_end, padding=0)

        # Volume plot: X is already linked; only need a Y reset
        if self._klines is not None:
            vis_v = self._klines.iloc[max(0, n_bars - init_bars):]
            max_v = float(vis_v["volume"].max()) if not vis_v.empty else 1.0
            self._plot_v.setYRange(0, max_v * 1.15, padding=0)

    def _set_xaxis_ticks(self, klines: pd.DataFrame) -> None:
        """Map integer bar indices to time_key strings on x-axis."""
        n    = len(klines)
        step = max(1, n // 10)
        ticks = [
            (i, str(klines.iloc[i]["time_key"])[5:16])
            for i in range(0, n, step)
        ]
        self._plot_c.getAxis("bottom").setTicks([ticks])
        self._plot_v.getAxis("bottom").setTicks([ticks])
        if self._ind("kd"):
            self._plot_kd.getAxis("bottom").setTicks([ticks])

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        self._stop_live()
        with self._ctx_lock:
            if self._ctx:
                self._ctx.close()
                self._ctx = None
        if self._fetcher:
            self._fetcher.quit()
            self._fetcher.wait(1000)
        event.accept()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_float(v) -> float:
    """Return float(v) or 0.0 on error."""
    try:
        return float(v) if v is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


# ── Entry point ───────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Trade Viewer Qt (PyQtGraph)")
    ap.add_argument("--code",    default="US.SNDK")
    ap.add_argument("--tf",      default="5m",
                    choices=list(TIMEFRAME_MAP.keys()))
    ap.add_argument("--mode",    default="Live",
                    choices=["Live", "Historical"])
    ap.add_argument("--date",    default=None)
    ap.add_argument("--host",    default="127.0.0.1")
    ap.add_argument("--port",    type=int, default=11111)
    ap.add_argument("--refresh", type=int, default=15)
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    app  = QApplication(sys.argv)
    app.setApplicationName("Trade Viewer Qt")
    win  = TradeViewerQt(args)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
