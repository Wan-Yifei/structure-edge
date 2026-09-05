#!/usr/bin/env python3
"""K-line Replay Trainer -- practice discretionary trade decisions on
historical data without seeing the future.

Usage:
    uv run main.py replay_trainer

Workflow:
    1. Load a code/timeframe/date range (same cached kline fetcher as the
       main viewer).
    2. Jump to a point in time (manual entry or Random) -- the chart only
       ever shows bars up to that point; FVG/OB/Volume-Profile/Chandelier
       overlays are recomputed fresh from the visible slice only.
    3. Place a simulated trade: Long/Short, N shares, and either a fixed
       SL/TP or an ATR chandelier trailing stop.
    4. Step or Play the replay forward bar-by-bar -- each new bar is checked
       against the open trade's exit condition -- or Skip straight to the
       settlement outcome.
    5. On settlement the trade is drawn on the chart (entry arrow, SL/TP or
       trailing-stop line, exit marker) and saved to db/sim_trades.duckdb;
       the session-stats panel (win rate, total R, total $ P&L) reflects the
       full saved history, not just this run.

Standalone window -- does not modify trade_viewer_qt.py's own state/behavior,
just imports its already-decoupled rendering primitives and dialog.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import uuid
from datetime import datetime

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QToolBar, QLabel, QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox,
    QPushButton, QCheckBox, QRadioButton, QButtonGroup, QMessageBox,
)
from PyQt6.QtCore import QThread, pyqtSignal

from feeds.fetcher import fetch_klines
from strategy.smc.fvg import detect_fvg
from strategy.smc.order_blocks import detect_order_blocks
from strategy.smc.market_structure import detect_bos_choch
from strategy.chandelier_exit.atr import wilder_atr
from strategy.chandelier_exit.chandelier import (
    rolling_extremes, simulate_chandelier_exit, current_stop,
)
from backtest.engine import _find_exit
from db.sim_trades import SimTradesDB
from core.time_utils import session_for_timestamp
from analysis.trade_viewer_qt import (
    CandlestickItem, FvgItem, ObItem, ChandelierParamsDialog,
    _compute_profile_bins, _compute_poc_vah_val,
    _RED, _GREEN, _GREY, _FG, _BG, _BG_TIP, _CROSS, _GOLD, _qc,
)

_TF_CHOICES = ["1m", "3m", "5m", "15m", "30m", "60m", "1d"]

_SESSION_CFG_PATH = pathlib.Path(__file__).parent.parent / "config" / "schedule.json"
# Same four session windows as config/schedule.json's "sessions" block -- used
# as a fallback if that file is missing or doesn't have a "sessions" key.
_DEFAULT_SESSIONS = {
    "overnight":  {"start": "20:00", "end": "04:00", "enabled": True},
    "premarket":  {"start": "04:00", "end": "09:30", "enabled": True},
    "regular":    {"start": "09:30", "end": "16:00", "enabled": True},
    "afterhours": {"start": "16:00", "end": "20:00", "enabled": True},
}
# (checkbox attribute name, session key) -- session key must match config/schedule.json's
# "sessions" dict keys (and core.time_utils.session_for_timestamp's expected keys).
_SESSION_FILTER_CBS = [
    ("_sess_overnight_cb",  "overnight"),
    ("_sess_premarket_cb",  "premarket"),
    ("_sess_regular_cb",    "regular"),
    ("_sess_afterhours_cb", "afterhours"),
]


def _load_sessions_config() -> dict:
    if _SESSION_CFG_PATH.exists():
        try:
            cfg = json.loads(_SESSION_CFG_PATH.read_text(encoding="utf-8"))
            sessions = cfg.get("sessions")
            if sessions:
                return sessions
        except Exception:
            pass
    return _DEFAULT_SESSIONS
_MAX_BARS_IN_TRADE = 300   # cap on how far a trade can run before forced timeout
_MIN_LOOKBACK       = 60   # bars of history required before a Jump/Random point (indicator warmup)
_MIN_TRAILING        = 20   # bars required after a Jump/Random point (room for a trade to play out)


def _compute_r_multiple(direction: str, entry_price: float, exit_price: float, risk_unit: float) -> float:
    diff = (exit_price - entry_price) if direction == "bull" else (entry_price - exit_price)
    return diff / risk_unit


def _compute_pnl_usd(direction: str, shares: int, entry_price: float, exit_price: float) -> float:
    diff = (exit_price - entry_price) if direction == "bull" else (entry_price - exit_price)
    return shares * diff


def validate_chandelier_stop(direction: str, ref_price: float, init_stop: float) -> str | None:
    """Sanity-check that a freshly-computed chandelier stop is on the correct
    side of the reference (entry/fill) price. HH(period)/LL(period) is
    anchored to recent price extremes, not to ref_price -- if price has
    already pulled back/rallied far enough from that extreme, `init_stop` can
    end up past ref_price entirely, which would mean the position starts out
    already effectively stopped (a bull stop at/above entry, or a bear stop
    at/below entry). Returns an error message, or None if the stop is valid.
    """
    if direction == "bull" and init_stop >= ref_price:
        return (
            f"Chandelier stop ({init_stop:.4f}) is at or above the entry/fill price "
            f"({ref_price:.4f}) -- price has pulled back too far from the recent high "
            f"for this ATR multiplier. Try a larger multiplier, a shorter period, or "
            f"a different entry point.")
    if direction == "bear" and init_stop <= ref_price:
        return (
            f"Chandelier stop ({init_stop:.4f}) is at or below the entry/fill price "
            f"({ref_price:.4f}) -- price has rallied too far from the recent low for "
            f"this ATR multiplier. Try a larger multiplier, a shorter period, or a "
            f"different entry point.")
    return None


def check_settlement(
    klines: pd.DataFrame | None, trade: dict | None, replay_idx: int,
    max_bars_in_trade: int = _MAX_BARS_IN_TRADE,
) -> dict | None:
    """Check whether `trade` would be settled given bars revealed up through
    `replay_idx`. Pure function -- no Qt/instance state. Returns None if not
    yet settled (either no touch, or not enough bars revealed yet to tell the
    difference between "no touch" and "still running").

    `trade` must have: exit_mode ("fixed"|"chandelier"), entry_idx,
    entry_price, direction ("bull"|"bear"), and either (sl, tp) for fixed
    mode or (chandelier_period, chandelier_multiplier, risk_unit) for
    chandelier mode.
    """
    if trade is None or klines is None:
        return None
    entry_idx = trade["entry_idx"]
    bars_available = replay_idx - entry_idx

    highs  = klines["high"].to_numpy(dtype=float)
    lows   = klines["low"].to_numpy(dtype=float)
    closes = klines["close"].to_numpy(dtype=float)
    times  = klines["time_key"].astype(str).to_numpy()

    if trade["exit_mode"] == "fixed":
        if bars_available <= 0:
            return None   # _find_exit never checks the entry bar itself
        window = min(bars_available, max_bars_in_trade)
        exit_bar, exit_price, outcome = _find_exit(
            lows, highs, closes, from_bar=entry_idx + 1,
            sl=trade["sl"], tp=trade["tp"], direction=trade["direction"],
            max_bars=window,
        )
        if outcome == "timeout" and window < max_bars_in_trade:
            return None   # haven't revealed enough bars yet, not a real timeout
        cause = "sl" if outcome == "loss" else ("tp" if outcome == "win" else "timeout")
        risk_unit = abs(trade["entry_price"] - trade["sl"])
        r_mult = _compute_r_multiple(trade["direction"], trade["entry_price"], exit_price, risk_unit)
        return {
            "exit_bar": exit_bar, "exit_price": exit_price, "exit_time": times[exit_bar],
            "cause": cause, "result": "win" if r_mult > 0 else "loss", "r_multiple": r_mult,
        }

    period, mult = trade["chandelier_period"], trade["chandelier_multiplier"]
    window = min(bars_available, max_bars_in_trade)
    atr    = wilder_atr(highs, lows, closes, period)
    hh, ll = rolling_extremes(highs, lows, period)
    res = simulate_chandelier_exit(
        highs, lows, closes, times, atr, hh, ll,
        entry_idx=entry_idx, entry_price=trade["entry_price"], direction=trade["direction"],
        period=period, multiplier=mult, risk_unit=trade["risk_unit"], max_bars=window,
        entry_stop_override=trade.get("entry_stop_override"),
    )
    if res is None:
        return None
    if res.cause == "timeout" and window < max_bars_in_trade:
        return None
    cause = "chandelier" if res.cause == "stopped" else "timeout"
    return {
        "exit_bar": res.exit_bar, "exit_price": res.exit_price, "exit_time": res.exit_time,
        "cause": cause, "result": "win" if res.r_multiple > 0 else "loss",
        "r_multiple": res.r_multiple, "stop_series": res.stop_series,
    }


def check_limit_fill(
    klines: pd.DataFrame | None, order: dict | None, replay_idx: int,
) -> tuple[int, float] | None:
    """Check whether a pending limit order would have filled by replay_idx.

    A buy limit ("bull") fills when price trades down to or through the limit
    (low <= limit_price); a sell/short limit ("bear") fills when price trades
    up to or through it (high >= limit_price). Fills at exactly the limit
    price -- no slippage/gap modeling, same convention as check_settlement().
    The order's own placement bar is never checked (matches _find_exit's
    from_bar+1 convention: you can't fill on the same bar you placed the order).

    Returns (fill_bar, fill_price) or None if not filled yet.
    """
    if order is None or klines is None:
        return None
    placed_idx = order["placed_idx"]
    if replay_idx <= placed_idx:
        return None
    highs = klines["high"].to_numpy(dtype=float)[placed_idx + 1: replay_idx + 1]
    lows  = klines["low"].to_numpy(dtype=float)[placed_idx + 1: replay_idx + 1]
    limit_price = order["limit_price"]
    mask = (lows <= limit_price) if order["direction"] == "bull" else (highs >= limit_price)
    if not mask.any():
        return None
    fill_bar = placed_idx + 1 + int(np.argmax(mask))
    return fill_bar, limit_price


class _KlineFetchWorker(QThread):
    """Fetch klines in a background thread so Load never freezes the UI."""
    done  = pyqtSignal(object)   # emits pd.DataFrame
    error = pyqtSignal(str)

    def __init__(self, code: str, tf: str, start: str, end: str):
        super().__init__()
        self._code, self._tf, self._start, self._end = code, tf, start, end

    def run(self) -> None:
        try:
            df = fetch_klines(code=self._code, ktype=self._tf, start=self._start, end=self._end)
            self.done.emit(df)
        except Exception as exc:
            self.error.emit(str(exc))


class ReplayTrainerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("K-line Replay Trainer")
        self.resize(1500, 950)

        self._klines: pd.DataFrame | None = None
        self._replay_idx: int = 0
        self._code: str = ""
        self._fetcher: _KlineFetchWorker | None = None

        self._open_trade: dict | None = None      # filled/live sim trade -- see _on_place_trade
        self._pending_order: dict | None = None     # unfilled limit order -- see _place_limit_order
        self._settled_trade: dict | None = None    # last settled trade -- for chart overlay
        self._trade_items: list = []                 # chart items for the trade overlay
        self._profile_render_items: list = []          # chart items for the volume-profile bars/POC/VAH/VAL

        # Range Profile: a draggable region on the main chart whose bar span
        # feeds the same side-panel profile instead of the whole visible
        # history -- lets you isolate e.g. a session's initial volume build
        # (IVB) or any other sub-range, mirroring trade_viewer_qt.py's Range
        # Profile feature (same _compute_profile_bins/_compute_poc_vah_val).
        self._range_region: pg.LinearRegionItem | None = None
        self._range_profile_timer = QTimer(self)
        self._range_profile_timer.setSingleShot(True)
        self._range_profile_timer.timeout.connect(self._rebuild_range_profile)
        self._range_last_indices: tuple[int, int] = (-1, -1)

        self._chandelier_period     = 20
        self._chandelier_multiplier = 2.0

        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._on_step)

        self._db = SimTradesDB()

        self._build_ui()
        self._refresh_session_stats()

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)

        # Row 1: load bar
        tb1 = QToolBar("Load", self)
        tb1.setMovable(False)
        self.addToolBar(tb1)
        tb1.addWidget(QLabel("Code:"))
        self._code_edit = QLineEdit("US.SOXL")
        self._code_edit.setFixedWidth(90)
        tb1.addWidget(self._code_edit)
        tb1.addWidget(QLabel("TF:"))
        self._tf_combo = QComboBox()
        self._tf_combo.addItems(_TF_CHOICES)
        self._tf_combo.setCurrentText("30m")
        self._tf_combo.currentTextChanged.connect(self._on_tf_changed)
        tb1.addWidget(self._tf_combo)
        tb1.addWidget(QLabel("Start:"))
        self._start_edit = QLineEdit("2025-05-22")
        self._start_edit.setFixedWidth(90)
        tb1.addWidget(self._start_edit)
        tb1.addWidget(QLabel("End:"))
        self._end_edit = QLineEdit(datetime.now().strftime("%Y-%m-%d"))
        self._end_edit.setFixedWidth(90)
        tb1.addWidget(self._end_edit)
        load_btn = QPushButton("Load")
        load_btn.clicked.connect(self._on_load)
        tb1.addWidget(load_btn)
        tb1.addSeparator()
        self._red_up_cb = QCheckBox("Red Up")
        self._red_up_cb.setChecked(True)
        self._red_up_cb.stateChanged.connect(self._render)
        tb1.addWidget(self._red_up_cb)

        # Row 2: replay position controls
        self.addToolBarBreak()
        tb2 = QToolBar("Replay", self)
        tb2.setMovable(False)
        self.addToolBar(tb2)
        tb2.addWidget(QLabel("Jump to:"))
        self._jump_edit = QLineEdit()
        self._jump_edit.setPlaceholderText("YYYY-MM-DD HH:MM")
        self._jump_edit.setFixedWidth(130)
        tb2.addWidget(self._jump_edit)
        jump_btn = QPushButton("Go")
        jump_btn.clicked.connect(self._on_jump)
        tb2.addWidget(jump_btn)
        random_btn = QPushButton("🎲 Random")
        random_btn.clicked.connect(self._on_random)
        tb2.addWidget(random_btn)
        tb2.addWidget(QLabel("  session:"))
        self._sess_overnight_cb  = QCheckBox("Overnight")
        self._sess_premarket_cb  = QCheckBox("Premarket")
        self._sess_regular_cb    = QCheckBox("Regular")
        self._sess_afterhours_cb = QCheckBox("Afterhours")
        for attr, _key in _SESSION_FILTER_CBS:
            cb = getattr(self, attr)
            cb.setChecked(True)
            cb.setToolTip(
                "Restricts \U0001F3B2 Random to bars whose time falls in the "
                "checked session(s) -- uncheck the ones you don't want to "
                "practice (e.g. only Premarket for IVB drills). Manual "
                "\"Jump to\" is unaffected -- typing an exact time already "
                "gives full control over which session you land in."
            )
            tb2.addWidget(cb)
        tb2.addSeparator()
        step_btn = QPushButton("Step ▶")
        step_btn.clicked.connect(self._on_step)
        tb2.addWidget(step_btn)
        self._play_btn = QPushButton("Play ▶▶")
        self._play_btn.setCheckable(True)
        self._play_btn.toggled.connect(self._on_play_toggled)
        tb2.addWidget(self._play_btn)
        tb2.addWidget(QLabel("Speed(ms):"))
        self._speed_spin = QSpinBox()
        self._speed_spin.setRange(100, 3000)
        self._speed_spin.setValue(500)
        self._speed_spin.setSingleStep(100)
        tb2.addWidget(self._speed_spin)
        skip_btn = QPushButton("Skip to Result ⏭")
        skip_btn.clicked.connect(self._on_skip_to_result)
        tb2.addWidget(skip_btn)
        tb2.addSeparator()
        new_round_btn = QPushButton("New Round")
        new_round_btn.clicked.connect(self._on_new_round)
        tb2.addWidget(new_round_btn)

        # Row 3: overlays
        self.addToolBarBreak()
        tb3 = QToolBar("Overlays", self)
        tb3.setMovable(False)
        self.addToolBar(tb3)
        tb3.addWidget(QLabel("Overlays:"))
        self._fvg_cb = QCheckBox("FVG")
        self._fvg_cb.setChecked(True)
        self._fvg_cb.stateChanged.connect(self._render)
        tb3.addWidget(self._fvg_cb)
        self._fvg_min_pct = QDoubleSpinBox()
        self._fvg_min_pct.setRange(0.05, 5.0)
        self._fvg_min_pct.setSingleStep(0.05)
        self._fvg_min_pct.setValue(1.00)
        self._fvg_min_pct.setDecimals(2)
        self._fvg_min_pct.setSuffix("%")
        self._fvg_min_pct.setFixedWidth(72)
        self._fvg_min_pct.setToolTip("FVG minimum gap size (% of price)")
        self._fvg_min_pct.valueChanged.connect(self._render)
        tb3.addWidget(self._fvg_min_pct)

        self._ob_cb = QCheckBox("OB")
        self._ob_cb.setChecked(True)
        self._ob_cb.stateChanged.connect(self._render)
        tb3.addWidget(self._ob_cb)
        self._ob_max_count = QSpinBox()
        self._ob_max_count.setRange(1, 20)
        self._ob_max_count.setValue(4)
        self._ob_max_count.setFixedWidth(50)
        self._ob_max_count.setToolTip("Max number of recent OBs to display")
        self._ob_max_count.valueChanged.connect(self._render)
        tb3.addWidget(self._ob_max_count)
        self._profile_cb = QCheckBox("Volume Profile")
        self._profile_cb.setChecked(True)
        self._profile_cb.stateChanged.connect(self._render)
        tb3.addWidget(self._profile_cb)
        self._range_profile_btn = QPushButton("Range Profile")
        self._range_profile_btn.setCheckable(True)
        self._range_profile_btn.setChecked(False)
        self._range_profile_btn.setToolTip(
            "Drag the shaded region on the chart to compute the volume profile "
            "for just that bar range instead of the whole visible history -- "
            "e.g. isolate a session's initial volume build (IVB) or any other "
            "sub-range you want to inspect. Overrides Volume Profile while active."
        )
        self._range_profile_btn.clicked.connect(self._toggle_range_profile)
        tb3.addWidget(self._range_profile_btn)
        self._vol_cb = QCheckBox("Volume")
        self._vol_cb.setChecked(True)
        self._vol_cb.stateChanged.connect(self._render)
        tb3.addWidget(self._vol_cb)
        self._dv_cb = QCheckBox("DV")
        self._dv_cb.setChecked(True)
        self._dv_cb.setToolTip(
            "Delta Volume proxy (moomoo formula: DV = VOL*(C-O)).\n"
            "No tick data in replay mode, so this approximates aggressor "
            "buy/sell imbalance from OHLCV: positive = bar closed above its "
            "open (net-buy proxy), negative = closed below (net-sell proxy).")
        self._dv_cb.stateChanged.connect(self._render)
        tb3.addWidget(self._dv_cb)
        self._chandelier_cb = QCheckBox("Chandelier")
        self._chandelier_cb.setChecked(True)
        self._chandelier_cb.stateChanged.connect(self._render)
        tb3.addWidget(self._chandelier_cb)
        chandelier_settings_btn = QPushButton("⚙")
        chandelier_settings_btn.setFixedWidth(28)
        chandelier_settings_btn.setToolTip("Chandelier ATR period / multiplier")
        chandelier_settings_btn.clicked.connect(self._on_chandelier_settings)
        tb3.addWidget(chandelier_settings_btn)

        # Main body: chart | side panel (order entry + profile + stats)
        body = QHBoxLayout()
        root.addLayout(body)

        self._chart_widget = pg.GraphicsLayoutWidget()
        self._chart_widget.setBackground(_BG)
        body.addWidget(self._chart_widget, stretch=4)

        self._plot_c: pg.PlotItem = self._chart_widget.addPlot(row=0, col=0)
        self._plot_c.showGrid(x=True, y=True, alpha=0.15)
        self._plot_c.setMenuEnabled(False)
        self._plot_c.getAxis("left").setTextPen(_qc(_FG))
        self._plot_c.getAxis("bottom").setTextPen(_qc(_FG))

        self._candle_item = CandlestickItem()
        self._plot_c.addItem(self._candle_item)
        self._fvg_item = FvgItem()
        self._plot_c.addItem(self._fvg_item)
        self._ob_item = ObItem()
        self._plot_c.addItem(self._ob_item)

        self._chandelier_label = pg.TextItem(anchor=(1.0, 1.0), fill=pg.mkBrush(_qc(_BG_TIP, 200)))
        self._chandelier_label.setFont(QFont("Monospace", 8))
        self._chandelier_label.setZValue(60)
        self._chandelier_label.setVisible(False)
        self._plot_c.addItem(self._chandelier_label, ignoreBounds=True)
        self._plot_c.vb.sigRangeChanged.connect(self._pin_chandelier_label)

        # Volume subplot (row 1)
        self._chart_widget.nextRow()
        self._plot_vol: pg.PlotItem = self._chart_widget.addPlot(row=1, col=0)
        self._plot_vol.showGrid(x=True, y=True, alpha=0.10)
        self._plot_vol.setLabel("left", "Volume", **{"color": _FG})
        self._plot_vol.getAxis("left").setTextPen(_qc(_FG))
        self._plot_vol.getAxis("bottom").setTextPen(_qc(_FG))
        self._plot_vol.setMenuEnabled(False)
        self._plot_vol.setXLink(self._plot_c)
        self._vol_item = pg.BarGraphItem(x=[0], height=[0], width=0.7, brush=_qc(_GREY))
        self._plot_vol.addItem(self._vol_item)

        # Delta Volume subplot (row 2) -- moomoo formula: DV:(VOL*(C-O)), VOLSTICK.
        # OHLCV-only proxy for aggressor buy-sell imbalance (no tick data in
        # replay mode): positive when the bar closes above its open (proxying
        # net buying), negative when it closes below (net selling), scaled by
        # that bar's volume.
        self._chart_widget.nextRow()
        self._plot_dv: pg.PlotItem = self._chart_widget.addPlot(row=2, col=0)
        self._plot_dv.showGrid(x=True, y=True, alpha=0.10)
        self._plot_dv.setLabel("left", "DV", **{"color": _FG})
        self._plot_dv.getAxis("left").setTextPen(_qc(_FG))
        self._plot_dv.getAxis("bottom").setTextPen(_qc(_FG))
        self._plot_dv.setMenuEnabled(False)
        self._plot_dv.setXLink(self._plot_c)
        self._plot_dv.addItem(pg.InfiniteLine(
            pos=0, angle=0, movable=False,
            pen=pg.mkPen(_GREY, width=1, style=Qt.PenStyle.DashLine),
        ))
        self._dv_item = pg.BarGraphItem(x=[0], height=[0], width=0.7, brush=_qc(_GREY))
        self._plot_dv.addItem(self._dv_item)

        self._chart_widget.ci.layout.setRowStretchFactor(0, 5)
        self._set_subplot_row_visible(self._plot_vol, 1, True)
        self._set_subplot_row_visible(self._plot_dv, 2, True)

        # Volume profile side panel
        self._profile_widget = pg.PlotWidget()
        self._profile_widget.setBackground(_BG)
        self._profile_widget.setMinimumWidth(160)
        self._profile_widget.setMaximumWidth(220)
        self._profile_widget.getPlotItem().setMenuEnabled(False)
        self._profile_widget.getPlotItem().getAxis("left").setTextPen(_qc(_FG))
        self._profile_widget.getPlotItem().getAxis("bottom").setTextPen(_qc(_FG))
        body.addWidget(self._profile_widget, stretch=1)

        # ── Crosshair: vertical line in every subplot (shared X), horizontal
        # line in the main chart + a synced line in the volume profile panel ──
        cross_pen = pg.mkPen(_CROSS, width=1, style=Qt.PenStyle.DashLine)

        self._vline = pg.InfiniteLine(angle=90, movable=False, pen=cross_pen)
        self._hline = pg.InfiniteLine(angle=0,  movable=False, pen=cross_pen)
        self._vline.setVisible(False)
        self._hline.setVisible(False)
        self._plot_c.addItem(self._vline, ignoreBounds=True)
        self._plot_c.addItem(self._hline, ignoreBounds=True)

        self._vline_vol = pg.InfiniteLine(angle=90, movable=False, pen=cross_pen)
        self._vline_vol.setVisible(False)
        self._plot_vol.addItem(self._vline_vol, ignoreBounds=True)

        self._vline_dv = pg.InfiniteLine(angle=90, movable=False, pen=cross_pen)
        self._vline_dv.setVisible(False)
        self._plot_dv.addItem(self._vline_dv, ignoreBounds=True)

        self._profile_hline = pg.InfiniteLine(angle=0, movable=False, pen=cross_pen)
        self._profile_hline.setVisible(False)
        self._profile_widget.addItem(self._profile_hline, ignoreBounds=True)

        # Price tag (left edge, tracks cursor Y) + OHLCV tooltip (near cursor)
        self._price_label = pg.TextItem(
            text="", color=_GOLD, fill=pg.mkBrush(_qc(_BG_TIP, 180)), anchor=(0.0, 0.5))
        self._price_label.setFont(QFont("Monospace", 7))
        self._price_label.setZValue(100)
        self._price_label.setVisible(False)
        self._plot_c.addItem(self._price_label, ignoreBounds=True)

        self._ohlcv_label = pg.TextItem(
            text="", color="#ffffff", fill=pg.mkBrush(_qc(_BG_TIP, 220)), anchor=(0.0, 1.0))
        self._ohlcv_label.setFont(QFont("Monospace", 8))
        self._ohlcv_label.setZValue(100)
        self._ohlcv_label.setVisible(False)
        self._plot_c.addItem(self._ohlcv_label, ignoreBounds=True)

        self._chart_widget.scene().sigMouseMoved.connect(self._on_mouse_move)

        # Side dock: order entry + session stats
        side = QVBoxLayout()
        body.addLayout(side, stretch=0)

        side.addWidget(QLabel("<b>Account</b>"))
        capital_row = QHBoxLayout()
        capital_row.addWidget(QLabel("Starting capital:"))
        self._starting_capital_spin = QDoubleSpinBox()
        self._starting_capital_spin.setRange(100.0, 100_000_000.0)
        self._starting_capital_spin.setDecimals(2)
        self._starting_capital_spin.setSingleStep(1000.0)
        self._starting_capital_spin.setValue(10_000.0)
        self._starting_capital_spin.setPrefix("$")
        self._starting_capital_spin.setToolTip(
            "A trade is rejected if shares * entry/limit price would exceed "
            "your current balance (starting capital + all-time P&L).")
        self._starting_capital_spin.valueChanged.connect(self._refresh_session_stats)
        capital_row.addWidget(self._starting_capital_spin)
        side.addLayout(capital_row)

        side.addWidget(QLabel("<b>Order Entry</b>"))
        dir_row = QHBoxLayout()
        self._long_radio  = QRadioButton("Long")
        self._short_radio = QRadioButton("Short")
        self._long_radio.setChecked(True)
        self._dir_group = QButtonGroup(self)
        self._dir_group.addButton(self._long_radio)
        self._dir_group.addButton(self._short_radio)
        dir_row.addWidget(self._long_radio)
        dir_row.addWidget(self._short_radio)
        side.addLayout(dir_row)

        shares_row = QHBoxLayout()
        shares_row.addWidget(QLabel("Shares:"))
        self._shares_spin = QSpinBox()
        self._shares_spin.setRange(1, 1_000_000)
        self._shares_spin.setValue(100)
        shares_row.addWidget(self._shares_spin)
        side.addLayout(shares_row)

        order_type_row = QHBoxLayout()
        self._market_radio = QRadioButton("Market")
        self._limit_radio  = QRadioButton("Limit")
        self._market_radio.setChecked(True)
        self._order_type_group = QButtonGroup(self)
        self._order_type_group.addButton(self._market_radio)
        self._order_type_group.addButton(self._limit_radio)
        self._market_radio.toggled.connect(self._on_order_type_toggled)
        order_type_row.addWidget(self._market_radio)
        order_type_row.addWidget(self._limit_radio)
        side.addLayout(order_type_row)

        limit_row = QHBoxLayout()
        limit_row.addWidget(QLabel("Limit price:"))
        self._limit_price_spin = QDoubleSpinBox()
        self._limit_price_spin.setRange(0.0, 1_000_000.0)
        self._limit_price_spin.setDecimals(4)
        self._limit_price_spin.setEnabled(False)
        self._limit_price_spin.setToolTip(
            "Order sits PENDING until price trades through this level, then "
            "fills at exactly this price -- not an immediate market fill.")
        limit_row.addWidget(self._limit_price_spin)
        side.addLayout(limit_row)

        exit_mode_row = QHBoxLayout()
        self._fixed_radio      = QRadioButton("Fixed SL/TP")
        self._chandelier_radio = QRadioButton("Chandelier trail")
        self._fixed_radio.setChecked(True)
        self._exit_mode_group = QButtonGroup(self)
        self._exit_mode_group.addButton(self._fixed_radio)
        self._exit_mode_group.addButton(self._chandelier_radio)
        exit_mode_row.addWidget(self._fixed_radio)
        exit_mode_row.addWidget(self._chandelier_radio)
        side.addLayout(exit_mode_row)

        sl_row = QHBoxLayout()
        sl_row.addWidget(QLabel("SL:"))
        self._sl_spin = QDoubleSpinBox()
        self._sl_spin.setRange(0.0, 1_000_000.0)
        self._sl_spin.setDecimals(4)
        sl_row.addWidget(self._sl_spin)
        side.addLayout(sl_row)

        tp_row = QHBoxLayout()
        tp_row.addWidget(QLabel("TP:"))
        self._tp_spin = QDoubleSpinBox()
        self._tp_spin.setRange(0.0, 1_000_000.0)
        self._tp_spin.setDecimals(4)
        tp_row.addWidget(self._tp_spin)
        side.addLayout(tp_row)

        self._place_trade_btn = QPushButton("Place Trade")
        self._place_trade_btn.clicked.connect(self._on_place_trade)
        side.addWidget(self._place_trade_btn)

        self._cancel_order_btn = QPushButton("Cancel Order")
        self._cancel_order_btn.setEnabled(False)
        self._cancel_order_btn.clicked.connect(self._on_cancel_order)
        side.addWidget(self._cancel_order_btn)

        self._trade_status_lbl = QLabel("No open trade")
        self._trade_status_lbl.setWordWrap(True)
        side.addWidget(self._trade_status_lbl)

        side.addWidget(QLabel("<b>Session Stats (all-time)</b>"))
        self._stats_lbl = QLabel("")
        self._stats_lbl.setWordWrap(True)
        side.addWidget(self._stats_lbl)

        side.addStretch(1)

    # ── data loading ──────────────────────────────────────────────────────────

    def _on_tf_changed(self, _text: str) -> None:
        """Auto-reload when the TF combo changes, same as clicking Load --
        matches trade_viewer_qt.py's _tf_combo convention."""
        self._on_load()

    def _on_load(self) -> None:
        if self._fetcher is not None and self._fetcher.isRunning():
            return
        code  = self._code_edit.text().strip()
        tf    = self._tf_combo.currentText()
        start = self._start_edit.text().strip()
        end   = self._end_edit.text().strip()
        self._trade_status_lbl.setText("Loading...")
        self._fetcher = _KlineFetchWorker(code, tf, start, end)
        self._fetcher.done.connect(lambda df: self._on_data_ready(df, code))
        self._fetcher.error.connect(self._on_load_error)
        self._fetcher.start()

    def _on_load_error(self, msg: str) -> None:
        self._trade_status_lbl.setText("No open trade")
        QMessageBox.critical(self, "Load failed", msg)

    def _on_data_ready(self, df: pd.DataFrame, code: str) -> None:
        if df is None or df.empty:
            QMessageBox.warning(self, "No data", "No klines returned for this code/range.")
            return
        self._klines  = df.reset_index(drop=True)
        self._code    = code
        self._replay_idx = min(_MIN_LOOKBACK, len(self._klines) - 1)
        self._clear_open_trade_state()
        if self._range_profile_btn.isChecked():
            self._range_profile_btn.setChecked(False)   # old bar indices are meaningless against fresh data
            self._toggle_range_profile(False)
        self._render()
        self._reset_view()
        self._trade_status_lbl.setText("No open trade")

    def _reset_view(self, lookback_bars: int = 150) -> None:
        """Fit the candle chart's X/Y range to the last `lookback_bars` of the
        currently-visible (replayed-so-far) slice. Needed because pyqtgraph
        does not automatically re-fit the view when set_data() is called on
        an already-shown item with a very different bar count (e.g. after a
        big Jump) -- without this the view can be left showing a stale range
        that barely overlaps the new data, looking like "almost no bars".
        """
        if self._klines is None or self._klines.empty:
            return
        n = self._replay_idx + 1
        x0 = max(0, n - lookback_bars)
        window = self._klines.iloc[x0:n]
        if window.empty:
            return
        self._plot_c.setXRange(x0 - 0.5, n - 0.5, padding=0.02)
        self._plot_c.setYRange(
            float(window["low"].min()), float(window["high"].max()), padding=0.05)

    def _nudge_view_if_offscreen(self) -> None:
        """During Step/Play, re-center the view around the newest bar once it
        drifts into the last 15% of the visible window -- keeps a
        long-running trade's newest candle comfortably in view instead of
        letting it hug or scroll past the right edge. Only recenters when
        needed (not every single step) to avoid a jittery view."""
        xlo, xhi = self._plot_c.vb.viewRange()[0]
        width  = xhi - xlo
        newest = self._replay_idx
        margin = width * 0.15
        if newest > xhi - margin:
            half = width / 2
            self._plot_c.setXRange(newest - half, newest + half, padding=0)

    # ── jump / random ──────────────────────────────────────────────────────────

    def _on_jump(self) -> None:
        if self._klines is None or self._klines.empty:
            return
        text = self._jump_edit.text().strip()
        if not text:
            return
        times = self._klines["time_key"].astype(str).to_numpy()
        pos = int(np.searchsorted(times, text, side="left"))
        pos = max(0, min(pos, len(self._klines) - 1))
        self._replay_idx = pos
        self._render()
        self._reset_view()

    def _active_session_filter(self) -> set[str]:
        return {key for attr, key in _SESSION_FILTER_CBS if getattr(self, attr).isChecked()}

    def _session_filtered_indices(self, lo: int, hi: int, active: set[str]) -> list[int]:
        sessions_cfg = _load_sessions_config()
        times = self._klines["time_key"].iloc[lo : hi + 1].astype(str)
        result = []
        for i, ts in zip(range(lo, hi + 1), times):
            try:
                dt = datetime.strptime(ts[:16], "%Y-%m-%d %H:%M")
            except ValueError:
                continue
            if session_for_timestamp(dt, sessions_cfg) in active:
                result.append(i)
        return result

    def _on_random(self) -> None:
        if self._klines is None or self._klines.empty:
            return
        lo = _MIN_LOOKBACK
        hi = len(self._klines) - 1 - _MIN_TRAILING
        if hi <= lo:
            self._replay_idx = len(self._klines) - 1
            self._render()
            self._reset_view()
            return

        active = self._active_session_filter()
        if len(active) == len(_SESSION_FILTER_CBS):
            # All sessions checked -- no filtering needed, matches prior behavior.
            self._replay_idx = random.randint(lo, hi)
        else:
            candidates = self._session_filtered_indices(lo, hi, active)
            if not candidates:
                QMessageBox.warning(
                    self, "No matching bars",
                    "No bars in the loaded range fall in the checked session(s). "
                    "Check a different session or load a wider date range.")
                return
            self._replay_idx = random.choice(candidates)
        self._render()
        self._reset_view()

    # ── rendering ──────────────────────────────────────────────────────────────

    def _set_subplot_row_visible(self, plot_item: pg.PlotItem, row: int, visible: bool) -> None:
        """Show/hide a subplot row, collapsing its layout space to zero when
        hidden (PlotItem.hide() alone leaves an empty gap)."""
        layout = self._chart_widget.ci.layout
        if visible:
            plot_item.show()
            layout.setRowStretchFactor(row, 1)
            layout.setRowMaximumHeight(row, 16777215)
        else:
            plot_item.hide()
            layout.setRowStretchFactor(row, 0)
            layout.setRowMaximumHeight(row, 0)

    def _render(self) -> None:
        if self._klines is None or self._klines.empty:
            return
        visible = self._klines.iloc[: self._replay_idx + 1].reset_index(drop=True)
        n = len(visible)
        red_up = self._red_up_cb.isChecked()

        self._candle_item.set_data(visible, None, 1, show_heatmap=False, red_up=red_up)

        self._fvg_min_pct.setEnabled(self._fvg_cb.isChecked())
        if self._fvg_cb.isChecked():
            gaps = detect_fvg(
                visible, min_gap_pct=self._fvg_min_pct.value() / 100.0, require_displacement=False)
            self._fvg_item.set_data(gaps, n, red_up=red_up)
        else:
            self._fvg_item.set_data([], n)

        self._ob_max_count.setEnabled(self._ob_cb.isChecked())
        if self._ob_cb.isChecked():
            bos_signals = detect_bos_choch(visible)
            blocks = detect_order_blocks(visible, bos_signals, max_count=self._ob_max_count.value())
            self._ob_item.set_data(blocks, n)
        else:
            self._ob_item.set_data([], n)

        if self._range_profile_btn.isChecked() and self._range_region is not None:
            self._rebuild_range_profile()
        elif self._profile_cb.isChecked():
            self._update_volume_profile(visible, label=f"Visible {n}b")
        else:
            for item in self._profile_render_items:
                self._profile_widget.removeItem(item)
            self._profile_render_items = []

        self._set_subplot_row_visible(self._plot_vol, 1, self._vol_cb.isChecked())
        if self._vol_cb.isChecked():
            self._update_volume(visible, red_up)

        self._set_subplot_row_visible(self._plot_dv, 2, self._dv_cb.isChecked())
        if self._dv_cb.isChecked():
            self._update_dv(visible, red_up)

        if self._chandelier_cb.isChecked():
            self._update_chandelier_label(visible)
        else:
            self._chandelier_label.setVisible(False)

        if n:
            self.setWindowTitle(
                f"K-line Replay Trainer  —  {self._code}  "
                f"{str(visible['time_key'].iloc[-1])}  ({n} bars visible)")

    def _update_volume_profile(self, visible: pd.DataFrame, label: str | None = None) -> None:
        # NOTE: PlotWidget.clear() would wipe every item in the scene, including
        # self._profile_hline (the crosshair line added once in _build_ui) --
        # only remove this method's own previously-added items instead.
        for item in self._profile_render_items:
            self._profile_widget.removeItem(item)
        self._profile_render_items = []

        n = len(visible)
        if n < 2:
            self._profile_widget.getPlotItem().setLabel("top", "")
            return
        centers, volumes, _, _ = _compute_profile_bins(visible, None, 1, n_bins=60)
        if centers.size == 0:
            return
        poc, vah, val = _compute_poc_vah_val(centers, volumes)
        self._profile_widget.getPlotItem().setLabel(
            "top", label or "", **{"color": "#42a5f5", "size": "7pt"})
        bin_h = (centers[1] - centers[0]) if len(centers) > 1 else 1.0
        bars = pg.BarGraphItem(
            x0=np.zeros_like(volumes), x1=volumes, y=centers, height=bin_h * 0.9,
            brush=_qc("#546e7a", 160), pen=pg.mkPen(None),
        )
        self._profile_widget.addItem(bars)
        self._profile_render_items.append(bars)
        for price, color, tag in ((poc, "#ffee58", "POC"), (vah, "#ef5350", "VAH"), (val, "#26a69a", "VAL")):
            line = pg.InfiniteLine(
                pos=price, angle=0, movable=False,
                pen=pg.mkPen(color, width=1, style=Qt.PenStyle.DashLine),
                label=f"{tag} {price:.2f}", labelOpts={"color": color, "position": 0.02},
            )
            self._profile_widget.addItem(line)
            self._profile_render_items.append(line)

        # Project the current (latest revealed) price onto the profile -- a
        # solid line distinct from POC/VAH/VAL's dashed style, so it reads as
        # "where price is now" vs. "where the volume concentrated".
        last_price = float(visible["close"].iloc[-1])
        price_line = pg.InfiniteLine(
            pos=last_price, angle=0, movable=False,
            pen=pg.mkPen("#42a5f5", width=1.5),
            label=f"Last {last_price:.2f}", labelOpts={"color": "#42a5f5", "position": 0.95},
        )
        self._profile_widget.addItem(price_line)
        self._profile_render_items.append(price_line)

        self._profile_widget.setYRange(float(visible["low"].min()), float(visible["high"].max()), padding=0.02)

    # ── Range Profile ────────────────────────────────────────────────────────

    def _toggle_range_profile(self, checked: bool) -> None:
        if checked:
            if self._klines is None or self._klines.empty:
                self._range_profile_btn.setChecked(False)
                return
            # Default span: middle 40% of the currently visible X range,
            # same convention as trade_viewer_qt.py's Range Profile.
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
            self._range_last_indices = (-1, -1)
            self._render()   # restores the whole-visible-history profile, if checked

    def _on_range_region_changed(self) -> None:
        self._range_profile_timer.start(150)

    def _rebuild_range_profile(self) -> None:
        if self._range_region is None or self._klines is None:
            return
        n = self._replay_idx + 1   # can't select bars not yet revealed
        if n == 0:
            return
        rx0, rx1 = self._range_region.getRegion()
        i0 = max(0, min(n - 1, int(round(min(rx0, rx1)))))
        i1 = max(0, min(n - 1, int(round(max(rx0, rx1)))))
        if (i0, i1) == self._range_last_indices:
            return
        self._range_last_indices = (i0, i1)
        klines_slice = self._klines.iloc[i0 : i1 + 1].reset_index(drop=True)
        self._update_volume_profile(klines_slice, label=f"Range {i1 - i0 + 1}b")

    def _update_volume(self, visible: pd.DataFrame, red_up: bool) -> None:
        n     = len(visible)
        x     = np.arange(n)
        vols  = visible["volume"].fillna(0).to_numpy(dtype=float)
        opens = visible["open"].to_numpy(dtype=float)
        closes = visible["close"].to_numpy(dtype=float)
        bull_col = _RED if red_up else _GREEN
        bear_col = _GREEN if red_up else _RED
        colors = [_qc(bull_col, 140) if c >= o else _qc(bear_col, 140) for o, c in zip(opens, closes)]
        self._vol_item.setOpts(x=x, height=vols, width=0.7, brushes=colors)

    def _update_dv(self, visible: pd.DataFrame, red_up: bool) -> None:
        """moomoo formula: DV:(VOL*(C-O)), VOLSTICK -- OHLCV-only proxy for
        aggressor buy/sell imbalance (no tick data available in replay mode)."""
        n     = len(visible)
        x     = np.arange(n)
        vols  = visible["volume"].fillna(0).to_numpy(dtype=float)
        opens = visible["open"].to_numpy(dtype=float)
        closes = visible["close"].to_numpy(dtype=float)
        dv = vols * (closes - opens)
        bull_col = _RED if red_up else _GREEN
        bear_col = _GREEN if red_up else _RED
        colors = [_qc(bull_col, 180) if v >= 0 else _qc(bear_col, 180) for v in dv]
        self._dv_item.setOpts(x=x, height=dv, width=0.7, brushes=colors)

    def _update_chandelier_label(self, visible: pd.DataFrame) -> None:
        highs  = visible["high"].to_numpy(dtype=float)
        lows   = visible["low"].to_numpy(dtype=float)
        closes = visible["close"].to_numpy(dtype=float)
        direction = "bull" if self._long_radio.isChecked() else "bear"
        r = current_stop(highs, lows, closes, self._chandelier_period, self._chandelier_multiplier, direction)
        if r is None:
            self._chandelier_label.setHtml(
                f"<span style='color:{_GREY}'>Chandelier: need &ge;{self._chandelier_period} bars</span>")
        else:
            dir_label = "Long" if direction == "bull" else "Short"
            col = _GREEN if direction == "bull" else _RED
            offset     = r["atr"] * self._chandelier_multiplier
            offset_pct = offset / r["price"] * 100 if r["price"] else 0.0
            self._chandelier_label.setHtml(
                f"<span style='color:{_FG}'>Chandelier {dir_label}  "
                f"(p{self._chandelier_period} x{self._chandelier_multiplier:g})</span><br>"
                f"<span style='color:{col}'>Stop: {r['stop']:.4f}</span><br>"
                f"<span style='color:{_GREY}'>Dist: {r['dist']:.4f}  ({r['dist_pct']:.2f}%)</span><br>"
                f"<span style='color:{_GREY}'>ATR&times;mult: {offset:.4f}  ({offset_pct:.2f}%)</span>"
            )
        self._chandelier_label.setVisible(True)
        self._pin_chandelier_label()

    def _pin_chandelier_label(self, *_) -> None:
        if not self._chandelier_label.isVisible():
            return
        xlo, xhi = self._plot_c.vb.viewRange()[0]
        ylo, yhi = self._plot_c.vb.viewRange()[1]
        x_pad = (xhi - xlo) * 0.01
        y_pad = (yhi - ylo) * 0.015
        self._chandelier_label.setPos(xhi - x_pad, ylo + y_pad)

    def _on_chandelier_settings(self) -> None:
        dlg = ChandelierParamsDialog(self._chandelier_period, self._chandelier_multiplier, "bull", self)
        if dlg.exec():
            period, mult, _direction = dlg.values()
            self._chandelier_period     = period
            self._chandelier_multiplier = mult
            self._render()

    # ── crosshair ────────────────────────────────────────────────────────────────

    def _on_mouse_move(self, pos) -> None:
        """Crosshair: vertical line synced across candle/volume/DV (shared X
        axis), horizontal line in the main chart synced to the volume-profile
        panel's horizontal line (same Y = price axis)."""
        in_candle = self._plot_c.sceneBoundingRect().contains(pos)
        in_vol    = self._plot_vol.isVisible() and self._plot_vol.sceneBoundingRect().contains(pos)
        in_dv     = self._plot_dv.isVisible()  and self._plot_dv.sceneBoundingRect().contains(pos)
        in_any    = in_candle or in_vol or in_dv

        if not in_any:
            for line in (self._vline, self._hline, self._vline_vol, self._vline_dv,
                         self._price_label, self._ohlcv_label, self._profile_hline):
                line.setVisible(False)
            return

        if in_candle:
            mouse_pt = self._plot_c.vb.mapSceneToView(pos)
        elif in_vol:
            mouse_pt = self._plot_vol.vb.mapSceneToView(pos)
        else:
            mouse_pt = self._plot_dv.vb.mapSceneToView(pos)
        x = mouse_pt.x()
        y = mouse_pt.y() if in_candle else self._plot_c.vb.mapSceneToView(pos).y()

        self._vline.setPos(x); self._vline.setVisible(True)
        if self._plot_vol.isVisible():
            self._vline_vol.setPos(x); self._vline_vol.setVisible(True)
        if self._plot_dv.isVisible():
            self._vline_dv.setPos(x); self._vline_dv.setVisible(True)
        self._hline.setPos(y); self._hline.setVisible(in_candle)

        self._profile_hline.setPos(y)
        self._profile_hline.setVisible(True)

        xlo, xhi = self._plot_c.vb.viewRange()[0]
        ylo, yhi = self._plot_c.vb.viewRange()[1]
        y_span  = yhi - ylo
        label_x = xlo + (xhi - xlo) * 0.01

        self._price_label.setPos(label_x, y)
        self._price_label.setText(f"{y:.4f}")
        self._price_label.setVisible(True)

        if self._klines is not None and not self._klines.empty:
            idx = max(0, min(int(round(x)), self._replay_idx))
            row = self._klines.iloc[idx]
            vol = int(row.get("volume", 0) or 0)
            tip_y = min(y + y_span * 0.14, yhi - y_span * 0.02)
            self._ohlcv_label.setPos(label_x, tip_y)
            self._ohlcv_label.setText(
                f"{str(row['time_key'])[:16]}\n"
                f"O {row['open']:.2f}  H {row['high']:.2f}\n"
                f"L {row['low']:.2f}  C {row['close']:.2f}\n"
                f"Vol {vol:,}"
            )
            self._ohlcv_label.setVisible(True)
        else:
            self._ohlcv_label.setVisible(False)

    # ── order entry ──────────────────────────────────────────────────────────────

    def _on_order_type_toggled(self, market_checked: bool) -> None:
        self._limit_price_spin.setEnabled(not market_checked)

    def _build_exit_config(self, direction: str, ref_price: float, warmup_idx: int | None) -> dict | None:
        """Validate + build the exit-mode fields of a trade/pending-order dict.

        `ref_price` is the price SL/TP get validated against -- the market
        entry price, or a pending limit order's limit price. `warmup_idx` is
        the bar to compute the chandelier risk_unit from right now; pass None
        to defer that calc to fill time instead (used for pending limit
        orders, since the fill bar/price aren't known yet at placement time).
        Shows a warning + returns None on invalid input.
        """
        if self._fixed_radio.isChecked():
            sl, tp = self._sl_spin.value(), self._tp_spin.value()
            if sl <= 0 or tp <= 0:
                QMessageBox.warning(self, "Missing SL/TP", "Enter both SL and TP prices.")
                return None
            if direction == "bull" and not (sl < ref_price < tp):
                QMessageBox.warning(self, "Invalid SL/TP", "For a long: SL must be below entry, TP above.")
                return None
            if direction == "bear" and not (tp < ref_price < sl):
                QMessageBox.warning(self, "Invalid SL/TP", "For a short: TP must be below entry, SL above.")
                return None
            return {"exit_mode": "fixed", "sl": sl, "tp": tp}

        period, mult = self._chandelier_period, self._chandelier_multiplier
        cfg = {"exit_mode": "chandelier", "chandelier_period": period, "chandelier_multiplier": mult}
        if warmup_idx is None:
            return cfg   # risk_unit + entry_stop_override computed later, at fill time
        highs  = self._klines["high"].to_numpy(dtype=float)
        lows   = self._klines["low"].to_numpy(dtype=float)
        closes = self._klines["close"].to_numpy(dtype=float)
        atr    = wilder_atr(highs, lows, closes, period)
        if np.isnan(atr[warmup_idx]):
            QMessageBox.warning(
                self, "Insufficient ATR warmup",
                f"Need >= {period} bars of history before this point for the chandelier stop.")
            return None
        # Entry-anchored initial stop (entry_price -+ mult*ATR), not the
        # classic indicator's HH/LL-anchored one -- guarantees by
        # construction that the starting stop sits on the correct side of
        # entry, regardless of how far price has drifted from the recent
        # high/low. The ratchet still trails via HH/LL from the next bar
        # onward exactly as before (see entry_stop_override in
        # strategy/chandelier_exit/chandelier.py:simulate_chandelier_exit).
        atr_v = float(atr[warmup_idx])
        entry_stop = (ref_price - atr_v * mult) if direction == "bull" else (ref_price + atr_v * mult)
        err = validate_chandelier_stop(direction, ref_price, entry_stop)
        if err:
            QMessageBox.warning(self, "Invalid chandelier setup", err)
            return None
        cfg["risk_unit"]            = abs(ref_price - entry_stop)
        cfg["entry_stop_override"] = entry_stop
        return cfg

    def _on_place_trade(self) -> None:
        if self._open_trade is not None:
            QMessageBox.warning(self, "Trade already open", "Step/Play/Skip to settle it first.")
            return
        if self._pending_order is not None:
            QMessageBox.warning(self, "Order already pending", "Cancel it first to place a different one.")
            return
        if self._klines is None or self._klines.empty:
            return

        direction = "bull" if self._long_radio.isChecked() else "bear"
        shares    = self._shares_spin.value()
        if self._limit_radio.isChecked():
            self._place_limit_order(direction, shares)
        else:
            self._place_market_trade(direction, shares)

    def _check_capital(self, shares: int, price: float) -> bool:
        """Reject the order if shares * price would exceed the current
        balance (starting capital + all-time P&L). Shows a warning and
        returns False on rejection."""
        cost    = shares * price
        balance = self._current_balance()
        if cost > balance:
            QMessageBox.warning(
                self, "Insufficient capital",
                f"This order needs ${cost:,.2f} ({shares} sh @ {price:.4f}) "
                f"but your balance is only ${balance:,.2f}.")
            return False
        return True

    def _place_market_trade(self, direction: str, shares: int) -> None:
        entry_idx   = self._replay_idx
        entry_price = float(self._klines["close"].iloc[entry_idx])
        entry_time  = str(self._klines["time_key"].iloc[entry_idx])

        if not self._check_capital(shares, entry_price):
            return

        exit_cfg = self._build_exit_config(direction, entry_price, entry_idx)
        if exit_cfg is None:
            return

        trade = {
            "trade_id": uuid.uuid4().hex[:12],
            "direction": direction, "entry_idx": entry_idx,
            "entry_price": entry_price, "entry_time": entry_time, "shares": shares,
            **exit_cfg,
        }
        self._open_trade = trade
        self._clear_trade_items()
        self._settled_trade = None
        self._update_order_panel_enabled("open")
        dir_label = "LONG" if direction == "bull" else "SHORT"
        self._trade_status_lbl.setText(
            f"OPEN: {dir_label} {shares} sh @ {entry_price:.4f}  ({trade['exit_mode']})")
        self._draw_open_trade_marker()

        # Chandelier entries can self-stop on the entry bar itself (by design) --
        # check immediately in case that's already true.
        settlement = self._check_settlement()
        if settlement is not None:
            self._settle(settlement)

    def _place_limit_order(self, direction: str, shares: int) -> None:
        limit_price = self._limit_price_spin.value()
        if limit_price <= 0:
            QMessageBox.warning(self, "Missing limit price", "Enter a limit price.")
            return
        current_price = float(self._klines["close"].iloc[self._replay_idx])
        if direction == "bull" and limit_price >= current_price:
            QMessageBox.warning(
                self, "Invalid limit price",
                "A buy limit must be below the current price (waiting for a pullback).")
            return
        if direction == "bear" and limit_price <= current_price:
            QMessageBox.warning(
                self, "Invalid limit price",
                "A sell/short limit must be above the current price (waiting for a rally).")
            return
        if not self._check_capital(shares, limit_price):
            return

        exit_cfg = self._build_exit_config(direction, limit_price, warmup_idx=None)
        if exit_cfg is None:
            return

        order = {
            "order_id": uuid.uuid4().hex[:12],
            "direction": direction, "limit_price": limit_price,
            "placed_idx": self._replay_idx, "shares": shares,
            **exit_cfg,
        }
        self._pending_order = order
        self._clear_trade_items()
        self._settled_trade = None
        self._update_order_panel_enabled("pending")
        dir_label = "LONG" if direction == "bull" else "SHORT"
        self._trade_status_lbl.setText(
            f"PENDING: {dir_label} {shares} sh limit @ {limit_price:.4f}  ({order['exit_mode']})")
        self._draw_pending_order_marker()

    def _on_cancel_order(self) -> None:
        if self._pending_order is None:
            return
        self._pending_order = None
        self._clear_trade_items()
        self._update_order_panel_enabled("idle")
        self._trade_status_lbl.setText("No open trade")

    def _try_fill_pending_order(self) -> None:
        """Check whether the pending limit order would fill given bars
        revealed so far; if so, convert it into an open trade (computing the
        chandelier risk_unit now, at the actual fill bar, if applicable)."""
        order = self._pending_order
        if order is None or self._klines is None:
            return
        fill = check_limit_fill(self._klines, order, self._replay_idx)
        if fill is None:
            return
        fill_bar, fill_price = fill
        direction = order["direction"]

        trade = {
            "trade_id": order["order_id"], "direction": direction, "entry_idx": fill_bar,
            "entry_price": fill_price, "entry_time": str(self._klines["time_key"].iloc[fill_bar]),
            "shares": order["shares"], "exit_mode": order["exit_mode"],
        }
        if order["exit_mode"] == "fixed":
            trade["sl"] = order["sl"]
            trade["tp"] = order["tp"]
        else:
            period, mult = order["chandelier_period"], order["chandelier_multiplier"]
            highs  = self._klines["high"].to_numpy(dtype=float)
            lows   = self._klines["low"].to_numpy(dtype=float)
            closes = self._klines["close"].to_numpy(dtype=float)
            atr    = wilder_atr(highs, lows, closes, period)
            # Bar count only grows as the replay advances, so if there was ever
            # enough warmup this succeeds; guarded anyway for a pathological
            # very-early fill.
            if np.isnan(atr[fill_bar]):
                QMessageBox.warning(
                    self, "Insufficient ATR warmup at fill",
                    f"Need >= {period} bars of history; order cancelled.")
                self._on_cancel_order()
                return
            # Entry-anchored initial stop -- see the matching comment in
            # _build_exit_config() for why (guarantees the right side of fill_price).
            atr_v = float(atr[fill_bar])
            entry_stop = (fill_price - atr_v * mult) if direction == "bull" else (fill_price + atr_v * mult)
            err = validate_chandelier_stop(direction, fill_price, entry_stop)
            if err:
                QMessageBox.warning(self, "Invalid chandelier setup at fill", err + " Order cancelled.")
                self._on_cancel_order()
                return
            trade["chandelier_period"]      = period
            trade["chandelier_multiplier"]  = mult
            trade["entry_stop_override"]    = entry_stop
            trade["risk_unit"] = abs(fill_price - entry_stop)

        self._pending_order = None
        self._open_trade    = trade
        self._clear_trade_items()
        self._update_order_panel_enabled("open")
        self._draw_open_trade_marker()
        dir_label = "LONG" if direction == "bull" else "SHORT"
        self._trade_status_lbl.setText(
            f"OPEN (filled): {dir_label} {trade['shares']} sh @ {fill_price:.4f}  ({trade['exit_mode']})")

    def _expire_pending_order(self) -> None:
        """Ran out of loaded data before a pending limit order ever filled."""
        self._pending_order = None
        self._clear_trade_items()
        self._update_order_panel_enabled("idle")
        self._trade_status_lbl.setText("No open trade")
        QMessageBox.information(self, "Order expired", "The limit order never filled within the loaded data.")

    def _update_order_panel_enabled(self, mode: str) -> None:
        """mode: 'idle' (can place a new order), 'pending' (limit order
        waiting to fill, can only Cancel), 'open' (trade live, can only
        Step/Play/Skip)."""
        idle = mode == "idle"
        for w in (self._long_radio, self._short_radio, self._shares_spin,
                  self._fixed_radio, self._chandelier_radio, self._sl_spin, self._tp_spin,
                  self._market_radio, self._limit_radio, self._place_trade_btn):
            w.setEnabled(idle)
        self._limit_price_spin.setEnabled(idle and self._limit_radio.isChecked())
        self._cancel_order_btn.setEnabled(mode == "pending")

    # ── settlement ──────────────────────────────────────────────────────────────

    def _check_settlement(self) -> dict | None:
        """Check whether the open trade would be settled given bars revealed so
        far. Pure query -- does not mutate replay/trade state. See
        check_settlement() (module-level, unit-tested independently of Qt)."""
        return check_settlement(self._klines, self._open_trade, self._replay_idx, _MAX_BARS_IN_TRADE)

    def _force_timeout_settle(self) -> None:
        """Ran out of loaded data before the trade settled naturally -- close
        it at the last available bar's close as a timeout."""
        trade = self._open_trade
        if trade is None or self._klines is None:
            return
        last_idx    = len(self._klines) - 1
        exit_price  = float(self._klines["close"].iloc[last_idx])
        exit_time   = str(self._klines["time_key"].iloc[last_idx])
        risk_unit   = (abs(trade["entry_price"] - trade["sl"]) if trade["exit_mode"] == "fixed"
                       else trade["risk_unit"])
        r_mult = _compute_r_multiple(trade["direction"], trade["entry_price"], exit_price, risk_unit)
        self._replay_idx = last_idx
        self._render()
        self._settle({
            "exit_bar": last_idx, "exit_price": exit_price, "exit_time": exit_time,
            "cause": "timeout", "result": "win" if r_mult > 0 else "loss", "r_multiple": r_mult,
        })

    def _on_step(self) -> None:
        """Advance the replay position by one bar. Works whether or not a
        trade/pending order is active -- Step/Play are general "reveal the
        next bar" browsing controls, not something that only activates once
        you've placed an order. When a limit order is pending, the newly-
        revealed bar is checked for a fill; when a trade IS open, it's
        checked for settlement."""
        if self._klines is None:
            self._play_timer.stop()
            self._play_btn.setChecked(False)
            return
        if self._replay_idx >= len(self._klines) - 1:
            self._play_timer.stop()
            self._play_btn.setChecked(False)
            if self._open_trade is not None:
                self._force_timeout_settle()
            elif self._pending_order is not None:
                self._expire_pending_order()
            return
        self._replay_idx += 1
        self._render()
        self._nudge_view_if_offscreen()

        if self._pending_order is not None:
            self._try_fill_pending_order()

        if self._open_trade is not None:
            self._draw_open_trade_marker()
            settlement = self._check_settlement()
            if settlement is not None:
                self._play_timer.stop()
                self._play_btn.setChecked(False)
                self._settle(settlement)

    def _on_play_toggled(self, checked: bool) -> None:
        if checked:
            self._play_timer.start(self._speed_spin.value())
        else:
            self._play_timer.stop()

    def _on_skip_to_result(self) -> None:
        if self._open_trade is None and self._pending_order is None:
            return
        if self._klines is None:
            return
        self._play_timer.stop()
        self._play_btn.setChecked(False)

        if self._pending_order is not None:
            self._replay_idx = len(self._klines) - 1
            self._try_fill_pending_order()
            if self._pending_order is not None:
                # Scanned to the end of the loaded data and it still never filled.
                self._expire_pending_order()
                return
            # Filled -- rewind the display to the fill bar before continuing
            # on to settlement, same treatment as the market-order path below.
            self._replay_idx = self._open_trade["entry_idx"]
            self._render()

        if self._open_trade is None:
            return
        self._replay_idx = len(self._klines) - 1
        settlement = self._check_settlement()
        if settlement is None:
            self._force_timeout_settle()
            return
        self._replay_idx = settlement["exit_bar"]
        self._render()
        self._reset_view()
        self._draw_open_trade_marker()
        self._settle(settlement)

    def _settle(self, settlement: dict) -> None:
        trade = self._open_trade
        if trade is None:
            return
        entry_price = trade["entry_price"]
        exit_price  = settlement["exit_price"]
        shares      = trade["shares"]
        direction   = trade["direction"]
        pnl_usd = _compute_pnl_usd(direction, shares, entry_price, exit_price)

        record = {
            "trade_id": trade["trade_id"], "symbol": self._code, "direction": direction,
            "entry_time": trade["entry_time"], "exit_time": str(settlement["exit_time"]),
            "entry_price": entry_price, "exit_price": exit_price,
            "sl_price": trade.get("sl"), "tp_price": trade.get("tp"),
            "exit_cause": settlement["cause"], "result": settlement["result"],
            "r_multiple": settlement["r_multiple"], "shares": shares, "pnl_usd": pnl_usd,
            "chandelier_period": trade.get("chandelier_period"),
            "chandelier_multiplier": trade.get("chandelier_multiplier"),
        }
        self._db.insert_trade(record)

        self._settled_trade = {
            **trade, **settlement, "pnl_usd": pnl_usd, "exit_price": exit_price,
        }
        self._open_trade = None
        self._update_order_panel_enabled("idle")
        self._draw_trade_outcome()
        self._refresh_session_stats()

        dir_label = "LONG" if direction == "bull" else "SHORT"
        result_word = "WIN" if settlement["result"] == "win" else "LOSS"
        self._trade_status_lbl.setText(
            f"SETTLED ({result_word}): {dir_label} {shares} sh  "
            f"{entry_price:.4f} -> {exit_price:.4f}  "
            f"R={settlement['r_multiple']:+.2f}  P&L=${pnl_usd:+.2f}  "
            f"[{settlement['cause']}]"
        )
        QMessageBox.information(
            self, "Trade settled",
            f"{result_word}\n\n"
            f"Direction: {dir_label}\nShares: {shares}\n"
            f"Entry: {entry_price:.4f}  ->  Exit: {exit_price:.4f}\n"
            f"Exit cause: {settlement['cause']}\n"
            f"R-multiple: {settlement['r_multiple']:+.2f}\n"
            f"P&L: ${pnl_usd:+.2f}",
        )

    # ── trade overlay drawing ──────────────────────────────────────────────────

    def _clear_trade_items(self) -> None:
        for item in self._trade_items:
            self._plot_c.removeItem(item)
        self._trade_items.clear()

    def _draw_pending_order_marker(self) -> None:
        """Dashed line at the limit price for an order that hasn't filled yet."""
        self._clear_trade_items()
        order = self._pending_order
        if order is None:
            return
        line = pg.InfiniteLine(
            pos=order["limit_price"], angle=0, movable=False,
            pen=pg.mkPen(_GOLD, width=1.5, style=Qt.PenStyle.DashDotLine),
            label=f"LIMIT {order['limit_price']:.2f}", labelOpts={"color": _GOLD, "position": 0.05},
        )
        self._plot_c.addItem(line)
        self._trade_items.append(line)

    def _draw_open_trade_marker(self) -> None:
        """Entry arrow + SL/TP lines for the currently-open (not yet settled) trade."""
        self._clear_trade_items()
        trade = self._open_trade
        if trade is None:
            return
        is_bull = trade["direction"] == "bull"
        arrow_col = _GREEN if is_bull else _RED
        arr = pg.ArrowItem(
            pos=(trade["entry_idx"], trade["entry_price"]),
            angle=90 if is_bull else -90, headLen=14, tipAngle=30,
            brush=_qc(arrow_col), pen=pg.mkPen(arrow_col, width=1.5),
        )
        self._plot_c.addItem(arr)
        self._trade_items.append(arr)

        if trade["exit_mode"] == "fixed":
            for price, color, tag in ((trade["sl"], _RED, "SL"), (trade["tp"], _GREEN, "TP")):
                line = pg.InfiniteLine(
                    pos=price, angle=0, movable=False,
                    pen=pg.mkPen(color, width=1, style=Qt.PenStyle.DashLine),
                    label=f"{tag} {price:.2f}", labelOpts={"color": color, "position": 0.05},
                )
                self._plot_c.addItem(line)
                self._trade_items.append(line)

    def _draw_trade_outcome(self) -> None:
        """Full outcome overlay for a just-settled trade: entry arrow, SL/TP or
        chandelier stop-series line, and a win/loss exit marker. Mirrors
        trade_viewer_qt._draw_trade_review()'s visual pattern."""
        self._clear_trade_items()
        trade = self._settled_trade
        if trade is None:
            return
        is_bull   = trade["direction"] == "bull"
        arrow_col = _GREEN if is_bull else _RED
        entry_idx, exit_idx = trade["entry_idx"], trade["exit_bar"]

        arr = pg.ArrowItem(
            pos=(entry_idx, trade["entry_price"]),
            angle=90 if is_bull else -90, headLen=14, tipAngle=30,
            brush=_qc(arrow_col), pen=pg.mkPen(arrow_col, width=1.5),
        )
        self._plot_c.addItem(arr)
        self._trade_items.append(arr)

        if trade["exit_mode"] == "fixed":
            for price, color, tag in ((trade["sl"], _RED, "SL"), (trade["tp"], _GREEN, "TP")):
                line = pg.InfiniteLine(
                    pos=price, angle=0, movable=False,
                    pen=pg.mkPen(color, width=1, style=Qt.PenStyle.DashLine),
                    label=f"{tag} {price:.2f}", labelOpts={"color": color, "position": 0.05},
                )
                self._plot_c.addItem(line)
                self._trade_items.append(line)
        elif "stop_series" in trade:
            stop_series = trade["stop_series"]
            xs = np.arange(entry_idx, entry_idx + len(stop_series))
            curve = pg.PlotCurveItem(
                x=xs, y=stop_series, pen=pg.mkPen("#ffa726", width=1.5, style=Qt.PenStyle.DashLine),
            )
            self._plot_c.addItem(curve)
            self._trade_items.append(curve)

        is_win  = trade["result"] == "win"
        exc_col = _GREEN if is_win else _RED
        sym     = "o" if is_win else "x"
        pt = pg.ScatterPlotItem(
            x=[exit_idx], y=[trade["exit_price"]], symbol=sym, size=12,
            brush=_qc(exc_col, 200), pen=pg.mkPen(exc_col),
        )
        self._plot_c.addItem(pt)
        self._trade_items.append(pt)
        lbl = pg.TextItem(
            text=f"{'✓' if is_win else '✕'} {trade['exit_price']:.2f}  R={trade['r_multiple']:+.2f}",
            color=exc_col, anchor=(0.0, 0.5),
        )
        lbl.setFont(QFont("Monospace", 7))
        lbl.setPos(exit_idx + 0.5, trade["exit_price"])
        self._plot_c.addItem(lbl)
        self._trade_items.append(lbl)

    # ── new round / session stats ──────────────────────────────────────────────

    def _clear_open_trade_state(self) -> None:
        self._play_timer.stop()
        self._play_btn.setChecked(False)
        self._open_trade    = None
        self._pending_order = None
        self._settled_trade = None
        self._clear_trade_items()
        self._update_order_panel_enabled("idle")
        self._trade_status_lbl.setText("No open trade")

    def _on_new_round(self) -> None:
        self._clear_open_trade_state()
        self._render()

    def _current_balance(self) -> float:
        """Starting capital + all-time P&L across every stored trade (the DB
        IS the cross-session history, so this is a running "career" balance,
        not reset per app restart)."""
        return self._starting_capital_spin.value() + self._db.session_stats()["total_pnl_usd"]

    def _refresh_session_stats(self) -> None:
        stats   = self._db.session_stats()
        balance = self._current_balance()
        self._stats_lbl.setText(
            f"Trades: {stats['n_trades']}\n"
            f"Win rate: {stats['win_rate']*100:.1f}%\n"
            f"Total R: {stats['total_r']:+.2f}\n"
            f"Total P&L: ${stats['total_pnl_usd']:+.2f}\n"
            f"Balance: ${balance:,.2f}"
        )


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="K-line Replay Trainer")
    p.parse_args(argv)
    app = QApplication(sys.argv[:1])
    win = ReplayTrainerWindow()
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
