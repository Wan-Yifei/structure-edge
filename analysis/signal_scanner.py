"""SMC Signal Scanner — standalone PyQt6 application.

Monitors a configurable watchlist, detects SMC entry setups using the same
strategy logic as the backtest engine, and fires alerts when a signal appears.
Signals are persisted to db/signals.db (SQLite WAL) so the trade viewer can
overlay them independently without any shared process state.

Usage:
    uv run main.py scanner
"""

from __future__ import annotations

import json
import pathlib
import sys
import uuid
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from PyQt6.QtCore import QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from backtest.engine import ALGO_VERSION, BacktestParams
from strategy.smc.market_structure import detect_bos_choch, determine_trend, find_swings
from strategy.smc.fvg import detect_fvg

_ROOT = pathlib.Path(__file__).parent.parent
_CFG_PATH = _ROOT / "config" / "schedule.json"
_SIGNALS_DB_PATH = _ROOT / "db" / "signals.db"

# ── tf mapping (fetcher uses "60m" key, viewer uses "1h") ────────────────────
_TF_ALIAS: dict[str, str] = {"1h": "60m", "60m": "60m", "4h": "240m"}

def _fetcher_tf(tf: str) -> str:
    """Convert viewer-style TF key to fetcher-compatible key."""
    return _TF_ALIAS.get(tf, tf)


# ── pure strategy logic ───────────────────────────────────────────────────────

class SignalDetector:
    """Pure (no Qt) SMC signal detector — directly testable."""

    @staticmethod
    def detect(
        symbol: str,
        htf: pd.DataFrame,
        params: BacktestParams,
    ) -> list[dict]:
        """Detect SMC entry signals from HTF klines.

        Returns a list of 0 or 1 signal dicts per call.  Each dict contains
        all fields required by SignalsDB.insert_signal().

        Args:
            symbol: moomoo stock code, e.g. 'US.AAPL'
            htf:    HTF klines DataFrame (time_key, open, high, low, close, volume)
            params: BacktestParams controlling strategy behaviour
        """
        if htf is None or len(htf) < max(params.htf_window_bars, 20):
            return []

        window = htf.iloc[-params.htf_window_bars:].reset_index(drop=True)

        # Trend detection
        signals_raw = detect_bos_choch(
            window,
            lookback=params.swing_lookback,
            trend_window=params.htf_window_bars,
            filter_choch=False,
        )
        trend = determine_trend(signals_raw)
        if trend is None:
            return []
        if trend == "bear" and not params.allow_short:
            return []

        # FVG detection
        fvgs = detect_fvg(
            window,
            min_width_pct=params.fvg_min_width_pct,
            require_displacement=params.displacement_required,
        )
        unfilled = [g for g in fvgs if not g.get("filled", False)]
        if not unfilled:
            return []

        bar_cls = float(window.iloc[-1]["close"])

        # Find the most recent actionable FVG in the trend direction
        target_fvg = None
        for g in reversed(unfilled):
            top    = float(g["top"])
            bottom = float(g["bottom"])
            mid    = (top + bottom) / 2.0
            if trend == "bull" and mid < bar_cls:
                target_fvg = g
                break
            if trend == "bear" and mid > bar_cls:
                target_fvg = g
                break
        if target_fvg is None:
            return []

        top    = float(target_fvg["top"])
        bottom = float(target_fvg["bottom"])
        mid    = (top + bottom) / 2.0

        # Optional entry-bar trend filter
        if params.require_ltf_trend_bar:
            last = window.iloc[-1]
            bar_bull = float(last["close"]) >= float(last["open"])
            if (trend == "bull" and not bar_bull) or (trend == "bear" and bar_bull):
                return []

        # SL / TP using swing structure
        swings   = find_swings(window, params.swing_lookback)
        lows_sw  = [s for s in swings if s["kind"] == "low"]
        highs_sw = [s for s in swings if s["kind"] == "high"]

        sl_price: Optional[float] = None
        tp_price: Optional[float] = None

        if trend == "bull":
            sl_cands = [s for s in lows_sw  if s["price"] < mid]
            tp_cands = [s for s in highs_sw if s["price"] > mid]
            if sl_cands:
                sl_price = sl_cands[-1]["price"] * (1.0 - params.sl_buffer_pct)
            if tp_cands:
                tp_price = min(s["price"] for s in tp_cands)
        else:
            sl_cands = [s for s in highs_sw if s["price"] > mid]
            tp_cands = [s for s in lows_sw  if s["price"] < mid]
            if sl_cands:
                sl_price = sl_cands[-1]["price"] * (1.0 + params.sl_buffer_pct)
            if tp_cands:
                tp_price = max(s["price"] for s in tp_cands)

        if sl_price is None or tp_price is None:
            return []

        sl_dist = abs(mid - sl_price)
        tp_dist = abs(tp_price - mid)
        if sl_dist <= 0:
            return []

        # Max SL % guard
        if sl_dist / mid > params.max_sl_pct:
            return []

        rr = tp_dist / sl_dist
        if rr < params.min_rr:
            return []

        # BOS price from the most recent signal
        bos_price = None
        if signals_raw:
            bos_price = float(signals_raw[-1].get("price", 0)) or None

        signal_time = str(window.iloc[-1]["time_key"])[:19]

        return [{
            "symbol":            symbol,
            "direction":         trend,
            "signal_time":       signal_time,
            "trend_tf":          params.trend_tf,
            "entry_tf":          params.entry_tf,
            "entry_zone_top":    top,
            "entry_zone_bottom": bottom,
            "sl_price":          sl_price,
            "tp_price":          tp_price,
            "rr_ratio":          round(rr, 2),
            "bos_price":         bos_price,
            "strategy":          "smc",
            "params_json":       json.dumps(asdict(params)),
            "algo_version":      ALGO_VERSION,
            "source":            "auto",
            "status":            "open",
            "created_at":        datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        }]


# ── background scan worker ────────────────────────────────────────────────────

class ScanWorker(QThread):
    """Scans all enabled symbols in a background thread.

    Emits:
        log(str)             — status / error messages
        new_signal(dict)     — a freshly detected signal dict
        status_update(symbol, status_str)  — per-symbol last-scan status
    """

    log           = pyqtSignal(str)
    new_signal    = pyqtSignal(dict)
    status_update = pyqtSignal(str, str)

    def __init__(self, cfg: dict, parent=None) -> None:
        super().__init__(parent)
        self._cfg     = cfg
        self._running = False
        self._bar_cache: dict[str, str] = {}  # symbol → last bar time_key

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        self._running = True
        scanner_cfg = self._cfg.get("scanner", {})
        interval_s  = int(scanner_cfg.get("scan_interval_s", 60))
        enabled     = scanner_cfg.get("enabled", [])

        self.log.emit(f"Scan worker started — {len(enabled)} symbol(s), interval {interval_s}s")

        while self._running:
            for symbol_cfg in enabled:
                if not self._running:
                    break
                symbol = symbol_cfg if isinstance(symbol_cfg, str) else symbol_cfg.get("symbol", "")
                if not symbol:
                    continue
                try:
                    self._scan_symbol(symbol, symbol_cfg if isinstance(symbol_cfg, dict) else {}, scanner_cfg)
                except Exception as exc:
                    self.log.emit(f"[{symbol}] scan error: {exc}")
                    self.status_update.emit(symbol, f"ERR: {exc}")

            if self._running:
                self.msleep(interval_s * 1000)

        self.log.emit("Scan worker stopped.")

    def _scan_symbol(self, symbol: str, sym_cfg: dict, scanner_cfg: dict) -> None:
        from feeds.fetcher import fetch_klines
        from moomoo import OpenQuoteContext, RET_OK

        overrides   = scanner_cfg.get("overrides", {}).get(symbol, {})
        auto_params = overrides.get("auto_params", scanner_cfg.get("default", {}).get("auto_params", True))

        params = self._resolve_params(symbol, sym_cfg, scanner_cfg, auto_params)
        if params is None:
            self.status_update.emit(symbol, "no params")
            return

        tf      = _fetcher_tf(params.trend_tf)
        end_dt  = datetime.now()
        start_dt = end_dt - timedelta(days=10)
        start   = start_dt.strftime("%Y-%m-%d")
        end     = end_dt.strftime("%Y-%m-%d")

        htf = fetch_klines(symbol, tf, start, end)
        if htf is None or htf.empty:
            self.status_update.emit(symbol, "no data")
            return

        last_bar = str(htf.iloc[-1]["time_key"])
        cache_key = f"{symbol}_{tf}"
        if self._bar_cache.get(cache_key) == last_bar:
            self.status_update.emit(symbol, f"no new bar ({last_bar[:16]})")
            return
        self._bar_cache[cache_key] = last_bar

        new_sigs = SignalDetector.detect(symbol, htf, params)

        # Deduplication against open signals in DB
        if new_sigs:
            try:
                from db.signals import SignalsDB
                with SignalsDB(_SIGNALS_DB_PATH) as db:
                    open_sigs = db.get_open_signals(symbol)
                    open_fvg_keys = {
                        (round(s["entry_zone_bottom"], 4), round(s["entry_zone_top"], 4))
                        for s in open_sigs
                    }
                    for sig in new_sigs:
                        fvg_key = (round(sig["entry_zone_bottom"], 4), round(sig["entry_zone_top"], 4))
                        if fvg_key in open_fvg_keys:
                            continue
                        sig["signal_id"] = str(uuid.uuid4())
                        db.insert_signal(sig)
                        self.new_signal.emit(sig)
                        self.log.emit(
                            f"[{symbol}] {sig['direction'].upper()} signal "
                            f"zone {sig['entry_zone_bottom']:.2f}–{sig['entry_zone_top']:.2f} "
                            f"RR {sig['rr_ratio']:.1f}"
                        )
                        self._alert(sig, scanner_cfg)
            except Exception as exc:
                self.log.emit(f"[{symbol}] DB write error: {exc}")

        ts = datetime.now().strftime("%H:%M:%S")
        self.status_update.emit(symbol, f"OK {ts} ({len(new_sigs)} signal(s))")

    def _resolve_params(
        self, symbol: str, sym_cfg: dict, scanner_cfg: dict, auto_params: bool
    ) -> Optional[BacktestParams]:
        default = scanner_cfg.get("default", {})
        overrides = scanner_cfg.get("overrides", {}).get(symbol, {})

        if auto_params:
            try:
                from backtest.db import BacktestDB
                with BacktestDB(read_only=True) as bdb:
                    best = bdb.get_best_params(
                        symbol,
                        lookback_months=int(overrides.get("lookback_months", default.get("lookback_months", 3))),
                        min_n_trades=int(overrides.get("min_n_trades", default.get("min_n_trades", 5))),
                        min_pf=float(overrides.get("min_pf", default.get("min_pf", 1.5))),
                    )
                if best and best.get("config_json"):
                    cfg = json.loads(best["config_json"])
                    return BacktestParams(**{k: v for k, v in cfg.items() if hasattr(BacktestParams, k) or k in BacktestParams.__dataclass_fields__})
            except Exception:
                pass

        # Manual / default params
        p = dict(default)
        p.update(overrides.get("params", {}))
        try:
            fields = BacktestParams.__dataclass_fields__
            return BacktestParams(**{k: v for k, v in p.items() if k in fields})
        except Exception:
            return BacktestParams()

    def _alert(self, sig: dict, scanner_cfg: dict) -> None:
        if scanner_cfg.get("alert_sound", True):
            QApplication.beep()


# ── params dialog ─────────────────────────────────────────────────────────────

class ParamsDialog(QDialog):
    """Edit scanner params for a specific symbol (or default)."""

    def __init__(self, symbol: str, cfg: dict, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Params — {symbol}")
        self.setMinimumWidth(400)
        self._symbol = symbol
        self._cfg    = cfg

        scanner = cfg.get("scanner", {})
        default = scanner.get("default", {})
        override = scanner.get("overrides", {}).get(symbol, {})

        layout = QVBoxLayout(self)

        # Mode
        mode_box = QGroupBox("Parameter source")
        mode_lay = QHBoxLayout(mode_box)
        self._auto_rb   = QRadioButton("Auto (from backtest DB)")
        self._manual_rb = QRadioButton("Manual")
        mode_lay.addWidget(self._auto_rb)
        mode_lay.addWidget(self._manual_rb)
        layout.addWidget(mode_box)

        # Auto section
        auto_box = QGroupBox("Auto params settings")
        auto_form = QFormLayout(auto_box)
        self._lookback_sp = QSpinBox()
        self._lookback_sp.setRange(1, 24)
        self._lookback_sp.setValue(int(override.get("lookback_months", default.get("lookback_months", 3))))
        self._min_trades_sp = QSpinBox()
        self._min_trades_sp.setRange(1, 200)
        self._min_trades_sp.setValue(int(override.get("min_n_trades", default.get("min_n_trades", 5))))
        self._min_pf_sp = QDoubleSpinBox()
        self._min_pf_sp.setRange(0.5, 10.0)
        self._min_pf_sp.setSingleStep(0.1)
        self._min_pf_sp.setValue(float(override.get("min_pf", default.get("min_pf", 1.5))))
        self._preview_lbl = QLabel("—")
        preview_btn = QPushButton("Preview match")
        preview_btn.clicked.connect(self._preview_match)
        auto_form.addRow("Lookback months:", self._lookback_sp)
        auto_form.addRow("Min trades:", self._min_trades_sp)
        auto_form.addRow("Min PF:", self._min_pf_sp)
        auto_form.addRow("", preview_btn)
        auto_form.addRow("Match:", self._preview_lbl)
        layout.addWidget(auto_box)

        # Manual section
        manual_box = QGroupBox("Manual params")
        manual_form = QFormLayout(manual_box)
        p_override = override.get("params", {})
        def _pval(key, default_val):
            return p_override.get(key, default.get(key, default_val))

        self._trend_tf_cb = QComboBox()
        self._trend_tf_cb.addItems(["1h", "4h", "1d", "15m", "30m"])
        self._trend_tf_cb.setCurrentText(_pval("trend_tf", "1h"))
        self._entry_tf_cb = QComboBox()
        self._entry_tf_cb.addItems(["15m", "5m", "30m", "1m"])
        self._entry_tf_cb.setCurrentText(_pval("entry_tf", "15m"))
        self._allow_short_cb  = QCheckBox()
        self._allow_short_cb.setChecked(bool(_pval("allow_short", True)))
        self._disp_cb = QCheckBox()
        self._disp_cb.setChecked(bool(_pval("displacement_required", False)))
        self._min_rr_sp = QDoubleSpinBox()
        self._min_rr_sp.setRange(0.5, 10.0)
        self._min_rr_sp.setSingleStep(0.1)
        self._min_rr_sp.setValue(float(_pval("min_rr", 2.0)))
        self._sl_buf_sp = QDoubleSpinBox()
        self._sl_buf_sp.setRange(0.0, 0.02)
        self._sl_buf_sp.setSingleStep(0.001)
        self._sl_buf_sp.setDecimals(4)
        self._sl_buf_sp.setValue(float(_pval("sl_buffer_pct", 0.001)))
        self._max_sl_sp = QDoubleSpinBox()
        self._max_sl_sp.setRange(0.001, 0.05)
        self._max_sl_sp.setSingleStep(0.001)
        self._max_sl_sp.setDecimals(4)
        self._max_sl_sp.setValue(float(_pval("max_sl_pct", 0.005)))
        self._htf_window_sp = QSpinBox()
        self._htf_window_sp.setRange(10, 200)
        self._htf_window_sp.setValue(int(_pval("htf_window_bars", 20)))
        manual_form.addRow("Trend TF:", self._trend_tf_cb)
        manual_form.addRow("Entry TF:", self._entry_tf_cb)
        manual_form.addRow("Allow short:", self._allow_short_cb)
        manual_form.addRow("Displacement required:", self._disp_cb)
        manual_form.addRow("Min RR:", self._min_rr_sp)
        manual_form.addRow("SL buffer %:", self._sl_buf_sp)
        manual_form.addRow("Max SL %:", self._max_sl_sp)
        manual_form.addRow("HTF window bars:", self._htf_window_sp)
        layout.addWidget(manual_box)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Wire mode toggle
        auto = bool(override.get("auto_params", default.get("auto_params", True)))
        self._auto_rb.setChecked(auto)
        self._manual_rb.setChecked(not auto)
        self._auto_rb.toggled.connect(lambda on: (auto_box.setEnabled(on), manual_box.setEnabled(not on)))
        auto_box.setEnabled(auto)
        manual_box.setEnabled(not auto)

    def _preview_match(self) -> None:
        try:
            from backtest.db import BacktestDB
            with BacktestDB(read_only=True) as db:
                best = db.get_best_params(
                    self._symbol,
                    lookback_months=self._lookback_sp.value(),
                    min_n_trades=self._min_trades_sp.value(),
                    min_pf=self._min_pf_sp.value(),
                )
            if best:
                meta = best["_meta"]
                self._preview_lbl.setText(
                    f"{best['trend_tf']}/{best['entry_tf']} "
                    f"PF {meta['pf']:.2f}  n={meta['n_trades']}"
                )
            else:
                self._preview_lbl.setText("No match found")
        except Exception as exc:
            self._preview_lbl.setText(f"Error: {exc}")

    def result_cfg(self) -> dict:
        """Return the updated override dict for this symbol."""
        auto = self._auto_rb.isChecked()
        out: dict = {"auto_params": auto}
        if auto:
            out["lookback_months"] = self._lookback_sp.value()
            out["min_n_trades"]    = self._min_trades_sp.value()
            out["min_pf"]          = self._min_pf_sp.value()
        else:
            out["params"] = {
                "trend_tf":             self._trend_tf_cb.currentText(),
                "entry_tf":             self._entry_tf_cb.currentText(),
                "allow_short":          self._allow_short_cb.isChecked(),
                "displacement_required": self._disp_cb.isChecked(),
                "min_rr":               self._min_rr_sp.value(),
                "sl_buffer_pct":        self._sl_buf_sp.value(),
                "max_sl_pct":           self._max_sl_sp.value(),
                "htf_window_bars":      self._htf_window_sp.value(),
            }
        return out


# ── main window ───────────────────────────────────────────────────────────────

class SignalScanner(QMainWindow):
    """Signal scanner main window."""

    def __init__(self, args=None) -> None:
        super().__init__()
        self.setWindowTitle("SMC Signal Scanner")
        self.resize(1000, 700)

        self._cfg    = self._load_cfg()
        self._worker: Optional[ScanWorker] = None

        self._build_ui()
        self._refresh_targets_table()
        self._refresh_signals_table()

    # ── config ────────────────────────────────────────────────────────────────

    def _load_cfg(self) -> dict:
        if _CFG_PATH.exists():
            try:
                return json.loads(_CFG_PATH.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _save_cfg(self) -> None:
        try:
            _CFG_PATH.parent.mkdir(parents=True, exist_ok=True)
            _CFG_PATH.write_text(
                json.dumps(self._cfg, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            self._log(f"Config save error: {exc}")

    def _scanner_cfg(self) -> dict:
        if "scanner" not in self._cfg:
            self._cfg["scanner"] = {
                "enabled": [],
                "scan_interval_s": 60,
                "alert_sound": True,
                "default": {
                    "strategy": "smc",
                    "auto_params": True,
                    "lookback_months": 3,
                    "min_n_trades": 5,
                    "min_pf": 1.5,
                },
                "overrides": {},
            }
        return self._cfg["scanner"]

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_lay = QVBoxLayout(central)
        main_lay.setContentsMargins(4, 4, 4, 4)

        # Toolbar
        tb = QToolBar("Main")
        self.addToolBar(tb)

        self._connect_btn = QPushButton("Connect")
        self._connect_btn.setCheckable(True)
        self._connect_btn.clicked.connect(self._on_connect_toggle)
        tb.addWidget(self._connect_btn)

        self._scan_btn = QPushButton("▶ Scan")
        self._scan_btn.setCheckable(True)
        self._scan_btn.setEnabled(False)
        self._scan_btn.clicked.connect(self._on_scan_toggle)
        tb.addWidget(self._scan_btn)

        tb.addSeparator()
        tb.addWidget(QLabel(" Interval (s): "))
        self._interval_sp = QSpinBox()
        self._interval_sp.setRange(10, 3600)
        self._interval_sp.setValue(int(self._scanner_cfg().get("scan_interval_s", 60)))
        self._interval_sp.valueChanged.connect(self._on_interval_changed)
        tb.addWidget(self._interval_sp)

        tb.addSeparator()
        self._sound_cb = QCheckBox("Sound")
        self._sound_cb.setChecked(bool(self._scanner_cfg().get("alert_sound", True)))
        self._sound_cb.stateChanged.connect(self._on_sound_changed)
        tb.addWidget(self._sound_cb)

        tb.addSeparator()
        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self._on_add)
        tb.addWidget(add_btn)

        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(self._on_remove)
        tb.addWidget(remove_btn)

        edit_btn = QPushButton("Edit Params")
        edit_btn.clicked.connect(self._on_edit_params)
        tb.addWidget(edit_btn)

        # Splitter: targets table (top) + signals table (bottom)
        splitter = QSplitter()
        splitter.setOrientation(splitter.orientation())
        from PyQt6.QtCore import Qt as _Qt
        splitter.setOrientation(_Qt.Orientation.Vertical)
        main_lay.addWidget(splitter)

        # Targets table
        targets_widget = QWidget()
        targets_lay = QVBoxLayout(targets_widget)
        targets_lay.setContentsMargins(0, 0, 0, 0)
        targets_lay.addWidget(QLabel("Targets"))
        self._targets_tbl = QTableWidget(0, 5)
        self._targets_tbl.setHorizontalHeaderLabels(
            ["Symbol", "Mode", "Lookback", "TFs", "Last scan"]
        )
        self._targets_tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._targets_tbl.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._targets_tbl.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        targets_lay.addWidget(self._targets_tbl)
        splitter.addWidget(targets_widget)

        # Signals table
        signals_widget = QWidget()
        signals_lay = QVBoxLayout(signals_widget)
        signals_lay.setContentsMargins(0, 0, 0, 0)
        signals_lay.addWidget(QLabel("Recent signals (last 50)"))
        self._signals_tbl = QTableWidget(0, 8)
        self._signals_tbl.setHorizontalHeaderLabels(
            ["Time", "Symbol", "Dir", "Entry zone", "SL", "TP", "RR", "Status"]
        )
        self._signals_tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._signals_tbl.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        signals_lay.addWidget(self._signals_tbl)
        splitter.addWidget(signals_widget)

        splitter.setSizes([300, 350])

        # Log area
        self._log_edit = QTextEdit()
        self._log_edit.setReadOnly(True)
        self._log_edit.setMaximumHeight(120)
        self._log_edit.setFont(QFont("Consolas", 8))
        main_lay.addWidget(self._log_edit)

        # Status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Ready")

    # ── toolbar actions ───────────────────────────────────────────────────────

    def _on_connect_toggle(self, checked: bool) -> None:
        self._connect_btn.setText("Disconnect" if checked else "Connect")
        self._scan_btn.setEnabled(checked)
        self._log("Connected." if checked else "Disconnected.")

    def _on_scan_toggle(self, checked: bool) -> None:
        if checked:
            self._start_scan()
        else:
            self._stop_scan()

    def _start_scan(self) -> None:
        cfg = self._load_cfg()
        self._worker = ScanWorker(cfg)
        self._worker.log.connect(self._log)
        self._worker.new_signal.connect(self._on_new_signal)
        self._worker.status_update.connect(self._on_status_update)
        self._worker.start()
        self._scan_btn.setText("⏹ Stop")
        self._log("Scan started.")

    def _stop_scan(self) -> None:
        if self._worker:
            self._worker.stop()
            self._worker.wait(3000)
            self._worker = None
        self._scan_btn.setText("▶ Scan")
        self._log("Scan stopped.")

    def _on_interval_changed(self, value: int) -> None:
        self._scanner_cfg()["scan_interval_s"] = value
        self._save_cfg()

    def _on_sound_changed(self, state: int) -> None:
        self._scanner_cfg()["alert_sound"] = bool(state)
        self._save_cfg()

    def _on_add(self) -> None:
        from PyQt6.QtWidgets import QInputDialog
        symbol, ok = QInputDialog.getText(self, "Add symbol", "Symbol (e.g. US.AAPL):")
        if not ok or not symbol.strip():
            return
        symbol = symbol.strip().upper()
        enabled = self._scanner_cfg().setdefault("enabled", [])
        if symbol not in enabled:
            enabled.append(symbol)
            self._save_cfg()
            self._refresh_targets_table()

    def _on_remove(self) -> None:
        rows = {i.row() for i in self._targets_tbl.selectedItems()}
        if not rows:
            return
        enabled = self._scanner_cfg().get("enabled", [])
        to_remove = {
            self._targets_tbl.item(r, 0).text()
            for r in rows
            if self._targets_tbl.item(r, 0)
        }
        self._scanner_cfg()["enabled"] = [s for s in enabled if s not in to_remove]
        self._save_cfg()
        self._refresh_targets_table()

    def _on_edit_params(self) -> None:
        rows = {i.row() for i in self._targets_tbl.selectedItems()}
        symbol = (
            self._targets_tbl.item(list(rows)[0], 0).text()
            if rows and self._targets_tbl.item(list(rows)[0], 0)
            else "__default__"
        )
        dlg = ParamsDialog(symbol, self._cfg, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            result = dlg.result_cfg()
            overrides = self._scanner_cfg().setdefault("overrides", {})
            overrides[symbol] = result
            self._save_cfg()
            self._refresh_targets_table()

    # ── table refresh ─────────────────────────────────────────────────────────

    def _refresh_targets_table(self) -> None:
        scanner = self._scanner_cfg()
        enabled  = scanner.get("enabled", [])
        overrides = scanner.get("overrides", {})
        default  = scanner.get("default", {})

        self._targets_tbl.setRowCount(len(enabled))
        for row, symbol in enumerate(enabled):
            ov = overrides.get(symbol, {})
            auto = ov.get("auto_params", default.get("auto_params", True))
            mode = "Auto" if auto else "Manual"
            lb   = ov.get("lookback_months", default.get("lookback_months", 3))
            p    = ov.get("params", {})
            tfs  = f"{p.get('trend_tf', default.get('trend_tf', '1h'))}/{p.get('entry_tf', default.get('entry_tf', '15m'))}"

            self._targets_tbl.setItem(row, 0, QTableWidgetItem(symbol))
            self._targets_tbl.setItem(row, 1, QTableWidgetItem(mode))
            self._targets_tbl.setItem(row, 2, QTableWidgetItem(str(lb) + " mo"))
            self._targets_tbl.setItem(row, 3, QTableWidgetItem(tfs if not auto else "—"))
            self._targets_tbl.setItem(row, 4, QTableWidgetItem("—"))

    def _refresh_signals_table(self) -> None:
        try:
            from db.signals import SignalsDB
            since = (datetime.utcnow() - timedelta(days=5)).strftime("%Y-%m-%d %H:%M:%S")
            with SignalsDB(_SIGNALS_DB_PATH, read_only=False) as db:
                sigs = db.query_signals("", since)  # empty symbol = all (not supported)
                # Fallback: get all open signals
                sigs = db.get_all_open_signals()
        except Exception:
            sigs = []
        self._populate_signals_table(sigs[:50])

    def _populate_signals_table(self, sigs: list[dict]) -> None:
        self._signals_tbl.setRowCount(len(sigs))
        for row, sig in enumerate(sigs):
            direction = sig.get("direction", "")
            color = QColor("#26a69a") if direction == "bull" else QColor("#ef5350")
            items = [
                sig.get("signal_time", "")[:16],
                sig.get("symbol", ""),
                direction.upper(),
                f"{sig.get('entry_zone_bottom', 0):.2f}–{sig.get('entry_zone_top', 0):.2f}",
                f"{sig.get('sl_price', 0):.2f}",
                f"{sig.get('tp_price', 0):.2f}",
                f"{sig.get('rr_ratio', 0):.1f}",
                sig.get("status", ""),
            ]
            for col, text in enumerate(items):
                item = QTableWidgetItem(text)
                if col == 2:  # direction column
                    item.setForeground(color)
                self._signals_tbl.setItem(row, col, item)

    # ── worker callbacks ──────────────────────────────────────────────────────

    def _on_new_signal(self, sig: dict) -> None:
        self._refresh_signals_table()
        self._status_bar.showMessage(
            f"New signal: {sig['symbol']} {sig['direction'].upper()} "
            f"RR {sig['rr_ratio']:.1f}  ({sig['signal_time'][:16]})"
        )

    def _on_status_update(self, symbol: str, status: str) -> None:
        for row in range(self._targets_tbl.rowCount()):
            item = self._targets_tbl.item(row, 0)
            if item and item.text() == symbol:
                self._targets_tbl.setItem(row, 4, QTableWidgetItem(status))
                break

    def _log(self, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self._log_edit.append(f"{ts}  {msg}")

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        self._stop_scan()
        super().closeEvent(event)


# ── entry point ───────────────────────────────────────────────────────────────

def main(argv=None) -> None:
    app = QApplication(sys.argv if argv is None else [sys.argv[0]] + list(argv))
    app.setApplicationName("SMC Signal Scanner")
    win = SignalScanner()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
