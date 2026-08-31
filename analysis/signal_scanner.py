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
import subprocess
import sys
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QIcon, QPixmap
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
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QSystemTrayIcon,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
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
_INDICATOR_ALERT_CFG_PATH = _ROOT / "config" / "scanner" / "indicator_alert_params.json"
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
            min_gap_pct=params.fvg_min_width_pct,
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
        new_signal(dict)     — a freshly detected entry signal dict
        new_fvg_watch_signal(dict) — a freshly detected lightweight FVG-formed signal
        new_indicator_alert(dict) — a rule's refreshed reading (emitted every cycle
                                     the rule is scanned, positive or not -- the UI
                                     decides whether that's worth a notification)
        status_update(symbol, status_str)  — per-symbol last-scan status
    """

    log                  = pyqtSignal(str)
    new_signal           = pyqtSignal(dict)
    new_fvg_watch_signal = pyqtSignal(dict)
    new_indicator_alert  = pyqtSignal(dict)
    status_update        = pyqtSignal(str, str)

    def __init__(self, cfg: dict, parent=None) -> None:
        super().__init__(parent)
        self._cfg     = cfg
        self._running = False
        self._bar_cache: dict[str, str] = {}  # symbol → last bar time_key
        self._indicator_alert_last_notify: dict[str, float] = {}  # rule_id → time.time() of last beep/tray notify

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        self._running = True
        scanner_cfg = self._cfg.get("scanner", {})
        interval_s  = int(scanner_cfg.get("scan_interval_s", 60))
        enabled     = scanner_cfg.get("enabled", [])

        self.log.emit(f"Scan worker started — {len(enabled)} symbol(s), interval {interval_s}s")

        cycle = 0
        while self._running:
            cycle += 1
            ts = datetime.now().strftime("%H:%M:%S")
            self.log.emit(f"[{ts}] Cycle #{cycle} — scanning {len(enabled)} symbol(s)")
            for symbol_cfg in enabled:
                if not self._running:
                    break
                symbol = symbol_cfg if isinstance(symbol_cfg, str) else symbol_cfg.get("symbol", "")
                if not symbol:
                    continue
                sym_cfg   = symbol_cfg if isinstance(symbol_cfg, dict) else {}
                default   = scanner_cfg.get("default", {})
                overrides = scanner_cfg.get("overrides", {}).get(symbol, {})
                try:
                    if overrides.get("entry_signal_enabled", default.get("entry_signal_enabled", True)):
                        self._scan_symbol(symbol, sym_cfg, scanner_cfg)
                    if overrides.get("fvg_watch_enabled", default.get("fvg_watch_enabled", False)):
                        self._scan_symbol_fvg_watch(symbol, scanner_cfg)
                    if overrides.get("session_vp_enabled", default.get("session_vp_enabled", False)):
                        self._scan_symbol_session_vp(symbol, scanner_cfg)
                    if overrides.get("vah_soxs_enabled", default.get("vah_soxs_enabled", False)):
                        self._scan_symbol_vah_soxs(symbol, scanner_cfg)
                    if overrides.get("val_kd_tp_enabled", default.get("val_kd_tp_enabled", False)):
                        self._scan_symbol_val_kd_tp(symbol, scanner_cfg)
                    if overrides.get("indicator_alert_enabled", default.get("indicator_alert_enabled", False)):
                        self._scan_symbol_indicator_alert(symbol, scanner_cfg)
                except Exception as exc:
                    import traceback
                    self.log.emit(f"[{symbol}] scan error: {exc}\n{traceback.format_exc()}")
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
        # 3-day window is enough for strategy logic and keeps API payloads small.
        # force_refresh=True bypasses the DuckDB kline cache so the scanner always
        # sees the latest bars from moomoo (cache tolerance of 7 days would otherwise
        # return stale data from the previous trading day indefinitely).
        start_dt = end_dt - timedelta(days=3)
        start   = start_dt.strftime("%Y-%m-%d")
        end     = end_dt.strftime("%Y-%m-%d")

        htf = fetch_klines(symbol, tf, start, end, force_refresh=True)
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
        if not scanner_cfg.get("alert_sound", True):
            return
        # QApplication.beep() is frequently silent on Windows -- it depends on
        # the app having a mapped "beep" sound in the current sound scheme,
        # which many schemes leave unset. winsound.MessageBeep() instead plays
        # a real system sound event (SystemExclamation), audible under the
        # default Windows sound scheme; QApplication.beep() is the fallback for
        # non-Windows platforms.
        try:
            import winsound
            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        except Exception:
            QApplication.beep()

    def _scan_symbol_fvg_watch(self, symbol: str, scanner_cfg: dict) -> None:
        """Lightweight 'FVG formed' alert — no trend/SL/TP/RR, independent of
        _scan_symbol's full entry-signal pipeline. Per-(symbol, tf) params come
        from config/scanner/fvg_watch_params.json, hand-picked from
        backtest/fvg_width_sweep.py output.
        """
        from analysis.fvg_watcher import load_fvg_watch_config, scan_symbol_tf
        from db.signals import SignalsDB

        watch_cfg = load_fvg_watch_config().get(symbol, [])
        if not watch_cfg:
            return

        end_dt   = datetime.now()
        start_dt = end_dt - timedelta(days=3)  # same rolling window as _scan_symbol
        start    = start_dt.strftime("%Y-%m-%d")
        end      = end_dt.strftime("%Y-%m-%d")

        for entry in watch_cfg:
            tf = _fetcher_tf(entry["tf"])
            try:
                hits = scan_symbol_tf(symbol, tf, entry, start, end, force_refresh=True)
            except Exception as exc:
                self.log.emit(f"[{symbol} {tf}] fvg_watch error: {exc}")
                continue

            if not hits:
                continue

            cache_key = f"{symbol}_{tf}_fvgwatch"
            last_formed = self._bar_cache.get(cache_key)
            newest = max(h["formed_time"] for h in hits)
            if newest == last_formed:
                continue
            self._bar_cache[cache_key] = newest

            try:
                with SignalsDB(_SIGNALS_DB_PATH) as db:
                    open_keys = {
                        (round(s["zone_bottom"], 4), round(s["zone_top"], 4))
                        for s in db.get_open_fvg_watch(symbol, tf)
                    }
                    for hit in hits:
                        key = (round(hit["zone_bottom"], 4), round(hit["zone_top"], 4))
                        if key in open_keys:
                            continue
                        hit["signal_id"] = str(uuid.uuid4())
                        db.insert_fvg_watch(hit)
                        self.new_fvg_watch_signal.emit(hit)
                        self.log.emit(
                            f"[{symbol} {tf}] FVG formed {hit['direction'].upper()} "
                            f"zone {hit['zone_bottom']:.2f}-{hit['zone_top']:.2f}"
                        )
                        self._alert(hit, scanner_cfg)
            except Exception as exc:
                self.log.emit(f"[{symbol} {tf}] fvg_watch DB write error: {exc}")

    def _scan_symbol_session_vp(self, symbol: str, scanner_cfg: dict) -> None:
        """Session Value-Area reversal watch -- writes into the same `signals`
        table as _scan_symbol (strategy="session_vp"), since its schema
        (symbol/direction/sl_price/tp_price/rr_ratio/...) is generic enough
        to cover this strategy too -- no new table, no new UI needed, the
        existing signals table view / tray notification path displays it
        automatically. Per-(symbol, session) params come from
        config/scanner/session_vp_params.json, hand-picked from
        backtest/results/session_vp_v1_review/REVIEW.md's out-of-sample
        validation (see that file's "validated" field per entry).
        """
        from analysis.session_vp_watcher import load_session_vp_config, scan_symbol_session_vp
        from db.signals import SignalsDB

        watch_cfg = load_session_vp_config().get(symbol, [])
        if not watch_cfg:
            return

        end_dt   = datetime.now()
        start_dt = end_dt - timedelta(days=3)  # same rolling window as _scan_symbol/_scan_symbol_fvg_watch
        start    = start_dt.strftime("%Y-%m-%d")
        end      = end_dt.strftime("%Y-%m-%d")

        with open(_CFG_PATH, encoding="utf-8") as f:
            schedule_sessions = json.load(f)["sessions"]

        for entry_cfg in watch_cfg:
            session = entry_cfg["session"]
            try:
                sigs = scan_symbol_session_vp(
                    symbol, entry_cfg, start, end, schedule_sessions, force_refresh=True,
                )
            except Exception as exc:
                self.log.emit(f"[{symbol} {session}] session_vp error: {exc}")
                continue

            if not sigs:
                continue

            try:
                with SignalsDB(_SIGNALS_DB_PATH) as db:
                    open_sigs = db.get_open_signals(symbol)
                    open_keys = {round(s["entry_zone_top"], 4) for s in open_sigs}
                    for sig in sigs:
                        if round(sig["entry_zone_top"], 4) in open_keys:
                            continue
                        sig["signal_id"] = str(uuid.uuid4())
                        db.insert_signal(sig)
                        self.new_signal.emit(sig)
                        validated = entry_cfg.get("validated", False)
                        tag = "" if validated else " [UNVALIDATED]"
                        self.log.emit(
                            f"[{symbol} {session}] session_vp signal{tag} "
                            f"entry {sig['entry_zone_top']:.2f} "
                            f"SL {sig['sl_price']:.2f} TP {sig['tp_price']:.2f} "
                            f"RR {sig['rr_ratio']:.2f}"
                        )
                        self._alert(sig, scanner_cfg)
            except Exception as exc:
                self.log.emit(f"[{symbol} {session}] session_vp DB write error: {exc}")

    def _scan_symbol_vah_soxs(self, symbol: str, scanner_cfg: dict) -> None:
        """Cross-asset VAH-short-proxy watch -- the signal comes from another
        symbol's (e.g. SOXL) VAH+RSI-overbought reversal, but the executable
        trade is a LONG on `symbol` here (e.g. SOXS), since the signal
        symbol can't be shorted directly. Writes into the same `signals`
        table (strategy="vah_soxs_proxy"). Per-(symbol, session) params come
        from config/scanner/vah_soxs_params.json -- see
        backtest/results/session_vp_v1_review/REVIEW.md section 8 for the
        full research trail (why this exists, the beta price translation,
        and the ATR noise filter).
        """
        from analysis.vah_soxs_watcher import load_vah_soxs_config, scan_vah_soxs
        from db.signals import SignalsDB

        watch_cfg = load_vah_soxs_config().get(symbol, [])
        if not watch_cfg:
            return

        end_dt   = datetime.now()
        start_dt = end_dt - timedelta(days=3)  # same rolling window as the other watches
        start    = start_dt.strftime("%Y-%m-%d")
        end      = end_dt.strftime("%Y-%m-%d")

        with open(_CFG_PATH, encoding="utf-8") as f:
            schedule_sessions = json.load(f)["sessions"]

        for entry_cfg in watch_cfg:
            session = entry_cfg["session"]
            signal_symbol = entry_cfg["signal_symbol"]
            try:
                sigs = scan_vah_soxs(
                    symbol, entry_cfg, start, end, schedule_sessions, force_refresh=True,
                )
            except Exception as exc:
                self.log.emit(f"[{symbol} {session}] vah_soxs error: {exc}")
                continue

            if not sigs:
                continue

            try:
                with SignalsDB(_SIGNALS_DB_PATH) as db:
                    open_sigs = db.get_open_signals(symbol)
                    open_keys = {round(s["entry_zone_top"], 4) for s in open_sigs}
                    for sig in sigs:
                        if round(sig["entry_zone_top"], 4) in open_keys:
                            continue
                        sig["signal_id"] = str(uuid.uuid4())
                        db.insert_signal(sig)
                        self.new_signal.emit(sig)
                        validated = entry_cfg.get("validated", False)
                        tag = "" if validated else " [UNVALIDATED]"
                        self.log.emit(
                            f"[{symbol} {session}] vah_soxs signal{tag} (from {signal_symbol}) "
                            f"entry {sig['entry_zone_top']:.2f} "
                            f"SL {sig['sl_price']:.2f} TP {sig['tp_price']:.2f} "
                            f"RR {sig['rr_ratio']:.2f}"
                        )
                        self._alert(sig, scanner_cfg)
            except Exception as exc:
                self.log.emit(f"[{symbol} {session}] vah_soxs DB write error: {exc}")

    def _scan_symbol_val_kd_tp(self, symbol: str, scanner_cfg: dict) -> None:
        """VAL-entry / KD-channel-TP watch -- same VAL+RSI-oversold+reversal
        entry as _scan_symbol_session_vp, but TP = up1 (fast EMA-channel
        upper band) instead of the session's frozen POC. Outperforms the
        POC-based TP on the regular session specifically -- see
        backtest/results/session_vp_v1_review/REVIEW.md section 9. Writes
        into the same `signals` table (strategy="val_kd_tp"). Per-(symbol,
        session) params come from config/scanner/val_kd_tp_params.json.
        """
        from analysis.val_kd_tp_watcher import load_val_kd_tp_config, scan_val_kd_tp
        from db.signals import SignalsDB

        watch_cfg = load_val_kd_tp_config().get(symbol, [])
        if not watch_cfg:
            return

        end_dt   = datetime.now()
        start_dt = end_dt - timedelta(days=3)  # same rolling window as the other watches
        start    = start_dt.strftime("%Y-%m-%d")
        end      = end_dt.strftime("%Y-%m-%d")

        with open(_CFG_PATH, encoding="utf-8") as f:
            schedule_sessions = json.load(f)["sessions"]

        for entry_cfg in watch_cfg:
            session = entry_cfg["session"]
            try:
                sigs = scan_val_kd_tp(
                    symbol, entry_cfg, start, end, schedule_sessions, force_refresh=True,
                )
            except Exception as exc:
                self.log.emit(f"[{symbol} {session}] val_kd_tp error: {exc}")
                continue

            if not sigs:
                continue

            try:
                with SignalsDB(_SIGNALS_DB_PATH) as db:
                    open_sigs = db.get_open_signals(symbol)
                    open_keys = {round(s["entry_zone_top"], 4) for s in open_sigs}
                    for sig in sigs:
                        if round(sig["entry_zone_top"], 4) in open_keys:
                            continue
                        sig["signal_id"] = str(uuid.uuid4())
                        db.insert_signal(sig)
                        self.new_signal.emit(sig)
                        validated = entry_cfg.get("validated", False)
                        tag = "" if validated else " [UNVALIDATED]"
                        self.log.emit(
                            f"[{symbol} {session}] val_kd_tp signal{tag} "
                            f"entry {sig['entry_zone_top']:.2f} "
                            f"SL {sig['sl_price']:.2f} TP {sig['tp_price']:.2f} "
                            f"RR {sig['rr_ratio']:.2f}"
                        )
                        self._alert(sig, scanner_cfg)
            except Exception as exc:
                self.log.emit(f"[{symbol} {session}] val_kd_tp DB write error: {exc}")

    def _scan_symbol_indicator_alert(self, symbol: str, scanner_cfg: dict) -> None:
        """Open-ended indicator threshold watch -- a continuous status
        monitor, not an event-stream detector like the other _scan_symbolX
        methods: every rule is upserted every cycle (see
        db.signals.SignalsDB.upsert_indicator_alert_state), positive or not,
        so the UI table always reflects the latest reading. Rules come from
        config/scanner/indicator_alert_params.json; see
        analysis/indicator_alert_watcher.py's INDICATOR_REGISTRY for how a
        new indicator (MACD, classic KDJ, ...) gets added later without
        touching this method, the DB schema, or the UI.
        """
        from analysis.indicator_alert_watcher import load_indicator_alert_config, scan_indicator_alert
        from db.signals import SignalsDB

        rules = load_indicator_alert_config().get(symbol, [])
        if not rules:
            return

        end_dt   = datetime.now()
        start_dt = end_dt - timedelta(days=3)  # same rolling window as the other watches
        start    = start_dt.strftime("%Y-%m-%d")
        end      = end_dt.strftime("%Y-%m-%d")

        for rule in rules:
            try:
                result = scan_indicator_alert(symbol, rule, start, end, force_refresh=True)
            except Exception as exc:
                self.log.emit(f"[{symbol}] indicator_alert error ({rule.get('indicator')}): {exc}")
                continue
            if result is None:
                continue

            try:
                with SignalsDB(_SIGNALS_DB_PATH) as db:
                    db.upsert_indicator_alert_state(result)
                    current = db.get_indicator_alert_state(result["rule_id"])
            except Exception as exc:
                self.log.emit(f"[{symbol}] indicator_alert DB write error: {exc}")
                continue

            is_muted = bool(current["is_muted"]) if current else False

            # A level-triggered rule stays positive for as long as the condition
            # holds, so without this throttle it would beep/notify every single
            # scan cycle (as often as every scan_interval_s) for as long as
            # it's positive -- indicator_alert_repeat_s lets the user space
            # repeat notifications out (default: same as scan_interval_s, i.e.
            # no throttle beyond the scan cadence itself).
            repeat_s  = float(scanner_cfg.get("indicator_alert_repeat_s", scanner_cfg.get("scan_interval_s", 60)))
            now_ts    = time.time()
            last_ts   = self._indicator_alert_last_notify.get(result["rule_id"], 0.0)
            should_notify = bool(result["is_positive"]) and not is_muted and (now_ts - last_ts >= repeat_s)
            if should_notify:
                self._indicator_alert_last_notify[result["rule_id"]] = now_ts

            self.new_indicator_alert.emit({**result, "is_muted": is_muted, "should_notify": should_notify})

            if should_notify:
                self.log.emit(
                    f"[{symbol}] indicator_alert {rule['indicator']}.{rule['field']} "
                    f"= {result['value']:.2f}  {result['condition']} {result['threshold']}"
                )
                self._alert(result, scanner_cfg)


# ── backscan worker ────────────────────────────────────────────────────────────

class BackscanWorker(QThread):
    """Runs a single read-only historical FVG-watch scan off the UI thread.

    Wraps analysis.fvg_backscan.run_backscan() — the exact function the CLI
    tool uses — so the GUI panel and the CLI never diverge in behavior.
    Never touches SignalsDB.
    """

    finished_ok = pyqtSignal(list)
    failed      = pyqtSignal(str)

    def __init__(self, symbol: str, tf: str, start: str, end: str, parent=None) -> None:
        super().__init__(parent)
        self._symbol = symbol
        self._tf     = tf
        self._start  = start
        self._end    = end

    def run(self) -> None:
        try:
            from analysis.fvg_backscan import run_backscan
            df = run_backscan(None, self._symbol or None, self._tf or None, self._start, self._end)
            self.finished_ok.emit(df.to_dict("records"))
        except Exception as exc:
            self.failed.emit(str(exc))


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

        # Signal-type toggles — the two alert types are independent of each other.
        toggle_box = QGroupBox("Signal types")
        toggle_lay = QHBoxLayout(toggle_box)
        self._entry_enabled_cb = QCheckBox("Entry signal (trend + FVG + SL/TP/RR)")
        self._entry_enabled_cb.setChecked(
            bool(override.get("entry_signal_enabled", default.get("entry_signal_enabled", True)))
        )
        self._fvg_watch_enabled_cb = QCheckBox("FVG watch (config/scanner/fvg_watch_params.json)")
        self._fvg_watch_enabled_cb.setChecked(
            bool(override.get("fvg_watch_enabled", default.get("fvg_watch_enabled", False)))
        )
        self._session_vp_enabled_cb = QCheckBox("Session-VP (Value-Area reversal, config/scanner/session_vp_params.json)")
        self._session_vp_enabled_cb.setChecked(
            bool(override.get("session_vp_enabled", default.get("session_vp_enabled", False)))
        )
        self._vah_soxs_enabled_cb = QCheckBox("VAH-SOXS proxy (cross-asset, config/scanner/vah_soxs_params.json)")
        self._vah_soxs_enabled_cb.setChecked(
            bool(override.get("vah_soxs_enabled", default.get("vah_soxs_enabled", False)))
        )
        self._val_kd_tp_enabled_cb = QCheckBox("VAL+KD-TP (config/scanner/val_kd_tp_params.json)")
        self._val_kd_tp_enabled_cb.setChecked(
            bool(override.get("val_kd_tp_enabled", default.get("val_kd_tp_enabled", False)))
        )
        self._indicator_alert_enabled_cb = QCheckBox("Indicator Alert (RSI/MACD/KD…, config/scanner/indicator_alert_params.json)")
        self._indicator_alert_enabled_cb.setChecked(
            bool(override.get("indicator_alert_enabled", default.get("indicator_alert_enabled", False)))
        )
        toggle_lay.addWidget(self._entry_enabled_cb)
        toggle_lay.addWidget(self._fvg_watch_enabled_cb)
        toggle_lay.addWidget(self._session_vp_enabled_cb)
        toggle_lay.addWidget(self._vah_soxs_enabled_cb)
        toggle_lay.addWidget(self._val_kd_tp_enabled_cb)
        toggle_lay.addWidget(self._indicator_alert_enabled_cb)
        layout.addWidget(toggle_box)

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
        out: dict = {
            "auto_params": auto,
            "entry_signal_enabled": self._entry_enabled_cb.isChecked(),
            "fvg_watch_enabled":    self._fvg_watch_enabled_cb.isChecked(),
            "session_vp_enabled":   self._session_vp_enabled_cb.isChecked(),
            "vah_soxs_enabled":     self._vah_soxs_enabled_cb.isChecked(),
            "val_kd_tp_enabled":    self._val_kd_tp_enabled_cb.isChecked(),
            "indicator_alert_enabled": self._indicator_alert_enabled_cb.isChecked(),
        }
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


# ── indicator alert config dialog ───────────────────────────────────────────

class IndicatorAlertConfigDialog(QDialog):
    """Table editor for config/scanner/indicator_alert_params.json -- lets the
    user add/edit/delete per-symbol indicator threshold rules without hand-
    editing JSON. Deliberately generic (Indicator/Params/Field are free text,
    not RSI-specific widgets) so it stays usable once MACD/KD-style indicators
    are registered in analysis/indicator_alert_watcher.py's INDICATOR_REGISTRY.
    """

    _COLUMNS = ["Symbol", "Indicator", "TF", "Params (JSON)", "Field", "Condition", "Threshold"]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Indicator Alert Rules — config/scanner/indicator_alert_params.json")
        self.setMinimumSize(760, 420)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "One row per rule. A rule fires every scan cycle (not just on the edge\n"
            "crossing) while its condition holds -- mute a noisy rule from the Live tab's\n"
            "Indicator Alerts panel instead of deleting it here.\n"
            "Condition must be \"above\" or \"below\"; Params must be a JSON object, e.g. {\"period\": 6}."
        ))

        self._tbl = QTableWidget(0, len(self._COLUMNS))
        self._tbl.setHorizontalHeaderLabels(self._COLUMNS)
        self._tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self._tbl)

        rows_btns = QHBoxLayout()
        add_btn = QPushButton("Add Rule")
        add_btn.clicked.connect(self._on_add_row)
        rows_btns.addWidget(add_btn)
        del_btn = QPushButton("Delete Selected")
        del_btn.clicked.connect(self._on_delete_rows)
        rows_btns.addWidget(del_btn)
        rows_btns.addStretch()
        layout.addLayout(rows_btns)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        from analysis.indicator_alert_watcher import load_indicator_alert_config
        config = load_indicator_alert_config(_INDICATOR_ALERT_CFG_PATH)
        for symbol, rules in config.items():
            for rule in rules:
                self._append_row(symbol, rule)

    def _append_row(self, symbol: str = "", rule: Optional[dict] = None) -> None:
        rule = rule or {}
        row = self._tbl.rowCount()
        self._tbl.insertRow(row)
        self._tbl.setItem(row, 0, QTableWidgetItem(symbol))
        self._tbl.setItem(row, 1, QTableWidgetItem(rule.get("indicator", "rsi")))
        self._tbl.setItem(row, 2, QTableWidgetItem(rule.get("tf", "1m")))
        self._tbl.setItem(row, 3, QTableWidgetItem(json.dumps(rule.get("params", {}))))
        self._tbl.setItem(row, 4, QTableWidgetItem(rule.get("field", "value")))
        self._tbl.setItem(row, 5, QTableWidgetItem(rule.get("condition", "below")))
        self._tbl.setItem(row, 6, QTableWidgetItem(str(rule.get("threshold", 30))))

    def _on_add_row(self) -> None:
        self._append_row()

    def _on_delete_rows(self) -> None:
        for row in sorted({i.row() for i in self._tbl.selectedItems()}, reverse=True):
            self._tbl.removeRow(row)

    def _on_save(self) -> None:
        def cell(row: int, col: int) -> str:
            item = self._tbl.item(row, col)
            return item.text().strip() if item else ""

        config: dict[str, list[dict]] = {}
        for row in range(self._tbl.rowCount()):
            symbol = cell(row, 0)
            if not symbol:
                continue
            try:
                params = json.loads(cell(row, 3) or "{}")
            except Exception as exc:
                QMessageBox.warning(self, "Invalid params", f"Row {row + 1}: params must be valid JSON ({exc})")
                return
            condition = cell(row, 5)
            if condition not in ("above", "below"):
                QMessageBox.warning(self, "Invalid condition", f"Row {row + 1}: condition must be \"above\" or \"below\"")
                return
            try:
                threshold = float(cell(row, 6))
            except ValueError:
                QMessageBox.warning(self, "Invalid threshold", f"Row {row + 1}: threshold must be a number")
                return

            rule = {
                "indicator": cell(row, 1),
                "tf":        cell(row, 2),
                "params":    params,
                "field":     cell(row, 4),
                "condition": condition,
                "threshold": threshold,
            }
            config.setdefault(symbol, []).append(rule)

        out = {
            "_note": (
                "Per-symbol list of indicator threshold rules for "
                "analysis/indicator_alert_watcher.py. `indicator` selects a compute "
                "function from INDICATOR_REGISTRY (rsi/kd/... -- extend by registering "
                "a new function there, no schema change needed); `field` picks which "
                "output series of that indicator to compare against `threshold`. Each "
                "rule fires (repeatedly, once per scan cycle, not just on the edge "
                "crossing) whenever the latest closed bar's value satisfies "
                "condition+threshold -- mute a noisy rule from the scanner UI to pause "
                "it until it goes negative again."
            ),
            **config,
        }
        try:
            _INDICATOR_ALERT_CFG_PATH.parent.mkdir(parents=True, exist_ok=True)
            _INDICATOR_ALERT_CFG_PATH.write_text(
                json.dumps(out, indent=4, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        self.accept()


# ── main window ───────────────────────────────────────────────────────────────

class SignalScanner(QMainWindow):
    """Signal scanner main window."""

    def __init__(self, args=None) -> None:
        super().__init__()
        self.setWindowTitle("SMC Signal Scanner")
        self.resize(1000, 700)

        self._cfg    = self._load_cfg()
        self._worker: Optional[ScanWorker] = None

        self._signal_data: list[dict] = []   # mirrors signals table rows (full dicts)
        self._fvg_watch_data: list[dict] = []  # mirrors fvg watch table rows (full dicts)
        self._indicator_alert_data: list[dict] = []  # mirrors indicator alert table rows (full dicts)
        self._last_new_signal: Optional[dict] = None  # most recent signal for tray click
        self._backscan_worker: Optional["BackscanWorker"] = None

        self._build_ui()
        self._build_tray()
        self._refresh_targets_table()
        self._refresh_signals_table()
        self._refresh_fvg_watch_table()
        self._refresh_indicator_alert_table()

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
                "indicator_alert_repeat_s": 60,
                "default": {
                    "strategy": "smc",
                    "auto_params": True,
                    "lookback_months": 3,
                    "min_n_trades": 5,
                    "min_pf": 1.5,
                    "entry_signal_enabled": True,
                    "fvg_watch_enabled": False,
                    "session_vp_enabled": False,
                    "vah_soxs_enabled": False,
                    "val_kd_tp_enabled": False,
                    "indicator_alert_enabled": False,
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

        indicator_rules_btn = QPushButton("Indicator Alert Rules...")
        indicator_rules_btn.clicked.connect(self._on_edit_indicator_alert_rules)
        tb.addWidget(indicator_rules_btn)

        tabs = QTabWidget()
        main_lay.addWidget(tabs)
        tabs.addTab(self._build_live_tab(), "Live")
        tabs.addTab(self._build_backscan_tab(), "Backscan")

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

    def _build_live_tab(self) -> QWidget:
        """Targets / (entry) Signals / FVG Watch Signals — the live-monitoring view."""
        tab = QWidget()
        tab_lay = QVBoxLayout(tab)
        tab_lay.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter()
        splitter.setOrientation(Qt.Orientation.Vertical)
        tab_lay.addWidget(splitter)

        # Targets table
        targets_widget = QWidget()
        targets_lay = QVBoxLayout(targets_widget)
        targets_lay.setContentsMargins(0, 0, 0, 0)
        targets_lay.addWidget(QLabel("Targets  (click Entry / FVG Watch to toggle)"))
        self._targets_tbl = QTableWidget(0, 7)
        self._targets_tbl.setHorizontalHeaderLabels(
            ["Symbol", "Entry", "FVG Watch", "Mode", "Lookback", "TFs", "Last scan"]
        )
        self._targets_tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._targets_tbl.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._targets_tbl.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._targets_tbl.itemChanged.connect(self._on_target_item_changed)
        targets_lay.addWidget(self._targets_tbl)
        splitter.addWidget(targets_widget)

        # Entry signals table
        signals_widget = QWidget()
        signals_lay = QVBoxLayout(signals_widget)
        signals_lay.setContentsMargins(0, 0, 0, 0)
        signals_header = QHBoxLayout()
        signals_header.addWidget(QLabel("Recent entry signals (last 50)"))
        signals_header.addStretch()
        del_sig_btn = QPushButton("Delete")
        del_sig_btn.clicked.connect(self._on_delete_signal)
        signals_header.addWidget(del_sig_btn)
        signals_lay.addLayout(signals_header)
        self._signals_tbl = QTableWidget(0, 8)
        self._signals_tbl.setHorizontalHeaderLabels(
            ["Time", "Symbol", "Dir", "Entry zone", "SL", "TP", "RR", "Status"]
        )
        self._signals_tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._signals_tbl.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._signals_tbl.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._signals_tbl.cellDoubleClicked.connect(self._on_signal_double_clicked)
        signals_lay.addWidget(self._signals_tbl)
        splitter.addWidget(signals_widget)

        # FVG watch signals table
        fvg_watch_widget = QWidget()
        fvg_watch_lay = QVBoxLayout(fvg_watch_widget)
        fvg_watch_lay.setContentsMargins(0, 0, 0, 0)
        fvg_watch_header = QHBoxLayout()
        fvg_watch_header.addWidget(QLabel("FVG watch signals (last 50)"))
        fvg_watch_header.addStretch()
        del_fvg_btn = QPushButton("Delete")
        del_fvg_btn.clicked.connect(self._on_delete_fvg_watch)
        fvg_watch_header.addWidget(del_fvg_btn)
        fvg_watch_lay.addLayout(fvg_watch_header)
        self._fvg_watch_tbl = QTableWidget(0, 7)
        self._fvg_watch_tbl.setHorizontalHeaderLabels(
            ["Time", "Symbol", "TF", "Dir", "Zone", "Filled", "Status"]
        )
        self._fvg_watch_tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._fvg_watch_tbl.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._fvg_watch_tbl.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        fvg_watch_lay.addWidget(self._fvg_watch_tbl)
        splitter.addWidget(fvg_watch_widget)

        # Indicator alerts panel — a continuous status monitor (one row per rule,
        # upserted every scan cycle), not an event list like the panels above.
        indicator_alert_widget = QWidget()
        indicator_alert_lay = QVBoxLayout(indicator_alert_widget)
        indicator_alert_lay.setContentsMargins(0, 0, 0, 0)
        indicator_alert_header = QHBoxLayout()
        indicator_alert_header.addWidget(QLabel("Indicator alerts"))
        indicator_alert_header.addStretch()
        indicator_alert_header.addWidget(QLabel("Repeat sound every (s):"))
        self._indicator_alert_repeat_sp = QSpinBox()
        self._indicator_alert_repeat_sp.setRange(10, 3600)
        self._indicator_alert_repeat_sp.setValue(int(self._scanner_cfg().get(
            "indicator_alert_repeat_s", self._scanner_cfg().get("scan_interval_s", 60)
        )))
        self._indicator_alert_repeat_sp.setToolTip(
            "A rule that stays positive keeps its status but doesn't stop being\n"
            "positive -- without this throttle it would beep/notify every single\n"
            "scan cycle. Raise this to space out repeat notifications for the\n"
            "same rule (takes effect on the next Scan start, like the main\n"
            "Interval spinbox)."
        )
        self._indicator_alert_repeat_sp.valueChanged.connect(self._on_indicator_alert_repeat_changed)
        indicator_alert_header.addWidget(self._indicator_alert_repeat_sp)
        self._hide_muted_alerts_cb = QCheckBox("Hide muted")
        self._hide_muted_alerts_cb.setToolTip(
            "A muted row keeps updating its value every scan cycle (it doesn't\n"
            "expire or disappear on its own) -- check this to hide muted rows\n"
            "from the table instead. Uncheck to see and unmute them again."
        )
        self._hide_muted_alerts_cb.stateChanged.connect(lambda _: self._refresh_indicator_alert_table())
        indicator_alert_header.addWidget(self._hide_muted_alerts_cb)
        del_indicator_alert_btn = QPushButton("Delete")
        del_indicator_alert_btn.setToolTip(
            "Remove selected row(s). If the rule is still active in\n"
            "indicator_alert_params.json it will reappear on the next scan\n"
            "cycle -- use this to clear stale rows left over from a rule\n"
            "you've since edited or removed, not to silence an active one\n"
            "(use Mute for that)."
        )
        del_indicator_alert_btn.clicked.connect(self._on_delete_indicator_alert)
        indicator_alert_header.addWidget(del_indicator_alert_btn)
        indicator_alert_lay.addLayout(indicator_alert_header)
        self._indicator_alert_tbl = QTableWidget(0, 9)
        self._indicator_alert_tbl.setHorizontalHeaderLabels(
            ["Symbol", "TF", "Indicator", "Params", "Condition", "Value", "Status", "Updated", ""]
        )
        self._indicator_alert_tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._indicator_alert_tbl.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._indicator_alert_tbl.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        indicator_alert_lay.addWidget(self._indicator_alert_tbl)
        splitter.addWidget(indicator_alert_widget)

        splitter.setSizes([250, 250, 250, 250])
        return tab

    def _build_backscan_tab(self) -> QWidget:
        """Read-only historical FVG-watch scan — same logic as analysis/fvg_backscan.py."""
        tab = QWidget()
        tab_lay = QVBoxLayout(tab)

        form = QHBoxLayout()
        form.addWidget(QLabel("Symbol:"))
        self._backscan_symbol_cb = QComboBox()
        self._backscan_symbol_cb.setEditable(True)
        self._backscan_symbol_cb.currentTextChanged.connect(self._on_backscan_symbol_changed)
        form.addWidget(self._backscan_symbol_cb)

        form.addWidget(QLabel("TF:"))
        self._backscan_tf_cb = QComboBox()
        form.addWidget(self._backscan_tf_cb)

        form.addWidget(QLabel("Start:"))
        self._backscan_start_edit = QLineEdit()
        self._backscan_start_edit.setPlaceholderText("YYYY-MM-DD")
        form.addWidget(self._backscan_start_edit)

        self._backscan_run_btn = QPushButton("Run")
        self._backscan_run_btn.clicked.connect(self._on_run_backscan)
        form.addWidget(self._backscan_run_btn)
        form.addStretch()
        tab_lay.addLayout(form)

        self._backscan_status_lbl = QLabel("Idle")
        tab_lay.addWidget(self._backscan_status_lbl)

        self._backscan_tbl = QTableWidget(0, 8)
        self._backscan_tbl.setHorizontalHeaderLabels(
            ["Symbol", "TF", "Direction", "Formed time", "Zone bottom", "Zone top", "Width %", "Filled"]
        )
        self._backscan_tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._backscan_tbl.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        tab_lay.addWidget(self._backscan_tbl)

        self._populate_backscan_symbol_choices()
        return tab

    # ── system tray ───────────────────────────────────────────────────────────

    def _build_tray(self) -> None:
        if not QSystemTrayIcon.isSystemTrayAvailable():
            self._tray = None
            return

        from PyQt6.QtWidgets import QMenu
        from PyQt6.QtGui import QAction

        pix = QPixmap(16, 16)
        pix.fill(QColor("#26a69a"))
        self._tray = QSystemTrayIcon(QIcon(pix), self)
        self._tray.setToolTip("SMC Signal Scanner")

        menu = QMenu()
        show_act = QAction("Show", self)
        show_act.triggered.connect(self._tray_show)
        quit_act = QAction("Quit", self)
        quit_act.triggered.connect(self._tray_quit)
        menu.addAction(show_act)
        menu.addSeparator()
        menu.addAction(quit_act)
        self._tray.setContextMenu(menu)

        self._tray.messageClicked.connect(self._on_tray_message_clicked)
        self._tray.activated.connect(self._on_tray_activated)
        self._tray.show()

    def _tray_show(self) -> None:
        self.showNormal()
        self.activateWindow()
        self.raise_()

    def _tray_quit(self) -> None:
        self._stop_scan()
        if self._tray is not None:
            self._tray.hide()
        QApplication.quit()

    def _on_tray_message_clicked(self) -> None:
        if self._last_new_signal:
            self._open_viewer(self._last_new_signal)
        self._tray_show()

    def _on_tray_activated(self, reason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self._tray_show()

    # ── signal row double-click ───────────────────────────────────────────────

    def _on_signal_double_clicked(self, row: int, _col: int) -> None:
        if 0 <= row < len(self._signal_data):
            self._open_viewer(self._signal_data[row])

    # ── open trade viewer for a signal ───────────────────────────────────────

    def _open_viewer(self, sig: dict) -> None:
        symbol     = sig.get("symbol", "")
        trend_tf   = sig.get("trend_tf", "1h")
        sig_time   = str(sig.get("signal_time", ""))
        date_str   = sig_time[:10] if sig_time else datetime.now().strftime("%Y-%m-%d")
        if not symbol:
            return
        try:
            subprocess.Popen(
                [
                    sys.executable,
                    str(_ROOT / "main.py"),
                    "trade_viewer_qt",
                    "--code", symbol,
                    "--tf",   trend_tf,
                    "--mode", "Historical",
                    "--date", date_str,
                ],
                cwd=str(_ROOT),
            )
            self._log(f"Opened viewer: {symbol} {trend_tf} {date_str}")
        except Exception as exc:
            self._log(f"Failed to open viewer: {exc}")

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
        self._worker.new_fvg_watch_signal.connect(self._on_new_fvg_watch)
        self._worker.new_indicator_alert.connect(self._on_new_indicator_alert)
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

    def _on_indicator_alert_repeat_changed(self, value: int) -> None:
        self._scanner_cfg()["indicator_alert_repeat_s"] = value
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

    def _on_edit_indicator_alert_rules(self) -> None:
        dlg = IndicatorAlertConfigDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._log("Indicator alert rules saved to config/scanner/indicator_alert_params.json")

    # ── table refresh ─────────────────────────────────────────────────────────

    def _refresh_targets_table(self) -> None:
        scanner = self._scanner_cfg()
        enabled  = scanner.get("enabled", [])
        overrides = scanner.get("overrides", {})
        default  = scanner.get("default", {})

        # Block signals while repopulating so the checkbox cells we create
        # below don't re-trigger _on_target_item_changed.
        self._targets_tbl.blockSignals(True)
        try:
            self._targets_tbl.setRowCount(len(enabled))
            for row, symbol in enumerate(enabled):
                ov = overrides.get(symbol, {})
                auto = ov.get("auto_params", default.get("auto_params", True))
                mode = "Auto" if auto else "Manual"
                lb   = ov.get("lookback_months", default.get("lookback_months", 3))
                p    = ov.get("params", {})
                tfs  = f"{p.get('trend_tf', default.get('trend_tf', '1h'))}/{p.get('entry_tf', default.get('entry_tf', '15m'))}"
                entry_on = bool(ov.get("entry_signal_enabled", default.get("entry_signal_enabled", True)))
                fvg_on   = bool(ov.get("fvg_watch_enabled", default.get("fvg_watch_enabled", False)))

                self._targets_tbl.setItem(row, 0, QTableWidgetItem(symbol))
                self._targets_tbl.setItem(row, 1, self._checkbox_item(entry_on))
                self._targets_tbl.setItem(row, 2, self._checkbox_item(fvg_on))
                self._targets_tbl.setItem(row, 3, QTableWidgetItem(mode))
                self._targets_tbl.setItem(row, 4, QTableWidgetItem(str(lb) + " mo"))
                self._targets_tbl.setItem(row, 5, QTableWidgetItem(tfs if not auto else "—"))
                self._targets_tbl.setItem(row, 6, QTableWidgetItem("—"))
        finally:
            self._targets_tbl.blockSignals(False)

    @staticmethod
    def _checkbox_item(checked: bool) -> QTableWidgetItem:
        item = QTableWidgetItem()
        item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        return item

    def _on_target_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() not in (1, 2):
            return
        symbol_item = self._targets_tbl.item(item.row(), 0)
        if symbol_item is None:
            return
        symbol = symbol_item.text()
        key = "entry_signal_enabled" if item.column() == 1 else "fvg_watch_enabled"
        overrides = self._scanner_cfg().setdefault("overrides", {}).setdefault(symbol, {})
        overrides[key] = item.checkState() == Qt.CheckState.Checked
        self._save_cfg()

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
        self._signal_data = list(sigs)  # keep full dicts for double-click jump
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

    def _on_delete_signal(self) -> None:
        rows = sorted({i.row() for i in self._signals_tbl.selectedItems()})
        ids = [self._signal_data[r]["signal_id"] for r in rows if r < len(self._signal_data)]
        if not ids:
            return
        if QMessageBox.question(
            self, "Delete signal(s)", f"Delete {len(ids)} signal(s)? This cannot be undone.",
        ) != QMessageBox.StandardButton.Yes:
            return
        from db.signals import SignalsDB
        with SignalsDB(_SIGNALS_DB_PATH) as db:
            for sid in ids:
                db.delete_signal(sid)
        self._refresh_signals_table()

    def _refresh_fvg_watch_table(self) -> None:
        try:
            from db.signals import SignalsDB
            with SignalsDB(_SIGNALS_DB_PATH, read_only=False) as db:
                sigs = db.get_all_open_fvg_watch()
        except Exception:
            sigs = []
        self._populate_fvg_watch_table(sigs[:50])

    def _populate_fvg_watch_table(self, sigs: list[dict]) -> None:
        self._fvg_watch_data = list(sigs)
        self._fvg_watch_tbl.setRowCount(len(sigs))
        for row, sig in enumerate(sigs):
            direction = sig.get("direction", "")
            color = QColor("#26a69a") if direction == "bull" else QColor("#ef5350")
            items = [
                sig.get("formed_time", "")[:16],
                sig.get("symbol", ""),
                sig.get("tf", ""),
                direction.upper(),
                f"{sig.get('zone_bottom', 0):.2f}–{sig.get('zone_top', 0):.2f}",
                "Yes" if sig.get("filled") else "No",
                sig.get("status", ""),
            ]
            for col, text in enumerate(items):
                item = QTableWidgetItem(text)
                if col == 3:  # direction column
                    item.setForeground(color)
                self._fvg_watch_tbl.setItem(row, col, item)

    def _on_delete_fvg_watch(self) -> None:
        rows = sorted({i.row() for i in self._fvg_watch_tbl.selectedItems()})
        ids = [self._fvg_watch_data[r]["signal_id"] for r in rows if r < len(self._fvg_watch_data)]
        if not ids:
            return
        if QMessageBox.question(
            self, "Delete FVG watch signal(s)", f"Delete {len(ids)} signal(s)? This cannot be undone.",
        ) != QMessageBox.StandardButton.Yes:
            return
        from db.signals import SignalsDB
        with SignalsDB(_SIGNALS_DB_PATH) as db:
            for sid in ids:
                db.delete_fvg_watch(sid)
        self._refresh_fvg_watch_table()

    def _refresh_indicator_alert_table(self) -> None:
        try:
            from db.signals import SignalsDB
            with SignalsDB(_SIGNALS_DB_PATH, read_only=False) as db:
                rows = db.get_all_indicator_alert_state()
        except Exception:
            rows = []
        if self._hide_muted_alerts_cb.isChecked():
            rows = [r for r in rows if not r.get("is_muted")]
        self._populate_indicator_alert_table(rows)

    def _populate_indicator_alert_table(self, rows: list[dict]) -> None:
        self._indicator_alert_data = list(rows)
        self._indicator_alert_tbl.setRowCount(len(rows))
        for row, st in enumerate(rows):
            is_positive = bool(st.get("is_positive"))
            is_muted    = bool(st.get("is_muted"))
            try:
                params = json.loads(st.get("params_json") or "{}")
            except Exception:
                params = {}
            params_str = ", ".join(f"{k}={v}" for k, v in params.items())
            value = st.get("value")
            status = "MUTED" if (is_positive and is_muted) else ("ALERT" if is_positive else "—")
            items = [
                st.get("symbol", ""),
                st.get("tf", ""),
                st.get("indicator", ""),
                params_str,
                f"{st.get('field', '')} {st.get('condition', '')} {st.get('threshold', '')}",
                f"{value:.2f}" if value is not None else "—",
                status,
                (st.get("updated_at") or "")[:16],
            ]
            for col, text in enumerate(items):
                item = QTableWidgetItem(text)
                if is_positive and not is_muted:
                    item.setBackground(QColor("#ef5350"))
                elif is_positive and is_muted:
                    item.setBackground(QColor("#555555"))
                self._indicator_alert_tbl.setItem(row, col, item)

            rule_id = st.get("rule_id", "")
            btn = QPushButton("Muted" if is_muted else "Mute")
            btn.clicked.connect(
                lambda _checked=False, rid=rule_id, muted=is_muted: self._on_mute_indicator_alert(rid, not muted)
            )
            self._indicator_alert_tbl.setCellWidget(row, 8, btn)

    def _on_mute_indicator_alert(self, rule_id: str, muted: bool) -> None:
        try:
            from db.signals import SignalsDB
            with SignalsDB(_SIGNALS_DB_PATH) as db:
                db.set_indicator_alert_muted(rule_id, muted)
        except Exception as exc:
            self._log(f"indicator_alert mute error: {exc}")
        self._refresh_indicator_alert_table()

    def _on_delete_indicator_alert(self) -> None:
        """Remove selected row(s) from indicator_alert_state. If the rule is still
        listed in indicator_alert_params.json it reappears on the next scan cycle
        (upsert re-inserts it) -- this is for clearing rows left behind by a rule
        that's since been edited (a threshold/condition change makes a new
        rule_id, orphaning the old row) or removed, not for silencing a live one."""
        rows = sorted({i.row() for i in self._indicator_alert_tbl.selectedItems()})
        rule_ids = [
            self._indicator_alert_data[r]["rule_id"]
            for r in rows
            if r < len(self._indicator_alert_data)
        ]
        if not rule_ids:
            return
        if QMessageBox.question(
            self, "Delete indicator alert(s)", f"Delete {len(rule_ids)} row(s)?",
        ) != QMessageBox.StandardButton.Yes:
            return
        from db.signals import SignalsDB
        with SignalsDB(_SIGNALS_DB_PATH) as db:
            for rid in rule_ids:
                db.delete_indicator_alert_state(rid)
        self._refresh_indicator_alert_table()

    # ── worker callbacks ──────────────────────────────────────────────────────

    def _on_new_signal(self, sig: dict) -> None:
        self._last_new_signal = sig
        self._refresh_signals_table()
        summary = (
            f"{sig['symbol']}  {sig['direction'].upper()}"
            f"  zone {sig['entry_zone_bottom']:.2f}–{sig['entry_zone_top']:.2f}"
            f"  RR {sig['rr_ratio']:.1f}"
        )
        self._status_bar.showMessage(f"New signal: {summary}  ({sig['signal_time'][:16]})")
        if self._tray is not None:
            self._tray.showMessage(
                f"SMC Signal  {sig['signal_time'][:16]}",
                summary + "\n(double-click to open chart)",
                QSystemTrayIcon.MessageIcon.Information,
                8000,
            )

    def _on_new_fvg_watch(self, sig: dict) -> None:
        """Lightweight FVG-formed alert — no RR/SL/TP, separate from _on_new_signal."""
        self._refresh_fvg_watch_table()
        summary = (
            f"{sig['symbol']}  {sig['tf']}  {sig['direction'].upper()}"
            f"  zone {sig['zone_bottom']:.2f}–{sig['zone_top']:.2f}"
        )
        self._status_bar.showMessage(f"FVG formed: {summary}  ({sig['formed_time'][:16]})")
        if self._tray is not None:
            self._tray.showMessage(
                f"FVG formed  {sig['formed_time'][:16]}",
                summary,
                QSystemTrayIcon.MessageIcon.Information,
                8000,
            )

    def _on_new_indicator_alert(self, payload: dict) -> None:
        """Fires every scan cycle for every rule (positive, negative, or muted) so
        the table always reflects the latest reading; only pop a notification when
        the worker's should_notify says this cycle passed the positive+unmuted+
        indicator_alert_repeat_s throttle check (see _scan_symbol_indicator_alert)."""
        self._refresh_indicator_alert_table()
        if not payload.get("should_notify"):
            return
        summary = (
            f"{payload['symbol']}  {payload['indicator']}.{payload['field']} "
            f"{payload['value']:.2f} {payload['condition']} {payload['threshold']}"
        )
        self._status_bar.showMessage(f"Indicator alert: {summary}")
        if self._tray is not None:
            self._tray.showMessage(
                f"Indicator alert  {(payload.get('last_bar_time') or '')[:16]}",
                summary,
                QSystemTrayIcon.MessageIcon.Warning,
                8000,
            )

    def _on_status_update(self, symbol: str, status: str) -> None:
        for row in range(self._targets_tbl.rowCount()):
            item = self._targets_tbl.item(row, 0)
            if item and item.text() == symbol:
                self._targets_tbl.setItem(row, 6, QTableWidgetItem(status))
                break

    def _log(self, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self._log_edit.append(f"{ts}  {msg}")

    # ── backscan tab ──────────────────────────────────────────────────────────

    def _populate_backscan_symbol_choices(self) -> None:
        from analysis.fvg_watcher import load_fvg_watch_config
        config = load_fvg_watch_config()
        self._backscan_symbol_cb.clear()
        self._backscan_symbol_cb.addItems(sorted(config.keys()))
        if config:
            self._on_backscan_symbol_changed(self._backscan_symbol_cb.currentText())

    def _on_backscan_symbol_changed(self, symbol: str) -> None:
        from analysis.fvg_watcher import load_fvg_watch_config
        config = load_fvg_watch_config()
        tfs = [entry["tf"] for entry in config.get(symbol, [])]
        self._backscan_tf_cb.clear()
        self._backscan_tf_cb.addItem("All")
        self._backscan_tf_cb.addItems(tfs)

    def _on_run_backscan(self) -> None:
        if self._backscan_worker is not None and self._backscan_worker.isRunning():
            return
        symbol = self._backscan_symbol_cb.currentText().strip()
        tf = self._backscan_tf_cb.currentText().strip()
        tf = "" if tf in ("", "All") else tf
        start = self._backscan_start_edit.text().strip()
        if not start:
            QMessageBox.warning(self, "Backscan", "Start date is required (YYYY-MM-DD).")
            return
        end = datetime.now().strftime("%Y-%m-%d")

        self._backscan_run_btn.setEnabled(False)
        self._backscan_status_lbl.setText("Running...")
        self._backscan_worker = BackscanWorker(symbol, tf, start, end, self)
        self._backscan_worker.finished_ok.connect(self._on_backscan_finished)
        self._backscan_worker.failed.connect(self._on_backscan_error)
        self._backscan_worker.start()

    def _on_backscan_finished(self, rows: list[dict]) -> None:
        self._backscan_run_btn.setEnabled(True)
        self._backscan_status_lbl.setText(f"{len(rows)} match(es)")
        self._backscan_tbl.setRowCount(len(rows))
        for row, sig in enumerate(rows):
            direction = sig.get("direction", "")
            color = QColor("#26a69a") if direction == "bull" else QColor("#ef5350")
            items = [
                sig.get("symbol", ""),
                sig.get("tf", ""),
                direction.upper(),
                str(sig.get("formed_time", "")),
                f"{sig.get('zone_bottom', 0):.2f}",
                f"{sig.get('zone_top', 0):.2f}",
                f"{sig.get('width_pct', 0):.4f}",
                "Yes" if sig.get("filled") else "No",
            ]
            for col, text in enumerate(items):
                item = QTableWidgetItem(text)
                if col == 2:  # direction column
                    item.setForeground(color)
                self._backscan_tbl.setItem(row, col, item)

    def _on_backscan_error(self, message: str) -> None:
        self._backscan_run_btn.setEnabled(True)
        self._backscan_status_lbl.setText(f"Error: {message}")

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        if self._tray is not None:
            # Minimize to tray instead of quitting; use tray menu Quit to exit.
            event.ignore()
            self.hide()
            self._tray.showMessage(
                "SMC Signal Scanner",
                "Scanner is still running in the background.\n"
                "Right-click the tray icon to quit.",
                QSystemTrayIcon.MessageIcon.Information,
                3000,
            )
        else:
            self._stop_scan()
            super().closeEvent(event)


# ── entry point ───────────────────────────────────────────────────────────────

def main(argv=None) -> None:
    app = QApplication(sys.argv if argv is None else [sys.argv[0]] + list(argv))
    app.setApplicationName("SMC Signal Scanner")
    app.setQuitOnLastWindowClosed(False)  # keep alive when minimized to tray
    win = SignalScanner()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
