"""
Depth of Market (DOM) window — resting order book bar chart with absorption detection.

Shows bid (teal) and ask (red) depth as vertical bars: price on X, volume on Y.

Features:
  - Configurable depth (10 / 20 / 30 / 50 levels per side)
  - Live mode: refreshes every second from order_book.db
  - Historical mode: pinned to bar timestamp via pin_timestamp(); syncs with
    trade_viewer_qt crosshair
  - Absorption detection overlay: highlights price levels where large passive
    orders absorbed significant aggressive flow within the current bar window.
      Gold outline = ASK wall held (sellers absorbing buyers)
      Blue outline = BID wall held (buyers absorbing sellers)
  - Hover tooltip: volume, cumulative depth, and absorption details per level

Usage (standalone):
    uv run analysis/dom_window.py --code US.SNDK
"""

from __future__ import annotations

import pathlib
import sqlite3
import sys
from datetime import datetime, timedelta

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
    QHBoxLayout, QLabel, QSpinBox, QVBoxLayout, QWidget,
)

from core.time_utils import candle_start

# ── Colour palette ────────────────────────────────────────────────────────────
_BG       = "#1a1a2e"
_BG_BAR   = "#16213e"
_FG       = "#e0e0e0"
_GREY     = "#546e7a"
_BID      = (38,  166, 154, 200)   # teal
_ASK      = (239,  83,  80, 200)   # red
_BID_LINE = "#26a69a"
_ASK_LINE = "#ef5350"
_ABS_ASK  = "#ffa726"              # gold  — sell-side absorption
_ABS_BID  = "#42a5f5"              # blue  — buy-side absorption

_DB_PATH       = pathlib.Path(__file__).parent.parent / "db" / "order_book.db"
_TICKS_DB_PATH = pathlib.Path(__file__).parent.parent / "db" / "ticks.db"

# ── DB helpers ────────────────────────────────────────────────────────────────

def _query_latest_snapshot(code: str, db_path: pathlib.Path) -> list[dict]:
    """Most-recent full order book snapshot for `code`."""
    if not db_path.exists():
        return []
    try:
        con = sqlite3.connect(str(db_path), check_same_thread=False)
        row = con.execute(
            "SELECT MAX(ts) FROM order_book_snapshots WHERE code = ?", [code]
        ).fetchone()
        if not row or not row[0]:
            con.close()
            return []
        cur = con.execute(
            "SELECT ts, side, price, volume FROM order_book_snapshots "
            "WHERE code = ? AND ts = ?",
            [code, row[0]],
        )
        result = [{"ts": datetime.fromisoformat(r[0]),
                   "side": r[1], "price": r[2], "volume": r[3]}
                  for r in cur.fetchall()]
        con.close()
        return result
    except Exception:
        return []


def _query_snapshot_at(code: str, ts: datetime, db_path: pathlib.Path) -> list[dict]:
    """Most-recent snapshot at or before `ts` for `code`."""
    if not db_path.exists():
        return []
    try:
        con = sqlite3.connect(str(db_path), check_same_thread=False)
        ts_str = ts.isoformat(sep=" ")
        row = con.execute(
            "SELECT MAX(ts) FROM order_book_snapshots WHERE code = ? AND ts <= ?",
            [code, ts_str],
        ).fetchone()
        if not row or not row[0]:
            con.close()
            return []
        cur = con.execute(
            "SELECT ts, side, price, volume FROM order_book_snapshots "
            "WHERE code = ? AND ts = ?",
            [code, row[0]],
        )
        result = [{"ts": datetime.fromisoformat(r[0]),
                   "side": r[1], "price": r[2], "volume": r[3]}
                  for r in cur.fetchall()]
        con.close()
        return result
    except Exception:
        return []


def _query_ob_window(code: str, start: datetime, end: datetime,
                     db_path: pathlib.Path) -> list[dict]:
    """All order book snapshots for `code` in [start, end], sorted by ts."""
    if not db_path.exists():
        return []
    try:
        con = sqlite3.connect(str(db_path), check_same_thread=False)
        cur = con.execute(
            "SELECT ts, side, price, volume FROM order_book_snapshots "
            "WHERE code = ? AND ts >= ? AND ts <= ? ORDER BY ts",
            [code, start.isoformat(sep=" "), end.isoformat(sep=" ")],
        )
        result = [{"ts": datetime.fromisoformat(r[0]),
                   "side": r[1], "price": r[2], "volume": r[3]}
                  for r in cur.fetchall()]
        con.close()
        return result
    except Exception:
        return []


def _query_ticks_window(code: str, start: datetime, end: datetime,
                        db_path: pathlib.Path) -> list[dict]:
    """All tick records for `code` in [start, end]."""
    if not db_path.exists():
        return []
    try:
        con = sqlite3.connect(str(db_path), check_same_thread=False)
        cur = con.execute(
            "SELECT ts, price, volume, direction FROM ticks "
            "WHERE code = ? AND ts >= ? AND ts <= ?",
            [code, start.isoformat(sep=" "), end.isoformat(sep=" ")],
        )
        result = [{"ts": datetime.fromisoformat(r[0]),
                   "price": r[1], "volume": r[2], "direction": r[3]}
                  for r in cur.fetchall()]
        con.close()
        return result
    except Exception:
        return []


def _tick_size(prices: list[float]) -> float:
    """Infer minimum tick size from a sorted list of distinct price levels."""
    if len(prices) < 2:
        return 0.01
    diffs = [prices[i + 1] - prices[i]
             for i in range(len(prices) - 1)
             if prices[i + 1] != prices[i]]
    return min(diffs) if diffs else 0.01


def _sep() -> QLabel:
    lbl = QLabel("|")
    lbl.setStyleSheet(f"color: {_GREY}; padding: 0 3px;")
    return lbl


# ── DOM Window ────────────────────────────────────────────────────────────────

class DomWindow(QWidget):
    """Depth-of-Market chart with absorption detection overlay."""

    _DEPTH_CHOICES = [10, 20, 30, 50]
    _REFRESH_MS    = 1000

    def __init__(self, code: str = "US.SNDK", live: bool = True,
                 db_path: pathlib.Path | None = None,
                 ticks_db_path: pathlib.Path | None = None):
        super().__init__()
        self._code          = code
        self._live          = live
        self._db_path       = db_path or _DB_PATH
        self._ticks_db_path = ticks_db_path or _TICKS_DB_PATH
        self._depth         = 10
        self._candle_mins   = 1
        self._pinned_ts: datetime | None = None

        # Cached rendering state (populated by _render)
        self._bid_prices: list[float] = []
        self._bid_vols:   list[int]   = []
        self._ask_prices: list[float] = []
        self._ask_vols:   list[int]   = []
        self._tick:       float       = 0.01

        # Absorption results: [(price, side, agg_vol, pass_vol, ratio), ...]
        self._absorption: list[tuple] = []

        self.setWindowTitle(f"DOM — {code}")
        self.resize(860, 440)
        self.setWindowFlags(Qt.WindowType.Window)

        self._build_ui()
        self._apply_style()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        if live:
            self._timer.start(self._REFRESH_MS)

        self._refresh()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 4)
        root.setSpacing(4)

        # ── Toolbar ───────────────────────────────────────────────────────────
        bar = QHBoxLayout()
        bar.setSpacing(6)

        self._code_lbl = QLabel(f"Code: {self._code}")
        f = QFont(); f.setBold(True)
        self._code_lbl.setFont(f)
        bar.addWidget(self._code_lbl)

        bar.addWidget(_sep())
        bar.addWidget(QLabel("Depth:"))
        self._depth_combo = QComboBox()
        self._depth_combo.addItems([str(d) for d in self._DEPTH_CHOICES])
        self._depth_combo.setCurrentText(str(self._depth))
        self._depth_combo.setFixedWidth(60)
        self._depth_combo.currentTextChanged.connect(self._on_depth_changed)
        bar.addWidget(self._depth_combo)

        bar.addWidget(_sep())

        # Absorption controls
        self._abs_cb = QCheckBox("Absorb")
        self._abs_cb.setToolTip(
            "Highlight price levels where passive orders absorbed aggressive flow\n"
            "within the current bar window (aligned to chart timeframe).\n"
            "  Gold outline = ASK wall held (sellers absorbing buyers)\n"
            "  Blue outline = BID wall held (buyers absorbing sellers)")
        self._abs_cb.stateChanged.connect(self._on_abs_toggle)
        bar.addWidget(self._abs_cb)

        bar.addWidget(QLabel("Pass:"))
        self._pass_spin = QDoubleSpinBox()
        self._pass_spin.setRange(0.1, 100.0)
        self._pass_spin.setSingleStep(0.5)
        self._pass_spin.setValue(3.0)
        self._pass_spin.setDecimals(1)
        self._pass_spin.setFixedWidth(60)
        self._pass_spin.setToolTip(
            "Passive threshold multiplier.\n"
            "Resting order must be ≥ avg_tick_vol × Pass to qualify as a wall.")
        self._pass_spin.valueChanged.connect(self._on_abs_param_changed)
        bar.addWidget(self._pass_spin)
        bar.addWidget(QLabel("×"))

        bar.addWidget(QLabel("Act:"))
        self._act_spin = QDoubleSpinBox()
        self._act_spin.setRange(0.1, 100.0)
        self._act_spin.setSingleStep(0.5)
        self._act_spin.setValue(1.0)
        self._act_spin.setDecimals(1)
        self._act_spin.setFixedWidth(60)
        self._act_spin.setToolTip(
            "Active threshold multiplier.\n"
            "Cumulative aggressive volume must be ≥ avg_tick_vol × Act.")
        self._act_spin.valueChanged.connect(self._on_abs_param_changed)
        bar.addWidget(self._act_spin)
        bar.addWidget(QLabel("×"))

        bar.addWidget(QLabel("Hit:"))
        self._hit_spin = QSpinBox()
        self._hit_spin.setRange(1, 99)
        self._hit_spin.setValue(30)
        self._hit_spin.setSuffix("%")
        self._hit_spin.setFixedWidth(62)
        self._hit_spin.setToolTip(
            "Hit ratio threshold.\n"
            "Aggressive volume / passive volume must exceed this percentage.")
        self._hit_spin.valueChanged.connect(self._on_abs_param_changed)
        bar.addWidget(self._hit_spin)

        bar.addStretch()

        self._ts_lbl = QLabel("")
        self._ts_lbl.setStyleSheet("font-size: 10px;")
        bar.addWidget(self._ts_lbl)

        root.addLayout(bar)

        # ── Chart ─────────────────────────────────────────────────────────────
        self._pw = pg.PlotWidget(background=_BG)
        self._pw.setAntialiasing(True)
        pi = self._pw.getPlotItem()
        pi.showGrid(x=False, y=True, alpha=0.15)
        pi.getAxis("bottom").setTextPen(QColor(_FG))
        pi.getAxis("left").setTextPen(QColor(_FG))
        pi.setLabel("bottom", "Price", **{"color": _FG})
        pi.setLabel("left",   "Volume", **{"color": _FG})
        pi.setMenuEnabled(False)

        # Depth bars (drawn first, underneath absorption overlay)
        self._bid_bars = pg.BarGraphItem(
            x=[], height=[], width=0.01,
            brush=pg.mkBrush(_BID), pen=pg.mkPen(None),
        )
        self._ask_bars = pg.BarGraphItem(
            x=[], height=[], width=0.01,
            brush=pg.mkBrush(_ASK), pen=pg.mkPen(None),
        )
        self._pw.addItem(self._bid_bars)
        self._pw.addItem(self._ask_bars)

        # Absorption overlays (transparent fill, coloured border on top)
        self._abs_ask_bars = pg.BarGraphItem(
            x=[], height=[], width=0.01,
            brush=pg.mkBrush(0, 0, 0, 0),
            pen=pg.mkPen(_ABS_ASK, width=2),
        )
        self._abs_bid_bars = pg.BarGraphItem(
            x=[], height=[], width=0.01,
            brush=pg.mkBrush(0, 0, 0, 0),
            pen=pg.mkPen(_ABS_BID, width=2),
        )
        self._pw.addItem(self._abs_ask_bars)
        self._pw.addItem(self._abs_bid_bars)

        # Best bid/ask dashed markers
        self._best_bid_line = pg.InfiniteLine(
            angle=90,
            pen=pg.mkPen(_BID_LINE, width=1, style=Qt.PenStyle.DashLine),
        )
        self._best_ask_line = pg.InfiniteLine(
            angle=90,
            pen=pg.mkPen(_ASK_LINE, width=1, style=Qt.PenStyle.DashLine),
        )
        self._best_bid_line.setVisible(False)
        self._best_ask_line.setVisible(False)
        self._pw.addItem(self._best_bid_line)
        self._pw.addItem(self._best_ask_line)

        # Hover tooltip
        tip_bg = QColor(_BG_BAR)
        tip_bg.setAlpha(220)
        self._tooltip = pg.TextItem(
            text="", color=_FG, anchor=(0.0, 1.0),
            fill=pg.mkBrush(tip_bg),
            border=pg.mkPen(_GREY),
        )
        self._tooltip.setZValue(20)
        self._tooltip.setVisible(False)
        self._pw.addItem(self._tooltip, ignoreBounds=True)

        self._proxy = pg.SignalProxy(
            self._pw.scene().sigMouseMoved,
            rateLimit=30, slot=self._on_hover,
        )

        root.addWidget(self._pw)

        # Status bar
        self._status_lbl = QLabel("No data")
        self._status_lbl.setStyleSheet("font-size: 10px; padding: 2px 0;")
        root.addWidget(self._status_lbl)

    def _apply_style(self) -> None:
        self.setStyleSheet(f"""
            QWidget           {{ background: {_BG}; color: {_FG}; }}
            QLabel            {{ background: transparent; color: {_FG}; }}
            QCheckBox         {{ color: {_FG}; spacing: 4px; }}
            QComboBox, QSpinBox, QDoubleSpinBox {{
                background: {_BG_BAR}; color: {_FG};
                border: 1px solid {_GREY}; padding: 2px 4px;
            }}
        """)

    # ── Public API (called by TradeViewerQt) ──────────────────────────────────

    def set_code(self, code: str) -> None:
        if code == self._code:
            return
        self._code = code
        self._code_lbl.setText(f"Code: {code}")
        self.setWindowTitle(f"DOM — {code}")
        self._refresh()

    def set_live(self, live: bool) -> None:
        self._live = live
        if live:
            self._timer.start(self._REFRESH_MS)
        else:
            self._timer.stop()

    def set_timeframe(self, candle_mins: int) -> None:
        """Update the bar-aligned absorption window when the chart TF changes."""
        if candle_mins == self._candle_mins:
            return
        self._candle_mins = candle_mins
        if self._abs_cb.isChecked():
            self._compute_absorption()

    def pin_timestamp(self, ts: datetime | None) -> None:
        """Display the order book snapshot nearest to `ts` (historical crosshair sync)."""
        if ts == self._pinned_ts:
            return
        self._pinned_ts = ts
        if not self._live:
            self._refresh()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _on_depth_changed(self, text: str) -> None:
        try:
            self._depth = int(text)
        except ValueError:
            return
        self._refresh()

    def _on_abs_toggle(self) -> None:
        self._compute_absorption()

    def _on_abs_param_changed(self) -> None:
        if self._abs_cb.isChecked():
            self._compute_absorption()

    def _get_window(self) -> tuple[datetime, datetime]:
        """Bar-aligned absorption window.

        Live:       [candle_start(now, cm), now]
        Historical: [time_key - cm,  time_key]   (moomoo time_key = bar end)
        """
        cm = self._candle_mins
        if not self._live and self._pinned_ts is not None:
            window_end   = self._pinned_ts
            window_start = window_end - timedelta(minutes=cm)
        else:
            now          = datetime.now()
            window_start = candle_start(now, cm)
            window_end   = now
        return window_start, window_end

    def _refresh(self) -> None:
        if self._live or self._pinned_ts is None:
            rows = _query_latest_snapshot(self._code, self._db_path)
        else:
            rows = _query_snapshot_at(self._code, self._pinned_ts, self._db_path)
        self._render(rows)
        if self._abs_cb.isChecked():
            self._compute_absorption()

    def _render(self, rows: list[dict]) -> None:
        if not rows:
            self._bid_bars.setOpts(x=[], height=[], width=0.01)
            self._ask_bars.setOpts(x=[], height=[], width=0.01)
            self._abs_ask_bars.setOpts(x=[], height=[], width=0.01)
            self._abs_bid_bars.setOpts(x=[], height=[], width=0.01)
            self._best_bid_line.setVisible(False)
            self._best_ask_line.setVisible(False)
            self._bid_prices = []; self._bid_vols = []
            self._ask_prices = []; self._ask_vols = []
            self._absorption = []
            self._status_lbl.setText(
                f"No data for {self._code} — is order_book_collector.py running?")
            return

        bids = sorted(
            [(r["price"], r["volume"]) for r in rows if r["side"] == "BID"],
            key=lambda x: -x[0],
        )[:self._depth]
        asks = sorted(
            [(r["price"], r["volume"]) for r in rows if r["side"] == "ASK"],
            key=lambda x: x[0],
        )[:self._depth]

        if not bids and not asks:
            self._status_lbl.setText("Snapshot contains no bid/ask levels.")
            return

        self._ts_lbl.setText(str(rows[0]["ts"])[:19])

        self._bid_prices = [b[0] for b in bids]
        self._bid_vols   = [b[1] for b in bids]
        self._ask_prices = [a[0] for a in asks]
        self._ask_vols   = [a[1] for a in asks]

        all_sorted = sorted(set(self._bid_prices + self._ask_prices))
        self._tick  = _tick_size(all_sorted)
        bar_width   = self._tick * 0.8

        self._bid_bars.setOpts(
            x=self._bid_prices, height=self._bid_vols, width=bar_width,
            brush=pg.mkBrush(_BID), pen=pg.mkPen(None),
        )
        self._ask_bars.setOpts(
            x=self._ask_prices, height=self._ask_vols, width=bar_width,
            brush=pg.mkBrush(_ASK), pen=pg.mkPen(None),
        )

        if self._bid_prices:
            self._best_bid_line.setPos(self._bid_prices[0])
            self._best_bid_line.setVisible(True)
        if self._ask_prices:
            self._best_ask_line.setPos(self._ask_prices[0])
            self._best_ask_line.setVisible(True)

        total_bid = sum(self._bid_vols)
        total_ask = sum(self._ask_vols)
        if self._bid_prices and self._ask_prices:
            spread = self._ask_prices[0] - self._bid_prices[0]
            self._status_lbl.setText(
                f"Bid {self._bid_prices[0]:.4f} ({total_bid:,})  "
                f"Ask {self._ask_prices[0]:.4f} ({total_ask:,})  "
                f"Spread {spread:.4f}"
            )
        else:
            self._status_lbl.setText("Data loaded.")

    def _compute_absorption(self) -> None:
        """Query window data, run detect_absorption, update overlay bars."""
        if not self._abs_cb.isChecked() or not (self._bid_prices or self._ask_prices):
            self._abs_ask_bars.setOpts(x=[], height=[], width=0.01)
            self._abs_bid_bars.setOpts(x=[], height=[], width=0.01)
            self._absorption = []
            return

        window_start, window_end = self._get_window()
        ob_win    = _query_ob_window(
            self._code, window_start, window_end, self._db_path)
        ticks_win = _query_ticks_window(
            self._code, window_start, window_end, self._ticks_db_path)

        if not ob_win or not ticks_win:
            self._abs_ask_bars.setOpts(x=[], height=[], width=0.01)
            self._abs_bid_bars.setOpts(x=[], height=[], width=0.01)
            self._absorption = []
            return

        all_prices = sorted(set(self._bid_prices + self._ask_prices))
        tick       = self._tick
        price_min  = all_prices[0] - tick
        price_max  = all_prices[-1] + tick
        n_bins     = max(1, round((price_max - price_min) / tick)) + 2

        from analysis.orderflow_detect import detect_absorption
        self._absorption = detect_absorption(
            ob_data   = ob_win,
            raw_ticks = ticks_win,
            bin_size  = tick,
            price_min = price_min,
            N_PRICE   = n_bins,
            passive_k = self._pass_spin.value(),
            active_k  = self._act_spin.value(),
            hit_ratio = self._hit_spin.value() / 100.0,
        )

        bar_width   = tick * 0.8
        abs_ask_p, abs_ask_v = [], []
        abs_bid_p, abs_bid_v = [], []

        for price, side, _agg, _pas, _ratio in self._absorption:
            if side == "ASK":
                src_p, src_v = self._ask_prices, self._ask_vols
                dst_p, dst_v = abs_ask_p, abs_ask_v
            else:
                src_p, src_v = self._bid_prices, self._bid_vols
                dst_p, dst_v = abs_bid_p, abs_bid_v
            for p, v in zip(src_p, src_v):
                if abs(p - price) < tick * 0.5:
                    dst_p.append(p); dst_v.append(v)
                    break

        self._abs_ask_bars.setOpts(
            x=abs_ask_p, height=abs_ask_v, width=bar_width,
            brush=pg.mkBrush(0, 0, 0, 0),
            pen=pg.mkPen(_ABS_ASK, width=2),
        )
        self._abs_bid_bars.setOpts(
            x=abs_bid_p, height=abs_bid_v, width=bar_width,
            brush=pg.mkBrush(0, 0, 0, 0),
            pen=pg.mkPen(_ABS_BID, width=2),
        )

    # ── Hover tooltip ─────────────────────────────────────────────────────────

    def _on_hover(self, evt) -> None:
        pos = evt[0]
        if not self._pw.sceneBoundingRect().contains(pos):
            self._tooltip.setVisible(False)
            return
        pi     = self._pw.getPlotItem()
        pt     = pi.vb.mapSceneToView(pos)
        px, py = pt.x(), pt.y()
        tip    = self._build_tooltip(px)
        if tip is None:
            self._tooltip.setVisible(False)
            return
        xlo, xhi = pi.vb.viewRange()[0]
        ylo, yhi = pi.vb.viewRange()[1]
        tx = min(px + self._tick * 0.5, xhi - (xhi - xlo) * 0.25)
        ty = min(py + (yhi - ylo) * 0.02, yhi - (yhi - ylo) * 0.05)
        self._tooltip.setPos(tx, ty)
        self._tooltip.setText(tip)
        self._tooltip.setVisible(True)

    def _build_tooltip(self, px: float) -> str | None:
        half = self._tick * 0.5

        def _abs_suffix(price: float, side: str) -> str:
            for abs_p, abs_s, agg, pas, ratio in self._absorption:
                if abs_s == side and abs(abs_p - price) < half:
                    label = "SELL ABSORPTION" if side == "ASK" else "BUY ABSORPTION"
                    return (
                        f"\n⚡ {label}\n"
                        f"  Aggressive {agg:,.0f}  /  Passive {pas:,.0f}\n"
                        f"  Hit ratio  {ratio:.0%}"
                    )
            return ""

        for i, (p, v) in enumerate(zip(self._bid_prices, self._bid_vols)):
            if abs(px - p) <= half:
                cum  = sum(self._bid_vols[:i + 1])
                best = self._bid_prices[0]
                return (
                    f"BID  {p:.4f}\n"
                    f"Volume : {v:,}\n"
                    f"Cum from best ({best:.4f}): {cum:,}\n"
                    f"Levels: {i + 1}"
                    + _abs_suffix(p, "BID")
                )

        for i, (p, v) in enumerate(zip(self._ask_prices, self._ask_vols)):
            if abs(px - p) <= half:
                cum  = sum(self._ask_vols[:i + 1])
                best = self._ask_prices[0]
                return (
                    f"ASK  {p:.4f}\n"
                    f"Volume : {v:,}\n"
                    f"Cum from best ({best:.4f}): {cum:,}\n"
                    f"Levels: {i + 1}"
                    + _abs_suffix(p, "ASK")
                )
        return None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        self._timer.stop()
        super().closeEvent(event)


# ── Standalone entry ──────────────────────────────────────────────────────────

def _main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="DOM window (standalone)")
    parser.add_argument("--code", default="US.SNDK")
    parser.add_argument("--db",   default=None, help="Path to order_book.db")
    args = parser.parse_args()
    db   = pathlib.Path(args.db) if args.db else None
    app  = QApplication(sys.argv)
    win  = DomWindow(code=args.code, live=True, db_path=db)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    _main()
