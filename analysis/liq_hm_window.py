"""Liquidity Heatmap — independent floating window.

Real-time resting order book depth as price × time heatmap.
  - X-axis: real wall-clock time, independent of K-line bars
  - Y-axis: price (same units as main chart)
  - Each column: one OB snapshot (latest state at that moment)
  - New columns appended on the right; old columns scroll off the left
  - Iceberg / spoof markers overlaid on the heatmap
  - Crosshair synced with the main Trade Viewer via pin_timestamp()
"""
from __future__ import annotations

import bisect
import pathlib
import sqlite3
import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
from datetime import datetime

from PyQt6.QtCore    import Qt, QRectF, QTimer, QThread, pyqtSignal
from PyQt6.QtGui     import QColor
from PyQt6.QtWidgets import (
    QCheckBox, QLabel, QPushButton, QSpinBox, QToolBar, QVBoxLayout, QWidget,
)
import pyqtgraph as pg

from core.time_utils import candle_start

# ── palette (matches trade_viewer_qt) ──────────────────────────────────────────
_BG   = "#0d1117"
_FG   = "#b0bec5"
_GRID = "#263238"
_TEAL = "#26a69a"   # bid
_RED  = "#ef5350"   # ask

_DB_PATH = pathlib.Path(__file__).parent.parent / "db" / "order_book.db"

N_PRICE      = 100   # price bins (y-resolution)
COL_SECS_DEF = 30    # seconds per column
MAX_COLS_DEF = 240   # columns kept in memory  (2 h at 30 s/col)


# ── data helpers ───────────────────────────────────────────────────────────────

def _query_latest_snapshot(code: str) -> list[dict]:
    """Return all rows from the most recent OB push for *code* (10-20 rows).

    Uses a plain sqlite3 connection (no URI mode) to avoid Windows path issues.
    ts is included so callers can use it for iceberg/spoof detection.
    """
    if not _DB_PATH.exists():
        return []
    try:
        con = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        cur = con.execute(
            "SELECT ts, side, price, volume FROM order_book_snapshots "
            "WHERE code = ? AND ts = ("
            "  SELECT MAX(ts) FROM order_book_snapshots WHERE code = ?"
            ")",
            [code, code],
        )
        rows = [
            {
                "ts":     datetime.fromisoformat(r[0]),
                "side":   r[1],
                "price":  float(r[2]),
                "volume": float(r[3]),
            }
            for r in cur.fetchall()
        ]
        con.close()
        return rows
    except Exception:
        return []


# ── background query worker ───────────────────────────────────────────────────

class _SnapshotWorker(QThread):
    """Fetch the latest OB snapshot in a background thread to avoid UI stalls."""
    done = pyqtSignal(list)   # emits list[dict] (may be empty)

    def __init__(self, code: str) -> None:
        super().__init__()
        self._code = code

    def run(self) -> None:
        self.done.emit(_query_latest_snapshot(self._code))


# ── color renderers ────────────────────────────────────────────────────────────

def _hot_rgba(grid: np.ndarray) -> np.ndarray | None:
    """Combined bid+ask: black → purple → amber → yellow."""
    if grid.max() <= 0:
        return None
    log_g = np.log1p(grid)
    norm  = (log_g / log_g.max()).astype(np.float32)
    rgba  = np.zeros((*norm.shape, 4), dtype=np.uint8)

    m = norm < 0.25
    t = norm[m] / 0.25
    rgba[m, 0] = (t * 60).astype(np.uint8)
    rgba[m, 2] = (t * 140).astype(np.uint8)

    m = (norm >= 0.25) & (norm < 0.5)
    t = (norm[m] - 0.25) / 0.25
    rgba[m, 0] = (60  + t * 100).astype(np.uint8)
    rgba[m, 2] = (140 + t * 60 ).astype(np.uint8)

    m = (norm >= 0.5) & (norm < 0.75)
    t = (norm[m] - 0.5) / 0.25
    rgba[m, 0] = (160 + t * 95 ).astype(np.uint8)
    rgba[m, 1] = (t   * 80     ).astype(np.uint8)
    rgba[m, 2] = (200 - t * 200).astype(np.uint8)

    m = norm >= 0.75
    t = (norm[m] - 0.75) / 0.25
    rgba[m, 0] = 255
    rgba[m, 1] = (80 + t * 155).astype(np.uint8)
    rgba[m, 2] = 0

    rgba[..., 3]         = np.clip((norm * 200 + 20).astype(np.uint8), 0, 220)
    rgba[norm < 0.02, 3] = 0
    return rgba


def _single_rgba(grid: np.ndarray, hex_color: str) -> np.ndarray | None:
    """Single-hue intensity map for bid or ask separately."""
    if grid.max() <= 0:
        return None
    log_g = np.log1p(grid)
    norm  = (log_g / log_g.max()).astype(np.float32)
    c     = QColor(hex_color)
    rgba  = np.zeros((*norm.shape, 4), dtype=np.uint8)
    rgba[..., 0] = c.red()
    rgba[..., 1] = c.green()
    rgba[..., 2] = c.blue()
    rgba[..., 3]         = np.clip((norm * 180).astype(np.uint8), 0, 180)
    rgba[norm < 0.02, 3] = 0
    return rgba


def _blend(color_rgb: tuple, alpha: float) -> str:
    """Blend color onto chart background at given alpha; return hex string."""
    bg = (13, 17, 23)  # _BG = "#0d1117"
    r = int(bg[0] * (1 - alpha) + color_rgb[0] * alpha)
    g = int(bg[1] * (1 - alpha) + color_rgb[1] * alpha)
    b = int(bg[2] * (1 - alpha) + color_rgb[2] * alpha)
    return f"#{r:02x}{g:02x}{b:02x}"


# ── custom time axis ───────────────────────────────────────────────────────────

class _TimeAxisItem(pg.AxisItem):
    """Bottom axis showing HH:MM labels derived from column timestamps."""

    def __init__(self) -> None:
        super().__init__(orientation="bottom")
        self._col_ts: list[datetime] = []

    def update_timestamps(self, ts_list: list[datetime]) -> None:
        self._col_ts = ts_list[:]

    def tickStrings(self, values, scale, spacing) -> list[str]:
        result = []
        for v in values:
            i = int(round(v))
            if 0 <= i < len(self._col_ts):
                result.append(self._col_ts[i].strftime("%H:%M"))
            else:
                result.append("")
        return result


# ── main window ────────────────────────────────────────────────────────────────

class LiqHmWindow(QWidget):
    """Floating liquidity heatmap: resting OB depth as a real-time price×time image.

    Iceberg and spoof markers are overlaid on the same price×time canvas so
    that detected order-flow events can be read in context of resting liquidity.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("Liquidity Heatmap")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.resize(1100, 420)

        self._code: str  = ""
        self._live: bool = True

        # Rolling grid (index 0 = oldest visible column)
        self._bid_grid = np.zeros((MAX_COLS_DEF, N_PRICE), dtype=np.float64)
        self._ask_grid = np.zeros((MAX_COLS_DEF, N_PRICE), dtype=np.float64)
        self._col_ts:  list[datetime] = []

        # Raw snapshot buffer for iceberg / spoof detection
        # Each entry: {ts, side, price, volume}
        self._raw_snaps: list[dict] = []

        # Best bid / ask from the most recent snapshot (盘口)
        self._best_bid: float | None = None
        self._best_ask: float | None = None

        # Price range (auto-detected from first snapshot)
        self._price_min: float = 0.0
        self._price_max: float = 0.0
        self._bin_size:  float = 0.0

        # Overlay items managed on _plot_widget
        self._iceberg_items: list = []
        self._spoof_items:   list = []

        # Background query worker (one at a time)
        self._worker: _SnapshotWorker | None = None

        self._build_toolbar()
        self._build_chart()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._legend_lbl)
        layout.addWidget(self._plot_widget)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)

    # ── construction ───────────────────────────────────────────────────────────

    def _build_toolbar(self) -> None:
        def _lbl(text: str) -> QLabel:
            w = QLabel(text)
            w.setStyleSheet(f"color:{_FG};padding:0 4px;")
            return w

        tb = QToolBar()
        tb.setMovable(False)

        self._bid_ask_cb = QCheckBox("Bid/Ask")
        self._bid_ask_cb.setChecked(False)
        self._bid_ask_cb.setToolTip(
            "Checked: teal = Bid, red = Ask (shown separately)\n"
            "Unchecked: combined with black → purple → yellow colormap")
        self._bid_ask_cb.stateChanged.connect(self._on_controls_changed)
        tb.addWidget(self._bid_ask_cb)

        tb.addSeparator()
        tb.addWidget(_lbl("Min.Vol:"))
        self._min_vol_spin = QSpinBox()
        self._min_vol_spin.setRange(0, 1_000_000)
        self._min_vol_spin.setSingleStep(100)
        self._min_vol_spin.setValue(0)
        self._min_vol_spin.setFixedWidth(75)
        self._min_vol_spin.setToolTip(
            "Minimum volume to include in heatmap and detection.\n"
            "Spoof detection: 0 = auto (median of latest snapshot levels)"
        )
        self._min_vol_spin.valueChanged.connect(self._on_controls_changed)
        tb.addWidget(self._min_vol_spin)

        tb.addSeparator()
        tb.addWidget(_lbl("Col(s):"))
        self._col_secs_spin = QSpinBox()
        self._col_secs_spin.setRange(5, 300)
        self._col_secs_spin.setSingleStep(5)
        self._col_secs_spin.setValue(COL_SECS_DEF)
        self._col_secs_spin.setFixedWidth(55)
        self._col_secs_spin.setToolTip("Seconds per column (time resolution)")
        self._col_secs_spin.valueChanged.connect(self._on_col_secs_changed)
        tb.addWidget(self._col_secs_spin)

        tb.addSeparator()
        tb.addWidget(_lbl("History:"))
        self._max_cols_spin = QSpinBox()
        self._max_cols_spin.setRange(60, 1440)
        self._max_cols_spin.setSingleStep(60)
        self._max_cols_spin.setValue(MAX_COLS_DEF)
        self._max_cols_spin.setFixedWidth(60)
        self._max_cols_spin.setToolTip("Number of columns kept in memory")
        self._max_cols_spin.valueChanged.connect(self._on_max_cols_changed)
        tb.addWidget(self._max_cols_spin)

        tb.addSeparator()

        # Iceberg detection
        self._ice_cb = QCheckBox("Iceberg")
        self._ice_cb.setChecked(False)
        self._ice_cb.setToolTip(
            "Cyan line: price level where resting volume repeatedly drops then\n"
            "refreshes — hallmark of a hidden large order refilling at a fixed price.")
        self._ice_cb.stateChanged.connect(self._on_controls_changed)
        tb.addWidget(self._ice_cb)

        tb.addWidget(_lbl("Min.Ref:"))
        self._ice_min_ref_spin = QSpinBox()
        self._ice_min_ref_spin.setRange(1, 50)
        self._ice_min_ref_spin.setSingleStep(1)
        self._ice_min_ref_spin.setValue(3)
        self._ice_min_ref_spin.setFixedWidth(50)
        self._ice_min_ref_spin.setToolTip("Minimum refreshes to classify as iceberg")
        self._ice_min_ref_spin.valueChanged.connect(self._on_controls_changed)
        tb.addWidget(self._ice_min_ref_spin)

        tb.addSeparator()

        # Spoof detection
        self._spoof_cb = QCheckBox("Spoof")
        self._spoof_cb.setChecked(False)
        self._spoof_cb.setToolTip(
            "Orange ▲/▼: large order that appears then vanishes without execution.\n"
            "▲ = bid spoof (false buy pressure)  ▼ = ask spoof (false sell pressure)")
        self._spoof_cb.stateChanged.connect(self._on_controls_changed)
        tb.addWidget(self._spoof_cb)

        tb.addWidget(_lbl("Max.Dur(s):"))
        self._spoof_dur_spin = QSpinBox()
        self._spoof_dur_spin.setRange(3, 300)
        self._spoof_dur_spin.setSingleStep(5)
        self._spoof_dur_spin.setValue(30)
        self._spoof_dur_spin.setFixedWidth(50)
        self._spoof_dur_spin.setToolTip("Max seconds a large order can live before being flagged")
        self._spoof_dur_spin.valueChanged.connect(self._on_controls_changed)
        tb.addWidget(self._spoof_dur_spin)

        tb.addSeparator()

        # Reset zoom button
        reset_btn = QPushButton("⟲ Reset")
        reset_btn.setToolTip("Reset zoom to full view (double-click chart also resets)")
        reset_btn.setFixedWidth(70)
        reset_btn.clicked.connect(self._reset_view)
        tb.addWidget(reset_btn)

        self._toolbar = tb

        # Legend bar — separate row below toolbar so it never gets truncated
        self._legend_lbl = QLabel()
        self._legend_lbl.setStyleSheet(
            f"background:{_BG};color:{_FG};padding:2px 8px;font-size:8pt;")
        self._update_legend()

    def _build_chart(self) -> None:
        self._time_axis = _TimeAxisItem()
        self._plot_widget = pg.PlotWidget(
            background=_BG,
            axisItems={"bottom": self._time_axis},
        )
        pi = self._plot_widget.getPlotItem()
        pi.showGrid(x=True, y=True, alpha=0.15)
        pi.setMenuEnabled(False)
        pi.getAxis("left").setTextPen(_FG)
        pi.getAxis("bottom").setTextPen(_FG)

        self._img_combined = pg.ImageItem()
        self._img_bid      = pg.ImageItem()
        self._img_ask      = pg.ImageItem()
        for img in (self._img_combined, self._img_bid, self._img_ask):
            img.setZValue(-10)
            self._plot_widget.addItem(img)

        _cross_pen = pg.mkPen(_FG, width=1, style=Qt.PenStyle.DashLine)

        # Vertical line — set by pin_timestamp() from main chart OR local mouse
        self._vline = pg.InfiniteLine(angle=90, movable=False, pen=_cross_pen)
        self._vline.setVisible(False)
        self._plot_widget.addItem(self._vline)

        # Horizontal line — follows mouse within this window
        self._hline = pg.InfiniteLine(angle=0, movable=False, pen=_cross_pen)
        self._hline.setVisible(False)
        self._plot_widget.addItem(self._hline)

        # Price label anchored to left edge at cursor Y
        self._price_lbl = pg.TextItem(anchor=(0.0, 0.5), color=_FG)
        self._price_lbl.setZValue(50)
        self._price_lbl.setVisible(False)
        self._plot_widget.addItem(self._price_lbl, ignoreBounds=True)

        # Time label anchored just above the bottom axis at cursor X
        self._time_lbl = pg.TextItem(anchor=(0.5, 1.0), color=_FG)
        self._time_lbl.setZValue(50)
        self._time_lbl.setVisible(False)
        self._plot_widget.addItem(self._time_lbl, ignoreBounds=True)

        # Best bid / ask horizontal lines (盘口)
        self._bid_line = pg.InfiniteLine(
            angle=0, movable=False,
            pen=pg.mkPen(_TEAL, width=1, style=Qt.PenStyle.DashLine),
        )
        self._bid_line.setVisible(False)
        self._bid_line.setZValue(20)
        self._plot_widget.addItem(self._bid_line)

        self._ask_line = pg.InfiniteLine(
            angle=0, movable=False,
            pen=pg.mkPen(_RED, width=1, style=Qt.PenStyle.DashLine),
        )
        self._ask_line.setVisible(False)
        self._ask_line.setZValue(20)
        self._plot_widget.addItem(self._ask_line)

        # Mouse tracking
        self._plot_widget.scene().sigMouseMoved.connect(self._on_mouse_move)

    # ── public API ─────────────────────────────────────────────────────────────

    def set_code(self, code: str) -> None:
        if code == self._code:
            return
        self._code = code
        self._reset_grid()

    def set_live(self, live: bool) -> None:
        self._live = live
        if live and self._code:
            self._timer.start(self._col_secs_spin.value() * 1000)
            self._on_tick()   # fetch immediately without waiting for first interval
        else:
            self._timer.stop()

    def pin_timestamp(self, ts: datetime) -> None:
        """Move vertical crosshair to the column nearest *ts*."""
        if not self._col_ts:
            self._vline.setVisible(False)
            return
        # Binary search on sorted col_ts
        i = bisect.bisect_left(self._col_ts, ts)
        if i == 0:
            idx = 0
        elif i >= len(self._col_ts):
            idx = len(self._col_ts) - 1
        else:
            before = (ts - self._col_ts[i - 1]).total_seconds()
            after  = (self._col_ts[i] - ts).total_seconds()
            idx    = i - 1 if before <= after else i
        self._vline.setValue(idx + 0.5)
        self._vline.setVisible(True)

    # ── internals ──────────────────────────────────────────────────────────────

    def _reset_grid(self) -> None:
        max_cols = self._max_cols_spin.value()
        self._bid_grid  = np.zeros((max_cols, N_PRICE), dtype=np.float64)
        self._ask_grid  = np.zeros((max_cols, N_PRICE), dtype=np.float64)
        self._col_ts    = []
        self._raw_snaps = []
        self._price_min = 0.0
        self._price_max = 0.0
        self._bin_size  = 0.0
        self._best_bid  = None
        self._best_ask  = None
        for img in (self._img_combined, self._img_bid, self._img_ask):
            img.clear()
        self._vline.setVisible(False)
        self._hline.setVisible(False)
        self._price_lbl.setVisible(False)
        self._time_lbl.setVisible(False)
        self._bid_line.setVisible(False)
        self._ask_line.setVisible(False)
        self._clear_overlay_items()

    def _on_mouse_move(self, pos) -> None:
        """Update local crosshair when cursor is inside the plot."""
        if not self._plot_widget.sceneBoundingRect().contains(pos):
            self._hline.setVisible(False)
            self._price_lbl.setVisible(False)
            self._time_lbl.setVisible(False)
            return

        pt = self._plot_widget.getPlotItem().vb.mapSceneToView(pos)
        x, y = pt.x(), pt.y()

        # Horizontal line + price label
        self._hline.setPos(y)
        self._hline.setVisible(True)
        self._price_lbl.setText(f"{y:.2f}")
        xlo, xhi = self._plot_widget.getPlotItem().vb.viewRange()[0]
        ylo, yhi = self._plot_widget.getPlotItem().vb.viewRange()[1]
        self._price_lbl.setPos(xlo + (xhi - xlo) * 0.005, y)
        self._price_lbl.setVisible(True)

        # Vertical line + time label (local mouse — independent of main chart sync)
        self._vline.setPos(x)
        self._vline.setVisible(True)
        col = int(round(x))
        if 0 <= col < len(self._col_ts):
            ts_str = self._col_ts[col].strftime("%H:%M:%S")
        else:
            ts_str = ""
        self._time_lbl.setText(ts_str)
        self._time_lbl.setPos(x, ylo + (yhi - ylo) * 0.02)
        self._time_lbl.setVisible(bool(ts_str))

    def _on_tick(self) -> None:
        """Kick off a background snapshot query; skip if one is already running."""
        if not self._code:
            return
        if self._worker is not None and self._worker.isRunning():
            return   # previous query still in flight — skip this tick
        self._worker = _SnapshotWorker(self._code)
        self._worker.done.connect(self._on_snap_ready)
        self._worker.start()

    def _on_snap_ready(self, snap: list) -> None:
        """Called in main thread when background query completes."""
        if not snap:
            return
        self._maybe_init_price_range(snap)
        self._push_column(snap)
        self._render()
        self._redraw_orderflow_markers()

    def _maybe_init_price_range(self, snap: list[dict]) -> None:
        prices = [r["price"] for r in snap]
        if not prices:
            return
        lo  = min(prices)
        hi  = max(prices)
        mid = (lo + hi) / 2
        span = max(hi - lo, mid * 0.02)   # at least ±1% of mid price
        new_min = lo  - span * 0.5
        new_max = hi  + span * 0.5

        if self._bin_size == 0.0:
            self._price_min = new_min
            self._price_max = new_max
            self._bin_size  = (new_max - new_min) / N_PRICE
        else:
            rng = self._price_max - self._price_min
            out_lo = (self._price_min - lo) / rng
            out_hi = (hi - self._price_max) / rng
            if out_lo > 0.3 or out_hi > 0.3:
                self._price_min = new_min
                self._price_max = new_max
                self._bin_size  = (new_max - new_min) / N_PRICE
                max_cols = self._max_cols_spin.value()
                self._bid_grid  = np.zeros((max_cols, N_PRICE), dtype=np.float64)
                self._ask_grid  = np.zeros((max_cols, N_PRICE), dtype=np.float64)
                self._col_ts    = []
                self._raw_snaps = []

    def _push_column(self, snap: list[dict]) -> None:
        """Append one OB snapshot as the rightmost column, rolling if full."""
        min_vol  = self._min_vol_spin.value()
        max_cols = self._max_cols_spin.value()

        if len(self._col_ts) >= max_cols:
            self._bid_grid = np.roll(self._bid_grid, -1, axis=0)
            self._ask_grid = np.roll(self._ask_grid, -1, axis=0)
            self._bid_grid[-1] = 0.0
            self._ask_grid[-1] = 0.0
            self._col_ts.pop(0)
            # Trim raw_snaps to match the new oldest column time
            if self._col_ts:
                cutoff = self._col_ts[0]
                self._raw_snaps = [s for s in self._raw_snaps if s["ts"] >= cutoff]

        col = len(self._col_ts)
        now = datetime.now()
        self._col_ts.append(now)

        for row in snap:
            if row["volume"] < min_vol:
                continue
            p_bin = int((row["price"] - self._price_min) / self._bin_size)
            if not (0 <= p_bin < N_PRICE):
                continue
            if row["side"] == "BID":
                self._bid_grid[col, p_bin] = row["volume"]
            else:
                self._ask_grid[col, p_bin] = row["volume"]

        # Track best bid / ask (盘口) from this snapshot
        bid_prices = [r["price"] for r in snap if r["side"] == "BID"]
        ask_prices = [r["price"] for r in snap if r["side"] == "ASK"]
        self._best_bid = max(bid_prices) if bid_prices else self._best_bid
        self._best_ask = min(ask_prices) if ask_prices else self._best_ask

        # Append raw snaps (with wall-clock ts, not DB ts) for detection
        for row in snap:
            self._raw_snaps.append({
                "ts":     now,
                "side":   row["side"],
                "price":  row["price"],
                "volume": row["volume"],
            })

    def _render(self) -> None:
        n = len(self._col_ts)
        if n == 0 or self._bin_size == 0.0:
            return

        rect     = QRectF(0.0, self._price_min,
                          float(n), self._price_max - self._price_min)
        bid_view = self._bid_grid[:n]
        ask_view = self._ask_grid[:n]

        if self._bid_ask_cb.isChecked():
            self._img_combined.setVisible(False)
            for img, grid, color in (
                (self._img_bid, bid_view, _TEAL),
                (self._img_ask, ask_view, _RED),
            ):
                rgba = _single_rgba(grid, color)
                if rgba is not None:
                    img.setImage(rgba)
                    img.setRect(rect)
                    img.setVisible(True)
                else:
                    img.setVisible(False)
        else:
            self._img_bid.setVisible(False)
            self._img_ask.setVisible(False)
            rgba = _hot_rgba(bid_view + ask_view)
            if rgba is not None:
                self._img_combined.setImage(rgba)
                self._img_combined.setRect(rect)
                self._img_combined.setVisible(True)
            else:
                self._img_combined.setVisible(False)

        self._time_axis.update_timestamps(self._col_ts)
        step = max(1, n // 10)
        self._plot_widget.getPlotItem().getAxis("bottom").setTicks(
            [[(i, self._col_ts[i].strftime("%H:%M")) for i in range(0, n, step)]]
        )

        if n == 1:
            self._plot_widget.setXRange(0, self._max_cols_spin.value(), padding=0)
            self._plot_widget.setYRange(self._price_min, self._price_max, padding=0.02)

        # Update best bid/ask spread lines
        if self._best_bid is not None:
            self._bid_line.setValue(self._best_bid)
            self._bid_line.setVisible(True)
        if self._best_ask is not None:
            self._ask_line.setValue(self._best_ask)
            self._ask_line.setVisible(True)

    # ── iceberg / spoof detection and drawing ──────────────────────────────────

    def _build_bucket_to_idx(self) -> tuple[dict, int]:
        """Build candle_start → column_index mapping from col_ts.

        Returns (bucket_to_idx, cm) where cm is the effective candle duration.
        Uses 1-minute buckets (col_secs < 60) or col_secs//60 minute buckets.
        """
        col_secs = self._col_secs_spin.value()
        cm = max(1, col_secs // 60)
        bucket_to_idx: dict[datetime, int] = {}
        for i, ts in enumerate(self._col_ts):
            bk = candle_start(ts, cm)
            if bk not in bucket_to_idx:
                bucket_to_idx[bk] = i
        return bucket_to_idx, cm

    def _redraw_orderflow_markers(self) -> None:
        self._clear_overlay_items()
        if not self._raw_snaps or self._bin_size == 0.0:
            return

        bucket_to_idx, cm = self._build_bucket_to_idx()
        min_vol = self._min_vol_spin.value()

        if self._ice_cb.isChecked():
            from analysis.orderflow_detect import detect_icebergs
            icebergs = detect_icebergs(
                self._raw_snaps, bucket_to_idx, self._bin_size, self._price_min,
                N_PRICE, cm,
                min_refreshes=self._ice_min_ref_spin.value(),
                vol_threshold=min_vol,
                best_bid=self._best_bid,
                best_ask=self._best_ask,
                col_secs=self._col_secs_spin.value(),
            )
            self._draw_iceberg_markers(icebergs)

        if self._spoof_cb.isChecked():
            from analysis.orderflow_detect import detect_spoofs
            spoofs = detect_spoofs(
                self._raw_snaps,
                bucket_to_idx, self._bin_size, self._price_min,
                N_PRICE, cm,
                min_vol=float(min_vol),   # 0 → auto median of latest snapshot
                max_duration_secs=self._spoof_dur_spin.value(),
            )
            self._draw_spoof_markers(spoofs)

    def _draw_iceberg_markers(self, icebergs: list[tuple]) -> None:
        """Cyan horizontal line segments at detected iceberg price levels.

        Brightness encodes refresh intensity (more refreshes = brighter).
        """
        if not icebergs:
            return
        max_ref = max(ice[3] for ice in icebergs)
        N_TIERS = 5
        tier_xs: dict[int, list] = {t: [] for t in range(N_TIERS)}
        tier_ys: dict[int, list] = {t: [] for t in range(N_TIERS)}
        for (bar_start, bar_end, price, n_ref) in icebergs:
            tier = min(int((n_ref - 1) / max(max_ref, 1) * N_TIERS), N_TIERS - 1)
            tier_xs[tier] += [float(bar_start), float(bar_end) + 0.9]
            tier_ys[tier] += [price, price]

        for tier in range(N_TIERS):
            if not tier_xs[tier]:
                continue
            alpha = int(70 + 185 * tier / max(N_TIERS - 1, 1))
            pen   = pg.mkPen(color=(0, 229, 255, alpha), width=2)
            item  = pg.PlotCurveItem(
                x=np.array(tier_xs[tier]),
                y=np.array(tier_ys[tier]),
                pen=pen, connect="pairs",
            )
            item.setZValue(10)
            self._plot_widget.addItem(item)
            self._iceberg_items.append(item)

    def _draw_spoof_markers(self, spoofs: list[tuple]) -> None:
        """Orange ▲/▼ triangles at detected spoof events + dotted duration line."""
        if not spoofs:
            return
        ORANGE = (255, 140, 0, 210)
        up_xs, up_ys     = [], []
        down_xs, down_ys = [], []
        line_xs, line_ys = [], []

        for (appear_bar, disappear_bar, price, side) in spoofs:
            if side == "BID":
                up_xs.append(float(appear_bar))
                up_ys.append(price)
            else:
                down_xs.append(float(appear_bar))
                down_ys.append(price)
            line_xs += [float(appear_bar), float(disappear_bar) + 0.9]
            line_ys += [price, price]

        outline = pg.mkPen("white", width=1)
        if up_xs:
            scat = pg.ScatterPlotItem(
                x=up_xs, y=up_ys,
                symbol="t", size=15,
                pen=outline,
                brush=pg.mkBrush(*ORANGE),
            )
            scat.setZValue(11)
            self._plot_widget.addItem(scat)
            self._spoof_items.append(scat)

        if down_xs:
            scat = pg.ScatterPlotItem(
                x=down_xs, y=down_ys,
                symbol="t2", size=15,
                pen=outline,
                brush=pg.mkBrush(*ORANGE),
            )
            scat.setZValue(11)
            self._plot_widget.addItem(scat)
            self._spoof_items.append(scat)

        if line_xs:
            line = pg.PlotCurveItem(
                x=np.array(line_xs), y=np.array(line_ys),
                pen=pg.mkPen(color=ORANGE, width=1,
                             style=Qt.PenStyle.DotLine),
                connect="pairs",
            )
            line.setZValue(10)
            self._plot_widget.addItem(line)
            self._spoof_items.append(line)

    def _clear_overlay_items(self) -> None:
        for item in self._iceberg_items + self._spoof_items:
            self._plot_widget.removeItem(item)
        self._iceberg_items.clear()
        self._spoof_items.clear()

    def _update_legend(self) -> None:
        n = 5
        max_a = 180 / 255
        parts: list[str] = []

        # Colormap section — use <font> tags (safest Qt HTML subset)
        if self._bid_ask_cb.isChecked():
            teal = (38, 166, 154)
            red  = (239, 83, 80)
            t_cells = "".join(
                f'<font color="{_blend(teal, max_a*(i+1)/n)}">■</font>'
                for i in range(n))
            r_cells = "".join(
                f'<font color="{_blend(red, max_a*(i+1)/n)}">■</font>'
                for i in range(n))
            parts.append(
                f'{t_cells}&nbsp;<font color="{_TEAL}">Bid</font>'
                f'&nbsp;&nbsp;'
                f'{r_cells}&nbsp;<font color="{_RED}">Ask</font>'
                f'&nbsp;&nbsp;<font color="{_FG}">Lo→Hi</font>'
            )
        else:
            # #333 instead of #111 so the "Empty" square is faintly visible
            hot = ["#333333", "#300080", "#6010c8", "#ffa726", "#ffff00"]
            cells = "".join(f'<font color="{c}">■</font>' for c in hot)
            parts.append(
                f'{cells}&nbsp;<font color="{_FG}">Empty→Low→Med→High→Peak</font>'
            )

        # Best bid / ask spread
        parts.append(
            f'&nbsp;&nbsp;<font color="{_TEAL}">-- Best Bid</font>'
            f'&nbsp;&nbsp;<font color="{_RED}">-- Best Ask</font>'
        )

        if self._ice_cb.isChecked():
            parts.append(f'&nbsp;&nbsp;<font color="#00e5ff">-- Iceberg</font>')

        if self._spoof_cb.isChecked():
            parts.append(
                f'&nbsp;&nbsp;<font color="orange">▲ Bid spoof&nbsp;▼ Ask spoof</font>'
            )

        self._legend_lbl.setText("".join(parts))

    # ── control callbacks ──────────────────────────────────────────────────────

    def _on_controls_changed(self) -> None:
        self._update_legend()
        self._render()
        self._redraw_orderflow_markers()

    def _on_col_secs_changed(self) -> None:
        if self._live and self._timer.isActive():
            self._timer.start(self._col_secs_spin.value() * 1000)

    def _on_max_cols_changed(self) -> None:
        self._reset_grid()

    def _reset_view(self) -> None:
        """Restore X/Y ranges to the full data bounds (undo any zoom/pan)."""
        n = len(self._col_ts)
        if n == 0 or self._bin_size == 0.0:
            return
        self._plot_widget.setXRange(0, self._max_cols_spin.value(), padding=0)
        self._plot_widget.setYRange(self._price_min, self._price_max, padding=0.02)

    def mouseDoubleClickEvent(self, event) -> None:
        """Double-click anywhere on the window resets the zoom."""
        self._reset_view()
        super().mouseDoubleClickEvent(event)


# ── standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    w = LiqHmWindow()
    w.set_code("US.SOXL")
    w.set_live(True)
    w.show()
    sys.exit(app.exec())
