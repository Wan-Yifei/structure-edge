"""
Depth of Market (DOM) window — resting order book visualised as a bar chart.

Shows bid (teal) and ask (red) depth as vertical bars with price on the X
axis and volume on the Y axis.  Supports:
  - Configurable depth (10 / 20 / 30 / 50 levels per side)
  - Live mode: refreshes every second from order_book.db
  - Historical mode: pinned to a bar timestamp via pin_timestamp()
  - Hover tooltip: volume at hovered level + cumulative volume from best
    price to that level (how much to "eat through" the book)

Usage (standalone):
    uv run analysis/dom_window.py --code US.SNDK
"""

from __future__ import annotations

import pathlib
import sqlite3
import sys
from datetime import datetime

import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox,
)

# ── Colour palette (shared with trade_viewer_qt) ──────────────────────────────
_BG     = "#1a1a2e"
_BG_BAR = "#16213e"
_FG     = "#e0e0e0"
_GREY   = "#546e7a"
_BID    = (38,  166, 154, 200)   # teal
_ASK    = (239,  83,  80, 200)   # red
_BID_LINE = "#26a69a"
_ASK_LINE = "#ef5350"

_DB_PATH = pathlib.Path(__file__).parent.parent / "db" / "order_book.db"

# ── Helpers ───────────────────────────────────────────────────────────────────

def _query_latest_snapshot(code: str, db_path: pathlib.Path) -> list[dict]:
    """Return the most-recent full order book snapshot rows for `code`."""
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
        latest_ts = row[0]
        cur = con.execute(
            "SELECT ts, side, price, volume FROM order_book_snapshots "
            "WHERE code = ? AND ts = ?",
            [code, latest_ts],
        )
        result = [
            {"ts": datetime.fromisoformat(r[0]),
             "side": r[1], "price": r[2], "volume": r[3]}
            for r in cur.fetchall()
        ]
        con.close()
        return result
    except Exception:
        return []


def _query_snapshot_at(code: str, ts: datetime, db_path: pathlib.Path) -> list[dict]:
    """Return the most-recent snapshot at or before `ts` for `code`."""
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
        snap_ts = row[0]
        cur = con.execute(
            "SELECT ts, side, price, volume FROM order_book_snapshots "
            "WHERE code = ? AND ts = ?",
            [code, snap_ts],
        )
        result = [
            {"ts": datetime.fromisoformat(r[0]),
             "side": r[1], "price": r[2], "volume": r[3]}
            for r in cur.fetchall()
        ]
        con.close()
        return result
    except Exception:
        return []


def _tick_size(prices: list[float]) -> float:
    """Infer minimum tick size from a sorted list of price levels."""
    if len(prices) < 2:
        return 0.01
    diffs = [prices[i + 1] - prices[i] for i in range(len(prices) - 1) if prices[i + 1] != prices[i]]
    return min(diffs) if diffs else 0.01


# ── DOM Window ────────────────────────────────────────────────────────────────

class DomWindow(QWidget):
    """Depth-of-Market chart window (resting order book bar chart).

    Instantiated and shown by TradeViewerQt via the DOM toolbar button.
    Can also run standalone via __main__.
    """

    _DEPTH_CHOICES = [10, 20, 30, 50]
    _REFRESH_MS    = 1000

    def __init__(self, code: str = "US.SNDK", live: bool = True,
                 db_path: pathlib.Path | None = None):
        super().__init__()
        self._code       = code
        self._live       = live
        self._db_path    = db_path or _DB_PATH
        self._depth      = 10
        self._pinned_ts: datetime | None = None

        # Cached data for hover hit-testing
        self._bid_prices: list[float] = []
        self._bid_vols:   list[int]   = []
        self._ask_prices: list[float] = []
        self._ask_vols:   list[int]   = []
        self._tick:       float       = 0.01

        self.setWindowTitle(f"DOM — {code}")
        self.resize(760, 400)
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

        # Toolbar row
        bar = QHBoxLayout()
        bar.setSpacing(10)

        self._code_lbl = QLabel(f"Code: {self._code}")
        f = QFont(); f.setBold(True)
        self._code_lbl.setFont(f)
        bar.addWidget(self._code_lbl)

        bar.addSpacing(8)
        bar.addWidget(QLabel("Depth:"))

        self._depth_combo = QComboBox()
        self._depth_combo.addItems([str(d) for d in self._DEPTH_CHOICES])
        self._depth_combo.setCurrentText(str(self._depth))
        self._depth_combo.setFixedWidth(60)
        self._depth_combo.currentTextChanged.connect(self._on_depth_changed)
        bar.addWidget(self._depth_combo)

        bar.addStretch()

        self._ts_lbl = QLabel("")
        self._ts_lbl.setStyleSheet("font-size: 10px;")
        bar.addWidget(self._ts_lbl)

        root.addLayout(bar)

        # Chart
        self._pw = pg.PlotWidget(background=_BG)
        self._pw.setAntialiasing(True)
        pi = self._pw.getPlotItem()
        pi.showGrid(x=False, y=True, alpha=0.15)
        pi.getAxis("bottom").setTextPen(QColor(_FG))
        pi.getAxis("left").setTextPen(QColor(_FG))
        pi.setLabel("bottom", "Price", **{"color": _FG})
        pi.setLabel("left",   "Volume", **{"color": _FG})
        pi.setMenuEnabled(False)

        # Bid bars
        self._bid_bars = pg.BarGraphItem(
            x=[], height=[], width=0.01,
            brush=pg.mkBrush(_BID), pen=pg.mkPen(None),
        )
        # Ask bars
        self._ask_bars = pg.BarGraphItem(
            x=[], height=[], width=0.01,
            brush=pg.mkBrush(_ASK), pen=pg.mkPen(None),
        )
        self._pw.addItem(self._bid_bars)
        self._pw.addItem(self._ask_bars)

        # Best bid / ask vertical marker lines
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

        # Hover proxy
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
            QWidget  {{ background: {_BG}; color: {_FG}; }}
            QLabel   {{ background: transparent; color: {_FG}; }}
            QComboBox {{
                background: {_BG_BAR}; color: {_FG};
                border: 1px solid {_GREY}; padding: 2px 4px;
            }}
        """)

    # ── Public API (called by TradeViewerQt) ──────────────────────────────────

    def set_code(self, code: str) -> None:
        """Update the symbol and force a refresh."""
        if code == self._code:
            return
        self._code = code
        self._code_lbl.setText(f"Code: {code}")
        self.setWindowTitle(f"DOM — {code}")
        self._refresh()

    def set_live(self, live: bool) -> None:
        """Switch between live (auto-refresh) and historical (pinned) mode."""
        self._live = live
        if live:
            self._timer.start(self._REFRESH_MS)
        else:
            self._timer.stop()

    def pin_timestamp(self, ts: datetime | None) -> None:
        """Display the order book snapshot at or just before `ts`."""
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

    def _refresh(self) -> None:
        if self._live or self._pinned_ts is None:
            rows = _query_latest_snapshot(self._code, self._db_path)
        else:
            rows = _query_snapshot_at(self._code, self._pinned_ts, self._db_path)
        self._render(rows)

    def _render(self, rows: list[dict]) -> None:
        if not rows:
            self._bid_bars.setOpts(x=[], height=[], width=0.01)
            self._ask_bars.setOpts(x=[], height=[], width=0.01)
            self._best_bid_line.setVisible(False)
            self._best_ask_line.setVisible(False)
            self._bid_prices = []; self._bid_vols = []
            self._ask_prices = []; self._ask_vols = []
            self._status_lbl.setText(
                f"No data for {self._code} — is order_book_collector.py running?")
            return

        # Split and sort: bids descending (best at index 0), asks ascending
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

        snap_ts = rows[0]["ts"]
        self._ts_lbl.setText(str(snap_ts)[:19])

        bid_prices = [b[0] for b in bids]
        bid_vols   = [b[1] for b in bids]
        ask_prices = [a[0] for a in asks]
        ask_vols   = [a[1] for a in asks]

        self._bid_prices = bid_prices
        self._bid_vols   = bid_vols
        self._ask_prices = ask_prices
        self._ask_vols   = ask_vols

        # Infer tick size for bar width
        all_sorted = sorted(set(bid_prices + ask_prices))
        self._tick  = _tick_size(all_sorted)
        bar_width   = self._tick * 0.8

        self._bid_bars.setOpts(
            x=bid_prices, height=bid_vols, width=bar_width,
            brush=pg.mkBrush(_BID), pen=pg.mkPen(None),
        )
        self._ask_bars.setOpts(
            x=ask_prices, height=ask_vols, width=bar_width,
            brush=pg.mkBrush(_ASK), pen=pg.mkPen(None),
        )

        if bid_prices:
            self._best_bid_line.setPos(bid_prices[0])
            self._best_bid_line.setVisible(True)
        if ask_prices:
            self._best_ask_line.setPos(ask_prices[0])
            self._best_ask_line.setVisible(True)

        # Status bar summary
        total_bid = sum(bid_vols)
        total_ask = sum(ask_vols)
        if bid_prices and ask_prices:
            spread = ask_prices[0] - bid_prices[0]
            self._status_lbl.setText(
                f"Bid {bid_prices[0]:.4f} ({total_bid:,})  "
                f"Ask {ask_prices[0]:.4f} ({total_ask:,})  "
                f"Spread {spread:.4f}"
            )
        else:
            self._status_lbl.setText("Data loaded.")

    # ── Hover tooltip ─────────────────────────────────────────────────────────

    def _on_hover(self, evt) -> None:
        pos = evt[0]
        if not self._pw.sceneBoundingRect().contains(pos):
            self._tooltip.setVisible(False)
            return
        pi = self._pw.getPlotItem()
        pt = pi.vb.mapSceneToView(pos)
        px = pt.x()
        py = pt.y()

        tip = self._build_tooltip(px)
        if tip is None:
            self._tooltip.setVisible(False)
            return

        xlo, xhi = pi.vb.viewRange()[0]
        ylo, yhi = pi.vb.viewRange()[1]
        # Clamp tooltip so it stays inside the view
        tx = min(px + self._tick * 0.5, xhi - (xhi - xlo) * 0.25)
        ty = min(py + (yhi - ylo) * 0.02, yhi - (yhi - ylo) * 0.05)
        self._tooltip.setPos(tx, ty)
        self._tooltip.setText(tip)
        self._tooltip.setVisible(True)

    def _build_tooltip(self, px: float) -> str | None:
        half = self._tick * 0.5
        # Check bid levels
        for i, (p, v) in enumerate(zip(self._bid_prices, self._bid_vols)):
            if abs(px - p) <= half:
                cum = sum(self._bid_vols[:i + 1])
                best = self._bid_prices[0]
                return (
                    f"BID  {p:.4f}\n"
                    f"Volume : {v:,}\n"
                    f"Cum from best ({best:.4f}): {cum:,}\n"
                    f"Levels: {i + 1}"
                )
        # Check ask levels
        for i, (p, v) in enumerate(zip(self._ask_prices, self._ask_vols)):
            if abs(px - p) <= half:
                cum = sum(self._ask_vols[:i + 1])
                best = self._ask_prices[0]
                return (
                    f"ASK  {p:.4f}\n"
                    f"Volume : {v:,}\n"
                    f"Cum from best ({best:.4f}): {cum:,}\n"
                    f"Levels: {i + 1}"
                )
        return None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        self._timer.stop()
        super().closeEvent(event)


# ── Standalone entry point ────────────────────────────────────────────────────

def _main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="DOM window (standalone)")
    parser.add_argument("--code", default="US.SNDK")
    parser.add_argument("--db",   default=None, help="Path to order_book.db")
    args = parser.parse_args()

    db = pathlib.Path(args.db) if args.db else None
    app = QApplication(sys.argv)
    win = DomWindow(code=args.code, live=True, db_path=db)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    _main()
