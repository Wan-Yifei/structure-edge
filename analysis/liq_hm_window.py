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
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")

from PyQt6.QtCore    import Qt, QRectF, QTimer, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui     import QColor, QPainterPath
from PyQt6.QtWidgets import (
    QCheckBox, QDoubleSpinBox, QLabel, QPushButton,
    QSpinBox, QToolBar, QVBoxLayout, QWidget,
)
import pyqtgraph as pg

# ── palette (matches trade_viewer_qt) ──────────────────────────────────────────
_BG   = "#0d1117"
_FG   = "#b0bec5"
_GRID = "#263238"
_TEAL = "#26a69a"   # bid side (also bull color in green-up / Western convention)
_RED  = "#ef5350"   # ask side (also bull color in red-up / CN convention)

_DB_PATH = pathlib.Path(__file__).parent.parent / "db" / "order_book.db"

# Spoof marker symbols — explicit QPainterPath so orientation is version-agnostic.
# Qt painter Y increases downward: y=-0.5 is visually top, y=+0.5 is visually bottom.
def _make_triangle(up: bool) -> QPainterPath:
    path = QPainterPath()
    if up:   # ▲  apex at top
        path.moveTo( 0.0, -0.5)
        path.lineTo( 0.5,  0.5)
        path.lineTo(-0.5,  0.5)
    else:    # ▼  apex at bottom
        path.moveTo( 0.0,  0.5)
        path.lineTo( 0.5, -0.5)
        path.lineTo(-0.5, -0.5)
    path.closeSubpath()
    return path

_SYM_UP   = _make_triangle(True)   # bid spoof  ▲
_SYM_DOWN = _make_triangle(False)  # ask spoof  ▼

N_PRICE      = 100   # price bins (y-resolution)
COL_SECS_DEF = 1     # seconds per column -- ORDER_BOOK is push-driven, not
                      # polled, so 1s keeps the heatmap visually current;
                      # adjustable via the "Col(s)" spinbox (range 1-300)
MAX_COLS_DEF = 240   # columns kept in memory  (4 min at 1 s/col -- raise via
                      # the "History" spinbox, range 60-1440, for more
                      # lookback at the cost of a wider/denser grid)
_NEAR_TOUCH_MIN_LEVELS = 5    # always keep at least this many levels/side,
                              # even if the very first gap looks large
_NEAR_TOUCH_MAX_LEVELS = 60   # hard cap (a full L2 snapshot's per-side depth)
_NEAR_TOUCH_MAD_MULT   = 8.0  # gap must exceed median + this many MADs to
                              # count as an outlier -- generous on purpose,
                              # this only needs to catch a level that's
                              # dramatically farther out than its neighbours
                              # (e.g. a stub quote $65 away), not flag normal
                              # unevenness in real resting-depth spacing
_WINDOW_SHRINK_FACTOR  = 0.5  # rebuild (tighten) if the freshly-computed
                              # near-touch band is narrower than this
                              # fraction of the current window -- otherwise
                              # a window only ever grows: once an early wide
                              # snapshot (e.g. a stub order that has since
                              # been cancelled) sets a wide band, the book
                              # can stay genuinely tight forever after while
                              # price never drifts close enough to the old
                              # band's edges to trigger the grow-only checks
                              # below, leaving most of the view permanently
                              # empty around a thin sliver of real depth


# ── data helpers ───────────────────────────────────────────────────────────────

def _query_latest_snapshot(code: str) -> list[dict]:
    """Return all rows from the most recent OB push for *code*.

    Row count depends on the account's quote depth entitlement (e.g. up to
    60 BID + 60 ASK for US LV2) -- no LIMIT is applied here, so whatever the
    collector stored for that timestamp comes back in full.

    Uses a plain sqlite3 connection (no URI mode) to avoid Windows path issues.
    ts is included so callers can use it for iceberg/spoof detection.
    """
    if not _DB_PATH.exists():
        return []
    con = None
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
        return rows
    except Exception:
        return []
    finally:
        if con is not None:
            con.close()


def _query_n_snapshots(code: str, n: int) -> list[list[dict]]:
    """Return the last *n* distinct OB snapshots for *code*, oldest first.

    Each inner list contains all rows for one timestamp (one full book state).
    """
    if not _DB_PATH.exists():
        return []
    con = None
    try:
        con = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        cur = con.execute(
            "SELECT DISTINCT ts FROM order_book_snapshots "
            "WHERE code = ? ORDER BY ts DESC LIMIT ?",
            [code, n],
        )
        ts_list = [r[0] for r in cur.fetchall()][::-1]  # reverse: oldest first
        snapshots: list[list[dict]] = []
        for ts_str in ts_list:
            cur2 = con.execute(
                "SELECT ts, side, price, volume FROM order_book_snapshots "
                "WHERE code = ? AND ts = ?",
                [code, ts_str],
            )
            rows = [
                {"ts": datetime.fromisoformat(r[0]), "side": r[1],
                 "price": float(r[2]), "volume": float(r[3])}
                for r in cur2.fetchall()
            ]
            if rows:
                snapshots.append(rows)
        return snapshots
    except Exception:
        return []
    finally:
        if con is not None:
            con.close()


_TICK_DB_PATH = pathlib.Path(__file__).parent.parent / "db" / "ticks.db"


def _query_ticks(code: str, start: datetime, end: datetime) -> list[dict]:
    """Return tick records for *code* in [start, end) from ticks.db."""
    if not _TICK_DB_PATH.exists():
        return []
    try:
        from feeds.tick_store import TickStore
        store = TickStore(_TICK_DB_PATH, read_only=True)
        rows  = store.query_ticks(code, start, end)
        store.close()
        return rows
    except Exception as exc:
        print(f"[QueryTicks] ERROR: {exc}", flush=True)
        return []


# ── background query workers ──────────────────────────────────────────────────

class _AbsorbTickWorker(QThread):
    """Load ticks for the current display window in a background thread."""
    done = pyqtSignal(list)   # emits list[dict]

    def __init__(self, code: str, start: datetime, end: datetime) -> None:
        super().__init__()
        self._code  = code
        self._start = start
        self._end   = end

    def run(self) -> None:
        self.done.emit(_query_ticks(self._code, self._start, self._end))


class _SnapshotWorker(QThread):
    """Fetch the latest OB snapshot in a background thread to avoid UI stalls."""
    done = pyqtSignal(list)   # emits list[dict] (may be empty)

    def __init__(self, code: str) -> None:
        super().__init__()
        self._code = code

    def run(self) -> None:
        self.done.emit(_query_latest_snapshot(self._code))


class _BulkSnapshotWorker(QThread):
    """Fetch the last N distinct OB snapshots in a background thread."""
    done = pyqtSignal(list)   # emits list[list[dict]]

    def __init__(self, code: str, n: int) -> None:
        super().__init__()
        self._code = code
        self._n    = n

    def run(self) -> None:
        self.done.emit(_query_n_snapshots(self._code, self._n))


# Keep-alive list for workers whose owner discarded them while still running.
_PENDING_WORKERS: list[QThread] = []


def _retire_worker(worker: QThread) -> None:
    """Detach a worker whose result is no longer wanted.

    Dropping the last Python reference to a QThread while it is still
    running calls its C++ destructor on a live thread, which makes Qt abort
    the process with "QThread: Destroyed while thread is still running".
    These workers run one blocking read with no event loop, so they always
    finish on their own shortly; just disconnect the signal (so no stale
    callback touches a dead window) and, if still running, hold a reference
    until `finished` confirms the OS thread actually exited.
    """
    try:
        worker.done.disconnect()
    except Exception:
        pass
    if worker.isRunning():
        _PENDING_WORKERS.append(worker)
        worker.finished.connect(lambda: _PENDING_WORKERS.remove(worker))


# ── color renderers ────────────────────────────────────────────────────────────

_COLOR_NORM_PCT = 99.0   # brightness reference = this percentile, not the max


def _percentile_norm(log_g: np.ndarray, pct: float = _COLOR_NORM_PCT) -> np.ndarray:
    """Normalize log-volume to [0, 1], clipped, using a high percentile of the
    nonzero values as the "100% bright" reference instead of the single max.

    Normalizing against grid.max() means one outlier block (a big resting
    order, an iceberg, a large snapshot elsewhere in the loaded history)
    sets the brightness ceiling for the *entire* heatmap -- ordinary top-of-
    book size near the current touch is real and present in the data but
    renders too dim to see next to it (reported: "no orders near the touch
    but the order book shows them"). The 99th percentile is still driven by
    genuinely large levels, just not by a single extreme one; anything above
    it clips to full brightness rather than desaturating everything below.
    """
    nonzero = log_g[log_g > 0]
    if nonzero.size == 0:
        return log_g
    ref = float(np.percentile(nonzero, pct))
    if ref <= 0:
        ref = float(log_g.max())
    return np.clip(log_g / ref, 0.0, 1.0).astype(np.float32)


def _hot_rgba(grid: np.ndarray, gamma: float = 1.0) -> np.ndarray | None:
    """Combined bid+ask: black → purple → amber → yellow.

    gamma > 1 suppresses midtones — only the densest zones remain bright.
    gamma < 1 boosts midtones — weaker zones become more visible.
    """
    if grid.max() <= 0:
        return None
    log_g = np.log1p(grid)
    norm  = _percentile_norm(log_g)
    if gamma != 1.0:
        np.power(norm, gamma, out=norm)
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


def _single_rgba(grid: np.ndarray, hex_color: str,
                 gamma: float = 1.0) -> np.ndarray | None:
    """Single-hue intensity map for bid or ask separately."""
    if grid.max() <= 0:
        return None
    log_g = np.log1p(grid)
    norm  = _percentile_norm(log_g)
    if gamma != 1.0:
        np.power(norm, gamma, out=norm)
    c     = QColor(hex_color)
    rgba  = np.zeros((*norm.shape, 4), dtype=np.uint8)
    rgba[..., 0] = c.red()
    rgba[..., 1] = c.green()
    rgba[..., 2] = c.blue()
    rgba[..., 3]         = np.clip((norm * 180).astype(np.uint8), 0, 180)
    rgba[norm < 0.02, 3] = 0
    return rgba


def _near_touch_cutoff(
    prices_from_touch: list[float],
    min_keep: int = _NEAR_TOUCH_MIN_LEVELS,
    max_keep: int = _NEAR_TOUCH_MAX_LEVELS,
    mad_mult: float = _NEAR_TOUCH_MAD_MULT,
) -> list[float]:
    """Keep consecutive depth levels outward from the touch until a gap that's
    a clear outlier relative to the typical inter-level spacing seen so far.

    prices_from_touch: one side's levels, nearest-to-touch first (bids sorted
    descending or asks sorted ascending). Returns a prefix of that list --
    everything from the first outlier gap onward is dropped.

    Adaptive alternative to a fixed level count: a tightly-clustered book
    keeps more levels (more of the real depth is visible), a book with a
    stub/block order sitting far from the touch cuts off right before it,
    regardless of what position that happened to be at. Median + MAD (not
    mean/stddev) because a single huge outlier gap would otherwise blow out
    the very average it's being compared against.

    O(n) on <=60 levels, called only when the price band is first set or
    rebuilt (not on every tick) -- negligible cost either way.
    """
    n = len(prices_from_touch)
    if n <= min_keep:
        return prices_from_touch
    gaps = np.abs(np.diff(prices_from_touch))
    keep = min_keep
    for i in range(min_keep - 1, len(gaps)):
        history = gaps[:i + 1]
        med = float(np.median(history))
        mad = float(np.median(np.abs(history - med)))
        threshold = med + mad_mult * mad if mad > 0 else max(med * 5.0, 0.05)
        if gaps[i] > threshold:
            break
        keep = i + 2   # i+1 gaps examined so far -> i+2 prices included
    return prices_from_touch[:min(keep, max_keep)]


def _calc_col_mid(snap: list[dict]) -> float | None:
    """Compute mid-price from one OB snapshot.  Returns None when snap is empty."""
    bid_prices = [r["price"] for r in snap if r["side"] == "BID"]
    ask_prices = [r["price"] for r in snap if r["side"] == "ASK"]
    col_bid = max(bid_prices) if bid_prices else None
    col_ask = min(ask_prices) if ask_prices else None
    if col_bid is not None and col_ask is not None:
        return (col_bid + col_ask) / 2.0
    return col_bid if col_bid is not None else col_ask


def _calc_depth_label(snap: list[dict],
                      best_bid: float, best_ask: float,
                      target: float) -> str:
    """Return depth-to-cursor annotation string.

    '[spread]' when target is inside the spread.
    'eat↑ N'  when target is above best_ask (N = ask volume to consume).
    'eat↓ N'  when target is below best_bid (N = bid volume to consume).
    """
    if best_bid < target < best_ask:
        return "[spread]"
    if target >= best_ask:
        vol = sum(
            r["volume"] for r in snap
            if r["side"] == "ASK" and best_ask <= r["price"] <= target
        )
        return f"eat↑ {vol:,.0f}"
    vol = sum(
        r["volume"] for r in snap
        if r["side"] == "BID" and target <= r["price"] <= best_bid
    )
    return f"eat↓ {vol:,.0f}"


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
        self.resize(1000, 460)

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
        self._red_up: bool = True  # mirrors trade_viewer red-up convention

        # Per-column mid-price for the price path line (None = no data that column)
        self._mid_prices: list[float | None] = []

        # Live mid from the most recent update_quote() call — appended to path tail
        self._live_mid: float | None = None

        # Latest OB snapshot — cached for fast depth-to-cursor calculation
        self._latest_snap: list[dict] = []

        # Price range (auto-detected from first snapshot)
        self._price_min: float = 0.0
        self._price_max: float = 0.0
        self._bin_size:  float = 0.0

        # Overlay items managed on _plot_widget
        self._iceberg_items: list = []
        self._spoof_items:   list = []
        self._simb_items:    list = []
        self._absorb_items:  list = []

        # Tick cache + worker for absorption bubble overlay
        self._absorb_ticks:         list[dict]               = []
        self._absorb_worker:        _AbsorbTickWorker | None = None
        self._absorb_reload_pending: bool                    = False
        # Tracks the newest tick ts already in cache; None = need full reload.
        self._absorb_last_ts:       datetime | None          = None

        # Background query workers (one at a time each)
        self._worker:      _SnapshotWorker | None      = None
        self._bulk_worker: _BulkSnapshotWorker | None  = None
        # True after _reset_grid(); cleared after initial bulk pre-fill completes
        self._needs_init:  bool                        = False
        # Timestamp of the last snapshot pushed; used to skip duplicate polls
        self._last_snap_ts: datetime | None            = None

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

        # ── Row 1: core display controls ──────────────────────────────────────
        row1 = QToolBar()
        row1.setMovable(False)

        self._bid_ask_cb = QCheckBox("Bid/Ask")
        self._bid_ask_cb.setChecked(True)
        self._bid_ask_cb.setToolTip(
            "Checked: teal = Bid, red = Ask (shown separately)\n"
            "Unchecked: combined with black → purple → yellow colormap")
        self._bid_ask_cb.stateChanged.connect(self._on_controls_changed)
        row1.addWidget(self._bid_ask_cb)

        row1.addWidget(_lbl("Gamma:"))
        self._gamma_spin = QDoubleSpinBox()
        self._gamma_spin.setRange(0.2, 10.0)
        self._gamma_spin.setSingleStep(0.1)
        self._gamma_spin.setDecimals(1)
        self._gamma_spin.setValue(3.0)
        self._gamma_spin.setFixedWidth(52)
        self._gamma_spin.setToolTip(
            "Colour gamma correction.\n"
            ">1: suppresses sparse zones — only the densest orders stay bright.\n"
            "<1: boosts dim zones — reveals weaker order clusters.")
        self._gamma_spin.valueChanged.connect(self._on_gamma_changed)
        row1.addWidget(self._gamma_spin)

        self._price_path_cb = QCheckBox("Price")
        self._price_path_cb.setChecked(True)
        self._price_path_cb.setToolTip(
            "Overlay the mid-price path ((bid+ask)/2) per column as a white line.")
        self._price_path_cb.stateChanged.connect(self._on_price_path_changed)
        row1.addWidget(self._price_path_cb)

        row1.addSeparator()
        row1.addWidget(_lbl("Min.Vol:"))
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
        row1.addWidget(self._min_vol_spin)

        row1.addSeparator()
        row1.addWidget(_lbl("Col(s):"))
        self._col_secs_spin = QSpinBox()
        self._col_secs_spin.setRange(1, 300)
        self._col_secs_spin.setSingleStep(1)
        self._col_secs_spin.setValue(COL_SECS_DEF)
        self._col_secs_spin.setFixedWidth(55)
        self._col_secs_spin.setToolTip("Seconds per column — also controls refresh rate")
        self._col_secs_spin.valueChanged.connect(self._on_col_secs_changed)
        row1.addWidget(self._col_secs_spin)

        row1.addSeparator()
        row1.addWidget(_lbl("History:"))
        self._max_cols_spin = QSpinBox()
        self._max_cols_spin.setRange(60, 1440)
        self._max_cols_spin.setSingleStep(60)
        self._max_cols_spin.setValue(MAX_COLS_DEF)
        self._max_cols_spin.setFixedWidth(60)
        self._max_cols_spin.setToolTip("Number of columns kept in memory")
        self._max_cols_spin.valueChanged.connect(self._on_max_cols_changed)
        row1.addWidget(self._max_cols_spin)

        row1.addSeparator()
        row1.addWidget(_lbl("Bins:"))
        self._n_price_spin = QSpinBox()
        self._n_price_spin.setRange(20, 500)
        self._n_price_spin.setSingleStep(20)
        self._n_price_spin.setValue(N_PRICE)
        self._n_price_spin.setFixedWidth(55)
        self._n_price_spin.setToolTip(
            "Price-axis resolution: number of bands the current price span is\n"
            "divided into. Higher = finer bands (each one covers less price\n"
            "range) at the cost of thinner rows. Current $/band is shown in\n"
            "the legend below.")
        self._n_price_spin.valueChanged.connect(self._on_n_price_changed)
        row1.addWidget(self._n_price_spin)

        row1.addSeparator()
        reset_btn = QPushButton("⟲ Reset")
        reset_btn.setToolTip("Reset zoom to full view (double-click chart also resets)")
        reset_btn.setFixedWidth(70)
        reset_btn.clicked.connect(self._reset_view)
        row1.addWidget(reset_btn)

        row1.addSeparator()
        self._pin_btn = QPushButton("📌 Pin")
        self._pin_btn.setCheckable(True)
        self._pin_btn.setFixedWidth(60)
        self._pin_btn.setToolTip("Keep this window on top of all other windows")
        self._pin_btn.toggled.connect(self._on_pin_toggled)
        row1.addWidget(self._pin_btn)

        # ── Row 2: detection overlay controls ─────────────────────────────────
        row2 = QToolBar()
        row2.setMovable(False)

        self._ice_cb = QCheckBox("Iceberg")
        self._ice_cb.setChecked(False)
        self._ice_cb.setToolTip(
            "Purple line: price level where resting volume repeatedly drops then\n"
            "refreshes — hallmark of a hidden large order refilling at a fixed price.")
        self._ice_cb.stateChanged.connect(self._on_controls_changed)
        row2.addWidget(self._ice_cb)

        row2.addWidget(_lbl("Min.Ref:"))
        self._ice_min_ref_spin = QSpinBox()
        self._ice_min_ref_spin.setRange(1, 50)
        self._ice_min_ref_spin.setSingleStep(1)
        self._ice_min_ref_spin.setValue(3)
        self._ice_min_ref_spin.setFixedWidth(50)
        self._ice_min_ref_spin.setToolTip("Minimum refreshes to classify as iceberg")
        self._ice_min_ref_spin.valueChanged.connect(self._on_controls_changed)
        row2.addWidget(self._ice_min_ref_spin)

        row2.addSeparator()

        self._spoof_cb = QCheckBox("Spoof")
        self._spoof_cb.setChecked(False)
        self._spoof_cb.setToolTip(
            "Orange ▲/▼: large order that appears then vanishes without execution.\n"
            "▲ = bid spoof (false buy pressure)  ▼ = ask spoof (false sell pressure)")
        self._spoof_cb.stateChanged.connect(self._on_controls_changed)
        row2.addWidget(self._spoof_cb)

        row2.addWidget(_lbl("Max.Dur(s):"))
        self._spoof_dur_spin = QSpinBox()
        self._spoof_dur_spin.setRange(3, 300)
        self._spoof_dur_spin.setSingleStep(5)
        self._spoof_dur_spin.setValue(30)
        self._spoof_dur_spin.setFixedWidth(50)
        self._spoof_dur_spin.setToolTip("Max seconds a large order can live before being flagged")
        self._spoof_dur_spin.valueChanged.connect(self._on_controls_changed)
        row2.addWidget(self._spoof_dur_spin)

        row2.addSeparator()

        self._simb_cb = QCheckBox("Imbalance")
        self._simb_cb.setChecked(False)
        self._simb_cb.setToolTip(
            "Lime bar   = bullish stacked imbalance (bid dominates N consecutive depth levels)\n"
            "Pink bar   = bearish stacked imbalance (ask dominates N consecutive depth levels)\n"
            "Bid/ask levels paired by depth rank; missing side counts as 0.")
        self._simb_cb.stateChanged.connect(self._on_controls_changed)
        row2.addWidget(self._simb_cb)

        row2.addWidget(_lbl("Lvl:"))
        self._simb_levels_spin = QSpinBox()
        self._simb_levels_spin.setRange(2, 10)
        self._simb_levels_spin.setSingleStep(1)
        self._simb_levels_spin.setValue(3)
        self._simb_levels_spin.setFixedWidth(45)
        self._simb_levels_spin.setToolTip("Minimum consecutive imbalanced depth levels")
        self._simb_levels_spin.valueChanged.connect(self._on_controls_changed)
        row2.addWidget(self._simb_levels_spin)

        row2.addWidget(_lbl("Ratio:"))
        self._simb_ratio_spin = QDoubleSpinBox()
        self._simb_ratio_spin.setRange(1.5, 20.0)
        self._simb_ratio_spin.setSingleStep(0.5)
        self._simb_ratio_spin.setValue(3.0)
        self._simb_ratio_spin.setDecimals(1)
        self._simb_ratio_spin.setFixedWidth(52)
        self._simb_ratio_spin.setToolTip("bid/ask (or ask/bid) volume ratio threshold per level")
        self._simb_ratio_spin.valueChanged.connect(self._on_controls_changed)
        row2.addWidget(self._simb_ratio_spin)

        row2.addWidget(_lbl("Depth:"))
        self._simb_depth_spin = QSpinBox()
        self._simb_depth_spin.setRange(3, 50)
        self._simb_depth_spin.setSingleStep(1)
        self._simb_depth_spin.setValue(10)
        self._simb_depth_spin.setFixedWidth(45)
        self._simb_depth_spin.setToolTip(
            "Max depth ranks to analyse (top-of-book only).\n"
            "Deep orders far from the spread are noise — keep this at 5–15.")
        self._simb_depth_spin.valueChanged.connect(self._on_controls_changed)
        row2.addWidget(self._simb_depth_spin)

        row2.addSeparator()

        self._absorb_cb = QCheckBox("Aggressor")
        self._absorb_cb.setChecked(False)
        self._absorb_cb.setToolTip(
            "Gold bubble   = dominant BUY aggression (net buyers > threshold).\n"
            "Purple bubble = dominant SELL aggression (net sellers > threshold).\n"
            "Bubble size encodes net delta volume.  Reads ticks.db.\n"
            "Whether the flow was absorbed is for the user to judge from the heatmap.")
        self._absorb_cb.stateChanged.connect(self._on_absorb_changed)
        row2.addWidget(self._absorb_cb)

        row2.addWidget(_lbl("MinΔ:"))
        self._absorb_min_vol_spin = QSpinBox()
        self._absorb_min_vol_spin.setRange(10, 100_000)
        self._absorb_min_vol_spin.setSingleStep(10)
        self._absorb_min_vol_spin.setValue(500)
        self._absorb_min_vol_spin.setFixedWidth(70)
        self._absorb_min_vol_spin.setToolTip(
            "Minimum |buy_vol − sell_vol| per column to show a bubble.")
        self._absorb_min_vol_spin.valueChanged.connect(self._on_absorb_changed)
        row2.addWidget(self._absorb_min_vol_spin)

        # ── Two-row toolbar container ──────────────────────────────────────────
        self._toolbar = QWidget()
        tb_lay = QVBoxLayout(self._toolbar)
        tb_lay.setContentsMargins(0, 0, 0, 0)
        tb_lay.setSpacing(0)
        tb_lay.addWidget(row1)
        tb_lay.addWidget(row2)

        # row1/row2 are QToolBar instances used purely as a convenient
        # addWidget()/addSeparator() container here -- they were never docked
        # via addToolBar(), so Qt's native toolbar overflow chevron (which
        # only engages for toolbars managed by a QMainWindow toolbar area)
        # never applies. Without it, shrinking the window just clips the
        # widest row's trailing controls off the edge with no way to reach
        # them. Floor the window width at the widest row's natural size so
        # that state is never reachable.
        widest = max(row1.sizeHint().width(), row2.sizeHint().width())
        self.setMinimumWidth(widest + 16)

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

        _cross_pen = pg.mkPen("#ffffff", width=1, style=Qt.PenStyle.DashLine)

        # Vertical line — set by pin_timestamp() from main chart OR local mouse
        self._vline = pg.InfiniteLine(angle=90, movable=False, pen=_cross_pen)
        self._vline.setVisible(False)
        self._plot_widget.addItem(self._vline)

        # Horizontal line — follows mouse within this window
        self._hline = pg.InfiniteLine(angle=0, movable=False, pen=_cross_pen)
        self._hline.setVisible(False)
        self._plot_widget.addItem(self._hline)

        _lbl_fill = pg.mkBrush(QColor(13, 17, 23, 200))  # semi-transparent _BG

        # Price label anchored to left edge at cursor Y
        self._price_lbl = pg.TextItem(anchor=(0.0, 0.5),
                                      color="#ffffff", fill=_lbl_fill)
        self._price_lbl.setZValue(50)
        self._price_lbl.setVisible(False)
        self._plot_widget.addItem(self._price_lbl, ignoreBounds=True)

        # Time label anchored just above the bottom axis at cursor X
        self._time_lbl = pg.TextItem(anchor=(0.5, 1.0),
                                     color="#ffffff", fill=_lbl_fill)
        self._time_lbl.setZValue(50)
        self._time_lbl.setVisible(False)
        self._plot_widget.addItem(self._time_lbl, ignoreBounds=True)

        # Mid-price path line — connects (bid+ask)/2 per column
        self._price_path_item = pg.PlotCurveItem(
            pen=pg.mkPen(color=(255, 255, 255, 180), width=1.5),
            connect="finite",
        )
        self._price_path_item.setZValue(8)
        self._plot_widget.addItem(self._price_path_item)

        # Best bid / ask horizontal lines (盘口)
        # White dashed so they're visible in both Combined and Bid/Ask colormap modes.
        _quote_fill = pg.mkBrush(QColor(13, 17, 23, 200))
        self._bid_line = pg.InfiniteLine(
            angle=0, movable=False,
            pen=pg.mkPen(_TEAL, width=1, style=Qt.PenStyle.DashLine),
        )
        self._bid_line.setVisible(False)
        self._bid_line.setZValue(20)
        self._plot_widget.addItem(self._bid_line)

        self._bid_label = pg.TextItem(anchor=(0.0, 0.0), color=_TEAL, fill=_quote_fill)
        self._bid_label.setZValue(21)
        self._bid_label.setVisible(False)
        self._plot_widget.addItem(self._bid_label, ignoreBounds=True)

        self._ask_line = pg.InfiniteLine(
            angle=0, movable=False,
            pen=pg.mkPen(_RED, width=1, style=Qt.PenStyle.DashLine),
        )
        self._ask_line.setVisible(False)
        self._ask_line.setZValue(20)
        self._plot_widget.addItem(self._ask_line)

        self._ask_label = pg.TextItem(anchor=(0.0, 1.0), color=_RED, fill=_quote_fill)
        self._ask_label.setZValue(21)
        self._ask_label.setVisible(False)
        self._plot_widget.addItem(self._ask_label, ignoreBounds=True)

        # Mouse tracking
        self._plot_widget.scene().sigMouseMoved.connect(self._on_mouse_move)

    # ── public API ─────────────────────────────────────────────────────────────

    def set_code(self, code: str) -> None:
        if code == self._code:
            return
        self._code = code
        self.setWindowTitle(f"Liquidity Heatmap  —  {code}")
        self._reset_grid()

    def set_live(self, live: bool) -> None:
        self._live = live
        if live and self._code:
            if self._needs_init:
                if self._bulk_worker is not None and self._bulk_worker.isRunning():
                    return   # prefill already in flight — _on_bulk_ready will clear _needs_init
                # Pre-fill with the last 5 historical snapshots, then start timer
                self._bulk_worker = _BulkSnapshotWorker(self._code, 5)
                self._bulk_worker.done.connect(self._on_bulk_ready)
                self._bulk_worker.start()
            else:
                self._timer.start(self._col_secs_spin.value() * 1000)
                self._on_tick()
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
        # Discard any in-flight bulk fetch for the old code
        if self._bulk_worker is not None:
            _retire_worker(self._bulk_worker)
            self._bulk_worker = None
        # Discard stale absorb worker so its callback can't overwrite the new code's ticks
        if self._absorb_worker is not None:
            _retire_worker(self._absorb_worker)
            self._absorb_worker = None
        self._absorb_ticks   = []
        self._absorb_last_ts = None
        self._needs_init = True

        self._timer.stop()
        max_cols = self._max_cols_spin.value()
        n_price  = self._n_price_spin.value()
        self._bid_grid  = np.zeros((max_cols, n_price), dtype=np.float64)
        self._ask_grid  = np.zeros((max_cols, n_price), dtype=np.float64)
        self._col_ts      = []
        self._raw_snaps   = []
        self._mid_prices  = []
        self._latest_snap = []
        self._last_snap_ts = None
        self._price_min = 0.0
        self._price_max = 0.0
        self._bin_size  = 0.0
        self._best_bid  = None
        self._best_ask  = None
        self._live_mid  = None
        for img in (self._img_combined, self._img_bid, self._img_ask):
            img.clear()
        self._price_path_item.setData([], [])
        self._vline.setVisible(False)
        self._hline.setVisible(False)
        self._price_lbl.setVisible(False)
        self._time_lbl.setVisible(False)
        self._bid_line.setVisible(False)
        self._bid_label.setVisible(False)
        self._ask_line.setVisible(False)
        self._ask_label.setVisible(False)
        self._clear_overlay_items()
        self._update_legend()   # bin_size reset to 0 -- drop the stale "$/band" reading

    def _depth_to_cursor(self, target: float) -> str:
        """Return depth-to-cursor annotation, or '' when data is unavailable."""
        if not self._latest_snap or self._best_bid is None or self._best_ask is None:
            return ""
        return _calc_depth_label(
            self._latest_snap, self._best_bid, self._best_ask, target
        )

    def _column_tick_delta(self, col: int) -> float | None:
        """Executed buy_vol - sell_vol from raw ticks belonging to column
        `col` -- the actual traded delta for that period, projected onto the
        heatmap's time axis under the cursor. Only depends on which column
        (time) the cursor is over, not which price row.

        Bucketed the *same way* detect_aggressor_bubbles() assigns ticks to
        columns -- bisect against the actual recorded column timestamps
        (col_ts[col] <= tick_ts < col_ts[col+1]), not an assumed fixed
        Col(s)-second width, which can drift slightly from real column
        spacing (timer jitter). Matching bucketing means hovering on/near a
        bubble reads exactly that bubble's own delta.

        Needs _absorb_ticks to be cached -- the same tick cache "Aggressor"
        bubbles use -- so it returns None until Aggressor has been enabled
        at least once this session (no extra background load is triggered
        from mouse-move, which fires far too often for that).
        """
        if not self._absorb_ticks or not (0 <= col < len(self._col_ts)):
            return None
        col_start = self._col_ts[col]
        col_end   = self._col_ts[col + 1] if col + 1 < len(self._col_ts) else None
        ts_list   = [t["ts"] for t in self._absorb_ticks]
        lo = bisect.bisect_left(ts_list, col_start)
        hi = bisect.bisect_left(ts_list, col_end) if col_end is not None else len(ts_list)
        buy = sell = 0.0
        for tk in self._absorb_ticks[lo:hi]:
            d = tk.get("direction", "NEUTRAL")
            if d == "BUY":
                buy += float(tk["volume"])
            elif d == "SELL":
                sell += float(tk["volume"])
        return buy - sell

    def _on_mouse_move(self, pos) -> None:
        """Update local crosshair when cursor is inside the plot."""
        if not self._plot_widget.sceneBoundingRect().contains(pos):
            self._hline.setVisible(False)
            self._price_lbl.setVisible(False)
            self._time_lbl.setVisible(False)
            return

        pt = self._plot_widget.getPlotItem().vb.mapSceneToView(pos)
        x, y = pt.x(), pt.y()
        col = int(round(x))

        # Horizontal line + price label (executed tick delta, bucketed the
        # same way as aggressor bubbles so it matches a bubble's own delta
        # when hovering on/near it, + depth-to-cursor annotation)
        self._hline.setPos(y)
        self._hline.setVisible(True)
        tick_delta  = self._column_tick_delta(col)
        delta_str   = f"  Δ{tick_delta:+,.0f}" if tick_delta is not None else ""
        depth_str   = self._depth_to_cursor(y)
        lbl_text    = f"{y:.2f}{delta_str}  {depth_str}" if depth_str else f"{y:.2f}{delta_str}"
        self._price_lbl.setText(lbl_text)
        xlo, xhi = self._plot_widget.getPlotItem().vb.viewRange()[0]
        ylo, yhi = self._plot_widget.getPlotItem().vb.viewRange()[1]
        self._price_lbl.setPos(xlo + (xhi - xlo) * 0.005, y)
        self._price_lbl.setVisible(True)

        # Vertical line + time label (local mouse — independent of main chart sync)
        self._vline.setPos(x)
        self._vline.setVisible(True)
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
        snap_ts = snap[0]["ts"]
        new_data = snap_ts != self._last_snap_ts
        if new_data:
            self._last_snap_ts = snap_ts
            self._maybe_init_price_range(snap)
        prev_n = len(self._col_ts)
        xlo, xhi = self._plot_widget.getPlotItem().vb.viewRange()[0] if prev_n > 0 else (0, 0)
        # Always push a column so the time axis keeps advancing. When the
        # book hasn't actually changed since the last tick (common once
        # polling faster than ORDER_BOOK pushes arrive, e.g. Col(s)=1 during
        # a quiet moment), forward-fill with the last known snapshot instead
        # of an empty one -- otherwise the grid/price-path gets a visible gap
        # for every tick that landed between two real pushes. is_fill=True
        # keeps this out of _raw_snaps so iceberg/spoof detection (which
        # looks for volume *refreshing* at a price) doesn't mistake a
        # repeated stale snapshot for a genuine refresh.
        self._push_column(snap if new_data else self._latest_snap,
                          ts=snap_ts, is_fill=not new_data)
        new_n = len(self._col_ts)
        self._render()
        # Auto-follow: while the rolling buffer is still filling up (column
        # count still growing), the newest column's index keeps advancing,
        # so a view left at its initial position falls behind and needs a
        # manual drag to see new data -- keep it pinned to the right edge
        # as long as it was already there, same as the main chart's
        # auto-scroll. Once the buffer hits capacity the column count stops
        # growing (new columns roll in at the same fixed rightmost index),
        # so this naturally stops firing and a user's manual zoom/pan is
        # left alone.
        visible = xhi - xlo
        max_cols = self._max_cols_spin.value()
        if new_n > prev_n > 0 and xhi >= prev_n - 2 and visible < max_cols - 1:
            # visible < max_cols excludes the default full-buffer-width view
            # (set once at n<=1) -- that already shows every column at once
            # and never needs to shift; only a genuinely zoomed-in view can
            # fall behind the newest column.
            new_xhi = new_n + 0.5
            self._plot_widget.setXRange(new_xhi - visible, new_xhi, padding=0)
        if new_data:
            self._redraw_orderflow_markers()
            self._load_absorb_ticks()

    def _maybe_init_price_range(self, snap: list[dict]) -> None:
        """Set/rebuild the visible price band from the levels *near the touch*.

        A full L2 snapshot can carry up to 60 BID + 60 ASK levels ("120挡").
        Using min/max over *all* of them lets one deep, far-from-touch resting
        order (common on a volatile leveraged ETF) blow the window out to
        cover its entire depth -- e.g. a stock trading at $145 with a lone
        level down at $80 forces an $80-170 window, crushing all the actually
        useful near-touch depth into a sliver a few pixels tall. This tool is
        for watching near-touch depth move, not visualizing a single distant
        block order, so the window is sized from each side's levels out to
        the first outlier gap (_near_touch_cutoff) rather than a fixed count
        -- a tightly-clustered book keeps more of its real depth visible, a
        book with a stub order far from the touch cuts off right before it.
        Anything past the cutoff just falls outside the bin range and isn't
        painted (same as it always has been for out-of-range levels).
        """
        bids = sorted((r["price"] for r in snap if r["side"] == "BID"), reverse=True)
        asks = sorted(r["price"] for r in snap if r["side"] == "ASK")
        near_prices = _near_touch_cutoff(bids) + _near_touch_cutoff(asks)
        if not near_prices:
            return
        lo  = min(near_prices)
        hi  = max(near_prices)
        mid = (lo + hi) / 2
        span = max(hi - lo, mid * 0.02)   # at least ±1% of mid price
        new_min = lo  - span * 0.5
        new_max = hi  + span * 0.5

        n_price = self._n_price_spin.value()
        if self._bin_size == 0.0:
            self._price_min = new_min
            self._price_max = new_max
            self._bin_size  = (new_max - new_min) / n_price
            self._update_legend()   # bin_size just became known -- show it
        else:
            # Rebuild trigger threshold is a fraction of *price* (mid), not
            # of the current window's own width. It used to be the latter
            # (out_lo/out_hi as a fraction of price_max - price_min) --
            # harmless back when the window always covered full 120-level
            # depth (tens of dollars wide, so 30% of it was a large, rare
            # move), but the near-touch-only window above is deliberately
            # tight (a few dollars), so that same 30%-of-window math now
            # trips on a ~1% price move -- reported as the heatmap
            # constantly wiping and losing history on completely ordinary
            # intraday movement. A window this tight for resolution and a
            # trigger this sensitive for stability can't both be relative
            # to the same (now-tiny) number.
            out_lo = (self._price_min - lo) / mid
            out_hi = (hi - self._price_max) / mid
            # A sharp, fast move (e.g. an aggressive sweep consuming several
            # levels at once) can push the touch itself completely outside
            # the window in a single tick, before cumulative drift crosses
            # the 5% threshold above -- the swept side then renders as
            # "missing" only because it no longer has any bin to land in,
            # not because the data is actually gone (reported: bid vanished
            # right where a large sell aggressor hit, recovered once a
            # reset happened to catch up). Force a rebuild immediately
            # whenever the *current* best bid/ask itself is unrenderable,
            # regardless of the 5% drift check, so a fast move never has to
            # wait for the next unrelated rebuild to become visible again.
            cur_best_bid = bids[0] if bids else None
            cur_best_ask = asks[0] if asks else None
            touch_outside = (
                (cur_best_bid is not None and cur_best_bid < self._price_min) or
                (cur_best_ask is not None and cur_best_ask > self._price_max)
            )
            cur_width = self._price_max - self._price_min
            new_width = new_max - new_min
            too_wide = cur_width > 0 and new_width < cur_width * _WINDOW_SHRINK_FACTOR
            if out_lo > 0.05 or out_hi > 0.05 or touch_outside or too_wide:
                self._price_min = new_min
                self._price_max = new_max
                self._bin_size  = (new_max - new_min) / n_price
                max_cols = self._max_cols_spin.value()
                self._bid_grid   = np.zeros((max_cols, n_price), dtype=np.float64)
                self._ask_grid   = np.zeros((max_cols, n_price), dtype=np.float64)
                self._col_ts     = []
                self._raw_snaps  = []
                self._mid_prices = []
                # _absorb_ticks feeds detect_aggressor_bubbles() with column
                # indices relative to self._col_ts; wiping col_ts above
                # without also wiping this leaves stale (tick, old-col-index)
                # entries whose index no longer means anything once columns
                # renumber from 0 again. Worse, the bubble markers already
                # drawn from those stale entries are never removed either
                # (only _redraw_orderflow_markers()'s own clear-then-draw
                # does that, and a rebuild doesn't call it) -- they just sit
                # at their old data-coordinate position forever, reported as
                # "aggressor bubbles stuck on screen, not moving/updating".
                self._absorb_ticks = []
                self._clear_overlay_items()
                self._update_legend()   # bin_size just changed on rebuild

    def _push_column(self, snap: list[dict],
                     ts: datetime | None = None,
                     is_fill: bool = False) -> None:
        """Append one OB snapshot as the rightmost column, rolling if full.

        *ts*: use for historical pre-fill (DB timestamp); omit for live updates
        (wall-clock time is used).
        *is_fill*: `snap` is the last known snapshot repeated because the book
        hasn't actually changed since -- still paints the grid/price-path (so
        the heatmap stays visually continuous), but does not record entries
        into _raw_snaps or overwrite _latest_snap, since those feed iceberg/
        spoof detection which look for volume genuinely refreshing.
        """
        min_vol  = self._min_vol_spin.value()
        max_cols = self._max_cols_spin.value()

        if len(self._col_ts) >= max_cols:
            self._bid_grid = np.roll(self._bid_grid, -1, axis=0)
            self._ask_grid = np.roll(self._ask_grid, -1, axis=0)
            self._bid_grid[-1] = 0.0
            self._ask_grid[-1] = 0.0
            self._col_ts.pop(0)
            if self._mid_prices:
                self._mid_prices.pop(0)
            # Trim raw_snaps to match the new oldest column time
            if self._col_ts:
                cutoff = self._col_ts[0]
                self._raw_snaps = [s for s in self._raw_snaps if s["ts"] >= cutoff]

        col    = len(self._col_ts)
        col_ts = ts if ts is not None else datetime.now(_ET).replace(tzinfo=None)
        self._col_ts.append(col_ts)

        for row in snap:
            if row["volume"] < min_vol:
                continue
            p_bin = int((row["price"] - self._price_min) / self._bin_size)
            if not (0 <= p_bin < self._bid_grid.shape[1]):
                continue
            if row["side"] == "BID":
                self._bid_grid[col, p_bin] = row["volume"]
            else:
                self._ask_grid[col, p_bin] = row["volume"]

        # Track best bid / ask and mid-price from this snapshot
        mid = _calc_col_mid(snap)
        bid_prices = [r["price"] for r in snap if r["side"] == "BID"]
        ask_prices = [r["price"] for r in snap if r["side"] == "ASK"]
        col_bid = max(bid_prices) if bid_prices else None
        col_ask = min(ask_prices) if ask_prices else None
        self._best_bid = col_bid if col_bid is not None else self._best_bid
        self._best_ask = col_ask if col_ask is not None else self._best_ask
        self._mid_prices.append(mid)

        if not is_fill:
            for row in snap:
                self._raw_snaps.append({
                    "ts":     col_ts,
                    "side":   row["side"],
                    "price":  row["price"],
                    "volume": row["volume"],
                })
            self._latest_snap = snap   # cache for depth-to-cursor calculation

    def _on_bulk_ready(self, snapshots: list) -> None:
        """Push historical pre-fill columns then switch to the normal timer."""
        self._needs_init  = False
        self._bulk_worker = None
        for snap in snapshots:
            if snap:
                self._maybe_init_price_range(snap)
                self._push_column(snap, ts=snap[0]["ts"])
        if self._col_ts:
            self._render()
            self._redraw_orderflow_markers()
            self._load_absorb_ticks()
            self._reset_view()   # fit view to the freshly loaded pre-fill data
        # Start normal one-by-one updates
        if self._live and self._code:
            self._timer.start(self._col_secs_spin.value() * 1000)
            self._on_tick()

    def _render(self) -> None:
        n = len(self._col_ts)
        if n == 0 or self._bin_size == 0.0:
            return

        rect     = QRectF(0.0, self._price_min,
                          float(n), self._price_max - self._price_min)
        bid_view = self._bid_grid[:n]
        ask_view = self._ask_grid[:n]

        gamma = self._gamma_spin.value()
        if self._bid_ask_cb.isChecked():
            self._img_combined.setVisible(False)
            for img, grid, color in (
                (self._img_bid, bid_view, _TEAL),
                (self._img_ask, ask_view, _RED),
            ):
                rgba = _single_rgba(grid, color, gamma)
                if rgba is not None:
                    img.setImage(rgba)
                    img.setRect(rect)
                    img.setVisible(True)
                else:
                    img.setVisible(False)
        else:
            self._img_bid.setVisible(False)
            self._img_ask.setVisible(False)
            rgba = _hot_rgba(bid_view + ask_view, gamma)
            if rgba is not None:
                self._img_combined.setImage(rgba)
                self._img_combined.setRect(rect)
                self._img_combined.setVisible(True)
            else:
                self._img_combined.setVisible(False)

        self._time_axis.update_timestamps(self._col_ts)
        max_cols = self._max_cols_spin.value()
        # Minimum tick gap = max_cols // 10 so labels are never crowded even
        # when only a few columns of data exist in the full-width view.
        step = max(max_cols // 10, 1)
        self._plot_widget.getPlotItem().getAxis("bottom").setTicks(
            [[(i, self._col_ts[i].strftime("%H:%M")) for i in range(0, n, step)]]
        )

        if n <= 1:
            self._plot_widget.setXRange(0, max_cols + self._right_margin_cols(), padding=0)
            self._plot_widget.setYRange(self._price_min, self._price_max, padding=0)

        # Mid-price path line
        if self._price_path_cb.isChecked() and self._mid_prices:
            self._update_price_path(n)
            self._price_path_item.setVisible(True)
        else:
            self._price_path_item.setVisible(False)

        # Update best bid/ask spread lines
        if self._best_bid is not None:
            self._set_quote_line(self._bid_line, self._bid_label, self._best_bid, "B")
        if self._best_ask is not None:
            self._set_quote_line(self._ask_line, self._ask_label, self._best_ask, "A")

        # Auto-follow: _price_min/_price_max (and the view Y-range set from
        # them at n<=1) only get refreshed on a full grid rebuild, which is
        # gated by a 30%-of-band dead zone (see _maybe_init_price_range) --
        # below that threshold price can already sit outside the still-fixed
        # view for a while, so the heatmap appears to drift off the top/
        # bottom of the window well before any rebuild fires. Re-center
        # (same zoom span, no rebuild) whenever price nears/passes the edge.
        if n > 1 and self._best_bid is not None and self._best_ask is not None:
            self._follow_price_view((self._best_bid + self._best_ask) / 2)

    def _follow_price_view(self, price: float) -> None:
        """Re-center the Y-range around `price` if it has drifted near/past
        the edge of the current view, preserving the view's zoom span so a
        manual zoom isn't fought every tick."""
        ylo, yhi = self._plot_widget.getPlotItem().vb.viewRange()[1]
        span = yhi - ylo
        if span <= 0:
            return
        margin = span * 0.1
        if ylo + margin <= price <= yhi - margin:
            return
        self._plot_widget.setYRange(price - span / 2, price + span / 2, padding=0)

    def set_red_up(self, red_up: bool) -> None:
        """Sync bid/ask line colors with the main chart's red-up convention.

        red-up=True  (CN): ask=RED  bid=TEAL
        red-up=False (WS): ask=TEAL bid=RED
        """
        self._red_up = red_up
        bid_col = _TEAL if red_up else _RED
        ask_col = _RED  if red_up else _TEAL
        self._bid_line.setPen(pg.mkPen(bid_col, width=1, style=Qt.PenStyle.DashLine))
        self._bid_label.setColor(bid_col)
        self._ask_line.setPen(pg.mkPen(ask_col, width=1, style=Qt.PenStyle.DashLine))
        self._ask_label.setColor(ask_col)
        if self._best_bid is not None:
            self._set_quote_line(self._bid_line, self._bid_label, self._best_bid, "B")
        if self._best_ask is not None:
            self._set_quote_line(self._ask_line, self._ask_label, self._best_ask, "A")

    def _set_quote_line(
        self,
        line: pg.InfiniteLine,
        label: pg.TextItem,
        price: float,
        prefix: str,
    ) -> None:
        """Position a bid/ask InfiniteLine and its price label at the left edge of the view."""
        line.setValue(price)
        line.setVisible(True)
        xlo, xhi = self._plot_widget.getPlotItem().vb.viewRange()[0]
        label.setText(f"{prefix} {price:.2f}")
        label.setPos(xlo + (xhi - xlo) * 0.005, price)
        label.setVisible(True)

    def _update_price_path(self, n: int | None = None) -> None:
        """Rebuild the price path curve, appending the live mid as a trailing point."""
        if n is None:
            n = len(self._col_ts)
        if not self._mid_prices:
            return
        xs = np.arange(n, dtype=np.float64) + 0.5
        ys = np.array(
            [m if m is not None else np.nan for m in self._mid_prices[:n]],
            dtype=np.float64,
        )
        if self._live_mid is not None:
            xs = np.append(xs, float(n))
            ys = np.append(ys, self._live_mid)
        self._price_path_item.setData(xs, ys)

    @pyqtSlot(float, float)
    def update_quote(self, bid: float, ask: float) -> None:
        """Update Best Bid/Ask spread lines only.

        Called from get_market_snapshot (regular-session data) — does NOT
        touch _live_mid to avoid a mismatch with after-hours OB prices.
        """
        if bid > 0:
            self._best_bid = bid
            self._set_quote_line(self._bid_line, self._bid_label, bid, "B")
        if ask > 0:
            self._best_ask = ask
            self._set_quote_line(self._ask_line, self._ask_label, ask, "A")

    @pyqtSlot(float, float)
    def update_live_price(self, bid: float, ask: float) -> None:
        """Update the price path trailing point from real-time QUOTE subscription.

        Only called from the QUOTE push handler so the mid always reflects the
        correct session (extended hours included).  Safe to invoke via
        QMetaObject.invokeMethod from a non-Qt thread.
        """
        if bid > 0 and ask > 0:
            self._live_mid = (bid + ask) / 2
            if self._price_path_cb.isChecked() and self._mid_prices:
                self._update_price_path()

    # ── iceberg / spoof detection and drawing ──────────────────────────────────

    def _build_bucket_to_idx(self) -> dict:
        """Map each column's exact timestamp to its column index.

        Detection functions place an event at the exact column it occurred
        in. This used to go through candle_start(ts, cm) with cm floored to
        1 minute -- fine when each column spanned tens of seconds, but at
        Col(s)=1 a single minute covers 60 actual columns, so every event
        detected anywhere within that minute got pinned to the *first* of
        those 60 columns regardless of when within that minute it actually
        happened. Two genuinely different moments (price having moved
        substantially between them) would then be drawn at the same
        x-position (reported: two imbalance markers, one at an ask price one
        at a bid price, both appearing at the same column). _raw_snaps
        entries are always tagged with the exact column timestamp they were
        recorded under (see _push_column), so a direct dict lookup gives
        exact per-column placement regardless of col_secs -- no coarsening.
        """
        return {ts: i for i, ts in enumerate(self._col_ts)}

    def _redraw_orderflow_markers(self) -> None:
        self._clear_overlay_items()
        if not self._raw_snaps or self._bin_size == 0.0:
            return

        bucket_to_idx = self._build_bucket_to_idx()
        min_vol = self._min_vol_spin.value()
        n_price = self._n_price_spin.value()

        if self._ice_cb.isChecked():
            from analysis.orderflow_detect import detect_icebergs
            icebergs = detect_icebergs(
                self._raw_snaps, bucket_to_idx, self._bin_size, self._price_min,
                n_price,
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
                n_price,
                min_vol=float(min_vol),   # 0 → auto median of latest snapshot
                max_duration_secs=self._spoof_dur_spin.value(),
            )
            self._draw_spoof_markers(spoofs)

        if self._simb_cb.isChecked():
            from analysis.orderflow_detect import detect_stacked_imbalance
            simbs = detect_stacked_imbalance(
                self._raw_snaps,
                bucket_to_idx, self._bin_size, self._price_min,
                n_price,
                min_levels=self._simb_levels_spin.value(),
                imbalance_ratio=self._simb_ratio_spin.value(),
                min_vol=float(min_vol),
                max_depth=self._simb_depth_spin.value(),
            )
            self._draw_simb_markers(simbs)

        if self._absorb_cb.isChecked() and self._absorb_ticks:
            from analysis.orderflow_detect import detect_aggressor_bubbles
            bubbles = detect_aggressor_bubbles(
                self._absorb_ticks,
                self._col_ts,
                self._mid_prices,
                col_secs=self._col_secs_spin.value(),
                min_delta_vol=float(self._absorb_min_vol_spin.value()),
            )
            self._draw_absorb_bubbles(bubbles)

    def _draw_iceberg_markers(self, icebergs: list[tuple]) -> None:
        """Bright-purple horizontal line segments at detected iceberg price levels.

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
            pen   = pg.mkPen(color=(224, 64, 251, alpha), width=2)
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
                symbol=_SYM_UP, size=15,
                pen=outline,
                brush=pg.mkBrush(*ORANGE),
            )
            scat.setZValue(11)
            self._plot_widget.addItem(scat)
            self._spoof_items.append(scat)

        if down_xs:
            scat = pg.ScatterPlotItem(
                x=down_xs, y=down_ys,
                symbol=_SYM_DOWN, size=15,
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

    def _draw_simb_markers(self, simbs: list[tuple]) -> None:
        """Vertical bars at stacked-imbalance zones.

        Lime = bullish (BID dominates); pink = bearish (ASK dominates).
        Each bar spans [price_lo, price_hi] at the snapshot's column.
        Opacity encodes mean_ratio strength (capped at 5×).
        """
        if not simbs:
            return

        _BID_COL = (178, 255,  89)   # Lime A200 — high contrast vs teal heatmap
        _ASK_COL = (255,  64, 129)   # Pink A200 — high contrast vs red-orange heatmap

        bid_xs: list[float] = []
        bid_ys: list[float] = []
        ask_xs: list[float] = []
        ask_ys: list[float] = []

        for bar_idx, price_lo, price_hi, direction, mean_ratio in simbs:
            cx = float(bar_idx) + 0.45   # centre of the column
            if direction == "BID":
                bid_xs += [cx, cx, np.nan]
                bid_ys += [price_lo, price_hi, np.nan]
            else:
                ask_xs += [cx, cx, np.nan]
                ask_ys += [price_lo, price_hi, np.nan]

        for xs, ys, col in (
            (bid_xs, bid_ys, _BID_COL),
            (ask_xs, ask_ys, _ASK_COL),
        ):
            if not xs:
                continue
            item = pg.PlotCurveItem(
                x=np.array(xs, dtype=np.float64),
                y=np.array(ys, dtype=np.float64),
                pen=pg.mkPen(color=(*col, 220), width=5),
                connect="finite",
            )
            item.setZValue(12)
            self._plot_widget.addItem(item)
            self._simb_items.append(item)

    def _draw_absorb_bubbles(self, bubbles: list[tuple]) -> None:
        """Draw ScatterPlotItem bubbles at aggressor events on the price path.

        Gold   = dominant BUY aggression (net buyers exceeded threshold).
        Purple = dominant SELL aggression (net sellers exceeded threshold).
        Bubble size scales with net delta volume (8–30 px range).
        Hovering a bubble shows a tooltip with direction and net delta volume.
        """
        if not bubbles:
            return
        _GOLD   = (255, 160,   0, 200)   # gold   — buy aggressor
        _PURPLE = (171,  71, 188, 200)   # purple — sell aggressor

        max_vol = max(b[3] for b in bubbles)
        outline = pg.mkPen("white", width=0.5)
        spots = []
        for col_idx, price, direction, vol in bubbles:
            size  = 8.0 + 22.0 * (vol / max_vol)
            color = _GOLD if direction == "BUY" else _PURPLE
            spots.append({
                "pos":   (float(col_idx) + 0.5, price),
                "size":  size,
                "pen":   outline,
                "brush": pg.mkBrush(*color),
                "data":  {"direction": direction, "vol": vol, "price": price},
            })

        scat = pg.ScatterPlotItem(spots=spots, hoverable=True, tip=None)
        scat.setZValue(12)
        scat.sigHovered.connect(self._on_absorb_hovered)
        self._plot_widget.addItem(scat)
        self._absorb_items.append(scat)

    def _on_absorb_hovered(self, scatter, points, ev) -> None:
        from PyQt6.QtWidgets import QToolTip
        from PyQt6.QtGui import QCursor
        if len(points) == 0:
            QToolTip.hideText()
            return
        d = points[0].data()
        label = "BUY aggressor" if d["direction"] == "BUY" else "SELL aggressor"
        QToolTip.showText(
            QCursor.pos(),
            f"{label}\nPrice: {d['price']:.2f}\nΔvol: {d['vol']:,.0f}")

    def _load_absorb_ticks(self) -> None:
        """Trigger background tick load for the current display window."""
        if not self._absorb_cb.isChecked() or not self._code or not self._col_ts:
            return
        if self._absorb_worker is not None and self._absorb_worker.isRunning():
            # Record that the window has advanced; re-load immediately after the
            # current worker finishes so bubbles stay current during busy markets.
            self._absorb_reload_pending = True
            return
        self._absorb_reload_pending = False
        col_secs = self._col_secs_spin.value()
        if self._absorb_last_ts is None:
            # Full load: query the entire visible window.
            start = self._col_ts[0] - timedelta(seconds=col_secs)
        else:
            # Incremental: fetch only ticks strictly after the last cached one.
            start = self._absorb_last_ts + timedelta(milliseconds=1)
        end = self._col_ts[-1] + timedelta(seconds=col_secs * 2)
        self._absorb_worker = _AbsorbTickWorker(self._code, start, end)
        self._absorb_worker.done.connect(self._on_absorb_ready)
        self._absorb_worker.start()

    def _on_absorb_ready(self, ticks: list) -> None:
        """Receive ticks from background worker; merge into cache incrementally."""
        if ticks:
            # Append new ticks and update the high-water mark.
            self._absorb_ticks.extend(ticks)
            self._absorb_last_ts = ticks[-1]["ts"]
            # Trim entries that have scrolled off the left edge of the heatmap.
            if self._col_ts:
                cutoff = self._col_ts[0]
                self._absorb_ticks = [t for t in self._absorb_ticks
                                      if t["ts"] >= cutoff]
        self._redraw_orderflow_markers()
        if self._absorb_reload_pending:
            self._load_absorb_ticks()

    def _on_absorb_changed(self) -> None:
        """Checkbox or MinΔ changed: redraw if ticks are cached, else load from DB."""
        self._update_legend()
        if self._absorb_cb.isChecked():
            if self._absorb_ticks:
                # Ticks already cached — threshold change only, redraw immediately.
                self._redraw_orderflow_markers()
            else:
                self._load_absorb_ticks()
        else:
            for item in self._absorb_items:
                self._plot_widget.removeItem(item)
            self._absorb_items.clear()

    def _clear_overlay_items(self) -> None:
        for item in (self._iceberg_items + self._spoof_items
                     + self._simb_items + self._absorb_items):
            self._plot_widget.removeItem(item)
        self._iceberg_items.clear()
        self._spoof_items.clear()
        self._simb_items.clear()
        self._absorb_items.clear()

    def _update_legend(self) -> None:
        n = 5
        max_a = 180 / 255
        parts: list[str] = []

        # Price-axis resolution: each band's height in $ (bin_size), so it's
        # never a mystery how much price range one row of the heatmap covers.
        # A band spans [price_min + i*bin_size, price_min + (i+1)*bin_size) --
        # i.e. its *lower* edge is the exact bin boundary; read it as covering
        # bin_size upward from wherever its bottom edge sits. The crosshair's
        # own price readout is continuous (not snapped to band edges).
        if self._bin_size > 0:
            parts.append(
                f'<font color="{_FG}">${self._bin_size:.3f}/band</font>&nbsp;&nbsp;'
            )

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
            parts.append(f'&nbsp;&nbsp;<font color="#e040fb">-- Iceberg</font>')

        if self._spoof_cb.isChecked():
            parts.append(
                f'&nbsp;&nbsp;<font color="orange">▲ Bid spoof&nbsp;▼ Ask spoof</font>'
            )

        if self._simb_cb.isChecked():
            parts.append(
                f'&nbsp;&nbsp;<font color="#b2ff59">| Bullish imbalance</font>'
                f'&nbsp;&nbsp;<font color="#ff4081">| Bearish imbalance</font>'
            )

        if self._absorb_cb.isChecked():
            parts.append(
                f'&nbsp;&nbsp;<font color="#ffa000">●</font>'
                f'&nbsp;<font color="{_FG}">BUY aggressor</font>'
                f'&nbsp;&nbsp;<font color="#ab47bc">●</font>'
                f'&nbsp;<font color="{_FG}">SELL aggressor</font>'
            )

        self._legend_lbl.setText("".join(parts))

    # ── control callbacks ──────────────────────────────────────────────────────

    def _on_gamma_changed(self) -> None:
        self._render()   # pure visual change — no overlay recalculation needed

    def _on_price_path_changed(self) -> None:
        self._render()   # toggle visibility only

    def _on_controls_changed(self) -> None:
        self._update_legend()
        self._render()
        self._redraw_orderflow_markers()

    def _on_col_secs_changed(self) -> None:
        if self._live and self._timer.isActive():
            self._timer.start(self._col_secs_spin.value() * 1000)

    def _on_max_cols_changed(self) -> None:
        self._reset_grid()
        if self._live and self._code:
            self.set_live(True)

    def _on_n_price_changed(self) -> None:
        self._reset_grid()
        if self._live and self._code:
            self.set_live(True)

    def _right_margin_cols(self) -> float:
        """Blank columns to leave to the right of the newest data.

        Without this, the rolling buffer's default/reset view spans exactly
        [0, max_cols] -- since the newest column is always painted at index
        max_cols-1 once the buffer is full, "now" sits flush against the
        right edge with nothing to visually mark it as "the current moment"
        rather than just where the chart happens to stop. 5% of the buffer
        width (min 5 columns) matches the main K-line chart's own "+3 bars"
        breathing-room convention.
        """
        return max(5, round(self._max_cols_spin.value() * 0.05))

    def _reset_view(self) -> None:
        """Restore X/Y ranges to the full data bounds (undo any zoom/pan)."""
        n = len(self._col_ts)
        if n == 0 or self._bin_size == 0.0:
            return
        self._plot_widget.setXRange(
            0, self._max_cols_spin.value() + self._right_margin_cols(), padding=0)
        self._plot_widget.setYRange(self._price_min, self._price_max, padding=0)

    def _on_pin_toggled(self, checked: bool) -> None:
        flags = self.windowFlags()
        if checked:
            self.setWindowFlags(flags | Qt.WindowType.WindowStaysOnTopHint)
        else:
            self.setWindowFlags(flags & ~Qt.WindowType.WindowStaysOnTopHint)
        self.show()

    def mouseDoubleClickEvent(self, event) -> None:
        """Double-click anywhere on the window resets the zoom."""
        self._reset_view()
        super().mouseDoubleClickEvent(event)

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        self._timer.stop()
        for w in (self._worker, self._bulk_worker, self._absorb_worker):
            if w is not None:
                _retire_worker(w)
        super().closeEvent(event)


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
