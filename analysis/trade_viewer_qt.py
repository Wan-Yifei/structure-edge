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
import socket
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
    QShortcut, QKeySequence,
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QToolBar, QLabel, QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox,
    QPushButton, QCheckBox, QButtonGroup, QRadioButton, QSizePolicy,
    QFrame, QStatusBar, QMessageBox, QDialog, QDialogButtonBox, QFormLayout,
    QSystemTrayIcon,
)

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from analysis import viewer_config as vcfg
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
    TickerHandlerBase, StockQuoteHandlerBase, CurKlineHandlerBase, RET_OK,
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


def _alpha_brush(hex_: str, alpha: int):
    """A brush at a fixed alpha, built once at import.

    The delta labels reassign their fill on up to 1440 items a render and
    it only ever takes two values, so building the brushes here beats
    mkBrush in the loop. Same colours and same alpha the per-label
    mkBrush(_qc(color, 180)) produced.
    """
    c = pg.mkColor(hex_)
    c.setAlpha(alpha)
    return pg.mkBrush(c)


_DELTA_UP_BRUSH = _alpha_brush(_UP, 180)
_DELTA_DN_BRUSH = _alpha_brush(_DOWN, 180)

# Screen pixels per bar below which the delta labels are hidden. They are
# Monospace 8pt and run to about five characters ("+1.2M"), so under this
# they overlap into a smear. At a 1600px wide plot this shows them from
# roughly 90 bars in.
_DELTA_MIN_PX = 18.0
# Pixels per bar below which the heatmap's per-level bins stop being
# readable and the body shows the bar's delta sign instead. Above it the
# heatmap is drawn as before. One switch, not a fade: a half-drawn
# heatmap is worse than either state.
_HM_MIN_PX = 8.0

# How a bin's volume share becomes opacity.
#
# Volume inside a candle is concentrated, not spread: measured over
# 11,262 drawn bins of one SOXL session, a bin holds a MEDIAN 20% of its
# candle's busiest bin and the 10th percentile holds 2%. Mapping that
# share to alpha linearly put the median bin at alpha 86 and left 66% of
# the painted area under alpha 100 -- gold and purple both wash out to
# the same dark smudge against the background, which is the whole of the
# "can't read the heatmap" complaint.
#
# The gamma is the fix: raising the share to a power below 1 lifts the
# crowded low end without reordering anything, the same move the
# liquidity heatmap's gamma slider makes. It costs dynamic range at the
# top -- a 1%-share bin and a 20%-share bin are closer in opacity than
# they were -- and that is the trade being made deliberately, because
# the ordering is still readable at alpha 120..245 and was not at 63.
_HM_GAMMA = 0.45          # exponent on the bin's share of its candle
_HM_ALPHA_MIN = 90        # a drawn bin never falls below this
_HM_ALPHA_MAX = 245       # ... nor above it
# Imbalance still dims a bin, but far less than it did. At a single
# price level almost every bin is one-sided -- the median |ratio| is
# 0.95 -- so the term mostly idled; where it does bite, a 50/50 bin, the
# gold/purple choice really is close to arbitrary and deserves to recede.
_HM_IMB_FLOOR = 0.55      # multiplier at a perfectly balanced bin

# How often the session/range profile panel is rebuilt, at most. It costs
# 82ms, and a session profile does not move visibly between two redraws
# half a second apart.
_PROFILE_MIN_MS = 5_000

# Inverse-leveraged pairs. Hovering a price level on the key symbol also shows
# where its inverse would be trading at that level, so a level picked on one
# can be read straight off as a level on the other.
#
# Both are +/-3x the same index and reset daily from the previous close, so
# from that shared anchor a SOXL return of 3r implies a SOXS return of -3r:
#
#     SOXS = SOXS_prev_close * (2 - SOXL / SOXL_prev_close)
#
# The anchor is the LIVE pair, refreshed with every fetch cycle, not the daily
# previous close. Both work -- the relation only needs a moment when both legs
# are known -- but the error is drift accumulated since the anchor, so a fresh
# one is worth a lot. Measured on 1m and 5m bars, regular session only,
# 2026-08-01..09-25, the two timeframes agreeing throughout:
#
#     anchor          median   p90
#     prev close      $0.030   $0.138
#     live,  1m old   $0.011   $0.035
#     live, 15m old   $0.021   $0.081
#     live, 60m old   $0.029   $0.139
#
# and against a fresh anchor the error scales with how far the hovered level is
# from the live price, which is what actually matters when reading a level off
# the chart:
#
#     within 0.1%     $0.009   $0.022
#     0.1 - 0.25%     $0.011   $0.027
#     0.25 - 0.5%     $0.017   $0.043
#     0.5 - 1%        $0.027   $0.080
#     beyond 1%       $0.050   $0.153
#
# Careful with historical validation: moomoo's daily and 5m series are
# split-adjusted while 1m is raw, so mixing a daily prev_close into 1m bars
# manufactures a constant ~$0.58 offset that looks like model error.
#
# Tracking error and fund drift make this an estimate, not an arbitrage
# relation.
_INVERSE_PAIR  = {"US.SOXL": "US.SOXS"}
_PAIR_COL      = "#ab47bc"   # purple -- must not read as the gold price tag


def _short_code(code: str) -> str:
    """US.SOXL -> SOXL. Market prefixes waste width in a crosshair tag."""
    return code.split(".")[-1] if code else code


def _pair_refs(base_row, pair_row) -> tuple[float, float] | None:
    """Pick the anchor both legs are measured from: live pair, else prev close.

    Both references must come from the same moment or the ratio between them is
    meaningless, so a missing live print on either leg drops BOTH to prev close
    rather than pairing one leg's live price with the other's close. Happens
    pre-market, when one leg has traded and the other has not.
    """
    for field in ("last_price", "prev_close_price"):
        try:
            b = float(base_row.get(field) or 0)
            q = float(pair_row.get(field) or 0)
        except (TypeError, ValueError):
            continue
        if b > 0 and q > 0:
            return b, q
    return None
_AVWAP_COL = "#ffeb3b"  # anchored VWAP line/label color
_ZERO_GAMMA_COL = "#ff8c00"  # option Zero Gamma line -- matches gex.py's own matplotlib chart
# Half-width, in percent of price, of the window the POC switch hysteresis
# sums volume over. Comparing two single bins does not work: the profile is
# re-binned from scratch on every rebuild over a lo/hi that grows with the
# session, so a peak bin routinely hands a large slice of its volume to a
# neighbour and one bin's volume is far too noisy to gate a switch on
# (measured: rival/held single-bin ratios of 1.6-2.3 during what was
# visually one stable cluster). Summing a fixed PRICE window around each
# candidate is stable across re-binning.
_POC_CLUSTER_PCT = 0.5

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

# How many recent bars get_cur_kline is asked for when topping up a live
# frame. It bounds the gap a top-up can bridge; anything longer is caught
# by the gap check and falls back to a full fetch.
_CUR_KLINE_BARS = 200

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

def _day_window(date_str: str) -> tuple[datetime, datetime]:
    """The 28-hour span a "day" of ticks covers: prior 20:00 through 23:59."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return (dt - timedelta(days=1) + timedelta(hours=20),
            dt.replace(hour=23, minute=59, second=59))


def _query_ticks(code: str, start: datetime, end: datetime,
                 with_offex: bool) -> list | None:
    """Rows from ticks.db, or None when there is no database.

    Split out so both the full and the incremental read go through one
    place, and so tests can stand in for the store without a file.
    """
    db_path = pathlib.Path(__file__).parent.parent / "db" / "ticks.db"
    if not db_path.exists():
        return None
    from feeds.tick_store import TickStore
    with TickStore(db_path, read_only=True) as store:
        return store.query_ticks(code, start, end, with_mid=with_offex)


def top_n_levels(vols: np.ndarray, n: int) -> np.ndarray:
    """Boolean mask over the *n* entries of *vols* with the most volume.

    Used once per side of the tick profile, so the buy bars and the sell
    bars are ranked against their own side rather than pooled -- on a
    one-sided candle a pooled ranking spends every mark on the side that was
    already obvious and says nothing about the other.

    Levels with no volume are never marked. Without that, a candle with
    fewer than n traded levels on a side would outline empty bars, which
    reads as "nothing traded here" being called noteworthy.
    """
    mask = np.zeros(len(vols), dtype=bool)
    if n <= 0:
        return mask
    nz = np.where(vols > 0)[0]
    if len(nz):
        # Stable sort so that equal volumes mark the lower index -- the
        # lower price -- rather than whatever order the sort happened to
        # leave them in, which would make the marks flicker between redraws
        # of identical data.
        order = nz[np.argsort(-vols[nz], kind="stable")]
        mask[order[:n]] = True
    return mask


def _size_thresholds(rows) -> tuple[float, float]:
    """Adaptive S/M/L boundaries: the day's own p33 and p67 by volume.

    Fewer than six ticks is not a distribution, and splitting one into
    thirds would label two prints "large" on no evidence -- everything is
    small until there is enough to say otherwise.
    """
    vols = np.array([r["volume"] for r in rows if r["volume"] > 0],
                    dtype=float)
    if len(vols) < 6:
        return float("inf"), float("inf")
    return float(np.percentile(vols, 33)), float(np.percentile(vols, 67))


def _new_tick_bin() -> dict:
    return {
        "buy": 0, "sell": 0, "neutral": 0,
        "buy_s": 0, "buy_m": 0, "buy_l": 0,
        "sell_s": 0, "sell_m": 0, "sell_l": 0,
        "neutral_s": 0, "neutral_m": 0, "neutral_l": 0,
        # Off-exchange flow, carried beside the feed's own counts rather
        # than folded into them: every indicator that reads buy/sell keeps
        # the exact meaning it had, and the one line that wants this reads
        # these two.
        "off_buy": 0, "off_sell": 0,
    }


def _bucket_ticks(rows, candle_mins: int,
                  thresh_m: float, thresh_l: float) -> dict:
    """Aggregate rows into {bucket: {price: bin}} with GIVEN thresholds.

    The thresholds are a parameter rather than derived here so an
    incremental read can keep using the ones the day was loaded with --
    re-deriving them would mean re-stratifying rows that are no longer in
    memory.
    """
    def _size(v) -> str:
        if v <= thresh_m:
            return "s"
        if v <= thresh_l:
            return "m"
        return "l"

    buckets: dict = defaultdict(lambda: defaultdict(_new_tick_bin))
    for r in rows:
        ts = (r["ts"] if isinstance(r["ts"], datetime)
              else datetime.fromisoformat(str(r["ts"])))
        bucket = candle_start(ts, candle_mins)
        key = {"BUY": "buy", "SELL": "sell"}.get(
            r["direction"].upper(), "neutral")
        vol = r["volume"]
        buckets[bucket][r["price"]][key] += vol
        buckets[bucket][r["price"]][f"{key}_{_size(vol)}"] += vol
        if key == "neutral":
            md = r.get("mid_dir")
            if md == "BUY":
                buckets[bucket][r["price"]]["off_buy"] += vol
            elif md == "SELL":
                buckets[bucket][r["price"]]["off_sell"] += vol
    return dict(buckets)


# Where the option-chain context connects. Named so the probe below and
# the constructor that follows it cannot drift apart.
_OPEND_HOST, _OPEND_PORT = "127.0.0.1", 11111


def _opend_listening(host: str, port: int, timeout: float = 0.5) -> bool:
    """Is anything accepting connections on the OpenD port?

    Worth asking because OpenQuoteContext does not answer it: given a
    refused port it retries forever on the calling thread instead of
    raising. Constructed during the viewer's __init__, that means the
    window never appears when OpenD is not running. Measured on this
    machine: the TCP connect is refused in ~1ms while the constructor was
    still spinning in network_manager.poll after twenty seconds.

    A listening socket is not a working OpenD -- it may still be starting
    up or wedged -- so this only rules out the case that is both common
    and cheap to detect.
    """
    s = socket.socket()
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def load_local_ticks(code: str, date_str: str, tf: str,
                     with_offex: bool = False) -> dict | None:
    """Load tick buckets from ticks.db for a given code and date.

    Each price bin contains:
        buy / sell / neutral  — totals (used by heatmap)
        buy_s / buy_m / buy_l — size-stratified buy volume
        sell_s / sell_m / sell_l — size-stratified sell volume

    S/M/L thresholds are computed from the day's volume distribution
    (p33 = small/medium boundary, p67 = medium/large boundary).
    """
    try:
        _, candle_mins = TIMEFRAME_MAP[tf]
        start, end = _day_window(date_str)
        rows = _query_ticks(code, start, end, with_offex)
        if not rows:
            return None
        tm, tl = _size_thresholds(rows)
        return _bucket_ticks(rows, candle_mins, tm, tl)
    except Exception:
        return None


# How far the newest day may grow on frozen S/M/L thresholds before they
# are recomputed from scratch. Bucket count is the clock: it rises with
# elapsed session time, is exact, and costs nothing to read. Doubling
# gives roughly ten full loads across a session rather than one per
# refresh cycle, which at 15s is 240.
_LIVE_RELOAD_GROWTH = 2.0


def load_day_live(code: str, date_str: str, tf: str, with_offex: bool,
                  state: dict) -> dict | None:
    """Today's buckets, re-reading only the part of the tape that is new.

    `state` is owned by the caller and updated in place; pass {} the first
    time. A full read happens when the state is empty, when it describes a
    different (code, tf, date, offex), or when the day has grown past
    _LIVE_RELOAD_GROWTH since the thresholds were taken.

    Incremental reads restart at the start of the bucket holding the last
    tick seen and REPLACE those buckets. Ticks share timestamps, so a
    `ts >` cursor would drop the rest of that second and `ts >=` would
    double-count it; a whole-bucket replace has no boundary case at all.
    """
    try:
        _, candle_mins = TIMEFRAME_MAP[tf]
        start, end = _day_window(date_str)
        key = (code, tf, date_str, with_offex, candle_mins)

        grown = (state.get("buckets")
                 and len(state["buckets"]) >= _LIVE_RELOAD_GROWTH
                 * max(1, state.get("n_at_thresh", 1)))
        if state.get("key") != key or not state.get("buckets") or grown:
            rows = _query_ticks(code, start, end, with_offex)
            if not rows:
                state.clear()
                return None
            tm, tl = _size_thresholds(rows)
            buckets = _bucket_ticks(rows, candle_mins, tm, tl)
            state.clear()
            state.update(key=key, buckets=buckets, thresh=(tm, tl),
                         n_at_thresh=len(buckets),
                         last_ts=_max_ts(rows), full_loads=1)
            return buckets

        last_ts = state["last_ts"]
        re_from = candle_start(last_ts, candle_mins)
        rows = _query_ticks(code, re_from, end, with_offex)
        if rows:
            tm, tl = state["thresh"]
            fresh = _bucket_ticks(rows, candle_mins, tm, tl)
            # A new top-level dict: the one handed out last cycle may still
            # be referenced by whatever drew from it, and mutating that
            # under a consumer is the kind of bug that shows up as a chart
            # changing while nothing asked it to.
            buckets = dict(state["buckets"])
            buckets.update(fresh)          # whole-bucket replace, never add
            state["buckets"] = buckets
            state["last_ts"] = _max_ts(rows)
        return state["buckets"]
    except Exception:
        state.clear()
        return load_local_ticks(code, date_str, tf, with_offex=with_offex)


def _max_ts(rows) -> datetime:
    def _as_dt(v):
        return v if isinstance(v, datetime) else datetime.fromisoformat(str(v))
    return max(_as_dt(r["ts"]) for r in rows)


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


# Trading days each Range setting spans. "7d" is a trading week, five
# sessions -- the label says 1W because that is the calendar span it covers.
RANGE_DAYS = {"1d": 1, "2d": 2, "3d": 3, "7d": 5}

# Bars DCVD measures the CVD slope over.
#
# This was 5, on the grounds that a 1-bar lookback pins 14.7% of bars to
# |DCVD| > 0.9. That figure was measured over ALL bars including pre- and
# post-market, where 88,418 thin bars push it to 17.3%; across RTH alone it
# is 1.6%, and the extended session is not traded here. It also contradicted
# DCVD_SCALE below in this same file, whose 1-bar p99 of 0.93 leaves no room
# for 14.7% of readings above 0.9.
#
# The window is 1 because that is the one that predicts. Over 60 walk-forward
# out-of-sample sessions, entering on |DCVD| >= 0.4 and risking 1 ATR for 1.5
# returns +0.207 R a trade long and +0.247 short, against -0.101 and -0.062
# for entering on every minute. The 5-bar version of the same rule is worth
# nothing: averaging over five minutes removes the imbalance it is measuring.
DCVD_LOOKBACK = 1

# Signed volume a bar needs before its DCVD means anything. Below it a single
# small print swings the ratio to +/-1, so the bar is drawn as a gap instead.
# It costs almost nothing -- 87 of 38,049 RTH bars, 0.2% -- but it keeps the
# panel showing the same quantity the backtest traded, which applies the same
# floor and never trades a bar below it.
DCVD_MIN_SIGNED = 300.0

# The flow gate, from the absorption study. A bar clears it when its signed
# volume, measured against its OWN recent normal, lands in the top fifth:
#
#     x = (buy - sell) / median(buy + sell over the prior 60 bars)
#     G = the 80th percentile of |x| over those same prior bars
#
# DCVD is a RATIO and says nothing about size: a 500-share bar and a
# 50,000-share bar both read 0.5 when two thirds of them is buying. The
# gate is the missing dimension, and over 60 walk-forward sessions the two
# together scored +0.351 a trade against +0.112 for DCVD alone. Encoded as
# opacity rather than a second panel: colour already carries the sign and
# height the ratio, so opacity is the free channel, and the pair reads as
# "the solid ones are the bars with real size behind them".
#
# POST-HOC. That 2x2 was found by looking, not registered in advance, so
# this is drawn to watch and to gather forward evidence -- not yet to act
# on. The pre-registered part is the +/-0.4 line.
DCVD_FLOW_WIN = 60
DCVD_FLOW_PCT = 0.80
_DCVD_DIM_ALPHA = 70      # bars that did not clear the gate
_DCVD_LIT_ALPHA = 215     # bars that did

# Absorption: |x| >= G and the price move |A| standard deviations away
# from what that flow should have delivered. Pre-registered in
# backtest/absorption.py and measured out of sample over 60 sessions --
# where TRADING it lost (-0.361 R long, -0.566 R short, against an
# unconditional -0.101/-0.072). It is drawn because seeing that heavy
# selling is being eaten is worth knowing while watching the tape; it is
# not drawn because it is an entry. The window and the gate are DCVD's
# own constants above, since it is literally the same gate.
ABS_THRESHOLD = 1.5
ABS_ATR_N = 14            # the study's ATR14, the unit A is measured in

# A slower reading drawn behind the bars for context. The bars are each
# minute's own imbalance; this is the direction of the stretch around them.
# It is deliberately not the decision value -- see the note on the window.
DCVD_CONTEXT = 5

# The threshold the backtest validated, drawn on both sides: the long and
# short rules are symmetric, their lift over their own baselines measured at
# +0.308 and +0.309 across reward ratios from 1:1 to 1:2.
DCVD_DECISION = 0.4

# The light reads the newest bar, and what it reads changed.
#
# It used to be the LEVEL alone: >= 0.4 green, 0..0.4 amber, below zero
# red. The flow gate -- the same one that lights the bars -- measures
# better on every RTH minute: over the study's 60 out-of-sample
# sessions, long only and one position at a time, the gate returned
# +0.363 R a trade against +0.264 for the level. The two disagree
# exactly where it matters: 0.4 on thin volume was +0.186, while heavy
# volume that never reached 0.4 was +0.300. The old light was green on
# the weaker of the two and silent on the stronger.
#
# Their union is not the answer either. Green on "0.4 OR heavy" measured
# +0.228, BELOW heavy alone (+0.319), because it drags the thin half
# back in. So the gate is green, the level is demoted to amber, and 0.4
# keeps its line on the panel -- it is still a general reference, it has
# just stopped being what turns the light green on its own.
#
# WHICH COLOUR, THOUGH. The first version of this painted the cleared
# gate "#26a69a" -- the same teal as a bull candle -- so a reading of
# DCVD -0.72 on heavy flow, which is heavy SELLING, arrived in the
# colour this chart uses everywhere else to mean up. A badge whose hue
# contradicts its own number is worse than no badge.
#
# So the light reads the way the rest of the chart reads, on two
# channels instead of a traffic-light palette:
#
#     HUE        the sign -- the same gold/purple pair as the heatmap
#                bins, the delta bars, the far-zoom bodies and the ABS
#                markers. Gold is buying, purple is selling, here too.
#     BRIGHTNESS the flow state -- full for a cleared gate, dimmed for
#                a 0.4 reading on thin volume, exactly as the DCVD bars
#                already dim themselves.
#
# Nothing green, nothing red: neither hue means anything in this
# chart's vocabulary, and the one thing the light must not do is imply
# a direction opposite to the number beside it.
DCVD_LIGHT_HEAVY_A = 255        # the flow gate cleared
DCVD_LIGHT_LEVEL_A = 150        # |DCVD| >= 0.4 but the volume is thin
DCVD_LIGHT_NONE = "#78909c"     # neither -- no sign worth colouring

# Panel colours. The bars carry the sign, so they reuse the chart's own
# buy/sell pair; the context line takes a hue used nowhere else so it cannot
# be read as a third direction, and the decision line is warmer still,
# because a reference line sharing a series' hue is what made the earlier
# panel ambiguous.
_DCVD_CONTEXT_COL = "#b39ddb"
_DCVD_DECISION_COL = "#ffd54f"

# |DCVD| quantiles by window, measured over 38,049 RTH minutes / 104 sessions
# of SOXL (2026-05-19..10-16): {bars: (p75, p90, p99)}. The panel is scaled
# from these rather than auto-ranged, so a reading means the same thing on
# any day -- the property that normalising exists for -- while still filling
# the panel instead of sitting in its middle seventh.
DCVD_SCALE = {
    1:  (0.517, 0.700, 0.93),
    3:  (0.371, 0.523, 0.76),
    5:  (0.316, 0.448, 0.68),
    10: (0.256, 0.365, 0.56),
    15: (0.224, 0.327, 0.51),
    30: (0.180, 0.270, 0.42),
}


def dcvd_scale(n: int) -> tuple[float, float, float]:
    """(p75, p90, p99) for a window, interpolated between measured ones."""
    keys = sorted(DCVD_SCALE)
    if n <= keys[0]:
        return DCVD_SCALE[keys[0]]
    if n >= keys[-1]:
        return DCVD_SCALE[keys[-1]]
    lo = max(k for k in keys if k <= n)
    hi = min(k for k in keys if k >= n)
    if lo == hi:
        return DCVD_SCALE[lo]
    f = (n - lo) / (hi - lo)
    return tuple(DCVD_SCALE[lo][i] + f * (DCVD_SCALE[hi][i] - DCVD_SCALE[lo][i])
                 for i in range(3))

# Resolve NEUTRAL prints to the side inferred from the prevailing midpoint,
# where scripts/classify_ticks.py has computed one. NEUTRAL is 57.5% of SOXL's
# volume and is off-exchange flow rather than noise, so delta without it covers
# about an eighth of the tape; with it, about 92%. Off until the user asks for
# it from the toolbar, because it changes what every flow indicator reports and
# the two should be comparable side by side.
USE_MID_DIR = False


def load_tick_days(code: str, end_date_str: str, tf: str, days: int,
                   cache: dict | None = None,
                   with_offex: bool = False,
                   live_state: dict | None = None) -> tuple[dict, dict]:
    """Merged tick buckets for the `days` calendar days ending at end_date_str.

    Returns (merged_buckets, newly_loaded) where newly_loaded maps date_str ->
    buckets for the days that were read from disk this call, so the caller can
    keep them. Past days never change, and re-reading them is what made this
    expensive: five days measured 13.3s against 0.2-2.8s for one, which is far
    too slow for a fetch on a 1s refresh timer. The newest day is always read
    fresh -- it is still growing -- and is deliberately left out of
    newly_loaded so it is never cached stale.

    Later days win on overlapping buckets. load_local_ticks() spans 28 hours
    (the prior 20:00 through 23:59), so consecutive days overlap by four hours;
    the buy/sell/neutral totals agree, only the adaptive S/M/L split differs,
    and the fresher day's thresholds are the better ones to keep.
    """
    cache = cache or {}
    end = datetime.strptime(end_date_str, "%Y-%m-%d")

    # Step back over CALENDAR days but count TRADING ones. RANGE_DAYS is in
    # trading days, so over a weekend "3D" is five calendar days back -- taking
    # three left the oldest session with no ticks and its CVD flat at zero,
    # which is the same symptom this whole change set out to fix, one layer
    # down. A day with no ticks costs ~0.01s and does not count against the
    # quota. The cap stops a long holiday break from walking back forever.
    collected: list[tuple[tuple, dict]] = []
    seen_trading_days: set = set()
    for back in range(days * 3 + 4):
        if len(seen_trading_days) >= days:
            break
        d = (end - timedelta(days=back)).strftime("%Y-%m-%d")
        is_newest = back == 0
        # with_offex is part of the cache key: a day loaded without the side
        # file has its off_* counts at zero, and must not be served to a caller
        # that asked for them.
        key = (code, tf, d, with_offex)
        if not is_newest and key in cache:
            day = cache[key]
            fresh = False
        elif is_newest and live_state is not None:
            # The one day that cannot be cached, because it is still being
            # written. Read incrementally instead -- see load_day_live.
            day = load_day_live(code, d, tf, with_offex, live_state) or {}
            fresh = True
        else:
            day = load_local_ticks(code, d, tf, with_offex=with_offex) or {}
            fresh = True
        collected.append((key, day) if fresh and not is_newest else (None, day))
        # Count trading days, not calendar days that happen to be non-empty.
        # load_local_ticks spans the prior 20:00 through 23:59, so a Sunday
        # carries the 240 buckets that open Monday's trading day -- counting it
        # as a day of its own retired the quota early and left the oldest
        # session with no ticks, the exact hole this is meant to close.
        seen_trading_days.update(_trading_day_of_bucket(bk) for bk in day)

    merged: dict = {}
    new: dict = {}
    for key, day in reversed(collected):          # oldest first; later wins
        merged.update(day)
        # Only non-empty days are worth caching. A weekend reads in ~0.01s, and
        # caching it would spend one of the bounded cache's slots on nothing.
        if key is not None and day:
            new[key] = day
    return merged, new


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
    return _trading_day_of_bucket(candle_start(bar_end - timedelta(minutes=cm), cm))


def _trading_day_of_bucket(bk):
    """Trading-day date for an already-computed bucket start.

    Split out because _draw_cvd() has the bucket in hand and used to repeat
    this expression -- the profile range and the CVD reset have to agree on
    where a day begins or the two would anchor to different boundaries.
    """
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
    n_days = RANGE_DAYS.get(range_val)
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


def _bucket_arrays(pd_: dict, cache: dict | None = None, key=None):
    """One bucket's price levels as arrays, built once.

    Returns (price, total, buy, sell, neutral, medium), keeping only
    levels whose three directions sum above zero -- a level that sums to
    zero is not tick coverage and must not stop a bar falling back to
    its OHLCV volume.

    With a cache, the entry holds the bucket dict itself and is reused
    while that is the very same object. The incremental tick loader
    replaces whole buckets, so identity separates changed from unchanged;
    holding the object is also what keeps the test honest, since an id()
    kept for something that has been freed can match a new dict at the
    same address.
    """
    if cache is not None:
        hit = cache.get(key)
        if hit is not None and hit[0] is pd_:
            return hit[1]

    prices, tot, buy, sell, neu, med = [], [], [], [], [], []
    for price, counts in pd_.items():
        b = counts.get("buy", 0)
        s_ = counts.get("sell", 0)
        nn = counts.get("neutral", 0)
        t = b + s_ + nn
        if t > 0:
            prices.append(float(price))
            tot.append(t)
            buy.append(b)
            sell.append(s_)
            neu.append(nn)
            med.append(counts.get("buy_m", 0) + counts.get("sell_m", 0))

    out = tuple(np.asarray(x, dtype=float)
                for x in (prices, tot, buy, sell, neu, med))
    if cache is not None:
        if len(cache) > 50_000:          # a long session, many symbols
            cache.clear()
        cache[key] = (pd_, out)
    return out


def _compute_profile_bins(
    kl: pd.DataFrame,
    ticks: dict | None,
    candle_mins: int,
    n_bins: int = 60,
    cache: dict | None = None,
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

    # Gather the buckets, then do everything numpy at once. This loop
    # used to read every price level of every bucket into six python
    # lists -- ~86k dict reads a render -- and before that it called
    # np.digitize once per level. Now a bucket is split into arrays once
    # and kept (see _bucket_arrays), and the per-bar work is a dict
    # lookup and six appends.
    parts: list[tuple] = []
    lens:  list[int]   = []
    rows:  list[int]   = []
    n_bars = len(kl)
    bks = bar_start_keys(kl["time_key"], candle_mins) if ticks else None
    for j in range(n_bars):
        if not ticks:
            break
        bk = bks[j]
        if bk is pd.NaT:
            continue
        pd_ = ticks.get(bk)
        if not pd_:
            continue
        arrs = _bucket_arrays(pd_, cache, bk)
        if arrs[0].size:
            parts.append(arrs)
            lens.append(arrs[0].size)
            rows.append(j)

    # A bar whose levels all fall outside [lo, hi] is NOT tick-covered
    # and keeps its OHLCV fallback, so which bars are covered can only be
    # decided after the range mask -- done here for every bar at once.
    covered = np.zeros(n_bars, dtype=bool)
    if parts:
        p  = np.concatenate([a[0] for a in parts])
        t  = np.concatenate([a[1] for a in parts])
        b  = np.concatenate([a[2] for a in parts])
        sl = np.concatenate([a[3] for a in parts])
        nu = np.concatenate([a[4] for a in parts])
        md = np.concatenate([a[5] for a in parts])
        owner = np.repeat(np.asarray(rows), np.asarray(lens))

        m = (p >= lo) & (p <= hi)
        if m.any():
            covered[np.unique(owner[m])] = True
            used_ticks = True
            # Same bins array and the same digitize as the per-price
            # version, so a price exactly on a boundary lands in the bin
            # it always did.
            bi = np.clip(np.digitize(p[m], bins) - 1, 0, n_bins - 1)
            volumes += np.bincount(bi, weights=t[m], minlength=n_bins)
            stats["total"]   = float(t[m].sum())
            stats["buy"]     = float(b[m].sum())
            stats["sell"]    = float(sl[m].sum())
            stats["neutral"] = float(nu[m].sum())
            stats["medium"]  = float(md[m].sum())

    for j in np.nonzero(~covered)[0]:
        mask  = (centers >= klo[j]) & (centers <= khi[j])
        n_hit = int(mask.sum())
        if n_hit:
            volumes[mask] += kvol[j] / n_hit

    return centers, volumes, used_ticks, stats


def _compute_poc_vah_val(
    centers: np.ndarray,
    volumes: np.ndarray,
    va_pct: float = 0.70,
    poc_idx: int | None = None,
) -> tuple[float, float, float]:
    """Return (poc_price, vah_price, val_price) from a volume profile.

    Uses the standard market-profile algorithm: start at POC and expand the
    Value Area outward (up or down, whichever adds more volume at each step)
    until va_pct of total volume is covered.  VAL ≤ POC ≤ VAH by construction.

    When all bins have equal volume (OHLCV flat distribution) argmax returns
    index 0 (bottom price), which would be misleading; in that case the median
    bin is used as POC instead.

    poc_idx pins the value area to a specific bin instead of the argmax one.
    Its caller is the viewer's POC switch hysteresis: when a held POC is being
    displayed, the value area has to expand from that same bin, or the panel
    would show a POC and a VAH/VAL that disagree about where the peak is.
    """
    if centers.size == 0:
        return 0.0, 0.0, 0.0
    if poc_idx is None:
        poc_idx = int(np.argmax(volumes))
        # OHLCV flat distribution: all bins equal → use median bin as POC
        if np.all(volumes == volumes[poc_idx]):
            poc_idx = len(volumes) // 2
    else:
        poc_idx = int(np.clip(poc_idx, 0, len(volumes) - 1))
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

    CHUNK = 128   # bars per frozen picture; see _ensure_pictures

    def __init__(self):
        super().__init__()
        self._klines:      pd.DataFrame | None = None
        self._buckets:     dict | None         = None
        self._delta:       np.ndarray | None   = None   # per-bar buy-sell
        # True once the view is too far out for the heatmap to read, so
        # the bodies are baked in delta colours instead. Flipping it is
        # what invalidates the pictures; see paint().
        self._far:         bool                = False
        self._candle_mins: int                 = 1
        self._show_heatmap: bool               = True
        self._red_up:      bool                = False   # True = red up, green down
        self._rect:        QRectF              = QRectF()

        # Baked pictures, replayed in order. See _ensure_pictures.
        self._chunks:   list[QPicture]         = []
        self._part:     QPicture | None        = None
        self._part_rng: tuple[int, int]        = (0, 0)
        self._last:     QPicture | None        = None

        # What the baked pictures were drawn from; a change in any of it
        # invalidates all of them.
        self._cfg:    tuple | None             = None
        self._anchor: str                      = ""

        # Per-bar arrays, read out of the frame once in set_data.
        self._o = self._h = self._l = self._c = None
        self._bk: list | None                  = None

    def set_data(self, klines: pd.DataFrame, buckets: dict | None,
                 candle_mins: int, show_heatmap: bool = True,
                 red_up: bool = False,
                 dirty_bars: set | None = None,
                 delta: np.ndarray | None = None) -> None:
        """Hand over the frame; `dirty_bars` says which bars moved.

        dirty_bars is None by default and then nothing is reused -- the
        same full rebuild this did before. A caller that knows what
        changed (the live path) passes the bar indices, and only the
        pictures covering those bars are dropped.

        `delta` is per-bar buy-minus-sell, used only when the view is
        too far out for the heatmap to be legible. None leaves the body
        plain at that zoom, which is what it did before.
        """
        n = len(klines)
        cfg = (candle_mins, show_heatmap, red_up)
        anchor = str(klines["time_key"].iloc[0]) if n else ""

        # A different series, a different setting, or a frame that
        # shrank: nothing baked can be trusted.
        reuse = (dirty_bars is not None and cfg == self._cfg
                 and anchor == self._anchor and self._klines is not None
                 and n >= len(self._klines) > 0)
        if not reuse:
            self._chunks.clear()
            self._part = None
            self._part_rng = (0, 0)
        elif dirty_bars:
            first = min(dirty_bars)
            del self._chunks[first // self.CHUNK:]
            ps, pe = self._part_rng
            if any(ps <= b < pe for b in dirty_bars):
                self._part = None
        # The final bar is live by definition.
        self._last = None

        self._klines       = klines
        self._buckets      = buckets
        self._delta        = delta
        self._candle_mins  = candle_mins
        self._show_heatmap = show_heatmap
        self._red_up       = red_up
        self._cfg          = cfg
        self._anchor       = anchor

        # Read the frame out once. iterrows() over 2641 rows, with a
        # strptime per row, was a large part of the cost of baking.
        if n:
            self._o = klines["open"].to_numpy(float)
            self._h = klines["high"].to_numpy(float)
            self._l = klines["low"].to_numpy(float)
            self._c = klines["close"].to_numpy(float)
            self._bk = list(bar_start_keys(klines["time_key"], candle_mins))
        else:
            self._o = self._h = self._l = self._c = None
            self._bk = None

        if not klines.empty:
            self._rect = QRectF(
                -0.5,
                float(klines["low"].min()),
                float(len(klines)),
                float(klines["high"].max() - klines["low"].min()),
            )
        self.prepareGeometryChange()
        self.update()

    # ── bake pictures per chunk, replay them all on every paint ──────────────

    def _ensure_pictures(self) -> None:
        """Bake whatever set_data dropped, and nothing else.

        Bars [0, n-1) are history and the final bar is live, so the split
        is: complete CHUNK-sized runs of history, then the remainder of
        history, then the one live bar. A new bar moves the previous live
        bar into `part`, which is why `part` is keyed on its range -- and
        once `part` is CHUNK bars long it becomes a chunk and is never
        rebaked.
        """
        n = len(self._klines)
        full = (n - 1) // self.CHUNK            # complete chunks in [0, n-1)
        del self._chunks[full:]
        while len(self._chunks) < full:
            i = len(self._chunks)
            self._chunks.append(self._bake(i * self.CHUNK,
                                           (i + 1) * self.CHUNK))

        rng = (full * self.CHUNK, n - 1)
        if self._part is None or self._part_rng != rng:
            self._part = self._bake(*rng) if rng[1] > rng[0] else None
            self._part_rng = rng

        if self._last is None:
            self._last = self._bake(n - 1, n)

    @staticmethod
    def _spans(a: int, b: int, x0: float, x1: float) -> bool:
        """Does the bar range [a, b) overlap the view span [x0, x1]?

        A bar sits from i-0.35 to i+0.35, so the range occupies
        [a-0.5, b-0.5) in x.
        """
        return a - 0.5 <= x1 and b - 0.5 >= x0

    def _bake(self, a: int, b: int) -> QPicture:
        """One picture covering bars [a, b)."""
        pic = QPicture()
        p   = QPainter(pic)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        buckets     = self._buckets or {}
        o_a, h_a    = self._o, self._h
        l_a, c_a    = self._l, self._c
        bk_a        = self._bk

        # Color scheme: green-up/red-down (Western) OR red-up/green-down (Chinese)
        _bull_col = _RED   if self._red_up else _GREEN
        _bear_col = _GREEN if self._red_up else _RED

        # Pre-compute per-candle data so we can do a two-pass draw
        # (bodies + heatmap first, wicks on top last).
        candle_data: list[tuple] = []
        for i in range(a, b):
            o = float(o_a[i])
            h = float(h_a[i])
            l = float(l_a[i])
            c = float(c_a[i])
            body_lo = min(o, c)
            body_hi = max(o, c)
            if body_hi <= body_lo:
                body_hi = body_lo + 0.0001
            bk = bk_a[i]
            pd_ = buckets.get(bk) if bk is not pd.NaT else None
            candle_data.append((i, o, h, l, c, body_lo, body_hi, pd_))

        # Pass 1 — bodies + heatmap overlay
        p.setPen(Qt.PenStyle.NoPen)
        delta = self._delta
        for i, o, h, l, c, body_lo, body_hi, pd_ in candle_data:
            is_bull  = c >= o
            if self._far and delta is not None and i < len(delta):
                # Too far out for the heatmap to say anything, so the
                # body says delta instead: fill by the sign of the flow,
                # outline in the candle's own colour so direction and
                # flow are two separate readings rather than one.
                d = float(delta[i])
                edge = _qc(_bull_col if is_bull else _bear_col)
                if d > 0:
                    p.setBrush(QBrush(_qc(_UP)))
                elif d < 0:
                    p.setBrush(QBrush(_qc(_DOWN)))
                else:
                    # No signed flow for this bar. Left in its own
                    # colour rather than coloured gold or purple, so a
                    # gap in the tape cannot be read as balanced flow.
                    p.setBrush(QBrush(edge))
                p.setPen(pg.mkPen(edge, width=0))
                p.drawRect(QRectF(i - 0.35, body_lo, 0.7, body_hi - body_lo))
                p.setPen(Qt.PenStyle.NoPen)
            elif self._show_heatmap and pd_:
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
        for i, o, h, l, c, body_lo, body_hi, pd_ in candle_data:
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
            # Body outline, only where the heatmap took the body over.
            # Those bins are buy/sell coloured, which leaves the bar's
            # own direction shown nowhere; the outline is the only place
            # it survives. Drawn here, in the wick pass, so the bins
            # cannot cover it.
            if self._show_heatmap and pd_:
                p.drawRect(QRectF(i - 0.35, body_lo, 0.7,
                                  body_hi - body_lo))

        p.end()
        return pic

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
        Alpha = f(volume density, directional imbalance), on the gamma
        curve described at _HM_GAMMA: _HM_ALPHA_MIN … _HM_ALPHA_MAX.
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
                # Volume share first, through the gamma, then imbalance
                # dims it. Gamma before the imbalance multiplier, not
                # after: the two answer different questions and raising
                # their product to a power would let a one-sided bin
                # borrow brightness it has no volume for.
                imbalance = abs(ratio)                      # 0..1
                dim = _HM_IMB_FLOOR + (1.0 - _HM_IMB_FLOOR) * imbalance
                span = _HM_ALPHA_MAX - _HM_ALPHA_MIN
                alpha = int(min(_HM_ALPHA_MAX,
                                _HM_ALPHA_MIN
                                + span * (vol_frac ** _HM_GAMMA) * dim))
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

        # Which mode the bodies are baked in has to be settled BEFORE
        # baking: the pictures outlive a zoom, so crossing the line is
        # the one moment they stop being valid.
        vr0 = self.viewRect()
        vb = self.getViewBox()
        far = self._far
        if (self._show_heatmap and self._delta is not None
                and vr0 is not None and vb is not None
                and vr0.width() > 0 and vb.width() > 0):
            far = (vb.width() / vr0.width()) < _HM_MIN_PX
        if far != self._far:
            self._far = far
            self._chunks.clear()
            self._part = None
            self._part_rng = (0, 0)
            self._last = None

        self._ensure_pictures()

        # Replay only the chunks the view can see. Qt pays for a
        # QPicture's commands whether or not they land on screen, and at
        # a 200-bar zoom 19 of the 21 chunks are off it. Each chunk's bar
        # range is known, which is what makes this a lookup rather than a
        # clip test. A bar is drawn from i-0.35 to i+0.35, so chunk k
        # occupies [k*CHUNK - 0.5, (k+1)*CHUNK - 0.5).
        vr = self.viewRect()
        if vr is None:
            x0, x1 = float("-inf"), float("inf")
        else:
            x0, x1 = vr.left() - 1.0, vr.right() + 1.0

        for idx, pic in enumerate(self._chunks):
            if idx * self.CHUNK - 0.5 > x1:
                break                       # chunks are in bar order
            if self._spans(idx * self.CHUNK, (idx + 1) * self.CHUNK, x0, x1):
                pic.play(p)
        if self._part is not None and self._spans(*self._part_rng, x0, x1):
            self._part.play(p)
        n = len(self._klines)
        if self._last is not None and self._spans(n - 1, n, x0, x1):
            self._last.play(p)

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

def _merge_klines(base, fresh):
    """Fold `fresh` bars into `base`, newer rows winning on an equal time_key.

    Overwrite, not append-if-absent. The newest bar is still forming, so the
    same time_key comes back repeatedly with a different close/high/low/volume
    -- keeping the first copy would freeze the live bar at whatever it looked
    like when first seen. The pre-existing get_cur_kline supplement had exactly
    that blind spot (it filtered to unseen keys), which went unnoticed only
    because the full history re-pull replaced the whole frame every cycle.
    """
    if base is None or len(base) == 0:
        return fresh.reset_index(drop=True)
    if fresh is None or len(fresh) == 0:
        return base.reset_index(drop=True)
    fresh = fresh.reindex(columns=base.columns)
    out = pd.concat([base, fresh], ignore_index=True)
    out["time_key"] = out["time_key"].astype(str)
    out = out.drop_duplicates("time_key", keep="last")   # keep="last" = fresh wins
    return out.sort_values("time_key").reset_index(drop=True)


_PUSH_COLS = ("open", "high", "low", "close", "volume")


def _norm_time_key(s: str) -> str:
    """Match the fetcher's K_DAY normalisation.

    A 1d bar arrives on the push stream as a bare "YYYY-MM-DD" while the
    fetched frame carries "YYYY-MM-DD 00:00:00" (the tf == "1d" branch in
    DataFetcher.run). Without this they never compare equal, so on 1d the fold
    silently did nothing -- and at a day boundary the bare key sorted ABOVE
    the normalised one, appending a row that skipped a normalisation every
    downstream [:16] slice and strptime is written against.
    """
    return s + " 00:00:00" if len(s) == 10 else s


def _push_values(row) -> dict:
    """OHLCV from a pushed bar, as floats. Raises on anything uncoercible.

    Volume is float, not int. Both the push and the history fetch read
    moomoo's hpVolume, a protobuf double (CurKlinePush.unpack_rsp and
    RequestHistoryKlineQuery.unpack_rsp agree on that), and nothing in
    DataFetcher casts the column down, so the live frame's volume is float64.
    An int() here would silently truncate rather than protect anything.

    Non-finite values are refused along with uncoercible ones: float("nan")
    does NOT raise, so without this a NaN price or volume would fold straight
    into the frame and every reader downstream would have to cope with it.
    """
    out = {}
    for c in _PUSH_COLS:
        v = row.get(c)
        if v is None:
            continue
        v = float(v)
        if not np.isfinite(v):
            raise ValueError(f"non-finite {c}")
        out[c] = v
    return out


def _fold_pushed_bar(df, row) -> tuple:
    """Fold one pushed K-line into a live frame. Returns (frame, appended).

    Which of two observations of the SAME bar is the later one is decided by
    volume, not by either side's clock: within one bar volume only grows, so
    the larger figure was taken later. That is what keeps a Refresh landing
    mid-bar from rolling the forming candle backwards to the snapshot the
    fetch took a second or two earlier.

    Only the newest bar and a brand-new one are considered. A push for a bar
    older than the frame's last is dropped: the stream concerns the bar being
    formed, and a genuine late correction to an older bar is picked up by the
    next full fetch through _merge_klines anyway. That keeps this O(1) on a
    frame of several thousand bars, at roughly one call a second.

    Never mutates `df`. The fetcher thread holds the same frame as its top-up
    cache while this runs on the GUI thread, so an in-place edit here would be
    a cross-thread write to a frame being read; a changed frame comes back as
    a copy and is rebound instead.
    """
    if df is None or len(df) == 0 or not row:
        return df, False
    tk = _norm_time_key(str(row.get("time_key", "")))
    if not tk:
        return df, False
    last = str(df["time_key"].iloc[-1])

    if tk == last:
        # Every coercion is inside the guard, not just the comparison. These
        # run in a QTimer slot, and an exception escaping a PyQt6 slot aborts
        # the process -- a malformed push must cost a dropped bar, not the
        # viewer.
        try:
            vol_new = float(row.get("volume", 0) or 0)
            vol_old = float(df["volume"].iloc[-1] or 0)
            if vol_new < vol_old:
                return df, False      # an earlier observation of this bar
            vals = _push_values(row)
        except (TypeError, ValueError, OverflowError):
            return df, False
        out = df.copy()
        for c, v in vals.items():
            if c in out.columns:
                # By position, not by label: a duplicate index label would
                # write every row that carries it.
                out.iloc[-1, out.columns.get_loc(c)] = v
        return out, False

    if tk > last:
        try:
            vals = _push_values(row)
        except (TypeError, ValueError, OverflowError):
            return df, False
        # Start from the bar before it. Every non-OHLCV column is inherited --
        # including turnover and last_close, which the push does in fact
        # carry; only OHLCV is taken from the push. Filling them with None
        # instead would turn a numeric column into object and change the
        # frame's schema under everything that reads it. Nothing reads those
        # columns today, and the next fetch overwrites the row through
        # _merge_klines within one cycle, so the stale values cost nothing --
        # which is exactly why filling them with None would have gone
        # unnoticed.
        new = df.iloc[-1].to_dict()
        new.update({c: v for c, v in vals.items() if c in df.columns})
        new["time_key"] = tk
        return (pd.concat([df, pd.DataFrame([new], columns=df.columns)],
                          ignore_index=True), True)

    return df, False                  # older than the frame's last bar


class _KlinePushHandler(CurKlineHandlerBase):
    """Hands OpenD's pushed bars to a viewer.

    At module level, not nested inside _start_live, so a test can drive the
    real thing. The nested first version went unexercised -- every test called
    _ingest_kline directly -- and the one line that could only be checked here
    was wrong: CurKlineHandlerBase.on_recv_rsp returns a DATAFRAME, built from
    its own col_list, not the list of dicts that CurKlinePush.unpack_rsp hands
    it. `if not data` on a DataFrame raises, OpenD's push thread catches and
    logs it, and the net effect was the old behaviour plus a traceback roughly
    once a second.

    So the payload is converted explicitly rather than assumed, and the
    viewer's half of the contract stays a list of dicts.
    """

    def __init__(self, viewer, code: str):
        super().__init__()
        self._viewer = viewer
        self._code   = code

    def on_recv_rsp(self, rsp_pb):
        ret, data = super().on_recv_rsp(rsp_pb)
        if ret != RET_OK or data is None:
            return ret, data
        rows = data.to_dict("records") if hasattr(data, "to_dict") else list(data)
        if rows:
            self._viewer._ingest_kline(rows, self._code)
        return ret, data


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
            df = None

            # Live top-up: when the caller hands back last cycle's frame, the
            # window it already covers does not need pulling again. The full
            # re-pull is 4 paginated request_history_kline calls per cycle (1m
            # looks back 5 days, ~7200 bars at 2000/page) -- the bulk of the
            # cycle time, and the reason Refresh could not go below 5s without
            # risking the history-kline rate limit.
            #
            # get_cur_kline returns the most recent _CUR_KLINE_BARS bars, so it
            # bridges any gap shorter than that. Rather than predict whether it
            # will, the gap is DETECTED: if its oldest bar is newer than the
            # cache's newest, bars are missing in between and this falls
            # through to the full fetch. That stays correct however long the
            # app was asleep, paused or disconnected -- no assumption about how
            # much time passed, just a look at what actually came back.
            cached = p.get("cached_klines")
            if not historical and cached is not None and len(cached) > 0:
                try:
                    r0, cur0 = self._ctx.get_cur_kline(
                        code, _CUR_KLINE_BARS, ktype, AuType.NONE)
                    if r0 == RET_OK and cur0 is not None and not cur0.empty:
                        oldest_fresh = str(cur0["time_key"].astype(str).min())
                        newest_cached = str(cached["time_key"].astype(str).max())
                        if oldest_fresh <= newest_cached:
                            df = _merge_klines(cached, cur0)
                            # Keep the window rolling; an incrementally fed
                            # frame would otherwise grow all session.
                            df = df[df["time_key"].astype(str) >= start]
                            df = df.reset_index(drop=True)
                        else:
                            self.error.emit(
                                f"Live top-up gap ({oldest_fresh} > "
                                f"{newest_cached}) — full fetch")
                    elif r0 != RET_OK:
                        self.error.emit(f"get_cur_kline warning: {cur0}")
                except Exception as exc:
                    self.error.emit(f"Live top-up failed ({exc}) — full fetch")

            if df is None:
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
                # select="strength" rather than the detector's default
                # "recent": recency filled every slot from whichever side
                # price last visited, so the lower half of the chart drew
                # nothing even when a dozen blocks were sitting there.
                # Overlapping same-direction blocks collapse to the one
                # with the most volume behind it, and blocks already
                # closed through are capped at one PER SIDE -- the
                # nearest level price could come back to from either
                # direction, and nothing behind it. The detector's default
                # is untouched because backtest/engine.py calls it too.
                raw_obs = detect_order_blocks(warmup, [
                    dict(s, idx=s["idx"] - disp_off,
                         from_idx=s.get("from_idx", s["idx"]) - disp_off)
                    for s in smc_signals
                ] if disp_off > 0 else smc_signals,
                    max_count=p.get("ob_max_count", 4),
                    select="strength",
                    max_breakers=1,
                )
                for b in raw_obs:
                    r = dict(b)
                    r["idx"]     = max(0, b["idx"]      + disp_off)
                    r["bos_idx"] = max(0, b.get("bos_idx", b["idx"]) + disp_off)
                    ob_blocks.append(r)

            # Tick data
            ticks: dict | None = None
            self._new_tick_days: dict = {}
            if historical:
                ticks, self._new_tick_days = load_tick_days(
                    code, date_str, tf, int(p.get("tick_days", 1) or 1),
                    p.get("tick_cache"),
                    with_offex=bool(p.get("with_offex")))
            else:
                # Live mode: load today's ticks from DB first (covers the whole
                # trading day even when the viewer is opened after hours), then
                # use the ticks accumulated in this session to fill any buckets
                # ticks.db doesn't have yet. ticks.db is kept fresh by an
                # external collector on the same feed, so a bucket already
                # present there is authoritative — adding the live snapshot on
                # top would double-count the same real trades.
                today = datetime.now().strftime("%Y-%m-%d")
                # As many days as the Range control accumulates CVD over --
                # loading one while CVD spans three left every bar before the
                # current trading day with no tick bucket, so its delta was 0
                # and the curve sat flat at zero until 20:00.
                ticks, self._new_tick_days = load_tick_days(
                    code, today, tf, int(p.get("tick_days", 1) or 1),
                    p.get("tick_cache"),
                    with_offex=bool(p.get("with_offex")),
                    # Owned by the widget and updated in place. Only one
                    # fetch runs at a time, and the GUI thread does not
                    # touch it while one is in flight.
                    live_state=p.get("live_state"))
                live_snap: dict = p.get("live_ticks") or {}
                for bk, pd_ in live_snap.items():
                    if bk not in ticks:
                        ticks[bk] = {price: dict(counts) for price, counts in pd_.items()}

            self.ready.emit({
                "new_tick_days": self._new_tick_days,
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


class AtrStopParamsDialog(QDialog):
    """ATR period, the multiples to draw, direction, and the anchor price.

    The anchor is blank by default, meaning the latest close -- "if I
    entered now". Typing a fill price switches it to a position already
    open, where the stop belongs relative to what was paid rather than to
    where the price has since gone.
    """

    def __init__(self, period: int, mults: str, direction: str,
                 anchor: float | None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ATR Stop Parameters")

        self._period_spin = QSpinBox()
        self._period_spin.setRange(2, 200)
        self._period_spin.setValue(period)

        self._mult_edit = QLineEdit(mults)
        self._mult_edit.setPlaceholderText("1, 1.5, 2")
        self._mult_edit.setToolTip(
            "Comma-separated ATR multiples; one line is drawn per value")

        self._dir_combo = QComboBox()
        self._dir_combo.addItems(["Long", "Short"])
        self._dir_combo.setCurrentIndex(0 if direction == "bull" else 1)

        self._anchor_edit = QLineEdit("" if anchor is None else f"{anchor:g}")
        self._anchor_edit.setPlaceholderText("latest close")
        self._anchor_edit.setToolTip(
            "Entry price to measure from. Leave blank to use the latest "
            "close.")

        form = QFormLayout()
        form.addRow("ATR period:", self._period_spin)
        form.addRow("Multiples:", self._mult_edit)
        form.addRow("Direction:", self._dir_combo)
        form.addRow("Anchor price:", self._anchor_edit)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def values(self) -> tuple[int, str, str, float | None]:
        """(period, multiples text, "bull"/"bear", anchor or None)."""
        direction = "bull" if self._dir_combo.currentIndex() == 0 else "bear"
        raw = self._anchor_edit.text().strip()
        anchor = None
        if raw:
            try:
                anchor = float(raw)
            except ValueError:          # a typo should not wipe the setting
                anchor = None
        return (self._period_spin.value(), self._mult_edit.text(),
                direction, anchor)


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

def bar_start_keys(time_key, cm: int):
    """Vectorised candle_start() over a time_key column, minus one candle.

    Mirrors core.time_utils.candle_start branch for branch rather than
    using pandas' floor(): floor() is epoch-aligned and candle_start()
    aligns within the hour, which agree only while 60 is divisible by the
    candle size. Every size in TIMEFRAME_MAP happens to satisfy that, so
    the shortcut would work today and break silently the day a 45m is
    added. tests/analysis/test_bar_start_keys.py pins the equivalence.

    Returns an array of datetimes, NaT where the stamp did not parse --
    the callers skip those exactly as the old try/except did.
    """
    ts = pd.to_datetime(time_key, errors="coerce") - pd.Timedelta(minutes=cm)
    if cm >= 1440:
        out = ts.dt.normalize()
    elif cm > 60:
        tot = ts.dt.hour * 60 + ts.dt.minute
        out = ts.dt.normalize() + pd.to_timedelta((tot // cm) * cm, unit="m")
    else:
        out = (ts.dt.floor("h")
               + pd.to_timedelta((ts.dt.minute // cm) * cm, unit="m"))
    return out.dt.to_pydatetime()


# Fetchers that were let go while still running, held until their OS thread
# actually exits. See _retire_fetcher.
_PENDING_FETCHERS: list = []


def _retire_fetcher(worker) -> None:
    """Detach a fetcher whose result is no longer wanted.

    Dropping the last Python reference to a QThread while it is still
    running calls its C++ destructor on a live thread, and Qt aborts the
    process for it ("QThread: Destroyed while thread is still running").
    That is not a theoretical window here: this is called when a fetch is
    abandoned mid-flight, which is precisely when the thread is alive.

    So: disconnect every signal, so no stale callback reaches the window,
    and if the thread has not finished, keep a reference until `finished`
    says the OS thread is gone. analysis/liq_hm_window.py carries the same
    helper for its own workers, for the same reason.
    """
    try:
        worker.ready.disconnect()
        worker.error.disconnect()
        worker.finished.disconnect()
    except Exception:
        pass
    try:
        running = worker.isRunning()
    except Exception:
        running = False          # not a live QThread; nothing to hold
    if not running:
        return
    _PENDING_FETCHERS.append(worker)

    def _release() -> None:
        if worker in _PENDING_FETCHERS:
            _PENDING_FETCHERS.remove(worker)

    worker.finished.connect(_release)


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
        # (code, tf) the frame in _klines belongs to. The live top-up reuses
        # _klines as its cache rather than keeping a second copy, so this is
        # what stops a frame being topped up with another symbol's or another
        # timeframe's bars after a switch.
        self._live_kl_key:  tuple | None        = None
        # Pushed K-lines waiting to be folded in, {time_key: row}. Written on
        # the subscription's callback thread, drained on the GUI thread -- see
        # _ingest_kline. Keyed so that several pushes for the same forming bar
        # between two repaints collapse to the one that is current.
        self._pending_kl:   dict                = {}
        # (key, row) of the last pushed bar actually folded in, re-applied
        # after a fetch lands so the forming candle does not jump back to the
        # fetch's older snapshot of it.
        self._last_kl_push: tuple | None        = None
        self._warmup:       pd.DataFrame | None = None
        self._ticks:        dict | None         = None
        # So the "OpenD not listening" line is logged on transitions
        # rather than on every poll.
        self._opend_warned: bool                = False
        # Incremental cursor for TODAY's ticks, the one day that cannot be
        # cached because it is still being written. See load_day_live.
        self._tick_live_state: dict                = {}
        # Session-profile POC state. POC is a from-scratch argmax on every
        # rebuild with no memory, so on a bimodal profile it teleports between
        # the two peaks as their totals cross (observed: four 5-dollar flips
        # inside four minutes on a genuinely two-cluster day). _poc_held is
        # the currently displayed level, kept until a rival bin beats it by
        # the hysteresis margin; _poc_trail is where it has been, newest
        # first, so the move reads as a direction rather than a surprise.
        # Both reset when the profile is redefined (see _poc_ctx_key).
        self._poc_held:     float | None        = None
        self._poc_trail:    list[float]         = []
        self._poc_ctx_key:  tuple | None        = None
        self._live_ticks:   dict                = defaultdict(
            lambda: defaultdict(lambda: {"buy": 0, "sell": 0, "neutral": 0}))
        self._tick_lock      = threading.Lock()
        self._last_tick_price: float            = 0.0
        self._last_nbbo:      tuple[float, float] = (0.0, 0.0)
        # (base_code, pair_code, base_ref, pair_ref) -- a moment when both legs
        # of an _INVERSE_PAIR were known, refreshed every fetch. The base
        # code rides along so a switch away from the pair symbol cannot be read
        # against the previous one's anchor: _trigger_fetch returns early on a
        # live code change, leaving the old anchor in place until the next load.
        self._pair_anchor: tuple[str, str, float, float] | None = None
        # (code, tf, "YYYY-MM-DD") -> tick buckets for that day. Past days are
        # immutable, so they are read once; the current day is always re-read.
        self._tick_day_cache: dict = {}
        self._fetcher:      DataFetcher | None  = None
        # Whether a fetch is running, tracked separately from the thread's own
        # isRunning(): Qt emits finished() before it clears that flag, so a
        # slot reading it races the worker's teardown. This is set beside
        # start() and cleared in the finished slot, where it is simply a fact
        # already decided rather than a question asked of a dying thread.
        self._fetch_inflight: bool              = False
        # A user request that arrived while one was running. One bit, not a
        # queue: several toolbar changes during a fetch mean one refetch,
        # which reads the toolbar when it fires and so sees the settled state
        # rather than replaying every intermediate one.
        self._refetch_pending: bool             = False
        self._closing: bool                     = False
        self._refresh_timer = QTimer(self)
        # Not _trigger_fetch directly: a timer tick must never queue a
        # refetch. If a fetch outruns the refresh interval, every tick would
        # queue one and the next would start the instant the last finished --
        # a permanent back-to-back loop against the history-kline rate limit.
        # Dropping the tick is the backpressure that keeps it to one fetch
        # per completion.
        self._refresh_timer.timeout.connect(self._on_refresh_tick)
        self._redraw_timer = QTimer(self)
        self._redraw_timer.timeout.connect(self._on_redraw_tick)
        self._candle_mins   = 1
        self._smc_signals:  list[dict]          = []
        self._fvg_gaps:     list[dict]          = []
        self._ob_blocks:    list[dict]          = []
        self._liq_hm_window: "LiqHmWindow | None" = None
        self._mvp_window = None
        self._dom_window:  QWidget | None      = None  # DOM depth-of-market window
        self._trade_record: dict | None         = None  # active trade review
        self._live_code: str = ""        # code currently subscribed for tick push
        self._chandelier_period:     int   = 20
        self._chandelier_multiplier: float = 2.0
        self._chandelier_direction:  str   = "bull"   # "bull" | "bear"
        # Fixed ATR stop: period 14 to match the risk unit every backtest in
        # this project measures R against, so a level drawn here is the same
        # 1R those results are quoted in.
        self._atrstop_period:    int = 14
        self._atrstop_mults:     str = "1, 1.5, 2"
        self._atrstop_direction: str = "bull"
        self._atrstop_anchor:    float | None = None   # None = latest close
        self._atrstop_items:     list = []
        self._live_kl_sub: object = None  # kline SubType subscribed alongside TICKER/QUOTE
        # Track which (code, tf) was last auto-ranged; prevents live-refresh
        # from resetting the user's manual pan/zoom on every tick.
        self._last_chart_key: tuple             = ("", "")
        # What the bars currently in self._klines actually are, taken from
        # the fetch that delivered them. The toolbar is NOT this: it changes
        # the moment the user picks a timeframe, while the bars stay the old
        # ones until the fetch lands, and a repaint in between (the redraw
        # timer, every second or two in Live) would otherwise record the new
        # timeframe as "already ranged" against the old bars -- spending the
        # reset on them, and leaving the real chart off-screen until Home.
        self._klines_key: tuple                 = ("", "")

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

        # Build UI
        self._build_toolbar(args)
        self._build_central()
        self._build_statusbar()

        # Anything wrong with config/viewer_qt.toml -- an unknown key, a
        # value of the wrong type -- is said out loud here rather than
        # swallowed, because a config whose typos do nothing looks like it
        # worked. It has to wait for the status bar: _log writes to it.
        for _msg in vcfg.problems():
            self._log(f"viewer_qt.toml: {_msg}")

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

        # An indicator that starts checked never emits stateChanged, so the
        # handler that acts on it never runs: the Options boxes were ticked
        # while the chain was never polled and no wall was ever drawn. Run
        # it once here, with the toolbar built and the plots in place.
        self._on_option_indicator_toggle()

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
        self._tf_combo.setCurrentText(getattr(args, "tf", "1m") or "1m")
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
        tb1.addWidget(_lbl("Refresh (s):"))
        self._refresh_spin = QSpinBox()
        # Floor was 5s because every live cycle re-pulled the whole lookback
        # window: 4 paginated request_history_kline calls, which at 1s would
        # have been ~240 history calls a minute. Live cycles now top up a
        # cached frame with one get_cur_kline instead, so the API cost per
        # cycle is one call and 1s is affordable. A full fetch still happens on
        # connect, on a code/timeframe switch, and whenever a gap is detected.
        self._refresh_spin.setRange(1, 300)
        self._refresh_spin.setValue(getattr(args, "refresh", 15) or 15)
        self._refresh_spin.setFixedWidth(55)
        self._refresh_spin.valueChanged.connect(self._on_refresh_changed)
        tb1.addWidget(self._refresh_spin)

        # Redraw (Live). Separate from Refresh because repainting needs no
        # API call at all: ticks are pushed, so the tape-driven overlays
        # can run far ahead of the K-line top-up without costing quota.
        # The floor is 0.5s because a render is ~220ms and 0.3s would put
        # the GUI thread at a 73% duty cycle.
        tb1.addWidget(_lbl("Redraw (s):"))
        self._redraw_spin = QDoubleSpinBox()
        self._redraw_spin.setRange(0.5, 10.0)
        self._redraw_spin.setSingleStep(0.5)
        self._redraw_spin.setDecimals(1)
        self._redraw_spin.setValue(vcfg.num("redraw_secs"))
        self._redraw_spin.setFixedWidth(60)
        self._redraw_spin.setToolTip(
            "How often the chart repaints from ticks already received. "
            "Costs no API calls; skipped when no new prints arrived.")
        self._redraw_spin.valueChanged.connect(self._on_redraw_changed)
        tb1.addWidget(self._redraw_spin)

        tb1.addSeparator()

        # Connect / Stop
        self._conn_btn = QPushButton("Connect")
        self._conn_btn.setCheckable(True)
        self._conn_btn.clicked.connect(self._on_connect_toggle)
        tb1.addWidget(self._conn_btn)

        self._ind_checks: dict[str, QCheckBox | QRadioButton] = {}
        # Which overlays a chart opens with lives in config/viewer_qt.toml,
        # not here -- see analysis/viewer_config.py for the key list and
        # the fallbacks used when the file is absent. Complaints about that
        # file are reported once the status bar exists, at the end of
        # __init__: _log writes to it, and it is built after this.

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
            cb.setChecked(vcfg.flag(key))
            cb.stateChanged.connect(self._on_indicator_toggle)
            self._ind_checks[key] = cb
            tb_smc.addWidget(cb)
            if key == "fvg":
                self._fvg_min_pct = QDoubleSpinBox()
                self._fvg_min_pct.setRange(0.05, 5.0)
                self._fvg_min_pct.setSingleStep(0.05)
                self._fvg_min_pct.setValue(vcfg.num("fvg_min_pct"))
                self._fvg_min_pct.setDecimals(2)
                self._fvg_min_pct.setSuffix("%")
                self._fvg_min_pct.setFixedWidth(72)
                self._fvg_min_pct.setEnabled(vcfg.flag(key))
                self._fvg_min_pct.setToolTip("FVG minimum gap size (% of price)")
                self._fvg_min_pct.valueChanged.connect(self._trigger_fetch)
                tb_smc.addWidget(self._fvg_min_pct)
            if key == "ob":
                self._ob_max_count = QSpinBox()
                self._ob_max_count.setRange(1, 20)
                self._ob_max_count.setValue(vcfg.num("ob_max_count"))
                self._ob_max_count.setFixedWidth(50)
                # follow the checkbox: _on_indicator_toggle only corrects
                # this on the next toggle, so a default-on OB would sit
                # drawn with its quota spinbox greyed out until touched.
                self._ob_max_count.setEnabled(vcfg.flag(key))
                self._ob_max_count.setToolTip(
                    "OB quota per side of price (strongest by volume). "
                    "Blocks already broken are capped at 1 per side on "
                    "top of this, as retest levels.")
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
            ("dcvd",      "DCVD"),  # normalised CVD slope, bounded -1..+1
            ("absorb",    "ABS"),   # absorption markers on the price chart
            ("offex",     "OffEx"), # off-exchange flow as its own CVD line
            ("ema",       "EMA"),
            ("avwap",     "AVWAP"), # Anchored VWAP -- cumulative from a user-picked date
            ("vol",       "MAVOL"), # Volume subplot toggle
            ("chandelier", "Chandelier"), # ATR trailing-stop suggestion, bottom-right corner label
            ("atr_stop",  "ATR Stop"),  # fixed k x ATR stop lines from an anchor
        ]:
            cb = QCheckBox(label)
            cb.setChecked(vcfg.flag(key))
            cb.stateChanged.connect(self._on_indicator_toggle)
            self._ind_checks[key] = cb
            tb_ind.addWidget(cb)
            if key == "dcvd":
                # 1 is the window that predicts; widening it is for looking
                # around, not for deciding. The panel rescales with the
                # window so either stays readable.
                self._dcvd_spin = QSpinBox()
                self._dcvd_spin.setRange(1, 30)
                self._dcvd_spin.setValue(vcfg.num("dcvd_lookback"))
                self._dcvd_spin.setFixedWidth(48)
                self._dcvd_spin.setToolTip(
                    "DCVD lookback, in bars (the coloured bars).\n"
                    f"The {DCVD_CONTEXT}-bar line behind them is context "
                    "only.\n\n"
                    "Over 60 walk-forward out-of-sample sessions, entering "
                    "on\n"
                    f"|DCVD| >= {DCVD_DECISION} at a 1-bar window returned "
                    "+0.207 R long\n"
                    "and +0.247 R short, risking 1 ATR to make 1.5. The same"
                    "\nrule on a 5-bar window is worth nothing: averaging "
                    "over\nfive minutes removes the imbalance it measures.\n\n"
                    "The y-axis follows the window (measured p99), so "
                    "changing it\nneither squashes the bars nor clips their "
                    "peaks.\n\n"
                    "BRIGHT vs DIM is a second rule, and the stronger one. "
                    "A bar\nis bright when this minute's signed volume "
                    f"cleared the flow\ngate: |x| in the top "
                    f"{100 - int(DCVD_FLOW_PCT * 100)}% of its own trailing "
                    f"{DCVD_FLOW_WIN} bars,\nwhere x = (buy-sell) / median("
                    "buy+sell). Over the same 60\nout-of-sample sessions, "
                    "following a bright bar's own sign\nreturned +0.352 R "
                    "short on heavy selling and +0.285 R long on\nheavy "
                    "buying -- against +0.207/+0.247 for the height rule "
                    "above.\n\n"
                    "Height and brightness answer different questions: the "
                    "height\nis the imbalance RATIO, the brightness is "
                    "whether the volume\nbehind it was large for this stock "
                    "right now. Only at a 1-bar\nwindow do the bar's sign "
                    "and the gate's x agree bar for bar.")
                self._dcvd_spin.valueChanged.connect(
                    self._on_dcvd_lookback_changed)
                tb_ind.addWidget(self._dcvd_spin)
            if key == "absorb":
                cb.setToolTip(
                    "Absorption markers on the price chart.\n\n"
                    "A hollow triangle appears where heavy flow failed to "
                    "move\nprice: the bar cleared the same flow gate that "
                    "lights the\nDCVD bars, AND its move was at least "
                    f"{ABS_THRESHOLD} standard deviations\naway from what "
                    "that flow should have delivered.\n\n"
                    "Gold triangle below the bar = heavy SELLING that did "
                    "not push\nprice down. Purple above = heavy BUYING that "
                    "did not lift it.\n\n"
                    "THIS IS NOT AN ENTRY. Measured over 60 walk-forward "
                    "out-of-\nsample sessions, trading these marks lost "
                    "-0.361 R long and\n-0.566 R short, against an "
                    "unconditional -0.101/-0.072. At the\nsame gate, going "
                    "WITH the flow earned +0.352/+0.285. The mark\nis here "
                    "to show you what the tape is doing, not what to do.\n\n"
                    "Calibrated on 1-minute RTH bars; on other timeframes "
                    "the\nthreshold has not been measured.\n\n"
                    "RTH only, and the gate is read off the SAME TIME OF "
                    "DAY in\nprevious sessions -- the top fifth of the net "
                    "flow that this\n15 minutes usually delivers. It needs "
                    "at least 3 loaded\nsessions before any mark can "
                    "appear.")
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
                    lambda: self._redraw())
                tb_ind.addWidget(self._avwap_anchor_edit)
                self._avwap_sigma_spin = QSpinBox()
                self._avwap_sigma_spin.setRange(0, 3)
                self._avwap_sigma_spin.setValue(vcfg.num("avwap_sigma"))
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
                    lambda: self._redraw())
                tb_ind.addWidget(_lbl("sigma:"))
                tb_ind.addWidget(self._avwap_sigma_spin)
            if key == "atr_stop":
                atrstop_settings_btn = QPushButton("\u2699")
                atrstop_settings_btn.setFixedWidth(28)
                atrstop_settings_btn.setToolTip(
                    "ATR stop parameters (period, multiples, direction, "
                    "anchor price)")
                atrstop_settings_btn.clicked.connect(self._on_atrstop_settings)
                tb_ind.addWidget(atrstop_settings_btn)
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
        cb_red_up.setChecked(vcfg.flag("red_up"))
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
        _sess_on = vcfg.sessions()
        for key, label in [
            ("regular", "Regular"), ("pre", "Pre"),
            ("post", "Post"),       ("night", "Night"),
        ]:
            cb = QCheckBox(label)
            cb.setChecked(key in _sess_on)
            cb.stateChanged.connect(self._on_session_toggle)
            self._ind_checks[f"sess_{key}"] = cb
            tb_sess.addWidget(cb)

        tb_sess.addSeparator()

        # Shared anchor for the session volume profile and the CVD reset
        range_lbl = _lbl("Range:")
        range_lbl.setToolTip(
            "Trading days covered by the session volume profile, and the\n"
            "window CVD accumulates over before resetting. 1D is one session,\n"
            "1W is five. Both read the same setting so a level read off the\n"
            "profile and the CVD under it describe the same stretch of tape.")
        tb_sess.addWidget(range_lbl)
        self._range_group = QButtonGroup(self)
        for val, label in [("1d", "1D"), ("2d", "2D"), ("3d", "3D"), ("7d", "1W")]:
            rb = QRadioButton(label)
            rb.setChecked(val == vcfg.get("session", "range"))
            rb.toggled.connect(self._on_range_radio_toggled)
            self._ind_checks[f"range_{val}"] = rb
            self._range_group.addButton(rb)
            tb_sess.addWidget(rb)

        tb_sess.addSeparator()
        tb_sess.addWidget(_lbl("Bins:"))
        self._profile_bins_spin = QSpinBox()
        self._profile_bins_spin.setRange(20, 300)
        self._profile_bins_spin.setValue(vcfg.num("profile_bins"))
        self._profile_bins_spin.setFixedWidth(48)
        self._profile_bins_spin.setToolTip(
            "Number of price bins in the session volume profile (right panel) --\n"
            "higher = thinner bars, finer price resolution.")
        self._profile_bins_spin.valueChanged.connect(self._on_range_changed)
        tb_sess.addWidget(self._profile_bins_spin)

        tb_sess.addWidget(_lbl("POCs:"))
        self._poc_trail_spin = QSpinBox()
        self._poc_trail_spin.setRange(1, 6)
        self._poc_trail_spin.setValue(vcfg.num("poc_trail"))
        self._poc_trail_spin.setFixedWidth(40)
        self._poc_trail_spin.setToolTip(
            "How many POCs to mark in the session profile, newest first.\n"
            "POC1 is the current one, POC2 the level it moved off, and so on,\n"
            "so the trail shows which way value is migrating. Older ones fade\n"
            "and are dotted. The trail resets when the profile is redefined\n"
            "(symbol / date / range / bins / session filter).")
        self._poc_trail_spin.valueChanged.connect(
            lambda _: self._rebuild_session_profile())
        tb_sess.addWidget(self._poc_trail_spin)

        tb_sess.addWidget(_lbl("Hold:"))
        self._poc_hyst_spin = QSpinBox()
        self._poc_hyst_spin.setRange(0, 50)
        self._poc_hyst_spin.setValue(vcfg.num("poc_hysteresis_pct"))
        self._poc_hyst_spin.setSuffix("%")
        self._poc_hyst_spin.setFixedWidth(56)
        self._poc_hyst_spin.setToolTip(
            "POC switch hysteresis. A rival volume cluster has to hold this\n"
            "much more volume than the one the POC currently sits in before\n"
            "the POC moves to it (clusters, not single bins -- re-binning\n"
            "makes one bin's volume far too noisy to gate on).\n"
            "Without it a two-cluster profile flips the POC back and forth\n"
            "every refresh: 2026-09-10 flipped four times across $5 in twelve\n"
            "minutes, which at the 5%% default becomes one move.\n"
            "0 disables it -- POC then tracks the raw argmax every rebuild.")
        self._poc_hyst_spin.valueChanged.connect(
            lambda _: self._rebuild_session_profile())
        tb_sess.addWidget(self._poc_hyst_spin)

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
            cb.setChecked(vcfg.flag(key))
            cb.setToolTip(tip)
            cb.stateChanged.connect(self._on_tick_size_toggle)
            self._ind_checks[key] = cb
            tb3.addWidget(cb)

        tb3.addSeparator()
        cb_vp = QCheckBox("VP")
        cb_vp.setChecked(vcfg.flag("tick_vp"))
        cb_vp.setToolTip(
            "Tick profile: show the hovered candle (or selected range) as a\n"
            "volume profile -- one bar per price level for TOTAL traded volume,\n"
            "with POC / VAH / VAL -- instead of the buy/sell order-flow split.\n"
            "Bars are linear here, not log-scaled, so bar length is proportional\n"
            "to volume the way a profile is meant to be read.\n"
            "Takes precedence over Net. The S/M/L filters still apply; the Top\n"
            "highlight does not (it marks each side separately, and this mode\n"
            "merges them).")
        cb_vp.stateChanged.connect(self._on_tick_size_toggle)
        self._ind_checks["tick_vp"] = cb_vp
        tb3.addWidget(cb_vp)

        cb_net = QCheckBox("Net")
        cb_net.setChecked(vcfg.flag("tick_net"))
        cb_net.setToolTip(
            "Tick profile: show a single net (buy − sell) bar per price level\n"
            "instead of separate diverging buy/sell bars")
        cb_net.stateChanged.connect(self._on_tick_size_toggle)
        self._ind_checks["tick_net"] = cb_net
        tb3.addWidget(cb_net)

        tb3.addSeparator()
        tb3.addWidget(_lbl("Top:"))
        self._tick_top_n = QSpinBox()
        self._tick_top_n.setRange(1, 10)
        self._tick_top_n.setValue(int(vcfg.num("tick_top_n")))
        self._tick_top_n.setFixedWidth(44)
        self._tick_top_n.setToolTip(
            "Tick profile: gold outline on the N price levels that traded the\n"
            "most, PER SIDE -- N biggest buy levels and N biggest sell levels,\n"
            "chosen independently, so both sides are always represented.\n"
            "The price is printed above each marked row.\n\n"
            "This replaced a ratio threshold that did not do what it said.\n"
            "Only 8.4% of raw price levels carry both buy and sell volume, so\n"
            "the ratio was almost always dividing by the 1-share floor: at the\n"
            "default 3.0 about 89 levels a candle qualified, every candle hit\n"
            "the display cap, and 72.7% of the time the three levels drawn\n"
            "were simply the three largest bars. This says that outright.")
        self._tick_top_n.valueChanged.connect(self._on_tick_size_toggle)
        tb3.addWidget(self._tick_top_n)

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

        self._mvp_btn = QPushButton("Multi-Day VP")
        self._mvp_btn.setCheckable(True)
        self._mvp_btn.setChecked(False)
        self._mvp_btn.setToolTip(
            "Opens a window with the last few sessions' volume profiles\n"
            "side by side, value-area edges joined between them -- for\n"
            "watching which way value is migrating.")
        self._mvp_btn.clicked.connect(self._on_mvp_toggle)
        tb3.addWidget(self._mvp_btn)

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
            # Walls draw on the chart; Alerts fire notifications. Turning the
            # drawing on by default is harmless, starting a notification
            # stream unasked is not, so only the first is in the set.
            cb.setChecked(vcfg.flag(key))
            cb.stateChanged.connect(self._on_option_indicator_toggle)
            self._ind_checks[key] = cb
            tb_opt.addWidget(cb)

        tb_opt.addSeparator()

        # Shared controls -- both indicators come from the same poll cycle.
        tb_opt.addWidget(_lbl("DTE:"))
        self._option_dte_spin = QSpinBox()
        self._option_dte_spin.setRange(0, 30)
        self._option_dte_spin.setValue(vcfg.num("option_dte"))
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
        self._option_wall_count_spin.setValue(vcfg.num("option_wall_count"))
        # 4 per side reads the GEX structure, not just its peak
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
        self._option_interval_spin.setValue(vcfg.num("option_poll_secs"))
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
        self._option_threshold_spin.setValue(vcfg.num("option_threshold"))
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
        # Kept so the whole stack can be folded away. Six rows is roughly a
        # fifth of the window's height, and none of it changes between
        # trades -- the chart is what is being read.
        self._toolbar_rows = toolbar_rows

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
        self._plot_cvd.vb.sigYRangeChanged.connect(lambda *_: self._sync_cvd_fill())
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

        self._plot_dcvd: pg.PlotItem = self._chart_widget.addPlot(row=5, col=0)
        self._plot_dcvd.showGrid(x=True, y=True, alpha=0.10)
        self._plot_dcvd.setLabel("left", "DCVD", **{"color": _FG})
        # DCVD lives inside +/-1, and pyqtgraph reads a range below 1 as a
        # reason to switch the axis into milli-units. With no unit string to
        # attach a prefix to it degrades into a "(x0.001)" suffix on the
        # label and ticks printed as 500 where the value is 0.5 -- while the
        # crosshair, which is not scaled, says 0.45 on the same pixel. The
        # 0.40 decision line then draws against ticks reading 400. The axis
        # has to speak the same numbers as the threshold it is there to show.
        self._plot_dcvd.getAxis("left").enableAutoSIPrefix(False)
        self._plot_dcvd.getAxis("left").setTextPen(_qc(_FG))
        self._plot_dcvd.getAxis("bottom").setTextPen(_qc(_FG))
        self._plot_dcvd.setMenuEnabled(False)
        self._plot_dcvd.setXLink(self._plot_c)
        # Fixed, not auto-ranged: a DCVD of 0.4 has to mean the same thing on
        # a quiet day as a busy one, and auto-ranging would throw that away.
        # But fixed AT +/-1.05 wasted the panel -- at the old 10-bar window
        # the median |DCVD| is 0.148 and 97.6% of bars never reach the 0.5
        # guides. So it is fixed PER WINDOW, from measured quantiles.
        self._plot_dcvd.vb.setMouseEnabled(y=False)
        self._dcvd_guides: list = []
        self._apply_dcvd_scale()

        # Row stretch factors — main chart is fixed; subplot rows collapse to
        # zero height when their indicator is off instead of leaving a gap.
        self._chart_widget.ci.layout.setRowStretchFactor(0, 5)
        # Read from the checkboxes, not from literals. These were hardcoded
        # and hand-synced to the default-on set -- the "CVD on by default"
        # comment was that sync, performed once and never repeated. Adding
        # DCVD to the set then checked its box while leaving its row
        # collapsed, because a box that starts checked never emits
        # stateChanged and the handler that reveals the row never ran.
        # Now that the set comes from config/viewer_qt.toml, any literal
        # here would go stale the first time the file is edited.
        for plot, row, key in ((self._plot_v,    1, "vol"),
                               (self._plot_kd,   2, "kd"),
                               (self._plot_cvd,  3, "cvd"),
                               (self._plot_adi,  4, "adi"),
                               (self._plot_dcvd, 5, "dcvd")):
            self._set_subplot_row_visible(plot, row, self._ind(key))

        # ── Graphics items ────────────────────────────────────────────────────
        self._candle_item = CandlestickItem()
        self._plot_c.addItem(self._candle_item)

        self._fvg_item = FvgItem()
        self._plot_c.addItem(self._fvg_item)

        self._ob_item = ObItem()
        self._plot_c.addItem(self._ob_item)

        # Last frame's drawing inputs, for _dirty_bars. Holding the
        # buckets mapping is deliberate: the id() comparison is only
        # valid while those dicts are alive.
        # The buckets as they came from ticks.db / the fetch, before
        # _render merges the pushed live ones in on top. _render
        # overwrites self._ticks with the merged result, so this is the
        # only mapping that can safely be handed back to it.
        self._db_ticks: dict | None = None
        # Bumped by the TICKER handler; _render records what it drew. A
        # redraw with no new prints is 220ms for an identical picture.
        self._tick_seq: int = 0
        self._drawn_tick_seq: int = -1

        self._prev_frame: dict | None = None
        # The dirty set for the render in progress, and the per-bar flow
        # totals carried between renders. See _dirty_bars / _bar_flow.
        self._frame_dirty: set | None = None
        self._flow_cache: dict | None = None
        # bucket key -> (bucket dict, its arrays); see _bucket_arrays.
        # (kind, item, y) for every label the profile panel pins to an
        # edge. Replaced wholesale by each rebuild and read by the one
        # permanent connection in _pin_profile_labels.
        self._profile_pins: list = []

        self._profile_cache: dict = {}
        # When the profile panel was last rebuilt; see the throttle in
        # _render.
        self._profile_drawn_ms: float = -1e9

        self._bos_items:     list = []  # PlotCurveItem + TextItem per signal
        # Pool of delta labels, one per candle. Kept in the scene across
        # renders and reused -- see _draw_delta.
        self._delta_items:   list = []  # TextItem per candle
        self._delta_used:    int  = 0   # how many of the pool are in use
        self._delta_x:  np.ndarray = np.empty(0)  # bar index per used label
        self._delta_vis: tuple[int, int] = (0, 0)  # pool span now visible
        self._ema_items:     list = []  # PlotCurveItem per EMA period
        self._avwap_items:   list = []  # anchored-VWAP curve + anchor line + label
        self._option_wall_items: list = []  # Call/Put Wall + Zero Gamma lines -- cleared+redrawn each update
        self._option_wall_labels: list = []  # (price, TextItem) subset of the above, for collision layout
        self._option_wall_x = 0              # last bar the wall labels right-anchor to
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

        self._vline_dcvd = pg.InfiniteLine(angle=90, movable=False, pen=cross_pen)
        self._vline_dcvd.setVisible(False)
        self._plot_dcvd.addItem(self._vline_dcvd, ignoreBounds=True)

        self._dcvd_label = pg.TextItem(
            text="", color=_FG,
            fill=pg.mkBrush(_qc(_BG_TIP, 200)),
            anchor=(0.0, 0.5),
        )
        self._dcvd_label.setFont(QFont("Monospace", 8))
        self._dcvd_label.setZValue(100)
        self._dcvd_label.setVisible(False)
        self._plot_dcvd.addItem(self._dcvd_label, ignoreBounds=True)

        self._dcvd_items: list = []
        self._dcvd_arr: np.ndarray | None = None

        self._abs_items: list = []

        self._offex_arr: np.ndarray | None = None
        self._offex_label = pg.TextItem(
            text="", color="#ab47bc",
            fill=pg.mkBrush(_qc(_BG_TIP, 200)),
            anchor=(0.0, 0.5),
        )
        self._offex_label.setFont(QFont("Monospace", 8))
        self._offex_label.setZValue(100)
        self._offex_label.setVisible(False)
        self._plot_cvd.addItem(self._offex_label, ignoreBounds=True)

        # Last candle index shown in the tick profile (used to re-render on
        # order-size filter toggle without waiting for the next mouse move).
        self._last_hover_idx: int | None = None

        # Profile panels: a horizontal line that follows the main chart's price
        # while the cursor is over the chart, and the panel's own cursor once it
        # is over the panel -- see _on_profile_hover.
        #
        # The price rides on the line as an InfiniteLine label rather than a
        # separate TextItem: both panels call clear() on every rebuild and
        # re-add the line afterwards, in seven places between them, and a
        # second item would have to be re-added at every one of them.
        def _hline() -> pg.InfiniteLine:
            ln = pg.InfiniteLine(
                angle=0, movable=False,
                pen=pg.mkPen(_CROSS, width=1, style=Qt.PenStyle.DashLine),
                label="{value:.2f}",
                # anchors, or the price runs off the panel. InfLineLabel's
                # default for a horizontal line is [(0.5, 0), (0.5, 1)] -- x
                # centred on the position -- so at 2% from the left edge half
                # the box hangs outside and gets clipped ("166.74" read "6.74").
                # x=0 left-aligns it so it grows inward; the y pair keeps
                # pyqtgraph's own above/below flip.
                labelOpts={"position": 0.02, "color": _CROSS,
                           "fill": _qc(_BG_TIP, 200), "movable": False,
                           "anchors": [(0, 0), (0, 1)]},
            )
            ln.setVisible(False)
            return ln

        self._profile_hline = _hline()
        self._tick_profile_hline = _hline()

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

        # Inverse-pair price tag: sits just BELOW the crosshair line (anchor
        # top-left) while the gold one moves just above it, so the two prices
        # read as one pair straddling the level rather than a stack of numbers.
        self._pair_price_label = pg.TextItem(
            text="", color=_PAIR_COL,
            fill=pg.mkBrush(_qc(_BG_TIP, 180)),
            anchor=(0.0, 0.0),   # left edge, top -> body hangs below the point
        )
        self._pair_price_label.setFont(QFont("Monospace", 7))
        self._pair_price_label.setZValue(100)
        self._pair_price_label.setVisible(False)
        self._plot_c.addItem(self._pair_price_label, ignoreBounds=True)

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
        self._tick_profile_widget.scene().sigMouseMoved.connect(
            lambda pos: self._on_profile_hover(
                pos, self._tick_profile_widget, self._tick_profile_hline))
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

        # ── DCVD light (bottom-right, above the chandelier label) ─────────────
        # Lives on the price chart rather than in the DCVD panel: at 7pt in a
        # strip a few dozen pixels tall it could not be read, and the reading
        # it carries is about the price in front of you.
        self._dcvd_light = pg.TextItem(
            anchor=(1.0, 1.0),
            fill=pg.mkBrush(_qc(_BG_TIP, 200)),
        )
        self._dcvd_light.setFont(QFont("Monospace", 13, QFont.Weight.Bold))
        self._dcvd_light.setZValue(61)
        self._dcvd_light.setVisible(False)
        self._plot_c.addItem(self._dcvd_light, ignoreBounds=True)
        self._plot_c.vb.sigRangeChanged.connect(self._pin_dcvd_light)
        self._plot_c.vb.sigRangeChanged.connect(self._cull_delta_labels)

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
        # Wall-label collision layout depends on the pixel-per-price scale, so
        # it has to be recomputed on every zoom/pan, not just on redraw.
        self._plot_c.vb.sigRangeChanged.connect(self._pin_option_wall_labels)

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
        # The only pin connection there is. Each rebuild used to connect
        # its own closures and disconnect the previous set, and the
        # disconnected ones stayed reachable -- four TextItems a render.
        # See _pin_profile_labels.
        self._profile_widget.getViewBox().sigRangeChanged.connect(
            self._pin_profile_labels)
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
        self._profile_widget.scene().sigMouseMoved.connect(
            lambda pos: self._on_profile_hover(
                pos, self._profile_widget, self._profile_hline))
        right_layout.addWidget(self._profile_widget, 2)

        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)

        self.setCentralWidget(splitter)

    # ── Status bar ────────────────────────────────────────────────────────────

    def _build_statusbar(self) -> None:
        self._status = QStatusBar()
        self._status.setStyleSheet(
            f"QStatusBar {{ background: {_BG_BAR}; color: {_FG}; font-size: 11px; }}")
        # The fold control lives here because the status bar is the one
        # strip that never hides: put it on a toolbar and folding them
        # would take the way back with it.
        self._fold_btn = QPushButton("\u25b2 Toolbars")
        self._fold_btn.setFlat(True)
        self._fold_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._fold_btn.setToolTip(
            "Hide the toolbars and give the space to the chart  (Ctrl+M)")
        self._fold_btn.setStyleSheet(
            f"QPushButton {{ color: {_GREY}; border: none; padding: 0 8px; }}"
            f"QPushButton:hover {{ color: {_FG}; }}")
        self._fold_btn.clicked.connect(self._toggle_toolbars)
        self._status.addPermanentWidget(self._fold_btn)
        QShortcut(QKeySequence("Ctrl+M"), self, self._toggle_toolbars)

        self.setStatusBar(self._status)
        self._log("Ready.")

    def _toggle_toolbars(self) -> None:
        """Fold the six toolbar rows away, and back."""
        rows = getattr(self, "_toolbar_rows", [])
        if not rows:
            return
        hide = rows[0].isVisible()
        for tb in rows:
            tb.setVisible(not hide)
        self._fold_btn.setText("\u25bc Toolbars" if hide else "\u25b2 Toolbars")

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
            # in-flight guard in _trigger_fetch doesn't block new fetches.
            self._abandon_fetcher()
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
            self._abandon_fetcher()
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
                    viewer._ingest_ticks(data, candle_mins)

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
            # The K-line SubType above was already being subscribed (moomoo
            # requires it before request_history_kline returns intraday bars),
            # but nothing consumed it, so OpenD's bar stream was decoded by the
            # SDK and dropped. Measured on US.SOXL: a push a median 0.81s apart,
            # 95 of 98 of them the forming bar moving rather than a closed one.
            # It costs no quota and no round trip -- the subscription is
            # already paid for.
            if kl_sub is not None:
                self._ctx.set_handler(_KlinePushHandler(self, code))
            self._live_code = code
            interval = self._refresh_spin.value() * 1000
            self._refresh_timer.start(interval)
            self._redraw_timer.start(int(self._redraw_spin.value() * 1000))
            self._log(f"Live: subscribed {code}, fetching every "
                      f"{self._refresh_spin.value()}s, "
                      f"redrawing every {self._redraw_spin.value():g}s")
        except Exception as exc:
            self._log(f"Live start error: {exc}")

    def _stop_live(self) -> None:
        self._refresh_timer.stop()
        self._redraw_timer.stop()
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
        # Nothing will drain these once the redraw timer is stopped, and on the
        # next Live start they would belong to whatever was showing before.
        with self._tick_lock:
            self._pending_kl = {}
        self._last_kl_push = None

    # ── Toolbar callbacks ─────────────────────────────────────────────────────

    def _on_tf_changed(self, tf: str) -> None:
        _, self._candle_mins = TIMEFRAME_MAP[tf]
        # The DCVD window is counted in bars but scaled in minutes, so the
        # axis and its guides change meaning with the timeframe.
        self._apply_dcvd_scale()
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

    def _on_redraw_changed(self, value: float) -> None:
        if self._redraw_timer.isActive():
            self._redraw_timer.start(int(value * 1000))

    def _ingest_ticks(self, data: pd.DataFrame, candle_mins: int) -> None:
        """Fold a pushed TICKER frame into the live buckets.

        Runs on the subscription's callback thread, so everything it
        touches is under _tick_lock -- including _tick_seq, which is the
        only thing telling the redraw timer there is anything new to
        paint. Bumped once per frame rather than per row: the timer only
        needs to know that something arrived.

        Lives here rather than inside the handler class so it can be
        tested without a subscription; the handler is three lines of
        unpacking around this call.
        """
        with self._tick_lock:
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
                self._live_ticks[bucket][price][key] += vol
                self._last_tick_price = price
            self._tick_seq += 1

    def _ingest_kline(self, rows, code: str) -> None:
        """Stage pushed K-line bars for the GUI thread to fold in.

        Runs on the subscription's callback thread, and deliberately does NOT
        touch self._klines. The GUI thread renders from that frame for ~200ms
        at a stretch and the fetcher thread reads it as its top-up cache;
        neither can hold a lock for that long, so a write from here would race
        both. All this does is drop rows in a staging dict under _tick_lock
        and bump _tick_seq -- _on_redraw_tick drains it on the GUI thread.

        _tick_seq is bumped only when something was actually staged, so a push
        that is filtered out here does not buy a repaint. Its meaning is
        unchanged: the one thing telling the redraw timer there is new data to
        paint, whether it came from the tape or from the bar stream.

        The code check is a cheap early-out, not the authority: _live_kl_key
        is written on the GUI thread WITHOUT this lock, so the lock buys
        nothing for that field. What makes it safe is that rebinding an
        attribute is a single STORE_ATTR -- a reader sees the old tuple or the
        new one, never a torn one -- and that _drain_pushed_klines re-checks
        every row against the current binding on the GUI thread.

        Takes a list of dicts. The handler converts explicitly rather than
        passing the SDK's payload through: CurKlineHandlerBase hands back a
        DataFrame, and iterating one here yields column NAMES.
        """
        if not rows:
            return
        with self._tick_lock:
            key = self._live_kl_key
            if key is None or key[0] != code:
                return
            staged = False
            for r in rows:
                tk = str(r.get("time_key", ""))
                if not tk:
                    continue
                self._pending_kl[tk] = r
                staged = True
            if staged:
                self._tick_seq += 1

    def _drain_pushed_klines(self) -> tuple:
        """Fold staged pushed bars into _klines. Returns (changed, appended).

        GUI thread only: this is where _klines is rebound, and where
        _live_kl_key -- also written on this thread -- is authoritative. A
        push staged just before a symbol or timeframe switch is dropped here
        rather than landing in the wrong chart's frame.
        """
        with self._tick_lock:
            staged = self._pending_kl
            self._pending_kl = {}
        if not staged or self._klines is None or self._live_kl_key is None:
            return False, False
        if self._mode_combo.currentText() != "Live":
            return False, False

        code, tf = self._live_kl_key
        want = str(TIMEFRAME_MAP.get(tf, (None, 0))[0])
        changed = appended = False
        for tk in sorted(staged):                   # oldest first
            row = staged[tk]
            if str(row.get("code", code)) != code:
                continue
            if str(row.get("k_type", want)) != want:
                continue                            # a push for the old tf
            df, grew = _fold_pushed_bar(self._klines, row)
            if df is not self._klines:
                self._klines = df
                changed  = True
                appended = appended or grew
                self._last_kl_push = (self._live_kl_key, row)
        return changed, appended

    def _follow_right_edge(self, prev_n: int, xlo: float, xhi: float) -> None:
        """Keep the newest candle in view after the frame grew.

        Called from both places a bar can appear: a fetch landing in
        _on_data_ready, and now a bar arriving on the push stream. prev_n and
        the X range are the ones read BEFORE the frame grew.

        Only in Live, and only when the view was already parked at the right
        edge -- panning left to read history is a deliberate act and must not
        be undone by the next bar.
        """
        if self._mode_combo.currentText() != "Live" or self._klines is None:
            return
        new_n = len(self._klines)
        if not new_n > prev_n > 0:
            return
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

    def _on_redraw_tick(self) -> None:
        """Repaint from data already in hand, if any of it is new."""
        if self._klines is None:
            return
        prev_n   = len(self._klines)
        xlo, xhi = self._plot_c.vb.viewRange()[0]
        changed, appended = self._drain_pushed_klines()
        if not changed and self._tick_seq == self._drawn_tick_seq:
            return            # nothing new since the last paint
        self._redraw()
        if appended:
            # A bar arrived from the stream rather than from a fetch, so the
            # follow-the-right-edge step that _on_data_ready does has to
            # happen here too -- without it the newest candle walks off the
            # right edge, and the stale xMax limit clips it.
            self._follow_right_edge(prev_n, xlo, xhi)

    def _redraw(self) -> None:
        """Repaint without fetching anything.

        The source must be self._db_ticks, not self._ticks: _render
        overwrites self._ticks with the merged mapping, and its merge
        skips a live bucket already present there, so handing it back
        would freeze the newest bucket -- the only one still moving.
        """
        if self._klines is None:
            return
        self._render(self._klines, self._db_ticks)

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
        show_dcvd = self._ind("dcvd")
        # The light lives on the price chart now, so turning the panel off has
        # to take it down too -- otherwise it hangs there showing whatever the
        # last draw left in it.
        if not show_dcvd:
            self._dcvd_light.setVisible(False)
        if show_dcvd:
            self._plot_dcvd.setTitle(f"{code}  DCVD", color=_FG, size="8pt")
        self._set_subplot_row_visible(self._plot_dcvd, 5, show_dcvd)
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
        self._redraw()
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

    def _on_range_radio_toggled(self, checked: bool) -> None:
        """One click on the range radios emits twice: toggled(False) on the
        button being left and toggled(True) on the one being taken. Both used
        to run a refetch -- harmless while the second was dropped by the
        in-flight guard, a redundant round trip now that it is queued.
        """
        if checked:
            self._on_range_changed()

    def _on_range_changed(self) -> None:
        self._rebuild_session_profile()
        # A refetch, not just a redraw: CVD anchors on this window, and the
        # extra trading days it now spans need their ticks loaded before the
        # curve means anything. Redrawing alone left the older bars at delta 0.
        if self._klines is not None:
            self._trigger_fetch()

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
        if self._fetch_inflight:
            # Queued rather than dropped. Dropping it meant the toolbar had
            # moved and the chart had not, with nothing but a log line to say
            # so -- and in Historical nothing else fetches (_stop_live stops
            # the refresh timer), so the chart stayed on the old bars until
            # the user acted again. The worst of it was Live -> Historical:
            # _on_mode_changed stops the timer and then makes the request
            # that gets dropped.
            self._refetch_pending = True
            self._log("Fetch in progress — queued, will refetch when it ends.")
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

        if historical:
            # prev_close from the snapshot API is today's, not the displayed
            # date's, so the estimate would be silently wrong. No tag is better
            # than a wrong one.
            self._pair_anchor = None

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
            "tick_days":       RANGE_DAYS.get(self._get_range_val(), 1),
            "tick_cache":      dict(self._tick_day_cache),
            "live_state":      None if historical else self._tick_live_state,
            "with_offex":      self._ind("offex"),
            # Only when it is this exact (code, tf); the fetcher additionally
            # refuses to use it if a gap shows up. Historical mode never tops
            # up -- its window is pinned to a date, not rolling.
            "cached_klines":   (self._klines
                                if (not historical
                                    and self._live_kl_key == (new_code, tf))
                                else None),
            "fvg_min_gap_pct": self._fvg_min_pct.value() / 100.0,
            "ob_max_count":    self._ob_max_count.value(),
        }
        self._log(f"Fetching K-lines ({tf}) ...")
        self._fetcher = DataFetcher(ctx, params)
        self._fetcher.ready.connect(self._on_data_ready)
        self._fetcher.error.connect(self._log)
        # finished, not ready: error is not terminal (DataFetcher.run emits it
        # as a warning and carries on in several places) and ready is skipped
        # on a fatal one, so neither fires exactly once per run. finished does.
        self._fetcher.finished.connect(self._on_fetch_finished)
        self._fetcher.start()
        self._fetch_inflight = True

        # Push real-time NBBO to the heatmap so its spread lines reflect the
        # true best bid/ask rather than values derived from ORDER_BOOK depth
        # (which on LITE accounts does not start at the actual NBBO).
        if not historical and self._live_code:
            # The pair's leg rides along in the same snapshot call rather than
            # costing a second round trip; prev_close is a daily constant, so
            # refetching it each load is enough to stay current across a
            # session rollover.
            pair_code = _INVERSE_PAIR.get(self._live_code)
            codes = [self._live_code] + ([pair_code] if pair_code else [])
            self._pair_anchor = None
            try:
                ret, df = ctx.get_market_snapshot(codes)
                if ret == RET_OK and not df.empty:
                    row = df.iloc[0]
                    bid = float(row.get("bid_price", 0) or 0)
                    ask = float(row.get("ask_price", 0) or 0)
                    if bid > 0 and ask > 0:
                        self._last_nbbo = (bid, ask)
                        if self._liq_hm_window is not None:
                            self._liq_hm_window.update_quote(bid, ask)
                    if pair_code is not None and len(df) > 1:
                        by_code = df.set_index("code")
                        refs = _pair_refs(by_code.loc[self._live_code],
                                          by_code.loc[pair_code])
                        if refs is not None:
                            self._pair_anchor = (self._live_code, pair_code,
                                                 refs[0], refs[1])
            except Exception:
                pass

    def _abandon_fetcher(self) -> None:
        """Let go of a fetcher whose result is no longer wanted.

        The flags are cleared here because nothing else will: the signals
        that would have cleared them are the ones just dropped, and a
        reconnect fetches immediately afterwards.

        _retire_fetcher disconnects finished with the rest, but that is
        defence in depth rather than what makes this safe. What makes it
        safe is the identity check in _on_fetch_finished: once self._fetcher
        is None, an abandoned thread's finished() is rejected there whether
        or not it is still connected. (A mutation removing that one
        disconnect survives the suite for exactly that reason, which is the
        honest reading of it.)
        """
        if self._fetcher is not None:
            _retire_fetcher(self._fetcher)
            self._fetcher = None
        self._fetch_inflight  = False
        self._refetch_pending = False

    def _on_refresh_tick(self) -> None:
        """The Live refresh timer. Skips a tick rather than queueing one.

        See the comment where this is connected: an automatic source must not
        be able to queue work, or a fetch that outruns the interval turns into
        an unbroken chain of them.
        """
        if self._fetch_inflight:
            return
        self._trigger_fetch()

    def _on_fetch_finished(self) -> None:
        """A fetch ended; run the one that was asked for while it ran."""
        sender = self.sender()
        if sender is not None and sender is not self._fetcher:
            # An abandoned fetcher (a reconnect replaced it) or a superseded
            # one. Its finished() says nothing about the fetch that is
            # actually running, and acting on it would clear the flag for
            # that one -- letting the retry start a second DataFetcher beside
            # it, on one OpenQuoteContext, both writing _tick_live_state in
            # place. That is the invariant the old guard enforced by refusing
            # the request outright, and the only thing holding it up here.
            return
        self._fetch_inflight = False
        if self._closing:
            return
        if self._refetch_pending:
            # Cleared before the call, not after: the refetch may legitimately
            # queue one of its own.
            self._refetch_pending = False
            self._trigger_fetch()

    def _on_data_ready(self, result: dict) -> None:
        prev_n = len(self._klines) if self._klines is not None else 0
        xlo, xhi = self._plot_c.vb.viewRange()[0] if prev_n > 0 else (0, 0)

        self._klines      = result["klines"]
        # Identity of the bars themselves, for the view-reset decision in
        # _render. Taken from the fetch that produced them, never from the
        # toolbar: the toolbar is already showing what the user asked for.
        # historical is in the tuple because Live and Historical can agree on
        # all three of the others and still be different charts: Historical
        # loads a multi-day lookback, Live one session, so the bar count can
        # change by hundreds while the key would not move at all.
        self._klines_key  = (result.get("code", ""), result.get("tf", ""),
                             result.get("date_str", ""),
                             bool(result.get("historical")))
        self._live_kl_key = (
            None if result.get("historical")
            else (result.get("code", ""), result.get("tf", "")))
        # A fetch snapshots the forming bar a second or two before it lands,
        # while the push stream has kept moving. Re-apply the last pushed bar
        # so the live candle does not visibly step backwards once a cycle;
        # _fold_pushed_bar drops it on volume if the fetch is in fact newer.
        if (self._last_kl_push is not None
                and self._live_kl_key is not None
                and self._last_kl_push[0] == self._live_kl_key):
            self._klines, _ = _fold_pushed_bar(self._klines,
                                               self._last_kl_push[1])
            # Let it go once its bar is no longer the one being formed.
            # Held past that, a downward volume correction from the exchange
            # would be overwritten by this pin on every fetch cycle, and
            # nothing would clear it while the stream is quiet.
            held = _norm_time_key(str(self._last_kl_push[1].get("time_key", "")))
            if (self._klines is None or len(self._klines) == 0
                    or held != str(self._klines["time_key"].iloc[-1])):
                self._last_kl_push = None
        else:
            self._last_kl_push = None
        self._ticks       = result["ticks"]
        self._db_ticks    = result["ticks"]
        for key, day in (result.get("new_tick_days") or {}).items():
            self._tick_day_cache[key] = day
        if len(self._tick_day_cache) > 12:
            # Bounded: a day of 1m buckets is small, but a long session of
            # symbol and timeframe switches would otherwise accumulate them all.
            for key in sorted(self._tick_day_cache)[:-12]:
                del self._tick_day_cache[key]
        self._warmup      = result["warmup"]
        self._smc_signals = result["smc_signals"]
        self._fvg_gaps    = result["fvg_gaps"]
        self._ob_blocks   = result["ob_blocks"]
        self._render(self._klines, self._db_ticks)
        self._follow_right_edge(prev_n, xlo, xhi)
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
        # Probe first. OpenQuoteContext blocks indefinitely rather than
        # raising when OpenD is not up, and this runs on the GUI thread
        # from __init__ -- see _opend_listening.
        if not _opend_listening(_OPEND_HOST, _OPEND_PORT):
            if not self._opend_warned:
                self._opend_warned = True
                self._log(f"OpenD not listening on {_OPEND_HOST}:"
                          f"{_OPEND_PORT} — option chain idle, will start "
                          f"on its own once it is up")
            return
        try:
            self._option_ctx = OpenQuoteContext(host=_OPEND_HOST,
                                                port=_OPEND_PORT)
            if self._opend_warned:
                self._opend_warned = False
                self._log("OpenD is up — option chain starting")
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
        # Whatever had arrived when this paint started is on screen after
        # it; the redraw timer uses this to skip repainting an unchanged
        # picture.
        self._drawn_tick_seq = self._tick_seq

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

        # Which bars moved since the last render. Computed once, against
        # the real buckets rather than the heatmap's view of them, because
        # _bar_flow needs it whether the heatmap is on or not; _dirty_bars
        # records the frame it is given, so a second call would report
        # nothing changed.
        self._frame_dirty = self._dirty_bars(klines, buckets, cm)

        # Candlesticks. The dirty set is what lets the item keep the
        # chunks it already baked; see CandlestickItem._ensure_pictures.
        # Per-bar delta for the zoomed-out body colour. _bar_flow is the
        # cache three overlays already share, so asking for it here costs
        # nothing it was not going to cost anyway.
        bar_delta = None
        if show_hm and buckets:
            _b, _s, _ = self._bar_flow(klines, buckets, cm)
            bar_delta = _b - _s
        self._candle_item.set_data(klines, buckets if show_hm else None, cm,
                                   show_heatmap=show_hm, red_up=red_up,
                                   dirty_bars=self._frame_dirty,
                                   delta=bar_delta)

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

        # DCVD subplot
        self._clear_dcvd_items()
        if self._ind("dcvd"):
            self._draw_dcvd(klines, buckets, cm)

        # Absorption markers (price chart)
        self._clear_abs_items()
        if self._ind("absorb"):
            self._draw_absorption(klines, buckets, cm)

        # Chandelier exit suggestion (bottom-right corner label)
        if self._ind("chandelier"):
            self._update_chandelier_label()
        else:
            self._chandelier_label.setVisible(False)

        # Fixed ATR stop levels (lines on the price panel)
        self._clear_atrstop_items()
        if self._ind("atr_stop"):
            self._draw_atr_stops(klines)

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

        # Session vol profile (or range profile panel if active).
        #
        # Throttled because it is 82ms of what was a 220ms render and a
        # session-wide volume profile does not visibly move in half a
        # second.
        #
        # It was also holding back a leak of four TextItems a rebuild;
        # that is fixed at the source now (see _pin_profile_labels), so
        # what is left here is the cost, not the leak.
        #
        # A frame that is not a simple extension of the last one
        # (_frame_dirty is None: new symbol, timeframe, day) always
        # rebuilds -- that is when the profile really is different.
        now_ms = time.monotonic() * 1000.0
        fresh = (self._frame_dirty is None
                 or now_ms - self._profile_drawn_ms >= _PROFILE_MIN_MS)
        if fresh:
            self._profile_drawn_ms = now_ms
            if self._range_region is not None:
                self._clear_range_inline()
                self._range_last_indices = (-1, -1)
                self._rebuild_range_profile()
            else:
                self._rebuild_session_profile()

        # Tick profile panel: normally hover-driven, so it holds whatever the
        # cursor last landed on. That is right for a closed candle -- its data
        # cannot change -- but the newest candle is still forming, and leaving
        # its panel frozen at the moment the cursor arrived meant it quietly
        # went stale while the candle beside it kept growing. Redraw it here so
        # it advances with the bar it describes. Only the last index qualifies;
        # anything older is finished and repainting it would be pure waste.
        # _show_tick_profile returns immediately in range mode, so the
        # range-selection panel is left alone.
        if self._last_hover_idx is not None and self._last_hover_idx == len(klines) - 1:
            self._show_tick_profile(self._last_hover_idx)

        # Set view range only when Code or TF changes (i.e. a genuinely new
        # chart).  Deferred via singleShot so profile / overlay drawing cannot
        # override the range we set here.
        #
        # Keyed on the bars being painted, not on the toolbar. The toolbar
        # moves first: between the user picking a timeframe and the fetch
        # landing, the redraw timer repaints the OLD bars, and keying on the
        # toolbar banked the new timeframe against them -- so the paint that
        # finally carried the new bars found its key already recorded and
        # left the view where it was. The bar count also changes with the
        # timeframe (1m 390 bars, 5m 78), so the stale range and the
        # setLimits inside _reset_view could leave the chart entirely
        # off-screen, and unreachable by scrolling, until Home was pressed.
        code = self._code_edit.text().strip()
        chart_key = self._klines_key
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

    def _bucket_keys(self, klines: pd.DataFrame, cm: int):
        """bar_start_keys for this frame, memoised for the render that
        needs it five times over."""
        tk = klines["time_key"]
        sig = (len(tk), str(tk.iloc[0]) if len(tk) else "",
               str(tk.iloc[-1]) if len(tk) else "", cm)
        if getattr(self, "_bk_sig", None) == sig:
            return self._bk_cache
        self._bk_sig = sig
        self._bk_cache = bar_start_keys(tk, cm)
        return self._bk_cache

    def _dirty_bars(self, klines: pd.DataFrame, buckets: dict | None,
                    cm: int) -> set | None:
        """Bar indices whose drawing inputs changed since the last render.

        None means "everything", which is what the caller assumed before
        there was a dirty set at all. Returned for the first render and
        for any frame that is not simply the previous one grown at the
        right-hand edge.
        """
        n = len(klines)
        anchor = str(klines["time_key"].iloc[0]) if n else ""
        prev = self._prev_frame
        key = (cm, anchor, buckets is None)

        if prev is None or prev["key"] != key or not n or n < prev["n"]:
            self._remember_frame(klines, buckets, cm, key)
            return None

        m = prev["n"]                       # bars both frames have
        # The bar that was last is no longer last, and anything appended
        # is new, so both are redrawn regardless of their contents.
        dirty = set(range(m - 1, n))

        o = klines["open"].to_numpy(float)
        h = klines["high"].to_numpy(float)
        low = klines["low"].to_numpy(float)
        c = klines["close"].to_numpy(float)
        moved = (  (o[:m] != prev["o"]) | (h[:m] != prev["h"])
                 | (low[:m] != prev["l"]) | (c[:m] != prev["c"]))
        dirty.update(np.nonzero(moved)[0].tolist())

        if buckets is not None:
            keys = self._bucket_keys(klines, cm)
            prev_ids = prev["bid"]
            for i in range(m):
                if id(buckets.get(keys[i])) != prev_ids[i]:
                    dirty.add(i)

        self._remember_frame(klines, buckets, cm, key)
        return dirty

    def _remember_frame(self, klines: pd.DataFrame, buckets: dict | None,
                        cm: int, key: tuple) -> None:
        n = len(klines)
        bid = []
        if buckets is not None and n:
            keys = self._bucket_keys(klines, cm)
            bid = [id(buckets.get(k)) for k in keys]
        self._prev_frame = {
            "key": key, "n": n,
            "o": klines["open"].to_numpy(float).copy(),
            "h": klines["high"].to_numpy(float).copy(),
            "l": klines["low"].to_numpy(float).copy(),
            "c": klines["close"].to_numpy(float).copy(),
            "bid": bid,
            # Keeps every bucket dict alive, so the ids above stay unique.
            "buckets": buckets,
        }

    def _bar_flow(self, klines: pd.DataFrame, buckets: dict, cm: int):
        """Per-bar classified buy and sell volume, plus the trading day.

        Three overlays want this and all three used to compute it from
        scratch every frame: 1440 buckets, every price level in each,
        three times over. Here it is computed once and kept, and only
        the bars in self._frame_dirty are redone -- one bar, usually.

        Returns (buy, sell, day) where day[i] is the trading day of bar
        i's bucket, or None for a bar whose stamp did not parse.
        """
        n = len(klines)
        anchor = str(klines["time_key"].iloc[0]) if n else ""
        cache = self._flow_cache if self._frame_dirty is not None else None
        if (cache and cache["cm"] == cm and cache["anchor"] == anchor
                and n >= cache["n"]):
            buy = np.zeros(n, dtype=float)
            sell = np.zeros(n, dtype=float)
            buy[:cache["n"]] = cache["buy"]
            sell[:cache["n"]] = cache["sell"]
            day = cache["day"] + [None] * (n - cache["n"])
            todo = [i for i in self._frame_dirty if i < n]
        else:
            buy = np.zeros(n, dtype=float)
            sell = np.zeros(n, dtype=float)
            day = [None] * n
            todo = range(n)

        keys = self._bucket_keys(klines, cm)
        for i in todo:
            bk = keys[i]
            if bk is pd.NaT:
                buy[i] = sell[i] = 0.0
                day[i] = None
                continue
            day[i] = _trading_day_of_bucket(bk)
            pd_ = buckets.get(bk)
            if not pd_:
                buy[i] = sell[i] = 0.0
                continue
            b = s_ = 0.0
            for counts in pd_.values():
                b += counts["buy"]
                s_ += counts["sell"]
            buy[i] = b
            sell[i] = s_

        self._flow_cache = {"cm": cm, "anchor": anchor, "n": n,
                            "buy": buy, "sell": sell, "day": day}
        return buy, sell, day

    def _clear_delta_items(self) -> None:
        # Hide, do not remove: the pool exists so that a redraw costs no
        # scene churn, and dropping it here would pay that cost on every
        # indicator toggle.
        #
        # This is also what keeps a short frame from showing the tail of a
        # long one. _draw_delta is only ever called right after this, so
        # it starts from an all-hidden pool and shows exactly the labels
        # it draws; it does not trim a tail of its own.
        # Only the visible span needs hiding; the rest already are, and
        # walking all 1440 to hide them cost 13.5ms a render.
        lo, hi = self._delta_vis
        for item in self._delta_items[lo:hi]:
            item.setVisible(False)
        self._delta_used = 0
        self._delta_vis = (0, 0)

    def _draw_delta(self, klines: pd.DataFrame, buckets: dict, cm: int) -> None:
        lows = klines["low"].to_numpy(float)
        buy, sell, _ = self._bar_flow(klines, buckets, cm)
        # Blank the cull's view of the pool before touching it. Adding
        # and moving items can make the view auto-range, which fires
        # sigRangeChanged mid-loop; a cull running then would read the
        # PREVIOUS frame's _delta_x against this frame's _delta_used and
        # light up labels this frame never drew -- which _delta_vis would
        # then forget, leaving them on screen for good.
        self._delta_x = np.empty(0)
        self._delta_vis = (0, 0)
        xs: list[float] = []
        for i in range(len(klines)):
            # int(): the counts are whole trades, and the under-1000
            # branch below prints the number raw.
            delta = int(buy[i] - sell[i])
            if delta == 0:
                continue
            sign = "+" if delta >= 0 else ""
            if abs(delta) >= 1_000_000:
                txt_str = f"{sign}{delta/1_000_000:.1f}M"
            elif abs(delta) >= 1_000:
                txt_str = f"{sign}{delta/1_000:.0f}K"
            else:
                txt_str = f"{sign}{delta}"

            # Bright white on a coloured fill tag so the text stays readable
            # against the dark background whatever the heatmap is doing.
            n = self._delta_used
            if n < len(self._delta_items):
                txt = self._delta_items[n]
            else:
                txt = pg.TextItem(text="", color="#ffffff", anchor=(0.5, 0.0))
                txt.setFont(QFont("Monospace", 8))
                # Born hidden. A TextItem is visible by default, and
                # visibility is _cull_delta_labels' job -- it only ever
                # changes the span it tracks, so anything it did not put
                # on screen must not already be there.
                txt.setVisible(False)
                self._plot_c.addItem(txt, ignoreBounds=True)
                self._delta_items.append(txt)
            txt.setText(txt_str)
            # No setter for the fill brush, but paint() reads the attribute
            # directly and the constructor does nothing more than this.
            # Two prebuilt brushes: the colour only ever takes two values,
            # and 1440 mkBrush calls a render is not free either.
            txt.fill = _DELTA_UP_BRUSH if delta >= 0 else _DELTA_DN_BRUSH
            txt.setPos(i, lows[i] * 0.9997)
            xs.append(i)
            self._delta_used = n + 1

        # Visibility is the cull's business, not the loop's: at most a
        # screenful is shown, and only when they are far enough apart to
        # read.
        self._delta_x = np.asarray(xs, dtype=float)
        self._cull_delta_labels()

    def _cull_delta_labels(self, *_) -> None:
        """Show delta labels only where they can be read.

        Qt charges for every item in the scene, not for the ones on
        screen, so 1440 labels cost the same at a 200-bar zoom as at a
        full-day one -- and at the full-day one they are packed into a
        few pixels where no number is legible. Both are fixed by showing
        a screenful at most, and nothing at all below _DELTA_MIN_PX.

        Runs on every range change, so it touches only the labels whose
        visibility actually changes.
        """
        if not self._delta_used:
            return
        vb = self._plot_c.vb
        x0, x1 = vb.viewRange()[0]
        span = x1 - x0
        width = vb.width()
        if span <= 0 or width <= 0 or width / span < _DELTA_MIN_PX:
            lo = hi = 0                     # too dense to read
        else:
            # Clamped to what is actually in use: a cull can land while
            # _draw_delta is still filling the pool.
            xs = self._delta_x[:self._delta_used]
            lo = int(np.searchsorted(xs, x0, "left"))
            hi = int(np.searchsorted(xs, x1, "right"))

        plo, phi = self._delta_vis
        if (lo, hi) == (plo, phi):
            return
        for i in range(plo, phi):
            if not (lo <= i < hi):
                self._delta_items[i].setVisible(False)
        for i in range(lo, hi):
            if not (plo <= i < phi):
                self._delta_items[i].setVisible(True)
        self._delta_vis = (lo, hi)

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

    # ── Multi-session volume profile window ──────────────────────────────────

    def _on_mvp_toggle(self, checked: bool) -> None:
        if checked:
            from analysis.multi_vp_window import MultiVPWindow
            self._mvp_window = MultiVPWindow()
            self._mvp_window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
            self._mvp_window.destroyed.connect(self._on_mvp_closed)
            self._mvp_window.set_code(self._code_edit.text().strip())
            self._mvp_window.reload()
            self._mvp_window.show()
        elif self._mvp_window is not None:
            _close_with_hang_watchdog(self._mvp_window, "MultiVPWindow")
            self._mvp_window = None

    def _on_mvp_closed(self) -> None:
        self._mvp_window = None
        self._mvp_btn.setChecked(False)

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

            # Label at right end -- anchor=(1.0, 0.5) (right-anchored, text grows
            # leftward from the last bar) so it stays within the already-visible/
            # pannable chart area instead of extending into the empty margin past
            # the last candle, where it can end up beyond the view's xMax pan
            # limit at typical zoom and only become reachable by zooming in.
            lbl = pg.TextItem(
                text=f"EMA{period}", color=color, anchor=(1.0, 0.5),
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
        # anchor=(1.0, 0.5): right-anchored so the label grows leftward from the
        # last bar, staying within the pannable range (see EMA label comment above).
        lbl = pg.TextItem(text=f"AVWAP {last_v:.2f}", color=_AVWAP_COL, anchor=(1.0, 0.5))
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
                # anchor=(1.0, 0.5): right-anchored, grows leftward from the last
                # bar (see EMA label comment above).
                lbl = pg.TextItem(
                    text=f"{tag}{k}s {band[last_i]:.2f}", color=_AVWAP_COL,
                    fill=pg.mkBrush(_qc(_BG_TIP, 180)), anchor=(1.0, 0.5),
                )
                lbl.setFont(QFont("Monospace", 8))
                lbl.setPos(x[last_i], band[last_i])
                self._plot_c.addItem(lbl, ignoreBounds=True)
                self._avwap_items.append(lbl)

    def _clear_option_wall_items(self) -> None:
        for item in self._option_wall_items:
            self._plot_c.removeItem(item)
        self._option_wall_items.clear()
        self._option_wall_labels.clear()

    def _draw_option_wall_line(self, price: float, color: str, label: str,
                               n: int, alpha: int = 255) -> None:
        line = pg.InfiniteLine(
            pos=price, angle=0, movable=False,
            pen=pg.mkPen(_qc(color, alpha), width=1, style=Qt.PenStyle.DashLine),
        )
        self._plot_c.addItem(line, ignoreBounds=True)
        self._option_wall_items.append(line)

        # anchor=(1.0, 1.0): right-anchored, grows leftward from the last bar --
        # was (0.0, 1.0) (left-anchored, extending rightward past the last
        # candle into the empty margin), which at typical zoom levels put the
        # label beyond the view's xMax pan limit, only reachable by zooming in
        # first (reported: Call/Put Wall labels drifting off the right edge).
        #
        # Text stays fully opaque with a background fill even though the *line*
        # fades out for weaker ranks -- same rule as the AVWAP sigma-band
        # labels: a faint line is visual hierarchy, faint text is unreadable.
        lbl = pg.TextItem(
            text=f"{label} {price:.2f}", color=color,
            fill=pg.mkBrush(_qc(_BG_TIP, 200)), anchor=(1.0, 1.0),
        )
        lbl.setFont(QFont("Monospace", 8))
        lbl.setZValue(55)
        lbl.setPos(n - 1, price)
        self._plot_c.addItem(lbl, ignoreBounds=True)
        self._option_wall_items.append(lbl)
        self._option_wall_labels.append((price, lbl))

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
        self._option_wall_x = n - 1

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

        self._pin_option_wall_labels()

    def _pin_option_wall_labels(self, *_) -> None:
        """Lay the wall labels out in columns so their text never overlaps.

        Walls cluster by nature: adjacent strikes (115 / 114), or a call wall
        and a put wall on the same strike, all map to nearly the same y, and
        right-anchoring every label at the last bar stacked their text into an
        illegible smudge (reported: overlapping Call/Put Wall labels).

        Each label stays glued to its own price -- nudging one vertically would
        misattribute it to the wrong level -- so collisions are resolved
        sideways: a label too close to the one above it moves into the next
        column leftward from the last bar. Columns share a uniform width (the
        labels differ only by a "#2"/"#3" suffix) so they stay aligned.
        """
        if not self._option_wall_labels:
            return
        x_upp, y_upp = self._plot_c.vb.viewPixelSize()
        if not x_upp or not y_upp:
            return

        # self._option_wall_x, not len(self._klines)-1: the walls are drawn
        # against the session-filtered frame, which can be shorter.
        x_right  = self._option_wall_x
        rects    = [lbl.textItem.boundingRect() for _, lbl in self._option_wall_labels]
        col_w_px = max(r.width() for r in rects) + 6
        row_h_px = max(r.height() for r in rects)

        # Descending price, so a column's last entry is always the nearest
        # label above the one being placed -- the only possible collision.
        placed: list[float] = []   # last price placed in each column
        for price, lbl in sorted(self._option_wall_labels, key=lambda t: -t[0]):
            col = next(
                (i for i, last in enumerate(placed)
                 if abs(last - price) / y_upp >= row_h_px),
                len(placed),
            )
            if col == len(placed):
                placed.append(price)
            else:
                placed[col] = price
            lbl.setPos(x_right - col * col_w_px * x_upp, price)

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
        self._cvd_fills = []
        self._offex_arr = None
        for item in self._cvd_items:
            self._plot_cvd.removeItem(item)
        self._cvd_items.clear()
        self._cvd_arr = None

    def _draw_cvd(self, klines: pd.DataFrame, buckets: dict, cm: int) -> None:
        """Draw Cumulative Volume Delta in the CVD subplot.

        Per-bar delta = buy_vol - sell_vol from tick-tagged trades (same
        source as the Delta overlay); bars with no tick coverage are NaN and
        the line simply stops there, because 0 reads as "balanced" when it
        means "not known" -- and on a frame holding more days of klines than
        the Range control loads ticks for, that was 72% of the panel.
        Neutral-tagged volume is
        excluded from the delta itself, same caveat as the Range Profile
        Buy/Sell stats -- moomoo's ticker_direction tags a large share of
        same-price prints NEUTRAL, so this only reflects a subset of volume.
        Resets to 0 at the start of each accumulation block, where a block is
        the number of trading days the Range control selects -- 1D (the
        default, matching TradingView's session reset), 2D, 3D or 1W. Blocks
        are counted back from the most recent trading day, so the newest block
        always holds exactly the last N days and the reset lands where the
        session profile's window starts. A trading day here runs 20:00 ET ->
        next 20:00 ET
        (overnight session start through the following after-hours close,
        matching the "night" session window used by _filter_sessions() and
        config/schedule.json) -- resetting on the calendar-date boundary
        instead would split the continuous 20:00-04:00 overnight session in
        half at midnight, showing a spurious jump/reset in the middle of a
        session that never actually paused.
        """
        n = len(klines)
        buy, sell, day = self._bar_flow(klines, buckets, cm)
        delta = buy - sell

        # Block index per trading day, counted BACK from the newest one so the
        # most recent block is always a full N days. Counting forward from the
        # oldest loaded day would put the boundary at an arbitrary place that
        # moves every time the lookback changes.
        n_days = RANGE_DAYS.get(self._get_range_val(), 1)
        present = sorted({d for d in day if d is not None})
        block = {d: (len(present) - 1 - i) // n_days
                 for i, d in enumerate(present)}

        # Which bars actually have ticks behind them. A bar whose delta
        # nets to zero with real trades in it is a different statement
        # from one with no tick coverage, and "buy == sell" cannot stand
        # in for "nothing known" -- same distinction the off-exchange
        # line below draws with off_seen.
        seen = np.zeros(n, dtype=bool)
        for i, bk in enumerate(self._bucket_keys(klines, cm)):
            if bk is not pd.NaT and buckets.get(bk):
                seen[i] = True

        cvd = np.zeros(n, dtype=float)
        running    = 0.0
        prev_block = None
        for i in range(n):
            if day[i] is not None:
                b = block[day[i]]
                if b != prev_block:
                    running    = 0.0
                    prev_block = b
            running += delta[i]
            cvd[i] = running
        # Drawn as NaN where nothing was known, not held flat at the
        # running total. The chart holds as many days of klines as were
        # fetched while the Range control loads ticks for one to five of
        # them, so the flat part was most of the panel: measured on a
        # 3,980-bar frame with one day of ticks, 2,881 points -- 72% --
        # sat at exactly 0, which reads as "delta is balanced" and is
        # really "no data". It also put a vertical stroke where the line
        # climbed out of that false zero into the first real value.
        cvd[~seen] = np.nan
        self._cvd_arr = cvd

        # Off-exchange flow, accumulated on the same blocks, as its own line.
        # Kept separate rather than added in: folding the two together made
        # delta track price worse (+0.276 to +0.225 against the minute's
        # return), and it hides the only thing this flow is good for -- saying
        # when the unhurried side disagrees with the aggressive one.
        #
        # It is not a direction signal. Scored against the feed's own tags the
        # midpoint rule is 75.6% right, so the side is real; but its one-minute
        # autocorrelation is -0.105 against +0.209 for exchange flow (it does
        # not persist, so nobody is working it), and it *follows* price with a
        # negative sign -- off-exchange buying leads the next minute's return
        # at -0.177. Uninformed flow, which is what makes a divergence with the
        # exchange side worth seeing.
        off_delta = np.zeros(n, dtype=float)
        # Which bars actually carry classified off-exchange prints. A bar can
        # net to zero with real data in it, so "off_delta == 0" is a different
        # question and cannot stand in for this.
        off_seen = np.zeros(n, dtype=bool)
        have_off = False
        for i, bk in enumerate(self._bucket_keys(klines, cm)):
            if bk is pd.NaT:
                continue
            pd_ = buckets.get(bk)
            if not pd_:
                continue
            ob = sum(pd_[p].get("off_buy", 0) for p in pd_)
            os_ = sum(pd_[p].get("off_sell", 0) for p in pd_)
            if ob or os_:
                have_off = True
                off_seen[i] = True
            off_delta[i] = ob - os_

        self._offex_arr = None
        if have_off and self._ind("offex"):
            off = np.zeros(n, dtype=float)
            running, prev_block = 0.0, None
            for i in range(n):
                if day[i] is not None:
                    b = block[day[i]]
                    if b != prev_block:
                        running, prev_block = 0.0, b
                running += off_delta[i]
                off[i] = running
            # The sides come from the nightly classifier (scheduler cron
            # "30 20 * * 1-5"), so the session being watched has none until it
            # has run. Every unclassified bar contributes 0, which carried the
            # line flat across the whole day -- and a flat line reads as
            # "off-exchange flow is steady at -22K" when it means "not known
            # since last night". Stop the curve at the last bar that had data
            # and leave the rest NaN, so the hover readout says so instead of
            # repeating a stale number.
            last = int(np.flatnonzero(off_seen)[-1]) if off_seen.any() else -1
            off[last + 1:] = np.nan
            self._offex_arr = off
            line = pg.PlotCurveItem(
                x=np.arange(last + 1), y=off[:last + 1],
                pen=pg.mkPen("#ab47bc", width=1.3, style=Qt.PenStyle.DashLine))
            self._plot_cvd.addItem(line)
            self._cvd_items.append(line)

        x = np.arange(n)

        # Zero-referenced green/red fill (buy-/sell-pressure bias readable at
        # a glance) plus a crisp outline on top for the precise curve shape --
        # a plain single-color 1px line made the trend hard to read. Each
        # fill curve pins to 0 on the opposite side of zero from its own
        # sign, matching the pattern of a signed histogram/area chart rather
        # than drawing two overlapping flat-at-zero pens (which would leave
        # a visible seam line cutting across the other side's fill).
        # The uncovered bars go to 0 in the FILLS rather than NaN: the
        # fill level is 0 too, so a run of zeros paints nothing, while a
        # NaN inside a filled curve leaves pyqtgraph closing the polygon
        # across the gap.
        fin = np.isfinite(cvd)
        cvd_pos = np.where(fin & (cvd >= 0), cvd, 0.0)
        cvd_neg = np.where(fin & (cvd < 0), cvd, 0.0)
        pos_fill = pg.PlotCurveItem(
            x=x, y=cvd_pos, pen=None,
            fillLevel=0.0, brush=_qc(_GREEN, 55),
        )
        neg_fill = pg.PlotCurveItem(
            x=x, y=cvd_neg, pen=None,
            fillLevel=0.0, brush=_qc(_RED, 55),
        )
        outline = pg.PlotCurveItem(
            x=x, y=cvd, connect="finite",
            pen=pg.mkPen("#42a5f5", width=1.3),
        )
        for item in (pos_fill, neg_fill, outline):
            self._plot_cvd.addItem(item)
            self._cvd_items.append(item)
        self._cvd_fills = [pos_fill, neg_fill]
        self._sync_cvd_fill()

    def _sync_cvd_fill(self) -> None:
        """Show the zero-referenced fill only while zero is on screen.

        CVD's fill runs from the curve to y=0, which reads as buy- or
        sell-pressure bias at a glance -- but only while 0 is in view. A day's
        CVD drifts hundreds of thousands of shares from zero, so any zoom into
        the recent shape leaves 0 far below the window and the fill washes the
        whole panel one colour, hiding the line it was meant to decorate.

        The line itself is never hidden, so zooming in gives the shape rather
        than a solid block.
        """
        fills = getattr(self, "_cvd_fills", None)
        if not fills:
            return
        ylo, yhi = self._plot_cvd.vb.viewRange()[1]
        # bool(), not the bare comparison: viewRange carries numpy scalars once
        # anything has set the range from numpy data, and the chain then yields
        # numpy.bool_, which PyQt6's setVisible rejects. The TypeError is raised
        # inside a signal handler, where Qt prints it to stderr and carries on --
        # so the fill silently freezes at whatever it last was.
        on = bool(ylo <= 0.0 <= yhi)
        for f in fills:
            f.setVisible(on)

    def _dcvd_lookback(self) -> int:
        spin = getattr(self, "_dcvd_spin", None)
        return int(spin.value()) if spin is not None else DCVD_LOOKBACK

    def _dcvd_minutes(self) -> int:
        """Minutes the window actually covers.

        The spinbox counts BARS, and DCVD_SCALE is keyed by minutes -- it was
        measured on 1-minute bars over N-bar windows. On a 5m chart a
        lookback of 1 is a five-minute reading, so passing the bar count
        would scale a 5-minute series against the 1-minute distribution and
        leave it sitting in the middle of the panel.
        """
        return max(1, self._dcvd_lookback() * max(1, self._candle_mins))

    def _apply_dcvd_scale(self) -> None:
        """Range and guide lines for the current window."""
        mins = self._dcvd_minutes()
        p75, p90, p99 = dcvd_scale(mins)
        self._plot_dcvd.setYRange(-p99 * 1.05, p99 * 1.05, padding=0)
        for item in getattr(self, "_dcvd_guides", []):
            self._plot_dcvd.removeItem(item)
        self._dcvd_guides = []
        # p75 is dropped: at the 1-bar window it sits at 0.517, close enough
        # to the 0.40 decision line that the two read as one thick band and
        # neither can be told from the other. p90 stays as the "unusual"
        # marker, well clear of it.
        guides = [(0.0, _GREY, 1, Qt.PenStyle.SolidLine),
                  (p90, _GREY, 1, Qt.PenStyle.DotLine),
                  (-p90, _GREY, 1, Qt.PenStyle.DotLine)]
        # The threshold was measured on ONE MINUTE of flow and does not
        # carry to other windows: the same rule on five minutes was worth
        # nothing out of sample. Drawing it over a 5-minute series would
        # invite trading a level never validated for it, so it appears only
        # when the window really is one minute.
        if mins == 1 and DCVD_DECISION < p99:
            guides += [
                (DCVD_DECISION, _DCVD_DECISION_COL, 1.4, Qt.PenStyle.DashLine),
                (-DCVD_DECISION, _DCVD_DECISION_COL, 1.4,
                 Qt.PenStyle.DashLine)]
        for lvl, col, wid, style in guides:
            ln = pg.InfiniteLine(pos=lvl, angle=0, movable=False,
                                 pen=pg.mkPen(col, width=wid, style=style))
            self._plot_dcvd.addItem(ln)
            self._dcvd_guides.append(ln)
        # Minutes, not bars: "DCVD(1)" on a 5m chart read as one minute.
        self._plot_dcvd.setLabel(
            "left", f"DCVD({mins}m)", **{"color": _FG})

    def _on_dcvd_lookback_changed(self, *_) -> None:
        self._apply_dcvd_scale()
        if self._klines is not None:
            self._redraw()

    def _draw_dcvd(self, klines: pd.DataFrame, buckets: dict, cm: int) -> None:
        """Draw DCVD: the CVD slope normalised to [-1, +1].

        CVD's slope over N bars is the net delta accumulated across them. The
        largest slope those same bars could have produced is every classified
        trade landing on one side, so dividing one by the other is a bound of
        1 by construction rather than a fitted scale:

            DCVD = sum(buy - sell) / sum(buy + sell)   over the last N bars

        +1 is every classified trade a buy, -1 every one a sell, 0 balanced.
        Because it is a ratio of volumes it reads the same on a quiet day as a
        busy one, which an unnormalised slope does not.

        NEUTRAL volume is left out of both halves, matching _draw_cvd's delta.
        It is the majority of SOXL's tape -- 57.5% measured over three days --
        and putting it in the denominator would damp DCVD so far that the
        median bar reads 0.08 and the range above +/-0.5 goes unused. The
        trade-off is that DCVD describes the flow that could be classified,
        not all of it.

        A bar whose signed volume falls under DCVD_MIN_SIGNED is drawn as a
        gap, not as zero and not as the rail a single small print would put
        it on. Roughly 6% of RTH minutes have no ticks at all, and they
        arrive in long stretches rather than scattered, so a gap that reads
        as a gap is the honest picture.
        """
        n = len(klines)
        buy, sell, _ = self._bar_flow(klines, buckets, cm)

        csum_d = np.concatenate([[0.0], np.cumsum(buy - sell)])
        csum_t = np.concatenate([[0.0], np.cumsum(buy + sell)])

        def series(look: int) -> np.ndarray:
            look = max(1, min(look, n))
            out = np.full(n, np.nan)
            for i in range(n):
                j = max(0, i + 1 - look)
                tot = csum_t[i + 1] - csum_t[j]
                if tot >= DCVD_MIN_SIGNED:
                    out[i] = (csum_d[i + 1] - csum_d[j]) / tot
            return out

        # The flow gate: is this bar's signed volume large for this stock
        # right now? Strictly trailing -- the rolling median and percentile
        # both end at the previous bar, so a bar never raises the hurdle it
        # is then asked to clear.
        tot = pd.Series(buy + sell).replace(0.0, np.nan)
        scale = tot.rolling(DCVD_FLOW_WIN, min_periods=20).median().shift(1)
        xf = pd.Series(buy - sell) / scale
        gate = (xf.abs().rolling(DCVD_FLOW_WIN, min_periods=20)
                .quantile(DCVD_FLOW_PCT).shift(1))
        # `gate` has no value until the rolling median and the rolling
        # quantile both have their 20 bars and the shift has landed -- about
        # the first 40 minutes of a session, and longer on a thin one, since
        # bars under DCVD_MIN_SIGNED do not count towards either window.
        # Folding that into False told the panel and the badge "the flow is
        # not large" when the truth is "the gate does not exist yet", and it
        # did so in the same grey, at the open, which is when the tape is
        # busiest. Keep the two apart.
        self._dcvd_gate_ok = gate.notna().to_numpy()
        self._dcvd_gate = (xf.abs() >= gate).fillna(False).to_numpy()

        look = max(1, min(self._dcvd_lookback(), n))
        dcvd = series(look)
        self._dcvd_arr = dcvd
        x = np.arange(n)

        # Each bar is one minute's own reading, not a point on a path between
        # readings. A line drawn through them invents the segment in between,
        # and the shorter the window the more of the picture is invented --
        # which is what made this panel look like noise at a 1-bar window.
        drawn = np.isfinite(dcvd)
        gated = self._dcvd_gate
        gate_ok = self._dcvd_gate_ok
        # Three states, not two. A bar the gate never judged is drawn at the
        # dim weight -- its height, the ratio, is real either way -- but
        # carries a neutral outline, so "not judged" cannot be read as "judged
        # and found small". The outline rather than a third alpha because a
        # fainter bar reads as even less flow, which is the opposite of what
        # is being said.
        for sign, col in ((1, _GREEN), (-1, _RED)):
            side = drawn & ((dcvd >= 0) if sign > 0 else (dcvd < 0))
            for sel, alpha, pen in (
                    (side & gate_ok & gated, _DCVD_LIT_ALPHA, None),
                    (side & gate_ok & ~gated, _DCVD_DIM_ALPHA, None),
                    (side & ~gate_ok, _DCVD_DIM_ALPHA,
                     pg.mkPen(_qc(DCVD_LIGHT_NONE, 190), width=1))):
                if not sel.any():
                    continue
                bars = pg.BarGraphItem(x=x[sel], height=dcvd[sel],
                                       width=0.78, brush=_qc(col, alpha),
                                       pen=pen)
                self._plot_dcvd.addItem(bars)
                self._dcvd_items.append(bars)

        # The light, on the PRICE chart. The newest bar is still forming in
        # Live mode, so this is the running reading for the minute in
        # progress, not a settled one -- it can change until the bar closes,
        # and that is the honest thing for it to do rather than lag a minute
        # behind the tape.
        last_i = next((i for i in range(n - 1, -1, -1)
                       if np.isfinite(dcvd[i])), None)
        if last_i is None:
            self._dcvd_light.setVisible(False)
        else:
            last_v = dcvd[last_i]
            # Read at the SAME bar as the number beside it. Taking the
            # last finite reading and the last gate flag separately would
            # pair a value with another bar's volume on any day with a
            # gap in the tape.
            gate_ok = bool(self._dcvd_gate_ok[last_i])
            heavy = gate_ok and bool(self._dcvd_gate[last_i])
            # Hue from the sign, brightness from the flow state. Taking
            # the hue from the same reading the number shows is what
            # keeps the badge from contradicting itself.
            #
            # The candle pair, not the gold/purple one the heatmap and
            # the delta labels use for buyer- and seller-dominant. This
            # badge is read as a traffic light, and gold in a traffic
            # light is the amber that means "not yet" -- a heavy +0.88
            # came out looking like a warning. The bug that started this
            # was a negative reading painted teal, i.e. a hue that
            # contradicted the sign; banning direction hues outright was
            # a step too far, since direction is exactly what the sign is.
            hue = _GREEN if last_v >= 0 else _RED
            if not gate_ok:
                # Not "thin volume" -- the gate has not been established on
                # this session yet, so the badge has nothing to say about
                # size. The reading itself stands, so the hue stays.
                col, note = _qc(hue, DCVD_LIGHT_LEVEL_A), "  flow n/a"
            elif heavy:
                col, note = _qc(hue, DCVD_LIGHT_HEAVY_A), "  heavy flow"
            elif abs(last_v) >= DCVD_DECISION:
                col, note = _qc(hue, DCVD_LIGHT_LEVEL_A), "  thin volume"
            else:
                col, note = _qc(DCVD_LIGHT_NONE), ""
            # Named, unlike the old badge: in the DCVD panel the context said
            # what the number was, and on the price chart nothing does.
            self._dcvd_light.setText(f"\u25cf DCVD {last_v:+.2f}{note}",
                                     color=col)
            self._dcvd_light.setVisible(True)
            self._pin_dcvd_light()

        # The slower reading sits behind as context. 'finite' breaks the line
        # at bars the floor rejected rather than bridging them.
        if DCVD_CONTEXT > look:
            ctx = series(DCVD_CONTEXT)
            if np.isfinite(ctx).any():
                line = pg.PlotCurveItem(
                    x=x, y=ctx, connect="finite",
                    pen=pg.mkPen(_DCVD_CONTEXT_COL, width=1.6))
                line.setZValue(5)
                self._plot_dcvd.addItem(line)
                self._dcvd_items.append(line)

    @staticmethod
    def _abs_series(
        h: np.ndarray, l: np.ndarray, c: np.ndarray,
        buy: np.ndarray, sell: np.ndarray,
        scale: np.ndarray | None = None,
        gate: np.ndarray | None = None,
        atr0: float | None = None,
        prev: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """(x, y, A, gate, last ATR) for one session's bars.

        Every rolling quantity ends at the previous bar, so a bar is
        never part of the fit, the scale or the gate it is judged
        against -- the no-lookahead rule the study registered.

        `scale` and `gate` may be supplied per bar, which is how the
        caller hands in a time-of-day profile built from PREVIOUS
        sessions. Left out, both fall back to the trailing 60-bar form:
        the median of this bar's own recent total volume, and the 80th
        percentile of its own recent |x|.

        `atr0` seeds Wilder's average from the previous session's last
        value, and `prev` is that session's trailing (x, y) for lambda
        and the residual spread. With both, a session can be marked from
        its first bars; without them three 20-bar windows run in series
        and nothing is marked before roughly bar 60.

        `dp` is never carried across the boundary, seeded or not: the
        overnight gap is a price move no flow explains, so a session's
        first bar is always unmarked.
        """
        m = len(c)
        nan = np.full(m, np.nan)
        if m < 2:
            return nan, nan, nan, nan, np.nan

        tr = np.empty(m)
        tr[0] = h[0] - l[0]          # the overnight gap is not a true range
        for i in range(1, m):
            tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]),
                        abs(l[i] - c[i - 1]))
        a = np.full(m, np.nan)
        n = ABS_ATR_N
        if atr0 is not None and np.isfinite(atr0):
            run = float(atr0)
            for i in range(m):
                run = (run * (n - 1) + tr[i]) / n
                a[i] = run
        elif m > n:
            # tr[0] is excluded here as it was before the seed existed.
            a[n] = np.nanmean(tr[1:n + 1])
            for i in range(n + 1, m):
                a[i] = (a[i - 1] * (n - 1) + tr[i]) / n

        tot = buy + sell
        ok = tot >= DCVD_MIN_SIGNED
        ofi = np.where(ok, buy - sell, np.nan)
        dp = np.r_[np.nan, np.diff(c)]
        if scale is None:
            scale = (pd.Series(np.where(ok, tot, np.nan))
                     .rolling(DCVD_FLOW_WIN, min_periods=20)
                     .median().shift(1).to_numpy())
        with np.errstate(invalid="ignore", divide="ignore"):
            y = dp / a
            x = ofi / np.asarray(scale, dtype=float)
        if gate is None:
            gate = (pd.Series(x).abs()
                    .rolling(DCVD_FLOW_WIN, min_periods=20)
                    .quantile(DCVD_FLOW_PCT).shift(1).to_numpy())

        xt, yt = (np.empty(0), np.empty(0)) if prev is None else prev
        xc = pd.Series(np.r_[xt, x])
        yc = pd.Series(np.r_[yt, y])
        k0 = len(xt)
        xy = (xc * yc).rolling(DCVD_FLOW_WIN, min_periods=20).sum().shift(1)
        xx = (xc * xc).rolling(DCVD_FLOW_WIN, min_periods=20).sum().shift(1)
        lam = xy / xx.replace(0.0, np.nan)
        eps = yc - lam * xc
        sd = eps.rolling(DCVD_FLOW_WIN, min_periods=20).std().shift(1)
        A = (eps / sd.replace(0.0, np.nan)).to_numpy()[k0:]
        return x, y, A, np.asarray(gate, dtype=float), a[-1]

    @staticmethod
    def _abs_day(h: np.ndarray, l: np.ndarray, c: np.ndarray,
                 buy: np.ndarray, sell: np.ndarray,
                 **kw) -> tuple[np.ndarray, np.ndarray]:
        """(bull, bear) masks for one session's bars.

        Bullish where heavy SELLING did not get its move, bearish where
        heavy buying did not. See _abs_series for the quantities and for
        what the keyword arguments do.
        """
        m = len(c)
        bull = np.zeros(m, dtype=bool)
        bear = np.zeros(m, dtype=bool)
        if m < 25:
            return bull, bear
        x, _y, A, gate, _a = TradeViewerQt._abs_series(
            h, l, c, buy, sell, **kw)
        fire = (np.isfinite(x) & np.isfinite(gate) & np.isfinite(A)
                & (np.abs(x) >= gate))
        return (fire & (x < 0) & (A >= ABS_THRESHOLD),
                fire & (x > 0) & (A <= -ABS_THRESHOLD))

    def _absorption_flags(self, klines: pd.DataFrame, buckets: dict,
                          cm: int):
        """(bull, bear) masks over the whole frame, a trading day at a time.

        Split by day because every window here is trailing: one reaching
        back over the 20:00 boundary would fit this session's flow
        against the previous session's, and the overnight gap would enter
        as a price move no flow explains.
        """
        n = len(klines)
        bull = np.zeros(n, dtype=bool)
        bear = np.zeros(n, dtype=bool)
        if n < 25:
            return bull, bear
        buy, sell, day = self._bar_flow(klines, buckets, cm)
        h = klines["high"].to_numpy(float)
        l = klines["low"].to_numpy(float)
        c = klines["close"].to_numpy(float)
        start = 0
        for i in range(1, n + 1):
            if i < n and day[i] == day[start]:
                continue
            sl = slice(start, i)
            b, e = self._abs_day(h[sl], l[sl], c[sl], buy[sl], sell[sl])
            bull[sl] = b
            bear[sl] = e
            start = i
        return bull, bear

    def _draw_absorption(self, klines: pd.DataFrame, buckets: dict,
                         cm: int) -> None:
        """Hollow triangles where the study's condition fired.

        Hollow, and clear of the bar rather than on it: this marks the
        candle, it does not replace reading it. Gold below for selling
        that was absorbed, purple above for buying -- the same two
        colours the delta labels use for buyer- and seller-dominant.
        """
        bull, bear = self._absorption_flags(klines, buckets, cm)
        if not (bull.any() or bear.any()):
            return
        low = klines["low"].to_numpy(float)
        high = klines["high"].to_numpy(float)
        idx = np.arange(len(klines))
        for mask, ys, sym, col in ((bull, low * 0.9985, "t1", _UP),
                                   (bear, high * 1.0015, "t", _DOWN)):
            if not mask.any():
                continue
            item = pg.ScatterPlotItem(
                x=idx[mask], y=ys[mask], symbol=sym, size=10,
                brush=None, pen=pg.mkPen(_qc(col, 235), width=1.6))
            item.setZValue(60)
            self._plot_c.addItem(item)
            self._abs_items.append(item)

    def _clear_abs_items(self) -> None:
        for item in self._abs_items:
            self._plot_c.removeItem(item)
        self._abs_items.clear()

    def _clear_dcvd_items(self) -> None:
        for item in self._dcvd_items:
            self._plot_dcvd.removeItem(item)
        self._dcvd_items.clear()
        self._dcvd_arr = None
        self._dcvd_gate = None
        self._dcvd_gate_ok = None

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
          1. Filled ribbon between mid1 and mid2 — gold alpha-35 when fast is
             above slow (bullish spread), purple alpha-35 when fast is below.
          2. mid1 line (blue, fast channel midpoint).
          3. mid2 line (gold, slow channel midpoint).

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

        # Populate toolbar. Date first: the combos fetch as a side effect of
        # being set (currentTextChanged -> _on_tf_changed / _on_mode_changed,
        # each ending in _trigger_fetch), so a date written afterwards meant
        # the Historical fetch those triggered went out for the PREVIOUS
        # date. The final _trigger_fetch below is the one that carries the
        # right combination, and it used to be the one the in-flight guard
        # dropped -- so the trade landed on the wrong bars.
        self._code_edit.setText(row.get("symbol", ""))
        self._date_edit.setText(date_str)
        self._date_edit.setEnabled(True)
        self._tf_combo.setCurrentText(tf)
        self._mode_combo.setCurrentText("Historical")

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

        if not sigs:
            self._log(f"Scanner Signals: no open entry signals for {code} in db/signals.db")
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

    def _clear_profile_pins(self) -> None:
        """Forget the labels the previous rebuild pinned.

        There is nothing to disconnect: the connection is made once, in
        the constructor, and survives every rebuild. See
        _pin_profile_labels.
        """
        self._profile_pins.clear()

    def _pin_profile_labels(self, *_) -> None:
        """Hold the profile panel's labels against the view edges.

        Connected once for the window's lifetime and re-reading a list
        that each rebuild replaces. Each rebuild used to define its own
        closures, connect them, and disconnect the previous set; the
        disconnected ones stayed reachable regardless and took their
        labels with them -- four TextItems a render. Five explanations
        were tested and none held (see the commit), so the pattern goes
        rather than the symptom.

        Three kinds: "r" rides the right edge (level labels, clear of
        bars growing out from x=0), "l" the left (the previous
        session's open and close), "tl" the top-left corner (the range
        panel's stats header, which is a header and not a level).
        """
        pins = self._profile_pins
        if not pins:
            return
        (xlo, xhi), (_ylo, yhi) = self._profile_widget.getViewBox().viewRange()
        for kind, item, y in pins:
            if kind == "l":
                item.setPos(xlo, y)
            elif kind == "tl":
                item.setPos(xlo, yhi)
            else:
                item.setPos(xhi, y)

    def _reset_poc_state_if_redefined(self, range_val: str, n_bins: int) -> None:
        """Drop the held POC and its trail when the profile is no longer the
        same profile. A POC computed over a different symbol, date, range,
        bin count or session filter is not a previous position of this one,
        so carrying it over would draw a trail between unrelated levels."""
        key = (
            self._code_edit.text().strip(), self._date_edit.text().strip(),
            self._mode_combo.currentText(), range_val, n_bins,
            tuple(sorted(self._active_sessions())), self._candle_mins,
        )
        if key != self._poc_ctx_key:
            self._poc_ctx_key = key
            self._poc_held = None
            self._poc_trail = []

    @staticmethod
    def _cluster_mass(centers: np.ndarray, volumes: np.ndarray,
                      price: float) -> float:
        """Total volume within _POC_CLUSTER_PCT of `price` -- the size of the
        volume cluster a candidate POC sits in, rather than of its own bin."""
        return float(volumes[np.abs(centers - price)
                             <= price * _POC_CLUSTER_PCT / 100.0].sum())

    def _resolve_poc(self, centers: np.ndarray,
                     volumes: np.ndarray) -> tuple[float, int]:
        """Return (poc_price, poc_bin_index) with switch hysteresis applied.

        The held level is stored as a PRICE and, while held, is reported back
        unchanged. Re-snapping it to the nearest bin centre each rebuild would
        make it jitter by a few cents on its own as lo/hi (and so every bin
        boundary) shift with each new bar -- movement that says nothing about
        where volume actually is.

        A switch needs the challenger's whole cluster to beat the held one's
        by the margin, not just its peak bin: see _POC_CLUSTER_PCT.
        """
        raw_i = int(np.argmax(volumes))
        # Same flat-distribution guard _compute_poc_vah_val applies.
        if np.all(volumes == volumes[raw_i]):
            raw_i = len(volumes) // 2
        raw_p = float(centers[raw_i])

        margin = self._poc_hyst_spin.value() / 100.0
        held = self._poc_held
        if margin <= 0 or held is None or not (centers[0] <= held <= centers[-1]):
            return raw_p, raw_i          # off, nothing held, or held out of range

        held_i = int(np.argmin(np.abs(centers - held)))
        bin_w = float(centers[1] - centers[0]) if centers.size > 1 else 0.0
        if abs(raw_p - held) <= bin_w:
            return held, held_i          # same level, just re-binned
        if (self._cluster_mass(centers, volumes, raw_p)
                > self._cluster_mass(centers, volumes, held) * (1.0 + margin)):
            return raw_p, raw_i
        return held, held_i

    def _update_poc_trail(self, poc: float, bin_w: float) -> list[float]:
        """Record `poc` as the current level and return the newest-first slice
        to draw.

        The trail is a history of where the POC has BEEN, not the profile's
        runner-up peaks -- POC2 is the level POC1 moved off, so the pair reads
        as a direction of travel. Entries closer together than half a bin are
        treated as the same level: without that a POC stepping between two
        neighbouring bins would fill the trail with what is visually one line,
        pushing the older level that actually matters off the end.
        """
        self._poc_held = poc
        tol = bin_w * 0.5
        if not self._poc_trail:
            self._poc_trail = [poc]
        elif abs(self._poc_trail[0] - poc) <= tol:
            # Still the same level, just re-binned a cent or two: refresh the
            # head in place rather than pushing a new entry. The head must
            # stay equal to the POC being drawn, or POC1's label reports a
            # stale price; only a move worth more than a bin earns a POC2.
            self._poc_trail[0] = poc
        else:
            # Drop any older entry that has become the same level as the new
            # one (price came back), so the trail keeps showing DISTINCT
            # levels instead of logging a round trip twice.
            self._poc_trail = [poc] + [
                p for p in self._poc_trail if abs(p - poc) > tol
            ]
            del self._poc_trail[8:]
        return self._poc_trail[: self._poc_trail_spin.value()]

    def _rebuild_session_profile(self) -> None:
        if self._klines is None:
            return
        self._clear_profile_pins()
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
            klines, self._ticks, self._candle_mins, n_bins=n_bins,
            cache=self._profile_cache)
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

        self._reset_poc_state_if_redefined(range_val, n_bins)
        poc, poc_idx = self._resolve_poc(centers, volumes)
        # POC comes from _resolve_poc (it may be a held level, which is kept at
        # its own price rather than re-snapped); the value area still expands
        # from that same bin so the three lines agree about where the peak is.
        _, vah, val = _compute_poc_vah_val(centers, volumes, poc_idx=poc_idx)
        # A held POC is an exact past price, while VAH/VAL are bin centres, so
        # a value area that collapsed onto the POC's own bin can come back a
        # fraction of a bin short of it and draw POC outside its own VA. Widen
        # to restore VAL <= POC <= VAH; the correction is sub-bin, i.e. below
        # the resolution either number is quoted at anyway.
        vah, val = max(vah, poc), min(val, poc)
        bin_w = float(centers[1] - centers[0]) if centers.size > 1 else 0.0
        trail = self._update_poc_trail(poc, bin_w)

        # POC1 = current, POC2.. = the levels it moved off, newest first.
        # Only the line fades and turns dotted with age; the text stays fully
        # opaque (same rule as the AVWAP band and option wall labels -- a
        # faint line is visual hierarchy, faint text is just unreadable).
        poc_labels: list[tuple[float, pg.TextItem]] = []
        for rank, price in enumerate(trail):
            poc_line = pg.InfiniteLine(
                pos=price, angle=0, movable=False,
                pen=pg.mkPen(_qc(_RED, max(70, 255 - rank * 70)), width=1,
                             style=Qt.PenStyle.SolidLine if rank == 0
                             else Qt.PenStyle.DotLine),
            )
            poc_label = pg.TextItem(
                text=f"POC{rank + 1} {price:.2f}", color=_RED,
                fill=pg.mkBrush(_qc(_BG_TIP, 180)),
                anchor=(1.0, 1.0),
            )
            poc_label.setFont(QFont("Monospace", 7))
            pw.addItem(poc_line,  ignoreBounds=True)
            pw.addItem(poc_label, ignoreBounds=True)
            poc_labels.append((price, poc_label))

        # Right edge, not left. The histogram grows rightward from x=0, so a
        # left-pinned label sits on top of every bar; the X range carries 15%
        # headroom past the longest one, which leaves the right edge clear.
        self._profile_pins += [("r", lbl_item, p) for p, lbl_item
                               in poc_labels]

        for price, lbl in [(vah, "VAH"), (val, "VAL")]:
            va_line = pg.InfiniteLine(
                pos=price, angle=0, movable=False,
                pen=pg.mkPen(_GOLD, width=1, style=Qt.PenStyle.DashLine),
            )
            va_label = pg.TextItem(
                text=f"{lbl} {price:.2f}", color=_GOLD,
                fill=pg.mkBrush(_qc(_BG_TIP, 180)),
                anchor=(1.0, 0.0),
            )
            va_label.setFont(QFont("Monospace", 7))
            pw.addItem(va_line,  ignoreBounds=True)
            pw.addItem(va_label, ignoreBounds=True)

            self._profile_pins.append(("r", va_label, price))

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
            anchor=(1.0, 1.0),
        )
        price_label.setFont(QFont("Monospace", 7))
        pw.addItem(price_line,  ignoreBounds=True)
        pw.addItem(price_label, ignoreBounds=True)

        self._profile_pins.append(("r", price_label, last_price))

        # Previous session's open and close, each drawn only when it
        # falls inside what the panel is showing. See _prev_session_levels
        # for the day boundary and for why they come from the unfiltered
        # frame.
        for price, tag in self._prev_session_levels():
            if not (lo <= price <= hi):
                continue
            ln = pg.InfiniteLine(
                pos=price, angle=0, movable=False,
                pen=pg.mkPen(_qc(_GREY), width=1,
                             style=Qt.PenStyle.DashDotLine))
            lb = pg.TextItem(
                text=f"{tag} {price:.2f}", color=_GREY,
                fill=pg.mkBrush(_qc(_BG_TIP, 180)), anchor=(0.0, 0.5))
            lb.setFont(QFont("Monospace", 7))
            pw.addItem(ln, ignoreBounds=True)
            pw.addItem(lb, ignoreBounds=True)

            # Pinned LEFT, unlike POC/VAH/VAL on the right: these are
            # reference prices rather than readings off this profile, and
            # stacking five labels in one corner makes all five unreadable.
            self._profile_pins.append(("l", lb, price))

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
        # Place them once now that the range is final. The permanent
        # connection keeps them there through later pans and zooms, but
        # a range that ends up unchanged emits nothing, so the first
        # placement cannot be left to the signal.
        self._pin_profile_labels()

    def _prev_session_levels(self) -> list[tuple[float, str]]:
        """[(price, tag)] for the previous trading day's open and close.

        "Previous" is relative to the last bar loaded, NOT to the
        profile's Range window: in 2D the levels then fall inside the
        distribution being drawn, which is intended -- where yesterday
        opened does not change with how many days the histogram is
        accumulating over.

        From self._klines, the UNFILTERED frame. The session-filtered one
        is what the histogram is built from, and with "Night" unchecked it
        holds no overnight bars -- the open would quietly become the
        09:30 one, which is a different price and a different claim.

        Trading day is _trading_day's 20:00 boundary, so the overnight
        session counts toward the day it leads into, the same as CVD's
        reset and the profile range.
        """
        k = self._klines
        if k is None or len(k) < 2:
            return []
        cm = max(1, int(self._candle_mins))
        days = [_trading_day(tk, cm) for tk in k["time_key"]]
        cur = days[-1]
        if cur is None:
            return []
        prev = next((d for d in reversed(days) if d is not None and d != cur),
                    None)
        if prev is None:
            return []
        rows = [i for i, d in enumerate(days) if d == prev]
        if not rows:
            return []
        return [(float(k["open"].iloc[rows[0]]), "PDO"),
                (float(k["close"].iloc[rows[-1]]), "PDC")]

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

    def _fit_profile_y(self, pw, prices, bin_h: float = 0.0,
                       y_lo: float | None = None,
                       y_hi: float | None = None) -> None:
        """Y range from the panel's OWN bars, not the price chart's viewport.

        Copying the chart's Y made the two line up, which is worth having
        only while the chart is zoomed near this candle. It is not, in the
        home view or any time an option wall stretches the axis: a minute
        of ticks spans cents against an axis spanning dollars, and every
        bar collapses into one sliver.

        The candle's own high/low are taken into the range when the caller
        has them, so a wick that traded nothing still shows as empty space
        rather than being cropped out of the picture.
        """
        if not prices:
            return
        lo, hi = min(prices), max(prices)
        if bin_h:
            hi += bin_h
        if y_lo is not None:
            lo = min(lo, y_lo)
        if y_hi is not None:
            hi = max(hi, y_hi)
        span = hi - lo
        if span <= 0:
            # Every print at one price -- a doji with one price level. Give
            # it a sliver of its own value to sit in; a zero-height view
            # renders nothing at all.
            span = max(abs(hi) * 0.001, 0.01)
            lo, hi = lo - span / 2, hi + span / 2
        pad = span * 0.10
        pw.setYRange(lo - pad, hi + pad, padding=0)

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
        # Every draw starts from X auto-range ON. The VP branch below pins an
        # explicit X range, and pyqtgraph's setRange disables auto-range as a
        # side effect (disableAutoRange defaults True) -- that stuck, so once
        # VP had been used the order-flow view kept VP's volume-scale X range
        # and drew its log-scale bars (about +/-10 wide) as an invisible sliver
        # at the left edge. Resetting here rather than in the order-flow branch
        # keeps the two modes independent of each other's ordering.
        pw.getPlotItem().vb.enableAutoRange(axis=pg.ViewBox.XAxis, enable=True)

        bin_h = (max(prices) - min(prices)) / max(len(prices), 1) * 0.9 if prices else 0.01
        bin_h = max(bin_h, 0.001)

        if self._ind("tick_vp"):
            self._draw_tick_vp(prices, buys, sells, neutrals, bin_h, header,
                               y_lo=y_lo, y_hi=y_hi)
            return

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

        # Heaviest-level highlight: the N price levels that traded the most,
        # taken PER SIDE, get a gold outline (instead of no pen) and the
        # price printed above the row. Per side rather than pooled, because
        # a pooled top-N on a one-sided candle spends every mark on the side
        # that was already obvious and says nothing about the other.
        #
        # This deliberately makes no claim about imbalance. It replaced a
        # ratio threshold that could not support one: only 8.4% of raw price
        # levels carry both buy and sell volume, so `dominant / max(minor, 1)`
        # was almost always dividing by the 1-share floor rather than by a
        # real opposing side. At the shipped default of 3.0 that passed about
        # 89 levels a candle, every candle hit the three-level cap, and 72.7%
        # of the time the three drawn were just the three largest bars. The
        # control now says what the code was already doing.
        #
        # Ties and zeros: argsort would hand a mark to a level with no volume
        # on that side at all on a quiet candle, which would outline an empty
        # bar, so each side only ever marks levels it actually traded on.
        top_n      = self._tick_top_n.value()
        buy_mark   = top_n_levels(buys_arr,  top_n)
        sell_mark  = top_n_levels(sells_arr, top_n)
        # A level can win on both sides; it gets one label, coloured by
        # whichever side put it there with more volume.
        marked     = buy_mark | sell_mark
        # Net mode draws one bar a row and so can only honour one side's
        # mark; it replaces both of these so the label follows the outline.
        label_buy  = None
        no_pen     = pg.mkPen(None)
        hi_pen     = pg.mkPen(_GOLD, width=1.5)

        if self._ind("tick_net"):
            # Single net (buy - sell) bar per price level: one-sided, colored
            # by sign, instead of two diverging buy/sell bars.
            net       = buys_arr - sells_arr
            net_log   = np.log1p(np.abs(net))
            net_x0    = np.where(net >= 0, zeros, -net_log)
            net_x1    = np.where(net >= 0, net_log, zeros)
            net_brush = [_qc(buy_col, 180) if d >= 0 else _qc(sell_col, 180) for d in net]
            # One bar per level here, and its sign says which side it stands
            # for, so it takes that side's mark -- not either side's, which
            # would outline a net-buy bar because the level was a top SELL.
            net_pens  = [hi_pen if (buy_mark[i] if n >= 0 else sell_mark[i])
                         else no_pen for i, n in enumerate(net)]
            # The label has to follow the same rule, not the union: a level
            # that tops the side its net sign does NOT name has no bar to
            # outline here, so naming it anyway prints a price in the other
            # side's colour with nothing beside it.
            marked    = np.where(net >= 0, buy_mark, sell_mark)
            label_buy = net >= 0
            net_bar = pg.BarGraphItem(
                x0=net_x0, x1=net_x1,
                y=prices, height=bin_h,
                brushes=net_brush, pens=net_pens,
            )
            pw.addItem(net_bar)
        else:
            buy_pens  = [hi_pen if m else no_pen for m in buy_mark]
            sell_pens = [hi_pen if m else no_pen for m in sell_mark]
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
            if not marked[i]:
                continue
            if label_buy is not None:
                dom_col = buy_col if label_buy[i] else sell_col
            elif buy_mark[i] and sell_mark[i]:
                dom_col = buy_col if buys_arr[i] >= sells_arr[i] else sell_col
            else:
                dom_col = buy_col if buy_mark[i] else sell_col
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

        # Force the X auto-range to recompute now. Re-enabling it (top of this
        # method) only sets a flag; without this the view keeps whatever range
        # it last held, which after a visit to VP mode is a volume scale tens
        # of thousands wide -- these log-scale bars are about +/-10, so they
        # collapse into an invisible sliver at the left edge and the panel
        # looks blank (reported). Recomputing beats pinning an explicit range
        # here: it keeps the padding behaviour this view has always had.
        pw.getPlotItem().vb.updateAutoRange()

        self._fit_profile_y(pw, prices, bin_h, y_lo, y_hi)

    def _draw_tick_vp(self, prices: list, buys: list, sells: list,
                      neutrals: list, bin_h: float, header: str,
                      y_lo: float | None = None,
                      y_hi: float | None = None) -> None:
        """Render the same tick data as a volume profile: one bar per price
        level for total traded volume, plus POC / VAH / VAL.

        The order-flow view this replaces answers "who was aggressing at each
        price"; this one answers "where did the volume actually trade", which
        is the market-profile question the right-hand session panel asks of a
        whole session -- here scoped to a single candle (or the selected
        range), so the two panels read the same way at two zoom levels.

        Volume is buy + sell + neutral after the S/M/L filters, matching what
        the session profile counts from the same tick source. Bars are linear,
        not log-scaled like the order-flow view: a profile is read by
        comparing bar lengths, which log scaling destroys.

        Tick-only by construction -- the caller reaches this with a price->
        counts dict from ticks.db, and a candle with no tick coverage never
        gets here (see _show_tick_profile's early return). Unlike the session
        panel there is no OHLCV fallback, so a level shown here is a price
        that genuinely traded rather than one a bar's range merely spanned.
        """
        pw = self._tick_profile_widget
        centers = np.asarray(prices, dtype=float)
        totals = (np.asarray(buys, dtype=float)
                  + np.asarray(sells, dtype=float)
                  + np.asarray(neutrals, dtype=float))
        if not totals.any():
            return

        poc, vah, val = _compute_poc_vah_val(centers, totals)

        # Inside the value area vs outside: the VA is the point of the view,
        # so it carries the saturated colour and the tails fade back.
        in_va = (centers >= val) & (centers <= vah)
        brushes = [_qc(_GOLD, 150) if inside else _qc(_GOLD, 60) for inside in in_va]
        pw.addItem(pg.BarGraphItem(
            x0=np.zeros(len(centers)), x1=totals,
            y=centers, height=bin_h, brushes=brushes, pen=pg.mkPen(None),
        ))

        max_vol = float(totals.max())
        pw.setXRange(0, max_vol * 1.15, padding=0)

        # Right-aligned at the right edge of the X range pinned just above.
        # Bars grow rightward from x=0, so the left edge is the one place every
        # bar passes through; the 15% headroom past the longest bar leaves the
        # right edge clear. Anchoring to a bar's tip instead would scatter the
        # labels across a 200-300px column and overlap the bars themselves.
        label_x = max_vol * 1.15
        for price, text, colour, anchor in (
            (val, "VAL", _GOLD, (1.0, 0.0)),
            (poc, "POC", _RED, (1.0, 1.0)),
            (vah, "VAH", _GOLD, (1.0, 0.0)),
        ):
            style = (Qt.PenStyle.SolidLine if text == "POC" else Qt.PenStyle.DashLine)
            line = pg.InfiniteLine(pos=price, angle=0, movable=False,
                                   pen=pg.mkPen(colour, width=1, style=style))
            line.setZValue(6)
            pw.addItem(line, ignoreBounds=True)
            lbl = pg.TextItem(text=f"{text} {price:.2f}", color=colour,
                              fill=pg.mkBrush(_qc(_BG_TIP, 180)), anchor=anchor)
            lbl.setFont(QFont("Monospace", 7))
            lbl.setPos(label_x, price)
            lbl.setZValue(20)
            pw.addItem(lbl, ignoreBounds=True)

        total_vol = float(totals.sum())
        vol_str = (f"{total_vol/1_000_000:.2f}M" if total_vol >= 1_000_000
                   else f"{total_vol/1000:.0f}K" if total_vol >= 1000
                   else f"{total_vol:.0f}")
        pw.getPlotItem().setLabel("bottom", "", **{})
        pw.getPlotItem().setLabel("top", (
            f"<span style='color:{_FG}'>{header}</span>"
            f"&nbsp;&nbsp;<span style='color:{_GOLD}'>VP {vol_str}</span>"
            f"&nbsp;<span style='color:{_RED}'>POC {poc:.2f}</span>"
        ), **{"size": "7pt"})

        self._fit_profile_y(pw, prices, bin_h, y_lo, y_hi)

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
        keys = self._bucket_keys(self._klines, cm)
        for idx in range(i0, min(i1 + 1, len(keys))):
            bk = keys[idx]
            if bk is pd.NaT:
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
        self._clear_profile_pins()
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
                anchor=(1.0, 1.0),   # right-aligned; _pin_panel_labels puts it on xhi
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

        # Level labels ride the right edge, clear of the bars growing out from
        # x=0. stats_item stays top-left: it is a panel header, not a level, and
        # keeping it there means it never collides with the four of them.
        # The text is fixed for this rebuild, so it is set here rather
        # than on every range change.
        poc_lbl.setText(f"POC {poc:.2f}")
        vah_lbl.setText(f"VAH {vah:.2f}")
        val_lbl.setText(f"VAL {val:.2f}")
        price_lbl.setText(f"Last {last_price:.2f}")
        self._profile_pins += [("r", poc_lbl, poc), ("r", vah_lbl, vah),
                               ("r", val_lbl, val),
                               ("r", price_lbl, last_price),
                               ("tl", stats_item, 0.0)]

        self._pin_profile_labels()

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
        """Sync tick profile Y axis with the main candle chart on every zoom/pan.

        The cumulative subplots go with it. Their Y was fitted only at
        load and on each new bar, so panning or zooming left the window
        where it was while the curve walked out of it -- a CVD that
        climbs 1.5M over a session leaves a 30K-tall window within
        minutes, and the panel then looks empty or shows one vertical
        stroke where the curve crosses it.
        """
        ylo, yhi = ranges[1]
        self._tick_profile_widget.setYRange(ylo, yhi, padding=0)
        xlo, xhi = ranges[0]
        self._autorange_cumulative_subplots(xlo, xhi)

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

    def _clear_atrstop_items(self) -> None:
        for item in self._atrstop_items:
            self._plot_c.removeItem(item)
        self._atrstop_items.clear()

    def _on_atrstop_settings(self) -> None:
        dlg = AtrStopParamsDialog(
            self._atrstop_period, self._atrstop_mults,
            self._atrstop_direction, self._atrstop_anchor, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            (self._atrstop_period, self._atrstop_mults,
             self._atrstop_direction, self._atrstop_anchor) = dlg.values()
            if self._ind("atr_stop") and self._klines is not None:
                self._clear_atrstop_items()
                self._draw_atr_stops(self._klines)

    def _draw_atr_stops(self, klines: pd.DataFrame) -> None:
        """One dashed line per ATR multiple, labelled with price and distance.

        Nothing is drawn when the calculation declines -- too few bars for
        the warmup, or a flat series with no range. A stop line at an
        invented price is worse than no line, because it looks exactly like
        a real one.
        """
        from strategy.atr_stop import fixed_stops, parse_multipliers

        mults = parse_multipliers(self._atrstop_mults)
        if klines is None or klines.empty or not mults:
            return
        r = fixed_stops(
            klines["high"].to_numpy(float), klines["low"].to_numpy(float),
            klines["close"].to_numpy(float), self._atrstop_period, mults,
            self._atrstop_direction, self._atrstop_anchor)
        if r is None:
            return

        bull = r["direction"] == "bull"
        red_up = self._ind("red_up")
        # Same convention as the chandelier label: the stop is the losing
        # side, so it takes the colour that side has in the current scheme.
        col = (_RED if red_up else _GREEN) if bull else (_GREEN if red_up else _RED)
        # The widest multiple is the real stop if several are on screen;
        # drawing them all at one weight makes the chart say nothing about
        # which is which, so opacity carries the ordering.
        widest = max(lv["mult"] for lv in r["levels"])
        for lv in r["levels"]:
            alpha = 110 + int(105 * (lv["mult"] / widest))
            line = pg.InfiniteLine(
                pos=lv["stop"], angle=0, movable=False,
                pen=pg.mkPen(_qc(col, alpha), width=1,
                             style=Qt.PenStyle.DashLine),
                label=(f"SL {lv['mult']:g}x {lv['stop']:.2f}  "
                       f"(-{lv['dist_pct']:.2f}%)" if bull else
                       f"SL {lv['mult']:g}x {lv['stop']:.2f}  "
                       f"(+{lv['dist_pct']:.2f}%)"),
                labelOpts={"color": col, "position": 0.12})
            self._plot_c.addItem(line)
            self._atrstop_items.append(line)

        # The anchor, when it is not simply the last close -- otherwise the
        # distances are measured from somewhere the chart does not show.
        if self._atrstop_anchor is not None:
            a = pg.InfiniteLine(
                pos=r["anchor"], angle=0, movable=False,
                pen=pg.mkPen(_qc(_FG, 120), width=1,
                             style=Qt.PenStyle.DotLine),
                label=f"Entry {r['anchor']:.2f}",
                labelOpts={"color": _FG, "position": 0.12})
            self._plot_c.addItem(a)
            self._atrstop_items.append(a)

    def _pin_dcvd_light(self, *_) -> None:
        """Bottom-right of the candle view, one row up when the chandelier
        label is there -- they share the corner."""
        if not self._dcvd_light.isVisible():
            return
        xlo, xhi = self._plot_c.vb.viewRange()[0]
        ylo, yhi = self._plot_c.vb.viewRange()[1]
        x_pad = (xhi - xlo) * 0.01
        y_pad = (yhi - ylo) * 0.015
        row = ((yhi - ylo) * 0.055
               if self._chandelier_label.isVisible() else 0.0)
        self._dcvd_light.setPos(xhi - x_pad, ylo + y_pad + row)

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

    def _pair_implied(self, price: float) -> tuple[str, str, float] | None:
        """Where the inverse-leveraged pair sits when this symbol is at *price*.

        Returns (base_code, pair_code, price), or None when there is nothing to
        show -- no anchor fetched yet, the symbol in the box is not the one the
        anchor was taken for, or historical mode, where the anchor would be the
        wrong day's.

        None is also returned once the implied price goes non-positive: both
        funds reset daily from the previous close, so the relation only holds
        while each leg is still solvent against that anchor, and a SOXL move
        past +200% would put SOXS below zero. Well outside anything that can
        happen in a session, but it costs one comparison to not print it.

        The current symbol is read from the code box, which is where the viewer
        actually keeps it -- an earlier version of this read a self._code that
        does not exist on the window, so every mouse move raised AttributeError
        and took the ordinary price tag down with it.
        """
        anchor = self._pair_anchor
        if anchor is None:
            return None
        base_code, pair_code, base_ref, pair_ref = anchor
        if base_code != self._code_edit.text().strip() or base_ref <= 0:
            return None
        implied = pair_ref * (2.0 - price / base_ref)
        return (base_code, pair_code, implied) if implied > 0 else None

    def _on_profile_hover(self, pos, widget: "pg.PlotWidget",
                          line: pg.InfiniteLine) -> None:
        """Drive a profile panel's price line from that panel's own cursor.

        The line otherwise mirrors the main chart's crosshair, which is useless
        the moment the cursor is over the panel -- it freezes wherever the
        chart last left it while you read the histogram. Each panel is its own
        PlotWidget with its own scene, so the chart's handler is not firing at
        all then; this takes over and the two decouple on their own.

        Y only. The panel's X is volume, which has nothing to say about price,
        and moving the main chart's vertical line from here would scrub the
        chart while the user is reading a profile.
        """
        pi = widget.getPlotItem()
        if not pi.sceneBoundingRect().contains(pos):
            return
        line.setVisible(True)          # before setPos -- see _on_mouse_move
        line.setPos(pi.vb.mapSceneToView(pos).y())

    @staticmethod
    def _fit_tag_y(vb, y: float, above_px: float, below_px: float) -> float:
        """Nearest price to y at which a pixel-sized tag still fits in view.

        A ViewBox clips its children to its own rect, and a TextItem is
        positioned in data units but drawn at a fixed pixel size, so a tag
        within one line height of the top or bottom edge gets cut in half --
        at every zoom, because the box does not shrink with the range. This
        is what the view-span percentages elsewhere in this handler were
        reaching for; a percentage of the span is not the height of the box,
        and the two drift apart as the window is resized.

        above_px / below_px are how far the block reaches over and under the
        price it marks, which depends on the item's anchor: anchor y=1 is all
        above, y=0 all below, y=0.5 half each.

        Returns y untouched everywhere but the last few pixels at an edge, so
        the tag sits on the crosshair line exactly as before and only steps
        aside when it would otherwise be clipped. Either way the number it
        prints is the cursor's own price -- only the box moves.

        Call it AFTER setText: the height measured is the text's.
        """
        ylo, yhi = vb.viewRange()[1]
        py       = vb.viewPixelSize()[1]      # data units per pixel, positive
        above, below = above_px * py, below_px * py
        if above + below >= yhi - ylo:
            return y          # taller than the whole view: nothing fits
        return min(max(y, ylo + below), yhi - above)

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
        in_dcvd   = (self._plot_dcvd.isVisible()
                     and self._plot_dcvd.sceneBoundingRect().contains(pos))
        in_any    = (in_candle or in_vol or in_kd or in_cvd or in_adi
                     or in_dcvd)

        if not in_any:
            for line in (self._vline, self._hline,
                         self._vline_v, self._vline_kd, self._vline_cvd,
                         self._vline_adi, self._vline_dcvd,
                         self._dcvd_label,
                         self._offex_label,
                         self._price_label, self._pair_price_label,
                         self._ohlcv_label,
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
        if self._plot_dcvd.isVisible():
            self._vline_dcvd.setPos(x); self._vline_dcvd.setVisible(True)

        # Horizontal line only in candle plot
        self._hline.setPos(y);  self._hline.setVisible(in_candle)

        # Profile panel sync lines (session profile + tick profile)
        # Visible first, then positioned: pyqtgraph's InfLineLabel.valueChanged
        # returns early while the label is hidden, so positioning a hidden line
        # and then showing it displays the previous value -- 0.00 on the first
        # hover of a session.
        self._profile_hline.setVisible(True)
        self._profile_hline.setPos(y)
        self._tick_profile_hline.setVisible(True)
        self._tick_profile_hline.setPos(y)

        xlo, xhi = self._plot_c.vb.viewRange()[0]
        ylo, yhi = self._plot_c.vb.viewRange()[1]
        y_span   = yhi - ylo
        label_x  = xlo + (xhi - xlo) * 0.01  # ~1% from left edge

        # Price tag: left edge, vertically centered on cursor price, nudged
        # inward by _fit_tag_y at the very top and bottom of the view where it
        # would otherwise be clipped in half.
        pair = self._pair_implied(y)
        if pair is None:
            self._price_label.setAnchor((0.0, 0.5))   # centred on the line
            self._price_label.setText(f"{y:.2f}")
            half = self._price_label.boundingRect().height() / 2.0
            self._price_label.setPos(
                label_x, self._fit_tag_y(self._plot_c.vb, y, half, half))
            self._pair_price_label.setVisible(False)
        else:
            base_code, pair_code, pair_px = pair
            # Tickers are shown only in pair mode: with two numbers on one line
            # they stop being decoration and start doing the disambiguating.
            self._price_label.setAnchor((0.0, 1.0))   # lifted above the line
            self._price_label.setText(
                f"{_short_code(base_code)} {y:.2f}")
            self._pair_price_label.setText(
                f"{_short_code(pair_code)} {pair_px:.2f}")
            # One fit for the two of them. Fitting each on its own would, at
            # the bottom edge, hold the gold tag still and walk the pair tag
            # up through it -- they straddle the line as one block, so they
            # move as one.
            tag_y = self._fit_tag_y(
                self._plot_c.vb, y,
                self._price_label.boundingRect().height(),
                self._pair_price_label.boundingRect().height())
            self._price_label.setPos(label_x, tag_y)
            self._pair_price_label.setPos(label_x, tag_y)
            self._pair_price_label.setVisible(True)
        self._price_label.setVisible(True)

        if self._klines is not None and not self._klines.empty:
            idx = int(round(x))
            idx = max(0, min(idx, len(self._klines) - 1))
            row = self._klines.iloc[idx]
            vol = int(row.get("volume", 0) or 0)

            # OHLCV tooltip: anchor bottom-left → body floats ABOVE position,
            # about a seventh of the view over the cursor. Set before placed:
            # the fit needs this bar's text. It used to reserve 2% of the view
            # span against the top edge for a box that is four lines (~60 px)
            # tall -- 9 px on a 451 px chart, so most of it was cut off anyway.
            self._ohlcv_label.setText(
                f"{str(row['time_key'])[:16]}\n"
                f"O {row['open']:.2f}  H {row['high']:.2f}\n"
                f"L {row['low']:.2f}  C {row['close']:.2f}\n"
                f"Vol {vol:,}"
            )
            self._ohlcv_label.setPos(label_x, self._fit_tag_y(
                self._plot_c.vb, y + y_span * 0.14,
                self._ohlcv_label.boundingRect().height(), 0.0))
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
            if (self._plot_cvd.isVisible() and self._cvd_arr is not None
                    and np.isfinite(self._cvd_arr[
                        max(0, min(idx, len(self._cvd_arr) - 1))])):
                # A bar with no tick coverage has no cumulative delta to
                # report. Printing the running total there would answer a
                # question the data cannot -- it is the flat-zero line in
                # label form.
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

            # Off-exchange line: show it next to CVD so a divergence can be
            # read off as numbers, not just eyeballed from two curves.
            off = getattr(self, "_offex_arr", None)
            if self._plot_cvd.isVisible() and off is not None:
                o_idx = max(0, min(idx, len(off) - 1))
                o_val = float(off[o_idx])
                xlo_o, xhi_o = self._plot_cvd.vb.viewRange()[0]
                lx_o = xlo_o + (xhi_o - xlo_o) * 0.12
                if not np.isfinite(o_val):
                    # Past the last classified bar. Say so rather than showing
                    # the last known figure as though it still held.
                    ylo_o, yhi_o = self._plot_cvd.vb.viewRange()[1]
                    self._offex_label.setPos(lx_o, (ylo_o + yhi_o) / 2)
                    self._offex_label.setText("off n/a")
                else:
                    sign = "+" if o_val >= 0 else ""
                    if abs(o_val) >= 1_000_000:
                        o_str = f"{sign}{o_val/1_000_000:.2f}M"
                    elif abs(o_val) >= 1_000:
                        o_str = f"{sign}{o_val/1_000:.0f}K"
                    else:
                        o_str = f"{sign}{o_val:.0f}"
                    self._offex_label.setPos(lx_o, o_val)
                    self._offex_label.setText(f"off {o_str}")
                self._offex_label.setVisible(True)
            else:
                self._offex_label.setVisible(False)

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

            # DCVD: already a fraction, so print it as one rather than
            # abbreviating -- two decimals is the whole useful resolution.
            if self._plot_dcvd.isVisible() and self._dcvd_arr is not None:
                d_idx = max(0, min(idx, len(self._dcvd_arr) - 1))
                d_val = float(self._dcvd_arr[d_idx])
                xlo_d, xhi_d = self._plot_dcvd.vb.viewRange()[0]
                self._dcvd_label.setPos(xlo_d + (xhi_d - xlo_d) * 0.01, d_val)
                self._dcvd_label.setText(f"{d_val:+.2f}")
                self._dcvd_label.setVisible(True)
            else:
                self._dcvd_label.setVisible(False)

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
            # NaN is "no tick coverage here", which is most of the series
            # whenever the chart holds more days than the Range control
            # loads ticks for. min() over that is NaN and setYRange(NaN)
            # leaves the panel blank.
            if len(vis) == 0 or not np.isfinite(vis).any():
                continue
            lo, hi = float(np.nanmin(vis)), float(np.nanmax(vis))
            pad = (hi - lo) * 0.1 or max(abs(hi), 1.0) * 0.1
            plot.setYRange(lo - pad, hi + pad, padding=0)

    def _set_xaxis_ticks(self, klines: pd.DataFrame) -> None:
        """Map integer bar indices to time_key strings on x-axis.

        Only the main candle plot shows text labels; subplots suppress labels
        to prevent crowding — they are X-linked so positions already align.

        Every subplot needs a line below. One that is missing does not fall
        back to blank, it falls back to pyqtgraph's auto ticks, which label
        the raw bar index -- "4320" where a time belongs. DCVD read that way
        from v0.44.0 until v0.45.2.
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
        if self._ind("dcvd"):
            self._plot_dcvd.getAxis("bottom").setTicks([pos_only])

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
        # Multi-VP was built without a parent, like the other two, and was
        # the one left out of this list. So closing the viewer left it on
        # screen with no owner, the event loop alive because of it, and
        # its 30s timer still running -- now against a moomoo context the
        # teardown below has already closed, and since v0.74.3 that timer
        # asks for fresh data, so an orphan opens a new connection every
        # half minute for as long as the window is up.
        if self._mvp_window is not None:
            _close_with_hang_watchdog(self._mvp_window, "MultiVPWindow")
            self._mvp_window = None
        if self._dom_window is not None:
            self._dom_window.close()
            self._dom_window = None

        # Before anything else: a fetch still in flight will deliver its
        # finished() after this window is closed, and a queued refetch fired
        # from there would run against a torn-down widget (self._ctx is
        # nulled below, so it would reach self._log -> a status bar whose C++
        # object may already be gone).
        self._closing = True

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
    ap.add_argument("--tf",      default="1m",
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
