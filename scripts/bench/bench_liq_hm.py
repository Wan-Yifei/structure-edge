"""Replay the LiqHm window's whole live pipeline against recorded books.

This is the ACCEPTANCE RULER for performance work on the heatmap. Every
before/after claim in CHANGELOG from v0.104.5 onward is meant to be
reproducible with it, which is why it lives here instead of in a scratch
directory that gets cleaned.

    uv run scripts/bench/bench_liq_hm.py warm US.QQQ "2026-11-24 14:50:00" 300

Read two numbers off it: `CPU ... / 模拟 300s` and `ImageItem.paint`.

HOW TO USE IT AS A RULER, because a single run says nothing. Run-to-run
spread was measured at +/-3.6% on one machine and +/-1.8% on another, so:
establish a baseline of three runs BEFORE the change, three after, and
require the two ranges not to overlap. A median that moved by less than the
baseline spread is not a result. `ImageItem.paint` must stay put: if it
moves, the change altered how often the thing repaints, which is a
different claim needing its own argument.

WHAT IT IS FAITHFUL TO
  * The real LiqHmWindow, the real grid, the real paint, the real 100 ms
    render coalescing, the configured max_cols/n_price/gamma read from
    config/viewer_qt.toml through viewer_config.
  * The whole live pipeline -- the column timer runs, so _on_tick rolls
    columns, redraws markers and loads absorb ticks, as set_live() arranges.
  * Real arrival gaps. Pushes replay at their recorded inter-arrival times
    (median 14 ms, p95 304 ms), not on a uniform timer.

WHAT IT IS NOT
  * Not the app. QT_QPA_PLATFORM=offscreen, so this is software rasterising
    into a buffer, not your GPU compositing a 1500x950 window. The SHAPE of
    the drawing cost can differ; py-spy --native on the running viewer is
    the only thing that settles that.
  * No OpenD. `_start_push` is a no-op and the three QThread workers are
    replaced by synchronous stand-ins, so the SDK's own threads are out of
    the measurement.
  * One window. The real session runs the main chart in the same process on
    the same GUI thread, and their costs add.

THE VIRTUAL CLOCK is what makes the numbers mean anything. Qt's QTimers are
replaced by fakes driven off the recorded timestamps, so simulated time and
wall time decouple WITHOUT changing the push:render:column ratios. An
earlier harness ran at 8x on a real clock and silently starved the renderer,
because the 100 ms coalescing fires on wall time while pushes arrive eight
times faster. CPU is reported per SIMULATED second for the same reason.

Scenarios (argv[1]):
  warm    -- full pipeline, grid pre-filled from the DB first. Use this.
  cold    -- full pipeline, empty grid at t0
  bypass  -- no column timer at all; reproduces a flawed earlier harness,
             kept only to show what it was missing

Switches, all off by default:
  BENCH_PROC=1      processEvents() after every push and timer fire. WITHOUT
                    IT NOTHING PAINTS: the run reports ~21 s instead of
                    ~104 s and ImageItem.paint drops from 2,881 to 7. Qt
                    drawing is about 79% of this window's CPU, so a reading
                    taken without this switch is measuring the other 21%.
  MOOMOO_CFG=<toml> load a different viewer config
  MOOMOO_OVL=freeze draw the overlays once and never again
  MOOMOO_NOADD=1    build the overlay items but never put them in the scene
  MOOMOO_IGB=1      force ignoreBounds=True on every addItem
The last three exist to apportion the overlay cost; see TODO.md.
"""
import json
import os
import pathlib
import sqlite3
import sys
import time
from datetime import datetime, timedelta

SCEN = sys.argv[1] if len(sys.argv) > 1 else "cold"
CODE = sys.argv[2] if len(sys.argv) > 2 else "US.QQQ"
START = sys.argv[3] if len(sys.argv) > 3 else "2026-11-25 14:00:00"
SECS = int(sys.argv[4]) if len(sys.argv) > 4 else 120

PROC = os.environ.get("BENCH_PROC", "")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import numpy as np                                            # noqa: E402
import psutil                                                 # noqa: E402
from PyQt6.QtWidgets import QApplication                      # noqa: E402

from analysis import viewer_config as vcfg                    # noqa: E402

_CFG = os.environ.get("MOOMOO_CFG")
if _CFG:
    vcfg.reload(_CFG)
    print(f"[CFG] {_CFG}", flush=True)
from analysis import liq_hm_window as L                       # noqa: E402
from analysis import orderflow_detect as OD                   # noqa: E402

DB = f"file:{_ROOT / 'db' / 'order_book.db'}?mode=ro"
_ET = L._ET


# ── load recorded pushes, keeping their real timestamps ───────────────────────
def load(code, start, secs):
    con = sqlite3.connect(DB, uri=True)
    end = con.execute("SELECT datetime(?, ?)", (start, f"+{secs} seconds")
                      ).fetchone()[0]
    rows = con.execute(
        "SELECT ts, bids, asks FROM ob_snapshots WHERE code=? "
        "AND ts >= ? AND ts < ? ORDER BY ts", (code, start, end)).fetchall()
    con.close()
    out = []
    for ts, b, a in rows:
        snap = [{"price": p, "volume": v, "side": "BID"}
                for p, v in json.loads(b)]
        snap += [{"price": p, "volume": v, "side": "ASK"}
                 for p, v in json.loads(a)]
        if snap:
            out.append((datetime.fromisoformat(ts), snap))
    return out


pushes = load(CODE, START, SECS)
if not pushes:
    sys.exit(f"no snapshots for {CODE} at {START}")

T0 = pushes[0][0]
gaps = [(pushes[i + 1][0] - pushes[i][0]).total_seconds()
        for i in range(len(pushes) - 1)]
gaps_s = sorted(gaps)
print(f"配置  max_cols={vcfg.liq('max_cols')} n_price={vcfg.liq('n_price')} "
      f"col_secs={vcfg.liq('col_secs')} gamma={vcfg.liq('gamma')} "
      f"absorb={vcfg.liq('absorb')} imbalance={vcfg.liq('imbalance')} "
      f"wall={vcfg.liq('wall')} ice={vcfg.liq('iceberg')} "
      f"spoof={vcfg.liq('spoof')}")
print(f"负载  {SCEN}  {CODE} {START}+{SECS}s  {len(pushes):,} 推送 "
      f"({len(pushes)/SECS:.2f}/s)  gap 中位 {gaps_s[len(gaps_s)//2]*1000:.0f}ms "
      f"p95 {gaps_s[int(len(gaps_s)*0.95)]*1000:.0f}ms")

# ── virtual clock ─────────────────────────────────────────────────────────────
V = {"t": 0.0}          # seconds since T0


def vnow():
    return V["t"]


def vdt():
    return T0 + timedelta(seconds=V["t"])


class VDatetime(datetime):
    """datetime whose now() follows the replay, so column stamps and the
    absorb-tick query land on the recorded session, not on the real clock."""
    @classmethod
    def now(cls, tz=None):
        d = vdt()
        return d.replace(tzinfo=tz) if tz is not None else d


L.datetime = VDatetime


# ── fake timers ───────────────────────────────────────────────────────────────
class FakeSignal:
    def __init__(self):
        self._slots = []

    def connect(self, fn):
        self._slots.append(fn)

    def disconnect(self, *a):
        self._slots = []

    def emit(self, *a):
        for s in list(self._slots):
            s(*a)


class FakeTimer:
    def __init__(self, name):
        self.name = name
        self.timeout = FakeSignal()
        self._single = False
        self._iv = 0.0
        self._due = None
        self.fires = 0

    def setSingleShot(self, v):
        self._single = v

    def start(self, ms=None):
        if ms is not None:
            self._iv = ms / 1000.0
        self._due = vnow() + self._iv

    def stop(self):
        self._due = None

    def isActive(self):
        return self._due is not None

    def due(self):
        return self._due

    def fire(self):
        if self._single:
            self._due = None
        else:
            self._due = self._due + self._iv
        self.fires += 1
        self.timeout.emit()


# ── synchronous stand-ins for the QThread workers ─────────────────────────────
class _SyncWorker:
    def __init__(self):
        self.done = FakeSignal()

    def isRunning(self):
        return False

    def start(self):
        self.done.emit(self._run())

    def quit(self):
        pass

    def wait(self, *a):
        return True


class SyncAbsorb(_SyncWorker):
    calls = 0
    secs = 0.0
    rows = 0

    def __init__(self, code, start, end):
        super().__init__()
        self._a = (code, start, end)

    def _run(self):
        t = time.perf_counter()
        r = L._query_ticks(*self._a)
        print(f"[ABS] start={self._a[1]} end={self._a[2]} rows={len(r)}",
              flush=True)
        SyncAbsorb.calls += 1
        SyncAbsorb.secs += time.perf_counter() - t
        SyncAbsorb.rows += len(r)
        return r


PREFILL = {"snaps": []}


class SyncPrefill(_SyncWorker):
    def __init__(self, code, cols, bucket):
        super().__init__()

    def _run(self):
        return PREFILL["snaps"]


L._AbsorbTickWorker = SyncAbsorb
L._WindowPrefillWorker = SyncPrefill


class SyncNoSnap(_SyncWorker):
    """Returns nothing, and that is load-bearing rather than a shortcut.

    _on_tick falls back to this worker while _push_ok is still False. The
    real one reads store.latest_ts(), which during a replay of an older
    session returns the NEWEST row in the database instead -- the known
    _load_history/latest_ts bug, still open in TODO.md. That stamp lands on
    a fill column, the absorb query's `end` jumps forward by whatever the
    gap is, and one call drags back the better part of a million ticks
    (measured: 877,105 against the 8,272 the real twelve-minute window
    holds). The replay then measures that instead of the heatmap.

    Live, this path returns almost nothing anyway, because latest_ts() is
    approximately now.
    """

    def __init__(self, code):
        super().__init__()

    def _run(self):
        return []


L._SnapshotWorker = SyncNoSnap

# ── instrumentation ───────────────────────────────────────────────────────────
STATS = {}
_stack = []


def _acct(name, dt):
    s = STATS.setdefault(name, {"n": 0, "incl": 0.0, "excl": 0.0})
    s["n"] += 1
    s["incl"] += dt


def wrap_obj(obj, name):
    orig = getattr(obj, name)

    def f(*a, **k):
        _stack.append(0.0)
        t = time.perf_counter()
        try:
            return orig(*a, **k)
        finally:
            d = time.perf_counter() - t
            child = _stack.pop()
            if _stack:
                _stack[-1] += d
            s = STATS.setdefault(name, {"n": 0, "incl": 0.0, "excl": 0.0})
            s["n"] += 1
            s["incl"] += d
            s["excl"] += d - child
    setattr(obj, name, f)


def wrap_mod(mod, name):
    orig = getattr(mod, name)

    def f(*a, **k):
        _stack.append(0.0)
        t = time.perf_counter()
        try:
            return orig(*a, **k)
        finally:
            d = time.perf_counter() - t
            child = _stack.pop()
            if _stack:
                _stack[-1] += d
            s = STATS.setdefault(name, {"n": 0, "incl": 0.0, "excl": 0.0})
            s["n"] += 1
            s["incl"] += d
            s["excl"] += d - child
    setattr(mod, name, f)


for fn in ("detect_stacked_imbalance", "detect_aggressor_bubbles",
           "detect_icebergs", "detect_spoofs"):
    wrap_mod(OD, fn)
# _single_rgba used to be patchable from here while the sparse rewrite was
# being proved out. That version shipped in v0.104.5, so the copy is gone --
# a benchmark carrying its own implementation of the thing it measures is
# how this file broke the last time (it still referenced _raw_snaps after
# v0.104.2 renamed it).
wrap_mod(L, "_single_rgba")
wrap_mod(L, "_hot_rgba")
wrap_mod(L, "_query_ticks")

PAINTS = {"n": 0}
import pyqtgraph as pg                                        # noqa: E402

# ANTIALIASING IS OFF HERE AND ON IN THE APP, which is a real gap between
# this ruler and what it stands in for. pyqtgraph defaults `antialias` to
# False; the app turns it on in analysis/trade_viewer_qt.py, and importing
# liq_hm_window alone -- which is all this harness does -- never reaches
# that line. Everything measured with the default setting is therefore
# measured without it.
#
# The default is left alone rather than "fixed", because the CHANGELOG
# numbers from v0.104.5 onward were taken with it off and silently moving
# the ruler would invalidate them without saying so. Set MOOMOO_AA=1 to
# match the app; both arms of any comparison have to use the same setting.
_AA = os.environ.get("MOOMOO_AA") == "1"
pg.setConfigOptions(antialias=_AA)
print(f"[AA] antialias={_AA}  (the app runs True)", flush=True)
_orig_paint = pg.ImageItem.paint


def _counting_paint(self, *a, **k):
    PAINTS["n"] += 1
    return _orig_paint(self, *a, **k)


pg.ImageItem.paint = _counting_paint

# ── build the window ──────────────────────────────────────────────────────────
app = QApplication(sys.argv[:1])
w = L.LiqHmWindow()
w.resize(1500, 950)
w.show()
app.processEvents()

def log_call(obj, name):
    orig = getattr(obj, name)

    def f(*a, **k):
        print(f"[EV] sim={V['t']:.1f} {name} cols={len(w._col_ts)} "
              f"raw={w._snap_rows:,}", flush=True)
        return orig(*a, **k)
    setattr(obj, name, f)


for m in ("_clear_columns", "_reset_grid", "_on_snap_ready",
          "_resubscribe_push", "_refill_after_rebuild", "_on_bulk_ready"):
    if hasattr(w, m):
        log_call(w, m)

wrap_mod(L, "_percentile_norm")
for m in ("_wall_fate", "_row_label", "_wall_label_x",
          "_on_push", "_on_tick", "_render", "_push_column", "_paint_column",
          "_maybe_init_price_range", "_redraw_orderflow_markers",
          "_load_absorb_ticks", "_on_absorb_ready", "_update_price_path",
          "_draw_wall_labels", "_build_bucket_to_idx", "_clear_overlay_items",
          "_draw_absorb_bubbles", "_draw_simb_markers"):
    if hasattr(w, m):
        wrap_obj(w, m)

ITEMS = {"add": 0, "rm": 0}
_pw = w._plot_widget
_oadd, _orm = _pw.addItem, _pw.removeItem


_IGB = os.environ.get("MOOMOO_IGB") == "1"
_NOADD = os.environ.get("MOOMOO_NOADD") == "1"


def _cadd(*a, **k):
    ITEMS["add"] += 1
    if _NOADD:
        return None
    if _IGB:
        k["ignoreBounds"] = True
    return _oadd(*a, **k)


def _crm(*a, **k):
    ITEMS["rm"] += 1
    if _NOADD:
        return None
    return _orm(*a, **k)


_pw.addItem, _pw.removeItem = _cadd, _crm

if os.environ.get("MOOMOO_OVL") == "freeze":
    _ord = w._redraw_orderflow_markers
    _once = {"done": False}

    def _froz():
        if _once["done"]:
            return
        _once["done"] = True
        return _ord()

    w._redraw_orderflow_markers = _froz
    print("[OVL] frozen after first redraw", flush=True)

SAME = {"n": 0, "same": 0, "prev": None}
_ordm = w._redraw_orderflow_markers


def _sig_items():
    out = []
    for lst in (w._iceberg_items, w._spoof_items, w._simb_items,
                w._absorb_items, w._wall_items):
        for it in lst:
            try:
                pos = it.pos()
                xy = (round(float(pos.x()), 4), round(float(pos.y()), 4))
            except Exception:
                xy = None
            txt = ""
            if hasattr(it, "toPlainText"):
                try:
                    txt = it.toPlainText()
                except Exception:
                    txt = "?"
            out.append((type(it).__name__, xy, txt))
    return tuple(out)


def _counting_redraw():
    r = _ordm()
    sig = _sig_items()
    SAME["n"] += 1
    if SAME["prev"] is not None and sig == SAME["prev"]:
        SAME["same"] += 1
    SAME["prev"] = sig
    return r


w._redraw_orderflow_markers = _counting_redraw

timers = {}
for attr, slot in (("_timer", "_on_tick"), ("_render_timer", "_render"),
                   ("_marker_timer", "_redraw_orderflow_markers"),
                   ("_refill_timer", "_refill_after_rebuild")):
    ft = FakeTimer(attr)
    ft.setSingleShot(attr != "_timer")
    ft.timeout.connect(getattr(w, slot))
    setattr(w, attr, ft)
    timers[attr] = ft

w._start_push = lambda code: True        # no OpenD; the SDK thread is excluded

# warm: hand the real pre-fill the bucketed snapshots live would have had
if SCEN == "warm":
    from feeds.order_book_store import OrderBookStore
    cols = w._max_cols_spin.value()
    bucket = w._col_secs_spin.value()
    with OrderBookStore(L._DB_PATH, read_only=True) as store:
        PREFILL["snaps"] = store.bucketed_snapshots(
            CODE, T0 - timedelta(seconds=cols * bucket), T0, bucket)[-cols:]
    print(f"预填  {len(PREFILL['snaps'])} 列 (目标 {cols})")

w.set_code(CODE)
if SCEN == "bypass":
    w._code, w._live = CODE, True        # exactly what the original did
else:
    w.set_live(True)

# ── drive ─────────────────────────────────────────────────────────────────────
proc = psutil.Process()
rss0 = proc.memory_info().rss
c0 = proc.cpu_times()
wall0 = time.perf_counter()
trace = []


def fire_due(until):
    while True:
        due = [t for t in timers.values()
               if t.due() is not None and t.due() <= until]
        if not due:
            return
        t = min(due, key=lambda x: x.due())
        V["t"] = t.due()
        t.fire()
        if PROC:
            app.processEvents()


for i, (ts, snap) in enumerate(pushes):
    tgt = (ts - T0).total_seconds()
    fire_due(tgt)
    V["t"] = tgt
    w._on_push(CODE, snap, time.time())
    if PROC == "1":
        app.processEvents()
    if i % 200 == 0:
        trace.append((round(V["t"]), len(w._col_ts), w._snap_rows,
                      proc.memory_info().rss / 1024 ** 2))
fire_due(V["t"] + 0.001)
app.processEvents()

wall = time.perf_counter() - wall0
c1 = proc.cpu_times()
cpu = (c1.user - c0.user) + (c1.system - c0.system)
rss1 = proc.memory_info().rss
sim = (pushes[-1][0] - T0).total_seconds()

print(f"\n模拟 {sim:.0f}s 用壁钟 {wall:.1f}s (x{sim/wall:.1f})")
print(f"CPU   {cpu:.2f}s / 模拟 {sim:.0f}s = 单核 {100*cpu/sim:.1f}%  "
      f"= 每次推送 {1000*cpu/len(pushes):.2f} ms")
print(f"RSS   {rss0/1024**2:,.0f} -> {rss1/1024**2:,.0f} MB "
      f"({(rss1-rss0)/1024**2:+,.0f})")
print(f"列数  {len(w._col_ts)}   网格 {w._bid_grid.shape}   "
      f"_raw_snaps {w._snap_rows:,}   ImageItem.paint {PAINTS['n']}")
print(f"定时器 " + "  ".join(f"{k}={v.fires}" for k, v in timers.items()))
print(f"absorb 查询 {SyncAbsorb.calls} 次 {SyncAbsorb.secs:.2f}s "
      f"{SyncAbsorb.rows:,} 行")

print(f"\n{'函数':34s} {'次数':>7s} {'含子 s':>9s} {'自身 s':>9s} "
      f"{'自身 ms/次':>10s}  占模拟%")
for k, s in sorted(STATS.items(), key=lambda kv: -kv[1]["excl"]):
    print(f"{k:34s} {s['n']:7,d} {s['incl']:9.3f} {s['excl']:9.3f} "
          f"{1000*s['excl']/max(s['n'],1):10.3f}  {100*s['excl']/sim:6.1f}%")

print(f"[SAME] {SAME['same']:,} of {SAME['n']:,} overlay redraws drew an identical set")
print(f"[ITEMS] addItem {ITEMS['add']:,} removeItem {ITEMS['rm']:,}")
print(f"absorb_ticks buffer = {len(w._absorb_ticks):,} rows")
print("\n轨迹 (模拟秒, 列, raw_snaps, RSS MB)")
for r in trace[::max(1, len(trace)//12)]:
    print("   ", r)

# ── the trim line, measured on the real buffer rather than guessed ────────────
rs = list(__import__("itertools").chain.from_iterable(w._snap_cols))
if rs:
    cut = rs[len(rs) // 2]["ts"]
    for n in (len(rs)//4, len(rs)//2, len(rs)):
        sub = rs[:n]
        t = time.perf_counter()
        for _ in range(5):
            _ = [s for s in sub if s["ts"] >= cut]
        dt = (time.perf_counter() - t) / 5
        print(f"裁剪  n={n:,} -> {dt*1000:.2f} ms/次")
    import sys as _s
    per = _s.getsizeof(rs[0]) + sum(_s.getsizeof(v) for v in rs[0].values())
    print(f"每条 _raw_snaps 约 {per} B (dict+值, 不含共享键)")
app.quit()
