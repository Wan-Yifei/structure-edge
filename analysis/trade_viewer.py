"""
Trade Viewer — Tkinter GUI + headless CLI (formerly orderflow.py)

Modes
-----
GUI (default):
    uv run analysis/trade_viewer.py
    uv run analysis/trade_viewer.py --code US.AAPL --tf 5m --mode Historical --date 2026-05-15

Trade Review (jump to a specific backtest / live trade by ID):
    uv run analysis/trade_viewer.py --trade-id <uuid>

Headless (save PNG without opening a window):
    uv run analysis/trade_viewer.py --code US.SNDK --date 2026-05-15 --output chart.png

Via main entry point:
    uv run main.py trade_viewer [same args as above]

Trade Review mode
-----------------
Enter a trade_id (from backtest.duckdb — either backtest trades or live/paper trades)
in the "Trade ID" field and click Load.  The viewer will:
  • Auto-populate Code / Timeframe / Date from the trade record.
  • Switch to Historical mode.
  • Overlay ONLY the structures relevant to the entry decision:
      – The single HTF FVG zone that was entered (semi-transparent band).
      – The last 1-2 BOS/CHoCH signals that confirmed the trend.
      – Entry arrow (▲ bull / ▼ bear) at entry_price.
      – Exit marker (○ win / ✕ loss) at exit_price.
      – SL and TP horizontal lines.
  All other BOS/CHoCH, FVG and OB overlays are suppressed to keep the chart clean.
"""

import argparse
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import time
import tkinter as tk
from tkinter import ttk, messagebox
import threading
from collections import defaultdict
from datetime import datetime, timedelta

import json
import tomllib

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import pandas as pd
import ta as _ta
from moomoo import (
    OpenQuoteContext, SubType, KLType, AuType,
    TickerHandlerBase, RET_OK,
)

from core.time_utils import candle_start
from core.chart import (
    make_annot, make_float_tip,
    BG_DARK, BG_BAR, BG_EDIT, BG_TIP,
    FG, GREEN, RED, GREY, GRID, GOLD, CROSS,
    UP, DOWN,
)
from core.draw import (
    draw_candles, draw_tick_profile_bars, draw_ohlcv_profile,
    draw_hybrid_profile, build_hybrid_profile,
    draw_candle_heatmap, draw_candle_deltas,
    draw_bos_choch, draw_fvg, draw_order_blocks, draw_kd,
    aggregate_buckets, bucket_coverage, prices_arrays,
    _VA_COLOR,
)
from strategy.smc.market_structure import detect_bos_choch
from strategy.smc.fvg import detect_fvg
from strategy.smc.order_blocks import detect_order_blocks
from strategy.smc.kd_trend import compute_kd

# ── Performance monitor ───────────────────────────────────────────────────────

class PerfStats:
    """Rolling per-operation timing, toggled with the P key."""

    WINDOW = 30   # samples kept per operation

    def __init__(self):
        self._data: dict[str, list[float]] = {}
        self._t0: dict[str, float] = {}

    def start(self, name: str) -> None:
        self._t0[name] = time.perf_counter()

    def end(self, name: str) -> float:
        dt = (time.perf_counter() - self._t0.pop(name, time.perf_counter())) * 1000
        bucket = self._data.setdefault(name, [])
        bucket.append(dt)
        if len(bucket) > self.WINDOW:
            del bucket[0]
        return dt

    def record(self, name: str, ms: float) -> None:
        bucket = self._data.setdefault(name, [])
        bucket.append(ms)
        if len(bucket) > self.WINDOW:
            del bucket[0]

    def summary(self) -> str:
        lines = ["── perf (ms) ──"]
        for name, vals in self._data.items():
            if not vals:
                continue
            avg = sum(vals) / len(vals)
            mx  = max(vals)
            lines.append(f"{name:<14} avg {avg:5.0f}  max {mx:5.0f}")
        return "\n".join(lines)

    def clear(self) -> None:
        self._data.clear()


class _SafeNavToolbar(NavigationToolbar2Tk):
    """NavigationToolbar2Tk that guards against _pan_info=None during chart redraws."""
    def drag_pan(self, event):
        if self._pan_info is None:
            return
        super().drag_pan(event)

    def release_pan(self, event):
        if self._pan_info is None:
            return
        super().release_pan(event)


# ── Viewer config (which indicators are available in the GUI) ─────────────────

_VIEWER_CFG_PATH = pathlib.Path(__file__).parent.parent / "config" / "trade_viewer.toml"

def _load_viewer_config() -> dict:
    """Return the trade_viewer.toml config, or empty dict if file is absent."""
    if _VIEWER_CFG_PATH.exists():
        try:
            with open(_VIEWER_CFG_PATH, "rb") as _f:
                return tomllib.load(_f)
        except Exception:
            pass
    return {}

# ── Timeframe config ──────────────────────────────────────────────────────────
TIMEFRAME_MAP: dict[str, tuple[KLType, int]] = {
    "1m":  (KLType.K_1M,   1),
    "5m":  (KLType.K_5M,   5),
    "15m": (KLType.K_15M, 15),
    "30m": (KLType.K_30M, 30),
    "1h":  (KLType.K_60M, 60),
    "4h":  (KLType.K_240M, 240),
}

# Candle count ≈ one trading day per timeframe (US/HK session, ~6.5 h)
_DAY_CANDLES: dict[str, int] = {
    "1m":  390,
    "5m":   78,
    "15m":  26,
    "30m":  14,
    "1h":    7,
    "4h":    6,
}

# Max BOS/CHoCH span in bars per timeframe.
# Shorter TFs use a 4H window; 4H uses a 1-week window (~5 trading days).
_BOS_MAX_SPAN: dict[str, int] = {
    "1m":  240,   # 4H
    "5m":   48,   # 4H
    "15m":  16,   # 4H
    "30m":  14,   # 1 day
    "1h":    7,   # 1 day
    "4h":   30,   # 1 week (5 days × 6 bars/day)
}

# ── Technical indicator registries ────────────────────────────────────────────

# Indicators drawn on the price chart (ax_c)
_TA_OVERLAY: dict[str, dict] = {
    "EMA 9":           {"color": "#5c9cf5", "lw": 1.2},
    "EMA 21":          {"color": "#ce93d8", "lw": 1.2},
    "EMA 50":          {"color": "#f9a825", "lw": 1.5},
    "EMA 200":         {"color": "#ef5350", "lw": 1.5},
    "Bollinger Bands": {"color": "#80cbc4", "lw": 0.9},
    "KD":              {"fast": 25, "slow": 90,
                        "color_fast": "#26a69a", "color_slow": "#5c9cf5", "lw": 1.0},
}

# Indicators drawn in a separate subplot below the price chart.
# "MAVOL" is special: it shares the "Vol" panel when both are active,
# so it does not create its own row in that case.
_TA_SUBPLOT: dict[str, dict] = {
    "MACD":  {},
    "RSI":   {},
    "ATR":   {},
    "Vol":   {},
    "MAVOL": {},
}


class OrderFlowApp(tk.Tk):
    def __init__(self, args: argparse.Namespace | None = None):
        super().__init__()
        self.title("Trade Viewer")
        self.configure(bg=BG_DARK)
        self.geometry("1280x760")
        self.minsize(960, 580)

        self.ctx          = None
        self.running      = False
        self.tick_lock    = threading.Lock()
        self.tick_buckets: dict = defaultdict(
            lambda: defaultdict(lambda: {"buy": 0, "sell": 0, "neutral": 0})
        )
        self._refresh_job = None

        # data for hover / crosshair lookup
        self._klines_data:   pd.DataFrame | None = None
        self._profile_ohlcv: tuple | None        = None
        self._profile_tick:  dict | None         = None

        # tracked session-profile artists
        self._profile_bar_rects: list = []      # bars on ax_p
        self._profile_axvline         = None
        self._profile_legend          = None
        self._candle_profile_artists: list = []  # POC + VA lines on ax_c
        self._hist_klines: pd.DataFrame | None = None   # full unfiltered klines for zoom rebuild
        self._profile_centers: np.ndarray | None = None  # bin centres from last profile draw
        self._profile_total:   np.ndarray | None = None  # total volume per bin (for tight ylim)
        self._live_buckets: dict = {}
        self._hist_buckets: dict | None = None

        # session filter checkboxes (populated by _build_toolbar)
        self._sess_vars: dict[str, tk.BooleanVar] = {}

        # tracked per-candle tick panel artists (ax_t — rebuilt on each hover)
        self._tick_bar_rects: list = []
        self._tick_axvline           = None
        self._tick_legend            = None
        self._tick_shown_idx: int | None = None
        self._tick_data_cache: dict[int, tuple] = {}  # idx -> (prices,buy,sell,neu,title)
        self._hovered_candle_idx: int | None = None

        # SMC overlay artists (cleared on each refresh)
        self._smc_artists: list = []

        # annotation / crosshair objects — recreated after each fig.clear()
        self._tip_c  = None
        self._tip_p  = None
        self._ch_hline_c = None
        self._ch_vline_c = None
        self._ch_hline_t = None
        self._ch_hline_p = None
        self._ch_label   = None
        self._ch_labels_sub: list = []   # one floating label per subplot axis

        # TA series for subplot crosshair readouts (list of pd.Series, one per subplot)
        self._ta_series_sub: list = []
        # full klines used for TA warmup (more bars than displayed)
        self._klines_warmup: pd.DataFrame | None = None

        # blitting: saved background for fast crosshair/tooltip rendering
        # _bg_t / _bg_p are per-axes snapshots used for the side-panel hline blit
        self._bg:   object = None
        self._bg_t: object = None
        self._bg_p: object = None
        self._draw_event_cid: int | None = None
        self._fetching: bool = False
        self._last_code: str = "US.SNDK"   # tracks last-applied code for change detection

        # performance monitoring (toggle with P key)
        self._perf = PerfStats()
        self._perf_visible = False
        self._perf_text = None   # Text artist in ax_c
        self._scroll_job: str | None = None  # debounce scroll redraws

        self.bind("<p>", self._toggle_perf)
        self.bind("<P>", self._toggle_perf)

        # indicator toggles — loaded from chart_config.json
        self._ind: dict[str, tk.BooleanVar] = {}
        self._cfg_path = pathlib.Path(__file__).parent.parent / "config" / "chart.json"

        # load availability config (trade_viewer.toml)
        _vcfg = _load_viewer_config()
        _all_of = ["heatmap", "delta", "bos_choch", "fvg", "ob"]
        self._avail_orderflow: set[str] = set(
            _vcfg.get("orderflow", {}).get("enabled", _all_of))
        self._avail_ta_overlay: list[str] = _vcfg.get("ta_overlay", {}).get(
            "enabled", list(_TA_OVERLAY.keys()))
        self._avail_ta_subplot: list[str] = _vcfg.get("ta_subplot", {}).get(
            "enabled", list(_TA_SUBPLOT.keys()))

        self._load_indicator_cfg()

        # trade review state — set by _load_trade_by_id()
        self._trade_record: dict | None = None
        self._trade_overlay_artists: list = []

        # technical indicators state
        self._ta_overlay: list[str] = []    # active overlay indicator names
        self._ta_subplot: list[str] = []    # active subplot indicator names
        self._ta_axes:    list      = []    # matplotlib axes for subplot indicators
        self._ch_vlines_sub: list  = []    # crosshair vlines for subplot axes

        self._build_toolbar()
        self._build_chart()
        self._build_statusbar()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        if args is not None:
            self.code_var.set(args.code)
            self.tf_var.set(args.tf)
            self.num_var.set(args.num)
            self.mode_var.set(args.mode)
            self.host_var.set(args.host)
            self.port_var.set(args.port)
            self.refresh_var.set(args.refresh)
            if args.date:
                self.date_var.set(args.date)
            if getattr(args, "trade_id", None):
                self._trade_id_var.set(args.trade_id)
            # override refresh from schedule.json if present
            cfg_path = pathlib.Path(__file__).parent.parent / "config" / "schedule.json"
            if cfg_path.exists():
                try:
                    import json as _json
                    with open(cfg_path) as _f:
                        _cfg = _json.load(_f)
                    secs = int(_cfg.get("live_refresh_seconds", 0))
                    if secs >= 5:
                        self.refresh_var.set(secs)
                except Exception:
                    pass

        self._on_mode_change()
        # Auto-load trade if --trade-id was passed on CLI
        if args is not None and getattr(args, "trade_id", None):
            self.after(200, self._load_trade_by_id)

    # ── Config helpers ────────────────────────────────────────────────────────

    _IND_DEFAULTS = {
        "heatmap": True, "delta": True,
        "bos_choch": False, "fvg": False, "ob": False,
        "trade_review": False,
    }

    def _load_indicator_cfg(self):
        cfg = {}
        if self._cfg_path.exists():
            try:
                cfg = json.loads(self._cfg_path.read_text(encoding="utf-8")).get("indicators", {})
            except Exception:
                pass
        for key, default in self._IND_DEFAULTS.items():
            # trade_review is always available; orderflow keys respect avail config
            if key != "trade_review" and key not in self._avail_orderflow:
                continue
            self._ind[key] = tk.BooleanVar(value=cfg.get(key, default))

    def _save_indicator_cfg(self):
        data = {"_comment": "Toggle each indicator overlay in the chart window.",
                "indicators": {k: v.get() for k, v in self._ind.items()}}
        try:
            self._cfg_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_toolbar(self):
        bar = tk.Frame(self, bg=BG_BAR, pady=7)
        bar.pack(fill=tk.X, side=tk.TOP)

        def lbl(text):
            tk.Label(bar, text=text, bg=BG_BAR, fg=FG).pack(side=tk.LEFT, padx=(10, 2))

        lbl("Code:")
        self.code_var = tk.StringVar(value="US.SNDK")
        code_entry = tk.Entry(bar, textvariable=self.code_var, width=12,
                              bg=BG_EDIT, fg=FG, insertbackground=FG)
        code_entry.pack(side=tk.LEFT, padx=(0, 10))
        code_entry.bind("<Return>",   lambda _: self._on_code_change())
        code_entry.bind("<FocusOut>", lambda _: self._on_code_change())

        lbl("Timeframe:")
        self.tf_var = tk.StringVar(value="15m")
        tf_cb = ttk.Combobox(bar, textvariable=self.tf_var, values=list(TIMEFRAME_MAP),
                              width=5, state="readonly")
        tf_cb.pack(side=tk.LEFT, padx=(0, 10))
        tf_cb.bind("<<ComboboxSelected>>", lambda _: self._on_tf_change())

        lbl("Candles:")
        self.num_var = tk.IntVar(value=_DAY_CANDLES.get(self.tf_var.get(), 26))
        num_spin = tk.Spinbox(bar, from_=5, to=1000, textvariable=self.num_var,
                              width=4, bg=BG_EDIT, fg=FG, buttonbackground=BG_BAR)
        num_spin.pack(side=tk.LEFT, padx=(0, 10))
        num_spin.bind("<<Increment>>", lambda _: self._on_num_change())
        num_spin.bind("<<Decrement>>", lambda _: self._on_num_change())
        num_spin.bind("<Return>",      lambda _: self._on_num_change())
        num_spin.bind("<FocusOut>",    lambda _: self._on_num_change())

        lbl("Mode:")
        self.mode_var = tk.StringVar(value="Live")
        mode_cb = ttk.Combobox(bar, textvariable=self.mode_var,
                               values=["Live", "Historical"], width=10, state="readonly")
        mode_cb.pack(side=tk.LEFT, padx=(0, 10))
        mode_cb.bind("<<ComboboxSelected>>", lambda _: self._on_mode_change())

        lbl("Date (YYYY-MM-DD):")
        self.date_var = tk.StringVar(
            value=(datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d"))
        self.date_entry = tk.Entry(bar, textvariable=self.date_var, width=12,
                                   bg=BG_EDIT, fg=FG, insertbackground=FG)
        self.date_entry.pack(side=tk.LEFT, padx=(0, 10))
        self.date_entry.bind("<Return>",   lambda _e: self._on_date_change())
        self.date_entry.bind("<FocusOut>", lambda _e: self._on_date_change())

        lbl("Refresh (s, min 5):")
        self.refresh_var = tk.IntVar(value=15)
        self.refresh_spin = tk.Spinbox(bar, from_=5, to=300, textvariable=self.refresh_var,
                                       width=4, bg=BG_EDIT, fg=FG, buttonbackground=BG_BAR)
        self.refresh_spin.pack(side=tk.LEFT, padx=(0, 10))

        lbl("Host:")
        self.host_var = tk.StringVar(value="127.0.0.1")
        tk.Entry(bar, textvariable=self.host_var, width=13,
                 bg=BG_EDIT, fg=FG, insertbackground=FG).pack(side=tk.LEFT, padx=(0, 4))

        lbl("Port:")
        self.port_var = tk.IntVar(value=11111)
        tk.Entry(bar, textvariable=self.port_var, width=6,
                 bg=BG_EDIT, fg=FG, insertbackground=FG).pack(side=tk.LEFT, padx=(0, 10))

        self.start_btn = tk.Button(bar, text="Start", bg=GREEN, fg=FG,
                                   width=7, relief=tk.FLAT, command=self._start)
        self.start_btn.pack(side=tk.LEFT, padx=4)

        self.stop_btn = tk.Button(bar, text="Stop", bg=RED, fg=FG,
                                  width=7, relief=tk.FLAT, command=self._stop,
                                  state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=4)

        # ── Indicator control row ──────────────────────────────────────────
        ind_bar = tk.Frame(self, bg=BG_BAR, pady=3)
        ind_bar.pack(fill=tk.X, side=tk.TOP)
        tk.Label(ind_bar, text="Indicators:", bg=BG_BAR, fg=GREY,
                 font=("TkDefaultFont", 8)).pack(side=tk.LEFT, padx=(10, 6))

        # Width vars for FVG and OB rectangles (bars)
        self._fvg_width_var = tk.IntVar(value=20)
        self._ob_width_var  = tk.IntVar(value=30)

        _IND_LABELS = [
            ("heatmap",      "Heatmap"),
            ("delta",        "Delta Δ"),
            ("bos_choch",    "BOS / CHoCH"),
            ("fvg",          "FVG"),
            ("ob",           "Order Blocks"),
        ]
        for key, label in _IND_LABELS:
            if key not in self._avail_orderflow:
                continue
            ttk.Checkbutton(
                ind_bar, text=label,
                variable=self._ind[key],
                command=self._on_indicator_toggle,
                style="Ind.TCheckbutton",
            ).pack(side=tk.LEFT, padx=(6, 2))
            # Width spinbox for FVG and OB
            if key == "fvg":
                tk.Spinbox(
                    ind_bar, from_=5, to=200, textvariable=self._fvg_width_var,
                    width=3, bg=BG_EDIT, fg=FG, buttonbackground=BG_BAR,
                    command=self._on_indicator_toggle,
                ).pack(side=tk.LEFT, padx=(0, 4))
            elif key == "ob":
                tk.Spinbox(
                    ind_bar, from_=5, to=200, textvariable=self._ob_width_var,
                    width=3, bg=BG_EDIT, fg=FG, buttonbackground=BG_BAR,
                    command=self._on_indicator_toggle,
                ).pack(side=tk.LEFT, padx=(0, 4))

        # separator + Trade Review toggle (right-aligned)
        tk.Label(ind_bar, text="|", bg=BG_BAR, fg=GREY).pack(side=tk.LEFT, padx=8)
        ttk.Checkbutton(
            ind_bar, text="Trade Review",
            variable=self._ind["trade_review"],
            command=self._on_indicator_toggle,
            style="Ind.TCheckbutton",
        ).pack(side=tk.LEFT, padx=6)

        # ── Profile session filter ────────────────────────────────────────
        tk.Label(ind_bar, text="|", bg=BG_BAR, fg=GREY).pack(side=tk.LEFT, padx=8)
        tk.Label(ind_bar, text="Profile:", bg=BG_BAR, fg=GREY,
                 font=("TkDefaultFont", 8)).pack(side=tk.LEFT, padx=(0, 4))
        _SESS_LABELS = [
            ("regular", "Regular"),
            ("pre",     "Pre"),
            ("post",    "Post"),
            ("night",   "Night"),
        ]
        for key, label in _SESS_LABELS:
            var = tk.BooleanVar(value=True)
            self._sess_vars[key] = var
            ttk.Checkbutton(
                ind_bar, text=label, variable=var,
                command=self._on_session_toggle,
                style="Ind.TCheckbutton",
            ).pack(side=tk.LEFT, padx=(2, 0))

        # ── Trade ID row ──────────────────────────────────────────────────
        self._trade_bar = tk.Frame(self, bg=BG_BAR, pady=4)
        self._trade_bar.pack(fill=tk.X, side=tk.TOP)

        tk.Label(self._trade_bar, text="Trade ID:", bg=BG_BAR, fg=GREY,
                 font=("TkDefaultFont", 8)).pack(side=tk.LEFT, padx=(10, 4))
        self._trade_id_var = tk.StringVar()
        tk.Entry(self._trade_bar, textvariable=self._trade_id_var, width=38,
                 bg=BG_EDIT, fg=FG, insertbackground=FG,
                 font=("Consolas", 8)).pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(self._trade_bar, text="Load Trade", bg=BG_BAR, fg=GOLD,
                  relief=tk.FLAT, font=("TkDefaultFont", 8),
                  command=self._load_trade_by_id).pack(side=tk.LEFT, padx=(0, 10))
        tk.Label(self._trade_bar,
                 text="(backtest or live/paper trade UUID — auto-fills Code / TF / Date)",
                 bg=BG_BAR, fg=GREY, font=("TkDefaultFont", 7)
                 ).pack(side=tk.LEFT)

    # ── Toolbar auto-refresh handlers ─────────────────────────────────────────

    def _on_code_change(self):
        new = self.code_var.get().strip()
        if new == getattr(self, "_last_code", None) or not self.running:
            self._last_code = new
            return
        self._last_code = new
        if self.mode_var.get() == "Live":
            self._stop(); self._start()
        else:
            self._refresh_chart()

    def _on_tf_change(self):
        self._tick_data_cache = {}   # per-candle cache is TF-specific
        self.num_var.set(_DAY_CANDLES.get(self.tf_var.get(), 26))
        if not self.running:
            return
        if self.mode_var.get() == "Live":
            self._stop(); self._start()
        else:
            self._refresh_chart()

    def _on_num_change(self):
        if self.running and self._klines_data is not None:
            self._refresh_chart()

    def _on_date_change(self):
        try:
            datetime.strptime(self.date_var.get().strip(), "%Y-%m-%d")
        except ValueError:
            return
        if self.running and self._klines_data is not None \
                and self.mode_var.get() != "Live":
            self._refresh_chart()

    def _on_indicator_toggle(self):
        self._save_indicator_cfg()
        if self.running and self._klines_data is not None:
            self._refresh_chart()

    def _build_chart(self):
        chart_frame = tk.Frame(self, bg=BG_DARK)
        chart_frame.pack(fill=tk.BOTH, expand=True)

        # ── left: matplotlib canvas ───────────────────────────────────────────
        canvas_frame = tk.Frame(chart_frame, bg=BG_DARK)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig = Figure(facecolor=BG_DARK)
        self.ax_c = self.fig.add_subplot(121)
        self.ax_p = self.fig.add_subplot(122)
        self._style_axes()

        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        tb_frame = tk.Frame(canvas_frame, bg=BG_DARK)
        tb_frame.pack(fill=tk.X)
        _SafeNavToolbar(self.canvas, tb_frame)

        self.canvas.mpl_connect("motion_notify_event", self._on_hover)
        self.canvas.mpl_connect("axes_leave_event",    self._on_axes_leave)
        self.canvas.mpl_connect("scroll_event",        self._on_scroll)
        self.canvas.mpl_connect(
            "figure_enter_event",
            lambda _e: self.canvas.get_tk_widget().focus_set(),
        )

        # ── right: indicator panel ────────────────────────────────────────────
        self._build_indicator_panel(chart_frame)

    def _build_indicator_panel(self, parent: tk.Frame):
        """Right-side panel: search box, overlay/subplot lists, selected chips."""
        panel = tk.Frame(parent, bg=BG_BAR, width=175)
        panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(3, 0))
        panel.pack_propagate(False)

        tk.Label(panel, text="Indicators", bg=BG_BAR, fg=FG,
                 font=("TkDefaultFont", 9, "bold")).pack(pady=(8, 2))

        self._ta_search_var = tk.StringVar()
        self._ta_search_var.trace_add("write", self._filter_indicators)
        tk.Entry(panel, textvariable=self._ta_search_var,
                 bg=BG_EDIT, fg=FG, insertbackground=FG,
                 width=18).pack(padx=6, pady=(0, 6), fill=tk.X)

        # scrollable lists area
        scroll_outer = tk.Frame(panel, bg=BG_BAR)
        scroll_outer.pack(fill=tk.BOTH, expand=True, padx=4)

        list_canvas = tk.Canvas(scroll_outer, bg=BG_BAR, bd=0,
                                highlightthickness=0)
        list_sb = tk.Scrollbar(scroll_outer, orient=tk.VERTICAL,
                               command=list_canvas.yview)
        list_canvas.configure(yscrollcommand=list_sb.set)
        list_sb.pack(side=tk.RIGHT, fill=tk.Y)
        list_canvas.pack(fill=tk.BOTH, expand=True)

        self._ind_list_inner = tk.Frame(list_canvas, bg=BG_BAR)
        list_canvas.create_window((0, 0), window=self._ind_list_inner, anchor="nw")
        self._ind_list_inner.bind(
            "<Configure>",
            lambda e: list_canvas.configure(scrollregion=list_canvas.bbox("all")),
        )
        self._ind_list_canvas = list_canvas

        self._overlay_btn_frame = tk.Frame(self._ind_list_inner, bg=BG_BAR)
        self._subplot_btn_frame = tk.Frame(self._ind_list_inner, bg=BG_BAR)
        self._selected_frame    = tk.Frame(self._ind_list_inner, bg=BG_BAR)

        self._rebuild_indicator_list()

    def _filter_indicators(self, *_):
        self._rebuild_indicator_list()

    def _rebuild_indicator_list(self):
        """Rebuild available indicator buttons filtered by search text."""
        query = self._ta_search_var.get().strip().lower() \
            if hasattr(self, "_ta_search_var") else ""

        # Destroy ALL children (fixes section-header accumulation bug)
        for w in self._ind_list_inner.winfo_children():
            w.destroy()
        self._overlay_btn_frame = tk.Frame(self._ind_list_inner, bg=BG_BAR)
        self._subplot_btn_frame = tk.Frame(self._ind_list_inner, bg=BG_BAR)
        self._selected_frame    = tk.Frame(self._ind_list_inner, bg=BG_BAR)

        def _section(label: str, frame: tk.Frame):
            tk.Label(self._ind_list_inner, text=label, bg=BG_BAR, fg=GREY,
                     font=("TkDefaultFont", 7)).pack(anchor=tk.W, pady=(6, 1))
            frame.pack(fill=tk.X)

        # ── Overlay ────────────────────────────────────────────────────────────
        _section("── Overlay ──", self._overlay_btn_frame)
        for name, cfg in _TA_OVERLAY.items():
            if name not in self._avail_ta_overlay:
                continue
            if query and query not in name.lower():
                continue
            active = name in self._ta_overlay
            btn = tk.Button(
                self._overlay_btn_frame,
                text=f"● {name}",
                bg="#1e2d42" if active else BG_BAR,
                fg=cfg["color"] if active else GREY,
                anchor=tk.W, relief=tk.FLAT, font=("TkDefaultFont", 8),
                command=lambda n=name: self._toggle_ta("overlay", n),
            )
            btn.pack(fill=tk.X, padx=2, pady=1)

        # ── Subplot ────────────────────────────────────────────────────────────
        _section("── Subplot ──", self._subplot_btn_frame)
        for name in _TA_SUBPLOT:
            if name not in self._avail_ta_subplot:
                continue
            if query and query not in name.lower():
                continue
            active = name in self._ta_subplot
            btn = tk.Button(
                self._subplot_btn_frame, text=name,
                bg="#1e2d42" if active else BG_BAR,
                fg=FG if active else GREY,
                anchor=tk.W, relief=tk.FLAT, font=("TkDefaultFont", 8),
                command=lambda n=name: self._toggle_ta("subplot", n),
            )
            btn.pack(fill=tk.X, padx=2, pady=1)

        # ── Active ─────────────────────────────────────────────────────────────
        active_all = self._ta_overlay + self._ta_subplot
        if active_all:
            _section("── Active ──", self._selected_frame)
            for name in active_all:
                kind  = "overlay" if name in self._ta_overlay else "subplot"
                tag   = "[O]" if kind == "overlay" else "[S]"
                color = _TA_OVERLAY[name]["color"] if kind == "overlay" else GOLD
                row = tk.Frame(self._selected_frame, bg="#1e2d42")
                row.pack(fill=tk.X, pady=1, padx=2)
                tk.Label(row, text=f"{tag} {name}", bg="#1e2d42", fg=color,
                         font=("TkDefaultFont", 8), anchor=tk.W).pack(
                             side=tk.LEFT, padx=(4, 0))
                tk.Button(row, text="×", bg="#1e2d42", fg=RED,
                          relief=tk.FLAT, font=("TkDefaultFont", 9),
                          command=lambda n=name, k=kind: self._toggle_ta(k, n),
                          ).pack(side=tk.RIGHT, padx=2)

        self._ind_list_canvas.update_idletasks()
        self._ind_list_canvas.configure(
            scrollregion=self._ind_list_canvas.bbox("all"))

    def _toggle_ta(self, kind: str, name: str):
        lst = self._ta_overlay if kind == "overlay" else self._ta_subplot
        if name in lst:
            lst.remove(name)
        else:
            lst.append(name)
        self._rebuild_indicator_list()
        if self.running and self._klines_data is not None:
            self._refresh_chart()

    def _build_statusbar(self):
        panel = tk.Frame(self, bg="#0d0d1a", height=80)
        panel.pack(fill=tk.X, side=tk.BOTTOM)
        panel.pack_propagate(False)

        sb = tk.Scrollbar(panel, orient=tk.VERTICAL)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        self._log_text = tk.Text(
            panel, height=4, bg="#0d0d1a", fg="#aaaacc",
            font=("Consolas", 8), state=tk.DISABLED,
            wrap=tk.NONE, relief=tk.FLAT, bd=0,
            yscrollcommand=sb.set,
        )
        self._log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0), pady=2)
        sb.config(command=self._log_text.yview)

        self._log("Ready — press Start to connect.")

    def _style_axes(self):
        for ax in (self.ax_c, self.ax_p):
            ax.set_facecolor(BG_DARK)
            ax.tick_params(colors=FG)
            for spine in ax.spines.values():
                spine.set_edgecolor("#444466")
        self.ax_p.set_title("Profile", color=FG)

    def _on_mode_change(self):
        live = self.mode_var.get() == "Live"
        if live:
            self.date_var.set(datetime.now().strftime("%Y-%m-%d"))
        self.refresh_spin.config(state=tk.NORMAL if live else tk.DISABLED)
        self.date_entry.config(state=tk.DISABLED if live else tk.NORMAL)
        if self.running:
            self._stop()
            self._start()

    # ── Crosshair ─────────────────────────────────────────────────────────────

    def _init_crosshair(self):
        """Create crosshair line objects after each chart rebuild."""
        kw = dict(color=CROSS, linewidth=0.7, linestyle="--",
                  alpha=0.65, visible=False, zorder=8, animated=True)
        self._ch_hline_c = self.ax_c.axhline(0, **kw)
        self._ch_vline_c = self.ax_c.axvline(0, **kw)
        self._ch_vlines_sub = [ax.axvline(0, **kw) for ax in self._ta_axes]
        self._ch_hline_t = self.ax_t.axhline(0, **kw)   # tick panel price guide
        self._ch_hline_p = self.ax_p.axhline(0, **kw)   # session profile price guide
        self._ch_label   = self.ax_c.annotate(
            "", xy=(0, 0), xytext=(-72, 0), textcoords="offset points",
            color=GOLD, fontsize=7, va="center", ha="right",
            visible=False, zorder=9, animated=True,
            bbox=dict(fc=BG_TIP, ec="#556688", alpha=0.88, pad=2),
        )

        # One floating readout label per subplot, pinned to the right edge
        self._ch_labels_sub = []
        for ax in self._ta_axes:
            lbl = ax.annotate(
                "", xy=(1, 0), xycoords=("axes fraction", "data"),
                xytext=(4, 0), textcoords="offset points",
                color=GOLD, fontsize=6, va="center", ha="left",
                visible=False, zorder=9, animated=True,
                bbox=dict(fc=BG_TIP, ec="#556688", alpha=0.88, pad=1.5),
            )
            self._ch_labels_sub.append(lbl)

    def _update_crosshair(self, event):
        in_chart = event.inaxes is self.ax_c or event.inaxes in self._ta_axes
        in_profile = event.inaxes in (self.ax_t, self.ax_p)

        if (not in_chart and not in_profile) or event.ydata is None:
            self._hide_crosshair()
            return

        y  = event.ydata
        yd = [y, y]

        # Mirror horizontal price line across all three panels
        if self._ch_hline_t is not None:
            self._ch_hline_t.set_ydata(yd)
            self._ch_hline_t.set_visible(True)
        if self._ch_hline_p is not None:
            self._ch_hline_p.set_ydata(yd)
            self._ch_hline_p.set_visible(True)

        if in_chart:
            # Full crosshair on main chart
            self._ch_hline_c.set_ydata(yd)
            self._ch_hline_c.set_visible(True)
            if event.xdata is not None:
                xd = [event.xdata, event.xdata]
                self._ch_vline_c.set_xdata(xd)
                self._ch_vline_c.set_visible(True)
                for vl in self._ch_vlines_sub:
                    vl.set_xdata(xd)
                    vl.set_visible(True)
            else:
                self._ch_vline_c.set_visible(False)
                for vl in self._ch_vlines_sub:
                    vl.set_visible(False)
            vol_str = self._vol_at_price(y)
            cx = event.xdata if event.xdata is not None else self.ax_c.get_xlim()[1]
            self._ch_label.xy = (cx, y)
            self._ch_label.set_text(f"{y:.2f}{vol_str}")
            self._ch_label.set_visible(True)
            # Update subplot right-edge readouts from stored TA series
            self._update_sub_labels(event.xdata)
        else:
            # Hovering profile panel — hide vertical lines and label
            self._ch_hline_c.set_visible(False)
            self._ch_vline_c.set_visible(False)
            for vl in self._ch_vlines_sub:
                vl.set_visible(False)
            self._ch_label.set_visible(False)
            # Hovering a subplot directly: show its own label at cursor y
            self._update_sub_labels_direct(event)

    def _vol_at_price(self, price: float) -> str:
        """Return a ' | Vol N' string for the nearest profile bin, or ''."""
        if self._profile_ohlcv is not None:
            centers, volumes = self._profile_ohlcv
            if len(centers):
                idx = int(np.argmin(np.abs(centers - price)))
                return f" | Vol {int(volumes[idx]):,}"
        elif self._profile_tick is not None and self._profile_tick:
            prices = sorted(self._profile_tick.keys())
            nearest = min(prices, key=lambda p: abs(p - price))
            d = self._profile_tick[nearest]
            total = d["buy"] + d["sell"] + d["neutral"]
            return f" | Vol {total:,}"
        return ""

    @staticmethod
    def _fmt_sub_val(panel_name: str, val: float) -> str:
        if panel_name in ("Vol", "MAVOL"):
            if abs(val) >= 1e6:
                return f"{val/1e6:.1f}M"
            if abs(val) >= 1e3:
                return f"{val/1e3:.0f}K"
            return str(int(val))
        return f"{val:.2f}"

    def _update_sub_labels(self, xdata):
        """Show subplot right-edge readouts at the TA-series value for bar xdata."""
        if xdata is None:
            for lbl in self._ch_labels_sub:
                lbl.set_visible(False)
            return
        idx    = int(round(xdata))
        panels = self._subplot_panels()
        for i, (lbl, series) in enumerate(zip(self._ch_labels_sub, self._ta_series_sub)):
            if series is None or not (0 <= idx < len(series)):
                lbl.set_visible(False)
                continue
            val = float(series.iloc[idx]) if hasattr(series, "iloc") else float(series[idx])
            if np.isnan(val):
                lbl.set_visible(False)
            else:
                panel_name = panels[i] if i < len(panels) else ""
                lbl.xy = (1, val)
                lbl.set_text(self._fmt_sub_val(panel_name, val))
                lbl.set_visible(True)

    def _update_sub_labels_direct(self, event):
        """When hovering directly over a subplot, show that subplot's y-readout."""
        panels = self._subplot_panels()
        for i, (ax, lbl, series) in enumerate(
                zip(self._ta_axes, self._ch_labels_sub, self._ta_series_sub)):
            if event.inaxes is ax and event.ydata is not None:
                val = event.ydata
                panel_name = panels[i] if i < len(panels) else ""
                lbl.xy = (1, val)
                lbl.set_text(self._fmt_sub_val(panel_name, val))
                lbl.set_visible(True)
            else:
                lbl.set_visible(False)

    def _hide_crosshair(self):
        for obj in (self._ch_hline_c, self._ch_vline_c,
                    self._ch_hline_t, self._ch_hline_p, self._ch_label):
            if obj is not None:
                obj.set_visible(False)
        for vl in self._ch_vlines_sub:
            vl.set_visible(False)
        for lbl in self._ch_labels_sub:
            lbl.set_visible(False)

    def _on_draw(self, event):
        """Save background bitmaps after every full canvas render (for blitting)."""
        try:
            self._bg   = self.canvas.copy_from_bbox(self.fig.bbox)
            self._bg_t = self.canvas.copy_from_bbox(self.ax_t.bbox)
            self._bg_p = self.canvas.copy_from_bbox(self.ax_p.bbox)
        except Exception:
            self._bg = self._bg_t = self._bg_p = None
        # complete any in-flight perf timers
        for op in ("full_render", "scroll_render"):
            if op in self._perf._t0:
                self._perf.end(op)

    def _blit_dynamic(self):
        """Restore background and draw animated crosshair/tooltip artists via blit.

        ax_c artists use the full-figure snapshot + figure-level blit (one pass).
        ax_t / ax_p hlines use their own per-axes snapshots + per-axes blit so
        that fig.draw_artist() clipping issues with side panels don't hide the lines.
        """
        if self._bg is None:
            self.canvas.draw_idle()
            return
        try:
            # ── ax_c panel: full-figure restore + draw + blit ──────────────
            self.canvas.restore_region(self._bg)
            for artist in (self._ch_hline_c, self._ch_vline_c,
                           self._ch_label, self._tip_c, self._tip_p):
                if artist is not None:
                    self.fig.draw_artist(artist)
            for vl in self._ch_vlines_sub:
                self.fig.draw_artist(vl)
            for lbl in self._ch_labels_sub:
                self.fig.draw_artist(lbl)
            self.canvas.blit(self.fig.bbox)

            # ── ax_t panel: per-axes restore + draw + blit ─────────────────
            if self._bg_t is not None and self._ch_hline_t is not None:
                self.canvas.restore_region(self._bg_t)
                self.ax_t.draw_artist(self._ch_hline_t)
                self.canvas.blit(self.ax_t.bbox)

            # ── ax_p panel: per-axes restore + draw + blit ─────────────────
            if self._bg_p is not None and self._ch_hline_p is not None:
                self.canvas.restore_region(self._bg_p)
                self.ax_p.draw_artist(self._ch_hline_p)
                self.canvas.blit(self.ax_p.bbox)

            self.canvas.flush_events()
        except Exception:
            self.canvas.draw_idle()

    # ── Performance overlay ───────────────────────────────────────────────────

    def _toggle_perf(self, _event=None):
        self._perf_visible = not self._perf_visible
        if self._perf_visible:
            self._perf.clear()          # fresh stats when turning on
            self._schedule_perf_refresh()
        self._update_perf_overlay()

    def _schedule_perf_refresh(self):
        """Auto-refresh the perf overlay every second while it is visible."""
        if not self._perf_visible:
            return
        self._update_perf_overlay()
        self.after(1000, self._schedule_perf_refresh)

    def _update_perf_overlay(self):
        """Rebuild the in-chart perf text artist."""
        if self.ax_c is None:
            return
        if self._perf_text is not None:
            try:
                self._perf_text.remove()
            except Exception:
                pass
            self._perf_text = None
        if not self._perf_visible:
            self._bg = self._bg_t = self._bg_p = None
            self.canvas.draw_idle()
            return
        n_artists = sum(len(ax.get_children()) for ax in (self.ax_c, self.ax_p))
        txt = self._perf.summary() + f"\n{'artists':<14} {n_artists}"
        self._perf_text = self.ax_c.text(
            0.99, 0.99, txt,
            transform=self.ax_c.transAxes,
            va="top", ha="right", fontsize=7,
            color="#ffdd44", family="monospace", zorder=20,
            bbox=dict(fc="#0d0d1a", ec="#555577", alpha=0.88, pad=4),
        )
        self._bg = self._bg_t = self._bg_p = None
        self.canvas.draw_idle()

    # ── Scroll zoom (debounced) ───────────────────────────────────────────────

    def _on_scroll(self, event):
        all_axes = {self.ax_c, self.ax_p} | set(self._ta_axes)
        if event.inaxes not in all_axes or self._klines_data is None:
            return
        factor = 0.85 if event.button == "up" else 1.0 / 0.85

        if event.inaxes is self.ax_c and event.xdata is not None:
            xlo, xhi = self.ax_c.get_xlim()
            cx = event.xdata
            self.ax_c.set_xlim(cx - (cx - xlo) * factor,
                               cx + (xhi - cx) * factor)

        if event.ydata is not None:
            ylo, yhi = self.ax_c.get_ylim()
            cy = event.ydata
            self.ax_c.set_ylim(cy - (cy - ylo) * factor,
                               cy + (yhi - cy) * factor)
            # keep full-session profile panel aligned
            if self._tick_shown_idx is None:
                self._sync_profile_ylim()

        # debounce: coalesce rapid scroll events into one redraw after 60 ms
        if self._scroll_job is not None:
            self.after_cancel(self._scroll_job)
        self._scroll_job = self.after(60, self._flush_scroll)

    def _flush_scroll(self):
        self._scroll_job = None
        self._bg = self._bg_t = self._bg_p = None
        self._perf.start("scroll_render")
        self._rebuild_zoomed_profile()
        self.canvas.draw_idle()

    def _on_axes_leave(self, event):
        self._hide_crosshair()
        if self._hovered_candle_idx is not None:
            self._hovered_candle_idx = None
            self._draw_tick_placeholder()
        self._blit_dynamic()

    # ── Hover tooltip ─────────────────────────────────────────────────────────

    def _on_hover(self, event):
        self._perf.start("hover")
        # 1. update crosshair
        if self._ch_hline_c is not None:
            self._update_crosshair(event)

        # 2. per-candle tick panel (ax_t): always update when hovering ax_c
        if self._klines_data is not None and event.inaxes is self.ax_c \
                and event.xdata is not None:
            idx = int(round(event.xdata))
            if 0 <= idx < len(self._klines_data):
                if idx != self._hovered_candle_idx:
                    self._hovered_candle_idx = idx
                    self._update_hover_tick(idx)
            elif self._hovered_candle_idx is not None:
                self._hovered_candle_idx = None
                self._draw_tick_placeholder()

        # 3. hide then selectively show tooltip
        for tip in (self._tip_c, self._tip_p):
            if tip is not None:
                tip.set_visible(False)

        if event.inaxes is self.ax_c and self._klines_data is not None \
                and event.xdata is not None and self._tip_c is not None:
            idx = int(round(event.xdata))
            if 0 <= idx < len(self._klines_data):
                row = self._klines_data.iloc[idx]
                vol = int(row.get("volume", 0) or 0)
                text = (
                    f"{row['time_key']}\n"
                    f"O {row['open']:.2f}   H {row['high']:.2f}\n"
                    f"L {row['low']:.2f}   C {row['close']:.2f}\n"
                    f"Vol {vol:,}"
                )
                self._tip_c.set_text(text)
                # anchor to candle high — flip below only when near chart top
                candle_high = float(row["high"])
                self._tip_c.xy = (idx, candle_high)
                ylo, yhi = self.ax_c.get_ylim()
                frac = (candle_high - ylo) / (yhi - ylo) if yhi > ylo else 0.5
                self._tip_c.set_position((0, -60) if frac > 0.85 else (0, 14))
                self._tip_c.set_visible(True)

        elif event.inaxes is self.ax_p and event.ydata is not None \
                and self._tip_p is not None:
            ylo, yhi = self.ax_p.get_ylim()
            frac = (event.ydata - ylo) / (yhi - ylo) if yhi > ylo else 0.5
            tip_offset = (0, -60) if frac > 0.80 else (0, 14)

            if self._profile_ohlcv is not None:
                centers, volumes = self._profile_ohlcv
                if len(centers) > 1:
                    bin_h = abs(centers[1] - centers[0])
                    idx   = int(np.argmin(np.abs(centers - event.ydata)))
                    if abs(centers[idx] - event.ydata) <= bin_h:
                        self._tip_p.set_text(
                            f"Price  {centers[idx]:.2f}\n"
                            f"Vol    {int(volumes[idx]):,}\n"
                            f"(OHLCV)"
                        )
                        self._tip_p.xy = (event.xdata, event.ydata)
                        self._tip_p.set_position(tip_offset)
                        self._tip_p.set_visible(True)

            elif self._profile_tick is not None and self._profile_tick:
                prices  = sorted(self._profile_tick.keys())
                nearest = min(prices, key=lambda p: abs(p - event.ydata))
                tick_sz = abs(prices[1] - prices[0]) if len(prices) > 1 else 0.5
                if abs(nearest - event.ydata) <= tick_sz:
                    d   = self._profile_tick[nearest]
                    net = d["buy"] - d["sell"]
                    self._tip_p.set_text(
                        f"Price   {nearest:.2f}\n"
                        f"Buy     {d['buy']:,}\n"
                        f"Sell    {d['sell']:,}\n"
                        f"Neutral {d['neutral']:,}\n"
                        f"Net     {net:+,}"
                    )
                    self._tip_p.xy = (event.xdata, event.ydata)
                    self._tip_p.set_position(tip_offset)
                    self._tip_p.set_visible(True)

        self._perf.end("hover")
        self._blit_dynamic()

    # ── Session control ───────────────────────────────────────────────────────

    def _start(self):
        if self.mode_var.get() == "Historical":
            self._start_historical()
        else:
            self._start_live()

    def _start_live(self):
        code = self.code_var.get().strip()
        self._log(f"Connecting to OpenD {self.host_var.get().strip()}:{self.port_var.get()} ...")
        try:
            self.ctx = OpenQuoteContext(host=self.host_var.get().strip(),
                                        port=self.port_var.get())
        except Exception as exc:
            messagebox.showerror("Connection Error", str(exc))
            self._log(f"Connection failed: {exc}")
            return
        self._log(f"Connected. Subscribing {code} TICKER ...")
        with self.tick_lock:
            self.tick_buckets.clear()
        self.ctx.set_handler(self._make_ticker_handler())
        ret, err = self.ctx.subscribe([code], [SubType.TICKER], subscribe_push=True)
        if ret != RET_OK:
            messagebox.showerror("Subscribe Error", str(err))
            self._log(f"Subscribe failed: {err}")
            self.ctx.close()
            self.ctx = None
            return
        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self._log(f"Subscribed {code} — waiting for ticks ...")
        self._schedule_refresh()

    def _start_historical(self):
        date_str = self.date_var.get().strip()
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            messagebox.showerror("Invalid Date", "Use format YYYY-MM-DD")
            return
        self._log(f"Connecting to OpenD {self.host_var.get().strip()}:{self.port_var.get()} ...")
        try:
            self.ctx = OpenQuoteContext(host=self.host_var.get().strip(),
                                        port=self.port_var.get())
        except Exception as exc:
            messagebox.showerror("Connection Error", str(exc))
            self._log(f"Connection failed: {exc}")
            return
        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self._log(f"Connected. Fetching K-lines: {self.code_var.get().strip()} {date_str} ...")
        self._refresh_chart()

    def _stop(self):
        self.running = False
        if self._refresh_job:
            self.after_cancel(self._refresh_job)
            self._refresh_job = None
        if self.ctx:
            try:
                if self.mode_var.get() == "Live":
                    self.ctx.unsubscribe([self.code_var.get().strip()], [SubType.TICKER])
                self.ctx.close()
            except Exception:
                pass
            self.ctx = None
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self._log("Stopped.")

    def _on_close(self):
        self._stop()
        self._perf_visible = False   # prevent _schedule_perf_refresh from re-arming
        self.destroy()
        self.quit()

    # ── Tick handler (live) ───────────────────────────────────────────────────

    def _make_ticker_handler(self):
        app = self

        class Handler(TickerHandlerBase):
            def on_recv_rsp(self, rsp_str):
                ret, data = super().on_recv_rsp(rsp_str)
                if ret != RET_OK or data is None or data.empty:
                    return
                _, cm = TIMEFRAME_MAP[app.tf_var.get()]
                with app.tick_lock:
                    for _, row in data.iterrows():
                        raw = str(row["time"])
                        fmt = "%Y-%m-%d %H:%M:%S.%f" if "." in raw else "%Y-%m-%d %H:%M:%S"
                        t   = datetime.strptime(raw, fmt)
                        bucket = candle_start(t, cm)
                        price  = float(row["price"])
                        vol    = int(row["volume"])
                        d      = str(row["direction"]).upper()
                        key    = "buy" if d == "BUY" else ("sell" if d == "SELL" else "neutral")
                        app.tick_buckets[bucket][price][key] += vol

        return Handler()

    # ── Data fetch ────────────────────────────────────────────────────────────

    def _fetch_klines(self) -> pd.DataFrame | None:
        code  = self.code_var.get().strip()
        num   = self.num_var.get()
        ktype, _ = TIMEFRAME_MAP[self.tf_var.get()]
        if self.mode_var.get() == "Historical":
            date_str = self.date_var.get().strip()
            dt   = datetime.strptime(date_str, "%Y-%m-%d")
            prev = (dt - timedelta(days=1)).strftime("%Y-%m-%d")
            # Extend end by 3 calendar days so the user can pan forward into
            # subsequent sessions after the anchor date.
            end_dt = dt + timedelta(days=3)
            start, end = f"{prev} 20:00:00", f"{end_dt.strftime('%Y-%m-%d')} 23:59:59"

            # Trade Review: expand window to cover pre-entry context + exit date
            tr_active = self._ind.get("trade_review") and self._ind["trade_review"].get()
            if tr_active and self._trade_record:
                cfg = self._trade_record.get("_config", {})
                trend_tf = cfg.get("trend_tf", "60m")
                # Derive HTF candle size so the lookback covers the full 80-bar
                # HTF context window used in _overlay_trade_review.
                _htf_mins = {"1m":1,"5m":5,"15m":15,"30m":30,"60m":60,"1h":60,"4h":240,"1d":1440}
                htf_candle_mins = _htf_mins.get(trend_tf, 60)
                # 80 HTF bars + 30% buffer, converted to calendar days (7 trading hr/day)
                htf_lookback_days = max(14, int(80 * htf_candle_mins / (60 * 7) * 1.5) + 7)
                _, candle_mins = TIMEFRAME_MAP[self.tf_var.get()]
                ltf_lookback_days = max(3, (num * candle_mins) // (60 * 7))
                lookback_days = max(htf_lookback_days, ltf_lookback_days)
                start = (dt - timedelta(days=lookback_days)).strftime("%Y-%m-%d 20:00:00")
                exit_time = str(self._trade_record.get("exit_time") or "")
                if exit_time and len(exit_time) >= 10:
                    exit_date = exit_time[:10]
                    if exit_date > end_dt.strftime("%Y-%m-%d"):
                        end = f"{exit_date} 23:59:59"
        else:
            end   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            start = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d %H:%M:%S")
        ret, df, _ = self.ctx.request_history_kline(
            code, start=start, end=end, ktype=ktype, autype=AuType.NONE,
            max_count=2000, extended_time=True)
        if ret != RET_OK:
            return None
        # Return all fetched bars; _render_chart controls the initial viewport
        # via xlim anchored to date_str, so the user can pan into adjacent sessions.
        self._klines_warmup = df.reset_index(drop=True)
        return df.reset_index(drop=True)

    # ── Chart refresh ─────────────────────────────────────────────────────────

    def _schedule_refresh(self):
        if not self.running:
            return
        self._refresh_chart()
        self._refresh_job = self.after(self.refresh_var.get() * 1000, self._schedule_refresh)

    def _refresh_chart(self):
        if not self.ctx:
            return
        if getattr(self, "_fetching", False):
            return  # previous fetch still in progress
        self._fetching = True

        # snapshot widget state on main thread before handing off
        historical = self.mode_var.get() == "Historical"
        code       = self.code_var.get().strip()
        tf         = self.tf_var.get()
        date_str   = self.date_var.get().strip()
        ind        = {k: v.get() for k, v in self._ind.items()}
        with self.tick_lock:
            live_mem = {k: dict(v) for k, v in self.tick_buckets.items()}

        self._log(f"Fetching K-lines ({tf}, last {self.num_var.get()}) ...")
        threading.Thread(
            target=self._fetch_data_bg,
            args=(historical, code, tf, date_str, ind, live_mem),
            daemon=True,
        ).start()

    def _fetch_data_bg(self, historical, code, tf, date_str, ind, live_mem):
        """Background thread: API + SQLite + SMC — no Tkinter calls allowed."""
        try:
            klines = self._fetch_klines()
            if klines is None or klines.empty:
                self.after(0, self._log, "No K-line data — market may be closed or invalid date")
                return

            _, candle_mins = TIMEFRAME_MAP[tf]

            if historical:
                ticks = self._load_local_ticks(code, date_str, tf) or {}
            else:
                ticks = self._load_local_ticks(code, datetime.now().strftime("%Y-%m-%d"), tf) or {}
                for bk, pd_ in live_mem.items():
                    if bk not in ticks:
                        ticks[bk] = pd_
                    else:
                        for price, counts in pd_.items():
                            if price not in ticks[bk]:
                                ticks[bk][price] = dict(counts)
                            else:
                                for k in ("buy", "sell", "neutral"):
                                    ticks[bk][price][k] += counts[k]

            # Trade Review mode suppresses all-candle SMC overlays — only the
            # entry-specific FVG/BOS are drawn later in _overlay_trade_review().
            trade_review_active = ind.get("trade_review", False)

            # Run SMC detection on the full warmup window so annotations are
            # stable regardless of the user's display candle count.  Signals
            # whose break bar falls outside the display slice are discarded;
            # those whose from_idx is before the slice are clamped to 0 by
            # draw_bos_choch() already.
            warmup   = self._klines_warmup
            if warmup is None or warmup.empty:
                warmup = klines
            disp_off = len(warmup) - len(klines)   # warmup_idx → display_idx offset
            n_disp   = len(klines)

            def _remap(sig: dict) -> dict | None:
                d = sig["idx"] - disp_off
                if not (0 <= d < n_disp):
                    return None
                r = dict(sig)
                r["idx"] = d
                if "from_idx" in sig:
                    r["from_idx"] = sig["from_idx"] - disp_off
                return r

            smc_raw: list[dict] = []
            if not trade_review_active and (ind["bos_choch"] or ind["ob"]):
                smc_raw = detect_bos_choch(warmup, max_span_bars=_BOS_MAX_SPAN.get(tf))

            smc_signals: list[dict] | None = None
            if not trade_review_active and ind["bos_choch"]:
                smc_signals = [r for s in smc_raw if (r := _remap(s)) is not None]

            fvg_gaps: list[dict] = []
            if ind["fvg"] and not trade_review_active:
                for g in detect_fvg(warmup):
                    d = g["idx"] - disp_off
                    if d < n_disp:
                        r = dict(g); r["idx"] = max(d, 0); fvg_gaps.append(r)

            ob_blocks: list[dict] = []
            if ind["ob"] and not trade_review_active:
                for ob in detect_order_blocks(warmup, smc_raw):
                    d = ob["idx"] - disp_off
                    if d < n_disp:
                        r = dict(ob); r["idx"] = max(d, 0); ob_blocks.append(r)

            self.after(0, self._render_chart,
                       klines, ticks, historical, tf, code, date_str,
                       ind, smc_signals, fvg_gaps, ob_blocks, candle_mins)
        except Exception as exc:
            self.after(0, self._log, f"Fetch error: {exc}")
        finally:
            self._fetching = False

    def _render_chart(self, klines, ticks, historical, tf, code, date_str,
                      ind, smc_signals, fvg_gaps, ob_blocks, candle_mins):
        """Main-thread: rebuild the matplotlib figure from pre-fetched data."""
        self._log(f"Got {len(klines)} candles. Building chart ...")

        if historical:
            self._hist_buckets = ticks
            if ticks:
                self._log(f"Local ticks loaded — {len(ticks)} candle buckets found.")
            else:
                self._log("No local tick data for this date — using OHLCV profile.")
        else:
            self._live_buckets = ticks

        # Save current zoom state before clearing (Live mode only: preserve user pan/zoom)
        _saved_xlim: tuple | None = None
        _saved_n: int | None = None
        if not historical and hasattr(self, "ax_c"):
            try:
                _saved_xlim = self.ax_c.get_xlim()
                _saved_n = len(self._klines_data) if hasattr(self, "_klines_data") else None
            except Exception:
                pass

        # rebuild figure — layout depends on active subplot indicators
        self.fig.clear()
        self._profile_hover_band     = None   # axes recreated; old refs are invalid
        self._profile_hover_line     = None
        self._profile_bar_rects      = []     # old axes gone; drop stale refs
        self._profile_axvline        = None
        self._profile_legend         = None
        self._candle_profile_artists = []
        self._profile_centers        = None
        self._profile_total          = None
        panels = self._subplot_panels()   # MAVOL merged into Vol when both active
        n_sub  = len(panels)
        if n_sub == 0:
            gs = self.fig.add_gridspec(1, 3, width_ratios=[4, 0.8, 1], wspace=0)
            self.ax_c = self.fig.add_subplot(gs[0, 0])
            self.ax_t = self.fig.add_subplot(gs[0, 1])   # per-candle tick panel
            self.ax_p = self.fig.add_subplot(gs[0, 2])   # session hybrid profile
            self._ta_axes = []
        else:
            h_ratios = [4] + [1.5] * n_sub
            gs = self.fig.add_gridspec(
                1 + n_sub, 3,
                width_ratios=[4, 0.8, 1], height_ratios=h_ratios,
                wspace=0, hspace=0.08,
            )
            self.ax_c = self.fig.add_subplot(gs[0, 0])
            self.ax_t = self.fig.add_subplot(gs[0, 1])
            self.ax_p = self.fig.add_subplot(gs[0, 2])
            self._ta_axes = [
                self.fig.add_subplot(gs[i + 1, 0], sharex=self.ax_c)
                for i in range(n_sub)
            ]
        self._style_axes()
        for tax in self._ta_axes:
            tax.set_facecolor(BG_DARK)
            tax.tick_params(colors=FG, labelsize=7)
            for sp in tax.spines.values():
                sp.set_edgecolor("#444466")
        for ax_panel in (self.ax_t, self.ax_p):
            ax_panel.set_facecolor(BG_DARK)
            ax_panel.tick_params(colors=FG, labelsize=7)
            for sp in ax_panel.spines.values():
                sp.set_edgecolor("#444466")
            ax_panel.spines["left"].set_visible(False)
            ax_panel.yaxis.set_tick_params(labelleft=False, left=False)
        # ax_t x-axis labels (volume scale) sit at the left edge of the panel and
        # overlap with ax_c's rotated date labels.  They are not meaningful during
        # the placeholder state and are restored per-hover by draw_tick_profile_bars.
        self.ax_t.tick_params(labelbottom=False)
        self.ax_t.set_xlabel("")

        mode_label = f"Historical  {date_str}" if historical else "Live"
        self.fig.suptitle(f"{code}   {tf}   {mode_label}",
                          color=FG, fontsize=13)

        # ── candles ───────────────────────────────────────────────────────────
        labels = draw_candles(self.ax_c, klines)
        n = len(klines)

        # Anchor index: last bar on or before date_str (supports cross-day pan).
        num_view = self.num_var.get()
        if historical and date_str:
            anchor_idx = next(
                (i for i in range(n - 1, -1, -1)
                 if str(klines.iloc[i]["time_key"])[:10] <= date_str),
                n - 1,
            )
        else:
            anchor_idx = n - 1

        # Target ~7 labels in the initial viewport.  ha='right' anchors the text at
        # the tick position so rotated labels extend left (into the chart) rather
        # than right (where they would overlap ax_t's x-axis labels).
        step = max(1, num_view // 7)
        shown = list(range(0, n, step))
        self.ax_c.set_xticks(shown)
        self.ax_c.set_xticklabels(
            [labels[i] for i in shown],
            rotation=45, ha='right', va='top', fontsize=7, color=FG,
        )
        self.ax_c.set_ylabel("Price", color=FG)
        self.ax_c.grid(axis="y", color=GRID, linewidth=0.5)
        self.ax_c.set_xlim(-0.5, n - 0.5)  # overridden below after TA subplots
        self._hovered_candle_idx  = None
        self._tick_shown_idx      = None
        self._tick_data_cache     = {}   # new chart data invalidates all cached profiles
        self._smc_artists = []

        # ── order flow overlays ───────────────────────────────────────────────
        tick_overlay = ticks or None
        if tick_overlay:
            if ind["heatmap"]:
                draw_candle_heatmap(self.ax_c, klines, tick_overlay, candle_mins)
            if ind["delta"]:
                draw_candle_deltas(self.ax_c, klines, tick_overlay, candle_mins)
                price_range = float(klines["high"].max() - klines["low"].min())
                _, yhi = self.ax_c.get_ylim()
                self.ax_c.set_ylim(float(klines["low"].min()) - price_range * 0.08, yhi)

        # ── SMC overlays ─────────────────────────────────────────────────────
        if ind["bos_choch"] and smc_signals:
            # Show only the most recent signals — stale consolidation noise is not actionable.
            disp_smc = smc_signals[-5:] if len(smc_signals) > 5 else smc_signals
            self._smc_artists += draw_bos_choch(self.ax_c, klines, disp_smc)
        if ind["fvg"] and fvg_gaps:
            self._smc_artists += draw_fvg(self.ax_c, klines, fvg_gaps,
                                          self._fvg_width_var.get())
        if ind["ob"] and ob_blocks:
            self._smc_artists += draw_order_blocks(self.ax_c, klines, ob_blocks,
                                                   self._ob_width_var.get())

        # ── technical indicators ──────────────────────────────────────────────
        self._draw_ta_overlays(klines)
        self._draw_ta_subplots(klines)

        # ── right-panel profile ───────────────────────────────────────────────
        if historical:
            self._hist_klines = klines   # stored for zoom-responsive rebuild
            filtered_klines   = self._filter_klines_by_session(klines)
            coverage, poc_price, _vah, _val = self._draw_hist_tick_profile(
                ticks or {}, filtered_klines, candle_mins
            )
            src_label = f"Hybrid profile ({coverage}% tick)  |  Hover candle for single-bar profile"
        else:
            self._hist_klines = None
            poc_price = None
            _vah = _val = None
            src_label = None
            self._draw_live_tick_profile(ticks, klines)

        # ax_t starts empty — placeholder until user hovers a candle
        self._draw_tick_placeholder()

        # fixed margins
        bot = 0.22 if self._ta_subplot else 0.14
        self.fig.subplots_adjust(left=0.08, right=0.99, top=0.92, bottom=bot, wspace=0)

        self._tip_c = make_float_tip(self.ax_c)
        self._tip_p = make_float_tip(self.ax_p)
        self._tip_c.set_animated(True)
        self._tip_p.set_animated(True)
        self._init_crosshair()
        self._klines_data = klines
        self._bg = self._bg_t = self._bg_p = None
        if self._draw_event_cid is None:
            self._draw_event_cid = self.canvas.mpl_connect("draw_event", self._on_draw)

        # Trade Review: draw entry/exit overlays + relevant HTF FVG/BOS
        if self._ind.get("trade_review") and self._ind["trade_review"].get() \
                and self._trade_record is not None:
            self._overlay_trade_review(klines)

        # Enforce ax_c ylim from the VISIBLE bars (anchor viewport), so that bars
        # loaded for cross-day panning don't artificially expand the price range.
        vis_start = max(0, anchor_idx - num_view + 1)
        vis_klines = klines.iloc[vis_start : anchor_idx + 1]
        kl_lo = float(vis_klines["low"].min())  if not vis_klines.empty else float(klines["low"].min())
        kl_hi = float(vis_klines["high"].max()) if not vis_klines.empty else float(klines["high"].max())
        pr    = max(kl_hi - kl_lo, 0.01)
        margin_bot = pr * 0.10 if ind.get("delta") else pr * 0.05
        self.ax_c.set_ylim(kl_lo - margin_bot, kl_hi + pr * 0.05)

        # Draw POC + VA on main chart after ylim is finalised (tracked for zoom rebuild).
        self._draw_poc_va_on_candles(poc_price, _vah, _val, kl_lo, kl_hi, anchor_idx + 0.5)

        # Re-sync profile ylim after main-chart ylim is finalised.
        self._sync_profile_ylim()

        # Apply anchor viewport last — overrides any set_xlim from _draw_ta_subplots.
        # Initial view: num_view bars ending at anchor_idx (the target date's close).
        # User can pan right to see subsequent sessions loaded by _fetch_klines.
        xlim_lo = max(-0.5, anchor_idx - num_view + 0.5)
        self.ax_c.set_xlim(xlim_lo, anchor_idx + 0.5)

        # Live mode: restore user's zoom/pan position after refresh.
        # - If user was at the right edge (following live bars) → advance window to show latest bar.
        # - If user had panned left → shift xlim by the number of new bars added.
        if not historical and _saved_xlim is not None and _saved_n is not None:
            delta_n = n - _saved_n
            prev_lo, prev_hi = _saved_xlim
            prev_width = max(prev_hi - prev_lo, 1.0)
            at_edge = prev_hi >= _saved_n - 1.5  # user was within 1.5 bars of the right edge
            if at_edge:
                new_hi = anchor_idx + 0.5
                new_lo = max(-0.5, new_hi - prev_width)
            else:
                new_lo = max(-0.5, prev_lo + delta_n)
                new_hi = min(n - 0.5, prev_hi + delta_n)
            self.ax_c.set_xlim(new_lo, new_hi)
            # Recompute ylim for the new visible window so price stays in frame.
            vis_s  = max(0, int(new_lo + 0.5))
            vis_e  = min(n - 1, int(new_hi + 0.5))
            vis_kl = klines.iloc[vis_s : vis_e + 1]
            if not vis_kl.empty:
                vlo = float(vis_kl["low"].min())
                vhi = float(vis_kl["high"].max())
                vpr = max(vhi - vlo, 0.01)
                mb  = vpr * 0.10 if ind.get("delta") else vpr * 0.05
                self.ax_c.set_ylim(vlo - mb, vhi + vpr * 0.05)
                self._sync_profile_ylim()

        self._perf.start("full_render")
        self.canvas.draw_idle()

        total_ticks = sum(p[k] for v in ticks.values() if v for p in v.values() if p for k in p)
        if historical:
            self._log(f"Chart ready  |  {n} candles  |  {src_label}")
        else:
            self._log(f"Chart refreshed  |  live ticks: {total_ticks}"
                      f"  |  next refresh in {self.refresh_var.get()}s")

    # ── Technical indicator rendering ─────────────────────────────────────────

    def _draw_ta_overlays(self, klines: pd.DataFrame):
        if not self._ta_overlay:
            return
        close = klines["close"].reset_index(drop=True)
        x = list(range(len(close)))
        for name in self._ta_overlay:
            cfg = _TA_OVERLAY[name]
            if name.startswith("EMA"):
                length = int(name.split()[1])
                series = _ta.trend.EMAIndicator(close, window=length,
                                                fillna=False).ema_indicator()
                self.ax_c.plot(x, series, color=cfg["color"], lw=cfg["lw"],
                               label=name, zorder=3)
            elif name == "Bollinger Bands":
                bb = _ta.volatility.BollingerBands(close, window=20, window_dev=2,
                                                    fillna=False)
                col = cfg["color"]
                self.ax_c.plot(x, bb.bollinger_hband(), color=col,
                               lw=cfg["lw"], linestyle="--", zorder=3)
                self.ax_c.plot(x, bb.bollinger_mavg(),  color=col,
                               lw=cfg["lw"], zorder=3)
                self.ax_c.plot(x, bb.bollinger_lband(), color=col,
                               lw=cfg["lw"], linestyle="--", zorder=3)
                self.ax_c.fill_between(x, bb.bollinger_lband(),
                                       bb.bollinger_hband(),
                                       color=col, alpha=0.05, zorder=2)
            elif name == "KD":
                kd_df = compute_kd(klines, fast=cfg["fast"], slow=cfg["slow"])
                draw_kd(self.ax_c, klines, kd_df)
        if any(n.startswith("EMA") for n in self._ta_overlay):
            self.ax_c.legend(
                loc="upper left", fontsize=7,
                facecolor=BG_DARK, edgecolor="#444466", labelcolor=FG,
            )

    def _subplot_panels(self) -> list[str]:
        """Return subplot indicator names that each get a physical panel row.
        MAVOL is merged into the Vol panel when both are active."""
        has_vol = "Vol" in self._ta_subplot
        return [n for n in self._ta_subplot if not (n == "MAVOL" and has_vol)]

    def _draw_ta_subplots(self, klines: pd.DataFrame):
        if not self._ta_subplot or not self._ta_axes:
            return
        n_display = len(klines)
        x = list(range(n_display))

        # Use the full fetched dataset for warmup so that indicators don't have
        # a flat/missing head due to insufficient lookback.
        warmup = self._klines_warmup
        if warmup is not None and len(warmup) > n_display:
            close_w = warmup["close"].reset_index(drop=True)
            high_w  = warmup["high"].reset_index(drop=True)
            low_w   = warmup["low"].reset_index(drop=True)
        else:
            close_w = klines["close"].reset_index(drop=True)
            high_w  = klines["high"].reset_index(drop=True)
            low_w   = klines["low"].reset_index(drop=True)

        self._ta_series_sub = []   # reset; filled below, one entry per panel

        import matplotlib.ticker as _ticker
        panels = self._subplot_panels()
        n_last = len(panels) - 1

        for i, name in enumerate(panels):
            ax = self._ta_axes[i]
            ax.grid(axis="y", color=GRID, linewidth=0.3)
            # xlim is set by _render_chart after all subplots are drawn (via sharex)

            primary_series = None   # will be stored for crosshair readout

            if name == "MACD":
                n_w = len(close_w)
                if n_w >= 35:
                    fast, slow, sign = 12, 26, 9
                elif n_w >= 15:
                    fast, slow, sign = 5, 12, 5
                else:
                    ax.text(0.5, 0.5, f"Need ≥15 bars\n(have {n_w})",
                            ha="center", va="center", transform=ax.transAxes,
                            color=GREY, fontsize=7)
                    ax.set_ylabel("MACD", color=FG, fontsize=7)
                if n_w >= 15:
                    macd_obj = _ta.trend.MACD(close_w, window_fast=fast,
                                              window_slow=slow, window_sign=sign,
                                              fillna=False)
                    # Slice to the display window — warmup bars are discarded
                    line   = macd_obj.macd().iloc[-n_display:].reset_index(drop=True)
                    signal = macd_obj.macd_signal().iloc[-n_display:].reset_index(drop=True)
                    hist   = macd_obj.macd_diff().iloc[-n_display:].reset_index(drop=True)
                    colors = [GREEN if v >= 0 else RED for v in hist.fillna(0)]
                    ax.bar(x, hist, color=colors, alpha=0.7, width=0.8, zorder=2)
                    ax.plot(x, line,   color="#5c9cf5", lw=0.9, zorder=3)
                    ax.plot(x, signal, color=GOLD,      lw=0.9, zorder=3)
                    ax.axhline(0, color=GREY, lw=0.4)
                    ax.set_ylabel(f"MACD {fast}/{slow}", color=FG, fontsize=7)
                    ax.yaxis.set_major_locator(_ticker.MaxNLocator(nbins=3, symmetric=True))
                    primary_series = line

            elif name == "RSI":
                rsi_full = _ta.momentum.RSIIndicator(close_w, window=14,
                                                     fillna=False).rsi()
                rsi = rsi_full.iloc[-n_display:].reset_index(drop=True)
                ax.plot(x, rsi, color="#ce93d8", lw=0.9)
                ax.axhline(70, color=RED,   lw=0.5, linestyle="--", alpha=0.7)
                ax.axhline(30, color=GREEN, lw=0.5, linestyle="--", alpha=0.7)
                ax.axhline(50, color=GREY,  lw=0.3)
                ax.set_ylim(0, 100)
                ax.set_yticks([30, 50, 70])
                ax.set_ylabel("RSI", color=FG, fontsize=7)
                primary_series = rsi

            elif name == "ATR":
                atr_full = _ta.volatility.AverageTrueRange(
                    high_w, low_w, close_w, window=14, fillna=False).average_true_range()
                atr = atr_full.iloc[-n_display:].reset_index(drop=True)
                ax.plot(x, atr, color=GOLD, lw=0.9)
                ax.set_ylabel("ATR", color=FG, fontsize=7)
                ax.yaxis.set_major_locator(_ticker.MaxNLocator(nbins=3))
                primary_series = atr

            elif name in ("Vol", "MAVOL"):
                # "Vol" = volume bars; "MAVOL" without Vol = MA-vol line only
                vol_display = klines["volume"].reset_index(drop=True)
                if name == "Vol":
                    # Color bars by candle direction
                    opens_d  = klines["open"].reset_index(drop=True)
                    closes_d = klines["close"].reset_index(drop=True)
                    bar_colors = [GREEN if c >= o else RED
                                  for o, c in zip(opens_d, closes_d)]
                    ax.bar(x, vol_display, color=bar_colors, alpha=0.7, width=0.8, zorder=2)
                    ax.set_ylabel("Vol", color=FG, fontsize=7)
                    primary_series = vol_display

                    # MAVOL overlay if both are active (MAVOL merged into this panel)
                    if "MAVOL" in self._ta_subplot:
                        vol_w = (self._klines_warmup["volume"].reset_index(drop=True)
                                 if self._klines_warmup is not None else vol_display)
                        mavol = (pd.Series(vol_w).rolling(20, min_periods=1).mean()
                                 .iloc[-n_display:].reset_index(drop=True))
                        ax.plot(x, mavol, color=GOLD, lw=1.0, zorder=3, label="MA20")

                else:
                    # MAVOL only (Vol panel not active) — just the MA line
                    vol_w = (self._klines_warmup["volume"].reset_index(drop=True)
                             if self._klines_warmup is not None else vol_display)
                    mavol = (pd.Series(vol_w).rolling(20, min_periods=1).mean()
                             .iloc[-n_display:].reset_index(drop=True))
                    ax.plot(x, mavol, color=GOLD, lw=1.0, zorder=3)
                    ax.set_ylabel("MA Vol", color=FG, fontsize=7)
                    primary_series = mavol

                ax.yaxis.set_major_formatter(
                    _ticker.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M" if v >= 1e6
                                          else (f"{v/1e3:.0f}K" if v >= 1e3 else str(int(v)))))
                ax.yaxis.set_major_locator(_ticker.MaxNLocator(nbins=3))

            self._ta_series_sub.append(primary_series)

            # x-axis: hide labels on all but the last subplot
            if i < n_last:
                ax.tick_params(labelbottom=False)
            else:
                ax.tick_params(axis="x", labelrotation=45, labelsize=7, colors=FG)

    # ── Profile drawing ───────────────────────────────────────────────────────

    def _load_local_ticks(self, code: str, date_str: str, tf: str) -> dict | None:
        """Query DuckDB for code+date; return candle buckets or None if unavailable."""
        import pathlib as _pl
        db_path = _pl.Path(__file__).parent.parent / "db" / "ticks.db"
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
                ts = r["ts"] if isinstance(r["ts"], datetime) else datetime.fromisoformat(str(r["ts"]))
                bucket = candle_start(ts, candle_mins)
                key    = {"BUY": "buy", "SELL": "sell"}.get(r["direction"].upper(), "neutral")
                buckets[bucket][r["price"]][key] += r["volume"]
            return dict(buckets)
        except Exception as exc:
            msg = str(exc)
            if "being used by another process" in msg or "Cannot open file" in msg:
                self._log("DB busy (collector is running) — tick overlays unavailable")
            else:
                self._log(f"Tick DB error: {exc}")
            return None

    def _sync_profile_ylim(self):
        """Set ax_p y-range. Uses tight bounds around actual volume when profile
        data is available; falls back to matching ax_c when the panel is empty."""
        if self._profile_centers is not None and self._profile_total is not None:
            nonzero = np.where(self._profile_total > 0)[0]
            if len(nonzero) >= 2:
                lo = float(self._profile_centers[nonzero[0]])
                hi = float(self._profile_centers[nonzero[-1]])
                margin = max((hi - lo) * 0.10, 0.5)
                self.ax_p.set_ylim(lo - margin, hi + margin)
                self.ax_p.grid(axis="y", color=GRID, linewidth=0.3, alpha=0.5)
                return
        self.ax_p.set_ylim(self.ax_c.get_ylim())
        self.ax_p.grid(axis="y", color=GRID, linewidth=0.3, alpha=0.5)

    def _draw_tick_placeholder(self):
        """Reset ax_t to its default 'hover for tick' state."""
        self._clear_tick_artists()
        self.ax_t.text(0.5, 0.5, "Hover\nfor tick",
                       ha="center", va="center", color=GREY, fontsize=8,
                       transform=self.ax_t.transAxes)
        self.ax_t.set_title("Tick", color=FG, fontsize=9)
        self.ax_t.set_xlabel("", color=FG, fontsize=7)
        self.ax_t.tick_params(labelbottom=False)
        self._tick_shown_idx = None

    # ── Session filter ────────────────────────────────────────────────────────

    # Session boundaries in minutes-since-midnight (US ET / exchange local time).
    # "night" wraps midnight: bars with mins >= 1200 OR mins < 240 qualify.
    _SESS_MINUTES: dict[str, tuple[int, int]] = {
        "pre":     (4 * 60,      9 * 60 + 30),   # 04:00–09:30
        "regular": (9 * 60 + 30, 16 * 60),        # 09:30–16:00
        "post":    (16 * 60,     20 * 60),         # 16:00–20:00
        "night":   (20 * 60,     4 * 60),          # 20:00–04:00 (wraps midnight)
    }

    def _filter_klines_by_session(self, klines: pd.DataFrame) -> pd.DataFrame:
        """Return klines filtered to the sessions selected by _sess_vars checkboxes."""
        if not self._sess_vars:
            return klines

        enabled = {k for k, v in self._sess_vars.items() if v.get()}
        if not enabled:
            return klines.iloc[:0].copy()   # all disabled → empty

        def _in_session(row) -> bool:
            try:
                t = datetime.strptime(str(row["time_key"])[:16], "%Y-%m-%d %H:%M")
            except ValueError:
                return True
            mins = t.hour * 60 + t.minute
            for key in enabled:
                lo, hi = self._SESS_MINUTES[key]
                if key == "night":
                    if mins >= lo or mins < hi:
                        return True
                else:
                    if lo <= mins < hi:
                        return True
            return False

        mask = klines.apply(_in_session, axis=1)
        return klines[mask]

    def _on_session_toggle(self):
        """Session checkbox changed — rebuild the profile for the current view."""
        self._rebuild_zoomed_profile()
        self._bg = self._bg_t = self._bg_p = None
        self.canvas.draw_idle()

    # ── POC + VA overlay on the candle axis ───────────────────────────────────

    def _clear_candle_profile_artists(self):
        """Remove POC + VA artists drawn on ax_c."""
        for artist in self._candle_profile_artists:
            try: artist.remove()
            except Exception: pass
        self._candle_profile_artists = []

    def _draw_poc_va_on_candles(
        self,
        poc_price: float | None,
        vah: float | None,
        val: float | None,
        kl_lo: float,
        kl_hi: float,
        x_right: float,
    ) -> None:
        """Draw POC dashed line + VA band on ax_c.  Tracks artists for later removal."""
        self._clear_candle_profile_artists()
        artists: list = []

        if poc_price is not None and kl_lo <= poc_price <= kl_hi:
            ln = self.ax_c.axhline(poc_price, color=GOLD, lw=0.9,
                                   linestyle="--", alpha=0.8, zorder=5)
            txt = self.ax_c.text(
                x_right, poc_price, f" POC {poc_price:.2f}",
                color=GOLD, fontsize=7, va="center", ha="left", zorder=6, clip_on=True,
                bbox=dict(fc=BG_TIP, ec="none", alpha=0.7, pad=1.5),
            )
            artists += [ln, txt]

        if vah is not None and val is not None:
            span = self.ax_c.axhspan(val, vah, color=_VA_COLOR, alpha=0.07, zorder=1)
            artists.append(span)
            for price, label in ((vah, "VAH"), (val, "VAL")):
                if kl_lo <= price <= kl_hi:
                    ln = self.ax_c.axhline(
                        price, color=_VA_COLOR, lw=0.8, linestyle="--", alpha=0.75, zorder=4,
                    )
                    txt = self.ax_c.text(
                        x_right, price, f" {label} {price:.2f}",
                        color=_VA_COLOR, fontsize=7, va="center", ha="left", zorder=6, clip_on=True,
                        bbox=dict(fc=BG_TIP, ec="none", alpha=0.7, pad=1.5),
                    )
                    artists += [ln, txt]

        self._candle_profile_artists = artists

    # ── Zoom-responsive profile rebuild ───────────────────────────────────────

    def _active_sessions_label(self) -> str:
        """Return a short label of active session keys, e.g. 'Reg+Post'."""
        _SHORT = {"regular": "Reg", "pre": "Pre", "post": "Post", "night": "Night"}
        active = [_SHORT[k] for k in ("regular", "pre", "post", "night")
                  if self._sess_vars.get(k) and self._sess_vars[k].get()]
        if len(active) == 4 or not active:
            return ""   # all on (or all off) — no extra label needed
        return "+".join(active)

    def _rebuild_zoomed_profile(self) -> None:
        """Rebuild ax_p profile for the currently visible candle x-range.

        Called from _flush_scroll (after debounce) and _on_session_toggle.
        No-op in live mode or when no klines are stored.
        """
        if self._hist_klines is None or self.mode_var.get() != "Historical":
            return
        if not hasattr(self, "ax_c") or self.ax_c is None:
            return

        xlo, xhi = self.ax_c.get_xlim()
        lo_idx = max(0, int(np.floor(xlo + 0.5)))
        hi_idx = min(len(self._hist_klines) - 1, int(np.floor(xhi + 0.5)))
        if hi_idx < lo_idx:
            return

        visible  = self._hist_klines.iloc[lo_idx : hi_idx + 1]
        filtered = self._filter_klines_by_session(visible)

        # Always reset ax_p before drawing so no old patches survive.
        self._reset_profile_panel()

        if filtered.empty:
            self.ax_p.set_title("Vol Profile\n(no session data)", color=FG, fontsize=9)
            return

        _, cm = TIMEFRAME_MAP[self.tf_var.get()]
        buckets = self._hist_buckets or {}
        enabled = {k for k, v in self._sess_vars.items() if v.get()}
        tick_only = (enabled == {"regular"})
        result  = build_hybrid_profile(filtered, buckets, cm, n_bins=40, tick_only=tick_only)

        if result is None:
            self.ax_p.set_title("Vol Profile\ninsufficient range", color=FG, fontsize=9)
            return

        centers, buy_v, sell_v, neutral_v, ohlcv_v, coverage = result
        self._profile_centers = centers
        self._profile_total   = buy_v + sell_v + neutral_v + ohlcv_v
        date_label = self.date_var.get().strip()
        rects, vl, leg, poc_price, vah, val = draw_hybrid_profile(
            self.ax_p, centers, buy_v, sell_v, neutral_v, ohlcv_v, coverage,
            date_label=date_label,
        )
        self._profile_bar_rects = rects
        self._profile_axvline   = vl
        self._profile_legend    = leg

        # Override title to show active session filter (visual confirmation of rebuild).
        sess_lbl = self._active_sessions_label()
        if sess_lbl:
            self.ax_p.set_title(
                f"Vol Profile ({coverage}% tick)\n{date_label}  [{sess_lbl}]",
                color=FG, fontsize=9,
            )

        self._sync_profile_ylim()

        kl_lo, kl_hi = self.ax_c.get_ylim()
        self._draw_poc_va_on_candles(poc_price, vah, val, kl_lo, kl_hi, xhi)

    def _draw_hist_tick_profile(
        self, buckets: dict, klines: pd.DataFrame, candle_mins: int
    ) -> tuple:
        """Draw hybrid profile (tick + OHLCV normal estimate).

        Returns (coverage_pct, poc_price, vah, val).
        VA lines on ax_c must be drawn by the caller after ylim is finalised.
        """
        self._profile_ohlcv = None
        self._profile_tick  = None

        _, cm = TIMEFRAME_MAP[self.tf_var.get()]
        enabled = {k for k, v in self._sess_vars.items() if v.get()}
        result = build_hybrid_profile(klines, buckets, cm, n_bins=40,
                                      tick_only=(enabled == {"regular"}))
        if result is None:
            self.ax_p.set_title("Vol Profile\ninsufficient range", color=FG, fontsize=9)
            return 0, None, None, None

        centers, buy_v, sell_v, neutral_v, ohlcv_v, coverage = result
        self._profile_centers = centers
        self._profile_total   = buy_v + sell_v + neutral_v + ohlcv_v
        date_label = self.date_var.get().strip()
        rects, vl, leg, poc_price, vah, val = draw_hybrid_profile(
            self.ax_p,
            centers, buy_v, sell_v, neutral_v, ohlcv_v, coverage,
            date_label=date_label,
        )
        self._profile_bar_rects    = rects
        self._profile_axvline      = vl
        self._profile_legend       = leg
        self._sync_profile_ylim()
        return coverage, poc_price, vah, val

    def _draw_live_tick_profile(self, buckets: dict, klines: pd.DataFrame):
        self._profile_ohlcv     = None
        self._profile_bar_rects = []
        self._profile_axvline   = None
        self._profile_legend    = None
        if not buckets:
            self._profile_tick = None
            self.ax_p.text(0.5, 0.5, "Waiting for\ntick data",
                           ha="center", va="center", color="#aaaaaa", fontsize=9,
                           transform=self.ax_p.transAxes)
            self.ax_p.set_title("Vol Profile", color=FG, fontsize=10)
            return
        # Aggregate ALL candle buckets into one session-level profile
        agg = aggregate_buckets(buckets)
        self._profile_tick = agg
        kl_lo = float(klines["low"].min())  * 0.997
        kl_hi = float(klines["high"].max()) * 1.003
        prices, buy_v, sell_v, neu_v = prices_arrays(agg, lo=kl_lo, hi=kl_hi)
        latest = max(buckets.keys())
        title  = f"Vol Profile\n{latest.strftime('%Y-%m-%d')}"
        rects, vl, leg = draw_tick_profile_bars(
            self.ax_p, prices, buy_v, sell_v, neu_v, title,
            lo=kl_lo, hi=kl_hi, max_bins=40)
        self._profile_bar_rects    = rects
        self._profile_axvline      = vl
        self._profile_legend       = leg
        self._sync_profile_ylim()

    # ── Session profile artists (ax_p) ────────────────────────────────────────

    def _reset_profile_panel(self) -> None:
        """Clear ax_p completely (cla) and recreate the crosshair line.

        More reliable than removing individual patches: cla() removes everything
        including any lingering artists that slipped through the tracking lists.
        Must be called before redrawing the session profile.
        """
        self.ax_p.cla()
        # Recreate the profile crosshair line that cla() just destroyed.
        kw = dict(color=CROSS, linewidth=0.7, linestyle="--",
                  alpha=0.65, visible=False, zorder=8, animated=True)
        self._ch_hline_p = self.ax_p.axhline(0, **kw)
        # Clear stored profile data so _sync_profile_ylim falls back to ax_c range.
        self._profile_centers = None
        self._profile_total   = None
        self._sync_profile_ylim()
        # Invalidate the per-axes blit cache (old bbox snapshot is now stale).
        self._bg_p = None
        # Reset tracking lists — all old references are now dead.
        self._profile_bar_rects = []
        self._profile_axvline   = None
        self._profile_legend    = None
        self._clear_candle_profile_artists()

    def _clear_profile_artists(self):
        """Remove profile artists from ax_p.  Prefer _reset_profile_panel() for
        dynamic rebuilds; this is kept for the live-mode path."""
        for patch in self._profile_bar_rects:
            try: patch.remove()
            except Exception: pass
        self._profile_bar_rects = []
        try:
            self.ax_p.containers.clear()
        except Exception:
            pass
        for attr in ("_profile_axvline", "_profile_legend"):
            obj = getattr(self, attr)
            if obj is not None:
                try: obj.remove()
                except Exception: pass
                setattr(self, attr, None)
        self._clear_candle_profile_artists()

    # ── Per-candle tick panel (ax_t) ──────────────────────────────────────────

    def _clear_tick_artists(self):
        for patch in self._tick_bar_rects:
            try: patch.remove()
            except Exception: pass
        self._tick_bar_rects = []
        self.ax_t.containers.clear()
        for attr in ("_tick_axvline", "_tick_legend"):
            obj = getattr(self, attr, None)
            if obj is not None:
                try: obj.remove()
                except Exception: pass
                setattr(self, attr, None)
        # Remove any text labels (placeholder "Hover for tick" / "No tick data")
        for txt in list(self.ax_t.texts):
            try: txt.remove()
            except Exception: pass

    def _update_hover_tick(self, candle_idx: int):
        """Update ax_t to show per-candle tick distribution for candle_idx."""
        if candle_idx == self._tick_shown_idx:
            return

        mode    = self.mode_var.get()
        buckets = self._live_buckets if mode == "Live" else self._hist_buckets
        if self._klines_data is None or not (0 <= candle_idx < len(self._klines_data)):
            return

        _, candle_mins = TIMEFRAME_MAP[self.tf_var.get()]
        tk_str = str(self._klines_data.iloc[candle_idx]["time_key"])
        try:
            bar_end    = datetime.strptime(tk_str[:16], "%Y-%m-%d %H:%M")
            bucket_key = candle_start(bar_end - timedelta(minutes=candle_mins), candle_mins)
        except ValueError:
            return

        self._clear_tick_artists()

        # No tick data at all for this session → show placeholder
        if not buckets:
            self._draw_tick_placeholder()
            return

        is_live_latest = (mode == "Live" and candle_idx == len(self._klines_data) - 1)
        cached = None if is_live_latest else self._tick_data_cache.get(candle_idx)

        pd_ = buckets.get(bucket_key) if cached is None else None

        if cached is None and not pd_:
            # This candle has no tick data
            self.ax_t.text(0.5, 0.5, "No tick\ndata",
                           ha="center", va="center", color=GREY, fontsize=8,
                           transform=self.ax_t.transAxes)
            self.ax_t.set_title(f"Tick\n{tk_str[5:16]}", color=FG, fontsize=9)
            row_no_tick = self._klines_data.iloc[candle_idx]
            _clo = float(row_no_tick["low"]);  _chi = float(row_no_tick["high"])
            _mg  = max((_chi - _clo) * 0.25, 0.01)
            self.ax_t.set_ylim(_clo - _mg, _chi + _mg)
            self.ax_t.grid(axis="y", color=GRID, linewidth=0.3, alpha=0.5)
            self._tick_shown_idx = candle_idx
            self._bg = self._bg_t = self._bg_p = None
            self.canvas.draw_idle()
            return

        if cached is not None:
            prices, buy_v, sell_v, neu_v, title = cached
        else:
            kl_lo = float(self._klines_data["low"].min())  * 0.997
            kl_hi = float(self._klines_data["high"].max()) * 1.003
            prices, buy_v, sell_v, neu_v = prices_arrays(pd_, lo=kl_lo, hi=kl_hi)
            delta     = int(buy_v.sum() - sell_v.sum())
            delta_str = f"+{delta:,}" if delta >= 0 else f"{delta:,}"
            title = f"Tick  Δ={delta_str}\n{tk_str[5:16]}"
            if not is_live_latest:
                self._tick_data_cache[candle_idx] = (prices, buy_v, sell_v, neu_v, title)

        row  = self._klines_data.iloc[candle_idx]
        c_lo = float(row["low"])
        c_hi = float(row["high"])
        rects, vl, leg = draw_tick_profile_bars(
            self.ax_t, prices, buy_v, sell_v, neu_v, title=title,
            lo=c_lo, hi=c_hi, max_bins=30)
        self._tick_bar_rects = rects
        self._tick_axvline   = vl
        self._tick_legend    = leg
        # Adaptive ylim: fit the candle's own price range so bars fill the panel.
        margin = max((c_hi - c_lo) * 0.25, 0.01)
        self.ax_t.set_ylim(c_lo - margin, c_hi + margin)
        self.ax_t.grid(axis="y", color=GRID, linewidth=0.3, alpha=0.5)

        self._tick_shown_idx = candle_idx
        self._bg = self._bg_t = self._bg_p = None
        self.canvas.draw_idle()

    # ── Trade Review ──────────────────────────────────────────────────────────

    def _load_trade_by_id(self) -> None:
        """Look up a trade_id in the backtest/review DBs and auto-populate the toolbar."""
        trade_id = self._trade_id_var.get().strip()
        if not trade_id:
            return

        row = None
        source = None

        # Check live_trades and backtest trades (read_only avoids blocking the grid).
        try:
            from backtest.db import BacktestDB
            with BacktestDB(read_only=True) as db:
                row = db.fetch_live_trade(trade_id)
                source = "live_trades"
                if row is None:
                    row = db.fetch_trade(trade_id)
                    source = "backtest"
        except Exception:
            pass  # DB locked or missing — continue to review DB

        # Check review_trades (separate file, never blocked by grid).
        if row is None:
            try:
                from backtest.db import ReviewTradesDB
                with ReviewTradesDB(read_only=True) as rdb:
                    row = rdb.fetch_trade(trade_id)
                    source = "review"
            except FileNotFoundError:
                pass  # review DB doesn't exist yet
            except Exception as exc:
                messagebox.showerror("DB Error", str(exc))
                return

        if row is None:
            messagebox.showerror(
                "Not found",
                f"Trade ID not found in any DB:\n{trade_id}\n\n"
                "If this is an audit trade, re-run audit.py to register it.",
            )
            return

        # Extract config to get entry_tf
        config: dict = {}
        raw_cfg = row.get("config_json") or row.get("signal_params")
        if raw_cfg:
            config = json.loads(raw_cfg) if isinstance(raw_cfg, str) else raw_cfg

        # Use HTF (trend_tf) as the display timeframe in review mode —
        # the FVG/BOS context is defined on HTF, matching the audit report view.
        tf_alias = {"60m": "1h"}
        trend_tf  = config.get("trend_tf", "")
        tf        = tf_alias.get(trend_tf, trend_tf)
        if tf not in TIMEFRAME_MAP:
            entry_tf = config.get("entry_tf", "15m")
            tf       = tf_alias.get(entry_tf, entry_tf)
        if tf not in TIMEFRAME_MAP:
            tf = "15m"

        # Date from entry_time — must come from DB, not today's date
        entry_str = str(row.get("entry_time") or "")
        candidate = entry_str[:10] if len(entry_str) >= 10 else ""
        try:
            datetime.strptime(candidate, "%Y-%m-%d")
            date_str = candidate
        except ValueError:
            self._log(f"Warning: entry_time '{entry_str}' could not be parsed; falling back")
            date_str = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")

        # Populate toolbar fields
        self.code_var.set(row.get("symbol", ""))
        self.tf_var.set(tf)
        self.mode_var.set("Historical")
        self.date_var.set(date_str)
        self._trade_record = {**row, "_source": source, "_config": config}

        # Turn on Trade Review mode
        self._ind["trade_review"].set(True)
        self._save_indicator_cfg()
        self._on_mode_change()
        self._log(f"Loaded trade {trade_id[:8]}… ({source})  →  {row.get('symbol')} {tf} {date_str}")

        # Auto-start
        self._start()

    def _clear_trade_overlays(self) -> None:
        for artist in self._trade_overlay_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._trade_overlay_artists.clear()

    def _overlay_trade_review(self, klines: pd.DataFrame) -> None:
        """Draw entry/exit markers, SL/TP lines, and the relevant HTF FVG + BOS.

        Only called when Trade Review mode is active and _trade_record is set.
        All artifacts are stored in _trade_overlay_artists for surgical removal.
        """
        self._clear_trade_overlays()
        trade  = self._trade_record
        config = trade.get("_config", {})

        direction   = str(trade.get("direction", "")).upper()   # 'LONG'|'SHORT'|'bull'|'bear'
        is_bull     = direction in ("LONG", "BULL")
        entry_price = float(trade.get("entry_price") or 0)
        exit_price  = float(trade.get("exit_price")  or 0)
        sl_price    = float(trade.get("sl_price")    or 0)
        tp_price    = float(trade.get("tp_price")    or 0)
        entry_time  = str(trade.get("entry_time") or "")
        exit_time   = str(trade.get("exit_time")  or "")
        result      = str(trade.get("result") or "")

        times = klines["time_key"].astype(str).values

        def _bar_idx(ts: str) -> int | None:
            """Return the kline bar index nearest to ts, or None."""
            if not ts:
                return None
            import numpy as _np
            pos = int(_np.searchsorted(times, ts[:16], side="left"))
            return min(pos, len(klines) - 1)

        entry_idx = _bar_idx(entry_time)
        exit_idx  = _bar_idx(exit_time)

        artists = self._trade_overlay_artists

        # ── SL / TP horizontal lines ──────────────────────────────────────
        if sl_price:
            line = self.ax_c.axhline(
                sl_price, color=RED, linewidth=1.2, linestyle="--",
                alpha=0.85, zorder=4, label=f"SL {sl_price:.2f}",
            )
            artists.append(line)
            txt = self.ax_c.text(
                len(klines) - 1, sl_price, f" SL {sl_price:.2f}",
                color=RED, fontsize=7, va="bottom", ha="right", zorder=5,
            )
            artists.append(txt)

        if tp_price:
            line = self.ax_c.axhline(
                tp_price, color=GREEN, linewidth=1.2, linestyle="--",
                alpha=0.85, zorder=4, label=f"TP {tp_price:.2f}",
            )
            artists.append(line)
            txt = self.ax_c.text(
                len(klines) - 1, tp_price, f" TP {tp_price:.2f}",
                color=GREEN, fontsize=7, va="top", ha="right", zorder=5,
            )
            artists.append(txt)

        # ── Entry arrow ───────────────────────────────────────────────────
        if entry_idx is not None and entry_price:
            arrow_color = GREEN if is_bull else RED
            bar = klines.iloc[entry_idx]
            # Place tail below/above the candle wick
            if is_bull:
                tail_y = float(bar["low"])  * 0.9985
                dy     = entry_price - tail_y
            else:
                tail_y = float(bar["high"]) * 1.0015
                dy     = entry_price - tail_y
            ann = self.ax_c.annotate(
                "", xy=(entry_idx, entry_price),
                xytext=(entry_idx, tail_y),
                arrowprops=dict(arrowstyle="-|>", color=arrow_color,
                                lw=1.8, mutation_scale=14),
                zorder=6,
            )
            artists.append(ann)
            lbl = self.ax_c.text(
                entry_idx, entry_price,
                f"  {'▲' if is_bull else '▼'} {entry_price:.2f}",
                color=arrow_color, fontsize=7, va="center", zorder=7,
            )
            artists.append(lbl)

        # ── Exit marker ───────────────────────────────────────────────────
        if exit_idx is not None and exit_price:
            is_win   = result == "win"
            exc_col  = GREEN if is_win else (RED if result == "loss" else GREY)
            marker   = "o" if is_win else ("x" if result == "loss" else "s")
            pt, = self.ax_c.plot(
                exit_idx, exit_price, marker=marker, markersize=9,
                color=exc_col, zorder=6, linestyle="None",
            )
            artists.append(pt)
            lbl = self.ax_c.text(
                exit_idx, exit_price,
                f"  {'✓' if is_win else '✕'} {exit_price:.2f}",
                color=exc_col, fontsize=7, va="center", zorder=7,
            )
            artists.append(lbl)

        # ── HTF FVG + BOS/CHoCH + OB context ────────────────────────────
        symbol   = str(trade.get("symbol", ""))
        trend_tf = config.get("trend_tf", "")
        if symbol and trend_tf:
            try:
                from feeds.fetcher import fetch_klines as _fetch
                from strategy.smc.fvg import detect_fvg
                from strategy.smc.market_structure import detect_bos_choch
                from strategy.smc.order_blocks import detect_order_blocks
                from matplotlib.patches import Rectangle as _MplRect
                import numpy as _np

                from datetime import datetime as _dt, timedelta as _td
                _entry_dt  = _dt.fromisoformat(entry_time[:19])
                _htf_start = (_entry_dt - _td(days=180)).strftime("%Y-%m-%d")
                _htf_end   = (_entry_dt + _td(days=1)).strftime("%Y-%m-%d")
                htf_df = _fetch(code=symbol, ktype=trend_tf,
                                start=_htf_start, end=_htf_end)
                if not htf_df.empty:
                    htf_times      = htf_df["time_key"].astype(str).values
                    htf_pos        = int(_np.searchsorted(htf_times, entry_time[:16], side="right")) - 1
                    htf_slice      = htf_df.iloc[max(0, htf_pos - 80): htf_pos + 1].reset_index(drop=True)
                    htf_slice_times = htf_slice["time_key"].astype(str).values
                    n_ltf          = max(len(klines) - 1, 1)

                    def _htf_to_ltf(htf_rel: int) -> int:
                        """htf_slice relative index → nearest LTF klines bar index."""
                        if 0 <= htf_rel < len(htf_slice_times):
                            return _bar_idx(htf_slice_times[htf_rel]) or 0
                        return 0

                    show_fvg = self._ind.get("fvg") and self._ind["fvg"].get()
                    show_bos = self._ind.get("bos_choch") and self._ind["bos_choch"].get()
                    show_ob  = self._ind.get("ob") and self._ind["ob"].get()

                    # ── BOS / CHoCH: detect first (OB needs it) ────────────
                    # No max_span_bars here — _BOS_MAX_SPAN is for the LTF
                    # display chart; on an 80-bar HTF slice it kills all signals.
                    all_bos = detect_bos_choch(htf_slice)
                    bos_dir   = "bull" if is_bull else "bear"
                    rel_bos   = [b for b in all_bos if b.get("direction") == bos_dir][-2:]
                    self._log(f"TR overlay: HTF slice {len(htf_slice)} bars, "
                              f"all_bos={len(all_bos)}, rel_bos={len(rel_bos)}")

                    # ── FVG: zone that triggered the entry ────────────────
                    # Prefer exact FVG stored in DB at trade creation time.
                    # Fall back to proximity search for trades logged before
                    # fvg_top/fvg_bottom columns were added.
                    fvg_dir    = "bull" if is_bull else "bear"
                    fvg_width  = config.get("fvg_min_width_pct", 0.002)
                    fvgs       = detect_fvg(htf_slice, fvg_width)
                    stored_top = trade.get("fvg_top")
                    stored_bot = trade.get("fvg_bottom")
                    if stored_top and stored_bot:
                        stored_top = float(stored_top)
                        stored_bot = float(stored_bot)
                        # Match back to a detected FVG to get the correct idx
                        # (formation bar), falling back to the entry bar.
                        matched = next(
                            (f for f in fvgs
                             if f["direction"] == fvg_dir
                             and abs(f["top"]    - stored_top) < 1e-6
                             and abs(f["bottom"] - stored_bot) < 1e-6),
                            None,
                        )
                        entry_fvg = matched or {
                            "direction": fvg_dir,
                            "top":       stored_top,
                            "bottom":    stored_bot,
                            "idx":       len(htf_slice) - 1,
                        }
                        self._log(f"TR overlay: stored FVG {stored_bot:.4f}-{stored_top:.4f}, "
                                  f"idx={'matched' if matched else 'fallback'}")
                    else:
                        _candidates = [
                            f for f in fvgs
                            if f["direction"] == fvg_dir
                            and abs((f["top"] + f["bottom"]) / 2 - entry_price)
                                / max(entry_price, 1e-9) < 0.10
                        ]
                        entry_fvg = (
                            min(_candidates,
                                key=lambda f: abs((f["top"] + f["bottom"]) / 2 - entry_price))
                            if _candidates else None
                        )
                        self._log(f"TR overlay: fvgs={len(fvgs)}, entry_fvg={'found' if entry_fvg else 'none'}, show_fvg={show_fvg}")
                    if show_fvg and entry_fvg:
                        # Cap formation bar at entry so the band never starts
                        # after the entry point.
                        _e = entry_idx if entry_idx is not None else n_ltf
                        fvg_ltf = min(_htf_to_ltf(entry_fvg["idx"]), _e)
                        fvg_color = GREEN if is_bull else RED
                        fvg_w = len(klines) - fvg_ltf
                        if fvg_w > 0:
                            # Rectangle in data coords: left edge at fvg_ltf
                            # (same x as the text label, no -0.5 shift).
                            rect = _MplRect(
                                (fvg_ltf, entry_fvg["bottom"]),
                                fvg_w,
                                entry_fvg["top"] - entry_fvg["bottom"],
                                facecolor=fvg_color, edgecolor="none",
                                alpha=0.18, zorder=2,
                            )
                            self.ax_c.add_patch(rect)
                            artists.append(rect)
                        lbl = self.ax_c.text(
                            fvg_ltf, (entry_fvg["bottom"] + entry_fvg["top"]) / 2,
                            " FVG", color=fvg_color,
                            fontsize=7, va="center", alpha=0.85, zorder=5,
                        )
                        artists.append(lbl)

                    # ── OB: detect now (always log, draw only when show_ob) ──
                    htf_obs = detect_order_blocks(htf_slice, all_bos)
                    rel_obs = [ob for ob in htf_obs
                               if ob.get("direction") == bos_dir][-3:]
                    self._log(f"TR overlay: htf_obs={len(htf_obs)}, "
                              f"rel_obs={len(rel_obs)} "
                              f"subtypes={[o.get('subtype') for o in rel_obs]}")

                    if show_bos:
                        # Re-index signals from HTF slice coords → LTF klines coords,
                        # then delegate to draw_bos_choch for the canonical visual style
                        # (floating line + vertical dotted ticks, same as live/historic).
                        ltf_signals = []
                        for bos in rel_bos:
                            from_ltf  = _htf_to_ltf(bos.get("from_idx", 0))
                            break_ltf = _htf_to_ltf(bos.get("idx", 0))
                            self._log(f"TR overlay: {bos.get('type')} price={bos.get('price',0):.2f} "
                                      f"from_ltf={from_ltf} break_ltf={break_ltf}")
                            if from_ltf == break_ltf:
                                continue  # zero-length — skip
                            # Skip signals whose price level is outside the
                            # current y-axis range — they would be invisible
                            # and confuse the auto-scaling.
                            ylo, yhi = self.ax_c.get_ylim()
                            bos_p = float(bos.get("price", 0))
                            if bos_p and not (ylo <= bos_p <= yhi):
                                self._log(f"TR overlay: {bos.get('type')} @ {bos_p:.2f} "
                                          f"outside ylim [{ylo:.2f}, {yhi:.2f}] — skipped")
                                continue
                            ltf_signals.append({
                                "type":      bos.get("type", "BOS"),
                                "direction": bos.get("direction", bos_dir),
                                "idx":       break_ltf,
                                "from_idx":  from_ltf,
                                "price":     float(bos.get("price", 0)),
                            })
                        artists.extend(draw_bos_choch(self.ax_c, klines, ltf_signals))

                    if show_ob:
                        for ob in rel_obs:
                            ob_ltf = _htf_to_ltf(ob.get("idx", 0))
                            ob_w   = len(klines) - ob_ltf
                            if ob_w <= 0:
                                continue
                            sub      = ob.get("subtype", "regular")
                            ob_alpha = 0.06 if sub == "breaker" else 0.16
                            rect = _MplRect(
                                (ob_ltf, ob["bottom"]),
                                ob_w,
                                ob["top"] - ob["bottom"],
                                facecolor=GOLD, edgecolor="none",
                                alpha=ob_alpha, zorder=2,
                            )
                            self.ax_c.add_patch(rect)
                            artists.append(rect)
                            lbl = self.ax_c.text(
                                ob_ltf,
                                (ob["bottom"] + ob["top"]) / 2,
                                f" OB{'!' if sub == 'breaker' else ''}",
                                color=GOLD, fontsize=7,
                                va="center", ha="left", alpha=0.90, zorder=5,
                                bbox=dict(fc=BG_TIP, ec=GOLD, alpha=0.70,
                                          pad=2, linewidth=0.6),
                            )
                            artists.append(lbl)

            except Exception as exc:
                self._log(f"Trade Review overlay warning: {exc}")

        # Invalidate background cache so the next draw_idle() (called by
        # _render_chart) picks up the new artists.
        self._bg = self._bg_t = self._bg_p = None

    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"{ts}  {msg}\n"
        self._log_text.config(state=tk.NORMAL)
        self._log_text.insert(tk.END, line)
        self._log_text.see(tk.END)
        self._log_text.config(state=tk.DISABLED)

    def _set_status(self, msg: str):
        self._log(msg)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="trade_viewer",
        description="Trade viewer: candlestick + order flow + backtest/live trade review.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # open GUI with defaults
  uv run analysis/orderflow.py

  # open GUI pre-filled for AAPL 5-min historical
  uv run analysis/orderflow.py --code US.AAPL --tf 5m --mode Historical --date 2026-05-15

  # headless: save PNG directly (no window)
  uv run analysis/orderflow.py --code US.SNDK --date 2026-05-15 --output sndk_15m.png
  uv run analysis/orderflow.py --code US.AAPL --tf 5m --num 30 --date 2026-05-15 \
      --output out/aapl.png
        """,
    )
    p.add_argument("--code",    default="US.SNDK",    help="stock code  (default: US.SNDK)")
    p.add_argument("--tf",      default="15m",         choices=list(TIMEFRAME_MAP),
                   help="timeframe  (default: 15m)")
    p.add_argument("--num",     default=26,  type=int, help="number of candles  (default: 26)")
    p.add_argument("--mode",    default="Live",        choices=["Live", "Historical"],
                   help="Live or Historical  (default: Live)")
    p.add_argument("--date",    default=None,          metavar="YYYY-MM-DD",
                   help="date for Historical mode  (default: 3 days ago)")
    p.add_argument("--refresh", default=15,  type=int, help="live refresh interval in s  (default: 15)")
    p.add_argument("--host",    default="127.0.0.1",   help="OpenD host  (default: 127.0.0.1)")
    p.add_argument("--port",    default=11111, type=int, help="OpenD port  (default: 11111)")
    p.add_argument("--output",   default=None,  metavar="FILE.png",
                   help="save chart to file and exit without opening GUI (implies Historical)")
    p.add_argument("--trade-id", default=None,  metavar="UUID",
                   help="jump directly to a trade: auto-fills code/tf/date and enables Trade Review")
    return p.parse_args(argv)


def _load_tick_buckets(code: str, date_str: str, tf: str) -> dict | None:
    """Return candle tick buckets from local DuckDB, or None if unavailable."""
    try:
        db_path = pathlib.Path(__file__).parent.parent / "db" / "ticks.db"
        if not db_path.exists():
            return None
        from feeds.tick_store import TickStore
        day            = datetime.strptime(date_str, "%Y-%m-%d").date()
        _, candle_mins = TIMEFRAME_MAP[tf]
        with TickStore(db_path) as store:
            rows = store.query_date(code, day)
        if not rows:
            return None
        buckets: dict = defaultdict(
            lambda: defaultdict(lambda: {"buy": 0, "sell": 0, "neutral": 0})
        )
        for r in rows:
            ts     = r["ts"] if isinstance(r["ts"], datetime) else datetime.fromisoformat(str(r["ts"]))
            bucket = candle_start(ts, candle_mins)
            key    = {"BUY": "buy", "SELL": "sell"}.get(r["direction"].upper(), "neutral")
            buckets[bucket][r["price"]][key] += r["volume"]
        return dict(buckets)
    except Exception:
        return None


def _draw_headless_tick_profile(ax_c, ax_p, buckets: dict, df: pd.DataFrame,
                                date_str: str, tf: str):
    """Aggregate all candles' ticks into one profile and draw on headless axes."""
    _, candle_mins = TIMEFRAME_MAP[tf]
    kline_candles: set[datetime] = set()
    for tk_str in df["time_key"]:
        try:
            dt = datetime.strptime(str(tk_str)[:16], "%Y-%m-%d %H:%M")
            kline_candles.add(candle_start(dt, candle_mins))
        except ValueError:
            pass
    covered  = kline_candles & set(buckets.keys())
    coverage = int(100 * len(covered) / len(kline_candles)) if kline_candles else 0

    agg: dict = defaultdict(lambda: {"buy": 0, "sell": 0, "neutral": 0})
    for price_levels in buckets.values():
        for price, counts in price_levels.items():
            agg[price]["buy"]     += counts["buy"]
            agg[price]["sell"]    += counts["sell"]
            agg[price]["neutral"] += counts["neutral"]

    prices = sorted(agg.keys())
    if not prices:
        return
    buy_v  = np.array([agg[p]["buy"]     for p in prices], dtype=float)
    sell_v = np.array([agg[p]["sell"]    for p in prices], dtype=float)
    neu_v  = np.array([agg[p]["neutral"] for p in prices], dtype=float)
    y      = np.array(prices, dtype=float)
    h      = (y[1] - y[0]) * 0.8 if len(y) > 1 else 0.05
    ax_p.barh(y, buy_v,             height=h, color=UP,   alpha=0.85, label="Buy")
    ax_p.barh(y, neu_v, left=buy_v, height=h, color=GREY, alpha=0.70, label="Neutral")
    ax_p.barh(y, sell_v, left=buy_v + neu_v, height=h, color=DOWN, alpha=0.85, label="Sell")
    ax_p.axvline(0, color=FG, linewidth=0.5, alpha=0.4)
    ax_p.set_title(f"Tick (local, {coverage}%)\n{date_str}", color=FG, fontsize=9)
    ax_p.set_xlabel("Volume", color=FG, fontsize=8)
    ax_p.tick_params(axis="x", colors=FG, labelsize=7)
    ax_p.grid(axis="x", color=GRID, linewidth=0.5)
    ax_p.legend(loc="lower right", fontsize=7, facecolor=BG_BAR, labelcolor=FG, edgecolor="#444466")
    ymin = min(df["low"].min(), min(prices)) * 0.999
    ymax = max(df["high"].max(), max(prices)) * 1.001
    ax_c.set_ylim(ymin, ymax)


def _render_headless(args: argparse.Namespace) -> None:
    """Fetch historical data and save a PNG — no window opened."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    import pathlib

    date_str = args.date or (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        print(f"error: invalid date '{date_str}', use YYYY-MM-DD", file=sys.stderr)
        sys.exit(1)

    ktype, _ = TIMEFRAME_MAP[args.tf]
    print(f"Connecting to OpenD {args.host}:{args.port} ...")
    ctx = OpenQuoteContext(host=args.host, port=args.port)
    ret, df, _ = ctx.request_history_kline(
        args.code, start=f"{date_str} 00:00:00", end=f"{date_str} 23:59:59",
        ktype=ktype, autype=AuType.NONE, max_count=args.num)
    ctx.close()
    if ret != RET_OK:
        print(f"error: {df}", file=sys.stderr); sys.exit(1)
    if df.empty:
        print(f"No data for {args.code} on {date_str}", file=sys.stderr); sys.exit(1)

    df = df.tail(args.num).reset_index(drop=True)
    print(f"Fetched {len(df)} candles  "
          f"({df['time_key'].iloc[0]} → {df['time_key'].iloc[-1]})")

    fig = Figure(figsize=(16, 7), facecolor=BG_DARK)
    gs  = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0)
    ax_c = fig.add_subplot(gs[0])
    ax_p = fig.add_subplot(gs[1], sharey=ax_c)
    for ax in (ax_c, ax_p):
        ax.set_facecolor(BG_DARK)
        ax.tick_params(colors=FG)
        for sp in ax.spines.values():
            sp.set_edgecolor("#444466")
    ax_p.spines["left"].set_visible(False)
    ax_p.yaxis.set_tick_params(labelleft=False, left=False)
    fig.suptitle(f"{args.code}  {args.tf}  Historical {date_str}",
                 color=FG, fontsize=13)

    labels = []
    for i, (_, row) in enumerate(df.iterrows()):
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        color = UP if c >= o else DOWN
        ax_c.bar(i, abs(c - o), bottom=min(o, c), color=color, width=0.6, zorder=2)
        ax_c.plot([i, i], [l, h], color=color, linewidth=1)
        tk_str = str(row["time_key"])
        labels.append(tk_str[5:16] if len(tk_str) >= 16 else tk_str)
    ax_c.set_xticks(range(len(df)))
    ax_c.set_xticklabels(labels, rotation=45, fontsize=7, color=FG)
    ax_c.set_ylabel("Price", color=FG)
    ax_c.grid(axis="y", color=GRID, linewidth=0.5)
    ax_c.set_xlim(-0.5, len(df) - 0.5)

    # ── profile: prefer local tick data, fallback to OHLCV ────────────────────
    local_buckets = _load_tick_buckets(args.code, date_str, args.tf)
    if local_buckets:
        print("Using local tick data from DuckDB")
        _draw_headless_tick_profile(ax_c, ax_p, local_buckets, df, date_str, args.tf)
    else:
        print("No local tick data — using OHLCV estimate")
        result = build_ohlcv_profile(df, n_bins=40)
        if result:
            centers, volumes = result
            bin_h   = (centers[1] - centers[0]) * 0.85
            max_vol = volumes.max() or 1
            clrs    = [GREEN if v >= max_vol * 0.7 else (FG if v >= max_vol * 0.4 else GREY)
                       for v in volumes]
            ax_p.barh(centers, volumes, height=bin_h, color=clrs, alpha=0.85)
            poc = centers[int(np.argmax(volumes))]
            for ax in (ax_c, ax_p):
                ax.axhline(poc, color=GOLD, linewidth=0.9, linestyle="--", alpha=0.8, zorder=5)
            ax_c.text(len(df) - 0.5, poc, f" POC {poc:.2f}",
                      color=GOLD, fontsize=7, va="center", ha="left")
            ax_p.set_title(f"Vol Profile (OHLCV est.)\n{date_str}", color=FG, fontsize=9)
            ax_p.set_xlabel("Volume", color=FG, fontsize=8)
            ax_p.tick_params(axis="x", colors=FG, labelsize=7)
            ax_p.grid(axis="x", color=GRID, linewidth=0.5)
            ymin, ymax = df["low"].min() * 0.999, df["high"].max() * 1.001
            ax_c.set_ylim(ymin, ymax)

    fig.tight_layout()
    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
    print(f"Saved: {out}")


def main(argv=None):
    args = _parse_args(argv)
    if args.output:
        args.mode = "Historical"
        _render_headless(args)
        return
    app = OrderFlowApp(args=args)
    app.mainloop()


if __name__ == "__main__":
    main()
