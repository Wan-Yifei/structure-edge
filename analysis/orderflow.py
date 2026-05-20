"""
Order Flow Analyzer — Tkinter GUI + headless CLI

Modes
-----
GUI (default):
    uv run analysis/orderflow.py
    uv run analysis/orderflow.py --code US.AAPL --tf 5m --mode Historical --date 2026-05-15

Headless (save PNG without opening a window):
    uv run analysis/orderflow.py --code US.SNDK --date 2026-05-15 --output chart.png
    uv run analysis/orderflow.py --code US.SNDK --date 2026-05-15 --tf 5m --num 30 \
        --output out/sndk_5m.png

Via main entry point:
    uv run main.py orderflow [same args as above]
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

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import pandas as pd
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
    draw_candle_heatmap, draw_candle_deltas,
    draw_bos_choch, draw_fvg, draw_order_blocks,
    aggregate_buckets, bucket_coverage, prices_arrays,
)
from strategy.smc.market_structure import detect_bos_choch
from strategy.smc.fvg import detect_fvg
from strategy.smc.order_blocks import detect_order_blocks

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


# ── Timeframe config ──────────────────────────────────────────────────────────
TIMEFRAME_MAP: dict[str, tuple[KLType, int]] = {
    "1m":  (KLType.K_1M,  1),
    "5m":  (KLType.K_5M,  5),
    "15m": (KLType.K_15M, 15),
    "30m": (KLType.K_30M, 30),
    "1h":  (KLType.K_60M, 60),
}


class OrderFlowApp(tk.Tk):
    def __init__(self, args: argparse.Namespace | None = None):
        super().__init__()
        self.title("Order Flow Analyzer")
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

        # tracked profile artists (surgically removed on hover update)
        self._profile_bar_rects: list = []
        self._profile_axvline         = None
        self._profile_legend          = None
        self._hovered_candle_idx: int | None = None
        self._profile_shown_idx: int | None = None   # candle idx currently in ax_p
        self._profile_data_cache: dict[int, tuple] = {}  # idx -> (prices,buy,sell,neu,title)
        self._live_buckets: dict = {}
        self._hist_buckets: dict | None = None

        # SMC overlay artists (cleared on each refresh)
        self._smc_artists: list = []

        # annotation / crosshair objects — recreated after each fig.clear()
        self._tip_c  = None
        self._tip_p  = None
        self._ch_hline_c = None
        self._ch_vline_c = None
        self._ch_hline_p  = None
        self._ch_label    = None
        self._ch_label_p  = None   # price label on ax_p crosshair

        # blitting: saved background for fast crosshair/tooltip rendering
        self._bg: object = None
        self._draw_event_cid: int | None = None
        self._fetching: bool = False

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
        self._load_indicator_cfg()

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

    # ── Config helpers ────────────────────────────────────────────────────────

    _IND_DEFAULTS = {
        "heatmap": True, "delta": True,
        "bos_choch": False, "fvg": False, "ob": False,
    }

    def _load_indicator_cfg(self):
        cfg = {}
        if self._cfg_path.exists():
            try:
                cfg = json.loads(self._cfg_path.read_text(encoding="utf-8")).get("indicators", {})
            except Exception:
                pass
        for key, default in self._IND_DEFAULTS.items():
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
        tk.Entry(bar, textvariable=self.code_var, width=12,
                 bg=BG_EDIT, fg=FG, insertbackground=FG).pack(side=tk.LEFT, padx=(0, 10))

        lbl("Timeframe:")
        self.tf_var = tk.StringVar(value="15m")
        ttk.Combobox(bar, textvariable=self.tf_var, values=list(TIMEFRAME_MAP),
                     width=5, state="readonly").pack(side=tk.LEFT, padx=(0, 10))

        lbl("Candles:")
        self.num_var = tk.IntVar(value=26)
        tk.Spinbox(bar, from_=5, to=200, textvariable=self.num_var,
                   width=4, bg=BG_EDIT, fg=FG, buttonbackground=BG_BAR
                   ).pack(side=tk.LEFT, padx=(0, 10))

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

        _IND_LABELS = [
            ("heatmap",   "Heatmap"),
            ("delta",     "Delta Δ"),
            ("bos_choch", "BOS / CHoCH"),
            ("fvg",       "FVG"),
            ("ob",        "Order Blocks"),
        ]
        for key, label in _IND_LABELS:
            ttk.Checkbutton(
                ind_bar, text=label,
                variable=self._ind[key],
                command=self._on_indicator_toggle,
                style="Ind.TCheckbutton",
            ).pack(side=tk.LEFT, padx=6)

    def _on_indicator_toggle(self):
        self._save_indicator_cfg()
        if self.running and self._klines_data is not None:
            self._refresh_chart()

    def _build_chart(self):
        self.fig = Figure(facecolor=BG_DARK)
        self.ax_c = self.fig.add_subplot(121)
        self.ax_p = self.fig.add_subplot(122)
        self._style_axes()

        chart_frame = tk.Frame(self, bg=BG_DARK)
        chart_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        tb_frame = tk.Frame(chart_frame, bg=BG_DARK)
        tb_frame.pack(fill=tk.X)
        NavigationToolbar2Tk(self.canvas, tb_frame)

        self.canvas.mpl_connect("motion_notify_event", self._on_hover)
        self.canvas.mpl_connect("axes_leave_event",    self._on_axes_leave)
        self.canvas.mpl_connect("scroll_event",        self._on_scroll)
        # grab focus on enter so motion events work without clicking first
        self.canvas.mpl_connect(
            "figure_enter_event",
            lambda _e: self.canvas.get_tk_widget().focus_set(),
        )

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
        self.ax_c.set_title("Candlestick", color=FG)
        self.ax_p.set_title("Profile", color=FG)

    def _on_mode_change(self):
        live = self.mode_var.get() == "Live"
        if live:
            self.date_var.set(datetime.now().strftime("%Y-%m-%d"))
        self.refresh_spin.config(state=tk.NORMAL if live else tk.DISABLED)
        self.date_entry.config(state=tk.DISABLED if live else tk.NORMAL)

    # ── Crosshair ─────────────────────────────────────────────────────────────

    def _init_crosshair(self):
        """Create crosshair line objects after each chart rebuild."""
        kw = dict(color=CROSS, linewidth=0.7, linestyle="--",
                  alpha=0.65, visible=False, zorder=8, animated=True)
        self._ch_hline_c = self.ax_c.axhline(0, **kw)
        self._ch_vline_c = self.ax_c.axvline(0, **kw)
        self._ch_hline_p = self.ax_p.axhline(0, **kw)
        self._ch_label_p = self.ax_p.annotate(
            "", xy=(0, 0), xytext=(4, 0), textcoords="offset points",
            color=GOLD, fontsize=7, va="center", ha="left",
            visible=False, zorder=9, animated=True,
            bbox=dict(fc=BG_TIP, ec="#556688", alpha=0.88, pad=2),
        )
        self._ch_label   = self.ax_c.annotate(
            "", xy=(0, 0), xytext=(-72, 0), textcoords="offset points",
            color=GOLD, fontsize=7, va="center", ha="right",
            visible=False, zorder=9, animated=True,
            bbox=dict(fc=BG_TIP, ec="#556688", alpha=0.88, pad=2),
        )

    def _update_crosshair(self, event):
        in_chart = event.inaxes in (self.ax_c, self.ax_p)
        if not in_chart or event.ydata is None:
            self._hide_crosshair()
            return

        y = event.ydata

        # horizontal lines — both panels (y-axis is shared)
        for line in (self._ch_hline_c, self._ch_hline_p):
            line.set_ydata([y, y])
            line.set_visible(True)

        # vertical line — only on candle panel
        if event.inaxes is self.ax_c and event.xdata is not None:
            self._ch_vline_c.set_xdata([event.xdata, event.xdata])
            self._ch_vline_c.set_visible(True)
        else:
            self._ch_vline_c.set_visible(False)

        # price + volume label: follows cursor in ax_c, offset 72pt to the left
        vol_str = self._vol_at_price(y)
        cx = event.xdata if event.xdata is not None else self.ax_c.get_xlim()[1]
        self._ch_label.xy = (cx, y)
        self._ch_label.set_text(f"{y:.2f}{vol_str}")
        self._ch_label.set_visible(True)

        # price label pinned to the left edge of ax_p on the same horizontal line
        if self._ch_label_p is not None:
            xlo_p = self.ax_p.get_xlim()[0]
            self._ch_label_p.xy = (xlo_p, y)
            self._ch_label_p.set_text(f"{y:.2f}")
            ylo_p, yhi_p = self.ax_p.get_ylim()
            self._ch_label_p.set_visible(ylo_p <= y <= yhi_p)

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

    def _hide_crosshair(self):
        for obj in (self._ch_hline_c, self._ch_vline_c, self._ch_hline_p,
                    self._ch_label, self._ch_label_p):
            if obj is not None:
                obj.set_visible(False)

    def _on_draw(self, event):
        """Save background bitmap after every full canvas render (for blitting)."""
        try:
            self._bg = self.canvas.copy_from_bbox(self.fig.bbox)
        except Exception:
            self._bg = None
        # complete any in-flight perf timers
        for op in ("full_render", "scroll_render"):
            if op in self._perf._t0:
                self._perf.end(op)

    def _blit_dynamic(self):
        """Restore background and draw animated crosshair/tooltip artists via blit."""
        if self._bg is None:
            self.canvas.draw_idle()
            return
        try:
            self.canvas.restore_region(self._bg)
            for artist in (self._ch_hline_c, self._ch_vline_c, self._ch_hline_p,
                           self._ch_label, self._ch_label_p, self._tip_c, self._tip_p):
                if artist is not None:
                    self.fig.draw_artist(artist)
            self.canvas.blit(self.fig.bbox)
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
            self._bg = None
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
        self._bg = None
        self.canvas.draw_idle()

    # ── Scroll zoom (debounced) ───────────────────────────────────────────────

    def _on_scroll(self, event):
        if event.inaxes not in (self.ax_c, self.ax_p) or self._klines_data is None:
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
            # keep full-session profile panel aligned; hover profile autoscales independently
            if self._profile_shown_idx is None:
                self._sync_profile_ylim()

        # debounce: coalesce rapid scroll events into one redraw after 60 ms
        if self._scroll_job is not None:
            self.after_cancel(self._scroll_job)
        self._scroll_job = self.after(60, self._flush_scroll)

    def _flush_scroll(self):
        self._scroll_job = None
        self._bg = None
        self._perf.start("scroll_render")
        self.canvas.draw_idle()

    def _on_axes_leave(self, event):
        self._hide_crosshair()
        mode = self.mode_var.get()
        hover_capable = (
            mode == "Live"
            or (mode == "Historical" and self._hist_buckets is not None)
        )
        if hover_capable and self._hovered_candle_idx is not None:
            self._hovered_candle_idx = None
            self._restore_full_profile()
        self._blit_dynamic()

    # ── Hover tooltip ─────────────────────────────────────────────────────────

    def _on_hover(self, event):
        self._perf.start("hover")
        # 1. update crosshair
        if self._ch_hline_c is not None:
            self._update_crosshair(event)

        # 2. hover profile: update right panel to show hovered candle's distribution
        mode = self.mode_var.get()
        hover_capable = (
            mode == "Live"
            or (mode == "Historical" and self._hist_buckets is not None)
        )
        if hover_capable and self._klines_data is not None:
            if event.inaxes is self.ax_c and event.xdata is not None:
                idx = int(round(event.xdata))
                if 0 <= idx < len(self._klines_data):
                    if idx != self._hovered_candle_idx:
                        self._hovered_candle_idx = idx
                        self._update_hover_profile(idx)
                elif self._hovered_candle_idx is not None:
                    self._hovered_candle_idx = None
                    self._restore_full_profile()
            elif event.inaxes is not self.ax_p and self._hovered_candle_idx is not None:
                # moving to profile panel keeps the per-candle view; only reset on full exit
                self._hovered_candle_idx = None
                self._restore_full_profile()

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
        self.destroy()

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
            start, end = f"{date_str} 00:00:00", f"{date_str} 23:59:59"
        else:
            end   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            start = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d %H:%M:%S")
        ret, df, _ = self.ctx.request_history_kline(
            code, start=start, end=end, ktype=ktype, autype=AuType.NONE,
            max_count=1000, extended_time=True)
        if ret != RET_OK:
            return None
        return df.tail(num).reset_index(drop=True)

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
                ticks = self._load_local_ticks(code, date_str, tf)
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

            smc_signals = None
            if ind["bos_choch"] or ind["ob"]:
                smc_signals = detect_bos_choch(klines)
            fvg_gaps  = detect_fvg(klines)         if ind["fvg"] else []
            ob_blocks = detect_order_blocks(klines, smc_signals or []) if ind["ob"] else []

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

        # rebuild figure
        self.fig.clear()
        gs = self.fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0)
        self.ax_c = self.fig.add_subplot(gs[0])
        self.ax_p = self.fig.add_subplot(gs[1])   # no sharey — we sync manually
        self._style_axes()
        self.ax_p.spines["left"].set_visible(False)
        self.ax_p.yaxis.set_tick_params(labelleft=False, left=False)

        mode_label = f"Historical  {date_str}" if historical else "Live"
        self.fig.suptitle(f"{code}   {tf}   {mode_label}  —  Order Flow",
                          color=FG, fontsize=13)

        # ── candles ───────────────────────────────────────────────────────────
        labels = draw_candles(self.ax_c, klines)
        n = len(klines)
        step = max(1, n // 20)        # show at most ~20 x-tick labels
        shown = list(range(0, n, step))
        self.ax_c.set_xticks(shown)
        self.ax_c.set_xticklabels([labels[i] for i in shown],
                                   rotation=45, fontsize=7, color=FG)
        self.ax_c.set_ylabel("Price", color=FG)
        self.ax_c.grid(axis="y", color=GRID, linewidth=0.5)
        self.ax_c.set_xlim(-0.5, n - 0.5)
        self._hovered_candle_idx  = None
        self._profile_shown_idx   = None
        self._profile_data_cache  = {}   # new chart data invalidates all cached profiles
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
            self._smc_artists += draw_bos_choch(self.ax_c, klines, smc_signals)
        if ind["fvg"] and fvg_gaps:
            self._smc_artists += draw_fvg(self.ax_c, klines, fvg_gaps)
        if ind["ob"] and ob_blocks:
            self._smc_artists += draw_order_blocks(self.ax_c, klines, ob_blocks)

        # ── right-panel profile ───────────────────────────────────────────────
        if historical:
            if ticks:
                coverage  = bucket_coverage(ticks, klines, candle_mins)
                src_label = f"Tick data (local DB, {coverage}% coverage)"
                self._draw_hist_tick_profile(ticks, klines, coverage)
            else:
                src_label = "OHLCV estimate  (no local tick data)"
                result = draw_ohlcv_profile(self.ax_p, self.ax_c, klines,
                                            date_label=date_str)
                self._profile_ohlcv = result
                self._profile_tick  = None
                self._sync_profile_ylim()
        else:
            src_label = None
            self._draw_live_tick_profile(ticks, klines)

        # fixed margins — avoids tight_layout() recomputing every refresh
        self.fig.subplots_adjust(left=0.08, right=0.99, top=0.92, bottom=0.14, wspace=0)

        self._tip_c = make_float_tip(self.ax_c)
        self._tip_p = make_float_tip(self.ax_p)
        self._tip_c.set_animated(True)
        self._tip_p.set_animated(True)
        self._init_crosshair()
        self._klines_data = klines
        self._bg = None
        if self._draw_event_cid is None:
            self._draw_event_cid = self.canvas.mpl_connect("draw_event", self._on_draw)
        self._perf.start("full_render")
        self.canvas.draw_idle()

        total_ticks = sum(p[k] for v in ticks.values() for p in v.values() for k in p)
        if historical:
            self._log(f"Chart ready  |  {n} candles  |  {src_label}"
                      f"  |  Hover candle for single-bar profile")
        else:
            self._log(f"Chart refreshed  |  live ticks: {total_ticks}"
                      f"  |  next refresh in {self.refresh_var.get()}s")

    # ── Profile drawing ───────────────────────────────────────────────────────

    def _load_local_ticks(self, code: str, date_str: str, tf: str) -> dict | None:
        """Query DuckDB for code+date; return candle buckets or None if unavailable."""
        import pathlib as _pl
        db_path = _pl.Path(__file__).parent.parent / "db" / "ticks.db"
        if not db_path.exists():
            return None
        from feeds.tick_store import TickStore
        try:
            day            = datetime.strptime(date_str, "%Y-%m-%d").date()
            _, candle_mins = TIMEFRAME_MAP[tf]
            with TickStore(db_path, read_only=True) as store:
                rows = store.query_date(code, day)
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
        """Align ax_p y-axis to ax_c (used for full-session profile view)."""
        self.ax_p.set_ylim(self.ax_c.get_ylim())

    def _draw_hist_tick_profile(self, buckets: dict, klines: pd.DataFrame, coverage: int):
        self._profile_ohlcv = None
        agg = aggregate_buckets(buckets)
        if not agg:
            self._profile_tick = None
            self.ax_p.set_title("Tick (local)\nno data", color=FG, fontsize=9)
            return
        self._profile_tick = agg
        kl_lo = float(klines["low"].min())  * 0.997
        kl_hi = float(klines["high"].max()) * 1.003
        prices, buy_v, sell_v, neu_v = prices_arrays(agg, lo=kl_lo, hi=kl_hi)
        title = f"Tick (local, {coverage}%)\n{self.date_var.get().strip()}"
        rects, vl, leg = draw_tick_profile_bars(
            self.ax_p, prices, buy_v, sell_v, neu_v, title,
            ax_c=self.ax_c, klines=klines)
        self._profile_bar_rects = rects
        self._profile_axvline   = vl
        self._profile_legend    = leg
        self._sync_profile_ylim()

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
            self.ax_p.set_title("Order Profile", color=FG, fontsize=10)
            return
        latest = max(buckets.keys())
        pd_    = buckets[latest]
        self._profile_tick = pd_
        kl_lo = float(klines["low"].min())  * 0.997
        kl_hi = float(klines["high"].max()) * 1.003
        prices, buy_v, sell_v, neu_v = prices_arrays(pd_, lo=kl_lo, hi=kl_hi)
        title = f"Tick\n{latest.strftime('%Y-%m-%d %H:%M')}"
        rects, vl, leg = draw_tick_profile_bars(
            self.ax_p, prices, buy_v, sell_v, neu_v, title,
            ax_c=self.ax_c, klines=klines)
        self._profile_bar_rects = rects
        self._profile_axvline   = vl
        self._profile_legend    = leg
        self._sync_profile_ylim()

    # ── Hover profile helpers ──────────────────────────────────────────────────

    def _clear_profile_artists(self):
        for patch in self._profile_bar_rects:
            try:
                patch.remove()
            except Exception:
                pass
        self._profile_bar_rects = []
        self.ax_p.containers.clear()
        for attr in ("_profile_axvline", "_profile_legend"):
            obj = getattr(self, attr)
            if obj is not None:
                try:
                    obj.remove()
                except Exception:
                    pass
                setattr(self, attr, None)

    def _update_hover_profile(self, candle_idx: int):
        # already showing this candle — nothing to do
        if candle_idx == self._profile_shown_idx:
            return

        mode    = self.mode_var.get()
        buckets = self._live_buckets if mode == "Live" else self._hist_buckets
        if not buckets or self._klines_data is None:
            return
        if not (0 <= candle_idx < len(self._klines_data)):
            return

        _, candle_mins = TIMEFRAME_MAP[self.tf_var.get()]
        tk_str = str(self._klines_data.iloc[candle_idx]["time_key"])
        try:
            # moomoo time_key is the END of the bar; subtract one period to get the
            # bucket start that candle_start() uses when indexing tick data.
            bar_end = datetime.strptime(tk_str[:16], "%Y-%m-%d %H:%M")
            bucket_key = candle_start(bar_end - timedelta(minutes=candle_mins), candle_mins)
        except ValueError:
            return

        # live mode: latest candle ticks change continuously — always recompute;
        # all other candles are immutable once closed, so we can cache them.
        is_live_latest = (mode == "Live" and candle_idx == len(self._klines_data) - 1)
        cached = None if is_live_latest else self._profile_data_cache.get(candle_idx)

        if cached is None:
            pd_ = buckets.get(bucket_key)
            if not pd_:
                self._clear_profile_artists()
                self.ax_p.set_title(f"Tick\n{tk_str[5:16]}\n(no data)", color=FG, fontsize=9)
                row = self._klines_data.iloc[candle_idx]
                lo, hi = float(row["low"]), float(row["high"])
                pad = max((hi - lo) * 0.15, 0.5)
                self.ax_p.set_ylim(lo - pad, hi + pad)
                self._profile_shown_idx = candle_idx
                self._bg = None
                self.canvas.draw_idle()
                return
            kl_lo = float(self._klines_data["low"].min())  * 0.997
            kl_hi = float(self._klines_data["high"].max()) * 1.003
            prices, buy_v, sell_v, neu_v = prices_arrays(pd_, lo=kl_lo, hi=kl_hi)
            delta     = int(buy_v.sum() - sell_v.sum())
            delta_str = f"+{delta:,}" if delta >= 0 else f"{delta:,}"
            title = f"Tick  Δ={delta_str}\n{tk_str[5:16]}"
            if not is_live_latest:
                self._profile_data_cache[candle_idx] = (prices, buy_v, sell_v, neu_v, title)
        else:
            prices, buy_v, sell_v, neu_v, title = cached

        row  = self._klines_data.iloc[candle_idx]
        c_lo = float(row["low"])
        c_hi = float(row["high"])
        pad  = max((c_hi - c_lo) * 0.15, 0.5)

        self._clear_profile_artists()
        rects, vl, leg = draw_tick_profile_bars(
            self.ax_p, prices, buy_v, sell_v, neu_v, title=title, max_bins=30)
        self._profile_bar_rects = rects
        self._profile_axvline   = vl
        self._profile_legend    = leg
        self.ax_p.set_ylim(c_lo - pad, c_hi + pad)

        self._profile_shown_idx = candle_idx
        self._bg = None
        self.canvas.draw_idle()

    def _restore_full_profile(self):
        if self._klines_data is None:
            return
        self._clear_profile_artists()
        self._profile_shown_idx = None
        if self.mode_var.get() == "Live":
            if self._live_buckets:
                self._draw_live_tick_profile(self._live_buckets, self._klines_data)
        else:
            if self._hist_buckets:
                _, candle_mins = TIMEFRAME_MAP[self.tf_var.get()]
                cov = bucket_coverage(self._hist_buckets, self._klines_data, candle_mins)
                self._draw_hist_tick_profile(self._hist_buckets, self._klines_data, cov)
        # re-sync ax_p to ax_c after full-session profile is drawn
        self._sync_profile_ylim()
        self._bg = None
        self.canvas.draw_idle()

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
        prog="orderflow",
        description="Order flow analyzer: candlestick chart + volume/tick profile.",
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
    p.add_argument("--output",  default=None,          metavar="FILE.png",
                   help="save chart to file and exit without opening GUI (implies Historical)")
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
    fig.suptitle(f"{args.code}  {args.tf}  Historical {date_str}  —  Order Flow",
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
