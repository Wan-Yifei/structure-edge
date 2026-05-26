"""
Market Session Scheduler — Tkinter GUI

Manages which US trading sessions the tick collector should run in.
Config is persisted to config/schedule.json.

Run:  uv run analysis/scheduler.py
      uv run main.py scheduler
"""

import json
import pathlib
import subprocess
import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime, time, timedelta, timezone
import threading

import pystray
from PIL import Image, ImageDraw

# ── Paths ─────────────────────────────────────────────────────────────────────
CONFIG_PATH = pathlib.Path(__file__).parent.parent / "config" / "schedule.json"
LOG_PATH    = pathlib.Path(__file__).parent.parent / "logs" / "analysis" / "scheduler.log"

# ── Colours (match orderflow palette) ────────────────────────────────────────
BG_DARK  = "#1a1a2e"
BG_BAR   = "#2a2a3e"
BG_EDIT  = "#333355"
BG_LOG   = "#0d0d1a"
FG       = "white"
GREEN    = "#26a69a"
RED      = "#ef5350"
YELLOW   = "#ffd700"
GREY     = "#888888"
DIM      = "#555577"

SESSION_ORDER = ["overnight", "premarket", "regular", "afterhours"]


# ── ET timezone helpers (no external deps) ────────────────────────────────────

def _is_edt(dt_utc: datetime) -> bool:
    """Return True if *dt_utc* falls in EDT (2nd Sun Mar → 1st Sun Nov)."""
    year = dt_utc.year

    def nth_sunday(month: int, n: int) -> datetime:
        d = datetime(year, month, 1, tzinfo=timezone.utc)
        d += timedelta(days=(6 - d.weekday()) % 7)   # first Sunday
        d += timedelta(weeks=n - 1)
        return d

    edt_start = nth_sunday(3, 2).replace(hour=7)    # 2 AM ET = 7 AM UTC
    edt_end   = nth_sunday(11, 1).replace(hour=6)   # 2 AM ET = 6 AM UTC (already back 1h)
    return edt_start <= dt_utc < edt_end


def utc_now() -> datetime:
    """Return the current UTC datetime with timezone info."""
    return datetime.now(tz=timezone.utc)


def et_now() -> datetime:
    """Return the current Eastern Time (EDT or EST depending on DST)."""
    utc = utc_now()
    offset = timedelta(hours=-4 if _is_edt(utc) else -5)
    return utc + offset


def beijing_now() -> datetime:
    """Return the current Beijing Time (UTC+8, no DST)."""
    utc = utc_now()
    return utc + timedelta(hours=8)


# ── Session state logic ───────────────────────────────────────────────────────

def _to_time(s: str) -> time:
    """Parse an 'HH:MM' string into a datetime.time object."""
    h, m = s.split(":")
    return time(int(h), int(m))


def session_status(cfg: dict, name: str, et: datetime) -> str:
    """
    Return one of: 'ACTIVE', 'STARTING_SOON', 'INACTIVE', 'DISABLED'.
    Handles overnight sessions that wrap midnight.
    """
    sess = cfg["sessions"][name]
    if not sess["enabled"]:
        return "DISABLED"

    prestart = cfg.get("prestart_minutes", 5)
    start_t  = _to_time(sess["start"])
    end_t    = _to_time(sess["end"])
    now_t    = et.time().replace(second=0, microsecond=0)

    # build datetime pairs relative to today for comparison
    today = et.date()
    dt_start = datetime.combine(today, start_t)
    dt_end   = datetime.combine(today, end_t)
    if end_t <= start_t:                         # overnight: end is next day
        if now_t >= start_t:
            dt_end = datetime.combine(today + timedelta(days=1), end_t)
        else:
            dt_start = datetime.combine(today - timedelta(days=1), start_t)

    dt_now    = datetime.combine(today, now_t)
    dt_trigger = dt_start - timedelta(minutes=prestart)

    if dt_start <= dt_now < dt_end:
        return "ACTIVE"
    if dt_trigger <= dt_now < dt_start:
        return "STARTING SOON"
    return "INACTIVE"


def next_event(cfg: dict, et: datetime) -> str:
    """Return a human-readable string for the next upcoming session trigger."""
    prestart  = cfg.get("prestart_minutes", 5)
    today     = et.date()
    upcoming  = []

    for name in SESSION_ORDER:
        sess = cfg["sessions"][name]
        if not sess["enabled"]:
            continue
        start_t   = _to_time(sess["start"])
        trigger_t = (datetime.combine(today, start_t) - timedelta(minutes=prestart)).time()
        dt_trigger = datetime.combine(today, trigger_t)
        if datetime.combine(today, et.time()) < dt_trigger:
            upcoming.append((dt_trigger, sess["label"]))

    if not upcoming:
        return "No more sessions today"
    upcoming.sort()
    dt, label = upcoming[0]
    delta = dt - datetime.combine(today, et.time())
    h, rem = divmod(int(delta.total_seconds()), 3600)
    m = rem // 60
    return f"{label}  (in {h}h {m:02d}m)"


# ── Stock Picker Dialog ───────────────────────────────────────────────────────

class StockPickerDialog(tk.Toplevel):
    """Popup for selecting collection targets from moomoo watchlist groups."""

    def __init__(self, parent, host: str = "127.0.0.1", port: int = 11111):
        super().__init__(parent)
        self._app    = parent
        self._host   = host
        self._port   = port
        self._selected: set[str] = set(parent.cfg.get("targets", []))
        self._groups: list[str]  = []
        self._group_cache: dict[str, list[tuple[str, str]]] = {}
        self._check_vars: dict[str, tk.BooleanVar] = {}

        self.title("Manage Collection Targets")
        self.configure(bg=BG_DARK)
        self.geometry("720x520")
        self.resizable(True, True)
        self.transient(parent)
        self.grab_set()

        self._build_ui()
        self._load_groups()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Status bar
        self._status_var = tk.StringVar(value="Connecting to OpenD…")
        tk.Label(self, textvariable=self._status_var, bg=BG_BAR, fg=YELLOW,
                 font=("Courier", 9), anchor="w", padx=8).pack(fill=tk.X)

        # ── Content row ──────────────────────────────────────────────────────
        content = tk.Frame(self, bg=BG_DARK)
        content.pack(fill=tk.BOTH, expand=True, padx=8, pady=(6, 0))

        # Left — group list
        left = tk.Frame(content, bg=BG_DARK, width=170)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 6))
        left.pack_propagate(False)

        tk.Label(left, text="Watchlist Groups", bg=BG_DARK, fg=GREY,
                 font=("Helvetica", 9, "bold")).pack(anchor="w")
        self._group_lb = tk.Listbox(
            left, bg=BG_EDIT, fg=FG, selectbackground=GREEN, selectforeground=FG,
            font=("Courier", 9), activestyle="none", exportselection=False,
        )
        grp_sb = ttk.Scrollbar(left, orient="vertical", command=self._group_lb.yview)
        self._group_lb.configure(yscrollcommand=grp_sb.set)
        grp_sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._group_lb.pack(fill=tk.BOTH, expand=True)
        self._group_lb.bind("<<ListboxSelect>>", self._on_group_select)

        # Right — stock checkboxes + target list
        right = tk.Frame(content, bg=BG_DARK)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Stock checkbox area
        tk.Label(right, text="Stocks in group  (check to add)", bg=BG_DARK, fg=GREY,
                 font=("Helvetica", 9, "bold")).pack(anchor="w")

        box = tk.Frame(right, bg=BG_EDIT, relief=tk.SUNKEN, bd=1)
        box.pack(fill=tk.BOTH, expand=True)

        self._canvas = tk.Canvas(box, bg=BG_EDIT, highlightthickness=0)
        s_sb = ttk.Scrollbar(box, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=s_sb.set)
        s_sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._stock_inner = tk.Frame(self._canvas, bg=BG_EDIT)
        self._cwin = self._canvas.create_window((0, 0), window=self._stock_inner, anchor="nw")
        self._stock_inner.bind("<Configure>", lambda e: self._canvas.configure(
            scrollregion=self._canvas.bbox("all")))
        self._canvas.bind("<Configure>", lambda e: self._canvas.itemconfig(
            self._cwin, width=e.width))

        # Target list
        tk.Label(right, text="Collection Targets", bg=BG_DARK, fg=GREY,
                 font=("Helvetica", 9, "bold")).pack(anchor="w", pady=(6, 0))

        tgt_row = tk.Frame(right, bg=BG_DARK)
        tgt_row.pack(fill=tk.X)
        self._targets_lb = tk.Listbox(
            tgt_row, bg=BG_EDIT, fg=GREEN, font=("Courier", 9),
            height=4, selectbackground=RED, exportselection=False,
        )
        self._targets_lb.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(tgt_row, text="Remove", bg=RED, fg=FG, width=8,
                  relief=tk.FLAT, command=self._remove_target).pack(side=tk.LEFT, padx=4)

        # ── Bottom bar ───────────────────────────────────────────────────────
        bot = tk.Frame(self, bg=BG_DARK, pady=6)
        bot.pack(fill=tk.X, padx=8)

        tk.Label(bot, text="Add code:", bg=BG_DARK, fg=FG,
                 font=("Helvetica", 9)).pack(side=tk.LEFT)
        self._add_var = tk.StringVar()
        add_entry = tk.Entry(bot, textvariable=self._add_var, width=14,
                             bg=BG_EDIT, fg=FG, insertbackground=FG)
        add_entry.pack(side=tk.LEFT, padx=4)
        add_entry.bind("<Return>", lambda e: self._add_custom())
        tk.Button(bot, text="+ Add", bg=BG_BAR, fg=FG, width=7,
                  relief=tk.FLAT, command=self._add_custom).pack(side=tk.LEFT, padx=(0, 16))

        tk.Button(bot, text="Save", bg=GREEN, fg=FG, width=10,
                  relief=tk.FLAT, command=self._save).pack(side=tk.RIGHT, padx=(4, 0))
        tk.Button(bot, text="Close", bg=BG_BAR, fg=FG, width=8,
                  relief=tk.FLAT, command=self.destroy).pack(side=tk.RIGHT)

        self._refresh_targets()

    # ── API loading (background threads) ─────────────────────────────────────

    def _load_groups(self):
        threading.Thread(target=self._do_load_groups, daemon=True).start()

    def _do_load_groups(self):
        try:
            from moomoo import OpenQuoteContext, RET_OK
            ctx = OpenQuoteContext(host=self._host, port=self._port)
            ret, data = ctx.get_user_security_group()
            ctx.close()
            if ret == RET_OK:
                groups = data["group_name"].tolist()
                self.after(0, self._populate_groups, groups)
            else:
                self.after(0, self._status_var.set, f"Error: {data}")
        except Exception as exc:
            self.after(0, self._status_var.set, f"OpenD error: {exc}")

    def _populate_groups(self, groups: list[str]):
        self._groups = groups
        self._group_lb.delete(0, tk.END)
        for g in groups:
            self._group_lb.insert(tk.END, g)
        self._status_var.set(f"Loaded {len(groups)} groups — click a group to browse stocks")

    def _on_group_select(self, _event=None):
        sel = self._group_lb.curselection()
        if not sel:
            return
        group = self._groups[sel[0]]
        if group in self._group_cache:
            self._show_stocks(self._group_cache[group])
        else:
            self._status_var.set(f"Loading {group}…")
            threading.Thread(target=self._do_load_stocks, args=(group,), daemon=True).start()

    def _do_load_stocks(self, group: str):
        try:
            from moomoo import OpenQuoteContext, RET_OK
            ctx = OpenQuoteContext(host=self._host, port=self._port)
            ret, data = ctx.get_user_security(group)
            ctx.close()
            if ret == RET_OK:
                names = data["stock_name"].tolist() if "stock_name" in data.columns else [""] * len(data)
                stocks = list(zip(data["code"].tolist(), names))
                self._group_cache[group] = stocks
                self.after(0, self._show_stocks, stocks)
                self.after(0, self._status_var.set, f"{group}: {len(stocks)} stocks")
            else:
                self.after(0, self._status_var.set, f"Error loading {group}: {data}")
        except Exception as exc:
            self.after(0, self._status_var.set, f"Error: {exc}")

    # ── Stock checkboxes ──────────────────────────────────────────────────────

    def _show_stocks(self, stocks: list[tuple[str, str]]):
        for w in self._stock_inner.winfo_children():
            w.destroy()
        self._check_vars.clear()

        for code, name in stocks:
            var = tk.BooleanVar(value=(code in self._selected))
            self._check_vars[code] = var
            label = f"{code}   {name[:24]}" if name else code
            tk.Checkbutton(
                self._stock_inner, text=label, variable=var,
                bg=BG_EDIT, fg=FG, selectcolor=BG_DARK,
                activebackground=BG_EDIT, activeforeground=FG,
                font=("Courier", 9), anchor="w",
                command=lambda c=code, v=var: self._toggle(c, v),
            ).pack(fill=tk.X, padx=6, pady=1)

    def _toggle(self, code: str, var: tk.BooleanVar):
        if var.get():
            self._selected.add(code)
        else:
            self._selected.discard(code)
        self._refresh_targets()

    # ── Target list helpers ───────────────────────────────────────────────────

    def _refresh_targets(self):
        self._targets_lb.delete(0, tk.END)
        for code in sorted(self._selected):
            self._targets_lb.insert(tk.END, code)

    def _remove_target(self):
        sel = self._targets_lb.curselection()
        if not sel:
            return
        code = self._targets_lb.get(sel[0])
        self._selected.discard(code)
        if code in self._check_vars:
            self._check_vars[code].set(False)
        self._refresh_targets()

    def _add_custom(self):
        code = self._add_var.get().strip().upper()
        if not code:
            return
        if "." not in code:
            self._status_var.set("Invalid format — use e.g. US.AAPL or HK.00700")
            return
        self._selected.add(code)
        self._add_var.set("")
        self._refresh_targets()
        self._status_var.set(f"Added {code}")

    def _save(self):
        self._app.cfg["targets"] = sorted(self._selected)
        self._app.stocks_var.set(", ".join(sorted(self._selected)))
        self._app._save_config()
        self.destroy()


# ── Main GUI ──────────────────────────────────────────────────────────────────

class SchedulerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Market Session Scheduler")
        self.configure(bg=BG_DARK)
        self.geometry("740x640")
        self.resizable(False, False)

        self.cfg = self._load_config()
        self._collector_running = False
        self._tick_job = None
        self._row_widgets: dict[str, dict] = {}
        self._proc: subprocess.Popen | None = None
        self._tray: pystray.Icon | None = None

        self._build_clock()
        self._build_sessions()
        self._build_stocks()
        self._build_controls()
        self._build_log()

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._setup_tray()
        self._tick()    # start clock + status loop

    # ── Config I/O ───────────────────────────────────────────────────────────

    def _load_config(self) -> dict:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, encoding="utf-8") as f:
                return json.load(f)
        return {
            "prestart_minutes": 5,
            "data_timeout_minutes": 5,
            "sessions": {
                "overnight":  {"enabled": False, "start": "20:00", "end": "04:00",
                               "label": "Overnight  20:00–04:00 ET"},
                "premarket":  {"enabled": True,  "start": "04:00", "end": "09:30",
                               "label": "Pre-market  04:00–09:30 ET"},
                "regular":    {"enabled": True,  "start": "09:30", "end": "16:00",
                               "label": "Regular  09:30–16:00 ET"},
                "afterhours": {"enabled": True,  "start": "16:00", "end": "20:00",
                               "label": "After-hours  16:00–20:00 ET"},
            },
            "targets": ["US.SNDK"],
        }

    def _save_config(self):
        for name, widgets in self._row_widgets.items():
            self.cfg["sessions"][name]["enabled"] = bool(widgets["enabled"].get())
        self.cfg["prestart_minutes"]      = int(self.prestart_var.get())
        self.cfg["data_timeout_minutes"]  = int(self.timeout_var.get())
        self.cfg["targets"] = [t.strip() for t in self.stocks_var.get().split(",") if t.strip()]
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(self.cfg, f, indent=2, ensure_ascii=False)
        self._log("Config saved.")

    # ── UI builders ───────────────────────────────────────────────────────────

    def _build_clock(self):
        frame = tk.Frame(self, bg=BG_BAR, pady=8)
        frame.pack(fill=tk.X)

        self.et_var      = tk.StringVar(value="ET  --:--:--")
        self.bj_var      = tk.StringVar(value="BJ  --:--:--")
        self.next_var    = tk.StringVar(value="Next: —")

        tk.Label(frame, textvariable=self.et_var,   bg=BG_BAR, fg=YELLOW,
                 font=("Courier", 13, "bold")).pack(side=tk.LEFT, padx=16)
        tk.Label(frame, textvariable=self.bj_var,   bg=BG_BAR, fg=FG,
                 font=("Courier", 13)).pack(side=tk.LEFT, padx=8)
        tk.Label(frame, textvariable=self.next_var, bg=BG_BAR, fg=GREY,
                 font=("Courier", 10)).pack(side=tk.RIGHT, padx=16)

    def _build_sessions(self):
        outer = tk.LabelFrame(self, text=" Sessions ", bg=BG_DARK, fg=FG,
                              font=("Helvetica", 10, "bold"))
        outer.pack(fill=tk.X, padx=12, pady=(10, 4))

        hdr_cols = ["Session", "Start (ET)", "End (ET)", "Enabled", "Status"]
        widths   = [26,        10,            10,          8,         14]
        for col, (h, w) in enumerate(zip(hdr_cols, widths)):
            tk.Label(outer, text=h, bg=BG_DARK, fg=GREY,
                     font=("Helvetica", 9, "bold"), width=w, anchor="w"
                     ).grid(row=0, column=col, padx=4, pady=(2, 4), sticky="w")

        for row_i, name in enumerate(SESSION_ORDER, start=1):
            sess = self.cfg["sessions"][name]
            widgets = {}

            tk.Label(outer, text=sess["label"], bg=BG_DARK, fg=FG,
                     font=("Courier", 9), width=26, anchor="w"
                     ).grid(row=row_i, column=0, padx=4, pady=3, sticky="w")

            # start / end (editable)
            for col, key in [(1, "start"), (2, "end")]:
                var = tk.StringVar(value=sess[key])
                widgets[key] = var
                tk.Entry(outer, textvariable=var, width=8, bg=BG_EDIT,
                         fg=FG, insertbackground=FG, font=("Courier", 9)
                         ).grid(row=row_i, column=col, padx=4)

            # enabled toggle
            en_var = tk.BooleanVar(value=sess["enabled"])
            widgets["enabled"] = en_var
            cb = tk.Checkbutton(outer, variable=en_var, bg=BG_DARK,
                                fg=GREEN, selectcolor=BG_EDIT,
                                activebackground=BG_DARK, command=self._save_config)
            cb.grid(row=row_i, column=3, padx=4)

            # status label (updated by clock loop)
            status_var = tk.StringVar(value="—")
            widgets["status"] = status_var
            tk.Label(outer, textvariable=status_var, bg=BG_DARK,
                     font=("Courier", 9), width=14, anchor="w"
                     ).grid(row=row_i, column=4, padx=4, sticky="w")

            self._row_widgets[name] = widgets

    def _build_stocks(self):
        frame = tk.Frame(self, bg=BG_DARK)
        frame.pack(fill=tk.X, padx=12, pady=4)

        tk.Label(frame, text="Stocks:", bg=BG_DARK, fg=FG,
                 font=("Helvetica", 9)).pack(side=tk.LEFT, padx=(0, 6))
        self.stocks_var = tk.StringVar(value=", ".join(self.cfg.get("targets", [])))
        tk.Entry(frame, textvariable=self.stocks_var, width=28,
                 bg=BG_EDIT, fg=FG, insertbackground=FG).pack(side=tk.LEFT)
        tk.Button(frame, text="Manage…", bg=BG_BAR, fg=FG, width=9,
                  relief=tk.FLAT, command=self._open_stock_picker
                  ).pack(side=tk.LEFT, padx=6)

        tk.Label(frame, text="Pre-start:", bg=BG_DARK, fg=FG,
                 font=("Helvetica", 9)).pack(side=tk.LEFT, padx=(8, 4))
        self.prestart_var = tk.IntVar(value=self.cfg.get("prestart_minutes", 5))
        tk.Spinbox(frame, from_=0, to=30, textvariable=self.prestart_var,
                   width=3, bg=BG_EDIT, fg=FG, buttonbackground=BG_BAR).pack(side=tk.LEFT)
        tk.Label(frame, text="min   No-data warn:", bg=BG_DARK, fg=FG,
                 font=("Helvetica", 9)).pack(side=tk.LEFT, padx=(4, 4))
        self.timeout_var = tk.IntVar(value=self.cfg.get("data_timeout_minutes", 5))
        tk.Spinbox(frame, from_=1, to=60, textvariable=self.timeout_var,
                   width=3, bg=BG_EDIT, fg=FG, buttonbackground=BG_BAR).pack(side=tk.LEFT)
        tk.Label(frame, text="min", bg=BG_DARK, fg=FG,
                 font=("Helvetica", 9)).pack(side=tk.LEFT, padx=(4, 0))

    def _build_controls(self):
        frame = tk.Frame(self, bg=BG_DARK, pady=6)
        frame.pack(fill=tk.X, padx=12)

        tk.Button(frame, text="Save Config", bg=BG_BAR, fg=FG,
                  width=14, relief=tk.FLAT, command=self._save_config
                  ).pack(side=tk.LEFT, padx=(0, 8))

        self.run_btn = tk.Button(frame, text="▶  Start Scheduler", bg=GREEN, fg=FG,
                                 width=18, relief=tk.FLAT, command=self._start_scheduler)
        self.run_btn.pack(side=tk.LEFT, padx=(0, 8))

        self.stop_btn = tk.Button(frame, text="■  Stop", bg=RED, fg=FG,
                                  width=10, relief=tk.FLAT, command=self._stop_scheduler,
                                  state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT)

    def _build_log(self):
        frame = tk.LabelFrame(self, text=" Log ", bg=BG_DARK, fg=FG,
                              font=("Helvetica", 10, "bold"))
        frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(4, 10))

        self.log_text = tk.Text(frame, bg=BG_LOG, fg=FG, font=("Courier", 9),
                                state=tk.DISABLED, height=10, wrap=tk.WORD)
        sb = ttk.Scrollbar(frame, command=self.log_text.yview)
        self.log_text.config(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    # ── Clock / status loop ───────────────────────────────────────────────────

    def _tick(self):
        et = et_now()
        bj = beijing_now()
        self.et_var.set(f"ET  {et.strftime('%H:%M:%S')}")
        self.bj_var.set(f"BJ  {bj.strftime('%H:%M:%S')}")
        self.next_var.set(f"Next: {next_event(self.cfg, et)}")

        # sync start/end edits back to cfg and refresh status
        for name, widgets in self._row_widgets.items():
            try:
                self.cfg["sessions"][name]["start"] = widgets["start"].get()
                self.cfg["sessions"][name]["end"]   = widgets["end"].get()
                self.cfg["sessions"][name]["enabled"] = bool(widgets["enabled"].get())
            except Exception:
                pass

            st = session_status(self.cfg, name, et)
            color = {
                "ACTIVE":       GREEN,
                "STARTING SOON": YELLOW,
                "INACTIVE":     DIM,
                "DISABLED":     GREY,
            }.get(st, FG)
            widgets["status"].set(st)
            # update label colour dynamically
            for w in self.nametowidget(self.winfo_children()[0].winfo_name() if False else ".").winfo_children():
                pass  # colour update handled via StringVar; colour set below

            # update colour via grid slave lookup
            try:
                from tkinter import _default_root  # noqa: F401
            except Exception:
                pass

        # check if scheduler is running → fire callbacks
        if self._collector_running:
            self._check_session_transitions(et)

        self._tick_job = self.after(1000, self._tick)

    def _check_session_transitions(self, et: datetime):
        """Launch / stop the tick collector at session boundaries."""
        for name in SESSION_ORDER:
            st = session_status(self.cfg, name, et)
            prev = getattr(self, f"_prev_{name}", None)
            if st != prev:
                targets = ", ".join(self.cfg.get("targets", []))
                if st == "STARTING SOON":
                    self._log(f"[{name.upper()}]  Pre-start: launching collector for {targets}")
                    self._launch_collector()
                elif st == "ACTIVE":
                    self._log(f"[{name.upper()}]  Session ACTIVE")
                    # Always attempt launch: no-op if already running, restarts if a
                    # preceding session's INACTIVE transition just killed the collector.
                    self._launch_collector()
                elif st == "INACTIVE" and prev in ("ACTIVE", "STARTING SOON"):
                    other_active = any(
                        session_status(self.cfg, s, et) in ("ACTIVE", "STARTING SOON")
                        for s in SESSION_ORDER if s != name
                    )
                    if other_active:
                        self._log(f"[{name.upper()}]  Session ended — keeping collector (another session active)")
                    else:
                        self._log(f"[{name.upper()}]  Session ended — stopping collector")
                        self._kill_collector()
            setattr(self, f"_prev_{name}", st)

    # ── Collector subprocess ──────────────────────────────────────────────────

    def _launch_collector(self):
        """Start tick_collector.py as a subprocess if not already running."""
        if self._proc is not None and self._proc.poll() is None:
            return  # already running
        collector = pathlib.Path(__file__).parent / "tick_collector.py"
        timeout   = int(self.timeout_var.get())
        cmd = [
            sys.executable, str(collector),
            "--config",  str(CONFIG_PATH),
            "--timeout", str(timeout),
        ]
        import os
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}   # force line-flush in child
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            self._log(f"Collector PID {self._proc.pid} started  (no-data warn: {timeout} min)")
            threading.Thread(target=self._tail_proc, daemon=True).start()
        except Exception as exc:
            self._log(f"Failed to start collector: {exc}")

    def _kill_collector(self):
        if self._proc is None:
            return
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._log(f"Collector PID {self._proc.pid} stopped")
        self._proc = None

    def _tail_proc(self):
        """Read collector stdout in a background thread and forward to the log."""
        proc = self._proc
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                self.after(0, self._log, f"  [collector] {line}")
        self.after(0, self._log, "  [collector] process exited")

    # ── Scheduler control ─────────────────────────────────────────────────────

    def _start_scheduler(self):
        self._save_config()
        self._collector_running = True
        self.run_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self._log("Scheduler started.  Watching session boundaries...")
        if self._tray:
            self._tray.icon = self._make_tray_image(active=True)
        for name in SESSION_ORDER:
            setattr(self, f"_prev_{name}", None)
        # immediately evaluate current sessions instead of waiting for next _tick
        self._check_session_transitions(et_now())

    def _stop_scheduler(self):
        self._collector_running = False
        self._kill_collector()
        self.run_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self._log("Scheduler stopped.")
        if self._tray:
            self._tray.icon = self._make_tray_image(active=False)

    def _open_stock_picker(self):
        StockPickerDialog(self)

    # ── System tray ───────────────────────────────────────────────────────────

    def _make_tray_image(self, active: bool = False) -> Image.Image:
        img  = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        fill = "#ef5350" if active else "#888888"
        draw.ellipse([6, 6, 58, 58], fill=fill, outline="#ffffff", width=2)
        return img

    def _setup_tray(self):
        menu = pystray.Menu(
            pystray.MenuItem("Show", self._tray_show, default=True),
            pystray.MenuItem("Exit", self._tray_exit),
        )
        self._tray = pystray.Icon(
            "scheduler", self._make_tray_image(), "Market Scheduler", menu
        )
        threading.Thread(target=self._tray.run, daemon=True).start()

    def _tray_show(self, icon=None, item=None):
        self.after(0, self._do_show)

    def _do_show(self):
        self.deiconify()
        self.lift()
        self.focus_force()

    def _tray_exit(self, icon=None, item=None):
        self.after(0, self._quit_app)

    def _quit_app(self):
        if self._tray:
            self._tray.stop()
        if self._tick_job:
            self.after_cancel(self._tick_job)
        self._kill_collector()
        self.destroy()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{ts}  {msg}\n"
        # GUI panel
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, line)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        # File log
        try:
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            pass

    def _on_close(self):
        """Minimize to tray instead of quitting."""
        self.withdraw()
        if self._tray:
            self._tray.icon = self._make_tray_image(active=self._collector_running)


def main(argv=None):  # argv accepted for uniform main.py dispatch, not used
    app = SchedulerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
