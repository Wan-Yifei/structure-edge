"""Several sessions' volume profiles side by side, to watch value migrate.

The main chart's profile panel answers "where did volume trade today".
This answers the question a day of that cannot: which way the value area
has been walking, and whether a breakout was accepted by the next
session's distribution or handed straight back.

Each session is a slab: its profile drawn as horizontal bars growing
rightward from the session's left edge, the price path laid over them,
and POC / VAH / VAL marked. Dashed lines join consecutive sessions'
value-area edges, because the migration is the point and the eye should
not have to carry it between slabs.

Profiles are built from 1m klines through the same compute_volume_profile
the main chart uses, with each bar's volume spread across the bins it
spans, and the value area from the same _compute_poc_vah_val -- so a
level here means the same thing as a level there. Ticks would be finer
but only exist for the archived window; klines go back as far as the
fetcher's cache.
"""
from __future__ import annotations

import sys
from datetime import timedelta

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (QApplication, QCheckBox, QLabel, QMainWindow,
                             QPushButton, QSpinBox, QToolBar, QWidget)

_BG = "#1a1a2e"
_FG = "#e0e0e0"
_GREY = "#546e7a"
_GOLD = "#ffa726"
_VA = "#42a5f5"
_PATH = "#cfd8dc"
_PROF = "#37474f"

RTH_LO, RTH_HI = 9.5, 16.0
# Width of a session's widest profile bar, as a fraction of the slab. Less
# than one so the heaviest bin stops short of the next session rather than
# running into it -- two slabs touching read as one.
BAR_SPAN = 0.78


def _qc(hex_: str, alpha: int = 255):
    c = pg.mkColor(hex_)
    c.setAlpha(alpha)
    return c


class MultiVPWindow(QMainWindow):
    """Daily volume profiles across N sessions, with the value area joined."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Multi-Day Volume Profile")
        self.resize(1180, 680)
        self._code = "US.SOXL"
        self._items: list = []

        tb = QToolBar()
        tb.setMovable(False)
        self.addToolBar(tb)

        tb.addWidget(QLabel("  Days "))
        self._days = QSpinBox()
        self._days.setRange(2, 30)
        self._days.setValue(6)
        self._days.setFixedWidth(52)
        self._days.setToolTip("How many trading sessions to look back")
        self._days.valueChanged.connect(self.reload)
        tb.addWidget(self._days)

        tb.addSeparator()
        tb.addWidget(QLabel("  Bins "))
        self._bins = QSpinBox()
        self._bins.setRange(20, 300)
        self._bins.setValue(80)
        self._bins.setSingleStep(10)
        self._bins.setFixedWidth(60)
        self._bins.setToolTip(
            "Price bins per session. Too many and a day's profile breaks\n"
            "into spikes; too few and one wide bin swallows the value-area\n"
            "edge.")
        self._bins.valueChanged.connect(self.reload)
        tb.addWidget(self._bins)

        tb.addSeparator()
        tb.addWidget(QLabel("  VA% "))
        self._va = QSpinBox()
        self._va.setRange(50, 95)
        self._va.setValue(70)
        self._va.setSingleStep(5)
        self._va.setFixedWidth(52)
        self._va.setToolTip("Share of volume the value area covers; 70 is the standard")
        self._va.valueChanged.connect(self.reload)
        tb.addWidget(self._va)

        tb.addSeparator()
        self._rth = QCheckBox("RTH only")
        self._rth.setChecked(True)
        self._rth.setToolTip(
            "Use only 09:30-16:00 bars. Extended-hours volume is thin and\n"
            "mixing it in stretches the value area.")
        self._rth.stateChanged.connect(self.reload)
        tb.addWidget(self._rth)

        self._join = QCheckBox("Join VA")
        self._join.setChecked(True)
        self._join.setToolTip("Dashed lines between consecutive sessions' VAH / VAL")
        self._join.stateChanged.connect(self.reload)
        tb.addWidget(self._join)

        self._shared = QCheckBox("Shared grid")
        self._shared.setChecked(True)
        self._shared.setToolTip(
            "One price grid across every session, so a bin is the same\n"
            "width everywhere and a level can be followed across the\n"
            "panel. Unchecked, each session is binned over its own range:\n"
            "a quiet day's shape reads better, but the days stop being\n"
            "comparable.")
        self._shared.stateChanged.connect(self.reload)
        tb.addWidget(self._shared)

        self._labels = QCheckBox("Level prices")
        self._labels.setChecked(True)
        self._labels.setToolTip("Print POC / VAH / VAL beside each session")
        self._labels.stateChanged.connect(self.reload)
        tb.addWidget(self._labels)

        tb.addSeparator()
        btn = QPushButton("↻")
        btn.setFixedWidth(30)
        btn.setToolTip("Refetch")
        btn.clicked.connect(self.reload)
        tb.addWidget(btn)

        self._status = QLabel("  ")
        tb.addWidget(self._status)

        self._plot = pg.PlotWidget()
        self._plot.setBackground(_BG)
        self._plot.showGrid(x=False, y=True, alpha=0.12)
        self._plot.getPlotItem().getAxis("left").setTextPen(_qc(_FG))
        self._plot.getPlotItem().getAxis("bottom").setTextPen(_qc(_FG))
        self._plot.setLabel("left", "Price", color=_FG)
        self.setCentralWidget(self._plot)

        # Crosshair. Built once and kept: rebuilding per redraw would add
        # a fresh pair to the plot every time a spin box moves.
        pen = pg.mkPen(_qc(_GREY, 170), width=1,
                       style=Qt.PenStyle.DashLine)
        self._hline = pg.InfiniteLine(angle=0, movable=False, pen=pen)
        self._vline = pg.InfiniteLine(angle=90, movable=False, pen=pen)
        for ln in (self._hline, self._vline):
            ln.setZValue(50)
            ln.setVisible(False)
            self._plot.addItem(ln, ignoreBounds=True)
        self._read = pg.TextItem(anchor=(0, 1), color=_qc(_FG),
                                 fill=_qc("#0f3460", 200))
        self._read.setFont(QFont("Monospace", 9))
        self._read.setZValue(51)
        self._read.setVisible(False)
        self._plot.addItem(self._read, ignoreBounds=True)
        # Session levels, filled by reload() and read on hover.
        self._levels: dict[int, tuple] = {}
        self._plot.scene().sigMouseMoved.connect(self._on_mouse)
        self._plot.getPlotItem().vb.setMouseEnabled(x=True, y=True)

    # ── crosshair ───────────────────────────────────────────────────────
    def _on_mouse(self, pos) -> None:
        vb = self._plot.getPlotItem().vb
        if not self._plot.sceneBoundingRect().contains(pos):
            for it in (self._hline, self._vline, self._read):
                it.setVisible(False)
            return
        p = vb.mapSceneToView(pos)
        x, y = float(p.x()), float(p.y())
        self._hline.setPos(y)
        self._vline.setPos(x)
        for it in (self._hline, self._vline):
            it.setVisible(True)

        txt = f"{y:.2f}"
        lv = self._levels.get(int(np.floor(x)))
        if lv is not None:
            day, poc, vah, val = lv
            # Where the cursor sits relative to that session's value area
            # -- the thing being compared when the pointer moves across.
            where = ("above VA" if y > vah else
                     "below VA" if y < val else "in VA")
            txt = (f"{y:.2f}   {day}   {where}\n"
                   f"VAH {vah:.2f}  POC {poc:.2f}  VAL {val:.2f}")
        self._read.setText(txt)
        self._read.setPos(x, y)
        self._read.setVisible(True)

    # ── data ────────────────────────────────────────────────────────────
    def set_code(self, code: str) -> None:
        if code and code != self._code:
            self._code = code
            self.reload()

    def _sessions(self) -> list[tuple[str, pd.DataFrame]]:
        """The last N trading days of 1m bars, oldest first."""
        from feeds.fetcher import fetch_klines

        want = int(self._days.value())
        # Calendar days to ask for: weekends and holidays are not sessions,
        # so a fixed multiple would come up short over a long break.
        span = want * 3 + 8
        end = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
        start = end - pd.Timedelta(days=span)
        k = fetch_klines(self._code, "1m", start.strftime("%Y-%m-%d"),
                         end.strftime("%Y-%m-%d"))
        if k is None or k.empty:
            return []
        k = k.copy()
        k["t"] = pd.to_datetime(k["time_key"])
        k = k.sort_values("t").drop_duplicates("t").set_index("t")
        out = []
        # The trading day runs 20:00 ET -> next 20:00, matching
        # _trading_day_of_bucket in trade_viewer_qt and the boundary CVD
        # resets on. Adding four hours before taking the date applies that
        # rule: 19:59 stays put, 20:00 rolls into the next day. Grouping by
        # calendar date would cut the overnight session in half.
        tday = (k.index + pd.Timedelta(hours=4)).normalize()
        for d, g in k.groupby(tday):
            if self._rth.isChecked():
                hh = g.index.hour + g.index.minute / 60
                g = g[(hh > RTH_LO) & (hh <= RTH_HI)]
            # A handful of bars is a holiday stub, not a session; its
            # "value area" would be one bar wide and would drag the scale.
            if len(g) < 60:
                continue
            out.append((str(pd.Timestamp(d).date()), g))
        return out[-want:]

    # ── drawing ─────────────────────────────────────────────────────────
    def _clear(self) -> None:
        for it in self._items:
            self._plot.removeItem(it)
        self._items.clear()
        self._levels.clear()

    def _add(self, item) -> None:
        self._plot.addItem(item)
        self._items.append(item)

    def reload(self) -> None:
        from strategy.smc.fvg import compute_volume_profile
        # Imported here rather than at module scope: trade_viewer_qt imports
        # THIS module lazily when the button is pressed, so a top-level
        # import back into it would be a cycle.
        from analysis.trade_viewer_qt import _compute_poc_vah_val

        self._clear()
        try:
            sess = self._sessions()
        except Exception as exc:
            self._status.setText(f"  fetch failed: {exc}")
            return
        if not sess:
            self._status.setText("  no sessions available")
            return

        nb = int(self._bins.value())
        va_pct = self._va.value() / 100.0
        slab = 1.0                      # one session occupies x in [i, i+1)
        prev = None
        lo_all, hi_all = [], []
        grid = None
        if self._shared.isChecked():
            p_lo = min(float(g.low.min()) for _, g in sess)
            p_hi = max(float(g.high.max()) for _, g in sess)
            if p_hi > p_lo:
                grid = np.linspace(p_lo, p_hi, nb + 1)
        for i, (day, g) in enumerate(sess):
            edges, vols = compute_volume_profile(g, n_bins=nb, edges=grid)
            if edges is None or not np.any(vols):
                continue
            centers = (edges[:-1] + edges[1:]) / 2
            poc, vah, val = _compute_poc_vah_val(centers, vols, va_pct)
            h = float(edges[1] - edges[0])   # same for every slab on a
            #                                    shared grid, per-day otherwise
            w = vols / vols.max() * slab * BAR_SPAN

            self._add(pg.BarGraphItem(
                x0=np.full(len(centers), float(i)), y=centers,
                width=w, height=h * 0.92,
                brush=_qc(_PROF, 210), pen=None))

            # VAH / VAL / POC as segments over this slab only. A
            # LinearRegionItem would have been easier but it spans the whole
            # view, so each session's band would run across its neighbours.
            for p, col, wd, nm in ((vah, _VA, 1, "VAH"), (val, _VA, 1, "VAL"),
                                   (poc, _GOLD, 2, "POC")):
                self._add(pg.PlotCurveItem(
                    x=[i, i + slab * BAR_SPAN], y=[p, p],
                    pen=pg.mkPen(_qc(col), width=wd)))
                if self._labels.isChecked():
                    # Anchored at the right end of its own segment, and
                    # above the line, so three labels on one slab do not
                    # sit on each other or on the next slab's bars.
                    t = pg.TextItem(text=f"{nm} {p:.2f}", color=_qc(col),
                                    anchor=(1.0, 1.0))
                    t.setFont(QFont("Monospace", 7))
                    self._add(t)
                    t.setPos(i + slab * BAR_SPAN, p)

            # Price path across the slab, so the profile has its sequence.
            c = g["close"].to_numpy(float)
            xs = np.linspace(i, i + slab * BAR_SPAN, len(c))
            self._add(pg.PlotCurveItem(
                x=xs, y=c, pen=pg.mkPen(_qc(_PATH, 150), width=1)))

            lbl = pg.TextItem(text=day[5:], color=_qc(_GREY), anchor=(0, 1))
            lbl.setFont(QFont("Monospace", 8))
            self._add(lbl)
            lbl.setPos(i, float(edges[0]))

            if prev is not None and self._join.isChecked():
                pi, (pvah, pval) = prev
                for a, b, col in ((pvah, vah, _VA), (pval, val, _VA)):
                    ln = pg.PlotCurveItem(
                        x=[pi + slab * BAR_SPAN, i], y=[a, b],
                        pen=pg.mkPen(_qc(col, 110), width=1,
                                     style=Qt.PenStyle.DashLine))
                    self._add(ln)
            prev = (i, (vah, val))
            self._levels[i] = (day, poc, vah, val)
            lo_all.append(float(edges[0]))
            hi_all.append(float(edges[-1]))

        if lo_all:
            pad = (max(hi_all) - min(lo_all)) * 0.04 or 0.1
            self._plot.setYRange(min(lo_all) - pad, max(hi_all) + pad,
                                 padding=0)
            self._plot.setXRange(-0.1, len(sess) + 0.1, padding=0)
        ax = self._plot.getPlotItem().getAxis("bottom")
        ax.setTicks([[(i + BAR_SPAN / 2, d[5:]) for i, (d, _) in
                      enumerate(sess)]])
        first, last = sess[0][0], sess[-1][0]
        self._status.setText(
            f"  {self._code}   {len(sess)} sessions   {first} .. {last}"
            f"   gold = POC, blue = VA edges")


def main() -> None:
    app = QApplication(sys.argv)
    w = MultiVPWindow()
    if len(sys.argv) > 1:
        w.set_code(sys.argv[1])
    w.reload()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
