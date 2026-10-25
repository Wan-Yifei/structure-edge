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
        self.setWindowTitle("多日成交分布")
        self.resize(1180, 680)
        self._code = "US.SOXL"
        self._items: list = []

        tb = QToolBar()
        tb.setMovable(False)
        self.addToolBar(tb)

        tb.addWidget(QLabel("  天数 "))
        self._days = QSpinBox()
        self._days.setRange(2, 30)
        self._days.setValue(6)
        self._days.setFixedWidth(52)
        self._days.setToolTip("往回取几个交易日")
        self._days.valueChanged.connect(self.reload)
        tb.addWidget(self._days)

        tb.addSeparator()
        tb.addWidget(QLabel("  价格档 "))
        self._bins = QSpinBox()
        self._bins.setRange(20, 300)
        self._bins.setValue(80)
        self._bins.setSingleStep(10)
        self._bins.setFixedWidth(60)
        self._bins.setToolTip(
            "每个交易日的分布切成几档。档太多，单日的分布会碎成毛刺；\n"
            "太少，价值区的边界会被一个宽档吃掉。")
        self._bins.valueChanged.connect(self.reload)
        tb.addWidget(self._bins)

        tb.addSeparator()
        tb.addWidget(QLabel("  VA% "))
        self._va = QSpinBox()
        self._va.setRange(50, 95)
        self._va.setValue(70)
        self._va.setSingleStep(5)
        self._va.setFixedWidth(52)
        self._va.setToolTip("价值区覆盖多少成交量，标准是 70%")
        self._va.valueChanged.connect(self.reload)
        tb.addWidget(self._va)

        tb.addSeparator()
        self._rth = QCheckBox("只看盘中")
        self._rth.setChecked(True)
        self._rth.setToolTip(
            "只用 09:30–16:00 的 K 线。盘前盘后成交稀薄，\n"
            "混进来会把价值区拉得很宽。")
        self._rth.stateChanged.connect(self.reload)
        tb.addWidget(self._rth)

        self._join = QCheckBox("连接价值区")
        self._join.setChecked(True)
        self._join.setToolTip("相邻交易日的 VAH / VAL 之间连虚线")
        self._join.stateChanged.connect(self.reload)
        tb.addWidget(self._join)

        tb.addSeparator()
        btn = QPushButton("↻")
        btn.setFixedWidth(30)
        btn.setToolTip("重新取数")
        btn.clicked.connect(self.reload)
        tb.addWidget(btn)

        self._status = QLabel("  ")
        tb.addWidget(self._status)

        self._plot = pg.PlotWidget()
        self._plot.setBackground(_BG)
        self._plot.showGrid(x=False, y=True, alpha=0.12)
        self._plot.getPlotItem().getAxis("left").setTextPen(_qc(_FG))
        self._plot.getPlotItem().getAxis("bottom").setTextPen(_qc(_FG))
        self._plot.setLabel("left", "价格", color=_FG)
        self.setCentralWidget(self._plot)

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
        for d, g in k.groupby(k.index.normalize()):
            if self._rth.isChecked():
                hh = g.index.hour + g.index.minute / 60
                g = g[(hh > RTH_LO) & (hh <= RTH_HI)]
            # A handful of bars is a holiday stub, not a session; its
            # "value area" would be one bar wide and would drag the scale.
            if len(g) < 60:
                continue
            out.append((str(d.date()), g))
        return out[-want:]

    # ── drawing ─────────────────────────────────────────────────────────
    def _clear(self) -> None:
        for it in self._items:
            self._plot.removeItem(it)
        self._items.clear()

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
            self._status.setText(f"  取数失败：{exc}")
            return
        if not sess:
            self._status.setText("  没有可用的交易日")
            return

        nb = int(self._bins.value())
        va_pct = self._va.value() / 100.0
        slab = 1.0                      # one session occupies x in [i, i+1)
        prev = None
        lo_all, hi_all = [], []
        for i, (day, g) in enumerate(sess):
            edges, vols = compute_volume_profile(g, n_bins=nb)
            if edges is None or not np.any(vols):
                continue
            centers = (edges[:-1] + edges[1:]) / 2
            poc, vah, val = _compute_poc_vah_val(centers, vols, va_pct)
            h = float(edges[1] - edges[0])
            w = vols / vols.max() * slab * BAR_SPAN

            self._add(pg.BarGraphItem(
                x0=np.full(len(centers), float(i)), y=centers,
                width=w, height=h * 0.92,
                brush=_qc(_PROF, 210), pen=None))

            # VAH / VAL / POC as segments over this slab only. A
            # LinearRegionItem would have been easier but it spans the whole
            # view, so each session's band would run across its neighbours.
            for p, col, wd in ((vah, _VA, 1), (val, _VA, 1), (poc, _GOLD, 2)):
                self._add(pg.PlotCurveItem(
                    x=[i, i + slab * BAR_SPAN], y=[p, p],
                    pen=pg.mkPen(_qc(col), width=wd)))

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
            f"  {self._code}　{len(sess)} 个交易日　{first} .. {last}"
            f"　金线 POC，蓝线 VA 上下沿")


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
