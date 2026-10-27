"""The multi-session profile window draws what it claims to.

A Qt window that opens without raising proves nothing -- the failure to
catch is an empty plot, or a value area that does not belong to the day
it is drawn over. So these count what reached the plot and check the
levels against the session they came from.

Klines are synthetic and fetch_klines is replaced, so the tests neither
touch the network nor depend on which days happen to be cached.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app
    app.processEvents()


def _session(day, base, n=390, amp=1.5):
    """One RTH session of 1m bars with a drift and a volume hump, so the
    profile has a real peak rather than a flat distribution.

    `amp` varies the day's range. Sessions of IDENTICAL range would make
    the per-session and shared grids agree by accident, and the control
    that proves the shared grid is doing anything would pass vacuously."""
    t = pd.date_range(f"{day} 09:31", periods=n, freq="1min")
    i = np.arange(n, dtype=float)
    c = base + np.sin(i / 40) * amp + i * 0.002
    vol = 1000 + 4000 * np.exp(-((i - n / 2) ** 2) / (2 * (n / 8) ** 2))
    return pd.DataFrame(
        {"time_key": t.strftime("%Y-%m-%d %H:%M:%S"),
         "open": c, "high": c + 0.25, "low": c - 0.25, "close": c,
         "volume": vol})


@pytest.fixture
def win(qapp, monkeypatch):
    days = ["2026-10-19", "2026-10-20", "2026-10-21"]
    frames = [_session(d, 100 + k * 2, amp=1.0 + k * 0.8)
              for k, d in enumerate(days)]
    import feeds.fetcher as fetcher
    monkeypatch.setattr(fetcher, "fetch_klines",
                        lambda *a, **k: pd.concat(frames, ignore_index=True))
    from analysis.multi_vp_window import MultiVPWindow
    w = MultiVPWindow()
    w._days.blockSignals(True)
    w._days.setValue(3)
    w._days.blockSignals(False)
    w.reload()
    yield w, days
    w.close()


def kinds(w):
    out = {}
    for it in w._items:
        out[type(it).__name__] = out.get(type(it).__name__, 0) + 1
    return out


class TestItDraws:
    def test_one_profile_per_session(self, win):
        w, days = win
        assert kinds(w).get("BarGraphItem", 0) == len(days)

    def test_levels_and_path_and_joins(self, win):
        """3 level lines + 1 price path a session, 2 joins a gap. Counting
        them is how an empty or half-drawn panel gets caught."""
        w, days = win
        want = len(days) * 4 + (len(days) - 1) * 2
        assert kinds(w).get("PlotCurveItem", 0) == want

    def test_joins_can_be_turned_off(self, win):
        w, days = win
        w._join.setChecked(False)
        w.reload()
        assert kinds(w).get("PlotCurveItem", 0) == len(days) * 4

    def test_a_redraw_does_not_pile_items_up(self, win):
        """reload() has to clear first; without that the plot accumulates a
        fresh copy on every parameter change and slows to a halt."""
        w, _ = win
        before = len(w._items)
        w.reload()
        w.reload()
        assert len(w._items) == before


class TestLevelsBelongToTheirSession:
    def test_va_is_ordered_and_inside_the_day(self, win):
        w, days = win
        from analysis.trade_viewer_qt import _compute_poc_vah_val
        from strategy.smc.fvg import compute_volume_profile
        for day, g in w._sessions():
            edges, vols = compute_volume_profile(g, n_bins=w._bins.value())
            c = (edges[:-1] + edges[1:]) / 2
            poc, vah, val = _compute_poc_vah_val(c, vols,
                                                 w._va.value() / 100)
            assert val <= poc <= vah
            assert float(g.low.min()) - 0.01 <= val
            assert vah <= float(g.high.max()) + 0.01

    def test_a_wider_value_area_contains_the_narrower_one(self, win):
        """70% to 90% can only grow. If it does not, the expansion is
        anchored somewhere it should not be."""
        w, _ = win
        from analysis.trade_viewer_qt import _compute_poc_vah_val
        from strategy.smc.fvg import compute_volume_profile
        day, g = w._sessions()[0]
        edges, vols = compute_volume_profile(g, n_bins=w._bins.value())
        c = (edges[:-1] + edges[1:]) / 2
        _, v70, l70 = _compute_poc_vah_val(c, vols, 0.70)
        _, v90, l90 = _compute_poc_vah_val(c, vols, 0.90)
        assert v90 >= v70 and l90 <= l70

    def test_each_session_keeps_its_own_x_slab(self, win):
        """The levels are segments, not full-width lines -- a session's VA
        must not be drawn across its neighbours."""
        w, days = win
        spans = [(min(it.xData), max(it.xData)) for it in w._items
                 if type(it).__name__ == "PlotCurveItem"
                 and it.xData is not None and len(it.xData) == 2
                 and it.yData[0] == it.yData[1]]
        assert spans, "no horizontal level segments were drawn"
        for lo, hi in spans:
            assert hi - lo < 1.0, f"a level line spans a whole slab: {lo}-{hi}"


class TestEmptyInput:
    def test_no_data_says_so_instead_of_raising(self, qapp, monkeypatch):
        import feeds.fetcher as fetcher
        monkeypatch.setattr(fetcher, "fetch_klines",
                            lambda *a, **k: pd.DataFrame())
        from analysis.multi_vp_window import MultiVPWindow
        w = MultiVPWindow()
        w.reload()
        assert "no sessions" in w._status.text()
        assert not w._items
        w.close()

    def test_a_stub_session_is_skipped(self, qapp, monkeypatch):
        """A holiday half-hour is not a session; its one-bar value area
        would drag the price scale over everything else."""
        import feeds.fetcher as fetcher
        full = _session("2026-10-20", 100)
        stub = _session("2026-10-21", 140, n=20)
        monkeypatch.setattr(
            fetcher, "fetch_klines",
            lambda *a, **k: pd.concat([full, stub], ignore_index=True))
        from analysis.multi_vp_window import MultiVPWindow
        w = MultiVPWindow()
        w.reload()
        assert [d for d, _ in w._sessions()] == ["2026-10-20"]
        w.close()


class TestTradingDayBoundary:
    """20:00 ET starts the next trading day, as everywhere else in the
    project (_trading_day_of_bucket, and the bar CVD resets on).

    Only visible with extended hours shown: under RTH the session sits
    inside one calendar date either way, so the default view cannot catch
    a calendar-date grouping.
    """

    @staticmethod
    def _overnight():
        """Evening of the 20th through the next afternoon -- one trading
        day by the project's rule, two calendar dates."""
        ev = pd.date_range("2026-10-20 20:00", periods=120, freq="1min")
        ng = pd.date_range("2026-10-21 04:00", periods=120, freq="1min")
        rt = pd.date_range("2026-10-21 09:31", periods=200, freq="1min")
        t = ev.append(ng).append(rt)
        c = 100 + np.linspace(0, 2, len(t))
        return pd.DataFrame(
            {"time_key": t.strftime("%Y-%m-%d %H:%M:%S"),
             "open": c, "high": c + 0.2, "low": c - 0.2, "close": c,
             "volume": np.full(len(t), 1000.0)})

    def test_the_overnight_session_is_not_split(self, qapp, monkeypatch):
        import feeds.fetcher as fetcher
        monkeypatch.setattr(fetcher, "fetch_klines",
                            lambda *a, **k: self._overnight())
        from analysis.multi_vp_window import MultiVPWindow
        w = MultiVPWindow()
        w._rth.setChecked(False)
        w.reload()
        days = [d for d, _ in w._sessions()]
        assert days == ["2026-10-21"], (
            f"20:00 之后应归入次日，实得 {days}")
        n = len(w._sessions()[0][1])
        assert n == 440, f"该交易日应有 440 根，实得 {n}"
        w.close()

    def test_rth_view_is_unaffected(self, qapp, monkeypatch):
        """The default view must not move: 09:30-16:00 is the same session
        under either rule, and this is the case that was already right."""
        import feeds.fetcher as fetcher
        monkeypatch.setattr(fetcher, "fetch_klines",
                            lambda *a, **k: self._overnight())
        from analysis.multi_vp_window import MultiVPWindow
        w = MultiVPWindow()
        w._rth.setChecked(True)
        w.reload()
        days = [d for d, _ in w._sessions()]
        assert days == ["2026-10-21"]
        w.close()


class TestSharedGrid:
    def test_every_session_gets_the_same_bin_width(self, win):
        """What the shared grid is for. Sessions differ in range, so a
        per-session grid gives each a different bin width and the same
        bin index means a different price in each slab."""
        w, _ = win
        from strategy.smc.fvg import compute_volume_profile
        import numpy as np
        sess = w._sessions()
        p_lo = min(float(g.low.min()) for _, g in sess)
        p_hi = max(float(g.high.max()) for _, g in sess)
        grid = np.linspace(p_lo, p_hi, w._bins.value() + 1)
        widths = set()
        for _, g in sess:
            e, _v = compute_volume_profile(g, n_bins=w._bins.value(),
                                           edges=grid)
            widths.add(round(float(e[1] - e[0]), 9))
        assert len(widths) == 1, f"共享网格下档宽应一致，实得 {widths}"

    def test_per_session_grids_differ(self, win):
        """The control: without a shared grid the widths must differ, or
        the previous test proves nothing."""
        w, _ = win
        from strategy.smc.fvg import compute_volume_profile
        widths = set()
        for _, g in w._sessions():
            e, _v = compute_volume_profile(g, n_bins=w._bins.value())
            widths.add(round(float(e[1] - e[0]), 9))
        assert len(widths) > 1

    def test_a_supplied_grid_keeps_the_volume(self, win):
        """Bars outside the grid are clipped into the end bins, not
        dropped -- otherwise a session near the edge of the window would
        quietly lose volume and its POC would move."""
        from strategy.smc.fvg import compute_volume_profile
        import numpy as np
        w, _ = win
        _, g = w._sessions()[0]
        # A grid that stops short of the session on both sides.
        lo, hi = float(g.low.min()), float(g.high.max())
        mid = (lo + hi) / 2
        grid = np.linspace(mid - 0.2, mid + 0.2, 21)
        _e, v = compute_volume_profile(g, n_bins=20, edges=grid)
        assert abs(v.sum() - float(g.volume.sum())) < 1.0

    def test_turning_it_off_still_draws(self, win):
        w, days = win
        w._shared.setChecked(False)
        w.reload()
        assert kinds(w).get("BarGraphItem", 0) == len(days)


class TestLevelLabels:
    def test_three_labels_a_session(self, win):
        """The in-plot day tag was removed -- the axis already carries the
        date, and two rows of the same thing read as a bug."""
        w, days = win
        assert kinds(w).get("TextItem", 0) == len(days) * 3

    def test_they_can_be_turned_off(self, win):
        w, _ = win
        w._labels.setChecked(False)
        w.reload()
        assert kinds(w).get("TextItem", 0) == 0

class TestLabelPlacement:
    def test_crowded_labels_are_pushed_apart(self, qapp, monkeypatch):
        """POC and VAL land within a few cents often enough that the two
        labels overlapped and neither could be read."""
        import feeds.fetcher as fetcher
        # A session that spends nearly all its time in a narrow band, so
        # POC and the value-area edges end up almost on top of each other.
        t = pd.date_range("2026-10-20 09:31", periods=390, freq="1min")
        c = 100 + np.where(np.arange(390) % 50 == 0, 0.6, 0.0)
        frame = pd.DataFrame(
            {"time_key": t.strftime("%Y-%m-%d %H:%M:%S"),
             "open": c, "high": c + 0.02, "low": c - 0.02, "close": c,
             "volume": np.full(390, 1000.0)})
        monkeypatch.setattr(fetcher, "fetch_klines", lambda *a, **k: frame)
        from analysis.multi_vp_window import MultiVPWindow
        w = MultiVPWindow()
        w.resize(900, 600)
        w.show()
        qapp.processEvents()
        w.reload()
        w._place_labels()
        ys = sorted(t_.pos().y() for t_, *_ in w._tags)
        vb = w._plot.getPlotItem().vb
        (_, _), (y0, y1) = vb.viewRange()
        gap = (y1 - y0) * (11.0 / max(vb.height(), 1))
        for a, b in zip(ys, ys[1:]):
            assert b - a >= gap * 0.99, f"标签仍然重叠：{a} 与 {b}"
        w.close()

    def test_a_clear_label_stays_on_its_level(self, win):
        """Only the crowded ones move. Spacing all three evenly would put
        every label slightly wrong to fix a problem two do not have.

        "Stays" allows the few pixels of clearance the label is lifted by
        so its own line does not run through the digits -- a nudge is a
        whole label height, several times that, so the two do not blur."""
        w, _ = win
        w._place_labels()
        vb = w._plot.getPlotItem().vb
        (_, _), (y0, y1) = vb.viewRange()
        px = (y1 - y0) / max(vb.height(), 1)
        lowest = min(w._tags, key=lambda r: r[2])
        off = lowest[0].pos().y() - lowest[2]
        assert 0 <= off <= 4 * px, f"最低的标签被挪了 {off / px:.1f}px"


def _rgb(hex_: str) -> tuple:
    return tuple(int(hex_[k:k + 2], 16) for k in (1, 3, 5))


def _longest_run(flags) -> int:
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def _row_runs(img, box, ref, tol=36):
    """Longest unbroken run of `ref`-coloured pixels per row of `box`, in
    pixels. The caller divides by the width where a fraction is what it
    means, and reads the pixels straight where the width is arbitrary."""
    x0, y0, x1, y1 = box
    out = []
    for yy in range(max(0, y0), min(img.height(), y1 + 1)):
        flags = []
        for xx in range(max(0, x0), min(img.width(), x1 + 1)):
            c = img.pixelColor(xx, yy)
            flags.append(all(abs(a - b) <= tol for a, b in
                             zip((c.red(), c.green(), c.blue()), ref)))
        out.append(_longest_run(flags))
    return out


def _label_box(w, t):
    """A label's own rectangle, in the plot widget's pixels."""
    r = t.mapRectToScene(t.boundingRect())
    tl = w._plot.mapFromScene(r.topLeft())
    br = w._plot.mapFromScene(r.bottomRight())
    x0, x1 = sorted((int(tl.x()), int(br.x())))
    y0, y1 = sorted((int(tl.y()), int(br.y())))
    return x0, y0, x1, y1


def _level_row(w, price):
    """The pixel row a price sits on in the plot widget."""
    from PyQt6.QtCore import QPointF
    vb = w._plot.getPlotItem().vb
    pt = w._plot.mapFromScene(vb.mapViewToScene(QPointF(0.0, float(price))))
    return int(pt.y())


class TestTheLineDoesNotStrikeThroughItsLabel:
    """A level label with its own line through the digits cannot be read.

    The glyphs wear the same colour as the line, so counting coloured
    pixels inside the label tells a strike-through from the digits not at
    all. What separates them is continuity: a strike-through is one
    unbroken run across the whole label and no row of 7pt monospace
    digits is. So these measure, per row, the longest unbroken run.
    """

    @pytest.fixture
    def shot(self, win, qapp):
        w, days = win
        w.resize(1100, 700)
        w.show()
        qapp.processEvents()
        w.reload()
        qapp.processEvents()
        w._place_labels()
        qapp.processEvents()
        yield w, w._plot.grab().toImage()

    def test_no_row_inside_a_label_is_an_unbroken_line(self, shot):
        w, img = shot
        from analysis.multi_vp_window import _GOLD, _VA
        gold, va = _rgb(_GOLD), _rgb(_VA)
        worst = []
        for t, _x, p, i in w._tags:
            txt = t.toPlainText()
            ref = gold if txt.startswith("POC") else va
            box = _label_box(w, t)
            runs = _row_runs(img, box, ref)
            assert runs, f"{txt} 的框落在画面外，量不到"
            worst.append((max(runs) / max(box[2] - box[0] + 1, 1), txt, i))
        hi, txt, i = max(worst)
        assert hi <= 0.35, (f"第 {i} 段的 {txt} 有一行 {hi:.0%} 是连续线色"
                            f"——线从数字中间穿过去了")

    def test_the_probe_can_see_a_line_where_there_is_one(self, shot):
        """The control. The same measurement along the level itself has to
        come back as a long run -- otherwise the test above passes because
        nothing is being measured.

        In pixels, not as a fraction: the price path crosses the level a
        few times per slab, so the span is broken into pieces whatever its
        width, and what matters is that one piece is far longer than the
        5-6px a digit's stroke gives."""
        from analysis.multi_vp_window import _VA
        w, img = shot
        va = _rgb(_VA)
        best = 0
        for t, _x, p, i in w._tags:
            if not t.toPlainText().startswith("VAH"):
                continue
            lx0, _, _, _ = _label_box(w, t)
            row = _level_row(w, p)
            box = (max(lx0 - 300, 0), row - 2, max(lx0 - 2, 1), row + 2)
            best = max(best, max(_row_runs(img, box, va)))
        assert best >= 25, (f"沿着 VAH 线量到的最长连续段只有 {best}px，"
                            f"探针根本看不见线")

    def test_every_label_carries_a_backdrop(self, shot):
        """The widest bar of a session ends exactly where the labels are
        anchored and they extend LEFT from there, so the profile is
        behind the digits by construction, not by luck -- which is what
        the backdrop is for.

        A configuration check and not a pixel one on purpose. The share
        of a label's box that stays background does separate the two
        cases -- 87% with the backdrop against 61% without, measured
        offscreen -- but the same pair comes back 77% against 65% on a
        real display, because subpixel antialiasing colours the glyph
        edges. A threshold between 65 and 77 would be asserting the
        font-rendering stack. The z-order test below is what makes the
        backdrop land over the bars rather than under them.
        """
        from analysis.multi_vp_window import _BG
        w, _img = shot
        bg = _rgb(_BG)
        from PyQt6.QtCore import Qt as _Qt
        for t, _x, _p, i in w._tags:
            br = t.fill
            assert br is not None and br.style() != _Qt.BrushStyle.NoBrush, (
                f"第 {i} 段的 {t.toPlainText()} 没有底色")
            c = br.color()
            assert (c.red(), c.green(), c.blue()) == bg, (
                f"底色不是画布背景色：{c.name()}")
            assert c.alpha() >= 200, f"底色太透：alpha={c.alpha()}"

    def test_a_label_is_painted_over_everything_it_names(self, shot):
        """The price path and the next session's bars are added after the
        labels, so equal z-values let them paint over the digits."""
        w, _img = shot
        import pyqtgraph as pg
        tags = [t.zValue() for t, *_ in w._tags]
        drawn = [it.zValue() for it in w._items
                 if isinstance(it, (pg.PlotCurveItem, pg.BarGraphItem))]
        assert tags and drawn
        assert min(tags) > max(drawn)

    def test_the_label_clears_its_own_level(self, shot):
        """Geometry, independent of the pixels: the box sits above the
        price it names, not centred on it."""
        w, _img = shot
        for t, _x, p, i in w._tags:
            _x0, _y0, _x1, y1 = _label_box(w, t)
            row = _level_row(w, p)
            # y grows downward: the box's bottom edge is above the level.
            assert y1 <= row, (f"第 {i} 段的 {t.toPlainText()} 下沿在 y={y1}，"
                               f"价位在 y={row}——框骑在线上")
