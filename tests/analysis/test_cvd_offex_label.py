"""The off-exchange CVD figure has to be readable, which means not at its
own value.

Both series are drawn in one short panel, and the whole reason the second
one is there is that it DIVERGES from the first -- that divergence is the
reading. So placing its label at its own y put it off the bottom edge
exactly when it had something to say: reported with the on-exchange
figure at +7.19M and off at -4.49M, 11.7M apart, the label was simply
gone from a panel a few hundred pixels tall.

The repair parks it beside the on-exchange figure and clears that text by
its MEASURED pixel width. The two properties worth pinning are therefore:

  * the off label sits at the on-exchange label's y, never at its own
  * the two do not overlap

The second one is weaker than it looks, and the file says so where it
matters: the horizontal spacing was never the reported defect. The old
gap of 11% of the view range measured about 157 px against a 74-96 px
label and cleared it comfortably, and a fraction of the range is
zoom-invariant in pixels anyway. Clearing by the measured text width is
a robustness change; the case that would separate the two -- a panel
narrow enough that 11% is less than the text -- cannot be built under
the offscreen platform, which does not propagate a window resize to the
view box.

Driven through the real hover handler on a real viewer, so a change to
how either label is positioned is caught rather than a change to a
helper nobody calls.
"""

import os
from datetime import datetime, timedelta

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtWidgets import QApplication   # noqa: E402

CM = 5
DAY0 = datetime(2026, 9, 21, 10, 0)
ON_VAL = 7.19e6          # the figures from the report
OFF_VAL = -4.49e6


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app
    # No processEvents() at teardown -- see test_absorption_marker.py


@pytest.fixture(scope="module")
def viewer(qapp):
    from analysis.trade_viewer_qt import TradeViewerQt

    v = TradeViewerQt()
    v.resize(1400, 900)
    v.show()
    for _ in range(30):
        qapp.processEvents()
    yield v
    v.close()
    qapp.processEvents()


@pytest.fixture
def wired(viewer, qapp):
    """A viewer with both CVD series present and the panel visible."""
    import pandas as pd

    n = 40
    rows = []
    for i in range(n):
        ts = (DAY0 + timedelta(minutes=CM * (i + 1))).strftime("%Y-%m-%d %H:%M")
        rows.append((ts, 100.0, 101.0, 99.0, 100.0, 1000.0))
    df = pd.DataFrame(rows, columns=["time_key", "open", "high", "low",
                                     "close", "volume"])
    # `_klines` is what the hover handler reads to find the bar; `_df` is
    # not on that path, and setting only it gave a green-looking fixture
    # that never reached the label code at all.
    viewer._klines = df
    viewer._cvd_arr = np.full(n, ON_VAL, dtype=float)
    viewer._offex_arr = np.full(n, OFF_VAL, dtype=float)
    viewer._plot_cvd.setVisible(True)
    # Pin the y range: the panel otherwise auto-ranges onto whatever the
    # curve holds, which would quietly remove the divergence this fixture
    # exists to reproduce.
    viewer._plot_cvd.vb.enableAutoRange(axis="y", enable=False)
    viewer._plot_cvd.setYRange(-1.0e6, 11.0e6, padding=0)
    for _ in range(5):
        qapp.processEvents()
    return viewer


def _hover(v, qapp, idx: int, xlo: float, xhi: float):
    """Drive the real handler with a real scene position.

    `_on_mouse_move` takes the QPointF that `scene.sigMouseMoved` emits
    and decides which panel it is in from the scene rectangles, so the
    test has to hand it a point that genuinely lands inside the CVD
    panel. Calling an inner helper instead would test a function the
    application does not use on this path.
    """
    from PyQt6.QtCore import QPointF

    v._plot_cvd.setXRange(xlo, xhi, padding=0)
    for _ in range(3):
        qapp.processEvents()
    ylo, yhi = v._plot_cvd.vb.viewRange()[1]
    pt = v._plot_cvd.vb.mapViewToScene(QPointF(float(idx), (ylo + yhi) / 2))
    assert v._plot_cvd.sceneBoundingRect().contains(pt), (
        "the test point missed the CVD panel, so nothing below is "
        "measuring the hover path")
    v._on_mouse_move(pt)
    for _ in range(3):
        qapp.processEvents()


def _rects(v):
    a = v._cvd_label.mapRectToScene(v._cvd_label.boundingRect())
    b = v._offex_label.mapRectToScene(v._offex_label.boundingRect())
    return a, b


class TestTheOffLabelIsWhereItCanBeRead:

    def test_it_sits_at_the_on_exchange_value_not_its_own(self, wired, qapp):
        _hover(wired, qapp, 20, 0, 40)
        assert wired._offex_label.isVisible()
        y_off = wired._offex_label.pos().y()
        y_on = wired._cvd_label.pos().y()
        assert y_off == pytest.approx(y_on), (
            f"off label is at y={y_off:,.0f}, on-exchange at y={y_on:,.0f}; "
            f"placing it at its own value ({OFF_VAL:,.0f}) is what put it "
            f"off the panel")

    def test_its_own_value_really_would_be_off_the_panel(self, wired, qapp):
        """Guards the test above from passing vacuously: if the fixture
        ever stopped producing a wide divergence, the first test would
        hold for a reason that has nothing to do with the fix."""
        _hover(wired, qapp, 20, 0, 40)
        ylo, yhi = wired._plot_cvd.vb.viewRange()[1]
        assert not (ylo <= OFF_VAL <= yhi), (
            f"this fixture no longer reproduces the complaint: "
            f"{OFF_VAL:,.0f} is inside the panel range "
            f"[{ylo:,.0f}, {yhi:,.0f}]")

    @pytest.mark.parametrize("xlo,xhi", [(0, 40), (10, 25), (19, 21)])
    def test_the_two_labels_never_overlap(self, wired, qapp, xlo, xhi):
        _hover(wired, qapp, 20, xlo, xhi)
        a, b = _rects(wired)
        assert b.left() >= a.right(), (
            f"at x range {xhi - xlo}: on-exchange ends at {a.right():.1f}px, "
            f"off starts at {b.left():.1f}px -- they overlap by "
            f"{a.right() - b.left():.1f}px")

    def test_the_gap_does_not_change_with_zoom(self, wired, qapp):
        """Zoom must not move the labels relative to each other.

        Note this does NOT distinguish a measured width from a fraction
        of the view range -- a fraction is the same pixel distance at any
        zoom, because the range and the pixels scale together. The two
        tests below are the ones that tell them apart.
        """
        gaps = []
        for xlo, xhi in ((0, 40), (19, 21)):
            _hover(wired, qapp, 20, xlo, xhi)
            a, b = _rects(wired)
            gaps.append(b.left() - a.right())
        assert gaps[0] == pytest.approx(gaps[1], abs=1.0), (
            f"gap moved from {gaps[0]:.1f}px to {gaps[1]:.1f}px across a "
            f"20x zoom")

    def test_a_longer_number_still_clears(self, wired, qapp):
        """A property worth holding, but it does NOT discriminate.

        Measured: the panel is about 1439 px here and the old 11%-of-
        range gap was about 157 px, against a label 74 px wide at
        "+7.19M" and 96 px at "+123.46M". Both implementations clear
        both. This test pins the property; it does not prove the measured
        width is needed, and the panel-width case that WOULD separate
        them cannot be built offscreen -- see the class docstring.
        """
        wired._cvd_arr = np.full(len(wired._cvd_arr), 123.456e6, dtype=float)
        try:
            _hover(wired, qapp, 20, 0, 40)
            assert wired._cvd_label.toPlainText().strip() == "+123.46M"
            a, b = _rects(wired)
            assert b.left() >= a.right(), (
                f"'+123.46M' ends at {a.right():.1f}px and the off label "
                f"starts at {b.left():.1f}px -- overlapping by "
                f"{a.right() - b.left():.1f}px")
        finally:
            wired._cvd_arr = np.full(len(wired._cvd_arr), ON_VAL, dtype=float)

    def test_the_panel_width_case_cannot_be_tested_here(self, wired, qapp):
        """Record why the discriminating case is missing, in runnable form.

        A narrow panel is what would separate a measured width from a
        fraction of the range. Offscreen it cannot be produced: resizing
        the window leaves the view box at its original width ("This
        plugin does not support propagateSizeHints()"). An earlier
        version of this file resized to 520 px and asserted no overlap --
        it passed under BOTH implementations and was measuring nothing.
        This asserts the limitation instead, so that if the platform ever
        starts propagating the resize, this test fails and tells whoever
        is reading to write the real one.
        """
        before = wired.width()
        try:
            w0 = wired._plot_cvd.vb.boundingRect().width()
            wired.resize(520, 900)
            for _ in range(8):
                qapp.processEvents()
            w1 = wired._plot_cvd.vb.boundingRect().width()
            assert w1 == pytest.approx(w0, abs=2.0), (
                f"the view box now follows the window ({w0:.0f} -> "
                f"{w1:.0f} px), so the narrow-panel case is finally "
                f"testable -- write it and delete this test")
        finally:
            wired.resize(before, 900)
            for _ in range(8):
                qapp.processEvents()

    def test_both_labels_land_inside_the_panel(self, wired, qapp):
        _hover(wired, qapp, 20, 0, 40)
        ylo, yhi = wired._plot_cvd.vb.viewRange()[1]
        for name, lab in (("cvd", wired._cvd_label),
                          ("off", wired._offex_label)):
            y = lab.pos().y()
            assert ylo <= y <= yhi, (
                f"{name} label at y={y:,.0f} is outside the panel "
                f"[{ylo:,.0f}, {yhi:,.0f}]")
