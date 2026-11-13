"""Spin boxes are wide enough for their own arrows, and inputs stay readable.

Both halves of this file exist because of one Windows 11 regression. The
Windows 11 style lays a spin box's two arrow buttons SIDE BY SIDE where
earlier styles stacked them in a column, taking the chrome from about 28 px
to 58. Every toolbar spin box was pinned to a hardcoded pixel width below
that, so the style returned a NEGATIVE x for SC_SpinBoxUp and painted the
arrows on top of the digits: fourteen boxes in a row with unreadable values.

The fix sizes each box from the style at runtime, and narrows a stylesheet
selector that had been painting over the radio and check indicators. That
second change took the combo popup and the QDoubleSpinBoxes with it -- the
catch-all had been the only rule colouring them -- which is what the
contrast tests below are for.

Nothing here asserts a pixel count. Every assertion is a relation that must
hold at any DPI and under any style, because the numbers themselves differ
per style and this suite does not run under the one that broke.

TWO THINGS TO KNOW BEFORE TRUSTING A GREEN RUN:

  * these run offscreen, which means the Fusion style, which means the
    STACKED arrow layout. The side-by-side branch -- the one that caused the
    bug -- is never exercised here. Treat this file as a regression gate,
    not as proof the Windows 11 path is right. Mutation-checked, two lines
    are provably undefended from this platform and are named here rather
    than left to be rediscovered: removing the `up.x() != down.x()` branch
    in _fit_spin, and picking the widest text by character count instead of
    by rendered width. Both only bite under the side-by-side layout.
  * the indicator damage that motivated narrowing the selector does not
    reproduce under Fusion either -- the dot is still drawn, merely grey.
    So that half is defended at the source level instead, by asserting no
    selector paints a background over every widget in the tree.
  * constructing TradeViewerQt opens an OpenD quote context. The option
    poll is stubbed out below so the test does not depend on a live
    connection, matching what the other GUI tests here do.
"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtCore import QRect                              # noqa: E402
from PyQt6.QtWidgets import (QAbstractSpinBox, QApplication,  # noqa: E402
                             QComboBox, QLineEdit, QStyle,
                             QStyleOptionSpinBox, QToolBar)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app
    app.processEvents()


@pytest.fixture(scope="module")
def viewer(qapp, module_mocker=None):
    from analysis import trade_viewer_qt as mod

    # The option chain poll needs a live OpenD; it has nothing to do with
    # widget geometry and makes the fixture depend on the network.
    original = mod.TradeViewerQt._on_option_indicator_toggle
    mod.TradeViewerQt._on_option_indicator_toggle = lambda self: None
    try:
        v = mod.TradeViewerQt()
        v.resize(1500, 900)
        v.show()
        for _ in range(30):
            qapp.processEvents()
        yield v
        v.close()
        qapp.processEvents()
    finally:
        mod.TradeViewerQt._on_option_indicator_toggle = original


def _rects(spin):
    """Where the style puts the arrows and the text, at the box's own width."""
    opt = QStyleOptionSpinBox()
    opt.initFrom(spin)
    opt.frame = spin.hasFrame()
    opt.buttonSymbols = spin.buttonSymbols()
    opt.subControls = (QStyle.SubControl.SC_SpinBoxUp
                       | QStyle.SubControl.SC_SpinBoxDown
                       | QStyle.SubControl.SC_SpinBoxFrame
                       | QStyle.SubControl.SC_SpinBoxEditField)
    opt.rect = QRect(0, 0, spin.width(), spin.height())
    st, cc = spin.style(), QStyle.ComplexControl.CC_SpinBox
    return (st.subControlRect(cc, opt, QStyle.SubControl.SC_SpinBoxUp, spin),
            st.subControlRect(cc, opt, QStyle.SubControl.SC_SpinBoxDown, spin),
            st.subControlRect(cc, opt,
                              QStyle.SubControl.SC_SpinBoxEditField, spin))


def _widest_text(spin):
    texts = []
    for value in (spin.minimum(), spin.maximum()):
        try:
            texts.append(spin.textFromValue(value))
        except (AttributeError, TypeError):
            texts.append(str(value))
    fm = spin.fontMetrics()
    return max((f"{spin.prefix()}{t}{spin.suffix()}" for t in texts),
               key=fm.horizontalAdvance)


def _lum(c):
    """Perceived luminance, 0..255. Only ever compared, never asserted raw."""
    return 0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue()


def _spins(v):
    return v.findChildren(QAbstractSpinBox)


class TestTheArrowsAreInsideTheBox:
    """The exact shape of the bug: a negative sub-control x means the style
    is drawing the arrows over the digits rather than beside them."""

    def test_there_are_spin_boxes_to_check(self, viewer):
        assert len(_spins(viewer)) >= 10

    def test_no_arrow_starts_left_of_the_box(self, viewer):
        bad = []
        for sp in _spins(viewer):
            up, down, _ = _rects(sp)
            if up.x() < 0 or down.x() < 0:
                bad.append((sp.objectName() or type(sp).__name__,
                            sp.width(), up.x(), down.x()))
        assert not bad, f"arrows painted over the digits: {bad}"

    def test_no_arrow_runs_past_the_right_edge(self, viewer):
        bad = []
        for sp in _spins(viewer):
            up, down, _ = _rects(sp)
            if up.right() > sp.width() or down.right() > sp.width():
                bad.append((sp.width(), up.right(), down.right()))
        assert not bad, f"arrows clipped by the box: {bad}"


class TestTheWidestValueStillFits:
    """Sized to the value it holds today, a box clips the moment the user
    spins it up. The assertion is against the widest value it can show."""

    def test_text_clears_the_arrows(self, viewer):
        bad = []
        for sp in _spins(viewer):
            up, down, edit = _rects(sp)
            need = sp.fontMetrics().horizontalAdvance(_widest_text(sp))
            arrow_left = min(up.x(), down.x())
            if edit.x() + need >= arrow_left:
                bad.append((_widest_text(sp), need, edit.x(), arrow_left))
        assert not bad, f"widest value would touch the arrows: {bad}"


class TestEverySpinBoxWasSized:
    """A box added later and never passed through the fit keeps whatever
    width the layout gives it, which is how this bug arrived in the first
    place. Fitting pins the width, so an unpinned box is an unfitted one."""

    def test_all_are_pinned(self, viewer):
        loose = [type(sp).__name__ for sp in _spins(viewer)
                 if not (0 < sp.minimumWidth() == sp.maximumWidth())]
        assert not loose, f"not sized by _fit_spin: {loose}"


class TestInputsStayReadable:
    """The stylesheet's catch-all selector was painting over the radio and
    check indicators, so it was narrowed -- and that took the combo popup
    and the QDoubleSpinBoxes with it, because the catch-all had been the
    only rule colouring them. Black text on a dark popup is unreadable but
    raises nothing, so it needs an assertion."""

    def test_the_combo_popup_has_contrast(self, viewer):
        combos = viewer.findChildren(QComboBox)
        assert combos, "no combo boxes to check"
        bad = []
        for cb in combos:
            pal = cb.view().palette()
            base = pal.base().color()
            text = pal.text().color()
            if abs(_lum(base) - _lum(text)) < 60:
                bad.append((cb.objectName() or "?", base.name(), text.name()))
        assert not bad, f"combo popup text does not contrast: {bad}"

    def test_the_combo_popup_stays_dark(self, viewer):
        """A light popup would be legible but would flash white over a dark
        chart, which is the thing the stylesheet exists to prevent."""
        light = [(cb.view().palette().base().color().name())
                 for cb in viewer.findChildren(QComboBox)
                 if _lum(cb.view().palette().base().color()) > 110]
        assert not light, f"combo popup is light: {light}"

    def test_text_inputs_have_contrast(self, viewer):
        bad = []
        for w in _spins(viewer) + viewer.findChildren(QLineEdit):
            pal = w.palette()
            base, text = pal.base().color(), pal.text().color()
            if abs(_lum(base) - _lum(text)) < 60:
                bad.append((type(w).__name__, base.name(), text.name()))
        assert not bad, f"input text does not contrast: {bad}"

    def test_text_inputs_stay_dark(self, viewer):
        """QDoubleSpinBox is not a QSpinBox subclass, so it fell out of the
        stylesheet's input rule and came back white on a dark toolbar."""
        light = [(type(w).__name__, w.palette().base().color().name())
                 for w in _spins(viewer) + viewer.findChildren(QLineEdit)
                 if _lum(w.palette().base().color()) > 110]
        assert not light, f"input is light on a dark toolbar: {light}"


class TestTheWindowCannotCollapseItsToolbars:
    """Collapsing a row moves trailing controls into an overflow popup,
    which reads as the controls having disappeared. The floor is what keeps
    that unreachable -- and it has to be computed after the spin boxes are
    sized, or it describes a layout that never reaches the screen."""

    def test_the_floor_matches_the_rows_as_drawn(self, viewer):
        """Equality, not `>=`. The floor is also computed once while the rows
        are being built -- before the stylesheet and before the spin boxes are
        sized -- and that earlier value happens to be larger, so `>=` passes
        whether or not the floor is ever recomputed. Only equality shows that
        the number describes the layout that reaches the screen."""
        rows = viewer.findChildren(QToolBar)
        assert rows, "no toolbars found"
        widest = max(tb.sizeHint().width() for tb in rows)
        assert viewer.minimumWidth() == widest + 16, (
            f"minimum width {viewer.minimumWidth()} does not match the "
            f"widest toolbar row as drawn ({widest} + 16): the floor was "
            f"measured in a state the window is never in")


class TestNoSelectorPaintsOverEveryWidget:
    """`QMainWindow, QWidget { background: ... }` matched every widget in the
    tree, radio buttons and check boxes among them, and on the Windows 11
    style that background is painted over the indicator the style has already
    drawn: the checked radio showed nothing at all, so the Range control read
    as dead.

    This cannot be caught by looking at pixels from here -- under Fusion the
    dot survives, merely grey -- so it is asserted against the stylesheet
    itself. A rule is suspect when a bare QWidget appears in its selector
    list and its body sets a background."""

    def test_the_stylesheet_has_no_catch_all_background(self, viewer):
        import re

        sheet = viewer.styleSheet()
        assert sheet, "the viewer has no stylesheet; this test is stale"
        offenders = []
        for selector, body in re.findall(r"([^{}]+)\{([^{}]*)\}", sheet):
            sel = selector.strip()
            if sel.startswith("/*"):
                continue
            parts = [p.strip() for p in sel.split(",")]
            if "QWidget" in parts and "background" in body:
                offenders.append(sel)
        assert not offenders, (
            f"these rules paint a background over every widget, including "
            f"the radio and check indicators: {offenders}")
