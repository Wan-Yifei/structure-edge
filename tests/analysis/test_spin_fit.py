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

  * WHICH PLATFORM YOU RUN THIS ON CHANGES WHAT IT CATCHES. Offscreen
    means the Fusion style, and it also gives the Windows 11 style 18 px
    arrow buttons where the native platform gives 24 -- so a defect that
    only shows when the buttons reach the digits is milder there, or gone.
    Mutation-checked both ways: dropping the overlap compensation in
    fit_spin SURVIVES offscreen and is CAUGHT natively, by the rendering
    test at the bottom. Run `QT_QPA_PLATFORM=windows` before trusting a
    green run to mean the Windows 11 path is right.

  * one mutant is undefended on either platform: picking the widest text by
    character count instead of by rendered width. That is not a platform
    gap -- none of the 26 boxes here has a min/max pair where the two
    disagree, so there is nothing for an assertion to catch. It would need
    a box whose longer value string renders narrower, and inventing one to
    assert against would be testing the test.
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

from PyQt6.QtCore import QRect, Qt                          # noqa: E402
from PyQt6.QtGui import QImage                              # noqa: E402
from PyQt6.QtWidgets import (QAbstractSpinBox, QApplication,  # noqa: E402
                             QComboBox, QLineEdit, QStyle,
                             QStyleOptionSpinBox, QToolBar)

from analysis.qt_widgets import SPIN_TEXT_GAP               # noqa: E402

# The one pixel count in this file, and it is here on purpose: see
# test_the_gap_is_at_least_what_was_measured. Importing it would make that
# test compare the implementation against itself.
MEASURED_MIN_GAP = 10


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


def _cramped(spins):
    """Boxes whose inner edit area cannot render their widest value.

    Measured against the inner QLineEdit, not against SC_SpinBoxEditField.
    The sub-control rect's left edge is not where the glyphs begin, and an
    earlier version of this test compared against it: it passed while every
    box in the heatmap toolbar was visibly cutting its last digit.

    The margin is SPIN_TEXT_GAP because horizontalAdvance is not the width a
    QLineEdit renders into -- it reserves room for the cursor, and ink can
    run past a glyph's advance. Sweeping the width pixel by pixel puts that
    shortfall at a flat 10 px whatever the text, which is where the constant
    comes from.
    """
    bad, seen = [], 0
    for sp in spins:
        le = sp.lineEdit()
        if le is None:
            continue
        seen += 1
        need = sp.fontMetrics().horizontalAdvance(_widest_text(sp))
        if le.width() - need < SPIN_TEXT_GAP:
            bad.append((_widest_text(sp), need, le.width(), sp.width()))
    assert seen == len(spins), (
        f"only {seen} of {len(spins)} boxes had a line edit to measure; "
        f"a silent skip here would let this check pass on nothing")
    return bad


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

    def test_the_edit_area_has_room_for_it(self, viewer):
        assert not _cramped(_spins(viewer)), (
            f"widest value does not fit the edit area: "
            f"{_cramped(_spins(viewer))}")

    def test_the_gap_is_at_least_what_was_measured(self):
        """The test above compares against SPIN_TEXT_GAP, so lowering the
        constant moves the implementation and the ruler together and nothing
        notices -- which is how a gap of 6 shipped while every box in the
        heatmap toolbar was cutting its last digit.

        MEASURED_MIN_GAP is therefore written out here rather than imported.
        It is the one number in this file that is a pixel count, and it is
        an EXACT LOWER BOUND, not a safety margin: rendering the heatmap's
        twelve boxes at each gap gives 12/12 cut at 6 and at 8, 11/12 cut at
        9, and 0/12 at 10. There is nothing spare in it. It is tied to this
        display -- re-measure it before trusting a green run at a different
        DPI or font size."""
        assert SPIN_TEXT_GAP >= MEASURED_MIN_GAP, (
            f"the gap was lowered to {SPIN_TEXT_GAP}; the measured minimum "
            f"is {MEASURED_MIN_GAP} and below it the last character is cut")


class TestEverySpinBoxWasSized:
    """A box added later and never passed through the fit keeps whatever
    width the layout gives it, which is how this bug arrived in the first
    place. Fitting pins the width, so an unpinned box is an unfitted one."""

    def test_all_are_pinned(self, viewer):
        loose = [type(sp).__name__ for sp in _spins(viewer)
                 if not (0 < sp.minimumWidth() == sp.maximumWidth())]
        assert not loose, f"not sized by fit_spin: {loose}"


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


@pytest.fixture(scope="module")
def heatmap(qapp):
    from analysis.liq_hm_window import LiqHmWindow

    w = LiqHmWindow()
    w.show()
    for _ in range(20):
        qapp.processEvents()
    yield w
    w.close()
    qapp.processEvents()


class TestTheHeatmapToolbarToo:
    """The heatmap window is a separate top level window with no stylesheet
    of its own, so it never inherited the trade viewer's -- and it had its
    own twelve pinned spin boxes, every one of them narrower than the
    Windows 11 chrome. Same invariants, asserted here because the fix lives
    in a shared module and a change to it has to be answerable for both."""

    def test_it_has_spin_boxes(self, heatmap):
        assert len(_spins(heatmap)) >= 10

    def test_no_arrow_starts_left_of_the_box(self, heatmap):
        bad = [(sp.width(), _rects(sp)[0].x(), _rects(sp)[1].x())
               for sp in _spins(heatmap)
               if _rects(sp)[0].x() < 0 or _rects(sp)[1].x() < 0]
        assert not bad, f"arrows painted over the digits: {bad}"

    def test_the_edit_area_has_room_for_it(self, heatmap):
        assert not _cramped(_spins(heatmap)), (
            f"widest value does not fit the edit area: "
            f"{_cramped(_spins(heatmap))}")

    def test_all_are_pinned(self, heatmap):
        loose = [type(sp).__name__ for sp in _spins(heatmap)
                 if not (0 < sp.minimumWidth() == sp.maximumWidth())]
        assert not loose, f"not sized by fit_spin: {loose}"

    def test_the_floor_covers_the_rows_as_drawn(self, heatmap):
        """Its two rows are plain QToolBars inside a QVBoxLayout, never
        docked, so Qt's overflow chevron never engages for them: too narrow
        a window simply clips the trailing controls off the edge."""
        rows = [heatmap._toolbar.layout().itemAt(i).widget() for i in range(2)]
        widest = max(r.sizeHint().width() for r in rows)
        assert heatmap.minimumWidth() >= widest, (
            f"minimum width {heatmap.minimumWidth()} is under the widest "
            f"row {widest}: trailing controls can be clipped away")


# A QStyle created here is owned by Python. Handing one to setStyle and
# letting it fall out of scope leaves Qt dereferencing freed memory, which
# shows up as an access violation at teardown rather than at the call.
_STYLE_KEEPALIVE = []


def _stand_in(original, style, sheet, app):
    """A copy of `original` under `style`, outside the viewer's widget tree.

    Everything fit_spin reads is carried over: range, decimals, prefix,
    suffix, button symbols and the stylesheet (whose border and padding are
    part of the chrome).

    Returns (host, spin). The host has to be held by the caller: it owns the
    spin box, and letting it go out of scope deletes the C++ object out from
    under the Python wrapper mid-test. Returns (host, None) for a box with
    no line edit, which the caller counts so a silent skip cannot quietly
    empty this test.
    """
    from PyQt6.QtWidgets import (QDoubleSpinBox, QHBoxLayout, QSpinBox,
                                 QWidget)

    host = QWidget()
    host.setStyleSheet(sheet)
    lay = QHBoxLayout(host)
    lay.setContentsMargins(0, 0, 0, 0)
    sp = (QDoubleSpinBox() if isinstance(original, QDoubleSpinBox)
          else QSpinBox())
    if isinstance(original, QDoubleSpinBox):
        sp.setDecimals(original.decimals())
    sp.setRange(original.minimum(), original.maximum())
    sp.setPrefix(original.prefix())
    sp.setSuffix(original.suffix())
    sp.setButtonSymbols(original.buttonSymbols())
    sp.setValue(original.value())
    sp.setStyle(style)
    lay.addWidget(sp)
    host.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    host.show()
    for _ in range(5):
        app.processEvents()
    return host, (sp if sp.lineEdit() is not None else None)


def _glyph_extent(app, sp, text):
    """(bounding-box width, ink pixels) of what `text` actually paints.

    Rendered twice at the same width -- once with the text, once with the
    field emptied -- and differenced. Everything that is not the glyphs
    cancels, so an arrow button drawn ON TOP of a digit removes that digit's
    ink exactly as clipping would. That is the point: the geometric checks
    in this file compare against the edit area, and with a stylesheet the
    edit area runs underneath the buttons, so they cannot see a covered
    digit at all.
    """
    import numpy as np

    def shot():
        img = sp.grab().toImage().convertToFormat(
            QImage.Format.Format_RGB32)
        ptr = img.constBits()
        ptr.setsize(img.sizeInBytes())
        a = np.frombuffer(ptr, np.uint32).reshape(
            img.height(), img.bytesPerLine() // 4)
        return a[:, :img.width()].copy()

    le = sp.lineEdit()
    was = le.text()
    le.setText(text)
    app.processEvents()
    with_text = shot()
    le.setText("")
    app.processEvents()
    without = shot()
    le.setText(was)
    app.processEvents()

    diff = with_text != without
    cols = np.flatnonzero(diff.any(axis=0))
    if cols.size == 0:
        return 0, 0
    return int(cols.max() - cols.min() + 1), int(diff.sum())


class TestTheDigitsAreActuallyVisible:
    """The one check here that looks at pixels, and the only one that can
    see the original bug.

    Every other assertion in this file compares geometry against the edit
    area. With a stylesheet the edit area extends underneath the arrow
    buttons, so a box whose arrows are painted over its last digit still
    measures as having room -- mutation-checked: dropping the overlap
    compensation in fit_spin puts 13 of the viewer's 14 boxes back into the
    original bug and every geometric assertion here still passes, on the
    Windows 11 style as well as on Fusion.

    So this one renders instead, under the Windows 11 style, because that is
    the only style whose edit rect overlaps the buttons -- under Fusion the
    same defect is invisible and this would pass vacuously.

    It builds stand-ins rather than restyling the viewer's own boxes. A
    QStyle handed to setStyle stays owned by Python, and swapping one onto
    a long-lived widget and back leaves Qt holding a pointer into an object
    the interpreter is free to collect: doing it to the shared viewer
    crashed the run at teardown with an access violation. The stand-ins
    carry the same range, decimals, prefix, suffix and stylesheet, which is
    everything fit_spin reads.
    """

    @pytest.fixture(scope="class")
    def win11(self):
        from PyQt6.QtWidgets import QStyleFactory

        style = QStyleFactory.create("windows11")
        if style is None:
            pytest.skip("the windows11 style is not available here; this "
                        "check is vacuous under any other style")
        # Held for the life of the class: see the docstring.
        _STYLE_KEEPALIVE.append(style)
        return style

    def test_every_digit_survives_at_the_fitted_width(self, qapp, viewer,
                                                      win11):
        pytest.importorskip("numpy")
        from analysis.qt_widgets import fit_spin

        sheet = viewer.styleSheet()
        checked, bad = 0, []
        for original in _spins(viewer):
            host, sp = _stand_in(original, win11, sheet, qapp)
            if sp is None:
                host.close()
                continue
            checked += 1
            text = _widest_text(sp)

            fit_spin(sp)
            qapp.processEvents()
            fitted = sp.width()
            tight = _glyph_extent(qapp, sp, text)

            sp.setFixedWidth(fitted + 30)
            qapp.processEvents()
            loose = _glyph_extent(qapp, sp, text)

            # A column of ClearType feathering at the right edge is not a
            # stroke; losing a whole glyph is not one pixel.
            if tight[0] < loose[0] - 1 or tight[1] < loose[1] - 8:
                bad.append((text, fitted, tight, loose))
            host.close()

        assert checked == len(_spins(viewer)), (
            f"only {checked} of {len(_spins(viewer))} boxes were rendered; "
            f"a silent skip here would make this test vacuous")
        assert not bad, (
            f"digits are covered or cut at the fitted width "
            f"(text, width, tight=(bbox, ink), loose=(bbox, ink)): {bad}")


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
