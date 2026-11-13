"""Widget sizing shared by the Qt windows.

Lives on its own because the same bug hit every one of them at once when
this machine moved to Windows 11, and a fix that only knows about the trade
viewer leaves the heatmap, the DOM and the multi-VP windows garbled.
"""
from __future__ import annotations

from PyQt6.QtCore import QRect
from PyQt6.QtWidgets import QAbstractSpinBox, QStyle, QStyleOptionSpinBox

# Room the box needs past the text's advance width. Not a cosmetic margin:
# horizontalAdvance is not what a QLineEdit renders into. It reserves space
# for the cursor, and a glyph's ink can run past its own advance, so a box
# sized to the advance alone cuts its last character -- which is exactly what
# the heatmap's twelve boxes did at a gap of 6.
#
# Derived, not chosen. Walking the width down pixel by pixel and watching for
# the first width where the rendered text stops matching a generous render
# gives a constant 10 px, independent of the text: 12 px of text needs 74,
# 18 needs 80, 27 needs 89, 42 needs 104, and `advance + chrome + 10`
# reproduces all four exactly.
#
# That measurement is from a box with no stylesheet, which is the binding
# case. With one, the same sweep says the chrome is already 14 px more than
# needed, so those boxes come out about 24 px wider than strictly necessary.
# Fat is the safe direction here; thin is the bug being fixed.
SPIN_TEXT_GAP = 10

# The width used to ask the style where its sub-controls go. Any value wide
# enough that none of them degenerate works; the answer is a difference, so
# the number itself does not reach the result.
_SPIN_PROBE_W = 400


def fit_spin(spin: QAbstractSpinBox, gap: int = SPIN_TEXT_GAP) -> None:
    """Widen a spin box until its own widest value clears the arrows.

    These were hardcoded pixel widths until Windows 11, where every one of
    them broke at once: the Windows 11 style puts the two arrow buttons SIDE
    BY SIDE (two 24 px buttons) where earlier styles stacked them in one
    16 px column, taking the chrome from about 28 px to 58. Worse, it then
    returns a NEGATIVE x for SC_SpinBoxUp when the box is too narrow, so the
    arrows are painted on top of the digits rather than beside them -- which
    is why the toolbars read as rows of garbled boxes rather than as text cut
    off at an edge.

    Qt will not answer this directly. sizeHint and sizeFromContents both
    over-report (86 px of chrome against 58 measured by sweeping the width
    and looking at the pixels), while SC_SpinBoxEditField sometimes
    under-reports: on the Windows 11 style WITH a stylesheet it hands back a
    rect that runs 21 px into the up button, having subtracted only one of
    the two. So take the edit field's answer and add a button back exactly
    when the rect it returned overlaps them.

    Asking about the overlap rather than about the layout matters. The first
    version of this asked whether the buttons sat side by side, which is the
    cause but not the symptom: on the Windows 11 style with NO stylesheet
    they are still side by side, but the edit rect already clears them, and
    adding a button back made every box in the heatmap window 24 px fatter
    than it needed to be.

    Checked against a width sweep on every style this machine offers, and
    against what each one really needs, in 96 combinations of style,
    stylesheet, int/double, button symbols and frame: 92 agree exactly, 4
    (windows11 with a stylesheet) come out 24 px wide, and NONE come out
    narrow. Fat is the safe direction; narrow is the bug.

    With a stylesheet windows11 needs 58 and without it 52; windowsvista
    needs 27 with and 18 without, Fusion 24 and 20. Quoting one of those
    against the other is how a 28 and a 26 ended up in this docstring
    claiming to be the same measurement.

    Call this AFTER any stylesheet is applied. Not because the box comes out
    too narrow otherwise -- measured, it comes out 18 px too WIDE (100
    against 82 for a plain QSpinBox, 103 against 88 for a double): the
    stylesheet's own border and padding replace a larger native frame. Run
    early and a fourteen-box row gains about 250 px, which is what would push
    it past the window's minimum width and start clipping controls off the
    edge. A window with no stylesheet has nothing to wait for and can fit its
    boxes as soon as they exist.
    """
    spin.ensurePolished()

    opt = QStyleOptionSpinBox()
    opt.initFrom(spin)
    opt.frame = spin.hasFrame()
    opt.buttonSymbols = spin.buttonSymbols()
    opt.subControls = (QStyle.SubControl.SC_SpinBoxUp
                       | QStyle.SubControl.SC_SpinBoxDown
                       | QStyle.SubControl.SC_SpinBoxFrame
                       | QStyle.SubControl.SC_SpinBoxEditField)
    opt.rect = QRect(0, 0, _SPIN_PROBE_W, spin.sizeHint().height())

    style, cc = spin.style(), QStyle.ComplexControl.CC_SpinBox
    up = style.subControlRect(cc, opt, QStyle.SubControl.SC_SpinBoxUp, spin)
    down = style.subControlRect(cc, opt, QStyle.SubControl.SC_SpinBoxDown, spin)
    edit = style.subControlRect(cc, opt,
                                QStyle.SubControl.SC_SpinBoxEditField, spin)

    chrome = _SPIN_PROBE_W - edit.width()
    if edit.right() > min(up.x(), down.x()):
        chrome += down.width()

    # The widest string the box can ever display, not the value it holds now:
    # a box sized to "1" clips the moment the user spins it to "300".
    texts = []
    for value in (spin.minimum(), spin.maximum()):
        try:
            texts.append(spin.textFromValue(value))
        except (AttributeError, TypeError):
            texts.append(str(value))
    # By rendered width, not by character count: with a prefix, a suffix or
    # a proportional font the longer string is not always the wider one.
    fm = spin.fontMetrics()
    text_w = max(fm.horizontalAdvance(f"{spin.prefix()}{t}{spin.suffix()}")
                 for t in texts)

    spin.setFixedWidth(text_w + chrome + gap)


def fit_all_spins(root, gap: int = SPIN_TEXT_GAP) -> int:
    """Fit every spin box under `root`. Returns how many were sized.

    findChildren rather than a list built while the toolbar is assembled,
    so a box does not have to be registered by hand -- but only boxes that
    exist WHEN THIS RUNS are covered. One created later is not; the test
    that every box is pinned is what catches that.
    """
    spins = root.findChildren(QAbstractSpinBox)
    for spin in spins:
        fit_spin(spin, gap)
    return len(spins)
