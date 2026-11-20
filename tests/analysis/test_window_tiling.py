"""The viewer and the heatmap have to fit side by side.

Both windows floor their own width at their widest toolbar row, so that
no control can ever be pushed into an overflow popup where it looks like
it has gone missing. That floor is a real constraint on a desktop: two
windows at 1337 and 1289 px need 2626 px, and the screen this was
specified against has 2560. They overlapped.

Nothing in either window's own tests could see that, because it is a
property of the PAIR. So it is pinned here, and the two ways it has
already been broken are pinned with it:

  * a spin box padded by more than the overlap it had to clear -- 50 px
    where 22 was enough, which with six boxes on one row cost 168 px
  * a toolbar row carrying more control groups than the others

PLATFORM: offscreen has no font directory, so Qt falls back to a face
that is about 30% wider and every width here is wrong. These tests run
on the native platform only; offscreen they skip rather than pass.
"""

import os

import pytest

pytest.importorskip("PyQt6.QtWidgets")

NATIVE = os.environ.get("QT_QPA_PLATFORM", "") in ("", "windows")

# The screen this requirement was written against. A machine with less
# than this genuinely cannot tile the two windows, and should say so
# rather than skip.
SCREEN_PX = 2560

pytestmark = pytest.mark.skipif(
    not NATIVE,
    reason="offscreen font fallback inflates widths ~30%; run with "
           "QT_QPA_PLATFORM=windows for a real measurement")


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app
    app.processEvents()


def _floor(qapp, cls):
    w = cls()
    w.show()
    for _ in range(40):
        qapp.processEvents()
    w.resize(300, 700)
    for _ in range(15):
        qapp.processEvents()
    floor, actual = w.minimumWidth(), w.width()
    w.close()
    for _ in range(10):
        qapp.processEvents()
    return floor, actual


def test_the_two_windows_fit_side_by_side(qapp):
    from analysis.trade_viewer_qt import TradeViewerQt
    from analysis.liq_hm_window import LiqHmWindow

    v, v_actual = _floor(qapp, TradeViewerQt)
    h, h_actual = _floor(qapp, LiqHmWindow)
    assert v == v_actual and h == h_actual, (
        "a window did not shrink to its own stated minimum, so these "
        "numbers do not describe what a user can do")
    assert v + h <= SCREEN_PX, (
        f"viewer {v} px + heatmap {h} px = {v + h} px, which does not fit "
        f"{SCREEN_PX} px. Whatever was just added to a toolbar row has "
        f"raised a window's floor; put it on a shorter row.")


def test_no_single_row_dominates_its_window(qapp):
    """A floor is the widest row, so one overloaded row costs the window.

    Not a style rule: it is the failure mode both windows hit. The
    heatmap carried five control groups on one row and ten on the other,
    and the crowded row alone set a floor 165 px above what the window
    needed.
    """
    from PyQt6.QtWidgets import QToolBar
    from analysis.liq_hm_window import LiqHmWindow

    w = LiqHmWindow()
    w.show()
    for _ in range(40):
        qapp.processEvents()
    rows = sorted((tb.sizeHint().width() for tb in
                   w._toolbar.findChildren(QToolBar)), reverse=True)
    w.close()
    for _ in range(10):
        qapp.processEvents()

    assert len(rows) >= 3, f"expected the toolbar to be split, got {rows}"
    assert rows[0] - rows[1] <= 0.55 * rows[0], (
        f"the widest row is {rows[0]} px against {rows[1]} px for the "
        f"next one; that row alone is setting the window's floor")
