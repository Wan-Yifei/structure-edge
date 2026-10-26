"""Construct the floating order-flow windows for real, once each.

These exist because of a refactor that replaced a block of liq_hm_window.py by
line index and silently took six module-level definitions with it -- including
the QObject bridge the constructor instantiates. Every unit test still passed,
because none of them ever built the window; the failure only showed up as a
NameError when a person opened it. A construction that touches __init__ is the
cheapest thing that would have caught it.

Nothing here asserts on rendering. The point is that the module's names resolve
and the widget tree builds.
"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtWidgets import QApplication   # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app
    app.processEvents()


def _settle(app, widget, rounds: int = 20):
    """Pump the event loop a little, then tear the widget down cleanly."""
    for _ in range(rounds):
        app.processEvents()
    widget.close()
    app.processEvents()


def test_liq_hm_window_constructs(qapp):
    from analysis.liq_hm_window import LiqHmWindow

    w = LiqHmWindow()
    w.set_code("US.SOXL")
    w.show()
    _settle(qapp, w)


def test_trade_viewer_constructs(qapp):
    """The main chart, including the crosshair tags built in __init__."""
    from analysis.trade_viewer_qt import TradeViewerQt

    w = TradeViewerQt()
    assert w._pair_price_label is not None, "inverse-pair crosshair tag"
    _settle(qapp, w)


def test_dom_window_constructs(qapp, tmp_path):
    from analysis.dom_window import DomWindow

    w = DomWindow("US.SOXL", db_path=tmp_path / "ob.db",
                  ticks_db_path=tmp_path / "ticks.db")
    w.show()
    _settle(qapp, w)


class TestClosingTheViewerTakesItsChildrenWithIt:
    """Three floating windows are built without a parent, so each is a
    top-level window in its own right and the viewer closing does not
    close any of them by itself. closeEvent has to, and Multi-VP was the
    one it had been left out of: the window stayed on screen with no
    owner, the event loop stayed alive because of it, and its 30s timer
    kept running against a moomoo context the teardown had already
    closed. Since v0.74.3 that timer asks for FRESH data, so an orphan
    opens a new connection every half minute.

    A plain QWidget stands in for each: the point is the teardown list,
    not what the windows do.
    """

    def _viewer_with_children(self, qapp):
        from PyQt6.QtWidgets import QWidget
        from analysis.trade_viewer_qt import TradeViewerQt

        v = TradeViewerQt()
        kids = {}
        for attr in ("_mvp_window", "_liq_hm_window", "_dom_window"):
            w = QWidget()
            w.show()
            setattr(v, attr, w)
            kids[attr] = w
        for _ in range(10):
            qapp.processEvents()
        return v, kids

    def test_every_child_window_is_closed(self, qapp):
        v, kids = self._viewer_with_children(qapp)
        v.close()
        qapp.processEvents()
        for attr, w in kids.items():
            assert not w.isVisible(), f"{attr} was left on screen"
            assert getattr(v, attr) is None, f"{attr} was not released"

    def test_multi_vp_in_particular(self, qapp):
        """The one that was reported: close the viewer, it is still there."""
        v, kids = self._viewer_with_children(qapp)
        v.close()
        qapp.processEvents()
        assert not kids["_mvp_window"].isVisible()
