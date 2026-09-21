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


def test_dom_window_constructs(qapp, tmp_path):
    from analysis.dom_window import DomWindow

    w = DomWindow("US.SOXL", db_path=tmp_path / "ob.db",
                  ticks_db_path=tmp_path / "ticks.db")
    w.show()
    _settle(qapp, w)
