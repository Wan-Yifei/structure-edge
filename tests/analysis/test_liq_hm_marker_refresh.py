"""Overlay markers must re-anchor on every column roll, not only on new data.

The heatmap positions iceberg/spoof/imbalance/absorption overlays by column
index. Columns roll on every Col(s) tick -- including ticks where the order
book has not changed, which forward-fill so the time axis keeps advancing. If
the redraw is skipped on those ticks the markers stay pinned to the column they
were drawn on while the grid scrolls out from under them.

Reported twice: once on the push path (fixed by redrawing in _on_tick) and
again in the after-hours session on the DB-poll path, where a quiet book means
most ticks carry no new data.
"""

import os
from datetime import datetime, timedelta

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtWidgets import QApplication   # noqa: E402

T0 = datetime(2026, 9, 21, 18, 0, 0)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app
    app.processEvents()


@pytest.fixture()
def win(qapp):
    from analysis.liq_hm_window import LiqHmWindow

    w = LiqHmWindow()
    w.set_code("US.SOXL")
    # The poll timer and the push feed would both drive columns on their own;
    # these tests call the handler directly so the roll count is exact.
    w._timer.stop()
    yield w
    w.close()
    qapp.processEvents()


def _snap(ts, mid=141.0):
    return ([{"ts": ts, "side": "BID", "price": mid - 0.01 * i, "volume": 100 + i}
             for i in range(3)]
            + [{"ts": ts, "side": "ASK", "price": mid + 0.01 * (i + 1), "volume": 200 + i}
               for i in range(3)])


class _Spy:
    def __init__(self, win, monkeypatch):
        self.redraws = 0
        self.loads = 0
        monkeypatch.setattr(win, "_redraw_orderflow_markers",
                            lambda: setattr(self, "redraws", self.redraws + 1))
        monkeypatch.setattr(win, "_load_absorb_ticks",
                            lambda: setattr(self, "loads", self.loads + 1))


def test_unchanged_book_still_redraws_markers(win, monkeypatch):
    """The after-hours report: same snapshot tick after tick, grid still rolls."""
    win._on_snap_ready(_snap(T0))
    spy = _Spy(win, monkeypatch)

    for _ in range(5):
        win._on_snap_ready(_snap(T0))      # identical ts -> new_data False

    assert spy.redraws == 5, "markers must re-anchor on every roll"
    assert spy.loads == 5


def test_unchanged_book_still_rolls_a_column(win):
    """Guards the premise: if the roll ever became conditional too, the test
    above would pass for the wrong reason."""
    win._on_snap_ready(_snap(T0))
    before = len(win._col_ts)
    for _ in range(3):
        win._on_snap_ready(_snap(T0))
    assert len(win._col_ts) == before + 3


def test_changed_book_redraws_too(win, monkeypatch):
    win._on_snap_ready(_snap(T0))
    spy = _Spy(win, monkeypatch)

    for i in range(1, 4):
        win._on_snap_ready(_snap(T0 + timedelta(seconds=i), mid=141.0 + i * 0.01))

    assert spy.redraws == 3


def test_redraw_count_matches_roll_count(win, monkeypatch):
    """Mixed quiet and active ticks: one redraw per column, whatever the mix."""
    win._on_snap_ready(_snap(T0))
    spy = _Spy(win, monkeypatch)

    before = len(win._col_ts)
    stamps = [T0, T0, T0 + timedelta(seconds=1), T0 + timedelta(seconds=1),
              T0 + timedelta(seconds=2), T0 + timedelta(seconds=2)]
    for ts in stamps:
        win._on_snap_ready(_snap(ts))

    assert spy.redraws == len(win._col_ts) - before == len(stamps)


def test_empty_snapshot_rolls_nothing_and_redraws_nothing(win, monkeypatch):
    """An empty poll result is 'no answer', not 'the book is empty'."""
    win._on_snap_ready(_snap(T0))
    spy = _Spy(win, monkeypatch)
    before = len(win._col_ts)

    win._on_snap_ready([])

    assert len(win._col_ts) == before
    assert spy.redraws == 0
