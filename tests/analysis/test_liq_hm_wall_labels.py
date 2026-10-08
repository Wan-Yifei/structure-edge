"""A wall label must not outlive the wall by much.

Walls are ranked on mean rendered brightness over the visible columns, which
has no notion of "now": a level consumed early in the window keeps its slot
until its lit columns scroll off the left edge -- measured at up to 5/6 of the
window. Reported from a live screenshot, where a teal "B 156.82  0 / 2,303"
sat stranded over what had since become a solid red ask band.

A grace period bounds that. The freed slot costs nothing, because the next
candidate takes it: over a 3h replay the label count fell 10,068 -> 10,031
while the share of labels reading zero went 35.6% -> 14.3% on the bid side.
"""

import os
import re
from datetime import datetime, timedelta

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtWidgets import QApplication   # noqa: E402

T0 = datetime(2026, 10, 8, 1, 50, 0)
N_COLS = 240
PRICE_MIN, PRICE_MAX = 155.0, 160.0


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app
    app.processEvents()


@pytest.fixture()
def win(qapp):
    """A window whose grid is written directly, so each test states its book."""
    from analysis.liq_hm_window import LiqHmWindow, N_PRICE

    w = LiqHmWindow()
    w.set_code("US.SOXL")
    w._timer.stop()
    w._price_min, w._price_max = PRICE_MIN, PRICE_MAX
    w._bin_size = (PRICE_MAX - PRICE_MIN) / N_PRICE
    w._col_ts = [T0 + timedelta(seconds=i) for i in range(N_COLS)]
    w._bid_grid[:] = 0.0
    w._ask_grid[:] = 0.0
    w._wall_cb.setChecked(True)
    w._bid_ask_cb.setChecked(True)     # rank each side on its own grid
    w._col_secs_spin.setValue(1)
    w._hist_bucket = 0
    w._gamma_spin.setValue(1.0)
    w._min_vol_spin.setValue(0)
    w.resize(900, 700)
    w.show()
    qapp.processEvents()
    w._plot_widget.getPlotItem().vb.setXRange(0, N_COLS, padding=0)
    qapp.processEvents()
    yield w
    w.close()
    qapp.processEvents()


def _bin_of(w, price):
    return int((price - w._price_min) / w._bin_size)


def _draw(w, qapp):
    w._clear_overlay_items()
    w._draw_wall_labels()
    qapp.processEvents()
    return [it.textItem.toPlainText() for it in w._wall_items]


def _priced(labels, side):
    """{price: size-text} for one side's labels."""
    out = {}
    for t in labels:
        m = re.match(r"([BA]) ([\d.]+)(?:±[\d.]+)?\s+(.*)", t)
        if m and m.group(1) == side:
            out[float(m.group(2))] = m.group(3).strip()
    return out


class TestGracePeriod:
    """How long an emptied level keeps its label."""

    def test_a_long_dead_wall_is_dropped(self, win, qapp):
        """Thick for the first half of the window, gone since: not a wall."""
        from analysis.liq_hm_window import _WALL_MAX_EMPTY_SECS

        dead = _bin_of(win, 156.82)
        win._bid_grid[: N_COLS // 2, dead] = 2303.0
        # someone else to take the slot, so the drop is not just "nothing left"
        live = _bin_of(win, 156.11)
        win._bid_grid[:N_COLS, live] = 1411.0

        labels = _priced(_draw(win, qapp), "B")
        assert N_COLS - N_COLS // 2 > _WALL_MAX_EMPTY_SECS, "fixture too short"
        assert not any(abs(p - 156.82) < 0.05 for p in labels), labels
        assert any(abs(p - 156.11) < 0.05 for p in labels), labels

    def test_a_just_eaten_wall_still_shows(self, win, qapp):
        """Inside the grace period the consumption is the point of the label."""
        from analysis.liq_hm_window import _WALL_MAX_EMPTY_SECS

        bi = _bin_of(win, 156.82)
        gone_for = int(_WALL_MAX_EMPTY_SECS) - 5      # 1s per column here
        win._bid_grid[: N_COLS - gone_for, bi] = 2303.0

        labels = _priced(_draw(win, qapp), "B")
        hit = [v for p, v in labels.items() if abs(p - 156.82) < 0.05]
        assert hit, labels
        assert hit[0].startswith("0 / "), hit

    def test_an_intact_wall_is_unaffected(self, win, qapp):
        bi = _bin_of(win, 158.38)
        win._ask_grid[:N_COLS, bi] = 828.0
        labels = _priced(_draw(win, qapp), "A")
        hit = [v for p, v in labels.items() if abs(p - 158.38) < 0.05]
        assert hit and hit[0] == "828", labels

    def test_a_partly_eaten_wall_is_unaffected(self, win, qapp):
        """276 / 771 is still liquidity; only *empty* starts the clock."""
        bi = _bin_of(win, 158.19)
        win._ask_grid[:N_COLS, bi] = 771.0
        win._ask_grid[N_COLS - 100:, bi] = 276.0
        labels = _priced(_draw(win, qapp), "A")
        hit = [v for p, v in labels.items() if abs(p - 158.19) < 0.05]
        assert hit and hit[0] == "276 / 771", labels

    def test_a_level_that_came_back_is_not_stale(self, win, qapp):
        """Emptied, then re-posted: the run of zeros ended, so it counts."""
        bi = _bin_of(win, 156.43)
        win._bid_grid[:50, bi] = 376.0
        win._bid_grid[N_COLS - 3:, bi] = 274.0
        labels = _priced(_draw(win, qapp), "B")
        hit = [v for p, v in labels.items() if abs(p - 156.43) < 0.05]
        assert hit and hit[0] == "274 / 376", labels

    def test_grace_is_measured_in_time_not_columns(self, win, qapp):
        """The allowance grows with the span on screen, so a wide replay
        bucket is not punished for packing more minutes into a column.

        The same 33-second-old wall, twice: dropped when the window covers 4
        minutes (allowance held at the 30s floor), kept when 3s replay buckets
        stretch it to 12 and 5% of that comes to 36s.
        """
        bi = _bin_of(win, 156.82)

        win._bid_grid[: N_COLS - 33, bi] = 2303.0       # 1s columns -> 33s ago
        assert not _priced(_draw(win, qapp), "B"), "past the 30s floor"

        win._bid_grid[:] = 0.0
        win._hist_bucket = 3
        win._col_ts = [T0 + timedelta(seconds=3 * i) for i in range(N_COLS)]
        win._bid_grid[: N_COLS - 11, bi] = 2303.0       # 3s columns -> 33s ago
        labels = _priced(_draw(win, qapp), "B")
        assert [v for p, v in labels.items() if abs(p - 156.82) < 0.05], labels


class TestGraceDoesNotCostLabels:
    """The slot goes to the next candidate rather than being left blank."""

    def test_the_count_is_held(self, win, qapp):
        dead = [156.82, 157.00, 157.20]
        live = [156.11, 156.43, 155.60]
        for p in dead:
            win._bid_grid[: N_COLS // 2, _bin_of(win, p)] = 5000.0
        for p in live:
            win._bid_grid[:N_COLS, _bin_of(win, p)] = 1200.0

        labels = _priced(_draw(win, qapp), "B")
        assert len(labels) == win._wall_n_spin.value()
        assert all(not any(abs(p - d) < 0.05 for d in dead) for p in labels), labels

    def test_nothing_recent_leaves_the_side_unlabelled(self, win, qapp):
        """Not an error: an empty side has no wall to point at."""
        win._bid_grid[: N_COLS // 2, _bin_of(win, 156.82)] = 5000.0
        assert _priced(_draw(win, qapp), "B") == {}
