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
    # No processEvents() at teardown -- see test_absorption_marker.py


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
        # Inside the grace window but not zero: derived from the constant, not
        # a fixed offset from it. The offset was -5, which silently became
        # "not empty at all" when the grace dropped from 30s to 5s and the
        # test then asserted against an intact wall.
        gone_for = max(1, int(_WALL_MAX_EMPTY_SECS) - 1)   # 1s per column here
        assert 0 < gone_for < _WALL_MAX_EMPTY_SECS, "fixture outside the grace"
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


class TestBrightnessIsComputedOnceAndOnlyOnOccupiedCells:
    """The sparse brightness helper, where the window actually uses it.

    test_liq_hm_grid_fidelity.py pins the arithmetic; these pin the two
    things this change really did, neither of which that file can see:
    that the helper is reached at all, and that with Bid/Ask combined it
    is reached ONCE rather than twice with the same array.

    A call count is the only thing that notices the second one. Computing
    the combined grid unconditionally, or recomputing it inside the loop,
    leaves every byte of output identical.
    """

    @staticmethod
    def _counting(monkeypatch):
        """Wrap _bright_norm, recording each call's input."""
        import analysis.liq_hm_window as L

        seen = []
        real = L._bright_norm

        def spy(src, gamma):
            seen.append(np.array(src, copy=True))
            return real(src, gamma)

        monkeypatch.setattr(L, "_bright_norm", spy)
        return seen

    def test_combined_computes_it_once(self, win, qapp, monkeypatch):
        win._bid_ask_cb.setChecked(False)
        win._bid_grid[:N_COLS, _bin_of(win, 156.82)] = 2303.0
        win._ask_grid[:N_COLS, _bin_of(win, 158.38)] = 828.0
        seen = self._counting(monkeypatch)
        _draw(win, qapp)
        assert len(seen) == 1, (
            f"the brightness pipeline ran {len(seen)} times for one redraw; "
            f"with Bid/Ask combined both sides read the same array")

    def test_split_computes_it_twice_on_different_grids(self, win, qapp,
                                                        monkeypatch):
        """The control. Without it, a helper that never ran at all would
        satisfy the test above."""
        win._bid_ask_cb.setChecked(True)
        win._bid_grid[:N_COLS, _bin_of(win, 156.82)] = 2303.0
        win._ask_grid[:N_COLS, _bin_of(win, 158.38)] = 828.0
        seen = self._counting(monkeypatch)
        _draw(win, qapp)
        assert len(seen) == 2, f"{len(seen)} calls, expected one per side"
        assert not np.array_equal(seen[0], seen[1]), (
            "both sides were handed the same array in split mode, so one of "
            "them is ranked on the wrong grid")

    def test_split_mode_does_not_build_the_combined_grid(self, win, qapp,
                                                         monkeypatch):
        """`if combined else None`, not an unconditional call.

        An unconditional combined grid is computed and discarded in split
        mode: no output changes, so only the inputs this spy records can
        tell. Each call must see one side's grid, never the sum.
        """
        win._bid_ask_cb.setChecked(True)
        bid_bin, ask_bin = _bin_of(win, 156.82), _bin_of(win, 158.38)
        win._bid_grid[:N_COLS, bid_bin] = 2303.0
        win._ask_grid[:N_COLS, ask_bin] = 828.0
        seen = self._counting(monkeypatch)
        _draw(win, qapp)
        for i, src in enumerate(seen):
            lit = {int(c) for c in np.nonzero(src.max(axis=0))[0]}
            assert lit in ({bid_bin}, {ask_bin}), (
                f"call {i} was handed {sorted(lit)}, which is neither side "
                f"on its own -- the combined grid was built in split mode")


class TestTheLabelsAreTheOnesTheDenseFormChose:
    """End to end: same window, same book, dense pipeline against sparse.

    The arithmetic test asserts bytes. This asserts the thing bytes are
    for -- which levels were picked, in which order, with which text --
    through the ranking, the gate and the collision layout.
    """

    @staticmethod
    def _dense(src, gamma):
        """The pipeline as of v0.105.1, before the occupied-cells rewrite."""
        from analysis.liq_hm_window import _percentile_norm

        norm = _percentile_norm(np.log1p(src))
        if gamma != 1.0:
            norm = np.power(norm, gamma)
        return norm

    def _book(self, win):
        """A book with a level sitting just above the brightness floor.

        Without one, both versions agree because every candidate is far
        from _WALL_MIN_BRIGHT and no rounding could move it across.
        """
        from analysis.liq_hm_window import _WALL_MIN_BRIGHT

        win._bid_grid[:N_COLS, _bin_of(win, 156.82)] = 90000.0
        for i, price in enumerate((156.40, 156.21, 157.03)):
            win._bid_grid[:N_COLS, _bin_of(win, price)] = 120.0 * (i + 1)
        win._ask_grid[:N_COLS, _bin_of(win, 158.38)] = 40000.0
        for i, price in enumerate((158.60, 158.81)):
            win._ask_grid[:N_COLS, _bin_of(win, price)] = 150.0 * (i + 1)

        # A column band lit for only a slice of the window, so its mean
        # brightness lands near the floor rather than at one end.
        faint = _bin_of(win, 157.44)
        span = max(1, int(N_COLS * _WALL_MIN_BRIGHT * 1.1))
        win._bid_grid[:span, faint] = 90000.0
        return faint

    def _compare(self, win, qapp, monkeypatch, combined):
        import analysis.liq_hm_window as L

        win._bid_ask_cb.setChecked(not combined)
        self._book(win)
        win._wall_n_spin.setValue(max(win._wall_n_spin.value(), 5))
        sparse = _draw(win, qapp)
        monkeypatch.setattr(L, "_bright_norm", self._dense)
        dense = _draw(win, qapp)
        assert sparse, "no labels at all; the fixture says nothing"
        assert sparse == dense, (
            f"combined={combined}\n sparse: {sparse}\n dense : {dense}")

    def test_split(self, win, qapp, monkeypatch):
        self._compare(win, qapp, monkeypatch, combined=False)

    def test_combined(self, win, qapp, monkeypatch):
        self._compare(win, qapp, monkeypatch, combined=True)

    def test_the_fixture_puts_a_level_near_the_floor(self, win, qapp):
        """Otherwise the two tests above compare two easy answers.

        Asserted on the brightness itself rather than on the labels: a
        level that sits near the floor may or may not be labelled, and
        which way it falls is exactly what must not depend on the rewrite.
        """
        from analysis.liq_hm_window import _WALL_MIN_BRIGHT, _bright_norm

        faint = self._book(win)
        norm = _bright_norm(win._bid_grid[:N_COLS], win._gamma_spin.value())
        b = float(norm.mean(axis=0)[faint])
        assert 0.5 * _WALL_MIN_BRIGHT < b < 2.0 * _WALL_MIN_BRIGHT, (
            f"the near-floor level renders at {b:.4f} against a floor of "
            f"{_WALL_MIN_BRIGHT}; the comparison tests are not probing the "
            f"boundary they claim to")


class TestTheGammaControlCannotReachZero:
    """_bright_norm and _single_rgba both normalise the occupied cells and
    leave the rest at 0. That is byte-identical to the dense form only
    while `0 ** gamma == 0`, which needs gamma > 0.

    np.power(0, 0) is 1. At gamma 0 the dense form turns every empty cell
    full-bright -- measured, a whole-grid mean of 1.0000 against 0.1077 --
    and the sparse form leaves it black. At a negative gamma the dense
    form produces inf.

    The guard belongs on the control rather than in a runtime branch:
    gamma <= 0 is unreachable through the UI, so a check for it would be
    dead code, and the change that would actually break this is someone
    widening the range. That is what this watches.
    """

    def test_the_minimum_is_above_zero(self, win):
        assert win._gamma_spin.minimum() > 0.0, (
            "gamma can reach 0, where np.power(0, 0) is 1 -- the dense and "
            "sparse forms of _bright_norm and _single_rgba stop agreeing")

    def test_zero_gamma_really_would_diverge(self, win):
        """The reason, measured, so the assertion above is not folklore."""
        from analysis.liq_hm_window import _bright_norm, _percentile_norm

        g = np.zeros((N_COLS, 8))
        g[:, 3] = 1000.0
        dense = np.power(_percentile_norm(np.log1p(g)), 0.0)
        sparse = _bright_norm(g, 0.0)
        assert dense.mean() == 1.0, dense.mean()
        assert sparse.mean() < 0.2, sparse.mean()
