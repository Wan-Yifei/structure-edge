"""The bin COUNT follows the window, so the band can follow the gate.

A fixed count sets the resolution from whatever decides the window, and
which of the two does varies by symbol. Measured on db/order_book.db, 400
snapshots a window:

    SOXL  window 2.42  ->  100 bins = $0.0121 half-band, 2.5x inside the
          $0.03 gate. Its near-touch span is 2.8-3.3% of price, so the
          +/-1% floor never binds and the band is the same at mid 33, 36,
          43 and 50 -- it does not move with price at all.
    QQQ   window 14.56 ->  100 bins = $0.0728, more than double the gate.
          Here the floor decides the window in every snapshot.

So "cheap stock, fine band" is not the rule; "whichever of book and floor
is wider" is. The suffix was therefore dead by default on
anything expensive, reported twice as "QQQ never shows it" before anyone
connected it to a control named Bins.

The widths below come from the real book, over three sessions: QQQ
12.8-14.8 (median 14.6), SOXL 2.2-2.8 (median 2.4).
"""
import math
import types

import pytest

from analysis.liq_hm_window import (
    LiqHmWindow, _WALL_FATE_MAX_BAND, _WALL_FATE_MAX_HALF)


def _win(floor: int = 100, step: int = 20, maximum: int = 500):
    """Just enough window to answer _bins_for_window.

    A stand-in rather than a real LiqHmWindow: the function is pure and
    reads only the spin box, and building the widget costs a second and an
    OpenD context.
    """
    s = types.SimpleNamespace(_n_price_spin=types.SimpleNamespace(
        value=lambda: floor, singleStep=lambda: step,
        maximum=lambda: maximum))
    # Bound, because _bins_for_window delegates to _bins_needed_for_gate --
    # the shared arithmetic that keeps the legend's advice and the grid's
    # real count from drifting. Same pattern the grid-fidelity fakes use.
    s._bins_needed_for_gate = types.MethodType(
        LiqHmWindow._bins_needed_for_gate, s)
    s._bins_for_window = types.MethodType(LiqHmWindow._bins_for_window, s)
    return s


def bins(width: float, floor: int = 100, **kw) -> int:
    return _win(floor, **kw)._bins_for_window(width)


def half_band(width: float, n: int) -> float:
    return width / n / 2


class TestTheGateIsClearedWhereItUsedToBeUnreachable:
    """QQQ is the case this exists for; SOXL is the control that must not
    move."""

    @pytest.mark.parametrize("width", [13.2, 15.48, 16.8])
    def test_a_high_priced_window_is_given_enough_bins(self, width):
        n = bins(width)
        assert n > 100, f"{width} still at the floor"
        assert half_band(width, n) <= _WALL_FATE_MAX_HALF, (
            f"{width} wide at {n} bins gives {half_band(width, n):.4f}, "
            f"over the {_WALL_FATE_MAX_HALF} gate")

    @pytest.mark.parametrize("width", [2.18, 2.52, 2.76])
    def test_a_low_priced_window_is_left_alone(self, width):
        """SOXL clears the gate 2.5x over at 100 bins -- measured
        ($0.0121 a half-band), not the 8x a floor-only calculation gives:
        its own book is wider than the 1% floor, so the floor never
        decides. Raising the count there would cost grid and render time
        for nothing, and would push the band finer than the $0.01 tick,
        which the module deliberately avoids."""
        assert bins(width) == 100
        assert half_band(width, 100) <= _WALL_FATE_MAX_HALF


class TestItIsAFloorAndNotARatchet:

    def test_a_finer_setting_is_never_lowered(self):
        """Finer than the gate needs is the user's call."""
        assert bins(13.2, floor=400) == 400
        assert bins(2.52, floor=300) == 300

    def test_the_count_comes_back_down_when_the_window_does(self):
        """Recomputed from the floor every time, not from the last value.

        An earlier design raised the spin box itself and only ever raised,
        so one wide pre-market book pinned it for the session and carried
        that into the next symbol -- SOXL would have inherited QQQ's 240+
        and drawn bins finer than a tick.
        """
        wide = bins(16.8)
        narrow = bins(2.52)
        assert wide > narrow == 100

    def test_the_control_is_never_written(self):
        """Pure. The spin box stays the user's, which is what removes the
        need for a cap, a reset on symbol change and a blockSignals dance.
        """
        w = _win(100)
        before = w._n_price_spin.value()
        bins_ = w._bins_for_window(16.8)
        assert bins_ > before
        assert w._n_price_spin.value() == before


class TestWhatItRefusesToDo:

    def test_a_window_too_wide_for_any_setting_falls_back_to_the_floor(self):
        """NOT to the maximum. At $40 wide even 500 bins gives $0.04 a
        half-band, still over the gate -- so paying 5x the grid and render
        cost buys nothing, and the legend's "too wide at any Bins" is then
        the true thing to show."""
        width = 40.0
        assert half_band(width, 500) > _WALL_FATE_MAX_HALF, (
            "fixture no longer exceeds the cap; pick a wider window")
        assert bins(width) == 100

    def test_it_never_exceeds_the_control_maximum(self):
        assert bins(29.0) <= 500

    @pytest.mark.parametrize("width", [0.0, -1.0])
    def test_a_window_with_no_width_returns_the_floor(self, width):
        assert bins(width) == 100

    def test_the_answer_lands_on_a_step_the_control_can_hold(self):
        """Advice the spin box would silently clamp is not advice, and the
        same rounding is what _wall_fate_min_bins already does."""
        for width in (13.2, 15.48, 16.8, 21.0):
            assert bins(width) % 20 == 0, width


class TestItAgreesWithTheAdviceItActsOn:
    """_wall_fate_min_bins tells the user what is needed; this computes the
    same thing and applies it. The two drifting apart is how the legend came
    to advise a number the gate did not actually want."""

    @pytest.mark.parametrize("width", [2.52, 13.2, 16.8, 21.0])
    def test_the_advice_and_the_action_are_the_same_number(self, width):
        """Not recomputed here: both read the one shared function, and the
        point is that neither has its own copy to drift."""
        w = _win()
        need = w._bins_needed_for_gate(width)
        assert bins(width) == max(100, need)
        # And that number really does clear the gate -- a shared mistake is
        # still a mistake.
        assert half_band(width, need) <= _WALL_FATE_MAX_HALF


class TestTheAdviceMeasuresTheGridNotTheControl:
    """A SYNTHETIC state in which the two numbers differ.

    Not reachable in production: with the derived count in place, the only
    time _wall_fate_min_bins runs is the fall-back one, where the grid is
    exactly the floor wide and reading either number gives the same answer.
    This pins the decoupling so the function does not come to depend on
    that coincidence.

    `_wall_fate_min_bins` derives the window as bin_size x <count>. Once
    the grid can run wider than the user's floor, reading the floor there
    under-reports the window by exactly that ratio, so the legend advises a
    number smaller than the one already in use. The existing wall-fate
    tests cannot see it: their helper sets the grid and the spin box to the
    same value.
    """

    def _win_with_mismatch(self):
        """A 13.2-wide window running at 240 bins behind a floor of 100."""
        import numpy as np
        w = _win(100)
        w._bin_size = 13.2 / 240
        w._bid_grid = np.zeros((4, 240))
        w._wall_fate_min_bins = types.MethodType(
            LiqHmWindow._wall_fate_min_bins, w)
        return w

    def test_the_advice_sees_the_real_window(self):
        w = self._win_with_mismatch()
        got = w._wall_fate_min_bins()
        # The true window is 13.2 wide and needs 220. Reading the spin box
        # instead makes it look like 13.2/240*100 = 5.5 wide, which needs
        # nothing at all -- the legend would advise the floor, 100, on a
        # panel whose wall labels are silent. "You are already fine" is the
        # worst possible thing to say there.
        assert got == w._bins_needed_for_gate(13.2) == 220, got
        assert got > 100, (
            f"advice {got} is the untouched floor; it measured the control "
            f"instead of the grid")

    def test_the_advice_matches_what_the_code_would_choose(self):
        w = self._win_with_mismatch()
        assert w._wall_fate_min_bins() == w._bins_needed_for_gate(13.2)
