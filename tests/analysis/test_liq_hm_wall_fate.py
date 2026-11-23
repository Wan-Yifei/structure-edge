"""Why a wall shrank: bought, or withdrawn.

A snapshot shows only that a level lost size. The two causes are opposite
signals and look identical in the depth alone -- on 2026-10-14 a 73,072-share
bid at 116.00 vanished in 588 ms with not one share printing there, while two
hours earlier a level took 2,518 shares and held. Matching the loss against
what printed AT that price separates them.

The suffix is an estimate and the tests say where it declines to guess: wide
bins, no tick feed, an intact wall.

It names the LOST amount, because the share it reports is a share of that
loss and not of the order: "300 / 1,020  pulled 100%" was read, reasonably,
as "all of it went" while 300 sat there still resting. With the loss on
screen the denominator is unambiguous, and at the extremes the percentage is
dropped -- there is nothing left to apportion. NEUTRAL volume is excluded throughout --
off-exchange prints consume no displayed size, and counting them would read a
withdrawal as absorption, which is the one mistake that matters here.
"""

import os
from datetime import datetime, timedelta

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtWidgets import QApplication        # noqa: E402

T0 = datetime(2026, 10, 14, 11, 10, 0)
N_COLS = 40
N_BINS = 150
PMIN, BIN = 115.00, 0.01
WALL_BIN = int((116.00 - PMIN) / BIN)      # the 116.00 band


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app
    app.processEvents()


@pytest.fixture
def win(qapp):
    from analysis.liq_hm_window import LiqHmWindow
    w = LiqHmWindow()
    w._price_min = PMIN
    w._bin_size = BIN
    w._col_ts = [T0 + timedelta(seconds=i) for i in range(N_COLS)]
    w._bid_grid = np.zeros((N_COLS, N_BINS), dtype=float)
    w._ask_grid = np.zeros((N_COLS, N_BINS), dtype=float)
    w._absorb_ticks = []
    # A finished load that happened to find nothing, not "not loaded yet".
    w._absorb_last_ts = T0 + timedelta(seconds=N_COLS)
    w._absorb_cb.setChecked(True)
    yield w
    w.close()
    qapp.processEvents()


def ticks(spec, price=116.00, direction="SELL"):
    """spec: list of (col_offset, volume)."""
    return [{"ts": T0 + timedelta(seconds=c), "price": price,
             "volume": v, "direction": direction} for c, v in spec]


def fate(win, peak, now, *, half=0.005, side="B", peak_col=10):
    grid = win._bid_grid if side == "B" else win._ask_grid
    grid[:, WALL_BIN] = now
    grid[peak_col, WALL_BIN] = peak
    return win._wall_fate(grid, WALL_BIN, 0, N_COLS, peak, now, half, side)


class TestItSeparatesTheTwo:
    def test_pulled_when_nothing_printed(self, win):
        """The 116.00 case: 73,072 gone, nothing printed there."""
        assert fate(win, 73_072, 0) == "  −73,072  100% pulled / 0% eaten"

    def test_eaten_when_the_loss_printed_at_that_price(self, win):
        win._absorb_ticks = ticks([(12, 40_000), (14, 33_072)])
        assert fate(win, 73_072, 0) == "  −73,072  0% pulled / 100% eaten"

    def test_a_mix_reports_both_shares(self, win):
        win._absorb_ticks = ticks([(12, 20_000)])
        got = fate(win, 73_072, 0)
        assert "pulled" in got and "eaten" in got
        f = (73_072 - 20_000) / 73_072
        assert got == f"  −73,072  {f:.0%} pulled / {1 - f:.0%} eaten"

    def test_more_printed_than_was_lost_is_fully_eaten(self, win):
        """Refills mean prints can exceed the net loss; never negative."""
        win._absorb_ticks = ticks([(12, 500_000)])
        assert fate(win, 73_072, 0) == "  −73,072  0% pulled / 100% eaten"


class TestWhatDoesNotCount:
    def test_neutral_prints_are_not_absorption(self, win):
        """Off-exchange consumes no displayed size."""
        win._absorb_ticks = ticks([(12, 73_072)], direction="NEUTRAL")
        assert fate(win, 73_072, 0) == "  −73,072  100% pulled / 0% eaten"

    def test_buys_do_not_explain_a_bid_shrinking(self, win):
        win._absorb_ticks = ticks([(12, 73_072)], direction="BUY")
        assert fate(win, 73_072, 0) == "  −73,072  100% pulled / 0% eaten"

    def test_sells_do_not_explain_an_ask_shrinking(self, win):
        win._absorb_ticks = ticks([(12, 5_000)], direction="SELL")
        assert fate(win, 5_000, 0, side="A") == "  −5,000  100% pulled / 0% eaten"

    def test_buys_do_explain_an_ask_shrinking(self, win):
        win._absorb_ticks = ticks([(12, 5_000)], direction="BUY")
        assert fate(win, 5_000, 0, side="A") == "  −5,000  0% pulled / 100% eaten"

    def test_prints_before_the_peak_are_not_counted(self, win):
        """The wall still held its peak then, so they hit something else."""
        win._absorb_ticks = ticks([(2, 73_072)])
        assert fate(win, 73_072, 0, peak_col=10) == "  −73,072  100% pulled / 0% eaten"

    def test_prints_at_another_price_are_not_counted(self, win):
        win._absorb_ticks = ticks([(12, 73_072)], price=116.40)
        assert fate(win, 73_072, 0) == "  −73,072  100% pulled / 0% eaten"


class TestItDeclinesToGuess:
    def test_a_wide_bin_gets_no_suffix(self, win):
        """The price label already reads '116.00±0.25' at this width."""
        win._absorb_ticks = ticks([(12, 10_000)])
        assert fate(win, 73_072, 0, half=0.25) == ""

    def test_aggressor_off_gets_no_suffix(self, win):
        """Ticks only load while Aggressor is checked."""
        win._absorb_cb.setChecked(False)
        assert fate(win, 73_072, 0) == ""

    def test_a_load_that_has_not_finished_gets_no_suffix(self, win):
        """Empty-because-pending must not read as empty-because-nothing-traded."""
        win._absorb_last_ts = None
        assert fate(win, 73_072, 0) == ""

    def test_an_intact_wall_gets_no_suffix(self, win):
        win._absorb_ticks = ticks([(12, 10_000)])
        assert fate(win, 73_072, 73_072) == ""

    def test_a_level_that_was_never_there_gets_no_suffix(self, win):
        grid = win._bid_grid
        grid[:, WALL_BIN] = 0.0
        assert win._wall_fate(grid, WALL_BIN, 0, N_COLS,
                              1000.0, 0.0, 0.005, "B") == ""


class TestTheGateIsReachable:
    """The gate is a constant in dollars; the band it judges scales with price.

    So the feature has a PRICE CEILING, and the first version shipped above
    it: the cap was one cent, which no instrument this was written for could
    reach. It was raised to three, which clears SOXL. It does not clear QQQ,
    and nothing here noticed, because the parametrize below stopped at $130
    while QQQ trades near $615.

    These tests therefore pin TWO things, and the second is the one that was
    missing: that the constant is not narrowed below what the instruments in
    use need, AND what the ceiling actually is, written down as a number
    rather than implied by whichever mids someone happened to list.

    Do not "fix" a failure here by raising _WALL_FATE_MAX_HALF. The cap is
    not a tuning knob: the attribution matches prints against the band they
    land in, so a wider band credits a neighbour's volume to this wall.
    Measured on QQQ over one session, a band at the default resolution held
    about 11 price levels against about 3 on SOXL, and about 85% of the
    volume matched to a wall belonged to another level. Raising the cap would
    not make that right, it would publish it.
    """

    @staticmethod
    def _half_band(mid, near_width, n_price):
        """Half-band for a book whose near-touch levels span `near_width`.

        Mirrors _rebuild_grid exactly -- `span = max(near_width, mid*2%)` and
        the window is `near_width + span` wide -- because the shortcut that
        used to stand here did not. It assumed `near_width == span` and
        returned `mid*0.04/n/2`, described as "the narrowest band on offer".
        It is neither bound. It is the boundary between the two regimes, and
        measured on 2026-11-20 it overstated QQQ's real half-band by 1.9x and
        understated SOXL's by 1.3x -- wrong in both directions, so it could
        not even be relied on as conservative.
        """
        span = max(near_width, mid * 0.02)
        return ((near_width + span) / n_price) / 2

    def test_the_formula_matches_a_reading_taken_off_the_screen(self):
        """QQQ, 2026-11-20: the legend read $0.134/band at 100 bins.

        Pinning the arithmetic against a number that was actually on screen,
        because every claim below is computed with it.
        """
        from analysis.liq_hm_window import N_PRICE
        band = self._half_band(615.6, 1.09, N_PRICE) * 2
        assert abs(band - 0.134) < 0.0005, f"got ${band:.4f}/band"

    @pytest.mark.parametrize("mid,near_width", [
        (45.0, 1.2), (103.0, 1.2), (116.0, 1.2), (130.0, 1.2),
    ])
    def test_default_resolution_can_show_a_suffix(self, win, mid, near_width):
        from analysis.liq_hm_window import N_PRICE
        win._absorb_ticks = ticks([(12, 73_072)], price=116.40)
        half = self._half_band(mid, near_width, N_PRICE)
        assert fate(win, 73_072, 0, half=half) != "", (
            f"no suffix at mid={mid} with N_PRICE={N_PRICE} "
            f"(half={half:.4f}) -- the gate is unreachable again")

    def test_a_high_priced_instrument_is_above_the_ceiling(self, win):
        """QQQ at the default resolution. Not a bug being pinned as correct:
        the suffix is refused because the attribution would be wrong, and
        TestItSaysWhyItIsSilent pins that the UI says so."""
        from analysis.liq_hm_window import N_PRICE
        win._absorb_ticks = ticks([(12, 73_072)], price=116.40)
        half = self._half_band(615.6, 1.09, N_PRICE)
        assert fate(win, 73_072, 0, half=half) == ""

    def test_the_ceiling_is_about_250_dollars(self):
        """The number the parametrize above only implied.

        For a book whose near-touch levels span ~$1 at the default
        resolution: half = (1 + mid*2%) / 200 <= 0.03 => mid <= $250. Stated
        as a fact so the next high-priced instrument is a known limitation
        rather than a surprise.
        """
        from analysis.liq_hm_window import N_PRICE, _WALL_FATE_MAX_HALF
        ok = [m for m in range(10, 1000)
              if self._half_band(m, 1.0, N_PRICE) <= _WALL_FATE_MAX_HALF]
        assert max(ok) == 250, f"ceiling moved to ${max(ok)}"

    def test_a_quarter_dollar_band_is_still_refused(self, win):
        win._absorb_ticks = ticks([(12, 10_000)])
        assert fate(win, 73_072, 0, half=0.25) == ""

    def test_the_cutoff_is_at_least_three_ticks(self):
        """Three cents is three ticks at US equity tick size; below that the
        default heatmap resolution stops clearing it."""
        from analysis.liq_hm_window import _WALL_FATE_MAX_HALF
        assert _WALL_FATE_MAX_HALF >= 0.03


class TestItSaysWhyItIsSilent:
    """An absence the user cannot account for is the actual defect.

    Reported as "QQQ never shows pulled/eaten". Everything about why was
    off screen: the gate is a half-band compared against a constant, and
    neither number was anywhere in the UI. The band width was -- in the
    legend, as $/band -- so the explanation goes beside it.

    The load-bearing test here is not that a note appears. It is that the
    Bins value the note names ACTUALLY CLEARS THE GATE: advice that does not
    work is worse than silence, because it costs the user a round trip to
    find that out.
    """

    @staticmethod
    def _legend_with(win, lo, hi, n_price):
        """Order matters: setValue calls _reset_grid, which clears bin_size.

        The first version of this helper set bin_size first and passed only
        because the fixture's spin box already held 100, so setValue fired no
        signal. It would have gone green against a window whose width had
        been zeroed.
        """
        win._n_price_spin.setValue(n_price)
        win._price_min, win._price_max = lo, hi
        win._bin_size = (hi - lo) / n_price
        win._update_legend()
        return win._legend_lbl.text()

    def test_the_advice_does_not_depend_on_the_current_bins(self, win):
        """Same window, different starting resolution -> same answer.

        The advice is about the WINDOW, so it must be invariant to the knob
        it is advising on. Reading it off price_min/price_max instead of off
        bin_size broke exactly this.
        """
        self._legend_with(win, 609.0, 622.4, 100)
        a = win._wall_fate_min_bins()
        self._legend_with(win, 609.0, 622.4, 200)
        b = win._wall_fate_min_bins()
        assert a == b == 240, f"got {a} at 100 bins and {b} at 200"

    def test_a_wide_band_says_so(self, win):
        txt = self._legend_with(win, 609.0, 622.4, 100)       # QQQ, $0.134
        assert "no pulled/eaten" in txt
        assert "needs Bins" in txt

    def test_the_threshold_it_prints_is_the_one_it_enforces(self, win):
        """The note quotes a BAND; the gate compares a HALF-band.

        It shipped quoting $0.03 -- the half-band constant -- right beside a
        "needs Bins" figure derived from the real $0.06 cut-off, so the two
        halves of one sentence disagreed by 2x and a reader taking the stated
        threshold at face value would have asked for twice the resolution
        they needed. Nothing pinned the printed number to the gate, so this
        reads it back out of the rendered text and flips the gate across it.
        """
        import re
        txt = self._legend_with(win, 609.0, 622.4, 100)
        m = re.search(r"band &gt; \$([0-9.]+)", txt)
        assert m, f"no threshold in legend: {txt}"
        printed = float(m.group(1))

        lo, bins = 600.0, 100
        for factor, want_note in ((0.99, False), (1.01, True)):
            band = printed * factor
            t = self._legend_with(win, lo, lo + band * bins, bins)
            assert ("no pulled/eaten" in t) is want_note, (
                f"band ${band:.4f} against a printed cut-off of ${printed:.2f}: "
                f"note {'missing' if want_note else 'should not be there'}")

    def test_the_bins_it_names_actually_clear_the_gate(self, win):
        """The one that matters: take its own advice and check it works."""
        from analysis.liq_hm_window import _WALL_FATE_MAX_HALF
        lo, hi = 609.0, 622.4
        self._legend_with(win, lo, hi, 100)
        need = win._wall_fate_min_bins()
        assert need is not None
        half = ((hi - lo) / need) / 2
        assert half <= _WALL_FATE_MAX_HALF, (
            f"it told the user Bins={need}, which still gives half={half:.4f} "
            f"against a limit of {_WALL_FATE_MAX_HALF}")

    def test_it_does_not_name_more_than_needed(self, win):
        """One spin-box step below its answer must NOT clear the gate, or it
        is quietly costing resolution it did not have to."""
        from analysis.liq_hm_window import _WALL_FATE_MAX_HALF
        lo, hi = 609.0, 622.4
        self._legend_with(win, lo, hi, 100)
        need = win._wall_fate_min_bins()
        step = win._n_price_spin.singleStep()
        half_below = ((hi - lo) / (need - step)) / 2
        assert half_below > _WALL_FATE_MAX_HALF, (
            f"Bins={need - step} would have been enough; {need} is wasteful")

    def test_the_value_it_names_can_be_set(self, win):
        """A number the spin box would clamp is not advice."""
        self._legend_with(win, 609.0, 622.4, 100)
        need = win._wall_fate_min_bins()
        win._n_price_spin.setValue(need)
        assert win._n_price_spin.value() == need

    def test_a_narrow_band_says_nothing(self, win):
        txt = self._legend_with(win, 46.5, 47.7, 100)         # SOXL, $0.012
        assert "pulled/eaten" not in txt

    def test_an_unreachable_width_says_so(self, win):
        """A book spreading past 2% of price outruns the spin box, and naming
        a number it cannot be set to would be a lie. Real case, not a
        defensive branch -- see _wall_fate_min_bins."""
        wide = 0.06 * win._n_price_spin.maximum() * 1.5
        txt = self._legend_with(win, 600.0, 600.0 + wide, 100)
        assert win._wall_fate_min_bins() is None
        assert "too wide at any Bins" in txt
        assert "needs Bins" not in txt

    def test_the_note_is_not_shown_when_the_suffix_can_appear(self, win):
        """Pins the direction of the condition: the first draft of this check
        can pass while inverted, because both branches mention the feature."""
        narrow = self._legend_with(win, 46.5, 47.7, 100)
        wide = self._legend_with(win, 609.0, 622.4, 100)
        assert "no pulled/eaten" in wide and "no pulled/eaten" not in narrow


class TestItNamesTheLoss:
    """The share is OF THE LOSS, so the loss has to be on screen.

    Reported from a live panel: "B 98.81±0.02  300 / 1,020  pulled 100%".
    The arithmetic was right -- 720 shares left and every one of them was
    withdrawn -- but a bare 100% beside a level still holding 300 reads as
    "all of it went". The number the percentage applies to was the only
    thing missing.
    """

    def test_a_partial_loss_names_what_left_not_what_remains(self, win):
        got = fate(win, 1_020, 300)
        assert "720" in got, got
        assert "1,020" not in got and "300" not in got, got

    def test_no_bare_percentage_without_its_denominator(self, win):
        """Any percentage printed must have the lost amount beside it."""
        for peak, now in ((1_020, 300), (73_072, 0), (5_000, 4_000)):
            got = fate(win, peak, now)
            if "%" in got:
                assert "\u2212" in got, got

    def test_a_whole_loss_still_prints_the_pair(self, win):
        """Dropping the share at the extremes was shorter and got asked
        about: one column in two formats reads as a missing field."""
        got = fate(win, 1_020, 0)
        assert got == "  \u22121,020  100% pulled / 0% eaten", got


class TestTheMixedFormIsASplit:
    """Two shares that sum to 100 declare themselves a partition.

    One share alone had to be read against a denominator that was never on
    screen, which is how "pulled 100%" came to sit beside a level still
    holding 300.
    """

    def test_the_two_shares_sum_to_one_hundred(self, win):
        win._absorb_ticks = ticks([(12, 20_000)])
        got = fate(win, 73_072, 0)
        import re
        pcts = [int(x) for x in re.findall(r"(\d+)%", got)]
        assert len(pcts) == 2, got
        assert sum(pcts) == 100, got

    def test_both_words_appear_whatever_the_split(self, win):
        """Including at the extremes: one shape to learn, not two."""
        win._absorb_ticks = ticks([(12, 20_000)])
        mixed = fate(win, 73_072, 0)
        win._absorb_ticks = ticks([])
        pure = fate(win, 73_072, 0)
        for got in (mixed, pure):
            assert "pulled" in got and "eaten" in got, got


class TestItAgreesWithTheSizeColumn:
    """The two fields must not contradict each other about one wall.

    The size column prints the peak alone once a wall is _WALL_INTACT_FRAC
    whole -- i.e. it reports no loss. The suffix used to explain a loss
    anyway, so a label read "925  −70 pulled": a loss, against a peak the
    label never showed, beside a field saying nothing had gone.
    """

    def test_a_nearly_whole_wall_gets_no_suffix(self, win):
        from analysis.liq_hm_window import _WALL_INTACT_FRAC

        win._absorb_ticks = ticks([])
        peak = 1_000
        now = int(peak * _WALL_INTACT_FRAC) + 1      # inside "intact"
        assert fate(win, peak, now) == ""

    def test_just_past_intact_does_get_one(self, win):
        from analysis.liq_hm_window import _WALL_INTACT_FRAC

        win._absorb_ticks = ticks([])
        peak = 1_000
        now = int(peak * _WALL_INTACT_FRAC) - 1      # outside it
        assert fate(win, peak, now) != ""
