"""The crosshair's inverse-pair price tag.

Hovering a level on SOXL also shows where SOXS would be at that level. Both
are +/-3x the same index and reset daily from the previous close, so from that
shared anchor a SOXL return of 3r implies a SOXS return of -3r.

These build a real TradeViewerQt and set the symbol through the code box the
viewer actually reads. The first version of these tests instead built a bare
object and assigned a `self._code` attribute -- one the window does not have.
That made every test pass while the shipped code raised AttributeError on each
mouse move and took the ordinary price tag down with it. Nothing here may
invent state: only _pair_anchor is set directly, because it is filled from a
live market snapshot.
"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtCore import QPointF        # noqa: E402
from PyQt6.QtWidgets import QApplication   # noqa: E402

# Live quotes taken while building this, used as the reference case.
SOXL_PC, SOXS_PC = 146.33, 33.63
SOXL_LAST, SOXS_LAST = 151.845, 32.350

# The shipped anchor is the live pair; this one is the previous close, which
# the code falls back to and which makes SOXL_LAST a genuine out-of-sample
# prediction rather than the anchor reading itself back.
ANCHOR = ("US.SOXL", "US.SOXS", SOXL_PC, SOXS_PC)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app
    app.processEvents()


@pytest.fixture(scope="module")
def viewer(qapp):
    from analysis.trade_viewer_qt import TradeViewerQt

    v = TradeViewerQt()
    v.resize(1400, 900)
    v.show()
    for _ in range(30):
        qapp.processEvents()
    yield v
    v.close()
    qapp.processEvents()


@pytest.fixture()
def soxl(viewer):
    """Viewer showing SOXL with the anchor a live snapshot would have left."""
    viewer._code_edit.setText("US.SOXL")
    viewer._pair_anchor = ANCHOR
    yield viewer
    viewer._pair_anchor = None


class TestImpliedPrice:
    def test_matches_the_live_quote_it_was_derived_from(self, soxl):
        """The whole feature rests on this relation actually holding."""
        base, pair, px = soxl._pair_implied(SOXL_LAST)
        assert (base, pair) == ("US.SOXL", "US.SOXS")
        assert px == pytest.approx(SOXS_LAST, abs=0.05), (
            f"predicted {px:.3f} against {SOXS_LAST} actual")

    def test_relative_error_is_under_a_tenth_of_a_percent(self, soxl):
        assert abs(soxl._pair_implied(SOXL_LAST)[2] - SOXS_LAST) / SOXS_LAST < 0.001

    def test_at_the_anchor_the_pair_reads_its_own_reference(self, soxl):
        """Why a live anchor beats the previous close: at the anchor price the
        estimate is exact, so error is only drift over the offset from it."""
        assert soxl._pair_implied(SOXL_PC)[2] == pytest.approx(SOXS_PC)

    def test_a_live_anchor_is_exact_at_the_live_price(self, soxl):
        soxl._pair_anchor = ("US.SOXL", "US.SOXS", SOXL_LAST, SOXS_LAST)
        assert soxl._pair_implied(SOXL_LAST)[2] == pytest.approx(SOXS_LAST)

    def test_moves_opposite_to_the_hovered_price(self, soxl):
        up   = soxl._pair_implied(SOXL_PC * 1.05)[2]
        down = soxl._pair_implied(SOXL_PC * 0.95)[2]
        assert up < SOXS_PC < down

    def test_symmetric_around_the_anchor(self, soxl):
        up   = soxl._pair_implied(SOXL_PC * 1.05)[2]
        down = soxl._pair_implied(SOXL_PC * 0.95)[2]
        assert (SOXS_PC - up) == pytest.approx(down - SOXS_PC)

    def test_leverage_cancels_so_percent_moves_mirror_one_to_one(self, soxl):
        """Both legs are 3x, so the 3s cancel: SOXL +4% <-> SOXS -4%."""
        assert soxl._pair_implied(SOXL_PC * 1.04)[2] / SOXS_PC == pytest.approx(0.96)


class TestWhenNothingIsShown:
    def test_no_anchor(self, soxl):
        soxl._pair_anchor = None
        assert soxl._pair_implied(150.0) is None

    def test_symbol_switched_away_from_the_anchor(self, soxl):
        """_trigger_fetch returns early on a live code change, so the old
        anchor outlives the switch until the next load completes."""
        soxl._code_edit.setText("US.AAPL")
        assert soxl._pair_implied(150.0) is None

    def test_empty_code_box(self, soxl):
        soxl._code_edit.setText("")
        assert soxl._pair_implied(150.0) is None

    def test_whitespace_in_the_box_does_not_break_the_match(self, soxl):
        soxl._code_edit.setText("  US.SOXL  ")
        assert soxl._pair_implied(SOXL_LAST) is not None

    def test_non_positive_implied_price_is_withheld(self, soxl):
        """Past +200% on SOXL the relation puts SOXS at or below zero."""
        assert soxl._pair_implied(SOXL_PC * 3.0) is None
        assert soxl._pair_implied(SOXL_PC * 2.0) is None   # exactly zero

    def test_just_inside_the_limit_still_shows(self, soxl):
        got = soxl._pair_implied(SOXL_PC * 1.99)
        assert got is not None and got[2] > 0

    def test_zero_anchor_is_not_divided_by(self, soxl):
        soxl._pair_anchor = ("US.SOXL", "US.SOXS", 0.0, SOXS_PC)
        assert soxl._pair_implied(150.0) is None


class TestCrosshairLabels:
    """Drives the real mouse-move handler. This is the layer that broke."""

    @staticmethod
    def _hover(v):
        v._on_mouse_move(QPointF(v._plot_c.sceneBoundingRect().center()))

    def test_ordinary_symbol_still_shows_its_price(self, viewer):
        """The regression: a throw in the pair lookup takes the gold tag with
        it, so the chart loses the price readout it always had."""
        viewer._code_edit.setText("US.AAPL")
        viewer._pair_anchor = None
        self._hover(viewer)
        assert viewer._price_label.isVisible()
        assert viewer._price_label.textItem.toPlainText() != ""
        assert not viewer._pair_price_label.isVisible()

    def test_pair_symbol_shows_both_tags(self, soxl):
        self._hover(soxl)
        assert soxl._price_label.isVisible()
        assert soxl._pair_price_label.isVisible()
        assert soxl._price_label.textItem.toPlainText().startswith("SOXL ")
        assert soxl._pair_price_label.textItem.toPlainText().startswith("SOXS ")

    def test_tickers_appear_only_in_pair_mode(self, viewer):
        viewer._code_edit.setText("US.AAPL")
        viewer._pair_anchor = None
        self._hover(viewer)
        assert "AAPL" not in viewer._price_label.textItem.toPlainText()

    def test_switching_away_hides_the_pair_tag(self, soxl):
        self._hover(soxl)
        assert soxl._pair_price_label.isVisible()
        soxl._code_edit.setText("US.AAPL")
        self._hover(soxl)
        assert not soxl._pair_price_label.isVisible()
        assert soxl._price_label.isVisible(), "the gold tag must survive"


class TestAnchorSelection:
    """_pair_refs picks what both legs are measured from."""

    @staticmethod
    def _refs(base, pair):
        from analysis.trade_viewer_qt import _pair_refs
        return _pair_refs(base, pair)

    def test_prefers_the_live_pair(self):
        assert self._refs({"last_price": 151.8, "prev_close_price": 146.3},
                          {"last_price": 32.3, "prev_close_price": 33.6}) == (151.8, 32.3)

    def test_falls_back_to_prev_close_when_neither_has_traded(self):
        assert self._refs({"last_price": 0, "prev_close_price": 146.3},
                          {"last_price": 0, "prev_close_price": 33.6}) == (146.3, 33.6)

    def test_one_leg_untraded_drops_both_to_prev_close(self):
        """Pairing one leg's live price with the other's close would compare
        two different moments and make the ratio meaningless."""
        assert self._refs({"last_price": 151.8, "prev_close_price": 146.3},
                          {"last_price": 0, "prev_close_price": 33.6}) == (146.3, 33.6)

    def test_nothing_usable(self):
        assert self._refs({"last_price": 0, "prev_close_price": 0},
                          {"last_price": 0, "prev_close_price": 0}) is None

    def test_missing_keys_are_not_an_error(self):
        assert self._refs({}, {}) is None

    def test_none_values_are_not_an_error(self):
        assert self._refs({"last_price": None, "prev_close_price": 146.3},
                          {"last_price": None, "prev_close_price": 33.6}) == (146.3, 33.6)

    def test_unparseable_value_falls_through(self):
        assert self._refs({"last_price": "n/a", "prev_close_price": 146.3},
                          {"last_price": "n/a", "prev_close_price": 33.6}) == (146.3, 33.6)


class TestPairTable:
    def test_only_soxl_is_configured(self):
        from analysis.trade_viewer_qt import _INVERSE_PAIR
        assert _INVERSE_PAIR == {"US.SOXL": "US.SOXS"}

    def test_short_code_strips_the_market_prefix(self):
        from analysis.trade_viewer_qt import _short_code
        assert _short_code("US.SOXL") == "SOXL"
        assert _short_code("SOXL") == "SOXL"
        assert _short_code("") == ""
