"""The crosshair's inverse-pair price tag.

Hovering a level on SOXL also shows where SOXS would be at that level. Both
are +/-3x the same index and reset daily from the previous close, so from that
shared anchor a SOXL return of 3r implies a SOXS return of -3r.

The estimator is pure arithmetic over a cached anchor, so it is tested against
the widget without a running quote feed: _pair_anchor is what a live snapshot
would have put there.
"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtWidgets import QApplication   # noqa: E402

# Live quotes taken while building this, used as the reference case.
SOXL_PC, SOXS_PC = 146.33, 33.63
SOXL_LAST, SOXS_LAST = 151.845, 32.350

ANCHOR = ("US.SOXS", SOXL_PC, SOXS_PC)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app
    app.processEvents()


@pytest.fixture()
def viewer(qapp):
    from analysis.trade_viewer_qt import TradeViewerQt

    v = TradeViewerQt.__new__(TradeViewerQt)   # no __init__: no feed, no timers
    v._code = "US.SOXL"
    v._pair_anchor = ANCHOR
    return v


class TestImpliedPrice:
    def test_matches_the_live_quote_it_was_derived_from(self, viewer):
        """The whole feature rests on this relation actually holding."""
        code, px = viewer._pair_implied(SOXL_LAST)
        assert code == "US.SOXS"
        assert px == pytest.approx(SOXS_LAST, abs=0.05), (
            f"predicted {px:.3f} against {SOXS_LAST} actual")

    def test_relative_error_is_under_a_tenth_of_a_percent(self, viewer):
        _, px = viewer._pair_implied(SOXL_LAST)
        assert abs(px - SOXS_LAST) / SOXS_LAST < 0.001

    def test_at_the_anchor_both_legs_read_their_own_prev_close(self, viewer):
        _, px = viewer._pair_implied(SOXL_PC)
        assert px == pytest.approx(SOXS_PC), "zero move must be exact"

    def test_moves_opposite_to_the_hovered_price(self, viewer):
        _, up   = viewer._pair_implied(SOXL_PC * 1.05)
        _, down = viewer._pair_implied(SOXL_PC * 0.95)
        assert up < SOXS_PC < down

    def test_symmetric_around_the_anchor(self, viewer):
        _, up   = viewer._pair_implied(SOXL_PC * 1.05)
        _, down = viewer._pair_implied(SOXL_PC * 0.95)
        assert (SOXS_PC - up) == pytest.approx(down - SOXS_PC)

    def test_leverage_cancels_so_percent_moves_mirror_one_to_one(self, viewer):
        """Both legs are 3x, so the 3s cancel: SOXL +4% <-> SOXS -4%."""
        _, px = viewer._pair_implied(SOXL_PC * 1.04)
        assert px / SOXS_PC == pytest.approx(0.96)


class TestWhenNothingIsShown:
    def test_no_anchor(self, viewer):
        viewer._pair_anchor = None
        assert viewer._pair_implied(150.0) is None

    def test_symbol_without_a_pair(self, viewer):
        viewer._code = "US.AAPL"
        assert viewer._pair_implied(150.0) is None

    def test_anchor_left_over_from_another_symbol(self, viewer):
        """Switching symbols must not read the previous one's anchor."""
        viewer._code = "US.TQQQ"
        assert viewer._pair_implied(150.0) is None

    def test_no_code_yet(self, viewer):
        viewer._code = ""
        assert viewer._pair_implied(150.0) is None

    def test_non_positive_implied_price_is_withheld(self, viewer):
        """Past +200% on SOXL the relation puts SOXS at or below zero."""
        assert viewer._pair_implied(SOXL_PC * 3.0) is None
        assert viewer._pair_implied(SOXL_PC * 2.0) is None   # exactly zero

    def test_just_inside_the_limit_still_shows(self, viewer):
        got = viewer._pair_implied(SOXL_PC * 1.99)
        assert got is not None and got[1] > 0

    def test_zero_anchor_is_not_divided_by(self, viewer):
        viewer._pair_anchor = ("US.SOXS", 0.0, SOXS_PC)
        assert viewer._pair_implied(150.0) is None


class TestPairTable:
    def test_only_soxl_is_configured(self):
        from analysis.trade_viewer_qt import _INVERSE_PAIR
        assert _INVERSE_PAIR == {"US.SOXL": "US.SOXS"}

    def test_short_code_strips_the_market_prefix(self):
        from analysis.trade_viewer_qt import _short_code
        assert _short_code("US.SOXL") == "SOXL"
        assert _short_code("SOXL") == "SOXL"
        assert _short_code("") == ""
