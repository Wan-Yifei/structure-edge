"""The tick profile scales to its own bars, not to the price chart.

Both draw paths used to end by copying the main chart's Y viewport "so
profile aligns spatially". That alignment is worth having only while the
chart happens to be zoomed near the candle in question. In the home
(reset) view it never is: a minute of ticks spans a few cents while the
axis, stretched by option walls, spans tens of dollars -- so every bar
lands in the same sliver and the panel reads as a smear. Reported with
the axis at 79-90 and the data at 83.

`_fit_profile_y` takes only a widget with setYRange, so these run without
Qt and without the viewer fixture -- which builds a real OpenQuoteContext
and blocks whenever OpenD is up but not answering.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import analysis.trade_viewer_qt as tv  # noqa: E402


class FakePanel:
    """Records the last range it was given."""

    def __init__(self):
        self.lo = self.hi = None

    def setYRange(self, lo, hi, padding=0):
        self.lo, self.hi = lo, hi


# The method touches nothing on `self`, so it is taken off the class and
# called with None. A QWidget subclass cannot be instantiated without Qt,
# and instantiating the real viewer is what drags in OpenD.
_fit = tv.TradeViewerQt.__dict__["_fit_profile_y"]


def fit(prices, bin_h=0.0, y_lo=None, y_hi=None):
    pw = FakePanel()
    _fit(None, pw, prices, bin_h, y_lo, y_hi)
    return pw


class TestItFitsItsOwnData:
    def test_a_minute_of_ticks_fills_the_panel(self):
        """The reported case: cents of data must not be drawn against an
        axis of dollars."""
        pw = fit([83.10, 83.15, 83.20], y_lo=83.08, y_hi=83.22)
        span = pw.hi - pw.lo
        assert span < 0.5, f"range is {span:.2f} wide for 14c of data"
        assert pw.lo < 83.08 and pw.hi > 83.22

    def test_the_candle_bounds_are_included_even_where_nothing_traded(self):
        """A wick that traded nothing is information about the candle; a
        range taken from traded prices alone would crop it off."""
        pw = fit([83.10], y_lo=82.00, y_hi=84.00)
        assert pw.lo < 82.00 and pw.hi > 84.00

    def test_the_padding_is_ten_percent_each_side(self):
        pw = fit([100.0, 110.0])
        assert pw.lo == 99.0 and pw.hi == 111.0

    def test_bin_height_counts_toward_the_top(self):
        """Bars are drawn upward from their price, so the top bar needs its
        own height inside the view or it is clipped."""
        pw = fit([100.0, 110.0], bin_h=2.0)
        assert pw.hi > 112.0

    def test_one_price_still_gets_a_view_to_sit_in(self):
        """A doji with every print at one price. lo == hi would be a
        zero-height view, which renders nothing at all."""
        pw = fit([100.0])
        assert pw.hi > pw.lo
        assert pw.hi - pw.lo < 1.0, "a single price should stay tight"

    def test_no_prices_leaves_the_range_alone(self):
        """Nothing to show: better to keep the last range than to invent
        one around zero and send the axis somewhere meaningless."""
        pw = fit([])
        assert pw.lo is None and pw.hi is None


class TestItIgnoresTheMainChart:
    def test_the_source_no_longer_reads_the_price_chart_viewport(self):
        """The old two-line sync is what this change removes. A test on
        _fit_profile_y alone would still pass with the sync re-added after
        it, because the sync had the last word."""
        src = (ROOT / "analysis" / "trade_viewer_qt.py").read_text(
            encoding="utf-8")
        head, _, tail = src.partition("def _draw_tick_profile")
        body = tail[:tail.index("def _show_range_tick_profile")]
        assert "_plot_c.vb.viewRange()" not in body, \
            "the tick profile is reading the price chart's Y again"
        assert body.count("_fit_profile_y") >= 2, \
            "both the order-flow and VP paths should fit their own data"
