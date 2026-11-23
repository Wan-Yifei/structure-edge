"""The option delta flow data layer.

Written after a review found the module shipped with no tests at all. Each
class below pins one thing that, if it broke, would produce a plausible
chart rather than an error -- that is the failure mode this module invites:
every quantity here is a number nobody can eyeball for correctness.
"""
from datetime import date

import pandas as pd
import pytest

from strategy.option.delta_flow import (
    CONTRACT_MULTIPLIER, MAX_GREEKS_AGE_S, MAX_QUOTE_AGE_S,
    attach_delta, by_bar, trading_day_of)

T0 = pd.Timestamp("2026-11-23 10:00:00")


def _ticks(rows):
    """rows: (offset_s, direction, volume, opt_type)."""
    return pd.DataFrame([{
        "code": f"US.SOXL261127{r[3][0]}36000", "ts": T0 + pd.Timedelta(seconds=r[0]),
        "price": 1.0, "volume": r[2], "direction": r[1],
        "expiry": "2026-11-27", "strike": 36.0, "opt_type": r[3],
    } for r in rows])


def _greeks(rows):
    """rows: (offset_s, delta, opt_type, upd_offset_s)."""
    return pd.DataFrame([{
        "code": f"US.SOXL261127{r[2][0]}36000",
        "ts": T0 + pd.Timedelta(seconds=r[0]),
        "delta": r[1], "spot": 40.0,
        "upd": T0 + pd.Timedelta(seconds=r[3]),
    } for r in rows])


class TestTheSignNeedsNoCallPutBranch:
    """The whole formula rests on this: delta already carries the type.

    If it did not, every put in the series would point the wrong way -- and
    a wrong-signed flow series still correlates with price (a review
    measured +0.430 for a deliberately broken rule against +0.532 for the
    correct one, indistinguishable at N=30). Nothing downstream can catch
    this, so it is caught here.
    """

    @pytest.mark.parametrize("direction,delta,opt_type,bullish", [
        ("BUY", 0.55, "CALL", True),
        ("BUY", -0.45, "PUT", False),
        ("SELL", 0.55, "CALL", False),
        ("SELL", -0.45, "PUT", True),
    ])
    def test_all_four_corners(self, direction, delta, opt_type, bullish):
        f, _ = attach_delta(_ticks([(10, direction, 3, opt_type)]),
                            _greeks([(0, delta, opt_type, 0)]))
        assert len(f) == 1
        got = f["signed_delta"].iloc[0]
        # bool(), not `is`: the comparison yields np.bool_, and
        # `np.True_ is True` is False -- the assertion would fail on a
        # perfectly correct value.
        assert bool(got > 0) is bullish, f"{direction} {opt_type} -> {got}"
        assert abs(got) == pytest.approx(abs(delta) * 3 * CONTRACT_MULTIPLIER)


class TestNothingIsPricedOffTheFuture:
    """A greeks sample taken after a print knows where the stock went.

    This is the same one-minute lookahead that invalidated the drill light
    on 2026-11-14, which is why it is pinned rather than trusted.
    """

    def test_a_later_sample_is_not_used(self):
        # Only sample is 60 s AFTER the print.
        f, rep = attach_delta(_ticks([(0, "BUY", 1, "CALL")]),
                              _greeks([(60, 0.5, "CALL", 60)]))
        assert len(f) == 0, "a future greeks row was used"
        assert rep["no_greeks"] == 1

    def test_an_earlier_sample_is_used(self):
        f, _ = attach_delta(_ticks([(60, "BUY", 1, "CALL")]),
                            _greeks([(0, 0.5, "CALL", 0)]))
        assert len(f) == 1 and f["greeks_age_s"].iloc[0] == 60

    def test_a_sample_older_than_the_tolerance_is_refused(self):
        f, rep = attach_delta(
            _ticks([(MAX_GREEKS_AGE_S + 60, "BUY", 1, "CALL")]),
            _greeks([(0, 0.5, "CALL", 0)]))
        assert len(f) == 0 and rep["no_greeks"] == 1

    def test_a_quote_stamped_after_the_print_is_dropped(self):
        """The feed's clock can run ahead of ours -- 260 rows did on
        2026-11-23, by up to 218 s. Our sample time looks safely earlier
        while the quote inside it postdates the trade."""
        f, rep = attach_delta(
            _ticks([(30, "BUY", 1, "CALL")]),
            _greeks([(0, 0.5, "CALL", 90)]))      # sampled before, stamped after
        assert len(f) == 0, "a quote newer than the print was used"
        assert rep["quote_ahead"] == 1


class TestAnImpossibleDeltaIsDroppedNotSigned:

    def test_a_call_with_negative_delta_is_refused(self):
        f, rep = attach_delta(_ticks([(10, "BUY", 1, "CALL")]),
                              _greeks([(0, -0.5, "CALL", 0)]))
        assert len(f) == 0 and rep["delta_sign_violation"] == 1

    def test_a_put_with_positive_delta_is_refused(self):
        f, rep = attach_delta(_ticks([(10, "BUY", 1, "PUT")]),
                              _greeks([(0, 0.5, "PUT", 0)]))
        assert len(f) == 0 and rep["delta_sign_violation"] == 1


class TestTheDroppedBucketsAreCountedApart:
    """NEUTRAL and 'N/A' are different losses and must not merge again."""

    def test_neutral_and_unknown_are_separate(self):
        f, rep = attach_delta(
            _ticks([(10, "BUY", 1, "CALL"), (11, "NEUTRAL", 1, "CALL"),
                    (12, "N/A", 1, "CALL"), (13, "N/A", 1, "CALL")]),
            _greeks([(0, 0.5, "CALL", 0)]))
        assert rep["neutral"] == 1
        assert rep["unknown"] == 2
        assert rep["used"] == 1

    def test_a_stale_quote_is_counted_but_still_used(self):
        """Counted, not dropped: on 2026-11-23 dropping them moved the
        session net by 11%, which is the caller's call, not this module's."""
        f, rep = attach_delta(
            _ticks([(MAX_QUOTE_AGE_S + 120, "BUY", 1, "CALL")]),
            _greeks([(0, 0.5, "CALL", 0)]))
        assert rep["stale_quote"] == 1
        assert rep["used"] == 1, "a stale quote must be used, only flagged"


class TestTheBarAggregation:

    def _flow(self):
        return attach_delta(
            _ticks([(0, "BUY", 2, "CALL"), (30, "SELL", 1, "CALL"),
                    (90, "BUY", 4, "CALL"), (150, "SELL", 7, "CALL")]),
            _greeks([(-1, 0.5, "CALL", -1)]))[0]

    def test_buy_plus_sell_is_net(self):
        b = by_bar(self._flow(), 1)
        assert (b["buy"] + b["sell"] - b["net"]).abs().max() < 1e-9
        assert (b["buy"] >= 0).all() and (b["sell"] <= 0).all()

    def test_cum_is_the_running_total(self):
        b = by_bar(self._flow(), 1)
        assert b["cum"].tolist() == pytest.approx(b["net"].cumsum().tolist())

    def test_row_order_does_not_change_the_answer(self):
        """buy/sell used to come from a second groupby whose row order was
        assumed to match the first's -- both sort the same way today, so the
        bug would only appear when a pandas default changed, as buy/sell
        attached to the wrong bars. No total would have moved."""
        f = self._flow()
        a = by_bar(f, 1)
        b = by_bar(f.sample(frac=1, random_state=7), 1)
        for c in ("net", "buy", "sell", "cum", "prints", "contracts"):
            assert a[c].to_numpy() == pytest.approx(b[c].to_numpy()), c

    def test_a_frame_with_no_deltas_is_refused_not_drawn_as_zero(self):
        """attach_delta's no-greeks path yields signed_delta = NA. Summed,
        NA becomes 0, so this frame would render as a full session of prints
        with exactly zero flow -- a convincing chart of nothing."""
        f, _ = attach_delta(_ticks([(0, "BUY", 1, "CALL")]), pd.DataFrame())
        assert by_bar(f, 1).empty


class TestTheTradingDayBoundary:
    """20:00 ET, matching _trading_day_of_bucket in trade_viewer_qt.

    Two indicators read side by side cannot be compared at all in the hours
    between two different boundaries, so this one is copied deliberately.
    """

    @pytest.mark.parametrize("stamp,expected", [
        ("2026-11-23 09:30", date(2026, 11, 23)),
        ("2026-11-23 19:59", date(2026, 11, 23)),
        ("2026-11-23 20:00", date(2026, 11, 24)),
        ("2026-11-23 23:59", date(2026, 11, 24)),
        ("2026-11-24 00:01", date(2026, 11, 24)),
    ])
    def test_the_boundary_is_2000_not_midnight(self, stamp, expected):
        assert trading_day_of(pd.Timestamp(stamp)) == expected

    def test_cum_restarts_at_the_boundary(self):
        f = attach_delta(
            _ticks([(0, "BUY", 5, "CALL")]), _greeks([(0, 0.5, "CALL", 0)]))[0]
        late = f.copy()
        late["ts"] = pd.Timestamp("2026-11-23 19:59:00")
        after = f.copy()
        after["ts"] = pd.Timestamp("2026-11-23 20:01:00")
        b = by_bar(pd.concat([late, after], ignore_index=True), 1)
        assert len(b) == 2
        assert b["cum"].iloc[0] == pytest.approx(b["cum"].iloc[1]), (
            "the second bar accumulated onto the first across the boundary")
