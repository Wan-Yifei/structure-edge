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


class TestTheBarKeysMatchTheChartsOwn:
    """by_bar's keys have to be the keys the chart produces, or nothing
    lands.

    The viewer maps option bars onto chart bars by looking each bar start up
    in a dict. A key the chart never produces is not an error -- it is a
    miss, and the row goes blank with no message. `bar_start_keys()` in
    trade_viewer_qt uses `candle_start()` and carries an explicit warning
    that `Series.dt.floor()` agrees with it only while 60 is divisible by
    the candle size; this module used floor() until 2026-11-23.
    """

    @pytest.mark.parametrize("minutes", [1, 5, 15, 30, 60])
    def test_sizes_that_divide_an_hour_agree_either_way(self, minutes):
        f = attach_delta(_ticks([(0, "BUY", 1, "CALL"), (1847, "BUY", 1, "CALL")]),
                         _greeks([(-1, 0.5, "CALL", -1)]))[0]
        b = by_bar(f, minutes)
        assert (b["bar"] == b["bar"].dt.floor(f"{minutes}min")).all()

    def test_a_size_that_does_not_divide_an_hour_follows_candle_start(self):
        """45 minutes is the case the viewer's warning names. Epoch-aligned
        floor() and within-the-hour candle_start() part company here, and
        only one of them is what the chart draws."""
        from core.time_utils import candle_start
        # The greeks sample has to sit inside MAX_GREEKS_AGE_S of the print,
        # or this module's own staleness guard drops it and the frame is
        # empty -- which fails for a reason that has nothing to do with
        # bar alignment.
        f = attach_delta(_ticks([(2820, "BUY", 1, "CALL")]),   # 10:47:00
                         _greeks([(2800, 0.5, "CALL", 2800)]))[0]
        b = by_bar(f, 45)
        got = b["bar"].iloc[0].to_pydatetime()
        assert got == candle_start(pd.Timestamp("2026-11-23 10:47").to_pydatetime(), 45)
        assert got != pd.Timestamp("2026-11-23 10:47").floor("45min"), (
            "this case is only worth pinning because the two disagree")


class TestTheKeysReachTheChart:
    """by_bar's keys must be lookup-compatible with the viewer's.

    The viewer maps option bars onto chart bars with `bk in by_start` -- a
    dict lookup. A key that does not hash equal is not an error, it is a
    miss, and the row goes blank with a title that still reads normally.

    The two sides do not even produce the same TYPE: bar_start_keys yields
    datetime.datetime and _bar_starts yields pandas.Timestamp. They hash
    equal today only because both land on whole seconds with no timezone,
    and nothing anywhere states that as a requirement -- which is why it is
    stated here.
    """

    @pytest.mark.parametrize("minutes", [1, 5, 15, 30, 60])
    def test_every_chart_bar_key_is_findable(self, minutes):
        """Build the lookup from by_bar's OUTPUT, then look the chart's
        keys up in it -- which is what the viewer does.

        An earlier version built the dict from the chart's own keys and
        then searched it with those same keys. Nothing this module produces
        took part, so it asserted that pandas hashes its Timestamps
        consistently and would have passed against any bucketing rule at
        all.
        """
        from analysis.trade_viewer_qt import bar_start_keys

        # time_key is the bar's END and these keys are its START -- that
        # offset is the other half of the join, and the test below spells
        # out what happens to anyone who joins on time_key directly.
        ends = pd.date_range("2026-11-23 09:31", periods=40,
                             freq=f"{minutes}min")
        chart_keys = bar_start_keys(pd.Series(ends), minutes)

        # One print per bar, each with its OWN greeks sample a second
        # earlier. A single shared sample cannot sit inside
        # MAX_GREEKS_AGE_S of forty bars, so most prints were dropped
        # before by_bar saw them: at 15 minutes and above exactly one bar
        # survived, and the assertion below held vacuously.
        tick_rows, greek_rows = [], []
        proto_t = _ticks([(0, "BUY", 1, "CALL")]).iloc[0]
        proto_g = _greeks([(0, 0.5, "CALL", 0)]).iloc[0]
        for k in chart_keys:
            start = pd.Timestamp(k)
            t = proto_t.copy()
            t["ts"] = start + pd.Timedelta(seconds=1)
            tick_rows.append(t)
            g = proto_g.copy()
            g["ts"] = start
            g["upd"] = start
            greek_rows.append(g)

        flow, report = attach_delta(pd.DataFrame(tick_rows),
                                    pd.DataFrame(greek_rows))
        assert report["used"] == len(chart_keys), (
            f"only {report['used']} of {len(chart_keys)} prints survived; "
            f"the fixture, not the join, is what is being measured")

        produced = {pd.Timestamp(b): True for b in by_bar(flow, minutes)["bar"]}
        missing = [k for k in chart_keys if k not in produced]
        assert not missing, (
            f"{len(missing)} of {len(chart_keys)} chart bars find nothing in "
            f"this module's keys at {minutes}m -- the row would be blank "
            f"with no error: {missing[:3]}")

    def test_a_naive_join_on_time_key_would_shift_every_bar_by_one(self):
        """Stated so the next person does not 'simplify' the lookup.

        time_key is the bar's END and these keys are bar STARTS, so for a
        1-minute chart the two sets overlap by all but one element --
        joining on time_key does NOT fail loudly, it silently hands each
        candle the NEXT bar's option flow.

        That is a one-bar lookahead, the same shape that invalidated the
        drill light on 2026-11-14: every number still looks plausible and
        every chart still draws. Nothing else in the repo would catch it.
        """
        from analysis.trade_viewer_qt import bar_start_keys

        ends = pd.date_range("2026-11-23 09:31", periods=10, freq="1min")
        starts = [pd.Timestamp(k) for k in bar_start_keys(pd.Series(ends), 1)]

        # Each start is exactly one bar before its own end -- which is why
        # a naive join lands on the neighbour rather than on nothing.
        assert all(e - s == pd.Timedelta(minutes=1)
                   for e, s in zip(ends, starts))
        overlap = set(ends) & set(starts)
        assert len(overlap) == 9, (
            f"{len(overlap)} of 10 keys collide; a time_key join would "
            f"silently shift the series, not fail")
