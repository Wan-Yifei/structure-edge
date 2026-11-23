"""Time to expiry comes from the exchange clock; the chart is dated by its data.

Two separate ways the old code used "today" where it should not have:

  * `_t_years` counted whole LOCAL CALENDAR DAYS. Measured on a real SOXL
    chain, dZG/dT is about -$1.80 per day, so the Zero Gamma line sat frozen
    through the session and jumped $1.80 at local midnight with the market
    shut. Against moomoo's own chart for the same chain it read 40.80 vs
    41.32, and solving for T put that entirely down to 7.1 hours of
    over-counting.
  * `as_of` was `date.today()`. A chart built on a Sunday from Friday's
    greeks titled itself "Data: 2026-11-22", and a saved PNG carries that
    for as long as it exists.
"""
from datetime import date, datetime, time, timedelta

import pandas as pd
import pytest

from strategy.option.gex import (
    _ET, _MIN_T_YEARS, _data_as_of, build_gex_stats, compute_gex,
)


def _chain(expiry: str, strikes=(90.0, 95.0, 100.0, 105.0, 110.0),
           update_times=None):
    """Puts stacked low, calls stacked high, so signed GEX really crosses zero.

    A symmetric chain cancels exactly and zero_gamma returns None -- the
    first version of this fixture did that and the tests failed for a reason
    that had nothing to do with the clock.
    """
    rows = []
    for i, k in enumerate(strikes):
        for t in ("CALL", "PUT"):
            oi = (900 if (t == "CALL") == (k >= 100.0) else 150)
            rows.append({
                "code": f"T{i}{t}", "option_type": t,
                "option_strike_price": k, "option_gamma": 0.04,
                "option_open_interest": oi, "option_delta": 0.5,
                "option_implied_volatility": 40.0, "strike_time": expiry,
                "volume": 10.0,
                "update_time": (update_times[i % len(update_times)]
                                if update_times else "2026-11-20 16:00:00"),
            })
    return pd.DataFrame(rows)


def _zero_gamma_for(expiry: str):
    df, by_strike = compute_gex(_chain(expiry), 100.0)
    return build_gex_stats(df, by_strike, 100.0, 1e6, [expiry])["zero_gamma"]


class TestTheClockIsTheExchangeClock:

    def test_zero_gamma_moves_within_a_single_day(self, monkeypatch):
        """The load-bearing one.

        Under whole-day rounding, two readings taken hours apart on the SAME
        calendar day were bit-identical, and then the line jumped $1.80 at
        local midnight. Freezing the clock at two moments inside one day and
        demanding a difference is the only thing that pins the fix: comparing
        two different EXPIRIES would pass against the old code too.
        """
        import strategy.option.gex as gex
        expiry = (date.today() + timedelta(days=5)).isoformat()

        def _at(hour: int):
            real = gex.datetime

            class _Frozen(real):
                @classmethod
                def now(cls, tz=None):
                    return real.combine(date.today(), time(hour, 0), tzinfo=tz)

            monkeypatch.setattr(gex, "datetime", _Frozen)
            try:
                return _zero_gamma_for(expiry)
            finally:
                monkeypatch.setattr(gex, "datetime", real)

        morning, evening = _at(9), _at(21)
        assert morning is not None and evening is not None, (morning, evening)
        assert morning != evening, (
            f"same calendar day, 12 hours apart, identical answer {morning} "
            f"-- time to expiry is still being rounded to whole days")
        # Later in the day = less time left = the flip point moves up.
        assert evening > morning, (morning, evening)

    def test_time_runs_out_toward_the_expiry_close(self):
        """Nearer expiries carry strictly less time, with no plateau.

        A plateau is what whole-day rounding looked like. Anchored on a
        stretch with no half session in it -- an earlier version ran across
        2026-11-27 and its one short day broke the uniform-gap assertion,
        correctly.
        """
        from strategy.option.gex import _SECONDS_PER_YEAR, _market_close_et
        base = date(2026, 12, 7)           # Mon; that week is all full days
        now = datetime.combine(base, time(9, 0), tzinfo=_ET)
        seen = []
        for d in range(1, 6):
            day = base + timedelta(days=d)
            close = datetime.combine(day, _market_close_et(day), tzinfo=_ET)
            seen.append((close - now).total_seconds() / _SECONDS_PER_YEAR)

        assert seen == sorted(seen), seen
        gaps = [b - a for a, b in zip(seen, seen[1:])]
        assert all(abs(g - 1 / 365) < 1e-12 for g in gaps), gaps

    def test_a_half_session_is_three_hours_short(self):
        """The same sweep across the Thanksgiving-Friday expiry: one gap
        must come up three hours shy of a whole day."""
        from strategy.option.gex import _SECONDS_PER_YEAR, _market_close_et
        base = date(2026, 11, 24)
        now = datetime.combine(base, time(9, 0), tzinfo=_ET)
        seen = []
        for d in range(1, 5):              # 11-25, 11-26, 11-27, 11-28
            day = base + timedelta(days=d)
            close = datetime.combine(day, _market_close_et(day), tzinfo=_ET)
            seen.append((close - now).total_seconds() / _SECONDS_PER_YEAR)

        gaps = [b - a for a, b in zip(seen, seen[1:])]
        short = [g for g in gaps if g < 1 / 365 - 1e-12]
        assert len(short) == 1, gaps
        hours_lost = (1 / 365 - short[0]) * 365 * 24
        assert abs(hours_lost - 3) < 1e-6, hours_lost

    def test_an_expired_chain_refuses_rather_than_answering(self):
        """An expired contract gets NO number, not a floored one.

        The server does serve expired chains -- distance -2 was seen live --
        and `zero_gamma()` is a public function while the expiry filter
        lives in a different one, so this is reachable.

        This test used to assert the opposite: that two differently-expired
        chains clamp to the SAME value. That was deliberate and it was
        wrong. Clamping answers an expired chain with a confident,
        plausible-looking price level computed from no remaining time --
        exactly the class of quietly-wrong number the rest of this work is
        about. Refusing is the only honest output.
        """
        for days in (3, 30):
            past = (date.today() - timedelta(days=days)).isoformat()
            assert _zero_gamma_for(past) is None, (
                f"expired {days} days ago still produced a level")

    def test_the_floor_still_protects_a_live_contract(self, monkeypatch):
        """The floor is for time running out, not for time already gone.

        Half an hour before the close must still yield a number: written as
        `years < _MIN_T_YEARS` instead of `years < 0`, the NaN guard would
        swallow this too and the last hour of every 0DTE session would go
        blank. The clock is frozen because "today" at the moment the suite
        happens to run is usually already past 16:00 ET.
        """
        import strategy.option.gex as gex
        expiry = date(2026, 12, 11)        # an ordinary Friday, 16:00 close
        real = gex.datetime

        class _Frozen(real):
            @classmethod
            def now(cls, tz=None):
                return real.combine(expiry, time(15, 30), tzinfo=tz)

        monkeypatch.setattr(gex, "datetime", _Frozen)
        try:
            assert _zero_gamma_for(expiry.isoformat()) is not None, (
                "30 minutes before the close produced nothing")
        finally:
            monkeypatch.setattr(gex, "datetime", real)

        assert 0 < _MIN_T_YEARS < 1 / 365      # a short floor, not a day


class TestHalfSessions:
    """Two of the next nine expiries close at 13:00, not 16:00.

    Worth about 0.22 on Zero Gamma each time one is the near expiry. Found
    by chasing a 0.18 gap against moomoo's own chart for 2026-11-27, which
    is the day after Thanksgiving.
    """

    @pytest.mark.parametrize("day,expected_hour,why", [
        ("2026-11-27", 13, "day after Thanksgiving"),
        ("2026-12-24", 13, "Christmas Eve"),
        ("2025-11-28", 13, "day after Thanksgiving, a different year"),
        ("2027-11-26", 13, "ditto, a third year"),
        ("2026-11-23", 16, "ordinary Monday"),
        ("2026-12-04", 16, "ordinary expiry Friday"),
        ("2026-12-31", 16, "New Year's Eve is a FULL session"),
        ("2026-12-23", 16, "the day before Christmas Eve is not"),
    ])
    def test_the_close_is_right(self, day, expected_hour, why):
        from strategy.option.gex import _market_close_et
        got = _market_close_et(date.fromisoformat(day))
        assert got.hour == expected_hour, f"{day} ({why}) -> {got}"

    def test_thanksgiving_is_the_fourth_thursday(self):
        from strategy.option.gex import _thanksgiving
        for year, expected in ((2025, "2025-11-27"), (2026, "2026-11-26"),
                               (2027, "2027-11-25"), (2028, "2028-11-23")):
            got = _thanksgiving(year)
            assert got.isoformat() == expected, (year, got)
            assert got.weekday() == 3, got        # Thursday

    def test_a_half_day_leaves_less_time_than_a_full_one(self):
        """The whole point: three fewer hours changes the answer."""
        from strategy.option.gex import _CLOSE_FULL_ET, _CLOSE_HALF_ET
        assert _CLOSE_HALF_ET < _CLOSE_FULL_ET
        gap = (_CLOSE_FULL_ET.hour - _CLOSE_HALF_ET.hour)
        assert gap == 3, gap


class TestTheChartIsDatedByItsData:

    def test_as_of_reports_the_newest_quote_not_today(self):
        df = _chain("2026-11-27", update_times=["2026-11-20 09:42:10",
                                                "2026-11-20 16:14:53",
                                                "2026-11-17 09:30:00"])
        out = _data_as_of(df)
        assert out.startswith("2026-11-20 16:14"), out
        assert date.today().isoformat() not in out or out.startswith("2026-11-20")

    def test_the_newest_wins_not_the_first_row(self):
        """Per-contract stamps differ by hours; a quiet strike must not date
        the chart."""
        df = _chain("2026-11-27", update_times=["2026-11-20 16:14:53",
                                                "2026-11-20 09:42:10",
                                                "2026-11-20 10:39:00"])
        assert _data_as_of(df).startswith("2026-11-20 16:14")

    def test_a_missing_column_says_it_fell_back_to_the_clock(self):
        df = _chain("2026-11-27").drop(columns=["update_time"])
        out = _data_as_of(df)
        assert "(clock)" in out, out
        assert date.today().isoformat() in out

    def test_stats_carry_the_data_date(self):
        df = _chain("2026-11-27", update_times=["2026-11-20 16:14:53"])
        d2, bs = compute_gex(df, 100.0)
        stats = build_gex_stats(d2, bs, 100.0, 1e6, ["2026-11-27"])
        assert stats["as_of"].startswith("2026-11-20 16:14"), stats["as_of"]
