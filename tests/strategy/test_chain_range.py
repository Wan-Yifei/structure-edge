"""The span invariant belongs to the module that owns it.

`chunk_by_span` moved to strategy/option/chain_range.py because both the
collector's pick_contracts and the viewer's wall fetch need it and neither
could take the other's imports. Its core invariant -- no chunk wider than
the documented range limit -- was then reachable only through the
collector's private alias, so a shared module's contract was being tested
through one of its two callers.

Why the invariant matters: get_option_chain refuses a range past about 30
days, and a refused call costs EVERY expiry inside it. On a single-chunk
ladder, which is what QQQ at a 7-day DTE produces, that is the whole fetch.
"""
from datetime import date

import pytest

from strategy.option.chain_range import CHAIN_SPAN_DAYS, chunk_by_span

# Real ladders, read off 2026-11-25's own logs rather than invented: eight
# QQQ dailies span about twelve days, seven SOXL weeklies about thirty-five.
QQQ = ["2026-11-25", "2026-11-27", "2026-11-30", "2026-12-01",
       "2026-12-02", "2026-12-03", "2026-12-04", "2026-12-07"]
SOXL = ["2026-11-27", "2026-12-04", "2026-12-11", "2026-12-18",
        "2026-12-24", "2026-12-31", "2027-01-08"]


class TestNoChunkOutgrowsTheRangeLimit:

    @pytest.mark.parametrize("ladder", [QQQ, SOXL])
    def test_every_chunk_fits(self, ladder):
        for chunk in chunk_by_span(ladder, CHAIN_SPAN_DAYS):
            span = (date.fromisoformat(chunk[-1])
                    - date.fromisoformat(chunk[0])).days
            assert span <= CHAIN_SPAN_DAYS, (chunk, span)

    def test_the_span_is_measured_from_the_chunks_first_date(self):
        """Measuring from the last element instead lets a chunk creep: each
        new date is only compared to its predecessor, so a run of small steps
        adds up past the limit. A 60-day ladder in weekly steps is the
        shape."""
        weekly = ["2026-11-27", "2026-12-04", "2026-12-11", "2026-12-18",
                  "2026-12-24", "2026-12-31", "2027-01-08", "2027-01-15",
                  "2027-01-22", "2027-01-29"]
        for chunk in chunk_by_span(weekly, CHAIN_SPAN_DAYS):
            span = (date.fromisoformat(chunk[-1])
                    - date.fromisoformat(chunk[0])).days
            assert span <= CHAIN_SPAN_DAYS, (chunk, span)

    @pytest.mark.parametrize("ladder", [QQQ, SOXL])
    def test_nothing_is_lost_or_duplicated(self, ladder):
        flat = [e for chunk in chunk_by_span(ladder, CHAIN_SPAN_DAYS)
                for e in chunk]
        assert flat == sorted(ladder)


class TestItKeepsItsOwnPrecondition:

    def test_an_unsorted_ladder_still_produces_valid_ranges(self):
        """Out of order, the chunks used to come out start > end -- measured,
        a reversed ladder gave start='2027-01-15', end='2026-12-04', which
        asks OpenD for an empty range and loses everything in it. Both
        callers happen to sort today; the function keeps the precondition
        itself because it is one call to do so."""
        for chunk in chunk_by_span(list(reversed(SOXL)), CHAIN_SPAN_DAYS):
            assert chunk[0] <= chunk[-1], chunk

    def test_it_matches_the_sorted_input(self):
        assert (chunk_by_span(list(reversed(SOXL)), CHAIN_SPAN_DAYS)
                == chunk_by_span(SOXL, CHAIN_SPAN_DAYS))


class TestAMalformedExpiryIsSkippedNotFatal:
    """The SDK's GetOptionExpirationDate substitutes a single space for a
    missing strikeTime, and get_near_expiry_dates appends it unchecked. The
    per-expiry loop this replaced asked OpenD for `start=' '`, got nothing
    and carried on one expiry short. Raising instead would cost a whole
    session, because contracts are picked once per run."""

    def test_the_junk_is_dropped_and_the_rest_survives(self):
        out = chunk_by_span([" ", "2026-11-27", "", "2026-12-04"],
                            CHAIN_SPAN_DAYS)
        assert [e for chunk in out for e in chunk] == ["2026-11-27",
                                                       "2026-12-04"]

    def test_a_ladder_of_nothing_but_junk_is_empty_not_an_error(self):
        assert chunk_by_span([" ", "", "not-a-date"], CHAIN_SPAN_DAYS) == []


class TestTheRealLaddersCostWhatWeThink:
    """The whole point of the change, in calls."""

    def test_a_qqq_week_is_one_call(self):
        week = [e for e in QQQ if e <= "2026-12-02"]      # DTE=7 on 11-25
        assert len(week) == 5
        assert len(chunk_by_span(week, CHAIN_SPAN_DAYS)) == 1

    def test_the_soxl_ladder_needs_two(self):
        assert len(chunk_by_span(SOXL, CHAIN_SPAN_DAYS)) == 2
