"""Picking contracts must not spend one chain call per expiry.

The budget is ten get_option_chain calls per thirty seconds and it is
ACCOUNT-WIDE, so it is shared by every underlying collected at once. One
call per expiry meant six for SOXL alone; add a second symbol and the pair
reached fourteen, four of them refused.

A refusal is the dangerous part, not the cost. pick_contracts logs a
warning, carries on, and the surviving candidates re-rank to fill all 200
slots -- so `subscribed 200 of 200`, the reconcile ratio and the tick
counters all stay green while the furthest-dated expiries are simply
absent for the whole session. Contracts are picked once per run; there is
no second chance.

This file is the first coverage pick_contracts has ever had.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from analysis.option_tick_collector import (
    _CHAIN_SPAN_DAYS, _chunk_by_span, pick_contracts)

SOXL_WEEKLIES = ["2026-11-27", "2026-12-04", "2026-12-11", "2026-12-18",
                 "2026-12-24", "2026-12-31", "2027-01-15", "2027-02-19"]
QQQ_DAILIES = ["2026-11-24", "2026-11-25", "2026-11-27", "2026-11-30",
               "2026-12-01", "2026-12-02", "2026-12-03", "2026-12-04"]


class TestChunkingIsBySpanNotByCount:
    """30 days is the documented limit on start..end. The number of
    expiries inside it is not fixed: eight SOXL weeklies reach 84 days
    apart, eight QQQ dailies about ten."""

    @pytest.mark.parametrize("expiries", [SOXL_WEEKLIES, QQQ_DAILIES])
    def test_no_chunk_exceeds_the_documented_span(self, expiries):
        for chunk in _chunk_by_span(expiries, _CHAIN_SPAN_DAYS):
            span = (date.fromisoformat(chunk[-1])
                    - date.fromisoformat(chunk[0])).days
            assert span <= _CHAIN_SPAN_DAYS, (chunk, span)

    @pytest.mark.parametrize("expiries", [SOXL_WEEKLIES, QQQ_DAILIES])
    def test_every_expiry_appears_exactly_once(self, expiries):
        flat = [e for c in _chunk_by_span(expiries, _CHAIN_SPAN_DAYS) for e in c]
        assert flat == expiries

    def test_the_two_shapes_really_do_differ(self):
        """Without this the span test passes on data where count-slicing
        would have worked too, i.e. against no change at all."""
        soxl = _chunk_by_span(SOXL_WEEKLIES, _CHAIN_SPAN_DAYS)
        qqq = _chunk_by_span(QQQ_DAILIES, _CHAIN_SPAN_DAYS)
        assert len(qqq) == 1, qqq
        assert len(soxl) > 1, soxl
        assert len(soxl[0]) != len(qqq[0]), "same grouping for both shapes"

    def test_a_single_expiry_is_one_chunk(self):
        assert _chunk_by_span(["2026-11-27"], _CHAIN_SPAN_DAYS) == [["2026-11-27"]]

    def test_no_expiries_is_no_calls(self):
        assert _chunk_by_span([], _CHAIN_SPAN_DAYS) == []


class _Ctx:
    """Records every chain range asked for, and answers from one table."""

    def __init__(self, expiries, strikes=(30.0, 40.0, 50.0), fail=()):
        self.calls: list[tuple[str, str]] = []
        self._expiries = list(expiries)
        self._strikes = list(strikes)
        self._fail = set(fail)

    # Answers two different questions: the underlying's spot (one code) and
    # the per-contract greeks that _score_contracts ranks with. Spot has to
    # sit inside the strike grid or the +/-pct window excludes everything
    # and the chain loop is never reached -- which looked like "zero calls"
    # the first time this stub was written.
    def get_market_snapshot(self, codes):
        rows = [{"code": c, "option_delta": 0.5, "option_open_interest": 100,
                 "last_price": 40.0, "update_time": "2026-11-23 09:30:00"}
                for c in codes]
        return 0, pd.DataFrame(rows)

    def get_option_chain(self, code, start=None, end=None, **kw):
        self.calls.append((start, end))
        if (start, end) in self._fail:
            return -1, "rate limited"
        rows = []
        for e in self._expiries:
            if not (start <= e <= end):
                continue
            for k in self._strikes:
                for t in ("CALL", "PUT"):
                    rows.append({
                        "code": f"{code}{e.replace('-', '')[2:]}"
                                f"{t[0]}{int(k * 1000)}",
                        "strike_price": k, "option_type": t,
                        # The field the expiry now comes from.
                        "strike_time": e,
                    })
        return 0, pd.DataFrame(rows)


def _pick(ctx, expiries, monkeypatch, spot=40.0):
    import analysis.option_tick_collector as m
    import strategy.option.gex as gex
    # Patched at the SOURCE module: pick_contracts imports it locally
    # (), so there is
    # no attribute on the collector module to replace.
    monkeypatch.setattr(gex, "get_near_expiry_dates", lambda *a, **k: expiries)
    return m.pick_contracts(ctx, "US.SOXL", 80.0)


class TestTheCallBudget:

    def test_a_weekly_chain_costs_far_fewer_calls_than_expiries(
            self, monkeypatch):
        """The regression that matters: back to one call per expiry and
        two collectors starting together blow the ten-per-thirty budget."""
        ctx = _Ctx(SOXL_WEEKLIES)
        try:
            _pick(ctx, SOXL_WEEKLIES, monkeypatch)
        except Exception:
            pass                      # the budget is the subject, not the result
        # A lower bound as well: the try/except above swallows anything
        # raised before the loop, and "fewer calls than expiries" is also
        # satisfied by zero. Measured -- an exception before the first call
        # left this test green.
        assert 1 <= len(ctx.calls) <= 3, ctx.calls
        assert len(ctx.calls) < len(SOXL_WEEKLIES), (
            f"{len(ctx.calls)} calls for {len(SOXL_WEEKLIES)} expiries -- "
            f"this is the per-expiry pattern the ranges replaced")

    def test_a_daily_chain_costs_one_call(self, monkeypatch):
        ctx = _Ctx(QQQ_DAILIES)
        try:
            _pick(ctx, QQQ_DAILIES, monkeypatch)
        except Exception:
            pass
        assert len(ctx.calls) == 1, ctx.calls   # exact: 1 is also the floor


class TestTheExpiryComesFromTheRow:
    """A range answers for every expiry inside it, so the row's own
    strike_time is the only thing that says which one a contract belongs
    to -- the loop variable no longer does."""

    def test_contracts_carry_their_own_expiry(self, monkeypatch):
        ctx = _Ctx(SOXL_WEEKLIES)
        contracts, info = _pick(ctx, SOXL_WEEKLIES, monkeypatch)
        got = {c["expiry"] for c in contracts}
        # EVERY requested expiry, not merely "more than one". Stamping each
        # contract with its chunk's start date also yields several distinct
        # values -- one per chunk -- so `len(got) > 1` passed against
        # exactly the mistake this test names.
        assert got == set(SOXL_WEEKLIES), sorted(set(SOXL_WEEKLIES) - got)

    def test_an_expiry_the_range_covers_but_we_did_not_ask_for_is_dropped(
            self, monkeypatch):
        """A 28-day range sweeps up expiries between the ones requested.
        Keeping them would widen the selection past _MAX_CHAINS without
        anyone asking."""
        asked = ["2026-11-27", "2026-12-18"]
        ctx = _Ctx(asked + ["2026-12-04", "2026-12-11"])
        contracts, _ = _pick(ctx, asked, monkeypatch)
        assert {c["expiry"] for c in contracts} == set(asked)


class TestAShortChainIsNamedNotSwallowed:

    def test_a_refused_range_is_reported_with_the_missing_expiries(
            self, monkeypatch, caplog):
        """The failure mode this change exists to expose. Before, a refusal
        left 200 contracts subscribed and every indicator green."""
        chunks = _chunk_by_span(SOXL_WEEKLIES, _CHAIN_SPAN_DAYS)
        assert len(chunks) > 1, "need a multi-chunk fixture to refuse one"
        doomed = chunks[-1]
        ctx = _Ctx(SOXL_WEEKLIES, fail={(doomed[0], doomed[-1])})

        with caplog.at_level("WARNING"):
            contracts, _ = _pick(ctx, SOXL_WEEKLIES, monkeypatch)

        # Only the MISSING record. caplog.text also holds the
        # "chain fetch failed for X..Y" line, which names the same dates --
        # so asserting against the whole text passed even with the names
        # stripped out of MISSING entirely.
        missing = [r.getMessage() for r in caplog.records
                   if "MISSING" in r.getMessage()]
        assert len(missing) == 1, caplog.text
        for e in doomed:
            assert e in missing[0], (e, missing[0])
        # And the contracts really are absent -- the warning is not cosmetic.
        assert not ({c["expiry"] for c in contracts} & set(doomed))


class TestAMalformedLadderEntryCostsOnlyItself:
    """The SDK fills a missing strikeTime with a single space.

    `strikeTime` is optional in the proto while the distance field beside
    it is required, so quote_query.py:3576 substitutes ' ' -- and
    get_near_expiry_dates appends it unchecked on its `dist` branch. The
    per-expiry loop this file replaced asked OpenD for `start=' '`, got
    nothing, logged a warning and carried on one expiry short.

    date.fromisoformat does not carry on. Measured before the guard: the
    new code raised ValueError after zero chain calls and produced zero
    contracts, where the old one made seven calls and 36 rows. Contracts
    are picked once per run, so that is the difference between a session
    missing one expiry and no session at all.
    """

    def test_chunking_skips_it(self, caplog):
        with caplog.at_level("WARNING"):
            chunks = _chunk_by_span(
                ["2026-11-27", " ", "2026-12-04"], _CHAIN_SPAN_DAYS)
        assert chunks == [["2026-11-27", "2026-12-04"]], chunks
        assert "malformed" in caplog.text

    def test_a_bad_entry_does_not_lose_the_session(self, monkeypatch, caplog):
        ladder = ["2026-11-27", " ", "2026-12-04"]
        ctx = _Ctx(["2026-11-27", "2026-12-04"])
        with caplog.at_level("WARNING"):
            contracts, _ = _pick(ctx, ladder, monkeypatch)
        assert contracts, "a malformed ladder entry emptied the selection"
        assert {c["expiry"] for c in contracts} == {"2026-11-27", "2026-12-04"}


class TestMissingMeasuresTheChainNotTheBand:
    """`MISSING` is about a range that came back short. An expiry whose
    strikes all fall outside +/-pct% is a different statement, and counting
    it as a missing chain points the reader at the wrong subsystem."""

    def test_an_expiry_with_no_in_band_strikes_is_not_called_missing(
            self, monkeypatch, caplog):
        asked = ["2026-11-27", "2026-12-04"]
        ctx = _Ctx(asked, strikes=(40.0,))
        # One expiry answers only far out of the band; the chain itself is
        # fine for both.
        real_chain = ctx.get_option_chain

        def chain(code, start=None, end=None, **kw):
            ret, df = real_chain(code, start=start, end=end, **kw)
            if ret == 0 and not df.empty:
                df.loc[df["strike_time"] == "2026-12-04", "strike_price"] = 9999.0
            return ret, df

        ctx.get_option_chain = chain
        with caplog.at_level("WARNING"):
            _pick(ctx, asked, monkeypatch)
        missing = [r.getMessage() for r in caplog.records
                   if "MISSING" in r.getMessage()]
        assert not missing, missing
