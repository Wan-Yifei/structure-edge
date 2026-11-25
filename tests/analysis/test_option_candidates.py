"""A selection that is not written down can never be graded.

The scores pick_contracts ranks by come from a chain snapshot taken at
09:15, and the chain is not re-fetchable. Two lines after the sort the
scores are popped off the dicts and nothing in the process ever sees them
again, so the window in which a selection can be recorded is that narrow --
and outside it, "was the ranking any good?" is permanently unanswerable.
That is not hypothetical: on 2026-11-24 the single biggest-volume contract
of the day was passed over, and establishing that afterwards took rebuilding
the candidate pool from a live snapshot taken 27 hours later.

The capture therefore has to sit BEFORE the pop, which is a thing a later
edit can silently undo -- `r.get("_score")` keeps working and quietly
returns None for every row. test_scores_survive_the_capture is the guard
for exactly that, and it is the reason this file exists.
"""
from __future__ import annotations

import pandas as pd
import pytest

import analysis.option_tick_collector as m
from feeds.option_tick_store import OptionTickStore

EXPIRY = "2026-11-27"
STRIKES = (30.0, 40.0, 50.0)
SESSION = "2026-11-25"

# delta x OI with both factors varying, so no two candidates tie and the
# rank order is fully determined: CALLs carry more delta, higher strikes
# more open interest.
#   50C 0.6x50=30   40C 24   50P 0.4x50=20   30C 18   40P 16   30P 12
EXPECTED_ORDER = ["C50000", "C40000", "P50000", "C30000", "P40000", "P30000"]


class _Ctx:
    """Spot, a one-expiry chain, and per-contract greeks that differ."""

    def get_market_snapshot(self, codes):
        rows = []
        for c in codes:
            # int(k * 1000) is five digits and the type letter sits just
            # before it: US.SOXL261127C30000. Slicing one character wide
            # made every contract look like the underlying, every score
            # fall back to _score_contracts' 0.0 default, and the ranking
            # assertions below compare an unranked list.
            if not c[-5:].isdigit():            # the underlying itself
                rows.append({"code": c, "last_price": 40.0,
                             "update_time": "2026-11-25 09:15:00"})
                continue
            strike = int(c[-5:]) / 1000.0
            rows.append({
                "code": c,
                "option_delta": 0.6 if c[-6] == "C" else 0.4,
                "option_open_interest": int(strike),
                "last_price": 40.0,
                "update_time": "2026-11-25 09:15:00"})
        return 0, pd.DataFrame(rows)

    def get_option_chain(self, code, start=None, end=None, **kw):
        rows = []
        for k in STRIKES:
            for t in ("CALL", "PUT"):
                rows.append({
                    "code": f"{code}{EXPIRY.replace('-', '')[2:]}"
                            f"{t[0]}{int(k * 1000)}",
                    "strike_price": k, "option_type": t,
                    "strike_time": EXPIRY})
        return 0, pd.DataFrame(rows)


@pytest.fixture
def picked(monkeypatch):
    import strategy.option.gex as gex
    # Patched at the source module: pick_contracts imports it locally, so
    # there is no attribute on the collector module to replace.
    monkeypatch.setattr(gex, "get_near_expiry_dates", lambda *a, **k: [EXPIRY])
    return m.pick_contracts(_Ctx(), "US.SOXL", 80.0, quota=2)


@pytest.fixture
def store(tmp_path):
    s = OptionTickStore(tmp_path / "opt.db")
    yield s
    s.close()


class TestTheWholeBandIsRecordedNotJustTheCut:

    def test_every_candidate_appears(self, picked):
        contracts, info = picked
        assert len(contracts) == 2, "the quota did not bite"
        assert len(info["scored"]) == 6, (
            f"{len(info['scored'])} candidates recorded of 6 -- the ones the "
            f"ranking passed over are the entire point")

    def test_picked_marks_exactly_the_cut(self, picked):
        contracts, info = picked
        taken = {c["code"] for c in contracts}
        marked = {c["code"] for c in info["scored"] if c["picked"]}
        assert marked == taken, (
            f"`picked` disagrees with what was subscribed: {marked ^ taken}")

    def test_the_rank_is_the_ranking(self, picked):
        _c, info = picked
        ranks = [c["rank"] for c in info["scored"]]
        assert ranks == list(range(1, 7)), ranks
        suffixes = [c["code"][-6:] for c in info["scored"]]
        assert suffixes == EXPECTED_ORDER, (
            f"not ordered by score: {suffixes}")


class TestTheScoresSurviveTheCapture:
    """The one failure this file was written for.

    `r.get("_score")` on a dict the pop already ran over returns None
    without raising, so moving the capture a few lines down leaves every
    other assertion here passing and the recorded scores empty.
    """

    def test_scores_survive_the_capture(self, picked):
        _c, info = picked
        scores = [c["score"] for c in info["scored"]]
        assert all(s is not None for s in scores), (
            f"{sum(s is None for s in scores)} of {len(scores)} candidates "
            f"recorded with no score -- the capture has moved after the pop "
            f"that strips _score, and the ranking is now unrecoverable")
        # NOT vacuous: _score_contracts defaults an unscored contract to
        # 0.0, so a stub whose snapshot answers nothing produces six zeros
        # that are all "not None" and trivially sorted. This assertion is
        # what caught that in this file's own fixture.
        assert len(set(scores)) > 1, (
            f"every candidate scored the same ({scores[0]}) -- the ranking "
            f"this is supposed to be checking did not happen")
        assert scores == sorted(scores, reverse=True)

    def test_the_scores_are_the_ones_the_ranking_used(self, picked):
        _c, info = picked
        by_code = {c["code"][-6:]: c["score"] for c in info["scored"]}
        assert by_code["C50000"] == pytest.approx(0.6 * 50)
        assert by_code["P30000"] == pytest.approx(0.4 * 30)

    def test_the_picked_contracts_still_have_no_score_attached(self, picked):
        """The pop must still happen: `_score` is an internal field and the
        store writes `contracts` straight into option_contracts."""
        contracts, _info = picked
        assert not any("_score" in c for c in contracts)


class TestItRoundTripsThroughTheStore:

    def test_what_was_picked_is_what_comes_back(self, picked, store):
        _c, info = picked
        n = store.record_candidates(SESSION, "US.SOXL", info["scored"])
        assert n == 6
        back = store.candidates_on(SESSION, "US.SOXL")
        assert [c["rank"] for c in back] == list(range(1, 7))
        assert [c["code"] for c in back] == [c["code"] for c in info["scored"]]
        assert sum(c["picked"] for c in back) == 2

    def test_a_restart_replaces_rather_than_keeps_the_first_list(
            self, picked, store):
        """2026-11-23 picked twice, at 09:15 and 09:39. The second selection
        is the one that produced the session's data, so INSERT OR IGNORE
        would describe a selection that was replaced."""
        _c, info = picked
        store.record_candidates(SESSION, "US.SOXL", info["scored"])
        flipped = [{**c, "picked": not c["picked"]} for c in info["scored"]]
        store.record_candidates(SESSION, "US.SOXL", flipped)
        back = store.candidates_on(SESSION, "US.SOXL")
        assert sum(c["picked"] for c in back) == 4, (
            "the first selection was kept; a re-pick must overwrite it")
