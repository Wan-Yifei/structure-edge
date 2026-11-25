"""The coverage ratio has to belong to ONE symbol.

`reconcile` filtered its contract list by underlying and then threw the
filtering away: the ratio, contract count and volumes came back from
`store.coverage_summary(session)`, which joined nothing and summed every
row for the day. With two collectors writing one table that produced a
figure belonging to neither symbol, and -- because the two reconciles start
in the same minute -- a figure that depended on which process wrote first.
Reproduced before the fix: QQQ, truly 25% covered, printed 62.5%.

That number is the single output of the 16:50 run and the one most likely
to be written down. Two denominators from one earlier session (78.6% and
83.5%) are already quoted in CHANGELOG and in a table comment, so "the
coverage figure gets repeated as fact" is not a hypothetical here.

The fix sums the rows reconcile already has, so the table's scope cannot
come into it at all. These tests pin the number, not the mechanism.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from analysis.option_tick_collector import (                    # noqa: E402
    _underlying_of, reconcile)
from feeds.option_tick_store import OptionTickStore             # noqa: E402

SESSION = "2026-11-24"
SOXL = ["US.SOXL261127C43000", "US.SOXL261127P43000"]
QQQ = ["US.QQQ261127C60000", "US.QQQ261127P60000"]


class FakeSnapshot:
    """get_market_snapshot over a fixed volume table."""

    def __init__(self, vols: dict[str, int]):
        self._vols = vols

    def get_market_snapshot(self, codes):
        from moomoo import RET_OK

        import pandas as pd
        return RET_OK, pd.DataFrame(
            [{"code": c, "volume": self._vols.get(c, 0),
              "update_time": f"{SESSION} 16:00:00"} for c in codes])


@pytest.fixture()
def store(tmp_path):
    s = OptionTickStore(tmp_path / "opt.db")
    for code, under in [(c, "US.SOXL") for c in SOXL] + \
                       [(c, "US.QQQ") for c in QQQ]:
        s.upsert_contracts([{"code": code, "underlying": under,
                             "strike": 43.0, "expiry": "2026-11-27",
                             "opt_type": "CALL"}])
    s.record_subscriptions(SESSION, SOXL + QQQ)
    yield s
    s.close()


def _tick(store, code, vol):
    store.insert_ticks([{
        "code": code, "ts": f"{SESSION} 10:00:00.000", "price": 1.0,
        "volume": vol, "direction": "BUY", "seq": 1,
    }])


class TestTheRatioBelongsToOneSymbol:
    """SOXL fully covered, QQQ a quarter covered, both in one table."""

    @pytest.fixture(autouse=True)
    def _populate(self, store):
        for c in SOXL:
            _tick(store, c, 1000)       # 1000 of 1000 -> 100%
        for c in QQQ:
            _tick(store, c, 250)        # 250 of 1000 -> 25%
        self.ctx = FakeSnapshot({c: 1000 for c in SOXL + QQQ})

    def test_soxl_reads_its_own_hundred_percent(self, store):
        s = reconcile(self.ctx, SOXL, store, SESSION, "US.SOXL")
        assert s["ratio"] == pytest.approx(1.0), s
        assert s["contracts"] == 2, s
        assert s["our_vol"] == 2000 and s["feed_vol"] == 2000, s

    def test_qqq_reads_its_own_twenty_five_percent(self, store):
        """The number the old code printed as 62.5%."""
        s = reconcile(self.ctx, QQQ, store, SESSION, "US.QQQ")
        assert s["ratio"] == pytest.approx(0.25), s
        assert s["contracts"] == 2, s
        assert s["our_vol"] == 500 and s["feed_vol"] == 2000, s

    def test_the_answer_does_not_depend_on_who_wrote_first(self, store):
        """The old version's value moved with the write order, because it
        read the table back. Both orders, both symbols, same answers."""
        a = reconcile(self.ctx, SOXL, store, SESSION, "US.SOXL")["ratio"]
        b = reconcile(self.ctx, QQQ, store, SESSION, "US.QQQ")["ratio"]
        c = reconcile(self.ctx, QQQ, store, SESSION, "US.QQQ")["ratio"]
        d = reconcile(self.ctx, SOXL, store, SESSION, "US.SOXL")["ratio"]
        assert (a, b) == (d, c) == (pytest.approx(1.0), pytest.approx(0.25))

    def test_neither_answer_is_the_mixed_one(self, store):
        """Guards against a fix that scopes one call and not the other.
        0.625 is what the whole table says; it must appear nowhere."""
        for codes, under in [(SOXL, "US.SOXL"), (QQQ, "US.QQQ")]:
            s = reconcile(self.ctx, codes, store, SESSION, under)
            assert s["ratio"] != pytest.approx(0.625), (under, s)
            assert s["contracts"] != 4, (under, s)


class TestTheSummaryCarriesTheLabel:

    def test_the_underlying_is_in_the_result(self, store):
        _tick(store, SOXL[0], 10)
        s = reconcile(FakeSnapshot({c: 10 for c in SOXL}), SOXL, store,
                      SESSION, "US.SOXL")
        assert s["underlying"] == "US.SOXL"

    def test_it_is_derived_when_the_caller_does_not_say(self, store):
        """A reconcile started by hand still names what it reconciled,
        rather than printing a figure that could belong to either."""
        _tick(store, QQQ[0], 10)
        s = reconcile(FakeSnapshot({c: 10 for c in QQQ}), QQQ, store,
                      SESSION)
        assert s["underlying"] == "US.QQQ"


class TestTheDerivation:
    """_underlying_of labels a line; it must never guess."""

    def test_one_symbol(self):
        assert _underlying_of(SOXL) == "US.SOXL"

    def test_a_mixed_list_is_none_not_the_first_one(self):
        assert _underlying_of(SOXL + QQQ) is None

    def test_an_empty_list_is_none(self):
        assert _underlying_of([]) is None

    def test_a_non_option_code_is_none(self):
        assert _underlying_of(["US.SOXL"]) == "US.SOXL"
        assert _underlying_of(["garbage"]) is None

    def test_the_market_prefix_is_part_of_it(self):
        """US.SOXL and HK.SOXL are different symbols."""
        assert _underlying_of(["HK.SOXL261127C43000"]) == "HK.SOXL"
        assert _underlying_of(["US.SOXL261127C43000",
                               "HK.SOXL261127C43000"]) is None


class TestTheStoreMethodMakesTheCallerChoose:

    def test_the_scope_argument_is_required(self, store):
        """It took only a session, and the un-scoped answer was being used
        where a per-symbol one was meant. Not a default -- a default is
        how it happened."""
        with pytest.raises(TypeError):
            store.coverage_summary(SESSION)

    def test_filtered_and_unfiltered_differ_on_a_mixed_table(self, store):
        for c in SOXL:
            _tick(store, c, 1000)
        for c in QQQ:
            _tick(store, c, 250)
        ctx = FakeSnapshot({c: 1000 for c in SOXL + QQQ})
        reconcile(ctx, SOXL, store, SESSION, "US.SOXL")
        reconcile(ctx, QQQ, store, SESSION, "US.QQQ")

        everyone = store.coverage_summary(SESSION, None)
        soxl = store.coverage_summary(SESSION, "US.SOXL")
        qqq = store.coverage_summary(SESSION, "US.QQQ")

        assert everyone["contracts"] == 4
        assert soxl["contracts"] == 2 and qqq["contracts"] == 2
        assert soxl["ratio"] == pytest.approx(1.0)
        assert qqq["ratio"] == pytest.approx(0.25)
        assert everyone["ratio"] == pytest.approx(0.625), (
            "the mixed figure is still what an unfiltered read means -- "
            "this is the number that was being printed per symbol")
