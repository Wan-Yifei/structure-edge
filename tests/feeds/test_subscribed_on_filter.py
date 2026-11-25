"""Reconcile must score one underlying's contracts, not every symbol's.

`subscribed_on(session)` returned every contract subscribed that day with no
regard for which symbol it belonged to. With one underlying that is
harmless, which is why it survived; with two it hands SOXL's reconcile all
of QQQ's contracts, and because SOXL's tape holds nothing for them they all
land in `caught_nothing` -- the single detector for a dropped subscription,
and one that nobody reads again after the first day it fires two hundred
false positives.

The fixture here therefore has TWO underlyings in ONE session. A test with
one symbol passes against the unfiltered query it is meant to reject.
"""
from __future__ import annotations

import pytest

from feeds.option_tick_store import OptionTickStore

SESSION = "2026-11-24"

SOXL = [
    {"code": "US.SOXL261127C43000", "underlying": "US.SOXL",
     "expiry": "2026-11-27", "strike": 43.0, "opt_type": "CALL"},
    {"code": "US.SOXL261127P43000", "underlying": "US.SOXL",
     "expiry": "2026-11-27", "strike": 43.0, "opt_type": "PUT"},
    {"code": "US.SOXL261204C45000", "underlying": "US.SOXL",
     "expiry": "2026-12-04", "strike": 45.0, "opt_type": "CALL"},
]
QQQ = [
    {"code": "US.QQQ261124C600000", "underlying": "US.QQQ",
     "expiry": "2026-11-24", "strike": 600.0, "opt_type": "CALL"},
    {"code": "US.QQQ261124P600000", "underlying": "US.QQQ",
     "expiry": "2026-11-24", "strike": 600.0, "opt_type": "PUT"},
]


@pytest.fixture()
def store(tmp_path):
    s = OptionTickStore(tmp_path / "opt.db")
    yield s
    s.close()


@pytest.fixture()
def both(store):
    """Two symbols, one session -- the situation the filter exists for."""
    for group in (SOXL, QQQ):
        store.upsert_contracts(group)
        store.record_subscriptions(SESSION, [c["code"] for c in group])
    return store


class TestTheFilter:

    def test_each_underlying_gets_only_its_own(self, both):
        assert both.subscribed_on(SESSION, "US.SOXL") == sorted(
            c["code"] for c in SOXL)
        assert both.subscribed_on(SESSION, "US.QQQ") == sorted(
            c["code"] for c in QQQ)

    def test_the_two_sets_do_not_overlap_and_cover_everything(self, both):
        soxl = set(both.subscribed_on(SESSION, "US.SOXL"))
        qqq = set(both.subscribed_on(SESSION, "US.QQQ"))
        assert not soxl & qqq
        assert soxl | qqq == set(both.all_subscribed_on(SESSION))

    def test_the_whole_day_has_its_own_name(self, both):
        """Asking for every symbol is a deliberate act, not something you
        get by forgetting an argument -- the collector's bug WAS a missing
        argument. Asserting the count also makes the filtered assertions
        above mean something: without it, a filter that returned nothing
        would satisfy them too."""
        assert len(both.all_subscribed_on(SESSION)) == len(SOXL) + len(QQQ)

    def test_the_filter_is_not_optional(self, both):
        """A default would have handed the next caller the same trap."""
        with pytest.raises(TypeError):
            both.subscribed_on(SESSION)

    def test_an_unknown_underlying_is_empty_not_everything(self, both):
        assert both.subscribed_on(SESSION, "US.NVDA") == []

    def test_another_session_is_not_mixed_in(self, store):
        store.upsert_contracts(SOXL)
        store.record_subscriptions("2026-11-23", [SOXL[0]["code"]])
        store.record_subscriptions("2026-11-24", [SOXL[1]["code"]])
        assert store.subscribed_on("2026-11-23", "US.SOXL") == [SOXL[0]["code"]]
        assert store.subscribed_on("2026-11-24", "US.SOXL") == [SOXL[1]["code"]]

    def test_the_filtered_query_really_is_ordered(self, store):
        """Pinned properly, after a first attempt that could not fail.

        The order was never coming from option_contracts' primary key, as
        an earlier comment here claimed: it comes from
        option_subscriptions' own UNIQUE(session, code) index, which the
        planner happens to drive the join from. Run ANALYZE and the
        planner drives from option_contracts instead and returns rowid
        order -- so the clause IS observable, and the first version of this
        test only passed because its fixture inserted in code order.
        """
        rows = [
            {"code": "US.SOXL261204C45000", "underlying": "US.SOXL",
             "expiry": "2026-12-04", "strike": 45.0, "opt_type": "CALL"},
            {"code": "US.SOXL261127P43000", "underlying": "US.SOXL",
             "expiry": "2026-11-27", "strike": 43.0, "opt_type": "PUT"},
            {"code": "US.SOXL261127C43000", "underlying": "US.SOXL",
             "expiry": "2026-11-27", "strike": 43.0, "opt_type": "CALL"},
        ]
        store.upsert_contracts(rows)
        store.record_subscriptions(SESSION, [r["code"] for r in rows])
        store._con.execute("ANALYZE")      # makes the planner switch sides
        got = store.subscribed_on(SESSION, "US.SOXL")
        assert got == sorted(got), got



class TestUnderlyingsOn:

    def test_it_counts_each_symbol(self, both):
        assert both.underlyings_on(SESSION) == {"US.SOXL": 3, "US.QQQ": 2}

    def test_an_empty_session_is_empty(self, both):
        assert both.underlyings_on("2026-01-02") == {}

    def test_a_subscription_with_no_contract_row_is_reported_not_dropped(
            self, both):
        """The join is INNER, and a missing contract row would make a symbol
        look quiet rather than unrecorded. Cannot happen through the
        collector -- upsert_contracts runs on the same list immediately
        before record_subscriptions -- so it is forced here."""
        from feeds.option_tick_store import ORPHAN_KEY
        both.record_subscriptions(SESSION, ["US.SPY261124C500000"])
        got = both.underlyings_on(SESSION)
        # The EXACT dict. Asserting only got[""] and got["US.SOXL"] passed
        # against a LEFT JOIN too, which folds the orphan in under a None
        # key as well -- two entries for one row, and a None key that
        # formats into the error message as the string "None".
        assert got == {"US.SOXL": 3, "US.QQQ": 2, ORPHAN_KEY: 1}, got
        # And it is not silently handed to whoever asks.
        assert "US.SPY261124C500000" not in both.subscribed_on(
            SESSION, "US.SOXL")


class TestTheJoinIsTotalForRealSelections:
    """The property the collector relies on: everything it subscribes has a
    contract row, because it writes both from the same list."""

    def test_every_subscribed_code_joins(self, both):
        from feeds.option_tick_store import ORPHAN_KEY
        counts = both.underlyings_on(SESSION)
        assert ORPHAN_KEY not in counts, counts
        # A concrete number as well: "both sides empty" satisfies an
        # equality between two things that are both zero.
        assert sum(counts.values()) == len(SOXL) + len(QQQ) == 5
        assert len(both.all_subscribed_on(SESSION)) == 5
        assert both.orphans_on(SESSION) == 0

    def test_a_code_keeps_its_first_underlying(self, both):
        """option_contracts is INSERT OR IGNORE on the code, so a later
        write cannot move a contract to another symbol mid-session."""
        both.upsert_contracts([dict(SOXL[0], underlying="US.QQQ")])
        assert SOXL[0]["code"] in both.subscribed_on(SESSION, "US.SOXL")
        assert SOXL[0]["code"] not in both.subscribed_on(SESSION, "US.QQQ")


class TestTheMatchIsExactNotAPrefix:
    """US.QQQ and US.QQQM are both real ETFs with real option chains.

    A prefix match -- `c.underlying LIKE ? || '%'` -- passed every other
    test in this file, because no fixture had one symbol's code as another
    symbol's prefix. QQQ's reconcile would then score QQQM's contracts, the
    exact failure the filter exists to prevent, with the exact symptom
    (everything in caught_nothing).
    """

    QQQM = [
        {"code": "US.QQQM261124C190000", "underlying": "US.QQQM",
         "expiry": "2026-11-24", "strike": 190.0, "opt_type": "CALL"},
        {"code": "US.QQQM261124P190000", "underlying": "US.QQQM",
         "expiry": "2026-11-24", "strike": 190.0, "opt_type": "PUT"},
    ]

    def test_a_longer_symbol_sharing_a_prefix_is_not_swept_in(self, both):
        both.upsert_contracts(self.QQQM)
        both.record_subscriptions(SESSION, [c["code"] for c in self.QQQM])
        got = both.subscribed_on(SESSION, "US.QQQ")
        assert got == sorted(c["code"] for c in QQQ), got
        assert not any("QQQM" in c for c in got), got

    def test_the_longer_symbol_still_gets_its_own(self, both):
        """The control: a filter that matched nothing would satisfy the
        test above."""
        both.upsert_contracts(self.QQQM)
        both.record_subscriptions(SESSION, [c["code"] for c in self.QQQM])
        assert both.subscribed_on(SESSION, "US.QQQM") == sorted(
            c["code"] for c in self.QQQM)


class TestTheOrphanKeyIsNotAnEmptyString:
    """Asserted on the CONSTANT, not through it.

    Every other assertion in this file imports ORPHAN_KEY and compares
    against it, so setting it back to "" moved the ruler along with the
    thing being measured -- all 23 tests stayed green. The property that
    matters is that the key cannot be mistaken for a symbol.
    """

    def test_the_key_is_a_nonempty_sentinel(self):
        from feeds.option_tick_store import ORPHAN_KEY
        assert ORPHAN_KEY, "an empty key reads as a symbol with a blank name"
        assert not ORPHAN_KEY.startswith("US."), ORPHAN_KEY

    def test_it_sorts_and_formats_as_something_a_reader_understands(self, both):
        """It ends up in an operator-facing message via sorted(items()), so
        it has to survive that trip legibly."""
        from feeds.option_tick_store import ORPHAN_KEY
        both.record_subscriptions(SESSION, ["US.SPY261124C500000"])
        rendered = ", ".join(f"{k}={v}" for k, v in
                             sorted(both.underlyings_on(SESSION).items()))
        assert f"{ORPHAN_KEY}=1" in rendered, rendered
        assert "=1" != rendered.split(", ")[0], (
            f"the orphan entry rendered without a name: {rendered}")


class TestOrphansAreCountedPerSession:
    """M16: the orphan subquery had its own session filter, and nothing
    tested it -- so another day's orphan would be reported as this day's."""

    def test_another_sessions_orphan_is_not_counted_here(self, both):
        from feeds.option_tick_store import ORPHAN_KEY
        both.record_subscriptions("2026-11-20", ["US.SPY261120C500000"])
        assert both.orphans_on(SESSION) == 0
        assert ORPHAN_KEY not in both.underlyings_on(SESSION)
        # And it IS counted on its own day -- otherwise this passes against
        # a method that always answers zero.
        assert both.orphans_on("2026-11-20") == 1
        assert both.underlyings_on("2026-11-20") == {ORPHAN_KEY: 1}

    def test_a_partial_orphan_shortens_the_answer_silently(self, both):
        """The shape the reconcile warning exists for: subscribed_on's
        INNER JOIN drops the orphan, and "one fewer code" looks exactly
        like "this symbol was quiet"."""
        both.record_subscriptions(SESSION, ["US.SOXL261231C50000"])
        assert len(both.subscribed_on(SESSION, "US.SOXL")) == 3   # not 4
        assert both.orphans_on(SESSION) == 1
