"""--reconcile-only must score the underlying it was asked about.

The store-level filter is covered in tests/feeds/test_subscribed_on_filter.py.
This file covers the WIRING, which is a separate failure: the filter can be
perfect and the collector can still call it without an underlying, which is
exactly what it did. Measured -- removing `args.code` from that call left
every store test green.

The error path matters as much as the happy one. For a year the "nothing
recorded as subscribed for %s" message has named `args.code` while the query
behind it ignored the argument, so an operator reading it would conclude
that symbol had no session when another symbol's contracts were sitting
right there.
"""
from __future__ import annotations

import pytest

import analysis.option_tick_collector as mod


class _FakeStore:
    """Records how subscribed_on was called, and answers per underlying."""

    def __init__(self, table, orphans=0):
        self._table = table
        self.orphans = orphans
        self.calls: list[tuple] = []
        self.closed = False

    def subscribed_on(self, session, underlying):
        self.calls.append((session, underlying))
        return sorted(self._table.get(underlying, []))

    def all_subscribed_on(self, session):
        return sorted(c for v in self._table.values() for c in v)

    def underlyings_on(self, session):
        out = {k: len(v) for k, v in self._table.items()}
        if self.orphans:
            from feeds.option_tick_store import ORPHAN_KEY
            out[ORPHAN_KEY] = self.orphans
        return out

    def orphans_on(self, session):
        return self.orphans

    def close(self):
        self.closed = True


class _FakeCtx:
    def __init__(self, *a, **kw):
        pass

    def close(self):
        pass


@pytest.fixture()
def wired(monkeypatch):
    """Patch out OpenD, the store and the reconcile itself.

    `OpenQuoteContext` is imported inside main(), so it is patched on the
    `moomoo` module rather than on the collector.
    """
    import moomoo
    monkeypatch.setattr(moomoo, "OpenQuoteContext", _FakeCtx)

    table = {"US.SOXL": ["US.SOXL261127C43000", "US.SOXL261127P43000"],
             "US.QQQ": ["US.QQQ261124C600000"]}
    made: list[_FakeStore] = []
    orphans = {"n": 0}

    def _store(*a, **kw):
        """A NEW store per call, as the real constructor gives.

        The first version reused one instance after the first call, so a
        test that ran main() twice would have silently accumulated into one
        `calls` list rather than failing.
        """
        s = _FakeStore(table, orphans=orphans["n"])
        made.append(s)
        return s

    monkeypatch.setattr(mod, "OptionTickStore", _store)

    scored: list[list[str]] = []
    labelled: list[str | None] = []

    def _log(ctx, codes, store, session, underlying=None):
        """Records the LABEL as well as the set.

        The set is what gets scored; the label is what the coverage line
        prints. Both reconciles used to print "coverage <session>: ..."
        with no symbol in it, so two processes finishing in the same minute
        produced two lines that could not be told apart -- and the figure
        for one symbol was quotable as the day's coverage.
        """
        scored.append(list(codes))
        labelled.append(underlying)
        return {"ok": True}

    monkeypatch.setattr(mod, "_log_reconcile", _log)
    return table, made, scored, orphans, labelled


class TestReconcileOnlyIsScopedToOneSymbol:

    def test_it_scores_only_the_requested_symbol(self, wired):
        table, made, scored, _, labelled = wired
        rc = mod.main(["--code", "US.SOXL", "--reconcile-only",
                       "--session", "2026-11-24"])
        assert rc == 0, rc
        assert scored == [table["US.SOXL"]], scored
        # And the other symbol's contracts never reached the scorer: with
        # SOXL's tape holding nothing for them, every one would have landed
        # in caught_nothing.
        assert not set(scored[0]) & set(table["US.QQQ"])

    def test_the_other_symbol_gets_its_own_set(self, wired):
        """The control. Without it, a call that always returned SOXL's
        contracts -- or an empty list -- would satisfy the test above."""
        table, made, scored, _, labelled = wired
        rc = mod.main(["--code", "US.QQQ", "--reconcile-only",
                       "--session", "2026-11-24"])
        assert rc == 0, rc
        assert scored == [table["US.QQQ"]], scored

    def test_the_underlying_is_actually_passed_down(self, wired):
        _, made, _, _, labelled = wired
        mod.main(["--code", "US.QQQ", "--reconcile-only",
                  "--session", "2026-11-24"])
        assert made[-1].calls == [("2026-11-24", "US.QQQ")], made[-1].calls

    def test_the_store_is_closed(self, wired):
        _, made, _, _, labelled = wired
        mod.main(["--code", "US.SOXL", "--reconcile-only",
                  "--session", "2026-11-24"])
        assert made[-1].closed


class TestTheEmptyCaseNamesWhatIsActuallyThere:

    def test_an_unknown_symbol_reports_what_the_session_does_hold(
            self, wired, caplog):
        table, made, scored, _, labelled = wired
        with caplog.at_level("ERROR"):
            rc = mod.main(["--code", "US.NVDA", "--reconcile-only",
                           "--session", "2026-11-24"])
        assert rc == 5, rc
        assert scored == [], "nothing should have been scored"
        msg = "\n".join(r.getMessage() for r in caplog.records
                        if r.levelname == "ERROR")
        assert "US.NVDA" in msg
        # The point of the message: the operator must be able to see that
        # the session was not empty, only empty FOR THIS SYMBOL.
        assert "US.SOXL" in msg and "US.QQQ" in msg, msg

class TestAPartialOrphanIsAnnounced:
    """M18/M17 and the review's first finding: the empty-set branch was the
    only place orphans were mentioned, so a PARTIAL orphan -- 200
    subscribed, 50 with no contract row -- returned 150 codes and read as a
    quiet symbol."""

    def test_it_warns_and_still_reconciles(self, wired, caplog):
        table, made, scored, orphans, labelled = wired
        orphans["n"] = 50
        with caplog.at_level("WARNING"):
            rc = mod.main(["--code", "US.SOXL", "--reconcile-only",
                           "--session", "2026-11-24"])
        assert rc == 0, rc
        warn = [r.getMessage() for r in caplog.records
                if r.levelname == "WARNING"]
        assert any("50" in m and "option_contracts" in m for m in warn), warn
        # It warns INSTEAD OF giving up: the 150 that do join still get
        # scored, because a short reconcile beats no reconcile.
        assert scored == [table["US.SOXL"]]

    def test_no_warning_when_there_are_none(self, wired, caplog):
        """The control -- a warning on every run is a warning nobody reads."""
        with caplog.at_level("WARNING"):
            mod.main(["--code", "US.SOXL", "--reconcile-only",
                      "--session", "2026-11-24"])
        assert not [r for r in caplog.records
                    if r.levelname == "WARNING"
                    and "option_contracts" in r.getMessage()]

    def test_the_empty_branch_closes_the_store(self, wired):
        """It returned 5 without closing. The process exits immediately so
        it costs little, but the happy path closes and this did not."""
        _, made, _, _, labelled = wired
        assert mod.main(["--code", "US.NVDA", "--reconcile-only",
                         "--session", "2026-11-24"]) == 5
        assert made[-1].closed

    def test_the_orphan_key_is_named_not_blank(self, wired, caplog):
        """M17: the message used to format a blank key through an `or`.
        A key named "" reads as a symbol whose name is empty."""
        from feeds.option_tick_store import ORPHAN_KEY
        _, _, _, orphans, _ = wired
        orphans["n"] = 3
        with caplog.at_level("ERROR"):
            mod.main(["--code", "US.NVDA", "--reconcile-only",
                      "--session", "2026-11-24"])
        msg = chr(10).join(r.getMessage() for r in caplog.records
                           if r.levelname == "ERROR")
        assert f"{ORPHAN_KEY}=3" in msg, msg
