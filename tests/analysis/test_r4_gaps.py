"""The properties the fourth review's fixes added, where a mutation lived.

Each class here exists because a mutation of the fix survived the tests
written alongside it. They are grouped by what the mutation broke, not by
module.
"""
from __future__ import annotations

import ast
import inspect
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import analysis.option_tick_collector as col                     # noqa: E402
import analysis.scheduler as sch                                 # noqa: E402
from feeds.option_tick_store import OptionTickStore              # noqa: E402

SESSION = "2026-11-24"
SOXL = ["US.SOXL261127C43000", "US.SOXL261127P43000"]


class _Snap:
    def __init__(self, vols):
        self._vols = vols

    def get_market_snapshot(self, codes):
        import pandas as pd
        from moomoo import RET_OK
        return RET_OK, pd.DataFrame(
            [{"code": c, "volume": self._vols.get(c, 0),
              "update_time": f"{SESSION} 16:00:00"} for c in codes])


@pytest.fixture()
def store(tmp_path):
    s = OptionTickStore(tmp_path / "opt.db")
    for c in SOXL:
        s.upsert_contracts([{"code": c, "underlying": "US.SOXL",
                             "strike": 43.0, "expiry": "2026-11-27",
                             "opt_type": "CALL"}])
    s.record_subscriptions(SESSION, SOXL)
    s.insert_ticks([{"code": SOXL[0], "ts": f"{SESSION} 10:00:00.000",
                     "price": 1.0, "volume": 500, "direction": "BUY",
                     "seq": 1}])
    yield s
    s.close()


class TestTheLineItselfNamesTheSymbol:
    """Asserting s["underlying"] left the LOG LINE untested, and the line
    is the artefact: two collectors reconcile in the same minute and both
    printed "coverage 2026-11-24: ...", which is how a figure for one
    symbol becomes "the day's coverage"."""

    def test_the_coverage_line_contains_it(self, store, caplog):
        with caplog.at_level("INFO"):
            col._log_reconcile(_Snap({c: 1000 for c in SOXL}), SOXL, store,
                               SESSION, "US.SOXL")
        line = next(m for m in (r.getMessage() for r in caplog.records)
                    if m.startswith("coverage"))
        assert "US.SOXL" in line, line
        assert SESSION in line, line

    def test_the_not_comparable_line_contains_it_too(self, store, caplog):
        """The other exit from _log_reconcile. A wrong-day run is the one
        most likely to be read as a finding, so it has to say whose."""
        snap = _Snap({c: 1000 for c in SOXL})

        def _stale(codes):
            import pandas as pd
            from moomoo import RET_OK
            return RET_OK, pd.DataFrame(
                [{"code": c, "volume": 1000,
                  "update_time": "2026-11-20 16:00:00"} for c in codes])

        snap.get_market_snapshot = _stale
        with caplog.at_level("INFO"):
            col._log_reconcile(snap, SOXL, store, SESSION, "US.SOXL")
        line = next(m for m in (r.getMessage() for r in caplog.records)
                    if "NOT COMPARABLE" in m)
        assert "US.SOXL" in line, line


class TestTheRatioIsNotReadBackFromTheTable:

    def test_the_function_does_not_call_coverage_summary(self):
        """The fix is that reconcile sums the rows it already has. A later
        edit that reaches for the table again reintroduces the scope bug
        whatever the argument says, because the table holds both symbols.
        """
        src = inspect.getsource(col.reconcile)
        tree = ast.parse(src.strip())
        called = {node.func.attr for node in ast.walk(tree)
                  if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Attribute)}
        assert "coverage_summary" not in called, sorted(called)

    def test_it_still_writes_the_table(self, store):
        """Write-only, not dropped. option_coverage is the record a later
        session reads; what changed is that this function no longer asks
        it a question it cannot answer per symbol.

        On the AST, because the comment above the fix contains the words
        "coverage_summary(session)" and a text search matches it -- that
        is how the first version of this test failed against correct code.
        """
        tree = ast.parse(inspect.getsource(col.reconcile).strip())
        called = {n.func.attr for n in ast.walk(tree)
                  if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Attribute)}
        assert "record_coverage" in called, sorted(called)

    def test_the_rows_and_the_table_agree_for_one_symbol(self, store):
        """The two routes must give the same answer when the table holds
        only this symbol -- otherwise the local sum is not a substitute,
        it is a second opinion."""
        s = col.reconcile(_Snap({c: 1000 for c in SOXL}), SOXL, store,
                          SESSION, "US.SOXL")
        back = store.coverage_summary(SESSION, "US.SOXL")
        assert (s["contracts"], s["our_vol"], s["feed_vol"]) == (
            back["contracts"], back["our_vol"], back["feed_vol"])
        assert s["ratio"] == back["ratio"]


class TestTheSchedulerCallSiteUsesTheSharedName:
    """`sch._opt_lock_name is col.lock_name` held while the call site had
    gone back to an f-string. The import is not the behaviour."""

    def test_no_literal_option_collector_prefix_remains(self):
        src = pathlib.Path(sch.__file__).read_text(encoding="utf-8")
        bad = [ln for ln in src.splitlines()
               if 'f"option_collector_' in ln or "f'option_collector_" in ln]
        assert not bad, bad

    def test_the_refusal_path_passes_the_shared_function(self):
        src = inspect.getsource(sch.SchedulerApp._check_option_window)
        tree = ast.parse(src.strip())
        failed = [n for n in ast.walk(tree)
                  if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Attribute)
                  and n.func.attr == "_collector_failed"]
        assert failed, "no _collector_failed call found"
        for call in failed:
            name_arg = call.args[2]
            assert isinstance(name_arg, ast.Call), ast.dump(name_arg)
            assert name_arg.func.id == "_opt_lock_name", ast.dump(name_arg)


class TestTheSymbolIsCaseFolded:
    """pick_contracts stores args.code verbatim as `underlying`, and
    SQLite's = is case sensitive on TEXT -- so `--code us.soxl` wrote rows
    subscribed_on("US.SOXL") could not see. The scheduler normalises, so
    this only ever bit a hand-typed run: which is what the first live
    session is."""

    def test_a_lowercase_code_reaches_the_store_upper_cased(self, tmp_path,
                                                             monkeypatch):
        """Through the real argument parser and the real main(), because
        the normalisation is one statement at the top of main and a
        hand-built Namespace missed fields that made main return early --
        the first version of this test asserted on an empty dict."""
        seen = {}

        class _Store:
            def __init__(self, *a, **k):
                pass

            def subscribed_on(self, session, under):
                seen["under"] = under
                return []

            def orphans_on(self, session):
                return 0

            def underlyings_on(self, session):
                return {}

            def close(self):
                pass

        monkeypatch.setattr(col, "OptionTickStore", _Store)

        class _Ctx:
            def __init__(self, *a, **k):
                pass

            def close(self):
                pass

        import moomoo
        monkeypatch.setattr(moomoo, "OpenQuoteContext", _Ctx)

        rc = col.main(["--code", "  us.soxl  ", "--reconcile-only",
                       "--session", SESSION,
                       "--db", str(tmp_path / "opt.db")])
        assert seen.get("under") == "US.SOXL", (seen, rc)

    def test_the_lowercase_form_finds_nothing_in_the_store(self, store):
        """Why it matters, measured rather than asserted: SQLite's = is
        case sensitive on TEXT, so the un-normalised spelling returns an
        empty set from a session that has 2 contracts."""
        assert store.subscribed_on(SESSION, "US.SOXL") == SOXL
        assert store.subscribed_on(SESSION, "us.soxl") == []

    def test_whitespace_is_stripped_too(self):
        """A trailing space from a .bat file is the same failure with no
        visible cause at all."""
        src = inspect.getsource(col.main)
        assert ".strip().upper()" in src, (
            "args.code must be normalised before anything stores it")


class TestTheImportFailureCrashesWhenItShould:
    """`if missing not in _OPTIONAL_GUI_MODULES: raise` mutated to
    `if False: raise` and survived -- the AST test only looked for a Raise
    node, which was still there."""

    def test_the_reraise_is_conditional_on_the_module(self):
        import conftest

        src = inspect.getsource(conftest.pytest_configure)
        tree = ast.parse(src.strip())
        raises = [n for n in ast.walk(tree) if isinstance(n, ast.Raise)]
        assert len(raises) == 1, ast.dump(tree)

        guards = [n for n in ast.walk(tree)
                  if isinstance(n, ast.If)
                  and any(isinstance(b, ast.Raise) for b in n.body)]
        assert len(guards) == 1, "the raise is not guarded by anything"
        test = guards[0].test
        assert isinstance(test, ast.Compare), ast.dump(test)
        # `missing not in _OPTIONAL_GUI_MODULES` -- not `False`, and not a
        # comparison against something unrelated.
        # The OPERAND, not merely a name somewhere inside it.
        # `missing not in (_OPTIONAL_GUI_MODULES | {"anything"})` still
        # contains the name while admitting every failed import as an
        # absent GUI, and that mutation survived the first version.
        assert len(test.comparators) == 1, ast.dump(test)
        right = test.comparators[0]
        assert isinstance(right, ast.Name), (
            "the membership test is against an expression, not the set "
            f"itself: {ast.dump(right)}")
        assert right.id == "_OPTIONAL_GUI_MODULES", right.id
        assert isinstance(test.ops[0], ast.NotIn), ast.dump(test)

    def test_the_optional_list_is_only_gui_modules(self):
        """If anything else joins the list, an ordinary broken import
        becomes "no PyQt6 here" again."""
        import conftest

        assert conftest._OPTIONAL_GUI_MODULES == frozenset(
            {"PyQt6", "pyqtgraph", "matplotlib"})


class TestTheFixtureFallsBackRatherThanSkipping:
    """Only --allow-opend tells the two apart, so the default run cannot
    observe it directly. Driven here by setting the module global the
    fixture reads."""

    def test_it_does_not_skip_when_nothing_was_stubbed(self, monkeypatch,
                                                       request):
        import conftest

        monkeypatch.setattr(conftest, "_REAL_ENSURE_OPTION_CTX", None)
        from analysis import trade_viewer_qt as tvq
        real = tvq.TradeViewerQt._ensure_option_ctx

        mp = pytest.MonkeyPatch()
        try:
            got = conftest.real_ensure_option_ctx.__wrapped__(mp)
        except BaseException as exc:                        # noqa: BLE001
            # BaseException, not Exception: Skipped derives straight from
            # BaseException (Skipped -> OutcomeException -> BaseException),
            # so `except Exception` did not catch it and the mutation
            # survived a SECOND time, still as a skip.
            #
            # A skip is caught and turned into a FAILURE on purpose. The
            # first version let pytest.skip propagate, so restoring the
            # skip branch made this test skip rather than fail -- and a
            # skip leaves the exit code at 0, which is what the mutation
            # run reads. The test detected the regression and the ruler
            # could not see it.
            import _pytest.outcomes
            if isinstance(exc, _pytest.outcomes.Skipped):
                pytest.fail(
                    "the fixture SKIPPED instead of falling back. Under "
                    "--allow-opend that removes the only four tests of "
                    "the probe-then-connect path from the one mode built "
                    f"for it: {exc}")
            raise
        finally:
            mp.undo()
        assert got is real, got

    def test_the_skip_reason_does_not_claim_a_dead_branch(self):
        """The old reason said "or PyQt6 is unavailable", which cannot
        happen: test_opend_probe imports trade_viewer_qt at module level
        with no importorskip, so a missing PyQt6 is a collection error."""
        import conftest

        src = inspect.getsource(conftest.real_ensure_option_ctx)
        assert "PyQt6 is unavailable" not in src, src


class TestEveryTestOfTheMethodAsksForTheRealOne:
    """The hole this whole class guards: one test in TestEnsureConsultsIt
    was missing the fixture, so it exercised conftest's no-op. Both
    assertions held, and its docstring says "The actual bug"."""

    def test_no_test_in_that_class_is_running_against_the_stub(self):
        path = ROOT / "tests" / "analysis" / "test_opend_probe.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        cls = next(n for n in tree.body
                   if isinstance(n, ast.ClassDef)
                   and n.name == "TestEnsureConsultsIt")
        missing = []
        for fn in cls.body:
            if not isinstance(fn, ast.FunctionDef):
                continue
            if not fn.name.startswith("test_"):
                continue
            args = {a.arg for a in fn.args.args}
            if "real_ensure_option_ctx" not in args:
                missing.append(fn.name)
        assert not missing, (
            "these call _ensure_option_ctx against conftest's no-op, so "
            f"their assertions hold for a method that does nothing: {missing}")
