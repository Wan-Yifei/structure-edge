"""Two underlyings are two claims on ONE account-wide subscription quota.

The option subscription limit is 200 contracts for the whole account, not
per collector. For one underlying that distinction never showed: the
collector asked for 200, got 200, and reconciled at 100%. Add a second and
the same code asks for 400 of a 200 budget, and nothing at runtime divides
it for them -- `option_used_quota` is not a contract count (50 contracts
read 1, 400 read 2, and adding 120 already-subscribed ones took it from 4
DOWN to 3), so it cannot be read and split. A static division in the config
is the only honest mechanism, which is what these guard.

What each case is about:

  * the sum must fit. This is the whole point, and a helper that computes a
    correct share while the launcher still passes none is the classic shape
    of this bug -- so the command line is asserted too, not just the
    arithmetic.
  * a NUMBER must mean the budget to divide, never "this many each". The
    second reading is the bug; the config spelling must not be able to
    express it.
  * a code with no share of its own takes what is left, because the config
    that will actually be hand-edited is the one that names only the symbol
    somebody cared about that week.
  * over-subscription is clamped and said out loud. Refusing would collect
    nothing from a typo; silence would show numbers the account never
    granted.
  * one underlying alone still gets the whole budget. The division must not
    quietly halve the case that has been running all along.

The clamp only makes the CONFIG add up. What was actually taken is read
back by the collector's verify_subscription immediately after it
subscribes -- see tests/analysis/test_option_telemetry.py.
"""
from __future__ import annotations

import pathlib
import sys
import types

import pytest

ROOT = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import analysis.scheduler as sched                               # noqa: E402
from analysis.scheduler import SchedulerApp                      # noqa: E402

_quotas = SchedulerApp._option_quotas
_pct = SchedulerApp._option_pct
BUDGET = sched._OPT_QUOTA_TOTAL
PAIR = ["US.SOXL", "US.QQQ"]


class TestTheSumFits:

    def test_two_codes_do_not_each_get_the_whole_budget(self):
        got = _quotas({}, PAIR)
        assert sum(got.values()) <= BUDGET, (
            f"{got} asks for {sum(got.values())} of an account-wide "
            f"{BUDGET} -- the two collectors would fight over it")
        assert all(v > 0 for v in got.values())

    @pytest.mark.parametrize("n", [1, 2, 3, 5])
    def test_any_number_of_codes_fits(self, n):
        codes = [f"US.SYM{i}" for i in range(n)]
        got = _quotas({}, codes)
        assert set(got) == set(codes)
        assert sum(got.values()) <= BUDGET, got

    def test_the_configured_pair_fits(self):
        """The shares actually in config/schedule.json."""
        got = _quotas({"quota": {"US.SOXL": 140, "US.QQQ": 60}}, PAIR)
        assert got == {"US.SOXL": 140, "US.QQQ": 60}
        assert sum(got.values()) == BUDGET


class TestANumberIsTheBudgetNotTheShare:

    def test_a_scalar_divides_rather_than_repeating(self):
        got = _quotas({"quota": 180}, PAIR)
        assert sum(got.values()) <= 180, (
            f"{got}: a scalar quota was read as 'this many EACH', which is "
            "the misreading the two spellings exist to prevent")
        assert got == {"US.SOXL": 90, "US.QQQ": 90}

    def test_a_scalar_replaces_the_collectors_own_figure(self):
        assert _quotas({"quota": 50}, ["US.SOXL"]) == {"US.SOXL": 50}

    def test_junk_falls_back_to_the_real_budget(self):
        assert sum(_quotas({"quota": "lots"}, PAIR).values()) <= BUDGET


class TestAMissingShareTakesWhatIsLeft:

    def test_the_named_code_keeps_its_number(self):
        got = _quotas({"quota": {"US.QQQ": 140}}, PAIR)
        assert got["US.QQQ"] == 140
        assert got["US.SOXL"] == BUDGET - 140

    def test_two_unnamed_codes_split_the_remainder(self):
        got = _quotas({"quota": {"US.SOXL": 100}},
                      ["US.SOXL", "US.QQQ", "US.SPY"])
        assert got["US.SOXL"] == 100
        assert got["US.QQQ"] == got["US.SPY"] == 50

    def test_a_code_not_in_the_list_does_not_take_a_share(self):
        """A leftover entry from a symbol that was dropped from `codes`."""
        got = _quotas({"quota": {"US.TSLA": 150}}, ["US.SOXL"])
        assert got == {"US.SOXL": BUDGET}


class TestOverSubscriptionIsClampedAndSaid:

    def test_the_sum_is_brought_back_under_the_budget(self, caplog):
        with caplog.at_level("WARNING"):
            got = _quotas({"quota": {"US.SOXL": 140, "US.QQQ": 140}}, PAIR)
        assert sum(got.values()) <= BUDGET, got
        assert all(v > 0 for v in got.values()), (
            "clamping must not take a code to zero -- a collector with no "
            "contracts runs all day and stores nothing")
        assert any("scaling" in r.message or "scaling" in str(r.msg)
                   for r in caplog.records), (
            "the config was changed silently; the panel would show shares "
            "the account never granted")

    def test_what_was_asked_for_is_logged_next_to_what_was_applied(self,
                                                                  caplog):
        with caplog.at_level("WARNING"):
            _quotas({"quota": {"US.SOXL": 300, "US.QQQ": 300}}, PAIR)
        text = " ".join(str(r.msg) % r.args if r.args else str(r.msg)
                        for r in caplog.records)
        assert "asked 300" in text, text


class TestOneUnderlyingIsUnchanged:

    def test_it_still_gets_the_whole_budget(self):
        assert _quotas({}, ["US.SOXL"]) == {"US.SOXL": BUDGET}

    def test_even_with_an_empty_map(self):
        assert _quotas({"quota": {}}, ["US.SOXL"]) == {"US.SOXL": BUDGET}


class TestThePreFilterIsPerCode:

    def test_a_map_is_applied_per_code(self):
        got = _pct({"pct": {"US.SOXL": 80.0, "US.QQQ": 15.0}}, PAIR)
        assert got == {"US.SOXL": 80.0, "US.QQQ": 15.0}

    def test_a_number_still_applies_to_every_code(self):
        assert _pct({"pct": 30.0}, PAIR) == {"US.SOXL": 30.0, "US.QQQ": 30.0}

    def test_a_code_missing_from_the_map_gets_the_wide_default(self):
        """Wide, not narrow. Selection is centred on last_price -- the
        previous REGULAR close -- and SOXL has opened 24% away from it, so
        an unconfigured code guessing narrow would collect the wrong
        strikes all session."""
        assert _pct({"pct": {"US.QQQ": 15.0}}, PAIR)["US.SOXL"] == 80.0

    def test_no_pct_at_all_is_the_old_behaviour(self):
        assert _pct({}, PAIR) == {"US.SOXL": 80.0, "US.QQQ": 80.0}


class TestTheLauncherActuallyPassesThem:
    """Arithmetic nobody puts on the command line changes nothing.

    _launch_opt is driven directly with a stub, because the thing under test
    is the argv it builds -- not Popen, not the thread, not the panel.
    """

    def _argv(self, monkeypatch, cfg, code):
        seen: dict = {}

        class _Proc:
            pid = 4321

            def poll(self):
                return None

        def _popen(cmd, **kw):
            seen["cmd"] = cmd
            return _Proc()

        monkeypatch.setattr(sched.subprocess, "Popen", _popen)
        monkeypatch.setattr(sched, "_kill_with_me", lambda pid: None)
        monkeypatch.setattr(sched.threading, "Thread",
                            lambda *a, **kw: types.SimpleNamespace(
                                start=lambda: None))
        stub = types.SimpleNamespace(
            cfg={"option_window": cfg},
            _DEFAULT_OPTION=SchedulerApp._DEFAULT_OPTION,
            _opt_procs={},
            _launched_at={},
            _option_codes=SchedulerApp._option_codes,
            _option_quotas=SchedulerApp._option_quotas,
            _option_pct=SchedulerApp._option_pct,
            _log=lambda msg: None,
            _tail_opt_proc=lambda *a, **kw: None,
        )
        SchedulerApp._launch_opt(stub, code)
        return seen.get("cmd")

    CFG = {"enabled": True, "codes": PAIR,
           "pct": {"US.SOXL": 80.0, "US.QQQ": 15.0},
           "quota": {"US.SOXL": 140, "US.QQQ": 60}}

    def test_the_quota_reaches_the_command_line(self, monkeypatch):
        argv = self._argv(monkeypatch, self.CFG, "US.QQQ")
        assert argv is not None, "the collector was never launched"
        assert "--quota" in argv, (
            f"no --quota in {argv}: each collector would default to the "
            f"whole account budget, which is the bug")
        assert argv[argv.index("--quota") + 1] == "60"

    def test_each_code_gets_its_own_share(self, monkeypatch):
        soxl = self._argv(monkeypatch, self.CFG, "US.SOXL")
        assert soxl[soxl.index("--quota") + 1] == "140"

    def test_the_pre_filter_is_this_codes_own(self, monkeypatch):
        qqq = self._argv(monkeypatch, self.CFG, "US.QQQ")
        soxl = self._argv(monkeypatch, self.CFG, "US.SOXL")
        assert qqq[qqq.index("--pct") + 1] == "15.0"
        assert soxl[soxl.index("--pct") + 1] == "80.0"

    def test_the_share_counts_the_other_codes_even_when_launched_alone(
            self, monkeypatch):
        """One collector at a time is how they start, so a share computed
        from `codes` rather than from who happens to be running already is
        what keeps the first one from taking the lot."""
        cfg = {"enabled": True, "codes": PAIR}      # no explicit shares
        argv = self._argv(monkeypatch, cfg, "US.SOXL")
        assert argv[argv.index("--quota") + 1] == str(BUDGET // 2)

    def test_a_code_outside_the_config_still_gets_a_divided_share(
            self, monkeypatch):
        """_launch_opt is reachable for a code mid-removal from the config.
        It must not hand that one the whole budget while the configured
        collectors are still holding theirs."""
        cfg = {"enabled": True, "codes": ["US.SOXL"]}
        argv = self._argv(monkeypatch, cfg, "US.QQQ")
        assert int(argv[argv.index("--quota") + 1]) <= BUDGET // 2


class TestThePanelDoesNotClobberTheMaps:
    """The panel writes four fields and merges the rest. A per-code map it
    knew nothing about used to be exactly the kind of thing a save flattened
    back to a default."""

    def test_saving_from_the_panel_keeps_quota_and_pct(self):
        cfg = {"option_window": {
            "enabled": True, "codes": PAIR,
            "pct": {"US.SOXL": 80.0, "US.QQQ": 15.0},
            "quota": {"US.SOXL": 140, "US.QQQ": 60},
            "reconcile_cron": "50 16 * * 1-5"}}
        stub = types.SimpleNamespace(
            cfg=cfg,
            _DEFAULT_OPTION=SchedulerApp._DEFAULT_OPTION,
            opt_enabled_var=types.SimpleNamespace(get=lambda: True),
            opt_code_var=types.SimpleNamespace(
                get=lambda: "US.SOXL, US.QQQ", set=lambda v: None),
            opt_start_var=types.SimpleNamespace(get=lambda: "09:15"),
            opt_end_var=types.SimpleNamespace(get=lambda: "16:25"),
            _option_codes=SchedulerApp._option_codes,
            _log=lambda msg: None,
        )
        out = SchedulerApp._option_config_from_panel(stub)
        assert out["quota"] == {"US.SOXL": 140, "US.QQQ": 60}
        assert out["pct"] == {"US.SOXL": 80.0, "US.QQQ": 15.0}
