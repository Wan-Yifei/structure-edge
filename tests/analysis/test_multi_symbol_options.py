"""Collecting more than one underlying.

Three things had to change and each had its own silent failure:

B2  Config held one `code` string, which the panel read into an entry and
    the save path wrote back from it -- two independent readings. A config
    file lacking that key showed the default in the entry and then
    PERSISTED it on the next save, so pressing Start could change which
    symbol gets collected while saying nothing about symbols.
B3  One `self._opt_proc` slot, and every branch of the relaunch logic said
    `return`. Correct for one symbol; with two, the first symbol being
    refused, dying young or sitting in a backoff abandoned the second for
    that whole tick -- and since the condition persists, for every tick.
B7  _kill_opt waited up to 20 seconds for its one child, serially. N
    symbols froze the Tk thread for up to N x 20, while the children were
    shutting down in parallel the whole time.
"""
from __future__ import annotations

import time as time_mod
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

import analysis.scheduler as mod

ET = ZoneInfo("America/New_York")
ORDINARY = "2026-11-25"


def _et(day, hhmm):
    return datetime.strptime(f"{day} {hhmm}", "%Y-%m-%d %H:%M").replace(tzinfo=ET)


class TestTheConfigNormaliser:
    """One function for the read and the write. The bug was having two."""

    N = staticmethod(mod.SchedulerApp._option_codes)

    def test_a_list_comes_back_as_itself(self):
        assert self.N({"codes": ["US.SOXL", "US.QQQ"]}) == ["US.SOXL", "US.QQQ"]

    def test_the_old_single_string_still_works(self):
        """An existing config has `code`, and must not silently become the
        default on first read.

        NOT "US.SOXL", which is also the fallback: with the default as the
        example, a normaliser that ignored `code` entirely fell through to
        the fallback and returned the same answer. Measured -- that
        mutation survived all 73 tests.
        """
        assert self.N({"code": "US.QQQ"}) == ["US.QQQ"]

    def test_the_entry_box_spelling_is_accepted(self):
        assert self.N({"codes": "US.SOXL, US.QQQ"}) == ["US.SOXL", "US.QQQ"]
        assert self.N({"codes": "US.SOXL; US.QQQ"}) == ["US.SOXL", "US.QQQ"]

    def test_case_and_blanks_are_normalised(self):
        assert self.N({"codes": " us.soxl ,, us.qqq "}) == ["US.SOXL", "US.QQQ"]

    def test_spaces_separate_too(self):
        """The label says "Codes:", so typing them space-separated is the
        natural thing to do. Treated as one token it became a single junk
        symbol and the status line read "running 0/1"."""
        assert self.N({"codes": "US.SOXL US.QQQ"}) == ["US.SOXL", "US.QQQ"]
        assert self.N({"codes": "US.SOXL, US.QQQ US.TQQQ"}) == [
            "US.SOXL", "US.QQQ", "US.TQQQ"]

    def test_pasted_quotes_are_stripped(self):
        """Copying a symbol out of a config file or a log brings its quotes
        along. With the shape check in place an unstripped quote is not
        merely ugly -- it fails the pattern and the code is dropped."""
        assert self.N({"codes": "'US.SOXL'"}) == ["US.SOXL"]
        assert self.N({"codes": '"US.SOXL", "US.QQQ"'}) == ["US.SOXL", "US.QQQ"]

    def test_duplicates_are_dropped_in_order(self):
        """Two collectors for one symbol would fight over the same
        single-instance lock, and the loser's backoff would grow forever."""
        assert self.N({"codes": ["US.QQQ", "US.SOXL", "US.QQQ"]}) == [
            "US.QQQ", "US.SOXL"]

    def test_codes_wins_over_code_when_both_are_present(self):
        """A file written by an older build and then edited has both. One
        of them has to lose, predictably."""
        assert self.N({"codes": ["US.QQQ"], "code": "US.SOXL"}) == ["US.QQQ"]

    def test_empty_falls_back_rather_than_collecting_nothing(self):
        """Silently collecting nothing is the failure mode this whole area
        keeps producing, so an empty config is not allowed to mean it."""
        for cfg in ({}, {"codes": []}, {"codes": ""}, {"code": "  "}):
            assert self.N(cfg) == ["US.SOXL"], cfg

    def test_the_default_config_is_already_a_list(self):
        assert mod.SchedulerApp._DEFAULT_OPTION["codes"] == ["US.SOXL"]

    def test_a_round_trip_through_the_entry_box_is_stable(self):
        """What the panel shows, parsed back, must be what it showed --
        otherwise saving a config you did not edit changes it."""
        codes = ["US.SOXL", "US.QQQ"]
        shown = ", ".join(self.N({"codes": codes}))
        assert self.N({"codes": shown}) == codes


class _Proc:
    #: Set by a test that wants the child to honour CTRL_BREAK.
    stops_on_signal = False
    #: Set by a test that needs wall time to pass inside wait().
    clock = None

    def __init__(self, pid, alive=True, rc=0):
        self.pid = pid
        self.stdout = []
        self._alive = alive
        self.returncode = rc
        self.signals: list = []
        self.waits: list = []
        self.grace_waits: list = []
        self.kill_waits: list = []
        self.killed = False

    def poll(self):
        return None if self._alive else self.returncode

    def send_signal(self, sig):
        self.signals.append(sig)

    def wait(self, timeout=None):
        # Tagged by which phase asked, not by magnitude. Splitting them on
        # `w > 5` broke the moment the budget was lowered past 5, and
        # lowering the threshold instead swept the terminate waits in.
        (self.kill_waits if self.killed else self.grace_waits).append(timeout)
        self.waits.append(timeout)
        # Burn the budget, so a shared deadline is observable at all. A stub
        # that returns instantly leaves the deadline untouched and hands
        # every child the full grace period, which looks like no sharing.
        if self.clock is not None and timeout:
            self.clock.advance(timeout)
        if self._alive and not self.stops_on_signal:
            raise mod.subprocess.TimeoutExpired("opt", timeout or 0)
        self._alive = False
        return self.returncode

    def terminate(self):
        self.killed = True
        self._alive = False

    def kill(self):
        self.killed = True
        self._alive = False


class _App:
    def __init__(self, codes=("US.SOXL", "US.QQQ")):
        self.cfg = {"option_window": {"enabled": True, "start": "09:15",
                                      "end": "16:25", "codes": list(codes)}}
        self._DEFAULT_OPTION = mod.SchedulerApp._DEFAULT_OPTION
        self._POST_CLOSE_PAD_MIN = mod.SchedulerApp._POST_CLOSE_PAD_MIN
        self._collector_running = True
        self._opt_procs: dict = {}
        self._launched_at: dict = {}
        self.opt_status_lbl = None
        self.launches: list[str] = []
        self.logged: list[str] = []

    _option_codes = staticmethod(mod.SchedulerApp._option_codes)
    _hhmm = staticmethod(mod.SchedulerApp._hhmm)
    _option_window_active = mod.SchedulerApp._option_window_active
    _collector_backoff_ok = mod.SchedulerApp._collector_backoff_ok
    _died_young = mod.SchedulerApp._died_young
    _collector_failed = mod.SchedulerApp._collector_failed
    _collector_started = mod.SchedulerApp._collector_started
    _kill_opt = mod.SchedulerApp._kill_opt

    def _launch_opt(self, code):
        self.launches.append(code)
        self._opt_procs[code] = _Proc(100 + len(self._opt_procs))

    def _log(self, msg):
        self.logged.append(msg)


def _tick(app, hhmm="10:00", day=ORDINARY):
    mod.SchedulerApp._check_option_window(app, _et(day, hhmm))


class TestOneProcessPerUnderlying:

    def test_both_symbols_are_launched(self):
        app = _App()
        _tick(app)
        assert app.launches == ["US.SOXL", "US.QQQ"], app.launches
        assert set(app._opt_procs) == {"US.SOXL", "US.QQQ"}

    def test_a_second_tick_launches_nothing_new(self):
        app = _App()
        _tick(app)
        _tick(app)
        assert app.launches == ["US.SOXL", "US.QQQ"], app.launches

    def test_only_the_dead_one_is_relaunched(self):
        app = _App()
        _tick(app)
        app._opt_procs["US.SOXL"]._alive = False
        app._launched_at["opt:US.SOXL"] = time_mod.monotonic() - 3600
        _tick(app)
        assert app.launches == ["US.SOXL", "US.QQQ", "US.SOXL"], app.launches


class TestTheFirstSymbolCannotAbandonTheSecond:
    """Every branch in the relaunch logic used to `return`. The condition
    that triggers it persists, so one refused symbol meant the other was
    never started at all -- not merely delayed by a tick."""

    def test_a_refusal_on_the_first_still_starts_the_second(self):
        app = _App()
        app._opt_procs["US.SOXL"] = _Proc(
            9, alive=False, rc=mod.single_instance.EXIT_ALREADY_RUNNING)
        _tick(app)
        assert "US.QQQ" in app.launches, app.launches
        assert "US.SOXL" not in app.launches, app.launches

    def test_the_first_dying_young_still_starts_the_second(self):
        app = _App()
        app._opt_procs["US.SOXL"] = _Proc(9, alive=False, rc=1)
        app._launched_at["opt:US.SOXL"] = time_mod.monotonic()
        _tick(app)
        assert app.launches == ["US.QQQ"], app.launches

    def test_a_backoff_on_the_first_still_starts_the_second(self):
        app = _App()
        app._opt_procs["US.SOXL"] = _Proc(
            9, alive=False, rc=mod.single_instance.EXIT_ALREADY_RUNNING)
        _tick(app)                      # arms SOXL's backoff
        app._opt_procs.pop("US.QQQ", None)
        app.launches.clear()
        _tick(app)
        assert app.launches == ["US.QQQ"], app.launches

    def test_the_backoff_is_per_symbol_not_shared(self):
        """A shared key would let one symbol's refusal hold the other back,
        which is the same bug wearing a dict."""
        app = _App()
        app._opt_procs["US.SOXL"] = _Proc(
            9, alive=False, rc=mod.single_instance.EXIT_ALREADY_RUNNING)
        _tick(app)
        assert set(app._relaunch_tries) == {"opt:US.SOXL"}, app._relaunch_tries


class TestALaterSymbolDyingIsTheCommonCase:
    """Every other test in this file kills the FIRST symbol.

    None of them reaches the live branch's `continue`, so swapping it for
    `break` survived the whole suite -- and `break` is worse here than
    anywhere else: with SOXL healthy, the loop would exit before QQQ on
    every tick, so QQQ stays down for the rest of the day.

    It is also the shape that actually happens: the first symbol is fine
    all session and the second dies at eleven.
    """

    def test_a_healthy_first_symbol_does_not_shield_a_dead_second(self):
        app = _App(codes=("US.SOXL", "US.QQQ", "US.TQQQ"))
        _tick(app)
        app.launches.clear()
        # SOXL and TQQQ stay up; the middle one dies after a long life.
        app._opt_procs["US.QQQ"]._alive = False
        app._launched_at["opt:US.QQQ"] = time_mod.monotonic() - 3600
        _tick(app)
        assert app.launches == ["US.QQQ"], (
            f"{app.launches} -- a `break` in the live branch would leave "
            f"this empty and QQQ down for the session")

    def test_the_last_symbol_is_reached_at_all(self):
        """The weakest form, and the one a `break` fails outright."""
        app = _App(codes=("US.SOXL", "US.QQQ", "US.TQQQ"))
        _tick(app)
        assert app.launches == ["US.SOXL", "US.QQQ", "US.TQQQ"], app.launches


class TestTheBackoffClearsOnceItSurvives:
    """There was no clearing point anywhere on the option path.

    tick and ob are cleared in the watchdog, on a process that is alive and
    past _died_young -- not in their launch, because a spawn is not yet a
    survival. The option path copied the launch half and not the clearing
    half, so three refusals at 09:15 still cost 64 seconds after an
    unrelated crash at 15:00.
    """

    def test_a_surviving_collector_clears_its_escalation(self):
        app = _App(codes=("US.SOXL",))
        app._opt_procs["US.SOXL"] = _Proc(
            9, alive=False, rc=mod.single_instance.EXIT_ALREADY_RUNNING)
        _tick(app)
        assert app._relaunch_tries.get("opt:US.SOXL") == 1
        # It starts, survives startup, and keeps running.
        app._opt_procs["US.SOXL"] = _Proc(10)
        app._launched_at["opt:US.SOXL"] = time_mod.monotonic() - 3600
        _tick(app)
        assert not app._relaunch_tries.get("opt:US.SOXL"), app._relaunch_tries

    def test_it_does_not_clear_while_still_inside_startup(self):
        """A process that spawned seconds ago has not survived anything --
        clearing here would make a crash-loop read as "attempt 1" forever,
        which is the bug the backoff was built for."""
        app = _App(codes=("US.SOXL",))
        app._opt_procs["US.SOXL"] = _Proc(
            9, alive=False, rc=mod.single_instance.EXIT_ALREADY_RUNNING)
        _tick(app)
        app._opt_procs["US.SOXL"] = _Proc(10)
        app._launched_at["opt:US.SOXL"] = time_mod.monotonic()   # just now
        _tick(app)
        assert app._relaunch_tries.get("opt:US.SOXL") == 1, app._relaunch_tries

    def test_one_symbol_clearing_does_not_clear_another(self):
        app = _App(codes=("US.SOXL", "US.QQQ"))
        for c in ("US.SOXL", "US.QQQ"):
            app._opt_procs[c] = _Proc(
                9, alive=False, rc=mod.single_instance.EXIT_ALREADY_RUNNING)
        _tick(app)
        assert set(app._relaunch_tries) == {"opt:US.SOXL", "opt:US.QQQ"}
        app._opt_procs["US.SOXL"] = _Proc(10)
        app._launched_at["opt:US.SOXL"] = time_mod.monotonic() - 3600
        _tick(app)
        assert "opt:US.SOXL" not in app._relaunch_tries
        assert app._relaunch_tries.get("opt:US.QQQ") == 1


class TestTheEntryBoxCannotOverwriteAGoodConfig:
    """M18 and M19 both survived the whole suite: the save path had no
    coverage, and it is where the headline bug lived.

    It needs no Tk -- the var is anything with get() and set().
    """

    class _Var:
        def __init__(self, text):
            self.text = text

        def get(self):
            return self.text

        def set(self, text):
            self.text = text

    class _Flag:
        def __init__(self, v=True):
            self.v = v

        def get(self):
            return self.v

    def _app(self, typed, stored=("US.SOXL", "US.QQQ")):
        app = _App(codes=stored)
        app.opt_enabled_var = self._Flag(True)
        app.opt_code_var = self._Var(typed)
        app.opt_start_var = self._Var("09:15")
        app.opt_end_var = self._Var("16:25")
        return app

    def _save(self, app):
        """Call the REAL method.

        The first version of this helper copied the save branch into the
        test, which kills nothing: both mutations it exists to catch --
        bypassing the normaliser, and keeping the old `code` key -- live in
        the production method, and a copy of the logic agrees with itself
        whatever the original does. The branch was extracted to
        _option_config_from_panel so this could reach it.
        """
        return mod.SchedulerApp._option_config_from_panel(app)

    def test_an_empty_box_keeps_the_stored_symbols(self):
        """Clear the box, press Start: the config used to be overwritten
        with the DEFAULT, because the normaliser's read-side fallback was
        applied on the write side too."""
        app = self._app("")
        opt = self._save(app)
        assert opt["codes"] == ["US.SOXL", "US.QQQ"], opt["codes"]

    def test_the_box_is_put_back_to_what_was_saved(self):
        """Otherwise the panel shows something that is not in the file."""
        app = self._app("")
        self._save(app)
        assert app.opt_code_var.get() == "US.SOXL, US.QQQ"
        assert any("not usable" in l for l in app.logged), app.logged

    def test_a_real_edit_is_saved(self):
        """The control. Without it, a save path that refused everything
        would pass both tests above."""
        app = self._app("US.TQQQ, US.SOXS")
        opt = self._save(app)
        assert opt["codes"] == ["US.TQQQ", "US.SOXS"], opt["codes"]

    def test_the_old_key_is_dropped_so_the_two_cannot_drift(self):
        app = self._app("US.TQQQ")
        app.cfg["option_window"]["code"] = "US.SOXL"
        opt = self._save(app)
        assert "code" not in opt, opt
        assert mod.SchedulerApp._option_codes(opt) == ["US.TQQQ"]

    def test_junk_in_the_box_does_not_become_a_symbol(self):
        app = self._app("banana")
        opt = self._save(app)
        assert opt["codes"] == ["US.SOXL", "US.QQQ"], opt["codes"]


class TestSymbolsDroppedFromTheConfig:

    def test_a_removed_symbol_is_stopped(self):
        """Left running it keeps 200 contracts booked against the
        account-wide quota for the rest of the day."""
        app = _App()
        _tick(app)
        app.cfg["option_window"]["codes"] = ["US.SOXL"]
        _tick(app)
        assert set(app._opt_procs) == {"US.SOXL"}
        assert any("no longer configured" in l for l in app.logged), app.logged

    def test_the_remaining_symbol_is_untouched(self):
        app = _App()
        _tick(app)
        kept = app._opt_procs["US.SOXL"]
        app.cfg["option_window"]["codes"] = ["US.SOXL"]
        _tick(app)
        assert app._opt_procs["US.SOXL"] is kept
        assert kept.signals == [], "the surviving collector was signalled"


class _Clock:
    """A monotonic clock the test moves by hand."""

    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


class TestStoppingIsBatched:

    def test_every_collector_is_signalled_before_any_is_waited_on(self):
        """The point of the change: signal everybody, then wait. Per-child
        signal-then-wait serialises shutdowns that overlap anyway."""
        app = _App()
        _tick(app)
        procs = dict(app._opt_procs)
        for p in procs.values():
            p.stops_on_signal = True    # a well-behaved child
        app._kill_opt()
        for code, p in procs.items():
            assert p.signals == [mod.signal.CTRL_BREAK_EVENT], (code, p.signals)
        assert not any(p.killed for p in procs.values()), (
            "a child that answered CTRL_BREAK was terminated anyway, so its "
            "unsubscribe and WAL checkpoint were skipped after all")

    def test_the_grace_period_is_shared_not_per_child(self, monkeypatch):
        """Measured on a clock that actually moves.

        Two stubborn children: the first consumes the whole 15 s and the
        second must then be offered what is LEFT, not another 15. Before
        this change the waits were 10 s each plus a 5 s terminate, serially
        -- on the Tk thread, so the window was frozen for all of it.
        """
        clock = _Clock()
        monkeypatch.setattr(mod.time_mod, "monotonic", clock)
        app = _App()
        _tick(app)
        procs = dict(app._opt_procs)
        for p in procs.values():
            p.clock = clock
        start = clock.t
        app._kill_opt()
        spent = clock.t - start
        # The terminate fallback is on top of the grace period and is
        # bounded per child at 5 s, which is the price of a child that
        # ignores CTRL_BREAK; the GRACE itself must not multiply.
        # Derived, not the literal 5. With _OPT_STOP_GRACE_S set to 5 or
        # less the literal made `graces` empty and `sum([]) <= 5.5` passed
        # vacuously.
        graces = [w for p in procs.values() for w in p.grace_waits if w]
        assert graces, "no grace waits observed -- the stub tagged nothing"
        assert sum(graces) <= mod._OPT_STOP_GRACE_S + 0.5, (
            f"grace waits {graces} exceed the "
            f"{mod._OPT_STOP_GRACE_S}s budget -- it is per child, not shared")
        assert spent <= mod._OPT_STOP_GRACE_S + 2 * 5 + 0.5, spent
        assert all(p.killed for p in procs.values()), "a stubborn child lived"

    def test_the_second_child_is_not_offered_a_fresh_budget(self, monkeypatch):
        """The sharpest form of the same statement, and the one that fails
        against a per-child timeout even if the total happens to fit."""
        clock = _Clock()
        monkeypatch.setattr(mod.time_mod, "monotonic", clock)
        app = _App()
        _tick(app)
        procs = [app._opt_procs[c] for c in ("US.SOXL", "US.QQQ")]
        for p in procs:
            p.clock = clock
        app._kill_opt()
        first = [w for w in procs[0].waits if w and w > 5][0]
        second = [w for w in procs[1].waits if w and w > 5]
        assert first == pytest.approx(mod._OPT_STOP_GRACE_S, abs=0.5), first
        assert not second or second[0] < 1.0, (
            f"the second child was given {second} after the first spent the "
            f"whole budget")

    def test_killing_a_subset_leaves_the_others_running(self):
        app = _App()
        _tick(app)
        other = app._opt_procs["US.QQQ"]
        app._kill_opt(["US.SOXL"])
        assert set(app._opt_procs) == {"US.QQQ"}
        assert other.signals == []

    def test_killing_nothing_is_a_no_op(self):
        app = _App()
        app._kill_opt()
        assert app.logged == []


class TestTheStatusLabelCountsThem:

    class _Lbl:
        def __init__(self):
            self.text = None

        def config(self, text):
            self.text = text

    def test_it_names_the_live_symbols(self):
        app = _App()
        app.opt_status_lbl = self._Lbl()
        _tick(app)
        assert app.opt_status_lbl.text == "running 2/2: US.QQQ, US.SOXL", (
            app.opt_status_lbl.text)

    def test_a_partial_start_is_visible_as_partial(self):
        """"running" with no count hid exactly the case worth seeing: one
        symbol up, one refused."""
        app = _App()
        app.opt_status_lbl = self._Lbl()
        app._opt_procs["US.SOXL"] = _Proc(
            9, alive=False, rc=mod.single_instance.EXIT_ALREADY_RUNNING)
        _tick(app)
        assert app.opt_status_lbl.text == "running 1/2: US.QQQ", (
            app.opt_status_lbl.text)
