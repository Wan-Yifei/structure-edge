"""The two guards that decide whether a collection session happens at all.

Both tested only `weekday() >= 5` before this, so a full closure read as an
ordinary trading day. On 2026-11-26 that would have meant: the collector
launched at 09:15, spent chain and snapshot calls against an account-wide
rate limit, raised "NO PRINTS in 3 minutes" against a shut exchange, wrote a
session of greeks rows carrying 11-25's spot, and closed with a 16:50
reconcile reporting all 200 contracts in `caught_nothing` -- the single
detector for a dropped subscription, spent on a day with no tape.

The dates here are real and this week, not synthetic: 11-26 is a full
closure, 11-27 a 13:00 half session, 11-25 an ordinary full one.
"""
from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

import pytest

ET = ZoneInfo("America/New_York")

THANKSGIVING = "2026-11-26"      # full closure
BLACK_FRIDAY = "2026-11-27"      # 13:00 close
ORDINARY = "2026-11-25"          # 16:00 close
SATURDAY = "2026-11-28"


def _et(day: str, hhmm: str) -> datetime:
    return datetime.strptime(f"{day} {hhmm}", "%Y-%m-%d %H:%M").replace(tzinfo=ET)


class TestCollectorInSession:
    """option_tick_collector.in_session gates greeks sampling."""

    @staticmethod
    def _fn():
        from analysis.option_tick_collector import in_session
        return in_session

    @pytest.mark.parametrize("hhmm", ["09:20", "12:00", "15:59", "16:10"])
    def test_a_full_closure_is_never_in_session(self, hhmm):
        assert self._fn()(_et(THANKSGIVING, hhmm)) is False, hhmm

    @pytest.mark.parametrize("hhmm", ["09:20", "12:00", "15:59", "16:10"])
    def test_the_same_clock_times_are_in_session_the_day_before(self, hhmm):
        """The control. Without it the closure test also passes against a
        guard that returns False for everything."""
        assert self._fn()(_et(ORDINARY, hhmm)) is True, hhmm

    def test_a_half_session_ends_at_the_early_close_plus_the_pad(self):
        from analysis.option_tick_collector import _POST_CLOSE_PAD_MIN
        assert _POST_CLOSE_PAD_MIN == 20
        fn = self._fn()
        assert fn(_et(BLACK_FRIDAY, "12:59")) is True
        assert fn(_et(BLACK_FRIDAY, "13:19")) is True      # inside the pad
        assert fn(_et(BLACK_FRIDAY, "13:21")) is False     # past it
        # And the ordinary day still runs to 16:20, so the early close is
        # being read per-date rather than applied to every day.
        assert fn(_et(ORDINARY, "13:21")) is True
        assert fn(_et(ORDINARY, "16:19")) is True
        assert fn(_et(ORDINARY, "16:21")) is False

    def test_before_the_open_is_out_on_every_kind_of_day(self):
        for day in (ORDINARY, BLACK_FRIDAY, THANKSGIVING):
            assert self._fn()(_et(day, "09:19")) is False, day

    def test_a_weekend_is_still_out(self):
        assert self._fn()(_et(SATURDAY, "12:00")) is False


class _FakeApp:
    """Enough of SchedulerApp to call the one method under test.

    Built by hand rather than by constructing the app: SchedulerApp opens a
    Tk window, a tray icon and an OpenD connection in __init__.
    """

    def __init__(self, start="09:15", end="16:25", enabled=True):
        self.cfg = {"option_window": {"enabled": enabled,
                                      "start": start, "end": end}}

    def __getattr__(self, name):
        import analysis.scheduler as mod
        attr = getattr(mod.SchedulerApp, name)
        # A plain instance method fetched from the CLASS is unbound, so
        # calling it through this stub would bind its first real argument
        # to `self` and the test would pass while measuring nonsense. The
        # two names actually needed are a staticmethod and a class
        # attribute, neither of which has that problem.
        if callable(attr) and not isinstance(
                mod.SchedulerApp.__dict__.get(name), staticmethod):
            raise AttributeError(
                f"_FakeApp refuses to hand out the unbound method {name!r} "
                f"-- give the stub a real implementation instead")
        return attr


def _window_active(app, when):
    import analysis.scheduler as mod
    return mod.SchedulerApp._option_window_active(app, when)


class TestSchedulerOptionWindow:
    """scheduler._option_window_active decides whether to LAUNCH."""

    def test_a_full_closure_never_opens_the_window(self):
        app = _FakeApp()
        for hhmm in ("09:15", "12:00", "16:20"):
            assert _window_active(app, _et(THANKSGIVING, hhmm)) is False, hhmm

    def test_the_same_times_open_it_the_day_before(self):
        app = _FakeApp()
        for hhmm in ("09:15", "12:00", "16:20"):
            assert _window_active(app, _et(ORDINARY, hhmm)) is True, hhmm

    def test_a_half_session_closes_the_window_early(self):
        app = _FakeApp()
        assert _window_active(app, _et(BLACK_FRIDAY, "13:24")) is True
        assert _window_active(app, _et(BLACK_FRIDAY, "13:26")) is False
        # 200 subscriptions would otherwise be held over three dead hours.
        assert _window_active(app, _et(BLACK_FRIDAY, "16:00")) is False

    def test_an_end_set_earlier_than_the_close_is_still_honoured(self):
        """Clamped, not replaced. A user who sets 11:00 means 11:00, and a
        half-day rule that overwrote it would be a different bug."""
        app = _FakeApp(end="11:00")
        assert _window_active(app, _et(ORDINARY, "10:59")) is True
        assert _window_active(app, _et(ORDINARY, "11:01")) is False
        assert _window_active(app, _et(BLACK_FRIDAY, "11:01")) is False

    def test_the_scheduler_pad_outlasts_the_collector_pad(self):
        """If the window shut first, the watchdog would kill the collector
        in the middle of its own last sampling minute."""
        import analysis.scheduler as mod
        from analysis.option_tick_collector import _POST_CLOSE_PAD_MIN
        assert mod.SchedulerApp._POST_CLOSE_PAD_MIN > _POST_CLOSE_PAD_MIN

    def test_disabled_beats_everything(self):
        app = _FakeApp(enabled=False)
        assert _window_active(app, _et(ORDINARY, "12:00")) is False

    def test_a_weekend_is_still_out(self):
        assert _window_active(_FakeApp(), _et(SATURDAY, "12:00")) is False


class TestTheTwoGuardsAgree:
    """They are separate functions with separate configs, and a day where
    one runs and the other does not is a collector the watchdog fights."""

    # Minutes of the day (at one-minute resolution) on which the collector
    # samples. Stated, because the implication test below asserts NOTHING
    # on a day with no sampling minutes -- measured, two of its four days
    # ran zero assertions, and only the mutation that reintroduces the bug
    # made them non-empty again. Pinning the counts makes each day carry a
    # claim of its own.
    SAMPLING_MINUTES = {ORDINARY: 421, BLACK_FRIDAY: 241,
                        THANKSGIVING: 0, SATURDAY: 0}

    @pytest.mark.parametrize("day", list(SAMPLING_MINUTES))
    def test_the_sampling_window_is_the_expected_length(self, day):
        """09:20 to the close + 20 min, inclusive at both ends: 421 minutes
        on a full session, 241 on a 13:00 one, none at all otherwise."""
        from analysis.option_tick_collector import in_session
        n = sum(in_session(_et(day, f"{h:02d}:{m:02d}"))
                for h in range(24) for m in range(60))
        assert n == self.SAMPLING_MINUTES[day], (day, n)

    @pytest.mark.parametrize("day", list(SAMPLING_MINUTES))
    def test_no_day_where_the_collector_samples_outside_the_window(self, day):
        from analysis.option_tick_collector import in_session
        app = _FakeApp()
        checked = 0
        for hour in range(0, 24):
            for minute in (0, 30):
                when = _et(day, f"{hour:02d}:{minute:02d}")
                if in_session(when):
                    checked += 1
                    assert _window_active(app, when) is True, when
        # Non-emptiness, per day: an implication is satisfied for free when
        # the antecedent never holds.
        assert (checked > 0) == (self.SAMPLING_MINUTES[day] > 0), (day, checked)


class TestTheCollectorRefusesToStartOnAClosure:
    """in_session guards the sampling LOOP, not the startup.

    A hand-started collector -- scripts/launch_option_collector.bat, the
    documented first-session move -- still took the single-instance lock,
    ran pick_contracts against the account-wide 10-per-30s chain budget and
    subscribed 200 contracts before idling all day on a shut exchange. The
    fix had to go in main(), before the context, so it holds whoever starts
    it.
    """

    @staticmethod
    def _run(monkeypatch, today, argv):
        """Call main() with the clock moved and OpenD made unreachable.

        The context is NOT stubbed: the assertion is that main returns
        before building one. If the guard is missing, this fails by
        reaching the stub and raising -- a louder failure than a quiet
        wrong return code.
        """
        import datetime as dt

        import analysis.option_tick_collector as mod

        class _Clock(dt.datetime):
            @classmethod
            def now(cls, tz=None):
                return dt.datetime.combine(today, dt.time(9, 30), tzinfo=tz)

        monkeypatch.setattr(mod, "datetime", _Clock)

        import moomoo

        def _refuse(*a, **kw):
            raise AssertionError(
                "main() built a quote context on a non-session day -- the "
                "holiday guard runs too late or not at all")

        monkeypatch.setattr(moomoo, "OpenQuoteContext", _refuse)
        return mod.main(argv)

    def test_it_refuses_on_a_full_closure(self, monkeypatch, caplog):
        from datetime import date
        with caplog.at_level("ERROR"):
            rc = self._run(monkeypatch, date(2026, 11, 26),
                           ["--code", "US.SOXL"])
        assert rc == 7, rc
        assert "exchange holiday" in caplog.text, caplog.text

    def test_it_refuses_on_a_weekend_and_says_which_reason(
            self, monkeypatch, caplog):
        from datetime import date
        with caplog.at_level("ERROR"):
            rc = self._run(monkeypatch, date(2026, 11, 28),
                           ["--code", "US.SOXL"])
        assert rc == 7, rc
        # The two reasons are kept apart on purpose: "it is Saturday" and
        # "the exchange is shut" send an operator to different places.
        assert "weekend" in caplog.text and "exchange holiday" not in caplog.text

    def test_always_overrides_it(self, monkeypatch):
        """--always is the documented escape hatch, and a guard with no way
        past it cannot be tested against a live OpenD on a quiet day."""
        from datetime import date
        with pytest.raises(AssertionError, match="built a quote context"):
            self._run(monkeypatch, date(2026, 11, 26),
                      ["--code", "US.SOXL", "--always"])

    def test_reconcile_only_is_exempt(self, monkeypatch):
        """Reconciling last Wednesday's tape ON a holiday is a reasonable
        request and spends no quota."""
        from datetime import date
        with pytest.raises(AssertionError, match="built a quote context"):
            self._run(monkeypatch, date(2026, 11, 26),
                      ["--code", "US.SOXL", "--reconcile-only",
                       "--session", "2026-11-25"])

    def test_an_ordinary_day_is_not_refused(self, monkeypatch):
        """The control. Without it, a guard that refused every day would
        pass both refusal tests above."""
        from datetime import date
        with pytest.raises(AssertionError, match="built a quote context"):
            self._run(monkeypatch, date(2026, 11, 25), ["--code", "US.SOXL"])


class _ReconcileApp:
    """Explicit stub for _check_option_reconcile_cron.

    Written out rather than reusing _FakeApp: that one now refuses to hand
    out unbound methods, and this path touches several. _cron_field_match
    is a staticmethod and is taken from the real class, because a stub of
    the cron parser would turn this into a test of the stub.
    """

    def __init__(self, cron="50 16 * * 1-5", codes=("US.SOXL",)):
        import analysis.scheduler as mod
        self.cfg = {"option_window": {"enabled": True,
                                      "reconcile_cron": cron,
                                      "codes": list(codes)}}
        self._DEFAULT_OPTION = mod.SchedulerApp._DEFAULT_OPTION
        # Module imported lazily throughout this file, so the normaliser is
        # bound here rather than at class-body time.
        self._option_codes = mod.SchedulerApp._option_codes
        self._last_opt_reconcile_minute = None
        # Mirrors SchedulerApp.__init__. Without it the cron raised
        # AttributeError, the enclosing except logged it, and every
        # assertion here read as "the cron did not fire" -- which is why
        # _assert_no_cron_error exists below.
        self._opt_reconciled_on = (None, set())
        self._opt_reconcile_proc = None
        self.logged: list[str] = []

    def _log(self, msg):
        self.logged.append(msg)

    def assert_no_cron_error(self):
        """The enclosing try logs and swallows, so a harness that has
        drifted from SchedulerApp.__init__ looks exactly like a cron that
        correctly declined to fire. Three tests here failed that way."""
        errs = [m for m in self.logged if "Reconcile cron error" in m]
        assert not errs, errs

    def _tail_opt_reconcile(self, proc=None, code=""):
        """Signature matches the real one. The stub took only self, which
        the stubbed-out threading.Thread hid."""
        pass

    @property
    def _cron_field_match(self):
        import analysis.scheduler as mod
        return mod.SchedulerApp._cron_field_match


class TestTheReconcileCronSkipsAClosure:
    """A cron's dow field knows about weekends and nothing else.

    At 16:50 on a full closure this fired and asked the collector to
    reconcile a day that never happened. The outcome is softer than a wrong
    number -- subscribed_on comes back empty and the run exits 5 with "was
    the collector running that day?" -- but it is still a false alarm from
    the one check that can catch real under-collection.
    """

    @staticmethod
    def _fire(monkeypatch, app, when):
        import analysis.scheduler as mod

        calls = []

        class _Proc:
            def poll(self):
                return None

        def _popen(cmd, **kw):
            calls.append(cmd)
            return _Proc()

        monkeypatch.setattr(mod.subprocess, "Popen", _popen)
        monkeypatch.setattr(mod.threading, "Thread",
                            lambda *a, **kw: type("T", (), {"start": lambda s: None})())
        mod.SchedulerApp._check_option_reconcile_cron(app, when)
        return calls

    def test_it_does_not_fire_on_thanksgiving(self, monkeypatch):
        app = _ReconcileApp()
        calls = self._fire(monkeypatch, app, _et(THANKSGIVING, "16:50"))
        assert calls == [], calls
        # And it did not consume the minute either: a closure must not look
        # like "already ran" to a later tick.
        assert app._last_opt_reconcile_minute is None

    def test_it_does_fire_on_the_day_before(self, monkeypatch):
        """The control, and the one that makes the test above mean
        something: without it, a method that never fired would pass."""
        app = _ReconcileApp()
        calls = self._fire(monkeypatch, app, _et(ORDINARY, "16:50"))
        assert len(calls) == 1, calls
        assert "--reconcile-only" in calls[0]
        assert "2026-11-25" in calls[0]

    def test_it_fires_on_a_half_session_too(self, monkeypatch):
        """A 13:00 close is still a session, and its tape still needs
        checking -- the early close must not be read as a closure."""
        app = _ReconcileApp()
        calls = self._fire(monkeypatch, app, _et(BLACK_FRIDAY, "16:50"))
        assert len(calls) == 1, calls

    def test_it_does_not_fire_on_a_saturday(self, monkeypatch):
        app = _ReconcileApp()
        assert self._fire(monkeypatch, app, _et(SATURDAY, "16:50")) == []

    def test_one_symbols_popen_failing_does_not_abandon_the_others(
            self, monkeypatch):
        """Worse here than anywhere else: the minute is claimed BEFORE the
        loop and the cron matches one minute a day, so a symbol skipped
        here is unreconciled for the whole session -- no retry."""
        import analysis.scheduler as mod

        app = _ReconcileApp(codes=("US.SOXL", "US.QQQ", "US.TQQQ"))
        attempted = []

        class _P:
            def poll(self):
                return None

        def _popen(cmd, **kw):
            code = cmd[cmd.index("--code") + 1]
            attempted.append(code)
            if code == "US.SOXL":
                raise OSError(24, "Too many open files")
            return _P()

        monkeypatch.setattr(mod.subprocess, "Popen", _popen)
        monkeypatch.setattr(
            mod.threading, "Thread",
            lambda *a, **kw: type("T", (), {"start": lambda s: None})())
        mod.SchedulerApp._check_option_reconcile_cron(
            app, _et(ORDINARY, "16:50"))

        assert attempted == ["US.SOXL", "US.QQQ", "US.TQQQ"], attempted
        assert any("US.SOXL" in l and "failed to start" in l
                   for l in app.logged), app.logged

    def test_disabled_still_beats_everything(self, monkeypatch):
        app = _ReconcileApp()
        app.cfg["option_window"]["enabled"] = False
        assert self._fire(monkeypatch, app, _et(ORDINARY, "16:50")) == []
