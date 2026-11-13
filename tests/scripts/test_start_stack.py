"""The launcher must never hand the scheduler a dead OpenD.

Running the real thing exercises one branch only -- the one where everything
is already up and it has nothing to do. The branch that matters is the one
after a restart, and it cannot be rehearsed on this machine without killing
the collection it exists to protect. So it is pinned here instead.

The property worth defending is NOT "it starts things". It is that a
scheduler is never started against an OpenD that is not answering: its
collectors would connect to a dead port, churn, and the session would look
healthy from the outside while recording nothing. Failing loudly and
starting nothing is the correct outcome, and the only one that is obvious
from the log afterwards.
"""

import importlib.util
import pathlib
import socket
import sys
import types

import pytest

_SRC = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "start_stack.py"


def _load():
    """Import by path: scripts/ is not a package, and making it one would
    change how the launcher is run everywhere else."""
    spec = importlib.util.spec_from_file_location("start_stack_under_test",
                                                  _SRC)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


ss = _load()


@pytest.fixture(autouse=True)
def _quiet(monkeypatch, tmp_path):
    """Keep the test out of the real log file."""
    monkeypatch.setattr(ss, "LOG", tmp_path / "start_stack.log")
    monkeypatch.setattr(ss, "POLL_S", 0.0)


class TestReadinessIsAPortNotAProcess:
    """OpenD runs for as long as it sits on its login screen. Treating the
    process as the ready signal starts the scheduler against a port nobody
    is listening on -- which is the state this machine was actually in."""

    def test_a_listening_port_is_up(self):
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        try:
            assert ss.api_up(*srv.getsockname()) is True
        finally:
            srv.close()

    def test_a_closed_port_is_not(self):
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        addr = srv.getsockname()
        srv.close()            # nothing listening there now
        assert ss.api_up(*addr) is False


class TestItRefusesRatherThanStartAHalfStack:

    def _run(self, monkeypatch, *, api_answers, opend=1234, sched=None):
        started = []
        monkeypatch.setattr(ss, "api_endpoint", lambda: ("127.0.0.1", 11111))
        monkeypatch.setattr(ss, "api_up", lambda h, p: api_answers)
        monkeypatch.setattr(ss, "running", lambda frag: opend)
        monkeypatch.setattr(ss, "scheduler_pid", lambda: sched)
        monkeypatch.setattr(ss.subprocess, "Popen",
                            lambda cmd, **k: started.append(cmd) or
                            types.SimpleNamespace(pid=1))
        monkeypatch.setattr(ss.time, "sleep", lambda s: None)
        monkeypatch.setattr(sys, "argv", ["start_stack.py", "--wait", "0"])
        rc = ss.main()
        return rc, started

    def test_a_silent_api_means_no_scheduler(self, monkeypatch):
        rc, started = self._run(monkeypatch, api_answers=False)
        assert started == [], (
            "started something while OpenD's API was closed; its collectors "
            "would come up against a dead port and churn")
        assert rc != 0, "a refusal has to be visible in the exit code"

    def test_it_says_why_in_the_log(self, monkeypatch, capsys):
        self._run(monkeypatch, api_answers=False)
        out = capsys.readouterr().out
        assert "NOT" in out and "scheduler" in out, out

    def test_a_running_scheduler_is_left_alone(self, monkeypatch):
        """Idempotent: this is the path that runs when the task fires on a
        logon into a session that is already working."""
        rc, started = self._run(monkeypatch, api_answers=True, sched=24544)
        assert started == [], started
        assert rc == 0

    def test_with_the_api_up_and_no_scheduler_it_starts_one(self, monkeypatch):
        """The branch that cannot be rehearsed on the live machine."""
        rc, started = self._run(monkeypatch, api_answers=True, sched=None)
        assert len(started) == 1, started
        assert "scheduler" in " ".join(started[0])

    def test_a_missing_opend_is_not_silent(self, monkeypatch):
        monkeypatch.setattr(ss, "OPEND", pathlib.Path("Z:/nope/OpenD.exe"))
        rc, started = self._run(monkeypatch, api_answers=False, opend=None)
        assert started == [] and rc != 0


class TestCheckModeAnswersTheQuestion:
    """Written because a mutation survived: making --check return 0 always
    left every test green. That is the one flag meant to be trusted without
    reading the output -- a cron line or a glance at the exit code -- and a
    --check that cannot say "no" is worse than not having it."""

    def _check(self, monkeypatch, *, api_answers, sched):
        started = []
        monkeypatch.setattr(ss, "api_endpoint", lambda: ("127.0.0.1", 11111))
        monkeypatch.setattr(ss, "api_up", lambda h, p: api_answers)
        monkeypatch.setattr(ss, "running", lambda frag: 1234)
        monkeypatch.setattr(ss, "scheduler_pid", lambda: sched)
        monkeypatch.setattr(ss.subprocess, "Popen",
                            lambda cmd, **k: started.append(cmd))
        monkeypatch.setattr(sys, "argv", ["start_stack.py", "--check"])
        return ss.main(), started

    def test_zero_only_when_both_halves_are_up(self, monkeypatch):
        rc, started = self._check(monkeypatch, api_answers=True, sched=24544)
        assert rc == 0
        assert started == [], "--check must not start anything"

    def test_a_closed_api_is_not_zero(self, monkeypatch):
        rc, _ = self._check(monkeypatch, api_answers=False, sched=24544)
        assert rc != 0

    def test_a_missing_scheduler_is_not_zero(self, monkeypatch):
        rc, _ = self._check(monkeypatch, api_answers=True, sched=None)
        assert rc != 0

    def test_it_never_starts_anything(self, monkeypatch):
        _, started = self._check(monkeypatch, api_answers=False, sched=None)
        assert started == []


class TestTheEndpointComesFromOpenDsOwnConfig:
    """Hardcoding 11111 works until the day it is changed in OpenD and not
    here, and then the launcher reports a closed port forever."""

    def test_it_reads_ip_and_port(self, monkeypatch, tmp_path):
        xml = tmp_path / "OpenD.xml"
        xml.write_text("<config><ip>127.0.0.1</ip>"
                       "<api_port>12345</api_port></config>",
                       encoding="utf-8")
        monkeypatch.setattr(ss, "OPEND_XML", xml)
        assert ss.api_endpoint() == ("127.0.0.1", 12345)

    def test_an_unreadable_config_falls_back_instead_of_crashing(
            self, monkeypatch, tmp_path):
        bad = tmp_path / "OpenD.xml"
        bad.write_text("not xml at all <<<", encoding="utf-8")
        monkeypatch.setattr(ss, "OPEND_XML", bad)
        assert ss.api_endpoint() == ("127.0.0.1", 11111)


class TestItLooksPastTheTrampoline:
    r"""`.venv\Scripts\python.exe` execs the real interpreter, so a naive
    search finds the wrapper. A launcher that matches the wrapper reports a
    scheduler that is not there and then starts nothing."""

    def test_the_venv_wrapper_is_not_mistaken_for_the_scheduler(
            self, monkeypatch):
        fake = types.SimpleNamespace(process_iter=lambda attrs: [
            types.SimpleNamespace(info={"pid": 111, "cmdline": [
                r"E:\project\moomoo\.venv\Scripts\python.exe",
                "main.py", "scheduler"]}),
        ])
        monkeypatch.setitem(sys.modules, "psutil", fake)
        assert ss.scheduler_pid() is None

    def test_the_real_interpreter_is_found(self, monkeypatch):
        fake = types.SimpleNamespace(process_iter=lambda attrs: [
            types.SimpleNamespace(info={"pid": 111, "cmdline": [
                r"E:\project\moomoo\.venv\Scripts\python.exe",
                "main.py", "scheduler"]}),
            types.SimpleNamespace(info={"pid": 24544, "cmdline": [
                r"C:\Users\x\AppData\Roaming\uv\python\cpython-3.12"
                r"\python.exe", "main.py", "scheduler"]}),
        ])
        monkeypatch.setitem(sys.modules, "psutil", fake)
        assert ss.scheduler_pid() == 24544
