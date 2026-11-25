"""The suite must not be able to reach the live OpenD by accident.

The option-context stub in conftest covers ONE entrance. Measured, these
are not covered at all: trade_viewer_qt._connect_opend (which has no
listening probe, so a test that passes `args` hangs forever),
liq_hm_window, scheduler, signal_scanner, the three collectors, and
fetcher.fetch_klines on a cache miss. Today no test reaches any of them --
that is discipline held up by every test file individually, and a new file
breaks it silently.

Worse, the stub itself could vanish without a sound: pytest_configure
swallowed every import failure and returned, and simulating one took a
two-file run to 71 passed, green, with 48 connections to OpenD and 24
option-API calls behind it.

The guard turns both into a failed test. Measured cost before installing
it: the whole default run makes 5 socket connections and none of them go to
OpenD, so nothing legitimate is caught.

A live OpenD run is still available with --allow-opend, which is also what
the collectors' own tooling uses.
"""
from __future__ import annotations

import socket

import pytest


def _guard_installed() -> bool:
    """Whether this run has the guard, i.e. is not --allow-opend."""
    return getattr(socket.socket.connect, "__name__", "") == "guarded"


pytestmark = pytest.mark.skipif(
    not _guard_installed(),
    reason="--allow-opend: the guard is deliberately absent")


class TestItRefusesOpenD:

    @pytest.mark.parametrize("host", ["127.0.0.1", "localhost"])
    def test_a_connect_to_the_opend_port_raises(self, host):
        s = socket.socket()
        try:
            with pytest.raises(RuntimeError, match="connected to OpenD"):
                s.connect((host, 11111))
        finally:
            s.close()

    def test_the_message_says_how_to_opt_in(self):
        """A guard that only says "no" gets deleted by the next person who
        legitimately needs a live run."""
        s = socket.socket()
        try:
            with pytest.raises(RuntimeError, match="--allow-opend"):
                s.connect(("127.0.0.1", 11111))
        finally:
            s.close()


class TestItDoesNotCatchAnythingElse:
    """The guard is only worth having if it never cries wolf. The probe's
    own tests connect to a refused port and to a throwaway listener, and
    fetcher-backed tests open local files and DuckDB handles."""

    def test_another_port_on_localhost_is_untouched(self):
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        assert port != 11111
        c = socket.socket()
        try:
            c.connect(("127.0.0.1", port))      # must NOT raise
        finally:
            c.close()
            srv.close()

    def test_a_refused_port_still_raises_its_own_error(self):
        """Port 1 is what _opend_listening is tested against. The guard
        must not convert a normal ConnectionRefusedError into its own."""
        s = socket.socket()
        s.settimeout(0.5)
        try:
            with pytest.raises(OSError) as exc:
                s.connect(("127.0.0.1", 1))
            assert not isinstance(exc.value, RuntimeError), exc.value
        finally:
            s.close()

    def test_the_opend_port_on_another_host_is_not_ours_to_block(self):
        """11111 elsewhere is somebody else's service. Blocking it would
        make the guard a general-purpose port ban."""
        s = socket.socket()
        s.settimeout(0.2)
        try:
            with pytest.raises(OSError) as exc:
                s.connect(("10.255.255.1", 11111))
            assert not isinstance(exc.value, RuntimeError), exc.value
        except Exception as e:                                  # noqa: BLE001
            assert not isinstance(e, RuntimeError), e
        finally:
            s.close()


class TestTheGuardSurvivesAMissingStub:
    """The hole it was built for: pytest_configure returning early.

    It cannot be reproduced from inside a running session (configure has
    already happened), so this pins the two properties that made the hole
    dangerous rather than re-staging it: the guard is installed BEFORE the
    import that could fail, and a failure of that import is no longer
    swallowed into a bare return.
    """

    def test_the_guard_is_installed_before_the_import(self):
        import ast
        import inspect
        import conftest

        src = inspect.getsource(conftest.pytest_configure)
        tree = ast.parse(src.strip())
        body = tree.body[0].body
        names = []
        for node in ast.walk(ast.Module(body=body, type_ignores=[])):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                names.append((node.lineno, node.func.id))
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                names.append((node.lineno, "<import>"))
        guard = next(ln for ln, n in names if n == "_install_opend_guard")
        imp = next(ln for ln, n in names if n == "<import>")
        assert guard < imp, (
            "the guard must be installed before the import that can fail, "
            "or the window it covers is exactly the window it misses")

    def test_an_unexpected_import_failure_is_not_swallowed(self):
        """`except Exception: return` hid a broken tree as "no PyQt6 here"
        and left the suite talking to the live OpenD.

        Read off the AST, not the text: the comment explaining the old
        shape contains the words "except Exception", so a substring check
        failed against the corrected code.
        """
        import ast
        import inspect
        import conftest

        src = inspect.getsource(conftest.pytest_configure)
        tree = ast.parse(src.strip())
        caught = []
        reraises = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                caught.append(getattr(node.type, "id", None))
                reraises = reraises or any(
                    isinstance(n, ast.Raise) for n in ast.walk(node))
        assert caught, "no handler at all"
        assert "Exception" not in caught and "BaseException" not in caught, (
            f"a blanket handler is back: {caught}")
        assert caught == ["ImportError"], caught
        assert reraises, (
            "a module that is not an optional GUI dependency must crash "
            "the run rather than silently skipping the stub")
