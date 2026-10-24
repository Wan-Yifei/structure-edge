"""The viewer must not block on OpenD that isn't there.

OpenQuoteContext does not raise on a refused port -- it retries the
connect forever, on the calling thread. The viewer builds it from
__init__ (the option toggle has to run once, because a checkbox that
starts ticked never emits stateChanged), so with Options on and OpenD
down the window never appeared at all. Measured: TCP refused in ~1ms,
the constructor still spinning in network_manager.poll twenty seconds
later.

These pin the probe and, more importantly, that _ensure_option_ctx
consults it before reaching for the SDK at all.
"""
import pathlib
import socket
import sys
import threading

ROOT = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import analysis.trade_viewer_qt as tv  # noqa: E402

_ensure = tv.TradeViewerQt.__dict__["_ensure_option_ctx"]


class FakeViewer:
    """Only what _ensure_option_ctx touches."""

    def __init__(self):
        self._option_ctx = None
        self._opend_warned = False
        self.logs = []

    def _log(self, msg):
        self.logs.append(msg)


class TestProbe:
    def test_a_refused_port_is_false_and_returns_promptly(self):
        import time
        # Port 1 is not something anything listens on.
        t0 = time.perf_counter()
        assert tv._opend_listening("127.0.0.1", 1, timeout=0.5) is False
        assert time.perf_counter() - t0 < 2.0

    def test_a_listening_port_is_true(self):
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        try:
            assert tv._opend_listening("127.0.0.1", port) is True
        finally:
            srv.close()

    def test_an_unroutable_host_does_not_hang_past_the_timeout(self):
        """A refused connect is instant; a dropped one is what the timeout
        is for, and the whole point is that this function always returns."""
        import time
        t0 = time.perf_counter()
        tv._opend_listening("10.255.255.1", 11111, timeout=0.3)
        assert time.perf_counter() - t0 < 3.0


class TestEnsureConsultsIt:
    def test_it_does_not_construct_the_sdk_when_nothing_listens(self,
                                                                monkeypatch):
        """The actual bug. Calling OpenQuoteContext here is what hangs, so
        the test is that it is never called, not that it failed."""
        built = []
        monkeypatch.setattr(tv, "OpenQuoteContext",
                            lambda **kw: built.append(kw))
        monkeypatch.setattr(tv, "_opend_listening", lambda *a, **k: False)
        v = FakeViewer()
        _ensure(v)
        assert built == [], "the SDK was constructed with OpenD down"
        assert v._option_ctx is None

    def test_it_connects_once_opend_is_up(self, monkeypatch):
        monkeypatch.setattr(tv, "OpenQuoteContext", lambda **kw: ("ctx", kw))
        monkeypatch.setattr(tv, "_opend_listening", lambda *a, **k: True)
        v = FakeViewer()
        _ensure(v)
        assert v._option_ctx[0] == "ctx"

    def test_a_later_poll_picks_it_up_without_a_restart(self, monkeypatch):
        """The option timer keeps calling in here, so coming up late has to
        be enough on its own -- no reconnect button, no restart."""
        up = {"yes": False}
        monkeypatch.setattr(tv, "OpenQuoteContext", lambda **kw: "ctx")
        monkeypatch.setattr(tv, "_opend_listening", lambda *a, **k: up["yes"])
        v = FakeViewer()
        _ensure(v)
        assert v._option_ctx is None
        up["yes"] = True
        _ensure(v)
        assert v._option_ctx == "ctx"

    def test_the_warning_is_logged_on_transitions_not_every_poll(self,
                                                                 monkeypatch):
        """At one line per poll this would be four a minute saying the same
        thing, which is how a log stops being read."""
        up = {"yes": False}
        monkeypatch.setattr(tv, "OpenQuoteContext", lambda **kw: "ctx")
        monkeypatch.setattr(tv, "_opend_listening", lambda *a, **k: up["yes"])
        v = FakeViewer()
        for _ in range(5):
            _ensure(v)
        assert len(v.logs) == 1, v.logs
        up["yes"] = True
        _ensure(v)
        assert len(v.logs) == 2, "coming back up is worth a line too"

    def test_the_probe_targets_the_port_the_sdk_is_given(self):
        """A probe of one address followed by a connect to another would
        pass every test above and still hang in the field."""
        seen = {}

        def _probe(host, port, timeout=0.5):
            seen["probe"] = (host, port)
            return True

        import unittest.mock as m
        with m.patch.object(tv, "_opend_listening", _probe), \
             m.patch.object(tv, "OpenQuoteContext",
                            lambda **kw: seen.setdefault("ctx", (kw["host"],
                                                                 kw["port"]))):
            _ensure(FakeViewer())
        assert seen["probe"] == seen["ctx"]
