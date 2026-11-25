"""close(reason=) must reach the base, because the reason IS the self-heal.

`OpenContextBase.close` clears `_auto_reconnect` only when the reason is
`CloseReason.Close`. The keepalive watchdog therefore passes
`CloseReason.KeepAliveFail` so the connection comes back -- and both
`OpenQuoteContext` and `OpenTradeContextBase` override close without the
parameter, so that call raised TypeError inside the SDK's own tick thread.

Measured in logs/analysis/scheduler.log: 140 `Long time not recv`
detections, 136 of them ending in that TypeError, 2026-06-05 to
2026-11-23. The SDK's self-heal has never once run here.

The assertion that matters is therefore NOT "no TypeError". It is
"_auto_reconnect survives a KeepAliveFail and is cleared by an explicit
close" -- a patch that swallowed the reason would satisfy the first and
fail the second.
"""
from __future__ import annotations

import threading

import pytest

from core.sdk_patches import patch_close_reason, patched_classes


@pytest.fixture(autouse=True)
def _patched():
    assert patch_close_reason() == 2, patched_classes()


def _probe(cls):
    """An instance that reaches close() without any socket.

    __init__ is bypassed deliberately: constructing a real context opens a
    connection to OpenD, and these assertions are about which branch close
    takes, not about networking.
    """
    from moomoo.common.constant import ContextStatus

    obj = cls.__new__(cls)
    obj._lock = threading.RLock()
    obj._opend_conn_id = 0
    obj._conn_id = 0
    # CLOSED makes close() return right after the reason branch, so no
    # networking is reached and the only observable is _auto_reconnect.
    obj._status = ContextStatus.CLOSED
    obj._auto_reconnect = True
    return obj


def _classes():
    from moomoo.quote.open_quote_context import OpenQuoteContext
    from moomoo.trade.open_trade_context import OpenTradeContextBase
    return [OpenQuoteContext, OpenTradeContextBase]


class TestTheReasonSurvivesTheOverride:

    @pytest.mark.parametrize("cls", _classes(), ids=lambda c: c.__name__)
    def test_keepalive_failure_leaves_auto_reconnect_on(self, cls):
        from moomoo.common.network_manager import CloseReason
        obj = _probe(cls)
        obj.close(reason=CloseReason.KeepAliveFail)
        assert obj._auto_reconnect is True, (
            "a keepalive failure cleared _auto_reconnect -- the connection "
            "will not come back, which is the bug with extra steps")

    @pytest.mark.parametrize("cls", _classes(), ids=lambda c: c.__name__)
    def test_an_explicit_close_still_turns_auto_reconnect_off(self, cls):
        """The control, and the half a reason-swallowing patch would break.
        Without it, a replacement that ignored `reason` entirely would pass
        the test above."""
        obj = _probe(cls)
        obj.close()
        assert obj._auto_reconnect is False, (
            "ctx.close() must stop the reconnect timer, or a deliberate "
            "shutdown reconnects itself")

    @pytest.mark.parametrize("cls", _classes(), ids=lambda c: c.__name__)
    def test_the_reason_can_be_passed_positionally_too(self, cls):
        from moomoo.common.network_manager import CloseReason
        obj = _probe(cls)
        obj.close(CloseReason.KeepAliveFail)
        assert obj._auto_reconnect is True

    @pytest.mark.parametrize("cls", _classes(), ids=lambda c: c.__name__)
    def test_the_signature_does_not_leak_the_captured_base(self, cls):
        """The base is captured in a closure, not parked in a default
        argument where inspect.signature shows it and a caller can pass it."""
        import inspect
        # Read off the class, so `self` is present -- the point is that
        # nothing FOLLOWS `reason`.
        params = list(inspect.signature(cls.close).parameters)
        assert params == ["self", "reason"], params


class TestItIsSafeToApply:

    def test_it_is_idempotent(self):
        before = cls_ids()
        assert patch_close_reason() == 2
        assert patch_close_reason() == 2
        assert cls_ids() == before, "a second call replaced the replacement"

    def test_it_reports_which_classes_it_patched(self):
        assert patched_classes() == {"OpenQuoteContext", "OpenTradeContextBase"}

    def test_close_keeps_its_name(self):
        for cls in _classes():
            assert cls.close.__name__ == "close"
            assert cls.close.__qualname__.endswith(".close")


def cls_ids():
    return {c.__name__: id(c.__dict__["close"]) for c in _classes()}


class TestTheGuardAgainstAChangedUpstream:
    """The forward goes straight to OpenContextBase.close, which is only
    sound while each override is a pure super() call. That is checked, not
    assumed -- a future SDK that puts real work there must not have its
    cleanup silently skipped."""

    def test_a_body_that_does_more_is_refused(self, monkeypatch, caplog):
        import core.sdk_patches as mod
        from moomoo.quote.open_quote_context import OpenQuoteContext

        def busy_close(self):
            self._did_extra_work = True          # noqa: F841
            super(OpenQuoteContext, self).close()

        monkeypatch.setattr(OpenQuoteContext, "close", busy_close)
        monkeypatch.setattr(mod, "_PATCHED", set())
        with caplog.at_level("WARNING"):
            n = mod.patch_close_reason()
        assert OpenQuoteContext.__dict__["close"] is busy_close, (
            "it patched over a body that does more than forward")
        assert "refusing to patch" in caplog.text
        # The other class is still handled -- one odd body must not stop
        # the repair everywhere.
        assert n == 1, n

    def test_the_fingerprint_accepts_the_real_thing(self):
        import core.sdk_patches as mod
        from moomoo.quote.open_quote_context import OpenQuoteContext as Q

        def pure(self):
            """Same shape as the SDK's own override."""
            super(OpenQuoteContext, self).close()   # noqa: F821

        # Rename so the fingerprint's class-name slot matches.
        assert mod._is_pure_super_forward(pure, "OpenQuoteContext")
        assert not mod._is_pure_super_forward(pure, "SomethingElse")
        assert not mod._is_pure_super_forward(Q.close, "OpenQuoteContext"), (
            "the patched close is not a super() forward and must not read "
            "as one, or a second patch would stack on top of it")
