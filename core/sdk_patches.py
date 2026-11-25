"""Repair an SDK signature mismatch that disables keepalive self-healing.

WHAT GOES WRONG
---------------
`OpenContextBase.close` takes a reason:

    def close(self, reason=CloseReason.Close):

and the keepalive watchdog in the same class calls it with one, because
the reason is what decides whether the connection comes back:

    # open_context_base.py:600, inside on_tick
    if now - self._last_recv_time >= self._conn_alive_timeout:
        logger.warning(f'Long time not recv: ...')
        self.close(reason=CloseReason.KeepAliveFail)

    # open_context_base.py:146, inside close
    if reason is CloseReason.Close:
        self._auto_reconnect = False        # <- only an explicit close

But `OpenQuoteContext` and `OpenTradeContextBase` both OVERRIDE close
without the parameter, each body being nothing but a docstring and a
`super().close()` call that drops it:

    # open_quote_context.py:102
    def close(self):
        <docstring>
        super(OpenQuoteContext, self).close()

So the watchdog raises `TypeError: OpenQuoteContext.close() got an
unexpected keyword argument 'reason'` inside the SDK's own tick thread,
and the connection is never closed, never disconnected, and never
reconnected. The self-heal does not fail -- it never runs.

MEASURED, not inferred
----------------------
In `logs/analysis/scheduler.log`: 140 `Long time not recv` detections and
**136 of them ended in this TypeError**, spanning 2026-06-05 to
2026-11-23, 124 on 2026-11-23 alone. Every "the collector crashed and
restarted itself" story in this project is really "the collector was
restarted by the scheduler's watchdog, because the SDK's own one could
not run".

WHAT THIS DOES
--------------
Gives both overrides the parameter back and forwards it to the base. The
forward goes straight to `OpenContextBase.close`, which is only safe
because each override is a pure super() call -- so that is CHECKED rather
than assumed, by fingerprinting the code object:

    co_names == ('super', '<ClassName>', 'close')  and  co_argcount == 1

If a future SDK release puts real work in one of those bodies the
fingerprint stops matching, this refuses to patch that class, and says so.
Refusing leaves the bug in place, which is the lesser harm: silently
skipping subclass cleanup would be a new bug nobody is looking for.

Call it once, before opening any context, next to quiet_sdk_file_log():

    from core.sdk_patches import patch_close_reason
    patch_close_reason()

Safe to call more than once, safe before the SDK is imported, and never
raises -- a process must not fail to start because a monkeypatch did.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

_PATCHED: set[str] = set()

#: Marker set on a replacement so a second call is a no-op and a test can
#: tell a patched method from the original without reading bytecode.
_MARK = "_moomoo_close_reason_patched"


def _is_pure_super_forward(fn, class_name: str) -> bool:
    """True when `fn` does nothing but call super().close().

    A fingerprint of the code object rather than a source match: source is
    not always available (a .pyc-only install), and a comment or a reflowed
    docstring would break a textual comparison without changing behaviour.
    """
    code = getattr(fn, "__code__", None)
    if code is None:
        return False
    return (code.co_argcount == 1
            and code.co_names == ("super", class_name, "close"))


def _targets():
    """The classes to patch, imported lazily so the SDK stays optional."""
    out = []
    try:
        from moomoo.quote.open_quote_context import OpenQuoteContext
        out.append(OpenQuoteContext)
    except Exception:                                       # noqa: BLE001
        pass
    try:
        from moomoo.trade.open_trade_context import OpenTradeContextBase
        out.append(OpenTradeContextBase)
    except Exception:                                       # noqa: BLE001
        pass
    return out


def _make_close(base_close, CloseReason):
    """Build the replacement, with the base captured in a CLOSURE.

    Not a default argument: that would put `_base` in the public signature,
    where `inspect.signature` shows it and a caller can pass it.
    """
    def close(self, reason=CloseReason.Close):
        """Close, honouring the reason the caller gave.

        The reason is not cosmetic: the base clears _auto_reconnect only
        for CloseReason.Close, so forwarding it is exactly what lets a
        KeepAliveFail come back.
        """
        return base_close(self, reason)
    return close


def patch_close_reason() -> int:
    """Give close() back its `reason` parameter. Returns classes patched.

    Returns rather than logs the count so a caller can report it in its own
    log, and so a test can tell "patched two" from "found nothing to patch"
    without inspecting the classes.
    """
    try:
        from moomoo.common.open_context_base import OpenContextBase
        from moomoo.common.network_manager import CloseReason
    except Exception:                                       # noqa: BLE001
        # SDK absent or unimportable. Nothing to repair, nothing to raise.
        return 0

    base_close = OpenContextBase.__dict__.get("close")
    if base_close is None or getattr(base_close, "__code__", None) is None \
            or base_close.__code__.co_argcount != 2:
        log.warning("moomoo OpenContextBase.close no longer takes a reason "
                    "-- not patching; the upstream signatures may now agree")
        return 0

    patched = 0
    for cls in _targets():
        name = cls.__name__
        current = cls.__dict__.get("close")
        if current is None:
            continue
        if getattr(current, _MARK, False):
            patched += 1                     # already done in this process
            continue
        if current.__code__.co_argcount == 2:
            # Upstream fixed it. Leave it alone and say so once, because
            # this module can then be deleted.
            log.info("moomoo %s.close already accepts a reason -- upstream "
                     "appears fixed, core.sdk_patches can go", name)
            continue
        if not _is_pure_super_forward(current, name):
            log.warning("moomoo %s.close does more than forward to super() "
                        "-- refusing to patch it. Keepalive self-healing "
                        "stays broken for this class, which is better than "
                        "skipping cleanup nobody knows about.", name)
            continue

        _close = _make_close(base_close, CloseReason)
        setattr(_close, _MARK, True)
        _close.__name__ = "close"
        _close.__qualname__ = f"{name}.close"
        cls.close = _close
        _PATCHED.add(name)
        patched += 1

    if patched:
        log.debug("moomoo close(reason=) patched on: %s",
                  ", ".join(sorted(_PATCHED)) or "nothing")
    return patched


def patched_classes() -> frozenset[str]:
    """Names of the classes this process patched."""
    return frozenset(_PATCHED)
