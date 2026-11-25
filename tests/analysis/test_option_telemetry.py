"""Telemetry must read keys query_subscription actually returns.

There was no test here at all, and that is how a read of a key the SDK does
not return survived: `conn_sub_list` is NOT in the result. The SDK builds a
fresh dict of eight keys and only ITERATES conn_sub_list to derive
`own_used` (open_quote_context.py:1199-1228), so

    conns = sub.get("conn_sub_list") or []

was always empty and the line printed `conns=0(ours 0, others 0)` on every
session regardless of the truth. A constant shaped like a measurement is
worse than no measurement: the figure that would have said "another process
took the quota" said "nobody did", every time.

The fixtures here are built from the SDK's own result shape, not from what
the caller wishes it returned.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

from analysis.option_tick_collector import log_telemetry

#: Exactly what OpenQuoteContext.query_subscription puts in its result.
SDK_KEYS = {"total_used", "remain", "option_used_quota", "option_remain_quota",
            "own_used", "own_security_firm", "own_option_used_quota",
            "sub_list"}


def _sub(total_used=200, remain=300, own_used=200, option_used=4,
         option_remain=0, codes=("US.SOXL261127C43000",)):
    return {
        "total_used": total_used,
        "remain": remain,
        "option_used_quota": option_used,
        "option_remain_quota": option_remain,
        "own_used": own_used,
        "own_security_firm": 0,
        "own_option_used_quota": option_used,
        "sub_list": {"TICKER": list(codes)},
    }


class _Ctx:
    def __init__(self, sub):
        self._sub = sub

    def query_subscription(self):
        from moomoo import RET_OK
        return RET_OK, self._sub


def _levels(caplog) -> set[str]:
    """The LEVELS that were emitted, for "this must not warn" assertions.

    caplog's messages do not contain the level name, so asserting
    `"WARNING" not in message` is true of every possible implementation.
    """
    return {r.levelname for r in caplog.records}


def _run(sub, codes, caplog):
    with caplog.at_level("INFO"):
        log_telemetry(_Ctx(sub), list(codes), {}, {})
    return [r.getMessage() for r in caplog.records]


class TestTheResultShapeIsWhatTheSdkReturns:
    """Read off the SDK, so a future version that adds or drops a key
    makes this fail rather than silently changing what gets logged."""

    def test_query_subscription_does_not_return_conn_sub_list(self):
        src = pathlib.Path(
            "./.venv/Lib/site-packages/moomoo/quote/open_quote_context.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(src)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef)
                  and n.name == "query_subscription")
        # Only assignments INTO ret_dict count. Walking every subscript in
        # the function finds conn_sub_list too -- it is read from
        # sub_table -- which is exactly the mistake that let this through.
        assigned = {
            t.slice.value
            for node in ast.walk(fn) if isinstance(node, ast.Assign)
            for t in node.targets
            if isinstance(t, ast.Subscript)
            and isinstance(t.value, ast.Name) and t.value.id == "ret_dict"
            and isinstance(t.slice, ast.Constant)
        }
        assert assigned == SDK_KEYS, assigned
        assert "conn_sub_list" not in assigned


class TestItReportsWhoHoldsTheQuota:

    def test_others_is_the_difference_not_a_dead_field(self, caplog):
        """450 of 500 used, 200 of them ours: 250 belong to someone else."""
        msgs = _run(_sub(total_used=450, remain=50, own_used=200),
                    ["US.SOXL261127C43000"], caplog)
        line = next(m for m in msgs if m.startswith("telemetry: subscribed"))
        assert "ours=200" in line, line
        assert "others=250" in line, line
        assert "total_used=450" in line and "remain=50" in line, line

    def test_a_positive_difference_does_NOT_warn(self, caplog):
        """Because it is the normal state, not a finding.

        The tick and order-book collectors connect to the same OpenD
        (127.0.0.1:11111) and their subscriptions count toward total_used
        but not toward this connection's own_used, so `theirs` is non-zero
        in every ordinary session -- and the two option collectors each
        count the other's 200 contracts. Warning on it fired roughly 28
        times a day saying nothing, in the one subsystem whose log the
        first live session tells the operator to read. Replacing a constant
        zero dressed as a measurement with a constant alarm is not a fix.
        """
        msgs = _run(_sub(total_used=450, own_used=200),
                    ["US.SOXL261127C43000"], caplog)
        # On the RECORDS, not on the text: getMessage() never contains the
        # level name, so `"WARNING" in m` would have held for any code at
        # all -- including the version this test exists to rule out.
        assert _levels(caplog) == {"INFO"}, _levels(caplog)
        line = next(m for m in msgs if m.startswith("telemetry: subscribed"))
        assert "others=250" in line, "the figure still has to be reported"

    def test_a_negative_difference_is_called_out_as_contradictory(self,
                                                                  caplog):
        """own_used > total_used cannot happen inside one response.

        It is not a quota reading at that point, it is a broken one -- and
        the first version printed it as "-150 subscription slots are held
        by other connections", a sentence that cannot be true.
        """
        msgs = _run(_sub(total_used=50, own_used=200),
                    ["US.SOXL261127C43000"], caplog)
        assert any("cannot happen in one response" in m for m in msgs), msgs

    def test_no_contradiction_warning_when_every_slot_is_ours(self, caplog):
        msgs = _run(_sub(total_used=200, own_used=200),
                    ["US.SOXL261127C43000"], caplog)
        assert not any("cannot happen" in m for m in msgs), msgs
        line = next(m for m in msgs if m.startswith("telemetry: subscribed"))
        assert "others=0" in line, line

    def test_a_missing_key_reads_as_unknown_not_as_zero(self, caplog):
        """An older OpenD, or a shape change. "?" is a different claim from
        "nobody else is holding anything"."""
        sub = _sub()
        del sub["own_used"]
        msgs = _run(sub, ["US.SOXL261127C43000"], caplog)
        line = next(m for m in msgs if m.startswith("telemetry: subscribed"))
        assert "others=?" in line, line
        assert not any("held by other connections" in m for m in msgs), msgs


class TestItStillReportsTheSubscriptionShortfall:
    """The part that already worked, kept honest while the quota half
    changed around it."""

    def test_it_counts_how_many_requested_codes_are_live(self, caplog):
        msgs = _run(_sub(codes=["A", "B"]), ["A", "B", "C"], caplog)
        line = next(m for m in msgs if m.startswith("telemetry: subscribed"))
        assert "subscribed 2 of 3 requested" in line, line

    def test_it_warns_about_the_missing_ones(self, caplog):
        msgs = _run(_sub(codes=["A"]), ["A", "B", "C"], caplog)
        assert any("2 of 3 requested contracts are NOT in the subscription "
                   "list" in m for m in msgs), msgs

    def test_no_shortfall_warning_when_all_are_live(self, caplog):
        msgs = _run(_sub(codes=["A", "B"]), ["A", "B"], caplog)
        assert not any("are NOT in the subscription list" in m for m in msgs)

    def test_an_EMPTY_subscription_list_is_the_loudest_case_not_a_question(
            self, caplog):
        """Every contract gone is what this line exists to catch.

        `live = len(...) if tick_subs else None` printed "subscribed ? of
        200" for exactly that, and because live was None the shortfall
        warning below it did not fire at all -- the 210->60 collapse the
        whole telemetry line was added for would have gone out as a
        question mark. A read that failed raises and is caught separately;
        an empty list is an answer.
        """
        msgs = _run(_sub(codes=[]), ["A", "B", "C"], caplog)
        line = next(m for m in msgs if m.startswith("telemetry: subscribed"))
        assert "subscribed 0 of 3 requested" in line, line
        assert any("3 of 3 requested contracts are NOT in the subscription "
                   "list" in m for m in msgs), msgs


class TestAFailedReadIsNotAZeroReading:

    def test_an_exception_is_reported_rather_than_swallowed(self, caplog):
        class _Broken:
            def query_subscription(self):
                raise RuntimeError("socket gone")

        with caplog.at_level("WARNING"):
            log_telemetry(_Broken(), ["A"], {}, {})
        assert any("subscription read failed" in r.getMessage()
                   for r in caplog.records), caplog.text
