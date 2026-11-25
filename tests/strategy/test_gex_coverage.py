"""A partial chain fetch must say so, everywhere a magnitude is shown.

OpenD refuses get_option_chain past 10 calls per 30 s. Each refusal used to
print one line to stderr and carry on -- and in the viewer stderr goes
nowhere. Measured on QQQ: dropping 2 of 4 expiries moved net GEX from
-97.0M to -53.9M (44% gone) while the Call/Put Walls stayed on 660/600 and
the status bar still read "updated". Nothing on screen betrayed it.

So coverage travels with the data, and these tests pin that it reaches each
surface that prints a magnitude.
"""
import matplotlib
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt                        # noqa: E402

from moomoo import RET_OK, RET_ERROR                   # noqa: E402

from strategy.option.gex import (                      # noqa: E402
    build_gex_stats, compute_gex, coverage_is_full, coverage_note,
    fetch_option_data, _plot,
)

EXPIRIES = ["2026-11-23", "2026-11-24", "2026-11-25", "2026-11-27"]


class FakeCtx:
    """A chain served by RANGE, the way get_option_chain actually answers.

    The fetch asks for start..end and gets back every expiry inside it, each
    row stamped with its own `strike_time` -- which is why coverage is read
    off the rows and not credited to the range. This fake used to answer one
    expiry per call and refuse by expiry; that model stopped existing when
    the fetch stopped making one call per expiry, and keeping it would have
    tested a shape the API does not have.

    `refuse_ranges` refuses any call whose range contains one of the given
    dates -- a refusal costs the WHOLE range, which is the real failure.
    `omit_expiries` answers normally but leaves those expiries' rows out,
    the other real failure: a range that comes back short.
    """

    def __init__(self, refuse_expiries=(), refuse_batches=(), per_expiry=2,
                 omit_expiries=(), universe=None):
        self.refuse_expiries = set(refuse_expiries)
        self.refuse_batches = set(refuse_batches)
        self.omit_expiries = set(omit_expiries)
        self.universe = list(universe) if universe else list(EXPIRIES)
        self.per_expiry = per_expiry        # raise above 200 to force batching
        self._batch_i = -1
        self.calls: list[tuple] = []

    def get_option_chain(self, code, start=None, end=None):
        self.calls.append((start, end))
        covered = [e for e in self.universe if start <= e <= end]
        if self.refuse_expiries & set(covered):
            return RET_ERROR, "rate limited"
        half = self.per_expiry // 2
        rows = []
        for e in covered:
            if e in self.omit_expiries:
                continue
            rows += [{"code": f"{code}{e}C{i}", "strike_time": e}
                     for i in range(half)]
            rows += [{"code": f"{code}{e}P{i}", "strike_time": e}
                     for i in range(half)]
        return RET_OK, pd.DataFrame(rows,
                                    columns=["code", "strike_time"])

    def get_market_snapshot(self, codes):
        self._batch_i += 1
        if self._batch_i in self.refuse_batches:
            return RET_ERROR, "snapshot refused"
        rows = []
        for i, c in enumerate(codes):
            rows.append({
                "code": c,
                # codes are "<code><expiry>C<i>" / "...P<i>": the side marker
                # is the last non-digit character
                "option_type": "CALL" if c.rstrip("0123456789")[-1] == "C" else "PUT",
                "option_strike_price": 100.0 + i,
                "option_gamma": 0.02, "option_open_interest": 500,
                "option_delta": 0.5, "option_implied_volatility": 30.0,
                "strike_time": "2026-11-27", "volume": 10.0,
            })
        return RET_OK, pd.DataFrame(rows)


def test_coverage_names_the_expiries_that_came_back():
    """The range answers, but two expiries' rows are not in it.

    This is how a partial looks now that one call serves a range: the old
    per-expiry refusal cannot happen any more, because a refused call takes
    every expiry in its range with it (pinned separately below).
    """
    ctx = FakeCtx(omit_expiries=["2026-11-24", "2026-11-25"])
    _, cov = fetch_option_data(ctx, "US.T", EXPIRIES)

    # The list, not just a count: the chart titles itself from it.
    assert cov["expiries_ok"] == ["2026-11-23", "2026-11-27"]
    assert cov["expiries_requested"] == EXPIRIES
    assert not coverage_is_full(cov)
    assert coverage_note(cov) == "PARTIAL: 2/4 expiries"


def test_full_coverage_is_reported_as_full():
    """Guards the opposite failure: a banner that cries wolf every poll."""
    ctx = FakeCtx()
    _, cov = fetch_option_data(ctx, "US.T", EXPIRIES)

    assert cov["expiries_ok"] == EXPIRIES
    assert cov["batches_ok"] == cov["batches_total"] > 0
    assert coverage_is_full(cov)
    assert coverage_note(cov) == ""


def test_a_dropped_snapshot_batch_counts_too():
    """Batches are the other way the sum silently shrinks.

    200 codes per call, so a real QQQ chain spans several; refusing one
    loses its contracts while every expiry still reports OK.
    """
    ctx = FakeCtx(refuse_batches=[1], per_expiry=200)   # 4 x 200 = 4 batches
    _, cov = fetch_option_data(ctx, "US.T", EXPIRIES)

    assert cov["expiries_ok"] == EXPIRIES, "expiries were fine; batches were not"
    assert cov["batches_total"] == 4
    assert cov["batches_ok"] == 3
    assert not coverage_is_full(cov)
    assert coverage_note(cov) == "PARTIAL: 3/4 batches"


def test_stats_are_labelled_with_the_fetched_expiries():
    """The bug this closes: the title named expiries that are not in the sum.

    build_gex_stats takes whatever list it is given, so passing the REQUESTED
    one makes the chart assert that missing contracts are included.
    """
    ctx = FakeCtx(omit_expiries=["2026-11-24", "2026-11-25"])
    df, cov = fetch_option_data(ctx, "US.T", EXPIRIES)
    df, by_strike = compute_gex(df, 100.0)
    stats = build_gex_stats(df, by_strike, 100.0, 1e6, cov["expiries_ok"], cov)

    assert stats["expiries"] == cov["expiries_ok"]
    assert "2026-11-24" not in stats["expiries"]
    assert stats["coverage"] is cov


def test_a_partial_chart_carries_the_warning_in_the_image():
    """A saved PNG outlives the terminal it was printed beside."""
    ctx = FakeCtx(omit_expiries=["2026-11-24"])
    df, cov = fetch_option_data(ctx, "US.T", EXPIRIES)
    df, by_strike = compute_gex(df, 100.0)
    stats = build_gex_stats(df, by_strike, 100.0, 1e6, cov["expiries_ok"], cov)

    _plot("US.T", by_strike, stats, None)
    texts = [t.get_text() for a in plt.gcf().axes for t in a.texts]
    assert any("PARTIAL" in t for t in texts), texts
    plt.close(plt.gcf())


def test_a_complete_chart_carries_no_warning():
    ctx = FakeCtx()
    df, cov = fetch_option_data(ctx, "US.T", EXPIRIES)
    df, by_strike = compute_gex(df, 100.0)
    stats = build_gex_stats(df, by_strike, 100.0, 1e6, cov["expiries_ok"], cov)

    _plot("US.T", by_strike, stats, None)
    texts = [t.get_text() for a in plt.gcf().axes for t in a.texts]
    assert not any("PARTIAL" in t or "⚠" in t for t in texts), texts
    plt.close(plt.gcf())


def test_an_absent_coverage_is_silent_not_alarming():
    """Synthetic callers pass no coverage; they must not get a false alarm,
    or the banner becomes something the eye learns to skip."""
    assert coverage_note(None) == ""


# ── the range fetch itself ───────────────────────────────────────────────────
#
# One call per expiry is what the account-wide budget of 10 get_option_chain
# calls per 30 s could not afford. QQQ lists dailies, so a 7-day DTE was five
# calls per poll and the viewer polls every 15 s -- exactly the limit before
# anything else in the account touches a chain, which is why the PARTIAL
# banner appeared and disappeared without the inputs changing.

LADDER = ["2026-11-23", "2026-11-27", "2026-12-31", "2027-01-15"]


def test_one_call_serves_a_whole_range():
    ctx = FakeCtx()
    _, cov = fetch_option_data(ctx, "US.T", EXPIRIES)
    assert len(ctx.calls) == 1, ctx.calls
    assert ctx.calls[0] == ("2026-11-23", "2026-11-27")
    assert coverage_is_full(cov)


def test_a_ladder_wider_than_the_span_is_split():
    """Not one call for everything either: the API refuses a range past
    about 30 days, so the ladder is chunked by span."""
    ctx = FakeCtx(universe=LADDER)
    _, cov = fetch_option_data(ctx, "US.T", LADDER)
    assert len(ctx.calls) == 2, ctx.calls
    assert cov["expiries_ok"] == LADDER


def test_a_refused_call_costs_its_whole_range():
    """The failure that replaced per-expiry refusal. Refusing the chunk that
    holds 2026-12-31 also loses 2027-01-15, and the banner must say 2/4 --
    not 3/4, which is what crediting the range would have reported."""
    ctx = FakeCtx(universe=LADDER, refuse_expiries=["2026-12-31"])
    _, cov = fetch_option_data(ctx, "US.T", LADDER)
    assert cov["expiries_ok"] == ["2026-11-23", "2026-11-27"]
    assert coverage_note(cov) == "PARTIAL: 2/4 expiries"


def test_an_expiry_outside_the_request_is_not_credited():
    """A range answers for every expiry inside it, including ones nobody
    asked for. Those rows are dropped AND never counted as coverage.

    Both halves are asserted. An earlier version checked only the coverage
    half, and coverage is already protected by intersecting with the
    requested set -- so deleting the row filter altogether left this green
    while two unrequested contracts went into the GEX sum. Mutation testing
    is what found it: the test was named after a behaviour it did not check.
    """
    ctx = FakeCtx(universe=EXPIRIES + ["2026-11-26"])
    df, cov = fetch_option_data(ctx, "US.T", EXPIRIES)

    assert "2026-11-26" not in cov["expiries_ok"]
    assert cov["expiries_ok"] == EXPIRIES
    # The rows themselves: two contracts per expiry, four expiries asked for.
    assert len(df) == 8
    assert not df["code"].str.contains("2026-11-26").any()
    assert cov["rows_outside"] == 2


def test_a_single_chunk_refusal_loses_the_whole_fetch():
    """QQQ at a 7-day DTE is ONE chunk, so a refusal is now all or nothing.

    That is the shape the rate limit actually produced on 2026-11-25 -- the
    viewer log carries OpenD's own words, "every 30 seconds at most 10
    times", against expiries 11-27 through 12-02. Raising is correct: the
    caller clears the walls and the status bar says "no data" rather than
    drawing a magnitude computed from nothing. Before the change the same
    pressure produced a PARTIAL banner over a wall that was quietly 44%
    short, which is worse.
    """
    import pytest

    ctx = FakeCtx(refuse_expiries=[EXPIRIES[0]])
    with pytest.raises(RuntimeError, match="No option chain data"):
        fetch_option_data(ctx, "US.T", EXPIRIES)
    # One call, refused -- not four attempts.
    assert len(ctx.calls) == 1


def test_a_response_without_strike_time_is_not_credited():
    """Then there is nothing to attribute by. The rows still go into the
    sum, but no expiry is credited, so the banner warns. Crediting the range
    on a shape we cannot read is the optimism this banner exists to
    prevent."""
    class _NoStamp(FakeCtx):
        def get_option_chain(self, code, start=None, end=None):
            ret, df = super().get_option_chain(code, start, end)
            return ret, df.drop(columns=["strike_time"])

    df, cov = fetch_option_data(_NoStamp(), "US.T", EXPIRIES)
    assert not df.empty
    assert cov["expiries_ok"] == []
    assert coverage_note(cov).startswith("PARTIAL: 0/4 expiries")
