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
    """Serves a two-contract chain per expiry; refuses the named ones."""

    def __init__(self, refuse_expiries=(), refuse_batches=(), per_expiry=2):
        self.refuse_expiries = set(refuse_expiries)
        self.refuse_batches = set(refuse_batches)
        self.per_expiry = per_expiry        # raise above 200 to force batching
        self._batch_i = -1

    def get_option_chain(self, code, start=None, end=None):
        if start in self.refuse_expiries:
            return RET_ERROR, "rate limited"
        half = self.per_expiry // 2
        codes = ([f"{code}{start}C{i}" for i in range(half)]
                 + [f"{code}{start}P{i}" for i in range(half)])
        return RET_OK, pd.DataFrame({"code": codes})

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
    ctx = FakeCtx(refuse_expiries=["2026-11-24", "2026-11-25"])
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
    ctx = FakeCtx(refuse_expiries=["2026-11-24", "2026-11-25"])
    df, cov = fetch_option_data(ctx, "US.T", EXPIRIES)
    df, by_strike = compute_gex(df, 100.0)
    stats = build_gex_stats(df, by_strike, 100.0, 1e6, cov["expiries_ok"], cov)

    assert stats["expiries"] == cov["expiries_ok"]
    assert "2026-11-24" not in stats["expiries"]
    assert stats["coverage"] is cov


def test_a_partial_chart_carries_the_warning_in_the_image():
    """A saved PNG outlives the terminal it was printed beside."""
    ctx = FakeCtx(refuse_expiries=["2026-11-24"])
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
