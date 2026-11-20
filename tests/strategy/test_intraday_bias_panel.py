"""The hour panel's two rulers have to agree, and the audit has to bite.

This file exists because of a specific failure. `assign_session` decided
whether a bar belonged to the next session from the RAW bar-end stamp,
while the hour block was cut from the stamp moved back one minute. The
two agree on every stamp of the day except one -- 20:00 -- and that one
stamp turned block 19 into a 24-hour return wearing a one-hour label,
with a mean thirty times too large. Nothing in the output looked wrong.

That is bar-END costing this project something for the third time
(CHANGELOG v0.92.0 was a forward leak from the same cause, and this
study's own first coverage table was the second). So the tests here are
not about hour means at all. They pin the two properties that would have
caught it in a second:

  * no block may span more than an hour of wall clock
  * the audit must REJECT corrupted panels -- the first version of
    reconcile() passed a panel with 500 blocks deleted, with an hour
    duplicated, and with every session label shifted, because it was
    comparing the panel against arithmetic derived from itself

Synthetic bars throughout: no database, no network, and the defect is
reproduced by monkeypatching the old behaviour back in.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategy.intraday_bias import panel as P

DAYS = ("2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09")


@pytest.fixture
def bars() -> pd.DataFrame:
    """Four whole days of 1-minute bars, 00:01 to 24:00, flat volume.

    Stamps are bar ENDS, so a day runs 00:01 .. 24:00 and the final bar
    of the 19:00 block carries the stamp 20:00 -- the one the defect
    turned on.
    """
    idx = []
    for d in DAYS:
        start = pd.Timestamp(d)
        idx.append(pd.date_range(start + pd.Timedelta(minutes=1),
                                 periods=1440, freq="1min"))
    idx = pd.DatetimeIndex(np.concatenate([i.to_numpy() for i in idx]))
    rng = np.random.default_rng(7)
    px = 100 * np.exp(np.cumsum(rng.normal(0, 1e-4, len(idx))))
    return pd.DataFrame({"open": px, "high": px * 1.0001,
                         "low": px * 0.9999, "close": px,
                         "volume": 100.0}, index=idx)


@pytest.fixture
def built(bars):
    days = P.tradeable_days(bars)
    b = P.prepare(bars, days)
    return bars, days, b, P.build_panel(b)


def test_the_fixture_is_the_shape_the_rest_assumes(built):
    """A silent skip here would let every assertion below pass on nothing."""
    _, days, _, pan = built
    assert len(days) == len(DAYS), f"expected {len(DAYS)} sessions, got {len(days)}"
    assert set(pan.hour) == set(range(24))


class TestOneRulerForBothCuts:
    """The hour block and the session must be cut from the same stamp."""

    def test_no_block_spans_more_than_an_hour(self, built):
        _, _, _, pan = built
        span = (pan.t_last - pan.t_first).dt.total_seconds() / 60
        worst = pan.loc[span.idxmax()]
        assert span.max() <= 70, (
            f"block (session {worst.session}, hour {worst.hour}) spans "
            f"{span.max():.0f} minutes")

    def test_the_2000_stamp_stays_with_the_block_it_closes(self, built):
        """20:00 is the last minute of block 19, not the first of anything.

        Sent to the next session instead, it drags that session's block
        19 back a full day. This asserts the stamp's session equals the
        session of 19:01, which is unambiguously in the same block.
        """
        _, _, b, _ = built
        for d in DAYS[:-1]:
            end = b.loc[pd.Timestamp(d) + pd.Timedelta(hours=20)]
            begin = b.loc[pd.Timestamp(d) + pd.Timedelta(hours=19, minutes=1)]
            assert end.hour == begin.hour == 19
            assert end.session == begin.session, (
                f"{d} 20:00 was filed under {end.session} while 19:01 of "
                f"the same block went to {begin.session}")

    def test_the_old_ruler_mismatch_is_caught(self, bars, monkeypatch):
        """Put the defect back and the build must refuse to produce it.

        A regression test for an assertion is worth having only if the
        assertion can be shown to fire, so this reintroduces the exact
        original code path -- the session decided from the raw stamp --
        rather than an invented corruption.
        """
        real = P.assign_session

        def buggy(idx, days):
            return real(idx + pd.Timedelta(minutes=1), days)

        monkeypatch.setattr(P, "assign_session", buggy)
        days = P.tradeable_days(bars)
        with pytest.raises(AssertionError, match="different rulers"):
            P.build_panel(P.prepare(bars, days))

    def test_a_night_before_a_closed_day_is_dropped_not_merged(self, bars):
        """Removing a session must not make its night join another one.

        With "next trading day", a holiday puts TWO nights into one
        session's hour 20: the night before the holiday and the night
        after it, 70-odd hours apart. Here the Wednesday is removed from
        the calendar. Wednesday night still belongs to Thursday -- it is
        the night immediately before it -- but TUESDAY night now precedes
        a day that does not trade, and must be dropped rather than
        travelling on to Thursday.
        """
        days = P.tradeable_days(bars).drop(pd.Timestamp(DAYS[2]))
        pan = P.build_panel(P.prepare(bars, days))
        span = (pan.t_last - pan.t_first).dt.total_seconds() / 60
        assert span.max() <= 70

        thu20 = pan[(pan.session == pd.Timestamp(DAYS[3]))
                    & (pan.hour == 20)]
        assert len(thu20) == 1
        assert thu20.t_first.iloc[0].normalize() == pd.Timestamp(DAYS[2]), (
            "Thursday's hour 20 should be Wednesday night, not "
            f"{thu20.t_first.iloc[0]}")

        tue_night = pan[(pan.t_first >= pd.Timestamp(DAYS[1])
                         + pd.Timedelta(hours=20))
                        & (pan.t_first < pd.Timestamp(DAYS[2]))]
        assert tue_night.empty, (
            "the night before a day that does not trade is not the "
            f"overnight leg of anything, but {len(tue_night)} blocks of "
            f"it survived as {sorted(set(tue_night.session))}")


class TestTheAuditBites:
    """Every one of these corruptions passed the first implementation."""

    def test_a_clean_panel_passes(self, built):
        _, days, b, pan = built
        rec = P.reconcile(b, pan, days)
        assert rec["cells"] == len(pan)

    def test_deleted_blocks_are_caught(self, built):
        _, days, b, pan = built
        rng = np.random.default_rng(0)
        cut = pan.drop(index=rng.choice(pan.index, 5, replace=False))
        with pytest.raises(AssertionError, match="cells in the panel"):
            P.reconcile(b, cut, days)

    def test_duplicated_blocks_are_caught(self, built):
        _, days, b, pan = built
        dup = pd.concat([pan, pan[pan.hour == 11]]).sort_values("t_first")
        with pytest.raises(AssertionError, match="cells in the panel"):
            P.reconcile(b, dup, days)

    def test_shifted_session_labels_are_caught(self, built):
        """Shifting the labels changes no return and no total, so only a
        check that re-derives the session from the timestamp sees it."""
        _, days, b, pan = built
        bad = pan.copy()
        bad["session"] = bad.session.shift(24).bfill()
        with pytest.raises(AssertionError, match="session their own first"):
            P.reconcile(b, bad, days)


class TestZeroSum:
    """alpha sums to zero every day, which is why a positive alpha is
    never on its own a finding."""

    def test_alpha_sums_to_zero_on_every_day(self, built):
        _, _, _, pan = built
        w = P.wide(pan, tuple(range(9, 16)))
        a = P.alpha(w)
        assert np.abs(a.sum(axis=1).to_numpy()).max() < 1e-12

    def test_winsorising_moves_nothing_it_should_not(self, built):
        _, _, _, pan = built
        w = P.wide(pan, tuple(range(9, 16)))
        v = P.winsorise(w)
        moved = (v != w).sum().sum()
        assert 0 < moved <= 2 * np.ceil(P.WINSOR_FRAC * len(w)) * w.shape[1]
