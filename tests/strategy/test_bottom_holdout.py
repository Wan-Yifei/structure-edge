"""The bottom-study holdout: append once, never rewrite, never peek.

Its value is entirely in not having been looked at. Two things would destroy
that silently, and both are tested here rather than trusted:

  contamination  a routine "--days 3" sweep walks back into sessions the
                 study already scored. It did exactly that on the first run,
                 adding seven events from 2026-10-12 before anyone noticed.
  rewriting      re-running a day after the selection or features change
                 would quietly restate history to agree with the new code.

The selection and label rules are copied from the study and are checked here
too, because a holdout scored by different rules than the sample is not
comparable with it -- which is the whole point of keeping one.
"""

import numpy as np
import pandas as pd
import pytest

from strategy import bottom_holdout as bh


@pytest.fixture
def cache(tmp_path):
    return tmp_path / "events.parquet"


def frame(rows):
    return pd.DataFrame(rows, columns=bh.COLUMNS)


def event(day="2026-10-13", tm="10:00", lab=1, delta_r=0.2):
    return {"day": day, "tm": tm, "low": 100.0, "close": 100.5, "atr": 1.0,
            "lab": lab, "delta_r": delta_r, "flip": 0.0, "sell_drop": 1.0,
            "signed": 5000.0, "neutral": 4000.0}


class TestTheBoundary:
    def test_the_study_period_is_refused(self, cache, monkeypatch):
        """2026-10-12 and earlier were already scored."""
        monkeypatch.setattr(bh, "build_day",
                            lambda c, d: frame([event(day=str(d))]))
        assert bh.append_day("US.SOXL", "2026-10-12", cache_path=cache) == 0
        assert bh.load("US.SOXL", cache_path=cache).empty

    def test_the_first_holdout_session_is_accepted(self, cache, monkeypatch):
        monkeypatch.setattr(bh, "build_day",
                            lambda c, d: frame([event(day=str(d))]))
        assert bh.append_day("US.SOXL", bh.HOLDOUT_START,
                             cache_path=cache) == 1

    def test_a_sweep_across_the_boundary_takes_only_the_new_side(
            self, cache, monkeypatch):
        monkeypatch.setattr(bh, "build_day",
                            lambda c, d: frame([event(day=str(
                                pd.Timestamp(d).date()))]))
        for day in ("2026-10-11", "2026-10-12", "2026-10-13", "2026-10-14"):
            bh.append_day("US.SOXL", day, cache_path=cache)
        got = sorted(bh.load("US.SOXL", cache_path=cache).day.unique())
        assert got == ["2026-10-13", "2026-10-14"]


class TestAppendOnly:
    def test_a_session_already_written_is_not_rebuilt(self, cache,
                                                      monkeypatch):
        """The guard against a code change restating history."""
        monkeypatch.setattr(bh, "build_day",
                            lambda c, d: frame([event(lab=1, delta_r=0.2)]))
        bh.append_day("US.SOXL", "2026-10-13", cache_path=cache)
        # The rules "change": same day, different answer.
        monkeypatch.setattr(bh, "build_day",
                            lambda c, d: frame([event(lab=0, delta_r=-0.9),
                                                event(tm="11:00", lab=0)]))
        assert bh.append_day("US.SOXL", "2026-10-13", cache_path=cache) == 0
        got = bh.load("US.SOXL", cache_path=cache)
        assert len(got) == 1
        assert got.iloc[0].lab == 1
        assert got.iloc[0].delta_r == pytest.approx(0.2)

    def test_a_later_session_appends_beside_it(self, cache, monkeypatch):
        monkeypatch.setattr(bh, "build_day",
                            lambda c, d: frame([event(day=str(
                                pd.Timestamp(d).date()))]))
        bh.append_day("US.SOXL", "2026-10-13", cache_path=cache)
        bh.append_day("US.SOXL", "2026-10-14", cache_path=cache)
        assert len(bh.load("US.SOXL", cache_path=cache)) == 2

    def test_a_session_with_no_events_is_not_an_error(self, cache,
                                                      monkeypatch):
        monkeypatch.setattr(bh, "build_day", lambda c, d: frame([]))
        assert bh.append_day("US.SOXL", "2026-10-13", cache_path=cache) == 0


class TestItDoesNotPeek:
    def test_coverage_reports_counts_not_outcomes(self, cache, monkeypatch):
        monkeypatch.setattr(bh, "build_day",
                            lambda c, d: frame([event(lab=1),
                                                event(tm="11:00", lab=0)]))
        bh.append_day("US.SOXL", "2026-10-13", cache_path=cache)
        cov = bh.coverage("US.SOXL", cache_path=cache)
        assert list(cov.columns) == ["events"]
        assert cov.loc["2026-10-13", "events"] == 2

    def test_loading_an_empty_holdout_is_empty_not_an_error(self, cache):
        assert bh.load("US.SOXL", cache_path=cache).empty


class TestTheRulesMatchTheStudy:
    def test_the_label_barriers(self):
        assert (bh.UP_ATR, bh.DN_ATR) == (2.0, 1.0)

    def test_the_selection_window(self):
        assert (bh.LOOK, bh.MIN_DROP_ATR, bh.GAP) == (30, 1.0, 15)

    def test_the_forward_horizon(self):
        assert bh.FWD == 30

    def test_the_flow_guards(self):
        assert bh.MIN_SIGNED == 300.0 and bh.MIN_BARS_IN == 25

    def test_regular_hours_only(self):
        assert bh.RTH == (9.5, 16.0)
