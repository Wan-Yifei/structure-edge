"""The cache must not lose the last day, and must not shrink.

Both failures were silent. `end="2026-10-14"` is midnight, so that session's
intraday bars were sliced off and the caller got a frame that merely looked
complete -- every drill question generated for a range's final day came out
with no delta at all. Separately, a rebuild overwrote the cache file with
just the rebuilt range, so the nightly holdout job (--days 3) cut five months
down to four days and the next caller silently got almost nothing.
"""
import pandas as pd
import pytest

from feeds import delta_bars


def _frame(start, periods):
    idx = pd.date_range(start, periods=periods, freq="1min")
    return pd.DataFrame({"buy": 1.0, "sell": 1.0, "neutral": 0.0,
                         "n": 1, "volume": 2.0}, index=idx)


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(delta_bars, "_CACHE", tmp_path)
    return tmp_path


class TestEndBound:
    def test_a_bare_date_keeps_that_whole_session(self, cache, monkeypatch):
        full = _frame("2026-10-13 00:00", 60 * 24 * 3)
        delta_bars._cache_path("US.X").parent.mkdir(parents=True, exist_ok=True)
        full.to_parquet(delta_bars._cache_path("US.X"))
        monkeypatch.setattr(delta_bars, "build",
                            lambda *a, **k: pytest.fail("should hit cache"))
        got = delta_bars.load("US.X", "2026-10-13", "2026-10-14")
        assert got.index.max() >= pd.Timestamp("2026-10-14 23:00"), \
            "the end date's intraday bars were cut off"

    def test_an_explicit_timestamp_is_respected(self, cache, monkeypatch):
        full = _frame("2026-10-13 00:00", 60 * 24 * 3)
        delta_bars._cache_path("US.X").parent.mkdir(parents=True, exist_ok=True)
        full.to_parquet(delta_bars._cache_path("US.X"))
        monkeypatch.setattr(delta_bars, "build",
                            lambda *a, **k: pytest.fail("should hit cache"))
        got = delta_bars.load("US.X", "2026-10-13",
                              pd.Timestamp("2026-10-14 10:00"))
        assert got.index.max() == pd.Timestamp("2026-10-14 10:00")


class TestCacheIsNotShrunk:
    def test_a_short_request_keeps_the_long_history(self, cache, monkeypatch):
        wide = _frame("2026-05-18 00:00", 60 * 24 * 10)
        delta_bars._cache_path("US.X").parent.mkdir(parents=True, exist_ok=True)
        wide.to_parquet(delta_bars._cache_path("US.X"))
        # Ask for three days past the end -- forces a build of the gap only.
        monkeypatch.setattr(delta_bars, "build",
                            lambda code, s, e: _frame(s, 60 * 24 * 3))
        delta_bars.load("US.X", "2026-05-25", "2026-05-30")
        after = pd.read_parquet(delta_bars._cache_path("US.X"))
        assert after.index.min() == wide.index.min(), \
            "the nightly short-range job must not truncate the history"
        assert len(after) >= len(wide)

    def test_the_returned_slice_is_the_requested_range(self, cache, monkeypatch):
        wide = _frame("2026-05-18 00:00", 60 * 24 * 10)
        delta_bars._cache_path("US.X").parent.mkdir(parents=True, exist_ok=True)
        wide.to_parquet(delta_bars._cache_path("US.X"))
        monkeypatch.setattr(delta_bars, "build",
                            lambda code, s, e: _frame(s, 60 * 24 * 3))
        got = delta_bars.load("US.X", "2026-05-20",
                              pd.Timestamp("2026-05-21 12:00"))
        assert got.index.min() >= pd.Timestamp("2026-05-20")
        assert got.index.max() <= pd.Timestamp("2026-05-21 12:00")


class TestLoadCachedNeverBuilds:
    """The viewer reads this inside a repaint.

    `load` rebuilds from ticks.db whenever the request reaches outside
    the cache, and that table holds 49M rows for SOXL alone -- minutes,
    inside a paint. So this one reads the parquet and says None rather
    than building, and the caller does without.
    """

    def test_a_missing_cache_is_none_not_a_build(self, cache, monkeypatch):
        monkeypatch.setattr(delta_bars, "build",
                            lambda *a, **k: pytest.fail("rebuilt the cache"))
        assert delta_bars.load_cached("US.X", "2026-10-13",
                                      "2026-10-14") is None

    def test_a_range_past_the_cache_is_not_a_build(self, cache, monkeypatch):
        full = _frame("2026-10-13 00:00", 60 * 24)
        delta_bars._cache_path("US.X").parent.mkdir(parents=True,
                                                    exist_ok=True)
        full.to_parquet(delta_bars._cache_path("US.X"))
        monkeypatch.setattr(delta_bars, "build",
                            lambda *a, **k: pytest.fail("rebuilt the cache"))
        got = delta_bars.load_cached("US.X", "2026-10-13", "2026-10-20")
        assert got is not None and len(got), \
            "有一天在缓存里，却什么都没返回"
        assert got.index.max() < pd.Timestamp("2026-10-15"), \
            "返回了缓存里没有的日子"

    def test_a_range_entirely_outside_is_none(self, cache):
        full = _frame("2026-10-13 00:00", 60 * 24)
        delta_bars._cache_path("US.X").parent.mkdir(parents=True,
                                                    exist_ok=True)
        full.to_parquet(delta_bars._cache_path("US.X"))
        assert delta_bars.load_cached("US.X", "2026-09-01",
                                      "2026-09-05") is None
