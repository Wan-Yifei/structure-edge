"""load_tick_days: enough tick history for the window CVD accumulates over.

The viewer loaded exactly one day of ticks. When CVD started spanning the
Range window, every bar older than the current trading day had no tick bucket,
so its delta was 0 and the curve sat flat at zero until 20:00 -- reported as
"CVD before tonight is all 0" after switching to 2D/3D.

Two things are easy to get wrong here and both were:

- Past days are immutable and must be cached. Re-reading five of them measured
  13.3s against 0.2-2.8s for one, on a fetch that runs on a 1s timer.
- The quota is in TRADING days. Counting calendar days misses weekends; counting
  non-empty calendar days is also wrong, because load_local_ticks spans the
  prior 20:00 through 23:59, so a Sunday carries the buckets that open Monday's
  trading day and would retire the quota without adding a session.
"""

from datetime import datetime, timedelta

import pytest

from analysis import trade_viewer_qt as tv

CODE, TF = "US.SOXL", "1m"
END = "2026-09-29"          # a Tuesday; 09-26/27 are the weekend


def _buckets(*hours):
    """Tick buckets at the given datetimes, value shape irrelevant here."""
    return {h: {100.0: {"buy": 1.0, "sell": 0.0, "neutral": 0.0}} for h in hours}


@pytest.fixture()
def fake_disk(monkeypatch):
    """date_str -> buckets, standing in for ticks.db. Records what was read."""
    reads = []
    disk = {}

    def _load(code, date_str, tf, with_offex=False):
        reads.append(date_str)
        return disk.get(date_str)

    monkeypatch.setattr(tv, "load_local_ticks", _load)
    return disk, reads


def _d(day, hour):
    return datetime(2026, 9, day, hour, 0)


class TestTradingDayQuota:
    def test_weekend_does_not_count_as_a_session(self, fake_disk):
        disk, reads = fake_disk
        disk["2026-09-29"] = _buckets(_d(29, 10))
        disk["2026-09-28"] = _buckets(_d(28, 10))
        disk["2026-09-25"] = _buckets(_d(25, 10))
        merged, _ = tv.load_tick_days(CODE, END, TF, 3)
        days = {tv._trading_day_of_bucket(bk) for bk in merged}
        assert days == {datetime(2026, 9, d).date() for d in (25, 28, 29)}

    def test_a_sunday_evening_does_not_retire_the_quota(self, fake_disk):
        """Sunday's file holds the 20:00 buckets that open Monday -- the same
        trading day Monday's file covers, so it must not count as its own."""
        disk, reads = fake_disk
        disk["2026-09-29"] = _buckets(_d(29, 10))
        disk["2026-09-28"] = _buckets(_d(28, 10))
        disk["2026-09-27"] = _buckets(_d(27, 21))      # -> trading day the 28th
        disk["2026-09-25"] = _buckets(_d(25, 10))
        merged, _ = tv.load_tick_days(CODE, END, TF, 3)
        days = {tv._trading_day_of_bucket(bk) for bk in merged}
        assert datetime(2026, 9, 25).date() in days, (
            "the Sunday file ate a slot and the oldest session was skipped")

    def test_one_day_is_just_today(self, fake_disk):
        disk, _ = fake_disk
        for d in (29, 28, 25):
            disk[f"2026-09-{d}"] = _buckets(_d(d, 10))
        merged, _ = tv.load_tick_days(CODE, END, TF, 1)
        assert {tv._trading_day_of_bucket(bk) for bk in merged} == {
            datetime(2026, 9, 29).date()}

    def test_stops_walking_back_when_history_runs_out(self, fake_disk):
        disk, reads = fake_disk
        disk["2026-09-29"] = _buckets(_d(29, 10))
        merged, _ = tv.load_tick_days(CODE, END, TF, 5)
        assert merged, "what exists is still returned"
        assert len(reads) <= 5 * 3 + 4, "the cap must bound the walk"


class TestCaching:
    def test_past_days_are_returned_for_caching(self, fake_disk):
        disk, _ = fake_disk
        for d in (29, 28, 25):
            disk[f"2026-09-{d}"] = _buckets(_d(d, 10))
        _, new = tv.load_tick_days(CODE, END, TF, 3)
        assert sorted(k[2] for k in new) == ["2026-09-25", "2026-09-28"]

    def test_the_newest_day_is_never_cached(self, fake_disk):
        """It is still growing; a cached copy would freeze the live session."""
        disk, _ = fake_disk
        for d in (29, 28):
            disk[f"2026-09-{d}"] = _buckets(_d(d, 10))
        _, new = tv.load_tick_days(CODE, END, TF, 2)
        assert all(k[2] != END for k in new)

    def test_a_warm_cache_skips_every_day_that_had_data(self, fake_disk):
        """Today is always re-read, and the empty weekend days are cheap enough
        to re-read too -- what must not happen is re-reading a real session."""
        disk, reads = fake_disk
        for d in (29, 28, 25):
            disk[f"2026-09-{d}"] = _buckets(_d(d, 10))
        _, new = tv.load_tick_days(CODE, END, TF, 3)
        reads.clear()
        tv.load_tick_days(CODE, END, TF, 3, dict(new))
        assert END in reads
        assert "2026-09-28" not in reads
        assert "2026-09-25" not in reads

    def test_a_warm_cache_returns_the_same_buckets(self, fake_disk):
        disk, _ = fake_disk
        for d in (29, 28, 25):
            disk[f"2026-09-{d}"] = _buckets(_d(d, 10))
        cold, new = tv.load_tick_days(CODE, END, TF, 3)
        warm, _ = tv.load_tick_days(CODE, END, TF, 3, dict(new))
        assert set(warm) == set(cold)

    def test_widening_the_range_only_reads_the_new_days(self, fake_disk):
        disk, reads = fake_disk
        for d in (29, 28, 25, 24, 23):
            disk[f"2026-09-{d}"] = _buckets(_d(d, 10))
        _, new = tv.load_tick_days(CODE, END, TF, 2)
        cache = dict(new)
        reads.clear()
        _, new2 = tv.load_tick_days(CODE, END, TF, 4, cache)
        assert END in reads, "today is always re-read"
        assert "2026-09-28" not in reads, "already cached"

    def test_the_cache_key_separates_the_classification(self, fake_disk):
        """A day loaded without the side file has its off_* counts at zero, so
        it must not be served to a caller that asked for them."""
        disk, reads = fake_disk
        for d in (29, 28):
            disk[f"2026-09-{d}"] = _buckets(_d(d, 10))
        _, new = tv.load_tick_days(CODE, END, TF, 2)
        reads.clear()
        tv.load_tick_days(CODE, END, TF, 2, dict(new), with_offex=True)
        assert "2026-09-28" in reads, "with_offex must not reuse the plain cache"

    def test_the_cache_key_separates_code_and_timeframe(self, fake_disk):
        disk, reads = fake_disk
        for d in (29, 28):
            disk[f"2026-09-{d}"] = _buckets(_d(d, 10))
        _, new = tv.load_tick_days(CODE, END, TF, 2)
        reads.clear()
        tv.load_tick_days("US.SOXS", END, TF, 2, dict(new))
        assert "2026-09-28" in reads, "another symbol must not reuse these"


class TestRangeMapping:
    def test_every_range_asks_for_its_own_day_count(self):
        assert tv.RANGE_DAYS == {"1d": 1, "2d": 2, "3d": 3, "7d": 5}

    def test_empty_history_is_not_an_error(self, fake_disk):
        merged, new = tv.load_tick_days(CODE, END, TF, 3)
        assert merged == {} and new == {}
