"""load_day_live: today's tape, read once and then only where it grew.

Today is the one day that cannot be cached -- it is still being written --
so it was re-read in full every refresh cycle, 2.5-8.7s measured against
250ms for the entire SMC recomputation that runs beside it.

The read restarts at the start of the bucket holding the last tick seen
and REPLACES those buckets. The alternative, a `ts >` cursor, looks
simpler and is wrong: ticks share timestamps, so it drops the rest of
that second, and `ts >=` counts it twice. Those two are what most of
these tests are about.
"""
import pathlib
import sys
from datetime import datetime, timedelta

import pytest

ROOT = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import analysis.trade_viewer_qt as tv  # noqa: E402

from . import _tick_ref as ref  # noqa: E402

CODE, TF, DAY = "US.SOXL", "1m", "2026-10-23"
BASE = datetime(2026, 10, 23, 10, 0, 0)


def tick(sec, price=100.0, vol=100, direction="BUY"):
    return {"ts": BASE + timedelta(seconds=sec), "price": price,
            "volume": vol, "direction": direction}


@pytest.fixture
def tape(monkeypatch):
    """A growing tape plus a count of how many rows each read pulled.

    The seam moved. The loader asks for price bins now, not rows, because
    the summing happens inside SQLite -- see test_tick_sql_agg.py, which is
    where that change is checked. This stub keeps the tape as rows and
    buckets them with the implementation that used to do it in the viewer,
    so what these tests are about is untouched: which window gets re-read,
    and that whole buckets are replaced rather than added to.
    """
    store = {"rows": [], "reads": []}

    def _q(code, start, end, with_offex, candle_mins, thresholds=None):
        got = [r for r in store["rows"] if start <= r["ts"] < end]
        store["reads"].append(len(got))
        if not got:
            return None
        tm, tl = thresholds if thresholds else ref.ref_thresholds(got)
        return (ref.ref_bucket(got, candle_mins, tm, tl), (tm, tl),
                max(r["ts"] for r in got))

    monkeypatch.setattr(tv, "_query_tick_bins", _q)
    return store


def totals(buckets):
    """buy/sell/neutral summed over every price in every bucket."""
    out = {"buy": 0, "sell": 0, "neutral": 0}
    for pd_ in buckets.values():
        for b in pd_.values():
            for k in out:
                out[k] += b[k]
    return out


class TestIncremental:
    def test_the_second_call_reads_far_less_than_the_first(self, tape):
        tape["rows"] = [tick(i) for i in range(0, 600, 2)]   # 10 minutes
        st = {}
        tv.load_day_live(CODE, DAY, TF, False, st)
        first = tape["reads"][-1]
        tape["rows"].append(tick(600))
        tv.load_day_live(CODE, DAY, TF, False, st)
        second = tape["reads"][-1]
        assert second < first / 5, f"re-read {second} of {first} rows"

    def test_totals_match_a_full_load_exactly(self, tape):
        """The whole point. Replace-by-bucket has to come out identical to
        reading the day from scratch, or the heatmap and CVD drift."""
        tape["rows"] = [tick(i, vol=10 + i) for i in range(0, 300, 3)]
        st = {}
        tv.load_day_live(CODE, DAY, TF, False, st)
        for i in range(300, 500, 3):
            tape["rows"].append(tick(i, vol=10 + i))
        inc = tv.load_day_live(CODE, DAY, TF, False, st)
        full = tv.load_day_live(CODE, DAY, TF, False, {})
        assert totals(inc) == totals(full)

    def test_the_incremental_read_keeps_the_frozen_boundaries(self, tape):
        """Totals cannot see this one, which is why it has its own test.

        S/M/L is cut at the day's own p33/p67, taken at the last full load.
        An incremental read that re-derives them from its own small window
        stratifies the newest prints against each other instead of against
        the day -- the buy/sell totals come out identical and the S/M/L
        bars in the tick profile quietly mean something else from one
        refresh to the next.

        The tape is built so the two readings disagree loudly: thirty
        small prints set the boundaries, then thirty prints two orders of
        magnitude larger arrive. Under the day's boundaries every one of
        them is large; under the re-read window's own they spread across
        all three classes.
        """
        tape["rows"] = [tick(i * 10, vol=i + 1) for i in range(30)]
        st = {}
        tv.load_day_live(CODE, DAY, TF, False, st)

        tape["rows"] += [tick(300 + i * 10, vol=100 + i) for i in range(30)]
        got = tv.load_day_live(CODE, DAY, TF, False, st)

        late = {bk: pd_ for bk, pd_ in got.items()
                if bk >= BASE + timedelta(minutes=5)}
        assert late, "the new prints never arrived"
        big = sum(b["buy_l"] for pd_ in late.values() for b in pd_.values())
        small = sum(b["buy_s"] + b["buy_m"]
                    for pd_ in late.values() for b in pd_.values())
        assert small == 0, "the newest prints were re-stratified"
        assert big == sum(100 + i for i in range(30))

    def test_ticks_sharing_the_boundary_second_are_not_doubled(self, tape):
        """Three prints on the same second as the last one seen. A `ts >=`
        cursor counts the first batch twice; a `ts >` cursor loses the
        second. Whole-bucket replace has to do neither."""
        tape["rows"] = [tick(90, vol=100) for _ in range(3)]
        st = {}
        tv.load_day_live(CODE, DAY, TF, False, st)
        tape["rows"] += [tick(90, vol=100) for _ in range(3)]
        got = tv.load_day_live(CODE, DAY, TF, False, st)
        assert totals(got)["buy"] == 600

    def test_a_tick_arriving_in_the_same_bucket_is_picked_up(self, tape):
        """The newest bucket is still forming; a cursor that skipped it
        would freeze the current minute until the next one opened."""
        tape["rows"] = [tick(10, vol=100)]
        st = {}
        tv.load_day_live(CODE, DAY, TF, False, st)
        tape["rows"].append(tick(20, vol=50))
        assert totals(tv.load_day_live(CODE, DAY, TF, False, st))["buy"] == 150

    def test_it_does_not_mutate_the_dict_it_handed_out(self, tape):
        """Whatever drew the last frame may still hold that dict."""
        tape["rows"] = [tick(10)]
        st = {}
        first = tv.load_day_live(CODE, DAY, TF, False, st)
        snapshot = {k: len(v) for k, v in first.items()}
        tape["rows"].append(tick(400, price=101.0))
        tv.load_day_live(CODE, DAY, TF, False, st)
        assert {k: len(v) for k, v in first.items()} == snapshot
        assert len(first) < len(st["buckets"])


class TestFullReloads:
    def test_a_different_day_or_code_starts_over(self, tape):
        tape["rows"] = [tick(i) for i in range(10)]
        st = {}
        tv.load_day_live(CODE, DAY, TF, False, st)
        tv.load_day_live("US.SOXS", DAY, TF, False, st)
        assert st["key"][0] == "US.SOXS"
        assert tape["reads"][-1] == len(tape["rows"]), "not a full read"

    def test_growth_triggers_a_fresh_threshold(self, tape):
        """Thresholds are frozen between full loads, so the day must not be
        allowed to grow without bound under the ones it started with."""
        tape["rows"] = [tick(i * 60, vol=10) for i in range(4)]   # 4 buckets
        st = {}
        tv.load_day_live(CODE, DAY, TF, False, st)
        assert st["n_at_thresh"] == 4
        for i in range(4, 9):                                     # now 9
            tape["rows"].append(tick(i * 60, vol=10_000))
        # The growth test reads the state, which only learns about the new
        # buckets once an incremental read has merged them -- so the reload
        # lands on the following call, not this one. One refresh of lag on
        # a threshold that is already approximate is not worth a second
        # read inside the same cycle.
        tv.load_day_live(CODE, DAY, TF, False, st)
        assert st["n_at_thresh"] == 4, "reloaded before it could see growth"
        tv.load_day_live(CODE, DAY, TF, False, st)
        assert tape["reads"][-1] == len(tape["rows"]), "expected a full read"
        assert st["n_at_thresh"] == 9

    def test_an_empty_day_clears_the_state(self, tape):
        st = {"key": "stale", "buckets": {}, "last_ts": BASE}
        assert tv.load_day_live(CODE, DAY, TF, False, st) is None
        assert st == {}

    def test_offex_is_part_of_the_identity(self, tape):
        """A day loaded without the side file has its off_* at zero and must
        not be served to a caller that asked for them."""
        tape["rows"] = [tick(i) for i in range(10)]
        st = {}
        tv.load_day_live(CODE, DAY, TF, False, st)
        tv.load_day_live(CODE, DAY, TF, True, st)
        assert tape["reads"][-1] == len(tape["rows"]), "not a full read"


class TestWiring:
    def test_load_tick_days_uses_it_only_for_the_newest_day(self, tape):
        """Past days go through the plain loader and the day cache; only
        the newest one, which is still growing, reads incrementally."""
        seen = []

        def _plain(code, d, tf, with_offex=False):
            seen.append(d)
            return {}

        import unittest.mock as m
        with m.patch.object(tv, "load_local_ticks", _plain):
            tape["rows"] = [tick(i) for i in range(10)]
            st = {}
            tv.load_tick_days(CODE, DAY, TF, 2, {}, live_state=st)
        assert DAY not in seen, "the newest day went through the full loader"
        assert st.get("key"), "the live state was never populated"

    def test_without_a_state_nothing_changes(self, tape):
        """Historical mode passes no state; the old path has to still work."""
        seen = []

        # Returning a bucket, not {}: load_tick_days counts TRADING days by
        # what came back, so an empty day fills no quota and it walks the
        # whole fallback range looking for one.
        def _plain(code, d, tf, with_offex=False):
            seen.append(d)
            return {BASE: {100.0: dict(tv._new_tick_bin(), buy=1)}}

        import unittest.mock as m
        with m.patch.object(tv, "load_local_ticks", _plain):
            tv.load_tick_days(CODE, DAY, TF, 1, {})
        assert seen == [DAY]
