"""The tape is bucketed inside SQLite now, and must come out unchanged.

What changed is where the summing happens, not what it sums. One SOXL day
is 1.33M prints; building a dict, a string and a datetime for each of them
cost ~861 MB of resident memory that existed only to be added up and freed
-- and load_day_live re-reads the whole day about ten times a session, so
that peak kept coming back. Two GROUP BYs produce the same price bins
without the rows ever reaching Python.

"The same price bins" is the whole specification, so almost every test
here drives the old implementation (tests/analysis/_tick_ref.py, the code
that was deleted) and the new one over the same database and compares
every cell of every bin. An assertion hand-written about what the old code
did would be a restatement of it; this is the thing itself.

Everything runs against a database built in tmp_path. Nothing here opens
db/ticks.db.
"""
from __future__ import annotations

import pathlib
import random
import sqlite3
import sys
from datetime import datetime, timedelta

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import analysis.trade_viewer_qt as tv          # noqa: E402
from core.time_utils import candle_start       # noqa: E402

from . import _tick_ref as ref                 # noqa: E402

CODE = "US.SOXL"
DAY  = "2026-10-23"
BASE = datetime(2026, 10, 22, 20, 0, 0)        # the window's first second
ALL_TF = ["1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"]


@pytest.fixture
def at(tmp_path, monkeypatch):
    """Point the loader at a database this test owns."""
    def _build(ticks, mids=()):
        db = ref.build_db(tmp_path, ticks, mids)
        monkeypatch.setattr(tv, "_tick_db_path", lambda: db)
        return db
    return _build


def window():
    return tv._day_window(DAY)


def both(db, tf, with_offex=True):
    """(new, reference) for the whole window, or (None, None)."""
    _, cm = tv.TIMEFRAME_MAP[tf]
    start, end = window()
    got = tv._query_tick_bins(CODE, start, end, with_offex, cm)
    exp = ref.ref_load(db, CODE, start, end, with_offex, cm)
    return got, exp


def tape(n, seed=0, code=CODE, span_hours=27):
    """A deterministic pseudo-tape across the whole 28-hour window."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        ts = BASE + timedelta(seconds=rng.uniform(0, span_hours * 3600))
        ts = ts.replace(microsecond=rng.choice([0, 123000, 456789]))
        out.append((code, ts,
                    round(100.0 + rng.randint(0, 40) * 0.01, 2),
                    rng.choice([1, 7, 50, 100, 101, 999, 5000]),
                    rng.choice(["BUY", "SELL", "NEUTRAL"])))
    return out


class TestItMatchesTheOldLoader:
    """The one thing that matters, over every timeframe the viewer offers."""

    @pytest.mark.parametrize("tf", ALL_TF)
    def test_every_cell_of_every_bin(self, at, tf):
        db = at(tape(400, seed=1))
        got, exp = both(db, tf, with_offex=False)
        assert got is not None and exp is not None
        diffs = ref.cells_equal(got[0], exp[0])
        assert diffs == [], diffs[:8]

    @pytest.mark.parametrize("tf", ALL_TF)
    def test_with_the_side_file_too(self, at, tf):
        ticks = tape(400, seed=2)
        mids = [(c, t, p, v, "BUY" if v % 2 else "SELL")
                for c, t, p, v, d in ticks if d == "NEUTRAL"]
        db = at(ticks, mids)
        got, exp = both(db, tf, with_offex=True)
        diffs = ref.cells_equal(got[0], exp[0])
        assert diffs == [], diffs[:8]

    @pytest.mark.parametrize("seed", [3, 4, 5, 6, 7])
    def test_over_several_tapes(self, at, seed):
        db = at(tape(300, seed=seed))
        got, exp = both(db, "5m", with_offex=False)
        assert ref.cells_equal(got[0], exp[0]) == []

    def test_the_thresholds_match(self, at):
        db = at(tape(400, seed=8))
        got, exp = both(db, "1m", with_offex=False)
        assert got[1] == exp[1]

    def test_the_last_timestamp_matches(self, at):
        db = at(tape(400, seed=9))
        got, exp = both(db, "1m", with_offex=False)
        assert got[2] == exp[2]

    def test_a_tape_of_one_print(self, at):
        db = at([(CODE, BASE + timedelta(hours=14), 100.0, 5, "BUY")])
        got, exp = both(db, "1m", with_offex=False)
        assert ref.cells_equal(got[0], exp[0]) == []
        assert got[1] == exp[1] == (float("inf"), float("inf"))


class TestTheBucketExpression:
    """_bucket_sql has to agree with candle_start, not with a retelling of
    its rules -- so the test calls candle_start."""

    @pytest.mark.parametrize("tf", ALL_TF)
    def test_keys_are_what_candle_start_says(self, at, tf):
        _, cm = tv.TIMEFRAME_MAP[tf]
        ticks = tape(200, seed=10)
        db = at(ticks)
        got, _ = both(db, tf, with_offex=False)
        want = {candle_start(t, cm) for _, t, _, _, _ in ticks}
        assert set(got[0]) == want

    @pytest.mark.parametrize("tf", ALL_TF)
    def test_every_print_lands_in_its_own_bucket(self, at, tf):
        """Totals must survive the regrouping: nothing dropped, nothing
        counted twice."""
        _, cm = tv.TIMEFRAME_MAP[tf]
        ticks = tape(300, seed=11)
        db = at(ticks)
        got, _ = both(db, tf, with_offex=False)
        summed = sum(b["buy"] + b["sell"] + b["neutral"]
                     for pd_ in got[0].values() for b in pd_.values())
        assert summed == sum(v for _, _, _, v, _ in ticks)

    def test_the_hour_is_kept_below_an_hour(self, at):
        """A minute-only floor would collapse 09:05 and 10:05 into one."""
        db = at([(CODE, BASE.replace(hour=9) + timedelta(days=1,
                                                         minutes=5), 100.0,
                  1, "BUY"),
                 (CODE, BASE.replace(hour=10) + timedelta(days=1,
                                                          minutes=5), 100.0,
                  1, "BUY")])
        got, _ = both(db, "5m", with_offex=False)
        assert len(got[0]) == 2

    def test_four_hour_buckets_land_on_the_clock(self, at):
        ticks = [(CODE, BASE + timedelta(hours=h, minutes=7), 100.0, 1, "BUY")
                 for h in range(0, 24, 3)]
        db = at(ticks)
        got, _ = both(db, "4h", with_offex=False)
        assert all(bk.hour % 4 == 0 and bk.minute == 0 for bk in got[0])

    def test_a_daily_bucket_is_midnight(self, at):
        db = at(tape(60, seed=12))
        got, _ = both(db, "1d", with_offex=False)
        assert all(bk.hour == 0 and bk.minute == 0 for bk in got[0])


class TestTheBinShape:
    """Two consumers read a bin's shape rather than its values:
    _draw_tick_profile sniffs for "buy_s" to decide whether it can draw the
    S/M/L breakdown, and _bar_flow's .get on off_buy tolerates a live
    bucket that has no such key. A bin from the database must keep all 14."""

    def test_every_bin_has_every_column(self, at):
        db = at(tape(200, seed=13))
        got, _ = both(db, "1m", with_offex=False)
        for pd_ in got[0].values():
            for b in pd_.values():
                assert set(b) == set(tv._BIN_KEYS)

    def test_the_columns_are_in_the_canonical_order(self, at):
        db = at(tape(50, seed=14))
        got, _ = both(db, "1m", with_offex=False)
        b = next(iter(next(iter(got[0].values())).values()))
        assert list(b) == list(tv._new_tick_bin())

    def test_the_counts_are_integers(self, at):
        """Volumes are summed, not averaged; a float here would show up as
        '1234.0' in the profile labels."""
        db = at(tape(50, seed=15))
        got, _ = both(db, "1m", with_offex=False)
        for pd_ in got[0].values():
            for b in pd_.values():
                assert all(isinstance(v, int) for v in b.values())


class TestSizeThresholds:

    def test_under_six_prints_everything_is_small(self, at):
        db = at([(CODE, BASE + timedelta(minutes=i), 100.0, 10 ** i, "BUY")
                 for i in range(5)])
        got, _ = both(db, "1d", with_offex=False)
        assert got[1] == (float("inf"), float("inf"))
        b = next(iter(next(iter(got[0].values())).values()))
        assert b["buy_m"] == 0 and b["buy_l"] == 0
        assert b["buy_s"] == b["buy"]

    def test_zero_volume_prints_do_not_set_the_boundaries(self, at):
        """_size_thresholds filtered them out; the SQL has to as well, or a
        day padded with zero-volume prints gets a p33 of zero and calls
        everything large."""
        ticks = [(CODE, BASE + timedelta(seconds=i), 100.0, 0, "BUY")
                 for i in range(50)]
        ticks += [(CODE, BASE + timedelta(seconds=100 + i), 100.0, i + 1,
                   "BUY") for i in range(10)]
        db = at(ticks)
        got, exp = both(db, "1d", with_offex=False)
        assert got[1] == exp[1]
        assert got[1][0] > 0

    # Volume sets where numpy's two interpolation branches give DIFFERENT
    # floats, found by search rather than by taste: at n=6 the p33 lands
    # at f=0.65 between the second and third value, and the two formulas
    # part company in the last bit. Without a case like this, a test of
    # which branch is used passes on any data at all -- which is how the
    # whole suite passed while the branch was missing.
    @pytest.mark.parametrize("vols", [[1, 1, 14, 15, 15, 15],
                                      [1, 1, 20, 21, 21, 21],
                                      [1, 1, 30, 31, 31, 31]])
    def test_the_boundary_is_the_float_numpy_would_give(self, at, vols):
        """np.percentile's default method switches formula at the halfway
        point so the result is exact at both ends. Using one of the two
        for the whole range is off by an ulp often enough that an == in
        any of these tests would be a matter of luck."""
        at([(CODE, BASE + timedelta(minutes=i), 100.0, v, "BUY")
            for i, v in enumerate(vols)])
        start, end = window()
        got = tv._query_tick_bins(CODE, start, end, False, 1440)

        want = (float(np.percentile(vols, 33)), float(np.percentile(vols, 67)))
        assert got[1] == want

        # ... and this case still tells the two apart.
        h = (len(vols) - 1) * 0.33
        lo = int(h)
        s = sorted(vols)
        naive = s[lo] + (h - lo) * (s[lo + 1] - s[lo])
        assert naive != want[0], "this case no longer reaches the branch"

    def test_all_volumes_equal(self, at):
        db = at([(CODE, BASE + timedelta(seconds=i), 100.0, 42, "BUY")
                 for i in range(20)])
        got, exp = both(db, "1d", with_offex=False)
        assert got[1] == exp[1]
        assert ref.cells_equal(got[0], exp[0]) == []

    def test_a_frozen_threshold_is_used_instead_of_the_window_s_own(self, at):
        """What the incremental read depends on: the second half of a day
        must be stratified on the boundaries the first half set, or every
        refresh silently relabels the bars already drawn."""
        start, end = window()
        mid = start + timedelta(hours=13)
        # Built rather than drawn: a random tape's halves came out with the
        # same p33/p67, which made the comparison below pass on nothing.
        at([(CODE, mid - timedelta(minutes=i + 1), 100.0, i + 1, "BUY")
            for i in range(100)]
           + [(CODE, mid + timedelta(minutes=i), 100.0, 1000 + i, "BUY")
              for i in range(100)])
        whole = tv._query_tick_bins(CODE, start, end, False, 1)
        free = tv._query_tick_bins(CODE, mid, end, False, 1)
        frozen = tv._query_tick_bins(CODE, mid, end, False, 1,
                                     thresholds=whole[1])
        assert frozen[1] == whole[1]
        assert free[1] != whole[1], "the halves happen to agree; weak test"
        assert ref.cells_equal(frozen[0], free[0]) != []

    def test_infinite_frozen_thresholds_still_query(self, at):
        """A day that began under six prints hands (inf, inf) to the next
        read. Infinity has to survive the round trip into SQL."""
        db = at(tape(100, seed=17))
        start, end = window()
        got = tv._query_tick_bins(CODE, start, end, False, 1,
                                  thresholds=(float("inf"), float("inf")))
        for pd_ in got[0].values():
            for b in pd_.values():
                assert b["buy_m"] == b["buy_l"] == 0
                assert b["buy_s"] == b["buy"]


class TestOffExchange:

    def test_the_side_file_is_folded_in(self, at):
        t = BASE + timedelta(hours=14)
        db = at([(CODE, t, 100.0, 60, "NEUTRAL")],
                [(CODE, t, 100.0, 60, "BUY")])
        got, exp = both(db, "1m", with_offex=True)
        b = next(iter(next(iter(got[0].values())).values()))
        assert b["off_buy"] == 60 and b["off_sell"] == 0
        assert ref.cells_equal(got[0], exp[0]) == []

    def test_it_is_left_out_when_not_asked_for(self, at):
        t = BASE + timedelta(hours=14)
        db = at([(CODE, t, 100.0, 60, "NEUTRAL")],
                [(CODE, t, 100.0, 60, "BUY")])
        got, exp = both(db, "1m", with_offex=False)
        b = next(iter(next(iter(got[0].values())).values()))
        assert b["off_buy"] == 0 and b["off_sell"] == 0
        assert ref.cells_equal(got[0], exp[0]) == []

    def test_a_missing_side_file_is_not_an_error(self, at):
        db = at(tape(100, seed=18))
        assert not (db.parent / "tick_mid.db").exists()
        got, exp = both(db, "1m", with_offex=True)
        assert ref.cells_equal(got[0], exp[0]) == []

    def test_a_missing_side_file_is_not_even_mentioned(self, at, capsys):
        """Opening it and letting the error be swallowed gets the same
        numbers, so only the log says which happened -- and on a machine
        that has never run the classifier that line would be printed on
        every load of every day."""
        at(tape(20, seed=25))
        start, end = window()
        assert tv._query_tick_bins(CODE, start, end, True, 1) is not None
        assert capsys.readouterr().out == ""

    def test_a_side_row_on_a_level_with_no_print_is_dropped(self, at):
        """It cannot happen while the classifier is the only writer, but
        the two are aggregated separately now, so say what happens."""
        t = BASE + timedelta(hours=14)
        db = at([(CODE, t, 100.0, 60, "NEUTRAL")],
                [(CODE, t, 999.0, 60, "BUY")])
        got, _ = both(db, "1m", with_offex=True)
        b = next(iter(next(iter(got[0].values())).values()))
        assert b["off_buy"] == 0

    def test_a_side_row_in_a_bucket_with_no_print_is_dropped(self, at):
        """The other half of that: not an unknown price in a known bucket,
        but a bucket the main query never produced. It takes a different
        branch -- the bucket-key cache misses, and the lookup is for a key
        that cannot be in the mapping at all."""
        t = BASE + timedelta(hours=14)
        db = at([(CODE, t, 100.0, 60, "NEUTRAL")],
                [(CODE, t + timedelta(hours=2), 100.0, 60, "BUY")])
        got, exp = both(db, "1m", with_offex=True)
        assert len(got[0]) == 1, "the side row invented a bucket"
        b = next(iter(next(iter(got[0].values())).values()))
        assert b["off_buy"] == 0
        assert ref.cells_equal(got[0], exp[0]) == []

    def test_sides_at_one_level_are_summed(self, at):
        t = BASE + timedelta(hours=14)
        ticks = [(CODE, t + timedelta(seconds=i), 100.0, 10 + i, "NEUTRAL")
                 for i in range(3)]
        db = at(ticks, [(c, ts, p, v, "SELL") for c, ts, p, v, _ in ticks])
        got, exp = both(db, "1m", with_offex=True)
        b = next(iter(next(iter(got[0].values())).values()))
        assert b["off_sell"] == 10 + 11 + 12
        assert ref.cells_equal(got[0], exp[0]) == []


class TestNothingToRead:

    def test_no_database_at_all(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tv, "_tick_db_path",
                            lambda: tmp_path / "nope.db")
        start, end = window()
        assert tv._query_tick_bins(CODE, start, end, True, 1) is None

    def test_an_empty_window(self, at):
        db = at(tape(50, seed=19))
        start, _ = window()
        assert tv._query_tick_bins(CODE, start - timedelta(days=30),
                                   start - timedelta(days=29), True, 1) is None

    def test_another_code_is_not_picked_up(self, at):
        db = at(tape(50, seed=20, code="US.SOXS"))
        start, end = window()
        assert tv._query_tick_bins(CODE, start, end, True, 1) is None

    def test_load_local_ticks_returns_none(self, at):
        at(tape(50, seed=21, code="US.SOXS"))
        assert tv.load_local_ticks(CODE, DAY, "1m") is None


class TestDirectionIsFoldedTheSameWay:
    """Anything that is not BUY or SELL counted as neutral, case-insensitively
    -- the feed has shipped lowercase tags before."""

    @pytest.mark.parametrize("tag", ["NEUTRAL", "neutral", "", "UNKNOWN",
                                     "buy", "Sell"])
    def test_one_tag(self, at, tag):
        t = BASE + timedelta(hours=14)
        db = at([(CODE, t, 100.0, 7, tag)])
        got, exp = both(db, "1m", with_offex=False)
        assert ref.cells_equal(got[0], exp[0]) == []


class TestLoadLocalTicks:

    def test_it_returns_just_the_buckets(self, at):
        db = at(tape(100, seed=22))
        day = tv.load_local_ticks(CODE, DAY, "1m")
        got, _ = both(db, "1m", with_offex=False)
        assert ref.cells_equal(day, got[0]) == []

    def test_a_broken_database_is_swallowed(self, at, tmp_path, monkeypatch):
        at(tape(10, seed=23))
        bad = tmp_path / "db" / "broken.db"
        bad.write_bytes(b"this is not a database")
        monkeypatch.setattr(tv, "_tick_db_path", lambda: bad)
        assert tv.load_local_ticks(CODE, DAY, "1m") is None

    def test_but_it_says_so(self, at, tmp_path, monkeypatch, capsys):
        """Swallowed silently, this is a chart with no tick overlay and
        nothing anywhere saying why -- and the catch now covers the whole
        SQL path, not just a store that would not open."""
        at(tape(10, seed=26))
        bad = tmp_path / "db" / "broken2.db"
        bad.write_bytes(b"this is not a database")
        monkeypatch.setattr(tv, "_tick_db_path", lambda: bad)
        tv.load_local_ticks(CODE, DAY, "1m")
        out = capsys.readouterr().out
        assert CODE in out and DAY in out, out

    def test_a_side_file_that_fails_some_other_way_is_forgiven_too(
            self, at, monkeypatch):
        """The side file is only ever an extra, so nothing it does may cost
        the day. Narrowing this to sqlite3.Error lets anything else escape
        into load_local_ticks' own catch-all, which returns None."""
        db = at(tape(100, seed=27))
        real = tv._ro

        def _boom(path):
            if path.name == "tick_mid.db":
                raise ValueError("something that is not a sqlite3.Error")
            return real(path)

        (db.parent / "tick_mid.db").write_bytes(b"")
        monkeypatch.setattr(tv, "_ro", _boom)
        day = tv.load_local_ticks(CODE, DAY, "1m", with_offex=True)
        assert day, "a side file failure took the whole day down"

    def test_a_broken_side_file_does_not_lose_the_day(self, at, tmp_path):
        """The feed's own tags still stand; off_buy/off_sell stay at zero."""
        db = at(tape(100, seed=24))
        (db.parent / "tick_mid.db").write_bytes(b"not a database either")
        day = tv.load_local_ticks(CODE, DAY, "1m", with_offex=True)
        assert day
        assert all(b["off_buy"] == 0 for pd_ in day.values()
                   for b in pd_.values())


class TestTheGeneratedSql:
    """Cheap structural checks on the two string builders, so a mistake
    shows up as a readable failure rather than a wrong number."""

    def test_every_column_is_selected_once(self):
        assert _agg_parts().__len__() == len(tv._BIN_KEYS)

    def test_the_off_columns_are_zero_literals(self):
        parts = _agg_parts()
        for k, part in zip(tv._BIN_KEYS, parts):
            if k.startswith("off_"):
                assert part.strip() == "0"
            else:
                assert part.startswith("SUM(")

    def test_each_size_class_is_asked_for_exactly_once_per_side(self):
        parts = " | ".join(_agg_parts())
        for k in range(3):
            assert parts.count(f"sz = {k}") == 3

    @pytest.mark.parametrize("cm", [1, 3, 5, 15, 30, 60, 240, 1440])
    def test_the_bucket_expression_compiles(self, cm):
        con = sqlite3.connect(":memory:")
        con.execute("CREATE TABLE ticks (ts TEXT)")
        con.execute("INSERT INTO ticks VALUES ('2026-10-23 09:37:12.5')")
        out = con.execute(
            f"SELECT {tv._bucket_sql(cm)} FROM ticks").fetchone()[0]
        assert out == candle_start(
            datetime(2026, 10, 23, 9, 37, 12, 500000), cm).strftime(
                "%Y-%m-%d %H:%M")


def _agg_parts():
    """_agg_select() split back into one expression per column.

    Splitting on ', ' would cut inside the CASE expressions, so this walks
    the string and only breaks at depth zero.
    """
    s, out, depth, cur = tv._agg_select(), [], 0, []
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur))
            cur = []
            continue
        cur.append(ch)
    out.append("".join(cur))
    return [p.strip() for p in out]
