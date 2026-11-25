"""`offered - inserted` is one number for two opposite things.

A re-pushed print was already recorded and losing it costs nothing. A split
fill arriving as two legs at the same millisecond, price and size is a real
trade that exists nowhere else once the key drops it. Both collide on
UNIQUE(code, ts, price, volume), so SQLite reports them together -- and the
first version of this counter was presented as "dropped by the UNIQUE key",
which is only half true and the wrong half to act on.

The feed separates them even though the key cannot: a replay carries the
SAME seq, a second leg carries a new one. That is the whole mechanism, and
these tests pin it plus the two bounds it comes with.

Why not just stop the replay at the source (is_first_push=False on
re-subscribe): _try_subscribe fires on a no-data timeout, and that first
push is the only evidence the subscription came back. Turning it off would
make a recovery look like a failure -- so the counting is what changes, not
the subscription.
"""
from analysis.tick_collector import (
    _RECENT_SECONDS, _classify_drops, _drop_residual)

TS = "2026-11-20 10:00:00.002"


def _row(seq, volume=100, ts=TS, code="US.SOXL", price=41.72,
         push="REALTIME"):
    return {"code": code, "ts": ts, "price": price, "volume": volume,
            "seq": seq, "push_type": push}


class TestTheTwoCausesAreToldApart:

    def test_a_new_seq_on_a_seen_key_is_a_real_print(self):
        """The split fill: same millisecond, same price, same size, and the
        feed numbered it separately. This is the one that cost a trade."""
        st: dict = {}
        _classify_drops([_row(101), _row(102)], st)
        assert st.get("session_key_victim") == 1
        assert st.get("session_replayed_dropped", 0) == 0

    def test_a_seen_seq_is_a_replay(self):
        st: dict = {}
        _classify_drops([_row(101), _row(101)], st)
        assert st.get("session_replayed_dropped") == 1
        assert st.get("session_key_victim", 0) == 0

    def test_both_in_one_batch(self):
        st: dict = {}
        _classify_drops([_row(101), _row(102), _row(101)], st)
        assert st.get("session_key_victim") == 1
        assert st.get("session_replayed_dropped") == 1

    def test_a_different_size_at_the_same_instant_is_neither(self):
        """Two legs the key can tell apart are both stored, so they are not
        a loss and must not be counted as one. This is the ordinary shape of
        a sweep -- 50 then 100 at one price and millisecond."""
        st: dict = {}
        _classify_drops([_row(101, volume=50), _row(102, volume=100)], st)
        assert st.get("session_key_victim", 0) == 0
        assert st.get("session_replayed_dropped", 0) == 0

    def test_a_different_millisecond_is_neither(self):
        """The real 2026-11-20 10:00:00 case: counter 2 and counter 7 are
        both 41.72 x 50, and both survived only because one landed on .010
        and the other on .011."""
        st: dict = {}
        _classify_drops([_row(2, volume=50, ts="2026-11-20 10:00:00.010"),
                         _row(7, volume=50, ts="2026-11-20 10:00:00.011")],
                        st)
        assert st.get("session_key_victim", 0) == 0


class TestItDoesNotMixUpSymbols:

    def test_one_codes_prints_do_not_collide_with_anothers(self):
        """seq is a snowflake derived from the trade's instant, not a
        per-code counter, so two symbols trading in the same millisecond get
        the SAME number -- measured, QQQ and SOXL shared 33,699 seq values
        over one 15-minute window. Sharing one set would read every one of
        those as a replay."""
        st: dict = {}
        _classify_drops([_row(101, code="US.SOXL"),
                         _row(101, code="US.QQQ")], st)
        assert st.get("session_replayed_dropped", 0) == 0
        assert st.get("session_key_victim", 0) == 0


class TestTheBoundsAreWhatTheyClaim:

    def test_old_seconds_are_pruned(self):
        st: dict = {}
        for s in range(_RECENT_SECONDS + 4):
            _classify_drops([_row(s, ts=f"2026-11-20 10:00:{s:02d}.001")], st)
        assert len(st["_recent"]["US.SOXL"]) <= _RECENT_SECONDS

    def test_a_collision_survives_the_usual_out_of_order_arrival(self):
        """The bound, stated accurately. The key needs an identical
        millisecond, so a twin is in the same SECOND -- but its bucket can
        still age out if three newer print-seconds arrive in between. That
        is not hypothetical (see the next test); it is just rare: measured
        over 2026-11-23 10:00-10:15, rows arriving behind the stored
        maximum were 70/36,843 (QQQ) and 300/76,957 (SOXL), worst case
        8 ms, none later than 3 seconds."""
        st: dict = {}
        for s in range(_RECENT_SECONDS + 4):
            _classify_drops(
                [_row(s * 10, ts=f"2026-11-20 10:00:{s:02d}.001"),
                 _row(s * 10 + 1, ts=f"2026-11-20 10:00:{s:02d}.001")], st)
        assert st.get("session_key_victim") == _RECENT_SECONDS + 4

    def test_a_missing_seq_is_not_read_as_a_replay(self):
        """Nothing before 2026-11-03 has seq, and a NULL must not collide
        with another NULL -- that would report every such print as a
        replay."""
        st: dict = {}
        _classify_drops([_row(None), _row(None)], st)
        assert st.get("session_replayed_dropped", 0) == 0
        # The pair is still a key collision: identical ts, price and size.
        assert st.get("session_key_victim") == 1


class TestItCannotCostTheSession:

    def test_a_row_without_seq_or_with_odd_types_does_not_raise(self):
        st: dict = {}
        _classify_drops([{"code": "US.SOXL", "ts": TS, "price": 41.72,
                          "volume": 0}], st)
        assert "_recent" in st

    def test_an_empty_batch_is_a_no_op(self):
        st: dict = {}
        _classify_drops([], st)
        assert st == {"_recent": {}}


class TestTheFeedsOwnLabelIsUsedFirst:
    """push_data_type has been in the handler's DataFrame all along
    (quote_response_handler.py:194) and was discarded. It beats the seq
    bookkeeping on every axis: per row, no state, no window, no gap across
    a restart, and it names BYDISCONN -- the catch-up after a dropped quote
    connection, up to 50 rows, which the seq window cannot reach past."""

    def test_a_cache_row_is_a_replay_whatever_its_seq(self):
        st: dict = {}
        _classify_drops([_row(101), _row(999, push="CACHE")], st)
        assert st.get("session_replayed_dropped") == 1
        assert st.get("session_key_victim", 0) == 0

    def test_a_bydisconn_row_is_a_replay(self):
        st: dict = {}
        _classify_drops([_row(101), _row(999, push="BYDISCONN")], st)
        assert st.get("session_replayed_dropped") == 1

    def test_an_unlabelled_row_falls_back_to_seq(self):
        """proto2 optional: the label can be absent, and then the seq
        method still has to answer."""
        st: dict = {}
        _classify_drops([_row(101, push="N/A"), _row(101, push="N/A")], st)
        assert st.get("session_replayed_dropped") == 1

    def test_a_missing_label_still_finds_a_key_victim(self):
        st: dict = {}
        _classify_drops([_row(101, push=None), _row(102, push=None)], st)
        assert st.get("session_key_victim") == 1


class TestZeroVolumeIsCountedApart:
    """All zero-volume prints are ODD_LOT and they are 24% of a QQQ
    session. `volume` arrives as a float (the SDK prefers hpVolume) and this
    collector casts it with int(), so a fractional-share fill is already 0
    before the key sees it -- the loss happened at the cast. Every reader of
    this table sums volume, so a zero row contributes exactly 0. Counting
    them as "real prints the key ate" makes that number unactionable."""

    def test_a_zero_volume_collision_is_not_a_lost_trade(self):
        st: dict = {}
        _classify_drops([_row(101, volume=0), _row(102, volume=0)], st)
        assert st.get("session_key_victim", 0) == 0
        assert st.get("session_key_victim_zero") == 1

    def test_a_real_collision_is_still_counted(self):
        st: dict = {}
        _classify_drops([_row(101, volume=0), _row(102, volume=0),
                         _row(103, volume=50), _row(104, volume=50)], st)
        assert st.get("session_key_victim") == 1
        assert st.get("session_key_victim_zero") == 1


class TestTheResidualSubtractsOnlyWhatWasRefused:
    """The residual held at -2 for a whole session and `max(0, ...)` printed
    it as a clean zero. The cause was subtracting repeats SQLite had KEPT:
    two CACHE rows replayed at 2026-11-24 23:52 are in the table, because
    the collector had just restarted and the dead process never stored them.
    A repeat is only lost when the key is already present."""

    def test_it_is_what_the_refused_classes_did_not_claim(self):
        st = {"session_dropped": 10, "session_replayed_dropped": 3,
              "session_key_victim": 2, "session_key_victim_zero": 1}
        assert _drop_residual(st) == 4

    def test_a_repeat_that_was_stored_is_not_subtracted(self):
        st = {"session_dropped": 4, "session_replayed_dropped": 1,
              "session_replayed_stored": 99, "session_key_victim": 1}
        assert _drop_residual(st) == 2

    def test_it_is_allowed_to_go_negative(self):
        """No clamp any more. A negative means a row counted as dropped was
        in fact stored -- a defect to look at, not a number to floor. The
        clamp is exactly what hid this for a day."""
        st = {"session_dropped": 1, "session_replayed_dropped": 5}
        assert _drop_residual(st) == -4

    def test_the_real_session_arithmetic_leaves_exactly_one(self):
        """2026-11-25 13:42, read off the live collector, with the split
        applied.

        EVERY replay that day was stored, not refused: the table holds five
        rows with a non-NULL push_type and the counter recorded five
        re-pushes, so replayed_dropped is 0. Checked against the rows, not
        assumed -- an earlier version of this test filled in 1 and 2 for a
        split nobody had observed and asserted the residual was 0, while its
        own docstring said it closed to one. The residual is +1: a single
        dropped row this process could not attribute, which is what an older
        repeat looks like.
        """
        st = {"session_dropped": 189270, "session_replayed_dropped": 0,
              "session_replayed_stored": 5,
              "session_key_victim": 167905, "session_key_victim_zero": 21364}
        assert _drop_residual(st) == 1


class TestTheCountsAreKeptPerUnderlying:
    """One collector serves several codes and the offline figure it is held
    against is per symbol, so a blended number cannot be compared to it."""

    def test_each_code_gets_its_own_tally(self):
        st: dict = {}
        _classify_drops([_row(1, code="US.SOXL"), _row(2, code="US.SOXL"),
                         _row(1, code="US.QQQ")], st)
        assert st["by_code"]["US.SOXL"]["session_key_victim"] == 1
        assert "US.QQQ" not in st["by_code"] or             st["by_code"]["US.QQQ"].get("session_key_victim", 0) == 0

    def test_the_total_still_adds_up(self):
        st: dict = {}
        _classify_drops([_row(1, code="US.SOXL"), _row(2, code="US.SOXL"),
                         _row(1, code="US.QQQ"), _row(2, code="US.QQQ")], st)
        per = sum(d.get("session_key_victim", 0)
                  for d in st["by_code"].values())
        assert per == st["session_key_victim"] == 2


class TestTheBucketConstantIsNotFreelyAdjustable:

    def test_one_second_would_prune_nothing(self):
        """At _RECENT_SECONDS = 1 the slice becomes [:0]. The constant
        reads as tunable and is not, which is why it carries a note."""
        assert _RECENT_SECONDS >= 2


class TestACollisionAfterThreeNewerSecondsIsMissed:
    """The counter-example behind the reworded bound above. Kept as a test
    rather than a comment so the claim cannot quietly drift back to
    "cannot"."""

    def test_a_very_late_second_leg_is_not_counted(self):
        st: dict = {}
        _classify_drops([_row(1, ts="2026-11-20 10:00:00.123")], st)
        for s in (1, 2, 3):
            _classify_drops([_row(10 + s, ts=f"2026-11-20 10:00:0{s}.001")],
                            st)
        _classify_drops([_row(99, ts="2026-11-20 10:00:00.123")], st)
        assert st.get("session_key_victim", 0) == 0


class TestSeqOutranksTheLabelWhenTheyDisagree:
    """A seq already seen means this exact trade arrived twice: the number
    comes from the trade's own instant, so a genuine second leg cannot reuse
    it. A REALTIME label on a repeated seq therefore means the label is
    wrong, not that a trade was lost.

    Deciding it the other way round -- label first -- made `key_victim`
    count a plain repeat as a lost trade. That count is the number that
    would justify rekeying a 16.6 GB table, so it has to err low.
    """

    def test_a_repeated_seq_labelled_realtime_is_still_a_replay(self):
        st: dict = {}
        _classify_drops([_row(101), _row(101, push="REALTIME")], st)
        assert st.get("session_replayed_dropped") == 1
        assert st.get("session_key_victim", 0) == 0

    def test_a_new_seq_labelled_cache_is_still_a_replay(self):
        """The other direction needs no tie-break: the label is the only
        evidence when seq is new, and a CACHE row is by definition one the
        feed is sending again."""
        st: dict = {}
        _classify_drops([_row(101), _row(102, push="CACHE")], st)
        assert st.get("session_replayed_dropped") == 1
        assert st.get("session_key_victim", 0) == 0


class TestARepeatIsOnlyRefusedWhenTheKeyIsAlreadyThere:
    """The case the whole split exists for, and the one mutation testing
    found unguarded: a repeat whose key is NOT already present is STORED.

    That is not hypothetical. 2026-11-24 23:52, the collector had just
    restarted -- the previous process died at 23:22 -- and the CACHE row the
    subscribe replayed was one nothing had stored, so SQLite kept it. The
    table shows it with push_type='CACHE'. Counting it as refused is what
    held the residual at -2 for a whole session, which max(0, ...) then
    printed as a clean zero.
    """

    def test_a_cache_row_on_a_fresh_bucket_is_stored_not_refused(self):
        st: dict = {}
        _classify_drops([_row(101, push="CACHE")], st)
        assert st.get("session_replayed_stored") == 1
        assert st.get("session_replayed_dropped", 0) == 0

    def test_a_cache_row_on_a_seen_key_is_refused(self):
        st: dict = {}
        _classify_drops([_row(101), _row(102, push="CACHE")], st)
        assert st.get("session_replayed_dropped") == 1
        assert st.get("session_replayed_stored", 0) == 0

    def test_the_two_paths_are_both_reachable_in_one_batch(self):
        """A BYDISCONN catch-up carries both at once: some of its rows are
        already stored, some are not."""
        st: dict = {}
        _classify_drops([_row(101),
                         _row(102, push="BYDISCONN"),
                         _row(103, volume=7, push="BYDISCONN")], st)
        assert st.get("session_replayed_dropped") == 1   # same key as 101
        assert st.get("session_replayed_stored") == 1    # volume 7, unseen

    def test_only_the_refused_one_leaves_the_residual(self):
        st: dict = {}
        _classify_drops([_row(101),
                         _row(102, push="CACHE"),
                         _row(103, volume=7, push="CACHE")], st)
        st["session_dropped"] = 1        # SQLite refused exactly one of them
        assert _drop_residual(st) == 0
