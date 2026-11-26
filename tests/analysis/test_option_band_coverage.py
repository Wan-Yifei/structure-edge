"""The coverage ratio improves when the selection gets worse.

reconcile() divides our stored volume by the feed's volume FOR THE
CONTRACTS WE SUBSCRIBED TO. That is our own selection grading itself, and
it is not merely incomplete -- it moves the wrong way. Subscribe to fewer
contracts and the ones left are the active ones, so the ratio walks toward
100% while real coverage falls. On 2026-11-24 the subscribed set reconciled
at 100.0% of "the feed's WHOLE-SESSION volume" while holding 62.4% of the
candidate band's volume, and the single biggest-volume contract of the day
-- 13% of the band, OI 203, ranked 230 of 780 by |delta| x OI -- was never
subscribed at all.

That gap is about to become the normal case: an account-wide subscription
quota divided between two underlyings is a smaller selection on both. So
the band denominator has to exist BEFORE the division happens, and it has
to come from a candidate list recorded at selection time -- the chain
snapshot it is scored from is not re-fetchable, so a selection not written
down is a selection nobody can ever grade.

These tests are mostly about one property: the two ratios must be able to
move in OPPOSITE directions, because that is the whole reason the second
one was added.
"""
import pytest

from moomoo import RET_OK, RET_ERROR

from analysis.option_tick_collector import reconcile
from feeds.option_tick_store import OptionTickStore

SESSION = "2026-11-25"
UNDER = "US.QQQ"

# Four contracts in the band. The ranking likes A and B; C is the one it
# passed over that carries most of the day's volume -- the shape of the
# 2026-11-24 miss.
A = "US.QQQ261127C600000"
B = "US.QQQ261127P590000"
C = "US.QQQ261231C640000"
D = "US.QQQ261127P500000"

BAND_VOL = {A: 1_000, B: 500, C: 8_000, D: 100}


class FakeSnapshot:
    """get_market_snapshot with per-code volume and update_time."""

    def __init__(self, volumes: dict, upd_date: str = SESSION, fail=False,
                 wrong_day: tuple = ()):
        self.volumes = volumes
        self.upd_date = upd_date
        self.fail = fail
        self.wrong_day = set(wrong_day)
        self.asked: list[list[str]] = []

    def get_market_snapshot(self, codes):
        import pandas as pd
        self.asked.append(list(codes))
        if self.fail:
            return RET_ERROR, "boom"
        return RET_OK, pd.DataFrame([
            {"code": c, "volume": self.volumes.get(c, 0),
             # Carried in the same response as the volume, which is the
             # whole reason recording them costs nothing.
             "option_delta": 0.5 if c.endswith("0000") else 0.25,
             "option_open_interest": 1000 + len(c),
             "update_time": ("2026-01-01 16:00:00" if c in self.wrong_day
                             else f"{self.upd_date} 16:00:00")}
            for c in codes
        ])


@pytest.fixture
def store(tmp_path):
    s = OptionTickStore(tmp_path / "opt.db")
    yield s
    s.close()


def _candidates(store, picked: list[str], order=(A, B, C, D)):
    """Record a band where `order` is the ranking and `picked` made the cut.

    Scores descend with rank so the stored list is a plausible ranking and
    not just an ordering; `picked` is what the quota took.
    """
    store.record_candidates(SESSION, UNDER, [
        {"code": c, "rank": i + 1, "score": 1000.0 - 10 * i,
         "expiry": "2026-11-27", "strike": 600.0 + i, "opt_type": "CALL",
         "picked": c in picked}
        for i, c in enumerate(order)])


def _ticks(store, code, vol, day=SESSION):
    store.insert_ticks([
        {"code": code, "ts": f"{day} 10:00:00.000", "price": 1.0,
         "volume": vol, "direction": "BUY", "ttype": "AUTO_MATCH",
         "seq": abs(hash(code)) % 10**6}])


class TestTheTwoRatiosMoveInOppositeDirections:
    """The finding, as a test. If this ever passes trivially, the band
    denominator has stopped being a different measurement."""

    def test_a_narrower_selection_raises_one_and_lowers_the_other(self, store):
        # Wide: A and C subscribed, and we caught all of both.
        _candidates(store, picked=[A, C])
        _ticks(store, A, 1_000)
        _ticks(store, C, 8_000)
        wide = reconcile(FakeSnapshot(BAND_VOL), [A, C], store, SESSION, UNDER)

        # Narrow: only A subscribed -- the ranking dropped C, which is most
        # of the band's volume. Same day, same band.
        store2_session = SESSION
        store.record_candidates(store2_session, UNDER, [
            {"code": c, "rank": i + 1, "score": 1000.0 - 10 * i,
             "picked": c == A} for i, c in enumerate((A, B, C, D))])
        narrow = reconcile(FakeSnapshot(BAND_VOL), [A], store, SESSION, UNDER)

        assert wide["ratio"] == pytest.approx(1.0)
        assert narrow["ratio"] == pytest.approx(1.0), (
            "the premise: the subscribed-set ratio cannot tell these apart, "
            "which is why it reads 100% either way")
        assert narrow["band"]["ratio"] < wide["band"]["ratio"], (
            f"band coverage did not fall when the selection dropped the "
            f"biggest contract in the band: {narrow['band']['ratio']} vs "
            f"{wide['band']['ratio']}")

    def test_the_band_denominator_is_the_whole_band(self, store):
        _candidates(store, picked=[A])
        _ticks(store, A, 1_000)
        s = reconcile(FakeSnapshot(BAND_VOL), [A], store, SESSION, UNDER)
        assert s["feed_vol"] == 1_000, "the headline denominator changed"
        assert s["band"]["feed_vol"] == sum(BAND_VOL.values())
        assert s["band"]["ratio"] == pytest.approx(1_000 / 9_600)


class TestItNamesWhatTheRankingPassedOver:

    def test_the_biggest_unsubscribed_contract_is_reported(self, store):
        _candidates(store, picked=[A, B])
        _ticks(store, A, 1_000)
        _ticks(store, B, 500)
        s = reconcile(FakeSnapshot(BAND_VOL), [A, B], store, SESSION, UNDER)
        missed = {code: (vol, rank) for code, vol, rank in s["band"]["missed"]}
        assert C in missed, (
            f"the contract carrying 8,000 of the band's 9,600 was not "
            f"reported: {s['band']['missed']}")
        assert missed[C][0] == 8_000
        assert missed[C][1] == 3, "its rank in the ranking is the diagnostic"

    def test_subscribed_contracts_are_not_called_missed(self, store):
        _candidates(store, picked=[A, B, C, D])
        for code, vol in BAND_VOL.items():
            _ticks(store, code, vol)
        s = reconcile(FakeSnapshot(BAND_VOL), list(BAND_VOL), store,
                      SESSION, UNDER)
        assert s["band"]["missed"] == []
        assert s["band"]["ratio"] == pytest.approx(1.0)


class TestItIsAbsentRatherThanWrong:
    """A session collected before the candidate list existed has no band.
    Reporting one anyway -- from the subscribed set, say -- would put a
    number next to the word BAND that is not one."""

    def test_no_recorded_candidates_gives_none(self, store):
        _ticks(store, A, 1_000)
        s = reconcile(FakeSnapshot(BAND_VOL), [A], store, SESSION, UNDER)
        assert s["band"] is None

    def test_another_underlyings_candidates_do_not_count(self, store):
        _candidates(store, picked=[A])
        _ticks(store, A, 1_000)
        s = reconcile(FakeSnapshot(BAND_VOL), [A], store, SESSION, "US.SOXL")
        assert s["band"] is None, (
            "QQQ's band was scored as SOXL's -- the same mixing that printed "
            "QQQ's true 25% as 62.5%")

    def test_an_underlying_that_cannot_be_derived_gives_none(self, store):
        """Passing underlying=None does NOT mean unknown -- reconcile
        derives it from the codes. It is unknown only when the codes are
        not one underlying's, and a band scored against a mixed list is not
        a band."""
        _candidates(store, picked=[A])
        _ticks(store, A, 1_000)
        s = reconcile(FakeSnapshot(BAND_VOL), [A, "US.SOXL261127C40000"],
                      store, SESSION, None)
        assert s["underlying"] is None
        assert s["band"] is None

    def test_a_derived_underlying_still_finds_its_band(self, store):
        """The other half of the same behaviour: one symbol's codes do
        resolve, and the band must not be skipped just because the caller
        left the argument out."""
        _candidates(store, picked=[A])
        _ticks(store, A, 1_000)
        s = reconcile(FakeSnapshot(BAND_VOL), [A], store, SESSION, None)
        assert s["underlying"] == UNDER
        assert s["band"] is not None

    def test_a_failed_snapshot_is_not_a_zero_band(self, store):
        """A refused snapshot means unknown, not "the band traded nothing"
        -- which would read as infinite coverage."""
        _candidates(store, picked=[A])
        _ticks(store, A, 1_000)
        s = reconcile(FakeSnapshot(BAND_VOL, fail=True), [A], store,
                      SESSION, UNDER)
        assert s["band"]["ratio"] is None
        assert s["band"]["unknown"] == 4


class TestTheWrongDayRuleStillApplies:
    """The band pass reuses _feed_volumes, so a contract whose figure is
    stamped another day is excluded from the band denominator too. It was
    the first version of this check's worst bug and must not come back
    through a second code path."""

    def test_a_stale_contract_leaves_the_denominator(self, store):
        _candidates(store, picked=[A])
        _ticks(store, A, 1_000)
        s = reconcile(FakeSnapshot(BAND_VOL, wrong_day=(C,)), [A], store,
                      SESSION, UNDER)
        assert s["band"]["feed_vol"] == sum(
            v for c, v in BAND_VOL.items() if c != C)
        assert s["band"]["unknown"] == 1


class TestTheOutcomeIsKeptNotJustSummed:
    """The snapshot is paid for either way; discarding the per-contract
    figures is what made a different ranking ungradeable. Pairing these rows
    with option_candidates is the whole dataset."""

    def test_every_candidate_gets_a_row(self, store):
        _candidates(store, picked=[A])
        _ticks(store, A, 1_000)
        reconcile(FakeSnapshot(BAND_VOL), [A], store, SESSION, UNDER)
        rows = store.band_volume_on(SESSION)
        assert set(rows) == {A, B, C, D}, (
            "only some candidates were recorded; the ones the ranking passed "
            "over are the ones a replacement ranking has to be scored on")

    def test_a_contract_that_did_not_trade_is_zero_not_missing(self, store):
        """0 and NULL are different facts and the difference is the point:
        folding them together turns "we could not read it" into "nothing
        happened there", which is this project's recorded trap."""
        _candidates(store, picked=[A])
        _ticks(store, A, 1_000)
        quiet = {**BAND_VOL, D: 0}
        reconcile(FakeSnapshot(quiet), [A], store, SESSION, UNDER)
        assert store.band_volume_on(SESSION)[D]["feed_vol"] == 0

    def test_an_unreadable_contract_is_null_not_zero(self, store):
        _candidates(store, picked=[A])
        _ticks(store, A, 1_000)
        reconcile(FakeSnapshot(BAND_VOL, wrong_day=(C,)), [A], store,
                  SESSION, UNDER)
        rows = store.band_volume_on(SESSION)
        assert rows[C]["feed_vol"] is None, (
            "a figure stamped another day was stored as a volume")
        assert rows[D]["feed_vol"] == 100

    def test_delta_and_open_interest_come_along(self, store):
        """Without them no alternative ranking can be scored, and they are
        in the response already."""
        _candidates(store, picked=[A])
        _ticks(store, A, 1_000)
        reconcile(FakeSnapshot(BAND_VOL), [A], store, SESSION, UNDER)
        row = store.band_volume_on(SESSION)[C]
        assert row["delta_close"] is not None
        assert row["oi_close"] is not None

    def test_a_wrong_day_row_keeps_its_delta(self, store):
        """The wrong-day rule invalidates the VOLUME. A delta reading is not
        a claim about any session's trading, and dropping it would make the
        contract unscoreable for no reason."""
        _candidates(store, picked=[A])
        _ticks(store, A, 1_000)
        reconcile(FakeSnapshot(BAND_VOL, wrong_day=(C,)), [A], store,
                  SESSION, UNDER)
        row = store.band_volume_on(SESSION)[C]
        assert row["feed_vol"] is None
        assert row["delta_close"] is not None

    def test_it_costs_no_extra_snapshot(self, store):
        """The premise of the whole change. If recording needed its own
        snapshot pass the band pass would cost twice the API budget."""
        _candidates(store, picked=[A])
        _ticks(store, A, 1_000)
        ctx = FakeSnapshot(BAND_VOL)
        reconcile(ctx, [A], store, SESSION, UNDER)
        # One for the subscribed set, one for the band. Not three.
        assert len(ctx.asked) == 2, [len(a) for a in ctx.asked]


class TestAFailedWriteDoesNotTakeTheReportDown:
    """_log_reconcile catches everything around reconcile() and returns None,
    so an exception from the persistence would lose the coverage report --
    the part somebody actually reads that evening."""

    def test_the_summary_survives(self, store, monkeypatch):
        _candidates(store, picked=[A])
        _ticks(store, A, 1_000)

        def _boom(*a, **kw):
            raise RuntimeError("disk full")

        monkeypatch.setattr(store, "record_band_volume", _boom)
        s = reconcile(FakeSnapshot(BAND_VOL), [A], store, SESSION, UNDER)
        assert s["band"]["ratio"] is not None, (
            "a failed write aborted the band pass")
        assert s["ratio"] is not None


class TestTheDecomposition:
    """One ratio cannot say whether low coverage is a bad selection or a
    dropped subscription, and the two call for opposite responses."""

    def test_a_perfect_subscription_on_a_bad_selection(self, store):
        """We caught everything we asked for; we asked for almost nothing."""
        _candidates(store, picked=[A])
        _ticks(store, A, 1_000)
        s = reconcile(FakeSnapshot(BAND_VOL), [A], store, SESSION, UNDER)
        band = s["band"]
        assert band["ranking_ratio"] == pytest.approx(1_000 / 9_600)
        assert band["ranking_feed_vol"] == 1_000
        # One round, and we subscribed to exactly its picks, so the ranking
        # and the held figure are the same number. They part only on a
        # re-pick, which TestARepickSplitsTheRatios covers.
        assert band["taken_feed_vol"] == band["ranking_feed_vol"]
        # Everything the selection reached, we stored.
        assert band["our_graded"] == band["taken_feed_vol"]
        assert band["delivered"] == pytest.approx(1.0)

    def test_a_good_selection_with_a_dropped_subscription(self, store):
        """The selection had the volume; we only stored part of it."""
        _candidates(store, picked=[A, C])
        _ticks(store, A, 1_000)
        _ticks(store, C, 2_000)                 # of the feed's 8,000
        s = reconcile(FakeSnapshot(BAND_VOL), [A, C], store, SESSION, UNDER)
        band = s["band"]
        assert band["ranking_ratio"] == pytest.approx(9_000 / 9_600)
        assert s["our_vol"] == 3_000
        assert band["ratio"] == pytest.approx(3_000 / 9_600)
        # band = held * collected, an identity now that both sides of the
        # second factor are over the same contracts.
        assert band["ratio"] == pytest.approx(
            band["taken_ratio"] * band["delivered"])


class TestTheBandIsOnlySnapshottedOnce:

    def test_the_candidate_codes_are_asked_for_in_batches_of_200(self, store):
        """3,200 QQQ candidates is 16 snapshot calls; the batching is the
        existing loop's, and this pins that the band pass goes through it
        rather than asking for all of them at once."""
        codes = [f"US.QQQ261127C{i:06d}" for i in range(450)]
        store.record_candidates(SESSION, UNDER, [
            {"code": c, "rank": i + 1, "score": 1.0, "picked": i == 0}
            for i, c in enumerate(codes)])
        _ticks(store, codes[0], 10)
        ctx = FakeSnapshot({c: 1 for c in codes})
        reconcile(ctx, [codes[0]], store, SESSION, UNDER)
        band_calls = [a for a in ctx.asked if len(a) > 1]
        assert band_calls, "the band was never snapshotted"
        assert max(len(a) for a in band_calls) <= 200, (
            f"asked for {max(len(a) for a in band_calls)} codes in one call")


def _round(store, picked, when, order=(A, B, C, D)):
    """One selection round, stamped so the rounds are distinguishable.

    Explicit stamps because datetime.now() does not tick between two calls
    on this machine -- measured, five consecutive calls identical -- so a
    test leaning on the clock to separate rounds would pass by timer
    resolution rather than by the code being right.
    """
    store.record_candidates(SESSION, UNDER, [
        {"code": c, "rank": i + 1, "score": 1000.0 - 10 * i,
         "expiry": "2026-11-27", "strike": 600.0 + i, "opt_type": "CALL",
         "picked": c in picked}
        for i, c in enumerate(order)], recorded=f"{SESSION} {when}")


def _subscribe(store, pairs):
    """Contract rows plus subscription stamps, for the window split.

    Both are needed: subscribed_times_on joins option_contracts to find the
    underlying, exactly as subscribed_on does.
    """
    store.upsert_contracts([
        {"code": c, "underlying": UNDER, "expiry": "2026-11-27",
         "strike": 600.0, "opt_type": "CALL"} for c, _t in pairs])
    for code, when in pairs:
        store.record_subscriptions(SESSION, [code])
        store._con.execute(
            "UPDATE option_subscriptions SET subscribed = ? "
            "WHERE session = ? AND code = ?",
            (f"{SESSION} {when}", SESSION, code))
    store._con.commit()


class TestARepickSplitsTheRatios:
    """The subscribed set and the final round's picks are DIFFERENT sets.

    Every case above makes them equal, which is how 2026-11-25 printed
    "the subscriptions delivered 205.0%": the numerator was our volume over
    the 73 contracts subscribed across two rounds, the denominator the feed
    volume of the 60 the last round picked. A ratio over two different sets
    is not wrong by a little, it is not a ratio.
    """

    def _two_rounds(self, store):
        # Round 1 took A and C. Round 2 dropped C and took A and B. C stays
        # subscribed and keeps collecting -- nothing unsubscribes it until
        # the process that owned it dies.
        _round(store, picked=[A, C], when="09:15:00.000000")
        _round(store, picked=[A, B], when="15:41:00.000000")

    def test_the_delivery_ratio_cannot_exceed_what_we_held(self, store):
        self._two_rounds(store)
        for code in (A, B, C):
            _ticks(store, code, BAND_VOL[code])
        band = reconcile(FakeSnapshot(BAND_VOL), [A, B, C], store, SESSION,
                         UNDER)["band"]
        # 9,500 held and all of it stored. The old arithmetic divided that
        # same 9,500 by the 1,500 the last round picked.
        assert band["taken_feed_vol"] == 9_500
        assert band["our_graded"] == 9_500
        assert band["delivered"] == pytest.approx(1.0)

    def test_the_ranking_figure_stays_narrower_than_what_we_held(self, store):
        self._two_rounds(store)
        for code in (A, B, C):
            _ticks(store, code, BAND_VOL[code])
        band = reconcile(FakeSnapshot(BAND_VOL), [A, B, C], store, SESSION,
                         UNDER)["band"]
        # Both reported. Collapsing them leaves no figure that grades the
        # RANKING, which is the only thing a --pct change moves.
        assert band["ranking_feed_vol"] == 1_500        # A + B, round 2
        assert band["taken_feed_vol"] == 9_500          # A + B + C
        assert band["ranking_ratio"] < band["taken_ratio"]
        assert band["rounds"] == 2

    def test_the_identity_holds(self, store):
        self._two_rounds(store)
        _ticks(store, A, 1_000)
        _ticks(store, C, 4_000)                     # of the feed's 8,000
        band = reconcile(FakeSnapshot(BAND_VOL), [A, B, C], store, SESSION,
                         UNDER)["band"]
        assert band["ratio"] == pytest.approx(
            band["taken_ratio"] * band["delivered"])

    def test_a_contract_we_subscribed_is_never_called_passed_over(self, store):
        """The bug that actually fired on 2026-11-25.

        US.SOXL261127C45000 was subscribed at 09:15 and we captured 99.5%
        of its tape, and the reconcile named it as a contract the ranking
        passed over, because the final round had not re-picked it. This is
        the only post-hoc detector for a dropped subscription, and a false
        positive in it costs more than a wrong ratio.
        """
        self._two_rounds(store)
        _ticks(store, C, 8_000)
        band = reconcile(FakeSnapshot(BAND_VOL), [A, B, C], store, SESSION,
                         UNDER)["band"]
        named = [code for code, _v, _r in band["missed"]]
        assert C not in named, (
            "a contract we subscribed to, and collected, was reported as "
            "one the ranking passed over")
        assert D in named, "the genuinely unsubscribed contract is gone too"

    def test_every_rounds_contracts_get_a_realised_volume_row(self, store):
        """record_band_volume must iterate all rounds, not the last.

        A contract only an earlier round proposed is exactly the one a
        re-pick has something to say about, and those rows are the offline
        scoring dataset.
        """
        _round(store, picked=[A], when="09:15:00.000000",
               order=(A, B, C, D))
        _round(store, picked=[A], when="15:41:00.000000", order=(A, B))
        _ticks(store, A, 1_000)
        reconcile(FakeSnapshot(BAND_VOL), [A], store, SESSION, UNDER)
        rows = store.band_volume_on(SESSION)
        assert set(rows) >= {A, B, C, D}, (
            f"only {sorted(rows)} got a row; the first round's contracts "
            f"lost their realised volume")


class TestSubscribedOutsideTheBandIsCountedNotDropped:
    """candidates_on answers with one round, so an earlier round's pick can
    sit outside it. On 2026-11-25 ten SOXL contracts carrying 4,434 of our
    volume did. Dividing them away while keeping them in the headline is
    how the 205% was built."""

    def _dropped_out_of_band(self, store):
        _round(store, picked=[A, C], when="09:15:00.000000", order=(A, C))
        _round(store, picked=[A], when="15:41:00.000000", order=(A, B))
        _ticks(store, A, 1_000)
        _ticks(store, C, 7_000)
        return reconcile(FakeSnapshot(BAND_VOL), [A, C], store, SESSION,
                         UNDER)["band"]

    def test_they_are_counted_and_named(self, store):
        band = self._dropped_out_of_band(store)
        assert band["outside_n"] == 1
        assert band["outside_our_vol"] == 7_000
        assert band["outside"] == [(C, 7_000)]

    def test_they_are_not_in_the_graded_numerator(self, store):
        band = self._dropped_out_of_band(store)
        # Graded over A alone; C has no row in this band to divide by.
        assert band["graded"] == 1
        assert band["our_graded"] == 1_000
        assert band["delivered"] == pytest.approx(1.0)


class TestTheFeedCanReportLessThanWeStored:
    """Not clamped, not hidden. With the sets consistent, a figure over
    100% is the snapshot disagreeing with the same broker's own tape -- six
    of 230 subscribed contracts on 2026-11-25, 50,075 contracts gross, the
    worst at 177.5%. This WARNING is the only measurement of how often that
    happens, so it has to survive."""

    def test_the_excess_is_named_and_totalled(self, store):
        _candidates(store, picked=[A, B])
        _ticks(store, A, 1_800)                 # the feed says 1,000
        _ticks(store, B, 500)
        band = reconcile(FakeSnapshot(BAND_VOL), [A, B], store, SESSION,
                         UNDER)["band"]
        assert band["over_feed_n"] == 1
        assert band["over_feed_excess"] == 800
        assert band["over_feed"] == [(A, 1_800, 1_000)]
        # And it is NOT clamped away.
        assert band["delivered"] > 1.0

    def test_nothing_is_flagged_when_the_feed_is_ahead(self, store):
        _candidates(store, picked=[A, B])
        _ticks(store, A, 900)
        band = reconcile(FakeSnapshot(BAND_VOL), [A, B], store, SESSION,
                         UNDER)["band"]
        assert band["over_feed_n"] == 0
        assert band["over_feed_excess"] == 0


class TestTheDeliveryRatioIsSplitByListeningWindow:
    """The feed's figure is a WHOLE-session total, so the ratio only means
    "did the subscription deliver" for a contract held all session. On
    2026-11-25 blending the two printed 98.0% over a 104.2% and a 51.9%."""

    def test_a_late_subscription_is_reported_separately(self, store):
        _round(store, picked=[A, C], when="09:15:00.000000")
        _round(store, picked=[A, B], when="15:41:00.000000")
        _subscribe(store, [(A, "09:15:00.000000"), (C, "09:15:00.000000"),
                           (B, "15:41:00.000000")])
        _ticks(store, A, 1_000)
        _ticks(store, B, 125)                   # of the feed's 500
        _ticks(store, C, 8_000)
        win = reconcile(FakeSnapshot(BAND_VOL), [A, B, C], store, SESSION,
                        UNDER)["band"]["window"]
        # Held from the open AND still picked by the final round: A only.
        # C was subscribed at the open but dropped at the re-pick, B came
        # in at the re-pick -- both are partial windows.
        assert win["full"]["n"] == 1
        assert win["full"]["ratio"] == pytest.approx(1.0)
        assert win["partial"]["n"] == 2
        assert win["partial"]["ratio"] == pytest.approx(8_125 / 8_500)
        assert win["approximate"] is False

    def test_three_rounds_say_the_split_is_approximate(self, store):
        """subscribed_times_on keeps only the FIRST subscribe time, so a
        contract dropped by round 2 and re-picked by round 3 reads as held
        all session. Said out loud rather than papered over, because the
        approximation is worst on the messiest days."""
        _round(store, picked=[A], when="09:15:00.000000")
        _round(store, picked=[B], when="12:00:00.000000")
        _round(store, picked=[A, B], when="15:41:00.000000")
        _subscribe(store, [(A, "09:15:00.000000"), (B, "12:00:00.000000"),
                           (C, "15:41:00.000000")])
        _ticks(store, A, 1_000)
        band = reconcile(FakeSnapshot(BAND_VOL), [A, B, C], store, SESSION,
                         UNDER)["band"]
        assert band["window"]["approximate"] is True

    def test_no_subscription_stamps_leaves_the_split_out(self, store):
        """Every other test in this file passes `codes` without recording a
        subscription, which is what a session collected before the stamps
        existed looks like. The split must be absent, not zero."""
        _candidates(store, picked=[A])
        _ticks(store, A, 1_000)
        band = reconcile(FakeSnapshot(BAND_VOL), [A], store, SESSION,
                         UNDER)["band"]
        assert band["window"] is None


class TestTheDetectorsBlindSpotIsCounted:
    """`missed` only sees the band being graded, so that `rank` means one
    thing. An earlier round's band can be wider -- on 2026-11-25 the 09:15
    SOXL chain reached 2027-01-08 and the 15:41 one did not -- and a
    contract only it proposed is invisible to the detector. Counted rather
    than silent, the same rule as `outside`."""

    def _wider_first_round(self, store):
        # Round 1's band reaches C and D; round 2's stops at B.
        _round(store, picked=[A], when="09:15:00.000000", order=(A, B, C, D))
        _round(store, picked=[A], when="15:41:00.000000", order=(A, B))
        _ticks(store, A, 1_000)
        return reconcile(FakeSnapshot(BAND_VOL), [A], store, SESSION,
                         UNDER)["band"]

    def test_the_earlier_rounds_own_contracts_are_counted(self, store):
        band = self._wider_first_round(store)
        # C (8,000) and D (100) are in round 1's band only and were never
        # subscribed, so no "passed over" line can reach them.
        assert band["earlier_band_n"] == 2
        assert band["earlier_band_vol"] == 8_100

    def test_they_are_not_silently_called_passed_over(self, store):
        """They must not leak into `missed` either -- their rank belongs to
        a different scoring than the one the band is graded against."""
        band = self._wider_first_round(store)
        named = [code for code, _v, _r in band["missed"]]
        assert C not in named and D not in named
        assert named == [B], named

    def test_a_single_round_day_has_no_blind_spot(self, store):
        _candidates(store, picked=[A])
        _ticks(store, A, 1_000)
        band = reconcile(FakeSnapshot(BAND_VOL), [A], store, SESSION,
                         UNDER)["band"]
        assert band["earlier_band_n"] == 0
        assert band["earlier_band_vol"] == 0


class TestTheApproximateFlagCountsRoundsNotStamps:
    """Counting distinct subscribe stamps instead fails in the direction
    that matters: a round whose picks were ALL already subscribed adds no
    new stamp, because record_subscriptions is INSERT OR IGNORE. A false
    "exact" is worse than a true "approximate"."""

    def test_a_third_round_that_adds_no_subscription_still_reads_as_three(
            self, store):
        _round(store, picked=[A, B], when="09:15:00.000000")
        _round(store, picked=[A], when="12:00:00.000000")
        # Round 3 re-picks only contracts already subscribed at the open,
        # so option_subscriptions gains nothing and the stamps show ONE
        # distinct value.
        _round(store, picked=[A, B], when="15:41:00.000000")
        _subscribe(store, [(A, "09:15:00.000000"), (B, "09:15:00.000000")])
        _ticks(store, A, 1_000)
        band = reconcile(FakeSnapshot(BAND_VOL), [A, B], store, SESSION,
                         UNDER)["band"]
        assert band["rounds"] == 3
        assert len({v for v in
                    store.subscribed_times_on(SESSION, UNDER).values()}) == 1
        assert band["window"]["approximate"] is True, (
            "a three-round day reported its window split as exact")
