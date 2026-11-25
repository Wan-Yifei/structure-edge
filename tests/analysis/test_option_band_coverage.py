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
        assert band["selection_ratio"] == pytest.approx(1_000 / 9_600)
        assert band["picked_feed_vol"] == 1_000
        # Everything the selection reached, we stored.
        assert s["our_vol"] == band["picked_feed_vol"]

    def test_a_good_selection_with_a_dropped_subscription(self, store):
        """The selection had the volume; we only stored part of it."""
        _candidates(store, picked=[A, C])
        _ticks(store, A, 1_000)
        _ticks(store, C, 2_000)                 # of the feed's 8,000
        s = reconcile(FakeSnapshot(BAND_VOL), [A, C], store, SESSION, UNDER)
        band = s["band"]
        assert band["selection_ratio"] == pytest.approx(9_000 / 9_600)
        assert s["our_vol"] == 3_000
        assert band["ratio"] == pytest.approx(3_000 / 9_600)


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
