"""One session can say a parameter is WRONG. It cannot say what it should be.

The advice this covers proposes and never writes, because a single session is
not enough to tune anything: on 2026-11-24 one contract carried 19.5% of the
day's delta-weighted flow, so a parameter fitted to one day is fitted to that
day's accident. Rankings, per-expiry slot caps and the quota split all wait
for the 8-10 sessions the band dataset is being collected for.

What survives a one-session read is only what is a FACT:

  * asked vs listed -- OpenD either listed the contracts or it did not
  * candidates vs slots -- the pre-filter either binds or it does not
  * the DIFFERENCE between those two, which is the reason this exists at all:
    `asked 60, listed 48` and `asked 60, listed 60 with 12 caught_nothing`
    produce the same coverage numbers and call for opposite actions. The
    first is an over-committed quota; the second is a dropped subscription,
    where lowering the quota would change nothing.

The tests below are mostly about that last distinction, and about the advice
staying advice.
"""
import pytest

from analysis.option_tick_collector import calibration_advice
from feeds.option_tick_store import OptionTickStore

SESSION = "2026-11-25"
UNDER = "US.QQQ"


@pytest.fixture
def store(tmp_path):
    s = OptionTickStore(tmp_path / "opt.db")
    yield s
    s.close()


def _cands(store, n_candidates, n_picked):
    store.record_candidates(SESSION, UNDER, [
        {"code": f"US.QQQ261127C{i:06d}", "rank": i + 1,
         "score": float(n_candidates - i), "picked": i < n_picked}
        for i in range(n_candidates)])


def _summary(band=None, caught_nothing=0):
    return {"band": band, "caught_nothing": caught_nothing,
            "underlying": UNDER, "session": SESSION}


def _text(lines):
    return "\n".join(lines)


def _proposals(lines):
    return [ln.strip() for ln in lines if ln.lstrip().startswith("->")]


class TestTheQuotaReadIsAFactNotAnEstimate:

    def test_a_short_list_proposes_lowering_this_codes_share(self, store):
        _cands(store, 300, 60)
        store.record_subscription_check(SESSION, UNDER, asked=60, listed=48)
        lines = calibration_advice(store, SESSION, UNDER, _summary())
        assert "asked 60, listed 48" in _text(lines)
        assert any("lower this code's share to 48" in p
                   for p in _proposals(lines)), _text(lines)

    def test_a_full_list_proposes_nothing_about_the_quota(self, store):
        _cands(store, 300, 60)
        store.record_subscription_check(SESSION, UNDER, asked=60, listed=60)
        lines = calibration_advice(store, SESSION, UNDER, _summary())
        assert "the split held" in _text(lines)
        assert "lower this code's share" not in _text(lines)

    def test_a_failed_read_is_not_reported_as_zero(self, store):
        """An empty subscription list is the loudest possible signal and a
        failed read is no signal at all; proposing a quota of 0 from a failed
        read would silence the collector."""
        _cands(store, 300, 60)
        store.record_subscription_check(SESSION, UNDER, asked=60, listed=None)
        lines = calibration_advice(store, SESSION, UNDER, _summary())
        assert "unknown, not zero" in _text(lines)
        assert "lower this code's share" not in _text(lines)

    def test_a_missing_check_says_so_rather_than_guessing(self, store):
        _cands(store, 300, 60)
        lines = calibration_advice(store, SESSION, UNDER, _summary())
        assert "no subscription check recorded" in _text(lines)
        assert "lower this code's share" not in _text(lines)


class TestTheTwoShortfallsAreToldApart:
    """The whole reason the asked/listed row is written down."""

    def test_a_dropped_subscription_is_not_called_a_quota_problem(self, store):
        _cands(store, 300, 60)
        store.record_subscription_check(SESSION, UNDER, asked=60, listed=60)
        lines = calibration_advice(store, SESSION, UNDER,
                                   _summary(caught_nothing=12))
        text = _text(lines)
        assert "NOT a quota problem" in text, text
        assert "subscription stopped" in text
        # The phrase that carries the ACTION, not the bare word: the
        # advice for this case says "lowering it would change nothing", so a
        # substring match on "lower" matches the negation too.
        assert "lower this code's share" not in text, (
            "it proposed lowering the quota for a failure lowering the quota "
            "cannot fix")

    def test_an_over_committed_quota_is_not_called_a_dropped_subscription(
            self, store):
        _cands(store, 300, 60)
        store.record_subscription_check(SESSION, UNDER, asked=60, listed=48)
        lines = calibration_advice(store, SESSION, UNDER,
                                   _summary(caught_nothing=12))
        text = _text(lines)
        assert "short" in text
        assert "NOT a quota problem" not in text, (
            "with 12 contracts missing from the subscription list, a quota "
            "share IS the thing to change")


class TestThePreFilterIsOnlyFlaggedWhenItBinds:

    def test_a_loose_band_proposes_nothing(self, store):
        """2,288 candidates for 60 slots: whatever the volume did, the band
        is nowhere near the constraint. Measured on 2026-11-24, the strikes
        between QQQ's +/-15% band and +/-40% held 0.01% of the day's
        delta-weighted flow."""
        _cands(store, 2288, 60)
        store.record_subscription_check(SESSION, UNDER, asked=60, listed=60)
        lines = calibration_advice(store, SESSION, UNDER, _summary())
        assert "38x" in _text(lines)
        assert not any("pct" in p for p in _proposals(lines))

    def test_a_binding_band_proposes_widening(self, store):
        """Candidates barely above the slots: the band, not the quota, is
        choosing -- which is the one case where width matters."""
        _cands(store, 70, 60)
        store.record_subscription_check(SESSION, UNDER, asked=60, listed=60)
        lines = calibration_advice(store, SESSION, UNDER, _summary())
        assert any("pct" in p and "widen" in p for p in _proposals(lines)), (
            _text(lines))

    def test_no_candidate_list_says_so(self, store):
        lines = calibration_advice(store, SESSION, UNDER, _summary())
        assert "no candidate list recorded" in _text(lines)

    def test_nothing_selected_is_not_reported_as_the_loosest_band(self, store):
        """The ratio is candidates/slots, and zero slots has no ratio.

        Guarding the division with `or 1` -- which this did -- turns the
        loudest possible failure into the healthiest possible reading:
        2,288 candidates over a fabricated single slot prints as "2288x, not
        binding" for a session that selected nothing and collected no tape.
        Same shape as reporting a failed read as zero.
        """
        _cands(store, 2288, 0)
        store.record_subscription_check(SESSION, UNDER, asked=0, listed=0)
        text = _text(calibration_advice(store, SESSION, UNDER, _summary()))
        assert "NOTHING was selected" in text, text
        assert "for 1 slots" not in text
        assert "2288x" not in text


class TestItStaysAdvice:

    def test_it_writes_nothing(self, store):
        """Proposal-only. A session that silently rewrote a parameter would
        be tuning on one day's noise, and would do it where nobody looked."""
        _cands(store, 70, 60)
        store.record_subscription_check(SESSION, UNDER, asked=60, listed=48)
        before = {t: store._con.execute(
                      f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                  for t in ("option_candidates", "option_subscription_check",
                            "option_band_volume", "option_coverage")}
        calibration_advice(store, SESSION, UNDER, _summary())
        after = {t: store._con.execute(
                     f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                 for t in before}
        assert before == after

    def test_a_quiet_session_says_so_explicitly(self, store):
        _cands(store, 2288, 60)
        store.record_subscription_check(SESSION, UNDER, asked=60, listed=60)
        lines = calibration_advice(store, SESSION, UNDER, _summary())
        assert _proposals(lines) == ["-> no change proposed"]

    def test_the_coverage_numbers_ride_along_when_present(self, store):
        _cands(store, 2288, 60)
        store.record_subscription_check(SESSION, UNDER, asked=60, listed=60)
        lines = calibration_advice(store, SESSION, UNDER, _summary(
            band={"ratio": 0.134, "ranking_ratio": 0.141,
                  "taken_ratio": 0.293}))
        assert "band 13.4%" in _text(lines)
        # The RANKING figure, not the held one. On a re-pick day the held
        # figure is the wider of the two (7.0% against 14.6% for QQQ on
        # 2026-11-25), so advising from it reads a messier day as a
        # better-aimed one.
        assert "ranking reached 14.1%" in _text(lines)
        assert "29.3%" not in _text(lines)

    def test_it_survives_a_session_with_nothing_at_all(self, store):
        lines = calibration_advice(store, SESSION, UNDER, _summary())
        assert lines and _proposals(lines) == ["-> no change proposed"]


class TestSlotsAreTheQuotaAsked:
    """`slots` was sum(picked) over candidates_on, and on 2026-11-25 that
    reader returned a union of rounds, so SOXL printed "546 candidates for
    150 slots (4x)" on the same screen as "asked 140" -- 150 being two
    rounds' picks added together, a number no round ever had.

    Every other case in this file makes asked equal sum(picked), which is
    why the mutation that reverts this line survived the whole suite.
    """

    def test_the_subscription_check_wins_over_counting_picks(self, store):
        _cands(store, 300, 150)                 # as if two rounds were summed
        store.record_subscription_check(SESSION, UNDER, asked=140, listed=140)
        lines = calibration_advice(store, SESSION, UNDER, _summary())
        assert "300 candidates for 140 slots" in _text(lines)
        assert "150 slots" not in _text(lines)

    def test_the_wrong_slot_count_would_fire_a_spurious_proposal(self, store):
        """Not cosmetic: the ratio is what decides whether advice is given.

        290 candidates is 2.07x against the 140 asked -- silent -- and
        1.93x against a summed 150, which trips the `< 2` gate and tells
        the operator to widen --pct. Widening the band is the opposite of
        what a day with two selection rounds needs.
        """
        _cands(store, 290, 150)
        store.record_subscription_check(SESSION, UNDER, asked=140, listed=140)
        lines = calibration_advice(store, SESSION, UNDER, _summary())
        assert 290 / 140 > 2 > 290 / 150, "the pair must straddle the gate"
        assert not any("pre-filter is binding" in p
                       for p in _proposals(lines)), _text(lines)

    def test_the_gate_still_fires_when_the_band_really_is_tight(self, store):
        """The guard above must not be passing because advice never fires."""
        _cands(store, 200, 140)
        store.record_subscription_check(SESSION, UNDER, asked=140, listed=140)
        lines = calibration_advice(store, SESSION, UNDER, _summary())
        assert any("pre-filter is binding" in p
                   for p in _proposals(lines)), _text(lines)

    def test_counting_picks_remains_the_fallback(self, store):
        """A session with no subscription check recorded -- one that
        predates that table -- still gets a reading rather than a crash."""
        _cands(store, 300, 60)
        lines = calibration_advice(store, SESSION, UNDER, _summary())
        assert "300 candidates for 60 slots" in _text(lines)
