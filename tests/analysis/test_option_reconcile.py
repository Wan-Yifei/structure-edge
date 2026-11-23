"""Reconciliation is the only way to learn a session was under-collected.

The option tape cannot be re-fetched, so "did we miss anything?" has to be
answered while the answer still matters, and the failure it looks for --
a subscription that quietly stops delivering -- leaves the process looking
perfectly healthy.

Which makes a reconciliation that reports a WRONG number worse than none.
The snapshot's `volume` is the LAST session's total, not the session being
asked about: run on a Sunday against today's date, the first version
compared Friday's 251,263 contracts with an empty table and announced 0%
coverage. An alarming number about nothing is exactly the class of defect
this check exists to catch.
"""
import pytest

from moomoo import RET_OK, RET_ERROR

from analysis.option_tick_collector import reconcile
from feeds.option_tick_store import OptionTickStore

SESSION = "2026-11-20"
CODES = ["US.SOXL261127C40000", "US.SOXL261127P25000"]


class FakeSnapshot:
    """get_market_snapshot with per-code volume and update_time."""

    def __init__(self, volumes: dict, upd_date: str = SESSION, fail=False):
        self.volumes = volumes
        self.upd_date = upd_date
        self.fail = fail

    def get_market_snapshot(self, codes):
        import pandas as pd
        if self.fail:
            return RET_ERROR, "boom"
        return RET_OK, pd.DataFrame([
            {"code": c, "volume": self.volumes.get(c, 0),
             "update_time": f"{self.upd_date} 16:00:00"}
            for c in codes
        ])


@pytest.fixture
def store(tmp_path):
    s = OptionTickStore(tmp_path / "opt.db")
    yield s
    s.close()


def _store_ticks(store, code, volumes, day=SESSION):
    store.insert_ticks([
        {"code": code, "ts": f"{day} 10:0{i}:00.000", "price": 1.0 + i,
         "volume": v, "direction": "BUY", "ttype": "AUTO_MATCH", "seq": 1000 + i}
        for i, v in enumerate(volumes)
    ])


class TestItRefusesToCompareAcrossDays:

    def test_a_feed_figure_from_another_day_is_not_used(self, store):
        """The bug this closes: Friday's volume scored against Sunday."""
        ctx = FakeSnapshot({c: 22_616 for c in CODES}, upd_date="2026-11-20")
        s = reconcile(ctx, CODES, store, session="2026-11-22")

        assert s["wrong_day"] == len(CODES)
        assert s["feed_vol"] == 0, "a different day's volume was totalled in"
        assert s["ratio"] is None or s["ratio"] == 0

    def test_the_same_day_does_compare(self, store):
        _store_ticks(store, CODES[0], [10, 20, 30])       # 60 of 100
        ctx = FakeSnapshot({CODES[0]: 100, CODES[1]: 0}, upd_date=SESSION)
        s = reconcile(ctx, CODES, store, session=SESSION)

        assert s["wrong_day"] == 0
        assert s["our_vol"] == 60
        assert s["feed_vol"] == 100
        assert s["ratio"] == pytest.approx(0.6)


class TestItNoticesWhatIsMissing:

    def test_a_contract_with_nothing_collected_is_flagged(self, store):
        _store_ticks(store, CODES[0], [50])
        ctx = FakeSnapshot({CODES[0]: 50, CODES[1]: 9_000})
        s = reconcile(ctx, CODES, store, session=SESSION)

        assert s["caught_nothing"] == 1, "a silent contract went unreported"
        assert s["worst"][0][0] == CODES[1]
        assert s["worst"][0][1:] == (0, 9_000)

    def test_full_coverage_reports_no_shortfall(self, store):
        _store_ticks(store, CODES[0], [40, 60])
        ctx = FakeSnapshot({CODES[0]: 100, CODES[1]: 0})
        s = reconcile(ctx, CODES, store, session=SESSION)

        assert s["caught_nothing"] == 0
        assert s["worst"] == []
        assert s["ratio"] == pytest.approx(1.0)

    def test_ticks_from_another_day_do_not_count(self, store):
        """Grouping is on the exchange date, so yesterday's prints must not
        inflate today's coverage."""
        _store_ticks(store, CODES[0], [1000], day="2026-11-19")
        _store_ticks(store, CODES[0], [25], day=SESSION)
        ctx = FakeSnapshot({CODES[0]: 100, CODES[1]: 0})
        s = reconcile(ctx, CODES, store, session=SESSION)

        assert s["our_vol"] == 25, "a different day's ticks were counted"


class TestItIsRecordedNotJustLogged:

    def test_the_result_persists_for_later_readers(self, store):
        _store_ticks(store, CODES[0], [30])
        ctx = FakeSnapshot({CODES[0]: 100, CODES[1]: 0})
        reconcile(ctx, CODES, store, session=SESSION)

        rows = dict(store._con.execute(
            "SELECT code, our_vol FROM option_coverage WHERE session = ?",
            (SESSION,)).fetchall())
        assert rows == {CODES[0]: 30, CODES[1]: 0}

    def test_rerunning_replaces_rather_than_duplicates(self, store):
        ctx = FakeSnapshot({c: 10 for c in CODES})
        reconcile(ctx, CODES, store, session=SESSION)
        _store_ticks(store, CODES[0], [10])
        reconcile(ctx, CODES, store, session=SESSION)

        n = store._con.execute(
            "SELECT COUNT(*) FROM option_coverage WHERE session = ?",
            (SESSION,)).fetchone()[0]
        assert n == len(CODES)
        assert store.coverage_summary(SESSION)["our_vol"] == 10

    def test_a_failed_snapshot_does_not_invent_coverage(self, store):
        _store_ticks(store, CODES[0], [30])
        s = reconcile(FakeSnapshot({}, fail=True), CODES, store, session=SESSION)

        assert s["feed_vol_unknown"] == len(CODES)
        assert s["ratio"] is None, "a ratio was reported with no feed figure"
