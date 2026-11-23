"""Split fills must survive, and a missing greek must not read as zero.

Both of these corrupt the exact quantity the option delta indicator totals
up, and neither is recoverable: the option tape cannot be re-fetched, so a
session collected under either bug is a session thrown away.

The numbers in the split-fill tests are the shapes measured on three real
SOXL option tapes from 2026-11-20 -- one 42-contract order filling as
21 + 21 at the same millisecond and price, and a six-print burst of
1/10/10/3/1/1 contracts. Under the original UNIQUE(code, ts, price, volume)
those collapsed to one and to three rows respectively, losing 8-15% of
prints and skewing net signed contracts by -4.4% to +4.7% -- a skew that
does NOT cancel, since the sign differed by contract.
"""
import pytest

from feeds.option_tick_store import OptionTickStore


@pytest.fixture
def store(tmp_path):
    s = OptionTickStore(tmp_path / "opt.db")
    yield s
    s.close()


def _tick(seq, volume=1, ts="2026-11-20 14:17:00.744", price=1.23,
          direction="BUY", code="US.SOXL261127C40000"):
    return {"code": code, "ts": ts, "price": price, "volume": volume,
            "direction": direction, "ttype": "AUTO_MATCH", "seq": seq}


class TestSplitFillsSurvive:

    def test_two_halves_of_one_order_are_both_kept(self, store):
        """21 + 21 at the same millisecond, same price, consecutive seq."""
        n = store.insert_ticks([_tick(100, 21), _tick(101, 21)])
        assert n == 2, "a split fill collapsed into one row"

    def test_a_six_print_burst_keeps_every_contract(self, store):
        """1,10,10,3,1,1 BUY -> 26 contracts. The old key stored 1,10,3 = 14."""
        vols = [1, 10, 10, 3, 1, 1]
        store.insert_ticks([_tick(200 + i, v) for i, v in enumerate(vols)])
        total = store._con.execute(
            "SELECT SUM(volume) FROM option_ticks").fetchone()[0]
        assert total == sum(vols) == 26, f"stored {total} of 26 contracts"

    def test_a_replay_still_de_duplicates(self, store):
        """The reason seq was left out of the key originally.

        It does not need to be: sequence was verified unique and monotonic
        within a contract, and two independent fetches of the same window
        returned identical (time, sequence) pairs. So a replay collapses on
        seq just as well.
        """
        batch = [_tick(300, 5), _tick(301, 7), _tick(302, 2)]
        assert store.insert_ticks(batch) == 3
        assert store.insert_ticks(batch) == 0, "replay was not de-duplicated"
        assert store._con.execute(
            "SELECT COUNT(*) FROM option_ticks").fetchone()[0] == 3

    def test_a_missing_seq_falls_back_instead_of_disabling_dedup(self, store):
        """NULL never collides in a SQLite UNIQUE index, so a NULL seq would
        turn de-duplication off entirely for those rows."""
        rows = [_tick(None, 9), _tick(None, 9)]
        store.insert_ticks(rows)
        kept = store._con.execute(
            "SELECT COUNT(*), MIN(seq) FROM option_ticks").fetchone()
        assert kept == (1, -1), kept

    def test_the_same_print_on_two_contracts_is_not_merged(self, store):
        a = _tick(400, 5, code="US.SOXL261127C40000")
        b = _tick(400, 5, code="US.SOXL261127P40000")
        assert store.insert_ticks([a, b]) == 2


class TestAMissingGreekIsNotAZero:

    def test_nogreeks_rows_store_null_not_zero(self, store):
        store.insert_greeks([{
            "code": "US.SOXL261127P70000", "ts": "2026-11-20 15:00:00",
            "delta": None, "gamma": None, "iv": None, "oi": 1921,
            "spot": 30.88, "upd": "2026-11-20 09:30:00", "nogreeks": 1,
        }])
        row = store._con.execute(
            "SELECT delta, gamma, iv, oi, nogreeks FROM option_greeks").fetchone()
        assert row == (None, None, None, 1921, 1), row

    def test_a_real_zero_is_distinguishable_from_a_missing_one(self, store):
        """The whole point of the flag: delta really can be ~0 on a far
        out-of-the-money contract, and that is a measurement."""
        store.insert_greeks([
            {"code": "A", "ts": "t1", "delta": 0.0, "gamma": 0.0, "iv": 180.0,
             "oi": 10, "spot": 31.0, "upd": "u", "nogreeks": 0},
            {"code": "B", "ts": "t1", "delta": None, "gamma": None, "iv": None,
             "oi": 10, "spot": 31.0, "upd": "u", "nogreeks": 1},
        ])
        rows = dict(store._con.execute(
            "SELECT code, nogreeks FROM option_greeks").fetchall())
        assert rows == {"A": 0, "B": 1}
        measured_zero = store._con.execute(
            "SELECT delta FROM option_greeks WHERE code='A'").fetchone()[0]
        assert measured_zero == 0.0, "a measured zero was turned into NULL"

    def test_nogreeks_defaults_to_zero_when_absent(self, store):
        store.insert_greeks([{"code": "C", "ts": "t1", "delta": 0.5,
                              "gamma": 0.03, "iv": 200.0, "oi": 5,
                              "spot": 31.0, "upd": "u"}])
        assert store._con.execute(
            "SELECT nogreeks FROM option_greeks").fetchone()[0] == 0


class TestHousekeeping:

    def test_contracts_are_idempotent(self, store):
        c = [{"code": "X", "underlying": "US.SOXL", "expiry": "2026-11-27",
              "strike": 31.0, "opt_type": "CALL"}]
        assert store.upsert_contracts(c) == 1
        assert store.upsert_contracts(c) == 0
        assert len(store.contracts_for("US.SOXL")) == 1

    def test_prune_drops_old_rows_from_both_tables(self, store):
        store.insert_ticks([_tick(1, 1, ts="2020-01-01 10:00:00")])
        store.insert_greeks([{"code": "X", "ts": "2020-01-01 10:00:00",
                              "delta": 0.5, "gamma": 0.1, "iv": 50.0,
                              "oi": 1, "spot": 10.0, "upd": "u"}])
        store.insert_ticks([_tick(2, 1)])     # 2026, stays
        t, g = store.prune_before(30)
        assert (t, g) == (1, 1), (t, g)
        assert store._con.execute(
            "SELECT COUNT(*) FROM option_ticks").fetchone()[0] == 1
