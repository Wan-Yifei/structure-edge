"""Two threads, one connection.

The store is opened with check_same_thread=False and handed to both the
SDK's push callback (insert_ticks) and the main loop's greeks sampler
(insert_greeks / checkpoint / session_counts). On 2026-11-23 that produced
two of these, each costing the whole push batch:

    cannot commit - no transaction is active
    cannot commit transaction - SQL statements in progress

What must not interleave is a statement and the commit that closes it, so
these tests drive both writers at once and assert on BOTH outcomes: no
exception escaped, and every row survived. The second half matters on its
own -- a lost batch raises in the collector's handler, where a
`except Exception` swallows it into a counter.
"""
import sqlite3
import threading

import pytest

from feeds.option_tick_store import OptionTickStore

ROUNDS = 60
BATCH = 8


def _ticks(worker: int, i: int):
    return [{
        "code": f"US.SOXL261127C{40000 + worker}",
        "ts": f"2026-11-23 10:{i % 60:02d}:{j:02d}.000",
        "price": 1.0 + j, "volume": 1, "direction": "BUY",
        "ttype": "AUTO_MATCH", "seq": i * BATCH + j,
        "recv_ts": None, "feed_ts": None,
    } for j in range(BATCH)]


def _greeks(worker: int, i: int):
    return [{
        "code": f"US.SOXL261127P{30000 + worker}",
        "ts": f"2026-11-23 10:{i % 60:02d}:{j:02d}.000",
        "delta": -0.4, "gamma": 0.01, "iv": 0.5, "oi": 100,
        "spot": 44.0, "upd": "2026-11-23 10:00:00", "nogreeks": 0,
    } for j in range(BATCH)]


@pytest.fixture
def store(tmp_path):
    s = OptionTickStore(tmp_path / "opt.db")
    s.upsert_contracts([
        {"code": f"US.SOXL261127C{40000 + w}", "underlying": "US.SOXL",
         "expiry": "2026-11-27", "strike": 40.0 + w, "opt_type": "CALL"}
        for w in range(4)
    ] + [
        {"code": f"US.SOXL261127P{30000 + w}", "underlying": "US.SOXL",
         "expiry": "2026-11-27", "strike": 30.0 + w, "opt_type": "PUT"}
        for w in range(4)
    ])
    yield s
    s.close()


def _run(targets):
    """Run the callables in parallel; return whatever they raised."""
    errors: list[BaseException] = []
    barrier = threading.Barrier(len(targets))

    def wrap(fn):
        def go():
            barrier.wait()          # start together, maximise overlap
            try:
                fn()
            except BaseException as exc:            # noqa: BLE001
                errors.append(exc)
        return go

    threads = [threading.Thread(target=wrap(f)) for f in targets]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert not any(t.is_alive() for t in threads), "a writer deadlocked"
    return errors


class TestConcurrentWritersDoNotCorruptEachOthersTransactions:

    def test_ticks_and_greeks_at_once(self, store):
        def push():
            for i in range(ROUNDS):
                store.insert_ticks(_ticks(0, i))

        def sample():
            for i in range(ROUNDS):
                store.insert_greeks(_greeks(0, i))

        errors = _run([push, sample])
        assert not errors, f"{type(errors[0]).__name__}: {errors[0]}"

        # Not just "nothing raised": a batch lost to a broken transaction
        # is exactly what the collector's handler swallows into a counter.
        n_t = store._con.execute("SELECT COUNT(*) FROM option_ticks").fetchone()[0]
        n_g = store._con.execute("SELECT COUNT(*) FROM option_greeks").fetchone()[0]
        assert n_t == ROUNDS * BATCH, f"lost ticks: {n_t} of {ROUNDS * BATCH}"
        assert n_g == ROUNDS * BATCH, f"lost greeks: {n_g} of {ROUNDS * BATCH}"

    def test_four_writers_plus_a_reader_and_a_checkpoint(self, store):
        """The real shape: several push batches, the sampler, and the
        periodic checkpoint/count that the main loop runs between sweeps."""
        def push(w):
            def go():
                for i in range(ROUNDS):
                    store.insert_ticks(_ticks(w, i))
            return go

        def housekeeping():
            for _ in range(ROUNDS // 4):
                store.checkpoint()
                store.session_counts("2026-11-23 00:00:00")

        errors = _run([push(0), push(1), push(2),
                       lambda: [store.insert_greeks(_greeks(1, i))
                                for i in range(ROUNDS)],
                       housekeeping])
        assert not errors, f"{type(errors[0]).__name__}: {errors[0]}"
        n_t = store._con.execute("SELECT COUNT(*) FROM option_ticks").fetchone()[0]
        assert n_t == 3 * ROUNDS * BATCH, f"lost ticks: {n_t}"

    def test_the_lock_exists_at_all(self, store):
        """Cheap and blunt: the decorator is what the two tests above rely
        on, and it is a single line anyone can delete while 'tidying'."""
        assert isinstance(getattr(store, "_lock", None),
                          type(threading.RLock())), "the store lost its lock"
