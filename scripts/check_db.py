"""Quick DB diagnostic for order_book.db."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from feeds.order_book_store import OrderBookStore

db = pathlib.Path(__file__).parent.parent / "db" / "order_book.db"
print(f"DB path : {db}")
print(f"DB exists: {db.exists()}")
if not db.exists():
    raise SystemExit("DB not found")

with OrderBookStore(db, read_only=True) as store:
    print(f"Total snapshots: {store.snapshot_count():,}")

    codes = store.codes()
    print(f"Codes: {codes}")

    for code in codes:
        cnt    = store.snapshot_count(code)
        latest = store.latest_snapshot(code)
        max_ts = latest[0]["ts"] if latest else None
        sample = [(r["side"], r["price"], r["volume"]) for r in latest]
        print(f"\n  {code}  snapshots={cnt:,}  latest_ts={max_ts}")
        print(f"  latest snapshot ({len(sample)} levels): {sample[:3]}...")
