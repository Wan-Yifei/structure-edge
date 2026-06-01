"""Quick DB diagnostic for order_book.db."""
import sqlite3
import pathlib

db = pathlib.Path(__file__).parent.parent / "db" / "order_book.db"
print(f"DB path : {db}")
print(f"DB exists: {db.exists()}")
if not db.exists():
    raise SystemExit("DB not found")

con = sqlite3.connect(str(db))
total = con.execute("SELECT COUNT(*) FROM order_book_snapshots").fetchone()[0]
print(f"Total rows: {total:,}")

codes = [r[0] for r in con.execute("SELECT DISTINCT code FROM order_book_snapshots").fetchall()]
print(f"Codes: {codes}")

for code in codes:
    max_ts = con.execute("SELECT MAX(ts) FROM order_book_snapshots WHERE code=?", [code]).fetchone()[0]
    cnt    = con.execute("SELECT COUNT(*) FROM order_book_snapshots WHERE code=?", [code]).fetchone()[0]
    sample = con.execute(
        "SELECT side, price, volume FROM order_book_snapshots WHERE code=? AND ts=?",
        [code, max_ts]
    ).fetchall()
    print(f"\n  {code}  rows={cnt:,}  latest_ts={max_ts}")
    print(f"  latest snapshot ({len(sample)} levels): {sample[:3]}...")

con.close()
