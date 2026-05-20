"""One-time migration: ticks.duckdb → ticks.db (SQLite)."""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

src = pathlib.Path(__file__).parent / "ticks.duckdb"
dst = pathlib.Path(__file__).parent / "ticks.db"

if not src.exists():
    print("ticks.duckdb not found — nothing to migrate.")
    sys.exit(0)

if dst.exists():
    print(f"ticks.db already exists ({dst}). Delete it first if you want to re-migrate.")
    sys.exit(1)

print(f"Reading {src} ...")
import duckdb
con = duckdb.connect(str(src), read_only=True)
rows = con.execute(
    "SELECT code, ts::VARCHAR, price, volume, direction FROM ticks ORDER BY ts"
).fetchall()
con.close()
print(f"  {len(rows)} ticks read.")

print(f"Writing {dst} ...")
from db.tick_store import TickStore
store = TickStore(dst)
store._con.executemany(
    "INSERT OR IGNORE INTO ticks VALUES (?, ?, ?, ?, ?)", rows
)
store._con.commit()
n = store.row_count()
store.close()
print(f"  {n} ticks written to SQLite.")

src.rename(src.with_name("ticks.duckdb.migrated"))
print("Done. Old file renamed to ticks.duckdb.migrated.")
