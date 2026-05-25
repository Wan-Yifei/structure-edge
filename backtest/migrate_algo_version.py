"""One-shot migration: re-tag runs written under the wrong ALGO_VERSION.

Usage:
    uv run backtest/migrate_algo_version.py --from smc_v2.1 --to smc_v2.2 \
        --symbols US.CSCO US.AMD US.NVDA US.QCOM [--dry-run]

What it does:
1. Finds all runs matching --from version + optional symbol filter.
2. Recomputes each trade_id using the new version string
   (SHA256(new_ver:entry_time:direction:entry_price:.6f:sl_price:.6f)[:8]).
3. Updates trades.trade_id and review_trades.trade_id in both DBs.
4. Updates runs.algo_version.

Run with --dry-run first to preview changes.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import duckdb

_DB_PATH        = pathlib.Path(__file__).parent.parent / "db" / "backtest.duckdb"
_REVIEW_DB_PATH = pathlib.Path(__file__).parent.parent / "db" / "review_trades.duckdb"


def _new_trade_id(algo_version: str, entry_time: str, direction: str,
                  entry_price: float, sl_price: float) -> str:
    key = f"{algo_version}:{entry_time}:{direction}:{entry_price:.6f}:{sl_price:.6f}"
    return hashlib.sha256(key.encode()).hexdigest()[:8]


def migrate(
    from_version: str,
    to_version: str,
    symbols: list[str] | None,
    dry_run: bool,
) -> None:
    conn        = duckdb.connect(str(_DB_PATH), read_only=dry_run)
    review_conn = duckdb.connect(str(_REVIEW_DB_PATH), read_only=dry_run)

    # Find matching runs
    sym_filter = ""
    params: list = [from_version]
    if symbols:
        placeholders = ",".join("?" * len(symbols))
        sym_filter = f"AND symbol IN ({placeholders})"
        params += symbols

    runs = conn.execute(
        f"SELECT run_id, symbol FROM runs WHERE algo_version = ? {sym_filter}",
        params,
    ).fetchall()

    if not runs:
        print(f"No runs found with algo_version='{from_version}'" +
              (f" and symbols {symbols}" if symbols else "") + ".")
        return

    run_ids = [r[0] for r in runs]
    print(f"Found {len(runs)} runs to update:")
    for run_id, symbol in runs:
        print(f"  {run_id}  {symbol}")

    # Fetch all affected trades
    placeholders = ",".join("?" * len(run_ids))
    trades = conn.execute(
        f"SELECT trade_id, direction, entry_time, entry_price, sl_price "
        f"FROM trades WHERE run_id IN ({placeholders})",
        run_ids,
    ).fetchall()

    print(f"\n{len(trades)} trades to re-ID ({from_version} → {to_version}):")

    mapping: dict[str, str] = {}  # old_id → new_id
    conflicts = 0
    for old_id, direction, entry_time, entry_price, sl_price in trades:
        new_id = _new_trade_id(to_version, entry_time, direction, entry_price, sl_price)
        if old_id == new_id:
            continue
        if new_id in mapping.values():
            print(f"  WARNING: collision on new_id={new_id}")
            conflicts += 1
        mapping[old_id] = new_id

    if conflicts:
        print(f"\nAborted — {conflicts} ID collision(s) detected.")
        return

    for old_id, new_id in list(mapping.items())[:5]:
        print(f"  {old_id} → {new_id}")
    if len(mapping) > 5:
        print(f"  ... and {len(mapping) - 5} more")

    if dry_run:
        print("\n[DRY RUN] No changes written.")
        return

    # Apply updates — update trade_id in trades table
    # DuckDB allows UPDATE on PK as long as new value doesn't already exist.
    # Process in two phases to avoid transient conflicts within the batch.
    print("\nApplying updates …")

    # Phase 1: rename to temp IDs to avoid self-conflict
    temp_mapping: dict[str, str] = {old: f"__tmp__{new}" for old, new in mapping.items()}
    for old_id, tmp_id in temp_mapping.items():
        conn.execute("UPDATE trades SET trade_id = ? WHERE trade_id = ?", [tmp_id, old_id])
        review_conn.execute(
            "UPDATE review_trades SET trade_id = ? WHERE trade_id = ?", [tmp_id, old_id]
        )

    # Phase 2: rename from temp IDs to final IDs
    for old_id, new_id in mapping.items():
        tmp_id = f"__tmp__{new_id}"
        conn.execute("UPDATE trades SET trade_id = ? WHERE trade_id = ?", [new_id, tmp_id])
        review_conn.execute(
            "UPDATE review_trades SET trade_id = ? WHERE trade_id = ?", [new_id, tmp_id]
        )

    # Update algo_version on runs
    conn.execute(
        f"UPDATE runs SET algo_version = ? WHERE run_id IN ({placeholders})",
        [to_version] + run_ids,
    )

    conn.close()
    review_conn.close()
    print(f"Done. {len(mapping)} trade IDs updated, {len(runs)} runs re-tagged to {to_version}.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-tag backtest runs to correct ALGO_VERSION")
    ap.add_argument("--from", dest="from_version", required=True,
                    help="Current (wrong) algo_version, e.g. smc_v2.1")
    ap.add_argument("--to", dest="to_version", required=True,
                    help="Correct algo_version, e.g. smc_v2.2")
    ap.add_argument("--symbols", nargs="+",
                    help="Limit to these symbols (e.g. US.CSCO US.AMD). Omit to update all.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Preview changes without writing")
    args = ap.parse_args()

    migrate(args.from_version, args.to_version, args.symbols, args.dry_run)


if __name__ == "__main__":
    main()
