"""Merge a run-specific backtest DB (from HPC / S3) into the local master DB.

The HPC container produces a self-contained backtest_<run_tag>.duckdb for
every run.  This script appends its records into the local backtest.duckdb
without touching the klines cache.

Merge semantics
───────────────
  runs       — INSERT by run_id     (ON CONFLICT DO NOTHING)
  run_stats  — INSERT by run_id     (ON CONFLICT DO UPDATE — keeps latest stats)
  trades     — INSERT by trade_id   (ON CONFLICT DO NOTHING)

Usage
─────
  # Merge a single file
  uv run backtest/merge_db.py --src db/backtest_20260529_1400_smc_v2.3.duckdb

  # Download from S3 then merge (requires AWS CLI and backtest.env)
  uv run backtest/merge_db.py --s3 s3://my-bucket/moomoo/db/backtest_20260529_1400_smc_v2.3.duckdb

  # Merge all run-specific DBs in a directory
  uv run backtest/merge_db.py --src-dir db/hpc_runs/

  # Preview without writing
  uv run backtest/merge_db.py --src db/... --dry-run
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import tempfile
from datetime import datetime

import duckdb

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

_DEFAULT_MASTER = pathlib.Path(__file__).parent.parent / "db" / "backtest.duckdb"

# Tables to merge and their primary key columns
_TABLES: list[tuple[str, str, str]] = [
    # (table_name, pk_column, conflict_action)
    ("runs",      "run_id",   "DO NOTHING"),
    ("trades",    "trade_id", "DO NOTHING"),
    # run_stats: update metrics in case a re-run produced better data
    ("run_stats", "run_id",   "DO UPDATE SET "
                              "n_trades=EXCLUDED.n_trades, "
                              "win_rate=EXCLUDED.win_rate, "
                              "total_r=EXCLUDED.total_r, "
                              "avg_r=EXCLUDED.avg_r, "
                              "profit_factor=EXCLUDED.profit_factor, "
                              "max_drawdown_r=EXCLUDED.max_drawdown_r, "
                              "max_loss_r=EXCLUDED.max_loss_r, "
                              "sharpe=EXCLUDED.sharpe, "
                              "sortino=EXCLUDED.sortino, "
                              "computed_at=now()"),
]


def _row_counts(con: duckdb.DuckDBPyConnection, alias: str) -> dict[str, int]:
    """Return row count for each mergeable table in the attached DB."""
    counts: dict[str, int] = {}
    for tbl, _, _ in _TABLES:
        try:
            counts[tbl] = con.execute(
                f"SELECT COUNT(*) FROM {alias}.{tbl}"
            ).fetchone()[0]
        except Exception:
            counts[tbl] = 0
    return counts


def merge_one(
    src_path: pathlib.Path,
    master_path: pathlib.Path = _DEFAULT_MASTER,
    dry_run: bool = False,
    verbose: bool = True,
) -> dict[str, int]:
    """Merge src_path into master_path.  Returns dict of inserted row counts."""

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    if not src_path.exists():
        raise FileNotFoundError(f"Source DB not found: {src_path}")
    if not master_path.exists():
        raise FileNotFoundError(
            f"Master DB not found: {master_path}\n"
            f"Run a local backtest first to create the schema, or copy an existing DB."
        )

    log(f"[merge] src    : {src_path}")
    log(f"[merge] master : {master_path}")

    con = duckdb.connect(str(master_path), read_only=dry_run)
    alias = "src_db"
    con.execute(f"ATTACH '{src_path}' AS {alias} (READ_ONLY)")

    src_counts  = _row_counts(con, alias)
    pre_counts  = _row_counts(con, "main")

    log(f"\n{'Table':<12} {'src':>8} {'master_before':>14} {'inserted':>10}")
    log("-" * 48)

    inserted: dict[str, int] = {}

    for tbl, pk, conflict_action in _TABLES:
        src_n = src_counts.get(tbl, 0)
        if src_n == 0:
            log(f"  {tbl:<12} {'0':>8}  (empty — skipped)")
            inserted[tbl] = 0
            continue

        # Build column list from master schema (ensures order match)
        cols_df = con.execute(f"DESCRIBE {tbl}").df()
        cols    = ", ".join(cols_df["column_name"].tolist())

        sql = (
            f"INSERT INTO {tbl} ({cols}) "
            f"SELECT {cols} FROM {alias}.{tbl} "
            f"ON CONFLICT ({pk}) {conflict_action}"
        )

        if dry_run:
            # Count what WOULD be inserted
            would_insert = con.execute(
                f"SELECT COUNT(*) FROM {alias}.{tbl} s "
                f"WHERE NOT EXISTS (SELECT 1 FROM {tbl} m WHERE m.{pk} = s.{pk})"
            ).fetchone()[0]
            log(f"  {tbl:<12} {src_n:>8} {pre_counts.get(tbl,0):>14} "
                f"{'~'+str(would_insert):>10}  (dry-run)")
            inserted[tbl] = would_insert
        else:
            con.execute(sql)
            post_n = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
            delta  = post_n - pre_counts.get(tbl, 0)
            log(f"  {tbl:<12} {src_n:>8} {pre_counts.get(tbl,0):>14} {delta:>10}")
            inserted[tbl] = delta

    con.execute(f"DETACH {alias}")
    con.close()

    total = sum(inserted.values())
    tag   = "(dry-run)" if dry_run else ""
    log(f"\n[merge] Done {tag}  total rows inserted: {total}")
    return inserted


def _s3_download(s3_uri: str, env_file: str | None) -> pathlib.Path:
    """Download a DB file from S3/Wasabi to a temp file.  Returns local path."""
    env: dict[str, str] = {}
    if env_file and pathlib.Path(env_file).exists():
        for line in pathlib.Path(env_file).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip()

    profile  = env.get("AWS_PROFILE", os.environ.get("AWS_PROFILE", "default"))
    endpoint = env.get("AWS_ENDPOINT_URL", os.environ.get("AWS_ENDPOINT_URL", ""))

    fname    = s3_uri.rstrip("/").split("/")[-1]
    tmp_path = pathlib.Path(tempfile.mkdtemp()) / fname

    cmd = ["aws", "s3", "cp", s3_uri, str(tmp_path), "--profile", profile]
    if endpoint:
        cmd += ["--endpoint-url", endpoint]

    print(f"[merge] Downloading {s3_uri} ...")
    subprocess.run(cmd, check=True)
    print(f"[merge] Downloaded → {tmp_path}")
    return tmp_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Merge a run-specific backtest DB into the local master DB"
    )
    src_group = ap.add_mutually_exclusive_group(required=True)
    src_group.add_argument(
        "--src", metavar="PATH",
        help="Path to source DB file (backtest_<run_tag>.duckdb)"
    )
    src_group.add_argument(
        "--src-dir", metavar="DIR",
        help="Merge all *.duckdb files in this directory"
    )
    src_group.add_argument(
        "--s3", metavar="S3_URI",
        help="S3/Wasabi URI of the source DB (downloaded then merged)"
    )
    ap.add_argument(
        "--master", default=str(_DEFAULT_MASTER), metavar="PATH",
        help=f"Master DB path (default: {_DEFAULT_MASTER})"
    )
    ap.add_argument(
        "--env-file", default="docker/backtest.env", metavar="PATH",
        help="backtest.env file for AWS credentials (used with --s3)"
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Preview what would be merged without writing"
    )
    args = ap.parse_args()

    master = pathlib.Path(args.master)
    tmp_to_delete: list[pathlib.Path] = []

    if args.s3:
        src_path = _s3_download(args.s3, args.env_file)
        tmp_to_delete.append(src_path.parent)
        sources = [src_path]
    elif args.src_dir:
        sources = sorted(pathlib.Path(args.src_dir).glob("*.duckdb"))
        if not sources:
            print(f"No *.duckdb files found in {args.src_dir}")
            sys.exit(1)
    else:
        sources = [pathlib.Path(args.src)]

    total_inserted: dict[str, int] = {}
    for src in sources:
        print(f"\n{'='*60}")
        result = merge_one(src, master, dry_run=args.dry_run)
        for tbl, n in result.items():
            total_inserted[tbl] = total_inserted.get(tbl, 0) + n

    if len(sources) > 1:
        print(f"\n{'='*60}")
        print("[merge] Grand total inserted:")
        for tbl, n in total_inserted.items():
            print(f"  {tbl:<12} {n:>8}")

    # Clean up temp downloads
    import shutil
    for p in tmp_to_delete:
        shutil.rmtree(p, ignore_errors=True)


if __name__ == "__main__":
    main()
