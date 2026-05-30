#!/usr/bin/env bash
# post_backtest.sh — run report, audit, fvg_inspect for a completed backtest run
#
# Usage:
#   bash backtest/post_backtest.sh <run_dir> <start> <end> <inspect_start> <inspect_end>
#
# Example:
#   bash backtest/post_backtest.sh \
#       backtest/results/20260529_0057_smc_v2.2_soxl_random_random_300 \
#       2025-05-22 2026-05-22 \
#       2025-10-01 2025-10-31
#
# Runs report.py + audit.py (rank 1, min 5 trades) + fvg_inspect.py for every
# stock subdirectory found under <run_dir>.  All outputs land in the same subdir.

set -euo pipefail

if [ $# -lt 5 ]; then
    echo "Usage: $0 <run_dir> <start> <end> <inspect_start> <inspect_end>"
    exit 1
fi

RUN_DIR=$1
START=$2
END=$3
INSPECT_START=$4
INSPECT_END=$5

echo "=== post_backtest: $RUN_DIR ==="
echo "    backtest range : $START -> $END"
echo "    inspect window : $INSPECT_START -> $INSPECT_END"
echo ""

found=0
for csv in "$RUN_DIR"/*/results_*.csv; do
    [ -f "$csv" ] || continue
    found=$((found + 1))
    out_dir=$(dirname "$csv")

    echo "--- [$found] $csv ---"

    echo "[1/3] report ..."
    uv run python -m backtest.report "$csv"

    echo "[2/3] audit (rank 1, min-trades 5) ..."
    uv run backtest/audit.py \
        --from-csv "$csv" \
        --rank 1 --min-trades 5 \
        --start "$START" --end "$END" \
        --out-dir "$out_dir"

    echo "[3/3] fvg_inspect ($INSPECT_START -> $INSPECT_END) ..."
    uv run backtest/fvg_inspect.py \
        --from-csv "$csv" \
        --rank 1 --min-trades 5 \
        --start "$START" --end "$END" \
        --inspect-start "$INSPECT_START" \
        --inspect-end   "$INSPECT_END" \
        --out-dir "$out_dir"

    echo ""
done

if [ "$found" -eq 0 ]; then
    echo "ERROR: no results_*.csv files found under $RUN_DIR"
    exit 1
fi

echo "=== Done. $found stock(s) processed. ==="
echo "Next: write REVIEW.md summarising the report findings."
