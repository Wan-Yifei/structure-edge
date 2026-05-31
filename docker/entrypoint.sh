#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# entrypoint.sh — moomoo backtest pipeline
#
# Stages:
#   1. Download kline cache from S3 (read-only; never uploaded back)
#   2. Run backtest grid  → writes db/backtest.duckdb + backtest/results/<tag>/
#   3. Generate per-stock audit reports
#   4. Generate per-stock FVG inspect reports
#   5. Upload results dir + run-specific DB to S3
#      backtest_<run_tag>.duckdb is uploaded separately from the master DB so
#      it can be safely merged locally with: uv run backtest/merge_db.py --s3 …
#
# All configuration comes from environment variables (see backtest.env.example).
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

cd /app

log() { echo "[entrypoint] $(date '+%H:%M:%S') $*"; }
die() { echo "[entrypoint] ERROR: $*" >&2; exit 1; }

# ── Validate required vars ────────────────────────────────────────────────────
: "${S3_BUCKET:?S3_BUCKET is required (set in backtest.env)}"
: "${AWS_PROFILE:?AWS_PROFILE is required (set in backtest.env)}"
: "${BACKTEST_CONFIG:?BACKTEST_CONFIG is required (set in backtest.env)}"

[[ -f "${BACKTEST_CONFIG}" ]] \
    || die "BACKTEST_CONFIG file not found: ${BACKTEST_CONFIG}"

# ── AWS CLI helper (injects endpoint for Wasabi if set) ───────────────────────
aws_cmd() {
    local endpoint_flag=""
    [[ -n "${AWS_ENDPOINT_URL:-}" ]] && endpoint_flag="--endpoint-url ${AWS_ENDPOINT_URL}"
    # shellcheck disable=SC2086
    aws --profile "${AWS_PROFILE}" ${endpoint_flag} "$@"
}

# ── 1. Download kline cache ───────────────────────────────────────────────────
log "=== Stage 1: Download kline cache ==="
mkdir -p db

# Only klines are downloaded — backtest.duckdb is generated fresh each run
# and uploaded under a run-specific name so the master DB is never overwritten.
IFS=' ' read -ra _DB_LIST <<< "${DB_FILES:-backtest_klines.duckdb}"
for dbfile in "${_DB_LIST[@]}"; do
    local_path="db/${dbfile}"
    remote_path="${S3_BUCKET}/db/${dbfile}"

    if [[ "${SKIP_DOWNLOAD_IF_EXISTS:-true}" == "true" && -f "${local_path}" ]]; then
        log "  Skipping ${dbfile} (already present on bind-mounted volume)"
        continue
    fi

    log "  Downloading ${remote_path} → ${local_path} ..."
    aws_cmd s3 cp "${remote_path}" "${local_path}" \
        || die "Failed to download ${remote_path}. " \
               "Check S3_BUCKET, AWS_PROFILE, and that the file was uploaded."
    log "  Done: $(du -sh "${local_path}" | cut -f1)"
done

# ── 2. Run backtest ───────────────────────────────────────────────────────────
log "=== Stage 2: Backtest ==="

EXTRA_ARGS="${BACKTEST_EXTRA_ARGS:-}"
WORKERS_ARG=""
[[ -n "${BACKTEST_WORKERS:-}" ]] && WORKERS_ARG="--workers ${BACKTEST_WORKERS}"

log "  Config : ${BACKTEST_CONFIG}"
[[ -n "${EXTRA_ARGS}" ]] && log "  Extra  : ${EXTRA_ARGS}"

# Timestamp before run so we can find the newly created results dir
touch /tmp/_run_start_marker

# shellcheck disable=SC2086
python -m backtest.run \
    --config "${BACKTEST_CONFIG}" \
    --no-viz \
    ${WORKERS_ARG} \
    ${EXTRA_ARGS} \
    2>&1 | tee /tmp/backtest_run.log

log "  Backtest finished."

# Find the results dir created after the run started.
# Match only timestamp-prefixed dirs (YYYYMMDD_HHMM_*) to avoid picking up
# the checkpoints/ subdirectory whose mtime is also updated during the run.
RESULTS_DIR=$(find backtest/results -mindepth 1 -maxdepth 1 -type d \
    -newer /tmp/_run_start_marker \
    -name "[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_*" \
    2>/dev/null \
    | xargs ls -dt 2>/dev/null | head -1)

# Fallback: newest timestamp-prefixed dir by mtime
[[ -z "${RESULTS_DIR}" ]] && \
    RESULTS_DIR=$(ls -td backtest/results/[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_*/ 2>/dev/null | head -1)

[[ -n "${RESULTS_DIR}" ]] || die "Could not locate results directory after backtest run."
RESULTS_DIR="${RESULTS_DIR%/}"
RUN_TAG=$(basename "${RESULTS_DIR}")
log "  Results dir : ${RESULTS_DIR}"
log "  Run tag     : ${RUN_TAG}"

# ── 3. Audit top-N combos per stock ──────────────────────────────────────────
log "=== Stage 3: Audit ==="
AUDIT_N="${AUDIT_TOP_N:-3}"
AUDIT_DIR="${RESULTS_DIR}/audit"
mkdir -p "${AUDIT_DIR}"

for csv in "${RESULTS_DIR}"/*/results_*.csv; do
    [[ -f "${csv}" ]] || continue
    code_slug=$(basename "$(dirname "${csv}")")
    log "  Auditing ${code_slug} (top ${AUDIT_N} combos) ..."
    for rank in $(seq 1 "${AUDIT_N}"); do
        python -m backtest.audit \
            --from-csv "${csv}" \
            --rank "${rank}" \
            --out-dir "${AUDIT_DIR}" \
            2>&1 | tail -3 \
            || log "  WARN: audit rank=${rank} failed for ${code_slug}"
    done
done

# ── 4. FVG inspect top combo per stock ───────────────────────────────────────
log "=== Stage 4: FVG Inspect ==="
INSPECT_DIR="${RESULTS_DIR}/inspect"
mkdir -p "${INSPECT_DIR}"

# Derive inspect window from config if not set
if [[ -z "${INSPECT_START:-}" || -z "${INSPECT_END:-}" ]]; then
    CFG_START=$(python -c "
import json, sys
cfg = json.load(open('${BACKTEST_CONFIG}'))
print(cfg.get('start') or cfg.get('start_date',''))
" 2>/dev/null)
    CFG_END=$(python -c "
import json, sys
cfg = json.load(open('${BACKTEST_CONFIG}'))
print(cfg.get('end') or cfg.get('end_date',''))
" 2>/dev/null)
    INSPECT_START="${INSPECT_START:-${CFG_START}}"
    INSPECT_END="${INSPECT_END:-${CFG_END}}"
fi

[[ -n "${INSPECT_START}" && -n "${INSPECT_END}" ]] \
    || die "Cannot determine inspect window. Set INSPECT_START/INSPECT_END in backtest.env."

log "  Window: ${INSPECT_START} → ${INSPECT_END}"

for csv in "${RESULTS_DIR}"/*/results_*.csv; do
    [[ -f "${csv}" ]] || continue
    code_slug=$(basename "$(dirname "${csv}")")
    log "  Inspecting ${code_slug} (rank 1) ..."
    python -m backtest.fvg_inspect \
        --from-csv "${csv}" \
        --rank 1 \
        --inspect-start "${INSPECT_START}" \
        --inspect-end   "${INSPECT_END}" \
        --out-dir "${INSPECT_DIR}" \
        2>&1 | tail -3 \
        || log "  WARN: inspect failed for ${code_slug}"
done

# ── 5. Upload results + run-specific DB ──────────────────────────────────────

# Always create a run-tagged local copy on the bind-mounted volume so the file
# is identifiable even when S3 upload is disabled or when inspecting docker_db/.
LOCAL_RUN_DB="db/backtest_${RUN_TAG}.duckdb"
if [[ -f "db/backtest.duckdb" ]]; then
    cp "db/backtest.duckdb" "${LOCAL_RUN_DB}"
    log "  Run-tagged local DB : ${LOCAL_RUN_DB}"
else
    log "  WARN: db/backtest.duckdb not found — skipping local run-tag copy"
fi

if [[ "${UPLOAD_RESULTS:-true}" == "true" ]]; then
    log "=== Stage 5: Upload ==="
    S3_RESULTS="${S3_BUCKET}/${RESULTS_S3_PREFIX:-results}/${RUN_TAG}/"
    S3_DB_DEST="${S3_BUCKET}/db/runs/backtest_${RUN_TAG}.duckdb"

    # 5a. Upload HTML reports, CSVs, logs (exclude checkpoints / pickle files)
    log "  Reports → ${S3_RESULTS}"
    aws_cmd s3 sync "${RESULTS_DIR}/" "${S3_RESULTS}" \
        --exclude "*.pkl" \
        --exclude "checkpoints/*"

    # 5b. Upload run-specific backtest DB (NOT the master — avoids overwriting)
    #     Local clients merge with: uv run backtest/merge_db.py --s3 <S3_DB_DEST>
    if [[ -f "${LOCAL_RUN_DB}" ]]; then
        log "  Run DB  → ${S3_DB_DEST}"
        aws_cmd s3 cp "${LOCAL_RUN_DB}" "${S3_DB_DEST}"
        log "  Merge locally with:"
        log "    uv run backtest/merge_db.py --s3 ${S3_DB_DEST}"
    else
        log "  WARN: ${LOCAL_RUN_DB} not found — skipping DB upload"
    fi
else
    log "=== Stage 5: Upload skipped (UPLOAD_RESULTS != true) ==="
fi

log "=== Pipeline complete ==="
log "Run tag : ${RUN_TAG}"
log "Results : ${S3_BUCKET}/${RESULTS_S3_PREFIX:-results}/${RUN_TAG}/"
