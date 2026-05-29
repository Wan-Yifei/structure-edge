#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# entrypoint.sh — moomoo backtest pipeline
#
# Stages:
#   1. Download database files from S3 / Wasabi
#   2. Run backtest grid (backtest/run.py)
#   3. Generate per-stock audit reports (backtest/audit.py)
#   4. Generate per-stock FVG inspect reports (backtest/fvg_inspect.py)
#   5. Upload results directory back to S3
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

# ── 1. Download databases ─────────────────────────────────────────────────────
log "=== Stage 1: Download databases ==="
mkdir -p db

IFS=' ' read -ra _DB_LIST <<< "${DB_FILES:-backtest_klines.duckdb}"
for dbfile in "${_DB_LIST[@]}"; do
    local_path="db/${dbfile}"
    remote_path="${S3_BUCKET}/db/${dbfile}"

    if [[ "${SKIP_DOWNLOAD_IF_EXISTS:-true}" == "true" && -f "${local_path}" ]]; then
        log "  Skipping ${dbfile} (already exists locally)"
        continue
    fi

    log "  Downloading ${remote_path} → ${local_path} ..."
    aws_cmd s3 cp "${remote_path}" "${local_path}" \
        || die "Failed to download ${remote_path}. " \
               "Check S3_BUCKET, AWS_PROFILE, and that the file exists."
    log "  Done: $(du -sh "${local_path}" | cut -f1)"
done

# ── 2. Run backtest ───────────────────────────────────────────────────────────
log "=== Stage 2: Backtest ==="

EXTRA_ARGS="${BACKTEST_EXTRA_ARGS:-}"
WORKERS_ARG=""
[[ -n "${BACKTEST_WORKERS:-}" ]] && WORKERS_ARG="--workers ${BACKTEST_WORKERS}"

log "  Config: ${BACKTEST_CONFIG}"
log "  Extra args: ${EXTRA_ARGS:-<none>}"

# Record time so we can find the new results dir afterwards
RUN_START=$(date +%s)

# shellcheck disable=SC2086
python -m backtest.run \
    --config "${BACKTEST_CONFIG}" \
    --no-viz \
    ${WORKERS_ARG} \
    ${EXTRA_ARGS} \
    2>&1 | tee /tmp/backtest_run.log

log "  Backtest finished."

# Find the most recently created results subdirectory
RESULTS_DIR=$(find backtest/results -mindepth 1 -maxdepth 1 -type d \
    -newer /tmp/backtest_run.log -o -mindepth 1 -maxdepth 1 -type d \
    | xargs ls -dt 2>/dev/null | head -1)

if [[ -z "${RESULTS_DIR}" ]]; then
    # Fallback: newest by mtime
    RESULTS_DIR=$(ls -td backtest/results/*/ 2>/dev/null | head -1)
fi

[[ -n "${RESULTS_DIR}" ]] || die "Could not locate results directory after backtest run."
RESULTS_DIR="${RESULTS_DIR%/}"  # strip trailing slash
log "  Results dir: ${RESULTS_DIR}"

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

# Derive inspect window from config dates if not explicitly set
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
    || die "INSPECT_START / INSPECT_END could not be determined. " \
           "Set them in backtest.env or ensure the config has 'start'/'end' keys."

log "  Inspect window: ${INSPECT_START} → ${INSPECT_END}"

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

# ── 5. Upload results to S3 ───────────────────────────────────────────────────
if [[ "${UPLOAD_RESULTS:-true}" == "true" ]]; then
    log "=== Stage 5: Upload results ==="
    RUN_TAG=$(basename "${RESULTS_DIR}")
    S3_DEST="${S3_BUCKET}/${RESULTS_S3_PREFIX:-results}/${RUN_TAG}/"
    log "  Syncing ${RESULTS_DIR}/ → ${S3_DEST}"
    aws_cmd s3 sync "${RESULTS_DIR}/" "${S3_DEST}" \
        --exclude "*.pkl" \
        --exclude "checkpoints/*"
    log "  Upload complete: ${S3_DEST}"
else
    log "=== Stage 5: Upload skipped (UPLOAD_RESULTS != true) ==="
fi

log "=== Pipeline complete ==="
log "Results: ${RESULTS_DIR}"
