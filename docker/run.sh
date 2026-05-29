#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# run.sh — Run the moomoo-backtest Docker container
#
# Prerequisites:
#   1. Build the image:   bash docker/build.sh
#   2. Create config:     cp docker/backtest.env.example docker/backtest.env
#                         (then edit docker/backtest.env)
#   3. AWS credentials:   ~/.aws/credentials must have the profile named in
#                         AWS_PROFILE.  For Wasabi, set AWS_ENDPOINT_URL in
#                         backtest.env and configure ~/.aws/credentials as:
#
#                           [myprofile]
#                           aws_access_key_id     = WASABI_KEY
#                           aws_secret_access_key = WASABI_SECRET
#
# Usage:
#   bash docker/run.sh                   # interactive (logs to stdout)
#   bash docker/run.sh --detach          # background; docker logs to follow
#   bash docker/run.sh --dry-run         # print docker run command, don't execute
#   bash docker/run.sh --shell           # open bash inside the container (debug)
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE_NAME="moomoo-backtest:latest"
ENV_FILE="${SCRIPT_DIR}/backtest.env"
DETACH=""
DRY_RUN=false
SHELL_MODE=false

# ── Parse args ────────────────────────────────────────────────────────────────
for arg in "$@"; do
    case "${arg}" in
        --detach)   DETACH="--detach" ;;
        --dry-run)  DRY_RUN=true ;;
        --shell)    SHELL_MODE=true ;;
        *)          echo "Unknown argument: ${arg}"; exit 1 ;;
    esac
done

# ── Validate env file ─────────────────────────────────────────────────────────
if [[ ! -f "${ENV_FILE}" ]]; then
    echo "ERROR: ${ENV_FILE} not found."
    echo "       Copy docker/backtest.env.example to docker/backtest.env and fill in your values."
    exit 1
fi

# ── Directories on the host (created if absent) ───────────────────────────────

# db/  : bind-mounted so the downloaded kline DB persists between runs
#        (avoids re-downloading on every docker run)
DB_HOST="${PROJECT_ROOT}/docker_db"

# results/ : bind-mounted so reports survive container exit
RESULTS_HOST="${PROJECT_ROOT}/backtest_results"

mkdir -p "${DB_HOST}" "${RESULTS_HOST}"

# ── Build docker run command ──────────────────────────────────────────────────
DOCKER_CMD=(
    docker run
    --rm
    --name moomoo-backtest
    ${DETACH}

    # AWS credentials (read-only mount of host ~/.aws)
    --volume "${HOME}/.aws:/root/.aws:ro"

    # Persistent volumes
    --volume "${DB_HOST}:/app/db"
    --volume "${RESULTS_HOST}:/app/backtest/results"

    # Configuration
    --env-file "${ENV_FILE}"

    # Resource limits (adjust for your HPC node)
    # --cpus and --memory can be overridden by env vars CPU_LIMIT / MEM_LIMIT
    ${CPU_LIMIT:+--cpus="${CPU_LIMIT}"}
    ${MEM_LIMIT:+--memory="${MEM_LIMIT}"}

    "${IMAGE_NAME}"
)

if [[ "${SHELL_MODE}" == "true" ]]; then
    DOCKER_CMD=(
        docker run
        --rm -it
        --volume "${HOME}/.aws:/root/.aws:ro"
        --volume "${DB_HOST}:/app/db"
        --volume "${RESULTS_HOST}:/app/backtest/results"
        --env-file "${ENV_FILE}"
        --entrypoint bash
        "${IMAGE_NAME}"
    )
fi

# ── Execute ───────────────────────────────────────────────────────────────────
echo "Host DB dir:      ${DB_HOST}"
echo "Host results dir: ${RESULTS_HOST}"
echo "Env file:         ${ENV_FILE}"
echo ""

if [[ "${DRY_RUN}" == "true" ]]; then
    echo "Dry run — docker command:"
    printf '%s \\\n  ' "${DOCKER_CMD[@]}"
    echo ""
    exit 0
fi

exec "${DOCKER_CMD[@]}"
