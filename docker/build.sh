#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# build.sh — Build the moomoo-backtest Docker image
#
# Usage:
#   bash docker/build.sh              # builds moomoo-backtest:latest
#   bash docker/build.sh 0.3.1        # also tags as moomoo-backtest:0.3.1
#   bash docker/build.sh --no-cache   # force clean build
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# Always run from project root regardless of where the script is invoked from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

IMAGE_NAME="backtest"
TAG="latest"
NO_CACHE=""

# Parse args
for arg in "$@"; do
    case "${arg}" in
        --no-cache) NO_CACHE="--no-cache" ;;
        -*)         echo "Unknown flag: ${arg}"; exit 1 ;;
        *)          TAG="${arg}" ;;
    esac
done

# Read system version from VERSION file
SYS_VERSION="$(cat VERSION 2>/dev/null || echo "unknown")"

echo "Building ${IMAGE_NAME}:${TAG}  (system version ${SYS_VERSION})"
echo "Project root: ${PROJECT_ROOT}"
echo ""

# shellcheck disable=SC2086
docker build \
    ${NO_CACHE} \
    --file docker/Dockerfile \
    --tag "${IMAGE_NAME}:${TAG}" \
    --build-arg SYS_VERSION="${SYS_VERSION}" \
    --label "org.opencontainers.image.version=${SYS_VERSION}" \
    --label "org.opencontainers.image.source=moomoo-backtest" \
    .

# Also tag as explicit version if TAG != latest
if [[ "${TAG}" != "latest" ]]; then
    docker tag "${IMAGE_NAME}:${TAG}" "${IMAGE_NAME}:latest"
    echo "Tagged: ${IMAGE_NAME}:${TAG}  and  ${IMAGE_NAME}:latest"
else
    echo "Tagged: ${IMAGE_NAME}:latest"
fi
