#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -f "deploy/.env.deploy" ]]; then
  set -a
  source "deploy/.env.deploy"
  set +a
fi

COMPOSE_FILE=""
if [[ -f "docker-compose.yml" ]]; then
  COMPOSE_FILE="docker-compose.yml"
elif [[ -f "compose.yml" ]]; then
  COMPOSE_FILE="compose.yml"
fi

if [[ -n "${COMPOSE_FILE}" ]]; then
  echo "[deploy] using compose file: ${COMPOSE_FILE}"
  docker compose -f "${COMPOSE_FILE}" pull || true
  docker compose -f "${COMPOSE_FILE}" up -d --build
  docker image prune -f || true
  echo "[deploy] done"
  exit 0
fi

echo "[deploy] no docker compose file found."
echo "[deploy] customize deploy/server_rebuild.sh for your stack."
exit 0
