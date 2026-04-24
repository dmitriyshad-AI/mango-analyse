#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:8010}"
API_KEY="${AI_OFFICE_API_KEY:-ai-office-local-key}"

curl -sS -H "X-API-Key: ${API_KEY}" "${BASE_URL}/api/integrations/amocrm/status"
echo
curl -sS -X POST -H "X-API-Key: ${API_KEY}" "${BASE_URL}/api/integrations/amocrm/refresh"
echo
curl -sS -X POST \
  -H "X-API-Key: ${API_KEY}" \
  -H 'Content-Type: application/json' \
  "${BASE_URL}/api/integrations/amocrm/deals/queue/build" \
  -d '{"days_back":30,"apply_writeback":false}'
echo
