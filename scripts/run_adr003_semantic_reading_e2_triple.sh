#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

REV_LABEL="${REV_LABEL:-postE1_$(git rev-parse --short HEAD)}"
SCEN="${SCEN:-product_data/telegram_dynamic_test_sets/adr003_semantic_reading_paket1_e2_20260703.jsonl}"
OUT="${OUT:-runs/adr003_semantic_reading_e2_triple_${REV_LABEL}}"

COMMON=(
  --scenarios "$SCEN"
  --client-mode scripted
  --parallel 4
  --judge-prompt-version v9.1
)

clean_semantic_env=(
  env
  -u TELEGRAM_SEMANTIC_FRAME_SHADOW
  -u TELEGRAM_SEMANTIC_FRAME_POSTHOC_SHADOW
  -u TELEGRAM_SEMANTIC_FRAME_DECISION_SHADOW
  -u TELEGRAM_SEMANTIC_FRAME_MANAGER_ACTION_GATE
  -u TELEGRAM_SEMANTIC_FRAME_SELF_ANSWER_SHADOW
  -u TELEGRAM_SEMANTIC_FRAME_EXISTENCE_PROOF_SHADOW
  -u TELEGRAM_SEMANTIC_FRAME_PROOF_RECONCILIATION_SHADOW
  TELEGRAM_SEMANTIC_READING_CLASSES=
  PYTHONPATH="$ROOT/src"
)

mkdir -p "$OUT"

echo "ADR003 E2 triple"
echo "rev=$(git rev-parse HEAD)"
echo "scenarios=$SCEN"
echo "out=$OUT"

echo "== B: baseline, semantic frame shadow OFF =="
"${clean_semantic_env[@]}" \
  python3 scripts/run_telegram_dynamic_client_sim.py "${COMMON[@]}" \
    --out-dir "$OUT/B"

echo "== I: inline SemanticFrame shadow ON, semantic reading masks empty =="
"${clean_semantic_env[@]}" \
  TELEGRAM_SEMANTIC_FRAME_SHADOW=1 \
  python3 scripts/run_telegram_dynamic_client_sim.py "${COMMON[@]}" \
    --out-dir "$OUT/I"

echo "== P: post-hoc SemanticFrame enrich over B =="
"${clean_semantic_env[@]}" \
  python3 scripts/run_telegram_dynamic_client_sim.py \
    --scenarios "$SCEN" \
    --semantic-frame-enrich-from "$OUT/B/dynamic_dialog_transcripts.jsonl" \
    --client-mode scripted \
    --judge-mode fake \
    --memory-mode fake \
    --parallel 4 \
    --out-dir "$OUT/P"

echo "== Report =="
PYTHONPATH="$ROOT/src" python3 scripts/report_adr003_semantic_frame_eval.py \
  --off-transcripts "$OUT/B/dynamic_dialog_transcripts.jsonl" \
  --off-summary "$OUT/B/dynamic_summary.json" \
  --on-transcripts "$OUT/I/dynamic_dialog_transcripts.jsonl" \
  --on-summary "$OUT/I/dynamic_summary.json" \
  --posthoc-transcripts "$OUT/P/dynamic_dialog_transcripts.jsonl" \
  --out-dir "$OUT/REPORT"

echo "Done: $OUT"
