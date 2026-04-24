#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/dmitrijfabarisov/Projects/Mango analyse"
PY="$ROOT/stable_runtime/venv_stable/bin/python"
CLI="$ROOT/stable_runtime/run-cli.sh"
RUN_ID="new_merge_benchmark_500x2_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$ROOT/stable_runtime/benchmarks/$RUN_ID"
LOG="$OUT_DIR/run.log"
MANIFEST="$OUT_DIR/selection_manifest.json"
SRC_DIR="$ROOT/2026-03-09--26"
KNOWN_DB="$ROOT/mango_mvp.db"
MINI_BATCH_DIR="$OUT_DIR/batches/mini_500"
FULL_BATCH_DIR="$OUT_DIR/batches/full_500"
MINI_DB="$OUT_DIR/mini_500.db"
FULL_DB="$OUT_DIR/full_500.db"
MINI_TRANSCRIPTS="$OUT_DIR/mini_transcripts"
FULL_TRANSCRIPTS="$OUT_DIR/full_transcripts"

mkdir -p "$OUT_DIR"
exec > >(tee -a "$LOG") 2>&1

echo "[run] run_id=$RUN_ID"
echo "[run] out_dir=$OUT_DIR"
echo "[run] prepare batches"
"$PY" "$ROOT/scripts/prepare_untranscribed_merge_batches.py" \
  --source-dir "$SRC_DIR" \
  --known-db "$KNOWN_DB" \
  --out-root "$OUT_DIR/batches" \
  --batch-size 500 >"$MANIFEST"
echo "[run] selection manifest: $MANIFEST"

for MODEL in "gpt-5.4-mini" "gpt-5.4"; do
  if [[ "$MODEL" == "gpt-5.4-mini" ]]; then
    DB="$MINI_DB"
    BATCH_DIR="$MINI_BATCH_DIR"
    TRANSCRIPTS_DIR="$MINI_TRANSCRIPTS"
    SUMMARY_OUT="$OUT_DIR/mini_summary.json"
  else
    DB="$FULL_DB"
    BATCH_DIR="$FULL_BATCH_DIR"
    TRANSCRIPTS_DIR="$FULL_TRANSCRIPTS"
    SUMMARY_OUT="$OUT_DIR/full_summary.json"
  fi

  mkdir -p "$TRANSCRIPTS_DIR"
  export DATABASE_URL="sqlite:////${DB#/}"
  export TRANSCRIBE_PROVIDER="mlx"
  export DUAL_TRANSCRIBE_ENABLED="1"
  export SECONDARY_TRANSCRIBE_PROVIDER="gigaam"
  export DUAL_MERGE_PROVIDER="codex_cli"
  export CODEX_TRANSCRIBE_MODEL="$MODEL"
  export CODEX_CLI_COMMAND="/Applications/Codex.app/Contents/Resources/codex"
  export CODEX_CLI_TIMEOUT_SEC="180"
  export CODEX_REASONING_EFFORT="medium"
  export TRANSCRIPT_EXPORT_DIR="$TRANSCRIPTS_DIR"
  export MONO_ROLE_ASSIGNMENT_MODE="rule"
  export LLM_CACHE_ENABLED="0"

  echo "[run] model=$MODEL init-db"
  rm -f "$DB"
  "$CLI" init-db
  echo "[run] model=$MODEL ingest"
  "$CLI" ingest --recordings-dir "$BATCH_DIR"
  echo "[run] model=$MODEL worker transcribe+backfill"
  "$CLI" worker --stage-limit 20 --stages transcribe,backfill-second-asr --poll-sec 10 --max-idle-cycles 30
  echo "[run] model=$MODEL summarize"
  "$PY" "$ROOT/scripts/summarize_merge_usage.py" --db "$DB" --out "$SUMMARY_OUT" --model "$MODEL"
done

echo "[run] build comparison"
"$PY" - "$OUT_DIR" <<'PY'
from __future__ import annotations
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
comparison = {
    "run_id": out_dir.name,
    "mini": json.loads((out_dir / "mini_summary.json").read_text(encoding="utf-8")),
    "full": json.loads((out_dir / "full_summary.json").read_text(encoding="utf-8")),
}
(out_dir / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(comparison, ensure_ascii=False, indent=2))
PY

echo "[run] comparison=$OUT_DIR/comparison.json"
