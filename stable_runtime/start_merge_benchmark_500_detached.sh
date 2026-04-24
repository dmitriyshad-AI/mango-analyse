#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/dmitrijfabarisov/Projects/Mango analyse"
PY="$ROOT/stable_runtime/venv_stable/bin/python"
RUN_ID="merge_benchmark_500_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$ROOT/stable_runtime/benchmarks/$RUN_ID"
LOG="$OUT_DIR/run.log"
DB="$ROOT/mango_mvp.db"

mkdir -p "$OUT_DIR"

nohup "$PY" "$ROOT/scripts/benchmark_codex_merge.py" \
  --db "$DB" \
  --out-dir "$OUT_DIR" \
  --sample-size 500 \
  --models "gpt-5.4-mini,gpt-5.4" \
  --reasoning medium \
  >"$LOG" 2>&1 &

PID=$!

echo "run_id=$RUN_ID"
echo "pid=$PID"
echo "out_dir=$OUT_DIR"
echo "log=$LOG"
echo "selection=$OUT_DIR/selection.json"
echo "comparison=$OUT_DIR/comparison.json"
