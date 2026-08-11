#!/usr/bin/env bash
set -euo pipefail

DRY_CHECK=0
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --dry-check)
      DRY_CHECK=1
      shift
      ;;
    *)
      echo "Usage: $0 [--dry-check]" >&2
      exit 2
      ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

REV_LABEL="${REV_LABEL:-final_pachka_$(git rev-parse --short HEAD)}"
SCEN="${SCEN:-product_data/telegram_dynamic_test_sets/adr003_final_m1_pachka_99b8169a_20260707.jsonl}"
SNAPSHOT="${SNAPSHOT:-product_data/knowledge_base/kb_release_20260612_v6_7_staging_r4_1/kb_release_v3_snapshot.json}"
RUN_ORDER="${RUN_ORDER:-ON_FIRST}"
if [[ "$RUN_ORDER" != "B_FIRST" && "$RUN_ORDER" != "ON_FIRST" ]]; then
  echo "RUN_ORDER must be B_FIRST or ON_FIRST, got: $RUN_ORDER" >&2
  exit 2
fi

OLD_READING_CLASSES="sense_seats,slots_gsf,off_topic,intent_actions,live_status_read"
TARGET_READING_CLASSES="fact_select_read"
NEW_FLAGS=(
  TELEGRAM_FACT_SELECT_FRAME
  TELEGRAM_TONE_CLOSE_FRAME_VETO
  TELEGRAM_P0_MODEL_LED
  TELEGRAM_PROSE_MODEL_LED
  TELEGRAM_PAYMENT_REFUND_DISPUTE_SPLIT
  TELEGRAM_SEATS_DEFAULT_OPEN
  TELEGRAM_P0_LATCH_AUTORELEASE_V2
)

if [[ "$DRY_CHECK" == "1" ]]; then
  OUT="${OUT:-runs/adr003_final_pachka_dry_${REV_LABEL}}"
else
  OUT="${OUT:-runs/adr003_final_pachka_pair_${REV_LABEL}}"
fi

COMMON=(
  --scenarios "$SCEN"
  --snapshot "$SNAPSHOT"
  --client-mode scripted
  --parallel 4
  --judge-prompt-version v9.1
)
if [[ "$DRY_CHECK" == "1" ]]; then
  COMMON+=(--limit 2)
fi

base_unsets=(
  env
  -u TELEGRAM_SEMANTIC_FRAME_POSTHOC_SHADOW
  -u TELEGRAM_SEMANTIC_FRAME_DECISION_SHADOW
  -u TELEGRAM_SEMANTIC_FRAME_MANAGER_ACTION_GATE
  -u TELEGRAM_SEMANTIC_FRAME_SELF_ANSWER_SHADOW
  -u TELEGRAM_SEMANTIC_FRAME_EXISTENCE_PROOF_SHADOW
  -u TELEGRAM_SEMANTIC_FRAME_PROOF_RECONCILIATION_SHADOW
)

base_assignments=(
  TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1
  TELEGRAM_DIRECT_PATH=1
  TELEGRAM_BOT_GOLD_REAL=1
  TELEGRAM_TEMPLATE_FROM_KB=1
  TELEGRAM_ROUTE_RUBRIC=1
  TELEGRAM_LLM_RETRIEVE=1
  TELEGRAM_SEMANTIC_FRAME_SHADOW=1
  PYTHONPATH="$ROOT/src"
)

on_env=(
  "${base_unsets[@]}"
)
for flag in "${NEW_FLAGS[@]}"; do
  on_env+=(-u "$flag")
done
on_env+=(
  -u TELEGRAM_SEMANTIC_READING_CLASSES
  "${base_assignments[@]}"
)

b_env=(
  "${base_unsets[@]}"
  "${base_assignments[@]}"
  TELEGRAM_SEMANTIC_READING_CLASSES="$OLD_READING_CLASSES"
)
for flag in "${NEW_FLAGS[@]}"; do
  b_env+=("$flag=0")
done

validate_leg() {
  local leg="$1"
  local mode="$2"
  if [[ "$mode" == "require" ]]; then
    PYTHONPATH="$ROOT/src" python3 scripts/validate_adr003_e3_leg.py \
      --summary "$OUT/$leg/dynamic_summary.json" \
      --transcripts "$OUT/$leg/dynamic_dialog_transcripts.jsonl" \
      --leg "$leg" \
      --expect-trace \
      --require-trace-class "$TARGET_READING_CLASSES" \
      --out-json "$OUT/$leg/e3_validation.json"
  else
    PYTHONPATH="$ROOT/src" python3 scripts/validate_adr003_e3_leg.py \
      --summary "$OUT/$leg/dynamic_summary.json" \
      --transcripts "$OUT/$leg/dynamic_dialog_transcripts.jsonl" \
      --leg "$leg" \
      --forbid-trace-class "$TARGET_READING_CLASSES" \
      --out-json "$OUT/$leg/e3_validation.json"
  fi
}

write_env_contract() {
  python3 - "$OUT" "$OLD_READING_CLASSES" "$TARGET_READING_CLASSES" "$RUN_ORDER" "${NEW_FLAGS[@]}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

out = Path(sys.argv[1])
old_reading = sys.argv[2]
target_reading = sys.argv[3]
run_order = sys.argv[4]
flags = sys.argv[5:]
contract = {
    "schema_version": "adr003_final_pachka_env_contract_v1_2026_07_07",
    "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    "run_order": run_order,
    "B": {
        "meaning": "old pilot profile emulated by explicit overrides",
        "TELEGRAM_SEMANTIC_READING_CLASSES": old_reading,
        "explicit_zero_flags": {flag: "0" for flag in flags},
        "forbid_trace_classes": target_reading,
    },
    "ON": {
        "meaning": "current HEAD pilot profile as-is; no manual env for target flags",
        "unset_flags": flags + ["TELEGRAM_SEMANTIC_READING_CLASSES"],
        "require_trace_classes": target_reading,
    },
}
out.mkdir(parents=True, exist_ok=True)
(out / "env_contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

write_sha_manifest() {
  python3 - "$OUT" "$SCEN" "$SNAPSHOT" "$ROOT/scripts/run_adr003_final_pachka_pair.sh" "$ROOT" <<'PY'
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

out_dir = Path(sys.argv[1])
root = Path(sys.argv[5]).resolve()
paths = {
    "scenario": Path(sys.argv[2]),
    "snapshot": Path(sys.argv[3]),
    "runner": Path(sys.argv[4]),
    "env_contract": out_dir / "env_contract.json",
}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def display_path(path: Path) -> str:
    resolved = path.resolve(strict=False)
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(path)

artifact_paths = {
    "B_transcripts": out_dir / "B" / "dynamic_dialog_transcripts.jsonl",
    "B_summary": out_dir / "B" / "dynamic_summary.json",
    "B_validation": out_dir / "B" / "e3_validation.json",
    "ON_transcripts": out_dir / "ON" / "dynamic_dialog_transcripts.jsonl",
    "ON_summary": out_dir / "ON" / "dynamic_summary.json",
    "ON_validation": out_dir / "ON" / "e3_validation.json",
    "REPORT_json": out_dir / "REPORT" / "adr003_semantic_frame_eval_report.json",
    "REPORT_markdown": out_dir / "REPORT" / "adr003_semantic_frame_eval_report.md",
}
artifacts = {
    name: {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
    for name, path in artifact_paths.items()
    if path.is_file()
}
manifest = {
    "schema_version": "adr003_final_pachka_pair_manifest_v1_2026_07_07",
    "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
    "paths": {name: display_path(path) for name, path in paths.items()},
    "sha256": {name: sha256(path) for name, path in paths.items() if path.is_file()},
    "artifacts": artifacts,
}
(out_dir / "sha_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"sha_manifest={out_dir / 'sha_manifest.json'}")
PY
}

run_on_leg() {
  echo "== ON: current HEAD profile as-is =="
  "${on_env[@]}" python3 scripts/run_telegram_dynamic_client_sim.py "${COMMON[@]}" \
    --out-dir "$OUT/ON" \
    --progress-json "$OUT/ON/progress.json" \
    --progress-leg ON
  validate_leg ON require
}

run_b_leg() {
  echo "== B: old pilot profile via explicit overrides =="
  "${b_env[@]}" python3 scripts/run_telegram_dynamic_client_sim.py "${COMMON[@]}" \
    --allow-non-pilot-profile \
    --out-dir "$OUT/B" \
    --progress-json "$OUT/B/progress.json" \
    --progress-leg B
  validate_leg B forbid
}

run_report() {
  echo "== Report =="
  PYTHONPATH="$ROOT/src" python3 scripts/report_adr003_semantic_frame_eval.py \
    --off-transcripts "$OUT/B/dynamic_dialog_transcripts.jsonl" \
    --off-summary "$OUT/B/dynamic_summary.json" \
    --on-transcripts "$OUT/ON/dynamic_dialog_transcripts.jsonl" \
    --on-summary "$OUT/ON/dynamic_summary.json" \
    --out-dir "$OUT/REPORT"
}

mkdir -p "$OUT"
write_env_contract

echo "ADR003 final pachka pair"
echo "rev=$(git rev-parse HEAD)"
echo "scenarios=$SCEN"
echo "snapshot=$SNAPSHOT"
echo "target_reading_classes=$TARGET_READING_CLASSES"
echo "old_reading_classes=$OLD_READING_CLASSES"
echo "run_order=$RUN_ORDER"
echo "out=$OUT"
if [[ "$DRY_CHECK" == "1" ]]; then
  echo "mode=dry-check limit=2"
fi

if [[ "$RUN_ORDER" == "ON_FIRST" ]]; then
  run_on_leg
  run_b_leg
else
  run_b_leg
  run_on_leg
fi

if [[ "$DRY_CHECK" == "1" ]]; then
  write_sha_manifest
  echo "Dry check passed: $OUT"
  exit 0
fi

run_report
write_sha_manifest
echo "Done: $OUT"
