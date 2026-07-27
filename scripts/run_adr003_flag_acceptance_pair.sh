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

TARGET_FLAG="${TARGET_FLAG:-}"
TARGET_FLAG_VALUE="${TARGET_FLAG_VALUE:-1}"
case "$TARGET_FLAG" in
  TELEGRAM_TEXT_HYGIENE_PAYMENT_FIX|TELEGRAM_DIALOG_SUMMARY_ROLLING|TELEGRAM_INTENT_MODEL_LED)
    ;;
  *)
    echo "TARGET_FLAG must be one of TELEGRAM_TEXT_HYGIENE_PAYMENT_FIX, TELEGRAM_DIALOG_SUMMARY_ROLLING, TELEGRAM_INTENT_MODEL_LED" >&2
    exit 2
    ;;
esac

PACKAGE_ID="${PACKAGE_ID:-adr003_flag_acceptance}"
REV_LABEL="${REV_LABEL:-${PACKAGE_ID}_$(git rev-parse --short HEAD)}"
SCEN="${SCEN:-}"
SNAPSHOT="${SNAPSHOT:-product_data/knowledge_base/kb_release_20260612_v6_7_staging_r4_1/kb_release_v3_snapshot.json}"
if [[ -z "$SCEN" ]]; then
  echo "SCEN is required" >&2
  exit 2
fi
if [[ "$DRY_CHECK" == "1" ]]; then
  OUT="${OUT:-runs/${REV_LABEL}_dry_check}"
else
  OUT="${OUT:-runs/${REV_LABEL}}"
fi

COMMON=(
  --scenarios "$SCEN"
  --snapshot "$SNAPSHOT"
  --client-mode scripted
  --parallel 4
  --judge-prompt-version v9.1
  --allow-non-pilot-profile
)
if [[ "$DRY_CHECK" == "1" ]]; then
  COMMON+=(--limit 2)
fi

base_env=(
  env
  -u TELEGRAM_TEXT_HYGIENE_PAYMENT_FIX
  -u TELEGRAM_DIALOG_SUMMARY_ROLLING
  -u TELEGRAM_INTENT_MODEL_LED
  -u TELEGRAM_SEMANTIC_READING_CLASSES
  -u TELEGRAM_SEMANTIC_FRAME_POSTHOC_SHADOW
  -u TELEGRAM_SEMANTIC_FRAME_DECISION_SHADOW
  -u TELEGRAM_SEMANTIC_FRAME_MANAGER_ACTION_GATE
  -u TELEGRAM_SEMANTIC_FRAME_SELF_ANSWER_SHADOW
  -u TELEGRAM_SEMANTIC_FRAME_EXISTENCE_PROOF_SHADOW
  -u TELEGRAM_SEMANTIC_FRAME_PROOF_RECONCILIATION_SHADOW
  TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1
  PYTHONDONTWRITEBYTECODE=1
  PYTHONPATH="$ROOT/src"
)

package_flags_off=(
  TELEGRAM_TEXT_HYGIENE_PAYMENT_FIX=0
  TELEGRAM_DIALOG_SUMMARY_ROLLING=0
  TELEGRAM_INTENT_MODEL_LED=0
)

validate_leg() {
  local leg="$1"
  PYTHONPATH="$ROOT/src" python3 scripts/validate_adr003_e3_leg.py \
    --summary "$OUT/$leg/dynamic_summary.json" \
    --transcripts "$OUT/$leg/dynamic_dialog_transcripts.jsonl" \
    --leg "$leg" \
    --expect-trace \
    --out-json "$OUT/$leg/e3_validation.json"
}

write_sha_manifest() {
  python3 - "$OUT" "$SCEN" "$SNAPSHOT" "$ROOT/scripts/run_adr003_flag_acceptance_pair.sh" "$PACKAGE_ID" "$TARGET_FLAG" "$TARGET_FLAG_VALUE" "$ROOT" <<'PY'
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

out_dir = Path(sys.argv[1])
root = Path(sys.argv[8]).resolve()
paths = {
    "scenario": Path(sys.argv[2]),
    "snapshot": Path(sys.argv[3]),
    "runner": Path(sys.argv[4]),
}
package_id = sys.argv[5]
target_flag = sys.argv[6]
target_flag_value = sys.argv[7]

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
    "B_progress": out_dir / "B" / "progress.json",
    "B_validation": out_dir / "B" / "e3_validation.json",
    "ON_transcripts": out_dir / "ON" / "dynamic_dialog_transcripts.jsonl",
    "ON_summary": out_dir / "ON" / "dynamic_summary.json",
    "ON_progress": out_dir / "ON" / "progress.json",
    "ON_validation": out_dir / "ON" / "e3_validation.json",
    "REPORT_json": out_dir / "REPORT" / "adr003_semantic_frame_eval_report.json",
    "REPORT_markdown": out_dir / "REPORT" / "adr003_semantic_frame_eval_report.md",
}
artifacts = {
    name: {
        "path": str(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }
    for name, path in artifact_paths.items()
    if path.is_file()
}

manifest = {
    "schema_version": "adr003_flag_acceptance_pair_v1_2026_07_04",
    "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
    "package_id": package_id,
    "target_flag": target_flag,
    "target_flag_value": target_flag_value,
    "paths": {name: display_path(path) for name, path in paths.items()},
    "sha256": {name: sha256(path) for name, path in paths.items()},
    "artifacts": artifacts,
    "env_contract": {
        "B": "pilot_gold_v1 profile with package flags explicitly set to 0",
        "ON": f"pilot_gold_v1 profile with package flags explicitly set to 0, plus {target_flag}={target_flag_value}",
        "TELEGRAM_SEMANTIC_READING_CLASSES": "unset in process env so pilot profile default classes apply",
    },
}
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "sha_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(f"sha_manifest={out_dir / 'sha_manifest.json'}")
PY
}

run_leg() {
  local leg="$1"
  shift
  echo "== $leg =="
  "$@" \
    python3 scripts/run_telegram_dynamic_client_sim.py "${COMMON[@]}" \
      --out-dir "$OUT/$leg" \
      --progress-json "$OUT/$leg/progress.json" \
      --progress-leg "$leg"
  validate_leg "$leg"
}

run_report() {
  echo "== Report =="
  local extra_args=()
  if [[ "$TARGET_FLAG" == "TELEGRAM_INTENT_MODEL_LED" ]]; then
    extra_args+=(--require-intent-model-led-application)
  fi
  PYTHONPATH="$ROOT/src" python3 scripts/report_adr003_semantic_frame_eval.py \
    --off-transcripts "$OUT/B/dynamic_dialog_transcripts.jsonl" \
    --off-summary "$OUT/B/dynamic_summary.json" \
    --on-transcripts "$OUT/ON/dynamic_dialog_transcripts.jsonl" \
    --on-summary "$OUT/ON/dynamic_summary.json" \
    --out-dir "$OUT/REPORT" \
    "${extra_args[@]}"
}

mkdir -p "$OUT"

echo "ADR003 flag acceptance pair"
echo "rev=$(git rev-parse HEAD)"
echo "package_id=$PACKAGE_ID"
echo "target_flag=$TARGET_FLAG"
echo "scenarios=$SCEN"
echo "snapshot=$SNAPSHOT"
echo "out=$OUT"
if [[ "$DRY_CHECK" == "1" ]]; then
  echo "mode=dry-check limit=2"
fi

run_leg B "${base_env[@]}" "${package_flags_off[@]}"
run_leg ON "${base_env[@]}" "${package_flags_off[@]}" "$TARGET_FLAG=$TARGET_FLAG_VALUE"

if [[ "$DRY_CHECK" == "1" ]]; then
  write_sha_manifest
  echo "Dry check passed: $OUT"
  exit 0
fi

report_rc=0
run_report || report_rc=$?
write_sha_manifest
echo "Done: $OUT"
exit "$report_rc"
