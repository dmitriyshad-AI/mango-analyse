#!/usr/bin/env bash
set -euo pipefail

DRY_CHECK=0
RESUME_ON_REPORT=""
FORCE_RESUME=0
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --dry-check)
      DRY_CHECK=1
      shift
      ;;
    --resume-on-report)
      if [[ -z "${2:-}" ]]; then
        echo "Usage: $0 [--dry-check] [--resume-on-report OUT_DIR] [--force]" >&2
        exit 2
      fi
      RESUME_ON_REPORT="$2"
      shift 2
      ;;
    --force)
      FORCE_RESUME=1
      shift
      ;;
    *)
      echo "Usage: $0 [--dry-check] [--resume-on-report OUT_DIR] [--force]" >&2
      exit 2
      ;;
  esac
done
if [[ "$DRY_CHECK" == "1" && -n "$RESUME_ON_REPORT" ]]; then
  echo "--dry-check and --resume-on-report are mutually exclusive" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

REV_LABEL="${REV_LABEL:-e3_$(git rev-parse --short HEAD)}"
SCEN="${SCEN:-product_data/telegram_dynamic_test_sets/adr003_semantic_reading_paket1_e2_20260703.jsonl}"
SNAPSHOT="${SNAPSHOT:-product_data/knowledge_base/kb_release_20260612_v6_7_staging_r4_1/kb_release_v3_snapshot.json}"
PROFILE_READING_CLASSES="$(
  PYTHONPATH="$ROOT/src" python3 - <<'PY'
from mango_mvp.channels.subscription_llm_parts.semantic_reading import PILOT_PROFILE_DEFAULT_READING_CLASSES
print(PILOT_PROFILE_DEFAULT_READING_CLASSES)
PY
)"
BASE_READING_CLASSES="${READING_CLASSES:-$PROFILE_READING_CLASSES}"
TARGET_READING_CLASS="${TARGET_READING_CLASS:-}"
TARGET_READING_CLASSES="${TARGET_READING_CLASSES:-$TARGET_READING_CLASS}"
normalize_csv() {
  python3 - "$1" <<'PY'
import sys
seen = set()
items = []
for raw in str(sys.argv[1] or "").split(","):
    item = raw.strip()
    if item and item not in seen:
        seen.add(item)
        items.append(item)
print(",".join(items))
PY
}
TARGET_READING_CLASSES="$(normalize_csv "$TARGET_READING_CLASSES")"
RUN_ORDER="${RUN_ORDER:-B_FIRST}"
if [[ "$RUN_ORDER" != "B_FIRST" && "$RUN_ORDER" != "ON_FIRST" ]]; then
  echo "RUN_ORDER must be B_FIRST or ON_FIRST, got: $RUN_ORDER" >&2
  exit 2
fi
READING_CLASSES="$(normalize_csv "$BASE_READING_CLASSES")"
if [[ -n "$TARGET_READING_CLASSES" ]]; then
  IFS=',' read -r -a target_reading_class_items <<< "$TARGET_READING_CLASSES"
  for target_reading_class in "${target_reading_class_items[@]}"; do
    case ",$READING_CLASSES," in
      *",$target_reading_class,"*)
        echo "TARGET_READING_CLASS '$target_reading_class' is already in profile/base READING_CLASSES; this would not be an attributable B/ON target." >&2
        exit 2
        ;;
      *) READING_CLASSES="${READING_CLASSES},${target_reading_class}" ;;
    esac
  done
fi
USE_ON_READING_ENV=0
if [[ -n "$TARGET_READING_CLASSES" ]]; then
  USE_ON_READING_ENV=1
fi
if [[ "$DRY_CHECK" == "1" ]]; then
  OUT="${OUT:-runs/adr003_semantic_reading_e3_dry_check_${REV_LABEL}}"
elif [[ -n "$RESUME_ON_REPORT" ]]; then
  OUT="$RESUME_ON_REPORT"
else
  OUT="${OUT:-runs/adr003_semantic_reading_e3_paired_${REV_LABEL}}"
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

base_env=(
  env
  -u TELEGRAM_SEMANTIC_FRAME_POSTHOC_SHADOW
  -u TELEGRAM_SEMANTIC_FRAME_DECISION_SHADOW
  -u TELEGRAM_SEMANTIC_FRAME_MANAGER_ACTION_GATE
  -u TELEGRAM_SEMANTIC_FRAME_SELF_ANSWER_SHADOW
  -u TELEGRAM_SEMANTIC_FRAME_EXISTENCE_PROOF_SHADOW
  -u TELEGRAM_SEMANTIC_FRAME_PROOF_RECONCILIATION_SHADOW
  -u TELEGRAM_SEMANTIC_READING_CLASSES
  TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1
  TELEGRAM_DIRECT_PATH=1
  TELEGRAM_BOT_GOLD_REAL=1
  TELEGRAM_TEMPLATE_FROM_KB=1
  TELEGRAM_ROUTE_RUBRIC=1
  TELEGRAM_LLM_RETRIEVE=1
  TELEGRAM_SEMANTIC_FRAME_SHADOW=1
  PYTHONPATH="$ROOT/src"
)

validate_leg() {
  local leg="$1"
  local expect_trace="$2"
  if [[ "$expect_trace" == "1" ]]; then
    PYTHONPATH="$ROOT/src" python3 scripts/validate_adr003_e3_leg.py \
      --summary "$OUT/$leg/dynamic_summary.json" \
      --transcripts "$OUT/$leg/dynamic_dialog_transcripts.jsonl" \
      --leg "$leg" \
      --expect-trace \
      --require-trace-class "$TARGET_READING_CLASSES" \
      --out-json "$OUT/$leg/e3_validation.json"
  elif [[ -n "$TARGET_READING_CLASSES" ]]; then
    PYTHONPATH="$ROOT/src" python3 scripts/validate_adr003_e3_leg.py \
      --summary "$OUT/$leg/dynamic_summary.json" \
      --transcripts "$OUT/$leg/dynamic_dialog_transcripts.jsonl" \
      --leg "$leg" \
      --forbid-trace-class "$TARGET_READING_CLASSES" \
      --out-json "$OUT/$leg/e3_validation.json"
  else
    PYTHONPATH="$ROOT/src" python3 scripts/validate_adr003_e3_leg.py \
      --summary "$OUT/$leg/dynamic_summary.json" \
      --transcripts "$OUT/$leg/dynamic_dialog_transcripts.jsonl" \
      --leg "$leg" \
      --out-json "$OUT/$leg/e3_validation.json"
  fi
}

write_sha_manifest() {
  python3 - "$OUT" "$SCEN" "$SNAPSHOT" "$ROOT/scripts/run_adr003_semantic_reading_e3_paired.sh" "$READING_CLASSES" "$TARGET_READING_CLASSES" "$USE_ON_READING_ENV" "$RUN_ORDER" "$ROOT" <<'PY'
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

out_dir = Path(sys.argv[1])
root = Path(sys.argv[9]).resolve()
paths = {
    "scenario": Path(sys.argv[2]),
    "snapshot": Path(sys.argv[3]),
    "runner": Path(sys.argv[4]),
}
reading_classes = sys.argv[5]
target_reading_classes = sys.argv[6]
use_on_reading_env = sys.argv[7] == "1"
run_order = sys.argv[8]

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
    name: {
        "path": str(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }
    for name, path in artifact_paths.items()
    if path.is_file()
}

manifest = {
    "schema_version": "adr003_semantic_reading_e3_paired_v2",
    "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
    "paths": {name: display_path(path) for name, path in paths.items()},
    "sha256": {name: sha256(path) for name, path in paths.items()},
    "artifacts": artifacts,
    "required_env": {
        "TELEGRAM_DIRECT_PATH_PILOT_CONFIG": "pilot_gold_v1",
        "TELEGRAM_DIRECT_PATH": "1",
        "TELEGRAM_BOT_GOLD_REAL": "1",
        "TELEGRAM_TEMPLATE_FROM_KB": "1",
        "TELEGRAM_ROUTE_RUBRIC": "1",
        "TELEGRAM_LLM_RETRIEVE": "1",
        "TELEGRAM_SEMANTIC_FRAME_SHADOW": "1",
        "TELEGRAM_SEMANTIC_READING_CLASSES": reading_classes if use_on_reading_env else "(profile default; env unset)",
        "TARGET_READING_CLASSES": target_reading_classes,
        "RUN_ORDER": run_order,
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

run_on_leg() {
  echo "== ON: B + semantic reading classes =="
  if [[ "$USE_ON_READING_ENV" == "1" ]]; then
    "${base_env[@]}" \
      TELEGRAM_SEMANTIC_READING_CLASSES="$READING_CLASSES" \
      python3 scripts/run_telegram_dynamic_client_sim.py "${COMMON[@]}" \
        --out-dir "$OUT/ON" \
        --progress-json "$OUT/ON/progress.json" \
        --progress-leg ON
  else
    "${base_env[@]}" python3 scripts/run_telegram_dynamic_client_sim.py "${COMMON[@]}" \
      --out-dir "$OUT/ON" \
      --progress-json "$OUT/ON/progress.json" \
      --progress-leg ON
  fi
  validate_leg ON 1
}

run_b_leg() {
  echo "== B: profile + inline frame, profile reading classes =="
  "${base_env[@]}" \
    python3 scripts/run_telegram_dynamic_client_sim.py "${COMMON[@]}" \
      --out-dir "$OUT/B" \
      --progress-json "$OUT/B/progress.json" \
      --progress-leg B
  validate_leg B 0
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

if [[ -n "$RESUME_ON_REPORT" ]]; then
  if [[ ! -d "$OUT" ]]; then
    echo "Resume OUT_DIR does not exist: $OUT" >&2
    exit 2
  fi
  for required in \
    "$OUT/B/dynamic_dialog_transcripts.jsonl" \
    "$OUT/B/dynamic_summary.json"
  do
    if [[ ! -f "$required" ]]; then
      echo "Resume OUT_DIR is missing required B artifact: $required" >&2
      exit 2
    fi
  done
  if [[ -e "$OUT/ON" || -e "$OUT/REPORT" || -e "$OUT/sha_manifest.json" ]]; then
    if [[ "$FORCE_RESUME" != "1" ]]; then
      echo "Refusing to overwrite existing ON/REPORT/sha_manifest under $OUT; pass --force to replace them." >&2
      exit 2
    fi
    rm -rf "$OUT/ON" "$OUT/REPORT" "$OUT/sha_manifest.json"
  fi
  echo "ADR003 E3 resume ON/report"
  echo "rev=$(git rev-parse HEAD)"
  echo "scenarios=$SCEN"
  echo "snapshot=$SNAPSHOT"
  echo "reading_classes=$READING_CLASSES"
  echo "target_reading_classes=$TARGET_READING_CLASSES"
  echo "use_on_reading_env=$USE_ON_READING_ENV"
  echo "run_order=$RUN_ORDER"
  echo "out=$OUT"
  echo "mode=resume-on-report"
  validate_leg B 0
  run_on_leg
  report_rc=0
  run_report || report_rc=$?
  write_sha_manifest
  echo "Done resume-on-report: $OUT"
  exit "$report_rc"
fi

mkdir -p "$OUT"

echo "ADR003 E3 paired"
echo "rev=$(git rev-parse HEAD)"
echo "scenarios=$SCEN"
echo "snapshot=$SNAPSHOT"
echo "reading_classes=$READING_CLASSES"
echo "target_reading_classes=$TARGET_READING_CLASSES"
echo "use_on_reading_env=$USE_ON_READING_ENV"
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

report_rc=0
run_report || report_rc=$?
write_sha_manifest
echo "Done: $OUT"
exit "$report_rc"
