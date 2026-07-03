#!/usr/bin/env bash
set -euo pipefail

DRY_CHECK=0
RESUME_P_REPORT=""
FORCE_RESUME=0
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --dry-check)
      DRY_CHECK=1
      shift
      ;;
    --resume-p-report)
      if [[ -z "${2:-}" ]]; then
        echo "Usage: $0 [--dry-check] [--resume-p-report OUT_DIR] [--force]" >&2
        exit 2
      fi
      RESUME_P_REPORT="$2"
      shift 2
      ;;
    --force)
      FORCE_RESUME=1
      shift
      ;;
    *)
      echo "Usage: $0 [--dry-check] [--resume-p-report OUT_DIR] [--force]" >&2
      exit 2
      ;;
  esac
done
if [[ "$DRY_CHECK" == "1" && -n "$RESUME_P_REPORT" ]]; then
  echo "--dry-check and --resume-p-report are mutually exclusive" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

REV_LABEL="${REV_LABEL:-postE1_$(git rev-parse --short HEAD)}"
SCEN="${SCEN:-product_data/telegram_dynamic_test_sets/adr003_semantic_reading_paket1_e2_20260703.jsonl}"
SNAPSHOT="${SNAPSHOT:-product_data/knowledge_base/kb_release_20260612_v6_7_staging_r4_1/kb_release_v3_snapshot.json}"
if [[ "$DRY_CHECK" == "1" ]]; then
  OUT="${OUT:-runs/adr003_semantic_reading_e2_dry_check_${REV_LABEL}}"
elif [[ -n "$RESUME_P_REPORT" ]]; then
  OUT="$RESUME_P_REPORT"
else
  OUT="${OUT:-runs/adr003_semantic_reading_e2_triple_${REV_LABEL}}"
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

clean_semantic_env=(
  env
  -u TELEGRAM_SEMANTIC_FRAME_SHADOW
  -u TELEGRAM_SEMANTIC_FRAME_POSTHOC_SHADOW
  -u TELEGRAM_SEMANTIC_FRAME_DECISION_SHADOW
  -u TELEGRAM_SEMANTIC_FRAME_MANAGER_ACTION_GATE
  -u TELEGRAM_SEMANTIC_FRAME_SELF_ANSWER_SHADOW
  -u TELEGRAM_SEMANTIC_FRAME_EXISTENCE_PROOF_SHADOW
  -u TELEGRAM_SEMANTIC_FRAME_PROOF_RECONCILIATION_SHADOW
  TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1
  TELEGRAM_DIRECT_PATH=1
  TELEGRAM_BOT_GOLD_REAL=1
  TELEGRAM_TEMPLATE_FROM_KB=1
  TELEGRAM_ROUTE_RUBRIC=1
  TELEGRAM_LLM_RETRIEVE=1
  TELEGRAM_SEMANTIC_READING_CLASSES=
  PYTHONPATH="$ROOT/src"
)

validate_direct_leg() {
  local leg="$1"
  python3 - "$OUT/$leg/dynamic_summary.json" "$OUT/$leg/dynamic_dialog_transcripts.jsonl" "$leg" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
transcripts_path = Path(sys.argv[2])
leg = sys.argv[3]

def fail(message: str) -> None:
    print(f"INVALID_{leg}: {message}", file=sys.stderr)
    raise SystemExit(1)

if not summary_path.is_file():
    fail(f"missing summary: {summary_path}")
if not transcripts_path.is_file():
    fail(f"missing transcripts: {transcripts_path}")

summary = json.loads(summary_path.read_text(encoding="utf-8"))
profile = (
    summary.get("run_config", {})
    .get("key_flags", {})
    .get("profile", {})
)
if profile.get("env") != "pilot_gold_v1" or profile.get("effective") is not True:
    fail(f"pilot profile not active: {profile!r}")

llm_calls = summary.get("llm_calls") or {}
if int(llm_calls.get("bot_direct_draft") or 0) <= 0:
    fail(f"bot_direct_draft is not positive: {llm_calls!r}")

dialogs = [
    json.loads(line)
    for line in transcripts_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
turns = [turn for dialog in dialogs for turn in (dialog.get("turns") or [])]
if not turns:
    fail("no turns in transcripts")
direct_turns = [
    turn
    for turn in turns
    if isinstance(turn.get("bot_direct_path"), dict) and turn.get("bot_direct_path")
]
if len(direct_turns) != len(turns):
    fail(f"direct path metadata on {len(direct_turns)}/{len(turns)} turns")

print(
    f"VALID_DIRECT_{leg}: dialogs={len(dialogs)} turns={len(turns)} "
    f"bot_direct_draft={llm_calls.get('bot_direct_draft')}"
)
PY
}

validate_inline_frame_leg() {
  local leg="$1"
  python3 - "$OUT/$leg/dynamic_summary.json" "$OUT/$leg/dynamic_dialog_transcripts.jsonl" "$leg" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
transcripts_path = Path(sys.argv[2])
leg = sys.argv[3]

def fail(message: str) -> None:
    print(f"INVALID_{leg}: {message}", file=sys.stderr)
    raise SystemExit(1)

P0_PREBLOCK_REASONS = {"p0_pre_gate", "direct_path_preblocked_p0"}

def direct_path(turn):
    value = turn.get("bot_direct_path")
    return value if isinstance(value, dict) else {}

def answerability_trace(turn):
    value = turn.get("bot_answerability_trace")
    return value if isinstance(value, dict) else {}

def provider_error(turn):
    candidates = [
        turn.get("bot_provider_error"),
        turn.get("provider_error"),
    ]
    reason = turn.get("reason_evidence")
    if isinstance(reason, dict):
        candidates.append(reason.get("provider_error"))
    direct = direct_path(turn)
    direct_reason = direct.get("reason_evidence")
    if isinstance(direct_reason, dict):
        candidates.append(direct_reason.get("provider_error"))
    trace = answerability_trace(turn)
    trace_direct = trace.get("direct_path")
    if isinstance(trace_direct, dict):
        trace_reason = trace_direct.get("reason_evidence")
        if isinstance(trace_reason, dict):
            candidates.append(trace_reason.get("provider_error"))
    for candidate in candidates:
        value = str(candidate or "").strip().casefold()
        if value:
            return value
    return ""

def is_timeout(turn):
    return provider_error(turn) == "timeout"

def is_p0_preblocked(turn):
    direct = direct_path(turn)
    return (
        direct.get("model_called") is False
        and direct.get("preblocked") is True
        and str(direct.get("preblock_reason") or "").strip() in P0_PREBLOCK_REASONS
    )

def has_frame(turn):
    frame = turn.get("bot_semantic_frame")
    return isinstance(frame, dict) and bool(frame)

summary = json.loads(summary_path.read_text(encoding="utf-8"))
llm_calls = summary.get("llm_calls") or {}

dialogs = [
    json.loads(line)
    for line in transcripts_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
turns = [turn for dialog in dialogs for turn in (dialog.get("turns") or [])]
if not turns:
    fail("no turns in transcripts")
frame_turns = [
    turn
    for turn in turns
    if has_frame(turn)
]
preblocked_p0 = [turn for turn in turns if is_p0_preblocked(turn)]
timeouts = [turn for turn in turns if is_timeout(turn)]
eligible_turns = [turn for turn in turns if not is_p0_preblocked(turn) and not is_timeout(turn)]
eligible_frame_turns = [turn for turn in eligible_turns if has_frame(turn)]
if not eligible_turns:
    fail("no eligible model-called turns for semantic frame validation")
eligible_frame_rate = len(eligible_frame_turns) / len(eligible_turns)
if len(eligible_frame_turns) * 100 < len(eligible_turns) * 97:
    fail(
        "semantic frame eligible emission "
        f"{len(eligible_frame_turns)}/{len(eligible_turns)} "
        f"(preblocked_p0={len(preblocked_p0)} timeouts={len(timeouts)})"
    )
non_inline_sources = [
    turn.get("bot_semantic_frame", {}).get("source")
    for turn in frame_turns
    if turn.get("bot_semantic_frame", {}).get("source") != "inline"
]
if non_inline_sources:
    fail(f"semantic frame source is not inline: {non_inline_sources[:5]!r}")

print(
    f"VALID_INLINE_FRAME_{leg}: dialogs={len(dialogs)} turns={len(turns)} "
    f"preblocked_p0={len(preblocked_p0)} timeouts={len(timeouts)} "
    f"model_called_eligible={len(eligible_turns)} frames={len(eligible_frame_turns)} "
    f"eligible_frame_rate={eligible_frame_rate:.4f} "
    f"bot_semantic_frame_shadow={llm_calls.get('bot_semantic_frame_shadow')}"
)
PY
}

write_sha_manifest() {
  python3 - "$OUT" "$SCEN" "$SNAPSHOT" "$ROOT/scripts/run_adr003_semantic_reading_e2_triple.sh" <<'PY'
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

out_dir = Path(sys.argv[1])
paths = {
    "scenario": Path(sys.argv[2]),
    "snapshot": Path(sys.argv[3]),
    "runner": Path(sys.argv[4]),
}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

artifact_paths = {
    "B_transcripts": out_dir / "B" / "dynamic_dialog_transcripts.jsonl",
    "B_summary": out_dir / "B" / "dynamic_summary.json",
    "I_transcripts": out_dir / "I" / "dynamic_dialog_transcripts.jsonl",
    "I_summary": out_dir / "I" / "dynamic_summary.json",
    "P_transcripts": out_dir / "P" / "dynamic_dialog_transcripts.jsonl",
    "P_summary": out_dir / "P" / "dynamic_summary.json",
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
    "schema_version": "adr003_semantic_reading_e2_triple_v3",
    "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
    "paths": {name: str(path) for name, path in paths.items()},
    "sha256": {name: sha256(path) for name, path in paths.items()},
    "artifacts": artifacts,
    "required_env": {
        "TELEGRAM_DIRECT_PATH_PILOT_CONFIG": "pilot_gold_v1",
        "TELEGRAM_DIRECT_PATH": "1",
        "TELEGRAM_BOT_GOLD_REAL": "1",
        "TELEGRAM_TEMPLATE_FROM_KB": "1",
        "TELEGRAM_ROUTE_RUBRIC": "1",
        "TELEGRAM_LLM_RETRIEVE": "1",
        "TELEGRAM_SEMANTIC_READING_CLASSES": "",
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

run_p_and_report() {
  if [[ -e "$OUT/P" || -e "$OUT/REPORT" ]]; then
    if [[ "$FORCE_RESUME" != "1" ]]; then
      echo "Refusing to overwrite existing P/REPORT under $OUT; pass --force to replace only those outputs." >&2
      exit 2
    fi
    rm -rf "$OUT/P" "$OUT/REPORT"
  fi

  echo "== P: post-hoc SemanticFrame enrich over B =="
  "${clean_semantic_env[@]}" \
    python3 scripts/run_telegram_dynamic_client_sim.py \
      --scenarios "$SCEN" \
      --snapshot "$SNAPSHOT" \
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
}

if [[ -n "$RESUME_P_REPORT" ]]; then
  if [[ ! -d "$OUT" ]]; then
    echo "Resume OUT_DIR does not exist: $OUT" >&2
    exit 2
  fi
  for required in \
    "$OUT/B/dynamic_dialog_transcripts.jsonl" \
    "$OUT/B/dynamic_summary.json" \
    "$OUT/I/dynamic_dialog_transcripts.jsonl" \
    "$OUT/I/dynamic_summary.json"
  do
    if [[ ! -f "$required" ]]; then
      echo "Resume OUT_DIR is missing required artifact: $required" >&2
      exit 2
    fi
  done
  echo "ADR003 E2 resume P/report"
  echo "rev=$(git rev-parse HEAD)"
  echo "scenarios=$SCEN"
  echo "snapshot=$SNAPSHOT"
  echo "out=$OUT"
  echo "mode=resume-p-report"
  validate_direct_leg B
  validate_direct_leg I
  validate_inline_frame_leg I
  run_p_and_report
  write_sha_manifest
  echo "Done resume-p-report: $OUT"
  exit 0
fi

mkdir -p "$OUT"

echo "ADR003 E2 triple"
echo "rev=$(git rev-parse HEAD)"
echo "scenarios=$SCEN"
echo "snapshot=$SNAPSHOT"
echo "out=$OUT"
if [[ "$DRY_CHECK" == "1" ]]; then
  echo "mode=dry-check limit=2"
fi

echo "== B: baseline, semantic frame shadow OFF =="
"${clean_semantic_env[@]}" \
  python3 scripts/run_telegram_dynamic_client_sim.py "${COMMON[@]}" \
    --out-dir "$OUT/B"
validate_direct_leg B

echo "== I: inline SemanticFrame shadow ON, semantic reading masks empty =="
"${clean_semantic_env[@]}" \
  TELEGRAM_SEMANTIC_FRAME_SHADOW=1 \
  python3 scripts/run_telegram_dynamic_client_sim.py "${COMMON[@]}" \
    --out-dir "$OUT/I"
validate_direct_leg I
validate_inline_frame_leg I

if [[ "$DRY_CHECK" == "1" ]]; then
  write_sha_manifest
  echo "Dry check passed: $OUT"
  exit 0
fi

run_p_and_report

write_sha_manifest
echo "Done: $OUT"
