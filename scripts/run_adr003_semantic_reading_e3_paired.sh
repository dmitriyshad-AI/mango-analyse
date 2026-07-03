#!/usr/bin/env bash
set -euo pipefail

DRY_CHECK=0
if [[ "${1:-}" == "--dry-check" ]]; then
  DRY_CHECK=1
  shift
fi
if [[ "$#" -ne 0 ]]; then
  echo "Usage: $0 [--dry-check]" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

REV_LABEL="${REV_LABEL:-e3_$(git rev-parse --short HEAD)}"
SCEN="${SCEN:-product_data/telegram_dynamic_test_sets/adr003_semantic_reading_paket1_e2_20260703.jsonl}"
SNAPSHOT="${SNAPSHOT:-product_data/knowledge_base/kb_release_20260612_v6_7_staging_r4_1/kb_release_v3_snapshot.json}"
READING_CLASSES="${READING_CLASSES:-sense_seats,off_topic,slots_gsf}"
if [[ "$DRY_CHECK" == "1" ]]; then
  OUT="${OUT:-runs/adr003_semantic_reading_e3_dry_check_${REV_LABEL}}"
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
  TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1
  TELEGRAM_DIRECT_PATH=1
  TELEGRAM_BOT_GOLD_REAL=1
  TELEGRAM_TEMPLATE_FROM_KB=1
  TELEGRAM_ROUTE_RUBRIC=1
  TELEGRAM_LLM_RETRIEVE=1
  TELEGRAM_SEMANTIC_FRAME_SHADOW=1
  TELEGRAM_RELIABLE_ANSWERER_STEP1=1
  PYTHONPATH="$ROOT/src"
)

validate_leg() {
  local leg="$1"
  local expect_trace="$2"
  python3 - "$OUT/$leg/dynamic_summary.json" "$OUT/$leg/dynamic_dialog_transcripts.jsonl" "$leg" "$expect_trace" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
transcripts_path = Path(sys.argv[2])
leg = sys.argv[3]
expect_trace = sys.argv[4] == "1"

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
provider_errors = [
    str(turn.get("bot_provider_error") or "")
    for turn in turns
    if str(turn.get("bot_provider_error") or "").strip()
]
if provider_errors:
    fail(f"provider errors: {provider_errors[:3]!r}")
direct_turns = [
    turn
    for turn in turns
    if isinstance(turn.get("bot_direct_path"), dict) and turn.get("bot_direct_path")
]
if len(direct_turns) != len(turns):
    fail(f"direct path metadata on {len(direct_turns)}/{len(turns)} turns")
frame_turns = [
    turn
    for turn in turns
    if isinstance(turn.get("bot_semantic_frame"), dict) and turn.get("bot_semantic_frame")
]
if len(frame_turns) != len(turns):
    fail(f"semantic frame metadata on {len(frame_turns)}/{len(turns)} turns")
trace_turns = [
    turn
    for turn in turns
    if isinstance(turn.get("bot_semantic_reading_trace"), list) and turn.get("bot_semantic_reading_trace")
]
if expect_trace and not trace_turns:
    fail("ON leg has no semantic_reading_trace records")
if not expect_trace and trace_turns:
    fail(f"B leg unexpectedly has semantic_reading_trace on {len(trace_turns)} turns")

print(
    f"VALID_E3_{leg}: dialogs={len(dialogs)} turns={len(turns)} "
    f"bot_direct_draft={llm_calls.get('bot_direct_draft')} trace_turns={len(trace_turns)}"
)
PY
}

write_sha_manifest() {
  python3 - "$OUT" "$SCEN" "$SNAPSHOT" "$ROOT/scripts/run_adr003_semantic_reading_e3_paired.sh" "$READING_CLASSES" <<'PY'
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
reading_classes = sys.argv[5]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

manifest = {
    "schema_version": "adr003_semantic_reading_e3_paired_v1",
    "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
    "paths": {name: str(path) for name, path in paths.items()},
    "sha256": {name: sha256(path) for name, path in paths.items()},
    "required_env": {
        "TELEGRAM_DIRECT_PATH_PILOT_CONFIG": "pilot_gold_v1",
        "TELEGRAM_DIRECT_PATH": "1",
        "TELEGRAM_BOT_GOLD_REAL": "1",
        "TELEGRAM_TEMPLATE_FROM_KB": "1",
        "TELEGRAM_ROUTE_RUBRIC": "1",
        "TELEGRAM_LLM_RETRIEVE": "1",
        "TELEGRAM_SEMANTIC_FRAME_SHADOW": "1",
        "TELEGRAM_RELIABLE_ANSWERER_STEP1": "1",
        "TELEGRAM_SEMANTIC_READING_CLASSES": reading_classes,
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

mkdir -p "$OUT"

echo "ADR003 E3 paired"
echo "rev=$(git rev-parse HEAD)"
echo "scenarios=$SCEN"
echo "snapshot=$SNAPSHOT"
echo "reading_classes=$READING_CLASSES"
echo "out=$OUT"
if [[ "$DRY_CHECK" == "1" ]]; then
  echo "mode=dry-check limit=2"
fi

echo "== B: profile + reliable + inline frame, reading classes OFF =="
"${base_env[@]}" \
  TELEGRAM_SEMANTIC_READING_CLASSES= \
  python3 scripts/run_telegram_dynamic_client_sim.py "${COMMON[@]}" \
    --out-dir "$OUT/B"
validate_leg B 0

echo "== ON: B + semantic reading classes =="
"${base_env[@]}" \
  TELEGRAM_SEMANTIC_READING_CLASSES="$READING_CLASSES" \
  python3 scripts/run_telegram_dynamic_client_sim.py "${COMMON[@]}" \
    --out-dir "$OUT/ON"
validate_leg ON 1

if [[ "$DRY_CHECK" == "1" ]]; then
  write_sha_manifest
  echo "Dry check passed: $OUT"
  exit 0
fi

echo "== Report =="
PYTHONPATH="$ROOT/src" python3 scripts/report_adr003_semantic_frame_eval.py \
  --off-transcripts "$OUT/B/dynamic_dialog_transcripts.jsonl" \
  --off-summary "$OUT/B/dynamic_summary.json" \
  --on-transcripts "$OUT/ON/dynamic_dialog_transcripts.jsonl" \
  --on-summary "$OUT/ON/dynamic_summary.json" \
  --out-dir "$OUT/REPORT"

write_sha_manifest
echo "Done: $OUT"
