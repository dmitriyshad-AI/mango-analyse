#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_PYTHON="${SCRIPT_DIR}/venv_stable/bin/python"
RUNS_DIR="${SCRIPT_DIR}/runs"
BACKUPS_DIR="${SCRIPT_DIR}/backups"
LOCK_DIR="${PROJECT_ROOT}/test_runs/.march_2026_batch500.lock"

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "Stable runtime is missing: ${VENV_PYTHON}"
  echo "Rebuild stable runtime first."
  exit 1
fi

SOURCE_RECORDINGS_DIR="${PROJECT_ROOT}/2026-03-09--26"
LIMIT=500
SAMPLE_DIR="${PROJECT_ROOT}/test_sets/march_2026_batch500"
DB_PATH="${PROJECT_ROOT}/test_runs/march_2026_batch500.db"
TRANSCRIPTS_DIR="${PROJECT_ROOT}/test_runs/march_2026_batch500_transcripts"
WORKBOOK_PATH="${PROJECT_ROOT}/test_runs/benchmarks/march_2026_batch500.xlsx"
PUBLISH_WORKBOOK_PATH="${PROJECT_ROOT}/sales_workbook.xlsx"
STAGE_LIMIT=20
POLL_SEC=10
MAX_IDLE_CYCLES=120
MERGE_MODEL="${CODEX_TRANSCRIBE_MODEL:-gpt-5.4}"
RESOLVE_MODEL="${CODEX_RESOLVE_MODEL:-gpt-5.4}"
ANALYZE_MODEL="${CODEX_ANALYZE_MODEL:-gpt-5.4-mini}"
REASONING="${CODEX_REASONING_EFFORT:-medium}"
CODEX_BIN_DEFAULT="/Applications/Codex.app/Contents/Resources/codex"
CODEX_BIN="${CODEX_CLI_COMMAND:-${CODEX_BIN_DEFAULT}}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --recordings-dir)
      SOURCE_RECORDINGS_DIR="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --sample-dir)
      SAMPLE_DIR="$2"
      shift 2
      ;;
    --db)
      DB_PATH="$2"
      shift 2
      ;;
    --transcripts-dir)
      TRANSCRIPTS_DIR="$2"
      shift 2
      ;;
    --workbook-out)
      WORKBOOK_PATH="$2"
      shift 2
      ;;
    --publish-workbook)
      PUBLISH_WORKBOOK_PATH="$2"
      shift 2
      ;;
    --stage-limit)
      STAGE_LIMIT="$2"
      shift 2
      ;;
    --poll-sec)
      POLL_SEC="$2"
      shift 2
      ;;
    --max-idle-cycles)
      MAX_IDLE_CYCLES="$2"
      shift 2
      ;;
    --merge-model)
      MERGE_MODEL="$2"
      shift 2
      ;;
    --resolve-model)
      RESOLVE_MODEL="$2"
      shift 2
      ;;
    --analyze-model)
      ANALYZE_MODEL="$2"
      shift 2
      ;;
    --reasoning)
      REASONING="$2"
      shift 2
      ;;
    --codex-bin)
      CODEX_BIN="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

mkdir -p "${RUNS_DIR}" "${BACKUPS_DIR}"

if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "Another march batch 500 run appears to be active."
  echo "Lock: ${LOCK_DIR}"
  exit 1
fi

cleanup_lock() {
  rmdir "${LOCK_DIR}" 2>/dev/null || true
}
trap cleanup_lock EXIT

RUN_ID="march_batch500_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUNS_DIR}/${RUN_ID}"
LOG_FILE="${RUN_DIR}/run.log"
CONTROLLER_SCRIPT="${RUN_DIR}/controller.sh"
mkdir -p "${RUN_DIR}"

cat >"${CONTROLLER_SCRIPT}" <<EOF
#!/bin/zsh
set -euo pipefail

cd "${PROJECT_ROOT}"
export PATH="/Applications/Codex.app/Contents/Resources:${PROJECT_ROOT}/.local/bin:\${PATH}"
unset PYTHONPATH || true

SOURCE_RECORDINGS_DIR="${SOURCE_RECORDINGS_DIR}"
LIMIT="${LIMIT}"
SAMPLE_DIR="${SAMPLE_DIR}"
DB_PATH="${DB_PATH}"
TRANSCRIPTS_DIR="${TRANSCRIPTS_DIR}"
WORKBOOK_PATH="${WORKBOOK_PATH}"
PUBLISH_WORKBOOK_PATH="${PUBLISH_WORKBOOK_PATH}"
STAGE_LIMIT="${STAGE_LIMIT}"
POLL_SEC="${POLL_SEC}"
MAX_IDLE_CYCLES="${MAX_IDLE_CYCLES}"
BACKUPS_DIR="${BACKUPS_DIR}"
LOCK_DIR="${LOCK_DIR}"
MERGE_MODEL="${MERGE_MODEL}"
RESOLVE_MODEL="${RESOLVE_MODEL}"
ANALYZE_MODEL="${ANALYZE_MODEL}"
REASONING="${REASONING}"
CODEX_BIN="${CODEX_BIN}"
VENV_PYTHON="${VENV_PYTHON}"
RUN_DIR="${RUN_DIR}"

mkdir -p "\$(dirname "\${DB_PATH}")" "\$(dirname "\${WORKBOOK_PATH}")" "\${TRANSCRIPTS_DIR}" "\${SAMPLE_DIR}" "\${BACKUPS_DIR}"
cleanup() {
  rmdir "\${LOCK_DIR}" 2>/dev/null || true
}
trap cleanup EXIT

if [[ -f "\${DB_PATH}" ]]; then
  cp "\${DB_PATH}" "\${BACKUPS_DIR}/\$(basename "\${DB_PATH}").before_\$(date +%Y%m%d_%H%M%S).bak"
  rm -f "\${DB_PATH}"
fi

if [[ -f "\${PUBLISH_WORKBOOK_PATH}" ]]; then
  cp "\${PUBLISH_WORKBOOK_PATH}" "\${BACKUPS_DIR}/\$(basename "\${PUBLISH_WORKBOOK_PATH}").before_\$(date +%Y%m%d_%H%M%S).bak"
fi

rm -rf "\${TRANSCRIPTS_DIR}"
mkdir -p "\${TRANSCRIPTS_DIR}"

"\${VENV_PYTHON}" - <<'PY'
from pathlib import Path

source_dir = Path("${SOURCE_RECORDINGS_DIR}")
sample_dir = Path("${SAMPLE_DIR}")
limit = int("${LIMIT}")
exts = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}

files = sorted(
    path for path in source_dir.rglob("*")
    if path.is_file() and path.suffix.lower() in exts
)
selected = files[:limit]
if len(selected) < limit:
    raise SystemExit(f"Not enough audio files in {source_dir}: need {limit}, got {len(selected)}")

sample_dir.mkdir(parents=True, exist_ok=True)
for path in sample_dir.iterdir():
    if path.is_symlink() or path.is_file():
        path.unlink()

for idx, path in enumerate(selected, start=1):
    target = sample_dir / path.name
    if target.exists():
        target = sample_dir / f"{idx:04d}__{path.name}"
    target.symlink_to(path.resolve())

manifest = sample_dir / "_manifest.txt"
manifest.write_text(
    "\\n".join(str(path.resolve()) for path in selected) + "\\n",
    encoding="utf-8",
)
print({"selected": len(selected), "sample_dir": str(sample_dir), "first": str(selected[0]), "last": str(selected[-1])})
PY

export DATABASE_URL="sqlite:///\${DB_PATH}"
export TRANSCRIPT_EXPORT_DIR="\${TRANSCRIPTS_DIR}"
export TRANSCRIBE_PROVIDER="mlx"
export DUAL_TRANSCRIBE_ENABLED="1"
export SECONDARY_TRANSCRIBE_PROVIDER="gigaam"
export DUAL_MERGE_PROVIDER="codex_cli"
export CODEX_TRANSCRIBE_MODEL="\${MERGE_MODEL}"
export CODEX_RESOLVE_MODEL="\${RESOLVE_MODEL}"
export CODEX_ANALYZE_MODEL="\${ANALYZE_MODEL}"
export CODEX_REASONING_EFFORT="\${REASONING}"
export CODEX_CLI_COMMAND="\${CODEX_BIN}"
export CODEX_CLI_TIMEOUT_SEC="180"
export RESOLVE_LLM_PROVIDER="codex_cli"
export RESOLVE_DIALOGUE_MODE="dialogue"
export RESOLVE_LLM_FOR_RISKY="false"
export ANALYZE_PROVIDER="codex_cli"
export WORKER_POLL_SEC="\${POLL_SEC}"
export WORKER_MAX_IDLE_CYCLES="0"

"${SCRIPT_DIR}/run-cli.sh" init-db
"${SCRIPT_DIR}/run-cli.sh" ingest --recordings-dir "\${SAMPLE_DIR}"
"${SCRIPT_DIR}/run-cli.sh" stats | tee "\${RUN_DIR}/stats_after_ingest.json"

echo "[start] whisper" >&2
"${SCRIPT_DIR}/run-cli.sh" worker \\
  --stage-limit "\${STAGE_LIMIT}" \\
  --stages "transcribe" \\
  --poll-sec "\${POLL_SEC}" \\
  --max-idle-cycles "\${MAX_IDLE_CYCLES}" \\
  >> "\${RUN_DIR}/whisper.log" 2>&1 &
TRANSCRIBE_PID=\$!

echo "[start] gigaam" >&2
"${SCRIPT_DIR}/run-cli.sh" worker \\
  --stage-limit "\${STAGE_LIMIT}" \\
  --stages "backfill-second-asr" \\
  --poll-sec "\${POLL_SEC}" \\
  --max-idle-cycles "\${MAX_IDLE_CYCLES}" \\
  >> "\${RUN_DIR}/gigaam.log" 2>&1 &
BACKFILL_PID=\$!

echo "[start] resolve" >&2
"${SCRIPT_DIR}/run-cli.sh" worker \\
  --stage-limit "\${STAGE_LIMIT}" \\
  --stages "resolve" \\
  --poll-sec "\${POLL_SEC}" \\
  --max-idle-cycles "\${MAX_IDLE_CYCLES}" \\
  >> "\${RUN_DIR}/resolve.log" 2>&1 &
RESOLVE_PID=\$!

echo "[start] analyze" >&2
"${SCRIPT_DIR}/run-cli.sh" worker \\
  --stage-limit "\${STAGE_LIMIT}" \\
  --stages "analyze" \\
  --poll-sec "\${POLL_SEC}" \\
  --max-idle-cycles "\${MAX_IDLE_CYCLES}" \\
  >> "\${RUN_DIR}/analyze.log" 2>&1 &
ANALYZE_PID=\$!

echo "{\\"whisper\\": \${TRANSCRIBE_PID}, \\"gigaam\\": \${BACKFILL_PID}, \\"resolve\\": \${RESOLVE_PID}, \\"analyze\\": \${ANALYZE_PID}}" | tee "\${RUN_DIR}/pids.json"

rc_whisper=0
rc_gigaam=0
rc_resolve=0
rc_analyze=0
wait "\${TRANSCRIBE_PID}" || rc_whisper=\$?
wait "\${BACKFILL_PID}" || rc_gigaam=\$?
wait "\${RESOLVE_PID}" || rc_resolve=\$?
wait "\${ANALYZE_PID}" || rc_analyze=\$?

echo "{\\"rc_whisper\\": \${rc_whisper}, \\"rc_gigaam\\": \${rc_gigaam}, \\"rc_resolve\\": \${rc_resolve}, \\"rc_analyze\\": \${rc_analyze}}" | tee "\${RUN_DIR}/worker_rcs.json"

"${SCRIPT_DIR}/run-cli.sh" stats | tee "\${RUN_DIR}/stats_final.json"
"${SCRIPT_DIR}/run-cli.sh" export-sales-workbook --out "\${WORKBOOK_PATH}" --only-done --limit 200000
cp "\${WORKBOOK_PATH}" "\${PUBLISH_WORKBOOK_PATH}"

echo "[done] workbook=\${WORKBOOK_PATH}"
echo "[done] published=\${PUBLISH_WORKBOOK_PATH}"
EOF

chmod +x "${CONTROLLER_SCRIPT}"

nohup "${CONTROLLER_SCRIPT}" >"${LOG_FILE}" 2>&1 &
PID=$!

echo "run_id=${RUN_ID}"
echo "pid=${PID}"
echo "log=${LOG_FILE}"
echo "db=${DB_PATH}"
echo "sample_dir=${SAMPLE_DIR}"
echo "transcripts_dir=${TRANSCRIPTS_DIR}"
echo "workbook=${WORKBOOK_PATH}"
echo "published_workbook=${PUBLISH_WORKBOOK_PATH}"
