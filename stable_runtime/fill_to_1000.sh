#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_CLI="${SCRIPT_DIR}/run-cli.sh"
PY_BIN="${SCRIPT_DIR}/venv_stable/bin/python"

PROFILE_PATH="${SCRIPT_DIR}/profiles/semi_prod_dual_asr_codex_merge.env"
DEFAULT_INPUT_DIR_1="${PROJECT_ROOT}/2026-03-05-21-06-49-ч1"
DEFAULT_INPUT_DIR_2="${PROJECT_ROOT}/2026-03-05-21-06-49-ч2"
typeset -a INPUT_DIRS
OUTPUT_DIR="${PROJECT_ROOT}/transcripts_semi_prod_400"
DB_PATH="${SCRIPT_DIR}/semi_prod_400.db"
ANALYSIS_SEED_DB="${SCRIPT_DIR}/semi_prod_300_direct_20260308_162239.db"
TARGET_DONE="1000"
STAGE_LIMIT="20"
PROGRESS_EVERY="20"
MAX_IDLE_ROUNDS="240"
SLEEP_SEC="5"
AUTO_REQUEUE_DEAD="1"
SEED_ANALYSIS="1"

usage() {
  cat <<'EOF'
Usage:
  stable_runtime/fill_to_1000.sh [options]

Goal:
  Bring pipeline to at least TARGET analyzed calls.
  Progress checkpoint is printed every PROGRESS_EVERY analyzed calls.

Options:
  --input <dir>             Recordings directory (can be repeated).
                            Default: both Mango export folders if found.
  --output <dir>            Transcript export dir (default: transcripts_semi_prod_400)
  --db <path>               SQLite DB path (default: stable_runtime/semi_prod_400.db)
  --analysis-seed-db <path> DB with already done analysis to reuse (default: semi_prod_300_direct_20260308_162239.db)
  --seed-analysis <0|1>     Copy done analysis from seed DB into target DB (default: 1)
  --target-done <n>         Target analyzed calls (default: 1000)
  --stage-limit <n>         Batch size per stage/cycle (default: 20)
  --progress-every <n>      Progress checkpoint step (default: 20)
  --profile <path>          Env profile (default: semi_prod_dual_asr_codex_merge.env)
  --max-idle-rounds <n>     Stop after N idle rounds (default: 240)
  --sleep-sec <n>           Sleep between idle rounds (default: 5)
  --auto-requeue-dead <0|1> Auto requeue dead-letter rows (default: 1)
  -h, --help                Show help

Example:
  ./stable_runtime/fill_to_1000.sh \
    --input "/Users/dmitrijfabarisov/Projects/Mango analyse/2026-03-05-21-06-49-ч1" \
    --input "/Users/dmitrijfabarisov/Projects/Mango analyse/2026-03-05-21-06-49-ч2" \
    --output "/Users/dmitrijfabarisov/Projects/Mango analyse/transcripts_semi_prod_400" \
    --db "/Users/dmitrijfabarisov/Projects/Mango analyse/stable_runtime/semi_prod_400.db" \
    --analysis-seed-db "/Users/dmitrijfabarisov/Projects/Mango analyse/stable_runtime/semi_prod_300_direct_20260308_162239.db" \
    --target-done 1000 \
    --stage-limit 20 \
    --progress-every 20
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)
      INPUT_DIRS+=("${2:-}")
      shift 2
      ;;
    --output)
      OUTPUT_DIR="${2:-}"
      shift 2
      ;;
    --db)
      DB_PATH="${2:-}"
      shift 2
      ;;
    --analysis-seed-db)
      ANALYSIS_SEED_DB="${2:-}"
      shift 2
      ;;
    --seed-analysis)
      SEED_ANALYSIS="${2:-}"
      shift 2
      ;;
    --target-done)
      TARGET_DONE="${2:-}"
      shift 2
      ;;
    --stage-limit)
      STAGE_LIMIT="${2:-}"
      shift 2
      ;;
    --progress-every)
      PROGRESS_EVERY="${2:-}"
      shift 2
      ;;
    --profile)
      PROFILE_PATH="${2:-}"
      shift 2
      ;;
    --max-idle-rounds)
      MAX_IDLE_ROUNDS="${2:-}"
      shift 2
      ;;
    --sleep-sec)
      SLEEP_SEC="${2:-}"
      shift 2
      ;;
    --auto-requeue-dead)
      AUTO_REQUEUE_DEAD="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 2
      ;;
  esac
done

if [[ ! -x "${RUN_CLI}" ]]; then
  echo "Missing launcher: ${RUN_CLI}"
  exit 2
fi
if [[ ! -x "${PY_BIN}" ]]; then
  echo "Missing stable python: ${PY_BIN}"
  exit 2
fi
if [[ ! -f "${PROFILE_PATH}" ]]; then
  echo "Profile not found: ${PROFILE_PATH}"
  exit 2
fi

if [[ "${#INPUT_DIRS[@]}" -eq 0 ]]; then
  if [[ -d "${DEFAULT_INPUT_DIR_1}" && -d "${DEFAULT_INPUT_DIR_2}" ]]; then
    INPUT_DIRS=("${DEFAULT_INPUT_DIR_1}" "${DEFAULT_INPUT_DIR_2}")
  else
    INPUT_DIRS=("${PROJECT_ROOT}")
  fi
fi

for dir in "${INPUT_DIRS[@]}"; do
  if [[ ! -d "${dir}" ]]; then
    echo "Input directory does not exist: ${dir}"
    exit 2
  fi
done

mkdir -p "${OUTPUT_DIR}"
mkdir -p "$(dirname "${DB_PATH}")"

set -a
source "${PROFILE_PATH}"
set +a

CODEX_BIN="${CODEX_CLI_COMMAND:-codex}"
if [[ "${CODEX_BIN}" == "codex" ]] && ! command -v codex >/dev/null 2>&1; then
  if [[ -x "/Applications/Codex.app/Contents/Resources/codex" ]]; then
    CODEX_BIN="/Applications/Codex.app/Contents/Resources/codex"
    export CODEX_CLI_COMMAND="${CODEX_BIN}"
    echo "[setup] using codex binary: ${CODEX_BIN}"
  fi
fi

if [[ "${DUAL_MERGE_PROVIDER:-}" == "codex_cli" || "${ANALYZE_PROVIDER:-}" == "codex_cli" || "${RESOLVE_LLM_PROVIDER:-}" == "codex_cli" ]]; then
  if [[ ! -x "${CODEX_BIN}" ]] && ! command -v "${CODEX_BIN}" >/dev/null 2>&1; then
    echo "codex CLI is required but not found: ${CODEX_BIN}"
    echo "Tip: set CODEX_CLI_COMMAND to absolute path, for example:"
    echo "  export CODEX_CLI_COMMAND=/Applications/Codex.app/Contents/Resources/codex"
    exit 2
  fi
fi

export PATH="${PROJECT_ROOT}/.local/bin:${PATH}"
export DATABASE_URL="sqlite:///${DB_PATH}"
export TRANSCRIPT_EXPORT_DIR="${OUTPUT_DIR}"
export SYNC_DRY_RUN="1"

json_int() {
  local payload="$1"
  local key="$2"
  "${PY_BIN}" -c "import json,sys; d=json.load(sys.stdin); v=d.get('${key}',0); 
try:
 print(int(v))
except Exception:
 print(0)" <<< "${payload}"
}

stats_int() {
  local payload="$1"
  local group="$2"
  local key="$3"
  "${PY_BIN}" -c "import json,sys; d=json.load(sys.stdin); g=d.get('${group}',{}) or {}; v=g.get('${key}',0);
try:
 print(int(v))
except Exception:
 print(0)" <<< "${payload}"
}

echo "[setup] init-db"
"${RUN_CLI}" init-db >/dev/null

echo "[setup] ingest (idempotent)"
INGEST_PROCESSED_TOTAL=0
INGEST_INSERTED_TOTAL=0
INGEST_SKIPPED_TOTAL=0
for dir in "${INPUT_DIRS[@]}"; do
  echo "[setup] ingest dir=${dir}"
  INGEST_PAYLOAD="$("${RUN_CLI}" ingest --recordings-dir "${dir}")"
  INGEST_PROCESSED="$(json_int "${INGEST_PAYLOAD}" processed)"
  INGEST_INSERTED="$(json_int "${INGEST_PAYLOAD}" inserted)"
  INGEST_SKIPPED="$(json_int "${INGEST_PAYLOAD}" skipped)"
  INGEST_PROCESSED_TOTAL="$((INGEST_PROCESSED_TOTAL + INGEST_PROCESSED))"
  INGEST_INSERTED_TOTAL="$((INGEST_INSERTED_TOTAL + INGEST_INSERTED))"
  INGEST_SKIPPED_TOTAL="$((INGEST_SKIPPED_TOTAL + INGEST_SKIPPED))"
  echo "[setup] ingest dir_processed=${INGEST_PROCESSED} dir_inserted=${INGEST_INSERTED} dir_skipped=${INGEST_SKIPPED}"
done
echo "[setup] ingest total_processed=${INGEST_PROCESSED_TOTAL} total_inserted=${INGEST_INSERTED_TOTAL} total_skipped=${INGEST_SKIPPED_TOTAL}"

if [[ "${SEED_ANALYSIS}" == "1" && -f "${ANALYSIS_SEED_DB}" ]]; then
  SEEDED_COUNT="$(
    TARGET_DB="${DB_PATH}" SEED_DB="${ANALYSIS_SEED_DB}" "${PY_BIN}" - <<'PY'
import os
import sqlite3

target_db = os.environ["TARGET_DB"]
seed_db = os.environ["SEED_DB"]

t_conn = sqlite3.connect(target_db)
s_conn = sqlite3.connect(seed_db)
t_cur = t_conn.cursor()
s_cur = s_conn.cursor()

seed_rows = s_cur.execute(
    """
    SELECT source_file, transcript_manager, transcript_client, transcript_text,
           transcript_variants_json, resolve_status, resolve_json, resolve_quality_score,
           analysis_json
    FROM call_records
    WHERE analysis_status='done' AND analysis_json IS NOT NULL AND analysis_json!=''
    """
).fetchall()

seed_map = {row[0]: row[1:] for row in seed_rows}
if not seed_map:
    print(0)
    t_conn.close()
    s_conn.close()
    raise SystemExit(0)

target_rows = t_cur.execute(
    """
    SELECT id, source_file, COALESCE(analysis_status, 'pending')
    FROM call_records
    """
).fetchall()

updated = 0
for row_id, source_file, analysis_status in target_rows:
    if source_file not in seed_map:
        continue
    if str(analysis_status).lower() == "done":
        continue
    (
        t_manager,
        t_client,
        t_text,
        t_variants,
        r_status,
        r_json,
        r_score,
        a_json,
    ) = seed_map[source_file]
    t_cur.execute(
        """
        UPDATE call_records
        SET transcript_manager = CASE WHEN transcript_manager IS NULL OR transcript_manager='' THEN ? ELSE transcript_manager END,
            transcript_client = CASE WHEN transcript_client IS NULL OR transcript_client='' THEN ? ELSE transcript_client END,
            transcript_text = CASE WHEN transcript_text IS NULL OR transcript_text='' THEN ? ELSE transcript_text END,
            transcript_variants_json = CASE WHEN transcript_variants_json IS NULL OR transcript_variants_json='' THEN ? ELSE transcript_variants_json END,
            resolve_status = CASE
                WHEN ? IS NOT NULL AND ? != '' THEN ?
                ELSE resolve_status
            END,
            resolve_json = CASE WHEN resolve_json IS NULL OR resolve_json='' THEN ? ELSE resolve_json END,
            resolve_quality_score = CASE WHEN resolve_quality_score IS NULL THEN ? ELSE resolve_quality_score END,
            resolve_attempts = CASE WHEN COALESCE(resolve_attempts, 0) < 1 THEN 1 ELSE resolve_attempts END,
            analysis_json = ?,
            analysis_status = 'done',
            analyze_attempts = CASE WHEN COALESCE(analyze_attempts, 0) < 1 THEN 1 ELSE analyze_attempts END,
            sync_status = 'pending',
            next_retry_at = NULL,
            dead_letter_stage = NULL,
            last_error = NULL
        WHERE id = ?
        """,
        (
            t_manager,
            t_client,
            t_text,
            t_variants,
            r_status,
            r_status,
            r_status,
            r_json,
            r_score,
            a_json,
            row_id,
        ),
    )
    updated += 1

t_conn.commit()
t_conn.close()
s_conn.close()
print(updated)
PY
  )"
  echo "[setup] seeded_analysis_from_db=${SEEDED_COUNT} source=${ANALYSIS_SEED_DB}"
fi

STATS="$("${RUN_CLI}" stats)"
TOTAL_CALLS="$(json_int "${STATS}" total_calls)"
AN_DONE="$(stats_int "${STATS}" analysis_status done)"
AN_PENDING="$(stats_int "${STATS}" analysis_status pending)"
TR_PENDING="$(stats_int "${STATS}" transcription_status pending)"
RS_PENDING="$(stats_int "${STATS}" resolve_status pending)"
echo "[setup] total_calls=${TOTAL_CALLS} analyzed_done=${AN_DONE} analyzed_pending=${AN_PENDING} transcribe_pending=${TR_PENDING} resolve_pending=${RS_PENDING}"

if [[ "${AN_DONE}" -ge "${TARGET_DONE}" ]]; then
  echo "[done] already reached target: analyzed_done=${AN_DONE} target=${TARGET_DONE}"
  exit 0
fi

NEXT_CHECKPOINT="$(( (AN_DONE / PROGRESS_EVERY + 1) * PROGRESS_EVERY ))"
if [[ "${NEXT_CHECKPOINT}" -lt "${PROGRESS_EVERY}" ]]; then
  NEXT_CHECKPOINT="${PROGRESS_EVERY}"
fi

CYCLE=0
IDLE_ROUNDS=0

echo "[run] start: target_analyzed=${TARGET_DONE}, stage_limit=${STAGE_LIMIT}, progress_every=${PROGRESS_EVERY}"

while true; do
  CYCLE="$((CYCLE + 1))"

  TRANS_PAYLOAD="$("${RUN_CLI}" transcribe --limit "${STAGE_LIMIT}")"
  RESOLVE_PAYLOAD="$("${RUN_CLI}" resolve --limit "${STAGE_LIMIT}")"
  ANALYZE_PAYLOAD="$("${RUN_CLI}" analyze --limit "${STAGE_LIMIT}")"

  TR_P="$(json_int "${TRANS_PAYLOAD}" processed)"
  TR_S="$(json_int "${TRANS_PAYLOAD}" success)"
  TR_F="$(json_int "${TRANS_PAYLOAD}" failed)"
  RS_P="$(json_int "${RESOLVE_PAYLOAD}" processed)"
  RS_S="$(json_int "${RESOLVE_PAYLOAD}" success)"
  RS_F="$(json_int "${RESOLVE_PAYLOAD}" failed)"
  RS_MANUAL="$(json_int "${RESOLVE_PAYLOAD}" manual)"
  AN_P="$(json_int "${ANALYZE_PAYLOAD}" processed)"
  AN_S="$(json_int "${ANALYZE_PAYLOAD}" success)"
  AN_F="$(json_int "${ANALYZE_PAYLOAD}" failed)"

  STATS="$("${RUN_CLI}" stats)"
  TOTAL_CALLS="$(json_int "${STATS}" total_calls)"
  TR_DONE="$(stats_int "${STATS}" transcription_status done)"
  TR_PENDING="$(stats_int "${STATS}" transcription_status pending)"
  TR_FAILED="$(stats_int "${STATS}" transcription_status failed)"
  TR_DEAD="$(stats_int "${STATS}" transcription_status dead)"
  RS_DONE="$(stats_int "${STATS}" resolve_status done)"
  RS_SKIPPED="$(stats_int "${STATS}" resolve_status skipped)"
  RS_PENDING="$(stats_int "${STATS}" resolve_status pending)"
  RS_FAILED="$(stats_int "${STATS}" resolve_status failed)"
  RS_DEAD="$(stats_int "${STATS}" resolve_status dead)"
  AN_DONE="$(stats_int "${STATS}" analysis_status done)"
  AN_PENDING="$(stats_int "${STATS}" analysis_status pending)"
  AN_FAILED="$(stats_int "${STATS}" analysis_status failed)"
  AN_DEAD="$(stats_int "${STATS}" analysis_status dead)"
  RESOLVED_DONE="$((RS_DONE + RS_SKIPPED))"
  DEAD_TOTAL="$((TR_DEAD + RS_DEAD + AN_DEAD))"

  echo "[cycle ${CYCLE}] tr p/s/f=${TR_P}/${TR_S}/${TR_F} | rs p/s/f/manual=${RS_P}/${RS_S}/${RS_F}/${RS_MANUAL} | an p/s/f=${AN_P}/${AN_S}/${AN_F}"
  echo "[state ${CYCLE}] total=${TOTAL_CALLS} tr_done=${TR_DONE} rs_done_or_skipped=${RESOLVED_DONE} an_done=${AN_DONE} pending(t/r/a)=${TR_PENDING}/${RS_PENDING}/${AN_PENDING} failed(t/r/a)=${TR_FAILED}/${RS_FAILED}/${AN_FAILED} dead=${DEAD_TOTAL}"

  while [[ "${AN_DONE}" -ge "${NEXT_CHECKPOINT}" && "${NEXT_CHECKPOINT}" -le "${TARGET_DONE}" ]]; do
    echo "[progress] analyzed ${NEXT_CHECKPOINT}/${TARGET_DONE}"
    NEXT_CHECKPOINT="$((NEXT_CHECKPOINT + PROGRESS_EVERY))"
  done

  if [[ "${AN_DONE}" -ge "${TARGET_DONE}" ]]; then
    echo "[done] target reached: analyzed_done=${AN_DONE} target=${TARGET_DONE}"
    break
  fi

  if [[ "${AUTO_REQUEUE_DEAD}" == "1" && "${DEAD_TOTAL}" -gt 0 ]]; then
    REQUEUE_PAYLOAD="$("${RUN_CLI}" requeue-dead --stage all --limit 200000)"
    REQUEUED="$(json_int "${REQUEUE_PAYLOAD}" updated)"
    echo "[requeue] dead_total=${DEAD_TOTAL} updated=${REQUEUED}"
  fi

  CYCLE_PROCESSED="$((TR_P + RS_P + AN_P))"
  if [[ "${CYCLE_PROCESSED}" -eq 0 ]]; then
    IDLE_ROUNDS="$((IDLE_ROUNDS + 1))"
    if [[ "${IDLE_ROUNDS}" -ge "${MAX_IDLE_ROUNDS}" ]]; then
      echo "[stop] max idle rounds reached (${MAX_IDLE_ROUNDS}) before target"
      echo "[stop] analyzed_done=${AN_DONE} target=${TARGET_DONE}"
      exit 1
    fi
    sleep "${SLEEP_SEC}"
  else
    IDLE_ROUNDS=0
  fi
done

FINAL_STATS="$("${RUN_CLI}" stats)"
echo "[final] ${FINAL_STATS}"
