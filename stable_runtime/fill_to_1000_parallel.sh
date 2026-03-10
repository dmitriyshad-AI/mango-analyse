#!/bin/zsh
set -euo pipefail
# Some environments enable BG_NICE and print noisy "nice(...) failed" for background jobs.
setopt NO_BG_NICE 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_CLI="${SCRIPT_DIR}/run-cli.sh"
PY_BIN="${SCRIPT_DIR}/venv_stable/bin/python"

PROFILE_PATH="${SCRIPT_DIR}/profiles/semi_prod_dual_asr_codex_balanced.env"
DEFAULT_INPUT_DIR_1="${PROJECT_ROOT}/2026-03-05-21-06-49-ч1"
DEFAULT_INPUT_DIR_2="${PROJECT_ROOT}/2026-03-05-21-06-49-ч2"
typeset -a INPUT_DIRS
OUTPUT_DIR="${PROJECT_ROOT}/transcripts_semi_prod_400"
DB_PATH="${SCRIPT_DIR}/semi_prod_400.db"
ANALYSIS_SEED_DB="${SCRIPT_DIR}/semi_prod_300_direct_20260308_162239.db"
SEED_ANALYSIS="1"
TARGET_DONE="1000"
TRANSCRIBE_LIMIT="6"
RESOLVE_LIMIT="10"
ANALYZE_LIMIT="10"
PROGRESS_EVERY="20"
POLL_SEC="5"
MAX_IDLE_ROUNDS="720"
AUTO_REQUEUE_DEAD="1"
RUN_ID="$(date +%Y%m%d_%H%M%S)"

usage() {
  cat <<'EOF'
Usage:
  stable_runtime/fill_to_1000_parallel.sh [options]

Goal:
  Bring pipeline to at least TARGET analyzed calls using parallel workers:
  - worker A: transcribe
  - worker B: resolve + analyze (weighted: resolve priority)

Options:
  --input <dir>             Recordings directory (can be repeated).
                            Default: both Mango export folders if found.
  --output <dir>            Transcript export dir (default: transcripts_semi_prod_400)
  --db <path>               SQLite DB path (default: stable_runtime/semi_prod_400.db)
  --analysis-seed-db <path> DB with already done analysis to reuse
                            (default: semi_prod_300_direct_20260308_162239.db)
  --seed-analysis <0|1>     Copy done analysis from seed DB into target DB (default: 1)
  --target-done <n>         Target analyzed calls (default: 1000)
  --transcribe-limit <n>    Batch size per transcribe call (default: 6)
  --resolve-limit <n>       Batch size per resolve call (default: 10)
  --analyze-limit <n>       Batch size per analyze call (default: 10)
  --progress-every <n>      Progress checkpoint step (default: 20)
  --poll-sec <n>            Monitor/idle poll seconds (default: 5)
  --max-idle-rounds <n>     Stop after N monitor rounds without progress (default: 720)
  --auto-requeue-dead <0|1> Auto requeue dead-letter rows (default: 1)
  --profile <path>          Env profile (default: semi_prod_dual_asr_codex_balanced.env)
  --run-id <id>             Explicit run id (default: timestamp)
  -h, --help                Show help

Example:
  ./stable_runtime/fill_to_1000_parallel.sh \
    --input "/Users/dmitrijfabarisov/Projects/Mango analyse/2026-03-05-21-06-49-ч1" \
    --input "/Users/dmitrijfabarisov/Projects/Mango analyse/2026-03-05-21-06-49-ч2" \
    --output "/Users/dmitrijfabarisov/Projects/Mango analyse/transcripts_semi_prod_400" \
    --db "/Users/dmitrijfabarisov/Projects/Mango analyse/stable_runtime/semi_prod_400.db" \
    --target-done 1000 \
    --transcribe-limit 6 \
    --resolve-limit 10 \
    --analyze-limit 10 \
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
    --transcribe-limit)
      TRANSCRIBE_LIMIT="${2:-}"
      shift 2
      ;;
    --resolve-limit)
      RESOLVE_LIMIT="${2:-}"
      shift 2
      ;;
    --analyze-limit)
      ANALYZE_LIMIT="${2:-}"
      shift 2
      ;;
    --progress-every)
      PROGRESS_EVERY="${2:-}"
      shift 2
      ;;
    --poll-sec)
      POLL_SEC="${2:-}"
      shift 2
      ;;
    --max-idle-rounds)
      MAX_IDLE_ROUNDS="${2:-}"
      shift 2
      ;;
    --auto-requeue-dead)
      AUTO_REQUEUE_DEAD="${2:-}"
      shift 2
      ;;
    --profile)
      PROFILE_PATH="${2:-}"
      shift 2
      ;;
    --run-id)
      RUN_ID="${2:-}"
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
if [[ "${TARGET_DONE}" != <-> || "${TRANSCRIBE_LIMIT}" != <-> || "${RESOLVE_LIMIT}" != <-> || "${ANALYZE_LIMIT}" != <-> || "${PROGRESS_EVERY}" != <-> || "${POLL_SEC}" != <-> || "${MAX_IDLE_ROUNDS}" != <-> ]]; then
  echo "Numeric options must be integers."
  exit 2
fi
if [[ "${TRANSCRIBE_LIMIT}" -le 0 || "${RESOLVE_LIMIT}" -le 0 || "${ANALYZE_LIMIT}" -le 0 || "${POLL_SEC}" -le 0 || "${MAX_IDLE_ROUNDS}" -le 0 ]]; then
  echo "Batch/poll/idle options must be > 0."
  exit 2
fi
if [[ "${PROGRESS_EVERY}" -le 0 ]]; then
  echo "--progress-every must be > 0."
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

RUN_DIR="${SCRIPT_DIR}/runs/parallel_${RUN_ID}"
mkdir -p "${RUN_DIR}"

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

LOCK_DIR="${DB_PATH}.parallel.lock"
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "Lock exists: ${LOCK_DIR}"
  echo "Another parallel runner may be active for this DB. Stop it first."
  exit 2
fi

STOP_FILE="${RUN_DIR}/stop.flag"
TRANS_LOG="${RUN_DIR}/transcribe_worker.log"
POST_LOG="${RUN_DIR}/post_worker.log"
: > "${TRANS_LOG}"
: > "${POST_LOG}"

TR_PID=""
POST_PID=""

cleanup() {
  touch "${STOP_FILE}" 2>/dev/null || true
  if [[ -n "${TR_PID}" ]]; then
    kill "${TR_PID}" 2>/dev/null || true
  fi
  if [[ -n "${POST_PID}" ]]; then
    kill "${POST_PID}" 2>/dev/null || true
  fi
  if [[ -n "${TR_PID}" ]]; then
    wait "${TR_PID}" 2>/dev/null || true
  fi
  if [[ -n "${POST_PID}" ]]; then
    wait "${POST_PID}" 2>/dev/null || true
  fi
  rmdir "${LOCK_DIR}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

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
TR_DONE="$(stats_int "${STATS}" transcription_status done)"
TR_PENDING="$(stats_int "${STATS}" transcription_status pending)"
RS_PENDING="$(stats_int "${STATS}" resolve_status pending)"
AN_DONE="$(stats_int "${STATS}" analysis_status done)"
AN_PENDING="$(stats_int "${STATS}" analysis_status pending)"
echo "[setup] total_calls=${TOTAL_CALLS} tr_done=${TR_DONE} tr_pending=${TR_PENDING} resolve_pending=${RS_PENDING} analyzed_done=${AN_DONE} analyzed_pending=${AN_PENDING}"

if [[ "${AN_DONE}" -ge "${TARGET_DONE}" ]]; then
  echo "[done] already reached target: analyzed_done=${AN_DONE} target=${TARGET_DONE}"
  exit 0
fi

cat > "${RUN_DIR}/run_config.txt" <<EOF
run_id=${RUN_ID}
db_path=${DB_PATH}
output_dir=${OUTPUT_DIR}
profile=${PROFILE_PATH}
target_done=${TARGET_DONE}
transcribe_limit=${TRANSCRIBE_LIMIT}
resolve_limit=${RESOLVE_LIMIT}
analyze_limit=${ANALYZE_LIMIT}
poll_sec=${POLL_SEC}
progress_every=${PROGRESS_EVERY}
max_idle_rounds=${MAX_IDLE_ROUNDS}
auto_requeue_dead=${AUTO_REQUEUE_DEAD}
started_at=$(date "+%Y-%m-%d %H:%M:%S %z")
EOF

transcribe_worker() {
  local cycle=0
  while [[ ! -f "${STOP_FILE}" ]]; do
    cycle=$((cycle + 1))
    local payload
    if payload="$("${RUN_CLI}" transcribe --limit "${TRANSCRIBE_LIMIT}")"; then
      local p s f
      p="$(json_int "${payload}" processed)"
      s="$(json_int "${payload}" success)"
      f="$(json_int "${payload}" failed)"
      if [[ "${p}" -gt 0 || "${f}" -gt 0 ]]; then
        echo "[tr-worker] cycle=${cycle} p/s/f=${p}/${s}/${f}" >> "${TRANS_LOG}"
      fi
      if [[ "${p}" -eq 0 ]]; then
        sleep "${POLL_SEC}"
      fi
    else
      local rc=$?
      echo "[tr-worker] cycle=${cycle} error rc=${rc}" >> "${TRANS_LOG}"
      sleep "${POLL_SEC}"
    fi
  done
}

post_worker() {
  local cycle=0
  while [[ ! -f "${STOP_FILE}" ]]; do
    cycle=$((cycle + 1))
    local resolve_payload analyze_payload
    local rp=0 rs=0 rf=0 rm=0 ap=0 as=0 af=0

    if resolve_payload="$("${RUN_CLI}" resolve --limit "${RESOLVE_LIMIT}")"; then
      rp="$(json_int "${resolve_payload}" processed)"
      rs="$(json_int "${resolve_payload}" success)"
      rf="$(json_int "${resolve_payload}" failed)"
      rm="$(json_int "${resolve_payload}" manual)"
    else
      echo "[post-worker] cycle=${cycle} resolve error rc=$?" >> "${POST_LOG}"
    fi

    # Resolve is prioritized to avoid backlog growth while analyze is heavy.
    # Analyze runs every 3 cycles, or immediately when resolve had no work.
    if [[ "${rp}" -eq 0 || $((cycle % 3)) -eq 0 ]]; then
      if analyze_payload="$("${RUN_CLI}" analyze --limit "${ANALYZE_LIMIT}")"; then
        ap="$(json_int "${analyze_payload}" processed)"
        as="$(json_int "${analyze_payload}" success)"
        af="$(json_int "${analyze_payload}" failed)"
      else
        echo "[post-worker] cycle=${cycle} analyze error rc=$?" >> "${POST_LOG}"
      fi
    fi

    if [[ "${rp}" -gt 0 || "${rf}" -gt 0 || "${ap}" -gt 0 || "${af}" -gt 0 || "${rm}" -gt 0 ]]; then
      echo "[post-worker] cycle=${cycle} rs p/s/f/manual=${rp}/${rs}/${rf}/${rm} | an p/s/f=${ap}/${as}/${af}" >> "${POST_LOG}"
    fi

    if [[ $((rp + ap)) -eq 0 ]]; then
      sleep "${POLL_SEC}"
    fi
  done
}

transcribe_worker &
TR_PID="$!"
post_worker &
POST_PID="$!"

echo "[run] started parallel workers"
echo "[run] run_id=${RUN_ID}"
echo "[run] db=${DB_PATH}"
echo "[run] worker_logs: transcribe=${TRANS_LOG} post=${POST_LOG}"
echo "[run] target_analyzed=${TARGET_DONE} checkpoint_every=${PROGRESS_EVERY}"

NEXT_CHECKPOINT="$(( (AN_DONE / PROGRESS_EVERY + 1) * PROGRESS_EVERY ))"
if [[ "${NEXT_CHECKPOINT}" -lt "${PROGRESS_EVERY}" ]]; then
  NEXT_CHECKPOINT="${PROGRESS_EVERY}"
fi

monitor_cycle=0
idle_rounds=0
last_signature=""
exit_code=0
stats_failures=0

while true; do
  monitor_cycle=$((monitor_cycle + 1))
  sleep "${POLL_SEC}"
  if ! STATS="$("${RUN_CLI}" stats 2>&1)"; then
    stats_failures=$((stats_failures + 1))
    STATS_ERR="$(printf "%s" "${STATS}" | tail -n 1)"
    echo "[warn] stats failed (attempt=${stats_failures}): ${STATS_ERR}"
    if [[ "${stats_failures}" -ge 20 ]]; then
      echo "[stop] too many consecutive stats failures (${stats_failures})"
      exit_code=1
      break
    fi
    continue
  fi
  stats_failures=0

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
  RS_MANUAL="$(stats_int "${STATS}" resolve_status manual)"
  AN_DONE="$(stats_int "${STATS}" analysis_status done)"
  AN_PENDING="$(stats_int "${STATS}" analysis_status pending)"
  AN_FAILED="$(stats_int "${STATS}" analysis_status failed)"
  AN_DEAD="$(stats_int "${STATS}" analysis_status dead)"
  RESOLVED_DONE="$((RS_DONE + RS_SKIPPED))"
  DEAD_TOTAL="$((TR_DEAD + RS_DEAD + AN_DEAD))"

  echo "[state ${monitor_cycle}] total=${TOTAL_CALLS} tr_done=${TR_DONE} rs_done_or_skipped=${RESOLVED_DONE} an_done=${AN_DONE} pending(t/r/a)=${TR_PENDING}/${RS_PENDING}/${AN_PENDING} failed(t/r/a)=${TR_FAILED}/${RS_FAILED}/${AN_FAILED} manual=${RS_MANUAL} dead=${DEAD_TOTAL}"

  while [[ "${AN_DONE}" -ge "${NEXT_CHECKPOINT}" && "${NEXT_CHECKPOINT}" -le "${TARGET_DONE}" ]]; do
    echo "[progress] analyzed ${NEXT_CHECKPOINT}/${TARGET_DONE}"
    NEXT_CHECKPOINT="$((NEXT_CHECKPOINT + PROGRESS_EVERY))"
  done

  if [[ "${AN_DONE}" -ge "${TARGET_DONE}" ]]; then
    echo "[done] target reached: analyzed_done=${AN_DONE} target=${TARGET_DONE}"
    exit_code=0
    break
  fi

  if [[ "${AUTO_REQUEUE_DEAD}" == "1" && "${DEAD_TOTAL}" -gt 0 ]]; then
    if REQUEUE_PAYLOAD="$("${RUN_CLI}" requeue-dead --stage all --limit 200000 2>&1)"; then
      REQUEUED="$(json_int "${REQUEUE_PAYLOAD}" updated)"
      echo "[requeue] dead_total=${DEAD_TOTAL} updated=${REQUEUED}"
    else
      REQUEUE_ERR="$(printf "%s" "${REQUEUE_PAYLOAD}" | tail -n 1)"
      echo "[warn] requeue failed: ${REQUEUE_ERR}"
    fi
  fi

  if [[ "${TR_PENDING}" -eq 0 && "${RS_PENDING}" -eq 0 && "${AN_PENDING}" -eq 0 && "${TR_FAILED}" -eq 0 && "${RS_FAILED}" -eq 0 && "${AN_FAILED}" -eq 0 ]]; then
    if [[ "${AN_DONE}" -lt "${TARGET_DONE}" ]]; then
      echo "[stop] queue drained before target: analyzed_done=${AN_DONE} target=${TARGET_DONE} manual=${RS_MANUAL}"
      exit_code=1
    else
      exit_code=0
    fi
    break
  fi

  signature="${TR_DONE}:${RESOLVED_DONE}:${AN_DONE}:${TR_PENDING}:${RS_PENDING}:${AN_PENDING}:${TR_FAILED}:${RS_FAILED}:${AN_FAILED}:${RS_MANUAL}"
  if [[ "${signature}" == "${last_signature}" ]]; then
    idle_rounds=$((idle_rounds + 1))
    if [[ "${idle_rounds}" -ge "${MAX_IDLE_ROUNDS}" ]]; then
      echo "[stop] max idle rounds reached (${MAX_IDLE_ROUNDS}) before target"
      echo "[stop] analyzed_done=${AN_DONE} target=${TARGET_DONE}"
      exit_code=1
      break
    fi
  else
    idle_rounds=0
    last_signature="${signature}"
  fi
done

touch "${STOP_FILE}" 2>/dev/null || true
kill "${TR_PID}" 2>/dev/null || true
kill "${POST_PID}" 2>/dev/null || true
wait "${TR_PID}" 2>/dev/null || true
wait "${POST_PID}" 2>/dev/null || true

if FINAL_STATS="$("${RUN_CLI}" stats 2>&1)"; then
  echo "[final] ${FINAL_STATS}"
  printf "%s\n" "${FINAL_STATS}" > "${RUN_DIR}/final_stats.json"
else
  echo "[final] stats unavailable: ${FINAL_STATS}"
  printf "{ \"error\": \"%s\" }\n" "$(printf "%s" "${FINAL_STATS}" | tr '"' "'" | tr '\n' ' ')" > "${RUN_DIR}/final_stats.json"
fi
date "+%Y-%m-%d %H:%M:%S %z" > "${RUN_DIR}/finished_at.txt"

exit "${exit_code}"
