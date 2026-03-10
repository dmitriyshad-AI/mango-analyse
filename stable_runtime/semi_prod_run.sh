#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_CLI="${SCRIPT_DIR}/run-cli.sh"
PY_BIN="${SCRIPT_DIR}/venv_stable/bin/python"
EVAL_SCRIPT="${PROJECT_ROOT}/scripts/evaluate_dialogue_quality.py"

PROFILE_PATH="${SCRIPT_DIR}/profiles/semi_prod_dual_asr.env"
INPUT_DIR=""
OUTPUT_DIR=""
DB_PATH="${SCRIPT_DIR}/semi_prod.db"
TARGET_CALLS="400"
STAGE_LIMIT="30"
QC_EVERY="50"
MAX_IDLE_ROUNDS="180"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
WITH_ANALYZE="0"
WITH_SYNC="0"

usage() {
  cat <<'EOF'
Usage:
  stable_runtime/semi_prod_run.sh --input <recordings_dir> --output <transcripts_dir> [options]

Options:
  --db <path>             SQLite DB path (default: stable_runtime/semi_prod.db)
  --target <n>            Number of calls to ingest for this run (default: 400)
  --stage-limit <n>       Batch size for transcribe/analyze/sync loops (default: 30)
  --qc-every <n>          Build QC report every N done transcripts, 0 disables periodic QC (default: 50)
  --profile <path>        Env profile file (default: stable_runtime/profiles/semi_prod_dual_asr.env)
  --run-id <id>           Explicit run id (default: timestamp)
  --max-idle-rounds <n>   Break after N idle rounds with remaining failed/pending (default: 180)
  --with-analyze          Run analyze loop after transcribe
  --with-sync             Run sync loop after analyze (requires --with-analyze)
  -h, --help              Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)
      INPUT_DIR="${2:-}"
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
    --target)
      TARGET_CALLS="${2:-}"
      shift 2
      ;;
    --stage-limit)
      STAGE_LIMIT="${2:-}"
      shift 2
      ;;
    --qc-every)
      QC_EVERY="${2:-}"
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
    --max-idle-rounds)
      MAX_IDLE_ROUNDS="${2:-}"
      shift 2
      ;;
    --with-analyze)
      WITH_ANALYZE="1"
      shift 1
      ;;
    --with-sync)
      WITH_SYNC="1"
      shift 1
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

if [[ -z "${INPUT_DIR}" || -z "${OUTPUT_DIR}" ]]; then
  echo "Both --input and --output are required."
  usage
  exit 2
fi
if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "Input directory does not exist: ${INPUT_DIR}"
  exit 2
fi
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
if [[ ! -f "${EVAL_SCRIPT}" ]]; then
  echo "QC script not found: ${EVAL_SCRIPT}"
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
mkdir -p "$(dirname "${DB_PATH}")"

RUN_DIR="${SCRIPT_DIR}/runs/${RUN_ID}"
mkdir -p "${RUN_DIR}"

set -a
source "${PROFILE_PATH}"
set +a

export PATH="${PROJECT_ROOT}/.local/bin:${PATH}"
export DATABASE_URL="sqlite:///${DB_PATH}"
export TRANSCRIPT_EXPORT_DIR="${OUTPUT_DIR}"

if [[ "${WITH_ANALYZE}" != "1" ]]; then
  export ANALYZE_PROVIDER="mock"
fi
if [[ "${WITH_SYNC}" != "1" ]]; then
  export SYNC_DRY_RUN="1"
fi

cat > "${RUN_DIR}/run_config.txt" <<EOF
run_id=${RUN_ID}
input_dir=${INPUT_DIR}
output_dir=${OUTPUT_DIR}
db_path=${DB_PATH}
target_calls=${TARGET_CALLS}
stage_limit=${STAGE_LIMIT}
qc_every=${QC_EVERY}
profile=${PROFILE_PATH}
with_analyze=${WITH_ANALYZE}
with_sync=${WITH_SYNC}
started_at=$(date "+%Y-%m-%d %H:%M:%S %z")
EOF

json_field() {
  local field="$1"
  "${PY_BIN}" -c "import json,sys; d=json.load(sys.stdin); print(d.get('${field}',0))"
}

stats_field() {
  local payload="$1"
  local group="$2"
  local key="$3"
  "${PY_BIN}" -c "import json,sys; d=json.load(sys.stdin); print(int((d.get('${group}',{}) or {}).get('${key}',0)))" <<< "${payload}"
}

run_qc() {
  local tag="$1"
  local out="${RUN_DIR}/qc_${tag}.json"
  "${PY_BIN}" "${EVAL_SCRIPT}" --transcripts-dir "${OUTPUT_DIR}" --out "${out}" >/dev/null
  local summary
  summary="$("${PY_BIN}" -c "import json,sys; d=json.load(open(sys.argv[1], encoding='utf-8')); s=d.get('summary',{}); print(f\"same_ts_events={s.get('same_ts_cross_speaker_events',0)} residual_near_dup_pairs={s.get('residual_cross_speaker_near_duplicate_pairs',0)} warnings={s.get('warnings_count',0)}\")" "${out}")"
  echo "[qc] ${tag}: ${summary} (${out})"
}

echo "[semi-prod] init-db"
"${RUN_CLI}" init-db | tee "${RUN_DIR}/init_db.json" >/dev/null

echo "[semi-prod] ingest up to ${TARGET_CALLS} calls"
"${RUN_CLI}" ingest --recordings-dir "${INPUT_DIR}" --limit "${TARGET_CALLS}" \
  | tee "${RUN_DIR}/ingest.json" >/dev/null

echo "[semi-prod] transcribe loop started"
cycle=0
idle_rounds=0
next_qc="${QC_EVERY}"
if [[ "${QC_EVERY}" -le 0 ]]; then
  next_qc=999999999
fi

while true; do
  cycle=$((cycle + 1))
  payload="$("${RUN_CLI}" transcribe --limit "${STAGE_LIMIT}")"
  printf "%s\n" "${payload}" > "${RUN_DIR}/transcribe_cycle_${cycle}.json"

  processed="$(json_field processed <<< "${payload}")"
  success="$(json_field success <<< "${payload}")"
  failed="$(json_field failed <<< "${payload}")"

  stats="$("${RUN_CLI}" stats)"
  printf "%s\n" "${stats}" > "${RUN_DIR}/stats_cycle_${cycle}.json"
  done_count="$(stats_field "${stats}" transcription_status done)"
  pending_count="$(stats_field "${stats}" transcription_status pending)"
  failed_count="$(stats_field "${stats}" transcription_status failed)"
  dead_count="$(stats_field "${stats}" transcription_status dead)"

  echo "[transcribe] cycle=${cycle} processed=${processed} success=${success} failed=${failed} done=${done_count} pending=${pending_count} failed_status=${failed_count} dead=${dead_count}"

  while [[ "${QC_EVERY}" -gt 0 && "${done_count}" -ge "${next_qc}" ]]; do
    qc_mark="${next_qc}"
    run_qc "done_${qc_mark}"
    next_qc=$((next_qc + QC_EVERY))
  done

  if [[ "${processed}" -eq 0 ]]; then
    idle_rounds=$((idle_rounds + 1))
    if [[ "${pending_count}" -eq 0 && "${failed_count}" -eq 0 ]]; then
      echo "[transcribe] queue drained"
      break
    fi
    if [[ $((done_count + dead_count)) -ge "${TARGET_CALLS}" ]]; then
      echo "[transcribe] reached target boundary: done+dead >= target"
      break
    fi
    if [[ "${idle_rounds}" -ge "${MAX_IDLE_ROUNDS}" ]]; then
      echo "[transcribe] max idle rounds reached (${MAX_IDLE_ROUNDS}), stopping loop"
      break
    fi
    sleep 10
  else
    idle_rounds=0
  fi
done

run_qc "final"

if [[ "${WITH_ANALYZE}" == "1" ]]; then
  echo "[semi-prod] analyze loop started"
  while true; do
    payload="$("${RUN_CLI}" analyze --limit "${STAGE_LIMIT}")"
    processed="$(json_field processed <<< "${payload}")"
    echo "[analyze] processed=${processed}"
    if [[ "${processed}" -eq 0 ]]; then
      break
    fi
  done
fi

if [[ "${WITH_SYNC}" == "1" ]]; then
  echo "[semi-prod] sync loop started"
  while true; do
    payload="$("${RUN_CLI}" sync --limit "${STAGE_LIMIT}")"
    processed="$(json_field processed <<< "${payload}")"
    echo "[sync] processed=${processed}"
    if [[ "${processed}" -eq 0 ]]; then
      break
    fi
  done
fi

echo "[semi-prod] done"
"${RUN_CLI}" stats | tee "${RUN_DIR}/final_stats.json"
date "+%Y-%m-%d %H:%M:%S %z" > "${RUN_DIR}/finished_at.txt"
