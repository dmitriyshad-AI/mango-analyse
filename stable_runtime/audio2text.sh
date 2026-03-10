#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_CLI="${SCRIPT_DIR}/run-cli.sh"

if [[ ! -x "${RUN_CLI}" ]]; then
  echo "Missing launcher: ${RUN_CLI}"
  exit 1
fi

INPUT_DIR=""
OUTPUT_DIR=""
DB_PATH="${SCRIPT_DIR}/audio2text.db"
STAGE_LIMIT="200"
PROVIDER="mlx"
LANGUAGE="ru"
DUAL_MODE="0"
SECONDARY_PROVIDER="gigaam"
MERGE_PROVIDER="rule"
MONO_ASSIGN_MODE="off"
MONO_ASSIGN_MIN_CONF="0.62"
MONO_ASSIGN_LLM_THRESHOLD="0.72"
ROLE_ASSIGN_MODEL="gpt-4o-mini"
OLLAMA_BASE_URL="http://127.0.0.1:11434"
OLLAMA_MODEL="gpt-oss:20b"
OLLAMA_THINK="medium"
OLLAMA_TEMPERATURE="0"
MLX_WORD_TIMESTAMPS="1"

usage() {
  cat <<'EOF'
Usage:
  stable_runtime/audio2text.sh --input <recordings_dir> --output <transcripts_dir> [options]

Options:
  --db <path>                 SQLite db path (default: stable_runtime/audio2text.db)
  --stage-limit <n>           Batch size per transcribe cycle (default: 200)
  --provider <mlx|gigaam|openai|mock>
  --language <code>           ASR language (default: ru)
  --mlx-word-ts <0|1>         Enable MLX word timestamps (default: 1)
  --dual                      Enable second ASR pass
  --secondary <provider>      Secondary provider for dual mode (default: gigaam)
  --merge <rule|ollama|openai|codex_cli|primary>
  --mono-assign <mode>        off|rule|ollama_selective|openai_selective (default: off)
  --mono-min-conf <0..1>      minimum confidence to apply role split (default: 0.62)
  --mono-llm-thr <0..1>       LLM trigger threshold for selective modes (default: 0.72)
  --role-model <model>        OpenAI model for role assignment (default: gpt-4o-mini)
  --ollama-base-url <url>     Ollama API base URL (default: http://127.0.0.1:11434)
  --ollama-model <model>       Ollama model (default: gpt-oss:20b)
  --ollama-think <level>       Ollama think: low|medium|high (default: medium)
  --ollama-temp <value>        Ollama temperature (default: 0)
  -h, --help                  Show help
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
    --stage-limit)
      STAGE_LIMIT="${2:-}"
      shift 2
      ;;
    --provider)
      PROVIDER="${2:-}"
      shift 2
      ;;
    --language)
      LANGUAGE="${2:-}"
      shift 2
      ;;
    --mlx-word-ts)
      MLX_WORD_TIMESTAMPS="${2:-}"
      shift 2
      ;;
    --dual)
      DUAL_MODE="1"
      shift 1
      ;;
    --secondary)
      SECONDARY_PROVIDER="${2:-}"
      shift 2
      ;;
    --merge)
      MERGE_PROVIDER="${2:-}"
      shift 2
      ;;
    --mono-assign)
      MONO_ASSIGN_MODE="${2:-}"
      shift 2
      ;;
    --mono-min-conf)
      MONO_ASSIGN_MIN_CONF="${2:-}"
      shift 2
      ;;
    --mono-llm-thr)
      MONO_ASSIGN_LLM_THRESHOLD="${2:-}"
      shift 2
      ;;
    --role-model)
      ROLE_ASSIGN_MODEL="${2:-}"
      shift 2
      ;;
    --ollama-base-url)
      OLLAMA_BASE_URL="${2:-}"
      shift 2
      ;;
    --ollama-model)
      OLLAMA_MODEL="${2:-}"
      shift 2
      ;;
    --ollama-think)
      OLLAMA_THINK="${2:-}"
      shift 2
      ;;
    --ollama-temp)
      OLLAMA_TEMPERATURE="${2:-}"
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

if [[ -z "${INPUT_DIR}" || -z "${OUTPUT_DIR}" ]]; then
  echo "Both --input and --output are required."
  usage
  exit 2
fi

if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "Input directory does not exist: ${INPUT_DIR}"
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
mkdir -p "$(dirname "${DB_PATH}")"

export DATABASE_URL="sqlite:///${DB_PATH}"
export TRANSCRIPT_EXPORT_DIR="${OUTPUT_DIR}"
export TRANSCRIBE_PROVIDER="${PROVIDER}"
export TRANSCRIBE_LANGUAGE="${LANGUAGE}"
export MLX_WORD_TIMESTAMPS="${MLX_WORD_TIMESTAMPS}"
export SPLIT_STEREO_CHANNELS="1"
export DUAL_TRANSCRIBE_ENABLED="${DUAL_MODE}"
export SECONDARY_TRANSCRIBE_PROVIDER="${SECONDARY_PROVIDER}"
export DUAL_MERGE_PROVIDER="${MERGE_PROVIDER}"
export MONO_ROLE_ASSIGNMENT_MODE="${MONO_ASSIGN_MODE}"
export MONO_ROLE_ASSIGNMENT_MIN_CONFIDENCE="${MONO_ASSIGN_MIN_CONF}"
export MONO_ROLE_ASSIGNMENT_LLM_THRESHOLD="${MONO_ASSIGN_LLM_THRESHOLD}"
export OPENAI_ROLE_ASSIGN_MODEL="${ROLE_ASSIGN_MODEL}"
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL}"
export OLLAMA_MODEL="${OLLAMA_MODEL}"
export OLLAMA_THINK="${OLLAMA_THINK}"
export OLLAMA_TEMPERATURE="${OLLAMA_TEMPERATURE}"
export ANALYZE_PROVIDER="mock"
export SYNC_DRY_RUN="1"

echo "[1/3] init-db"
"${RUN_CLI}" init-db >/dev/null

echo "[2/3] ingest"
"${RUN_CLI}" ingest --recordings-dir "${INPUT_DIR}" >/dev/null

echo "[3/3] transcribe loop"
while true; do
  PAYLOAD="$("${RUN_CLI}" transcribe --limit "${STAGE_LIMIT}")"
  PROCESSED="$(printf "%s" "${PAYLOAD}" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(int(d.get("processed",0)))')"
  SUCCESS="$(printf "%s" "${PAYLOAD}" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(int(d.get("success",0)))')"
  FAILED="$(printf "%s" "${PAYLOAD}" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(int(d.get("failed",0)))')"
  echo "processed=${PROCESSED} success=${SUCCESS} failed=${FAILED}"
  if [[ "${PROCESSED}" -eq 0 ]]; then
    break
  fi
done

echo "Done. Transcripts:"
echo "  ${OUTPUT_DIR}"
echo "DB:"
echo "  ${DB_PATH}"
