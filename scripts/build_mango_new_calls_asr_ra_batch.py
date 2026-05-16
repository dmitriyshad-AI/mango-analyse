#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = PROJECT_ROOT / "product_data/mango_canonical_rebuild_plan_20260516_v1/needs_asr_ra_21.csv"
DEFAULT_OUT = PROJECT_ROOT / "product_data/mango_new_21_asr_ra_20260516_v1"


def clean_part(value: str) -> str:
    value = (value or "").strip()
    value = re.sub(r"[\\/:\n\r\t]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value[:120] or "unknown"


def decode_provider_call_id(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return ""
    try:
        padded = raw + "=" * (-len(raw) % 4)
        decoded = base64.b64decode(padded).decode("utf-8").strip()
        if decoded.isdigit():
            return decoded
    except Exception:
        pass
    return raw


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hardlink_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if sha256_file(src) != sha256_file(dst):
            raise RuntimeError(f"target exists with different content: {dst}")
        return "exists_same_hash"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def started_parts(started_at_utc: str) -> tuple[str, str, str]:
    raw = (started_at_utc or "").strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt.strftime("%Y-%m-%d"), dt.strftime("%H-%M-%S"), dt.isoformat(sep=" ")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def shell_quote(path: Path | str) -> str:
    return "'" + str(path).replace("'", "'\"'\"'") + "'"


def write_launcher_scripts(out_dir: Path, db_path: Path) -> None:
    audio_dir = out_dir / "audio"
    transcripts_dir = out_dir / "transcripts"
    metadata_csv = out_dir / "metadata.csv"
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    common = f"""#!/bin/zsh
set -euo pipefail

PROJECT_ROOT={shell_quote(PROJECT_ROOT)}
BATCH_ROOT={shell_quote(out_dir)}
DB_PATH={shell_quote(db_path)}
AUDIO_DIR={shell_quote(audio_dir)}
METADATA_CSV={shell_quote(metadata_csv)}
TRANSCRIPTS_DIR={shell_quote(transcripts_dir)}
LOGS_DIR={shell_quote(logs_dir)}

_pick_python() {{
  local candidates=(
    "/usr/bin/python3"
    "${{PROJECT_ROOT}}/.venv/bin/python"
    "${{PROJECT_ROOT}}/stable_runtime/venv_stable/bin/python"
  )
  for py in "${{candidates[@]}}"; do
    [[ -x "${{py}}" ]] || continue
    PYTHONPATH="${{PROJECT_ROOT}}/src" "${{py}}" - <<'PY' >/dev/null 2>&1
import importlib
for mod in ("sqlalchemy", "dotenv", "mango_mvp.cli", "openai"):
    importlib.import_module(mod)
PY
    if [[ $? -eq 0 ]]; then
      echo "${{py}}"
      return 0
    fi
  done
  return 1
}}

VENV_PYTHON="$(_pick_python)"
mkdir -p "${{TRANSCRIPTS_DIR}}" "${{LOGS_DIR}}"
cd "${{PROJECT_ROOT}}"
"""

    (out_dir / "run_01_init_ingest.sh").write_text(
        common
        + """
DATABASE_URL="sqlite:///${DB_PATH}" \\
PYTHONPATH="${PROJECT_ROOT}/src" \\
PYTHONDONTWRITEBYTECODE=1 \\
"${VENV_PYTHON}" -m mango_mvp.cli init-db

DATABASE_URL="sqlite:///${DB_PATH}" \\
PYTHONPATH="${PROJECT_ROOT}/src" \\
PYTHONDONTWRITEBYTECODE=1 \\
"${VENV_PYTHON}" -m mango_mvp.cli ingest \\
  --recordings-dir "${AUDIO_DIR}" \\
  --metadata-csv "${METADATA_CSV}"

DATABASE_URL="sqlite:///${DB_PATH}" \\
PYTHONPATH="${PROJECT_ROOT}/src" \\
PYTHONDONTWRITEBYTECODE=1 \\
"${VENV_PYTHON}" -m mango_mvp.cli stats
""",
        encoding="utf-8",
    )

    (out_dir / "run_02_transcribe_21.sh").write_text(
        common
        + """
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${LOGS_DIR}/transcribe_${RUN_ID}.log"
CODEX_HOME_ROOT="/private/tmp/mango_codex_home_transcribe21_${RUN_ID}"
mkdir -p "${CODEX_HOME_ROOT}/worker/sessions" "${CODEX_HOME_ROOT}/worker/logs" "${CODEX_HOME_ROOT}/worker/tmp"
for f in auth.json config.toml AGENTS.md installation_id models_cache.json; do
  if [[ -f "${HOME}/.codex/${f}" ]]; then
    cp "${HOME}/.codex/${f}" "${CODEX_HOME_ROOT}/worker/${f}"
  fi
done

echo "Transcribing 21 calls. Log: ${LOG_PATH}"
DATABASE_URL="sqlite:///${DB_PATH}" \\
TRANSCRIPT_EXPORT_DIR="${TRANSCRIPTS_DIR}" \\
CODEX_HOME="${CODEX_HOME_ROOT}/worker" \\
PYTHONPATH="${PROJECT_ROOT}/src" \\
PYTHONDONTWRITEBYTECODE=1 \\
PYTHONPYCACHEPREFIX="/private/tmp/mango_pycache_transcribe21_${RUN_ID}" \\
PATH="/Applications/Codex.app/Contents/Resources:${PROJECT_ROOT}/.local/bin:${PATH}" \\
SQLITE_BUSY_TIMEOUT_MS=60000 \\
TRANSCRIBE_PROVIDER=mlx \\
DUAL_TRANSCRIBE_ENABLED=1 \\
SECONDARY_TRANSCRIBE_PROVIDER=gigaam \\
DUAL_MERGE_PROVIDER=codex_cli \\
TRANSCRIBE_LANGUAGE=ru \\
SPLIT_STEREO_CHANNELS=1 \\
GIGAAM_DEVICE=cpu \\
CODEX_CLI_COMMAND="/opt/homebrew/bin/codex" \\
CODEX_MERGE_MODEL="gpt-5.4" \\
CODEX_REASONING_EFFORT=medium \\
CODEX_CLI_TIMEOUT_SEC=360 \\
"${VENV_PYTHON}" -m mango_mvp.cli worker \\
  --stages "transcribe,backfill-second-asr" \\
  --stage-limit 1 \\
  --poll-sec 5 \\
  --max-idle-cycles 5 | tee "${LOG_PATH}"
""",
        encoding="utf-8",
    )

    (out_dir / "run_03_resolve_analyze_21.sh").write_text(
        common
        + """
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${LOGS_DIR}/parallel_ra_${RUN_ID}"
CODEX_HOME_ROOT="/private/tmp/mango_codex_home_ra21_${RUN_ID}"
mkdir -p "${RUN_DIR}" "${CODEX_HOME_ROOT}"

_prepare_codex_home() {
  local worker_name="$1"
  local worker_home="${CODEX_HOME_ROOT}/${worker_name}"
  mkdir -p "${worker_home}/sessions" "${worker_home}/logs" "${worker_home}/tmp"
  for f in auth.json config.toml AGENTS.md installation_id models_cache.json; do
    if [[ -f "${HOME}/.codex/${f}" ]]; then
      cp "${HOME}/.codex/${f}" "${worker_home}/${f}"
    fi
  done
  echo "${worker_home}"
}

_run_worker() {
  local worker_name="$1"
  local stages="$2"
  local log_path="${RUN_DIR}/${worker_name}.log"
  local codex_home
  codex_home="$(_prepare_codex_home "${worker_name}")"
  {
    echo "started_at=$(date -Iseconds)"
    echo "worker=${worker_name}"
    echo "stages=${stages}"
    DATABASE_URL="sqlite:///${DB_PATH}" \\
    TRANSCRIPT_EXPORT_DIR="${TRANSCRIPTS_DIR}" \\
    CODEX_HOME="${codex_home}" \\
    PYTHONPATH="${PROJECT_ROOT}/src" \\
    PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONPYCACHEPREFIX="/private/tmp/mango_pycache_${worker_name}_${RUN_ID}" \\
    PATH="/Applications/Codex.app/Contents/Resources:${PROJECT_ROOT}/.local/bin:${PATH}" \\
    SQLITE_BUSY_TIMEOUT_MS=60000 \\
    RESOLVE_LLM_PROVIDER=codex_cli \\
    RESOLVE_DIALOGUE_MODE=dialogue \\
    RESOLVE_RESCUE_PROVIDER=none \\
    RESOLVE_RESCUE_DUAL_ENABLED=0 \\
    ANALYZE_PROVIDER=codex_cli \\
    CODEX_CLI_COMMAND="/opt/homebrew/bin/codex" \\
    CODEX_RESOLVE_MODEL="gpt-5.4" \\
    CODEX_ANALYZE_MODEL="gpt-5.4-mini" \\
    CODEX_REASONING_EFFORT=medium \\
    CODEX_CLI_TIMEOUT_SEC=360 \\
    "${VENV_PYTHON}" -m mango_mvp.cli worker \\
      --stages "${stages}" \\
      --stage-limit 1 \\
      --poll-sec 5 \\
      --max-idle-cycles 60
    echo "finished_at=$(date -Iseconds)"
  } >> "${log_path}" 2>&1
}

pids=()
for idx in 1 2; do
  _run_worker "resolve_${idx}" "resolve" &
  pids+=("$!")
done
for idx in 1 2 3 4; do
  _run_worker "analyze_${idx}" "analyze" &
  pids+=("$!")
done
printf '%s\\n' "${pids[@]}" > "${RUN_DIR}/pids.txt"
echo "Parallel Resolve+Analyze started"
echo "run_dir=${RUN_DIR}"
trap 'echo "Stopping workers"; kill "${pids[@]}" 2>/dev/null || true' INT TERM
wait "${pids[@]}"
echo "Parallel Resolve+Analyze finished at $(date -Iseconds)"
""",
        encoding="utf-8",
    )

    (out_dir / "run_04_stats.sh").write_text(
        common
        + """
DATABASE_URL="sqlite:///${DB_PATH}" \\
PYTHONPATH="${PROJECT_ROOT}/src" \\
PYTHONDONTWRITEBYTECODE=1 \\
"${VENV_PYTHON}" -m mango_mvp.cli stats
""",
        encoding="utf-8",
    )

    for script in out_dir.glob("run_*.sh"):
        script.chmod(0o755)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-csv", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    plan_csv = args.plan_csv.resolve()
    out_dir = args.out_dir.resolve()
    audio_dir = out_dir / "audio"
    db_path = out_dir / "mango_new_21_asr_ra.sqlite"

    with plan_csv.open("r", encoding="utf-8-sig", newline="") as fh:
        source_rows = list(csv.DictReader(fh))

    rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    link_actions: dict[str, int] = {}

    for idx, row in enumerate(source_rows, start=1):
        audio_src = (PROJECT_ROOT / row["audio_store_path"]).resolve()
        if not audio_src.is_file():
            raise FileNotFoundError(audio_src)
        date_part, time_part, started_at = started_parts(row["started_at_utc"])
        manager = clean_part(row.get("manager_name") or "Неизвестный менеджер")
        phone = (row.get("phone") or "").strip()
        phone_digits = re.sub(r"\D+", "", phone) or "no_phone"
        call_id = decode_provider_call_id(row.get("provider_call_id") or "")
        filename = f"{date_part}__{time_part}__{manager}__{phone_digits}_{call_id}.mp3"
        target = audio_dir / filename
        action = hardlink_or_copy(audio_src, target)
        link_actions[action] = link_actions.get(action, 0) + 1

        manifest_row = {
            **row,
            "batch_index": idx,
            "batch_audio_file": str(target.relative_to(PROJECT_ROOT)),
            "batch_filename": filename,
            "decoded_call_id": call_id,
            "started_at_for_ingest": started_at,
            "link_action": action,
            "batch_db": str(db_path.relative_to(PROJECT_ROOT)),
        }
        rows.append(manifest_row)
        metadata_rows.append(
            {
                "filename": filename,
                "call_id": call_id,
                "provider_call_id": row.get("provider_call_id") or "",
                "recording_id": row.get("recording_id") or "",
                "phone": phone,
                "manager": manager,
                "manager_ref": row.get("manager_ref") or "",
                "manager_quality": row.get("manager_quality") or "",
                "started_at": started_at,
                "direction": "",
                "source_event_key": row.get("event_key") or "",
                "source_manifest": row.get("source_manifest") or "",
                "source_audio_store_path": row.get("audio_store_path") or "",
                "source_original_audio_path": row.get("original_audio_path") or "",
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_fields = list(rows[0].keys()) if rows else []
    metadata_fields = list(metadata_rows[0].keys()) if metadata_rows else []
    write_csv(out_dir / "asr_ra_21_manifest.csv", rows, manifest_fields)
    write_jsonl(out_dir / "asr_ra_21_manifest.jsonl", rows)
    write_csv(out_dir / "metadata.csv", metadata_rows, metadata_fields)
    write_launcher_scripts(out_dir, db_path)

    summary = {
        "schema_version": "mango_new_21_asr_ra_batch_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "plan_csv": str(plan_csv),
        "out_dir": str(out_dir),
        "db_path": str(db_path),
        "audio_dir": str(audio_dir),
        "rows": len(rows),
        "link_actions": link_actions,
        "manager_quality_counts": {},
        "manager_counts": {},
        "safety": {
            "stable_runtime_writes": False,
            "crm_writes": False,
            "tallanto_writes": False,
            "asr_started_by_builder": False,
            "resolve_analyze_started_by_builder": False,
        },
        "next_scripts": [
            str((out_dir / "run_01_init_ingest.sh").relative_to(PROJECT_ROOT)),
            str((out_dir / "run_02_transcribe_21.sh").relative_to(PROJECT_ROOT)),
            str((out_dir / "run_03_resolve_analyze_21.sh").relative_to(PROJECT_ROOT)),
            str((out_dir / "run_04_stats.sh").relative_to(PROJECT_ROOT)),
        ],
    }
    for row in rows:
        mq = row.get("manager_quality") or ""
        mn = row.get("manager_name") or ""
        summary["manager_quality_counts"][mq] = summary["manager_quality_counts"].get(mq, 0) + 1
        summary["manager_counts"][mn] = summary["manager_counts"].get(mn, 0) + 1
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "README.md").write_text(
        "# Mango new 21 ASR/R+A batch\n\n"
        "Изолированный батч для 21 майского звонка, у которых есть аудио, но нет ASR/R+A.\n\n"
        "Порядок запуска:\n"
        "1. `run_01_init_ingest.sh` - создать SQLite-БД и загрузить 21 запись.\n"
        "2. `run_02_transcribe_21.sh` - ASR: MLX Whisper + GigaAM, без CRM/Tallanto.\n"
        "3. `run_03_resolve_analyze_21.sh` - Resolve+Analyze через Codex CLI, без sync.\n"
        "4. `run_04_stats.sh` - проверить статусы.\n\n"
        "Папка не меняет `stable_runtime` и не пишет во внешние системы.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
