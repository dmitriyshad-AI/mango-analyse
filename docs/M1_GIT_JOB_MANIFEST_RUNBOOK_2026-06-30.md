# M1 Git Job Manifest Runbook

ТЗ-155 переводит быстрые eval-замеры с тяжёлых `mango_clean_*` копий на git checkout + маленький job-манифест. Вотчер M1 не оживляется: запуск ручной.

## One-Time Setup On M1

```bash
cd ~/Projects
git clone ~/Yandex.Disk.localized/OpenClaw/mango_repo.git mango_run
cd mango_run
git remote rename origin yandex
```

Если клон уже есть:

```bash
cd ~/Projects/mango_run
git fetch yandex
```

## Build Job On Main Mac

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/build_job_manifest.py \
  --set product_data/telegram_dynamic_test_sets/pilot_smoke18_2026-06-10.jsonl \
  --snapshot product_data/knowledge_base/kb_release_20260813_v6_8_owner_approved/kb_release_v3_snapshot.json \
  --env-flag TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1 \
  --parallel 4 \
  --max-hours 3 \
  --out-dir "$HOME/Yandex.Disk.localized/OpenClaw/Actual Mango Tests/_jobs"
```

Манифест `job_<sha>.json` содержит `commit_sha`, относительные пути набора и KB snapshot, SHA256 обоих файлов, env-флаги, `parallel`, `max_hours` и готовый `run_cmd`. Размер задания должен быть в килобайтах.

## Manual Run On M1

```bash
JOB="$HOME/Yandex.Disk.localized/OpenClaw/Actual Mango Tests/_jobs/job_<sha>.json"
cd ~/Projects/mango_run
git fetch yandex
SHA="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["commit_sha"])' "$JOB")"
git checkout "$SHA"
test "$(git rev-parse HEAD)" = "$SHA"

SET="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["set_rel_path"])' "$JOB")"
SET_SHA="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["set_sha256"])' "$JOB")"
test "$(shasum -a 256 "$SET" | awk '{print $1}')" = "$SET_SHA"

SNAPSHOT="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["snapshot_rel_path"])' "$JOB")"
SNAPSHOT_SHA="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["snapshot_sha256"])' "$JOB")"
test "$(shasum -a 256 "$SNAPSHOT" | awk '{print $1}')" = "$SNAPSHOT_SHA"

python3 - <<'PY' "$JOB" > /tmp/mango_job_cmd.sh
import json, shlex, sys
job = json.load(open(sys.argv[1]))
print(" ".join(shlex.quote(str(x)) for x in job["run_cmd"]))
PY
bash /tmp/mango_job_cmd.sh
```

## Wappi Replay Exam Set Staging

Для Wappi replay-exam пакетов `PROMPT_M1.md` не должен запускать
`scripts/run_wappi_replay_exam.py` напрямую с `--set "$PKG/set/..."`.
Перед dry-check и full-run набор нужно скопировать в локальный scrubbed-корень
M1, сверить SHA-256 и во всех командах использовать только эту копию:

```bash
M1_SET_DIR="$HOME/.mango_local/replay_exam/scrubbed/<package_id>"
M1_SET="$M1_SET_DIR/replay_exam_set_v1.jsonl"
mkdir -p "$M1_SET_DIR"
cp "$PKG/set/replay_exam_set_v1.jsonl" "$M1_SET"
printf '<set_sha256>  %s\n' "$M1_SET" | shasum -a 256 -c -
```

Линт пакета обязан падать, если команда replay-runner не содержит `--set`
или если значение `--set` не резолвится под
`~/.mango_local/replay_exam/scrubbed`. `$M1_SET_DIR` обязательно добавлять в
retention-пути для удаления после экзамена.

## Heavy Bundle Fallback

`scripts/build_mango_clean_bundle.py` оставлен только как явный аварийный fallback. Его CLI требует `--allow-heavy-bundle`, чтобы случайно не собрать 800МБ-папку.
