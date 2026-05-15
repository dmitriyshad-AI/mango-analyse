# Runbook

Дата обновления: 2026-05-15

Назначение: безопасные команды и рабочие правила для проекта.

## Перед любой работой

Проверить ветку и рабочую папку:

```bash
git branch --show-current
git status --short
```

Посмотреть актуальное состояние:

```bash
sed -n '1,220p' docs/CURRENT_STATE.md
sed -n '1,220p' docs/DECISIONS_LOG.md
sed -n '1,220p' docs/ROADMAP.md
```

## Безопасный сбор тестов

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest --collect-only -q
```

## Безопасный точечный запуск тестов

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q <tests>
```

Пример:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_post_backfill_amo_ready_export.py
```

## Что не запускать без отдельного подтверждения

- ASR;
- Resolve+Analyze по реальным данным;
- live AMO write;
- live CRM write;
- Tallanto write;
- массовые batch/start/run-ui скрипты;
- скрипты, которые пишут в `stable_runtime` как рабочее состояние;
- удаление или перенос runtime-папок.

## AMO Snapshot / Rollback

Live-запись deal-aware полей теперь обязана создать:

```text
pre_write_snapshot.jsonl
pre_write_snapshot.csv
rollback_manifest.json
live_write_report.csv
live_write_report.json
summary.json
```

Rollback по умолчанию запускать только в dry-run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/rollback_deal_aware_amo_fields.py \
  --live-run-root <live_run_root> \
  --dry-run
```

Реальный rollback запрещен без отдельного подтверждения Дмитрия. Для `--apply` нужен отдельный token:

```text
ROLLBACK_DEAL_AWARE_AMO_FIELDS
```

Первый live-микропилот после Блока A ограничен 1-5 сделками.

## Runtime

Текущие указатели можно читать:

```bash
sed -n '1,220p' stable_runtime/CURRENT_RUNTIME.json
cat stable_runtime/CANONICAL_EXPORT.txt
```

Но нельзя менять `stable_runtime` DB/audio/transcripts без отдельного подтверждения.

## Audit Pack

После значимого блока создать:

```text
audits/_inbox/<block>_<timestamp>/
```

Минимальные файлы:

```text
implementation_notes.md
changed_files.txt
test_output.txt
risk_review.md
backward_compatibility.md
```

Для AMO/writeback блоков добавить:

```text
snapshot_contract.md
rollback_contract.md
dry_run_summary.md
readback_plan.md
live_write_status.md
```

Если live-write не запускался, явно написать:

```text
Live write was not executed.
```

## Коммиты

Коммитить только после:

1. понятного diff;
2. тестов;
3. audit pack;
4. проверки, что в коммит не попали runtime-артефакты;
5. проверки, что нет чужих unrelated изменений.

Не смешивать в один коммит:

- код разных блоков;
- документы и runtime-выгрузки;
- cleanup и функциональную реализацию;
- live отчеты и исходный код.

## Текущий порядок реализации

```text
G -> A -> PBF -> B -> C -> D -> E
```

Где:

- `G` - git-границы и рабочее состояние;
- `A` - AMO pre-write snapshot и rollback;
- `PBF` - красный post-backfill тест;
- `B` - коммерческие поля;
- `C` - структурные возражения;
- `D` - связь каталога вопросов и deal-aware gate;
- `E` - customer timeline как read-only источник истории клиента.

## Customer Timeline Coverage

Безопасный read-only отчет покрытия:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/audit_customer_timeline_coverage.py \
  --deal-aware-candidates <path/to/deal_stage4_deal_candidates.csv> \
  --timeline-db <path/to/customer_timeline.sqlite> \
  --out-root <path/to/audit_output>
```

Нельзя писать output в `stable_runtime` без отдельного подтверждения.

Stage 4 preview может читать customer timeline только по явному флагу:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/build_deal_aware_stage4_preview.py \
  --customer-timeline-db <path/to/customer_timeline.sqlite> \
  --enable-customer-timeline-context
```

Этот контекст остается только в preview/report полях и не попадает в AMO payload.

## Навыки Codex

Для security-аудита использовать skills:

- `security-best-practices`;
- `security-threat-model`;
- `security-ownership-map`.

Для PDF:

- `pdf`;
- Python-среда: `/Users/dmitrijfabarisov/.codex/skill-venv/bin/python`.

Для notebooks:

- `jupyter-notebook`;
- Python-среда: `/Users/dmitrijfabarisov/.codex/skill-venv/bin/python`.

Для долговечных CLI:

- `cli-creator`.

Не использовать эти skills для простых вопросов и мелких правок.
