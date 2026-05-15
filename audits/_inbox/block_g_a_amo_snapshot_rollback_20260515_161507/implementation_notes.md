# Implementation Notes

Дата: 2026-05-15
Блоки: G + A
Статус: выполнено без live-запуска AMO

## Что сделано

1. Зафиксированы git-границы текущей параллельной разработки:
   - создан `docs/CURRENT_DEVELOPMENT_BOUNDARIES_2026-05-15.md`;
   - текущий ТЗ ссылается на эту карту;
   - runtime-артефакты и unrelated направления отделены от Блока A.

2. Добавлен чистый слой AMO snapshot/rollback:
   - `src/mango_mvp/deal_aware/amo_rollback.py`;
   - сбор старых значений AMO lead fields;
   - запись `pre_write_snapshot.jsonl` и `pre_write_snapshot.csv`;
   - `rollback_manifest.json`;
   - retry/backoff для 429/5xx;
   - resume по успешным строкам;
   - защита от стирания ручных правок менеджера.

3. Обновлен live-write скрипт:
   - `scripts/write_deal_aware_amo_fields.py`;
   - перед PATCH теперь сначала читает текущий lead;
   - сохраняет snapshot;
   - если snapshot не сохранен, PATCH не вызывается;
   - добавлены `--batch-size`, `--delay-ms`, `--max-retries`, `--resume-from-report`;
   - добавлены `live_write_report.csv` и `live_write_report.json`;
   - промежуточные отчеты сохраняются после обработки строк.

4. Добавлен rollback-скрипт:
   - `scripts/rollback_deal_aware_amo_fields.py`;
   - dry-run по умолчанию;
   - `--apply` требует отдельный token `ROLLBACK_DEAL_AWARE_AMO_FIELDS`;
   - rollback восстанавливает только поля из snapshot;
   - если текущее значение уже изменено менеджером, поле пропускается;
   - если старое значение пустое, строка уходит в `manual_restore_required`.

5. В коммит включается минимальный deal-aware Stage6 baseline, без которого A-блок не будет воспроизводимым в чистой ветке:
   - `src/mango_mvp/deal_aware/__init__.py`;
   - `src/mango_mvp/deal_aware/deal_text_builder.py`;
   - `src/mango_mvp/deal_aware/deal_writeback.py`;
   - `src/mango_mvp/deal_aware/stage1_snapshot.py`;
   - `scripts/readback_deal_aware_amo_fields.py`;
   - `tests/test_deal_aware_stage6_writeback.py`.

6. Обновлены документы состояния:
   - `docs/CURRENT_STATE.md`;
   - `docs/DECISIONS_LOG.md`;
   - `docs/ROADMAP.md`;
   - `docs/RUNBOOK.md`.

## Что не делалось

- Не запускался live-write в AMO.
- Не запускался реальный rollback.
- Не запускался ASR.
- Не запускался Resolve+Analyze по реальным данным.
- Не менялись `stable_runtime` DB/audio/transcripts.
- Не удалялись файлы и папки.

## Известное осознанное решение

Очистка AMO-поля при rollback не реализована через существующий `build_custom_fields_values()`, потому что этот helper сейчас намеренно пропускает пустые строки. Чтобы не менять старую совместимость, пустой `old_value` помечается как `manual_restore_required`.
