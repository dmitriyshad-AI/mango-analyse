# Implementation Notes

Дата: 2026-05-15
Блок: PBF
Статус: выполнено

## Что сделано

Исправлено дублирование одиночного последнего содержательного звонка в post-backfill истории контакта.

Правило теперь такое:

- если содержательный звонок один, `Краткая история общения` содержит компактную сводку, а `Хронология общения (последние 5 касаний)` остается пустой;
- если содержательных звонков два и больше, хронология сохраняется в порядке от старого касания к новому.

## Файлы

- `scripts/build_post_backfill_amo_ready_export.py`
- `tests/test_post_backfill_amo_ready_export.py`
- `docs/CURRENT_STATE.md`
- `docs/DECISIONS_LOG.md`
- `docs/ROADMAP.md`

## Что не делалось

- Не запускался ASR.
- Не запускался Resolve+Analyze.
- Не было live AMO/CRM/Tallanto write.
- `stable_runtime` не менялся.
