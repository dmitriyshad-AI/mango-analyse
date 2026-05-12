# Transcript Quality Phase 5 Full Dry Run Report

Дата: 2026-05-09.

## Что сделано

Выполнен read-only full dry-run hard-gate нормализации по всем БД из финального coverage:

`stable_runtime/final_processing_coverage_report_20260507_v5/included_dbs.tsv`

Выходная папка:

`stable_runtime/non_conversation_hard_gate_owner_dry_run_20260509_v5_gpt_policy_preview/`

Базы данных не изменялись. SQLite открывались в read-only режиме.

## Итоговые цифры

| Метрика | Baseline v4 | Phase 5 run |
|---|---:|---:|
| Terminal rows selected | 64 832 | 68 771 |
| Rows normalized | 64 832 | 68 771 |
| Would update | 5 404 | 5 761 |
| Unchanged | 59 428 | 63 010 |
| Parse errors | 0 | 0 |
| Protected live dialogues | 42 669 | 45 288 |

Рост корпуса с 64 832 до 68 771 не является регрессией: после baseline v4 часть RA/analysis статусов дозавершилась, поэтому текущий dry-run увидел больше terminal rows.

## Что именно предлагается исправить

Все `5 761` кандидатов нормализуются в `non_conversation`.

Разбивка старых типов:

| Старый тип | Количество |
|---|---:|
| service_call | 2 785 |
| technical_call | 2 528 |
| sales_call | 422 |
| existing_client_progress | 26 |

Разбивка подтипов:

| Подтип | Количество |
|---|---:|
| no_live_or_voicemail | 4 540 |
| outbound_voicemail | 1 221 |

## По месяцам

| Месяц | Selected | Would update |
|---|---:|---:|
| 2025-01 | 1 299 | 117 |
| 2025-02 | 3 132 | 468 |
| 2025-03 | 3 020 | 244 |
| 2025-04 | 2 857 | 312 |
| 2025-05 | 2 299 | 207 |
| 2025-06 | 2 456 | 151 |
| 2025-07 | 2 440 | 109 |
| 2025-08 | 6 436 | 268 |
| 2025-09 | 12 796 | 1 314 |
| 2025-10 | 6 549 | 571 |
| 2025-11 | 4 467 | 466 |
| 2025-12 | 3 809 | 279 |
| 2026-01 | 2 715 | 196 |
| 2026-02 | 6 081 | 602 |
| 2026-03 | 3 560 | 162 |
| 2026-04 | 4 396 | 283 |
| 2026-05 | 456 | 12 |

## Вывод

Фаза 5 успешно завершена: полный список deterministic hard-gate candidates собран, parse errors нет.

Важно: этот dry-run не вызывал GPT по всем `5 761` кандидатам. Production policy `hard_gate_gpt_policy_v1` уже реализована, но для полного корпуса следующий шаг должен построить apply-plan и решить, какие кандидаты требуют GPT-review перед apply.

## Следующий шаг

Фаза 6: построить full-corpus apply-plan:

1. взять `would_update_candidates.csv` из phase 5;
2. подготовить очередь GPT-only review для `5 761` кандидатов или явно выделить deterministic auto-apply subset;
3. разделить `auto_apply`, `gpt_review_required`, `manual_review`, `blocked`;
4. не писать в SQLite до backup/rollback manifest.
