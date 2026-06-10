# TZ-12 D4: поставка 1 (РП-0..РП-2)

Дата: 2026-06-10  
Ветка: `codex/tz12-d4-client-history-profile`

## Коммиты

- `fc7e9b8f` — РП-0: git-гигиена ПДн (`all_whatsapp_chats.txt`, `product_data/customer_profiles/`) + NEG `git check-ignore`.
- `2ef64732` — РП-1: `analysis_meta` для свежего analyze; миграция старых `analysis_json` не добавляет мета.
- `cc04121b` — РП-2: A/B runner, `analysis_model`/`analysis_prompt_version` в отчётах, совместимость с `canonical_calls`, безопасные обезличенные отчёты.

## A/B Analyze

Путь к audit-пакету: `audits/_inbox/ab_analyze_2026-06-10/final_report/`

Источник: `stable_runtime/canonical_master_20260523_audio_working_store_v1/canonical_calls_master.db`, read-only; каждое плечо работало на своей SQLite-копии в `audits/_inbox`.

Выборка: 50 содержательных звонков, `started_at >= 2026-01-01`, `duration_sec >= 120`, без `non_conversation`.

| Плечо | Модель | Промпт | Успех | target_product | next_step | objections | summary |
|---|---|---|---:|---:|---:|---:|---:|
| `mini_v6` | `gpt-5.4-mini` | compact (`v6,v7`) | 50/50 | 80% | 96% | 98% | 100% |
| `mini_v7` | `gpt-5.4-mini` | full (`v7`) | 50/50 | 86% | 98% | 100% | 100% |
| `gpt54_v6` | `gpt-5.4` | compact (`v6,v7`) | 50/50 | 76% | 98% | 98% | 100% |
| `gpt55_v6` | `gpt-5.5` | compact (`v6,v7`) | 50/50 | 82% | 98% | 96% | 100% |

`analysis_model_missing=0`, `analysis_prompt_version_missing=0` во всех плечах.

## LLM Calls

- `llm_calls_total`: 201 успешный analyze-вызов.
- Официальная матрица: 200 вызовов (`4 x 50`).
- Live-пилот runner: 1 успешный вызов.
- Mock-smoke: 0 LLM-вызовов.
- Один ранний инфраструктурный запуск упал до полезного analyze из-за `service_tier` в пользовательском Codex config; исправлено флагом `--ignore-user-config`.

## Проверки

- Точечные тесты РП-0: `2 passed`.
- Точечные тесты РП-1/РП-2: `55 passed`.
- Полный pytest перед отчётом поставки 1: `2913 passed, 2 skipped, 1 warning in 43.37s`.
- NEG: `migrate_analysis_payload` и AI Office export не добавляют `analysis_meta` к историческим payload.
- NEG: raw WhatsApp и `product_data/customer_profiles/` игнорируются git.
- NEG: A/B `calls.json` не хранит полный `history_summary` и `next_step_action`; итоговые отчёты не содержат телефонов/email.

## Ограничения

- Решение по модели не принято: по ТЗ это делает Дмитрий после регрейда Claude.
- `prompt_version=v6,v7` у compact-плеч означает штатную эскалацию части звонков с compact на full при нехватке структурных полей.
- Глобальный `git status` не чистый из-за изменений, существовавших до старта TZ-12; мои коммиты добавляют только файлы этого блока.
