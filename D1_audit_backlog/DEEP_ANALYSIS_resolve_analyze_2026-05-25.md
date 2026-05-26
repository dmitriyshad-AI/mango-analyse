# Глубокий разбор этапов RESOLVE и ANALYZE (read-only). 2026-05-25

Claude, read-only код-ревью (сервисы НЕ запускались — правило проекта). Подробные логи:
`logs/deep_resolve.md`, `logs/deep_analyze.md`. Критичное перепроверено Claude лично.

## Конвейер (контекст)
ingest (назначает телефон, ingest.py:162) → transcribe (ASR) → **resolve** → **analyze** → sync (в AMO, под гейтом).
Привязка к клиенту/сделке и запись в CRM — НЕ в resolve/analyze, а в sync_amocrm / amocrm_runtime / deal_aware.

## RESOLVE (resolve.py, 2062 стр.) — НЕ то, что по названию
Главное: это НЕ «резолв личности/сделки в CRM». Проверено Claude: 0 упоминаний amocrm/tallanto/crm/lead/deal.
Что делает: сливает и чистит ДВА ASR-варианта транскрипта (A и B) одного звонка в единый диалог — выбирает
лучший из baseline / LLM-слитый / rescue-ASR по score (resolve.py:1759–1838). Пишет ТОЛЬКО в локальную БД +
локальные CSV. LLM (codex CLI) запускается `--sandbox read-only` (resolve.py:827, 1080) — записи во вне нет.
Риски:
- P2 путаница ролей manager↔client: метки тянутся из ASR, модель может их менять (resolve.py:1187–1200) →
  в карточку может попасть «менеджер сказал» вместо «клиент» и наоборот.
- P2 тихий fallback LLM→правило при сбое модели (resolve.py:902–911) — маскирует деградацию качества.
- P2 lease-гонка без heartbeat (pipeline_claims.py:59–62) — при сбое/нескольких воркерах возможна двойная обработка.
- P2 нет явного таймаута к openai/ollama (у codex CLI таймаут есть).
- P3 имя стадии вводит в заблуждение — стоит переименовать (transcript_merge/reconcile).

## ANALYZE (analyze.py, 2340 стр.) — LLM-карточка звонка
Что делает: transcript → через LLM формирует `analysis_json` (history_summary + structured_fields:
люди/контакты/ученик/интересы/коммерция/возражения/next_step/приоритет, теги, follow_up_score). После успеха
ставит sync_status=pending (analyze.py:2308). Выбор записей — claim/lease/batch с worker_id, обработка по одной с commit.
Риски:
- **P1 (внутренний) — НЕТ бренд-контроля в выводе.** Проверено: 0 упоминаний brand/active_brand/Фотон/УНПК.
  Карточка собирается из полей модели; смешение Фотон/УНПК не флагается. По CLAUDE.md CRM-тексты тоже требуют
  бренд-разделения + semantic_pass. (Карточка manager-facing, не клиент — потому P1-внутренний, не клиентский P0.)
- **P2 — OpenAI-вызов без `max_tokens`, `timeout`, `seed`** (analyze.py:2078), temperature=0.1 (не 0) →
  риск зависания воркера, неконтролируемой длины/стоимости, недетерминизма.
- **P2 — PII в `analysis_json`** (email/телефон/ФИО, analyze.py:1697–1703) + fallback summary = первые 600 символов
  СЫРОГО транскрипта (analyze.py:1651). Карточка для менеджера PII содержать вправе; РИСК — если эту карточку
  переиспользуют в клиентском/боте слое без чистки (insights/sanitizers здесь НЕ зовётся — только локальные guard).
- P2 claim-гонка на не-SQLite/нескольких воркерах: `UPDATE…WHERE id IN (SELECT…LIMIT)` без FOR UPDATE
  (analyze.py:283). На текущем SQLite + 1 воркер безопасно.
- P3 ошибка экспорта файлов до commit может откатить корректный анализ в failed (analyze.py:2309).
Хорошее: парсинг кривого JSON устойчив (json→ast→fence→срез {…}, _coerce_score клампит 0..100); ошибки НЕ
глотаются тихо (last_error + failed/dead с backoff); есть защита от дампа диалога (_looks_like_dialogue_dump) и
hard-валидация non_conversation.

## Главные выводы (для решения)
1. «resolve» — это слияние ASR, а не привязка клиента. Где реально происходит привязка к клиенту/сделке и запись в
   CRM — отдельный аудит sync_amocrm/amocrm_runtime/deal_aware (там и должны жить гейты записи и бренд-контроль).
2. Безопасность по записи: resolve и analyze в CRM НЕ пишут (resolve — read-only sandbox; analyze — только локальная БД).
3. Топ-риски к правке: analyze OpenAI без timeout/max_tokens (операционно); нет бренд-контроля в карточке analyze;
   путаница ролей и тихие fallback в resolve (качество данных).
4. Что проверить у Дмитрия: допустима ли смена роли моделью в проде; нужен ли бренд-флаг в analysis_json;
   переименовать ли стадию resolve.
