# Wappi autoresolver ops guard

Дата: 2026-06-26
Ветка: `codex/wappi-autoresolver-ops-guard`
Base: `main e55262b`

## Что сделано

- Из старой ветки `codex/tz20-autoresolver` перенесены только эксплуатационные ограничения, без старой логики матчинга.
- Добавлен `DRAFT_LOOP_AUTO_RESOLVER_CHANNELS` и CLI `--auto-resolver-channels`.
  - По умолчанию: `telegram,max`.
  - Можно сузить rollout до `telegram`, чтобы Max временно не резолвился автоматически.
- Добавлен управляемый throttle для всех AMO GET внутри `AmoAutoResolver`.
  - Env: `DRAFT_LOOP_AUTO_RESOLVER_THROTTLE_SEC`.
  - CLI: `--auto-resolver-throttle-sec`.
  - Default: `0.2`.
- Добавлен один retry на AMO 429.
  - Env: `DRAFT_LOOP_AUTO_RESOLVER_RETRY_AFTER_SEC`.
  - CLI: `--auto-resolver-retry-after-sec`.
  - Default: `3.0`.
- Новый AMO-event resolver сохранён как главный источник истины.

## Что сознательно не перенесено

- Старый primary matcher по Telegram ID / Max phone не перенесён как главный путь: в `main` безопаснее AMO-event matcher через `incoming_chat_message`, `talk_id` и sequence-confirmation.
- File-cache `profile_id:chat_id -> lead_id` не перенесён в live-путь: после успешного матча auto-pair уже сохраняется, а дополнительный cache мог бы стать вторым источником истины. Его можно отдельно оценить позже как read-only/performance cache.
- `auto_resolver_started_at` / same-turn draft не перенесён: текущее поведение `main` безопаснее, потому что создаёт auto-pair и пишет черновики только на будущие входящие.

## Проверки

- Targeted:
  - `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_run_amo_wappi_draft_loop.py tests/test_draft_loop.py`
  - `52 passed`
- Full:
  - `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q`
  - `3661 passed, 5 skipped, 1 warning`
- `git diff --check` чистый.

## Safety

- AMO/Tallanto/CRM write не запускались.
- Клиентам ничего не отправлялось.
- Live Wappi loop не запускался и не перезапускался.
- `DRAFT_LOOP_AUTO_RESOLVER` остаётся default OFF.
- AMO `/events` по-прежнему только read-only, запись заметок остаётся отдельно за `--live-write`.

## Остаточный риск

- Это `formal_pass`: смысл клиентских черновиков не менялся и отдельно не оценивался.
- Перед удалением `Mango_tz20_autoresolver` стоит сохранить тег на `4edb83a`; после этой правки полезные safety-части уже перенесены, но ветка остаётся исторической страховкой.
