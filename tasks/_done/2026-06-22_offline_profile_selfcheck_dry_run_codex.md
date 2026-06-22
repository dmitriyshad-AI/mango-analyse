# Offline dry-run самопроверки public Telegram bot

Дата: 2026-06-22  
Исполнитель: Codex  
Ветка: `codex/tz-profile-selfcheck`  
База до задачи: `3af1a460f7ad5d11586f1e865c277a2e982062b2`

## Цель

Проверить включение `ENFORCE_CANONICAL_PROFILE=1` без Telegram, без токена настоящего бота, без `poll`, без отправки сообщений и без доступа к live-процессу PID `73545`.

## Что изменено

- `scripts/run_telegram_public_pilot_bots.py`
  - добавлен режим `--mode dry-run-startup`;
  - режим выполняет startup-последовательность: env-файл -> sync в `os.environ` -> `ensure_canonical_pilot_profile()` -> самопроверка -> запись heartbeat -> выход;
  - Telegram API не импортируется и не вызывается;
  - `--duration-sec` в этом режиме только удерживает dry-run процесс живым, чтобы checker мог проверить `pid`;
  - heartbeat теперь пишет объединённые active/quality guards, включая `semantic_output_verifier` и `output_sanitizer`.
- `scripts/check_public_bot_live.py`
  - добавлен `--heartbeat-only`;
  - режим читает реальный heartbeat, проверяет freshness/pid/profile/guards и не запускает smoke-диалоги;
  - checker требует в heartbeat также `semantic_output_verifier=true` и `output_sanitizer=true`.
- Тесты:
  - добавлены offline startup positive/negative;
  - добавлен checker heartbeat-only positive/negative;
  - добавлена явная очистка env после тестов, которые вызывают `main()`.

## Offline прогон

Запуск был через `env -i` и временный env-файл в `/tmp`, с fake token `offline-token`. Боевой `.codex/mango_telegram_pilot_bots.env` не использовался.

Команда dry-run: `run_telegram_public_pilot_bots.py --mode dry-run-startup --duration-sec 12`.

Результат:

- dry-run rc: `0`
- stderr: пустой
- Telegram polling: не запускался
- Telegram send: не выполнялся

Heartbeat:

```json
{
  "schema_version": "public_pilot_bot_heartbeat_v2_2026_06_21",
  "status": "polling",
  "event": "dry_run_startup",
  "effective_profile": "pilot_gold_v1",
  "draft_path": "direct_path",
  "brands": ["foton"],
  "active_guards": {
    "presale_safety": true,
    "presale_pii_memory": true,
    "pii_relation_stopwords": true,
    "verifier_handoff_claims": true,
    "semantic_output_verifier": true,
    "output_sanitizer": true,
    "number_gate_scope_aware": true
  },
  "summary_offline": true
}
```

Checker green:

```json
{
  "rc": 0,
  "ok": true,
  "heartbeat_only": true,
  "turns_count": 0,
  "bot_heartbeat_ok": true,
  "failures": []
}
```

## Negative

Профиль `TELEGRAM_DIRECT_PATH_PILOT_CONFIG=on`:

- dry-run rc: `1`
- selfcheck остановил старт;
- stderr содержит `pilot_gold_profile_disabled`.

Required guard off `TELEGRAM_PRESALE_SAFETY=0`:

- dry-run rc: `1`
- selfcheck остановил старт;
- stderr содержит `presale_safety_disabled`.

Checker по сломанному heartbeat:

```json
{
  "rc": 2,
  "ok": false,
  "failures": [
    {"reason": "public_bot_profile_off", "effective_profile": "on"},
    {"reason": "public_bot_guard_off", "guard": "output_sanitizer"}
  ]
}
```

## Тесты

Команды:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m py_compile scripts/run_telegram_public_pilot_bots.py scripts/check_public_bot_live.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_telegram_public_pilot_bots.py tests/test_check_public_bot_live.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q
git diff --check
```

Результат:

- py_compile: PASS
- focused pytest: `69 passed`
- full pytest: `3501 passed, 5 skipped, 1 warning in 76.94s`
- diff check: PASS

Первый полный прогон поймал env-leak в новых checker-тестах; исправлено явной очисткой env после вызова `main()`, затем полный прогон зелёный.

## Safety

- `--mode poll` не запускался.
- Telegram API не вызывался.
- Сообщения не отправлялись.
- Боевой public bot PID `73545` не трогался.
- AMO/Tallanto/stable_runtime/M1 не трогались.
- Merge/push не выполнялись.
- Секреты не выводились.

## ACK

ACK: offline dry-run самопроверки и heartbeat/checker проверены, корректный конфиг зелёный, negative красный, STOP на регрейд Claude #1.
