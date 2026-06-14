# TZ-109 Package 6 - child identity in dialogue memory

Дата: 2026-06-14

## Что сделано

- Влит в `main` утвержденный LLM-резолвер дублей детей из ветки `codex/tz24-dubli-deti`.
- Добавлен безопасный флаг `TELEGRAM_CHILD_IDENTITY_MODEL`, по умолчанию выключен.
- В `build_dialogue_memory()` добавлен read-only вход `resolved_children`.
- В `build_telegram_pilot_context()` добавлен проброс `resolved_children` в память диалога.
- При включенном флаге и готовом `current_child_key` один текущий детский слот использует модельно разрешенный ключ, а не жёсткое правило `дочка/младший -> child_2`.
- При двух детях в одном сообщении старое разделение `child_1`/`child_2` сохранено.

## Чего не делали

- Не вызывали `child_resolver_llm.py` из живого Telegram-пути.
- Не запускали ASR, Resolve+Analyze, live AMO/Tallanto/CRM.
- Не меняли runtime-артефакты.
- Не включали флаг по умолчанию.
