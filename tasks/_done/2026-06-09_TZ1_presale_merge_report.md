# TZ1 presale merge report — 2026-06-09

## Итог

Выполнено на `main`.

- Presale-пакет П1-П4 влит fast-forward из `codex/presale-safety-fixes`.
- Дополнительно закрыт блокер склонённых одиночных имён из TZ1.
- Симулятор/прогоны качества не запускались.

## Коммиты

- `bf558405` — Add presale safety guards.
- `8937c1c7` — Add presale source id output sanitizer.
- `1609869b` — Cover inflected presale PII echoes.

## Read-only сверка

До мержа проверено:

- Ветка `codex/presale-safety-fixes` отличается от `main` только:
  - `src/mango_mvp/channels/subscription_llm.py`;
  - `tests/test_subscription_llm_draft_provider.py`.
- Пакет содержит флаги:
  - `TELEGRAM_PRESALE_SAFETY`;
  - `TELEGRAM_PRESALE_PII_MEMORY`;
  - `TELEGRAM_PRESALE_VERIFIER_FAILSOFT`;
  - `TELEGRAM_PRESALE_META_RU`;
  - `TELEGRAM_PRESALE_SOURCE_ID`.
- Флаги не включались глобально в env; включение профилем пилота оставлено для TZ2.

## NEG / проверки

Targeted:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_subscription_llm_draft_provider.py -k 'presale or pii_echo or source_id or verifier_failsoft'
20 passed, 381 deselected
```

Соседняя регрессия после добавления нового NEG была найдена и исправлена:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_subscription_llm_draft_provider.py -k 'presale or pii_echo or source_id or verifier_failsoft or live_availability_data_needed or invoice_monthly_payment_method or foton_offline_free_trial or weekend_schedule_question'
24 passed, 377 deselected
```

Полный pytest:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests
2834 passed, 2 skipped, 1 warning
```

## Склонённые имена

Закрыто.

Добавлен NEG:

- память содержит `client_name=Ирина`, `child_name=Артём`;
- черновик содержит `Передайте Ирине: для Артёма есть группа.`;
- результат не содержит `Ирине` и `Артёма`;
- reason: `client_name_echo`.

Побочный эффект первого варианта фикса пойман полным pytest: self-marker начал воспринимать обычное слово после «я» как имя. Исправлено: дополнительные single-name candidates из self-marker берутся только из слов, которые в исходном тексте начинаются с заглавной буквы.

## Остаточный риск

PII de-echo остаётся эвристическим. Он закрывает подтверждённые кейсы `Петру`, `Артёму`, `Ирине`, телефон и имена из recent window / memory slots, но не является полноценным морфологическим NER.
