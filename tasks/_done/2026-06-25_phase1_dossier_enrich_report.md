# Фаза 1: обогащение bot-safe досье из звонков

Дата: 2026-06-25
Ветка: `codex/phase1-dossier-enrich`
База ветки: `4caa5eb`

## Что изменено

- Bot-safe summary стал многострочным досье: шапка `Бренд / Стадия / Интерес / Следующий шаг` + секции `Обсуждали`, `Интерес / возражения`, `Договорённость / следующий шаг`.
- Звонки берутся только из `timeline_events` с `event_type='mango_call'` и `source_system='mango_processed_summary'`.
- Новый `source_system='mango_call'` не создавался и не используется.
- Добавлен скраб имён по всему досье, не только в interest.
- Добавлены фильтры точных деталей: адреса, группы/старт/сроки, реквизиты, назначение платежа, даты/время/цены/проценты/коды.
- Процедурные обещания из памяти не выводятся и не сохраняются в bot-safe `metadata.next_step` как клиентский текст.
- При `ambiguous_identity_open` содержательные секции из звонков не строятся.
- Direct path и runtime context подняты до лимитов 1200/3000, чтобы многострочное досье не резалось сразу.

## Проверки

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_customer_timeline* tests/test_*bot_safe* tests/test_*direct_path*` → `314 passed`.
- `python3 scripts/preflight.py --tz tasks/_running/2026-06-25_TZ_FAZA1_obogashchenie_dosye_zvonki_dlya_Kodeksa.md` → `PREFLIGHT: OK`.
- Sandbox micro-check: `12/12 passed`, модель не вызывалась.

## Audit pack

`audits/_inbox/phase1_dossier_enrich_20260625_205038/`

Содержит: `implementation_notes.md`, `semantic_review.md`, `risk_review.md`, `backward_compatibility.md`, `test_output.txt`, `micro_check.json`, `micro_check.md`, `manifest.json`.

## Маскированный пример

До:

```text
Бренд: Фотон. Стадия: Ожидание оплаты. Интерес: Фотон ОГЭ физика онлайн. Следующий шаг: активный шаг не найден.
```

После:

```text
Бренд: Фотон.
Стадия: Ожидание оплаты.
Интерес: Фотон ОГЭ физика онлайн.
Следующий шаг: семья сравнит онлайн и очный формат.
Обсуждали:
- Обсуждали подготовку к ОГЭ по физике и онлайн-формат.
Интерес / возражения:
- Интерес: ОГЭ физика
- Возражение: нужен мягкий темп
Договорённость / следующий шаг:
- [active] семья сравнит онлайн и очный формат
```

## Границы

- Клиенту сообщений: 0.
- AMO/Tallanto/CRM write: 0.
- Боевую timeline на запись не трогал.
- Секреты не печатались.
- ASR/Analyze не запускались.
- Вердикт «в прод» не выносится: нужен M1 ON-vs-OFF замер и регрейд Claude #1.

## Готовый ON-прогон для M1

После регрейда кода использовать тот же reliable replay набор `mango_clean_reliable_replay_20260625`, включив `TELEGRAM_BOT_SAFE_CRM_CONTEXT=1` и подключив пересобранную bot-safe timeline copy. Нули для гейта: рост `derived_product_claim`, бренд-утечка, P0-регрессия, чужой человек, ПДн.
