# D1 CRM card source-label cleanup

Дата: 2026-06-26

## Контекст

Последний AMO card dry-run на 5 `ready_yes` карточках был заблокирован quality-gate:
в `Авто история общения` / `AI-история по сделке` попадала служебная метка
`mango_processed_summary`.

## Что изменено

- В `crm_card_aggregator` добавлен человекочитаемый label для источников истории:
  `mango_processed_summary` / `mango` / `mango_office` -> `Звонок`,
  `amocrm_snapshot` -> `AMO`, `mail_archive*` -> `Письмо`,
  `channel_snapshot` / messenger history -> `Сообщение`,
  `tallanto_snapshot` -> `Tallanto`.
- `_history_summary_source` и `_chronology_text` больше не выводят сырой
  `source_system` в manager-only историю.
- Добавлен regression-тест: fallback-история из D4-style `mango_call.summary`
  сохраняет полезный текст, но не пропускает `mango_processed_summary`
  в contact/deal payload и preview.

## Проверки

Целевые тесты:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_crm_card_aggregator.py tests/test_crm_text_quality_detector.py
# 62 passed in 0.67s
```

Полный pytest:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q
# 3382 passed, 5 skipped, 1 warning in 52.55s
```

Preview read-only на боевой Customer Timeline:

```text
product_data/customer_timeline/crm_card_preview_20260626_source_label_cleanup_prod_ro/crm_cards_preview.xlsx
```

Итоги preview:

- rows: 200
- ready_yes: 60
- ready_no: 140
- AMO/Tallanto/CRM write: 0
- Customer Timeline write: 0
- `mango_processed_summary` в текстовых payload/preview колонках: 0
- `mango_call` в текстовых payload/preview колонках: 0
- `source_system` в текстовых payload/preview колонках: 0

## Semantic review

Verdict: `PASS_WITH_NOTES`.

Что прошло:

- Manager-visible history больше не показывает машинную метку источника.
- Полезная fallback-сводка звонка сохраняется.
- Quality-gate на служебные/тестовые маркеры не ослаблен.
- Live-write не выполнялся.

Notes:

- Это закрывает только payload-noise blocker из последнего dry-run.
- Следующий AMO dry-run на 5 карточках нужно повторить отдельно через
  `foton-crm-readonly` / existing writeback dry-run: identity-gate,
  field allowlist, clobber-protection и payload-quality должны пройти заново.
- Этот отчёт не даёт разрешения на AMO live-write.
