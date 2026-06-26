# D1 P0 Three Classes On Main

Дата: 2026-06-26

Ветка: `codex/p0-three-classes-on-main`
Base: `main@b543991`

## Что сделано

Пересобран P0-fix как чистая ветка поверх актуального локального `main`.

Перенесены только P0-коммиты:

- `fbcf9dd` (`01d049d`) — добавляет recall для трёх классов P0:
  - снятие/отмена записи оплаченной смены -> `refund`;
  - перенос-возврат оплаченной смены -> `refund`;
  - претензия по договору/дате/ФИО/паспорту -> `legal`.
- `f461117` (`0876fad`) — сужает service-exit detector, чтобы benign фразы вроде "снять стресс/усталость ребёнку" не уходили в `refund`.

## Scope Check

Дифф к `main` содержит только:

- `src/mango_mvp/channels/p0_recall_spec.py`
- `tests/test_answer_safety_classifier.py`
- `tests/test_p0_perifraz.py`
- `tasks/_done/2026-06-26_p0_detector_three_classes_report.md`
- этот отчёт

Venue/autonomy слой, `fact_venue_scope.py`, большие KB-изменения и Wappi-код в эту ветку не попали.

## Проверки

Целевые P0-тесты:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q \
  tests/test_p0_perifraz.py \
  tests/test_answer_safety_classifier.py
```

Результат: `135 passed in 0.90s`

Полный pytest:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q
```

Результат: `3631 passed, 5 skipped, 1 warning in 78.35s`

`git diff --check`: clean.

## Safety

- Live Telegram bot не трогался.
- AMO/Tallanto/CRM write: 0.
- Клиентам ничего не отправлялось.
- `stable_runtime` не трогался.

## Остаточные риски

Это `formal_pass`. Для клиентского safety-слоя нужен регрейд по сырью: целевые P0 должны уходить к менеджеру, benign "снять стресс/усталость" и похожие неплатёжные фразы не должны становиться P0.

## Вывод

Ветка готова к регрейду как чистый P0-кандидат поверх актуального `main`.
