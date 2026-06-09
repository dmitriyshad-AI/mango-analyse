# TZ-7 route rubric report

Дата: 2026-06-10

Кодовый коммит: `411984f6` (`Add direct path route rubric`)

## Что сделано

- Добавлен флаг `TELEGRAM_ROUTE_RUBRIC`, default OFF.
- Флаг не добавлен в профиль `pilot_gold_v1`.
- В прямой путь добавлена рубрика выбора маршрута в prompt только при включенном флаге.
- Строка миссии про «менеджер свяжется» под тем же флагом скоупится только на черновик для менеджера.
- Добавлена одна перегенерация строго после успешного direct draft-вызова и до semantic verifier/gate, только для `draft_for_manager` без `missing_facts` при наличии фактов.
- Перегенерация не меняет `direct_path_regenerated`; отдельные поля: `rubric_enabled`, `rubric_regenerated`, `rubric_reason`.
- Добавлен annotate-only индикатор `deferral_text_in_self`.
- В summary раннера добавлен блок `direct_path_rubric`.

## NEG / контроль

- `pilot_gold_v1` не включает `TELEGRAM_ROUTE_RUBRIC`.
- Golden prompt OFF совпадает с прежней структурой, ON добавляет рубрику и скоупит строку миссии.
- Матрица регенерации: self-route, `missing_facts`, отсутствие фактов, OFF и preblock не вызывают повтор.
- Ошибка второго вызова оставляет первый draft и пишет `regen_failed:*`.
- Перегенерированный self-текст всё равно проходит authoritative gate; неподтверждённое продуктовое число блокируется.
- Анти-Волна-5: рубрика не повышает route кодом при OFF/без успешной перегенерации.

## Проверки

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_subscription_llm_draft_provider.py tests/test_telegram_dynamic_client_sim.py`
  - `517 passed`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests`
  - `2893 passed, 2 skipped, 1 warning`

## Кандидаты для будущего смоука

- Тревожный родитель: модель должна не уходить в draft без `missing_facts`, если факты покрывают вопрос.
- «За два предмета выйдет X»: проверка, что route-рубрика не разблокирует продуктовые числа без факта.
- Вопрос про порядок записи/оформления: рассказать процесс по факту, но действие записи оставить draft.
- Вопрос с реальным отсутствующим процессным фактом: `draft_for_manager` должен содержать конкретные `missing_facts`.

Симулятор и М1-прогоны не запускались: по ТЗ замер позже отдельной задачей.
