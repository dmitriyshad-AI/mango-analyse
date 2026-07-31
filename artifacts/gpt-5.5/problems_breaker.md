# ТЗ 200 — breaker report

Дата проверки: 2026-07-31
Worktree: `/Users/dmitrijfabarisov/Projects/Mango_regex_map_d1_20260731`
HEAD: `ca1c9ce534b9f64b8d0c775df5753694cfbb101f`
Режим: read-only по `src/**`; правки только в этом файле.

## Бизнес-обещание

Карта regex/marker-долга должна отделить клиентское "понимание", которое можно
передавать модели, от детерминированных полов безопасности. Ошибка разметки в
сторону "понимания" для платежей, возвратов, P0 или бренда может разрешить боту
подтверждать деньги без двух источников либо давать опасный ответ там, где нужен
менеджер.

## Пересчёт первоисточников

Вызвал существующие функции из `tests/test_adr003_regex_understanding_moratorium.py`,
а не отдельный обходчик:

- `_regex_snapshot(repo)`: `197`; фикстура: `197`; `actual == fixture`: `True`.
- Живой regex-периметр без `telegram_pilot_reporting.py`: `190`.
- `telegram_pilot_reporting.py`: `7`; поэтому `190 + 7 = 197`.
- `_direct_path_text_pattern_snapshot(repo)`: `832`; фикстура: `832`; `actual == fixture`: `True`.
- Разбивка `832`: `regex_call=381`, `string_contains=167`, `text_table=154`, `marker_helper_call=130`.
- Внутри `regex_call=381`: `re.compile=166`, inline `re.*=215`.
- `_channel_marker_helper_call_counts(repo)`: фактических вызовов `172`.
- `CHANNEL_MARKER_HELPER_BUDGET`: бюджетный потолок `255`.

Проверка самого моратория:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python3 -m pytest -q -p no:cacheprovider tests/test_adr003_regex_understanding_moratorium.py

7 passed in 14.25s
```

## Проблемы

### P1 — measurement_bug — `255` является бюджетом, а не фактическим числом marker-helper calls

Факт: `tests/test_adr003_regex_understanding_moratorium.py:84-95` задаёт
`CHANNEL_MARKER_HELPER_BUDGET`, а `:607-620` проверяет только `count <= budget`.
Текущий фактический счётчик `_channel_marker_helper_call_counts()` дал `172`, не
`255`. В `832`-снапшоте marker-helper строк ещё меньше: `130`, потому что
`_direct_path_text_pattern_snapshot()` пишет только вызовы с литеральными
marker-аргументами.

Чем грозит бизнесу: карта может завысить "понимательный" долг на `83` записи
и дать ложный прогресс или ложный объём работ. Хуже: budget можно трактовать
как список живых правил и размечать несуществующие записи, пока реальные
строки остаются без владельца.

Минимальный фикс: в карте и приёмке разделить `actual_marker_helper_calls=172`,
`literal_marker_helper_rows=130` и `budget_ceiling=255`. Для полноты карты
использовать actual rows/actual calls, а бюджет оставить только stop-гейтом.

### P1 — measurement_bug — inline `re.*` зависит от периметра; число `233` не является выводом текущего генератора

По тем же AST-правилам из теста:

- `DIRECT_PATH_PATTERN_FILES`: inline `re.* = 215`.
- весь `src/mango_mvp/channels/**/*.py`: inline `re.* = 325`.
- весь `src/**/*.py`: inline `re.* = 770`.

Число `233` не воспроизводится из текущего `_direct_path_text_pattern_snapshot()`.
Если его использовать как обязательную сумму карты, появятся "фантомные" строки
или пропадут реальные строки из другого периметра.

Чем грозит бизнесу: параллельные разметчики будут сверять разные знаменатели.
Итоговая карта может выглядеть полной, но не иметь `0 unmarked` относительно
реального генератора.

Минимальный фикс: ввести один машинный периметр для приёмки. Если source of
truth — `adr003_direct_path_text_patterns_snapshot.json`, то inline-база сейчас
`215`, а не `233`. Если нужно `233`, нужно явно перечислить другой список файлов
и добавить его в генератор как отдельный режим.

### P2 — measurement_bug — regex budget допускает slack и не доказывает точное снятие

`_regex_snapshot()` сходится с фикстурой на `197`, а живой периметр сходится на
`190`. Но сумма `CHANNEL_REGEX_BUDGET` сейчас `200`, а без reporting — `193`.
Тест `:587-590` проверяет `count <= budget`, не точное равенство.

Чем грозит бизнесу: удаление или перенос до трёх regex может не отразиться в
budget-гейте. Для шага 5 ТЗ это опасно: "бюджет строго уменьшился" нельзя
доказывать только текущим `<=`.

Минимальный фикс: для снятия класса требовать diff точной фикстуры и отдельную
проверку, что соответствующий budget уменьшен ровно на снятое число.

## Защитные regex-ловушки

### `PAYMENT_CONFIRMATION_RE` — не bucket 1, а bucket 2 / пол денег

Источник:

- `src/mango_mvp/channels/subscription_llm_parts/policy_routing.py:557` —
  сам regex.
- `src/mango_mvp/channels/subscription_llm_parts/policy_routing.py:1587` —
  `apply_payment_confirmation_guard()`.
- `src/mango_mvp/channels/subscription_llm_parts/policy_routing.py:1595-1607` —
  требует согласованные AMO и Tallanto, иначе заменяет черновик на manager-only.
- `src/mango_mvp/channels/subscription_llm_parts/policy_routing.py:3685-3692` —
  ставит `payment_confirmation_guarded`, checklist и безопасный fallback.

Воспроизведение существующими тестами:

```text
tests/test_bot_policy_v2.py::test_payment_confirmation_requires_matching_amo_tallanto
tests/test_bot_policy_v2.py::test_payment_conflict_forces_manager_only
tests/test_wappi_stabilization_smoke.py::test_payment_status_is_not_autoconfirmed_without_two_sources

в составе точечного набора: 7 passed in 0.52s
```

Минимальный импорт-вызов без сети/LLM:

- без источников: `route=manager_only`, подтверждение оплаты удалено,
  флаг `payment_confirmation_without_two_sources`.
- только AMO: `route=manager_only`, подтверждение оплаты удалено.
- AMO + Tallanto = paid/paid: `route=draft_for_manager`, подтверждение оплаты
  сохранено.
- конфликт AMO/Tallanto: `route=manager_only`, флаг `payment_source_conflict`.

Классификация: object_bug не подтверждён; это защитный объект. Риск — будущий
measurement/object bug, если карту разметят как "понимание" и отдадут модели.

### `_ROUTE_REFUND_RE` — не bucket 1, а bucket 2 / P0-деньги

Источник:

- `src/mango_mvp/channels/subscription_llm_parts/text_hygiene.py:76` —
  сам regex.
- `src/mango_mvp/channels/subscription_llm_parts/text_hygiene.py:272` —
  manager-only + refund/payment client message удерживаются как P0 hygiene.
- `src/mango_mvp/channels/subscription_llm_parts/text_hygiene.py:375-380` —
  legacy kind становится `refund`, кроме включённого payment-fix исключения.

Воспроизведение существующими тестами:

```text
tests/test_direct_p0_text_hygiene.py::test_direct_p0_text_hygiene_provider_level_scrubs_refund_sales_tail
tests/test_direct_p0_text_hygiene.py::test_direct_p0_text_hygiene_final_hook_scrubs_post_gate_refund_manager_only
tests/test_direct_p0_text_hygiene.py::test_direct_p0_text_hygiene_keeps_benign_presale_refund_without_high_risk_route
tests/test_direct_p0_text_hygiene.py::test_text_hygiene_payment_fix_keeps_real_refund_as_refund

в составе точечного набора: 7 passed in 0.52s
```

Минимальный импорт-вызов без сети/LLM:

- реальный возврат: `route=manager_only`, флаг `direct_p0_text_hygiene`,
  `kind=refund`, продающий хвост удалён.
- предпродажный вопрос "до оплаты": объект не меняется, route остаётся
  `bot_answer_self_for_pilot`.

Классификация: object_bug не подтверждён; regex ведёт себя как safety floor.
Риск — будущий object_bug, если при переносе "понимания" он будет удалён вместе
с обычными интентными маркерами.

## Итоговая классификация

- `object_bug`: не подтверждён в проверенных защитных путях.
- `measurement_bug`: подтверждён. Минимум три проблемы: `255` как budget вместо
  actual, невоспроизводимое inline `233`, slack между regex budget и snapshot.
- `infrastructure_bug`: не подтверждён. Тесты и импорт-вызовы проходят локально,
  без сети/LLM и без записи в `src/**`.

## Вердикты

- `formal_pass`: да, мораторий и точечные safety tests зелёные.
- `semantic_pass`: частично. Защитные payment/refund regex семантически должны
  остаться в bucket 2, но численные критерии карты в ТЗ надо уточнить.
- `business_pass`: нет для текущей формулы приёмки `190 + 233 + 255 + 75`.
  Она смешивает actual rows, budget ceilings и разные периметры.
- `data_pass`: да для проверенных первоисточников `197/832/190`; нет для
  `233/255` как фактических чисел.
- `runtime_pass`: не заявляю. Live-путь, сеть, LLM, AMO/Tallanto/CRM не
  запускались и не должны были запускаться в этой проверке.
