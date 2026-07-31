# Review notes v2 GPT-5.5 для ТЗ №200 D1

Дата проверки: 2026-07-31
Worktree: `/Users/dmitrijfabarisov/Projects/Mango_regex_map_d1_20260731`
HEAD worktree: `9f3f859b`
База сравнения: `main@ca1c9ce534b9f64b8d0c775df5753694cfbb101f`
Режим: без LLM/live, без записи в `src/**`, ветку и коммиты не трогал.

## Вердикт

`PASS_WITH_NOTES`

Жёсткие стоп-условия D1 v2 на последней проверенной версии не сработали:

- snapshot: `832` строки;
- `row_id`: `832/832` уникальны и текущим генератором воспроизводятся;
- `node_kind`: `marker_helper_call=130`, `regex_call=381`, `string_contains=167`, `text_table=154`;
- YAML: только bucket `2_verification` и `3_format_hygiene`, неизвестных `row_id` нет;
- bucket2: `313`, выше порога `110`;
- bucket3: `28`, только разрешённые format/hygiene classes;
- `255` используется как `marker_helper_budget_ceiling`, не как знаменатель;
- `src/**` diff против `main@ca1c9ce5` пустой;
- бюджеты `CHANNEL_REGEX_BUDGET`, `CHANNEL_MARKER_HELPER_BUDGET`, `ACTIVE_BEHAVIOR_ALLOWED_FALSE_BUDGET` не выросли.

Не даю чистый `PASS` из-за двух схемных проблем YAML и одного workflow-риска.

## Проверки

1. Graphify использован только read-only: локальная карта показала ревизию `ca1c9ce534b9`; все выводы ниже подтверждены сырьём в указанном worktree.
2. `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_adr003_regex_understanding_moratorium.py`
   - финальный результат: `11 passed in 13.83s`.
3. Payment/refund reproduction:
   - `tests/test_bot_policy_v2.py::test_payment_confirmation_requires_matching_amo_tallanto`
   - `tests/test_bot_policy_v2.py::test_payment_conflict_forces_manager_only`
   - `tests/test_wappi_stabilization_smoke.py::test_payment_status_is_not_autoconfirmed_without_two_sources`
   - `tests/test_direct_p0_text_hygiene.py::test_direct_p0_text_hygiene_provider_level_scrubs_refund_sales_tail`
   - `tests/test_direct_p0_text_hygiene.py::test_direct_p0_text_hygiene_final_hook_scrubs_post_gate_refund_manager_only`
   - `tests/test_direct_p0_text_hygiene.py::test_direct_p0_text_hygiene_keeps_benign_presale_refund_without_high_risk_route`
   - `tests/test_direct_p0_text_hygiene.py::test_text_hygiene_payment_fix_keeps_real_refund_as_refund`
   - `tests/test_direct_path_payment_refund_split.py::test_real_payment_dispute_stays_manager_only_and_not_payment_link_text`
   - результат: `8 passed in 0.48s`.
4. `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_semantic_roles.py`
   - результат: `52 passed in 0.40s`.
5. `git diff --check -- tests/test_adr003_regex_understanding_moratorium.py tests/fixtures/adr003_direct_path_text_patterns_snapshot.json`
   - результат: PASS.
6. Полный `git diff --check ca1c9ce5` падает на старом tracked-артефакте `artifacts/gpt-5.5/problems_breaker.md:182: new blank line at EOF`. Это фон ветки от предыдущего STOP, не текущие D1 v2 файлы.

Во время ревью worktree менялся параллельно: первый запуск мораторий-теста дал `2 failed / 9 passed` на рассинхроне `row_id`, затем текущие файлы были обновлены и повторные проверки стали зелёными. Финальный вердикт относится к последней проверенной версии.

## Фактическая карта

`tests/fixtures/adr003_direct_path_text_patterns_snapshot.json`

- текущий `sha256`: `c6a94c894a23d4a73a36d070bdec96ceb178efafd61473e50bfbdc062af10cb2`;
- base и current оба имеют `832` строки;
- после удаления новых полей `row_id`, `lineno`, `col_offset`, `literal_string_count`, `cyrillic_string_count` multiset строк равен base snapshot;
- порядок части строк изменился, но содержимое base-набора не выросло и не исчезло.

`docs/adr003_understanding_map.yaml`

- `source.snapshot_sha256` совпадает с текущей фикстурой;
- `classification_counts`: `2_verification=313`, `3_format_hygiene=28`, `unassigned_for_parallel_owner=491`;
- `classes`: 16, включая новый `refund_semantic_safety_floor`;
- YAML файл сейчас untracked, поэтому его нужно явно добавить в будущий коммит D1.

## Findings

### P1 - YAML `symbol` не является тем же полем, что `symbol` в canonical snapshot

Проверка дала `129` несовпадений `map_row.symbol != snapshot_row.symbol`.

Пример:

- YAML: `docs/adr003_understanding_map.yaml`, row `adr003:e45a4d51827aadbbe07d65b3`, `symbol: PAYMENT_CONFIRMATION_RE`.
- Snapshot: та же строка имеет `symbol: re.compile`, `target: PAYMENT_CONFIRMATION_RE`.

То же видно для `_ROUTE_REFUND_RE`: YAML кладёт `_ROUTE_REFUND_RE`, snapshot хранит `symbol: re.compile`, `target: _ROUTE_REFUND_RE`.

Почему это важно: сейчас `symbol` в YAML стал human/display name, а не canonical field. Тест сверяет `path`, `lineno`, `col_offset`, `node_kind`, но не сверяет `symbol`, поэтому схема допускает тихое расхождение.

Точная правка:

- либо в YAML хранить `symbol` ровно из snapshot и добавить отдельное поле `target`/`display_symbol`;
- либо переименовать YAML-поле в `display_symbol`, а тестом явно проверять соответствие `display_symbol == source_row.get("target", source_row["symbol"])`.

Минимальный тест: в цикле `for row_id, map_row in mapped.items()` добавить проверку выбранного контракта для `symbol`/`target`.

### P1 - YAML строки не несут content hash

Во всех `341` YAML rows нет ни одного из canonical content hash полей:

- `pattern_sha256`;
- `args_sha256`;
- `value_sha256`;
- `expression_sha256`.

Сейчас восстановление возможно только через `row_id` и snapshot lookup. Для ревью это работает, но требование D1 про существующий content hash в строке карты не выполнено в самом YAML-артефакте.

Точная правка:

- добавить в каждую YAML row два поля: `content_hash_kind` и `content_sha256`;
- значение брать из соответствующего source row: `pattern_sha256` для regex, `args_sha256` для marker helper, `value_sha256` для text_table, `expression_sha256` для string_contains;
- тестом проверить, что `map_row["content_sha256"] == source_row[map_row["content_hash_kind"]]`.

### P2 - `text_table` является машинным правилом, а не смысловой таблицей

Текущее правило задокументировано и воспроизводится: uppercase assignment, имя содержит один из `TEXT_TABLE_NAME_PARTS`. Counts верные: `154` rows, `1348` string literals, `767` Cyrillic-containing literals, `29` tables with `>=8` Cyrillic literals.

Но правило намеренно широкое и ловит не только таблицы фраз:

- `INTERNAL_SERVICE_MARKER_RE` как `text_table`, хотя значение это `re.compile(...)`;
- `DEAL_ACTION_DECISION_SCHEMA_VERSION` как `text_table`, хотя это строковая версия схемы;
- `_SEMANTIC_FRAME_MANAGER_ACTION_GATE_CONFIDENCE` как `text_table`, хотя это число `0.8`.

Это не STOP для D1, потому что source явно называет rule машинным. Но в отчётах нельзя называть `154` "смысловыми таблицами".

Точная правка:

- в YAML переименовать описание в `machine_text_table_rule`;
- добавить в snapshot/YAML `value_kind` для `text_table`: `regex_call`, `collection`, `string`, `number`, `other`;
- в human report писать `machine text_table rows`, а не `semantic tables`.

### P2 - `row_id` устойчив на frozen snapshot, но не является вечным физическим якорем для дублей

Проверка нашла `15` duplicate identity groups, `30` rows, если исключить `lineno` и `col_offset`. Примеры: два `":" in text` в `dialogue_memory.py`, пары `"онлайн" in text`, `"очно" in text`, одинаковые `_has_any_marker("пробн", "попроб")`.

Текущий генератор решает это occurrence-index после сортировки и даёт `832` уникальных id. Это детерминированно для frozen snapshot и проходит тест. Но если в будущем одинаковые выражения будут переставлены, `row_id` может поменяться местами между физическими строками.

Точная правка: не менять текущий D1, но сохранить обязательную сверку `lineno`/`col_offset`; для будущего генератора добавить `identity_duplicate_index` или включить координаты в отдельный `physical_anchor`, не в stable semantic id.

## Payment/refund proof

### `PAYMENT_CONFIRMATION_RE`

Canonical row:

- `row_id`: `adr003:e45a4d51827aadbbe07d65b3`;
- source: `src/mango_mvp/channels/subscription_llm_parts/policy_routing.py:557`;
- snapshot `target`: `PAYMENT_CONFIRMATION_RE`;
- YAML bucket/class: `2_verification` / `brand_money_promise_floor`.

Source behavior:

- `_draft_confirms_payment()` checks `PAYMENT_CONFIRMATION_RE` at `policy_routing.py:3639-3642`;
- `apply_payment_confirmation_guard()` at `policy_routing.py:1587-1607` requires both AMO and Tallanto paid;
- conflict or missing second source returns manager-only fallback via `_payment_guarded_result()` at `policy_routing.py:3685-3692`.

Reproduction: payment tests above passed. Классификация bucket2 корректна: это money verification floor, не обычное intent-understanding.

### `_ROUTE_REFUND_RE`

Canonical row:

- `row_id`: `adr003:51c3f3de13a38d815c973dfc`;
- source: `src/mango_mvp/channels/subscription_llm_parts/text_hygiene.py:76`;
- snapshot `target`: `_ROUTE_REFUND_RE`;
- YAML bucket/class: `2_verification` / `p0_output_hygiene_floor`.

Source behavior:

- manager-only + refund/payment client message activates P0 hygiene at `text_hygiene.py:272`;
- legacy kind becomes `refund` at `text_hygiene.py:375-380`, except explicit payment-fix branches;
- benign presale refund question is preserved by `_is_benign_presale_refund_question()` at `text_hygiene.py:283-292`.

Reproduction: refund/P0 tests above passed. Классификация bucket2 корректна: это P0/payment hygiene floor, не переносимый "понимательный" regex.

## Verification vs understanding

`refund_semantic_safety_floor` спорный, но допустимый в D1:

- строки в `semantic_roles.py:137`, `semantic_roles.py:191`, `semantic_roles.py:558-569` реально классифицируют смысл `refund_dispute` vs `refund_presale`;
- `tag_message_roles()` добавляет `refund_dispute`/`refund_presale` на `semantic_roles.py:318-321`;
- прямое воспроизведение: "Верните деньги за курс, уже оплатили." даёт `refund_frame=dispute`; "До оплаты хочу понять условия возврата." даёт `refund_frame=presale_policy`; "Это не про возврат, хочу расписание." даёт `refund_frame=none`;
- `tests/test_semantic_roles.py` зелёный: `52 passed`.

Риск: если этот class читать как обычное "понимание", его можно ошибочно отдать bucket1/4. В текущей формулировке `refund_semantic_safety_floor` он должен оставаться bucket2 только как safety boundary для P0/refund split.

Точная правка: в `classes.refund_semantic_safety_floor.meaning` прямо написать, что это "semantic classifier used as a safety boundary for refund/P0 split", и добавить в map/test отдельную проверку, что эти rows не смешиваются с обычными intent/topic aliases.

## Итоговые обязательные правки перед чистым PASS

1. Уточнить YAML schema: разделить canonical `symbol` и display `target`.
2. Добавить `content_hash_kind` + `content_sha256` в каждую YAML row и test gate на совпадение со snapshot.
3. Переименовать/уточнить `text_table` как machine rule, чтобы `154` не читалось как число смысловых таблиц.
4. В итоговом коммите явно добавить untracked `docs/adr003_understanding_map.yaml`.
5. Не считать полный `git diff --check ca1c9ce5` красным по D1, пока причина только старый `artifacts/gpt-5.5/problems_breaker.md`; но перед общим merge это лучше поправить отдельным tracked-cleanup.

## Не проверял

- LLM, AMO, Tallanto, CRM, live Wappi, ASR и batch-процессы не запускались.
- Полную смысловую разметку bucket1/4 не проверял, она заявлена за параллельным исполнителем.

## Final Recheck 2026-07-31

Вердикт после исправлений: `PASS`.

Проверял только текущую карту D1 v2 и связанные безопасные unit/AST-гейты. `src/**`, YAML, фикстуру и тесты не менял.

Что прошло:

- Целевой тест: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_adr003_regex_understanding_moratorium.py` -> `12 passed in 14.39s`.
- Exact `evidence_cases`: 14 YAML cases collect-only разворачиваются в 68 pytest items; запуск этих exact cases -> `68 passed in 1.30s`.
- Canonical schema: `symbol` совпадает со snapshot у `363/363` YAML rows; `display_symbol == target or symbol` у `363/363`.
- Content hash: `content_hash_kind/content_sha256` есть и совпадает со snapshot у `363/363` YAML rows.
- Counts: snapshot `832`; YAML rows `363`; bucket2 `336`; bucket3 `27`; unassigned `469`.
- Node kinds: `regex_call=381`, `string_contains=167`, `text_table=154`, `marker_helper_call=130`.
- Membership scope: `literal_left_membership_total=321 = 167 in canonical text scope + 154 outside canonical text scope`.
- `machine_text_table_rule` присутствует; старое ambiguous wording `text_table_rule` не используется.
- `deal_action_safety_floor`: `24` rows, все в `src/mango_mvp/channels/subscription_llm_parts/post_layers.py`, только `_DEAL_ACTION_*` declarations и `_deal_action_*` safety/helper functions; evidence cases по payment link и CRM identity прошли.
- Safety anchors `PAYMENT_CONFIRMATION_RE`, `_ROUTE_REFUND_RE`, `BRAND_FORBIDDEN_TERMS`, `UNSUPPORTED_PROMISE_PATTERNS`, `_DEAL_ACTION_*` и весь `p0_recall_spec.py` остаются bucket2.

Остаточных notes по заявленным пунктам нет.
