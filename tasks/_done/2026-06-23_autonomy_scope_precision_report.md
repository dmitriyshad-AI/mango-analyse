# TELEGRAM_AUTONOMY_SCOPE_PRECISION — отчёт D1

Дата: 2026-06-23 07:11 MSK  
Ветка: `codex/autonomy-scope-precision`  
База: `main` / `a9f80ba`  
ТЗ: `/Users/dmitrijfabarisov/Claude Projects/Foton/2026-06-23_TZ_autonomiya_za_flagom_DRAFT.md`, секция v2, только Кандидат 1.

## Read-only карта

- Живая точка проблемы: `src/mango_mvp/channels/dialogue_contract_pipeline.py`, `_wrong_intent_fact_findings` вызывает `_asks_address(contract)` и при `False` блокирует адресный факт как `wrong_intent_fact`.
- Узкое место: `_asks_address` ловил `адрес|площадк|где вы|где находит|куда ехать|куда ездить`, но не ловил `как доехать/добраться/проезд/маршрут`.
- ЛВШ/камп-ветка отдельная: `_draft_uses_camp_or_lvsh_fact` / `_contract_mentions_camp_or_lvsh`; её не менял.
- `_draft_uses_address_fact` не менял.

## Что изменено

- Добавлен флаг `TELEGRAM_AUTONOMY_SCOPE_PRECISION`, default OFF.
- Ветвление сделано внутри `_asks_address`: при ON добавлены адресные синонимы `как доехать`, `как добраться`, `как попасть`, `как пройти`, `как проехать`, `доехать`, `добраться`, `проезд`, `маршрут`.
- Флаг не добавлен в `DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS`; есть тест на это.
- Добавлены unit NEG/POS:
  - default OFF и не в профиле;
  - `Как доехать...` OFF даёт `wrong_intent_fact`, ON не даёт;
  - C1: вопрос про цену + адресный факт всё ещё блокируется;
  - C3: ЛВШ вне контекста всё ещё блокируется.
- Добавлен микро-набор и комбинированный набор частые-20 + микро.

## Наборы

- Микро: `product_data/telegram_dynamic_test_sets/autonomy_scope_precision_micro_20260623.jsonl`
  - sha256: `2b29c45ba865154507ea330db1cda01e15df94a275d976867a5fc3d65ee66950`
- Частые-20 + микро: `product_data/telegram_dynamic_test_sets/autonomy_scope_precision_freq20_plus_micro_20260623.jsonl`
  - sha256: `5db121790b6dbee5431c1cbc0b7ab282eeefbc3d75c933514cc07400a9553d2c`
  - 23 диалога: 20 частых + 3 микро.

## Тесты

- Точечные: `9 passed, 501 deselected in 1.43s`.
- Полный pytest: `3603 passed, 5 skipped, 1 warning in 78.15s`.
- Warning: системный `urllib3 NotOpenSSLWarning` из-за LibreSSL, не по изменённому коду.

## Прогоны OFF/ON

Окружение обоих прогонов:

- `TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1`
- `TELEGRAM_STEP4_KEEP_ANSWER=0`
- `--judge-prompt-version v9.1`
- `--parallel 4`
- `--model gpt-5.5`
- временный `CODEX_HOME=/tmp/mango_codex_home_autonomy_scope_precision`

OFF:

- Флаг: `TELEGRAM_AUTONOMY_SCOPE_PRECISION=0`
- Папка: `runs/20260623_autonomy_scope_precision_OFF`
- Итог: `dialogs=23`, `turns=57`, `pass=8`, `pass_with_notes=8`, `fail=7`, `hard_gate_failures=7`, `ok=true`
- `config_validity.invalid=false`

ON:

- Флаг: `TELEGRAM_AUTONOMY_SCOPE_PRECISION=1`
- Папка: `runs/20260623_autonomy_scope_precision_ON`
- Итог: `dialogs=23`, `turns=65`, `pass=11`, `pass_with_notes=8`, `fail=4`, `hard_gate_failures=4`, `ok=true`
- `config_validity.invalid=false`

## Микро-кейсы

OFF:

- `asp_pos_foton_address_how_to_get`: `PASS`, route на первом ходу `bot_answer_self_for_pilot`, но safety flags включали `authoritative_gate:wrong_intent_fact` с последующим promote.
- `asp_neg_c1_price_question_address_only`: `PASS_WITH_NOTES`; адрес в клиентский текст не выдан.
- `asp_neg_c3_lvsh_out_of_context`: `PASS_WITH_NOTES`; ЛВШ/Менделеево не раскрыты, первый ход остался `draft_for_manager`.

ON:

- `asp_pos_foton_address_how_to_get`: `PASS`, адрес Фотона выдан на прямой вопрос `Как доехать...`.
- `asp_neg_c1_price_question_address_only`: `PASS`, адрес в клиентский текст не выдан.
- `asp_neg_c3_lvsh_out_of_context`: `PASS_WITH_NOTES`, ЛВШ/Менделеево/МФТИ не раскрыты.

## Частые-20: что отдать на регрейд

OFF FAIL:

- `sm_f_install` — `wrong_intent_fact_leak`
- `sm_f_format_both` — `wrong_intent_fact_leak`
- `sm_f_price_short` — `wrong_intent_fact_leak`
- `sm_u_price_weekend` — `fabrication`
- `sm_f_platform` — `timeout` инфраструктуры Codex
- `sm_f_camp1` — `timeout` инфраструктуры Codex
- `sm_u_camp1` — `fabrication`

ON FAIL:

- `sm_f_format_both` — `wrong_intent_fact_leak`
- `sm_f_price_short` — `fabrication` / факт вне текущего вопроса
- `sm_f_camp1` — `fabrication` / адрес вне текущего вопроса
- `sm_u_address` — `wrong_intent_fact_leak` из-за ухода в ЛВШ на московскую площадку УНПК

Вердикт по качеству не выношу; это сырьё для Claude #1.

## Дополнительная проверка утечек клиентского текста

По `bot_text` в `dynamic_dialog_transcripts.jsonl` для OFF и ON:

- `[данные у менеджера]`: 0
- `fact_id`: 0
- `source_id`: 0
- `client_blocked`: 0
- `internal_only`: 0
- `клиенту суммы не называть`: 0
- `presentation_format_facts`: 0

Сырые JSONL содержат эти строки в metadata/fact payload, поэтому проверял именно клиентский текст, не весь файл.

## Границы

- `TELEGRAM_AUTONOMY_SCOPE_PRECISION` default OFF.
- В боевой профиль не добавлял.
- Live bot, AMO, Tallanto, `stable_runtime`, M1, push/merge не трогал.
- `TELEGRAM_STEP4_KEEP_ANSWER` на замере держал `0`.
