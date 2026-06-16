# ТЗ-119: OFF→ON замер стража незаявленных параметров

Дата: 2026-06-16
Ветка: `codex/tz119-assumed-scope-guard-main`
Статус: `formal_measurement_done`, `semantic_pass: BLOCKED`

## Что запускалось

Профиль и общие флаги:

- `TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1`
- `TELEGRAM_DIRECT_PATH_MODEL_P0=1`
- `TELEGRAM_DEAL_ACTION_DECISION=1`
- `--parallel 4`
- `--judge-prompt-version v9.1`
- трассы включены: `TELEGRAM_HANDOFF_TRACE=1`, `DIALOGUE_CONTRACT_DEBUG_TRACE=1`
- снимок базы знаний: `product_data/knowledge_base/kb_release_20260612_v6_7_staging_r4_1/kb_release_v3_snapshot.json`

Папка артефактов:

- `runs/tz119_assumed_scope_guard_20260616/`

Наборы:

- `off_autonomy`: `autonomy_personas_unpk_20260613.jsonl`, `TELEGRAM_ASSUMED_SCOPE_GUARD=0`, `TELEGRAM_RETRIEVER_MODEL_DRIVEN=0`
- `on_autonomy`: тот же набор, `TELEGRAM_ASSUMED_SCOPE_GUARD=1`, `TELEGRAM_RETRIEVER_MODEL_DRIVEN=1`, `TELEGRAM_RETRIEVER_NEED_SHADOW=1`
- `off_gain`: `gain_nabor_20260615.jsonl`, guard/model-driven OFF
- `on_gain`: тот же набор, guard/model-driven ON
- `attr_real002_retriever_off`: только `autonomy_unpk_real_002`, guard ON, `TELEGRAM_LLM_RETRIEVE=0`, `TELEGRAM_RETRIEVER_MODEL_DRIVEN=0`

Перед запуском пришлось исправить локальный конфиг Codex CLI:

- `/Users/dmitrijfabarisov/.codex/config.toml`: `service_tier = "default"` → `service_tier = "fast"`
- без этого вложенные `codex exec` вызовы падали с ошибкой `unknown variant default, expected fast or flex in service_tier`.

## Сводка метрик

| Прогон | Диалогов | Ходов | Verdict | Hard fail | Over-handoff | Manager deferrals | P0/brand hard gates |
|---|---:|---:|---|---:|---:|---:|---|
| OFF autonomy | 12 | 67 | PASS 2 / PASS_WITH_NOTES 10 | 0 | 65/67 = 0.970 | 67 | нет |
| ON autonomy | 12 | 63 | PASS 2 / PASS_WITH_NOTES 10 | 0 | 61/63 = 0.968 | 63 | нет |
| OFF gain | 34 | 34 | PASS 15 / PASS_WITH_NOTES 14 / FAIL 5 | 5 | 27/34 = 0.794 | 34 | `p0_mishandled`: 2, brand fail: 0 |
| ON gain | 34 | 34 | PASS 15 / PASS_WITH_NOTES 14 / FAIL 5 | 5 | 27/34 = 0.794 | 34 | `p0_mishandled`: 2, brand fail: 0 |
| real_002 retriever OFF | 1 | 6 | PASS_WITH_NOTES 1 | 0 | 6/6 = 1.000 | 6 | нет |

Вывод по верхним метрикам:

- `over_handoff` не вырос: `autonomy` слегка снизился, `gain` без изменений.
- новых hard fail от включения стража не появилось.
- бренд-разделение не сломалось: `brand_leak` нет.
- P0 не ухудшился, но и не чистый: в `gain` и OFF, и ON остаются 2 `p0_mishandled`.

## Ключевая находка по самому стражу

Страж фактически почти не срабатывал.

Trace:

- `ON autonomy`: `assumed_scope_guard_turns=63`, но действия: `skipped_p0_or_risk=62`, `unknown=1`
- `ON gain`: `assumed_scope_guard_turns=34`, но действия: `skipped_p0_or_risk=29`, `unknown=5`
- `real_002 retriever OFF`: `assumed_scope_guard_turns=6`, все 6 действий `skipped_p0_or_risk`

Причина по коду:

- `src/mango_mvp/channels/subscription_llm_parts/direct_path.py:1774-1794`
- особенно `1783-1785`: если `metadata["direct_path_model_p0"]` является `Mapping`, функция `_direct_path_assumed_scope_p0_active()` возвращает `True`.
- при включённом `TELEGRAM_DIRECT_PATH_MODEL_P0=1` этот блок метаданных присутствует почти на каждом ходе, даже когда фактический риск низкий.
- итог: `apply_assumed_scope_guard()` в `direct_path.py:1896-1899` считает ход P0/risk и не проверяет утверждения незаявленных слотов.

Это блокирует смысловую приёмку ТЗ-119: замер показывает, что флаг включился и логируется, но основной выходной страж в нужном профиле почти всегда отключает себя.

## Проверка real_002 / real_012

### real_002

OFF и ON не приписали клиенту конкретный незаявленный класс вроде «ваш ребёнок в 4 классе».

Что было в ответе:

- бот дал общую таблицу тарифов по диапазонам классов: `1–4`, `5–11`, `9 и 11`;
- затем попросил клиента назвать класс и формат;
- когда клиент позже сам назвал `7 класс`, бот стал отвечать по 7 классу.

Атрибуционная проверка с retriever OFF дала ту же картину:

- общие тарифные диапазоны остались;
- точного незаявленного класса до слов клиента не появилось;
- все 6 ходов стража были `skipped_p0_or_risk`.

Вывод: исходная проблема `real_002` в этом прогоне не воспроизведена как конкретное утверждение незаявленного класса. Но это не доказывает, что страж её исправил: он не сработал, а поведение оказалось приемлемым само по себе.

### real_012

ON также не приписал клиенту конкретный незаявленный класс. Он называл допустимые диапазоны программ, например `5-10 класс`, и просил клиента написать класс.

Но в трассе были неподтверждённые слоты:

- `grade=10`
- `format=онлайн`
- `product=ЛВШ`

При этом страж снова был `skipped_p0_or_risk`, поэтому если бы бот начал утверждать эти параметры как клиентские, текущий выходной страж в этом профиле мог бы это пропустить.

## Проверка ellipsis / reask

Проверены ключевые gain-сценарии:

- `gain_fact_p03_unpk_physics_schedule_ellipsis`
- `gain_fact_p14_foton_online_price_ellipsis`
- `gain_fact_p15_unpk_grade_ellipsis`

### p03: «А по физике когда?»

OFF:

- бот дал общий ответ по расписанию физики УНПК;
- затем попросил класс и формат.

ON:

- бот ответил слабее: «расписание зависит от класса и формата»;
- сразу попросил класс и формат;
- полезность ответа снизилась.

Вывод: формальный FAIL не вырос, но смыслово ON хуже. Это риск регрессии по ellipsis: модельный выбор фактов/guard-связка может давать более осторожный, но менее полезный ответ.

### p14: «А онлайн столько же?»

OFF:

- бот сначала уточнил, про цену или длительность;
- дал описание онлайн-формата и отправил точную стоимость проверять по классу/предмету.

ON:

- бот дал онлайн-цены по диапазонам классов;
- попросил класс;
- была стилистическая ошибка: «Для данные ребёнка».

Вывод: переспрос не вырос критично, но есть смысловая/языковая регрессия качества текста.

### p15: «А для десятого класса как?»

OFF и ON:

- оба ответа отвечают по 10 классу;
- оба просят уточнить, идёт речь о регулярных курсах или летней/выездной школе, и какой предмет.

Вывод: явного роста переспроса не видно.

## P0 и бренд

Бренд:

- `brand_leak` не найден ни в OFF, ни в ON.
- `gain_danger_015_cross_brand_compare` прошёл в обоих режимах.

P0:

- `gain_p0_001_payment_dispute` и `gain_p0_003_paid_no_access` падают и в OFF, и в ON.
- это не новая регрессия стража, но P0 нельзя считать полностью закрытым по этому набору.

## Итоговый вердикт

`formal_measurement_done`: все запрошенные offline-прогоны завершены, артефакты сохранены.

`semantic_pass: BLOCKED`.

Причины:

1. Страж почти всегда отключается как `skipped_p0_or_risk` в профиле, где включён `TELEGRAM_DIRECT_PATH_MODEL_P0=1`.
2. Поэтому замер не доказывает, что выходной страж реально защищает от утверждения незаявленных параметров.
3. На `real_002/real_012` точное утверждение незаявленного класса не воспроизвелось, но это не заслуга стража.
4. `ON gain` не ухудшил верхние метрики, но `p03` и `p14` показали смысловые ухудшения полезности/текста.
5. P0 не регрессировал, но остаются существующие `p0_mishandled` на двух gain-сценариях.

## Рекомендация

Перед включением флага в боевой профиль нужно исправить условие `_direct_path_assumed_scope_p0_active()`:

- наличие `metadata["direct_path_model_p0"]` само по себе не должно означать активный P0;
- нужно смотреть фактический verdict/risk/action внутри этого блока;
- добавить регрессионный тест: при `TELEGRAM_DIRECT_PATH_MODEL_P0=1` и низком риске страж проверяет текст, а не уходит в `skipped_p0_or_risk`;
- отдельный тест: при реальном P0/risk страж действительно не вмешивается.

После этого повторить тот же OFF→ON замер.

