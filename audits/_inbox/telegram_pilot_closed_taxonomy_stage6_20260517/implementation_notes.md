# Telegram Pilot Closed Taxonomy Stage 6

Дата: 2026-05-17

## Что сделано

Закрыт риск, что LLM придумывает `topic_id` вне утвержденного списка тем РОПа.

Изменения:

- В prompt для черновиков добавлен закрытый список из 32 тем и 5 служебных категорий из `src/mango_mvp/question_catalog/themes_taxonomy.yaml`.
- В prompt явно запрещены любые новые `topic_id` и `alternative_themes` вне закрытого списка.
- Добавлены специальные правила выбора темы для возврата, материнского капитала, налогового вычета, статуса оплаты, способа оплаты, рассрочки, скидок, пробного занятия, материалов, отсутствующих ссылок и явного отказа клиента.
- Добавлена post-LLM проверка: если модель все равно возвращает неизвестную тему, результат переводится в `service:S2_unclear`, маршрут становится `manager_only`, а в флаги и metadata добавляется причина.
- Повторен Stage 6 прогон на тех же 20 Telegram-диалогах.

## Результат Stage 6

Артефакты локального исторического прогона:

- `.codex_local/telegram_pilot/eval_packs/20260517_contextual_layer_smoke/llm_drafts_stage6_taxonomy_20260517/stage6_llm_drafts_for_manual_review.csv`
- `.codex_local/telegram_pilot/eval_packs/20260517_contextual_layer_smoke/llm_drafts_stage6_taxonomy_20260517/stage6_llm_drafts_for_manual_review.xlsx`
- `.codex_local/telegram_pilot/eval_packs/20260517_contextual_layer_smoke/llm_drafts_stage6_taxonomy_20260517/stage6_llm_drafts_full.jsonl`
- `.codex_local/telegram_pilot/eval_packs/20260517_contextual_layer_smoke/llm_drafts_stage6_taxonomy_20260517/summary.json`
- `.codex_local/telegram_pilot/eval_packs/20260517_contextual_layer_smoke/llm_drafts_stage6_taxonomy_20260517/quality_checks.json`

Ключевые числа:

- строк: 20;
- ошибок выполнения: 0;
- неизвестных `topic_id`: 0;
- маршруты: `draft_for_manager` - 8, `manager_only` - 12;
- типы сообщений: `question` - 12, `context_update` - 5, `non_question` - 2, `wait_for_more` - 1.

Контрольные кейсы:

- `1063099421`: вопрос про возврат теперь классифицирован как `theme:009_refund`, маршрут `manager_only`;
- `1084253673`: вопрос про пропущенный урок и материалы классифицирован как `theme:018_materials_homework`, маршрут `draft_for_manager`.

## Ограничения

- Это исторический локальный прогон, не live-режим.
- Сообщения клиентам не отправлялись.
- Записей в AMO/CRM/Tallanto не было.
- `stable_runtime` не менялся.
- Вопрос качества самих черновиков остается предметом ручной проверки Дмитрия/Насти; текущий блок закрывает именно контроль тем и безопасный маршрут.
