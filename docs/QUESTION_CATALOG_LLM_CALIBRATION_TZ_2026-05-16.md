# ТЗ: LLM-калибровка каталога клиентских вопросов

Дата: 2026-05-16

## Контекст

Telegram-пилот не сможет давать качественные черновики, если система плохо понимает тему вопроса клиента. Сейчас в проекте уже есть каталог из 9 969 клиентских вопросов, таксономия из 32 бизнес-тем и 5 служебных категорий, а также 100 вручную размеченных строк для проверки качества.

Проблема: старый rule-only классификатор на 100 строках показывает около 37% точности и macro-F1 около 0.326. Старый Codex A/B прогон лучше, но тоже ниже целевого порога 0.85. Значит полный прогон 9 969 вопросов нельзя считать готовым к боевому использованию без безопасной обвязки, проверки качества и ручной доразметки сложных мест.

## Цель

Сделать управляемый контур для LLM-калибровки question catalog:

1. Классифицировать 9 969 вопросов через Codex CLI по подписке, без API-ключей.
2. Сохранять прогресс по батчам, чтобы прогон можно было продолжать после сбоя.
3. Валидировать ответы модели до принятия результата.
4. Пересобирать каталог в отдельную новую папку, не перезаписывая текущий рабочий каталог.
5. Дать проверки, что бот и deal-aware слой получают валидные темы и не обходят правила РОПа.

## Жесткие границы

- Не запускать ASR.
- Не запускать Resolve+Analyze по реальным данным.
- Не писать в AMO, CRM, Tallanto.
- Не отправлять сообщения клиентам.
- Не писать в `stable_runtime`.
- Не перезаписывать текущие `product_data/question_catalog/customer_question_items.jsonl` и `customer_question_classes.csv`.
- Полный прогон 9 969 вопросов запускать только как восстанавливаемый локальный batch-run.
- Сырые ответы модели хранить только в локальной `.codex_local`, не в audit pack.

## Реализация

### Блок 1. Full-run runner

Добавить скрипт:

`scripts/run_question_catalog_codex_full_v2.py`

Он читает:

- `product_data/question_catalog/customer_question_items.jsonl`
- `src/mango_mvp/question_catalog/themes_taxonomy.yaml`

Пишет только в:

- `.codex_local/question_catalog/codex_full_v2/<run_id>/`

Каждый batch должен иметь:

- `raw/<batch>.response.txt`
- `raw/<batch>.response.json`
- `raw/<batch>.meta.json`
- `predictions/<batch>.jsonl`
- `cache/<sha>.json`

Итоговые файлы:

- `run_manifest.json`
- `batch_plan.jsonl`
- `progress.json`
- `predictions_all.jsonl`
- `predictions_all.csv`
- `low_confidence_review_queue.csv`
- `summary.json`

### Блок 2. Проверка ответа модели

Batch считается завершенным только если:

- все входные `question_item_id` вернулись ровно один раз;
- нет лишних `question_item_id`;
- `theme_id` есть в текущей таксономии;
- `confidence` в диапазоне 0..1;
- JSON валиден;
- predictions записаны атомарно.

### Блок 3. Пересборка каталога

Добавить скрипт:

`scripts/rebuild_question_catalog_from_llm_predictions_v2.py`

Он читает:

- исходные 9 969 вопросов;
- `predictions_all.jsonl` из full-run.

Пишет новый каталог в отдельную папку, например:

`product_data/question_catalog/rebuild_llm_v2_20260516/`

Текущий каталог не трогать.

### Блок 4. Проверка для Telegram-бота

После пересборки нужно проверить:

- темы распознаны валидно;
- опасные темы остаются только для менеджера;
- вопросы про цену и расписание не дают финальный ответ без свежих фактов;
- неутвержденные темы не уходят в автоответ;
- Telegram-пилот остается в режиме черновиков для менеджера.

## Acceptance criteria

- Full-run runner умеет `--dry-run-plan-only`.
- Full-run runner умеет `--max-rows` для безопасного малого прогона.
- Full-run runner не требует `OPENAI_API_KEY`.
- Full-run runner запускает Codex CLI с `--sandbox read-only`.
- Output под `stable_runtime` запрещен.
- Неполный или битый batch не считается завершенным.
- Resume пропускает только валидные complete batch.
- Пересборка сохраняет количество вопросов.
- Пересборка меняет `question_class_id` и metadata согласно LLM prediction.
- Текущий каталог не перезаписывается.

## Тесты

Добавить тесты:

- `test_full_prompt_marks_client_text_as_untrusted`
- `test_validate_predictions_rejects_missing_duplicate_unknown_theme_and_bad_confidence`
- `test_batch_resume_requires_meta_and_prediction_count`
- `test_full_run_output_rejects_stable_runtime`
- `test_rebuild_from_predictions_updates_theme_metadata_and_preserves_question_count`
- `test_source_index_json_roundtrip_preserves_manager_only_mode`

## Порядок запуска

1. Прогнать тесты.
2. Сделать dry-run plan.
3. Сделать малый прогон на 5-10 строк.
4. Посмотреть `low_confidence_review_queue.csv`.
5. Только после этого принимать решение о полном прогоне 9 969 вопросов.

## Что не закрывает это ТЗ

- Ручную разметку 250-300 строк.
- Финальное улучшение prompt до целевого качества 0.85.
- Автоматическую отправку сообщений клиентам.
- Обновление AMO/Tallanto.
- Полную переобработку звонков.

