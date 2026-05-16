# Question Catalog v2 Migration Notes

Дата: 2026-05-14

## Этап C

Сделано:

- Добавлен `src/mango_mvp/question_catalog/classifier.py` как новая точка входа классификации.
- `classify_question()` возвращает `theme_id`, извлеченные параметры, уверенность, способ классификации, факты и режим ответа из таксономии.
- `assign_theme()` пока является заглушкой: LLM не используется, работает только простая логика на существующих регулярках из `normalization.py`.
- `normalization.infer_question_metadata()` переведен в тонкую обертку над новым классификатором.
- Старые функции склейки и fallback-дробления физически удалены из `normalization.py`.
- `extractors.py` переведен на `theme_id` как ключ группировки; старый `class_key` больше не пишется в metadata новых `QuestionItem`.
- `builder.py` при fallback использует `theme_id` и человекочитаемое название темы.
- Старые тесты вокруг `normalization.py` переписаны под v2-контракт: теперь они проверяют `theme_id`, параметры и отсутствие старых полей `intent/subclass_key/class_key/canonical_question`.

## Что осталось историческим до этапов E/F

- `contracts.QuestionClass` пока сохраняет поле `class_key`, потому что вокруг него завязаны старые экспортные файлы и вспомогательные отчеты. На этапе C туда передается `theme_id`, а не старое произведение `intent/subclass/product/subject/grade/format`.
- Поле `canonical_question` в старых выгрузках используется как человекочитаемый заголовок. Оно больше не строится старой функцией, а наполняется названием темы.
- Старые тесты v1 на подробные subclass-key больше не являются контрактом. Если на этапах E/F останутся экспортные проверки со старыми названиями колонок, их нужно переводить на `theme_id` и названия тем.

## Дополнительный аудит callers после этапа C

Проверено `class_key` и `canonical_question` по `src/`, `scripts/`, `tests/`, `docs/`.

Результат:

- Парсинга старого `class_key` через `intent=`, `subclass=` или `split("|")` в production-коде нет.
- Тестовые fixtures с `class_key="intent=..."` переведены на `theme:*`/`service:*`.
- Связка audit/ROP pack через `canonical_question` усилена: новые `rop_review_priority_top100` строки получают `ID класса`, а `answer_review_pack.py` и `build_rop_blocker_markup_pack.py` сначала связывают строки по `question_class_id`, затем fallback по `canonical_question` для старых файлов.
- `canonical_question` остается только отображаемым названием класса/темы и сортировочным label.

TODO на этап E/F:

- В `builder.py` осталась legacy-защита широких v1 fallback buckets (`base_price`, `base_schedule`, `general_*`, `*_manual_review`). Для новых v2 items это не активный контракт, потому что `question_subclass_key` теперь равен `theme_id`; блок оставлен только для чтения старых CSV до полной миграции экспортов.
