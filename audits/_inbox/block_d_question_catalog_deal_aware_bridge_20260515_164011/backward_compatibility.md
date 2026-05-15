# Backward Compatibility

## Сохраняется

- Старый `source_ref` у call-вопросов не ломается.
- Без `question_catalog_source_index` deal-aware quality gate работает как раньше.
- Старые Stage5 outputs сохраняются.

## Добавлено

- Явные source-ключи рядом с хэшированным `source_ref`.
- Индекс `call_id -> themes/services/policies`.
- Optional CLI-аргумент:
  `--question-catalog-source-index`.

## Не добавлено

- Нет live-write.
- Нет обязательной зависимости Stage5 от question catalog.
