# post_layers.py KB-backed price few-shot

Дата: 2026-06-16
Ветка: `codex/postlayers-kb-prices`

## Что исправлено

В `build_semantic_output_verifier_prompt` был бренд-ошибочный few-shot:

- бренд: Фотон;
- цены: `49 000 ₽ / 82 000 ₽`;
- по актуальной r4.1 KB эти цены относятся к УНПК очно 5-11, а не к Фотону.

Правка:

- удалены ценовые литералы из `post_layers.py`;
- few-shot теперь строится из Фотон-фактов KB:
  - `prices_regular_2026_27.offline_5_11_class.before_2026_07_01.semester`;
  - `prices_regular_2026_27.offline_5_11_class.before_2026_07_01.year`;
- используется существующая проверка client-safe + `valid_until` через `_direct_path_fact_by_brand_key`;
- если факты отсутствуют/просрочены, prompt использует нейтральный пример без чисел.

## Проверка KB

Актуальный r4.1 snapshot:

`product_data/knowledge_base/kb_release_20260612_v6_7_staging_r4_1/kb_release_v3_snapshot.json`

Подтверждённые Фотон-факты:

- семестр: `44 600 ₽`, `valid_until=2026-07-01`;
- год: `74 500 ₽`, `valid_until=2026-07-01`.

Сгенерированный prompt содержит:

`Фотон: цены на 2026/27 учебный год, 5-11 класс, очно, семестр — 44 600 ₽.`

`Фотон: цены на 2026/27 учебный год, 5-11 класс, очно, год — 74 500 ₽.`

И не содержит старые `49 000 ₽ / 82 000 ₽` в Фотон few-shot.

## Tests

Точечно:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_subscription_llm_draft_provider.py -k "semantic_output_verifier"
```

Результат: `23 passed, 460 deselected`.

Полный pytest:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests
```

Результат: `3306 passed, 5 skipped, 1 warning in 47.57s`.

## Semantic review

Verdict: `PASS_WITH_NOTES`.

Что прошло:

- брендовая ошибка устранена;
- значения берутся из KB, а не из кода;
- `valid_until` защищает от использования просроченного факта;
- при отсутствии валидного факта prompt не держит старую цену.

Остаточный риск:

- после `2026-07-01` эти Фотон-факты станут просроченными, и few-shot перейдёт на нейтральный вариант без чисел. Это ожидаемое fail-soft поведение, но после обновления KB стоит проверить, что новые цены снова доступны через те же или обновлённые fact_key.

