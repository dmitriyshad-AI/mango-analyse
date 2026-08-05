Ветка: main
Зоны: product_data/knowledge_base/kb_release_20260612_v6_7_staging_r4_1_sources/, product_data/knowledge_base/kb_release_20260612_v6_7_staging_r4_1/, tests/test_kb_semantic_review.py, tests/test_kb_validity_window_runtime_gate.py, audits/, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_kb_semantic_review.py tests/test_kb_validity_window_runtime_gate.py tests/test_kb_release_v6_1_builder_sources.py
Семантический-аудит: да

# ТЗ: переподтвердить или закрыть истёкшие факты текущей KB

## Проблема

Read-only semantic review текущего snapshot на 2026-08-05:

- snapshot SHA256: `f99ea55c11b589f93976818e6918016a3ce86e7d1f7633aac616fe267df05bb3`;
- `semantic_pass=false`;
- 9 P1: четыре истёкших ценовых окна и пять уже закончившихся периодов;
- 304 P2 по нарушению SLA свежести.

Формальная сборка остаётся зелёной, но это не разрешение использовать факты как
актуальные. Ослаблять semantic gate или просто менять дату проверки запрещено.

## Образ результата

Каждый из девяти P1 и каждый бизнес-критичный P2 либо подтверждён свежим первичным
источником с датой/владельцем, либо исключён из клиентского доступа. Текущий builder
используется повторно; нового сборщика, regex-парсера или параллельной KB не появляется.

## Порядок

1. Сначала инвентарь уже имеющихся свежих source overlays, Google/DOCX-экспортов,
   решений владельца и существующего сырьевого контура.
2. Сформировать таблицу `fact_id -> источник -> решение keep/update/block`.
3. Менять только source overlay; Python не использовать для бизнес-патчей.
4. Пересобрать через `scripts/build_kb_release_v6_1_team_answers.py` в новый staging output.
5. Выполнить formal и независимый semantic review, затем diff старого и нового релиза.

## Приёмка

- P1 = 0;
- для всех 304 P2 есть замкнутый баланс: revalidated + client_blocked + явно
  оставленные internal-only = 304;
- ни один истёкший факт не стал свежим только из-за подмены даты;
- цены, даты, скидки и места сверены с первичными источниками;
- current runtime не переключён без отдельного решения;
- добавлено 0 новых production-модулей, флагов и зависимостей.

## СТОП

Остановиться и запросить решение владельца, если первичный источник противоречит
текущему факту или свежего подтверждения коммерческого условия нет.
