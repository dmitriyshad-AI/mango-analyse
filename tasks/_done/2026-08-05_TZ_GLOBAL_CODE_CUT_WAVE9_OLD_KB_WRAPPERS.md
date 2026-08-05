> DONE 2026-08-05 03:11 | ветка main | codex

> TAKE 2026-08-05 02:59 | ветка main | codex

Ветка: main
Зоны: scripts/build_kb_release_v2_from_claude_and_codex.py, scripts/build_kc_knowledge_snapshot.py, scripts/build_kc_night_audit_pack.py, tests/test_kb_release_v2_import.py, tests/test_kc_knowledge_snapshot.py, tests/test_kc_night_audit_pack.py, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_kc_knowledge_snapshot.py tests/test_build_full_kc_knowledge_base.py tests/test_build_kc_final_release.py tests/test_extract_kc_google_doc_facts.py tests/test_kb_release_v6_1_builder_sources.py tests/test_kb_validity_window_runtime_gate.py tests/test_kb_semantic_review.py
Семантический-аудит: да

# ТЗ: Wave 9 — удалить устаревшие KB-обвязки, не потеряв сырьевой импорт

## Проблема

В репозитории одновременно лежат майские поколения сборки KB и текущий канон
`build_kb_release_v6_1_team_answers.py` + source overlay v6.7. Массовое удаление всех
старых файлов опасно: `build_kc_final_release.py` ->
`build_full_kc_knowledge_base.py` -> `extract_kc_google_doc_facts.py` остаётся
единственным воспроизводимым офлайн-входом для сырых Google/DOCX/site-источников.

Одновременно три CLI не имеют живых вызовов и не входят в текущий runbook:

- старый v2 release builder;
- старый CLI metadata-only snapshot;
- старый ночной audit-pack builder.

Тест `test_kc_knowledge_snapshot.py` смешанный: только один тест относится к
удаляемому CLI, остальные пять проверяют живой публичный `fact_registry.py`.

## Образ результата и бизнес-польза

Текущая v6.7 KB продолжает загружаться тем же runtime-кодом. Возможность заново
принять сырые Google/DOCX-экспорты не потеряна. При этом из проекта исчезает
примерно 2,7 тыс. строк ложных альтернативных точек запуска, которые могут
собрать устаревший релиз или отчёт и ввести исполнителя в заблуждение.

## Рассмотренные варианты

1. Удалить все 12 файлов: отвергнуто, теряется уникальный сырьевой вход.
2. Удалить только v2 и night-audit пары: безопасно, но оставляет доказанно мёртвый snapshot CLI.
3. Удалить три CLI, два их чистых теста и один CLI-тест из смешанного файла: выбран минимальный полный вариант.

## Изменения

Удалить целиком:

- `scripts/build_kb_release_v2_from_claude_and_codex.py`;
- `tests/test_kb_release_v2_import.py`;
- `scripts/build_kc_knowledge_snapshot.py`;
- `scripts/build_kc_night_audit_pack.py`;
- `tests/test_kc_night_audit_pack.py`.

В `tests/test_kc_knowledge_snapshot.py` удалить только импорт CLI и
`test_build_kc_knowledge_snapshot_cli_writes_snapshot`. Остальные пять тестов
оставить без смысловых изменений.

Не менять и не удалять:

- `scripts/build_kc_final_release.py`;
- `scripts/build_full_kc_knowledge_base.py`;
- `scripts/extract_kc_google_doc_facts.py`;
- их тесты;
- `src/mango_mvp/knowledge_base/fact_registry.py`;
- текущие source overlays и KB runtime.

## Приёмка

1. Поиск удалённых имён не находит живых вызовов; исторические документы не переписываются.
2. Пять оставшихся тестов `test_kc_knowledge_snapshot.py` проходят.
3. Три теста уникального сырьевого контура проходят.
4. Импорт `scripts.build_kb_release_v6_1_team_answers` проходит.
5. Текущий KB snapshot и source overlays побайтово не меняются.
6. Полный pytest не получает новых падений относительно исходного HEAD.
7. Отрицательный контроль: импорт удалённого CLI падает, импорт текущего builder проходит.
8. Отдельно зафиксировать, что свежая смысловая проверка текущей KB уже до этой
   волны находит истёкшие факты; это не исправлять и не маскировать уборкой.

## СТОП

Остановиться, если найден живой runbook/launchd/importlib/subprocess-вызов удаляемого
CLI или если после удаления меняется текущий KB snapshot/runtime output.

## Бритва

Новых файлов кода, флагов, зависимостей и абстракций: 0. Добавление нетестового
кода: 0. Эта задача только удаляет ложных владельцев и сохраняет единственный
доказанный сырьевой путь.
