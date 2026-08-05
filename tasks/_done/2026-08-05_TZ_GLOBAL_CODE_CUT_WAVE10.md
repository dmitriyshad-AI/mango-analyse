> DONE 2026-08-05 12:13 | ветка codex/global-code-cut-wave10 | codex

> TAKE 2026-08-05 11:40 | ветка codex/global-code-cut-wave10 | codex

Ветка: codex/global-code-cut-wave10
Зоны: src/mango_mvp/crm_card_aggregator.py, src/mango_mvp/crm_card_amo_writeback.py, src/mango_mvp/crm_card_history_summary.py, src/mango_mvp/crm_card_workbook.py, src/mango_mvp/customer_timeline/crm_export_package.py, src/mango_mvp/deal_aware/, src/mango_mvp/quality/crm_text_quality_detector.py, scripts/, tests/, D1_audit_backlog/codex_tests/, docs/, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest --collect-only -q tests/test_subscription_llm_draft_provider.py tests/test_draft_loop.py tests/test_customer_timeline_nightly_service.py tests/test_deal_aware_amo_rollback.py tests/test_amo_write_safety.py tests/test_amo_writeback_guards.py tests/test_question_catalog_deal_aware_bridge.py tests/test_customer_timeline_context_provider.py tests/test_customer_timeline_next_step_resolver.py tests/test_customer_timeline_safe_copy.py
Семантический-аудит: нет

# Уборка волна 10: старый файловый deal-aware/CRM-writeback

## Бизнес-результат

В целевом пути остаются только Wappi -> черновик AMO -> менеджер и Customer
Timeline. Старый файловый конвейер Stage 2-6/709 и CRM-card больше не имеет
runtime-входа, а его производные данные удалены решением D-032. Удаляем весь
замкнутый конвейер, чтобы не оставить Stage 2/3 без потребителя или Stage 6 без
генератора. Текущий `amocrm_runtime`, writeback его API и rollback сохраняются.

## До правок

1. Зафиксировать SHA, чистоту worktree и число собираемых тестов.
2. Graphify использовать только как навигацию; все зависимости перепроверить
   `rg` на текущем SHA.
3. Доказать отсутствие кандидатов в launchd, `docs/RUNBOOK.md`, `deploy/`,
   `Makefile`, `pyproject.toml`, активных ТЗ и импортном замыкании Wappi/Timeline.
4. Отдельно найти смешанные тесты и сохранить проверки живых владельцев.

## Изменение

- удалить старые CRM-card/history/export, deal-aware Stage 2-6/709, их CLI,
  прямые тесты и frozen fixtures;
- удалить ручной D1-тест этого же конвейера и его инструкции запуска;
- в смешанных тестах удалить только кейсы и импорты удалённого контура;
- сохранить Stage 1, потому что Timeline использует его XLSX-reader;
- сохранить `amo_rollback`, `amo_write_safety`, текущий `amocrm_runtime` и
  `write_recent_actionable_deals`;
- перенести в rollback-CLI только загрузку env и DB-preflight из удаляемого
  писателя; не создавать новый helper/module;
- не менять P0, direct-path, Customer Timeline ingestion/read API, Wappi,
  внешние системы и runtime-данные;
- не добавлять флаги, зависимости или совместимые заглушки; добавление кода
  ограничено переносом двух rollback-функций (до 30 строк).

## Приёмка

1. `rg` по удалённым модулям в `src/`, `scripts/`, `tests/`, `deploy/`,
   `pyproject.toml` не находит исполняемых ссылок.
2. Импорты `mango_mvp.channels`, draft-loop и Customer Timeline проходят.
3. Сохранённые Stage 1/rollback/AMO-runtime тесты, живые тесты и полный collect-only зелёные.
4. Полный pytest не имеет новых падений относительно `61bc54da`.
5. Число удалённых строк существенно больше добавленных; новых файлов, флагов,
   зависимостей и LLM-вызовов нет.
6. Ломатель подтверждает, что удалённый код нельзя вызвать из целевого пути и
   что Stage 1/rollback/AMO-runtime не потеряны.

## Стоп

- найден живой/плановый запускатель или активное ТЗ на файловый Stage 2-6/709;
- удаление требует менять живую семантику Wappi/Timeline;
- rollback не отделяется переносом двух уже существующих функций;
- появились чужие изменения в worktree.

## Результат

- итог коммита: удалено 26 259 строк, добавлено 158 (включая это ТЗ),
  чистое уменьшение 26 101 строка;
- удалены 11 модулей, 14 CLI, 14 прямых тестовых файлов и 5 фикстур старого
  файлового конвейера; новых файлов кода, флагов, зависимостей и LLM-вызовов нет;
- Stage 1, rollback, текущий `amocrm_runtime`, Wappi и Customer Timeline
  сохранены; rollback-CLI больше не импортирует удалённый writer;
- `rg` и import-smoke не нашли исполняемых ссылок на удалённые модули;
- 565 целевых тестов прошли; полный прогон: 4 952 passed, 3 skipped и те же
  8 KB-падений, что на исходном `61bc54da`;
- рассмотрено более простое удаление rollback и current AMO API, но отвергнуто:
  rollback нужен для уже существующих snapshot, а внешнее использование API
  нельзя опровергнуть только поиском по репозиторию.
- ломатель дал PASS; найденный им ручной D1-пакет также удалён целиком: два
  файла импортировали отсутствующие модули, а Timeline-тест закреплял отменённое
  правило «общий телефон = конфликт».
