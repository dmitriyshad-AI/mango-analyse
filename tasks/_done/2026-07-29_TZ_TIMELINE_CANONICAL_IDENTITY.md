> DONE 2026-07-29 08:19 | ветка codex/timeline-canonical-identity | codex

> TAKE 2026-07-29 07:53 | ветка codex/timeline-canonical-identity | codex

Ветка: codex/timeline-canonical-identity
Зоны: src/mango_mvp/customer_timeline/ingestion.py, src/mango_mvp/customer_timeline/family_graph.py, src/mango_mvp/customer_timeline/store.py, scripts/build_customer_timeline_nightly_dv2_sources.py, tests/, docs/worktrees_registry.md, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_customer_timeline_ingestion.py tests/test_customer_timeline_codex_task.py
Семантический-аудит: да

# ТЗ: канонический клиент Tallanto/AMO

## Доказанный дефект

Полный staging-проход 20260729T042041Z скачал 19 202 карточки Tallanto и
остановился на смене канонического клиента: ранее временный ID был сопоставлен
с точным владельцем Tallanto, а новый разрешитель выбрал другой существующий
AMO-контакт по алфавиту. Одновременно AMO page_limit=50 воспроизводимо даёт
обрезанный ответ; page_limit=20 проходит.

## Сделать

1. При доказанном объединении выбирать существующего уникального владельца
   `tallanto_student_id`, а не минимальный строковый ID.
2. `amo_id` из карточки ученика Tallanto считать точной связью с семьёй, но не
   доказательством, что AMO-контакт и ученик — один человек. Родителя и ребёнка
   объединять только в семейном графе.
3. Не создавать `customer_id_mapping` для неизменившегося клиента A→A.
   Существующая старая A→A не должна блокировать будущую точную A→B.
4. Не склеивать разных учеников по общему телефону, email или AMO-контакту
   родителя; существующие полы семьи сохранить.
5. Установить AMO `page_limit=20` в канонической ночной конфигурации.
6. Добавить сквозные отрицательные тесты: выбор владельца Tallanto, повторный
   импорт без новых mappings, два ребёнка остаются раздельными, AMO page limit.

## Приёмка

- карточка с точными Tallanto+AMO ID сохраняет владельца Tallanto;
- повтор той же карточки не меняет канонический ID и не создаёт mappings;
- два Tallanto student_id общего родителя не объединяются;
- целевые тесты и полный pytest зелёные;
- новых флагов, зависимостей и параллельного механизма идентификации нет.

## СТОП

Не менять staging/prod данные в этом worktree, не запускать внешние системы,
не публиковать Timeline и не создавать вторую базу.
