> DONE 2026-07-25 13:59 | ветка codex/ai-employee-final | codex

Ветка: codex/ai-employee-final
Зоны: src/, scripts/, tests/, docs/, tasks/, launchd/, m1_exam/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest --collect-only -q tests/test_customer_timeline_store.py tests/test_wappi_history_import_to_timeline.py tests/test_customer_timeline_manager_dossier.py
Семантический-аудит: да

# Консолидация веток ИИ-сотрудника

Источник требований: `/Users/dmitrijfabarisov/Claude Projects/Foton/2026-07-25_TZ_CLAUDE_dovesti_Mango_do_polnocennogo_AI_sotrudnika.md`, этап 0.

## Цель

Собрать одну чистую интеграционную линию поверх актуального `main`, перенеся только подтверждённую полезную работу из локальных веток Customer Timeline, Wappi, AMO, Tallanto и измерителей. Не переносить snapshot/stash-коммиты, чужой `uv.lock`, незавершённый Owner50 и устаревшие копии документов.

## Порядок

1. Зафиксировать ветки, worktree, статусы, уникальные коммиты, patch-id и процессы.
2. Создать `branch_commit_matrix.md` со статусами `PORT`, `SUPERSEDED`, `DUPLICATE`, `TEST_ONLY`, `REJECT`.
3. Перенести spine, затем source/nightly contracts, family/P0, mail, Tallanto, Wappi, AMO idempotency и принятые измерители.
4. После каждого блока запускать его точечные тесты; в конце — полный безопасный pytest, import smoke и semantic review.
5. Создать один audit pack. Не удалять исходные ветки/worktree до приёмки интеграции.

## Стоп-условия

- активный или грязный worktree затрагивается переносом;
- непонятный конфликт меняет P0, бренд, ПДн, identity или write-контракт;
- коммит содержит runtime/ПДн/боевую SQLite;
- тест или смысловой аудит красный.

## Приёмка

- одна чистая интеграционная ветка и зарегистрированный worktree;
- все локальные ветки и уникальные SHA получили доказанный статус;
- перенесены только принятые коммиты без snapshot/stash и чужого `uv.lock`;
- точечные и полные безопасные тесты зелёные;
- audit pack содержит diff, тесты, риски и смысловой вердикт.
