# Git/project order, 2026-05-12

Цель: вернуть проект к управляемому состоянию без удаления данных, без `git reset/checkout` и без изменения runtime DB/audio/transcripts.

## Текущий снимок

- Ветка: `codex/saas-productization-baseline`.
- Локально есть отдельный hygiene-коммит: `efb7f87 Ignore generated runtime and audit artifacts`.
- Рабочее дерево все еще большое: 36 измененных tracked-файлов и 243 видимых untracked-файла.
- `stable_runtime` занимает около `32G`, весь проект около `68G`.
- `.git` после `git gc` уменьшен примерно с `4.2G` до `3.2G`.
- В `.git` осталось около `881M` временных `tmp_pack/tmp_obj`; удалять их можно только отдельным подтвержденным шагом.

## Что уже сделано безопасно

- Расширен `.gitignore`, чтобы новые audit-паки, runtime-артефакты, офисные выгрузки, карантинные папки и внешние handoff-папки не засоряли `git status`.
- Выполнен обычный `git gc`; рабочие файлы не трогались.
- `git diff --check` проходит без ошибок.

## Что не должно попадать в Git

- Новые `stable_runtime/*` artifact-директории, кроме явно выбранных маленьких pointer/runbook-файлов.
- `audits/_inbox/*` и `audits/_results/*`, кроме `.gitkeep` и уже отслеживаемых исторических файлов.
- Офисные выгрузки: `.xls`, `.xlsx`, `.docx`, `.zip`.
- `_cleanup_quarantine_*`, `_external_handoffs`, локальные `.venv*`, `.codex_*`, `.cache`.
- Сырые аудио, большие DB, транскрипты и runtime-кэши.

## Что нужно разложить по коммитам

1. Transcript-quality / safety slice:
   `src/mango_mvp/quality/`, transcript-quality scripts, tests `test_transcript_quality*`, `test_non_conversation_quality.py`, bot/CRM quality tests, профильные docs.

2. Canonical / CRM writeback slice:
   `src/mango_mvp/maintenance/`, AMO manual/duplicate/waiting modules, AMO writeback scripts, CRM quality scripts, AMO/CRM docs, related tests.

3. Productization / operator runtime slice:
   `src/mango_mvp/productization/current_runtime.py`, `operator_status.py`, product API/http/scheduler changes, operator/status scripts, related tests.

4. Channels slice:
   `src/mango_mvp/channels/demo.py`, `persistence.py`, `workspace.py`, channel demo script, channel tests and docs.

5. Insight/analyze improvements slice:
   changes in `knowledge_base.py`, `pilot_extraction.py`, `rop_validation_pack.py`, `services/analyze.py` and their tests.

6. Runtime pointer / tunnel helper slice:
   `stable_runtime/CANONICAL_EXPORT.txt`, `stable_runtime/start-amocrm-shared-db-tunnel.sh`, optional `stable_runtime/bring-up-amocrm-shared-db-tunnel.sh`.
   This slice needs extra review because it changes operational pointers and tunnel behavior.

7. Docs consolidation slice:
   current status, SaaS contours, runbooks and completion reports. Before commit, decide whether to keep all reports or compress old drafts into fewer current documents.

## Verification order

- Always run `git diff --check`.
- For each slice, run only focused tests for that slice first.
- Run full test suite only after slices are staged cleanly and runtime-heavy scripts are excluded.
- Do not push until branch history is readable and each commit has a clear scope.

## Pending manual approvals

- Physical deletion of `.git/objects/**/tmp_*` garbage.
- Moving or deleting old root-level spreadsheets/docs.
- Archiving or deleting old `stable_runtime` artifact directories.
- Any reset/checkout/rebase operation.
