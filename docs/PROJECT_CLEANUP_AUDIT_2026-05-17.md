# Аудит порядка в проекте от 2026-05-17

Статус: выполнена безопасная уборка без удаления файлов.

## Что было сделано

- Проверен `git status`.
- Проверены размеры и назначение локальных незакоммиченных папок.
- Полезные скрипты сборки базы знаний зафиксированы отдельным коммитом.
- Добавлены правила `.gitignore`, чтобы локальные runtime, аудио, кэши и промежуточные сборки не мешали Git.
- Удалений, переносов, ASR, Resolve+Analyze, AMO/Tallanto/CRM write не было.

## Git-граница

Рабочая ветка: `codex/git-order-20260513`.

Коммиты, сделанные в рамках уборки:

- `cd3887bb8 Add KC knowledge base builders`
- cleanup-коммит с этим отчётом и `.gitignore` создаётся отдельно после проверки.

## Группа 1. Оставлено и зафиксировано в Git

Эти файлы являются полезным кодом воспроизводимости базы знаний и не являются мусором:

- `scripts/build_full_kc_knowledge_base.py`
- `scripts/build_kc_final_release.py`
- `scripts/extract_kc_google_doc_facts.py`
- `tests/test_build_full_kc_knowledge_base.py`
- `tests/test_build_kc_final_release.py`
- `tests/test_extract_kc_google_doc_facts.py`

Проверка:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q \
  tests/test_build_full_kc_knowledge_base.py \
  tests/test_build_kc_final_release.py \
  tests/test_extract_kc_google_doc_facts.py
```

Результат: `8 passed`.

## Группа 2. Оставлено локально и добавлено в ignore

Эти папки нужны как локальный след работы, входные данные или промежуточные результаты, но не должны попадать в Git.

### База знаний

- `Claude Mango_Bot_Knowledge_Base_FINAL_2026-05-17/` — исходный пакет Клода, около 6.3M.
- `product_data/knowledge_base/full_kb_20260517_v1/` — промежуточная полная база, около 16M.
- `product_data/knowledge_base/google_drive_structured_facts_20260517_v1/` — кандидаты фактов из Google Docs, около 448K.
- `product_data/knowledge_base/kb_release_20260517_v1/` — старый v1 release, около 87M.
- `product_data/knowledge_base/kb_release_20260517_v1_agent_pack/` — вход v1 для сборщика v2, около 87M.
- `product_data/knowledge_base/kb_release_20260517_v2/stage6_codex_smoke/` — подробный LLM smoke-прогон, около 380K.
- `product_data/knowledge_base/kb_release_20260517_v2/stage6_fake_smoke/` — подробный fake smoke-прогон, около 288K.

Причина: v2 уже зафиксирована в Git, а подробные smoke CSV/XLSX/LLM cache могут содержать клиентские тексты. Handoff-папка уже содержит безопасные summary.

### Mango/API/audio

- `product_data/mango_api_reconcile_20260517_v1/` — сверка Mango API, около 22M.
- `product_data/mango_audio_update_20260517_v1/` — отчёт audio update, около 28K.
- `product_data/mango_missing_1372_asr_only_20260517_v1/` — 1372 mp3 + ASR-only пакет, около 891M.
- `stable_runtime/ra_pending_mango_api_20260517_v1/` — pending R+A runtime, около 52M.

Причина: это runtime/аудио/промежуточные данные, а не исходный код. Удалять рано.

## Группа 3. Кандидаты на архив или удаление только после подтверждения

Не удалять автоматически.

Самые безопасные кандидаты после отдельного решения:

- `product_data/knowledge_base/kb_release_20260517_v2/stage6_fake_smoke/`
- `product_data/knowledge_base/kb_release_20260517_v2/stage6_codex_smoke/`
- `product_data/knowledge_base/google_drive_structured_facts_20260517_v1/`
- `product_data/knowledge_base/full_kb_20260517_v1/`

Крупные кандидаты, которые нельзя трогать до следующего runtime-решения:

- `product_data/mango_missing_1372_asr_only_20260517_v1/`
- `stable_runtime/ra_pending_mango_api_20260517_v1/`
- `product_data/knowledge_base/kb_release_20260517_v1/`
- `product_data/knowledge_base/kb_release_20260517_v1_agent_pack/`

## Группа 4. Что не трогать сейчас

- `stable_runtime/CURRENT_RUNTIME.json`
- `stable_runtime/CANONICAL_EXPORT.txt`
- `stable_runtime/canonical_master_*`
- аудио и transcripts, пока не завершён новый rebuild поверх актуального runtime
- v1 knowledge-base папки, пока v2 provenance окончательно не нормализован

## Важное наблюдение

`stable_runtime/ra_pending_mango_api_20260517_v1/` не является завершённым canonical runtime. Фактическое состояние по read-only проверке: ASR готов, Resolve/Analyze почти завершены, но остаются manual/pending строки и sync pending. Текущий `CURRENT_RUNTIME` указывает на слой `20260516_after_mango_update_v1`, и это корректно.

## Следующие решения Дмитрия

1. Когда закрывать 17 manual/pending строк в `ra_pending_mango_api_20260517_v1`.
2. Когда делать rebuild нового canonical/current runtime поверх Mango 20260517.
3. Можно ли после rebuild удалить или архивировать тяжёлый пакет `product_data/mango_missing_1372_asr_only_20260517_v1/`.
4. Нужно ли сохранить v1 knowledge-base источники в отдельный архив для воспроизводимости или достаточно v2 snapshot + handoff.

## Рекомендация

На моём месте я бы сейчас ничего не удалял. Я бы продолжал разработку от чистого Git-статуса, а удаление тяжёлых папок сделал бы отдельным шагом после завершения Mango 20260517 rebuild и ревью базы знаний v2.
