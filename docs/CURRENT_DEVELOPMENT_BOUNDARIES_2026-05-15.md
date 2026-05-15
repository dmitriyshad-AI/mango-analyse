# Current Development Boundaries

Дата: 2026-05-15
Блок: G - git-границы и рабочее состояние
Статус: зафиксировано перед реализацией Блока A

## 1. Назначение

Этот документ фиксирует текущее грязное состояние рабочей папки перед реализацией AMO snapshot/rollback.

Цель: не смешать Блок A с параллельными направлениями разработки и не потерять незакоммиченные изменения, которые уже есть в проекте.

## 2. Ветка и последние коммиты

Текущая ветка:

`codex/git-order-20260513`

Последние коммиты:

- `9b8d02126 Implement TZ X analysis refactor`
- `555b41c82 Implement TZ Z hygiene checks`
- `b4da574a3 Implement TZ Y sanitizer quality fixes`
- `c16e27203 Improve customer question catalog approval workflow`
- `29bdd69f6 Document question catalog improvement plan`

## 3. Общая картина git status

В рабочей папке есть незакоммиченные изменения и новые файлы из нескольких направлений:

- deal-aware;
- question catalog / ROP bot policy;
- customer timeline / channels / Telegram-history;
- mail archive;
- CRM text quality / AMO runtime shared;
- документы, ТЗ и audit packs;
- runtime-артефакты в `stable_runtime`;
- зависимости.

Это состояние не является мусором само по себе. Его нельзя чистить, откатывать или переносить без отдельного решения.

## 4. Что входит в Блок A

Блок A может менять только файлы, которые нужны для AMO pre-write snapshot, rollback, fake-тестов и audit pack.

Разрешенная зона Блока A:

- `scripts/write_deal_aware_amo_fields.py`
- `scripts/readback_deal_aware_amo_fields.py` только если нужна обратная совместимость/readback context
- новый `scripts/rollback_deal_aware_amo_fields.py`
- `src/mango_mvp/deal_aware/deal_writeback.py`
- новые helper-функции внутри `src/mango_mvp/deal_aware/`, если это лучше, чем раздувать скрипты
- `tests/test_deal_aware_amo_rollback.py`
- при необходимости существующие `tests/test_deal_aware_stage6_writeback.py`
- audit pack в `audits/_inbox/`

Осторожная зона:

- `src/mango_mvp/amocrm_runtime/amo_integration.py`
- `src/mango_mvp/amocrm_runtime/deals.py`
- `tests/test_amocrm_deals.py`

Эти файлы уже затронуты или являются общими для AMO. Их можно менять только если без этого нельзя выполнить Блок A. Перед изменением нужно прочитать текущий diff и сохранить совместимость.

## 5. Что вне Блока A

### 5.1. Mail archive

Не трогать:

- `docs/MAIL_ARCHIVE_RUNBOOK_2026-05-12.md`
- `scripts/mango_office_mail_archive.py`
- `scripts/mango_office_mail_full_60d_remaining.py`
- `src/mango_mvp/productization/mail_archive.py`
- `tests/test_mango_office_mail_full_60d_remaining.py`
- `tests/test_productization_mail_archive.py`

### 5.2. Question catalog / ROP bot policy

Не трогать:

- `src/mango_mvp/question_catalog/`
- `scripts/apply_rop_questionnaire_to_catalog_v2.py`
- `scripts/build_question_answer_quality_review_pack.py`
- `scripts/build_question_catalog_stratified_calibration_v2.py`
- `scripts/build_rop_blocker_markup_pack.py`
- `scripts/build_rop_bot_policy_questionnaire.py`
- `scripts/run_question_catalog_codex_ab_v2.py`
- `scripts/run_question_catalog_llm_calibration_v2.py`
- `tests/test_question_catalog_*`
- `tests/test_theme_assigner_llm_v2.py`
- `tests/test_parameters_registry_v2.py`

### 5.3. Channels / Telegram-history

Не трогать:

- `src/mango_mvp/channels/__init__.py`
- `src/mango_mvp/channels/conversation_orchestrator.py`
- `tests/test_channels_conversation_orchestrator.py`

### 5.4. Dependencies

Не трогать без необходимости:

- `pyproject.toml`
- `requirements.txt`
- `uv.lock`

Блок A должен обходиться текущими зависимостями стандартной библиотеки и уже установленными пакетами.

### 5.5. Runtime artifacts

Не коммитить и не менять:

- `stable_runtime/deal_aware_*`
- `stable_runtime/tenant_text_normalizer_gate_*`
- любые DB/audio/transcripts в `stable_runtime`

Эти папки можно читать только при необходимости аудита, но Блок A должен тестироваться на fake fixtures и `tmp_path`.

### 5.6. Старые ТЗ и документы

Не переносить и не удалять в рамках Блока A:

- `Mango_Analyse_TZ_*`
- старые audit docs;
- roadmap UI;
- ROP review docs.

## 6. Runtime-артефакты, которые не входят в git

На момент фиксации есть много untracked папок:

- `stable_runtime/deal_aware_stage1_snapshot_*`
- `stable_runtime/deal_aware_stage2_attribution_*`
- `stable_runtime/deal_aware_stage3_deal_state_*`
- `stable_runtime/deal_aware_stage4_preview_*`
- `stable_runtime/deal_aware_stage5_quality_gate_*`
- `stable_runtime/deal_aware_stage6_writeback_preflight_*`
- `stable_runtime/deal_aware_stage709_*`
- `stable_runtime/deal_aware_writeback_batch1_*`

Они отражают историю разработки deal-aware, но не являются кодом. В рамках Блока A их не удалять и не добавлять в коммит.

## 7. Правило для коммита Блока A

В коммит Блока A могут войти только:

- проектные инструкции и документы порядка, если они еще не зафиксированы;
- код snapshot/rollback;
- тесты snapshot/rollback;
- audit pack Блока A;
- минимальные связанные документы.

В коммит Блока A не должны попасть:

- `stable_runtime/*`;
- mail archive изменения;
- question catalog изменения;
- Telegram/channel изменения;
- unrelated docs;
- root-level старые ТЗ, если они не нужны для текущего коммита;
- зависимости, если Блок A не требует их изменения.

## 8. Следующий шаг

Переходить к Блоку A:

1. Изучить текущий `scripts/write_deal_aware_amo_fields.py`.
2. Изучить `src/mango_mvp/deal_aware/deal_writeback.py`.
3. Найти или добавить безопасные helper-функции для snapshot/rollback.
4. Написать fake-тесты без live AMO.
5. Не запускать live-write.
