Ветка: codex/next-step-proactivity-port
Зоны: src/mango_mvp/customer_timeline/bot_safe_summary.py, src/mango_mvp/customer_timeline/bot_safe_runtime_context.py, src/mango_mvp/customer_timeline/approval_workspace.py, src/mango_mvp/channels/subscription_llm_parts/__init__.py, src/mango_mvp/channels/subscription_llm_parts/direct_path.py, src/mango_mvp/channels/subscription_llm_parts/post_layers.py, src/mango_mvp/channels/subscription_llm_parts/provider.py, tests/test_customer_timeline_bot_safe_summary.py, tests/test_bot_safe_runtime_context.py, tests/test_bot_safe_memory_step_guard.py, tests/test_subscription_llm_draft_provider.py, tests/test_bot_safe_direct_path_context.py, tests/test_customer_timeline_approval_workspace.py, tests/test_customer_timeline_read_api.py, tests/test_adr003_regex_understanding_moratorium.py, tests/fixtures/adr003_runtime_channel_regex_snapshot.json, tests/fixtures/adr003_direct_path_text_patterns_snapshot.json, docs/ADR003_REGEX_UNDERSTANDING_MORATORIUM.md, docs/worktrees_registry.md, tasks/_running/2026-07-10_TZ_next_step_proaktivnost_phase1_port.md, audits/_inbox/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest --collect-only -q tests/test_customer_timeline_bot_safe_summary.py tests/test_bot_safe_runtime_context.py tests/test_bot_safe_memory_step_guard.py tests/test_subscription_llm_draft_provider.py tests/test_bot_safe_direct_path_context.py tests/test_customer_timeline_approval_workspace.py tests/test_customer_timeline_read_api.py
Семантический-аудит: да

# Phase 1: proactivity next-step safety port

Источник: `/Users/dmitrijfabarisov/Claude Projects/Foton/2026-07-08_TZ_Codex_proaktivnost_FINAL_posle_audita.md`.

Фаза 0 уже выполнена в `Foton/2026-07-10_PHASE0_proaktivnost_next_step_sverka_D1.md`.

Выполнить только безопасный Phase 1 slice:

1. Точечно портировать next-step семью из `codex/email-pipeline-restore` / `edf58b68`, не вливая EPR целиком.
2. Убрать прайминг пустого next_step в `bot_safe_summary.py`: не рендерить `Следующий шаг: активный шаг не найден`.
3. Добавить bot-path strip неподтверждённого next_step на чтении, не трогая manager-facing читалки.
4. Добавить мягко-рамочный next-step output guard для фраз вида `Следующий шаг — уточнить класс...` без active next_step.
5. Встроить guard в canonical direct-path chain без потери semantic-frame/read-trace цепочки.
6. Перенести/адаптировать регрессионные тесты из EPR; не запускать live/AMO/Wappi/M1.

## Phase 1b follow-up

Источник: `/Users/dmitrijfabarisov/Claude Projects/Foton/2026-07-10_REGRADE_phase1_port_next_step_15fb1c9b.md` и `/Users/dmitrijfabarisov/Claude Projects/Foton/2026-07-10_TZ_Codex_PORT_next_step_fix_v_kanon.md`.

Довести порт без включения памяти:

1. Расширить next-step frame regex на 1-2 вставных слова после `шаг`, закрыть `Следующий шаг сейчас — уточнить класс ребёнка`; не трогать прошлое `Следующим шагом была оплата`.
2. Убрать самостоятельный no-memory post-layer путь из direct-path chain; смысловая защита должна идти через канонический bot-safe memory step guard и только при review/empty next_step.
3. Завести отдельный default-OFF флаг guard'а, не добавлять его в профиль.
4. Добавить try/except fail-open для guard, женский голос в rewrites, `next_step_status` в approval workspace и read-api regression assert.
5. Проверить, что безопасный текущий `оплата по ссылке` не переписывается, а конкретный неподтверждённый шаг из памяти заминается.

Не делать в этом заходе:

- whole merge EPR;
- Фаза 2 / включение памяти live;
- P2 slot/recap provenance gate, кроме если он нужен как минимальная регрессия для next-step;
- форматный трек A1;
- новые профильные default-ON флаги.

## Phase 1c: combo overclaim guard, direct-path format, AMO check

Источник: `/Users/dmitrijfabarisov/Claude Projects/Foton/2026-07-10_TZ_Codex_COMBO_overclaim_format_amocheck.md` v2.

Последовательно после Phase 1b:

1. Read-only проверить фактический механизм запуска Wappi/AMO draft-loop: cron, launchd, LaunchAgents, screen, PID/cwd/HEAD и безопасные профильные флаги; live-процесс не менять.
2. Добавить default-OFF direct-path guard против неподтверждённой пресуппозиции о подходящей группе/формате в сырых формулировках `memory_rich_07` и `memory_rich_20`. Переписывать только целевое предложение до semantic verifier и authoritative gate, маршрут не менять.
3. Не затрагивать `memory_rich_09`, менеджерские оговорки, подтверждённые client-safe факты, P0/manager_only, бренды и соседние числа.
4. Добавить отдельную default-OFF инструкцию форматирования непосредственно в живой `_build_direct_path_prompt`: короткие абзацы для многофактного ответа, максимум один эмодзи, ноль эмодзи на серьёзных темах; без жирного и без изменения монолитного prompt builder.
5. После юнитов и безопасного pytest выполнить отдельный semantic review. LLM micro/full/M1 не запускать без следующего этапа сырьевой приёмки.

### Phase 1c: принятые решения и фактическая приёмка

1. Локальная пара держит bot-safe CRM memory и Phase 1b guard одинаково включёнными в B/ON; различаются только Phase 1c overclaim/format флаги. Иначе эффект памяти смешался бы с эффектом новых механизмов.
2. Первая незавершённая попытка с `client-mode=scripted` на исходных служебных `behaviors` признана `measurement_bug`: клиентом уходило буквальное `Спроси:`. Корректная пара использует отдельные точные `scripted_behaviors`, один прогон каждой ноги без ретраев.
3. Сырьё выявило runtime-разрыв: grade/subject могут находиться только в `result.metadata.direct_path.semantic_frame`, а не в old `known_slots`. Guard теперь определяет candidate по готовому предложению, exact-fact exception читает scope из frame только при confidence >= 0.90; 0.89, active p0_latch и risk_flags закрыты регрессиями.
4. Локальный итог: ON 0 FAIL, B 1 judge measurement bug на дословно одинаковом ответе; rich-format 12.2% -> 66.7%; p0_route_lost=0; unverified numbers=0; dangerous route flips=0. Естественных overclaim-replacement traces не было, поэтому Part A остаётся default-OFF до M1-фокуса.
5. M1-фокус использует 20 обезличенных synthetic/canonical персон с `initial_history_lines`, не реальную 4.5-GB CRM DB. Это проверяет A/B переносимо и не передаёт Customer Timeline. Runner ON-first, чистит все унаследованные `TELEGRAM_*`/`CUSTOMER_TIMELINE_*`/`MANGO_TELEGRAM_*`, затем задаёт явный контракт ног.
6. Read-only AMO/Wappi аудит: launchd `KeepAlive=true`, активный процесс работает из EPR cwd, точный загруженный SHA не доказан. Live-процесс не менялся; подробности в `Foton/2026-07-10_REPORT_C_Wappi_AMO_autorestart_D1.md`.

## Phase 1d: Part A hardening and authorized live swap

Источник: `/Users/dmitrijfabarisov/Claude Projects/Foton/2026-07-10_TZ_Codex_SWAP_i_vklyuchenie_featur.md`.

Владелец явно разрешил остановить старый Wappi/AMO draft-loop, удалить только подтвержденные пустые stale `index.lock`, переключить `launchd` на чистый потомок `7da115e5` и включить принятые флаги. Выполнить:

1. Part A: сопоставлять exact fact с классом/предметом каждого конкретного candidate-предложения, а не со слотом диалога; конфликт `slot/fact=9` против предложения `7 класс` обязан заминаться.
2. Не переписывать честную явную передачу проверки менеджеру; добавить trace-счетчики без клиентского текста и сохранить fail-open.
3. Part A и Part B коммитятся default-OFF; Part A не включать live до отдельного M1 micro.
4. После тестов и независимого аудита создать чистый candidate worktree, выполнить `--once --dry-run`, затем переключить `com.mango.wappi-draft-loop` с готовым откатом.
5. Live включить только: Phase 1b memory-step guard, bot-safe CRM context, timeline memory prompt и Part B format guidance. При любом расхождении cwd/SHA/env, более чем одном PID или P0/brand-регрессии немедленно откатить старый EPR plist/script.
6. Собрать, но не запускать самостоятельно, единый M1-пакет: подтверждение включенных механизмов и отдельная приемка Part A до его включения.
