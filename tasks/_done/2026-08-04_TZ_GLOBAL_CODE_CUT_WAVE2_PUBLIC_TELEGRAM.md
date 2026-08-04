> DONE 2026-08-04 22:48 | ветка main | codex

> TAKE 2026-08-04 22:30 | ветка main | codex

Ветка: main
Зоны: CLAUDE.md, docs/RUNBOOK.md, scripts/run_telegram_public_pilot_bots.py, scripts/restart_telegram_public_pilot_bots.sh, scripts/check_public_bot_live.py, scripts/build_telegram_public_pilot_feedback_report.py, scripts/run_telegram_night_shadow_replay.py, scripts/adr003_deploy_swap_dry_run.py, scripts/project_now.py, scripts/skills/live_truth.py, scripts/make_audit_pack.py, src/mango_mvp/channels/night_funnel_shadow.py, tests/test_telegram_public_pilot_bots.py, tests/test_check_public_bot_live.py, tests/test_telegram_public_pilot_feedback_report.py, tests/test_adr003_deploy_swap_dry_run.py, tests/test_exact_runtime_dedup_contract.py, tests/test_kb_r4_1_owner_gap_answers.py, tests/test_kb_v67_staging_content.py, tests/test_skills_top5_tools.py, tests/test_audit_pack_pii.py, tasks/, audits/_inbox/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_draft_loop.py tests/test_run_amo_wappi_draft_loop.py tests/test_wappi_draft_loop_ops.py tests/test_bot_safe_runtime_context.py tests/test_kb_r4_1_owner_gap_answers.py tests/test_kb_v67_staging_content.py tests/test_exact_runtime_dedup_contract.py tests/test_adr003_regex_understanding_moratorium.py tests/test_graphify_structural.py tests/test_single_owner_registry.py tests/test_project_now.py tests/test_skills_top5_tools.py tests/test_audit_pack_pii.py
Семантический-аудит: да

# Глобальная резка кода, волна 2: удалить публичный Telegram-контур

## Образ результата и бизнес-польза

В актуальном `main` остаётся один клиентский рабочий путь: `Wappi -> черновик AMO
-> менеджер`. Публичный Telegram-бот, которым никто не пользуется, нельзя
запустить из канонического кода или принять за целевую архитектуру. Старые
worktree пока сохраняются, поэтому read-only страж обязан замечать их запрещённый
процесс. Общий provider, Customer Timeline, память, P0 и выходные полы не меняются.

Результат полезен бизнесу тем, что разработка и тесты концентрируются на канале,
где менеджер реально получает черновик. Удаление не добавляет новых слоёв,
флагов, зависимостей или вызовов модели.

## Основание

- D-105 исключает публичный Telegram из целевой архитектуры и запрещает его запуск;
- текущий live-путь один: `scripts/run_amo_wappi_draft_loop.py` с
  `sends_client_replies=False`;
- Graphify на HEAD `4a088d47` и raw-поиск не нашли production caller публичного
  runner/night-shadow вне их собственного контура, прямых тестов и старых документов;
- перед удалением обязательна read-only проверка процессов и скрытых запускателей.

## Удалить целиком

1. Публичный runner и его restart/live-check/feedback-report.
2. Ночную Telegram-тень и replay: они обслуживают только исключённый публичный путь.
3. ADR003 deploy-swap helper и прямой тест: helper целиком печатает команды запуска
   исключённого публичного бота и больше не является допустимым deploy-путём.
4. Все прямые тесты удаляемых модулей.

## Точечные правки

1. Убрать публичный runner из `project_now.py`. В `live_truth.py` оставить его
   отдельным запрещённым маркером: старый worktree не должен незаметно запустить
   удалённый контур. В `make_audit_pack.py` оставить путь как маркер, требующий
   смыслового аудита при попытке вернуть удалённый runner.
2. В актуальной инструкции `CLAUDE.md` удалить устаревшую строку о рестарте
   публичного бота; решение D-105 оставить.
3. В KB-тестах заменить чтение публичного/night-shadow runner на проверку
   канонического Wappi runner. Не ослаблять проверяемые требования KB.
4. Удалить test-only контракт writer-алиасов night-shadow из
   `test_exact_runtime_dedup_contract.py`; остальные владельцы helper-контрактов
   оставить.
5. В актуальном `RUNBOOK.md` заменить условное разрешение будущего запуска
   публичного бота на безусловный запрет D-105 и проверку запрещённого процесса.

## Не делать

- не удалять общий `SubscriptionLlmDraftProvider`, direct path, P0 или output floor;
- не удалять Telegram/Wappi ingestion из Customer Timeline;
- не трогать manager draft/store/feedback модули в этой волне;
- не переписывать исторические решения и runbook как будто публичного контура не было;
- не создавать заглушку, compatibility wrapper или feature flag вместо удалённого кода;
- не запускать live, не отправлять сообщения и не писать в AMO/Tallanto/CRM.

## Ломающие проверки

1. До удаления `live_truth.py --no-write` и raw `ps` не находят процесс публичного бота.
2. После удаления `rg` не находит runtime/import/subprocess/launchd-ссылок на
   удалённые файлы вне исторических документов и audit-backlog.
3. Wappi runner импортируется и его целевой набор тестов остаётся зелёным.
4. KB-тесты проверяют тот же смысл на Wappi runner, а не просто теряют assertions.
5. Полный `pytest --collect-only` проходит; полный pytest не получает новых падений
   относительно baseline на том же HEAD.
6. Архитектор, ломатель и бизнес-аудитор независимо подтверждают отсутствие
   уникальной функции, нужной Wappi -> AMO draft.

## Приёмка

- удалено не менее 6 000 строк прямого кода и тестов;
- новых рабочих файлов, функций, флагов, зависимостей и LLM-вызовов — 0;
- `formal_pass`: collect, целевой набор и сравнение полного pytest;
- `semantic_pass`: Wappi-черновик сохраняет память, KB, P0 и запрет client-send;
- `breaker_pass`: скрытых entry point и единственных общих helpers нет;
- один audit pack, один коммит, push в `origin/main` и `yandex/main`.

## СТОП

Остановить конкретное удаление, если найден фактический процесс, launchd/cron,
production caller или уникальная функция текущего Wappi-пути. Остальной независимо
доказанный контур удалить той же волной.
