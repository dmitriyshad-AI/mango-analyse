# Project Risk Audit Response Plan

Дата: 2026-05-09

Источник: `docs/PROJECT_RISK_AND_CLEANUP_AUDIT_2026-05-09.md`

## Назначение

Этот файл переводит большой аудит проекта в практический план действий. Цель:
не чинить все подряд, а разложить findings по владельцам, приоритетам, решениям,
проверкам и статусам.

Важно: часть findings относится к SaaS/productization ветке этого диалога, часть
относится к отдельному processing-диалогу, который чинит модули обработки
транскрибированных звонков, а часть является ручной операционной политикой.

## Позиция по локальным секретам и живой CRM

Для внутреннего этапа проекта принято рабочее решение: локальные secrets могут
лежать внутри рабочей папки, потому что проект используется с двух ноутбуков и
нужен быстрый запуск live-интеграций.

Это не считается блокером разработки, если соблюдаются правила:

1. Secrets не должны попадать в git.
2. Secrets не должны попадать в support bundle, zip, публичные логи и демо.
3. Secrets лучше хранить в малом количестве понятных ignored-файлов, а не
   размножать по snapshot-папкам.
4. Live CRM доступ допустим для ручного тестирования, но опасные writeback
   entrypoints должны иметь явный dry-run/confirmation/live флаг.
5. Перед внешним клиентским demo или передачей проекта третьим лицам secrets
   нужно вынести в отдельную управляемую схему.

## Приоритеты

- P0: риск случайной записи во внешнюю систему, потери данных или утечки при
  обычной разработке.
- P1: риск операционной путаницы, неправильного запуска или поломки runtime.
- P2: технический долг, который может замедлить развитие или усложнить поддержку.
- P3: cleanup/документация/улучшение качества, не блокирует ближайшую работу.

## Владельцы

- `this dialog`: SaaS/productization, Product API, safety gates, docs/runbook,
  script safety, CRM writeback guards.
- `processing dialog`: модули обработки транскрибированных звонков, transcript
  quality, ASR/R+A quality guardrails.
- `manual security`: ручные решения по secrets, live credentials, ротации,
  доступам и политике двух ноутбуков.
- `later cleanup`: большие архивы, перенос данных, исторические DB/audio/export
  folders, legacy script consolidation.

## Status legend

- `open`: нужно сделать.
- `in_progress_elsewhere`: уже идет в другом диалоге.
- `accepted_risk_internal`: риск осознанно принят для внутреннего режима.
- `deferred`: отложено до отдельного cleanup/product этапа.
- `done`: закрыто.
- `rejected`: не делаем, причина указана.

## Findings response table

| ID | Priority | Finding | Owner | Decision / solution | Verification | Status |
|---|---|---|---|---|---|---|
| RA-001 | P0 | Локальные secrets, `.env`, `*.env.private`, Codex auth snapshots, SSH key лежат внутри дерева проекта. | manual security | Для внутреннего этапа это принимается как рабочая схема. Улучшение: держать secrets в ignored-файлах, не коммитить, не включать в support bundles, сократить дубли snapshot secrets после подтверждения доступа. Ротацию делать перед внешним demo/client handoff. | `git ls-files` не должен показывать secrets; `git status --ignored` можно использовать для контроля; перед demo проверить support bundle на tokens. | accepted_risk_internal |
| RA-002 | P0 | Нужен live CRM доступ для быстрой проверки результатов. | manual security + this dialog | Live доступ сохранен, но live write entrypoints получили явное подтверждение. По умолчанию CLI writeback делает dry-run/preview, HTTP writeback отказывается без confirmation. | `tests/test_amo_writeback_guards.py`; ручной smoke только на тестовом/ограниченном batch. | done |
| RA-003 | P0 | `scripts/write_amo_ready_contacts.py` может выполнять live AMO writes без достаточного friction. | this dialog | Добавлен dry-run default и live gate: `--execute-live-write --live-confirmation WRITE_AMO_LIVE`. | Unit tests проверяют default dry-run и отказ без confirmation. | done |
| RA-004 | P0 | `scripts/write_recent_actionable_deals.py` может выполнять live AMO writeback. | this dialog | Добавлен dry-run default и live gate: `--execute-live-write --live-confirmation WRITE_AMO_LIVE`. | Unit tests проверяют default dry-run и отказ без confirmation. | done |
| RA-005 | P0 | FastAPI `/deals/writeback` и `apply_writeback` могут стать live write entrypoints. | this dialog | Добавлен HTTP policy gate: live write только при `execute_live_write=true` и `live_confirmation=WRITE_AMO_LIVE`. | `TestClient` tests на отказ без confirmation и live path с mock write. | done |
| RA-006 | P0 | Legacy `sync_amocrm` может писать notes/fields/tasks при env-флаге. | this dialog | Не удалять, но усилить документацию и tests: default disabled, live mode явно маркирован, в runbook указать как dangerous legacy. | Existing/added tests на disabled default и explicit env gate. | open |
| RA-007 | P1 | `scripts/` вырос до 98 файлов, каталог описывает около 49. | this dialog | Обновить `docs/CLI_AND_SCRIPTS_CATALOG_2026-05-07.md` или создать canonical `docs/CLI_AND_SCRIPTS_CATALOG.md` со всеми scripts. | Script inventory command сравнивает количество с каталогом. | open |
| RA-008 | P1 | Нет `SCRIPT_SAFETY_MATRIX.md`. | this dialog | Создана матрица безопасности скриптов: read-only, report writes, DB/runtime, network, CRM, ASR/R+A, approval required, recommended/default command. | `docs/SCRIPT_SAFETY_MATRIX.md`; targeted tests для live write gates. | done |
| RA-009 | P1 | README упоминает auto commit/push scripts как нормальный путь. | this dialog | Обновить README: auto commit/push scripts пометить historical/dangerous, нормальный workflow через ручной commit/push. | README review; grep `autocommit_push`. | open |
| RA-010 | P1 | Runbook говорит про safety, но live write scripts существуют отдельно. | this dialog | В runbook добавлен раздел safety matrix и правило live amoCRM confirmation. | Docs review; runbook links `docs/SCRIPT_SAFETY_MATRIX.md`. | done |
| RA-011 | P1 | Нет canonical AMO/Tallanto field mapping и writeback policy. | this dialog | Создать `docs/AMO_TALLANTO_FIELD_MAPPING_PROD.md`: поля, владелец, allowed writes, dry-run/live rollout, rollback. | Review by owner; tests for mapping helpers later. | open |
| RA-012 | P1 | Нет canonical `docs/ARCHITECTURE_CURRENT.md` и `docs/DATA_MODEL.md`. | this dialog | Создать текущую архитектуру и data model/status docs после принятия audit response, чтобы UI/API строились на понятных контрактах. | Docs review; links from development plan/runbook. | open |
| RA-013 | P1 | Старый `mango_office_download_recordings.py` опаснее нового guarded downloader. | this dialog | В safety matrix рекомендован guarded `mango_office_recording_capture_download.py`; старый downloader помечен `DANGEROUS_LEGACY`. Код не удалялся. | Docs review; optional CLI guard later. | done |
| RA-014 | P1 | `make test-smoke` может запускать `stable_runtime/rebuild_snapshot.sh`. | this dialog | Пересмотреть Makefile/test-smoke: либо исключить unsafe smoke, либо явно назвать команду dangerous/integration. | `make test-smoke` review; tests not run if unsafe. | open |
| RA-015 | P1 | `make audit` пишет в `stable_runtime/project_audit_*`. | this dialog | Решить: оставить как explicit local audit command или перенести default output в ignored `_local_archive`/`/private/tmp`. Документировать writes. | `--help`/docs; no surprise writes. | open |
| RA-016 | P1 | Transcript quality/processing problems найдены отдельно. | processing dialog | Оставить processing-диалогу. Этот диалог не меняет processing modules, пока второй диалог активно правит качество обработки транскрибированных звонков. | Дождаться plan/results от processing-диалога; не конфликтовать по файлам. | in_progress_elsewhere |
| RA-017 | P2 | Большие runtime/data folders внутри repo tree: audio, stable_runtime, Telegram exports, local archives. | later cleanup | Не удалять. Сначала сделать manifest, подтвердить владельца данных, затем план переноса в data volume/archive. | Manifest + size report; backup before move. | deferred |
| RA-018 | P2 | `.git` раздута до примерно 2.4G. | later cleanup | Не переписывать историю сейчас. Позже проверить large tracked history и необходимость git gc/filter-repo только после backup. | `git count-objects -vH`, `git-sizer` if available. | deferred |
| RA-019 | P2 | `__pycache__`, `.DS_Store`, `.pytest_cache`, local caches. | later cleanup | Можно чистить только после отдельного approval; добавить cleanup script dry-run first. | Dry-run list before delete. | deferred |
| RA-020 | P2 | Дубли/legacy families в scripts. | later cleanup + this dialog | Сначала safety matrix и catalog. Потом пометить canonical/legacy/special-case. Не удалять до владельца данных. | Catalog diff; no deletions. | open |
| RA-021 | P2 | Productization destructive primitives существуют, хотя guard rails есть. | this dialog | Добавить/проверить negative tests для path guards, restore/import/archive commands; усилить docs по dangerous functions. | `tests/test_productization_*.py`; targeted negative tests. | open |
| RA-022 | P2 | FastAPI routers/auth покрыты неравномерно. | this dialog | Добавить `TestClient` coverage для auth, deals/integrations/tallanto refusal/default modes, особенно writeback refusal. | New tests pass with mock env. | open |
| RA-023 | P2 | External clients `clients.amocrm`, `clients.ollama` без прямых HTTP/error tests. | this dialog or later cleanup | Добавить unit tests с fake session/monkeypatch. Не выполнять реальные network calls. | Tests verify timeout/error/token/cache behavior. | deferred |
| RA-024 | P2 | `utils.phone`, `llm_response_cache` имеют слабое прямое покрытие. | this dialog or later cleanup | Добавить недорогие unit tests. | New focused tests. | deferred |
| RA-025 | P2 | Insight generators требуют больше negative tests. | this dialog | Добавить tests на пустые/битые CSV/JSON, partial schema, missing columns. | Focused insight tests. | open |
| RA-026 | P2 | GUI почти без тестов, при этом запускает subprocess/workers. | this dialog or later cleanup | Не трогать UI глубоко сейчас. Позже добавить command-builder tests без Tk mainloop и docs warning для dangerous buttons. | Py_compile + future unit tests. | deferred |
| RA-027 | P3 | Много dated audit docs; часть superseded. | later cleanup | Не удалять. Позже создать docs/audits/YYYY-MM-DD или index с canonical/superseded статусом. | Docs index. | deferred |
| RA-028 | P3 | Root Excel/CSV/JSON exports и Postman collection лежат рядом с кодом. | later cleanup | Не удалять. Включить в data manifest и позже вынести в data/archive folder. | Manifest. | deferred |

## Recommended next work packages for this dialog

### WP-1. Script Safety Matrix - done

Цель: создать `docs/SCRIPT_SAFETY_MATRIX.md`.

Scope:

- inventory всех scripts;
- классификация по рискам;
- recommended safe command;
- dangerous/live command marker;
- owner stream: productization, processing, CRM, runtime, legacy.

Почему первым: это не конфликтует с processing-диалогом и сразу уменьшает риск
случайного запуска опасных команд.

### WP-2. AMO writeback guards - done for current entrypoints

Цель: закрыть RA-003, RA-004, RA-005. RA-006 остается отдельным follow-up,
потому что legacy `sync_amocrm` требует отдельного review.

Scope:

- dry-run by default;
- explicit live confirmation;
- focused tests;
- docs update.

Перед началом: проверить, не меняет ли processing-диалог эти же файлы.

### WP-3. Canonical writeback policy docs

Цель: закрыть RA-011.

Scope:

- AMO/Tallanto field mapping;
- allowed/disallowed writes;
- staged rollout;
- rollback;
- owner approval.

### WP-4. Router/auth safety tests

Цель: закрыть RA-022.

Scope:

- `TestClient` tests;
- no real CRM;
- mock env;
- refusal/default behavior.

## Manual security checklist

Эти пункты не блокируют внутреннюю разработку, но должны быть сделаны перед
внешним demo/client handoff:

1. Проверить, что secrets не tracked:
   `git ls-files | rg 'env|auth.json|id_ed25519|token|secret'`.
2. Сформировать список live credentials и где они лежат.
3. Решить, какие secrets можно оставить локально, а какие перенести.
4. Проверить, что support/demo bundles не включают secrets.
5. Ротировать токены, которые могли попасть в zip/snapshot/чужие машины.

## Coordination with processing dialog

Processing-диалог владеет:

- transcript quality;
- ASR/R+A correctness;
- modules that process transcribed calls;
- quality guardrails for transcript interpretation;
- related tests and docs:
  - `docs/TRANSCRIPT_QUALITY_ADVERSARIAL_AUDIT_2026-05-09.md`;
  - `docs/TRANSCRIPT_QUALITY_GUARDRAILS_AUDIT_2026-05-09.md`;
  - `docs/TRANSCRIPT_QUALITY_FIX_IMPLEMENTATION_PLAN_2026-05-09.md`;
  - `src/mango_mvp/quality/`;
  - `tests/test_transcript_quality_baseline.py`;
  - related `stable_runtime/transcript_quality_*` artifacts.

Этот диалог не должен редактировать эти файлы без явного согласования.

## Current response-plan status

Summary:

- accepted internal risk: RA-001;
- done in this pass: RA-002, RA-003, RA-004, RA-005, RA-008, RA-010, RA-013;
- in progress elsewhere: RA-016;
- ready for this dialog: RA-006, RA-007, RA-009, RA-011, RA-012, RA-014, RA-015, RA-021, RA-022, RA-025;
- deferred cleanup: RA-017 to RA-020, RA-023, RA-024, RA-026 to RA-028.

Recommended immediate next step: WP-3 `docs/AMO_TALLANTO_FIELD_MAPPING_PROD.md`.
