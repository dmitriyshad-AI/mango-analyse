# SaaS/Productization Development Plan

Дата: 2026-05-09

## Назначение документа

Этот документ фиксирует дальнейший план развития Mango Analyse после появления
первого productization baseline. Он отделен от параллельного audit-диалога:
аудит ищет риски и мусор, а этот план описывает, как развивать продукт дальше.

Правило синхронизации: аудит-диалог может читать проект и писать свой audit-файл,
но не должен править проверяемые файлы. Разработка в этой ветке может идти
параллельно, если перед каждой правкой проверяется `git status`, а предложения
аудита применяются только после отдельного инженерного решения.

## Текущее состояние

Ветка: `codex/saas-productization-baseline`.

Сейчас в проекте появились семь продуктовых слоев:

1. Mango capture/productization layer.
2. Изолированная product appliance SQLite DB.
3. Read-only Product API и HTTP layer.
4. Scheduler/appliance loop/ASR safety gates.
5. Sales insight/knowledge layer.
6. Opt-in agent runtime preview API.
7. Ops/docs/git housekeeping layer.

Ключевой принцип остается прежним:

- не менять `stable_runtime` DB/audio/transcripts без отдельного решения;
- не запускать ASR/R+A из этой ветки;
- не писать в AMO/Tallanto/CRM;
- все опасные действия сначала должны иметь dry-run, preview, gate и audit trail;
- текущий pipeline должен оставаться рабочим, пока новый product appliance слой
  не станет достаточно зрелым.

## Целевая картина продукта

Продукт должен пройти три практических состояния.

### Состояние A. Внутренний автономный appliance

Сервис работает на нашем ноутбуке или сервере, сам забирает данные из Mango,
показывает очереди, состояние обработки, инсайты и управленческие отчеты.
CRM-записи остаются dry-run или staged-writeback.

Критерий готовности: руководитель может открыть локальный UI/API и понять,
какие звонки пришли, что с ними произошло, где риски и какие действия предлагаются.

### Состояние B. Демо-продукт для клиентов

Сервис можно показать клиенту без доступа к приватному runtime. Есть демо-tenant,
обезличенные данные, стабильный UI, понятный сценарий демонстрации и безопасный
read-only режим.

Критерий готовности: клиент видит не набор скриптов, а продуктовый workflow:
подключение телефонии, список звонков, аналитика, очередь РОПа, knowledge base,
writeback preview.

### Состояние C. Ограниченный client-hosted SaaS/appliance

Сервис можно поставить нескольким компаниям на отдельный ноутбук/сервер под их
управлением. Multi-client сначала означает повторяемую установку и изоляцию
конфигураций, а не обязательно один общий облачный multi-tenant кластер.

Критерий готовности: второй клиент подключается конфигурацией и адаптерами,
а не форком кода.

## Главные архитектурные решения

### 1. SQLite пока остается

SQLite остается правильным решением для текущего appliance-этапа:

- проще установка на клиентский ноутбук/сервер;
- легче backup/restore;
- меньше администрирования;
- соответствует текущей стратегии "сначала внутренний автономный appliance".

PostgreSQL не отменяется, но переносится на этап, где появятся реальные причины:
параллельные writers, несколько пользователей, много tenant-ов, очереди с высокой
конкурентностью, централизованный хостинг.

### 2. Product DB отделена от runtime DB

Product appliance DB должна оставаться отдельной от `stable_runtime`. Новый сервис
читает, нормализует и показывает данные, но не должен незаметно менять историческую
рабочую БД.

### 3. Product API является контрактом для UI

UI v1 должен строиться поверх Product API, а не напрямую дергать batch scripts.
Это делает интерфейс стабильнее и позволяет позже заменить storage/scheduler без
переписывания UI.

### 4. Все writeback-действия идут через preview и gates

CRM/AMO/Tallanto writeback нельзя делать "просто кнопкой". Нужна цепочка:

```text
proposal -> preview diff -> policy gate -> staged batch -> audit log -> rollback plan
```

### 5. Agent runtime остается opt-in

Agent runtime preview API не должен включаться по умолчанию. Он нужен для будущей
автоматизации, но в текущем продукте должен оставаться безопасным экспериментальным
слоем.

## Roadmap по фазам

### Фаза 1. Baseline hardening и приемка аудита

Цель: убедиться, что текущий baseline не несет скрытых рисков.

Work packages:

1. Дождаться файла параллельного аудита:
   `docs/PROJECT_RISK_AND_CLEANUP_AUDIT_2026-05-09.md`.
2. Разделить findings на группы:
   - P0: секреты, риск потери данных, случайные CRM writes;
   - P1: риск сломать pipeline/runtime;
   - P2: тестовые пробелы и опасные defaults;
   - P3: cleanup/docs/naming.
3. Создать отдельный response-plan:
   `docs/PROJECT_RISK_AUDIT_RESPONSE_PLAN_2026-05-09.md`.
4. Исправлять только согласованные findings маленькими коммитами.
5. После каждого исправления запускать целевые тесты.

Acceptance criteria:

- рабочая копия чистая;
- нет новых секретов в git;
- runtime artifacts игнорируются;
- productization tests проходят;
- audit findings имеют владельца: fix now, defer, reject with reason.

Команды проверки:

```bash
git status --short --branch
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_productization_*.py
```

### Фаза 2. Supervised local Product API service

Цель: превратить read-only Product API из скрипта в локальный сервис, который
можно стабильно запускать.

Work packages:

1. Добавить service profile для Product API:
   - host/port;
   - product root;
   - product DB path;
   - log path;
   - health endpoint.
2. Добавить supervisor dry-run manifest:
   - какая команда будет запущена;
   - какие env vars нужны;
   - какие paths будут читаться;
   - куда пишутся logs.
3. Добавить health/readiness smoke command.
4. Добавить тесты, что сервис read-only:
   - GET routes работают;
   - POST/PUT/PATCH/DELETE запрещены;
   - invalid query дает JSON error.
5. Добавить docs runbook для запуска сервиса.

Acceptance criteria:

- сервис стартует локально на `127.0.0.1`;
- `/dashboard/summary` отвечает;
- mutation routes заблокированы;
- запуск не пишет в runtime DB и CRM.

Безопасный тест:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_product_api_http.py readiness
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_product_api_http.py serve --host 127.0.0.1 --port 8765
```

### Фаза 3. UI v1 поверх Product API

Цель: сделать первый настоящий интерфейс продукта, а не панель запуска скриптов.

Минимальные экраны:

1. Dashboard:
   - сколько звонков в product DB;
   - сколько новых capture candidates;
   - какие gates заблокированы;
   - последние scheduler runs.
2. Capture:
   - последние Mango events;
   - duplicate/no-recording/new statuses;
   - raw payload link/status.
3. Processing Queue:
   - что готово к обработке;
   - что заблокировано;
   - почему заблокировано.
4. ASR Gates:
   - approval status;
   - sandbox readiness;
   - human approval missing/present.
5. Knowledge:
   - playbook summaries;
   - ROP validation status;
   - sales moments.
6. Settings:
   - tenant;
   - Mango adapter;
   - CRM adapter;
   - safety policies.

Implementation rule:

- UI читает только Product API;
- UI не запускает ASR/R+A;
- UI не пишет в AMO/Tallanto;
- любые кнопки действия сначала создают preview, а не выполняют действие.

Acceptance criteria:

- UI открывается без терминальных знаний;
- работает на demo/local product DB;
- browser/smoke verification подтверждает, что основные экраны не пустые;
- нет прямых вызовов batch scripts из UI.

### Фаза 4. Mango capture from shadow to controlled ingest

Цель: перейти от shadow-poll к контролируемому ingest новых Mango звонков.

Work packages:

1. Зафиксировать Mango API credentials policy:
   - где хранится API key/salt;
   - как не коммитить secrets;
   - как проверить подключение.
2. Улучшить Mango shadow poll:
   - стабильные provider ids;
   - rate limit handling;
   - retry/backoff;
   - raw payload archive;
   - clear report.
3. Добавить controlled recording metadata ingest:
   - не скачивать аудио автоматически;
   - сначала сохранять только recording refs;
   - отличать no-recording от delayed-recording.
4. Сделать capture inbox:
   - новые звонки;
   - duplicates;
   - missing recording;
   - blocked by policy.
5. После audit approval добавить controlled download в отдельную quarantine папку.

Acceptance criteria:

- shadow poll стабильно показывает новые звонки;
- повторный запуск не создает дубликаты;
- no-recording понятно объяснен;
- скачивание аудио не происходит без явного command/gate;
- runtime DB не меняется.

### Фаза 5. Processing orchestration bridge

Цель: связать product appliance capture с существующим pipeline без поломки
текущей обработки.

Work packages:

1. Описать единый lifecycle звонка:
   - captured;
   - recording_available;
   - quarantined;
   - approved_for_asr;
   - asr_done;
   - resolve_done/skipped;
   - analyze_done;
   - insight_ready;
   - writeback_preview_ready.
2. Сделать bridge dry-run:
   - какие звонки могли бы уйти в ASR;
   - какие заблокированы;
   - почему.
3. Добавить idempotency:
   - один provider_call_id не должен попасть в pipeline дважды;
   - один recording_id не должен скачиваться дважды.
4. Добавить worker handoff manifest:
   - входы;
   - ожидаемые выходы;
   - checksum;
   - rollback/cleanup.
5. Не включать auto-trigger, пока ASR approval gates не будут приняты.

Acceptance criteria:

- bridge report показывает candidates и blockers;
- не запускает ASR/R+A;
- не пишет в stable runtime DB;
- покрыт negative tests.

### Фаза 6. Controlled CRM writeback

Цель: подготовить безопасную запись в AMO/CRM, но не включать ее без staged rollout.

Work packages:

1. Финализировать AMO field mapping:
   - contact fields;
   - lead fields;
   - AI summary;
   - next step;
   - priority;
   - source metadata.
2. Добавить writeback preview diff:
   - что было;
   - что хотим записать;
   - почему;
   - confidence;
   - blockers.
3. Добавить staged queue:
   - batch 10;
   - batch 50;
   - batch 300;
   - full only after approval.
4. Добавить policy gates:
   - no close lead automatically;
   - no delete/merge contacts;
   - no direct client messages;
   - L3/L4 require approval or are forbidden.
5. Добавить audit log и rollback plan.

Acceptance criteria:

- можно показать AMO writeback preview без записи;
- есть список blockers;
- есть human approval path;
- live write path выключен по умолчанию.

### Фаза 7. Knowledge and ROP productization

Цель: превратить insight-layer в понятную продуктовую ценность для РОПа и
руководителя продаж.

Work packages:

1. Стабилизировать extraction:
   - вопросы клиента;
   - ответы менеджера;
   - возражения;
   - следующий шаг;
   - признаки buying intent.
2. Улучшить outcome linking:
   - AMO;
   - Tallanto;
   - действующий клиент;
   - повторная сделка;
   - отказ;
   - в работе.
3. Ввести rubric scoring:
   - полнота ответа;
   - работа с возражением;
   - конкретность следующего шага;
   - корректность;
   - коммерческая сила.
4. Сделать ROP review workflow:
   - top examples;
   - risky examples;
   - approve/reject knowledge item;
   - comments.
5. Подготовить playbook:
   - лучшие ответы;
   - плохие паттерны;
   - рекомендации по follow-up;
   - сегменты клиентов.

Acceptance criteria:

- РОП может открыть пакет и проверить выводы;
- рекомендации отделены от сырых наблюдений;
- LLM score не выдается за доказанную истину;
- есть экспорт для ручной валидации.

### Фаза 8. Packaging for client-hosted appliance

Цель: сделать установку повторяемой на отдельном ноутбуке или сервере клиента.

Work packages:

1. Описать install profile:
   - Python version;
   - dependencies;
   - local paths;
   - product DB;
   - logs;
   - backups.
2. Добавить config template:
   - tenant id;
   - Mango credentials placeholders;
   - CRM mode;
   - retention settings;
   - API host/port.
3. Добавить backup/restore commands:
   - product DB backup;
   - config backup;
   - logs retention;
   - no audio by default unless policy allows.
4. Добавить healthcheck command:
   - API reachable;
   - DB readable;
   - scheduler state;
   - disk space;
   - credentials present but not printed.
5. Подготовить update procedure:
   - pull new version;
   - run migrations;
   - run tests/smoke;
   - rollback.

Acceptance criteria:

- установку можно повторить по инструкции;
- secrets не попадают в git/logs;
- backup/restore проверены на test DB;
- клиент может управлять сервисом без знания внутренних скриптов.

### Фаза 9. Demo readiness

Цель: подготовить демонстрационный сценарий для клиента.

Work packages:

1. Demo tenant:
   - обезличенные звонки;
   - sample Mango events;
   - sample CRM context;
   - sample knowledge items.
2. Demo script:
   - показать dashboard;
   - показать new calls;
   - показать blocked ASR gate;
   - показать ROP queue;
   - показать writeback preview;
   - показать knowledge/playbook.
3. Browser verification:
   - desktop;
   - mobile/tablet if UI web-based;
   - screenshots;
   - no blank states.
4. Demo safety:
   - no real CRM writes;
   - no real client data;
   - no secrets on screen.

Acceptance criteria:

- демо можно провести за 10-15 минут;
- оно не зависит от live Mango/AMO;
- оно показывает ценность продукта, а не технические детали.

### Фаза 10. Multi-client readiness

Цель: подготовить продукт к нескольким компаниям.

Work packages:

1. Tenant isolation:
   - tenant id в каждой product table;
   - separate config;
   - separate credentials;
   - separate retention policy.
2. Adapter registry:
   - Mango first;
   - AMO first;
   - другие телефонии/CRM позже;
   - common normalized contracts.
3. Support runbook:
   - как диагностировать capture;
   - как диагностировать API;
   - как проверить DB;
   - как собрать support bundle без secrets.
4. Compliance:
   - personal data policy;
   - retention;
   - deletion/export request;
   - access control.
5. Upgrade path:
   - SQLite appliance;
   - optional PostgreSQL;
   - optional central hosting later.

Acceptance criteria:

- второй tenant не требует копирования и переименования кода;
- конфигурация отделена от core logic;
- support/debug не раскрывает персональные данные и secrets.

## Управление параллельным аудитом

Да, можно дорабатывать проект и параллельно проводить аудит в другом диалоге, но
только при следующих правилах:

1. Аудит-диалог не меняет код, кроме собственного audit-файла.
2. Этот development-диалог перед каждой правкой смотрит `git status`.
3. Если аудит нашел проблему в файле, который здесь активно меняется, сначала
   фиксируется владелец правки.
4. Findings не применяются автоматически. Каждое предложение проходит triage:
   - согласен, исправляем сейчас;
   - согласен, но позже;
   - не согласен, причина;
   - нужно проверить.
5. Не запускать два агента, которые одновременно редактируют один файл.
6. После каждого пакета изменений делать маленький commit.

Рекомендуемый цикл обработки audit findings:

```text
Audit finding -> triage -> fix plan -> small patch -> focused tests -> commit -> update response plan
```

Практическая оговорка: аудит-диалог не является полностью read-only, потому что
по задаче он должен создать один собственный файл отчета. Это допустимо, если он
пишет только `docs/PROJECT_RISK_AND_CLEANUP_AUDIT_2026-05-09.md` и не меняет
исходный код, runtime artifacts, тесты и текущие productization-файлы.

## Ownership map для параллельной работы

Этот development-диалог может менять:

- `src/mango_mvp/productization/`;
- `tests/test_productization_*.py`;
- `scripts/mango_office_*.py`;
- productization docs: `docs/SAAS_*.md`, `docs/MANGO_*.md`;
- новые UI/API файлы, если они относятся к Product API/UI v1.

Аудит-диалог может создавать и обновлять:

- `docs/PROJECT_RISK_AND_CLEANUP_AUDIT_2026-05-09.md`;
- позже, после согласования, отдельный response-plan может быть создан этим
  development-диалогом.

Не менять одновременно без явного согласования:

- `.gitignore`;
- `Makefile`;
- `src/mango_mvp/gui.py`;
- `src/mango_mvp/services/resolve.py`;
- `src/mango_mvp/amocrm_runtime/`;
- любые файлы в `stable_runtime/`.

Если `git status` показывает неожиданный untracked/modified файл от другого
диалога, текущий диалог должен остановиться на этом файле и не включать его в
свой commit.

## Audit finding triage template

Каждое найденное аудитом замечание должно быть перенесено в response-plan в
таком формате:

```text
ID:
Priority: P0/P1/P2/P3
Source: PROJECT_RISK_AND_CLEANUP_AUDIT_2026-05-09.md#section
Finding:
Risk:
Affected files:
Decision: fix now / defer / reject / needs research
Reason:
Patch owner:
Verification:
Commit:
```

Правила принятия:

- P0 исправляется до следующего продуктового этапа, если подтвержден.
- P1 исправляется до UI/demo work, если влияет на safety или runtime.
- P2 можно группировать в hardening pass.
- P3 не должен блокировать развитие, если не создает путаницу в эксплуатации.

## Testing matrix

Минимальная матрица тестов по слоям:

| Слой | Когда запускать | Команда |
|---|---|---|
| Productization core | После изменений в `src/mango_mvp/productization` | `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_productization_*.py` |
| Insight layer | После изменений в `src/mango_mvp/insights` | `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_insight_readiness.py tests/test_knowledge_base.py tests/test_llm_review.py tests/test_llm_review_merge.py tests/test_outcome_linker.py tests/test_pilot_extraction.py tests/test_rop_validation_pack.py` |
| Agent runtime | После изменений в opt-in agent API | `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_agent_runtime.py` |
| Resolve safety | После изменений в Resolve/ASR rescue logic | `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_resolve.py` |
| Product API HTTP | После изменений в read-only API routes | `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_productization_product_api.py tests/test_productization_product_api_http.py` |
| Full new baseline | Перед push/PR | `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_productization_*.py tests/test_insight_readiness.py tests/test_knowledge_base.py tests/test_llm_review.py tests/test_llm_review_merge.py tests/test_outcome_linker.py tests/test_pilot_extraction.py tests/test_rop_validation_pack.py tests/test_agent_runtime.py tests/test_resolve.py` |

Не запускать в этой ветке без отдельного подтверждения:

- команды, которые запускают ASR/R+A;
- batch/start/run-ui scripts;
- scripts, которые пишут в AMO/Tallanto/CRM;
- scripts, которые меняют `stable_runtime` DB/audio/transcripts.

## Commit and branch discipline

Рекомендуемый порядок работы:

1. Перед началом: `git status --short --branch`.
2. Один work package = один небольшой commit.
3. Не смешивать docs cleanup, runtime changes, Product API, UI и tests в одном
   commit.
4. Перед commit: `git diff --cached --check`.
5. После commit: `git status --short --branch`.
6. После серии commits: `git push`.

Если появляется audit finding P0/P1, новые feature-commits лучше поставить на
паузу до triage этого finding.

## Приоритеты ближайших 10 implementation steps

1. Принять и разобрать параллельный risk/cleanup audit.
2. Закрыть P0/P1 findings, если они будут.
3. Сделать supervised Product API service profile.
4. Добавить Product API service runbook и smoke test.
5. Начать UI v1 против read-only Product API.
6. Улучшить Mango shadow poll reliability: retries, rate limits, delayed recordings.
7. Сделать capture inbox как продуктовую очередь.
8. Сформировать writeback preview contract для AMO.
9. Стабилизировать ROP/knowledge playbook export.
10. Подготовить client-hosted install profile.

## Execution horizons

### Горизонт 1. Следующие 1-2 рабочих дня

Цель: не наращивать функциональность поверх потенциальных рисков.

Действия:

1. Дождаться audit-файла от параллельного диалога.
2. Создать response-plan по findings.
3. Закрыть подтвержденные P0/P1.
4. Перепроверить baseline tests.
5. После этого выбрать один следующий feature package: supervised Product API
   service или UI v1 skeleton.

Ожидаемый результат: понятно, какие риски блокируют развитие, а какие можно
отложить.

### Горизонт 2. Ближайшие 1-2 недели

Цель: сделать продукт запускаемым как локальный сервис.

Действия:

1. Supervised Product API service.
2. Health/readiness/smoke commands.
3. Product API runbook.
4. Первый UI v1 dashboard против read-only API.
5. Capture inbox screen или API endpoint для новых Mango events.
6. Начальный install profile для внутреннего appliance.

Ожидаемый результат: сервис можно запустить и показать внутри компании без
ручного чтения JSON-отчетов.

### Горизонт 3. Ближайший месяц

Цель: довести продукт до внутреннего автономного MVP.

Действия:

1. Mango shadow poll hardening.
2. Controlled recording metadata ingest.
3. Processing bridge dry-run.
4. ROP/knowledge playbook product screen/export.
5. AMO writeback preview contract.
6. Backup/restore для product appliance DB.

Ожидаемый результат: продукт не просто показывает старые данные, а ведет
контролируемую работу с новыми звонками.

### Горизонт 4. После внутреннего MVP

Цель: подготовить демонстрацию и первые client-hosted установки.

Действия:

1. Demo tenant и anonymized dataset.
2. Demo script на 10-15 минут.
3. Client install package.
4. Secrets/retention policy для клиента.
5. Support bundle без secrets.
6. Решение, когда SQLite еще достаточно, а когда нужен PostgreSQL.

Ожидаемый результат: продукт можно показать и поставить второму клиенту без
ручной инженерной импровизации.

## Что пока не делать

- Не мигрировать всю систему на PostgreSQL без реальной необходимости.
- Не включать auto-ASR для Mango capture.
- Не включать live CRM writeback.
- Не делать multi-tenant cloud до локального appliance.
- Не строить красивый UI, который напрямую дергает batch scripts.
- Не удалять runtime artifacts без отдельного cleanup approval.
- Не смешивать processing debt и SaaS productization в одном коммите.

## Definition of done для следующего крупного этапа

Следующий крупный этап считается завершенным, когда:

- есть supervised local Product API service;
- UI v1 читает Product API;
- Mango capture работает в shadow/controlled mode;
- все опасные действия имеют gates;
- audit findings P0/P1 закрыты или явно отложены с причиной;
- есть install/runbook для локального appliance;
- тесты productization/insight/agent runtime проходят;
- ветка запушена на GitHub и готова к review.

## Self-audit log

### Audit 1. Проверка полноты

Риск: план мог описать только SaaS UI/API и забыть про аудит, упаковку, CRM и
knowledge layer.

Правка: добавлены отдельные фазы для audit intake, controlled CRM writeback,
knowledge productization, client-hosted packaging, demo readiness и multi-client
readiness.

### Audit 2. Проверка безопасности

Риск: план мог нечаянно предложить включить ASR, скачивание аудио или CRM writeback
слишком рано.

Правка: во все опасные фазы добавлены gates, dry-run, preview, opt-in и acceptance
criteria, запрещающие silent runtime/CRM writes.

### Audit 3. Проверка тестируемости

Риск: план мог остаться набором желаний без критериев проверки.

Правка: у каждой фазы добавлены acceptance criteria, а у ближайших фаз добавлены
конкретные безопасные команды проверки.

### Audit 4. Проверка параллельной работы

Риск: параллельный audit-диалог может конфликтовать с development-диалогом.

Правка: добавлены правила ownership, triage, запрет одновременного редактирования
одних файлов и цикл обработки audit findings.

### Audit 5. Проверка исполнимости

Риск: даже хороший roadmap может быть слишком общим и не давать ответа, что
делать завтра.

Правка: добавлены execution horizons на 1-2 дня, 1-2 недели, месяц и период
после внутреннего MVP.
