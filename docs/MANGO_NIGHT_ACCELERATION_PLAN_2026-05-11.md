# Mango night acceleration plan

Дата: 2026-05-11

Ветка фокуса: `productization/channel`

## Назначение

Этот документ фиксирует ночной 6-8 часовой план ускорения productization/channel
направления. Цель ночи - превратить уже начатый channel runtime из набора
безопасных контрактов и read-only adapter-ов в более демонстрируемый продуктовый
контур: operator-facing inbox, API/readiness surface, demo payloads и ясные gates
для следующего controlled-send этапа.

План намеренно не пересекается с processing/AMO duplicate направлением второго
диалога. Ночью не решаем AMO multi-contact rows, не выбираем правильные AMO
контакты, не пишем duplicate-resolution policy и не двигаем CRM writeback queue.

## Цель ночи

К утру получить проверяемый channel/productization слой, который можно показать
как часть AI Office:

- входящие сообщения из Telegram/Web/CRM chat нормализуются в единый формат;
- для сообщений строится безопасный draft reply, который всегда требует approval;
- recommended actions видны оператору как proposal, а не выполняются;
- inbox/approval workspace имеет стабильный snapshot contract;
- Product/API слой может отдать channel status/demo data в read-only режиме;
- весь контур подтвержден focused tests и кратким runbook.

Главный критерий: демонстрация должна выглядеть как управляемый workflow
`message -> preview -> approval queue -> mock outcome`, а не как набор разрозненных
модулей.

## Жесткие ограничения

- Не менять `stable_runtime`, кроме чтения существующих документов/контрактов.
- Не менять processing pipeline, ASR, Resolve+Analyze, transcript quality gates.
- Не менять AMO/Tallanto writeback, AMO duplicate/manual resolution, CRM queue
  classification и readback logic.
- Не делать live send в Telegram/Web/CRM chat.
- Не добавлять реальные tokens, secrets, webhooks, polling или external network
  calls.
- Не включать LLM/RAG в channel reply generation.
- Не импортировать legacy Telegram bot source как runtime dependency.
- Любые новые действия остаются dry-run/mock/preview и имеют audit trail.
- Перед каждым блоком правок проверять `git status --short`, потому что другие
  агенты могут работать параллельно.

## Не область этой ночи

Явно исключено:

- processing/AMO duplicate plan второго диалога;
- разбор `needs_human.csv`, выбор AMO contact id, merge duplicate contacts;
- live CRM/AMO/Tallanto writes;
- backfill, staged writeback, post-writeback readback;
- массовая переработка исторических звонков;
- смена product DB storage strategy;
- production deploy на VPS/Render;
- Telegram controlled live send.

Если в процессе всплывает зависимость от этих тем, она фиксируется как blocker или
handoff note, но не решается в этой ветке.

## План на 6-8 часов

### Час 0:30. Orientation и guardrails

Задачи:

- Проверить текущий `git status --short`.
- Прочитать актуальные channel/productization документы и последние operator
  status notes.
- Зафиксировать список файлов, которые можно трогать в рамках channel ветки.
- Подтвердить, что AMO/manual-resolution изменения остаются за другим диалогом.

Deliverable:

- Короткая рабочая заметка в итоговом отчете ночи: touched files, skipped areas,
  known parallel changes.

Готовность:

- Нет правок вне channel/productization docs/API/tests.
- Нет конфликтов с уже измененными чужими файлами.

### Час 0:30-2:00. Channel inbox API contract

Задачи:

- Спроектировать read-only endpoint contract для channel inbox snapshot.
- Связать `ChannelMemoryStore.snapshot()` с Product/API style response, без
  persistent DB migration.
- Добавить фильтры уровня contract: channel, status, tenant/source, limit.
- Убедиться, что `raw_payload` по умолчанию не выходит наружу.

Deliverables:

- Read-only channel inbox summary/detail contract.
- Deterministic JSON examples for demo use.
- Focused tests for filtering, redaction and empty-state behavior.

Готовность:

- Endpoint/contract не имеет mutation methods.
- Snapshot стабилен и JSON-ready.
- Empty inbox выглядит как нормальное состояние, а не ошибка.

### Час 2:00-3:30. Demo workflow pack

Задачи:

- Подготовить компактный demo fixture pack для трех каналов:
  `telegram_business`, `site_chat`, `crm_chat`.
- Прогнать каждый payload через parse -> preview -> store -> snapshot.
- Добавить cases: текстовый вопрос, вложение без текста, коммерческий сигнал,
  handoff/manual-review signal.
- Сохранить все в локальный deterministic demo command или тестовый helper без
  записи в `stable_runtime`.

Deliverables:

- Demo payload fixtures.
- One-command/readme-style сценарий для локальной демонстрации.
- Expected snapshot artifacts в tests/fixtures или docs example, если это
  соответствует текущему стилю проекта.

Готовность:

- Demo работает без network, secrets, Telegram SDK и CRM credentials.
- В каждом сценарии `requires_approval=true`.
- Recommended actions не исполняются и остаются proposals.

### Час 3:30-5:00. Approval lifecycle hardening

Задачи:

- Проверить lifecycle transitions для drafts/actions на product-grade edge cases:
  duplicate message, duplicate draft, reject after approval, superseded draft,
  mock send only after approval.
- Уточнить blocked reasons и audit events, чтобы оператор видел причину.
- Добавить regression tests на запрещенные transitions.

Deliverables:

- Более строгий approval lifecycle.
- Tests на idempotency и fail-closed transitions.
- Документированный список разрешенных статусов.

Готовность:

- Нельзя получить `mock_sent` без approval.
- Нельзя получить live `sent/executed` в текущем слое.
- Повторный payload не создает хаотичные дубли в inbox.

### Час 5:00-6:30. Operator-facing readiness

Задачи:

- Добавить channel readiness summary в существующий product/operator стиль, если
  это можно сделать изолированно.
- Проверить, что readiness честно показывает: live send disabled, LLM disabled,
  CRM writes disabled, adapters read-only.
- Подготовить краткий runbook: как открыть demo, какие команды прогнать, что
  считается green/red.

Deliverables:

- Channel readiness summary.
- Runbook section or dedicated docs note.
- Focused tests/readiness command.

Готовность:

- Оператор с первого экрана понимает, что контур безопасный и read-only.
- Красные состояния объясняют next action, а не только fail flag.
- Readiness не зависит от AMO duplicate/manual-resolution artifacts.

### Час 6:30-7:30. Regression и cleanup

Задачи:

- Запустить focused channel/productization tests.
- Если есть время, запустить минимальный Product API readiness smoke.
- Проверить `git diff --check`.
- Проверить `git status --short` и отделить свои изменения от чужих.

Deliverables:

- Список пройденных команд.
- Список не запущенных команд с причиной.
- Финальный changed-files список только по зоне ночной работы.

Готовность:

- Focused tests green.
- Нет форматных ошибок в diff.
- Нет случайных изменений в `src/tests/scripts/stable_runtime`, если они не были
  заранее согласованы для отдельной реализации. Для текущего документа эти
  директории вообще не трогаются.

### Час 7:30-8:00. Handoff

Задачи:

- Сформировать короткий итог: что готово, что осталось, какие blockers.
- Отдельно отметить, что AMO duplicate/manual-resolution остается за другим
  планом.
- Подготовить next step на следующий дневной блок: controlled-send design или UI
  inbox, но без live send.

Deliverables:

- Night completion note.
- Next-step list на 2-4 пункта.
- Risk list с владельцами.

Готовность:

- У следующего агента есть ясный вход в работу без чтения всего diff.
- Нет скрытых production действий.

## Итоговые deliverables ночи

Минимальный успешный набор:

1. Read-only channel inbox/API contract.
2. Demo workflow pack для Telegram Business, site chat и CRM chat.
3. Approval lifecycle tests с fail-closed статусами.
4. Channel readiness summary/runbook.
5. Focused test log.
6. Handoff note с changed files и explicit non-overlap with AMO duplicate plan.

Расширенный набор, если остается время:

1. Static operator demo snapshot для channel inbox.
2. Дополнительные fixtures для вложений и edited messages.
3. Draft/action sorting для удобства операторского UI.
4. Negative tests на raw payload leakage.

## Критерии готовности к утру

Ночной блок считается готовым, если:

- channel workflow можно объяснить и показать за 3 минуты;
- все новые channel endpoints/commands read-only;
- live-send, CRM write и LLM/RAG явно выключены;
- draft reply всегда требует manager approval;
- lifecycle transitions fail-closed;
- tests покрывают happy path, duplicate path и forbidden transition path;
- документы называют границы ответственности и не требуют AMO duplicate решений;
- changed files не заходят в processing/AMO duplicate/manual-resolution область.

## Риски и стоп-сигналы

Стоп-сигналы:

- Нужно выбрать правильный AMO contact id или объединить дубли контактов.
- Нужно изменить `stable_runtime` как источник правды.
- Нужно включить Telegram webhook/polling/live send.
- Нужно писать в CRM/Tallanto/AMO.
- Нужно запускать ASR/R+A/backfill.

При любом стоп-сигнале действие не выполняется. Фиксируется handoff note с
описанием зависимости и владельцем соседнего направления.

## Рекомендованные команды проверки

Перед началом:

```bash
git status --short
```

Focused channel checks:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_channels_*.py
```

Focused productization checks, если touched files затрагивают Product API/operator
surface:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_productization_product_api.py tests/test_productization_product_api_http.py
```

Diff hygiene:

```bash
git diff --check
git status --short
```
