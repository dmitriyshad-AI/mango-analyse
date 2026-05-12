# Customer Timeline Approval Workspace Runbook

Дата: 2026-05-12

## Назначение

Approval workspace - это статический read-only экран для оператора. Он собирает из `customer_timeline.sqlite` карточку клиента, ленту событий, bot-context, конфликты и safety gates.

Этап 6 не поднимает сервер и не сохраняет решения оператора. Он создает переносимые артефакты:

- JSON view model;
- HTML workspace.

## Что workspace не делает

- не пишет в `customer_timeline.sqlite`;
- не пишет в AMO;
- не пишет в Tallanto;
- не отправляет письма и сообщения;
- не запускает ASR;
- не запускает Resolve + Analyze;
- не читает файлы артефактов по локальным путям;
- не меняет `stable_runtime`.

## Команда

```bash
PYTHONPATH=src python3 scripts/customer_timeline_approval_workspace.py \
  --tenant-id foton \
  --timeline-db /path/to/product_root/customer_timeline/customer_timeline.sqlite \
  --allowed-root /path/to/product_root \
  --customer-id customer:example \
  --query "стоимость" \
  --out-json /path/to/product_root/reports/customer_timeline_approval_workspace.json \
  --out-html /path/to/product_root/reports/customer_timeline_approval_workspace.html
```

Если `--customer-id` не указан, workspace выбирает первого клиента из результата поиска `--query`.

## Как читать HTML

- `Customer search` - найденные клиенты с замаскированными контактами.
- `Selected customer` - выбранная карточка клиента.
- `Timeline` - события клиента: звонки, письма, CRM snapshots, каналы.
- `Bot context readiness` - какие фрагменты можно использовать будущему боту, а какие требуют проверки.
- `Conflicts` - открытые конфликты идентичности.
- `Safety gates` - подтверждение, что live-действия заблокированы.

## Status

- `ready_for_review` - можно смотреть оператору, явных блокеров нет.
- `blocked_by_conflict` - есть конфликт идентичности, автоматический бот/approval нельзя считать безопасным.
- `needs_context` - не хватает bot-safe контекста.

`validation_ok=true` означает, что workspace корректно построен. Это не означает, что оператор уже одобрил клиента.

## Exit codes

- `0` - workspace построен;
- `1` - workspace построен, но validation failed;
- `2` - ошибка аргументов, небезопасный путь, проблема открытия БД.

## Redaction

HTML и JSON не должны отдавать:

- сырые payload;
- `record_json`;
- локальные пути артефактов;
- байты вложений;
- полные телефоны/email;
- live-write команды.

## Проверка

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_customer_timeline_*.py
```

Pass criteria:

- static HTML генерируется;
- JSON view model детерминирован при fixed `generated_at`;
- БД открывается только через read-only API;
- `stable_runtime` и outside-root outputs блокируются;
- malicious customer text экранируется в HTML;
- live actions остаются disabled/blocked.
