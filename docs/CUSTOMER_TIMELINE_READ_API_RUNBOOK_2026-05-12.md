# Customer Timeline Read API Runbook

Дата: 2026-05-12

## Назначение

Read API слой дает UI, операторским отчетам и будущему боту безопасный способ читать `customer_timeline.sqlite`.

Ключевая идея: внешний код не читает SQLite-таблицы напрямую. Он обращается к фасаду `CustomerTimelineReadApi`, который открывает БД только в read-only режиме и возвращает стабильные JSON-контракты.

## Что слой не делает

- не пишет в `customer_timeline.sqlite`;
- не пишет в AMO;
- не пишет в Tallanto;
- не отправляет письма или сообщения;
- не запускает ASR;
- не запускает Resolve + Analyze;
- не читает файлы артефактов по путям;
- не меняет `stable_runtime`.

## Основные возможности

- health/status по timeline-БД;
- общий summary по tenant;
- список клиентов с поиском и пагинацией;
- карточка клиента: связи, возможности, события, сигналы, контекст для бота, конфликты;
- лента событий клиента;
- bot-context только из разрешенных chunks;
- поиск по событиям, сигналам и bot-context;
- список открытых конфликтов идентичности;
- CLI-отчет для проверки готовности.

## CLI report

```bash
PYTHONPATH=src python3 scripts/customer_timeline_read_report.py \
  --tenant-id foton \
  --timeline-db /path/to/product_root/customer_timeline/customer_timeline.sqlite \
  --allowed-root /path/to/product_root \
  --customer-id customer:example \
  --query "стоимость" \
  --out /path/to/product_root/reports/customer_timeline_read_report.json
```

Exit codes:

- `0` - отчет построен, `validation_ok=true`;
- `1` - отчет построен, но есть ожидаемый blocker;
- `2` - ошибка аргументов, небезопасный путь или БД не открылась.

## Программное использование

```python
from pathlib import Path
from mango_mvp.customer_timeline import CustomerTimelineReadApi, CustomerTimelineReadApiConfig

config = CustomerTimelineReadApiConfig(
    timeline_db=Path("/path/to/customer_timeline.sqlite"),
    allowed_root=Path("/path/to/product_root"),
)

with CustomerTimelineReadApi.open(config) as api:
    profile = api.customer_profile("foton", "customer:example")
```

## Read-only routes

Этап 5 не поднимает HTTP-сервер. Но уже есть route-функция для будущего HTTP слоя:

- `GET /health`
- `GET /summary`
- `GET /customers`
- `GET /customer`
- `GET /customer/timeline`
- `GET /customer/bot-context`
- `GET /search`
- `GET /conflicts`

Любой `POST`, `PUT`, `PATCH`, `DELETE` возвращает read-only блокировку.

## Redaction

Ответы API не отдают:

- сырые payload;
- `record_json`;
- локальные пути артефактов;
- байты файлов/вложений;
- внутренние record hashes;
- полные телефоны и email в списках.

Для bot-context дополнительно возвращаются только chunks, у которых:

- `allowed_for_bot=true`;
- `requires_manager_review=false`.

## Проверка

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_customer_timeline_*.py
```

Pass criteria:

- все тесты customer timeline проходят;
- read-only API открывает SQLite через `mode=ro`;
- `PRAGMA query_only=1`;
- missing DB не создается;
- `stable_runtime` и runtime-looking DB names заблокированы;
- bot-context не отдает review-required chunks в режиме `allowed_only=true`.
