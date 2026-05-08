# DB engine options for client-hosted product mode

Дата: 2026-05-07

Контекст: продукт предполагается ставить на отдельный ноутбук или сервер клиента, с управлением у клиента. Нужно решение серьезнее текущего прямого SQLite-файла, но без сложности PostgreSQL/multi-cloud SaaS.

## Короткий вывод

Рекомендуемый путь:

1. Для текущей исторической обработки оставить SQLite.
2. Для нового product operational слоя выбрать MariaDB/MySQL как самый прагматичный non-Postgres server-DB вариант.
3. DuckDB добавить позже только для аналитики и insight-запросов.
4. Если хочется вообще не ставить DB-сервер, можно оставить SQLite, но только через single-writer service: все процессы ходят в локальный API, а не пишут в `.db` напрямую.

## Варианты

### SQLite with single-writer service

Это не замена SQLite, а способ убрать главную проблему: много процессов не пишут в файл напрямую. Capture, UI, scheduler и workers обращаются к локальному backend API, а уже backend один пишет в SQLite.

Плюсы:

- минимальная установка;
- легко бэкапить как продуктовый appliance;
- текущий код ближе всего к этому варианту;
- подходит для одного ноутбука/сервера клиента.

Минусы:

- остается один writer;
- нужно дисциплинированно делать WAL/checkpoint/backup;
- нельзя класть DB на сетевой диск;
- масштабирование ограничено.

SQLite WAL официально дает конкурентное чтение и запись, но writer все равно один, а WAL требует, чтобы процессы работали на одной машине и использовали shared memory.

Решение: годится для appliance v1, если все записи проходят через один локальный backend.

### MariaDB/MySQL

Это главный кандидат, если нужен шаг серьезнее SQLite, но без PostgreSQL.

Плюсы:

- полноценный DB-сервер;
- нормальная многопроцессная запись;
- понятная установка через Docker/пакет;
- привычные SQLAlchemy/Python драйверы;
- InnoDB транзакции, индексы, constraints;
- стандартные backup tools: `mariadb-dump`, MariaDB Backup;
- проще объяснить клиенту как "локальная база сервиса".

Минусы:

- все равно появляется DB service, пользователь, пароль, порт, backup;
- нужны миграции схемы;
- чуть больше эксплуатации, чем файл SQLite;
- JSON/аналитика слабее, чем в Postgres, но для operational слоя это не критично.

Решение: лучший компромисс для client-hosted Mango Analyse v1.

### DuckDB

DuckDB хорош для аналитики по большому корпусу: отчеты, группировки, correlation/insight-layer, выгрузки.

Плюсы:

- быстрые аналитические запросы;
- простой файл;
- отлично подходит для read-heavy reports.

Минусы:

- не предназначен для множества мелких operational транзакций;
- запись из нескольких процессов в один DB-файл не является штатным режимом;
- не подходит как основная очередь capture/scheduler/workers.

Решение: использовать рядом с operational DB для insight-layer, но не вместо SQLite/Postgres/MariaDB.

### Firebird

Firebird технически интересен: легкий RDBMS, есть server/embedded режимы, официальный Python driver.

Плюсы:

- легче Postgres;
- mature SQL database;
- может быть хорош для embedded/on-prem сценариев.

Минусы:

- меньше Python/SQLAlchemy практики в обычных SaaS-командах;
- меньше готовых рецептов для FastAPI/analytics/deploy;
- сложнее нанимать/передавать поддержку.

Решение: держать как запасной вариант, но не брать первым.

### PostgreSQL

Postgres остается лучшим вариантом для полноценного SaaS/multi-tenant, но в текущем product appliance подходе его можно не брать первым.

Решение: вернуться к Postgres, если появятся несколько tenant на одном сервере, сложная аналитика прямо в operational DB, row-level security или managed cloud deployment.

## Практическая рекомендация

Для Mango Analyse client-hosted v1:

```text
SQLite:
  historical batch runtime
  portable worker packs
  local legacy compatibility

MariaDB/MySQL:
  product operational layer
  tenants
  captured calls
  scheduler runs
  queue statuses
  CRM writeback audit
  UI state

DuckDB:
  insight/reporting layer
  question/answer/outcome aggregates
  offline analytics
```

## Migration strategy

Не мигрировать все сразу.

1. Зафиксировать product data contracts.
2. Создать repository interfaces под operational entities.
3. Реализовать MariaDB schema только для новых product tables.
4. Старую SQLite историю читать через bridge/export.
5. Переносить исторические records только после появления конкретного UI/insight use case.

## Когда SQLite все еще допустим

SQLite можно оставить как product DB v1, если:

- сервис установлен на одной машине;
- только один backend процесс пишет в DB;
- UI/capture/scheduler/workers не открывают `.db` напрямую;
- настроены WAL, busy timeout и checkpoint;
- backup делается через SQLite backup API или остановку сервиса, а не копированием `.db` без `-wal/-shm`.

Если хочется минимальной установки, этот вариант можно сделать первым, а MariaDB держать как следующий profile.

## Decision

Для этой ветки не делаем немедленную миграцию. В коде сначала строим DB-agnostic contracts и read-only Mango capture. После shadow polling POC следующий безопасный шаг: `product repository` interface с двумя будущими профилями:

- `sqlite_single_writer` для appliance v1;
- `mariadb` для более серьезной client-hosted установки.

Источники:

- SQLite WAL: `https://www.sqlite.org/wal.html`
- DuckDB concurrency: `https://duckdb.org/docs/stable/connect/concurrency`
- MariaDB backup overview: `https://mariadb.com/docs/server/server-usage/backup-and-restore/backup-and-restore-overview`
- MariaDB dump: `https://mariadb.com/docs/server/clients-and-utilities/backup-restore-and-import-clients/mariadb-dump`
- Firebird Python driver: `https://www.firebirdsql.org/en/python-driver/`
