# Mango Analyse

Единый проект для внутренних процессов Mango/Foton/UNPK:

- публичные Telegram-боты Фотон и УНПК отвечают клиентам через одно модельное ядро;
- Wappi принимает сообщения и создаёт менеджерские черновики в AMO;
- `customer_timeline` объединяет разрешённую историю клиента;
- конвейер звонков забирает записи Mango, распознаёт и анализирует их;
- почтовый конвейер загружает письма read-only, обрабатывает и добавляет их в timeline;
- база знаний даёт боту подтверждённые цены, адреса, расписание и условия.

Финансовая модель вынесена в отдельный проект
`/Users/dmitrijfabarisov/Projects/Foton_Finance` и не является частью этого
репозитория.

## Текущее устройство

Каноническая ветка продукта — `main`. Временные страховочные worktree не входят
в live-контур и перечислены в `docs/worktrees_registry.md`. Основная папка на
этом Mac:

```text
/Users/dmitrijfabarisov/Projects/Mango analyse
```

Живой контур черновиков:

```text
Wappi -> integrations/draft_loop.py
      -> SubscriptionLlmDraftProvider.build_draft()
      -> direct_path
      -> post_layers + output_verification_floor
      -> черновик-заметка в AMO
```

Клиенту этот контур ничего не отправляет. Автоответ запрещён; результат остаётся
черновиком для менеджера.

Публичный Telegram — отдельный тонкий транспорт: один процесс на бренд вызывает
то же ядро и отправляет модельный текст только при конечном разрешённом маршруте
и пройденной выходной проверке. Старый публичный runner не используется.

Основные защитные правила:

- P0: реальные возвраты, споры оплаты, серьёзные жалобы и юридические вопросы
  передаются менеджеру;
- бренд: факты Foton и UNPK не смешиваются;
- ПДн: телефон, email и служебные идентификаторы не попадают в клиентский текст;
- факты: цены, даты, адреса, расписание и обещания допустимы только при наличии
  подтверждённого источника;
- маршрут: последующий слой не может повысить `manager_only` до самостоятельного
  ответа.

Подробная карта: [ARCHITECTURE.md](ARCHITECTURE.md).

Контракт live: [docs/LIVE_BOT_CONTRACT.md](docs/LIVE_BOT_CONTRACT.md).

Рабочие команды: [docs/RUNBOOK.md](docs/RUNBOOK.md).

## Источник правды

Перед работой читать в таком порядке:

1. `AGENTS.md`;
2. `README.md`;
3. `ARCHITECTURE.md`;
4. `docs/PROJECT_NOW.md`;
5. `docs/RUNBOOK.md`;
6. `docs/DECISIONS_LOG.md`.

Актуальное ТЗ и свежий audit pack дополняют эту шестёрку только в границах
конкретной задачи. `docs/_archive/` хранит историю и не является источником
текущей правды.

Состояние процесса проверяется по PID, cwd, env, startup manifest и heartbeat,
а не по старому имени папки в документе:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  python3 scripts/skills/live_truth.py --no-write
```

## Установка для разработки

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Требования:

- Python 3.9+;
- `ffmpeg` и `ffprobe` для работы со звонками;
- локальные секреты только вне Git;
- `PyYAML` входит в зависимости проекта.

## Безопасные проверки

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  python3 -m pytest --collect-only -q

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  python3 -m pytest -q <tests>
```

Импорт ядра бота:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -c \
  "from mango_mvp.channels.subscription_llm_parts import SubscriptionLlmDraftProvider"
```

Зелёные тесты означают только формальную исправность. Для базы знаний,
клиентских ответов, CRM-текстов и коммерческих фактов дополнительно обязателен
смысловой аудит по `docs/SEMANTIC_REVIEW_RULES.md`.

## Основные данные

Текущая база знаний бота:

```text
product_data/knowledge_base/kb_release_20260813_v6_8_owner_approved/
  kb_release_v3_snapshot.json
```

Текущая customer timeline:

```text
product_data/customer_timeline/customer_timeline_prod_20260621/
  customer_timeline.sqlite
```

Runtime-указатели звонков читаются из `stable_runtime/CURRENT_RUNTIME.json`.
Менять базы, внешние системы или live-службы можно только по отдельному
подтверждённому ТЗ.

## Конвейер звонков

Базовые стадии CLI:

```bash
mango-mvp ingest --recordings-dir <dir> --metadata-csv <calls.csv>
mango-mvp transcribe --limit <n>
mango-mvp resolve --limit <n>
mango-mvp analyze --limit <n>
mango-mvp stats
```

ASR, Resolve+Analyze и реальные пакетные запуски по клиентским данным не
запускать без отдельного подтверждения. Штатные фоновые службы описаны в
`docs/RUNBOOK.md`.

## Безопасность Git

- не использовать `git add -A`;
- не смешивать код, runtime-данные и внешние выгрузки в одном коммите;
- секреты, ПДн, `.env`, токены, аудио, транскрипты и customer timeline не
  коммитить;
- Graphify использовать только как навигационную карту, затем проверять вывод
  по исходникам.
