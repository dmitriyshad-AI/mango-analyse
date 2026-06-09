# AMO + Wappi draft loop

Режим: Wappi polling -> bot draft -> AMO note. Код не отправляет сообщения клиентам.

## Внешние файлы

Все файлы с ПДн и секретами лежат вне репозитория:

- `~/.mango_secrets/amo_wappi.env` - Wappi/AMO токены.
- `~/.mango_secrets/amo_wappi_profiles.json` - Wappi `profile_id -> brand`.
- `~/.mango_secrets/draft_loop_pairs.json` - явные пары `(profile_id, chat_id) -> lead_id`.
- `~/.mango_secrets/amo_wappi_phase1.json` - allowlist AMO сделок, если используется.
- `~/.mango_secrets/STOP_DRAFT_LOOP` - полный стоп цикла.
- `~/.mango_local/draft_loop/` - state, journal, manager edit log.

`draft_loop_pairs.json` пример:

```json
[
  {
    "profile_id": "ec2eed50-b55f",
    "chat_id": "<chat_id>",
    "lead_id": "49832125",
    "expected_brand": "foton"
  }
]
```

Голый `chat_id` запрещён: ключ всегда составной `(profile_id, chat_id)`.

## Безопасный dry-run

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
python3 scripts/run_amo_wappi_draft_loop.py --once --dry-run
```

Dry-run читает Wappi, собирает контекст и строит черновик, но не пишет note в AMO и не помечает сообщения обработанными.

## Live note в тестовую сделку

Только после отдельного подтверждения Дмитрия:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
python3 scripts/run_amo_wappi_draft_loop.py --once --live-write
```

Даже в live-write запись проходит только если:

- есть явная пара в `draft_loop_pairs.json`;
- `expected_brand` совпадает с брендом Wappi-профиля;
- `lead_id` есть в allowlist phase1 или явно следует из пары;
- транспорт разрешает только `POST /api/v4/leads/{id}/notes`.

## STOP

Если существует `~/.mango_secrets/STOP_DRAFT_LOOP`, цикл только читает Wappi и пишет raw journal. Бот не вызывается, AMO note не пишется, входящие не помечаются обработанными.

## Step 0 status

Living API discovery: `D1_audit_backlog/WAPPI_DRAFT_LOOP_STEP0_2026-06-10.md`.

Endpoint history подтверждён. Видимость исходящих, отправленных именно из AMO-интерфейса, требует ручной проверки на тестовой сделке 49832125. До этого `unedited_rate` считать экспериментальным.

