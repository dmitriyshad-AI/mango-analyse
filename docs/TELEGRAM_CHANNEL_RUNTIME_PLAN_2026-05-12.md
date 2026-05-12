# Telegram Channel Runtime Plan

Дата: 2026-05-12
Контур: Mango Analyse / AI Office channel adapters
Статус: read-only skeleton + integration plan

## Позиция

Telegram не должен становиться отдельным продуктом и не должен тащить legacy RAG.
Он входит в общий channel слой как adapter:

```text
Telegram history / Telegram Bot / Telegram Business
  -> ChannelMessage
  -> identity matching
  -> customer timeline
  -> draft reply / recommended action
```

На текущем этапе включены только read-only парсинг, нормализация, локальный
archive/import skeleton и отчеты без персональных текстов. Live Telegram Bot API,
CRM/Tallanto writes, ASR и Resolve+Analyze не запускаются.

## Что Уже Есть

В `src/mango_mvp/channels/` уже есть foundation:

- `contracts.py`: `ChannelMessage`, `ChannelSession`, `BotReply`,
  `RecommendedAction`, `ChannelAdapter`, idempotency helpers.
- `telegram_adapter.py`: `TelegramReadOnlyAdapter` для обычных Telegram updates,
  Telegram Business updates и Mini App `web_app_data`.
- `persistence.py`: `ChannelSQLiteStore`, который запрещает `stable_runtime`
  DB paths и не сохраняет `raw_payload`.
- `tests/test_channels_telegram_adapter.py`: покрывает parse/render/no-live-send.

Новый изолированный слой для history:

- `src/mango_mvp/channels/telegram_history.py`;
- `tests/test_channels_telegram_history.py`.

Он добавляет:

- inventory исторического export без вывода текстов сообщений;
- historical export -> `ChannelMessage` с каналом `telegram_history`;
- idempotent import в `ChannelSQLiteStore`;
- generic identity map: Telegram id / username / phone -> candidate customer;
- Tallanto CSV identity reader;
- matching classes `strong_unique`, `ambiguous`, `unmatched`;
- timeline-ready records: `telegram_message`, `telegram_identity_link`;
- safety contract: no network, no Telegram API, no CRM/Tallanto writes, no ASR/R+A.

## Inventory Telegram Export

Источник:

```text
telegram_exports (2)/local_vm_2024-04-01
```

Файлы:

- `summary.json`;
- `dialogs.jsonl`;
- `messages.jsonl`.

Агрегаты read-only inventory:

| Метрика | Значение |
|---|---:|
| dialogs | 1653 |
| messages | 13223 |
| dialog top-message range | 2022-09-07T10:54:03+00:00 - 2026-04-15T14:56:54+00:00 |
| message range | 2024-04-01T06:46:28+00:00 - 2026-04-15T14:56:54+00:00 |
| inbound messages | 7865 |
| outbound messages | 5358 |
| user dialogs | 1651 |
| group dialogs | 1 |
| channel dialogs | 1 |
| user messages | 12859 |
| chat messages | 175 |
| channel messages | 189 |
| text only | 11694 |
| media only | 794 |
| text + media | 482 |
| empty/no media | 253 |
| reply messages | 2245 |

Доступные поля связи в raw export:

| Поле связи | Наличие |
|---|---:|
| `dialog_id` / Telegram-like id | 1653 dialogs |
| `sender_id` | 13223 messages |
| `name` / `dialog_name` | есть, но PII и не используется для auto-match |
| `phone` | нет отдельного поля |
| `username` | нет отдельного поля |
| `telegram_id` named field | нет, используется `dialog_id` |

Вывод: этот export сам по себе почти не дает надежных CRM/Tallanto связей,
кроме числового Telegram/dialog id и имени. Имена нельзя использовать как
strong match. Phone/username coverage появляется только в старых enrichment
артефактах.

## Полезные Legacy / OpenClaw Artifacts

Legacy handoff:

- `_external_handoffs/telegram_bot_legacy_20260509/CHANNEL_ADAPTER_EXTRACTION_PLAN_2026-05-09.md`
- `_external_handoffs/telegram_bot_legacy_20260509/BOT_ENV_CONTRACT_2026-05-09.md`
- `_external_handoffs/telegram_bot_legacy_20260509/MANGO_TRANSFER_AUDIT_2026-05-09.md`
- `_external_handoffs/telegram_bot_legacy_20260509/RENDER_DEPLOYMENT_NOTES_2026-05-09.md`

Что берем:

- Telegram update mapping в `ChannelMessage`;
- Business thread id: `business_connection_id:chat_id`;
- Mini App `web_app_data` как structured event;
- duplicate protection / idempotency;
- draft -> approval -> controlled-send lifecycle;
- webhook queue/retry как будущий adapter boundary pattern;
- startup diagnostics / smoke ideas.

Что не берем:

- legacy bot repo целиком;
- `python-telegram-bot` types/handlers внутрь Mango core;
- старую RAG/vector-store систему;
- старый `.env`, tokens, DB, `data/`, `webapp/dist`;
- прямой Telegram send;
- Telegram-only state machine как общий backend.

OpenClaw/enrichment useful facts:

| Artifact | Полезность |
|---|---|
| `telegram_phone_live_enrichment.jsonl` | structured phone-level join: `phone`, `telegram`, `canonical_contact`, `amo`, `tallanto`, `utility_score` |
| `telegram_live_enrichment_refined.csv` | полный enrichment + OpenClaw draft fields; PII, не коммитить |
| `telegram_openclaw_summary.json` | aggregate counts |
| `telegram_outreach_summary.json/md` | dialog-level coverage and segment counts |
| `telegram_high_utility_drafts_96_summary.json` | high utility count and pack refs |

Aggregate coverage from old enrichment:

| Метрика | Значение |
|---|---:|
| phones_total | 725 |
| matched_to_working_layer | 100 |
| matched_to_amo | 185 |
| matched_to_tallanto | 218 |
| offer_possible | 655 |
| high_utility_phones | 96 |
| dialogs_with_phone | 720 |
| dialogs_with_username | 941 |
| dialogs_phone_or_username | 1321 |
| CRM matched dialogs in outreach | 100 |
| outreach candidates | 314 |
| OpenClaw shortlist dialogs | 76 |

High utility pack:

```text
stable_runtime/audits/telegram_high_utility_drafts_20260416/
```

It has 96 high utility rows and a top-30 slice. CSV/XLSX contents include phone,
CRM/Tallanto context, prompts and drafts, so they remain local ignored artifacts.

## Tallanto Export Check

Источник:

```text
_external_handoffs/tallanto_students_export_2026-05-12/Ученики.csv
```

Encoding: `cp1251`
Rows: 18126

Telegram-related field presence:

| Поле | Наличие |
|---|---:|
| `Telegram ID` | 0 |
| `Telegram` | 8 |
| `Подписан в Telegram` | 18126, all `0` |
| `Тел. цифровой (моб.)` | 18001 |
| `Тел. (родителя)` | 18001 |

Прямой match `telegram_export.dialog_id -> Tallanto.Telegram ID` сейчас дает:

| class | dialogs |
|---|---:|
| strong_unique | 0 |
| ambiguous | 0 |
| unmatched | 1653 |

Вывод: новый Tallanto export полезен для phone identity map, но не для прямого
Telegram ID linking. Для Telegram coverage пока нужны phone enrichment/OpenClaw
артефакты или новый Telegram export с contacts.

## Identity Map Design

Наблюдение Telegram:

```text
channel_thread_id
telegram_user_id
username
phone
display_name_present
source_refs
```

Candidate customer:

```text
customer_id
source_system: mango | amocrm | tallanto | working_layer
phones[]
telegram_user_ids[]
telegram_usernames[]
source_refs[]
metadata
```

Правила:

| Evidence | Match class |
|---|---|
| один candidate по phone и/или telegram_user_id | `strong_unique` |
| один candidate только по username | `strong_unique`, lower confidence |
| несколько candidates по одному или нескольким evidence keys | `ambiguous` |
| только name/display name | `unmatched`, flag `name_only_not_matched` |
| нет evidence | `unmatched` |

Confidence:

- phone + telegram_user_id: 0.98;
- phone: 0.96;
- telegram_user_id: 0.94;
- username: 0.78;
- ambiguous: 0.45;
- unmatched: 0.0.

Conflict flags:

- `phone_conflict`;
- `telegram_user_id_conflict`;
- `username_conflict`;
- `multiple_candidate_customers`;
- `evidence_disagreement`;
- `name_only_not_matched`.

## Unified Customer Timeline Output

Telegram history message event:

```text
event_type: telegram_message
source_system: telegram_history
source_id: ChannelMessage.idempotency_key
source_ref: telegram_history:{export_id}:messages.jsonl:{line}
event_at
direction
channel_thread_id
channel_message_id
actor_ref
attachment_count
identity_link
confidence
conflict_flags
```

Default timeline preview is text-redacted. A local product archive can opt into
text preview later, but reports stay aggregate-only.

Telegram identity link record:

```text
event_type: telegram_identity_link
source_system: telegram_history
channel_thread_id
match_class
candidate_customer_ids
confidence
evidence_keys
conflict_flags
source_refs
```

Telegram dialog event for the future:

```text
event_type: telegram_dialog
source_system: telegram_history
channel_thread_id
first_message_at
last_message_at
message_count
direction_counts
identity_link
```

## Storage

For real personal Telegram history, use only local ignored paths:

```text
product_data/channel_archive/
```

This path is now ignored in `.gitignore`. SQLite files are also ignored by the
existing `*.sqlite` / `*.db` rules.

Safe local DB target example:

```text
product_data/channel_archive/telegram_history_channel.sqlite
```

`ChannelSQLiteStore` still rejects `stable_runtime` paths and runtime-looking DB
names. It persists `ChannelMessage.text`, so the DB is a private artifact and
must not be committed.

## Development Stages

### Stage 1. Read-only History Adapter

Done:

- inventory historical export;
- parser `messages.jsonl -> ChannelMessage`;
- media-only placeholder attachments;
- skip empty/no-media messages;
- idempotent import to `ChannelSQLiteStore`;
- tests for parsing, idempotency, duplicates, no API/send/CRM writes.

### Stage 2. Identity Map Import

Done as skeleton:

- generic `CustomerIdentityRecord`;
- Telegram observations from dialogs;
- Tallanto CSV reader with `Telegram ID`, `Telegram`, phone fields;
- strong/ambiguous/unmatched classes.

Next:

- add AMO/working-layer identity record readers from safe read-only snapshots;
- add OpenClaw phone-enrichment reader for aggregate/local reports;
- emit conflict report under `product_data/channel_archive/`.

### Stage 3. Timeline Bridge

Next:

- write `telegram_message` and `telegram_identity_link` events into the future
  `customer_timeline.sqlite`;
- keep raw Telegram DB separate from canonical timeline;
- include only source refs and redacted previews in shared reports.

### Stage 4. Bot Context / Draft Preview

Next:

- use linked Telegram events as context for `ChannelPreviewService`;
- never call legacy RAG;
- draft replies require manager review;
- recommended actions target common action contracts, not Telegram-specific code.

### Stage 5. Controlled Send Design

Future only:

- explicit approval gate;
- Telegram Bot API abstraction outside Mango core;
- webhook secret validation;
- idempotent send log;
- separate dev/prod bot tokens;
- manual production approval before any live send.

## Non-Commit Artifacts

Do not commit:

- `telegram_exports*/`;
- `stable_runtime/audits/telegram_*` CSV/XLSX/JSONL;
- `_external_handoffs/tallanto_students_export_2026-05-12/Ученики.csv`;
- local `product_data/channel_archive/`;
- any SQLite DB with Telegram text;
- tokens, `.env`, webhook secrets, CRM/Tallanto credentials.

Safe to commit:

- source code under `src/mango_mvp/channels/`;
- focused tests under `tests/`;
- this architecture/runtime plan;
- `.gitignore` rule for local channel archive.

## Next Step

The next useful implementation slice is a local matching report builder that
combines:

```text
telegram_exports (2)/local_vm_2024-04-01
stable_runtime/audits/telegram_phone_live_enrichment_20260416/telegram_phone_live_enrichment.jsonl
_external_handoffs/tallanto_students_export_2026-05-12/Ученики.csv
```

It should write only aggregate JSON/Markdown under
`product_data/channel_archive/`, with no message texts, no prompts/drafts and no
CRM/Tallanto writes.
