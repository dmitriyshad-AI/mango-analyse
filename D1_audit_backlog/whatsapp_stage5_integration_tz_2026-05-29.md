# ТЗ Этап 5 — врезка WhatsApp в боевой customer timeline. Claude #2, 2026-05-29

Адресат: Codex / Claude #1. Это read-only спецификация. Код, тесты и коммиты — за Codex
после решения Дмитрия. Я (Claude #2) ничего в боевом коде не менял.

## 0. Главный вывод (разворот относительно первоначального плана)

Изучил боевой код `src/mango_mvp/customer_timeline/`. **Отдельный провайдер
`whatsapp_context_provider.py` (мой Этап 3) для боевой врезки не нужен как первичный путь.**
В пайплайне уже есть всё необходимое:

- `ingestion.ChannelMessageNormalizer` — превращает сообщение канала (telegram/max/web)
  в события timeline;
- `ingestion.load_sqlite_source_records(...)` — читает источник-SQLite строго read-only
  (`mode=ro`, `PRAGMA query_only=ON`, запрет write-ключевых слов);
- `context_provider.get_customer_context_for_phone(...)` — уже отдаёт контекст по телефону
  из единой `customer_timeline.sqlite`.

Значит WhatsApp надо завести как **ещё один канал в ту же базу timeline**, после чего контекст
по нему отдаётся существующим провайдером автоматически. Это прямое следование правилу
CLAUDE.md «не изобретай параллельную инфраструктуру / используй то, что уже есть».

Мой `whatsapp_context_provider.py` остаётся полезным как автономный быстрый доступ к
`whatsapp_chats.sqlite` (отладка, аналитика), но НЕ как второй боевой timeline.

## 1. Что уже готово (мой вклад, Этапы 1–4.5)

- `product_data/transcripts/whatsapp_chats.sqlite` — нормализованная БД: 4538 чатов,
  61 540 сообщений, роли client/manager/service, `brand_hint`, `client_phone` (=+7…),
  `ts` в ISO. Верифицирована (см. `whatsapp_verification_2026-05-29.md`, вердикт PASS).
- `chat_id` = телефон клиента (98%+), что даёт сильную identity-привязку по телефону.

## 2. Рекомендуемая архитектура врезки

Источник `whatsapp_chats.sqlite` (таблица `messages` + `chats`) → `load_sqlite_source_records`
→ новый `WhatsAppMessageNormalizer` → `TimelineImportService.import_records(dry_run=True)` →
новая дата­рованная `customer_timeline.sqlite` (НЕ перезаписывать canonical_readonly).

Ключевое отличие WhatsApp от текущего `ChannelMessageNormalizer`: **у нас есть телефон.**
Текущий канал-нормализатор делает только `INFERRED` identity по `channel_user_id` (conf 0.6),
без телефона — поэтому его события НЕ склеиваются с звонками/AMO по номеру. Для WhatsApp это
надо исправить: добавить **`phone` IdentityLink (STRONG)**, тогда WhatsApp-события клиента
сядут на ту же карточку, что звонки и AMO/Tallanto. Это и есть главная ценность канала.

## 3. Точные точки правки (хирургически)

| Файл | Правка | Зачем |
|---|---|---|
| `customer_timeline/contracts.py` | + `TimelineEventType.WHATSAPP_MESSAGE = "whatsapp_message"` | сейчас в enum нет WhatsApp (есть telegram/max/web). Единственная правка контракта. |
| `customer_timeline/ingestion.py` | + класс `WhatsAppMessageNormalizer` (source_system="whatsapp_export"); либо ветка `channel="whatsapp"` в `channel_link_type`/`channel_event_type` + явный phone-link | завести WhatsApp как канал и привязать к телефону |
| `scripts/whatsapp_timeline_import.py` (новый) | тонкий загрузчик: `load_sqlite_source_records(whatsapp_chats.sqlite, table="messages", source_system="whatsapp_export", where_sql="is_service_message=0")` → `TimelineImportService`, по умолчанию dry-run | воспроизводимый импорт |
| `tests/test_customer_timeline_*` | новые тесты (см. §6) | критерии выхода |

Нормализатор WhatsApp ОБЯЗАН использовать те же `safe_phone`/`normalize_phone` и
`normalize_identity_value("phone", …)`, что и `MangoCallSummaryNormalizer`, иначе ключ телефона
не сойдётся со звонками. Не вводить свою нормализацию.

## 4. Соответствие полей (whatsapp_chats.sqlite → timeline)

| Источник (messages/chats) | Поле события timeline |
|---|---|
| `chats.client_phone` (=chat_id, +7…) | identity link `phone` (STRONG) + customer.primary_phone |
| `messages.ts` | `event_at` (ISO, UTC-приведение делает `parse_source_datetime`) |
| `messages.role` (client/manager) | `direction` (inbound/outbound) + участник client/manager |
| `messages.text` | `text_preview`/`summary` (compact_text, лимит) + BotContextChunk |
| `messages.brand_hint` / `chats.brand_hint` | **tenant_id** (см. §5) — НЕ кладётся в клиентский текст |
| `chats.chat_id` | `source_id`/`source_ref` = `whatsapp:<phone>:<msg_id>` |

`allowed_for_bot=False`, `requires_manager_review=True` для всех чанков — как уже сделано в
`ChannelMessageNormalizer`. На пилоте бот эти данные напрямую клиенту не отдаёт.

## 5. Бренд-безопасность (критично, CLAUDE.md)

`tenant_id` в пайплайне разделяет бренды (по умолчанию "foton"). Правило маппинга строгое:

- `brand_hint = foton` → `tenant_id="foton"`;
- `brand_hint = unpk` → `tenant_id="unpk"`;
- `brand_hint = mixed` или `null` → **НЕ присваивать брендовый tenant**. Импортировать в
  нейтральный `tenant_id="whatsapp_unassigned"` (только manager-review) либо пропускать.
  active_brand задаётся каналом, а не угадывается — поэтому угадывать бренд для mixed/null
  запрещено.

Контрольный смысл: ни одно UNPK-сообщение не должно попасть в foton-tenant и наоборот; mixed
никогда не уходит в брендовый tenant автоматически. Это проверяется отрицательным тестом (§6).

Рекомендация по умолчанию (не критично, могу решить сам): mixed/null → `whatsapp_unassigned`,
manager-review, до отдельного решения Дмитрия о ручной разметке. 917 mixed и 1817 null НЕ
вливать в брендовые карточки без подтверждения.

## 6. План тестов (вход/выход + контрольный отрицательный)

Тест входа (до старта): зелёный существующий `test_customer_timeline_ingestion.py`,
`whatsapp_chats.sqlite` верифицирована (PASS).

Тесты выхода (должны стать зелёными):

1. **Phone-join (позитив):** клиент с телефоном P имеет звонок и WhatsApp →
   `get_customer_context_for_phone(P)` возвращает события ОБОИХ источников на одной карточке.
2. **Бренд-изоляция (контрольный отрицательный):** UNPK-чат после импорта НЕ виден в
   `tenant_id="foton"`; mixed-чат НЕ попал ни в foton, ни в unpk tenant. Тест должен ПАДАТЬ,
   если кто-то ослабит маппинг §5.
3. **Идемпотентность:** повторный импорт того же снапшота не плодит дубли (upsert,
   `source_unchanged=True`).
4. **Dry-run по умолчанию:** без `--apply` боевая БД не пишется.
5. **Read-only контракт:** импорт не пишет в CRM/Tallanto/stable_runtime; источник открыт
   `mode=ro` (проверка `timeline_ingestion_safety_contract`).
6. **Backward-compat enum — ПРОВЕРЕНО Claude #2 (read-only):** добавление значения enum
   `WHATSAPP_MESSAGE` и необходимо, и достаточно. Факты: схема `timeline_events.event_type` —
   `TEXT NOT NULL` БЕЗ CHECK-ограничения (`store.py:381`); store сохраняет `event.event_type.value`
   (`store.py:694, 1661`); выборки по типу параметризованы через `normalize_key` без белого
   списка (`store.py:1126, 1899`). Единственный гейт — `event_type = TimelineEventType(self.event_type)`
   в `contracts.py:394`: незарегистрированная строка падает с ValueError. Поэтому новый тип
   обязателен в enum и после добавления полностью совместим. Тест на запись/чтение всё равно
   оставить как регресс-страховку.

## 7. Чего НЕ делать

- не перезаписывать `product_data/customer_timeline/canonical_readonly_20260521_v5/` —
  это read-only канон; импорт WhatsApp идёт в новую датированную БД;
- не трогать `stable_runtime`;
- не ставить `allowed_for_bot=True` для WhatsApp на пилоте;
- не включать авто-слияние identity (`identity_conflicts_auto_merge=False` оставить);
- не угадывать бренд для mixed/null;
- не менять мою `whatsapp_chats.sqlite` как источник (только чтение).

## 8. Порядок (с тестом выхода на каждом шаге)

1. ~~Подтвердить, что новый event_type безопасен~~ — уже проверено Claude #2 (§6.6): безопасно.
2. Добавить enum `WHATSAPP_MESSAGE` + `WhatsAppMessageNormalizer` + phone-link (STRONG).
3. Загрузчик-скрипт (dry-run по умолчанию).
4. Тесты §6 (включая контрольный отрицательный §6.2) — зелёные.
5. Прогон dry-run на полном снапшоте, ручная сверка 10–15 карточек (звонок+WhatsApp на
   одном телефоне; бренд-изоляция).
6. Только после «ок» Дмитрия — `--apply` в новую БД.

## 9. Открытый вопрос Дмитрию (не блокирует написание кода)

Маршрут mixed/null: нейтральный `whatsapp_unassigned` manager-review (рекомендую) или пропуск.
По твоей установке «несрочное решаю сам» — по умолчанию беру нейтральный tenant, в брендовые
карточки mixed/null не вливаю. Перед `--apply` подтверди.
