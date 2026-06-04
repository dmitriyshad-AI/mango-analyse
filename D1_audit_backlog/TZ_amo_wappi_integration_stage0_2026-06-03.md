# ТЗ — AMO + Wappi интеграция бота, ЭТАП 0 (черновик в карточке, dry-run). Для Кодекса. 2026-06-03.
<!-- v1. Заменяет TZ_amo_integration_stage0_designfirst (тот был до уточнения про Wappi). Дизайн согласован с Кодексом
     в 2 круга: Wappi=транспорт, amoCRM=рабочее место+контекст, наш сервис=мозг+гейт. -->

Автор: Клод #1. Дизайн уже продуман Кодексом (2 круга) и сошёлся. Это ТЗ на РЕАЛИЗАЦИЮ Этапа 0 (dry-run, без
автоотправки клиенту). **ВАЖНО: класть на принятый тип `9cc70d2b` (worktree `.phase12`) ПОСЛЕ сверки базы Level A
(задача #103), НЕ на главное дерево (оно на 98 коммитов позади).** Полный `pytest tests/` зелёный — критерий коммита.

## Реальная схема (после уточнения Дмитрия про Wappi)
К AMO уже подключён **Wappi** — через него привязаны Telegram- и MAX-аккаунты Фотона и УНПК. Поэтому:
- **Wappi = транспорт** TG/MAX (сырой входящий текст + profile_id + chatId + from + contact_phone/username; и API
  для исходящих — на будущее).
- **amoCRM = рабочее место менеджера и карточка** (сделка, статус, бренд-воронка, история; read-only-контекст; место
  для черновика-ноты).
- **Наш сервис = мозг + безопасность** (planner → rules → composer → авторитетный output gate). Гейт и бренд-логика
  НЕ выносятся ни в Wappi, ни в amoCRM, ни в Salesbot.

```text
Telegram/MAX клиент
  -> Wappi webhook (incoming_message)           [ПЕРВИЧНЫЙ вход: сырой текст раньше и чище, чем amo chat webhook]
  -> наш ingress (быстрый 200 + очередь + идемпотентность)
  -> amoCRM read-only lookup (phone/chat/profile -> lead/contact/deal/pipeline/custom_fields/events)
  -> brand resolver (КАНАЛ Wappi-профиль = источник; amo pipeline = подтверждение; расхождение -> fail-closed)
  -> наш боевой draft-пайплайн (planner -> rules -> composer -> output gate)
  -> draft artifact
  -> ЭТАП 0: локальный dry-run журнал ВСЕГДА; internal draft-note в карточку — ТОЛЬКО после отдельного OK Дмитрия
  -> менеджер правит и шлёт сам через Wappi в amoCRM
```

## Железные принципы (не нарушать)
1. **Мозг и гейт — у нас.** AMO/Wappi/Salesbot — канал и интерфейс, не место безопасности.
2. **Бренд из КАНАЛА (CLAUDE.md правило №1).** active_brand = Wappi `profile_id` (профиль привязан к конкретному
   бренду). amo поле сделки `Организация` — независимое ПОДТВЕРЖДЕНИЕ (pipeline_id НЕ бренд — live-подтверждено D3).
   **Расхождение или неизвестный профиль → fail-closed**:
   черновик НЕ генерим, пишем менеджеру служебную пометку, БЕЗ клиентского текста. Бренды не угадываем и не смешиваем.
3. **Draft-only.** Этап 0 НЕ отправляет клиенту ничего. Только предлагает; шлёт менеджер.
4. **CRM read-only**, кроме ОДНОЙ записи — internal draft-note, и та за флагом + отдельным OK Дмитрия. Без изменения
   полей, без создания сделок, без Tallanto, без Salesbot-логики.
5. **Быстрый ответ вебхуку.** amoCRM ждёт ≤2с; Wappi аналогично. Значит: принять → 200 → очередь → асинхронный воркер.
6. **Идемпотентность.** Дубликаты/ретраи вебхука не должны плодить черновики/ноты.
7. **PII.** Телефон/имя из Wappi — только для контекста и хендоффа; в журнале — с политикой хранения (решает Дмитрий);
   не эхо клиенту; не в CRM-поля.
8. **Никакой DOM-инъекции** в чужой виджет Wappi/amo (хрупко, опасно). «Предзаполнить поле Wappi» официально — только
   если есть документированный API/hook; нет — не закладывать.
9. **Пайплайн = БОЕВОЙ draft-путь** (тот, что использует бот: `subscription_llm` → `dialogue_contract_pipeline`),
   НЕ симулятор `run_telegram_dynamic_client_sim`. Мозг идентичен.
10. **Identity policy C** сохраняется в самом черновике (бот честно «помощник менеджера», если прямо спросят) — это
    делает пайплайн, слой AMO только показывает черновик.

## ФАЗА 0 — live-discovery на ТЕСТОВОМ аккаунте (до продакшн-кода). Нужны доступы Дмитрия.
Кодекс проверяет на реальном тесте (а не по докам), потому что поведение каналов неочевидно:
1. При входящем TG/MAX через Wappi — что приходит и КОГДА: Wappi webhook (`wh_type=incoming_message`, `body`,
   `profile_id`, `chatId`, `from`, `contact_phone/username`)? Параллельно ли amoCRM-событие/сообщение? Есть ли связь с
   `lead_id`?
2. Как СТАБИЛЬНО связать Wappi `chatId/profile_id` с конкретной сделкой amoCRM (по phone/contact?).
3. Появляется ли сообщение, отправленное через Wappi API, обратно в amoCRM-ленте (важно для будущего human-click send
   и метрики).
4. Есть ли в Wappi/amoCRM-виджете ОФИЦИАЛЬНЫЙ «draft/prefill» hook. Если нет — не пытаться (см. принцип №8).
5. Какие Wappi `profile_id` соответствуют Фотону и УНПК (для бренд-карты).
**Выход Фазы 0:** подтверждённые факты (что и откуда приходит) + заполненная бренд-карта. Только потом — Фаза 1.

## ФАЗА 1 — реализация Этапа 0 (dry-run). Полуфабрикаты ниже.

### Полуфабрикат 1 — конфиг бренд-карты (канал authoritative)
```python
# КАНАЛ (Wappi profile) = источник правды по бренду — CLAUDE.md правило №1. Заполнить из Фазы 0.
WAPPI_PROFILE_BRAND = {
    "<foton_tg_profile_id>": "foton",
    "<foton_max_profile_id>": "foton",
    "<unpk_tg_profile_id>":   "unpk",
    "<unpk_max_profile_id>":  "unpk",
}
# ПОПРАВКА по live Фазе 0 (D3): бренд в AMO живёт в ПОЛЕ сделки `Организация` (доп. Филиал/utm), НЕ в pipeline_id
# (две live-сделки разных брендов в ОДНОЙ воронке 10408062/статусе 83489762 → pipeline = процесс, НЕ бренд).
AMO_ORG_FIELD_BRAND = {"unpk_or_mfti": "unpk", "foton": "foton"}   # значения поля `Организация` → бренд (live-подтверждено)
# WAPPI_PROFILE_BRAND: 4 профиля (TG Фотон/УНПК, MAX Фотон/УНПК) — заполнить реальными profile_id (нужен Wappi token/таблица).
```

### Полуфабрикат 2 — ingress: быстрый 200 + очередь + идемпотентность
Стек — существующий в репо async/web (проверить; если нет — минимальный FastAPI, согласовать). Endpoint только
принимает и кладёт в очередь:
```python
def wappi_inbound(payload: dict, headers: dict) -> int:
    if not _valid_source(payload, headers):        # allowlist account/profile + секретный URL (+ X-Signature если есть)
        _log_risk("weak_source", payload); return 200   # не доверяем, но и не падаем
    key = _idempotency_key(payload)
    if _seen(key):                                  # дубликат вебхука → ничего не делаем
        return 200
    _mark_seen(key)
    _enqueue({"text": payload.get("body"), "profile_id": payload.get("profile_id"),
              "chat_id": payload.get("chatId"), "from": payload.get("from"),
              "contact_phone": payload.get("contact_phone"), "raw_id": key})
    return 200                                       # быстро; вся работа — в воркере

def _idempotency_key(p: dict) -> str:
    return f"{p.get('wh_type')}:{p.get('profile_id')}:{p.get('chatId')}:{p.get('message_id') or p.get('time')}"
```
Только реагируем на `wh_type == "incoming_message"` (исходящие/системные игнорируем на Этапе 0).

### Полуфабрикат 3 — brand resolver (двусигнальный, канал-authoritative, fail-closed)
```python
def resolve_brand(*, wappi_profile_id: str, amo_org_field: str | None) -> str | None:
    primary = WAPPI_PROFILE_BRAND.get(wappi_profile_id)        # КАНАЛ (Wappi профиль) = источник правды
    if primary is None:
        return None                                            # неизвестный профиль → fail-closed
    if amo_org_field is not None:                              # ПОДТВЕРЖДЕНИЕ = поле сделки `Организация` (live), НЕ pipeline
        confirm = AMO_ORG_FIELD_BRAND.get(str(amo_org_field).strip().casefold())
        if confirm is not None and confirm != primary:
            return None                                        # РАСХОЖДЕНИЕ брендов → fail-closed (НЕ угадывать)
    return primary
```
`None` → воркер пишет менеджеру служебную пометку «не удалось однозначно определить бренд, нужен ручной разбор», БЕЗ
клиентского черновика. Это прямая защита правила №1 (бренды не смешиваются).
**Live Фаза 0 (D3) подтвердила связку:** телефон/channel-id → AMO contact → `Telegram ID`/`Max User ID` → linked lead →
поле `Организация`. Рабочие read-only маршруты AI Office: `contacts/by-phone`, `leads/by-phone` (live 200/matched). НЕ
выставлены: `contacts/chats`, `events`, `notes` — значит ЧИТАТЬ историю чата/события пока нельзя (нужно расширить AI
Office API read-only ИЛИ прямой AMO read-only токен). Для Этапа 0 это ок: текст входящего берём из Wappi-вебхука, бренд
из profile_id, контекст сделки/бренда — из by-phone lookup; история диалога — позже.

### Полуфабрикат 4 — воркер: read-only контекст + наш пайплайн + журнал + (за флагом) нота
```python
AMO_NOTE_WRITE_ENABLED = env_flag("AMO_NOTE_WRITE_ENABLED", default=False)   # write — только после OK Дмитрия

def process(job: dict) -> None:
    ctx = amo_readonly_lookup(phone=job["contact_phone"], chat_id=job["chat_id"])  # lead/contact/pipeline/fields/events
    brand = resolve_brand(wappi_profile_id=job["profile_id"], amo_org_field=ctx.get("org_field"))  # `Организация`, не pipeline
    if brand is None:
        _journal(job, brand=None, verdict="brand_fail_closed")
        _service_note_for_manager(ctx.get("lead_id"), "Бренд не определён однозначно — нужен ручной разбор.")  # без клиентского текста
        return
    draft = run_production_draft(text=job["text"], active_brand=brand, crm_context=ctx)  # БОЕВОЙ путь subscription_llm
    note_id = None
    if AMO_NOTE_WRITE_ENABLED and draft.client_safe:
        note_id = amo_add_internal_note(ctx["lead_id"], f"🤖 Черновик ({brand}): {draft.text}")  # внутренняя common note
    _journal(job, brand=brand, draft=draft, note_id=note_id, verdict=draft.gate_verdict)
```
`run_production_draft` — обёртка над боевым draft-входом (НЕ симулятор). `crm_context` идёт как СПРАВКА; при конфликте
клиентский текст и KB-факт ПРИОРИТЕТНЕЕ CRM-полей (они могут устареть).

### Полуфабрикат 5 — amo read-only client (OAuth2 + backoff + кэш + лимиты)
```python
class AmoReadOnlyClient:
    # OAuth2: code -> /oauth2/access_token -> access/refresh; токены как СЕКРЕТЫ (token store, не в коде/логах)
    # лимиты amoCRM: ~7 req/s на интеграцию, ~50 req/s на аккаунт → троттлинг + экспоненциальный backoff + кэш
    def lookup(self, *, phone=None, chat_id=None) -> dict:
        # GET /api/v4/leads?query=... + with=contacts; читать custom_fields_values, pipeline_id, status_id, events/notes
        # ТОЛЬКО чтение. Никаких POST/PATCH здесь.
        ...
```

### Полуфабрикат 6 — OAuth token store (секреты)
Хранить access/refresh как секреты (не в репо, не в логах), авто-refresh по истечении. Интерфейс: `get_access_token()`,
`refresh()`. Конкретный бэкенд (env/keyring/secret-manager) — согласовать с Дмитрием.

### Полуфабрикат 7 — internal draft-note writer (ЕДИНСТВЕННАЯ запись; за флагом + OK Дмитрия)
```python
def amo_add_internal_note(lead_id: int, text: str) -> int:
    # POST /api/v4/leads/{lead_id}/notes, note_type=common — ВНУТРЕННЯЯ запись, клиенту НЕ уходит.
    # Вызывается ТОЛЬКО при AMO_NOTE_WRITE_ENABLED. До этого — dry-run журнал.
    ...
```
Замечание: нота — append-only write, rollback грязнее, чем правка поля; поэтому строго за флагом и с журналом note_id.

### Полуфабрикат 8 — журнал (dry-run всегда; основа метрик и регрейда)
Поля на каждое событие: `inbound_raw_id, ts, lead_id, contact_id, brand, channel(profile_id), context_hash,
facts_used[], route, gate_verdict, draft_text, note_id|null, error|null`. Хранение ПДн (срок) — решает Дмитрий.
Журнал — источник для еженедельного регрейда по сырью (Claude) и будущей метрики `unedited_rate`.

### Полуфабрикат 9 — «предотправленные сообщения»: лестница (по убыванию практичности)
- **(Этап 0) Нота «🤖 Черновик» в карточке.** Просто и безопасно. Минус: менеджер копирует руками. Достаточно для старта.
- **(Этап 1) Виджет в правой колонке amoCRM (Web SDK card).** Последний черновик + кнопки «Скопировать», «Обновить»,
  «Отклонить». Лучший UX, и даёт честную метрику действий менеджера (для `unedited_rate`). Спроектировать параллельно,
  НЕ начинать с него.
- **(Этап 1+) Кнопка «Отправить через Wappi» в нашем виджете.** Human-click send во внешний канал — отдельный OK +
  kill-switch + журнал + readback (появилось ли сообщение в amo-ленте, проверка из Фазы 0 п.3). Это уже live-send, не
  Этап 0.
- **ОТКЛОНЕНО: авто-вставка/инъекция в поле Wappi-чата.** Без официального API хрупко (ломается на обновлениях, риск
  отправить не туда). Не закладывать (принцип №8).
**Метрика `unedited_rate`:** через ноту мы знаем «черновик создан», но не «отправлен как есть». Честно: точная метрика
требует виджета с кнопками ИЛИ read-only сверки исходящего Wappi-сообщения с черновиком. На Этапе 0 — приблизительно
(журнал создания), точная — с Этапа 1 (виджет).

## Минимальный scope Этапа 0 (что делаем СЕЙЧАС)
Dry-run ingress + очередь + идемпотентность; OAuth token store; amo read-only client (lead/contact/fields/events);
brand resolver (двусигнальный, fail-closed); draft-генератор через БОЕВОЙ пайплайн; локальный журнал. **После отдельного
OK Дмитрия** — добавить internal draft-note. БЕЗ авто-отправки, БЕЗ правки полей, БЕЗ создания сделок, БЕЗ Tallanto, БЕЗ
Salesbot-логики, БЕЗ виджета (его — Этап 1).

## Будущее (НЕ Этап 0, но дизайнить совместимо): ночной трафик с сайта
Цель Дмитрия — как можно раньше в нерабочее время вести лидов с сайта СРАЗУ в нашего бота. Для этого: сайт → НАШ сервис
напрямую (не «черновик для менеджера»), режим сначала shadow, потом ограниченный live на низкорисковых классах; Wappi —
транспорт, если клиент пришёл в TG/MAX; amoCRM получает запись/сделку постфактум. Наш сервис = мозг+гейт при ЛЮБОЙ точке
входа (вебхук Wappi или прямой сайт) — поэтому пайплайн и гейт держим независимыми от канала. Слой автоотправки и
kill-switch — это Этап 3-4 плана пилота, не сейчас.

## Что нужно от Дмитрия (доступы/решения)
- Субдомен amoCRM, `account_id`, тестовая сделка/контакт по TG и по MAX.
- Список Wappi `profile_id` для TG и MAX + какие из них Фотон, какие УНПК (бренд-карта).
- Подтверждённая карта `pipeline_id/status` → Фотон/УНПК (для подтверждающего сигнала).
- Доступ к настройке Wappi webhook в тестовом режиме.
- OAuth-приложение amoCRM (credentials) + права на read-only нужных сущностей.
- Какие поля сделки читать: класс, предмет, формат, источник, статус оплаты, ответственный менеджер.
- Решение: можно ли писать internal draft-note в ТЕСТОВУЮ карточку (включить флаг `AMO_NOTE_WRITE_ENABLED`).
- Срок хранения локальных логов/ПДн.
- Кто (роль) смотрит черновики утром/в моменте.
- Подтверждение: Этап 0 клиенту ничего не отправляет.

## Риски
- Входящий webhook может не дать полной истории/текста для всех каналов → Фаза 0 проверяет (Wappi-first снижает риск).
- Нота — append-only write; не такой чистый откат, как правка поля → строго за флагом + журнал note_id.
- Дубликаты/ретраи вебхуков с разными правилами → обязательная идемпотентность.
- CRM-поля могут устареть → клиентский текст и KB-факт приоритетнее CRM.
- Без виджета метрика действий менеджера приблизительна → точная с Этапа 1.
- Связка Wappi chatId ↔ amo сделка может быть нетривиальной → Фаза 0 п.2.

## Что НЕ делать
Не выносить гейт/безопасность в AMO/Wappi/Salesbot; не угадывать бренд (расхождение→fail-closed); не автоотправлять
клиенту; не писать в CRM кроме согласованной ноты за флагом; не DOM-инъекция; не Tallanto; не класть на главное дерево
(только на `9cc70d2b` после сверки базы); не кодить продакшн до Фазы 0; полный pytest зелёный; не git reset.
