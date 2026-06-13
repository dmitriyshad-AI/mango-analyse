# ТЗ-22 (D7 «Расшифровки для CRM»): починка текстового слоя CRM по итогам аудита 13.06

Дата: 2026-06-13. Постановка: Claude (архитектор трека «действующие клиенты») + аудит вторым архитектором. Исполнитель: Codex (диалог «D7. Расшифровки для CRM»). Опора: `audits/_inbox/crm_layer_audit_2026-06-13/REPORT_crm_layer_audit_2026-06-13.md` (полный отчёт + частные дорожки `parts/`).

Все проблемы ниже **перепроверены лично по сырью** (файл+строка указаны). Гипотез в этом ТЗ нет — то, что не подтвердилось, исключено.

---

## 0. Роль, дисциплина, границы (читать обязательно)

- Это **только код**, без живой записи в AMO/Tallanto. Ни один шаг не пишет в боевые системы. Чистка испорченного контакта 76062310 — отдельно, рукой Дмитрия.
- **Каждая правка — за флагом, default OFF**, кроме случаев, явно помеченных «default ON» (там, где старое поведение и есть баг). У каждой правки — **контрольный отрицательный тест (NEG)**: при OFF-флаге поведение байт-в-байт как сейчас.
- 4 правила Codex: сначала read-only анализ → простота → хирургическая правка → цель = проверяемый критерий + NEG.
- НЕ запускать analyze/ASR/перепрогоны. НЕ менять исходные YAML, stable_runtime, измерительные машины. НЕ делать git reset/checkout/clean.
- Снимок базы для тестов профилей: `product_data/customer_profiles/tz16_profiles_v7_20260612/` (только чтение, `mode=ro`).
- Каждый блок — отдельный коммит с зелёными тестами; полный `pytest` гонять самому, не подмножество.

## 1. Состав работ (по приоритету)

| ID | Приоритет | Что | Файл (проверено) | Флаг |
|---|---|---|---|---|
| A1 | P0 | Бренд-фильтр живой карточки не работает (active_brand не прокинут) | `deal_dossier.py:215`, `deals.py:857,830` | `CRM_LIVE_CARD_BRAND_FAILCLOSED` (**OFF**, реш. Дмитрия 13.06) + часть 1-бот ОБЯЗАТЕЛЬНА |
| A2 | P0 | Гейт не ловит служебные/тест-строки в клиентском тексте | `quality/crm_text_quality_detector.py:273` | default ON (новый детект) |
| B1 | P1 | Возражение >90 символов выбрасывается целиком | `deal_aware/deal_text_builder.py:1305` | `CRM_OBJECTION_COMPACT` (ON) |
| B2 | P1 | Итог «Авто история» не ограничен MAX_AUTO_HISTORY_CHARS | `scripts/write_amo_ready_contacts.py:186,329` | `CRM_AUTO_HISTORY_HARD_LIMIT` (ON) |
| B3 | P1 | `normalize_manager_text` молча стирает маркер обрезки | `deal_aware/deal_text_builder.py:1091` | `CRM_KEEP_TRUNCATION_MARK` (OFF) |
| B4 | P2 | Жёсткие срезы не по границе слова | `deal_text_builder.py:785` (`short_evidence`), `:1248` | в составе B-блока |
| C1 | P1 | Дубль-дети: безымянные слоты не сливаются | `customer_profile/builder.py:594` | `PROFILE_CHILD_MERGE_BY_TRAIT` (OFF) |
| D1 | P2 | N+1 запросов к Tallanto в живой карточке | `tallanto_api.py:528,546,409` | `TALLANTO_BATCH_FETCH` (OFF) |
| D2 | P2 | N+1 `fetch_lead` в цикле по сделкам | `amocrm_runtime/deals.py:937` | `AMO_LEADS_BATCH_FETCH` (OFF) |
| E1 | P3 | 5 реализаций нормализации телефона | см. §E1 | рефактор, без флага (поведение сохранить) |
| E2 | P3 | Полный скан таблицы при поиске по телефону | `customer_profile/crm_summary.py:169`, `store.py:44` | `PROFILE_PHONE_INDEX` (OFF) |
| E3 | P3 | Захардкоженная дата 2026-05-13 в дефолтах | `deal_quality_gate.py:84`, `deal_text_builder.py:126,631`, `deal_writeback.py:37` | без флага (дефолт → None+явная передача) |
| E4 | P3 | analyze не пишет модель/версию-промпта/факт обрезки | `services/analyze.py:699` | аддитивно в analysis_json |
| E5 | P3 | Мёртвый код и дубль-скрипты | см. §E5 | только пометка, без удаления |

**Вне зоны этого ТЗ (данные, отдельные ТЗ с приёмкой):** перепрогон 57 blacklist (TZ-20), вливание хвоста 3439 (TZ-21), пересборка профилей, смена дефолта analyze compact→full (решение Дмитрия). Здесь — только код.

---

## БЛОК A — P0

### A1. Бренд-фильтр живой карточки (главный риск проекта)

**Проблема [ПРОВЕРЕНО].** `build_deal_dossier` зовёт `build_live_tallanto_context` без аргумента `active_brand` (`deal_dossier.py:215-219`). Внутри (`tallanto_context.py:261`) стоит `if active and scope not in {active, "shared"}` — при `active_brand=None` переменная `active=""`, условие короткозамыкается, и **проверка бренд-мисматча никогда не выполняется** (строки 261 и 263 обе пропускаются). Карточка строится по филиалу как есть. Это нарушение главного правила «бренды не смешиваются».

**Важно по влиянию [ПРОВЕРЕНО].** Живые запросы к Tallanto идут только при `CRM_TALLANTO_MODE=http` (дефолт `mock`, `config.py:228`; при `mock` `build_live_tallanto_context` возвращает `disabled`). Значит сегодня утечки нет — но баг сработает в ту же секунду, когда http включат в бою. Это **латентная P0: закрыть ДО включения http**.

**Корень.** Бренд канала (его знает потребитель — цикл черновиков/бот) не доводится по цепочке `_build_dossier_and_analysis (deals.py:830) → build_deal_dossier → build_live_tallanto_context`.

**Правка (две части).**

Часть 1 — прокинуть `active_brand` по цепочке (пломбировка, поведение без брендового аргумента не меняется):

```python
# deal_dossier.py — build_deal_dossier(...)
def build_deal_dossier(
    *,
    phone_context: PhoneContext,
    contact: dict[str, Any],
    lead: dict[str, Any],
    notes: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    pipeline_name: str,
    status_name: str,
    user_map: dict[int, str],
    active_brand: str | None = None,          # <-- НОВОЕ
    transcript_excerpt_chars: int = 2200,
    max_transcript_calls: int = 8,
) -> dict[str, Any]:
    ...
    tallanto_live = build_live_tallanto_context(
        phone=phone_context.phone,
        tallanto_id=phone_context.tallanto_id,
        tallanto_match_status=phone_context.tallanto_match_status,
        active_brand=active_brand,            # <-- НОВОЕ
    )
```

```python
# deals.py — _build_dossier_and_analysis(...) добавить параметр active_brand: str | None = None
# и проброс active_brand=active_brand в build_deal_dossier.
```

**ЧЕСТНО про phone-вход (правка ревью).** Три вызова `_build_dossier_and_analysis` (`deals.py:1050,1488,1506`) приходят из REST-роутера `routers/deals.py` строго по телефону; `PhoneContext` поля бренда НЕ имеет — взять `active_brand` там негде, он останется `None`. Поэтому проброс по этой ветке — только пломбировка на будущее, НЕ решение P0. Реальный источник бренда — публичный бот (ниже). Не выдавать part-1-deals за защиту.

Часть 1-бот (ГЛАВНАЯ И ОБЯЗАТЕЛЬНАЯ — решение Дмитрия 13.06 о выключенной страховке делает её ЕДИНСТВЕННОЙ защитой брендов; см. часть 2). Прокинуть бренд по брендо-осведомлённому пути живой карты `scripts/run_telegram_public_pilot_bots.py`. Бот знает `self.config.brand` (foton/unpk), и бренд уже доходит до префетча (`brand=self.config.brand`, строки 683/729), но до Tallanto не доносится:

```python
# run_telegram_public_pilot_bots.py:1567 — добавить параметр active_brand и прокинуть:
def build_live_tallanto_context_readonly(
    phone: str,
    *,
    tallanto_id: str = "",
    tallanto_match_status: str = "",
    active_brand: str | None = None,          # <-- НОВОЕ
) -> dict[str, Any]:
    ...
        payload = build_live_tallanto_context(
            phone=phone,
            tallanto_id=tallanto_id or None,
            tallanto_match_status=tallanto_match_status or None,
            active_brand=active_brand,         # <-- НОВОЕ
            max_related_records=20,
        )
# и на вызове ~1332 передать active_brand из brand префетча (он уже в области видимости как
# self.config.brand / параметр brand функции построения read-only контекста — прокинуть до сюда).
```

Часть 2 — предохранитель «fail-closed» в источнике. Флаг `CRM_LIVE_CARD_BRAND_FAILCLOSED`, **default OFF (решение Дмитрия 13.06)** — на пилоте важнее полные карточки. Реализовать флаг ВСЁ РАВНО нужно (как аварийный рубильник), просто по умолчанию выключен. ВНИМАНИЕ (правка ревью): `brand_scope_from_filial` (`tallanto_context.py:60-71`) возвращает только `unpk / shared / unknown / skip_shd` — **`"foton"` не возвращается никогда** (Фотон-филиалы → `unknown`). Поэтому условие НЕ должно ссылаться на `"foton"`, иначе предохранитель пропустит Фотон. Правильно — блокировать при пустом бренде ЛЮБОЙ scope (skip_shd уже отсечён выше):

```python
# tallanto_context.py — добавить в начало файла: import os
# внутри build_tallanto_live_card(...) (это РЕАЛЬНОЕ имя, НЕ build_one_card),
# сразу после строки `active = _safe_text(active_brand).casefold()` и проверки skip_shd:
    fail_closed = os.getenv("CRM_LIVE_CARD_BRAND_FAILCLOSED", "0") == "1"   # default OFF (реш. Дмитрия 13.06)
    if fail_closed and not active:
        # бренд канала не подтверждён — брендовые живые данные не отдаём
        # (skip_shd уже обработан выше; сюда попадают unpk/shared/unknown)
        return _no_card("brand_unverified", matched_via=matched_via,
                        contacts_found=1, brand_scope=scope, skipped=skipped)
    if active and scope not in {active, "shared"}:
        return _no_card("brand_mismatch", matched_via=matched_via,
                        contacts_found=1, brand_scope=scope, skipped=skipped)
    # ... далее существующая логика ...
```

**Решение Дмитрия (зафиксировано 13.06): `CRM_LIVE_CARD_BRAND_FAILCLOSED` = OFF по умолчанию** — на пилоте приоритет у полноты карточки. 

**Следствие, которое Codex обязан выдержать:** при выключенной страховке защита от смешения брендов держится ИСКЛЮЧИТЕЛЬНО на проверке `brand_mismatch` (строка `if active and scope not in {active, "shared"}`), а она срабатывает ТОЛЬКО когда бот реально передал `active_brand`. Поэтому **часть 1-бот (проброс бренда) — обязательна и является блокером перед включением `CRM_TALLANTO_MODE=http` в бою**. Без части 1-бот и с выключенной страховкой http включать нельзя — иначе вернётся ровно та утечка, которую закрываем. Эту строку вынести в раздел приёмки.

**NEG-тесты (имя функции — `build_tallanto_live_card`):**
1. `mode=mock` → `disabled` (флаг не влияет; ранний выход по mode остаётся).
2. `mode=http`, `active_brand="unpk"`, филиал mfti(unpk) → карточка отдаётся (scope==active).
3. `mode=http`, `active_brand="foton"`, филиал mfti(unpk) → `brand_mismatch`.
4. `mode=http`, `active_brand=""/None`, филиал mfti(unpk), флаг ON → `brand_unverified`, карточки нет.
5. `mode=http`, `active_brand=""/None`, филиал «онлайн»(shared), флаг ON → `brand_unverified` (без бренда не отдаём даже shared).
6. `mode=http`, `active_brand=None`, флаг OFF (ДЕФОЛТ) → карточка по scope — фиксируем «OFF = байт-в-байт старое поведение».
7. **Часть 1-бот (ключевой тест защиты при дефолтном OFF):** бот с `config.brand="foton"`, контакт-филиал unpk, http → итог `brand_mismatch` (бренд дошёл до источника, чужая карточка НЕ выдана).
8. То же, что 7, но `config.brand="unpk"`, филиал mfti(unpk) → карточка отдаётся (свой бренд).

**Критерий:** (а) тесты №3 и №7 доказывают, что при ПЕРЕДАННОМ бренде чужая брендовая карточка блокируется (`brand_mismatch`) — это и есть защита при дефолтном OFF; (б) часть 1-бот реально доносит бренд (тест №7 зелёный); (в) флаг ON (тесты 4/5) работает как аварийный рубильник; (г) OFF воспроизводит старое поведение.

### A2. Гейт ловит служебные/тест-строки

**Проблема [ПРОВЕРЕНО ЖИВЫМ AMO].** В боевом поле «Авто история общения» контакта 76062310 лежит «Второй боевой smoke test из AI Office… [smoke-test, match-status, ai-priority | Тестовый ИИ]». Сплошной проход по снимку 13 895 контактов: такой мусор ровно один, но защиты на запись нет.

**Корень.** Детектор качества `detect_crm_text_quality_risks` (`crm_text_quality_detector.py:273`) не проверяет наличие служебных/тестовых маркеров в клиентских полях.

**Правка.** Новый детект (severity P0/блокирующий), default ON. Добавить в модуль и подключить в общий проход:

```python
# crm_text_quality_detector.py
# ВАЖНО (правка ревью раунд-2): regex БЕЗ ветки «тестов… + истори», иначе ловит
# людскую заметку «Тестовая история» как ложный P0. Только однозначные служебные маркеры,
# которые в реальной порче шли вместе (smoke test / AI Office / Тестовый ИИ / match-status / ai-priority).
SERVICE_TEST_MARKERS_RE = re.compile(
    r"smoke[\s\-]*test|\bai\s*office\b|тестовый\s+ии"
    r"|match-status|ai-priority|\bplaceholder\b|\blorem\b|\bfoobar\b|\bdummy\b",
    re.IGNORECASE,
)

def _detect_service_test_markers(payload: object) -> list[CrmTextQualityFinding]:
    findings: list[CrmTextQualityFinding] = []
    # Скоуп СТРОГО по TARGET_CRM_TEXT_FIELDS (НЕ через _iter_target_text_fields:
    # тот добавляет любое поле с «истори»/«AI» в имени и затянул бы ручное «История общения»).
    if not isinstance(payload, Mapping):
        return findings
    for field in TARGET_CRM_TEXT_FIELDS:
        value = _safe_text(payload.get(field))
        if not value:
            continue
        text = value
        m = SERVICE_TEST_MARKERS_RE.search(text)
        if m:
            findings.append(CrmTextQualityFinding(
                class_id="Q-service-marker", risk_type="service_test_marker",   # литерал: _class_id_of НЕ существует (правка ревью)
                severity="P0", field=field, matched_text=m.group(0)[:60],
                reason="Служебный/тестовый текст в клиентском поле — нельзя выгружать в бой",
            ))
    return findings

# внутри detect_crm_text_quality_risks(...) добавить:
    findings.extend(_detect_service_test_markers(payload))
```

Существование символов (проверено ревью): `TARGET_CRM_TEXT_FIELDS` содержит только АВТО-поля («Авто история общения», «AI-история общения», «Краткая история общения» и др.), ручного «История общения» там НЕТ (проверено по коду). **Почему важен строгий scope (правка ревью раунд-2):** общий `_iter_target_text_fields` (`crm_text_quality_detector.py:931-936`) дополнительно отдаёт ЛЮБОЕ поле, в имени которого есть «AI» или «истори» — то есть затянул бы и ручное «История общения». Поэтому наш детект скоупится строго по кортежу `TARGET_CRM_TEXT_FIELDS`, а не через общий итератор. Гейт реально стоит на пути записи (`write_amo_ready_contacts.py:620`, `min_severity="P2"`). Убедиться, что `_allowed_severities` знает "P0" (подтверждено: `P0:0` в шкале) — иначе P0-finding выпадет из блокирующих.

**NEG-тесты:**
1. «…smoke test из AI Office… [match-status, ai-priority | Тестовый ИИ]» в поле «Авто история общения» → finding P0, `has_blocking_...`=True.
2. Чистый клиентский текст → 0 findings.
3. «дз по мат в виде тестов» (живая фраза) → 0 findings (маркеры не ловят подстроку «тест»).
4. «Тестовая история» в РУЧНОМ поле «История общения» → 0 findings — по двум причинам сразу: (а) детект скоупится строго по `TARGET_CRM_TEXT_FIELDS`, куда ручное поле не входит; (б) regex БЕЗ ветки «тестов…+истори» (её убрали) — даже если такой текст попадёт в авто-поле, ложного P0 не будет. Проверить оба условия отдельными тестами.
5. «Тестовая история» в авто-поле → 0 findings (regex сужен, см. п.4б).

**Критерий:** smoke-контакт (76062310) заблокировался бы на выгрузке; ноль ложных на живых сводках и на ручном поле (проверить на 50 строках снимка `product_data/customer_profiles/tz14_amo_step1_full_20260612/amo_contacts_raw.jsonl`).

---

## БЛОК B — P1/P2 текст карточки

### B1. Возражения длиннее 90 символов не выбрасывать

**Проблема [ПРОВЕРЕНО].** `normalize_objection` (`deal_text_builder.py:1306`): `if len(text) > 90: return ""` — развёрнутое возражение исчезает целиком. Для продаж это ключевой сигнал.

**Правка.** Флаг `CRM_OBJECTION_COMPACT` (рекомендую ON). ВНИМАНИЕ (правка ревью): НЕ использовать `fit_text` — он клеит поле-суффикс « Текст сокращен до лимита поля.» (29 симв.) и ищет границу предложения только при `last_stop>=160`, что при лимите 90 не сработает → получится нелепое возражение с суффиксом про «лимит поля». Нужен отдельный аккуратный сжиматель по границе слова с многоточием:

```python
def compact_objection(text: str, *, max_chars: int = 90) -> str:
    if len(text) <= max_chars:
        return text
    budget = max(20, max_chars - 1)            # место под «…»
    chunk = text[:budget].rstrip()
    cut = max(chunk.rfind(" "), chunk.rfind(","), chunk.rfind(";"))
    if cut >= int(budget * 0.6):
        chunk = chunk[:cut]
    return chunk.rstrip(" ,;:.") + "…"

def normalize_objection(value: Any) -> str:
    text = normalize_manager_text(value).strip(" .;:").casefold()
    if not text:
        return ""
    # ... словарные замены (цена/время/доверие/неактуально) без изменений ...
    if len(text) <= 2 or text.isdigit():
        return ""
    if len(text) > 90:
        if os.getenv("CRM_OBJECTION_COMPACT", "1") == "1":
            return compact_objection(text, max_chars=90)   # сжать, не терять
        return ""                                          # старое поведение при OFF
    return text
```

(`import os` в `deal_text_builder.py` — проверить наличие; если нет, добавить.)

**NEG:** возражение 150 символов при OFF → "" (старое); при ON → непустой текст ≤90, заканчивается на границе слова с «…», БЕЗ суффикса про лимит поля. Короткие словарные возражения («цена») — без изменений.
**Критерий:** ни одно длинное возражение не теряется при ON; конец — целое слово; OFF = старое.

### B2. Применить MAX_AUTO_HISTORY_CHARS к итогу

**Проблема [ПРОВЕРЕНО].** `MAX_AUTO_HISTORY_CHARS=1600` объявлена (`write_amo_ready_contacts.py:70`), но `_compose_auto_history` (186-226) собирает блоки и возвращает их без лимита; результат пишется в поле (строка 329) без обрезки. Риск: текст превышает размер поля AMO и обрезается уже на стороне CRM — молча и не по границе слова (двойная обрезка).

**Правка.** Флаг `CRM_AUTO_HISTORY_HARD_LIMIT` (рекомендую ON). Заменить финальный `return "\n\n".join(...)` (`write_amo_ready_contacts.py:226`) на переменную и применить `_compact_without_ellipsis` (он уже режет по границе слова и ставит «[сжато]», `import os` в файле есть — строка 8):

```python
    # было: return "\n\n".join(block for block in blocks if block.strip()).strip()
    composed = "\n\n".join(block for block in blocks if block.strip()).strip()
    if os.getenv("CRM_AUTO_HISTORY_HARD_LIMIT", "1") == "1" and len(composed) > MAX_AUTO_HISTORY_CHARS:
        composed = _compact_without_ellipsis(composed, limit=MAX_AUTO_HISTORY_CHARS)
    return composed
```

**Обязательный NEG-«читаю-после-записи» (ловит молчаливую обрезку поля).** Тест: собрать текст из всех заполненных блоков (длинный случай), при ON длина ≤ MAX_AUTO_HISTORY_CHARS; имитация «записал → прочитал» возвращает тот же текст (никакой внешней обрезки). При OFF — длина не ограничена (старое).
**Критерий:** при ON итог никогда не превышает лимит; конец текста — целое слово.

### B3. Не стирать факт обрезки в normalize_manager_text

**Проблема [ПРОВЕРЕНО].** `normalize_manager_text` (`deal_text_builder.py:1091`): `.replace("[сжато]", "").replace("[truncated]", "")` — если текст уже был сжат и маркирован, маркер исчезает при ре-нормализации, читателю не видно, что текст неполный.

**Правка.** Флаг `CRM_KEEP_TRUNCATION_MARK` (default OFF — поведение меняется незаметно, поэтому осторожно). При ON — не вырезать маркер, а схлопнуть к одному виду «…» в конце фрагмента:

```python
def normalize_manager_text(value: Any) -> str:
    keep_mark = os.getenv("CRM_KEEP_TRUNCATION_MARK", "0") == "1"
    text = safe_text(value)
    if keep_mark:
        text = text.replace("[truncated]", "[сжато]")   # унифицировать, не удалять
        text = text.replace("…", " ").replace("...", " ")
    else:
        text = text.replace("…", " ").replace("...", " ")
        text = text.replace("[сжато]", "").replace("[truncated]", "")
    # ... остальное без изменений ...
```

**Единый маркер (правка ревью, согласование B3↔B4).** Маркер обрезки во ВСЁМ слое — один: `[сжато]`. B4 (ниже) не должен вводить второй маркер (« Текст сокращен до лимита поля.» из `fit_text` — это про лимит поля сделки, оставить только там, где он сейчас, не тащить в evidence/историю звонка). 

**ВНИМАНИЕ (правка ревью): `normalize_manager_text` — центральная функция**, её зовут ~16 мест внутри билдера и 7 файлов-потребителей (draft_loop, tenant_normalizer, build_post_backfill и др.). Поэтому NEG обязан доказать **байт-идентичность при OFF по ВСЕМ путям-потребителям**, а не только в билдере (прогнать характеристический тест на каждом потребителе).

**NEG:** текст с «[сжато]» при OFF → маркер удалён (старое поведение, все 7 потребителей байт-в-байт); при ON → маркер сохранён ровно один раз, без дублей; B3 ON + B4 ON совместно → единый маркер `[сжато]`, не два разных.
**Критерий:** при ON наличие обрезки видно в готовом тексте; OFF = старое по всем потребителям.

### B4. Срезы по границе слова (косметика, в составе B-блока)

**Проблема [ПРОВЕРЕНО].** `short_evidence` (`deal_text_builder.py:785`): `text[:151].rstrip() + " [сжато]"` — жёсткий срез без границы слова. `history_call_summary` (`:1248`) при `chunk<80` делает `selected[:max_chars-26]` — тоже жёсткий fallback.

**Правка (правка ревью: НЕ `fit_text`).** `fit_text` приклеивает суффикс « Текст сокращен до лимита поля.» — для evidence/истории звонка это лишнее и ввело бы ВТОРОЙ маркер (конфликт с B3). Завести общий помощник по границе слова с единым маркером `[сжато]` и использовать его:

```python
def _cut_on_word_boundary(text: str, *, limit: int, suffix: str = " [сжато]") -> str:
    if len(text) <= limit:
        return text
    budget = max(20, limit - len(suffix))
    chunk = text[:budget].rstrip()
    cut = max(chunk.rfind(" "), chunk.rfind(","), chunk.rfind(";"), chunk.rfind("."))
    if cut >= int(budget * 0.6):
        chunk = chunk[:cut]
    return chunk.rstrip(" ,;:.") + suffix

def short_evidence(row: dict[str, str]) -> str:
    text = safe_text(row.get("call_summary"))
    if len(text) <= 160:
        return text
    return _cut_on_word_boundary(text, limit=160)   # вместо text[:151] + " [сжато]"
```

Для `history_call_summary` — в fallback заменить жёсткий `selected[:max_chars-26].rstrip(...)` на `_cut_on_word_boundary(selected, limit=max_chars, suffix=". Детали в полном звонке.")` (его собственный осмысленный суффикс — это не маркер обрезки, а отсылка к полному звонку, оставить).

**NEG:** длинный evidence → конец на границе слова с `[сжато]`, не на середине; короткий (≤160) — без изменений; маркер ровно один (согласовано с B3).
**Критерий:** ни один срез текста карточки не рвёт слово посередине; во всём слое единый маркер обрезки `[сжато]`.

---

## БЛОК C — P1 профиль (дубль-дети)

### C1. Склейка безымянных детских слотов

**Проблема [ПРОВЕРЕНО].** `child_slots_match` (`builder.py:594-597`) возвращает True только если у обоих слотов есть имя и имена равны: `bool(left_name and right_name and left_name == right_name)`. Слот без имени (например `child_<hash>`, где hash от класса/числа) никогда не сливается → у одной семьи копятся 13-19 «детей», хотя ребёнок один.

**Корень.** Единственный критерий слияния — имя. Класс/предмет/бренд не используются как вторичный признак для безымянных слотов.

**Факт по коду (правка ревью).** Слот строится в `collect_child_slots` (`builder.py:550`) как `{"names","grades","subjects"}` — **ключа `"brands"` НЕТ**, бренд на уровне детского слота не хранится. Поэтому критерий «бренд» — фикция (`∅==∅` всегда True). Сливаем безымянных по (КЛАСС ∧ ПРЕДМЕТ), без бренда. Также в `builder.py` **нет `import os`** — добавить.

**Правка (осторожная, флаг `PROFILE_CHILD_MERGE_BY_TRAIT`, default OFF).** Сливать только когда у ОБОИХ слотов имени нет И совпадают класс и предмет. Если у одного имя есть, у другого нет — НЕ сливать (риск приклеить разных детей):

```python
# builder.py — добавить в начало: import os
def child_slots_match(left: Mapping[str, set[str]], right: Mapping[str, set[str]]) -> bool:
    left_name = child_slot_name_key(left.get("names", set()))
    right_name = child_slot_name_key(right.get("names", set()))
    if left_name and right_name:
        return left_name == right_name
    if os.getenv("PROFILE_CHILD_MERGE_BY_TRAIT", "0") == "1" and not left_name and not right_name:
        # оба безымянны — слить только при совпадении класса И предмета (бренда в слоте нет)
        left_grades = frozenset(left.get("grades", set()))
        right_grades = frozenset(right.get("grades", set()))
        left_subj = frozenset(left.get("subjects", set()))
        right_subj = frozenset(right.get("subjects", set()))
        return bool(left_grades and left_grades == right_grades and left_subj and left_subj == right_subj)
    return False
```

Риск ложного слияния (два РАЗНЫХ безымянных ребёнка одной семьи, один класс, один предмет) — осознанный и низкий (бренд в слоте отсутствует, разделить их нечем; такие случаи редки и сейчас и так висят дублями). Поэтому флаг default OFF и обязательный замер ниже. Если позже понадобится бренд как признак — это отдельная задача: добавить сбор бренда в `collect_child_slots` (источник бренда на уровне поля профиля надо сперва найти).

**Замер до/после (обязательно).** На снимке v7: число детских слотов на семью до и после флага; показать, что (а) при OFF число не меняется; (б) при ON сливаются только безымянные с одинаковыми признаками; (в) ни один именованный слот не поглощён. Привести 3-5 примеров обезличенно.

**NEG:** два слота с разными именами → не слиты; именованный + безымянный → не слиты; два безымянных одного класса+предмета+бренда при ON → слиты, при OFF → нет.
**Критерий:** раздувание слотов падает, ложных слияний именованных детей — ноль.

---

## БЛОК D — P2 производительность

### D1. N+1 запросов к Tallanto в живой карточке

**Проблема [ПРОВЕРЕНО].** (а) `classes_by_ids` (`tallanto_api.py:528-543`) — отдельный GET на каждый class_id; (б) `build_contact_context` (`:546-575`) — 7 последовательных вызовов на каждый контакт; (в) `search_contacts_by_phone` (`:409-427`) — двойной цикл `CONTACT_PHONE_FIELDS × candidates`, GET на каждую пару, ранняя остановка только при наборе max_records. При нескольких контактах — десятки запросов, риск упереться в лимит коннектора 60/мин и в таймаут карточки.

**Правка (флаг `TALLANTO_BATCH_FETCH`, default OFF).**
- `classes_by_ids`: если API поддерживает выборку по списку id (проверить `get_entries_by_ids`/фильтр `id IN`) — один запрос вместо N. Если не поддерживает — оставить, но добавить кэш в пределах одной сборки карточки (один class_id не запрашивается дважды; сейчас `seen` спасает только от дублей в одном вызове).
- `search_contacts_by_phone`: остановиться на первом поле, давшем контакт (телефон обычно лежит в одном поле); перебор остальных полей — только если первое пусто. Полуфабрикат:

```python
        for field_name in self.CONTACT_PHONE_FIELDS:
            for candidate in candidates:
                ... # как сейчас
                records.extend(_extract_record_list(payload))
                if records and os.getenv("TALLANTO_BATCH_FETCH", "0") == "1":
                    return _dedupe_dicts(records)[:max_records]   # ранний выход
                if len(records) >= max_records:
                    return _dedupe_dicts(records)[:max_records]
```

- `build_contact_context`: не грузить блоки, которые `live_card` не использует. Проверить по `tallanto_context.build_tallanto_live_card` (`tallanto_context.py:242`): реально нужны `finances`, `abonements`, `classes` (через `class_relations`). `opportunities`, `requests`, `course_relations` в live_card не входят — грузить их только для `compact_contexts`-пути, а в live-режиме пропускать (за тем же флагом).

**NEG:** при OFF — число и порядок запросов как сейчас (зафиксировать счётчиком вызовов в моке Tallanto); при ON — меньше запросов, итоговая карточка побайтово та же (данные не изменились, только число обращений).
**Критерий:** при ON число запросов к Tallanto на типовой контакт падает (показать до/после на моке), содержимое карточки идентично.

### D2. N+1 fetch_lead в цикле по сделкам

**Проблема [ПРОВЕРЕНО].** `deals.py:937-940`: `for lead_id in lead_ids: leads.append(fetch_lead(session, lead_id=lead_id, with_fields="contacts"))`. AMO поддерживает пакетную выборку `GET /leads?filter[id][]=...`.

**Правка (флаг `AMO_LEADS_BATCH_FETCH`, default OFF).** Сама `fetch_lead` лежит в `amo_integration.py:1103` (НЕ в `deals.py` — там только её вызов); новую `fetch_leads_batch` писать в `amo_integration.py` рядом. **Сначала подтвердить (правка ревью)**, что AMO отдаёт пакет по `GET /leads?filter[id][]=...` — на доке/моке AMO; без подтверждённого контракта API флаг не включать. Затем использовать вместо цикла:

```python
        if lead_ids:
            if os.getenv("AMO_LEADS_BATCH_FETCH", "0") == "1":
                leads = fetch_leads_batch(session, lead_ids=lead_ids, with_fields="contacts")
            else:
                leads = [fetch_lead(session, lead_id=lid, with_fields="contacts") for lid in lead_ids]
        else:
            leads = fetch_related_leads(session, contact_id=contact_id)
```

**NEG:** при OFF — N запросов как сейчас; при ON — 1 запрос на ≤50 id, результат (множество сделок) совпадает с поэлементной выборкой.
**Критерий:** число запросов к AMO падает, набор сделок идентичен.

---

## БЛОК E — P3 технический долг

### E1. Единый нормализатор телефона

**Проблема [ПРОВЕРЕНО].** 5 реализаций с возможным расхождением: `utils/phone.py:7`, `insights/phone_identity.py:10`, `productization/mail_archive.py:473`, `channels/telegram_history.py:794`, `customer_timeline/context_provider.py:354` (`normalize_phone_for_match`). Телефон — главный ключ склейки семьи; расхождение форматов = тихие промахи матчинга.

**Правка (рефактор без смены поведения).** Сверить поведение всех пяти на наборе номеров (международный, 8/7, со скобками/дефисами, короткий, мусор). Выбрать каноном `utils/phone.py`. Прочие свести к импорту из него; если где-то нужна спец-логика (например `last10`), оставить тонкую обёртку поверх канона. **Сначала характеристический тест**: зафиксировать текущий выход каждой версии на ~30 номерах, затем доказать, что после унификации выход канона совпадает (или осознанно объяснить расхождение).
**NEG:** характеристический тест зелёный — на всех 30 номерах унифицированная функция даёт прежний результат там, где он был верным.
**Критерий:** один нормализатор, остальные — обёртки; ключи сшивки не изменились на снимке.

### E2. Индекс для поиска по телефону в профилях

**Проблема [ПРОВЕРЕНО].** `_profile_ids_by_phone` (`crm_summary.py:169`) делает `SELECT ... FROM customer_profiles ORDER BY profile_id` (полная таблица, ~18К строк) и нормализует каждую строку в Python. Индекса на телефон нет (`store.py:44-52` создаёт индекс только на `profile_fields`).

**Правка (флаг `PROFILE_PHONE_INDEX`, default OFF; это меняет схему — осторожно).** `primary_phone` хранится «как есть» (не нормализован), поэтому обычный индекс не поможет. Добавить **нормализованную колонку + индекс** и поиск по ней:

```sql
ALTER TABLE customer_profiles ADD COLUMN primary_phone_norm TEXT;   -- заполнить при сборке
CREATE INDEX IF NOT EXISTS idx_customer_profiles_phone_norm ON customer_profiles(primary_phone_norm);
```

В сборщике профиля заполнять `primary_phone_norm = normalize_phone(primary_phone)` (канон из E1). Поиск — `WHERE primary_phone_norm = ? OR primary_phone_norm LIKE '%'||?` по last10. Для существующих баз без колонки — fallback на старый полный скан (флаг OFF).
**NEG:** при OFF — старый путь; при ON на новой базе — тот же набор profile_id, что и полный скан, но по индексу.
**Критерий:** поиск по телефону не делает полный скан на индексированной базе; результат идентичен.

### E3. Убрать захардкоженную дату из дефолтов

**Проблема [ПРОВЕРЕНО].** `analysis_date: str = "2026-05-13"` в трёх датаклассах (`deal_quality_gate.py:84`, `deal_text_builder.py:126`, `deal_writeback.py:37`), fallback `datetime(2026,5,13)` (`deal_text_builder.py:631`), токен `WRITE_AMO_DEAL_AWARE_STAGE20_20260513` (`deal_writeback.py:389`). Запуск без явной даты тихо проставит май → «дата следующего касания» уже устарела на месяц.

**Правка (без флага, но аккуратно).** Дефолт `analysis_date` → `None`; при `None` подставлять текущую дату сборки (UTC) явной функцией `resolve_analysis_date()`, чтобы было видно происхождение. Фиксированный fallback `datetime(2026,5,13)` заменить на дату из контекста. Токен stage20 не трогать (исторический идентификатор пройденного шага), но пометить комментарием «# исторический токen прогона 13.05, не использовать для новых записей».
**NEG:** при явно переданной дате — поведение прежнее; при `None` — берётся сегодняшняя, не май.
**Критерий:** новые сборки не проставляют май; тесты, завязанные на 13.05, переведены на явную дату-фикстуру.

### E4. analyze пишет модель/версию-промпта/факт обрезки (аддитивно)

**Проблема [ПРОВЕРЕНО].** При обрезке промпта (`analyze.py:699-707`) середина транскрипта выбрасывается; флаг `truncated` вычисляется, но в `analysis_json` модель/версия-промпта/факт обрезки не фиксируются (известный пробел из STATE-документа). Без этого нельзя разобрать, какой звонок чем посчитан, и где была потеря середины.

**Правка (аддитивно, без смены логики разбора).** В `analysis_json` дописать поля: `analyze_model`, `analyze_prompt_profile`, `analyze_prompt_truncated` (bool), `analyze_prompt_chars`. Это не запускает analyze, а лишь фиксируется при будущих прогонах.
**NEG:** структура `analysis_json` расширяется только новыми ключами; существующие чтения не ломаются (тест парсинга старого и нового json зелёный).
**Критерий:** в новой выжимке видно модель, профиль промпта и факт обрезки.

> Примечание Дмитрию: смена дефолта analyze с `compact` (режет середину, `analyze.py:565`) на `full` (v7) — это **решение Дмитрия + перепрогон зоны использования**, в код этого ТЗ не входит. Здесь только фиксация метаданных, чтобы потом было видно, что чем посчитано.

### E5. Мёртвый код и дубль-скрипты — только пометить

**Проблема [ПРОВЕРЕНО частично].** Пара `scripts/prepare_message_archive_history_full_cycle.py` vs `prepare_message_archives_history_full_cycle.py` (ед./мн. число — вероятный дубль); `scripts/build_student_card_manual_review_pack.py` — вызовов в коде не видно (только в доках). 

**Правка.** НЕ удалять. Прогнать `grep` вызовов по `src/ scripts/ tests/` для каждого подозрительного файла, собрать список «кандидаты в мёртвые» с числом вхождений, положить в `audits/_inbox/crm_layer_audit_2026-06-13/dead_code_candidates.md`. Удаление — отдельным решением Дмитрия.
**Критерий:** список кандидатов с доказательством (0 вызовов вне определения), без удаления.

---

## Порядок работ (рекомендация)

1. **A2** (защита гейта от тест-строк) — мал, ценен, ничего не ломает.
2. **A1** (бренд-фильтр) — критично до включения http; согласовать с потребителем (главный архитектор) проброс active_brand.
3. **B1–B4** (текст карточки) — один логический блок, общий регрейд по сырью.
4. **C1** (дубль-дети) — с замером до/после на снимке v7.
5. **D1, D2** (производительность) — со счётчиком запросов на моках.
6. **E1, E2, E3, E4, E5** (долг) — последними, E1 перед E2 (E2 опирается на канон-нормализатор).

## Общая приёмка ТЗ

- Полный `pytest` зелёный; для каждой правки есть NEG, доказывающий «OFF = старое поведение».
- A1: при http без бренда брендовая карточка не утекает (тест на mismatch + brand_unverified).
- A2: smoke-строка заблокирована, ложных на живых сводках нет (проверка на 50 строках снимка).
- B2: тест «читаю-после-записи» подтверждает отсутствие молчаливой обрезки поля.
- C1: замер слотов до/после, ноль ложных слияний именованных детей.
- D1/D2: число запросов до/после на моке, содержимое идентично.
- Регрейд по сырью (не по сводке) для A1/A2/B/C — обязателен; архитектор (Claude) сверяет независимо.
- Ни одной живой записи в AMO/Tallanto за всё ТЗ.
- **БЛОКЕР перед включением `CRM_TALLANTO_MODE=http` в бою:** часть 1-бот (проброс `active_brand`) смержена и проверена тестом №7 (бот с чужим брендом получает `brand_mismatch`). Поскольку страховка `CRM_LIVE_CARD_BRAND_FAILCLOSED` по решению Дмитрия выключена, это ЕДИНСТВЕННАЯ защита от смешения брендов в живой карточке.

## Решения Дмитрия по флагам (зафиксированы 13.06, обязательны к исполнению)

1. **`CRM_LIVE_CARD_BRAND_FAILCLOSED` = OFF по умолчанию.** На пилоте приоритет — полнота карточки. Флаг реализовать как аварийный рубильник, но выключенным. Защита брендов переносится на часть 1-бот (обязательна, см. блокер выше).
2. **B1 (`CRM_OBJECTION_COMPACT`) и B2 (`CRM_AUTO_HISTORY_HARD_LIMIT`) = ON по умолчанию.** Это починка явных багов; OFF оставить только как аварийный откат.
3. **Остальные флаги (B3, C1, D1, D2, E2) = OFF по умолчанию**, включать последовательно, по одному, после регрейда каждого.

---

## След ревью (для Codex и Дмитрия)

Это ТЗ доведено двумя раундами аудита вторым архитектором (независимый read-only по коду):
- **Раунд 1 — вердикт BLOCKED.** Закрыто: имя функции `build_tallanto_live_card` (было неверное `build_one_card`); условие fail-closed расширено на Фотон (его scope = `unknown`, бренд «foton» не существует в `brand_scope_from_filial`); добавлен РЕАЛЬНЫЙ путь бренда через публичный бот (phone-вход `deals.py` бренда не имеет — помечено как пломбировка, не защита); убран несуществующий `_class_id_of` → литерал; в C1 убран фиктивный критерий «бренд» (ключа нет в слоте) → слияние по (класс ∧ предмет); B1 переведён с `fit_text` на отдельный `compact_objection`; B4 — на `_cut_on_word_boundary` с единым маркером `[сжато]`; добавлены недостающие `import os` (tallanto_context, builder, deal_text_builder); D2 — `fetch_lead` в `amo_integration.py`, добавлен шаг подтверждения контракта AMO.
- **Раунд 2 — вердикт PASS_WITH_NOTES.** Закрыто: A2 скоупится строго по `TARGET_CRM_TEXT_FIELDS` (общий `_iter_target_text_fields` затянул бы ручное «История общения»), regex сужен (убрана ветка «тестов…+истори», ловившая людскую «Тестовая история»), NEG#4 переписан; снята висячая ссылка `build_one_card` в §D1.
- **Статус: финал, готов к исполнению.** Номера строк — ориентир (возможен дрейф на ±10), символы названы однозначно — искать по имени, не по номеру.
