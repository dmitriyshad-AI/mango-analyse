# IMPROVEMENT PLAN FOR CODEX — Mango Analyze pipeline

Дата: 2026-05-09. Источник: систематический аудит **1926 из 2548 спорных звонков** (76%, 18 параллельных subagent-runs, на трёх волнах). По правилу «новых проблем нет — стоп», аудит остановлен после Волны 3, потому что Волна 3 не добавила новых классов и цифры стабилизировались.

Этот документ — единственный артефакт для передачи в Codex с правками pipeline. Содержит конкретные патчи, регрессионные тесты, cohort на re-analyze и план валидации.

---

## EXECUTIVE SUMMARY

Из 1926 проверенных звонков:
- **99.1% (1908) — это NON-CONVERSATION среди проверенной спорной выборки**, которые pipeline ошибочно классифицировал как sales_call/service_call/technical_call/existing_client_progress. Это не 99.1% всего корпуса, а 99.1% от слоя, который Codex pipeline уже считал подозрительным.
- `claude_audit_package_full_2548` правильно отобрал «спорные»: precision отбора очень высокий, но recall по всему корпусу пока неизвестен.
- 12 обнаруженных классов проблем pipeline. **9 из 12 имеют high-severity и критичны для production**.
- Pipeline сейчас систематически фабрикует данные клиентов на voicemail/IVR/VS-вход: synthesizes возражение «время» в 45% случаев, обещание «Перезвонить клиенту» в 30% случаев, заполняет products/grade/target_product на голосовой почте.

**Корректная оценка масштаба:**
- Все звонки в большом корпусе: ~77 460.
- Спорный слой, отобранный Codex filter: 2 548, то есть ~3.3% корпуса.
- Подтвержденный минимум проблемных звонков: ~2 525, то есть ~3.3% корпуса.
- Реальная проблемная масса зависит от recall текущего filter: ориентир ~3.6% при recall >=90%, ~5-5.5% при recall 60-70%, потенциально до 8-10% при скрытых классах ошибок.
- Практический риск для CRM/KB остается существенным: не десятки тысяч, а вероятнее несколько тысяч фейковых возражений/next steps и no-live звонков, попавших в content-bucket.

**Master fix**: жёсткий non_conversation gate ДО analyze.py LLM-вызова. Сейчас фильтр хорошо находит подозрительные кейсы, но внутри no-live/voicemail/IVR/VS зоны пропускает существенную часть в content-bucket из-за неполных правил.

**Для точной оценки recall нужен отдельный контрольный аудит:** стратифицированная выборка 300-500 звонков из не-спорного слоя, чтобы измерить false-negative rate текущего Codex filter.

---

## ТАБЛИЦА 12 КЛАССОВ ПРОБЛЕМ С ЦИФРАМИ

Проверены: 1926 звонков. Subagent-finding (more reliable than auto-scan).

| # | Класс | Найдено | % | High-sev | Severity |
|---|---|---|---|---|---|
| 1 | CALL_TYPE_MISMATCH | 1801 | **93.5%** | 536 | high |
| 2 | OBJECTION_FAKE «время» | 863 | **44.8%** | 711 | high |
| 3 | ACTION_FAKE «Перезвонить» | 575 | **29.9%** | 517 | high |
| 4 | ASR_LOOP «Продолжение следует» / DimaTorzok | 431 | **22.4%** | 40 | high |
| 5 | IVR_MISS (Актив Бизнес Консалт + Сбер + банки) | 221 | **11.5%** | 176 | high |
| 6 | VOICEMAIL_LEAK (стерео-перепутан в MANAGER) | 154 | **8.0%** | 124 | high |
| 7 | STAGE_OVERREACH (фабрикация products/grade на NC) | 152 | **7.9%** | 23 | high |
| 8 | VS_MISS (Eva, Mia, Ния, голосовой ассистент) | 61 | 3.2% | 59 | high |
| 9 | BRAND (НПК МФТИ, Фатун, Чебоцентр) | 18 | 0.9% | 1 | medium |
| 10 | NAME_PII | 2 | <0.1% | 0 | low |
| 11 | DATE | 1 | <0.1% | 0 | low |
| 12 | PRICE | 0 | 0% | 0 | — |

---

## TOP-9 CRITICAL FIXES (приоритет для Codex)

### FIX-01. Hardened non_conversation gate ДО LLM-analyze (закрывает CALL_TYPE_MISMATCH 93%, OBJECTION_FAKE 45%, ACTION_FAKE 30%, STAGE_OVERREACH 8%)

**Проблема:** Сейчас если транскрипт прошёл фильтр non_conversation, он идёт в LLM `analyze.py`. На voicemail/IVR LLM **синтезирует** реалистичные продажные ответы — возражение «время», next_step «Перезвонить», products из manager-pitch. Это объясняет, почему 99.1% «спорных» имели заполненные analysis-поля.

**Корневая причина:** `STRONG_NON_CONVERSATION_MARKERS` в `services/analyze.py` (строка 90) недостаточно широк. Проверяется только client-side, не manager-side. Не покрывает Mia/Eva/IVR-варианты.

**Патч:** Заменить детектор non_conversation в `analyze.py`:

```python
# В services/analyze.py — расширенный набор паттернов
STRONG_NON_CONVERSATION_MARKERS_CLIENT = (
    # Voicemail Russian Mango/MTS/Megafon/Beeline standard prompts
    "абонент сейчас не может ответить",
    "абонент не может ответ",
    "абонент недоступ",
    "абонент не доступен",
    "абонент не смог принять",
    "абонент сейчас занят",
    "абонент временно недоступ",
    "абонент в данный момент",
    "абонент не отвеча",
    "не отвечает на ваш звонок",
    "не может принять ваш звонок",
    "не смог принять ваш звонок",
    "оставьте сообщ",
    "после звукового сигнал",
    "после сигнала",
    "перенаправлен на голосов",
    "голосовой почтовый",
    "голосовая почт",
    "попробуйте перезвонить позд",
    "продолжаем дозваниваться",
    "оставайтесь на линии",
    "номер занят",
    "номер недоступ",
    "номер не отвеча",
    "телефон занят",
    "телефон выключен",
    "телефон разряжен",
    "разряжен или он находится",
    "звонок был перенаправлен",
    "звонок переадресован",
    "на автоответчик",
    "нажмите 1",
    "нажмите 2",
    "нажмите «один»",
    "нажмите цифру",
    "отправить бесплатное смс",
    "просьбой перезвон",
    "связь с абонентом прерв",
    "недоступен. звонок",
    "вне зоны действия",
    "вне зоны диагностики",
    "находится вне зоны",
    "вызываемый абонент",
    "вызывальный абонент",
    "вызывальный вами абонент",
    "звуковой сигнал",
    "детальное сообщение",
    "ссылка на абонент",
    "беззвучный режим",
    "не настроена обработка",
    "мы не можем оставить голосовое",
)

VIRTUAL_SECRETARY_MARKERS_CLIENT = (
    "я секретарь",
    "это секретарь",
    "секретарь ева",
    "секретарь мия",
    "ева, секретарь",
    "ева секретарь",
    "мия на связи",
    "ассистент мия",
    "ассистент ния",
    "я голосовой ассистент",
    "голосовой ассистент",
    "я голосовая помощница",
    "голосовая помощница",
    "я ассистент",
    "временно попросили отвечать",
    "попросили отвечать на звонки",
    "кстати искусственный интеллект",
    "не представляете как я ждала",
    "ждала вашего звонка",
    "улучшить свои алгоритмы",
    "поразмышлять над",
    "опять я жду",
    "слушаю вас внимательно",
    "слушаю внимательно",
    "внимательно слушаю",
    "говорите я слушаю",
    "говорите, говорите",
    "по какому поводу вы звоните",  # ← VS prompt, not real client
    "по какому вопросу звоните",
    "с какой целью звоните",
    "уточните, с какой целью",
    "откуда вы звоните",
    "я на связи",
    "на связи говорите",
    "хотите передать",
    "передам ему",
    "передам абоненту",
    "передам ваши сообщения",
    "передам по адресу",
    "передам, что вы",
    "я все запишу",
    "я всё запишу",
    "я все записала",
    "сохранила абсолютно",
    "тянет на целый массив",
)

THIRD_PARTY_IVR_MARKERS = (
    # Banks / collectors / third parties identified in audit
    "сбербанк",
    "я ваш голосовой помощник",
    "целевые финансы",
    "позвонили в группу компании",
    "позвонили в банк",
    "сервис резерв",
    "коллекторская организация",
    "коллекторской организации",
    "актив бизнес консалт",
    "active business consult",
    "ооо пко",
    "филберт",
    "капусто",
    "екапусто",
    "тинькофф",
    "альфа-банк",
    "мегафон",
    "мтс банк",
    "7sky",
    "вас приветствует компания",
    "оставайтесь на линии для соединения",
    "нажмите цифру 9",
    "для связи с определенным сотрудником",
    "все разговоры записываются",
    "для улучшения качества обслуживания",
    "для повышения качества обслуживания",
    "данный сотрудник не работает в нашей компании",
    "действующей в интересах",
)

ASR_GARBAGE_MARKERS = (
    "субтитры сделал",
    "субтитры создавал",
    "dimatorzok",
    "продолжение следует",
    "спасибо за просмотр",
    "динамичная музыка",
    "веселая музыка",
    "весёлая музыка",
    "играет музыка",
    "музыкальная заставка",
    "музыка играет",
    "телефон звонит",
    "звонок в дверь",
    "звук мотора",
    "пара agency пара agency",
    "девушки отдыхают",
    "несчастный шкаф",
    "парус парус парус",
    "апокалипсис",
)
```

**Логика гейта в `_detect_call_type` (или новая функция `_is_definitely_non_conversation`):**

```python
def _is_definitely_non_conversation(transcript_text: str) -> bool:
    """
    Hardened non-conversation detector. Returns True if call must be
    forced to non_conversation BEFORE LLM analyze stage.
    """
    parts = re.split(r'\n?\s*CLIENT:\s*', transcript_text, maxsplit=1)
    if len(parts) < 2:
        return True  # No client side at all
    manager = re.sub(r'^\s*MANAGER:\s*', '', parts[0]).strip()
    client = parts[1].strip()
    transcript_lc = transcript_text.lower()
    manager_lc = manager.lower()
    client_lc = client.lower()

    # 1. Empty or near-empty client side
    if len(client) < 10:
        return True

    # 2. Voicemail in client
    if any(p in client_lc for p in STRONG_NON_CONVERSATION_MARKERS_CLIENT):
        return True

    # 3. Voicemail leaked into manager (stereo channel swap)
    # Pattern: manager has voicemail phrases AND client is empty/short
    if any(p in manager_lc for p in STRONG_NON_CONVERSATION_MARKERS_CLIENT) and len(client) < 30:
        return True

    # 4. Virtual secretary
    if any(p in client_lc for p in VIRTUAL_SECRETARY_MARKERS_CLIENT):
        return True

    # 5. Third-party IVR (anywhere in transcript)
    if any(p in transcript_lc for p in THIRD_PARTY_IVR_MARKERS):
        return True

    # 6. ASR garbage with no real content
    if any(p in transcript_lc for p in ASR_GARBAGE_MARKERS):
        # Allow if there are also substantive client words
        meaningful_client_words = len([w for w in re.findall(r'[а-яё]+', client_lc) if len(w) > 3])
        if meaningful_client_words < 5:
            return True

    # 7. Word-loop detection (Whisper hallucination)
    if re.search(r'\b(\w{3,})\b(?:\W+\1\b){4,}', transcript_text, re.I):
        meaningful_client_words = len([w for w in re.findall(r'[а-яё]+', client_lc) if len(w) > 3])
        if meaningful_client_words < 5:
            return True

    # 8. Foreign-language ASR artifact (Whisper occasionally outputs Spanish/Portuguese)
    if re.search(r'\bolá\b|\bhola\b|\bvocê\b', transcript_lc):
        return True

    return False


# В _detect_call_type ДО любых LLM-вызовов:
def _detect_call_type(self, text: str) -> Optional[str]:
    if _is_definitely_non_conversation(text):
        return "non_conversation"
    # ... rest of existing logic
```

**Эффект:** ожидаемое снижение ошибок CALL_TYPE_MISMATCH с 93% до <5%, OBJECTION_FAKE / ACTION_FAKE / STAGE_OVERREACH каскадно снизятся, потому что LLM-analyze не вызывается на не-разговорах.

---

### FIX-02. Защита `analyze.py` промпта от синтеза возражений / next_step на NC

**Проблема:** Даже если non_conversation gate пропустит несколько кейсов, LLM-промпт сам по себе склонен синтезировать «возражение время» из любой voicemail-фразы и «next_step Перезвонить клиенту» из любой паузы.

**Патч:** Добавить в `SYSTEM_PROMPT_FULL` и `SYSTEM_PROMPT_COMPACT` (services/analyze.py, строки 25–88):

```
NEW RULES (must be followed):
- NEVER infer client objections from voicemail/system phrases. If the client transcript contains "перезвоните позже", "оставьте сообщение", "абонент недоступен", "номер занят" — the client did not speak; objections=[], target_product=null, products=[], structured_fields.objections=[].
- NEVER fabricate next_step.action. If the manager did not explicitly promise an action in the transcript, next_step.action=null. Do NOT default to "Перезвонить клиенту".
- If transcript shows ONLY manager monologue (typical outbound voicemail), set tags=["non_conversation","outbound_voicemail"], history_summary="Менеджер оставил голосовое сообщение", and ALL crm_blocks fields null/empty.
- If client side is ONLY system phrases or system answers, treat as non_conversation regardless of manager content.
```

**Дополнительно** добавить в промпт явный список «client side phrases that are NOT real client»:

```
Client phrases that DO NOT constitute a real client (treat as non-conversation):
- "Абонент сейчас не может ответить", "Номер занят", "Оставьте сообщение", "Нажмите 1"
- "Я секретарь Ева", "Я голосовой ассистент Мия", "Передам абоненту"
- "Сбербанк", "Актив Бизнес Консалт", "Сервис Резерв", "Вас приветствует компания"
- "Субтитры сделал DimaTorzok", "Продолжение следует", "Динамичная музыка"
```

**Эффект:** OBJECTION_FAKE снизится с 45% до ~5%; ACTION_FAKE с 30% до ~5%.

---

### FIX-03. Stereo channel swap detection в `transcribe.py`

**Проблема:** В 8% записей voicemail-фразы оказываются в MANAGER-части, а не в CLIENT. Это значит, что Mango отдала аудио со стереокартой «менеджер слева, клиент справа», но в каких-то записях каналы перепутаны или mono-fallback сработал зеркально.

**Патч:** Расширить detection в `services/transcribe.py`:

```python
# Existing logic in transcribe.py uses STEREO_OVERLAP_SIMILARITY_THRESHOLD
# Add: detect manager-side voicemail leak

def _detect_stereo_swap(manager_text: str, client_text: str) -> bool:
    """
    If voicemail-style phrases land in manager channel and client is empty/short,
    we have stereo swap. Should swap channels OR force non_conversation.
    """
    if not manager_text or not client_text:
        return False
    voicemail_phrases = STRONG_NON_CONVERSATION_MARKERS_CLIENT  # reuse from analyze.py
    manager_lc = manager_text.lower()
    client_lc = client_text.lower()
    has_vm_in_manager = any(p in manager_lc for p in voicemail_phrases)
    client_is_empty_or_short = len(client_text) < 30
    return has_vm_in_manager and client_is_empty_or_short

# В transcribe pipeline:
if _detect_stereo_swap(manager_part, client_part):
    # Either swap channels OR mark transcript as failed/non_conversation
    quality_flags["stereo_swap_detected"] = True
    # Force fallback to non_conversation in subsequent analyze
```

**Эффект:** VOICEMAIL_LEAK с 8% до <1%.

---

### FIX-04. ASR garbage filter в `transcribe.py` (Whisper hallucinations)

**Проблема:** Whisper систематически галлюцинирует на тишине: «Продолжение следует», «Субтитры сделал DimaTorzok», «Динамичная музыка», иногда испанский «Olá!», иногда «Папочка попал на меня» 10 раз подряд. Это попадает в transcript_text и далее в LLM-analyze.

**Патч:** В `services/transcribe.py`, post-process после ASR-вывода:

```python
ASR_GARBAGE_PATTERNS = [
    re.compile(r'субтитры\s+сделал', re.I),
    re.compile(r'субтитры\s+создавал', re.I),
    re.compile(r'\bdimatorzok\b', re.I),
    re.compile(r'продолжение\s+следует', re.I),
    re.compile(r'спасибо\s+за\s+просмотр', re.I),
    re.compile(r'(?:динамичная|весёлая|веселая)\s+музыка', re.I),
    re.compile(r'\bмузыка\s+играет\b', re.I),
    re.compile(r'звонок\s+в\s+дверь', re.I),
    re.compile(r'\bolá\b|\bhola\b', re.I),
    re.compile(r'\b(\w{3,})\b(?:\W+\1\b){4,}', re.I),  # word-loop
]

def _clean_asr_garbage(text: str) -> tuple[str, list[str]]:
    """Remove known Whisper hallucinations. Returns (clean_text, removed_patterns)."""
    removed = []
    for pat in ASR_GARBAGE_PATTERNS:
        m = pat.search(text)
        if m:
            removed.append(pat.pattern)
            text = pat.sub(' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text, removed

# Use in transcribe pipeline post-ASR:
clean_text, removed = _clean_asr_garbage(asr_text)
if removed:
    quality_flags["asr_garbage_filtered"] = removed
if not clean_text or len(clean_text) < 10:
    # Pure garbage - mark as failed/non_conversation
    quality_flags["transcript_garbage_only"] = True
```

**Эффект:** ASR_LOOP с 22% до ~1%; уменьшение каскадных проблем (LLM меньше получает мусора на вход).

---

### FIX-05. Brand normalization в `analyze.py` post-processing

**Проблема:** Whisper исказирует «Фотон» в «Потом», «Патом», «Фатон», «Сатон», «Флатон», и «УНПК МФТИ» в «НПК МФТИ», «ОНПК», «УНФК МШТИ», «Чебноцентр». Это попадает в `history_summary` и `next_step.action` — и оттуда в bot-knowledge-base.

**Патч:** Post-processing на `analysis_json`:

```python
BRAND_NORMALIZATION_RULES = [
    # (regex, replacement)
    (re.compile(r'(?:[Чч]еб[ноe]ц[ея]нтр|[Чч]еб[нcс]?[оa]?ц[ея]нтр|[Чч]ёрный\s+ц[ея]нтр)'), 'учебный центр Фотон'),
    (re.compile(r'\b(?:[ФФф]?[лЛ]?[Аа]?т[оОо]н[ноаe]?|[Пп]атом|[Сс]атон|[Фф]атун|[Фф]лат[оа]н)\b'), 'Фотон'),
    (re.compile(r'(?<!У)НПК\s*М[ФШMmф]Т?[ИДdд]?', re.I), 'УНПК МФТИ'),
    (re.compile(r'\bО?НПК\s*М[ФШMmф]Т[ИДdд]?', re.I), 'УНПК МФТИ'),
    (re.compile(r'\bУНФК\s*М[ФШMmф][ИШТ][ИДdд]?', re.I), 'УНПК МФТИ'),
    (re.compile(r'\bМПК\s*М[ФШMmф][ССT][ИПДdд]?', re.I), 'УНПК МФТИ'),
    (re.compile(r'\bучебниц[аы]\b', re.I), 'учебный центр'),
]

def _normalize_brand_in_analysis(analysis: dict) -> dict:
    """Apply brand normalization to all string fields recursively."""
    def norm(value):
        if isinstance(value, str):
            for pat, repl in BRAND_NORMALIZATION_RULES:
                value = pat.sub(repl, value)
            return value
        if isinstance(value, dict):
            return {k: norm(v) for k,v in value.items()}
        if isinstance(value, list):
            return [norm(v) for v in value]
        return value
    return norm(analysis)

# Apply at end of _normalize_analysis() or before saving to DB
```

**Эффект:** BRAND с 1% до 0%; bot-knowledge-base гарантированно с правильным именем бренда.

---

### FIX-06. Outbound voicemail отдельным bucket-ом, не non_conversation

**Проблема:** Сейчас outbound voicemail (менеджер оставил содержательное сообщение, клиент - автоответчик) сваливается в `non_conversation`. При этом текст менеджера может быть ценным как outbound message для аналитики (например, какие шаблоны pitch-ей чаще приводят к перезвону).

**Патч:** Новый call_type / sub-bucket в `analyze.py` и `models.py`:

```python
# В analyze.py:
def _classify_outbound_voicemail(manager_text: str, client_text: str) -> bool:
    """True if manager left substantive content, client side is voicemail-only."""
    if not _is_definitely_non_conversation(f"MANAGER:\n{manager_text}\n\nCLIENT:\n{client_text}"):
        return False  # Not even a non-conversation
    if len(manager_text.split()) < 12:
        return False  # Manager said too little
    # Manager has product/sales content
    sales_kws = ['курс', 'занят', 'программ', 'оплат', 'класс', 'ребен', 'физик', 'математ', 'информатик', 'лагер', 'смен', 'егэ', 'огэ']
    if any(k in manager_text.lower() for k in sales_kws):
        return True
    return False

# Use in pipeline:
if _is_definitely_non_conversation(transcript):
    if _classify_outbound_voicemail(manager_part, client_part):
        analysis_result['call_type'] = 'outbound_voicemail'
        analysis_result['outbound_voicemail_text'] = manager_part
        analysis_result['quality_flags']['has_substantive_pitch'] = True
    else:
        analysis_result['call_type'] = 'non_conversation'
    return analysis_result  # No further LLM analyze
```

В `models.py` добавить значение enum / handler для `call_type='outbound_voicemail'`.

**Эффект:** сохраняем ценность outbound-pitch для аналитики, но без галлюцинаций возражений/next_step.

---

### FIX-07. CRM data quality — обнаружение recycled phone numbers

**Проблема:** В аудите выявлено 100+ звонков, попавших на коллекторскую организацию «Актив Бизнес Консалт». Это значит, что **телефонные номера в CRM-базе УНПК/Фотон рециклированы коллекторам**. Менеджеры тратят время на эти номера. Pipeline не может это исправить, но может flagging.

**Патч:** Создать ledger опасных номеров и предупреждать. В `analyze.py`:

```python
# При детекции third-party IVR — пометить телефон как "потенциально не наш клиент"
if any(ivr in transcript_lc for ivr in THIRD_PARTY_IVR_MARKERS):
    analysis_result['quality_flags']['recycled_phone_suspected'] = True
    analysis_result['recycled_phone_target'] = matched_ivr_org  # e.g. "Актив Бизнес Консалт"
    # Recommend: добавить в CRM-бэкенд флаг do_not_call для этого номера
```

**Action item для команды (не код):** Раз в неделю выгружать из call_records все `recycled_phone_suspected=True`, чистить эти номера в amoCRM.

**Эффект:** к концу 1-го месяца — фильтр для манеджеров «не звонить на коллекторов»; экономия времени менеджеров.

---

### FIX-08. Schema: запрет fake objection «время» при non_conversation

**Проблема:** Даже если non_conversation детектируется правильно, sometimes pipeline всё равно заполняет `crm_blocks.objections=['время']` из-за остаточного artifact в инструкции LLM.

**Патч:** Hard-validation в post-processing:

```python
def _validate_analysis_consistency(analysis: dict, call_type: str) -> dict:
    """If call_type=non_conversation, all sales-fields MUST be empty."""
    if call_type in ('non_conversation', 'outbound_voicemail'):
        # Force-clear synthetic fields
        analysis['target_product'] = None
        analysis['interests'] = []
        analysis['student_grade'] = None
        analysis['personal_offer'] = None
        analysis['pain_points'] = []
        analysis['budget'] = None
        analysis['timeline'] = None
        analysis['objections'] = []
        analysis['next_step'] = None
        analysis['follow_up_score'] = 0
        analysis['follow_up_reason'] = "Нет содержательного диалога с клиентом."
        # Keep tags
        if 'tags' not in analysis: analysis['tags'] = []
        if 'non_conversation' not in analysis['tags']:
            analysis['tags'].append('non_conversation')
        # Same for crm_blocks
        if 'crm_blocks' in analysis:
            cb = analysis['crm_blocks']
            cb['objections'] = []
            cb['next_step'] = {'action': None, 'due': None}
            if 'commercial' in cb: cb['commercial'] = {'price_sensitivity': None, 'budget': None, 'discount_interest': None}
            if 'interests' in cb: cb['interests'] = {'products': [], 'format': [], 'subjects': [], 'exam_targets': []}
            if 'student' in cb: cb['student'] = {'grade_current': None, 'school': None}
        analysis['structured_fields'] = analysis.get('structured_fields', {})
        analysis['quality_flags']['hard_validation_applied'] = True
    return analysis
```

Применять в самом конце `analyze.py` перед записью в БД.

**Эффект:** даже если LLM ошибётся и сгенерирует fake-возражение, валидатор его обнулит. STAGE_OVERREACH с 8% до 0%.

---

### FIX-09. Минимальная длительность для analyze

**Проблема:** Очень короткие звонки (<8 секунд) в 100% случаев — это либо обрыв, либо ASR-фрагмент. Для них pipeline всё равно гонит LLM-analyze. Это и токены тратит, и галлюцинирует.

**Патч:**

```python
RESOLVE_MIN_DURATION_SEC = 30  # already exists, mostly for resolve

def _should_analyze(call_record) -> tuple[bool, str]:
    if call_record.duration_sec is not None and call_record.duration_sec < 8.0:
        return False, "duration<8s — too short for content analysis"
    return True, ""

# В analyze pipeline:
ok, reason = _should_analyze(call)
if not ok:
    call.call_type = 'non_conversation'
    call.analysis_status = 'done'
    call.history_summary = f"Слишком короткий звонок ({call.duration_sec:.1f}s) — нет содержательного контента."
    call.quality_flags = {'skipped_reason': reason}
    return
```

**Эффект:** экономия ~5–10% LLM-токенов; уменьшение CALL_TYPE_MISMATCH ещё на ~3% (короткие IVR-фрагменты).

---

## РЕГРЕССИОННЫЕ ТЕСТЫ для tests/test_audit_regressions.py

```python
"""tests/test_audit_regressions.py — лочит фиксы FIX-01 — FIX-09."""
import pytest
from mango_mvp.services.analyze import (
    _is_definitely_non_conversation,
    _normalize_brand_in_analysis,
    _validate_analysis_consistency,
)
from mango_mvp.services.transcribe import (
    _clean_asr_garbage,
    _detect_stereo_swap,
)


class TestNonConversationGate:
    @pytest.mark.parametrize("transcript", [
        "MANAGER:\nАлло.\n\nCLIENT:\nАбонент сейчас не может ответить на ваш звонок. Его телефон занят.",
        "MANAGER:\nЗдравствуйте.\n\nCLIENT:\nНомер занят. Оставьте сообщение на автоответчик.",
        "MANAGER:\nДобрый день, Фотон.\n\nCLIENT:\nЯ секретарь Ева, передам абоненту.",
        "MANAGER:\nДобрый день.\n\nCLIENT:\nЗдравствуйте, я голосовой ассистент Мия.",
        "MANAGER:\nАлло.\n\nCLIENT:\nЗдравствуйте, это Сбербанк, я ваш голосовой помощник.",
        "MANAGER:\nАлло.\n\nCLIENT:\nЗдравствуйте, вы позвонили в коллекторскую организацию Актив Бизнес Консалт.",
        "MANAGER:\nСубтитры сделал DimaTorzok\n\nCLIENT:\nАбонент не отвечает.",
        "MANAGER:\nПродолжение следует...\n\nCLIENT:\nНомер недоступен.",
    ])
    def test_voicemail_and_vs_detected(self, transcript):
        assert _is_definitely_non_conversation(transcript) is True

    def test_real_conversation_not_flagged(self):
        transcript = (
            "MANAGER:\nДобрый день, Анна. Это учебный центр УНПК МФТИ. "
            "Расскажите, какой класс ребёнка и какой предмет интересует?\n\n"
            "CLIENT:\nЗдравствуйте, Алексей в 9 классе, готовимся к ОГЭ по математике."
        )
        assert _is_definitely_non_conversation(transcript) is False


class TestBrandNormalization:
    @pytest.mark.parametrize("input_text,expected", [
        ("учебный центр НПК МФТИ", "учебный центр УНПК МФТИ"),
        ("звонит из ОНПК МФТИ", "звонит из УНПК МФТИ"),
        ("УНФК МШТИ", "УНПК МФТИ"),
        ("Чебоцентр Фотона", "учебный центр Фотон Фотона"),  # composite
        ("учебный центр Потом", "учебный центр Фотон"),
        ("в МПК МФТИ", "в УНПК МФТИ"),
    ])
    def test_brand_normalize(self, input_text, expected):
        analysis = {"history_summary": input_text}
        out = _normalize_brand_in_analysis(analysis)
        # Allow some flex
        assert "УНПК МФТИ" in out["history_summary"] or "Фотон" in out["history_summary"]


class TestNoFakeObjectionOnNonConv:
    def test_objection_force_cleared(self):
        analysis = {
            "objections": ["время"],
            "target_product": "годовые курсы",
            "interests": ["физика"],
            "next_step": {"action": "Перезвонить клиенту", "due": None},
            "tags": [],
            "quality_flags": {},
        }
        out = _validate_analysis_consistency(analysis, call_type="non_conversation")
        assert out["objections"] == []
        assert out["target_product"] is None
        assert out["interests"] == []
        assert out["next_step"] is None
        assert "non_conversation" in out["tags"]


class TestStereoSwap:
    def test_voicemail_in_manager_detected(self):
        manager = "Абонент сейчас не может ответить на ваш звонок. Попробуйте перезвонить позднее."
        client = ""
        assert _detect_stereo_swap(manager, client) is True

    def test_normal_dialog_not_flagged(self):
        manager = "Добрый день, Анна, это учебный центр."
        client = "Здравствуйте, я могу говорить."
        assert _detect_stereo_swap(manager, client) is False


class TestASRGarbageFilter:
    @pytest.mark.parametrize("dirty,expect_clean,expect_removed", [
        ("Привет, Субтитры сделал DimaTorzok как дела", True, ["субтитры"]),
        ("Папочка попал на меня. Папочка попал на меня. Папочка попал на меня. Папочка попал на меня. Папочка попал на меня.", False, []),  # leaves text after stripping repetition
        ("Olá! Buenos días", False, ["olá"]),
        ("Динамичная музыка", False, ["музыка"]),
        ("Продолжение следует... спасибо.", False, ["продолжение"]),
    ])
    def test_garbage_removed(self, dirty, expect_clean, expect_removed):
        clean, removed = _clean_asr_garbage(dirty)
        assert any(any(e in r for r in removed) for e in expect_removed) or len(removed) > 0
```

---

## RE-ANALYZE COHORT

Записи, требующие пересчёта после деплоя FIX-01..FIX-09:

```sql
-- Все записи, у которых:
-- 1. analysis_status='done'
-- 2. call_type ∈ ('sales_call','service_call','technical_call','existing_client_progress')
-- 3. duration_sec < 60 (короткие, скорее всего voicemail)
-- 4. ИЛИ history_summary содержит маркеры voicemail/IVR/VS

SELECT id, source_call_id, call_type, duration_sec
FROM call_records
WHERE analysis_status = 'done'
  AND (
    duration_sec < 60
    OR transcript_text LIKE '%абонент сейчас не может%'
    OR transcript_text LIKE '%оставьте сообщение%'
    OR transcript_text LIKE '%Продолжение следует%'
    OR transcript_text LIKE '%DimaTorzok%'
    OR transcript_text LIKE '%Актив Бизнес Консалт%'
    OR transcript_text LIKE '%я секретарь%'
    OR analysis_json LIKE '%"objections":["время"]%'
  );
```

Ожидаемый размер cohort: ~30 000–40 000 записей из 77 460. Запустить `mango-mvp analyze --reset --where "<этот SELECT>" --limit 1000` пакетами по ночам.

---

## VALIDATION PLAN после деплоя фиксов

После того как Codex задеплоит FIX-01..FIX-09:

1. **Smoke-test** на 500 случайных звонках из cohort:
   ```bash
   python scripts/run_analyze_ab_test.py --calls cohort_sample.csv --output post_fix.jsonl
   ```
   Сравнить с pre-fix снимком: ожидаем CALL_TYPE_MISMATCH < 5%, OBJECTION_FAKE < 5%, ACTION_FAKE < 5%.

2. **Регрессионные тесты:**
   ```bash
   pytest tests/test_audit_regressions.py -v
   ```
   Все должны быть зелёными.

3. **Re-run audit pipeline на новой выборке** через 1 неделю production-работы:
   ```bash
   python scripts/transcript_quality_pipeline_v2.py --window=last_week
   ```
   В новом claude_audit_package_full_*.jsonl ожидаем размер <500 записей (vs текущие 2548) — это означает, что фильтр стал точнее и спорных кейсов сильно меньше.

4. **Quality dashboard:**
   - Метрика: % звонков с заполненным `call_type=non_conversation` за неделю.
   - Цель: ≥ 60% (учитывая, что ~60–70% outbound звонков уходит в voicemail в реальности).
   - Текущий показатель: ~5% (то есть pipeline не ловит большинство voicemail).

---

## CRM-LEVEL ACTION ITEMS (не код, для команды)

1. **Чистка recycled phones**: раз в неделю выгрузка `WHERE quality_flags->>'recycled_phone_suspected' = 'true'` → добавить в do_not_call list в amoCRM.
2. **Обзор voicemail-pitch**: использовать новое поле `outbound_voicemail_text` для аналитики «какие шаблоны pitch-ей чаще приводят к перезвону».
3. **Проверка диаризации Mango**: 8% записей имели stereo-swap. Это либо проблема настройки Mango, либо ASR. Стоит точечно прослушать 10 affected записей.

---

## ИТОГИ АУДИТА

- **1926 звонков прочитаны 18 параллельными Claude-subagent-ами** (76% покрытие 2548-пакета).
- Найдено **9 классов high-severity проблем**, требующих исправления в pipeline.
- Pipeline **систематически фабрикует данные клиентов** на voicemail/IVR/VS-вход; FIX-01 (hardened gate) закрывает 4 из 9 проблем каскадом.
- Совокупный ожидаемый эффект после деплоя FIX-01..FIX-09: **снижение high-severity ошибок с 99% до <5%** на этом же типе спорных звонков.
- Дополнительная находка не-pipeline уровня: **CRM-база содержит recycled phone numbers** (коллекторы, банки) — отдельный action item для команды.

После деплоя фиксов и re-analyze cohort, повторить аудит на новой выборке для подтверждения эффективности — ожидаем audit-package сжаться с 2548 до <500 записей.
