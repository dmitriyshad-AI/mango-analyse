# Аудит CRM-слоя: трек 2 — analyze + quality

Дата: 2026-06-13. Автор: Claude (субагент read-only). Метод: статический анализ кода + ревью задач D4 и blacklist_77. SQLite-прогон заблокирован (workspace concurrent lock); распределение классов в DB — гипотеза, не проверено.

---

## A. Карта модулей

### src/mango_mvp/services/analyze.py (2 370 строк)

**Назначение.** Единственный сервис конвейерного анализа звонков. Читает `call_records.transcript_text`, вызывает LLM (OpenAI/Ollama/Codex CLI), нормализует выход, пишет в `call_records.analysis_json`.

**Точки входа:**
- `AnalyzeService.run(session, limit)` — основной батч-цикл (строка 2 317): берёт до `limit` звонков со статусом `pending` или `failed`/`in_progress`-зависаний, прогоняет по одному, пишет результат.
- `AnalyzeService._analyze_text(call, text)` — внутренняя точка: выбирает провайдера, проверяет non_conversation-ворота, отдаёт в LLM.
- `AnalyzeService._normalize_analysis(call, text, raw)` — нормализация и сборка history_summary.

**Версии промптов (строки 161–164):**
- `ANALYZE_PROMPT_VERSION_COMPACT = "v6"` (профиль compact, дефолт если `analyze_prompt_profile != "full"`)
- `ANALYZE_PROMPT_VERSION_FULL = "v7"` (профиль full)
- Промпт выбирается в `_analysis_system_prompt()` / `_analysis_prompt_version()`.

**Провайдеры:** `openai`, `ollama`, `codex_cli`, `mock`. Codex CLI: до 5 попыток с backoff, таймаут настраивается. Все трое используют LLM-кэш (`LLMResponseCache`) по хешу промпта+модели+версии.

### src/mango_mvp/quality/non_conversation.py

**Назначение.** Детерминированный классификатор «живой диалог vs автоответчик/IVR». Экспортирует `detect_non_conversation_signals(...)` → `NonConversationSignals`. Не меняет DB, только читает текст.

**Ключевые сигналы (строки 21–200+):** HARD_NO_LIVE_RE, SYSTEM_NO_DIALOGUE_RE, THIRD_PARTY_IVR_RE, VIRTUAL_SECRETARY_RE, ASR_ARTIFACT_RE, RISKY_KEYWORD_RE, LIVE_DIALOGUE_RE. Скоринг: балльная система ±, пороги `should_force_non_conversation` и `requires_manual_review`.

**Сейфгарды живого диалога (строки 437–511):** `long_client_live_safeguard`, `edtech_live_safeguard`, `proxy_parent_safeguard`, `sales_live_safeguard`, `transfer_after_live_dialogue`, `ambiguous_service_attempt_safeguard` — именно они лечили «болезнь v6» (живой длинный разговор → автоответчик).

### src/mango_mvp/quality/crm_text_quality_detector.py

**Назначение.** Детектор проблем в CRM-тексте (выжимках, полях AMO). Проверяет `TARGET_CRM_TEXT_FIELDS` (~22 поля) на типовые дефекты: диалог-дамп, размытый следующий шаг, пустые возражения-заглушки, исторические артефакты в активных полях.

Не вмешивается в analyze-пайплайн — используется отдельно в export/writeback QA.

### Прочие quality-модули

- `transcript_quality_*.py` (15 файлов) — ревью и бэкфилл quality-флагов транскриптов, инструменты аудита; в production-конвейере напрямую не вызываются.
- `stage14_quality_comparison.py`, `stage15_export_quality_gate.py` — стейджинговые гейты.
- `tenant_text_normalizer.py`, `amo_loss_reason_policy.py` — вспомогательные.

---

## B. Поток сырьё → выжимка

```
transcript_text (DB)
  ↓
_analyze_text()
  ├─ detect_non_conversation_signals()   [детерминированно]
  │    └─ should_force_non_conversation → _non_conversation_analysis()  [без LLM]
  ├─ _is_non_conversation()              [второй детерм. ворот]
  │    └─ True → _non_conversation_analysis()
  └─ LLM вызов (_openai / _ollama / _codex_cli)
       ├─ _compact_transcript_for_prompt()  [компакция + обрезка]
       ├─ _analysis_prompt_context()        [сборка промпта]
       └─ raw JSON ответ
  ↓
_normalize_analysis()
  ├─ извлечение полей из raw JSON
  ├─ детерминированное слияние с паттернами (products/subjects/formats/…)
  ├─ _detect_call_type()                 [финальный класс звонка]
  ├─ if non_conversation → обнуление всех полей
  └─ _compose_history_summary()          [сборка history_summary]
  ↓
analysis_json → call_records.analysis_json
```

Разделение: LLM понимает смысл + черновит, детерминизм верифицирует класс и обнуляет поля.

---

## C. Неэффективности

1. **Двойной вызов `detect_non_conversation_signals`** в `_analyze_text` (строки 2 288–2 295): сначала `signals.should_force_non_conversation`, потом `_is_non_conversation()` → который внутри вызывает `_detect_call_type()` → который снова вызывает `detect_non_conversation_signals`. Итого: минимум 2, иногда 3 вызова одного детерминированного классификатора на один звонок. Не критично по скорости (regexp), но семантически дублирует логику.

2. **Каскадный escalate-к-full:** если compact-ответ LLM признан неполным (`_should_escalate_full_profile`), вызывается повторный LLM-вызов с full-профилем (строки 2 308–2 314). Кэш снижает частоту, но это потенциально 2× стоимость на спорных звонках.

3. **`_normalize_analysis` вызывается дважды** в `run()` (строка 2 336) и дополнительно может быть вызвана через `_openai_analysis` → `_normalize_analysis` уже там. Проверено: в коде `_normalize_analysis` вызывается снаружи в `run()` после `_analyze_text`, а `_analyze_text` возвращает raw; двойного вызова нет. (УТОЧНЕНИЕ: ок, дублирования нет.)

4. **Паттерны `products/subjects` добавляются детерминированно поверх LLM-ответа** (строки 1 675–1 691) — всегда, даже если LLM уже их вернул. Это безопасно (unique-фильтр), но может добавить шум для коротких технических звонков с упоминанием EdTech-слов.

5. **Heuristic next_step** добавляется при `"перезвон"/"отправ" in text.lower()` (строки 1 762–1 766) — без контекстной проверки: слово «перезвон» в фразе «клиент не перезванивает» даст ложный next_step «Перезвонить клиенту».

---

## D. Места обрезки текста + риски

### D1. Обрезка транскрипта для промпта

**Файл:** `src/mango_mvp/services/analyze.py`, строки 212–217, 697–706.

**Константы:**
```
FULL-профиль:    max=10 000 симв, head=7 000, tail=2 800
COMPACT-профиль: max=6 500 симв, head=4 600, tail=1 600
```

**Механизм (строки 699–706):**
```python
if len(prompt_transcript) > max_chars:
    head = prompt_transcript[:head_chars].rstrip()
    tail = prompt_transcript[-tail_chars:].lstrip()
    prompt_transcript = f"{head}\n\n[... transcript truncated ...]\n\n{tail}"
    truncated = True
```

**Что теряется:** середина разговора — как правило, ключевой блок переговоров о цене, возражениях, конкретных классах. При длинных звонках (>10 000 символов до компакции — ориентировочно >20–30 мин в стерео) средний сегмент диалога полностью выпадает из контекста LLM.

**Сигнал в quality_flags:** `analyze_prompt_truncated: true` + `analyze_transcript_chars_original` vs `analyze_transcript_chars_prompt` — данные записываются (строки 742–748), т.е. можно grep по DB.

**Риск:** у самых длинных звонков (многие сервисные, в т.ч. 1 800 с = 30 мин, как call_id 19055 в blacklist) высока вероятность truncation. Именно такие звонки v6 ошибочно классифицировал как non_conversation: compact-промпт не «видел» живой диалог в середине.

### D2. Обрезка текста evidence-цитат

**Файл:** `src/mango_mvp/services/analyze.py`, строка 1 856.
```python
"text": text_item[:260],
```
Каждая цитата из LLM-evidence режется до 260 символов. Потеря незначительна — это вспомогательное поле.

### D3. Обрезка history_summary

**Файл:** строки 1 511–1 512, 1 547–1 548.
```python
if len(compact) > 32000:
    compact = compact[:31974].rstrip() + " [обрезано по лимиту поля]"
```
Срабатывает только на аномально длинных выжимках (>32 000 символов). На практике маловероятно, но маркер `[обрезано по лимиту поля]` может появиться в CRM.

### D4. Обрезка summary-фолбека (mock mode)

**Файл:** строки 2 036, 2 657 (mock analysis).
```python
"summary": text[:600],
```
Только в mock-провайдере, в production не активен.

### D5. Компакция транскрипта (до обрезки)

**Файл:** строки 631–724 (`_compact_transcript_for_prompt`).

- Удаляются временные метки (`[00:05.2]`).
- Схлопываются повторяющиеся «ага»/«да»/«угу» одного спикера подряд.
- Многократные «ага, ага, ага» → один «ага».

**Риск «болезни»:** компакция убирает много строк-заполнителей, уменьшая `chars_compacted`. Если после компакции transcript всё ещё >max_chars — идёт обрезка середины. Именно это (компакция + усечение) при compact-промпте (max 6 500) давало потерю живого контекста у длинных звонков.

---

## E. Blacklist_77 и «хвост 3 439»

### Blacklist_77

**Проверено:** файл `blacklist_77.txt` находится в:
`/Users/dmitrijfabarisov/Yandex.Disk.localized/OpenClaw/Actual Mango Tests/analyze_rerun_20260611/blacklist_77.txt`

По задаче TZ-19 (отчёт `tasks/_done/2026-06-13_TZ19_D4_tail_bundle_blacklist_batch_ingest_prep_final_report.md`):
- 77 звонков = исторически bad-классифицированные при v6.
- Батч-15 прогнан с v7/full (строки ревью `REVIEW_blacklist_batch15_2026-06-13.md`).
- **Ошибка v6:** compact-профиль + truncation → модель не видела живой диалог в середине → давала tag `non_conversation` + убирала target_product.
- **Фикс v7:** full-профиль (max 10 000), расширенные сейфгарды в `non_conversation.py` → 8/8 содержательных звонков из батча-15 корректно восстановлены.
- Остаток: 57 звонков ждут прогона (TZ-20 в `tasks/_inbox_codex/2026-06-13_TZ20_PROMPT_for_D4_blacklist57_finish.md`).
- Вливание в canonical DB — **ещё не выполнено**, ожидает Дмитрия (регрейд цифр + `--apply`).

**Гипотеза (не проверена по DB):** у записей из blacklist_77 в `analysis_json.quality_flags.analyze_prompt_truncated` = true И `analyze_prompt_profile` = "compact" — это должно коррелировать с ошибкой.

### Хвост 3 439

**Проверено:** бандл M1 (`analyze_tail_20260612/`):
- 3 439 звонков, transcript chars = 15 682 102 (≈4 560 символов в среднем).
- Промпт-версия: v7/full.
- Срез `slice_zone.db` на M1: 116 МБ, quick_check OK.
- Пересечение с blacklist = 0.
- Статус: задача `.task.yaml` + `.ready` отправлены в `_inbox_m1`, станция ещё не подхватила.

**Что не проверено в DB:** распределение классов (sales_call / service_call / non_conversation / ...) в canonical_calls_master.db. SQLite недоступен через bash (concurrent lock). Метаданные модели/версии промпта в `analysis_json` — не проверены напрямую.

---

## Итог: проверено vs гипотеза

| Утверждение | Статус |
|---|---|
| analyze.py — единственный сервис анализа, run() — точка входа | ПРОВЕРЕНО (код) |
| Промпты v6=compact / v7=full, строки 162–163 | ПРОВЕРЕНО (код) |
| Обрезка транскрипта: full max 10 000 / head 7 000 / tail 2 800 | ПРОВЕРЕНО (код, стр. 212–217) |
| Обрезка транскрипта: compact max 6 500 / head 4 600 / tail 1 600 | ПРОВЕРЕНО (код, стр. 215–217) |
| Механизм среза: `[:head_chars]` + `[-tail_chars:]` | ПРОВЕРЕНО (код, стр. 700–706) |
| Evidence цитаты режутся до 260 символов | ПРОВЕРЕНО (код, стр. 1 856) |
| History_summary режется до 32 000 символов с маркером | ПРОВЕРЕНО (код, стр. 1 511–1 512) |
| Болезнь v6 = compact+truncation → non_conversation на длинных | ПРОВЕРЕНО (ревью blacklist-батча-15, 8/8 восстановлены) |
| Фикс v7 = full-профиль + новые сейфгарды non_conversation.py | ПРОВЕРЕНО (код + ревью) |
| blacklist_77 ещё не влит в canonical DB | ПРОВЕРЕНО (TZ-19 отчёт, TZ-20 в инбоксе) |
| 3 439 хвост ещё не прогнан на M1 | ПРОВЕРЕНО (TZ-19, задача в _inbox_m1 не подхвачена) |
| Двойной вызов detect_non_conversation_signals в _analyze_text | ПРОВЕРЕНО (код, стр. 2 288–2 295) |
| Heuristic next_step «перезвон» может дать ложный результат | ПРОВЕРЕНО (код), риск не подтверждён на данных |
| Распределение call_type в canonical DB | ГИПОТЕЗА (SQLite недоступен) |
| Версия промпта в analysis_json конкретных записей | ГИПОТЕЗА (DB недоступна) |
| Доля truncated звонков в production базе | ГИПОТЕЗА (quality_flags.analyze_prompt_truncated есть, но не агрегировано) |
