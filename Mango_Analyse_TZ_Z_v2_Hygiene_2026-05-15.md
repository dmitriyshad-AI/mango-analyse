# ТЗ-Z-v2: Hygiene infrastructure — переписанная версия после реверс-аудита Codex

Дата: 2026-05-15
Автор: Claude (наставник Дмитрия) после critique Codex'а и точной grep-проверки через субагент
Заменяет: `Mango_Analyse_TZ_Z_Hygiene_Infrastructure_2026-05-14.md` (v1, имела серьёзные ошибки)
Адресат: Codex как primary implementation agent

---

## 0. Что изменилось относительно v1

Codex выявил 5 серьёзных ошибок в v1:
- Z.1: `_merge_texts` возвращает `str`, не dict — я предлагал ломать API на 5 callsites
- Z.2: Stage15 **не** принимает tenant_config — указанные ключи не существуют, реальное место правки — 2 скрипта
- Z.3: `100% результат` сейчас санитизируется (Codex прав) — мой test был FN. Жёсткая зависимость от TZ-Y
- Z.4: список промптов неполный (11 в реальности, не 7), пропущен реальный риск в `_merge_with_openai`. Plus `theme_assigner_llm.py` не существует (будет создан только в catalog D.1)
- Z.5: `ingest_call/transcribe_call/analyze_call` не существуют, пропущен слой `resolve`, существует `tests/test_smoke.py` — нужно расширять, не дублировать

После точной grep-проверки через субагент картина прояснилась. **v2 разбит на 2 «готовых сейчас» трека (Z-1, Z-2) и 3 «future-work» (Z-3, Z-4, Z-5) с явными зависимостями.**

---

## ТРЕК Z-1: _merge_variant_pair counter for dropped suspicious chunks

### Контекст

**Точные факты из grep-проверки:**

- `_merge_texts(primary_text: str, secondary_text: str) -> str` (transcribe.py:1215) — возвращает строку, не dict.
- `_is_suspicious_chunk(tokens: list[str]) -> bool` (transcribe.py:1204) — использует `_chunk_quality(tokens)` для score/max_run/repetition_ratio.
- Все 5 callers `_merge_texts` находятся внутри `_merge_variant_pair` (transcribe.py:1865-2015). Этот caller **уже** оборачивает результат в dict `{text, selection, confidence, provider, notes, similarity}` — есть готовое место для counter.
- В DB у `CallRecord.transcript_variants_json` есть структура `merge_meta` (transcribe.py:2410, 2416, 2546).

### Правка Z-1.1 — добавить counter в `_merge_variant_pair`, не в `_merge_texts`

**Файл:** `src/mango_mvp/services/transcribe.py:1865-2015 (_merge_variant_pair)`

**Что сделать:**

`_merge_texts` оставить **без изменений** (возвращает str). Counter формируется в `_merge_variant_pair` через **отдельный публичный helper**, который повторяет логику `_is_suspicious_chunk` на opcodes:

```python
def _count_suspicious_drops_in_merge(self, primary_text: str, secondary_text: str) -> dict:
    """Считает chunks, которые _merge_texts выкинул как suspicious. 
    Логика та же, что в _merge_texts (opcodes delete/insert + _is_suspicious_chunk).
    Возвращает counter + samples без изменения merged result."""
    from difflib import SequenceMatcher
    
    primary_tokens = self._tokenize(primary_text)
    secondary_tokens = self._tokenize(secondary_text)
    matcher = SequenceMatcher(a=primary_tokens, b=secondary_tokens)
    
    dropped = []
    total_chars = 0
    for opcode, ai, aj, bi, bj in matcher.get_opcodes():
        if opcode in ("delete", "insert"):
            chunk_tokens = primary_tokens[ai:aj] if opcode == "delete" else secondary_tokens[bi:bj]
            if self._is_suspicious_chunk(chunk_tokens):
                chunk_text = " ".join(chunk_tokens)
                score, max_run, repetition_ratio = self._chunk_quality(chunk_tokens)
                dropped.append({
                    "opcode": opcode,
                    "sample": chunk_text[:50],
                    "length": len(chunk_text),
                    "score": round(score, 2),
                    "max_run": max_run,
                    "repetition_ratio": round(repetition_ratio, 3),
                })
                total_chars += len(chunk_text)
    
    return {
        "count": len(dropped),
        "total_chars": total_chars,
        "samples": dropped[:5],
    }
```

В `_merge_variant_pair` после вызова `_merge_texts` добавить:

```python
merged_text = self._merge_texts(primary_text, secondary_text)
suspicious_drops = self._count_suspicious_drops_in_merge(primary_text, secondary_text)
# ... existing dict construction ...
return {
    "text": merged_text,
    "selection": "...",
    "confidence": ...,
    "provider": "rule",
    "notes": {
        # ... existing notes ...
        "suspicious_drops": suspicious_drops,
    },
    "similarity": ...,
}
```

**Это безопасно потому что:**
- `_merge_texts` не меняет behaviour (возвращает тот же str)
- Counter формируется отдельно, не ломает API
- 5 callsites `_merge_texts` не меняются
- В `transcript_variants_json` появляется `merge_meta.notes.suspicious_drops` — это новое поле, downstream его игнорирует (backward compat)

### Тесты Z-1

**Файл:** `tests/test_transcribe_z1_suspicious_drops_counter.py` (новый)

```python
import pytest
from mango_mvp.services.transcribe import TranscribeService
from mango_mvp.config import get_settings


@pytest.fixture
def service():
    return TranscribeService(get_settings())


def test_count_suspicious_drops_returns_zero_for_clean_merge(service):
    """На чистых текстах без артефактов — counter = 0."""
    primary = "Добрый день, по вашему запросу о подготовке к ЕГЭ по математике."
    secondary = "Добрый день по вашему запросу о подготовке к ЕГЭ по математике."
    result = service._count_suspicious_drops_in_merge(primary, secondary)
    assert result["count"] == 0
    assert result["total_chars"] == 0
    assert result["samples"] == []


def test_count_suspicious_drops_logs_dimatorzok_like_artifact(service):
    """Whisper-loop типа DimaTorzok ловится в suspicious."""
    primary = (
        "Нормальный текст. "
        "DimaTorzok DimaTorzok DimaTorzok DimaTorzok DimaTorzok DimaTorzok DimaTorzok DimaTorzok DimaTorzok. "
        "Продолжение нормального текста."
    )
    secondary = "Нормальный текст. Продолжение нормального текста."
    result = service._count_suspicious_drops_in_merge(primary, secondary)
    assert result["count"] >= 1
    assert result["total_chars"] > 50
    assert any("DimaTorzok" in s["sample"] for s in result["samples"])


def test_merge_texts_signature_unchanged(service):
    """Regression: _merge_texts всё ещё возвращает str."""
    result = service._merge_texts("текст A", "текст A с правкой")
    assert isinstance(result, str)


def test_merge_variant_pair_includes_suspicious_drops_in_notes(service):
    """_merge_variant_pair кладёт suspicious_drops в notes."""
    # ... realistic test через _merge_variant_pair ...
    pair_result = service._merge_variant_pair(
        primary_text="чистый текст",
        secondary_text="чистый текст",
        # ... остальные аргументы ...
    )
    assert "suspicious_drops" in pair_result["notes"]
```

### Acceptance трека Z-1

1. `_merge_texts` signature **не меняется** (regression test).
2. `_merge_variant_pair` возвращает dict с `notes.suspicious_drops`.
3. На 50 случайных звонках из `tests/fixtures/` (НЕ stable_runtime) — поле `suspicious_drops` появляется в `merge_meta.notes`.
4. `merged_text` (фактический результат merge) **идентичен** до и после правки на 50 контрольных звонках (verified diff).
5. Existing tests `tests/test_transcribe*.py`, `tests/test_llm_review_merge.py` passed.

### Deliverables Z-1

- `src/mango_mvp/services/transcribe.py` (добавлен `_count_suspicious_drops_in_merge`, обновлён `_merge_variant_pair`)
- `tests/test_transcribe_z1_suspicious_drops_counter.py` (новый)
- `audits/_inbox/transcribe_z1_<timestamp>/`:
  - `AUDIT_SCOPE.md`
  - `MERGE_DROPS_SAMPLE_50_CALLS.csv` — выборка из 50 звонков с counter
  - `BACKWARD_COMPAT_DIFF.md` — proof что merged_text identical before/after

---

## ТРЕК Z-2: tenant_config sha256 pin-check в CRM writeback и export gates

### Контекст

**Точные факты из grep:**

- Stage15 `quality/stage15_export_quality_gate.py` **не** содержит ни одного упоминания tenant_config — это in-process gate, не tenant-aware.
- Реальный loader `productization/tenant_config.py:20` `load_tenant_config(path)` возвращает dataclass `TenantConfigLoadResult(path, sha256, config)`.
- `tenant_config_summary(result)` (tenant_config.py:58-73) возвращает **плоский** dict: `{"loaded", "path", "sha256", "tenant_id", "schema_version"}`. Не вложенный.
- Только 2 callers в production: `scripts/run_crm_writeback_quality_gate.py:212` и `scripts/build_post_backfill_amo_ready_export.py:1208`. Оба кладут результат под ключ `"tenant_config"`.

### Правка Z-2.1 — sha256 pin constants

**Файл:** новый `src/mango_mvp/productization/tenant_config_pinning.py`

```python
"""Pinned tenant_config sha256 — изменение требует явного bump."""

# Pinned значения — обновляются ТОЛЬКО при сознательном изменении tenant_config_v1.json
EXPECTED_TENANT_CONFIG_SHA256 = "REPLACE_WITH_ACTUAL_HASH"  # Codex считает текущий sha256 и подставляет
EXPECTED_TENANT_ID = "foton"
EXPECTED_SCHEMA_VERSION = "tenant_config_v1"


def check_tenant_config_pin(summary: dict) -> tuple[bool, str]:
    """Проверяет, что summary соответствует pinned значениям.
    
    Args:
        summary: результат tenant_config_summary(load_result). Имеет плоские ключи:
                 loaded, path, sha256, tenant_id, schema_version.
    
    Returns:
        (passed, reason)
    """
    if not summary.get("loaded"):
        return False, "tenant_config not loaded"
    
    actual_sha = summary.get("sha256")
    if actual_sha != EXPECTED_TENANT_CONFIG_SHA256:
        return False, (
            f"tenant_config sha256 changed without bump: "
            f"expected {EXPECTED_TENANT_CONFIG_SHA256[:16]}..., got {(actual_sha or '')[:16]}..."
        )
    
    actual_tenant = summary.get("tenant_id")
    if actual_tenant != EXPECTED_TENANT_ID:
        return False, f"tenant_id mismatch: expected {EXPECTED_TENANT_ID}, got {actual_tenant!r}"
    
    actual_schema = summary.get("schema_version")
    if actual_schema != EXPECTED_SCHEMA_VERSION:
        return False, f"schema_version mismatch: expected {EXPECTED_SCHEMA_VERSION}, got {actual_schema!r}"
    
    return True, "tenant_config_pin_ok"
```

### Правка Z-2.2 — integration в 2 script-callers

**Файлы:**
- `scripts/run_crm_writeback_quality_gate.py:212` (где формируется `"tenant_config": tenant_config_summary(...)`)
- `scripts/build_post_backfill_amo_ready_export.py:1208` (то же)

**Что сделать:**

Добавить вызов `check_tenant_config_pin` после `tenant_config_summary`:

```python
from mango_mvp.productization.tenant_config import load_tenant_config, tenant_config_summary
from mango_mvp.productization.tenant_config_pinning import check_tenant_config_pin

tenant_config_result = load_tenant_config(...)
tenant_summary = tenant_config_summary(tenant_config_result)

pin_passed, pin_reason = check_tenant_config_pin(tenant_summary)
if not pin_passed:
    # Логировать в gate result + НЕ пропускать дальше (или warning — на выбор Дмитрия)
    raise RuntimeError(f"tenant_config pin failed: {pin_reason}")
    # Или: result["warnings"].append({"type": "tenant_config_drift", "message": pin_reason})

result_payload["tenant_config"] = tenant_summary
result_payload["tenant_config_pin"] = {"passed": pin_passed, "reason": pin_reason}
```

### Правка Z-2.3 — документация bump procedure

**Файл:** `docs/TENANT_CONFIG_BUMP_PROCEDURE_2026-05-XX.md` (новый)

```markdown
# Tenant Config Bump Procedure

При любом изменении `tenants/foton/config/tenant_config_v1.json`:

1. Изменить JSON.
2. Запустить: `python -m mango_mvp.productization.tenant_config_pinning --print-current`
3. Скопировать новый sha256 в `EXPECTED_TENANT_CONFIG_SHA256` в `tenant_config_pinning.py`.
4. Если изменилась схема — поднять `EXPECTED_SCHEMA_VERSION` и обновить loader.
5. Документировать в `docs/` changelog: что изменилось, почему.
6. Один PR с обоими изменениями (json + pin constants).

Если gate упал с "sha256 changed without bump" — это интенциональное защитное поведение.
Либо поднять pin constants и продокументировать, либо откатить изменение JSON.
```

### Тесты Z-2

**Файл:** `tests/test_tenant_config_pin_z2.py` (новый)

```python
def test_pin_check_passes_on_current_config():
    """Текущий tenant_config matches pinned."""
    result = load_tenant_config(Path("..."))
    summary = tenant_config_summary(result)
    passed, reason = check_tenant_config_pin(summary)
    assert passed, reason


def test_pin_check_fails_on_changed_sha256():
    """Если sha256 не соответствует pinned — fail."""
    summary = {
        "loaded": True,
        "path": "/test/path",
        "sha256": "different_hash_deadbeef",
        "tenant_id": "foton",
        "schema_version": "tenant_config_v1",
    }
    passed, reason = check_tenant_config_pin(summary)
    assert not passed
    assert "sha256 changed" in reason


def test_pin_check_fails_on_not_loaded():
    summary = {"loaded": False}
    passed, reason = check_tenant_config_pin(summary)
    assert not passed
    assert "not loaded" in reason


def test_pin_check_fails_on_schema_drift():
    summary = {
        "loaded": True,
        "sha256": EXPECTED_TENANT_CONFIG_SHA256,
        "tenant_id": "foton",
        "schema_version": "tenant_config_v999_fake",
    }
    passed, reason = check_tenant_config_pin(summary)
    assert not passed
    assert "schema_version mismatch" in reason
```

### Acceptance трека Z-2

1. `tenant_config_pinning.py` существует, EXPECTED_TENANT_CONFIG_SHA256 заполнен реальным хешем.
2. 2 каллера в scripts/ интегрированы.
3. 4 теста зелёные.
4. `docs/TENANT_CONFIG_BUMP_PROCEDURE_2026-05-XX.md` создан.
5. На текущем tenant_config: pin passes.

### Deliverables Z-2

- `src/mango_mvp/productization/tenant_config_pinning.py` (новый)
- `scripts/run_crm_writeback_quality_gate.py` (правка integration)
- `scripts/build_post_backfill_amo_ready_export.py` (правка integration)
- `tests/test_tenant_config_pin_z2.py` (новый)
- `docs/TENANT_CONFIG_BUMP_PROCEDURE_2026-05-XX.md` (новый)
- `audits/_inbox/tenant_config_pin_z2_<timestamp>/AUDIT_SCOPE.md`

---

## ТРЕК Z-3: Negative-overblock corpus (ОТЛОЖЕН — зависит от TZ-Y)

**НЕ реализуется в v2 как самостоятельный трек.**

**Причина:** Точная grep-проверка показала, что `100% результат` **сейчас санитизируется** (через PERCENT_RE → [PERCENT] → [PAYMENT_OPTIONS]). Это behaviour будет изменено только после реализации **TZ-Y трек Y-A** (правка Y-A.2 — PERCENT_RE context-aware). Если делать Z-3 сейчас:
- Negative test «100% результат не должно санитизироваться» **упадёт** на текущем коде
- Это создаст ложные позитивы в frozen corpus
- Корпус будет требовать постоянной синхронизации с TZ-Y

**Правильный план:**

1. Реализовать TZ-Y трек Y-A (spaced-thousands + PERCENT context).
2. **После** закрытия Y-A — реализовать Z-3 negative corpus, где acceptance соответствует новой behaviour.

В этом ТЗ Z-3 явно отложен. Промпт для будущего ТЗ-Z-3 будет создан после закрытия TZ-Y.

---

## ТРЕК Z-4: Prompt drift protection (ОТЛОЖЕН — зависит от TZ-X + расширенный scope)

**НЕ реализуется в v2 как самостоятельный трек.**

**Причина 1:** TZ-X-v2 трек X-A.1 включает bump `ANALYZE_PROMPT_VERSION_FULL = "v6" → "v7"`. Если зафиксировать hash сейчас, после X-A.1 он сразу станет stale.

**Причина 2:** Точная grep-проверка показала, что в проекте **11 промптов**, не 7 как в моей v1:
1. `services/transcribe.py:30` `MERGE_SYSTEM_PROMPT`
2. `services/transcribe.py:41` `CODEX_MERGE_PROMPT_TEMPLATE`
3. `services/transcribe.py:61` `ROLE_ASSIGN_SYSTEM_PROMPT`
4. `services/analyze.py:26` `SYSTEM_PROMPT_FULL`
5. `services/analyze.py:58` `SYSTEM_PROMPT_COMPACT`
6. `services/resolve.py:28` `RESOLVE_SYSTEM_PROMPT`
7. `services/resolve.py:43` `DIALOGUE_RESOLVE_SYSTEM_PROMPT`
8. `amocrm_runtime/deal_llm.py:36-37` `deal_llm_v2`
9. `insights/llm_review.py:20` `sales_moment_llm_review_v1`
10. `quality/hard_gate_gpt_review.py:20` `hard_gate_gpt_review_prompt_v1`
11. `quality/transcript_quality_llm_review.py:21` `transcript_quality_llm_review_v2`

**Также:** `theme_assigner_llm.py` ещё **не существует** в коде — он создаётся в Catalog v2 D.1. Если он будет создан до Z-4, добавится 12-й промпт.

**Причина 3:** В коде **уже есть** `TRANSCRIBE_MERGE_PROMPT_VERSION` как drift-protection mechanism. Z-4 должно **расширить** этот механизм на остальные промпты, не создавать параллельную систему.

**Реальный риск (из grep):** `_merge_with_openai` (transcribe.py:1630-1696) использует `MERGE_SYSTEM_PROMPT + CODEX_MERGE_PROMPT_TEMPLATE.format(...)`. Если эти константы меняются без bump `TRANSCRIBE_MERGE_PROMPT_VERSION` — кэш отдаёт устаревшее.

**Правильный план:**

1. Реализовать TZ-X-v2 (включая X-A.1 bump v6→v7).
2. **После** закрытия TZ-X — реализовать Z-4, фиксируя hashes 11+ промптов как контрольные.
3. Интегрировать с существующим `TRANSCRIBE_MERGE_PROMPT_VERSION` mechanism.

---

## ТРЕК Z-5: E2E smoke test (ОТЛОЖЕН — зависит от стабилизации pipeline в git)

**НЕ реализуется в v2 как самостоятельный трек.**

**Причина 1:** В коде **уже существует** `tests/test_smoke.py` — пишет пустые mp3, прогоняет `init-db → ingest → worker --once`. Z-5 должно **расширить** существующий smoke, не создавать параллельный.

**Причина 2:** Точная grep-проверка показала, что real pipeline идёт через CLI / Service-объекты, не через функции `ingest_call/transcribe_call/analyze_call`, которых **не существует**. Реальные точки входа:
- `cmd_ingest` (cli.py:193) → `ingest_from_directory(session, recordings_dir=..., metadata_csv=..., limit=...)`
- `cmd_transcribe` → `TranscribeService(settings).run(session, limit=...)`
- `cmd_resolve` → `ResolveService(settings).run_with_progress(session, limit=...)` (**slой resolve между transcribe и analyze!** Был пропущен в v1)
- `cmd_analyze` → `AnalyzeService(settings).run(session, limit=...)`
- Per-call: `TranscribeService._transcribe_call(call)`

**Причина 3:** Codex прав, что E2E нельзя строить пока `question_catalog` и `deal_aware` ещё в активной разработке — каждый раз сдвигаются контракты на стыках. Лучше E2E **после** того, как catalog v2 и deal-aware writeback rollout закрыты и зафиксированы в git.

**Правильный план:**

1. Закрыть catalog v2 (этапы D-F) и deal-aware writeback rollout.
2. **После** этого — реализовать Z-5: расширить `tests/test_smoke.py`, добавить:
   - mock LLM client для analyze/classify steps
   - проверку контрактов на 4 стыках: transcribe→resolve, resolve→analyze, analyze→catalog, analyze→deal-aware
   - assertion на schema_version v2 propagation
   - assertion на prompt_version в LLM cache key
3. Использовать **реальные** API: `TranscribeService`, `ResolveService`, `AnalyzeService`.

---

## Использование субагентов

Для Z-1 и Z-2 (готовые сейчас) — субагенты не критичны, правки локальные. Можно использовать 1-2 для:

- **Sub-A**: точный расчёт actual sha256 текущего tenant_config_v1.json для заполнения `EXPECTED_TENANT_CONFIG_SHA256`.
- **Sub-B**: реализация Z-1 + Z-2 параллельно + общий audit pack.

---

## Граничные условия

— **Z-3, Z-4, Z-5 НЕ реализуются** в этом ТЗ. Они имеют жёсткие зависимости от других треков.
— **НЕ менять** signature `_merge_texts` (Z-1). 5 callsites не должны быть тронуты.
— **НЕ добавлять** sha256 pin-check в Stage15 (Z-2). Stage15 — это не tenant-aware gate. Правильное место — 2 script-callers.
— **НЕ использовать** `tests/fixtures/` файлы, которые ещё не существуют. Если нужен fixture — создавать новый.
— **Acceptance не через stable_runtime/** — это read-only зона.

---

## Если что-то непонятно

Создай `audits/_inbox/hygiene_zv2_<track>_clarifications_REQUEST_<timestamp>/QUESTIONS_FOR_CLAUDE.md`.

Вероятные точки уточнения:
- Z-2: должен ли pin-check **блокировать** скрипт при mismatch (raise RuntimeError) или только **logging** в warnings? Дмитрий выбирает политику.
- Z-1: формат `suspicious_drops` в `merge_meta.notes` — нужны ли дополнительные поля кроме count/total_chars/samples?

---

## Резюме v1 → v2

| Аспект | v1 | v2 |
|---|---|---|
| Структура | 5 правок одним блоком | 2 готовых трека + 3 отложенных с явными зависимостями |
| Z.1 (merge logging) | Изменить signature `_merge_texts` | Добавить counter в `_merge_variant_pair` (caller-side, без слома API) |
| Z.2 (tenant_config) | В Stage15 export gate | В 2 скриптах callers (Stage15 не tenant-aware) |
| Z.3 (negative corpus) | Сейчас | Отложен до закрытия TZ-Y трек Y-A |
| Z.4 (prompt drift) | 7 промптов | 11 промптов + отложен до закрытия TZ-X (избежать stale hash) |
| Z.5 (E2E smoke) | Новый файл с несуществующими API | Расширить существующий `tests/test_smoke.py` после стабилизации catalog/deal-aware |
| Acceptance | Через stable_runtime | Через tests/fixtures (read-only contract) |

После реализации Z-1 + Z-2 — есть observability merge drops и защита tenant_config от silent drift. Z-3, Z-4, Z-5 — в pipeline, делаются по очереди по мере закрытия зависимостей.
