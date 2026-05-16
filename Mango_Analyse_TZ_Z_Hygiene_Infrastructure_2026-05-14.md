# ТЗ-Z: Hygiene infrastructure — наблюдаемость, защита от регрессий, E2E

Дата: 2026-05-14
Автор: Claude (наставник Дмитрия по ИИ-проекту Mango Analyse)
Адресат: Codex как primary implementation agent в отдельном диалоге
Связанный документ: `Foton/Mango_Analyse_Data_Quality_Improvement_TZ_FOR_CODEX_2026-05-14.md` (системное ТЗ из 18 правок, этот трек — часть P3 hygiene)

---

## 0. Контекст и проблема

После полного аудита pipeline (14 мая) выявлены **5 инфраструктурных пробелов**, которые не блокируют текущую работу, но **накапливают долгосрочный технический долг** и риски регрессий. Каждый — small fix, но вместе они дают:
- Наблюдаемость потерь данных (где сейчас «исчезает молча»)
- Защиту от регрессий при правках промптов
- Защиту от негативной over-sanitization
- End-to-end smoke test pipeline'а

Этот ТЗ — **infrastructure track**, можно запускать параллельно с другими (ТЗ-X analyze.py, ТЗ-Y sanitizer/quality, catalog v2 D.2, deal-aware writeback rollout). Не пересекается с ними по файлам.

**Граница ответственности этого ТЗ:** `services/transcribe.py`, `quality/stage15_export_quality_gate.py`, `quality/bot_safety_frozen_corpus.py`, `tests/fixtures/bot_safety_*.jsonl`, новые тесты в `tests/`. **Не трогать** `services/analyze.py` или `services/resolve.py` (это ТЗ-X), не трогать `insights/sanitizers.py`, `quality/tenant_text_normalizer.py`, `quality/crm_text_quality_detector.py` (это ТЗ-Y).

---

## 1. Пять правок

### Правка Z.1 — _merge_texts logging

**Файл:** `src/mango_mvp/services/transcribe.py:1215-1265` (_merge_texts), 1178-1206 (_is_suspicious_chunk)

**Проблема:** Smoking gun из аудита ASR: `_merge_texts` при opcodes delete/insert удаляет «suspicious chunks» через `_is_suspicious_chunk` **без логирования** — что именно выкинуто и сколько токенов. Метрика «токенов выброшено» не сохраняется в `variants_json`. Это потеря содержания не-аудируемо.

**Что сделать:**

В `_merge_texts` накапливать counter и samples:

```python
def _merge_texts(text_a: str, text_b: str, ...) -> dict:
    # ... existing logic ...
    
    dropped_chunks = []
    total_dropped_chars = 0
    
    for opcode, ai, aj, bi, bj in matcher.get_opcodes():
        if opcode in ("delete", "insert"):
            chunk = text_a[ai:aj] if opcode == "delete" else text_b[bi:bj]
            if _is_suspicious_chunk(chunk):
                dropped_chunks.append({
                    "opcode": opcode,
                    "sample": chunk[:50],  # первые 50 char для аудита
                    "length": len(chunk),
                    "reason": _suspicious_chunk_reason(chunk),  # score / max_run / repetition_ratio
                })
                total_dropped_chars += len(chunk)
                continue  # выкидываем — как было раньше
            # ... existing handling for non-suspicious chunks ...
    
    return {
        "merged_text": merged,
        "selection": selection,
        "confidence": confidence,
        "notes": {
            "merge_dropped_suspicious_chunks_count": len(dropped_chunks),
            "merge_dropped_suspicious_chunks_total_chars": total_dropped_chars,
            "merge_dropped_suspicious_chunks_samples": dropped_chunks[:5],  # top-5
        }
    }
```

Это **не меняет** behaviour merge (контент по-прежнему удаляется), но **делает удаление наблюдаемым** — теперь в БД (через `transcript_variants_json.merge_meta.notes`) видно, сколько и что выкинуто.

**Дополнительно:** добавить добавлять `_suspicious_chunk_reason` функцию которая возвращает короткий код причины:

```python
def _suspicious_chunk_reason(chunk: str) -> str:
    """Возвращает причину, по которой chunk помечен как suspicious."""
    score = _calc_suspicious_score(chunk)
    max_run = _calc_max_run(chunk)
    rep_ratio = _calc_repetition_ratio(chunk)
    
    if score >= 4.0:
        return f"high_score_{score:.1f}"
    if max_run >= 8:
        return f"max_run_{max_run}"
    if rep_ratio >= 0.45:
        return f"repetition_{rep_ratio:.2f}"
    return "unknown"
```

**Тест-якорь:** `tests/test_transcribe_z1_merge_logging.py`:

```python
def test_suspicious_chunks_logged_in_merge_meta():
    text_a = "Это нормальный текст. Subtitle by DimaTorzok. Продолжение текста."
    text_b = "Это нормальный текст. Продолжение текста."
    result = _merge_texts(text_a, text_b, ...)
    notes = result["notes"]
    assert notes["merge_dropped_suspicious_chunks_count"] >= 1
    assert any("Torzok" in s["sample"] for s in notes["merge_dropped_suspicious_chunks_samples"])
    assert notes["merge_dropped_suspicious_chunks_total_chars"] > 20

def test_no_suspicious_chunks_no_logging():
    text_a = "Это полностью нормальный текст без артефактов."
    text_b = "Это нормальный текст без артефактов вообще."
    result = _merge_texts(text_a, text_b, ...)
    assert result["notes"]["merge_dropped_suspicious_chunks_count"] == 0
```

**Acceptance:**
1. Counter и samples появляются в `variants_json` для звонков, где сработал _is_suspicious_chunk.
2. На звонках без suspicious chunks — counter = 0, samples = [].
3. Behaviour merge не меняется (regression на 50 случайных звонках — итоговый merged_text идентичен до и после правки).

---

### Правка Z.2 — tenant_config sha256 pin-check в Stage15 gate

**Файл:** `src/mango_mvp/quality/stage15_export_quality_gate.py:183` (новый check)

**Проблема:** `tenant_config.py:34` считает sha256 от загруженного tenant_config (brand_aliases, products, и т.д.). Через `tenant_config_summary` (line 70) экспортируется. Но **в gate-checks нигде не верифицируется** — нет comparison `expected_sha256 == actual_sha256`. Это значит: любая правка `tenant_config_v1.json` (добавили новый brand alias, поменяли продукт) проходит **молча**, без сигнала. Это нарушает семантику frozen-gate.

В моих прошлых аудитах я подсвечивал это 3 раза, не закрыто.

**Что сделать:**

**Шаг 1.** Зафиксировать expected sha256 в `tenant_config.py` или в отдельной константе:

```python
# В tenant_config.py или src/mango_mvp/quality/tenant_config_pinning.py
EXPECTED_TENANT_CONFIG_SHA256 = "a3b8f9c2..."  # реальный hash на дату фиксации
EXPECTED_TENANT_CONFIG_VERSION = "tenant_config_v1_2026_05_14"
```

Hash считается на момент пины (Codex запускает `tenant_config_summary()`, получает текущий sha256, фиксирует как expected).

**Шаг 2.** Добавить check в Stage15 gate (`stage15_export_quality_gate.py:183`):

```python
def _check_tenant_config_sha256_matches_expected(summary: dict) -> tuple[bool, str]:
    """Tenant config sha256 не должен меняться без bump version."""
    from mango_mvp.quality.tenant_config_pinning import (
        EXPECTED_TENANT_CONFIG_SHA256,
        EXPECTED_TENANT_CONFIG_VERSION,
    )
    actual_sha = summary.get("tenant_config_sha256")
    actual_version = summary.get("tenant_config_version")
    
    if actual_version != EXPECTED_TENANT_CONFIG_VERSION:
        return False, f"tenant_config_version mismatch: expected {EXPECTED_TENANT_CONFIG_VERSION}, got {actual_version}"
    
    if actual_sha != EXPECTED_TENANT_CONFIG_SHA256:
        return False, f"tenant_config_sha256 changed without bump (expected {EXPECTED_TENANT_CONFIG_SHA256[:16]}..., got {actual_sha[:16]}...)"
    
    return True, "tenant_config_sha256_matches"
```

Добавить вызов этого check в общую last_gate список:
```python
checks = [
    # ... existing checks ...
    _check_tenant_config_sha256_matches_expected,
]
```

**Шаг 3.** Документировать процедуру **bump tenant_config**:

```markdown
# docs/TENANT_CONFIG_BUMP_PROCEDURE_2026-05-XX.md

При любом изменении tenant_config (новый brand alias, продукт, etc.):
1. Изменить `tenant_config_v1.json`
2. Запустить `python -m mango_mvp.quality.tenant_config_pinning --print-current`
3. Скопировать новый sha256 в `EXPECTED_TENANT_CONFIG_SHA256`
4. Поднять version: `tenant_config_v1` → `tenant_config_v2_2026_XX_XX`
5. Документировать в changelog: что изменилось
6. PR с обоими изменениями (json + pinning constants)
```

**Тест-якорь:** `tests/test_stage15_gate_z2_tenant_config_pin.py`:

```python
def test_gate_passes_with_matching_sha256():
    summary = {
        "tenant_config_sha256": EXPECTED_TENANT_CONFIG_SHA256,
        "tenant_config_version": EXPECTED_TENANT_CONFIG_VERSION,
    }
    passed, reason = _check_tenant_config_sha256_matches_expected(summary)
    assert passed is True

def test_gate_fails_with_mismatched_sha256():
    summary = {
        "tenant_config_sha256": "deadbeef",
        "tenant_config_version": EXPECTED_TENANT_CONFIG_VERSION,
    }
    passed, reason = _check_tenant_config_sha256_matches_expected(summary)
    assert passed is False
    assert "sha256 changed" in reason

def test_gate_fails_with_mismatched_version():
    summary = {
        "tenant_config_sha256": EXPECTED_TENANT_CONFIG_SHA256,
        "tenant_config_version": "tenant_config_v999_fake",
    }
    passed, reason = _check_tenant_config_sha256_matches_expected(summary)
    assert passed is False
    assert "version mismatch" in reason
```

**Acceptance:**
1. Stage15 gate включает new check.
2. Тесты зелёные.
3. На текущем tenant_config gate passes (sha256 matches).
4. Документ docs/ описывает bump procedure.

---

### Правка Z.3 — Negative-overblock layer в frozen corpus

**Файл:** `src/mango_mvp/quality/bot_safety_frozen_corpus.py`, `tests/fixtures/bot_safety_*.jsonl` (новый файл)

**Проблема:** Frozen corpus 1312 кейсов = past Claude/GPT findings (sanitizer должен поймать) + 200 real-data random + 12 hand-curated. **Нет negative-overblock guard layer** — то есть нет кейсов «эти числа/имена НЕ должны санитизироваться». Это **C8/F8 self-validation loop** из THREAT_MODEL: detector ловит ровно то, что sanitizer уже почистил, и пропускает классы, которые sanitizer не знает.

**Что сделать:**

**Шаг 1.** Создать новый файл фикстур `tests/fixtures/bot_safety_negative_overblock_cases.jsonl` с минимум 100 кейсами:

```jsonl
{"id": "neg_001", "text": "100% результат — это наша гарантия", "expected_sanitized": "100% результат — это наша гарантия", "class": "percent_not_money"}
{"id": "neg_002", "text": "5000 человек прошли курсы", "expected_sanitized": "5000 человек прошли курсы", "class": "non_money_number"}
{"id": "neg_003", "text": "ребёнок учится в 10 классе", "expected_sanitized": "ребёнок учится в 10 классе", "class": "grade_class"}
{"id": "neg_004", "text": "Иванов отлично преподавал физику", "expected_sanitized": "Иванов отлично преподавал физику", "class": "teacher_name_positive_context"}
{"id": "neg_005", "text": "класс 7 А расписание", "expected_sanitized": "класс 7 А расписание", "class": "class_letter"}
... (100+ кейсов)
```

Классы negative cases (примеры):
- `percent_not_money` — «100% результат», «95% посещаемость» (Y.1 правка покрывает)
- `non_money_number` — «5000 человек», «250 учеников», «100 уроков» (Y.1)
- `grade_class` — «10 класс», «7А», «11Б»
- `age_number` — «10 лет ребёнку», «12 лет ученику»
- `teacher_name_positive_context` — преподаватель упомянут позитивно, имя сохраняется
- `historical_year` — «2024 год», «учебный год 2025-26»
- `score_egé_oge` — «250 баллов», «80 баллов»
- `count_units` — «12 уроков», «3 часа», «5 занятий»
- `address_city` — «г. Москва», «Московская область» (не персональный адрес)
- `school_name` — «школа №1234», «лицей №16» (имя школы, не адрес)

**Шаг 2.** Расширить `bot_safety_frozen_corpus.py` для прогона negative cases:

```python
def run_negative_overblock_validation(fixture_path: Path) -> dict:
    """Прогон negative cases — проверка что sanitizer НЕ санитизирует то, что не должен."""
    results = {"total": 0, "passed": 0, "failed": [], "by_class": defaultdict(lambda: {"total": 0, "passed": 0})}
    
    with open(fixture_path) as f:
        for line in f:
            case = json.loads(line)
            actual_sanitized = sanitize_text(case["text"])
            class_name = case["class"]
            results["total"] += 1
            results["by_class"][class_name]["total"] += 1
            
            if actual_sanitized == case["expected_sanitized"]:
                results["passed"] += 1
                results["by_class"][class_name]["passed"] += 1
            else:
                results["failed"].append({
                    "id": case["id"],
                    "text": case["text"],
                    "expected": case["expected_sanitized"],
                    "actual": actual_sanitized,
                    "class": class_name,
                })
    
    return results
```

**Шаг 3.** Тест-якорь `tests/test_negative_overblock_z3.py`:

```python
def test_negative_overblock_corpus_passes():
    """Прогон 100+ negative cases — sanitizer не должен санитизировать ничего из них."""
    results = run_negative_overblock_validation(NEGATIVE_OVERBLOCK_FIXTURE)
    pass_rate = results["passed"] / results["total"]
    assert pass_rate >= 0.95, f"Negative overblock pass rate {pass_rate:.2%} < 95%: {len(results['failed'])} cases"
    # Дополнительно: каждый класс должен иметь pass_rate >= 80% (нет слабых классов)
    for class_name, stats in results["by_class"].items():
        class_pass_rate = stats["passed"] / stats["total"]
        assert class_pass_rate >= 0.80, f"Class {class_name} pass rate {class_pass_rate:.2%} < 80%"
```

**Acceptance:**
1. Фикстура `bot_safety_negative_overblock_cases.jsonl` создана с минимум 100 кейсами, минимум 10 классов покрыто (по 8-10 кейсов на класс).
2. Прогон даёт pass_rate ≥ 95% (после правки Y.1 значительная часть neg-cases уже passes — но Y.1 это отдельный трек, в этом ТЗ только инфраструктура).
3. Каждый класс pass_rate ≥ 80%.
4. Документ `docs/NEGATIVE_OVERBLOCK_CORPUS_RATIONALE.md` объясняет, почему этот слой нужен и как добавлять новые кейсы.

---

### Правка Z.4 — CI-проверка prompt drift

**Файл:** новый `tests/test_prompt_drift_protection.py`

**Проблема:** `LLMResponseCache` ключ = sha256(provider+model+`prompt_version`+полный `prompt`). Если Codex меняет system_prompt **без bump prompt_version** (например, фикс опечатки или добавление одной фразы), хеш `prompt` поля меняется → кеш инвалидируется (правильно), но `prompt_version` остаётся `v6` (неправильно). Это **silent drift**: версии в логах и кэше остаются прежними, но реальный промпт другой. Future debugging становится невозможным.

**Что сделать:**

Создать `tests/test_prompt_drift_protection.py` с зафиксированными hash'ами:

```python
import hashlib
from mango_mvp.services import analyze
from mango_mvp.services import transcribe
from mango_mvp.services import resolve
from mango_mvp.amocrm_runtime import deal_llm

EXPECTED_PROMPT_HASHES = {
    # analyze.py
    "SYSTEM_PROMPT_FULL_v6": "abc123...",  # реальный hash на дату фиксации
    "SYSTEM_PROMPT_COMPACT_v6": "def456...",
    # transcribe.py
    "MERGE_SYSTEM_PROMPT_v2": "ghi789...",
    "ROLE_ASSIGN_SYSTEM_PROMPT_v2": "jkl012...",
    # resolve.py
    "RESOLVE_SYSTEM_PROMPT_v2": "mno345...",
    "DIALOGUE_RESOLVE_SYSTEM_PROMPT_v2": "pqr678...",
    # deal_llm.py
    "deal_llm_v2": "stu901...",
}

def _hash_prompt(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def test_analyze_full_prompt_unchanged():
    actual = _hash_prompt(analyze.SYSTEM_PROMPT_FULL)
    expected = EXPECTED_PROMPT_HASHES["SYSTEM_PROMPT_FULL_v6"]
    assert actual == expected, (
        f"SYSTEM_PROMPT_FULL changed without bumping ANALYZE_PROMPT_VERSION_FULL. "
        f"If intentional: bump version to v7 and update EXPECTED_PROMPT_HASHES."
    )

def test_analyze_compact_prompt_unchanged():
    actual = _hash_prompt(analyze.SYSTEM_PROMPT_COMPACT)
    expected = EXPECTED_PROMPT_HASHES["SYSTEM_PROMPT_COMPACT_v6"]
    assert actual == expected, "SYSTEM_PROMPT_COMPACT changed without bumping version"

# ... аналогично для transcribe, resolve, deal_llm
```

**Процедура при изменении промпта:**

```markdown
# docs/PROMPT_BUMP_PROCEDURE.md

При изменении любого system_prompt:
1. Поднять prompt_version в соответствующем модуле (e.g. `ANALYZE_PROMPT_VERSION_FULL = "v7"`)
2. Запустить `python tests/test_prompt_drift_protection.py --print-current-hashes`
3. Скопировать новые hash'и в `EXPECTED_PROMPT_HASHES`
4. Обновить ключ — `"SYSTEM_PROMPT_FULL_v6"` → `"SYSTEM_PROMPT_FULL_v7"`
5. Документировать в changelog: что изменилось в промпте

Если pytest падает с error "promp changed without bumping version" — это интенциональное защитное поведение. Либо poднимай version, либо откати изменение промпта.
```

**Шаг 2.** Добавить в `.github/workflows/regression.yml` или существующий CI workflow:

```yaml
- name: Prompt drift protection
  run: pytest tests/test_prompt_drift_protection.py
```

**Acceptance:**
1. Тесты зелёные на текущих промптах (Codex фиксирует expected hash'и).
2. Если в промпте что-то менять — тест падает с понятной ошибкой.
3. Документ `docs/PROMPT_BUMP_PROCEDURE.md` объясняет процедуру.
4. CI integration done.

---

### Правка Z.5 — End-to-end smoke test

**Файл:** новый `tests/test_pipeline_smoke_e2e.py`

**Проблема:** Полного end-to-end (mp3 → AMO write) теста **нет**. Есть unit-тесты по слоям, но контракт «выход analyze попадает в catalog без потерь» / «output catalog → input для deal-aware» / «output deal-aware → AMO write» — не покрыт ни одним тестом. Это значит: каждое изменение в любом слое может молча сломать стык между слоями.

**Что сделать:**

Создать `tests/test_pipeline_smoke_e2e.py` с одним базовым smoke-сценарием:

```python
import json
from pathlib import Path

import pytest

from mango_mvp.services.ingest import ingest_call
from mango_mvp.services.transcribe import transcribe_call
from mango_mvp.services.analyze import analyze_call
from mango_mvp.question_catalog.classifier import classify_question
from mango_mvp.deal_aware.deal_attribution import attribute_call_to_deal


class _MockLLMClient:
    """Возвращает зафиксированные ответы по входу — не делает реальных LLM-вызовов."""
    def chat_completion(self, prompt, **kwargs):
        # Простой fixture-based mock
        if "Сколько стоит" in prompt:
            return {"theme_id": "theme:001_pricing", "confidence": 0.92, "reasoning": "test"}
        if "extracted_params" in prompt:
            return {"theme_id": "service:S2_unclear", "confidence": 0.55, "reasoning": "test"}
        return {"theme_id": "service:S5_general_consultation", "confidence": 0.5}


@pytest.fixture
def smoke_audio_fixture(tmp_path):
    """Создаёт mock mp3 fixture (в реальности заменён на existing test audio file)."""
    return Path("tests/fixtures/smoke_test_call_001.mp3")


def test_smoke_e2e_pipeline_call_to_writeback(smoke_audio_fixture, tmp_path):
    """Полный pipeline: ingest → transcribe → analyze → classify → attribute → writeback dry-run.
    
    Цель: не проверять глубину каждого слоя, а доказать что контракты на стыках стыкуются.
    """
    
    # 1. Ingest
    call_record = ingest_call(smoke_audio_fixture, metadata={"phone": "+79161234567"})
    assert call_record.id is not None
    assert call_record.audio_path is not None
    
    # 2. Transcribe (mocked ASR)
    transcript = transcribe_call(call_record, provider="mock")
    assert transcript is not None
    assert len(transcript) > 0
    
    # 3. Analyze (mocked LLM)
    analysis = analyze_call(call_record, transcript, llm_client=_MockLLMClient())
    assert "analysis_schema_version" in analysis
    assert analysis["analysis_schema_version"] == "v2"
    assert "structured_fields" in analysis
    
    # 4. Classify (catalog v2)
    questions = [
        {"text": "Сколько стоит подготовка к ЕГЭ?", "source": "call"}
    ]
    for q in questions:
        result = classify_question(q["text"], source=q["source"])
        assert result.theme_id in {"theme:001_pricing", "service:S2_unclear", "service:S5_general_consultation"}
    
    # 5. Deal attribution (deal-aware Phase 2)
    candidates = [
        {"deal_id": 12345, "is_active_deal": True, "is_duplicate_or_existing_client": False, "candidate_source_score": 100}
    ]
    attribution = attribute_call_to_deal(call_record, candidates)
    assert attribution["attribution_decision"] in {"linked_single_deal_candidate", "manual_review_multiple_active_deals", "manual_review_all_candidates_terminal"}
    
    # 6. Writeback dry-run (no real AMO call)
    # ... mock AMO writer ...
    assert True  # dry-run проходит


def test_smoke_e2e_schema_version_propagates():
    """analysis_schema_version="v2" попадает во все downstream слои."""
    # Здесь проверяется, что catalog читает schema_version из analyze output
    # и не работает с другой версией без миграции
    ...

def test_smoke_e2e_prompt_version_in_cache():
    """LLM cache использует prompt_version в ключе."""
    # Проверка что cache key содержит actual prompt_version
    ...
```

**Тесты сознательно простые** — это smoke, не deep. Главное — что контракты на стыках работают и любое breaking change на стыке падает в этом тесте.

**Шаг 2.** Создать тестовую фикстуру (если ещё нет):

```python
# tests/fixtures/smoke_test_call_001/ или smoke_test_call_001.mp3
# Это minimal fixture: либо реальный короткий звонок, либо .wav 5-секундный мок
```

**Acceptance:**
1. `pytest tests/test_pipeline_smoke_e2e.py` зелёные.
2. Если в любом слое pipeline что-то breaking — smoke тест падает с понятной ошибкой.
3. Документ `docs/E2E_SMOKE_TEST.md` объясняет назначение и как расширять.

---

## 2. Использование субагентов

Codex, можешь использовать до 6 параллельных субагентов:

- **Sub-A**: правка Z.1 (_merge_texts logging) — изолированный фикс в transcribe.py.
- **Sub-B**: правка Z.2 (tenant_config sha256 pin-check) + документация процедуры bump.
- **Sub-C**: правка Z.3 (negative-overblock corpus) — самая трудоёмкая. Нужно создать 100+ кейсов из 10+ классов. Использовать реальные данные из call_records для генерации realistic cases.
- **Sub-D**: правка Z.4 (prompt drift protection) — собрать текущие hash'и всех 7 промптов в проекте, документировать процедуру.
- **Sub-E**: правка Z.5 (E2E smoke test) — самая сложная по интеграционной части, требует понимания контрактов между слоями.
- **Sub-F**: общий sanity check + audit pack + регрессия на старых тестах.

---

## 3. Acceptance Criteria (вся правка Z)

### Hard requirements

1. Все 5 правок реализованы и тесты зелёные.
2. Существующие тесты `tests/test_transcribe*.py`, `tests/test_stage15*.py`, `tests/test_bot_safety*.py` всё ещё passed.
3. Z.1: counter и samples в variants_json, behaviour merge не меняется.
4. Z.2: gate включает check, текущий tenant_config passes, mismatch detected.
5. Z.3: 100+ negative cases, pass rate ≥ 95%, каждый класс ≥ 80%.
6. Z.4: 7 prompt hash'ей зафиксированы, тесты зелёные, CI integration done.
7. Z.5: smoke E2E pipeline проходит, контракты verified.

### Soft requirements

8. Документация всех 5 правок в docs/.
9. Производительность: Z.5 smoke тест выполняется ≤ 30 секунд на CI.

---

## 4. Deliverables

1. **Изменённые файлы:**
   - `src/mango_mvp/services/transcribe.py` (Z.1)
   - `src/mango_mvp/quality/stage15_export_quality_gate.py` (Z.2)
   - `src/mango_mvp/quality/tenant_config_pinning.py` (новый, Z.2)
   - `src/mango_mvp/quality/bot_safety_frozen_corpus.py` (Z.3)

2. **Новые фикстуры и тесты:**
   - `tests/fixtures/bot_safety_negative_overblock_cases.jsonl` (Z.3)
   - `tests/fixtures/smoke_test_call_001/` (Z.5)
   - `tests/test_transcribe_z1_merge_logging.py`
   - `tests/test_stage15_gate_z2_tenant_config_pin.py`
   - `tests/test_negative_overblock_z3.py`
   - `tests/test_prompt_drift_protection.py` (Z.4)
   - `tests/test_pipeline_smoke_e2e.py` (Z.5)

3. **CI integration:**
   - `.github/workflows/regression.yml` обновлён (если есть) с новыми тестами

4. **Документация:**
   - `docs/TENANT_CONFIG_BUMP_PROCEDURE_2026-05-XX.md`
   - `docs/PROMPT_BUMP_PROCEDURE.md`
   - `docs/NEGATIVE_OVERBLOCK_CORPUS_RATIONALE.md`
   - `docs/E2E_SMOKE_TEST.md`

5. **Audit pack:**
   - `audits/_inbox/hygiene_z_<timestamp>/`:
     - `AUDIT_SCOPE.md`
     - `MERGE_DROPS_SAMPLE.csv` — выборка из 50 звонков с merge_dropped chunks (Z.1)
     - `NEGATIVE_OVERBLOCK_PASS_RATE.json` — pass rate по классам (Z.3)
     - `PROMPT_HASHES_PINNED.json` — текущие hash'и всех 7 промптов (Z.4)
     - `E2E_SMOKE_RESULT.json` — итог smoke прогона (Z.5)

---

## 5. Граничные условия

— **НЕ менять** behaviour merge в Z.1 (только добавить logging — контент по-прежнему удаляется).
— **НЕ менять** структуру `findings` в Z.2 — только добавить новый check.
— **НЕ удалять** existing frozen corpus 1312 кейсов в Z.3 — только **дополнить** negative-overblock layer.
— **НЕ менять** промпты в Z.4 (только фиксировать hash'и — это защитный слой).
— **НЕ делать smoke test глубоким** в Z.5 — это smoke, не integration test. Главное — что контракты работают.

— **НЕ trogать** файлы вне scope этого ТЗ. Если кажется что нужно — спрашивай через QUESTIONS_FOR_CLAUDE.md.

---

## 6. Контроль качества

Перед сдачей Codex отвечает сам:

1. Все 5 правок реализованы?
2. Все 5 новых тестов зелёные?
3. Старые тесты всё ещё passed?
4. Z.1: counter появляется в variants_json для звонков с suspicious chunks?
5. Z.2: gate включает sha256 check?
6. Z.3: 100+ кейсов, 10+ классов, pass rate ≥ 95%?
7. Z.4: 7 prompt hash'ей зафиксированы и протестированы?
8. Z.5: smoke E2E проходит за ≤ 30 секунд?

---

## 7. Если что-то непонятно

Создай `audits/_inbox/hygiene_z_clarifications_REQUEST_<timestamp>/QUESTIONS_FOR_CLAUDE.md`.

Вероятные точки уточнения:
- Какие именно 7 промптов фиксировать в Z.4 — список зашит выше, но могут быть скрытые промпты
- Какая структура negative-overblock fixture в Z.3 — выше пример, но можно расширить
- Smoke E2E fixture в Z.5 — использовать существующий test audio или создать новый mock
- Tenant config pinning — где хранить EXPECTED_TENANT_CONFIG_SHA256 (в отдельном модуле или в tenant_config.py)

---

## 8. Ожидаемый эффект

После реализации ТЗ-Z:
- **Наблюдаемость потерь данных** — видно, сколько и что _merge_texts молча выкидывает.
- **Защита tenant_config** от silent drift — любое изменение требует явного bump version.
- **Защита от over-sanitization** — 100+ negative cases ловят регрессии санитайзера.
- **Защита от prompt drift** — изменение промпта без bump version падает в CI.
- **End-to-end защита** — любое breaking change на стыке слоёв ловится smoke тестом.

Это **infrastructure layer** для устойчивости pipeline. Каждая правка по отдельности — мелкая, вместе — значительное снижение долгосрочного техдолга.
