# ТЗ-Z-v3: Delta-исправления к TZ-Z-v2 после 2-го реверс-аудита Codex

Дата: 2026-05-15
Дополняет (не заменяет): `Mango_Analyse_TZ_Z_v2_Hygiene_2026-05-15.md`
Адресат: Codex как primary implementation agent

---

## Контекст

Codex провёл 2-й реверс-аудит TZ-Z-v2 и нашёл 6 уточнений. Структура (Z-1 готово сейчас, Z-2 готово сейчас, Z-3/Z-4/Z-5 отложены) сохраняется. Меняются только детали реализации Z-1 и Z-2.

---

## ТРЕК Z-1: Delta-правки

### Z-1.1 — НЕ превращать `notes` в dict

**Было в v2:**
```python
return {
    "text": merged_text,
    "notes": {
        "suspicious_drops": {...},
    },
}
```

**Проблема:** сейчас `notes` в `merge_meta` — **строка**, не dict. Тесты, скрипты, debugging кода ожидают строку. Превращение в dict сломает backward compat.

**Стало в v3:**

`notes` оставить строкой. Добавить **отдельное** поле `suspicious_drops` на том же уровне:

```python
return {
    "text": merged_text,
    "selection": selection,
    "confidence": confidence,
    "provider": "rule",
    "notes": "rule merge applied",  # строка как было
    "similarity": similarity,
    "suspicious_drops": {  # NEW: отдельное поле
        "count": int,
        "total_chars": int,
        "samples": list,
    },
}
```

Это backward compat: downstream код, который читает `notes` как строку, продолжает работать. Новое поле `suspicious_drops` появляется отдельно — кто хочет, тот его читает.

---

### Z-1.2 — `_count_suspicious_drops` точно повторять merge logic

**Было в v2:**
```python
matcher = SequenceMatcher(a=primary_tokens, b=secondary_tokens)
# raw tokens сравнение
```

**Проблема:** Реальный `_merge_texts` в transcribe.py:1230 использует:
- **Нормализованные токены** (через `_normalize_token` или подобный), не сырые
- `SequenceMatcher(autojunk=False)`

Если `_count_suspicious_drops_in_merge` использует другие токены или другой autojunk — counter будет расходиться с реальным merge result.

**Стало в v3:**

Codex точно повторяет токенизацию и SequenceMatcher arguments из `_merge_texts`:

```python
def _count_suspicious_drops_in_merge(self, primary_text: str, secondary_text: str) -> dict:
    """ВНИМАНИЕ: должна точно повторять токенизацию и matcher-args из _merge_texts."""
    # Использовать TE ЖЕ функции токенизации что и _merge_texts:
    primary_tokens = self._normalize_tokens_for_merge(primary_text)  # точное имя функции из реального кода
    secondary_tokens = self._normalize_tokens_for_merge(secondary_text)
    
    # Точно те же arguments что в _merge_texts:1230
    matcher = SequenceMatcher(a=primary_tokens, b=secondary_tokens, autojunk=False)
    
    # ... остальное счётчика ...
```

**Acceptance addition:** на 20 тестовых merge case'ах counter совпадает с реальным merge (т.е. количество выкинутых suspicious кусков соответствует фактическим opcodes).

---

### Z-1.3 — 4 callsites, не 5

**Было в v2:** «5 callsites _merge_texts».

**Стало в v3:** по фактическому коду — **4 callsites** внутри `_merge_variant_pair`. Минорная поправка, не критичная.

---

### Z-1.4 — Acceptance fixtures: не 50 из tests/fixtures

**Было в v2:** «На 50 случайных звонках из `tests/fixtures/`...»

**Проблема:** `tests/fixtures/` содержит только 4 файла. 50 звонков там нет.

**Стало в v3:** один из вариантов:
- (a) **Создать синтетическую выборку** в `tests/fixtures/transcribe_merge_corpus_z1/` — 30-50 сгенерированных пар transcripts (известные artifacts типа DimaTorzok-loop, нормальные звонки, edge cases). Документ что в каждой паре ожидается counter > 0 vs counter = 0.
- (b) **Безопасная копия данных** из `stable_runtime/`, но **читаем only**, в отдельный `audits/_inbox/transcribe_z1_data_<timestamp>/`.

Вариант (a) предпочтительнее — meaningful test cases, не зависит от production data.

**Acceptance updated:**
- Тестовая фикстура `tests/fixtures/transcribe_merge_corpus_z1/` создана с минимум 20 парами.
- Counter > 0 на парах с known artifacts.
- Counter = 0 на чистых парах.
- `merged_text` идентичен до и после правки (verified на этих 20 парах).

---

## ТРЕК Z-2: Delta-правки

### Z-2.1 — Политика блок/warn явная

**Было в v2:**
```python
pin_passed, pin_reason = check_tenant_config_pin(tenant_summary)
if not pin_passed:
    raise RuntimeError(f"tenant_config pin failed: {pin_reason}")
    # Или: result["warnings"].append(...)
```

Я оставил выбор Дмитрию. Codex прав: нужно **явно решить и зафиксировать**.

**Стало в v3:**

**Политика по контексту запуска:**

```python
# В каждом скрипте — отдельная политика, явно зафиксирована

# scripts/run_crm_writeback_quality_gate.py:
TENANT_CONFIG_PIN_MODE = "strict"  # для live writeback — блокируем
if not pin_passed:
    raise RuntimeError(f"tenant_config pin failed: {pin_reason}")

# scripts/build_post_backfill_amo_ready_export.py:
TENANT_CONFIG_PIN_MODE = "strict_if_live_else_warn"
if write_mode == "live":
    raise RuntimeError(f"...")
elif write_mode in {"dry_run", "preview"}:
    warnings.append({"type": "tenant_config_drift", "message": pin_reason})
```

**Принцип:** для **live/writeback** в реальный AMO — `strict` (блок). Для **dry-run/preview/audit** — `warn`. Это даёт защиту в production без ломания локальных проверок.

---

### Z-2.2 — `--print-current` команда реально работающая

**Было в v2:** упомянули команду `python -m mango_mvp.productization.tenant_config_pinning --print-current`, но в коде такого блока нет.

**Стало в v3:** в файл `tenant_config_pinning.py` явно добавить:

```python
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-current", action="store_true", help="Print current tenant_config sha256")
    args = parser.parse_args()
    
    if args.print_current:
        from mango_mvp.productization.tenant_config import load_tenant_config, DEFAULT_TENANT_CONFIG_PATH
        result = load_tenant_config(DEFAULT_TENANT_CONFIG_PATH)
        print(f"path: {result.path}")
        print(f"sha256: {result.sha256}")
        print(f"tenant_id: {result.config.get('tenant_id')}")
        print(f"schema_version: {result.config.get('schema_version')}")
        print()
        print(f"To pin this hash, update EXPECTED_TENANT_CONFIG_SHA256 in this file.")
```

Тогда документация `docs/TENANT_CONFIG_BUMP_PROCEDURE` имеет рабочую команду.

---

### Z-2.3 — Перед стартом Z-2 стабилизировать `test_post_backfill_amo_ready_export`

**Codex:** «В test_post_backfill_amo_ready_export.py сейчас 1 падение, связанное с текущей разработкой истории контакта, не с этим ТЗ напрямую».

**Стало в v3:** перед стартом интеграции Z-2 в `build_post_backfill_amo_ready_export.py` Codex проверяет что test_post_backfill_amo_ready_export.py зелёный (после стабилизации параллельной разработки). Если красный — Z-2 интеграция **только** в `run_crm_writeback_quality_gate.py`, второй скрипт ждёт.

---

## ТРЕК Z-4: Update — список промптов вырос

**Было в v2:** 11 промптов в списке.

**Стало в v3:** `theme_assigner_llm.py` уже создан в catalog v2 D.1. **12+ промптов** теперь. Перед Z-4 (когда будет делаться, после TZ-X) — Codex заново сканирует проект `grep -r "SYSTEM_PROMPT\|PROMPT_TEMPLATE\|PROMPT_VERSION"` для актуального списка.

---

## ТРЕК Z-3, Z-5: подтверждение, что отложены

Z-3 ждёт TZ-Y (`100% результат` сейчас санитизируется на текущем коде).
Z-5 ждёт стабилизации catalog/deal-aware в git + расширяет существующий `tests/test_smoke.py`.

---

## Resume v2 → v3

| Аспект | v2 | v3 |
|---|---|---|
| Z-1 notes structure | `notes: dict` | `notes: str` + отдельное `suspicious_drops` field |
| Z-1 token matching | raw tokens + default matcher | normalized tokens + `autojunk=False` (повторяет _merge_texts) |
| Z-1 callsites count | 5 | 4 (минорная поправка) |
| Z-1 acceptance fixtures | 50 из tests/fixtures (не существует) | 20+ синтетических в новой `tests/fixtures/transcribe_merge_corpus_z1/` |
| Z-2 политика | Выбор Дмитрия | strict для live, warn для dry_run |
| Z-2 print-current | Упомянут в doc, нет в коде | Реализован `__main__` блок |
| Z-2 prerequisite | — | Стабилизировать test_post_backfill_amo_ready_export перед стартом 2-го скрипта |
| Z-4 список промптов | 11 | 12+ (theme_assigner_llm.py добавился) — пересканировать |
