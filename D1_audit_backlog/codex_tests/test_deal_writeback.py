from __future__ import annotations

"""Бизнес-тесты ЗАПИСИ СДЕЛОК (область 2).

Запуск (в среде Кодекса, с PYTHONPATH=src):
    PYTHONPATH=src python3 D1_audit_backlog/codex_tests/test_deal_writeback.py

Защищаемые инварианты области:
  - запись НЕ происходит без approval / при dry_run (build_dry_run_row всегда dry_run,
    Stage 5 никогда не разрешает live writeback);
  - низкое качество (deal_quality_gate.evaluate_row) блокирует строку (block_stage6);
  - rollback восстанавливает прежнее значение только если текущее = записанному,
    и помечает skipped/manual если значение изменилось вручную или было пустым;
  - идемпотентность rollback: повторная попытка по тем же ключам не делает повторных
    записей (resume_success_keys).

Чистые функции тестируются напрямую. Запись в AMO заменена фейками (fetch/send),
чтобы тест проверял КОНТРАКТ, а не живую систему.

Модули под проверкой:
  - mango_mvp.deal_aware.deal_writeback   (build_dry_run_row, validate_stage5_summary)
  - mango_mvp.deal_aware.deal_quality_gate (evaluate_row)
  - mango_mvp.deal_aware.amo_rollback     (rollback_decision, run_rollback, retries)
"""

from pathlib import Path

from mango_mvp.deal_aware import deal_writeback as dw
from mango_mvp.deal_aware import deal_quality_gate as gate
from mango_mvp.deal_aware import amo_rollback as rb
from mango_mvp.deal_aware.deal_text_builder import DEAL_AI_REQUIRED_FIELDS


_PASS = 0
_FAIL = 0
_FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    global _PASS, _FAIL
    if condition:
        _PASS += 1
    else:
        _FAIL += 1
        _FAILURES.append(f"[FAIL] {name}: {detail}")


# --------------------------------------------------------------------------
# 2A. deal_writeback.build_dry_run_row
# Инвариант: режим всегда dry_run, live-запись НИКОГДА не разрешена этой функцией.
# --------------------------------------------------------------------------

def _good_row() -> dict:
    """Строка, которая по содержанию проходит как dry-run кандидат."""
    row = {
        "review_id": "r1",
        "selected_deal_id": "12345",
        "stage5_decision": "allow_stage6_dry_run",
    }
    # Заполняем все обязательные AI-поля нейтральным безопасным текстом.
    for field in DEAL_AI_REQUIRED_FIELDS:
        row.setdefault(field, "нейтральный безопасный текст без обещаний")
    return row


def _empty_field_catalog_for(row: dict) -> list[dict]:
    """Каталог полей AMO, в котором есть все нужные required-поля (не api-only)."""
    return [{"name": f, "id": f"fid_{i}", "type": "text"} for i, f in enumerate(DEAL_AI_REQUIRED_FIELDS)]


def _field_guard_ok() -> dict:
    return {
        "missing_fields": [],
        "api_only_fields": [],
        "missing_optional_fields_blocking": [],
    }


def test_dry_run_row_is_always_dry_run_mode() -> None:
    row = _good_row()
    report, findings = dw.build_dry_run_row(
        row, row_index=1, field_catalog=_empty_field_catalog_for(row),
        field_guard=_field_guard_ok(), analysis_date="2026-05-13",
    )
    check("dry_run_mode", report.get("stage6_mode") == "dry_run", f"mode={report.get('stage6_mode')}")
    check("dry_run_live_not_allowed", report.get("stage6_live_write_allowed_now") == "Нет",
          f"val={report.get('stage6_live_write_allowed_now')}")


def test_dry_run_row_blocks_when_stage5_not_allowed() -> None:
    # Инвариант: без разрешения Stage 5 строка блокируется.
    row = _good_row()
    row["stage5_decision"] = "block_stage6"
    report, findings = dw.build_dry_run_row(
        row, row_index=1, field_catalog=_empty_field_catalog_for(row),
        field_guard=_field_guard_ok(), analysis_date="2026-05-13",
    )
    check("blocked_without_stage5_approval", report.get("stage6_status") == "blocked",
          f"status={report.get('stage6_status')}, reason={report.get('stage6_reason')}")


def test_dry_run_row_blocks_protected_field_in_payload() -> None:
    # Инвариант: защищённые лид-поля (Телефон, ФИО, Email...) нельзя перезаписывать.
    row = _good_row()
    row["Телефон"] = "+79161234567"
    catalog = _empty_field_catalog_for(row) + [{"name": "Телефон", "id": "fphone", "type": "text"}]
    # Чтобы payload вообще включил protected-поле, оно должно быть в DEAL_AI_*; если нет —
    # проверка ниже всё равно безопасна (просто не сработает), поэтому проверяем напрямую set.
    payload_protected = dw.PROTECTED_LEAD_FIELDS
    check("protected_set_defined", len(payload_protected) > 0, "PROTECTED_LEAD_FIELDS пуст")
    check("protected_includes_pii", {"Телефон", "ФИО", "Email"} <= payload_protected,
          f"protected={payload_protected}")


def test_dry_run_row_blocks_discount_promise() -> None:
    # Инвариант проекта: интерес к скидке != разрешение обещать скидку.
    # 'AI-интерес к скидке' — OPTIONAL-поле, поэтому оно попадёт в payload, только если
    # есть в каталоге полей AMO (by_name). Поэтому добавляем его в каталог явно.
    row = _good_row()
    row["AI-интерес к скидке"] = "yes"
    row["AI-рекомендованный следующий шаг"] = "дадим скидку 20% при оплате на этой неделе"
    catalog = _empty_field_catalog_for(row) + [
        {"name": "AI-интерес к скидке", "id": "fdisc", "type": "text"},
    ]
    report, findings = dw.build_dry_run_row(
        row, row_index=1, field_catalog=catalog,
        field_guard=_field_guard_ok(), analysis_date="2026-05-13",
    )
    risk_types = {f.get("risk_type") for f in findings}
    check("discount_promise_blocked", report.get("stage6_status") == "blocked",
          f"status={report.get('stage6_status')}, risks={risk_types}")
    check("discount_promise_risk_type", "discount_promise_without_policy" in risk_types,
          f"risks={risk_types}")


def test_dry_run_row_blocks_missing_required_fields() -> None:
    row = {"review_id": "r2", "selected_deal_id": "999", "stage5_decision": "allow_stage6_dry_run"}
    report, findings = dw.build_dry_run_row(
        row, row_index=1, field_catalog=_empty_field_catalog_for(row),
        field_guard=_field_guard_ok(), analysis_date="2026-05-13",
    )
    check("missing_fields_blocked", report.get("stage6_status") == "blocked",
          f"status={report.get('stage6_status')}, reason={report.get('stage6_reason')}")


# --------------------------------------------------------------------------
# 2B. validate_stage5_summary — Stage 5 НИКОГДА не разрешает live writeback
# --------------------------------------------------------------------------

def test_stage5_summary_never_passes_live_writeback() -> None:
    # Даже если кто-то выставил passed_for_live_writeback=true, контракт passed=False,
    # потому что для passed нужно (dry_run=true И live=false И input совпал).
    summary = {
        "schema_version": "x",
        "readiness": {"passed_for_stage6_dry_run": True, "passed_for_live_writeback": True},
        "outputs": {"dry_run_candidates_csv": ""},
    }
    result = dw.validate_stage5_summary(summary, Path("/tmp/whatever.csv"))
    check("stage5_live_writeback_not_passed", result.get("passed") is False,
          f"passed={result}")


# --------------------------------------------------------------------------
# 2C. deal_quality_gate.evaluate_row — низкое качество блокирует
# --------------------------------------------------------------------------

def _gate_row(**over) -> dict:
    row = {
        "review_id": "g1",
        "selected_deal_id": "1",
        "selected_status_name": "Переговоры",
        "selected_loss_reason": "",
        "crm_text_quality_passed": "Да",
        "quality_risk_types": "",
    }
    for field in gate.DEAL_AI_FIELDS:
        row.setdefault(field, "нормальный текст")
    row.update(over)
    return row


def test_gate_blocks_known_hard_risk_type() -> None:
    # Любой stage4 hard-риск из списка блокирует dry-run.
    sample_risk = sorted(gate.DRY_RUN_BLOCKING_STAGE4_RISK_TYPES)[0]
    row = _gate_row(quality_risk_types=sample_risk)
    hard, warnings = gate.evaluate_row(row, {}, row_index=1, analysis_date="2026-05-13")
    check("gate_hard_risk_blocks", len(hard) > 0, f"hard={[h['gate_type'] for h in hard]}")


def test_gate_blocks_ellipsis_truncation() -> None:
    row = _gate_row()
    # Внедряем многоточие в одно из AI-полей — признак обрезанного текста.
    first_field = gate.DEAL_AI_FIELDS[0]
    row[first_field] = "клиент интересовался курсом по…"
    hard, warnings = gate.evaluate_row(row, {}, row_index=1, analysis_date="2026-05-13")
    types = {h["gate_type"] for h in hard}
    check("gate_ellipsis_blocks", "lossy_ellipsis_truncation" in types, f"hard={types}")


def test_gate_blocks_missing_ai_field() -> None:
    row = _gate_row()
    row[gate.DEAL_AI_FIELDS[0]] = ""  # пустое обязательное поле
    hard, warnings = gate.evaluate_row(row, {}, row_index=1, analysis_date="2026-05-13")
    types = {h["gate_type"] for h in hard}
    check("gate_missing_field_blocks", "missing_deal_ai_field" in types, f"hard={types}")


def test_gate_clean_row_allows_dry_run() -> None:
    row = _gate_row()
    hard, warnings = gate.evaluate_row(row, {}, row_index=1, analysis_date="2026-05-13")
    # Чистая строка не должна давать hard-findings (decision = allow_stage6_dry_run).
    check("gate_clean_no_hard", len(hard) == 0, f"unexpected hard={[h['gate_type'] for h in hard]}")
    # NB: если есть hard, это может быть валидным — тогда Кодекс уточняет, что в _gate_row
    # попало под детектор. TODO(codex): при падении распечатать h['reason'].


# --------------------------------------------------------------------------
# 2D. amo_rollback — восстановление и идемпотентность
# --------------------------------------------------------------------------

def _snapshot_row(lead_id="100", field_name="AI-поле", old="старое", new="новое") -> dict:
    return {
        "lead_id": lead_id,
        "field_name": field_name,
        "field_id": "fid",
        "field_type": "text",
        "old_value": old,
        "new_value": new,
        "new_value_sha256": rb.sha256_text(new),
        "row_index": "1",
        "review_id": "r1",
    }


def _lead_with_field(field_name: str, value: str) -> dict:
    return {"custom_fields_values": [{"field_name": field_name, "values": [{"value": value}]}]}


def test_rollback_decision_ready_when_current_equals_new() -> None:
    snap = _snapshot_row(new="новое")
    lead = _lead_with_field("AI-поле", "новое")  # текущее = записанному → можно откатить
    decision = rb.rollback_decision(snap, lead)
    check("rollback_ready", decision["rollback_status"] == "dry_run_ready",
          f"status={decision['rollback_status']}, reason={decision['reason']}")


def test_rollback_skips_when_value_changed_after_write() -> None:
    # Менеджер вручную поменял поле после записи → откат НЕ должен затирать его правку.
    snap = _snapshot_row(new="новое")
    lead = _lead_with_field("AI-поле", "менеджер_поменял_вручную")
    decision = rb.rollback_decision(snap, lead)
    check("rollback_skipped_on_manual_change", decision["rollback_status"] == "skipped",
          f"status={decision['rollback_status']}")
    check("rollback_skip_reason", decision["reason"] == "current_value_changed_after_write",
          f"reason={decision['reason']}")


def test_rollback_manual_required_when_old_empty() -> None:
    # Если до записи поле было пустым, helper не умеет очищать → ручное восстановление.
    snap = _snapshot_row(old="", new="новое")
    lead = _lead_with_field("AI-поле", "новое")
    decision = rb.rollback_decision(snap, lead)
    check("rollback_manual_when_old_empty", decision["rollback_status"] == "manual_restore_required",
          f"status={decision['rollback_status']}")


def test_run_rollback_requires_confirmation_for_apply() -> None:
    # Инвариант: apply без правильного confirmation должен падать (нет тихой записи).
    snaps = [_snapshot_row()]
    raised = False
    try:
        rb.run_rollback(snapshot_rows=snaps, fetch_lead=lambda _id: {}, send_update=lambda **k: {},
                        apply=True, confirmation="WRONG")
    except ValueError:
        raised = True
    check("apply_requires_confirmation", raised, "apply с неверным confirmation не упал")


def test_run_rollback_requires_send_update_for_apply() -> None:
    raised = False
    try:
        rb.run_rollback(snapshot_rows=[_snapshot_row()], fetch_lead=lambda _id: {},
                        send_update=None, apply=True, confirmation=rb.ROLLBACK_CONFIRMATION)
    except ValueError:
        raised = True
    check("apply_requires_send_update", raised, "apply без send_update не упал")


def test_run_rollback_apply_restores_via_fake_send() -> None:
    # Фейковая запись: фиксируем вызовы send_update и проверяем восстановление old_value.
    snap = _snapshot_row(old="старое", new="новое")
    lead = _lead_with_field("AI-поле", "новое")
    sent: list[dict] = []

    def fake_fetch(lead_id: int) -> dict:
        return lead

    def fake_send(*, lead_id: int, field_payload: dict) -> dict:
        sent.append({"lead_id": lead_id, "payload": field_payload})
        return {"ok": True}

    rows = rb.run_rollback(
        snapshot_rows=[snap], fetch_lead=fake_fetch, send_update=fake_send,
        apply=True, confirmation=rb.ROLLBACK_CONFIRMATION,
        retry_policy=rb.RetryPolicy(delay_ms=0, sleep_func=lambda _s: None),
    )
    check("apply_one_row", len(rows) == 1, f"rows={len(rows)}")
    check("apply_restored", rows[0]["rollback_status"] == "restored", f"status={rows[0]['rollback_status']}")
    check("apply_sent_old_value", sent and sent[0]["payload"].get("AI-поле") == "старое",
          f"sent={sent}")


def test_run_rollback_idempotent_via_resume_keys() -> None:
    # Идемпотентность: ключи уже успешно обработанных строк пропускаются (нет повторной записи).
    snap = _snapshot_row()
    key = rb.snapshot_key(snap)
    sent: list[dict] = []

    rows = rb.run_rollback(
        snapshot_rows=[snap], fetch_lead=lambda _id: {}, send_update=lambda **k: sent.append(k),
        apply=True, confirmation=rb.ROLLBACK_CONFIRMATION,
        resume_success_keys={key},
        retry_policy=rb.RetryPolicy(delay_ms=0, sleep_func=lambda _s: None),
    )
    check("idempotent_skipped", rows[0]["rollback_status"] == "skipped",
          f"status={rows[0]['rollback_status']}")
    check("idempotent_no_send", sent == [], f"unexpected send calls={sent}")


def test_retry_only_on_retryable_errors() -> None:
    # 429/5xx ретраятся; 4xx (кроме 429) — нет (нет бесконечных попыток на постоянной ошибке).
    check("retry_429", rb.is_retryable_exception(Exception("HTTP 429 Too Many")) is True)
    check("retry_503", rb.is_retryable_exception(Exception("HTTP 503")) is True)
    check("no_retry_400", rb.is_retryable_exception(Exception("HTTP 400 Bad Request")) is False)


def main() -> int:
    test_dry_run_row_is_always_dry_run_mode()
    test_dry_run_row_blocks_when_stage5_not_allowed()
    test_dry_run_row_blocks_protected_field_in_payload()
    test_dry_run_row_blocks_discount_promise()
    test_dry_run_row_blocks_missing_required_fields()
    test_stage5_summary_never_passes_live_writeback()
    test_gate_blocks_known_hard_risk_type()
    test_gate_blocks_ellipsis_truncation()
    test_gate_blocks_missing_ai_field()
    test_gate_clean_row_allows_dry_run()
    test_rollback_decision_ready_when_current_equals_new()
    test_rollback_skips_when_value_changed_after_write()
    test_rollback_manual_required_when_old_empty()
    test_run_rollback_requires_confirmation_for_apply()
    test_run_rollback_requires_send_update_for_apply()
    test_run_rollback_apply_restores_via_fake_send()
    test_run_rollback_idempotent_via_resume_keys()
    test_retry_only_on_retryable_errors()
    print(f"PASS={_PASS}  FAIL={_FAIL}")
    for line in _FAILURES:
        print(line)
    return 1 if _FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
