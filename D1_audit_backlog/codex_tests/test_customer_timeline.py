from __future__ import annotations

"""Бизнес-тесты ИСТОРИИ КЛИЕНТА (область 3).

Запуск (в среде Кодекса, с PYTHONPATH=src):
    PYTHONPATH=src python3 D1_audit_backlog/codex_tests/test_customer_timeline.py

Защищаемые инварианты области:
  - импорт строго read-only: контракты безопасности запрещают write_crm/write_tallanto/
    run_asr/send_* и любую запись в источник; импорт не лезет в сеть/subprocess;
  - PII (телефон, email, raw payload, пути к файлам) НЕ попадает в клиентский слой
    (audience='bot') и маскируется в менеджерском слое;
  - связывание личности корректно: один телефон у ДВУХ разных customer_id → конфликт
    'ambiguous_identity'; один телефон у одного клиента → конфликта нет.

Чистые функции тестируются напрямую. SQLite read_api тестируется на проекциях
(project_* — чистые), а живой store вынесен в TODO-скелет.

Модули под проверкой:
  - mango_mvp.customer_timeline.safety
  - mango_mvp.customer_timeline.read_api       (mask_*, project_*, redaction_summary, FORBIDDEN_OUTPUT_KEYS)
  - mango_mvp.customer_timeline.ingestion       (infer_identity_conflicts, identity_*, safety contract, SQL guard)
  - mango_mvp.customer_timeline.canonical_readonly_import (safety contract)
"""

from datetime import datetime, timezone

from mango_mvp.customer_timeline import safety
from mango_mvp.customer_timeline import read_api as ra
from mango_mvp.customer_timeline import ingestion as ing
from mango_mvp.customer_timeline.canonical_readonly_import import (
    canonical_readonly_timeline_safety_contract,
)
from mango_mvp.customer_timeline.contracts import (
    IdentityLink,
    IdentityLinkType,
    IdentityMatchClass,
    IdentityStatus,
)


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
# 3A. Контракты безопасности read-only
# --------------------------------------------------------------------------

def test_safety_contract_is_read_only() -> None:
    contract = safety.customer_timeline_safety_contract()
    # Не должен бросать — это валидный read-only контракт.
    raised = False
    try:
        safety.assert_customer_timeline_safety_contract(contract)
    except ValueError:
        raised = True
    check("base_safety_contract_valid", not raised, "валидный контракт упал на assert")
    for action in safety.blocked_live_actions():
        check(f"blocked_action_false[{action}]", contract.get(action) is False,
              f"{action}={contract.get(action)}")
    check("read_only_source_true", contract.get("read_only_source_systems") is True)


def test_safety_contract_rejects_write_flags() -> None:
    bad = dict(safety.customer_timeline_safety_contract())
    bad["write_crm"] = True  # подменили — должен упасть
    raised = False
    try:
        safety.assert_customer_timeline_safety_contract(bad)
    except ValueError:
        raised = True
    check("safety_rejects_write_crm", raised, "контракт с write_crm=True прошёл проверку")


def test_ingestion_safety_read_only_and_no_automerge() -> None:
    c = ing.timeline_ingestion_safety_contract()
    for key in ("write_crm", "write_tallanto", "send_email", "send_messenger",
                "run_asr", "run_ra", "network_calls", "subprocess_calls"):
        check(f"ingestion_{key}_false", c.get(key) is False, f"{key}={c.get(key)}")
    check("ingestion_source_read_only", c.get("source_reads_are_read_only") is True)
    check("ingestion_no_auto_merge", c.get("identity_conflicts_auto_merge") is False,
          "автослияние личностей не должно быть включено")
    check("ingestion_idempotent_upsert", c.get("idempotent_upsert") is True)


def test_canonical_readonly_never_writes_external() -> None:
    for flag in (True, False):
        c = canonical_readonly_timeline_safety_contract(write_customer_timeline_db=flag)
        for key in ("write_crm", "write_tallanto", "send_email", "send_messenger",
                    "run_asr", "run_ra", "stable_runtime_writes"):
            check(f"canonical_{key}_false[db={flag}]", c.get(key) is False, f"{key}={c.get(key)}")
        check(f"canonical_no_raw_pii_in_reports[db={flag}]",
              c.get("raw_personal_values_in_reports") is False)


def test_sql_write_keyword_guard() -> None:
    # Инвариант: источник читается query-only, write-ключевые слова детектируются.
    check("sql_select_clean", ing.contains_sql_write_keyword("SELECT * FROM t") is False)
    for stmt in ("INSERT INTO t VALUES (1)", "update t set x=1", "DROP TABLE t",
                 "ATTACH DATABASE x", "pragma writable_schema=1"):
        check(f"sql_write_detected[{stmt[:12]}]", ing.contains_sql_write_keyword(stmt) is True,
              f"не задетектирован: {stmt}")


# --------------------------------------------------------------------------
# 3B. PII не попадает в клиентский слой (audience='bot') и маскируется иначе
# --------------------------------------------------------------------------

def test_mask_phone_email_identifier() -> None:
    check("mask_phone_tail4", ra.mask_phone("+79161234567") == "+***4567",
          f"got={ra.mask_phone('+79161234567')}")
    check("mask_phone_short", ra.mask_phone("123") == "***", f"got={ra.mask_phone('123')}")
    check("mask_phone_none", ra.mask_phone(None) is None)
    em = ra.mask_email("ivanov@example.com")
    check("mask_email_hides_local", em is not None and "ivanov" not in em and "@example.com" in em,
          f"got={em}")
    ident = ra.mask_identifier("abcdef1234567")
    check("mask_identifier_partial", ident is not None and "abcdef1234567" != ident,
          f"got={ident}")


def test_project_identity_link_bot_audience_hides_value() -> None:
    item = {"link_id": "l1", "customer_id": "c1", "link_type": "phone",
            "link_value": "+79161234567", "source_system": "amocrm"}
    bot = ra.project_identity_link(item, audience="bot")
    check("bot_link_value_none", bot.get("link_value") is None,
          f"bot leaked phone: {bot.get('link_value')}")
    mgr = ra.project_identity_link(item, audience="manager")
    check("manager_link_value_masked", mgr.get("link_value") not in (None, "+79161234567"),
          f"manager value not masked: {mgr.get('link_value')}")


def test_project_bot_context_strips_ids_for_bot() -> None:
    item = {"chunk_id": "ch1", "customer_id": "c1", "opportunity_id": "o1", "event_id": "e1",
            "text": "клиент спрашивал про расписание", "allowed_for_bot": True}
    bot = ra.project_bot_context(item, audience="bot")
    check("bot_customer_id_hidden", bot.get("customer_id") is None,
          f"customer_id leaked: {bot.get('customer_id')}")
    check("bot_event_id_hidden", bot.get("event_id") is None,
          f"event_id leaked: {bot.get('event_id')}")
    check("bot_text_kept", bot.get("text") is not None, "текст контекста должен остаться")
    mgr = ra.project_bot_context(item, audience="manager")
    check("manager_keeps_ids", mgr.get("customer_id") == "c1")


def test_safe_mapping_drops_forbidden_keys() -> None:
    # Инвариант: raw_payload и пути к файлам не должны просочиться через summary.
    raw = {"ok": "видно", "raw_payload": "СЕКРЕТ", "audio_path": "/x/y.mp3", "path": "/z"}
    cleaned = ra.safe_mapping(raw)
    check("safe_mapping_keeps_ok", cleaned.get("ok") == "видно")
    for forbidden in ("raw_payload", "audio_path", "path"):
        check(f"safe_mapping_drops[{forbidden}]", forbidden not in cleaned,
              f"{forbidden} протёк: {cleaned}")


def test_forbidden_output_keys_cover_pii_payloads() -> None:
    must_have = {"raw_payload", "telegram_raw_update", "email_raw_body", "audio_bytes",
                 "audio_path", "transcript_path", "source_path"}
    missing = must_have - set(ra.FORBIDDEN_OUTPUT_KEYS)
    check("forbidden_keys_cover_pii", not missing, f"не хватает запретных ключей: {missing}")


def test_project_artifact_does_not_leak_path() -> None:
    art = {"artifact_id": "a1", "path": "/secret/file.eml", "sha256": "deadbeef"}
    projected = ra.project_artifact(art)
    check("artifact_no_path_field", "path" not in projected, f"path протёк: {projected}")
    check("artifact_has_path_bool", projected.get("has_path") is True,
          "должен быть только булев флаг наличия пути")


def test_redaction_summary_flags() -> None:
    s = ra.redaction_summary(bot_safe=True)
    for key in ("source_payload_removed", "artifact_paths_removed", "storage_rows_removed"):
        check(f"redaction_{key}_true", s.get(key) is True)
    check("redaction_bot_safe_propagated", s.get("bot_safe") is True)


# --------------------------------------------------------------------------
# 3C. Связывание личности: один человек ↔ один id; разные ↔ разные/конфликт
# --------------------------------------------------------------------------

def _link(customer_id: str, value: str, link_type=IdentityLinkType.PHONE) -> IdentityLink:
    return IdentityLink(
        tenant_id="t1",
        link_type=link_type,
        link_value=value,
        source_system="amocrm",
        source_ref=f"ref:{customer_id}:{value}",
        customer_id=customer_id,
        match_class=IdentityMatchClass.STRONG_UNIQUE,
    )


def _batch(links: list[IdentityLink]) -> ing.TimelineNormalizedBatch:
    rec = ing.TimelineSourceRecord(source_system="amocrm", source_ref="batch_ref", payload={})
    return ing.TimelineNormalizedBatch(source_record=rec, identity_links=tuple(links))


def test_same_phone_two_customers_is_conflict() -> None:
    # Один телефон закреплён за ДВУМЯ разными customer_id → ambiguous_identity.
    batch = _batch([
        _link("cust_A", "+79161234567"),
        _link("cust_B", "+79161234567"),
    ])
    conflicts = ing.infer_identity_conflicts([batch])
    types = {c["conflict_type"] for c in conflicts}
    check("two_customers_one_phone_conflict", "ambiguous_identity" in types,
          f"conflicts={[c['conflict_type'] for c in conflicts]}")


def test_same_phone_one_customer_no_conflict() -> None:
    # Один человек, один телефон, две записи (разные источники) → НЕТ конфликта.
    batch = _batch([
        _link("cust_A", "+79161234567"),
        _link("cust_A", "+79161234567"),
    ])
    conflicts = ing.infer_identity_conflicts([batch])
    check("one_customer_one_phone_no_conflict", len(conflicts) == 0,
          f"ложный конфликт: {[c['conflict_type'] for c in conflicts]}")


def test_different_phones_different_customers_no_conflict() -> None:
    # Два разных человека с разными телефонами → НЕ должны слипнуться в конфликт.
    batch = _batch([
        _link("cust_A", "+79161234567"),
        _link("cust_B", "+79169999999"),
    ])
    conflicts = ing.infer_identity_conflicts([batch])
    check("different_phones_no_conflict", len(conflicts) == 0,
          f"ложный конфликт: {[c['conflict_type'] for c in conflicts]}")


def test_identity_match_class_from_payload() -> None:
    check("match_class_default_strong",
          ing.identity_match_class_from_payload({}) == IdentityMatchClass.STRONG_UNIQUE)
    check("match_class_ambiguous",
          ing.identity_match_class_from_payload({"match_class": "ambiguous"}) == IdentityMatchClass.AMBIGUOUS)
    check("match_class_duplicate",
          ing.identity_match_class_from_payload({"resolution_status": "duplicate_merge_required"}) == IdentityMatchClass.DUPLICATE)


def test_identity_status_from_match() -> None:
    # Неоднозначный/дубль → статус ambiguous (не strong), даже если есть телефон.
    check("status_ambiguous",
          ing.identity_status_from_match(phone="+79161234567", email=None,
                                         match_class=IdentityMatchClass.AMBIGUOUS) == IdentityStatus.AMBIGUOUS)
    check("status_unmatched",
          ing.identity_status_from_match(phone=None, email=None,
                                         match_class=IdentityMatchClass.UNMATCHED) == IdentityStatus.UNMATCHED)
    check("status_strong_with_phone",
          ing.identity_status_from_match(phone="+79161234567", email=None,
                                         match_class=IdentityMatchClass.STRONG_UNIQUE) == IdentityStatus.STRONG)
    check("status_partial_without_identifiers",
          ing.identity_status_from_match(phone=None, email=None,
                                         match_class=IdentityMatchClass.STRONG_UNIQUE) == IdentityStatus.PARTIAL)


# --------------------------------------------------------------------------
# 3D. read_api на живом store — СКЕЛЕТ (требует customer_timeline.sqlite)
# TODO(codex): подставить фикстуру БД.
# --------------------------------------------------------------------------

def test_read_api_read_only_skeleton() -> None:
    """СКЕЛЕТ. CustomerTimelineReadApi требует read-only store и не имеет mutate-методов.

    Инвариант для проверки Кодексом:
      - CustomerTimelineReadApi(store с read_only=False) -> ValueError;
      - bot_context(audience='bot') не возвращает customer_id/phone/email/raw payload;
      - tenant_id фильтрует строки (нельзя достать чужого арендатора).

    TODO(codex): подставить фикстуру:
      1. собрать временную customer_timeline.sqlite через
         CustomerTimelineSQLiteStore (или взять тестовую из tests/);
      2. открыть CustomerTimelineReadApi.open(CustomerTimelineReadApiConfig(timeline_db=..., allowed_root=...));
      3. проверить, что health()['read_only'] is True и bot_context не содержит PII.
    """
    from mango_mvp.customer_timeline.read_api import CustomerTimelineReadApi

    class _FakeStore:
        read_only = False

    raised = False
    try:
        CustomerTimelineReadApi(_FakeStore())  # type: ignore[arg-type]
    except ValueError:
        raised = True
    check("read_api_rejects_writable_store", raised,
          "ReadApi принял store с read_only=False")
    # TODO(codex): добавить позитивный путь с реальной БД и проверкой отсутствия PII в bot_context.


def main() -> int:
    test_safety_contract_is_read_only()
    test_safety_contract_rejects_write_flags()
    test_ingestion_safety_read_only_and_no_automerge()
    test_canonical_readonly_never_writes_external()
    test_sql_write_keyword_guard()
    test_mask_phone_email_identifier()
    test_project_identity_link_bot_audience_hides_value()
    test_project_bot_context_strips_ids_for_bot()
    test_safe_mapping_drops_forbidden_keys()
    test_forbidden_output_keys_cover_pii_payloads()
    test_project_artifact_does_not_leak_path()
    test_redaction_summary_flags()
    test_same_phone_two_customers_is_conflict()
    test_same_phone_one_customer_no_conflict()
    test_different_phones_different_customers_no_conflict()
    test_identity_match_class_from_payload()
    test_identity_status_from_match()
    test_read_api_read_only_skeleton()
    print(f"PASS={_PASS}  FAIL={_FAIL}")
    for line in _FAILURES:
        print(line)
    return 1 if _FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
