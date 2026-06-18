from __future__ import annotations

import csv
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mango_mvp.customer_timeline import (
    CanonicalReadonlyTimelineConfig,
    CustomerTimelineSQLiteStore,
    build_canonical_readonly_customer_timeline,
    canonical_readonly_timeline_safety_contract,
)
from mango_mvp.customer_timeline.canonical_readonly_import import infer_brand, infer_offline_brand, split_ids, tallanto_match_class
from mango_mvp.customer_timeline.read_api import CustomerTimelineReadApi, CustomerTimelineReadApiConfig


NOW = datetime(2026, 5, 21, 9, 0, tzinfo=timezone.utc)


def test_infer_brand_cyrillic_v2_foton_root_and_cross_brand_fail_closed() -> None:
    assert infer_brand(["Фотона математика"], mode="legacy") == "foton"
    assert infer_brand(["Фотоны онлайн"], mode="cyrillic_v2") == "foton"
    assert infer_brand(["Фотону"], mode="cyrillic_v2") == "foton"
    assert infer_brand(["в Фотоне"], mode="cyrillic_v2") == "foton"
    assert infer_brand(["ЦДПФОТОН"], mode="cyrillic_v2") == "foton"
    assert infer_brand(["ЦИДПОФОТОН"], mode="cyrillic_v2") == "foton"
    assert infer_brand(["олимпиада МФТИ"], mode="legacy") == "unknown"
    assert infer_brand(["олимпиада МФТИ"], mode="cyrillic_v2") == "unpk"
    assert infer_brand(["Фотон и УНПК"], mode="legacy") == "unpk"
    assert infer_brand(["Фотон и УНПК"], mode="cyrillic_v2") == "unknown"
    assert infer_brand(["Фотон МФТИ"], mode="cyrillic_v2") == "unknown"
    assert infer_brand(["мотивация через фотончики"], mode="cyrillic_v2") == "unknown"
    assert infer_brand(["олимпиада Фотоний"], mode="cyrillic_v2") == "unknown"
    assert infer_offline_brand({"История": "клиент занимался у Фотона", "Филиал Tallanto": "МФТИ"}) == "foton"


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_runtime(path: Path, active_export_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"paths": {"active_export_root": str(active_export_root)}}, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_mail_bridge(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as con:
        con.execute(
            """
            CREATE TABLE candidate_phone_refs (
                candidate_key TEXT,
                normalized_phone TEXT,
                phone_match_class TEXT
            )
            """
        )
        con.execute(
            """
            CREATE TABLE candidate_mango_preview (
                candidate_key TEXT,
                mail_message_count INTEGER,
                first_mail_date_iso TEXT,
                last_mail_date_iso TEXT,
                bridge_status TEXT,
                blocked_reason TEXT,
                tallanto_id TEXT,
                amocrm_id TEXT
            )
            """
        )
        con.execute(
            """
            INSERT INTO candidate_phone_refs VALUES
            ('candidate-1', '+79161234567', 'strong_unique'),
            ('candidate-2', '+79160000000', 'ambiguous')
            """
        )
        con.execute(
            """
            INSERT INTO candidate_mango_preview VALUES
            ('candidate-1', 3, '2026-05-10T10:00:00+00:00', '2026-05-12T10:00:00+00:00', 'preview_ready', '', 'student-1', 'contact-1'),
            ('candidate-2', 5, '2026-05-10T10:00:00+00:00', '2026-05-12T10:00:00+00:00', 'blocked', 'ambiguous_phone', '', '')
            """
        )


def _write_mail_handoff(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as con:
        con.execute("CREATE TABLE mail_customer_links (id INTEGER PRIMARY KEY)")
        con.executemany("INSERT INTO mail_customer_links DEFAULT VALUES", [(), (), ()])


def _config(tmp_path: Path) -> CanonicalReadonlyTimelineConfig:
    runtime_root = tmp_path / "runtime_source"
    current_runtime = tmp_path / "stable_runtime" / "CURRENT_RUNTIME.json"
    _write_runtime(current_runtime, runtime_root)
    _write_csv(
        runtime_root / "master_contacts_ru.csv",
        [
            {
                "Телефон клиента": "+79161234567",
                "Email": "Parent@Example.COM ",
                "ФИО родителя": "Иван Петров",
                "Первый звонок": "2026-05-01 10:00:00",
                "Последний звонок": "2026-05-02 11:00:00",
                "Всего звонков в истории": "2",
                "Содержательных звонков в истории": "2",
                "Статус матчинга Tallanto": "exact_phone_single",
                "ID Tallanto": "student-1",
                "AMO contact IDs": "contact-1",
                "AMO lead IDs": "lead-1",
                "Краткая история общения": "Клиент выбирает курс математики.",
                "Рекомендуемый продукт": "Фотон математика",
            },
            {
                "Телефон клиента": "+79160000000",
                "ФИО родителя": "Мария Сидорова",
                "Первый звонок": "2026-05-03 10:00:00",
                "Последний звонок": "2026-05-03 10:00:00",
                "Всего звонков в истории": "0",
                "Статус матчинга Tallanto": "missing",
                "Нужна ручная проверка": "Да",
                "Рекомендуемый продукт": "УНПК",
            },
        ],
    )
    _write_csv(
        runtime_root / "master_calls_ru.csv",
        [
            {
                "ID звонка": "call-1",
                "Дата и время звонка": "2026-05-01 10:00:00",
                "Телефон клиента": "+79161234567",
                "Менеджер": "Анна",
                "Направление звонка": "Входящий",
                "Длительность, сек": "120",
                "Содержательный звонок": "Да",
                "Краткое резюме разговора": "Клиент спросил про расписание.",
                "Тип звонка": "sales_call",
            },
            {
                "ID звонка": "call-2",
                "Дата и время звонка": "2026-05-02 11:00:00",
                "Телефон клиента": "+79161234567",
                "Менеджер": "Анна",
                "Направление звонка": "Исходящий",
                "Длительность, сек": "90",
                "Содержательный звонок": "Да",
                "Краткое резюме разговора": "Клиент уточнил оплату.",
                "Тип звонка": "sales_call",
            },
        ],
    )
    amo_root = tmp_path / "amo"
    _write_csv(
        amo_root / "amo_contacts_snapshot.csv",
        [
            {
                "contact_id": "contact-1",
                "contact_name": "Иван Петров",
                "phones": "+7 916 123-45-67, +7 916 000-00-00",
                "emails": "parent@example.com",
                "linked_lead_ids": "lead-1",
                "responsible_user_name": "Анна",
                "created_at": "1770000000",
                "updated_at": "1770003600",
            }
        ],
    )
    _write_csv(
        amo_root / "amo_deals_snapshot.csv",
        [
            {
                "lead_id": "lead-1",
                "lead_name": "Фотон математика",
                "linked_contact_ids": "contact-1",
                "pipeline_name": "Сделки B2C",
                "status_name": "Переговоры",
                "created_at": "1770000000",
                "updated_at": "1770003600",
            }
        ],
    )
    mail_handoff = tmp_path / "mail" / "mail_customer_history_handoff.sqlite"
    mail_bridge = tmp_path / "mail" / "mail_mango_bridge_preview.sqlite"
    _write_mail_handoff(mail_handoff)
    _write_mail_bridge(mail_bridge)
    out_root = tmp_path / "product_data" / "customer_timeline" / "canonical_readonly_test"
    return CanonicalReadonlyTimelineConfig(
        project_root=tmp_path,
        out_root=out_root,
        timeline_db=out_root / "customer_timeline.sqlite",
        current_runtime_json=current_runtime,
        amo_contacts_csv=amo_root / "amo_contacts_snapshot.csv",
        amo_deals_csv=amo_root / "amo_deals_snapshot.csv",
        mail_handoff_db=mail_handoff,
        mail_bridge_db=mail_bridge,
        generated_at=NOW,
    )


def _table_count(db_path: Path, table: str) -> int:
    with sqlite3.connect(db_path) as con:
        return int(con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def test_builds_canonical_readonly_timeline_with_aggregate_coverage(tmp_path: Path) -> None:
    config = _config(tmp_path)

    report = build_canonical_readonly_customer_timeline(config)
    coverage = json.loads((config.out_root / "coverage_report.json").read_text(encoding="utf-8"))

    assert report["summary"]["total_customers"] == 2
    assert report["summary"]["with_mango_calls"] == 1
    assert report["summary"]["with_amo_context"] == 2
    assert report["summary"]["with_email_context"] == 1
    assert report["summary"]["duplicate_amo_contact_ids"] == 1
    assert report["summary"]["duplicate_amo_lead_ids"] == 1
    assert coverage["brand_counts"] == {"foton": 1, "unpk": 1}
    assert coverage["manual_review_reason_counts"]["shared_amo_contact_across_customers"] == 2
    assert coverage["manual_review_reason_counts"]["shared_amo_lead_across_customers"] == 2
    assert coverage["primary_read_blockers"]
    assert coverage["safety"]["timeline_primary_read_enabled_allowed"] is False
    assert coverage["safety"]["write_crm"] is False
    assert coverage["safety"]["telegram_import_enabled"] is False
    assert _table_count(config.timeline_db, "customer_identities") == 2
    assert _table_count(config.timeline_db, "timeline_events") >= 7


def test_canonical_family_phone_keeps_tallanto_students_split_and_conflicted(tmp_path: Path) -> None:
    config = _config(tmp_path)
    runtime_root = tmp_path / "runtime_source"
    _write_csv(
        runtime_root / "master_contacts_ru.csv",
        [
            {
                "Телефон клиента": "+79161234567",
                "Email": "parent-one@example.com",
                "ФИО родителя": "Иван Петров",
                "Первый звонок": "2026-05-01 10:00:00",
                "Последний звонок": "2026-05-02 11:00:00",
                "Всего звонков в истории": "2",
                "Содержательных звонков в истории": "2",
                "Статус матчинга Tallanto": "exact_phone_multiple",
                "Количество кандидатов Tallanto": "2",
                "ID Tallanto": "student-1",
                "Краткая история общения": "Первый ребенок интересуется математикой.",
                "Рекомендуемый продукт": "Фотон математика",
            },
            {
                "Телефон клиента": "+79161234567",
                "Email": "parent-two@example.com",
                "ФИО родителя": "Иван Петров",
                "Первый звонок": "2026-05-01 10:00:00",
                "Последний звонок": "2026-05-02 11:00:00",
                "Всего звонков в истории": "2",
                "Содержательных звонков в истории": "2",
                "Статус матчинга Tallanto": "exact_phone_multiple",
                "Количество кандидатов Tallanto": "2",
                "ID Tallanto": "student-2",
                "Краткая история общения": "Второй ребенок интересуется русским.",
                "Рекомендуемый продукт": "Фотон русский",
            },
        ],
    )

    report = build_canonical_readonly_customer_timeline(config)
    with sqlite3.connect(config.timeline_db) as con:
        phone_links = [
            json.loads(row[0])
            for row in con.execute("SELECT record_json FROM identity_links WHERE link_type = 'phone'")
        ]
        mango_events = [
            json.loads(row[0])
            for row in con.execute("SELECT record_json FROM timeline_events WHERE event_type = 'mango_call'")
        ]
        conflicts = [
            json.loads(row[0])
            for row in con.execute("SELECT record_json FROM timeline_conflicts WHERE conflict_type = 'shared_family_phone'")
        ]
        split_mappings = [
            json.loads(row[0])
            for row in con.execute("SELECT record_json FROM customer_id_mappings WHERE mapping_kind = 'split'")
        ]

    assert report["summary"]["total_customers"] == 2
    assert report["summary"]["manual_review_customers_estimated"] == 2
    assert report["summary"]["with_mango_calls"] == 2
    assert _table_count(config.timeline_db, "customer_identities") == 2
    assert _table_count(config.timeline_db, "customer_id_mappings") == 4
    assert len({item["customer_id"] for item in phone_links}) == 2
    assert len(mango_events) == 4
    assert {item["match_status"] for item in mango_events} == {"ambiguous"}
    assert len({item["customer_id"] for item in mango_events}) == 2
    assert len(conflicts) == 1
    assert {"tallanto_student:student-1", "tallanto_student:student-2"} <= set(conflicts[0]["entity_refs"])
    assert len(split_mappings) == 2
    assert len({item["old_customer_id"] for item in split_mappings}) == 1
    assert {item["new_customer_id"] for item in split_mappings} == {item["customer_id"] for item in phone_links}


def test_canonical_family_phone_splits_single_row_with_multiple_tallanto_ids(tmp_path: Path) -> None:
    config = _config(tmp_path)
    runtime_root = tmp_path / "runtime_source"
    _write_csv(
        runtime_root / "master_contacts_ru.csv",
        [
            {
                "Телефон клиента": "+79161234567",
                "Email": "parent@example.com",
                "ФИО родителя": "Иван Петров",
                "Первый звонок": "2026-05-01 10:00:00",
                "Последний звонок": "2026-05-02 11:00:00",
                "Всего звонков в истории": "2",
                "Содержательных звонков в истории": "2",
                "Статус матчинга Tallanto": "exact_phone_multiple",
                "Количество кандидатов Tallanto": "3",
                "ID Tallanto": "student-1;student-2;student-3",
                "Краткая история общения": "В семье несколько учеников.",
                "Рекомендуемый продукт": "Фотон математика",
            },
        ],
    )

    report = build_canonical_readonly_customer_timeline(config)
    with sqlite3.connect(config.timeline_db) as con:
        identities = [
            json.loads(row[0])
            for row in con.execute("SELECT record_json FROM customer_identities ORDER BY customer_id")
        ]
        phone_links = [
            json.loads(row[0])
            for row in con.execute("SELECT record_json FROM identity_links WHERE link_type = 'phone'")
        ]
        conflicts = [
            json.loads(row[0])
            for row in con.execute("SELECT record_json FROM timeline_conflicts WHERE conflict_type = 'shared_family_phone'")
        ]
        split_mappings = [
            json.loads(row[0])
            for row in con.execute("SELECT record_json FROM customer_id_mappings WHERE mapping_kind = 'split'")
        ]

    assert report["summary"]["total_customers"] == 3
    assert {item["identity_status"] for item in identities} == {"ambiguous"}
    assert len({item["customer_id"] for item in identities}) == 3
    assert {item["match_class"] for item in phone_links} == {"ambiguous"}
    assert len(conflicts) == 1
    assert {"tallanto_student:student-1", "tallanto_student:student-2", "tallanto_student:student-3"} <= set(
        conflicts[0]["entity_refs"]
    )
    assert len(split_mappings) == 3
    assert len({item["old_customer_id"] for item in split_mappings}) == 1


def test_canonical_no_exact_tallanto_match_is_partial_not_strong(tmp_path: Path) -> None:
    config = _config(tmp_path)
    runtime_root = tmp_path / "runtime_source"
    _write_csv(
        runtime_root / "master_contacts_ru.csv",
        [
            {
                "Телефон клиента": "+79161234567",
                "Email": "parent@example.com",
                "ФИО родителя": "Иван Петров",
                "Первый звонок": "2026-05-01 10:00:00",
                "Последний звонок": "2026-05-02 11:00:00",
                "Всего звонков в истории": "2",
                "Содержательных звонков в истории": "2",
                "Статус матчинга Tallanto": "no_exact_phone_match",
                "Краткая история общения": "Клиент выбирает курс математики.",
                "Рекомендуемый продукт": "Фотон математика",
            },
        ],
    )

    build_canonical_readonly_customer_timeline(config)
    with sqlite3.connect(config.timeline_db) as con:
        identity = json.loads(con.execute("SELECT record_json FROM customer_identities").fetchone()[0])
        tallanto_event = json.loads(
            con.execute("SELECT record_json FROM timeline_events WHERE event_type = 'tallanto_student_snapshot'").fetchone()[0]
        )

    assert tallanto_match_class("exact_phone_single").value == "strong_unique"
    assert tallanto_match_class("no_exact_phone_match").value == "unmatched"
    assert identity["identity_status"] == "partial"
    assert tallanto_event["match_status"] == "unmatched"


def test_canonical_brand_history_does_not_split_same_identity(tmp_path: Path) -> None:
    config = _config(tmp_path)
    runtime_root = tmp_path / "runtime_source"
    _write_csv(
        runtime_root / "master_contacts_ru.csv",
        [
            {
                "Телефон клиента": "+79161234567",
                "Email": "parent@example.com",
                "ФИО родителя": "Иван Петров",
                "Первый звонок": "2026-05-01 10:00:00",
                "Последний звонок": "2026-05-02 11:00:00",
                "Всего звонков в истории": "2",
                "Содержательных звонков в истории": "2",
                "Статус матчинга Tallanto": "exact_phone_single",
                "ID Tallanto": "student-1",
                "Краткая история общения": "Клиент выбирает курс.",
                "Рекомендуемый продукт": "Фотон математика",
            },
            {
                "Телефон клиента": "+79161234567",
                "Email": "parent@example.com",
                "ФИО родителя": "Иван Петров",
                "Первый звонок": "2026-05-01 10:00:00",
                "Последний звонок": "2026-05-02 11:00:00",
                "Всего звонков в истории": "2",
                "Содержательных звонков в истории": "2",
                "Статус матчинга Tallanto": "exact_phone_single",
                "ID Tallanto": "student-1",
                "Краткая история общения": "Тот же клиент интересуется олимпиадой МФТИ.",
                "Рекомендуемый продукт": "УНПК олимпиада МФТИ",
            },
        ],
    )

    build_canonical_readonly_customer_timeline(config)
    store = CustomerTimelineSQLiteStore.open_read_only(config.timeline_db, allowed_root=config.out_root)
    try:
        customers = store.list_customers("foton", limit=10)["items"]
        conflicts = store.summary()["counts"]["timeline_conflicts"]
    finally:
        store.close()

    assert len(customers) == 1
    assert customers[0]["summary"]["brands"] == ["foton", "unpk"]
    assert conflicts == 0


def test_reports_do_not_leak_raw_identity_values(tmp_path: Path) -> None:
    config = _config(tmp_path)
    build_canonical_readonly_customer_timeline(config)

    report_text = (config.out_root / "coverage_report.json").read_text(encoding="utf-8")
    manifest_text = (config.out_root / "source_manifest.json").read_text(encoding="utf-8")
    public_text = report_text + manifest_text

    assert "+79161234567" not in public_text
    assert "parent@example.com" not in public_text
    assert "Иван Петров" not in public_text
    assert "Мария Сидорова" not in public_text


def test_idempotent_rerun_keeps_store_counts_stable(tmp_path: Path) -> None:
    config = _config(tmp_path)

    first = build_canonical_readonly_customer_timeline(config)
    second = build_canonical_readonly_customer_timeline(config)

    assert first["summary"] == second["summary"]
    assert _table_count(config.timeline_db, "customer_identities") == 2
    assert _table_count(config.timeline_db, "timeline_events") >= 7


def test_built_timeline_is_readable_by_existing_read_api(tmp_path: Path) -> None:
    config = _config(tmp_path)
    build_canonical_readonly_customer_timeline(config)

    api = CustomerTimelineReadApi.open(CustomerTimelineReadApiConfig(timeline_db=config.timeline_db, allowed_root=config.out_root))
    try:
        customers = api.list_customers("foton", q="+79161234567")
        assert customers["items"]
        customer_id = customers["items"][0]["customer_id"]
        timeline = api.customer_timeline("foton", customer_id, limit=20)
    finally:
        api.close()

    event_types = [item["event_type"] for item in timeline["items"]]
    assert event_types.count("mango_call") == 2
    assert event_types.count("email_message") == 1


def test_output_root_must_not_be_under_stable_runtime(tmp_path: Path) -> None:
    config = _config(tmp_path)
    unsafe = CanonicalReadonlyTimelineConfig(
        project_root=tmp_path,
        out_root=tmp_path / "stable_runtime" / "customer_timeline",
        timeline_db=tmp_path / "stable_runtime" / "customer_timeline" / "customer_timeline.sqlite",
        current_runtime_json=config.current_runtime_json,
        amo_contacts_csv=config.amo_contacts_csv,
        amo_deals_csv=config.amo_deals_csv,
        mail_handoff_db=config.mail_handoff_db,
        mail_bridge_db=config.mail_bridge_db,
        generated_at=NOW,
    )

    with pytest.raises(ValueError, match="stable_runtime"):
        build_canonical_readonly_customer_timeline(unsafe)


def test_safety_contract_blocks_live_actions() -> None:
    safety = canonical_readonly_timeline_safety_contract(write_customer_timeline_db=True)

    assert safety["write_customer_timeline_db"] is True
    assert safety["write_crm"] is False
    assert safety["write_tallanto"] is False
    assert safety["send_email"] is False
    assert safety["send_messenger"] is False
    assert safety["run_asr"] is False
    assert safety["run_ra"] is False
    assert safety["stable_runtime_writes"] is False
    assert safety["raw_personal_values_in_reports"] is False


def test_split_ids_handles_spaces_and_common_separators() -> None:
    assert split_ids("lead-1 lead-2,lead-3|lead-4;lead-5") == [
        "lead-1",
        "lead-2",
        "lead-3",
        "lead-4",
        "lead-5",
    ]
