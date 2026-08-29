from __future__ import annotations

import importlib.util
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from openpyxl import load_workbook


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_owner_gate_semantic_sample.py"
SPEC = importlib.util.spec_from_file_location("audit_owner_gate_semantic_sample", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _review_db(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.executescript(
        """
        CREATE TABLE family_links_v1 (
          tenant_id TEXT, family_id TEXT, customer_id TEXT, child_key TEXT,
          canonical_name TEXT, grades_json TEXT, subjects_json TEXT, status TEXT, brand TEXT
        );
        CREATE TABLE family_members_v1 (
          tenant_id TEXT, family_id TEXT, customer_id TEXT, membership_status TEXT
        );
        CREATE TABLE customer_opportunities (
          tenant_id TEXT, customer_id TEXT, opportunity_id TEXT, title TEXT,
          status TEXT, opened_at TEXT
        );
        CREATE TABLE customer_purchases_v1 (
          tenant_id TEXT, customer_id TEXT, period TEXT, money_kind TEXT,
          total_in REAL, last_purchase_at TEXT, deals_cnt INTEGER
        );
        CREATE TABLE derived_signals (tenant_id TEXT, customer_id TEXT);
        CREATE TABLE timeline_events (
          tenant_id TEXT, customer_id TEXT, event_id TEXT, event_at TEXT,
          event_type TEXT, source_system TEXT, direction TEXT, subject TEXT,
          summary TEXT, text_preview TEXT, source_ref TEXT, superseded_by TEXT,
          record_json TEXT
        );
        CREATE TABLE timeline_conflicts (
          tenant_id TEXT, conflict_id TEXT, conflict_type TEXT, severity TEXT,
          status TEXT, created_at TEXT, record_json TEXT
        );
        """
    )
    con.execute(
        "INSERT INTO family_links_v1 VALUES (?,?,?,?,?,?,?,?,?)",
        ("foton", "family:1", "customer:1", "child:1", "Ученик", '["8"]', '["математика"]', "confident", "foton"),
    )
    con.execute(
        "INSERT INTO customer_opportunities VALUES (?,?,?,?,?,?)",
        ("foton", "customer:1", "lead:1", "Курс", "open", "2026-07-01"),
    )
    con.execute(
            "INSERT INTO customer_purchases_v1 VALUES (?,?,?,?,?,?,?)",
        ("foton", "customer:1", "all_time", "fact", 50000, "2026-07-02", 1),
    )
    con.executemany(
        "INSERT INTO timeline_events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            ("foton", "customer:1", "event:mail", "2026-07-03", "email_message", "mail", "inbound", "Вопрос", "Коротко", "Полный исходный текст письма", "mail:1", None, '{}'),
            ("foton", "customer:1", "event:visit", "2026-07-04", "tallanto_attendance", "tallanto", "system", "математика", "Посещение", "", "tallanto:1", None, '{}'),
            ("foton", "customer:1", "event:future-visit", "2030-01-01", "tallanto_attendance", "tallanto_attendance_api", "system", "физика", "Будущее расписание", "", "tallanto:future", None, '{"record":{"attendance_confirmed":false}}'),
            ("foton", "customer:1", "event:student", "2026-07-04", "tallanto_student_snapshot", "tallanto_snapshot", "system", "Ученик", "", "", "tallanto:student:1", None, '{"record":{"payload":{"student_type":"8_klass"}}}'),
            ("foton", "customer:other", "event:other", "2026-07-05", "email_message", "mail", "inbound", "Чужое", "Не должно попасть", "Чужой полный текст", "mail:other", None, '{}'),
        ],
    )
    con.execute(
        "INSERT INTO timeline_conflicts VALUES (?,?,?,?,?,?,?)",
        ("foton", "conflict:1", "shared_family_phone", "high", "open", "2026-07-05", '{"entity_refs":["customer:1"]}'),
    )
    con.commit()
    return con


def test_acceptance_workbook_has_five_raw_review_sheets(tmp_path: Path, monkeypatch) -> None:
    con = _review_db(tmp_path / "review.sqlite")
    monkeypatch.setattr(MODULE, "_family_scope_customer_ids", lambda *_args, **_kwargs: ("customer:1",))
    monkeypatch.setattr(
        MODULE,
        "build_customer_dossier",
        lambda *_args, **_kwargs: SimpleNamespace(
            display_name="Родитель", phone="+70000000000", email="parent@example.com",
            brand="foton", next_step="Ответить", next_step_source="derived_signals:signal:1",
        ),
    )
    families, chronology, evidence, conflicts = MODULE._acceptance_family_data(
        con, tenant_id="foton", sample=[{"id": "customer:1", "family_id": "family:1"}],
    )
    assert families[0][15:17] == ["2026-07-03", "mail"]
    assert families[0][13:15] == ["2026-07-04", "математика"]
    out = tmp_path / "review.xlsx"
    MODULE._write_acceptance_workbook(
        out,
        {
            "Семьи 30": (("family_id", "Родитель"), [[families[0][1], families[0][3]]]),
            "Хронология": (("family_id", "event_id", "Полный текст"), [[row[0], row[2], row[9]] for row in chronology]),
            "Доказательства": (("family_id", "Тип", "source_system", "event_id"), [[row[0], row[1], row[5], row[6]] for row in evidence]),
            "Конфликты": (("family_id", "conflict_id"), [[row[0], row[1]] for row in conflicts]),
            "Owner50": (("family_id", "Статус"), [["family:1", "READY"]]),
        },
    )
    wb = load_workbook(out, read_only=True)
    assert wb.sheetnames == list(MODULE._ACCEPTANCE_SHEETS)
    chronology_values = [value for row in wb["Хронология"].iter_rows(values_only=True) for value in row]
    assert "event:mail" in chronology_values
    assert "Полный исходный текст письма" in chronology_values
    assert "event:other" not in chronology_values
    assert "tallanto_scheduled_lesson" in [row[4] for row in chronology]
    assert "Запланированное занятие" in [row[1] for row in evidence]
    assert "customer_identities" in [value for row in wb["Доказательства"].iter_rows(values_only=True) for value in row]
    assert "conflict:1" in [value for row in wb["Конфликты"].iter_rows(values_only=True) for value in row]
    assert out.stat().st_mode & 0o777 == 0o600
    assert len(MODULE._ACCEPTANCE_BUSINESS_REVIEW_COLUMNS) == 5


def test_acceptance_owner50_keeps_candidate_and_excluded_families() -> None:
    control = [
        ("family:1", "candidate", "brand_unproven", "Бренд не подтвержден", *("",) * (len(MODULE.OWNER50_CONTROL_COLUMNS) - 4)),
        ("family:1", "candidate", "product_missing", "Нет продукта", *("",) * (len(MODULE.OWNER50_CONTROL_COLUMNS) - 4)),
        ("family:2", "excluded", "opt_out", "Просили не писать", *("",) * (len(MODULE.OWNER50_CONTROL_COLUMNS) - 4)),
    ]

    rows = MODULE._acceptance_owner50_rows([], control, {"family:1", "family:2"})

    assert len(rows) == 2
    assert rows[0][:4] == [
        "family:1", "CANDIDATE", "brand_unproven; product_missing",
        "Бренд не подтвержден; Нет продукта",
    ]
    assert rows[1][:4] == ["family:2", "EXCLUDED", "opt_out", "Просили не писать"]


def test_acceptance_owner50_blocks_when_sample_family_is_not_classified() -> None:
    with pytest.raises(RuntimeError, match="family:missing"):
        MODULE._acceptance_owner50_rows([], [], {"family:missing"})


def test_acceptance_blocks_when_owner50_classification_fails(tmp_path: Path, monkeypatch) -> None:
    con = _review_db(tmp_path / "review.sqlite")
    monkeypatch.setattr(MODULE, "_connect_ro", lambda _db: con)
    monkeypatch.setattr(MODULE, "_source_freshness", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(MODULE, "manager_freshness_gate", lambda _rows: {"passed": True, "blockers": []})
    monkeypatch.setattr(MODULE, "_dossier_population", lambda *_args, **_kwargs: [{
        "id": "customer:1", "family_id": "family:1", "brand": "foton", "channel": "email",
        "child_bucket": "1", "has_payment": True, "has_conflict": False, "has_signal": True,
        "has_mail": True, "has_call": False, "has_attendance": True,
    }])
    monkeypatch.setattr(MODULE, "_acceptance_family_data", lambda *_args, **_kwargs: ([], [], [], []))
    monkeypatch.setattr(MODULE, "_owner50_family_rows", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("broken")))
    args = SimpleNamespace(db=tmp_path / "review.sqlite", out_root=tmp_path / "out", tenant_id="foton", seed=1, count=1)
    legacy = args.out_root / ".codex_local" / "acceptance_30_families.xlsx"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("old", encoding="utf-8")

    rc = MODULE.cmd_acceptance(args)

    manifest = json.loads((args.out_root / "acceptance_selection_manifest.json").read_text(encoding="utf-8"))
    assert rc == 4
    assert manifest["status"] == "semantic_review_blocked_by_owner50"
    assert manifest["current_artifact"] is None
    assert manifest["legacy_artifact_present"] is True
    assert legacy.read_text(encoding="utf-8") == "old"
    assert not list(legacy.parent.glob("acceptance_30_families_*.xlsx"))


def test_cli_rejects_freshness_bypass(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", [str(SCRIPT), "acceptance", "--db", "x", "--out-root", "y", "--skip-freshness-gate"])

    with pytest.raises(SystemExit) as exc:
        MODULE.main()

    assert exc.value.code == 2


def test_acceptance_blocks_when_sample_loses_attendance_layer(tmp_path: Path, monkeypatch) -> None:
    con = _review_db(tmp_path / "review.sqlite")
    base = {
        "brand": "foton", "channel": "email", "child_bucket": "1", "has_payment": True,
        "has_conflict": False, "has_signal": True, "has_mail": True, "has_call": False,
    }
    population = [
        {**base, "id": "customer:1", "family_id": "family:1", "has_attendance": False},
        {**base, "id": "customer:2", "family_id": "family:2", "has_attendance": True},
    ]
    monkeypatch.setattr(MODULE, "_connect_ro", lambda _db: con)
    monkeypatch.setattr(MODULE, "_source_freshness", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(MODULE, "manager_freshness_gate", lambda _rows: {"passed": True, "blockers": []})
    monkeypatch.setattr(MODULE, "_dossier_population", lambda *_args, **_kwargs: population)
    monkeypatch.setattr(MODULE, "stratified_sample", lambda *_args, **_kwargs: population[:1])
    args = SimpleNamespace(db=tmp_path / "review.sqlite", out_root=tmp_path / "out", tenant_id="foton", seed=1, count=1)

    rc = MODULE.cmd_acceptance(args)

    manifest = json.loads((args.out_root / "acceptance_selection_manifest.json").read_text(encoding="utf-8"))
    assert rc == 4
    assert manifest["status"] == "semantic_review_blocked_by_sample_coverage"
    assert manifest["missing_layers"] == ["has_attendance"]
    assert not (args.out_root / ".codex_local" / "acceptance_30_families.xlsx").exists()


def test_population_is_unique_by_family_and_conflicts_match_exact_refs(tmp_path: Path) -> None:
    con = _review_db(tmp_path / "population.sqlite")
    con.execute(
        "INSERT INTO family_links_v1 VALUES (?,?,?,?,?,?,?,?,?)",
        ("foton", "family:1", "customer:1b", "child:2", "Второй ребёнок", '["6"]', '["физика"]', "confident", "foton"),
    )
    con.executemany(
        "INSERT INTO family_members_v1 VALUES (?,?,?,?)",
        [
            ("foton", "family:1", "customer:1", "confident"),
            ("foton", "family:1", "customer:1b", "confident"),
        ],
    )
    con.execute("DELETE FROM timeline_conflicts")
    con.execute("DELETE FROM customer_purchases_v1")
    con.execute("DELETE FROM timeline_events")
    con.execute(
        "INSERT INTO customer_purchases_v1 VALUES (?,?,?,?,?,?,?)",
        ("foton", "customer:1b", "all_time", "fact", 1000, "2026-07-06", 1),
    )
    con.execute(
        "INSERT INTO timeline_events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("foton", "customer:1b", "event:call", "2026-07-06", "mango_call", "mango", "inbound", "Звонок", "", "", "call:1", None, '{}'),
    )
    con.execute(
        "INSERT INTO timeline_events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            "foton", "customer:1", "event:student", "2026-07-06", "tallanto_student_snapshot",
            "tallanto_snapshot", "system", "Ученик", "", "", "tallanto:student:1", None,
            '{"record":{"payload":{"student_type":"8_klass"}}}',
        ),
    )
    con.execute(
        "INSERT INTO timeline_conflicts VALUES (?,?,?,?,?,?,?)",
        ("foton", "conflict:10", "shared_family_phone", "high", "open", "2026-07-05", '{"entity_refs":["customer:10"]}'),
    )
    con.commit()
    population = MODULE._dossier_population(con, tenant_id="foton")
    assert len(population) == 1
    assert population[0]["family_id"] == "family:1"
    assert population[0]["has_payment"] is True
    assert population[0]["has_call"] is True
    assert population[0]["has_conflict"] is False

    con.execute(
        "INSERT INTO timeline_conflicts VALUES (?,?,?,?,?,?,?)",
        ("foton", "conflict:family", "family_identity_conflict", "high", "open", "2026-07-06", '{"entity_refs":["family:1"]}'),
    )
    con.commit()
    assert MODULE._dossier_population(con, tenant_id="foton")[0]["has_conflict"] is True


def test_population_includes_canonical_family_without_child_link(tmp_path: Path) -> None:
    con = _review_db(tmp_path / "population-without-child.sqlite")
    con.execute("DELETE FROM family_links_v1")
    con.execute(
        "INSERT INTO family_members_v1 VALUES (?,?,?,?)",
        ("foton", "family:without-child", "customer:1", "confident"),
    )
    con.commit()

    population = MODULE._dossier_population(con, tenant_id="foton")

    assert len(population) == 1
    assert population[0]["family_id"] == "family:without-child"
    assert population[0]["child_bucket"] == "0"


def test_population_does_not_count_future_or_unconfirmed_api_attendance(tmp_path: Path) -> None:
    con = _review_db(tmp_path / "future-attendance.sqlite")
    con.execute("DELETE FROM timeline_events WHERE event_id='event:visit'")
    con.execute(
        "UPDATE timeline_events SET source_system='tallanto_attendance_api', "
        "record_json=? WHERE event_id='event:future-visit'",
        ('{"record":{"attendance_confirmed":false}}',),
    )
    con.commit()

    assert MODULE._dossier_population(con, tenant_id="foton")[0]["has_attendance"] is False


@pytest.mark.parametrize(
    "record_json",
    (
        '{"record":{"attendance_confirmed":false,"physical_absence_confirmed":false}}',
        '{"record":{"attendance_confirmed":false,"physical_absence_confirmed":true}}',
    ),
)
def test_acceptance_does_not_report_unconfirmed_or_absent_lesson_as_visit(
    tmp_path: Path, monkeypatch, record_json: str
) -> None:
    con = _review_db(tmp_path / "not-a-visit.sqlite")
    monkeypatch.setattr(MODULE, "_family_scope_customer_ids", lambda *_args, **_kwargs: ("customer:1",))
    monkeypatch.setattr(
        MODULE,
        "build_customer_dossier",
        lambda *_args, **_kwargs: SimpleNamespace(
            display_name="Контакт", phone="", email="", brand="foton",
            next_step="", next_step_source="",
        ),
    )
    con.execute(
        "UPDATE timeline_events SET source_system='tallanto_attendance_api', record_json=? "
        "WHERE event_id='event:visit'",
        (record_json,),
    )
    con.commit()

    families, *_ = MODULE._acceptance_family_data(
        con, tenant_id="foton", sample=[{"id": "customer:1", "family_id": "family:1"}]
    )

    assert families[0][13:15] == ["", ""]


@pytest.mark.parametrize(
    ("student_type", "expected"),
    (("Listener", False), ("1_klass", True), ("10_klass", True), ("11_klass", False), ("vypusknik", False)),
)
def test_business_population_is_anchored_in_current_tallanto_students(
    tmp_path: Path,
    student_type: str,
    expected: bool,
) -> None:
    con = _review_db(tmp_path / "current-cohort.sqlite")
    con.execute(
        "UPDATE timeline_events SET record_json=? WHERE event_id='event:student'",
        (json.dumps({"record": {"payload": {"student_type": student_type}}}),),
    )
    con.commit()

    population = MODULE._dossier_population(con, tenant_id="foton")

    assert bool(population) is expected

def test_human_review_active_input_requires_exact_eight_unique_hashes(tmp_path: Path) -> None:
    rows = [
        {"customer_sha256": f"{index:064x}", "position": index, "reason_code": "no_explicit_next_step"}
        for index in range(8)
    ]
    path = tmp_path / "exam.json"
    path.write_text(json.dumps({"cards": {"active_deal_closed_or_empty": rows}}), encoding="utf-8")

    assert len(MODULE._active_no_step_refs(path, expected_count=8)) == 8

    rows[-1]["customer_sha256"] = rows[0]["customer_sha256"]
    path.write_text(json.dumps({"cards": {"active_deal_closed_or_empty": rows}}), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        MODULE._active_no_step_refs(path, expected_count=8)


def _ambiguous_case(index: int) -> dict[str, object]:
    return {
        "customer_sha256": f"{index + 1:064x}",
        "case_event_sha256": f"{1000 + index:064x}",
        "reason_codes": ["multiple_amo_contacts"],
        "candidate_amo_contact_sha256s": [f"{2000 + index:064x}"],
        "candidate_amo_lead_sha256s": [],
        "resolution_status": "unresolved_no_authoritative_lead",
        "resolved_amo_lead_sha256": None,
    }


def test_human_review_ambiguous_input_requires_primary_exact_nineteen(tmp_path: Path) -> None:
    path = tmp_path / "ambiguous.json"
    rows = [_ambiguous_case(index) for index in range(19)]
    rows[1]["candidate_amo_contact_sha256s"] = [f"{3001:064x}", f"{3002:064x}"]
    rows[1]["candidate_amo_lead_sha256s"] = [f"{4001:064x}", f"{4002:064x}"]
    path.write_text(json.dumps({"schema_version": MODULE._AMBIGUOUS_INPUT_SCHEMA, "rows": rows}), encoding="utf-8")

    refs = MODULE._ambiguous_link_refs(path, expected_count=19)
    assert len(refs) == 19
    assert "AMO leads=0" in refs[0]["position"]
    assert "AMO contacts=2" in refs[1]["position"]
    assert "AMO leads=2" in refs[1]["position"]

    path.write_text(json.dumps({"schema_version": MODULE._AMBIGUOUS_INPUT_SCHEMA, "rows": rows[:-1]}), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly 19"):
        MODULE._ambiguous_link_refs(path, expected_count=19)


def test_human_review_ambiguous_input_rejects_duplicate_case(tmp_path: Path) -> None:
    path = tmp_path / "ambiguous.json"
    rows = [_ambiguous_case(index) for index in range(19)]
    rows[-1]["case_event_sha256"] = rows[0]["case_event_sha256"]
    path.write_text(json.dumps({"schema_version": MODULE._AMBIGUOUS_INPUT_SCHEMA, "rows": rows}), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate cases"):
        MODULE._ambiguous_link_refs(path, expected_count=19)


def test_human_review_ambiguous_input_rejects_duplicate_customer_and_non_string_hash(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ambiguous.json"
    rows = [_ambiguous_case(index) for index in range(19)]
    rows[-1]["customer_sha256"] = rows[0]["customer_sha256"]
    path.write_text(json.dumps({"schema_version": MODULE._AMBIGUOUS_INPUT_SCHEMA, "rows": rows}), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate customers"):
        MODULE._ambiguous_link_refs(path, expected_count=19)

    rows = [_ambiguous_case(index) for index in range(19)]
    rows[0]["customer_sha256"] = 123
    path.write_text(json.dumps({"schema_version": MODULE._AMBIGUOUS_INPUT_SCHEMA, "rows": rows}), encoding="utf-8")
    with pytest.raises(ValueError, match="must be a SHA256 string"):
        MODULE._ambiguous_link_refs(path, expected_count=19)


def test_human_review_ambiguous_input_rejects_raw_amo_ids_and_invalid_candidate_hashes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ambiguous.json"
    rows = [_ambiguous_case(index) for index in range(19)]
    rows[0]["amo_lead_id"] = "raw-lead-1"
    path.write_text(json.dumps({"schema_version": MODULE._AMBIGUOUS_INPUT_SCHEMA, "rows": rows}), encoding="utf-8")
    with pytest.raises(ValueError, match="only hashed AMO candidates"):
        MODULE._ambiguous_link_refs(path, expected_count=19)

    del rows[0]["amo_lead_id"]
    rows[0]["candidate_amo_contact_sha256s"] = ["not-a-sha256"]
    path.write_text(json.dumps({"schema_version": MODULE._AMBIGUOUS_INPUT_SCHEMA, "rows": rows}), encoding="utf-8")
    with pytest.raises(ValueError, match="64-character SHA256"):
        MODULE._ambiguous_link_refs(path, expected_count=19)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("reason_codes", ["multiple_amo_contacts:contact_id=123"], "allowed codes"),
        ("resolution_status", "unresolved:lead_id=456", "allowed code"),
        ("case_ref", "amo:lead:456", "only hashed AMO candidates"),
        ("customer_id", "customer:raw", "only hashed AMO candidates"),
    ),
)
def test_human_review_ambiguous_input_rejects_freeform_identity_fields(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    path = tmp_path / "ambiguous.json"
    rows = [_ambiguous_case(index) for index in range(19)]
    rows[0][field] = value
    path.write_text(json.dumps({"schema_version": MODULE._AMBIGUOUS_INPUT_SCHEMA, "rows": rows}), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        MODULE._ambiguous_link_refs(path, expected_count=19)


def test_human_review_ambiguous_input_rejects_unknown_top_level_field(tmp_path: Path) -> None:
    path = tmp_path / "ambiguous.json"
    rows = [_ambiguous_case(index) for index in range(19)]
    path.write_text(json.dumps({
        "schema_version": MODULE._AMBIGUOUS_INPUT_SCHEMA,
        "rows": rows,
        "raw_note": "lead_id=456",
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected top-level fields"):
        MODULE._ambiguous_link_refs(path, expected_count=19)


def test_human_review_ambiguous_input_rejects_duplicate_candidate_hash(tmp_path: Path) -> None:
    path = tmp_path / "ambiguous.json"
    rows = [_ambiguous_case(index) for index in range(19)]
    candidate = f"{6001:064x}"
    rows[0]["candidate_amo_lead_sha256s"] = [candidate, candidate]
    path.write_text(json.dumps({"schema_version": MODULE._AMBIGUOUS_INPUT_SCHEMA, "rows": rows}), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate SHA256"):
        MODULE._ambiguous_link_refs(path, expected_count=19)


def test_human_review_ambiguous_input_rejects_unresolved_singleton_lead(tmp_path: Path) -> None:
    path = tmp_path / "ambiguous.json"
    rows = [_ambiguous_case(index) for index in range(19)]
    rows[0]["candidate_amo_lead_sha256s"] = [f"{7001:064x}"]
    path.write_text(json.dumps({"schema_version": MODULE._AMBIGUOUS_INPUT_SCHEMA, "rows": rows}), encoding="utf-8")

    with pytest.raises(ValueError, match="zero or multiple"):
        MODULE._ambiguous_link_refs(path, expected_count=19)

    rows[0]["candidate_amo_lead_sha256s"] = []
    rows[0]["resolution_status"] = "ambiguous_multiple_authoritative_leads"
    path.write_text(json.dumps({"schema_version": MODULE._AMBIGUOUS_INPUT_SCHEMA, "rows": rows}), encoding="utf-8")
    with pytest.raises(ValueError, match="requires multiple"):
        MODULE._ambiguous_link_refs(path, expected_count=19)


def test_human_review_ambiguous_input_allows_resolved_lead_only_for_authoritative_singleton(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ambiguous.json"
    rows = [_ambiguous_case(index) for index in range(19)]
    resolved = f"{5001:064x}"
    rows[0].update({
        "candidate_amo_lead_sha256s": [resolved],
        "resolution_status": "resolved_authoritative_singleton",
        "resolved_amo_lead_sha256": resolved,
    })
    path.write_text(json.dumps({"schema_version": MODULE._AMBIGUOUS_INPUT_SCHEMA, "rows": rows}), encoding="utf-8")
    assert len(MODULE._ambiguous_link_refs(path, expected_count=19)) == 19

    rows[0]["candidate_amo_lead_sha256s"] = [resolved, f"{5002:064x}"]
    path.write_text(json.dumps({"schema_version": MODULE._AMBIGUOUS_INPUT_SCHEMA, "rows": rows}), encoding="utf-8")
    with pytest.raises(ValueError, match="authoritative singleton"):
        MODULE._ambiguous_link_refs(path, expected_count=19)


def test_human_review_workbook_is_one_private_owner_sheet(tmp_path: Path) -> None:
    out = tmp_path / "human.xlsx"
    MODULE._write_human_review_workbook(
        out,
        [{
            "Когорта": "8 active/no-step",
            "customer_id": "customer:1",
            "Клиент": "Клиент с ПД",
            "История верна и полна?": "",
            "Досье экономит время?": "",
            "Действие верно сейчас?": "",
        }],
    )

    wb = load_workbook(out, read_only=True)
    assert wb.sheetnames == ["Human review"]
    headers = [cell.value for cell in next(wb["Human review"].iter_rows())]
    assert "Владелец/семья верны?" in headers
    assert "История верна и полна?" in headers
    assert "Досье экономит время?" in headers
    assert "Действие верно сейчас?" in headers
    assert out.stat().st_mode & 0o777 == 0o600


def test_human_review_student_classes_use_current_family_scope_contract(tmp_path: Path) -> None:
    con = _review_db(tmp_path / "human-review-current-contract.sqlite")

    finished, next_grade, graduate = MODULE._student_classes(
        con,
        tenant_id="foton",
        customer_id="customer:1",
        as_of=datetime(2026, 8, 29, tzinfo=timezone.utc),
    )

    assert finished == "8"
    assert next_grade == "9"
    assert graduate == "нет"
