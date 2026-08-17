from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from datetime import datetime
from pathlib import Path

from mango_mvp.models import CallRecord
from mango_mvp.services import dialogue_contract as contract
from mango_mvp.services.export_excel import (
    CALLS_HEADERS,
    CONTACTS_HEADERS,
    build_call_rows,
    build_contact_rows,
    call_to_row,
    write_workbook,
)
from tests import mango_provider_fixture as fx
from tests.test_ai_office_export import valid_v3_analysis


# Excel now reads every stored analysis through the shared fail-closed role
# guard, so a call whose sides Mango never proved keeps no sales content.  These
# fixtures are the proven case; the untrusted case has its own tests below.
PROVEN_LINES = fx.dialogue_lines()


def proven_variants_json(source_call_id: str, turns=fx.DEFAULT_TURNS) -> str:
    variants = fx.proven_variants(turns)
    variants[contract.PROVIDER_EVIDENCE_FIELD] = fx.evidence(
        turns, source_call_id=source_call_id
    )
    return json.dumps(variants, ensure_ascii=False)


class ExportExcelTest(unittest.TestCase):
    def test_build_contact_rows_aggregates_by_phone(self) -> None:
        turns = (
            ("client", "right", "Нас интересует математика для 9 класса."),
            ("operator", "left", "Хорошо, отправлю программу."),
        )
        analysis_1 = {
            "analysis_schema_version": "v2",
            "history_summary": "Первый звонок.",
            "structured_fields": {
                "people": {"parent_fio": "Иванова Мария", "child_fio": "Иванов Кирилл"},
                "contacts": {
                    "email": "maria@example.com",
                    "phone_from_filename": "79990000000",
                    "preferred_channel": "telegram",
                },
                "student": {"grade_current": "8", "school": None},
                "interests": {
                    "products": ["годовые курсы"],
                    "format": ["онлайн"],
                    "subjects": ["математика"],
                    "exam_targets": [],
                },
                "commercial": {"price_sensitivity": "medium", "budget": None, "discount_interest": True},
                "objections": ["цена"],
                "next_step": {"action": "Отправить материалы", "due": None},
                "lead_priority": "warm",
            },
            "follow_up_score": 70,
            "follow_up_reason": "Есть согласованный следующий шаг.",
            "tags": [],
            "quality_flags": {
                "mode": "stereo",
                "secondary_provider": "gigaam",
                "call_type": "sales_call",
                "needs_review": False,
                "review_reasons": [],
            },
        }
        analysis_2 = {
            "analysis_schema_version": "v2",
            "history_summary": "Второй звонок.",
            "structured_fields": {
                "people": {"parent_fio": "Иванова Мария", "child_fio": "Иванов Кирилл"},
                "contacts": {
                    "email": "maria@example.com",
                    "phone_from_filename": "79990000000",
                    "preferred_channel": "telegram",
                },
                "student": {"grade_current": "8", "school": None},
                "interests": {
                    "products": ["летний лагерь"],
                    "format": ["оффлайн"],
                    "subjects": ["математика", "физика"],
                    "exam_targets": [],
                },
                "commercial": {"price_sensitivity": "low", "budget": None, "discount_interest": False},
                "objections": [],
                "next_step": {"action": "Перезвонить клиенту", "due": "2026-03-25"},
                "lead_priority": "hot",
            },
            "target_product": "летний лагерь",
            "follow_up_score": 90,
            "follow_up_reason": "Клиент заинтересован, нужно быстро вернуться.",
            "tags": [],
            "quality_flags": {
                "mode": "stereo",
                "secondary_provider": "gigaam",
                "call_type": "sales_call",
                "needs_review": True,
                "review_reasons": ["sales_missing_next_step"],
            },
        }

        calls = [
            CallRecord(
                id=1,
                source_call_id="call-1",
                source_recording_id=fx.RECORDING_ID,
                source_file="/tmp/a.mp3",
                source_filename="a.mp3",
                phone="79990000000",
                manager_name="Менеджер 1",
                duration_sec=120.0,
                started_at=datetime(2026, 3, 20, 10, 0, 0),
                transcript_variants_json=proven_variants_json("call-1", turns),
                analysis_json=json.dumps(analysis_1, ensure_ascii=False),
            ),
            CallRecord(
                id=2,
                source_call_id="call-2",
                source_recording_id=fx.RECORDING_ID,
                source_file="/tmp/b.mp3",
                source_filename="b.mp3",
                phone="79990000000",
                manager_name="Менеджер 2",
                duration_sec=180.0,
                started_at=datetime(2026, 3, 21, 11, 0, 0),
                transcript_variants_json=proven_variants_json("call-2", turns),
                analysis_json=json.dumps(analysis_2, ensure_ascii=False),
            ),
        ]
        # The exporter no longer trusts legacy v2 facts.  Use the current
        # evidenced contract so this test keeps exercising aggregation rather
        # than accidentally testing the fail-closed migration path.
        for call in calls:
            call.analysis_json = json.dumps(valid_v3_analysis(call), ensure_ascii=False)

        call_rows = build_call_rows(calls)
        self.assertEqual(len(call_rows), 2)
        self.assertEqual(call_rows[0]["recommended_followup_date"], "")
        self.assertEqual(call_rows[1]["recommended_followup_date"], "")
        self.assertIn(
            "не согласован",
            call_rows[0]["recommended_followup_reason"],
        )
        self.assertEqual(call_rows[0]["needs_review"], False)

        contact_rows = build_contact_rows(call_rows)
        self.assertEqual(len(contact_rows), 1)
        row = contact_rows[0]
        self.assertEqual(row["phone"], "79990000000")
        self.assertEqual(row["calls_count"], 2)
        self.assertEqual(row["latest_manager_name"], "Менеджер 2")
        self.assertEqual(row["recommended_product"], "")
        self.assertEqual(row["lead_priority"], "warm")
        self.assertEqual(row["needs_review"], False)
        self.assertIn("математика", row["interests_subjects"])

    def test_write_workbook_creates_valid_xlsx(self) -> None:
        calls_rows = [
            {
                "id": 1,
                "started_at": "2026-03-21 11:00:00",
                "phone": "79990000000",
                "manager_name": "Менеджер",
                "duration_sec": 180.0,
                "source_filename": "call.mp3",
                "source_file": "/tmp/call.mp3",
                "history_summary": "Тестовый конспект.",
                "parent_fio": "Иванова Мария",
                "child_fio": "Иванов Кирилл",
                "email": "maria@example.com",
                "preferred_channel": "telegram",
                "grade_current": "8",
                "school": "",
                "interests_products": "годовые курсы",
                "interests_format": "онлайн",
                "interests_subjects": "математика",
                "exam_targets": "",
                "recommended_product": "годовые курсы",
                "price_sensitivity": "medium",
                "budget": "",
                "discount_interest": "True",
                "objections": "цена",
                "next_step_action": "Отправить материалы",
                "next_step_due_raw": "",
                "lead_priority": "warm",
                "sale_probability_pct": 70,
                "sale_probability_reason": "Есть согласованный следующий шаг.",
                "recommended_followup_date": "2026-03-23",
                "recommended_followup_reason": "После отправки материалов оптимален follow-up через 2 дня.",
                "call_type": "sales_call",
                "needs_review": False,
                "review_reasons": "",
                "quality_mode": "stereo",
                "secondary_provider": "gigaam",
                "secondary_backfill_status": "",
                "tags": "",
                "analysis_schema_version": "v2",
            }
        ]
        contacts_rows = [
            {
                "contact_key": "79990000000",
                "phone": "79990000000",
                "calls_count": 1,
                "first_call_at": "2026-03-21 11:00:00",
                "last_call_at": "2026-03-21 11:00:00",
                "latest_manager_name": "Менеджер",
                "latest_history_summary": "Тестовый конспект.",
                "parent_fio": "Иванова Мария",
                "child_fio": "Иванов Кирилл",
                "email": "maria@example.com",
                "preferred_channel": "telegram",
                "grade_current": "8",
                "interests_products": "годовые курсы",
                "interests_format": "онлайн",
                "interests_subjects": "математика",
                "exam_targets": "",
                "recommended_product": "годовые курсы",
                "lead_priority": "warm",
                "sale_probability_pct": 70,
                "sale_probability_reason": "Есть согласованный следующий шаг.",
                "recommended_followup_date": "2026-03-23",
                "recommended_followup_reason": "После отправки материалов оптимален follow-up через 2 дня.",
                "latest_call_type": "sales_call",
                "needs_review": False,
                "review_reasons_latest": "",
                "last_next_step_action": "Отправить материалы",
                "last_next_step_due_raw": "",
                "objections_latest": "цена",
                "source_call_ids": "1",
            }
        ]

        with tempfile.TemporaryDirectory(prefix="mango_xlsx_") as td:
            out_path = write_workbook(
                Path(td) / "sales_workbook.xlsx",
                calls_rows=calls_rows,
                contacts_rows=contacts_rows,
            )
            self.assertTrue(out_path.exists())
            with zipfile.ZipFile(out_path, "r") as zf:
                names = set(zf.namelist())
                self.assertIn("xl/workbook.xml", names)
                self.assertIn("xl/worksheets/sheet1.xml", names)
                self.assertIn("xl/worksheets/sheet2.xml", names)
                sheet1 = zf.read("xl/worksheets/sheet1.xml").decode("utf-8")
                sheet2 = zf.read("xl/worksheets/sheet2.xml").decode("utf-8")
                workbook_text = sheet1 + "\n" + sheet2
                if "xl/sharedStrings.xml" in names:
                    shared_strings = zf.read("xl/sharedStrings.xml").decode("utf-8")
                    workbook_text += "\n" + shared_strings
                self.assertIn("Тестовый конспект.", workbook_text)
                self.assertIn("79990000000", workbook_text)
                self.assertIn("<worksheet", sheet1)
                self.assertIn("<worksheet", sheet2)

    def test_build_call_rows_repairs_mojibake_manager_and_filename_for_export(self) -> None:
        analysis = {
            "analysis_schema_version": "v2",
            "history_summary": "26.03.2026 менеджер КЂлз•Ґ† Д†амп обсудил с клиентом программу.",
            "structured_fields": {
                "people": {},
                "contacts": {},
                "student": {},
                "interests": {"products": [], "format": [], "subjects": [], "exam_targets": []},
                "commercial": {},
                "objections": [],
                "next_step": {"action": None, "due": None},
                "lead_priority": "warm",
            },
            "quality_flags": {"call_type": "sales_call"},
            "tags": [],
        }
        calls = [
            CallRecord(
                id=1,
                source_call_id="call-1",
                source_recording_id=fx.RECORDING_ID,
                source_file="/tmp/raw.mp3",
                source_filename="2026-03-09__10-52-01__КЂлз•Ґ† Д†амп__79801983922.mp3",
                phone="79801983922",
                manager_name="КЂлз•Ґ† Д†амп",
                duration_sec=60.0,
                started_at=datetime(2026, 3, 9, 10, 52, 1),
                transcript_variants_json=proven_variants_json("call-1"),
                analysis_json=json.dumps(analysis, ensure_ascii=False),
            ),
        ]
        row = build_call_rows(calls)[0]
        self.assertEqual(row["manager_name"], "Клычева Дарья")
        self.assertEqual(row["history_summary"], contract.INVALID_STORED_SUMMARY)
        self.assertEqual(row["recommended_followup_date"], "")
        self.assertIn("повторный анализ", row["recommended_followup_reason"])
        self.assertNotIn("КЂлз", row["history_summary"])
        self.assertEqual(
            row["source_filename"],
            "2026-03-09__10-52-01__Клычева Дарья__79801983922.mp3",
        )

    def _untrusted_call(self, analysis):
        return CallRecord(
            id=1,
            source_call_id="call-1",
            source_file="/tmp/a.mp3",
            source_filename="a.mp3",
            phone="79990000000",
            manager_name="Менеджер 1",
            duration_sec=120.0,
            started_at=datetime(2026, 3, 20, 10, 0, 0),
            # Real production shape: channel roles guessed from the text alone.
            transcript_variants_json=json.dumps(
                {
                    "mode": "stereo",
                    "role_mapping": {
                        "status": "confirmed_multi_signal",
                        "confirmed": True,
                        "manager_quality_allowed": True,
                        "topology": "simple_two_party",
                        "left": "manager",
                        "right": "client",
                    },
                    "dialogue_lines": PROVEN_LINES,
                },
                ensure_ascii=False,
            ),
            analysis_json=json.dumps(analysis, ensure_ascii=False),
        )

    def test_an_unproven_call_never_gets_a_recommended_followup_in_excel(self) -> None:
        """A follow-up date reads as an agreement with a specific side."""
        analysis = {
            "analysis_schema_version": "v2",
            "history_summary": "Клиент Мария согласилась оплатить курс.",
            "structured_fields": {
                "people": {"parent_fio": "Иванова Мария", "child_fio": "Иванов Пётр"},
                "contacts": {"email": "mama@example.com"},
                "next_step": {"action": "Отправить материалы", "due": None},
                "lead_priority": "hot",
            },
            "target_product": "летний лагерь",
            "follow_up_score": 90,
            "tags": [],
            "quality_flags": {"call_type": "sales_call", "needs_review": False},
        }

        row = build_call_rows([self._untrusted_call(analysis)])[0]

        self.assertEqual(row["recommended_followup_date"], "")
        self.assertEqual(
            row["recommended_followup_reason"], contract.UNTRUSTED_FOLLOW_UP_REASON
        )
        self.assertEqual(row["next_step_action"], "")
        self.assertEqual(row["next_step_due_raw"], "")
        self.assertEqual(row["parent_fio"], "")
        self.assertEqual(row["child_fio"], "")
        self.assertEqual(row["email"], "")
        self.assertEqual(row["recommended_product"], "")
        self.assertEqual(row["lead_priority"], "")
        self.assertEqual(row["sale_probability_pct"], "")
        self.assertTrue(row["needs_review"])
        self.assertIn("требует ручной проверки", row["history_summary"])
        self.assertNotIn("Мария", row["history_summary"])

    def test_direct_call_to_row_cannot_bypass_the_stored_analysis_guard(self) -> None:
        analysis = {
            "analysis_schema_version": "v2",
            "history_summary": "Клиент уже оплатил.",
            "structured_fields": {
                "people": {}, "contacts": {}, "student": {}, "interests": {},
                "commercial": {}, "objections": [],
                "next_step": {"action": "Отправить договор", "due": "сегодня"},
                "lead_priority": "hot",
            },
            "follow_up_score": 95,
        }

        row = call_to_row(self._untrusted_call(analysis), analysis)

        self.assertEqual(row["next_step_action"], "")
        self.assertEqual(row["sale_probability_pct"], "")
        self.assertTrue(row["needs_review"])

    def test_the_excel_review_reason_of_an_unproven_call_is_russian(self) -> None:
        row = build_call_rows(
            [self._untrusted_call({"analysis_schema_version": "v2", "tags": []})]
        )[0]

        self.assertIn("Mango", row["review_reasons"])
        self.assertNotIn("role_attribution_untrusted", row["review_reasons"])

    def test_a_text_cell_is_never_compiled_into_a_spreadsheet_formula(self) -> None:
        """The XML is inspected, not the Python value that went in.

        A manager name, a summary or a model answer beginning with ``=``, ``+``,
        ``-`` or ``@`` is data.  XlsxWriter compiles such a string into a
        formula by default, and the cell then executes on open — the value the
        reader sees is not the value that was exported.  Asserting on the row
        dict would prove nothing here: the defect only exists in the file.
        """
        dangerous = {
            "manager_name": "=1+1",
            "history_summary": "+7 999 000 00 00 перезвонить",
            "objections": "-цена высокая",
            "next_step_action": "@Иванову отправить договор",
            "review_reasons": "=HYPERLINK(\"http://evil\",\"click\")",
        }
        calls_row = {header: "" for header in CALLS_HEADERS}
        calls_row.update(
            {"id": 1, "started_at": "2026-03-21 11:00:00", "duration_sec": 180.0}
        )
        calls_row.update(dangerous)
        contacts_row = {header: "" for header in CONTACTS_HEADERS}
        contacts_row.update({"contact_key": "79990000000", "calls_count": 1})
        contacts_row["latest_history_summary"] = "=2+2"

        with tempfile.TemporaryDirectory(prefix="mango_xlsx_formula_") as td:
            out_path = write_workbook(
                Path(td) / "wb.xlsx",
                calls_rows=[calls_row],
                contacts_rows=[contacts_row],
            )
            with zipfile.ZipFile(out_path, "r") as zf:
                names = set(zf.namelist())
                sheets = "\n".join(
                    zf.read(name).decode("utf-8")
                    for name in names
                    if name.startswith("xl/worksheets/")
                )
                strings = (
                    zf.read("xl/sharedStrings.xml").decode("utf-8")
                    if "xl/sharedStrings.xml" in names
                    else ""
                )

        # No cell carries a formula element at all.
        self.assertNotIn("<f>", sheets)
        self.assertNotIn("<f ", sheets)
        # And the dangerous text survived as text rather than being evaluated
        # away.  Quotes are left out of the comparison on purpose: their XML
        # escaping differs between the two writers, the leading character does
        # not.
        body = sheets + "\n" + strings
        for marker in ("=1+1", "+7 999 000 00 00", "-цена высокая", "@Иванову", "HYPERLINK"):
            self.assertIn(marker, body)
