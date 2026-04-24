from __future__ import annotations

import json
import math
import zipfile
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from xml.sax.saxutils import escape

from mango_mvp.models import CallRecord
from mango_mvp.utils.filename_repair import (
    repair_filename_display,
    repair_manager_name,
    repair_text_manager_names,
)

try:
    import xlsxwriter
except ImportError:  # pragma: no cover - optional dependency in some environments
    xlsxwriter = None


CALLS_HEADERS = [
    "id",
    "started_at",
    "phone",
    "manager_name",
    "duration_sec",
    "source_filename",
    "source_file",
    "history_summary",
    "parent_fio",
    "child_fio",
    "email",
    "preferred_channel",
    "grade_current",
    "school",
    "interests_products",
    "interests_format",
    "interests_subjects",
    "exam_targets",
    "recommended_product",
    "price_sensitivity",
    "budget",
    "discount_interest",
    "objections",
    "next_step_action",
    "next_step_due_raw",
    "lead_priority",
    "sale_probability_pct",
    "sale_probability_reason",
    "recommended_followup_date",
    "recommended_followup_reason",
    "call_type",
    "needs_review",
    "review_reasons",
    "quality_mode",
    "secondary_provider",
    "secondary_backfill_status",
    "tags",
    "analysis_schema_version",
]

CONTACTS_HEADERS = [
    "contact_key",
    "phone",
    "calls_count",
    "first_call_at",
    "last_call_at",
    "latest_manager_name",
    "latest_history_summary",
    "parent_fio",
    "child_fio",
    "email",
    "preferred_channel",
    "grade_current",
    "interests_products",
    "interests_format",
    "interests_subjects",
    "exam_targets",
    "recommended_product",
    "lead_priority",
    "sale_probability_pct",
    "sale_probability_reason",
    "recommended_followup_date",
    "recommended_followup_reason",
    "latest_call_type",
    "needs_review",
    "review_reasons_latest",
    "last_next_step_action",
    "last_next_step_due_raw",
    "objections_latest",
    "source_call_ids",
]


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _join_unique(values: Iterable[str]) -> str:
    seen: set[str] = set()
    result: list[str] = []
    for item in values:
        text = _clean_text(item)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return " | ".join(result)


def _parse_any_date(value: str) -> Optional[datetime]:
    text = _clean_text(value)
    if not text:
        return None
    for fmt in (
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%d.%m.%Y",
        "%d.%m.%Y %H:%M",
        "%d.%m.%Y %H:%M:%S",
    ):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _format_dt(value: Optional[datetime]) -> str:
    if value is None:
        return ""
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _format_date(value: Optional[datetime]) -> str:
    if value is None:
        return ""
    return value.strftime("%Y-%m-%d")


def _priority_rank(value: str) -> int:
    normalized = _clean_text(value).lower()
    if normalized == "hot":
        return 3
    if normalized == "warm":
        return 2
    if normalized == "cold":
        return 1
    return 0


def _recommend_followup(
    *,
    started_at: Optional[datetime],
    next_step_due_raw: str,
    next_step_action: str,
    lead_priority: str,
    sale_probability_pct: Optional[int],
    tags: str,
) -> tuple[str, str]:
    due_dt = _parse_any_date(next_step_due_raw)
    if due_dt is not None:
        return (_format_date(due_dt), "Явная дата следующего шага из анализа звонка.")

    lowered_tags = tags.lower()
    if "non_conversation" in lowered_tags:
        return ("", "Нет содержательного диалога: follow-up автоматически не ставится.")

    base_dt = started_at
    if base_dt is None:
        return ("", "Нет даты звонка: рекомендованную дату follow-up вычислить нельзя.")

    score = sale_probability_pct if isinstance(sale_probability_pct, int) else None
    action = next_step_action.lower()
    priority = lead_priority.lower()

    days = 3
    reason = "Правило по умолчанию для следующего контакта."
    if "отправ" in action:
        days = 2
        reason = "После отправки материалов оптимален follow-up через 2 дня."
    elif "перезвон" in action or "созвон" in action or "позвон" in action:
        days = 1 if (score or 0) >= 75 else 3
        reason = "Для согласованного повторного звонка ставится быстрый follow-up."
    elif priority == "hot" or (score or 0) >= 75:
        days = 1
        reason = "Горячий лид: следующий контакт на следующий день."
    elif priority == "warm" or (score or 0) >= 45:
        days = 3
        reason = "Теплый лид: follow-up через 3 дня."
    elif priority == "cold" or (score or 0) > 0:
        days = 14
        reason = "Холодный лид: мягкий follow-up через 2 недели."

    return (_format_date(base_dt + timedelta(days=days)), reason)


def _mode_value(analysis: Dict[str, Any]) -> tuple[str, str, str]:
    quality_flags = _as_dict(analysis.get("quality_flags"))
    return (
        _clean_text(quality_flags.get("mode")),
        _clean_text(quality_flags.get("secondary_provider")),
        _clean_text(quality_flags.get("secondary_backfill_status")),
    )


def call_to_row(call: CallRecord, analysis: Dict[str, Any]) -> Dict[str, Any]:
    blocks = _as_dict(analysis.get("structured_fields")) or _as_dict(analysis.get("crm_blocks"))
    people = _as_dict(blocks.get("people"))
    contacts = _as_dict(blocks.get("contacts"))
    student = _as_dict(blocks.get("student"))
    interests = _as_dict(blocks.get("interests"))
    commercial = _as_dict(blocks.get("commercial"))
    next_step = _as_dict(blocks.get("next_step"))

    history_summary = (
        _clean_text(analysis.get("history_summary"))
        or _clean_text(analysis.get("history_short"))
        or _clean_text(analysis.get("summary"))
    )
    history_summary = repair_text_manager_names(history_summary)
    recommended_product = (
        _clean_text(analysis.get("target_product"))
        or (_join_unique(_as_list(interests.get("products"))).split(" | ")[0] if _join_unique(_as_list(interests.get("products"))) else "")
    )
    objections = _join_unique(_as_list(blocks.get("objections")) + _as_list(analysis.get("objections")))
    next_step_action = _clean_text(next_step.get("action")) or _clean_text(analysis.get("next_step"))
    next_step_due_raw = _clean_text(next_step.get("due")) or _clean_text(analysis.get("timeline"))
    lead_priority = _clean_text(blocks.get("lead_priority"))
    sale_probability_pct = analysis.get("follow_up_score")
    if not isinstance(sale_probability_pct, int):
        try:
            sale_probability_pct = int(sale_probability_pct)
        except (TypeError, ValueError):
            sale_probability_pct = None
    tags = _join_unique(_as_list(analysis.get("tags")))
    quality_flags = _as_dict(analysis.get("quality_flags"))
    call_type = _clean_text(quality_flags.get("call_type"))
    needs_review = bool(analysis.get("needs_review") if analysis.get("needs_review") is not None else quality_flags.get("needs_review"))
    review_reasons = _join_unique(
        _as_list(analysis.get("review_reasons")) + _as_list(quality_flags.get("review_reasons"))
    )
    recommended_followup_date, recommended_followup_reason = _recommend_followup(
        started_at=call.started_at,
        next_step_due_raw=next_step_due_raw,
        next_step_action=next_step_action,
        lead_priority=lead_priority,
        sale_probability_pct=sale_probability_pct,
        tags=tags,
    )
    quality_mode, secondary_provider, secondary_backfill_status = _mode_value(analysis)

    return {
        "id": call.id,
        "started_at": _format_dt(call.started_at),
        "phone": _clean_text(call.phone) or _clean_text(contacts.get("phone_from_filename")),
        "manager_name": _clean_text(repair_manager_name(call.manager_name)),
        "duration_sec": round(float(call.duration_sec or 0.0), 3),
        "source_filename": repair_filename_display(_clean_text(call.source_filename)),
        "source_file": _clean_text(call.source_file),
        "history_summary": history_summary,
        "parent_fio": _clean_text(people.get("parent_fio")),
        "child_fio": _clean_text(people.get("child_fio")),
        "email": _clean_text(contacts.get("email")),
        "preferred_channel": _clean_text(contacts.get("preferred_channel")),
        "grade_current": _clean_text(student.get("grade_current")),
        "school": _clean_text(student.get("school")),
        "interests_products": _join_unique(_as_list(interests.get("products"))),
        "interests_format": _join_unique(_as_list(interests.get("format"))),
        "interests_subjects": _join_unique(_as_list(interests.get("subjects"))),
        "exam_targets": _join_unique(_as_list(interests.get("exam_targets"))),
        "recommended_product": recommended_product,
        "price_sensitivity": _clean_text(commercial.get("price_sensitivity")),
        "budget": _clean_text(commercial.get("budget")) or _clean_text(analysis.get("budget")),
        "discount_interest": _clean_text(commercial.get("discount_interest")),
        "objections": objections,
        "next_step_action": next_step_action,
        "next_step_due_raw": next_step_due_raw,
        "lead_priority": lead_priority,
        "sale_probability_pct": sale_probability_pct if sale_probability_pct is not None else "",
        "sale_probability_reason": _clean_text(analysis.get("follow_up_reason")),
        "recommended_followup_date": recommended_followup_date,
        "recommended_followup_reason": recommended_followup_reason,
        "call_type": call_type,
        "needs_review": needs_review,
        "review_reasons": review_reasons,
        "quality_mode": quality_mode,
        "secondary_provider": secondary_provider,
        "secondary_backfill_status": secondary_backfill_status,
        "tags": tags,
        "analysis_schema_version": _clean_text(analysis.get("analysis_schema_version")),
    }


def build_call_rows(calls: Iterable[CallRecord]) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for call in calls:
        raw = (call.analysis_json or "").strip()
        if not raw:
            continue
        try:
            analysis = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(analysis, dict):
            continue
        rows.append(call_to_row(call, analysis))
    return rows


def _choose_latest(rows: list[Dict[str, Any]], key: str) -> str:
    for row in rows:
        value = _clean_text(row.get(key))
        if value:
            return value
    return ""


def _choose_common(rows: list[Dict[str, Any]], key: str) -> str:
    values = [_clean_text(row.get(key)) for row in rows]
    values = [value for value in values if value]
    if not values:
        return ""
    return Counter(values).most_common(1)[0][0]


def build_contact_rows(call_rows: Iterable[Dict[str, Any]]) -> list[Dict[str, Any]]:
    grouped: Dict[str, list[Dict[str, Any]]] = {}
    for row in call_rows:
        phone = _clean_text(row.get("phone"))
        if phone:
            key = phone
        else:
            key = f"call:{_clean_text(row.get('id'))}"
        grouped.setdefault(key, []).append(dict(row))

    out_rows: list[Dict[str, Any]] = []
    for key, rows in grouped.items():
        sorted_rows = sorted(
            rows,
            key=lambda item: (
                _parse_any_date(_clean_text(item.get("started_at"))) or datetime.min,
                int(item.get("id") or 0),
            ),
            reverse=True,
        )
        latest = sorted_rows[0]
        chronological = list(reversed(sorted_rows))
        first_dt = _parse_any_date(_clean_text(chronological[0].get("started_at")))
        last_dt = _parse_any_date(_clean_text(latest.get("started_at")))
        interests_products = _join_unique(row.get("interests_products", "") for row in rows)
        interests_format = _join_unique(row.get("interests_format", "") for row in rows)
        interests_subjects = _join_unique(row.get("interests_subjects", "") for row in rows)
        exam_targets = _join_unique(row.get("exam_targets", "") for row in rows)
        lead_priority = _clean_text(latest.get("lead_priority"))
        if not lead_priority:
            lead_priority = max(
                (_clean_text(row.get("lead_priority")) for row in rows),
                key=_priority_rank,
                default="",
            )
        out_rows.append(
            {
                "contact_key": key,
                "phone": _clean_text(latest.get("phone")),
                "calls_count": len(rows),
                "first_call_at": _format_dt(first_dt),
                "last_call_at": _format_dt(last_dt),
                "latest_manager_name": _clean_text(repair_manager_name(latest.get("manager_name"))),
                "latest_history_summary": repair_text_manager_names(_clean_text(latest.get("history_summary"))),
                "parent_fio": _choose_common(rows, "parent_fio"),
                "child_fio": _choose_common(rows, "child_fio"),
                "email": _choose_latest(sorted_rows, "email"),
                "preferred_channel": _choose_latest(sorted_rows, "preferred_channel"),
                "grade_current": _choose_latest(sorted_rows, "grade_current"),
                "interests_products": interests_products,
                "interests_format": interests_format,
                "interests_subjects": interests_subjects,
                "exam_targets": exam_targets,
                "recommended_product": _choose_latest(sorted_rows, "recommended_product"),
                "lead_priority": lead_priority,
                "sale_probability_pct": latest.get("sale_probability_pct", ""),
                "sale_probability_reason": _clean_text(latest.get("sale_probability_reason")),
                "recommended_followup_date": _clean_text(latest.get("recommended_followup_date")),
                "recommended_followup_reason": _clean_text(latest.get("recommended_followup_reason")),
                "latest_call_type": _clean_text(latest.get("call_type")),
                "needs_review": any(bool(row.get("needs_review")) for row in rows),
                "review_reasons_latest": _clean_text(latest.get("review_reasons")),
                "last_next_step_action": _clean_text(latest.get("next_step_action")),
                "last_next_step_due_raw": _clean_text(latest.get("next_step_due_raw")),
                "objections_latest": _clean_text(latest.get("objections")),
                "source_call_ids": " | ".join(str(int(row.get("id") or 0)) for row in sorted_rows),
            }
        )

    return sorted(
        out_rows,
        key=lambda row: (
            _parse_any_date(_clean_text(row.get("last_call_at"))) or datetime.min,
            _clean_text(row.get("phone")),
        ),
        reverse=True,
    )


def _column_name(index: int) -> str:
    result = ""
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result


def _xml_cell(ref: str, value: Any) -> str:
    if value is None or value == "":
        return f'<c r="{ref}"/>'
    if isinstance(value, bool):
        text = "TRUE" if value else "FALSE"
        return f'<c r="{ref}" t="inlineStr"><is><t>{escape(text)}</t></is></c>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number):
            if isinstance(value, int) or number.is_integer():
                text = str(int(number))
            else:
                text = str(number)
            return f'<c r="{ref}"><v>{text}</v></c>'
    text = escape(_clean_text(value))
    return f'<c r="{ref}" t="inlineStr"><is><t xml:space="preserve">{text}</t></is></c>'


def _sheet_xml(headers: list[str], rows: list[Dict[str, Any]]) -> str:
    row_xml: list[str] = []
    header_cells = [
        _xml_cell(f"{_column_name(col_idx)}1", header)
        for col_idx, header in enumerate(headers, start=1)
    ]
    row_xml.append(f'<row r="1">{"".join(header_cells)}</row>')

    for row_idx, row in enumerate(rows, start=2):
        cells = []
        for col_idx, header in enumerate(headers, start=1):
            ref = f"{_column_name(col_idx)}{row_idx}"
            cells.append(_xml_cell(ref, row.get(header, "")))
        row_xml.append(f'<row r="{row_idx}">{"".join(cells)}</row>')

    max_col = _column_name(max(1, len(headers)))
    max_row = max(1, len(rows) + 1)
    auto_filter_ref = f"A1:{max_col}{max_row}"
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheetViews>'
        '<sheetView workbookViewId="0">'
        '<pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>'
        '</sheetView>'
        '</sheetViews>'
        "<sheetFormatPr defaultRowHeight=\"15\"/>"
        f"<autoFilter ref=\"{auto_filter_ref}\"/>"
        f"<sheetData>{''.join(row_xml)}</sheetData>"
        "</worksheet>"
    )


def _xlsxwriter_widths(headers: list[str], rows: list[Dict[str, Any]]) -> list[int]:
    widths: list[int] = []
    for header in headers:
        max_len = len(header)
        for row in rows:
            text = _clean_text(row.get(header, ""))
            if text:
                max_len = max(max_len, min(len(text), 120))
        widths.append(min(max(max_len + 2, 12), 60))
    return widths


def _write_workbook_xlsxwriter(
    workbook_path: Path,
    *,
    calls_rows: list[Dict[str, Any]],
    contacts_rows: list[Dict[str, Any]],
) -> Path:
    if xlsxwriter is None:  # pragma: no cover - protected by caller
        raise RuntimeError("XlsxWriter is not installed")

    with xlsxwriter.Workbook(workbook_path) as workbook:
        header_fmt = workbook.add_format(
            {
                "bold": True,
                "bg_color": "#D9EAF7",
                "border": 1,
                "text_wrap": True,
                "valign": "top",
            }
        )
        text_fmt = workbook.add_format({"text_wrap": True, "valign": "top"})
        int_fmt = workbook.add_format({"num_format": "0", "valign": "top"})
        float_fmt = workbook.add_format({"num_format": "0.000", "valign": "top"})

        def write_sheet(name: str, headers: list[str], rows: list[Dict[str, Any]]) -> None:
            worksheet = workbook.add_worksheet(name)
            worksheet.freeze_panes(1, 0)
            worksheet.autofilter(0, 0, max(len(rows), 1), len(headers) - 1)
            worksheet.set_row(0, 24)

            for col_idx, header in enumerate(headers):
                worksheet.write(0, col_idx, header, header_fmt)

            for row_idx, row in enumerate(rows, start=1):
                for col_idx, header in enumerate(headers):
                    value = row.get(header, "")
                    if value is None or value == "":
                        worksheet.write_blank(row_idx, col_idx, None, text_fmt)
                    elif isinstance(value, bool):
                        worksheet.write_boolean(row_idx, col_idx, value, text_fmt)
                    elif isinstance(value, int):
                        worksheet.write_number(row_idx, col_idx, value, int_fmt)
                    elif isinstance(value, float):
                        if math.isfinite(value):
                            worksheet.write_number(row_idx, col_idx, value, float_fmt)
                        else:
                            worksheet.write(row_idx, col_idx, str(value), text_fmt)
                    else:
                        worksheet.write(row_idx, col_idx, _clean_text(value), text_fmt)

            for col_idx, width in enumerate(_xlsxwriter_widths(headers, rows)):
                worksheet.set_column(col_idx, col_idx, width)

        write_sheet("Calls", CALLS_HEADERS, calls_rows)
        write_sheet("Contacts", CONTACTS_HEADERS, contacts_rows)

    return workbook_path


def _write_workbook_minimal_xml(
    out_path: Path,
    *,
    calls_rows: list[Dict[str, Any]],
    contacts_rows: list[Dict[str, Any]],
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    workbook_path = out_path if out_path.suffix.lower() == ".xlsx" else out_path.with_suffix(".xlsx")

    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/worksheets/sheet2.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>
"""
    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
"""
    workbook = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
          xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="Calls" sheetId="1" r:id="rId1"/>
    <sheet name="Contacts" sheetId="2" r:id="rId2"/>
  </sheets>
</workbook>
"""
    workbook_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>
"""
    styles = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>
  <fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills>
  <borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>
"""
    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    core = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
 xmlns:dc="http://purl.org/dc/elements/1.1/"
 xmlns:dcterms="http://purl.org/dc/terms/"
 xmlns:dcmitype="http://purl.org/dc/dcmitype/"
 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>Mango Calls Workbook</dc:title>
  <dc:creator>mango-mvp</dc:creator>
  <cp:lastModifiedBy>mango-mvp</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>
"""
    app = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
 xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>mango-mvp</Application>
</Properties>
"""

    with zipfile.ZipFile(workbook_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("xl/workbook.xml", workbook)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        zf.writestr("xl/styles.xml", styles)
        zf.writestr("xl/worksheets/sheet1.xml", _sheet_xml(CALLS_HEADERS, calls_rows))
        zf.writestr("xl/worksheets/sheet2.xml", _sheet_xml(CONTACTS_HEADERS, contacts_rows))
        zf.writestr("docProps/core.xml", core)
        zf.writestr("docProps/app.xml", app)

    return workbook_path


def write_workbook(
    out_path: Path,
    *,
    calls_rows: list[Dict[str, Any]],
    contacts_rows: list[Dict[str, Any]],
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    workbook_path = out_path if out_path.suffix.lower() == ".xlsx" else out_path.with_suffix(".xlsx")

    if xlsxwriter is not None:
        return _write_workbook_xlsxwriter(
            workbook_path,
            calls_rows=calls_rows,
            contacts_rows=contacts_rows,
        )

    return _write_workbook_minimal_xml(
        workbook_path,
        calls_rows=calls_rows,
        contacts_rows=contacts_rows,
    )
