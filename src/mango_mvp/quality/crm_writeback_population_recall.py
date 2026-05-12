from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from mango_mvp.quality.crm_writeback_quality_detector import detect_crm_writeback_quality_risks


SCHEMA_VERSION = "crm_writeback_population_recall_v1"


@dataclass(frozen=True)
class CrmWritebackPopulationMarker:
    marker_id: str
    class_id: str
    precision: str
    pattern: re.Pattern[str]
    description: str
    excludes: tuple[re.Pattern[str], ...] = ()


@dataclass(frozen=True)
class CrmWritebackPopulationHit:
    row_index: int
    phone: str
    marker_id: str
    class_id: str
    precision: str
    matched_text: str
    detector_risk_types: tuple[str, ...]
    text_preview: str

    @property
    def detector_covered(self) -> bool:
        return bool(self.detector_risk_types)

    def to_row(self) -> dict[str, Any]:
        return {
            "row_index": self.row_index,
            "phone": self.phone,
            "marker_id": self.marker_id,
            "class_id": self.class_id,
            "precision": self.precision,
            "matched_text": self.matched_text,
            "detector_covered": "Да" if self.detector_covered else "Нет",
            "detector_risk_types": " | ".join(self.detector_risk_types),
            "text_preview": self.text_preview,
        }


def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.I)


DEFAULT_POPULATION_MARKERS: tuple[CrmWritebackPopulationMarker, ...] = (
    CrmWritebackPopulationMarker(
        "hp_self_label_necelevoi",
        "A2/A3",
        "high",
        _rx(r"\bнецелев\w+\b|\bнерелевантн\w+\s+(?:звонок|обращени\w+|контакт)\b"),
        "Self-label that row is non-target / non-relevant.",
    ),
    CrmWritebackPopulationMarker(
        "hp_contentful_dialog_not_happened",
        "A1/A3",
        "high",
        _rx(r"\bсодержательн\w+\s+(?:разговор|диалог|контакт)\w*\s+не\s+(?:произошл|состоял|получил|сложил|возникл)\w*"),
        "Explicit marker that contentful dialogue did not happen.",
    ),
    CrmWritebackPopulationMarker(
        "hp_edtech_not_confirmed",
        "A2/A3",
        "high",
        _rx(r"\bedtech[-\s]?услугами?\s+не\s+подтвержд\w*|\bучебн\w+\s+центр\w*\s+не\s+подтвержд\w*"),
        "Explicit no EdTech / learning-center intent confirmation.",
    ),
    CrmWritebackPopulationMarker(
        "hp_wrong_person_or_identity_mismatch",
        "A1/A3",
        "high",
        _rx(
            r"\bконтакт\s+не\s+подтвердил\w*\b|"
            r"\bпутаниц\w+\s+с\s+именем\b|"
            r"\bна\s+линии\s+был[ао]?\s+не\s+т[ао]\b|"
            r"\bобсуждени\w+\s+(?:программ\w+,\s*)?интерес\w+\s+к\s+продукт\w+\s+и\s+следующ\w+\s+шаг\w+\s+не\s+состоял"
        ),
        "Wrong person / identity mismatch or explicit no productive discussion.",
    ),
    CrmWritebackPopulationMarker(
        "hp_remote_signing_vendor",
        "A2",
        "high",
        _rx(r"\bподпислон\b|\bсервис\w+\s+удаленн\w+\s+подписани\w+\s+документ\w+"),
        "Remote document signing vendor request.",
    ),
    CrmWritebackPopulationMarker(
        "hp_wrong_site_or_resource",
        "A2",
        "high",
        _rx(r"\bне\s+тот\s+сайт\b|\bне\s+на\s+том\s+сайте\b|\bдруг\w+\s+ресурс\w+\b|\bзаявк\w+\s+не\s+относит\w+\s+к\s+учебн\w+\s+центр\w+"),
        "Wrong site/resource marker.",
    ),
    CrmWritebackPopulationMarker(
        "hp_other_company_delivery",
        "A2",
        "high",
        _rx(r"\b(?:сотрудник|представител)\w+\s+друг\w+\s+компани\w+.{0,120}\bдоставк\w+|\bдоставк\w+\s+хозтовар\w+"),
        "Other-company delivery / household goods request.",
    ),
    CrmWritebackPopulationMarker(
        "rv_dialog_not_established",
        "A1/A3",
        "review",
        _rx(r"\b(?:разговор|диалог|связь|контакт|консультаци\w+)\s+не\s+(?:состоял|произошл|сложил|возникл|получил)\w*"),
        "Broad dialogue-not-established grammar; needs detector/context check.",
    ),
    CrmWritebackPopulationMarker(
        "rv_learning_not_discussed",
        "A3/A4",
        "review",
        _rx(r"\b(?:обучени\w+|уч[её]б\w+|продукт\w+|предмет\w+)\b.{0,80}\bне\s+(?:обсуждал|подтвержд|выявл|уточн|определ)\w*"),
        "Learning/product not discussed marker; may be valid service/outcome context.",
    ),
    CrmWritebackPopulationMarker(
        "rv_carrier_or_corporate_number",
        "A2/B2",
        "review",
        _rx(r"\b(?:ростелеком|мтс|мегафон|билайн|tele2|yota|корпоративн\w+\s+номер|многоканальн\w+\s+(?:номер|телефон))\b.{0,140}\b(?:не\s+может\s+определить|кто\s+звонил|не\s+оставлял\w+\s+заявк|удалить\s+из\s+баз\w+|запроса\s+на\s+обучени\w+\s+нет)"),
        "Carrier/corporate number with no-lead context.",
        excludes=(_rx(r"\bмтс\s+линк\b"),),
    ),
)


def scan_crm_writeback_population_recall(
    rows: Sequence[Mapping[str, Any]],
    *,
    text_fields: Iterable[str],
    detector_min_severity: str = "P2",
    high_precision_uncovered_max: int = 0,
    markers: Sequence[CrmWritebackPopulationMarker] = DEFAULT_POPULATION_MARKERS,
) -> dict[str, Any]:
    hits: list[CrmWritebackPopulationHit] = []
    detector_rows: set[int] = set()
    marker_rows: set[int] = set()
    uncovered_rows: set[int] = set()
    high_uncovered_rows: set[int] = set()
    review_uncovered_rows: set[int] = set()

    for row_index, row in enumerate(rows, start=1):
        text = _row_text(row, text_fields)
        detector_findings = detect_crm_writeback_quality_risks(text, min_severity=detector_min_severity)
        detector_risk_types = tuple(sorted({finding.risk_type for finding in detector_findings}))
        if detector_risk_types:
            detector_rows.add(row_index)
        for marker in markers:
            if any(exclude.search(text) for exclude in marker.excludes):
                continue
            for match in marker.pattern.finditer(text):
                matched = match.group(0).strip()
                if not matched:
                    continue
                marker_rows.add(row_index)
                hit = CrmWritebackPopulationHit(
                    row_index=row_index,
                    phone=str(row.get("Телефон клиента", "") or "").strip(),
                    marker_id=marker.marker_id,
                    class_id=marker.class_id,
                    precision=marker.precision,
                    matched_text=matched,
                    detector_risk_types=detector_risk_types,
                    text_preview=text[:1000],
                )
                hits.append(hit)
                if not hit.detector_covered:
                    uncovered_rows.add(row_index)
                    if marker.precision == "high":
                        high_uncovered_rows.add(row_index)
                    else:
                        review_uncovered_rows.add(row_index)

    hit_rows = [hit.to_row() for hit in hits]
    uncovered_hit_rows = [hit.to_row() for hit in hits if not hit.detector_covered]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "rows_scanned": len(rows),
        "detector_blocking_rows": len(detector_rows),
        "class_marker_prevalence_rows": len(marker_rows),
        "marker_uncovered_by_detector_rows": len(uncovered_rows),
        "high_precision_marker_uncovered_rows": len(high_uncovered_rows),
        "review_marker_uncovered_rows": len(review_uncovered_rows),
        "detector_blocking_without_marker_rows": len(detector_rows - marker_rows),
        "passed_for_live": len(high_uncovered_rows) <= high_precision_uncovered_max,
        "thresholds": {
            "high_precision_marker_uncovered_max": high_precision_uncovered_max,
        },
        "by_marker": dict(Counter(hit.marker_id for hit in hits)),
        "by_class": dict(Counter(hit.class_id for hit in hits)),
        "by_precision": dict(Counter(hit.precision for hit in hits)),
    }
    return {
        "summary": summary,
        "hits": hit_rows,
        "uncovered": uncovered_hit_rows,
        "audit_sample": uncovered_hit_rows[:200],
    }


def write_population_recall_outputs(out_root: Path, result: Mapping[str, Any]) -> dict[str, str]:
    outputs = {
        "summary_json": out_root / "crm_writeback_population_marker_summary.json",
        "marker_hits_csv": out_root / "crm_writeback_population_marker_hits.csv",
        "marker_uncovered_csv": out_root / "crm_writeback_population_marker_uncovered.csv",
        "marker_audit_sample_csv": out_root / "crm_writeback_population_marker_audit_sample.csv",
    }
    out_root.mkdir(parents=True, exist_ok=True)
    outputs["summary_json"].write_text(_json_dumps(result.get("summary") or {}), encoding="utf-8")
    _write_csv(outputs["marker_hits_csv"], list(result.get("hits") or []))
    _write_csv(outputs["marker_uncovered_csv"], list(result.get("uncovered") or []))
    _write_csv(outputs["marker_audit_sample_csv"], list(result.get("audit_sample") or []))
    return {key: str(path) for key, path in outputs.items()}


def _row_text(row: Mapping[str, Any], fields: Iterable[str]) -> str:
    return " ".join(str(row.get(field, "") or "") for field in fields).strip()


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    if not fieldnames:
        fieldnames = ["row_index", "phone", "marker_id", "class_id", "precision", "matched_text", "detector_covered", "detector_risk_types", "text_preview"]
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _json_dumps(value: object) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


__all__ = [
    "CrmWritebackPopulationHit",
    "CrmWritebackPopulationMarker",
    "DEFAULT_POPULATION_MARKERS",
    "scan_crm_writeback_population_recall",
    "write_population_recall_outputs",
]
