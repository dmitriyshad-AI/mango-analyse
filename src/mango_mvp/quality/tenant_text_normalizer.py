from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable


BRAND_ALIASES_RE = re.compile(
    r"\b(?:(?:[А-ЯA-Z]?МПК|(?!УНПК)[А-ЯA-Z]?НПК|О\s*Н\s*П\s*К|Н\s*П\s*К|УНФК|УНП|ЛНПК|"
    r"У\s*Н\s*И\s*П\s*К)\s*М\s*[ФШ]\s*(?:[ТДП](?:\s*[ИI])?|[ИI])|"
    r"У\s*Н\s*П\s*К\s*М\s*[ФШ]\s*(?:[ДП](?:\s*[ИI])?|Т\b|[ИI]\b))\b",
    re.IGNORECASE,
)
UNPK_MFTI_TAIL_VARIANT_RE = re.compile(
    r"\bУ\s*Н\s*П\s*К\s*М\s*Ф\s*Т\s*[ИI](?:Ш|К|Й|В|НГ)\b",
    re.IGNORECASE,
)
DETECTOR_KNOWN_BRAND_VARIANTS = (
    "МПК МФТИ",
    "НПК МФТИ",
    "ОНПК МФТИ",
    "ВНПК МФТИ",
    "МНПК МФТИ",
    "УНПК МФП",
    "ЛНПК МФТИ",
    "УНФК МФТИ",
    "УНП МФТИ",
    "УНИПК МФТИ",
    "УНПК МФТИШ",
    "УНПК МФТИК",
    "УНПК МФТИЙ",
    "УНПК МФТИВ",
    "УНПК МФТИНГ",
)
DETECTOR_BRAND_GENERAL_PATTERNS = (
    re.compile(r"\bУ\s*Н\s*П\s*К\s*М\s*Ф\s*Т\s*[ИI][А-ЯЁA-Z]{1,2}\b", re.IGNORECASE),
    re.compile(r"\bМ[ФШ][А-ЯЁA-Z]{3,}\b"),
)

SUMMER_NIGHT_SCHOOL_PATTERNS = (
    (re.compile(r"\bлетняя\s+ночная\s+школа\b", re.IGNORECASE), "летняя очная школа"),
    (re.compile(r"\bлетней\s+ночной\s+школой\b", re.IGNORECASE), "летней очной школой"),
    (re.compile(r"\bлетнюю\s+ночную\s+школу\b", re.IGNORECASE), "летнюю очную школу"),
    (re.compile(r"\bлетние\s+ночные\s+школы\b", re.IGNORECASE), "летние очные школы"),
    (re.compile(r"\bлетних\s+ночных\s+школах\b", re.IGNORECASE), "летних очных школах"),
    (re.compile(r"\bлетними\s+ночными\s+школами\b", re.IGNORECASE), "летними очными школами"),
    (re.compile(r"\bлетн(?:яя|ей|юю)\s+ночн(?:ая|ой|ую)\s+школ\w*", re.IGNORECASE), "летняя очная школа"),
    (re.compile(r"\bлетн(?:ие|их|ими)\s+ночн(?:ые|ых|ыми)\s+школ\w*", re.IGNORECASE), "летние очные школы"),
)

COUNT_SUFFIX_RE = re.compile(r"\s*\(\s*\d+\s+касани[йя]\s*\)\s*$", re.IGNORECASE)
COUNTED_LABEL_RE = re.compile(r"^(?P<label>.+?)\s*:\s*\d+\s*$")
DATE_PREFIX_RE = re.compile(r"^\s*\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?\s*:\s*")
PRODUCT_COUNT_ARTIFACT_RE = re.compile(r"\(\s*\d+\s+касани[йя]\s*\)|\b[^|:]{3,80}:\s*\d+\b", re.IGNORECASE)


@dataclass(frozen=True)
class TenantTextArtifact:
    class_id: str
    matched_text: str
    reason: str


def normalize_manager_text(value: Any) -> str:
    """Normalize tenant-specific ASR/LLM artifacts in manager-facing CRM text."""
    text = "" if value is None else str(value).strip()
    if not text:
        return ""
    text = BRAND_ALIASES_RE.sub("УНПК МФТИ", text)
    text = UNPK_MFTI_TAIL_VARIANT_RE.sub("УНПК МФТИ", text)
    for pattern, replacement in SUMMER_NIGHT_SCHOOL_PATTERNS:
        text = pattern.sub(replacement, text)
    return re.sub(r"\s+", " ", text).strip()


def detect_residual_manager_text_artifacts(value: Any) -> list[TenantTextArtifact]:
    """Find closed-class artifacts that must not survive after tenant normalization."""
    text = "" if value is None else str(value)
    if not text:
        return []
    findings: list[TenantTextArtifact] = []
    seen_matches: set[str] = set()

    def add_finding(class_id: str, matched_text: str, reason: str) -> None:
        key = re.sub(r"\s+", " ", matched_text).strip().casefold()
        if not key or key in seen_matches:
            return
        seen_matches.add(key)
        findings.append(TenantTextArtifact(class_id=class_id, matched_text=matched_text, reason=reason))

    for variant in DETECTOR_KNOWN_BRAND_VARIANTS:
        pattern = re.compile(rf"(?<!\w){re.escape(variant)}(?!\w)", re.IGNORECASE)
        for match in pattern.finditer(text):
            add_finding(
                "known_brand_variant_residual",
                match.group(0),
                "Known ASR/LLM brand artifact must be normalized to УНПК МФТИ",
            )
    for pattern in DETECTOR_BRAND_GENERAL_PATTERNS:
        for match in pattern.finditer(text):
            add_finding(
                "suspicious_brand_pattern",
                match.group(0),
                "Suspicious manager-facing brand pattern must be reviewed or normalized",
            )
    for match in BRAND_ALIASES_RE.finditer(text):
        add_finding(
            "tenant_brand_alias",
            match.group(0),
            "ASR/LLM brand artifact must be normalized to УНПК МФТИ",
        )
    for pattern, _replacement in SUMMER_NIGHT_SCHOOL_PATTERNS:
        for match in pattern.finditer(text):
            add_finding(
                "summer_night_school_asr_artifact",
                match.group(0),
                "ASR artifact must be normalized to летняя/летние очная/очные школа/школы",
            )
    return findings


def detect_product_list_artifacts(value: Any) -> list[TenantTextArtifact]:
    text = "" if value is None else str(value)
    if not text:
        return []
    return [
        TenantTextArtifact(
            class_id="product_count_artifact",
            matched_text=match.group(0),
            reason="Manager-facing product list must not expose technical touch counts",
        )
        for match in PRODUCT_COUNT_ARTIFACT_RE.finditer(text)
    ]


def normalize_product_label(value: Any) -> str:
    text = normalize_manager_text(_strip_count(value)).strip(" .;,")
    if not text:
        return ""
    key = text.casefold()
    if "летн" in key and "школ" in key and ("очн" in key or "ночн" in key):
        return "летняя очная школа"
    if key in {"летняя школа", "летние школы", "летняя летняя школа"}:
        return "летняя очная школа"
    if ("летн" in key and ("лагер" in key or "выездн" in key)) or key in {"лвш", "летняя выездная школа"}:
        return "летний лагерь"
    if "индивидуаль" in key:
        return "индивидуальные занятия"
    if "годов" in key:
        return "годовые курсы"
    return text


def format_product_list(values: Any | Iterable[Any], *, max_items: int | None = None) -> str:
    result: list[str] = []
    seen: set[str] = set()
    for item in _iter_items(values):
        normalized = normalize_product_label(item)
        key = normalized.casefold()
        if not normalized or key in seen:
            continue
        seen.add(key)
        result.append(normalized)
        if max_items is not None and len(result) >= max_items:
            break
    return " | ".join(result)


def normalize_objection_label(value: Any) -> str:
    text = normalize_manager_text(_strip_count(value)).strip(" .;,")
    if not text:
        return ""
    text = re.sub(r"^(?:Актуальные|Исторические)\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    text = DATE_PREFIX_RE.sub("", text).strip()
    key = text.casefold()
    if "муж" in key and ("договор" in key or "обсуд" in key or "прочит" in key):
        return "нужно согласовать договор с мужем"
    return text


def objection_key(value: Any) -> str:
    return re.sub(r"\s+", " ", normalize_objection_label(value).casefold()).strip(" .;,")


def format_objection_list(values: Any | Iterable[Any], *, max_items: int | None = None) -> str:
    result: list[str] = []
    seen: set[str] = set()
    for item in _iter_items(values):
        normalized = normalize_objection_label(item)
        key = objection_key(normalized)
        if not normalized or key in seen:
            continue
        seen.add(key)
        result.append(normalized)
        if max_items is not None and len(result) >= max_items:
            break
    return " | ".join(result)


def _strip_count(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    text = COUNT_SUFFIX_RE.sub("", text)
    match = COUNTED_LABEL_RE.match(text)
    if match:
        text = match.group("label").strip()
    return text


def _iter_items(values: Any | Iterable[Any]) -> Iterable[str]:
    if values is None:
        return
    if isinstance(values, str):
        raw_values: Iterable[Any] = [values]
    else:
        raw_values = values
    for raw in raw_values:
        text = "" if raw is None else str(raw)
        for part in re.split(r"\s*\|\s*|;\s*", text.replace("\n", " | ")):
            value = part.strip(" .;,")
            if value:
                yield value
