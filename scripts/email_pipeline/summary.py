from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RE_EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.[a-z]{2,}", re.I)
RE_PHONE_FULL = re.compile(
    r"(?<!\d)(?:\+7|8)\s*\(?\d{3,4}\)?[\s.-]*\d{2,3}[\s.-]*\d{2}[\s.-]*\d{2}(?!\d)|"
    r"(?<!\d)7[\s.-]+\(?\d{3,4}\)?[\s.-]+\d{2,3}[\s.-]+\d{2}[\s.-]+\d{2}(?!\d)"
)
RE_TELEGRAM = re.compile(r"@\w{4,}")
RE_PHONE_FRAGMENT = re.compile(r"\b\d{3}[-\s]\d{2}[-\s]\d{2}\b")
RE_RU_SURNAME_INITIALS = re.compile(r"\b[А-ЯЁ][а-яё-]{2,}\s+[А-ЯЁ]\.\s*[А-ЯЁ]\.")
RE_RU_NAME_PAIR = re.compile(r"\b[А-ЯЁ][а-яё]{2,}\s+[А-ЯЁ][а-яё]{2,}\b")
RE_LATIN_NAME_PAIR = re.compile(r"\b[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}\b")
RE_GREETING_NAME = re.compile(r"(^|\n)(\s*)[А-ЯЁ][а-яё]{2,}(?:\s+[А-ЯЁ][а-яё]{2,})?\s*,")
RE_LONG_PRIVATE_NUMBER = re.compile(r"\b\d{6,}\b")
EVENT_TYPES = {"payment", "refund", "application", "scheduling", "assessment", "contract", "tax", "medical", "broadcast", "other"}
MONEY_DIRECTIONS = {"in", "out", "none"}
AMOUNT_KINDS = {"quote", "actual_payment", "refund"}
SUBJECT_ALIASES = {
    "мат": "математика",
    "математика": "математика",
    "олимпиадная математика": "математика",
    "физ": "физика",
    "физика": "физика",
    "инфа": "информатика",
    "информатика": "информатика",
    "фм": "математика,физика",
    "физмат": "математика,физика",
    "физико-математический": "математика,физика",
    "английский": "английский",
    "русский": "русский",
    "химия": "химия",
    "биология": "биология",
}

QUOTE_HEADER = re.compile(
    r"^\s*(-{2,}\s*original message|-{2,}\s*пересылаемое|исходное сообщение|>+|on .+ wrote:|"
    r".+ (написал|написала|wrote)\s*:|\d{1,2}\.\d{1,2}\.\d{2,4}.*(пишет|написал)|"
    r"от кого:|кому:|отправлено:|sent:|from:\s)",
    re.I,
)
SIGNATURE_DIVIDER = re.compile(r"^\s*--\s*$")
FOOTER_HINT = re.compile(r"(с уважением|best regards|данное сообщение.*конфиденц|отписаться|unsubscribe|©)", re.I)


@dataclass(frozen=True)
class SummaryItem:
    message_sha256: str
    direction: str
    brand: str
    brand_source: str
    subject: str
    body: str


@dataclass(frozen=True)
class SummaryResult:
    summaries: dict[str, dict[str, Any]]
    llm_calls_total: int
    provider: str
    model: str
    reasoning: str


def split_thread_context(text: str) -> tuple[str, str]:
    if not text:
        return "", ""
    lines: list[str] = []
    context: list[str] = []
    in_context = False
    for line in text.replace("\r\n", "\n").split("\n"):
        stripped = line.strip()
        if QUOTE_HEADER.match(stripped) or SIGNATURE_DIVIDER.match(stripped):
            in_context = True
            context.append(line)
            continue
        if in_context:
            context.append(line)
            continue
        if stripped.startswith(">"):
            context.append(line)
            continue
        if FOOTER_HINT.search(stripped) and len("\n".join(lines)) > 120:
            in_context = True
            context.append(line)
            continue
        lines.append(line)
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    thread_context = re.sub(r"\n{3,}", "\n\n", "\n".join(context)).strip()
    return cleaned, thread_context


def clean_body(text: str, *, limit: int | None = None) -> str:
    cleaned, _ = split_thread_context(text)
    if limit is not None:
        return cleaned[:limit]
    return cleaned


def mask_pii(text: str, *, mask_names: bool = True, mask_requisites: bool = True) -> str:
    value = (text or "").replace("&nbsp;", " ")
    value = RE_EMAIL.sub("[email]", value)
    value = RE_PHONE_FULL.sub("[phone]", value)
    value = RE_PHONE_FRAGMENT.sub("[phone]", value)
    value = RE_TELEGRAM.sub("[handle]", value)
    if mask_names:
        value = RE_RU_SURNAME_INITIALS.sub("[name]", value)
        value = RE_RU_NAME_PAIR.sub("[name]", value)
        value = RE_LATIN_NAME_PAIR.sub("[name]", value)
        value = RE_GREETING_NAME.sub(r"\1\2[name],", value)
    if mask_requisites:
        value = RE_LONG_PRIVATE_NUMBER.sub("[id]", value)
    return value


def summarize_items(
    items: list[SummaryItem],
    *,
    provider: str,
    model: str,
    reasoning: str,
    batch_size: int,
    max_llm_calls: int,
    project_root: Path,
    codex_home: Path | None = None,
    timeout_sec: int = 240,
) -> SummaryResult:
    if not items:
        return SummaryResult({}, 0, provider, model, reasoning)
    resolved_provider = provider
    if provider == "auto":
        resolved_provider = "openai" if os.getenv("OPENAI_API_KEY") else "codex_cli"
    summaries: dict[str, dict[str, Any]] = {}
    llm_calls = 0
    for offset in range(0, len(items), batch_size):
        if llm_calls >= max_llm_calls:
            raise RuntimeError(f"LLM call limit exceeded: {llm_calls} >= {max_llm_calls}")
        batch = items[offset : offset + batch_size]
        payload = _call_summary_provider(
            batch,
            provider=resolved_provider,
            model=model,
            reasoning=reasoning,
            project_root=project_root,
            codex_home=codex_home,
            timeout_sec=timeout_sec,
        )
        llm_calls += 1
        expected = {item.message_sha256 for item in batch}
        for row in _extract_summaries(payload, expected=expected, require_complete=False):
            summaries[str(row["message_sha256"])] = _normalize_summary_row(row)
        missing = expected - summaries.keys()
        for item in batch:
            if item.message_sha256 not in missing:
                continue
            if llm_calls >= max_llm_calls:
                raise RuntimeError(f"LLM call limit exceeded while repairing missing summaries: {llm_calls} >= {max_llm_calls}")
            repair_payload = _call_summary_provider(
                [item],
                provider=resolved_provider,
                model=model,
                reasoning=reasoning,
                project_root=project_root,
                codex_home=codex_home,
                timeout_sec=timeout_sec,
            )
            llm_calls += 1
            for row in _extract_summaries(repair_payload, expected={item.message_sha256}, require_complete=True):
                summaries[str(row["message_sha256"])] = _normalize_summary_row(row)
    return SummaryResult(summaries, llm_calls, resolved_provider, model, reasoning)


def _call_summary_provider(
    batch: list[SummaryItem],
    *,
    provider: str,
    model: str,
    reasoning: str,
    project_root: Path,
    codex_home: Path | None,
    timeout_sec: int,
) -> dict[str, Any]:
    prompt = build_summary_prompt(batch)
    if provider == "openai":
        return _call_openai_json(prompt, model=model, reasoning=reasoning, timeout_sec=timeout_sec)
    if provider == "codex_cli":
        return _call_codex_json(
            prompt,
            model=model,
            reasoning=reasoning,
            project_root=project_root,
            codex_home=codex_home,
            timeout_sec=timeout_sec,
        )
    if provider == "stub":
        return {
            "summaries": [
                {
                    "message_sha256": item.message_sha256,
                    "summary": "Тестовая заглушка сводки.",
                    "topic": "stub",
                    "next_step": None,
                    "confidence": 0.0,
                    "extraction_source": "fallback",
                    "event_type": "other",
                    "money_direction": "none",
                    "student_name": None,
                    "grade": None,
                    "subject_area": None,
                    "amount_rub": None,
                    "amount_kind": None,
                    "amount_is_total": False,
                    "amount_items": [],
                    "amount_uncertain": False,
                    "deadline_date": None,
                    "contract_no": None,
                    "document_no": None,
                    "requisites": [],
                    "has_attachment": False,
                    "payer_name": None,
                    "contact_name": None,
                    "is_plain_acknowledgement": False,
                }
                for item in batch
            ]
        }
    raise RuntimeError(f"Unsupported summary provider: {provider}")


def _normalize_summary_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    normalized["summary"] = _humanize_mask_tokens(str(normalized.get("summary") or "").strip())
    normalized["topic"] = _humanize_mask_tokens(str(normalized.get("topic") or "").strip())
    next_step = normalized.get("next_step")
    normalized["next_step"] = _humanize_mask_tokens(str(next_step).strip()) if next_step not in (None, "") else None
    normalized["event_type"] = _enum_value(normalized.get("event_type"), EVENT_TYPES, "other")
    normalized["money_direction"] = _enum_value(normalized.get("money_direction"), MONEY_DIRECTIONS, "none")
    normalized["amount_kind"] = _enum_value(normalized.get("amount_kind"), AMOUNT_KINDS, None)
    normalized["student_name"] = _nullable_string(normalized.get("student_name"))
    normalized["grade"] = _canonical_grade(normalized.get("grade"))
    normalized["subject_area"] = _canonical_subject_area(normalized.get("subject_area"))
    normalized["amount_rub"] = _nullable_int(normalized.get("amount_rub"))
    normalized["amount_is_total"] = bool(normalized.get("amount_is_total")) if normalized.get("amount_is_total") is not None else False
    normalized["amount_items"] = _normalize_amount_items(normalized.get("amount_items"))
    normalized["amount_uncertain"] = bool(normalized.get("amount_uncertain")) if normalized.get("amount_uncertain") is not None else False
    normalized["deadline_date"] = _nullable_string(normalized.get("deadline_date"))
    normalized["contract_no"] = _nullable_string(normalized.get("contract_no"))
    normalized["document_no"] = _nullable_string(normalized.get("document_no"))
    normalized["payer_name"] = _nullable_string(normalized.get("payer_name"))
    normalized["contact_name"] = _nullable_string(normalized.get("contact_name"))
    normalized["is_plain_acknowledgement"] = bool(normalized.get("is_plain_acknowledgement"))
    requisites = normalized.get("requisites")
    normalized["requisites"] = [str(item).strip() for item in requisites if str(item).strip()] if isinstance(requisites, list) else []
    normalized["has_attachment"] = bool(normalized.get("has_attachment")) if normalized.get("has_attachment") is not None else False
    if str(normalized.get("extraction_source") or "") not in {"model", "fallback"}:
        normalized["extraction_source"] = "fallback"
    return normalized


def _enum_value(value: object, allowed: set[str], default: str | None) -> str | None:
    text = str(value or "").strip()
    return text if text in allowed else default


def _nullable_string(value: object) -> str | None:
    text = str(value or "").strip()
    return text if text else None


def _nullable_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_amount_items(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    items: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        amount = _nullable_int(item.get("amount_rub"))
        kind = _enum_value(item.get("amount_kind"), AMOUNT_KINDS, None)
        if amount is None and kind is None:
            continue
        items.append(
            {
                "amount_rub": amount,
                "amount_kind": kind,
                "description": _nullable_string(item.get("description")),
                "is_total": bool(item.get("is_total")),
            }
        )
    return items


def _canonical_grade(value: object) -> str | None:
    text = _nullable_string(value)
    if not text:
        return None
    match = re.search(r"\b([1-9]|1[0-1])\b", text)
    return match.group(1) if match else text


def _canonical_subject_area(value: object) -> str | None:
    text = _nullable_string(value)
    if not text:
        return None
    lowered = re.sub(r"\s+", " ", text.casefold()).strip()
    if lowered in SUBJECT_ALIASES:
        return SUBJECT_ALIASES[lowered]
    parts: list[str] = []
    for chunk in re.split(r"[,;/+]| и ", lowered):
        cleaned = chunk.strip()
        item = SUBJECT_ALIASES.get(cleaned, cleaned)
        if item and item not in parts:
            parts.append(item)
    return ",".join(parts) if parts else text


def _humanize_mask_tokens(text: str) -> str:
    return (
        (text or "")
        .replace("[phone]", "телефон")
        .replace("[Phone]", "телефон")
        .replace("[PHONE]", "телефон")
        .replace("[number]", "число")
        .replace("[Number]", "число")
        .replace("[NUMBER]", "число")
        .replace("[email]", "email")
        .replace("[Email]", "email")
        .replace("[EMAIL]", "email")
        .replace("[name]", "имя")
        .replace("[Name]", "имя")
        .replace("[NAME]", "имя")
        .replace("[handle]", "контакт")
        .replace("[Handle]", "контакт")
        .replace("[HANDLE]", "контакт")
    )


def build_summary_prompt(items: list[SummaryItem]) -> str:
    records = []
    for item in items:
        records.append(
            {
                "message_sha256": item.message_sha256,
                "direction": item.direction,
                "brand": item.brand,
                "brand_source": item.brand_source,
                "subject": mask_pii(item.subject, mask_names=False, mask_requisites=False)[:800],
                "body": mask_pii(clean_body(item.body, limit=6000), mask_names=False, mask_requisites=False),
            }
        )
    return (
        "Ты строишь manager-only сводки e-mail для учебного центра. "
        "Верни строго JSON object с ключом summaries. Не используй markdown. "
        "На каждый входной message_sha256 верни одну сводку. "
        "Не выдумывай факты, суммы, бренды и следующие шаги. "
        "Если brand='none', не называй Фотон, УНПК, МФТИ, Физтех, cdpofoton.ru, kmipt.ru или другие бренд-домены "
        "в summary/topic/next_step; вместо этого пиши нейтрально 'ссылка' или 'сайт'. "
        "Ключевое: сохраняй конкретику из письма. Не заменяй время, расписание, цену, дату, учебный год, класс, курс или группу "
        "общими словами вроде 'указанное расписание', 'указанная стоимость', 'данное время'. "
        "Если в письме есть 'Воскресенье 10:50-12:30', '2025-2026 уч.г.', '126 000 руб.', '8 класс' или название группы, "
        "перенеси эту конкретику в summary/next_step, если она относится к сути письма. "
        "ПДн во входе уже скрыты маркерами, не восстанавливай и не выдумывай их; не уничтожай учебные числа, даты, время, цены и группы. "
        "Имена учеников и родителей во входе НЕ скрыты: различай роли. Ребёнка клади в student_name, взрослого/плательщика в payer_name/contact_name; "
        "не клади первого попавшегося взрослого в student_name. "
        "Дополнительно извлеки закрытые поля: event_type один из payment/refund/application/scheduling/assessment/contract/tax/medical/broadcast/other; "
        "money_direction один из in/out/none; amount_kind один из quote/actual_payment/refund или null; "
        "student_name, payer_name, contact_name, grade, subject_area, amount_rub integer RUB или null, amount_is_total boolean, "
        "amount_items list, amount_uncertain boolean, deadline_date, contract_no, document_no, requisites list, has_attachment boolean, "
        "is_plain_acknowledgement boolean. "
        "amount_kind: quote = цена/стоимость/предложение, actual_payment = факт оплаты/поступления, refund = возврат. "
        "amount_is_total=true только если модель видит, что сумма итоговая; отдельные позиции положи в amount_items. "
        "Если сумма неоднозначна, слишком странна или непонятно, это цена или факт оплаты, ставь amount_uncertain=true и не угадывай. "
        "Не ставь amount_uncertain=true только потому, что в письме несколько ясных цен/скидок/вариантов; если каждая сумма читается нормально, это quote и amount_uncertain=false. "
        "is_plain_acknowledgement=true только для чистого подтверждения факта без новой инструкции, спора или возврата. "
        "Если денег в письме нет, amount_rub=null и amount_uncertain=false. "
        "extraction_source всегда 'model', если ты понял поля сам из письма. "
        "Схема каждой строки: {message_sha256, summary, topic, next_step, confidence, extraction_source, event_type, money_direction, student_name, payer_name, contact_name, grade, subject_area, amount_rub, amount_kind, amount_is_total, amount_items, amount_uncertain, deadline_date, contract_no, document_no, requisites, has_attachment, is_plain_acknowledgement}. "
        "summary: 2-4 предложения по-русски, с фактами и конкретными значениями; topic: 2-8 слов; "
        "next_step: конкретный следующий шаг или null; confidence: 0..1.\n\n"
        + json.dumps({"emails": records}, ensure_ascii=False)
    )


def _call_openai_json(prompt: str, *, model: str, reasoning: str, timeout_sec: int) -> dict[str, Any]:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for provider=openai")
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    instructions = "Return one valid JSON object only. No markdown."
    try:
        response = client.responses.create(
            model=model,
            instructions=instructions,
            input=prompt,
            reasoning={"effort": reasoning},
            text={"format": {"type": "json_object"}},
            timeout=timeout_sec,
        )
        text = getattr(response, "output_text", "") or ""
    except Exception:
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": instructions}, {"role": "user", "content": prompt}],
            timeout=timeout_sec,
        )
        text = response.choices[0].message.content if response.choices else ""
    return _extract_json_object(text)


def _call_codex_json(
    prompt: str,
    *,
    model: str,
    reasoning: str,
    project_root: Path,
    codex_home: Path | None,
    timeout_sec: int,
) -> dict[str, Any]:
    codex_bin = "codex"
    if shutil.which(codex_bin) is None:
        raise RuntimeError("codex binary is not available")
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["summaries"],
        "properties": {
            "summaries": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "message_sha256",
                        "summary",
                        "topic",
                        "next_step",
                        "confidence",
                        "extraction_source",
                        "event_type",
                        "money_direction",
                        "student_name",
                        "payer_name",
                        "contact_name",
                        "grade",
                        "subject_area",
                        "amount_rub",
                        "amount_kind",
                        "amount_is_total",
                        "amount_items",
                        "amount_uncertain",
                        "deadline_date",
                        "contract_no",
                        "document_no",
                        "requisites",
                        "has_attachment",
                        "is_plain_acknowledgement",
                    ],
                    "properties": {
                        "message_sha256": {"type": "string"},
                        "summary": {"type": "string"},
                        "topic": {"type": "string"},
                        "next_step": {"type": ["string", "null"]},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "extraction_source": {"type": "string", "enum": ["model", "fallback"]},
                        "event_type": {"type": "string", "enum": sorted(EVENT_TYPES)},
                        "money_direction": {"type": "string", "enum": sorted(MONEY_DIRECTIONS)},
                        "student_name": {"type": ["string", "null"]},
                        "payer_name": {"type": ["string", "null"]},
                        "contact_name": {"type": ["string", "null"]},
                        "grade": {"type": ["string", "null"]},
                        "subject_area": {"type": ["string", "null"]},
                        "amount_rub": {"type": ["integer", "null"]},
                        "amount_kind": {"type": ["string", "null"], "enum": ["quote", "actual_payment", "refund", None]},
                        "amount_is_total": {"type": "boolean"},
                        "amount_items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["amount_rub", "amount_kind", "description", "is_total"],
                                "properties": {
                                    "amount_rub": {"type": ["integer", "null"]},
                                    "amount_kind": {"type": ["string", "null"], "enum": ["quote", "actual_payment", "refund", None]},
                                    "description": {"type": ["string", "null"]},
                                    "is_total": {"type": "boolean"},
                                },
                            },
                        },
                        "amount_uncertain": {"type": "boolean"},
                        "deadline_date": {"type": ["string", "null"]},
                        "contract_no": {"type": ["string", "null"]},
                        "document_no": {"type": ["string", "null"]},
                        "requisites": {"type": "array", "items": {"type": "string"}},
                        "has_attachment": {"type": "boolean"},
                        "is_plain_acknowledgement": {"type": "boolean"},
                    },
                },
            }
        },
    }
    with tempfile.NamedTemporaryFile(prefix="email_summary_", suffix=".json") as out_file, tempfile.NamedTemporaryFile(
        prefix="email_summary_schema_", suffix=".json", mode="w", encoding="utf-8"
    ) as schema_file:
        schema_file.write(json.dumps(schema, ensure_ascii=False))
        schema_file.flush()
        cmd = [
            codex_bin,
            "exec",
            "--skip-git-repo-check",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--sandbox",
            "read-only",
            "--cd",
            str(project_root.resolve()),
            "--model",
            model,
            "--output-schema",
            str(Path(schema_file.name)),
            "--output-last-message",
            str(Path(out_file.name)),
            "-c",
            f'model_reasoning_effort="{reasoning}"',
            "-",
        ]
        env = os.environ.copy()
        if codex_home is not None:
            env["CODEX_HOME"] = str(codex_home.expanduser().resolve())
        proc = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_sec,
            env=env,
        )
        raw = Path(out_file.name).read_text(encoding="utf-8", errors="ignore").strip()
    for candidate in (raw, proc.stdout or "", proc.stderr or ""):
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            return _extract_json_object(candidate)
        except Exception:
            continue
    stderr_tail = (proc.stderr or "").strip().splitlines()[-1:] or [""]
    raise RuntimeError(f"codex exec returned no JSON; rc={proc.returncode}; stderr_tail={stderr_tail[0]}")


def _extract_summaries(payload: dict[str, Any], *, expected: set[str], require_complete: bool = True) -> list[dict[str, Any]]:
    rows = payload.get("summaries")
    if not isinstance(rows, list):
        raise RuntimeError("summary payload must contain summaries list")
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        sha = str(row.get("message_sha256") or "")
        if sha in expected:
            seen.add(sha)
            output.append(row)
    missing = expected - seen
    if missing and require_complete:
        raise RuntimeError(f"summary payload missing {len(missing)} message_sha256 values")
    return output


def _extract_json_object(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(raw[start : end + 1])
    if not isinstance(payload, dict):
        raise RuntimeError("JSON root must be object")
    return payload
