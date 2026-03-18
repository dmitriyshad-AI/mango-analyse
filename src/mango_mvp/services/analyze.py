from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from openai import OpenAI
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from mango_mvp.clients.ollama import OllamaClient
from mango_mvp.config import Settings
from mango_mvp.models import CallRecord

SYSTEM_PROMPT = """You are a sales call analyst for Russian EdTech phone calls.
Return strict JSON only with these keys:
- analysis_schema_version (string, always "v2")
- history_summary (string, concise CRM-ready call history in Russian).
  Format requirement: 3-6 sentences in Russian prose.
  Sentence 1 MUST include call date/time and manager name if available.
  Then include what was discussed, key constraints/objections, and exact agreement/next step.
  Do NOT output raw dialogue or timestamped replicas.
- structured_fields (object):
  - people: {parent_fio: string|null, child_fio: string|null}
  - contacts: {email: string|null, phone_from_filename: string|null, preferred_channel: string|null}
  - student: {grade_current: string|null, school: string|null}
  - interests: {products: string[], format: string[], subjects: string[], exam_targets: string[]}
  - commercial: {price_sensitivity: "high"|"medium"|"low"|null, budget: string|null, discount_interest: boolean|null}
  - objections: string[]
  - next_step: {action: string|null, due: string|null}
  - lead_priority: "hot"|"warm"|"cold"|null
- evidence (array of objects): [{speaker: string|null, ts: string|null, text: string}]
- quality_flags (object)

Also include legacy compatibility keys:
- history_short (string)
- crm_blocks (object, same structure as structured_fields)
- summary (string)
- interests (array of strings)
- student_grade (string|null)
- target_product (string|null)
- personal_offer (string|null)
- pain_points (array of strings)
- budget (string|null)
- timeline (string|null)
- objections (array of strings)
- next_step (string|null)
- follow_up_score (integer 0..100)
- follow_up_reason (string)
- tags (array of strings)

No markdown, no comments, no extra keys."""

NON_CONVERSATION_MARKERS = (
    "продолжение следует",
    "голосовой ассистент",
    "абонент не может ответить",
    "номер недоступен",
    "оставьте сообщение",
    "после сигнала",
    "оставайтесь на линии",
    "дозванивайтесь",
    "дозваниваться",
)

LATEST_ANALYSIS_SCHEMA_VERSION = "v2"

EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
GRADE_RE = re.compile(r"\b((?:[1-9]|1[0-1]))(?:-?й)?\s*класс(?:а)?\b", re.I)
SPEAKER_LINE_RE = re.compile(
    r"^\[(?P<ts>[^\]]+)\]\s*(?P<speaker>[^:]{1,60}):\s*(?P<text>.+)$",
    re.U,
)
SUBJECT_PATTERNS = {
    "математика": re.compile(r"математ\w*", re.I),
    "физика": re.compile(r"физик\w*", re.I),
    "информатика": re.compile(r"информат\w*|python|питон|алгоритм", re.I),
    "русский язык": re.compile(r"русск\w* язык", re.I),
    "химия": re.compile(r"хими\w*", re.I),
    "биология": re.compile(r"биолог\w*", re.I),
    "английский язык": re.compile(r"английск\w*", re.I),
}
FORMAT_PATTERNS = {
    "онлайн": re.compile(r"\bонлайн\b", re.I),
    "оффлайн": re.compile(r"\bоффлайн\b|\bочно\b", re.I),
    "групповой": re.compile(r"\bгрупп\w*", re.I),
    "индивидуальный": re.compile(r"индивидуал\w*", re.I),
    "с проживанием": re.compile(r"с проживанием|проживани[ея]", re.I),
    "без проживания": re.compile(r"без проживания", re.I),
}
PRODUCT_PATTERNS = {
    "годовые курсы": re.compile(r"годов\w* курс|на год|годов\w* программ", re.I),
    "летний лагерь": re.compile(r"летн(?:ий|его) лаг|летн\w* смен", re.I),
    "интенсив": re.compile(r"интенсив\w*", re.I),
}
EXAM_PATTERNS = {
    "ЕГЭ": re.compile(r"\bегэ\b", re.I),
    "ОГЭ": re.compile(r"\bогэ\b", re.I),
    "олимпиады": re.compile(r"олимпиад\w*", re.I),
}
OBJECTION_PATTERNS = {
    "цена": re.compile(r"цен\w*|стоимост\w*|дорог\w*|дешев\w*|бюджет\w*", re.I),
    "время": re.compile(r"нет времени|занят\w*|нагрузк\w*|расписан\w*", re.I),
    "доверие": re.compile(r"кто вы|не слышал\w* о вас|отзыв\w*|гаранти", re.I),
    "неактуально": re.compile(r"не актуальн\w*|не интерес\w*|не нужно", re.I),
}
DIALOGUE_DUMP_LINE_RE = re.compile(r"^\[(?:~)?\d{2}:\d{2}(?:\.\d+)?\]\s*", re.M)

ANALYSIS_PROMPT_TRANSCRIPT_MAX_CHARS = 10000
ANALYSIS_PROMPT_TRANSCRIPT_HEAD_CHARS = 7000
ANALYSIS_PROMPT_TRANSCRIPT_TAIL_CHARS = 2800


class AnalyzeService:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._client: Optional[OpenAI] = None
        self._ollama_client_instance: Optional[OllamaClient] = None

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)

    def _retry_delay(self, attempts: int) -> timedelta:
        base = max(1, self._settings.retry_base_delay_sec)
        multiplier = max(1, 2 ** max(0, attempts - 1))
        return timedelta(seconds=base * multiplier)

    def _openai_client(self) -> OpenAI:
        if not self._settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for openai analyze provider")
        if self._client is None:
            self._client = OpenAI(api_key=self._settings.openai_api_key)
        return self._client

    def _ollama_client(self) -> OllamaClient:
        if self._ollama_client_instance is None:
            self._ollama_client_instance = OllamaClient(self._settings.ollama_base_url)
        return self._ollama_client_instance

    @staticmethod
    def _clean_text(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @classmethod
    def _clean_list(cls, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        result: list[str] = []
        for item in value:
            text = cls._clean_text(item)
            if text:
                result.append(text)
        return result

    @staticmethod
    def _unique(values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for item in values:
            key = item.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            result.append(item.strip())
        return result

    @staticmethod
    def _nested_dict(payload: Dict[str, Any], key: str) -> Dict[str, Any]:
        value = payload.get(key)
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _coerce_score(value: Any) -> Optional[int]:
        try:
            score = int(value)
        except (TypeError, ValueError):
            return None
        if score < 0:
            return 0
        if score > 100:
            return 100
        return score

    @staticmethod
    def _priority_from_score(score: int) -> str:
        if score >= 75:
            return "hot"
        if score >= 45:
            return "warm"
        return "cold"

    def _detect_from_patterns(self, text: str, patterns: Dict[str, re.Pattern[str]]) -> list[str]:
        lowered = text.lower()
        detected: list[str] = []
        for label, pattern in patterns.items():
            if pattern.search(lowered):
                detected.append(label)
        return detected

    def _detect_preferred_channel(self, text: str) -> Optional[str]:
        lowered = text.lower()
        if "телеграм" in lowered or "telegram" in lowered:
            return "telegram"
        if "ватсап" in lowered or "whatsapp" in lowered or "вотсап" in lowered:
            return "whatsapp"
        if "почт" in lowered or "email" in lowered or "e-mail" in lowered:
            return "email"
        if "сайт" in lowered or "заявк" in lowered:
            return "site"
        return None

    @staticmethod
    def _extract_email(text: str) -> Optional[str]:
        match = EMAIL_RE.search(text or "")
        if not match:
            return None
        return match.group(0).lower()

    @staticmethod
    def _extract_grade(text: str) -> Optional[str]:
        match = GRADE_RE.search(text or "")
        if not match:
            return None
        return match.group(1)

    @staticmethod
    def _extract_evidence(text: str, limit: int = 3) -> list[Dict[str, str]]:
        evidence: list[Dict[str, str]] = []
        for line in (text or "").splitlines():
            match = SPEAKER_LINE_RE.match(line.strip())
            if not match:
                continue
            snippet = (match.group("text") or "").strip()
            if len(snippet) < 12:
                continue
            evidence.append(
                {
                    "speaker": (match.group("speaker") or "").strip(),
                    "ts": (match.group("ts") or "").strip(),
                    "text": snippet[:260],
                }
            )
            if len(evidence) >= limit:
                break
        return evidence

    @staticmethod
    def _extract_json_payload(text: str) -> Dict[str, Any]:
        raw = (text or "").strip()
        if not raw:
            raise RuntimeError("empty response")
        try:
            payload = json.loads(raw)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass

        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL)
        if fence:
            payload = json.loads(fence.group(1))
            if isinstance(payload, dict):
                return payload

        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            payload = json.loads(raw[start : end + 1])
            if isinstance(payload, dict):
                return payload
        raise RuntimeError("response does not contain JSON object")

    @staticmethod
    def _format_started_at(started_at: Optional[datetime]) -> Optional[str]:
        if started_at is None:
            return None
        dt = started_at
        if dt.tzinfo is not None:
            dt = dt.astimezone()
        return dt.strftime("%d.%m.%Y %H:%M")

    @staticmethod
    def _sentence(text: Optional[str]) -> Optional[str]:
        if not text:
            return None
        compact = re.sub(r"\s+", " ", text).strip()
        if not compact:
            return None
        if compact[-1] not in ".!?":
            compact = f"{compact}."
        return compact

    @staticmethod
    def _looks_like_dialogue_dump(text: str) -> bool:
        raw = (text or "").strip()
        if not raw:
            return False
        if DIALOGUE_DUMP_LINE_RE.search(raw):
            return True
        line_count = len([line for line in raw.splitlines() if line.strip()])
        if line_count >= 3 and ("клиент:" in raw.lower() or "менеджер" in raw.lower()):
            return True
        return False

    def _analysis_user_prompt(self, call: CallRecord, text: str) -> str:
        started_at = self._format_started_at(call.started_at) or "unknown"
        manager = self._clean_text(call.manager_name) or "unknown"
        phone = self._clean_text(call.phone) or "unknown"
        direction = self._clean_text(call.direction) or "unknown"
        transcript = text or ""
        if len(transcript) > ANALYSIS_PROMPT_TRANSCRIPT_MAX_CHARS:
            head = transcript[:ANALYSIS_PROMPT_TRANSCRIPT_HEAD_CHARS].rstrip()
            tail = transcript[-ANALYSIS_PROMPT_TRANSCRIPT_TAIL_CHARS :].lstrip()
            transcript = (
                f"{head}\n\n"
                "[... transcript truncated for prompt budget ...]\n\n"
                f"{tail}"
            )
        return (
            "Analyze this call transcript and return JSON only.\n"
            "Call metadata:\n"
            f"- source_filename: {call.source_filename}\n"
            f"- started_at: {started_at}\n"
            f"- manager_name: {manager}\n"
            f"- client_phone: {phone}\n"
            f"- direction: {direction}\n\n"
            f"Transcript:\n{transcript}"
        )

    def _compose_history_summary(
        self,
        call: CallRecord,
        *,
        draft_history_summary: Optional[str],
        summary: Optional[str],
        structured_fields: Dict[str, Any],
        objections: list[str],
        next_step_action: Optional[str],
        due: Optional[str],
        follow_up_reason: Optional[str],
    ) -> str:
        started_at = self._format_started_at(call.started_at) or "дата/время не указаны"
        manager_name = self._clean_text(call.manager_name) or "не указан"
        opening = f"{started_at} менеджер {manager_name} общался с клиентом."

        cleaned_draft = self._clean_text(draft_history_summary)
        if cleaned_draft and self._looks_like_dialogue_dump(cleaned_draft):
            cleaned_draft = None
        if cleaned_draft:
            parts = [opening, self._sentence(cleaned_draft) or ""]
            if next_step_action and "договор" not in cleaned_draft.lower():
                agreement = next_step_action
                if due:
                    agreement = f"{agreement} (срок: {due})"
                parts.append(f"Договорились: {agreement}.")
            return re.sub(r"\s+", " ", " ".join(parts)).strip()

        blocks: list[str] = [opening]
        people = self._nested_dict(structured_fields, "people")
        contacts = self._nested_dict(structured_fields, "contacts")
        student = self._nested_dict(structured_fields, "student")
        interests = self._nested_dict(structured_fields, "interests")

        child_fio = self._clean_text(people.get("child_fio"))
        parent_fio = self._clean_text(people.get("parent_fio"))
        grade = self._clean_text(student.get("grade_current"))
        student_bits: list[str] = []
        if child_fio:
            student_bits.append(f"ребенок: {child_fio}")
        if parent_fio:
            student_bits.append(f"родитель: {parent_fio}")
        if grade:
            student_bits.append(f"класс: {grade}")
        if student_bits:
            blocks.append(f"Уточнили данные: {'; '.join(student_bits)}.")

        topic_parts: list[str] = []
        products = self._clean_list(interests.get("products"))
        formats = self._clean_list(interests.get("format"))
        subjects = self._clean_list(interests.get("subjects"))
        exams = self._clean_list(interests.get("exam_targets"))
        if products:
            topic_parts.append(f"продукты: {', '.join(products)}")
        if formats:
            topic_parts.append(f"формат: {', '.join(formats)}")
        if subjects:
            topic_parts.append(f"предметы: {', '.join(subjects)}")
        if exams:
            topic_parts.append(f"цели: {', '.join(exams)}")
        if topic_parts:
            blocks.append(f"Обсудили: {'; '.join(topic_parts)}.")
        else:
            summary_sentence = self._sentence(summary)
            if summary_sentence and not self._looks_like_dialogue_dump(summary_sentence):
                blocks.append(f"Суть обращения: {summary_sentence}")

        if objections:
            blocks.append(f"Ограничения/возражения: {', '.join(objections)}.")

        if next_step_action:
            agreement = next_step_action
            if due:
                agreement = f"{agreement} (срок: {due})"
            blocks.append(f"Договорились: {agreement}.")
        elif follow_up_reason:
            reason_sentence = self._sentence(follow_up_reason)
            if reason_sentence:
                blocks.append(f"Итог: {reason_sentence}")

        contact_bits: list[str] = []
        email = self._clean_text(contacts.get("email"))
        preferred_channel = self._clean_text(contacts.get("preferred_channel"))
        if email:
            contact_bits.append(f"email: {email}")
        if preferred_channel:
            contact_bits.append(f"канал: {preferred_channel}")
        if contact_bits:
            blocks.append(f"Контакты: {'; '.join(contact_bits)}.")

        compact = re.sub(r"\s+", " ", " ".join(blocks)).strip()
        if len(compact) > 1100:
            compact = compact[:1097].rstrip() + "..."
        return compact

    def _quality_flags_from_call(self, call: CallRecord) -> Dict[str, Any]:
        flags: Dict[str, Any] = {}
        raw = (call.transcript_variants_json or "").strip()
        if raw:
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = {}
            if isinstance(payload, dict):
                mode = payload.get("mode")
                if mode in {"stereo", "mono_or_fallback"}:
                    flags["mode"] = mode
                    flags["mono_fallback"] = mode == "mono_or_fallback"
                secondary_provider = payload.get("secondary_provider")
                if secondary_provider is not None:
                    flags["secondary_provider"] = str(secondary_provider or "") or None
                secondary_backfill_meta = payload.get("secondary_backfill_meta")
                if isinstance(secondary_backfill_meta, dict):
                    flags["secondary_backfill_status"] = str(
                        secondary_backfill_meta.get("status") or ""
                    ).strip() or None
                    flags["secondary_backfill_exhausted"] = bool(
                        secondary_backfill_meta.get("exhausted")
                    )
                warnings = payload.get("warnings")
                if isinstance(warnings, list):
                    clean_warnings = [str(item) for item in warnings if str(item).strip()]
                    flags["warnings_count"] = len(clean_warnings)
                    flags["has_secondary_empty_warning"] = any(
                        "secondary" in item.lower() and "empty" in item.lower()
                        for item in clean_warnings
                    )
        if "mono_fallback" not in flags:
            flags["mono_fallback"] = False
        return flags

    def _analysis_export_paths(self, call: CallRecord) -> Optional[tuple[Path, Path]]:
        export_dir = (self._settings.transcript_export_dir or "").strip()
        if not export_dir:
            return None
        source_path = Path(call.source_file)
        target_dir = Path(export_dir) / source_path.parent.name
        stem = source_path.stem
        return (
            target_dir / f"{stem}_history_summary.txt",
            target_dir / f"{stem}_structured_fields.json",
        )

    def _export_analysis_files(self, call: CallRecord, analysis: Dict[str, Any]) -> None:
        paths = self._analysis_export_paths(call)
        if not paths:
            return
        summary_path, structured_path = paths
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        history_summary = self._clean_text(analysis.get("history_summary"))
        if not history_summary:
            history_summary = self._clean_text(analysis.get("history_short"))
        if not history_summary:
            history_summary = self._clean_text(analysis.get("summary"))
        summary_path.write_text((history_summary or "") + "\n", encoding="utf-8")

        structured_fields = analysis.get("structured_fields")
        if not isinstance(structured_fields, dict):
            structured_fields = self._nested_dict(analysis, "crm_blocks")
        structured_path.write_text(
            json.dumps(structured_fields, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _normalize_analysis(
        self,
        call: CallRecord,
        text: str,
        raw_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        raw = raw_analysis if isinstance(raw_analysis, dict) else {}
        blocks = self._nested_dict(raw, "crm_blocks")
        if not blocks:
            blocks = self._nested_dict(raw, "structured_fields")
        people = self._nested_dict(blocks, "people")
        contacts = self._nested_dict(blocks, "contacts")
        student = self._nested_dict(blocks, "student")
        interests = self._nested_dict(blocks, "interests")
        commercial = self._nested_dict(blocks, "commercial")
        next_step_block = self._nested_dict(blocks, "next_step")

        summary = (
            self._clean_text(raw.get("summary"))
            or self._clean_text(raw.get("history_summary"))
            or self._clean_text(raw.get("history_short"))
        )
        if not summary:
            summary = (text or "").strip()[:600]
        history_short = (
            self._clean_text(raw.get("history_short"))
            or self._clean_text(raw.get("history_summary"))
            or summary
            or ""
        )
        raw_history_summary = self._clean_text(raw.get("history_summary")) or history_short

        target_product = self._clean_text(raw.get("target_product"))
        legacy_interests = self._clean_list(raw.get("interests"))

        products = self._unique(
            self._clean_list(interests.get("products"))
            + ([target_product] if target_product else [])
            + self._detect_from_patterns(text, PRODUCT_PATTERNS)
        )
        formats = self._unique(
            self._clean_list(interests.get("format"))
            + self._detect_from_patterns(text, FORMAT_PATTERNS)
        )
        subjects = self._unique(
            self._clean_list(interests.get("subjects"))
            + self._detect_from_patterns(text, SUBJECT_PATTERNS)
        )
        exam_targets = self._unique(
            self._clean_list(interests.get("exam_targets"))
            + self._detect_from_patterns(text, EXAM_PATTERNS)
        )
        if not target_product and products:
            target_product = products[0]

        grade_current = (
            self._clean_text(student.get("grade_current"))
            or self._clean_text(raw.get("student_grade"))
            or self._extract_grade(text)
        )
        budget = self._clean_text(commercial.get("budget")) or self._clean_text(raw.get("budget"))
        timeline = self._clean_text(raw.get("timeline")) or self._clean_text(next_step_block.get("due"))

        raw_price_sensitivity = self._clean_text(commercial.get("price_sensitivity"))
        if raw_price_sensitivity in {"high", "medium", "low"}:
            price_sensitivity = raw_price_sensitivity
        elif OBJECTION_PATTERNS["цена"].search(text.lower()):
            price_sensitivity = "high"
        else:
            price_sensitivity = None

        raw_discount_interest = commercial.get("discount_interest")
        if isinstance(raw_discount_interest, bool):
            discount_interest = raw_discount_interest
        else:
            discount_interest = bool(re.search(r"скидк\w*|акци\w*|рассрочк\w*", text.lower()))

        objections = self._unique(
            self._clean_list(blocks.get("objections"))
            + self._clean_list(raw.get("objections"))
            + self._detect_from_patterns(text, OBJECTION_PATTERNS)
        )

        next_step_action = self._clean_text(next_step_block.get("action")) or self._clean_text(
            raw.get("next_step")
        )
        if (
            not next_step_action
            and ("перезвон" in text.lower() or "созвон" in text.lower() or "позвон" in text.lower())
        ):
            next_step_action = "Перезвонить клиенту"
        if not next_step_action and "отправ" in text.lower():
            next_step_action = "Отправить материалы"

        score = self._coerce_score(raw.get("follow_up_score"))
        if score is None:
            if self._is_non_conversation(text):
                score = 0
            elif next_step_action:
                score = 70
            elif objections:
                score = 55
            else:
                score = 60

        raw_lead_priority = self._clean_text(blocks.get("lead_priority"))
        if raw_lead_priority in {"hot", "warm", "cold"}:
            lead_priority = raw_lead_priority
        else:
            lead_priority = self._priority_from_score(score)

        tags = self._unique(self._clean_list(raw.get("tags")))
        if self._is_non_conversation(text) and "non_conversation" not in [t.lower() for t in tags]:
            tags.append("non_conversation")
            score = 0
            lead_priority = "cold"
            next_step_action = None
            objections = []

        follow_up_reason = self._clean_text(raw.get("follow_up_reason"))
        if not follow_up_reason:
            if self._is_non_conversation(text):
                follow_up_reason = "Нет содержательного диалога."
            elif next_step_action:
                follow_up_reason = "Есть согласованный следующий шаг."
            else:
                follow_up_reason = "Оценка на основе содержания звонка."

        phone_from_filename = self._clean_text(contacts.get("phone_from_filename")) or self._clean_text(
            call.phone
        )
        email = self._clean_text(contacts.get("email")) or self._extract_email(text)
        preferred_channel = self._clean_text(contacts.get("preferred_channel")) or self._detect_preferred_channel(
            text
        )

        pain_points = self._unique(self._clean_list(raw.get("pain_points")) + objections)
        personal_offer = self._clean_text(raw.get("personal_offer"))
        school = self._clean_text(student.get("school"))
        parent_fio = self._clean_text(people.get("parent_fio"))
        child_fio = self._clean_text(people.get("child_fio"))
        due = self._clean_text(next_step_block.get("due")) or timeline

        legacy_interests_out = self._unique(legacy_interests + products + formats + subjects + exam_targets)
        quality_flags = self._quality_flags_from_call(call)
        raw_quality = raw.get("quality_flags")
        if isinstance(raw_quality, dict):
            quality_flags.update(raw_quality)

        evidence: list[Dict[str, Any]] = []
        raw_evidence = raw.get("evidence")
        if isinstance(raw_evidence, list):
            for item in raw_evidence:
                if not isinstance(item, dict):
                    continue
                text_item = self._clean_text(item.get("text"))
                if not text_item:
                    continue
                evidence.append(
                    {
                        "speaker": self._clean_text(item.get("speaker")),
                        "ts": self._clean_text(item.get("ts")),
                        "text": text_item[:260],
                    }
                )
                if len(evidence) >= 5:
                    break
        if not evidence:
            evidence = self._extract_evidence(text, limit=3)

        structured_fields = {
            "people": {
                "parent_fio": parent_fio,
                "child_fio": child_fio,
            },
            "contacts": {
                "email": email,
                "phone_from_filename": phone_from_filename,
                "preferred_channel": preferred_channel,
            },
            "student": {
                "grade_current": grade_current,
                "school": school,
            },
            "interests": {
                "products": products,
                "format": formats,
                "subjects": subjects,
                "exam_targets": exam_targets,
            },
            "commercial": {
                "price_sensitivity": price_sensitivity,
                "budget": budget,
                "discount_interest": discount_interest,
            },
            "objections": objections,
            "next_step": {
                "action": next_step_action,
                "due": due,
            },
            "lead_priority": lead_priority,
        }
        history_summary = self._compose_history_summary(
            call,
            draft_history_summary=raw_history_summary,
            summary=summary,
            structured_fields=structured_fields,
            objections=objections,
            next_step_action=next_step_action,
            due=due,
            follow_up_reason=follow_up_reason,
        )
        if not history_short or self._looks_like_dialogue_dump(history_short):
            history_short = history_summary

        normalized: Dict[str, Any] = {
            "analysis_schema_version": LATEST_ANALYSIS_SCHEMA_VERSION,
            "history_summary": history_summary,
            "structured_fields": structured_fields,
            "history_short": history_short,
            "crm_blocks": structured_fields,
            "evidence": evidence,
            "quality_flags": quality_flags,
            # Legacy-compatible keys for existing downstream sync.
            "summary": summary,
            "interests": legacy_interests_out,
            "student_grade": grade_current,
            "target_product": target_product,
            "personal_offer": personal_offer,
            "pain_points": pain_points,
            "budget": budget,
            "timeline": timeline,
            "objections": objections,
            "next_step": next_step_action,
            "follow_up_score": score,
            "follow_up_reason": follow_up_reason,
            "tags": tags,
        }
        return normalized

    @staticmethod
    def analysis_schema_version(payload: Dict[str, Any]) -> str:
        raw = payload.get("analysis_schema_version")
        if raw is None:
            return "v1"
        value = str(raw).strip().lower()
        return value or "v1"

    def migrate_analysis_payload(self, call: CallRecord, payload: Dict[str, Any]) -> Dict[str, Any]:
        text = (call.transcript_text or "").strip()
        raw = payload if isinstance(payload, dict) else {}
        return self._normalize_analysis(call, text, raw)

    def _mock_analysis(self, call: CallRecord, text: str) -> Dict[str, Any]:
        _ = call
        lowered = text.lower()
        tags = []
        if "дорого" in lowered or "expensive" in lowered:
            tags.append("price_sensitive")
        if "перезвон" in lowered or "follow" in lowered:
            tags.append("needs_follow_up")
        score = 60 + (15 if "needs_follow_up" in tags else 0)
        return {
            "summary": text[:600],
            "interests": [],
            "student_grade": None,
            "target_product": None,
            "personal_offer": None,
            "pain_points": [],
            "budget": None,
            "timeline": None,
            "objections": [],
            "next_step": "Перезвонить с персональным предложением.",
            "follow_up_score": min(score, 100),
            "follow_up_reason": "MVP mock-анализ по ключевым словам транскрипта.",
            "tags": tags,
        }

    def _is_non_conversation(self, text: str) -> bool:
        raw = (text or "").strip()
        if not raw:
            return True
        lowered = raw.lower()
        if len(raw) <= 40 and not self._looks_like_dialogue_dump(raw):
            return True
        return any(marker in lowered for marker in NON_CONVERSATION_MARKERS)

    def _non_conversation_analysis(self) -> Dict[str, Any]:
        return {
            "summary": "Нецелевой звонок: автоответчик/короткий технический дозвон.",
            "interests": [],
            "student_grade": None,
            "target_product": None,
            "personal_offer": None,
            "pain_points": [],
            "budget": None,
            "timeline": None,
            "objections": [],
            "next_step": None,
            "follow_up_score": 0,
            "follow_up_reason": "Нет содержательного диалога менеджер-клиент для анализа продаж.",
            "tags": ["non_conversation"],
        }

    def _openai_analysis(self, call: CallRecord, text: str) -> Dict[str, Any]:
        client = self._openai_client()
        response = client.chat.completions.create(
            model=self._settings.openai_analysis_model,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": self._analysis_user_prompt(call, text),
                },
            ],
        )
        content = response.choices[0].message.content if response.choices else None
        if not content:
            raise RuntimeError("OpenAI analysis returned empty content")
        data = json.loads(content)
        if not isinstance(data, dict):
            raise RuntimeError("OpenAI analysis must return object JSON")
        return data

    def _ollama_analysis(self, call: CallRecord, text: str) -> Dict[str, Any]:
        client = self._ollama_client()
        payload = client.generate_json(
            model=self._settings.ollama_model,
            think=self._settings.ollama_think,
            temperature=self._settings.ollama_temperature,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=self._analysis_user_prompt(call, text),
            num_predict=max(200, int(self._settings.analyze_ollama_num_predict)),
        )
        if not isinstance(payload, dict):
            raise RuntimeError("Ollama analysis must return object JSON")
        return payload

    def _codex_cli_analysis(self, call: CallRecord, text: str) -> Dict[str, Any]:
        codex_bin = (self._settings.codex_cli_command or "codex").strip() or "codex"
        if shutil.which(codex_bin) is None:
            raise RuntimeError(f"codex binary is not available: {codex_bin}")

        prompt = f"{SYSTEM_PROMPT}\n\n{self._analysis_user_prompt(call, text)}"
        timeout_sec = max(15, int(self._settings.codex_cli_timeout_sec))
        with tempfile.NamedTemporaryFile(prefix="mango_codex_analyze_", suffix=".txt") as out_file:
            cmd = [
                codex_bin,
                "exec",
                "--skip-git-repo-check",
                "--ephemeral",
                "--sandbox",
                "read-only",
                "--model",
                self._settings.codex_merge_model,
                "--output-last-message",
                out_file.name,
            ]
            reasoning_effort = (self._settings.codex_reasoning_effort or "").strip().lower()
            if reasoning_effort in {"low", "medium", "high"}:
                cmd.extend(["-c", f'model_reasoning_effort="{reasoning_effort}"'])
            cmd.append(prompt)
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_sec,
            )
            if proc.returncode != 0:
                stderr_tail = (proc.stderr or "").strip().splitlines()[-1:] or [""]
                raise RuntimeError(f"codex exec failed rc={proc.returncode}: {stderr_tail[0].strip()}")
            raw = Path(out_file.name).read_text(encoding="utf-8", errors="ignore")
        payload = self._extract_json_payload(raw)
        if not isinstance(payload, dict):
            raise RuntimeError("Codex analysis must return object JSON")
        return payload

    def _analyze_text(self, call: CallRecord, text: str) -> Dict[str, Any]:
        if self._is_non_conversation(text):
            return self._non_conversation_analysis()
        provider = self._settings.analyze_provider
        if provider == "mock":
            return self._mock_analysis(call, text)
        if provider == "openai":
            return self._openai_analysis(call, text)
        if provider == "ollama":
            return self._ollama_analysis(call, text)
        if provider == "codex_cli":
            return self._codex_cli_analysis(call, text)
        raise RuntimeError(f"Unsupported ANALYZE_PROVIDER={provider}")

    def run(self, session: Session, limit: int) -> Dict[str, int]:
        now = self._utc_now()
        max_attempts = max(1, self._settings.analyze_max_attempts)
        calls = session.scalars(
            select(CallRecord)
            .where(CallRecord.transcription_status == "done")
            .where(or_(CallRecord.resolve_status.in_(["done", "skipped"]), CallRecord.resolve_status.is_(None)))
            .where(CallRecord.dead_letter_stage.is_(None))
            .where(CallRecord.analysis_status.in_(["pending", "failed"]))
            .where(CallRecord.analyze_attempts < max_attempts)
            .where(or_(CallRecord.next_retry_at.is_(None), CallRecord.next_retry_at <= now))
            .order_by(CallRecord.id.asc())
            .limit(limit)
        ).all()
        success = 0
        failed = 0
        for call in calls:
            call.analyze_attempts = int(call.analyze_attempts or 0) + 1
            attempt = call.analyze_attempts
            try:
                text = (call.transcript_text or "").strip()
                if not text:
                    raise RuntimeError("Empty transcript_text")
                raw_analysis = self._analyze_text(call, text)
                analysis = self._normalize_analysis(call, text, raw_analysis)
                call.analysis_json = json.dumps(analysis, ensure_ascii=False)
                self._export_analysis_files(call, analysis)
                call.analysis_status = "done"
                call.sync_status = "pending"
                call.next_retry_at = None
                call.dead_letter_stage = None
                call.last_error = None
                success += 1
            except Exception as exc:  # noqa: BLE001
                call.last_error = f"analyze: {exc}"
                if attempt >= max_attempts:
                    call.analysis_status = "dead"
                    call.dead_letter_stage = "analyze"
                    call.next_retry_at = None
                else:
                    call.analysis_status = "failed"
                    call.next_retry_at = self._utc_now() + self._retry_delay(attempt)
                failed += 1
            session.add(call)
        session.commit()
        return {"processed": len(calls), "success": success, "failed": failed}
