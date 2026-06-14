from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from openai import OpenAI
from sqlalchemy import or_, select, text
from sqlalchemy.orm import Session

from mango_mvp.clients.ollama import OllamaClient
from mango_mvp.config import Settings
from mango_mvp.models import CallRecord
from mango_mvp.quality.non_conversation import detect_non_conversation_signals
from mango_mvp.services.llm_response_cache import LLMResponseCache
from mango_mvp.utils.filename_repair import repair_manager_name

SYSTEM_PROMPT_FULL = """Strict analyst for Russian EdTech phone calls.
Return a single-line minified JSON object only. No markdown, comments, or extra keys.

Rules:
- Use only transcript + metadata + deterministic hints when supported by transcript facts. Never invent facts from hints; if unsupported, return null or [].
- All text in Russian except emails and phone numbers.
- history_summary: 3-5 factual sentences, dense CRM note, no dialogue dump, no MANAGER/CLIENT prefixes, no date/time or manager preamble.
- For meaningful calls mention: request/topic, what the manager clarified/offered/explained, student data (grade/subject/product) when available, key constraint/objection when available, and the agreed next step when available.
- Do not collapse a meaningful call into one generic sentence.
- For long transcripts or multi-turn MANAGER/CLIENT dialogue, do not classify as non_conversation/voicemail/IVR just because words like "абонент", "секретарь", "коллекторская организация", "перезвонить", or company auto-greeting markers appear. Use non_conversation only when the client side is exclusively a system/IVR/voicemail/no-live message and there is no human response.
- next_step.action must be in Russian.
- target_product must be one of: "годовые курсы", "летний лагерь", "интенсив", "индивидуальные занятия", null.
- Use non_conversation only for unanswered/voicemail/wrong-number/no meaningful human dialogue.
- technical_call, service_call, existing_client_progress are not non_conversation.

Return exactly these keys:
{
  "analysis_schema_version": "v2",
  "history_summary": "",
  "structured_fields": {
    "people": {"parent_fio": null, "child_fio": null},
    "contacts": {"email": null, "phone_from_filename": null, "preferred_channel": null},
    "student": {"grade_current": null, "school": null},
    "interests": {"products": [], "format": [], "subjects": [], "exam_targets": []},
    "commercial": {"price_sensitivity": null, "budget": null, "discount_interest": null},
    "objections": [],
    "next_step": {"action": null, "due": null},
    "lead_priority": null
  },
  "target_product": null,
  "tags": []
}"""

SYSTEM_PROMPT_COMPACT = """Strict analyst for Russian EdTech phone calls.
Return a single-line minified JSON object only. No markdown, comments, or extra keys.

Rules:
- Use only transcript + metadata + deterministic hints. Never invent facts.
- All text in Russian except emails and phone numbers.
- If unsupported, return null or [].
- history_summary: 3-5 factual sentences, dense CRM note, no dialogue dump, no date/time or manager preamble.
- For meaningful calls mention: request/topic, what the manager clarified/offered/explained, student data (grade/subject/product) when available, key constraint/objection when available, and the agreed next step when available.
- Do not collapse a meaningful call into one generic sentence.
- For long transcripts or multi-turn MANAGER/CLIENT dialogue, do not use tag non_conversation just because words like "абонент", "секретарь", "коллекторская организация", "перезвонить", or company auto-greeting markers appear. Use non_conversation only when the client side is exclusively a system/IVR/voicemail/no-live message and there is no human response.
- next_step.action must be in Russian.
- target_product must be one of: "годовые курсы", "летний лагерь", "интенсив", "индивидуальные занятия", null.
- Use tag non_conversation only for unanswered/voicemail/wrong-number/no meaningful human dialogue.
- technical_call, service_call, existing_client_progress are not non_conversation.

Return exactly these keys:
{
  "analysis_schema_version": "v2",
  "history_summary": "",
  "structured_fields": {
    "people": {"parent_fio": null, "child_fio": null},
    "contacts": {"email": null, "phone_from_filename": null, "preferred_channel": null},
    "student": {"grade_current": null, "school": null},
    "interests": {"products": [], "format": [], "subjects": [], "exam_targets": []},
    "commercial": {"price_sensitivity": null, "budget": null, "discount_interest": null},
    "objections": [],
    "next_step": {"action": null, "due": null},
    "lead_priority": null
  },
  "target_product": null,
  "tags": []
}"""

STRONG_NON_CONVERSATION_MARKERS = (
    "продолжение следует",
    "голосовой ассистент",
    "голосовой помощник",
    "я секретарь",
    "на связи я секретарь",
    "ассистент миа",
    "временно попросили отвечать",
    "абонент не может ответить",
    "абонент временно недоступен",
    "абонент занят",
    "вызываемый абонент",
    "номер недоступен",
    "вне зоны действия",
    "телефон выключен",
    "звонок был перенаправлен",
    "оставьте сообщение",
    "после сигнала",
    "отправить бесплатное смс",
    "нажмите 1",
    "целевые финансы",
    "7sky",
    "сервис резерв",
    "актив бизнес консалт",
    "коллекторская организация",
)

WEAK_NON_CONVERSATION_MARKERS = (
    "оставайтесь на линии",
    "дозванивайтесь",
    "дозваниваться",
)

TECHNICAL_CALL_PATTERNS = (
    re.compile(
        r"личн\w* кабинет|не открыва\w*|не работа\w*|ошибк\w*|ссылк\w*|подключ\w*|"
        r"логин|парол\w*|код подтвержден\w*|смс|вебинар|zoom|зум|платформ\w*|"
        r"доступ\w*|тест\b|онлайн[- ]?тест",
        re.I,
    ),
)

SERVICE_CALL_PATTERNS = (
    re.compile(
        r"оплат\w*|счет\w*|чек\w*|договор\w*|расписан\w*|перенос\w*|отмен\w*|возврат\w*|"
        r"заняти\w*|урок\w*|преподавател\w*|куратор\w*|домашн\w*|пробник\w*|срез\w*|"
        r"посещаемост\w*|доступ к урокам|доступ к материалам",
        re.I,
    ),
)

EXISTING_CLIENT_PROGRESS_PATTERNS = (
    re.compile(
        r"обратн\w* связ\w*|как проходит|как вам курс|втор\w* семестр|"
        r"продолж\w* обучен\w*|ранее обучал\w*|уже обуча\w*|результат\w*|"
        r"успеваемост\w*|по текущему курсу|на следующий год",
        re.I,
    ),
)

CALL_TYPE_TAGS = {
    "non_conversation",
    "technical_call",
    "service_call",
    "existing_client_progress",
    "sales_call",
}

LATEST_ANALYSIS_SCHEMA_VERSION = "v2"
ANALYZE_PROMPT_VERSION_COMPACT = "v6"
ANALYZE_PROMPT_VERSION_FULL = "v7"
TRANSCRIPT_QUALITY_GUARDRAILS_VERSION = "non_conversation_v4_live_safeguards"
NON_CONVERSATION_ADVISORY_ENV = "TELEGRAM_NON_CONVERSATION_ADVISORY"
TRUE_ENV_VALUES = {"1", "true", "yes", "y", "on", "да"}

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
    "летний лагерь": re.compile(r"летн(?:ий|его) лаг|летн\w* смен|летн\w* школ|выездн\w* школ", re.I),
    "интенсив": re.compile(r"интенсив\w*", re.I),
    "индивидуальные занятия": re.compile(r"индивидуал\w* занят|репетиторств|индивидуальн\w* формат", re.I),
}
EXAM_PATTERNS = {
    "ЕГЭ": re.compile(r"\bегэ\b", re.I),
    "ОГЭ": re.compile(r"\bогэ\b", re.I),
    "олимпиады": re.compile(r"олимпиад\w*", re.I),
}
OBJECTION_PATTERNS = {
    "цена": re.compile(
        r"\bцен(?:а|е|у|ы|ой|ам|ами|ник\w*)\b|\bстоимост\w*\b|\bдорог\w*\b|\bдешев\w*\b|\bбюджет\w*\b",
        re.I,
    ),
    "время": re.compile(r"нет времени|занят\w*|нагрузк\w*|расписан\w*", re.I),
    "доверие": re.compile(r"кто вы|не слышал\w* о вас|отзыв\w*|гаранти", re.I),
    "неактуально": re.compile(r"не актуальн\w*|не интерес\w*|не нужно", re.I),
}
DIALOGUE_DUMP_LINE_RE = re.compile(r"^\[(?:~)?\d{2}:\d{2}(?:\.\d+)?\]\s*", re.M)
ROLE_PREFIX_RE = re.compile(r"^\s*(manager|client|менеджер|клиент)\s*:\s*", re.I | re.M)

ANALYSIS_PROMPT_TRANSCRIPT_MAX_CHARS_FULL = 10000
ANALYSIS_PROMPT_TRANSCRIPT_HEAD_CHARS_FULL = 7000
ANALYSIS_PROMPT_TRANSCRIPT_TAIL_CHARS_FULL = 2800
ANALYSIS_PROMPT_TRANSCRIPT_MAX_CHARS_COMPACT = 6500
ANALYSIS_PROMPT_TRANSCRIPT_HEAD_CHARS_COMPACT = 4600
ANALYSIS_PROMPT_TRANSCRIPT_TAIL_CHARS_COMPACT = 1600
PROMPT_COMPACTION_FILLER_TOKENS = {
    "ага",
    "алло",
    "да",
    "ладно",
    "понятно",
    "спасибо",
    "угу",
    "хорошо",
    "ясно",
}
PROMPT_COMPACTION_COMMITMENT_TOKENS = {"да", "спасибо"}
PROMPT_COMPACTION_REPEAT_RE = re.compile(
    r"\b(?P<token>ага|алло|да|ладно|понятно|спасибо|угу|хорошо|ясно)\b"
    r"(?:[\s,.;:!?-]+(?P=token)\b)+",
    re.I,
)


def _truthy_env_flag(name: str) -> bool:
    value = os.getenv(name)
    return str(value or "").strip().casefold() in TRUE_ENV_VALUES


class AnalyzeService:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._client: Optional[OpenAI] = None
        self._ollama_client_instance: Optional[OllamaClient] = None
        self._llm_cache = LLMResponseCache(
            enabled=settings.llm_cache_enabled,
            root_dir=settings.llm_cache_dir,
        )

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _analysis_worker_id() -> str:
        return f"an-{uuid.uuid4().hex[:12]}"

    def _analysis_lease_cutoff(self, now: datetime) -> datetime:
        timeout_sec = max(60, int(self._settings.analyze_lease_timeout_sec))
        return now - timedelta(seconds=timeout_sec)

    def _release_stale_claims(self, session: Session, now: datetime) -> int:
        cutoff = self._analysis_lease_cutoff(now)
        result = session.execute(
            text(
                """
                UPDATE call_records
                   SET analysis_status = 'pending',
                       analysis_worker_id = NULL,
                       analysis_claimed_at = NULL,
                       updated_at = :now
                 WHERE analysis_status = 'in_progress'
                   AND (
                        analysis_claimed_at IS NULL
                        OR analysis_claimed_at <= :cutoff
                   )
                """
            ),
            {"now": now, "cutoff": cutoff},
        )
        return int(result.rowcount or 0)

    def _claim_batch(self, session: Session, limit: int, worker_id: str) -> list[int]:
        if limit <= 0:
            return []
        now = self._utc_now()
        max_attempts = max(1, self._settings.analyze_max_attempts)
        self._release_stale_claims(session, now)
        session.execute(
            text(
                """
                UPDATE call_records
                   SET analysis_status = 'in_progress',
                       analysis_worker_id = :worker_id,
                       analysis_claimed_at = :now,
                       updated_at = :now
                 WHERE id IN (
                    SELECT id
                      FROM call_records
                     WHERE transcription_status = 'done'
                       AND (resolve_status IN ('done', 'skipped') OR resolve_status IS NULL)
                       AND dead_letter_stage IS NULL
                       AND analysis_status IN ('pending', 'failed')
                       AND analyze_attempts < :max_attempts
                       AND (next_retry_at IS NULL OR next_retry_at <= :now)
                     ORDER BY id ASC
                     LIMIT :limit
                 )
                """
            ),
            {
                "worker_id": worker_id,
                "now": now,
                "max_attempts": max_attempts,
                "limit": int(limit),
            },
        )
        ids = [
            int(row[0])
            for row in session.execute(
                text(
                    """
                    SELECT id
                      FROM call_records
                     WHERE analysis_status = 'in_progress'
                       AND analysis_worker_id = :worker_id
                     ORDER BY id ASC
                    """
                ),
                {"worker_id": worker_id},
            ).all()
        ]
        session.commit()
        return ids

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
    def _parse_object_candidate(text: str) -> Optional[Dict[str, Any]]:
        raw = (text or "").strip()
        if not raw:
            return None
        try:
            payload = json.loads(raw)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass
        try:
            payload = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _extract_json_payload(text: str) -> Dict[str, Any]:
        raw = (text or "").strip()
        if not raw:
            raise RuntimeError("empty response")
        payload = AnalyzeService._parse_object_candidate(raw)
        if payload is not None:
            return payload

        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL)
        if fence:
            payload = AnalyzeService._parse_object_candidate(fence.group(1))
            if payload is not None:
                return payload

        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            payload = AnalyzeService._parse_object_candidate(raw[start : end + 1])
            if payload is not None:
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
        if ROLE_PREFIX_RE.search(raw):
            return True
        line_count = len([line for line in raw.splitlines() if line.strip()])
        lowered = raw.lower()
        if line_count >= 3 and (
            "клиент:" in lowered
            or "менеджер" in lowered
            or "manager:" in lowered
            or "client:" in lowered
        ):
            return True
        return False

    @staticmethod
    def _normalize_next_step_action(value: Optional[str]) -> Optional[str]:
        text = (value or "").strip()
        if not text:
            return None
        lowered = text.lower()
        mappings = (
            (r"call back|callback|follow[\s-]?up", "Перезвонить клиенту"),
            (r"wait for decision", "Дождаться решения клиента"),
            (r"schedule|arrange", "Согласовать следующий контакт"),
        )
        for pattern, replacement in mappings:
            if re.search(pattern, lowered):
                return replacement
        if re.search(r"ссылк\w*.*оплат|оплат\w*.*ссылк", lowered):
            return "Отправить ссылку на оплату"
        if re.search(r"(telegram|whatsapp|email|e-mail|телеграм|ватсап|вотсап|почт)", lowered) and re.search(
            r"(написа\w*|отправ\w*|вышл\w*|пришл\w*|направ\w*|скин\w*)",
            lowered,
        ):
            return "Отправить материалы"
        return text

    def _analysis_prompt_profile(self, override: Optional[str] = None) -> str:
        profile = (override or self._settings.analyze_prompt_profile or "").strip().lower()
        if profile not in {"compact", "full"}:
            return "compact"
        return profile

    def _analysis_system_prompt(self, profile: Optional[str] = None) -> str:
        normalized = self._analysis_prompt_profile(profile)
        if normalized == "full":
            return SYSTEM_PROMPT_FULL
        return SYSTEM_PROMPT_COMPACT

    def _analysis_prompt_version(self, profile: Optional[str] = None) -> str:
        normalized = self._analysis_prompt_profile(profile)
        if normalized == "full":
            return ANALYZE_PROMPT_VERSION_FULL
        return ANALYZE_PROMPT_VERSION_COMPACT

    def _analysis_prompt_limits(self, profile: Optional[str] = None) -> tuple[int, int, int]:
        normalized = self._analysis_prompt_profile(profile)
        if normalized == "full":
            return (
                ANALYSIS_PROMPT_TRANSCRIPT_MAX_CHARS_FULL,
                ANALYSIS_PROMPT_TRANSCRIPT_HEAD_CHARS_FULL,
                ANALYSIS_PROMPT_TRANSCRIPT_TAIL_CHARS_FULL,
            )
        return (
            ANALYSIS_PROMPT_TRANSCRIPT_MAX_CHARS_COMPACT,
            ANALYSIS_PROMPT_TRANSCRIPT_HEAD_CHARS_COMPACT,
            ANALYSIS_PROMPT_TRANSCRIPT_TAIL_CHARS_COMPACT,
        )

    @staticmethod
    def _compact_prompt_filler_body(text: str) -> str:
        compact = re.sub(r"\s+", " ", text or "").strip()
        if not compact:
            return ""
        previous = None
        while previous != compact:
            previous = compact
            compact = PROMPT_COMPACTION_REPEAT_RE.sub(lambda match: match.group("token"), compact)
            compact = re.sub(r"\s+([,.;:!?])", r"\1", compact)
            compact = re.sub(r"([,.;:!?])(?=[^\s])", r"\1 ", compact)
            compact = re.sub(r"\s+", " ", compact).strip(" ,")
        return compact

    @staticmethod
    def _filler_only_signature(text: str) -> Optional[str]:
        lowered = (text or "").lower()
        tokens = re.findall(r"[a-zа-яё0-9]+", lowered, flags=re.I)
        if not tokens:
            return None
        if not all(token in PROMPT_COMPACTION_FILLER_TOKENS for token in tokens):
            return None
        if len(tokens) == 1:
            return None
        if any(token in PROMPT_COMPACTION_COMMITMENT_TOKENS for token in tokens):
            return None
        return " ".join(tokens)

    @staticmethod
    def _prompt_speaker_label(speaker: str) -> str:
        lowered = (speaker or "").strip().lower()
        if "менедж" in lowered or "manager" in lowered:
            return "Менеджер"
        if "клиент" in lowered or "client" in lowered:
            return "Клиент"
        return "Спикер"

    def _compact_transcript_for_prompt(
        self,
        text: str,
        profile: Optional[str] = None,
        *,
        apply_compaction: Optional[bool] = None,
    ) -> Dict[str, Any]:
        normalized = self._analysis_prompt_profile(profile)
        original = text or ""
        use_compaction = (
            self._settings.analyze_transcript_compaction_enabled
            if apply_compaction is None
            else bool(apply_compaction)
        )
        compacted = original
        shortened_lines = 0
        deduped_lines = 0
        removed_lines = 0
        timestamp_removed_lines = 0

        if use_compaction and original:
            compacted_lines: list[str] = []
            prev_filler_signature: Optional[str] = None
            prev_speaker: Optional[str] = None
            for raw_line in original.splitlines():
                stripped = raw_line.strip()
                if not stripped:
                    if compacted_lines and compacted_lines[-1] != "":
                        compacted_lines.append("")
                    prev_filler_signature = None
                    prev_speaker = None
                    continue

                match = SPEAKER_LINE_RE.match(stripped)
                speaker = None
                body = stripped
                prefix = ""
                if match:
                    speaker = self._prompt_speaker_label(self._clean_text(match.group("speaker")) or "")
                    body = (match.group("text") or "").strip()
                    prefix = f"{speaker}: "
                    timestamp_removed_lines += 1
                compact_body = self._compact_prompt_filler_body(body)
                if compact_body != body:
                    shortened_lines += 1
                filler_signature = self._filler_only_signature(compact_body)
                if (
                    filler_signature
                    and filler_signature == prev_filler_signature
                    and (speaker or "") == (prev_speaker or "")
                ):
                    deduped_lines += 1
                    removed_lines += 1
                    continue
                rendered = f"{prefix}{compact_body}".strip()
                if rendered:
                    compacted_lines.append(rendered)
                prev_filler_signature = filler_signature
                prev_speaker = speaker

            compacted = "\n".join(compacted_lines).strip()
            if not compacted:
                compacted = original

        compacted = re.sub(r"[ \t]+", " ", compacted)
        prompt_transcript = compacted
        max_chars, head_chars, tail_chars = self._analysis_prompt_limits(normalized)
        truncated = False
        if len(prompt_transcript) > max_chars:
            head = prompt_transcript[:head_chars].rstrip()
            tail = prompt_transcript[-tail_chars:].lstrip()
            prompt_transcript = (
                f"{head}\n\n"
                "[... transcript truncated for prompt budget ...]\n\n"
                f"{tail}"
            )
            truncated = True

        chars_original = len(original)
        chars_compacted = len(compacted)
        chars_prompt = len(prompt_transcript)
        return {
            "profile": normalized,
            "transcript": prompt_transcript,
            "transcript_chars_original": chars_original,
            "transcript_chars_compacted": chars_compacted,
            "transcript_chars_prompt": chars_prompt,
            "transcript_chars_saved": max(0, chars_original - chars_prompt),
            "transcript_compacted": bool(use_compaction and chars_compacted < chars_original),
            "transcript_truncated": truncated,
            "transcript_compaction_removed_lines": removed_lines,
            "transcript_compaction_shortened_lines": shortened_lines,
            "transcript_compaction_deduped_lines": deduped_lines,
            "transcript_prompt_timestamps_removed_lines": timestamp_removed_lines,
        }

    @staticmethod
    def _with_analysis_prompt_quality_flags(
        payload: Dict[str, Any],
        *,
        metrics: Dict[str, Any],
        prompt_version: str,
        cache_hit: bool,
    ) -> Dict[str, Any]:
        merged = dict(payload) if isinstance(payload, dict) else {}
        raw_quality = merged.get("quality_flags")
        quality_flags = dict(raw_quality) if isinstance(raw_quality, dict) else {}
        quality_flags.update(
            {
                "analyze_prompt_profile": metrics.get("profile"),
                "analyze_prompt_version": prompt_version,
                "analyze_prompt_compacted": bool(metrics.get("transcript_compacted")),
                "analyze_prompt_truncated": bool(metrics.get("transcript_truncated")),
                "analyze_llm_cache_hit": bool(cache_hit),
                "analyze_transcript_chars_original": int(metrics.get("transcript_chars_original", 0) or 0),
                "analyze_transcript_chars_compacted": int(metrics.get("transcript_chars_compacted", 0) or 0),
                "analyze_transcript_chars_prompt": int(metrics.get("transcript_chars_prompt", 0) or 0),
                "analyze_transcript_chars_saved": int(metrics.get("transcript_chars_saved", 0) or 0),
                "analyze_prompt_compaction_removed_lines": int(
                    metrics.get("transcript_compaction_removed_lines", 0) or 0
                ),
                "analyze_prompt_compaction_shortened_lines": int(
                    metrics.get("transcript_compaction_shortened_lines", 0) or 0
                ),
                "analyze_prompt_compaction_deduped_lines": int(
                    metrics.get("transcript_compaction_deduped_lines", 0) or 0
                ),
                "analyze_prompt_timestamps_removed_lines": int(
                    metrics.get("transcript_prompt_timestamps_removed_lines", 0) or 0
                ),
            }
        )
        merged["quality_flags"] = quality_flags
        return merged

    @classmethod
    def _prune_prompt_payload(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        pruned: Dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(value, dict):
                nested = cls._prune_prompt_payload(value)
                if nested:
                    pruned[key] = nested
                continue
            if isinstance(value, list):
                cleaned_items = []
                for item in value:
                    if isinstance(item, dict):
                        nested = cls._prune_prompt_payload(item)
                        if nested:
                            cleaned_items.append(nested)
                    elif item not in (None, "", False):
                        cleaned_items.append(item)
                if cleaned_items:
                    pruned[key] = cleaned_items
                continue
            if value in (None, "", False):
                continue
            pruned[key] = value
        return pruned

    def _analysis_prompt_context(
        self,
        call: CallRecord,
        text: str,
        profile: Optional[str] = None,
    ) -> Dict[str, Any]:
        normalized = self._analysis_prompt_profile(profile)
        started_at = self._format_started_at(call.started_at) or "unknown"
        manager = self._clean_text(call.manager_name) or "unknown"
        phone = self._clean_text(call.phone) or "unknown"
        direction = self._clean_text(call.direction) or "unknown"
        transcript_meta = self._compact_transcript_for_prompt(text, normalized)
        metadata_payload = {
            "source_filename": call.source_filename,
            "started_at": started_at,
            "manager_name": manager,
            "client_phone": phone,
            "direction": direction,
        }
        prompt = (
            "Analyze this call transcript and return JSON only.\n"
            "Call metadata JSON:\n"
            f"{json.dumps(metadata_payload, ensure_ascii=False, separators=(',', ':'))}\n"
        )
        if normalized in {"compact", "full"}:
            hints_payload = self._prune_prompt_payload(self._analysis_rule_hints(call, text))
            prompt += (
                "\nDeterministic hints JSON (may be incomplete; use only if supported by transcript):\n"
                f"{json.dumps(hints_payload, ensure_ascii=False, separators=(',', ':'))}\n"
            )
        prompt += "\n" f"Transcript:\n{transcript_meta['transcript']}"
        system_prompt = self._analysis_system_prompt(normalized)
        return {
            "profile": normalized,
            "system_prompt": system_prompt,
            "user_prompt": prompt,
            "llm_prompt": f"{system_prompt}\n\n{prompt}",
            "metrics": transcript_meta,
        }

    def _candidate_next_step_action(self, text: str) -> Optional[str]:
        lowered = (text or "").lower()
        signals = detect_non_conversation_signals(transcript_text=text)
        if signals.should_force_non_conversation or (
            signals.strong_no_live_marker and not signals.protected_live_dialogue and signals.score <= 1
        ):
            return None
        if "перезвон" in lowered or "созвон" in lowered or "позвон" in lowered:
            return "Перезвонить клиенту"
        if re.search(r"ссылк\w*.*оплат|оплат\w*.*ссылк", lowered):
            return "Отправить ссылку на оплату"
        if "отправ" in lowered:
            return "Отправить материалы"
        if "уточн" in lowered:
            return "Уточнить информацию и сообщить клиенту"
        return None

    @staticmethod
    def _has_explicit_sales_signal(
        *,
        raw_sales_signal: bool,
        products: Optional[list[str]] = None,
        formats: Optional[list[str]] = None,
        exam_targets: Optional[list[str]] = None,
    ) -> bool:
        return bool(raw_sales_signal or products or formats or exam_targets)

    def _analysis_rule_hints(self, call: CallRecord, text: str) -> Dict[str, Any]:
        hints: Dict[str, Any] = {
            "target_product_candidates": self._detect_from_patterns(text, PRODUCT_PATTERNS),
            "subject_candidates": self._detect_from_patterns(text, SUBJECT_PATTERNS),
            "format_candidates": self._detect_from_patterns(text, FORMAT_PATTERNS),
            "exam_target_candidates": self._detect_from_patterns(text, EXAM_PATTERNS),
            "objection_candidates": self._detect_from_patterns(text, OBJECTION_PATTERNS),
            "grade_candidate": self._extract_grade(text),
            "email_candidate": self._extract_email(text),
            "preferred_channel_candidate": self._detect_preferred_channel(text),
            "next_step_candidate": self._candidate_next_step_action(text),
            "call_type_candidate": self._detect_call_type(text),
            "non_conversation_candidate": self._is_non_conversation(text),
            "phone_from_filename": self._clean_text(call.phone),
        }
        return hints

    def _analysis_user_prompt(self, call: CallRecord, text: str, profile: Optional[str] = None) -> str:
        return str(self._analysis_prompt_context(call, text, profile)["user_prompt"])

    def _analysis_llm_prompt(self, call: CallRecord, text: str, profile: Optional[str] = None) -> str:
        return str(self._analysis_prompt_context(call, text, profile)["llm_prompt"])

    def _analysis_cache_lookup(
        self,
        *,
        provider: str,
        model: str,
        reasoning: str,
        prompt_version: str,
        prompt: str,
    ) -> Optional[Dict[str, Any]]:
        return self._llm_cache.get(
            namespace="analyze",
            provider=provider,
            model=model,
            reasoning=reasoning,
            prompt_version=prompt_version,
            prompt=prompt,
        )

    def _analysis_cache_store(
        self,
        *,
        provider: str,
        model: str,
        reasoning: str,
        prompt_version: str,
        prompt: str,
        response: Dict[str, Any],
    ) -> None:
        self._llm_cache.put(
            namespace="analyze",
            provider=provider,
            model=model,
            reasoning=reasoning,
            prompt_version=prompt_version,
            prompt=prompt,
            response=response,
        )

    def _should_escalate_full_profile(self, text: str, raw_analysis: Dict[str, Any]) -> bool:
        if not self._settings.analyze_escalate_full_on_ambiguity:
            return False
        if self._analysis_prompt_profile() != "compact":
            return False
        raw = raw_analysis if isinstance(raw_analysis, dict) else {}
        summary = self._clean_text(raw.get("history_summary")) or self._clean_text(raw.get("summary"))
        if not summary or self._looks_like_dialogue_dump(summary):
            return True
        blocks = self._nested_dict(raw, "structured_fields") or self._nested_dict(raw, "crm_blocks")
        interests = self._nested_dict(blocks, "interests")
        next_step_block = self._nested_dict(blocks, "next_step")
        llm_subjects = self._clean_list(interests.get("subjects"))
        llm_products = self._clean_list(interests.get("products"))
        llm_formats = self._clean_list(interests.get("format"))
        llm_exam_targets = self._clean_list(interests.get("exam_targets"))
        llm_next_step = self._clean_text(next_step_block.get("action")) or self._clean_text(raw.get("next_step"))
        llm_target_product = self._clean_text(raw.get("target_product"))
        heuristic_subjects = self._detect_from_patterns(text, SUBJECT_PATTERNS)
        heuristic_products = self._detect_from_patterns(text, PRODUCT_PATTERNS)
        heuristic_formats = self._detect_from_patterns(text, FORMAT_PATTERNS)
        heuristic_exam_targets = self._detect_from_patterns(text, EXAM_PATTERNS)
        heuristic_next_step = self._candidate_next_step_action(text)
        tags = [str(item).strip().lower() for item in self._clean_list(raw.get("tags"))]
        heuristic_call_type = self._detect_call_type(
            text,
            products=heuristic_products,
            subjects=heuristic_subjects,
            formats=heuristic_formats,
            exam_targets=heuristic_exam_targets,
            next_step_action=heuristic_next_step,
        )
        llm_call_type = self._detect_call_type(
            text,
            tags=tags,
            products=llm_products + ([llm_target_product] if llm_target_product else []),
            subjects=llm_subjects,
            formats=llm_formats,
            exam_targets=llm_exam_targets,
            next_step_action=llm_next_step,
        )
        if heuristic_products and not (llm_target_product or llm_products):
            return True
        if heuristic_subjects and not llm_subjects:
            return True
        if heuristic_next_step and not llm_next_step:
            return True
        if "non_conversation" in tags and heuristic_call_type != "non_conversation":
            return True
        if llm_call_type == "non_conversation" and heuristic_call_type != "non_conversation":
            return True
        return False

    def _has_meaningful_sales_signal(self, text: str) -> bool:
        raw = (text or "").strip()
        if not raw:
            return False
        lowered = raw.lower()
        products = self._detect_from_patterns(raw, PRODUCT_PATTERNS)
        subjects = self._detect_from_patterns(raw, SUBJECT_PATTERNS)
        formats = self._detect_from_patterns(raw, FORMAT_PATTERNS)
        exam_targets = self._detect_from_patterns(raw, EXAM_PATTERNS)
        lead_interest = bool(
            re.search(
                r"интерес\w*|хотел\w*|хочу|ищ\w*|рассматрива\w*|подобрат\w*|"
                r"запис\w*|узнат\w*|подход\w*|выбрат\w*",
                lowered,
            )
        )
        needs_training = bool(
            re.search(
                r"нуж(?:ен|на|ны)\s+(?:курс\w*|лагер\w*|школ\w*|обучен\w*|заняти\w*|подготов\w*|программ\w*)",
                lowered,
            )
        )
        training_noun = bool(
            re.search(r"курс\w*|лагер\w*|школ\w*|обучен\w*|заняти\w*|подготов\w*|программ\w*", lowered)
        )
        existing_client_context = self._matches_any_pattern(raw, TECHNICAL_CALL_PATTERNS) or self._matches_any_pattern(
            raw, SERVICE_CALL_PATTERNS
        ) or self._matches_any_pattern(raw, EXISTING_CLIENT_PROGRESS_PATTERNS)
        if products:
            return True
        if existing_client_context and not lead_interest and not needs_training:
            return False
        if (lead_interest or needs_training) and (subjects or formats or exam_targets or training_noun):
            return True
        if subjects and formats and training_noun:
            return True
        if exam_targets and (subjects or training_noun or lead_interest):
            return True
        if subjects and self._extract_grade(raw) and (lead_interest or needs_training):
            return True
        return False

    @staticmethod
    def _matches_any_pattern(text: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
        raw = text or ""
        return any(pattern.search(raw) for pattern in patterns)

    def _has_substantial_dialogue(self, text: str) -> bool:
        raw = (text or "").strip()
        if not raw:
            return False
        word_count = len(re.findall(r"\w+", raw, re.U))
        speaker_markers = len(re.findall(r"^\s*(manager|client|менеджер|клиент)\s*:", raw, re.I | re.M))
        line_count = len([line for line in raw.splitlines() if line.strip()])
        return word_count >= 18 and (speaker_markers >= 1 or line_count >= 3)

    def _detect_call_type(
        self,
        text: str,
        *,
        tags: Optional[list[str]] = None,
        products: Optional[list[str]] = None,
        subjects: Optional[list[str]] = None,
        formats: Optional[list[str]] = None,
        exam_targets: Optional[list[str]] = None,
        next_step_action: Optional[str] = None,
    ) -> str:
        raw = (text or "").strip()
        if not raw:
            return "non_conversation"
        signals = detect_non_conversation_signals(transcript_text=raw)
        if signals.should_force_non_conversation:
            return "non_conversation"
        if signals.label == "manual_review_borderline_live_context" and any(
            str(reason).startswith("safeguard_") for reason in signals.reason_codes
        ):
            return "service_call"
        lowered = raw.lower()
        semantic_tags = {str(item).strip().lower() for item in (tags or []) if str(item).strip()}
        raw_sales_signal = self._has_meaningful_sales_signal(raw)
        explicit_sales_signal = self._has_explicit_sales_signal(
            raw_sales_signal=raw_sales_signal,
            products=products,
            formats=formats,
            exam_targets=exam_targets,
        )
        meaningful_dialogue = self._has_substantial_dialogue(raw)
        technical_signal = self._matches_any_pattern(raw, TECHNICAL_CALL_PATTERNS)
        service_signal = self._matches_any_pattern(raw, SERVICE_CALL_PATTERNS)
        progress_signal = self._matches_any_pattern(raw, EXISTING_CLIENT_PROGRESS_PATTERNS)
        has_followup = bool(self._clean_text(next_step_action))
        has_business_content = explicit_sales_signal or technical_signal or service_signal or progress_signal or (
            has_followup and meaningful_dialogue
        )

        if any(marker in lowered for marker in STRONG_NON_CONVERSATION_MARKERS) and not has_business_content:
            return "non_conversation"
        if len(raw) <= 40 and not meaningful_dialogue and not has_business_content:
            return "non_conversation"

        if "existing_client_progress" in semantic_tags and not explicit_sales_signal:
            return "existing_client_progress"
        if "technical_call" in semantic_tags and not explicit_sales_signal:
            return "technical_call"
        if "service_call" in semantic_tags and not explicit_sales_signal:
            return "service_call"

        if progress_signal and not explicit_sales_signal:
            return "existing_client_progress"
        if technical_signal and not explicit_sales_signal:
            return "technical_call"
        if service_signal and not explicit_sales_signal:
            return "service_call"

        if "non_conversation" in semantic_tags and not raw_sales_signal and not meaningful_dialogue:
            return "non_conversation"
        if explicit_sales_signal:
            return "sales_call"
        if meaningful_dialogue:
            if technical_signal:
                return "technical_call"
            if progress_signal:
                return "existing_client_progress"
            return "service_call"
        if any(marker in lowered for marker in WEAK_NON_CONVERSATION_MARKERS):
            return "non_conversation"
        if len(re.findall(r"\w+", raw, re.U)) < 12:
            return "non_conversation"
        return "service_call"

    def _non_conversation_advisory_call_type(
        self,
        text: str,
        *,
        tags: list[str],
        products: list[str],
        formats: list[str],
        exam_targets: list[str],
        next_step_action: Optional[str],
    ) -> str:
        lowered_tags = {str(item).strip().lower() for item in tags if str(item).strip()}
        if "existing_client_progress" in lowered_tags or self._matches_any_pattern(text, EXISTING_CLIENT_PROGRESS_PATTERNS):
            return "existing_client_progress"
        if "technical_call" in lowered_tags or self._matches_any_pattern(text, TECHNICAL_CALL_PATTERNS):
            return "technical_call"
        if "service_call" in lowered_tags or self._matches_any_pattern(text, SERVICE_CALL_PATTERNS):
            return "service_call"
        if self._has_explicit_sales_signal(
            raw_sales_signal=self._has_meaningful_sales_signal(text),
            products=products,
            formats=formats,
            exam_targets=exam_targets,
        ):
            return "sales_call"
        if self._clean_text(next_step_action) or self._has_substantial_dialogue(text):
            return "service_call"
        return "non_conversation"

    def _build_review_flags(
        self,
        call: CallRecord,
        *,
        text: str,
        call_type: str,
        products: list[str],
        formats: list[str],
        exam_targets: list[str],
        target_product: Optional[str],
        next_step_action: Optional[str],
        history_summary: Optional[str],
    ) -> Dict[str, Any]:
        reasons: list[str] = []
        product_present = bool(products or self._clean_text(target_product))
        summary_lower = (history_summary or "").lower()
        technical_signal = self._matches_any_pattern(text, TECHNICAL_CALL_PATTERNS)
        service_signal = self._matches_any_pattern(text, SERVICE_CALL_PATTERNS)
        progress_signal = self._matches_any_pattern(text, EXISTING_CLIENT_PROGRESS_PATTERNS)
        explicit_sales_signal = self._has_explicit_sales_signal(
            raw_sales_signal=self._has_meaningful_sales_signal(text),
            products=products,
            formats=formats,
            exam_targets=exam_targets,
        )

        if call_type == "sales_call":
            if not product_present and not next_step_action:
                reasons.append("sales_missing_product_and_next_step")
            elif not product_present:
                reasons.append("sales_missing_product")
            elif not next_step_action:
                reasons.append("sales_missing_next_step")
            if (technical_signal or service_signal or progress_signal) and not product_present:
                reasons.append("sales_service_overlap")

        if call_type == "non_conversation" and float(call.duration_sec or 0.0) >= 30:
            reasons.append("long_non_conversation")

        if call_type != "non_conversation" and (
            "нецелевой звонок" in summary_lower or "автоответчик/короткий технический дозвон" in summary_lower
        ):
            reasons.append("legacy_summary_conflict")

        if (
            call_type in {"service_call", "technical_call", "existing_client_progress"}
            and explicit_sales_signal
            and not next_step_action
        ):
            reasons.append("non_sales_with_sales_signal")

        return {
            "needs_review": bool(reasons),
            "review_reasons": self._unique(reasons),
        }

    def _transcript_quality_guardrails(
        self,
        call: CallRecord,
        *,
        text: str,
        history_summary: Optional[str],
        call_type: str,
        products: list[str],
        subjects: list[str],
        objections: list[str],
        next_step_action: Optional[str],
    ) -> Dict[str, Any]:
        signals = detect_non_conversation_signals(
            transcript_text=text,
            history_summary=history_summary or "",
            call_type=call_type,
            next_step=next_step_action or "",
            products=products,
            subjects=subjects,
            objections=objections,
            duration_sec=getattr(call, "duration_sec", None),
        )
        return {
            "version": TRANSCRIPT_QUALITY_GUARDRAILS_VERSION,
            "mode": "dry_run",
            "label": signals.label,
            "score": signals.score,
            "reason_codes": list(signals.reason_codes),
            "strong_no_live_marker": signals.strong_no_live_marker,
            "asr_artifact_marker": signals.asr_artifact_marker,
            "system_no_dialogue_phrase": signals.system_no_dialogue_phrase,
            "risky_keyword_marker": signals.risky_keyword_marker,
            "outbound_voicemail_marker": signals.outbound_voicemail_marker,
            "protected_live_dialogue": signals.protected_live_dialogue,
            "should_force_non_conversation": signals.should_force_non_conversation,
            "requires_manual_review": signals.requires_manual_review,
            "recommended_call_type": signals.recommended_call_type,
            "recommended_contentful": signals.recommended_contentful,
            "recommended_contact_subtype": signals.recommended_contact_subtype,
            "manager_chars": signals.manager_chars,
            "client_chars": signals.client_chars,
            "transcript_chars": signals.transcript_chars,
        }

    def _non_conversation_summary(self, call: CallRecord, *, contact_subtype: Optional[str] = None) -> str:
        started_at = self._format_started_at(call.started_at) or "дата/время не указаны"
        manager_name = self._clean_text(repair_manager_name(call.manager_name)) or "не указан"
        reason = "автоответчик, IVR, голосовой ассистент или технический недозвон"
        if contact_subtype == "outbound_voicemail":
            reason = "менеджер оставил сообщение на автоответчике, живого диалога с клиентом не было"
        return (
            f"{started_at} менеджер {manager_name} пытался связаться с клиентом. "
            f"Содержательного диалога не было: {reason}."
        )

    def _apply_non_conversation_hard_validation(
        self,
        call: CallRecord,
        normalized: Dict[str, Any],
    ) -> Dict[str, Any]:
        quality_flags = normalized.get("quality_flags")
        if not isinstance(quality_flags, dict):
            quality_flags = {}
        if quality_flags.get("non_conversation_advisory"):
            return normalized
        if quality_flags.get("call_type") != "non_conversation":
            return normalized

        existing_structured = (
            normalized.get("structured_fields") if isinstance(normalized.get("structured_fields"), dict) else {}
        )
        existing_contacts = (
            existing_structured.get("contacts") if isinstance(existing_structured.get("contacts"), dict) else {}
        )
        phone_from_filename = self._clean_text(existing_contacts.get("phone_from_filename")) or self._clean_text(
            call.phone
        )

        contact_subtype = self._clean_text(quality_flags.get("transcript_quality_recommended_contact_subtype"))
        summary = self._non_conversation_summary(call, contact_subtype=contact_subtype)
        structured_fields = {
            "people": {
                "parent_fio": None,
                "child_fio": None,
            },
            "contacts": {
                "email": None,
                "phone_from_filename": phone_from_filename,
                "preferred_channel": None,
            },
            "student": {
                "grade_current": None,
                "school": None,
            },
            "interests": {
                "products": [],
                "format": [],
                "subjects": [],
                "exam_targets": [],
            },
            "commercial": {
                "price_sensitivity": None,
                "budget": None,
                "discount_interest": None,
            },
            "objections": [],
            "next_step": {
                "action": None,
                "due": None,
            },
            "lead_priority": "cold",
        }

        quality_flags["call_type"] = "non_conversation"
        quality_flags["non_conversation_hard_validation_applied"] = True
        normalized.update(
            {
                "history_summary": summary,
                "structured_fields": structured_fields,
                "history_short": summary,
                "crm_blocks": structured_fields,
                "summary": "Нет содержательного диалога менеджер-клиент для анализа продаж.",
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
                "quality_flags": quality_flags,
            }
        )
        return normalized

    def _clean_history_summary_draft(self, call: CallRecord, draft: str) -> str:
        cleaned = self._clean_text(draft) or ""
        if not cleaned:
            return ""
        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        started_at = self._format_started_at(call.started_at) or ""
        started_at_alt = ""
        if call.started_at is not None:
            started_at_alt = call.started_at.strftime("%d.%m.%Y в %H:%M")
        manager_name = self._clean_text(call.manager_name) or ""
        non_empty_sentences = [self._clean_text(sentence) for sentence in sentences if self._clean_text(sentence)]
        pruned: list[str] = []
        skipping_context = len(non_empty_sentences) > 1
        for sentence in non_empty_sentences:
            compact = self._clean_text(sentence)
            if not compact:
                continue
            lowered = compact.lower()
            has_context = bool(
                (started_at and started_at in compact)
                or (started_at_alt and started_at_alt in compact)
            ) or bool(
                manager_name and manager_name.lower() in lowered
            )
            if skipping_context and has_context:
                if started_at_alt:
                    compact = re.sub(rf"^{re.escape(started_at_alt)}\s*", "", compact, flags=re.I)
                if started_at:
                    compact = re.sub(rf"^{re.escape(started_at)}\s*", "", compact, flags=re.I)
                if manager_name:
                    compact = re.sub(
                        rf"^(?:менеджер\s+)?{re.escape(manager_name)}[\s,:-]*",
                        "",
                        compact,
                        flags=re.I,
                    )
                compact = compact.lstrip(" ,.-")
                lowered = compact.lower()
                if not compact or re.fullmatch(r"(?:менеджер\s+)?общал\w*\s+с\s+клиентом\.?", lowered):
                    continue
            skipping_context = False
            pruned.append(compact)
        return re.sub(r"\s+", " ", " ".join(pruned)).strip()

    @staticmethod
    def _summary_mentions_any(text: str, values: list[Optional[str]]) -> bool:
        lowered = (text or "").lower()
        if not lowered:
            return False
        for value in values:
            cleaned = (value or "").strip().lower()
            if cleaned and cleaned in lowered:
                return True
        return False

    @staticmethod
    def _is_empty_budget_value(value: Optional[str]) -> bool:
        lowered = (value or "").strip().lower()
        return not lowered or lowered in {"не указан", "не указано", "нет", "none", "null", "-"}

    def _build_commercial_lines(self, structured_fields: Dict[str, Any]) -> list[str]:
        commercial = self._nested_dict(structured_fields, "commercial")
        price_sensitivity = self._clean_text(commercial.get("price_sensitivity"))
        budget = self._clean_text(commercial.get("budget"))
        discount_interest = commercial.get("discount_interest")
        price_labels = {
            "high": "высокая",
            "medium": "средняя",
            "low": "низкая",
        }
        bits: list[str] = []
        if price_sensitivity in price_labels:
            bits.append(f"чувствительность к цене: {price_labels[price_sensitivity]}")
        if not self._is_empty_budget_value(budget):
            bits.append(f"бюджет: {budget}")
        if discount_interest is True:
            bits.append("интересуется скидками")
        if not bits:
            return []
        return [f"Коммерческий контекст: {'; '.join(bits)}."]

    def _build_school_line(self, structured_fields: Dict[str, Any]) -> Optional[str]:
        student = self._nested_dict(structured_fields, "student")
        school = self._clean_text(student.get("school"))
        if not school:
            return None
        return f"Школа: {school}."

    def _build_lead_priority_line(self, structured_fields: Dict[str, Any]) -> Optional[str]:
        priority = self._clean_text(structured_fields.get("lead_priority"))
        labels = {
            "hot": "горячий",
            "warm": "теплый",
        }
        if priority not in labels:
            return None
        return f"Приоритет лида: {labels[priority]}."

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
        manager_name = self._clean_text(repair_manager_name(call.manager_name)) or "не указан"
        opening = f"{started_at} менеджер {manager_name} общался с клиентом."

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

        products = self._clean_list(interests.get("products"))
        formats = self._clean_list(interests.get("format"))
        subjects = self._clean_list(interests.get("subjects"))
        exams = self._clean_list(interests.get("exam_targets"))
        school_line = self._build_school_line(structured_fields)
        commercial_lines = self._build_commercial_lines(structured_fields)
        lead_priority_line = self._build_lead_priority_line(structured_fields)
        topic_parts: list[str] = []
        if products:
            topic_parts.append(f"продукты: {', '.join(products)}")
        if formats:
            topic_parts.append(f"формат: {', '.join(formats)}")
        if subjects:
            topic_parts.append(f"предметы: {', '.join(subjects)}")
        if exams:
            topic_parts.append(f"цели: {', '.join(exams)}")

        email = self._clean_text(contacts.get("email"))
        preferred_channel = self._clean_text(contacts.get("preferred_channel"))
        contact_bits: list[str] = []
        if email:
            contact_bits.append(f"email: {email}")
        if preferred_channel:
            contact_bits.append(f"канал: {preferred_channel}")

        cleaned_draft = self._clean_history_summary_draft(call, draft_history_summary or "")
        if cleaned_draft and self._looks_like_dialogue_dump(cleaned_draft):
            cleaned_draft = None
        if cleaned_draft:
            compact_draft = re.sub(r"\s+", " ", cleaned_draft).strip()
            summary_sentence = self._sentence(summary)
            draft_sentences = [item for item in re.split(r"(?<=[.!?])\s+", compact_draft) if item.strip()]
            draft_is_sparse = len(compact_draft) < 180 or len(draft_sentences) < 2
            parts = [opening]
            sentence = self._sentence(compact_draft)
            if sentence:
                parts.append(sentence)
            if (
                summary_sentence
                and not self._looks_like_dialogue_dump(summary_sentence)
                and summary_sentence.lower() not in compact_draft.lower()
                and compact_draft.lower() not in summary_sentence.lower()
                and draft_is_sparse
            ):
                parts.append(f"Суть обращения: {summary_sentence}")
            if student_bits and not self._summary_mentions_any(
                compact_draft,
                [child_fio, parent_fio, grade],
            ):
                parts.append(f"Уточнили данные: {'; '.join(student_bits)}.")
            if topic_parts and not self._summary_mentions_any(
                compact_draft,
                products + formats + subjects + exams,
            ):
                parts.append(f"Обсудили: {'; '.join(topic_parts)}.")
            elif not topic_parts:
                if (
                    summary_sentence
                    and not self._looks_like_dialogue_dump(summary_sentence)
                    and not self._summary_mentions_any(compact_draft, [summary_sentence])
                ):
                    parts.append(f"Суть обращения: {summary_sentence}")
            if objections and not self._summary_mentions_any(compact_draft, objections):
                parts.append(f"Ограничения/возражения: {', '.join(objections)}.")
            for extra_line in [school_line, *commercial_lines, lead_priority_line]:
                if not extra_line:
                    continue
                current_text = " ".join(parts)
                if not self._summary_mentions_any(current_text, [extra_line]):
                    parts.append(extra_line)
            if next_step_action and not any(
                token in compact_draft.lower()
                for token in ("договор", "следующ", "перезвон", "отправ", "созвон", "соедин")
            ):
                agreement = next_step_action
                if due:
                    agreement = f"{agreement} (срок: {due})"
                parts.append(f"Договорились: {agreement}.")
            elif follow_up_reason and not self._summary_mentions_any(compact_draft, [follow_up_reason]):
                reason_sentence = self._sentence(follow_up_reason)
                if reason_sentence:
                    parts.append(f"Итог: {reason_sentence}")
            if contact_bits and not self._summary_mentions_any(compact_draft, [email, preferred_channel]):
                parts.append(f"Контакты: {'; '.join(contact_bits)}.")
            compact = re.sub(r"\s+", " ", " ".join(parts)).strip()
            if len(compact) > 32000:
                compact = compact[:31974].rstrip() + " [обрезано по лимиту поля]"
            return compact

        blocks: list[str] = [opening]
        if student_bits:
            blocks.append(f"Уточнили данные: {'; '.join(student_bits)}.")

        if topic_parts:
            blocks.append(f"Обсудили: {'; '.join(topic_parts)}.")
        else:
            summary_sentence = self._sentence(summary)
            if summary_sentence and not self._looks_like_dialogue_dump(summary_sentence):
                blocks.append(f"Суть обращения: {summary_sentence}")

        if objections:
            blocks.append(f"Ограничения/возражения: {', '.join(objections)}.")

        for extra_line in [school_line, *commercial_lines, lead_priority_line]:
            if extra_line:
                blocks.append(extra_line)

        if next_step_action:
            agreement = next_step_action
            if due:
                agreement = f"{agreement} (срок: {due})"
            blocks.append(f"Договорились: {agreement}.")
        elif follow_up_reason:
            reason_sentence = self._sentence(follow_up_reason)
            if reason_sentence:
                blocks.append(f"Итог: {reason_sentence}")

        if contact_bits:
            blocks.append(f"Контакты: {'; '.join(contact_bits)}.")

        compact = re.sub(r"\s+", " ", " ".join(blocks)).strip()
        if len(compact) > 32000:
            compact = compact[:31974].rstrip() + " [обрезано по лимиту поля]"
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
        llm_sales_signal_sources: list[str] = []
        if self._clean_list(interests.get("products")):
            llm_sales_signal_sources.append("interests.products")
        if self._clean_text(raw.get("target_product")):
            llm_sales_signal_sources.append("target_product")
        if self._clean_text(next_step_block.get("action")) or (
            not isinstance(raw.get("next_step"), dict) and self._clean_text(raw.get("next_step"))
        ):
            llm_sales_signal_sources.append("next_step.action")
        if self._clean_list(blocks.get("objections")) or self._clean_list(raw.get("objections")):
            llm_sales_signal_sources.append("objections")

        summary = (
            self._clean_text(raw.get("summary"))
            or self._clean_text(raw.get("history_summary"))
            or self._clean_text(raw.get("history_short"))
        )
        if summary and self._looks_like_dialogue_dump(summary):
            summary = None
        if not summary:
            transcript_fallback = (text or "").strip()[:600]
            if not self._looks_like_dialogue_dump(transcript_fallback):
                summary = transcript_fallback
        history_short = (
            self._clean_text(raw.get("history_short"))
            or self._clean_text(raw.get("history_summary"))
            or summary
            or ""
        )
        if history_short and self._looks_like_dialogue_dump(history_short):
            history_short = None
        raw_history_summary = self._clean_text(raw.get("history_summary")) or history_short
        if raw_history_summary and self._looks_like_dialogue_dump(raw_history_summary):
            raw_history_summary = None

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
        if target_product and target_product not in PRODUCT_PATTERNS:
            target_product = None

        grade_current = (
            self._clean_text(student.get("grade_current"))
            or self._clean_text(raw.get("student_grade"))
            or self._extract_grade(text)
        )
        school = self._clean_text(student.get("school"))
        parent_fio = self._clean_text(people.get("parent_fio"))
        child_fio = self._clean_text(people.get("child_fio"))
        budget = self._clean_text(commercial.get("budget")) or self._clean_text(raw.get("budget"))
        timeline = self._clean_text(raw.get("timeline")) or self._clean_text(next_step_block.get("due"))
        due = self._clean_text(next_step_block.get("due")) or timeline
        phone_from_filename = self._clean_text(contacts.get("phone_from_filename")) or self._clean_text(call.phone)
        email = self._clean_text(contacts.get("email")) or self._extract_email(text)
        preferred_channel = self._clean_text(contacts.get("preferred_channel")) or self._detect_preferred_channel(text)
        pain_points = self._unique(self._clean_list(raw.get("pain_points")))
        personal_offer = self._clean_text(raw.get("personal_offer"))

        price_signal = bool(OBJECTION_PATTERNS["цена"].search(text.lower()))
        raw_price_sensitivity = self._clean_text(commercial.get("price_sensitivity"))
        if raw_price_sensitivity in {"high", "medium", "low"}:
            price_sensitivity = raw_price_sensitivity
        elif price_signal:
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
        if not price_signal:
            objections = [
                item
                for item in objections
                if not any(token in item.lower() for token in ("цен", "стоим", "дорог", "бюджет"))
            ]
            if price_sensitivity == "high":
                price_sensitivity = None

        next_step_action = self._clean_text(next_step_block.get("action")) or self._clean_text(
            raw.get("next_step")
        )
        next_step_action = self._normalize_next_step_action(next_step_action)
        next_step_signals = detect_non_conversation_signals(
            transcript_text=text,
            duration_sec=getattr(call, "duration_sec", None),
        )
        non_conversation_advisory_enabled = _truthy_env_flag(NON_CONVERSATION_ADVISORY_ENV)
        pre_llm_non_conversation_advisory = (
            non_conversation_advisory_enabled and next_step_signals.should_force_non_conversation
        )
        allow_heuristic_next_step = not (
            next_step_signals.should_force_non_conversation
            or (
                next_step_signals.strong_no_live_marker
                and not next_step_signals.protected_live_dialogue
                and next_step_signals.score <= 1
            )
        )
        if (
            not next_step_action
            and allow_heuristic_next_step
            and ("перезвон" in text.lower() or "созвон" in text.lower() or "позвон" in text.lower())
        ):
            next_step_action = "Перезвонить клиенту"
        if not next_step_action and allow_heuristic_next_step and "отправ" in text.lower():
            next_step_action = "Отправить материалы"

        score = self._coerce_score(raw.get("follow_up_score"))
        if score is None:
            if self._is_non_conversation(text) and not non_conversation_advisory_enabled:
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
        detected_call_type = self._detect_call_type(
            text,
            tags=tags,
            products=products,
            subjects=subjects,
            formats=formats,
            exam_targets=exam_targets,
            next_step_action=next_step_action,
        )
        call_type = detected_call_type
        non_conversation_advisory_sources: list[str] = []
        if non_conversation_advisory_enabled and detected_call_type == "non_conversation":
            if pre_llm_non_conversation_advisory:
                non_conversation_advisory_sources.append("pre_llm_guardrail")
            non_conversation_advisory_sources.append("post_llm_detector")
            call_type = self._non_conversation_advisory_call_type(
                text,
                tags=tags,
                products=products,
                formats=formats,
                exam_targets=exam_targets,
                next_step_action=next_step_action,
            )
        tags = [item for item in tags if item.lower() not in CALL_TYPE_TAGS]
        non_conversation_soft_warning_sources = (
            self._unique(llm_sales_signal_sources) if detected_call_type == "non_conversation" else []
        )
        if detected_call_type == "non_conversation" and not non_conversation_advisory_enabled:
            tags.append("non_conversation")
            products = []
            formats = []
            subjects = []
            exam_targets = []
            target_product = None
            grade_current = None
            school = None
            parent_fio = None
            child_fio = None
            email = None
            preferred_channel = None
            budget = None
            timeline = None
            price_sensitivity = None
            discount_interest = None
            objections = []
            next_step_action = None
            due = None
            pain_points = []
            personal_offer = None
            score = 0
            lead_priority = "cold"
        elif call_type != "sales_call":
            tags.append(call_type)
        if non_conversation_advisory_sources:
            tags.append("non_conversation_advisory")

        follow_up_reason = self._clean_text(raw.get("follow_up_reason"))
        if not follow_up_reason:
            if call_type == "non_conversation":
                follow_up_reason = "Нет содержательного диалога."
            elif next_step_action:
                follow_up_reason = "Есть согласованный следующий шаг."
            else:
                follow_up_reason = "Оценка на основе содержания звонка."
        pain_points = self._unique(pain_points + objections)

        legacy_interests_out = self._unique(legacy_interests + products + formats + subjects + exam_targets)
        quality_flags = self._quality_flags_from_call(call)
        raw_quality = raw.get("quality_flags")
        if isinstance(raw_quality, dict):
            quality_flags.update(raw_quality)
        quality_flags["call_type"] = call_type

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

        transcript_quality_guardrails = self._transcript_quality_guardrails(
            call,
            text=text,
            history_summary=history_summary,
            call_type=call_type,
            products=products,
            subjects=subjects,
            objections=objections,
            next_step_action=next_step_action,
        )
        quality_flags["transcript_quality_guardrails"] = transcript_quality_guardrails
        quality_flags["transcript_quality_guardrails_version"] = transcript_quality_guardrails["version"]
        quality_flags["transcript_quality_guardrails_mode"] = transcript_quality_guardrails["mode"]
        quality_flags["transcript_quality_label"] = transcript_quality_guardrails["label"]
        quality_flags["transcript_quality_score"] = transcript_quality_guardrails["score"]
        quality_flags["transcript_quality_reason_codes"] = transcript_quality_guardrails["reason_codes"]
        quality_flags["transcript_quality_should_force_non_conversation"] = transcript_quality_guardrails[
            "should_force_non_conversation"
        ]
        quality_flags["transcript_quality_requires_manual_review"] = transcript_quality_guardrails[
            "requires_manual_review"
        ]
        quality_flags["transcript_quality_protected_live_dialogue"] = transcript_quality_guardrails[
            "protected_live_dialogue"
        ]
        quality_flags["transcript_quality_recommended_call_type"] = transcript_quality_guardrails[
            "recommended_call_type"
        ]
        quality_flags["transcript_quality_recommended_contact_subtype"] = transcript_quality_guardrails[
            "recommended_contact_subtype"
        ]

        review_flags = self._build_review_flags(
            call,
            text=text,
            call_type=call_type,
            products=products,
            formats=formats,
            exam_targets=exam_targets,
            target_product=target_product,
            next_step_action=next_step_action,
            history_summary=history_summary,
        )
        review_reasons = list(review_flags["review_reasons"])
        needs_review = bool(review_flags["needs_review"])
        if non_conversation_advisory_sources:
            quality_flags["non_conversation_advisory"] = True
            quality_flags["non_conversation_advisory_env"] = NON_CONVERSATION_ADVISORY_ENV
            quality_flags["non_conversation_advisory_sources"] = self._unique(non_conversation_advisory_sources)
            quality_flags["non_conversation_advisory_recommended_call_type"] = "non_conversation"
            quality_flags["non_conversation_advisory_final_call_type"] = call_type
            review_reasons = self._unique(review_reasons + ["non_conversation_advisory"])
            needs_review = True
        if non_conversation_soft_warning_sources:
            quality_flags["non_conversation_soft_warning_llm_sales_signal"] = True
            quality_flags["non_conversation_soft_warning_sources"] = non_conversation_soft_warning_sources
            review_reasons = self._unique(
                review_reasons + ["non_conversation_llm_sales_signal_soft_warning"]
            )
            needs_review = True
        quality_flags["needs_review"] = needs_review
        quality_flags["review_reasons"] = review_reasons

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
            "needs_review": needs_review,
            "review_reasons": review_reasons,
        }
        return self._apply_non_conversation_hard_validation(call, normalized)

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

    def _analysis_model_for_provider(self, provider: str) -> str:
        if provider == "codex_cli":
            return (self._settings.codex_analyze_model or "").strip() or "unknown"
        if provider == "openai":
            return (self._settings.openai_analysis_model or "").strip() or "unknown"
        if provider == "ollama":
            return (self._settings.ollama_model or "").strip() or "unknown"
        if provider == "mock":
            return "mock"
        return provider or "unknown"

    def _build_analysis_meta(self, analysis: Dict[str, Any]) -> Dict[str, str]:
        provider = (self._settings.analyze_provider or "").strip().lower() or "mock"
        quality_flags = analysis.get("quality_flags") if isinstance(analysis.get("quality_flags"), dict) else {}
        prompt_version = quality_flags.get("analyze_prompt_version") or self._analysis_prompt_version()
        return {
            "analysis_model": self._analysis_model_for_provider(provider),
            "analysis_provider": provider,
            "analysis_prompt_version": str(prompt_version),
            "analyzed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    @staticmethod
    def _with_analysis_runtime_metadata(analysis: Dict[str, Any]) -> Dict[str, Any]:
        enriched = dict(analysis)
        quality_flags = enriched.get("quality_flags") if isinstance(enriched.get("quality_flags"), dict) else {}
        analysis_meta = enriched.get("analysis_meta") if isinstance(enriched.get("analysis_meta"), dict) else {}
        enriched["analyze_model"] = str(analysis_meta.get("analysis_model") or "")
        enriched["analyze_prompt_profile"] = str(quality_flags.get("analyze_prompt_profile") or "")
        enriched["analyze_prompt_truncated"] = bool(quality_flags.get("analyze_prompt_truncated"))
        enriched["analyze_prompt_chars"] = int(quality_flags.get("analyze_transcript_chars_prompt", 0) or 0)
        return enriched

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
        return self._detect_call_type(text) == "non_conversation"

    def _non_conversation_analysis(self, signals: Optional[Any] = None) -> Dict[str, Any]:
        reason_codes = list(getattr(signals, "reason_codes", ()) or ())
        contact_subtype = getattr(signals, "recommended_contact_subtype", None)
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
            "quality_flags": {
                "pre_llm_non_conversation_gate": bool(signals is not None),
                "transcript_quality_guardrails_version": TRANSCRIPT_QUALITY_GUARDRAILS_VERSION,
                "transcript_quality_label": getattr(signals, "label", None),
                "transcript_quality_score": getattr(signals, "score", None),
                "transcript_quality_reason_codes": reason_codes,
                "transcript_quality_should_force_non_conversation": bool(
                    getattr(signals, "should_force_non_conversation", False)
                ),
                "transcript_quality_recommended_call_type": getattr(signals, "recommended_call_type", None),
                "transcript_quality_recommended_contact_subtype": contact_subtype,
            },
        }

    def _openai_analysis(self, call: CallRecord, text: str, profile: Optional[str] = None) -> Dict[str, Any]:
        client = self._openai_client()
        prompt_context = self._analysis_prompt_context(call, text, profile)
        prompt = prompt_context["llm_prompt"]
        user_prompt = prompt_context["user_prompt"]
        metrics = prompt_context["metrics"]
        prompt_version = self._analysis_prompt_version(profile)
        cached = self._analysis_cache_lookup(
            provider="openai",
            model=self._settings.openai_analysis_model,
            reasoning="temperature=0.1",
            prompt_version=prompt_version,
            prompt=prompt,
        )
        if cached is not None:
            return self._with_analysis_prompt_quality_flags(
                cached,
                metrics=metrics,
                prompt_version=prompt_version,
                cache_hit=True,
            )
        response = client.chat.completions.create(
            model=self._settings.openai_analysis_model,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": prompt_context["system_prompt"]},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content if response.choices else None
        if not content:
            raise RuntimeError("OpenAI analysis returned empty content")
        data = json.loads(content)
        if not isinstance(data, dict):
            raise RuntimeError("OpenAI analysis must return object JSON")
        data = self._with_analysis_prompt_quality_flags(
            data,
            metrics=metrics,
            prompt_version=prompt_version,
            cache_hit=False,
        )
        self._analysis_cache_store(
            provider="openai",
            model=self._settings.openai_analysis_model,
            reasoning="temperature=0.1",
            prompt_version=prompt_version,
            prompt=prompt,
            response=data,
        )
        return data

    def _ollama_analysis(self, call: CallRecord, text: str, profile: Optional[str] = None) -> Dict[str, Any]:
        client = self._ollama_client()
        prompt_context = self._analysis_prompt_context(call, text, profile)
        prompt = prompt_context["llm_prompt"]
        user_prompt = prompt_context["user_prompt"]
        metrics = prompt_context["metrics"]
        prompt_version = self._analysis_prompt_version(profile)
        reasoning = f"think={self._settings.ollama_think}"
        cached = self._analysis_cache_lookup(
            provider="ollama",
            model=self._settings.ollama_model,
            reasoning=reasoning,
            prompt_version=prompt_version,
            prompt=prompt,
        )
        if cached is not None:
            return self._with_analysis_prompt_quality_flags(
                cached,
                metrics=metrics,
                prompt_version=prompt_version,
                cache_hit=True,
            )
        payload = client.generate_json(
            model=self._settings.ollama_model,
            think=self._settings.ollama_think,
            temperature=self._settings.ollama_temperature,
            system_prompt=prompt_context["system_prompt"],
            user_prompt=user_prompt,
            num_predict=max(200, int(self._settings.analyze_ollama_num_predict)),
        )
        if not isinstance(payload, dict):
            raise RuntimeError("Ollama analysis must return object JSON")
        payload = self._with_analysis_prompt_quality_flags(
            payload,
            metrics=metrics,
            prompt_version=prompt_version,
            cache_hit=False,
        )
        self._analysis_cache_store(
            provider="ollama",
            model=self._settings.ollama_model,
            reasoning=reasoning,
            prompt_version=prompt_version,
            prompt=prompt,
            response=payload,
        )
        return payload

    def _codex_cli_analysis(self, call: CallRecord, text: str, profile: Optional[str] = None) -> Dict[str, Any]:
        codex_bin = (self._settings.codex_cli_command or "codex").strip() or "codex"
        if shutil.which(codex_bin) is None:
            raise RuntimeError(f"codex binary is not available: {codex_bin}")

        prompt_context = self._analysis_prompt_context(call, text, profile)
        prompt = prompt_context["llm_prompt"]
        metrics = prompt_context["metrics"]
        prompt_version = self._analysis_prompt_version(profile)
        timeout_sec = max(15, int(self._settings.codex_cli_timeout_sec))
        retryable_marker = "no last agent message"
        max_attempts = 5
        last_error: Optional[str] = None
        reasoning_effort = (self._settings.codex_reasoning_effort or "").strip().lower()
        cached = self._analysis_cache_lookup(
            provider="codex_cli",
            model=self._settings.codex_analyze_model,
            reasoning=reasoning_effort,
            prompt_version=prompt_version,
            prompt=prompt,
        )
        if cached is not None:
            return self._with_analysis_prompt_quality_flags(
                cached,
                metrics=metrics,
                prompt_version=prompt_version,
                cache_hit=True,
            )

        for attempt in range(1, max_attempts + 1):
            with tempfile.NamedTemporaryFile(
                prefix="mango_codex_analyze_",
                suffix=".txt",
            ) as out_file:
                cmd = [
                    codex_bin,
                    "exec",
                    "--skip-git-repo-check",
                    "--ephemeral",
                    "--ignore-user-config",
                    "--sandbox",
                    "read-only",
                    "--model",
                    self._settings.codex_analyze_model,
                    "--output-last-message",
                    out_file.name,
                ]
                if reasoning_effort in {"low", "medium", "high"}:
                    cmd.extend(["-c", f'model_reasoning_effort="{reasoning_effort}"'])
                cmd.append("-")
                proc = subprocess.run(
                    cmd,
                    input=prompt,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=timeout_sec,
                )
                raw = Path(out_file.name).read_text(encoding="utf-8", errors="ignore")

            for candidate in (raw, proc.stdout or "", proc.stderr or ""):
                candidate = (candidate or "").strip()
                if not candidate:
                    continue
                try:
                    payload = self._extract_json_payload(candidate)
                except RuntimeError:
                    continue
                if isinstance(payload, dict):
                    payload = self._with_analysis_prompt_quality_flags(
                        payload,
                        metrics=metrics,
                        prompt_version=prompt_version,
                        cache_hit=False,
                    )
                    self._analysis_cache_store(
                        provider="codex_cli",
                        model=self._settings.codex_analyze_model,
                        reasoning=reasoning_effort,
                        prompt_version=prompt_version,
                        prompt=prompt,
                        response=payload,
                    )
                    return payload

            stderr = (proc.stderr or "").strip()
            if proc.returncode == 0:
                last_error = "Codex analysis returned empty content"
                if attempt < max_attempts:
                    time.sleep(min(5, attempt + 1))
                    continue
                raise RuntimeError(last_error)

            stderr_tail = stderr.splitlines()[-1:] or [""]
            last_error = f"codex exec failed rc={proc.returncode}: {stderr_tail[0].strip()}"
            if retryable_marker in stderr.lower() and attempt < max_attempts:
                time.sleep(min(6, attempt * 2))
                continue
            raise RuntimeError(last_error)

        raise RuntimeError(last_error or "Codex analysis failed")

    def _analyze_text(self, call: CallRecord, text: str) -> Dict[str, Any]:
        signals = detect_non_conversation_signals(
            transcript_text=text,
            duration_sec=getattr(call, "duration_sec", None),
        )
        non_conversation_advisory_enabled = _truthy_env_flag(NON_CONVERSATION_ADVISORY_ENV)
        if signals.should_force_non_conversation and not non_conversation_advisory_enabled:
            return self._non_conversation_analysis(signals)
        if self._is_non_conversation(text) and not non_conversation_advisory_enabled:
            return self._non_conversation_analysis()
        provider = self._settings.analyze_provider
        profile = self._analysis_prompt_profile()
        if provider == "mock":
            return self._mock_analysis(call, text)
        if provider == "openai":
            payload = self._openai_analysis(call, text, profile)
        elif provider == "ollama":
            payload = self._ollama_analysis(call, text, profile)
        elif provider == "codex_cli":
            payload = self._codex_cli_analysis(call, text, profile)
        else:
            raise RuntimeError(f"Unsupported ANALYZE_PROVIDER={provider}")
        if profile == "compact" and self._should_escalate_full_profile(text, payload):
            if provider == "openai":
                return self._openai_analysis(call, text, "full")
            if provider == "ollama":
                return self._ollama_analysis(call, text, "full")
            if provider == "codex_cli":
                return self._codex_cli_analysis(call, text, "full")
        return payload

    def run(self, session: Session, limit: int) -> Dict[str, int]:
        max_attempts = max(1, self._settings.analyze_max_attempts)
        worker_id = self._analysis_worker_id()
        claimed_ids = self._claim_batch(session, limit=limit, worker_id=worker_id)
        success = 0
        failed = 0
        for call_id in claimed_ids:
            call = session.get(CallRecord, call_id)
            if call is None:
                continue
            if call.analysis_status != "in_progress" or call.analysis_worker_id != worker_id:
                continue
            call.analyze_attempts = int(call.analyze_attempts or 0) + 1
            attempt = call.analyze_attempts
            try:
                text = (call.transcript_text or "").strip()
                if not text:
                    raise RuntimeError("Empty transcript_text")
                raw_analysis = self._analyze_text(call, text)
                analysis = self._normalize_analysis(call, text, raw_analysis)
                analysis["analysis_meta"] = self._build_analysis_meta(analysis)
                analysis = self._with_analysis_runtime_metadata(analysis)
                call.analysis_json = json.dumps(analysis, ensure_ascii=False)
                self._export_analysis_files(call, analysis)
                call.analysis_status = "done"
                call.analysis_worker_id = None
                call.analysis_claimed_at = None
                call.sync_status = "pending"
                call.next_retry_at = None
                call.dead_letter_stage = None
                call.last_error = None
                success += 1
            except Exception as exc:  # noqa: BLE001
                call.last_error = f"analyze: {exc}"
                if attempt >= max_attempts:
                    call.analysis_status = "dead"
                    call.analysis_worker_id = None
                    call.analysis_claimed_at = None
                    call.dead_letter_stage = "analyze"
                    call.next_retry_at = None
                else:
                    call.analysis_status = "failed"
                    call.analysis_worker_id = None
                    call.analysis_claimed_at = None
                    call.next_retry_at = self._utc_now() + self._retry_delay(attempt)
                failed += 1
            session.add(call)
            session.commit()
        return {
            "processed": len(claimed_ids),
            "claimed": len(claimed_ids),
            "success": success,
            "failed": failed,
            "worker_id": worker_id,
        }
