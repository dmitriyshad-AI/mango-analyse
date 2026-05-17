from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from mango_mvp.channels.draft_prompt_builder import (
    IDENTITY_DISCLOSURE_FORBIDDEN_PHRASES,
    build_draft_prompt,
    safe_schedule_template,
    should_force_manager_only,
)
from mango_mvp.question_catalog.classifier import load_valid_theme_and_service_ids


SUBSCRIPTION_LLM_SCHEMA_VERSION = "subscription_llm_draft_v1_2026_05_16"
DEFAULT_CODEX_MODEL = "gpt-5.5"
DEFAULT_CODEX_REASONING_EFFORT = "medium"
SAFE_FALLBACK_DRAFT_TEXT = "Спасибо за сообщение. Передам вопрос менеджеру, он вернется с проверенным ответом."
UNKNOWN_TOPIC_FALLBACK_ID = "service:S2_unclear"

ALLOWED_ROUTES = {"draft_for_manager", "manager_only", "blocked"}
ALLOWED_MESSAGE_TYPES = {"question", "non_question", "context_update", "wait_for_more", "manager_only"}
BASE_SAFETY_FLAGS = ("manager_approval_required", "no_auto_send")
HIGH_RISK_THEME_IDS = {
    "theme:003_payment_status",
    "theme:005_discounts",
    "theme:007_matkap_payment",
    "theme:008_tax_deduction",
    "theme:009_refund",
    "theme:012_certificates",
    "theme:019b_negative_feedback",
    "theme:029_legal_question",
}
HIGH_RISK_MARKERS = (
    "refund",
    "matkap",
    "tax",
    "legal",
    "negative",
    "payment_status",
    "documents",
    "discount",
    "возврат",
    "маткап",
    "налог",
    "юрид",
    "жалоб",
    "подтверждение оплаты",
    "статус оплаты",
    "документ",
    "скид",
)
HIGH_RISK_INPUT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("refund", re.compile(r"\bвозврат\w*|\bвернуть\s+(?:деньги|оплату|плат[её]ж)|\bкак\s+получить\s+возврат", re.I)),
    ("matkap", re.compile(r"мат(?:еринск(?:ий|ого|им|ому)?\s*)?капитал|маткап|сертификат\s+мат", re.I)),
    ("tax", re.compile(r"налогов(?:ый|ого|ому|ым)?\s+вычет|справк\w*\s+для\s+налог", re.I)),
    ("legal", re.compile(r"юрид|суд|претензи|досудеб|законн|правомер", re.I)),
    ("complaint", re.compile(r"жалоб|конфликт|недовольн|претензи|обман|ужасн|плохо\s+провел", re.I)),
    ("discount", re.compile(r"скидк|промокод|льгот|рассрочк", re.I)),
    ("payment_status", re.compile(r"прошл[ао]\s+ли\s+оплат|подтверждени[ея]\s+оплат|статус\s+оплат|поступил[а]?\s+оплат|чек", re.I)),
)
_RETRYABLE_MARKERS = (
    "no last agent message",
    "temporarily unavailable",
    "temporary",
    "timeout",
    "timed out",
    "rate limit",
    "429",
    "500",
    "502",
    "503",
    "504",
    "overloaded",
)

_Runner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class SubscriptionDraftResult:
    message_type: str = "question"
    broad_group: str = ""
    topic_id: str = "service:S2_unclear"
    topic_confidence: float = 0.0
    confidence_group: float = 0.0
    alternative_themes: tuple[str, ...] = field(default_factory=tuple)
    risk_level: str = "unknown"
    route: str = "manager_only"
    draft_text: str = SAFE_FALLBACK_DRAFT_TEXT
    manager_checklist: tuple[str, ...] = field(default_factory=tuple)
    missing_facts: tuple[str, ...] = field(default_factory=tuple)
    forbidden_promises_detected: tuple[str, ...] = field(default_factory=tuple)
    crm_recommendations: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    safety_flags: tuple[str, ...] = BASE_SAFETY_FLAGS
    context_used: tuple[str, ...] = field(default_factory=tuple)
    context_warnings: tuple[str, ...] = field(default_factory=tuple)
    manager_followup_required: bool = False
    manager_followup_deadline: Optional[str] = None
    provider: str = "codex_exec"
    schema_version: str = SUBSCRIPTION_LLM_SCHEMA_VERSION
    raw_response: Optional[str] = None
    error: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        route = str(self.route or "manager_only").strip()
        if route not in ALLOWED_ROUTES:
            route = "manager_only"
        text = str(self.draft_text or "").strip() or SAFE_FALLBACK_DRAFT_TEXT
        message_type = str(self.message_type or "question").strip()
        if message_type not in ALLOWED_MESSAGE_TYPES:
            message_type = "manager_only"
        flags = tuple(dict.fromkeys([*BASE_SAFETY_FLAGS, *(_clean_list(self.safety_flags, max_items=16, max_chars=80))]))
        object.__setattr__(self, "message_type", message_type)
        object.__setattr__(self, "broad_group", str(self.broad_group or "").strip()[:80])
        object.__setattr__(self, "route", route)
        object.__setattr__(self, "draft_text", text)
        object.__setattr__(self, "topic_id", str(self.topic_id or "service:S2_unclear").strip() or "service:S2_unclear")
        object.__setattr__(self, "topic_confidence", _clamp_float(self.topic_confidence))
        object.__setattr__(self, "confidence_group", _clamp_float(self.confidence_group))
        object.__setattr__(self, "alternative_themes", tuple(_clean_list(self.alternative_themes, max_items=5, max_chars=120)))
        object.__setattr__(self, "risk_level", str(self.risk_level or "unknown").strip()[:80] or "unknown")
        object.__setattr__(self, "manager_checklist", tuple(_clean_list(self.manager_checklist, max_items=12, max_chars=240)))
        object.__setattr__(self, "missing_facts", tuple(_clean_list(self.missing_facts, max_items=12, max_chars=160)))
        object.__setattr__(
            self,
            "forbidden_promises_detected",
            tuple(_clean_list(self.forbidden_promises_detected, max_items=12, max_chars=160)),
        )
        object.__setattr__(self, "crm_recommendations", tuple(_clean_crm_recommendations(self.crm_recommendations)))
        object.__setattr__(self, "safety_flags", flags)
        object.__setattr__(self, "context_used", tuple(_clean_list(self.context_used, max_items=12, max_chars=100)))
        object.__setattr__(self, "context_warnings", tuple(_clean_list(self.context_warnings, max_items=12, max_chars=120)))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_json_dict(self, *, include_raw_response: bool = False) -> Mapping[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "provider": self.provider,
            "message_type": self.message_type,
            "broad_group": self.broad_group,
            "topic_id": self.topic_id,
            "topic_confidence": self.topic_confidence,
            "confidence_theme": self.topic_confidence,
            "confidence_group": self.confidence_group,
            "alternative_themes": list(self.alternative_themes),
            "risk_level": self.risk_level,
            "route": self.route,
            "draft_text": self.draft_text,
            "manager_checklist": list(self.manager_checklist),
            "missing_facts": list(self.missing_facts),
            "forbidden_promises_detected": list(self.forbidden_promises_detected),
            "crm_recommendations": [dict(item) for item in self.crm_recommendations],
            "manager_followup_required": self.manager_followup_required,
            "manager_followup_deadline": self.manager_followup_deadline,
            "safety_flags": list(self.safety_flags),
            "context_used": list(self.context_used),
            "context_warnings": list(self.context_warnings),
            "error": self.error,
            "metadata": dict(self.metadata),
        }
        if include_raw_response:
            payload["raw_response"] = self.raw_response
        return payload


class SubscriptionLlmDraftProvider:
    def __init__(
        self,
        *,
        codex_bin: str = "codex",
        model: str = DEFAULT_CODEX_MODEL,
        reasoning_effort: str = DEFAULT_CODEX_REASONING_EFFORT,
        timeout_sec: int = 90,
        max_attempts: int = 2,
        cache_dir: Optional[Path | str] = None,
        runner: Optional[_Runner] = None,
        sleep: Callable[[float], None] = time.sleep,
        base_env: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.codex_bin = str(codex_bin or "codex").strip() or "codex"
        self.model = str(model or DEFAULT_CODEX_MODEL).strip() or DEFAULT_CODEX_MODEL
        self.reasoning_effort = str(reasoning_effort or DEFAULT_CODEX_REASONING_EFFORT).strip() or DEFAULT_CODEX_REASONING_EFFORT
        self.timeout_sec = max(1, int(timeout_sec))
        self.max_attempts = max(1, int(max_attempts))
        self.runner = runner or subprocess.run
        self.sleep = sleep
        self.base_env = dict(base_env) if base_env is not None else None
        self.cache_dir = _guard_cache_dir(cache_dir) if cache_dir is not None else None

    def build_draft(
        self,
        client_message: str,
        *,
        context: Optional[Mapping[str, Any]] = None,
    ) -> SubscriptionDraftResult:
        prompt = build_draft_prompt(client_message, context=context)
        result = self.generate_from_prompt(prompt, force_manager_only=should_force_manager_only(context))
        return apply_input_policy_guards(result, client_message=client_message, context=context)

    def generate(self, prompt: str) -> SubscriptionDraftResult:
        return self.generate_from_prompt(prompt)

    def generate_from_prompt(self, prompt: str, *, force_manager_only: bool = False) -> SubscriptionDraftResult:
        prompt_text = str(prompt or "").strip()
        if not prompt_text:
            return safe_fallback_draft(reason="empty_prompt")

        cache_key = _cache_key(
            {
                "schema_version": SUBSCRIPTION_LLM_SCHEMA_VERSION,
                "provider": "codex_exec",
                "model": self.model,
                "reasoning_effort": self.reasoning_effort,
                "prompt": prompt_text,
                "force_manager_only": force_manager_only,
            }
        )
        cached = self._cache_get(cache_key)
        if cached is not None:
            return _with_metadata(cached, {"cache_hit": True})

        last_error = "codex_exec_failed"
        for attempt in range(1, self.max_attempts + 1):
            try:
                result = self._run_once(prompt_text, force_manager_only=force_manager_only)
            except subprocess.TimeoutExpired:
                return safe_fallback_draft(reason="timeout", metadata={"attempt": attempt, "timeout_sec": self.timeout_sec})
            except FileNotFoundError:
                return safe_fallback_draft(reason="codex_binary_not_found", metadata={"codex_bin": self.codex_bin})
            except _CodexRetryableError as exc:
                last_error = str(exc) or "retryable_codex_error"
                if attempt < self.max_attempts:
                    self.sleep(min(3.0, float(attempt)))
                    continue
                return safe_fallback_draft(reason="codex_retryable_error", metadata={"last_error": last_error})
            except Exception as exc:  # noqa: BLE001
                return safe_fallback_draft(reason="invalid_json_or_codex_error", metadata={"last_error": str(exc)[:400]})
            self._cache_put(cache_key, result)
            return result
        return safe_fallback_draft(reason=last_error)

    def _run_once(self, prompt: str, *, force_manager_only: bool) -> SubscriptionDraftResult:
        with tempfile.NamedTemporaryFile(prefix="mango_draft_codex_", suffix=".json") as out_file:
            output_path = Path(out_file.name)
            cmd = build_codex_exec_command(
                output_path=output_path,
                codex_bin=self.codex_bin,
                model=self.model,
                reasoning_effort=self.reasoning_effort,
            )
            proc = self.runner(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                check=False,
                timeout=self.timeout_sec,
                env=build_codex_exec_env(self.base_env),
            )
            raw = output_path.read_text(encoding="utf-8", errors="ignore")

        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            message = f"codex exec failed rc={proc.returncode}: {' '.join(stderr.splitlines()[-2:])[:400]}"
            if _is_retryable(stderr):
                raise _CodexRetryableError(message)
            raise RuntimeError(message)

        payload = extract_json_object(raw or proc.stdout or proc.stderr or "")
        result = normalize_subscription_draft_payload(payload, raw_response=raw)
        if force_manager_only and result.route != "manager_only":
            result = replace(
                result,
                route="manager_only",
                safety_flags=tuple(dict.fromkeys([*result.safety_flags, "forced_manager_only_by_rop_policy"])),
                metadata={**dict(result.metadata), "forced_route": "manager_only"},
            )
        return guard_identity_disclosure(result)

    def _cache_get(self, cache_key: str) -> Optional[SubscriptionDraftResult]:
        if self.cache_dir is None:
            return None
        path = self.cache_dir / f"{cache_key}.json"
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return normalize_subscription_draft_payload(payload)
        except Exception:
            return None

    def _cache_put(self, cache_key: str, result: SubscriptionDraftResult) -> None:
        if self.cache_dir is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / f"{cache_key}.json"
        path.write_text(json.dumps(result.to_json_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


class FakeSubscriptionLlmDraftProvider:
    def __init__(self, result: Optional[SubscriptionDraftResult | Mapping[str, Any]] = None) -> None:
        self.result = normalize_subscription_draft_payload(result) if result is not None else safe_fallback_draft(
            reason="fake_provider_default"
        )
        self.prompts: list[str] = []

    def build_draft(
        self,
        client_message: str,
        *,
        context: Optional[Mapping[str, Any]] = None,
    ) -> SubscriptionDraftResult:
        prompt = build_draft_prompt(client_message, context=context)
        result = self.generate_from_prompt(prompt, force_manager_only=should_force_manager_only(context))
        return apply_input_policy_guards(result, client_message=client_message, context=context)

    def generate(self, prompt: str) -> SubscriptionDraftResult:
        return self.generate_from_prompt(prompt)

    def generate_from_prompt(self, prompt: str, *, force_manager_only: bool = False) -> SubscriptionDraftResult:
        self.prompts.append(prompt)
        result = self.result
        if force_manager_only:
            result = replace(
                result,
                route="manager_only",
                safety_flags=tuple(dict.fromkeys([*result.safety_flags, "forced_manager_only_by_rop_policy"])),
            )
        return guard_identity_disclosure(result)


def build_codex_exec_command(
    *,
    output_path: Path | str,
    codex_bin: str = "codex",
    model: str = DEFAULT_CODEX_MODEL,
    reasoning_effort: str = DEFAULT_CODEX_REASONING_EFFORT,
) -> list[str]:
    cmd = [
        str(codex_bin or "codex").strip() or "codex",
        "exec",
        "--skip-git-repo-check",
        "--ephemeral",
        "--sandbox",
        "read-only",
        "--model",
        str(model or DEFAULT_CODEX_MODEL).strip() or DEFAULT_CODEX_MODEL,
    ]
    reasoning = str(reasoning_effort or "").strip()
    if reasoning:
        cmd.extend(["-c", f'model_reasoning_effort="{reasoning}"'])
    cmd.extend(["--output-last-message", str(output_path), "-"])
    return cmd


def build_codex_exec_env(base_env: Optional[Mapping[str, str]] = None, *, codex_home: Optional[Path | str] = None) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    env.pop("OPENAI_API_KEY", None)
    if codex_home is not None:
        env["CODEX_HOME"] = str(codex_home)
    return env


@dataclass(frozen=True)
class CodexExecConfig:
    codex_bin: str = "codex"
    model: str = DEFAULT_CODEX_MODEL
    reasoning_effort: str = DEFAULT_CODEX_REASONING_EFFORT

    def build_command(self, output_path: Path | str) -> list[str]:
        return build_codex_exec_command(
            output_path=output_path,
            codex_bin=self.codex_bin,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
        )


def normalize_subscription_draft_payload(payload: Mapping[str, Any] | SubscriptionDraftResult, *, raw_response: Optional[str] = None) -> SubscriptionDraftResult:
    if isinstance(payload, SubscriptionDraftResult):
        return payload
    if not isinstance(payload, Mapping):
        raise RuntimeError("subscription draft response JSON root must be an object")
    schedule = payload.get("safe_schedule_template")
    manager_followup_required = bool(payload.get("manager_followup_required"))
    manager_followup_deadline = _optional_text(payload.get("manager_followup_deadline"))
    if isinstance(schedule, Mapping) and schedule.get("manager_followup_required") is True:
        manager_followup_required = True
        manager_followup_deadline = manager_followup_deadline or _optional_text(
            schedule.get("manager_followup_deadline") or schedule.get("deadline_at")
        )
    result = SubscriptionDraftResult(
        message_type=str(payload.get("message_type") or "question"),
        broad_group=str(payload.get("broad_group") or ""),
        topic_id=str(payload.get("topic_id") or "service:S2_unclear"),
        topic_confidence=_clamp_float(payload.get("confidence_theme", payload.get("topic_confidence"))),
        confidence_group=_clamp_float(payload.get("confidence_group")),
        alternative_themes=tuple(_clean_list(payload.get("alternative_themes"), max_items=5, max_chars=120)),
        risk_level=str(payload.get("risk_level") or "unknown"),
        route=str(payload.get("route") or "manager_only"),
        draft_text=str(payload.get("draft_text") or SAFE_FALLBACK_DRAFT_TEXT),
        manager_checklist=tuple(_clean_list(payload.get("manager_checklist"), max_items=12, max_chars=240)),
        missing_facts=tuple(_clean_list(payload.get("missing_facts"), max_items=12, max_chars=160)),
        forbidden_promises_detected=tuple(_clean_list(payload.get("forbidden_promises_detected"), max_items=12, max_chars=160)),
        crm_recommendations=tuple(_clean_crm_recommendations(payload.get("crm_recommendations"))),
        safety_flags=tuple(_clean_list(payload.get("safety_flags"), max_items=16, max_chars=80)),
        context_used=tuple(_clean_list(payload.get("context_used"), max_items=12, max_chars=100)),
        context_warnings=tuple(_clean_list(payload.get("context_warnings"), max_items=12, max_chars=120)),
        manager_followup_required=manager_followup_required,
        manager_followup_deadline=manager_followup_deadline,
        raw_response=raw_response,
    )
    return guard_identity_disclosure(apply_taxonomy_topic_guard(apply_subscription_policy_guards(result)))


def safe_fallback_draft(*, reason: str, metadata: Optional[Mapping[str, Any]] = None) -> SubscriptionDraftResult:
    extra_flags = ("codex_exec_timeout",) if reason == "timeout" else ()
    return SubscriptionDraftResult(
        message_type="manager_only",
        route="manager_only",
        draft_text=SAFE_FALLBACK_DRAFT_TEXT,
        manager_checklist=("Проверить вопрос вручную.",),
        missing_facts=("llm_response",),
        safety_flags=(*BASE_SAFETY_FLAGS, "llm_fallback", "draft_only", *extra_flags),
        error=reason,
        metadata=dict(metadata or {}),
    )


def extract_json_object(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        raise RuntimeError("empty subscription draft response")
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise RuntimeError("subscription draft response does not contain JSON object")
        payload = json.loads(raw[start : end + 1])
    if not isinstance(payload, dict):
        raise RuntimeError("subscription draft response JSON root must be an object")
    return payload


def parse_llm_json(text: str) -> SubscriptionDraftResult:
    try:
        return normalize_subscription_draft_payload(extract_json_object(text), raw_response=text)
    except Exception as exc:  # noqa: BLE001
        return safe_fallback_draft(reason="invalid_json", metadata={"parse_error": str(exc)[:300]})


def draft_has_identity_disclosure(text: str) -> bool:
    return bool(find_identity_disclosure_phrases(text))


def find_identity_disclosure_phrases(text: str) -> tuple[str, ...]:
    lowered = str(text or "").casefold()
    return tuple(phrase for phrase in IDENTITY_DISCLOSURE_FORBIDDEN_PHRASES if phrase.casefold() in lowered)


def guard_identity_disclosure(result: SubscriptionDraftResult) -> SubscriptionDraftResult:
    phrases = find_identity_disclosure_phrases(result.draft_text)
    if not phrases:
        return result
    return replace(
        result,
        route="manager_only",
        draft_text=SAFE_FALLBACK_DRAFT_TEXT,
        forbidden_promises_detected=tuple(dict.fromkeys([*result.forbidden_promises_detected, *phrases])),
        safety_flags=tuple(dict.fromkeys([*result.safety_flags, "identity_disclosure_guarded", "bot_identity_disclosure", "llm_fallback"])),
        error=result.error or "identity_disclosure_guarded",
    )


def apply_subscription_policy_guards(result: SubscriptionDraftResult) -> SubscriptionDraftResult:
    route = result.route
    flags = list(result.safety_flags)
    manager_checklist = list(result.manager_checklist)
    metadata = dict(result.metadata)

    if result.topic_confidence < 0.70:
        route = "manager_only"
        flags.append("low_confidence_manager_only")
        manager_checklist.append("Модель не уверена в теме: проверить вручную.")
        metadata["forced_route_low_confidence"] = True

    if is_high_risk_result(result):
        route = "manager_only"
        flags.append("high_risk_manager_only")
        manager_checklist.append("Высокорисковая тема: не отправлять клиенту без ручной проверки.")
        metadata["forced_route_high_risk"] = True

    if result.message_type in {"non_question", "context_update", "wait_for_more", "manager_only"}:
        route = "manager_only"
        flags.append(f"message_type_{result.message_type}")
        metadata["forced_route_message_type"] = result.message_type

    if route == result.route and tuple(flags) == result.safety_flags and tuple(manager_checklist) == result.manager_checklist:
        return result
    return replace(
        result,
        route=route,
        safety_flags=tuple(dict.fromkeys(flags)),
        manager_checklist=tuple(dict.fromkeys(manager_checklist)),
        metadata=metadata,
    )


def apply_input_policy_guards(
    result: SubscriptionDraftResult,
    *,
    client_message: str,
    context: Optional[Mapping[str, Any]] = None,
) -> SubscriptionDraftResult:
    markers = detect_high_risk_input_markers(client_message, context=context)
    if not markers:
        return result
    flags = tuple(dict.fromkeys([*result.safety_flags, "high_risk_input_manager_only", "high_risk_manager_only"]))
    checklist = tuple(
        dict.fromkeys(
            [
                *result.manager_checklist,
                "Исходное сообщение клиента содержит высокорисковую тему: проверить вручную.",
            ]
        )
    )
    return replace(
        result,
        route="manager_only",
        safety_flags=flags,
        manager_checklist=checklist,
        metadata={
            **dict(result.metadata),
            "forced_route_high_risk_input": list(markers),
        },
    )


def apply_taxonomy_topic_guard(result: SubscriptionDraftResult) -> SubscriptionDraftResult:
    valid_ids = load_valid_theme_and_service_ids()
    topic_id = str(result.topic_id or "").strip()
    valid_alternatives = tuple(item for item in result.alternative_themes if item in valid_ids)
    invalid_alternatives = tuple(item for item in result.alternative_themes if item and item not in valid_ids)
    if topic_id in valid_ids and valid_alternatives == result.alternative_themes:
        return result

    flags = list(result.safety_flags)
    checklist = list(result.manager_checklist)
    metadata = dict(result.metadata)

    if topic_id not in valid_ids:
        flags.append("invalid_topic_id_normalized")
        checklist.append("LLM вернула тему не из утвержденного списка: проверить вручную.")
        metadata["original_invalid_topic_id"] = topic_id
        topic_id = UNKNOWN_TOPIC_FALLBACK_ID
    if invalid_alternatives:
        flags.append("invalid_alternative_themes_removed")
        metadata["invalid_alternative_themes"] = list(invalid_alternatives)

    return replace(
        result,
        topic_id=topic_id,
        alternative_themes=valid_alternatives,
        route="manager_only",
        safety_flags=tuple(dict.fromkeys(flags)),
        manager_checklist=tuple(dict.fromkeys(checklist)),
        metadata=metadata,
    )


def is_high_risk_result(result: SubscriptionDraftResult) -> bool:
    topic = result.topic_id.strip()
    if topic in HIGH_RISK_THEME_IDS:
        return True
    haystack = " ".join(
        [
            topic,
            result.broad_group,
            result.risk_level,
            *result.alternative_themes,
            *result.safety_flags,
            *result.context_warnings,
        ]
    ).casefold()
    return any(marker.casefold() in haystack for marker in HIGH_RISK_MARKERS)


def detect_high_risk_input_markers(client_message: str, *, context: Optional[Mapping[str, Any]] = None) -> tuple[str, ...]:
    texts = [str(client_message or "")]
    if isinstance(context, Mapping):
        recent = context.get("recent_messages")
        if isinstance(recent, Sequence) and not isinstance(recent, (str, bytes, bytearray)):
            texts.extend(str(item or "") for item in recent[-3:])
        for key in ("risk_flags", "context_warnings", "missing_facts"):
            value = context.get(key)
            if isinstance(value, str):
                texts.append(value)
            elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
                texts.extend(str(item or "") for item in value)
    haystack = "\n".join(texts)
    markers = [name for name, pattern in HIGH_RISK_INPUT_PATTERNS if pattern.search(haystack)]
    return tuple(dict.fromkeys(markers))


DraftGenerationResult = SubscriptionDraftResult
CodexExecDraftProvider = SubscriptionLlmDraftProvider
FakeDraftProvider = FakeSubscriptionLlmDraftProvider
contains_bot_identity_disclosure = draft_has_identity_disclosure


def subscription_llm_safety_contract() -> Mapping[str, Any]:
    return {
        "schema_version": SUBSCRIPTION_LLM_SCHEMA_VERSION,
        "provider": "codex_exec",
        "uses_openai_api_key": False,
        "client_auto_send_allowed": False,
        "crm_write_allowed": False,
        "tallanto_write_allowed": False,
        "stable_runtime_write_allowed": False,
        "fallback_text": SAFE_FALLBACK_DRAFT_TEXT,
        "identity_disclosure_forbidden_phrases": list(IDENTITY_DISCLOSURE_FORBIDDEN_PHRASES),
        "safe_schedule_template": safe_schedule_template(),
    }


def _clean_list(value: Any, *, max_items: int, max_chars: int) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values: Sequence[Any] = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = value
    else:
        return []
    result: list[str] = []
    for item in values:
        text = str(item or "").strip()
        if not text:
            continue
        result.append(" ".join(text.split())[:max_chars])
        if len(result) >= max_items:
            break
    return result


def _clean_crm_recommendations(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    result: list[Mapping[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        recommendation = {
            "target": str(item.get("target") or "").strip()[:80],
            "action": str(item.get("action") or "").strip()[:80],
            "text": str(item.get("text") or "").strip()[:500],
            "requires_manager_approval": True,
        }
        if recommendation["target"] and recommendation["action"] and recommendation["text"]:
            result.append(recommendation)
        if len(result) >= 8:
            break
    return result


def _clamp_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, parsed))


def _optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _cache_key(payload: Mapping[str, Any]) -> str:
    import hashlib

    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _with_metadata(result: SubscriptionDraftResult, extra: Mapping[str, Any]) -> SubscriptionDraftResult:
    return replace(result, metadata={**dict(result.metadata), **dict(extra)})


def _guard_cache_dir(path: Path | str) -> Path:
    resolved = Path(path).expanduser().resolve(strict=False)
    if "stable_runtime" in resolved.parts:
        raise ValueError("subscription LLM cache must not be inside stable_runtime")
    return resolved


def _is_retryable(stderr: str) -> bool:
    lowered = (stderr or "").casefold()
    return any(marker in lowered for marker in _RETRYABLE_MARKERS)


class _CodexRetryableError(RuntimeError):
    pass
