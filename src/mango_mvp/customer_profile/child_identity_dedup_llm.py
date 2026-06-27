from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from mango_mvp.customer_profile.contracts import ProfileFieldCandidate
from mango_mvp.services.llm_response_cache import LLMResponseCache
from mango_mvp.utils.codex_cli import append_codex_service_tier
from mango_mvp.utils.phone import normalize_phone


NAMESPACE = "child_resolver_v1"
PROMPT_VERSION = "v5"
ESCALATION_PROMPT_VERSION = "v6"
PROVIDER_OPENAI = "openai"
PROVIDER_CODEX_CLI = "codex_cli"
DEFAULT_STOPLIST_PATH = Path.home() / ".mango_secrets" / "shared_phones_stoplist.json"
CHILD_FIELDS = {"child_name", "grade", "subject"}
MERGE_CONFIDENCE_VALUES = {"high", "low"}
ROOT_RESPONSE_KEYS = {"children"}
CHILD_RESPONSE_KEYS = {
    "child_id",
    "canonical_name",
    "name_variants",
    "grades",
    "subjects",
    "mention_ids",
    "merge_confidence",
}
CHILD_REQUIRED_RESPONSE_KEYS = CHILD_RESPONSE_KEYS - {"merge_confidence"}
JOINT_NAME_HINTS = {
    "александр",
    "александра",
    "алексей",
    "алиса",
    "анастасия",
    "анна",
    "артем",
    "артём",
    "артемий",
    "борис",
    "валентин",
    "валентина",
    "варвара",
    "василий",
    "вера",
    "вероника",
    "виктор",
    "виктория",
    "владимир",
    "вячеслав",
    "глеб",
    "даниил",
    "данил",
    "дарья",
    "денида",
    "дмитрий",
    "егор",
    "елена",
    "елизавета",
    "иван",
    "кирилл",
    "константин",
    "ксения",
    "лев",
    "макар",
    "мария",
    "матвей",
    "михаил",
    "милана",
    "никита",
    "олег",
    "олеся",
    "павел",
    "петр",
    "пётр",
    "родион",
    "роман",
    "савва",
    "семен",
    "семён",
    "софия",
    "степан",
    "тимофей",
    "тимур",
    "федор",
    "фёдор",
    "юлия",
    "ярослав",
}
JOINT_SEPARATOR_RE = re.compile(r"\s+(?:и)\s+|\s*,\s*", re.IGNORECASE)


class ChildResolverError(RuntimeError):
    def __init__(self, message: str, *, diagnostics: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.diagnostics = dict(diagnostics or {})


class ChatCompletionClient(Protocol):
    chat: Any


@dataclass(frozen=True)
class ChildResolverConfig:
    provider: str = PROVIDER_CODEX_CLI
    model: str = "gpt-5.4-mini"
    reasoning_effort: str = "medium"
    temperature: float = 0.0
    prompt_version: str = PROMPT_VERSION
    cache_enabled: bool = True
    cache_root_dir: str | Path = ".cache/llm_responses"
    cache_only: bool = False
    max_concurrency: int = 4
    max_retries: int = 2
    retry_initial_seconds: float = 0.5
    request_timeout_seconds: float = 180.0
    stoplist_path: str | Path = DEFAULT_STOPLIST_PATH
    codex_cli_command: str = "codex"
    codex_home: str | Path | None = None
    project_root: str | Path = "."
    trace_path: str | Path | None = None
    name_diagnostics_path: str | Path | None = None
    escalation_enabled: bool = False
    escalation_model: str = "gpt-5.5"
    escalation_reasoning_effort: str = "high"
    escalation_max_concurrency: int = 2
    escalation_request_timeout_seconds: float = 300.0
    strict_confidence_prompt: bool = False
    resolver_tier: str = "tier1"

    @classmethod
    def from_env(cls) -> "ChildResolverConfig":
        return cls(
            provider=os.getenv("PROFILE_LLM_CHILD_RESOLVER_PROVIDER", PROVIDER_CODEX_CLI).strip().lower()
            or PROVIDER_CODEX_CLI,
            model=os.getenv("PROFILE_LLM_CHILD_RESOLVER_MODEL", "gpt-5.4-mini").strip() or "gpt-5.4-mini",
            reasoning_effort=os.getenv("PROFILE_LLM_CHILD_RESOLVER_REASONING", "medium").strip() or "medium",
            cache_enabled=_bool_env("LLM_CACHE_ENABLED", True),
            cache_root_dir=os.getenv("LLM_CACHE_DIR", ".cache/llm_responses").strip() or ".cache/llm_responses",
            cache_only=_bool_env("PROFILE_LLM_CHILD_RESOLVER_CACHE_ONLY", False),
            max_concurrency=_int_env("PROFILE_LLM_CHILD_RESOLVER_MAX_CONCURRENCY", 4),
            max_retries=_int_env("PROFILE_LLM_CHILD_RESOLVER_MAX_RETRIES", 2),
            request_timeout_seconds=_float_env("PROFILE_LLM_CHILD_RESOLVER_TIMEOUT_SECONDS", 180.0),
            stoplist_path=os.getenv("PROFILE_LLM_CHILD_RESOLVER_STOPLIST", str(DEFAULT_STOPLIST_PATH)).strip()
            or str(DEFAULT_STOPLIST_PATH),
            codex_cli_command=os.getenv("CODEX_CLI_COMMAND", "codex").strip() or "codex",
            codex_home=os.getenv("PROFILE_LLM_CHILD_RESOLVER_CODEX_HOME") or None,
            project_root=os.getenv("PROFILE_LLM_CHILD_RESOLVER_PROJECT_ROOT", ".").strip() or ".",
            trace_path=os.getenv("PROFILE_LLM_CHILD_RESOLVER_TRACE_PATH") or None,
            name_diagnostics_path=os.getenv("PROFILE_LLM_CHILD_RESOLVER_NAME_DIAGNOSTICS_PATH") or None,
            escalation_enabled=_bool_env("PROFILE_LLM_CHILD_RESOLVER_ESCALATION", False),
            escalation_model=os.getenv("PROFILE_LLM_CHILD_RESOLVER_ESCALATION_MODEL", "gpt-5.5").strip()
            or "gpt-5.5",
            escalation_reasoning_effort=os.getenv("PROFILE_LLM_CHILD_RESOLVER_ESCALATION_REASONING", "high").strip()
            or "high",
            escalation_max_concurrency=_int_env("PROFILE_LLM_CHILD_RESOLVER_ESCALATION_MAX_CONCURRENCY", 2),
            escalation_request_timeout_seconds=_float_env(
                "PROFILE_LLM_CHILD_RESOLVER_ESCALATION_TIMEOUT_SECONDS",
                300.0,
            ),
        )


@dataclass(frozen=True)
class ChildMention:
    profile_id: str
    mention_id: str
    child_key: str
    child_name: str = ""
    grades: tuple[str, ...] = ()
    subjects: tuple[str, ...] = ()
    event_at: datetime | None = None
    brand: str = "unknown"
    source_ref: str = ""

    def prompt_payload(self) -> Mapping[str, Any]:
        return {
            "mention_id": self.mention_id,
            "name": self.child_name,
            "grade": "; ".join(self.grades),
            "subjects": list(self.subjects),
            "event_at": self.event_at.isoformat(timespec="seconds") if self.event_at else "",
            "evidence": self.source_ref,
        }


@dataclass(frozen=True)
class ChildResolverCase:
    profile_id: str
    profile_phone: str
    mentions: tuple[ChildMention, ...]
    original_child_keys: tuple[str, ...]

    @property
    def case_id(self) -> str:
        raw = json.dumps(
            {
                "profile_id": self.profile_id,
                "mentions": [mention.prompt_payload() for mention in self.mentions],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return "family_" + sha256(raw.encode("utf-8")).hexdigest()[:16]

    def prompt_input(self) -> Mapping[str, Any]:
        return {
            "case_id": self.case_id,
            "mentions": [mention.prompt_payload() for mention in self.mentions],
        }


@dataclass(frozen=True)
class ChildResolverFamilyResult:
    case_id: str
    profile_id: str
    accepted: bool
    mention_to_child_key: Mapping[str, str] = field(default_factory=dict)
    raw_response: Mapping[str, Any] = field(default_factory=dict)
    error_code: str = ""
    error_detail: str = ""
    cache_hit: bool = False
    llm_attempted: bool = False
    tier: str = "tier1"
    escalated: bool = False
    manual_review_reason: str = ""


@dataclass(frozen=True)
class ChildResolverApplyResult:
    fields: list[ProfileFieldCandidate]
    summary: Mapping[str, Any]


def apply_llm_child_resolver_to_fields(
    fields: Sequence[ProfileFieldCandidate],
    *,
    profile_phones: Mapping[str, str] | None = None,
    config: ChildResolverConfig | None = None,
    client: ChatCompletionClient | None = None,
    cache: LLMResponseCache | None = None,
) -> ChildResolverApplyResult:
    resolved_config = config or ChildResolverConfig.from_env()
    stoplist = load_shared_phone_stoplist(Path(resolved_config.stoplist_path), required=True)
    profile_phones = profile_phones or {}
    all_cases = build_child_resolver_cases(fields, profile_phones=profile_phones)
    scoped_cases = [case for case in all_cases if not phone_in_stoplist(case.profile_phone, stoplist)]
    skipped_shared = [case for case in all_cases if phone_in_stoplist(case.profile_phone, stoplist)]
    resolved_cache = cache or LLMResponseCache(enabled=resolved_config.cache_enabled, root_dir=resolved_config.cache_root_dir)
    family_results = resolve_child_families(
        scoped_cases,
        config=resolved_config,
        client=client,
        cache=resolved_cache,
    )
    tier1_cache_hits = sum(1 for result in family_results if result.cache_hit)
    tier1_llm_attempts = sum(1 for result in family_results if result.llm_attempted)
    escalation_summary: Mapping[str, Any] = {}
    if resolved_config.escalation_enabled:
        family_results, escalation_summary = apply_escalation_to_low_confidence_families(
            scoped_cases,
            family_results,
            config=resolved_config,
            client=client,
            cache=resolved_cache,
        )
    skipped_results = [
        ChildResolverFamilyResult(
            case_id=case.case_id,
            profile_id=case.profile_id,
            accepted=False,
            error_code="shared_phone_stoplist",
            error_detail="profile phone is in shared family stoplist",
        )
        for case in skipped_shared
    ]
    for case, result in zip(skipped_shared, skipped_results):
        write_trace_event(case, result, config=resolved_config)
    all_results = [*family_results, *skipped_results]
    result_by_profile = {result.profile_id: result for result in all_results}
    mention_lookup = {mention.mention_id: mention for case in all_cases for mention in case.mentions}
    field_mention_ids = field_mention_index(all_cases)

    rewritten: list[ProfileFieldCandidate] = []
    seen_ids: set[str] = set()
    rekeyed_fields = 0
    brand_changed = 0
    for field in fields:
        result = result_by_profile.get(field.profile_id)
        mention_id = field_mention_ids.get((field.profile_id, field.source_ref, field.child_key))
        target_child_key = result.mention_to_child_key.get(mention_id or "") if result and result.accepted else None
        original_brand = field.brand
        if target_child_key and target_child_key != field.child_key and field.field in CHILD_FIELDS:
            field = replace(field, child_key=target_child_key, field_id=None)
            rekeyed_fields += 1
        if field.brand != original_brand:
            brand_changed += 1
        if field.field_id in seen_ids:
            continue
        seen_ids.add(field.field_id or "")
        rewritten.append(field)

    profiles_with_2plus_before = sum(1 for case in all_cases if len(case.original_child_keys) >= 2)
    after_counts = child_slot_counts_after_resolution(fields, all_cases, result_by_profile, mention_lookup)
    profiles_with_2plus_after = sum(1 for count in after_counts.values() if count >= 2)
    accepted = [result for result in all_results if result.accepted]
    failed = [result for result in all_results if not result.accepted and result.error_code != "shared_phone_stoplist"]
    confidence_counts = merge_confidence_counts(all_results)

    summary = {
        "profiles_with_2plus_children_before": profiles_with_2plus_before,
        "profiles_with_2plus_children_after": profiles_with_2plus_after,
        "merge_candidate_groups": 0,
        "child_slots_marked_merge_candidate": 0,
        "child_slot_fields_rekeyed": rekeyed_fields,
        "merge_candidate_markers_written": 0,
        "llm_child_resolver_enabled": 1,
        "llm_child_resolver_namespace": NAMESPACE,
        "llm_child_resolver_prompt_version": resolved_config.prompt_version,
        "llm_child_resolver_provider": resolved_config.provider,
        "llm_child_resolver_model": resolved_config.model,
        "llm_cases_total": len(all_cases),
        "llm_calls_total": tier1_llm_attempts,
        "llm_cache_hits": tier1_cache_hits,
        "llm_cache_only": 1 if resolved_config.cache_only else 0,
        "llm_cache_misses_without_call": sum(
            1 for result in family_results if result.error_code == "cache_miss_cache_only"
        ),
        "llm_families_resolved": len(accepted),
        "llm_families_failed_soft": len(failed),
        "llm_families_skipped_shared_phone": len(skipped_shared),
        "llm_brand_changed_fields": brand_changed,
        "llm_input_mentions_total": sum(len(case.mentions) for case in all_cases),
        "llm_output_children_total": sum(len(set(result.mention_to_child_key.values())) for result in accepted),
        "llm_merge_confidence_high_children": confidence_counts["high"],
        "llm_merge_confidence_low_children": confidence_counts["low"],
        "llm_merge_confidence_missing_children": confidence_counts["missing"],
    }
    if resolved_config.escalation_enabled:
        summary.update(escalation_summary)
    return ChildResolverApplyResult(fields=rewritten, summary=summary)


def apply_escalation_to_low_confidence_families(
    cases: Sequence[ChildResolverCase],
    results: Sequence[ChildResolverFamilyResult],
    *,
    config: ChildResolverConfig,
    client: ChatCompletionClient | None,
    cache: LLMResponseCache,
) -> tuple[list[ChildResolverFamilyResult], Mapping[str, Any]]:
    tier2_config = replace(
        config,
        model=config.escalation_model,
        reasoning_effort=config.escalation_reasoning_effort,
        max_concurrency=config.escalation_max_concurrency,
        request_timeout_seconds=config.escalation_request_timeout_seconds,
        prompt_version=ESCALATION_PROMPT_VERSION,
        strict_confidence_prompt=True,
        resolver_tier="tier2",
        trace_path=None,
        name_diagnostics_path=None,
    )
    final_results: list[ChildResolverFamilyResult] = list(results)
    escalation_targets: list[tuple[int, ChildResolverCase]] = []
    counters = Counter()
    for index, (case, result) in enumerate(zip(cases, results)):
        reason = name_review_reason(case, result)
        joint_name = first_joint_child_name(case)
        if joint_name:
            final = replace(
                result,
                accepted=False,
                mention_to_child_key={},
                error_code="joint_mention",
                error_detail=f"joint child mention detected: {joint_name}"[:500],
                tier="tier3",
                manual_review_reason="joint_mention",
            )
            counters["llm_escalation_joint_mentions"] += 1
            if reason == "low_confidence_multi_named":
                counters["llm_escalation_candidates"] += 1
                counters["llm_escalation_joint_skipped_escalation"] += 1
            write_name_review_diagnostic(case, final, config=config)
            write_trace_event(case, final, config=config)
            final_results[index] = final
            continue
        if reason != "low_confidence_multi_named":
            continue
        counters["llm_escalation_candidates"] += 1
        escalation_targets.append((index, case))

    def worker(item: tuple[int, ChildResolverCase]) -> tuple[int, ChildResolverCase, ChildResolverFamilyResult]:
        index, case = item
        return index, case, resolve_child_family(case, config=tier2_config, client=client, cache=cache)

    with ThreadPoolExecutor(max_workers=max(1, int(config.escalation_max_concurrency))) as pool:
        escalated_items = list(pool.map(worker, escalation_targets))

    for index, case, escalated in escalated_items:
        counters["llm_escalation_cache_hits"] += 1 if escalated.cache_hit else 0
        counters["llm_escalation_calls_total"] += 1 if escalated.llm_attempted else 0
        if escalated.accepted and result_all_confidence(escalated, "high"):
            final = replace(escalated, escalated=True)
            counters["llm_escalation_resolved_high"] += 1
            write_name_review_diagnostic(case, final, config=config)
            write_trace_event(case, final, config=config)
            final_results[index] = final
            continue

        if escalated.accepted:
            final = replace(
                escalated,
                accepted=False,
                mention_to_child_key={},
                error_code="escalation_low_confidence",
                error_detail="gpt-5.5 high escalation returned low confidence",
                tier="tier3",
                escalated=True,
                manual_review_reason="escalation_low_confidence",
            )
            counters["llm_escalation_still_low"] += 1
        else:
            final = replace(
                escalated,
                mention_to_child_key={},
                tier="tier3",
                escalated=True,
                manual_review_reason="escalation_validation_failed",
            )
            counters["llm_escalation_failed_verification"] += 1
        write_name_review_diagnostic(case, final, config=config)
        write_trace_event(case, final, config=config)
        final_results[index] = final

    return final_results, {
        "llm_escalation_enabled": 1,
        "llm_escalation_model": config.escalation_model,
        "llm_escalation_reasoning": config.escalation_reasoning_effort,
        "llm_escalation_max_concurrency": config.escalation_max_concurrency,
        "llm_escalation_timeout_seconds": config.escalation_request_timeout_seconds,
        "llm_escalation_prompt_version": ESCALATION_PROMPT_VERSION,
        "llm_escalation_candidates": counters["llm_escalation_candidates"],
        "llm_escalation_calls_total": counters["llm_escalation_calls_total"],
        "llm_escalation_cache_hits": counters["llm_escalation_cache_hits"],
        "llm_escalation_resolved_high": counters["llm_escalation_resolved_high"],
        "llm_escalation_still_low": counters["llm_escalation_still_low"],
        "llm_escalation_failed_verification": counters["llm_escalation_failed_verification"],
        "llm_escalation_joint_mentions": counters["llm_escalation_joint_mentions"],
        "llm_escalation_joint_skipped_escalation": counters["llm_escalation_joint_skipped_escalation"],
        "llm_escalation_manual_review": (
            counters["llm_escalation_still_low"]
            + counters["llm_escalation_failed_verification"]
            + counters["llm_escalation_joint_mentions"]
        ),
    }


def build_child_resolver_cases(
    fields: Sequence[ProfileFieldCandidate],
    *,
    profile_phones: Mapping[str, str] | None = None,
) -> list[ChildResolverCase]:
    profile_phones = profile_phones or {}
    grouped: dict[tuple[str, str, str], list[ProfileFieldCandidate]] = {}
    child_keys_by_profile: dict[str, set[str]] = {}
    names_by_profile_key: dict[tuple[str, str], set[str]] = {}
    for field in fields:
        if not field.child_key or field.field not in CHILD_FIELDS:
            continue
        grouped.setdefault((field.profile_id, field.source_ref, field.child_key), []).append(field)
        child_keys_by_profile.setdefault(field.profile_id, set()).add(field.child_key)
        if field.field == "child_name":
            names_by_profile_key.setdefault((field.profile_id, field.child_key), set()).add(field.value)

    mentions_by_profile: dict[str, list[ChildMention]] = {}
    for (profile_id, source_ref, child_key), items in grouped.items():
        names = unique_texts(field.value for field in items if field.field == "child_name")
        grades = tuple(unique_texts(field.value for field in items if field.field == "grade"))
        subjects = tuple(
            unique_texts(part.strip() for field in items if field.field == "subject" for part in field.value.split(";"))
        )
        event_at = max((field.event_at for field in items), default=None)
        brand = latest_text((field.event_at, field.brand) for field in items)
        mention_id = stable_mention_id(profile_id=profile_id, source_ref=source_ref, child_key=child_key)
        mentions_by_profile.setdefault(profile_id, []).append(
            ChildMention(
                profile_id=profile_id,
                mention_id=mention_id,
                child_key=child_key,
                child_name=names[0] if names else "",
                grades=grades,
                subjects=subjects,
                event_at=event_at,
                brand=brand or "unknown",
                source_ref=source_ref,
            )
        )

    cases: list[ChildResolverCase] = []
    for profile_id, mentions in mentions_by_profile.items():
        child_keys = tuple(sorted(child_keys_by_profile.get(profile_id, set()), key=child_key_sort))
        has_unnamed = any(not names_by_profile_key.get((profile_id, child_key)) for child_key in child_keys)
        if len(child_keys) < 2 and not has_unnamed:
            continue
        cases.append(
            ChildResolverCase(
                profile_id=profile_id,
                profile_phone=str(profile_phones.get(profile_id) or ""),
                mentions=tuple(sorted(mentions, key=lambda item: (item.event_at or datetime.min.replace(tzinfo=timezone.utc), item.source_ref, item.child_key))),
                original_child_keys=child_keys,
            )
        )
    return sorted(cases, key=lambda case: case.case_id)


def resolve_child_families(
    cases: Sequence[ChildResolverCase],
    *,
    config: ChildResolverConfig,
    client: ChatCompletionClient | None = None,
    cache: LLMResponseCache,
) -> list[ChildResolverFamilyResult]:
    case_list = list(cases)
    if not case_list:
        return []

    def worker(case: ChildResolverCase) -> ChildResolverFamilyResult:
        return resolve_child_family(case, config=config, client=client, cache=cache)

    with ThreadPoolExecutor(max_workers=max(1, int(config.max_concurrency))) as pool:
        return list(pool.map(worker, case_list))


def resolve_child_family(
    case: ChildResolverCase,
    *,
    config: ChildResolverConfig,
    client: ChatCompletionClient | None = None,
    cache: LLMResponseCache,
) -> ChildResolverFamilyResult:
    prompt = build_child_resolver_prompt(case, strict_confidence=config.strict_confidence_prompt)
    reasoning = resolver_reasoning(config)
    cached = cache.get(
        namespace=NAMESPACE,
        provider=config.provider,
        model=config.model,
        reasoning=reasoning,
        prompt_version=config.prompt_version,
        prompt=prompt,
    )
    cache_hit = cached is not None
    llm_attempted = False
    response_payload: Mapping[str, Any] = cached or {}
    normalized_payload: Mapping[str, Any] | None = None
    try:
        if cached is None and config.cache_only:
            raise ChildResolverError("cache_miss_cache_only")
        llm_attempted = cached is None
        response_payload = cached if cached is not None else call_llm_json(prompt, config=config, client=client)
        normalized_payload = normalize_child_resolver_response(response_payload)
        result = validate_child_resolver_response(case, normalized_payload, cache_hit=cache_hit)
        result = replace(result, tier=config.resolver_tier, llm_attempted=llm_attempted)
        if not cache_hit:
            cache.put(
                namespace=NAMESPACE,
                provider=config.provider,
                model=config.model,
                reasoning=reasoning,
                prompt_version=config.prompt_version,
                prompt=prompt,
                response=normalized_payload,
            )
        write_name_review_diagnostic(case, result, config=config)
        write_trace_event(case, result, config=config)
        return result
    except Exception as exc:  # noqa: BLE001 - fail-soft is an explicit TZ-100 requirement.
        result_payload = normalized_payload if isinstance(normalized_payload, Mapping) else response_payload
        result = ChildResolverFamilyResult(
            case_id=case.case_id,
            profile_id=case.profile_id,
            accepted=False,
            raw_response=dict(result_payload) if isinstance(result_payload, Mapping) else {},
            error_code=child_resolver_error_code(exc),
            error_detail=str(exc)[:500],
            cache_hit=cache_hit,
            llm_attempted=llm_attempted,
            tier=config.resolver_tier,
        )
        write_name_review_diagnostic(case, result, config=config, exc=exc)
        write_trace_event(case, result, config=config)
        return result


def build_child_resolver_prompt(case: ChildResolverCase, *, strict_confidence: bool = False) -> str:
    payload_json = json.dumps(case.prompt_input(), ensure_ascii=False, sort_keys=True, indent=2)
    confidence_rule = (
        "- Для каждого ребёнка выставь merge_confidence строго одним из двух значений: high или low. "
        "Никакие другие значения не разрешены.\n"
        if strict_confidence
        else "- Для каждого ребёнка выставь merge_confidence: high, если упоминания явно про одного ребёнка; "
        "low, если есть сомнение, разные имена, слабая связь или безымянные упоминания.\n"
    )
    return (
        "Ты помогаешь собрать семейную карточку учебного центра. "
        "Нужно понять, сколько реальных детей есть в семье, по уже извлечённым упоминаниям.\n\n"
        "Правила:\n"
        "- Идентичность ребёнка по именам решаешь ты по смыслу, не по строковому совпадению.\n"
        "- Сливай ФИО, имя, уменьшительное, опечатки и ASR-варианты одного ребёнка.\n"
        "- Разных детей, включая братьев/сестёр с общей фамилией, но разными именами, оставляй раздельно.\n"
        "- Класс может расти со временем, например 9 -> 11 за длительный период может быть один ребёнок.\n"
        "- Не выдумывай имена: canonical_name должен быть ровно одним из входных name; "
        "name_variants должны перечислять только входные написания этой группы.\n"
        "- Не отбрасывай упоминания: каждый mention_id должен быть ровно у одного ребёнка.\n"
        "- Если у ребёнка нет имени во входе, canonical_name верни пустой строкой.\n\n"
        f"{confidence_rule}"
        "- merge_confidence только отражает твою уверенность; оно не меняет формат ответа.\n\n"
        "Верни строго JSON object без markdown:\n"
        "{\n"
        "  \"children\": [\n"
        "    {\n"
        "      \"child_id\": \"child_1\",\n"
        "      \"canonical_name\": \"\",\n"
        "      \"name_variants\": [],\n"
        "      \"grades\": [],\n"
        "      \"subjects\": [],\n"
        "      \"mention_ids\": [],\n"
        "      \"merge_confidence\": \"high\"\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "INPUT_JSON:\n"
        f"{payload_json}\n"
    )


def call_llm_json(
    prompt: str,
    *,
    config: ChildResolverConfig,
    client: ChatCompletionClient | None = None,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(max(1, int(config.max_retries))):
        try:
            provider = str(config.provider or "").strip().lower()
            if provider == PROVIDER_OPENAI:
                return call_openai_json(prompt, config=config, client=client)
            if provider == PROVIDER_CODEX_CLI:
                return call_codex_cli_json(prompt, config=config)
            raise ChildResolverError(f"unsupported child resolver provider: {config.provider}")
        except Exception as exc:  # noqa: BLE001 - caller converts to family fail-soft.
            last_error = exc
            if attempt >= max(1, int(config.max_retries)) - 1:
                break
            time.sleep(max(0.0, float(config.retry_initial_seconds)) * (2**attempt))
    raise ChildResolverError(f"child resolver LLM failed: {last_error}") from last_error


def call_openai_json(
    prompt: str,
    *,
    config: ChildResolverConfig,
    client: ChatCompletionClient | None = None,
) -> dict[str, Any]:
    resolved_client = client or openai_client(config)
    response = resolved_client.chat.completions.create(
        model=config.model,
        temperature=config.temperature,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )
    content = response.choices[0].message.content if getattr(response, "choices", None) else None
    if not content:
        raise ChildResolverError("child resolver returned empty content")
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise ChildResolverError("child resolver response must be a JSON object")
    return payload


def call_codex_cli_json(prompt: str, *, config: ChildResolverConfig) -> dict[str, Any]:
    codex_bin = (config.codex_cli_command or "codex").strip() or "codex"
    if shutil.which(codex_bin) is None:
        raise ChildResolverError(f"codex binary is not available: {codex_bin}")
    with tempfile.NamedTemporaryFile(prefix="child_resolver_", suffix=".json") as out_file, tempfile.NamedTemporaryFile(
        prefix="child_resolver_schema_",
        suffix=".json",
        mode="w",
        encoding="utf-8",
    ) as schema_file:
        schema_file.write(json.dumps(child_resolver_output_json_schema(), ensure_ascii=False))
        schema_file.flush()
        cmd = build_codex_cli_command(config, output_path=Path(out_file.name), schema_path=Path(schema_file.name))
        proc = subprocess.run(
            cmd,
            input=(
                "Верни строго один JSON object по схеме. Не используй markdown. "
                "Не запускай shell-команды, работай только с текстом задания.\n\n"
                + prompt
            ),
            capture_output=True,
            text=True,
            check=False,
            timeout=max(30, int(config.request_timeout_seconds)),
            env=codex_cli_env(config),
        )
        raw = Path(out_file.name).read_text(encoding="utf-8", errors="ignore").strip()
    for candidate in (raw, proc.stdout or "", proc.stderr or ""):
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            return extract_json_object(candidate)
        except Exception:
            continue
    stderr_tail = (proc.stderr or "").strip().splitlines()[-1:] or [""]
    raise ChildResolverError(f"codex exec returned no JSON; rc={proc.returncode}; stderr_tail={stderr_tail[0]}")


def build_codex_cli_command(config: ChildResolverConfig, *, output_path: Path, schema_path: Path) -> list[str]:
    cmd = [
        (config.codex_cli_command or "codex").strip() or "codex",
        "exec",
        "--skip-git-repo-check",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--sandbox",
        "read-only",
        "--cd",
        str(Path(config.project_root).expanduser().resolve(strict=False)),
        "--model",
        config.model,
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(output_path),
    ]
    append_codex_service_tier(cmd)
    reasoning = str(config.reasoning_effort or "").strip().lower()
    if reasoning:
        cmd.extend(["-c", f'model_reasoning_effort="{reasoning}"'])
    cmd.append("-")
    return cmd


def codex_cli_env(config: ChildResolverConfig) -> dict[str, str]:
    env = os.environ.copy()
    if config.codex_home is not None:
        env["CODEX_HOME"] = str(Path(config.codex_home).expanduser().resolve(strict=False))
    elif not env.get("CODEX_HOME"):
        env["CODEX_HOME"] = str(prepare_default_codex_home())
    return env


def prepare_default_codex_home() -> Path:
    target = Path("/private/tmp/mango_child_resolver_codex_home")
    target.mkdir(parents=True, exist_ok=True)
    (target / "sessions").mkdir(parents=True, exist_ok=True)
    source = Path.home() / ".codex"
    for name in ("auth.json", "config.toml", "installation_id", "models_cache.json", ".codex-global-state.json"):
        src = source / name
        dst = target / name
        copy_file_if_fresher(src, dst)
    return target


def copy_file_if_fresher(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    try:
        should_copy = not dst.exists()
        if not should_copy:
            should_copy = src.stat().st_mtime_ns > dst.stat().st_mtime_ns or src.stat().st_size != dst.stat().st_size
        if should_copy:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    except OSError:
        return


def openai_client(config: ChildResolverConfig) -> ChatCompletionClient:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ChildResolverError("OPENAI_API_KEY is required for child resolver openai provider")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ChildResolverError("openai package is required for child resolver openai provider") from exc
    return OpenAI(api_key=api_key, timeout=config.request_timeout_seconds)


def normalize_child_resolver_response(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ChildResolverError("response must be a JSON object")
    if set(payload.keys()) - ROOT_RESPONSE_KEYS:
        raise ChildResolverError("response has unexpected fields")
    children_raw = payload.get("children")
    if not isinstance(children_raw, list):
        raise ChildResolverError("response.children must be a list")
    children: list[dict[str, Any]] = []
    for index, item in enumerate(children_raw, start=1):
        if not isinstance(item, Mapping):
            raise ChildResolverError("each child must be a JSON object")
        item_keys = set(item.keys())
        if CHILD_REQUIRED_RESPONSE_KEYS - item_keys:
            raise ChildResolverError("response child has missing required fields")
        if item_keys - CHILD_RESPONSE_KEYS:
            raise ChildResolverError("response child has unexpected fields")
        mention_ids = clean_string_list(item.get("mention_ids"))
        children.append(
            {
                "child_id": str(item.get("child_id") or f"child_{index}").strip() or f"child_{index}",
                "canonical_name": str(item.get("canonical_name") or "").strip(),
                "name_variants": clean_name_variant_list(item.get("name_variants")),
                "grades": clean_string_list(item.get("grades")),
                "subjects": clean_string_list(item.get("subjects")),
                "mention_ids": mention_ids,
                "merge_confidence": normalize_merge_confidence(item.get("merge_confidence")),
            }
        )
    return {"children": children}


def validate_child_resolver_response(
    case: ChildResolverCase,
    payload: Mapping[str, Any],
    *,
    cache_hit: bool,
) -> ChildResolverFamilyResult:
    mentions_by_id = {mention.mention_id: mention for mention in case.mentions}
    children = payload.get("children") if isinstance(payload.get("children"), list) else []
    if not children:
        raise ChildResolverError("response has no children")
    if len(children) > len(case.mentions):
        raise ChildResolverError("children_count_exceeds_mentions")

    input_name_norms = {normalize_full_name(mention.child_name) for mention in case.mentions if mention.child_name}
    input_name_norms.discard("")
    input_name_spellings = {input_name_key(mention.child_name) for mention in case.mentions if mention.child_name}
    input_name_spellings.discard("")
    has_unnamed = any(not mention.child_name for mention in case.mentions)
    max_children = len(input_name_norms) + (1 if has_unnamed else 0)
    if len(children) > max(1, max_children):
        raise ChildResolverError("children_count_exceeds_name_bound")

    seen_mentions: set[str] = set()
    mention_to_child_key: dict[str, str] = {}
    for unnamed_index, child in enumerate(children, start=1):
        if not isinstance(child, Mapping):
            raise ChildResolverError("child item is not object")
        child_name_values = [str(child.get("canonical_name") or "").strip(), *clean_string_list(child.get("name_variants"))]
        child_name_spellings = {input_name_key(value) for value in child_name_values if value}
        child_name_spellings.discard("")
        invented = child_name_spellings - input_name_spellings
        if invented:
            raise ChildResolverError("invented_child_name")
        mention_ids = clean_string_list(child.get("mention_ids"))
        if not mention_ids:
            raise ChildResolverError("child_with_empty_mention_ids")
        for mention_id in mention_ids:
            if mention_id not in mentions_by_id:
                raise ChildResolverError("unknown_mention_id")
            if mention_id in seen_mentions:
                raise ChildResolverError("duplicate_mention_id")
            seen_mentions.add(mention_id)
        target_key = target_child_key_for_child(child, unnamed_index=unnamed_index)
        for mention_id in mention_ids:
            mention_to_child_key[mention_id] = target_key

    if seen_mentions != set(mentions_by_id):
        raise ChildResolverError("incomplete_mention_mapping")
    reject_incompatible_grades(case, payload)
    return ChildResolverFamilyResult(
        case_id=case.case_id,
        profile_id=case.profile_id,
        accepted=True,
        mention_to_child_key=mention_to_child_key,
        raw_response=dict(payload),
        cache_hit=cache_hit,
    )


def reject_incompatible_grades(case: ChildResolverCase, payload: Mapping[str, Any]) -> None:
    mentions_by_id = {mention.mention_id: mention for mention in case.mentions}
    for child in payload.get("children", []):
        if not isinstance(child, Mapping):
            continue
        mentions = [mentions_by_id[item] for item in clean_string_list(child.get("mention_ids")) if item in mentions_by_id]
        parsed: list[tuple[ChildMention, int]] = []
        for mention in mentions:
            for grade_text in mention.grades:
                grade = parse_single_grade(grade_text)
                if grade is not None:
                    parsed.append((mention, grade))
        for left_index, (left, left_grade) in enumerate(parsed):
            for right, right_grade in parsed[left_index + 1 :]:
                if abs(left_grade - right_grade) < 3:
                    continue
                if left.event_at is None or right.event_at is None:
                    raise ChildResolverError("incompatible_grade_without_dates")
                days = abs((left.event_at - right.event_at).days)
                if days <= 200:
                    raise ChildResolverError("incompatible_grades_same_period")


def target_child_key_for_child(child: Mapping[str, Any], *, unnamed_index: int) -> str:
    mention_ids = sorted(clean_string_list(child.get("mention_ids")))
    if mention_ids:
        raw = json.dumps(mention_ids, ensure_ascii=False, separators=(",", ":"))
        return f"child_{stable_hash(raw)}"
    return f"child_unnamed_{unnamed_index}"


def child_slot_counts_after_resolution(
    fields: Sequence[ProfileFieldCandidate],
    cases: Sequence[ChildResolverCase],
    result_by_profile: Mapping[str, ChildResolverFamilyResult],
    mention_lookup: Mapping[str, ChildMention],
) -> dict[str, int]:
    field_mention_ids = field_mention_index(cases)
    profile_slots: dict[str, set[str]] = {}
    for field in fields:
        if not field.child_key or field.field not in CHILD_FIELDS:
            continue
        child_key = field.child_key
        result = result_by_profile.get(field.profile_id)
        mention_id = field_mention_ids.get((field.profile_id, field.source_ref, field.child_key))
        if result and result.accepted and mention_id in result.mention_to_child_key and mention_id in mention_lookup:
            child_key = result.mention_to_child_key[mention_id]
        profile_slots.setdefault(field.profile_id, set()).add(child_key)
    return {profile_id: len(slots) for profile_id, slots in profile_slots.items()}


def field_mention_index(cases: Sequence[ChildResolverCase]) -> dict[tuple[str, str, str], str]:
    return {
        (mention.profile_id, mention.source_ref, mention.child_key): mention.mention_id
        for case in cases
        for mention in case.mentions
    }


def load_shared_phone_stoplist(path: Path, *, required: bool) -> set[str]:
    target = path.expanduser()
    if not target.exists():
        if required:
            raise ChildResolverError(f"shared phone stoplist is required but missing: {target}")
        return set()
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ChildResolverError(f"shared phone stoplist is invalid JSON: {target}") from exc
    raw = payload.get("phones") if isinstance(payload, Mapping) else payload
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise ChildResolverError("shared phone stoplist must contain a phones list")
    phones = {normalize_phone(item) for item in raw}
    phones.discard("")
    if required and not phones:
        raise ChildResolverError("shared phone stoplist is empty")
    return phones


def phone_in_stoplist(phone: str, stoplist: set[str]) -> bool:
    normalized = normalize_phone(phone)
    if not normalized:
        return False
    return normalized in stoplist or any(normalized[-10:] == item[-10:] for item in stoplist if len(item) >= 10)


def child_resolver_output_json_schema() -> dict[str, Any]:
    child_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "child_id",
            "canonical_name",
            "name_variants",
            "grades",
            "subjects",
            "mention_ids",
            "merge_confidence",
        ],
        "properties": {
            "child_id": {"type": "string"},
            "canonical_name": {"type": "string"},
            "name_variants": {"type": "array", "items": {"type": "string"}},
            "grades": {"type": "array", "items": {"type": "string"}},
            "subjects": {"type": "array", "items": {"type": "string"}},
            "mention_ids": {"type": "array", "items": {"type": "string"}},
            "merge_confidence": {"type": "string"},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["children"],
        "properties": {"children": {"type": "array", "items": child_schema}},
    }


def extract_json_object(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("empty JSON text")
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("JSON object not found")
    payload = json.loads(raw[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("JSON object not found")
    return payload


def stable_mention_id(*, profile_id: str, source_ref: str, child_key: str) -> str:
    raw = json.dumps(
        {"profile_id": profile_id, "source_ref": source_ref, "child_key": child_key},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "m_" + sha256(raw.encode("utf-8")).hexdigest()[:16]


def stable_hash(value: str) -> str:
    return sha256(str(value or "").encode("utf-8")).hexdigest()[:8]


def resolver_reasoning(config: ChildResolverConfig) -> str:
    if config.provider == PROVIDER_OPENAI:
        return f"temperature={config.temperature}"
    return str(config.reasoning_effort or "").strip().lower()


def parse_single_grade(value: str) -> int | None:
    text = str(value or "").lower().replace("ё", "е")
    numbers = [int(item) for item in re.findall(r"(?<!\d)(1[01]|[1-9])(?!\d)", text)]
    unique = sorted(set(num for num in numbers if 1 <= num <= 11))
    if len(unique) != 1:
        return None
    return unique[0]


def normalize_full_name(value: str) -> str:
    return " ".join(re.findall(r"[a-zа-яё]+", str(value or "").lower().replace("ё", "е"), flags=re.IGNORECASE))


def input_name_key(value: str) -> str:
    return str(value or "").strip().casefold()


def normalize_merge_confidence(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text if text in MERGE_CONFIDENCE_VALUES else "low"


def clean_name_variant_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item or "").strip()
        key = text.casefold()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def clean_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item or "").strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def unique_texts(values: Sequence[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def latest_text(values: Sequence[tuple[datetime, str]]) -> str:
    ordered = sorted((item for item in values if item[1]), key=lambda item: item[0])
    return ordered[-1][1] if ordered else ""


def child_key_sort(value: str) -> tuple[int, int | str]:
    match = re.fullmatch(r"child_(\d+)", str(value or ""))
    if match:
        return (0, int(match.group(1)))
    return (1, str(value or ""))


def child_resolver_error_code(exc: Exception) -> str:
    if isinstance(exc, ChildResolverError):
        text = str(exc)
        if re.fullmatch(r"[a-z0-9_]+", text):
            return text
    return type(exc).__name__


def merge_confidence_counts(results: Sequence[ChildResolverFamilyResult]) -> dict[str, int]:
    counts = {"high": 0, "low": 0, "missing": 0}
    for result in results:
        children = result.raw_response.get("children") if isinstance(result.raw_response, Mapping) else []
        if not isinstance(children, list):
            continue
        for child in children:
            if not isinstance(child, Mapping):
                continue
            confidence = str(child.get("merge_confidence") or "").strip().lower()
            if confidence in ("high", "low"):
                counts[confidence] += 1
            else:
                counts["missing"] += 1
    return counts


def result_all_confidence(result: ChildResolverFamilyResult, expected: str) -> bool:
    children = result.raw_response.get("children") if isinstance(result.raw_response, Mapping) else []
    values = [
        normalize_merge_confidence(child.get("merge_confidence"))
        for child in children
        if isinstance(child, Mapping)
    ]
    return bool(values) and all(value == expected for value in values)


def first_joint_child_name(case: ChildResolverCase) -> str:
    for mention in case.mentions:
        if child_name_has_joint_mention(mention.child_name):
            return mention.child_name
    return ""


def child_name_has_joint_mention(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    parts = [part for part in JOINT_SEPARATOR_RE.split(text) if part.strip()]
    if len(parts) < 2:
        return False
    for left, right in zip(parts, parts[1:]):
        if contains_first_name_hint(left) and contains_first_name_hint(right):
            return True
    return False


def contains_first_name_hint(value: str) -> bool:
    tokens = re.findall(r"[a-zа-яё]+", str(value or "").lower().replace("ё", "е"), flags=re.IGNORECASE)
    return any(token in JOINT_NAME_HINTS for token in tokens)


_TRACE_LOCK = threading.Lock()
_NAME_DIAGNOSTIC_LOCK = threading.Lock()


def write_trace_event(case: ChildResolverCase, result: ChildResolverFamilyResult, *, config: ChildResolverConfig) -> None:
    if not config.trace_path:
        return
    path = Path(config.trace_path).expanduser()
    event = anonymized_trace_event(case, result)
    try:
        with _TRACE_LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
    except OSError:
        return


def anonymized_trace_event(case: ChildResolverCase, result: ChildResolverFamilyResult) -> Mapping[str, Any]:
    name_map: dict[str, str] = {}

    def anonymize_name(value: str) -> str:
        norm = normalize_full_name(value)
        if not norm:
            return ""
        if norm not in name_map:
            name_map[norm] = f"child_name_{len(name_map) + 1}"
        return name_map[norm]

    input_mentions = []
    for mention in case.mentions:
        input_mentions.append(
            {
                "mention_id": mention.mention_id,
                "name": anonymize_name(mention.child_name),
                "has_name": bool(mention.child_name),
                "grades": list(mention.grades),
                "subjects": list(mention.subjects),
                "brand": mention.brand,
                "event_at": mention.event_at.isoformat(timespec="seconds") if mention.event_at else "",
            }
        )
    children = []
    for child in result.raw_response.get("children", []) if isinstance(result.raw_response, Mapping) else []:
        if not isinstance(child, Mapping):
            continue
        children.append(
            {
                "child_id": str(child.get("child_id") or ""),
                "canonical_name": anonymize_name(str(child.get("canonical_name") or "")),
                "name_variants": [anonymize_name(item) for item in clean_string_list(child.get("name_variants"))],
                "grades": clean_string_list(child.get("grades")),
                "subjects": clean_string_list(child.get("subjects")),
                "mention_ids": clean_string_list(child.get("mention_ids")),
                "merge_confidence": normalize_merge_confidence(child.get("merge_confidence")),
            }
        )
    event = {
        "case_id": case.case_id,
        "profile_hash": stable_hash(case.profile_id)[:12],
        "accepted": result.accepted,
        "error_code": result.error_code,
        "error_detail": result.error_detail,
        "cache_hit": result.cache_hit,
        "llm_attempted": result.llm_attempted,
        "input_mentions": input_mentions,
        "model_children": children,
        "applied_child_keys": dict(sorted(result.mention_to_child_key.items())),
    }
    if result.tier != "tier1":
        event["tier"] = result.tier
    if result.escalated:
        event["escalated"] = True
    if result.manual_review_reason:
        event["manual_review_reason"] = result.manual_review_reason
    return event


def write_name_review_diagnostic(
    case: ChildResolverCase,
    result: ChildResolverFamilyResult,
    *,
    config: ChildResolverConfig,
    exc: Exception | None = None,
) -> None:
    reason = name_review_reason(case, result, exc)
    if not reason:
        return
    path = name_diagnostics_path(config)
    if path is None:
        return
    event = name_review_diagnostic_event(case, result, reason=reason, exc=exc)
    try:
        with _NAME_DIAGNOSTIC_LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
    except OSError:
        return


def name_diagnostics_path(config: ChildResolverConfig) -> Path | None:
    if config.name_diagnostics_path:
        return Path(config.name_diagnostics_path).expanduser()
    if config.trace_path:
        return Path(config.trace_path).expanduser().with_name("name_review_diagnostics.local.jsonl")
    return None


def name_review_reason(
    case: ChildResolverCase,
    result: ChildResolverFamilyResult,
    exc: Exception | None = None,
) -> str:
    if result.manual_review_reason:
        return result.manual_review_reason
    if exc is not None and child_resolver_error_code(exc) in {"invented_child_name", "children_count_exceeds_name_bound"}:
        return child_resolver_error_code(exc)
    if not result.accepted:
        return ""
    children = result.raw_response.get("children") if isinstance(result.raw_response, Mapping) else []
    child_items = children if isinstance(children, list) else []
    has_low = any(
        isinstance(child, Mapping) and normalize_merge_confidence(child.get("merge_confidence")) == "low"
        for child in child_items
    )
    has_two_named_roots = len({mention.child_key for mention in case.mentions if mention.child_name}) >= 2
    has_multi_named_merge = any(child_named_root_count(case, child) >= 2 for child in child_items if isinstance(child, Mapping))
    if has_low and has_two_named_roots:
        return "low_confidence_multi_named"
    if has_multi_named_merge:
        return "accepted_multi_named_merge"
    return ""


def child_named_root_count(case: ChildResolverCase, child: Mapping[str, Any]) -> int:
    mentions_by_id = {mention.mention_id: mention for mention in case.mentions}
    mention_ids = clean_string_list(child.get("mention_ids"))
    return len({mentions_by_id[mention_id].child_key for mention_id in mention_ids if mention_id in mentions_by_id and mentions_by_id[mention_id].child_name})


def name_review_diagnostic_event(
    case: ChildResolverCase,
    result: ChildResolverFamilyResult,
    *,
    reason: str,
    exc: Exception | None = None,
) -> Mapping[str, Any]:
    children = result.raw_response.get("children") if isinstance(result.raw_response, Mapping) else []
    model_children = []
    for child in children if isinstance(children, list) else []:
        if not isinstance(child, Mapping):
            continue
        model_children.append(
            {
                "child_id": str(child.get("child_id") or ""),
                "canonical_name": str(child.get("canonical_name") or ""),
                "name_variants": clean_string_list(child.get("name_variants")),
                "mention_ids": clean_string_list(child.get("mention_ids")),
                "merge_confidence": normalize_merge_confidence(child.get("merge_confidence")),
                "named_root_count": child_named_root_count(case, child),
            }
        )
    return {
        "case_id": case.case_id,
        "profile_hash": stable_hash(case.profile_id)[:12],
        "reason": reason,
        "accepted": result.accepted,
        "error_code": child_resolver_error_code(exc) if exc else result.error_code,
        "error_detail": (str(exc) if exc else result.error_detail)[:500],
        "diagnostics": getattr(exc, "diagnostics", {}) if exc else {},
        "grouped_name_spellings": grouped_name_spellings(case),
        "input_mentions": [
            {
                "mention_id": mention.mention_id,
                "child_key": mention.child_key,
                "name": mention.child_name,
                "name_norm": normalize_full_name(mention.child_name),
                "grades": list(mention.grades),
                "subjects": list(mention.subjects),
                "brand": mention.brand,
                "event_at": mention.event_at.isoformat(timespec="seconds") if mention.event_at else "",
            }
            for mention in case.mentions
        ],
        "model_children": model_children,
    }


def grouped_name_spellings(case: ChildResolverCase) -> list[Mapping[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for mention in case.mentions:
        norm = normalize_full_name(mention.child_name)
        if not norm:
            continue
        group = grouped.setdefault(norm, {"name_norm": norm, "spellings": set(), "mention_ids": [], "child_keys": set()})
        group["spellings"].add(mention.child_name)
        group["mention_ids"].append(mention.mention_id)
        group["child_keys"].add(mention.child_key)
    return [
        {
            "name_norm": norm,
            "spellings": sorted(group["spellings"]),
            "mention_ids": list(group["mention_ids"]),
            "child_keys": sorted(group["child_keys"]),
        }
        for norm, group in sorted(grouped.items())
    ]


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default
