#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import date as real_date
from datetime import datetime as real_datetime
from datetime import timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


SCHEMA_VERSION = "subscription_llm_equivalence_replay_v1_2026_06_11"
DEFAULT_CASES = Path("D1_audit_backlog/subscription_llm_refactor_cases_2026-06-11.jsonl")
DEFAULT_BASELINE = Path("D1_audit_backlog/subscription_llm_refactor_baseline_2026-06-11.jsonl")
DEFAULT_UNEXPECTED_FALLBACK_ERRORS = {
    "codex_exec_failed",
    "codex_binary_not_found",
    "codex_retryable_error",
    "empty_prompt",
    "direct_path_error",
    "fake_provider_default",
    "invalid_json",
    "invalid_json_or_codex_error",
    "timeout",
}
DEFAULT_UNEXPECTED_METADATA_REASONS = {
    "classifier_error",
    "classifier_invalid_payload",
    "classifier_unavailable",
    "classifier_unsupported_route",
    "direct_path_error",
    "empty_selection",
    "hard_verification_failed",
    "invalid_json",
    "no_candidates",
    "no_draft_fn",
    "provider_runtime",
    "retriever_fn_missing",
    "runtime_error",
    "semantic_check_unavailable",
    "semantic_verifier_downgrade",
    "semantic_verifier_unavailable",
    "timeout",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


STATE_TIME_KEYS = {"updated_at"}
SLOT_HISTORY_TIME_KEYS = {
    "at",
    "created_at",
    "recorded_at",
    "started_at",
    "timestamp",
    "ts",
    "updated_at",
}


def canonicalize_for_hash(value: Any, *, path: tuple[str, ...] = ()) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            next_path = (*path, key_text)
            if key_text in {"snapshot_path", "knowledge_snapshot_path", "kb_snapshot_path"}:
                result[key_text] = "<snapshot_path>"
            elif key_text in STATE_TIME_KEYS:
                result[key_text] = "<state_time>"
            elif "slot_history" in path and (key_text in SLOT_HISTORY_TIME_KEYS or key_text.endswith("_at")):
                result[key_text] = "<slot_history_time>"
            else:
                result[key_text] = canonicalize_for_hash(item, path=next_path)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [canonicalize_for_hash(item, path=(*path, "[]")) for item in value]
    return value


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=json_default) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"missing JSONL: {path}")
    result = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            try:
                result.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: {exc}") from exc
    return result


@contextlib.contextmanager
def clean_telegram_env() -> Any:
    saved = {key: value for key, value in os.environ.items() if key.startswith("TELEGRAM_")}
    try:
        for key in list(os.environ):
            if key.startswith("TELEGRAM_"):
                os.environ.pop(key, None)
        yield
    finally:
        for key in list(os.environ):
            if key.startswith("TELEGRAM_"):
                os.environ.pop(key, None)
        os.environ.update(saved)


class ExplodingDate:
    @staticmethod
    def today() -> real_date:
        raise AssertionError("date.today() is forbidden in subscription_llm replay")

    @staticmethod
    def fromisoformat(value: str) -> real_date:
        return real_date.fromisoformat(value)


class ExplodingDateTime:
    @staticmethod
    def now(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("datetime.now() is forbidden in subscription_llm replay")


class FixedDateTime:
    @staticmethod
    def now(tz: Any = None) -> real_datetime:
        fixed = real_datetime(2026, 6, 11, 12, 0, 0, tzinfo=timezone.utc)
        if tz is not None:
            return fixed.astimezone(tz)
        return fixed


@contextlib.contextmanager
def time_sentinel() -> Any:
    import mango_mvp.channels.subscription_llm as llm
    import mango_mvp.channels.dialogue_memory as dialogue_memory
    import mango_mvp.channels.draft_prompt_builder as draft_prompt_builder

    patched: list[tuple[Any, str, Any]] = []

    def patch_module(module: Any) -> None:
        for attr, replacement in (("date", ExplodingDate), ("datetime", ExplodingDateTime)):
            if hasattr(module, attr):
                patched.append((module, attr, getattr(module, attr)))
                setattr(module, attr, replacement)

    patch_module(llm)
    for name, module in list(sys.modules.items()):
        if name.startswith("mango_mvp.channels.subscription_llm_parts") and module is not None:
            patch_module(module)
    for module in (draft_prompt_builder, dialogue_memory):
        patched.append((module, "datetime", getattr(module, "datetime")))
        setattr(module, "datetime", FixedDateTime)
    try:
        yield
    finally:
        for module, attr, original in reversed(patched):
            setattr(module, attr, original)


def base_context(**extra: Any) -> dict[str, Any]:
    context = {
        "active_brand": "foton",
        "now_msk_hour": 12,
        "TELEGRAM_DIRECT_PATH": "0",
        "TELEGRAM_DIALOGUE_CONTRACT_PIPELINE": "0",
        "TELEGRAM_ROUTE_RUBRIC": "0",
        "TELEGRAM_LLM_RETRIEVE": "0",
        "TELEGRAM_SEMANTIC_OUTPUT_VERIFIER": "0",
        "TELEGRAM_NIGHT_HOURS_NOTE": "0",
    }
    context.update(extra)
    return context


def result_json(result: Any) -> dict[str, Any]:
    return dict(result.to_json_dict(include_raw_response=True))


def prompt_payload(prompts: Sequence[str]) -> list[dict[str, str | int]]:
    return [
        {
            "index": index,
            "sha256": sha256_text(prompt),
            "bytes_len": len(prompt.encode("utf-8")),
            "utf8": prompt,
        }
        for index, prompt in enumerate(prompts, 1)
    ]


def selected_metadata(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
    result: dict[str, Any] = {}
    for key in (
        "authoritative_output_gate",
        "direct_path",
        "dialogue_contract_pipeline",
        "night_hours_note",
        "cache_hit",
        "codex_exec",
        "parse_error",
        "last_error",
        "semantic_output_verifier",
        "semantic_diagnosis_guard",
        "output_sanitizer",
    ):
        if key in metadata:
            result[key] = metadata[key]
    return result


def fallback_markers(payload: Mapping[str, Any]) -> list[str]:
    markers: list[str] = []
    error = str(payload.get("error") or "").strip()
    if error:
        markers.append(f"error:{error}")
    for flag in payload.get("safety_flags") or []:
        if str(flag) == "llm_fallback" or "fallback" in str(flag):
            markers.append(f"flag:{flag}")
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
    direct = metadata.get("direct_path") if isinstance(metadata.get("direct_path"), Mapping) else {}
    reason_class = str(direct.get("reason_class") or "").strip()
    if reason_class:
        markers.append(f"direct_path.reason_class:{reason_class}")
    text_source = str(direct.get("text_composition_source") or "").strip()
    if text_source:
        markers.append(f"direct_path.text_composition_source:{text_source}")
    evidence = direct.get("reason_evidence") if isinstance(direct.get("reason_evidence"), Mapping) else {}
    if evidence.get("provider_error"):
        markers.append(f"direct_path.reason_evidence.provider_error:{evidence.get('provider_error')}")
    llm_retrieve = direct.get("llm_retrieve") if isinstance(direct.get("llm_retrieve"), Mapping) else {}
    if llm_retrieve.get("fallback") is True:
        markers.append(f"direct_path.llm_retrieve.fallback_reason:{llm_retrieve.get('fallback_reason') or 'unknown'}")
    rubric_reason = str(direct.get("rubric_reason") or "").strip()
    if rubric_reason.startswith("regen_failed:"):
        markers.append(f"direct_path.rubric_reason:{rubric_reason}")
    reason = str(metadata.get("reason") or "").strip()
    if reason:
        markers.append(f"metadata.reason:{reason}")
    if metadata.get("last_error"):
        markers.append(f"metadata.last_error:{metadata.get('last_error')}")
    semantic = metadata.get("semantic_output_verifier") if isinstance(metadata.get("semantic_output_verifier"), Mapping) else {}
    if semantic.get("unavailable") is True:
        markers.append("semantic_output_verifier.unavailable:true")
    if semantic.get("fallback_reason"):
        markers.append(f"semantic_output_verifier.fallback_reason:{semantic.get('fallback_reason')}")
    if semantic.get("regen_error"):
        markers.append(f"semantic_output_verifier.regen_error:{semantic.get('regen_error')}")
    diagnosis = metadata.get("semantic_diagnosis_guard") if isinstance(metadata.get("semantic_diagnosis_guard"), Mapping) else {}
    if diagnosis.get("fallback_reason"):
        markers.append(f"semantic_diagnosis_guard.fallback_reason:{diagnosis.get('fallback_reason')}")
    pipeline = metadata.get("dialogue_contract_pipeline") if isinstance(metadata.get("dialogue_contract_pipeline"), Mapping) else {}
    if pipeline.get("fallback_reason"):
        markers.append(f"dialogue_contract_pipeline.fallback_reason:{pipeline.get('fallback_reason')}")
    sanitizer = metadata.get("output_sanitizer") if isinstance(metadata.get("output_sanitizer"), Mapping) else {}
    if sanitizer.get("fallback") is True:
        markers.append("output_sanitizer.fallback:true")
    gate = metadata.get("authoritative_output_gate") if isinstance(metadata.get("authoritative_output_gate"), Mapping) else {}
    if gate.get("action") in {"block", "downgrade", "downgrade_keep_text"}:
        markers.append(f"authoritative_output_gate.action:{gate.get('action')}")
    return markers


def assert_no_unexpected_fallback(record: Mapping[str, Any], case: Mapping[str, Any]) -> None:
    if case.get("expect_fallback"):
        return
    payload = record.get("result_json") if isinstance(record.get("result_json"), Mapping) else {}
    markers = fallback_markers(payload)
    bad = []
    for marker in markers:
        value = marker.split(":", 1)[1]
        if marker.startswith("authoritative_output_gate.action:") and case.get("allow_policy_downgrade"):
            continue
        if value in DEFAULT_UNEXPECTED_FALLBACK_ERRORS or value in DEFAULT_UNEXPECTED_METADATA_REASONS or "provider_runtime" in value:
            bad.append(marker)
    if bad:
        raise AssertionError(f"{record['case_id']} unexpected fallback markers: {bad}")


@dataclass
class StubRunner:
    payloads: list[Mapping[str, Any]]
    returncodes: list[int] | None = None
    stderrs: list[str] | None = None
    timeout_on_call: int | None = None

    def __post_init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, cmd: Sequence[str], input: str, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        call_no = len(self.calls) + 1
        self.calls.append({"cmd": list(cmd), "input": input, "timeout": kwargs.get("timeout")})
        if self.timeout_on_call == call_no:
            raise subprocess.TimeoutExpired(cmd=list(cmd), timeout=kwargs.get("timeout") or 1)
        rc = self.returncodes[call_no - 1] if self.returncodes and call_no <= len(self.returncodes) else 0
        stderr = self.stderrs[call_no - 1] if self.stderrs and call_no <= len(self.stderrs) else ""
        output_path = Path(list(cmd)[list(cmd).index("--output-last-message") + 1])
        if rc == 0:
            payload = self.payloads[min(call_no - 1, len(self.payloads) - 1)]
            output_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return subprocess.CompletedProcess(list(cmd), rc, stdout="", stderr=stderr)

    @property
    def prompts(self) -> list[str]:
        return [str(item["input"]) for item in self.calls]


def make_record(case: Mapping[str, Any], *, branch: str, context: Mapping[str, Any], result: Any, prompts: Sequence[str], captured: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload = result_json(result)
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": case["case_id"],
        "branch": branch,
        "client_message": case.get("client_message", ""),
        "context_sha256": sha256_text(stable_json(canonicalize_for_hash(context))),
        "prompt_bytes": prompt_payload(prompts),
        "prompt_sha256": [item["sha256"] for item in prompt_payload(prompts)],
        "route": payload.get("route"),
        "draft_text": payload.get("draft_text"),
        "error": payload.get("error"),
        "selected_metadata": selected_metadata(payload),
        "fallback_markers": fallback_markers(payload),
        "captured": dict(captured or {}),
        "result_json": payload,
    }


def case_legacy_ok(case: Mapping[str, Any]) -> dict[str, Any]:
    from mango_mvp.channels.subscription_llm import SubscriptionLlmDraftProvider

    context = base_context(confirmed_facts={"fact.general": "Фотон: есть курсы для школьников."})
    runner = StubRunner(
        [
            {
                "route": "draft_for_manager",
                "draft_text": "Здравствуйте! Менеджер проверит подходящую группу и вернётся с ответом.",
                "message_type": "question",
                "topic_id": "service:S5_general_consultation",
                "confidence_theme": 0.81,
                "manager_checklist": ["Проверить группу для 6 класса."],
            }
        ]
    )
    provider = SubscriptionLlmDraftProvider(runner=runner, cache_dir=None, sleep=lambda _: None)
    result = provider.build_draft(str(case["client_message"]), context=context)
    return make_record(case, branch="legacy", context=context, result=result, prompts=runner.prompts)


def case_legacy_retryable_rc(case: Mapping[str, Any]) -> dict[str, Any]:
    from mango_mvp.channels.subscription_llm import SubscriptionLlmDraftProvider

    context = base_context()
    sleeps: list[float] = []
    runner = StubRunner(
        [
            {
                "route": "draft_for_manager",
                "draft_text": "Передам вопрос менеджеру, он сверит данные и вернётся с ответом.",
                "message_type": "question",
                "topic_id": "service:S2_unclear",
                "confidence_theme": 0.7,
            }
        ],
        returncodes=[503, 0],
        stderrs=["temporarily unavailable", ""],
    )
    provider = SubscriptionLlmDraftProvider(runner=runner, cache_dir=None, sleep=sleeps.append, max_attempts=2)
    result = provider.build_draft(str(case["client_message"]), context=context)
    return make_record(
        case,
        branch="legacy",
        context=context,
        result=result,
        prompts=runner.prompts,
        captured={"runner_returncodes": [503, 0], "sleep_calls": sleeps},
    )


def case_legacy_non_retryable_rc(case: Mapping[str, Any]) -> dict[str, Any]:
    from mango_mvp.channels.subscription_llm import SubscriptionLlmDraftProvider

    context = base_context()
    runner = StubRunner([], returncodes=[2], stderrs=["fatal non retryable error"])
    provider = SubscriptionLlmDraftProvider(runner=runner, cache_dir=None, sleep=lambda _: None, max_attempts=1)
    result = provider.build_draft(str(case["client_message"]), context=context)
    return make_record(case, branch="legacy", context=context, result=result, prompts=runner.prompts)


def case_legacy_timeout(case: Mapping[str, Any]) -> dict[str, Any]:
    from mango_mvp.channels.subscription_llm import SubscriptionLlmDraftProvider

    context = base_context()
    runner = StubRunner([], timeout_on_call=1)
    provider = SubscriptionLlmDraftProvider(runner=runner, cache_dir=None, sleep=lambda _: None, timeout_sec=1)
    result = provider.build_draft(str(case["client_message"]), context=context)
    return make_record(case, branch="legacy", context=context, result=result, prompts=runner.prompts)


def case_legacy_cache_put_hit(case: Mapping[str, Any]) -> dict[str, Any]:
    from mango_mvp.channels.subscription_llm import SubscriptionLlmDraftProvider

    context = base_context()
    runner = StubRunner(
        [
            {
                "route": "draft_for_manager",
                "draft_text": "Менеджер проверит условия и вернётся с ответом.",
                "message_type": "question",
                "topic_id": "service:S2_unclear",
            }
        ]
    )
    with tempfile.TemporaryDirectory(prefix="subscription_llm_replay_cache_") as tmp_dir:
        provider = SubscriptionLlmDraftProvider(runner=runner, cache_dir=tmp_dir, sleep=lambda _: None)
        first = provider.generate_from_prompt("Replay cache prompt")
        second = provider.generate_from_prompt("Replay cache prompt")
        cache_files = sorted(Path(tmp_dir).glob("*.json"))
    return make_record(
        case,
        branch="legacy_cache",
        context=context,
        result=second,
        prompts=runner.prompts,
        captured={
            "first_route": first.route,
            "runner_calls": len(runner.calls),
            "cache_files_count": len(cache_files),
            "second_cache_hit": bool(second.metadata.get("cache_hit")),
        },
    )


def case_guard_cache_dir_stable_runtime(case: Mapping[str, Any]) -> dict[str, Any]:
    from mango_mvp.channels import subscription_llm as llm

    context = base_context()
    try:
        llm._guard_cache_dir(repo_root() / "stable_runtime" / "subscription_llm_cache")
    except ValueError as exc:
        result = llm.SubscriptionDraftResult(route="manager_only", draft_text=str(exc), metadata={"guard_cache_dir_error": str(exc)})
    else:
        raise AssertionError("_guard_cache_dir accepted stable_runtime path")
    return make_record(case, branch="cache_guard", context=context, result=result, prompts=())


def case_fake_provider_guarded(case: Mapping[str, Any]) -> dict[str, Any]:
    from mango_mvp.channels.subscription_llm import FakeSubscriptionLlmDraftProvider, SubscriptionDraftResult

    context = base_context()
    provider = FakeSubscriptionLlmDraftProvider(
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            draft_text="Да, курс есть. Менеджер поможет подобрать группу.",
            topic_id="service:S5_general_consultation",
        )
    )
    result = provider.build_draft(str(case["client_message"]), context=context)
    return make_record(case, branch="fake", context=context, result=result, prompts=provider.prompts)


class DirectProviderBase:
    def __init__(self, *results: Any, retriever_payload: Mapping[str, Any] | str | Exception | None = None) -> None:
        from mango_mvp.channels.subscription_llm import SubscriptionLlmDraftProvider

        class _Provider(SubscriptionLlmDraftProvider):
            def __init__(self, outer: "DirectProviderBase") -> None:
                super().__init__(sleep=lambda _: None)
                self.outer = outer

            def _direct_path_draft_runner(self, prompt: str) -> Any:
                self.outer.prompts.append(prompt)
                self.outer.calls += 1
                if not self.outer.results:
                    raise AssertionError("unexpected direct path draft call")
                value = self.outer.results.pop(0)
                if isinstance(value, Exception):
                    raise value
                return value

            def _direct_path_llm_retrieve_runner(self, prompt: str) -> Mapping[str, Any] | str:
                self.outer.retriever_prompts.append(prompt)
                self.outer.retriever_calls += 1
                value = self.outer.retriever_payload
                if isinstance(value, Exception):
                    raise value
                if value is None:
                    raise AssertionError("unexpected direct path retrieve call")
                return value

        self.calls = 0
        self.prompts: list[str] = []
        self.retriever_calls = 0
        self.retriever_prompts: list[str] = []
        self.retriever_payload = retriever_payload
        self.results = list(results)
        self.provider = _Provider(self)


def direct_context(**extra: Any) -> dict[str, Any]:
    return base_context(
        **{
            "TELEGRAM_DIRECT_PATH": "1",
            "TELEGRAM_LLM_RETRIEVE": "0",
            "confirmed_facts": {"fact.price": "Фотон: годовой курс стоит 59 000 ₽."},
            **extra,
        }
    )


def case_direct_ok(case: Mapping[str, Any]) -> dict[str, Any]:
    from mango_mvp.channels.subscription_llm import SubscriptionDraftResult

    context = direct_context()
    holder = DirectProviderBase(
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            draft_text="Годовой курс Фотона стоит 59 000 ₽.",
            topic_id="theme:001_pricing",
            context_used=("fact.price",),
        )
    )
    result = holder.provider.build_draft(str(case["client_message"]), context=context)
    return make_record(case, branch="direct_path", context=context, result=result, prompts=holder.prompts, captured={"calls": holder.calls})


def case_direct_p0_preblock(case: Mapping[str, Any]) -> dict[str, Any]:
    from mango_mvp.channels.subscription_llm import SubscriptionDraftResult

    context = direct_context()
    holder = DirectProviderBase(SubscriptionDraftResult(route="bot_answer_self_for_pilot", draft_text="should not be called"))
    result = holder.provider.build_draft(str(case["client_message"]), context=context)
    return make_record(case, branch="direct_path", context=context, result=result, prompts=holder.prompts, captured={"calls": holder.calls})


def case_direct_default_gold_pack_path(case: Mapping[str, Any]) -> dict[str, Any]:
    from mango_mvp.channels import subscription_llm as llm
    from mango_mvp.channels.subscription_llm import SubscriptionDraftResult

    os.environ.pop(llm.BOT_GOLD_REAL_PACK_ENV, None)
    assert llm.DIRECT_PATH_REAL_MANAGER_GOLD_PACK_PATH.exists(), llm.DIRECT_PATH_REAL_MANAGER_GOLD_PACK_PATH
    context = direct_context(TELEGRAM_BOT_GOLD_REAL="1")
    holder = DirectProviderBase(
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            draft_text="Да, рассрочка в Фотоне доступна.",
            topic_id="theme:006_installment",
        )
    )
    result = holder.provider.build_draft(str(case["client_message"]), context=context)
    return make_record(
        case,
        branch="direct_path",
        context=context,
        result=result,
        prompts=holder.prompts,
        captured={
            "default_gold_path": str(llm.DIRECT_PATH_REAL_MANAGER_GOLD_PACK_PATH),
            "default_gold_path_exists": llm.DIRECT_PATH_REAL_MANAGER_GOLD_PACK_PATH.exists(),
            "prompt_has_gold_block": bool(holder.prompts and "Живые образцы менеджерского стиля" in holder.prompts[0]),
        },
    )


def write_snapshot_file(payload: Mapping[str, Any]) -> Path:
    handle = tempfile.NamedTemporaryFile(prefix="subscription_llm_replay_snapshot_", suffix=".json", delete=False)
    path = Path(handle.name)
    handle.close()
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def case_direct_llm_retrieve_fake(case: Mapping[str, Any]) -> dict[str, Any]:
    from mango_mvp.channels.subscription_llm import SubscriptionDraftResult

    snapshot_path = write_snapshot_file(
        {
            "facts": [
                {
                    "brand": "foton",
                    "fact_key": "foton.price.year",
                    "fact_type": "price",
                    "product": "regular_course",
                    "allowed_for_client_answer": True,
                    "forbidden_for_client": False,
                    "internal_only": False,
                    "valid_until": "",
                    "client_safe_text": "Фотон: годовой курс стоит 59 000 ₽.",
                },
                {
                    "brand": "foton",
                    "fact_key": "foton.installment",
                    "fact_type": "payment",
                    "product": "regular_course",
                    "allowed_for_client_answer": True,
                    "forbidden_for_client": False,
                    "internal_only": False,
                    "valid_until": "",
                    "client_safe_text": "Фотон: доступна рассрочка.",
                },
            ]
        }
    )
    try:
        context = direct_context(
            TELEGRAM_LLM_RETRIEVE="1",
            snapshot_path=str(snapshot_path),
            confirmed_facts={},
            conversation_intent_plan={"primary_intent": "pricing", "answer_topics": ["pricing"]},
        )
        holder = DirectProviderBase(
            SubscriptionDraftResult(
                route="bot_answer_self_for_pilot",
                draft_text="Годовой курс Фотона стоит 59 000 ₽.",
                topic_id="theme:001_pricing",
                context_used=("foton.price.year",),
            ),
            retriever_payload={"exact_ids": ["foton.price.year"], "adjacent_ids": ["foton.installment"]},
        )
        result = holder.provider.build_draft(str(case["client_message"]), context=context)
        return make_record(
            case,
            branch="direct_path_llm_retrieve",
            context=context,
            result=result,
            prompts=[*holder.retriever_prompts, *holder.prompts],
            captured={"draft_calls": holder.calls, "retriever_calls": holder.retriever_calls},
        )
    finally:
        snapshot_path.unlink(missing_ok=True)


def case_route_rubric_regen(case: Mapping[str, Any]) -> dict[str, Any]:
    from mango_mvp.channels.subscription_llm import SubscriptionDraftResult

    context = direct_context(TELEGRAM_ROUTE_RUBRIC="1")
    holder = DirectProviderBase(
        SubscriptionDraftResult(route="draft_for_manager", draft_text="Передам менеджеру."),
        SubscriptionDraftResult(route="bot_answer_self_for_pilot", draft_text="Годовой курс Фотона стоит 59 000 ₽."),
    )
    result = holder.provider.build_draft(str(case["client_message"]), context=context)
    return make_record(case, branch="direct_path", context=context, result=result, prompts=holder.prompts, captured={"calls": holder.calls})


def case_night_note(case: Mapping[str, Any]) -> dict[str, Any]:
    from mango_mvp.channels import subscription_llm as llm

    context = base_context(TELEGRAM_NIGHT_HOURS_NOTE="1", now_msk_hour=22)
    source = llm.SubscriptionDraftResult(
        route="draft_for_manager",
        draft_text="Передам вопрос менеджеру, он вернётся с ответом.",
        topic_id="service:S2_unclear",
    )
    result = llm.apply_authoritative_output_gate(source, client_message=str(case["client_message"]), context=context)
    return make_record(case, branch="final_gate", context=context, result=result, prompts=())


def case_dialogue_contract_ok(case: Mapping[str, Any]) -> dict[str, Any]:
    from mango_mvp.channels.subscription_llm import SubscriptionDraftResult, SubscriptionLlmDraftProvider

    class Provider(SubscriptionLlmDraftProvider):
        def __init__(self) -> None:
            super().__init__(sleep=lambda _: None)
            self.calls = 0

        def _build_dialogue_contract_pipeline_draft(self, client_message: str, *, context: Mapping[str, Any] | None = None) -> SubscriptionDraftResult:
            self.calls += 1
            return SubscriptionDraftResult(
                route="draft_for_manager",
                draft_text="Фотон: годовой курс стоит 59 000 ₽. Менеджер проверит подходящую группу.",
                topic_id="theme:001_pricing",
                confidence_group=0.9,
                topic_confidence=0.9,
                context_used=("dialogue_contract",),
                metadata={
                    "dialogue_contract_pipeline": {
                        "contract": {"question": "цена", "answerability": "answer_self"},
                        "retrieved_fact_keys": ["fact.price"],
                        "retrieved_facts": {"fact.price": "Фотон: годовой курс стоит 59 000 ₽."},
                        "missing_fact_keys": [],
                        "findings": [],
                        "fallback_reason": "",
                    }
                },
            )

    context = base_context(TELEGRAM_DIALOGUE_CONTRACT_PIPELINE="1", client_safe_fact_verified=True)
    provider = Provider()
    result = provider.build_draft(str(case["client_message"]), context=context)
    return make_record(case, branch="dialogue_contract", context=context, result=result, prompts=(), captured={"pipeline_calls": provider.calls})


def case_brand_separation(case: Mapping[str, Any]) -> dict[str, Any]:
    from mango_mvp.channels import subscription_llm as llm

    context = base_context(active_brand="unpk")
    source = llm.SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="У Фотона и УНПК одинаковые условия по рассрочке.",
        message_type="question",
        topic_id="service:S5_general_consultation",
    )
    result = llm.apply_brand_separation_guard(source, client_message=str(case["client_message"]), context=context)
    return make_record(case, branch="guard", context=context, result=result, prompts=())


def case_payment_confirmation(case: Mapping[str, Any]) -> dict[str, Any]:
    from mango_mvp.channels import subscription_llm as llm

    context = base_context(active_brand="unpk")
    source = llm.SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Оплата получена, доступ откроем.",
        message_type="question",
        topic_id="theme:003_payment_status",
    )
    result = llm.apply_payment_confirmation_guard(source, client_message=str(case["client_message"]), context=context)
    return make_record(case, branch="guard", context=context, result=result, prompts=())


def case_known_context_no_reask(case: Mapping[str, Any]) -> dict[str, Any]:
    from mango_mvp.channels.subscription_llm import FakeSubscriptionLlmDraftProvider

    context = base_context(
        active_brand="unpk",
        autonomy_enabled=True,
        client_safe_fact_verified=True,
        known_client_fields={"student_name": "Колосов Даниил Максимович"},
        known_dialog_fields={"grade": "9", "subject": "физика"},
        facts_context={"client_safe_fact_verified": True, "fresh": True},
    )
    provider = FakeSubscriptionLlmDraftProvider(
        {
            "route": "bot_answer_self_for_pilot",
            "draft_text": "Напишите, пожалуйста, ФИО ребёнка, какой класс и какой предмет интересует.",
            "message_type": "question",
            "topic_id": "theme:016_program",
            "confidence_theme": 0.91,
        }
    )
    result = provider.build_draft(str(case["client_message"]), context=context)
    return make_record(case, branch="guard", context=context, result=result, prompts=provider.prompts)


def case_valid_until_2099(case: Mapping[str, Any]) -> dict[str, Any]:
    from mango_mvp.channels import subscription_llm as llm

    context = base_context()
    checks = {
        "empty": llm._direct_path_valid_until_ok("", today=real_date(2026, 6, 11)),
        "future_2099": llm._direct_path_valid_until_ok("2099-12-31", today=real_date(2026, 6, 11)),
    }
    result = llm.SubscriptionDraftResult(route="bot_answer_self_for_pilot", draft_text=json.dumps(checks, sort_keys=True), metadata={"valid_until": checks})
    return make_record(case, branch="direct_path_time", context=context, result=result, prompts=())


def case_memory_provenance_prompt(case: Mapping[str, Any]) -> dict[str, Any]:
    from mango_mvp.channels.dialogue_memory import build_dialogue_memory
    from mango_mvp.channels.subscription_llm import SubscriptionDraftResult

    os.environ["TELEGRAM_MEMORY_PROVENANCE"] = "1"
    os.environ["TELEGRAM_MEMORY_PROVENANCE_COMPACT"] = "1"
    memory = build_dialogue_memory(
        current_message="Ребёнок Артём в 6 классе, нужна физика очно.",
        active_brand="foton",
        recent_messages=("Клиент: Здравствуйте",),
        context={"current_message_id": "replay-message-1"},
        session_id="replay-memory",
    )
    memory_view = memory.to_prompt_view()
    context = direct_context(
        TELEGRAM_MEMORY_PROVENANCE="1",
        TELEGRAM_MEMORY_PROVENANCE_COMPACT="1",
        dialogue_memory_view=memory_view,
        confirmed_facts={"fact.program": "Фотон: есть очные занятия по физике для школьников."},
    )
    holder = DirectProviderBase(
        SubscriptionDraftResult(
            route="draft_for_manager",
            draft_text="Менеджер подберёт очную группу по физике для 6 класса.",
            topic_id="theme:020_enrollment",
        )
    )
    result = holder.provider.build_draft(str(case["client_message"]), context=context)
    prompt = holder.prompts[0] if holder.prompts else ""
    return make_record(
        case,
        branch="direct_path_memory",
        context=context,
        result=result,
        prompts=holder.prompts,
        captured={
            "memory_known_slots": memory_view.get("known_slots", {}),
            "memory_slot_provenance_keys": sorted((memory_view.get("slot_provenance") or {}).keys()),
            "prompt_has_slot_provenance": "slot_provenance" in prompt or "memory_provenance" in prompt,
        },
    )


CASE_HANDLERS: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
    "legacy_ok": case_legacy_ok,
    "legacy_retryable_rc": case_legacy_retryable_rc,
    "legacy_non_retryable_rc": case_legacy_non_retryable_rc,
    "legacy_timeout": case_legacy_timeout,
    "legacy_cache_put_hit": case_legacy_cache_put_hit,
    "guard_cache_dir_stable_runtime": case_guard_cache_dir_stable_runtime,
    "fake_provider_guarded": case_fake_provider_guarded,
    "direct_ok": case_direct_ok,
    "direct_p0_preblock": case_direct_p0_preblock,
    "direct_default_gold_pack_path": case_direct_default_gold_pack_path,
    "direct_llm_retrieve_fake": case_direct_llm_retrieve_fake,
    "route_rubric_regen": case_route_rubric_regen,
    "dialogue_contract_ok": case_dialogue_contract_ok,
    "brand_separation": case_brand_separation,
    "payment_confirmation": case_payment_confirmation,
    "known_context_no_reask": case_known_context_no_reask,
    "night_note": case_night_note,
    "valid_until_2099": case_valid_until_2099,
    "memory_provenance_prompt": case_memory_provenance_prompt,
}


def run_cases(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    records = []
    with clean_telegram_env(), time_sentinel():
        for case in cases:
            kind = str(case.get("kind") or case.get("case_id") or "")
            handler = CASE_HANDLERS.get(kind)
            if handler is None:
                raise SystemExit(f"unknown replay case kind={kind!r}")
            record = handler(case)
            assert_no_unexpected_fallback(record, case)
            records.append(record)
    return records


def compare_records(actual: Sequence[Mapping[str, Any]], expected: Sequence[Mapping[str, Any]]) -> list[str]:
    errors: list[str] = []
    actual_by_id = {str(row["case_id"]): row for row in actual}
    expected_by_id = {str(row["case_id"]): row for row in expected}
    if set(actual_by_id) != set(expected_by_id):
        errors.append(f"case id mismatch actual={sorted(actual_by_id)} expected={sorted(expected_by_id)}")
        return errors
    for case_id in sorted(expected_by_id):
        a = actual_by_id[case_id]
        e = expected_by_id[case_id]
        if stable_json(a) != stable_json(e):
            errors.append(case_id)
    return errors


def write_summary(out_dir: Path, actual: Sequence[Mapping[str, Any]], errors: Sequence[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "actual.jsonl", actual)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "case_count": len(actual),
        "case_ids": [row["case_id"] for row in actual],
        "errors": list(errors),
        "status": "failed" if errors else "ok",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# subscription_llm equivalence replay",
        "",
        f"- status: `{summary['status']}`",
        f"- cases: `{summary['case_count']}`",
        f"- errors: `{', '.join(errors) if errors else 'none'}`",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic subscription_llm equivalence replay.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--freeze-baseline", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    cases_path = args.cases if args.cases.is_absolute() else root / args.cases
    baseline_path = args.baseline if args.baseline.is_absolute() else root / args.baseline
    out_dir = args.out if args.out is None or args.out.is_absolute() else root / args.out
    cases = read_jsonl(cases_path)
    actual = run_cases(cases)
    if args.freeze_baseline:
        if baseline_path.exists() and not args.force:
            raise SystemExit(f"baseline already exists and must not be re-frozen: {baseline_path}")
        write_jsonl(baseline_path, actual)
        errors: list[str] = []
    else:
        expected = read_jsonl(baseline_path)
        errors = compare_records(actual, expected)
        if errors:
            print(f"replay mismatch: {errors}", file=sys.stderr)
    if out_dir is not None:
        write_summary(out_dir, actual, errors)
    print(json.dumps({"status": "failed" if errors else "ok", "cases": len(actual), "errors": errors}, ensure_ascii=False, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
