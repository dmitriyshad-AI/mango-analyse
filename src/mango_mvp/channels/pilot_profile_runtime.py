from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Mapping, MutableMapping

from mango_mvp.channels.dialogue_contract_pipeline import AUTONOMY_SCOPE_PRECISION_ENV, NUMBER_GATE_SCOPE_AWARE_ENV
from mango_mvp.channels.fact_venue_scope import FACT_VENUE_SCOPE_ENV
from mango_mvp.channels.subscription_llm_parts.direct_path import _presale_safety_enabled
from mango_mvp.channels.subscription_llm_parts.post_layers import (
    _output_sanitizer_enabled,
    _semantic_output_verifier_enabled,
    _verifier_handoff_claims_enabled,
)
from mango_mvp.channels.subscription_llm_parts.support import (
    DIRECT_PATH_PILOT_CONFIG_ENV,
    DIRECT_PATH_PILOT_CONFIG_VERSION,
    PII_RELATION_STOPWORDS_ENV,
    P0_MODEL_LED_ENV,
    PRESALE_PII_MEMORY_ENV,
    PRESALE_SAFETY_ENV,
    PROSE_MODEL_LED_ENV,
    VERIFIER_HANDOFF_CLAIMS_ENV,
    _explicit_truthy_setting,
    _pilot_gold_profile_enabled,
    _pilot_profile_default_on_flag_enabled,
)


ENFORCE_CANONICAL_PROFILE_ENV = "ENFORCE_CANONICAL_PROFILE"

REQUIRED_GUARD_KEYS = (
    "presale_safety",
    "presale_pii_memory",
    "pii_relation_stopwords",
    "verifier_handoff_claims",
)

QUALITY_GUARD_KEYS = (
    "semantic_output_verifier",
    "output_sanitizer",
    "number_gate_scope_aware",
)

_CANONICAL_ONE_ENV_NAMES = {
    PRESALE_SAFETY_ENV: "presale_safety",
    PRESALE_PII_MEMORY_ENV: "presale_pii_memory",
    PII_RELATION_STOPWORDS_ENV: "pii_relation_stopwords",
    VERIFIER_HANDOFF_CLAIMS_ENV: "verifier_handoff_claims",
}


@dataclass(frozen=True)
class PilotProfileActivation:
    enforce_enabled: bool
    profile_value: str
    action: str
    warnings: tuple[str, ...] = ()

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "enforce_enabled": self.enforce_enabled,
            "profile_value": self.profile_value,
            "action": self.action,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class PilotProfileSelfCheck:
    ok: bool
    required: bool
    effective_profile: str
    draft_path: str
    active_guards: Mapping[str, bool]
    quality_guards: Mapping[str, bool]
    failures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "required": self.required,
            "effective_profile": self.effective_profile,
            "draft_path": self.draft_path,
            "active_guards": dict(self.active_guards),
            "quality_guards": dict(self.quality_guards),
            "failures": list(self.failures),
            "warnings": list(self.warnings),
        }


def sync_env_to_process(env: Mapping[str, str], *, environ: MutableMapping[str, str] | None = None) -> None:
    target = os.environ if environ is None else environ
    for key, value in env.items():
        target[str(key)] = str(value)


def strict_one_enabled(env: Mapping[str, str], key: str) -> bool:
    return str(env.get(key) or "").strip() == "1"


def _strict_one_warning(env: Mapping[str, str], key: str) -> str:
    value = str(env.get(key) or "").strip()
    if not value or value == "1":
        return ""
    return f"{key}=non_canonical:{value[:40]}"


def ensure_canonical_pilot_profile(
    *,
    environ: MutableMapping[str, str] | None = None,
    warn: Callable[[str], None] | None = None,
) -> PilotProfileActivation:
    target = os.environ if environ is None else environ
    warnings: list[str] = []
    enforce_value = str(target.get(ENFORCE_CANONICAL_PROFILE_ENV) or "").strip()
    if enforce_value and enforce_value != "1":
        warnings.append(f"{ENFORCE_CANONICAL_PROFILE_ENV}=non_canonical:{enforce_value[:40]}")
    if enforce_value != "1":
        return PilotProfileActivation(
            enforce_enabled=False,
            profile_value=str(target.get(DIRECT_PATH_PILOT_CONFIG_ENV) or "").strip(),
            action="disabled",
            warnings=tuple(warnings),
        )

    profile_value = str(target.get(DIRECT_PATH_PILOT_CONFIG_ENV) or "").strip()
    if not profile_value:
        target[DIRECT_PATH_PILOT_CONFIG_ENV] = DIRECT_PATH_PILOT_CONFIG_VERSION
        profile_value = DIRECT_PATH_PILOT_CONFIG_VERSION
        action = "set_default"
    elif profile_value == DIRECT_PATH_PILOT_CONFIG_VERSION:
        action = "already_set"
    else:
        action = "operator_override_kept"
        warnings.append(f"{DIRECT_PATH_PILOT_CONFIG_ENV}=override:{profile_value[:80]}")

    if warnings and warn is not None:
        for item in warnings:
            warn(item)
    return PilotProfileActivation(
        enforce_enabled=True,
        profile_value=profile_value,
        action=action,
        warnings=tuple(warnings),
    )


def _guard_status() -> dict[str, bool]:
    return {
        "presale_safety": _presale_safety_enabled(None),
        "presale_pii_memory": _presale_safety_enabled(None, subflag=PRESALE_PII_MEMORY_ENV),
        "pii_relation_stopwords": _pilot_profile_default_on_flag_enabled(None, PII_RELATION_STOPWORDS_ENV),
        "verifier_handoff_claims": _verifier_handoff_claims_enabled(None),
        "p0_model_led": _explicit_truthy_setting(None, P0_MODEL_LED_ENV) is True,
        "prose_model_led": _explicit_truthy_setting(None, PROSE_MODEL_LED_ENV) is True,
        "fact_venue_scope": _explicit_truthy_setting(None, FACT_VENUE_SCOPE_ENV) is True,
        "autonomy_scope_precision": _explicit_truthy_setting(None, AUTONOMY_SCOPE_PRECISION_ENV) is True,
    }


def _quality_guard_status() -> dict[str, bool]:
    return {
        "semantic_output_verifier": _semantic_output_verifier_enabled(None),
        "output_sanitizer": _output_sanitizer_enabled(None),
        "number_gate_scope_aware": _pilot_profile_default_on_flag_enabled(None, NUMBER_GATE_SCOPE_AWARE_ENV),
    }


def current_draft_path(*, dialogue_contract_pipeline_enabled: bool) -> str:
    if _pilot_gold_profile_enabled(None):
        return "direct_path"
    if dialogue_contract_pipeline_enabled:
        return "dialogue_contract_pipeline"
    return "legacy"


def pilot_profile_selfcheck(
    *,
    require: bool | None = None,
    dialogue_contract_pipeline_enabled: bool = True,
    env: Mapping[str, str] | None = None,
) -> PilotProfileSelfCheck:
    source = os.environ if env is None else env
    required = strict_one_enabled(source, ENFORCE_CANONICAL_PROFILE_ENV) if require is None else bool(require)
    profile_value = str(source.get(DIRECT_PATH_PILOT_CONFIG_ENV) or "").strip()
    active_guards = _guard_status()
    quality_guards = _quality_guard_status()
    failures: list[str] = []
    warnings: list[str] = []

    enforce_warning = _strict_one_warning(source, ENFORCE_CANONICAL_PROFILE_ENV)
    if enforce_warning:
        warnings.append(enforce_warning)
        if require is None:
            failures.append("enforce_canonical_profile_non_canonical")

    for env_name, guard_name in _CANONICAL_ONE_ENV_NAMES.items():
        warning = _strict_one_warning(source, env_name)
        if warning:
            warnings.append(warning)
            failures.append(f"{guard_name}_non_canonical_env_value")

    profile_enabled = _pilot_gold_profile_enabled(None)
    if required and not profile_enabled:
        failures.append("pilot_gold_profile_disabled")
    if required:
        for key in REQUIRED_GUARD_KEYS:
            if not active_guards.get(key):
                failures.append(f"{key}_disabled")
    for key in QUALITY_GUARD_KEYS:
        if not quality_guards.get(key):
            warnings.append(f"{key}_disabled")

    return PilotProfileSelfCheck(
        ok=not failures,
        required=required,
        effective_profile=profile_value if profile_enabled else "",
        draft_path=current_draft_path(dialogue_contract_pipeline_enabled=dialogue_contract_pipeline_enabled),
        active_guards=active_guards,
        quality_guards=quality_guards,
        failures=tuple(dict.fromkeys(failures)),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def raise_for_failed_selfcheck(check: PilotProfileSelfCheck) -> None:
    if check.ok:
        return
    detail = ", ".join(check.failures)
    raise SystemExit(f"Canonical pilot profile self-check failed: {detail}")


def stderr_warning(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)
