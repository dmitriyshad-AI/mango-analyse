from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml

from mango_mvp.question_catalog import normalization


REGISTRY_PATH = Path(__file__).with_name("parameters_registry.yaml")
SCHEMA_PATH = Path(__file__).parent / "schemas" / "parameters_registry_schema_v1.yaml"


@dataclass(frozen=True)
class ParameterMatch:
    parameter_id: str
    value: str
    matched_by: str | None = None


@lru_cache(maxsize=1)
def load_parameters_registry(path: str | Path | None = None) -> Mapping[str, Any]:
    registry_path = Path(path) if path else REGISTRY_PATH
    return yaml.safe_load(registry_path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_parameters_schema(path: str | Path | None = None) -> Mapping[str, Any]:
    schema_path = Path(path) if path else SCHEMA_PATH
    return yaml.safe_load(schema_path.read_text(encoding="utf-8"))


def validate_parameters_registry(registry: Mapping[str, Any] | None = None) -> list[str]:
    registry = registry or load_parameters_registry()
    schema = load_parameters_schema()
    errors: list[str] = []

    parameters = registry.get("parameters", [])
    required_ids = set(schema["required_parameter_ids"])
    actual_ids = {item.get("parameter_id") for item in parameters}
    if len(parameters) != schema["expected_counts"]["parameters"]:
        errors.append(f"expected 8 parameters, got {len(parameters)}")
    if actual_ids != required_ids:
        errors.append(f"parameter ids mismatch: missing={sorted(required_ids - actual_ids)}, extra={sorted(actual_ids - required_ids)}")

    allowed_methods = set(schema["allowed_extraction_methods"])
    required_fields = set(schema["parameter_required_fields"])
    missing_value = schema["required_missing_value"]
    for item in parameters:
        parameter_id = item.get("parameter_id", "<missing>")
        missing = required_fields - set(item)
        if missing:
            errors.append(f"{parameter_id}: missing fields {sorted(missing)}")
        values = item.get("values", [])
        if not isinstance(values, list) or not values:
            errors.append(f"{parameter_id}: values must be a non-empty list")
            continue
        if len(values) != len(set(values)):
            errors.append(f"{parameter_id}: duplicate values")
        if missing_value not in values:
            errors.append(f"{parameter_id}: missing '{missing_value}'")
        if item.get("fallback_value") not in values:
            errors.append(f"{parameter_id}: fallback_value is outside values")
        if item.get("extraction_method") not in allowed_methods:
            errors.append(f"{parameter_id}: bad extraction_method {item.get('extraction_method')!r}")
        patterns = item.get("extraction_patterns", {})
        if item.get("extraction_method") == "regex":
            for value in values:
                if value == missing_value:
                    continue
                if value not in patterns:
                    errors.append(f"{parameter_id}: value {value!r} has no extraction_patterns entry")
                    continue
                aliases = patterns[value].get("aliases", [])
                raw_patterns = patterns[value].get("patterns", [])
                if not aliases and not raw_patterns and value not in {"низкая", "нейтральный"}:
                    errors.append(f"{parameter_id}: value {value!r} has neither aliases nor literal patterns")
                for alias in aliases:
                    try:
                        _resolve_alias(alias)
                    except KeyError as exc:
                        errors.append(f"{parameter_id}: cannot resolve alias {alias!r}: {exc}")
                for pattern in raw_patterns:
                    try:
                        re.compile(pattern, re.I)
                    except re.error as exc:
                        errors.append(f"{parameter_id}: bad regex for {value!r}: {exc}")
        elif patterns:
            errors.append(f"{parameter_id}: external_lookup must not define regex patterns")
    return errors


def extract_parameters(text: Any, registry: Mapping[str, Any] | None = None) -> dict[str, str]:
    return {
        match.parameter_id: match.value
        for match in extract_parameter_matches(text, registry=registry).values()
    }


def extract_parameter_matches(
    text: Any,
    registry: Mapping[str, Any] | None = None,
) -> dict[str, ParameterMatch]:
    registry = registry or load_parameters_registry()
    cleaned = normalization.clean_text(text)
    matches: dict[str, ParameterMatch] = {}
    for parameter in registry["parameters"]:
        parameter_id = parameter["parameter_id"]
        if parameter["extraction_method"] == "external_lookup":
            matches[parameter_id] = ParameterMatch(parameter_id, parameter["fallback_value"], "external_lookup_missing")
            continue
        value, matched_by = _extract_regex_parameter(cleaned, parameter)
        matches[parameter_id] = ParameterMatch(parameter_id, value, matched_by)
    return matches


def _extract_regex_parameter(text: str, parameter: Mapping[str, Any]) -> tuple[str, str | None]:
    patterns = parameter.get("extraction_patterns", {})
    order = parameter.get("extraction_order") or [
        value for value in parameter["values"] if value != "не_указано"
    ]
    for value in order:
        value_patterns = patterns.get(value, {})
        for alias in value_patterns.get("aliases", []):
            if _resolve_alias(alias).search(text):
                return value, alias
        for pattern in value_patterns.get("patterns", []):
            if re.search(pattern, text, re.I):
                return value, pattern
    return parameter["fallback_value"], None


@lru_cache(maxsize=None)
def _resolve_alias(alias: str) -> re.Pattern[str]:
    if alias.startswith("normalization.PRODUCT_PATTERNS."):
        key = alias.rsplit(".", 1)[-1]
        return _pattern_from_sequence(normalization.PRODUCT_PATTERNS, key)
    if alias.startswith("normalization.SUBJECT_PATTERNS."):
        key = alias.rsplit(".", 1)[-1]
        return _pattern_from_sequence(normalization.SUBJECT_PATTERNS, key)
    if alias.startswith("normalization.INTENT_PATTERNS."):
        key = alias.rsplit(".", 1)[-1]
        for item_key, _label, _facts, pattern in normalization.INTENT_PATTERNS:
            if item_key == key:
                return pattern
        raise KeyError(key)
    if alias.startswith("normalization.INTENT_SUBCLASS_PATTERNS."):
        tail = alias.removeprefix("normalization.INTENT_SUBCLASS_PATTERNS.")
        group, key = tail.split(".", 1)
        for item_key, _label, pattern in normalization.INTENT_SUBCLASS_PATTERNS.get(group, ()):
            if item_key == key:
                return pattern
        raise KeyError(alias)
    raise KeyError(alias)


def _pattern_from_sequence(items: tuple[Any, ...], key: str) -> re.Pattern[str]:
    for item_key, _label, pattern in items:
        if item_key == key:
            return pattern
    raise KeyError(key)
