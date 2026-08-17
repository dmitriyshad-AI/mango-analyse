from __future__ import annotations

import json
from pathlib import Path

import pytest

from mango_mvp.services.llm_response_cache import LLMResponseCache, input_sha256


IDENTITY = {
    "namespace": "analysis",
    "provider": "openai",
    "model": "gpt-test",
    "reasoning": "high",
    "prompt_version": "v3",
    "prompt": "exact prompt",
}
RESPONSE = {"answer": "cached"}


def _cache_path(cache: LLMResponseCache, **overrides: str) -> Path:
    request = {**IDENTITY, **overrides}
    return cache._cache_path(**request)


def _put(cache: LLMResponseCache) -> None:
    cache.put(**IDENTITY, response=RESPONSE)


def test_cache_persists_and_verifies_complete_identity(tmp_path: Path) -> None:
    cache = LLMResponseCache(enabled=True, root_dir=tmp_path)

    _put(cache)

    payload = json.loads(_cache_path(cache).read_text(encoding="utf-8"))
    assert payload == {
        "namespace": IDENTITY["namespace"],
        "provider": IDENTITY["provider"],
        "model": IDENTITY["model"],
        "reasoning": IDENTITY["reasoning"],
        "prompt_version": IDENTITY["prompt_version"],
        "input_sha256": input_sha256(IDENTITY["prompt"]),
        "response": RESPONSE,
    }
    assert cache.get(**IDENTITY) == RESPONSE


@pytest.mark.parametrize(
    ("field", "tampered_value"),
    [
        ("namespace", "other-namespace"),
        ("provider", "other-provider"),
        ("model", "other-model"),
        ("reasoning", "low"),
        ("prompt_version", "v2"),
        ("input_sha256", "0" * 64),
    ],
)
def test_tampered_identity_is_safe_miss(
    tmp_path: Path, field: str, tampered_value: str
) -> None:
    cache = LLMResponseCache(enabled=True, root_dir=tmp_path)
    _put(cache)
    path = _cache_path(cache)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = tampered_value
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert cache.get(**IDENTITY) is None


def test_entry_moved_to_another_namespace_is_safe_miss(tmp_path: Path) -> None:
    cache = LLMResponseCache(enabled=True, root_dir=tmp_path)
    _put(cache)
    moved_path = _cache_path(cache, namespace="other-namespace")
    moved_path.parent.mkdir(parents=True)
    moved_path.write_bytes(_cache_path(cache).read_bytes())

    assert cache.get(**{**IDENTITY, "namespace": "other-namespace"}) is None


def test_legacy_entry_without_namespace_is_safe_miss(tmp_path: Path) -> None:
    cache = LLMResponseCache(enabled=True, root_dir=tmp_path)
    _put(cache)
    path = _cache_path(cache)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("namespace")
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert cache.get(**IDENTITY) is None
