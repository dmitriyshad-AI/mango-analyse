from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from mango_mvp.productization.owner_only_io import read_stable_regular_bytes


EXPECTED_CONFIG_SHA_ENV = "MANGO_CALLS_EXPECTED_CONFIG_SHA256"


def strict_service_flags(payload: Mapping[str, Any]) -> tuple[bool, bool]:
    authority_mode = str(
        payload.get("runtime_authority_mode") or "service_cutover"
    ).strip()
    if authority_mode not in {"service_cutover", "isolated_controlled"}:
        raise ValueError("runtime_authority_mode is invalid")
    values: list[bool] = []
    for name in ("require_cutover_authority", "strict_ready_provenance"):
        default = False if (
            authority_mode == "isolated_controlled"
            and name == "require_cutover_authority"
        ) else True
        value = payload[name] if name in payload else default
        if type(value) is not bool:
            raise ValueError(f"{name} must be a JSON boolean")
        values.append(value)
    invalid = (
        tuple(values) != (False, True)
        if authority_mode == "isolated_controlled"
        else values[0] != values[1]
    )
    if invalid:
        raise ValueError(
            "runtime authority flags must be enabled together or match "
            "isolated runtime_authority_mode"
        )
    return values[0], values[1]


def decode_runtime_config_bytes(raw: bytes) -> Mapping[str, Any]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("config must be valid UTF-8 JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("config must be a JSON object")
    strict_service_flags(payload)
    return payload


def load_owner_only_runtime_config_snapshot(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[Mapping[str, Any], str]:
    raw = read_stable_regular_bytes(
        path,
        label="mango_calls_runtime_config",
        owner_only_mode=0o600,
    )
    digest = hashlib.sha256(raw).hexdigest()
    expected = (
        expected_sha256
        if expected_sha256 is not None
        else os.getenv(EXPECTED_CONFIG_SHA_ENV, "")
    ).strip()
    if expected and (
        not re.fullmatch(r"[0-9a-f]{64}", expected)
        or not hmac.compare_digest(expected, digest)
    ):
        raise RuntimeError("mango_calls_runtime_config_snapshot_mismatch")
    return decode_runtime_config_bytes(raw), digest


def load_owner_only_runtime_config(
    path: Path, *, expected_sha256: str | None = None
) -> Mapping[str, Any]:
    return load_owner_only_runtime_config_snapshot(
        path, expected_sha256=expected_sha256
    )[0]


def runtime_shell_fields(payload: Mapping[str, Any]) -> tuple[str, str, str, str]:
    python_executable = str(payload.get("python_executable") or "").strip()
    pipeline_root = str(payload.get("pipeline_root") or "").strip()
    codex_binary = str(payload.get("codex_binary") or "codex").strip()
    if not python_executable or not pipeline_root:
        raise ValueError("python_executable and pipeline_root are required")
    if any(
        character in python_executable + pipeline_root + codex_binary
        for character in "\r\n\0"
    ):
        raise ValueError("runtime paths must not contain control characters")
    require_cutover, strict_provenance = strict_service_flags(payload)
    strict_mode = "true" if require_cutover or strict_provenance else "false"
    return python_executable, pipeline_root, strict_mode, codex_binary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    payload, digest = load_owner_only_runtime_config_snapshot(args.config)
    for value in (*runtime_shell_fields(payload), digest):
        print(value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
