from __future__ import annotations

import json
import os
import re
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional


DEFAULT_CODEX_MODEL = "gpt-5.5"


DEFAULT_CODEX_REASONING_EFFORT = "medium"


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


def build_codex_exec_command(
    *,
    output_path: Path | str,
    codex_bin: str = "codex",
    model: str = DEFAULT_CODEX_MODEL,
    reasoning_effort: str = DEFAULT_CODEX_REASONING_EFFORT,
    isolated: bool = False,
    cwd: Optional[Path | str] = None,
) -> list[str]:
    cmd = [
        str(codex_bin or "codex").strip() or "codex",
        "--ask-for-approval",
        "never",
        "exec",
    ]
    if isolated:
        cmd.extend(["--ignore-user-config", "--ignore-rules"])
    cmd.extend(["--skip-git-repo-check", "--ephemeral", "--sandbox", "read-only"])
    if cwd is not None:
        cmd.extend(["-C", str(cwd)])
    cmd.extend(["--model", str(model or DEFAULT_CODEX_MODEL).strip() or DEFAULT_CODEX_MODEL])
    reasoning = str(reasoning_effort or "").strip()
    if reasoning:
        cmd.extend(["-c", f'model_reasoning_effort="{reasoning}"'])
    cmd.extend(["--output-last-message", str(output_path), "-"])
    return cmd


@contextmanager
def codex_isolation_cwd(enabled: bool):
    if not enabled:
        yield None
        return
    with tempfile.TemporaryDirectory(prefix="mango_bot_codex_empty_") as tmp_dir:
        yield Path(tmp_dir)


def _with_codex_exec_metadata(metadata: Mapping[str, Any], *, isolated: bool) -> dict[str, Any]:
    current = dict(metadata or {})
    existing = current.get("codex_exec")
    codex_exec = dict(existing) if isinstance(existing, Mapping) else {}
    codex_exec.update(
        {
            "isolated": bool(isolated),
            "ignore_user_config": bool(isolated),
            "ignore_rules": bool(isolated),
        }
    )
    current["codex_exec"] = codex_exec
    return current


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
    isolated: bool = False
    cwd: Optional[Path | str] = None

    def build_command(self, output_path: Path | str) -> list[str]:
        return build_codex_exec_command(
            output_path=output_path,
            codex_bin=self.codex_bin,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            isolated=self.isolated,
            cwd=self.cwd,
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


def _cache_key(payload: Mapping[str, Any]) -> str:
    import hashlib

    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


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


class _PromptProviderError(RuntimeError):
    pass
