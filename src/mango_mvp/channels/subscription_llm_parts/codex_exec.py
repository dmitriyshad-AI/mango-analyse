from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


DEFAULT_CODEX_MODEL = "gpt-5.5"


DEFAULT_CODEX_REASONING_EFFORT = "medium"

CODEX_ENV_ISOLATION_ENV = "TELEGRAM_CODEX_ENV_ISOLATION"

CODEX_EXEC_ENV_ALLOWLIST = frozenset(
    {
        "PATH",
        "HOME",
        "TMPDIR",
        "TEMP",
        "TMP",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "NO_PROXY",
        "https_proxy",
        "http_proxy",
        "no_proxy",
        "SSL_CERT_FILE",
        "CODEX_CA_CERTIFICATE",
        "CODEX_HOME",
    }
)
CODEX_EXEC_SECRET_EXACT = frozenset({"OPENAI_API_KEY"})
CODEX_EXEC_SECRET_PREFIXES = ("AMO_", "WAPPI_", "CRM_", "AI_OFFICE_")
CODEX_EXEC_SECRET_SUFFIXES = ("_TOKEN", "_SECRET")
CODEX_HOME_COPY_ALLOWLIST = ("auth.json", "models_cache.json", "installation_id")
CODEX_HOME_MARKER = ".mango_isolated_codex_home"
CODEX_HOME_TEMP_PREFIX = "mango_codex_home_"
CODEX_EXEC_NEUTRAL_CONFIG = """approval_policy = "never"
sandbox_mode = "read-only"
model_reasoning_effort = "medium"

[features]
multi_agent = false
js_repl = false
"""


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


def _truthy_env_value(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def codex_env_isolation_enabled(base_env: Optional[Mapping[str, str]] = None) -> bool:
    source = os.environ if base_env is None else base_env
    return _truthy_env_value(source.get(CODEX_ENV_ISOLATION_ENV))


def _csv_env_names(value: object) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(value or "").split(",") if part.strip())


def _is_secret_env_name(name: str) -> bool:
    upper = str(name or "").upper()
    if upper in CODEX_EXEC_SECRET_EXACT:
        return True
    if upper.endswith(CODEX_EXEC_SECRET_SUFFIXES):
        return True
    return upper.startswith(CODEX_EXEC_SECRET_PREFIXES)


def _is_allowed_codex_env_name(name: str, extra_passthrough: Iterable[str]) -> bool:
    if name == "CODEX_HOME":
        return False
    return name in CODEX_EXEC_ENV_ALLOWLIST or name.startswith("LC_") or name in extra_passthrough


def _safe_codex_home_entry(entry: str) -> Path | None:
    candidate = Path(str(entry or "").strip())
    if not candidate.parts or candidate.is_absolute() or ".." in candidate.parts:
        return None
    return candidate


def _chmod_tree(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_dir():
            path.chmod(0o700)
        else:
            path.chmod(0o600)
    root.chmod(0o700)


def _source_codex_home(base_env: Optional[Mapping[str, str]] = None) -> Path:
    source = os.environ if base_env is None else base_env
    raw = source.get("CODEX_HOME") if source is not None else None
    if raw:
        return Path(str(raw)).expanduser()
    return Path.home() / ".codex"


def prepare_isolated_codex_home(
    *,
    source_home: Optional[Path | str] = None,
    allowlist: Iterable[str] = CODEX_HOME_COPY_ALLOWLIST,
    neutral_config: str = CODEX_EXEC_NEUTRAL_CONFIG,
    prefix: str = CODEX_HOME_TEMP_PREFIX,
) -> str:
    source_root = Path(source_home).expanduser() if source_home is not None else _source_codex_home()
    runtime_root = Path(tempfile.mkdtemp(prefix=prefix)).resolve()
    runtime_root.chmod(0o700)
    (runtime_root / CODEX_HOME_MARKER).write_text("managed by mango codex env isolation\n", encoding="utf-8")
    (runtime_root / CODEX_HOME_MARKER).chmod(0o600)

    if source_root.exists():
        for entry in allowlist:
            safe_entry = _safe_codex_home_entry(str(entry))
            if safe_entry is None:
                continue
            source_path = source_root / safe_entry
            target_path = runtime_root / safe_entry
            if not source_path.exists():
                continue
            if source_path.is_dir():
                shutil.copytree(source_path, target_path, dirs_exist_ok=True)
                _chmod_tree(target_path)
            else:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, target_path)
                target_path.chmod(0o600)

    config_path = runtime_root / "config.toml"
    config_path.write_text(str(neutral_config or CODEX_EXEC_NEUTRAL_CONFIG), encoding="utf-8")
    config_path.chmod(0o600)
    return str(runtime_root)


def cleanup_isolated_codex_home(path: Optional[Path | str]) -> None:
    if path is None:
        return
    runtime_root = Path(path).expanduser().resolve()
    if not (runtime_root / CODEX_HOME_MARKER).exists():
        return
    shutil.rmtree(runtime_root, ignore_errors=True)


def build_codex_exec_env(
    base_env: Optional[Mapping[str, str]] = None,
    *,
    codex_home: Optional[Path | str] = None,
    extra_passthrough: Iterable[str] = (),
) -> dict[str, str]:
    source = os.environ if base_env is None else base_env
    configured_passthrough = _csv_env_names(source.get("TASK_CONTAINER_ENV_PASSTHROUGH"))
    passthrough = frozenset((*configured_passthrough, *tuple(extra_passthrough or ())))
    env: dict[str, str] = {}
    for key, value in source.items():
        key = str(key)
        if _is_secret_env_name(key):
            continue
        if _is_allowed_codex_env_name(key, passthrough):
            env[key] = str(value)
    if codex_home is not None:
        env["CODEX_HOME"] = str(Path(codex_home).expanduser().resolve())
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
