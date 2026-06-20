from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

from mango_mvp.channels.subscription_llm_parts.codex_exec import extract_json_object
from mango_mvp.quality.tenant_text_normalizer import normalize_manager_text


CRM_CARD_HISTORY_SUMMARY_PROMPT_VERSION = "crm_card_history_summary_v1_2026_06_20"
DEFAULT_HISTORY_SUMMARY_PROVIDER = "off"
DEFAULT_HISTORY_SUMMARY_MODEL = "gpt-5.5"
DEFAULT_HISTORY_SUMMARY_REASONING = "low"
DEFAULT_HISTORY_SUMMARY_TIMEOUT_SEC = 120
DEFAULT_HISTORY_SUMMARY_CACHE_DIR = Path(".cache/crm_card_history_summary")

BOILERPLATE_MARKERS = (
    "end of history",
    "apache 2.0 license",
    "apache license",
    "licensed under the apache",
    "copyright",
    "all rights reserved",
)
SERVICE_TAIL_RE = re.compile(
    r"(?:\s|^)(?:Итог|Контакты)\s*:\s*[^.\n]*(?:\.|$)",
    re.IGNORECASE,
)


@dataclass
class CrmHistorySummaryStats:
    provider: str
    model: str
    prompt_version: str = CRM_CARD_HISTORY_SUMMARY_PROMPT_VERSION
    cache_hits: int = 0
    cache_misses: int = 0
    llm_calls: int = 0
    rule_fallbacks: int = 0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "llm_calls": self.llm_calls,
            "rule_fallbacks": self.rule_fallbacks,
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class CrmHistorySummaryConfig:
    provider: str = DEFAULT_HISTORY_SUMMARY_PROVIDER
    cache_dir: Path = DEFAULT_HISTORY_SUMMARY_CACHE_DIR
    model: str = DEFAULT_HISTORY_SUMMARY_MODEL
    reasoning_effort: str = DEFAULT_HISTORY_SUMMARY_REASONING
    timeout_sec: int = DEFAULT_HISTORY_SUMMARY_TIMEOUT_SEC
    codex_bin: str = "codex"
    temperature: float = 0.1

    @classmethod
    def from_env(cls) -> "CrmHistorySummaryConfig":
        return cls(
            provider=os.getenv("CRM_CARD_HISTORY_SUMMARY_PROVIDER", DEFAULT_HISTORY_SUMMARY_PROVIDER),
            cache_dir=Path(os.getenv("CRM_CARD_HISTORY_SUMMARY_CACHE_DIR", str(DEFAULT_HISTORY_SUMMARY_CACHE_DIR))),
            model=os.getenv("CRM_CARD_HISTORY_SUMMARY_MODEL", DEFAULT_HISTORY_SUMMARY_MODEL),
            reasoning_effort=os.getenv("CRM_CARD_HISTORY_SUMMARY_REASONING", DEFAULT_HISTORY_SUMMARY_REASONING),
            timeout_sec=max(15, int(os.getenv("CRM_CARD_HISTORY_SUMMARY_TIMEOUT_SEC", str(DEFAULT_HISTORY_SUMMARY_TIMEOUT_SEC)))),
            codex_bin=os.getenv("CRM_CARD_HISTORY_SUMMARY_CODEX_BIN", "codex"),
            temperature=float(os.getenv("CRM_CARD_HISTORY_SUMMARY_TEMPERATURE", "0.1")),
        )


class CrmHistorySummarizer:
    def __init__(
        self,
        config: CrmHistorySummaryConfig | None = None,
        *,
        runner: Callable[[str], str] | None = None,
    ) -> None:
        self.config = config or CrmHistorySummaryConfig.from_env()
        self.provider = str(self.config.provider or "off").strip().casefold()
        self.stats = CrmHistorySummaryStats(provider=self.provider, model=self.config.model)
        self._runner = runner

    def __call__(self, source_text: str) -> str:
        source = clean_history_source_text(source_text)
        if not source:
            return ""
        if self.provider in {"", "off", "none"}:
            return rule_history_summary(source)
        key = self._cache_key(source)
        cached = self._read_cache(key)
        if cached:
            self.stats.cache_hits += 1
            return cached
        self.stats.cache_misses += 1
        cacheable = True
        try:
            if self.provider == "rule":
                summary = rule_history_summary(source)
                self.stats.rule_fallbacks += 1
            elif self.provider == "codex_cli":
                summary = self._call_codex(source)
                self.stats.llm_calls += 1
            else:
                raise RuntimeError(f"unsupported CRM history summary provider: {self.provider}")
        except Exception as exc:  # pragma: no cover - live provider fail-soft
            self.stats.errors.append(str(exc))
            summary = rule_history_summary(source)
            self.stats.rule_fallbacks += 1
            cacheable = self.provider == "rule"
        summary = normalize_history_summary(summary)
        if summary and cacheable:
            self._write_cache(key, summary, source)
        return summary

    def summary(self) -> Mapping[str, object]:
        return self.stats.as_dict()

    def _cache_key(self, source_text: str) -> str:
        payload = {
            "prompt_version": CRM_CARD_HISTORY_SUMMARY_PROMPT_VERSION,
            "provider": self.provider,
            "model": self.config.model,
            "reasoning_effort": self.config.reasoning_effort,
            "temperature": self.config.temperature,
            "source_sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.config.cache_dir.expanduser().resolve(strict=False) / f"{key}.json"

    def _read_cache(self, key: str) -> str:
        path = self._cache_path(key)
        if not path.exists():
            return ""
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return ""
        return normalize_history_summary(payload.get("history"))

    def _write_cache(self, key: str, summary: str, source_text: str) -> None:
        path = self._cache_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "crm_history_summary_cache_v1",
            "prompt_version": CRM_CARD_HISTORY_SUMMARY_PROMPT_VERSION,
            "provider": self.provider,
            "model": self.config.model,
            "reasoning_effort": self.config.reasoning_effort,
            "temperature": self.config.temperature,
            "source_sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
            "history": summary,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _call_codex(self, source_text: str) -> str:
        prompt = build_history_summary_prompt(source_text)
        if self._runner is not None:
            raw = self._runner(prompt)
        else:
            codex_bin = (self.config.codex_bin or "codex").strip() or "codex"
            if shutil.which(codex_bin) is None:
                raise RuntimeError(f"codex binary is not available: {codex_bin}")
            with tempfile.NamedTemporaryFile(prefix="mango_crm_history_", suffix=".txt") as out_file:
                cmd = [
                    codex_bin,
                    "exec",
                    "--skip-git-repo-check",
                    "--ephemeral",
                    "--ignore-user-config",
                    "--sandbox",
                    "read-only",
                    "--model",
                    self.config.model,
                    "--output-last-message",
                    out_file.name,
                ]
                reasoning = (self.config.reasoning_effort or "").strip().lower()
                if reasoning in {"minimal", "low", "medium", "high", "xhigh"}:
                    cmd.extend(["-c", f'model_reasoning_effort="{reasoning}"'])
                cmd.append(prompt)
                started_at = time.time()
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=max(15, int(self.config.timeout_sec)),
                )
                if proc.returncode != 0:
                    tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-1:] or [""]
                    raise RuntimeError(f"codex exec failed rc={proc.returncode}: {tail[0]}")
                raw = Path(out_file.name).read_text(encoding="utf-8", errors="ignore")
                _ = time.time() - started_at
        try:
            payload = extract_json_object(raw)
            history = normalize_history_summary(payload.get("history"))
        except Exception:
            history = _extract_loose_history(raw)
        if not history:
            raise RuntimeError("CRM history summarizer returned empty history")
        return history


def build_history_summary_prompt(source_text: str) -> str:
    return (
        "Ты сжимаешь внутреннюю историю общения клиента для менеджера учебного центра.\n"
        "Нужно вернуть только полезный смысл, без выдумок и без новых фактов.\n"
        "Оставь: кто клиент/ребёнок, что интересовало, важные договорённости, возражения, риски, следующий шаг.\n"
        "Выкинь веб-мусор, лицензии, футеры, навигацию, 'End of History', 'Apache 2.0 License', "
        "служебные хвосты вида 'Итог: ...' и 'Контакты: канал: ...'.\n"
        "Не вставляй HTML/Markdown. Plain-text с короткими блоками и переносами строк.\n"
        "Верни строгий JSON: {\"history\":\"...\"}.\n\n"
        "История для сжатия:\n"
        f"{source_text}"
    )


def clean_history_source_text(value: str) -> str:
    text = "" if value is None else str(value)
    if not text.strip():
        return ""
    cleaned_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if cleaned_lines and cleaned_lines[-1]:
                cleaned_lines.append("")
            continue
        if any(marker in line.casefold() for marker in BOILERPLATE_MARKERS):
            continue
        line = SERVICE_TAIL_RE.sub("", line).strip(" ;,")
        if line:
            cleaned_lines.append(line)
    while cleaned_lines and cleaned_lines[-1] == "":
        cleaned_lines.pop()
    return "\n".join(cleaned_lines)


def normalize_history_summary(value: object) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        return ""
    lines = [normalize_manager_text(line) for line in text.splitlines()]
    result: list[str] = []
    blank = False
    for line in lines:
        if any(marker in line.casefold() for marker in BOILERPLATE_MARKERS):
            continue
        line = SERVICE_TAIL_RE.sub("", line).strip(" ;,")
        if line:
            result.append(line)
            blank = False
        elif not blank and result:
            result.append("")
            blank = True
    while result and result[-1] == "":
        result.pop()
    return "\n".join(result)


def rule_history_summary(source_text: str) -> str:
    cleaned = clean_history_source_text(source_text)
    if not cleaned:
        return ""
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if not lines:
        return ""
    selected = lines[:8]
    return normalize_history_summary("Сводка:\n" + "\n".join(f"- {line}" for line in selected))


def _extract_loose_history(raw: str) -> str:
    text = (raw or "").strip()
    match = re.search(r'"history"\s*:\s*"(.*)"\s*}\s*$', text, flags=re.S)
    if not match:
        return ""
    value = match.group(1)
    value = value.replace(r"\"", '"').replace(r"\n", "\n")
    return normalize_history_summary(value)


__all__ = [
    "CRM_CARD_HISTORY_SUMMARY_PROMPT_VERSION",
    "CrmHistorySummaryConfig",
    "CrmHistorySummarizer",
    "clean_history_source_text",
    "normalize_history_summary",
    "rule_history_summary",
]
