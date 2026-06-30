from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RE_EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.[a-z]{2,}", re.I)
RE_PHONE = re.compile(r"(?:\+?\d[\s().-]*){7,}\d")
RE_LONG_NUMBER = re.compile(r"\b\d{4,}\b")
RE_TELEGRAM = re.compile(r"@\w{4,}")
RE_PHONE_FRAGMENT = re.compile(r"\b\d{2,3}[-\s]\d{2}[-\s]\d{2}\b")
RE_RU_NAME_PAIR = re.compile(r"\b[А-ЯЁ][а-яё]{2,}\s+[А-ЯЁ][а-яё]{2,}\b")
RE_LATIN_NAME_PAIR = re.compile(r"\b[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}\b")
RE_GREETING_NAME = re.compile(r"(^|\n)(\s*)[А-ЯЁ][а-яё]{2,}(?:\s+[А-ЯЁ][а-яё]{2,})?\s*,")

QUOTE_HEADER = re.compile(
    r"^\s*(-{2,}\s*original message|-{2,}\s*пересылаемое|исходное сообщение|>+|on .+ wrote:|"
    r".+ (написал|написала|wrote)\s*:|\d{1,2}\.\d{1,2}\.\d{2,4}.*(пишет|написал)|"
    r"от кого:|кому:|отправлено:|sent:|from:\s)",
    re.I,
)
SIGNATURE_DIVIDER = re.compile(r"^\s*--\s*$")
FOOTER_HINT = re.compile(r"(с уважением|best regards|данное сообщение.*конфиденц|отписаться|unsubscribe|©)", re.I)


@dataclass(frozen=True)
class SummaryItem:
    message_sha256: str
    direction: str
    brand: str
    brand_source: str
    subject: str
    body: str


@dataclass(frozen=True)
class SummaryResult:
    summaries: dict[str, dict[str, Any]]
    llm_calls_total: int
    provider: str
    model: str
    reasoning: str


def clean_body(text: str, *, limit: int = 2600) -> str:
    if not text:
        return ""
    lines: list[str] = []
    for line in text.replace("\r\n", "\n").split("\n"):
        stripped = line.strip()
        if QUOTE_HEADER.match(stripped) or SIGNATURE_DIVIDER.match(stripped):
            break
        if stripped.startswith(">"):
            continue
        if FOOTER_HINT.search(stripped) and len("\n".join(lines)) > 120:
            break
        lines.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()[:limit]


def mask_pii(text: str) -> str:
    value = (text or "").replace("&nbsp;", " ")
    value = RE_EMAIL.sub("[email]", value)
    value = RE_PHONE.sub("[phone]", value)
    value = RE_PHONE_FRAGMENT.sub("[phone]", value)
    value = RE_TELEGRAM.sub("[handle]", value)
    value = RE_LONG_NUMBER.sub("[number]", value)
    value = RE_RU_NAME_PAIR.sub("[name]", value)
    value = RE_LATIN_NAME_PAIR.sub("[name]", value)
    value = RE_GREETING_NAME.sub(r"\1\2[name],", value)
    return value


def summarize_items(
    items: list[SummaryItem],
    *,
    provider: str,
    model: str,
    reasoning: str,
    batch_size: int,
    max_llm_calls: int,
    project_root: Path,
    codex_home: Path | None = None,
    timeout_sec: int = 240,
) -> SummaryResult:
    if not items:
        return SummaryResult({}, 0, provider, model, reasoning)
    resolved_provider = provider
    if provider == "auto":
        resolved_provider = "openai" if os.getenv("OPENAI_API_KEY") else "codex_cli"
    summaries: dict[str, dict[str, Any]] = {}
    llm_calls = 0
    for offset in range(0, len(items), batch_size):
        if llm_calls >= max_llm_calls:
            raise RuntimeError(f"LLM call limit exceeded: {llm_calls} >= {max_llm_calls}")
        batch = items[offset : offset + batch_size]
        prompt = build_summary_prompt(batch)
        if resolved_provider == "openai":
            payload = _call_openai_json(prompt, model=model, reasoning=reasoning, timeout_sec=timeout_sec)
        elif resolved_provider == "codex_cli":
            payload = _call_codex_json(
                prompt,
                model=model,
                reasoning=reasoning,
                project_root=project_root,
                codex_home=codex_home,
                timeout_sec=timeout_sec,
            )
        elif resolved_provider == "stub":
            payload = {
                "summaries": [
                    {
                        "message_sha256": item.message_sha256,
                        "summary": "Тестовая заглушка сводки.",
                        "topic": "stub",
                        "next_step": None,
                        "confidence": 0.0,
                    }
                    for item in batch
                ]
            }
        else:
            raise RuntimeError(f"Unsupported summary provider: {resolved_provider}")
        llm_calls += 1
        for row in _extract_summaries(payload, expected={item.message_sha256 for item in batch}):
            row = dict(row)
            row["summary"] = mask_pii(str(row.get("summary") or "")).strip()
            row["topic"] = mask_pii(str(row.get("topic") or "")).strip()
            next_step = row.get("next_step")
            row["next_step"] = mask_pii(str(next_step)).strip() if next_step not in (None, "") else None
            summaries[str(row["message_sha256"])] = row
    return SummaryResult(summaries, llm_calls, resolved_provider, model, reasoning)


def build_summary_prompt(items: list[SummaryItem]) -> str:
    records = []
    for item in items:
        records.append(
            {
                "message_sha256": item.message_sha256,
                "direction": item.direction,
                "brand": item.brand,
                "brand_source": item.brand_source,
                "subject": mask_pii(item.subject)[:500],
                "body": mask_pii(clean_body(item.body))[:2400],
            }
        )
    return (
        "Ты строишь manager-only сводки e-mail для учебного центра. "
        "Верни строго JSON object с ключом summaries. Не используй markdown. "
        "На каждый входной message_sha256 верни одну сводку. "
        "Не выдумывай факты, суммы, бренды и следующие шаги. "
        "Если brand='none', не называй Фотон, УНПК, МФТИ или Физтех в summary/topic/next_step. "
        "Не возвращай телефоны, e-mail, ФИО, адреса или номера документов; если встретились, замени смысловым описанием. "
        "Схема каждой строки: {message_sha256, summary, topic, next_step, confidence}. "
        "summary: 1-2 коротких предложения по-русски; topic: 2-6 слов; next_step: строка или null; confidence: 0..1.\n\n"
        + json.dumps({"emails": records}, ensure_ascii=False)
    )


def _call_openai_json(prompt: str, *, model: str, reasoning: str, timeout_sec: int) -> dict[str, Any]:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for provider=openai")
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    instructions = "Return one valid JSON object only. No markdown."
    try:
        response = client.responses.create(
            model=model,
            instructions=instructions,
            input=prompt,
            reasoning={"effort": reasoning},
            text={"format": {"type": "json_object"}},
            timeout=timeout_sec,
        )
        text = getattr(response, "output_text", "") or ""
    except Exception:
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": instructions}, {"role": "user", "content": prompt}],
            timeout=timeout_sec,
        )
        text = response.choices[0].message.content if response.choices else ""
    return _extract_json_object(text)


def _call_codex_json(
    prompt: str,
    *,
    model: str,
    reasoning: str,
    project_root: Path,
    codex_home: Path | None,
    timeout_sec: int,
) -> dict[str, Any]:
    codex_bin = "codex"
    if shutil.which(codex_bin) is None:
        raise RuntimeError("codex binary is not available")
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["summaries"],
        "properties": {
            "summaries": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["message_sha256", "summary", "topic", "next_step", "confidence"],
                    "properties": {
                        "message_sha256": {"type": "string"},
                        "summary": {"type": "string"},
                        "topic": {"type": "string"},
                        "next_step": {"type": ["string", "null"]},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                },
            }
        },
    }
    with tempfile.NamedTemporaryFile(prefix="email_summary_", suffix=".json") as out_file, tempfile.NamedTemporaryFile(
        prefix="email_summary_schema_", suffix=".json", mode="w", encoding="utf-8"
    ) as schema_file:
        schema_file.write(json.dumps(schema, ensure_ascii=False))
        schema_file.flush()
        cmd = [
            codex_bin,
            "exec",
            "--skip-git-repo-check",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--sandbox",
            "read-only",
            "--cd",
            str(project_root.resolve()),
            "--model",
            model,
            "--output-schema",
            str(Path(schema_file.name)),
            "--output-last-message",
            str(Path(out_file.name)),
            "-c",
            f'model_reasoning_effort="{reasoning}"',
            "-",
        ]
        env = os.environ.copy()
        if codex_home is not None:
            env["CODEX_HOME"] = str(codex_home.expanduser().resolve())
        proc = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_sec,
            env=env,
        )
        raw = Path(out_file.name).read_text(encoding="utf-8", errors="ignore").strip()
    for candidate in (raw, proc.stdout or "", proc.stderr or ""):
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            return _extract_json_object(candidate)
        except Exception:
            continue
    stderr_tail = (proc.stderr or "").strip().splitlines()[-1:] or [""]
    raise RuntimeError(f"codex exec returned no JSON; rc={proc.returncode}; stderr_tail={stderr_tail[0]}")


def _extract_summaries(payload: dict[str, Any], *, expected: set[str]) -> list[dict[str, Any]]:
    rows = payload.get("summaries")
    if not isinstance(rows, list):
        raise RuntimeError("summary payload must contain summaries list")
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        sha = str(row.get("message_sha256") or "")
        if sha in expected:
            seen.add(sha)
            output.append(row)
    missing = expected - seen
    if missing:
        raise RuntimeError(f"summary payload missing {len(missing)} message_sha256 values")
    return output


def _extract_json_object(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(raw[start : end + 1])
    if not isinstance(payload, dict):
        raise RuntimeError("JSON root must be object")
    return payload
