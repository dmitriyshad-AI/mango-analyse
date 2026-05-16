from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

import yaml

from mango_mvp.services.llm_response_cache import LLMResponseCache


TAXONOMY_PATH = Path(__file__).with_name("themes_taxonomy.yaml")
NAMESPACE = "theme_assigner_v2"
PROMPT_VERSION = "v1"
PROVIDER_OPENAI = "openai"


class ThemeAssignerError(RuntimeError):
    pass


class ChatCompletionClient(Protocol):
    chat: Any


@dataclass(frozen=True)
class ThemeAssignerConfig:
    provider: str = PROVIDER_OPENAI
    primary_model: str = "gpt-4o-mini"
    escalation_model: str = "gpt-5.5"
    temperature: float = 0.0
    llm_confidence_threshold: float = 0.7
    macro_f1_threshold: float = 0.85
    prompt_version: str = PROMPT_VERSION
    cache_enabled: bool = True
    cache_root_dir: str | Path = ".cache/llm_responses"
    max_concurrency: int = 10
    max_retries: int = 3
    retry_initial_seconds: float = 0.35
    request_timeout_seconds: float = 60.0
    enable_escalation: bool = True

    @classmethod
    def from_env(cls) -> "ThemeAssignerConfig":
        return cls(
            primary_model=os.getenv("QUESTION_CATALOG_PRIMARY_LLM_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini",
            escalation_model=os.getenv("QUESTION_CATALOG_ESCALATION_LLM_MODEL", "gpt-5.5").strip() or "gpt-5.5",
            llm_confidence_threshold=_float_env("QUESTION_CATALOG_LLM_CONFIDENCE_THRESHOLD", 0.7),
            macro_f1_threshold=_float_env("QUESTION_CATALOG_MACRO_F1_THRESHOLD", 0.85),
            cache_enabled=_bool_env("LLM_CACHE_ENABLED", True),
            cache_root_dir=os.getenv("LLM_CACHE_DIR", ".cache/llm_responses").strip() or ".cache/llm_responses",
            max_concurrency=_int_env("QUESTION_CATALOG_LLM_MAX_CONCURRENCY", 10),
            max_retries=_int_env("QUESTION_CATALOG_LLM_MAX_RETRIES", 3),
            enable_escalation=_bool_env("QUESTION_CATALOG_LLM_ENABLE_ESCALATION", True),
        )


@dataclass(frozen=True)
class ThemeAssignmentResult:
    theme_id: str
    confidence: float
    reasoning: str
    provider: str = PROVIDER_OPENAI
    model: str = ""
    cache_hit: bool = False
    raw_response: Mapping[str, Any] = field(default_factory=dict)


def assign_theme_llm(
    raw_text: str,
    params: Mapping[str, str],
    *,
    config: ThemeAssignerConfig | None = None,
    client: ChatCompletionClient | None = None,
    cache: LLMResponseCache | None = None,
    taxonomy_path: str | Path | None = None,
) -> ThemeAssignmentResult:
    resolved_config = config or ThemeAssignerConfig.from_env()
    primary = _assign_with_model(
        raw_text,
        params,
        model=resolved_config.primary_model,
        config=resolved_config,
        client=client,
        cache=cache,
        taxonomy_path=taxonomy_path,
    )
    if (
        primary.confidence < resolved_config.llm_confidence_threshold
        and resolved_config.enable_escalation
        and resolved_config.escalation_model
        and resolved_config.escalation_model != resolved_config.primary_model
    ):
        escalation = _assign_with_model(
            raw_text,
            params,
            model=resolved_config.escalation_model,
            config=resolved_config,
            client=client,
            cache=cache,
            taxonomy_path=taxonomy_path,
        )
        if escalation.confidence >= primary.confidence:
            return escalation
    return primary


def batch_assign_theme_llm(
    rows: Iterable[Mapping[str, Any]],
    *,
    config: ThemeAssignerConfig | None = None,
    client: ChatCompletionClient | None = None,
    cache: LLMResponseCache | None = None,
    taxonomy_path: str | Path | None = None,
) -> list[ThemeAssignmentResult]:
    resolved_config = config or ThemeAssignerConfig.from_env()
    row_list = list(rows)

    def worker(row: Mapping[str, Any]) -> ThemeAssignmentResult:
        raw_text = str(row.get("raw_text") or row.get("question") or "")
        params = row.get("params") if isinstance(row.get("params"), Mapping) else row.get("extracted_params")
        if not isinstance(params, Mapping):
            params = {}
        return assign_theme_llm(
            raw_text,
            {str(key): str(value) for key, value in params.items()},
            config=resolved_config,
            client=client,
            cache=cache,
            taxonomy_path=taxonomy_path,
        )

    with ThreadPoolExecutor(max_workers=max(1, resolved_config.max_concurrency)) as pool:
        return list(pool.map(worker, row_list))


def build_theme_prompt(
    raw_text: str,
    params: Mapping[str, str],
    *,
    taxonomy_path: str | Path | None = None,
) -> str:
    params_json = json.dumps(dict(sorted(params.items())), ensure_ascii=False, sort_keys=True)
    return (
        "Ты классификатор клиентских вопросов для образовательной компании Фотон / УНПК МФТИ.\n\n"
        "Тебе даётся:\n"
        "1. Текст вопроса клиента (одно сообщение или фрагмент звонка)\n"
        "2. Уже извлечённые параметры: {product, subject, grade, format}\n\n"
        "Твоя задача: определить, к какой ОДНОЙ из 32 тем относится вопрос.\n\n"
        "Список тем:\n"
        f"{_taxonomy_prompt_block(taxonomy_path)}\n\n"
        "Если вопрос не подходит ни к одной теме — выбери одну из служебных категорий:\n"
        "  service:S1_non_question — обрывок без вопроса, развернутый пересказ\n"
        "  service:S2_unclear — слишком короткий контекст\n"
        "  service:S3_out_of_scope — отказ, неактуально, ошибся номером\n"
        "  service:S4_status_request — клиент спрашивает «когда ответят»\n"
        "  service:S5_general_consultation — общий запрос без конкретной темы\n\n"
        "Правила:\n"
        "- Выбирай ОДНУ тему, ту, что ЛУЧШЕ ВСЕГО подходит\n"
        "- Не путай \"вопрос о возврате денег\" (theme:009_refund) с \"вопросом о налоговом вычете\" (theme:008_tax_deduction) — это разные процедуры\n"
        "- Не путай \"статус оплаты\" (theme:003_payment_status) и \"способ оплаты\" (theme:002_payment_method)\n"
        "- Если клиент спрашивает про оплату маткапиталом — это theme:007_matkap_payment, не общая оплата\n"
        "- Обратная связь делится по тону: позитив/благодарность → theme:019a_positive_feedback; негатив/жалоба/претензия → theme:019b_negative_feedback\n"
        "- Запрос родителя о прогрессе/посещаемости/отметках ребёнка — это theme:032_student_progress_inquiry, не обратная связь\n"
        "- Если контекст слишком короткий чтобы понять — выбирай service:S2_unclear, не угадывай\n\n"
        "Верни JSON:\n"
        "{\n"
        "  \"theme_id\": \"theme:001_pricing\",\n"
        "  \"confidence\": 0.95,\n"
        "  \"reasoning\": \"Клиент явно спрашивает цену; параметр product=летняя_школа, subject=математика.\"\n"
        "}\n\n"
        f"Вопрос клиента: {raw_text}\n"
        f"Извлечённые параметры: {params_json}\n"
    )


def valid_theme_and_service_ids(taxonomy_path: str | Path | None = None) -> set[str]:
    taxonomy = load_taxonomy(taxonomy_path)
    return {
        *(str(item["theme_id"]) for item in taxonomy["themes"]),
        *(str(item["service_id"]) for item in taxonomy["service_categories"]),
    }


@lru_cache(maxsize=4)
def load_taxonomy(path: str | Path | None = None) -> Mapping[str, Any]:
    taxonomy_path = Path(path) if path else TAXONOMY_PATH
    return yaml.safe_load(taxonomy_path.read_text(encoding="utf-8"))


@lru_cache(maxsize=4)
def _taxonomy_prompt_block(path: str | Path | None = None) -> str:
    taxonomy = load_taxonomy(path)
    lines: list[str] = []
    for item in taxonomy["themes"]:
        examples = ", ".join(f"\"{text}\"" for text in item.get("example_phrasings", [])[:3])
        lines.append(
            f"  {item['theme_id']} — \"{item['theme_name']}\" "
            f"({item['short_description']} Например: {examples})"
        )
    return "\n".join(lines)


def _assign_with_model(
    raw_text: str,
    params: Mapping[str, str],
    *,
    model: str,
    config: ThemeAssignerConfig,
    client: ChatCompletionClient | None,
    cache: LLMResponseCache | None,
    taxonomy_path: str | Path | None,
) -> ThemeAssignmentResult:
    prompt = build_theme_prompt(raw_text, params, taxonomy_path=taxonomy_path)
    resolved_cache = cache or LLMResponseCache(enabled=config.cache_enabled, root_dir=config.cache_root_dir)
    reasoning = f"temperature={config.temperature}"
    cached = resolved_cache.get(
        namespace=NAMESPACE,
        provider=config.provider,
        model=model,
        reasoning=reasoning,
        prompt_version=config.prompt_version,
        prompt=prompt,
    )
    if cached is not None:
        return _parse_llm_response(cached, model=model, provider=config.provider, cache_hit=True, taxonomy_path=taxonomy_path)

    response_payload: dict[str, Any] | None = None
    last_error: Exception | None = None
    for attempt in range(max(1, config.max_retries)):
        try:
            response_payload = _call_openai_json(
                prompt,
                model=model,
                config=config,
                client=client,
            )
            break
        except Exception as exc:  # noqa: BLE001 - classifier catches and falls back to rules.
            last_error = exc
            if attempt >= config.max_retries - 1:
                break
            time.sleep(config.retry_initial_seconds * (2**attempt))
    if response_payload is None:
        raise ThemeAssignerError(f"theme assigner LLM failed: {last_error}") from last_error

    result = _parse_llm_response(response_payload, model=model, provider=config.provider, cache_hit=False, taxonomy_path=taxonomy_path)
    resolved_cache.put(
        namespace=NAMESPACE,
        provider=config.provider,
        model=model,
        reasoning=reasoning,
        prompt_version=config.prompt_version,
        prompt=prompt,
        response=dict(response_payload),
    )
    return result


def _call_openai_json(
    prompt: str,
    *,
    model: str,
    config: ThemeAssignerConfig,
    client: ChatCompletionClient | None,
) -> dict[str, Any]:
    resolved_client = client or _openai_client(config)
    response = resolved_client.chat.completions.create(
        model=model,
        temperature=config.temperature,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )
    content = response.choices[0].message.content if getattr(response, "choices", None) else None
    if not content:
        raise ThemeAssignerError("theme assigner returned empty content")
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise ThemeAssignerError("theme assigner response must be a JSON object")
    return payload


def _parse_llm_response(
    payload: Mapping[str, Any],
    *,
    model: str,
    provider: str,
    cache_hit: bool,
    taxonomy_path: str | Path | None,
) -> ThemeAssignmentResult:
    theme_id = str(payload.get("theme_id") or "").strip()
    if theme_id not in valid_theme_and_service_ids(taxonomy_path):
        raise ThemeAssignerError(f"LLM returned unknown theme_id: {theme_id!r}")
    try:
        confidence = float(payload.get("confidence"))
    except (TypeError, ValueError) as exc:
        raise ThemeAssignerError("LLM returned invalid confidence") from exc
    if not 0 <= confidence <= 1:
        raise ThemeAssignerError(f"LLM confidence must be in 0..1, got {confidence}")
    return ThemeAssignmentResult(
        theme_id=theme_id,
        confidence=confidence,
        reasoning=str(payload.get("reasoning") or ""),
        provider=provider,
        model=model,
        cache_hit=cache_hit,
        raw_response=dict(payload),
    )


def _openai_client(config: ThemeAssignerConfig) -> ChatCompletionClient:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ThemeAssignerError("OPENAI_API_KEY is required for theme_assigner_v2")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ThemeAssignerError("openai package is required for theme_assigner_v2") from exc
    return OpenAI(api_key=api_key, timeout=config.request_timeout_seconds)


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
