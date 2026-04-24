from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI

from mango_mvp.amocrm_runtime.config import get_settings
from mango_mvp.services.llm_response_cache import LLMResponseCache

settings = get_settings()

ALLOWED_CLOSE_VERDICTS = {
    "closed_valid",
    "closed_too_early",
    "follow_up_needed",
    "reopen_recommended",
    "alternative_offer_needed",
    "manual_review",
}
ALLOWED_RISK_VALUES = {
    "no_risk",
    "low",
    "medium",
    "high",
    "critical",
    "manual_review",
}
PROMPT_VERSION = "deal_llm_v2"
SYSTEM_PROMPT = """Ты анализируешь сделки российского EdTech по агрегированному досье сделки.
Верни только одну строку с minified JSON object. Никакого markdown, комментариев и лишних ключей.

Задача:
- определить, была ли сделка закрыта слишком рано;
- оценить риск premature close;
- предложить следующий шаг;
- учитывать всю доступную историю звонков, заметок, задач, хронологию общения и контекст сделки.

Правила:
- Используй только факты из dossier и heuristic_screening. Не выдумывай факты.
- heuristic_screening это вспомогательный слой, а не истина.
- Если матчинг сделки или вывод неуверенный, ставь manual_review.
- Если отказ жесткий и окончательный, не рекомендуй reopen без сильных контрсигналов.
- Если клиент говорит "позже", "после экзаменов", "не сейчас", "вернемся", это не жесткий отказ.
- Не пиши сырой transcript в ответ.
- Все тексты на русском. Даты в ISO формате YYYY-MM-DD или null.
- confidence от 0 до 1.
- evidence_signals: только факты и сигналы, которые поддерживают итоговый вердикт.
- conflict_flags: только блокирующие причины неопределенности. Используй их только когда автоматическая запись небезопасна.
- В conflict_flags НЕ нужно писать бизнес-сигналы в пользу reopen/follow-up.
- Примеры того, что должно идти в evidence_signals, а не в conflict_flags:
  - причина закрытия не подтверждается содержанием звонков;
  - сделка закрыта без выполненного follow-up;
  - после закрытия были содержательные касания;
  - согласован следующий шаг, но сделка закрыта.
- Примеры того, что может идти в conflict_flags:
  - несколько одинаково вероятных сделок;
  - критически не хватает данных;
  - противоречивые источники, из-за которых нельзя уверенно выбрать verdict;
  - неверный или неуверенный match сделки к телефону/контакту.

Верни ровно такие ключи:
{
  "analysis_schema_version": "deal_llm_v2",
  "close_verdict": "closed_valid|closed_too_early|follow_up_needed|reopen_recommended|alternative_offer_needed|manual_review",
  "premature_close_risk": "no_risk|low|medium|high|critical|manual_review",
  "close_reason_summary": "",
  "recommended_next_step": "",
  "follow_up_due_at": null,
  "deal_summary": "",
  "manager_action_summary": "",
  "confidence": 0.0,
  "needs_manual_review": false,
  "evidence_signals": [],
  "conflict_flags": []
}
"""


class DealLLMError(RuntimeError):
    pass


class DealLLMAnalyzer:
    def __init__(self) -> None:
        self._settings = settings
        self._client: Optional[OpenAI] = None
        self._cache = LLMResponseCache(
            enabled=settings.crm_analysis_llm_cache_enabled,
            root_dir=settings.crm_analysis_llm_cache_dir,
        )

    def _prepare_runtime_codex_home(self) -> str:
        source_root = Path.home() / ".codex"
        runtime_root = Path(self._settings.crm_analysis_llm_cache_dir).resolve().parent / "codex_home"
        runtime_root.mkdir(parents=True, exist_ok=True)

        allowlist = tuple(
            entry.strip()
            for entry in self._settings.task_container_codex_home_copy_allowlist
            if str(entry).strip()
        )
        if source_root.exists():
            for entry in allowlist:
                source_path = source_root / entry
                target_path = runtime_root / entry
                if not source_path.exists():
                    continue
                if target_path.exists():
                    if target_path.is_dir():
                        shutil.rmtree(target_path, ignore_errors=True)
                    else:
                        target_path.unlink(missing_ok=True)
                if source_path.is_dir():
                    shutil.copytree(source_path, target_path, dirs_exist_ok=True)
                else:
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_path, target_path)

        shutil.rmtree(runtime_root / "sessions", ignore_errors=True)
        shutil.rmtree(runtime_root / "sqlite", ignore_errors=True)
        for entry in ("state_5.sqlite", "state_5.sqlite-wal", "state_5.sqlite-shm"):
            (runtime_root / entry).unlink(missing_ok=True)
        (runtime_root / "sessions").mkdir(parents=True, exist_ok=True)
        return str(runtime_root)

    @staticmethod
    def _safe_text(value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @staticmethod
    def _parse_object_candidate(text: str) -> Optional[dict[str, Any]]:
        raw = (text or "").strip()
        if not raw:
            return None
        try:
            payload = json.loads(raw)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass
        try:
            payload = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            return None
        return payload if isinstance(payload, dict) else None

    @classmethod
    def _extract_json_payload(cls, text: str) -> dict[str, Any]:
        raw = (text or "").strip()
        if not raw:
            raise DealLLMError("empty response")
        payload = cls._parse_object_candidate(raw)
        if payload is not None:
            return payload

        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            payload = cls._parse_object_candidate(raw[start : end + 1])
            if payload is not None:
                return payload
        raise DealLLMError("response does not contain JSON object")

    @staticmethod
    def _clip_confidence(value: Any) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.0
        if score < 0:
            return 0.0
        if score > 1:
            return 1.0
        return round(score, 3)

    @staticmethod
    def _clean_list(value: Any, *, max_items: int = 8, item_max_chars: int = 240) -> list[str]:
        if not isinstance(value, list):
            return []
        result: list[str] = []
        seen: set[str] = set()
        for item in value:
            text = str(item or "").strip()
            if not text:
                continue
            text = text[:item_max_chars]
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            result.append(text)
            if len(result) >= max_items:
                break
        return result

    @classmethod
    def _compact_dossier_for_prompt(cls, dossier: dict[str, Any], heuristic_analysis: dict[str, Any]) -> dict[str, Any]:
        call_history = dossier.get("call_history") or []
        transcript_context = dossier.get("transcript_context") or []
        notes = dossier.get("notes") or []
        tasks = dossier.get("tasks") or []
        compact = {
            "dossier_schema_version": dossier.get("dossier_schema_version"),
            "phone": dossier.get("phone"),
            "contact": dossier.get("contact"),
            "lead": dossier.get("lead"),
            "contact_rollup": dossier.get("contact_rollup"),
            "tallanto_live": dossier.get("tallanto_live"),
            "manager_history": dossier.get("manager_history"),
            "call_history": call_history[:25],
            "transcript_context": transcript_context[: settings.crm_analysis_max_transcript_calls],
            "notes": notes[:20],
            "tasks": tasks[:20],
            "heuristic_screening": {
                "close_verdict": heuristic_analysis.get("close_verdict"),
                "premature_close_risk": heuristic_analysis.get("premature_close_risk"),
                "close_reason_summary": heuristic_analysis.get("close_reason_summary"),
                "recommended_next_step": heuristic_analysis.get("recommended_next_step"),
                "follow_up_due_at": heuristic_analysis.get("follow_up_due_at"),
                "match_confidence": heuristic_analysis.get("match_confidence"),
                "match_reason": heuristic_analysis.get("match_reason"),
                "loss_reason_summary": heuristic_analysis.get("loss_reason_summary"),
                "close_too_fast": heuristic_analysis.get("close_too_fast"),
            },
        }
        return compact

    @classmethod
    def _build_prompt(cls, dossier: dict[str, Any], heuristic_analysis: dict[str, Any]) -> str:
        compact = cls._compact_dossier_for_prompt(dossier, heuristic_analysis)
        dossier_json = json.dumps(compact, ensure_ascii=False, sort_keys=True)
        return (
            f"{SYSTEM_PROMPT}\n\n"
            "Ниже агрегированное досье сделки. Прими решение по закрытию и риску преждевременного закрытия.\n"
            "Если по истории клиента нужен возврат в работу или follow-up, отрази это в verdict и recommended_next_step.\n\n"
            f"dossier={dossier_json}"
        )

    def _cache_lookup(self, *, provider: str, model: str, reasoning: str, prompt: str) -> Optional[dict[str, Any]]:
        return self._cache.get(
            namespace="amocrm-deal-analysis",
            provider=provider,
            model=model,
            reasoning=reasoning,
            prompt_version=PROMPT_VERSION,
            prompt=prompt,
        )

    def _cache_store(self, *, provider: str, model: str, reasoning: str, prompt: str, response: dict[str, Any]) -> None:
        self._cache.put(
            namespace="amocrm-deal-analysis",
            provider=provider,
            model=model,
            reasoning=reasoning,
            prompt_version=PROMPT_VERSION,
            prompt=prompt,
            response=response,
        )

    def _openai_client(self) -> OpenAI:
        api_key = self._safe_text(getattr(self._settings, "crm_amo_api_token", ""))
        # no-op path: runtime currently relies on codex_cli by default; openai path requires OPENAI_API_KEY in env.
        from os import getenv
        openai_api_key = self._safe_text(getenv("OPENAI_API_KEY"))
        if not openai_api_key:
            raise DealLLMError("OPENAI_API_KEY is required for openai CRM analysis provider")
        if self._client is None:
            self._client = OpenAI(api_key=openai_api_key)
        return self._client

    def _normalize_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        verdict = self._safe_text(payload.get("close_verdict"))
        if verdict not in ALLOWED_CLOSE_VERDICTS:
            verdict = "manual_review"

        risk = self._safe_text(payload.get("premature_close_risk"))
        if risk not in ALLOWED_RISK_VALUES:
            risk = "manual_review" if verdict == "manual_review" else "medium"

        confidence = self._clip_confidence(payload.get("confidence"))
        needs_manual_review = bool(payload.get("needs_manual_review")) or verdict == "manual_review" or confidence < 0.45

        result = {
            "analysis_schema_version": PROMPT_VERSION,
            "close_verdict": verdict,
            "premature_close_risk": risk,
            "close_reason_summary": self._safe_text(payload.get("close_reason_summary"))[:2500],
            "recommended_next_step": self._safe_text(payload.get("recommended_next_step"))[:1200],
            "follow_up_due_at": self._safe_text(payload.get("follow_up_due_at")) or None,
            "deal_summary": self._safe_text(payload.get("deal_summary"))[:2500],
            "manager_action_summary": self._safe_text(payload.get("manager_action_summary"))[:1200],
            "confidence": confidence,
            "needs_manual_review": needs_manual_review,
            "evidence_signals": self._clean_list(payload.get("evidence_signals"), max_items=8, item_max_chars=300),
            "conflict_flags": self._clean_list(payload.get("conflict_flags"), max_items=8, item_max_chars=220),
        }
        if needs_manual_review:
            result["close_verdict"] = "manual_review"
            result["premature_close_risk"] = "manual_review"
        return result

    def _analyze_openai(self, *, prompt: str) -> dict[str, Any]:
        model = self._safe_text(self._settings.crm_analysis_model) or "gpt-5.4"
        cached = self._cache_lookup(provider="openai", model=model, reasoning="temperature=0.1", prompt=prompt)
        if cached is not None:
            return cached
        client = self._openai_client()
        response = client.chat.completions.create(
            model=model,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content if response.choices else None
        if not content:
            raise DealLLMError("OpenAI CRM analysis returned empty content")
        payload = self._extract_json_payload(content)
        normalized = self._normalize_response(payload)
        self._cache_store(provider="openai", model=model, reasoning="temperature=0.1", prompt=prompt, response=normalized)
        return normalized

    def _analyze_codex_cli(self, *, prompt: str) -> dict[str, Any]:
        codex_bin = (self._settings.codex_cli_path or "codex").strip() or "codex"
        if shutil.which(codex_bin) is None:
            raise DealLLMError(f"codex binary is not available: {codex_bin}")
        model = self._safe_text(self._settings.crm_analysis_model) or "gpt-5.4"
        reasoning = self._safe_text(self._settings.crm_analysis_reasoning_effort).lower()
        runtime_codex_home = self._prepare_runtime_codex_home()
        cached = self._cache_lookup(provider="codex_cli", model=model, reasoning=reasoning, prompt=prompt)
        if cached is not None:
            return cached

        max_attempts = 4
        timeout_sec = max(15, int(self._settings.crm_analysis_timeout_seconds))
        last_error: Optional[str] = None
        retryable_marker = "no last agent message"

        for attempt in range(1, max_attempts + 1):
            with tempfile.NamedTemporaryFile(prefix="mango_deal_llm_", suffix=".txt") as out_file:
                cmd = [
                    codex_bin,
                    "exec",
                    "--skip-git-repo-check",
                    "--ephemeral",
                    "--sandbox",
                    "read-only",
                    "--model",
                    model,
                    "--output-last-message",
                    out_file.name,
                ]
                if reasoning in {"low", "medium", "high"}:
                    cmd.extend(["-c", f'model_reasoning_effort="{reasoning}"'])
                cmd.append("-")
                proc = subprocess.run(
                    cmd,
                    input=prompt,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=timeout_sec,
                    env={
                        **os.environ,
                        "CODEX_HOME": runtime_codex_home,
                    },
                )
                raw = Path(out_file.name).read_text(encoding="utf-8", errors="ignore")

            for candidate in (raw, proc.stdout or "", proc.stderr or ""):
                candidate = (candidate or "").strip()
                if not candidate:
                    continue
                try:
                    payload = self._extract_json_payload(candidate)
                except DealLLMError:
                    continue
                normalized = self._normalize_response(payload)
                self._cache_store(provider="codex_cli", model=model, reasoning=reasoning, prompt=prompt, response=normalized)
                return normalized

            stderr = (proc.stderr or "").strip()
            if proc.returncode == 0:
                last_error = "Codex deal analysis returned empty content"
                if attempt < max_attempts:
                    time.sleep(min(4, attempt + 1))
                    continue
                raise DealLLMError(last_error)

            stderr_tail = stderr.splitlines()[-1:] or [""]
            last_error = f"codex exec failed rc={proc.returncode}: {stderr_tail[0].strip()}"
            if retryable_marker in stderr.lower() and attempt < max_attempts:
                time.sleep(min(6, attempt * 2))
                continue
            raise DealLLMError(last_error)

        raise DealLLMError(last_error or "Codex deal analysis failed")

    def analyze(self, *, dossier: dict[str, Any], heuristic_analysis: dict[str, Any]) -> dict[str, Any]:
        prompt = self._build_prompt(dossier, heuristic_analysis)
        provider = self._safe_text(self._settings.crm_analysis_provider).lower() or "codex_cli"
        if provider == "openai":
            result = self._analyze_openai(prompt=prompt)
        elif provider == "codex_cli":
            result = self._analyze_codex_cli(prompt=prompt)
        else:
            raise DealLLMError(f"Unsupported CRM_ANALYSIS_PROVIDER={provider}")
        return {
            **result,
            "llm_provider": provider,
            "llm_model": self._safe_text(self._settings.crm_analysis_model),
            "llm_prompt_version": PROMPT_VERSION,
        }
