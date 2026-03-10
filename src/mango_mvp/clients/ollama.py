from __future__ import annotations

import json
from typing import Any, Dict, Optional

import requests


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, base_url: str):
        self._base_url = (base_url or "http://127.0.0.1:11434").rstrip("/")
        self._session = requests.Session()

    def _post(self, path: str, payload: Dict[str, Any], timeout_sec: int) -> Dict[str, Any]:
        url = f"{self._base_url}{path}"
        try:
            response = self._session.post(url, json=payload, timeout=timeout_sec)
        except requests.RequestException as exc:
            raise OllamaError(f"Ollama request failed: {exc}") from exc
        if response.status_code >= 300:
            raise OllamaError(
                f"Ollama error HTTP {response.status_code}: {response.text[:500]}"
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise OllamaError("Ollama returned non-JSON response") from exc
        if not isinstance(data, dict):
            raise OllamaError("Ollama returned unexpected payload type")
        return data

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        stripped = text.strip()
        if not stripped.startswith("```"):
            return stripped
        lines = stripped.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
        return stripped

    def _extract_object_json(self, text: str) -> Dict[str, Any]:
        stripped = self._strip_code_fences(text)
        candidates = [stripped]
        left = stripped.find("{")
        right = stripped.rfind("}")
        if left != -1 and right > left:
            candidates.append(stripped[left : right + 1])
        for candidate in candidates:
            if not candidate:
                continue
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
        raise OllamaError("Ollama response is not valid object JSON")

    def generate_json(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        think: Optional[str],
        temperature: float,
        num_predict: Optional[int] = None,
        timeout_sec: int = 600,
    ) -> Dict[str, Any]:
        options: Dict[str, Any] = {"temperature": float(temperature)}
        if num_predict is not None:
            options["num_predict"] = int(num_predict)
        payload: Dict[str, Any] = {
            "model": model,
            "system": system_prompt,
            "prompt": user_prompt,
            "stream": False,
            "options": options,
        }
        think_value = (think or "").strip().lower()
        if think_value:
            payload["think"] = think_value
        data = self._post("/api/generate", payload, timeout_sec=timeout_sec)
        response_text = str(data.get("response") or "").strip()
        if not response_text:
            done_reason = str(data.get("done_reason") or "").strip()
            if done_reason == "length":
                raise OllamaError(
                    "Ollama response hit token limit before final JSON (done_reason=length)"
                )
            raise OllamaError("Ollama returned empty response")
        return self._extract_object_json(response_text)
