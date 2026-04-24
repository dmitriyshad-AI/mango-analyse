from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional


class LLMResponseCache:
    def __init__(self, *, enabled: bool, root_dir: str | Path):
        self._enabled = bool(enabled)
        self._root_dir = Path(root_dir)

    def _cache_path(
        self,
        *,
        namespace: str,
        provider: str,
        model: str,
        reasoning: str,
        prompt_version: str,
        prompt: str,
    ) -> Path:
        key_payload = {
            "namespace": namespace,
            "provider": provider,
            "model": model,
            "reasoning": reasoning,
            "prompt_version": prompt_version,
            "prompt": prompt,
        }
        raw = json.dumps(key_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()
        return self._root_dir / namespace / f"{digest}.json"

    def get(
        self,
        *,
        namespace: str,
        provider: str,
        model: str,
        reasoning: str,
        prompt_version: str,
        prompt: str,
    ) -> Optional[Dict[str, Any]]:
        if not self._enabled:
            return None
        path = self._cache_path(
            namespace=namespace,
            provider=provider,
            model=model,
            reasoning=reasoning,
            prompt_version=prompt_version,
            prompt=prompt,
        )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return None
        response = payload.get("response")
        if isinstance(response, dict):
            return response
        return None

    def put(
        self,
        *,
        namespace: str,
        provider: str,
        model: str,
        reasoning: str,
        prompt_version: str,
        prompt: str,
        response: Dict[str, Any],
    ) -> None:
        if not self._enabled or not isinstance(response, dict):
            return
        path = self._cache_path(
            namespace=namespace,
            provider=provider,
            model=model,
            reasoning=reasoning,
            prompt_version=prompt_version,
            prompt=prompt,
        )
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(".tmp")
            tmp_path.write_text(
                json.dumps(
                    {
                        "provider": provider,
                        "model": model,
                        "reasoning": reasoning,
                        "prompt_version": prompt_version,
                        "response": response,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            tmp_path.replace(path)
        except OSError:
            return
