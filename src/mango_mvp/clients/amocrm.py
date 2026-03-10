from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from mango_mvp.config import Settings
from mango_mvp.utils.phone import last10


class AmoCRMError(RuntimeError):
    pass


class AmoCRMClient:
    def __init__(self, settings: Settings):
        if not settings.amocrm_base_url:
            raise AmoCRMError("AMOCRM_BASE_URL is required")
        self._settings = settings
        self._base_url = settings.amocrm_base_url.rstrip("/")
        self._session = requests.Session()
        self._access_token = settings.amocrm_access_token
        self._refresh_token = settings.amocrm_refresh_token
        self._token_cache_path = Path(settings.amocrm_token_cache_path)
        self._load_token_cache()

    def _load_token_cache(self) -> None:
        if not self._token_cache_path.exists():
            return
        try:
            payload = json.loads(self._token_cache_path.read_text(encoding="utf-8"))
            self._access_token = payload.get("access_token") or self._access_token
            self._refresh_token = payload.get("refresh_token") or self._refresh_token
        except json.JSONDecodeError:
            return

    def _save_token_cache(self) -> None:
        payload = {
            "access_token": self._access_token,
            "refresh_token": self._refresh_token,
            "saved_at_unix": int(time.time()),
        }
        self._token_cache_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")

    def _refresh_access_token(self) -> None:
        required = [
            self._settings.amocrm_client_id,
            self._settings.amocrm_client_secret,
            self._settings.amocrm_redirect_uri,
            self._refresh_token,
        ]
        if any(not value for value in required):
            raise AmoCRMError("Cannot refresh token: missing OAuth refresh credentials")

        payload = {
            "client_id": self._settings.amocrm_client_id,
            "client_secret": self._settings.amocrm_client_secret,
            "grant_type": "refresh_token",
            "refresh_token": self._refresh_token,
            "redirect_uri": self._settings.amocrm_redirect_uri,
        }
        url = f"{self._base_url}/oauth2/access_token"
        response = self._session.post(url, json=payload, timeout=30)
        if response.status_code >= 300:
            raise AmoCRMError(
                f"Failed to refresh amoCRM token: HTTP {response.status_code} {response.text}"
            )
        data = response.json()
        self._access_token = data.get("access_token")
        self._refresh_token = data.get("refresh_token") or self._refresh_token
        if not self._access_token:
            raise AmoCRMError("amoCRM refresh response does not contain access_token")
        self._save_token_cache()

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Any] = None,
        retry_auth: bool = True,
    ) -> Any:
        if not self._access_token:
            self._refresh_access_token()
        headers = {"Authorization": f"Bearer {self._access_token}"}
        url = f"{self._base_url}{path}"
        response = self._session.request(
            method=method,
            url=url,
            params=params,
            json=json_body,
            headers=headers,
            timeout=30,
        )
        if response.status_code == 401 and retry_auth:
            self._refresh_access_token()
            return self._request(
                method,
                path,
                params=params,
                json_body=json_body,
                retry_auth=False,
            )
        if response.status_code >= 300:
            raise AmoCRMError(f"amoCRM error: HTTP {response.status_code} {response.text}")
        if not response.text:
            return {}
        return response.json()

    def find_contact_by_phone(self, phone: str) -> Optional[Dict[str, Any]]:
        query = last10(phone)
        if not query:
            return None
        payload = self._request("GET", "/api/v4/contacts", params={"query": query})
        contacts = (payload.get("_embedded") or {}).get("contacts") or []
        if not contacts:
            return None
        return contacts[0]

    def add_contact_note(self, contact_id: int, text: str) -> None:
        body = [{"entity_id": contact_id, "note_type": "common", "params": {"text": text}}]
        self._request("POST", "/api/v4/contacts/notes", json_body=body)

    def update_contact_fields(self, contact_id: int, custom_fields_values: List[Dict[str, Any]]) -> None:
        if not custom_fields_values:
            return
        body = [{"id": contact_id, "custom_fields_values": custom_fields_values}]
        self._request("PATCH", "/api/v4/contacts", json_body=body)

    def create_task(
        self,
        *,
        contact_id: int,
        text: str,
        complete_till_unix: int,
        task_type_id: Optional[int],
        responsible_user_id: Optional[int],
    ) -> None:
        task: Dict[str, Any] = {
            "text": text,
            "entity_id": contact_id,
            "entity_type": "contacts",
            "complete_till": complete_till_unix,
        }
        if task_type_id:
            task["task_type_id"] = task_type_id
        if responsible_user_id:
            task["responsible_user_id"] = responsible_user_id
        self._request("POST", "/api/v4/tasks", json_body=[task])
