from __future__ import annotations

import csv
import io
import json
import socket
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mango_mvp.existing_clients import amo_step1_snapshot as step1


NOW = datetime(2026, 6, 12, 9, 0, tzinfo=timezone.utc)


class FakeMcpClient:
    def __init__(self) -> None:
        self.calls = 0

    def amo_api_get(self, *, path: str, params: dict | None = None, limit: int = 50) -> dict:
        self.calls += 1
        page = int((params or {}).get("page") or 1)
        if path == "leads/pipelines":
            return {
                "_embedded": {
                    "pipelines": [
                        {
                            "id": 1,
                            "name": "B2C",
                            "_embedded": {
                                "statuses": [
                                    {"id": 1, "name": "Новая"},
                                    {"id": 142, "name": "Успешно"},
                                    {"id": 143, "name": "Закрыто и не реализовано"},
                                ]
                            },
                        }
                    ]
                }
            }
        if path == "contacts":
            pages = {
                1: [
                    contact(101, "Роман Иванов", "+7 999 000 00 00", parent="Елена", tallanto="t-1", leads=[501]),
                    contact(102, "Рома Иванов", "8 (999) 000-00-00", parent="Елена", tallanto="t-2", leads=[502]),
                    contact(103, "Софья Иванова", "9990000000", parent="Елена", tallanto="t-3", leads=[]),
                    contact(104, "Петр", "+7 999 111 22 33", parent="Анна", tallanto="t-4", leads=[503]),
                ],
                2: [
                    contact(105, "Олег", "+7 999 111 22 33", parent="Борис", tallanto="t-5", leads=[]),
                    contact(106, "Мария", "+7 999 222 33 44", parent="", tallanto="", leads=[]),
                    contact(107, "Иван", "+7 999 222 33 44", parent="", tallanto="", leads=[]),
                ],
            }
            payload = {"_embedded": {"contacts": pages.get(page, [])}}
            if page == 1:
                payload["_links"] = {"next": {"href": "/api/v4/contacts?page=2"}}
            return payload
        if path == "leads":
            if page > 1:
                return {"_embedded": {"leads": []}}
            return {
                "_embedded": {
                    "leads": [
                        lead(501, 1, contacts=[101]),
                        lead(502, 1, contacts=[102]),
                        lead(503, 143, contacts=[104]),
                    ]
                }
            }
        raise AssertionError(path)


def contact(contact_id: int, name: str, phone: str, *, parent: str, tallanto: str, leads: list[int]) -> dict:
    fields = [
        {"field_name": "Телефон", "field_code": "PHONE", "values": [{"value": phone}]},
    ]
    if parent:
        fields.append({"field_name": "ФИО Родителя", "values": [{"value": parent}]})
    if tallanto:
        fields.append({"field_name": "Id Tallanto", "values": [{"value": tallanto}]})
    return {
        "id": contact_id,
        "name": name,
        "created_at": 1781200000 + contact_id,
        "updated_at": 1781205000 + contact_id,
        "custom_fields_values": fields,
        "_embedded": {"leads": [{"id": lead_id} for lead_id in leads]},
    }


def lead(lead_id: int, status_id: int, *, contacts: list[int]) -> dict:
    return {
        "id": lead_id,
        "name": f"Lead {lead_id}",
        "status_id": status_id,
        "_embedded": {"contacts": [{"id": contact_id} for contact_id in contacts]},
    }


def test_step1_snapshot_classifies_duplicates_common_and_multi_child(tmp_path: Path) -> None:
    project = tmp_path / "project"
    out = project / "product_data" / "customer_profiles" / "tz14"

    summary = step1.build_amo_step1_snapshot(
        project_root=project,
        out_root=out,
        client=FakeMcpClient(),
        page_limit=50,
        sleep_sec=0,
        generated_at=NOW,
    )

    assert summary["read_only"] is True
    assert summary["write_crm"] is False
    assert summary["contacts_seen"] == 7
    assert summary["leads_seen"] == 3
    assert summary["counts"]["duplicate_groups_total"] == 1
    assert summary["counts"]["live_duplicate_groups"] == 1
    assert summary["counts"]["possible_common_phone_groups"] == 1
    assert summary["counts"]["multi_child_family_groups"] == 1
    assert summary["counts"]["ambiguous_missing_parent_groups"] == 1
    assert (out / "amo_step1_snapshot.sqlite").exists()

    duplicates = rows(out / "duplicate_candidates.csv")
    assert duplicates[0]["phone"] == "+79990000000"
    assert duplicates[0]["contact_ids"] == "101 | 102"
    assert duplicates[0]["live_status"] == "live_candidate"

    common = rows(out / "common_phone_review.csv")
    assert common[0]["phone"] == "+79991112233"
    assert common[0]["review_class"] == "possible_common_phone_distinct_parents"

    multi_child = rows(out / "multi_child_families.csv")
    assert multi_child[0]["phone"] == "+79990000000"
    assert multi_child[0]["review_class"] == "multi_child_same_parent"


def test_step1_snapshot_csv_is_stable_on_rerun(tmp_path: Path) -> None:
    project = tmp_path / "project"
    out = project / "product_data" / "customer_profiles" / "tz14"
    kwargs = {
        "project_root": project,
        "out_root": out,
        "page_limit": 50,
        "sleep_sec": 0,
        "generated_at": NOW,
    }

    step1.build_amo_step1_snapshot(client=FakeMcpClient(), **kwargs)
    first = (out / "duplicate_candidates.csv").read_bytes()
    step1.build_amo_step1_snapshot(client=FakeMcpClient(), **kwargs)
    second = (out / "duplicate_candidates.csv").read_bytes()

    assert first == second


def test_phone_and_name_normalization_are_conservative() -> None:
    assert step1.normalize_phone("8 (999) 000-00-00") == "+79990000000"
    assert step1.normalize_phone("+7 999 000 00 00") == "+79990000000"
    assert step1.normalize_phone("9990000000") == "+79990000000"
    assert step1.normalize_person_key("Рома Иванов") == step1.normalize_person_key("Роман Иванов")
    assert step1.normalize_person_key("Саша") != step1.normalize_person_key("Александр")


def test_step1_refuses_project_output_outside_customer_profiles(tmp_path: Path) -> None:
    project = tmp_path / "project"
    with pytest.raises(ValueError, match="product_data/customer_profiles"):
        step1.build_amo_step1_snapshot(
            project_root=project,
            out_root=project / "runs" / "bad",
            client=FakeMcpClient(),
            sleep_sec=0,
        )


def test_mcp_client_retries_429(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}
    user_agents: list[str] = []

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def read(self) -> bytes:
            result = {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "content": [{"type": "text", "text": json.dumps({"_embedded": {"users": []}})}],
                    "isError": False,
                },
            }
            return json.dumps(result).encode("utf-8")

    def fake_urlopen(request, timeout):
        calls["count"] += 1
        user_agents.append(request.headers.get("User-agent") or request.headers.get("User-Agent") or "")
        if calls["count"] == 1:
            raise urllib.error.HTTPError(
                request.full_url,
                429,
                "too many",
                hdrs={"Retry-After": "0.1"},
                fp=io.BytesIO(b'{"detail":"rate"}'),
            )
        return FakeResponse()

    monkeypatch.setattr(step1.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(step1.time, "sleep", lambda seconds: None)
    client = step1.AmoMcpClient(
        step1.AmoMcpConfig(connector_url="https://connector.test", bearer_token="token", max_retries=1)
    )

    assert client.amo_api_get(path="users", limit=1)["_embedded"]["users"] == []
    assert calls["count"] == 2
    assert user_agents == [step1.DEFAULT_USER_AGENT, step1.DEFAULT_USER_AGENT]


def test_mcp_client_retries_socket_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def read(self) -> bytes:
            result = {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "content": [{"type": "text", "text": json.dumps({"_embedded": {"users": []}})}],
                    "isError": False,
                },
            }
            return json.dumps(result).encode("utf-8")

    def fake_urlopen(request, timeout):
        calls["count"] += 1
        if calls["count"] == 1:
            raise socket.timeout("slow connector")
        return FakeResponse()

    monkeypatch.setattr(step1.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(step1.time, "sleep", lambda seconds: None)
    client = step1.AmoMcpClient(
        step1.AmoMcpConfig(connector_url="https://connector.test", bearer_token="token", max_retries=1)
    )

    assert client.amo_api_get(path="users", limit=1)["_embedded"]["users"] == []
    assert calls["count"] == 2


def rows(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open(encoding="utf-8-sig")))
