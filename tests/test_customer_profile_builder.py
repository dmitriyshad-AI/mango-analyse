from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mango_mvp.customer_profile import CustomerProfileBuilder, CustomerProfileBuildOptions
from mango_mvp.customer_profile.build_cli import safe_field_preview
from mango_mvp.customer_profile.builder import apply_child_slot_merge_candidates, child_slot_groups
from mango_mvp.customer_profile.child_identity_dedup_llm import (
    ChildResolverConfig,
    ChildResolverError,
    apply_llm_child_resolver_to_fields,
    build_child_resolver_prompt,
    build_child_resolver_cases,
    child_resolver_output_json_schema,
    normalize_child_resolver_response,
)
from mango_mvp.customer_profile.contracts import ProfileFieldCandidate, apply_superseded_rules
from mango_mvp.customer_profile.store import CustomerProfileSQLiteStore
from mango_mvp.services.llm_response_cache import LLMResponseCache
from mango_mvp.customer_timeline.contracts import (
    CustomerIdentity,
    IdentityLink,
    IdentityMatchClass,
    IdentityStatus,
    TimelineDirection,
    TimelineEvent,
    TimelineEventType,
)
from mango_mvp.customer_timeline.store import CustomerTimelineSQLiteStore


NOW = datetime(2026, 6, 10, 12, 0, tzinfo=timezone.utc)


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, payloads: list[dict[str, object] | str]):
        self.payloads = list(payloads)
        self.calls = 0

    def create(self, **kwargs: object) -> _FakeResponse:
        self.calls += 1
        if not self.payloads:
            raise AssertionError("unexpected child resolver LLM call")
        payload = self.payloads.pop(0)
        if callable(payload):
            payload = payload(str(kwargs["messages"][0]["content"]))
        if isinstance(payload, str):
            return _FakeResponse(payload)
        return _FakeResponse(json.dumps(payload, ensure_ascii=False))


class _FakeChat:
    def __init__(self, completions: _FakeCompletions):
        self.completions = completions


class _FakeClient:
    def __init__(self, payloads: list[dict[str, object] | str]):
        self.completions = _FakeCompletions(payloads)
        self.chat = _FakeChat(self.completions)


def _prompt_mention_ids(prompt: str) -> list[str]:
    payload = json.loads(prompt.split("INPUT_JSON:\n", 1)[1])
    return [str(item["mention_id"]) for item in payload["mentions"]]


def _children_payload(prompt: str, children: list[dict[str, object]]) -> dict[str, object]:
    mention_ids = _prompt_mention_ids(prompt)
    translated: list[dict[str, object]] = []
    for child in children:
        child_payload = {key: value for key, value in child.items() if key != "brands"}
        ids = child.get("mention_ids")
        if ids == "all":
            resolved_ids = mention_ids
        elif isinstance(ids, list) and ids and isinstance(ids[0], int):
            resolved_ids = [mention_ids[int(item)] for item in ids]
        else:
            resolved_ids = ids
        translated.append({**child_payload, "mention_ids": resolved_ids, "merge_confidence": child.get("merge_confidence", "high")})
    return {"children": translated}


def _resolver_config(
    tmp_path: Path,
    *,
    stoplist_phone: str = "+70000000000",
    trace_path: Path | None = None,
    name_diagnostics_path: Path | None = None,
) -> ChildResolverConfig:
    stoplist = tmp_path / "shared_phones_stoplist.json"
    stoplist.write_text(json.dumps({"phones": [stoplist_phone]}), encoding="utf-8")
    return ChildResolverConfig(
        provider="openai",
        model="fake-child-resolver",
        cache_root_dir=tmp_path / "cache",
        max_concurrency=2,
        max_retries=1,
        stoplist_path=stoplist,
        trace_path=trace_path,
        name_diagnostics_path=name_diagnostics_path,
    )


def _child_field(
    field: str,
    value: str,
    *,
    profile_id: str = "cust-1",
    child_key: str = "child_1",
    source_ref: str = "mango:1",
    event_at: datetime = NOW,
    brand: str = "foton",
) -> ProfileFieldCandidate:
    return ProfileFieldCandidate(
        profile_id=profile_id,
        field=field,
        value=value,
        child_key=child_key,
        source_system="mango_processed_summary",
        source_ref=source_ref,
        event_at=event_at,
        brand=brand,
    )


def _timeline_db(tmp_path: Path, *, duplicate_phone_customer: bool = False) -> Path:
    db = tmp_path / "customer_timeline.sqlite"
    store = CustomerTimelineSQLiteStore(db, allowed_root=tmp_path)
    customer = CustomerIdentity(
        tenant_id="foton",
        customer_id="cust-1",
        identity_status=IdentityStatus.STRONG,
        display_name="Клиент",
        primary_phone="+79990000000",
        source_ref="test",
        first_seen_at=NOW,
        last_seen_at=NOW,
    )
    store.upsert_customer(customer)
    store.upsert_identity_link(
        IdentityLink(
            tenant_id="foton",
            customer_id="cust-1",
            link_type="phone",
            link_value="+79990000000",
            source_system="test",
            source_ref="test:phone",
            match_class=IdentityMatchClass.STRONG_UNIQUE,
            confidence=1.0,
            first_seen_at=NOW,
            last_seen_at=NOW,
        )
    )
    store.upsert_event(
        TimelineEvent(
            tenant_id="foton",
            customer_id="cust-1",
            event_type=TimelineEventType.MANGO_CALL,
            event_at=NOW,
            source_system="mango_processed_summary",
            source_id="100",
            source_ref="mango:100",
            direction=TimelineDirection.INBOUND,
            record={"brand": "foton"},
            created_at=NOW,
        )
    )
    if duplicate_phone_customer:
        second = CustomerIdentity(
            tenant_id="foton",
            customer_id="cust-2",
            identity_status=IdentityStatus.STRONG,
            display_name="Другой клиент",
            primary_phone="+79990000000",
            source_ref="test2",
            first_seen_at=NOW,
            last_seen_at=NOW,
        )
        store.upsert_customer(second)
        store.upsert_identity_link(
            IdentityLink(
                tenant_id="foton",
                customer_id="cust-2",
                link_type="phone",
                link_value="+79990000000",
                source_system="test",
                source_ref="test2:phone",
                match_class=IdentityMatchClass.STRONG_UNIQUE,
                confidence=1.0,
                first_seen_at=NOW,
                last_seen_at=NOW,
            )
        )
    store.close()
    return db


def _master_calls_db(tmp_path: Path, rows: list[tuple[int, str, dict]]) -> Path:
    db = tmp_path / "canonical_calls_master.db"
    con = sqlite3.connect(db)
    try:
        con.execute(
            """
            CREATE TABLE canonical_calls (
              canonical_call_id INTEGER PRIMARY KEY,
              phone TEXT,
              started_at TEXT,
              analysis_status TEXT,
              analysis_json TEXT
            )
            """
        )
        for call_id, started_at, analysis in rows:
            phone = analysis.pop("_phone", "+79990000000")
            con.execute(
                "INSERT INTO canonical_calls VALUES (?, ?, ?, ?, ?)",
                (call_id, phone, started_at, "done", json.dumps(analysis, ensure_ascii=False)),
            )
        con.commit()
    finally:
        con.close()
    return db


def _analysis(*, grade: str = "8", child_name: str = "Ребенок", subjects: list[str] | None = None) -> dict:
    return {
        "structured_fields": {
            "people": {"parent_fio": "Родитель", "child_fio": child_name},
            "student": {"grade_current": grade},
            "interests": {"subjects": subjects or ["математика"], "format": ["онлайн"]},
            "next_step": {"action": "Отправить материалы"},
        },
        "target_product": "годовые курсы",
        "objections": ["цена"],
    }


def _children_analysis(children: list[dict], *, parent_name: str = "Родитель") -> dict:
    return {
        "structured_fields": {
            "children": children,
            "people": {"parent_fio": parent_name},
            "interests": {},
            "next_step": {},
        },
        "target_product": "годовые курсы",
    }


def test_builder_marks_superseded_conflicting_grade_and_is_idempotent(tmp_path: Path) -> None:
    timeline_db = _timeline_db(tmp_path)
    master_db = _master_calls_db(
        tmp_path,
        [
            (100, "2026-01-10T10:00:00+00:00", _analysis(grade="7")),
            (101, "2026-02-10T10:00:00+00:00", _analysis(grade="8")),
        ],
    )
    profiles_db = tmp_path / "profiles.sqlite"
    options = CustomerProfileBuildOptions(
        timeline_db=timeline_db,
        profiles_db=profiles_db,
        master_calls_db=master_db,
        customer_ids=("cust-1",),
        build_id="test-build",
    )

    first = CustomerProfileBuilder(options).build()
    with CustomerProfileSQLiteStore(profiles_db) as store:
        active_first = store.active_fields("cust-1")
        summary_first = store.summary()
    second = CustomerProfileBuilder(options).build()
    with CustomerProfileSQLiteStore(profiles_db) as store:
        active_second = store.active_fields("cust-1")
        all_grade_rows = sqlite3.connect(profiles_db).execute(
            "SELECT value, superseded_by FROM profile_fields WHERE field='grade' ORDER BY event_at"
        ).fetchall()

    active_grade = [row for row in active_first if row["field"] == "grade"]
    assert first["build_id"] == "test-build"
    assert second["fields_written"] == first["fields_written"]
    assert active_first == active_second
    assert summary_first["counts"]["customer_profiles"] == 1
    assert active_grade[0]["value"] == "8"
    assert all_grade_rows[0][0] == "7"
    assert all_grade_rows[0][1]
    assert all_grade_rows[1][0] == "8"
    assert all_grade_rows[1][1] == ""


def test_builder_opens_read_only_sources_under_path_with_space(tmp_path: Path) -> None:
    root = tmp_path / "profile source with space"
    root.mkdir()
    timeline_db = _timeline_db(root)
    master_db = _master_calls_db(root, [(100, "2026-01-10T10:00:00+00:00", _analysis())])
    profiles_db = root / "profiles.sqlite"

    report = CustomerProfileBuilder(
        CustomerProfileBuildOptions(
            timeline_db=timeline_db,
            profiles_db=profiles_db,
            master_calls_db=master_db,
            customer_ids=("cust-1",),
        )
    ).build()

    assert report["profiles_built"] == 1
    with CustomerProfileSQLiteStore(profiles_db) as store:
        assert any(row["field"] == "grade" for row in store.active_fields("cust-1"))


def test_builder_marks_duplicate_child_slots_as_merge_candidate(tmp_path: Path) -> None:
    timeline_db = _timeline_db(tmp_path)
    master_db = _master_calls_db(
        tmp_path,
        [
            (100, "2026-01-10T10:00:00+00:00", _analysis(grade="7", child_name="Рома")),
            (101, "2026-02-10T10:00:00+00:00", _analysis(grade="8", child_name="Роман")),
        ],
    )
    profiles_db = tmp_path / "profiles.sqlite"
    options = CustomerProfileBuildOptions(
        timeline_db=timeline_db,
        profiles_db=profiles_db,
        master_calls_db=master_db,
        customer_ids=("cust-1",),
        build_id="test-build",
    )

    first = CustomerProfileBuilder(options).build()
    with CustomerProfileSQLiteStore(profiles_db) as store:
        active_first = store.active_fields("cust-1")
    second = CustomerProfileBuilder(options).build()
    with CustomerProfileSQLiteStore(profiles_db) as store:
        active_second = store.active_fields("cust-1")

    child_keys = {
        row["child_key"]
        for row in active_first
        if row["field"] in {"child_name", "grade", "subject"}
    }
    marker_rows = [row for row in active_first if row["field"] == "child_slot_merge_candidate"]
    active_grade = [row for row in active_first if row["field"] == "grade"]

    assert len(child_keys) == 1
    assert len(marker_rows) == 1
    assert "merge_candidate" in marker_rows[0]["value"]
    assert active_grade[0]["value"] == "8"
    assert first["child_slot_merge"]["profiles_with_2plus_children_before"] == 1
    assert first["child_slot_merge"]["profiles_with_2plus_children_after"] == 0
    assert first["child_slot_merge"]["merge_candidate_groups"] == 1
    assert second["fields_written"] == first["fields_written"]
    assert active_second == active_first


def test_child_slot_trait_merge_is_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PROFILE_CHILD_MERGE_BY_TRAIT", raising=False)

    groups = child_slot_groups(
        {
            "child_1": {"names": set(), "grades": {"7"}, "subjects": {"математика"}},
            "child_2": {"names": set(), "grades": {"7"}, "subjects": {"математика"}},
        }
    )

    assert groups == [("child_1", ["child_1"]), ("child_2", ["child_2"])]


def test_child_slot_trait_merge_combines_nameless_same_grade_and_subject(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROFILE_CHILD_MERGE_BY_TRAIT", "1")

    groups = child_slot_groups(
        {
            "child_1": {"names": set(), "grades": {"7"}, "subjects": {"математика"}},
            "child_2": {"names": set(), "grades": {"7"}, "subjects": {"математика"}},
            "child_3": {"names": set(), "grades": {"7"}, "subjects": {"физика"}},
        }
    )

    assert groups == [("child_1", ["child_1", "child_2"]), ("child_3", ["child_3"])]


def test_child_slot_trait_merge_does_not_absorb_named_slots(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROFILE_CHILD_MERGE_BY_TRAIT", "1")

    groups = child_slot_groups(
        {
            "child_1": {"names": {"Анна"}, "grades": {"7"}, "subjects": {"математика"}},
            "child_2": {"names": set(), "grades": {"7"}, "subjects": {"математика"}},
            "child_3": {"names": {"Олег"}, "grades": {"7"}, "subjects": {"математика"}},
        }
    )

    assert groups == [("child_1", ["child_1"]), ("child_2", ["child_2"]), ("child_3", ["child_3"])]


def test_llm_child_resolver_flag_off_matches_existing_merge(monkeypatch: pytest.MonkeyPatch) -> None:
    fields = [
        _child_field("child_name", "Рома", child_key="child_a", source_ref="mango:1"),
        _child_field("grade", "7", child_key="child_a", source_ref="mango:1"),
        _child_field("child_name", "Роман", child_key="child_b", source_ref="mango:2"),
        _child_field("grade", "8", child_key="child_b", source_ref="mango:2"),
    ]
    monkeypatch.delenv("PROFILE_LLM_CHILD_RESOLVER", raising=False)
    old_fields, old_summary = apply_child_slot_merge_candidates(fields)
    monkeypatch.setenv("PROFILE_LLM_CHILD_RESOLVER", "0")
    new_fields, new_summary = apply_child_slot_merge_candidates(fields)

    assert old_summary == new_summary
    assert old_fields == new_fields


def test_llm_child_resolver_schema_and_prompt_include_merge_confidence() -> None:
    schema = child_resolver_output_json_schema()
    child_schema = schema["properties"]["children"]["items"]

    assert schema["additionalProperties"] is False
    assert child_schema["additionalProperties"] is False
    assert "merge_confidence" in child_schema["required"]
    assert child_schema["properties"]["merge_confidence"] == {"type": "string"}
    assert "brands" not in child_schema["required"]
    assert "brands" not in child_schema["properties"]

    fields = [
        _child_field("child_name", "Степан", child_key="child_a", source_ref="mango:1"),
        _child_field("child_name", "Стёпа", child_key="child_b", source_ref="mango:2"),
    ]
    case = build_child_resolver_cases(fields, profile_phones={"cust-1": "+79990000000"})[0]
    prompt = build_child_resolver_prompt(case)

    assert '"merge_confidence": "high"' in prompt
    assert '"brand"' not in prompt
    assert '"brands"' not in prompt
    assert "ФИО" in prompt
    assert "high" in prompt
    assert "low" in prompt


def test_llm_child_resolver_normalizes_confidence_and_dedups_name_variants() -> None:
    payload = {
        "children": [
            {
                "child_id": "child_1",
                "canonical_name": "Степан",
                "name_variants": ["Стёпа", "Степа", "стёпа"],
                "grades": [],
                "subjects": [],
                "mention_ids": ["m_1"],
                "merge_confidence": "LOW",
            }
        ]
    }

    normalized = normalize_child_resolver_response(payload)

    assert normalized["children"][0]["merge_confidence"] == "low"
    assert normalized["children"][0]["name_variants"] == ["Стёпа", "Степа"]


def test_llm_child_resolver_normalize_rejects_extra_and_missing_fields() -> None:
    child = {
        "child_id": "child_1",
        "canonical_name": "",
        "name_variants": [],
        "grades": [],
        "subjects": [],
        "mention_ids": ["m_1"],
        "merge_confidence": "high",
    }

    with pytest.raises(ChildResolverError, match="unexpected fields"):
        normalize_child_resolver_response({"children": [child], "debug": True})

    with pytest.raises(ChildResolverError, match="unexpected fields"):
        normalize_child_resolver_response({"children": [{**child, "reason": "guess"}]})

    missing_confidence = dict(child)
    missing_confidence.pop("merge_confidence")
    normalized_missing = normalize_child_resolver_response({"children": [missing_confidence]})
    assert normalized_missing["children"][0]["merge_confidence"] == "low"

    invalid_confidence = {**child, "merge_confidence": "medium"}
    normalized_invalid = normalize_child_resolver_response({"children": [invalid_confidence]})
    assert normalized_invalid["children"][0]["merge_confidence"] == "low"


def test_llm_child_resolver_merges_typo_name_variants(tmp_path: Path) -> None:
    fields = [
        _child_field("child_name", "Степан", child_key="child_a", source_ref="mango:1"),
        _child_field("grade", "7", child_key="child_a", source_ref="mango:1"),
        _child_field("child_name", "Стёпа", child_key="child_b", source_ref="mango:2"),
        _child_field("subject", "физика", child_key="child_b", source_ref="mango:2"),
    ]
    client = _FakeClient(
        [
            lambda prompt: _children_payload(
                prompt,
                [
                    {
                        "child_id": "child_1",
                        "canonical_name": "Степан",
                        "name_variants": ["Степан", "Стёпа"],
                        "grades": ["7"],
                        "subjects": ["физика"],
                        "mention_ids": "all",
                    }
                ],
            )
        ]
    )

    result = apply_llm_child_resolver_to_fields(
        fields,
        profile_phones={"cust-1": "+79990000000"},
        config=_resolver_config(tmp_path),
        client=client,
        cache=LLMResponseCache(enabled=True, root_dir=tmp_path / "cache"),
    )

    child_keys = {field.child_key for field in result.fields if field.field in {"child_name", "grade", "subject"}}
    assert len(child_keys) == 1
    assert result.summary["llm_families_resolved"] == 1
    assert client.completions.calls == 1


def test_llm_child_resolver_low_confidence_is_logged_without_behavior_change(tmp_path: Path) -> None:
    fields = [
        _child_field("child_name", "Степан", child_key="child_a", source_ref="mango:1"),
        _child_field("child_name", "Стёпа", child_key="child_b", source_ref="mango:2"),
    ]
    trace_path = tmp_path / "trace.jsonl"
    client = _FakeClient(
        [
            lambda prompt: _children_payload(
                prompt,
                [
                    {
                        "child_id": "child_1",
                        "canonical_name": "Степан",
                        "name_variants": ["Степан", "Стёпа"],
                        "grades": [],
                        "subjects": [],
                        "mention_ids": "all",
                        "merge_confidence": "low",
                    }
                ],
            )
        ]
    )

    result = apply_llm_child_resolver_to_fields(
        fields,
        profile_phones={"cust-1": "+79990000000"},
        config=_resolver_config(tmp_path, trace_path=trace_path),
        client=client,
        cache=LLMResponseCache(enabled=True, root_dir=tmp_path / "cache"),
    )
    trace_event = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[0])

    assert result.summary["llm_families_resolved"] == 1
    assert result.summary["llm_merge_confidence_low_children"] == 1
    assert len({field.child_key for field in result.fields if field.field == "child_name"}) == 1
    assert trace_event["model_children"][0]["merge_confidence"] == "low"


def test_llm_child_resolver_child_key_is_stable_across_canonical_choice(tmp_path: Path) -> None:
    fields = [
        _child_field("child_name", "Дубинина Елизавета Андреевна", child_key="child_a", source_ref="mango:1"),
        _child_field("child_name", "Лиза", child_key="child_b", source_ref="mango:2"),
    ]

    first = apply_llm_child_resolver_to_fields(
        fields,
        profile_phones={"cust-1": "+79990000000"},
        config=_resolver_config(tmp_path),
        client=_FakeClient(
            [
                lambda prompt: _children_payload(
                    prompt,
                    [
                        {
                            "child_id": "child_1",
                            "canonical_name": "Дубинина Елизавета Андреевна",
                            "name_variants": ["Дубинина Елизавета Андреевна", "Лиза"],
                            "grades": [],
                            "subjects": [],
                            "mention_ids": "all",
                            "merge_confidence": "high",
                        }
                    ],
                )
            ]
        ),
        cache=LLMResponseCache(enabled=False, root_dir=tmp_path / "cache1"),
    )
    second = apply_llm_child_resolver_to_fields(
        fields,
        profile_phones={"cust-1": "+79990000000"},
        config=_resolver_config(tmp_path),
        client=_FakeClient(
            [
                lambda prompt: _children_payload(
                    prompt,
                    [
                        {
                            "child_id": "child_1",
                            "canonical_name": "Лиза",
                            "name_variants": ["Дубинина Елизавета Андреевна", "Лиза"],
                            "grades": [],
                            "subjects": [],
                            "mention_ids": "all",
                            "merge_confidence": "high",
                        }
                    ],
                )
            ]
        ),
        cache=LLMResponseCache(enabled=False, root_dir=tmp_path / "cache2"),
    )

    first_keys = {field.child_key for field in first.fields if field.field == "child_name"}
    second_keys = {field.child_key for field in second.fields if field.field == "child_name"}
    assert len(first_keys) == 1
    assert first_keys == second_keys


def test_llm_child_resolver_rejects_children_count_growth_above_input_names(tmp_path: Path) -> None:
    fields = [
        _child_field("child_name", "Анна", child_key="child_a", source_ref="mango:1"),
        _child_field("subject", "физика", child_key="child_b", source_ref="mango:2"),
    ]
    client = _FakeClient(
        [
            lambda prompt: _children_payload(
                prompt,
                [
                    {
                        "child_id": "child_1",
                        "canonical_name": "Анна",
                        "name_variants": ["Анна"],
                        "grades": [],
                        "subjects": [],
                        "mention_ids": [0],
                        "merge_confidence": "high",
                    },
                    {
                        "child_id": "child_2",
                        "canonical_name": "",
                        "name_variants": [],
                        "grades": [],
                        "subjects": ["физика"],
                        "mention_ids": [1],
                        "merge_confidence": "low",
                    },
                    {
                        "child_id": "child_3",
                        "canonical_name": "",
                        "name_variants": [],
                        "grades": [],
                        "subjects": [],
                        "mention_ids": [],
                        "merge_confidence": "low",
                    },
                ],
            )
        ]
    )

    result = apply_llm_child_resolver_to_fields(
        fields,
        profile_phones={"cust-1": "+79990000000"},
        config=_resolver_config(tmp_path),
        client=client,
        cache=LLMResponseCache(enabled=True, root_dir=tmp_path / "cache"),
    )

    assert result.fields == fields
    assert result.summary["llm_families_failed_soft"] == 1


def test_llm_child_resolver_merges_full_name_and_short_name_as_one_child(tmp_path: Path) -> None:
    fields = [
        _child_field("child_name", "Дубинина Елизавета Андреевна", child_key="child_a", source_ref="mango:1"),
        _child_field("child_name", "Лиза", child_key="child_b", source_ref="mango:2"),
    ]
    client = _FakeClient(
        [
            lambda prompt: _children_payload(
                prompt,
                [
                    {
                        "child_id": "child_1",
                        "canonical_name": "Дубинина Елизавета Андреевна",
                        "name_variants": ["Дубинина Елизавета Андреевна", "Лиза"],
                        "grades": [],
                        "subjects": [],
                        "mention_ids": "all",
                        "merge_confidence": "high",
                    }
                ],
            )
        ]
    )

    result = apply_llm_child_resolver_to_fields(
        fields,
        profile_phones={"cust-1": "+79990000000"},
        config=_resolver_config(tmp_path),
        client=client,
        cache=LLMResponseCache(enabled=True, root_dir=tmp_path / "cache"),
    )

    assert result.summary["llm_families_resolved"] == 1
    assert result.summary["llm_families_failed_soft"] == 0
    assert len({field.child_key for field in result.fields if field.field == "child_name"}) == 1


def test_llm_child_resolver_merges_alias_spellings_as_one_child(tmp_path: Path) -> None:
    fields = [
        _child_field("child_name", "Колосов Даниил", child_key="child_a", source_ref="mango:1"),
        _child_field("child_name", "Даня", child_key="child_b", source_ref="mango:2"),
    ]
    client = _FakeClient(
        [
            lambda prompt: _children_payload(
                prompt,
                [
                    {
                        "child_id": "child_1",
                        "canonical_name": "Колосов Даниил",
                        "name_variants": ["Колосов Даниил", "Даня"],
                        "grades": [],
                        "subjects": [],
                        "mention_ids": "all",
                        "merge_confidence": "high",
                    }
                ],
            )
        ]
    )

    result = apply_llm_child_resolver_to_fields(
        fields,
        profile_phones={"cust-1": "+79990000000"},
        config=_resolver_config(tmp_path),
        client=client,
        cache=LLMResponseCache(enabled=True, root_dir=tmp_path / "cache"),
    )

    assert result.summary["llm_families_resolved"] == 1
    assert result.summary["llm_families_failed_soft"] == 0
    assert len({field.child_key for field in result.fields if field.field == "child_name"}) == 1


def test_llm_child_resolver_merges_asr_variant_and_writes_review_diagnostics(tmp_path: Path) -> None:
    fields = [
        _child_field("child_name", "Майчислав Леонидович", child_key="child_a", source_ref="mango:1"),
        _child_field("child_name", "Вячеслав Леонидович", child_key="child_b", source_ref="mango:2"),
    ]
    trace_path = tmp_path / "trace.jsonl"
    diagnostics_path = tmp_path / "name_review_diagnostics.local.jsonl"
    client = _FakeClient(
        [
            lambda prompt: _children_payload(
                prompt,
                [
                    {
                        "child_id": "child_1",
                        "canonical_name": "Майчислав Леонидович",
                        "name_variants": ["Майчислав Леонидович", "Вячеслав Леонидович"],
                        "grades": [],
                        "subjects": [],
                        "mention_ids": "all",
                        "merge_confidence": "low",
                    }
                ],
            )
        ]
    )

    result = apply_llm_child_resolver_to_fields(
        fields,
        profile_phones={"cust-1": "+79990000000"},
        config=_resolver_config(tmp_path, trace_path=trace_path, name_diagnostics_path=diagnostics_path),
        client=client,
        cache=LLMResponseCache(enabled=True, root_dir=tmp_path / "cache"),
    )
    trace_event = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[0])
    diagnostics_event = json.loads(diagnostics_path.read_text(encoding="utf-8").splitlines()[0])

    assert result.summary["llm_families_resolved"] == 1
    assert result.summary["llm_merge_confidence_low_children"] == 1
    assert len({field.child_key for field in result.fields if field.field == "child_name"}) == 1
    assert trace_event["error_code"] == ""
    assert trace_event["model_children"][0]["merge_confidence"] == "low"
    assert diagnostics_event["reason"] == "low_confidence_multi_named"
    assert diagnostics_event["accepted"] is True
    assert [item["spellings"] for item in diagnostics_event["grouped_name_spellings"]]


def test_llm_child_resolver_rejects_extra_response_fields(tmp_path: Path) -> None:
    fields = [_child_field("grade", "7", child_key="child_a"), _child_field("subject", "физика", child_key="child_b")]
    payload = {
        "children": [
            {
                "child_id": "child_1",
                "canonical_name": "",
                "name_variants": [],
                "grades": ["7"],
                "subjects": ["физика"],
                "mention_ids": [],
                "merge_confidence": "high",
                "reason": "debug",
            }
        ]
    }
    client = _FakeClient([lambda prompt: {**payload, "children": [{**payload["children"][0], "mention_ids": _prompt_mention_ids(prompt)}]}])

    result = apply_llm_child_resolver_to_fields(
        fields,
        profile_phones={"cust-1": "+79990000000"},
        config=_resolver_config(tmp_path),
        client=client,
        cache=LLMResponseCache(enabled=True, root_dir=tmp_path / "cache"),
    )

    assert result.fields == fields
    assert result.summary["llm_families_failed_soft"] == 1
    assert result.summary["child_slot_fields_rekeyed"] == 0


def test_llm_child_resolver_defaults_invalid_merge_confidence_to_low(tmp_path: Path) -> None:
    fields = [_child_field("grade", "7", child_key="child_a"), _child_field("subject", "физика", child_key="child_b")]
    trace_path = tmp_path / "trace.jsonl"
    client = _FakeClient(
        [
            lambda prompt: _children_payload(
                prompt,
                [
                    {
                        "child_id": "child_1",
                        "canonical_name": "",
                        "name_variants": [],
                        "grades": ["7"],
                        "subjects": ["физика"],
                        "mention_ids": "all",
                        "merge_confidence": "medium",
                    }
                ],
            )
        ]
    )

    result = apply_llm_child_resolver_to_fields(
        fields,
        profile_phones={"cust-1": "+79990000000"},
        config=_resolver_config(tmp_path, trace_path=trace_path),
        client=client,
        cache=LLMResponseCache(enabled=True, root_dir=tmp_path / "cache"),
    )
    trace_event = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[0])

    assert result.summary["llm_families_resolved"] == 1
    assert result.summary["llm_families_failed_soft"] == 0
    assert result.summary["llm_merge_confidence_low_children"] == 1
    assert trace_event["model_children"][0]["merge_confidence"] == "low"


def test_llm_child_resolver_keeps_different_names_separate(tmp_path: Path) -> None:
    fields = [
        _child_field("child_name", "Кулаков Никита", child_key="child_a", source_ref="mango:1"),
        _child_field("child_name", "Кулакова Дарья", child_key="child_b", source_ref="mango:2"),
    ]
    client = _FakeClient(
        [
            lambda prompt: _children_payload(
                prompt,
                [
                    {
                        "child_id": "child_1",
                        "canonical_name": "Кулаков Никита",
                        "name_variants": ["Кулаков Никита"],
                        "grades": [],
                        "subjects": [],
                        "mention_ids": [0],
                    },
                    {
                        "child_id": "child_2",
                        "canonical_name": "Кулакова Дарья",
                        "name_variants": ["Кулакова Дарья"],
                        "grades": [],
                        "subjects": [],
                        "mention_ids": [1],
                    },
                ],
            )
        ]
    )

    result = apply_llm_child_resolver_to_fields(
        fields,
        profile_phones={"cust-1": "+79990000000"},
        config=_resolver_config(tmp_path),
        client=client,
        cache=LLMResponseCache(enabled=True, root_dir=tmp_path / "cache"),
    )

    child_keys = {field.child_key for field in result.fields if field.field == "child_name"}
    assert len(child_keys) == 2
    assert result.summary["llm_families_resolved"] == 1


def test_llm_child_resolver_attaches_nameless_mentions_to_named_child(tmp_path: Path) -> None:
    fields = [
        _child_field("child_name", "Анна", child_key="child_name_hash", source_ref="mango:1"),
        _child_field("grade", "7", child_key="child_grade_hash", source_ref="mango:2"),
        _child_field("subject", "математика", child_key="child_subject_hash", source_ref="mango:3"),
    ]
    client = _FakeClient(
        [
            lambda prompt: _children_payload(
                prompt,
                [
                    {
                        "child_id": "child_1",
                        "canonical_name": "Анна",
                        "name_variants": ["Анна"],
                        "grades": ["7"],
                        "subjects": ["математика"],
                        "mention_ids": "all",
                    }
                ],
            )
        ]
    )

    result = apply_llm_child_resolver_to_fields(
        fields,
        profile_phones={"cust-1": "+79990000000"},
        config=_resolver_config(tmp_path),
        client=client,
        cache=LLMResponseCache(enabled=True, root_dir=tmp_path / "cache"),
    )

    child_keys = {field.child_key for field in result.fields if field.field in {"child_name", "grade", "subject"}}
    assert len(child_keys) == 1
    assert result.summary["child_slot_fields_rekeyed"] == 3


def test_llm_child_resolver_rejects_incompatible_same_period_grades(tmp_path: Path) -> None:
    fields = [
        _child_field("grade", "5 класс", child_key="child_a", source_ref="mango:1", event_at=NOW),
        _child_field("grade", "9 класс", child_key="child_b", source_ref="mango:2", event_at=NOW.replace(month=7)),
    ]
    client = _FakeClient(
        [
            lambda prompt: _children_payload(
                prompt,
                [
                    {
                        "child_id": "child_1",
                        "canonical_name": "",
                        "name_variants": [],
                        "grades": ["5 класс", "9 класс"],
                        "subjects": [],
                        "mention_ids": "all",
                    }
                ],
            )
        ]
    )

    result = apply_llm_child_resolver_to_fields(
        fields,
        profile_phones={"cust-1": "+79990000000"},
        config=_resolver_config(tmp_path),
        client=client,
        cache=LLMResponseCache(enabled=True, root_dir=tmp_path / "cache"),
    )

    assert result.fields == fields
    assert result.summary["llm_families_failed_soft"] == 1


def test_llm_child_resolver_allows_grade_progression_9_to_11(tmp_path: Path) -> None:
    fields = [
        _child_field(
            "child_name",
            "Анна",
            child_key="child_a",
            source_ref="mango:1",
            event_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        ),
        _child_field(
            "grade",
            "9 класс",
            child_key="child_a",
            source_ref="mango:1",
            event_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        ),
        _child_field(
            "child_name",
            "Анна",
            child_key="child_b",
            source_ref="mango:2",
            event_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ),
        _child_field(
            "grade",
            "11 класс",
            child_key="child_b",
            source_ref="mango:2",
            event_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ),
    ]
    client = _FakeClient(
        [
            lambda prompt: _children_payload(
                prompt,
                [
                    {
                        "child_id": "child_1",
                        "canonical_name": "Анна",
                        "name_variants": ["Анна"],
                        "grades": ["9 класс", "11 класс"],
                        "subjects": [],
                        "mention_ids": "all",
                    }
                ],
            )
        ]
    )

    result = apply_llm_child_resolver_to_fields(
        fields,
        profile_phones={"cust-1": "+79990000000"},
        config=_resolver_config(tmp_path),
        client=client,
        cache=LLMResponseCache(enabled=True, root_dir=tmp_path / "cache"),
    )
    active = [field for field in apply_superseded_rules(result.fields) if not field.superseded_by and field.field == "grade"]

    assert result.summary["llm_families_resolved"] == 1
    assert len(active) == 1
    assert active[0].value == "11 класс"


def test_llm_child_resolver_requires_stoplist_file(tmp_path: Path) -> None:
    with pytest.raises(ChildResolverError, match="stoplist"):
        apply_llm_child_resolver_to_fields(
            [_child_field("grade", "7", child_key="child_a"), _child_field("subject", "физика", child_key="child_b")],
            profile_phones={"cust-1": "+79990000000"},
            config=ChildResolverConfig(provider="openai", stoplist_path=tmp_path / "missing.json"),
            client=_FakeClient([]),
            cache=LLMResponseCache(enabled=True, root_dir=tmp_path / "cache"),
        )


def test_llm_child_resolver_skips_shared_phone_from_stoplist(tmp_path: Path) -> None:
    fields = [_child_field("grade", "7", child_key="child_a"), _child_field("subject", "физика", child_key="child_b")]
    result = apply_llm_child_resolver_to_fields(
        fields,
        profile_phones={"cust-1": "+79990000000"},
        config=_resolver_config(tmp_path, stoplist_phone="+79990000000"),
        client=_FakeClient([]),
        cache=LLMResponseCache(enabled=True, root_dir=tmp_path / "cache"),
    )

    assert result.fields == fields
    assert result.summary["llm_families_skipped_shared_phone"] == 1


def test_llm_child_resolver_invalid_json_is_fail_soft(tmp_path: Path) -> None:
    fields = [_child_field("grade", "7", child_key="child_a"), _child_field("subject", "физика", child_key="child_b")]

    result = apply_llm_child_resolver_to_fields(
        fields,
        profile_phones={"cust-1": "+79990000000"},
        config=_resolver_config(tmp_path),
        client=_FakeClient(["not json"]),
        cache=LLMResponseCache(enabled=True, root_dir=tmp_path / "cache"),
    )

    assert result.fields == fields
    assert result.summary["llm_families_failed_soft"] == 1


def test_llm_child_resolver_rejects_empty_mention_ids(tmp_path: Path) -> None:
    fields = [_child_field("grade", "7", child_key="child_a"), _child_field("subject", "физика", child_key="child_b")]
    client = _FakeClient(
        [
            {
                "children": [
                    {
                        "child_id": "child_1",
                        "canonical_name": "",
                        "name_variants": [],
                        "grades": ["7"],
                        "subjects": ["физика"],
                        "mention_ids": [],
                        "merge_confidence": "high",
                    }
                ]
            }
        ]
    )

    result = apply_llm_child_resolver_to_fields(
        fields,
        profile_phones={"cust-1": "+79990000000"},
        config=_resolver_config(tmp_path),
        client=client,
        cache=LLMResponseCache(enabled=True, root_dir=tmp_path / "cache"),
    )

    assert result.fields == fields
    assert result.summary["llm_families_failed_soft"] == 1


def test_llm_child_resolver_uses_cache_on_repeat(tmp_path: Path) -> None:
    fields = [_child_field("grade", "7", child_key="child_a"), _child_field("subject", "физика", child_key="child_b")]
    cache = LLMResponseCache(enabled=True, root_dir=tmp_path / "cache")
    config = _resolver_config(tmp_path)
    first_client = _FakeClient(
        [
            lambda prompt: _children_payload(
                prompt,
                [
                    {
                        "child_id": "child_1",
                        "canonical_name": "",
                        "name_variants": [],
                        "grades": ["7"],
                        "subjects": ["физика"],
                        "mention_ids": "all",
                    }
                ],
            )
        ]
    )

    first = apply_llm_child_resolver_to_fields(
        fields,
        profile_phones={"cust-1": "+79990000000"},
        config=config,
        client=first_client,
        cache=cache,
    )
    second_client = _FakeClient([])
    second = apply_llm_child_resolver_to_fields(
        fields,
        profile_phones={"cust-1": "+79990000000"},
        config=config,
        client=second_client,
        cache=cache,
    )

    assert first.summary["llm_cache_hits"] == 0
    assert second.summary["llm_cache_hits"] == 1
    assert second.summary["llm_calls_total"] == 0
    assert first_client.completions.calls == 1
    assert second_client.completions.calls == 0


def test_llm_child_resolver_preserves_brand_per_row(tmp_path: Path) -> None:
    fields = [
        _child_field("child_name", "Анна", child_key="child_a", source_ref="mango:1", brand="unpk"),
        _child_field("grade", "7", child_key="child_b", source_ref="mango:2", brand="foton"),
    ]
    client = _FakeClient(
        [
            lambda prompt: _children_payload(
                prompt,
                [
                    {
                        "child_id": "child_1",
                        "canonical_name": "Анна",
                        "name_variants": ["Анна"],
                        "grades": ["7"],
                        "subjects": [],
                        "mention_ids": "all",
                    }
                ],
            )
        ]
    )

    result = apply_llm_child_resolver_to_fields(
        fields,
        profile_phones={"cust-1": "+79990000000"},
        config=_resolver_config(tmp_path),
        client=client,
        cache=LLMResponseCache(enabled=True, root_dir=tmp_path / "cache"),
    )

    brands_by_field = {field.field: field.brand for field in result.fields}
    assert brands_by_field == {"child_name": "unpk", "grade": "foton"}
    assert result.summary["llm_brand_changed_fields"] == 0


def test_builder_does_not_merge_different_children_or_ambiguous_diminutive(tmp_path: Path) -> None:
    timeline_db = _timeline_db(tmp_path)
    master_db = _master_calls_db(
        tmp_path,
        [
            (
                100,
                "2026-01-10T10:00:00+00:00",
                _children_analysis(
                    [
                        {"name": "Ермаков Тимур", "grade": "7", "subjects": ["математика"]},
                        {"name": "Ермаков Олег", "grade": "7", "subjects": ["физика"]},
                    ]
                ),
            ),
            (
                101,
                "2026-02-10T10:00:00+00:00",
                _children_analysis(
                    [
                        {"name": "Саша", "grade": "8", "subjects": ["математика"]},
                        {"name": "Александр", "grade": "8", "subjects": ["физика"]},
                    ]
                ),
            ),
        ],
    )
    profiles_db = tmp_path / "profiles.sqlite"

    report = CustomerProfileBuilder(
        CustomerProfileBuildOptions(
            timeline_db=timeline_db,
            profiles_db=profiles_db,
            master_calls_db=master_db,
            customer_ids=("cust-1",),
        )
    ).build()

    with CustomerProfileSQLiteStore(profiles_db) as store:
        fields = store.active_fields("cust-1")

    assert report["child_slot_merge"]["merge_candidate_groups"] == 0
    assert not [row for row in fields if row["field"] == "child_slot_merge_candidate"]
    assert {row["value"] for row in fields if row["field"] == "child_name"} == {
        "Ермаков Тимур",
        "Ермаков Олег",
        "Саша",
        "Александр",
    }


def test_builder_child_slot_merge_stays_within_profile(tmp_path: Path) -> None:
    timeline_db = _timeline_db(tmp_path, duplicate_phone_customer=True)
    master_db = _master_calls_db(
        tmp_path,
        [
            (100, "2026-01-10T10:00:00+00:00", _analysis(grade="7", child_name="Рома")),
            (101, "2026-02-10T10:00:00+00:00", _analysis(grade="8", child_name="Роман")),
        ],
    )
    profiles_db = tmp_path / "profiles.sqlite"

    report = CustomerProfileBuilder(
        CustomerProfileBuildOptions(
            timeline_db=timeline_db,
            profiles_db=profiles_db,
            master_calls_db=master_db,
            customer_ids=("cust-1", "cust-2"),
        )
    ).build()

    with CustomerProfileSQLiteStore(profiles_db) as store:
        fields_1 = store.active_fields("cust-1")
        fields_2 = store.active_fields("cust-2")

    assert report["ambiguous_calls"] == 2
    assert report["child_slot_merge"]["merge_candidate_groups"] == 0
    assert not [row for row in fields_1 + fields_2 if row["source_system"] == "mango_processed_summary"]


def test_builder_keeps_explicit_children_separate_by_child_key(tmp_path: Path) -> None:
    timeline_db = _timeline_db(tmp_path)
    master_db = _master_calls_db(
        tmp_path,
        [
            (
                100,
                "2026-01-10T10:00:00+00:00",
                {
                    "structured_fields": {
                        "children": [
                            {"name": "Первый", "grade": "7", "subjects": ["математика"]},
                            {"name": "Второй", "grade": "9", "subjects": ["физика"]},
                        ],
                        "people": {},
                        "interests": {},
                        "next_step": {},
                    }
                },
            )
        ],
    )
    profiles_db = tmp_path / "profiles.sqlite"

    CustomerProfileBuilder(
        CustomerProfileBuildOptions(
            timeline_db=timeline_db,
            profiles_db=profiles_db,
            master_calls_db=master_db,
            customer_ids=("cust-1",),
        )
    ).build()

    with CustomerProfileSQLiteStore(profiles_db) as store:
        fields = store.active_fields("cust-1")

    names = {row["value"]: row["child_key"] for row in fields if row["field"] == "child_name"}
    grades = {row["child_key"]: row["value"] for row in fields if row["field"] == "grade"}
    assert set(names) == {"Первый", "Второй"}
    assert names["Первый"] != names["Второй"]
    assert grades[names["Первый"]] == "7"
    assert grades[names["Второй"]] == "9"


def test_builder_does_not_mix_single_child_from_different_calls(tmp_path: Path) -> None:
    timeline_db = _timeline_db(tmp_path)
    master_db = _master_calls_db(
        tmp_path,
        [
            (100, "2026-01-10T10:00:00+00:00", _analysis(grade="7", child_name="Первый")),
            (101, "2026-02-10T10:00:00+00:00", _analysis(grade="9", child_name="Второй", subjects=["физика"])),
        ],
    )
    profiles_db = tmp_path / "profiles.sqlite"

    CustomerProfileBuilder(
        CustomerProfileBuildOptions(
            timeline_db=timeline_db,
            profiles_db=profiles_db,
            master_calls_db=master_db,
            customer_ids=("cust-1",),
        )
    ).build()

    with CustomerProfileSQLiteStore(profiles_db) as store:
        fields = store.active_fields("cust-1")

    grades = {row["value"] for row in fields if row["field"] == "grade"}
    assert {"7", "9"}.issubset(grades)


def test_builder_skips_ambiguous_phone_and_counts_it(tmp_path: Path) -> None:
    timeline_db = _timeline_db(tmp_path, duplicate_phone_customer=True)
    master_db = _master_calls_db(tmp_path, [(100, "2026-01-10T10:00:00+00:00", _analysis())])
    profiles_db = tmp_path / "profiles.sqlite"

    report = CustomerProfileBuilder(
        CustomerProfileBuildOptions(
            timeline_db=timeline_db,
            profiles_db=profiles_db,
            master_calls_db=master_db,
            customer_ids=("cust-1", "cust-2"),
        )
    ).build()

    with CustomerProfileSQLiteStore(profiles_db) as store:
        fields_1 = store.active_fields("cust-1")
        fields_2 = store.active_fields("cust-2")
    assert report["ambiguous_calls"] == 1
    assert not [row for row in fields_1 + fields_2 if row["source_system"] == "mango_processed_summary"]


def test_builder_matches_master_call_by_last_10_digits(tmp_path: Path) -> None:
    timeline_db = _timeline_db(tmp_path)
    analysis = _analysis()
    analysis["_phone"] = "9990000000"
    master_db = _master_calls_db(tmp_path, [(100, "2026-01-10T10:00:00+00:00", analysis)])
    profiles_db = tmp_path / "profiles.sqlite"

    report = CustomerProfileBuilder(
        CustomerProfileBuildOptions(
            timeline_db=timeline_db,
            profiles_db=profiles_db,
            master_calls_db=master_db,
            customer_ids=("cust-1",),
        )
    ).build()

    assert report["unmatched_calls"] == 0
    with CustomerProfileSQLiteStore(profiles_db) as store:
        assert any(row["field"] == "grade" for row in store.active_fields("cust-1"))


def test_builder_counts_master_call_without_phone_as_unmatched(tmp_path: Path) -> None:
    timeline_db = _timeline_db(tmp_path)
    analysis = _analysis()
    analysis["_phone"] = None
    master_db = _master_calls_db(tmp_path, [(100, "2026-01-10T10:00:00+00:00", analysis)])
    profiles_db = tmp_path / "profiles.sqlite"

    report = CustomerProfileBuilder(
        CustomerProfileBuildOptions(
            timeline_db=timeline_db,
            profiles_db=profiles_db,
            master_calls_db=master_db,
            customer_ids=("cust-1",),
        )
    ).build()

    with CustomerProfileSQLiteStore(profiles_db) as store:
        fields = store.active_fields("cust-1")

    assert report["unmatched_calls"] == 1
    assert not [row for row in fields if row["source_system"] == "mango_processed_summary"]


def test_store_enforces_profile_field_foreign_key(tmp_path: Path) -> None:
    profiles_db = tmp_path / "profiles.sqlite"
    with CustomerProfileSQLiteStore(profiles_db):
        pass
    con = sqlite3.connect(profiles_db)
    con.execute("PRAGMA foreign_keys = ON")
    try:
        with pytest.raises(sqlite3.IntegrityError):
            con.execute(
                """
                INSERT INTO profile_fields (
                  field_id, profile_id, field, value, child_key, brand, source_system,
                  source_ref, event_at, quote, superseded_by
                ) VALUES ('f1', 'missing', 'grade', '8', '', 'unknown', 'test', 'test', ?, '', '')
                """,
                (NOW.isoformat(),),
            )
    finally:
        con.close()


def test_cli_field_preview_does_not_expose_raw_value() -> None:
    preview = safe_field_preview({"field": "child_name", "value": "Иван Петров", "brand": "foton"})

    assert preview["field"] == "child_name"
    assert preview["has_value"] is True
    assert preview["value_len"] == len("Иван Петров")
    assert "value" not in preview


def test_profile_field_requires_origin_and_timezone() -> None:
    with pytest.raises(ValueError, match="source_ref"):
        ProfileFieldCandidate(
            profile_id="cust-1",
            field="grade",
            value="8",
            source_system="mango_processed_summary",
            source_ref="",
            event_at=NOW,
        )
    with pytest.raises(ValueError, match="timezone-aware"):
        ProfileFieldCandidate(
            profile_id="cust-1",
            field="grade",
            value="8",
            source_system="mango_processed_summary",
            source_ref="mango:1",
            event_at=datetime(2026, 1, 1, 10, 0),
        )


def test_profile_build_records_timeline_sha256(tmp_path: Path) -> None:
    timeline_db = _timeline_db(tmp_path)
    master_db = _master_calls_db(tmp_path, [(100, "2026-01-10T10:00:00+00:00", _analysis())])
    profiles_db = tmp_path / "profiles.sqlite"

    CustomerProfileBuilder(
        CustomerProfileBuildOptions(
            timeline_db=timeline_db,
            profiles_db=profiles_db,
            master_calls_db=master_db,
            customer_ids=("cust-1",),
        )
    ).build()

    con = sqlite3.connect(profiles_db)
    try:
        row = con.execute(
            "SELECT build_id, timeline_db_sha256, profiles_built FROM profile_builds"
        ).fetchone()
    finally:
        con.close()
    assert row[0]
    assert len(row[1]) == 64
    assert row[2] == 1
