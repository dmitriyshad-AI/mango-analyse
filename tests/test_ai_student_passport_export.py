import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_ai_student_passport_export as export


def save(path, data):
    path.write_text(json.dumps(data))


def freeze(root):
    path = root / "R5_SLICE_MANIFEST.json"
    manifest = export.read_json(path)
    manifest["slice_sha256"] = hashlib.sha256((root / "r5_13_slice.sqlite").read_bytes()).hexdigest()
    save(path, manifest)


@pytest.fixture
def dataset(tmp_path):
    db = sqlite3.connect(tmp_path / "r5_13_slice.sqlite")
    db.executescript("""
        CREATE TABLE identity_links (
            link_id TEXT PRIMARY KEY, tenant_id TEXT, customer_id TEXT,
            link_type TEXT, link_value TEXT, source_system TEXT, source_ref TEXT,
            match_class TEXT, record_json TEXT);
        CREATE TABLE family_links_v1 (
            tenant_id TEXT, customer_id TEXT, child_key TEXT, status TEXT,
            confidence TEXT, record_json TEXT);
        CREATE TABLE family_members_v1 (
            tenant_id TEXT, customer_id TEXT, family_id TEXT,
            membership_status TEXT, confidence TEXT);
        CREATE TABLE timeline_conflicts (
            tenant_id TEXT, status TEXT, conflict_type TEXT, record_json TEXT);
        CREATE TABLE timeline_events (
            event_id TEXT PRIMARY KEY, tenant_id TEXT, customer_id TEXT,
            event_at TEXT, source_system TEXT, source_id TEXT, source_ref TEXT,
            superseded_by TEXT, record_json TEXT);
        CREATE TABLE event_child_attribution_v1 (
            tenant_id TEXT, event_id TEXT, customer_id TEXT,
            child_key TEXT, status TEXT, confidence TEXT);
        CREATE TABLE email_summary_cache_v1 (
            message_sha256 TEXT PRIMARY KEY, summary_text TEXT, text_sha256 TEXT);
    """)
    rows = []
    selections = []
    for number in (1, 2):
        student = "student-" + str(number)
        db.execute("INSERT INTO identity_links VALUES (?,?,?,?,?,?,?,?,?)", (
            "link-" + str(number), "foton", "customer:test", "tallanto_student_id",
            student, "tallanto_snapshot", "student:" + student, "strong_unique", "{}"))
        db.execute("INSERT INTO family_links_v1 VALUES (?,?,?,?,?,?)", (
            "foton", "customer:test", "child-" + str(number), "confident", "high",
            json.dumps({"tallanto_student_ids": [student]})))
        rows.append([number, student, "shared-phone", "parent[at]example.invalid"] + [""] * 17 + [student, "passport-" + str(number)])
        rows[-1][7] = "registered, attendance unconfirmed"
        selections.append({"number": number, "tallanto_id": student, "passport_key": "passport-" + str(number)})
    db.execute("INSERT INTO family_members_v1 VALUES (?,?,?,?,?)", ("foton", "customer:test", "family:test", "confident", "high"))
    for number in range(503):
        event = {
            "event_id": "event:" + str(number), "tenant_id": "foton", "customer_id": "customer:test",
            "event_at": "2026-01-02T03:04:05+00:00", "source_system": "synthetic",
            "source_id": str(number), "source_ref": "synthetic:" + str(number),
            "summary": "Complete evidence. " * 80, "direction": "inbound",
            "match_status": "strong_unique", "metadata": {"brand": "foton"}, "record": {},
        }
        db.execute("INSERT INTO timeline_events VALUES (?,?,?,?,?,?,?,?,?)", (
            event["event_id"], "foton", "customer:test", event["event_at"], "synthetic",
            str(number), event["source_ref"], "event:1" if number == 502 else None, json.dumps(event)))
        db.execute("INSERT INTO event_child_attribution_v1 VALUES (?,?,?,?,?,?)", (
            "foton", event["event_id"], "customer:test", "child-1", "matched", "high"))
    db.commit()
    db.close()
    cells = lambda row: {"values": [{"userEnteredValue": {"stringValue": str(v)}} for v in row]}
    save(tmp_path / "GOOGLE_BEFORE.json", {"structuredContent": {"sheets": [{"data": [
        {"startRow": 1, "rowData": [cells(["", "Advanced group 2026"]) ]},
        {"startRow": 5, "rowData": [cells(r) for r in rows]},
    ]}]}})
    save(tmp_path / "HISTORY_BEFORE.json", {"structuredContent": {"sheets": [{"data": [
        {"startRow": 5, "rowData": [cells([1, "legacy child", "date"])]},
        {"startRow": 5, "startColumn": 14, "rowData": [cells(["uncertain", "", "", "", "passport-1"])]},
    ]}]}})
    save(tmp_path / "HISTORY_FULLTEXT_BEFORE.json", {"structuredContent": {
        "range": "'History'!N6:N6", "values": [["Original full conversation. " * 30]]}})
    save(tmp_path / "CANONICAL_CALLS_TARGETED.json", {"rows": {}})
    save(tmp_path / "R5_SLICE_MANIFEST.json", {
        "tenant_id": "foton", "as_of": "2026-09-06T15:01:48+00:00", "selections": selections,
        "source_dates": {"roster_contacts": "2026-09-05"}, "extra_evidence": [],
    })
    freeze(tmp_path)
    return tmp_path


def semantic(source):
    return {
        "as_of": source["as_of"], "source_digest": export.stable_digest(source),
        "review_status": "claude_codex_reviewed", "event_summaries": {},
        "rows": [{**{k: r[k] for k in ("name", "phones", "email", "tallanto_id", "group", "academic_year", "group_status", "group_verified_at",
                                      "identity_scope_status", "identity_conflict", "legacy_quarantine_warning", "family_scope_complete")},
                  "passport_key": r["passport_key"], "manager_summary": "Reviewed facts.",
                  "learning_profile": "Unknown diagnostic level.", "situation_risks": "Evidence limited.",
                  "quality_note": "Shared contact, separate child evidence.",
                  "next_action": "Review placement at first lesson.", "evidence_refs": ["roster"] +
                  [e["event_id"] for e in r["history"] if e["scope"] != "other_child"][:1],
                  "legacy_source_refs": [e["ref"] for e in r["legacy_history"]],
                  "history": [{**{k: e[k] for k in ("event_id", "at", "source", "scope", "author", "direction",
                                                   "text_source", "canonical_summary_verified")},
                               "parts": export.parts("".join(e["full_text_parts"]))} for e in r["history"]]}
                 for r in source["rows"]],
    }


def test_pagination_full_text_two_children_and_inactive(dataset):
    source = export.collect(dataset)
    first, second = source["rows"]
    assert len(first["history"]) == 502
    assert len({e["event_id"] for e in first["history"]}) == 502
    assert first["history"][0]["scope"] == "child"
    assert second["history"][0]["scope"] == "other_child"
    assert len("".join(first["history"][0]["full_text_parts"])) > 290
    assert first["raw_inactive"][0]["active"] is False
    assert first["raw_inactive"][0]["source"]["event_id"] not in {e["event_id"] for e in first["history"]}
    assert first["academic_year"] == "2026/27"
    assert "unconfirmed" in first["group_status"]
    assert first["legacy_history"][0]["scope"] == "unresolved_legacy"
    assert len("".join(first["legacy_history"][0]["full_text_parts"])) > 290
    assert first["legacy_history"][0]["cells"]["14"] == "uncertain"


@pytest.mark.parametrize("change", ["conflict", "pending", "wrong_owner", "missing", "low", "brand", "record_brand", "payload_brand"])
def test_identity_negative_controls(change):
    event = {"customer_id": "owner", "match_status": "strong_unique", "metadata": {"brand": "foton"}}
    attr = {"customer_id": "owner", "child_key": "child", "status": "matched", "confidence": "high"}
    child = {"child_key": "child", "status": "confident", "confidence": "high"}
    if change == "pending":
        event["metadata"]["pending_attribution"] = True
    if change == "wrong_owner":
        attr["customer_id"] = "another-owner"
    if change == "missing":
        attr = {}
    if change == "low":
        child["confidence"] = "low"
    if change == "brand":
        event["metadata"]["brand"] = "unpk"
    if change == "record_brand":
        event["record"] = {"brand": "unpk"}
    if change == "payload_brand":
        event["record"] = {"payload": {"brand": "unpk"}}
    assert export.attribution(event, attr, ["owner"], [child], change == "conflict") != "child"


@pytest.mark.parametrize("scenario", ["exact", "conflict", "record_brand"])
def test_old_google_warning_does_not_override_current_evidence(dataset, scenario):
    if scenario != "exact":
        with sqlite3.connect(dataset / "r5_13_slice.sqlite") as con:
            if scenario == "conflict":
                con.execute("INSERT INTO timeline_conflicts VALUES (?,?,?,?)", (
                    "foton", "open", "family_identity_conflict", json.dumps({"entity_refs": ["customer:test"]})))
            else:
                e = json.loads(con.execute("SELECT record_json FROM timeline_events WHERE event_id='event:0'").fetchone()[0])
                e["metadata"] = {}
                e["record"] = {"brand": "unpk", "payload": {"finished_grade": 4}}
                con.execute("UPDATE timeline_events SET record_json=? WHERE event_id='event:0'", (json.dumps(e),))
    before = export.collect(dataset)["rows"][0]["history"][0]["scope"]
    path = dataset / "GOOGLE_BEFORE.json"
    data = export.read_json(path)
    data["structuredContent"]["sheets"][0]["data"][1]["rowData"][0]["values"][20] = {
        "userEnteredValue": {"stringValue": "\u041a\u0410\u0420\u0410\u041d\u0422\u0418\u041d"}}
    save(path, data)
    row = export.collect(dataset)["rows"][0]
    assert row["legacy_quarantine_warning"]
    assert row["history"][0]["scope"] == before == ("child" if scenario == "exact" else "unresolved")


@pytest.mark.parametrize("statement", ["DELETE FROM event_child_attribution_v1", "DROP TABLE event_child_attribution_v1"])
def test_missing_attribution_keeps_history(dataset, statement):
    with sqlite3.connect(dataset / "r5_13_slice.sqlite") as con:
        con.execute(statement)
    row = export.collect(dataset)["rows"][0]
    assert len(row["history"]) == 502
    assert {e["scope"] for e in row["history"]} == {"family_context"}


def test_parts_preserve_every_character():
    text = "A whole sentence.\n" * 4000
    split = export.parts(text)
    assert len(split) > 2
    assert "".join(split) == text
    assert all(len(p.encode("utf-16-le")) // 2 <= 40000 for p in split)
    assert all(p[-1].isspace() for p in split[:-1])
    with pytest.raises(ValueError, match="unbreakable"):
        export.parts("x" * 20001)


def test_email_context_and_targeted_cache(dataset):
    original = "Body evidence. " * 40 + "\n--\nSignature\nPlease move the lesson.\n> Older reply"
    with sqlite3.connect(dataset / "r5_13_slice.sqlite") as con:
        e = json.loads(con.execute("SELECT record_json FROM timeline_events WHERE event_id='event:0'").fetchone()[0])
        e.update(source_system="mail_archive_stage2", record={"message_sha256": "target", "full_clean_text": original})
        con.execute("UPDATE timeline_events SET source_system=?,record_json=? WHERE event_id='event:0'", (e["source_system"], json.dumps(e)))
        con.executemany("INSERT INTO email_summary_cache_v1 VALUES (?,?,?)", [("target", "Reviewed head", "sha"), ("not-target", "Do not export", "other-sha")])
    row = export.collect(dataset)["rows"][0]
    e = next(e for e in row["history"] if e["event_id"] == "event:0")
    assert "Please move" not in "".join(e["parts"])
    assert "Please move" in "".join(e["context_parts"])
    assert "".join(e["full_text_parts"]) == original
    assert e["email_cache"]["message_sha256"] == "target"
    assert "Do not export" not in json.dumps(row)


def test_call_id_and_importer_time_contract():
    event = {"source_id": "7", "source_ref": "mango:7", "event_at": "2025-01-02T10:42:19+00:00"}
    call = {"canonical_call_id": 7, "started_at": "2025-01-02 10:42:19"}
    assert export.call_matches(event, call)
    assert not export.call_matches(event, dict(call, canonical_call_id=8))
    assert not export.call_matches(event, dict(call, started_at="2025-01-02 11:42:19"))
    assert not export.call_matches(event, {})


def test_readonly_store_required():
    with pytest.raises(ValueError, match="writable"):
        list(export.all_events(SimpleNamespace(read_only=False), "foton", "customer:test"))


def test_full_canonical_summary_not_one_side_transcript(dataset):
    full = "Full canonical summary, both speakers. " * 70
    with sqlite3.connect(dataset / "r5_13_slice.sqlite") as con:
        e = json.loads(con.execute("SELECT record_json FROM timeline_events WHERE event_id='event:0'").fetchone()[0])
        e.update(source_system="mango_processed_summary", source_id="7", source_ref="mango:7")
        con.execute("UPDATE timeline_events SET record_json=? WHERE event_id='event:0'", (json.dumps(e),))
    save(dataset / "CANONICAL_CALLS_TARGETED.json", {"rows": {"7": {
        "canonical_call_id": 7, "started_at": "2026-01-02 03:04:05",
        "analysis_json": json.dumps({"history_summary": full}), "transcript_client": "One side only",
    }}})
    e = export.collect(dataset)["rows"][0]["history"][0]
    assert e["canonical_summary_verified"]
    assert "".join(e["full_text_parts"]) == full


@pytest.mark.parametrize("damage", ["duplicate", "wrong_id"])
def test_roster_identity_gate(dataset, damage):
    path = dataset / "R5_SLICE_MANIFEST.json"
    manifest = export.read_json(path)
    if damage == "duplicate":
        manifest["selections"].append(manifest["selections"][0])
    else:
        manifest["selections"][0]["tallanto_id"] = "wrong-student"
    save(path, manifest)
    with pytest.raises(ValueError, match="roster"):
        export.collect(dataset)


def test_unowned_only_with_exact_evidence(dataset):
    with sqlite3.connect(dataset / "r5_13_slice.sqlite") as con:
        con.row_factory = sqlite3.Row
        con.execute("UPDATE timeline_events SET customer_id=NULL,source_system='mail_archive_stage2',source_id='sha',source_ref='mail:sha' WHERE event_id='event:0'")
        con.execute("UPDATE identity_links SET source_system='mail_archive_stage2',source_ref='mail:sha',record_json=? WHERE link_id='link-1'", (json.dumps({"evidence": {"message_sha256": "sha"}}),))
        proof = {"tallanto_id": "student-1", "link_id": "link-1", "event_id": "event:0"}
        assert export.verified_extra(con, proof, "foton", "student-1")["customer_id"] is None
        with pytest.raises(ValueError, match="evidence mismatch"):
            export.verified_extra(con, dict(proof, event_id="event:1"), "foton", "student-1")
    row = export.collect(dataset)["rows"][0]
    assert "event:0" not in {e["event_id"] for e in row["history"]}


def test_repeat_digest_and_stale_semantic_atomic_rejection(dataset):
    output = dataset / "output.json"
    before = (dataset / "r5_13_slice.sqlite").read_bytes()
    first = export.run(dataset, output, allowed_root=dataset)
    original = output.read_bytes()
    second = export.run(dataset, output, allowed_root=dataset)
    assert first["source_digest"] == second["source_digest"]
    assert second["reused"] == 2
    assert output.read_bytes() == original
    assert (dataset / "r5_13_slice.sqlite").read_bytes() == before
    source = export.read_json(output)["source"]
    payload = dataset / "semantic.json"
    save(payload, semantic(source))
    export.run(dataset, output, payload, allowed_root=dataset)
    accepted = output.read_bytes()
    with pytest.raises(ValueError, match="downgrade"):
        export.run(dataset, output, allowed_root=dataset)
    with sqlite3.connect(dataset / "r5_13_slice.sqlite") as con:
        con.execute("UPDATE event_child_attribution_v1 SET confidence='low' WHERE event_id='event:0'")
    freeze(dataset)
    with pytest.raises(ValueError, match="stale semantic"):
        export.run(dataset, output, payload, allowed_root=dataset)
    assert output.read_bytes() == accepted


@pytest.mark.parametrize("damage", ["as_of", "digest", "missing_row", "review", "blank", "alien_ref"])
def test_semantic_gate(dataset, damage):
    source = export.collect(dataset)
    payload = semantic(source)
    if damage == "as_of":
        payload["as_of"] = "2020-01-01T00:00:00+00:00"
    if damage == "digest":
        payload["source_digest"] = "old"
    if damage == "missing_row":
        payload["rows"].pop()
    if damage == "review":
        payload["review_status"] = "draft"
    if damage == "blank":
        payload["rows"][0]["manager_summary"] = ""
    if damage == "alien_ref":
        payload["rows"][0]["evidence_refs"] = ["foreign-child"]
    with pytest.raises(ValueError):
        export.validate_semantic(payload, source, export.stable_digest(source))


def test_source_overwrite_escape_and_wal(dataset):
    with pytest.raises(ValueError, match="overwrite"):
        export.run(dataset, dataset / "GOOGLE_BEFORE.json", allowed_root=dataset)
    with pytest.raises(ValueError):
        export.run(dataset, Path(__file__).resolve().parents[1] / "escape.json")
    (dataset / "r5_13_slice.sqlite-wal").write_bytes(b"not frozen")
    with pytest.raises(ValueError, match="WAL"):
        export.run(dataset, dataset / "output.json", allowed_root=dataset)


def test_unassigned_legacy_and_shifted_fulltext(dataset):
    path = dataset / "HISTORY_BEFORE.json"
    data = export.read_json(path)
    data["structuredContent"]["sheets"][0]["data"][0]["rowData"][0]["values"][0] = {
        "userEnteredValue": {"stringValue": "unknown student"}}
    save(path, data)
    source = export.collect(dataset)
    assert len(source["unassigned_legacy_rows"]) == 1
    assert sum(len(r["legacy_history"]) for r in source["rows"]) == 0
    path = dataset / "HISTORY_FULLTEXT_BEFORE.json"
    full = export.read_json(path)
    full["structuredContent"]["range"] = "N7:N7"
    save(path, full)
    with pytest.raises(ValueError, match="fulltext range"):
        export.collect(dataset)


def test_increment_full_summary_and_empty_sha(dataset):
    with sqlite3.connect(dataset / "r5_13_slice.sqlite") as con:
        e = json.loads(con.execute("SELECT record_json FROM timeline_events WHERE event_id='event:0'").fetchone()[0])
        e["summary"] = "Short preview"
        e["source_system"] = "mango_processed_summary"
        e["record"] = {"call": {"call_id": e["source_id"], "source_ref": e["source_ref"],
                                 "event_at": e["event_at"], "analysis_json": {"history_summary": "Full proof. " * 100}}}
        con.execute("UPDATE timeline_events SET record_json=? WHERE event_id='event:0'", (json.dumps(e),))
        con.execute("INSERT INTO email_summary_cache_v1 VALUES ('','must not attach','')")
    result = export.collect(dataset)["rows"][0]["history"][0]
    assert result["text_source"] == "increment_summary"
    assert "".join(result["full_text_parts"]) == "Full proof. " * 100
    assert result["email_cache"] is None
    e["record"]["call"]["call_id"] = "other-id"
    assert export.increment_summary(e) == ""


def test_other_child_not_valid_semantic_evidence(dataset):
    source = export.collect(dataset)
    payload = semantic(source)
    payload["rows"][1]["evidence_refs"] = [source["rows"][1]["history"][0]["event_id"]]
    with pytest.raises(ValueError, match="outside passport"):
        export.validate_semantic(payload, source, export.stable_digest(source))


def test_hot_rollback_journal(dataset):
    (dataset / "r5_13_slice.sqlite-journal").write_bytes(b"unfinished")
    with pytest.raises(ValueError, match="journal"):
        export.run(dataset, dataset / "output.json", allowed_root=dataset)


@pytest.mark.parametrize("damage", ["row_contact", "row_quality", "event_scope", "event_text", "event_lost", "legacy_lost", "inactive_summary"])
def test_complete_projection_gate(dataset, damage):
    source = export.collect(dataset)
    payload = semantic(source)
    row = payload["rows"][0]
    if damage == "row_contact":
        row["phones"] = "foreign-contact"
    elif damage == "row_quality":
        row["quality_note"] = ""
    elif damage == "event_scope":
        row["history"][0]["scope"] = "unresolved"
    elif damage == "event_text":
        row["history"][0]["parts"] = ["truncated"]
    elif damage == "event_lost":
        row["history"].pop()
    elif damage == "legacy_lost":
        row["legacy_source_refs"] = []
    elif damage == "inactive_summary":
        payload["event_summaries"]["event:502"] = {"summary": "Revived"}
    with pytest.raises(ValueError):
        export.validate_semantic(payload, source, export.stable_digest(source))


def test_email_requires_finished_summary_without_scope_upgrade(dataset):
    source = export.collect(dataset)
    for row in source["rows"]:
        row["history"][0]["source"] = "mail_archive_stage2"
        row["history"][0]["scope"] = "unresolved"
    payload = semantic(source)
    with pytest.raises(ValueError, match="finished event summary missing"):
        export.validate_semantic(payload, source, export.stable_digest(source))
    ref = source["rows"][0]["history"][0]["event_id"]
    note = "The parent mentions a classmate and that classmate's parents, not this student's learning result."
    payload["event_summaries"][ref] = {"summary": note, "basis": "source_text_reviewed", "evidence_refs": [ref]}
    for row in payload["rows"]:
        row["history"][0]["parts"] = export.parts(note)
    export.validate_semantic(payload, source, export.stable_digest(source))
    payload["rows"][0]["history"][0]["scope"] = "child"
    with pytest.raises(ValueError, match="identity or scope"):
        export.validate_semantic(payload, source, export.stable_digest(source))


@pytest.mark.parametrize("key", ["identity_scope_status", "identity_conflict", "legacy_quarantine_warning", "family_scope_complete"])
def test_projection_cannot_hide_identity_markers(dataset, key):
    source = export.collect(dataset)
    payload = semantic(source)
    payload["rows"][0].pop(key)
    with pytest.raises(ValueError, match="projection roster mismatch"):
        export.validate_semantic(payload, source, export.stable_digest(source))


@pytest.mark.parametrize("kind", ["missing_call", "task_action", "missing_provenance"])
def test_fallback_is_not_silently_presented_as_full_text(dataset, kind):
    source = export.collect(dataset)
    for row in source["rows"]:
        event = row["history"][0]
        if kind == "missing_call":
            event["source"] = "mango_processed_summary"
        elif kind == "task_action":
            event["raw"]["record"]["action_text"] = "Actual recorded instruction, not the task stub."
    payload = semantic(source)
    if kind == "missing_provenance":
        payload["rows"][0]["history"][0].pop("text_source")
    with pytest.raises(ValueError):
        export.validate_semantic(payload, source, export.stable_digest(source))


def test_cache_detects_same_stat_json_replacement(dataset):
    import os
    output = dataset / "export.json"
    export.run(dataset, output, allowed_root=dataset)
    path = dataset / "R5_SLICE_MANIFEST.json"
    before = path.stat()
    old = path.read_text()
    path.write_text(old.replace("2026-09-05", "2026-09-04"))
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert path.stat().st_size == before.st_size
    report = export.run(dataset, output, allowed_root=dataset)
    assert report["mode"] == "full"
    assert export.read_json(output)["source"]["rows"][0]["group_verified_at"] == "2026-09-04"


@pytest.mark.parametrize("damage", [None, "phone", "scope", "time", "text", "id", "legacy_owner"])
def test_supplemental_call_is_exact_contact_not_child(damage):
    values = ["42", "2026-08-27 17:49:55", "Manager", "Inbound"] + [""] * 2 + ["+70000000001"] + [""] * 8 + ["Complete two-sided source."]
    event = {"passport_key": "p1", "sheet_row": 20, "legacy_ref": "sheet:6",
             "event_id": "google_call:sheet:0:42", "at": "2026-08-27T17:49:55+03:00",
             "source": "google_calls_sheet", "scope": "family_context", "author": "Manager", "direction": "Inbound",
             "full_text_parts": [values[15]], "canonical_summary_verified": False,
             "text_source": "google_full_transcript", "raw": {"record": {"values": list(values)}}}
    old = {5: {0: "1", 2: "27.08.2026 17:49", 13: values[15], 18: "p1"}}
    manifest = {"supplemental_calls": [event], "supplemental_call_evidence": {
        "spreadsheet_id": "sheet", "sheet_id": 0, "records": [{"sheet_row": 20, "values": values}],
        "raw_cells": {"structuredContent": {"properties": {"timeZone": "Europe/Moscow"}}}}}
    cells = {2: "+70000000001; +70000000002 (parent)"}
    if damage == "phone":
        cells[2] = "+70000000003"
    elif damage == "scope":
        event["scope"] = "child"
    elif damage == "time":
        event["at"] = "2026-08-28T17:49:55+03:00"
    elif damage == "text":
        event["full_text_parts"] = ["Incomplete"]
    elif damage == "id":
        event["event_id"] = "google_call:sheet:0:43"
    elif damage == "legacy_owner":
        old[5][18] = "p2"
    invoke = lambda: list(export.verified_supplemental_calls(manifest, {"passport_key": "p1", "number": 1}, cells, old))
    if damage:
        with pytest.raises(ValueError, match="supplemental call evidence mismatch"):
            invoke()
    else:
        assert invoke() == [event]
        assert list(export.verified_supplemental_calls(manifest, {"passport_key": "p2", "number": 2}, cells, old)) == []


@pytest.mark.parametrize("damage", [None, "event_digest", "evidence_ref", "evidence_digest", "summary_sha256", "direction", "scope", "foreign_event"])
def test_primary_review_can_correct_summary_not_identity(dataset, damage):
    source = export.collect(dataset)
    source["rows"] = source["rows"][:1]
    original = source["rows"][0]["history"][0]
    original["text_source"] = "increment_summary"
    original["full_text_parts"] = ["Two certificates were incorrectly described as two lessons."]
    payload = semantic(source)
    event = payload["rows"][0]["history"][0]
    ref = original["event_id"]
    corrected = "The parent will send two certificates. The number of missed lessons and credit amount were not specified."
    payload["event_summaries"][ref] = {"summary": corrected, "basis": "source_text_reviewed", "evidence_refs": [ref]}
    event["parts"] = [corrected]
    event["direction"] = "outbound"
    proof = source["rows"][0]["legacy_history"][0]
    patch = {"event_digest": export.stable_digest(original), "evidence_ref": proof["ref"],
             "evidence_digest": export.stable_digest(proof), "direction": "outbound",
             "summary_sha256": hashlib.sha256(corrected.encode()).hexdigest()}
    payload["source_corrections"] = {ref: patch}
    if damage in {"event_digest", "evidence_ref", "evidence_digest", "summary_sha256", "direction"}:
        patch[damage] = "unverified"
    elif damage == "scope":
        event["scope"] = "unresolved"
    elif damage == "foreign_event":
        payload["source_corrections"]["foreign"] = patch
    if damage:
        with pytest.raises(ValueError):
            export.validate_semantic(payload, source, export.stable_digest(source))
    else:
        export.validate_semantic(payload, source, export.stable_digest(source))
        assert original["full_text_parts"] != event["parts"]
        payload.pop("source_corrections")
        with pytest.raises(ValueError):
            export.validate_semantic(payload, source, export.stable_digest(source))
