"""Local passport export from an owner-approved, frozen, targeted source slice."""
import argparse
import hashlib
import json
import re
import tempfile
import time
from datetime import datetime
from pathlib import Path

from email_pipeline.summary import split_thread_context
from mango_mvp.customer_timeline.canonical_readonly_import import parse_datetime_guess
from mango_mvp.customer_timeline.ids import stable_digest
from mango_mvp.utils.phone import normalize_phone
from mango_mvp.customer_timeline.safety import guard_customer_timeline_output_path
from mango_mvp.customer_timeline.store import (
    CustomerTimelineSQLiteStore, has_open_family_identity_conflict,
    trusted_family_customer_ids_by_customer,
)


def read_json(path):
    return json.loads(Path(path).read_text())


def grid(path):
    data = read_json(path)["structuredContent"]
    if "values" in data:
        match = re.search(r"!?([A-Z]+)(\d+):", data["range"])
        column = 0
        for letter in match[1]:
            column = column * 26 + ord(letter) - 64
        return {int(match[2]) - 1 + i: {column - 1 + j: v for j, v in enumerate(row)}
                for i, row in enumerate(data["values"])}
    result = {}
    for block in data["sheets"][0]["data"]:
        for i, row in enumerate(block.get("rowData", []), block.get("startRow", 0)):
            for j, cell in enumerate(row.get("values", []), block.get("startColumn", 0)):
                result.setdefault(i, {})[j] = cell.get("formattedValue", next(iter(cell.get("userEnteredValue", {}).values()), ""))
    return result


def parts(text, limit=20000):
    if limit < 1 or limit > 20000:
        raise ValueError("invalid cell budget")
    result = []
    while len(text) > limit:
        boundaries = [m.end() for m in re.finditer(r"\s+", text[:limit])]
        if not boundaries:
            raise ValueError("unbreakable text exceeds cell budget")
        boundary = boundaries[-1]
        result.append(text[:boundary])
        text = text[boundary:]
    return result + ([text] if text else [])


# SAFETY BEGIN: identity is evidence, never a shared phone/name guess.
def attribution(event, attr, owners, children, conflict):
    meta = event.get("metadata", {})
    record = event.get("record", {})
    blocked = conflict or meta.get("pending_attribution") or len(owners) != 1
    brands = {str(b or "").lower() for b in
              (meta.get("brand"), record.get("brand"), record.get("payload", {}).get("brand"))}
    if blocked or event.get("match_status") not in {"strong_unique", "manual"}:
        return "unresolved"
    if brands - {"foton", "unknown", ""} or meta.get("brand_context_authorized") is False:
        return "unresolved"
    if (len(children) == 1 and attr.get("customer_id") == event.get("customer_id") == owners[0]
            and attr.get("status") == "matched" and attr.get("confidence") in {"high", "medium"}
            and children[0]["status"] == "confident" and children[0]["confidence"] in {"high", "medium"}):
        return "child" if attr.get("child_key") == children[0]["child_key"] else "other_child"
    return "family_context"


def call_matches(event, call):
    try:
        def minute(value):
            return parse_datetime_guess(value).replace(second=0, microsecond=0)
        return (str(call["canonical_call_id"]) == event["source_id"]
                and event.get("source_ref") == "mango:" + event["source_id"]
                and minute(call["started_at"]) == minute(event["event_at"]))
    except (KeyError, ValueError, TypeError, AttributeError):
        return False


def increment_summary(event):
    call = event.get("record", {}).get("call", {})
    if (event.get("source_system") == "mango_processed_summary" and parse_datetime_guess(event.get("event_at"))
            and call.get("call_id") == event.get("source_id") and call.get("source_ref") == event.get("source_ref")
            and parse_datetime_guess(call.get("event_at")) == parse_datetime_guess(event.get("event_at"))):
        return call.get("analysis_json", {}).get("history_summary") or call.get("analysis_summary") or ""
    return ""


def all_events(store, tenant, customer):
    if not store.read_only:
        raise ValueError("writable store is forbidden")
    cursor = None
    while True:
        page = store.list_events_by_customer(tenant, customer, sort="asc", limit=500, cursor=cursor)
        yield from page["items"]
        cursor = page["next_cursor"]
        if cursor is None:
            break


def verified_extra(con, proof, tenant, student):
    if proof["tallanto_id"] != student:
        return None
    link = con.execute("SELECT * FROM identity_links WHERE link_id=? AND tenant_id=?", (proof["link_id"], tenant)).fetchone()
    row = con.execute("SELECT * FROM timeline_events WHERE event_id=? AND tenant_id=?", (proof["event_id"], tenant)).fetchone()
    if not link or not row or link["link_type"] != "tallanto_student_id" or link["link_value"] != student:
        raise ValueError("unproven extra event")
    evidence = json.loads(link["record_json"]).get("evidence", {})
    if (row["source_system"] != link["source_system"] or row["source_ref"] != link["source_ref"]
            or row["source_id"] != (evidence.get("message_sha256") or evidence.get("source_id"))):
        raise ValueError("extra event evidence mismatch")
    return row


def validate_semantic(payload, source, digest):
    if payload.get("as_of") != source["as_of"] or payload.get("source_digest") != digest:
        raise ValueError("stale semantic payload")
    expected = {r["passport_key"]: r for r in source["rows"]}
    rows = payload.get("rows", [])
    if len(rows) != len(expected) or {r["passport_key"] for r in rows} != set(expected):
        raise ValueError("semantic passport set mismatch")
    if payload.get("review_status") != "claude_codex_reviewed":
        raise ValueError("semantic review missing")
    for row in rows:
        if any(not isinstance(row.get(k), str) or not row[k].strip() for k in
               ("manager_summary", "learning_profile", "situation_risks", "next_action", "quality_note")):
            raise ValueError("incomplete semantic fields")
        passport = expected[row["passport_key"]]
        refs = {e["event_id"] for e in passport["history"] if e["scope"] != "other_child"}
        refs.update(e["ref"] for e in passport["legacy_history"])
        if not row.get("evidence_refs") or not set(row["evidence_refs"]).issubset(refs | {"roster"}):
            raise ValueError("semantic evidence outside passport")
        if refs and not set(row["evidence_refs"]).intersection(refs):
            raise ValueError("semantic ignores available history")
    validate_projection(payload, source)


def validate_projection(payload, source):
    events = {e["event_id"]: e for r in source["rows"] for e in r["history"]}
    summaries = payload.get("event_summaries", {})
    corrections = payload.get("source_corrections", {})
    if set(corrections) - set(events):
        raise ValueError("source correction outside active events")
    if set(summaries) - set(events):
        raise ValueError("event summary outside active sources")
    for event_id, note in summaries.items():
        if (not isinstance(note.get("summary"), str) or not note["summary"].strip()
                or note.get("evidence_refs") != [event_id]
                or note.get("basis") not in {"cache_verified_against_source", "source_text_reviewed", "source_metadata_only"}):
            raise ValueError("unreviewed event summary")
    for base in source["rows"]:
        row = next(r for r in payload["rows"] if r["passport_key"] == base["passport_key"])
        for key in ("name", "phones", "email", "tallanto_id", "group", "academic_year", "group_status", "group_verified_at",
                    "identity_scope_status", "identity_conflict", "legacy_quarantine_warning", "family_scope_complete"):
            if row.get(key) != base[key]:
                raise ValueError("projection roster mismatch")
        history = row.get("history", [])
        expected = {e["event_id"]: e for e in base["history"]}
        if len(history) != len(expected) or {e["event_id"] for e in history} != set(expected):
            raise ValueError("incomplete projected history")
        for e in history:
            original = expected[e["event_id"]]
            correction = corrections.get(e["event_id"], {})
            proofs = {item["ref"]: item for item in base["legacy_history"]}
            proofs[e["event_id"]] = original
            if correction and (
                    set(correction) - {"event_digest", "evidence_ref", "evidence_digest", "summary_sha256", "direction"}
                    or correction.get("event_digest") != stable_digest(original)
                    or correction.get("evidence_ref") not in proofs
                    or correction.get("evidence_digest") != stable_digest(proofs[correction["evidence_ref"]])
                    or ("direction" in correction and correction["direction"] not in {"inbound", "outbound", "unknown"})
                    or ("summary_sha256" in correction and correction["summary_sha256"] != hashlib.sha256(
                        summaries.get(e["event_id"], {}).get("summary", "").encode()).hexdigest())):
                raise ValueError("unbound source correction")
            if e.get("direction") != correction.get("direction", original["direction"]):
                raise ValueError("unreviewed direction change")
            if any(e.get(k) != original[k] for k in ("at", "source", "scope", "author",
                                                    "text_source", "canonical_summary_verified")):
                raise ValueError("projection changes event identity or scope")
            if original["text_source"] in {"canonical_summary", "increment_summary"} and "summary_sha256" not in correction:
                text = "".join(original["full_text_parts"])
            elif e["event_id"] in summaries:
                text = summaries[e["event_id"]]["summary"]
            elif (original["source"] in {"mail_archive_stage2", "mango_processed_summary", "google_calls_sheet"}
                  or original["raw"].get("record", {}).get("action_text")
                  or not "".join(original["full_text_parts"])):
                raise ValueError("finished event summary missing")
            else:
                text = "".join(original["full_text_parts"])
            if e.get("parts") != parts(text):
                raise ValueError("projected text lost or changed")
        if row.get("legacy_source_refs") != [e["ref"] for e in base["legacy_history"]]:
            raise ValueError("legacy source coverage changed")


def validate_roster(selections, seeds):
    for key in ("number", "passport_key", "tallanto_id"):
        if len({r[key] for r in selections}) != len(selections):
            raise ValueError("duplicate roster key")
    for row in selections:
        cells = seeds[row["number"] + 4]
        if cells.get(21) != row["tallanto_id"] or cells.get(22) != row["passport_key"]:
            raise ValueError("roster identity mismatch")


def verified_supplemental_calls(manifest, seed, cells, old):
    evidence = manifest.get("supplemental_call_evidence", {})
    for event in manifest.get("supplemental_calls", []):
        if event["passport_key"] != seed["passport_key"]:
            continue
        record = next((r for r in evidence.get("records", []) if r["sheet_row"] == event["sheet_row"]), None)
        if not record or evidence["raw_cells"]["structuredContent"]["properties"]["timeZone"] != "Europe/Moscow":
            raise ValueError("supplement lacks primary call evidence")
        values = record["values"]
        phones = {normalize_phone(p) for p in str(cells.get(2, "")).split(";")}
        legacy = old[int(event["legacy_ref"].split(":")[1]) - 1]
        # ponytail: this supplement is the owner-verified Moscow 2026 sheet, not a timezone importer.
        at = datetime.fromisoformat(values[1]).isoformat() + "+03:00"
        identity = f"google_call:{evidence['spreadsheet_id']}:{evidence['sheet_id']}:{values[0]}"
        if (not normalize_phone(values[6]) or normalize_phone(values[6]) not in phones
                or event["event_id"] != identity or event["at"] != at
                or event["scope"] != "family_context" or event["source"] != "google_calls_sheet"
                or legacy.get(18) != seed["passport_key"] or str(legacy.get(0)) != str(seed["number"])
                or legacy.get(2) != datetime.fromisoformat(values[1]).strftime("%d.%m.%Y %H:%M")
                or legacy.get(13) != values[15] or event["full_text_parts"] != parts(values[15])
                or event["raw"]["record"]["values"] != values or event["author"] != values[2]
                or event["direction"] != values[3] or event["canonical_summary_verified"]
                or event["text_source"] != "google_full_transcript"):
            raise ValueError("supplemental call evidence mismatch")
        yield event
# SAFETY END


def collect(root):
    manifest = read_json(root / "R5_SLICE_MANIFEST.json")
    seeds, old = grid(root / "GOOGLE_BEFORE.json"), grid(root / "HISTORY_BEFORE.json")
    validate_roster(manifest["selections"], seeds)
    fulltext = grid(root / "HISTORY_FULLTEXT_BEFORE.json")
    if set(fulltext) - set(old):
        raise ValueError("fulltext range does not match history rows")
    for number, cells in fulltext.items():
        old.setdefault(number, {}).update(cells)
    calls = read_json(root / "CANONICAL_CALLS_TARGETED.json")
    source = {"as_of": manifest["as_of"], "manifest": manifest, "calls": calls, "rows": []}
    matched_legacy = set()
    tenant = manifest["tenant_id"]
    with CustomerTimelineSQLiteStore.open_read_only(root / "r5_13_slice.sqlite") as store:
        con = store._con
        has_attribution = con.execute("SELECT 1 FROM sqlite_master WHERE name='event_child_attribution_v1'").fetchone()
        for seed in manifest["selections"]:
            cells = seeds[seed["number"] + 4]
            links = [dict(r) for r in con.execute(
                "SELECT * FROM identity_links WHERE tenant_id=? AND link_type='tallanto_student_id' AND link_value=?",
                (tenant, seed["tallanto_id"]))]
            owners = sorted({r["customer_id"] for r in links if r["customer_id"] and r["match_class"] in {"strong_unique", "manual"}})
            scopes = trusted_family_customer_ids_by_customer(con, tenant_id=tenant, customer_ids=owners)
            members = sorted({m for scope in scopes.values() for m in scope})
            family = [dict(r) for m in members for r in con.execute("SELECT * FROM family_links_v1 WHERE tenant_id=? AND customer_id=?", (tenant, m))]
            children = [r for r in family if r["customer_id"] in owners
                        and seed["tallanto_id"] in json.loads(r["record_json"]).get("tallanto_student_ids", [])]
            conflict = has_open_family_identity_conflict(con, tenant, family_id="", customer_ids=members)
            quarantine = "\u041a\u0410\u0420\u0410\u041d\u0422\u0418\u041d" in str(cells.get(20, "")).upper()
            events = {e["event_id"]: e for m in members for e in all_events(store, tenant, m)}
            inactive = {r["event_id"]: dict(r) for m in members for r in con.execute(
                "SELECT * FROM timeline_events WHERE tenant_id=? AND customer_id=? AND superseded_by IS NOT NULL", (tenant, m))}
            for proof in manifest["extra_evidence"]:
                extra = verified_extra(con, proof, tenant, seed["tallanto_id"])
                if extra is not None:
                    if extra["superseded_by"] is not None:
                        inactive[extra["event_id"]] = dict(extra)
                    else:
                        events[extra["event_id"]] = dict(json.loads(extra["record_json"]), customer_id=extra["customer_id"])
            history = []
            for e in sorted(events.values(), key=lambda e: (e["event_at"], e["event_id"])):
                a = con.execute("SELECT * FROM event_child_attribution_v1 WHERE tenant_id=? AND event_id=?",
                                (tenant, e["event_id"])).fetchone() if has_attribution else None
                attr, record = dict(a) if a else {}, e.get("record", {})
                call = calls["rows"].get(e["source_id"], {}) if e["source_system"] == "mango_processed_summary" else {}
                canonical_ok = call_matches(e, call)
                summary = json.loads(call["analysis_json"]).get("history_summary", "") if canonical_ok else ""
                text_source, text = next(((kind, value) for kind, value in (
                    ("canonical_summary", summary), ("increment_summary", increment_summary(e)),
                    ("full_clean_text", record.get("full_clean_text")), ("message_text", record.get("message", {}).get("text")),
                    ("record_summary", record.get("summary")), ("payload_summary", record.get("payload", {}).get("summary")),
                    ("summary", e.get("summary")), ("text_preview", e.get("text_preview"))) if value), ("none", ""))
                sha = record.get("message_sha256")
                cache = con.execute("SELECT * FROM email_summary_cache_v1 WHERE message_sha256=?", (sha,)).fetchone() if sha else None
                head, context = split_thread_context(text) if e["source_system"] == "mail_archive_stage2" else (text, "")
                scope = attribution(e, attr, owners, children, conflict) if e.get("customer_id") in members else "unresolved"
                history.append({
                    "event_id": e["event_id"], "at": e["event_at"], "source": e["source_system"],
                    "author": e.get("actor_name"), "direction": e["direction"], "scope": scope,
                    "parts": parts(head), "context_parts": parts(context), "full_text_parts": parts(text),
                    "canonical_summary_verified": bool(summary), "text_source": text_source,
                    "email_cache": dict(cache, llm_derived=True) if cache else None,
                    "attribution": attr, "raw": e,
                })
            history.extend(verified_supplemental_calls(manifest, seed, cells, old))
            history.sort(key=lambda e: (e["at"], e["event_id"]))
            legacy = [{"ref": "sheet:" + str(i + 1), "scope": "unresolved_legacy", "cells": row,
                       "full_text_parts": parts(str(row.get(13, "")))}
                      for i, row in sorted(old.items())
                      if str(row.get(0)) == str(seed["number"]) and row.get(18) == seed["passport_key"]]
            matched_legacy.update(e["ref"] for e in legacy)
            source["rows"].append({
                "number": seed["number"], "passport_key": seed["passport_key"], "tallanto_id": seed["tallanto_id"],
                "name": cells[1], "phones": cells.get(2, ""), "email": cells.get(3, ""), "group": seeds[1][1],
                "academic_year": "2026/27", "group_status": cells.get(7, ""),
                "group_verified_at": manifest["source_dates"]["roster_contacts"],
                "identity_links": links, "owners": owners, "family_members": members, "family_links": family,
                "identity_scope_status": "blocked" if conflict else "owner_found" if len(owners) == 1 else "unresolved",
                "family_scope_complete": False, "identity_conflict": conflict, "legacy_quarantine_warning": quarantine,
                "history": history, "legacy_history": legacy,
                "raw_inactive": [{"active": False, "source": r} for r in inactive.values()], "raw_before": cells,
            })
    source["unassigned_legacy_rows"] = {i + 1: r for i, r in old.items() if "sheet:" + str(i + 1) not in matched_legacy}
    return json.loads(json.dumps(source, ensure_ascii=False))


# SAFETY BEGIN: fixed source binding, no live writes or stale partial releases.
def run(root, output, semantic=None, *, allowed_root=None):
    started = time.perf_counter()
    root = Path(root).resolve()
    allowed = allowed_root or Path(__file__).resolve().parents[1] / ".codex_local"
    output = guard_customer_timeline_output_path(output, allowed)
    files = [root / name for name in (
        "R5_SLICE_MANIFEST.json", "GOOGLE_BEFORE.json", "HISTORY_BEFORE.json",
        "HISTORY_FULLTEXT_BEFORE.json", "CANONICAL_CALLS_TARGETED.json", "r5_13_slice.sqlite")]
    if output in [p.resolve() for p in files] or (semantic and output == Path(semantic).resolve()):
        raise ValueError("source overwrite forbidden")
    stamp = lambda: [(str(p), p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ino,
                      hashlib.sha256(p.read_bytes()).hexdigest() if p.suffix == ".json" else None) for p in files]
    before = stamp()
    hot = lambda: any(p.exists() and p.stat().st_size for p in
                      (Path(str(files[-1]) + suffix) for suffix in ("-wal", "-journal")))
    if hot():
        raise ValueError("source must be frozen without WAL/journal")
    manifest = read_json(files[0])
    if datetime.fromisoformat(manifest["as_of"]).utcoffset() is None:
        raise ValueError("as_of must be timezone-aware")
    if hashlib.sha256(files[-1].read_bytes()).hexdigest() != manifest["slice_sha256"]:
        raise ValueError("slice differs from frozen manifest")
    signature = stable_digest({"inputs": before, "code": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    previous = read_json(output) if output.exists() else {}
    reuse = previous.get("input_signature") == signature and stable_digest(previous.get("source", {})) == previous.get("source_digest")
    source = previous["source"] if reuse else collect(root)
    digest = stable_digest(source)
    payload = read_json(semantic) if semantic else None
    if payload is None and previous.get("status") == "reviewed_local":
        raise ValueError("refusing to downgrade an accepted export")
    if payload is not None:
        validate_semantic(payload, source, digest)
    result = {"input_signature": signature, "source_digest": digest, "source": source,
              "semantic_payload": payload, "status": "reviewed_local" if payload else "source_draft"}
    result["rows"] = payload["rows"] if payload else []
    if stamp() != before or hot():
        raise ValueError("source changed during export")
    encoded = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists() or output.read_text() != encoded:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent, delete=False) as temporary:
            temporary.write(encoded)
        Path(temporary.name).replace(output)
    return {"mode": "incremental" if reuse else "full", "passports": len(source["rows"]),
            "reused": len(source["rows"]) if reuse else 0, "source_digest": digest,
            "seconds": round(time.perf_counter() - started, 4)}
# SAFETY END


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--semantic", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.inputs, args.output, args.semantic)))
