from __future__ import annotations

import csv
import hashlib
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from mango_mvp.customer_timeline.wappi_history_import import (
    WappiChatResolution,
    WappiFetchLimits,
    WappiFetchStats,
    WappiHistoryClient,
    WappiPairCustomerResolver,
    WappiProfileSpec,
    assert_readonly_wappi_client,
    extract_chat_id,
    extract_wappi_items,
    fetch_chat_messages,
    open_readonly_sqlite,
)


HINT_SCHEMA_VERSION = "wappi_pending_hint_v1"
DECISION_VALUES = {"pending", "approve", "reject"}
REVIEW_COLUMNS = (
    "hint_id",
    "proposal_fingerprint",
    "profile_id",
    "chat_id",
    "channel",
    "brand",
    "pending_message_count",
    "previous_reason",
    "proposal_status",
    "proposed_customer_id",
    "proposed_lead_id",
    "proposed_contact_id",
    "match_key",
    "exact_match_kind",
    "single_active_lead",
    "organization_brand",
    "organization_value_count",
    "timeline_identity_sources",
    "evidence_complete",
    "rationale",
    "rationale_ru",
    "review_gate",
    "decision",
    "reviewer",
    "review_note",
)


@dataclass(frozen=True)
class PendingWappiChat:
    profile_id: str
    chat_id: str
    source_system: str
    brand: str
    pending_message_count: int
    previous_reason: str

    @property
    def channel(self) -> str:
        return "telegram" if self.source_system == "wappi_telegram" else "max"


def load_pending_wappi_chats(db_path: Path, *, tenant_id: str) -> tuple[PendingWappiChat, ...]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    with open_readonly_sqlite(db_path) as con:
        rows = con.execute(
            """
            SELECT record_json
            FROM timeline_conflicts
            WHERE tenant_id = ?
              AND conflict_type = 'pending_attribution'
              AND status = 'open'
              AND json_extract(record_json, '$.metadata.source_system') IN ('wappi_telegram', 'wappi_max')
            """,
            (tenant_id,),
        )
        for row in rows:
            payload = json.loads(str(row["record_json"] or "{}"))
            metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
            profile_id = str(metadata.get("profile_id") or "").strip()
            chat_id = str(metadata.get("chat_id") or "").strip()
            if not profile_id or not chat_id:
                continue
            key = (profile_id, chat_id)
            item = grouped.setdefault(
                key,
                {
                    "profile_id": profile_id,
                    "chat_id": chat_id,
                    "source_system": str(metadata.get("source_system") or "").strip(),
                    "brand": str(metadata.get("brand") or "").strip(),
                    "pending_message_count": 0,
                    "reasons": Counter(),
                },
            )
            if item["source_system"] != str(metadata.get("source_system") or "").strip() or item["brand"] != str(metadata.get("brand") or "").strip():
                raise ValueError(f"pending Wappi chat has inconsistent brand/source: profile_id={profile_id}")
            item["pending_message_count"] += 1
            item["reasons"][str(metadata.get("resolution_reason") or "unknown").strip()] += 1

    result: list[PendingWappiChat] = []
    for item in grouped.values():
        reasons: Counter[str] = item.pop("reasons")
        previous_reason = reasons.most_common(1)[0][0] if reasons else "unknown"
        result.append(PendingWappiChat(**item, previous_reason=previous_reason))
    return tuple(sorted(result, key=lambda item: (item.profile_id, item.chat_id)))


def collect_pending_dialogs(
    *,
    client: WappiHistoryClient,
    profiles: Sequence[WappiProfileSpec],
    pending_chats: Sequence[PendingWappiChat],
    page_size: int = 100,
    request_limit: int = 100,
    sleep_seconds: float = 0.2,
) -> tuple[dict[tuple[str, str], Mapping[str, Any]], Mapping[str, Any]]:
    assert_readonly_wappi_client(client)
    targets_by_profile: dict[str, set[str]] = {}
    for item in pending_chats:
        targets_by_profile.setdefault(item.profile_id, set()).add(item.chat_id)

    found: dict[tuple[str, str], Mapping[str, Any]] = {}
    requests = 0
    scanned = Counter()
    for profile in profiles:
        remaining = set(targets_by_profile.get(profile.profile_id, ()))
        offset = 0
        while remaining and requests < request_limit:
            payload = client.list_chats(
                channel=profile.channel,
                profile_id=profile.profile_id,
                limit=page_size,
                offset=offset,
                order="desc",
                show_all=False,
            )
            requests += 1
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            dialogs = extract_wappi_items(payload, "dialogs", "chats", "items", "data")
            if not dialogs:
                break
            scanned[profile.profile_id] += len(dialogs)
            for dialog in dialogs:
                chat_id = extract_chat_id(dialog)
                if chat_id in remaining:
                    found[(profile.profile_id, chat_id)] = dialog
                    remaining.remove(chat_id)
            if len(dialogs) < page_size:
                break
            offset += page_size
    return found, {
        "list_requests": requests,
        "dialogs_scanned_by_profile": dict(scanned),
        "pending_chats_found": len(found),
        "pending_chats_not_found": len(pending_chats) - len(found),
        "request_limit_hit": requests >= request_limit,
    }


def build_pending_hints(
    *,
    client: WappiHistoryClient,
    profiles: Sequence[WappiProfileSpec],
    resolver: WappiPairCustomerResolver,
    pending_chats: Sequence[PendingWappiChat],
    page_size: int = 100,
    list_request_limit: int = 100,
    messages_per_chat: int = 5,
    sleep_seconds: float = 0.2,
    amo_pause_seconds_per_call: float = 1.05,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
    profile_by_id = {profile.profile_id: profile for profile in profiles}
    dialogs, scan = collect_pending_dialogs(
        client=client,
        profiles=profiles,
        pending_chats=pending_chats,
        page_size=page_size,
        request_limit=list_request_limit,
        sleep_seconds=sleep_seconds,
    )
    rows: list[dict[str, Any]] = []
    message_requests = 0
    for pending in pending_chats:
        profile = profile_by_id.get(pending.profile_id)
        dialog = dialogs.get((pending.profile_id, pending.chat_id))
        if profile is None:
            resolution = WappiChatResolution(
                status="pending_attribution",
                expected_brand=pending.brand,
                reason="profile_not_configured",
            )
        elif pending.brand != profile.brand or pending.channel != profile.channel:
            resolution = WappiChatResolution(
                status="pending_attribution",
                expected_brand=profile.brand,
                reason="pending_profile_brand_or_channel_mismatch",
            )
        elif dialog is None:
            resolution = WappiChatResolution(
                status="pending_attribution",
                expected_brand=pending.brand,
                reason="chat_not_returned_by_wappi",
            )
        else:
            stats = WappiFetchStats()
            limits = WappiFetchLimits(
                chat_limit_per_profile=1,
                messages_per_chat=messages_per_chat,
                message_limit_total=messages_per_chat,
                request_limit_total=1,
                page_size=min(100, messages_per_chat),
                sleep_seconds=sleep_seconds,
                show_all_chats=False,
            )
            messages = fetch_chat_messages(
                client,
                profile=profile,
                chat_id=pending.chat_id,
                limits=limits,
                request_counter=stats,
                request_budget=limits.request_limit_total,
            )
            message_requests += stats.requests
            calls_before = resolver.amo_auto_calls
            resolution = resolver.resolve_chat(profile=profile, dialog=dialog, messages=messages)
            calls_delta = max(0, resolver.amo_auto_calls - calls_before)
            if calls_delta and amo_pause_seconds_per_call > 0:
                sleep(calls_delta * amo_pause_seconds_per_call)
        rows.append(_hint_row(pending, resolution))

    status_counts = Counter(row["proposal_status"] for row in rows)
    reason_counts = Counter(row["rationale"] for row in rows)
    return rows, {
        **scan,
        "message_requests": message_requests,
        "amo_read_calls": resolver.amo_auto_calls,
        "proposal_status_counts": dict(status_counts),
        "rationale_counts": dict(reason_counts),
        "writes": 0,
        "human_decisions_applied": 0,
    }


def write_hint_pack(
    out_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    *,
    review_limit: int = 50,
) -> Mapping[str, str]:
    if not 30 <= int(review_limit) <= 50:
        raise ValueError("Wappi human review sample must contain 30-50 rows when available")
    target = out_dir.expanduser().resolve(strict=False)
    if ".codex_local" not in target.parts:
        raise ValueError("Wappi hint pack contains identifiers and must stay under .codex_local")
    target.mkdir(parents=True, exist_ok=True)
    jsonl_path = target / "wappi_pending_hints.jsonl"
    review_path = target / "wappi_pending_hints_review.csv"
    sample_jsonl_path = target / "wappi_pending_review_sample.jsonl"
    sample_review_path = target / "wappi_pending_review_sample.csv"
    summary_path = target / "summary.json"
    jsonl_path.write_text("".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    _write_review_csv(review_path, rows)
    sample_rows = _frozen_review_sample(rows, limit=review_limit)
    sample_jsonl_path.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in sample_rows),
        encoding="utf-8",
    )
    _write_review_csv(sample_review_path, sample_rows)
    report = {
        "schema_version": HINT_SCHEMA_VERSION,
        "pending_chat_count": len(rows),
        "pending_set_sha256": hashlib.sha256(
            "\n".join(sorted(str(row["hint_id"]) for row in rows)).encode()
        ).hexdigest(),
        "proposal_set_sha256": hashlib.sha256(jsonl_path.read_bytes()).hexdigest(),
        "review_sample": {
            "rows": len(sample_rows),
            "proposed": sum(1 for row in sample_rows if row.get("proposal_status") == "proposed"),
            "sha256": hashlib.sha256(sample_jsonl_path.read_bytes()).hexdigest(),
        },
        "counts": dict(summary),
        "proposal_duplicate_counts": _proposal_duplicate_counts(rows),
        "policy": {
            "read_only": True,
            "automatic_binding": False,
            "chunk_visibility_changed": False,
            "accuracy_available": False,
            "accuracy_blocker": "human_decisions_required",
        },
        "files": {
            "hints_jsonl": str(jsonl_path),
            "review_csv": str(review_path),
            "review_sample_jsonl": str(sample_jsonl_path),
            "review_sample_csv": str(sample_review_path),
        },
    }
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "jsonl": str(jsonl_path),
        "review_csv": str(review_path),
        "review_sample_jsonl": str(sample_jsonl_path),
        "review_sample_csv": str(sample_review_path),
        "summary": str(summary_path),
    }


def _write_review_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "decision": "pending", "reviewer": "", "review_note": ""})


def _frozen_review_sample(rows: Sequence[Mapping[str, Any]], *, limit: int) -> list[Mapping[str, Any]]:
    size = max(0, min(int(limit), len(rows)))
    ordered = sorted(
        rows,
        key=lambda row: (
            0 if row.get("proposal_status") == "proposed" else 1,
            hashlib.sha256(str(row.get("hint_id") or "").encode()).hexdigest(),
        ),
    )
    return ordered[:size]


def validate_human_decisions(hints_jsonl: Path, decisions_csv: Path) -> Mapping[str, Any]:
    hints = {row["hint_id"]: row for row in _read_jsonl(hints_jsonl)}
    decisions: dict[str, Mapping[str, str]] = {}
    with decisions_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            hint_id = str(row.get("hint_id") or "").strip()
            if not hint_id or hint_id in decisions:
                raise ValueError("decision rows must have unique non-empty hint_id")
            decisions[hint_id] = row
    if set(decisions) != set(hints):
        raise ValueError("decision file must contain exactly the frozen hint set")

    counts = Counter()
    for hint_id, decision in decisions.items():
        hint = hints[hint_id]
        if str(decision.get("proposal_fingerprint") or "") != str(hint.get("proposal_fingerprint") or ""):
            raise ValueError(f"proposal fingerprint changed for hint_id={hint_id}")
        value = str(decision.get("decision") or "pending").strip().casefold()
        if value not in DECISION_VALUES:
            raise ValueError(f"unsupported decision={value!r} for hint_id={hint_id}")
        if value == "approve" and hint.get("proposal_status") != "proposed":
            raise ValueError(f"cannot approve a row without a proposal: hint_id={hint_id}")
        counts[value] += 1
    proposed_reviewed = 0
    approved_proposals = 0
    proposed_pending = 0
    for hint_id, decision in decisions.items():
        value = str(decision.get("decision") or "pending").strip().casefold()
        if hints[hint_id].get("proposal_status") == "proposed":
            proposed_pending += int(value == "pending")
            if value in {"approve", "reject"}:
                proposed_reviewed += 1
                approved_proposals += int(value == "approve")
    return {
        "total": len(hints),
        "counts": dict(counts),
        "review_complete": proposed_pending == 0,
        "proposed_pending": proposed_pending,
        "proposed_reviewed": proposed_reviewed,
        "approved_proposals": approved_proposals,
        "precision": (approved_proposals / proposed_reviewed) if proposed_reviewed else None,
        "binding_executed": False,
        "chunk_visibility_changed": False,
    }


def _hint_row(pending: PendingWappiChat, resolution: WappiChatResolution) -> dict[str, Any]:
    proposed = resolution.resolved
    basis = {
        "profile_id": pending.profile_id,
        "chat_id": pending.chat_id,
        "brand": pending.brand,
        "customer_id": resolution.customer_id or "",
        "lead_id": resolution.lead_id,
        "contact_id": resolution.contact_id,
        "match_key": resolution.match_key,
        "evidence": dict(resolution.evidence),
        "status": resolution.status,
        "reason": resolution.reason,
    }
    hint_id = hashlib.sha256(f"{pending.profile_id}:{pending.chat_id}".encode()).hexdigest()[:20]
    fingerprint = hashlib.sha256(json.dumps(basis, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    rationale = _rationale(resolution)
    return {
        "hint_id": hint_id,
        "proposal_fingerprint": fingerprint,
        "profile_id": pending.profile_id,
        "chat_id": pending.chat_id,
        "channel": pending.channel,
        "brand": pending.brand,
        "pending_message_count": pending.pending_message_count,
        "previous_reason": pending.previous_reason,
        "proposal_status": "proposed" if proposed else "no_proposal",
        "proposed_customer_id": resolution.customer_id or "",
        "proposed_lead_id": resolution.lead_id,
        "proposed_contact_id": resolution.contact_id,
        "match_key": resolution.match_key,
        "exact_match_kind": str(resolution.evidence.get("exact_match_kind") or resolution.match_key),
        "single_active_lead": bool(resolution.evidence.get("single_active_lead")),
        "organization_brand": str(resolution.evidence.get("organization_brand") or ""),
        "organization_value_count": int(resolution.evidence.get("organization_value_count") or 0),
        "timeline_identity_sources": " | ".join(str(item) for item in resolution.evidence.get("timeline_identity_sources") or ()),
        "evidence_complete": _evidence_complete(resolution),
        "rationale": rationale,
        "rationale_ru": _rationale_ru(resolution),
        "review_gate": (
            "recheck_current_amo_lead_state"
            if pending.previous_reason in {"closed_lead", "no_active_lead", "multi_active_lead", "max_phone_missing"}
            else "verify_existing_pair"
            if resolution.resolution_source == "draft_loop_pair"
            else "verify_proposal_evidence"
        ),
        "decision": "pending",
        "reviewer": "",
        "review_note": "",
    }


def _rationale(resolution: WappiChatResolution) -> str:
    if resolution.resolved:
        source = resolution.resolution_source or "resolver"
        if source == "draft_loop_pair":
            return f"existing_pair:{resolution.pair_source or 'unknown'}:current_amo_not_rechecked"
        match = resolution.match_key or "existing_pair"
        organization_brand = str(resolution.evidence.get("organization_brand") or "pair_brand")
        identity_sources = "+".join(str(item) for item in resolution.evidence.get("timeline_identity_sources") or ()) or "pair"
        return f"{source}:{match}:one_active_lead:org_{organization_brand}:timeline_{identity_sources}"
    return f"no_proposal:{resolution.reason or resolution.status or 'unresolved'}"


def _evidence_complete(resolution: WappiChatResolution) -> bool:
    return bool(
        resolution.resolved
        and resolution.evidence.get("single_active_lead")
        and resolution.evidence.get("organization_brand") in {"foton", "unpk"}
        and int(resolution.evidence.get("organization_value_count") or 0) == 1
        and resolution.evidence.get("timeline_identity_sources")
    )


def _rationale_ru(resolution: WappiChatResolution) -> str:
    if not resolution.resolved:
        return f"Предложение не сформировано: {resolution.reason or resolution.status or 'нет однозначного совпадения'}."
    if resolution.resolution_source == "draft_loop_pair":
        return "В текущем файле уже есть явная пара; её актуальность в AMO этим прогоном ещё не подтверждена."
    match = "Telegram ID" if "telegram" in resolution.match_key.casefold() else "телефону MAX"
    brand = "Фотон" if resolution.evidence.get("organization_brand") == "foton" else "УНПК"
    source_names = {
        "amo_contact_id": "контакту AMO",
        "amo_lead_id": "сделке AMO",
        "amo_opportunity": "карточке сделки в памяти",
    }
    sources = ", ".join(source_names.get(str(item), str(item)) for item in resolution.evidence.get("timeline_identity_sources") or ())
    return f"Точное совпадение по {match}; в AMO один активный лид; организация {brand}; клиент подтверждён по {sources}."


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError("hint jsonl rows must be objects")
            rows.append(dict(payload))
    return rows


def _proposal_duplicate_counts(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, int]:
    proposed = [row for row in rows if row.get("proposal_status") == "proposed"]
    result: dict[str, int] = {}
    for field_name in ("hint_id", "chat_id", "proposed_customer_id", "proposed_lead_id", "proposed_contact_id"):
        values = [str(row.get(field_name) or "") for row in proposed]
        result[field_name] = len(values) - len(set(values))
    return result
