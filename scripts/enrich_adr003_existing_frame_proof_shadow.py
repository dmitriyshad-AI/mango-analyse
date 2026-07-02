#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from mango_mvp.channels.subscription_llm_parts import (
    SEMANTIC_FRAME_EXISTENCE_PROOF_SHADOW_ENV,
    SEMANTIC_FRAME_PROOF_RECONCILIATION_SHADOW_ENV,
    SEMANTIC_FRAME_SELF_ANSWER_SHADOW_ENV,
    SubscriptionDraftResult,
    apply_semantic_frame_existence_proof_shadow,
    apply_semantic_frame_proof_reconciliation_shadow,
    apply_semantic_frame_self_answer_shadow,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL row: {exc}") from exc
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return ""


def _clean_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _clean_list(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return tuple(str(item) for item in value if str(item or "").strip())


def _active_brand(dialog: Mapping[str, Any], turn: Mapping[str, Any]) -> str:
    candidates = (
        turn.get("brand"),
        dialog.get("brand"),
        _clean_mapping(dialog.get("persona")).get("brand"),
    )
    for candidate in candidates:
        value = str(candidate or "").strip().casefold()
        if value in {"foton", "unpk"}:
            return value
    return ""


def _turn_key(dialog: Mapping[str, Any], turn: Mapping[str, Any]) -> str:
    return f"{dialog.get('dialog_id') or ''}#{turn.get('turn') or ''}"


def _posthoc_shadow(turn: Mapping[str, Any], direct: Mapping[str, Any]) -> dict[str, Any]:
    for raw in (turn.get("bot_semantic_frame_posthoc_shadow"), direct.get("semantic_frame_posthoc_shadow")):
        if isinstance(raw, Mapping) and raw:
            return dict(raw)
    # Existing-frame replay deliberately does not call the model; this marker lets
    # proof/readiness code treat the already stored frame as a valid posthoc frame.
    return {"status": "ok", "mode": "existing_frame_replay", "model_calls_added": 0}


def _metadata_from_turn(turn: Mapping[str, Any]) -> dict[str, Any]:
    direct = _clean_mapping(turn.get("bot_direct_path"))
    metadata: dict[str, Any] = {
        "direct_path": dict(direct),
        "direct_path_model_p0": _clean_mapping(direct.get("model_p0") or turn.get("bot_direct_path_model_p0")),
        "direct_path_model_intent": _clean_mapping(turn.get("bot_model_intent")),
        "conversation_intent_plan": _clean_mapping(turn.get("bot_conversation_intent_plan")),
        "action_decision": _clean_mapping(turn.get("bot_action_decision")),
        "close_detect": _clean_mapping(turn.get("bot_close_detect")),
        "authoritative_output_gate": _clean_mapping(turn.get("bot_authoritative_output_gate")),
        "semantic_output_verifier": _clean_mapping(turn.get("bot_semantic_output_verifier")),
        "reason_class": str(turn.get("bot_reason_class") or ""),
        "semantic_frame_replay": {
            "mode": "existing_frame_proof_shadow_replay",
            "no_llm": True,
        },
    }
    frame = _clean_mapping(turn.get("bot_semantic_frame"))
    if frame:
        posthoc = _posthoc_shadow(turn, direct)
        metadata["semantic_frame"] = dict(frame)
        metadata["semantic_frame_shadow"] = dict(frame)
        metadata["semantic_frame_posthoc_shadow"] = dict(posthoc)
        metadata["direct_path"]["semantic_frame"] = dict(frame)
        metadata["direct_path"]["semantic_frame_shadow"] = dict(frame)
        metadata["direct_path"]["semantic_frame_posthoc_shadow"] = dict(posthoc)
    return metadata


def _result_from_turn(turn: Mapping[str, Any]) -> SubscriptionDraftResult:
    return SubscriptionDraftResult(
        route=str(turn.get("bot_route") or "draft_for_manager"),
        draft_text=str(turn.get("bot_text") or ""),
        safety_flags=_clean_list(turn.get("bot_safety_flags")),
        manager_checklist=_clean_list(turn.get("bot_manager_checklist")),
        missing_facts=_clean_list(turn.get("bot_missing_facts")),
        forbidden_promises_detected=_clean_list(turn.get("bot_forbidden_promises_detected")),
        topic_id=str(turn.get("bot_topic_id") or "unknown"),
        message_type=str(turn.get("bot_message_type") or "question"),
        risk_level=str(turn.get("bot_risk_level") or "low"),
        metadata=_metadata_from_turn(turn),
    )


def _context(*, brand: str, snapshot_path: Path) -> dict[str, str]:
    return {
        "active_brand": brand,
        "brand": brand,
        "snapshot_path": str(snapshot_path),
        "knowledge_snapshot_path": str(snapshot_path),
        "kb_snapshot_path": str(snapshot_path),
        SEMANTIC_FRAME_EXISTENCE_PROOF_SHADOW_ENV: "1",
        SEMANTIC_FRAME_PROOF_RECONCILIATION_SHADOW_ENV: "1",
        SEMANTIC_FRAME_SELF_ANSWER_SHADOW_ENV: "1",
        "semantic_frame_existence_proof_shadow": "1",
        "semantic_frame_proof_reconciliation_shadow": "1",
        "semantic_frame_self_answer_shadow": "1",
        "semantic_frame_posthoc_shadow": "0",
    }


def _protected_turn_fields(turn: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "bot_route": turn.get("bot_route"),
        "bot_text": turn.get("bot_text"),
        "bot_safety_flags": turn.get("bot_safety_flags"),
        "bot_manager_checklist": turn.get("bot_manager_checklist"),
        "bot_missing_facts": turn.get("bot_missing_facts"),
        "judge_result": turn.get("judge_result"),
    }


def _enrich_turn(
    dialog: Mapping[str, Any],
    turn: Mapping[str, Any],
    *,
    snapshot_path: Path,
    counters: Counter[str],
    examples: list[dict[str, Any]],
) -> dict[str, Any]:
    enriched = dict(turn)
    counters["turns_total"] += 1
    route_before = str(turn.get("bot_route") or "")
    text_before = str(turn.get("bot_text") or "")
    protected_before = _protected_turn_fields(turn)
    frame = _clean_mapping(turn.get("bot_semantic_frame"))
    if not frame:
        counters["turns_missing_frame"] += 1
        enriched["existing_frame_proof_shadow_replayed"] = False
        enriched["existing_frame_proof_shadow_skip_reason"] = "no_bot_semantic_frame"
        return enriched

    counters["turns_with_frame"] += 1
    brand = _active_brand(dialog, turn)
    if not brand:
        counters["turns_unknown_brand"] += 1

    result = _result_from_turn(turn)
    ctx = _context(brand=brand, snapshot_path=snapshot_path)
    result = apply_semantic_frame_existence_proof_shadow(result, context=ctx)
    result = apply_semantic_frame_proof_reconciliation_shadow(result, context=ctx)
    result = apply_semantic_frame_self_answer_shadow(result, context=ctx)

    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    direct = metadata.get("direct_path") if isinstance(metadata.get("direct_path"), Mapping) else {}
    proof = metadata.get("semantic_frame_existence_proof_shadow")
    if not isinstance(proof, Mapping):
        proof = direct.get("semantic_frame_existence_proof_shadow") if isinstance(direct, Mapping) else {}
    reconciliation = metadata.get("semantic_frame_proof_reconciliation_shadow")
    if not isinstance(reconciliation, Mapping):
        reconciliation = direct.get("semantic_frame_proof_reconciliation_shadow") if isinstance(direct, Mapping) else {}
    self_shadow = metadata.get("semantic_frame_self_answer_shadow")
    if not isinstance(self_shadow, Mapping):
        self_shadow = direct.get("semantic_frame_self_answer_shadow") if isinstance(direct, Mapping) else {}

    if isinstance(proof, Mapping) and proof:
        enriched["bot_semantic_frame_existence_proof_shadow"] = dict(proof)
        counters[f"existence_proof_status:{proof.get('status') or ''}"] += 1
        counters[f"existence_proof_reason:{proof.get('reason') or ''}"] += 1
    if isinstance(reconciliation, Mapping) and reconciliation:
        enriched["bot_semantic_frame_proof_reconciliation_shadow"] = dict(reconciliation)
        counters[f"proof_reconciliation_status:{reconciliation.get('status') or ''}"] += 1
        counters[f"proof_reconciliation_reason:{reconciliation.get('reason') or ''}"] += 1
        if reconciliation.get("status") == "would_reconcile_to_safe_reference":
            counters["proof_reconciliation_would_reconcile_to_safe_reference"] += 1
            if len(examples) < 20:
                examples.append(
                    {
                        "dialog_id": str(dialog.get("dialog_id") or ""),
                        "turn": turn.get("turn"),
                        "brand": brand,
                        "route": route_before,
                        "client_message": str(turn.get("client_message") or "")[:240],
                        "proof_status": str(proof.get("status") or "") if isinstance(proof, Mapping) else "",
                        "source_fact_key": str(reconciliation.get("source_fact_key") or ""),
                        "frame_requested_action": str(frame.get("requested_action") or ""),
                        "frame_risk_class": str(frame.get("risk_class") or ""),
                        "frame_answerability": str(frame.get("answerability") or ""),
                        "active_blockers": list(reconciliation.get("active_blockers") or []),
                    }
                )
    if isinstance(self_shadow, Mapping) and self_shadow:
        enriched["bot_semantic_frame_self_answer_shadow"] = dict(self_shadow)
        counters[f"self_answer_shadow_status:{self_shadow.get('status') or ''}"] += 1
        counters[f"self_answer_shadow_reason:{self_shadow.get('reason') or ''}"] += 1
        if self_shadow.get("status") == "would_demote_to_self":
            counters["self_answer_shadow_would_demote_to_self"] += 1

    if isinstance(direct, Mapping) and direct:
        enriched["bot_direct_path"] = dict(direct)
    enriched["existing_frame_proof_shadow_replayed"] = True
    enriched["existing_frame_proof_shadow_model_calls_added"] = 0

    if str(enriched.get("bot_route") or "") != route_before or str(enriched.get("bot_text") or "") != text_before:
        counters["route_text_diff_count"] += 1
    if _protected_turn_fields(enriched) != protected_before:
        counters["protected_turn_field_diff_count"] += 1
    return enriched


def enrich_dialogs(
    dialogs: Sequence[Mapping[str, Any]],
    *,
    snapshot_path: Path,
    input_source_rev: str = "",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    counters: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    enriched_dialogs: list[dict[str, Any]] = []
    for dialog in dialogs:
        payload = dict(dialog)
        turns = dialog.get("turns") if isinstance(dialog.get("turns"), list) else []
        payload["turns"] = [
            _enrich_turn(dialog, turn, snapshot_path=snapshot_path, counters=counters, examples=examples)
            if isinstance(turn, Mapping)
            else turn
            for turn in turns
        ]
        payload["existing_frame_proof_shadow_replayed"] = True
        enriched_dialogs.append(payload)
        counters["dialogs_total"] += 1

    summary = {
        "schema_version": "adr003_existing_frame_proof_shadow_replay_v1",
        "script_source_rev": _git_rev(),
        "input_source_rev": str(input_source_rev or ""),
        "model_calls_added": 0,
        "route_text_diff_count": counters.get("route_text_diff_count", 0),
        "protected_turn_field_diff_count": counters.get("protected_turn_field_diff_count", 0),
        "counts": dict(sorted(counters.items())),
        "examples": examples,
    }
    return enriched_dialogs, summary


def _markdown(summary: Mapping[str, Any], *, transcripts: Path, snapshot: Path) -> str:
    counts = summary.get("counts") if isinstance(summary.get("counts"), Mapping) else {}
    lines = [
        "# ADR003 Existing-Frame Proof Shadow Replay",
        "",
        f"- script_source_rev: `{summary.get('script_source_rev') or ''}`",
        f"- input_source_rev: `{summary.get('input_source_rev') or ''}`",
        f"- transcripts: `{transcripts}`",
        f"- kb_snapshot: `{snapshot}`",
        "- model_calls_added: `0`",
        f"- route_text_diff_count: `{summary.get('route_text_diff_count', 0)}`",
        f"- protected_turn_field_diff_count: `{summary.get('protected_turn_field_diff_count', 0)}`",
        f"- dialogs: `{counts.get('dialogs_total', 0)}`",
        f"- turns_total: `{counts.get('turns_total', 0)}`",
        f"- turns_with_frame: `{counts.get('turns_with_frame', 0)}`",
        f"- proof_reconciliation_would_reconcile_to_safe_reference: `{counts.get('proof_reconciliation_would_reconcile_to_safe_reference', 0)}`",
        f"- self_answer_shadow_would_demote_to_self: `{counts.get('self_answer_shadow_would_demote_to_self', 0)}`",
        "",
        "## Reconciliation Status",
    ]
    for key, value in sorted(counts.items()):
        if key.startswith("proof_reconciliation_status:"):
            lines.append(f"- `{key.removeprefix('proof_reconciliation_status:') or '<empty>'}`: `{value}`")
    lines.append("")
    lines.append("## Existence Proof Reasons")
    for key, value in sorted(counts.items()):
        if key.startswith("existence_proof_reason:"):
            lines.append(f"- `{key.removeprefix('existence_proof_reason:') or '<empty>'}`: `{value}`")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Replay ADR-003 proof/self-answer shadow layers over saved transcripts using existing "
            "bot_semantic_frame values. Does not call LLM or change route/text."
        )
    )
    parser.add_argument("--transcripts", required=True, type=Path)
    parser.add_argument("--kb-snapshot", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--input-source-rev", default="", help="Source revision of the saved transcripts, if known.")
    args = parser.parse_args(argv)

    transcripts = args.transcripts.expanduser().resolve()
    snapshot = args.kb_snapshot.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    if not transcripts.exists():
        raise SystemExit(f"transcripts not found: {transcripts}")
    if not snapshot.exists():
        raise SystemExit(f"kb snapshot not found: {snapshot}")
    if any(part == "stable_runtime" for part in out_dir.parts):
        raise SystemExit(f"refusing to write replay artifacts under stable_runtime: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    dialogs = _read_jsonl(transcripts)
    enriched, summary = enrich_dialogs(dialogs, snapshot_path=snapshot, input_source_rev=args.input_source_rev)
    summary = {
        **summary,
        "inputs": {
            "transcripts": str(transcripts),
            "transcripts_sha256": _sha256(transcripts),
            "kb_snapshot": str(snapshot),
            "kb_snapshot_sha256": _sha256(snapshot),
        },
    }
    out_transcripts = out_dir / "dynamic_dialog_transcripts.jsonl"
    _write_jsonl(out_transcripts, enriched)
    _write_json(out_dir / "existing_frame_proof_shadow_summary.json", summary)
    (out_dir / "existing_frame_proof_shadow_summary.md").write_text(
        _markdown(summary, transcripts=transcripts, snapshot=snapshot),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
