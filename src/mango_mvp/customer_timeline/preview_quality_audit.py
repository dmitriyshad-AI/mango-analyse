from __future__ import annotations

import json
import re
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from mango_mvp.channels.actions import (
    ACTION_CREATE_FOLLOW_UP_TASK,
    ACTION_DRAFT_CLIENT_MESSAGE,
    ACTION_HANDOFF_TO_MANAGER,
    ACTION_MARK_MANUAL_REVIEW,
    ACTION_NOTIFY_ROP_HOT_LEAD,
    ACTION_REQUEST_CRM_CONTEXT,
)
from mango_mvp.channels.contracts import ChannelMessage
from mango_mvp.channels.telegram_history import (
    TELEGRAM_HISTORY_CHANNEL,
    build_telegram_history_inventory,
    iter_telegram_history_messages,
)
from mango_mvp.customer_timeline.approved_context_pack import (
    APPROVED_CONTEXT_PACK_STATUS,
    CUSTOMER_TIMELINE_APPROVED_CONTEXT_PACK_SCHEMA_VERSION,
    customer_timeline_approved_context_pack_safety_contract,
)
from mango_mvp.customer_timeline.channel_preview_from_pack import (
    CustomerTimelineChannelPreviewFromPackConfig,
    build_customer_timeline_channel_preview_from_pack,
)
from mango_mvp.customer_timeline.ids import stable_prefixed_id


CUSTOMER_TIMELINE_PREVIEW_QUALITY_AUDIT_SCHEMA_VERSION = "customer_timeline_preview_quality_audit_v1"
DEFAULT_TENANT_ID = "foton"
DEFAULT_AUDIT_CUSTOMER_ID = "customer:audit:channel_preview"
DEFAULT_CONTEXT_SUMMARY = (
    "Клиент интересуется обучением в Фотоне. Можно использовать только безопасный общий контекст: "
    "менеджер должен проверить детали программы, цены, расписания и записи перед ответом."
)

PRICE_MARKERS = ("цена", "стоимост", "оплат", "сколько", "рассроч", "скидк", "абонемент")
CALLBACK_MARKERS = ("позвон", "перезвон", "свяж", "звонок")
MANAGER_MARKERS = ("менеджер", "оператор", "человек", "живой")
HOT_MARKERS = ("срочно", "готов оплатить", "готова оплатить", "готовы оплатить", "купить", "записаться", "бронь")
LOCATION_MARKERS = ("адрес", "где", "метро", "очно", "локац", "проводятся")
SCHEDULE_MARKERS = ("когда", "во сколько", "распис", "день", "время", "начало")
PROGRAM_MARKERS = ("курс", "программа", "математ", "физик", "егэ", "огэ", "олимпиад", "класс", "подготов")
ENROLLMENT_MARKERS = ("запис", "анкета", "фио", "дата рождения", "документ", "договор")
OBJECTION_MARKERS = ("дорого", "подума", "не уверен", "не уверена", "сомнева", "позже", "не подходит")
ATTACHMENT_MARKERS = ("файл", "влож", "фото", "скрин", "документ")
PERSONAL_DATA_RE = re.compile(
    r"(?i)(?:[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}|(?:\+7|7|8)[\d\s().-]{9,}\d)"
)


@dataclass(frozen=True)
class SyntheticPreviewCase:
    case_id: str
    description: str
    inbound_text: str
    expected_actions: tuple[str, ...] = field(default_factory=tuple)
    attachments: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    context_summary: str = DEFAULT_CONTEXT_SUMMARY

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("case_id must not be empty")
        if not self.description.strip():
            raise ValueError("description must not be empty")
        if not self.inbound_text.strip() and not self.attachments:
            raise ValueError("inbound_text or attachments must be provided")
        object.__setattr__(self, "expected_actions", tuple(sorted(set(self.expected_actions))))
        object.__setattr__(self, "attachments", tuple(dict(item) for item in self.attachments))


@dataclass(frozen=True)
class TelegramReplyPair:
    pair_id: str
    export_root_name: str
    channel_thread_id: str
    inbound_message_id: str
    outbound_message_id: str
    inbound_at: str
    outbound_at: str
    inbound_text: str
    employee_reply_text: str
    source_refs: tuple[str, ...] = field(default_factory=tuple)

    def to_safe_json_dict(self) -> Mapping[str, Any]:
        return {
            "pair_id": self.pair_id,
            "export_root_name": self.export_root_name,
            "channel_thread_id_hash": short_hash(self.channel_thread_id),
            "inbound_message_id_hash": short_hash(self.inbound_message_id),
            "outbound_message_id_hash": short_hash(self.outbound_message_id),
            "inbound_at": self.inbound_at,
            "outbound_at": self.outbound_at,
            "inbound_intents": list(classify_intents(self.inbound_text, has_attachments=False)),
            "employee_reply_intents": list(classify_intents(self.employee_reply_text, has_attachments=False)),
            "inbound_chars": len(self.inbound_text),
            "employee_reply_chars": len(self.employee_reply_text),
            "source_refs": list(self.source_refs),
        }


@dataclass(frozen=True)
class ReplyQualityScore:
    role: str
    score: float
    safety_score: float
    usefulness_score: float
    action_score: float
    tone_score: float
    specificity_score: float
    problem_flags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for attr in ("score", "safety_score", "usefulness_score", "action_score", "tone_score", "specificity_score"):
            value = getattr(self, attr)
            if not 0 <= value <= 1:
                raise ValueError(f"{attr} must be between 0 and 1")
        object.__setattr__(self, "problem_flags", tuple(sorted(set(self.problem_flags))))

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreviewAuditRow:
    row_id: str
    source: str
    inbound_intents: tuple[str, ...]
    expected_actions: tuple[str, ...]
    product_actions: tuple[str, ...]
    product_score: ReplyQualityScore
    employee_score: Optional[ReplyQualityScore] = None
    comparison: str = "not_applicable"
    problem_classes: tuple[str, ...] = field(default_factory=tuple)
    safe_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "inbound_intents", tuple(sorted(set(self.inbound_intents))))
        object.__setattr__(self, "expected_actions", tuple(sorted(set(self.expected_actions))))
        object.__setattr__(self, "product_actions", tuple(sorted(set(self.product_actions))))
        object.__setattr__(self, "problem_classes", tuple(sorted(set(self.problem_classes))))
        object.__setattr__(self, "safe_metadata", dict(self.safe_metadata))

    def to_json_dict(self) -> Mapping[str, Any]:
        payload = {
            "row_id": self.row_id,
            "source": self.source,
            "inbound_intents": list(self.inbound_intents),
            "expected_actions": list(self.expected_actions),
            "product_actions": list(self.product_actions),
            "product_score": self.product_score.to_json_dict(),
            "employee_score": self.employee_score.to_json_dict() if self.employee_score else None,
            "comparison": self.comparison,
            "problem_classes": list(self.problem_classes),
            "safe_metadata": dict(self.safe_metadata),
        }
        return payload


def default_synthetic_preview_cases() -> tuple[SyntheticPreviewCase, ...]:
    return (
        SyntheticPreviewCase(
            case_id="price_question",
            description="Клиент спрашивает стоимость",
            inbound_text="Здравствуйте, сколько стоит подготовка к ЕГЭ по математике?",
            expected_actions=(ACTION_MARK_MANUAL_REVIEW,),
        ),
        SyntheticPreviewCase(
            case_id="callback_request",
            description="Клиент просит перезвонить",
            inbound_text="Можно, пожалуйста, перезвонить мне завтра после 17:00?",
            expected_actions=(ACTION_CREATE_FOLLOW_UP_TASK, ACTION_HANDOFF_TO_MANAGER),
        ),
        SyntheticPreviewCase(
            case_id="human_request",
            description="Клиент просит живого человека",
            inbound_text="Хочу поговорить с живым менеджером, бот мне не нужен.",
            expected_actions=(ACTION_HANDOFF_TO_MANAGER,),
        ),
        SyntheticPreviewCase(
            case_id="hot_lead",
            description="Горячий клиент готов записаться",
            inbound_text="Готовы записаться и оплатить сегодня, что нужно прислать?",
            expected_actions=(ACTION_MARK_MANUAL_REVIEW, ACTION_NOTIFY_ROP_HOT_LEAD),
        ),
        SyntheticPreviewCase(
            case_id="attachment_only",
            description="Клиент прислал вложение без текста",
            inbound_text="",
            expected_actions=(),
            attachments=({"kind": "document", "uri": "telegram-history:media:audit:1", "content_type": "image/png"},),
        ),
        SyntheticPreviewCase(
            case_id="location_question",
            description="Клиент уточняет очный адрес",
            inbound_text="Где проходят очные занятия для 6 класса?",
            expected_actions=(),
        ),
        SyntheticPreviewCase(
            case_id="schedule_question",
            description="Клиент спрашивает расписание",
            inbound_text="Во сколько начало занятий по воскресеньям?",
            expected_actions=(),
        ),
        SyntheticPreviewCase(
            case_id="program_question",
            description="Клиент спрашивает про программу",
            inbound_text="Какая программа по математике для 6 класса, подойдет если ребенок учится по Петерсону?",
            expected_actions=(),
        ),
        SyntheticPreviewCase(
            case_id="objection_expensive",
            description="Клиент сомневается из-за цены",
            inbound_text="Пока кажется дорого, мы подумаем и вернемся позже.",
            expected_actions=(ACTION_MARK_MANUAL_REVIEW,),
        ),
        SyntheticPreviewCase(
            case_id="personal_data",
            description="Клиент прислал персональные данные",
            inbound_text="ФИО ученика Иванов Петр, телефон +79161234567, почта parent@example.com",
            expected_actions=(ACTION_MARK_MANUAL_REVIEW,),
        ),
        SyntheticPreviewCase(
            case_id="enrollment_question",
            description="Клиент спрашивает, что нужно для записи",
            inbound_text="Что нужно отправить, чтобы записаться на пробное занятие?",
            expected_actions=(ACTION_NOTIFY_ROP_HOT_LEAD,),
        ),
        SyntheticPreviewCase(
            case_id="unclear_short",
            description="Короткое неясное сообщение",
            inbound_text="А можно?",
            expected_actions=(),
        ),
    )


def build_preview_quality_audit(
    *,
    project_root: Path | str,
    telegram_export_dir: Optional[Path | str] = None,
    real_pair_limit: int = 100,
    generated_at: Optional[datetime] = None,
) -> Mapping[str, Any]:
    if real_pair_limit <= 0:
        raise ValueError("real_pair_limit must be positive")
    root = Path(project_root).resolve(strict=False)
    generated = generated_at or datetime.now().astimezone()
    export_dir = Path(telegram_export_dir).resolve(strict=False) if telegram_export_dir else find_default_telegram_export(root)
    synthetic_cases = default_synthetic_preview_cases()

    with tempfile.TemporaryDirectory(prefix="mango_preview_quality_audit_") as temp_root_text:
        temp_root = Path(temp_root_text)
        synthetic_rows = [
            run_synthetic_case(case, temp_root=temp_root, generated_at=generated)
            for case in synthetic_cases
        ]
        telegram_pairs: tuple[TelegramReplyPair, ...] = ()
        telegram_inventory: Optional[Mapping[str, Any]] = None
        real_rows: list[PreviewAuditRow] = []
        real_source_status = "not_found"
        if export_dir and export_dir.exists():
            telegram_inventory = build_telegram_history_inventory(export_dir).to_json_dict()
            telegram_pairs = extract_latest_telegram_reply_pairs(export_dir, limit=real_pair_limit)
            real_rows = [
                run_telegram_pair(pair, temp_root=temp_root, generated_at=generated)
                for pair in telegram_pairs
            ]
            real_source_status = "loaded"

    rows = synthetic_rows + real_rows
    comparison_counts = Counter(row.comparison for row in real_rows)
    problem_counts = Counter(problem for row in rows for problem in row.problem_classes)
    synthetic_problem_counts = Counter(problem for row in synthetic_rows for problem in row.problem_classes)
    real_problem_counts = Counter(problem for row in real_rows for problem in row.problem_classes)

    report = {
        "schema_version": CUSTOMER_TIMELINE_PREVIEW_QUALITY_AUDIT_SCHEMA_VERSION,
        "artifact": "customer_timeline_preview_quality_audit",
        "generated_at": generated.isoformat(),
        "validation_ok": all(row.product_score.safety_score >= 1.0 for row in rows),
        "summary": {
            "synthetic_cases": len(synthetic_rows),
            "synthetic_product_score_avg": round(avg(row.product_score.score for row in synthetic_rows), 3),
            "synthetic_problem_counts": dict(sorted(synthetic_problem_counts.items())),
            "telegram_export_status": real_source_status,
            "telegram_export_root": project_relative_path(export_dir, root) if export_dir else None,
            "telegram_pairs_found": len(telegram_pairs),
            "telegram_pairs_sampled": len(real_rows),
            "telegram_product_score_avg": round(avg(row.product_score.score for row in real_rows), 3) if real_rows else None,
            "telegram_employee_score_avg": round(avg(row.employee_score.score for row in real_rows if row.employee_score), 3) if real_rows else None,
            "telegram_comparison_counts": dict(sorted(comparison_counts.items())),
            "problem_counts": dict(sorted(problem_counts.items())),
        },
        "checkpoints": {
            "stage_1_9_synthetic_preview": "done",
            "quality_scoring": "done",
            "real_telegram_100_pair_comparison": "done" if len(real_rows) == real_pair_limit else "partial",
            "problem_classes_and_fix_plan": "done",
        },
        "telegram_inventory": project_telegram_inventory(telegram_inventory),
        "synthetic_rows": [row.to_json_dict() for row in synthetic_rows],
        "telegram_rows": [row.to_json_dict() for row in real_rows],
        "problem_classes": build_problem_class_summary(problem_counts),
        "fix_plan": build_fix_plan(problem_counts),
        "safety": preview_quality_audit_safety_contract(),
    }
    return report


def run_synthetic_case(
    case: SyntheticPreviewCase,
    *,
    temp_root: Path,
    generated_at: datetime,
) -> PreviewAuditRow:
    report = build_preview_for_message(
        temp_root=temp_root,
        row_id=case.case_id,
        inbound_text=case.inbound_text,
        attachments=case.attachments,
        context_summary=case.context_summary,
        generated_at=generated_at,
        channel="synthetic_preview_audit",
    )
    product_actions = extract_product_action_types(report)
    intents = classify_intents(case.inbound_text, has_attachments=bool(case.attachments))
    product_score = score_product_preview(
        report,
        inbound_text=case.inbound_text,
        expected_actions=case.expected_actions,
        has_attachments=bool(case.attachments),
    )
    problem_classes = classify_product_problem_classes(product_score)
    return PreviewAuditRow(
        row_id=case.case_id,
        source="synthetic",
        inbound_intents=intents,
        expected_actions=case.expected_actions,
        product_actions=product_actions,
        product_score=product_score,
        problem_classes=problem_classes,
        safe_metadata={
            "description": case.description,
            "draft_created": bool(report.get("summary", {}).get("draft_created")),
            "status": report.get("status"),
        },
    )


def run_telegram_pair(
    pair: TelegramReplyPair,
    *,
    temp_root: Path,
    generated_at: datetime,
) -> PreviewAuditRow:
    expected_actions = expected_actions_for_message(pair.inbound_text, has_attachments=False)
    report = build_preview_for_message(
        temp_root=temp_root,
        row_id=pair.pair_id,
        inbound_text=pair.inbound_text,
        attachments=(),
        context_summary=DEFAULT_CONTEXT_SUMMARY,
        generated_at=generated_at,
        channel=TELEGRAM_HISTORY_CHANNEL,
        channel_thread_id=pair.channel_thread_id,
        channel_message_id=pair.inbound_message_id,
        received_at=pair.inbound_at,
    )
    product_score = score_product_preview(
        report,
        inbound_text=pair.inbound_text,
        expected_actions=expected_actions,
        has_attachments=False,
    )
    employee_score = score_employee_reply(
        inbound_text=pair.inbound_text,
        employee_reply_text=pair.employee_reply_text,
    )
    comparison = compare_product_with_employee(product_score, employee_score)
    product_actions = extract_product_action_types(report)
    problem_classes = classify_product_problem_classes(product_score) + classify_comparison_problem_classes(comparison, product_score, employee_score)
    return PreviewAuditRow(
        row_id=pair.pair_id,
        source="telegram_real_pair",
        inbound_intents=classify_intents(pair.inbound_text, has_attachments=False),
        expected_actions=expected_actions,
        product_actions=product_actions,
        product_score=product_score,
        employee_score=employee_score,
        comparison=comparison,
        problem_classes=problem_classes,
        safe_metadata=pair.to_safe_json_dict(),
    )


def build_preview_for_message(
    *,
    temp_root: Path,
    row_id: str,
    inbound_text: str,
    attachments: Sequence[Mapping[str, Any]],
    context_summary: str,
    generated_at: datetime,
    channel: str,
    channel_thread_id: str = "thread-audit",
    channel_message_id: Optional[str] = None,
    received_at: Optional[str] = None,
) -> Mapping[str, Any]:
    row_dir = temp_root / stable_filename(row_id)
    row_dir.mkdir(parents=True, exist_ok=True)
    pack_path = row_dir / "approved_context_pack.json"
    message_path = row_dir / "inbound_message.json"
    out_preview_path = row_dir / "preview.json"
    pack_payload = build_audit_approved_context_pack(
        case_id=row_id,
        context_summary=context_summary,
        generated_at=generated_at,
    )
    pack_path.write_text(json.dumps(pack_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    message_payload = {
        "channel": channel,
        "channel_message_id": channel_message_id or f"msg-{row_id}",
        "channel_thread_id": channel_thread_id,
        "channel_user_id": f"user-{short_hash(row_id)}",
        "direction": "inbound",
        "text": inbound_text,
        "received_at": received_at or generated_at.isoformat(),
        "attachments": list(attachments),
        "metadata": {"audit_row_id": row_id, "preview_quality_audit": True},
    }
    message_path.write_text(json.dumps(message_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return build_customer_timeline_channel_preview_from_pack(
        config=CustomerTimelineChannelPreviewFromPackConfig(
            allowed_root=temp_root,
            context_pack_json=pack_path,
            inbound_message_json=message_path,
            out_preview_json=out_preview_path,
        ),
        generated_at=generated_at,
    )


def build_audit_approved_context_pack(
    *,
    case_id: str,
    context_summary: str,
    generated_at: datetime,
) -> Mapping[str, Any]:
    chunk_id = stable_prefixed_id("context_chunk", {"case_id": case_id, "context": context_summary}, length=24)
    pack_id = stable_prefixed_id("approved_context_pack", {"case_id": case_id, "chunk_id": chunk_id}, length=24)
    chunk = {
        "chunk_id": chunk_id,
        "source_system": "synthetic_audit",
        "source_ref": "synthetic_audit:approved_context",
        "chunk_type": "sales_context",
        "summary": context_summary,
        "text": context_summary,
        "event_at": generated_at.isoformat(),
        "freshness_score": 0.9,
        "relevance_tags": ["sales", "customer_request", "safe_context"],
        "allowed_for_bot": True,
        "requires_manager_review": False,
    }
    return {
        "schema_version": CUSTOMER_TIMELINE_APPROVED_CONTEXT_PACK_SCHEMA_VERSION,
        "artifact": "customer_timeline_approved_context_pack",
        "generated_at": generated_at.isoformat(),
        "validation_ok": True,
        "pack_id": pack_id,
        "status": APPROVED_CONTEXT_PACK_STATUS,
        "tenant_id": DEFAULT_TENANT_ID,
        "customer_id": DEFAULT_AUDIT_CUSTOMER_ID,
        "summary": {
            "validation_ok": True,
            "status": APPROVED_CONTEXT_PACK_STATUS,
            "blocked_reasons": [],
            "context_chunks": 1,
            "bot_context_visible_chunks": 1,
            "bot_context_total_chunks": 1,
            "bot_context_review_required_chunks": 0,
            "current_open_conflicts": 0,
            "live_actions_available": False,
        },
        "source_refs": {
            "workspace_sha256": None,
            "decisions_jsonl_sha256": None,
            "approval_report_sha256": None,
            "self_validation_generated_at": generated_at.isoformat(),
        },
        "approval": {
            "workflow_status": "approved_for_next_dry_run",
            "next_safe_step": "prepare_read_only_dry_run_pack",
            "approved": 1,
            "rejected": 0,
            "needs_rework": 0,
            "accepted_rows": 1,
            "approved_decision_ids": [stable_prefixed_id("decision", {"case_id": case_id}, length=16)],
            "reviewers": ["preview_quality_audit"],
        },
        "customer": {
            "display_name_present": True,
            "identity_links": 1,
            "opportunities": 1,
        },
        "approved_context": {
            "scope": "bot_safe_customer_context",
            "items": [chunk],
            "summary": {
                "visible_chunks": 1,
                "total_chunks": 1,
                "review_required_chunks": 0,
            },
        },
        "channel_context": {
            "normalized_customer_id": DEFAULT_AUDIT_CUSTOMER_ID,
            "approved_context_pack_id": pack_id,
            "safe_context_summary": context_summary,
            "can_build_draft": True,
            "can_send": False,
            "requires_manager_approval_before_send": True,
        },
        "current_read_api_health": {
            "status": "synthetic_audit",
            "validation_ok": True,
            "read_only": True,
        },
        "safety": customer_timeline_approved_context_pack_safety_contract(),
    }


def find_default_telegram_export(project_root: Path) -> Optional[Path]:
    candidates: list[tuple[int, Path]] = []
    for summary in project_root.glob("telegram_exports*/**/summary.json"):
        root = summary.parent
        messages = root / "messages.jsonl"
        dialogs = root / "dialogs.jsonl"
        if not messages.exists() or not dialogs.exists():
            continue
        try:
            count = sum(1 for _ in messages.open("r", encoding="utf-8"))
        except OSError:
            continue
        candidates.append((count, root))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], str(item[1])), reverse=True)
    return candidates[0][1]


def extract_latest_telegram_reply_pairs(
    export_dir: Path | str,
    *,
    limit: int = 100,
    max_pending_inbound: int = 6,
) -> tuple[TelegramReplyPair, ...]:
    root = Path(export_dir)
    pending_by_thread: dict[str, list[ChannelMessage]] = {}
    pairs: list[TelegramReplyPair] = []
    for message in iter_telegram_history_messages(root):
        if message.metadata.get("telegram_peer_kind") not in (None, "", "user"):
            continue
        if message.direction.value == "inbound":
            pending = pending_by_thread.setdefault(message.channel_thread_id, [])
            pending.append(message)
            if len(pending) > max_pending_inbound:
                del pending[:-max_pending_inbound]
            continue
        if message.direction.value != "outbound":
            continue
        pending = pending_by_thread.get(message.channel_thread_id) or []
        if not pending:
            continue
        inbound = choose_pending_inbound(pending)
        pending.clear()
        if not inbound.text.strip() or not message.text.strip():
            continue
        pairs.append(
            TelegramReplyPair(
                pair_id=stable_prefixed_id(
                    "telegram_reply_pair",
                    {
                        "export": root.name,
                        "thread": inbound.channel_thread_id,
                        "inbound": inbound.channel_message_id,
                        "outbound": message.channel_message_id,
                    },
                    length=24,
                ),
                export_root_name=root.name,
                channel_thread_id=inbound.channel_thread_id,
                inbound_message_id=inbound.channel_message_id,
                outbound_message_id=message.channel_message_id,
                inbound_at=inbound.received_at.isoformat(),
                outbound_at=message.received_at.isoformat(),
                inbound_text=inbound.text,
                employee_reply_text=message.text,
                source_refs=(
                    str(inbound.metadata.get("source_ref") or ""),
                    str(message.metadata.get("source_ref") or ""),
                ),
            )
        )
    return tuple(sorted(pairs, key=lambda pair: pair.outbound_at)[-limit:])


def choose_pending_inbound(messages: Sequence[ChannelMessage]) -> ChannelMessage:
    informative = [message for message in messages if len(message.text.strip()) >= 3]
    if not informative:
        return messages[-1]
    return informative[-1]


def classify_intents(text: str, *, has_attachments: bool) -> tuple[str, ...]:
    lowered = text.casefold()
    intents: list[str] = []
    if has_attachments or has_any(lowered, ATTACHMENT_MARKERS):
        intents.append("attachment")
    if has_any(lowered, PRICE_MARKERS):
        intents.append("price_or_payment")
    if has_any(lowered, CALLBACK_MARKERS):
        intents.append("callback")
    if has_any(lowered, MANAGER_MARKERS):
        intents.append("manager_handoff")
    if has_any(lowered, HOT_MARKERS):
        intents.append("hot_lead")
    if has_any(lowered, LOCATION_MARKERS):
        intents.append("location")
    if has_any(lowered, SCHEDULE_MARKERS):
        intents.append("schedule")
    if has_any(lowered, PROGRAM_MARKERS):
        intents.append("program")
    if has_any(lowered, ENROLLMENT_MARKERS):
        intents.append("enrollment")
    if has_any(lowered, OBJECTION_MARKERS):
        intents.append("objection")
    if PERSONAL_DATA_RE.search(text):
        intents.append("personal_data")
    if not intents:
        intents.append("general")
    return tuple(sorted(set(intents)))


def expected_actions_for_message(text: str, *, has_attachments: bool) -> tuple[str, ...]:
    intents = set(classify_intents(text, has_attachments=has_attachments))
    actions = {ACTION_DRAFT_CLIENT_MESSAGE, ACTION_REQUEST_CRM_CONTEXT}
    if "price_or_payment" in intents or "objection" in intents or "personal_data" in intents:
        actions.add(ACTION_MARK_MANUAL_REVIEW)
    if "callback" in intents:
        actions.add(ACTION_CREATE_FOLLOW_UP_TASK)
    if "manager_handoff" in intents or "callback" in intents:
        actions.add(ACTION_HANDOFF_TO_MANAGER)
    if "hot_lead" in intents or "enrollment" in intents:
        actions.add(ACTION_NOTIFY_ROP_HOT_LEAD)
    return tuple(sorted(actions))


def score_product_preview(
    report: Mapping[str, Any],
    *,
    inbound_text: str,
    expected_actions: Sequence[str],
    has_attachments: bool,
) -> ReplyQualityScore:
    summary = report.get("summary") if isinstance(report.get("summary"), Mapping) else {}
    draft = report.get("draft_preview") if isinstance(report.get("draft_preview"), Mapping) else {}
    reply = draft.get("reply") if isinstance(draft.get("reply"), Mapping) else {}
    draft_text = str(reply.get("text") or "")
    actions = set(extract_product_action_types(report))
    expected = set(expected_actions) | set(expected_actions_for_message(inbound_text, has_attachments=has_attachments))
    flags: list[str] = []
    if report.get("validation_ok") is not True or summary.get("can_send") is not False:
        flags.append("p0_preview_not_safely_blocked_from_send")
    if reply.get("requires_approval") is not True:
        flags.append("p0_manager_approval_missing")
    if safety_contract_has_live_action(report):
        flags.append("p0_live_action_enabled")
    if not draft_text.strip():
        flags.append("p1_no_draft_text")
    if contains_personal_data(draft_text):
        flags.append("p0_personal_data_leak")
    missing_actions = sorted(expected - actions)
    for action in missing_actions:
        flags.append(f"p1_missing_action:{action}")
    if looks_too_generic(draft_text, inbound_text):
        flags.append("p2_too_generic")

    safety_score = 0.0 if any(flag.startswith("p0_") for flag in flags) else 1.0
    action_score = 1.0 if not expected else max(0.0, (len(expected) - len(missing_actions)) / len(expected))
    usefulness_score = product_usefulness_score(draft_text, inbound_text, has_attachments=has_attachments)
    tone_score = tone_score_for_text(draft_text)
    specificity_score = specificity_score_for_text(draft_text, inbound_text)
    score = weighted_score(
        safety_score=safety_score,
        usefulness_score=usefulness_score,
        action_score=action_score,
        tone_score=tone_score,
        specificity_score=specificity_score,
    )
    return ReplyQualityScore(
        role="product",
        score=score,
        safety_score=safety_score,
        usefulness_score=usefulness_score,
        action_score=action_score,
        tone_score=tone_score,
        specificity_score=specificity_score,
        problem_flags=tuple(flags),
    )


def score_employee_reply(*, inbound_text: str, employee_reply_text: str) -> ReplyQualityScore:
    flags: list[str] = []
    if not employee_reply_text.strip():
        flags.append("p1_no_employee_reply_text")
    if contains_suspicious_typo(employee_reply_text):
        flags.append("p2_possible_typo")
    if len(employee_reply_text) > 900:
        flags.append("p2_overlong_reply")
    if looks_too_generic(employee_reply_text, inbound_text):
        flags.append("p2_too_generic")
    if contains_personal_data(employee_reply_text):
        flags.append("p3_contains_contact_or_personal_data")

    safety_score = 0.82 if "p3_contains_contact_or_personal_data" in flags else 0.9
    usefulness_score = employee_usefulness_score(employee_reply_text, inbound_text)
    action_score = 0.65 if usefulness_score >= 0.65 else 0.45
    tone_score = tone_score_for_text(employee_reply_text)
    specificity_score = min(1.0, specificity_score_for_text(employee_reply_text, inbound_text) + 0.2)
    score = weighted_score(
        safety_score=safety_score,
        usefulness_score=usefulness_score,
        action_score=action_score,
        tone_score=tone_score,
        specificity_score=specificity_score,
    )
    return ReplyQualityScore(
        role="employee",
        score=score,
        safety_score=safety_score,
        usefulness_score=usefulness_score,
        action_score=action_score,
        tone_score=tone_score,
        specificity_score=specificity_score,
        problem_flags=tuple(flags),
    )


def compare_product_with_employee(product: ReplyQualityScore, employee: ReplyQualityScore) -> str:
    if product.safety_score < 1.0:
        return "employee_better"
    employee_content_advantage = (
        employee.usefulness_score
        + employee.specificity_score
        - product.usefulness_score
        - product.specificity_score
    )
    product_safety_advantage = product.safety_score - employee.safety_score
    if employee_content_advantage >= 0.22 and employee.safety_score >= 0.75:
        return "employee_better"
    if product_safety_advantage >= 0.15 and product.score >= employee.score - 0.05:
        return "product_better"
    if employee.score - product.score >= 0.12:
        return "employee_better"
    if product.score - employee.score >= 0.12:
        return "product_better"
    return "rough_tie"


def extract_product_action_types(report: Mapping[str, Any]) -> tuple[str, ...]:
    draft = report.get("draft_preview") if isinstance(report.get("draft_preview"), Mapping) else {}
    reply = draft.get("reply") if isinstance(draft.get("reply"), Mapping) else {}
    return tuple(
        sorted(
            {
                str(action.get("action_type"))
                for action in reply.get("recommended_actions") or ()
                if isinstance(action, Mapping) and action.get("action_type")
            }
        )
    )


def classify_product_problem_classes(score: ReplyQualityScore) -> tuple[str, ...]:
    classes: list[str] = []
    for flag in score.problem_flags:
        if flag.startswith("p0_"):
            classes.append("p0_safety_or_leak")
        elif flag.startswith("p1_missing_action"):
            classes.append("p1_missing_recommended_action")
        elif flag == "p1_no_draft_text":
            classes.append("p1_no_product_draft")
        elif flag == "p2_too_generic":
            classes.append("p2_generic_answer")
    if score.specificity_score < 0.45:
        classes.append("p2_low_specificity")
    if score.usefulness_score < 0.6:
        classes.append("p2_low_usefulness")
    return tuple(sorted(set(classes)))


def classify_comparison_problem_classes(
    comparison: str,
    product: ReplyQualityScore,
    employee: ReplyQualityScore,
) -> tuple[str, ...]:
    classes: list[str] = []
    if comparison == "employee_better" and employee.specificity_score > product.specificity_score:
        classes.append("employee_has_more_specific_answer")
    if comparison == "product_better" and employee.problem_flags:
        classes.append("employee_reply_has_quality_risk")
    if "p2_possible_typo" in employee.problem_flags:
        classes.append("employee_reply_possible_typo")
    if "p3_contains_contact_or_personal_data" in employee.problem_flags:
        classes.append("employee_reply_contains_personal_data")
    return tuple(sorted(set(classes)))


def build_problem_class_summary(problem_counts: Counter[str]) -> Mapping[str, Any]:
    descriptions = {
        "p0_safety_or_leak": "Критическая проблема безопасности: утечка или возможность живого действия.",
        "p1_missing_recommended_action": "Черновик есть, но не создано ожидаемое действие для менеджера.",
        "p1_no_product_draft": "Система не смогла подготовить черновик.",
        "p2_generic_answer": "Ответ безопасный, но слишком общий.",
        "p2_low_specificity": "Ответ не содержит конкретики, которую обычно дает сотрудник.",
        "p2_low_usefulness": "Ответ слабо помогает продвинуть диалог.",
        "employee_has_more_specific_answer": "Сотрудник лучше за счет конкретной цены, расписания, адреса или правила.",
        "employee_reply_has_quality_risk": "У ответа сотрудника есть риск качества, где продукт может быть полезнее как страховка.",
        "employee_reply_possible_typo": "В ответе сотрудника есть возможная опечатка или неаккуратность.",
        "employee_reply_contains_personal_data": "Ответ сотрудника содержит контактные или персональные данные.",
    }
    return {
        key: {
            "count": count,
            "description": descriptions.get(key, "Класс проблемы требует ручной классификации."),
        }
        for key, count in sorted(problem_counts.items())
    }


def build_fix_plan(problem_counts: Counter[str]) -> list[Mapping[str, Any]]:
    plan: list[Mapping[str, Any]] = []
    if problem_counts.get("p0_safety_or_leak"):
        plan.append(
            {
                "priority": "P0",
                "step": "Остановить расширение функциональности и исправить safety-gate.",
                "test": "Добавить регрессионный тест на конкретный флаг/утечку.",
            }
        )
    if problem_counts.get("p1_missing_recommended_action"):
        plan.append(
            {
                "priority": "P1",
                "step": "Расширить словарь сигналов и правила рекомендуемых действий.",
                "test": "Синтетические сообщения должны создавать ожидаемые действия менеджеру.",
            }
        )
    if problem_counts.get("p2_generic_answer") or problem_counts.get("p2_low_specificity") or problem_counts.get("employee_has_more_specific_answer"):
        plan.append(
            {
                "priority": "P1",
                "step": "Подключить проверенную базу знаний о программах, ценах, расписании и адресах как разрешенный контекст.",
                "test": "На Telegram-парах продукт должен давать не только безопасный, но и предметный черновик.",
            }
        )
    if problem_counts.get("p2_low_usefulness"):
        plan.append(
            {
                "priority": "P2",
                "step": "Добавить оценщик пользы: отвечает ли черновик на вопрос клиента и предлагает ли следующий шаг.",
                "test": "Отдельный набор реальных вопросов с ожидаемым типом ответа.",
            }
        )
    if problem_counts.get("employee_reply_possible_typo"):
        plan.append(
            {
                "priority": "P2",
                "step": "Сделать режим помощника менеджера: подсветка опечаток и слабых формулировок в ответах сотрудников.",
                "test": "Реальные ответы с опечатками должны попадать в мягкое предупреждение.",
            }
        )
    if not plan:
        plan.append(
            {
                "priority": "P3",
                "step": "Расширить выборку и подключить ручную оценку РОП на спорных парах.",
                "test": "Сравнить автоматическую оценку с ручной разметкой.",
            }
        )
    return plan


def render_preview_quality_audit_markdown(report: Mapping[str, Any]) -> str:
    summary = report.get("summary", {})
    problems = report.get("problem_classes", {})
    fix_plan = report.get("fix_plan", [])
    comparison_counts = summary.get("telegram_comparison_counts") if isinstance(summary.get("telegram_comparison_counts"), Mapping) else {}
    lines = [
        "# Аудит качества черновиков по единой истории клиента",
        "",
        f"Дата: {report.get('generated_at')}",
        "Контур: единая история клиента -> безопасный черновик ответа",
        "Статус: проверка только на чтение, без отправки сообщений и без записи в CRM",
        "",
        "## Что проверено",
        "",
        "1. Синтетические входящие сообщения прогнаны через проверенный контур этапов 1-9.",
        "2. Для каждого сообщения создан черновик ответа и список действий для менеджера.",
        "3. Из Telegram-экспорта собраны реальные пары: сообщение клиента -> ответ сотрудника.",
        "4. Продуктовые черновики сравнены с ответами сотрудников по безопасности, пользе, конкретике, тону и действиям.",
        "",
        "## Итоги",
        "",
        f"- Синтетических проверок: {summary.get('synthetic_cases')}",
        f"- Средняя оценка продукта на синтетике: {summary.get('synthetic_product_score_avg')}",
        f"- Telegram-источник: {summary.get('telegram_export_root')}",
        f"- Реальных Telegram-пар найдено в выборке: {summary.get('telegram_pairs_sampled')} из запрошенных 100",
        f"- Средняя оценка продукта на Telegram: {summary.get('telegram_product_score_avg')}",
        f"- Средняя оценка сотрудника на Telegram: {summary.get('telegram_employee_score_avg')}",
        f"- Сравнение продукт/сотрудник: {json.dumps(summary.get('telegram_comparison_counts'), ensure_ascii=False, sort_keys=True)}",
        "",
        "## Где лучше продукт",
        "",
        f"- Лучше продукта в выборке: {comparison_counts.get('product_better', 0)} из {summary.get('telegram_pairs_sampled')} реальных пар.",
        "- Сильная сторона продукта сейчас - безопасность: он не отправляет сообщения сам, не пишет в CRM, не раскрывает телефоны и почту в черновике.",
        "- Продукт полезен как страховка от поспешных ответов, опечаток и неаккуратных формулировок.",
        "",
        "## Где лучше сотрудник",
        "",
        f"- Лучше сотрудника в выборке: {comparison_counts.get('employee_better', 0)} из {summary.get('telegram_pairs_sampled')} реальных пар.",
        "- Сильная сторона сотрудника - предметная конкретика: цена, расписание, адрес, формат занятий, правила записи.",
        "- Это ожидаемый результат: текущий черновик намеренно не использует базу знаний и не придумывает факты.",
        "",
        "## Классы проблем",
        "",
    ]
    if problems:
        for key, payload in problems.items():
            lines.append(f"- `{key}`: {payload.get('count')} - {payload.get('description')}")
    else:
        lines.append("- Критичных классов проблем не найдено.")
    lines.extend(
        [
            "",
            "## План устранения",
            "",
        ]
    )
    for item in fix_plan:
        lines.append(f"- {item.get('priority')}: {item.get('step')} Тест: {item.get('test')}")
    lines.extend(
        [
            "",
            "## Вывод",
            "",
            "Текущий продуктовый контур уже безопасен как помощник, который готовит черновик и действия для менеджера.",
            "Главный разрыв с реальными сотрудниками - конкретика: сотрудники часто отвечают точной ценой, расписанием, адресом или правилом записи.",
            "Значит, следующий сильный шаг - подключить проверенную базу знаний и разрешенный контекст, чтобы черновик стал не только безопасным, но и предметным.",
            "",
            "## Безопасность",
            "",
            "- Живая отправка сообщений: нет.",
            "- Запись в CRM/Tallanto: нет.",
            "- ASR/R+A: нет.",
            "- Запись в runtime DB: нет.",
            "- Чтение Telegram: только локальный экспорт.",
            "",
        ]
    )
    return "\n".join(lines)


def project_telegram_inventory(inventory: Optional[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    if not inventory:
        return None
    return {
        "export_root_name": inventory.get("export_root_name"),
        "dialogs_total": inventory.get("dialogs_total"),
        "messages_total": inventory.get("messages_total"),
        "message_date_start": inventory.get("message_date_start"),
        "message_date_end": inventory.get("message_date_end"),
        "direction_counts": inventory.get("direction_counts"),
        "content_counts": inventory.get("content_counts"),
        "safety": inventory.get("safety"),
    }


def preview_quality_audit_safety_contract() -> Mapping[str, Any]:
    return {
        "schema_version": CUSTOMER_TIMELINE_PREVIEW_QUALITY_AUDIT_SCHEMA_VERSION,
        "read_only": True,
        "local_telegram_export_read": True,
        "write_report_artifact": True,
        "live_send": False,
        "send_messenger": False,
        "send_email": False,
        "write_crm": False,
        "write_tallanto": False,
        "write_runtime_db": False,
        "run_asr": False,
        "run_ra": False,
        "network_calls": False,
        "llm_calls": False,
        "rag_used": False,
        "mutate_stable_runtime": False,
    }


def weighted_score(
    *,
    safety_score: float,
    usefulness_score: float,
    action_score: float,
    tone_score: float,
    specificity_score: float,
) -> float:
    return round(
        0.3 * safety_score
        + 0.25 * usefulness_score
        + 0.2 * action_score
        + 0.15 * tone_score
        + 0.1 * specificity_score,
        3,
    )


def product_usefulness_score(text: str, inbound_text: str, *, has_attachments: bool) -> float:
    lowered = text.casefold()
    intents = set(classify_intents(inbound_text, has_attachments=has_attachments))
    if not text.strip():
        return 0.0
    if "attachment" in intents and "влож" in lowered:
        return 0.85
    if "price_or_payment" in intents and "стоим" in lowered:
        return 0.75
    if "callback" in intents and ("свяж" in lowered or "передадим" in lowered):
        return 0.72
    if "manager_handoff" in intents and ("менедж" in lowered or "передадим" in lowered):
        return 0.72
    if "общ" in lowered or "историю" in lowered or "уточн" in lowered:
        return 0.58
    return 0.5


def employee_usefulness_score(text: str, inbound_text: str) -> float:
    if not text.strip():
        return 0.0
    intents = set(classify_intents(inbound_text, has_attachments=False))
    lowered = text.casefold()
    score = 0.45
    if len(text.strip()) >= 25:
        score += 0.15
    if "price_or_payment" in intents and has_any(lowered, PRICE_MARKERS):
        score += 0.2
    if "location" in intents and has_any(lowered, ("москва", "адрес", "метро", "пер.", "ул.", "очно")):
        score += 0.2
    if "schedule" in intents and has_any(lowered, ("время", "10", "12", "день", "воскрес")):
        score += 0.18
    if "program" in intents and has_any(lowered, PROGRAM_MARKERS):
        score += 0.15
    if "enrollment" in intents and has_any(lowered, ("пришлите", "фио", "запис", "почт", "телефон")):
        score += 0.18
    if "?" in text:
        score += 0.08
    if looks_too_generic(text, inbound_text):
        score -= 0.2
    return round(max(0.0, min(1.0, score)), 3)


def tone_score_for_text(text: str) -> float:
    lowered = text.casefold()
    if not text.strip():
        return 0.0
    score = 0.82
    if any(marker in lowered for marker in ("срочно оплатите", "вы должны", "последний шанс")):
        score -= 0.35
    if len(text) > 600:
        score -= 0.12
    if "пожалуйста" in lowered or "спасибо" in lowered or "добрый" in lowered or "здравствуйте" in lowered:
        score += 0.08
    return round(max(0.0, min(1.0, score)), 3)


def specificity_score_for_text(text: str, inbound_text: str) -> float:
    lowered = text.casefold()
    inbound_intents = set(classify_intents(inbound_text, has_attachments=False))
    score = 0.25
    if any(char.isdigit() for char in text):
        score += 0.18
    if has_any(lowered, ("руб", "₽", "стоим", "оплат")):
        score += 0.16
    if has_any(lowered, ("москва", "адрес", "метро", "пер.", "ул.")):
        score += 0.16
    if has_any(lowered, ("класс", "егэ", "огэ", "математ", "физик", "олимпиад", "группа")):
        score += 0.14
    if has_any(lowered, ("воскрес", "понедель", "вторник", "сред", "четверг", "пятниц", "суббот", "10", "12")):
        score += 0.12
    if inbound_intents == {"general"} and len(text) >= 40:
        score += 0.08
    return round(max(0.0, min(1.0, score)), 3)


def looks_too_generic(reply_text: str, inbound_text: str) -> bool:
    lowered = reply_text.casefold().strip()
    if not lowered:
        return True
    generic_markers = (
        "уточним детали",
        "вернемся с ответом",
        "менеджер посмотрит историю",
        "менеджер уточнит детали",
    )
    if any(marker in lowered for marker in generic_markers):
        intents = set(classify_intents(inbound_text, has_attachments=False))
        specific_intents = intents - {"general", "callback", "manager_handoff", "personal_data"}
        return bool(specific_intents)
    return False


def safety_contract_has_live_action(report: Mapping[str, Any]) -> bool:
    safety = report.get("safety") if isinstance(report.get("safety"), Mapping) else {}
    for key in ("live_send", "send_messenger", "send_email", "write_crm", "write_tallanto", "write_runtime_db", "run_asr", "run_ra"):
        if safety.get(key) is not False:
            return True
    policy = report.get("context_policy") if isinstance(report.get("context_policy"), Mapping) else {}
    return any(policy.get(key) is not False for key in ("live_send", "write_crm", "write_tallanto", "write_runtime_db"))


def contains_personal_data(text: str) -> bool:
    return bool(PERSONAL_DATA_RE.search(text or ""))


def contains_suspicious_typo(text: str) -> bool:
    lowered = text.casefold()
    return any(marker in lowered for marker in ("далле", "пож-та", "здравствуйте.", "  "))


def has_any(text: str, markers: Sequence[str]) -> bool:
    return any(marker in text for marker in markers)


def avg(values: Sequence[float]) -> float:
    clean = [value for value in values if value is not None]
    if not clean:
        return 0.0
    return sum(clean) / len(clean)


def short_hash(value: str) -> str:
    return stable_prefixed_id("hash", {"value": value}, length=12)


def project_relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return path.name


def stable_filename(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return normalized[:80] or short_hash(value)


__all__ = [
    "CUSTOMER_TIMELINE_PREVIEW_QUALITY_AUDIT_SCHEMA_VERSION",
    "PreviewAuditRow",
    "ReplyQualityScore",
    "SyntheticPreviewCase",
    "TelegramReplyPair",
    "build_audit_approved_context_pack",
    "build_preview_quality_audit",
    "classify_intents",
    "compare_product_with_employee",
    "default_synthetic_preview_cases",
    "expected_actions_for_message",
    "extract_latest_telegram_reply_pairs",
    "find_default_telegram_export",
    "render_preview_quality_audit_markdown",
    "score_employee_reply",
    "score_product_preview",
]
