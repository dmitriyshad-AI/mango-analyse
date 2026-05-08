from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Sequence

from mango_mvp.productization.repository import ManagerRollupItem, ProductCallRecord, ProductRepository


UI_CONTRACT_SCHEMA_VERSION = "saas_ui_contracts_v1"


@dataclass(frozen=True)
class CallListItemDTO:
    event_key: str
    source_filename: str
    started_at: Optional[str]
    duration_sec: Optional[float]
    provider: str
    provider_call_id: str
    recording_id: str
    manager_extension: str
    manager_display_name: Optional[str]
    manager_crm_owner_id: Optional[int]
    manager_crm_owner_name: Optional[str]
    manager_crm_match_status: Optional[str]
    raw_payload_ref: Optional[str]

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ManagerFilterDTO:
    manager_extension: str
    label: str
    call_count: int
    crm_owner_status: str
    crm_owner_id: Optional[int]
    crm_owner_name: Optional[str]

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ManualReviewDTO:
    manager_extension: str
    mango_name: Optional[str]
    mango_email: Optional[str]
    call_count: int
    required_action: str
    reason: str

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def build_dashboard_contract(repo: ProductRepository, call_limit: int = 50) -> Mapping[str, Any]:
    summary = repo.summary().to_json_dict()
    calls = [call_item_dto(call).to_json_dict() for call in repo.list_calls(limit=call_limit)]
    managers = [manager_filter_dto(row).to_json_dict() for row in repo.manager_rollup()]
    review_queue = [manual_review_dto(row).to_json_dict() for row in repo.manual_owner_review_queue()]
    return {
        "schema_version": UI_CONTRACT_SCHEMA_VERSION,
        "summary": summary,
        "filters": {
            "managers": managers,
            "crm_owner_status": ["all", "present", "missing"],
            "providers": sorted({item["provider"] for item in calls}),
        },
        "views": {
            "call_list": {
                "items": calls,
                "limit": call_limit,
                "total_available": summary["enriched_view_rows"],
            },
            "manual_owner_review_queue": {
                "items": review_queue,
                "total_available": len(review_queue),
            },
        },
        "actions": {
            "allowed": ["shadow_poll_dry_run", "tenant_owner_mapping_review", "export_json_report"],
            "blocked": ["download_audio", "run_asr", "run_ra", "write_crm", "write_runtime_db"],
        },
        "provenance": {
            "db_path": summary["db_path"],
            "primary_call_key": "event_key",
            "raw_payload_ref_field": "raw_payload_ref",
        },
    }


def call_item_dto(call: ProductCallRecord) -> CallListItemDTO:
    return CallListItemDTO(
        event_key=call.event_key,
        source_filename=call.source_filename,
        started_at=call.started_at,
        duration_sec=call.duration_sec,
        provider=call.provider,
        provider_call_id=call.provider_call_id,
        recording_id=call.recording_id,
        manager_extension=call.manager_extension,
        manager_display_name=call.manager_display_name,
        manager_crm_owner_id=call.manager_crm_owner_id,
        manager_crm_owner_name=call.manager_crm_owner_name,
        manager_crm_match_status=call.manager_crm_match_status,
        raw_payload_ref=call.raw_payload_ref,
    )


def manager_filter_dto(row: ManagerRollupItem) -> ManagerFilterDTO:
    label_name = row.mango_name or f"extension {row.manager_extension}"
    return ManagerFilterDTO(
        manager_extension=row.manager_extension,
        label=f"{label_name} ({row.manager_extension})",
        call_count=row.call_count,
        crm_owner_status="present" if row.crm_owner_id is not None else "missing",
        crm_owner_id=row.crm_owner_id,
        crm_owner_name=row.crm_owner_name,
    )


def manual_review_dto(row: ManagerRollupItem) -> ManualReviewDTO:
    if row.mapping_status != "mapped_mango_user":
        reason = "missing_mango_user"
        required_action = "map_mango_extension_to_user"
    else:
        reason = "crm_owner_missing"
        required_action = "set_or_confirm_crm_owner"
    return ManualReviewDTO(
        manager_extension=row.manager_extension,
        mango_name=row.mango_name,
        mango_email=row.mango_email,
        call_count=row.call_count,
        required_action=required_action,
        reason=reason,
    )
