from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Sequence

from mango_mvp.productization.repository import ManagerRollupItem, ProductCallRecord, ProductRepository


INSIGHT_SEED_SCHEMA_VERSION = "product_insight_seed_v1"


@dataclass(frozen=True)
class InsightEvidenceRef:
    event_key: str
    source_filename: str
    raw_payload_ref: Optional[str]

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InsightSeed:
    seed_id: str
    tenant_id: str
    provider: str
    topic: str
    priority: str
    manager_extension: str
    manager_name: Optional[str]
    crm_owner_id: Optional[int]
    crm_owner_name: Optional[str]
    call_count: int
    question: str
    evidence_refs: Sequence[Mapping[str, Any]]
    next_action: str

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def build_insight_seed_report(
    repo: ProductRepository,
    max_evidence_per_manager: int = 3,
) -> Mapping[str, Any]:
    rollup = list(repo.manager_rollup())
    seeds = []
    for index, manager in enumerate(rollup, start=1):
        calls = repo.list_calls(limit=max_evidence_per_manager, manager_extension=manager.manager_extension)
        if manager.crm_owner_id is None:
            seeds.append(manual_owner_seed(index, manager, calls).to_json_dict())
        else:
            seeds.append(manager_volume_seed(index, manager, calls).to_json_dict())
    summary = {
        "schema_version": INSIGHT_SEED_SCHEMA_VERSION,
        "db_path": str(repo.db_path),
        "seeds": len(seeds),
        "manual_owner_seeds": sum(1 for seed in seeds if seed["topic"] == "manual_crm_owner_mapping"),
        "manager_volume_seeds": sum(1 for seed in seeds if seed["topic"] == "manager_call_volume"),
        "evidence_refs": sum(len(seed["evidence_refs"]) for seed in seeds),
        "validation_ok": True,
    }
    return {"summary": summary, "items": seeds}


def manual_owner_seed(index: int, manager: ManagerRollupItem, calls: Sequence[ProductCallRecord]) -> InsightSeed:
    return InsightSeed(
        seed_id=f"insight-seed-{index:04d}",
        tenant_id=manager.tenant_id,
        provider=manager.provider,
        topic="manual_crm_owner_mapping",
        priority="high" if manager.call_count >= 20 else "medium",
        manager_extension=manager.manager_extension,
        manager_name=manager.mango_name,
        crm_owner_id=None,
        crm_owner_name=None,
        call_count=manager.call_count,
        question="Which CRM owner should be responsible for this Mango manager's calls?",
        evidence_refs=[evidence_ref(call).to_json_dict() for call in calls],
        next_action="confirm tenant owner mapping before CRM write automation",
    )


def manager_volume_seed(index: int, manager: ManagerRollupItem, calls: Sequence[ProductCallRecord]) -> InsightSeed:
    return InsightSeed(
        seed_id=f"insight-seed-{index:04d}",
        tenant_id=manager.tenant_id,
        provider=manager.provider,
        topic="manager_call_volume",
        priority="medium" if manager.call_count >= 20 else "low",
        manager_extension=manager.manager_extension,
        manager_name=manager.mango_name,
        crm_owner_id=manager.crm_owner_id,
        crm_owner_name=manager.crm_owner_name,
        call_count=manager.call_count,
        question="What should the SaaS dashboard show for this manager's captured conversation volume and outcomes?",
        evidence_refs=[evidence_ref(call).to_json_dict() for call in calls],
        next_action="attach ASR/R+A insights after processing is explicitly enabled",
    )


def evidence_ref(call: ProductCallRecord) -> InsightEvidenceRef:
    return InsightEvidenceRef(
        event_key=call.event_key,
        source_filename=call.source_filename,
        raw_payload_ref=call.raw_payload_ref,
    )
