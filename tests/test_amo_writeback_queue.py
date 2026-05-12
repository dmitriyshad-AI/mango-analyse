from __future__ import annotations

import csv
from pathlib import Path

from scripts import build_amo_writeback_queue as queue


def test_classify_excludes_already_written_by_phone() -> None:
    context = _context(written_by_phone={"+79000000001": {"status": "written", "contact_id": "111"}})

    bucket, row = queue.classify_queue_row(_row(phone="+7 900 000-00-01", amo_ids="111"), index=1, context=context)

    assert bucket == queue.BUCKET_ALREADY_WRITTEN
    assert row["queue_reason"] == "already_written_by_phone_or_contact_id"


def test_manual_review_takes_priority_over_already_written() -> None:
    context = _context(
        written_by_phone={"+79000000002": {"status": "written", "contact_id": "222"}},
        manual_review_by_phone={"+79000000002": {"__report_path": "manual.csv"}},
    )

    bucket, row = queue.classify_queue_row(_row(phone="+7 900 000-00-02", amo_ids="222"), index=1, context=context)

    assert bucket == queue.BUCKET_TEXT_QUALITY
    assert row["queue_reason"] == "manual_review_input:already_written"


def test_classify_blocks_contact_id_mismatch_from_dry_run() -> None:
    context = _context(
        dry_run_by_phone={
            "+79000000003": {
                "status": "skipped",
                "reason": "contact_id_mismatch_with_source_amo_contact_ids",
                "contact_id": "999",
            }
        }
    )

    bucket, _row_out = queue.classify_queue_row(_row(phone="+7 900 000-00-03", amo_ids="333"), index=1, context=context)

    assert bucket == queue.BUCKET_CONTACT_ID_MISMATCH


def test_classify_routes_multi_contact_to_manager_review() -> None:
    context = _context(dry_run_by_phone={"+79000000004": {"status": "dry_run", "contact_id": "444"}})

    bucket, row = queue.classify_queue_row(_row(phone="+7 900 000-00-04", amo_ids="333 | 444"), index=1, context=context)

    assert bucket == queue.BUCKET_MULTI_CONTACT
    assert "ambiguous_multiple_contact_ids" in row["queue_reason"]


def test_classify_routes_quality_blocks_to_text_review() -> None:
    context = _context(
        dry_run_by_phone={"+79000000005": {"status": "dry_run", "contact_id": "555"}},
        quality_by_phone={"+79000000005": {"decision": "block", "risk_types": "lossy_ellipsis_truncation"}},
    )

    bucket, row = queue.classify_queue_row(_row(phone="+7 900 000-00-05", amo_ids="555"), index=1, context=context)

    assert bucket == queue.BUCKET_TEXT_QUALITY
    assert "lossy_ellipsis_truncation" in row["queue_reason"]


def test_classify_routes_service_context_to_deferred() -> None:
    context = _context(dry_run_by_phone={"+79000000006": {"status": "dry_run", "contact_id": "666"}})

    bucket, row = queue.classify_queue_row(
        _row(phone="+7 900 000-00-06", amo_ids="666", call_type="service_call"), index=1, context=context
    )

    assert bucket == queue.BUCKET_DEFERRED_NON_SALES
    assert "service_call" in row["queue_reason"]


def test_classify_requires_real_tunnel_dry_run_before_ready() -> None:
    context = _context()

    bucket, row = queue.classify_queue_row(_row(phone="+7 900 000-00-07", amo_ids="777"), index=1, context=context)

    assert bucket == queue.BUCKET_MULTI_CONTACT
    assert row["queue_reason"] == "not_verified_by_real_tunnel_dry_run"


def test_classify_allows_single_contact_after_quality_and_dry_run() -> None:
    context = _context(dry_run_by_phone={"+79000000008": {"status": "dry_run", "contact_id": "888"}})

    bucket, row = queue.classify_queue_row(_row(phone="+7 900 000-00-08", amo_ids="888"), index=1, context=context)

    assert bucket == queue.BUCKET_READY_SINGLE
    assert row["effective_contact_id"] == "888"


def test_build_queue_reads_reports_and_writes_buckets(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    live = tmp_path / "20260510T000000Z" / "contact_writeback_report.csv"
    dry = tmp_path / "20260510T010000Z" / "contact_writeback_report.csv"
    quality = tmp_path / "quality.csv"
    manual = tmp_path / "manual.csv"
    _write_csv(
        source,
        [
            _row(phone="+79000000009", amo_ids="999"),
            _row(phone="+79000000010", amo_ids="1010"),
            _row(phone="+79000000011", amo_ids="1111"),
        ],
    )
    _write_csv(live, [{"phone": "+79000000009", "status": "written", "contact_id": "999"}])
    _write_csv(dry, [{"phone": "+79000000010", "status": "dry_run", "contact_id": "1010"}])
    _write_csv(quality, [{"phone": "+79000000011", "decision": "block", "risk_types": "out_of_domain_b2b"}])
    _write_csv(manual, [{"Телефон клиента": "+79000000010"}])

    summary = queue.build_amo_writeback_queue(
        queue.AmoWritebackQueueConfig(
            input_csv=source,
            out_root=tmp_path / "queue",
            writeback_reports=(live,),
            dry_run_reports=(dry,),
            quality_reports=(quality,),
            manual_review_inputs=(manual,),
        )
    )

    assert summary["bucket_counts"][queue.BUCKET_ALREADY_WRITTEN] == 1
    assert summary["bucket_counts"][queue.BUCKET_TEXT_QUALITY] == 2
    assert (tmp_path / "queue" / f"{queue.BUCKET_TEXT_QUALITY}.csv").exists()


def _context(
    *,
    written_by_phone: dict[str, dict[str, object]] | None = None,
    written_by_contact_id: dict[str, dict[str, object]] | None = None,
    dry_run_by_phone: dict[str, dict[str, object]] | None = None,
    dry_run_by_contact_id: dict[str, dict[str, object]] | None = None,
    quality_by_phone: dict[str, dict[str, object]] | None = None,
    quality_by_row_index: dict[str, dict[str, object]] | None = None,
    manual_review_by_phone: dict[str, dict[str, object]] | None = None,
) -> queue.QueueContext:
    return queue.QueueContext(
        written_by_phone=written_by_phone or {},
        written_by_contact_id=written_by_contact_id or {},
        dry_run_by_phone=dry_run_by_phone or {},
        dry_run_by_contact_id=dry_run_by_contact_id or {},
        quality_by_phone=quality_by_phone or {},
        quality_by_row_index=quality_by_row_index or {},
        manual_review_by_phone=manual_review_by_phone or {},
    )


def _row(
    *,
    phone: str,
    amo_ids: str,
    call_type: str = "sales_call",
    ready: str = "Да",
    policy: str = "live_update_ready",
    entity_policy: str = "update_existing_single_amo_contact",
) -> dict[str, str]:
    return {
        "Телефон клиента": phone,
        "AMO contact IDs": amo_ids,
        "Тип последнего свежего звонка": call_type,
        "Готово к записи в AMO": ready,
        "CRM writeback policy": policy,
        "CRM writeback blockers": "",
        "AMO entity policy": entity_policy,
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
