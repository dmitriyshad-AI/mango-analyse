from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path('/Users/dmitrijfabarisov/Projects/Mango analyse')
REPORT_ROOT = PROJECT_ROOT / 'stable_runtime' / 'amocrm_runtime' / 'deal_writebacks'
ENV_FILES = (
    PROJECT_ROOT / 'stable_runtime' / 'amocrm_runtime' / '.env.private',
    PROJECT_ROOT / 'prod_runtime_transfer' / '.env.private',
)
ACTIONABLE_VERDICTS = {'reopen_recommended', 'closed_too_early', 'follow_up_needed', 'alternative_offer_needed'}
LIVE_WRITE_CONFIRMATION = 'WRITE_AMO_LIVE'


def _live_write_enabled(args: argparse.Namespace) -> bool:
    execute_live_write = bool(getattr(args, 'execute_live_write', False))
    confirmation = str(getattr(args, 'live_confirmation', '') or '').strip()
    if execute_live_write and confirmation != LIVE_WRITE_CONFIRMATION:
        raise ValueError(
            f"Live amoCRM writeback requires --live-confirmation {LIVE_WRITE_CONFIRMATION!r}."
        )
    if confirmation and not execute_live_write:
        raise ValueError('--live-confirmation is only valid together with --execute-live-write.')
    return execute_live_write


def _safe_text(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, float) and pd.isna(value):
        return ''
    return str(value).strip()


def _load_env_files() -> None:
    for path in ENV_FILES:
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding='utf-8').splitlines():
            line = raw_line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            os.environ.setdefault(key.strip(), value.strip())
    os.environ.setdefault('DATABASE_URL', f"sqlite:///{(PROJECT_ROOT / 'stable_runtime' / 'amocrm_runtime' / 'amo_runtime.db').resolve()}")


def _collect_previously_written_leads() -> set[int]:
    result: set[int] = set()
    root = PROJECT_ROOT / 'stable_runtime' / 'amocrm_runtime' / 'deal_analysis'
    for path in sorted(root.glob('*/all_results.json')):
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            continue
        if not isinstance(payload, list):
            continue
        for row in payload:
            if not isinstance(row, dict):
                continue
            writeback = row.get('writeback_result') or {}
            if isinstance(writeback, dict) and _safe_text(writeback.get('status')) == 'written':
                lead_id = int(row.get('matched_lead_id') or 0)
                if lead_id:
                    result.add(lead_id)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description='Build fresh recent closed queue and dry-run/live write actionable deals to amoCRM.')
    parser.add_argument('--days-back', type=int, default=30)
    parser.add_argument('--max-leads', type=int, default=None)
    parser.add_argument('--source-all-results', default=None, help='Готовый all_results.json из прошлого live run; если передан, fresh queue не строится.')
    parser.add_argument(
        '--execute-live-write',
        action='store_true',
        help='Разрешить live-запись в amoCRM. Без этого флага скрипт делает только dry-run отчет.',
    )
    parser.add_argument(
        '--live-confirmation',
        default='',
        help=f'Контрольная строка для live-записи: {LIVE_WRITE_CONFIRMATION}.',
    )
    args = parser.parse_args()
    try:
        live_write = _live_write_enabled(args)
    except ValueError as exc:
        print(f'Refusing live amoCRM writeback: {exc}', file=sys.stderr)
        return 2

    _load_env_files()

    from mango_mvp.amocrm_runtime.db import SessionLocal
    from mango_mvp.amocrm_runtime.deals import _prepare_writeback_payload, build_recent_closed_queue, write_analysis_to_lead

    previous_written = _collect_previously_written_leads()

    session = SessionLocal()
    try:
        if args.source_all_results:
            source_path = Path(args.source_all_results).resolve()
            analyses = json.loads(source_path.read_text(encoding='utf-8'))
            queue_summary = {
                'run_id': 'source_reuse',
                'source_all_results_json': str(source_path),
                'days_back': args.days_back,
                'max_leads': args.max_leads,
            }
        else:
            queue_summary = build_recent_closed_queue(
                session,
                days_back=args.days_back,
                apply_writeback=False,
                max_leads=args.max_leads,
            )
            analyses = json.loads(Path(queue_summary['files']['all_results_json']).read_text(encoding='utf-8'))

        actionable: list[dict[str, Any]] = []
        for row in analyses:
            if not isinstance(row, dict):
                continue
            lead_id = int(row.get('matched_lead_id') or 0)
            if not lead_id or lead_id in previous_written:
                continue
            if _safe_text(row.get('close_verdict')) not in ACTIONABLE_VERDICTS:
                continue
            if not bool(row.get('writeback_allowed')):
                continue
            actionable.append(row)

        run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        run_dir = REPORT_ROOT / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        report_rows: list[dict[str, Any]] = []
        total = len(actionable)
        for index, analysis in enumerate(actionable, start=1):
            lead_id = int(analysis.get('matched_lead_id') or 0)
            report_row = {
                'row_index': index,
                'mode': 'live_write' if live_write else 'dry_run',
                'lead_id': lead_id,
                'contact_id': int(analysis.get('matched_contact_id') or 0),
                'phone': _safe_text(analysis.get('phone')),
                'verdict': _safe_text(analysis.get('close_verdict')),
                'risk': _safe_text(analysis.get('premature_close_risk')),
                'status': '',
                'reason': '',
                'updated_fields': [],
                'skipped_fields': [],
                'unchanged_fields': [],
                'preview_analysis': {},
                'preview_payload': {},
            }
            try:
                if not live_write:
                    preview_payload = _prepare_writeback_payload(analysis)
                    report_row['status'] = 'dry_run'
                    report_row['reason'] = 'live_write_not_confirmed'
                    report_row['updated_fields'] = list(preview_payload.keys())
                    report_row['preview_analysis'] = {
                        'matched_lead_id': lead_id,
                        'close_verdict': report_row['verdict'],
                        'premature_close_risk': report_row['risk'],
                        'recommended_next_step': _safe_text(analysis.get('recommended_next_step')),
                        'follow_up_due_at': _safe_text(analysis.get('follow_up_due_at')),
                    }
                    report_row['preview_payload'] = preview_payload
                    report_rows.append(report_row)
                    continue

                result = write_analysis_to_lead(session, analysis=analysis)
                session.commit()
                report_row['status'] = _safe_text(result.get('status')) or 'written'
                report_row['reason'] = _safe_text(result.get('reason'))
                report_row['updated_fields'] = result.get('updated_fields') or []
                report_row['skipped_fields'] = result.get('skipped_fields') or []
                report_row['unchanged_fields'] = result.get('unchanged_fields') or []
            except Exception as exc:
                session.rollback()
                report_row['status'] = 'failed'
                report_row['reason'] = str(exc)
            report_rows.append(report_row)
            if index % 25 == 0 or index == total:
                written = sum(1 for row in report_rows if row['status'] == 'written')
                dry_run = sum(1 for row in report_rows if row['status'] == 'dry_run')
                skipped = sum(1 for row in report_rows if row['status'] == 'skipped')
                failed = sum(1 for row in report_rows if row['status'] == 'failed')
                print(f'[{index}/{total}] written={written} dry_run={dry_run} skipped={skipped} failed={failed}', flush=True)
    finally:
        session.close()

    summary = {
        'run_id': run_id,
        'mode': 'live_write' if live_write else 'dry_run',
        'live_write': live_write,
        'queue_summary': queue_summary,
        'previous_written_leads': len(previous_written),
        'actionable_candidates': len(actionable),
        'written': sum(1 for row in report_rows if row['status'] == 'written'),
        'dry_run': sum(1 for row in report_rows if row['status'] == 'dry_run'),
        'skipped': sum(1 for row in report_rows if row['status'] == 'skipped'),
        'failed': sum(1 for row in report_rows if row['status'] == 'failed'),
        'report_dir': str(run_dir),
    }

    (run_dir / 'deal_writeback_report.json').write_text(json.dumps({'summary': summary, 'rows': report_rows}, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.DataFrame(report_rows).to_csv(run_dir / 'deal_writeback_report.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(report_rows).to_excel(run_dir / 'deal_writeback_report.xlsx', index=False)
    (run_dir / 'deal_writeback_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary['failed'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
