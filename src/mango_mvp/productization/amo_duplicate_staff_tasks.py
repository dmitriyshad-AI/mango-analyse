from __future__ import annotations

import csv
import html
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional


AMO_DUPLICATE_STAFF_TASKS_SCHEMA_VERSION = "amo_duplicate_staff_tasks_v1"
DEFAULT_DUPLICATE_PACK_ROOT = Path("stable_runtime/amo_duplicate_resolution_20260511_v1")
DEFAULT_OUT_ROOT = Path("stable_runtime/amo_duplicate_staff_tasks_20260511_v1")

STAFF_TASK_COLUMNS = [
    "task_id",
    "owner",
    "manager",
    "phone",
    "status",
    "priority",
    "sale_probability_percent",
    "suggested_keep_contact_id",
    "candidate_contact_ids",
    "contact_links",
    "lead_links",
    "client_name_hint",
    "latest_call_date",
    "next_step",
    "instruction_ru",
    "post_merge_recheck_required",
    "post_merge_recheck_input_row_id",
]

MANAGER_SUMMARY_COLUMNS = [
    "owner",
    "tasks",
    "high_priority_tasks",
    "phones",
    "candidate_contacts",
    "instruction_ru",
]


def build_amo_duplicate_staff_tasks(
    *,
    duplicate_pack_root: Path = DEFAULT_DUPLICATE_PACK_ROOT,
    out_root: Path = DEFAULT_OUT_ROOT,
    generated_at: Optional[datetime] = None,
) -> Mapping[str, Any]:
    """Build a read-only staff task pack for AMO duplicate cleanup."""

    duplicate_pack_root = duplicate_pack_root.expanduser().resolve(strict=False)
    out_root = out_root.expanduser().resolve(strict=False)
    now = generated_at or datetime.now(timezone.utc)
    out_root.mkdir(parents=True, exist_ok=True)

    queue_rows = _read_csv(duplicate_pack_root / "duplicate_merge_queue.csv")
    candidate_rows = _read_csv(duplicate_pack_root / "candidate_contacts.csv")
    candidates_by_resolution = _candidates_by_resolution(candidate_rows)
    task_rows = [_task_row(row, candidates_by_resolution.get(_safe_text(row.get("resolution_id")), [])) for row in queue_rows]
    manager_rows = _manager_summary_rows(task_rows)

    outputs = {
        "staff_tasks_csv": out_root / "staff_tasks.csv",
        "manager_summary_csv": out_root / "manager_summary.csv",
        "staff_tasks_html": out_root / "staff_tasks.html",
        "summary_json": out_root / "summary.json",
        "readme_md": out_root / "README.md",
    }
    _write_csv(outputs["staff_tasks_csv"], task_rows, STAFF_TASK_COLUMNS)
    _write_csv(outputs["manager_summary_csv"], manager_rows, MANAGER_SUMMARY_COLUMNS)
    outputs["staff_tasks_html"].write_text(_render_html(task_rows, manager_rows, now), encoding="utf-8")

    status_counts = dict(Counter(row["status"] for row in task_rows))
    owner_counts = dict(Counter(row["owner"] for row in task_rows))
    summary = {
        "schema_version": AMO_DUPLICATE_STAFF_TASKS_SCHEMA_VERSION,
        "generated_at": now.isoformat(timespec="seconds"),
        "duplicate_pack_root": str(duplicate_pack_root),
        "out_root": str(out_root),
        "task_rows": len(task_rows),
        "manager_summary_rows": len(manager_rows),
        "candidate_contact_rows": len(candidate_rows),
        "status_counts": status_counts,
        "owner_counts": owner_counts,
        "outputs": {key: str(path) for key, path in outputs.items()},
        "policy": {
            "read_only": True,
            "write_crm": False,
            "write_amo": False,
            "merge_amo_contacts_automatically": False,
            "post_merge_recheck_required": True,
            "fail_closed_until_recheck_passes": True,
        },
        "next_actions": [
            {
                "action": "assign_duplicate_merge_tasks_to_staff",
                "rows": len(task_rows),
                "description_ru": "Передать задачи ответственным: открыть AMO-карточки, объединить реальные дубли, затем запустить post-merge recheck.",
            }
        ],
    }
    _write_json(outputs["summary_json"], summary)
    outputs["readme_md"].write_text(_render_readme(summary), encoding="utf-8")
    return summary


def _task_row(row: Mapping[str, Any], candidates: list[Mapping[str, str]]) -> dict[str, str]:
    manager = _safe_text(row.get("last_call_manager")) or "AMO operator"
    owner = manager if _safe_text(row.get("owner_hint")) == "manager_who_owns_client_context" else "AMO operator / " + manager
    candidate_ids = _safe_text(row.get("all_candidate_contact_ids")) or " | ".join(_safe_text(item.get("candidate_contact_id")) for item in candidates)
    status = _safe_text(row.get("duplicate_resolution_status"))
    source_ids = _safe_text(row.get("source_amo_contact_ids"))
    dry_ids = _safe_text(row.get("dry_run_contact_ids"))
    client_hint = _join_non_empty([row.get("fio_parent"), row.get("fio_child")], " / ")
    instruction = _instruction(row, source_ids=source_ids, dry_ids=dry_ids, candidate_ids=candidate_ids)
    return {
        "task_id": _safe_text(row.get("resolution_id")),
        "owner": owner,
        "manager": manager,
        "phone": _safe_text(row.get("phone")),
        "status": status,
        "priority": _safe_text(row.get("merge_priority")) or _safe_text(row.get("priority")),
        "sale_probability_percent": _safe_text(row.get("sale_probability_percent")),
        "suggested_keep_contact_id": _safe_text(row.get("suggested_keep_contact_id")),
        "candidate_contact_ids": candidate_ids,
        "contact_links": _safe_text(row.get("contact_links")),
        "lead_links": _safe_text(row.get("lead_links")),
        "client_name_hint": client_hint,
        "latest_call_date": _safe_text(row.get("latest_call_date")),
        "next_step": _safe_text(row.get("next_step")),
        "instruction_ru": instruction,
        "post_merge_recheck_required": "yes",
        "post_merge_recheck_input_row_id": _safe_text(row.get("resolution_id")),
    }


def _instruction(row: Mapping[str, Any], *, source_ids: str, dry_ids: str, candidate_ids: str) -> str:
    status = _safe_text(row.get("duplicate_resolution_status"))
    if status == "contact_id_mismatch_requires_operator":
        return (
            "Открыть source AMO contact_id и contact_id из live dry-run, проверить где реальная карточка клиента/семьи, "
            "объединить дубли при необходимости. Если правильная карточка не совпадает с source_id, зафиксировать причину. "
            "После правки запустить post-merge dry-run recheck. "
            f"Source: {source_ids or 'нет'}, dry-run: {dry_ids or 'нет'}, кандидаты: {candidate_ids or 'нет'}."
        )
    return (
        "Открыть все AMO contact links по телефону, убедиться что это один клиент/семья, объединить дубли в одну основную карточку "
        "со сделками и историей. Не писать AI-поля до recheck. После склейки запустить post-merge dry-run recheck. "
        f"Рекомендуемый keep contact_id: {_safe_text(row.get('suggested_keep_contact_id')) or 'нужно выбрать вручную'}; кандидаты: {candidate_ids or 'нет'}."
    )


def _manager_summary_rows(task_rows: list[Mapping[str, str]]) -> list[dict[str, str]]:
    grouped: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in task_rows:
        grouped[_safe_text(row.get("owner")) or "AMO operator"].append(row)
    rows: list[dict[str, str]] = []
    for owner, items in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        rows.append(
            {
                "owner": owner,
                "tasks": str(len(items)),
                "high_priority_tasks": str(sum(1 for row in items if _safe_text(row.get("priority")) == "high")),
                "phones": " | ".join(_safe_text(row.get("phone")) for row in items),
                "candidate_contacts": " | ".join(_safe_text(row.get("candidate_contact_ids")) for row in items),
                "instruction_ru": "Разобрать все телефоны этого owner, объединить AMO-дубли и вернуть на post-merge recheck; live-write до recheck запрещен.",
            }
        )
    return rows


def _render_html(task_rows: list[Mapping[str, str]], manager_rows: list[Mapping[str, str]], now: datetime) -> str:
    cards = "\n".join(_render_card(row) for row in task_rows)
    manager_table = "\n".join(
        "<tr>"
        f"<td>{html.escape(row['owner'])}</td>"
        f"<td>{html.escape(row['tasks'])}</td>"
        f"<td>{html.escape(row['phones'])}</td>"
        f"<td>{html.escape(row['instruction_ru'])}</td>"
        "</tr>"
        for row in manager_rows
    )
    return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title>AMO duplicate staff tasks</title>
  <style>
    body {{ margin:0; padding:28px; background:#f5f7fb; color:#111827; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    h1 {{ margin:0 0 6px; }} .meta {{ color:#667085; margin-bottom:18px; }}
    table {{ width:100%; border-collapse:collapse; background:#fff; margin:0 0 18px; }}
    th,td {{ border:1px solid #d0d5dd; padding:9px; text-align:left; vertical-align:top; }} th {{ background:#1f2937; color:white; }}
    .card {{ background:#fff; border:1px solid #d0d5dd; border-radius:14px; margin:0 0 14px; padding:16px; box-shadow:0 8px 24px rgba(17,24,39,.06); }}
    .phone {{ font-size:20px; font-weight:800; }} .status {{ color:#b42318; font-weight:700; }}
    .grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; margin:12px 0; }}
    .box {{ background:#f9fafb; border:1px solid #eaecf0; border-radius:10px; padding:10px; white-space:pre-wrap; }}
    .label {{ color:#667085; font-size:12px; text-transform:uppercase; font-weight:700; }}
    a {{ color:#0b5cab; }}
    @media(max-width:900px) {{ .grid {{ grid-template-columns:1fr; }} body {{ padding:14px; }} }}
  </style>
</head>
<body>
  <h1>AMO duplicate staff tasks</h1>
  <div class="meta">Сгенерировано: {html.escape(now.isoformat(timespec='seconds'))}. Read-only пакет; автоматической склейки и записи нет.</div>
  <h2>Сводка по ответственным</h2>
  <table><thead><tr><th>Ответственный</th><th>Задач</th><th>Телефоны</th><th>Инструкция</th></tr></thead><tbody>{manager_table}</tbody></table>
  <h2>Задачи</h2>
  {cards}
</body>
</html>
"""


def _render_card(row: Mapping[str, str]) -> str:
    contact_links = _links(row.get("contact_links"))
    lead_links = _links(row.get("lead_links"))
    return f"""<section class="card">
  <div class="phone">{html.escape(_safe_text(row.get('phone')))}</div>
  <div class="status">{html.escape(_safe_text(row.get('status')))}</div>
  <div class="grid">
    <div class="box"><div class="label">Ответственный</div>{html.escape(_safe_text(row.get('owner')))}</div>
    <div class="box"><div class="label">Клиент</div>{html.escape(_safe_text(row.get('client_name_hint')))}</div>
    <div class="box"><div class="label">Приоритет</div>{html.escape(_safe_text(row.get('priority')))} / {html.escape(_safe_text(row.get('sale_probability_percent')))}%</div>
    <div class="box"><div class="label">Контакты AMO</div>{contact_links}</div>
    <div class="box"><div class="label">Сделки</div>{lead_links}</div>
    <div class="box"><div class="label">Следующий шаг</div>{html.escape(_safe_text(row.get('next_step')))}</div>
  </div>
  <div class="box"><div class="label">Что сделать</div>{html.escape(_safe_text(row.get('instruction_ru')))}</div>
</section>"""


def _links(value: Any) -> str:
    links = [_safe_text(part) for part in _safe_text(value).splitlines() if _safe_text(part)]
    if not links:
        return ""
    return "<br>".join(f'<a href="{html.escape(link)}">{html.escape(link)}</a>' for link in links)


def _candidates_by_resolution(rows: list[Mapping[str, str]]) -> dict[str, list[Mapping[str, str]]]:
    result: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        result[_safe_text(row.get("resolution_id"))].append(row)
    return result


def _join_non_empty(values: list[Any], sep: str) -> str:
    return sep.join(_safe_text(value) for value in values if _safe_text(value))


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _write_csv(path: Path, rows: list[Mapping[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _render_readme(summary: Mapping[str, Any]) -> str:
    return f"""# AMO duplicate staff tasks

Read-only task pack for staff/operator duplicate cleanup.

- Task rows: `{summary.get('task_rows')}`
- Manager summary rows: `{summary.get('manager_summary_rows')}`
- Candidate contact rows: `{summary.get('candidate_contact_rows')}`

This pack does not merge contacts automatically and does not write CRM fields.
Every row remains blocked for live writeback until manual merge/mismatch resolution and a green post-merge dry-run recheck.
"""
