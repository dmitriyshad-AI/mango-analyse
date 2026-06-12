#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mango_mvp.amocrm_runtime.tallanto_api import (  # noqa: E402
    TallantoApiClient,
    TallantoApiConfig,
    TallantoApiError,
)


ENV_PATH = PROJECT_ROOT / "mango_tallanto_transfer" / ".env.private"
OUT_DIR = PROJECT_ROOT / "Финансовая модель" / "03_выгрузки_tallanto" / "och_lsh"

COURSE_FIELDS = [
    "name",
    "description",
    "subject_name",
    "date_start",
    "date_finish",
    "program_name",
    "filial",
    "status",
    "course_number_seats",
    "course_tags",
    "course_type",
]
REL_FIELDS = [
    "most_courses_id",
    "contact_id",
    "most_courses_contacts_status",
    "seller_id",
    "notice",
    "date_entry",
    "date_modified",
]
CONTACT_FIELDS = [
    "first_name",
    "last_name",
    "title",
    "phone_mobile",
    "email1",
    "type_client_c",
    "filial",
    "discount",
    "discount_type",
    "discount_comment",
    "contact_notice",
    "tags",
]
FINANCE_FIELDS = [
    "name",
    "description",
    "type",
    "direction",
    "cost",
    "date_payment",
    "contact_id",
    "account_id",
    "account_name",
    "most_abonements_id",
    "most_class_id",
    "invoice_id",
    "filial",
    "print_check_status",
    "print_refund_status",
    "school_out_type",
    "tags",
]
ABONEMENT_FIELDS = [
    "name",
    "description",
    "contact_id",
    "type",
    "form",
    "category",
    "rate",
    "duration",
    "start_date",
    "finish_date",
    "cost",
    "num_visit",
    "num_visit_left",
    "filial",
    "invoice_id",
    "discount",
    "discount_comment",
    "internal_notice",
    "tags",
    "tags2",
]

PAYMENT_METHODS_DEFAULT = ["Альфа-банк", "Webmoney", "СБП", "Единая касса", "Наличные"]
DATE_FROM = date(2025, 6, 1)
DATE_TO = date(2026, 9, 30)


def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def client_from_env() -> TallantoApiClient:
    values = {**load_env(ENV_PATH), **os.environ}
    return TallantoApiClient(
        TallantoApiConfig(
            base_url=values["CRM_TALLANTO_BASE_URL"],
            api_token=values["CRM_TALLANTO_API_TOKEN"],
            rest_path=values.get("CRM_TALLANTO_STUDENT_PATH", "/service/api/rest.php"),
        )
    )


def norm(value: Any) -> str:
    return str(value or "").replace("ё", "е").lower()


def text_blob(*records: dict[str, Any] | None) -> str:
    parts: list[str] = []
    for record in records:
        if not record:
            continue
        for value in record.values():
            if value is not None:
                parts.append(str(value))
    return norm(" | ".join(parts))


def parse_amount(value: Any) -> float:
    raw = str(value or "").replace("\u00a0", "").replace(" ", "").replace(",", ".")
    try:
        return float(raw)
    except ValueError:
        return 0.0


def parse_day(value: Any) -> date | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", raw)
    if match:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    match = re.search(r"(\d{2})\.(\d{2})\.(\d{4})", raw)
    if match:
        return date(int(match.group(3)), int(match.group(2)), int(match.group(1)))
    return None


def period_from_date(raw: Any) -> str:
    parsed = parse_day(raw)
    return parsed.strftime("%Y-%m") if parsed else ""


def date_iso(raw: Any) -> str:
    parsed = parse_day(raw)
    return parsed.isoformat() if parsed else str(raw or "")[:10]


def round_money(value: float) -> int | float:
    if abs(value - round(value)) < 0.005:
        return int(round(value))
    return round(value, 2)


def is_selected_course(course: dict[str, Any]) -> bool:
    name = norm(course.get("name"))
    if "лш 2026" not in name:
        return False
    if "очно" not in name:
        return False
    excluded = ["лвш", "лист ожидания", "онлайн", "шд", "дизайн", "пробн"]
    return not any(marker in name for marker in excluded)


def is_related_but_excluded(course: dict[str, Any]) -> bool:
    name = norm(course.get("name"))
    if "лш 2026" not in name:
        return False
    return not is_selected_course(course)


def event_from_course(course: dict[str, Any]) -> dict[str, Any]:
    name = str(course.get("name") or "")
    name_n = norm(name)
    start = parse_day(course.get("date_start"))
    finish = parse_day(course.get("date_finish"))
    brand = "ФОТОН" if "фотон" in name_n else "УНПК"
    legal = "ЦДПО ФОТОН" if brand == "ФОТОН" else "АНО УНПК МФТИ"
    location = "МФТИ" if "мфти" in name_n or norm(course.get("filial")) == "мфти" else "Москва"
    date_code = start.strftime("%d%m") if start else "DATE"
    brand_code = "FOTON" if brand == "ФОТОН" else "UNPK"
    loc_code = "MFTI" if location == "МФТИ" else "MSK"
    suffix = "_FM" if re.search(r"(^|\W)фм($|\W)", name_n) else ""
    event_id = f"OCH_LSH_{loc_code}_{brand_code}_{date_code}{suffix}"
    title = f"ЛШ очная {location} {start.strftime('%d.%m') if start else ''}-{finish.strftime('%d.%m') if finish else ''} {brand}".strip()
    if suffix:
        title += " ФМ"
    return {
        "event_id": event_id,
        "brand": brand,
        "legal_entity": legal,
        "location": location,
        "title": title,
        "course_id": course.get("id"),
        "course_name": name,
        "date_start": start.isoformat() if start else "",
        "date_finish": finish.isoformat() if finish else "",
        "filial": course.get("filial") or "",
    }


def fio(contact: dict[str, Any] | None) -> str:
    if not contact:
        return ""
    parts = [
        str(contact.get("last_name") or "").strip(),
        str(contact.get("first_name") or "").strip(),
        str(contact.get("title") or "").strip(),
    ]
    result: list[str] = []
    seen: set[str] = set()
    for part in parts:
        if not part or norm(part) in seen:
            continue
        seen.add(norm(part))
        result.append(part)
    return " ".join(result)


def payment_method(raw_value: Any) -> str:
    raw = norm(raw_value)
    if "альфа" in raw or "alfa" in raw or "alfabank" in raw:
        return "Альфа-банк"
    if "webmoney" in raw or "веб" in raw:
        return "Webmoney"
    if "сбп" in raw or "sbp" in raw:
        return "СБП"
    if "единая" in raw or "kassa" in raw or "касс" in raw:
        return "Единая касса"
    if "нал" in raw or "cash" in raw:
        return "Наличные"
    return str(raw_value or "").strip() or "Не указан"


def has_lsh_marker(blob: str) -> bool:
    return any(marker in blob for marker in ["лш", "летняя школ", "летн. школ", "летней школ"])


def has_strong_exclusion(blob: str) -> bool:
    return any(marker in blob for marker in ["лвш", "выезд", "менделеево", "онлайн"])


def event_score(event: dict[str, Any], blob: str) -> int:
    score = 0
    brand = event["brand"]
    location = event["location"]
    if brand == "ФОТОН" and "фотон" in blob:
        score += 3
    if brand == "УНПК" and ("унпк" in blob or "мфти" in blob):
        score += 2
    if location == "МФТИ" and "мфти" in blob:
        score += 3
    if location == "Москва" and ("москва" in blob or "сретен" in blob):
        score += 3
    for raw_date in [event.get("date_start"), event.get("date_finish")]:
        parsed = parse_day(raw_date)
        if not parsed:
            continue
        tokens = {
            parsed.strftime("%d.%m"),
            parsed.strftime("%-d.%m") if hasattr(parsed, "strftime") else "",
            parsed.strftime("%d.%m.%Y"),
        }
        score += 2 if any(token and token in blob for token in tokens) else 0
    if event["event_id"].endswith("_FM") and re.search(r"(^|\W)фм($|\W)", blob):
        score += 2
    return score


def assign_event(
    record: dict[str, Any],
    contact_events: list[dict[str, Any]],
    abonement: dict[str, Any] | None,
) -> tuple[str | None, str]:
    blob = text_blob(record, abonement)
    if not has_lsh_marker(blob):
        return None, "нет признака ЛШ в названии/абонементе"
    if has_strong_exclusion(blob):
        return None, "исключено: текст похож на ЛВШ/выездную/онлайн оплату"
    if not contact_events:
        return None, "контакт не найден в выбранных очных ЛШ"
    scored = sorted(((event_score(event, blob), event) for event in contact_events), reverse=True, key=lambda item: item[0])
    best_score, best_event = scored[0]
    if len(contact_events) == 1 and best_score >= 0:
        return best_event["event_id"], "единственная очная ЛШ ученика + признак ЛШ"
    if best_score >= 2 and (len(scored) == 1 or best_score > scored[1][0]):
        return best_event["event_id"], f"совпали признаки смены, score={best_score}"
    return None, "неоднозначная очная ЛШ для ученика"


def iter_with_pause(
    client: TallantoApiClient,
    *,
    module: str,
    select_fields: list[str],
    field_values: dict[str, Any] | None = None,
    query: str | None = None,
    order_by: str | None = None,
    max_records: int | None = None,
) -> list[dict[str, Any]]:
    records = client.iter_entry_list(
        module=module,
        select_fields=select_fields,
        field_values=field_values,
        query=query,
        order_by=order_by,
        max_records=max_records,
    )
    time.sleep(0.08)
    return records


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_tsv(path: Path, rows: list[list[Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerows(rows)


def choose_event_plan_prices(events: list[dict[str, Any]], abonements_by_event: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for event in events:
        costs = [parse_amount(row.get("cost")) for row in abonements_by_event.get(event["event_id"], [])]
        costs = [cost for cost in costs if cost > 0]
        if not costs:
            result[event["event_id"]] = 0.0
            continue
        rounded = [round(cost / 100) * 100 for cost in costs]
        counts = Counter(rounded)
        repeated = [cost for cost, count in counts.items() if count >= 2]
        result[event["event_id"]] = float(max(repeated or rounded))
    return result


def build_wide_rows(
    events: list[dict[str, Any]],
    participant_rows: list[dict[str, Any]],
    payment_methods: list[str],
) -> list[list[Any]]:
    block_width = 3 + len(payment_methods) + 6 + 1
    rows_by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in participant_rows:
        rows_by_event[row["событие_id"]].append(row)
    for event_id in rows_by_event:
        rows_by_event[event_id].sort(key=lambda row: norm(row.get("фио")))
    max_len = max([len(rows_by_event[event["event_id"]]) for event in events] + [0])

    total_cols = block_width * len(events)
    sheet_rows: list[list[Any]] = [[""] * total_cols for _ in range(max_len + 4)]
    for event_index, event in enumerate(events):
        start = event_index * block_width
        end = start + block_width - 2
        sheet_rows[0][start + 1] = f"{event['title']} ({event['event_id']})"
        sheet_rows[1][start + 1] = "Участников"
        sheet_rows[1][start + 2] = len(rows_by_event[event["event_id"]])
        sheet_rows[1][start + 3] = "Поступления"
        sheet_rows[1][start + 4] = round_money(
            sum(parse_amount(row.get("общий_итог")) for row in rows_by_event[event["event_id"]])
        )
        sheet_rows[1][start + 5] = "Нет оплаты"
        sheet_rows[1][start + 6] = sum(1 for row in rows_by_event[event["event_id"]] if row.get("статус") == "нет оплаты")
        headers = [
            "ученик_id",
            "№",
            "ФИО",
            *payment_methods,
            "Общий итог",
            "К учету",
            "Ожидаемая цена",
            "Остаток",
            "Статус",
            "Комментарий",
        ]
        sheet_rows[2][start : start + len(headers)] = headers
        sheet_rows[3][start + 2] = "Суммы"
        for idx, method in enumerate(payment_methods):
            sheet_rows[3][start + 3 + idx] = round_money(
                sum(parse_amount(row.get(method, 0)) for row in rows_by_event[event["event_id"]])
            )
        total_idx = start + 3 + len(payment_methods)
        sheet_rows[3][total_idx] = round_money(
            sum(parse_amount(row.get("общий_итог")) for row in rows_by_event[event["event_id"]])
        )
        sheet_rows[3][total_idx + 1] = round_money(
            sum(parse_amount(row.get("к_учету")) for row in rows_by_event[event["event_id"]])
        )
        sheet_rows[3][total_idx + 2] = round_money(
            sum(parse_amount(row.get("ожидаемая_цена", row.get("плановая_цена"))) for row in rows_by_event[event["event_id"]])
        )
        sheet_rows[3][total_idx + 3] = round_money(
            sum(parse_amount(row.get("остаток")) for row in rows_by_event[event["event_id"]])
        )
        for row_index, participant in enumerate(rows_by_event[event["event_id"]], start=4):
            values = [
                participant.get("ученик_id", ""),
                row_index - 3,
                participant.get("фио", ""),
            ]
            values.extend(participant.get(method, 0) or "" for method in payment_methods)
            values.extend(
                [
                    participant.get("общий_итог", 0) or "",
                    participant.get("к_учету", 0) or "",
                    participant.get("ожидаемая_цена", participant.get("плановая_цена", "")) or "",
                    participant.get("остаток", 0) if participant.get("остаток") != "" else "",
                    participant.get("статус", ""),
                    participant.get("комментарий", ""),
                ]
            )
            sheet_rows[row_index][start : start + len(values)] = values
        if end + 1 < total_cols:
            for row in sheet_rows:
                row[end + 1] = ""
    return sheet_rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = client_from_env()

    print("fetch courses", file=sys.stderr, flush=True)
    courses = iter_with_pause(
        client,
        module="most_courses",
        select_fields=COURSE_FIELDS,
        query='DATE(most_courses.date_start) BETWEEN "2026-05-01" AND "2026-09-15"',
        order_by="most_courses.date_start ASC",
    )
    selected_courses = [course for course in courses if is_selected_course(course)]
    excluded_courses = [course for course in courses if is_related_but_excluded(course)]
    events = [event_from_course(course) for course in selected_courses]
    event_by_course_id = {event["course_id"]: event for event in events}
    event_by_id = {event["event_id"]: event for event in events}
    print(
        f"courses fetched={len(courses)} selected={len(events)} excluded_related={len(excluded_courses)}",
        file=sys.stderr,
        flush=True,
    )

    relations: list[dict[str, Any]] = []
    relation_errors: list[dict[str, Any]] = []
    for event in events:
        print(f"fetch relations {event['event_id']} {event['course_name']}", file=sys.stderr, flush=True)
        try:
            course_relations = iter_with_pause(
                client,
                module="CoursesContactsRelationship",
                select_fields=REL_FIELDS,
                field_values={"most_courses_id": event["course_id"]},
                max_records=1000,
            )
        except TallantoApiError as exc:
            relation_errors.append(
                {
                    "event_id": event["event_id"],
                    "course_id": event["course_id"],
                    "course_name": event["course_name"],
                    "error": str(exc),
                }
            )
            course_relations = []
        for relation in course_relations:
            relation = dict(relation)
            relation.update(
                {
                    "event_id": event["event_id"],
                    "brand": event["brand"],
                    "event_title": event["title"],
                    "course_name": event["course_name"],
                }
            )
            relations.append(relation)
        print(f"relations total={len(relations)}", file=sys.stderr, flush=True)

    if relation_errors:
        details = "; ".join(f"{row['event_id']} ({row['course_id']}): {row['error']}" for row in relation_errors)
        raise TallantoApiError(f"Failed to fetch course participants; refusing to build incomplete model: {details}")

    participant_pairs: set[tuple[str, str]] = set()
    contact_ids: set[str] = set()
    for relation in relations:
        contact_id = str(relation.get("contact_id") or "").strip()
        event_id = str(relation.get("event_id") or "").strip()
        if not contact_id or not event_id or relation.get("_error"):
            continue
        participant_pairs.add((event_id, contact_id))
        contact_ids.add(contact_id)
    print(
        f"participant_pairs={len(participant_pairs)} contacts={len(contact_ids)}",
        file=sys.stderr,
        flush=True,
    )

    contacts: dict[str, dict[str, Any]] = {}
    for idx, contact_id in enumerate(sorted(contact_ids), start=1):
        try:
            contacts[contact_id] = client.contact_by_id(contact_id, select_fields=CONTACT_FIELDS)
        except TallantoApiError as exc:
            contacts[contact_id] = {"id": contact_id, "_error": str(exc)}
        if idx % 5 == 0:
            print(f"contacts {idx}/{len(contact_ids)}", file=sys.stderr, flush=True)
        time.sleep(0.05)

    contact_events: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event_id, contact_id in sorted(participant_pairs):
        contact_events[contact_id].append(event_by_id[event_id])

    finances: list[dict[str, Any]] = []
    abonements: list[dict[str, Any]] = []
    for idx, contact_id in enumerate(sorted(contact_ids), start=1):
        try:
            finances.extend(
                iter_with_pause(
                    client,
                    module="most_finances",
                    select_fields=FINANCE_FIELDS,
                    field_values={"contact_id": contact_id},
                    max_records=700,
                )
            )
        except TallantoApiError as exc:
            finances.append({"contact_id": contact_id, "_error": str(exc)})
        try:
            abonements.extend(
                iter_with_pause(
                    client,
                    module="most_abonements",
                    select_fields=ABONEMENT_FIELDS,
                    field_values={"contact_id": contact_id},
                    max_records=700,
                )
            )
        except TallantoApiError as exc:
            abonements.append({"contact_id": contact_id, "_error": str(exc)})
        if idx % 5 == 0:
            print(
                f"finance+abonements {idx}/{len(contact_ids)} finance_rows={len(finances)} abonement_rows={len(abonements)}",
                file=sys.stderr,
                flush=True,
            )

    abonement_by_id = {str(row.get("id")): row for row in abonements if row.get("id")}

    assigned_abonements: list[dict[str, Any]] = []
    for abonement in abonements:
        contact_id = str(abonement.get("contact_id") or "").strip()
        event_id, reason = assign_event(abonement, contact_events.get(contact_id, []), None)
        if not event_id:
            continue
        row = dict(abonement)
        row["event_id"] = event_id
        row["assign_reason"] = reason
        assigned_abonements.append(row)
    abonements_by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    abonements_by_contact_event: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in assigned_abonements:
        event_id = row["event_id"]
        contact_id = str(row.get("contact_id") or "").strip()
        abonements_by_event[event_id].append(row)
        abonements_by_contact_event[(event_id, contact_id)].append(row)

    event_plan_prices = choose_event_plan_prices(events, abonements_by_event)

    confirmed_payments: list[dict[str, Any]] = []
    excluded_payments: list[dict[str, Any]] = []
    ambiguous_payments: list[dict[str, Any]] = []
    seen_finance_ids: set[str] = set()
    for finance in finances:
        finance_id = str(finance.get("id") or "").strip()
        if finance_id and finance_id in seen_finance_ids:
            continue
        if finance_id:
            seen_finance_ids.add(finance_id)
        contact_id = str(finance.get("contact_id") or "").strip()
        amount = parse_amount(finance.get("cost"))
        direction = norm(finance.get("direction"))
        payment_day = parse_day(finance.get("date_payment"))
        if amount <= 0 or direction not in {"in", "приход", "income"}:
            continue
        if payment_day and not (DATE_FROM <= payment_day <= DATE_TO):
            continue
        abonement = abonement_by_id.get(str(finance.get("most_abonements_id") or ""))
        event_id, reason = assign_event(finance, contact_events.get(contact_id, []), abonement)
        if not event_id:
            candidate = dict(finance)
            candidate["exclude_reason"] = reason
            if "неоднознач" in reason:
                ambiguous_payments.append(candidate)
            elif has_lsh_marker(text_blob(finance, abonement)):
                excluded_payments.append(candidate)
            continue
        event = event_by_id[event_id]
        contact = contacts.get(contact_id, {})
        method = payment_method(finance.get("type"))
        confirmed_payments.append(
            {
                "дата": date_iso(finance.get("date_payment")),
                "период": period_from_date(finance.get("date_payment")),
                "бренд": event["brand"],
                "юрлицо": event["legal_entity"],
                "формат": "Очная ЛШ",
                "событие_id": event_id,
                "продукт": event["title"],
                "ученик_id": contact_id,
                "ученик_фио": fio(contact),
                "сумма_валовая": round_money(amount),
                "метод_оплаты": method,
                "скидка_сумма": 0,
                "возврат_флаг": "FALSE",
                "tallanto_finance_id": finance_id,
                "tallanto_group_id": event["course_id"],
                "tallanto_abonement_id": finance.get("most_abonements_id") or "",
                "проверка_очной_ЛШ": f"OK: {reason}",
                "комментарий": (
                    f"Tallanto: {finance.get('name') or ''} | method_code={finance.get('type') or ''} "
                    f"| abonement_id={finance.get('most_abonements_id') or ''}"
                ).strip(),
            }
        )

    payments_by_contact_event: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in confirmed_payments:
        payments_by_contact_event[(row["событие_id"], row["ученик_id"])].append(row)

    methods_seen = sorted({row["метод_оплаты"] for row in confirmed_payments})
    payment_methods = PAYMENT_METHODS_DEFAULT + [method for method in methods_seen if method not in PAYMENT_METHODS_DEFAULT]

    participant_rows: list[dict[str, Any]] = []
    price_check_rows: list[dict[str, Any]] = []
    for event in events:
        event_id = event["event_id"]
        plan_price = event_plan_prices.get(event_id, 0.0)
        event_relations = [row for row in relations if row.get("event_id") == event_id and row.get("contact_id")]
        event_relations.sort(key=lambda row: fio(contacts.get(str(row.get("contact_id") or ""))))
        for relation in event_relations:
            contact_id = str(relation.get("contact_id") or "").strip()
            contact = contacts.get(contact_id, {})
            payments = payments_by_contact_event.get((event_id, contact_id), [])
            paid = sum(parse_amount(row.get("сумма_валовая")) for row in payments)
            by_method = {method: 0.0 for method in payment_methods}
            for payment in payments:
                by_method[payment["метод_оплаты"]] = by_method.get(payment["метод_оплаты"], 0.0) + parse_amount(
                    payment.get("сумма_валовая")
                )
            abons = abonements_by_contact_event.get((event_id, contact_id), [])
            abon_costs = [parse_amount(row.get("cost")) for row in abons if parse_amount(row.get("cost")) > 0]
            effective_price = 0.0
            if paid > 0 and abon_costs:
                effective_price = min(abon_costs, key=lambda cost: abs(cost - paid))
            elif abon_costs:
                effective_price = max(abon_costs)
            target_price = effective_price if effective_price > 0 else plan_price
            residual = max((target_price or plan_price) - paid, 0) if paid > 0 else (plan_price or "")

            if paid <= 0:
                status = "нет оплаты"
                note = "Участник есть в группе Tallanto, подтвержденной оплаты очной ЛШ не найдено"
                if abon_costs:
                    note += f"; найден абонемент на {', '.join(str(round_money(cost)) for cost in sorted(set(abon_costs)))}"
            elif effective_price and abs(paid - effective_price) <= 1:
                if plan_price and effective_price < plan_price:
                    discount_pct = (plan_price - effective_price) / plan_price
                    if discount_pct <= 0.25:
                        status = "оплачено со скидкой/ранним бронированием"
                    else:
                        status = "скидка >25%, проверить"
                elif plan_price and effective_price > plan_price * 1.05:
                    status = "переплата/проверить"
                else:
                    status = "оплачено"
                note = "Стоимость абонемента совпадает с оплаченной суммой"
            elif effective_price and paid < effective_price:
                status = "частично до стоимости абонемента"
                note = "Оплата меньше стоимости найденного абонемента"
            elif plan_price and paid >= plan_price:
                status = "оплачено" if paid <= plan_price * 1.05 else "переплата/проверить"
                note = "Оплата не ниже плановой цены"
            else:
                status = "частично/долг"
                note = "Оплата меньше плановой цены; абонемент на фактическую цену не найден"

            row = {
                "событие_id": event_id,
                "бренд": event["brand"],
                "смена": event["title"],
                "группа_tallanto": event["course_name"],
                "tallanto_group_id": event["course_id"],
                "ученик_id": contact_id,
                "фио": fio(contact),
                "статус_записи": relation.get("most_courses_contacts_status") or "",
                "плановая_цена": round_money(plan_price) if plan_price else "",
                "оплачено_валовая": round_money(paid) if paid else 0,
                "остаток_с_учетом_скидки": round_money(residual) if residual != "" else "",
                "статус_с_учетом_скидки": status,
                "типы_оплаты": ", ".join(sorted({payment["метод_оплаты"] for payment in payments})),
                "abonement_costs": ", ".join(str(round_money(cost)) for cost in sorted(set(abon_costs))),
                "проверка": "OK: участник группы Tallanto",
                "комментарий": note,
            }
            participant_rows.append(row)

            wide_row: dict[str, Any] = {
                "событие_id": event_id,
                "ученик_id": contact_id,
                "фио": row["фио"],
                "общий_итог": round_money(paid) if paid else 0,
                "к_учету": round_money(paid) if paid else 0,
                "остаток": row["остаток_с_учетом_скидки"],
                "статус": status,
                "комментарий": note,
            }
            for method in payment_methods:
                wide_row[method] = round_money(by_method.get(method, 0.0)) if by_method.get(method, 0.0) else ""
            price_check_rows.append(
                {
                    "событие_id": event_id,
                    "ученик_id": contact_id,
                    "фио": row["фио"],
                    "плановая_цена": round_money(plan_price) if plan_price else "",
                    "оплачено_валовая": round_money(paid) if paid else 0,
                    "эффективная_цена_по_абонементу": round_money(effective_price) if effective_price else "",
                    "остаток_уточненный": row["остаток_с_учетом_скидки"],
                    "разница_к_прайсу_pct": round((plan_price - effective_price) / plan_price, 4)
                    if plan_price and effective_price
                    else "",
                    "статус_цены": status,
                    "abonement_ids": ", ".join(str(abon.get("id") or "") for abon in abons if abon.get("id")),
                    "abonement_costs": row["abonement_costs"],
                    "abonement_discounts": ", ".join(
                        str(abon.get("discount") or "") for abon in abons if abon.get("discount")
                    ),
                    "abonement_comments": " | ".join(
                        str(abon.get("discount_comment") or "") for abon in abons if abon.get("discount_comment")
                    ),
                    "price_check_note": note,
                    "типы_оплаты": row["типы_оплаты"],
                }
            )
            row.update(wide_row)

    participants_by_event = Counter(row["событие_id"] for row in participant_rows)
    payments_by_event = Counter(row["событие_id"] for row in confirmed_payments)
    sum_by_event: dict[str, float] = defaultdict(float)
    for row in confirmed_payments:
        sum_by_event[row["событие_id"]] += parse_amount(row["сумма_валовая"])
    status_counts = Counter(row["статус_с_учетом_скидки"] for row in participant_rows)

    summary = {
        "selected_courses_count": len(selected_courses),
        "selected_events": events,
        "excluded_related_courses": [
            {
                "id": course.get("id"),
                "name": course.get("name"),
                "date_start": course.get("date_start"),
                "date_finish": course.get("date_finish"),
                "filial": course.get("filial"),
            }
            for course in excluded_courses
        ],
        "participants": len(participant_rows),
        "participants_by_event": dict(participants_by_event),
        "confirmed_payments": len(confirmed_payments),
        "confirmed_sum": round_money(sum(parse_amount(row["сумма_валовая"]) for row in confirmed_payments)),
        "payments_by_event": dict(payments_by_event),
        "sum_by_event": {key: round_money(value) for key, value in sorted(sum_by_event.items())},
        "payments_by_method": dict(Counter(row["метод_оплаты"] for row in confirmed_payments)),
        "status_counts": dict(status_counts),
        "event_plan_prices": {key: round_money(value) if value else "" for key, value in event_plan_prices.items()},
        "abonement_cost_distribution_by_event": {
            event_id: dict(
                sorted(
                    Counter(round_money(parse_amount(row.get("cost"))) for row in rows if parse_amount(row.get("cost")) > 0).items()
                )
            )
            for event_id, rows in abonements_by_event.items()
        },
        "ambiguous_lsh_payment_candidates": len(ambiguous_payments),
        "excluded_lsh_payment_candidates": len(excluded_payments),
    }

    payment_fields = [
        "дата",
        "период",
        "бренд",
        "юрлицо",
        "формат",
        "событие_id",
        "продукт",
        "ученик_id",
        "ученик_фио",
        "сумма_валовая",
        "метод_оплаты",
        "скидка_сумма",
        "возврат_флаг",
        "tallanto_finance_id",
        "tallanto_group_id",
        "tallanto_abonement_id",
        "проверка_очной_ЛШ",
        "комментарий",
    ]
    participant_fields = [
        "событие_id",
        "бренд",
        "смена",
        "группа_tallanto",
        "tallanto_group_id",
        "ученик_id",
        "фио",
        "статус_записи",
        "плановая_цена",
        "оплачено_валовая",
        "остаток_с_учетом_скидки",
        "статус_с_учетом_скидки",
        "типы_оплаты",
        "abonement_costs",
        "проверка",
        "комментарий",
    ]
    price_fields = [
        "событие_id",
        "ученик_id",
        "фио",
        "плановая_цена",
        "оплачено_валовая",
        "эффективная_цена_по_абонементу",
        "остаток_уточненный",
        "разница_к_прайсу_pct",
        "статус_цены",
        "abonement_ids",
        "abonement_costs",
        "abonement_discounts",
        "abonement_comments",
        "price_check_note",
        "типы_оплаты",
    ]
    write_csv(OUT_DIR / "och_lsh_courses_candidates.csv", courses, ["id", *COURSE_FIELDS])
    write_csv(OUT_DIR / "och_lsh_courses_selected.csv", selected_courses, ["id", *COURSE_FIELDS])
    write_csv(OUT_DIR / "och_lsh_relations.csv", relations, ["id", *REL_FIELDS, "event_id", "brand", "event_title", "course_name"])
    write_csv(OUT_DIR / "och_lsh_confirmed_payments_import.csv", confirmed_payments, payment_fields)
    write_csv(OUT_DIR / "och_lsh_participants_import.csv", participant_rows, participant_fields)
    write_csv(OUT_DIR / "och_lsh_participants_price_check.csv", price_check_rows, price_fields)
    write_csv(OUT_DIR / "och_lsh_assigned_abonements.csv", assigned_abonements, ["id", *ABONEMENT_FIELDS, "event_id", "assign_reason"])
    write_csv(OUT_DIR / "och_lsh_ambiguous_payments.csv", ambiguous_payments, ["id", *FINANCE_FIELDS, "exclude_reason"])
    write_csv(OUT_DIR / "och_lsh_excluded_payment_candidates.csv", excluded_payments, ["id", *FINANCE_FIELDS, "exclude_reason"])
    (OUT_DIR / "och_lsh_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "och_lsh_raw_courses.json").write_text(json.dumps(courses, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "och_lsh_raw_finances.json").write_text(json.dumps(finances, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "och_lsh_raw_abonements.json").write_text(json.dumps(abonements, ensure_ascii=False, indent=2), encoding="utf-8")

    participants_tsv = [participant_fields] + [[row.get(field, "") for field in participant_fields] for row in participant_rows]
    payments_tsv = [payment_fields] + [[row.get(field, "") for field in payment_fields] for row in confirmed_payments]
    wide_rows = build_wide_rows(events, participant_rows, payment_methods)
    write_tsv(OUT_DIR / "och_lsh_participants_paste.tsv", participants_tsv)
    write_tsv(OUT_DIR / "och_lsh_payments_paste.tsv", payments_tsv)
    write_tsv(OUT_DIR / "och_lsh_wide_sheet.tsv", wide_rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
