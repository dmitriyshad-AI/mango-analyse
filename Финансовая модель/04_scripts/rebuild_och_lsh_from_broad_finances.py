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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mango_mvp.amocrm_runtime.tallanto_api import TallantoApiClient, TallantoApiConfig, TallantoApiError  # noqa: E402
from fetch_och_lsh_tallanto import (  # noqa: E402
    ABONEMENT_FIELDS,
    CONTACT_FIELDS,
    FINANCE_FIELDS,
    OUT_DIR,
    PAYMENT_METHODS_DEFAULT,
    build_wide_rows,
    date_iso,
    event_score,
    fio,
    iter_with_pause,
    parse_amount,
    parse_day,
    period_from_date,
    round_money,
    text_blob,
    write_csv,
    write_tsv,
)


ENV_PATH = PROJECT_ROOT / "mango_tallanto_transfer" / ".env.private"
CURRENT_FINANCE_QUERY = (
    'DATE(most_finances.date_payment) BETWEEN "2026-01-01" AND "2026-06-30" '
    'AND most_finances.name LIKE "%ЛШ%"'
)
MISSING_FIO_LABEL = "(ФИО не заполнено в Tallanto)"


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


def display_fio(value: Any) -> str:
    return str(value or "").strip() or MISSING_FIO_LABEL


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def purpose_blob(record: dict[str, Any] | None, abonement: dict[str, Any] | None = None) -> str:
    parts: list[str] = []
    for source in [record, abonement]:
        if not source:
            continue
        for key in ["name", "description", "tags", "tags2", "discount_comment", "internal_notice"]:
            value = source.get(key)
            if value is not None:
                parts.append(str(value))
    return norm(" | ".join(parts))


def has_lsh_marker(blob: str) -> bool:
    return any(marker in blob for marker in ["лш", "летняя школ", "летней школ", "летн. школ"])


def is_refund_like(blob: str) -> bool:
    return any(marker in blob for marker in ["возврат", "возвращение", "refund"])


def is_current_2026_lsh(
    record: dict[str, Any],
    abonement: dict[str, Any] | None,
    *,
    contact_events: list[dict[str, Any]],
) -> tuple[bool, str]:
    blob = purpose_blob(record, abonement)
    if not has_lsh_marker(blob):
        return False, "нет признака ЛШ в назначении"
    if is_refund_like(blob):
        return False, "исключено: похоже на возврат/возвращение средств"
    if any(marker in blob for marker in ["лвш", "звш", "онлайн", "менделеево"]):
        return False, "исключено: ЛВШ/ЗВШ/онлайн/Менделеево"
    if "выезд" in blob and "очно" not in blob:
        return False, "исключено: выездная школа"
    if "2026" in blob or re.search(r"лш\s*-?\s*26", blob):
        return True, "текущий цикл: 2026/ЛШ-26 в назначении"

    payment_day = parse_day(record.get("date_payment"))
    abonement_day = parse_day((abonement or record).get("start_date"))
    has_current_day = bool(payment_day and payment_day.year == 2026) or bool(abonement_day and abonement_day.year == 2026)
    if has_current_day and "очная" in blob:
        return True, "текущий цикл: generic ЛШ очная с датой/абонементом 2026"
    if has_current_day and contact_events:
        return True, "текущий цикл: ЛШ с датой/абонементом 2026 и есть связь с очной сменой"
    return False, "исключено: похоже на старую generic ЛШ без 2026"


def payment_method(record: dict[str, Any]) -> str:
    translated = str(record.get("type_translated") or "").strip()
    if translated in {"Альфа-банк", "Webmoney", "СБП", "Единая касса", "Наличные"}:
        return translated
    raw = norm(record.get("type"))
    if "alfabank" in raw or "альфа" in raw:
        return "Альфа-банк"
    if "webmoney" in raw:
        return "Webmoney"
    if "sbp" in raw or "сбп" in raw:
        return "СБП"
    if raw == "w1" or "касс" in raw:
        return "Единая касса"
    if "cash" in raw or "нал" in raw:
        return "Наличные"
    if "bank" in raw:
        return "Безналичный расчет"
    return translated or str(record.get("type") or "").strip() or "Не указан"


def assign_event(
    record: dict[str, Any],
    contact_events: list[dict[str, Any]],
    abonement: dict[str, Any] | None,
    events: list[dict[str, Any]],
) -> tuple[str | None, str]:
    blob = purpose_blob(record, abonement)
    if contact_events:
        scored_contact = sorted(
            ((event_score(event, blob), event) for event in contact_events),
            reverse=True,
            key=lambda item: item[0],
        )
        best_contact_score, best_contact_event = scored_contact[0]
        if len(contact_events) == 1:
            return best_contact_event["event_id"], "единственная очная ЛШ ученика; строка участника важнее текста платежа"
        if best_contact_score >= 2 and (
            len(scored_contact) == 1 or best_contact_score > scored_contact[1][0]
        ):
            return best_contact_event["event_id"], f"совпали признаки смены среди групп ученика, score={best_contact_score}"

    scored_all = sorted(((event_score(event, blob), event) for event in events), reverse=True, key=lambda item: item[0])
    best_score, best_event = scored_all[0] if scored_all else (0, None)
    if best_event and best_score >= 5:
        return best_event["event_id"], f"явное название смены, score={best_score}"
    if best_event and best_score >= 2:
        return best_event["event_id"], f"распознано по названию без строки участника, score={best_score}"
    return None, "неоднозначная очная ЛШ"


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


def choose_personal_expected_price(
    *,
    abon_costs: list[float],
    paid: float,
    fallback_event_price: float,
) -> tuple[float, str]:
    if paid > 0 and abon_costs:
        return min(abon_costs, key=lambda cost: abs(cost - paid)), "абонемент, ближайший к оплате"
    if abon_costs:
        return max(abon_costs), "созданный абонемент"
    if paid > 0:
        return paid, "нет абонемента; ожидаемая цена принята равной подтвержденной оплате"
    return 0.0, "нет оплаты и персонального абонемента; ожидаемая цена не определена"


def dedupe_by_id(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        row_id = str(row.get("id") or "").strip()
        signature = row_id or json.dumps(row, ensure_ascii=False, sort_keys=True)
        if signature in seen:
            continue
        seen.add(signature)
        result.append(row)
    return result


def read_json_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def main() -> None:
    use_cached_tallanto = os.environ.get("FIN_MODEL_USE_CACHED_TALLANTO") == "1"
    client = None if use_cached_tallanto else client_from_env()
    summary = json.loads((OUT_DIR / "och_lsh_summary.json").read_text(encoding="utf-8"))
    events = summary["selected_events"]
    event_by_id = {event["event_id"]: event for event in events}

    relation_rows = [row for row in read_csv(OUT_DIR / "och_lsh_relations.csv") if row.get("contact_id")]
    previous_participants = read_csv(OUT_DIR / "och_lsh_participants_import.csv")
    participant_template: dict[tuple[str, str], dict[str, Any]] = {
        (row["событие_id"], row["ученик_id"]): dict(row)
        for row in previous_participants
        if row.get("статус_записи") != "нет строки участника Tallanto"
    }
    fio_by_contact: dict[str, str] = {}
    for row in previous_participants:
        if row.get("фио"):
            fio_by_contact[row["ученик_id"]] = row["фио"]

    contact_events: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in relation_rows:
        event = event_by_id.get(row.get("event_id"))
        if event:
            contact_events[row["contact_id"]].append(event)

    if use_cached_tallanto:
        print("use cached broad current finances", file=sys.stderr, flush=True)
        broad_finances = read_json_rows(OUT_DIR / "och_lsh_current_finances_broad.json")
    else:
        print("fetch broad current finances", file=sys.stderr, flush=True)
        assert client is not None
        broad_finances = iter_with_pause(
            client,
            module="most_finances",
            select_fields=FINANCE_FIELDS,
            query=CURRENT_FINANCE_QUERY,
            order_by="most_finances.date_payment ASC",
            max_records=2000,
        )
    raw_finances = json.loads((OUT_DIR / "och_lsh_raw_finances.json").read_text(encoding="utf-8"))
    finances = dedupe_by_id([*raw_finances, *broad_finances])
    if not use_cached_tallanto:
        (OUT_DIR / "och_lsh_current_finances_broad.json").write_text(
            json.dumps(broad_finances, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(f"broad={len(broad_finances)} merged_finances={len(finances)}", file=sys.stderr, flush=True)

    raw_abonements = json.loads((OUT_DIR / "och_lsh_raw_abonements.json").read_text(encoding="utf-8"))
    cached_broad_abonements = read_json_rows(OUT_DIR / "och_lsh_abonements_by_id_broad.json")
    abonement_by_id = {str(row.get("id")): row for row in [*raw_abonements, *cached_broad_abonements] if row.get("id")}
    missing_abonement_ids = sorted(
        {
            str(row.get("most_abonements_id") or "").strip()
            for row in finances
            if str(row.get("most_abonements_id") or "").strip()
            and str(row.get("most_abonements_id") or "").strip() not in abonement_by_id
        }
    )
    if use_cached_tallanto and missing_abonement_ids:
        print(f"cached mode: missing abonements {len(missing_abonement_ids)}", file=sys.stderr, flush=True)
    if not use_cached_tallanto:
        assert client is not None
        for idx, abonement_id in enumerate(missing_abonement_ids, start=1):
            try:
                abonement_by_id[abonement_id] = client.get_entry_by_id(
                    module="most_abonements",
                    entry_id=abonement_id,
                    select_fields=ABONEMENT_FIELDS,
                )
            except TallantoApiError as exc:
                abonement_by_id[abonement_id] = {"id": abonement_id, "_error": str(exc)}
            if idx % 20 == 0:
                print(f"abonements by id {idx}/{len(missing_abonement_ids)}", file=sys.stderr, flush=True)
            time.sleep(0.05)
    abonements = dedupe_by_id(list(abonement_by_id.values()))
    if not use_cached_tallanto:
        (OUT_DIR / "och_lsh_abonements_by_id_broad.json").write_text(
            json.dumps(abonements, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    contacts_by_id: dict[str, dict[str, Any]] = {
        str(contact_id): contact
        for contact_id, contact in (
            json.loads((OUT_DIR / "och_lsh_broad_contacts_by_id.json").read_text(encoding="utf-8"))
            if (OUT_DIR / "och_lsh_broad_contacts_by_id.json").exists()
            else {}
        ).items()
    }
    for contact_id, contact in contacts_by_id.items():
        if fio(contact):
            fio_by_contact[contact_id] = fio(contact)
    missing_contact_ids = sorted(
        {
            str(row.get("contact_id") or "").strip()
            for row in finances
            if str(row.get("contact_id") or "").strip()
            and str(row.get("contact_id") or "").strip() not in fio_by_contact
        }
    )
    if use_cached_tallanto and missing_contact_ids:
        print(f"cached mode: missing contacts {len(missing_contact_ids)}", file=sys.stderr, flush=True)
    if not use_cached_tallanto:
        assert client is not None
        for idx, contact_id in enumerate(missing_contact_ids, start=1):
            try:
                contact = client.contact_by_id(contact_id, select_fields=CONTACT_FIELDS)
                contacts_by_id[contact_id] = contact
                if fio(contact):
                    fio_by_contact[contact_id] = fio(contact)
            except TallantoApiError as exc:
                contacts_by_id[contact_id] = {"id": contact_id, "_error": str(exc)}
            if idx % 20 == 0:
                print(f"contacts by id {idx}/{len(missing_contact_ids)}", file=sys.stderr, flush=True)
            time.sleep(0.05)
        (OUT_DIR / "och_lsh_broad_contacts_by_id.json").write_text(
            json.dumps(contacts_by_id, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    assigned_abonements: list[dict[str, Any]] = []
    for abonement in abonements:
        contact_id = str(abonement.get("contact_id") or "").strip()
        ok, _reason_current = is_current_2026_lsh(abonement, None, contact_events=contact_events.get(contact_id, []))
        if not ok:
            continue
        event_id, reason = assign_event(abonement, contact_events.get(contact_id, []), None, events)
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

    confirmed_payments: list[dict[str, Any]] = []
    excluded_payments: list[dict[str, Any]] = []
    ambiguous_payments: list[dict[str, Any]] = []
    for finance in finances:
        amount = parse_amount(finance.get("cost"))
        if amount <= 0 or norm(finance.get("direction")) != "in":
            continue
        contact_id = str(finance.get("contact_id") or "").strip()
        abonement = abonement_by_id.get(str(finance.get("most_abonements_id") or ""))
        ok, current_reason = is_current_2026_lsh(finance, abonement, contact_events=contact_events.get(contact_id, []))
        if not ok:
            if has_lsh_marker(purpose_blob(finance, abonement)):
                candidate = dict(finance)
                candidate["exclude_reason"] = current_reason
                excluded_payments.append(candidate)
            continue
        event_id, assign_reason = assign_event(finance, contact_events.get(contact_id, []), abonement, events)
        if not event_id:
            candidate = dict(finance)
            candidate["exclude_reason"] = assign_reason
            ambiguous_payments.append(candidate)
            continue
        event = event_by_id[event_id]
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
                "ученик_фио": display_fio(fio_by_contact.get(contact_id, "")),
                "сумма_валовая": round_money(amount),
                "метод_оплаты": payment_method(finance),
                "скидка_сумма": 0,
                "возврат_флаг": "FALSE",
                "tallanto_finance_id": finance.get("id") or "",
                "tallanto_group_id": event["course_id"],
                "tallanto_abonement_id": finance.get("most_abonements_id") or "",
                "проверка_очной_ЛШ": f"OK: {current_reason}; {assign_reason}",
                "комментарий": (
                    f"Tallanto: {finance.get('name') or ''} | method_code={finance.get('type') or ''} "
                    f"| abonement_id={finance.get('most_abonements_id') or ''}"
                ).strip(),
            }
        )

    payments_by_contact_event: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in confirmed_payments:
        payments_by_contact_event[(row["событие_id"], row["ученик_id"])].append(row)

    payments_without_participant: list[dict[str, Any]] = []
    for row in confirmed_payments:
        key = (row["событие_id"], row["ученик_id"])
        if key not in participant_template:
            event = event_by_id[row["событие_id"]]
            payments_without_participant.append(
                {
                    "событие_id": row["событие_id"],
                    "бренд": event["brand"],
                    "смена": event["title"],
                    "группа_tallanto": event["course_name"],
                    "tallanto_group_id": event["course_id"],
                    "ученик_id": row["ученик_id"],
                    "фио": row["ученик_фио"],
                    "сумма_валовая": row["сумма_валовая"],
                    "метод_оплаты": row["метод_оплаты"],
                    "tallanto_finance_id": row["tallanto_finance_id"],
                    "tallanto_abonement_id": row["tallanto_abonement_id"],
                    "проверка": "Оплата есть, но строки участника в выбранной группе Tallanto нет",
                    "комментарий": row["комментарий"],
                }
            )

    # Payment-only rows are a reconciliation problem, not real participants.
    # Keep them in the payment register and audit CSV, but do not add them to
    # participant tables or 2022-style participant counts.

    methods_seen = sorted({row["метод_оплаты"] for row in confirmed_payments})
    payment_methods = PAYMENT_METHODS_DEFAULT + [method for method in methods_seen if method not in PAYMENT_METHODS_DEFAULT]
    event_plan_prices = choose_event_plan_prices(events, abonements_by_event)

    participant_rows: list[dict[str, Any]] = []
    price_check_rows: list[dict[str, Any]] = []
    for key in sorted(participant_template, key=lambda item: (item[0], norm(participant_template[item].get("фио")), item[1])):
        event_id, contact_id = key
        base = dict(participant_template[key])
        event = event_by_id[event_id]
        payments = payments_by_contact_event.get(key, [])
        paid = sum(parse_amount(row.get("сумма_валовая")) for row in payments)
        by_method = {method: 0.0 for method in payment_methods}
        for payment in payments:
            by_method[payment["метод_оплаты"]] = by_method.get(payment["метод_оплаты"], 0.0) + parse_amount(
                payment.get("сумма_валовая")
            )
        abons = abonements_by_contact_event.get(key, [])
        abon_costs = [parse_amount(row.get("cost")) for row in abons if parse_amount(row.get("cost")) > 0]
        reference_event_price = event_plan_prices.get(event_id, 0.0)
        expected_price, expected_reason = choose_personal_expected_price(
            abon_costs=abon_costs,
            paid=paid,
            fallback_event_price=reference_event_price,
        )
        residual = max(expected_price - paid, 0) if expected_price else ""

        if paid <= 0:
            status = "нет оплаты"
            note = f"Участник есть в группе Tallanto, подтвержденной оплаты очной ЛШ 2026 не найдено; {expected_reason}"
        elif expected_price and paid >= expected_price - 1:
            if paid > expected_price * 1.05:
                status = "переплата/проверить"
                note = f"Оплата выше ожидаемой персональной цены; {expected_reason}"
            elif abon_costs:
                status = "оплачено"
                note = f"Ожидаемая цена взята из абонемента; {expected_reason}"
            else:
                status = "оплачено"
                note = expected_reason
        elif expected_price and paid < expected_price:
            status = "частично/долг"
            note = f"Оплата меньше ожидаемой персональной цены; {expected_reason}"
        else:
            status = "оплачено/проверить цену"
            note = expected_reason
        if base.get("статус_записи") == "нет строки участника Tallanto":
            note = f"{note}; оплата есть, строки участника в группе Tallanto нет"

        row = {
            "событие_id": event_id,
            "бренд": event["brand"],
            "смена": event["title"],
            "группа_tallanto": event["course_name"],
            "tallanto_group_id": event["course_id"],
            "ученик_id": contact_id,
            "фио": display_fio(base.get("фио") or fio_by_contact.get(contact_id, "")),
            "статус_записи": base.get("статус_записи", ""),
            "ожидаемая_цена": round_money(expected_price) if expected_price else "",
            "оплачено_валовая": round_money(paid) if paid else 0,
            "остаток_с_учетом_скидки": round_money(residual) if residual != "" else "",
            "статус_с_учетом_скидки": status,
            "типы_оплаты": ", ".join(sorted({payment["метод_оплаты"] for payment in payments})),
            "abonement_costs": ", ".join(str(round_money(cost)) for cost in sorted(set(abon_costs))),
            "проверка": base.get("проверка", "OK: участник группы Tallanto"),
            "комментарий": note,
            "общий_итог": round_money(paid) if paid else 0,
            "к_учету": round_money(paid) if paid else 0,
            "остаток": round_money(residual) if residual != "" else "",
            "статус": status,
        }
        for method in payment_methods:
            row[method] = round_money(by_method.get(method, 0.0)) if by_method.get(method, 0.0) else ""
        participant_rows.append(row)
        price_check_rows.append(
            {
                "событие_id": event_id,
                "ученик_id": contact_id,
                "фио": row["фио"],
                "ожидаемая_цена": row["ожидаемая_цена"],
                "оплачено_валовая": row["оплачено_валовая"],
                "цена_из_абонемента": round_money(expected_price) if abon_costs and expected_price else "",
                "остаток_уточненный": row["остаток_с_учетом_скидки"],
                "способ_ожидаемой_цены": expected_reason,
                "справочная_цена_смены": round_money(reference_event_price) if reference_event_price else "",
                "статус_цены": status,
                "abonement_ids": ", ".join(str(abon.get("id") or "") for abon in abons if abon.get("id")),
                "abonement_costs": row["abonement_costs"],
                "abonement_discounts": ", ".join(str(abon.get("discount") or "") for abon in abons if abon.get("discount")),
                "abonement_comments": " | ".join(
                    str(abon.get("discount_comment") or "") for abon in abons if abon.get("discount_comment")
                ),
                "price_check_note": note,
                "типы_оплаты": row["типы_оплаты"],
            }
        )

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
        "ожидаемая_цена",
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
        "ожидаемая_цена",
        "оплачено_валовая",
        "цена_из_абонемента",
        "остаток_уточненный",
        "способ_ожидаемой_цены",
        "справочная_цена_смены",
        "статус_цены",
        "abonement_ids",
        "abonement_costs",
        "abonement_discounts",
        "abonement_comments",
        "price_check_note",
        "типы_оплаты",
    ]
    payments_without_participant_fields = [
        "событие_id",
        "бренд",
        "смена",
        "группа_tallanto",
        "tallanto_group_id",
        "ученик_id",
        "фио",
        "сумма_валовая",
        "метод_оплаты",
        "tallanto_finance_id",
        "tallanto_abonement_id",
        "проверка",
        "комментарий",
    ]
    write_csv(OUT_DIR / "och_lsh_confirmed_payments_import.csv", confirmed_payments, payment_fields)
    write_csv(OUT_DIR / "och_lsh_participants_import.csv", participant_rows, participant_fields)
    write_csv(OUT_DIR / "och_lsh_participants_price_check.csv", price_check_rows, price_fields)
    write_csv(
        OUT_DIR / "och_lsh_payments_without_participant.csv",
        payments_without_participant,
        payments_without_participant_fields,
    )
    write_csv(OUT_DIR / "och_lsh_assigned_abonements.csv", assigned_abonements, ["id", *ABONEMENT_FIELDS, "event_id", "assign_reason"])
    write_csv(OUT_DIR / "och_lsh_ambiguous_payments.csv", ambiguous_payments, ["id", *FINANCE_FIELDS, "exclude_reason"])
    write_csv(OUT_DIR / "och_lsh_excluded_payment_candidates.csv", excluded_payments, ["id", *FINANCE_FIELDS, "exclude_reason"])

    participants_by_event = Counter(row["событие_id"] for row in participant_rows)
    payments_by_event = Counter(row["событие_id"] for row in confirmed_payments)
    sum_by_event: dict[str, float] = defaultdict(float)
    for row in confirmed_payments:
        sum_by_event[row["событие_id"]] += parse_amount(row["сумма_валовая"])
    summary_out = {
        **summary,
        "participants": len(participant_rows),
        "participants_by_event": dict(participants_by_event),
        "confirmed_payments": len(confirmed_payments),
        "confirmed_sum": round_money(sum(parse_amount(row["сумма_валовая"]) for row in confirmed_payments)),
        "payments_by_event": dict(payments_by_event),
        "sum_by_event": {key: round_money(value) for key, value in sorted(sum_by_event.items())},
        "payments_by_method": dict(Counter(row["метод_оплаты"] for row in confirmed_payments)),
        "status_counts": dict(Counter(row["статус_с_учетом_скидки"] for row in participant_rows)),
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
        "broad_finances_rows": len(broad_finances),
        "merged_finances_rows": len(finances),
        "payment_only_participant_rows": 0,
        "payments_without_participant_rows": len(payments_without_participant),
        "payments_without_participant_keys": len(
            {(row["событие_id"], row["ученик_id"]) for row in payments_without_participant}
        ),
        "payments_without_participant_by_event": dict(Counter(row["событие_id"] for row in payments_without_participant)),
    }
    (OUT_DIR / "och_lsh_summary.json").write_text(json.dumps(summary_out, ensure_ascii=False, indent=2), encoding="utf-8")

    participants_tsv = [participant_fields] + [[row.get(field, "") for field in participant_fields] for row in participant_rows]
    payments_tsv = [payment_fields] + [[row.get(field, "") for field in payment_fields] for row in confirmed_payments]
    compact_participant_fields = [
        "событие_id",
        "бренд",
        "смена",
        "фио",
        "статус_записи",
        "ожидаемая_цена",
        "оплачено_валовая",
        "остаток_с_учетом_скидки",
        "статус_с_учетом_скидки",
        "типы_оплаты",
        "abonement_costs",
        "проверка",
    ]
    compact_payment_fields = [
        "дата",
        "период",
        "бренд",
        "юрлицо",
        "формат",
        "событие_id",
        "продукт",
        "ученик_фио",
        "сумма_валовая",
        "метод_оплаты",
        "скидка_сумма",
        "возврат_флаг",
        "tallanto_finance_id",
        "проверка_очной_ЛШ",
    ]
    participants_compact_tsv = [compact_participant_fields] + [
        [row.get(field, "") for field in compact_participant_fields] for row in participant_rows
    ]
    payments_compact_tsv = [compact_payment_fields] + [
        [row.get(field, "") for field in compact_payment_fields] for row in confirmed_payments
    ]
    wide_rows = build_wide_rows(events, participant_rows, payment_methods)
    write_tsv(OUT_DIR / "och_lsh_participants_paste.tsv", participants_tsv)
    write_tsv(OUT_DIR / "och_lsh_payments_paste.tsv", payments_tsv)
    write_tsv(OUT_DIR / "och_lsh_participants_compact_paste.tsv", participants_compact_tsv)
    write_tsv(OUT_DIR / "och_lsh_payments_compact_paste.tsv", payments_compact_tsv)
    write_tsv(OUT_DIR / "och_lsh_wide_sheet.tsv", wide_rows)
    print(json.dumps(summary_out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
