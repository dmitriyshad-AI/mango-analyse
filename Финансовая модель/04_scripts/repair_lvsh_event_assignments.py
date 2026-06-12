#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "Финансовая модель" / "03_выгрузки_tallanto"

PAYMENT_FIELDS = [
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
    "проверка_ЛВШ",
    "комментарий",
]
PARTICIPANT_FIELDS = [
    "событие_id",
    "бренд",
    "смена",
    "группа_tallanto",
    "tallanto_group_id",
    "ученик_id",
    "фио",
    "статус_записи",
    "сумма_к_оплате",
    "типы_оплаты",
    "проверка",
    "комментарий",
]
PRICE_FIELDS = [
    "sheet_row",
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
    "исходный_комментарий",
]

EVENT_META = {
    "LVSH1_FOTON": ("ФОТОН", "ЦДПО ФОТОН", "ЛВШ 2026, 1 смена ФОТОН"),
    "LVSH2_FOTON": ("ФОТОН", "ЦДПО ФОТОН", "ЛВШ 2026, 2 смена ФОТОН"),
    "LVSH2_UNPK": ("УНПК", "АНО УНПК МФТИ", "ЛВШ 2026, 2 смена УНПК"),
    "LVSH3_UNPK": ("УНПК", "АНО УНПК МФТИ", "ЛВШ 2026, 3 смена УНПК"),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_tsv(path: Path, rows: list[list[Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(rows)


def amount(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    text = str(value).replace("\xa0", "").replace(" ", "").replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return 0.0


def money(value: float) -> str:
    if abs(value - round(value)) < 0.001:
        return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def formula_payment_commission_rate(row_number: int) -> str:
    return f'=IF(A{row_number}="";"";IFERROR(VLOOKUP(K{row_number};\'Справочники\'!$A$17:$B$40;2;FALSE);0))'


def formula_payment_commission(row_number: int) -> str:
    return f'=IF(A{row_number}="";"";J{row_number}*N{row_number})'


def formula_payment_net(row_number: int) -> str:
    return (
        f'=IF(A{row_number}="";"";'
        f'IF(OR(M{row_number}=TRUE;M{row_number}="да";M{row_number}="TRUE");'
        f'0;J{row_number}-IFERROR(L{row_number};0)-IFERROR(O{row_number};0)))'
    )


def formula_participant_paid(row_number: int) -> str:
    return (
        f'=IF(A{row_number}="";"";'
        f'SUMIFS(\'Поступления\'!$P:$P;\'Поступления\'!$F:$F;A{row_number};'
        f'\'Поступления\'!$H:$H;F{row_number}))'
    )


def formula_participant_residual(row_number: int) -> str:
    return f'=IF(A{row_number}="";"";MAX(0;I{row_number}-J{row_number}))'


def formula_participant_status(row_number: int) -> str:
    return (
        f'=IF(A{row_number}="";"";'
        f'IF(J{row_number}=0;"нет оплаты";IF(L{row_number}>0;"частично";"оплачено")))'
    )


def extract_abonement_id(comment: str) -> str:
    match = re.search(r"abonement_id=([0-9a-fA-F-]+)", comment or "")
    return match.group(1) if match else ""


def rebuild_logged_reassignments(payments: list[dict[str, str]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    pattern = re.compile(r"перенесено по фактической группе ученика: ([A-Z0-9_]+) -> ([A-Z0-9_]+)")
    for row in payments:
        match = pattern.search(row.get("проверка_ЛВШ", ""))
        if not match:
            continue
        result.append(
            {
                "ученик_id": row["ученик_id"],
                "фио": row["ученик_фио"],
                "tallanto_finance_id": row["tallanto_finance_id"],
                "сумма": row["сумма_валовая"],
                "метод_оплаты": row["метод_оплаты"],
                "старое_событие_id": match.group(1),
                "новое_событие_id": match.group(2),
                "основание": "у ученика ровно одна фактическая ЛВШ-группа, отличная от события платежа",
            }
        )
    return result


def main() -> None:
    participants = read_csv(OUT_DIR / "lvsh_participants_import.csv")
    payments = read_csv(OUT_DIR / "lvsh_confirmed_payments_import.csv")
    old_price_rows = read_csv(OUT_DIR / "lvsh_participants_price_check.csv")
    abonements = {row["id"]: row for row in read_csv(OUT_DIR / "lvsh_abonements_by_id.csv") if row.get("id")}

    participant_by_key = {(row["событие_id"], row["ученик_id"]): row for row in participants}
    events_by_contact: dict[str, set[str]] = defaultdict(set)
    participant_event_row: dict[tuple[str, str], dict[str, str]] = {}
    for row in participants:
        event_id = row["событие_id"]
        contact_id = row["ученик_id"]
        events_by_contact[contact_id].add(event_id)
        participant_event_row[(event_id, contact_id)] = row

    reassignments: list[dict[str, Any]] = []
    for row in payments:
        contact_id = row["ученик_id"]
        current_event = row["событие_id"]
        possible_events = sorted(events_by_contact.get(contact_id, set()))
        if current_event in possible_events or len(possible_events) != 1:
            continue
        new_event = possible_events[0]
        participant = participant_event_row[(new_event, contact_id)]
        old_event = row["событие_id"]
        brand, legal_entity, product = EVENT_META[new_event]
        row["бренд"] = brand
        row["юрлицо"] = legal_entity
        row["событие_id"] = new_event
        row["продукт"] = product
        row["tallanto_group_id"] = participant["tallanto_group_id"]
        row["проверка_ЛВШ"] = (
            f'{row.get("проверка_ЛВШ") or "OK ЛВШ-2026"}; '
            f"перенесено по фактической группе ученика: {old_event} -> {new_event}"
        )
        reassignments.append(
            {
                "ученик_id": contact_id,
                "фио": row["ученик_фио"],
                "tallanto_finance_id": row["tallanto_finance_id"],
                "сумма": row["сумма_валовая"],
                "метод_оплаты": row["метод_оплаты"],
                "старое_событие_id": old_event,
                "новое_событие_id": new_event,
                "основание": "у ученика ровно одна фактическая ЛВШ-группа, отличная от события платежа",
            }
        )
    if not reassignments:
        reassignments = rebuild_logged_reassignments(payments)

    payments_by_key: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in payments:
        payments_by_key[(row["событие_id"], row["ученик_id"])].append(row)

    old_price_by_key = {(row["событие_id"], row["ученик_id"]): row for row in old_price_rows}
    price_rows: list[dict[str, Any]] = []
    for index, row in enumerate(participants, start=1):
        key = (row["событие_id"], row["ученик_id"])
        participant_payments = payments_by_key.get(key, [])
        paid = sum(amount(payment["сумма_валовая"]) for payment in participant_payments)
        types = ", ".join(sorted({payment["метод_оплаты"] for payment in participant_payments}))
        plan = amount(row["сумма_к_оплате"])
        row["типы_оплаты"] = types
        if paid <= 0:
            row["комментарий"] = "Подтвержденных поступлений ЛВШ-2026 не найдено"
        elif paid >= plan and plan > 0:
            row["комментарий"] = "Оплачено"
        else:
            row["комментарий"] = "Оплата меньше плановой цены; возможно скидка/частичная оплата"

        old_price = old_price_by_key.get(key, {})
        abonement_ids = []
        abonement_costs = []
        abonement_discounts = []
        abonement_comments = []
        for payment in participant_payments:
            abonement_id = extract_abonement_id(payment.get("комментарий", ""))
            if not abonement_id or abonement_id in abonement_ids:
                continue
            abonement = abonements.get(abonement_id, {})
            abonement_ids.append(abonement_id)
            if abonement.get("cost"):
                abonement_costs.append(money(amount(abonement["cost"])))
            if abonement.get("discount"):
                abonement_discounts.append(money(amount(abonement["discount"])))
            if abonement.get("discount_comment"):
                abonement_comments.append(abonement["discount_comment"])

        effective_candidates = [amount(value) for value in abonement_costs if amount(value) > 0]
        effective = paid if paid > 0 and not effective_candidates else (min(effective_candidates, key=lambda x: abs(x - paid)) if effective_candidates else plan)
        residual = max(effective - paid, 0) if effective else 0
        diff_pct = (plan - paid) / plan if plan else 0
        if paid <= 0:
            status = "нет оплаты"
            note = "Подтвержденной оплаты ЛВШ-2026 не найдено"
        elif effective and paid >= effective - 1 and plan and 0 <= diff_pct <= 0.25:
            status = "оплачено со скидкой/ранним бронированием"
            note = "Стоимость абонемента или оплаты принята как персональная цена; разница к прайсу не больше 25%"
        elif effective and paid >= effective - 1:
            status = "оплачено"
            note = "Оплата покрывает персональную цену"
        else:
            status = "частично/долг"
            note = "Оплата меньше персональной цены"

        price_rows.append(
            {
                "sheet_row": index,
                "событие_id": row["событие_id"],
                "ученик_id": row["ученик_id"],
                "фио": row["фио"],
                "плановая_цена": money(plan),
                "оплачено_валовая": money(paid),
                "эффективная_цена_по_абонементу": money(effective) if effective else "",
                "остаток_уточненный": money(residual),
                "разница_к_прайсу_pct": f"{diff_pct:.4f}".rstrip("0").rstrip("."),
                "статус_цены": status,
                "abonement_ids": ", ".join(abonement_ids) or old_price.get("abonement_ids", ""),
                "abonement_costs": ", ".join(abonement_costs) or old_price.get("abonement_costs", ""),
                "abonement_discounts": ", ".join(abonement_discounts) or old_price.get("abonement_discounts", ""),
                "abonement_comments": " | ".join(abonement_comments) or old_price.get("abonement_comments", ""),
                "price_check_note": note,
                "типы_оплаты": types,
                "исходный_комментарий": row["комментарий"],
            }
        )

    write_csv(OUT_DIR / "lvsh_confirmed_payments_import.csv", payments, PAYMENT_FIELDS)
    write_csv(OUT_DIR / "lvsh_participants_import.csv", participants, PARTICIPANT_FIELDS)
    write_csv(OUT_DIR / "lvsh_participants_price_check.csv", price_rows, PRICE_FIELDS)
    write_csv(
        OUT_DIR / "lvsh_event_reassignments.csv",
        reassignments,
        ["ученик_id", "фио", "tallanto_finance_id", "сумма", "метод_оплаты", "старое_событие_id", "новое_событие_id", "основание"],
    )

    payment_paste_rows: list[list[Any]] = []
    payment_a_m_rows: list[list[Any]] = []
    payment_q_t_rows: list[list[Any]] = []
    for index, row in enumerate(payments, start=2):
        base = [
            row["дата"],
            row["период"],
            row["бренд"],
            row["юрлицо"],
            row["формат"],
            row["событие_id"],
            row["продукт"],
            row["ученик_id"],
            row["ученик_фио"],
            money(amount(row["сумма_валовая"])),
            row["метод_оплаты"],
            money(amount(row["скидка_сумма"])),
            row["возврат_флаг"],
        ]
        tail = [
            row["tallanto_finance_id"],
            row["tallanto_group_id"],
            row["проверка_ЛВШ"],
            (row["комментарий"] or "").replace("Tallanto: ", "").split(" | method_code=", 1)[0],
        ]
        payment_paste_rows.append([*base, formula_payment_commission_rate(index), formula_payment_commission(index), formula_payment_net(index), *tail])
        payment_a_m_rows.append(base)
        payment_q_t_rows.append(tail)
    write_tsv(OUT_DIR / "lvsh_payments_paste.tsv", payment_paste_rows)
    write_tsv(OUT_DIR / "lvsh_payments_paste_short.tsv", payment_paste_rows)
    write_tsv(OUT_DIR / "paste_payments_A_M.tsv", payment_a_m_rows)
    write_tsv(OUT_DIR / "paste_payments_Q_T.tsv", payment_q_t_rows)

    participant_paste_rows: list[list[Any]] = []
    participant_a_i_rows: list[list[Any]] = []
    participant_k_rows: list[list[Any]] = []
    participant_n_o_rows: list[list[Any]] = []
    for index, row in enumerate(participants, start=2):
        base = [
            row["событие_id"],
            row["бренд"],
            row["смена"],
            row["группа_tallanto"],
            row["tallanto_group_id"],
            row["ученик_id"],
            row["фио"],
            row["статус_записи"],
            money(amount(row["сумма_к_оплате"])),
        ]
        check_short = str(row["проверка"]).replace("OK: участник группы Tallanto", "OK группа Tallanto")
        comment_short = str(row["комментарий"]).replace("; возможно скидка/частичная оплата", "")
        participant_paste_rows.append(
            [
                *base,
                formula_participant_paid(index),
                row["типы_оплаты"],
                formula_participant_residual(index),
                formula_participant_status(index),
                row["проверка"],
                row["комментарий"],
            ]
        )
        participant_a_i_rows.append(base)
        participant_k_rows.append([row["типы_оплаты"]])
        participant_n_o_rows.append([check_short, comment_short])
    write_tsv(OUT_DIR / "lvsh_participants_paste.tsv", participant_paste_rows)
    write_tsv(OUT_DIR / "lvsh_participants_paste_short.tsv", participant_paste_rows)
    write_tsv(OUT_DIR / "paste_participants_A_I.tsv", participant_a_i_rows)
    write_tsv(OUT_DIR / "paste_participants_K.tsv", participant_k_rows)
    write_tsv(OUT_DIR / "paste_participants_N_O.tsv", participant_n_o_rows)

    price_paste_rows = [["эффективная_цена", "остаток_уточненный", "разница_к_прайсу_%", "статус_цены", "стоимости_абонементов", "скидки_абонементов", "комментарий_проверки_цены", "abonement_ids"]]
    for row in price_rows:
        price_paste_rows.append(
            [
                row["эффективная_цена_по_абонементу"],
                row["остаток_уточненный"],
                row["разница_к_прайсу_pct"],
                row["статус_цены"],
                row["abonement_costs"],
                row["abonement_discounts"],
                row["price_check_note"],
                row["abonement_ids"],
            ]
        )
    write_tsv(OUT_DIR / "paste_participants_price_check_P_W.tsv", price_paste_rows)

    payments_by_event = Counter(row["событие_id"] for row in payments)
    sum_by_event: dict[str, float] = defaultdict(float)
    for row in payments:
        sum_by_event[row["событие_id"]] += amount(row["сумма_валовая"])
    no_payment = sum(1 for row in participants if not payments_by_key.get((row["событие_id"], row["ученик_id"])))
    print(f"reassignments={len(reassignments)}")
    print(f"payments_by_event={dict(payments_by_event)}")
    print(f"sum_by_event={{{', '.join(f'{key}: {money(value)}' for key, value in sorted(sum_by_event.items()))}}}")
    print(f"no_payment_participants={no_payment}")


if __name__ == "__main__":
    main()
