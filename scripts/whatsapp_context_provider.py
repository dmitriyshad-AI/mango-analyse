#!/usr/bin/env python3
"""WhatsApp context provider (Этап 3) — по образцу customer timeline context_provider.py.

Вход: primary_phone. Выход: набор matched_whatsapp_rows для timeline-импорта,
совместимый со схемой timeline_import_source.csv (расширение полями whatsapp_*).

Безопасность: read-only по whatsapp_chats.sqlite, без сети/подпроцессов/записи в CRM/runtime.
Сводка — фактическая (из реальных сообщений), без выдумки. Опциональный summary_fn
(например, Haiku) можно подать колбэком; по умолчанию — детерминированная сводка.
"""
from __future__ import annotations
import re, sqlite3, json, os, sys
from collections import Counter
from typing import Any, Callable, Mapping, Sequence

WHATSAPP_CONTEXT_PROVIDER_SCHEMA_VERSION = "whatsapp_context_provider_v1"

# P0-маркеры в клиентских сообщениях (согласовано с p0_recall_spec по смыслу).
P0_MARKERS = {
    "refund": ("верните деньги", "верните оплат", "требую возврат", "хочу возврат",
               "деньги назад", "вернуть деньги", "расторг", "отказ от обучен"),
    "legal": ("суд", "прокуратур", "роспотребнадзор", "претензи", "юрист", "адвокат", "незаконн"),
    "complaint": ("жалоб", "пожалуюсь", "обман", "мошенн", "недовол", "возмущ", "ужасн", "плохо учит"),
    "payment_dispute": ("двойн", "списали дважды", "не списал", "оплатил а", "чарджб", "оспор"),
}
# Тематические маркеры для фактической сводки тем.
TOPIC_MARKERS = (
    ("цена/оплата", ("цена", "стоим", "сколько", "рассроч", "оплат", "скидк", "руб")),
    ("расписание", ("расписан", "когда", "во сколько", "дни", "время занят")),
    ("лагерь/ЛВШ", ("лвш", "лагер", "смен", "менделеево", "выездн", "прожив")),
    ("формат/онлайн-очно", ("онлайн", "очно", "вебинар", "запис")),
    ("пробное/запись", ("пробн", "записать", "заявк", "оформ")),
    ("документы/справка", ("справк", "договор", "документ", "вычет", "маткап")),
    ("олимпиады", ("олимпиад", "физтех", "перечнев")),
)


def normalize_phone_for_match(value: Any) -> str:
    digits = re.sub(r"\D+", "", str(value or "").strip())
    if len(digits) == 11 and digits.startswith("8"):
        digits = "7" + digits[1:]
    if len(digits) == 10:
        digits = "7" + digits
    return "+" + digits if digits else ""


def context_provider_safety_contract() -> dict[str, Any]:
    return {
        "schema_version": WHATSAPP_CONTEXT_PROVIDER_SCHEMA_VERSION,
        "read_whatsapp_db": True,
        "write_whatsapp_db": False,
        "write_crm": False,
        "network_calls": False,
        "subprocess_calls": False,
        "live_writeback_required": False,
    }


def _detect_risk_classes(client_text: str) -> list[str]:
    low = (client_text or "").lower()
    out = []
    for cls, markers in P0_MARKERS.items():
        if any(m in low for m in markers):
            out.append(cls)
    return out


def _topics(text: str) -> list[str]:
    low = (text or "").lower()
    return [name for name, ms in TOPIC_MARKERS if any(m in low for m in ms)]


def _deterministic_summary(rows: Sequence[Mapping[str, Any]], brand: str | None,
                           first_ts: str, last_ts: str, n_client: int, n_manager: int) -> str:
    text = " ".join(str(r.get("text") or "") for r in rows)
    topics = _topics(text)
    parts = [f"WhatsApp: {len(rows)} содержательных сообщений ({first_ts[:10]}–{last_ts[:10]})."]
    if brand:
        parts.append(f"Бренд: {brand}.")
    parts.append(f"Клиент: {n_client}, менеджер: {n_manager}.")
    if topics:
        parts.append("Темы: " + ", ".join(topics) + ".")
    return " ".join(parts)


def get_whatsapp_context_for_phone(
    phone: str,
    *,
    whatsapp_db: str,
    limit: int = 50,
    summary_fn: Callable[[Sequence[Mapping[str, Any]]], str] | None = None,
) -> dict[str, Any]:
    """Read-only контекст WhatsApp по телефону для timeline-импорта."""
    normalized = normalize_phone_for_match(phone)
    base = {
        "schema_version": WHATSAPP_CONTEXT_PROVIDER_SCHEMA_VERSION,
        "primary_phone": normalized,
        "matched_whatsapp_rows": 0,
        "whatsapp_first_ts": "",
        "whatsapp_last_ts": "",
        "whatsapp_brand_hint": None,
        "whatsapp_summary": "",
        "risk_classes": [],
        "found": False,
        "warnings": [],
        "safety": context_provider_safety_contract(),
    }
    if not normalized or not os.path.exists(whatsapp_db):
        base["warnings"].append("whatsapp_db_not_available" if not os.path.exists(whatsapp_db) else "empty_phone")
        return base
    con = sqlite3.connect(f"file:{whatsapp_db}?mode=ro", uri=True)
    try:
        cur = con.cursor()
        chat = cur.execute(
            "SELECT chat_id, first_ts, last_ts, brand_hint FROM chats WHERE client_phone=? ORDER BY last_ts DESC",
            (normalized,)).fetchall()
        if not chat:
            base["warnings"].append("whatsapp_customer_not_found")
            return base
        chat_ids = [c[0] for c in chat]
        brands = {c[3] for c in chat if c[3]}
        brand = "mixed" if ("mixed" in brands or len(brands) > 1) else (next(iter(brands)) if brands else None)
        ph_ph = ",".join("?" * len(chat_ids))
        rows = cur.execute(
            f"SELECT ts, role, text FROM messages WHERE chat_id IN ({ph_ph}) AND is_service_message=0 ORDER BY ts DESC LIMIT ?",
            (*chat_ids, limit)).fetchall()
        items = [{"event_at": r[0], "role": r[1], "text": r[2]} for r in rows]
        n_client = sum(1 for r in rows if r[1] == "client")
        n_manager = sum(1 for r in rows if r[1] == "manager")
        client_text = " ".join(r[2] for r in rows if r[1] == "client")
        first_ts = min((c[1] for c in chat if c[1]), default="")
        last_ts = max((c[2] for c in chat if c[2]), default="")
        summary = summary_fn(items) if summary_fn else _deterministic_summary(items, brand, first_ts, last_ts, n_client, n_manager)
        return {
            **base,
            "matched_whatsapp_rows": len(items),
            "whatsapp_first_ts": first_ts,
            "whatsapp_last_ts": last_ts,
            "whatsapp_brand_hint": brand,
            "whatsapp_summary": summary,
            "risk_classes": _detect_risk_classes(client_text),
            "found": True,
            "chat_ids": chat_ids,
        }
    finally:
        con.close()


if __name__ == "__main__":
    db = os.environ.get("MANGO_DB", "/tmp/wadb/whatsapp_chats.sqlite")
    phone = sys.argv[1] if len(sys.argv) > 1 else None
    if not phone:
        # демо: первый matched-чат
        c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        phone = c.execute("SELECT primary_phone FROM crm_match LIMIT 1").fetchone()[0]
        c.close()
    print(json.dumps(get_whatsapp_context_for_phone(phone, whatsapp_db=db), ensure_ascii=False, indent=2))
