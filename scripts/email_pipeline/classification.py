from __future__ import annotations

import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


OWN_DOMAINS = {"kmipt.ru"}
NOTIFY_DOMAINS = {
    "gosuslugi.ru",
    "id.yandex.ru",
    "eljur.ru",
    "reg.ru",
    "hosting.reg.ru",
    "vk.com",
    "vkontakte.ru",
    "sberbank.ru",
    "online.sberbank.ru",
    "sberbank.com",
    "notify.yandex.ru",
    "passport.yandex.ru",
    "money.yandex.ru",
    "yoomoney.ru",
    "nalog.ru",
    "nalog.gov.ru",
    "pochta.ru",
    "mvideo.ru",
    "ozon.ru",
    "wildberries.ru",
    "tinkoff.ru",
    "tbank.ru",
    "alfabank.ru",
    "vtb.ru",
    "mos.ru",
    "mail.yandex.ru",
}
ESP_DOMAINS = {
    "sendsay.ru",
    "unisender.com",
    "mindbox.ru",
    "mailchimp.com",
    "mailchimpapp.com",
    "sendpulse.com",
    "getresponse.com",
    "dashamail.com",
    "carrotquest.io",
    "amocrm.com",
    "amocrm.ru",
    "bitrix24.ru",
    "notikum.ru",
    "cmail19.com",
    "mailgun.org",
    "sparkpostmail.com",
    "sendgrid.net",
    "ms.am",
}
NOREPLY_LOCALS = {
    "noreply",
    "no-reply",
    "no_reply",
    "donotreply",
    "do-not-reply",
    "mailer-daemon",
    "mailerdaemon",
    "postmaster",
    "notification",
    "notifications",
    "notify",
    "robot",
    "auto",
    "automailer",
    "news",
    "newsletter",
    "mailing",
    "rassylka",
    "support",
    "bounce",
    "bounces",
    "feedback",
}

EML_HDR_BYTES = 16384
RE_LIST_UNSUB = re.compile(rb"^List-Unsubscribe\s*:", re.I | re.M)
RE_PRECEDENCE_BULK = re.compile(rb"^Precedence\s*:\s*(bulk|list|junk)", re.I | re.M)
RE_AUTO_SUBMITTED = re.compile(rb"^Auto-Submitted\s*:\s*auto", re.I | re.M)
RE_CAMPAIGN = re.compile(rb"^(X-Mailer|X-Campaign|X-Mailru-Msgtype|X-MC-User|Feedback-ID)\s*:", re.I | re.M)
RE_REPLY = re.compile(r"^\s*(re|fwd|fw)\s*:", re.I)


@dataclass(frozen=True)
class ClassificationInput:
    kind: str
    mailbox: str
    from_email: str
    from_dom: str
    from_local: str
    to_doms: tuple[str, ...]
    subject: str
    body_chars: int
    eml_flags: dict[str, bool]
    is_outbound: bool


def domain_of(email: str | None) -> str:
    if not email or "@" not in email:
        return ""
    return email.rsplit("@", 1)[1].strip().lower()


def local_of(email: str | None) -> str:
    if not email or "@" not in email:
        return ""
    return email.split("@", 1)[0].strip().lower()


def norm_subject(value: str | None) -> str:
    text = (value or "").strip().lower()
    text = re.sub(r"^\s*(re|fwd|fw)\s*:\s*", "", text)
    return re.sub(r"\s+", " ", text).strip()


def scan_eml_header(path: Path | str | None) -> dict[str, bool]:
    flags = {"list_unsub": False, "bulk": False, "auto": False, "campaign": False}
    if not path:
        return flags
    try:
        head = Path(path).read_bytes()[:EML_HDR_BYTES]
    except Exception:
        return flags
    sep = head.find(b"\r\n\r\n")
    if sep == -1:
        sep = head.find(b"\n\n")
    header = head[:sep] if sep != -1 else head
    flags["list_unsub"] = bool(RE_LIST_UNSUB.search(header))
    flags["bulk"] = bool(RE_PRECEDENCE_BULK.search(header))
    flags["auto"] = bool(RE_AUTO_SUBMITTED.search(header))
    flags["campaign"] = bool(RE_CAMPAIGN.search(header))
    return flags


def participants_for(con: sqlite3.Connection) -> dict[str, dict[str, object]]:
    output = defaultdict(lambda: {"from": None, "to": [], "cc": [], "reply_to": None})
    for sha, header, name, email, domain in con.execute(
        "SELECT message_sha256, header_name, display_name, email_normalized, domain FROM message_participants"
    ):
        key = (header or "").lower()
        rec = (name or "", email or "", (domain or "").lower())
        if key == "from":
            output[sha]["from"] = rec
        elif key == "to":
            output[sha]["to"].append(rec)
        elif key == "cc":
            output[sha]["cc"].append(rec)
        elif key == "reply-to":
            output[sha]["reply_to"] = rec
    return dict(output)


def build_outbound_templates(db_paths: list[Path], threshold: int = 10) -> set[str]:
    freq: Counter[str] = Counter()
    for path in db_paths:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as con:
            con.execute("PRAGMA query_only=ON")
            for subject, mailbox in con.execute("SELECT subject, mailbox FROM messages"):
                mailbox_text = mailbox or ""
                if mailbox_text in ("Sent", "Sent Messages", "Drafts", "Templates") or "Шаблоны" in mailbox_text:
                    normalized = norm_subject(subject)
                    if normalized:
                        freq[normalized] += 1
    return {subject for subject, count in freq.items() if count >= threshold}


def classify_message(msg: ClassificationInput, outbound_templates: set[str]) -> tuple[str, str]:
    kind = msg.kind
    flags = msg.eml_flags
    from_dom = msg.from_dom
    from_local = msg.from_local
    subject_lower = (msg.subject or "").lower()

    own_from = from_dom in OWN_DOMAINS
    has_external_party = any(domain and domain not in OWN_DOMAINS for domain in (from_dom, *msg.to_doms))
    if kind == "internal" and not has_external_party:
        return "internal", "kind_internal_no_external"
    if own_from and msg.to_doms and all((domain in OWN_DOMAINS or domain == "") for domain in msg.to_doms):
        return "internal", "own_to_own"

    if from_local in {"mailer-daemon", "mailerdaemon", "postmaster"} or subject_lower.startswith(
        ("undelivered", "mail delivery", "delivery status", "returned mail", "недоставлен", "доставка")
    ):
        return "bounce", "mailer_daemon_or_subject"

    if kind == "service":
        return "service_notification", "kind_service"
    if from_dom in NOTIFY_DOMAINS:
        return "service_notification", f"notify_domain:{from_dom}"
    if flags.get("auto") and not (flags.get("list_unsub") or flags.get("bulk")):
        return "service_notification", "auto_submitted"

    if from_dom in ESP_DOMAINS:
        return "bulk_newsletter", f"esp_domain:{from_dom}"
    if flags.get("list_unsub") or flags.get("bulk") or flags.get("campaign"):
        return "bulk_newsletter", "list_unsub_or_bulk_or_campaign"
    if from_local in NOREPLY_LOCALS:
        return "bulk_newsletter", f"noreply_local:{from_local}"

    if (msg.body_chars or 0) < 8 and not msg.is_outbound:
        return "spam_or_empty", "empty_body"

    if msg.is_outbound:
        if not any(domain and domain not in OWN_DOMAINS for domain in msg.to_doms):
            return "internal", "outbound_no_external"
        if RE_REPLY.match(msg.subject or ""):
            return "real_correspondence", "outbound_reply"
        if norm_subject(msg.subject) in outbound_templates:
            return "outbound_campaign", "template_mass_send"
        return "real_correspondence", "outbound_personal"
    return "real_correspondence", "inbound_human"

