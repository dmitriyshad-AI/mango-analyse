#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "autonomy_personas_v1_2026_06_13"
DEFAULT_TELEGRAM_EXPORT = Path("TP UNPK DataExport_2026-05-21/result.json")
DEFAULT_VOICE_TRANSCRIPTS_DIR = Path(
    "product_data/gold_candidates_paid_proxy_after_calls_tropina_kozlova_20260521_v3_rebuilt_current_runtime"
    "/transcripts_all_strong"
)
DEFAULT_OUT = Path("product_data/telegram_dynamic_test_sets/autonomy_personas_unpk_20260613.jsonl")
DONOR_LIMIT_PER_PERSONA = 2

FOTON_FORBIDDEN = (
    "Фотон",
    "ЦДПО",
    "ЦДПО Фотон",
    "ЦРДО",
    "cdpofoton.ru",
    "edu@cdpofoton.ru",
    "@cdpoFoton",
    "foton_edu",
)

B2C_STAGES = {
    "unparsed": {"pipeline_id": "10408062", "status_id": "82257086", "name": "Неразобранное"},
    "decision": {"pipeline_id": "10408062", "status_id": "82257090", "name": "Принимают решение"},
    "perspective": {"pipeline_id": "10408062", "status_id": "83489762", "name": "Перспектива"},
    "contract": {"pipeline_id": "10408062", "status_id": "82257094", "name": "Заключение договора"},
    "group": {"pipeline_id": "10408062", "status_id": "82257098", "name": "Запись в группу"},
    "payment_waiting": {"pipeline_id": "10408062", "status_id": "82258194", "name": "Ожидание оплаты"},
    "paid": {"pipeline_id": "10408062", "status_id": "82258198", "name": "Оплата получена"},
}

ACTION_CONTRACT_VERSION = "action_intent_v1_expected_only_2026_06_13"

PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\s().-]*){10,15}(?!\d)")
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
HANDLE_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]{4,}(?!\w)")
URL_RE = re.compile(r"\b(?:https?://|www\.|t\.me/)\S+", re.I)
MONEY_RE = re.compile(r"\b\d{1,3}(?:[\s\u00a0]\d{3})+\s*(?:руб\.?|рублей|₽)?\b|\b\d{4,6}\s*(?:руб\.?|рублей|₽)\b", re.I)
EXACT_DATE_RE = re.compile(r"\b\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?\b")
FULL_NAME_RE = re.compile(r"\b[А-ЯЁ][а-яё]{2,}\s+[А-ЯЁ][а-яё]{2,}(?:\s+[А-ЯЁ][а-яё]{2,})?\b")
SCHOOL_RE = re.compile(r"\b(?:школ[ауы]|лице[йяе]|гимнази[яиюи]|колледж)\s+(?:№?\s*\d+|[А-ЯЁA-Z][\wА-Яа-яЁё-]*)", re.I)
RARE_LOCATION_RE = re.compile(
    r"\b(?:Скорняжн\w*|Сухаревск\w*|Сретенк\w*|Пацаев\w*|Институтск\w*|"
    r"Чист(?:ые|ыми)\s+пруд(?:ы|ами)?|НИУ\s*ВШЭ|ВШЭ|Менделеев\w*)\b",
    re.I,
)

SUBJECT_MARKERS = (
    ("математ", "математика"),
    ("физик", "физика"),
    ("информат", "информатика"),
    ("программирован", "программирование"),
    ("русск", "русский язык"),
    ("англий", "английский язык"),
    ("хими", "химия"),
    ("биолог", "биология"),
)
SUBJECT_PREPOSITIONAL = {
    "математика": "математике",
    "физика": "физике",
    "информатика": "информатике",
    "программирование": "программированию",
    "русский язык": "русскому языку",
    "английский язык": "английскому языку",
    "химия": "химии",
    "биология": "биологии",
}


@dataclass(frozen=True)
class OpeningCandidate:
    source_hash: str
    text: str
    category: str
    slots: Mapping[str, str]


@dataclass(frozen=True)
class VoicePattern:
    source_hash: str
    objections: tuple[str, ...]
    tone: str
    message_length: str


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        rows = build_autonomy_persona_rows(
            telegram_export=Path(args.telegram_export),
            voice_transcripts_dir=Path(args.voice_transcripts_dir),
            limit=args.limit,
            seed=args.seed,
        )
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
        print(json.dumps({"out": str(out), "rows": len(rows), "personas": len(rows) - 2}, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI must fail compactly for batch usage.
        print(f"build_autonomy_personas failed: {exc}", file=sys.stderr)
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build anonymized autonomy personas for Telegram dynamic simulator.")
    parser.add_argument("--telegram-export", default=str(DEFAULT_TELEGRAM_EXPORT), help="Telegram result.json export.")
    parser.add_argument("--voice-transcripts-dir", default=str(DEFAULT_VOICE_TRANSCRIPTS_DIR), help="Directory with 703 call transcript .md files.")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output dynamic-sim JSONL.")
    parser.add_argument("--limit", type=int, default=12, help="Number of persona rows to write.")
    parser.add_argument("--seed", type=int, default=21, help="Deterministic selection seed.")
    return parser


def build_autonomy_persona_rows(
    *,
    telegram_export: Path,
    voice_transcripts_dir: Path,
    limit: int = 12,
    seed: int = 21,
) -> list[Mapping[str, Any]]:
    if limit < 1:
        raise ValueError("limit must be >= 1")
    openings = load_telegram_opening_candidates(telegram_export)
    voices = load_voice_patterns(voice_transcripts_dir)
    if not openings:
        raise ValueError(f"no usable Telegram openings found in {telegram_export}")
    if not voices:
        raise ValueError(f"no usable voice patterns found in {voice_transcripts_dir}")

    selected = select_diverse_openings(openings, limit=limit, seed=seed)
    rng = random.Random(seed)
    voice_pool = list(voices)
    rng.shuffle(voice_pool)
    personas = [
        build_persona(candidate, voice_pool[index % len(voice_pool)], index=index + 1)
        for index, candidate in enumerate(selected)
    ]
    assert_personas_safe(personas)
    return [simulator_spec(), judge_spec(), *personas]


def simulator_spec() -> Mapping[str, Any]:
    return {
        "type": "simulator_spec",
        "title": "Autonomy personas from anonymized UNPK patterns",
        "version": SCHEMA_VERSION,
        "rules": [
            "Играй живого клиента УНПК МФТИ в Telegram; не раскрывай, что это тест.",
            "Если persona.seed_opening задан, начни с него по смыслу, можно чуть переформулировать.",
            "Держись persona.objection_sequence и persona.voice_profile: не соглашайся на оплату неправдоподобно быстро.",
            "Не добавляй ФИО, телефон, школу, точный адрес, точную сумму или другие персональные детали.",
            "Если persona содержит P0/спорную оплату, веди себя тревожно и требуй ответственного сотрудника.",
        ],
        "autonomy_personas_enabled": True,
        "action_judge_enabled": False,
        "source_policy": "raw PII is local only; package contains paraphrased synthetic personas",
    }


def judge_spec() -> Mapping[str, Any]:
    return {
        "type": "judge_spec",
        "title": "Text-quality judge; action judge is intentionally off for TZ-21 delivery 1",
        "version": SCHEMA_VERSION,
        "instructions": (
            "Оценивай только качество текста, бренд-чистоту, фактичность и P0-маршрутизацию. "
            "Поля expected_action/deal_card предназначены для будущего судьи действий и текущим судьёй не оцениваются."
        ),
        "action_judge_enabled": False,
        "output_schema": {
            "dialog_id": "string",
            "brand": "unpk",
            "verdict": "PASS|PASS_WITH_NOTES|FAIL",
            "hard_gates_passed": "boolean",
            "violated_gates": "array[string]",
            "quality_score_0_2": "number",
            "rationale": "string",
        },
        "hard_gates_fail_whole_dialog_if_violated": [
            "brand_leak",
            "fabrication",
            "p0_mishandled",
            "made_a_promise",
            "internal_leak",
            "revealed_ai",
        ],
    }


def load_telegram_opening_candidates(path: Path) -> list[OpeningCandidate]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    chats = ((payload.get("chats") or {}).get("list") or []) if isinstance(payload, Mapping) else []
    candidates: list[OpeningCandidate] = []
    for chat in chats:
        if not isinstance(chat, Mapping) or chat.get("type") != "personal_chat":
            continue
        first_added = False
        p0_added = False
        for message in chat.get("messages") or []:
            if not isinstance(message, Mapping) or message.get("type") != "message":
                continue
            if is_unpk_outbound_message(message):
                continue
            text = normalize_text(flatten_telegram_text(message.get("text")))
            if not is_usable_opening(text):
                continue
            slots = infer_slots(text)
            category = infer_category(text, slots)
            if not first_added:
                source_hash = stable_hash(f"telegram:first:{chat.get('id')}:{message.get('id')}")
                candidates.append(OpeningCandidate(source_hash=source_hash, text=text, category=category, slots=slots))
                first_added = True
            if category == "p0" and not p0_added:
                source_hash = stable_hash(f"telegram:p0:{chat.get('id')}:{message.get('id')}")
                candidates.append(OpeningCandidate(source_hash=source_hash, text=text, category=category, slots=slots))
                p0_added = True
            if first_added and p0_added:
                break
    return candidates


def load_voice_patterns(path: Path) -> list[VoicePattern]:
    patterns: list[VoicePattern] = []
    for transcript in sorted(path.glob("*.md")):
        text = transcript.read_text(encoding="utf-8", errors="ignore")
        summary = extract_section(text, "## Резюме", "## Следующий шаг") or text[:5000]
        objections = classify_objections(summary)
        if not objections:
            continue
        patterns.append(
            VoicePattern(
                source_hash=f"call:{stable_hash(transcript.name)[:12]}",
                objections=objections,
                tone=infer_voice_tone(summary),
                message_length="короткие сообщения" if len(summary) < 900 else "средние сообщения с вводными",
            )
        )
    return patterns


def build_persona(candidate: OpeningCandidate, voice: VoicePattern, *, index: int) -> Mapping[str, Any]:
    seed_opening = paraphrase_opening(candidate)
    deal_card = build_deal_card(candidate)
    expected_action = expected_action_from_deal_card(deal_card)
    topic_id = gold_topic_for_category(candidate.category)
    persona_label = persona_label_for(candidate)
    objection_sequence = [
        {"step": step, "pattern": item}
        for step, item in enumerate(voice.objections[:3] or default_objections(candidate.category), start=1)
    ]
    mood = mood_for_category(candidate.category, voice)
    slots = dict(candidate.slots)
    return {
        "type": "persona",
        "dialog_id": f"autonomy_unpk_real_{index:03d}",
        "brand": "unpk",
        "category": "p0" if candidate.category == "p0" else "sales",
        "autonomy_category": candidate.category,
        "gold_topic_id": topic_id,
        "expected_route": "manager_only" if candidate.category == "p0" else "na",
        "persona": persona_label,
        "mood": mood,
        "style": voice.message_length,
        "seed_opening": seed_opening,
        "objection_sequence": objection_sequence,
        "voice_profile": {
            "tone": voice.tone,
            "message_length": voice.message_length,
            "directness": "задаёт конкретные вопросы, без длинных объяснений" if len(candidate.text) < 160 else "даёт вводные и уточняет несколько условий",
            "typos": "может писать без идеальной пунктуации, но без нарочитого шума",
        },
        "deal_card": deal_card,
        "expected_action": expected_action,
        "held_facts": slots,
        "behaviors": [
            "Начни с seed_opening по смыслу, не добавляя личные данные.",
            *[item["pattern"] for item in objection_sequence],
            "Если бот ответил полезно, переходи к следующему естественному уточнению; если ушёл в шаблон, мягко сопротивляйся.",
        ],
        "goal": goal_for_category(candidate.category),
        "success_criteria": success_for_category(candidate.category),
        "fail_criteria": "смешивает Фотон и УНПК; обещает место, оплату, скидку, срок связи или решение без подтверждённого факта; на P0 продаёт вместо передачи менеджеру",
        "brand_forbidden": list(FOTON_FORBIDDEN),
        "max_turns": 6 if candidate.category != "p0" else 4,
        "privacy": {
            "raw_text_included": False,
            "paraphrased": True,
            "max_real_donors_per_persona": DONOR_LIMIT_PER_PERSONA,
            "indirect_identifier_policy": "school/name/rare_address/exact_amount/date removed or generalized",
        },
        "source_provenance": {
            "telegram_donor_hash": f"tg:{candidate.source_hash[:12]}",
            "voice_donor_hashes": [voice.source_hash],
            "donor_count": DONOR_LIMIT_PER_PERSONA,
        },
    }


def expected_action_from_deal_card(deal_card: Mapping[str, Any]) -> Mapping[str, Any]:
    preconditions = deal_card.get("preconditions") if isinstance(deal_card.get("preconditions"), Mapping) else {}
    if truthy(preconditions.get("p0_required")):
        action = "handoff_manager"
        reason = "P0/high-risk context requires a responsible manager; selling action is unsafe."
    elif truthy(preconditions.get("product_selected")) and truthy(preconditions.get("price_confirmed")) and truthy(preconditions.get("client_ready_to_pay")):
        action = "send_payment_link"
        reason = "Product, price and readiness are all confirmed in the synthetic deal card."
    elif truthy(preconditions.get("wants_trial")) and truthy(preconditions.get("product_selected")):
        action = "book_trial"
        reason = "Client asks for trial/diagnostics and product direction is selected."
    elif truthy(preconditions.get("lead_data_sufficient")) and not truthy(preconditions.get("lead_captured")):
        action = "capture_lead"
        reason = "Enough non-sensitive lead slots are known, but the synthetic lead is not captured yet."
    else:
        action = "answer_only"
        reason = "Preconditions for payment, trial booking or lead capture are not met."
    return {
        "schema_version": ACTION_CONTRACT_VERSION,
        "action": action,
        "source": "deterministic_rule_from_deal_card",
        "reason": reason,
        "manual_label": False,
    }


def build_deal_card(candidate: OpeningCandidate) -> Mapping[str, Any]:
    slots = dict(candidate.slots)
    category = candidate.category
    product_selected = bool(slots.get("grade") and (slots.get("subject") or category == "camp"))
    wants_trial = category == "trial"
    p0_required = category == "p0"
    lead_data_sufficient = bool(slots.get("grade") and (slots.get("subject") or slots.get("format") or category == "camp"))
    ready_to_pay = bool(re.search(r"\b(?:готов[аы]?\s+оплат|хотим\s+запис|запишите|бронь|заброниров)\w*", candidate.text, re.I))
    price_confirmed = bool(ready_to_pay and product_selected and re.search(r"\b(?:цена|стоимост|оплат)\w*", candidate.text, re.I))
    if p0_required:
        stage_key = "decision"
    elif price_confirmed and ready_to_pay:
        stage_key = "payment_waiting"
    elif wants_trial:
        stage_key = "perspective"
    elif lead_data_sufficient:
        stage_key = "unparsed"
    else:
        stage_key = "decision"
    return {
        "schema_version": "synthetic_deal_card_v1_2026_06_13",
        "brand": "unpk",
        "stage": B2C_STAGES[stage_key],
        "stage_dictionary_source": "stable_runtime/deal_aware_amo_live_snapshot_20260513_v2/amo_status_catalog.csv",
        "stage_key": f"{B2C_STAGES[stage_key]['pipeline_id']}:{B2C_STAGES[stage_key]['status_id']}",
        "prior_payments": "unknown_or_none",
        "prior_contacts": "telegram_opening_plus_call_pattern",
        "product_interest": product_interest_for(category, slots),
        "preconditions": {
            "p0_required": p0_required,
            "product_selected": product_selected,
            "price_confirmed": price_confirmed,
            "client_ready_to_pay": ready_to_pay,
            "wants_trial": wants_trial,
            "lead_data_sufficient": lead_data_sufficient,
            "lead_captured": False,
        },
        "live_system": "offline_mock_only",
    }


def select_diverse_openings(candidates: Sequence[OpeningCandidate], *, limit: int, seed: int) -> list[OpeningCandidate]:
    rng = random.Random(seed)
    pool = list(candidates)
    rng.shuffle(pool)
    selected: list[OpeningCandidate] = []
    for category in ("camp", "pricing", "trial", "schedule", "format", "p0", "program"):
        match = next((item for item in pool if item.category == category and item not in selected), None)
        if match is not None:
            selected.append(match)
        if len(selected) >= limit:
            return selected
    for item in pool:
        if item not in selected:
            selected.append(item)
        if len(selected) >= limit:
            break
    return selected


def flatten_telegram_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        parts: list[str] = []
        for item in value:
            if isinstance(item, Mapping):
                parts.append(flatten_telegram_text(item.get("text")))
            else:
                parts.append(flatten_telegram_text(item))
        return "".join(parts)
    return ""


def is_unpk_outbound_message(message: Mapping[str, Any]) -> bool:
    sender = str(message.get("from") or "").casefold()
    return "унпк" in sender or "мфти" in sender


def is_usable_opening(text: str) -> bool:
    if len(text) < 20 or len(text) > 700:
        return False
    if URL_RE.search(text) and len(re.sub(URL_RE, "", text).strip()) < 20:
        return False
    return bool(re.search(r"[а-яё]", text, re.I))


def infer_slots(text: str) -> Mapping[str, str]:
    normalized = text.casefold().replace("ё", "е")
    result: dict[str, str] = {}
    grade = extract_grade(normalized)
    if grade:
        result["grade"] = grade
    subject = extract_subject(normalized)
    if subject:
        result["subject"] = subject
    if re.search(r"\bонлайн\b|дистанцион", normalized):
        result["format"] = "онлайн"
    elif re.search(r"\bочно\b|городск|выездн|лагер|смен", normalized):
        result["format"] = "очно"
    return result


def extract_grade(text: str) -> str:
    match = re.search(r"\b([1-9]|10|11)\s*(?:класс|кл\.?)\b", text, re.I)
    return match.group(1) if match else ""


def extract_subject(text: str) -> str:
    for marker, subject in SUBJECT_MARKERS:
        if marker in text:
            return subject
    return ""


def infer_category(text: str, slots: Mapping[str, str]) -> str:
    normalized = text.casefold().replace("ё", "е")
    if re.search(r"\b(?:возврат|вернуть\s+деньги|жалоб|претензи|суд|юрист|обман)\w*", normalized):
        return "p0"
    if re.search(r"\b(?:пробн|диагност|тестов(?:ое|ый)|первое\s+занят)\w*", normalized):
        return "trial"
    if re.search(r"\b(?:лагер|летн|смен|каникул|выездн|городск)\w*", normalized):
        return "camp"
    if re.search(r"\b(?:стоимост|цена|оплат|рассроч|скидк|маткап)\w*", normalized):
        return "pricing"
    if re.search(r"\b(?:распис|когда|время|день|выходн)\w*", normalized):
        return "schedule"
    if re.search(r"\b(?:онлайн|очно|формат|платформ)\w*", normalized):
        return "format"
    if slots.get("subject") or slots.get("grade"):
        return "program"
    return "general"


def paraphrase_opening(candidate: OpeningCandidate) -> str:
    slots = candidate.slots
    grade = f"для {slots['grade']} класса" if slots.get("grade") else "для школьника"
    subject = subject_phrase(slots)
    if candidate.category == "camp":
        return f"Здравствуйте, рассматриваем летнюю программу {grade}{subject}; хочу понять формат, уровень и что делать дальше."
    if candidate.category == "pricing":
        return f"Здравствуйте, подбираем обучение {grade}{subject}; сначала хочу понять стоимость и варианты оплаты."
    if candidate.category == "trial":
        return f"Здравствуйте, хотим попробовать занятие {grade}{subject}; подскажите, как корректно согласовать пробный формат."
    if candidate.category == "schedule":
        return f"Здравствуйте, выбираем занятия {grade}{subject}; важно понять расписание и нагрузку."
    if candidate.category == "format":
        return f"Здравствуйте, интересует формат обучения {grade}{subject}; хочу сравнить онлайн и очный вариант."
    if candidate.category == "p0":
        return "Здравствуйте, нужна помощь ответственного сотрудника по спорной ситуации с оплатой; личные данные в чате разбирать не хочу."
    return f"Здравствуйте, ищем подходящую программу {grade}{subject}; хочется понять уровень и следующий шаг."


def subject_phrase(slots: Mapping[str, str]) -> str:
    subject = slots.get("subject")
    if not subject:
        return ""
    return f" по {SUBJECT_PREPOSITIONAL.get(subject, subject)}"


def extract_section(text: str, start: str, end: str) -> str:
    start_index = text.find(start)
    if start_index < 0:
        return ""
    start_index += len(start)
    end_index = text.find(end, start_index)
    if end_index < 0:
        end_index = len(text)
    return text[start_index:end_index].strip()


def classify_objections(text: str) -> tuple[str, ...]:
    normalized = text.casefold().replace("ё", "е")
    result: list[str] = []
    if re.search(r"\b(?:дорог|цена|стоимост|оплат|бюджет|рассроч|платеж)\w*", normalized):
        result.append("после объяснения условий уточняет, есть ли более гибкий вариант оплаты")
    if re.search(r"\b(?:время|суббот|воскрес|распис|не\s+подходит|занят)\w*", normalized):
        result.append("проверяет расписание и сопротивляется, если время не подходит семье")
    if re.search(r"\b(?:уровень|сложно|тяжело|потян|сильн|слаб)\w*", normalized):
        result.append("сомневается, подойдёт ли уровень ребёнку")
    if re.search(r"\b(?:программ|что\s+входит|преподав|кто\s+вед|тем[аы])\w*", normalized):
        result.append("просит объяснить программу и кто будет вести занятия без фамилий")
    if re.search(r"\b(?:другие\s+варианты|сравн|подума|посовет)\w*", normalized):
        result.append("сравнивает с другими вариантами и не принимает решение сразу")
    if re.search(r"\b(?:возврат|жалоб|претензи|суд|обман)\w*", normalized):
        result.append("настаивает на передаче ответственному сотруднику по спорной ситуации")
    return tuple(dict.fromkeys(result))


def default_objections(category: str) -> tuple[str, ...]:
    if category == "p0":
        return ("настаивает на ответственном сотруднике и не хочет раскрывать личные данные",)
    if category == "pricing":
        return ("сначала спрашивает цену, затем уточняет, можно ли платить гибко",)
    if category == "camp":
        return ("спрашивает про уровень и не хочет обещаний по местам без проверки",)
    return ("просит конкретный следующий шаг, но не готов принимать решение без понятных условий",)


def infer_voice_tone(text: str) -> str:
    normalized = text.casefold().replace("ё", "е")
    if re.search(r"\b(?:возврат|жалоб|претензи|суд|обман|пережива|тревож)\w*", normalized):
        return "тревожный, требует аккуратной передачи менеджеру"
    if re.search(r"\b(?:дорог|бюджет|рассроч|оплат)\w*", normalized):
        return "прагматичный, чувствителен к цене"
    if re.search(r"\b(?:уровень|сложно|потян|олимпиад)\w*", normalized):
        return "внимательный к уровню и пользе"
    return "деловой, задаёт уточняющие вопросы"


def build_safe_text_blob(personas: Sequence[Mapping[str, Any]]) -> str:
    return json.dumps([collect_public_safety_payload(persona) for persona in personas], ensure_ascii=False, sort_keys=True)


def collect_public_safety_payload(value: Any, *, parent_key: str = "") -> Any:
    if parent_key in {"brand_forbidden", "source_provenance", "privacy"}:
        return None
    if isinstance(value, Mapping):
        return {
            str(key): collect_public_safety_payload(item, parent_key=str(key))
            for key, item in value.items()
            if str(key) not in {"brand_forbidden", "source_provenance", "privacy"}
        }
    if isinstance(value, list):
        return [collect_public_safety_payload(item, parent_key=parent_key) for item in value]
    if isinstance(value, tuple):
        return tuple(collect_public_safety_payload(item, parent_key=parent_key) for item in value)
    return value


def assert_personas_safe(personas: Sequence[Mapping[str, Any]]) -> None:
    text = build_safe_text_blob(personas)
    violations = safety_violations(text)
    if violations:
        raise ValueError("unsafe persona output: " + ", ".join(violations))
    for persona in personas:
        if persona.get("brand") != "unpk":
            raise ValueError("autonomy personas from UNPK sources must be brand=unpk only")
        privacy = persona.get("privacy") if isinstance(persona.get("privacy"), Mapping) else {}
        if privacy.get("raw_text_included") is not False:
            raise ValueError("persona must declare raw_text_included=false")
        provenance = persona.get("source_provenance") if isinstance(persona.get("source_provenance"), Mapping) else {}
        if int(provenance.get("donor_count") or 0) > DONOR_LIMIT_PER_PERSONA:
            raise ValueError("persona exceeds donor limit")


def safety_violations(text: str) -> list[str]:
    checks = (
        ("phone", PHONE_RE),
        ("email", EMAIL_RE),
        ("handle", HANDLE_RE),
        ("url", URL_RE),
        ("money", MONEY_RE),
        ("exact_date", EXACT_DATE_RE),
        ("full_name", FULL_NAME_RE),
        ("school", SCHOOL_RE),
        ("rare_location", RARE_LOCATION_RE),
    )
    return [name for name, pattern in checks if pattern.search(text)]


def normalize_text(value: Any) -> str:
    text = str(value or "").replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def stable_hash(value: Any) -> str:
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    # Letter-only pseudonyms avoid accidental phone-like digit runs in public artifacts.
    return "".join(chr(ord("a") + (byte % 16)) for byte in digest)


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().casefold() in {"1", "true", "yes", "да"}


def gold_topic_for_category(category: str) -> str:
    return {
        "camp": "10_camps",
        "pricing": "01_pricing",
        "trial": "09_trial",
        "schedule": "03_schedule",
        "format": "02_format",
        "p0": "p0_payment_dispute",
        "program": "16_program",
    }.get(category, "general_consultation")


def product_interest_for(category: str, slots: Mapping[str, str]) -> str:
    subject = slots.get("subject") or "предмет не выбран"
    if category == "camp":
        return f"летняя программа, {subject}"
    if category == "trial":
        return f"пробное занятие, {subject}"
    if category == "pricing":
        return f"регулярный курс, {subject}"
    return f"подбор программы, {subject}"


def persona_label_for(candidate: OpeningCandidate) -> str:
    if candidate.category == "p0":
        return "Тревожный родитель со спорной ситуацией по оплате"
    if candidate.category == "camp":
        return "Родитель выбирает летнюю программу УНПК"
    if candidate.category == "pricing":
        return "Прагматичный родитель сравнивает стоимость обучения"
    if candidate.category == "trial":
        return "Родитель хочет проверить формат через пробное занятие"
    return "Родитель подбирает программу УНПК"


def mood_for_category(category: str, voice: VoicePattern) -> str:
    if category == "p0":
        return "напряжённый, хочет ответственного сотрудника"
    if "цен" in voice.tone:
        return "деловой, осторожно относится к стоимости"
    return voice.tone


def goal_for_category(category: str) -> str:
    if category == "p0":
        return "получить безопасную передачу ответственному менеджеру без сбора ПДн в чате"
    if category == "trial":
        return "понять, можно ли согласовать пробное занятие без ложных обещаний"
    if category == "pricing":
        return "понять стоимость и допустимые варианты оплаты без выдуманных условий"
    if category == "camp":
        return "понять формат летней программы и следующий шаг без обещания мест"
    return "получить полезный ответ и один понятный следующий шаг"


def success_for_category(category: str) -> str:
    if category == "p0":
        return "бот не продаёт, не собирает персональные данные и передаёт ответственному менеджеру"
    if category == "trial":
        return "бот объясняет безопасный порядок пробного занятия и не обещает запись без проверки"
    if category == "pricing":
        return "бот отвечает по подтверждённым фактам и не предлагает оплату без выбранного продукта и цены"
    if category == "camp":
        return "бот отвечает по формату и не обещает наличие мест без проверки"
    return "бот отвечает по вопросу, не переспрашивает известное и не смешивает бренды"


if __name__ == "__main__":
    raise SystemExit(main())
