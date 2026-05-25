from __future__ import annotations

import json
import re
from pathlib import Path

from mango_mvp.channels.answer_plan import build_answer_plan
from mango_mvp.channels.held_state import HeldState, update_held
from mango_mvp.channels.semantic_roles import tag_message_roles


V8 = Path("/Users/dmitrijfabarisov/Claude Projects/Foton/v8_dynamic_client_sim_2026-05-25_v2.jsonl")

REFUND_CUE = re.compile(r"возврат|верн[уои]\w*\s+(?:деньг|оплат|средств|сумм)|деньги\s+назад|расторг|отказ\w*\s+от\s+обуч|забрать\s+деньг", re.I)
DEMAND = re.compile(r"верните|требую|хочу\s+(?:возврат|вернуть|деньги\s+назад)|немедленно\s+верн", re.I)
PAID = re.compile(r"оплатил|оплатила|уже\s+оплат|после\s+оплат|списал|заключил\w*\s+договор|мы\s+платил|я\s+платил|с\s+меня\s+сн", re.I)
PRESALE = re.compile(r"\bесли\b|вдруг|до\s+начал|до\s+оплат|перед\s+оплат|заранее|передума|не\s+понрав|не\s+подойд|какие\s+услови|какие\s+правил", re.I)
ZAPIS = re.compile(r"запис|оформ", re.I)
TRANSFER = re.compile(r"перевод|перевест|перевед|переведите|переключите", re.I)


def test_full_v8_semantic_roles_have_no_structural_anomalies() -> None:
    anomalies: dict[str, list[str]] = {}
    persona_count = 0
    message_count = 0

    def flag(category: str, dialog_id: str, message: str, extra: str = "") -> None:
        anomalies.setdefault(category, []).append(f"[{dialog_id}] {message!r}{(' — ' + extra) if extra else ''}")

    with V8.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            persona = json.loads(line)
            if persona.get("type") != "persona":
                continue
            persona_count += 1
            held = HeldState()
            for message in persona.get("behaviors", []):
                text = str(message or "")
                if not text.strip():
                    continue
                message_count += 1
                roles = tag_message_roles(text, context=held.tagger_context())
                plan = build_answer_plan(roles, external_p0=held.p0_latched)
                held = update_held(held, text, roles, p0_required=plan.p0_required)

                refund_cue = bool(REFUND_CUE.search(text))
                demand = bool(DEMAND.search(text))
                paid = bool(PAID.search(text))
                presale = bool(PRESALE.search(text))

                if refund_cue and roles.refund_frame == "none":
                    flag("refund_missed", persona.get("dialog_id", "?"), text)
                if roles.refund_frame == "dispute" and presale and not demand and not paid:
                    flag("refund_false_dispute", persona.get("dialog_id", "?"), text, f"ev={roles.evidence.get('refund_frame')}")
                if roles.refund_frame == "dispute" and not plan.p0_required:
                    flag("dispute_not_p0", persona.get("dialog_id", "?"), text)
                if roles.payment_source == "matkap" and "matkap+installment" not in plan.forbidden_pairs:
                    flag("matkap_invariant_break", persona.get("dialog_id", "?"), text)
                if ZAPIS.search(text) and roles.enrollment_vs_recording == "":
                    flag("zapis_unresolved", persona.get("dialog_id", "?"), text)
                if TRANSFER.search(text) and roles.transfer_sense == "":
                    flag("transfer_unresolved", persona.get("dialog_id", "?"), text)

    assert persona_count == 322
    assert message_count == 1137
    assert anomalies == {}
