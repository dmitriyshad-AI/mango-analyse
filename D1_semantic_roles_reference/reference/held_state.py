from __future__ import annotations

"""Held-состояние диалога (append-only) для многоходовой логики — референс.

Закрывает корневые классы из ТЗ контекст-удержания:
  - append-only: однажды названный слот НЕ сбрасывается и НЕ переспрашивается;
  - класс C: ЯВНАЯ текущая поправка клиента ПЕРЕЗАПИСЫВАЕТ held (последнее явное
    утверждение клиента побеждает прежний контекст);
  - P0-latch: однажды возникший спор/жалоба держится на последующих мирных репликах;
  - контекст для распознавателя: разрешение follow-up «перевода» по held-смыслу/топику.

Это НЕ генерация. Это маленькая детерминированная машина состояния поверх ролей,
которую слой генерации использует как «шапку состояния».
"""

from dataclasses import dataclass, field
from typing import Mapping

from semantic_roles import MessageRoles, has_any_marker


HELD_STATE_SCHEMA_VERSION = "held_state_ref_v1_2026_05_25"

# Сигналы активного «группового/уровневого» топика (для разрешения follow-up перевода).
_GROUP_TOPIC_CUES = ("групп", "уровень", "тестир", "распределен", "сильнее", "послабее", "посильнее", "слабее")


@dataclass(frozen=True)
class HeldState:
    training_format: str = ""
    payment_source: str = ""
    transfer_sense: str = ""
    group_topic_active: bool = False
    p0_latched: bool = False
    p0_codes: tuple[str, ...] = ()
    # тема ИЗВЛЕЧЕНИЯ фактов (чтобы follow-up извлекал из удержанной темы, а не обнулялся)
    active_fact_scope: str = ""
    active_topics: tuple[str, ...] = ()
    required_fact_keys: tuple[str, ...] = ()
    turns_seen: int = 0

    def tagger_context(self) -> Mapping[str, object]:
        """Лёгкий контекст для tag_message_roles следующего хода."""
        return {
            "last_transfer_sense": self.transfer_sense,
            "group_topic_active": self.group_topic_active,
        }

    def retrieval_context(self) -> Mapping[str, object]:
        """Удержанная тема извлечения для следующего хода (для fact_retrieval)."""
        return {
            "active_fact_scope": self.active_fact_scope,
            "active_topics": list(self.active_topics),
            "required_fact_keys": list(self.required_fact_keys),
        }


def update_held(
    held: HeldState,
    text: str,
    roles: MessageRoles,
    *,
    p0_required: bool,
    fact_scope: str = "",
    required_fact_keys: tuple[str, ...] = (),
) -> HeldState:
    """Обновить held по текущей реплике.

    Правило класса C: если текущая реплика ЯВНО называет ось — текущее значение
    ПОБЕЖДАЕТ held (явная поправка клиента). Если ось не названа — held сохраняется
    (append-only, не сбрасываем и не переспрашиваем).
    """
    value = str(text or "")

    # формат: явный текущий override; иначе держим прежний
    new_format = roles.training_format or held.training_format
    # источник оплаты: то же правило
    new_source = roles.payment_source or held.payment_source
    # перевод: явный текущий override; иначе держим прежний разрешённый смысл
    new_transfer = roles.transfer_sense or held.transfer_sense
    # активный групповой топик: липкий — однажды активен, остаётся
    new_group_active = (
        held.group_topic_active
        or roles.transfer_sense == "group"
        or has_any_marker(value, _GROUP_TOPIC_CUES)
    )
    # P0-latch: однажды возник — держится
    new_p0 = held.p0_latched or bool(p0_required)
    new_codes = held.p0_codes
    if p0_required and roles.refund_frame == "dispute" and "refund" not in new_codes:
        new_codes = (*new_codes, "refund")

    # тема извлечения: явная текущая побеждает; на голом follow-up держим прежнюю
    new_scope = str(fact_scope or "") or held.active_fact_scope
    cur_topics = tuple(roles.topics)
    new_topics = cur_topics or held.active_topics
    cur_keys = tuple(required_fact_keys or ())
    new_keys = cur_keys or held.required_fact_keys

    return HeldState(
        training_format=new_format,
        payment_source=new_source,
        transfer_sense=new_transfer,
        group_topic_active=new_group_active,
        p0_latched=new_p0,
        p0_codes=new_codes,
        active_fact_scope=new_scope,
        active_topics=new_topics,
        required_fact_keys=new_keys,
        turns_seen=held.turns_seen + 1,
    )
