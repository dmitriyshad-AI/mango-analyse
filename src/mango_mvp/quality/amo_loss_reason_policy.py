from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class AmoLossReasonPolicy:
    class_id: str
    risk_type: str
    severity: str
    category: str
    label: str
    manager_status: str
    manager_action: str
    matched_text: str
    reason: str


@dataclass(frozen=True)
class _PolicyDefinition:
    pattern: re.Pattern[str]
    class_id: str
    risk_type: str
    severity: str
    category: str
    label: str
    manager_status: str
    manager_action: str
    reason: str


ACTIVE_CLIENT_LOSS_REASON_RE = re.compile(
    r"\b(?:действующ\w+\s+клиент\w*|текущ\w+\s+клиент\w*|"
    r"действующ\w+\s+ученик\w*|текущ\w+\s+ученик\w*|уже\s+учит\w+|уже\s+занима\w+)\b",
    re.I,
)
DUPLICATE_LOSS_REASON_RE = re.compile(r"\b(?:дубл\w+|объедин[её]н\w+\s+карточк\w+)\b", re.I)
NO_APPLICATION_LOSS_REASON_RE = re.compile(
    r"\b(?:не\s+оставлял[а-я]*\s+заявк\w+|заявк\w+\s+не\s+оставлял[а-я]*|не\s+тот\s+клиент)\b",
    re.I,
)
NO_CONTACT_ARCHIVE_LOSS_REASON_RE = re.compile(
    r"\b(?:архив|нет\s+связи|недозвон|не\s+дозвон|не\s+дозвонил[а-я]*)\b",
    re.I,
)
TERMINAL_LOST_LOSS_REASON_RE = re.compile(
    r"\b(?:не\s*актуальн\w+|дорого|уш[её]л\w*\s+к\s+конкурент\w+|конкурент\w+|"
    r"выбрал[а-я]*\s+репетитор\w+|выбрали\s+репетитор\w+|репетитор\w+)\b",
    re.I,
)
NOT_QUALIFIED_OR_OUT_OF_SCOPE_LOSS_REASON_RE = re.compile(
    r"\b(?:не\s*квал\w*|неквал\w*|не\s+целев\w+|нецелев\w+|жуковск\w+|"
    r"не\s+подходит\s+формат|шд\s+жако)\b",
    re.I,
)
INVALID_OR_TEST_LOSS_REASON_RE = re.compile(r"\b(?:спам|тест)\b", re.I)
FUTURE_PROSPECT_LOSS_REASON_RE = re.compile(r"\b(?:перспектив\w*)\b", re.I)
COMPANY_SIDE_LOSS_REASON_RE = re.compile(r"\b(?:закрыл[а-я]*\s+групп\w+)\b", re.I)
REFUND_OR_POSTSALE_LOSS_REASON_RE = re.compile(r"\b(?:возврат)\b", re.I)
GRADUATE_LOSS_REASON_RE = re.compile(r"\b(?:выпускник\w*)\b", re.I)
AMBIGUOUS_LOSS_REASON_RE = re.compile(r"^\s*другое\s*$", re.I)


_POLICIES = (
    _PolicyDefinition(
        pattern=ACTIVE_CLIENT_LOSS_REASON_RE,
        class_id="Q4j",
        risk_type="active_client_loss_reason_requires_entity_resolution",
        severity="P1",
        category="active_client_entity_resolution",
        label="действующий клиент",
        manager_status="закрыта как действующий клиент; нужна актуальная карточка/сделка",
        manager_action="Найти актуальную карточку контакта/сделку/другой номер, не считать эту сделку потерянной продажей.",
        reason="AMO loss reason says the person is an active client; do not treat this closed lead as a lost sale or active sales target",
    ),
    _PolicyDefinition(
        pattern=DUPLICATE_LOSS_REASON_RE,
        class_id="Q4k",
        risk_type="duplicate_loss_reason_requires_entity_resolution",
        severity="P1",
        category="duplicate_entity_resolution",
        label="дубль/объединенная карточка",
        manager_status="закрыта как дубль; нужна каноническая карточка/сделка",
        manager_action="Найти основную карточку/сделку и писать AI-контекст только туда.",
        reason="AMO loss reason marks this deal as duplicate/merged; writeback must target the canonical contact/deal only",
    ),
    _PolicyDefinition(
        pattern=NO_APPLICATION_LOSS_REASON_RE,
        class_id="Q4l",
        risk_type="no_application_loss_reason_blocks_sales_writeback",
        severity="P1",
        category="no_application_wrong_direction",
        label="не оставлял заявку / неверный контакт",
        manager_status="закрыта как неверное обращение; активная продажная рекомендация запрещена",
        manager_action="Не писать следующий шаг продаж; нужен свежий подтвержденный лид или исправление матчинга.",
        reason="AMO loss reason says the person did not leave an application or the contact direction is wrong",
    ),
    _PolicyDefinition(
        pattern=TERMINAL_LOST_LOSS_REASON_RE,
        class_id="Q4m",
        risk_type="terminal_lost_reason_blocks_active_sales_writeback",
        severity="P1",
        category="lost_or_not_actual",
        label="не актуально / конкурент / репетитор",
        manager_status="закрыта как потерянная или неактуальная; активный следующий шаг запрещен",
        manager_action="Не реанимировать закрытую сделку без нового входящего сигнала.",
        reason="AMO loss reason indicates a closed/lost or not-actual lead; do not generate active sales writeback",
    ),
    _PolicyDefinition(
        pattern=NO_CONTACT_ARCHIVE_LOSS_REASON_RE,
        class_id="Q4n",
        risk_type="no_contact_archive_loss_reason_requires_no_action",
        severity="P2",
        category="no_contact_archive",
        label="архив / недозвон / нет связи",
        manager_status="закрыта как недозвон/архив; нужен новый свежий сигнал",
        manager_action="Не создавать активную задачу продаж без нового звонка, заявки, сообщения или оплаты.",
        reason="AMO loss reason is archive/no-contact; this should not become an active sales task without a fresh signal",
    ),
    _PolicyDefinition(
        pattern=INVALID_OR_TEST_LOSS_REASON_RE,
        class_id="Q4q",
        risk_type="invalid_or_test_loss_reason_blocks_writeback",
        severity="P1",
        category="invalid_or_test_no_action",
        label="спам / тест",
        manager_status="закрыта как спам/тест; sales-writeback запрещен",
        manager_action="Не писать AI-продажный контекст; строка не является реальным лидом.",
        reason="AMO loss reason is spam/test; this is not a real sales target",
    ),
    _PolicyDefinition(
        pattern=FUTURE_PROSPECT_LOSS_REASON_RE,
        class_id="Q4r",
        risk_type="future_prospect_loss_reason_requires_reactivation_policy",
        severity="P2",
        category="future_prospect_reactivation",
        label="перспектива",
        manager_status="закрыта как перспектива; нужен отдельный план реактивации",
        manager_action="Не писать активный следующий шаг без даты/условия реактивации и свежего основания.",
        reason="AMO loss reason marks a future prospect; it requires a reactivation policy, not generic active sales writeback",
    ),
    _PolicyDefinition(
        pattern=COMPANY_SIDE_LOSS_REASON_RE,
        class_id="Q4s",
        risk_type="company_side_loss_reason_requires_review",
        severity="P2",
        category="company_side_unavailable",
        label="закрыли группу",
        manager_status="закрыта по причине со стороны компании; нужна проверка альтернатив",
        manager_action="Проверить, есть ли альтернативная группа/формат; не писать общий sales-next-step автоматически.",
        reason="AMO loss reason is caused by company-side capacity/group closure; review alternatives before writeback",
    ),
    _PolicyDefinition(
        pattern=REFUND_OR_POSTSALE_LOSS_REASON_RE,
        class_id="Q4t",
        risk_type="refund_or_postsale_loss_reason_requires_service_review",
        severity="P1",
        category="refund_or_postsale_service_review",
        label="возврат",
        manager_status="закрыта как возврат; это service/post-sale контекст",
        manager_action="Не писать продажный следующий шаг; проверить финансы, возврат и сервисную историю.",
        reason="AMO loss reason indicates refund/post-sale context; route to service/finance review",
    ),
    _PolicyDefinition(
        pattern=GRADUATE_LOSS_REASON_RE,
        class_id="Q4u",
        risk_type="graduate_loss_reason_requires_alumni_policy",
        severity="P2",
        category="graduate_or_alumni",
        label="выпускник",
        manager_status="закрыта как выпускник; нужен alumni/повторная продажа сценарий",
        manager_action="Не писать обычный следующий шаг продаж; применять отдельную политику выпускников/повторных продаж.",
        reason="AMO loss reason says graduate/alumni; generic lead sales writeback is unsafe",
    ),
    _PolicyDefinition(
        pattern=NOT_QUALIFIED_OR_OUT_OF_SCOPE_LOSS_REASON_RE,
        class_id="Q4o",
        risk_type="not_qualified_or_out_of_scope_loss_reason_requires_review",
        severity="P2",
        category="not_qualified_or_out_of_scope",
        label="не квалифицирован / вне продукта",
        manager_status="закрыта как неквалифицированная или вне целевого продукта",
        manager_action="Не писать продажный следующий шаг; нужна проверка, есть ли новый подходящий продукт/филиал.",
        reason="AMO loss reason says the lead is not qualified or out of scope; route to review instead of active writeback",
    ),
    _PolicyDefinition(
        pattern=AMBIGUOUS_LOSS_REASON_RE,
        class_id="Q4p",
        risk_type="ambiguous_loss_reason_requires_manual_review",
        severity="P2",
        category="ambiguous_other_manual_review",
        label="другое",
        manager_status="закрыта с неоднозначной причиной; нужна ручная проверка",
        manager_action="Перед записью AI-полей менеджер должен понять реальную причину закрытия.",
        reason="AMO loss reason is ambiguous; manager review is required before writing deal guidance",
    ),
)


def classify_amo_loss_reason(reason: str) -> list[AmoLossReasonPolicy]:
    result: list[AmoLossReasonPolicy] = []
    for policy in _POLICIES:
        match = policy.pattern.search(reason or "")
        if not match:
            continue
        result.append(
            AmoLossReasonPolicy(
                class_id=policy.class_id,
                risk_type=policy.risk_type,
                severity=policy.severity,
                category=policy.category,
                label=policy.label,
                manager_status=policy.manager_status,
                manager_action=policy.manager_action,
                matched_text=match.group(0),
                reason=policy.reason,
            )
        )
    return result


def primary_amo_loss_reason_policy(reason: str) -> AmoLossReasonPolicy | None:
    policies = classify_amo_loss_reason(reason)
    return policies[0] if policies else None
