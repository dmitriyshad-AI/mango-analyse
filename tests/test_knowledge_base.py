from __future__ import annotations

import csv
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import mango_mvp.insights.sanitizers as sanitizer_module
from mango_mvp.insights.knowledge_base import (
    KnowledgeBaseConfig,
    build_sales_insight_knowledge_base,
    classify_answer_pattern,
    commercial_usefulness,
    enrich_review_row,
    is_trusted_llm_review,
    outcome_group,
    quality_band,
)
from mango_mvp.insights.sanitizers import (
    has_brand_risk,
    has_money_or_terms_risk,
    has_personal_data_risk,
    sanitize_answer,
)


def _row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "moment_id": "pilot-00001",
        "phone": "79000000000",
        "source_filename": "call.mp3",
        "started_at": "2026-04-01 10:00:00",
        "manager_name": "Менеджер",
        "llm_customer_signal_type": "price_question",
        "llm_hidden_sales_stage": "price_discussion",
        "final_outcome_label": "payment_pending",
        "outcome_confidence_tier": "strong",
        "customer_question": "Сколько стоит курс?",
        "manager_answer": "Менеджер объяснил стоимость, ценность программы и отправил ссылку на оплату.",
        "overall_quality_score": 82,
        "extraction_confidence": 0.88,
        "what_manager_did_well": "Ответил на цену",
        "what_manager_missed": "",
        "ideal_reaction": "Объяснить ценность и следующий шаг.",
        "ideal_answer_example": "Стоимость зависит от формата. Я пришлю расчет, а сейчас кратко объясню, что входит в курс.",
        "risk_flags": "",
        "avoid_using_when": "",
        "customer_quote": "Сколько стоит?",
        "manager_quote": "Отправлю ссылку",
        "history_summary": "Клиент спрашивал цену.",
        "rubric_factual_correctness": 80,
        "rubric_completeness": 80,
        "rubric_persuasiveness": 80,
        "rubric_personalization": 80,
        "rubric_objection_handling": 80,
        "rubric_next_step_clarity": 80,
        "rubric_empathy_tone": 80,
        "rubric_sales_discipline": 80,
        "provider": "codex_cli",
        "review_source": "live_gpt55",
    }
    base.update(overrides)
    return base


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_quality_and_outcome_grouping() -> None:
    assert quality_band(80) == "high"
    assert quality_band(60) == "medium"
    assert quality_band(40) == "low"
    assert outcome_group("won_paid_or_active") == "paid_or_payment_path"
    assert outcome_group("reopen_or_follow_up_opportunity") == "follow_up_opportunity"
    assert outcome_group("lost_or_refused") == "lost_or_churn"


def test_classify_answer_pattern_for_core_cases() -> None:
    assert classify_answer_pattern(_row()) == "price_payment_handled_with_value_or_instruction"
    assert classify_answer_pattern(
        _row(
            llm_customer_signal_type="schedule_question",
            manager_answer="Перезвоним как только будет понятно.",
            risk_flags="Нет точной даты follow-up.",
        )
    ) == "vague_or_missing_next_step"
    assert classify_answer_pattern(
        _row(
            llm_customer_signal_type="technical_or_access_issue",
            llm_hidden_sales_stage="existing_client_service",
            manager_answer="Передам в поддержку, проверим доступ и продублируем ссылку.",
        )
    ) == "service_resolution_or_escalation"
    assert classify_answer_pattern(_row(manager_answer="Абонент недоступен, оставили голосовое сообщение.")) == "no_live_contact_or_voicemail"
    assert classify_answer_pattern(_row(manager_answer="Продолжение следует...")) == "no_live_contact_or_voicemail"


def test_enrich_review_row_adds_business_fields() -> None:
    enriched = enrich_review_row(_row())

    assert enriched["review_trust_status"] == "trusted_llm_review"
    assert enriched["quality_band"] == "high"
    assert enriched["outcome_group"] == "paid_or_payment_path"
    assert enriched["commercial_usefulness"] == "playbook_candidate"
    assert enriched["bot_seed_status"] == "ready_for_bot_draft"
    assert enriched["signal_ru"] == "Вопрос о цене"
    assert enriched["final_outcome_ru"] == "Есть путь к оплате"
    assert enriched["answer_pattern_ru"] == "Цена/оплата объяснены через ценность или инструкцию"
    assert "мессенджеры" in enriched["data_scope_note"]
    assert commercial_usefulness(_row(final_outcome_label="lost_or_refused"), 40, "lost_or_refused") == "revenue_leakage_risk"
    assert (
        commercial_usefulness(_row(manager_answer="Продолжение следует...", final_outcome_label="lost_or_refused"), 30, "lost_or_refused")
        == "process_fix_needed"
    )


def test_untrusted_dry_run_review_is_not_ready_for_kb_or_bot() -> None:
    row = _row(provider="dry_run", review_source="deterministic_fallback_needs_llm_refresh")
    enriched = enrich_review_row(row)

    assert not is_trusted_llm_review(enriched)
    assert enriched["review_trust_status"] == "needs_live_llm_refresh"
    assert enriched["commercial_usefulness"] == "needs_llm_refresh"
    assert enriched["bot_seed_status"] == "not_ready_needs_llm_refresh"
    assert "LLM-refresh" in enriched["rop_action"]


def test_sanitize_answer_normalizes_brand_money_terms_and_personal_data() -> None:
    raw = (
        "Ольга Михайловна, в НПК МФТИ стоимость 50 000 рублей, скидка 10%, "
        "рассрочка на 12 месяцев и возврат до 15 мая. Пишите на test@example.com или +7 900 123-45-67."
    )

    manager = sanitize_answer(raw, mode="manager")
    bot = sanitize_answer(raw, mode="bot")

    assert "Фотон" in manager.text
    assert "НПК" not in manager.text
    assert "Ольга Михайловна" not in manager.text
    assert "50 000" not in manager.text
    assert "10%" not in manager.text
    assert "test@example.com" not in manager.text
    assert "+7 900" not in manager.text
    assert "наш учебный центр" in bot.text
    assert "Точные условия менеджер подтвердит" in bot.text
    assert not has_brand_risk(bot.text)
    assert not has_money_or_terms_risk(bot.text)
    assert not has_personal_data_risk(bot.text)
    assert "brand_normalized" in manager.flags
    assert "price_redacted" in manager.flags
    assert "person_name_redacted" in manager.flags


def test_sanitize_answer_catches_hidden_dates_spoken_percent_and_single_names() -> None:
    raw = (
        "Мария, бронь держим до пятницы, 10 апреля. Оплата 50к, скидка 10 процентов. "
        "Михаил занимается в субботу 10:00-12:00."
    )

    bot = sanitize_answer(raw, mode="bot")

    assert "Мария" not in bot.text
    assert "Михаил" not in bot.text
    assert "пятницы" not in bot.text.lower()
    assert "10 апреля" not in bot.text.lower()
    assert "50к" not in bot.text.lower()
    assert "10 процентов" not in bot.text.lower()
    assert "10:00" not in bot.text
    assert not has_money_or_terms_risk(bot.text)
    assert not has_personal_data_risk(bot.text)
    assert "deadline_redacted" in bot.flags
    assert "price_redacted" in bot.flags
    assert "percent_redacted" in bot.flags
    assert "person_name_redacted" in bot.flags


def test_sanitize_answer_catches_stage15_adversarial_bot_export_risks() -> None:
    raw = "Максим получит ссылку @anna_photon, стоимость пятьдесят тысяч рублей или 50 т.р., звонить с 10 до 22."

    bot = sanitize_answer(raw, mode="bot")

    assert "Максим" not in bot.text
    assert "@anna_photon" not in bot.text
    assert "пятьдесят тысяч" not in bot.text.lower()
    assert "50 т" not in bot.text.lower()
    assert "с 10 до 22" not in bot.text.lower()
    assert not has_money_or_terms_risk(bot.text)
    assert not has_personal_data_risk(bot.text)


def test_sanitize_answer_catches_claude_stage15_price_leak_patterns() -> None:
    raw = (
        "По оплате: физика 7900 за 4 занятия. "
        "Первый семестр за 88000, год целиком за 147000. "
        "При ранней оплате 78400."
    )

    bot = sanitize_answer(raw, mode="bot")

    assert "7900" not in bot.text
    assert "88000" not in bot.text
    assert "147000" not in bot.text
    assert "78400" not in bot.text
    assert "актуальную стоимость" in bot.text
    assert "price_redacted" in bot.flags
    assert not has_money_or_terms_risk(bot.text)


def test_sanitize_answer_catches_claude_stage15_location_teacher_deadline_and_promise_patterns() -> None:
    raw = (
        "Преподаватель Лукина ждет вас в Долгопрудном: проспект Пацаева, 7 корпус 1, "
        "4 этаж, кабинет 49, рядом со Скорняжным переулком и Чистыми прудами. До конца дня вернемся с подтверждением. "
        "Письмо от Альфа-банка придет на почту vidu@. . . в районе Сухаревки. "
        "Файл Word «Разбивка 1» отправим отдельно."
    )

    bot = sanitize_answer(raw, mode="bot")

    assert "Лукина" not in bot.text
    assert "Долгопруд" not in bot.text
    assert "Пацаева" not in bot.text
    assert "Скорняж" not in bot.text
    assert "Чист" not in bot.text
    assert "кабинет 49" not in bot.text
    assert "до конца дня" not in bot.text.lower()
    assert "Альфа" not in bot.text
    assert "vidu@" not in bot.text
    assert "Сухарев" not in bot.text
    assert "Разбивка 1" not in bot.text
    assert "адрес, который подтвердит менеджер" in bot.text
    assert "менеджер свяжется с вами после проверки" in bot.text
    assert "role_name_redacted" in bot.flags
    assert "location_redacted" in bot.flags
    assert "service_promise_redacted" in bot.flags
    assert not has_money_or_terms_risk(bot.text)
    assert not has_personal_data_risk(bot.text)


def test_sanitize_answer_catches_claude_reaudit_orphan_names_dates_and_compensation() -> None:
    raw = (
        "По ученик Николаевне нет статистики, но физику будет вести Кондрашова. "
        "Преподаватель - ученик Гамзяков, очная группа ученик Еделькина. "
        "Будет ли Камаринцев вести информатику? По Камаринцеву уточним. "
        "Скажите фамилию Николаев, подойдите к вахте в КПМ на Майской, кабинет 324. "
        "Пусть Катерина восстановится, действует до 15 числа, тестирование до 17 числа, "
        "важно компенсировать занятие."
    )

    bot = sanitize_answer(raw, mode="bot")

    for leaked in (
        "Николаев",
        "Кондраш",
        "Гамзяк",
        "Еделькин",
        "Камаринц",
        "Майск",
        "КПМ",
        "кабинет 324",
        "Катерин",
        "15 числа",
        "17 числа",
        "компенсировать",
    ):
        assert leaked not in bot.text
    assert "актуальное окно записи" in bot.text
    assert "адрес, который подтвердит менеджер" in bot.text
    assert "менеджер свяжется с вами после проверки" in bot.text
    assert "person_name_redacted" in bot.flags
    assert "location_redacted" in bot.flags
    assert "deadline_redacted" in bot.flags
    assert "service_promise_redacted" in bot.flags
    assert not has_money_or_terms_risk(bot.text)
    assert not has_personal_data_risk(bot.text)


def test_sanitize_answer_is_strongly_idempotent_on_adversarial_bot_cases() -> None:
    cases = [
        (
            "Ольга Михайловна, в НПК МФТИ стоимость 50 000 рублей, скидка 10%, "
            "рассрочка до 15 мая, пишите на test@example.com."
        ),
        "Первый семестр за 88000, год целиком за 147000. Физика 7900 за 4 занятия.",
        (
            "Преподаватель Лукина ждет в Долгопрудном: проспект Пацаева 7 корпус 1, "
            "кабинет 49. До конца дня вернемся."
        ),
        (
            "По ученик Николаевне нет статистики, будет вести Кондрашова. "
            "Преподаватель - ученик Гамзяков, действует до 17 числа."
        ),
        "Менеджер свяжется после проверки. Точные условия менеджер подтвердит по актуальным правилам.",
    ]

    for raw in cases:
        first = sanitize_answer(raw, mode="bot")
        second = sanitize_answer(first.text, mode="bot")

        assert first.fixpoint_reached is True
        assert first.pass_count >= 1
        assert second.text == first.text
        assert second.fixpoint_reached is True


def test_sanitize_answer_blocks_when_fixpoint_is_not_reached(monkeypatch) -> None:
    def unstable_pass(text: object, *, mode: sanitizer_module.SanitizerMode = "bot") -> sanitizer_module.SanitizedText:
        return sanitizer_module.SanitizedText(
            f"{sanitizer_module.clean_text(text)} x",
            ("person_name_redacted",),
            "safe_with_placeholders",
            pass_count=1,
        )

    monkeypatch.setattr(sanitizer_module, "_sanitize_answer_pass", unstable_pass)

    result = sanitizer_module.sanitize_answer("циклический текст", mode="bot", max_passes=3)

    assert result.status == "fixpoint_not_reached"
    assert result.text == ""
    assert result.fixpoint_reached is False
    assert result.pass_count == 3


def test_sanitize_customer_text_blocks_when_fixpoint_is_not_reached(monkeypatch) -> None:
    def unstable_pass(text: object, *, mode: sanitizer_module.SanitizerMode = "customer") -> sanitizer_module.SanitizedText:
        return sanitizer_module.SanitizedText(
            f"{sanitizer_module.clean_text(text)} x",
            ("phone_redacted",),
            "safe_with_placeholders",
            pass_count=1,
        )

    monkeypatch.setattr(sanitizer_module, "_sanitize_answer_pass", unstable_pass)

    result = sanitizer_module.sanitize_answer("циклический клиентский текст", mode="customer", max_passes=3)

    assert result.status == "fixpoint_not_reached"
    assert result.text == ""
    assert result.fixpoint_reached is False
    assert result.pass_count == 3


def test_sanitize_manager_text_keeps_diagnostic_text_when_fixpoint_is_not_reached(monkeypatch) -> None:
    def unstable_pass(text: object, *, mode: sanitizer_module.SanitizerMode = "manager") -> sanitizer_module.SanitizedText:
        return sanitizer_module.SanitizedText(
            f"{sanitizer_module.clean_text(text)} x",
            ("price_redacted",),
            "safe_with_placeholders",
            pass_count=1,
        )

    monkeypatch.setattr(sanitizer_module, "_sanitize_answer_pass", unstable_pass)

    result = sanitizer_module.sanitize_answer("циклический текст", mode="manager", max_passes=3)

    assert result.status == "fixpoint_not_reached"
    assert result.text == "циклический текст x x x"
    assert result.fixpoint_reached is False
    assert result.pass_count == 3


def test_sanitize_answer_stable_text_keeps_existing_client_safe_parity() -> None:
    result = sanitize_answer("Менеджер подтвердит актуальную стоимость.", mode="bot")

    assert result.status == "safe_no_changes"
    assert result.text == "Менеджер подтвердит актуальную стоимость."
    assert result.fixpoint_reached is True


def test_enrich_review_row_adds_manager_and_bot_safe_answers() -> None:
    enriched = enrich_review_row(
        _row(
            ideal_answer_example=(
                "Ольга Михайловна, в НПК МФТИ стоимость 50 000 рублей, "
                "скидка 10% действует до 15 мая, есть рассрочка."
            )
        )
    )

    assert enriched["bot_seed_status"] == "needs_rop_validation"
    assert "Идеальный" not in enriched["bot_safe_answer"]
    assert "[CLIENT_NAME]" not in enriched["bot_safe_answer"]
    assert "Фотон" in enriched["ideal_answer_manager_sanitized"]
    assert "НПК" not in enriched["ideal_answer_manager_sanitized"]
    assert "50 000" not in enriched["ideal_answer_manager_sanitized"]
    assert "10%" not in enriched["bot_safe_answer"]
    assert "до 15 мая" not in enriched["bot_safe_answer"].lower()
    assert not has_money_or_terms_risk(enriched["bot_safe_answer"])
    assert not has_brand_risk(enriched["bot_safe_answer"])
    assert "price_redacted" in enriched["sanitizer_flags"]
    assert enriched["brand_risk_flag"] == "Да"
    assert enriched["money_or_discount_flag"] == "Да"
    assert enriched["installment_flag"] == "Да"
    assert enriched["deadline_or_promise_flag"] == "Да"
    assert enriched["personal_data_flag"] == "Да"


def test_build_sales_insight_knowledge_base_outputs_workbook(tmp_path: Path) -> None:
    rows = [
        _row(moment_id="pilot-00001", manager_name="Анна", overall_quality_score=82),
        _row(
            moment_id="pilot-00002",
            manager_name="Олег",
            llm_customer_signal_type="schedule_question",
            final_outcome_label="lost_or_refused",
            manager_answer="Перезвоним позже.",
            risk_flags="Нет точной даты следующего контакта.",
            overall_quality_score=42,
        ),
        _row(
            moment_id="pilot-00003",
            manager_name="Олег",
            llm_customer_signal_type="technical_or_access_issue",
            final_outcome_label="service_or_existing_context",
            manager_answer="Передам в поддержку и продублирую ссылку.",
            overall_quality_score=73,
        ),
        _row(
            moment_id="pilot-00004",
            manager_name="Анна",
            provider="dry_run",
            review_source="deterministic_fallback_needs_llm_refresh",
            overall_quality_score=95,
            ideal_answer_example="Этот fallback выглядит хорошо, но не должен попасть в базу бота без live LLM-review.",
        ),
        _row(
            moment_id="pilot-00005",
            manager_name="Анна",
            ideal_answer_example="В НПК МФТИ стоимость 50 000 рублей, скидка 10% до 15 мая.",
            customer_question="Сколько стоит для Ольги Михайловны?",
            customer_quote="Ольга Михайловна спросила про +79001234567",
        ),
    ]
    reviews_csv = tmp_path / "reviews.csv"
    _write_csv(reviews_csv, rows)

    summary = build_sales_insight_knowledge_base(
        KnowledgeBaseConfig(
            project_root=tmp_path,
            reviews_csv=reviews_csv,
            out_root=tmp_path / "kb",
            min_group_count=1,
            top_examples=10,
        )
    )

    assert summary["totals"]["reviews"] == 5
    assert summary["llm_review"]["trusted_llm_reviews"] == 4
    assert summary["llm_review"]["needs_live_llm_refresh"] == 1
    assert summary["sanitizer"]["bot_safe_answer_rows"] >= 1
    assert summary["quality"]["low_quality_count"] == 1
    workbook_path = tmp_path / "kb" / "sales_insight_knowledge_base.xlsx"
    assert workbook_path.exists()
    assert (tmp_path / "kb" / "bot_knowledge_seeds.csv").exists()
    assert (tmp_path / "kb" / "llm_refresh_queue.csv").exists()
    brief = (tmp_path / "kb" / "signal_summary.csv").read_text(encoding="utf-8-sig")
    assert "Вопрос о цене" in brief
    assert "Только звонки" in brief
    bot_seeds = (tmp_path / "kb" / "bot_knowledge_seeds.csv").read_text(encoding="utf-8-sig")
    assert "pilot-00004" not in bot_seeds
    assert "50 000" not in bot_seeds
    assert "НПК" not in bot_seeds
    assert "Безопасный ответ для бота" in bot_seeds
    llm_refresh = (tmp_path / "kb" / "llm_refresh_queue.csv").read_text(encoding="utf-8-sig")
    assert "pilot-00004" in llm_refresh
    with zipfile.ZipFile(workbook_path) as xlsx:
        workbook_xml = xlsx.read("xl/workbook.xml")
    root = ElementTree.fromstring(workbook_xml)
    sheet_names = [sheet.attrib["name"] for sheet in root.findall(".//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}sheet")]
    assert "Сводка РОПа" in sheet_names
    assert "Лучшие ответы" in sheet_names
    assert "LLM refresh" in sheet_names
    assert "ROP brief" not in sheet_names
