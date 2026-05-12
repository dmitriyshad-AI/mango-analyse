from __future__ import annotations

import csv
import json
from pathlib import Path

from mango_mvp.insights.sanitizers import SanitizedText
import mango_mvp.quality.stage15_export_quality_gate as stage15_gate
from mango_mvp.quality.stage15_export_quality_gate import (
    BOT_EXPORT_COLUMNS,
    Stage15ExportGateConfig,
    build_stage15_export_quality_gate,
)


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _stage14_summary(root: Path, kb: Path, rop: Path, baseline: Path, *, passed: bool = True) -> dict[str, object]:
    checks = {
        "required_kb_columns_present": passed,
        "required_rop_columns_present": passed,
        "bot_seed_safe_columns_present": passed,
        "no_residual_bot_safe_risks": passed,
        "kb_no_live_revenue_risk_zero": passed,
        "rop_p0_no_live_or_artifact_zero": passed,
        "rop_revenue_no_live_or_artifact_zero": passed,
        "kb_bot_ready_money_or_terms_zero": passed,
        "rop_bot_candidate_money_or_terms_zero": passed,
        "bot_ready_rows_have_safe_answer": passed,
        "audit_sample_built": passed,
    }
    return {
        "acceptance": {"passed": passed, "checks": checks, "warnings": []},
        "residual_risk_samples": {"rows": 0 if passed else 1},
        "inputs": {
            "after_kb_summary": str((kb / "summary.json").resolve()),
            "after_kb_enriched": str((kb / "enriched_reviews.csv").resolve()),
            "after_kb_bot_seeds": str((kb / "bot_knowledge_seeds.csv").resolve()),
            "after_rop_summary": str((rop / "summary.json").resolve()),
            "after_rop_validation": str((rop / "rop_validation.csv").resolve()),
            "after_rop_bot_drafts": str((rop / "bot_knowledge_drafts.csv").resolve()),
            "after_baseline_summary": str((baseline / "summary.json").resolve()),
        },
        "outputs": {"summary_json": str((root / "summary.json").resolve())},
    }


def _baseline_summary(*, money: int = 0) -> dict[str, object]:
    return {
        "baseline_risks": {
            "kb_no_live_revenue_risk": 0,
            "kb_bot_ready_money_or_terms": money,
            "kb_ideal_answer_brand_risk": 0,
            "kb_bot_safe_answer_brand_risk": 0,
            "kb_bot_safe_answer_personal_data_risk": 0,
            "rop_p0_no_live_or_artifact": 0,
            "rop_revenue_risk_no_live_or_artifact": 0,
            "rop_bot_candidate_money_or_terms": 0,
            "rop_bot_safe_answer_brand_risk": 0,
            "rop_bot_safe_answer_personal_data_risk": 0,
            "kb_raw_ideal_answer_brand_risk": 5,
            "kb_raw_ideal_answer_money_or_terms": 50,
        },
        "kb_metrics": {"reviews": 10},
        "rop_metrics": {"rows": 5},
    }


def _build_fixture(
    root: Path,
    *,
    stage14_passed: bool = True,
    baseline_money_risk: int = 0,
    unsafe_bot_answer: bool = False,
    duplicate_audit_id: bool = False,
    over_sanitization_rows: int = 1,
    block_bot_on_over_sanitization: bool = True,
) -> Stage15ExportGateConfig:
    stage14 = root / "stage14"
    kb = root / "kb"
    rop = root / "rop"
    baseline = root / "baseline"
    out = root / "out"
    _write_json(stage14 / "summary.json", _stage14_summary(stage14, kb, rop, baseline, passed=stage14_passed))
    _write_json(baseline / "summary.json", _baseline_summary(money=baseline_money_risk))
    _write_json(kb / "summary.json", {"totals": {"reviews": 10}, "sanitizer": {"bot_safe_answer_rows": 2}})
    _write_json(rop / "summary.json", {"totals": {"source_reviews": 10}})
    audit_rows = []
    for idx in range(120):
        moment_id = "dup" if duplicate_audit_id and idx in {0, 1} else f"m-{idx:03d}"
        audit_rows.append({"audit_bucket": "money_terms_sanitized", "moment_id": moment_id})
    _write_csv(stage14 / "audit_sample.csv", audit_rows)
    _write_csv(stage14 / "residual_risk_sample.csv", [], fieldnames=["source", "moment_id", "text"])
    _write_csv(stage14 / "over_sanitization_candidates.csv", [{"moment_id": f"o-{idx}"} for idx in range(over_sanitization_rows)])

    safe_answer = "Стоимость 50 000 рублей." if unsafe_bot_answer else "Актуальные условия менеджер подтвердит по текущим правилам."
    _write_csv(
        kb / "bot_knowledge_seeds.csv",
        [
            {
                "ID момента": "kb-1",
                "Код сигнала": "price_question",
                "Сигнал клиента": "Вопрос о цене",
                "Стадия": "Выбор курса",
                "Код паттерна": "price_payment_handled",
                "Паттерн ответа": "Объяснить актуальные условия",
                "Пример вопроса клиента": "Сколько стоит курс?",
                "Черновик идеального ответа": "В НПК МФТИ цена 50 000 рублей.",
                "Безопасный ответ для бота": safe_answer,
                "Идеальный ответ для менеджера": "В Фотоне менеджер уточнит актуальные условия.",
                "Когда не использовать": "Если клиент просит персональную скидку.",
                "Статус sanitizer": "Безопасно после sanitization",
                "Флаги sanitizer": "price_redacted | brand_normalized",
                "Ограничение данных": "Только звонки.",
                "Оценка": 80,
                "Код итога": "follow_up",
                "Итог сделки": "Нужен follow-up",
            }
        ],
    )
    _write_csv(kb / "enriched_reviews.csv", [{"moment_id": "kb-1"}])
    _write_csv(
        rop / "bot_knowledge_drafts.csv",
        [
            {
                "ID момента": "rop-1",
                "Телефон": "+79990000000",
                "Менеджер": "Менеджер",
                "Файл звонка": "call.mp3",
                "Вопрос клиента": "Можно онлайн?",
                "Ответ менеджера": "Да.",
                "Идеальный ответ": "Да, можно онлайн.",
                "Безопасный ответ для бота": "Да, формат зависит от выбранной программы; менеджер подтвердит доступные варианты.",
                "Сигнал клиента": "Вопрос о формате",
                "Стадия": "Выбор формата",
                "Паттерн ответа": "Объяснить формат",
                "Статус sanitizer": "Безопасно без замен",
            }
        ],
    )
    _write_csv(rop / "rop_validation.csv", [{"ID момента": "rop-1", "Телефон": "+79990000000"}])
    return Stage15ExportGateConfig(
        project_root=root,
        stage14_root=stage14,
        kb_root=kb,
        rop_root=rop,
        baseline_root=baseline,
        out_root=out,
        min_audit_sample_rows=100,
        block_bot_production_on_over_sanitization_queue=block_bot_on_over_sanitization,
    )


def test_stage15_gate_builds_safe_bot_allowlist_and_blocks_autonomous_bot_until_review(tmp_path: Path) -> None:
    summary = build_stage15_export_quality_gate(_build_fixture(tmp_path))

    assert summary["passed"] is True
    assert summary["readiness"]["bot_allowlist_export_ready"] is True
    assert summary["readiness"]["bot_autonomous_production_ready"] is False
    assert summary["readiness"]["bot_autonomous_production_blockers"] == [
        "over_sanitization_queue_requires_rop_review_before_autonomous_bot"
    ]
    allowlist_path = Path(summary["outputs"]["bot_export_allowlist_csv"])
    rows = list(csv.DictReader(allowlist_path.open("r", encoding="utf-8-sig")))
    assert rows
    assert set(rows[0].keys()) == set(BOT_EXPORT_COLUMNS)
    forbidden = " ".join(rows[0].keys()).lower()
    assert "телефон" not in forbidden
    assert "менеджер" not in forbidden
    assert "идеальный ответ" not in forbidden
    assert "bot_safe_answer" in rows[0]


def test_stage15_gate_can_mark_bot_production_ready_after_usefulness_queue_policy_is_relaxed(tmp_path: Path) -> None:
    summary = build_stage15_export_quality_gate(
        _build_fixture(tmp_path, over_sanitization_rows=0, block_bot_on_over_sanitization=True)
    )

    assert summary["passed"] is True
    assert summary["readiness"]["bot_autonomous_production_ready"] is True


def test_stage15_gate_fails_when_stage14_acceptance_is_not_green(tmp_path: Path) -> None:
    summary = build_stage15_export_quality_gate(_build_fixture(tmp_path, stage14_passed=False))

    assert summary["passed"] is False
    assert summary["checks"]["stage14_acceptance_passed"] is False
    assert summary["checks"]["stage14_required_checks_passed"] is False


def test_stage15_gate_fails_when_required_baseline_risk_regresses(tmp_path: Path) -> None:
    summary = build_stage15_export_quality_gate(_build_fixture(tmp_path, baseline_money_risk=1))

    assert summary["passed"] is False
    assert summary["checks"]["baseline_required_risks_zero"] is False
    assert summary["baseline"]["non_zero_required_risks"] == {"kb_bot_ready_money_or_terms": 1}


def test_stage15_gate_fails_when_bot_safe_answer_contains_prices_or_terms(tmp_path: Path) -> None:
    summary = build_stage15_export_quality_gate(_build_fixture(tmp_path, unsafe_bot_answer=True))

    assert summary["passed"] is False
    assert summary["checks"]["source_bot_safe_answers_have_zero_risks"] is False
    assert summary["checks"]["bot_export_allowlist_has_zero_risks"] is False
    assert summary["row_counts"]["blocked_bot_export_rows"] == 1


def test_stage15_gate_blocks_claude_price_leak_patterns(tmp_path: Path) -> None:
    config = _build_fixture(tmp_path)
    _write_csv(
        config.kb_root / "bot_knowledge_seeds.csv",
        [
            {
                "ID момента": "kb-price-leak",
                "Сигнал клиента": "Вопрос о цене",
                "Пример вопроса клиента": "Сколько стоит?",
                "Безопасный ответ для бота": "Первый семестр за 88000, год целиком за 147000. Физика 7900 за 4 занятия.",
            }
        ],
    )

    summary = build_stage15_export_quality_gate(config)

    assert summary["passed"] is False
    risks = summary["bot_export"]["source_risk_counts"]["kb_bot_knowledge_seeds"]
    assert risks["money_or_terms"] == 1
    assert summary["row_counts"]["blocked_bot_export_rows"] == 1


def test_stage15_gate_blocks_claude_location_teacher_and_promise_patterns(tmp_path: Path) -> None:
    config = _build_fixture(tmp_path)
    _write_csv(
        config.kb_root / "bot_knowledge_seeds.csv",
        [
            {
                "ID момента": "kb-location-teacher",
                "Сигнал клиента": "Вопрос о филиале",
                "Пример вопроса клиента": "Куда приезжать?",
                "Безопасный ответ для бота": (
                    "Преподаватель Лукина ждет в Долгопрудном: проспект Пацаева, 7 корпус 1, "
                    "кабинет 49, Скорняжный переулок, рядом с Чистыми прудами. "
                    "До конца дня вернемся с подтверждением."
                ),
            }
        ],
    )

    summary = build_stage15_export_quality_gate(config)

    assert summary["passed"] is False
    risks = summary["bot_export"]["source_risk_counts"]["kb_bot_knowledge_seeds"]
    assert risks["money_or_terms"] == 1
    assert risks["personal_data"] == 1
    assert summary["row_counts"]["blocked_bot_export_rows"] == 1


def test_stage15_gate_blocks_claude_reaudit_orphan_names_dates_and_compensation(tmp_path: Path) -> None:
    config = _build_fixture(tmp_path)
    _write_csv(
        config.kb_root / "bot_knowledge_seeds.csv",
        [
            {
                "ID момента": "kb-re-audit-leak",
                "Сигнал клиента": "Вопрос о преподавателе",
                "Пример вопроса клиента": "Кто ведет?",
                "Безопасный ответ для бота": (
                    "По ученик Николаевне нет статистики, будет вести Кондрашова. "
                    "Преподаватель - ученик Гамзяков, группа ученик Еделькина. "
                    "Скажите фамилию Николаев, подойдите в КПМ Майская, кабинет 324. "
                    "Действует до 15 числа, тестирование до 17 числа, нужно компенсировать занятие."
                ),
            }
        ],
    )

    summary = build_stage15_export_quality_gate(config)

    assert summary["passed"] is False
    risks = summary["bot_export"]["source_risk_counts"]["kb_bot_knowledge_seeds"]
    assert risks["money_or_terms"] == 1
    assert risks["personal_data"] == 1
    assert summary["row_counts"]["blocked_bot_export_rows"] == 1


def test_stage15_gate_fails_on_independent_adversarial_export_risks(tmp_path: Path) -> None:
    config = _build_fixture(tmp_path)
    _write_csv(
        config.kb_root / "bot_knowledge_seeds.csv",
        [
            {
                "ID момента": "kb-unsafe",
                "Сигнал клиента": "Вопрос о цене",
                "Пример вопроса клиента": "Сколько стоит?",
                "Безопасный ответ для бота": "Максим получит ссылку @anna_photon, стоимость пятьдесят тысяч рублей.",
            }
        ],
    )

    summary = build_stage15_export_quality_gate(config)

    assert summary["passed"] is False
    risks = summary["bot_export"]["source_risk_counts"]["kb_bot_knowledge_seeds"]
    assert risks["spoken_money_or_terms"] == 1
    assert risks["messenger_handle"] == 1
    assert risks["likely_single_name"] == 1
    assert summary["row_counts"]["blocked_bot_export_rows"] == 1


def test_stage15_gate_blocks_rows_when_sanitizer_fixpoint_is_not_reached(tmp_path: Path, monkeypatch) -> None:
    config = _build_fixture(tmp_path, over_sanitization_rows=0)
    _write_csv(
        config.kb_root / "bot_knowledge_seeds.csv",
        [
            {
                "ID момента": "kb-fixpoint-loop",
                "Сигнал клиента": "Вопрос о курсе",
                "Пример вопроса клиента": "Что дальше?",
                "Безопасный ответ для бота": "циклический ответ должен быть заблокирован",
            }
        ],
    )
    real_sanitize_answer = stage15_gate.sanitize_answer

    def unstable_sanitize(text: object, *, mode: str = "bot") -> SanitizedText:
        if "циклический" in str(text):
            return SanitizedText(
                str(text),
                ("person_name_redacted",),
                "fixpoint_not_reached",
                pass_count=5,
                fixpoint_reached=False,
            )
        return real_sanitize_answer(text, mode=mode)

    monkeypatch.setattr(stage15_gate, "sanitize_answer", unstable_sanitize)

    summary = build_stage15_export_quality_gate(config)

    assert summary["passed"] is False
    assert summary["checks"]["source_bot_safe_answers_have_zero_risks"] is False
    assert summary["bot_export"]["source_risk_counts"]["kb_bot_knowledge_seeds"]["fixpoint_not_reached"] == 1
    assert summary["row_counts"]["blocked_bot_export_rows"] == 1


def test_stage15_gate_fails_on_duplicate_audit_sample_moment_ids(tmp_path: Path) -> None:
    summary = build_stage15_export_quality_gate(_build_fixture(tmp_path, duplicate_audit_id=True))

    assert summary["passed"] is False
    assert summary["checks"]["audit_sample_sufficient_and_unique"] is False
    assert summary["audit_sample"]["duplicate_moment_ids"] == ["dup"]
