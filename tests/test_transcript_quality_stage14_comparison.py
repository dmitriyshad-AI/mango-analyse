from __future__ import annotations

import csv
import json
from pathlib import Path

from mango_mvp.quality.stage14_quality_comparison import Stage14ComparisonConfig, build_stage14_quality_comparison


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
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
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _baseline_summary(*, money: int, brand: int, rop_money: int, raw_brand: int = 0, raw_money: int = 0) -> dict[str, object]:
    return {
        "baseline_risks": {
            "kb_no_live_revenue_risk": 0,
            "kb_bot_ready_money_or_terms": money,
            "kb_ideal_answer_brand_risk": brand,
            "kb_bot_safe_answer_brand_risk": 0,
            "kb_bot_safe_answer_personal_data_risk": 0,
            "rop_p0_no_live_or_artifact": 0,
            "rop_revenue_risk_no_live_or_artifact": 0,
            "rop_bot_candidate_money_or_terms": rop_money,
            "rop_bot_safe_answer_brand_risk": 0,
            "rop_bot_safe_answer_personal_data_risk": 0,
        },
        "kb_metrics": {
            "raw_ideal_answer_brand_risk": raw_brand,
            "raw_ideal_answer_money_or_terms": raw_money,
        },
        "rop_metrics": {"rows": 3},
    }


def _kb_row(idx: int, *, flags: str = "", status: str = "ready_for_bot_draft", safe: str | None = None) -> dict[str, object]:
    return {
        "moment_id": f"m-{idx:04d}",
        "source_filename": f"call-{idx}.mp3",
        "started_at": "2026-04-01 10:00:00",
        "manager_name": "Менеджер",
        "signal_ru": "Вопрос о цене",
        "stage_ru": "Обсуждение цены",
        "answer_pattern_ru": "Цена/оплата объяснены",
        "final_outcome_ru": "Есть путь к оплате",
        "overall_quality_score": 80,
        "bot_seed_status": status,
        "bot_seed_status_ru": "Можно брать как черновик для бота" if status == "ready_for_bot_draft" else "Нужна проверка РОПом",
        "bot_safety_status": "safe_with_placeholders" if flags else "safe_no_changes",
        "bot_safety_status_ru": "Безопасно после sanitization" if flags else "Безопасно без замен",
        "sanitizer_flags": flags,
        "brand_risk_flag": "Да" if "brand" in flags else "Нет",
        "money_or_discount_flag": "Да" if "price" in flags or "discount" in flags else "Нет",
        "installment_flag": "Да" if "installment" in flags else "Нет",
        "legal_or_refund_flag": "Да" if "refund" in flags else "Нет",
        "deadline_or_promise_flag": "Да" if "deadline" in flags else "Нет",
        "personal_data_flag": "Да" if "person" in flags or "email" in flags or "phone" in flags else "Нет",
        "customer_question": "Сколько стоит курс?",
        "customer_question_sanitized": "Сколько стоит курс?",
        "manager_answer": "Менеджер объяснил стоимость.",
        "ideal_answer_example": "В НПК МФТИ стоимость 50 000 рублей, скидка 10% до 15 мая.",
        "ideal_answer_manager_sanitized": "В Фотоне актуальную стоимость и варианты менеджер уточнит по правилам.",
        "bot_safe_answer": safe or "Актуальную стоимость и варианты менеджер подтвердит по текущим правилам.",
    }


def _rop_row(idx: int, *, category: str = "Черновик для бота", safe: str | None = None) -> dict[str, object]:
    return {
        "Категория проверки": category,
        "Приоритет": "P1 бот",
        "Сигнал клиента": "Вопрос о цене",
        "Стадия": "Обсуждение цены",
        "Паттерн ответа": "Цена/оплата объяснены",
        "Итог сделки": "Есть путь к оплате",
        "Оценка": 80,
        "Менеджер": "Менеджер",
        "Дата звонка": "2026-04-01 10:00:00",
        "Вопрос клиента": "Сколько стоит курс?",
        "Ответ менеджера": "Менеджер объяснил стоимость.",
        "Идеальный ответ": "В Фотоне актуальную стоимость и варианты менеджер уточнит по правилам.",
        "Идеальный ответ для менеджера": "В Фотоне актуальную стоимость и варианты менеджер уточнит по правилам.",
        "Безопасный ответ для бота": safe or "Актуальную стоимость и варианты менеджер подтвердит по текущим правилам.",
        "Статус sanitizer": "Безопасно после sanitization",
        "Флаги sanitizer": "brand_normalized | price_redacted",
        "Риск бренда": "Да",
        "Риск цены/скидки": "Да",
        "Риск рассрочки": "Нет",
        "Риск договора/возврата": "Нет",
        "Риск срока/обещания": "Нет",
        "Риск персональных данных": "Нет",
        "ID момента": f"m-{idx:04d}",
        "Файл звонка": f"call-{idx}.mp3",
    }


def _build_fixture(root: Path, *, unsafe_after_bot: bool = False) -> Stage14ComparisonConfig:
    before_kb = root / "before_kb"
    after_kb = root / "after_kb"
    before_rop = root / "before_rop"
    after_rop = root / "after_rop"
    before_baseline = root / "before_baseline"
    after_baseline = root / "after_baseline"
    _write_json(before_baseline / "summary.json", _baseline_summary(money=552, brand=13, rop_money=85))
    _write_json(after_baseline / "summary.json", _baseline_summary(money=0, brand=0, rop_money=0, raw_brand=75, raw_money=1568))
    _write_json(before_kb / "summary.json", {"totals": {"reviews": 220}})
    _write_json(after_kb / "summary.json", {"totals": {"reviews": 220}, "sanitizer": {"bot_safe_answer_rows": 220}})
    _write_json(before_rop / "summary.json", {"totals": {"source_reviews": 220}})
    _write_json(after_rop / "summary.json", {"totals": {"source_reviews": 220}})

    safe_text = "Стоимость 50 000 рублей." if unsafe_after_bot else None
    def flags_for_idx(i: int) -> str:
        if i < 30:
            return "brand_normalized"
        if i < 70:
            return "price_redacted | discount_terms_redacted"
        if i < 110:
            return "refund_policy_redacted | deadline_redacted"
        if i < 145:
            return "installment_terms_redacted"
        if i < 185:
            return "person_name_redacted | email_redacted"
        return ""

    after_rows = [
        _kb_row(i, flags=flags_for_idx(i), status=("needs_rop_validation" if flags_for_idx(i) else "ready_for_bot_draft"), safe=safe_text if i == 0 else None)
        for i in range(220)
    ]
    before_rows = [_kb_row(i, safe="Стоимость 50 000 рублей, скидка 10%.") for i in range(220)]
    _write_csv(before_kb / "enriched_reviews.csv", before_rows)
    _write_csv(after_kb / "enriched_reviews.csv", after_rows)
    _write_csv(before_kb / "bot_knowledge_seeds.csv", [{"ID момента": f"m-{i:04d}", "Код сигнала": "price_question", "Пример вопроса клиента": "Сколько стоит курс?", "Черновик идеального ответа": "Стоимость 50 000 рублей."} for i in range(20)])
    _write_csv(after_kb / "bot_knowledge_seeds.csv", [{"ID момента": f"m-{i:04d}", "Код сигнала": "price_question", "Пример вопроса клиента": "Сколько стоит курс?", "Безопасный ответ для бота": safe_text if unsafe_after_bot and i == 0 else "Актуальную стоимость менеджер подтвердит по правилам.", "Флаги sanitizer": "price_redacted"} for i in range(20)])
    _write_csv(before_rop / "rop_validation.csv", [_rop_row(i, safe="Стоимость 50 000 рублей.") for i in range(20)])
    _write_csv(after_rop / "rop_validation.csv", [_rop_row(i, category="Риск потери выручки" if i < 5 else "Черновик для бота", safe=safe_text if unsafe_after_bot and i == 0 else None) for i in range(20)])
    _write_csv(before_rop / "bot_knowledge_drafts.csv", [_rop_row(i, safe="Стоимость 50 000 рублей.") for i in range(20)])
    _write_csv(after_rop / "bot_knowledge_drafts.csv", [_rop_row(i, safe=safe_text if unsafe_after_bot and i == 0 else None) for i in range(20)])
    return Stage14ComparisonConfig(
        project_root=root,
        before_kb_root=before_kb,
        after_kb_root=after_kb,
        before_rop_root=before_rop,
        after_rop_root=after_rop,
        before_baseline_root=before_baseline,
        after_baseline_root=after_baseline,
        out_root=root / "out",
        audit_sample_limit=120,
    )


def test_stage14_comparison_builds_acceptance_package(tmp_path: Path) -> None:
    summary = build_stage14_quality_comparison(_build_fixture(tmp_path))

    assert summary["acceptance"]["passed"] is True
    assert summary["metric_deltas"]["kb_bot_ready_money_or_terms"]["before"] == 552
    assert summary["metric_deltas"]["kb_bot_ready_money_or_terms"]["after"] == 0
    assert summary["metric_deltas"]["rop_bot_candidate_money_or_terms"]["delta"] == -85
    assert summary["schema"]["kb_required_columns_missing"] == []
    assert summary["schema"]["rop_required_columns_missing"] == []
    assert summary["audit_sample"]["rows"] >= 100
    assert (tmp_path / "out" / "STAGE14_QUALITY_COMPARISON_REPORT.md").exists()
    assert (tmp_path / "out" / "audit_sample.csv").exists()
    assert (tmp_path / "out" / "AUDIT_PROMPT_FOR_CLAUDE_OR_GPT.md").exists()


def test_stage14_comparison_fails_acceptance_on_residual_bot_risk(tmp_path: Path) -> None:
    summary = build_stage14_quality_comparison(_build_fixture(tmp_path, unsafe_after_bot=True))

    assert summary["acceptance"]["passed"] is False
    assert summary["acceptance"]["checks"]["no_residual_bot_safe_risks"] is False
    assert summary["residual_risk_samples"]["rows"] > 0
