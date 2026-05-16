import csv
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "build_rop_blocker_markup_pack.py"

spec = importlib.util.spec_from_file_location("build_rop_blocker_markup_pack", SCRIPT_PATH)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def _fixture_catalog(tmp_path: Path) -> Path:
    root = tmp_path / "question_catalog"
    _write_csv(
        root / "customer_question_classes.csv",
        [
            {
                "question_class_id": "class:payment",
                "parent_question_class": "оплата / возврат / чек",
                "question_subclass": "ручной разбор: общий вопрос по оплате",
                "canonical_question": "оплата / возврат / чек / ручной разбор: общий вопрос по оплате",
                "count_total": "3",
                "count_calls": "1",
                "count_telegram": "1",
                "count_email": "1",
            },
            {
                "question_class_id": "class:discount",
                "parent_question_class": "скидки",
                "question_subclass": "общий вопрос о скидке",
                "canonical_question": "скидки / общий вопрос о скидке",
                "count_total": "2",
                "count_calls": "0",
                "count_telegram": "2",
                "count_email": "0",
            },
        ],
    )
    _write_csv(
        root / "rop_review_priority_top100.csv",
        [
            {
                "Место": "6",
                "ID класса": "class:payment",
                "Класс вопроса": "оплата / возврат / чек / ручной разбор: общий вопрос по оплате",
                "Можно ли утверждать сейчас": "нет",
                "Причина блокировки утверждения": "thematic_fallback_needs_split",
            },
            {
                "Место": "7",
                "ID класса": "class:discount",
                "Класс вопроса": "скидки / общий вопрос о скидке",
                "Можно ли утверждать сейчас": "да, проверить черновик",
                "Причина блокировки утверждения": "",
            },
        ],
    )
    (root / "answer_quality_check_report.json").write_text(
        json.dumps(
            {
                "findings": [
                    {
                        "severity": "p1",
                        "code": "thematic_fallback_needs_split",
                        "question_class_id": "class:payment",
                        "canonical_question": "оплата / возврат / чек / ручной разбор: общий вопрос по оплате",
                    },
                    {
                        "severity": "p1",
                        "code": "thematic_fallback_needs_split",
                        "question_class_id": "class:discount",
                        "canonical_question": "скидки / общий вопрос о скидке",
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        root / "customer_question_items.jsonl",
        [
            {
                "question_class_id": "class:payment",
                "question_item_id": "item:1",
                "source_channel": "telegram",
                "occurred_at": "2026-01-01T00:00:00+00:00",
                "customer_text_redacted": "Можно оплатить частями и получить договор?",
                "manager_text_redacted": "Уточню условия.",
                "metadata": {"customer_text_for_rop": "Можно оплатить частями и получить договор?"},
            },
            {
                "question_class_id": "class:payment",
                "question_item_id": "item:2",
                "source_channel": "email",
                "occurred_at": "2026-01-02T00:00:00+00:00",
                "customer_text_redacted": "Нужен чек после оплаты.",
                "manager_text_redacted": "Отправим актуальную стоимость и актуальное окно записи.",
                "metadata": {"customer_text_for_rop": "Нужен чек после оплаты."},
            },
            {
                "question_class_id": "class:discount",
                "question_item_id": "item:3",
                "source_channel": "telegram",
                "occurred_at": "2026-01-03T00:00:00+00:00",
                "customer_text_redacted": "Есть скидка?",
                "manager_text_redacted": "Проверим.",
                "metadata": {"customer_text_for_rop": "Есть скидка?"},
            },
        ],
    )
    return root


def test_build_pack_uses_only_top100_blockers_by_default(tmp_path: Path) -> None:
    catalog_root = _fixture_catalog(tmp_path)
    output_csv = tmp_path / "pack.csv"
    output_summary = tmp_path / "summary.json"
    output_guide = tmp_path / "guide.md"

    result = module.build_pack(
        catalog_root,
        output_csv,
        output_summary,
        output_guide,
        limit_per_class=10,
    )

    assert result["blocked_classes"] == 1
    rows = list(csv.DictReader(output_csv.open(encoding="utf-8-sig")))
    example_rows = [row for row in rows if row["Источник примера"] != "ИТОГО ПО КЛАССУ"]
    assert len(example_rows) == 2
    assert all(row["ID класса"] == "class:payment" for row in example_rows)
    assert rows[0]["Решение РОПа"] == ""
    assert "Новый узкий класс 2, если в вопросе несколько тем" in rows[0]
    assert "Идеальный ответ или шаблон РОПа" in rows[0]
    assert "Исторический ответ менеджера (не утверждено)" in rows[0]
    assert "[актуальная стоимость]" in rows[1]["Исторический ответ менеджера (не утверждено)"]
    assert "[[" not in rows[1]["Исторический ответ менеджера (не утверждено)"]
    guide = output_guide.read_text(encoding="utf-8")
    assert "должен подтвердить РОП" in guide
    assert "[CLIENT_NAME]" in guide

    summary = json.loads(output_summary.read_text(encoding="utf-8"))
    assert summary["totals"]["blocked_classes"] == 1
    assert summary["totals"]["example_rows"] == 2


def test_include_all_quality_blockers_adds_non_top100_p1_classes(tmp_path: Path) -> None:
    catalog_root = _fixture_catalog(tmp_path)
    output_csv = tmp_path / "pack.csv"
    output_summary = tmp_path / "summary.json"

    module.build_pack(
        catalog_root,
        output_csv,
        output_summary,
        tmp_path / "guide.md",
        limit_per_class=1,
        include_all_quality_blockers=True,
    )

    summary = json.loads(output_summary.read_text(encoding="utf-8"))
    assert summary["totals"]["blocked_classes"] == 2
    assert summary["totals"]["example_rows"] == 2


def test_suggest_bucket_detects_multi_topic_and_specific_payment_cases() -> None:
    bucket, reason = module.suggest_bucket("Можно оплатить частями и получить договор?")
    assert bucket == "много тем в одном вопросе"
    assert "оплата" in reason
    assert "договор" in reason

    bucket, reason = module.suggest_bucket("Нужен чек после оплаты")
    assert bucket == "оплата: чек или квитанция"
    assert reason


def test_suggest_bucket_does_not_use_canonical_title_as_customer_evidence() -> None:
    bucket, reason = module.suggest_bucket(
        "Сканы документов отправлять прямо сюда?",
        "документы / договор",
    )

    assert bucket != "много тем в одном вопросе"
    assert bucket == "документы: общий вопрос, требуется уточнение"
    assert "договор" not in reason


def test_suggest_bucket_handles_user_feedback_examples() -> None:
    assert module.suggest_bucket("Пришлите, пожалуйста, квитанцию для оплаты 2 семестра.")[0] == (
        "оплата: отправить квитанцию для оплаты"
    )
    assert module.suggest_bucket("Нет, мне нужно забрать договор для оплаты мат капиталом.")[0] == (
        "материнский капитал: договор для оплаты обучения"
    )
    assert module.suggest_bucket("У вас не появилась оплата региональным мат капиталом?")[0] == (
        "материнский капитал: статус оплаты региональным маткапиталом"
    )
    assert module.suggest_bucket("Не совсем понимаю,что нужно написать в назначении платежа?")[0] == (
        "оплата: назначение платежа"
    )


def test_noise_examples_are_filtered_before_markup(tmp_path: Path) -> None:
    catalog_root = _fixture_catalog(tmp_path)
    with (catalog_root / "customer_question_items.jsonl").open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(
                {
                    "question_class_id": "class:payment",
                    "question_item_id": "item:noise",
                    "source_channel": "email",
                    "occurred_at": "2026-01-04T00:00:00+00:00",
                    "customer_text_redacted": "Промокод на следующую покупку. Электронная копия чека.",
                    "manager_text_redacted": "",
                    "metadata": {"customer_text_for_rop": "Промокод на следующую покупку. Электронная копия чека."},
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    output_csv = tmp_path / "pack.csv"
    module.build_pack(
        catalog_root,
        output_csv,
        tmp_path / "summary.json",
        tmp_path / "guide.md",
        limit_per_class=10,
    )

    rows = list(csv.DictReader(output_csv.open(encoding="utf-8-sig")))
    assert all("Промокод на следующую покупку" not in row["Пример вопроса клиента"] for row in rows)
