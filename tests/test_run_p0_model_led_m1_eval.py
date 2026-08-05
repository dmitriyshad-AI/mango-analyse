from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from mango_mvp.channels.subscription_llm import SubscriptionDraftResult
from scripts import run_p0_model_led_m1_eval as p0_eval


class _Provider:
    def __init__(self, result: SubscriptionDraftResult | None = None) -> None:
        self.calls = 0
        self.result = result or SubscriptionDraftResult(
            metadata={
                "direct_path_model_p0": {
                    "is_p0": True,
                    "is_p0_present": True,
                    "is_p0_valid": True,
                    "p0_kind": "complaint",
                }
            }
        )

    def _direct_path_draft_runner(self, prompt: str) -> SubscriptionDraftResult:
        self.calls += 1
        assert "is_p0" in prompt
        return self.result


def test_evaluate_case_uses_one_model_call_and_does_not_return_input_or_pii() -> None:
    provider = _Provider()
    row = p0_eval.evaluate_case(
        {
            "case_index": 1,
            "text": "Жалоба от parent@example.ru, телефон +7 999 123-45-67.",
            "label": "p0",
            "class": "complaint",
            "source": "paraphrase",
            "recent_messages": ("Ранее оплатила, parent@example.ru, +7 999 123-45-67.",),
        },
        provider=provider,  # type: ignore[arg-type]
    )

    serialized = json.dumps(row, ensure_ascii=False)
    assert provider.calls == 1
    assert len(row["case_id"]) == 30
    assert row["model_is_p0"] is True
    assert row["model_effective_is_p0"] is True
    assert row["model_contract_status"] == "valid"
    assert row["model_signal_route"] == "manager_only"
    assert "model_draft_text" in row
    assert row["model_led_external_calls"] == row["legacy_external_calls"] == 0
    assert "parent@example.ru" not in serialized
    assert "999" not in serialized
    assert "text" not in row


def test_load_cases_validates_set_and_summary_uses_fixed_denominator(tmp_path: Path) -> None:
    path = tmp_path / "set.jsonl"
    path.write_text(
        "\n".join(
            (
                json.dumps({"case_id": "call_refund_001", "text": "Верните оплату.", "label": "p0", "class": "refund", "source": "traffic_hit"}),
                json.dumps({"text": "Когда занятия?", "label": "benign", "class": "none", "source": "traffic_miss"}),
            )
        ),
        encoding="utf-8",
    )
    cases = p0_eval.load_cases(path)
    assert cases[0]["case_id"].startswith("synthetic_")
    assert "refund" not in cases[0]["case_id"]
    assert cases[0]["case_id"] != cases[1]["case_id"]
    assert cases[0]["review_status"] == "single_reviewer"
    rows = [
        {
            **cases[0],
            "model_is_p0": True,
            "model_effective_is_p0": True,
            "regex_is_p0": True,
            "model_field_present": True,
            "model_field_valid": True,
        },
        {
            **cases[1],
            "model_is_p0": False,
            "model_effective_is_p0": False,
            "regex_is_p0": False,
            "model_field_present": True,
            "model_field_valid": True,
        },
    ]

    summary = p0_eval.summarize(rows, denominator=27_507)

    assert summary["source_corpus_denominator"] == 27_507
    assert summary["classification_denominator"] == 2
    assert summary["counters"]["model_tp"] == 1
    assert summary["counters"]["model_tn"] == 1


def test_ambiguous_case_is_report_only(tmp_path: Path) -> None:
    path = tmp_path / "set.jsonl"
    path.write_text(
        json.dumps(
            {
                "text": "Можно ли расторгнуть договор?",
                "p0_label": "ambiguous",
                "class": "refund",
                "source": "traffic_hit",
                "review_status": "needs_context",
                "expected_route": "manual_review",
            }
        ),
        encoding="utf-8",
    )
    case = p0_eval.load_cases(path)[0]
    summary = p0_eval.summarize(
        [
            {
                **case,
                "model_is_p0": False,
                "model_effective_is_p0": False,
                "regex_is_p0": True,
                "model_field_present": False,
                "model_field_valid": False,
            }
        ],
        denominator=27_507,
    )

    assert case["expected_route"] == "manual_review"
    assert summary["counters"] == {"model_field_missing": 1, "report_only_ambiguous": 1}


def test_evaluate_case_does_not_override_physical_false_with_risk_metadata() -> None:
    provider = _Provider(
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            message_type="answer",
            draft_text="Проверка.",
            risk_level="high",
            metadata={
                "direct_path_model_p0": {
                    "is_p0": False,
                    "is_p0_present": True,
                    "is_p0_valid": True,
                    "risk_level": "high",
                    "p0_kind": "refund",
                }
            },
        )
    )

    row = p0_eval.evaluate_case(
        {
            "case_index": 1,
            "text": "Подскажите порядок действий.",
            "label": "p0",
            "class": "refund",
            "source": "paraphrase",
            "brand": "foton",
        },
        provider=provider,  # type: ignore[arg-type]
    )

    assert provider.calls == 1
    assert row["model_is_p0"] is False
    assert row["model_effective_is_p0"] is False
    assert row["model_led_route"] == "bot_answer_self_for_pilot"


def test_evaluate_case_replays_one_model_result_through_both_build_draft_routes() -> None:
    provider = _Provider(
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            message_type="answer",
            draft_text="Рада помочь.",
            risk_level="low",
            metadata={
                "direct_path_model_p0": {
                    "is_p0": True,
                    "is_p0_present": True,
                    "is_p0_valid": True,
                    "risk_level": "low",
                    "p0_kind": "complaint",
                }
            },
        )
    )

    row = p0_eval.evaluate_case(
        {
            "case_index": 1,
            "text": "Подскажите расписание.",
            "label": "p0",
            "class": "complaint",
            "source": "paraphrase",
            "brand": "foton",
        },
        provider=provider,  # type: ignore[arg-type]
    )

    assert provider.calls == 1
    assert row["model_led_route"] == "manager_only"
    assert row["legacy_route"] == "bot_answer_self_for_pilot"
    assert row["model_led_replay_calls"] == row["legacy_replay_calls"] == 1
    assert row["model_led_external_calls"] == row["legacy_external_calls"] == 0


def test_evaluate_case_deterministic_replay_blocks_environment_model_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TELEGRAM_DIRECT_PATH_PILOT_CONFIG", "pilot_gold_v1")
    monkeypatch.setenv("TELEGRAM_SEMANTIC_OUTPUT_VERIFIER", "1")
    row = p0_eval.evaluate_case(
        {
            "case_index": 1,
            "text": "Подскажите расписание.",
            "label": "p0",
            "class": "complaint",
            "source": "paraphrase",
            "brand": "foton",
        },
        provider=_Provider(),  # type: ignore[arg-type]
    )

    assert row["model_led_external_calls"] == row["legacy_external_calls"] == 0
    assert row["model_signal_route"] == "manager_only"


def test_signal_probe_exposes_dead_model_route_even_when_regex_preblock_is_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(p0_eval, "_apply_direct_path_model_p0_route", lambda result, **kwargs: result)
    row = p0_eval.evaluate_case(
        {
            "case_index": 1,
            "text": "Верните деньги за курс.",
            "label": "p0",
            "class": "refund",
            "source": "traffic_hit",
            "brand": "foton",
        },
        provider=_Provider(),  # type: ignore[arg-type]
    )
    summary = p0_eval.summarize([row], denominator=p0_eval.TRAFFIC_CORPUS_DENOMINATOR)

    assert row["model_led_route"] == "manager_only"
    assert row["model_signal_route"] == "bot_answer_self_for_pilot"
    assert summary["counters"]["model_signal_p0_route_miss"] == 1


def test_summary_counts_only_actual_regex_false_positive_traffic_hits() -> None:
    rows = [
        {
            "label": "benign",
            "class": "none",
            "source": "traffic_hit",
            "model_is_p0": False,
            "model_effective_is_p0": False,
            "regex_is_p0": False,
            "model_field_present": True,
            "model_field_valid": True,
        },
        {
            "label": "benign",
            "class": "none",
            "source": "traffic_hit",
            "model_is_p0": False,
            "model_effective_is_p0": False,
            "regex_is_p0": True,
            "model_field_present": True,
            "model_field_valid": True,
        },
    ]

    summary = p0_eval.summarize(rows, denominator=p0_eval.TRAFFIC_CORPUS_DENOMINATOR)

    assert summary["counters"]["regex_false_positive_traffic_hits"] == 1
    assert summary["counters"]["regex_tn"] == 1
    assert summary["counters"]["regex_fp"] == 1


def test_summary_counts_exact_child_safety_kind() -> None:
    summary = p0_eval.summarize(
        [
            {
                "label": "p0",
                "class": "child_safety",
                "p0_classes": ["child_safety"],
                "source": "paraphrase",
                "model_is_p0": True,
                "model_p0_kind": "child_safety",
                "regex_is_p0": False,
                "model_field_present": True,
                "model_field_valid": True,
            }
        ],
        denominator=p0_eval.TRAFFIC_CORPUS_DENOMINATOR,
    )

    assert summary["counters"]["child_safety_total"] == 1
    assert summary["counters"]["child_safety_model_p0"] == 1
    assert summary["counters"]["child_safety_exact_kind"] == 1


def test_validate_case_counts_rejects_missing_child_safety_before_model() -> None:
    cases = (
        [{"label": "p0", "p0_classes": ("child_safety",)}] * 38
        + [{"label": "p0", "p0_classes": ("complaint",)}] * 260
        + [{"label": "benign", "p0_classes": ()}] * 496
        + [{"label": "ambiguous", "p0_classes": ()}] * 21
    )
    with pytest.raises(ValueError, match="39 child_safety"):
        p0_eval._validate_case_counts(cases)


def test_diagnostic_quality_rejects_all_benign_and_all_p0_models() -> None:
    base = {
        "rows": 815,
        "label_counts": p0_eval.EXPECTED_LABEL_COUNTS,
        "classification_denominator": 794,
        "counters": {
            "model_field_missing": 0,
            "model_field_invalid": 0,
            "model_led_replay_one": 815,
            "legacy_replay_one": 815,
            "route_pair_rows": 794,
            "child_safety_total": 39,
            "child_safety_model_p0": 39,
            "child_safety_exact_kind": 39,
        },
    }

    assert p0_eval.diagnostic_quality_passed(
        {**base, "counters": {**base["counters"], "model_fn": 298, "model_fp": 0}}, errors=0
    ) is False
    assert p0_eval.diagnostic_quality_passed(
        {**base, "counters": {**base["counters"], "model_fn": 0, "model_fp": 496}}, errors=0
    ) is False
    assert p0_eval.diagnostic_quality_passed(
        {**base, "counters": {**base["counters"], "model_fn": 0, "model_fp": 10}}, errors=0
    ) is True
    for failed_counter in ("model_signal_p0_route_miss", "model_led_p0_autonomous_route", "replay_external_calls"):
        assert p0_eval.diagnostic_quality_passed(
            {**base, "counters": {**base["counters"], failed_counter: 1}}, errors=0
        ) is False
    assert p0_eval.diagnostic_quality_passed(
        {**base, "counters": {**base["counters"], "replay_call_invalid": 1}}, errors=0
    ) is False
    assert p0_eval.diagnostic_quality_passed(
        {**base, "counters": {**base["counters"], "child_safety_exact_kind": 38}}, errors=0
    ) is False
    assert p0_eval.diagnostic_quality_passed(
        {**base, "counters": {**base["counters"], "model_led_replay_one": 814}}, errors=0
    ) is False
    assert p0_eval.diagnostic_quality_passed(
        {**base, "counters": {**base["counters"], "model_led_replay_one": 0, "model_led_replay_preblocked": 815}}, errors=0
    ) is False
    assert p0_eval.diagnostic_quality_passed(
        {**base, "counters": {**base["counters"], "model_p0_kind_missing_p0": 1}}, errors=0
    ) is False


def test_parse_args_rejects_decorative_traffic_denominator(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        p0_eval.parse_args(
            [
                "--set",
                str(tmp_path / "set.jsonl"),
                "--out-dir",
                str(tmp_path),
                "--expected-code-commit",
                "deadbeef",
                "--codex-bin",
                "/bin/echo",
                "--codex-home",
                str(tmp_path),
                "--expected-codex-version",
                "echo",
                "--traffic-denominator",
                "1",
            ]
        )


def test_main_rejects_replaced_set_before_loading_or_llm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    replaced = tmp_path / "same-counts-different-content.jsonl"
    replaced.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(p0_eval, "load_cases", lambda path: (_ for _ in ()).throw(AssertionError("must not load")))

    with pytest.raises(ValueError, match="unexpected M1 set sha256"):
        p0_eval.main(
            [
                "--set",
                str(replaced),
                "--out-dir",
                str(tmp_path / "out"),
                "--expected-code-commit",
                "deadbeef",
                "--codex-bin",
                "/bin/echo",
                "--codex-home",
                str(tmp_path),
                "--expected-codex-version",
                "echo",
                "--validate-only",
            ]
        )


def test_main_rejects_unexpected_code_commit_before_loading_cases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "set.jsonl"
    path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(p0_eval, "EXPECTED_SET_SHA256", p0_eval._sha256(path))
    monkeypatch.setattr(p0_eval, "_git_head", lambda: "actual")
    monkeypatch.setattr(p0_eval, "load_cases", lambda value: (_ for _ in ()).throw(AssertionError("must not load")))

    with pytest.raises(ValueError, match="unexpected code commit: actual"):
        p0_eval.main(
            [
                "--set",
                str(path),
                "--out-dir",
                str(tmp_path / "out"),
                "--expected-code-commit",
                "expected",
                "--codex-bin",
                "/bin/echo",
                "--codex-home",
                str(tmp_path),
                "--expected-codex-version",
                "echo",
                "--validate-only",
            ]
        )


def test_git_head_rejects_dirty_worktree(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(args: tuple[str, ...], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert kwargs["cwd"] == p0_eval.REPO_ROOT
        assert not any(name.startswith("GIT_") for name in kwargs["env"])
        stdout = str(p0_eval.REPO_ROOT) if args[1:] == ("rev-parse", "--show-toplevel") else " M src/example.py\n"
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(p0_eval.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="clean git worktree"):
        p0_eval._git_head()


def test_git_head_ignores_foreign_git_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GIT_DIR", "/tmp/foreign.git")
    monkeypatch.setenv("GIT_WORK_TREE", "/tmp/foreign")
    outputs = iter((str(p0_eval.REPO_ROOT), "", "deadbeef"))

    def fake_run(args: tuple[str, ...], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert not any(name.startswith("GIT_") for name in kwargs["env"])
        return subprocess.CompletedProcess(args, 0, stdout=next(outputs), stderr="")

    monkeypatch.setattr(p0_eval.subprocess, "run", fake_run)
    assert p0_eval._git_head() == "deadbeef"


def test_evaluator_clears_telegram_runtime_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_DIRECT_PATH_PILOT_CONFIG", "pilot_gold_v1")
    monkeypatch.setenv("TELEGRAM_SEMANTIC_OUTPUT_VERIFIER", "1")
    monkeypatch.setenv("UNRELATED_SETTING", "kept")

    p0_eval._clear_telegram_runtime_env()

    assert "TELEGRAM_DIRECT_PATH_PILOT_CONFIG" not in p0_eval.os.environ
    assert "TELEGRAM_SEMANTIC_OUTPUT_VERIFIER" not in p0_eval.os.environ
    assert p0_eval.os.environ["UNRELATED_SETTING"] == "kept"


def test_model_environment_does_not_inherit_home_path_endpoint_or_passthrough(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = tmp_path / "codex"
    binary.write_text("binary", encoding="utf-8")
    codex_home = tmp_path / "profile" / ".codex"
    codex_home.mkdir(parents=True)
    monkeypatch.setenv("TASK_CONTAINER_ENV_PASSTHROUGH", "OPENAI_BASE_URL,FOREIGN_PROFILE")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://foreign.invalid")
    monkeypatch.setenv("FOREIGN_PROFILE", "foreign")
    monkeypatch.setenv("HOME", "/tmp/foreign-home")
    monkeypatch.setenv("PATH", "/tmp/foreign-bin")
    monkeypatch.setattr(
        p0_eval.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, stdout="codex-cli 1.2.3\n", stderr=""),
    )

    resolved_bin, env, version = p0_eval._codex_runtime(binary, codex_home, "codex-cli 1.2.3")

    assert "TASK_CONTAINER_ENV_PASSTHROUGH" not in env
    assert "OPENAI_BASE_URL" not in env
    assert "FOREIGN_PROFILE" not in env
    assert resolved_bin == str(binary.resolve())
    assert env["HOME"] == str(codex_home.parent.resolve())
    assert env["CODEX_HOME"] == str(codex_home.resolve())
    assert env["PATH"] == p0_eval.EVALUATION_SYSTEM_PATH
    assert version == "codex-cli 1.2.3"
    with pytest.raises(ValueError, match="unexpected codex version"):
        p0_eval._codex_runtime(binary, codex_home, "codex-cli 9.9.9")
    with pytest.raises(ValueError, match="absolute paths"):
        p0_eval._codex_runtime(Path("codex"), codex_home, "codex-cli 1.2.3")
    with pytest.raises(ValueError, match="absolute paths"):
        p0_eval._codex_runtime(binary, Path("~/.codex"), "codex-cli 1.2.3")


def test_load_cases_stops_before_llm_when_input_contains_person_name(tmp_path: Path) -> None:
    path = tmp_path / "set.jsonl"
    path.write_text(
        json.dumps({"text": "Менеджер Анна Иванова обещала перезвонить.", "p0_label": "benign"}),
        encoding="utf-8",
    )

    try:
        p0_eval.load_cases(path)
    except ValueError as exc:
        assert "PII signals are forbidden" in str(exc)
        assert "person_name" in str(exc)
    else:
        raise AssertionError("PII input must be rejected before any model call")


def test_load_cases_rejects_single_colloquial_names(tmp_path: Path) -> None:
    path = tmp_path / "set.jsonl"
    path.write_text(
        json.dumps({"text": "Лёш, подскажите, когда Юлия перезвонит?", "p0_label": "benign"}),
        encoding="utf-8",
    )

    try:
        p0_eval.load_cases(path)
    except ValueError as exc:
        assert "person_name" in str(exc)
    else:
        raise AssertionError("colloquial names must be rejected before any model call")


@pytest.mark.parametrize(
    "raw_case_id",
    (
        "79991234567",
        "call_79991234567",
        "call_Anna_Ivanova",
        "12345678901234567890",
    ),
)
def test_load_cases_hashes_pii_like_case_ids_before_report(tmp_path: Path, raw_case_id: str) -> None:
    path = tmp_path / "set.jsonl"
    path.write_text(
        json.dumps({"case_id": raw_case_id, "text": "Когда занятия?", "label": "benign"}), encoding="utf-8"
    )

    case = p0_eval.load_cases(path)[0]

    assert case["case_id"].startswith("synthetic_")
    assert raw_case_id not in case["case_id"]


@pytest.mark.parametrize(
    "row",
    (
        {"case_id": "call_safe_001", "text": "Когда занятия?", "label": "benign", "class": "Anna_Ivanova"},
        {"case_id": "call_safe_001", "text": "Когда занятия?", "label": "benign", "source": "79991234567"},
    ),
)
def test_load_cases_rejects_pii_like_or_unknown_report_metadata(tmp_path: Path, row: dict[str, object]) -> None:
    path = tmp_path / "set.jsonl"
    path.write_text(json.dumps(row), encoding="utf-8")

    with pytest.raises(ValueError, match="opaque safe"):
        p0_eval.load_cases(path)


def test_summary_rejects_more_than_one_deterministic_replay_call() -> None:
    row = {
        "label": "benign",
        "class": "none",
        "source": "paraphrase",
        "model_field_present": True,
        "model_field_valid": True,
        "model_effective_is_p0": False,
        "regex_is_p0": False,
        "model_led_route": "bot_answer_self_for_pilot",
        "legacy_route": "bot_answer_self_for_pilot",
        "model_led_replay_calls": 2,
        "legacy_replay_calls": 2,
    }

    summary = p0_eval.summarize([row], denominator=p0_eval.TRAFFIC_CORPUS_DENOMINATOR)

    assert summary["counters"]["replay_call_invalid"] == 2


def test_diagnostic_quality_rejects_completely_preblocked_replay() -> None:
    summary = {
        "label_counts": p0_eval.EXPECTED_LABEL_COUNTS,
        "classification_denominator": 794,
        "counters": {
            "model_led_replay_preblocked": 815,
            "legacy_replay_preblocked": 815,
            "route_pair_rows": 794,
        },
    }

    assert p0_eval.diagnostic_quality_passed(summary, errors=0) is False


def test_load_cases_rejects_pii_in_report_metadata(tmp_path: Path) -> None:
    path = tmp_path / "set.jsonl"
    path.write_text(
        json.dumps({"text": "Когда занятия?", "p0_label": "benign", "class": "Анна Иванова"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="class must be an opaque safe token"):
        p0_eval.load_cases(path)
