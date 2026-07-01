from __future__ import annotations

from pathlib import Path


CHANNEL_REGEX_BUDGET = {
    "src/mango_mvp/channels/answer_quality_rewriter.py": 4,
    "src/mango_mvp/channels/contracts.py": 1,
    "src/mango_mvp/channels/dialogue_contract_pipeline.py": 27,
    "src/mango_mvp/channels/dialogue_memory.py": 27,
    "src/mango_mvp/channels/fact_claim_audit.py": 1,
    "src/mango_mvp/channels/few_shot_reference.py": 1,
    "src/mango_mvp/channels/humanity_guards.py": 1,
    "src/mango_mvp/channels/humanity_linter.py": 2,
    "src/mango_mvp/channels/manager_handoff_summary.py": 1,
    "src/mango_mvp/channels/p0_recall_spec.py": 11,
    "src/mango_mvp/channels/rules_engine.py": 7,
    "src/mango_mvp/channels/subscription_llm_parts/contracts.py": 11,
    "src/mango_mvp/channels/subscription_llm_parts/direct_path.py": 10,
    "src/mango_mvp/channels/subscription_llm_parts/policy_routing.py": 16,
    "src/mango_mvp/channels/subscription_llm_parts/post_layers.py": 73,
    "src/mango_mvp/channels/subscription_llm_parts/provider.py": 3,
    "src/mango_mvp/channels/subscription_llm_parts/reliable_answerer.py": 11,
    "src/mango_mvp/channels/subscription_llm_parts/support.py": 3,
    "src/mango_mvp/channels/subscription_llm_parts/text_hygiene.py": 7,
    "src/mango_mvp/channels/telegram_pilot_reporting.py": 7,
}


def test_adr003_no_new_runtime_channel_regex_without_review() -> None:
    repo = Path(__file__).resolve().parents[1]
    channel_files = sorted((repo / "src/mango_mvp/channels").rglob("*.py"))
    files_with_regex = {
        str(path.relative_to(repo)): path.read_text(encoding="utf-8").count("re.compile(")
        for path in channel_files
        if "re.compile(" in path.read_text(encoding="utf-8")
    }

    unexpected = sorted(set(files_with_regex) - set(CHANNEL_REGEX_BUDGET))
    assert unexpected == [], (
        "ADR-003 forbids adding new runtime channel regex/keyword understanding. "
        "If this is an output scrub or infrastructure parser, document it in "
        "docs/ADR003_REGEX_UNDERSTANDING_MORATORIUM.md and update this allowlist."
    )
    for rel_path, count in files_with_regex.items():
        assert count <= CHANNEL_REGEX_BUDGET[rel_path], (
            f"{rel_path} added {count - CHANNEL_REGEX_BUDGET[rel_path]} runtime regex. "
            "New client-meaning detection must go through SemanticFrame/eval, not regex."
        )
