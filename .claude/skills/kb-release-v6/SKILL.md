---
name: kb-release-v6
description: Use for Mango knowledge-base releases and KB fact edits when Codex must use source YAML plus release_manifest and the approved build_kb_release_v6_1_team_answers.py builder, verify quality and semantic gates, compare v6 releases, and avoid deprecated KB builders.
---

# KB Release v6

Use this skill for Mango KB changes and releases.

## Hard Rules

- Edit source YAML and `release_manifest.yaml`; do not patch snapshot JSONL directly.
- Build only with `scripts/build_kb_release_v6_1_team_answers.py`.
- Do not use `scripts/build_kb_release_v3_from_claude_handoff.py` for new releases.
- Preserve brand separation: Foton facts do not answer for UNPK and vice versa.
- `quality_passed=true` is only formal pass; customer-facing facts still need semantic review.

## Release Workflow

1. Read the KB TZ and source YAML.
2. Identify exact facts to add/change/remove and their brand.
3. Update source YAML / manifest only.
4. Run the approved builder:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/build_kb_release_v6_1_team_answers.py
```

5. Check the produced release folder includes:
   - `kb_release_v3_snapshot.json`
   - `client_safe_facts_foton.jsonl`
   - `client_safe_facts_unpk.jsonl`
   - `manager_only_or_internal_facts.jsonl`
   - `quality_report.json`
   - `semantic_review.json`
6. Verify gates: `quality_passed=true`, `semantic_pass=true`, `text_number_grounded`, `field_ranges_ok`, `weekly_frequency_is_plausible`, and `control_numbers_present`.
7. Diff old vs new release: added facts, removed facts, changed client-safe text, changed structured values, changed permissions.
8. Add negative controls: internal facts did not become client-safe accidentally, brand files do not mention the other brand unless approved, prices/percentages/dates match structured values.

## Helper

Use `scripts/check_kb_release.py <release-dir>` for a local gate summary. It does not replace semantic review.
