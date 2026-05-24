# Backward Compatibility

- `post_filter_registry.json` keeps legacy `phrases` as a global phrase list.
- New consumers should use `global_phrases` plus `phrases_by_active_brand[active_brand]`.
- `gold_answers_v3_payload()` remains available for tests, but reads YAML instead of hardcoding payloads.
- `scripts/build_kb_release_v6_1_team_answers.py` still supports existing CLI args and default paths.
- Builder no longer mutates source YAML; missing `release_manifest.yaml` now fails fast.
