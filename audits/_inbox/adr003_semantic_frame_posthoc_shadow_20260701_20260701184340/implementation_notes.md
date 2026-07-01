# ADR-003 SemanticFrame Post-hoc Shadow

Date: 2026-07-01
Branch: `codex/adr003-semanticframe-migration`

## What changed

- Added default-OFF `TELEGRAM_SEMANTIC_FRAME_POSTHOC_SHADOW`.
- Added post-hoc SemanticFrame runner after the final direct-path result and before `frame_decision_shadow`.
- The post-hoc runner writes only metadata: `semantic_frame`, `semantic_frame_shadow`, `semantic_frame_posthoc_shadow`, and mirrored `direct_path.*` keys.
- Provider errors are fail-soft: route/text/safety_flags/manager_checklist are preserved.
- Dynamic simulator now counts `bot_semantic_frame_shadow` calls.
- ADR-003 report now accepts extra model calls only when they are fully explained by expected post-hoc frame-shadow calls; route/text diff remains a stopper.

## Important design correction

The previous same-payload `TELEGRAM_SEMANTIC_FRAME_SHADOW` path was measured on a 2-case smoke and failed route/text no-op. It added `semantic_frame` into the main draft prompt and changed final text. This patch does not promote that path.
