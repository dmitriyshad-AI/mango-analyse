# ADR-003 Regex Moratorium Guard

Date: 2026-07-01
Branch: `codex/adr003-semanticframe-migration`

## What changed

- Strengthened `tests/test_adr003_regex_understanding_moratorium.py`.
- Added frozen runtime regex snapshot:
  - `tests/fixtures/adr003_runtime_channel_regex_snapshot.json`
- Added frozen direct-path text pattern snapshot:
  - `tests/fixtures/adr003_direct_path_text_patterns_snapshot.json`
- Updated ADR-003 docs to point to the 20260701 runnable M1 bundle and explain the CI guard.

## Guard coverage

- All current `re.compile` calls in `src/mango_mvp/channels`.
- Direct-path `re.compile`, inline `re.search/match/fullmatch/findall/finditer/split/sub/subn`.
- Direct-path uppercase keyword/marker/topic/scope/action tables.
- Direct-path string contains checks like `"..." in text` on text-like expressions.

The guard intentionally freezes existing legacy regex/keyword understanding while SemanticFrame migration proceeds. It does not claim that existing regex is good; it prevents silently adding more.

## Safety

- No live bot process was touched.
- No AMO/Tallanto/CRM writes.
- No runtime behavior changed.
- Changes are docs/tests/fixtures only.
