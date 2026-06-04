---
name: tz-contract-implementation
description: Use when implementing an approved Mango TZ/spec in code, especially bot, KB, verifier, routing, safety, or test changes that require rule #1 source confirmation, scoped edits, negative controls, tests, audit pack, and clean logical commits.
---

# TZ Contract Implementation

Use this skill before touching code for an approved Mango TZ.

## Workflow

1. Read source of truth before editing: `AGENTS.md`, the current TZ, the exact code points named in the TZ, and `git status --short`.
2. Confirm rule #1 explicitly: if a TZ line/function/diagnosis is wrong, stop and report the mismatch. Do not silently implement the nearest plausible fix.
3. Keep the edit scoped. Change only the requested subsystem. Do not touch KB, judge, router, P0, warmth, or runtime artifacts unless the TZ names them.
4. Add positive tests and negative controls. A safety-layer change without negative control is not complete.
5. Run targeted tests, neighbor tests for the touched subsystem, and safe smoke when allowed. If full pytest fails on known infra, record exact failures.
6. Create or update an audit pack for significant bot/KB/customer-facing changes.
7. Commit one logical step at a time.

## Negative Control Checklist

Use the relevant subset:

- Brand: cross-brand answer does not go autonomous and does not leak another brand.
- P0: refund claim, payment dispute, legal threat, complaint stay manager-only.
- Promise: result/admission/score/guarantee claims still block without fact.
- Facts: numbers/dates/discounts not in retrieved facts still block.
- Scope: camp vs regular course, online vs offline, payment method A vs B do not substitute.
- Meta: no debug/internal/client-in-third-person text reaches the client.
- Fallback: no protected safe template is bypassed by a convenience recovery.

## Reporting

Report files/functions changed, positive tests, negative controls, test commands/results, known untested risk, and commit hash.
