---
name: audit-pack-generator
description: Use after Mango implementation, bot, KB, CRM, Telegram, verifier, routing, or semantic changes to create an audits/_inbox audit pack with implementation notes, changed files, test output, semantic review, risk review, and backward compatibility notes.
---

# Audit Pack Generator

Create a compact audit pack for meaningful Mango changes.

## When To Create A Pack

Create a pack for customer-facing bot text or routing changes, KB releases, judge/verifier/safety changes, CRM/AMO/Tallanto-facing drafts or payloads, large refactors, and multi-commit TZ blocks. For tiny code-only edits, skip unless the TZ asks for it.

## Required Files

Create `audits/_inbox/<block>_<YYYYMMDD_HHMMSS>/` with:

- `implementation_notes.md`
- `changed_files.txt`
- `test_output.txt`
- `semantic_review.md` for customer/manager/KB/CRM-facing work
- `risk_review.md`
- `backward_compatibility.md`

Use `scripts/create_audit_pack.py` to create the skeleton when useful.

## Content Rules

- Distinguish `formal_pass` from `semantic_pass`.
- Include exact test commands and outcomes.
- Record full pytest failures if they are infrastructure-bound.
- Name negative controls, not only positive tests.
- Mention what was not checked and why.
- Do not copy raw private data into reviewable packs; mask PII when needed.

## Minimal Command

```bash
python3 .claude/skills/audit-pack-generator/scripts/create_audit_pack.py <block-name>
```

Then fill in the placeholders from actual git diff and test output.
