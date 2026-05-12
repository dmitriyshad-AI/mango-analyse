# Audit Command

Проведи контрольный аудит пакета по пути `$ARGUMENTS`.

## Inputs

Read:

- `CLAUDE.md`
- `docs/AI_WORKFLOW.md`
- `docs/THREAT_MODEL.md`
- current formal release artifacts referenced by the audit package, especially Stage15 and frozen-corpus summaries
- audit package at `$ARGUMENTS`
- previous results in `audits/_results/`, if relevant to the package

## Scope

Цель зависит от типа пакета:

- для bot/allowlist пакета: проверить, что финальный `bot_export_allowlist.csv` не нарушает классы из `docs/THREAT_MODEL.md`, и что предыдущие findings устранены;
- для CRM/AMO/writeback пакета: проверить, что transcript-derived история общения, AMO-ready строки и dry-run/offline-preview payload безопасны, не используют старый export pointer, не содержат очевидно несодержательные/wrong-number/no-dialogue кейсы и не пишут protected fields;
- для ROP/internal пакета: проверить полноту, отсутствие искусственного обрезания и корректное отделение internal-only данных от bot-safe/CRM-safe данных.

Не расширяй scope на бесконечный поиск новых классов. Если находишь новую проблему, классифицируй ее:

- `known_class` - известный класс из threat model;
- `future_threat_model_class` - новый класс для будущего расширения threat model.

## Hard Rules

Do not edit anything outside `audits/_results/`.

Do not:

- write to CRM/AMO/Tallanto;
- run ASR;
- run R+A;
- delete files;
- mutate `stable_runtime/`;
- change implementation files;
- change tests;
- change docs outside the result folder.

If you need to suggest a fix, write it as a finding. Do not implement it.

## Required Output

Create a new result folder:

```text
audits/_results/<today>_<phase>/
```

Use a short filesystem-safe `<phase>` derived from the audit package name.

Write:

- `CLAUDE_REAUDIT_RESULT.md`
- `findings.csv`
- `row_decisions.csv`

## Required Verdict

Final verdict must answer the relevant question for the package:

```text
Can this package safely proceed to the next staged step?
```

Use one of:

- `PASS`
- `PASS_WITH_LIMITATIONS`
- `FAIL`

## Findings CSV Schema

Use this header:

```csv
finding_id,severity,class,status,row_id,column,evidence,recommendation
```

Severity values:

- `P0`
- `P1`
- `P2`
- `P3`
- `INFO`

Status values:

- `open`
- `fixed`
- `not_reproducible`
- `accepted_risk`
- `future_threat_model_class`

## Row Decisions CSV Schema

Use this header:

```csv
row_id,decision,reason,known_class,future_class_note
```

Decision values:

- `allow`
- `block`
- `needs_review`

For CRM/AMO packages, `allow` means “safe for the next staged dry-run/live pilot under existing writeback guards”, not permission for unrestricted full live writeback.

## Result Markdown Structure

`CLAUDE_REAUDIT_RESULT.md` must contain:

1. Audit package path.
2. Date.
3. Verdict.
4. What was checked.
5. Summary counts by severity.
6. Known-class findings.
7. Future threat-model class candidates.
8. Previous findings status.
9. Commands/files inspected.
10. Final recommendation.
