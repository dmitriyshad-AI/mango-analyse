# Risk Review

Hard safety:

- P0/high-risk preblocks still execute before direct model call.
- Output gate remains the final authority after direct draft and optional semantic verifier.
- Direct-path wide pack does not include forbidden or internal facts and does not bypass number grounding.

Business risks:

- Adjacent facts can help the model answer broader questions, but also create scope ambiguity. The prompt and tests constrain this; M1 transcript review is required before treating it as accepted.
- v6.6 is now the default snapshot in code and bundle metadata. Pilot-readiness still depends on Dmitry/Claude regreeding summer school and new fact behavior.

Operational risks:

- Watcher now accepts `max_hours=9` for the full Claude smoke and defaults task judge to v9.
- `config_invalid` status is reported when runner detects zero direct model calls; this is intentional fail-fast behavior.

Known uncovered item:

- The final smoke set was not present in this local pass, so no M1 task `.ready` files should be created until its path and SHA are inserted.
