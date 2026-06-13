Risk review

Primary risks addressed:
- A single nameless child can accumulate many duplicate slots when no name is present.
- False merges of named children remain blocked.

Guardrails:
- PROFILE_CHILD_MERGE_BY_TRAIT defaults OFF.
- Unit NEG proves OFF keeps old nameless-slot grouping.
- ON tests cover merge and non-merge boundaries.
- No production profile database was modified.

Known gaps:
- Real distribution on tz16_profiles_v7 was not measured in this worktree.
- The logic does not normalize grade aliases such as "7" vs "7 класс"; that is outside this TZ and avoids broadening the merge criterion.
