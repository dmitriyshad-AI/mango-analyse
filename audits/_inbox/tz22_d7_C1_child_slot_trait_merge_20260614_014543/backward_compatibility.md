Backward compatibility

Default behavior:
- PROFILE_CHILD_MERGE_BY_TRAIT is OFF, so nameless slots do not newly merge by grade+subject.
- Existing normalized-name merge remains unchanged.

NEG:
- With the flag unset/OFF, two nameless slots with the same grade and subject still produce two groups.

Changed behavior when enabled:
- With PROFILE_CHILD_MERGE_BY_TRAIT=1, two nameless slots with identical non-empty grades and subjects collapse into one group.

Unaffected:
- Named+nameless slots remain separate.
- Differently named children remain separate.
- Child slot data model still has no brand key.
