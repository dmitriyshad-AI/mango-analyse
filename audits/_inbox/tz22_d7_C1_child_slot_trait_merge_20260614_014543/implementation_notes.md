TZ22 D7 block C1: nameless child slot merge

Scope:
- Added PROFILE_CHILD_MERGE_BY_TRAIT with default OFF.
- Existing named-child merge by normalized name is unchanged.
- When the flag is ON, two child slots merge only if both have no normalized name and their grade sets and subject sets are non-empty and equal.
- Brand is intentionally not part of the criterion because child slots do not store brands.

Implementation:
- Added os import in customer_profile/builder.py.
- Extended child_slots_match() with the flag-gated nameless grade+subject branch.
- Kept named+nameless slots separate.
- Kept two differently named slots separate even when grade and subject match.
- Marker reason is now grade_subject_match for nameless trait merges instead of falsely reporting a name match.

Out of scope:
- No rebuild of ignored tz16_profiles_v7 snapshot.
- No profile runtime writes outside tmp test databases.
