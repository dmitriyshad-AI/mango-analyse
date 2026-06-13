Semantic review for profile child slot merge

Checks:
- Default OFF preserves old behavior for two nameless slots with identical grade and subject: they remain separate.
- ON merges only the narrow class where both slots are nameless and grade+subject match.
- Named+nameless is not merged, even if grade+subject match.
- Different named children are not merged, even if grade+subject match.
- Brand was not used as a hidden criterion because it is absent from the child slot structure.

Residual semantic risk:
- Two real different nameless children in one family with the same grade and subject can merge when the flag is ON. This is the explicit risk accepted by the TZ and why the flag defaults OFF.
- Snapshot-level before/after slot measurements were not run here because ignored large snapshots are absent; Claude will perform that source-data regread.
