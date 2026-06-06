# Backward compatibility

Expected unchanged:
- Main Step 4 flags and number grounding behavior.
- Existing hard P0, complaint, legal, payment dispute handling.
- Existing output gate and KB snapshot paths.
- Live bot wiring and dirty main worktree.

Observed:
- Full `pytest tests/ -q`: 2637 passed, 5 skipped.
- No tracked runtime artifacts changed.

