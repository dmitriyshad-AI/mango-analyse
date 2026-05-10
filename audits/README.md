# Audits

This folder is the formal handoff surface between Codex and Claude Code.

1. Codex writes audit packages to `audits/_inbox/<phase_name>/`.
2. Dmitry runs Claude Code from repo root.
3. Command:

   ```bash
   claude /audit audits/_inbox/<phase_name>
   ```

4. Claude Code reads `CLAUDE.md`, `.claude/commands/audit.md`, `docs/THREAT_MODEL.md` and the pack.
5. Claude Code writes results to `audits/_results/<YYYY-MM-DD>_<phase_name>/`.
6. Required result files: `CLAUDE_REAUDIT_RESULT.md`, `findings.csv`, `row_decisions.csv`.
7. Claude Code must not edit files outside `audits/_results/`.
8. Codex must not edit completed Claude result folders.
9. If a new issue class appears, classify it as known class or future threat-model class.
10. Keep audits focused: verify the pack and previous findings, do not expand scope endlessly.

Last audit: none yet.
