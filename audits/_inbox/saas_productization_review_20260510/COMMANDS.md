# Suggested Read-Only Checks

Claude Code may inspect files and run safe read-only or test commands if permitted by its tool policy.

Preferred checks:

```bash
git status --short
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_channels_*.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_productization_*.py tests/test_agent_runtime.py tests/test_amo_writeback_guards.py
rg -n "write|live|send|post|put|delete|commit|execute|requests|httpx" src/mango_mvp/productization src/mango_mvp/channels scripts/mango_office_*.py
```

Do not run:

```bash
scripts/start*
scripts/run-ui*
scripts/*asr*
scripts/*ra*
scripts/write_amo_ready_contacts.py
scripts/write_recent_actionable_deals.py
```

If a command looks risky, do not run it. Report the risk in `findings.csv`.
