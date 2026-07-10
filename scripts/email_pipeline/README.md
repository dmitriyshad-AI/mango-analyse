# Email pipeline restore

Read-only rescue of the old stage-2 e-mail pipeline from `_external_handoffs`.

Tracked code lives here because `_external_handoffs` is not durable enough for
pipeline logic. Source archives, extracted mail bodies, SQLite databases and
pilot outputs with personal data stay outside git.

Main entry point:

```bash
PYTHONPATH=src:. python3 scripts/email_pipeline/pilot_100.py
```

Safety rules:

- mail archives and prod customer timeline are opened read-only;
- brand is inferred only from message content, never from folder/from/domain;
- brand conflict or silence returns `brand=none`;
- LLM is used only for manager-only summaries and is capped by
  `--max-llm-calls` (default 100);
- the report masks phones, e-mails, handles and long numeric identifiers.

