# Coverage parity

`subscription_llm_parts` does not exist in wave 1, so strict parts coverage is not applicable yet.

Proxy coverage was collected for current wave 1 files:

Focused pytest:

```text
Name                                                   Stmts   Miss  Cover
--------------------------------------------------------------------------
src/mango_mvp/channels/dialogue_contract_pipeline.py    4310    518    88%
src/mango_mvp/channels/subscription_llm.py              6694    943    86%
--------------------------------------------------------------------------
TOTAL                                                  11004   1461    87%
```

Replay:

```text
Name                                                   Stmts   Miss  Cover
--------------------------------------------------------------------------
src/mango_mvp/channels/dialogue_contract_pipeline.py    4310   3494    19%
src/mango_mvp/channels/subscription_llm.py              6694   3553    47%
--------------------------------------------------------------------------
TOTAL                                                  11004   7047    36%
```

Rule for wave 2+:

- once `subscription_llm_parts` appears, coverage source switches to `src/mango_mvp/channels/subscription_llm_parts`;
- every line covered before a move must be covered after the move in the corresponding parts module;
- any coverage parity drop for moved code fails the wave.
