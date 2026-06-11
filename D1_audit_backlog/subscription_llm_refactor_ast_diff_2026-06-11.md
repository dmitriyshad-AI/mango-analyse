# subscription_llm AST refresh before baseline freeze

- source commit: `87df0ecd8fd4`
- local names: `766`
- compat imported names: `5`
- export names frozen for facade: `771`

| metric | phase0 plan section 3 | current AST | drift |
|---|---:|---:|---:|
| lines | 13463 | 13522 | +59 |
| ast_top_level_items | 793 | 799 | +6 |
| defs | 474 | 476 | +2 |
| classes | 8 | 8 | +0 |
| assigns | 278 | 282 | +4 |
| uppercase_constants | 273 | 277 | +4 |
| env_constants | 49 | 52 | +3 |
| regex_RE | 82 | 82 | +0 |

Notes:
- phase0 section 3 is kept as historical map;
- future facade checks must use the frozen export snapshot, not current AST;
- compat imported names are only aliases imported from subscription_llm by src/scripts/tests.
