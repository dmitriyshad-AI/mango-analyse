# Backward Compatibility

- Direct path is still flag/config gated. `pilot_gold_v1` intentionally enables direct path and gold pack as a named pilot configuration.
- Legacy layered pipeline code is not removed.
- Default KB snapshot paths now point to v6.6 for new runs and bundles.
- Judge default in watcher tasks is v9; runner CLI default remains unchanged unless `--judge-prompt-version v9` is passed.
- The runner fail-fast only activates when direct path is explicitly enabled or `TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1`.
