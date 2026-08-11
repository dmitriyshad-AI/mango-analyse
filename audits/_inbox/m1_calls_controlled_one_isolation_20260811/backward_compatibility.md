# Backward compatibility

- default scope остаётся `service`;
- service capture/pipeline/Process A/B/watchdog продолжают использовать общий
  cutover authority и shared lineage marker;
- controlled read-only proof вызывается только allowlist/probe/controlled-one/
  stage-ticket путями;
- legacy broad worker не меняет назначение, но controlled env принудительно
  ограничивает стадии и exact target;
- AMO остаётся default-off и дополнительно запрещена в controlled scope;
- основной Calls regression profile прошёл: `746 passed`.

Customer Timeline worktree, stable_runtime и main не изменялись.
