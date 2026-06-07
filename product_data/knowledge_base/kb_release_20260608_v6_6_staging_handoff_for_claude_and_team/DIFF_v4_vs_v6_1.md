# DIFF v4 -> v6.3

## Source of truth

- Business facts are read from `facts/*.yaml`.
- Release metadata and control-number policy are read from `release_manifest.yaml`.
- The Python builder validates source paths and assembles artifacts; it does not patch prices, contacts, transfers, brand rules, bot policy, or gold answers over YAML.

## Not run by builder

- Full MEGA smoke.
- Stage6/Codex smoke.
- Any live write to AMO/CRM/Tallanto.
