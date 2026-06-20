# TZ139 Work A Real-Data Report

## Summary

Read-only проверка подтвердила два реальных дефекта старого Work A и они исправлены до отчета:

- `8340` строк `no_exact_phone_match` в старой live SQLite были `strong`; после текущей логики все `8340` стали `partial`, `strong=0`.
- `50` семейных телефонов с `3+` Tallanto students были в одной строке и не попадали в family split; после текущей логики они дают `50` конфликтных групп и `161` split customer entries.

## До / После

| Метрика | До: текущая live SQLite read-only | После: Work A in-memory |
| --- | ---: | ---: |
| `customer_identities` / customer entries | `16239` | `16901` |
| `identity_status=strong` | `16239` | `7298` |
| `identity_status=partial` | `0` | `8340` |
| `identity_status=ambiguous` | `0` | `1263` |
| `timeline_conflicts` | `0` | `601` family-phone conflict groups |
| `customer_id_mappings` table | нет | будет создан при apply |
| `no_exact_phone_match` as `strong` | `8340` | `0` |
| `no_exact_phone_match` as `partial` | `0` | `8340` |

## A. Shared AMO Contact / Lead

- Duplicate AMO contact IDs: `367`.
- Duplicate AMO lead IDs: `207`.
- Phones with `shared_amo_contact_across_customers`: `694`.
- Phones with `shared_amo_lead_across_customers`: `265`.
- Work A policy: these IDs are not used as customer key and get `ambiguous` link policy, not merge policy.
- Selected manual reasons after current logic:
  - `shared_amo_contact_across_customers`: `785` customer entries.
  - `shared_amo_lead_across_customers`: `316` customer entries.

## B. Families 3+ Students

- Source phone groups with `3+` Tallanto students: `50`.
- Max Tallanto students on one phone: `6`.
- Work A conflict groups total for `2+` Tallanto students on one phone: `601`.
- Work A conflict groups for `3+`: `50`.
- Split customer entries for the `3+` subset: `161`.
- Phone links for shared family phone are now `ambiguous`, not `strong_unique`.

## C. Foreign / Malformed Phones

- Foreign phones by normalized value not starting with `+7`: `25` rows / `25` unique.
- Malformed `+7` with total normalized string length `13` (`+` + 12 digits): `1` row / `1` unique.
- Invalid phones where normalizer returns `None`: `0`.
- No raw phone values included in this report.

## D. `no_exact_phone_match`

- Source rows: `8340`.
- Unique phones in those rows: `8340`.
- Before live SQLite: `8340` had `identity_status=strong`.
- After current Work A logic:
  - `no_exact_source_row_after_status_counts`: `partial=8340`.
  - `no_exact_customer_status_counts`: `partial=8340`.
  - `no_exact_strong_example_phone_hashes`: none.
  - `tallanto_match_class("no_exact_phone_match")`: `unmatched`.
  - `tallanto_match_class("exact_phone_single")`: `strong_unique`.

## Stop Condition

Work B не начат. Следующий шаг: Claude regreade по этому audit pack и сырью. Read-time resolver old -> new `customer_id` оставлен в Work D, как указал Claude.
