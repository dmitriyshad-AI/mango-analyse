# Dry-run

## Статус

`BLOCKED_ENV`, не `object_bug`.

Кодовый dry-run готов, но реальный AMO GET diff не выполнен, потому что в доступном runtime нет активного read-only AMO контекста:

- worktree-local runtime DB: нет таблицы `amo_integration_connections`;
- main runtime DB: таблица есть, но `count(*) = 0`;
- env содержит `CRM_AMO_BASE_URL`, но не содержит `CRM_AMO_API_TOKEN`;
- OAuth refresh запрещён в этом dry-run.

## Blocked summary

```json
{
  "schema_version": "crm_card_amo_writeback_dry_run_blocked_v1",
  "status": "blocked",
  "reason": "AMO read-only dry-run blocked: active OAuth connection is missing.",
  "safety": {
    "dry_run_only": true,
    "write_amo": false,
    "write_tallanto": false,
    "refresh_oauth_token": false,
    "patch_function_available": false
  }
}
```

## Что доказано тестами

- payload строится и для contact, и для deal;
- `active_brand/open_deal_count` обязательны на входе guard;
- brand conflict блокируется;
- `open_deal_count>1` включает D7 `multiple_open_deals` blocker;
- ручная правка между snapshot и PATCH даёт `clobber_protected`;
- dry-run пишет snapshot/journal локально и не имеет PATCH-функции.
