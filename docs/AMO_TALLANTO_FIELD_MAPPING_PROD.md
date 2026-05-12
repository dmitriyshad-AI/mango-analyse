# AMO/Tallanto Field Mapping Production Policy

Дата: 2026-05-09

Назначение: зафиксировать, какие данные продуктовый слой читает из amoCRM и
Tallanto, какие поля можно записывать обратно, какие поля защищены от записи и
какой порядок безопасного rollout использовать.

## Operating mode

Проект остается в режиме внутреннего продукта с live credentials. Это допустимо,
но live writeback должен быть явным:

```zsh
--execute-live-write --live-confirmation WRITE_AMO_LIVE
```

HTTP writeback также требует:

```json
{
  "execute_live_write": true,
  "live_confirmation": "WRITE_AMO_LIVE"
}
```

Без этих признаков CLI делает dry-run/preview, а HTTP writeback возвращает отказ.

## amoCRM contacts

### Read

Контакты используются для:

- поиска контакта по телефону;
- проверки единственного точного match;
- чтения текущего каталога custom fields;
- проверки готовности интеграции.

### Allowed writes

Текущий guarded contact writeback пишет только AI/context fields:

| Field | Source | Owner | Write policy |
|---|---|---|---|
| `Статус матчинга` | Tallanto/phone matching result | AI/product layer | Allowed if live confirmed |
| `AI-приоритет` | sales insight/contact scoring | AI/product layer | Allowed if live confirmed |
| `AI-рекомендованный следующий шаг` | call insight/history analysis | AI/product layer | Allowed if live confirmed |
| `Последняя AI-сводка` | latest fresh call summary | AI/product layer | Allowed if live confirmed |
| `Авто история общения` | compact contact history + Tallanto context | AI/product layer | Allowed if live confirmed |

### Protected fields

Эти поля можно читать и показывать в отчетах, но нельзя перезаписывать через
текущий contact writeback:

| Field | Why protected |
|---|---|
| `Id Tallanto` | Идентификатор внешней системы, должен приходить из Tallanto/matching owner |
| `Филиал Tallanto` | Операционное поле филиала, не AI-generated |

Кодовая защита: `CONTACT_WRITE_PROTECTED_FIELDS` в
`src/mango_mvp/amocrm_runtime/amo_integration.py`.

## amoCRM leads/deals

### Read

Deal layer читает:

- pipeline/status catalog;
- recent closed leads;
- lead details and current custom field values;
- related contacts;
- notes/tasks for dossier context.

### Allowed writes

Lead writeback использует logical field map. Если `CRM_AMO_LEAD_FIELD_MAP` не
задан, применяются default field names:

| Logical key | Default amoCRM field | Source | Write policy |
|---|---|---|---|
| `close_verdict` | `AI-вердикт по закрытию` | deal analysis | Closed actionable deals only |
| `premature_close_risk` | `AI-risk: premature close` | deal analysis | Closed actionable deals only |
| `close_reason_summary` | `AI-основание вердикта` | deal analysis evidence | Closed actionable deals only |
| `recommended_next_step` | `AI-рекомендованный следующий шаг` | deal/contact context | Open deals and closed actionable deals |
| `follow_up_due_at` | `AI-дата следующего касания` | deal/contact context | Open deals and closed actionable deals |
| `deal_summary` | `AI-сводка по сделке` | latest call/history/objections | Open deals and closed actionable deals |

Опциональное служебное поле `AI office` выключено по умолчанию и включается
только через `CRM_ANALYSIS_WRITE_AI_OFFICE_FIELD=true`.

### Safe mode

`CRM_AMO_DEAL_WRITEBACK_SAFE_MODE=true` по умолчанию. В этом режиме code path
не перезаписывает непустые значения в сделке:

- пустое поле можно заполнить;
- такое же значение считается unchanged;
- другое непустое значение пропускается как protected by safe mode.

## Tallanto

### Read

Tallanto используется как read/context provider:

| Module / area | Purpose |
|---|---|
| `Contact` | ФИО, телефоны, филиал, assigned user, email, amo id, тип клиента |
| `Opportunity` | История возможностей/сделок в Tallanto |
| `Request` | Заявки и статусы обращений |
| `most_finances` | Платежный контекст |
| `most_sip_log` | Контекст звонков, если нужен для сверки |
| `most_courses` | Курсы и продуктовая привязка |
| `CoursesContactsRelationship` | Связь контакта с курсами |
| `ClassContactsRelationship` | Связь контакта с классами |
| `User` | Пользователи/ответственные |

Schema discovery выполняется read-only через `scripts/export_tallanto_schema.py`.

### Writes

На текущем этапе SaaS/productization ветки Tallanto writeback запрещен. Tallanto
остается источником правды для своих полей, а AI-слой только читает и
компактно переносит контекст в отчеты/amoCRM AI fields.

## Rollout policy

1. `shadow`: читать amoCRM/Tallanto, формировать preview reports, ничего не писать.
2. `dry_run`: строить payload для writeback и сохранять report rows со статусом
   `dry_run`.
3. `limited_live`: live write только на малом batch с ручным просмотром отчета.
4. `supervised_live`: live write по расписанию, но только guarded entrypoints и
   safe mode.
5. `client_appliance`: отдельный tenant config, отдельный data folder, отдельные
   credentials, documented rollback.

## Rollback

Минимальный rollback для live write:

- сохранять `*_writeback_report.json` и `*_writeback_report.xlsx`;
- не перезаписывать непустые lead fields при safe mode;
- перед большим live batch делать export текущих target fields;
- при ошибке использовать report rows с `updated_fields` и entity id для ручного
  восстановления.

## Current implementation references

| Area | File |
|---|---|
| Contact writeback CLI | `scripts/write_amo_ready_contacts.py` |
| Deal writeback CLI | `scripts/write_recent_actionable_deals.py` |
| HTTP deal writeback gate | `src/mango_mvp/amocrm_runtime/routers/deals.py` |
| amoCRM field update helpers | `src/mango_mvp/amocrm_runtime/amo_integration.py` |
| Deal payload mapping | `src/mango_mvp/amocrm_runtime/deals.py` |
| Tallanto context | `src/mango_mvp/amocrm_runtime/tallanto_context.py` |
| Tallanto schema export | `src/mango_mvp/amocrm_runtime/tallanto_export.py` |
| Safety tests | `tests/test_amo_writeback_guards.py` |
