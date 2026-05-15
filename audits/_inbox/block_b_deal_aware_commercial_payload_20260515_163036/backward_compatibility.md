# Backward Compatibility

## Сохраняется

- `DEAL_AI_FIELDS` оставлен как alias на старые 12 обязательных полей.
- `validate_field_catalog()` по умолчанию требует только старые 12 полей.
- Stage6 dry-run не ломается, если новых AMO-полей нет.
- Readback старых 12 полей не менялся.
- Live-write confirmation и snapshot/rollback из Блока A не менялись.

## Добавлено

- `DEAL_AI_REQUIRED_FIELDS`;
- `DEAL_AI_OPTIONAL_FIELDS`;
- `build_commercial_payload()`;
- `classify_budget_range()`;
- CLI-флаг `--require-commercial-fields`.

## Намеренное поведение

Если optional-поле заполнено в строке, но отсутствует в AMO-каталоге, оно не попадает в `preview_payload` по умолчанию. Строка остается dry-run-ready для старых 12 полей.
