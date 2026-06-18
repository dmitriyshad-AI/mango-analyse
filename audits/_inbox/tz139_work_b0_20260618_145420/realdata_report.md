# TZ139 Work B0 Real-Data Report

## Источник

- Использован доступный снимок `sales_master_export_20260523_audio_working_store_v1/master_contacts_ru.csv`.
- Строк контактов: `16239`.
- SHA совпадает с manifest канонической SQLite: да.
- Путь из старого manifest `sales_master_export_20260521_after_mango_update_v4_runtime_acceptance` локально отсутствует; это уже фиксировалось в Work A real-data pack.

## Числа B0

| Метрика | Значение |
| --- | ---: |
| Customer entries после Work A | `16901` |
| Shared family phone groups | `601` |
| Family customer entries | `1263` |
| Family groups 3+ | `50` |
| Max Tallanto students на одном телефоне | `6` |
| `phone` links already ambiguous before B0 | `1263` |
| `mango_client_phone` links strong before B0 | `1263` |
| `mango_client_phone` links ambiguous after B0 | `1263` |
| Non-family `mango_client_phone` links remain strong | `15638` |

## Проверка инварианта

После B0 для семейного телефона:

- `phone`: `ambiguous/0.55`.
- `mango_client_phone`: `ambiguous/0.55`.

Сырые телефоны, ФИО и email в отчет не включались.
