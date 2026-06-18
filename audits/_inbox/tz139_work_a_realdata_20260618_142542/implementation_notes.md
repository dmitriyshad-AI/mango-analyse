# TZ139 Work A Real-Data Regrade Notes

Дата: 2026-06-18

Статус: Work B не начинался. По условию Claude выполнен read-only прогон Work A на живом снимке и поставлен стоп на регрейд.

## Что изменено перед отчетом

- `no_exact_phone_match` больше не классифицируется как `strong_unique` из-за подстроки `exact`.
- Customer identity для `no_exact_phone_match` получает `partial`, а Tallanto event получает `unmatched`.
- Shared family-phone теперь разделяет не только несколько строк на один телефон, но и одну строку с несколькими `ID Tallanto`.
- Phone identity-link для shared family-phone получает `ambiguous` и confidence `0.55`, чтобы общий телефон не выглядел сильной связью.

## Источники

- Каноническая SQLite открывалась только read-only: `/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/canonical_readonly_20260521_v5/customer_timeline.sqlite`.
- `source_manifest.json` старой базы указывает на отсутствующий путь `sales_master_export_20260521_after_mango_update_v4_runtime_acceptance`.
- Фактически использован доступный снимок `/Users/dmitrijfabarisov/Projects/Mango analyse/stable_runtime/sales_master_export_20260523_audio_working_store_v1/master_contacts_ru.csv`.
- SHA256 использованного `master_contacts_ru.csv` совпадает с SHA из manifest: `2fb85bbbde32899cb5b7c30e62af3618f97a10d951f38144bcbf33d36a3e99ea`.
- AMO snapshot: `deal_aware_amo_live_snapshot_20260513_v2`.

## Safety

- AMO/Tallanto/CRM writes: нет.
- `stable_runtime` writes: нет.
- ASR/R+A/analyze: нет.
- Live DB rewrite: нет.
- В отчет не включались сырые телефоны, ФИО или email; только агрегаты и короткие хэши.
