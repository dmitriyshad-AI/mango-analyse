# TZ-22 D7 A1: live card active brand

Дата: 2026-06-14

## Что сделано

- Добавлен параметр `active_brand` в `build_deal_dossier(...)`.
- Добавлен параметр `active_brand` в `_build_dossier_and_analysis(...)` и проброс в `build_deal_dossier(...)`.
- В `scripts/run_telegram_public_pilot_bots.py` бренд канала передаётся в `build_live_tallanto_context_readonly(...)`, а wrapper передаёт его в `build_live_tallanto_context(...)`.
- В `build_tallanto_live_card(...)` добавлен аварийный флаг `CRM_LIVE_CARD_BRAND_FAILCLOSED`, default OFF:
  - `OFF`: поведение без `active_brand` остаётся старым;
  - `ON`: при пустом бренде карточка блокируется как `brand_unverified`.

## Важные границы

- REST phone-входы в `deals.py` остаются с `active_brand=None`, потому что в этой ветке нет источника бренда.
- `brand_scope_from_filial(...)` не менялся: Фотон по-прежнему не является scope, неизвестные филиалы остаются `unknown`.
- Server-mode public bot route не менялся: A1 закрывает live path `CRM_TALLANTO_MODE=http` через local live wrapper.
- Live AMO/Tallanto write не запускался.
- ASR/analyze/Resolve+Analyze не запускались.
- `stable_runtime` не менялся.
