# WhatsApp-обработка — индекс. Claude #2, 2026-05-29

Одна точка входа по всей обработке `all_whatsapp_chats.txt`. Этапы 0–5 закрыты
(5 — это ТЗ, не код). Всё read-only к источнику; боевой код не менялся.

## Данные

- `product_data/transcripts/whatsapp_chats.sqlite` — нормализованная БД (4538 чатов,
  61 540 сообщений). Каноничный артефакт WhatsApp. Только телефоны-ID; реклама, длинные
  WhatsApp-ID, иностранные, «WhatsApp Calls» исключены (решение Дмитрия).
- `whatsapp_normalize_report.json`, `whatsapp_crm_match_report.json` — рядом, статистика.

## Скрипты (в `scripts/`)

| Скрипт | Этап | Что делает |
|---|---|---|
| `whatsapp_normalize.py` | 1 | txt → SQLite, роли, бренд-маркеры, телефоны |
| `whatsapp_match_crm.py` | 2 | матчинг по телефону с `master_contacts_ru.csv` |
| `whatsapp_context_provider.py` | 3 | автономный read-only контекст по телефону (fallback) |
| `whatsapp_analytics.py` | 4 | сезонные темы, реальные P0, бренд-смешение |
| `whatsapp_analytics_part2.py` | 4 | сравнение с звонками, тон менеджеров |
| `whatsapp_p0_frequency.py` | 4 | частота P0 по бренду/сезону + базовая доля |
| `whatsapp_verify.py` | 4.5 | проверка целостности/бренда/матчинга |

## Отчёты (в `D1_audit_backlog/`)

- `whatsapp_step0_existing_infrastructure_2026-05-29.md` — разведка.
- `whatsapp_seasonal_topics_2026-05-29.md` — темы по месяцам/брендам.
- `whatsapp_real_p0_phrasings_2026-05-29.md` — живые формулировки претензий.
- `whatsapp_brand_leaks_managers_2026-05-29.md` — бренд-смешение в исходящих.
- `whatsapp_channel_comparison_2026-05-29.md` — WhatsApp vs звонки.
- `whatsapp_manager_tone_2026-05-29.md` — тон менеджеров для X2.
- `whatsapp_p0_frequency_by_brand_season_2026-05-29.md` — частота/базовая доля P0.
- `whatsapp_verification_2026-05-29.md` — вердикт качества PASS.
- `whatsapp_stage5_integration_tz_2026-05-29.md` — ТЗ врезки в timeline.

## Ключевые факты

- Матчинг с CRM: 2764 чата (61%), `primary_phone == norm(chat_id)` сходится на 100%.
- Бренд-метки точные: риски «фотон»(физика)/«долями»(частями) проверены — не подтвердились.
- Звонки только с 01.2025 → весь 2024 WhatsApp = единственный письменный след; 1773 клиента
  (39%) вообще без звонков.
- P0 редок: 0,46% клиентских сообщений.

## Что дальше и чья зона

| Действие | Владелец | Статус |
|---|---|---|
| Реализация Этапа 5 (врезка в timeline) | Codex + «ок» Дмитрия | ждёт решения |
| Маршрут mixed/null (нейтральный tenant) | Дмитрий подтверждает перед `--apply` | рекомендация дана |
| Ручная разметка mixed (917) | Дмитрий (просил пока не делать) | отложено |
| Расширенный smoke до публичного трафика | позже | не сейчас |

Claude #2 свою часть (анализ + ТЗ, read-only) по WhatsApp закрыл. Реализация — за Codex.
