# Block4 — Tallanto as clean CRM card field

Дата: 2026-06-20/21  
Ветка/worktree: `/Users/dmitrijfabarisov/Projects/Mango_etap1_crm_cards` (`codex/etap1-crm-card-assembler`)  
Live-write: не запускался. `TallantoApiClient` не вызывался.

## Что изменено

- Поле сделки `AI-Tallanto статус по сделке` заменено на понятное менеджеру поле `Статус оплат и занятий`.
- Tallanto-блок убран из `Авто история общения`, чтобы лента касаний не смешивала общение с техническим статусом источника.
- Сырые маркеры `exact_phone_single` / `no_exact_phone_match` по-прежнему нормализуются в человекочитаемый текст.
- Данные берутся только из `timeline_events` и manager facts; внешних вызовов нет.

## Preview-проверка

Preview:

`/Users/dmitrijfabarisov/Projects/Mango_etap1_crm_cards/product_data/customer_timeline/crm_card_preview_20260621_with_channels/crm_cards_preview.xlsx`

Машинная проверка по 200 строкам:

- `Статус оплат и занятий` присутствует: 200
- старое поле `AI-Tallanto статус по сделке` присутствует: 0
- Tallanto-текст в `Авто история общения`: 0
- Tallanto-текст в `AI-история по сделке`: 0

Примеры статусов без ПДн:

- `Tallanto: точного совпадения по телефону нет.`
- `Tallanto: найден один ученик по телефону.`

## Тест

Точечный тест:

`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_crm_card_aggregator.py`

Результат: `7 passed`.

Полный pytest будет запущен после Block5 перед коммитом, как требует ночное ТЗ.
