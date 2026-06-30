> DONE 2026-06-30 16:18 | ветка main | codex

> TAKE 2026-06-30 16:05 | ветка main | codex

Ветка: main
Зоны: src/mango_mvp/channels/subscription_llm_parts/, src/mango_mvp/channels/p0_recall_spec.py, product_data/telegram_dynamic_test_sets/, scripts/, tests/, tasks/, docs/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_direct_path_semantic_frame_shadow.py tests/test_subscription_llm_draft_provider.py
Семантический-аудит: да

# ТЗ-154 — ADR-003 Этап 2: развести payment-intent / refund / dispute

Дата: 2026-06-30. Постановщик: Claude #1 + аудитор. Исполнитель: главный Codex D1. Регрейд — Claude #1 по сырью. Канон: main.

Основание: ADR-003 v2. Это первый этап со сменой поведения после shadow-этапа ТЗ-153. Все изменения поведения должны быть за default-OFF флагом; A/B-замер на замороженном eval ТЗ-153.

## Предусловие

Стартовать после ТЗ-153. `payment_class` выводить из полей SemanticFrame (`payment_readiness`, `requested_action`, `risk_class`), а не добавлять новый детерминированный detector по словам.

## Цель

Развести классы:

- `forward_payment`
- `presale_refund_policy`
- `refund_claim`
- `payment_dispute`

Закрыть дефект #16: запрос ссылки/пути оплаты не должен превращаться в текст про возврат.

## Обязательные инварианты

- `refund_claim` и `payment_dispute` остаются P0 / `manager_only`; P0 floor не трогать.
- `presale_refund_policy` остаётся benign self-answer исключением.
- `forward_payment` не попадает в P0-union, не close-message, не обещает live-send, не выдумывает ссылку.
- Готовый текст `PAYMENT_LINK_SAFE_TEXT` использовать как источник истины.
- Флаг default OFF гейтит только новую forward-payment ветку; OFF = прежнее поведение.
- Forward-payment проверять после benign-presale исключения.

## Точки изменения

- `text_hygiene.py`: убрать слепой дефолт-возврат; добавить `forward_payment -> PAYMENT_LINK_SAFE_TEXT`; нераспознанный kind -> нейтральный manager-deferral без слова «возврат».
- `text_hygiene.py`: `_direct_path_p0_hygiene_kind` должен принимать класс из SemanticFrame/metadata и возвращать `forward_payment` до keyword-эвристик; снять keyword-падение «оплата прошла» -> `payment_dispute` для forward-контекста.
- `post_layers.py`: `_direct_path_p0_text` не должен классифицировать по подстроке `оплат`; `payment_dispute` только при реальном floor/payment-dispute code.
- `provider.py`: пробросить frame-derived payment class в scrub context/metadata за флагом.
- Тесты: #16 forward-payment, real dispute/refund, benign presale refund, P0 floor unchanged, inner+outer scrub idempotent, default OFF.

## Явно не делать

- Не менять P0 floor.
- Не включать флаг в профиль.
- Не писать в AMO/Tallanto/CRM.
- Не делать push.
