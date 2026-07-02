Ветка: codex/adr003-semanticframe-migration
Зоны: src/mango_mvp/channels/subscription_llm_parts/, tests/test_direct_path_semantic_frame_shadow.py, audits/_inbox/, tasks/_done/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_direct_path_semantic_frame_shadow.py tests/test_report_adr003_*.py
Семантический-аудит: да

# ADR-003 F2l: existence/format proof shadow

## Контекст

F2j улучшил `requested_action` на existence/format subset, но не дал активного
рычага: `risk_class/answerability/must_handoff` остались слишком осторожными.
F2k подтвердил, что простое prompt-only ослабление опасно: без проверенного
факта можно получить красивую метрику и риск выдумки.

## Что сделано

Добавлен новый default-OFF shadow-флаг:

`TELEGRAM_SEMANTIC_FRAME_EXISTENCE_PROOF_SHADOW`.

При включении флага после появления SemanticFrame provider строит только
телеметрию:

- берёт `requested_product` из SemanticFrame;
- ищет exact proof в KB через существующий
  `product_existence_axes_catalog`;
- пишет результат в `metadata.direct_path.semantic_frame_existence_proof_shadow`;
- self-answer shadow учитывает этот proof как отдельный shadow-source в
  freshness trace.

## Что НЕ сделано

- route не меняется;
- draft_text не меняется;
- profile/live не меняются;
- P0-floor/preblock не тронуты;
- manager-only не понижается;
- AMO/CRM/Tallanto/Wappi/live не тронуты.

## Границы

Proof-shadow не является live-availability checker.

Он не доказывает:

- свободные места;
- подходящую группу;
- запись/бронь/лист ожидания;
- оплату/чек/реквизиты;
- документы;
- личный статус клиента или ребёнка.

Это только proof стабильного существования/формата продукта, если KB содержит
fresh client-safe exact fact.

## Проверки

- флаг default OFF и не включён профилем;
- OFF не добавляет proof и не меняет freshness;
- ON добавляет proof-source в metadata и freshness trace;
- route/text остаются прежними;
- ADR003 report tests не сломаны.

## Вердикт

F2l даёт недостающий измерительный слой для будущей автономности, но active
по-прежнему NO-GO до полного paired shadow eval и регрейда Claude #1.
