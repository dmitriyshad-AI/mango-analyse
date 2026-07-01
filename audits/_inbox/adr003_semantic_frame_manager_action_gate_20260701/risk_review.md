# Risk review

## Основные риски

1. Ложные handoff из-за слишком осторожного SemanticFrame.
   - Митигация: default-OFF, confidence >= 0.8, узкий набор действий, posthoc-only, тесты на safe/forward-payment. Первый offline eval поймал широкое `check_availability`; правило сужено до поздних/операционных стадий.

2. Случайное использование старого inline `semantic_frame_shadow` как активного источника.
   - Митигация: gate требует `semantic_frame_posthoc_shadow.status == "ok"`.

3. Строковое значение `"false"` в `must_handoff` могло трактоваться как True.
   - Митигация: добавлен строгий `_semantic_frame_bool`, тест покрывает `"false"`.

4. Snapshot ADR-003 мог скрыть новое regex-понимание.
   - Митигация: snapshot обновлен только на технические константы OFF-гейта; причина зафиксирована в `docs/ADR003_REGEX_UNDERSTANDING_MORATORIUM.md`.

## Не трогалось

- Live bot / Wappi / Telegram sends.
- AMO / Tallanto / CRM.
- Profile default-on flags.
- P0 floor, brand/number/output gates.
