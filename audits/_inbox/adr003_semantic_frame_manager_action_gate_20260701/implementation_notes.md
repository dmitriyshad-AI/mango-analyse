# ADR-003 SemanticFrame manager-action gate

Дата: 2026-07-01

## Что сделано

- Добавлен default-OFF флаг `TELEGRAM_SEMANTIC_FRAME_MANAGER_ACTION_GATE`.
- В direct-path цепочку добавлен узкий гейт после posthoc `SemanticFrame` и перед `frame_decision_shadow`.
- Гейт может только повысить автономный маршрут `bot_answer_self*` до `draft_for_manager`.
- Гейт не меняет текст ответа, не поднимает в `manager_only`, не понижает существующий `draft_for_manager`/`manager_only`.
- Активный сигнал берется только из posthoc frame со статусом `ok`; старый inline `semantic_frame_shadow` не используется как активный источник.
- Исправлено чтение `must_handoff`: строка `"false"` больше не трактуется как `True`.
- Обновлен ADR-003 snapshot моратория: добавлены только технические константы нового OFF-гейта, без новых regex/keyword-правил сырого клиентского текста.

## Условия срабатывания

Гейт срабатывает только если все условия выполнены:

- флаг включен явно;
- route до гейта автономный;
- posthoc `SemanticFrame` есть и `semantic_frame_posthoc_shadow.status == "ok"`;
- `confidence >= 0.8`;
- `risk_class == "manager_action"`;
- `must_handoff is True` или `answerability == "manager_only"`;
- действие из узкого набора: `handoff_manager`, `send_document`, `check_availability` только в стадиях `closing/post_payment/support`, либо `enroll` в closing/post-payment контексте, либо `paid/dispute` с manager-oriented action.

## Offline eval

На сохраненном full131 paired-enrichment артефакте:

- первый вариант: 17 promoted / 241, 0 text changes;
- после сужения `check_availability`: 12 promoted / 241, 0 text changes.

Сужение убрало ранние справочные `check_availability` на стадиях `interest/qualification`, где SemanticFrame был слишком осторожен.

## Почему это не глобальная миграция

Gold-review показал, что SemanticFrame пока часто слишком осторожен. Поэтому этот шаг не делает frame главным источником всех решений и не включает флаг в профиль. Это проверяемый маленький мост: frame может только остановить потенциально опасный self-answer там, где он уверенно видит действие менеджера.
