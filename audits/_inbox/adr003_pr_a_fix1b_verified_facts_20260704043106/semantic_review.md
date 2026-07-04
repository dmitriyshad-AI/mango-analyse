# Semantic review

## Проверка смысла

Фикс не генерирует новый текст и не добавляет новые факты. Он меняет только маршрут уже готового черновика из `draft_for_manager` в `bot_answer_self_for_pilot` при полном доказательстве, что текст безопасен и покрыт свежими client-safe фактами.

## Бренды

Чужой бренд рядом с подтверждённым фактом блокирует коридор. В микро-замере `neg_foreign_brand` остался `draft_for_manager`, `neg_promoted=[]`.

## Деньги/числа

Неподтверждённое дополнительное число блокирует коридор. В микро-замере `neg_extra_number` остался `draft_for_manager`.

## Live-наличие

Обещание мест/группы/брони блокирует коридор. В микро-замере `neg_live_availability` остался `draft_for_manager`. Старый live-status floor не обходится.

## P0

P0/high-risk input и high-risk topic не проходят коридор. В микро-замере `neg_p0_input` и `neg_high_risk_topic` ушли в `manager_only`.

## Вывод

`semantic_pass` для PR-A как локального default-OFF коридора: смысловой риск ограничен, потому что изменение не создаёт новый ответ и не ослабляет P0/live/brand/fabrication полы.
