# Semantic review

Вердикт: `formal_pass`, `semantic_pass` на включение флага еще не выдан.

## Проверено смыслово

- Гейт не добавляет новое regex/keyword-понимание клиента.
- Гейт не решает P0 и не снимает существующие P0/manager routes.
- Гейт не меняет клиентский текст, только переводит автономный результат в `draft_for_manager`.
- Forward-payment/self-safe сценарий не переводится в manager route только из-за `send_payment_link`.
- Safe trial/signup scenario со строковым `"must_handoff": "false"` остается self-answer.
- Inline frame без posthoc status `ok` не считается активным источником.
- Offline eval на full131 после сужения `check_availability`: 12/241 ходов повышаются в `draft_for_manager`, текст меняется 0 раз.
- Первый offline eval дал 17 promoted, но 5 были слишком широкими `check_availability` на раннем `interest/qualification`; правило сужено до `closing/post_payment/support`.

## Что осталось на semantic-регрейд

- Прогнать ON-флаг на frozen eval/gold очереди manager-action.
- Проверить false-positive на цены, адрес, платформа, формат, порядок записи, рассрочка, pre-sale refund, "подумаем/оплачу позже", safe deferral.
- Проверить, что реальные кейсы менеджерского действия поднимаются в `draft_for_manager`: наличие мест/бронь, чек/оплата, запрос администратора, документы/запись.

До этого флаг не включать в профиль и не использовать в live.
