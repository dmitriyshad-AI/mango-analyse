# Risk Review

## Главный риск

Перепутать "курс/формат существует" с "есть места / можно записаться / можно оплатить".

## Митигаторы в реализации

- `availability` facts исключаются как положительные доказательства, кроме явного отрицательного факта.
- Payment/refund/installment/contact/documents facts исключены из каталога.
- Payment/enrollment/manager-action факты не используются как positive proof даже если они client-safe.
- Отрицательные факты не имеют приоритета над положительными на широком запросе: отдельная отменённая смена не означает, что весь лагерь не предлагается.
- В docstring и rules каталога явно зафиксировано: live availability, enrollment, booking и payment не доказываются.
- Runtime-проводки нет.

## Остаточный риск

Перед включением в direct path нужен отдельный этап, который:

- строит клиентский текст только из `client_safe_text`;
- запрещает утверждения про свободные места;
- сохраняет manager route при P0/money/brand ambiguity;
- измеряет ложные `exists` на замороженном eval.

## Live safety

Live bot, Wappi, AMO, Tallanto, CRM и профиль не тронуты.
