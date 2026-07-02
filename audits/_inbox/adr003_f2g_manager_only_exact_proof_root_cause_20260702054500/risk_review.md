# Risk Review

## Главный риск

Принять `manager_only + exact proof` за разрешение понижать маршрут.

## Защита

F2g делает только report. В коде нет runtime-проводки.

Acceptance отчёта всегда оставляет `active_readiness=no_go`.

## Что блокирует active

- route уже `manager_only`;
- runtime retrieval не доставил exact fact keys;
- в одном случае confidence ниже порога;
- во втором frame сам считает `manager_action/check_availability`;
- в обоих есть `missing_facts`.

## Остаточный риск

Следующий этап должен быть отдельной shadow-фазой. Любая active-логика должна пройти новый регрейд Claude #1 и отдельное "да" Дмитрия.
