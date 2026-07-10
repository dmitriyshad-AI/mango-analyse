# Risk Review

Runtime risk: низкий. Изменён только offline/report-only скорер.

Основные риски:
- ложное чувство готовности renderer из-за brand-aware lookup;
- регрессия отчёта при duplicate `fact_key`;
- слишком широкий marker `ит` внутри слов вроде `питание`.

Митигаторы:
- добавлен source-axis blocker перед renderer;
- добавлены тесты на duplicate `fact_key` и mismatch фактов про питание/медицину/охрану/места/доступ после оплаты;
- широкий marker `ит` заменён на явные формы `ит-направ` / `ит направление`;
- полный pytest зелёный.

Запрещённые зоны не затронуты: live bot, Wappi, AMO/Tallanto/CRM, профиль, provider/direct_path runtime.
