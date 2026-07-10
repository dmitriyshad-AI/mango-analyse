# Обратная совместимость

- Форматы: production DB schema не менялась. Изменения в репо касаются тестовой fixture и pytest-покрытия CRM export gates.
- Потребители: live bot, AMO writer, Tallanto и Wappi live-loop не запускались и не менялись.
- Transfer package: пересобран локально и теперь указывает на свежий CRM export; старые пакеты не удалялись.
- Golden negative gates: новый тест расширяет регрессию и не ослабляет существующие гейты.
