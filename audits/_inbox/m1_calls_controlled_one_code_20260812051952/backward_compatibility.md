# Обратная совместимость

- Форматы: штатный `service_cutover` сохранён; новый режим
  `isolated_controlled` включается только явным scope/config.
- Потребители: существующие service callers не переводятся на controlled-one;
  production Customer Timeline, AMO, Tallanto и CRM не затронуты.
- Старый runtime остаётся доступным как откат и не изменялся.
