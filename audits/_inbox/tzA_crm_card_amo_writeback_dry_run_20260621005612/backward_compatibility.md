# Обратная совместимость

- Существующие contact/deal live writers не изменены.
- `customer_timeline` и read_api не изменены.
- Новый CLI отдельный; без прямого запуска не влияет на runtime.
- Live-флаги в новом CLI только отказывают, не пишут.
- Новые тесты не требуют AMO/Tallanto/live credentials.
