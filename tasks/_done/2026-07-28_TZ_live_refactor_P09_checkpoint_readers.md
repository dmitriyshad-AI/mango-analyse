> DONE 2026-07-28 20:56 | ветка main | codex

> TAKE 2026-07-28 20:49 | ветка main | codex

Ветка: main
Зоны: src/mango_mvp/customer_timeline/wappi_history_import.py, src/mango_mvp/customer_timeline/amo_incremental.py, src/mango_mvp/customer_timeline/tallanto_cards_sync.py, tests/test_wappi_history_checkpoint.py, tests/test_customer_timeline_amo_incremental.py, tests/test_customer_timeline_tallanto_cards_sync.py, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_wappi_history_checkpoint.py tests/test_customer_timeline_amo_incremental.py tests/test_customer_timeline_tallanto_cards_sync.py
Семантический-аудит: нет

# P09: безопасное чтение контрольных точек источников

## Цель

Повреждённая UTF-8 запись или контрольная точка чужой версии не должна ронять
ночной процесс и не должна использовать несовместимое состояние продолжения.

## Минимальный дифф

- Wappi: считать `UnicodeDecodeError` повреждением файла; существующую проверку
  версии не дублировать.
- AMO и Tallanto cards: считать `UnicodeDecodeError` повреждением и принимать
  состояние только при точном совпадении уже существующей версии схемы.
- Не строить общий helper: три функции малы и живут у разных владельцев данных.

## Приёмка

- Усечённый UTF-8 во всех трёх файлах даёт пустое безопасное состояние.
- Чужая версия AMO/Tallanto даёт пустое состояние; Wappi уже покрыт текущим кодом.
- Корректные файлы продолжают читаться.
- Целевой и полный pytest зелёные.

## СТОП

- Не менять реальные контрольные точки, базы или внешние системы.
