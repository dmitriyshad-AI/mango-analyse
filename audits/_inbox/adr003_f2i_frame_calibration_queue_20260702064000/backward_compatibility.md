# Backward Compatibility

- Код бота не менялся.
- `provider.py`, `direct_path.py`, `post_layers.py`, profile flags и P0-floor не трогались.
- Новый скрипт не импортируется runtime-путём.
- Новый тест изолирован и использует синтетические JSONL/KB fixtures.
- Live process не перезапускался.

Ожидаемое влияние на существующее поведение: ноль.
