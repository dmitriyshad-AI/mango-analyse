# Обратная совместимость

- Форматы: новые поля не добавлены; меняется только дефолт включения существующих `semantic_frame`/`semantic_reading_trace` под `pilot_gold_v1`.
- Потребители: явный `TELEGRAM_SEMANTIC_FRAME_SHADOW=0` выключает frame; явный `TELEGRAM_SEMANTIC_READING_CLASSES=""` выключает reading classes. Вне профиля поведение остается default-off.
