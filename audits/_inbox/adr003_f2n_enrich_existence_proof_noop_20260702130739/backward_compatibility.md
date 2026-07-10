# Backward Compatibility

Runtime behavior не меняется.

Изменён только offline/eval режим `--semantic-frame-enrich-from`. Он сохраняет
frozen route/text/safety/checklist и добавляет metadata.

Совместимость отчётов сохраняется: существующие поля SemanticFrame/self-answer
остались, добавляется proof metadata внутри `bot_direct_path`.
