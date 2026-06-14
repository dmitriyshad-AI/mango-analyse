# Backward compatibility

- Новые флаги default OFF.
- Новые поля metadata добавляются аддитивно.
- Старые поля `llm_retrieve.selected_exact_ids`, `selected_adjacent_ids`, `invalid_ids`, `fallback_reason` сохранены.
- При `TELEGRAM_LLM_RETRIEVE=0` shadow/model-driven режимы не создают отдельный LLM-вызов.
- `dynamic_turns.csv` получил новое поле `bot_fact_retrieval_trace`, остальные существующие поля не переименованы.
- Фактический выбор в режиме B ограничен только direct-path retriever seam; scope/autonomy/missing/memory не переподключались.
