# Semantic Output Verifier

Дата: 2026-06-06

Ветка: `codex/semantic-output-verifier-20260606`

Сделано:
- добавлен единый смысловой верификатор финального текста за флагом `TELEGRAM_SEMANTIC_OUTPUT_VERIFIER` (default OFF);
- верификатор ставится перед финальным `apply_authoritative_output_gate`, поэтому детерминированный gate остаётся последним словом;
- поглощены классы старого diagnosis guard в единую схему findings;
- добавлены действия `annotate` и `downgrade_keep_text` в существующий `GATE_BLOCKING_CODES`;
- `downgrade_keep_text` понижает маршрут до `draft_for_manager`, сохраняет исходный текст для менеджера и добавляет `manager_approval_required`/`no_auto_send`;
- `annotate` сохраняет все findings в metadata/summary, но в manager checklist показывает только сжатую заметку;
- fail-soft делает один retry; повторная недоступность даёт `semantic_verifier_unavailable`, маршрут не меняется;
- добавлены `relation_to_base` и `nearest_fact_key` для проверки менеджером;
- runner пишет per-turn `bot_semantic_output_verifier`, summary-блок и отдельные llm call роли.

Не включено:
- будущий автономный fail-closed при недоступном verifier не реализован; оставлен комментарий у флага;
- live exit-прогоны 13+25+hp_topic_change не запускались, по ТЗ их гонит M1 после коммита.
