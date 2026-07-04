# Что сделано

- Добавлен default-OFF флаг `TELEGRAM_FIX1B_AUTONOMY_VERIFIED_FACTS`.
- В `apply_autonomy_matrix_guard()` добавлен узкий коридор, который может снять только два ложных демоута:
  - `autonomy_default_cautious_missing_facts`;
  - `autonomy_default_cautious_unverified_fact`.
- Коридор требует одновременно:
  - `message_type == question`;
  - topic из `AUTONOMY_MATRIX_SAFE_TOPIC_IDS` и разрешён policy;
  - активный бренд определён;
  - нет P0/high-risk по result и input;
  - есть fresh/client-safe факт;
  - весь черновик поддержан фактами через existing fact support helpers;
  - нет неподтверждённых чисел/дат;
  - нет чужого бренда;
  - нет live-обещаний мест/групп/брони.
- Коридор также не срабатывает, если результат уже несёт `conversation_intent_plan_live_availability`: этот live-status пол остаётся выше `fix1b`, а trace пишет `no_op`, не ложный `fix1b_promote`.
- Добавлен trace-класс `fix1b` через `semantic_reading_trace_record`, но он не добавлен в `TELEGRAM_SEMANTIC_READING_CLASSES`.
- Обновлены `docs/ADR003_REGEX_UNDERSTANDING_MORATORIUM.md` и `docs/ADR003_ETAP_T_DECISIONS.md`.
- Добавлены unit-тесты: POS, OFF, 8 NEG-условий и 3 partial-support стоп-кейса.

# Почему так

Fix1b лечит не понимание клиента, а ложное понижение уже готового проверенного ответа. Поэтому новый код не добавляет regex/marker-понимание и не меняет P0/live/brand/fabrication полы.

Вместо полной строки `_claim_supported_by_facts(result.draft_text, facts)` использован более устойчивый вариант: точная поддержка или hard-anchor support, плюс отдельные выходные проверки неподтверждённых чисел/дат, чужого бренда и live-claim. Это нужно, потому что человеческий черновик часто содержит вежливую обвязку вокруг факта.

# Как проверялось

См. `test_output.txt`. Дополнительно проведён локальный deterministic микро-замер на 20 кейсах: 10 POS verified справок и 10 NEG/stop условий.

# Что осталось

- Флаг не включён в профиль и не включён в live.
- Формулировочный дефект `наличии` -> live-marker в налоговом тексте обнаружен на промежуточном микро-кейсе, но не исправлялся здесь: это отдельная legacy-lazagna задача, не PR-A.
- Динамический model-run на M1 не выполнялся по прямому указанию Дмитрия: измерения делались локально.
