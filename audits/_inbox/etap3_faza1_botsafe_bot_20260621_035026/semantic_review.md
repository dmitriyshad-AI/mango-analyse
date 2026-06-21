# Semantic Review

Verdict: PASS_WITH_NOTES.

Artifact: подключение bot-safe выжимки клиента к черновику Telegram-бота.

Audience: менеджер, который проверяет черновик перед отправкой клиенту.

Formal pass:

- Полный `pytest` зеленый: 3493 passed, 5 skipped.
- Флаг `TELEGRAM_BOT_SAFE_CRM_CONTEXT` выключен по умолчанию.
- Подключение не меняет live/autoreply режим.
- PII/service-id фильтр покрыт тестами.

Semantic pass:

- Боту передается не весь профиль клиента, а только `bot_context(allowed_only=True)`.
- Выжимки фильтруются по активному бренду канала через `relevance_tags`.
- Память используется как контекст продолжения диалога, но промпт прямо запрещает брать цены, даты и условия из памяти вместо подтвержденных фактов.
- В промпт не попадают `customer_id`, `source_ref`, `chunk_id`, `event_id`, `opportunity_id`.
- При неизвестном бренде, неоднозначной identity, отсутствующей DB или PII в выжимке контекст не подмешивается.

Blocking issues:

- Нет.

Non-blocking risks:

- Эффект качества не доказан локальными unit-тестами: нужен M1 off/on замер по сырью.
- Большая часть боевой памяти имеет brand tag `unknown`; такие выжимки не попадут в prompt до уточнения бренда.
- Если будущий генератор bot-safe summary начнет писать клиентские телефоны/email в текст, этот слой заблокирует такие chunks, но процент полезной памяти снизится.

Required regression checks:

- Default OFF: блок памяти отсутствует в direct prompt.
- Active brand filter: Foton не получает UNPK summary и наоборот.
- PII/service ids: телефон, email, `customer:`, `botsafe:`, `timeline_event:`, `bot_context_chunk:` не попадают в prompt/output.
- Ambiguous identity: context не подмешивается.

Recommended next action:

- Запустить M1 пару OFF/ON на подготовленном target set и разбирать `dynamic_dialog_transcripts.jsonl`: память должна уменьшать лишние переспросы, не создавать brand/PII/P0 нарушений и не переводить черновик в автоответ.
