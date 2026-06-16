# ТЗ-126 (после аудитора) — честная метрика over-handoff (классификатор уходов, observe-only)

- **Дата:** 2026-06-16. Заказчик: Дмитрий. Постановщик: Claude #1 (+ аудитор по коду). Исполнитель: D4. Раннер: `scripts/run_telegram_dynamic_client_sim.py`.
- **Зачем:** метрика over_handoff сливает все уходы → «32%» страшнее реальности. Нужно разложить каждый уход на корзины, чтобы видеть ИСТИННЫЙ дожим. Это МЕТРИКА, поведение бота НЕ меняем.

## Главный принцип (находка аудитора): НЕ строить новый детектор
Половина сигналов уже есть в раннере/проде — переиспользовать, не плодить второй:
- `bot_close_detect.status` (`fired/suppressed_handoff/suppressed_pending`) — боевой детектор закрытия (`_tone_close_detect_is_close_message`, уже отсекает вопрос/exit/«спасибо НО списали»/`message_type==question`). НЕ писать свой regex благодарности.
- `_handoff_fact_level(turn)` → `retrieved_match/same_brand_global_match/wrong_scope/no_match` — есть ли ответуемый факт.
- `_turn_is_real_p0(turn)`; `contact_requested`/action=capture_lead; `bot_route`.

## Классификатор `_classify_handoff_bucket(turn)` — только для turn, где `_is_over_handoff_turn(turn)`
Порядок (первый сработавший = приоритет безопасности):
1. `bot_close_detect.status ∈ {fired, suppressed_handoff, suppressed_pending}` И в client_message НЕТ вопроса → **closing**.
2. `_turn_is_real_p0`: если P0-флаг + `fact_level==retrieved_match` + домен ОТВЕТУЕМ (не refund/оплата/CRM) → **disputed_p0** (видимая под-корзина, не прятать); иначе → **legitimate**.
3. route manager_only + домен оплата/CRM/реквизиты ИЛИ `contact_requested` (захват лида БЕЗ открытого вопроса) ИЛИ `fact_level==no_match` (пробел KB) → **legitimate**.
4. `fact_level ∈ {retrieved_match, same_brand_global_match}` И не P0 И не closing → **upsell_miss** (истинный дожим).
5. `fact_level==wrong_scope` → **upsell_miss** (склоняем к дожиму — факт есть, но не туда).
6. иначе → **unclassified**.

**Асимметрия (защита от скрытия проблемы):** при сомнении между «дожим» и «закрытие/законный» — относить к ДОЖИМУ. Спорное падает в `unclassified`, а не размазывается в безопасные корзины. «Закрытие+вопрос» → всегда дожим (вопрос перебивает благодарность). `wrong_scope` НЕ зачитывать как законный.

## Observe-only (жёстко)
- Классификатор — отдельная постобработка над готовым `transcripts` (рядом с `_over_handoff_metrics`), ничего не мутирует, не импортируется в исполнение бота (`provider/post_layers/_run_once`).
- Тест-инвариант: прогон с классификатором и без — БАЙТ-в-байт идентичные `dynamic_dialog_transcripts.jsonl` и `bot_text/bot_route`; отличается только `summary.json`.
- Считать по ТЕМ ЖЕ handoff-turn, что и `_is_over_handoff_turn` (тот же знаменатель); доли — внутри множества уходов.

## Вывод
В summary блок `over_handoff.buckets`: счётчики 5 корзин (closing/legitimate/disputed_p0/upsell_miss/unclassified) + доли от handoff_turns + 8-10 примеров на корзину (turn id, client_message, bot_route, fact_level). Вывести в `render_summary_md`.

## Валидация
- Прогнать на C0, сверить с ручным разбором: closing ≈39%, legitimate(+disputed_p0) ≈10%, upsell_miss ≈15-20%, остаток unclassified; допуск ±5-7 п.п.
- Confusion matrix классификатор×ручная разметка C0; **ключевой критерий: 0 ручных «дожимов», попавших в closing/legitimate** (FN дожима = 0).
- Observe-only регресс: diff транскриптов до/после пуст; юнит на ~10-15 размеченных turn (закрытие-с-вопросом, «ок», wrong_scope, disputed P0, захват лида).

## Файлы
`scripts/run_telegram_dynamic_client_sim.py` (`_over_handoff_metrics`, `_is_over_handoff_turn`, `_turn_is_real_p0`, `_handoff_fact_level`, `_close_detect_summary`, `render_summary_md`); боевой детектор — `post_layers.py:_tone_close_detect_is_close_message`. Только Codex, без ключей; полный pytest зелёный; мой регрейд.
