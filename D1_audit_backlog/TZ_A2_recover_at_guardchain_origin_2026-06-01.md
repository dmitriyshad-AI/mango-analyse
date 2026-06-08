# ТЗ MAIN (Кодекс) — A2: recover в МЕСТЕ РОЖДЕНИЯ ухода (guard-chain subscription_llm). 2026-06-01.

Автор: Клод 1. Точки подтверждены чтением HEAD ae62faa4 (`subscription_llm.py`). Это главный рычаг —
доказан ДВУМЯ регрейдами (`REGRADE_A_recover_M1` + `REGRADE_AB_humane`). A жил ВНУТРИ run_pipeline, но в
боевом пути уход рождается НИЖЕ — в guard-chain subscription_llm. Отдельный коммит.

## Что доказано (по сырью)

A (recover) сработал в живом прогоне 1 раз из ~24 позитивных уходов. Корни:
1. **9/24 рождаются в guard-chain ПОСЛЕ run_pipeline.** Флаги хода: `*_safe_template_applied`
   (matkap/terminal/olympiad) + `dialogue_contract_text_change_blocked` + `route_permission_autonomous_candidate`.
   Пайплайн дал валидного автономного кандидата-ответа, а downstream его перебил.
2. **Вариативность 11%** — путь «ответить/уйти» недетерминирован; A2 (детерминированный recover в финале)
   её тоже снизит.

### Точные места убийства ответа (прочитано)
- `_reverify_dialogue_contract_text_change` (1358): если safe_template заменил draft и перепроверка нашла
  findings → возврат с `route="draft_for_manager"`, `draft_text=SAFE_FALLBACK_DRAFT_TEXT`,
  флаг `dialogue_contract_text_change_blocked` (строки 1408-1415). ← генерик-уход вместо валидного ответа.
- `_dialogue_contract_v2_route_permission_guard` (1417): на 1464 при `draft_for_manager`+автономия+тема
  разрешена ставится только ФЛАГ `..._autonomous_candidate`, БЕЗ повышения до `bot_answer_self`. ← кандидат
  остаётся уходом.

## Решение — пайплайн отдаёт ВАЛИДИРОВАННОГО кандидата, guard-chain его использует

Идея: не реконструировать объекты в subscription_llm и НЕ трогать safe_template'ы. Пайплайн уже умеет
строить безопасного cite-only кандидата (`_cite_only_recover_before_handoff`, оба замка). Пусть он ВСЕГДА,
даже когда решает уйти, кладёт результат в metadata; guard-chain в точках коллапса отдаёт его вместо
генерик-ухода.

### Правка 1 — пайплайн стАшИт валидированного кандидата в metadata
В `run_pipeline` (dialogue_contract_pipeline.py): когда итог — уход (любой safe_fallback/handoff), но
`_cite_only_recover_before_handoff` смог построить прошедший `_hard_check` cite-only ответ — положить его в
результат как поле/метадату `recovery_candidate` (текст + признак client_safe + что НЕ P0). Сам маршрут
пайплайна НЕ менять (пусть остаётся как есть — это вход в guard-chain). Кандидат = строка или None.
Прокинуть его в `metadata["dialogue_contract_pipeline"]["recovery_candidate"]`, доступную subscription_llm.

### Правка 2 — ослабить scope-замок, чтобы кандидат строился чаще
В `_cite_only_recover_before_handoff` на точке `hard_verification_failed` (и при стАшинге кандидата)
вызывать с `allow_key_coverage=True` (покрытие ключей вместо строгого `_has_exact_retrieved_answer_part`).
Безопасно: выход всё равно проходит `_hard_check`. Это поднимет долю построенных кандидатов (живьём
understanding/retrieval редко дают точный exact-scope).

### Правка 3 — guard-chain отдаёт кандидата вместо генерик-ухода (ДВЕ точки)
В `subscription_llm.py`:
- **`_reverify_dialogue_contract_text_change`, перед возвратом SAFE_FALLBACK (1408):** если в metadata есть
  `recovery_candidate` И НЕ P0/жалоба/возврат (проверить `is_high_risk_result`/markers) → вернуть
  `route="bot_answer_self"`, `draft_text=recovery_candidate`, флаг `cite_only_recover_at_guardchain`,
  вместо SAFE_FALLBACK. Иначе — текущий SAFE_FALLBACK.
- **`_dialogue_contract_v2_route_permission_guard`, точка 1464** (draft_for_manager + автономия + тема
  разрешена + autonomous_candidate): если есть `recovery_candidate` и не high-risk → повысить до
  `route="bot_answer_self"` с `draft_text=recovery_candidate`. Иначе — как сейчас (флаг).

### Правка 4 (мелкая, из регрейда B) — анти-повтор на P0 держит manager_only
`P0b_complaint_02` t3: анти-повтор-эскалация дала `route=bot_answer_self` на жалобе (текст безопасен, метка
слетела). В B-анти-повторе: если контекст P0/жалоба/возврат — маршрут остаётся `manager_only`, меняется
только формулировка.

## НЕ трогать (предохранители — оставить жёсткими)
- safe_template'ы `cross_brand`, `result_guarantee`, `admission_guarantee` — это бренд/обещания, НЕ
  обходить кандидатом (для них recovery_candidate не строить: контракт P0/high-risk → recover и так
  возвращает "").
- P0/жалоба/возврат/юр → кандидат НЕ строится и НЕ отдаётся (первый замок recover + high-risk проверка в
  guard-chain).
- `_hard_check` на кандидате обязателен (бренд/мета/числа/p0) — без него не отдавать.

## Тесты + НЕГАТИВНЫЙ контроль (критично)
ПОЗИТИВ:
- маткапитал (`V5_tax`): пайплайн дал кандидата → reverify-коллапс заменён на ответ, route=bot_answer_self
  (воспроизводит matkap_safe_template+text_change_blocked → ответ вместо «Спасибо за сообщение»).
- адрес/расписание/документооборот при autonomous_candidate → повышение до ответа (точка 1464).
- hard_verification_failed с key-coverage фактом → кандидат строится → ответ.
НЕГАТИВ:
- P0/жалоба/возврат → кандидат не строится, генерик/эмпатичный уход остаётся, route=manager_only.
- cross_brand/обещание результата/поступления → НЕ обходить, safe_template держит.
- кандидат не прошёл `_hard_check` (чужой бренд/число вне факта) → не отдавать, уход остаётся.
- анти-повтор на жалобе → route остаётся manager_only (Правка 4).
- автономия выключена / бренд unknown / high-risk → кандидат не повышается (логика 1429-1463 не ослаблена).

## Замер (ВСЕГДА --parallel 4)
Кодекс гонит pytest+smoke сам; Клод 1 проверяет дифф в песочнике. Критерий выхода: на целевом наборе
`cite_only_recover_at_guardchain` replaced > 0 заметно; over-handoff на позитивах падает; выдумки 0;
P0/бренд держат; вариативность ответ/уход снижается. ПОТОМ мерить A+B+A2 vs A на ОДНОЙ машине (изолировать
тон B на стабильной базе).

## Правило #1
Точки `_reverify_dialogue_contract_text_change` (1358, возврат 1408), `_dialogue_contract_v2_route_permission_guard`
(1417, точка 1464), `_cite_only_recover_before_handoff`, поле metadata пайплайна — подтвердить чтением перед
правкой. Хирургически: не трогать safe_template'ы и логику high-risk/autonomy-блокировки.
