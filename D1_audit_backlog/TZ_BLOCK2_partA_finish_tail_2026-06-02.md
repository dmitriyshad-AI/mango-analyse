# ТЗ MAIN (Кодекс) — Блок 2, Часть A: добить хвост уходов. 2026-06-02.

Автор: Клод 1. Точки/причина подтверждены чтением сырья прогона BLOCK1. Эта часть НЕ трогает P0/безопасность
и НЕ зависит от исхода замера Block 1.1 — её можно делать сейчас, параллельно замеру 1.1 на M1. Часть B
(тон + «чёткий шаг») — ОТДЕЛЬНО, после регрейда Block 1.1. Отдельный коммит.

## Точная причина остатка (по сырью, не по догадке)
Остаточные 9 уходов BLOCK1 — НЕ retrieval-мисс (факты достаются). Разбор:

1. **terminal/платформа** (`foton_V4_platform_02` t4 «как зайти в личный кабинет»): `recovery_candidate`
   ПОСТРОЕН и `validated=True` («В личный кабинет ученик заходит на учебной платформе…»), но НЕ применён,
   потому что `terminal_safe_template_applied` поставил `route="manager_only"`, а
   `_validated_guardchain_recovery_candidate` (subscription_llm ~119, строка ~130) ОТКАЗЫВАЕТ при
   `route=="manager_only"`. То есть информационный шаблон сам ставит manager_only и этим блокирует
   собственный валидный ответ. ← ГЛАВНЫЙ корень хвоста.
2. **tax** (`unpk_V5_tax_02` t2 «налоговый вычет потом можно?»): `recovery_candidate` ПУСТ — композитор не
   собрал tax-ответ (scope/builder не сработал), хотя факты налога в retrieved.
3. **trial/пробное** (`unpk_E9_trial_02`): не в списке информационных + понимание поставило manager_only,
   кандидат не строится.
4. **hard_verification** (`program_15`, `program_02`): scope-замок recover не пустил кандидата.

## Правки

### A1 — информационный safe_template уступает валидному кандидату ДАЖЕ при manager_only (главное)
В `_validated_guardchain_recovery_candidate` (subscription_llm): сейчас блок `if result.route ==
"manager_only" ...: return ""` рубит всё. Разрешить yield при `route=="manager_only"`, ЕСЛИ manager_only
поставлен ИНФОРМАЦИОННЫМ safe_template (флаги `matkap_safe_template_applied`, `tax_safe_template_applied`,
`olympiad_online_safe_template_applied`, `terminal_safe_template_applied`, платформа) И НЕТ P0/high-risk и
НЕТ блокирующих флагов (`_GUARDCHAIN_RECOVERY_BLOCKING_FLAGS`). Т.е. различать manager_only «от
информационного шаблона» (можно уступить) vs manager_only «от P0/безопасности» (НЕЛЬЗЯ). Кандидат всё равно
проходит повторный `verify_dialogue_contract_output` — безопасность держится.
Подтвердить чтением: точную строку manager_only-блока в `_validated_guardchain_recovery_candidate`, набор
информационных флагов, `_GUARDCHAIN_RECOVERY_BLOCKING_FLAGS`.

### A2 — строить кандидат для tax надёжнее
`unpk_V5_tax_02`: кандидат пуст. Убедиться, что для информационных тем (tax/matkap) `recovery_candidate`
строится из client_safe факта (через тот же cite-only/coverage путь, что A2.1), когда факт темы в retrieved.
Если builder для tax не покрывает — добавить tax в источник кандидата (cite-only из `tax.*`/`matkap.*`
client_safe). После Block 1.1 — БЕЗ производных сумм (строгий fact_grounding уже введён в 1.1, не ослабить).

### A3 — trial/пробное в информационные
Если есть client_safe факт про пробное (`*trial*`/`online_trial_fragment.client_safe_text`) — добавить
trial в информационные темы, уступающие валидному ответу (как matkap). NEG: если client_safe факта про
пробное нет — уход остаётся (не выдумывать).

### A4 — ослабить scope-замок на hard_verification_failed
`program_15/02`: на `hard_verification_failed` строить `recovery_candidate` с `allow_key_coverage=True`
(покрытие ключей вместо строгого exact-scope). Выход перепроверяется `_hard_check` — безопасно. NEG:
смена≠курс / чужой бренд / число вне факта → кандидат не строится.

## Тесты + НЕГАТИВНЫЙ контроль (критично)
ПОЗИТИВ:
- «как зайти в личный кабинет» (terminal) → отвечает из факта (route bot_answer_self), а не уход
  (воспроизводит V4_platform_02).
- налоговый вычет с фактом → отвечает (V5_tax_02), БЕЗ производных сумм.
- пробное с client_safe фактом → отвечает.
- hard_verification с key-coverage фактом → отвечает.
НЕГАТИВ (не ослабить):
- manager_only от P0/high-risk/жалобы/возврата → кандидат НЕ применяется (yield только для информационных).
- защитные safe_template (cross_brand/result_guarantee/admission_guarantee/unsupported_promise/zero_collect/
  payment_dispute) → НЕ обходятся.
- кандидат не прошёл `verify` (чужой бренд/число вне факта) → уход остаётся.
- Block 1.1 не ослаблен (presale-refund подавление, tax строгий fact_grounding держат).

## Замер
Кодекс гонит pytest+smoke сам; Клод 1 проверяет в песочнике. После — в общий замер Блока 2 (вместе с
Частью B позже). Критерий Части A: остаточные платформа/налог/пробное/hard_verification отвечают; выдумки 0;
HARD-P0/бренд держат. Правило #1: точки `_validated_guardchain_recovery_candidate`, информационные флаги,
builder кандидата, hard_verification scope — подтвердить чтением. Часть B (тон) — НЕ в этом коммите.
