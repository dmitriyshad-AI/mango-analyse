# ТЗ — Шаг 4 Фаза 2 (физический снос мёртвого монолита). Для Кодекса. На cf77d789. 2026-06-03.

Автор: Клод #1. ЗАПУСКАТЬ только ПОСЛЕ чистого финального прогона (3b+Фаза1 закрыты). Основание: карта сноса
`MAP_monolith_demolition_step4_2026-06-02.md` (классификация), актуализированная под cf77d789.

## Предусловие
Фаза 1 (инверсия) сделана: доменные safe_template УСТУПАЮТ диспетчеру (`_safe_template_can_yield_to_dispatcher`
:1089). Значит для мигрированных интентов монолитные доменные шаблоны уже НЕ выигрывают — они мёртвый груз.
Фаза 2 = физически их убрать + их вызовы, БЕЗ изменения поведения (правила уже отвечают).

## СНЕСТИ (17 доменных safe_template + их вызовы + 4 реестровые + мёртвые константы)
Функции (rules_engine их заменил, после инверсии не выигрывают):
`_camp_safe_template, _direct_process_safe_template, _discount_safe_template, _docs_safe_template,
_format_choice_safe_template, _installment_safe_template, _matkap_safe_template, _olympiad_online_safe_template,
_payment_method_safe_template, _price_installment_multitopic_safe_template, _pricing_safe_template,
_recordings_safe_template, _schedule_confirmation_safe_template, _schedule_frequency_safe_template,
_tax_safe_template, _teacher_safe_template, _trial_safe_template`
+ реестровые обёртки `_produce_matkap/tax/olympiad_online/trial_template` (зовут те же функции).
+ их ВЫЗОВЫ в зелёном блоке `apply_high_risk_content_guards` (1833/1851/2535/2545 — проверить ВСЕ места) и
  записи в `DIALOGUE_CONTRACT_V2_TEMPLATE_REGISTRY` (matkap/tax/olympiad/trial specs).
+ текст-константы `*_SAFE_TEXT`, используемые ТОЛЬКО снесёнными функциями (грепнуть каждую: 0 использований
  вне снесённого → удалить; иначе оставить).

## ОСТАВИТЬ (НЕ трогать — безопасность + инфраструктура)
- safety-шаблоны: `cross_brand`/`_cross_brand_safe_template`, `terminal`/`_terminal_safe_template`
  (identity-injection/off-topic/бренд-лояльность), `result_guarantee`, `admission_guarantee` + их specs/produce;
- yield-логика `_safe_template_can_yield_to_dispatcher` (1089) и identity-policy-lock — они теперь маршрутный
  каркас;
- инфраструктура диспетчера (`_safe_template_already_applied`, `_apply_safe_template_spec`,
  `_safe_template_yield_result` и пр.);
- выходной гейт Шага 1, p0_pregate, планнер.

## Хирургия и порядок
1. Сначала удалить ВЫЗОВЫ доменных safe_template в зелёном блоке + доменные записи реестра (matkap/tax/olympiad/
   trial). 2. Затем сами функции (теперь без вызовов). 3. Затем мёртвые константы (после грепа). Зелёный блок
   усохнет до безопасностных (cross_brand/terminal/guarantees); реестр — до 4 safety specs.

## Тест выхода — КРИТИЧНО (снос НЕ должен менять поведение)
1. **Офлайн-регрессия на 58 диалогах Прогона 1 + тон-наборе:** ДОМЕННЫЕ ответы идентичны до/после сноса
   (правила уже отвечали; монолит не выигрывал → удаление не меняет текст). Дифф доменных ответов = 0.
2. **НЕГАТИВ-контроль (безопасность не потеряна со сносом монолита):**
   - identity «это бот?» → политика C (terminal-identity safety цел);
   - cross-brand → каноничная фраза (cross_brand цел);
   - prompt-injection → блок (terminal цел);
   - гарантия результата/поступления → менеджер (result/admission_guarantee целы);
   - P0-возврат → manager_only (p0_pregate цел);
   - бренд/мета/выдумки = 0.
3. **Полный pytest** зелёный (вкл. тесты безопасностных шаблонов; кроме ~9 инфра).
4. **Один точечный прогон 8-10** (`--parallel 4`, X2 off) — поведение идентично до сноса (это контроль, не
   улучшение): домен отвечает правилами, identity/cross/P0/гарантии целы, hard_gate=0.

## Что НЕ делать
НЕ трогать safety-шаблоны/yield/гейт/планнер/p0_pregate/stable_runtime/KB. Удалять константу только после грепа
0-использований. Не git reset/checkout/clean.

## Отчёт
Удалённые функции/вызовы/константы (список) + почему безопасно; дифф офлайн-регрессии (0 на доменных); NEG-
результаты (безопасностные шаблоны живы); ревизия коммита для моей проверки.

После Фазы 2 ЧИСТАЯ АРХИТЕКТУРА ПОЛНОСТЬЮ ГОТОВА: планнер(понимание) → правила(данные) → композер(текст) →
выходной гейт(безопасность), монолитная лазанья снесена, остался только safety-фолбэк. Дальше — карта качества
до пилота (продающее понимание + хвост over-handoff + тон).
