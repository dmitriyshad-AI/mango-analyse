# Phase 12 · TZ-02 — миграция `terminal`/identity в v2

Снимок: HEAD `36e23cb8`. Приоритет диспетчера: 20. Зависит от: дизайна диспетчера.

## 1. Цель
Не раскрывать природу бота («я бот/ИИ/GPT/Claude») и безопасно обрабатывать терминальные служебные случаи (адрес, контакты, офф-топик, мягкий отказ). Главный P0-аспект — **identity** (мета-leak природы бота, 1/212 в прогоне, но юр./доверие-критично).

## 2. Текущее состояние в legacy
- Селектор: `_terminal_safe_template(result, *, client_message, context)` — `subscription_llm.py:4663`. Применение: `:2315-2368` (флаг `terminal_safe_template_applied`); `direct_info_template` ветка → route `draft_for_manager`, иначе `manager_only` (`:2356`).
- Тексты identity: `IDENTITY_PROMPT_SAFE_TEXT:348`, `IDENTITY_FOTON_SAFE_TEXT:352`, `IDENTITY_UNPK_SAFE_TEXT:356`. Прочие терминальные: `ADDRESS_*:365-373`, `CONTACT_*:374-375`, `OFF_TOPIC_*:376-378`, `SOFT_NEGATIVE_HANDOFF_SAFE_TEXT:93`.
- Отдельно есть `guard_identity_disclosure` (`:1556`) — ВЫХОДНОЙ страж по подстроке драфта (находка 11.5). Вызывается в legacy-композициях (`:1316/1389/1475`).
- **В v2-цепочке** `_terminal_safe_template` НЕ вызывается. `guard_identity_disclosure` — ОТКРЫТО, входит ли в v2 (в eval флаг `identity_disclosure_guarded` появлялся → где-то да; подтвердить).

## 3. Точка вставки в v2
Диспетчер, priority 20 (после cross_brand). Identity-часть — критична; address/contact/off-topic — можно мигрировать тем же селектором (P1-инфо, но он внутри terminal). Мигрировать `_terminal_safe_template` целиком как один спец.
**Дополнительно:** убедиться, что выходной `guard_identity_disclosure` стоит в v2 ПОСЛЕ драфта (как safety-net) — если его нет, добавить в v2-цепочку перед sanitize. И применить фикс 11.5: матч по границе слова вместо подстроки (`find_identity_disclosure_phrases:1541`, `phrase in lowered` → `\bphrase\b`).

## 4. Зависимости (KB)
KB-факты не нужны (тексты identity/адрес/контакты — константы в коде + KB-факты адресов/контактов уже есть в v6.3: `locations_*`, `contacts_*`). Блок Г: OK.

## 5. Точная правка
```python
_TERMINAL_SPEC = TemplateSpec(
    name="terminal", priority=20,
    produce=lambda r, cm, ctx: _terminal_safe_template(r, client_message=cm, context=ctx),
    route_on_apply="terminal",  # direct_info → draft_for_manager, иначе manager_only (как :2356)
    flag="terminal_safe_template_applied",
    checklist="Терминальный случай: identity/адрес/контакты/офф-топик — безопасный шаблон.",
)
# Выходной identity-net (если не в v2): result = guard_identity_disclosure(result)  # перед sanitize
# Фикс 11.5 в find_identity_disclosure_phrases: матч по \bphrase\b, не подстрока.
```

## 6. Регрессы
Positive:
1. `«ты бот/ИИ?»` → IDENTITY_*_SAFE_TEXT («я помощник менеджера…»), без «я бот».
2. `«ты GPT/Claude/нейросеть?»` → identity-safe.
3. `«дай адрес»` → ADDRESS_*_SAFE_TEXT (active brand).
4. `«какой телефон?»` → CONTACT_*_SAFE_TEXT.
5. `«расскажи анекдот»` (офф-топик) → OFF_TOPIC_*.
Контрольные negative:
6. Драфт содержит «как и интенсив» / «России» — НЕ ловится identity-стражем (фикс границы слова; иначе ложное `identity_disclosure_guarded`, находка 11.5/`foton C2_multi_subj_02`).
7. Реальная утечка в драфте «я бот»/«GPT» как отдельное слово → `identity_disclosure_guarded` срабатывает (контроль безопасности).
8. Обычный вопрос про курс → terminal НЕ срабатывает.

## 7. Backward compatibility
- Legacy identity-тесты зелёные.
- `V2_are_you_bot_01/02` — identity-safe ответ сохраняется.
- Мета-утечки = 0 (не ухудшать).
- Фикс границы слова не должен пропустить реальные identity-фразы (контроль 7).

## 8. Открытые вопросы Кодексу
1. Входит ли `guard_identity_disclosure` уже в v2 (через `apply_input_policy_guards:2154` или pipeline)? Если да — мигрировать только `_terminal_safe_template` без дубля стража.
2. `IDENTITY_DISCLOSURE_FORBIDDEN_PHRASES` (`draft_prompt_builder.py:22`) — какие короткие фразы дают подстрочные ложные срабатывания (ревизия при переходе на границу слова).
3. Address/contact-часть terminal — оставить в этом спеце (P0-вместе) или вынести в P1.
