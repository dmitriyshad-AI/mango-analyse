# Промт MAIN — фикс veto_category в 4a. 2026-05-30.

> Скопировать в окно MAIN Кодекса.

---

Claude #1 прогнал `fa9b285b` в песочнике: **431 passed, 1 failed.** Cross-brand фикс прошёл.

**Новое падение** (после исправленного cross-brand тест дошёл дальше):
`test_pravka4_router_veto_shield_keeps_all_manager_routes` (~строка 3093):
`assert forced_manager_only.veto_category == "force_manager_only"`.

**Диагноз (проверил):** маршрут ВЕРЕН — `route == "manager_only"` (assert строкой выше прошёл), защита
держится, это НЕ дыра. `decide_route` проставляет `veto_category="force_manager_only"` в `RouteDecision`
(subscription_llm.py:6565), но это поле НЕ доходит до итогового результата, который проверяет тест.

**Фикс (правило #1 — подтверди чтением):** в месте, где `decide_route` применяется к результату
(~subscription_llm.py:1418, route-permission guard), перенести `decision.veto_category` (и при
необходимости `safety_flags`/`metadata`) из `RouteDecision` в итоговый `SubscriptionDraftResult`, чтобы
все 4 категории, которые decide_route помечает (`force_manager_only`, `high_risk`, `unknown_brand`,
`autonomy_policy_missing`), проставляли `veto_category` в результате.

**Прогони этот тест целиком** (он идёт дальше — `semantic_unavailable`, `no_draft_fn`): убедись, что
все категории зелёные, не только force_manager_only.

**Важно:** это нужно до 4b — там `decision.veto_category` используется (subscription_llm.py:1432).
Маршруты НЕ менять (они верны). Тесты не гонять (лимит) — Claude #1 прогонит щит + smoke.