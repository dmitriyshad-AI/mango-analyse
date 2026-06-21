# Semantic Review

Verdict: PASS_WITH_NOTES.

Artifact: правило видимости bot-safe памяти в черновике бота.

What passed:

- Бот видит `active_brand` chunks и `unknown` chunks.
- Явный чужой бренд исключается до prompt.
- Если chunk ошибочно имеет `unknown` и чужой известный бренд одновременно, чужой известный бренд выигрывает и chunk скрывается.
- PII/service-id scan не ослаблялся.
- Выходной бренд-верификатор и P0 backstop не менялись.

Blocking issues:

- Нет.

Non-blocking risks:

- `unknown` chunks теперь видны обоим брендам. Это соответствует решению Дмитрия, но требует постоянной проверки, что текст unknown chunks не содержит явных brand-specific условий чужого бренда.
- Текущая production DB изменилась параллельно после создания test-copy; финальный общий перегон нужно делать на свежей production DB после интеграции D8.

Regression checks:

- Unknown chunk виден активному Foton/UNPK.
- Explicit foreign chunk скрыт.
- Cross-brand customer: Foton prompt содержит Foton, не UNPK; UNPK prompt содержит UNPK, не Foton.
- `unknown_contains_brand_marker=0`, `foton_contains_unpk=0`, `unpk_contains_foton=0` по client-safe text.
