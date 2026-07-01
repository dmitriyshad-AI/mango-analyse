# Semantic review

- Бренды раздельны: клиентский ответ не меняется, fact retrieval/brand guards не менялись.
- Цены/даты/условия из KB: retrieval/output verification не менялись, новый слой не добавляет facts и не меняет текст.
- P0 и ПДн не ослаблены: P0 floor и scrub не менялись; evidence в frame уже проходит существующую маскировку SemanticFrame, новый shadow пишет только count evidence, не raw evidence.
- Semantic verdict: `formal_pass`; клиентский смысл не изменяется, но активное включение SemanticFrame для решений требует отдельного semantic regread.
