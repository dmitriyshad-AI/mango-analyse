# Semantic review

Статус: local semantic pass for the implemented guard class; full behavioral semantic pass still requires Claude/dynamic transcript review.

## Что проверено

- Positive cases: адрес, маткапитал, электронные документы, точная цена после плохого черновика, факт при недоступном critic теперь могут быть отвечены автономно из retrieved-фактов.
- Negative cases: P0/refund, жалоба, смена/лагерь против регулярного курса, прямой перевод против Т-Банк, cross-brand текст и неподтвержденное число не проходят в клиентский ответ.
- Клиентский текст recover строится только из retrieved-фактов или уже существующих verified/composition helpers.

## Смысловой риск

- Recover расширяет автономность, поэтому главный риск - ложный перенос соседнего факта. Для этого добавлены явные NEG-тесты по P0, refund, complaint, camp-vs-regular, payment-method и cross-brand.
- Полный semantic_pass по реальному поведению должен подтвердить Claude на динамических транскриптах: тесты проверяют класс, но не все реальные формулировки клиента.
