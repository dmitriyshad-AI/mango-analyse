# Semantic review

Verdict: `PASS_WITH_NOTES`

## Что прошло

- Отчет не генерирует и не экспортирует клиентский текст.
- Route-only active candidates = `0`; активное включение не предлагается.
- Danger-adjacent строки остаются заблокированными.
- `manager_only` не понижается.
- `platform_current` выделен как taxonomy/proof-axis gap, а не как повод
  демоутить маршрут.

## Риски

- Это диагностический отчет, не улучшение автономности.
- Нельзя чинить `platform_current` новым live regex; нужна структурная ось факта
  или измерительный proof-axis слой.

## Следующее правило

Следующий шаг должен доказать, что platform facts покрывают `platform_current`
структурно и брендово, без клиентского текста и без route/text изменений.
