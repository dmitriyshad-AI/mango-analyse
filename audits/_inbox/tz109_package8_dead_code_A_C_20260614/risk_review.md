# Risk review

## Риск

Удаление файлов безопасно только если нет импортёров.

## Проверка

`rg` по `src`, `tests`, `scripts` не нашёл ссылок на `tallanto_deal_ranking`, `tallanto_matching`, `tallanto_premature_close`.

## Остаточный риск

Внешний код вне репозитория теоретически мог импортировать эти приватные модули. Внутри проекта импортёров нет.

`subscription_llm_parts/monolith.py` не удалялся из-за запаркованной Части B.
