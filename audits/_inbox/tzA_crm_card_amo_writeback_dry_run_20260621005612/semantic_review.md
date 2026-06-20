# Semantic review

Verdict: `PASS_WITH_NOTES`.

## Что прошло

- Менеджерская карточка остаётся manager-only; bot-safe слой не трогался.
- В AMO payload не попадают телефон, ФИО, email, ручная `История общения`, статус/этап/ответственный.
- Бренд не угадывается: если `active_brand` не найден, сделочная запись блокируется.
- Multi-deal риск не замалчивается: `open_deal_count` передаётся в guard.
- Tallanto суммы не добавляются в bot-safe и не пишутся отдельно.
- Live-write не запускался.

## Notes

- Реальный AMO diff не получен из-за отсутствия активного read-only AMO контекста.
- По текущей D7-политике `open_deal_count>1` блокирует сделочную запись. Это безопасно, но требует сверки с фразой ТЗ "несколько открытых сделок -> в каждую своя".
- LLM-сжатие в CLI по умолчанию не запускается; default `rule` выбран, чтобы dry-run не делал внешние LLM-вызовы без отдельного решения.

## Regression

- `test_deal_guard_requires_brand_and_open_deal_count_before_write`
- `test_deal_guard_blocks_brand_conflict_and_multiple_open_deals`
- `test_dry_run_entity_saves_snapshot_journal_and_blocks_clobber`
