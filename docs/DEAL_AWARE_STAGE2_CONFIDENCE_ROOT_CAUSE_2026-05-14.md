# Stage2 Confidence Root Cause

## Короткий вывод

Старый массовый Stage2 warning заменён на stage2_confidence_low. Теперь предупреждение появляется только у действительно низкого confidence, а обычный medium не засоряет ROP/live-кандидаты массовой технической меткой.

## Политика для live-pilot

Для РОП-workbook низкий confidence остаётся строковым предупреждением. Для будущего live-pilot он должен проверяться через Stage1 source, frozen corpus, readback/rollback и Claude preflight.

## Распределение Stage2 по всему корпусу

| decision | confidence | rows |
|---|---:|---:|
| linked_single_deal_candidate | high | 1642 |
| linked_single_deal_candidate | low | 1290 |
| linked_single_deal_candidate | medium | 2337 |
| manual_review_all_candidates_terminal | low | 2226 |
| manual_review_missing_phone | none | 1053 |
| manual_review_multiple_active_deals | low | 570 |
| manual_review_no_deal_candidate | none | 20810 |
| manual_review_single_terminal_deal_candidate | low | 5155 |
| skipped_non_contentful_call | none | 21325 |
| skipped_non_sales_call | none | 8424 |

## Что это значит

- 709 строк Stage6 не являются 709 плохими строками.
- Это 723 кандидата Stage5 минус 14 ранних блокеров; затем Stage6 оставил 680 dry-run и 29 текстовых блокеров.
- Массовая техническая метка старого Stage2 больше не должна появляться в новых пакетах.
- Новый сигнал `stage2_confidence_low` означает действительно низкую уверенность привязки.
- Для следующих live-партий нужен не ручной запрет по этой метке, а отдельные защиты: проверка deal_id, качество текста, dry-run, readback, rollback.
