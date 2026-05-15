# Timeline Coverage Sample

Добавлен read-only скрипт:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/audit_customer_timeline_coverage.py \
  --deal-aware-candidates <path/to/deal_stage4_deal_candidates.csv> \
  --timeline-db <path/to/customer_timeline.sqlite> \
  --out-root <path/to/audit_output>
```

Выходные файлы:

- `timeline_coverage_report.csv`
- `summary.json`

Проверенные счетчики:

- количество deal-aware строк;
- количество уникальных телефонов;
- сколько телефонов найдено в timeline;
- сколько телефонов отсутствует;
- наличие звонков, AMO, Tallanto, Telegram/email по телефону;
- warnings по fallback.

Output в `stable_runtime` заблокирован guard-ом.
