# ADR-003 F2g Manager-Only Exact-Proof Root Cause

## Что сделано

Добавлен report-only scorer `scripts/report_adr003_manager_only_exact_proof_root_cause.py`.

Он строится поверх F2f readiness report и диагностирует только строки `manager_only + exact product proof`.

## Что это НЕ делает

- Не меняет route/text.
- Не подключается к direct path/provider/profile.
- Не включает флаги.
- Не понижает `manager_only`.
- Не добавляет regex-понимание.

## Реальный пересчёт 36ea110

См. `real_36ea110/adr003_manager_only_exact_proof_root_cause_report.md`:

- manager-only exact-proof rows: 2;
- runtime exact proof missing: 2;
- conversation plan lacks product scope: 2;
- frame says manager action: 1;
- low confidence: 1.

## Вывод

Ф3 active остаётся NO-GO.

Обе строки имеют exact proof только в offline F2e/F2f catalog, но runtime retrieval/direct metadata этот fact не доставили (`candidate_count=0`, `selected_exact_ids=[]`).

Следующий безопасный шаг, если Claude #1 согласен: отдельная shadow-фаза для retrieval/evidence injection диагностики или калибровки frame. Не active route demotion.
