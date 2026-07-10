> DONE 2026-07-07 01:06 | ветка codex/email-pipeline-restore | codex

> TAKE 2026-07-06 23:58 | ветка codex/email-pipeline-restore | codex

Ветка: codex/email-pipeline-restore
Зоны: src/mango_mvp/customer_timeline/, src/mango_mvp/channels/subscription_llm_parts/, scripts/, tests/, docs/DECISIONS_LOG.md, tasks/_running/, tasks/_done/, audits/_inbox/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest --collect-only -q
Семантический-аудит: да

# TZ FINAL wrapper: память, письма, M1-стенд, D0

Исполнять полное согласованное ТЗ:

`/Users/dmitrijfabarisov/Claude Projects/Foton/2026-07-06_TZ_FINAL_megaplan_pamyat_pisma_stend_D0_dlya_Codex2.md`

Ключевые рамки:

- прод-БД/AMO/Tallanto/live: 0 записей;
- M1 не запускать, только готовить пакет;
- P0/бренд/payment-гейты только ужесточать;
- overlay v3.1 без новых текстов, только разрешённый whitelist изменений;
- D0 nightly только read-only;
- semantic review обязателен.
