# Что сделано

- Добавлен dry-run контур `src/mango_mvp/crm_card_amo_writeback.py`.
  - Карточка строится из `read_api.customer_profile` через `build_crm_card_projection`.
  - D8 next-step используется из `profile["next_step_resolution"]`, который уже протаскивает read_api.
  - История сжимается через `CrmHistorySummarizer`; CLI default = `rule`, без внешнего LLM.
- Добавлен CLI `scripts/dry_run_crm_card_amo_writeback.py`.
  - Пишет отчёты только в `audits/_inbox`.
  - Live-флаги есть только как hard refusal: без отдельной live-команды ничего не пишет.
- AMO payload маппится в две сущности:
  - contact: `AI-рекомендованный следующий шаг`, `Последняя AI-сводка`, `Авто история общения`;
  - lead/deal: `AI-рекомендованный следующий шаг`, `AI-сводка по сделке`, `AI-история по сделке`.
- Для D7 guard вход явно заполняется:
  - `active_brand`;
  - `deal_brand`;
  - `open_deal_count`.
- Brand/multi-deal guard вызываются явно; без `active_brand/open_deal_count` запись блокируется.
- D7 snapshot/anti-clobber используется на dry-run:
  - `build_pre_write_snapshot_rows`;
  - `pre_patch_write_decisions`;
  - local dry-run journal `written-dry`;
  - PATCH/send функций в dry-run контуре нет.
- Для brand сделки добавлен read-only SQLite lookup `customer_opportunities.record_json`, потому что read_api manager projection не отдаёт `product_context.brand`.

# Что не делал

- Не запускал live AMO write.
- Не писал в Tallanto.
- Не отправлял сообщения.
- Не refresh-ил OAuth token.
- Не писал в `stable_runtime`.
- Не менял `customer_timeline`.

# Реальный dry-run

Реальный AMO GET dry-run заблокирован до сети:

`AMO read-only dry-run blocked: active OAuth connection is missing.`

Проверено также, что runtime DB в главной папке содержит таблицу `amo_integration_connections`, но активных соединений в ней `0`. Поэтому корректный результат текущего окружения — blocked, а не попытка refresh/write.
