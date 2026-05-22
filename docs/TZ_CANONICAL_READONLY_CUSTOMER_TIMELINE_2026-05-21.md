# ТЗ: canonical read-only customer_timeline

Дата: 2026-05-21

## Цель

Собрать первый канонический read-only слой истории клиента в `product_data/customer_timeline/canonical_readonly_20260521_v5/`.

Слой нужен как единая локальная база чтения, чтобы по клиенту видеть звонки, AMO/Tallanto-контекст, email-связки и качество идентификации. Это не live-система и не источник записи во внешние сервисы.

## Зачем бизнесу

- Менеджеру и будущему боту нужен не отдельный звонок или сделка, а цельная история клиента.
- Deal-aware слой показал, что без общей истории много строк уходят в ручную проверку.
- Единый timeline позволит понять, где у нас надежная история, где конфликт AMO/Tallanto, где есть письма, а где данных не хватает.

## Жесткие границы

- Не писать в AMO/CRM/Tallanto.
- Не отправлять Telegram/email.
- Не запускать ASR и Resolve+Analyze.
- Не менять `stable_runtime`.
- Читать `stable_runtime` можно только как read-only источник.
- Сырые письма, вложения, OCR-текст, Telegram exports и пароли не переносить в git и не писать в отчеты.
- Не включать `timeline_preview_enabled` и `timeline_primary_read_enabled`.
- Не менять текущий Telegram-пилот, prompt, обработку УНПК-чата и тесты качества Telegram-ответов.
- Не коммитить `customer_timeline.sqlite`, raw CSV/SQLite с персональными данными и derived PII-артефакты.
- Не писать в публичные отчеты хеши телефонов/email, если по ним можно восстанавливать или сопоставлять персональные значения. В отчетах только агрегаты; PII-ключи допустимы только внутри локальной ignored SQLite.

## Источники v5

1. Активный runtime из `stable_runtime/CURRENT_RUNTIME.json`.
2. `master_contacts_ru.csv` - основная строка клиента по телефону.
3. `master_calls_ru.csv` - события звонков Mango.
4. AMO read-only snapshot `stable_runtime/deal_aware_amo_live_snapshot_20260513_v2/`.
5. Tallanto-контекст из `master_contacts_ru.csv` и, если безопасно, identity map/email handoff.
6. Email read-only handoff:
   `_external_handoffs/mail_archive_2026-05-12/regru_edu/full_all_mail_combined_20260513/customer_history_handoff_full_all_mail/`.

Telegram в v5 не импортировать в каноническую базу: надежная связь Telegram -> клиент пока слабая и идет отдельным пилотом.

Критерий "если безопасно" для email/identity:

- можно читать агрегированный email handoff и identity map только локально;
- можно переносить в timeline SQLite только candidate/customer refs, match class, счетчики, даты, source ids и статусы;
- нельзя переносить raw `.eml`, тексты писем, тексты вложений, OCR-текст, raw filenames, raw paths, пароли;
- ambiguous/missing/internal/service email не дают уверенную склейку и уходят в manual review/diagnostic.

## Выходные артефакты

- `customer_timeline.sqlite` - локальная SQLite-база под `product_data/customer_timeline/canonical_readonly_20260521_v5/`.
- `import_report.json` - агрегированный отчет импорта без сырых персональных значений.
- `coverage_report.json` и `coverage_report.md` - покрытие источников и причины ручной проверки.
- `source_manifest.json` - какие источники читались, только пути, размеры, хеши и счетчики.

## Acceptance criteria

- База создается вне `stable_runtime`.
- Все внешние источники открываются только для чтения.
- В отчеты не попадают телефоны, email, ФИО, сырые тексты писем/вложений.
- По каждому клиенту из `master_contacts_ru.csv` создается identity/event summary.
- Каждое событие содержит брендовый контекст: `foton`, `unpk` или `unknown`; если бренд неизвестен, он не повышается до уверенного.
- По звонкам создаются timeline events, но без запуска ASR/R+A.
- AMO/Tallanto добавляются только как snapshot-события/связки, без live-запросов.
- Email добавляется только агрегированно: количество писем, даты, match class, без raw mail/text.
- Отчет показывает:
  - сколько клиентов всего;
  - сколько имеют звонки;
  - сколько имеют Tallanto;
  - сколько имеют AMO;
  - сколько имеют email;
  - свежесть AMO snapshot и Tallanto snapshot;
  - сколько имеют конфликты или требуют ручной проверки;
  - почему `timeline_primary_read_enabled` пока нельзя включать.
- Текущие Telegram-тесты и обработка УНПК-чата не меняются.
- `git status --short` показывает, что SQLite/raw artifacts не подготовлены к коммиту; если нужны локальные данные, они должны быть ignored.

## Риски и решения

- Несколько клиентов на один телефон: не склеивать автоматически, ставить manual review.
- Один клиент на несколько телефонов/email: фиксировать как multi-contact risk и не делать уверенное объединение без источника, который уже это подтверждает.
- Один AMO contact/lead на несколько телефонов: не создавать уверенную сделку для всех клиентов; помечать как shared AMO risk и оставлять для ручной проверки.
- AMO/Tallanto конфликтуют: ставить manual review.
- Email ambiguous/missing: не использовать для уверенной склейки.
- Telegram identity слабая: в v1 не включать.
- Старая AMO snapshot дата: явно показать свежесть источника в отчете.
- Смешение Фотон/УНПК: хранить брендовый трек отдельно; не использовать единый timeline для брендовых ответов без active_brand.

## Фактическая сборка 2026-05-21

Пилотные каталоги `canonical_readonly_20260521_v1` - `v4` оставлены как частичные локальные ignored artifacts после остановок на реальных проблемах данных:

- AMO даты создания/обновления могли приходить в обратном порядке.
- Один AMO lead/contact мог быть связан с несколькими телефонами через разные источники.
- Разбор списков AMO ID должен поддерживать пробелы, запятые, `|` и `;`.

Финальный целевой каталог текущего этапа: `product_data/customer_timeline/canonical_readonly_20260521_v5/`.

## Следующий шаг после v5

После отчета решить:

1. Какие manual review причины являются реальными проблемами данных.
2. Какие являются слишком строгими правилами.
3. Можно ли включить `timeline_preview_enabled` на ограниченной выборке.
4. Что нужно для добавления Telegram как отдельного безопасного слоя.
