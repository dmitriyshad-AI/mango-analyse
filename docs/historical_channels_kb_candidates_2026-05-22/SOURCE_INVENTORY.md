# Source Inventory

Пакет собран read-only. Сырые письма, вложения, транскрипты, имена, телефоны, email, ссылки, username и внешние ID в этот отчет не переносились.

## Current KB

Источник: текущий employee/bot pack `kb_release_20260520_v6_3_team_answers`.

- Всего фактов: 833.
- Client-safe фактов: 472.
- Фотон client-safe: 222.
- УНПК client-safe: 250.
- Manager/internal facts: 361.
- Existing package status: `formal_pass=true`, `semantic_pass=true`.
- Ограничение: текущий контракт бота по-прежнему рассчитан на черновик для менеджера; клиентская автоотправка не подтверждена.

## Telegram

Источники: Telegram UNPK history audit, full-history enrichment, next analysis, gold-answer integration audit.

- Чатов: 892.
- Сообщений: 8261.
- Клиентских сообщений: 4068.
- Вопросов-источников во втором проходе: 2484.
- Очередь KB enrichment: 2321.
- P0-кейсы в источнике: 706.
- Reviewed regression tests: 80.
- Автономность в reviewed tests: 0.
- Gold phrase candidates после проверки: 20.

Вывод: Telegram полезен для реальных формулировок и тестов, но не для прямого импорта фактов.

## Email

Источники: mail archive runbook, email pipeline audit, full available mail handoff, question catalog summary.

- Писем в full available handoff: 49480.
- Внешних писем: 39355.
- Внутренних писем: 5446.
- Служебных писем: 4679.
- Strong unique привязок: 29711.
- Ambiguous: 6045.
- Missing: 3599.
- Email-вопросов в общем каталоге: 5957.
- Phone lift strong-unique: 214.
- Mango bridge resolved candidates with calls: 192.

Вывод: email уже полезен как read-only история и источник вопросов. Email-черновики пока заблокированы: нет устойчивого thread-слоя, проверки получателя и отдельного no-send контура.

## Calls

Источники: gold-candidates paid-proxy reports and audits.

- Paid-proxy candidates: 331.
- Top shortlist: 80.
- Метод: paid-proxy, не доказательство причинности покупки.
- Ограничение: один менеджер получил все совпадения в локальном snapshot; второй менеджер не получил paid-match в доступных данных.
- Следствие: стиль из paid-proxy нельзя утверждать как стиль всей команды без ручной балансировки источников.

Вывод: звонки полезны для стиля, структуры продажи и возражений, но не как источник клиентских фактов.

## Question Catalog

Источник: общий каталог вопросов.

- Всего question items: 9969.
- Вопросов из звонков: 2337.
- Вопросов из email: 5957.
- Вопросов из Telegram: 1675.
- Question classes: 2209.
- Dynamic fact classes: 1662.

Крупные классы: возвраты, оплата, чек/квитанция, летняя школа, расписание, онлайн-формат.

## Attachments

Источники: safe attachment inventory, text index, OCR reports.

- Вложений: 56951.
- Писем с вложениями: 26506.
- Общий размер вложений: около 27.1 ГБ.
- Text/PDF/OCR дают extracted/evidence слой, но raw OCR text может содержать персональные данные.
- Архивы, macro-файлы, большие и неизвестные типы остаются manual review.

Вывод: вложения можно использовать только как evidence-candidate с ручной проверкой, не как source of truth.
