> DONE 2026-07-23 05:23 | ветка codex/ai-employee-timeline | codex

> TAKE 2026-07-23 01:20 | ветка codex/ai-employee-timeline | codex

Ветка: codex/ai-employee-timeline
Зоны: src/mango_mvp/customer_timeline/, scripts/, tests/, docs/DECISIONS_LOG.md, tasks/, audits/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest --collect-only -q tests/test_customer_timeline_mail_link_enrich.py tests/test_wappi_history_import_to_timeline.py tests/test_customer_timeline_ingestion.py tests/test_customer_timeline_manager_dossier.py
Семантический-аудит: да

# Надёжная привязка почты и Wappi + owner review 30

Цель: повысить долю надёжно привязанных писем и Wappi-диалогов, используя уже
имеющиеся Tallanto/AMO/звонки/family/identity данные, без опасной массовой
склейки и без внешних write-операций.

## Порядок

1. По staging-сырью агрегатно разложить причины mail/Wappi unmatched, weak и
   ambiguous; ПДн в отчёт не выводить.
2. Проверить, какие сигналы уже существуют: нормализованные телефоны/email,
   Tallanto student/parent contacts, AMO contact/lead, family graph, Wappi
   pair/auto-pair, исходящий ответ менеджера и самоидентификация в диалоге.
3. Добавить только доказуемые правила identity resolution с provenance,
   confidence и veto для shared-family/conflict. Cross-brand не скрывает
   личность человека, но запрещает авторизацию брендового bot-контекста.
4. Сначала dry-run на staging, затем применить только к staging после
   агрегатного readback; prod Timeline не писать.
5. Повторить применение и доказать отсутствие дублей и перепривязок.
6. Собрать один XLSX на 30 клиентов для Дмитрия: отдельно manager-known и
   bot-safe-known, источники, свежесть, конфликты и пробелы.
7. Провести focused tests, SQLite quick_check и независимый смысловой аудит.

## Жёсткие границы

- AMO/Tallanto/CRM/Wappi/Telegram/email: только чтение, write/send = 0.
- Production Customer Timeline и stable_runtime не менять.
- Raw ПДн не коммитить; XLSX хранить только локально вне Git.
- Weak/ambiguous/conflicting match нельзя повышать до strong без двух
  независимых согласующихся доказательств или одного канонического strong ID.
- Ответ менеджера сам по себе не доказывает личность; это только дополнительный
  сигнал после подтверждения пары/контакта.
- Новые bot-safe chunks не открывать автоматически: сначала manager-only и
  отдельный semantic gate.

## Приёмка

- Есть таблица причин непривязки до/после без ПДн.
- Каждое новое strong-сопоставление имеет provenance и не нарушает семью,
  opt-out и существующий customer_id; брендовая авторизация хранится отдельно.
- Повторный run: created=0, remap=0, дублей=0.
- SQLite quick_check=ok, foreign_key_check=0.
- XLSX содержит 30 уникальных клиентов и явно показывает, что знает менеджер,
  что разрешено боту и чего система не знает.
- Тесты зелёные; production_ready не заявляется.

## СТОП

- Нужна запись во внешнюю систему или production Timeline.
- Рост привязки достигается только ослаблением ambiguous/cross-brand veto.
- Нельзя доказать provenance связи.
- SQLite quick_check не ok либо появляется перепривязка существующего события.
