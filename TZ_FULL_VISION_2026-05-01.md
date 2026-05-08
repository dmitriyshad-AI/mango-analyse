# ТЗ: Mango Analyse → автономный ИИ-сотрудник отдела продаж

Дата: 1 мая 2026
Версия: 1.0
Автор: Claude (наставник Дмитрия по ИИ)
Адресат: Дмитрий + Codex как исполнитель
Дополняет: `docs/ROADMAP_2026-03-07.md`, `docs/TZ_UPDATED_2026-03-09.md`, `docs/AMO_OAUTH_RECOVERY_PLAN_2026-04-14.md`, `docs/TALLANTO_API_NOTES_2026-04-13.md`

---

## 0. Как пользоваться этим документом

ТЗ разбито на **6 фаз** (Phase 1 — Phase 6). Каждая фаза имеет:
- **Цель** — одно предложение, что должно работать к концу фазы.
- **Состав** — какие компоненты включены.
- **Задачи** — детальные подзадачи с оценкой в днях focused work.
- **Архитектурные решения** — что нужно решить до старта фазы.
- **Acceptance Criteria** — проверяемые пункты, при которых фаза считается закрытой.
- **Verification Gate** — что должно быть верифицировано вручную до перехода к следующей фазе.

**Правила работы:**

1. Phase Gates непропускаемы. Если acceptance criteria не выполнены — следующая фаза не стартует. Не «пока запущу следующее, тут доделаю».
2. Codex используется для генерации кода и тестов. Решения по архитектуре, схемам данных, рискам и validation gates принимает человек (Дмитрий) до того, как давать задание Codex.
3. Каждая фаза заканчивается ручным запуском кода на реальных данных и сравнением результата с тем, что должно быть. Это часть acceptance criteria, не дополнительный шаг.
4. Тесты пишутся в той же фазе, что и код, а не «потом докроем». Юнит-тесты — обязательны для новой логики; интеграционные — для каждого сервиса; e2e — на каждом phase gate.
5. Шапочные оценки времени — focused work одного разработчика, ассистируемого Codex. Реальная скорость в part-time ≈ 1.6× от focused.

---

## 1. Бизнес-цели

**Главная цель:** превратить существующий batch-pipeline Mango Analyse в **автономного ИИ-сотрудника отдела продаж**, который без участия человека:

1. Получает звонок из Mango Office сразу после завершения.
2. Распознаёт диалог и анализирует его (это уже работает).
3. Записывает структурированный результат в карточку клиента в AMO CRM (и при необходимости в Tallanto).
4. Уведомляет ответственного менеджера с ключевыми точками разговора.
5. Накапливает базу знаний по тематикам и сезонным паттернам, выявляет, какие ответы менеджеров приводят к покупке.
6. Использует эту базу как источник для бота-консультанта в Telegram/MAX, который ведёт диалог с потенциальным клиентом до записи на консультацию, опираясь на проверенные паттерны и историю клиента из CRM.

**Бизнес-эффект:**
- Менеджеры получают саммари каждого звонка автоматически — экономия 10–15 минут на звонок.
- РОП видит топ возражений и эффективность по менеджерам в реальном времени.
- Клиент получает консультацию 24/7 через бота, конверсия посетитель→заявка растёт.
- Ответы бота построены на статистически проверенных паттернах, а не на догадках.

**Стратегический эффект:**
- Mango Analyse становится первым внедрённым ИИ-кейсом компании, на котором можно показывать клиентам B2B-продукта «Внедрение ИИ в организации».

---

## 2. Нефункциональные требования

| Требование | Целевое значение |
|---|---|
| Latency capture → notify | ≤ 5 минут после завершения звонка |
| Pipeline throughput | ≥ 50 звонков/час в нормальном режиме, ≥ 200/час в catchup |
| Uptime live-сервиса | ≥ 99% (допустимо ≤ 7 часов простоя в месяц) |
| Sync errors (доля fail) | ≤ 1% при стабильной работе AMO API |
| ASR-качество (на чистом аудио) | WER ≤ 12% |
| Анализ-качество (поля заполнены корректно) | ≥ 85% звонков без quality_flags |
| Recovery time после падения | ≤ 15 минут (auto-restart) |
| Storage retention для аудио | ≥ 90 дней |
| Storage retention для transcripts | ≥ 365 дней |

---

## 3. Существующие активы

### 3.1 Слой Analyze — почти готов

В `src/mango_mvp/`:
- `cli.py` — CLI команды `ingest`, `transcribe`, `resolve`, `analyze`, `worker`, `migrate-analysis-schema`.
- `db.py`, `models.py`, `config.py` — SQLite по умолчанию, БД ~89 МБ с реальными данными.
- `clients/ollama.py`, `clients/amocrm.py` — провайдеры.
- `utils/audio.py`, `utils/phone.py`, `utils/filename_repair.py` — вспомогательные функции.
- Dual-ASR (MLX Whisper + GigaAM) с merge.
- Schema v2: `history_short`, `crm_blocks`, `evidence`, `quality_flags`.
- Stable runtime + GUI launcher.
- Retry/dead-letter/worker.
- 22 теста в `tests/`.

### 3.2 Слой Sync — частично готов

В `src/mango_mvp/amocrm_runtime/`:
- `auth.py` — OAuth (см. AMO_OAUTH_RECOVERY_PLAN).
- `db.py`, `models.py`, `schemas.py`, `config.py`.
- `amo_integration.py`, `deals.py`, `leads_extension.py`, `deal_dossier.py`, `deal_llm.py`.
- `tallanto_api.py`, `tallanto_context.py`, `tallanto_export.py`, `tallanto_matching.py`, `tallanto_premature_close.py`, `tallanto_deal_ranking.py`.
- `phone_context.py` — контекст по номеру телефона из обеих CRM.
- `main.py`, `__main__.py` — entry points.
- **Sync работает в dry-run по умолчанию.** Production-mode заблокирован задачами P0.1 (карта полей) и P0.3 (staged rollout) из текущего roadmap.

### 3.3 Слой Capture — отсутствует

Текущий ingest батчевый: ручной экспорт из Mango Office → файлы в папке → `mango-mvp ingest`. Live auto-pull нужно построить.

### 3.4 Слой Insight — отсутствует

Топик-классификации, привязки к outcome конверсии, паттерн-экстракции и базы идеальных ответов нет.

### 3.5 Слой Bot — отсутствует в этом проекте

Существует отдельный `/Foton/Projects/TG_sale_bot/` (заморожен, минимальный функционал). Будет реанимирован в рамках Phase 4a.

---

## 4. Целевая архитектура

### 4.1 Сервисы и их обязанности

```
┌─────────────────────────────────────────────────────────────┐
│  EXTERNAL                                                    │
│  Mango Office PBX  ↔  AMO CRM  ↔  Tallanto CRM              │
└──────┬───────────────────┬───────────────┬──────────────────┘
       │ webhook/poll      │ REST API      │ REST API
       ▼                   ▼               ▼
┌──────────────────────────────────────────────────────────────┐
│  CAPTURE SERVICE  (новое, Phase 2)                          │
│  • FastAPI webhook receiver на /mango/call_end               │
│  • Polling fallback на recent calls API                      │
│  • Скачивание аудио, проверка идемпотентности                │
│  • Запись метаданных в БД, файлов в storage                  │
└──────────────────────┬───────────────────────────────────────┘
                       │ new ingest record
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  ANALYZE PIPELINE  (существует, Phase 1 — productionize)    │
│  ingest → transcribe (dual-ASR) → resolve → analyze (v2)    │
│  worker daemon mode, auto-trigger на новых записях           │
└──────────────────────┬───────────────────────────────────────┘
                       │ analysis JSON + quality flags
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  SYNC SERVICE  (существует, Phase 1 — production-mode)      │
│  • Field mapping crm_blocks → AMO custom fields              │
│  • Quality gate (sync только без критичных flags)            │
│  • Staged rollout                                            │
│  • Tallanto writeback (Phase 3)                              │
│  • Error reporting                                           │
└──────┬───────────────────────────────────────────────────────┘
       │
       ├──► AMO/Tallanto: notes, fields, tasks
       │
       └──► EVENT: analysis_synced
                  │
                  ▼
┌──────────────────────────────────────────────────────────────┐
│  NOTIFICATION SERVICE  (новое, Phase 2)                     │
│  • TG-бот для сотрудников                                    │
│  • Per-owner routing                                         │
│  • Формат: ключевые точки звонка + ссылка в AMO              │
└──────────────────────────────────────────────────────────────┘

──────────── ПОСЛЕ ФАЗ 1-3: AUTONOMOUS PIPELINE РАБОТАЕТ ────────────

┌──────────────────────────────────────────────────────────────┐
│  TOPIC CLASSIFIER  (новое, Phase 4b)                         │
│  • Таксономия тем (программы, цены, расписание, возражения) │
│  • LLM-based классификатор + few-shot                       │
└──────────────────────┬───────────────────────────────────────┘
                       │ topic per call
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  OUTCOME LINKER  (новое, Phase 4b)                           │
│  • Чтение AMO/Tallanto deals по phone+time                   │
│  • Матчинг звонков с outcome (купил/не купил/в работе)       │
└──────────────────────┬───────────────────────────────────────┘
                       │ topic + outcome per call
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  PATTERN EXTRACTOR  (новое, Phase 5)                         │
│  • Группировка: тема × ответ менеджера → outcome             │
│  • Статистические/LLM-инсайты                                │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  IDEAL ANSWERS KB  (новое, Phase 5)                          │
│  • Структурированный markdown/JSON                           │
│  • Ручная валидация с РОП и Анной                            │
│  • Update mechanism                                          │
└──────────────────────┬───────────────────────────────────────┘
                       │ feeds KB to bot
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  TG/MAX SALES BOT  (новое, Phase 4a + 6)                     │
│  • Conversation engine (LLM)                                 │
│  • KB lookup                                                 │
│  • CRM context lookup по номеру (phone_context.py)           │
│  • Handoff менеджеру                                         │
│  • Audit log                                                 │
└──────────────────────────────────────────────────────────────┘
```

### 4.2 Tech stack

| Слой | Решение | Обоснование |
|---|---|---|
| Язык | Python 3.12 | Существующий |
| Web/API | FastAPI | Современный, async-ready, OpenAPI бесплатно |
| БД (operational) | PostgreSQL | SQLite не подходит для multi-process production. Migration plan — отдельная подзадача в Phase 1. |
| БД (analytical, Phase 4-5) | DuckDB | Отдельный analytical store для outcome linking и pattern extraction |
| Очередь | Redis Streams | Простота, не нужен Kafka для этого объёма |
| Storage аудио | S3-совместимый (MinIO локально или Selectel S3 в production) | |
| ASR | MLX Whisper + GigaAM (existing) | |
| LLM | OpenAI API (gpt-4 или gpt-4-turbo) для analyze, claude-haiku для классификации | Существующий ollama-провайдер сохраняется как fallback |
| Bot framework | aiogram 3.x | TG_sale_bot уже на aiogram, продолжаем |
| Деплой | Docker Compose на VPS (Phase 1-3), миграция на k8s — после | M4 Max не подходит для live |
| CI/CD | Существующий GitHub Actions | Продолжаем `.github/workflows/ci.yml` |
| Мониторинг | Prometheus + Grafana, Sentry для ошибок | Phase 1 — базовые health checks, Phase 6 — полный observability |
| Secrets | .env + sops (gitops-friendly) или Doppler | |

### 4.3 Структура репозитория после фаз 1-6

```
mango-analyse/
├── src/
│   ├── mango_mvp/                       # существующее, продолжаем
│   │   ├── cli.py
│   │   ├── db.py, models.py, config.py
│   │   ├── amocrm_runtime/              # уже есть
│   │   ├── clients/
│   │   └── utils/
│   ├── capture_service/                 # NEW (Phase 2)
│   │   ├── webhook.py
│   │   ├── poller.py
│   │   ├── mango_office_client.py
│   │   └── ingest_bridge.py
│   ├── analyze_daemon/                  # NEW (Phase 1)
│   │   ├── watcher.py
│   │   └── worker_loop.py
│   ├── sync_service/                    # NEW (Phase 1, refactor of amocrm_runtime)
│   │   ├── field_mapper.py
│   │   ├── staged_rollout.py
│   │   ├── quality_gate.py
│   │   └── tallanto_writer.py           # Phase 3
│   ├── notification_service/            # NEW (Phase 2)
│   │   ├── tg_notifier.py
│   │   └── routing.py
│   ├── insight_layer/                   # NEW (Phase 4b-5)
│   │   ├── topic_classifier.py
│   │   ├── outcome_linker.py
│   │   ├── pattern_extractor.py
│   │   └── kb_generator.py
│   └── sales_bot/                       # NEW (Phase 4a-6)
│       ├── bot.py
│       ├── conversation.py
│       ├── kb_lookup.py
│       └── crm_context.py
├── deploy/
│   ├── docker-compose.yml               # NEW (Phase 1)
│   ├── Dockerfile.capture
│   ├── Dockerfile.analyze
│   ├── Dockerfile.sync
│   ├── Dockerfile.notify
│   └── server_rebuild.sh                # существующее
├── docs/                                # существующее, дополняем
├── tests/
│   ├── test_*.py                        # 22 существующих
│   ├── test_capture_*.py                # NEW
│   ├── test_sync_*.py                   # NEW
│   ├── test_notify_*.py                 # NEW
│   ├── test_insight_*.py                # NEW
│   └── test_bot_*.py                    # NEW
└── alembic/                             # NEW (Phase 1, для PG migrations)
    └── versions/
```

---

## 5. Phased Delivery Plan

### Общая карта фаз

| Фаза | Период (focused work) | Цель |
|---|---|---|
| **Phase 1** | 7–10 дней | Production-ready Sync существующего pipeline (без live capture) |
| **Phase 2** | 8–12 дней | Live capture + notification, конвейер end-to-end в shadow |
| **Phase 3** | 5–8 дней | Стабилизация в production + Tallanto writeback |
| **Phase 4a** | 14–20 дней | Sales bot foundation на ручной KB, в shadow |
| **Phase 4b** | 12–18 дней (parallel с 4a) | Topic classifier + outcome linker — фундамент insight |
| **Phase 5** | 15–22 дня | Pattern extraction → Ideal Answers KB → enriched bot |
| **Phase 6** | 7–10 дней | Production polish: monitoring, alerting, документация, handoff |
| **ИТОГО** | **68–100 дней focused** (~14–20 недель focused, ~5–8 месяцев part-time 4ч/день) |

---

### PHASE 1 — Production-ready Sync существующего pipeline

**Цель:** AMO sync работает в production-режиме (не dry-run) на тестовом сегменте 300+ звонков с ошибками ≤ 1% и quality gate включённым.

**Состав компонентов:** B (productionize Analyze частично) + C (CRM sync prod-mode).

**Ключевая зависимость:** P0.1 и P0.3 из существующего ROADMAP_2026-03-07.md.

#### Задачи Phase 1

**1.1. Миграция БД с SQLite на PostgreSQL (2–3 дня)**

Зачем: SQLite заблокирует параллельные процессы capture и analyze. Production требует мульти-процесс.

Подзадачи:
- Установить Alembic, создать первичную миграцию из существующих моделей.
- Настроить connection pool через SQLAlchemy.
- Скрипт миграции данных из текущего `mango_mvp.db` (89 МБ) в PostgreSQL.
- Параметризовать через `DATABASE_URL` в `.env`.
- Тесты: смоук-тест на копии production-данных.

Acceptance:
- Все существующие тесты проходят на PostgreSQL.
- `mango-mvp stats` даёт идентичный результат на SQLite и PostgreSQL.
- Миграционный скрипт идемпотентен (повторный запуск не ломает данные).

**1.2. Финальная карта полей AMO (P0.1 из roadmap) (2–3 дня)**

Подзадачи:
- Выгрузить через AMO API список всех custom fields на сущности «контакт» и «сделка».
- Сопоставить каждый блок `crm_blocks` (people, contacts, student, interests, commercial, objections, next_step, lead_priority) с конкретным `field_id`.
- Зафиксировать mapping в `src/sync_service/field_mapper.py` как структуру с версионированием.
- Документировать в `docs/AMO_FIELD_MAPPING.md` — какой блок куда едет, какие enum-значения, валидация.
- Покрыть тестами: для каждого блока есть фикстура-вход и ожидаемый payload в AMO формате.

Acceptance:
- Mapping покрывает 100% полей schema v2.
- Юнит-тесты field mapping проходят.
- На 50 случайных звонках dry-run выдаёт корректные payload без warnings.

**1.3. Quality gate (1–2 дня)**

Подзадачи:
- Определить категории `quality_flags`, при которых sync блокируется (например, `low_confidence_transcript`, `manager_role_uncertain`, `no_phone_extracted`).
- Реализовать `src/sync_service/quality_gate.py` с правилами.
- Звонки, не прошедшие gate, складываются в `manual_review_queue` с указанием причины.
- CLI: `mango-mvp review-queue --list`, `--mark-resolved`.

Acceptance:
- На тестовом сегменте 500 звонков gate отсекает X% звонков (X замеряется и фиксируется в отчёте).
- Ручная проверка 20 случайных отсечённых звонков подтверждает: ≥ 80% действительно не должны были синкаться (true positive rate).
- Ручная проверка 20 случайных пропущенных звонков подтверждает: ≥ 80% действительно качественные (true negative rate).

**1.4. Staged rollout (P0.3) (2–3 дня)**

Подзадачи:
- Реализовать `src/sync_service/staged_rollout.py` с уровнями: `dry-run`, `staged-50`, `staged-300`, `full`.
- На уровне `staged-50`: sync только 50 случайных звонков, выводит подробный отчёт.
- Каждое повышение уровня требует явного флага — никакого `--auto-promote`.
- Логи всех sync-попыток с timestamp, payload, response, status.
- CLI: `mango-mvp sync --stage staged-50 --report-out report.json`.

Acceptance:
- Уровень `staged-50` отрабатывает на 50 реальных карточках без ошибок API.
- Каждая записанная карточка проверена вручную (5 минут на карточку × 50 = ~4 часа верификации) — данные корректны.
- Перед `staged-300` есть явный manual gate.

**1.5. Productionize analyze pipeline (B компонент) (2–3 дня)**

Подзадачи:
- Daemon-mode: file watcher на `ingest_dir`, автозапуск pipeline на новых файлах.
- Конкурентность: ограничить параллелизм по CPU/RAM (профили из P1.1 текущего roadmap).
- Health endpoint: `GET /health` возвращает статус сервиса и очередей.
- Graceful shutdown: SIGTERM завершает текущие задачи, не теряя состояние.
- Restart policy через systemd или Docker Compose `restart: always`.

Acceptance:
- Сервис работает 48 часов без перезапуска на тестовых данных.
- 100 файлов, положенных в ingest_dir в течение часа, обрабатываются без потерь.
- SIGTERM не теряет в работе ни одного звонка.

#### Архитектурные решения, которые принять до старта Phase 1

1. **PostgreSQL: managed service или self-hosted?** Рекомендация: managed (Selectel Cloud DB, Yandex Cloud). Проще и дешевле, чем self-hosted на VPS.
2. **Где хостинг компонентов?** Рекомендация: VPS Selectel или TimeWeb с 4 vCPU / 8 GB RAM на старт. Можно начать с одного, разделить по сервисам в Phase 6.
3. **Управление секретами.** Рекомендация: для Phase 1-3 хватает `.env` + git-ignored файла, передаваемого через scp при деплое. Для Phase 4-6 — sops или Doppler.
4. **Куда писать логи.** Рекомендация: stdout → docker logs → Loki или Grafana Cloud (free tier). Sentry для exceptions.

#### Verification Gate Phase 1 → Phase 2

Перед стартом Phase 2 проверить:

- [ ] Sync прогнан в `staged-300` режиме, ≥ 99% успехов.
- [ ] 30 случайных карточек после sync проверены вручную, данные совпадают с transcript ≥ 90% случаев.
- [ ] Daemon работает 7 дней непрерывно без manual restart.
- [ ] Все тесты `pytest tests/` проходят.
- [ ] AMO_FIELD_MAPPING.md написан и согласован с пониманием менеджеров продаж (5 минут разговора с РОП — «вот сюда мы пишем такое-то поле»).
- [ ] Quality gate true-positive и true-negative rates ≥ 80% на ручной выборке.

**Если хотя бы один пункт не закрыт — Phase 2 не стартует.**

---

### PHASE 2 — Live capture + notification

**Цель:** Звонок завершается в Mango Office → саммари появляется в карточке AMO + сотрудник получает TG-уведомление за ≤ 5 минут. Работает в shadow-режиме (всё пишется в AMO, но менеджер вручную подтверждает корректность через TG-кнопки) 7 дней.

**Состав компонентов:** A (capture) + D (notification) + интеграция с Phase 1.

#### Задачи Phase 2

**2.1. Изучение Mango Office API (1 день)**

- Прочитать актуальную документацию VPBX API: https://www.mango-office.ru/support/programmistam/api/
- Получить ключи API в личном кабинете Mango Office.
- Тестовые запросы через Postman: `GET /vpbx/calls`, `GET /vpbx/recordings/{id}`.
- Изучить webhook-конфигурацию: какие события доступны, как настроить URL.
- Зафиксировать в `docs/MANGO_OFFICE_API_NOTES.md`: rate limits, auth, формат payload.

Acceptance: документ готов, ручные запросы работают.

**2.2. Capture Service: webhook receiver (3–4 дня)**

Подзадачи:
- FastAPI приложение `src/capture_service/webhook.py` с эндпоинтом `POST /mango/call_end`.
- Валидация подписи webhook (Mango даёт HMAC).
- Парсинг payload, извлечение `call_id`, `phone`, `manager_extension`, `recording_url`.
- Идемпотентность: проверка по `call_id` перед обработкой (в БД).
- Скачивание аудио через `recording_url` с retry.
- Запись метаданных в PostgreSQL, файла в S3-совместимый storage.
- Триггер на ingest pipeline (через Redis Stream или прямой DB record + watcher).
- Тесты: моковый webhook payload, проверка полного цикла до записи в БД.

Acceptance:
- Эндпоинт принимает реальный webhook от Mango Office (тестовый звонок) и обрабатывает за ≤ 30 секунд.
- 10 повторных webhook на один call_id создают только одну запись.
- Невалидная подпись отбрасывается с 401.

**2.3. Capture Service: polling fallback (2–3 дня)**

Зачем: webhook может теряться (network issues, Mango downtime). Polling страхует.

Подзадачи:
- Cron-задача (или daemon с sleep): каждые 15 минут запрашивает `GET /vpbx/calls?from=...&to=...`.
- Сравнение с локальной БД: новые `call_id` идут в обработку.
- Catch-up окно: при первом запуске после простоя — догон до 24 часов.
- CLI: `mango-mvp capture poll --since=2026-05-04T00:00:00`.

Acceptance:
- Если webhook отключить, polling догоняет все звонки за окно ≤ 30 минут.
- При длительном простое (например, 6 часов) catch-up не дублирует записи.

**2.4. Storage аудио (1–2 дня)**

Подзадачи:
- Выбор: MinIO в Docker Compose локально (Phase 2) или сразу Selectel S3.
- Naming convention: `mango/{YYYY}/{MM}/{DD}/{call_id}.mp3`.
- Lifecycle policy: ≥ 90 дней retention.
- Доступ только из приватной сети (capture, analyze читают; внешний доступ закрыт).

Acceptance:
- 100 тестовых файлов сохранены и читаются.
- Permissions проверены: вне whitelist S3 не отдаёт файл.

**2.5. Триггер на pipeline (1 день)**

Подзадачи:
- Когда capture закончил скачивание и записал метаданные — триггер на analyze daemon (Phase 1.5).
- Реализация: Redis pub/sub или прямой select в БД с poll интервалом 5 секунд.

Acceptance: новый файл в storage → analyze начинается за ≤ 30 секунд.

**2.6. Notification Service (3–4 дня)**

Подзадачи:
- Создать отдельный TG-бот `@foton_call_notifier_bot` (отдельный от продающего).
- `src/notification_service/tg_notifier.py`: при событии `analysis_synced` → отправляет сообщение ответственному менеджеру.
- Per-owner routing: маппинг `manager_extension → telegram_user_id` в БД (таблица `notification_routing`).
- Формат сообщения: 5–7 строк — кто звонил, тема, ключевые точки, next step, ссылка на карточку AMO. Inline-кнопки «✓ верно / ✗ ошибка / ⚠ требует правки».
- Сохранение feedback от кнопок в БД для метрик качества.

Acceptance:
- Один тестовый звонок проходит full pipeline: webhook → analyze → sync → notify за ≤ 5 минут.
- Менеджер получает корректно отформатированное сообщение и может нажать кнопку — feedback сохраняется.

**2.7. Shadow режим: 7 дней наблюдения (вшито в Definition of Done)**

В Phase 2 sync уже идёт в production-mode из Phase 1. «Shadow» здесь — это **shadow тебя как валидатора**: каждое сообщение notification ты лично проверяешь, нажимаешь кнопку, делаешь итерации. 50+ звонков подряд должны быть с feedback ≥ 80% ✓.

#### Архитектурные решения для Phase 2

1. **Webhook URL — публичный домен.** Нужен субдомен `capture.foton.example.com` с TLS. Caddy или nginx + Let's Encrypt.
2. **HMAC signing key для Mango.** Сгенерировать, прописать в `.env`.
3. **Manager extension → telegram user id mapping.** Таблица создаётся вручную для команды (8–15 записей).

#### Verification Gate Phase 2 → Phase 3

- [ ] Capture service работает 7 дней непрерывно.
- [ ] За 7 дней обработано ≥ 100 реальных звонков live.
- [ ] Latency capture → notify ≤ 5 минут на 95-м перцентиле.
- [ ] 50+ notification-сообщений получены менеджерами, нажато ≥ 30 кнопок ✓.
- [ ] Polling fallback протестирован: webhook отключали, polling догнал.
- [ ] Тесты `tests/test_capture_*.py` и `tests/test_notify_*.py` написаны и проходят.

---

### PHASE 3 — Стабилизация + Tallanto writeback

**Цель:** Live конвейер работает 14 дней автономно, объём поднят со shadow-сегмента до полного потока всех звонков. Tallanto writeback включён.

**Состав:** E (Tallanto writeback) + стабилизация всего, что построено.

#### Задачи Phase 3

**3.1. Tallanto writeback policy (1–2 дня)**

Подзадачи:
- Решить какие поля писать в Tallanto (некоторые дублируются с AMO, нужно избежать конфликта).
- Конфликт-резолюция: AMO как primary, Tallanto обновляется только если соответствующее поле в AMO «свежее» этого звонка.
- `src/sync_service/tallanto_writer.py` — реализация.
- Документировать в `docs/TALLANTO_WRITEBACK_POLICY.md`.

Acceptance:
- 30 тестовых звонков syncнуты в обе CRM, конфликтов нет, данные консистентны.

**3.2. Расширение объёма shadow → full (2–3 дня)**

Подзадачи:
- Снять ограничение «только сегмент менеджеров X, Y» — пускать в обработку все звонки.
- Включить notification всем сотрудникам отдела продаж по mapping.
- Мониторить throughput: если поток превышает обработку — нужен либо больший instance, либо ограничение параллелизма.

Acceptance:
- 7 дней full-flow без manual intervention.
- Нет накопления очереди (ingestion rate ≤ processing rate).

**3.3. Error handling polish (2–3 дня)**

Подзадачи:
- Любая ошибка API (AMO 401/429/500) → retry с exponential backoff.
- При исчерпании retry — звонок в `dead_letter_queue` с причиной.
- Daily report: сколько звонков в DLQ, по каким причинам, top 5 типов ошибок.
- Alerts в TG-канал @foton_ops_alerts при resh-критичных событиях (auth fail > 5 минут, queue lag > 30 минут).

Acceptance:
- Симуляция AMO 500: pipeline восстанавливается без потерь после возврата API.
- Симуляция OAuth token истёк: автоматическое refresh через AMO_OAUTH_RECOVERY_PLAN.

**3.4. Documentation для команды (1 день)**

Подзадачи:
- `docs/RUNBOOK.md`: что делать если notification не пришло, если карточка не обновилась, как посмотреть DLQ.
- Короткое видео для менеджеров «как читать notification, что значат кнопки» (5 минут).

#### Verification Gate Phase 3 → Phase 4

- [ ] Live конвейер 14 дней без manual restart.
- [ ] DLQ ≤ 1% от объёма.
- [ ] РОП и 2 менеджера подтверждают, что notification реально полезны и используются в работе (короткое интервью).
- [ ] Документация написана.

**К этой точке у тебя есть полностью работающий «Mango Analyse как автономный сервис».** Это уже самостоятельный кейс, который можно показывать клиентам B2B-продукта. Phase 4-6 — расширение до полной визии с ботом и insight, но Phase 1-3 уже даёт продукт.

---

### PHASE 4a — Sales bot foundation

**Цель:** TG-бот летней школы Фотона ведёт диалог с потенциальным клиентом, опираясь на ручную KB. В shadow-режиме (твоё подтверждение каждого ответа) 7+ дней, потом частичный auto-mode.

**Состав:** J (bot) + K (CRM context lookup, минимально).

**Может идти параллельно с Phase 4b.**

#### Задачи Phase 4a

**4a.1. Реанимация TG_sale_bot (2–3 дня)**

Подзадачи:
- Перенос актуального кода `/Foton/Projects/TG_sale_bot/` в `src/sales_bot/`.
- Аудит существующих handlers, выкинуть нерабочее.
- Обновить aiogram до 3.x если нужно.
- Запустить локально: бот отвечает на `/start`.

**4a.2. Conversation Engine (4–6 дней)**

Подзадачи:
- `src/sales_bot/conversation.py` — состояние диалога в Redis (state machine + LLM).
- System prompt с задачей бота: консультант летней школы Фотона.
- Few-shot examples диалогов.
- Rate limiting per user.
- Anti-jailbreak: фильтр промпт-инъекций.

Acceptance:
- 20 тестовых диалогов с тобой в роли клиента — бот отвечает по делу, не путает программы и цены.

**4a.3. Manual Knowledge Base (2–3 дня)**

Подзадачи:
- Структура `kb/programs/`, `kb/prices/`, `kb/schedule/`, `kb/faq/`.
- Заполнение от руки + с помощью Codex (но финальная проверка — твоя): программы летней школы 2026, цены, расписание, FAQ из 30 вопросов.
- `src/sales_bot/kb_lookup.py` — vector embedding или просто keyword search для начала.

Acceptance:
- На 30 тестовых вопросах бот находит правильный фрагмент KB ≥ 80%.

**4a.4. CRM context lookup минимальный (2–3 дня)**

Подзадачи:
- При первом сообщении бот спрашивает имя/телефон.
- По телефону через `phone_context.py` (уже существует) — запрос в AMO + Tallanto.
- Если клиент уже есть: бот учитывает историю в первой реплике («вижу, вы интересовались курсом X в прошлом году»).
- Если нет — стандартный диалог.

Acceptance:
- Тест с твоим личным номером (ты есть в Tallanto): бот корректно подтягивает контекст.
- Тест с новым номером: бот не падает, ведёт диалог как с новым клиентом.

**4a.5. Handoff менеджеру (2–3 дня)**

Подзадачи:
- Триггеры на handoff: клиент готов записаться, бот не знает ответа на 2 подряд вопроса, клиент явно просит человека, обнаружен сложный кейс (жалоба, эмоциональный разговор).
- Передача: бот сообщает клиенту «передаю менеджеру», менеджер получает уведомление с историей диалога.
- Менеджер продолжает в том же чате (бот уходит).

Acceptance:
- 10 тестовых диалогов: 5 заканчиваются записью на консультацию через бота, 5 — handoff. Все обработаны корректно.

**4a.6. Shadow-режим: 7 дней (1 неделя)**

Подзадачи:
- Бот развёрнут на сайте летней школы.
- **Каждый ответ бота сначала идёт тебе в TG**, ты подтверждаешь / правишь / отвергаешь — только потом отправляется клиенту.
- 50+ предложений ботом, валидация ≥ 80% корректных.
- После 7 дней — частичный auto-mode (только на whitelist тем).

Acceptance:
- 50+ ответов проверены, 80%+ корректны.
- Зафиксирован вердикт: какие темы переводим в auto, какие остаются в shadow.

#### Verification Gate Phase 4a → Phase 6

- [ ] Бот в auto-mode 7 дней без эскалаций.
- [ ] ≥ 5 реальных клиентов записались на консультацию через бота.
- [ ] Логи разговоров просмотрены, явных провалов нет.

---

### PHASE 4b — Insight foundation (parallel с 4a)

**Цель:** Все накопленные звонки разнесены по темам, каждый звонок привязан к outcome конверсии.

**Состав:** F (topic classifier) + G (outcome linker).

#### Задачи Phase 4b

**4b.1. Топик-таксономия (2 дня)**

Подзадачи:
- Совместная сессия с РОП (1 час): какие темы клиенты обсуждают?
- Финальный список 12–18 тем (программы, цены, расписание, преподаватели, формат, возражения по цене, возражения по нагрузке, возражения по результату, технические вопросы, регистрация, оплата, повторные клиенты, рекомендации, отзывы, и т.д.).
- Документ `docs/TOPIC_TAXONOMY.md` с описанием и примерами каждой темы.

**4b.2. Topic Classifier (3–4 дня)**

Подзадачи:
- `src/insight_layer/topic_classifier.py` — LLM-classifier с few-shot examples.
- Один звонок может иметь несколько тем (multi-label).
- Валидация на 100 ручно размеченных звонках: precision и recall по каждой теме.
- Iteration на промпте до достижения ≥ 80% accuracy на macro-F1.

Acceptance:
- На 100 валидационных звонках macro-F1 ≥ 0.80.
- Классификатор прогнан на всей исторической базе (89 МБ) — каждый звонок имеет topics.

**4b.3. Outcome Linker (4–5 дней)**

Подзадачи:
- Чтение AMO/Tallanto deals: `phone`, `created_at`, `status`, `closed_at`, `amount`.
- Алгоритм матчинга: для каждого звонка ищем сделку, у которой `phone` совпадает и `created_at` в окне ±7 дней (настраиваемо).
- Edge cases:
  - Один номер → несколько сделок: берём ближайшую по времени.
  - Сделка без явного outcome (в работе): помечаем `pending`.
  - Звонок без сделки: помечаем `no_deal_found`.
- Outcome categories: `won`, `lost`, `pending`, `no_deal_found`.
- Документ `docs/OUTCOME_LINKING_ALGORITHM.md` с правилами.

Acceptance:
- На 200 ручно проверенных линках точность ≥ 85% (правильно сматчили).
- Звонки с `won` имеют известную сумму сделки.

**4b.4. Analytical store (DuckDB) (2–3 дня)**

Подзадачи:
- Развернуть DuckDB рядом с PostgreSQL.
- ETL: каждый день переливать `calls + topics + outcomes` в DuckDB для аналитики.
- Базовые view: «звонки по теме за период», «конверсия по теме», «менеджер × тема × конверсия».

Acceptance:
- DuckDB запросы работают, базовые view возвращают разумные числа.

#### Verification Gate Phase 4b → Phase 5

- [ ] Topic classifier macro-F1 ≥ 0.80.
- [ ] Outcome linker точность ≥ 85%.
- [ ] DuckDB analytical store работает.
- [ ] Аудит CRM-данных: насколько чисто менеджеры выставляют статусы сделок? Если хаос — Phase 5 буксует. Возможно, нужен отдельный мини-проект «выровнять CRM-дисциплину» до Phase 5.

---

### PHASE 5 — Pattern extraction → Ideal Answers KB

**Цель:** Готов файл «идеальных ответов» — для каждой темы видны паттерны успешных и неуспешных ответов, есть валидированные drafts. Бот переключён на enriched KB.

**Состав:** H (pattern extractor) + I (ideal answers KB) + интеграция с ботом.

#### Задачи Phase 5

**5.1. Pattern Extractor (5–8 дней)**

Подзадачи:
- Для каждой темы (из 4b.1):
  - Группировать звонки по теме.
  - Для каждого звонка выделить «ответ менеджера на ключевой вопрос» (через LLM extraction).
  - Сравнить ответы won-звонков vs lost-звонков.
  - LLM-аналитика: «какие отличия в формулировках/аргументах ведут к won?»
- Output: `patterns/{topic}/successful_patterns.md` и `patterns/{topic}/failed_patterns.md`.
- Это исследовательский компонент, требует итераций промптов.

Acceptance:
- Для топ-5 тем (по объёму) есть выделенные паттерны.
- Каждый паттерн имеет 3+ примера и явное отличие от противоположного.

**5.2. Ideal Answers Generation (3–5 дней)**

Подзадачи:
- На основе successful_patterns LLM генерирует drafts «идеальных ответов» по каждой теме.
- Структура: тема → типичные вопросы клиента → рекомендованные формулировки ответа → запрещённые формулировки → доп. контекст.
- Output: `kb/ideal_answers/{topic}.md` — структурированный markdown.

**5.3. Validation с РОП и Анной (3–5 дней — их время)**

Подзадачи:
- Спланировать 2 встречи по 2 часа каждая с РОП и Анной.
- На встречах: проходим топ-5 тем, валидируем drafts. Они говорят «да, так работает» или «нет, в реальности по-другому».
- Правки в `kb/ideal_answers/`.

Это блокирующее звено. Если РОП и Анна не вовлечены — Phase 5 не закроется.

**5.4. Integration с ботом (2–3 дня)**

Подзадачи:
- `src/sales_bot/kb_lookup.py` расширить: vector search + topic-aware retrieval.
- На вопрос клиента: классификатор определяет тему → берёт `kb/ideal_answers/{topic}.md` → формирует ответ.
- Старые ручные KB остаются как fallback для тем, по которым ideal_answers ещё не готовы.

Acceptance:
- На 50 тестовых диалогах бот с enriched KB даёт ответы, которые РОП оценивает выше, чем бот с ручной KB (subjective А/Б тест).

**5.5. Continuous update (2 дня)**

Подзадачи:
- Cron-задача (раз в неделю): пересобирать паттерны и ideal_answers с новыми данными.
- Diff-отчёт: что изменилось в паттернах за неделю.
- Manual approval: изменения в KB деплоятся только после твоего ✓.

#### Verification Gate Phase 5 → Phase 6

- [ ] Ideal answers KB покрывает топ-10 тем.
- [ ] РОП и Анна валидировали ≥ 80% drafts.
- [ ] А/Б тест бота: enriched > manual KB (по оценке РОП).

---

### PHASE 6 — Production polish & handoff

**Цель:** Система переведена в полноценный production: monitoring, alerting, документация, готовность к расширению на другие компании (B2B-продукт).

#### Задачи Phase 6

**6.1. Monitoring stack (2–3 дня)**

Подзадачи:
- Prometheus exporters для всех сервисов: capture, analyze, sync, notify, bot.
- Grafana дашборды: throughput по часам, latency распределение, error rates, queue depth.
- Алерты: Sentry для exceptions, PagerDuty или TG-бот для critical alerts.

**6.2. Operational documentation (2 дня)**

Подзадачи:
- `docs/RUNBOOK.md` — расширенный с Phase 3.
- `docs/ARCHITECTURE.md` — обновлённый с финальной архитектурой.
- `docs/API_REFERENCE.md` — описание внутренних API.
- `docs/DEPLOYMENT.md` — как развернуть с нуля.

**6.3. Тестовое покрытие polish (2–3 дня)**

Подзадачи:
- Coverage ≥ 75% на новый код Phase 1–5.
- Integration tests для всех phase boundaries (capture → analyze, sync → notify, и т.д.).
- E2E test: симулированный звонок проходит полный pipeline → проверка в AMO + notification + bot context.

**6.4. B2B-readiness (1–2 дня)**

Подзадачи:
- Конфигурация на multi-tenant: возможность развернуть для другой компании, поменяв `.env`.
- Описание архитектуры для презентации B2B-клиентам.

#### Final Verification Gate

- [ ] 30 дней непрерывной работы Phase 1-5 без сбоев.
- [ ] РОП и менеджеры используют систему ежедневно (опрос — 100% pull adoption в отделе продаж).
- [ ] Бот обработал ≥ 200 диалогов с конверсией ≥ X% (X фиксируется как baseline).
- [ ] Документация полная.
- [ ] Можно показывать B2B-клиенту как готовое решение.

---

## 6. Тестовая стратегия

| Уровень | Что покрывает | Когда добавляется |
|---|---|---|
| Юнит | Чистые функции, валидация, мапперы | В той же фазе, что код |
| Интеграционные | Сервисы с реальной БД, реальным S3 | В той же фазе |
| Контрактные | API-моки внешних систем (AMO, Tallanto, Mango) | В той же фазе |
| E2E | Полные сценарии end-to-end | Перед каждым phase gate |
| Smoke | Быстрая проверка на CI и production deploy | С Phase 1 |
| Performance | Throughput, latency на нагрузке | Phase 3 и Phase 6 |

**CI требования:**
- Все юнит и интеграционные тесты должны проходить на каждом push.
- E2E запускается на тегах релиза.
- Покрытие отслеживается, регрессия > 5% — блокирует merge.

---

## 7. Безопасность и приватность

**Чувствительные данные в системе:**
- Записи звонков (содержат ФИО клиентов, телефоны, иногда финансовые данные).
- Транскрипты.
- AMO/Tallanto ключи и OAuth токены.
- Mango Office ключи.

**Меры:**
1. Все ключи только в `.env`, gitignored. Production — через sops или Doppler.
2. Storage аудио: только private bucket, доступ через IAM.
3. БД доступна только из VPN/private network.
4. HTTPS обязателен на всех публичных эндпоинтах.
5. Audit log для всех CRM-write операций: кто (какой сервис), что записал, когда.
6. Retention: 90 дней аудио, 365 дней transcripts, далее анонимизация ФИО.
7. Согласие клиентов на запись и обработку: проверить, что Mango Office сам это даёт (стандартная фраза в начале звонка).

**Compliance:** ФЗ-152 «О персональных данных». Это требование к продукту, а не разовая задача — оформить политику обработки персональных данных, отдельный документ.

---

## 8. Риски и митигации

| Риск | Вероятность | Impact | Митигация |
|---|---|---|---|
| Mango Office webhook нестабилен | Средняя | Высокий | Polling fallback с самого начала Phase 2 |
| AMO OAuth срывы | Высокая (был случай) | Высокий | Авто-refresh + alerting + recovery plan |
| ASR качество падает на live (vs batch) | Средняя | Средний | Shadow-режим 7 дней с твоей ручной валидацией |
| CRM-дисциплина менеджеров не позволяет outcome linking | Высокая | Высокий | Аудит CRM-данных перед Phase 5; возможно, отдельный проект |
| РОП/Анна не вовлечены в Phase 5 | Высокая | Высокий | Решить **до Phase 4b**, бронировать их время заранее |
| Throughput не справляется с пиками | Низкая | Средний | Horizontal scaling в Phase 3; metrics покажут |
| LLM API rate limits | Средняя | Средний | Batching + caching + fallback на ollama |
| Деплой VPS падает | Низкая | Высокий | Daily backup БД и storage; runbook восстановления |
| Бот галлюцинирует на ценах/программах | Средняя | Высокий | Strict KB lookup с цитированием; запрет на «придумать» цену |

---

## 9. Решения, которые принять до старта Phase 1

1. **PostgreSQL: managed или self-hosted?** Рекомендация: managed (Selectel Cloud DB).
2. **VPS provider и конфигурация?** Рекомендация: Selectel или TimeWeb, 4 vCPU / 8 GB RAM.
3. **Storage: MinIO local в Phase 1-2 или сразу Selectel S3?** Рекомендация: Selectel S3 с самого начала, не делать локальную миграцию потом.
4. **Domain для webhook URL?** Например, `capture.foton.ru` или `api-mango.foton.ru`.
5. **Управление секретами: .env (Phase 1-3) или sops (Phase 4-6)?**
6. **Бюджет на инфраструктуру в месяц.** Грубо: VPS 4ГБ ≈ 1500 ₽, S3 ≈ 500 ₽, Cloud DB ≈ 1500 ₽, домен и TLS бесплатно. Итого ≈ 3500–5000 ₽/мес.
7. **Объём fallback testing данных.** Сколько исторических звонков использовать для валидации классификатора и outcome linker — 200, 500, 1000?

---

## 10. Что входит в Модуль 3 программы обучения (18–31 мая)

В рамках 2 недель программы реалистично только Phase 1 + начало Phase 2 (subtasks 2.1–2.3 + 2.6 в минимальном виде).

То есть к 31 мая у тебя должно быть:
- AMO sync в production-mode на staged-300 без ошибок.
- PostgreSQL вместо SQLite.
- Базовый capture webhook принимает и обрабатывает live-звонки.
- Базовое TG-уведомление приходит.
- Shadow-режим (твоя ручная валидация) — 7 дней.

Phase 2.4–2.5–2.7 (storage refinement, polling fallback, full shadow) — продолжаются после программы.
Phase 3, 4, 5, 6 — после программы по фазам, в твоём темпе.

---

## 11. Принципы работы с Codex по этому ТЗ

Эти принципы встроены в структуру фаз, но проговариваю явно:

**1. Одна задача — один промпт Codex.** Не «сделай Phase 2». А «сделай subtask 2.2 — webhook receiver, по acceptance criteria из ТЗ». После завершения subtask — проверяешь acceptance, потом переходишь к 2.3.

**2. Перед запуском Codex прочитай задачу сам.** Если ты сам не понимаешь, что должно получиться, ты не сможешь проверить результат.

**3. После Codex запускай код руками.** Не через «он же написал тесты, наверное прошли». Запусти на реальной задаче, посмотри вывод.

**4. На phase gate выноси полную проверку acceptance criteria.** Это не формальность. Если acceptance не проходит — следующая фаза не стартует, что бы ни было сгенерено.

**5. Документация пишется в той же фазе.** `docs/RUNBOOK.md`, `docs/ARCHITECTURE.md` — обновляются по мере работы. Не «потом задокументирую».

**6. Memory эффект.** Через 2 недели после фазы ты забудешь, какие решения принимал. Поэтому каждую фазу заканчивай коммит в git с осмысленным сообщением + опционально запись в `docs/PHASE_X_SUMMARY.md` — что сделано, что осталось, какие сюрпризы.

**7. Когда что-то пошло не так — стоп и анализ.** Если Codex три раза подряд выдаёт нерабочее — это сигнал, что промпт или сама задача нечётко сформулирована. Не пытайся «ещё раз попробовать», возвращайся к ТЗ и переформулируй.

---

## 12. Обновление этого ТЗ

Этот документ — живой. По мере прохождения фаз пиши в `docs/PHASE_X_SUMMARY.md` — что было запланировано, что фактически сделано, какие отклонения. Если отклонения существенные (например, Phase 2 показала, что polling fallback нужно делать раньше) — обновляй ТЗ в соответствующем разделе и помечай changelog.

Не накапливай долг по обновлению ТЗ. Документ должен отражать текущее понимание системы, а не план двухмесячной давности.
