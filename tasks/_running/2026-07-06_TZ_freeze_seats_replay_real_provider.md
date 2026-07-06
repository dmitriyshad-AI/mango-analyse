> TAKE 2026-07-06 16:43 | ветка codex/adr003-semanticframe-migration | codex

Ветка: codex/adr003-semanticframe-migration
Зоны: scripts/, src/mango_mvp/channels/, src/mango_mvp/replay_exam/, tests/, docs/, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest --collect-only -q tests/test_direct_p0_text_hygiene.py tests/test_wappi_replay_scripts.py tests/test_wappi_replay_exporter.py tests/test_wappi_replay_slicer.py tests/test_wappi_replay_pseudonymizer.py tests/test_wappi_replay_judge.py tests/test_subscription_llm_draft_provider.py
Семантический-аудит: да

# TZ: ADR003 freeze live, seats default-open, replay real provider

Контекст: решение владельца и регрейд Claude/Fable 2026-07-06 после `d0357d79`.

## Цели

1. FREEZE LIVE: снять свежий read-only снимок состояния живого Telegram-бота и подготовить swap/rollback план, исполнимый человеком без ИИ. Ничего live не переключать.
2. Политика мест: реализовать `TELEGRAM_SEATS_DEFAULT_OPEN` default-OFF по ТЗ `/Users/dmitrijfabarisov/Claude Projects/Foton/2026-07-06_TZ_politika_mest_default_open.md`: регулярные группы по умолчанию открыты, исключения-полы сохранены.
3. Replay real-provider adapter: разрешить локальный replay на реальном draft-provider только по scrubbed cases, без live-write, за явным флагом. Полный M1 replay exam не собирать без отдельного GO.

## Жёсткие запреты

- Не отправлять Telegram/Wappi/AMO/CRM/Tallanto сообщения.
- Не писать во внешние системы.
- Не останавливать live-процессы и не выполнять swap/rollback.
- Не раскрывать значения secrets/env в документах или stdout.
- Не менять `stable_runtime`.
- Не включать новые флаги в live/profile без отдельного решения владельца.

## Приёмка

- Для каждого этапа: отдельная проверка субагентом-аудитором, правки учтены или письменно отклонены.
- Для клиентских текстов и KB-факта: semantic review.
- Для кода: focused pytest + `git diff --check`.
- Для freeze/deploy: документ с read-only фактурой и dry-run/print-only rollback/swap планом.
- Для policy seats: unit/NEG + локальная микро-пара или честный стоп, если нужен M1/регрейд.
- Для replay adapter: pilot-10 или честный стоп, если нет безопасного scrubbed set/provider prerequisites.
