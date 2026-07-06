> DONE 2026-07-06 08:17 | ветка codex/adr003-semanticframe-migration | codex

> TAKE 2026-07-06 07:48 | ветка codex/adr003-semanticframe-migration | codex

Ветка: codex/adr003-semanticframe-migration
Зоны: scripts/, src/mango_mvp/channels/, src/mango_mvp/replay_exam/, tests/, docs/, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest --collect-only -q
Семантический-аудит: да

# TZ: ADR003 final dovetail + replay pilot + deploy prep

Контекст: владелец разрешил read-only Wappi-чтение для replay-пилота. M1 пара фокусного набора уже отправлена отдельно. Этот заход не должен писать во внешние системы и не должен менять live runtime.

## Цели

1. Довесок ADR003 после `live_status_read`:
   - удалить оставшиеся legacy stem/facet участки `мест` только по измеренному манифесту атомарного №2;
   - сохранить fail-closed пол на live availability;
   - понизить соответствующие бюджеты/снапшоты в том же коммите;
   - добавить/сохранить NEG seats контроль и локальный smoke.

2. Replay pilot:
   - довести read-only exporter до реальной Wappi GET-пагинации `mark_all=false`;
   - raw dump писать только в `~/.mango_local/replay_exam/raw/`;
   - сразу делать pseudonymize/scrub, auto-grep 0 leaks, slice, offline pilot-10;
   - report pack: методика, scrubbed сырьё, 5 полных тестов с памятью/raw trace;
   - полный M1 replay exam не собирать до отдельного GO.

3. Deploy-prep data-only:
   - карта отличий live-ветки/процесса от текущего канона;
   - инвентарь `~/.mango_secrets` и draft_loop конфигурации только с маскированием значений;
   - черновик swap/rollback/stop-crane плана документом.

## Запреты

- Не отправлять сообщения клиентам.
- Не писать в AMO/CRM/Tallanto/Wappi.
- Не менять live process/env/runtime.
- Не коммитить raw Wappi dump, секреты, ПДн, `~/.mango_local`.
- Не запускать полный M1 replay exam.

## Проверки

- `git diff --check`
- `bash -n` для новых/изменённых shell/python CLI где применимо
- targeted pytest по ADR003 и replay
- `pytest --collect-only`
- audit pack + semantic review
