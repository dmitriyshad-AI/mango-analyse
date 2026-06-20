# Semantic review

## Verdict

PASS_WITH_NOTES.

Это safety-code для AMO writeback, не клиентский текст. Смысловая проверка применена к бизнес-инвариантам CRM/AMO: не затирать менеджера, не писать нецелевые поля, не смешивать бренды в сделке, оставаться dry-run.

## What passed

- Ручная правка менеджера защищена: fresh GET перед PATCH сравнивается с snapshot/journal sha.
- Contact path получил GET + snapshot + rollback через общий формат.
- Нижний helper не пропускает телефон, ФИО, email, ручную `История общения`, статус/этап/ответственного.
- Deal writer блокирует бренд-конфликт по каналу и несколько открытых сделок, если эти признаки есть во входной строке.
- Dry-run journal не становится источником "мы это уже записали".

## Blocking issues

- Нет live AMO проверки и настоящего readback: это запрещено текущим ТЗ, поэтому не является дефектом блока.

## Non-blocking risks

- Бренд-чек зависит от наличия brand-полей во входном Stage6 row. Если upstream их не отдаёт, guard fail-open по этому признаку.
- `multiple_open_deals` тоже зависит от входного счётчика. Если upstream его не отдаёт, блокировки не будет.
- `~/.mango_local/amo_writeback/journal.jsonl` станет source-of-truth только после настоящего `action=written`; dry-run строки намеренно не участвуют в anti-clobber state.

## Missing checks

- Реальный AMO field inventory/contact+lead mapping не выполнен: live/read-only AMO не вызывался в этом блоке.
- Реальный rollback по AMO не выполнялся.

## Required regression tests or gate rules

- `clobber_protected` перед PATCH.
- `unchanged` повтор без записи.
- contact snapshot/rollback.
- protected/allowlist для contact и lead.
- brand conflict block.
- multiple open deals block.

## Recommended next action

Отдать код и audit pack на регрейд Claude. До отдельного "да" Дмитрия не запускать `--execute-live-write` и rollback `--apply` на реальном AMO.
