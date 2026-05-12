# AMO waiting autonomous work: Claude audit instruction

Дата: 2026-05-11
Scope: безопасная автономная работа, которую можно делать, пока сотрудники вручную объединяют AMO/Tallanto дубли.

## Что проверяем

Аудит должен проверить не частные телефоны, а классы рисков:

1. Live-write не выполнялся и не разрешен ни в одном generated command.
2. Non-duplicate кандидат проходит только через quality gate -> real-tunnel dry-run -> audit/approval -> staged live -> readback.
3. Refresh по 40 уже записанным контактам является diff-based, а не broad rewrite-all.
4. 15 строк без readback не допускаются к refresh до успешного readback.
5. Contact-id mismatch остается заблокированным и не принимается автоматически.
6. Operator/API статус честно показывает: сотрудники чинят дубли, dry-run можно готовить при поднятом tunnel, live-write запрещен до approval.

Важно: `WRITE_AMO_LIVE` может встречаться в read-only Product API metadata как описание обязательной confirmation phrase. Это не нарушение само по себе. Нарушение - если live-флаг или confirmation argument есть в executable generated command, либо если API/action реально включает write.

## Команда для запуска Claude

```zsh
cd "/Users/dmitrijfabarisov/Projects/Mango analyse"
claude -p --model opus --effort high --permission-mode acceptEdits \
  "/audit audits/_inbox/amo_waiting_autonomous_work_20260511_v1"
```

## Fallback prompt, если slash-command не сработает

```text
Проведи независимый read-only аудит пакета audits/_inbox/amo_waiting_autonomous_work_20260511_v1.

Прочитай CLAUDE.md, docs/AI_WORKFLOW.md и docs/THREAT_MODEL.md.
Не редактируй ничего вне audits/_results/.
Не трогай stable_runtime вне чтения.
Не запускай ASR/R+A.
Не пиши в CRM/AMO/Tallanto.

Проверь:
- нет ли live-write flags в generated commands;
- non-duplicate candidate не может быть записан без quality gate + real-tunnel dry-run + explicit approval;
- refresh candidates являются diff-based, а не rewrite-all;
- readback_missing rows заблокированы до успешного readback;
- contact_id_mismatch rows fail-closed;
- operator/dashboard/API статус не вводит в заблуждение: dry-run отдельно, live-write отдельно.

Сохрани результат в audits/_results/2026-05-11_amo_waiting_autonomous_work_v1/:
- CLAUDE_REAUDIT_RESULT.md
- findings.csv
- row_decisions.csv

Финальный verdict: PASS / PASS_WITH_LIMITATIONS / FAIL.
```

## Ожидаемые числа

- Non-duplicate candidate rows: 1.
- Refresh candidate rows: 40.
- Readback missing rows: 15.
- Contact-id mismatch rows: 1.
- Live-write allowed now: false.
- Write CRM in this package: false.
