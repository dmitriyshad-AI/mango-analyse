# AMO live-stage plan after tunnel dry-run

Дата: 2026-05-11
Статус: план, live-write не выполняется.

## Текущее состояние

Сотрудники вручную объединяют дубли в AMO/Tallanto. Этот процесс не блокирует безопасные read-only и dry-run задачи, но блокирует массовый live-write по дублям.

Готовые автономные очереди:

| Очередь | Строк | Что можно делать сейчас |
|---|---:|---|
| Non-duplicate strict candidate | 1 | Quality gate уже готов; следующий шаг - real-tunnel dry-run |
| Refresh already-written | 40 | Только diff-based dry-run; live после readback/approval |
| Missing readback | 15 | Сначала readback, потом решать refresh |
| Contact-id mismatch | 1 | Оставить blocked до ручной сверки |

## Последовательность после поднятия tunnel

1. Поднять shared DB tunnel к AMO runtime.
2. Запустить readback по 15 missing-readback строкам.
3. Запустить real-tunnel dry-run по 1 non-duplicate кандидату.
4. Запустить real-tunnel dry-run по 40 refresh-кандидатам.
5. Собрать audit pack по фактическим dry-run/readback отчетам.
6. Получить независимый audit verdict.
7. Создать explicit operator approval artifact на конкретный stage.
8. Выполнить минимальный live stage: сначала 1 non-duplicate или refresh canary 5-10 строк.
9. Немедленно выполнить post-writeback readback gate.
10. Только после зеленого readback расширять stage.

## Запрещено

- Не делать broad rewrite всех AMO-ready строк.
- Не писать строки с duplicate/multi-contact до post-merge recheck.
- Не писать contact-id mismatch без ручного подтверждения.
- Не refresh-ить уже записанные строки без readback и diff.
- Не объединять dry-run и live-write в одну команду.

## Exit criterion для следующего live stage

Следующий live stage разрешается только если одновременно выполнено:

- CRM quality gate passed для точного input CSV.
- Real-tunnel dry-run выполнен end-to-end, failed=0.
- Protected fields не являются write targets.
- Claude/operator audit PASS или PASS_WITH_LIMITATIONS без P0/P1/P2 blockers.
- Есть explicit approval artifact на конкретный CSV и limit.
- После live выполнен readback gate и expected count совпал.
