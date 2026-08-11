# Codex audits

## Architect-auditor

Первый STOP: controlled allowlist создавал общий service lineage marker.
Исправлено отдельным read-only proof; service callers оставлены на старом gate.

Второй STOP: ticket мог повторно хешировать уже заменённый manifest. Исправлено:
ticket получает digest именно проверенных `cutover_before` bytes.

Третий STOP: cleanup `OSError` мог маскировать первичную stage-ошибку.
Исправлено точечным catch и двумя отрицательными тестами.

Финал: GO, P0/P1 нет; 349 tests + 5 subtests в его выбранном контуре,
Python 3.12 AST и diff-check прошли.

## Breaker

Проверены stale/mismatch/race, broad service unlock, exact-one, non-target,
missing/tamper/unlink/nonempty cleanup и primary-error preservation.

Финал: GO, P0/P1 нет. В cleanup-аудите выполнены живые атаки; остатки всегда
блокировали следующий запуск до первой стадии.

## Business-auditor / optimizer

GO к коду controlled-one; production/service и business остаются STOP.
Подтверждено, что `status=ok` — только машинный локальный результат, а все
business/runtime/human flags остаются false до человека.

## Manic cleaner

GO. Мёртвого или дублированного P0/P1 кода не найдено. Новые функции имеют
реальные callers; старого пути пользователя в изменённых файлах нет.
Неблокирующий backlog вынесен в risk review.
