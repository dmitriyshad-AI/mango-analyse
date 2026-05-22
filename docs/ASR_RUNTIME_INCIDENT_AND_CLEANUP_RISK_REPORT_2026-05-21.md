# ASR Runtime Incident and Cleanup Risk Report, 2026-05-21

## Короткий вывод

Проблема была не в данных Mango и не в самих аудио. Сломался запускной слой: UI мог стартовать в Python-окружении, где были GUI-зависимости, но не было ASR-библиотек `mlx_whisper` и `gigaam`. Из-за этого batch сначала массово ушёл в `dead`.

Сейчас безопасно разрулена только инфраструктурная часть:

- добавлен read-only preflight `scripts/check_asr_runtime_contract.py`;
- runbook обновлён: перед ASR теперь обязательна проверка окружения;
- cleanup-док обновлён: `.venv-asrbench` больше не кандидат на удаление;
- 53 shell-launchers в `stable_runtime` переведены с удалённого/неполного `venv_stable*` на `.venv-asrbench`;
- активные ASR DB/audio/transcripts не изменялись;
- старые `stable_runtime` DB/audio/transcripts не переписывались во время активного ASR batch.

## Что произошло

1. Был собран ASR-only batch для свежих Mango-записей после 2026-05-12.
2. UI сначала запускался нестабильно из-за выбора Python-окружения.
3. После запуска выяснилось, что часть окружений проходит GUI-проверки, но не имеет `mlx_whisper`/`gigaam`.
4. Строки с ошибкой окружения были безопасно переочередлены отдельным script, после чего ASR пошёл дальше.
5. При очистке корзины macOS показал сообщение, что объект `messages(35)` используется. Проверка не нашла активных процессов, держащих `messages35` или `messages(35)`.

## Причины

### Причина 1: неполная проверка окружения

Старый launcher проверял только часть зависимостей. Этого хватало, чтобы открыть UI, но не хватало, чтобы гарантировать реальный ASR.

### Причина 2: удалённый legacy venv остался в старых scripts

`stable_runtime/venv_stable.broken_20260407` был удалён после cleanup, но старые launchers всё ещё ссылаются на него как fallback Python. Это риск для повторного запуска старых batch-скриптов.

### Причина 3: `.venv-asrbench` был ошибочно похож на мусор

В cleanup-доках `.venv-asrbench` был описан как rebuildable benchmark env. Сейчас это фактически текущий рабочий ASR runtime, поэтому удалять его нельзя.

## Что проверено read-only

Команда:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
python3 scripts/check_asr_runtime_contract.py
```

Результат:

- `active_runtime_ok=true`;
- preferred runtime: `.venv-asrbench/bin/python`;
- `.venv` не хватает `mlx_whisper` и `gigaam`;
- `stable_runtime/venv_stable` не хватает `sqlalchemy`, `dotenv`, `mlx_whisper`, `gigaam`;
- до ремонта shell-launchers было найдено 40 старых ссылок на `venv_stable.broken_20260407` и 14 прямых ссылок на неполный `venv_stable`;
- после ремонта `runtime_launcher_count=0` для `venv_stable.broken_20260407`;
- после ремонта `rg 'venv_stable(.broken_20260407)?/bin/python' stable_runtime -g '*.sh'` возвращает 0 строк;
- 9 оставшихся ссылок находятся в документах/служебном preflight и не являются launchers.

ASR batch status на момент проверки:

- всего: 848;
- `done`: 794;
- `in_progress`: 6;
- `pending`: 48;
- `dead_letter_stage`: `{}`.

## Что не трогалось

- Не удалялись файлы и папки.
- Не переносились аудио.
- Не менялись SQLite DB/WAL/SHM.
- Не запускались ASR/R+A заново из Codex.
- Не менялись `stable_runtime` runtime-артефакты.
- Не запускались live-записи во внешние системы.

## Почему `messages35_asr_only_20260506` нельзя считать мусором

`stable_runtime/messages35_asr_only_20260506/messages35_asr_only_20260506.db` входит в lineage текущих canonical master слоёв. Даже если часть аудио дублируется по содержимому с canonical audio root, сама DB является источником обработанных ASR/R+A данных.

Удаление возможно только после отдельного archive/backup контракта и проверки, что новая canonical база больше не зависит от этого source DB.

## Что делать дальше

1. Дождаться завершения текущего ASR batch.
2. Проверить `mango_mvp.cli stats`: `done=848`, `dead_letter_stage={}`.
3. После отдельного подтверждения Дмитрия запускать Resolve+Analyze.
4. После accepted rebuild обновить runtime-указатели и audit pack.
5. Отдельным спокойным этапом решить, какие исторические launchers в `stable_runtime` оставить, а какие пометить deprecated. Удалять их можно только после отдельного подтверждения.
