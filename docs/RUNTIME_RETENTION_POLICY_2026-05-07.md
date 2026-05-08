# Runtime Retention Policy

Дата: 2026-05-07

Цель: определить, что хранить, что архивировать и что можно удалять в локальном runtime/data слое.

## Классы артефактов

| Класс | Примеры | Решение |
|---|---|---|
| Source audio | `2026-03-09--26` | Хранить до отдельного external storage/backup |
| Final coverage | `stable_runtime/final_processing_coverage_report_*` | Хранить все, они маленькие |
| Current production DB | DB из актуального `included_dbs.tsv` | Хранить до следующего full rebuild |
| Final batch DB | `messages35`, `ra_missing_all`, `mar2026_client_asr_tail` | Хранить до contact-layer rebuild и backup |
| Backup DB | `.before_*`, `*_backup_*` | Архивировать после coverage v4/v5, локально оставить 1-2 последних на важный batch |
| Research DB | `ab_tests`, `benchmarks` | Сжать/вынести во внешний архив |
| External worker results | `external_m1_*` | После import report хранить одну canonical-копию + zip/manifest во внешнем архиве |
| Telegram exports | `telegram_exports (2)` | После outreach/CRM merge report вынести из корня проекта |
| Generated cache | `__pycache__`, `.pytest_cache`, `.DS_Store`, `egg-info` | Удалять безопасно |
| Broken/old venv | `venv_stable.broken_*` | Не удалять, пока используется launcher/supervisor |

## Текущее решение по `venv_stable.broken_20260407`

Не удалять сейчас. Несмотря на название, последние успешные R+A прогоны используют именно этот Python fallback. Удалять можно только после восстановления нормального `stable_runtime/venv_stable` и проверки:

```zsh
stable_runtime/venv_stable/bin/python - <<'PY'
import sqlalchemy, mango_mvp.cli, mango_mvp.gui
PY
```

## Минимальный safe cleanup

Можно делать без бизнес-риска:

```zsh
find src tests scripts -type d -name '__pycache__' -prune -exec rm -r {} +
rm -rf .pytest_cache
find . -type f -name '.DS_Store' ! -path './.git/*' -delete
```

## DB cleanup gate

Перед удалением/архивацией любой DB ответить на 4 вопроса:

1. Есть ли она в последнем `included_dbs.tsv`?
2. Есть ли более свежая DB, которая покрывает те же `source_filename`?
3. Есть ли import/coverage report, где результат этой DB уже учтен?
4. Есть ли backup/архив вне проекта?

Если хотя бы один ответ неизвестен - не удалять.

## Рекомендуемый следующий cleanup

1. Восстановить нормальный `stable_runtime/venv_stable`, чтобы перестать зависеть от `.broken_20260407`.
2. После закрытия `468` manual tails пересобрать coverage v5.
3. После contact-layer rebuild архивировать `.before_*` DB.
4. Вынести raw Telegram/external worker артефакты из корня проекта.
