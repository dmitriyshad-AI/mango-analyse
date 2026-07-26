> DONE 2026-07-26 21:52 | ветка codex/integrate-ai-employee-20260726 | codex

> TAKE 2026-07-26 17:34 | ветка unknown | codex

Ветка: codex/integrate-ai-employee-20260726
Зоны: scripts/, src/mango_mvp/, tests/, docs/worktrees_registry.md, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q
Семантический-аудит: да

# Интеграция AI employee A-F

## Цель

Перенести проверенную работу из `codex/ai-employee-implementation` в чистую
ветку от текущего `main`, не сохраняя случайный коммит `5d417ddf init` и не
возвращая более старый код из donor Owner50.

## Обязательные результаты

1. Принять правдивую Wappi-диагностику, fail-loud nightly, AMO checkpoint,
   HTTP retry, attendance-классификацию, Owner50 и существующий semantic harness.
2. Подключить `tallanto_cards_sync` к generated nightly config и доказательству
   свежести; переносить полезные поля карточки, не только имя и телефон.
3. Не писать в prod/AMO/Tallanto/CRM/Wappi и не запускать live-службы.
4. Прогнать точечные тесты, полный pytest и независимый смысловой аудит.
5. Собрать один audit pack. После зелёной проверки влить в `main`; удаление
   временных веток/worktree выполнять только по отдельному подтверждению Дмитрия.

## Запреты

- Никаких `git reset`, `git clean`, `git checkout --`, `git add -A`.
- Никаких новых флагов, зависимостей, второго классификатора или второго runner.
- Никаких тяжёлых реальных nightly-прогонов и публикации рабочей Timeline.

## Приёмка

- Полный pytest зелёный на Mac.
- Generated config реально содержит `tallanto_cards_sync`.
- Повторный импорт Tallanto-карточек не создаёт дублей; конфликт не склеивает
  разные семьи; нужные бизнес-поля не теряются.
- Смысловой аудит Owner50 не находит ложных READY и недоказанных предложений.
- `main`, runtime и внешние системы не менялись до отдельного этапа слияния.

## СТОП

- Любой подтверждённый P0/brand/ПДн/числовой пробой.
- Невоспроизводимый тестовый провал в изменённой зоне.
- Live-процесс появился в рабочей папке или потребовалась запись во внешнюю
  систему/рабочую базу.
