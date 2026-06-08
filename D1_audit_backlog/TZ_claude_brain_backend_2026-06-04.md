# ТЗ — Сменный мозг бота: Claude-бэкенд (Sonnet 4.6) рядом с Codex/GPT. Для Кодекса. 2026-06-04.

Автор: Клод #1. Идея Дмитрия: дать боту альтернативный «мозг» — Claude CLI (Sonnet 4.6, высокий reasoning) рядом с
текущим Codex CLI (gpt-5.5). Зачем: (1) честный A/B «какой мозг лучше для НАШЕГО бота» — замерить на одном наборе +
регрейд по сырью, не гадать; (2) фундамент под SaaS «выбор мозгов».

## Железный принцип (почему это безопасно)
Архитектура **модель-агностична**: детерминированный выходной гейт (бренд/P0/выдумки/число-гейт/leak/faithfulness)
проверяет ВЫХОД независимо от того, какая модель его написала. Claude-бэкенд — это ДОБАВОЧНЫЙ провайдер инференса, он
НЕ трогает слой безопасности и НЕ обходит гейты. Текст от Claude идёт через ТОТ ЖЕ output-gate, что и от GPT.
Дефолт остаётся `codex`/gpt-5.5. Claude — opt-in режимом. Бот-логика/гейты/правила НЕ меняются.

## Текущая архитектура (подтверждено по коду `scripts/run_telegram_dynamic_client_sim.py`)
- Интерфейс модели: объект с `generate(prompt: str) -> Mapping[str, Any]`. Есть `Fake*` модели и боевой
  `CodexJsonModel` (строка ~219).
- `CodexJsonModel.generate`: вызывает `codex exec --skip-git-repo-check --ephemeral --sandbox read-only --model <model>
  [-c model_reasoning_effort="<effort>"] --output-last-message <path> -`, подаёт prompt в stdin, читает файл, парсит
  `extract_json_object(raw)`. Env через `_codex_env()`.
- Раннер выбирает backend: `--bot-mode {codex,fake}`, `--model gpt-5.5`, `--bot-reasoning medium`; отдельно
  `--memory-mode/--semantic-mode/--selling-mode {codex,fake,off}` + свои `--*-model/--*-reasoning`.

Значит Claude = новый класс `ClaudeJsonModel` (зеркало `CodexJsonModel`) + опция `claude` в режимах.

## ФАЗА 0 — короткая сверка (Кодекс подтверждает 2 неизвестных, потом кодит)
Я НЕ хардкожу флаги Claude CLI наугад. Кодекс подтверждает по `claude --help` на M1/маке:
1. **Точная команда неинтерактивного one-shot вызова** Claude CLI: print-режим (`claude -p`), выбор модели
   (`--model claude-sonnet-4-6` или `sonnet`), формат вывода (`--output-format text` — вернуть текст ответа модели),
   и КАК выключить tool-use/MCP (это «мозг», не агент — нужен чистый prompt→текст, без инструментов и без правки файлов).
2. **Как задать «высокий reasoning» (extended thinking) Sonnet 4.6** в CLI (флаг/конфиг/ключевое слово) и поролевое
   соответствие уровням gpt `low/medium/high`.
Где иначе и ПОЧЕМУ; риски; вердикт. Потом — реализация по полуфабрикатам.

## ФАЗА 2 — полуфабрикаты

### Полуфабрикат 1 — `ClaudeJsonModel` (зеркало `CodexJsonModel`, тот же интерфейс `generate`)
```python
class ClaudeJsonModel:
    def __init__(self, *, model: str = "claude-sonnet-4-6", reasoning: str = "high",
                 timeout_sec: int = 180, claude_bin: str = "claude") -> None:
        self.model = model
        self.reasoning = reasoning          # маппинг см. Полуфабрикат 3
        self.timeout_sec = timeout_sec
        self.claude_bin = claude_bin

    def generate(self, prompt: str) -> Mapping[str, Any]:
        cmd = [
            self.claude_bin, "-p",                       # print-режим, неинтерактивно
            "--model", self.model,
            "--output-format", "text",                   # текст ответа модели → потом extract_json_object
            # ВЫКЛЮЧИТЬ tool-use/MCP (это мозг, не агент): точный флаг подтвердить в Фазе 0,
            # ориентир: --disallowedTools "*" / --permission-mode plan / запуск без MCP-конфига.
            *_claude_reasoning_args(self.reasoning),     # extended thinking, см. Полуфабрикат 3
        ]
        proc = subprocess.run(
            cmd, input=prompt, capture_output=True, text=True,
            timeout=self.timeout_sec, check=False, env=_claude_env(),
        )
        if proc.returncode != 0:
            raise RuntimeError(f"claude -p failed rc={proc.returncode}: {(proc.stderr or '')[-500:]}")
        # тот же парсер, что у codex: вытащить JSON-объект из текста ответа
        return extract_json_object(proc.stdout or proc.stderr)
```
Замечание: боковые промпты бота просят модель вернуть JSON-объект; `extract_json_object` находит его в тексте (как у
codex). Если extended thinking печатает рассуждение перед ответом — финальный текст должен содержать JSON; парсер его
выцепит. Проверить на 1-2 реальных промптах в Фазе 0.

### Полуфабрикат 2 — выбор backend в раннере (добавить `claude` в режимы)
```python
parser.add_argument("--bot-mode", choices=("codex", "claude", "fake"), default="codex")   # +claude
# аналогично при желании: --memory-mode/--semantic-mode/--selling-mode += "claude"
def build_bot_model(args):
    if args.bot_mode == "fake":   return FakeBotModel(...)
    if args.bot_mode == "claude": return CountingGenerateModel(ClaudeJsonModel(
        model=args.model if args.model.startswith("claude") else "claude-sonnet-4-6",
        reasoning=args.bot_reasoning, timeout_sec=args.timeout_sec))
    return CountingGenerateModel(CodexJsonModel(model=args.model, reasoning_effort=args.bot_reasoning,
                                                timeout_sec=args.timeout_sec))
```
`CountingGenerateModel`-обёртка сохраняется (счётчик LLM-вызовов в summary работает для обоих мозгов одинаково).
v1-приоритет: `--bot-mode claude` (главный draft/understand-мозг). Память/семантика/продажа — следующим шагом
(можно временно оставить на codex или fake), чтобы изолировать эффект СМЕНЫ ГЛАВНОГО мозга.

### Полуфабрикат 3 — `_claude_env` + маппинг reasoning
```python
def _claude_env() -> dict:
    env = dict(os.environ)
    # ключ/конфиг Claude (ANTHROPIC_API_KEY или конфиг Claude CLI) — НЕ логировать, как _codex_env
    return env

_CLAUDE_REASONING = {"low": [], "medium": [...], "high": [...]}   # точные args/флаги extended thinking — из Фазы 0
def _claude_reasoning_args(level: str) -> list[str]:
    return list(_CLAUDE_REASONING.get(level, _CLAUDE_REASONING["high"]))
```
Sonnet 4.6 «максимальный reasoning» = высокий уровень extended thinking на ролях understand/draft/critic; memory — low
(дёшево). Точную CLI-механику thinking подтвердить в Фазе 0.

### Полуфабрикат 4 — паритет промптов (для ЧЕСТНОГО A/B)
Промпты/few-shot могли быть подкручены под gpt. Перед сравнением: НЕ менять промпты под Claude (иначе сравниваем
промпты, а не мозги) — сравнить на ОДИНАКОВЫХ промптах. Если Claude систематически ломает формат JSON — это РЕЗУЛЬТАТ
сравнения (записать), а не повод чинить промпт только под него. Любую правку формата применять к ОБОИМ мозгам.

### Полуфабрикат 5 — гейты и безопасность НЕ меняются
Выход Claude идёт через тот же `apply_authoritative_output_gate` / faithfulness / число-гейт / бренд / P0. Никаких
обходов. NEG-контроль: прогон `--bot-mode claude` на наборе с продуктовой выдумкой/бренд-провокацией/P0 — гейты
держат так же, как для codex (выдумок к клиенту 0, бренд 0, P0→менеджер).

## ПЛАН A/B (как замерим «какой мозг лучше»)
Один и тот же набор (напр. Level A + память нити + продающий) гоняем дважды: `--bot-mode codex` и `--bot-mode claude`,
`--parallel 4`, тот же snapshot/registry. Я регрейжу ОБА по сырью на одной машине (межмашинное не мешать): сравнить
выдумки (число/нечисловые), тон/живость, over-handoff, удержание нити, формат. Вывод — таблица codex vs claude по
сырью, не по агрегату судьи. Решение по «мозгу по умолчанию» — после данных.

## SaaS-замечание (на будущее, не сейчас)
Это фундамент: позже выбор мозга вынести в конфиг рантайма (на деплой/бренд/тариф), а не только CLI-флаг. Архитектура
уже позволяет — гейт модель-агностичен.

## Что НЕ делать
Не менять бот-логику/гейты/правила/промпты ради Claude; не делать Claude дефолтом; не обходить output-gate; не
хардкодить непроверенные флаги claude CLI (Фаза 0 подтверждает); ключи не логировать; не git reset. Это ПАРАЛЛЕЛЬНЫЙ
трек — не тормозит конвейер качества (Блоки 1-6).
