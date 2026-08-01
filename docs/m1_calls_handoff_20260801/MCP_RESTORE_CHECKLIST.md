# Восстановление Codex на M1

Этот список переносит возможности, но не авторизацию. Токены, `auth.json`,
`config.toml`, браузерные профили, журналы и локальные базы Codex не копировать.

## Локальные навыки

1. До распаковки сверить SHA-256 архива с
   `codex_profile_checksums_20260801.txt`.
2. Распаковать `codex_skills_clean_20260801.tar.gz` в `~/.codex/skills`.
3. Для `multi-model-analysis-review` переустановить Node-зависимости из его
   lock-файла; каталог `node_modules` в архив не входит.

Полный ожидаемый список записан в `codex_profile_manifest_20260801.json`.
Системные skills устанавливаются вместе с Codex и намеренно не копируются;
архив содержит пользовательские skills без `node_modules`.

## Plugins и подключения

Заново установить или включить доступные в аккаунте plugins из manifest:
browser, build-web-apps, chrome, computer-use, documents, github, pdf,
presentations, sites, spreadsheets, template-creator и visualize. Дополнительные
кэшированные figma, google-drive и openai-templates ставить только если они
доступны этому аккаунту и реально нужны.

Заново авторизовать GitHub, Google Drive и Todoist через интерфейс. Не переносить
их токены с другого Mac.

## MCP

- `node_repl`: установить вместе с актуальным Codex runtime и проверить через
  `codex mcp list`; старые абсолютные пути не копировать.
- `computer-use`: включать только через установленный Desktop/runtime; на
  исходном Mac он был отключён.
- GitHub: удалённое подключение, нужна новая авторизация на M1.
- Todoist: connector, нужна новая авторизация на M1.
- Google Drive: plugin/connector, нужна новая авторизация на M1.

Resolve и Analyze намеренно запускаются в изолированном профиле без skills,
plugins и MCP. Восстановление пользовательского профиля нужно для разработки и
аудита, но не влияет на штатную обработку звонков.
