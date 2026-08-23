# Read-only M1 runtime inventory

Проверено 2026-08-11T21:40:38Z без вызова Mango API, моделей и внешних систем.

## Машина

- runtime user: `dmitriy`;
- RAM: 32 GiB;
- свободный диск: около 1.0 TiB;
- Python: 3.12.13;
- ffmpeg: 8.1.1;
- Codex CLI: установлен, login status OK;
- активного Mango Calls worker/pipeline не найдено;
- во время проверки шёл тяжёлый Customer Timeline audit, поэтому Calls workload
  не запускался.

## Модели и окружение

- Whisper large-v3 MLX weights: найдены, около 2.9 GiB;
- GigaAM v2 RNNT checkpoint: найден, около 448 MiB;
- `mlx_whisper`, `gigaam`, `torch`, `transformers` в системном Python 3.12: не
  установлены;
- штатный `$HOME/.mango_local/mango_calls_runtime` venv: отсутствует;
- рабочий M1 Calls config/DB/cursor под `$HOME/.mango_local`: отсутствуют;
- owner-only Mango credentials file существует, содержимое и API-доступ не
  проверялись.

Следствие: model weights есть, но runtime к реальному пилоту пока не готов.

## Исторический handoff snapshot

Read-only проверена рабочая SQLite из пакета
`/Users/dmitriy/Projects/Mango_m1_full_handoff_20260807`.

- всего строк: 5,248;
- первый звонок: 2026-07-09 06:44:31;
- последний скачанный звонок: 2026-08-03 17:05:57;
- последний полностью обработанный звонок: 2026-08-03 12:01:56;
- полностью обработано: 4,966;
- ASR не завершён: 30;
- Resolve waiting (`pending + manual`): 245;
- Analyze waiting (`pending + dead`): 282;
- пустых call IDs: 0;
- дублирующихся call IDs: 0.

Это состояние handoff snapshot, а не доказательство актуального состояния
сервера Mango. После 3 августа на M1 ничего не догружалось. Source config
содержит пути исходного Mac и не должен использоваться как M1 config; требуется
штатная relocation/bootstrap-процедура, а не строковая замена SQLite.
