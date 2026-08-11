# Risk review

## Закрытые P1

- controlled prep больше не пишет shared service lineage marker;
- ticket digest связан с проверенными bytes, а не с повторным чтением после
  proof;
- cleanup failure не уничтожает `before/stages/after`;
- cleanup `OSError` не маскирует primary stage exception.

## Принятый P2 backlog

- same-UID — доверенная локальная граница, ticket не криптографическая
  авторизация; nonce/replay не защищают от произвольного процесса того же UID;
- обычный worker без controlled env остаётся широким service worker по
  назначению; официальный пилот использует только orchestrator command;
- bare `stage_limit=1` не является изоляцией и так документирован;
- `source_cursor_lineage_ok` в controlled report семантически похоже на service
  lineage, но дополнено `lineage_mode=controlled_read_only` и marker=false;
- при primary stage failure внешний аварийный отчёт не включает весь mutable
  cleanup evidence, хотя исходная ошибка сохраняется и cleanup выполняется;
- общий stop_reason cleanup одинаков для tamper/unlink/rmdir, точная причина
  находится в `asr_input_snapshot_cleanup.errors`;
- direct-child путь не даёт переносимый `peak_rss_raw`;
- parser читает последний подходящий JSON из owner-only лога;
- allowlist numeric поля допускают Python `bool` как `int`;
- повторные Git/DB digest проверки стоят времени, но усиливают fail-closed;
- cleanup каталогов не имеет отдельной remediation receipt и parent fsync.

## Бизнес/runtime STOP

- актуальный cursor и последние обработанные звонки старого Mac не перенесены и
  не доказаны на этом M1;
- зависимости, модели, доступы, память и скорость на реальном controlled звонке
  не проверены;
- Google/Yandex publication и внешний watchdog не развёрнуты;
- имя клиента и закрытая ссылка относятся к последующей publication-фазе;
- один звонок, нулевой повтор, 10 звонков, сутки, 3 дня РОПа и 7 суток службы
  не пройдены.
