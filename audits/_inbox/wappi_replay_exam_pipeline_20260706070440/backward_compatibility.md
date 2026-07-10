# Обратная совместимость

- Форматы: новые replay artifacts используют отдельные схемы `wappi_replay_raw_v1`, `wappi_replay_exam_summary_v1`, `wappi_replay_m1_manifest_v1`; существующие ADR003/E3 runner formats не менялись.
- Потребители: direct-path, Wappi draft loop, AMO/Tallanto/CRM clients не импортируют новый replay package.
- Runtime: `AmoWappiDraftLoop.run_once` не используется и не менялся.
