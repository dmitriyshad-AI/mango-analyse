# ADR003 F2p Posthoc SemanticFrame Scope Calibration

## Изменение

Уточнён только post-hoc prompt `build_direct_path_semantic_frame_posthoc_prompt()`.
Inline SemanticFrame в основном direct-path prompt не тронут.

Новая инструкция заставляет frame сначала классифицировать смысл запроса клиента,
а не копировать осторожность текущего `final_route` или текста "менеджер проверит".
`missing_facts` теперь должен ставиться только если нет проверенной client-safe
опоры ни в `final_draft_text`, ни в переданных metadata.

## Почему

F2o показал, что реальный остаток автономности не в price/ack route-only, а в
границе "стабильная справка о существовании/формате" против "живые места/запись".
При этом active demotion остаётся недопустимым: чистых route-only кандидатов 0,
основной остаток требует проверенного fact path.

## Замер

Локальный paired no-op enrichment на OFF-транскриптах M1-прогона 36ea110:

- dialogs: 131;
- turns: 241;
- route/text/input diff: 0;
- SemanticFrame present: 241/241;
- required fields complete: 241/241;
- post-hoc frame model calls: 241;
- non-frame ON calls: 0;
- self-answer demotion candidates: 0;
- P0/money/operational/manager-only lowered: 0;
- too_confident: 0;
- too_cautious: 10 (было 14 на F2o/F2n);
- strict_active_candidates_now: 0.

## Вывод

Prompt-калибровка полезна как shadow/report-only шаг: frame стал менее
осторожным без роста `too_confident`. Но active автономность всё ещё NO-GO:
чистого route-only рычага нет, следующий содержательный рычаг — проверенный
client-safe fact path для стабильных справочных вопросов.
