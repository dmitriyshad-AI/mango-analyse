# TZ-141 — MTS-Link fix + M1 bundle

Дата: 2026-06-18
Ветка: codex/tz135-direct-wow-tone

## Что изменено

- Обновлён снапшот v6.5:
  `product_data/knowledge_base/kb_release_20260603_v6_5_summer_format_cleanup/kb_release_v3_snapshot.json`.
- Целевые факты переведены на текущую платформу SohoLMS:
  - `foton / online_platform.name`
  - `unpk / online_platform.name`
  - `foton / presentation_format_facts_2026_05_21.client_facts.online_lesson_format.client_safe_text`
  - `foton / presentation_format_facts_2026_05_21.client_facts.online_technical_requirements.client_safe_text`
  - `unpk / tg_unpk_verified_2026_05_21.client_facts.online_courses_format.client_safe_text`
- MTS-Link оставлен только как историческая оговорка: для курсов, начатых до лета 2026.
- Ссылки `https://my.mts-link.ru/...` не изменялись.
- `_BRAND_TOKENS` не изменялся.
- Добавлен набор:
  `product_data/telegram_dynamic_test_sets/p0_stability_set_20260617.jsonl`
  (10 персон, sha256 `238fc075215f7eb28e94268b0bc936b0606f03680da73f64e1a658ae82287250`).

## Греп client_safe_text

Изменённые клиентские тексты:

- `Фотон: онлайн-платформа для новых онлайн-курсов — SohoLMS; MTS-Link используется только для курсов, начатых до лета 2026.`
- `УНПК: онлайн-платформа для новых онлайн-курсов — SohoLMS; MTS-Link используется только для курсов, начатых до лета 2026.`
- `Онлайн-занятия проходят в формате вебинаров на SohoLMS: ученики задают вопросы в чате, преподаватель может вызвать в голосовой или видео-чат, а записи уроков доступны для пересмотра.`
- `Для онлайн-занятий нужен стабильный интернет от 2,5 Мбит/с. Можно подключаться с телефона, планшета или компьютера; полный функционал удобнее на компьютере в Chrome. Если есть техническая проблема, можно написать @support в чате занятия.`
- `Онлайн-курсы УНПК проходят на платформе SohoLMS. Обычно это группы до 20 человек, занятия 2 раза в неделю по 90 минут; после каждого урока доступны записи. Есть домашние задания и контрольные срезы, обычно 5-6 раз в год.`

Проверка остаточных MTS-упоминаний в снапшоте:

- `suspicious_old_mentions = 0`
- оставшиеся упоминания: только историческая оговорка и 4 URL-факта `my.mts-link.ru`.

## Semantic review

Verdict: PASS.

- Новые тексты не называют MTS-Link текущей платформой.
- SohoLMS указан как текущая платформа для новых онлайн-курсов.
- Историческая оговорка ограничена курсами, начатыми до лета 2026.
- Бренды не смешаны.
- Новых цен, дат старта, гарантий, обещаний менеджера или ссылок не добавлено.

## Проверки

- Полный pytest:
  `3334 passed, 2 skipped, 1 warning in 63.40s`.
- Симулятор не запускался по ТЗ.

## Бандл

- Точный bundle id зависит от финального hash коммита и фиксируется в `BUNDLE_INFO.txt`.
- Путь: `~/Yandex.Disk.localized/OpenClaw/Actual Mango Tests/mango_clean_<head>`.
- `manifest.json` записывается сборщиком последним.
- `BUNDLE_INFO.txt` должен содержать:
  - branch: `codex/tz135-direct-wow-tone`
  - kb_snapshot: `product_data/knowledge_base/kb_release_20260603_v6_5_summer_format_cleanup/kb_release_v3_snapshot.json`
- В бандле есть оба набора:
  - `tz135_wow_tone_coverage_20260617.jsonl`
  - `p0_stability_set_20260617.jsonl`
- Проверка снапшота в бандле: `suspicious_old_mentions = 0`.
