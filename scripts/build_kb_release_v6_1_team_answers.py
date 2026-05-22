#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from scripts import build_kb_release_v3_from_claude_handoff as kb_builder
from scripts.build_kb_distribution_packs import build_distribution_packs
from scripts.run_kb_semantic_review import run_kb_semantic_review


DEFAULT_SOURCE_ROOT = Path("/Users/dmitrijfabarisov/Claude Projects/Foton/kb_for_bot_review_2026-05-18")
DEFAULT_RUN_ID = "kb_release_20260520_v6_3_team_answers"
DEFAULT_SOURCE_OUT = Path("product_data/knowledge_base/kb_release_20260520_v6_3_team_answers_sources")
DEFAULT_RELEASE_OUT = Path("product_data/knowledge_base/kb_release_20260520_v6_3_team_answers")
DEFAULT_HANDOFF_OUT = Path("product_data/knowledge_base/kb_release_20260520_v6_3_team_answers_handoff_for_claude_and_team")
DEFAULT_BOT_PACK_OUT = Path("product_data/knowledge_base/kb_release_20260520_v6_3_team_answers_bot_pack")
DEFAULT_EMPLOYEE_PACK_OUT = Path("product_data/knowledge_base/kb_release_20260520_v6_3_team_answers_employee_pack")
DEFAULT_SMOKE_NOT_RUN = Path("product_data/knowledge_base/kb_release_20260520_v6_3_team_answers_smoke_not_run")
GOLD_ANSWERS_V3_DOCX = (
    "/Users/dmitrijfabarisov/Claude Projects/Foton/gold_answers_2026-05-21/GOLD_ANSWERS_v3_2026-05-21.docx"
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build KB v6.1 from team answers without MEGA or live smoke runs.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--source-out", type=Path, default=DEFAULT_SOURCE_OUT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--release-out", type=Path, default=DEFAULT_RELEASE_OUT)
    parser.add_argument("--handoff-out", type=Path, default=DEFAULT_HANDOFF_OUT)
    parser.add_argument("--bot-pack-out", type=Path, default=DEFAULT_BOT_PACK_OUT)
    parser.add_argument("--employee-pack-out", type=Path, default=DEFAULT_EMPLOYEE_PACK_OUT)
    parser.add_argument("--smoke-dir", type=Path, default=DEFAULT_SMOKE_NOT_RUN)
    args = parser.parse_args(argv)

    source_out = prepare_source_overlay(args.source_root, args.source_out)
    patch_sources(source_out)
    register_gold_answers_source()

    # v6.x removes outdated preschool, expired intensive and closed early-minimum prices from client/current control scope.
    kb_builder.CONTROL_NUMBERS = tuple(
        number
        for number in kb_builder.CONTROL_NUMBERS
        if number not in {"11900", "56500", "94000", "16900", "27720", "89900", "75000", "83800"}
    )
    kb_builder.CONTROL_NUMBERS = tuple(dict.fromkeys((*kb_builder.CONTROL_NUMBERS, "93100", "114000", "120000", "98000")))
    kb_builder.BUILDER_VERSION = "kb_release_v6_1_builder_2026_05_20"
    kb_builder.FRESHNESS_CHECK_DATE = "2026-05-20"

    build_result = kb_builder.build_kb_release_v3(
        run_id=args.run_id,
        handoff_dir=source_out,
        out_dir=args.release_out,
        handoff_out_dir=args.handoff_out,
    )

    semantic_report = run_kb_semantic_review(args.handoff_out, out_dir=args.handoff_out)
    copy_if_exists(args.handoff_out / "semantic_review.json", args.release_out / "semantic_review.json")
    copy_if_exists(args.handoff_out / "semantic_review.md", args.release_out / "semantic_review.md")

    create_not_run_smoke_summaries(args.smoke_dir)
    pack_result = build_distribution_packs(
        release_dir=args.handoff_out,
        full_release_dir=args.release_out,
        smoke_dir=args.smoke_dir,
        employee_out=args.employee_pack_out,
        bot_out=args.bot_pack_out,
    )
    write_diff_summary(args.release_out, args.handoff_out)

    result = {
        "schema_version": "kb_release_v6_3_build_result_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id,
        "source_out": str(args.source_out),
        "release_out": str(args.release_out),
        "handoff_out": str(args.handoff_out),
        "bot_pack_out": str(args.bot_pack_out),
        "employee_pack_out": str(args.employee_pack_out),
        "smoke_status": "not_run_by_instruction",
        "build_result": dict(build_result),
        "semantic_pass": bool(semantic_report.get("semantic_pass")),
        "semantic_blocking_findings": semantic_report.get("blocking_findings"),
        "pack_result": dict(pack_result),
    }
    (args.release_out / "v6_1_build_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if semantic_report.get("semantic_pass") else 2


def prepare_source_overlay(source_root: Path, source_out: Path) -> Path:
    source = source_root.expanduser().resolve(strict=False)
    target = source_out.expanduser().resolve(strict=False)
    if not source.exists():
        raise FileNotFoundError(source)
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
    return target


def patch_sources(source_root: Path) -> None:
    ensure_expected_source_files(source_root)
    facts_dir = source_root / "facts"
    foton_path = facts_dir / "facts_for_bot_FOTON.yaml"
    unpk_path = facts_dir / "facts_for_bot_UNPK.yaml"
    brand_rules_path = facts_dir / "brand_rules.yaml"
    bot_policy_path = facts_dir / "bot_policy.yaml"
    internal_path = facts_dir / "facts_internal_only.yaml"
    gold_answers_path = facts_dir / "gold_answers_v3.yaml"

    foton = load_yaml(foton_path)
    unpk = load_yaml(unpk_path)
    brand_rules = load_yaml(brand_rules_path)
    bot_policy = load_yaml(bot_policy_path)
    internal = load_yaml(internal_path)

    patch_foton_facts(foton)
    patch_unpk_facts(unpk)
    patch_brand_rules(brand_rules)
    patch_bot_policy(bot_policy)
    patch_internal_facts(internal)
    gold_answers = gold_answers_v3_payload()

    write_yaml(foton_path, foton)
    write_yaml(unpk_path, unpk)
    write_yaml(brand_rules_path, brand_rules)
    write_yaml(bot_policy_path, bot_policy)
    write_yaml(internal_path, internal)
    write_yaml(gold_answers_path, gold_answers)


def patch_foton_facts(data: dict[str, Any]) -> None:
    data["schema_version"] = "kb_facts_foton_v6_1_2026_05_20_team_answers"
    data["generated_at"] = "2026-05-20"
    remove_future_price_branches(data)
    data["presentation_format_facts_2026_05_21"] = {
        "status": "verified",
        "freshness_status": "document_verified",
        "brand": "foton",
        "source": "Презентации ЦДПО «Фотон» по онлайн-подготовке, очной подготовке и лагерю; цены, даты и адреса из презентаций не переносились.",
        "client_facts": {
            "lesson_load_by_age": {
                "client_safe_text": (
                    "В Фотоне нагрузка зависит от класса и формата: для 5-11 классов обычно 1 занятие в неделю по 3 академических часа "
                    "(очное занятие длится около 2 часов); для младших групп 1-4 очно и 3-4 онлайн — по 2 академических часа; "
                    "для 9 и 11 классов онлайн возможен формат 2 раза в неделю по 2 академических часа по будням."
                )
            },
            "levels_explained": {
                "client_safe_text": (
                    "В Фотоне есть три уровня подготовки: базовый — закрыть пробелы и уверенно идти по школьной программе; "
                    "продвинутый — углублённая программа и темы за пределами школы с элементами олимпиад; "
                    "олимпиадный — подготовка к перечневым олимпиадам и более сложным задачам. Уровень группы можно поменять, если ребёнку будет слишком легко или сложно."
                )
            },
            "subjects_by_class": {
                "client_safe_text": (
                    "По предметам в Фотоне: онлайн есть математика для 3-11 классов, информатика для 5-11 классов, физика для 7-11 классов; "
                    "очно есть математика для 3-11 классов, информатика для 5-11 классов, физика для 5-11 классов и русский язык для 9-11 классов."
                )
            },
            "student_account_access": {
                "client_safe_text": (
                    "У ученика есть личный кабинет на учебной платформе. Если пароль забыт, его восстанавливают через кнопку «Забыли пароль»; "
                    "центр не видит пароли учеников и не может назвать старый пароль."
                )
            },
            "placement_and_first_feedback": {
                "client_safe_text": (
                    "До старта ребёнок проходит вступительное тестирование и заполняет анкету — это помогает подобрать группу и адаптировать программу. "
                    "Первые 3-4 занятия преподаватель оценивает уровень и мотивацию, после чего можно дать первую обратную связь и при необходимости обсудить переход в другую группу."
                )
            },
            "knowledge_control": {
                "client_safe_text": (
                    "Контроль знаний строится не только на занятиях: есть вступительное и финальное тестирование, контрольные срезы примерно раз в 2-3 месяца, "
                    "обратная связь от преподавателя в личном кабинете в течение 2 недель. Для 9-11 классов предусмотрены пробные экзамены 2 раза в год, для 3-8 классов — олимпиадные активности."
                )
            },
            "online_lesson_format": {
                "client_safe_text": (
                    "Онлайн-занятия проходят в формате вебинаров на МТС Линк: ученики задают вопросы в чате, преподаватель может вызвать в голосовой или видео-чат, "
                    "а записи уроков доступны для пересмотра."
                )
            },
            "offline_lesson_format": {
                "client_safe_text": (
                    "Очные занятия проходят как семинары: теория и практика, разбор у доски, самостоятельная работа, конспекты. "
                    "Домашнее задание задаётся не всегда, а когда оно действительно нужно для закрепления материала."
                )
            },
            "course_rules_safe": {
                "client_safe_text": (
                    "На онлайн- и очных занятиях есть правила поведения и цифровой этикет. Это нужно, чтобы детям было комфортно учиться, задавать вопросы и не отвлекаться."
                )
            },
            "offline_organization": {
                "client_safe_text": (
                    "Перед очным стартом дату, время, адрес и кабинет присылают отдельным письмом. Основной канал информирования — email; "
                    "Telegram-чаты с преподавателем и администратором обычно подключают на 2-м занятии. С собой нужны письменные принадлежности и тетрадь."
                )
            },
            "attendance_tracking": {
                "client_safe_text": (
                    "По очным занятиям следят за посещаемостью: если ребёнок пропускает 2 занятия подряд без предупреждения, администратор связывается с родителями."
                )
            },
            "parent_access_offline": {
                "client_safe_text": (
                    "Вход родителей в здание школы может быть ограничен правилами площадки, поэтому ожидание обычно организуют рядом с площадкой."
                )
            },
            "online_technical_requirements": {
                "client_safe_text": (
                    "Для онлайн-занятий нужен стабильный интернет от 2,5 Мбит/с. Можно подключаться с телефона или планшета через приложение МТС Линк, "
                    "но полный функционал удобнее на компьютере в Chrome. Если есть техническая проблема, можно написать @support в чате занятия."
                )
            },
            "age_goals": {
                "client_safe_text": (
                    "Цели по возрастам разные: в 3-4 классах — интерес к предмету и логика; в 5-6 — математическая культура, сильные школы и элементы олимпиад; "
                    "в 7-8 — глубокая проработка и поступление в сильные школы; в 9-11 — подготовка к ОГЭ/ЕГЭ; олимпиадные группы готовят к перечневым олимпиадам."
                )
            },
        },
        "manager_only_facts": {
            "internal_only": True,
            "rules_sanction": {
                "client_safe_text": (
                    "В презентациях указано, что нарушение правил поведения может привести к отчислению с курса без возврата денег. "
                    "Бот не озвучивает это клиенту сам; при конфликте или вопросе о возврате маршрут только к менеджеру."
                )
            },
            "brand_domain_signal": {
                "client_safe_text": (
                    "В презентациях Фотона встречались edu@kmipt.ru и kmipt.tallanto.com. Это внутренний сигнал на проверку общего бэк-офиса; "
                    "клиенту бота Фотон давать только подтверждённые контакты Фотона, например edu@cdpofoton.ru."
                )
            },
        },
    }
    contacts = data.setdefault("contacts_foton", {})
    contacts["vk"] = "vk.ru/foton_edu"
    lvsh = data.setdefault("lvsh_mendeleevo_2026", {})
    pricing = lvsh.setdefault("pricing_2026", {})
    for key in ("current_price_valid_until", "current_price_status", "current_price_confirmed_by_dmitry"):
        pricing.pop(key, None)
    for key in ("main_min", "main_min_status", "main_min_note_internal", "booking_deposit"):
        pricing.pop(key, None)
    pricing["current_price"] = 93100
    pricing["client_safe_text_when_price_asked"] = "ЛВШ Менделеево у Фотона сейчас стоит 93 100 ₽. Полная стоимость — 98 000 ₽"
    pricing.pop("dolyami_lvsh", None)
    pricing.pop("tbank_installment_lvsh", None)
    pricing.pop("dolyami_short_courses", None)
    lvsh["payment_options_2026"] = {
        "client_safe_text": (
            "Для ЛВШ Фотона доступны варианты оплаты частями на 6, 10 или 12 месяцев, "
            "а также сервис Долями. Менеджер поможет выбрать способ оплаты и оформить его дистанционно."
        ),
        "note_internal": "Текущие условия подтверждены Дмитрием 2026-05-22; применимо к продуктам Фотона без разделения на старые условия по типу продукта.",
    }
    lvsh["payment_options_2026_internal"] = {
        "internal_only": True,
        "terms_months_options": [6, 10, 12],
        "dolyami_available": True,
        "client_safe_rule": "не обещать одобрение или точную комиссию без оформления",
    }
    accommodation = lvsh.setdefault("accommodation", {})
    accommodation["meals_description"] = "Шведский стол, ресторанный уровень"
    accommodation["security_24_7"] = True
    lvsh["transfer_from_moscow"] = {
        "available": True,
        "included_in_price": True,
        "client_safe_text": "Трансфер до ЛВШ Фотона включён в стоимость; ориентир места сбора — метро Ховрино, точные детали отправляем перед сменой.",
    }
    lvsh["documents_for_shift"] = {
        "client_safe_text": (
            "Перед сменой понадобятся договор, анкета и согласия, копия документа ребёнка, копия полиса, "
            "медицинская справка 079У и справка о санэпидокружении."
        ),
        "manager_should_send_full_list": True,
    }
    lvsh["parent_visit_and_communication"] = {
        "client_safe_text": (
            "Связаться с ребёнком можно; для 5-6 классов телефоны забирают с 23:00 до 14:00. "
            "Фото и видео со смены публикуются в Telegram-канале смены."
        )
    }
    lvsh["teachers_general_phrase"] = "Преподаватели из МГУ и МФТИ, вожатые из топовых вузов Москвы"
    city = data.setdefault("ls_city_2026_foton", {})
    city["free_morning_club"] = {
        "name": "Утренний клуб Предлёнка",
        "time": "09:45-11:45",
        "from_day": 2,
        "cost": "полностью бесплатно",
        "description_for_client": "Предлёнка начинается со второго дня занятий в летней городской школе Фотона. Предлёнка полностью бесплатно",
    }
    academic = data.setdefault("academic_year_2026_27", {})
    academic["start_by_location"] = {"moscow": "12-13 сентября 2026", "online": "19-20 сентября 2026"}
    academic["start"] = "12-20 сентября 2026 в зависимости от формата"
    social = data.get("results_social_proof")
    if isinstance(social, dict):
        social["industry_rating_2025"] = "Лидер отрасли 2025"
        social["industry_rating_2025_confirmation"] = "Подтверждено Дмитрием 2026-05-20"
    ensure_matkap_age_over_18_phrase(data)
    ensure_matkap_docs_manager_only(data)
    ensure_offline_group_size(data)
    patch_foton_installment_client_terms(data)
    intensives = data.get("intensives_2026")
    if isinstance(intensives, dict):
        oge = intensives.get("oge_foton")
        if isinstance(oge, dict):
            oge.pop("prices_before_2026_04_07", None)
            oge.pop("prices_after_2026_04_07", None)
            oge.pop("price_status", None)
            oge["price_note_internal"] = "Интенсив ОГЭ уже не актуален: цены не называем клиенту без проверки живого менеджера."
    legal = data.get("legal_entities")
    if isinstance(legal, dict):
        for entity in legal.get("entities", []):
            if isinstance(entity, dict) and entity.get("brand") == "foton":
                used_for = entity.get("used_for")
                if isinstance(used_for, list):
                    entity["used_for"] = [
                        "Курсы Фотон Москва (Верхняя Красносельская, 30)"
                        if str(item) == "Курсы Фотон Москва (Сретенка/Скорняжный)"
                        else item
                        for item in used_for
                    ]


def patch_unpk_facts(data: dict[str, Any]) -> None:
    data["schema_version"] = "kb_facts_unpk_v6_1_2026_05_20_team_answers"
    data["generated_at"] = "2026-05-20"
    remove_future_price_branches(data)
    prices = data.setdefault("prices_regular_2026_27", {})
    if isinstance(prices, dict):
        offline_1_4 = prices.setdefault("offline_1_4_class", {})
        if isinstance(offline_1_4, dict):
            offline_1_4.pop("location_only", None)
            offline_1_4["note_internal"] = "Очная площадка 1-4 классов хранится в locations_unpk, а не внутри price-ветки, чтобы адрес не превращался в цену."
        prices["patsayeva_2x_week"] = {
            "status": "removed_2026_05_20",
            "programs": [],
            "note_internal": "Программы 2 раза в неделю на Пацаева сняты. Если клиент спрашивает — передать менеджеру.",
        }
        online_phystech = prices.setdefault("online_olympiad_phystech_9_and_11", {})
        if isinstance(online_phystech, dict):
            online_phystech["status"] = "stale_previous_year_not_current"
            online_phystech["product"] = "Будничные онлайн-курсы прошлого учебного года"
            online_phystech.pop("semester", None)
            online_phystech.pop("year", None)
            online_phystech["note_internal"] = (
                "41 800/69 900 ₽ относились к будничным онлайн-курсам прошлого учебного года. "
                "На 2026/27 актуальных цен и расписания онлайн УНПК пока нет; клиенту суммы не называть."
            )
            online_phystech["bot_behavior_when_asked"] = (
                "По онлайн-курсам УНПК на 2026/27 актуальные цены и расписание ещё уточняются; "
                "менеджер проверит и свяжется с вами."
            )
        online_regular = prices.setdefault("online_5_11_class_regular", {})
        if isinstance(online_regular, dict):
            online_regular["status"] = "prices_and_schedule_pending_2026_27"
            online_regular["bot_behavior_when_asked"] = (
                "Да, онлайн-направление есть, но актуальные цены и расписание УНПК на 2026/27 пока уточняются; "
                "менеджер проверит условия под ваш класс и предмет."
            )
    lvsh = data.setdefault("lvsh_mendeleevo_2026", {})
    lvsh["transfer_from_moscow"] = {"available": True, "cost": "бесплатно", "confirmed_by_team": "2026-05-20"}
    pricing = lvsh.setdefault("pricing_2026", {})
    for key in ("current_price_valid_until", "current_price_status", "current_price_confirmed_by_dmitry"):
        pricing.pop(key, None)
    for key in ("main_with_25_pct_discount", "main_min", "main_min_status", "main_min_note_internal", "booking_deposit"):
        pricing.pop(key, None)
    pricing["main_full"] = 120000
    pricing["current_price"] = 114000
    pricing["client_safe_text_when_price_asked"] = (
        "ЛВШ Менделеево в УНПК сейчас стоит 114 000 ₽. "
        "Места уже почти распроданы, поэтому наличие и запись проверяет живой менеджер"
    )
    lvsh["availability_2026"] = {
        "status": "almost_sold_out",
        "client_safe_text": "По ЛВШ УНПК места уже почти распроданы; запись сейчас только через живого сотрудника.",
        "route_policy": "draft_for_manager",
        "confirmed_by_dmitry": "2026-05-22",
    }
    for shift in lvsh.get("smeny_2026", []):
        if isinstance(shift, dict) and "18-26 июля" in str(shift.get("dates") or ""):
            shift["brand_internal"] = "unpk_parallel_independent_smena"
            shift["note_internal"] = (
                "В эти же даты проходит независимая смена Фотона на той же базе; "
                "клиенту бота УНПК показывать только смену УНПК."
            )
        if isinstance(shift, dict) and "15-25 августа" in str(shift.get("dates") or ""):
            shift["status"] = "closed"
            shift["client_safe_text"] = "Августовская смена УНПК 15-25 августа закрыта."
    lvsh["booking_policy"] = {
        "client_safe_text": "Запись на ЛВШ УНПК сейчас проверяет живой менеджер.",
        "bot_must_not_say_note_internal": "Бот не говорит «места есть» и не пишет «можно забронировать автоматически».",
    }
    city = data.setdefault("ls_city_2026_unpk", {})
    city["dolgoprudny"] = {
        **as_dict(city.get("dolgoprudny")),
        "location": "Долгопрудный, Институтский пер., 9 (Главный корпус МФТИ)",
    }
    moscow = as_dict(city.get("moscow") or city.pop("moscow_ano", {}))
    city["moscow"] = {
        **moscow,
        "dates": moscow.get("dates") or "6-17 июля",
        "location": "Москва, Верхняя Красносельская ул., 30",
    }
    city["patsayeva"] = {
        "available": False,
        "note_internal": "2026-05-20: команда подтвердила, что на Пацаева летних школ нет.",
    }
    data["preschool_patsayeva"] = {
        "status": "removed_2026_05_20",
        "brand": "unpk",
        "prices": {},
        "programs": [],
        "note_internal": "Дошкольные программы закрываются у обоих брендов. Клиенту не называть старые тарифы; передать менеджеру.",
    }
    contacts = data.setdefault("contacts_unpk", {})
    contacts["email"] = "edu@kmipt.ru"
    locations = data.setdefault("locations_unpk", {})
    addresses = [item for item in locations.get("addresses", []) if "лобн" not in json.dumps(item, ensure_ascii=False).casefold()]
    locations["addresses"] = addresses
    locations["free_trial_offline"] = {
        "available": False,
        "offer": "Очного бесплатного пробного на Пацаева и в МФТИ сейчас нет. По онлайн-формату можно прислать фрагмент занятия.",
    }
    locations["online_trial_fragment"] = {
        "available": True,
        "client_safe_text": "По онлайн-формату можно прислать фрагмент занятия для знакомства с подачей и уровнем.",
    }
    academic = data.setdefault("academic_year_2026_27", {})
    academic["start_by_location"] = {
        "moscow": "12-13 сентября 2026",
        "online": "19-20 сентября 2026",
        "mfti_institutsky_9": "20 сентября 2026",
        "patsayeva": "26-27 сентября 2026",
    }
    academic["start"] = "12-27 сентября 2026 в зависимости от площадки"
    teachers_lobnya = data.get("teachers_lobnya")
    if isinstance(teachers_lobnya, dict):
        teachers_lobnya["status"] = "archived_2026_05_20"
        teachers_lobnya["internal_only"] = True
    social = data.get("results_social_proof")
    if isinstance(social, dict):
        social["total_alumni"] = "100 000 учеников"
        social["total_alumni_confirmation"] = "Подтверждено Дмитрием 2026-05-20"
    data["unavailable_programs_2026_27"] = {
        "client_safe_text": "Химия 10-11 и английский 1-4 в УНПК на 2026/27 не запускаются.",
        "note_internal": "Подтверждено Дмитрием 2026-05-22. Бот не предлагает химию 10-11 и английский 1-4 как доступные программы УНПК.",
    }
    tg_internal = data.get("tg_unpk_internal_2026_05_21")
    if isinstance(tg_internal, dict):
        facts = tg_internal.setdefault("client_safe_facts", {})
        if isinstance(facts, dict):
            facts.pop("online_trial_week", None)
            facts["online_trial_fragment"] = {
                "client_safe_text": "По онлайн-формату можно прислать фрагмент занятия для знакомства с подачей и уровнем.",
            }
            facts["unavailable_chemistry_english"] = {
                "client_safe_text": "Химия 10-11 и английский 1-4 на 2026/27 не запускаются.",
            }
    tg_verified = data.get("tg_unpk_verified_2026_05_21")
    if isinstance(tg_verified, dict):
        facts = tg_verified.setdefault("client_safe_facts", {})
        if isinstance(facts, dict):
            facts.pop("online_trial_week", None)
            facts["online_trial_fragment"] = {
                "client_safe_text": "По онлайн-формату можно прислать фрагмент занятия для знакомства с подачей и уровнем.",
            }
        client_facts = tg_verified.setdefault("client_facts", {})
        if isinstance(client_facts, dict):
            client_facts.pop("online_trial_week", None)
            client_facts["online_trial_fragment"] = {
                "client_safe_text": "По онлайн-формату можно прислать фрагмент занятия для знакомства с подачей и уровнем.",
            }
    ensure_matkap_age_over_18_phrase(data)
    ensure_matkap_docs_manager_only(data)
    ensure_offline_group_size(data)


def ensure_matkap_age_over_18_phrase(data: dict[str, Any]) -> None:
    matkap = data.setdefault("matkap", {})
    if not isinstance(matkap, dict):
        return
    client_text = matkap.setdefault("client_safe_text", {})
    if not isinstance(client_text, dict):
        return
    client_text["when_age_over_18"] = (
        "Если ученику уже 18 лет или больше, по возрастным условиям маткапитала есть ограничения. "
        "Уточню у менеджера, подходит ли ваш случай — он свяжется в течение рабочего дня."
    )


def ensure_matkap_docs_manager_only(data: dict[str, Any]) -> None:
    matkap = data.setdefault("matkap", {})
    if not isinstance(matkap, dict):
        return
    docs = matkap.get("required_docs")
    if docs is None:
        return
    if isinstance(docs, dict) and docs.get("internal_only"):
        return
    items = docs if isinstance(docs, list) else [docs]
    matkap["required_docs"] = {
        "internal_only": True,
        "client_facing": False,
        "bot_route": "manager_handoff_only",
        "items": items,
        "client_safe_summary": "Менеджер пришлёт перечень документов и поможет с оформлением через СФР.",
    }


def ensure_offline_group_size(data: dict[str, Any]) -> None:
    platform = data.setdefault("online_platform", {})
    if not isinstance(platform, dict):
        return
    platform["offline_group_size"] = "6-12 человек"


def patch_brand_rules(data: dict[str, Any]) -> None:
    data["schema_version"] = "brand_rules_v6_1_2026_05_20_team_answers"
    data["generated_at"] = "2026-05-20"
    forbidden = data.setdefault("forbidden_client_mentions", {})
    foton_terms = (((forbidden.setdefault("when_active_brand_is_foton", {})).setdefault("blocked_terms", [])))
    ensure_terms(foton_terms, ["edu@kmipt.ru", "Сретенка", "Сретенка, 20"])
    unpk_terms = (((forbidden.setdefault("when_active_brand_is_unpk", {})).setdefault("blocked_terms", [])))
    ensure_terms(unpk_terms, ["vk.ru/foton_edu", "foton_edu"])
    remove_terms_containing(unpk_terms, ["Верхняя Красносельская", "vk.com/kmipt_edu"])
    data["shared_brand_entities_allowed"] = [
        {"entity": "Менделеево", "contexts": ["lvsh_mendeleevo"]},
        {"entity": "Верхняя Красносельская ул., 30", "contexts": ["ls_city_2026_foton", "ls_city_2026_unpk"]},
    ]


def patch_bot_policy(data: dict[str, Any]) -> None:
    data["schema_version"] = "bot_policy_v6_1_2026_05_20_team_answers"
    data["generated_at"] = "2026-05-20"
    data["sales_playbook_from_calls_2026_05_21"] = {
        "status": "approved_by_dmitry_for_bot_use",
        "source_docx": "/Users/dmitrijfabarisov/Claude Projects/Foton/gold_candidates_paid_proxy_after_calls_tropina_kozlova_20260521_v3_rebuilt_current_runtime/SALES_PLAYBOOK_FROM_CALLS_2026-05-21.docx",
        "source_scope": "703 звонка Тропиной Анны и Козловой Екатерины, после которых есть подтверждённая платная учебная активность.",
        "use_as": "tone_and_dialog_strategy_not_price_or_schedule_source",
        "client_facing": False,
        "do_not_import_as_facts": [
            "цены из звонков",
            "даты из звонков",
            "расписание из звонков",
            "непроверенные условия возврата",
            "непроверенные дедлайны оплаты",
        ],
        "core_style": [
            "Тон спокойного заботливого администратора, а не продажника.",
            "По возможности использовать известный контекст клиента: имя, класс, предмет, прошлый вопрос или заявку.",
            "Сначала понять цель семьи: подтянуть базу, углубиться, готовиться к олимпиадам или экзаменам.",
            "Не вываливать прайс без контекста; сначала связать цену и формат с задачей ребёнка.",
            "Снижать тревогу родителя до возражения: объяснять уровни, переход между группами, записи, поддержку и право семьи решить.",
            "Давать честную мягкую срочность только на основе проверенного факта; не давить и не придумывать дедлайны.",
            "Заканчивать ответ понятным следующим шагом.",
            "Честно говорить об ограничениях вместо приукрашивания.",
        ],
        "answer_structure": [
            "короткое живое подтверждение вопроса",
            "прямой ответ по сути",
            "связка с целью ребёнка или контекстом семьи",
            "условия или варианты только из проверенных фактов",
            "один понятный следующий шаг",
        ],
        "safe_phrasing_patterns": {
            "goal_discovery": "Подскажите, что сейчас важнее: подтянуть школьную базу, углубиться или готовиться к олимпиадам/экзаменам?",
            "level_anxiety": "Уровень нужен, чтобы подобрать группу, а не чтобы «отсеять» ребёнка. Если после старта будет некомфортно, менеджер поможет подобрать другой уровень.",
            "mid_year_join": "Присоединиться можно не только с начала года: менеджер подскажет, как мягче войти в программу и что ребёнку нужно догнать.",
            "payment_friction": "Менеджер пришлёт удобный способ оплаты и подскажет, куда отправить чек, если он нужен для подтверждения.",
            "next_step": "Давайте зафиксируем класс, предмет и удобный формат — тогда подберём самый близкий вариант.",
        },
        "hard_limits": [
            "Не обещать возврат, перенос, место, группу, оплату или скидку без проверенного факта и разрешённого маршрута.",
            "В high-risk темах не применять продающие приёмы; безопасность и manager_only важнее полезности.",
            "Не сравнивать Фотон и УНПК и не использовать условия одного бренда в ответе другого.",
            "Не копировать фразы playbook дословно как скрипт; использовать как ориентир по тону и логике.",
        ],
    }
    routes = data.setdefault("theme_routes", {})
    installment = routes.setdefault("installment", {})
    unpk_specific = installment.setdefault("unpk_specific", {})
    unpk_specific["fallback_phrase"] = (
        "В УНПК можно платить помесячно, за семестр или за год. "
        "При оплате за семестр действует скидка 10%, за год - 14%. "
        "Если нужно растянуть оплату, менеджер подскажет варианты под вашу ситуацию."
    )
    complaint = routes.setdefault("complaint", {})
    complaint["bot_phrase_p0"] = "Ваше обращение принято. Передам его ответственному сотруднику, он свяжется с вами."
    complaint["p0_response_rule"] = (
        "При срабатывании любого p0_trigger бот отвечает строго bot_phrase_p0. "
        "Без слов «автоматический ответ», «бот» или «ИИ». Тема дальше — только менеджер."
    )
    routes["enrollment"] = {
        "risk": "low",
        "requires_brand_filter": True,
        "route": "draft_for_manager",
        "collect_required_fields": ["класс ребёнка", "предмет", "формат (онлайн или очно)", "email родителя", "ФИО ребёнка", "телефон родителя"],
        "collection_rule": "Перед запросом каждого поля бот проверяет, есть ли оно уже в контексте. Просим только недостающие данные.",
        "bot_response_deadline": "в ближайшее время",
    }
    routes["theme_routes_part2"] = {
        "materials_homework": {"route": "draft_for_manager"},
        "results_social_proof": {
            "route": "draft_for_manager",
            "notes": "Использовать только подтверждённые соцдоказательства: УНПК — 100 000 учеников; Фотон — Лидер отрасли 2025.",
        },
        "forgot_password": {"route": "draft_for_manager", "collect": ["email", "ФИО ребёнка"]},
        "missing_link": {"route": "draft_for_manager"},
        "adult_18_plus": {
            "route": "bot_answer_self_for_pilot",
            "notes": "Курсов 18+ нет у обоих брендов; если вопрос про маткапитал 19+, приоритет у matkap.when_age_over_18.",
        },
        "preschool": {"route": "bot_answer_self_for_pilot", "notes": "Дошкольные программы закрываются у обоих брендов."},
        "reschedule_lesson": {"route": "draft_for_manager", "collect": ["ФИО ребёнка"]},
        "change_group": {"route": "draft_for_manager", "manager_must_check": ["наличие мест в целевой группе"]},
        "illness": {
            "route": "draft_for_manager",
            "rules": {
                "online": "Бот может сказать, что запись урока сохранится.",
                "offline": "Перенос или компенсация — всегда менеджер.",
                "certificate": "Справка от ребёнка не нужна; 079у только для лагеря.",
            },
        },
        "b2b": {"route": "manager_only", "collect": ["название школы/организации", "контактное лицо"], "deadline": "в течение рабочего дня"},
        "group_discount": {"route": "draft_for_manager", "notes": "Для школ/классов — handoff, бот процент не называет."},
        "free_trial_online": {"route": "bot_answer_self", "applies_to_brands": ["foton", "unpk"]},
        "free_trial_offline_foton": {"route": "draft_for_manager", "applies_to_brands": ["foton"]},
        "free_trial_offline_unpk": {"route": "draft_for_manager", "applies_to_brands": ["unpk"]},
        "diagnostic_test": {"route": "draft_for_manager", "notes": "Диагностика после оплаты, не предлагать до оплаты."},
        "combined_question": {
            "route": "bot_answer_self_for_pilot",
            "rules": {
                "forbid_future_prices": True,
                "combined_with_high_risk": {
                    "trigger_topics": ["refund", "complaint", "legal_threat"],
                    "override_route": "manager_only",
                    "client_send": False,
                    "manager_draft_allowed": True,
                },
            },
        },
    }
    post_filter = data.setdefault("post_filter_draft_text", {})
    forbidden_any = post_filter.setdefault("forbidden_in_any_brand", [])
    ensure_terms(forbidden_any, ["Это автоматический ответ", "автоматический ответ", "до 36 месяцев", "3-36 месяцев"])
    post_filter["no_future_prices"] = {
        "rule": "Бот называет только актуальные цены. После периодов повышения цены — handoff менеджеру за актуальной цифрой.",
        "forbid_phrases": ["будущая цена", "после повышения", "цена вырастет до"],
        "notes": "Если клиент спрашивает будущую цену, не угадываем и не называем число.",
    }
    data["gold_answers_v3"] = gold_answers_v3_payload()
    data["dialog_quality_rules_v3"] = {
        "status": "approved_by_dmitry_2026_05_22",
        "source": GOLD_ANSWERS_V3_DOCX,
        "use_as": "quality_reference_not_literal_script",
        "rules": [
            "Говорить тепло и по-человечески, как хороший консультант, а не как строгий регламент.",
            "Сначала отвечать на прямой вопрос, затем давать один понятный следующий шаг.",
            "Если факта нет, не выдумывать: честно объяснить, от чего зависит ответ, и задать один полезный вопрос.",
            "Не приглашать на очную встречу по умолчанию: запись и оформление дистанционные, встреча только по согласованию.",
            "В составном вопросе отвечать на безопасную часть, но если есть P0/high-risk — маршрут к менеджеру.",
            "Gold-ответы задают тон и структуру, но не являются дословным скриптом.",
        ],
    }


def patch_internal_facts(data: dict[str, Any]) -> None:
    data["schema_version"] = "kb_facts_internal_v6_1_2026_05_20_team_answers"
    data["generated_at"] = "2026-05-20"
    analytics = data.setdefault("returns_analytics_25_26", {})
    analytics["insight_for_bot"] = (
        "2026-05-20 команда поправила: диагностика идёт ПОСЛЕ оплаты и включена в стоимость обучения. "
        "После диагностики — распределение на учебные группы по уровню. Бот не предлагает диагностику как способ снизить риск возврата ДО оплаты."
    )
    analytics["client_facing"] = False


def register_gold_answers_source() -> None:
    kb_builder.SOURCE_FILES.setdefault(
        "gold_answers_v3",
        {
            "filename": "gold_answers_v3.yaml",
            "source_id": "dmitry_approved:gold_answers_v3_2026_05_22",
            "kind": "yaml",
            "brand": "brand_neutral",
            "description": "Эталонные ответы и правила живого тона для Telegram-бота.",
        },
    )


def patch_foton_installment_client_terms(data: dict[str, Any]) -> None:
    installment = data.setdefault("installment", {})
    if not isinstance(installment, dict):
        return
    safe_text = (
        "В Фотоне можно оплатить обучение частями: доступны варианты на 6, 10 или 12 месяцев, "
        "а также сервис Долями. Это относится к очным и онлайн-курсам, ЛВШ, ЛШ и другим программам Фотона. "
        "Конкретные условия и оформление зависят от выбранного способа оплаты; менеджер поможет подобрать удобный вариант."
    )
    installment["client_confirmed_terms"] = {
        "status": "approved_by_dmitry_2026_05_22",
        "client_safe_text": safe_text,
        "bot_must_not_say_note_internal": "Не использовать старые разные условия по типам продукта; клиенту говорить единое подтверждённое правило: 6, 10 или 12 месяцев и Долями для продуктов Фотона.",
    }
    installment["term_months"] = {
        "client_facing": False,
        "internal_note": "Старая широкая формулировка срока рассрочки не используется в клиентских ответах после подтверждения Дмитрия 2026-05-22.",
    }
    services = installment.get("services")
    if isinstance(services, dict):
        rassrochka = services.get("rassrochka")
        if isinstance(rassrochka, dict):
            rassrochka["term_months"] = "6, 10 или 12"
            rassrochka["term_months_confirmed_by_dmitry"] = "2026-05-22"
            rassrochka["applies_to_all_foton_products"] = True
            rassrochka.pop("lvsh_terms_months_options", None)
            rassrochka.pop("lvsh_commission_range", None)
            rassrochka.pop("lvsh_note_internal", None)
            rassrochka["note_internal"] = "2026-05-22: действует для очных/онлайн курсов, ЛВШ, ЛШ и других программ Фотона."
        dolyami = services.get("dolyami")
        if isinstance(dolyami, dict):
            dolyami["available"] = True
            dolyami["applies_to_all_foton_products"] = True
            dolyami.pop("best_for", None)
            dolyami["products_scope_client_safe"] = "Продукты Фотона: очные и онлайн-курсы, ЛВШ, ЛШ и другие программы"
            dolyami.pop("parts", None)
            dolyami.pop("interest_for_client", None)
            dolyami["note_internal"] = "2026-05-22: Долями разрешён как вариант оплаты для продуктов Фотона; конкретные условия оформляет менеджер."
        six_twelve = services.get("rassrochka_6_12_months")
        if isinstance(six_twelve, dict):
            six_twelve["terms_months_options"] = [6, 10, 12]
            six_twelve["applies_to_all_foton_products"] = True
            six_twelve["bot_behavior"] = (
                "Бот может сказать, что доступны варианты на 6, 10 или 12 месяцев; "
                "конкретные условия зависят от выбранного способа оплаты."
            )
    products = installment.get("products")
    if isinstance(products, dict):
        for product in products.values():
            if isinstance(product, dict):
                product["term_months"] = "6, 10 или 12"
                product["term_months_confirmed_by_dmitry"] = "2026-05-22"
    client_text = installment.setdefault("client_safe_text", {})
    if isinstance(client_text, dict):
        client_text["when_asked"] = safe_text
    installment.pop("forbidden_client_claims", None)


def gold_answers_v3_payload() -> dict[str, Any]:
    return {
        "schema_version": "gold_answers_v3_2026_05_22",
        "status": "approved_by_dmitry_for_bot_quality",
        "source_docx": GOLD_ANSWERS_V3_DOCX,
        "client_facing": False,
        "use_as": "answer_quality_tone_and_confirmed_business_rules_not_raw_script",
        "global_rules": [
            "Отвечать тепло, живо и по делу: как внимательный консультант, которому важно помочь семье.",
            "Не звучать как строгая формальная организация или нейросеть.",
            "Сначала прямой ответ, затем один безопасный следующий шаг.",
            "Не выдумывать цены, даты, наличие мест, сроки связи и расписание.",
            "Если точного факта нет, дать полезный общий ответ без цифр и задать 1-2 уточнения.",
            "Запись и оформление по умолчанию дистанционные; очная встреча только по согласованию.",
            "Не смешивать бренды Фотон и УНПК МФТИ.",
            "Gold-ответы — эталон структуры и тона, не дословный скрипт.",
        ],
        "confirmed_rules": {
            "foton_installment": "В Фотоне доступны варианты оплаты частями на 6, 10 или 12 месяцев, а также сервис Долями. Это правило действует для очных и онлайн-курсов, ЛВШ, ЛШ и других программ Фотона.",
            "unpk_installment": "Помесячно, семестр или год; скидка 10% за семестр и 14% за год; банковскую рассрочку не обещать.",
            "future_price": "Можно сказать, что цена скоро подрастёт, но не называть дату или будущую сумму без подтверждения.",
            "remote_enrollment_default": "Запись и оформление дистанционные; встреча с сотрудником только по согласованию.",
            "foton_trial": "По онлайн-формату можно прислать фрагмент занятия; бесплатность и лимит бесплатных бот не озвучивает.",
            "unpk_trial": "Очного пробного на Пацаева и в МФТИ сейчас нет; по онлайн-формату можно прислать фрагмент занятия.",
            "foton_moscow_address": "Верхняя Красносельская, 30 — основной московский адрес Фотона для регулярных занятий и летней школы.",
            "unpk_moscow_regular": "Регулярные курсы УНПК в Москве — Сретенка, 20; Верхняя Красносельская, 30 только для летней школы/городского лагеря.",
            "camp_question_key": "По лагерям уточнять класс ребёнка, а не возраст.",
            "foton_camps": "ЛВШ Менделеево у Фотона сейчас стоит 93 100 ₽; полная стоимость — 98 000 ₽.",
            "unpk_lvsh_price": "ЛВШ Менделеево в УНПК сейчас стоит 114 000 ₽; полная стоимость — 120 000 ₽; места почти распроданы, запись проверяет живой менеджер.",
            "unpk_online_prices": "41 800/69 900 ₽ — старые цены будничных онлайн-курсов прошлого учебного года; актуальных цен и расписания онлайн УНПК на 2026/27 пока нет.",
            "unpk_unavailable_subjects": "Химия 10-11 и английский 1-4 на 2026/27 не запускаются.",
            "parallel_lvsh_shift": "18-26 июля у Фотона и УНПК проходят независимые смены на одной базе; бот каждого бренда говорит только про свой бренд.",
            "camp_seats": "Бот не говорит «места есть»; формулировка только «проверим наличие / подберём смену / закрепим заявку».",
        },
        "topics": {
            "pricing": {
                "foton": {
                    "gold_answer_example": (
                        "Да, сориентирую. Для 7 класса очно сейчас семестр стоит 44 600 ₽, год — 74 500 ₽. "
                        "Цена скоро подрастёт, поэтому если формат подходит, лучше закрепить текущие условия. "
                        "Напишите, пожалуйста, предмет — подберём ближайшую группу."
                    ),
                    "must_include": ["цена", "период оплаты", "следующий шаг"],
                    "must_not_include": ["будущая цена", "точная дата повышения без факта"],
                },
                "unpk": {
                    "gold_answer_example": (
                        "Да, подскажу. Для 8 класса очно в УНПК сейчас семестр стоит 49 000 ₽, год — 82 000 ₽. "
                        "Цена скоро подрастёт, поэтому текущие условия лучше закрепить заранее. "
                        "Какой предмет и уровень подготовки рассматриваете?"
                    ),
                    "must_include": ["цена", "период оплаты", "следующий шаг"],
                    "must_not_include": ["Фотон", "Долями", "Т-Банк"],
                },
            },
            "installment": {
                "foton": {
                    "gold_answer_example": (
                        "Да, в Фотоне можно оплатить обучение частями: доступны варианты на 6, 10 или 12 месяцев, "
                        "а также сервис Долями. Это относится к очным и онлайн-курсам, ЛВШ, ЛШ и другим программам Фотона. "
                        "Конкретные условия и оформление зависят от выбранного способа оплаты; менеджер поможет подобрать удобный вариант."
                    ),
                    "must_include": ["6, 10 или 12 месяцев", "Долями"],
                    "must_not_include": ["до 36 месяцев", "3-36 месяцев", "3, 6 или 10 месяцев", "4 части"],
                },
                "unpk": {
                    "gold_answer_example": (
                        "В УНПК можно платить помесячно, за семестр или за год. "
                        "При оплате за семестр действует скидка 10%, за год — 14%. "
                        "Если нужно растянуть оплату, менеджер подскажет варианты под вашу ситуацию."
                    ),
                    "must_include": ["помесячно", "10%", "14%"],
                    "must_not_include": ["Т-Банк", "Долями", "Фотон"],
                },
            },
            "trial_class": {
                "foton": {
                    "gold_answer_example": (
                        "По онлайн-формату можно прислать фрагмент занятия, чтобы вы посмотрели подачу и уровень. "
                        "Сначала подберём класс, предмет и формат, а менеджер отправит подходящий материал."
                    ),
                    "must_not_include": ["бесплатное", "сколько угодно", "очное пробное"],
                },
                "unpk": {
                    "gold_answer_example": (
                        "Очного пробного на Пацаева и в МФТИ сейчас нет. "
                        "По онлайн-формату можно прислать фрагмент занятия, чтобы вы посмотрели подачу и уровень."
                    ),
                    "must_not_include": ["бесплатное очное пробное", "пробная неделя"],
                },
            },
            "platform_records": {
                "common": {
                    "gold_answer_example": (
                        "Онлайн-занятия проходят в МТС Линк, записи уроков доступны для пересмотра. "
                        "Если ребёнок пропустит занятие, можно вернуться к записи и материалам."
                    ),
                    "must_not_include": ["Zoom", "срок хранения без факта"],
                }
            },
            "schedule_groups": {
                "common": {
                    "gold_answer_example": (
                        "Расписание зависит от класса, предмета, формата и площадки. "
                        "Напишите класс ребёнка и предмет — подберём ближайший удобный вариант."
                    ),
                    "must_not_include": ["точный день без факта", "места есть"],
                }
            },
            "matkap": {
                "common": {
                    "gold_answer_example": (
                        "Да, работаем с федеральным материнским капиталом. СФР рассматривает заявление до 10 рабочих дней, "
                        "перевод занимает ещё до 5 рабочих дней; решение принимает СФР. "
                        "Менеджер пришлёт перечень документов и поможет с оформлением."
                    ),
                    "must_include": ["федеральный", "СФР", "до 15 рабочих дней"],
                    "must_not_include": ["региональный принимаем", "одобрение гарантировано"],
                }
            },
            "tax": {
                "common": {
                    "gold_answer_example": (
                        "По налоговому вычету можно вернуть до 14 300 ₽ в год: это 13% с расходов до 110 000 ₽. "
                        "За 2023 год и ранее лимит был 50 000 ₽ — возврат до 6 500 ₽. "
                        "Справку готовим до 10 рабочих дней, решение принимает ФНС."
                    ),
                    "must_include": ["14 300 ₽", "110 000 ₽", "ФНС"],
                    "must_not_include": ["ФНС точно вернёт"],
                }
            },
            "addresses": {
                "foton": {
                    "gold_answer_example": (
                        "В Москве Фотон занимается на Верхней Красносельской, 30, рядом с метро Красносельская. "
                        "Запись и оформление можно пройти дистанционно."
                    ),
                    "must_include": ["Верхняя Красносельская, 30"],
                },
                "unpk": {
                    "gold_answer_example": (
                        "Регулярные занятия УНПК в Москве проходят на Сретенке, 20. "
                        "Также есть площадки в Долгопрудном: МФТИ, Институтский переулок, 9 и Пацаева, 7к1."
                    ),
                    "must_include": ["Сретенка, 20"],
                    "must_not_include": ["Фотон"],
                },
            },
            "discounts": {
                "foton": {
                    "gold_answer_example": (
                        "Есть скидка на второй и последующий предмет одного ребёнка: 20% очно и 30% онлайн. "
                        "Для многодетных семей — 10% по удостоверению, скидки не суммируются."
                    ),
                    "must_include": ["20%", "30%", "не суммируются"],
                },
                "unpk": {
                    "gold_answer_example": (
                        "В УНПК на второй предмет одного ребёнка действует скидка 20%. "
                        "При оплате за семестр — 10%, за год — 14%; скидки не суммируются."
                    ),
                    "must_include": ["20%", "10%", "14%"],
                    "must_not_include": ["30%", "Долями"],
                },
            },
            "camps": {
                "foton": {
                    "gold_answer_example": (
                        "ЛВШ Менделеево у Фотона сейчас стоит 93 100 ₽. "
                        "Полная стоимость — 98 000 ₽. "
                        "Напишите класс ребёнка — подберём подходящую смену и проверим наличие мест."
                    ),
                    "must_include": ["93 100 ₽"],
                    "must_not_include": ["места есть", "старая минимальная цена как текущая", "возраст"],
                },
                "unpk": {
                    "gold_answer_example": (
                        "ЛВШ Менделеево в УНПК сейчас стоит 114 000 ₽, полная стоимость — 120 000 ₽. "
                        "Места уже почти распроданы, поэтому запись проверяет живой менеджер. "
                        "Напишите класс ребёнка — менеджер проверит, можем ли ещё закрепить место."
                    ),
                    "must_include": ["114 000 ₽", "120 000 ₽", "класс"],
                    "must_not_include": ["места есть", "старая минимальная цена как текущая", "возраст"],
                },
            },
        },
    }


def remove_future_price_branches(value: Any) -> None:
    if isinstance(value, dict):
        for key in list(value):
            if "after_2026_" in str(key):
                value.pop(key, None)
            else:
                remove_future_price_branches(value[key])
    elif isinstance(value, list):
        for item in value:
            remove_future_price_branches(item)


def ensure_expected_source_files(source_root: Path) -> None:
    """The v3 builder treats missing Claude markdown files as formal blockers."""
    required = {
        "UPDATE_REPORT_2026-05-19.md": (
            "# UPDATE REPORT 2026-05-19\n\n"
            "Этот файл добавлен сборщиком v6.1 как техническая заглушка для воспроизводимого source registry.\n\n"
            "Актуальные подтверждения команды для v6.1 зафиксированы в:\n\n"
            "- docs/TZ_KB_RELEASE_V6_1_TEAM_ANSWERS_CLEAN_2026-05-20.md\n"
            "- facts/*.yaml после применения scripts/build_kb_release_v6_1_team_answers.py\n\n"
            "Файл не является самостоятельным источником клиентских фактов.\n"
        )
    }
    for rel, content in required.items():
        path = source_root / rel
        if not path.exists():
            path.write_text(content, encoding="utf-8")


def create_not_run_smoke_summaries(root: Path) -> None:
    for brand in ("foton", "unpk"):
        target = root / brand
        target.mkdir(parents=True, exist_ok=True)
        (target / "stage6_eval_summary.json").write_text(
            json.dumps(
                {
                    "brand": brand,
                    "status": "not_run_by_instruction",
                    "rows_total": 0,
                    "errors": 0,
                    "brand_separation_violation": 0,
                    "note": "MEGA and live smoke were intentionally not run before final KB corrections.",
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )


def write_diff_summary(release_out: Path, handoff_out: Path) -> None:
    text = """# DIFF v4 -> v6.1

## Added or changed

- Foton VK changed to `vk.ru/foton_edu`.
- UNPK email `edu@kmipt.ru` added.
- UNPK city summer school Moscow address changed to `Верхняя Красносельская ул., 30`.
- UNPK Lobnya removed from client facts.
- UNPK Patsayeva 2x/week removed from client facts.
- UNPK preschool prices removed from client facts.
- Future price branches after known price increase dates removed from client-facing KB.
- P0 complaint phrase kept neutral without `Это автоматический ответ` or `зафиксировано`.
- UNPK installment fallback mentions confirmed discounts: 10% for semester and 14% for year.
- Offline mini-group size `6-12 человек` added for both brands.
- Matkap required document list is manager-only; client-safe text says that manager will send the list.
- Shared physical entities recorded for `Менделеево` and `Верхняя Красносельская ул., 30`.
- Gold answers v3 added as a structured source and bot/employee pack artifact.
- Foton installment client-facing term synchronized to `6, 10 или 12 месяцев` plus `Долями` for all Foton products; old split by product type is blocked and removed from client-safe facts.
- Bot prompt and guards updated for warm human tone, composite questions, remote enrollment default, and camp grade-vs-age clarification.

## Not run by instruction

- Full MEGA smoke.
- Small live `codex exec` smoke.
- Any live write to AMO/CRM/Tallanto.
"""
    (release_out / "DIFF_v4_vs_v6_1.md").write_text(text, encoding="utf-8")
    (handoff_out / "DIFF_v4_vs_v6_1.md").write_text(text, encoding="utf-8")


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in {path}")
    return payload


def write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(yaml.safe_dump(dict(payload), allow_unicode=True, sort_keys=False), encoding="utf-8")


def copy_if_exists(source: Path, target: Path) -> None:
    if source.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def ensure_terms(items: list[Any], terms: Sequence[str]) -> None:
    existing = {str(item).casefold() for item in items}
    for term in terms:
        if term.casefold() not in existing:
            items.append(term)
            existing.add(term.casefold())


def remove_terms_containing(items: list[Any], needles: Sequence[str]) -> None:
    lowered = [needle.casefold() for needle in needles]
    items[:] = [item for item in items if not any(needle in str(item).casefold() for needle in lowered)]


def as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


if __name__ == "__main__":
    raise SystemExit(main())
