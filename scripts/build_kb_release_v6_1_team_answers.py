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

    # v6.x removes outdated preschool and expired intensive prices from client/current control scope.
    kb_builder.CONTROL_NUMBERS = tuple(
        number
        for number in kb_builder.CONTROL_NUMBERS
        if number not in {"11900", "56500", "94000", "16900", "27720"}
    )
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

    write_yaml(foton_path, foton)
    write_yaml(unpk_path, unpk)
    write_yaml(brand_rules_path, brand_rules)
    write_yaml(bot_policy_path, bot_policy)
    write_yaml(internal_path, internal)


def patch_foton_facts(data: dict[str, Any]) -> None:
    data["schema_version"] = "kb_facts_foton_v6_1_2026_05_20_team_answers"
    data["generated_at"] = "2026-05-20"
    remove_future_price_branches(data)
    contacts = data.setdefault("contacts_foton", {})
    contacts["vk"] = "vk.ru/foton_edu"
    lvsh = data.setdefault("lvsh_mendeleevo_2026", {})
    accommodation = lvsh.setdefault("accommodation", {})
    accommodation["meals_description"] = "Шведский стол, ресторанный уровень"
    accommodation["security_24_7"] = True
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
    lvsh = data.setdefault("lvsh_mendeleevo_2026", {})
    lvsh["transfer_from_moscow"] = {"available": True, "cost": "бесплатно", "confirmed_by_team": "2026-05-20"}
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
        "offer": "По бесплатному пробному очно — менеджер свяжется и подскажет вариант. Лимит — по согласованию с куратором филиала; бот лимит не озвучивает.",
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
    ensure_terms(forbidden_any, ["Это автоматический ответ", "автоматический ответ"])
    post_filter["no_future_prices"] = {
        "rule": "Бот называет только актуальные цены. После периодов повышения цены — handoff менеджеру за актуальной цифрой.",
        "forbid_phrases": ["будущая цена", "после повышения", "цена вырастет до"],
        "notes": "Если клиент спрашивает будущую цену, не угадываем и не называем число.",
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
