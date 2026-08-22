#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, re, shutil, sys, zipfile
from collections import Counter, defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_kb_distribution_packs import clean_text, load_json, write_csv, write_json, write_jsonl
DEFAULT_SNAPSHOT = ROOT / "product_data/knowledge_base/kb_release_20260813_v6_8_owner_approved/kb_release_v3_snapshot.json"
DEFAULT_OUT = ROOT / "product_data/knowledge_base/kb_release_20260813_v6_8_partner_pack"
FIELDS = ("fact_id", "brand", "fact_type", "product", "client_safe_text", "risk_level", "freshness_status", "valid_from", "valid_until", "bot_template_required")
VOLATILE_TYPES = frozenset({"price", "discount", "schedule", "payment", "contact", "contacts"})
BLOCKED_MARKERS = ("/Users/", ".mango_local", "source_path", "source_sha256", "manager_check_text", "manager_display_text", "internal_text", "safety_block_reasons", "Claude", "Codex", "GPT")
EXTERNAL_EXCLUDE_MARKERS = ("подтверждено дмитрием",)
COMPLETED_HANDOFF_MARKERS = ("я уже передал", "мы уже передали")
PARTNER_HANDOFF_TEXT = "По возврату после оплаты нужен расчёт менеджера: он проверит курс, период, оплату и условия договора и назовёт точную сумму."
MANAGER_ACTION_MARKERS = ("менеджер передаст", "менеджер проверит", "менеджер отправит", "менеджер пришлёт", "менеджер свяжется", "менеджер уточнит", "менеджер рассчитает", "менеджер оформит", "менеджер перезвонит", "менеджер подберёт")
CLIENT_CONTACT_RE = re.compile(r"(?:(?:\+7|8)[\s\-()]*(?:\d[\s\-()]*){10}|\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b)")
REFUND_MARKERS = ("возврат", "возвращается", "вернуть за него", "средства не сгорают")
def main() -> int:
    parser = argparse.ArgumentParser(description="Build an external partner-safe pack from canonical KB v6.8.")
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--guide-docx", type=Path)
    args = parser.parse_args()
    print(json.dumps(build_partner_pack(args.snapshot, args.out, guide_docx=args.guide_docx), ensure_ascii=False, indent=2))
    return 0
def build_partner_pack(snapshot_path: Path, out_dir: Path, *, guide_docx: Path | None = None) -> dict[str, object]:
    snapshot_path = snapshot_path.expanduser().resolve(strict=True)
    out = out_dir.expanduser().resolve(strict=False)
    if "stable_runtime" in out.parts:
        raise ValueError("Partner pack must not be written under stable_runtime")
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Partner output must be empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    snapshot = load_json(snapshot_path)
    facts = snapshot.get("facts") or []
    if not isinstance(facts, list):
        raise ValueError("Snapshot facts must be a list")
    records: list[dict[str, object]] = []
    source_client_safe = external_excluded = action_claims_neutralized = 0
    for fact in facts:
        if not isinstance(fact, dict) or fact.get("allowed_for_client_answer") is not True:
            continue
        if fact.get("internal_only") or fact.get("forbidden_for_client") or not clean_text(fact.get("client_safe_text")):
            raise ValueError(f"Invalid client-safe permission contract: {fact.get('fact_id', '<unknown>')}")
        source_client_safe += 1
        brand = str(fact.get("brand") or "")
        if brand not in {"foton", "unpk"}:
            raise ValueError(f"Unsupported partner brand: {brand}")
        client_text = clean_text(fact.get("client_safe_text"))
        if any(marker in client_text.casefold() for marker in EXTERNAL_EXCLUDE_MARKERS):
            external_excluded += 1
            continue
        record = {field: fact.get(field, "") for field in FIELDS}
        record["bot_template_required"] = bool(fact.get("bot_template_required"))
        if "это часы связи, а не расписание занятий групп" in client_text.casefold():
            record["client_safe_text"] = f"{'Фотон' if brand == 'foton' else 'УНПК МФТИ'}: актуальные часы связи проверьте на официальной странице контактов; по будням и выходным они могут отличаться."
        completed_handoff = any(marker in client_text.casefold() for marker in COMPLETED_HANDOFF_MARKERS)
        if completed_handoff:
            record["client_safe_text"] = PARTNER_HANDOFF_TEXT
            action_claims_neutralized += 1
        requires_manager = completed_handoff or any(marker in client_text.casefold() for marker in MANAGER_ACTION_MARKERS + REFUND_MARKERS)
        record["usage_mode"] = "manager_review_required" if requires_manager or fact.get("route_policy") != "bot_answer_self_for_pilot" else "answer_source"
        record["refresh_before_launch"] = completed_handoff or any(marker in client_text.casefold() for marker in REFUND_MARKERS) or str(fact.get("fact_type") or "") in VOLATILE_TYPES or bool(CLIENT_CONTACT_RE.search(client_text))
        records.append(record)
    records.sort(key=lambda item: (str(item["brand"]), str(item["fact_type"]), str(item["fact_id"])))
    by_brand = {brand: [item for item in records if item["brand"] == brand] for brand in ("foton", "unpk")}
    for brand, items in by_brand.items():
        write_jsonl(out / f"kb_{brand}.jsonl", items)
        (out / f"KB_{brand.upper()}.md").write_text(render_brand_markdown(brand, items, drafts_only=True), encoding="utf-8")
        autonomous = [item for item in items if item["usage_mode"] == "answer_source"]
        (out / f"KB_{brand.upper()}_AUTO.md").write_text(render_brand_markdown(brand, autonomous, drafts_only=False), encoding="utf-8")
    write_csv(out / "KB_ALL.csv", records)
    (out / "README.md").write_text(render_readme(records), encoding="utf-8")
    (out / "SYSTEM_PROMPT_TEMPLATE.md").write_text(SYSTEM_PROMPT_TEMPLATE, encoding="utf-8")
    properties = {field: {"type": "string"} for field in FIELDS}
    properties.update({"brand": {"enum": ["foton", "unpk"]}, "usage_mode": {"enum": ["answer_source", "manager_review_required"]}, "bot_template_required": {"type": "boolean"}, "refresh_before_launch": {"type": "boolean"}, "valid_from": {"type": "string", "pattern": r"^$|^\d{4}-\d{2}-\d{2}$"}, "valid_until": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"}})
    write_json(out / "schema.json", {"$schema": "https://json-schema.org/draft/2020-12/schema", "$id": "kb_partner_fact_v1", "type": "object", "additionalProperties": False, "required": list(records[0]) if records else [], "properties": properties})
    if guide_docx:
        shutil.copy2(guide_docx.expanduser().resolve(strict=True), out / "README_FOR_PARTNERS.docx")
    content_files = sorted(path for path in out.iterdir() if path.is_file())
    joined = "\n".join(readable_text(path) for path in content_files)
    leaked = [marker for marker in BLOCKED_MARKERS if marker.casefold() in joined.casefold()]
    if leaked:
        raise ValueError(f"Partner leak gate failed: {leaked}")
    hashes = {path.name: sha256(path) for path in content_files}
    counts = Counter(str(item["brand"]) for item in records)
    manifest = {
        "schema_version": "kb_partner_pack_v1",
        "source_release": snapshot.get("run_id"),
        "source_snapshot_sha256": sha256(snapshot_path),
        "source_facts_total": len(facts),
        "source_client_safe_facts": source_client_safe,
        "facts_total": len(records),
        "facts_by_brand": dict(counts),
        "usage_modes": dict(Counter(str(item["usage_mode"]) for item in records)),
        "excluded_non_client_safe_facts": len(facts) - source_client_safe,
        "excluded_by_external_policy": external_excluded,
        "completed_action_claims_neutralized": action_claims_neutralized,
        "public_business_contacts_included": True,
        "volatile_facts_require_refresh_before_launch": True,
        "files": hashes,
    }
    write_json(out / "manifest.json", manifest)
    checksum_files = sorted(path for path in out.iterdir() if path.is_file() and path.name != "SHA256SUMS.txt")
    (out / "SHA256SUMS.txt").write_text("".join(f"{sha256(path)}  {path.name}\n" for path in checksum_files), encoding="utf-8")
    return manifest
def render_brand_markdown(brand: str, records: list[dict[str, object]], *, drafts_only: bool) -> str:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[str(record["fact_type"])].append(record)
    mode = "Полная база только для режима черновиков: часть фактов требует обязательной проверки менеджером." if drafts_only else "Автономный поднабор: содержит только answer_source. Перед запуском всё равно проверьте актуальность."
    lines = [f"# База знаний: {'Фотон' if brand == 'foton' else 'УНПК МФТИ'}", "", "Используйте только для указанного бренда.", mode, ""]
    for fact_type, items in sorted(grouped.items()):
        lines.extend([f"## {fact_type}", ""])
        for item in items:
            suffix = (" [перепроверить перед запуском]" if item["refresh_before_launch"] else "") + (" Проверка менеджером обязательна." if item["usage_mode"] == "manager_review_required" else "")
            lines.append(f"- {clean_text(item['client_safe_text'])} [действует до {item['valid_until'] or 'ручной перепроверки'}].{suffix}")
        lines.append("")
    return "\n".join(lines)
def render_readme(records: list[dict[str, object]]) -> str:
    counts = Counter(str(item["brand"]) for item in records)
    return f"""# KB v6.8: пакет для партнёров

Пакет содержит внешние клиентские факты: всего {len(records)}, Фотон — {counts['foton']}, УНПК — {counts['unpk']}.

Начните с `README_FOR_PARTNERS.docx`. Для режима автоответов после проверки интеграции используйте только `KB_FOTON_AUTO.md` или `KB_UNPK_AUTO.md`. Полные Markdown-файлы допустимы только в режиме черновиков. Для RAG используйте JSONL и фильтры `brand` и `usage_mode`. `manager_review_required` нельзя отправлять без проверки человеком. Перед запуском перепроверьте цены, расписание, скидки, способы оплаты и контакты.
"""
SYSTEM_PROMPT_TEMPLATE = """# Системная инструкция для бота

Ты готовишь точные ответы клиентам на основе переданной базы знаний.

1. Активный бренд задаётся системой до вызова модели: `foton` или `unpk`. Никогда не определяй его по похожести и не смешивай бренды.
2. Используй только факты активного бренда с неистёкшим `valid_until`.
3. Не придумывай факты. Если подходящего факта нет или факты конфликтуют, напиши, что вопрос нужно уточнить у менеджера.
4. `usage_mode=manager_review_required` означает: подготовь черновик, но не отправляй его автоматически.
5. `refresh_before_launch=true` означает: не используй факт, пока владелец базы не подтвердит его актуальность.
6. Отвечай на вопрос прямо, коротко и человеческим языком. Не показывай JSON, идентификаторы, маршруты или внутренние правила.
7. Публичные контакты относятся только к указанному бренду. Если база адаптируется под другую организацию, сначала замените их.
"""
def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
def readable_text(path: Path) -> str:
    if path.suffix.casefold() != ".docx":
        return path.read_text(encoding="utf-8", errors="ignore")
    with zipfile.ZipFile(path) as archive:
        return "\n".join(archive.read(name).decode("utf-8", "ignore") for name in archive.namelist() if name.endswith(".xml"))
if __name__ == "__main__":
    raise SystemExit(main())
