#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "tallanto_money_snapshot_from_local_exports_v1"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build deterministic Tallanto money snapshot from local JSON exports.")
    parser.add_argument("--allowed-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--finance-input", action="append", default=[])
    parser.add_argument("--abonement-input", action="append", default=[])
    parser.add_argument("--class-input", action="append", default=[])
    args = parser.parse_args(argv)

    report = build_snapshot(
        allowed_root=Path(args.allowed_root),
        output=Path(args.output),
        finance_inputs=[Path(item) for item in args.finance_input],
        abonement_inputs=[Path(item) for item in args.abonement_input],
        class_inputs=[Path(item) for item in args.class_input],
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


def build_snapshot(
    *,
    allowed_root: Path,
    output: Path,
    finance_inputs: Sequence[Path],
    abonement_inputs: Sequence[Path],
    class_inputs: Sequence[Path],
) -> Mapping[str, Any]:
    root = allowed_root.expanduser().resolve(strict=False)
    out = _guard_output(output, root)
    source_groups = {
        "most_finances": tuple(_guard_input(path, root) for path in finance_inputs),
        "most_abonements": tuple(_guard_input(path, root) for path in abonement_inputs),
        "most_class": tuple(_guard_input(path, root) for path in class_inputs),
    }
    finances, finance_stats = _dedupe_records(
        _records_from_files(source_groups["most_finances"], nested_keys=("finances", "records")),
        id_fields=("id", "finance_id", "payment_id"),
    )
    abonements, abonement_stats = _dedupe_records(
        _records_from_files(source_groups["most_abonements"], nested_keys=("abonements", "records")),
        id_fields=("id", "abonement_id", "most_abonements_id"),
    )
    classes, class_stats = _dedupe_records(
        _records_from_files(source_groups["most_class"], nested_keys=("classes", "courses", "records")),
        id_fields=("id", "class_id", "most_class_id"),
    )
    sources = {
        key: [{"path": str(path), "sha256": _sha256_file(path)} for path in paths]
        for key, paths in source_groups.items()
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "deterministic_build": True,
        "source_sha256": _stable_digest(sources),
        "sources": sources,
        "most_finances": finances,
        "most_abonements": abonements,
        "most_class": classes,
        "stats": {
            "most_finances": finance_stats,
            "most_abonements": abonement_stats,
            "most_class": class_stats,
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    out.write_text(text, encoding="utf-8")
    repeat_sha = _sha256_text(text)
    return {
        "schema_version": SCHEMA_VERSION,
        "output": str(out),
        "output_sha256": repeat_sha,
        "source_sha256": payload["source_sha256"],
        "counts": {
            "most_finances": len(finances),
            "most_abonements": len(abonements),
            "most_class": len(classes),
        },
        "stats": payload["stats"],
    }


def _records_from_files(paths: Sequence[Path], *, nested_keys: Sequence[str]) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    for path in paths:
        for item in _load_json_records(path):
            records.extend(_expand_record(item, nested_keys=nested_keys))
    return records


def _load_json_records(path: Path) -> list[Any]:
    if path.suffix == ".jsonl":
        items: list[Any] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    items.append(json.loads(line))
        return items
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return list(payload)
    if isinstance(payload, dict):
        if all(isinstance(value, Mapping) for value in payload.values()) and not any(
            key in payload for key in ("records", "items", "result", "content")
        ):
            return list(payload.values())
        return [payload]
    return []


def _expand_record(item: Any, *, nested_keys: Sequence[str]) -> list[Mapping[str, Any]]:
    if isinstance(item, list):
        return [record for value in item for record in _expand_record(value, nested_keys=nested_keys)]
    if not isinstance(item, Mapping):
        return []
    result: list[Mapping[str, Any]] = []
    if _looks_like_tallanto_row(item):
        result.append(dict(item))
    for key in nested_keys:
        value = item.get(key)
        if isinstance(value, list):
            result.extend(record for value_item in value for record in _expand_record(value_item, nested_keys=nested_keys))
        elif isinstance(value, Mapping):
            result.extend(record for value_item in value.values() for record in _expand_record(value_item, nested_keys=nested_keys))
    return result


def _looks_like_tallanto_row(item: Mapping[str, Any]) -> bool:
    return any(key in item for key in ("id", "finance_id", "payment_id", "abonement_id", "most_abonements_id", "class_id"))


def _dedupe_records(records: Sequence[Mapping[str, Any]], *, id_fields: Sequence[str]) -> tuple[list[Mapping[str, Any]], Mapping[str, int]]:
    seen: dict[str, Mapping[str, Any]] = {}
    first_hash: dict[str, str] = {}
    stats: Counter[str] = Counter({"input_rows": len(records)})
    for index, record in enumerate(records, start=1):
        record_id = _record_id(record, id_fields=id_fields)
        if not record_id:
            stats["skipped_without_id"] += 1
            continue
        normalized = dict(record)
        digest = _stable_digest(normalized)
        if record_id in seen:
            stats["duplicate_id_rows"] += 1
            if first_hash[record_id] != digest:
                stats["duplicate_id_conflicts"] += 1
            continue
        normalized["_snapshot_source_index"] = index
        seen[record_id] = normalized
        first_hash[record_id] = digest
    return [seen[key] for key in sorted(seen)], {
        "input_rows": stats["input_rows"],
        "output_rows": len(seen),
        "skipped_without_id": stats["skipped_without_id"],
        "duplicate_id_rows": stats["duplicate_id_rows"],
        "duplicate_id_conflicts": stats["duplicate_id_conflicts"],
    }


def _record_id(record: Mapping[str, Any], *, id_fields: Sequence[str]) -> str:
    for field in id_fields:
        value = record.get(field)
        if value not in (None, ""):
            return str(value).strip()
    return ""


def _guard_input(path: Path, allowed_root: Path) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    try:
        resolved.relative_to(allowed_root)
    except ValueError as exc:
        raise ValueError(f"input must stay under allowed root: {allowed_root}: {resolved}") from exc
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _guard_output(path: Path, allowed_root: Path) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    try:
        resolved.relative_to(allowed_root)
    except ValueError as exc:
        raise ValueError(f"output must stay under allowed root: {allowed_root}: {resolved}") from exc
    parts = tuple(part.casefold() for part in resolved.parts)
    if not any(part == ".codex_local" and parts[index + 1] == "staging" for index, part in enumerate(parts[:-1])):
        raise ValueError("output must be under .codex_local/staging")
    return resolved


def _stable_digest(payload: Any) -> str:
    return _sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
