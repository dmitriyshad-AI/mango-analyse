#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from mango_mvp.services.analyze import ANALYZE_PROMPT_VERSION_FULL, SYSTEM_PROMPT_FULL
from scripts.compute_tz16_rerun_tail import (
    ACTIVE_AMO_EXCLUDED_STATUSES,
    MIN_DURATION_SEC,
    RERUN_SINCE,
    load_zone,
    parse_date,
    read_int_set,
)


EXPECTED_TAIL_CALLS = 3439
TARGET_ARM = "mini_v7:gpt-5.4-mini:full"
TASK_ID = "2026-06-13_analyze_tail_20260612_d4_codex"
DEFAULT_TIMELINE_DB = Path(
    "/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_profiles/"
    "tz16_profiles_v7_20260612/customer_timeline.sqlite"
)
DEFAULT_MASTER_CALLS_DB = Path(
    "/Users/dmitrijfabarisov/Projects/Mango analyse/"
    "stable_runtime/canonical_master_20260523_audio_working_store_v1/canonical_calls_master.db"
)
DEFAULT_PREVIOUS_RERUN_PACKAGE = Path(
    "/Users/dmitrijfabarisov/Yandex.Disk.localized/OpenClaw/Actual Mango Tests/analyze_rerun_20260611"
)
DEFAULT_BUNDLE_DIR = Path(
    "/Users/dmitrijfabarisov/Yandex.Disk.localized/OpenClaw/Actual Mango Tests/analyze_tail_20260612"
)
DEFAULT_INBOX_M1_DIR = Path(
    "/Users/dmitrijfabarisov/Yandex.Disk.localized/OpenClaw/Actual Mango Tests/tasks/_inbox_m1"
)


@dataclass(frozen=True)
class TailCall:
    call_id: int
    started_at: str
    duration_sec: int
    transcript_chars: int


@dataclass(frozen=True)
class TailBundleConfig:
    timeline_db: Path
    master_calls_db: Path
    previous_rerun_package: Path
    bundle_dir: Path
    inbox_m1_dir: Path
    repo_root: Path
    expected_calls: int = EXPECTED_TAIL_CALLS
    parts: int = 4


def build_tail_bundle(config: TailBundleConfig) -> Mapping[str, Any]:
    cfg = normalize_config(config)
    blacklist_ids = read_int_set(cfg.previous_rerun_package / "blacklist_77.txt")
    selected = select_tail_calls(cfg, blacklist_ids)
    if len(selected) != cfg.expected_calls:
        raise RuntimeError(f"tail call count mismatch: expected {cfg.expected_calls}, got {len(selected)}")
    selected_ids = [item.call_id for item in selected]
    overlap = sorted(set(selected_ids) & blacklist_ids)
    if overlap:
        raise RuntimeError(f"selected tail intersects blacklist: {overlap[:10]}")

    data_dir = cfg.bundle_dir / "data"
    scripts_dir = cfg.bundle_dir / "scripts_pkg"
    prompt_dir = cfg.bundle_dir / "prompt"
    code_dir = cfg.bundle_dir / "code"
    for path in (data_dir, scripts_dir, prompt_dir, code_dir):
        path.mkdir(parents=True, exist_ok=True)

    prompt_text = SYSTEM_PROMPT_FULL.strip() + "\n"
    prompt_sha256 = sha256_text(prompt_text)
    write_if_same_or_absent(prompt_dir / "analyze_prompt_full_v7.txt", prompt_text)

    ids_text = "\n".join(str(item) for item in selected_ids) + "\n"
    ids_sha256 = sha256_text(ids_text)
    write_if_same_or_absent(data_dir / "ids_all.txt", ids_text)
    parts = write_part_files(data_dir, selected_ids, cfg.parts)

    slice_db = data_dir / "slice_zone.db"
    create_or_validate_slice_db(cfg.master_calls_db, slice_db, selected_ids)
    create_deterministic_zip(slice_db, data_dir / "slice_zone.db.zip")
    copy_if_same_or_absent(cfg.previous_rerun_package / "blacklist_77.txt", cfg.bundle_dir / "blacklist_77.txt")
    copy_support_scripts(cfg.previous_rerun_package, scripts_dir)

    git_commit = current_git_commit(cfg.repo_root)
    short_commit = git_commit[:8]
    code_archive = code_dir / f"mango_clean_{short_commit}.tar.gz"
    create_git_archive(cfg.repo_root, code_archive)
    write_if_same_or_absent(
        code_dir / f"VERSION_{short_commit}.txt",
        "\n".join(
            [
                f"git_commit={git_commit}",
                "contains=TZ16 merge + TZ19 bundle builder",
                "run_from=git archive HEAD",
                "",
            ]
        ),
    )

    manifest = build_manifest(
        cfg=cfg,
        selected=selected,
        ids_sha256=ids_sha256,
        prompt_sha256=prompt_sha256,
        parts=parts,
        git_commit=git_commit,
        code_archive=code_archive,
        slice_db=slice_db,
    )
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    write_if_same_or_absent(data_dir / "manifest.json", manifest_text)

    config_payload = {
        "schema_version": "tz19_m1_analyze_tail_config_v1",
        "target_arm": TARGET_ARM,
        "provider": "codex_cli",
        "model": "gpt-5.4-mini",
        "prompt_profile": "full",
        "parallel_parts": cfg.parts,
        "timeout_sec": 180,
        "expected_calls": cfg.expected_calls,
        "expected_prompt_version": ANALYZE_PROMPT_VERSION_FULL,
        "prompt_sha256": prompt_sha256,
        "forbidden": ["import_results", "write_crm", "write_amo", "write_tallanto", "run_asr"],
    }
    write_if_same_or_absent(
        cfg.bundle_dir / "config.json",
        json.dumps(config_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    write_bundle_readme(cfg.bundle_dir, short_commit, cfg.parts)
    write_m1_prompt(cfg.bundle_dir)

    task_path = cfg.inbox_m1_dir / f"{TASK_ID}.task.yaml"
    task_yaml = build_task_yaml(cfg, manifest, short_commit)
    write_if_same_or_absent(task_path, task_yaml)
    ready_path = Path(str(task_path) + ".ready")
    write_if_same_or_absent(ready_path, f"sha256:{sha256_text(task_yaml)}\n")

    result = {
        "bundle_dir": str(cfg.bundle_dir),
        "task_yaml": str(task_path),
        "ready_marker": str(ready_path),
        "manifest": manifest,
        "prompt_sha256": prompt_sha256,
        "ids_sha256": ids_sha256,
        "blacklist_overlap": 0,
        "git_commit": git_commit,
        "safety": {
            "write_crm": False,
            "write_tallanto": False,
            "write_amo": False,
            "run_asr": False,
            "run_resolve_analyze": False,
            "llm_calls": 0,
        },
    }
    write_if_same_or_absent(
        cfg.bundle_dir / "BUILD_SUMMARY.json",
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return result


def normalize_config(config: TailBundleConfig) -> TailBundleConfig:
    return TailBundleConfig(
        timeline_db=config.timeline_db.expanduser().resolve(strict=False),
        master_calls_db=config.master_calls_db.expanduser().resolve(strict=False),
        previous_rerun_package=config.previous_rerun_package.expanduser().resolve(strict=False),
        bundle_dir=config.bundle_dir.expanduser().resolve(strict=False),
        inbox_m1_dir=config.inbox_m1_dir.expanduser().resolve(strict=False),
        repo_root=config.repo_root.expanduser().resolve(strict=False),
        expected_calls=config.expected_calls,
        parts=config.parts,
    )


def select_tail_calls(config: TailBundleConfig, blacklist_ids: set[int]) -> list[TailCall]:
    zone = load_zone(config.timeline_db)
    call_ids = sorted(int(item) for item in zone["call_ids"])
    con = sqlite3.connect(f"file:{config.master_calls_db}?mode=ro&immutable=1", uri=True)
    con.row_factory = sqlite3.Row
    selected: list[TailCall] = []
    try:
        for batch in chunked(call_ids, 800):
            placeholders = ",".join("?" for _ in batch)
            rows = con.execute(
                f"""
                SELECT canonical_call_id,
                       COALESCE(started_at, '') AS started_at,
                       CAST(COALESCE(duration_sec, 0) AS INTEGER) AS duration_sec,
                       COALESCE(analysis_status, '') AS analysis_status,
                       COALESCE(transcript_chars, LENGTH(COALESCE(transcript_text, '')), 0) AS transcript_chars,
                       CASE
                         WHEN analysis_json IS NOT NULL
                          AND json_valid(analysis_json)
                         THEN COALESCE(json_extract(analysis_json, '$.analysis_meta.analysis_prompt_version'), '')
                         ELSE ''
                       END AS prompt_version
                FROM canonical_calls
                WHERE canonical_call_id IN ({placeholders})
                ORDER BY canonical_call_id
                """,
                tuple(batch),
            ).fetchall()
            for row in rows:
                cid = int(row["canonical_call_id"])
                if cid in blacklist_ids:
                    continue
                if str(row["analysis_status"] or "") != "done":
                    continue
                if int(row["transcript_chars"] or 0) <= 0:
                    continue
                if int(row["duration_sec"] or 0) < MIN_DURATION_SEC:
                    continue
                if parse_date(str(row["started_at"] or "")) >= RERUN_SINCE:
                    continue
                if str(row["prompt_version"] or "") == ANALYZE_PROMPT_VERSION_FULL:
                    continue
                selected.append(
                    TailCall(
                        call_id=cid,
                        started_at=str(row["started_at"] or ""),
                        duration_sec=int(row["duration_sec"] or 0),
                        transcript_chars=int(row["transcript_chars"] or 0),
                    )
                )
    finally:
        con.close()
    return sorted(selected, key=lambda item: item.call_id)


def create_or_validate_slice_db(source_db: Path, target_db: Path, ids: Sequence[int]) -> None:
    if target_db.exists():
        validate_slice_db(target_db, ids)
        return

    src = sqlite3.connect(f"file:{source_db}?mode=ro&immutable=1", uri=True)
    src.row_factory = sqlite3.Row
    try:
        ddl_row = src.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='canonical_calls'"
        ).fetchone()
        if not ddl_row or not ddl_row["sql"]:
            raise RuntimeError("canonical_calls table not found in source db")
        target_db.parent.mkdir(parents=True, exist_ok=True)
        out = sqlite3.connect(target_db)
        try:
            out.execute(ddl_row["sql"])
            cols = [str(row[1]) for row in src.execute("PRAGMA table_info(canonical_calls)").fetchall()]
            insert_sql = f"INSERT INTO canonical_calls ({','.join(cols)}) VALUES ({','.join('?' for _ in cols)})"
            for batch in chunked(list(ids), 500):
                placeholders = ",".join("?" for _ in batch)
                rows = src.execute(
                    f"SELECT * FROM canonical_calls WHERE canonical_call_id IN ({placeholders})",
                    tuple(batch),
                ).fetchall()
                by_id = {int(row["canonical_call_id"]): row for row in rows}
                missing = [cid for cid in batch if cid not in by_id]
                if missing:
                    raise RuntimeError(f"missing canonical rows for ids: {missing[:10]}")
                out.executemany(insert_sql, [tuple(by_id[cid][col] for col in cols) for cid in batch])
            for row in src.execute(
                "SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name='canonical_calls' AND sql IS NOT NULL"
            ).fetchall():
                try:
                    out.execute(row["sql"])
                except sqlite3.OperationalError:
                    pass
            out.commit()
        finally:
            out.close()
    finally:
        src.close()
    validate_slice_db(target_db, ids)


def validate_slice_db(path: Path, ids: Sequence[int]) -> None:
    con = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    try:
        rows = [int(row[0]) for row in con.execute("SELECT canonical_call_id FROM canonical_calls ORDER BY canonical_call_id")]
    finally:
        con.close()
    expected = list(ids)
    if rows != expected:
        raise RuntimeError(f"existing slice db does not match selected ids: {len(rows)} rows vs {len(expected)} expected")


def create_deterministic_zip(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    info = zipfile.ZipInfo(source.name)
    info.date_time = (2026, 6, 12, 0, 0, 0)
    info.compress_type = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(target, "w") as archive:
        archive.writestr(info, source.read_bytes())


def build_manifest(
    *,
    cfg: TailBundleConfig,
    selected: Sequence[TailCall],
    ids_sha256: str,
    prompt_sha256: str,
    parts: Sequence[Mapping[str, Any]],
    git_commit: str,
    code_archive: Path,
    slice_db: Path,
) -> Mapping[str, Any]:
    return {
        "schema_version": "tz19_analyze_tail_manifest_v1",
        "bundle_id": "analyze_tail_20260612",
        "source_db": str(cfg.master_calls_db),
        "timeline_db": str(cfg.timeline_db),
        "previous_rerun_package": str(cfg.previous_rerun_package),
        "selector": {
            "zone": "active AMO customers OR strong Tallanto student customers",
            "old_prompt_version_not": ANALYZE_PROMPT_VERSION_FULL,
            "started_at": f"< {RERUN_SINCE.isoformat()}",
            "min_duration_sec": MIN_DURATION_SEC,
            "analysis_status": "done",
            "transcript_chars": "> 0",
            "exclude_blacklist_77": True,
            "active_amo_excluded_statuses": list(ACTIVE_AMO_EXCLUDED_STATUSES),
        },
        "rows": len(selected),
        "transcript_chars_sum": sum(item.transcript_chars for item in selected),
        "duration_sec_sum": sum(item.duration_sec for item in selected),
        "ids_sha256": ids_sha256,
        "prompt_profile": "full",
        "prompt_version": ANALYZE_PROMPT_VERSION_FULL,
        "prompt_sha256": prompt_sha256,
        "target_arm": TARGET_ARM,
        "parts": list(parts),
        "code_commit": git_commit,
        "code_archive": str(code_archive.relative_to(cfg.bundle_dir)),
        "code_archive_sha256": sha256_file(code_archive),
        "slice_db": str(slice_db.relative_to(cfg.bundle_dir)),
        "slice_db_bytes": slice_db.stat().st_size,
        "slice_db_sha256": sha256_file(slice_db),
        "calls": [
            {
                "canonical_call_id": item.call_id,
                "started_at": item.started_at,
                "duration_sec": item.duration_sec,
                "transcript_chars": item.transcript_chars,
            }
            for item in selected
        ],
        "safety": {
            "blacklist_overlap": 0,
            "write_crm": False,
            "write_tallanto": False,
            "write_amo": False,
            "run_asr": False,
            "run_resolve_analyze": False,
            "llm_calls": 0,
        },
    }


def write_part_files(data_dir: Path, ids: Sequence[int], parts_count: int) -> list[Mapping[str, Any]]:
    parts: list[Mapping[str, Any]] = []
    for index in range(parts_count):
        part_ids = list(ids[index::parts_count])
        filename = f"ids_part{index + 1}.txt"
        text = "\n".join(str(item) for item in part_ids) + "\n"
        write_if_same_or_absent(data_dir / filename, text)
        parts.append({"file": filename, "count": len(part_ids), "ids_sha256": sha256_text(text)})
    return parts


def copy_support_scripts(previous_package: Path, scripts_dir: Path) -> None:
    for name in ("export_analyze_results.py", "m1_run_cli_shim.sh"):
        copy_if_same_or_absent(previous_package / "scripts_pkg" / name, scripts_dir / name)


def write_bundle_readme(bundle_dir: Path, short_commit: str, parts: int) -> None:
    lines = [
        "# Analyze tail 20260612: old >=60 sec customer-zone calls",
        "",
        "Пакет для М1. Он запускает только перепрогон выжимок analyze.",
        "Вливание результатов в каноническую базу здесь запрещено.",
        "",
        "## Состав",
        "",
        "- `data/slice_zone.db` и `data/slice_zone.db.zip` — срез canonical_calls только на 3 439 звонков.",
        "- `data/ids_all.txt`, `data/ids_part1..4.txt` — список id и 4 части.",
        "- `data/manifest.json` — контрольные суммы, длительности и размеры расшифровок.",
        "- `prompt/analyze_prompt_full_v7.txt` — полная подсказка v7 с анти-автоответчик правкой.",
        "- `scripts_pkg/export_analyze_results.py` — выгрузка результатов после прогона.",
        "- `scripts_pkg/m1_run_cli_shim.sh` — запасной shim для CLI.",
        f"- `code/mango_clean_{short_commit}.tar.gz` — архив кода для М1.",
        "",
        "## Запуск",
        "",
        "```bash",
        f"mkdir -p ~/mango_m1_work/mango_clean_{short_commit}",
        f"tar -xzf code/mango_clean_{short_commit}.tar.gz -C ~/mango_m1_work/mango_clean_{short_commit}",
        f"cd ~/mango_m1_work/mango_clean_{short_commit}",
        "export PYTHONDONTWRITEBYTECODE=1",
        "export PYTHONPATH=src",
        f"DATA=\"{bundle_dir}/data\"",
        "RUNS=\"$HOME/mango_m1_work/analyze_tail_20260612_runs\"",
        "mkdir -p \"$RUNS\"",
        "",
        f"for N in $(seq 1 {parts}); do",
        "  COUNT=$(grep -c . \"$DATA/ids_part$N.txt\")",
        "  nohup python3 scripts/run_analyze_ab_test.py \\",
        "    --source-db \"$DATA/slice_zone.db\" \\",
        "    --ids-file \"$DATA/ids_part$N.txt\" \\",
        "    --sample-size \"$COUNT\" \\",
        "    --arms mini_v7:gpt-5.4-mini:full \\",
        "    --provider codex_cli \\",
        "    --cli stable_runtime/m1_run_cli_shim.sh \\",
        "    --out-dir \"$RUNS/part$N\" > \"$RUNS/part$N.log\" 2>&1 &",
        "done",
        "```",
        "",
        "Перед стартом сверить `python3 scripts/run_analyze_ab_test.py --help`.",
        "Если `stable_runtime/m1_run_cli_shim.sh` отсутствует, скопировать его из `scripts_pkg/` в `stable_runtime/` и сделать исполняемым.",
        "",
        "## После каждой части",
        "",
        "```bash",
        f"for N in $(seq 1 {parts}); do",
        "  python3 scripts_pkg/export_analyze_results.py \\",
        "    --db \"$RUNS/part$N/mini_v7/test.db\" \\",
        "    --ids-file \"$DATA/ids_part$N.txt\" \\",
        "    --out \"results_part$N.jsonl.gz\"",
        "done",
        "```",
        "",
        "Вернуть в папку пакета `results_part1..4.jsonl.gz`, `ab_summary.json` частей, хвосты логов и `READY_YYYY-MM-DD.md`.",
        "",
        "Запрещено: вливание в боевую базу, изменение кода, изменение настроек М1, ASR, CRM/AMO/Tallanto write.",
        "",
    ]
    write_if_same_or_absent(bundle_dir / "README_RUN.md", "\n".join(lines))


def write_m1_prompt(bundle_dir: Path) -> None:
    prompt = f"""# Промт для Кодекса на М1

Задача: пакетный перепрогон выжимок звонков analyze на gpt-5.4-mini с полной подсказкой v7.

Пакет: Яндекс-диск, папка `OpenClaw/Actual Mango Tests/analyze_tail_20260612/`.
Прочитай `README_RUN.md` и выполни строго по нему.

Обязательные проверки:

1. `data/manifest.json` содержит `rows=3439`.
2. `prompt_sha256` в manifest совпадает с sha256 файла `prompt/analyze_prompt_full_v7.txt`.
3. `blacklist_overlap=0`.
4. Запуск только 4 частей `mini_v7:gpt-5.4-mini:full`, provider `codex_cli`.
5. После прогона выгрузить `results_part1..4.jsonl.gz` и вернуть summary/log tails/READY.

Запрещено: импорт результатов в боевую базу, изменение кода, изменение настроек М1, ASR, CRM/AMO/Tallanto write.
"""
    write_if_same_or_absent(bundle_dir / "PROMPT_for_M1_codex.md", prompt)


def build_task_yaml(config: TailBundleConfig, manifest: Mapping[str, Any], short_commit: str) -> str:
    return "\n".join(
        [
            f"id: {TASK_ID}",
            "kind: analyze_tail_rerun",
            f"requires_bundle: mango_clean_{short_commit}",
            'package: "OpenClaw/Actual Mango Tests/analyze_tail_20260612"',
            f'bundle_path: "{config.bundle_dir}"',
            "brain: codex",
            "parallel: 4",
            "max_hours: 8",
            "target_arm: mini_v7:gpt-5.4-mini:full",
            "provider: codex_cli",
            "prompt_profile: full",
            f"expected_calls: {manifest['rows']}",
            f"transcript_chars_sum: {manifest['transcript_chars_sum']}",
            f"ids_sha256: {manifest['ids_sha256']}",
            f"prompt_sha256: {manifest['prompt_sha256']}",
            "outputs:",
            "  - results_part1.jsonl.gz",
            "  - results_part2.jsonl.gz",
            "  - results_part3.jsonl.gz",
            "  - results_part4.jsonl.gz",
            "  - READY_YYYY-MM-DD.md",
            "forbidden:",
            "  - import_results",
            "  - write_crm",
            "  - write_amo",
            "  - write_tallanto",
            "  - run_asr",
            "env:",
            '  TELEGRAM_DIRECT_PATH_PILOT_CONFIG: ""',
            "",
        ]
    )


def current_git_commit(repo_root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()


def create_git_archive(repo_root: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    subprocess.run(
        ["git", "archive", "--format=tar.gz", "--output", str(target), "HEAD"],
        cwd=repo_root,
        check=True,
    )


def write_if_same_or_absent(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != text:
            raise RuntimeError(f"refusing to overwrite different generated file: {path}")
        return
    path.write_text(text, encoding="utf-8")


def copy_if_same_or_absent(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256_file(source) != sha256_file(target):
            raise RuntimeError(f"refusing to overwrite different generated file: {target}")
        return
    shutil.copy2(source, target)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def chunked(values: Sequence[int], size: int) -> Iterable[Sequence[int]]:
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build TZ-19 M1 analyze tail bundle.")
    parser.add_argument("--timeline-db", type=Path, default=DEFAULT_TIMELINE_DB)
    parser.add_argument("--master-calls-db", type=Path, default=DEFAULT_MASTER_CALLS_DB)
    parser.add_argument("--previous-rerun-package", type=Path, default=DEFAULT_PREVIOUS_RERUN_PACKAGE)
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    parser.add_argument("--inbox-m1-dir", type=Path, default=DEFAULT_INBOX_M1_DIR)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--expected-calls", type=int, default=EXPECTED_TAIL_CALLS)
    parser.add_argument("--parts", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_tail_bundle(
        TailBundleConfig(
            timeline_db=args.timeline_db,
            master_calls_db=args.master_calls_db,
            previous_rerun_package=args.previous_rerun_package,
            bundle_dir=args.bundle_dir,
            inbox_m1_dir=args.inbox_m1_dir,
            repo_root=args.repo_root,
            expected_calls=args.expected_calls,
            parts=args.parts,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
