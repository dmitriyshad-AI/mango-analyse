from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional


def _clean(value: object) -> str:
    return str(value or "").strip()


@dataclass(frozen=True)
class AudioStoreRecord:
    record_type: str
    record_id: str
    source_set: str
    source_audio_path: str
    source_audio_basename: str
    canonical_audio_path: str
    sha256: str
    size_bytes: int
    ext: str
    queue_item_id: str = ""
    event_key: str = ""
    provider_call_id: str = ""


class AudioStoreIndex:
    """Read-only index over canonical audio store mapping.csv.

    The index deliberately does not mutate DBs or move files. It only resolves
    old source references to the canonical audio-store path.
    """

    def __init__(self, mapping_csv: Path, project_root: Path | None = None) -> None:
        self.mapping_csv = mapping_csv.resolve(strict=False)
        self.project_root = (project_root or Path.cwd()).resolve(strict=False)
        self.records = self._read_mapping(self.mapping_csv)
        self.by_record: dict[tuple[str, str], AudioStoreRecord] = {}
        self.by_source_path: dict[str, AudioStoreRecord] = {}
        self.by_source_basename: dict[str, list[AudioStoreRecord]] = {}
        self.by_queue_item_id: dict[str, AudioStoreRecord] = {}
        self.by_event_key: dict[str, AudioStoreRecord] = {}
        self.by_provider_call_id: dict[str, AudioStoreRecord] = {}
        self.by_sha256: dict[str, list[AudioStoreRecord]] = {}
        for record in self.records:
            if record.record_type and record.record_id:
                self.by_record[(record.record_type, record.record_id)] = record
            if record.source_audio_path:
                self.by_source_path[self._norm_path(record.source_audio_path)] = record
            if record.source_audio_basename:
                self.by_source_basename.setdefault(record.source_audio_basename, []).append(record)
            if record.queue_item_id:
                self.by_queue_item_id[record.queue_item_id] = record
            if record.event_key:
                self.by_event_key[record.event_key] = record
            if record.provider_call_id:
                self.by_provider_call_id[record.provider_call_id] = record
            if record.sha256:
                self.by_sha256.setdefault(record.sha256, []).append(record)

    def resolve(
        self,
        *,
        record_type: str = "",
        record_id: str = "",
        source_audio_path: str = "",
        source_filename: str = "",
        queue_item_id: str = "",
        event_key: str = "",
        provider_call_id: str = "",
        sha256: str = "",
    ) -> Optional[AudioStoreRecord]:
        if record_type and record_id:
            found = self.by_record.get((record_type, record_id))
            if found:
                return found
        if queue_item_id:
            found = self.by_queue_item_id.get(queue_item_id)
            if found:
                return found
        if event_key:
            found = self.by_event_key.get(event_key)
            if found:
                return found
        if provider_call_id:
            found = self.by_provider_call_id.get(provider_call_id)
            if found:
                return found
        if source_audio_path:
            found = self.by_source_path.get(self._norm_path(source_audio_path))
            if found:
                return found
        if source_filename:
            matches = self.by_source_basename.get(Path(source_filename).name, [])
            if len(matches) == 1:
                return matches[0]
        if sha256:
            matches = self.by_sha256.get(sha256.lower(), [])
            if len(matches) == 1:
                return matches[0]
        return None

    def canonical_path(self, record: AudioStoreRecord) -> Path:
        path = Path(record.canonical_audio_path)
        return path if path.is_absolute() else self.project_root / path

    def unresolved_canonical_paths(self, records: Iterable[AudioStoreRecord] | None = None) -> list[AudioStoreRecord]:
        result = []
        for record in records or self.records:
            path = self.canonical_path(record)
            if not path.exists() or not path.is_file() or path.stat().st_size != record.size_bytes:
                result.append(record)
        return result

    def _norm_path(self, value: str) -> str:
        text = _clean(value)
        if not text:
            return ""
        path = Path(text)
        if path.is_absolute():
            try:
                return str(path.resolve(strict=False).relative_to(self.project_root))
            except ValueError:
                return str(path.resolve(strict=False))
        return str(path)

    @staticmethod
    def _read_mapping(path: Path) -> list[AudioStoreRecord]:
        with path.open(encoding="utf-8") as fh:
            rows = []
            for row in csv.DictReader(fh):
                rows.append(
                    AudioStoreRecord(
                        record_type=_clean(row.get("record_type")),
                        record_id=_clean(row.get("record_id")),
                        source_set=_clean(row.get("source_set")),
                        source_audio_path=_clean(row.get("source_audio_path")),
                        source_audio_basename=_clean(row.get("source_audio_basename")),
                        canonical_audio_path=_clean(row.get("canonical_audio_path")),
                        sha256=_clean(row.get("sha256")).lower(),
                        size_bytes=int(row.get("size_bytes") or 0),
                        ext=_clean(row.get("ext")),
                        queue_item_id=_clean(row.get("queue_item_id")),
                        event_key=_clean(row.get("event_key")),
                        provider_call_id=_clean(row.get("provider_call_id")),
                    )
                )
            return rows
