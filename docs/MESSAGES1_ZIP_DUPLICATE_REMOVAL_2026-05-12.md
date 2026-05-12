# messages(1).zip duplicate cleanup, 2026-05-12

Status: completed after owner approval.

Rule: remove only audio entries whose byte size and SHA-256 hash match an audio file under `2026-03-09--26`. Unique audio and non-audio entries were kept in the zip.

- Zip: `_local_archive_20260424/source_archives/messages(1).zip`
- Removed duplicate audio entries: `1880`
- Removed duplicate uncompressed bytes: `804065184`
- Kept unique audio entries: `231`
- Kept non-audio entries: `1`
- Zip size before: `901236473` bytes
- Zip size after: `96833095` bytes
- Disk reduction for zip file: `804403378` bytes
- Local JSON manifest: `audits/_results/MESSAGES1_ZIP_DUPLICATE_REMOVAL_2026-05-12.json`

The JSON manifest is stored outside Git because filenames include client identifiers.
