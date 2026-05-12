# Exact audio duplicate delete candidates, 2026-05-12

Status: deleted after owner approval on 2026-05-12.

Rule used: a file is listed only when it has the same filename, same byte size, and same SHA-256 hash as a canonical audio file under `2026-03-09--26`.

- Exact duplicate files: `6370`
- Deleted bytes: `2162964384` (`2.0 GiB`, about `2.0 GB` by `du -h`)
- Skipped/non-exact files: `0`
- Full CSV manifest: `audits/_results/EXACT_AUDIO_DUPLICATES_DELETE_CANDIDATES_2026-05-12.csv`

The full manifest is intentionally stored in `audits/_results`, which is ignored by Git, because audio filenames include client identifiers.

## Candidate groups

| Delete candidate parent | Files | Size |
|---|---:|---:|
| `stable_runtime/external_m1_jan2025_test300_20260503/batch_asr_only` | 300 | 111.0 MB |
| `stable_runtime/external_m1_jan_mar_2025_asr_only_20260504/batch_asr_only` | 6070 | 1.9 GB |

## Deletion result

Deleted files were rechecked immediately before removal: allowed parent path, existing canonical file, matching size, and matching SHA-256 hash.

After deletion, both old ASR batch folders have `0` regular audio files in `batch_asr_only`.
