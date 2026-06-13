TZ22 D7 E5: dead-code candidates, no deletion

Command:

```bash
rg -n "prepare_message_archive_history_full_cycle|prepare_message_archives_history_full_cycle|build_student_card_manual_review_pack" src scripts tests
```

Result:
- No references in file contents under `src/`, `scripts/`, or `tests/`.
- Files exist:
  - `scripts/prepare_message_archive_history_full_cycle.py`
  - `scripts/prepare_message_archives_history_full_cycle.py`
  - `scripts/build_student_card_manual_review_pack.py`

Candidates:

| file | references outside own filename | note |
| --- | ---: | --- |
| `scripts/prepare_message_archive_history_full_cycle.py` | 0 | Possible duplicate of plural-name script. |
| `scripts/prepare_message_archives_history_full_cycle.py` | 0 | Possible duplicate of singular-name script. |
| `scripts/build_student_card_manual_review_pack.py` | 0 | No code/test caller found. |

Decision:
- Do not delete in TZ22.
- Deletion or consolidation requires a separate Dmitry decision after owner/use-case check.
