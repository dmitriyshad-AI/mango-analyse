# Risk Review

## Low Risk

- Only git tracking was changed for current v6.3 generated outputs; files were not deleted from disk.
- `.gitignore` rules are exact directory rules and do not match `*_sources/`.
- Frozen old releases v2, v3, v3_2, v3_3 and their packs were restored in the working tree before commit.
- `src/mango_mvp/` was left out of the change scope.

## Residual Risks

- Some existing docs and scripts still reference generated v6.3 output paths. This remains valid locally because the files stay on disk and are regenerable, but fresh checkouts must run the builder before using those paths.
- The builder writes timestamp and output-root path metadata, so byte-for-byte comparison across two different output roots is not expected. The normalized comparison confirms content equivalence after replacing volatile timestamps and output roots.

## Controls

- `git ls-files` for current v6.3 generated dirs returns `0`.
- `git ls-files` for v6.3 sources returns tracked files.
- Targeted tests passed.
- Full pytest collection passed.
