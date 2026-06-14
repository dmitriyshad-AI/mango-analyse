# Risk review

Main risks:
- False merge of two different children.
- PII exposure through raw LLM prompt/cache.
- Accidental resolver run without shared-phone stoplist.
- Regression of existing deterministic OFF behavior.
- Full 7,512-family run started accidentally instead of microprobe.

Mitigations:
- `PROFILE_LLM_CHILD_RESOLVER` default OFF.
- ON path requires `~/.mango_secrets/shared_phones_stoplist.json`; missing/invalid file raises before run.
- Seven deterministic veto classes implemented, including invented names, misattribution, incomplete mapping, incompatible grades, child-count sanity, brand preservation, and fail-soft.
- Additional deterministic guard rejects merging incompatible different names even if the model puts both into `name_variants`.
- Bad/invalid model output leaves the family unchanged.
- Brand is preserved per `ProfileFieldCandidate` row; microprobe reports `llm_brand_changed_fields=0`.
- Microprobe script limits selected families and does not call the full 7,512 scope.
- Raw artifacts remain under ignored paths.

Residual risk:
- The verifier is conservative and can reject potentially valid model resolutions; microprobe had 7 fail-soft/vetoed families.
- The name-compatibility guard is intentionally conservative and may need tuning after Claude raw regrade.
- Full-run runtime and cost are not measured because the full run was explicitly out of scope.
