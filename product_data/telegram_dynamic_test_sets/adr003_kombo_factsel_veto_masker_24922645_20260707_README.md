# ADR003 combo factsel/veto/masker set 24922645 2026-07-07

- Output: `adr003_kombo_factsel_veto_masker_24922645_20260707.jsonl`
- Lines: 50 = 2 spec + 48 personas
- Source factsel package: `/Users/dmitrijfabarisov/Yandex.Disk.localized/OpenClaw/Actual Mango Tests/adr003_fact_select_frame_pair_9b2b9315_20260707/adr003_fact_select_frame_focus_only_9b2b9315_20260707.jsonl`
- `simulator_spec` and `judge_spec` copied from package 9b2b9315 and calibrated for 29a/29b; `wrong_product_fact` is a semantic hard-gate rule in the judge prompt, not a separate stable machine counter.
- B/ON pair contract was superseded by the mega-TZ: this set is now an input/focus source for the final consolidated M1 exam. `fact_select_read` is profile-default after the profile reveal; final ON composition is defined by the M1 package README.

## Composition
### factsel_33
- `factsel_foton_online_physics_vs_math_01` — source `adr003_fact_select_frame_focus_only_9b2b9315_20260707.jsonl`
- `factsel_foton_online_math_vs_physics_02` — source `adr003_fact_select_frame_focus_only_9b2b9315_20260707.jsonl`
- `factsel_unpk_math_not_russian_03` — source `adr003_fact_select_frame_focus_only_9b2b9315_20260707.jsonl`
- `factsel_unpk_physics_not_programming_04` — source `adr003_fact_select_frame_focus_only_9b2b9315_20260707.jsonl`
- `factsel_foton_grade6_not8_05` — source `adr003_fact_select_frame_focus_only_9b2b9315_20260707.jsonl`
- `factsel_foton_grade10_not11_06` — source `adr003_fact_select_frame_focus_only_9b2b9315_20260707.jsonl`
- `factsel_unpk_grade5_boundary_07` — source `adr003_fact_select_frame_focus_only_9b2b9315_20260707.jsonl`
- `factsel_foton_grade7_after_history_08` — source `adr003_fact_select_frame_focus_only_9b2b9315_20260707.jsonl`
- `factsel_foton_online_not_offline_09` — source `adr003_fact_select_frame_focus_only_9b2b9315_20260707.jsonl`
- `factsel_foton_offline_not_online_10` — source `adr003_fact_select_frame_focus_only_9b2b9315_20260707.jsonl`
- `factsel_unpk_offline_venue_11` — source `adr003_fact_select_frame_focus_only_9b2b9315_20260707.jsonl`
- `factsel_foton_online_platform_12` — source `adr003_fact_select_frame_focus_only_9b2b9315_20260707.jsonl`
- `factsel_foton_krasnoselskaya_not_unpk_13` — source `adr003_fact_select_frame_focus_only_9b2b9315_20260707.jsonl`
- `factsel_unpk_venue_not_foton_14` — source `adr003_fact_select_frame_focus_only_9b2b9315_20260707.jsonl`
- `factsel_foton_lvesh_camp_not_regular_15` — source `adr003_fact_select_frame_focus_only_9b2b9315_20260707.jsonl`
- `factsel_foton_regular_not_camp_16` — source `adr003_fact_select_frame_focus_only_9b2b9315_20260707.jsonl`
- `factsel_foton_camp_age_not_regular_grade_17` — source `adr003_fact_select_frame_focus_only_9b2b9315_20260707.jsonl`
- `factsel_foton_second_shift_not_regular_18` — source `adr003_fact_select_frame_focus_only_9b2b9315_20260707.jsonl`
- `factsel_foton_group_size_regular_19` — source `adr003_fact_select_frame_focus_only_9b2b9315_20260707.jsonl`
- `factsel_foton_group_size_camp_20` — source `adr003_fact_select_frame_focus_only_9b2b9315_20260707.jsonl`
- `factsel_unpk_group_size_not_foton_21` — source `adr003_fact_select_frame_focus_only_9b2b9315_20260707.jsonl`
- `factsel_foton_tax_docs_not_refund_22` — source `adr003_fact_select_frame_focus_only_9b2b9315_20260707.jsonl`
- `factsel_unpk_tax_docs_not_refund_23` — source `adr003_fact_select_frame_focus_only_9b2b9315_20260707.jsonl`
- `factsel_foton_paid_no_access_manager_24` — source `adr003_fact_select_frame_focus_only_9b2b9315_20260707.jsonl`
- `factsel_unpk_refund_real_manager_25` — source `adr003_fact_select_frame_focus_only_9b2b9315_20260707.jsonl`
- `factsel_foton_presale_refund_safe_26` — source `adr003_fact_select_frame_focus_only_9b2b9315_20260707.jsonl`
- `factsel_foton_place_vs_places_27` — source `adr003_fact_select_frame_focus_only_9b2b9315_20260707.jsonl`
- `factsel_foton_places_live_manager_28` — source `adr003_fact_select_frame_focus_only_9b2b9315_20260707.jsonl`
- `factsel_fallback_wrong_venue_lvsh_29a` — calibrated explicit regular-year request; source `adr003_fact_select_frame_focus_only_9b2b9315_20260707.jsonl`
- `factsel_fallback_wrong_venue_lvsh_29b` — calibrated ambiguous season request; success accepts clarification or explicit year/summer split; source `adr003_fact_select_frame_focus_only_9b2b9315_20260707.jsonl`
- `factsel_fallback_unpk_not_foton_30` — source `adr003_fact_select_frame_focus_only_9b2b9315_20260707.jsonl`
- `factsel_history_axis_switch_31` — source `adr003_fact_select_frame_focus_only_9b2b9315_20260707.jsonl`
- `factsel_conflict_subject_format_32` — source `adr003_fact_select_frame_focus_only_9b2b9315_20260707.jsonl`

### hot_lead_close_5
- `cf142_pos_foton_online_info_record` — source `adr003_semantic_reading_paket1_e2_20260703.jsonl`
- `cf142_pos_unpk_camp_dates_signup` — source `adr003_semantic_reading_paket1_e2_20260703.jsonl`
- `payfix_foton_link_01` — source `adr003_focus_reask_roles_payment_20260706.jsonl`
- `wappi_pair_missing_72h_006` — source `adr003_semantic_reading_paket1_e2_20260703.jsonl`
- `wappi_pair_missing_72h_022` — source `adr003_semantic_reading_paket1_e2_20260703.jsonl`

### legit_closing_3
- `cf142_over_handoff_foton_clean_thanks` — source `adr003_semantic_reading_paket1_e2_20260703.jsonl`
- `cf142_over_handoff_unpk_clean_ready` — source `adr003_semantic_reading_paket1_e2_20260703.jsonl`
- `tz147_ft_benign_platform_01` — source `adr003_semantic_reading_paket1_e2_20260703.jsonl`

### p0_veto_2
- `p1_ft_seats_refund_p0_01` — source `adr003_semantic_reading_paket1_e2_20260703.jsonl`
- `p0_un_legal_threat_01` — source `adr003_semantic_reading_paket1_e2_20260703.jsonl`

### masker_controls_5
- `p0_model_led_neg_child_left_alone` — source `adr003_semantic_reading_paket1_e2_20260703.jsonl`
- `rz_unpk_refund_paid_dispute_02` — source `adr003_semantic_reading_paket1_e2_20260703.jsonl`
- `focus_roles_payment_dispute_real_01` — source `adr003_focus_reask_roles_payment_20260706.jsonl`
- `payfix_foton_not_paid_access_01` — source `adr003_focus_reask_roles_payment_20260706.jsonl`
- `p0_ctrl_benign_presale_refund_01` — source `adr003_semantic_reading_paket1_e2_20260703.jsonl`
