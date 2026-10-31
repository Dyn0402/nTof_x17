# Moved from the user EOS area to the n_TOF experiment area, 2026-10-31

Tool: nTof_x17 `tools/eos_move/eosmove.py` (xrdcp across instances, then size + adler32 of every file on both
sides from server-side listings; only then the user copy is removed and a symlink left at the old path).
The symlinks resolve through FUSE (`/eos/...`), not through `root://eosuser.cern.ch` (another instance):
read moved data by its new path in any xrootd URL.

**Total: 291283 files, 387.8 GB, all verified.**

| old path | new path | files | GB | links | not verified |
|---|---|---|---|---|---|
| `/eos/user/d/dneff/asym_beam` | `/eos/experiment/ntof/data/x17/analysis/user_misc/asym_beam` | 7265 | 0.36 | 0 | 0 |
| `/eos/user/d/dneff/bands` | `/eos/experiment/ntof/data/x17/analysis/user_misc/bands` | 3 | 0.01 | 0 | 0 |
| `/eos/user/d/dneff/calorimetry` | `/eos/experiment/ntof/data/x17/analysis/user_misc/calorimetry` | 59 | 0.06 | 0 | 0 |
| `/eos/user/d/dneff/cloud_basics` | `/eos/experiment/ntof/data/x17/analysis/user_misc/cloud_basics` | 181 | 0.01 | 0 | 0 |
| `/eos/user/d/dneff/mx17_hdr_audit_20261016` | `/eos/experiment/ntof/data/x17/analysis/user_misc/mx17_hdr_audit_20261016` | 8 | 2.96 | 0 | 0 |
| `/eos/user/d/dneff/pairing_sim` | `/eos/experiment/ntof/data/x17/analysis/user_misc/pairing_sim` | 80 | 0.04 | 0 | 0 |
| `/eos/user/d/dneff/plane_ratio` | `/eos/experiment/ntof/data/x17/analysis/user_misc/plane_ratio` | 91 | 0.09 | 0 | 0 |
| `/eos/user/d/dneff/satblob` | `/eos/experiment/ntof/data/x17/analysis/user_misc/satblob` | 254 | 0.01 | 0 | 0 |
| `/eos/user/d/dneff/scratch` | `/eos/experiment/ntof/data/x17/analysis/user_misc/scratch` | 130 | 0.20 | 0 | 0 |
| `/eos/user/d/dneff/seedfit` | `/eos/experiment/ntof/data/x17/analysis/user_misc/seedfit` | 20 | 0.00 | 0 | 0 |
| `/eos/user/d/dneff/sim` | `/eos/experiment/ntof/data/x17/analysis/user_misc/sim` | 21 | 0.00 | 0 | 0 |
| `/eos/user/d/dneff/test` | `/eos/experiment/ntof/data/x17/analysis/user_misc/test` | 1 | 0.00 | 0 | 0 |
| `/eos/user/d/dneff/tmp/d0ar` | `/eos/experiment/ntof/data/x17/response_sim/scratch_d0ar_2026-10` | 45 | 0.06 | 0 | 0 |
| `/eos/user/d/dneff/tmp/gx` | `/eos/experiment/ntof/data/x17/response_sim/scratch_gx_2026-10` | 68 | 30.05 | 0 | 0 |
| `/eos/user/d/dneff/vladder` | `/eos/experiment/ntof/data/x17/analysis/user_misc/vladder` | 30 | 0.01 | 0 | 0 |
| `/eos/user/d/dneff/x17/cosmics_fullpass` | `/eos/experiment/ntof/data/x17/analysis/cosmics_fullpass` | 1040 | 0.46 | 0 | 0 |
| `/eos/user/d/dneff/x17/ear2_smoke` | `/eos/experiment/ntof/data/x17/analysis/ear2_smoke` | 1 | 0.03 | 0 | 0 |
| `/eos/user/d/dneff/x17/pair_displays` | `/eos/experiment/ntof/data/x17/analysis/pair_displays` | 3 | 0.01 | 0 | 0 |
| `/eos/user/d/dneff/x17/reproc/prod_v12` | `/eos/experiment/ntof/data/x17/reproc/from_user_2026-10-31/prod_v12` | 71 | 174.83 | 0 | 0 |
| `/eos/user/d/dneff/x17/reproc/prod_v12_recover` | `/eos/experiment/ntof/data/x17/reproc/from_user_2026-10-31/prod_v12_recover` | 0 | 0.00 | 0 | 0 |
| `/eos/user/d/dneff/x17/same_arm_pairs` | `/eos/experiment/ntof/data/x17/analysis/same_arm_pairs` | 200 | 0.04 | 0 | 0 |
| `/eos/user/d/dneff/x17/sept26_calib` | `/eos/experiment/ntof/data/x17/analysis/sept26_calib` | 1040 | 0.18 | 0 | 0 |
| `/eos/user/d/dneff/x17/sept26_fullpass` | `/eos/experiment/ntof/data/x17/analysis/sept26_fullpass` | 12929 | 26.12 | 0 | 0 |
| `/eos/user/d/dneff/x17/sept26_fullpass_is2_v1` | `/eos/experiment/ntof/data/x17/analysis/sept26_fullpass_is2_v1` | 702 | 1.49 | 0 | 0 |
| `/eos/user/d/dneff/x17/sept26_fullpass_is2_v1_smoke` | `/eos/experiment/ntof/data/x17/analysis/sept26_fullpass_is2_v1_smoke` | 14 | 0.03 | 0 | 0 |
| `/eos/user/d/dneff/x17/sept26_fullpass_is2_v2` | `/eos/experiment/ntof/data/x17/analysis/sept26_fullpass_is2_v2` | 6464 | 13.36 | 0 | 0 |
| `/eos/user/d/dneff/x17/sept26_slim` | `/eos/experiment/ntof/data/x17/analysis/sept26_slim` | 310 | 5.80 | 0 | 0 |
| `/eos/user/d/dneff/x17/sept26_stage2` | `/eos/experiment/ntof/data/x17/analysis/sept26_stage2` | 12705 | 1.71 | 0 | 0 |
| `/eos/user/d/dneff/x17/sps_run53_det4_check` | `/eos/experiment/ntof/data/x17/p2_sps_july/sps_run53_det4_check` | 1531 | 100.68 | 0 | 0 |
| `/eos/user/d/dneff/x17/stack_edep` | `/eos/experiment/ntof/data/x17/analysis/stack_edep` | 194 | 0.01 | 0 | 0 |
| `/eos/user/d/dneff/x17/thermal_accounting` | `/eos/experiment/ntof/data/x17/analysis/thermal_accounting` | 225 | 0.00 | 0 | 0 |
| `/eos/user/d/dneff/x17/two_track_limit` | `/eos/experiment/ntof/data/x17/analysis/two_track_limit` | 260 | 0.09 | 0 | 0 |
| `/eos/user/d/dneff/x17dig_out` | `/eos/experiment/ntof/data/x17/analysis/user_misc/x17dig_out` | 96 | 0.01 | 0 | 0 |
| `/eos/user/d/dneff/x17rerun/Analysis_audit` | `/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/Analysis_audit` | 494 | 0.23 | 0 | 0 |
| `/eos/user/d/dneff/x17rerun/Analysis_deadseed` | `/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/Analysis_deadseed` | 38 | 0.03 | 0 | 0 |
| `/eos/user/d/dneff/x17rerun/Analysis_dryrun` | `/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/Analysis_dryrun` | 213 | 0.03 | 0 | 0 |
| `/eos/user/d/dneff/x17rerun/code` | `/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/code` | 58406 | 4.06 | 0 | 0 |
| `/eos/user/d/dneff/x17rerun/code_3f6db25` | `/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/code_3f6db25` | 8397 | 0.58 | 0 | 0 |
| `/eos/user/d/dneff/x17rerun/code_440a76c` | `/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/code_440a76c` | 8452 | 0.59 | 0 | 0 |
| `/eos/user/d/dneff/x17rerun/code_884def7` | `/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/code_884def7` | 8471 | 0.59 | 0 | 0 |
| `/eos/user/d/dneff/x17rerun/code_89000668` | `/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/code_89000668` | 2068 | 0.11 | 0 | 0 |
| `/eos/user/d/dneff/x17rerun/code_95ad31f` | `/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/code_95ad31f` | 8444 | 0.59 | 0 | 0 |
| `/eos/user/d/dneff/x17rerun/code_ba79a73` | `/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/code_ba79a73` | 8476 | 0.59 | 0 | 0 |
| `/eos/user/d/dneff/x17rerun/code_ccb272e` | `/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/code_ccb272e` | 8487 | 0.59 | 0 | 0 |
| `/eos/user/d/dneff/x17rerun/code_d93c687` | `/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/code_d93c687` | 8404 | 0.59 | 0 | 0 |
| `/eos/user/d/dneff/x17rerun/code_ddc7a60` | `/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/code_ddc7a60` | 8468 | 0.59 | 0 | 0 |
| `/eos/user/d/dneff/x17rerun/code_df8b3bd` | `/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/code_df8b3bd` | 8465 | 0.59 | 0 | 0 |
| `/eos/user/d/dneff/x17rerun/code_e383a27b` | `/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/code_e383a27b` | 2084 | 0.11 | 0 | 0 |
| `/eos/user/d/dneff/x17rerun/code_fd127cd9` | `/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/code_fd127cd9` | 2084 | 0.11 | 0 | 0 |
| `/eos/user/d/dneff/x17rerun/code_ws_t0` | `/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/code_ws_t0` | 23 | 0.00 | 0 | 0 |
| `/eos/user/d/dneff/x17rerun/code_ws_widthtilt` | `/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/code_ws_widthtilt` | 73 | 0.00 | 0 | 0 |
| `/eos/user/d/dneff/x17rerun/decode_audit` | `/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/decode_audit` | 3836 | 0.06 | 0 | 0 |
| `/eos/user/d/dneff/x17rerun/decode_audit_sps` | `/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/decode_audit_sps` | 773 | 0.01 | 0 | 0 |
| `/eos/user/d/dneff/x17rerun/det4y_ablation` | `/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/det4y_ablation` | 100 | 0.04 | 0 | 0 |
| `/eos/user/d/dneff/x17rerun/effstage` | `/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/effstage` | 58 | 1.77 | 0 | 0 |
| `/eos/user/d/dneff/x17rerun/fd_toys_pass1` | `/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/fd_toys_pass1` | 765 | 0.04 | 0 | 0 |
| `/eos/user/d/dneff/x17rerun/followups` | `/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/followups` | 56299 | 7.39 | 0 | 0 |
| `/eos/user/d/dneff/x17rerun/m3_audit` | `/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/m3_audit` | 135 | 0.14 | 12 | 0 |
| `/eos/user/d/dneff/x17rerun/m3clus` | `/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/m3clus` | 46 | 0.57 | 0 | 0 |
| `/eos/user/d/dneff/x17rerun/maskgate` | `/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/maskgate` | 65 | 0.04 | 0 | 0 |
| `/eos/user/d/dneff/x17rerun/mb_fine` | `/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/mb_fine` | 2466 | 0.00 | 0 | 0 |
| `/eos/user/d/dneff/x17rerun/model_audit` | `/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/model_audit` | 551 | 0.24 | 0 | 0 |
| `/eos/user/d/dneff/x17rerun/paper1_charge` | `/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/paper1_charge` | 330 | 0.01 | 0 | 0 |
| `/eos/user/d/dneff/x17rerun/paper1_eff` | `/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/paper1_eff` | 1167 | 0.22 | 0 | 0 |
| `/eos/user/d/dneff/x17rerun/paper1_res` | `/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/paper1_res` | 3942 | 0.11 | 0 | 0 |
| `/eos/user/d/dneff/x17rerun/pass1_backup_1030` | `/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/pass1_backup_1030` | 8 | 0.18 | 0 | 0 |
| `/eos/user/d/dneff/x17rerun/pass1_backup_2026-10-30` | `/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/pass1_backup_2026-10-30` | 284 | 0.15 | 0 | 0 |
| `/eos/user/d/dneff/x17rerun/pass1_backup_2026-10-31` | `/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/pass1_backup_2026-10-31` | 60 | 0.04 | 0 | 0 |
| `/eos/user/d/dneff/x17rerun/pass1_dry` | `/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/pass1_dry` | 215 | 0.04 | 0 | 0 |
| `/eos/user/d/dneff/x17rerun/precheck_ga` | `/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/precheck_ga` | 21 | 0.00 | 0 | 0 |
| `/eos/user/d/dneff/x17rerun/regression` | `/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/regression` | 5 | 0.02 | 0 | 0 |
| `/eos/user/d/dneff/x17rerun/s4_followup` | `/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/s4_followup` | 9453 | 0.70 | 0 | 0 |
| `/eos/user/d/dneff/x17rerun/s4_inputs` | `/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/s4_inputs` | 9422 | 1.09 | 0 | 0 |
| `/eos/user/d/dneff/x17rerun/s4_recheck` | `/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/s4_recheck` | 38 | 0.06 | 0 | 0 |
| `/eos/user/d/dneff/x17rerun/satfin` | `/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/satfin` | 322 | 0.22 | 0 | 0 |
| `/eos/user/d/dneff/x17rerun/satur` | `/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/satur` | 552 | 0.17 | 0 | 0 |
| `/eos/user/d/dneff/x17rerun/spotcheck` | `/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/spotcheck` | 418 | 0.00 | 0 | 0 |
| `/eos/user/d/dneff/x17rerun/sps_audit` | `/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/sps_audit` | 652 | 0.53 | 0 | 0 |
| `/eos/user/d/dneff/x17rerun/sps_final2_stage` | `/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/sps_final2_stage` | 142 | 0.09 | 0 | 0 |
| `/eos/user/d/dneff/x17rerun/sps_final_stage` | `/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/sps_final_stage` | 142 | 0.09 | 0 | 0 |
| `/eos/user/d/dneff/x17rerun/sps_followup` | `/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/sps_followup` | 177 | 3.73 | 0 | 0 |
| `/eos/user/d/dneff/x17rerun/t15` | `/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/t15` | 309 | 0.07 | 0 | 0 |
| `/eos/user/d/dneff/x17rerun/t16` | `/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/t16` | 10 | 0.01 | 0 | 0 |
| `/eos/user/d/dneff/x17rerun/vt_all` | `/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/vt_all` | 394 | 0.00 | 0 | 0 |
| `/eos/user/d/dneff/x17rerun/vt_check` | `/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/vt_check` | 9558 | 1.16 | 0 | 0 |
| `/eos/user/d/dneff/x17rerun/vt_correction` | `/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/vt_correction` | 107 | 0.01 | 0 | 0 |
| `/eos/user/d/dneff/x17rerun/ws_sps` | `/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/ws_sps` | 873 | 0.20 | 0 | 0 |

## Duplicates kept in the user area (60 files, 76.6 GB)

Identical (same size and adler32) to a file already in the experiment area; not copied, not deleted.

| user file | GB | adler32 | identical experiment-area file |
|---|---|---|---|
| `/eos/user/d/dneff/x17/reproc/prod_v12/224709/completed/224709/run224709_0013.root` | 2.24 | a5503f02 | `/eos/experiment/ntof/data/x17/reproc/prod_v12/224709/completed/224709/run224709_0013.root` |
| `/eos/user/d/dneff/x17/reproc/prod_v12/224709/completed/224709/run224709_0041.root` | 2.50 | e21e2bd2 | `/eos/experiment/ntof/data/x17/reproc/prod_v12/224709/completed/224709/run224709_0041.root` |
| `/eos/user/d/dneff/x17/reproc/prod_v12/224709/completed/224709/run224709_0045.root` | 2.44 | 8e62dfd3 | `/eos/experiment/ntof/data/x17/reproc/prod_v12/224709/completed/224709/run224709_0045.root` |
| `/eos/user/d/dneff/x17/reproc/prod_v12/224709/completed/224709/run224709_0046.root` | 2.51 | 2b30db2d | `/eos/experiment/ntof/data/x17/reproc/prod_v12/224709/completed/224709/run224709_0046.root` |
| `/eos/user/d/dneff/x17/reproc/prod_v12/224709/completed/224709/run224709_0053.root` | 2.63 | 3155dbc3 | `/eos/experiment/ntof/data/x17/reproc/prod_v12/224709/completed/224709/run224709_0053.root` |
| `/eos/user/d/dneff/x17/reproc/prod_v12/224709/completed/224709/run224709_0056.root` | 2.72 | a595e559 | `/eos/experiment/ntof/data/x17/reproc/prod_v12/224709/completed/224709/run224709_0056.root` |
| `/eos/user/d/dneff/x17/reproc/prod_v12/224709/completed/224709/run224709_0059.root` | 2.77 | 183fb179 | `/eos/experiment/ntof/data/x17/reproc/prod_v12/224709/completed/224709/run224709_0059.root` |
| `/eos/user/d/dneff/x17/reproc/prod_v12/224709/completed/224709/run224709_0062.root` | 2.86 | 3639f153 | `/eos/experiment/ntof/data/x17/reproc/prod_v12/224709/completed/224709/run224709_0062.root` |
| `/eos/user/d/dneff/x17/reproc/prod_v12/224709/completed/224709/run224709_0063.root` | 2.85 | eef6e915 | `/eos/experiment/ntof/data/x17/reproc/prod_v12/224709/completed/224709/run224709_0063.root` |
| `/eos/user/d/dneff/x17/reproc/prod_v12/224709/completed/224709/run224709_0064.root` | 2.87 | 75bb3c40 | `/eos/experiment/ntof/data/x17/reproc/prod_v12/224709/completed/224709/run224709_0064.root` |
| `/eos/user/d/dneff/x17/reproc/prod_v12/224709/completed/224709/run224709_0065.root` | 2.88 | 7f117ade | `/eos/experiment/ntof/data/x17/reproc/prod_v12/224709/completed/224709/run224709_0065.root` |
| `/eos/user/d/dneff/x17/reproc/prod_v12/224709/completed/224709/run224709_0067.root` | 2.93 | 6480f949 | `/eos/experiment/ntof/data/x17/reproc/prod_v12/224709/completed/224709/run224709_0067.root` |
| `/eos/user/d/dneff/x17/reproc/prod_v12/224709/completed/224709/run224709_0068.root` | 2.95 | 8ad0c4ba | `/eos/experiment/ntof/data/x17/reproc/prod_v12/224709/completed/224709/run224709_0068.root` |
| `/eos/user/d/dneff/x17/reproc/prod_v12/224709/completed/224709/run224709_0071.root` | 2.54 | 8dd751c3 | `/eos/experiment/ntof/data/x17/reproc/prod_v12/224709/completed/224709/run224709_0071.root` |
| `/eos/user/d/dneff/x17/reproc/prod_v12/224709/completed/224709/run224709_0072.root` | 2.51 | ceba3989 | `/eos/experiment/ntof/data/x17/reproc/prod_v12/224709/completed/224709/run224709_0072.root` |
| `/eos/user/d/dneff/x17/reproc/prod_v12/224709/completed/224709/run224709_0074.root` | 2.87 | cbccc559 | `/eos/experiment/ntof/data/x17/reproc/prod_v12/224709/completed/224709/run224709_0074.root` |
| `/eos/user/d/dneff/x17/reproc/prod_v12/224709/completed/224709/run224709_0075.root` | 2.87 | 21278e9a | `/eos/experiment/ntof/data/x17/reproc/prod_v12/224709/completed/224709/run224709_0075.root` |
| `/eos/user/d/dneff/x17/reproc/prod_v12_recover/224526/history_224526.root` | 0.00 | 5a65469e | `/eos/experiment/ntof/data/x17/reproc/prod_v12_recover/224526/completed/224526/history_224526.root` |
| `/eos/user/d/dneff/x17/reproc/prod_v12_recover/224526/run224526_0001.root` | 0.70 | f6607014 | `/eos/experiment/ntof/data/x17/reproc/prod_v12_recover/224526/completed/224526/run224526_0001.root` |
| `/eos/user/d/dneff/x17/reproc/prod_v12_recover/224526/run224526_0002.root` | 0.75 | 7a34bfb6 | `/eos/experiment/ntof/data/x17/reproc/prod_v12_recover/224526/completed/224526/run224526_0002.root` |
| `/eos/user/d/dneff/x17/reproc/prod_v12_recover/224526/run224526_0003.root` | 0.76 | d41eb2cc | `/eos/experiment/ntof/data/x17/reproc/prod_v12_recover/224526/completed/224526/run224526_0003.root` |
| `/eos/user/d/dneff/x17/reproc/prod_v12_recover/224526/run224526_0004.root` | 0.75 | d6c315f7 | `/eos/experiment/ntof/data/x17/reproc/prod_v12_recover/224526/completed/224526/run224526_0004.root` |
| `/eos/user/d/dneff/x17/reproc/prod_v12_recover/224526/run224526_0005.root` | 0.75 | 4effd944 | `/eos/experiment/ntof/data/x17/reproc/prod_v12_recover/224526/completed/224526/run224526_0005.root` |
| `/eos/user/d/dneff/x17/reproc/prod_v12_recover/224526/run224526_0006.root` | 0.76 | fda8f34f | `/eos/experiment/ntof/data/x17/reproc/prod_v12_recover/224526/completed/224526/run224526_0006.root` |
| `/eos/user/d/dneff/x17/reproc/prod_v12_recover/224526/run224526_0007.root` | 0.75 | da9f9334 | `/eos/experiment/ntof/data/x17/reproc/prod_v12_recover/224526/completed/224526/run224526_0007.root` |
| `/eos/user/d/dneff/x17/reproc/prod_v12_recover/224526/run224526_0008.root` | 0.75 | aad7b4a2 | `/eos/experiment/ntof/data/x17/reproc/prod_v12_recover/224526/completed/224526/run224526_0008.root` |
| `/eos/user/d/dneff/x17/reproc/prod_v12_recover/224526/run224526_0009.root` | 0.75 | 4ce75d3b | `/eos/experiment/ntof/data/x17/reproc/prod_v12_recover/224526/completed/224526/run224526_0009.root` |
| `/eos/user/d/dneff/x17/reproc/prod_v12_recover/224526/run224526_0010.root` | 0.74 | 6157353e | `/eos/experiment/ntof/data/x17/reproc/prod_v12_recover/224526/completed/224526/run224526_0010.root` |
| `/eos/user/d/dneff/x17/reproc/prod_v12_recover/224526/run224526_0011.root` | 0.76 | 09447b0f | `/eos/experiment/ntof/data/x17/reproc/prod_v12_recover/224526/completed/224526/run224526_0011.root` |
| `/eos/user/d/dneff/x17/reproc/prod_v12_recover/224526/run224526_0012.root` | 0.75 | 2834b7ec | `/eos/experiment/ntof/data/x17/reproc/prod_v12_recover/224526/completed/224526/run224526_0012.root` |
| `/eos/user/d/dneff/x17/reproc/prod_v12_recover/224526/run224526_0013.root` | 0.75 | d707b554 | `/eos/experiment/ntof/data/x17/reproc/prod_v12_recover/224526/completed/224526/run224526_0013.root` |
| `/eos/user/d/dneff/x17/reproc/prod_v12_recover/224526/run224526_0014.root` | 0.76 | 5b74cff9 | `/eos/experiment/ntof/data/x17/reproc/prod_v12_recover/224526/completed/224526/run224526_0014.root` |
| `/eos/user/d/dneff/x17/reproc/prod_v12_recover/224526/run224526_0015.root` | 0.75 | cb4fff12 | `/eos/experiment/ntof/data/x17/reproc/prod_v12_recover/224526/completed/224526/run224526_0015.root` |
| `/eos/user/d/dneff/x17/reproc/prod_v12_recover/224526/run224526_0016.root` | 0.75 | 3c37367a | `/eos/experiment/ntof/data/x17/reproc/prod_v12_recover/224526/completed/224526/run224526_0016.root` |
| `/eos/user/d/dneff/x17/reproc/prod_v12_recover/224526/run224526_0017.root` | 0.75 | 8a105e93 | `/eos/experiment/ntof/data/x17/reproc/prod_v12_recover/224526/completed/224526/run224526_0017.root` |
| `/eos/user/d/dneff/x17/reproc/prod_v12_recover/224526/run224526_0018.root` | 0.75 | 5c2c12cf | `/eos/experiment/ntof/data/x17/reproc/prod_v12_recover/224526/completed/224526/run224526_0018.root` |
| `/eos/user/d/dneff/x17/reproc/prod_v12_recover/224526/run224526_0019.root` | 0.76 | 60c138d3 | `/eos/experiment/ntof/data/x17/reproc/prod_v12_recover/224526/completed/224526/run224526_0019.root` |
| `/eos/user/d/dneff/x17/reproc/prod_v12_recover/224526/run224526_0020.root` | 0.75 | fdcc5d63 | `/eos/experiment/ntof/data/x17/reproc/prod_v12_recover/224526/completed/224526/run224526_0020.root` |
| `/eos/user/d/dneff/x17/reproc/prod_v12_recover/224526/run224526_0021.root` | 0.75 | 543c94a0 | `/eos/experiment/ntof/data/x17/reproc/prod_v12_recover/224526/completed/224526/run224526_0021.root` |
| `/eos/user/d/dneff/x17/reproc/prod_v12_recover/224526/run224526_0022.root` | 0.75 | 2390c07b | `/eos/experiment/ntof/data/x17/reproc/prod_v12_recover/224526/completed/224526/run224526_0022.root` |
| `/eos/user/d/dneff/x17/reproc/prod_v12_recover/224526/run224526_0023.root` | 0.75 | 8e48e076 | `/eos/experiment/ntof/data/x17/reproc/prod_v12_recover/224526/completed/224526/run224526_0023.root` |
| `/eos/user/d/dneff/x17/reproc/prod_v12_recover/224526/run224526_0024.root` | 0.76 | 907fb350 | `/eos/experiment/ntof/data/x17/reproc/prod_v12_recover/224526/completed/224526/run224526_0024.root` |
| `/eos/user/d/dneff/x17/reproc/prod_v12_recover/224526/run224526_0025.root` | 0.75 | b22bf14e | `/eos/experiment/ntof/data/x17/reproc/prod_v12_recover/224526/completed/224526/run224526_0025.root` |
| `/eos/user/d/dneff/x17/reproc/prod_v12_recover/224526/run224526_0026.root` | 0.75 | 5ef2eb91 | `/eos/experiment/ntof/data/x17/reproc/prod_v12_recover/224526/completed/224526/run224526_0026.root` |
| `/eos/user/d/dneff/x17/reproc/prod_v12_recover/224526/run224526_0027.root` | 0.75 | d7e1ab0a | `/eos/experiment/ntof/data/x17/reproc/prod_v12_recover/224526/completed/224526/run224526_0027.root` |
| `/eos/user/d/dneff/x17/reproc/prod_v12_recover/224526/run224526_0028.root` | 0.75 | 879621a4 | `/eos/experiment/ntof/data/x17/reproc/prod_v12_recover/224526/completed/224526/run224526_0028.root` |
| `/eos/user/d/dneff/x17/reproc/prod_v12_recover/224526/run224526_0029.root` | 0.76 | 29600066 | `/eos/experiment/ntof/data/x17/reproc/prod_v12_recover/224526/completed/224526/run224526_0029.root` |
| `/eos/user/d/dneff/x17/reproc/prod_v12_recover/224526/run224526_0030.root` | 0.75 | 8b8f7016 | `/eos/experiment/ntof/data/x17/reproc/prod_v12_recover/224526/completed/224526/run224526_0030.root` |
| `/eos/user/d/dneff/x17/reproc/prod_v12_recover/224526/run224526_0031.root` | 0.69 | 6f130279 | `/eos/experiment/ntof/data/x17/reproc/prod_v12_recover/224526/completed/224526/run224526_0031.root` |
| `/eos/user/d/dneff/x17/reproc/prod_v12_recover/224526/run224526_0032.root` | 0.57 | ad627507 | `/eos/experiment/ntof/data/x17/reproc/prod_v12_recover/224526/completed/224526/run224526_0032.root` |
| `/eos/user/d/dneff/x17/reproc/prod_v12_recover/224526/run224526_0033.root` | 0.76 | b5a3efbc | `/eos/experiment/ntof/data/x17/reproc/prod_v12_recover/224526/completed/224526/run224526_0033.root` |
| `/eos/user/d/dneff/x17/reproc/prod_v12_recover/224526/run224526_0034.root` | 0.63 | 28ac8039 | `/eos/experiment/ntof/data/x17/reproc/prod_v12_recover/224526/completed/224526/run224526_0034.root` |
| `/eos/user/d/dneff/x17/reproc/prod_v12_recover/224526/run224526_0035.root` | 0.70 | f59ef2cc | `/eos/experiment/ntof/data/x17/reproc/prod_v12_recover/224526/completed/224526/run224526_0035.root` |
| `/eos/user/d/dneff/x17/reproc/prod_v12_recover/224526/run224526_0036.root` | 0.72 | 3553c7af | `/eos/experiment/ntof/data/x17/reproc/prod_v12_recover/224526/completed/224526/run224526_0036.root` |
| `/eos/user/d/dneff/x17/reproc/prod_v12_recover/224526/run224526_0037.root` | 0.62 | 5278dfab | `/eos/experiment/ntof/data/x17/reproc/prod_v12_recover/224526/completed/224526/run224526_0037.root` |
| `/eos/user/d/dneff/x17/reproc/prod_v12_recover/224526/run224526_0038.root` | 0.74 | aa5e9d1f | `/eos/experiment/ntof/data/x17/reproc/prod_v12_recover/224526/completed/224526/run224526_0038.root` |
| `/eos/user/d/dneff/x17/reproc/prod_v12_recover/224526/run224526_0039.root` | 0.75 | 28ec8461 | `/eos/experiment/ntof/data/x17/reproc/prod_v12_recover/224526/completed/224526/run224526_0039.root` |
| `/eos/user/d/dneff/x17/reproc/prod_v12_recover/224526/run224526_0040.root` | 0.75 | 6c3326a6 | `/eos/experiment/ntof/data/x17/reproc/prod_v12_recover/224526/completed/224526/run224526_0040.root` |
| `/eos/user/d/dneff/x17/reproc/prod_v12_recover/224526/run224526_0041.root` | 0.76 | bb38a6fc | `/eos/experiment/ntof/data/x17/reproc/prod_v12_recover/224526/completed/224526/run224526_0041.root` |
| `/eos/user/d/dneff/x17/reproc/prod_v12_recover/224526/run224526_0042.root` | 0.41 | 4802f0a9 | `/eos/experiment/ntof/data/x17/reproc/prod_v12_recover/224526/completed/224526/run224526_0042.root` |
