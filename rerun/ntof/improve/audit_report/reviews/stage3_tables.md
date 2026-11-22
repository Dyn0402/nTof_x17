<!-- Verbatim report of the 'Review stage-3 table code' code review, part of the 2026-11-21 n_TOF tracking audit (session be7f0e66).
     Scratch scripts it cites under /home/dylan/.claude/jobs/be7f0e66/tmp/ were copied to
     $WORK_DATA/ntof_pass1/ana/tracking_audit/review_scripts/. -->

# Review of the n_TOF pass-1 table-level chain (stage 3 → v(t)/κ → map → censor → analysis view)

Nothing in the repo or under `/media/ucla/work/ntof_pass1` was modified. Scratch scripts are in `/home/dylan/.claude/jobs/be7f0e66/tmp/review_stage3/`.

The geometry, the rescale, the map application and the gates are all internally consistent. The real problems are a sign convention in the view's `tan_raw_y`, and missing time since flash on 14.7 % of beam rows, which also leaks beam events into the "cosmic" lines.

## Findings, ranked

### 1. The view's `tan_raw_y` has the opposite sign from the stage-3 contract, and the scint stack mirrors the y tangent — CONFIRMED
- **Where:** `rerun/ntof/stage3/vt_view.py:30` (`RAW = {'tan_raw_y': 'tany_vt'}`) and `rerun/ntof/stage3/map_view.py:125`.
- **What is wrong:** In stage 3, `tan_raw_y = +y_tan_theta` (reco sign) and `tany = TAN_SIGN_Y · tan_raw_y` (`build_tracks.py:260, 273`). The view sets `tan_raw_y = tany`, already in the local sign. Inside the same table, `tan_raw_y_frozen` still has the reco sign.
- **Consumer that breaks:** `ntof_scint_stack/extract.py:137` computes `tan_v_raw = TAN_SIGN_Y * tan_raw_y`, which flips the sign a second time.
- **Checks:**
  - `ana/scint_stack/tracks/stack_run_145.parquet` joined to the view gives `tan_v_raw / tany = −1.0000` on A, C and D (23 k gated tracks).
  - `ana/scint_stack/ana/pointing.parquet` gives `lam_v = 0.000` on A, C and D, clipped at the grid floor, with the residual rising monotonically with λ. The is2-era product gave 0.87–1.03 (A, B, C) and 0.61 (D).
  - `scales.parquet` wall-v reads s = −0.4 to −16.
  - The same holds in `ana_nomap`.
- **Consequence:** Every pass-1 scint-stack v prediction is either not extrapolated (λ_v = 0) or extrapolated the wrong way where `tan_v_raw` is used directly (`ana.py:384`, `ana.py:821`, `ntof_calorimetry/wall_plastic.py:216`).
  - At the wall (97 mm lever) the error is about 97 · 0.85 · tan_y, roughly 17 mm rms and 40 mm at |tan| = 0.5.
  - At the plastic and LS it is about twice that.
  - The u-view numbers (λ = 0.847 etc.) are not affected.
- **Not checked for sign use:** other readers of `tan_raw_y` (`ntof_calorimetry/mip_sample.py:386`, `ntof_cosmics/*`).

### 2. 14.7 % of beam rows have no time since flash, and null time is then read as "beam-off" — CONFIRMED
- **Where:** `sept26_prelim_analysis/trigger_time.py:174–181` and `build_tracks.py:533–538` fill NaN silently (only the sidecar's `n_no_trigger_in_slim` records it). `rerun/ntof/improve/line_residuals.py:48` and `bd_map.py:43` select beam-off as `t_since_flash_ns is null`.
- **Size of the gap (all 543 view tables):**
  - 6.86 M of 46.7 M production rows are untimed, and 1.83 M of 12.45 M gated tracks.
  - 187 of 293 production sub-runs are affected; 454 of 3,233 acquisitions are entirely untimed.
  - Worst runs: run_114 35 %, run_162 26 %, run_147 21 %.
- **Cause as far as I could trace it:** The trigtime and slim products cover only part of those sub-runs. For run_116/stat090_0027 they cover events 1–7231 of 110,526. I could not reach EOS to see why; `trigger_time.slim_path` takes only the first `*.root` of the directory, which is one candidate.
- **Join itself is correct:** Event ids align with the old stage 1 (99.5 % against 40 % for id ± 1), and untimed rows have null `wall_*`, `coinc_*` and `bunch`, not False.
- **Consequence (a):** Every ToF-gated study drops these tracks. A yield per trigger or per proton is low by the same fraction unless the denominator uses covered triggers only.
- **Consequence (b):** `lines.parquet` contains beam events.
  - 14,742 of 46,171 lines (32 %) come from 32 production runs. The README's "72 beam-off runs" is 40 cosmic plus 32 production.
  - Their x residual MAD is 0.128, against 0.029 for real cosmics.
  - README finding 1 (13–14 % of views off by more than 0.15) is 5.7–7.0 % on cosmics alone and 28–32 % on the beam part.
  - 15 % of the map's clean training lines are beam events.
- **Effect on the shipped map is small:** I refitted on cosmic-only lines. On late run_145 tracks Δtan is 0.0007–0.0010 rms (0.15–0.24 mm of pointing), and the scale change is below 0.1 %. The largest per-cell changes are |Δa| = 0.021 and |Δs| = 0.055.
- **Not affected:** `bd_lines.parquet` (0 production lines) and κ (`vt_line_campaign.py` uses the registry class).

### 3. The acceptance edge on D (and slightly C) in early runs is set by the reco cut, not by |tan| < 0.5 — CONFIRMED, small
- **Where:** `vt_join.py:23–27` argues the reco cut's true reach stays at or above 0.52. After κ, v(t) and the map this no longer holds.
- **Check:** The raw 0.6 cut maps to 0.48–0.50 for 25–35 % of D's gated tracks (x view, runs 79 and 104) and for 2–5 % of C's. In run_104 D cells with reach below 0.495, the count ratio of |tan| 0.47–0.50 to 0.44–0.47 is 0.17–0.41, against 0.82–0.88 elsewhere.
- **Consequence:** A time- and position-dependent loss in the last 4 % of the tan range on D. Runs 145 and 149 are clean.

### 4. Stage 3 ignores the bow mesh height that the reco stores — PLAUSIBLE
- **Where:** `build_tracks.py:256–263` places `x_p0`/`y_p0` at w = 0. The reco writes `x_bow_h`/`y_bow_h` with "position at the nominal plane = p0 − tan · h" (`wft/reco.py:461, 2167`; `wft/compat.py:46`), and the n_TOF bundles carry `h_x`/`h_y`.
- **Size:**
  - Late beam tracks: |tan · h| median 0.05–0.14 mm, 95th percentile 0.4–0.9 mm.
  - Cosmics: 95th percentile 0.9–2.1 mm.
  - Mean h / 235 mm is 0.2–0.45 % for beam tracks, which acts as a lever-arm (focus-scale) error.
  - On the A–C line the lever arm is short by h_A + h_C ≈ 2.7 mm of 469, about 0.6 % on the joined slope, which κ and the map's scale term absorb.
- **Limit:** Whether the bench h is physical at n_TOF is not established; `bow_check.py` explicitly left it out. The candidates files do not carry `bow_h`, so applying it needs `wft.reco.bow_h` recomputed.

### 5. The stage-1 merge duplicates rows — CONFIRMED, 2 rows
- **Where:** `build_tracks.py:515–517` merges on `event_id` with no `validate`.
- **Check:** The old stage 1 has two eventIds with two classes each (NONE and SINGLE). The duplicated rows are run_139/stat090_0002 C event 16717 (gated, counted twice) and run_139/stat090_0004 A event 41694.

### 6. The v(t)/κ rescale also scales the bow tilt term — PLAUSIBLE, small
- **Where:** `vt_join.py:113–114` multiplies the whole reco tan by v_used/(κ · v_t). The bow tilt a(U, W) inside that tan is v-independent by construction.
- **Size:** The error is a · (r − 1). On C it is about 0.002 rms and 0.005 at most (a rms 0.024, max 0.066; r − 1 up to 7 %). On D it is about 0.002 rms.

### 7. Re-run hazards (no current data error)
- `map_view.py` defaults to `--src stage3_vt`, which is now the mapped view. `apply()` has no guard and would map twice and overwrite the `_nomap` columns.
- `vt_view.py --force` with its default `--out` would replace the analysis view with the unmapped, uncensored one.
- `line_residuals build` now reads the mapped and censored view, so a re-run of `angle_map fit` would fit a residual map that `map_view` then applies to unmapped input.
- Current products are fine: view tans equal the shipped map applied once to the no-map tans (max difference 0, all six maps), and `lines.parquet` and `bd_lines.parquet` were built on no-map slopes.

## Checked and found correct
- **Signs and frames:** `R` equals `U_HAT`/`W_HAT` for all four arms in runs 79, 104, 145 and 149. The x position flip, the y tan flip and the direction built toward the cathode agree with the model (`p0 + w·u`, later charge is deeper).
- **External sign checks:** corr(position, tan) is positive in both views on A, C and D, and the late y target is about −27 to −29 mm.
- **Geometry recomputation:** `p0_*`, `d_*`, `target_*`, `dca_axis_mm`, `angle_to_beam_deg`, `path_len_mm`, `q_per_len`, `drift_len_mm`, `in_bore` and the `pred_*` columns, recomputed independently from `x_p0, y_p0, tanx, tany`, match the stored values to 1e-8 for the view, `_nomap` and `_frozen` sets.
- **Reference depth of `p0`:** The model's `p0` is the position of charge arriving at `t0`, which is the mesh, and stage 3 places it at w = 0. The fitted column starts within ±15 ns of `t0` on A and 25–50 ns early on C. On D it starts about 55 ns late, which is either 1.8 mm of depth or a 6 % error in D's v_t; I could not separate the two.
- **v(t)/κ:** `tan_nomap = tan_frozen · v_used / (κ · v_t)` to 2e-6 on A, C and D.
- **Map:** The sign g per view (C y and D x are −1), the (x_local, y_local) lookup and the global-slope frame are the same in fit and application. The lookup is clamped at the outer cell centres, never extrapolated. All 64 cells are fitted on A and C; 2 of 16 on D fall back to the chamber-wide value.
- **Gates, all 543 tables (49.6 M rows):** `gated = gated_reco & |tan| < 0.5 (mapped) & ~censored_window`, with B as `gated_reco & ~censored`. Zero mismatches. `censored_window` equals its rule, and the join asserts `n_strips` equality.
- **Row counts and keys:** Identical across `stage3`, `stage3_vt_nomap`, `stage3_vtmap` and `stage3_vt`. No event id appears in two tags of a sub-run.
- **Time base:** In 40 trigtime files, 23,424 of 23,425 bunches have exactly one flash trigger at t = 0. The y guard passes on 291 of 293 production sub-runs; one is insufficient and one has no time product.
- **Readers:** `pass1_tracks.py`, `paths.py` (the `stage3_pass1` link points at `stage3_vt`) and `reco_tree.py` have no errors. The reco tree carries frozen-v reco tans, with no κ and no map.
