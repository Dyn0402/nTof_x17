# Y_FLIP_AUDIT — what the stage-3 y mirror touched, and what the fix changes

**Started 2026-10-13** with the code fix (branch `y-flip-fix`). Frame, evidence and the rule are in
`COORDINATES.md`; the plan is `HANDOFF_Y_FLIP_FIX.md`.

> **State, 2026-10-13: the code fix is done. The data has NOT been regenerated yet.** The stage-2/3 tables, the
> slim, the scint stack and the g4_digi samples live only on the desktop, which was offline when the fix was made.
> All the reruns are in one chain script, `Y_FIX_REGEN.sh`, to run there. Every "after" value below marked
> **pending** comes from that run. Until it has run, **every stored stage-3 table on disk is still mirrored**.
> Flag-aware loaders (listed in §2) read those tables correctly anyway. Every other reader still sees the mirror.

## 0 · The fix in one paragraph

`build_tracks.local_and_global` now writes `y_local = +(y_p0 − 199.29)` and `tany = −y_tan_theta · k`. The
constants are `run145_target_imaging.IN_PLANE_SIGN_Y = +1` and `TAN_SIGN_Y = −1`, with helpers `local_y` and
`local_tan_y`. The x view is untouched. Relative to the 2026-09-07 convention this is an exact global y mirror:
`p0_y`, `d_y`, `tany`, `target_y_mm` and `y_local` change sign and nothing else moves. A unit test checks this
(`sept26_prelim_analysis/tests/test_y_convention.py`). The stage-3 schema is now `sept26_prelim/tracks/2`, and every
sidecar says `y_convention: "up"`.

The quantities that are *not* a pure sign flip are those that meet y-asymmetric geometry. They are recomputed on
regeneration:
- `in_bore` (the He-3 gas spans y −29.5…+50.7);
- `pred_sipm_bar`, `pred_plastic`, `pred_ls`, `pred_n_cross`, `pred_sipm_s_mm` (scintillator extents in v);
- every y-window cut and every acceptance that is not y-symmetric.

## 1 · The guard

| check | where | runs | expected after the fix |
|---|---|---|---|
| late (≥ 20 ms) A/C y band crossing < 0 at > 3σ, i.e. at the capsule nose | `y_guard.late_y_crossing` | in **every** `build_tracks.build` (refuses to write on a mirror unless `--allow-y-fail`), and on the combined table in `campaign_tracks` | ≈ −31 (A) / −35 (C) mm |
| cosmic muons hit the truly higher arm first (run_149 A–C lines, wall mean-time) | `y_orientation_check.py verdict`, `test_cosmic_muons_hit_the_higher_arm_first` | `Y_FIX_REGEN.sh` steps `cosmic`, `verdict`, `tests_data` | stored-higher arm **earlier**, p ≪ 1e-3 (it was later, 88 % vs 7 %, p = 4e-11) |
| the fix is an exact mirror of the 09-07 convention; both views obey strip(w) = p0 + tan_raw·k·w; y strip = HALF + global y | `tests/test_y_convention.py` | pytest (10 tests: 9 pass now, the data test skips until run_149 is rebuilt) | pass |
| each regenerated table is an exact y mirror of the old one, row by row | `regen_stage3.compare` | every stage-3 rebuild | `y` exact, x/z/gates identical; `pred_*`, `in_bore` counted |

## 2 · Code changed

| file | was | now |
|---|---|---|
| `ntof_tracking/run145_target_imaging.py` | `track_lines` and the y pointing fit used `IN_PLANE_SIGN` (the x sign) on the y position, with a raw tan | owns `IN_PLANE_SIGN_Y = +1`, `TAN_SIGN_Y = −1`, `local_y`, `local_tan_y`; used in both places |
| `sept26_prelim_analysis/build_tracks.py` | `yl = −(y_p0 − H)`, `ty = +raw·k` | `yl = +(y_p0 − H)`, `ty = −raw·k`; schema 2, `y_convention`, `y_guard` in the sidecar; `stored_y_sign()` / `y_up()` for loaders |
| `campaign_tracks.py` | skipped any table newer than its inputs | also rebuilds a table without `y_convention: up`; refuses to combine mixed conventions; guard on the combined table; `--k-source` |
| `y_guard.py` (new), `regen_stage3.py` (new), `tests/test_y_convention.py` (new), `Y_FIX_REGEN.sh` (new) | — | guard, regeneration with mirror proof, tests, chain |
| `pairs.py`, `make_figures.py`, `hit_maps.py` | y position via `IN_PLANE_SIGN_Y`, raw tan | tan via `TAN_SIGN_Y` (directions now consistent with the flipped position) |
| `k_arm.py` | sample `yl` mirrored, `ty` raw | `yl = local_y`, `ty = local_tan_y`. The k objective is XZ-only, so **k is unchanged** |
| `capsule_y.py` | reads `k_arm`'s `yl`/`ty` | unchanged code; now in true y through `k_arm` |
| `beam_cache.py` | corridor sign from one constant | position and tan signs separate; corridor bit-identical to before |
| `k_robustness.Masks._raw` | x sign applied to y | `IN_PLANE_SIGN_Y` |
| `normal_incidence.py` | x sign applied to y | `local_y` |
| `ntof_cosmics/g4_digi/digitise.py` | y strip = `HALF − v` (matched the mirror) | y strip = `HALF + v` (hardware); `strip_v_sign` stamped per row; `y_view()` reads either vintage |
| `g4_digi/analyse.py`, `residual_checks.py`, `capsule_tof.py` (g4 step), `capsule_pose.py` (g4) | sim y as `−(y_p0 − H)` | `digitise.y_view` |
| `capsule_tof.load` | stored frame | true y via `build_tracks.y_up` (flag-aware) |
| `capsule_pose.load_data` | mirrored on load, unconditionally (would double-flip after the fix) | delegates to `capsule_tof.load`; no mirror of its own |
| `det_a_intra.tracks`, `focus_split.load`, `make_overhead_figure.noisy_columns`, `y_image.one_run` | stored frame | `y_up` on load (flag-aware) |
| `det_a_intra` pairs output | — | `y_up = True` column; `same_chamber_physics`/`compare` refuse pairs without it |
| `same_chamber_physics.CAPSULE_V` / `same_chamber_compare.CAPSULE_SHIFT_MM` | +32.2 mm shift (the mirror) | −4.0 mm (capsule_pose: ~4 mm below the Geant4 placement), one constant |
| `ntof_scint_stack/extract.py` | `v_mm = stored y_local`, raw tan | true y via `y_up`; adds `tan_v_raw = −tan_raw_y`; schema 5, `y_convention` |
| `ntof_scint_stack/ana.py` | `CAPSULE_XYZ` y = +31.3; v extrapolation with `tan_raw_y` | y = −31.3; `tan_v_raw`; schema-4 extracts are mirrored on load |
| `ntof_calorimetry/mip_sample.build_beam` | — | refuses a stage-3/scint-stack convention mismatch |
| `y_orientation_check.py` | `cosmic` printed counts only; `pmt` was the `else` | `verdict` step and `cosmic_verdict()`; `pmt` explicit |

**Not changed, on purpose.**
- `ntof_tracking/run145_event_display_impl.py`, `run79_event_display.py`, `run79_wall_segment_gif.py`: July/August
  displays in the raw strip frame in **both** views (they predate the 08-20 x fix as well). They are historical, with
  self-consistent raw coordinates. Do not reuse them for geometry.
- `ntof_active_area/scint_acceptance.py`: run_79, raw frame in both views, consistent.
- The ~60 readers that take stage-3 columns as stored. They become correct when the tables are regenerated, with no
  code change. Until then they are mirrored, as they have been since 09-07.

## 3 · The audit table

Verdict classes (handoff §1.3): **U** unaffected · **Co** cosmetic (axis/label mirrored) · **N** numbers change ·
**C** conclusion changes. "after" values in *italics* follow exactly from the mirror. **pending** means
`Y_FIX_REGEN.sh` must produce them.

| # | product | y it uses | class | before (mirrored) | after | action |
|---|---|---|---|---|---|---|
| 1 | **Capsule height** — `capsule_y.py`, `HANDOFF_CAPSULE_Y.md` Outcome, `<out>/capsule_y/` | band crossing of tan_y vs v; acceptance ray trace | **C** | y_s ≈ **+33 ± 3** (A +31.1, C +35.6, D +29.4); "the CAD +0.8 is wrong by ~30 mm" | band crossing *A −31.1, C −35.6, D −29.4* → y_s ≈ −33 ± 3, **at the nose**. Geant4's nose by the same estimator: −28; capsule_pose: ~4 mm below the Geant4 placement. **The CAD/Geant4 height is right to ~5 mm. The "30 mm" was the mirror.** Ray-trace fits (+39/+31): **pending** (scintillators are not y-symmetric) | Outcome box corrected (retraction note); rerun `capsule_y --campaign` |
| 1b | Athens side view, `ntof_athens_26/README.md`, `sideview_run145.json` `capsule_y_calibration` | band crossings; CAD polycone moved +30.5 mm | **C** | capsule drawn at +31.3 ± 0.7, "sits ~30 mm up the beam" | *−31.3 ± 0.7*; the shift on the CAD becomes ≈ −4…−5 mm, not +30.5 | README errata; rerun `make_overhead_figure --projection y` |
| 2 | **Same-chamber pairs, vertex y** — `same_chamber_physics`, `same_chamber_compare`, `cmp_vertex_y.csv`, report/deck | `vy_mm`, `CAPSULE_V`, the 32.2 mm G4 shift | **C** (vertex location) / **U** (angles, 1:8 rate) | "A: 73 of 91 excess pairs at y 0–75 mm, capsule +3…+83"; G4 moved +32.2 | excess sits at *y −75…0* (the nose end); capsule window −33.5…+46.7; G4 shift −4. Already retracted as an artefact on 10-13 (early-ToF junk), so the physics conclusion does not move. Vertex-y comparison: **pending** | rerun det_a_intra ×6, physics, compare, report, deck |
| 3 | **Scintillator stack** — `ntof_scint_stack`, scint-stack note | `v_mm`, `CAPSULE_XYZ` y, v extrapolation | **C** (liquid open question) / **N** (v-maps) | "vertical liquids B, C show no rise toward their PMT (C: 0.07 % top vs 0.34 % bottom)" | v-maps flip: C is **0.34 % top vs 0.07 % bottom**, a rise toward the PMT, as A and D show. **The open question closes.** Matching rates: expected to barely move (`LS_OFFSET_V` ≤ 0.7 mm) — **pending** | HANDOFF next-step 1 closed; rerun extract/ana/report; republish note |
| 4 | **Capsule focus split** — `focus_split.py`, HANDOFF_TRACKING §21–22 | `target_y_mm` in `CAPSULE_Y = (−29.5, 50.7)` | **N** (record) | "capsule height inside / outside 1.100 / 1.092" selected the **valve** end of the mirrored image | inside/outside re-selected on true y: **pending**. k itself is y-free (unchanged) | rerun focus_split |
| 5 | Two-chamber pairs, vertex imaging, IPC/X17 opening angles (`pairs.py`, `pair_qa`, `source_imaging`, `pair_vertex_imaging/*`, `opening_angle`, acceptance, fold) | angles; vertex y; y windows | **U** (angles) / **N** (vertex-y images, any acceptance with the source at +33) | — | angle results unchanged by construction. y images flip (*Co*). Acceptance with the source at ±33: **pending** (geometry is near y-symmetric; expect small) | rerun after regeneration; y_image is flag-aware |
| 6 | Tracking QA web (`trackqa_shards`, `/x17/qa-tracks.html`) | y displays, `target_y_mm` | **Co** | mirrored y panels | flipped | regenerate shards (`--force`) |
| 7 | Hit maps, chamber B (`hit_maps.py`, `chamber_b.py`) | v from `IN_PLANE_SIGN_Y` | **Co** | maps upside down | correct on next run (code reads stage 2, now +1) | rerun when next needed |
| 8a | **Athens talk figures** (`make_overhead_figure --projection y`, `y_image.py`) | side view, capsule height | **C** (as shown) | capsule "30 mm up the beam" | erratum: the capsule is where the CAD puts it, to ~5 mm | erratum on the board and in the README; do not edit the talk |
| 8b | MPGD26 slides (`mpgd26/`) | — | **U** | the deck has no side view or height claim (grep: no `p0_y`/`target_y`/`IN_PLANE_SIGN_Y`; `make_run145_pointing` is XZ) | — | none |
| 8c | Site notes: pair-vertex-y, pair-vertex-capsule-image, capsule-imaging, same-chamber-pairs, scint-stack, ac-insitu-angles, beam-off-cosmics | heights, v-maps | **C/Co** | — | republish each from its generator after the rerun; add an erratum line to any that quoted +33 | **pending** |
| 9 | **Before 2026-09-07**: `run145_target_imaging` era (08-20 → 09-07), July hits analysis | y position raw (right), tan raw (direction mirrored) | **N** (3D directions) | `target_y_mm`, `angle_to_beam_deg`, LS/plastic crossings and inter-chamber opening angles had a mirrored vertical direction | every sept26 pair/opening-angle result is from 79429e0 (09-07 23:01) or later: `pairs` b3f0aa9 23:05, `opening angle` 293c46b 09-08. **Redone, confirmed.** They were mirrored as a whole (consistent), and are now right in position and direction | none |
| 10 | Geant4 comparisons using the frame shift | `+32.2`, `HE3_GAS_Y` | **C** | G4 moved +32 mm to match data | data sit ≈ 4 mm below G4 nominal; shift −4 | done in code (#2) |
| 11 | `capsule_pose.py`, HANDOFF_CAPSULE_POSE_FIT Outcome | mirrored on load | **U** (results) | it already mirrored on load, so its numbers are in true y | identical numbers; now read through the flag (no double flip) | none beyond the loader |
| 12 | `capsule_tof.py` (`tof.csv` y crossing, tilt, g4) | stored y | **N** | y crossing +31.3 (A) / +34.5 (C) at ≥ 20 ms | *−31.3 / −34.5*; g4 reco step: **pending** (digitiser fixed) | rerun `capsule_tof all` |
| 13 | g4_digi samples (`g4_{A,C}_is2_full`) | digitiser y strip = HALF − v | **N** | sim mirrored to match the mirrored data | `strip_v_sign = +1`. Must reproduce the Geant4 truth y crossing (−28 mm) through the reco: **pending** | `run_full.sh` |
| 14 | `ntof_calorimetry` (mip_sample, mm_charge, mm_dedx_cosmics) | v, tv for scint crossings | **N** (small) | — | **pending**; mip_sample now refuses mixed vintages | rerun after the scint stack |
| 15 | `det_a_scint`, `scintillators.py`, `make_det_a_scint_report` | `pred_*`, v | **N** (small) | — | `pred_*` recomputed at regeneration; **pending** | rerun after regeneration |
| 16 | ntof_cosmics (cosmic_tracks, angle_response, bd_feasibility, capsule_focus, inbeam_through_goers, insitu_calib, seed tests, yield_gate, pilot_compare) | p0_y/d_y/tany; mostly \|tan\| | **U** mostly | \|tan\|-only quantities unchanged; `capsule_focus` uses p0_y/d_y in pointing (XZ miss: unchanged) | rerun only on next use | none now |

## 4 · Run log (fill from `Y_FIX_REGEN.sh`)

| step | result |
|---|---|
| tests (laptop, 2026-10-13) | y convention: 9 passed, 1 skipped (the run_149 data test); wft + slim_pipeline + sept26: 67 passed |
| code on main | 9ea8b72 (merge of `y-flip-fix`, 361719d) |
| board | log entry + pair-vertex-y erratum, deployed 2026-10-13 |
| y-orientation slide "fixed on, what moved" | **pending**: its generator needs ac149/eff_map/tof.csv (desktop) |
| stage-3 regeneration (cf, is2_v2, fullpass, campaign, pilot, cosmic k_run_147/150) | **pending** — desktop offline |
| guard: late crossing | **pending** |
| guard: cosmic verdict | **pending** |
| g4_digi truth y crossing | **pending** |
