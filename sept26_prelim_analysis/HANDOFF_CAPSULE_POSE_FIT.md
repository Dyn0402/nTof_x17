# HANDOFF — fit the capsule's pose in the beam from beam data

**Written 2026-10-13** at Dylan's request. This is an exploratory study: it may not converge to anything useful.
That is an acceptable result, as long as the report says which parameters the data constrain and which they do not.

Read first:
- `SAME_CHAMBER_PAIRS.md`, section "Follow-up 2026-10-13 (day)".
- The slide note <https://dylan-neff.web.cern.ch/notes/same-chamber-pairs.html>, slides 2–3 and 6–7.
- `HANDOFF_CAPSULE_Y.md` (the earlier capsule-height work and its trap list).
- CLAUDE.md: "Take positions from pointing, not from the shape of a distribution".

> ## ⚠ Correction — 2026-10-14: the beam model was the resolution function. Refit on the real beam.
>
> **The beam.** `Lambda2D` (`lamda2DvsEn_EAR2.root`) is the n_TOF **resolution function**, not a beam radius:
> - its axis is titled "lambda (cm)";
> - it is `transport.cc`'s `HIST_LAMBDA`;
> - it sits beside its SAMMY resolution export.
>
> The real beam comes from `ear2_beam/`: n_TOF's phase-4 FLUKA phase space through the EAR2 collimation.
> - It matches the PPACMon beam-interception factors at 19.95 m to 1–2 %.
> - At 19.5 m: r50 10.5 mm, FWHM 22 mm, sharp edge at r ≈ 22 mm. The old model had r50 2.5 mm.
> - A centred 20 mm bore sees 0.46 of it (0.35 at 19.95 m).
> - The beam sits on the pipe axis to 1–2 mm.
> - The flux is now the Phase-4 one (×1.19 thermal).
> - `capsule_pose.py` now samples `ear2_beam.beam` with the ray slopes.
> - Everything below this box was computed on the old beam, and its two "readings" are void.
>
> **Refit** (`capsule_pose pose2`, `<out>/capsule_pose_ear2/pose2*.csv`; late window; A/C x crossings
> σ_sys 1 mm; x-vs-height shape over the populated bins with the toy's library spread):
> - **Tilt is robust.** The capsule top leans toward −x by ~7–15° (φ 135–225°). The untilted shape is χ² ≈ 46–67
>   for 10 points; tilted it is 7–12. φ = 0 (top toward +x, nose outward, the original hypothesis) is excluded.
> - **Height is robust.** Every good pose needs the capsule 8–12 mm lower than drawn (dy_c).
> - **Sideways it is degenerate.** The x crossing mixes the beam and capsule positions. Best fits:
>   - beam 0 / capsule −16: χ² 23.7, ³He sees 0.23;
>   - beam −5 / capsule −15: 14.8, 0.42;
>   - beam −10 / capsule −10: 12.1, 0.53;
>   - beam −10 / capsule −15: **8.3**, 0.58.
>   The shape mildly prefers the beam off the frame (Δχ² ≈ 15). That shape error is toy-library-limited, and
>   the literature puts the beam on the pipe axis. So if the frame is centred on the pipe, the capsule is about
>   15 mm off the beam axis and the ³He sees ~0.2–0.25 of the beam at 19.5 m.
> - **Not yet done:**
>   - B/D z-vs-height, to pin the tilt azimuth (φ 135 vs 225 = the z component's sign);
>   - the 19.95 m beam;
>   - more Geant4 track library, to tighten the shape χ²;
>   - the late population's illumination. Direct beam at 20–75 ms is only 1.3 % of 4–12 ms in n_TOF's phase
>     space, yet the data's late capsule tracks dominate. The fit assumes they are lit like the beam.
> - **Geant4 rerun with the real gun:** condor 4454431 (capsule at 19.5 m) and 4454432 (19.95 m) →
>   `full_sim/neutrons_thermal_ear2_L19p{5,95}`. Smoke test: ³He 0.448 per neutron (was ~0.95),
>   Al 0.0028 (was 0.0045), air 0.0025.

> ## Outcome — 2026-10-13 (evening), revised the same night. Read this first.
>
> Code `capsule_pose.py` (beam / closure / data / tof / shear / early / image / alternatives / halo / scan / fit),
> report `make_capsule_pose_report.py` → `<out>/capsule_pose/report.html`; deck slide `pose`.
>
> - **Model.** Rays get an energy from the EAR2 Phase-3 flux in the window and a radius from that energy's Lambda2D row
>   (the Geant4 gun's recipe), capture ∝ 1/v. Neutron scattering in the CFRP/Al is needed for closure (without it barrel
>   grazing is 2× Geant4). Al captures by radius then match Geant4 bin by bin; crossings close to 0.4 mm. Tracks are
>   g4_digi tracks translated to the toy capture points.
> - **Measured: one compact source** at x −10.3, z −3.6 mm (chamber frame), at the nose height (−31.3 A / −34.6 C,
>   mirrored). Same x from 2 to 75 ms (pointing-excess peak; the band crossing is unusable below ~10 ms), in every run,
>   before and after the 27 July access. A and C agree although their u axes are opposite, so only a *common* shift of
>   the A–C pair against the beam could fake it; same for B–D in z.
> - **RETRACTED: "the beam is at x −10.3".** The source is where the illuminating flux meets Al. A compact nose source
>   needs a narrow flux centred within ~8 mm of the nose (capsule within ±5 mm of its drawn height). Every Lambda2D-like
>   beam centred on the frame that reaches x ≈ −10 (capsule 13–18 mm off; flux ×10 wider; tail-lit) puts the source
>   14–25 mm too high and leaves 1–15 % of the beam on the ³He. The direct-beam window shows no source near x = 0
>   (2σ: ≤ 34 % A / 10 % C at 4–6 ms, ≤ 5 % at 6–12 ms), but the direct-beam share of those tracks is unknown: the
>   after-the-flux late population images the same spot, which an isotropic hall flux would not (y +5…+8).
> - **Two readings, not separable with these data:** (1) the flux is centred on the capsule ~10 mm toward B / 4 mm
>   toward C from the frame centre, i.e. the beam is not where the frame says or the frame is not where it was aligned
>   (Dylan: centred to mm; survey positions in run_config). Capsule ~4 mm below drawn; ≥ 85 % of the beam on the ³He.
>   (2) The beam is centred as aligned and the capsule is 13–18 mm off it and ≥ 14 mm lower than drawn, lit by a
>   broader-than-Lambda2D beam or its tail; ³He yield 1–15 % of nominal.
> - **Not discriminating:** the height shape (the uncalibrated y scale absorbs it), the scintillators (they share the
>   structure), the 2D pointing image (no beam-line ridge, e.g. ¹⁴N in air, is visible at any x).
> - **x vs height step** ~4 mm, 3.6σ vs a point source (Geant4-library-limited): hint of a second, higher source.
> - **Needed:** how the beam centring was done and against what; the capsule tip height and its holder; the beam spot
>   at the station (collimator, distance, halo beyond Lambda2D's 3 cm). The stage-3 y fix is separate.

---

## 1 · The question

Where is the ³He capsule relative to the **neutron beam**, and relative to the **chambers**? We want:
- its transverse position;
- its height along the beam;
- its tilt, as two angles;
- if the data can separate it, the beam's own transverse centre.

Fit a forward model of capture positions to the measured track pointing.

**Why it matters.**
- Beam neutrons that enter the 500 bar ³He are absorbed within ~0.15 mm, so the gas is effectively black.
- Aluminium captures therefore happen where the beam first meets aluminium: the 5.5 mm nose at the upstream end,
  or, if the capsule is off the beam, the 0.6 mm barrel wall the rays graze along.
- **Rate of ³He captures (so IPC and X17 yield):** this depends on how much of the beam the capsule intercepts.
- **Pairs per trigger:** this depends on how much aluminium surrounds the capture point. The nose is thick and the
  barrel wall is thin.
- Same-chamber e⁺e⁻ data run at ~1/8 of Geant4 (centred, nose-first). A capsule that is half out of the beam, with
  barrel-grazing captures, is a candidate explanation that nobody has tested.

**Dylan's hypothesis.**
- The capsule is ~1 cm off the beam, toward the B–C corner.
- It is tilted with the bottom (nose) furthest out, so much of the nose is out of the beam.

---

## 2 · Step 0: the y sign. Fix it first.

**Hardware (Dylan, 2026-10-13):** the y strips were wired with low FEU channels at the bottom of each chamber and
high channels at the top. +y is the beam direction (upward).

**Confirmed independently (later 10-13):** beam-off cosmic muon timing (p = 4e-11) and plastic-PMT timing, `ntof_cosmics/y_orientation_check.py`.

**Software:**
- `mx17_m1_map.csv` gives `y_position_mm` = 0.78 × channel, so the raw strip coordinate `y_p0` increases upward.
- `DetTransform.R` is a rotation about y only, so local y is global +y.
- `build_tracks.py` applies `IN_PLANE_SIGN_Y = −1` to the position: `yl = −(y_p0 − 199.29)`.
- So **every stage-3 table has global y mirrored.** The 2026-09-07 sign test, corr(y, tan y) > 0, was satisfied by
  flipping positions. The correct fix was to flip the tans.
- This is consistent with x. There, positions were flipped and tans were not, which implies the reco's
  `tan_raw = −d(strip)/d(depth)`. For a y view whose strips run upward, that convention gives
  `tan_y,true = −tan_y,raw` with the position unflipped.

**Corroborating evidence:**
- Data y band crossing: +31 mm (A) and +35 mm (C) as stored. Geant4's nose-first capsule gives −28 mm with the same
  estimator.
- The July hits analysis, which applied no position flip, had the source "3 cm low".
- Liquid B (PMT on top) brightens toward stored −y.

**What to do in this study:**
- Mirror in the loader: `p0_y → −p0_y` and `d_y → −d_y` (equivalently `tany → −tany`, `y_local → −y_local`). Do not
  rerun stage 3 for this study.
- Angles between tracks are unchanged by the mirror.
- Changing `build_tracks.IN_PLANE_SIGN_Y` and regenerating stage 3 is a separate decision for Dylan. Raise it in the
  report.
- Corrected heights: source at **−31 mm (A) and −35 mm (C)**, against Geant4 −28 mm.

---

## 3 · What is already measured (all pointing estimators, post-access runs)

| quantity | value | where |
|---|---|---|
| source x (A, C band crossing, global) | A −8.3, C −10.3 (campaign medians); −10.2 / −10.7 on is2_v2 cf ≥ 12 ms | `<out>/imaging_campaign/per_arm.csv`; `<out>/capsule_tof/tilt.csv` |
| source z (B, D) | B −4.1 (few tracks), D −3.5 | `per_arm.csv` (D needs the hot-channel cleaning; production D through `capsule_tof.tilt` gives garbage, −65) |
| source height, y band crossing | stored +31.3 (A) / +34.5 (C) at ≥ 20 ms, i.e. **−31 / −35 corrected** | `<out>/capsule_tof/tof.csv` (`y_cross`) |
| same, early (beam) window | stored A +36…+46 at 7–12 ms, C +42…+54, i.e. lower than late once corrected | `tof.csv` |
| x vs reconstructed height (shear) | x −14 mm at stored y −30 → −9.3 at stored +50, in A and C and in both chains; Geant4 (untilted) flat within ±1 mm | `tilt.csv`; slide 7 |
| capsule-pointing purity vs ToF | 6–25 % at 2–10 ms, 46 % (A) / 42 % (C) after 20 ms; Geant4 digitised 31 % / 34 % | `tof.csv` (`excess`), `g4.csv` |
| capsule-pointing tracks per ms | rise to a maximum at 15–25 ms, half of them after 30 ms | `tof.csv` (`focus_per_ms`) |
| beam captures (flux × 1/v) | 87 % before 12 ms, peak 5–6 ms | `<out>/capsule_tof/beam_capture.csv` |

**Two things to note before modelling:**
- Under the corrected y, the shear means the source is further toward −x (toward B) at the TOP than at the bottom.
  That is the opposite sense to Dylan's hypothesis, if it is a capsule tilt. It may instead be an in-plane rotation
  common to A and C, which A–C cosmics cannot see. Check it on D/B with z against height.
- **Most capsule-pointing tracks are NOT beam captures.**
  - 92 % are after 12 ms and half are after 30 ms. That is a delayed, thermalised population with τ ≈ 35–40 ms
    (`ntof_cosmics/HANDOFF_TRACKING_2026-10-06.md` §10, the late-trigger clock).
  - Its illumination of the capsule is unknown: isotropic, or streaming along the beam pipe.
  - **The beam's footprint is in the 4–12 ms tracks**: ~30–60 k capsule-pointing tracks per arm, at low purity.
  - Fit the two windows separately, or with two source components.

---

## 4 · Beam profile: narrow core, long tail

From `MX17_Full_Geant/data/lamda2DvsEn_EAR2.root` (`Lambda2D`, radial, normalised per energy row), at 5–200 meV:
**50 % of the beam within r = 2.5 mm, 68 % within 4.5 mm, 90 % within 13 mm, 99 % within 27 mm.**

If the beam is centred on the chamber frame and the capsule axis is offset by d:

| d (mm) | beam on the capsule (r < 10.6) | on the nose core (r < 6) | grazing the barrel wall (10 < r < 10.6) |
|---|---|---|---|
| 0 | 0.86 | 0.75 | 0.011 |
| 5 | 0.84 | 0.54 | 0.015 |
| 8 | 0.76 | 0.11 | 0.043 |
| 10 | 0.52 | 0.05 | 0.111 |
| 12 | 0.19 | 0.03 | 0.056 |

A grazing ray can run ~40 mm along the wall. At Σ_capture(Al, 25 meV) ≈ 0.014 /cm that is ~5 % capture per ray,
against ~0.8 % through the 5.5 mm nose. So at d ≈ 10 mm, **barrel captures can dominate** and pull the source
height up the barrel. This is the main lever the height measurement gives.

**The imaging measures captures relative to the chambers, not relative to the beam.** A 10 mm offset of the
capture centroid from the chamber-frame centre is ambiguous between two cases:
1. Capsule and beam are offset together. Then the nose is hit and the centroid sits at the nose.
2. The beam is centred on the chambers and the capsule is off it. Then the centroid is pulled toward the beam side
   of the capsule and up the barrel.

The height and its spread are what can separate them.

---

## 5 · Suggested method

1. **Geometry.**
   - Al, CFRP and gas polycones: `MX17_Full_Geant/src/DetectorConstruction.cc` (nose-first, tip at local −35 mm) or
     `ntof_tracking/reco/geometry.py` `HE3_GAS_Y/R`. The Al outer profile is only in the C++.
   - Pose parameters: capsule centre (x_c, z_c, y_c) in the chamber frame; tilt (θ, φ); beam centre (x_b, z_b).
     The beam is along +y with a Lambda2D radial profile.
2. **Captures, by ray-tracing.**
   - Sample beam rays from the profile at a 1/v-weighted energy.
   - Walk each ray through the posed capsule: Al capture probability per step Σ(E)·ds, with the gas as a black
     absorber (stop the ray at gas entry) and CFRP's own capture (small; in Geant4 it is ~5 % of Al).
   - Record capture points, weighted.
   - For the delayed component, try (a) isotropic thermal flux (captures ∝ Al volume near the surface) and
     (b) the same beam profile at meV energies.
3. **Observables, through the measured detector.**
   - Emit isotropic γ, or simply take the capture points as track sources. Pair and Compton kinematics hardly move a
     centroid; Geant4 showed truth ≈ reco for the crossings.
   - Project each source to arms A–D with |tan| < 0.6.
   - Smear tan by the measured resolution (`ntof_cosmics/results/tracking/pooled/resolution.csv`, σ ≈ 0.028 core
     plus tails).
   - Apply exactly the data's estimators:
     - x and z band crossings, overall and per height bin;
     - y band crossing;
     - the reconstructed-height distribution, to use its width; its shape is a nuisance, so prefer pointing summaries.
4. **Closure first.** Run the toy at Geant4's nominal pose (centred, untilted, beam centred). Compare it with the
   same estimators on `g4_digi` (`/media/dylan/data/x17/ntof_cosmics/g4_digi/g4_{A,C}_is2_full`; y crossing
   −28 mm, x flat with height). If the toy does not close, fix it before fitting data.
5. **Fit.**
   - Use a χ² over the pointing summaries, per ToF window (5–12 ms beam-dominated; ≥ 20 ms delayed).
   - Scan the parameters and report the profiles. Expect degeneracies: (capsule offset) vs (beam offset), and tilt
     vs a common chamber rotation.
   - Use B/D to break x-only degeneracies where they have statistics.
6. **Consequences, if the fit says something.**
   - Fraction of the beam on the capsule (rate of ³He captures, hence IPC/X17 yield).
   - Fraction of Al captures in the nose vs the barrel. Then the expected pairs per trigger: either reweight the
     Geant4 truth by capture region (the conversion origins `ov`/`oy` in the `same_arm_pairs` parts give the region
     split), or propose a Geant4 run with the fitted pose. That needs a pose option in `DetectorConstruction`; the
     `lnl` branch is checked out, so make a branch, and submit via lxplus condor per CLAUDE.md storage rules.

---

## 6 · Data access and cuts (copy from `sept26_prelim_analysis/capsule_tof.py`)

- **Tracks:**
  - is2_v2 at the capsule-focus k, A/C only: `~/x17_local/stage3_is2_v2_cf/tracks_campaign.parquet`. It lives on the
    root disk because the data disk was full.
  - Production, all arms: `/media/dylan/data/x17/sept26_prelim/stage3_fullpass/tracks_campaign.parquet`.
    Use it for B/D; the crossings are scale-free, so its capsule-k is harmless.
- **Cuts:**
  - exclude run_79 and run_81 (pre-access);
  - `gated_reco` (or `gated`) and `angle_calibrated`;
  - |tanx|, |tany| < 0.6; x_t0, y_t0 ≤ 300;
  - time windows from `t_since_flash_ns`.
- **Helpers:** `capsule_tof.load()` gives per-track `pos`, `disp`, `hgt` in global coordinates, and
  `capsule_tof.crossing()` is the robust zero crossing.
  - **They do not apply the y mirror yet.** Add it to `load()` (one line on `p0_y`, `d_y`) and rerun `tof`/`tilt`.
- **D:** use `campaign_imaging`'s cleaned sample (hot-channel masks; `HANDOFF_D_NOISY_CHANNELS.md`), or the
  crossing is junk.
- **The y angle scale is not calibrated** (HANDOFF_CAPSULE_Y §6.2). Heights *binned* by reconstructed y carry it.
  The y band crossing does not.

## 7 · Deliverables

- `sept26_prelim_analysis/capsule_pose.py`: toy, closure, fit.
- `make_capsule_pose_report.py` → `<out>/capsule_pose/report.html`, following CLAUDE.md's report rules (verdict,
  numbers, what is not ruled out).
- One slide or two added to `make_same_chamber_deck.py`, or a new note if it grows.
- Update `SAME_CHAMBER_PAIRS.md` and this file's Outcome box.
- Commit, merge to main and push (CLAUDE.md git rules).

## 8 · Open items this handoff does not cover

- The y-sign fix in `build_tracks` and a stage-3 regeneration. Dylan decides. Many downstream heights move:
  `target_y_mm`, `CAPSULE_V` in `same_chamber_physics.py`, `CAPSULE_SHIFT_MM = 32.2` in `same_chamber_compare.py`
  (likely the mirror image of the true offset), and the scint-stack v-maps (liquid B/C PMT puzzle).
- The 1 : 8 pair rate. This study feeds it but does not close it.
