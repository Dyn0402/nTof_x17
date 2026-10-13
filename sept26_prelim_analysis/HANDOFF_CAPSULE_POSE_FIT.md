# HANDOFF — fit the capsule's pose in the beam from beam data

**Written 2026-10-13** at Dylan's request. This is an exploratory study: it may not converge to anything useful.
That is an acceptable result, as long as the report says which parameters the data constrain and which they do not.

Read first:
- `SAME_CHAMBER_PAIRS.md`, section "Follow-up 2026-10-13 (day)".
- The slide note <https://dylan-neff.web.cern.ch/notes/same-chamber-pairs.html>, slides 2–3 and 6–7.
- `HANDOFF_CAPSULE_Y.md` (the earlier capsule-height work and its trap list).
- CLAUDE.md: "Take positions from pointing, not from the shape of a distribution".

> ## Outcome — run 2026-10-13 (evening). Read this first.
>
> Code `capsule_pose.py` (beam / closure / data / tof / shear / scan / fit), report
> `make_capsule_pose_report.py` → `<out>/capsule_pose/report.html`; one slide (`pose`) in
> `make_same_chamber_deck.py`.
>
> - **Toy closes on Geant4** to 0.4 mm in x and y (Geant4's own error 0.37), but only after adding neutron
>   *scattering* along each ray (CFRP 1.5 /cm, tuned on the Geant4 radius distribution of Al captures; Al
>   1.4 b). Without it barrel-grazing captures were 2× Geant4's. Tracks are Geant4 g4_digi tracks translated
>   from their capture point to the toy's, so kinematics and reco are Geant4's.
> - **The beam meets the capsule on its nose.** Late (≥ 20 ms) y crossings, mirrored: A −31.3, C −34.6 mm;
>   flat from 20 to 75 ms. The toy at the Geant4 pose gives −28.6 / −29.4, so a beam on the axis needs the
>   capsule ~4 mm lower than Geant4 (A−C 3.3 mm still open). Isotropic illumination would give +5…+8.
> - **The x crossing measures the beam, not the capsule axis**: beam at x −10.3, z −3.6 mm (chamber frame).
>   It moves < 1 mm for any offset up to 10 mm.
> - **The beam–capsule offset is NOT determined.** The y crossing is degenerate with the (unmeasured) capsule
>   height, and a tilt with the nose toward the beam keeps the nose in the beam at any offset. Untilted, d = 10
>   (Dylan's hypothesis, nose out) needs the capsule ~25 mm lower than drawn. **What is bounded are the
>   consequences:** with the capsule within ±5 mm of the Geant4 placement, ≥ 87 % of the beam enters the ³He,
>   ≤ 4 % of Al captures are on the barrel, and the nose tip is ≤ 8 mm from the beam axis. At ±10 mm anything goes.
> - **The pose does not explain the 1 : 8.** Untilted d = 10 keeps Al captures per neutron at 0.95× nominal
>   (barrel grazing replaces the nose) while the ³He share falls to 0.45. A ×8 cut needs the capsule mostly off the beam.
> - **The x-vs-height "shear" is a step, not a slope**: ~4 mm toward −x for tracks pointing above the nose.
>   Against a point-like nose source through Geant4 that is a 3.6σ hint (combined A, C) of a second, higher
>   source at more negative x. It is limited by the Geant4 library (~10 k tracks/arm). It is not a capsule tilt
>   (the ±25 mm height spread is kinematics) and not an in-plane rotation (that would be linear).
> - **Earlier windows read lower** (bg-subtracted 4–6 ms: −39 A / −45 C). No pose moves the beam window *down*
>   relative to the late one, so this is an unsubtracted beam-line source below the capsule, not pose.
> - **Not done:** B/D modelling (z from campaign crossings, uncorrected); a Geant4 run with a fitted pose (no pose
>   to fit). **Needed from Dylan:** the capsule tip height against the chamber mid-plane (it turns the scan into a
>   measurement of d), the stage-3 y fix, optionally a 10× g4_digi sample for the step.

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
