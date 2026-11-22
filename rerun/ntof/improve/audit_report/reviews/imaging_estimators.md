<!-- Verbatim report of the 'Review capsule imaging estimators' code review, part of the 2026-11-21 n_TOF tracking audit (session be7f0e66).
     Scratch scripts it cites under /home/dylan/.claude/jobs/be7f0e66/tmp/ were copied to
     $WORK_DATA/ntof_pass1/ana/tracking_audit/review_scripts/. -->

# Read-only review of the capsule-imaging estimators

No estimator has a sign or frame bug. Four of the listed oddities are produced or inflated by how an estimator is built; the x ≈ −10 mm offset is not one of them.

Real-data checks used 8 runs (139–154) of the analysis view, and runs 143/145/147 for the pointing-coincident sample. Scratch scripts and outputs are in `/home/dylan/.claude/jobs/be7f0e66/tmp/review_imaging/`. Nothing was written to the repo or under `/media/ucla/work/ntof_pass1`.

## Findings, ranked

### 1. `capsule_tof.py:199` — the `tilt` z crossing for D is not a measurement (CONFIRMED)

- **What is wrong.** `tilt` cuts every arm at ≥ 12 ms. For D, 59 % of that sample is 12–25 ms tracks that do not point (band slope −0.02, crossing +177 mm).
- **Why the fit fails.** `crossing` (lines 132–136) clips at 3× the MAD of the residuals. On D the MAD is 52 mm, so nothing is clipped (kept 1.00) and the fit is a plain least-squares line with slope −0.15 instead of −1.
- **Consequence.** The campaign value reads −1.71 (no map), +7.48 (map), −2.70 (map + censored cut). My 8-run subset gives −13.4. The quoted ±0.3 is statistical only.
- **Where it propagates.** `compare_chains.py:68-71` copies it into `chains.csv`, and `PASS1_DOWNSTREAM.md` quotes "D z −1.7 ± 0.3".
- **The repo's own D selection works.** With ≥ 25 ms and wall (`capsule_pose.py:1007`), four estimators give −3.97 (CT), −3.75 (30–130 window), −4.08 (recentred window), −4.84 (CT within 100 mm of the crossing).

### 2. `capsule_tof.py:123-143` — `crossing` fails abruptly with background (CONFIRMED by injection)

I added real < 4 ms tracks to the real late samples and refitted.

| arm | +5 % | +10 % | +20 % | +40 % |
|---|---|---|---|---|
| A (from −10.73) | −10.75 | −10.78 | −11.65 | −18.2 |
| C (from −10.85) | −10.91 | −11.00 | −11.95 | −14.3 |
| D late + wall (from −3.97) | −4.45 | −10.6 | −13.8 | −17.9 |

- A and C late samples are safe. D sits at the edge.
- The same mechanism makes the 4–12 ms window crossings and the outer height bins unusable (h = ±100, +60, and +40 on D have slope −0.3 to −0.6).
- Fitting only within 100 mm of the crossing is far more robust (A −10.71 → −10.95 at +40 %).
- For a pure source with the |tan| < 0.5 cut and the real dead regions, the toy bias is ≤ 0.05 mm on A and C, and +0.5 mm on D.

### 3. `source_imaging.py:319`, `k_arm.py:148`, `run145_target_imaging.py:258-260` — `campaign_imaging` has three built-in biases (CONFIRMED)

**(a) Asymmetric truncation of the band.**
- The fit window is 30–130 mm about the surveyed foot, not about the source.
- The coincidence also requires the predicted plastic position within ±201.7 mm of the chamber centre. The band reaches that edge at lever +105 to +112 mm on one side and −122 to −131 mm on the other.
- In the data, A has 550 tracks at lever +110–130 against 3,212 at −110–130, with a median tan residual of −0.35.
- Toy (source at x = −10, z = −4): A +0.47, C −0.41, D −0.43 mm. The window part alone is +0.33 mm per 10 mm of offset (raw cut 0.6), and zero for an on-axis source.
- Data, 30–90 window against 30–130: A −8.32 against −7.81, C −9.27 against −10.10. A − C falls from +2.29 to +0.95 mm.
- So about half of the "A–C half-difference 1.16 mm" is this artefact, and |x| is understated by roughly 0.3–0.9 mm.
- The reduction keeps `variant='baseline'` (`campaign_imaging.py:169`); the recentred variants are already in `crossings.csv`.

**(b) No ToF cut** (`k_arm.py:152-238`).
- 5–13 % of the sample is < 12 ms.
- D run_145: −6.10 with all ToF, −5.31 at ≥ 20 ms. D on three runs: −5.39 → −4.97 (≥ 12 ms) → −4.62 (≥ 20 ms).
- C moves 0.18 mm; A does not move.
- With a ToF cut, the result no longer depends on the charge window.

**(c) It does not read the analysis view.**
- `per_arm.csv` is byte-identical in `ana/` and `ana_nomap/`. It uses the raw `x_tan_theta`, with no `gated`, no map and no censored-window cut.
- On the same late coincident rows the map tan moves A by −1.95 and C by −1.33 mm.
- So "−9.09 (imaging)" and "−10.9 (tilt, pose)" are not the same quantity.

**(d) Hot-strip drop on one run only** (`k_arm.py:213-223`).
- `hot_strata/` holds run_145 only; the log has 140 "cut NOT applied" lines.
- Effect on run_145 D is 0.14 mm, so this is minor.

### 4. `focus_split.py:79`, `capsule_focus.py:63,71` — the tan-sign split is the nominal axis (CONFIRMED on data and toy)

- The miss is taken about x = 0, while the source is at −10.8 mm.
- Re-centred on the measured source, the splits collapse:
  - A: 0.860 / 1.141 → 1.002 / 0.995
  - C: 1.140 / 0.853 → 0.975 / 1.006
- A toy point source at −10 mm gives 0.87 / 1.125 with nothing else in it.
- The pooled value is also biased low. Both arms have more tracks on the +x side of the source, and that sign focuses below 1.
  - A all: 0.984 → 1.000. C all: 0.974 → 0.988.
  - Common scale: 0.979 → 0.994 (map view), 0.992 → 1.013 (no map).
- So "(A + C)/2 = 0.977" in `compare_chains.py:53` is about 0.015–0.02 low.
- The no-map A − C asymmetry survives re-centring (0.046), so that reading holds.

### 5. `capsule_tof.py:55,146-153` — "45–58 % point within 15 mm" is not a purity (CONFIRMED on Geant4)

- True capsule captures in the Geant4 library, through the pass-1 reconstruction, point within 15 mm only 41–48 % of the time. Their excess over the shuffled null is 0.28–0.35 and their median miss is 16–20 mm.
- The late data give f = 0.57–0.60 and excess 0.43–0.46, which is better than a pure simulated capsule sample.
- R_FOCUS = 15 mm is about one core width, so the number is resolution and scattering, not background.

### 6. Which D z is less biased, and why they differ (answer to 2c)

- The tilt value (−1.7) is finding 1 and should be dropped.
- −5.55 is a real band crossing, but it carries −0.4 to −0.8 mm from early tracks and about −0.4 mm from the truncation in 3(a).
- Corrected, it is about −4.3 to −4.8 mm. The late wall-confirmed view sample gives −3.7 to −4.8 mm.
- I would quote D z ≈ −4.5 ± 1 mm at all heights.
- What remains is not explained:
  - The 30–70 window gives −7.8 against −5.35 for 30–130, so D's band is not straight.
  - D's source foot (x_local ≈ 20) lies in its dead region (x_local 5–65).
  - In run_145, late wall tracks inside the coincident sample give −5.2 ± 0.4 and those outside give −2.2 ± 0.4.
- z also changes by about 5 mm with height, so any all-height number depends on the height weighting.

### 7. `capsule_tof.py:59`, `capsule_pose.py:101,501,983` — the toy's fixed A − C offset is an artefact (CONFIRMED for the constant; the library-size share of the offset is an inference)

- `FOOT_SIM['C'] = 16.4`, but the Geant4 geometry has 17.3 (`SimConfig.hh:286`, `run_digi.py:33`).
- The sim's own truth crossing for C is 17.11 ± 0.34 (nose) and 17.45 ± 0.65 (ear2).
- This makes the toy's x_C 0.9 mm too negative.
- The rest of the "+2.3–2.7 mm for all 287 poses" is plausibly library size. The ear2 library has 2,690 (A) and 2,186 (C) capsule tracks, so each crossing carries ±0.6–0.9 mm. The same tracks are reused in every pose, which is why the spread over poses is only 0.06 mm.
- `SIG_X_SYS = 1.0` does not cover this.
- It sets a floor of about 2.8 on the crossings term. That is small against the shape term (22–49), so it does not drive the pose ranking. It does account for the reported rise of the crossings term with the map.

### 8. `track_quality.py:132-138` — the null keeps part of the pointing (CONFIRMED)

- Shuffling displacements inside |tan| bins mostly randomises only the sign.
- Null within 15 mm: 0.296 against 0.140 for a full shuffle. Excess: 0.305 against 0.461.
- `exc1` and `exc2` in `scan_*.csv` are therefore about one third low and are not comparable with `capsule_tof`'s excess.

### 9. Smaller items

- **`track_quality.py:54-58`, stale cache.** `tracks_AC.parquet` and `scan_late.csv` are dated 11-20 23:12. The view was rebuilt with map and cut on 11-21 01:23. `load_tracks()` returns the cache whatever the view, and `angle_map.py beam` reads it.
- **`capsule_pose.py:689,953`, stale constant.** `Z_BD` is from the is2 era (B −4.12, D −3.45). Their unweighted mean, −3.79, is the capsule z in every pose. I did not measure the effect.
- **`capsule_y.py:891-910`, y crossing (PLAUSIBLE).** A toy with the trigger plastic clip (|v + 189·tan| < 150) gives −0.5 mm for a point-like nose and −1.3 mm for nose plus barrel. Without the clip it is unbiased (≤ 0.4 mm). The data are stable to ±0.5 mm under selections: A −30.4, C −29.4, D −32.1.
- **`capsule_pose.py:1140-1159`, depth from band slopes (PLAUSIBLE, not tested).** The slope is diluted by background (−0.93 in data), differently per arm and height bin; I did not measure the size of the effect.
- **`capsule_focus.py:75-79`.** The null bins the permuted tan by the original track's |tan| bin. Display only.
- **`capsule_point.py:29`.** The axis (−9.5, −3.7) is from the is2 era. The fraction within 15 mm changes by < 0.001.

## Selection consistency (item 3)

| estimator | sample | ToF | tan | wall |
|---|---|---|---|---|
| `campaign_imaging` | reco tree, pointing-coincident, charge 25–75 % | none | raw, cut ≈ 0.6, no `gated` | segment + bar matched |
| `capsule_tof` tof / tilt | view `gated`, t0 ≤ 300 | ≥ 12 ms | view, < 0.5 | none |
| `capsule_pose` data (A, C) | as `capsule_tof` | 20–75 ms | view | none |
| `capsule_pose` data_z (D) | as `capsule_tof` | 25–75 ms | view | wall flag |
| `capsule_focus`, `focus_split` | view `gated` | all, about 45 % early | view, < 0.5 | none; nominal axis |
| `track_quality` | stale no-map cache | ≥ 20 ms | \|tan x\| ≥ 0.1 | none |

The t0 ≤ 300 cut differs between scripts but removes < 0.1 % of tracks.

## Checked and found correct

- **Frames and signs.** pos = x_local − 16.35 (A), 17.3 − x_local (C), 15.5 − x_local (D). `U_HAT`, `W_HAT` and `to_global` agree with this.
- **Lever arm.** p0 is on the strip plane for every track: 234.6 mm (A, C) and 234.1 mm (D).
- **Scale and depth independence.** The band crossing does not depend on the tan scale or on the source depth (algebra, and toy with z = −4).
- **Source x on the map view.** Four estimators agree: A −10.4 to −10.7, C −10.5 to −11.4. The offset is not an estimator artefact. The only bias toward the axis is 3(a), so |x| is if anything larger than −9.09.
- **Shear with height.** In the bins the pose fit uses (h = −60 to +20), x (−9.5 → −14.3, A and C) and z (−6.5 → −1.7, D) agree across three estimators to 0.3 mm (A, C) and 1 mm (D). The shear is real.
- **Geometric reading of the focus asymmetry.** The toy gives A 1.013 and C 0.976 for z = −4, which is the distance ratio. So "4.6 % ↔ 5.4 mm" is right.
- **`focus_excess` null.** The full shuffle is a valid no-pointing null. The ToF binning has no off-by-one.
- **Dead-channel masking.** `raw = IN_PLANE_SIGN·xl + HALF` is the correct inverse.
- **Pose χ².** It is dominated by the shape term, and the shape comparison removes each pose's own x and y offsets as documented.
- **Wall flags.** They are plain booleans, so `focus_split.py:68` is safe.

One thing that is not mine: an untracked `rerun/ntof/improve/tracking_audit.py` and `/media/ucla/work/ntof_pass1/ana/tracking_audit/` appeared during the run, from another session.
