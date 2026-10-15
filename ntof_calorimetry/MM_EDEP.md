# MM energy deposit per track — how accurately, and does it work on n_TOF data

Started 2026-10-14 (dylan-MS-7C84). Replaces the "2-MIP flag" framing of PLAN.md M2–M3:
the question here is **how well the MM measures the energy a reconstructed track
deposits in its 30 mm gap**, and whether that adds to the scintillators' picture
of the track. Status (2026-10-15): stages 1–4 have run on A and C, cosmics and beam. The beam calibration is fixed,
and stage 4 was rerun with the measured σ. The per-track combination with every other layer is the
**track passport** (`track_passport.py`, below).

## Chain

| stage | module | what |
|---|---|---|
| 1 truth | `heed_edep.py` (lxplus, Garfield/Heed) | ionisation in 30 mm Ar/iC4H10 90/10, 730 Torr, per track: e⁻ 0.1–16 MeV, μ 3 GeV; fixed θ 0/30/45 and continuous-angle pools. → `<calo>/heed/*.npz` (EOS copy `/eos/user/d/dneff/calorimetry/heed`) |
| 2 sim | `mm_edep_sim.py` | Heed electrons through `ntof_cosmics/g4_digi` (diffusion, Polya, is2 template+kernel, real quiet run_145 noise, production reco), geometry/t0/v copied from real tracks; stores truth at each stage. → `<calo>/mm_edep/sim_{cosmic,beam}_{A,C}.parquet` (+`_prof.pkl`) |
| 3 data | `mm_edep_data.py` | `mm_charge` estimator on run_145 waveforms, stage-3 fullpass gated single tracks A/C, joined to wall/plastic. → `<calo>/mm_edep/beam_run_145.pkl`. Cosmics: existing `m2/charge_run149.pkl` |
| 3 ana | `mm_edep_ana.py` | window-corrected estimator E_hat = Q/(k c(f)), accuracy, forward fold, gain maps, halves test, flash curve; `main()` → `<calo>/mm_edep/results/` |
| 4 combo | `mm_edep_combo.py` | Geant4 neutron-beam events (wp_g4/parts ⨝ g4_angle reduced, same files) + measured resolutions: MM charge per scintillator category; kNN energy estimate with/without MM; C2 singles for the X17 range |

`<calo>` = `/media/dylan/data/x17/calorimetry`.

## Results so far

**Physics (Heed truth).** Gap MPV ≈ 5–6 keV (170–210 electrons; W = 27.5 eV);
the old "~3 keV" was wrong. FWHM/MPV ≈ 0.6. MPV rises ~24 % from 1.5 to 16 MeV
(relativistic rise; 3 GeV μ ≈ 16 MeV e⁻), +14 % median at 0.5 MeV, ×1.6 at 0.2 MeV.
Geant4 opt4 (95/5) reads ~19 % higher MPV than Heed and has a fatter tail.

**Accuracy in simulation (E_hat vs true ionisation, robust σ of ln).**
A, drift inside the window: **10 %** (ionisation 2.6 %, avalanche 4.7 %, containment 2.4 %,
noise+CM+reco road 6.1 %). A with the drift cut by the window (f_obs ≲ 1.1): ~40 %.
C (drift almost never fits, v 26–28 µm/ns): **16 %** window-corrected, 17 % plateau estimator.
Sim noise is pessimistic for cosmics: data x/y spread 0.096 vs sim 0.132, noise road 2.5 % vs 3.9 %
(sim borrows run_145 beam-time triggers).

**What the real chamber adds (data).**
- A position-dependent gain, **one-dimensional along x and in both views**: same x, |Δy|>80 mm
  r = 0.26 in Qx and Qy; same y, far x: 0.01. Correlation length ~20–30 mm; raw r(<4 mm) 0.42 (A),
  0.33 (C). An amplification structure along x, not strip electronics (would move one view).
  The bundles' channel gains are all exactly 1.0, so the sim has none of this.
- A cross-validated local (kNN, 25 tracks) map removes it (r → 0.0–0.07).
- **Per-track common gain jitter remains**: halves test (first vs second half of the drift,
  truth-free) cov excess over sim → σ_g = 0.42 raw, **0.27 after the map, 0.22 in the centre** (A cosmics).
  Not time (lag corr ≤ 0.02), not angle/t0 (≤ 5 %). Fold fit agrees: σ_x 0.19–0.28 (A), 0.16–0.19 (C).
- Realistic cosmic per-track accuracy ≈ √(0.10² + 0.2²) ≈ **22–30 % (A)**, ≈ 25 % (C).
- **Beam (run_145, ≥ 12 ms) is better**: halves σ_g 0.18 raw, **0.09 after a beam self-map**; fold σ_x 0.10–0.14.
  → beam per-track accuracy ≈ √(0.10² + 0.09²) ≈ **13–17 % (A)** (sim-beam calibration caveat below).
- **The cosmic map does not transfer to beam** (makes it worse: 0.367 → 0.393; maps correlate 0.30 A / 0.46 C).
  Calibrate in situ on beam electrons.
- **Flash recovery**: gain ×0.3 for < 8 ms, recovering 8–12 ms, flat ±3 % from 12 to 80 ms; before ~9 ms
  the x/y ratio collapses (0.2–0.7 vs 0.95), so the MM charge is not a deposit measurement there.
- Beam/cosmic gain 0.63 (A), 0.70 (C).

**Combination with the scintillators.**
- Data (A, late, in-window, pointing at the plastic): no counter fired → MM charge **+29 %** (n=109),
  wall only +10 % (n=71), relative to plastic-reaching tracks. Within plastic tracks MM rises +5 %
  from 3 to >8 MeV plastic, and +8 % for wall > 1 MeV.
- Geant4 predicts +15 % (all particles in the gap) / +2 % (leading lepton only, Heed) for "none",
  −3 % for wall-only: data excess ~2× the prediction. Not yet explained (second particles?).
- χ² is NOT an independent quality cut: q rises 0.73 → 1.69 over χ² quintiles even in plastic tracks.
- Energy (Geant4, placeholder σ_MM = 0.15): MM adds nothing in the X17 range (4–16 MeV);
  it narrows vertex-T for 1–3.5 MeV electrons that never reach the plastic (68 % width 1.25 → 0.75 MeV,
  coarse kNN); slow-electron tag (T<1 MeV) among no-plastic tracks AUC wall 0.50 / MM 0.59 / both 0.63.
  **Rerun with the measured σ (≈0.15 beam, 0.25 cosmic) before quoting.**

## Chamber C and the beam calibration (2026-10-15)

- `mm_edep_ana.main()` A+C ran (`results/summary.json`). C: sim accuracy 16 % (cosmic, window-corrected),
  cosmic fold σ_x 0.16–0.19 after the map; the halves test is undefined on C (almost no contained drifts).
- **Beam k fixed** in `track_passport.mm_beam_calibration`: window curve and k from the *beam* sim,
  data/sim scale from the fold after the beam self map. Beam sim bias −0.46 → **0.00**.
  | | k_beam (ADC/keV) | sim σ | data/sim | σ_x (extra) | σ total | MIP (keV, normal) |
  |---|---|---|---|---|---|---|
  | A (contained) | 2471 | 0.096 | 1.226 ± 0.017 | 0.13 ± 0.03 | **0.16** | 5.68 |
  | C (all) | 1233 | 0.224 | 1.163 ± 0.014 | 0.01 ± 0.04 | **0.22** | 5.65 |
  Data median at normal incidence: 5.6 keV per 30 mm, matching Heed's MIP.
  (`chamber()` itself still applies the cosmic k to beam; its `sim_beam.bias` −0.46 is that gain ratio.)
- **Stage 4 rerun at σ = 0.16** (`results/combo_sigma016.txt`). The energy conclusions are unchanged:
  the MM adds nothing at 4–16 MeV, and narrows the 1–3.5 MeV capsule singles (C2) from 1.7 to 0.45–0.8 MeV.
  The slow-electron tag (T < 1 MeV, no plastic) now reads AUC wall 0.74 / MM 0.66 / both 0.80. The
  2026-10-14 record (0.50 / 0.59 / 0.63) was not reproduced; the cause was not traced.

## Track passport (2026-10-15)

`track_passport.py` → `<calo>/passport/`, report `make_passport_report.py` → `passport/report.html`.
On run_145 at ≥ 12 ms it uses A+C self-triggered single tracks: 15,444 tracks, 12,818 with an MM deposit.
For each track it collects the direction, position and pointing, the MM keV, the wall, plastic and liquid
MeV (on the Geant4 scale), and E_n from the time of flight. Geant4-trained kNN estimators add T at the gap,
P(several particles in the gap) and P(conversion). An example card for each class is in the report.
- **Path length (cosmics).** The MM deposit goes as sec^0.97 ± 0.10 (A) and 1.03 ± 0.05 (C); Heed
  predicts 1.07. The wall goes as sec^1.00 ± 0.19 (A) and 1.29 ± 0.14 (C). On beam electrons the wall barely
  follows the gap angle (data 0.5, Geant4 0.1): the electrons scatter before the wall.
- **Several particles in the gap** (Geant4: 30 % of triggered tracks). The MM tags them (AUC 0.84); the wall
  cannot (0.46–0.48). The MM–wall Spearman correlation is ~0 in both data and Geant4, and the "both high"
  lift agrees: data 1.22/1.19, Geant4 1.28/1.40.
- **Energy at the gap (Geant4, triggered).** w68 0.86 MeV from the plastic alone. Adding the track angle
  takes sec ≥ 1.1 from 1.40 to 1.07 MeV and multi-particle tracks from 1.70 to 1.08 MeV; adding the MM takes
  T < 3 MeV from 1.56 to 1.35 MeV.
- **Origin.** Capsule vs elsewhere from the track geometry: AUC 0.75 (all gap tracks), 0.62 (triggered,
  where 91 % are from the capsule). Conversion vs Compton: 0.65.
- Data vs Geant4: the share of tracks below 0.8 MIP × sec in the plastic is 0.08 / 0.13 (A) and 0.28 / 0.33
  (C). The estimator outputs (T̂, P(multi)) match Geant4's distributions. The C liquid almost never fires.

## Known issues

- Sim reco t0 is +47 ns (spread 82) later than the input t0 → sim tracks are more truncated than data;
  compare in f_obs bins / contained tracks only.
- Fold fits are on the core (ln within −0.7…+0.45 of the median); tails are reported separately.
