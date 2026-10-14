# MM energy deposit per track — how accurately, and does it work on n_TOF data

Started 2026-10-14 (dylan-MS-7C84). Replaces the "2-MIP flag" framing of PLAN.md M2–M3:
the question here is **how well the MM measures the energy a reconstructed track
deposits in its 30 mm gap**, and whether that adds to the scintillators' picture
of the track. Status: stages 1–4 have run on A cosmics and A beam; C beam sim and
the report are still to do (HANDOFF.md).

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

## Known issues

- Beam sim was generated at 10.7 ADC/e but `chamber()` applies the **cosmic** k and window curve, so
  `sim_beam.bias` = −0.46 is the gain ratio by design, and the beam fold `scale` 1.2 is relative to that.
  Use a beam-sim k (`window_curve(Sb)`) for absolute keV on beam.
- Sim reco t0 is +47 ns (spread 82) later than the input t0 → sim tracks are more truncated than data;
  compare in f_obs bins / contained tracks only.
- Fold fits are on the core (ln within −0.7…+0.45 of the median); tails are reported separately.
