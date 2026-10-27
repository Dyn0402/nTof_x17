# ws_sps — F20 / F14 / F18(σ0): what the SPS det4 data can deliver, 2026-10-27

(Written by the coordinating session from the workstream's final message. Commits 0f3aedf, 7505338, d20ef98.)

## Cache bug (invalidates F16/F20 as measured)
EOS-built ZS caches (`/eos/.../model_audit/sps/*.pkl`, `sps_build_cache.py`, `sps_reco.py`) never subtracted the ZS
baseline: kept samples sit 256 ADC above the suppressed zeros (nonzero p5 256, median 298); RAW rows sat on raw
pedestals 300–700. Fixed 7505338. **F16 and F20 numbers came from the offset caches and must be redone.** Local
wf-npz caches were corrected upstream (unaffected). In run_71 / wf-npz caches `ref_mesh` is det4's own leading strip,
not the uRWELL prediction (docstrings fixed).

## σ0 / Dp from head-on RAW run_71 (1200 clean tracks / view / field; bootstrap errors; χ²/dof 9–10)
| field | σ0_X [mm] | Dp_X [mm/√ns] | σ0_Y [mm] | Dp_Y | D_rc,Y [mm²/ns] | Magboltz Dp |
|---|---|---|---|---|---|---|
| 243 V/cm | 0.306 ± 0.007 | 0.0109 | 0.297 ± 0.020 | 0.0086 ± 0.002 | 2.74e-4 | 0.0057 |
| 156 V/cm | 0.287 ± 0.006 | 0.0100 | 0.332 ± 0.012 | 0.0035 ± 0.001 | 2.77e-4 | 0.0052 |
| 96 V/cm | 0.285 ± 0.008 | 0.0100 | 0.324 ± 0.014 | 0.0046 ± 0.001 | 2.97e-4 | 0.0050 |
- σ0_Y 0.32 ± 0.02 vs fleet production 0.343 (det3): within 7 % → one physical Y footprint is supported.
- σ0_X 0.29 is an upper bound: X width grows with drift time at 2× Magboltz, Y does not — the bench X-width excess
  (cloud_basics §29) in a second gas → detector effect, likely the unmodelled one-sided X copy. Defensible 0.24–0.29.
- σ0 flat ±0.03 mm vs amplitude (thirds) at every field; per-event σ0 unconstrained (0.19–0.41), 8–14 % pick ≥ 0.42
  by scatter → the per-event wide-footprint choice has no physical basis for unsaturated pulses. Saturation untested
  (beam max 3342 ADC).
- Proposal: fix σ0_Y ≈ 0.33, σ0_X 0.24–0.29 fleet-wide; drop the per-event profile (decision with A/H).

## ZS treatment (ZS emulated on RAW run_71, same events)
Suppressed samples as measured zeros: t0 ≤ 27 ns, position 15–27 µm, slope 0.03–0.07 µm/ns from RAW (best).
Censored (what F16/F20 used): 44–66 ns, 45–86 µm, 0.14–0.31 µm/ns (worst). `WFT_ZS_BOUND` (new opt-in) in between.
With zeros measured, ZS recovers σ0_X (−0.02) and Dp_X; it cannot split σ0 from RC in Y (σ0_Y +0.08–0.10, D_rc,Y to
grid floor).

## Position resolution (run_63, width of position − uRWELL prediction, no reference subtraction → upper bounds)
| | waveform fit | amplitude centroid |
|---|---|---|
| 0°, X / Y | 391 / 311 µm | 307 / 312 µm |
| 25.64°, X (not inclined) | 440–460 | 414–494 |
| 25.64°, Y (inclined) | 675–710, 34 % > 1 mm | meaningless |
Inclined view costs ~0.6 mm in quadrature at 25.6° (pessimistic: ZS-censored, 148/113 V/cm plateaus). Reference term
here ~250 µm (worse than the 3-pitch method's 176 µm). Bench 0.5 mm consistent with M3 pointing + angle.
x–y t0 difference on clean RAW head-on: robust σ 84–118 ns → 60–80 ns/plane, intrinsic to the free charge profile.

## Quantity → SPS → transfer → constraint
Bench gas Ar/iso 95/5; SPS wet Ar/CO₂/iso and Ar/CF₄/iso with O₂ → no gas property transfers numerically.
| quantity | SPS measurement | precision / caveat | transfers? | constraint |
|---|---|---|---|---|
| σ0_Y | 0.32 ± 0.02 mm | ZS-censored reads +0.1 | yes (readout) | fix ~0.33 fleet-wide |
| σ0_X | ≤ 0.29 ± 0.01 | bound (X copy) | as bound | 0.24–0.29 |
| σ0 vs amplitude | flat ±0.03 | unsaturated only | yes | drop per-event profile (unsaturated) |
| D_rc,Y | 2.8 ± 0.15e-4 per event; 2.2e-4 stacks | correlated with σ0_Y | det4 only | keep 2.2e-4, quote 2.2–3.0 |
| Dp | X 0.010–0.011 (2× Magboltz); Y ≈ Magboltz, weak | O₂ front-loads charge | no (gas) | don't force Magboltz Dp on X |
| sharing | Y continuous RC, X none, ±1 delay +60 ns, 6 conditions | settled | yes | in phys model |
| electronics shape | identical bench/beam | ZS distorts tails | shape | bench templates |
| v(E), gap | CF₄ ladders agree 0.6 %; absolute disputed | workstream C | no | none |
| gain | det4 ×20 swings in 12 mm bands | ZS amp ≠ gain | no | none |
| angle scale 25.64° | void until redone on fixed caches | — | model test | none yet |
| angle resolution 25.64° | void until redone | — | det4 only | paper number once redone |
| position 0° | 176 µm (run_53, earlier) | head-on | technology bound | det4 intrinsic for M3 deconvolution |
| position 25.64° | ~0.6 mm extra (inclined view) | pessimistic | det4 only | none |
| t0 | 60–80 ns/plane | intrinsic | yes | x–y gate follows it |
| dead channels | X 9, Y 5 from data | — | by position (F12) | G's h4 mask |

## Other changes
`sps_reco.py` censors exact zeros on RAW rows (23 % front-end-dropped samples had been fitted as zeros: χ² +30 %,
σ0 +0.01). `wft/model.py`: `WFT_ZS_BOUND`, `P['missing']` in prep_plane (both off by default, identical when absent).
Condor nodes served a stale copy of an overwritten EOS script (17/48 jobs) → always stage code into a fresh directory.

## Still running / not collected
- condor 4580495: X footprint grid on 22 real ZS flat rows (run_55 CO₂, run_63 flat, run_66 gain scan, run_70 drift
  scan) → `/eos/user/d/dneff/x17rerun/ws_sps/out/headon_zsscan_*.pkl` (σ0_X vs gain, Dp_X vs field, two gases).
- Fixed rot25 caches rebuilding on lxplus (`/tmp/dneff_build_rot25.log`) → `/eos/user/d/dneff/x17rerun/ws_sps/data/fixed/`.
  F16 known-angle arms (censored / zeros / bound at d675, d575, d475) must be re-run on them; old DAG outputs invalid.
- 2 zeros-measured chunks died on memory (5/6 used).

## Risks
Grid holds position/slope at one base fit, steps 0.05 σ0 / 0.005 Dp, parabolic interpolation. One chamber (det4,
worst), one board, flat 10 ADC noise. Bound arm adds its penalty post-solve (rough χ² surface).

Figures: `figures/sigma0_profiles.png`, `sigma0_vs_amplitude.png`, `dp_vs_field.png`, `zs_treatment.png`;
results JSONs in `results/`.
