# ws_floor — F3: the physics floor on the angle resolution (toy model for the paper), 2026-10-27

(Written by the coordinating session from the workstream's final message. Commit 964275e. Per-event .npy dumps moved to
`~/x17/cosmic_bench/model_audit/ws_floor/`; every number below is in `results/summary.json`.)

## Result
Electron-by-electron toy (Fischle cluster sizes + δ-ray ranges, Magboltz diffusion, Polya gain, measured footprint,
template, sampling, noise, ADC ceiling), fitted by the production estimator with the toy's exact mean model, started
at truth; self-contained (no `wft`). **Floor 0.30–0.35° (s68) at every angle 0–45°, all five bench chambers.**
Measured: det3 X/Y 1.06/1.02°, det2 1.10/1.34, det4 1.65/1.71, det6 1.76/1.89, det7 1.57/1.56 → 3–6× the floor.
The old "~1° physics floor" (script 18, `FLOOR_TAN = 0.018` in `wft/reco.py`) is withdrawn: script 18 needed 0.30 mm
centroid jitter per 60 ns bin; the physical value is 0.185 mm rms.

det3 vs angle (s68 °, ±0.02–0.04):
| angle | 0 | 10 | 20 | 25 | 30 | 35 | 40 | 45 |
|---|---|---|---|---|---|---|---|---|
| every electron seen | 0.18 | 0.18 | 0.18 | 0.17 | 0.13 | 0.13 | 0.11 | 0.11 |
| noise only (Cramér–Rao) | 0.003 | 0.006 | 0.011 | 0.013 | 0.015 | 0.015 | 0.013 | 0.011 |
| perfect model, 20 ns bins | 0.34 | 0.34 | 0.30 | 0.27 | 0.27 | 0.30 | 0.33 | 0.41 |
| perfect model, 30 ns bins | 0.34 | 0.35 | 0.31 | 0.29 | 0.33 | 0.46 | 0.60 | 0.66 |
| perfect model, 60 ns (production) | 0.35 | 0.36 | 0.34 | 0.63 | 0.86 | 1.18 | 1.35 | 1.36 |
| measured det3 X / Y (vs M3, ≤ 0.15° M3 not subtracted) | 1.21/1.10 | 1.00/0.91 | 1.02/0.85 | | | | | |

Breakdown (20 ns, one effect off, at 5/20/35°; all on 0.34/0.30/0.30): no diffusion 0.02/0.05/0.09 (diffusion IS the
floor); no cluster-size fluctuation 0.25/0.20/0.19; no gain fluctuation 0.29/0.29/0.24; noise negligible (covers det4's
4× lower charge); ionisation alone 0.01/0.02/0.11.
Closed form: σ_θ ≈ D_T √(6(1+f)/(N·G)), N drift electrons, G gap, f = 1/(1+θ_Polya) → det3 0.29°. D_T 0.110–0.126
mm/√mm on all bench chambers → same floor everywhere.

## By-product (F4, n_TOF)
With 60 ns charge bins even a perfect model degrades and biases steep tracks shallow: −0.08 / −0.23 / −0.51° at
25 / 30 / 40° (≤ 0.05° flat to 45° with 20 ns). Mechanism: v·Δt·tanθ ≳ pitch, so "uniform charge within a bin" can't
represent clustered ionisation (vanishes without clustering; weaker on slow det6). Bench: ~−0.05° at 20° (small part of
F4). n_TOF pairs at 30–45°: the main term. Recommendation: 20 ns or interpolated charge grid (~3× columns), untested on data.

## Estimator notes
Weighting noise² + 0.19·(ADC/e)·prediction is the correct error model (toy χ²/dof 0.9–1.0), gains 0.02–0.03°; F17's
variance ∝ prediction² was the wrong form. Toy χ²/dof with production weighting 4–40 vs data 60–700 → data excess = model error.

## Recommendation to H (F3)
Per-chamber/view "model floor" stored in the bundle at finalize (≈0.018 tan det3, 0.028–0.033 det4/6/7); never the
physics floor (0.006 tan).

## Literature
Bortfeldt et al., NPPS 273–275 (2016) 1173, arXiv:1508.00802 (floating-strip µTPC "below 5°", abstract only);
ATLAS NSW µTPC arXiv:1810.01394, Alexopoulos et al., NIM A 937 (2019) 125 (LMU seminar ≈2° at 20°, ≈3° at 26.5°, not
citable); Fischle, Heintze, Schmidt, NIM A 301 (1991) 202; Sauli, Gaseous Radiation Detectors (2014). No published
first-principles µTPC angle floor found — re-check before claiming.

## Proposed paper paragraph
> To separate detector physics from reconstruction, we simulate the chain electron by electron — primary clusters with
> the argon cluster-size distribution including δ-ray ranges, Magboltz diffusion, Polya gain, the measured prompt
> footprint, strip pitch, pulse shape, 60 ns sampling and noise — and fit the waveforms with the simulation's exact mean
> model. The resulting 0.30–0.35° for all chambers and angles up to 45° is the floor of the method. It is set by
> transverse diffusion, σ_θ ≈ D_T√(6(1+f)/(NG)), with gain and cluster-size fluctuations next and electronic noise below
> 0.02°; reading every electron would give 0.1–0.2°. The measured 1.0° (best chamber) is three times this floor: model
> error of the charge-dispersion response, not a detector limit. The charge profile must also be sampled finer than one
> strip per depth bin (20 ns): with 60 ns bins, tracks steeper than 25° lose resolution and read 0.2–0.5° too shallow.

## Limits
X view only (Y along-strip spreading not modelled), projected angle, ideal start; no dead strips, CM residual, gain
non-uniformity; bench gas only (n_TOF gas ~½ D_T → lower floor, not run).

Files: floor_toy.py, measured.py, make_figure.py, run_all.sh, results/*.json; figures/floor_vs_angle_det3.png (paper
candidate), floor_breakdown_det3.png, floor_bias_grid_det3.png, floor_fleet.png.
