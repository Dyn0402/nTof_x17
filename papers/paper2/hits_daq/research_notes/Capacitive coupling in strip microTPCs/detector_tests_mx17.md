# MX17 tests of same-time neighbour coupling (2026-11-09)

Own analysis, not literature. Code: `papers/paper2/hits_daq/coupling_scan.py`. Chamber A = det3, bench cosmics, Run 1 (`sat_det3`, 12,366 planes in 4 window files) and Run 2 (`p2_det3`, 38,364 planes). Working tables: `$WORK_DATA/paper2_hits/capcoup/`.

Coupled model: M_i -> M_i + kappa (M_{i-1} + M_{i+1}), same time and same shape, on the pass model, strip list padded by 4. Charge profile q free. Pass model has no such term: sigma_p0 = 0.30 mm is its only same-time sharing. Our notes put the avalanche footprint near 33 um and the static induction floor at 0.07 mm, so 0.30 mm is not a width that any identified physical process sets.

Sections 1-5 use the same-shape form. Section 6 tests a derivative (bipolar) neighbour term.

## 1. Waveform chi2 versus (sigma_p0, kappa)

Run 1 file 0, 1,540 X / 1,536 Y planes, track fixed at the pass values; 7-point refit below. Median chi2 relative to (0.30, 0):

| sigma \ kappa (X) | 0 | 0.03 | 0.06 | 0.10 | 0.15 | 0.20 | 0.30 |
|---|---|---|---|---|---|---|---|
| 0.07 | 1.104 | 1.027 | 0.991 | 0.987 | 1.055 | 1.168 | 1.524 |
| 0.15 | 1.039 | 0.992 | 0.975 | 1.003 | 1.090 | 1.228 | 1.621 |
| 0.20 | 1.005 | 0.979 | 0.983 | 1.035 | 1.143 | 1.317 | 1.737 |
| 0.25 | 0.987 | 0.987 | 1.019 | 1.095 | 1.248 | 1.437 | 1.883 |
| 0.30 | 1.000 | 1.034 | 1.091 | 1.209 | 1.392 | 1.605 | 2.082 |

Y is the same picture, minima 0.934-0.940 at (0.07, 0.06), (0.15, 0.03-0.06). A valley runs from (0.30, 0) to (0.07, 0.10). kappa >= 0.15 is disfavoured at every sigma (chi2 up by 4 % or more against the minimum at sigma 0.07; much more elsewhere); kappa 0.3 up 50-120 %.

Track refitted per grid point (Nelder-Mead p0, w, t0, 120 evaluations, 3,076 planes), median chi2 ratio to (0.30, 0):

| (sigma, kappa) | X | Y |
|---|---|---|
| (0.25, 0.03) | 0.981 | 0.980 |
| (0.20, 0.06) | 0.973 | 0.973 |
| (0.15, 0.08) | 0.964 | 0.963 |
| (0.15, 0.10) | 0.994 | 0.992 |
| (0.07, 0.10) | 0.956 | 0.954 |
| (0.07, 0.15) | 1.043 | 1.032 |

Caveat: chi2/dof is about 66 here (median chi2 25-29k, dof ~380), so model error dominates and these are not statistical significances.

## 2. Waveform-fit angle against M3 along the valley

Same refit, tan = w/v remapped linearly per grid point, s68 against the M3 reference. X base 0.966 deg; Y base 0.985 deg. Paired bootstrap change in s68 stays within +0.030 +- 0.014..0.018 (X; largest at sigma 0.07) and +0.008 +- 0.010 (Y). Angle scale 1.026-1.033 X, 1.011-1.019 Y. The waveform fit's angle does not care where the point sits on the valley.

## 3. How much compression coupling makes

Re-simulation W' = coupled model + (W - M), arms on the event's own track. 400 planes, |w| > 2.5e-3, V0 ladder slope / true slope, median:

| kappa | base X/Y | metallic-like X/Y | point X/Y (no diffusion, no RC, sigma 0.07) |
|---|---|---|---|
| 0 | 1.250 / 1.469 | 1.191 / 1.299 | 1.067 / 1.107 |
| 0.06 | 1.307 / 1.532 | 1.243 / 1.352 | 1.092 / 1.148 |
| 0.15 | 1.379 / 1.652 | 1.310 / 1.411 | 1.152 / 1.215 |
| 0.30 | 1.460 / 1.671 | 1.389 / 1.477 | 1.224 / 1.301 |

So coupling alone, with no diffusion, can compress the ladder, the size being comparable with the observed X 1.19 only if kappa is near 0.2-0.3; the waveforms disfavour that.

Decomposition with pure model ladders (white noise, no fit residual), 3,076 planes, |w| > 2.5e-3:

| view | sigma, kappa | full | no kappa | no diffusion | no diffusion, no RC | floor (no kappa/diffusion/RC) |
|---|---|---|---|---|---|---|
| X | 0.30, 0 | 1.259 | 1.256 | 1.106 | 1.106 | 1.106 |
| X | 0.20, 0.06 | 1.265 | 1.229 | 1.110 | 1.109 | 1.067 |
| X | 0.15, 0.08 | 1.264 | 1.217 | 1.109 | 1.110 | 1.051 |
| X | 0.07, 0.10 | 1.269 | 1.204 | 1.103 | 1.103 | 1.035 |
| Y | 0.30, 0 | 1.595 | 1.585 | 1.345 | 1.145 | 1.142 |
| Y | 0.15, 0.08 | 1.594 | 1.556 | 1.352 | 1.149 | 1.065 |
| Y | 0.07, 0.10 | 1.581 | 1.549 | 1.354 | 1.149 | 1.049 |

Total compression is invariant along the valley (X 1.26, Y 1.58-1.60). Moving along it only re-attributes: kappa takes the part sigma_p0 gives up. At (0.15, 0.08), removing kappa alone costs 0.047 of X's 0.264 excess (18 %), removing diffusion costs 0.155 (59 %). Y model 1.59 vs real-data 1.47 in the same selection: the pure model overshoots on Y (known, not from coupling). Order of removal matters; these are differences, not a unique split.

## 4. Model-free reach test

Near-normal planes (|w| < 1.2e-3), early window t0 - 60 to + 180 ns (mesh-end charge), ratio of strip charge to centre strip, position-in-strip |phase| >= 0.25 for the far-side strips. Run 1 (n 386 X, 463 Y) / Run 2 (n 1,219 X, 1,369 Y):

| strip (far side) | X data R1 / R2 | Y data R1 / R2 | model (0.30, 0) X / Y | (0.2, 0.06) X / Y |
|---|---|---|---|---|
| -1 | 0.0596 / 0.0631 | 0.1088 / 0.1024 | 0.054 / 0.107 | 0.073 / 0.115 |
| -2 | 0.0139 / 0.0128 | 0.0168 / 0.0167 | 0.0001 / 0.0013 | 0.0009 / 0.0038 |
| -3 | 0.0011 / 0.0011 | 0.0016 / 0.0024 | | |
| -4 | 0.0007 / 0.0000 | 0.0000 / -0.0004 | | |

Statistical errors on the medians ~0.002-0.003 (-1) and ~0.001 (-2). -3 and -4 are the control: no baseline or zero-suppression excess. Data are 2.5 % zero samples, censored.

The -2 strip carries 1.3-1.7 % in data against kappa^2 ~ 0.4 % at most: a nearest-neighbour coupling cannot make it; models with kappa <= 0.06 predict at most 0.4 %. Lorentzian footprint tails (eta 0.15-0.3, gamma 0.3-0.5) reach 0.9-1.3 % on X but 0.2 % on Y.

Position-in-strip dependence of the far neighbour at early time (Run 1, n ~ 110-210 per bin; phase bins 0-0.1, ..., 0.4-0.5):
- X data 0.120, 0.068, 0.056, 0.058, 0.066 (error 0.018, 0.007, 0.003, 0.002, 0.005); model (0.30, 0) 0.167, 0.118, 0.079, 0.056, 0.039.
- Y data 0.151, 0.116, 0.115, 0.104, 0.106; model (0.30, 0) 0.244, 0.190, 0.145, 0.114, 0.087.
A Gaussian footprint falls with phase; data flatten to a floor (X ~0.06, Y ~0.105). Injecting a Gaussian position error of 0.1, 0.2, 0.3 mm in the model leaves it at 0.043-0.044 (X) at phase 0.4-0.5, so position error does not make the floor. The X floor corresponds to kappa ~ 0.06 at small sigma. Near the strip centre the data are lower than every model tried.

## 5. Pitch-beat prediction (0.80 mm resistive vs 0.78 mm readout, 31.2 mm period)

Far -1 neighbour, X view, early window, 8 bins of x mod 31.2 mm. Run 2: 0.063, 0.059, 0.071, 0.070, 0.060, 0.056, 0.064, 0.062 (errors 0.003-0.008). Scatter of the size of the errors, no clear modulation. Only the far-side strip was tested.

## 6. Derivative-form neighbour term (added after sections 1-5)

M_i -> M_i + kd x (central difference per 60 ns sample of M_{i-1} + M_{i+1}): bipolar, zero net, the form the mechanism notes expect for direct strip-to-strip coupling into a charge amplifier. 3,076 planes, track fixed at the pass values, grid sigma/kappa along the valley, kd -0.3..+0.3. At every point the median chi2 is minimal at kd = 0 on both views: kd = +-0.05 raises it by 0.5-1.9 % (X +0.019/+0.005 at the base point; Y +0.006/+0.012), +-0.1 by 2-6 %, +-0.3 by 24-47 %. The waveforms do not ask for a derivative-shaped neighbour above a few percent per sample. The 60 ns sampling limits how sharp a transient this can see.

## Status and limits

- One chamber, near-normal samples for sections 4-5.
- Same-shape and derivative forms only; no term with its own delay.
- chi2 dominated by model error.
- The pass was calibrated at kappa = 0.
