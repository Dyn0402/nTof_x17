# Per-view sharing kernel — findings log (2026-10-09, overnight)

Running record of the PAPER_PLAN C2 study ("build the per-plane c2/c1 first").
The final write-up is generated from the result JSONs; this file is the log.

## 1 · Model-free (README.md): the two views differ in SHAPE

Head-on ±2/±1 area ratio, Y ≈ 0.40 everywhere (det4 SPS, det3/det4 bench),
X 0.10–0.23. X's ±1 is one-sided in H4 (+1 0.27 prompt, −1 0.43 delayed),
identical in the flat and 25.6° mounts and at 81–233 V/cm, so not a tilt.

## 2 · Corrections to the record

- **R06_GATE §8 says the beam's c2/c1 = 0.45 was measured on det4's X view.
  It is Y's** (`sharing_kernel/fit_kernel.json`: Y 0.42–0.47; X fits put c2
  at 0 with 3× Y's residual). The proposal "pin X at the beam value, leave Y
  free" was built on that mislabel.
- `wft/model.py:288` says "X cannot have resistive sharing". X's ±1 carries
  0.27–0.43 of the centre with real delays; what X lacks is the ±2 reach.
- **The hyper `c2/c1` is not the observable ±2/±1.** In r06, c1 sits on the
  fit floor `C1_MIN = 0.05` on det2/3/4/7; most of the observed ±1 (and much
  of ±2) is carried by the prompt Gaussian `sigma_p0` (0.39–0.43 mm, shared by
  both views, ~3× the H4 diffusion bound of 0.16 mm).
- **r06's recorded training chi2 (1.10e8) cannot be reproduced locally.** Its
  own hypers score 1.92e8 under a global t0 profile on the local
  `calib_work/calib_cache.pkl` (dated 08-05; r06 was fitted 08-19 elsewhere).
  Also: `wft.calibrate._event_chi2` warm-starts t0 in ±60 ns around the
  previous evaluation's minimum, on a surface with near-degenerate minima
  ~60 ns apart, so that objective is path-dependent. This study uses a
  deterministic cold global t0 profile (`recal.py --objective cold`).

## 3 · Closure (posterior predictive), r06 on det3, near-vertical

Model and data run through the SAME neighbour estimator (closure.py).

| view | observable | data | r06 model | pull |
|---|---|---|---|---|
| Y | ±2 area | 0.25–0.27 | 0.17–0.18 | −13σ |
| Y | ±2/±1 | 0.40 | 0.31 | −18σ |
| Y | ±2 delay | 240–290 ns | 190–207 ns | −6…−13σ |
| X | +2 area | 0.05 | 0.08 | +11σ |
| X | ±2 delay | 58–75 ns | 166 ns | +12σ |

Y needs MORE and SLOWER ±2; X's ±2 is prompt (geometric), not a delayed copy.
With p0 free, the fitted position sits 0.30 mm (rsig) from M3 — M3's pointing
error, which a reference-pinned calibration can only absorb into sigma_p0.

H4 det4 (run_71 RAW, K = 48 depth bins) with det4's bench kernel: X sharing too
wide (+13…+24σ), X timing one-sided (−1 delayed 112 ns, +1 prompt; +2 prompt,
−2 delayed), Y copies 76–97 ns too EARLY (−18…−22σ).

## 4 · The bench calibration chi2 is model-error dominated

chi2/dof ≈ 800 on the bench (18–40 on the beam).  With the ratio freed per view,
the bench fit moves sigma_s (the copies' time smear) 9 → 220 ns and wanders
between basins at < 3 % chi2: the kernel is used to patch template error.
Kernel constants fitted on this objective are effective, not physical.
`MODEL_FRAC` (fractional model-error weighting, archived model, RECO_BENCH
2026-07-29 §3) is ported to wft as opt-in and tested as a calibration arm.

## 5 · Arms (recal.py), judged on held-out reco (plane_bench.py) and the gate

g06 (control), pv (per-view ratio), pv_cx (+ X copy amplitude, one-sided
X), pv_sp (per-view sigma_p0), pv_all (+ tau_y_fac), diag_pv_free (ratio
unbounded, diagnostic only) × {pinned, p0-profiled} × {MODEL_FRAC 0, 0.05}.

_(results appended below as they land)_

### 5a · Model-free per-view pattern on all five bench chambers (|tan| < 0.05, big caches)

| chamber | X ±2/±1 | Y ±2/±1 | Y ±2 delay | X ±1 asymmetry (−1 vs +1) |
|---|---|---|---|---|
| det2 | 0.146 ± 0.005 | 0.511 ± 0.010 | 182–242 ns | +0.03 ± 0.04 |
| det3 | 0.151 ± 0.004 | 0.406 ± 0.005 | 240–280 ns | **+0.13 ± 0.03** |
| det4 | 0.172 ± 0.006 | 0.361 ± 0.007 | 206–259 ns | +0.06 ± 0.04 (H4: +0.10) |
| det6 | 0.148 ± 0.006 | 0.522 ± 0.015 | 240–302 ns | −0.01 ± 0.06 |
| det7 | 0.195 ± 0.014 | 0.540 ± 0.014 | 165–232 ns | +0.08 ± 0.06 |

Universal: Y's ±2 reach is 2.4–3.5× X's on every chamber, and delayed by
~150–250 ns more. X's one-sidedness is chamber-dependent (clear on det3 and on
det4 in H4, consistent with zero elsewhere) — a per-chamber, not a universal,
term.  This table is model-free and is Paper II §1 material as it stands.

### 5b · H4 baseline: det4's production kernel, gas refit only (run_71 RAW, head-on)

| plateau | sigma_p0 | Dp | X ±1 / ±2 vs data | Y ±1 / ±2 vs data | timing |
|---|---|---|---|---|---|
| raw700 (243 V/cm) | **0.145 mm** | 0.0135 | +9σ / +20σ | 0σ / +5σ | X ±2 +150 ns late; Y copies 70–90 ns early |
| raw450 | 0.050 mm | 0.0195 | +10…15σ / +20σ | +7σ / +12σ | same |

With a sharp reference (telescope; p0 profiled) sigma_p0 lands at the H4
diffusion bound (< 0.16 mm), against 0.26–0.45 mm in every bench fit — the bench
value carries M3's pointing error.  Even with the gas refit, the production
kernel over-shares on X and has the wrong timing on both views: the gas cannot
absorb the kernel's shape.  This is the bar each arm's kernel must clear in H4.

### 5c · First arm refits (training chi2; NOT the judge)

Freed per view, the Y ratio runs to its 0.95 cap on every chamber, and past 1
where allowed (diag: det4 2.59, det7 1.16; det6 0.66); the X ratio goes low
(det7 0.001, det6 0.10, det2 0.32, det3-mf05 0.12; det4 0.61 is the exception).
Chi2 gains over the control g06 are 0.3–1 %.  Note the control itself — r06's
global 0.6, refitted on the deterministic objective — already beats production
on the held-out bench (det3: s68 −0.02…−0.03 deg), so per-view effects are
measured against g06, not against production.

### 5d · First held-out benches (aggregate.py; 2000 held-out events; blind / ref start)

Paired Δs68 [deg], negative = better. "ctl" = against the same recipe with
r06's single ratio (the per-view effect alone).

| arm | X vs ctl (all / head-on) | Y vs ctl (all / head-on) |
|---|---|---|
| det4 pv | +0.00 ± 0.03 / −0.00 ± 0.03 | **−0.03…−0.055 ± 0.025 / −0.06…−0.09 ± 0.04** |
| det4 diag (Y ratio 2.6) | −0.04…−0.06 ± 0.025 / 0.00 | −0.055 ± 0.027 / −0.07…−0.10 ± 0.04 |
| det3 pv_mf05 | −0.03…−0.05 ± 0.02 / −0.01…−0.04 | −0.01…+0.01 ± 0.02 / ±0.05 |
| det6 diag | −0.01…−0.03 ± 0.04 | +0.02…+0.03 ± 0.03 |

- MODEL_FRAC 0.05 calibration+reco is clearly WORSE than production (det3 Y
  +0.30…+0.46 deg; det4 Y +0.04…+0.25) despite an 8–10 % lower training chi2.
  Closed at 0.05.
- det4: the refit control g06 is 0.13–0.16 deg WORSE than production on X
  (production det4 = lp_t0p, unslaved c2 = 0.67 c1, fitted on another cache).
  Any candidate must beat what ships on its own chamber, not just its control.

### 5e · Second batch: refitting the ratio is unreliable; p0-profiling; H4 carry-over

Per-view effect against its own control (blind start, Δs68 deg):

| chamber | pv X (all / head-on) | pv Y (all / head-on) |
|---|---|---|
| det2 | −0.004 ± 0.010 / −0.011 ± 0.015 | −0.017 ± 0.019 / −0.016 ± 0.024 |
| det3 | +0.017 ± 0.013 / **+0.060 ± 0.020** | **+0.040 ± 0.015 / +0.118 ± 0.026** |
| det4 | +0.002 ± 0.026 / −0.004 ± 0.028 | −0.030 ± 0.025 / −0.064 ± 0.038 |
| det6 | −0.025 ± 0.044 / −0.050 ± 0.066 | +0.017 ± 0.035 / +0.033 ± 0.045 |

The freed-ratio refit is NOT a reliable improvement: det3 gets worse (its refit
walked to sigma_s ≈ 280 ns), det4 slightly better, det2/det6 neutral.  The 180-
event bench chi2 cannot pin the per-view ratio; the freedom is spent elsewhere.

p0-profiled calibration barely moves the bench sigma_p0 (det3 0.42 → 0.39, det2
→ 0.36, det4 0.38–0.44, det7 0.43), so **M3 pointing error is a minor part of
it**; the bench/beam gap (≈ 0.4 vs 0.12–0.15 mm) is mostly the gas (Ar/iso 95/5
vs Ar/CF4/iso).  It does rescue det7's control on Y (+0.195 → +0.010 vs
production) and det6 pv_pp beats production on both views (X −0.08, Y −0.12).

H4 carry-over (kernel fixed, gas refit, head-on closure; Σ pull² over 16
observables): production 3150 (700 V) / 2906 (450 V); bench-refit g06 11345 /
10504; pv 13489 / 13140 — pv overshoots Y ±2 by +50σ on the beam.  Bench-
refitted kernels carry bench gas physics; the carry-over mixes kernel and gas
and is a weaker judge than hoped.

**Next (running):** the per-view ratios PINNED from the measurement on the
production hypers, no refit — x ∈ {0.1, 0.2, 0.3, 0.6}, y ∈ {0.6, 0.8, 0.95},
all five chambers.  The smoke test of that form (det3, x 0.2 / y 0.9) gave
Y head-on −0.07 ± 0.035 deg against production.

### 5f · PINNED per-view ratios on the production hypers (no refit) — the clean result

Held-out bench, blind start, Δs68 [deg] against production (negative = better),
Y ratio 0.95 with X left at production:

| chamber | Y all | Y head-on (\|θ\| < 5°) | X all |
|---|---|---|---|
| det2 | −0.041 ± 0.022 | **−0.146 ± 0.033** | 0.000 |
| det3 | **−0.036 ± 0.012** | **−0.070 ± 0.022** | 0.000 |
| det4 | −0.032 ± 0.023 | −0.078 ± 0.031 | −0.016 ± 0.014 (x 0.6 vs prod 0.67) |
| det7 | −0.025 ± 0.026 | **−0.164 ± 0.040** | 0.000 |
| det6 | +0.027 ± 0.04 | +0.061 ± 0.057 | −0.03 ± 0.04 (x 0.6 vs prod 0.82) |

Monotonic in the Y ratio (0.6 → 0.8 → 0.95) on det2/3/4/7.  det6 (whose
production is a different representation: sigma_p0 0.039 mm, c1 0.064, c2 free
at 0.82) is neutral.  **Lowering X's ratio does NOT help reconstruction**
(det3 head-on +0.04 ± 0.016, det7 all +0.04 ± 0.02 at x 0.2) — the measured
X pattern (little ±2) does not translate into a lower X copy ratio in this
representation, where X's sharing is carried by the prompt sigma_p0.

H4 det4 head-on closure (Σ pull², 700 V): production 3150; pin x0.2/y0.6 2340;
x0.1/y0.95 2612; x0.2/y0.95 2705; x0.6/y0.95 3422.  On the beam a LOW X ratio
is right (X ±2 pull +19σ → +2…+7σ), matching the model-free measurement, while
Y fits best at 0.6 and 0.95 overshoots ±2 (+5σ → +17σ).  Every variant leaves
Y's copies 13–15σ too EARLY on the beam.  So the in-model ratio is not a pure
board constant: it trades against the gas-dependent prompt spread and the copy
timing.  The beam says X's ratio should be low; the bench reco is indifferent
to X and wants Y higher.

Gate (full reco, golden keys) running: candidate `calib_bundle_pvy95` =
production hypers with c2_over_c1_y = 0.95 (X unchanged), against
`calib_bundle_prodt0` = production hypers with the SAME t0 re-measurement.

### 5g · Diagnostic: Y past the c2 < c1 gate (pinned, production hypers, NOT shippable)

Y Δs68 head-on vs production [deg]: y = 0.95 / 1.2 / 1.6

| chamber | 0.95 | 1.2 | 1.6 |
|---|---|---|---|
| det2 | −0.146 ± 0.036 | −0.161 ± 0.039 | −0.167 ± 0.048 |
| det3 | −0.070 ± 0.023 | −0.124 ± 0.027 | **−0.168 ± 0.031** |
| det7 | −0.164 ± 0.041 | −0.194 ± 0.046 | −0.193 ± 0.059 |
| det4 | −0.078 ± 0.028 | −0.078 ± 0.034 | −0.030 ± 0.044 |
| det6 | +0.061 | +0.014 | −0.025 (flat) |

(all-angle: det3 −0.036 / −0.056 / −0.061; det7 −0.025 / −0.058 / −0.103.)

**The hyper-level c2 < c1 gate is the binding constraint on det2/3/7.**  In this
representation the prompt sigma_p0 already carries most of the observed ±1, so
the *delayed* ±2 copy can exceed the *delayed* ±1 copy while the observable
±2/±1 stays 0.36–0.54 (5a) — the physical ordering (±2 reached only through
±1) is a statement about the observable, which every arm respects.  Whether to
move the gate from the hyper to the observable is Dylan's call; the shippable
candidate stays at y = 0.95.

Refit arms in this batch confirm 5e: freeing more hypers is no better than
pinning (det3 pv_all/pv_sp worse head-on; det4 pv_sp_pp Y −0.11/−0.18 vs prod is
the one strong refit, det7 pv_all Y −0.135 vs its (bad) control).

### 6 · FULL-RECO GATE: pvy95 (production hypers, c2_over_c1_y = 0.95) vs prodt0

Golden keys, full matched sample, same code, paired bootstrap on identical
events; prodt0 = production hypers with the same t0 re-measurement (and it
reproduces shipped production within errors — `prodref` rows in
`<wft>/plane_ratio/gate_pvy95.json`).

| key | Y s68 | Y s68 \|θ\|<5° | Y implied-v spread | X | within 5 mm / core σ |
|---|---|---|---|---|---|
| sat_det3 | 1.227 → **1.187** (−0.040 ± 0.007, 5.4σ) | 1.429 → **1.327** (−0.101 ± 0.013, 7.7σ) | 0.017 → 0.008 | ±0.001 | 93.39 → 93.42 % / 0.446 → 0.449 mm |
| o22_long_det2 | 1.711 → **1.648** (−0.063 ± 0.018, 3.5σ) | 2.056 → **1.938** (−0.118 ± 0.024, 5.0σ) | 0.074 → 0.060 | ±0.001 | 91.97 → 92.06 % / 0.439 → 0.435 mm |

This undoes most of what r06 cost on Y (R06_GATE: det3 Y +0.061, head-on
1.22 → 1.43; det2 Y +0.122) while keeping r06's X and the physical ordering.
