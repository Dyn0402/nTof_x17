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
