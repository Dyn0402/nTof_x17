# Using all the information: the error model of the waveform fit (2026-10-29 night)

> **CLOSED (Dylan, 2026-10-29): the simple model is locked.** No cap, no per-event width, no variance model, no strip
> nuisance, no rails in the pass; the "Next" list below is not pursued. The decision, the measured cost (~0.05° on
> det3) and the list of known simplifications carried into the pass: `DECISION_2026-10-29.md`.

## Decision (2026-10-29 late): NOT adopted -- the recipe fails two gates, and so does the cap alone

`wft/model.py` defaults are unchanged. Two independent failures:

1. **Full reconstruction (det6, det7): efficiency lost.** The held-out harness fits windows cut along the M3 corridor
   with t0 pinned; the full reconstruction fits seeded windows and picks wells itself. There the cap costs det6
   −0.70 pt (recipe) / −0.82 pt (cap alone) of interior detection at 5 mm and det7 −1.62 pt (cap alone), against a noise
   level of ±0.16 pt (LM gate), and det7's angles get WORSE (+0.07 / +0.13°). Mechanism, read event by event: on a
   near-vertical, heavily clipped column (≥ 3 clipped strips) every bright sample is only "≥ 1200", so a steep line
   predicting ≥ 1200 there fits as well as the true vertical one; the fit runs off (det6: 65 X + 20 Y planes go from
   |tan| < 0.6 to > 1, 64 of them with ≥ 3 clipped strips; e.g. M3 at 260.0 mm, base p0 258.0 tan −0.09, capped p0 307.7
   tan −2.61). Those rays move detected → far (det6 78 vs 20 back; det7 also detected → hw_known 137 vs 44 back, the
   track lands in the known-fault zone), and the lost tracks also lose their x-y pair and their two-track split
   (det6 splits 102 → 56). On det2/det3/det4 the net detection change is −0.15…+0.12 pt; the far fraction still rises
   +0.1…+0.5 pt everywhere because the relative χ²/dof flag now passes far tracks it used to reject (rejected → far
   21–34 per chamber vs 10–18 back).
2. **Cross-dataset (det3): the per-strip amplitude nuisance flips sign with the drift field.** The recipe is worse on
   det3's 500 V drift point (X +0.080 ± 0.041°, Y +0.022 ± 0.040°, 704 held-out events). `STRIP_TAU` alone is worse on
   every det3 dataset below 1000 V (500 V +0.196 ± 0.044 / +0.113 ± 0.037; 700 V +0.039 / +0.013; the 6-25 day, also
   500 V, +0.046 ± 0.011 / +0.025 ± 0.015) and neutral at 1000 V (6-28 night −0.008 / −0.027). A correction whose sign
   depends on the dataset is what the rule excludes.

**What survives:** in the held-out harness the cap alone (`SAT_CAP=1200` + `QSOLVE_PLAIN`) wins on all six det3
datasets and is neutral to helpful elsewhere, with the charge intact -- the angle information really is in where and
when the charge arrives. It cannot go in as a cap on every fit; the next form to try is a cap that cannot move the
well: the uncapped fit chooses (p0, w, t0) as today, and the capped likelihood only polishes inside that well (or is
refused when the polish moves |tan| by more than its error / leaves the plausibility window). That needs its own
held-out + full-reconstruction gate; the infrastructure for both is in place (below).

**Rails (`SAT_RAIL`) are now production-ready** (`wft.io` carries them, opt-in, below) but add nothing to the cap's
geometry: under `SAT_CAP=1200` a rail only matters for a sample clipped below 1200 ADC (0 windows on det2/det3,
9 / 62 / 15 of 3000 on det4 / det6 / det7, all coherent CM excursions); recipe with vs without rails differs by
≤ 0.012° on every chamber. The rail does change the CHARGE through the plain charge solve: on det4 blobs x-y pairing
power 0.84 → 0.92 (held-out, with or without the cap). Full reconstruction with the rail alone: see the table.

### Full reconstruction (gate), five golden rows, 8000 matched events, pinned FE bundles of the LM gate

`make_gate_dag.py --arms sat_base sat_cand sat_cap` (code `c290771`) and `sat_candr sat_rail` (code `bacf7fe`, rails
carried by `wft.io`), on `/eos/user/d/dneff/x17rerun/followups/fe/Analysis_fe`; tables `gate_compare.py --base sat_base`
and `satgate_pairs.py` (`results/satgate_pairs.{json,md}`). Interior x-y detection at 5 mm, Δ in points; s68 vs M3 on
the production alignment. recipe+rail = `sat_candr`.

| chamber | arm | Δ detected | Δ far | rejected (base) | Δs68 X / Y [°] | splits (base) | n_tracks ≥ 2 (base) | x-y log-q σ / pairing (base) | steep runaways X+Y |
|---|---|---|---|---|---|---|---|---|---|
| det2 | recipe | −0.15 | +0.47 | 1.53 (1.86) | −0.058 / −0.079 | 16 (18) | 50 (52) | 0.068 / 0.922 (0.065 / 0.921) | 4 |
| det2 | cap alone | −0.15 | +0.44 | 1.56 | −0.052 / −0.064 | 17 | 50 | 0.068 / 0.924 | 4 |
| det2 | rail alone | +0.06 | −0.12 | 1.92 | −0.005 / −0.005 | 18 | 50 | 0.064 / 0.924 | 1 |
| det3 | recipe | +0.12 | +0.11 | 0.91 (1.13) | −0.051 / −0.055 | 39 (41) | 101 (100) | 0.060 / 0.909 (0.057 / 0.908) | 7 |
| det3 | cap alone | +0.06 | +0.19 | 0.90 | −0.050 / −0.036 | 40 | 102 | 0.059 / 0.910 | 9 |
| det3 | rail alone | +0.02 | 0.00 | 1.12 | 0.000 / −0.001 | 40 | 98 | 0.057 / 0.908 | 0 |
| det4 | recipe | +0.11 | +0.14 | 3.16 (3.24) | −0.215 / −0.077 | 24 (33) | 88 (86) | 0.115 / 0.933 (0.116 / 0.929) | 31 |
| det4 | cap alone | −0.08 | +0.11 | 3.20 | −0.095 / −0.012 | 15 | 85 | 0.113 / 0.935 | 20 |
| det4 | rail alone | −0.02 | 0.00 | 3.24 | −0.034 / −0.008 | 25 | 86 | 0.116 / 0.928 | 4 |
| det6 | recipe | **−0.70** | +0.75 | 2.18 (2.36) | +0.011 / −0.122 | 59 (102) | 83 (92) | 0.155 / 0.817 (0.153 / 0.816) | **86** |
| det6 | cap alone | **−0.82** | +0.87 | 2.20 | −0.009 / −0.073 | 56 | 86 | 0.158 / 0.817 | **85** |
| det6 | rail alone | +0.01 | +0.04 | 2.29 | −0.016 / −0.023 | 77 | 89 | **0.139 / 0.865** | 7 |
| det7 | recipe+rail | **−1.36** | +0.40 | 2.50 (2.62) | −0.061 / −0.053 | 47 (65) | 37 (37) | 0.175 / 0.835 (0.169 / 0.827) | **64** |
| det7 | cap alone | **−1.62** | +0.61 | 2.41 | **+0.067 / +0.125** | 50 | 35 | 0.185 / 0.827 | **46** |
| det7 | recipe | **−1.37** | +0.41 | 2.50 | −0.061 / −0.049 | 42 | 34 | 0.180 / 0.832 | **68** |
| det7 | rail alone | +0.13 | −0.12 | 2.56 | −0.028 / −0.002 | 60 | 40 | 0.166 / 0.829 | 6 |

recipe+rail equals the recipe on det2/det3/det4/det6 to ≤ 0.13 pt and ≤ 0.001° (det6 −0.83 pt). Per event (recipe vs
base): the same w on 83–95 % of planes, dθ > 1° on 6–9 % (det2/det3) and 14–23 % (det4/det6/det7), against 1.4–1.7 % /
3–11 % for the LM gate; the quality flag agrees on 91–95 %; q_sum ratio p50 0.998–1.000. Budget moves per M3 ray
(interior, 5 mm): det6 recipe detected → far 78, far → detected 28; det7 cap alone detected → far 63 (19 back) and
detected → hw_known 137 (44 back). The rail alone moves 2–82 rays per chamber, symmetric, and on det6 tightens the x-y
charge agreement (0.153 → 0.139) with pairing power 0.816 → 0.865 at no cost in efficiency or angle (det7 +0.13 pt,
−0.028 / −0.002°; det2/det3/det4 −0.02…+0.06 pt): the per-sample rail is a real charge improvement where the CM moves
the rail (det6), neutral on det3 -- a separate candidate for Dylan, which by the rule ("wins on det3") is not adoptable
as a gain, only as a correctness fix.

### Held-out, base = the current algorithm without rails (the production path), Δs68 [°], all events, each arm's own scale

| dataset | events | recipe X | recipe Y | cap alone X | cap alone Y | strip nuisance alone X / Y |
|---|---|---|---|---|---|---|
| det3 golden (6-27 sat, 1000 V) | 1320 | −0.067 ± 0.026 | −0.016 ± 0.021 | −0.047 ± 0.024 | −0.006 ± 0.020 | |
| det3 golden, second sample | 4818 | −0.047 ± 0.010 | −0.049 ± 0.011 | **−0.052 ± 0.010** | **−0.037 ± 0.010** | |
| det3 6-28 night (1000 V) | 4000 | −0.043 ± 0.015 | −0.037 ± 0.012 | **−0.034 ± 0.012** | −0.020 ± 0.011 | −0.008 / −0.027 |
| det3 6-27 drift 700 V | 734 | −0.028 ± 0.027 | −0.044 ± 0.028 | −0.026 ± 0.026 | −0.035 ± 0.031 | +0.039 / +0.013 |
| det3 6-27 drift 500 V | 704 | **+0.080 ± 0.041** | +0.022 ± 0.040 | **−0.064 ± 0.027** | −0.050 ± 0.027 | **+0.196 / +0.113** |
| det3 6-25 day (bottom slot, 500 V) | 4000 | −0.041 ± 0.016 | −0.064 ± 0.019 | **−0.056 ± 0.015** | −0.032 ± 0.021 | **+0.046 / +0.025** |
| det2 golden | 1320 | −0.032 ± 0.023 | −0.083 ± 0.030 | −0.053 ± 0.023 | −0.062 ± 0.028 | |
| det2 6-22 long | 2000 | −0.067 ± 0.019 | −0.073 ± 0.025 | −0.064 ± 0.021 | −0.050 ± 0.025 | |
| det4 golden | 1320 | −0.062 ± 0.046 | −0.044 ± 0.033 | −0.065 ± 0.038 | +0.006 ± 0.029 | |
| det4 6-23 night | 2000 | −0.089 ± 0.039 | +0.006 ± 0.028 | −0.035 ± 0.035 | +0.039 ± 0.024 | |
| det6 golden | 1320 | +0.023 ± 0.035 | −0.085 ± 0.046 | +0.011 ± 0.033 | −0.039 ± 0.042 | |
| det6 6-26 longer | 2000 | −0.033 ± 0.032 | −0.090 ± 0.041 | −0.047 ± 0.025 | −0.093 ± 0.032 | |
| det7 golden | 1312 | −0.142 ± 0.043 | −0.072 ± 0.036 | −0.124 ± 0.046 | −0.011 ± 0.043 | |
| det7 6-26 longer | 1975 | −0.084 ± 0.035 | −0.041 ± 0.030 | −0.095 ± 0.031 | +0.042 ± 0.028 | |

Charge on every row: median q ratio arm / base ≥ 0.997 (recipe on det3 500 V 0.993), x-y log-charge σ within
+0.005 and pairing power within ±0.008 of the base (cap alone; the recipe alike), positions equal or better (cap alone
−0.03…+0.03 mm). Far fractions within ±1 pp. (The fleet table further down was scored against the `SAT_RAIL` arm; against
the production path without rails det7 X reads −0.14 rather than −0.07, because the rail alone gains det7 X −0.08.)

Dylan: "if we aren't using all the information our detector gives us, we're probably doing something wrong."
Follow-up to the clean-track check (`SATURATION_STUDY.md`): on clean single det3 tracks the angle got *better* the
more of the amplitude was hidden (clip at 300 ADC: X −0.085, Y −0.052°) while the charge got worse.

## Verdict of the evening (golden rows, held-out only; superseded by the Decision above)
- **Recipe that uses the information correctly, now on five chambers:** geometry from a fit that treats every sample
  ≥ 1200 ADC as "at least 1200" (`SAT_CAP=1200`) with a per-strip amplitude nuisance (`STRIP_TAU=0.05`), and the
  charge re-solved at that geometry from every sample (`QSOLVE_PLAIN`). Angles improve or are neutral on every chamber
  and view, positions improve (det4 by 0.03 mm), and the charge is exactly preserved (x-y charge σ and pairing
  power unchanged). It meets the rule "wins on det3, reasonable elsewhere"; it is **not yet** checked across det3's
  other datasets (no caches; the pass gate will do it) nor through the full reconstruction (efficiency, multi-track,
  head-on rescue read χ²).
- **Why hiding amplitude helps, in information terms:** the electron-by-electron toy with an EXACT model
  (`rerun/model_audit/ws_floor`, `--sat`) loses nothing when clipped at 300 ADC (s68 0.33–0.35° at every ceiling), and
  its far outliers (rms 4–5°) vanish. With a free charge per 60 ns depth slice, the angle information sits in where
  and when charge arrives; bright-sample amplitudes add mostly fluctuation. A correct likelihood (the toy's Poisson
  variance) gains only 7–10 % (0.36 → 0.33°). In the data the clip *helps* (−0.05°) and the Poisson variance *hurts*
  (+0.10…+0.33°), both unlike the toy: the data's bright samples carry a systematic mean-model error (centre strip
  under-filled 5–11 %, T9) that the noise-only weights trust 10–20× too much. So the amplitude is informative for
  the charge (x-y matching) and only weakly for the angle, and our mean model is what stops us using it.
- **The charge does use the amplitude**: the cap alone reads charge 3–23 % low and loosens the x-y charge agreement
  (0.060 → 0.081); the plain re-solve restores it exactly. A 2 % fractional term in the charge solve tightens x-y
  agreement on det3 (0.059 → 0.048) and det4 but loosens det6/det7 and reads charge 4–11 % low: not adopted.

## Fleet (1320 held-out events per chamber; Δs68 [°] vs the true-clips-only fit; scale/zero from all events)

| chamber | angle X | angle Y | position X / Y [mm] | far X / Y [%] (base) | charge |
|---|---|---|---|---|---|
| det2 | −0.036 ± 0.022 | **−0.073 ± 0.030** | −0.007 / −0.014 | 6.7 / 6.4 (6.8 / 6.4) | unchanged |
| det3 | **−0.066 ± 0.025** | −0.015 ± 0.021 | +0.002 / −0.009 | 3.9 / 2.5 (3.9 / 2.7) | unchanged |
| det3, second sample (4818) | **−0.047 ± 0.010** | **−0.049 ± 0.011** | −0.006 / **−0.011** | 2.8 / 1.3 (3.1 / 1.5) | unchanged |
| det4 | −0.065 ± 0.044 | −0.031 ± 0.032 | **−0.035 / −0.027** | 9.0 / 9.5 (10.8 / 10.5) | unchanged |
| det6 | +0.029 ± 0.035 | −0.064 ± 0.043 | −0.004 / **−0.038** | 6.0 / 3.8 (6.2 / 4.3) | unchanged |
| det7 | −0.071 ± 0.045 | −0.057 ± 0.034 | −0.024 / −0.001 | 13.0 / 7.2 (12.0 / 6.6) | unchanged |

det3 clean tracks: X −0.053, Y −0.043 (338 events); second sample X −0.054 ± 0.022, Y −0.038 ± 0.015 (1312).
The pieces separately (det2/det3): the cap carries most of X, the strip nuisance most of Y; together is best overall.

## What was tried (det3, 4818 held-out events, all event-paired; `results/err_model_final.json`)

| error model | angle X | angle Y | x-y charge σ (0.060) |
|---|---|---|---|
| Poisson variance σ² = n² + K·m, K = 3 / 11.5 / 40 (prediction-based) | +0.02 / +0.13 / +0.34 | +0.10 / +0.20 / +0.33 | 0.088 / 0.140 / 0.205 |
| fractional σ² = n² + (F·m)², F = 0.02 / 0.03 / 0.06 | −0.018 / −0.011 / +0.033 | −0.021 / −0.027 / −0.012 | 0.044 / 0.050 / 0.074 |
| per-strip amplitude nuisance, τ = 0.05 / 0.15 | −0.011 / −0.012 | −0.019 / −0.020 | 0.068 / 0.069 |
| cap 1200 (charge from the capped fit) | −0.052 | −0.037 | 0.081 |
| cap 1200, charge re-solved plain | −0.052 | −0.037 | 0.061 |
| **cap 1200 + τ 0.05, charge re-solved plain** | **−0.047** | **−0.049** | **0.061** |

## Scoring correction found on the way
Scoring a class subset with an angle scale and zero re-fitted on that subset alone is noisy: on the 338 clean det3
events it put the zero +1.5° off and turned a −0.09° gain into a +0.15° loss. `clean_check.py` now uses each arm's
scale and zero from all its events of the view (as kw / w0 are measured per row on all events). Earlier clean/blob
tables in `SATURATION_STUDY.md` used the per-subset scale; their large-class numbers barely move.

## Switches (`wft/model.py`, all off = bit-identical; `wft/tests/test_err_model.py`)
`SAT_CAP` (study session), `STRIP_TAU`, `ERR_K`, `ERR_F`, `ERR_PASSES`, `QSOLVE_PLAIN`, `QSOLVE_F`. The error-model
passes wrap `fit_plane_raw` (fit, derive σ / strip scales from the prediction, refit from the solution).

## Next
1. ~~Full-reconstruction gate~~ and ~~det3's other datasets~~: done (Decision above); both failed the recipe.
2. A cap that cannot move the well (uncapped fit picks the well, capped likelihood polishes inside it, refused when it
   leaves the plausibility window): held-out on all 14 datasets (`make_satfin_dag.py --arms ... --no-build`) and the
   full-reconstruction gate (`make_gate_dag.py`), with det6/det7 efficiency and the steep-runaway count as the first
   checks.
3. The real fix is the mean model at the bright samples; when it is right, the Poisson variance should start to
   pay (toy: −7…−10 %).

## Rails in production (`wft.io`, opt-in, code `bacf7fe`)
`FeuReader.iter_events(with_raw=True)` also yields the undecoded samples; `extract_window(..., raw_all=)` puts
`rail_hi = W + 4095 − raw` and `rail_lo = W − raw` on the window (`wft.io.rail_arrays`). `wft.reco` carries them on
every seeded and head-on-rescue window only when `io.rails_wanted()` (model `SAT_RAIL` / `SAT_FLOOR` on, or
`io.CARRY_RAILS`); otherwise nothing extra is read or stored (bit-identical; 135 `wft/tests` pass). `build_cache(rails=)`
and `build_cache_wft.py --rails` store them in calibration caches, so the study sidecar (`rails.py`) is no longer
needed. Checked against the sidecar on all 3000 det3 golden windows from the local decoded_root: raw identical,
0 ADC difference; `wft/tests/test_io_rails.py`. Used by the gate's `sat_rail` / `sat_candr` arms on condor.

Files: `clean_check.py`, `fleet_check.py`, `results/err_model*.json`, `results/fleet_<det>.json`; toy
`rerun/model_audit/ws_floor/results/det3_full_sat*.json`; condor outputs `/eos/user/d/dneff/x17rerun/satur/{big,fleet}`.
Gate (10-29 late): `make_satfin_dag.py` (held-out DAG: HITS → CACHE → arms; caches `/eos/user/d/dneff/x17rerun/satfin/caches`,
arms `satfin/ho`; DAGs in `~/x17rerun/satfin/ho` on lxplus), `satfin_score.py` / `satfin_capscore.py` (scoring,
`results/satfin_*.json`); full reconstruction `make_gate_dag.py --arms sat_base sat_cand sat_cap sat_candr sat_rail`
(DAGs `~/x17rerun/satfin/gate/satgate{,2}.dag`, outputs `<row>/wft/sat_*` in `followups/fe/Analysis_fe`),
`gate_compare.py --base sat_base` (`results/satgate_compare.{json,md}`), `satgate_pairs.py`
(`results/satgate_pairs.{json,md}`), `satgate_lost.py` (lost-ray anatomy), `pull_gate.sh`.
