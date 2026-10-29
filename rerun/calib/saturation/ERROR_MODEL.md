# Using all the information: the error model of the waveform fit (2026-10-29 night)

Dylan: "if we aren't using all the information our detector gives us, we're probably doing something wrong."
Follow-up to the clean-track check (`SATURATION_STUDY.md`): on clean single det3 tracks the angle got *better* the
more of the amplitude was hidden (clip at 300 ADC: X −0.085, Y −0.052°) while the charge got worse.

## Verdict
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
1. Full-reconstruction gate on the five golden rows with the recipe (efficiency, multi-track, x-y pairing, head-on
   rescue, far fraction), condor.
2. det3's other datasets (6-28 night, 500/700 V, 6-25): build caches and re-run (the pass gate does this anyway).
3. The real fix is the mean model at the bright samples; when it is right, the Poisson variance should start to
   pay (toy: −7…−10 %).

Files: `clean_check.py`, `fleet_check.py`, `results/err_model*.json`, `results/fleet_<det>.json`; toy
`rerun/model_audit/ws_floor/results/det3_full_sat*.json`; condor outputs `/eos/user/d/dneff/x17rerun/satur/{big,fleet}`.
