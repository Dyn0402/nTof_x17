# Saturation (ADC-clipped pulses) in the wft reconstruction — a critical study (2026-10-29)

Dylan: "we are treating it as well as we can" is not an answer; find real improvements or show with numbers that there
are none. Held-out angle s68 vs M3, event-paired against the planned-pass base, det3 first, then det2/4/6/7.

## Verdict

> **CLOSED (Dylan, 2026-10-29): the simple model is locked** (`DECISION_2026-10-29.md`): the 3550 rule, lower-bound
> censoring, noise-only weights, no cap. The findings below stand as known simplifications K1, K3–K5, K7 of that file.

> **Latest (2026-10-29 night): `ERROR_MODEL.md`** — the recipe that keeps both the angle gain and the charge, on
> five chambers, and the toy result on why hiding amplitude helps. Then the clean-track check below.
>
> **Clean-track check (2026-10-29 evening).** The cap result holds on clean single tracks (not
> a δ artefact), but it is a symptom: the fit's noise-only error model over-trusts bright samples. Amplitude carries
> real information (the charge, the x-y matching, and the toy's 0.34°); the fix to try is the error model, not a cap.


**There is room, and it is not in the censoring rule.** Changing how the clipped samples themselves are treated does
nothing on det3: the per-sample rail and the floor bound are null to ±0.002–0.005° (all planes) on a second sample of
4818 events, and every other censor variant is null or worse on the first 1320. The fit's present treatment of the
clip is not what makes saturated planes worse: clipping planes that never saturated at an emulated rail costs them
nothing (they get 0.03° *better*). The lever is the opposite of what was asked about: **the bright samples are where
the model is wrong, and censoring every sample above ~1200 ADC as a lower bound improves det3 by −0.052 ± 0.011° (X)
and −0.037 ± 0.010° (Y)** on a second det3 sample of 4818 events, −0.10 / −0.07° on saturated planes, far fraction
−0.3 / −0.2 pp. The gain is there on det2 (−0.05 / −0.06) and det7 (X; Y far fraction +2 pp is the one cost) and
neutral on det4/det6 (within ±0.06, 1–1.5σ). It is a candidate for Dylan, **not adopted**: charge must be re-solved
uncapped (the capped fit reads 3–23 % low), and C was scanned on these rows.

Also found and fixed in the switches (opt-in): the 3550 rule misses 5–46 % of the truly clipped samples (the CM moves
the rail per sample; det4/6/7 worst). The per-sample rail (`SAT_RAIL`) removes 14–40 % of the χ² of clipped planes
and gains −0.08 ± 0.03° on det7 X (both halves of its sample), null elsewhere. The floor after a clip (raw 0) is the
ADC clipping a real undershoot: an upper bound (`SAT_FLOOR=1`) is the right treatment; it lowers χ² and is
angle-neutral. Down-weighting bright samples by their signal variance (the toy's estimator) is strongly **worse**
everywhere (+0.2…+0.8°, far ×2–4), as MODEL_FRAC was: what pins the track is the bright samples' *shape*, so they must
be censored where wrong, not de-weighted everywhere. A Huber loss, tail censoring, whole-strip censoring, unbounded
clips and a compression margin are all null or worse.

Saturated planes stay worse than unsaturated ones after the best treatment (det3 sat 1.06 vs all 1.02 X at cap 1200,
far 10 % on the 1320-event sample): that remainder is the deposits themselves (≥ 5 clipped strips: far 38–50 %),
which no censoring rule reaches.

## Clean-track check (2026-10-29 evening, Dylan: "clipping at 1500 costs nothing can't be right")

Dylan's two objections: is "clipping costs nothing" an artefact of saturation correlating with δ blobs / non-track
events, and doesn't the charge matter (x-y matching)? `clean_check.py`, det3, the 4818-event held-out sample, every
window clipped at C ADC (`--emulate-rail C` = `SAT_CAP`), event-paired against the true-clips-only fit. Event classes
from the RAW waveforms and M3 only (no fit): **clean** = one cluster, width ≤ expected + 3 strips, charge ≤ 2 × the
median at that angle, in both views (1312 events, far fraction 0.5 %); **blob** = a second cluster > 20 % of
the main charge, width > expected + 6 or charge > 3 × median, in either view (775 events, far 12 / 6 %); other
2731.

**Clean tracks:**

| clip level (every window) | angle Δs68 X | angle Δs68 Y | far X / Y % | charge ratio X / Y | x-y log-charge σ | x-y pairing power |
|---|---|---|---|---|---|---|
| none (true clips only) | +0.000 ± 0.000 | +0.000 ± 0.000 | 0.5 / 0.5 | 1.000 / 1.000 | 0.054 | 0.900 |
| 300 ADC | -0.085 ± 0.025 | -0.052 ± 0.023 | 0.4 / 0.4 | 0.907 / 0.916 | 0.173 | 0.786 |
| 600 ADC | -0.071 ± 0.021 | -0.039 ± 0.023 | 0.5 / 0.4 | 0.941 / 0.932 | 0.094 | 0.855 |
| 800 ADC | -0.066 ± 0.020 | -0.019 ± 0.020 | 0.5 / 0.4 | 0.956 / 0.944 | 0.072 | 0.878 |
| 1000 ADC | -0.061 ± 0.018 | -0.013 ± 0.016 | 0.5 / 0.4 | 0.971 / 0.957 | 0.062 | 0.889 |
| 1200 ADC | -0.057 ± 0.018 | -0.007 ± 0.016 | 0.5 / 0.4 | 0.982 / 0.971 | 0.056 | 0.891 |
| 1500 ADC | -0.030 ± 0.013 | -0.010 ± 0.013 | 0.4 / 0.5 | 0.995 / 0.988 | 0.055 | 0.896 |
| 2000 ADC | -0.025 ± 0.011 | +0.000 ± 0.011 | 0.4 / 0.5 | 1.000 / 1.000 | 0.053 | 0.899 |
| 2500 ADC | -0.012 ± 0.008 | -0.001 ± 0.009 | 0.4 / 0.5 | 1.000 / 1.000 | 0.053 | 0.898 |

**Blobs:**

| clip level (every window) | angle Δs68 X | angle Δs68 Y | far X / Y % | charge ratio X / Y | x-y log-charge σ | x-y pairing power |
|---|---|---|---|---|---|---|
| none (true clips only) | +0.000 ± 0.000 | +0.000 ± 0.000 | 12.4 / 6.3 | 1.000 / 1.000 | 0.093 | 0.899 |
| 300 ADC | +0.070 ± 0.053 | +0.440 ± 0.064 | 11.5 / 10.1 | 0.874 / 0.738 | 0.464 | 0.721 |
| 600 ADC | -0.020 ± 0.042 | +0.092 ± 0.047 | 11.1 / 7.0 | 0.909 / 0.828 | 0.284 | 0.778 |
| 800 ADC | -0.039 ± 0.041 | -0.026 ± 0.044 | 11.0 / 6.1 | 0.926 / 0.850 | 0.240 | 0.808 |
| 1000 ADC | -0.079 ± 0.042 | -0.039 ± 0.043 | 11.6 / 6.2 | 0.934 / 0.874 | 0.193 | 0.832 |
| 1200 ADC | -0.086 ± 0.037 | -0.077 ± 0.037 | 11.6 / 6.2 | 0.948 / 0.893 | 0.165 | 0.838 |
| 1500 ADC | -0.040 ± 0.035 | -0.070 ± 0.033 | 11.9 / 6.1 | 0.955 / 0.915 | 0.141 | 0.859 |
| 2000 ADC | -0.018 ± 0.033 | -0.053 ± 0.033 | 11.9 / 6.3 | 0.976 / 0.946 | 0.115 | 0.878 |
| 2500 ADC | -0.015 ± 0.022 | -0.025 ± 0.029 | 12.0 / 6.2 | 0.988 / 0.964 | 0.100 | 0.888 |

- **Not a blob artefact.** On clean single tracks the angle gets *better* the more of the amplitude is hidden, down to
  a 300 ADC clip (X −0.085 ± 0.025°, Y −0.052 ± 0.023°), where the fit sees little more than where and when the
  strips are above 300. Blobs are where the amplitude is needed (Y +0.44° at 300).
- **The charge does carry information, and the clip destroys it**: x-y log-charge σ 0.054 → 0.173 and pairing power
  0.90 → 0.79 on clean tracks at 300 ADC; the cap at 1200 already costs 0.054 → 0.056 (clean) and 0.093 → 0.165
  (blobs). Any cap must keep an uncapped charge solve.
- **What it means.** Amplitudes are physically informative: the electron-by-electron toy with an exact model reaches
  0.34° *using* them (`rerun/model_audit/ws_floor`). That hiding them helps here says the fit's **error model** is wrong
  for bright samples: it weights every sample by electronics noise alone (σ ~ 10 ADC), while the real sample-to-sample
  scatter of a bright pulse around any mean model is ~6 % of the peak (T16: 0.06 of the peak, plane to plane), i.e.
  50–200 ADC, and it is correlated within a strip (one cluster's charge shapes all its samples). So the brightest
  samples are trusted 10–20× too much and their fluctuations pull the line; the cap is a crude way of not trusting them.
- **Not yet tested, and the likely real fix:** (1) a model-based signal variance, σ² = noise² + k·M (Pearson form,
  iterated with the prediction M). The study's `SIG_VAR_K` used the DATA in place of the prediction (Neyman form), which
  is known to bias towards low-fluctuating samples; that may be why it came out much worse. (2) A per-strip amplitude
  nuisance (each strip's scale free within a prior ~ the cluster fluctuation), so the shape and timing of every strip
  are used but its absolute amplitude only loosely, with the total charge from the uncapped solve.

## Headline numbers (det3, 1320 held-out events, s68 [°] vs M3; far = |err| > 5°)

| det3 | planes | s68 X | s68 Y | far X | far Y | χ²/dof X / Y |
|---|---|---|---|---|---|---|
| unsaturated planes (no raw 4095 sample) | 889 / 885 | 1.029 | 0.926 | 1.0 % | 0.8 % | 32 / 34 |
| saturated planes | 431 / 435 | 1.213 | 1.246 | 10.2 % | 6.0 % | 158 / 231 |
| sat. planes, 1–2 clipped strips | 311 / 313 | 1.13 | 1.18 | 4.2 % | 2.2 % | |
| sat. planes, ≥ 5 clipped strips | 40 / 37 | 1.30 | 1.83 | 50 % | 38 % | ~1000 / ~2900 (far ones) |

**The decisive diagnostic (r4, `--emulate-rail`).** Planes that never saturated were clipped at an emulated rail
(1500 ADC: 54 % of det3 planes; 2500: 24 %) and fitted with the same censored treatment. If the clip, or the way the
fit treats it, cost resolution, these planes would degrade towards 1.2°. They do not; they get slightly *better*:

| emulated rail | chamber | planes clipped only by it | s68 X: real → clipped | s68 Y: real → clipped |
|---|---|---|---|---|
| 1500 | det3 | 54 % | 0.973 → 0.944 (−0.028 ± 0.023) | 0.948 → 0.917 (−0.031 ± 0.019) |
| 2500 | det3 | 24 % | 0.975 → 0.940 (−0.035 ± 0.028) | 0.978 → 0.975 (−0.003 ± 0.023) |
| 1500 | det2 | 42–44 % | 0.983 → 0.957 (−0.027 ± 0.028) | 1.422 → 1.325 (−0.098 ± 0.033) |
| 1500 | det7 | 15–18 % | 1.096 → 1.058 (−0.038 ± 0.080) | 1.402 → 1.450 (+0.048 ± 0.050) |

So **clipping per se costs nothing** with the present censored treatment, and the 0.2–0.3° excess and the ten-fold far
fraction of saturated planes belong to *which* planes saturate (large deposits: δ electrons, Landau tail, two
particles — the satblob finding) together with the model's error on the bright neighbours that then carry the fit, not
to the information the clip removes.

## The one lever that works: censor the bright samples (the cap)

What the diagnostic implies, tested directly: censor every sample at or above a fixed level C as a lower bound at C
(the fit only learns "at least C" there), on every plane, rails kept (`SAT_CAP = C` + `SAT_RAIL`; the harness
emulation `--emulate-rail C` is the same thing, checked identical on 80 det3 planes). These are the samples where the
template and footprint are most wrong in absolute ADC (T9: centre under-filled 5–11 %, the peak sample +3…+15 % in
data − model here). With noise-only weights they dominate χ² by the square of a model error, not of a measurement
error. At C = 1200 ADC the cap censors 15 % (det3), 19 % (det2), 17 % (det4), 28 % (det6), 35 % (det7) of the samples
above 5σ.

**det3, second and larger held-out sample** (`big_det3.py`: 4818 events 1500–6318 of an 8000-event rr1-reproducing
cache, disjoint from the first sample and from the base bundle's calibration cache; event-paired, 400 bootstraps):

| arm | X all | X sat | Y all | Y sat | far X / Y (base) | χ²/dof X / Y (base) | q X / Y |
|---|---|---|---|---|---|---|---|
| base | s68 1.072 (n 4818) | 1.162 (n 1528) | 1.022 | 1.203 (n 1490) | 3.1 / 1.6 | 49 / 55 | |
| rail | +0.000 ± 0.002 | +0.002 ± 0.007 | -0.000 ± 0.002 | +0.002 ± 0.006 | 3.1 / 1.5 (3.1 / 1.6) | 49 / 55 | 1.000 / 1.000 |
| railub | +0.000 ± 0.002 | +0.000 ± 0.007 | -0.003 ± 0.005 | -0.021 ± 0.016 | 3.1 / 1.6 (3.1 / 1.6) | 49 / 51 | 1.000 / 1.000 |
| cap2000 | -0.024 ± 0.007 | -0.065 ± 0.023 | -0.015 ± 0.007 | -0.023 ± 0.022 | 2.8 / 1.5 (3.1 / 1.6) | 39 / 44 | 0.997 / 0.988 |
| cap1500 | -0.035 ± 0.009 | -0.061 ± 0.027 | -0.027 ± 0.009 | -0.062 ± 0.024 | 2.8 / 1.4 (3.1 / 1.6) | 32 / 37 | 0.980 / 0.966 |
| cap1200 | -0.052 ± 0.011 | -0.098 ± 0.028 | -0.037 ± 0.010 | -0.073 ± 0.025 | 2.8 / 1.4 (3.1 / 1.6) | 26 / 31 | 0.967 / 0.949 |
| cap1200ub | -0.051 ± 0.011 | -0.100 ± 0.027 | -0.040 ± 0.010 | -0.080 ± 0.027 | 2.8 / 1.5 (3.1 / 1.6) | 26 / 28 | 0.967 / 0.968 |
| cap1000 | -0.049 ± 0.012 | -0.086 ± 0.028 | -0.046 ± 0.011 | -0.080 ± 0.026 | 2.8 / 1.4 (3.1 / 1.6) | 22 / 27 | 0.953 / 0.937 |
| cap800 | -0.048 ± 0.013 | -0.049 ± 0.028 | -0.053 ± 0.011 | -0.081 ± 0.030 | 2.7 / 1.4 (3.1 / 1.6) | 18 / 23 | 0.940 / 0.925 |

**First sample (1320 events), all five chambers** — X all | Y all (Δs68 [°], Δfar pp), `results/arms/cap_scan.json`:

| cap | det3 | det2 | det4 | det6 | det7 |
|---|---|---|---|---|---|
| 2000 | −0.033 ± 0.019 \| −0.018 ± 0.015 | −0.026 ± 0.018 \| −0.040 ± 0.025 | +0.005 ± 0.028 \| −0.025 ± 0.028 | +0.001 ± 0.030 \| −0.066 ± 0.035 | −0.115 ± 0.051 \| −0.061 ± 0.034 |
| 1500 | −0.041 ± 0.020 \| −0.012 ± 0.018 | −0.025 ± 0.023 \| −0.073 ± 0.025 | −0.012 ± 0.035 \| −0.012 ± 0.029 | +0.000 ± 0.033 \| −0.025 ± 0.047 | −0.099 ± 0.044 \| −0.029 ± 0.036 (+1.2) |
| 1200 | −0.047 ± 0.021 \| −0.006 ± 0.020 | −0.053 ± 0.025 \| −0.062 ± 0.028 | −0.061 ± 0.037 \| +0.015 ± 0.030 | +0.013 ± 0.034 \| −0.029 ± 0.044 | −0.126 ± 0.044 \| −0.023 ± 0.039 (+2.0) |
| 1000 | −0.019 ± 0.024 \| −0.011 ± 0.021 | −0.046 ± 0.026 \| −0.066 ± 0.029 | −0.044 ± 0.039 \| −0.020 ± 0.030 | +0.007 ± 0.034 \| −0.035 ± 0.047 | −0.138 ± 0.048 \| −0.033 ± 0.041 (+3.2) |
| 800 | −0.043 ± 0.025 \| −0.011 ± 0.025 | −0.056 ± 0.026 \| −0.086 ± 0.029 | −0.057 ± 0.042 \| −0.025 ± 0.033 | −0.004 ± 0.038 \| −0.067 ± 0.046 | −0.097 ± 0.049 \| −0.023 ± 0.041 (+5.5) |
| 1200 + floor ub | −0.047 ± 0.021 \| −0.002 ± 0.020 | −0.053 ± 0.025 \| −0.069 ± 0.028 | −0.055 ± 0.039 \| −0.007 ± 0.032 | +0.025 ± 0.035 \| −0.051 ± 0.045 | −0.144 ± 0.046 \| −0.026 ± 0.040 (+1.4) |

(det7's column contains its rail gain, −0.08 X; the cap adds −0.02…−0.06 on top. Far fractions move by < 1 pp except
det7 Y, which grows +1…+5.5 pp below C = 1500.)

Costs and caveats of the cap:
- **Charge reads low** (q ratio det3 0.97/0.95 at 1200; det6 0.77/0.81): the capped samples carry only "≥ C". The
  geometry can be fitted capped and the charge re-solved uncapped at the final (p0, w, t0) — a one-line follow-up
  solve, not implemented here (it belongs in `wft/reco.py`, which the bow session is editing). Until then a capped fit
  must not feed the x–y charge pairing or the head-on rescue's charge match.
- **χ²/dof falls** (det3 52 → 26 X at 1200) because the worst-modelled samples leave the sum. The relative χ² flag
  (5 × median) re-normalises itself; absolute χ² cuts would not.
- **det7 Y far fraction** grows at low caps (+2.0 pp at 1200, +1.2 at 1500, +0.3 at 2000): the brightest chamber loses
  a third of its signal samples at 1200. A cap scaled to the chamber's gain (det3-equivalent) is the obvious variant;
  not run.
- C is one ADC number for all chambers here. It was scanned, i.e. chosen on held-out data from the same rows. The
  det3 second sample confirms it there; the other chambers have one sample each.


## What a clipped DREAM pulse looks like (Q1) — `rails.py`, `char.py`, `resid.py`, `examples.py`

Rails rebuilt per sample from the raw decoded_root (`rails.py`: raw − ped − CM reproduces the cache W bit-exactly on
det2/3/6/7, ≤ 1.75 ADC on det4's older decode; consumers take the rails relative to W, so they are exact).

| chamber | planes with a raw-4095 sample | top rail in W p10/p50/p90 | clipped samples the 3550 rule fits as data | W ≥ 3550 censored but not clipped | strips → raw 0 after a clip | bright *unclipped* strips → raw 0 | planes with a chip-wide CM > 1000 |
|---|---|---|---|---|---|---|---|
| det3 | 32.8 % | 3684 / 3763 / 3830 | 467 of 9 955 (4.7 %) | 986 | 49 % | 27 % | 0.3 % |
| det2 | 49.2 % | 3575 / 3750 / 3882 | 1 711 of 17 598 (9.7 %) | 1 554 | 49 % | 25 % | 0.8 % |
| det4 | 28.6 % | 3281 / 3661 / 3969 | 6 762 of 18 309 (37 %) | 686 | 50 % | 25 % | 2.8 % |
| det6 | 82.5 % | 3312 / 3627 / 3916 | 20 583 of 44 934 (46 %) | 1 660 | 44 % | 20 % | 8.3 % |
| det7 | 77.2 % | 3386 / 3739 / 3977 | 17 093 of 74 500 (23 %) | 3 415 | 51 % | 25 % | 6.9 % |

(`results/char_<det>.json`; held-out events 180–1500.)

- **The clip is a hard ADC clip at raw 4095.** Raw strip maxima pile up at exactly 4095 (det3: 1 896 strips, against
  82 in 4000–4094 and 90 in 3900–3999). In W units the rail is per channel *and per sample* (4095 − ped − CM): its
  spread (p10–p90 ≈ 150 ADC on det3, 600 on det4/det6) is the common mode, so on det4/6/7 a quarter to half of the
  clipped samples sit below 3550 and are fitted as exact data. They carry 14 % (det3), 16 % (det2), 40 % (det4),
  26 % (det6), 34 % (det7) of the χ² of clipped planes in the base fit (`resid.py`, χ² shares).
- **No compression below the rail.** The peak residual of unclipped strips (data − model, / peak) is +0.043 / +0.047 /
  +0.048 on det3 for peaks 2000–3000 / 3000–3550 / 3550–4100 ADC (det2/4/6 alike); a soft saturation would turn it
  negative near the rail. (det7's excess falls with amplitude from 500 ADC up, 0.16 → 0.07: a shape trend, not a rail
  effect.) The `m150` arm (150 ADC margin below the rail censored) is null.
- **After the clip.** The strip falls fast (4–5 samples for plateaus ≥ 10 samples, 8 for 1-sample clips) and in about
  half the cases to raw 0, where it sits for the rest of the window (P(floor) 34 % for L = 1 → 69 % for L ≥ 10 on
  det3). This is *not* outside the linear model: aligned on the last clipped sample, the fit's own model (with the
  true, unclipped charge the lower bound forces) follows the fall to within ~±150 ADC and undershoots to −464 ADC
  where the data sit at the floor (−290): **the floor is the ADC clipping a real undershoot**, so the right bound
  there is model ≤ W (upper), not data and not model ≥ W. 20–27 % of bright *unclipped* strips reach the floor too.
- **Neighbours.** The ±1 strips of a clipped strip are over-predicted by up to 60 (det3) – 320 (det6) ADC before
  the clip ends and under-predicted after it; ±2 under-predicted by ~60 ADC: the same centre/±1/±2 footprint error as
  T9, now carrying the fit because the centre is censored. On clipped det3 planes, after rail + floor bound, the χ²
  is 35 % post-clip samples, 38 % neighbours, 20 % rising edge, 0.5 % plateau (from 8 %).
- **Coherent chip events** (64-channel block median > 1000 ADC: a whole chip clipped or at the floor) exist on det6
  (8.3 % of planes) and det7 (6.9 %); there the CM-subtracted W of every strip in the block is not a measurement.
  Not treated here (see limits).

Figures: `figures/resid_base_<det>.png` (stacks A/B/C), `figures/examples_base_det3.png` (displays).

## The candidates (Q2/Q3) — held-out, event-paired vs base (Δs68 [°] ± paired bootstrap; far Δ in pp)

Base = the planned pass (T15 base bundles, σ0 0.30, LM, t0 pinned), `heldout_arm.py --t0-fix --rails`, events
180–1500, lxplus condor. "sat" = the planes with a raw-clipped sample in the base. Every switch is opt-in in
`wft/model.py` (via `WFT_DIAG=model.<NAME>=<v>`); off = bit-identical (checked on 80 det3 planes against f5841f2).

### Censor rules and weights (r1–r3), det3 — Δs68 [°] ± paired bootstrap (Δfar pp)

| arm | what | X all | X sat | Y all | Y sat | χ²/dof sat X / Y (base 158 / 231) |
|---|---|---|---|---|---|---|
| `rail` | clip mask = raw 4095 (per-sample rail) | −0.001 ± 0.004 | +0.004 ± 0.020 | −0.001 ± 0.005 | −0.010 ± 0.017 | 159 / 231 |
| `floorub0` | floor (raw 0) = upper bound, 3550 rule | +0.003 ± 0.004 | +0.023 ± 0.012 | −0.008 ± 0.011 | −0.009 ± 0.032 | 159 / 181 |
| `railub` | rail + floor upper bound | −0.002 ± 0.004 | +0.001 ± 0.021 | −0.008 ± 0.011 | −0.009 ± 0.032 | 159 / 190 |
| `railfree` | rail + floor censored, no bound | −0.002 ± 0.006 | +0.020 ± 0.022 | −0.011 ± 0.011 | −0.014 ± 0.035 | 158 / 185 |
| `tail` | rail + post-clip tail free | +0.011 ± 0.022 | +0.056 ± 0.065 | +0.002 ± 0.016 | +0.083 ± 0.056 (+0.7) | 76 / 123 |
| `tailub` | tail + floor bound | +0.009 ± 0.021 | +0.065 ± 0.066 | +0.006 ± 0.015 | +0.095 ± 0.053 (+0.9) | 81 / 128 |
| `tailL4ub` | tail only for plateaus ≥ 4 + floor bound | +0.003 ± 0.017 | −0.010 ± 0.059 | +0.004 ± 0.012 | +0.050 ± 0.043 | 101 / 147 |
| `strip` | whole clipped strip censored | +0.011 ± 0.024 | +0.034 ± 0.076 | +0.022 ± 0.017 | +0.127 ± 0.064 (+0.9) | 46 / 78 |
| `m150` | 150 ADC below the rail censored | +0.001 ± 0.003 | +0.004 ± 0.012 | −0.001 ± 0.004 | −0.012 ± 0.015 | 158 / 231 |
| `clipfree` | rail + clipped samples unbounded | −0.015 ± 0.010 | −0.027 ± 0.036 | −0.006 ± 0.012 | −0.000 ± 0.029 | 147 / 214 |
| `clipfree_ub` | + floor bound | −0.014 ± 0.010 | −0.029 ± 0.037 | −0.003 ± 0.012 | −0.002 ± 0.034 | 148 / 173 |
| `sv` | σ² = noise² + K·W (K 11.5) | **+0.177 ± 0.040 (+5.0)** | **+0.596 ± 0.107 (+13.7)** | **+0.191 ± 0.038 (+8.3)** | **+0.735 ± 0.129 (+21.6)** | 19 / 166 |
| `sv_half` | K/2 | +0.098 ± 0.035 (+3.7) | +0.379 ± 0.104 (+10.7) | +0.152 ± 0.039 (+6.0) | +0.654 ± 0.134 (+16.6) | 23 / 155 |
| `rob10` | Huber loss, C = 10 σ | −0.014 ± 0.024 | +0.052 ± 0.051 | −0.004 ± 0.019 | +0.008 ± 0.049 (+1.4) | |
| `rob25` | C = 25 σ | +0.008 ± 0.020 | +0.012 ± 0.044 | −0.003 ± 0.016 | −0.006 ± 0.042 (+1.4) | |
| `rob50` | C = 50 σ | −0.015 ± 0.013 | −0.011 ± 0.035 | −0.017 ± 0.013 | −0.005 ± 0.034 | |

T15 shape arms on det3 saturated planes (reanalysed here; T15 never split them): core fraction 0.2/0.4, Dp ×0.85/×1.15,
template stretch 0.84–1.04 — all within ±0.06 (best stretch 0.84 X −0.094 ± 0.056, Y +0.014 ± 0.048).

### The same, other chambers — X all | Y all (Δs68 [°]); saturated planes in the JSON (`results/arms/arms_<det>.json`)

| arm | det2 | det4 | det6 | det7 |
|---|---|---|---|---|
| `rail` | +0.004 ± 0.009 \| −0.010 ± 0.010 | +0.009 ± 0.020 \| −0.013 ± 0.019 | −0.005 ± 0.019 \| −0.017 ± 0.021 | **−0.080 ± 0.028 \| −0.019 ± 0.025** |
| `railub` | +0.004 ± 0.009 \| −0.010 ± 0.015 | +0.010 ± 0.020 \| −0.009 ± 0.023 | −0.004 ± 0.019 \| −0.003 ± 0.024 | **−0.068 ± 0.028 \| −0.065 ± 0.032** |
| `railfree` | +0.001 ± 0.009 \| −0.006 ± 0.015 | −0.002 ± 0.024 \| −0.020 ± 0.023 | +0.002 ± 0.020 \| −0.027 ± 0.029 | −0.083 ± 0.029 \| −0.062 ± 0.031 |
| `tail` | +0.023 ± 0.023 \| +0.017 ± 0.024 | +0.003 ± 0.034 \| −0.036 ± 0.026 | +0.089 ± 0.039 \| +0.030 ± 0.038 | −0.077 ± 0.055 \| −0.044 ± 0.041 |
| `strip` | +0.079 ± 0.032 \| +0.025 ± 0.024 | +0.055 ± 0.044 \| +0.026 ± 0.028 | +0.236 ± 0.058 \| +0.036 ± 0.044 | +0.176 ± 0.078 \| +0.044 ± 0.041 |
| `clipfree_ub` | −0.006 ± 0.016 \| −0.011 ± 0.018 | +0.014 ± 0.029 \| −0.012 ± 0.027 | +0.009 ± 0.026 \| −0.006 ± 0.040 | −0.068 ± 0.039 \| −0.122 ± 0.036 (far +2.3) |
| `sv` | +0.256 ± 0.037 \| +0.293 ± 0.065 | +0.009 ± 0.056 \| −0.034 ± 0.048 | +0.379 ± 0.064 \| +0.442 ± 0.061 | +0.217 ± 0.055 \| +0.475 ± 0.071 |
| `rob25` | −0.006 ± 0.019 \| −0.019 ± 0.019 | +0.005 ± 0.029 \| −0.015 ± 0.026 | +0.023 ± 0.024 \| −0.000 ± 0.035 | +0.021 ± 0.039 \| −0.013 ± 0.025 |

det7's rail gain holds in both halves of its held-out sample (rail X −0.054 / −0.071, railub Y −0.081 / −0.098).

### The cap (r5)

See the cap section above (`results/arms/cap_scan.json`, `results/arms/big_det3.json`).


## What it does not rule out

- **One dataset per chamber.** The harness has one 1500-event rr1 cache per chamber (golden row). The det7 rail gain
  holds in both halves of the held-out sample (−0.04…−0.10 in each half and view), but it is not checked on a second
  run, which Dylan's rule asks for. The held-out samples are small (s68 errors ±0.01–0.05° all planes, ±0.02–0.10° sat
  planes): effects below ~0.03° on det3 are not resolved.
- **The big-deposit planes are not fixed.** Planes with ≥ 3 clipped strips (det3: 9 %) carry most of the far
  fraction. Nothing here (censoring rules, Huber loss, unbounded clips) moves it; satblob showed a compact blob term
  does not either. A δ-track component in the model is the untested direction.
- **The neighbour mismatch** that dominates clipped-plane χ² is the T9/T16 footprint error; T15's shape arms (core
  fraction, Dp, stretch) do not help saturated planes either (det3 sat planes: all within ±0.06, best stretch 0.84 X
  −0.09 ± 0.06).
- **Coherent chip events** (det6/det7 7–8 % of planes) are not censored. A rule (a block whose CM exceeds ~1000 →
  censor the block's samples) is easy with the rails sidecar but was not run.
- **The toy floor** (0.30–0.35°) includes an ADC ceiling; it is not split by deposit size, so how much of the
  saturated-plane excess is physics (δ rays) is not quantified.
- **The cap was tested in the held-out harness only** (`wft.reco.fit_plane`, pivot start, t0 pinned, M3-corridor
  windows). Seeding, candidate selection, the two-track fit and the head-on rescue read the charge or χ² of a fit; a
  full reco on a golden row (efficiency, multi-track, pairing) with `SAT_CAP` is the next check before any adoption.
  A cap scaled to each chamber's gain was not run.
- **Test-order note.** `wft/tests/test_lm_fitter.py` fails when run after `test_two_track.py` in one pytest session
  (state left behind); the same happens on the pre-study code (f5841f2), and both pass alone. Not caused by this work.
- **Production wiring.** The rails exist only in the study sidecar (`rails.py`); `wft.io.FeuReader` would have to
  carry ped + CM (or raw) into the windows for SAT_RAIL / SAT_FLOOR to act in the pass.

## Files

| what | where |
|---|---|
| rails sidecar builder (raw → per-sample top/bottom rail) | `rails.py` → `$CLAUDE_JOB_DIR/tmp/satur/rails_<det>.pkl`; slim (raw only) on EOS `/eos/user/d/dneff/x17rerun/satur/rails/` |
| raw characterisation | `char.py` → `results/char_<det>.json` |
| fit residual stacks, χ² shares | `resid.py run/stack` → `results/resid_<tag>.json`, `figures/resid_<tag>.png` |
| displays | `examples.py` → `figures/examples_base_det3.png` |
| arms, DAGs, comparison | `sat_arms.py build/compare/summary/emu`, `cap_table.py`, `big_det3.py` (second det3 sample; EOS `satur/big/`) → `results/arms/arms_<det>.json`, `results/arms/emu.json`; outputs EOS `/eos/user/d/dneff/x17rerun/satur/r1…r5/` |
| model switches | `wft/model.py` SAT_CAP, SAT_RAIL, SAT_MARGIN, SAT_FLOOR (1 upper bound, 2 free), SAT_TAIL (+_SKIP, _MINLEN), SAT_STRIP, SAT_CLIP_FREE, SIG_VAR_K, ROBUST_C (+_ITERS); `Censor` mask (lower / upper / free bounds); test `wft/tests/test_sat_switches.py` |
| harness | `rerun/model_audit/heldout_arm.py --rails R.pkl [--emulate-rail C]` (two/three columns appended to `per`) |
