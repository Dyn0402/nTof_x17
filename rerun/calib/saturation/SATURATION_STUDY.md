# Saturation (ADC-clipped pulses) in the wft reconstruction — a critical study (2026-10-29)

Dylan: "we are treating it as well as we can" is not an answer; find real improvements or show with numbers that there
are none. Held-out angle s68 vs M3, event-paired against the planned-pass base, det3 first, then det2/4/6/7.

## Verdict

@@VERDICT@@

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

@@CAP@@

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

@@TABLES@@

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
- **Production wiring.** The rails exist only in the study sidecar (`rails.py`); `wft.io.FeuReader` would have to
  carry ped + CM (or raw) into the windows for SAT_RAIL / SAT_FLOOR to act in the pass.

## Files

| what | where |
|---|---|
| rails sidecar builder (raw → per-sample top/bottom rail) | `rails.py` → `$CLAUDE_JOB_DIR/tmp/satur/rails_<det>.pkl`; slim (raw only) on EOS `/eos/user/d/dneff/x17rerun/satur/rails/` |
| raw characterisation | `char.py` → `results/char_<det>.json` |
| fit residual stacks, χ² shares | `resid.py run/stack` → `results/resid_<tag>.json`, `figures/resid_<tag>.png` |
| displays | `examples.py` → `figures/examples_base_det3.png` |
| arms, DAGs, comparison | `sat_arms.py build/compare/summary/emu` → `results/arms/arms_<det>.json`, `results/arms/emu.json`; outputs EOS `/eos/user/d/dneff/x17rerun/satur/r1…r5/` |
| model switches | `wft/model.py` SAT_RAIL, SAT_MARGIN, SAT_FLOOR (1 upper bound, 2 free), SAT_TAIL (+_SKIP, _MINLEN), SAT_STRIP, SAT_CLIP_FREE, SIG_VAR_K, ROBUST_C (+_ITERS); `Censor` mask (lower / upper / free bounds); test `wft/tests/test_sat_switches.py` |
| harness | `rerun/model_audit/heldout_arm.py --rails R.pkl [--emulate-rail C]` (two/three columns appended to `per`) |
