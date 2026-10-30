# S4: the physics inputs re-measured on pass 1 (2026-10-30)

PASS_PLAN §0: the physics inputs in `physics_inputs.json` are measured outside the reconstruction. S4 re-measures them on
the pass. If one moves by more than its error, that chamber's calibration phase is re-run and the cost is written down.
**Nothing was re-run here.** This file is the measurement and the cost. The decision is Dylan's.

## Verdict

**det2 and det3 hold, apart from det3's gap at +2σ. Two inputs move beyond their errors in a way the data support:
det4's timing v (−5 %, −2.4σ) and det7's Dp (−17 %, −2.0σ).** Each would mean re-running that chamber's calibration.
The other flags are the estimator failing on that row, not a moved input:
- **det4 gap +2.6 mm:** the two tan halves disagree by 3.7 mm (0.3 mm before);
- **det6 gap and Dp:** the 32-sample window cuts them, as gaps.py already says;
- **det3 gap:** +0.38 mm, which is 2.0σ of its stated error but 1.4σ combined.

- **det3, the reference chamber, is unchanged:** v 35.75 against 35.8 (−0.08σ), Dp 0.0230 against 0.023. Its gap reads
  28.46 ± 0.20 against 28.08 ± 0.19 (+0.38 mm). That is beyond the stated error (2.0σ) but only 1.4σ once both errors are
  combined.
- **det2:** every input is inside its error (v −0.5σ, Dp +0.1σ, gap +0.6σ).
- **det4 v: 31.3 ± 0.4 against 33.0 ± 0.7 µm/ns (−2.4σ).**
  - The stack T rose from 924 to 973 ns. The new stack has 1799 head-on events against 476 before.
  - The pass's own ladder (angle-implied v) reads 30.8 µm/ns (X 30.3, Y 31.3), so the new timing v and the ladder now agree
    to 2 %. Against the old input they were 7 % apart.
  - Dp holds: 0.0195 against 0.020.
- **det7 Dp: 0.0187 ± 0.0013 against 0.0225 ± 0.0019 (−2.0σ).**
  - v (36.83 against 36.8) and gap (+0.8σ) hold.
  - det7's Dp was already sensitive to the saturation cut (0.018–0.024 across `unsat_width.json`'s @sat variants), and the
    new stack has 554 events against 144.
- **det6:** v 23.2 ± 0.4 against 24.17 ± 0.75 (−1.2σ), inside 2σ.
  - Its Dp (0.0085) and gap (32.7, mesh edge at z = +8.5 mm) are window-limited, as before (`gaps.py`: "det6 cannot be
    measured this way"). Neither is a measurement.
  - The adopted 28.0 mm and Dp 0.018 stand on the estimates they already rested on.
- **σ0 and det4 D_rc,y are not re-measured.** Both are SPS / H4 inputs.
  - The bench stack's free σ0 (0.29–0.35 mm) is listed only for information. Bench free σ0 is not a measurement (FC).

## What was measured

The inputs' own estimators, unchanged, run on the five golden rows with the pass's inputs in place of the pre-pass ones:
- **the reference:** M3 rays through the pass's waveform alignment (`Analysis_pass1/<row>/wft/alignment/alignment.json`), not the hits-chain
  `alignment_tpc_veto50` (memory: that one biases v by 3–15 %);
- **the waveforms:** the rr1 decode the pass read;
- **the template:** the pass bundle's `tmpl_x`.

| input | estimator (source of the input) | S4 tool |
|---|---|---|
| timing v | gap / T, where T is the free column duration of the unsaturated head-on X stack (`unsat_width.py` → `x_width.measure`, < 3500 ADC, \|tan_x\| < 0.03) | `rerun/calib/s4/s4_stacks.py` |
| Dp | √(2 Dd) from the same stack fit | same |
| gap | `gap_extent.py`: half-height edges of the inclined-track charge against z, X view, mean of the default and `--satmask` selections, half their difference in quadrature (`gaps.py` §39) | same |
| cross-checks | the pass ladder (Huber slope of w on M3 tan, `vgas.ladder`, on `angles/per_event.npz`); FA's model-free column duration `colsum.py` with the pass bundle | `s4_inputs.py`, `s4_row.sh` |

Errors:
- new values: bootstrap (20 resamples);
- the input's v error: `vladder/ladder_summary.py` TIMING (det6 scaled ×28/30);
- Dp error: the bootstrap error of the stack the input came from (`unsat_width.json`, 'unsat');
- gap error: `physics_inputs.json` (det6 1.0).

The pull is (new − input) / the input's stated error, the PASS_PLAN test. "comb." also folds in the new error.

## Per chamber

| chamber | input | old ± err | new ± err (n) | pull | comb. |
|---|---|---|---|---|---|
| det2 | v (µm/ns) | 39.3 ± 1.2 | 38.72 ± 0.61 (248) | −0.5 | |
| | Dp (mm/√ns) | 0.025 ± 0.0027 | 0.0253 ± 0.0010 | +0.1 | |
| | gap (mm) | 31.12 ± 0.42 | 31.36 ± 0.32 (232) | +0.6 | +0.4 |
| det3 | v | 35.8 ± 0.6 | 35.75 ± 0.29 (654) | −0.1 | |
| | Dp | 0.023 ± 0.0007 | 0.0230 ± 0.0005 | 0.0 | |
| | gap | 28.08 ± 0.19 | 28.46 ± 0.20 (913) | **+2.0** | +1.4 |
| det4 | v | 33.0 ± 0.7 | **31.34 ± 0.38** (1799) | **−2.4** | −2.1 |
| | Dp | 0.020 ± 0.0011 | 0.0195 ± 0.0007 | −0.4 | |
| | gap | 30.48 ± 0.44 | 33.13 ± 1.85 (1315) | (+6.0) | +1.4 (estimator unstable, below) |
| det6 | v | 24.17 ± 0.75 | 23.25 ± 0.44 (327) | −1.2 | |
| | Dp | 0.018 (own stack 0.0142 ± 0.0044) | 0.0085 ± 0.0025 | (−2.2) | window-limited, not a measurement |
| | gap | 28.0 ± 1.0 | 32.7 ± 2.7 (195) | (+4.7) | window-limited, not a measurement |
| det7 | v | 36.8 ± 1.1 | 36.83 ± 0.55 (554) | 0.0 | |
| | Dp | 0.0225 ± 0.0019 | **0.0187 ± 0.0013** | **−2.0** | −1.7 |
| | gap | 29.44 ± 1.12 | 30.36 ± 1.27 (343) | +0.8 | +0.5 |

v is gap_input / T_stack. The stack T values are 804 / 785 / 973 / 1204 / 799 ns (det2/3/4/6/7). Before they were
793 / 784 / 924 / 1160 / 800.

**Cross-checks:**
- **The pass ladder** (X, Y, µm/ns): det2 38.9 / 39.2, det3 37.0 / 36.4, det4 30.3 / 31.3, det6 24.5 / 24.8, det7 36.0 / 36.4.
  The ladder × stack T (C19 route) gives gaps of 31.4 / 28.8 / 29.9 / 29.7 / 29.0 mm.
- **FA colsum T with the pass bundle:** 829 / 804 / 743 / 1157 / 853 ns.
  - With the audit bundle (FA, 10-27) it was 831 / 802 / 706 / 1149 / 844, so the pass template moves colsum's T by
    ≤ 1 % except on det4 (+5 %).
  - Colsum's absolute T is not the inputs' estimator: det4 reads 743 against the stack's 973. It is used only as a relative
    check, as FA did.

## The gap estimator on det4 with the pass reference

The ± tan halves give 33.8 and 30.0 mm. The pre-pass gap_extent gave 30.9 and 30.6 mm on the hits-chain reference.

A tilt between the drift field and the M3 frame splits the halves symmetrically and cancels in their mean (the docstring of
`gap_extent.py`). This split is 3.7 mm, its mean is 31.9, and the mesh edge sits at z = −2.9 mm. So the pass alignment of
g_det4 does not give `gap_extent` a clean reference.

The −tan half agrees with the input, and the C19 route gives 29.9. **I read this as "det4's gap is not re-measured", not as
a +2.6 mm move.** det3's halves split 0.7 mm (28.6 / 27.9), close to before (28.5 / 27.6).

det4's pass alignment would need its own look before this estimator can use it. That is the same chamber as step 1's det4 Y
question, but this is the X view.

## Cost of re-running a chamber's calibration (PASS_PLAN §0, §3)

A chamber's calibration phase is the `calib` DAG for that chamber (~20 nodes, three reco iterations) and its G1 line.
Everything built on its golden `calib_bundle` follows:
- vrefit of its own-v rows;
- the pass of every row;
- post (01–04);
- the bow gate;
- the ledger.

From the pass's own timings (S1 4–6 h for all five chambers in parallel; vrefit ~1 h per 56 nodes; pass overnight for 156 rows):
- **det4 (v):**
  - calib ~4–6 h, then 17 bench rows, about half a night;
  - **plus the 34 SPS rows that have no v_span.** They take the bench det4 bundle's v (`sps_reco.py`), and they are already
    flagged as uncalibrated (review P3). The 37 rows with a v_span do not depend on it.
  - What changes: v sets the reconstruction's windows (`u_window`, the lever arm, the start w, K), not the angles (kw·v is
    re-measured against M3 per row). So the expected effect is on window-edge losses and far fractions, not on the angle
    scale.
- **det7 (Dp):** calib ~4–6 h, then 25 bench rows. Dp is in the footprint σ(u) of every fit, so χ², resolution and
  detection can all move. Its size is unknown until it is run.
- **det3 (gap +2σ, if acted on):** the gap enters only through v = gap / T, and the timing v is unchanged (−0.1σ). A
  re-run would change nothing at first order. Writing it down is enough.

## What this does not rule out

- **What moved det4's stack T (+5 %) is not separated.** The candidates are the reference (the pass alignment selects 4×
  more head-on events), the template and the decode.
  - An arm with the old template on the new cache would split them. It was not run.
  - The agreement with the pass ladder (30.8) is the evidence that the new value is the better one.
- **det7's Dp depends on the saturation cut** (0.018–0.024 across @sat variants before). One cut (3500 ADC, the
  estimator's default) is quoted, and no sat scan was run on the new stack.
- **The gap is used only through v = gap / T for the angle-free windows.** The ±2σ gap movements do not reach the
  angles, which are scaled by kw against M3 per row.
- **σ0 (0.30 mm, SPS FC) and det4 D_rc,y (H4)** are beam measurements. The pass's bench products do not re-measure them.
  The SPS run71 cache fix (`a72b889`) postdates FC; whether FC used RAW run71 was not checked.
- **v inside a row is not addressed here.** The det6 golden row sits in the 6-26 night, where v drifts ~10 % (FA). One T per row is
  quoted, as the input was.
- **The five stack rules** (no pulse alignment; RAW drops = NaN; ZS additive; smooth Magboltz η; single-field degenerate)
  are those of the unchanged cloud_basics code. No new stack was written.

## Files

- Tools: `rerun/calib/s4/`
  - `s4_stacks.py`: the stacks on the pass reference/template; condor `s4_stack.sh`/`.sub`, `s4_stack_jobs.in`;
  - `s4_row.sh`/`.sub`, `s4_jobs.txt`: colsum with the pass bundle;
  - `s4_inputs.py`: the comparison;
  - `results/s4_inputs.json`: every number above.
- Products: `/eos/user/d/dneff/x17rerun/s4_inputs/Analysis/<row>/{s4/{headon,incl}.pkl, s4/s4_stacks.json, fa/colsum.npz}`
  (340 MB). Code `/eos/user/d/dneff/x17rerun/s4_inputs/code_75287c0`. Condor 4599742 (colsum) and 4599748 (stacks), lxplus
  `~/x17rerun/s4_inputs/`.
- Analysis_pass1 was only read.

## Follow-up (2026-10-30 evening): what moved det4's v and det7's Dp

Dylan asked for two checks before any re-calibration: what moved det4's v, and a saturation-cut scan for det7's Dp.

### Verdict

**Neither move is a changed physics input. Neither chamber needs re-calibrating.**

- **det4 v:** the −1.66 µm/ns splits into two parts.
  - **The pass template, −0.7 µm/ns.** It is real: the template is the better electronics response (DET4Y_ABLATION).
  - **The event sample, −0.9 µm/ns.** S4's sample is dominated by low-amplitude events, and the stack's T depends on
    amplitude.
  - At matched amplitude (1000–3500 ADC), with the pass template, **v = 32.36 ± 0.50 µm/ns**, which is −0.9σ from the input
    33.0 ± 0.7. The decode contributes nothing and the alignment nothing.
- **det7 Dp:** the move is the X region with the readout faults (ch < 256) entering S4's sample.
  - On ch 256–511, the only part of det7 X that is quoted, **Dp = 0.0214 ± 0.0023**, which is −0.6σ from 0.0225 ± 0.0019.
  - It holds across saturation cuts of 2000–3500 ADC (0.019–0.021).
  - The cut scan on S4's full sample does not rescue the input: every cut ≤ 3500 ADC gives 0.010–0.019. So the cut was
    never the cause; the region was.

### How it was split

S4 changed three things at once.
- **The cache:** S4 used every M3 ray through the pass alignment ("ref-only"). The input used the hits chain's events
  (has_x and has_y, radial residual < 10 mm).
- **The decode:** rr1 against the pre-rerun `decoded_root`.
- **The template:** the pass bundle against `calib_bundle_lp` (det4) / `calib_bundle_r06` (det7).

`s4f_build.py` rebuilt the ref-only cache on the pre-rerun decode (`--gen prod`) and wrote both chains' hits-chain event
lists. `s4f_fit.py` fits the S4 stack (x_width.measure, |tan_x| < 0.03, bootstrap 20) on every combination, plus
amplitude bins on the X window maximum and the det7 channel restriction. The pass alignment and the hits alignment of
g_det4 agree to 0.05° and 1 mm, and both chains' hits selections are the same events (2740 det4, 3166 det7).

### det4: stack T (ns), gap 30.48 mm

| decode | selection | old template | pass template |
|---|---|---|---|
| prod | hits chain (n 473) | **924.1 ± 13.9** (the input: 924.5) | 944.8 ± 14.3 |
| rr1 | hits chain (n 476) | 925.2 ± 12.7 | 945.6 ± 14.8 |
| prod | ref-only (n 1794) | 950.7 ± 13.8 | 972.0 ± 13.6 |
| rr1 | ref-only (n 1799) | 951.4 ± 11.6 | **972.6 ± 11.9** (S4) |
| rr1 | ref-only, 1000–3500 ADC (n 266) | 919.0 ± 13.3 | **941.8 ± 14.6** |
| rr1 | hits chain, 1000–3500 ADC (n 215) | 922.2 ± 19.4 | 942.6 ± 18.8 |

- **Closure:** the prod / hits / old arm gives 924.1 against the input's 924.5.
- **Decode:** ≤ 1.1 ns.
- **Template:** +20 to +23 ns in every row, which is −0.7 µm/ns.
- **Selection:** +27 ns (−0.9 µm/ns) at all amplitudes, but **−0.8 ns at matched amplitude**.
  - The stack's T depends on amplitude: 1002 ± 17 ns below 1000 ADC against 942 ± 15 above (ref-only, pass template).
  - 45 % of the hits-chain head-on events are ≥ 1000 ADC (215/476) against 15 % of the ref-only ones (266/1799). The ref-only
    sample adds ~1300 events that the hits chain had dropped, almost all below 1000 ADC.
  - The amplitude bins are in `results/s4f/amp_det4.json`.
- **Dp:** 0.0212 ± 0.0011 at matched amplitude (+1.1σ against 0.020). Below 1000 ADC it reads 0.0145.

**Which v to believe:** 32.36 ± 0.50 µm/ns, from the pass template and events ≥ 1000 ADC.
- The low-amplitude stacks fit a smaller Dp and a larger σ0, and their extra length is most likely the stack estimator on
  weak signals rather than a slower drift. That is an inference: the data cannot separate the two.
- S4's agreement with the pass ladder (30.8) is therefore no longer evidence for 31.3. The ladder's v is angle-implied and
  carries kw, so it is not a timing measurement.

### det7: Dp (mm/√ns)

| sample | old template | pass template |
|---|---|---|
| hits chain, < 3500 ADC (n 144) | 0.0229 ± 0.0020 | **0.0225 ± 0.0019** (the input exactly) |
| ref-only, < 3500 ADC (n 554) | 0.0191 ± 0.0014 | **0.0187 ± 0.0013** (S4) |
| ref-only, ch 256–511, < 3500 ADC (n 170) | 0.0220 ± 0.0023 | **0.0214 ± 0.0023** |
| ref-only, ch 256–511, 1000–3500 ADC (n 88) | 0.0232 ± 0.0024 | 0.0224 ± 0.0025 |
| hits chain, ch 256–511 (n 67) | 0.0218 ± 0.0029 | 0.0213 ± 0.0029 |
| not hits chain, ch 256–511 (n 103) | 0.0222 ± 0.0034 | 0.0216 ± 0.0034 |

- **Decode:** the prod and rr1 caches give identical fits for this row.
- **Template:** ≤ 0.0006.
- **Where the move comes from:** the events S4 adds sit on the bad X region. 73 % of the non-hits-chain events at
  1000–3500 ADC peak on ch < 256, against 48 % of the hits-chain ones. Those stacks fit σ0 ≈ 0.2 mm and Dp 0.012–0.015.
  On ch 256–511, the hits-chain and added events agree (0.0213 against 0.0216).

**Saturation-cut scan (X window maximum, ADC):**

| cut | S4 sample (n) | S4 sample, ch 256–511 (n) | input sample (n) |
|---|---|---|---|
| 1000 | degenerate (332) | | < 120 events |
| 1500 | 0.0098 ± 0.0031 (390) | | < 120 |
| 2000 | 0.0153 ± 0.0033 (432) | 0.0190 ± 0.0024 (123) | < 120 (old cache: 0.024) |
| 2500 | 0.0159 ± 0.0022 (470) | 0.0194 ± 0.0027 (138) | < 120 (old cache: 0.024) |
| 3000 | 0.0186 ± 0.0020 (515) | 0.0203 ± 0.0022 (158) | 0.0227 ± 0.0027 (123) |
| 3500 | 0.0187 ± 0.0013 (554) | 0.0214 ± 0.0023 (170) | 0.0229 ± 0.0020 (144) |
| 4000 | 0.0273 (1252, saturated events enter) | | 0.0281 (680) |
| none | 0.0358 (1672) | | 0.0294 (821) |

- Above 3500 ADC, saturated events enter and Dp climbs in every sample. 3500 stays the right ceiling.
- Below it, the full S4 sample falls with the cut because the faulty-region and low-amplitude events make up more of it.

### Updated recommendation

- **det4: do not re-calibrate.**
  - The timing v at matched amplitude is 32.4 ± 0.5 against 33.0 ± 0.7 (−0.9σ), inside the PASS_PLAN test.
  - The remaining shift is the template, which the pass already uses.
  - v sets only the reconstruction windows, since the angles are scaled against M3 per row.
  - Optionally, record 32.4 ± 0.5 (pass template, ≥ 1000 ADC) in `physics_inputs.json` for bookkeeping. No re-run follows from
    it.
  - A re-run would cost 4–6 h of calibration, then 17 bench rows and the 34 SPS rows that borrow det4's bench v.
- **det7: do not re-calibrate.**
  - On the quoted X region, Dp 0.0214 ± 0.0023 is −0.6σ from the input.
  - A re-run would cost 4–6 h of calibration, then 25 bench rows.
- **The S4 estimator itself needs two changes before it is used again:**
  - apply each chamber's documented masks (det7 X ch 256–511);
  - quote the head-on stack at an amplitude floor (≥ 1000 ADC), or report T and Dp per amplitude bin.
  - As run, its sample is set by whatever the ref-only reference admits, and both T and Dp depend on that mix.

### What this does not rule out

- **Not re-checked with the floor and masks:** det2, det3 and det6. Their S4 values held, but they were measured the
  same way. det3 is the reference and has no documented X fault; det6's stack is window-limited anyway.
- **Why T rises at low amplitude is not separated.** It could be the estimator on weak signals or a real drift difference
  in the weak-gain regions (det4's dead stripes). Both read the same in the stack.
- **The matched-amplitude det4 number rests on 266 events.** Its 0.50 µm/ns error is bootstrap only.
- **det7's ch 256–511 numbers rest on 67–170 events.** The 120-event floor was lowered to 50 for the matched subsets.

### Files

- Tools:
  - `rerun/calib/s4/s4f_build.py` (+ `s4f_build.sh` / `.sub`): the head-on cache on a chosen decode, plus the hits-chain
    event lists;
  - `s4f_fit.py`: the arm grid;
  - `s4f_specs.py`: every grid above.
- Results: `rerun/calib/s4/results/s4f/*.json`.
- Products:
  - `/eos/user/d/dneff/x17rerun/s4_followup/Analysis_prod/<row>/s4f/{headon.pkl, selections.json}`;
  - the fit inputs and results are in `/eos/user/d/dneff/x17rerun/s4_followup/fits/`.
- Code: `code_16587ac6`. Condor 4599810, lxplus `~/x17rerun/s4_followup/`. The fits ran locally (seconds each).
- Analysis_pass1, the audit trees and `physics_inputs.json` were not modified.

## Re-check (2026-10-30 night): det2, det3, det6 with masks — and what the "low-amplitude" T really was

Dylan asked for det2, det3 and det6 to be re-measured the way the follow-up said the estimator must be used: masked, and
at an amplitude floor.

### Verdict

**All three inputs hold, and so does det4's.** Nothing needs re-calibrating, and `physics_inputs.json` is unchanged.

**The amplitude floor was the wrong fix.** The longer T of the low-amplitude events is not amplitude, and not the
estimator on weak signals. It is two things the low-amplitude sample over-represents:
- **unmasked dead strips:** det4 has 77 dead X channels;
- **the outer 30 mm of the active area:** the stack's T differs there by −90 to +60 ns depending on the chamber, and det6's
  edge fits do not constrain T at all.

With the mask applied and the reference ≥ 30 mm inside the active area, T is flat in amplitude:
- det4: 917 ± 34 ns below 1000 ADC, 926 ± 18 above;
- det4 over the whole area, masked: 935 ± 13 against 947 ± 14.

**An injection test rules out the estimator itself.** The same 2000–3500 ADC events, scaled to ~800 ADC with their own
noise added back, change T by −5 ns (det3) and −8 ns (det4), not +60–90 ns.

**The estimator to reuse is therefore: the bundle's dead list plus strip_gain_audit's flags masked, the M3 reference
≥ 30 mm inside the active area, the pass template, every unsaturated amplitude.** That is the "interior ≥ 30 mm" region
Paper I quotes anyway (PASS_PLAN §4). The ≥ 1000 ADC floor is only a cross-check. On top of the mask and the interior
cut it moves T by 1 ns (det2, det3), 3 ns (det6) and 6 ns (det4).

### Per chamber (golden rows; masked, interior ≥ 30 mm, pass template)

| chamber | input | input ± err | S4 as run | re-check (n) | pull |
|---|---|---|---|---|---|
| det2 | v (µm/ns) | 39.3 ± 1.2 | 38.72 | **38.25 ± 0.60** (158) | −0.9 |
| | Dp (mm/√ns) | 0.025 ± 0.0027 | 0.0253 | **0.0241 ± 0.0011** | −0.3 |
| | gap (mm) | 31.12 ± 0.42 | 31.36 | **31.55 ± 0.30** (210) | +1.0 |
| det3 | v | 35.8 ± 0.6 | 35.75 | **36.03 ± 0.34** (519) | +0.4 |
| | Dp | 0.023 ± 0.0007 | 0.0230 | **0.0228 ± 0.0006** | −0.3 |
| | gap | 28.08 ± 0.19 | 28.46 (+2.0σ) | **28.19 ± 0.19** (848) | **+0.6** |
| det6 | v | 24.17 ± 0.75 | 23.25 (−1.2σ) | **24.78 ± 0.78** (131) | +0.8 |
| | Dp | 0.018 | 0.0085 | 0.0109 ± 0.0026 | still window-limited, not a measurement |
| | gap | 28.0 ± 1.0 | 32.7 | 24.8–38.2 depending on the selection | still not a measurement |
| det4 (for reference) | v | 33.0 ± 0.7 | 31.34 (−2.4σ) | **32.70 ± 0.45** (1293) | −0.4 |
| | Dp | 0.020 ± 0.0011 | 0.0195 | 0.0215 ± 0.0012 | +1.4 |

- v = gap_input / T. The pull is against the input's stated error, the PASS_PLAN test.
- **det3's gap:** its +2.0σ in S4 was the edge. Interior only, the ± tan halves give 28.44 / 27.89.
- **det2's gap:** the ± tan halves still split 30.35 / 32.86. That is unchanged by the mask and was already there in S4.
- **det6's Dp:** it reads 0.0094–0.0134 across arms, and the low-amplitude fits collapse to Dp = 0. Its gap swings by
  13 mm with the selection: audit flags 32.7, + bundle dead 38.2, interior 24.8. **Neither is quoted.** 0.018 and
  28.0 ± 1.0 stand on the estimates they already rested on.

### Stack T (ns), step by step

| | det2 | det3 | det6 | det4 |
|---|---|---|---|---|
| S4 as run (all events, no mask) | 803.8 | 785.4 | 1204.4 | 972.6 |
| + mask, all area | 800.3 | 784.8 | 1153.2 | — |
| no mask, interior ≥ 30 mm | 816.7 | 781.1 | 1161.0 | 950.3 |
| **mask, interior ≥ 30 mm (quoted)** | **813.5 ± 12.8** | **779.4 ± 7.3** | **1130.1 ± 35.6** | **932.1 ± 12.9** |
| same, interior ≥ 20 / ≥ 40 mm | 806.2 / 824.4 | 781.1 / 777.8 | 1148.6 / 1118.7 | 939.9 / 936.2 |
| same, pre-pass template | 816.6 | 777.8 | 1120.7 | 923.7 |
| mask, edge zone (< 30 mm) | 727.6 ± 32.7 | 819.3 ± 18.7 | not constrained | 991.6 ± 63.2 |

- **Template (pre-pass → pass), interior:** −3.0 / +1.6 / +9.4 / +8.5 ns. That is ≤ 0.8 % for every chamber.
  - det4's +20–23 ns in the follow-up shrinks to +8.5 once the mask and the interior cut are applied.
  - So the template's effect on the stack was mostly through the dead-strip events.
- **Mask:** small on det2/det3 (≤ 3.5 ns; 7 and 8 channels masked), large on det6 (−31 to −51 ns; 40 channels, including
  every other strip over ch 36–62 and a block at 208–224) and on det4 (−18 ns interior, −67 ns at < 1000 ADC).
- **Edge zone:** it is chamber-specific in sign: det2 −86 ns, det3 +40, det4 +60. det6's edge-zone fits stay at their
  starting value (T0 = 1241.2 ns), so T is not constrained there.
  - That fits drift-field distortion near the cage, but this re-check does not identify the cause.
  - The interior value is stable from 20 to 40 mm on det3 and det4 (within 4 ns). On det2 it rises by 18 ns across that
    range (1.1σ), and on det6 it falls by 30 ns.

### T against amplitude (X window maximum)

| | det3 | det4 | det6 |
|---|---|---|---|
| no mask, all area: < 1000 / 1000–3500 ADC | 870 ± 38 (80) / 784 ± 7 | **1002 ± 17 / 942 ± 15** (the follow-up's split) | not constrained / 1180 ± 34 |
| mask, all area | 872 ± 72 / 784 ± 7 | **935 ± 13 / 947 ± 14** | not constrained / 1160 ± 34 |
| mask, interior | 12 events / 779 ± 7 | **917 ± 34 / 926 ± 18** | 22 events / 1128 ± 28 |
| share of the < 1000 ADC events in the edge zone | 68 / 80 | 424 / 1533 | 124 / 146 |
| injection: 2000–3500 ADC events at ×1 / ×0.5 / ×0.3 | 782.3 / 778.2 / 777.1 | 953.0 / 945.8 / 944.9 | — |

- det2 has only 48 head-on events below 1000 ADC, too few to fit.
- **det4's +60 ns at low amplitude is the dead strips.** Masked, it is −12 ± 19 ns over the whole area and −9 ± 38 in
  the interior.
- **det3's and det6's +85 ns at low amplitude is the edge zone.** Their low-amplitude events are 85 % edge, and too few
  of them are left in the interior to fit.
- **The injection** keeps each event's drift and scales only the signal. Its T moves down by 5–8 ns (3 noise seeds each,
  paired, so far below the quoted ±7–27 errors). So the estimator does not lengthen T on weak signals. This corrects the
  follow-up's inference above ("most likely the stack estimator on weak signals").
- **det4's Dp still depends on amplitude in the interior** (0.0163 below 1000 ADC against 0.0235 above), with σ0 trading
  against it. T does not. The quoted det4 Dp (0.0215 ± 0.0012, all amplitudes) is +1.4σ. This was not pursued further:
  det4 was not part of this request.

### What this does not rule out

- **What makes the edge zone different is not identified.** The candidates are the drift-field fringe at the cage, the
  reference near the M3 acceptance edge, or the chamber edge itself. The quoted numbers avoid the zone rather than
  explain it.
- **The input sample was not rebuilt for these three chambers.** That is the hits-chain events, `s4f_build.py --gen`, a
  condor job. So there is no closure against the inputs' own T here. The S4-as-run arm reproduces S4 exactly (det2 803.8).
- **det6's interior v rests on 131 events.** The readout window that limits det6's Dp and gap (S4, `gaps.py`) may also
  touch its T. The interior fits converge (1119–1149 ns, away from the 1241 ns start), but a window effect on T is not
  excluded.
- **The ≥ 30 mm cut is a choice.** The 20/40 mm variants are listed above; det2 and det6 move by 1σ across them.
- **det7 was not re-checked with the interior cut.** Its quoted region is still X ch 256–511 (the follow-up above).
- **Errors are bootstrap only** (20 resamples).

### Files

- Tools in `rerun/calib/s4/`:
  - `s4f_fit.py` gained `mask_x` / `guard` (censor or exclude the masked strips) and `edge` (distance of the M3 reference
    to the active-area box);
  - `s4r_specs.py`: every arm grid of this section;
  - `s4r_gap.py`: the gap with the masks and the interior cut;
  - `s4r_inject.py`: the amplitude injection;
  - `s4r_table.py`: the tables, written to `results/s4r/summary.json`.
- Results: `rerun/calib/s4/results/s4r/` (`r_*`, `where_*`, `gap_r`, `inj_*`, `mask_*`, `summary.json`).
- Inputs (copies) and results: `/eos/user/d/dneff/x17rerun/s4_recheck/fits/`.
  - S4's head-on caches (the inclined ones are read from `s4_inputs/Analysis/<row>/s4/incl.pkl`);
  - the pass bundles;
  - the pre-pass bundles, which came from the laptop mirror `~/x17/cosmic_bench/Analysis/<row>/wft/calib_bundle_{r06,lp}`.
- Everything ran locally (seconds per fit), with no condor. Analysis_pass1, the audit trees and `physics_inputs.json` were
  only read.
