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
