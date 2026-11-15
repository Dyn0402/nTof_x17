# D run_104: L1 or L2 latency pin (2026-11-15/16)

**Answer: take L1 for all of run_104. Its timing on beam is the same as the runs on both sides, and the L1 pin fits
D's beam data a little better everywhere, on L2-era runs as well. So the L1 edge on run_104 does not show which epoch
it belongs to. It shows that the D beam pin is better at L1 than at L2.** Provisional on 23 of the 24 control jobs
(the last was still running at 11-16 00:02).

## What L1 and L2 are

The t0 pin (`t0_abs`, soft, σ 15 ns) for chamber D. It was measured as free-t0 latency on cosmic runs, where D stepped
by ~36 ns between run_103 (07-29) and run_105 (07-30): L1 x −17.8 / y −11.2 ns (runs ≤ 103), L2 x 18.3 / y 20.0 ns
(≥ 105; the r149 pin). run_104 (07-29 19:22 → 07-30 11:29) lies between, so pass 1 reconstructed its D arm under both
pins (`Analysis_pass1/side/run104_D_{L1,L2}`). Plan §5–6, `tests/prefreeze/FINDINGS.md` §1 and §4.

## Evidence

1. **Fit, run_104** (`d104_latency.py`, `d104_by_acq.csv`). The same events were fitted under both pins. With each
   pin's prior term removed, the waveform χ² prefers L1 on all 198 acquisitions (59 % of x planes and 56 % of y planes
   better), and capsule pointing (< 15 mm) gives 65,547 tracks under L1 against 64,569 under L2 (+1.5 %). The preference
   does not change across the 16 h, so there is no change point. The fitted t0 sits on whichever pin it was given
   (median pull ≤ 1.5 ns), so t0 itself cannot tell the pins apart.
2. **Hits, model-free timing** (`d104_hits.py`, `d104_hits_{beam,cosmic}.csv`). This is timing QA only, with no
   geometry. Per event and FEU, the earliest time_of_max of significant, non-truncated hits; then the median per
   acquisition. On beam triggers D does not move:

   | beam | D x | D y | D−A x | D−A y |
   |---|---|---|---|---|
   | L1 runs 100, 102 (12 acq.) | 308.2 ± 1.2 | 358.2 ± 1.1 | −101.6 ± 2.4 | −60.0 ± 1.2 |
   | run_104 (29 acq., all 17 sub-runs) | 305.3 ± 0.9 | 358.1 ± 0.6 | −104.6 ± 1.5 | −58.9 ± 1.5 |
   | L2 runs 106, 108 (10 acq.) | 308.7 ± 1.6 | 356.8 ± 1.0 | −98.8 ± 3.8 | −60.7 ± 1.9 |

   A 36 ns DAQ latency step would move every trigger class by 36 ns, so it is ruled out on beam. On cosmics D x moves
   +20 ± 10 ns (run_97–103 → 105–111) but D y moves −7 ± 2 ns, so the cosmic free-t0 "step" is not a common latency
   change either. Not followed further.
3. **Control, condor 4673558** (`control_by_acq.csv`): 24 L2-era beam acquisitions (runs 106: 14, 108: 10)
   reconstructed under the L1 pin (EOS `Analysis_pass1/tests/d104_control_L1`), compared with the main pass's L2-pin
   outputs of the same acquisitions. L1 wins on all 23 finished acquisitions, by the same margin as on run_104: 58–61 %
   of x planes better, pointing 7,338 against 7,241 (+1.3 %). The pre-freeze check had the same result on L1-era runs
   79/81 (+1.5 / +2.5 %).

## Implication for the main pass (not acted on: Dylan's call)

D's beam data in the L2 era (runs 105–159, main pass on `D_E2_beam`) carry a pin that fits about 1 % worse in pointing
than L1. The plan already noted that D's beam tracks pull the pinned t0 to −12 / −15 ns below its cosmic latency
(§6 11-05). The best D beam pin is therefore probably below L2, and possibly between the two pins, rather than at L1
itself. This was not scanned. The geometry effect is small (soft pin; t0 at fixed geometry is flat). A re-run of D's
L2-era beam acquisitions on one beam pin would remove the inconsistency.

## What this does not rule out

- The hits estimator assumes D's earliest beam hits are trigger-correlated (D's own trigger). If they were not, a
  latency shift would not move them. Their stability across acquisitions (±4 ns) suggests they are correlated, but this
  was not shown directly.
- The comparison is between two pins only, so it says which pin is better, not where the optimum lies.
