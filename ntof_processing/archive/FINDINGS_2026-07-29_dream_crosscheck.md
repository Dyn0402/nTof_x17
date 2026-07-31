> # ⛔ RETIRED — do not build on this
>
> **Superseded by `../../ntof_dream_merge/DREAM_NTOF_CALIBRATION.md`.** Archived 2026-07-30.
>
> The first full-statistics DREAM cross-check of v12. All three of its sections are superseded:
>
> - **§1, the matcher** (95.7 % / 0.5 %) was measured with the old ±150 ns + [250,450] ns window and the official-file `K`, `T0`. The re-derived numbers are 95.84 % at **0.049 %** — same efficiency, 7× less background.
> - **§2, the MM activity cross-check** used the same window, so its arm assignment carries ~0.5 % accidental contamination and 0.46 % two-arm ambiguity. **It should be re-run at ±25 ns**, where those are 0.049 % and 0.15 %.
> - **§3, the liquids**, was already superseded on 07-30 by `../FINDINGS_2026-07-30_liquid_leg_fullpair.md` (both sub-runs, correct saturation cut).
>
> **Read `../../ntof_dream_merge/DREAM_NTOF_CALIBRATION.md`.**

---

# DREAM cross-check of the v12 reprocessing — full statistics + first physics

**2026-07-29.** This executes Section 9 of
`HANDOFF_2026-07-29_dream_vs_reprocessed.md` (the matcher on the whole
reference pair) and then uses the merge for its first physics: do the
Micromegas see tracks where the reprocessed n_TOF says a wall+plastic singles
fired, and do the liquids see coincident light?

Everything below ran locally against the complete 16-partial
`v12_liqpileup/224572` (32.2 GB, download completed today) and the official
merged `run224572.root`, through `dream_regression.py` (now with a `--repair`
flag for the production-baseline mode) and two new tools in
`ntof_dream_merge/`: `mm_activity_crosscheck.py` and `liq_coincidence.py`.

## 1. The matcher on the whole reference pair — v12 wins by 3.3 points

Thresholded wall AND plastic singles matcher, per sub-run (1012 + 1049
bunches, 105k + 108k DREAM events — the full pair, vs the 252 bunches the
headline rested on):

| file / mode                        | sub-run 0000 | sub-run 0001 | plastic-leg cost |
|---|---|---|---|
| **v12, its own tflash (repair OFF)** | **95.7 % / 0.5 %** | **95.7 % / 0.6 %** | 2.7 % |
| official, laptop repair ON         | 92.5 % / 0.5 % | 92.3 % / 0.5 % | 5.9 % |
| official, repair OFF (control)     | 12.2 % / 0.1 % | — | 75.4 % |

Notes:

- The two sub-runs are fully independent hours of DREAM data on disjoint
  bunch ranges (146–1157 / 1165–2213) and agree to 0.0 points on both files.
  The number is stable, not a sample fluctuation.
- v12's measured per-arm time-base offsets are +2.5/+1.5/+0.5/−3.0 ns —
  the stored tflash needs no repair. `match_window` gives 99.7–99.8 % with a
  100.0 % plastic partner rate for wall-matched events.
- The official file *with the laptop repair* — the best it can do — is 3.2–3.4
  points lower at the same false rate, loses disproportionately at late times
  (87.4 % vs 95.2 % in 40–80 ms), and its plastic leg costs 5.9 % vs 2.7 %.
  The earlier "93.7 %" baseline was a 100-bunch number; on full statistics
  the official+repair figure is 92.4 %.
- Wall-only efficiency is 98.4 % on both files: the gain is all in the
  plastic leg, exactly where the v8→v12 shape-fitting change aimed.
- The control row is why the reprocessing exists: on its own stored tflash
  the official file gives 12.2 %, with the plastic leg alone costing 75
  points (the PSS tflash mis-tags of FINDINGS_2026-07-28_pss_tflash.md, seen
  at full statistics). v12 needs no repair to beat the repaired official.

## 2. The Micromegas confirm the matches are physical

`mm_activity_crosscheck.py` (300 bunches of stat090_0000, 31,432 non-flash
DREAM events, 96.3 % matched to ≥1 arm, 95.8 % to exactly one): for events
matched exclusively to one arm, MM chamber activity (≥2 strips in both
planes; chambers mapped via `ntof_tracking/reco/io.py`, candidate-level only
per `RECONSTRUCTION_BASIS.md`):

```
  matched to      chA     chB     chC     chD      n
    arm A only   81.7%   57.4%   26.5%   64.7%   6526
    arm B only   38.2%   77.5%   33.0%   68.8%   8237
    arm C only   36.8%   59.8%   76.8%   67.3%   7987
    arm D only   33.7%   57.6%   28.7%   85.6%   7358
    unmatched    54.5%   62.7%   43.2%   74.5%   1163
```

- The diagonal is enhanced in every row, on top of chamber-dependent
  occupancy/gain floors (chambers B and D are busier; chamber B's median max
  amplitude is ~900 ADC vs ~250–500 elsewhere). A large-pulse tier
  (amax > 500 in both planes) sharpens it: A 15.9 % vs ~4 % off-arm,
  C 21.6 % vs ~4 %, D 36.3 % vs ~20 %, B 14.2 % vs ~8 %.
- **The matcher does not select against MM content**: events it misses have
  a cluster in some chamber 96.4 % of the time vs 96.5 % for its hits. The
  residual ~4 % inefficiency is scintillator-leg, not fake DREAM triggers —
  consistent with the analysis-side candidates in handoff §8.3.
- The unmatched row looks like a mixture of all four arms, as it should if
  misses are random in arm.

## 3. The liquids see the same events, on the same clock

`liq_coincidence.py` on the 30,108 exclusively-matched events (hits with
`amp > 31 000` dropped; `satuflag` ignored — see pre-ship findings): residual
t_LIQ − t_wall per (matched arm, liquid), coincident hits per event in
±100 ns around the residual peak, vs the same window with the wall time
shifted +100 µs (accidental floor):

```
  matched arm    LIQA          LIQB          LIQC          LIQD
    A         0.159/0.025@-5   0.025/0.032   0.009/0.002   0.019/0.015
    B         0.047/0.035      0.150/0.031@-5   0.008/0.005   0.039/0.014
    C         0.038/0.037      0.047/0.033   0.020/0.003@-25  0.030/0.024
    D         0.031/0.037      0.045/0.025   0.008/0.003   0.090/0.019@-15
```

- Every same-arm cell is a 5–7× excess over accidentals; every off-arm cell
  sits at the floor with an unstable peak position.
- **The v12 liquid time base is aligned with the walls to −5…−25 ns.** No
  trace of the official file's ~350 ns per-tree feature offsets. This is the
  first end-to-end validation of the LIQ leg of v12 against an external
  detector.
- 9–16 % of matched events have same-arm liquid light (LIQC lower — it also
  has 3.4× fewer hits overall); enough statistics for the Phase-5 physics.

## 4. Anything to change before the full-campaign reprocessing?

**No.** Nothing found here motivates touching the UserInput:

- The wall+plastic configuration is confirmed at full statistics with
  margin over the official processing; the remaining plastic-leg 2.7 % was
  already shown to be analysis-side (three reconstructions identical — §8.3
  of the handoff), and the MM comparison independently supports that (misses
  are not fake events, and their MM content is normal).
- The liquid time base and hit content pass their first external test.
- The known output caveats (~~ADC wrap~~, `satuflag`, missing slow component)
  are PSA/DAQ properties, already in the n_TOF handoff README list, and none
  of them bit this analysis after the documented cuts.
  **Corrected 2026-07-29 evening:** there is no ADC wrap (signed-vs-unsigned
  decoding error), and `satuflag` is good on the liquids — see
  `FINDINGS_2026-07-29_signed_decoding.md`. The `amp > 31 000` cut above was
  therefore stricter than needed: it drops legitimate half-scale pulses rather
  than corrupted ones. It is conservative, so the conclusions here stand, but
  the cut should be redone at the real ceiling (~63 800) if these numbers are
  ever quoted per-hit.

Ship the campaign on v12 as staged. (n_TOF call it by its staged UserInput;
there is no "v14" — v13 was rejected, v12 is the production candidate.)

## Reproduce

```bash
.venv/bin/python ntof_processing/dream_regression.py \
    /media/dylan/data/x17/ntof_reproc/v12_liqpileup run_79 stat090_0000 1200
# add --repair and the official file path for the baseline rows
.venv/bin/python ntof_dream_merge/mm_activity_crosscheck.py \
    /media/dylan/data/x17/ntof_reproc/v12_liqpileup run_79 stat090_0000 300 \
    --out /tmp/mm_match_sub0.npz
.venv/bin/python ntof_dream_merge/liq_coincidence.py \
    /media/dylan/data/x17/ntof_reproc/v12_liqpileup /tmp/mm_match_sub0.npz
```
