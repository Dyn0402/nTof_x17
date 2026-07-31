# n_TOF reprocessing: current state

**Keep this file current.** It is the resume point if a session drops. Detail
lives in `FINDINGS_2026-07-28_psa_optimization.md` (what was measured),
`FLASH_TIME_BASE.md` (the divert and the flash), `userinputs/README.md` (how to
run one) and `flash_timing/README.md` (the PKUP-referenced calibration).

Last updated: 2026-07-30 (evening). **The n_TOF side is closed; the analysis has
started, and the DREAM<->n_TOF time calibration is now locked.**

> **The authority on the match is
> `../ntof_dream_merge/DREAM_NTOF_CALIBRATION.md`.** Constants, the per-run
> re-derivation recipe, what transfers and what does not. Tooling and slides in
> `../ntof_dream_merge/match_study/`; machine-readable constants exported to
> `../../nTof_x17_DAQ/calibrations/dream_ntof/`. Everything earlier is retired
> into `archive/` (see `archive/README.md`) -- do not build on it.

Headline, on the complete reference pair (2061 bunches, 213 420 triggers):

- **accept window ±25 ns, one band.** 95.84 % efficiency at a **measured**
  0.049 % accidental rate, two-arm ambiguity 0.15 %. The old
  ±150 ns + [250,450] ns window bought +0.17 points of efficiency for 7× the
  background; the satellite band carries no signal on v12 at all.
- **the window was never a resolution.** The DREAM timestamp clock drifts
  ~1 ppm bunch to bunch, smearing the residual in proportion to time since flash
  (9 ns at 1 ms, 37 ns at 40-80 ms). Fitted per bunch and cross-validated the
  residual is **flat at 6 ns** over the whole 80 ms. This corrects
  `time_align.py`'s "36 ns of DREAM trigger jitter".
- **the per-bunch fit was audited for self-fulfilment** and passes five tests:
  107 triggers per bunch for 2 parameters, split-half rho = +0.996 (0.92 ppm of
  real drift against 0.06 ppm of fit noise), a 3-5 % in-sample-vs-cross-validated
  gap, wide-window efficiency **identical to five decimals**, and a wrong-bunch
  parameter swap that makes things worse. `match_study/scripts/bias_check.py`.
- **nothing transfers between processings.** K and T0 refit on v12 to
  1.103724e-4 (+1.35 %) and -253.64 ns, plus per-arm trigger-path offsets
  A -16.81 / B +7.55 / C +1.62 / D -0.83 ns. The wall top/bottom "cable offsets"
  are ±32-39 ns on the official file and within ±5.5 ns on v12 -- a flash-finder
  artifact, not cabling; `dream_trigger.py`'s stored table is flagged.
- **do NOT run the tflash repair on v12**: it would shift LIQC/D by 15 ns and add
  25 ns RMS on PSSC, and the stored time base already has the liquids within 1 ns
  of the walls.
- **the reprocessing checks out in situ**: the v12 liquid flash times reproduce
  the divert-off `flash_timing` calibration to 0.1-0.5 ns; walls spread 4.0 ns;
  per wall channel RMS 2.3 ns; liquid-vs-wall -0.8..+0.2 ns. No internal offsets
  are needed.
- coverage (accidental-subtracted): 96.00 % of triggers get a wall AND plastic
  partner, 98.59 % wall-only -- the plastic leg still costs 2.58 %.
- new: `ntof_dream_merge/fast_singles.py`, a vectorised `dream_trigger`
  (validated bit-identical) -- the original is O(N_hits x N_bunches) and cannot
  run on 2061 bunches.

Earlier the same day: **the handoff package now carries the corrected
saturation story.** `ntof_handoff/README.md` §8b had been left on the retracted
ADC-wrap text (its last commit predated the signed-int16 finding): (a) now says
`satuflag` is reliable on the liquids and structurally absent on the walls, and
recommends cutting `satuflag` **or** `amp` > ~63 800 -- neither alone is complete
(satuflag misses ~9 % of over-ceiling hits; the amp cut misses flagged hits whose
extrapolated amp lands back in range); (b) now says the ADC clips at its rails
with no wrap, gives the ±950 mV baseline-offset table, and adds that the wall
**front end** limits at ~34 600 counts (~half of ADC full scale) where no rail
test can see it. The two `adc_wrap_*.png` figures were withdrawn from the package
and replaced with `sat_examples_liq.png` / `sat_population_liq.png`. The stale
question we were about to ask n_TOF about a per-channel `start` offset is
retracted in the README too -- it was our parser's 259 pre-samples.
**Do not re-issue the old "cut amp > 31 000" advice**: on LIQA that cut removes
2 099 hits of which 1 561 (74 %) are ordinary half-scale pulses.
**New today, and they answer the saturation question end to end** —
`FINDINGS_2026-07-30_saturation_walls_plastics.md` and
`FINDINGS_2026-07-30_liquid_leg_fullpair.md`:

- **The walls have a hard ceiling BELOW the readout limit.** Reported `amp`
  terminates at 43 220-44 915 on all four and never reaches 63 800; the raw
  excursion limits at 32 888-34 635 counts. So it is analogue, and no rail test
  can see it — **cut WAL `amp` > 34 600 in post-processing**. Fires only in the
  flash (physics-time wall amplitudes stop below ~25 000). Now in the handoff.
- **The plastics are the opposite**: PSSA/B/C do reach the ADC rail, so
  `satuflag` fires on them. **PSSD does not** — it is analogue-limited at 44 806
  (70 % of range), which is the real reason it never sets the flag.
- **`amp > 63 800` on a plastic is a FIT-QUALITY flag, not a saturation flag.**
  Correcting what this file said earlier: about half the over-ceiling PSSA/PSSC
  hits are *unflagged because they never clipped* — measured peak 58-62 k against
  a 63 568 rail — and the fit merely overshoots (1.45x on PSSD up to 22-80x on
  genuinely clipped hits). `satuflag` is right about both halves.
- **Flag implemented**: `ntof_io.saturated(tree, amp, satuflag)` +
  `saturation_ceiling(tree)` — one definition, per-family ceilings (WAL 34 600,
  LIQ/PSS 63 800, SILI 59 100, PKUP 59 400). `liq_coincidence.py` and
  `liq_saturated_study.py` now call it.
- **`area` is proportional to `amp` BY CONSTRUCTION, and the PSA guide says so.**
  With AMPLITUDE OPTION=2 "both the final amplitude and area will be determined
  from the fitted pulse", so area = amp x integral(shape): `area/amp` takes
  exactly one value per `pulseshape`, matching the per-shape counts to the hit.
  **The measured pair is `amp_0` (pre-fit maximum) and `area_0` (pre-fit
  integration)** — use those for a real integral or an un-extrapolated amplitude.
  `amp`/`amp_0` is the best saturation diagnostic in the file: ~1.0 clean,
  1.24-1.30 wall flash artifacts, 1.45 PSSD overshoot, 22-80 clipped plastics.
- Figures: `liq_study/pss_over_ceiling_PSSC.png` (the flash plunge to the rail),
  `liq_study/wal_front_end_WALB.png` (the divert step parked at ADC zero),
  `liq_study/wal_pss_saturation.png` (spectra + width vs amplitude).
- **Method** (`liq_study/sat_curve.py`): width-vs-amplitude plateau departure,
  calibrated on the liquids where `satuflag` is truth. The liquids stay flat to
  0.1 ns up to the ceiling *even inside the flash*, so a departure below the
  ceiling is real and not a flash artifact. Do **not** use the automatic
  "1.2x the plateau" rule — it mis-called both LIQ and PSS; read the table.
- **A physics-time clipped liquid pulse keeps its time** (`clipped_timing_check.py`):
  dt to a fixed-depth raw crossing is 3.5-3.8 ns against 3.6-3.7 for unclipped
  controls, so saturated hits are usable as TIME hits with `amp` as a lower
  bound of 63 800. The 114-129 ns mistiming tail is entirely flash-region.
  `area` cannot help recover amplitude — it is exactly proportional to `amp`.

The §8b(a) table is now a real whole-run census (it had been a single partial,
3.4 M LIQA hits, labelled as the run: the run has 51 M) — reproduce it with
`liq_study/amp_ceiling_census.py`, output kept as
`liq_study/amp_ceiling_census_v12_224572.json`.
`ntof_dream_merge/liq_coincidence.py` carried the same wrong cut and now uses
`ntof_io.saturated()`. **Re-run on the whole of `stat090_0000`** (1012 bunches,
100 083 exclusively-matched events, 3.3× the old sample) it reproduces the 07-29
table cell by cell — same-arm diagonal at −5…−25 ns, excess **3.6-5.9×** over the
shifted control. **`stat090_0001` replicates it** on a disjoint hour — diagonal
cells 0.164/0.146/0.016/0.092 against 0000's 0.165/0.151/0.018/0.094, same
−5…−25 ns offsets. Restate the excess as **2.7-5.9×** over both sub-runs, not the
"5-7×" from 300 bunches (LIQC is the weak cell, 2.7-3.6×, on a better-measured
floor with an unchanged signal cell).

**The merge tooling is ~46× faster as of 2026-07-30** and bit-identical: the
per-bunch/per-event Python double loop in `liq_coincidence.py` is now one
`window_residuals()` call (bunch and time packed into a sorted float64 key, two
`searchsorted`, one ragged gather), and `ntof_io.variant_cache()` replaces the
`tempfile.mkdtemp()` sandbox in all five scripts with a persistent directory
fingerprinted on the file set — same isolation, but the bunch index is not
rebuilt every run. **1 h 52 min → 2 min 25 s** cold, **1 min 55 s** warm, with all
33 residual histograms identical bin-for-bin (363 038 entries both ways).

§3 of `archive/FINDINGS_2026-07-29_dream_crosscheck.md` (300 bunches, wrong cut) is now
**superseded** by `FINDINGS_2026-07-30_liquid_leg_fullpair.md` — both sub-runs,
correct cut. Read the 07-30 file for any liquid number.

Earlier, 2026-07-29 (evening): **the DREAM cross-check has run on the
FULL reference pair** -- see `archive/FINDINGS_2026-07-29_dream_crosscheck.md`
(retired: its matcher numbers predate the re-derived time map, and its MM
cross-check needs re-running at ±25 ns). On all
2061 bunches / 213k DREAM events of both run_79 sub-runs: **v12 95.7 % / 0.5 %
on its own tflash**, vs official+repair 92.4 % and official-alone 12.2 %.
Both sub-runs agree to 0.0 points. First physics through the merge both pass:
MM chamber activity concentrates in the matched arm, and the liquids show
5-7x same-arm coincidence excesses at -5..-25 ns offset (the v12 LIQ time
base is wall-aligned). **Nothing found motivates another UserInput variant --
ship the campaign on v12.**

Earlier the same day: the pre-ship tests (`FINDINGS_2026-07-29_pre_ship_tests.md`);
headline confirmed at 252 bunches, T2/T3 pass. ~~Two output-integrity problems --
ADC wrap-around and an unusable `satuflag` -- go in the handoff~~ **both
retracted the same evening**: the raw samples are signed int16 and the tooling
read them unsigned, so there is no wrap, and `satuflag` is verified good on the
liquids (119/123 clipped runs matched per pulse). T4's per-hit liquid check is
no longer blocked either -- the raw-to-`tof` offset is a constant 259 samples.
See `FINDINGS_2026-07-29_signed_decoding.md`.

**Open, for a separate session:** the shared raw parser
`nTof_x17_DAQ/stream1_monitor/ntof_raw.py:163` still decodes samples as `<u2`.
The full write-up — evidence, the fix, the two operational consumers to re-check
(`stream1_size_controller.py`, `wall_probe.py`), the `0x8000` fill collision and
the 259-pre-sample offset — is in that repo as
`stream1_monitor/SIGNED_DECODE_FIX_NOTE.md`. Nothing there has been changed yet.

**Auditing this?** Start at `REVIEW.md` -- it maps every claim to the tool that
produced it, says what is reproducible and what is ephemeral, and lists the
mistakes I made and corrected so you know where the error modes were.

---

## Where things are

| | |
|---|---|
| variant studies (run 224572) | `/eos/experiment/ntof/data/x17/reproc/<variant>/completed/224572/` |
| production runs (224573-224579) | `/eos/experiment/ntof/data/x17/reproc/prod_v11/<run>/completed/<run>/` |
| processing scratch (must be /afs) | `/afs/cern.ch/work/d/dneff/x17_reproc/` |
| UserInputs, staged | `/afs/cern.ch/work/d/dneff/x17_reproc/userinputs/<variant>/` |
| UserInputs, source | `ntof_processing/userinputs/<variant>/` |
| package for n_TOF | `ntof_handoff/` |
| DREAM-vs-reprocessed entry point | `../ntof_dream_merge/DREAM_NTOF_CALIBRATION.md` |
| retired documentation | `archive/` -- do not build on it |
| local copy of 224572 v12 | `/media/dylan/data/x17/ntof_reproc/v12_liqpileup/` |

Note on 224572: it has no directory under `prod_v11/` because it is the
reference run every variant was built on -- its production-configuration output
is `reproc/v12_liqpileup/completed/224572/`. Nothing is missing; the naming just
differs. The user EOS (`/eos/user/d/dneff/x17/reproc/`) is kept empty: every
output is copied to the ntof disk and verified file-by-file before the source
is removed.

`ssh -K` is mandatory. Without delegated credentials there is no AFS token and
no condor auth, and `/eos/user/d/dneff` looks like it does not exist.

**Never merge a run.** The official merge node cannot (condor file-transfer cap
at 1024 MB), hadd over EOS dies and leaves a truncated file that still opens.
`ntof_io.ntof_paths()` chains `run<run>.parts/run<run>_NNNN.root` instead.

## Division of labour, settled

- **Time base** comes from `flash_timing/` -- `t_flash = tof_PKUP + C`, ~1 ns.
  Not from stored `tflash`, even reprocessed.
- **Hit content** comes from the reprocessing. That is what these variants are
  optimising, and it is what the calibration cannot give.

## Variants

| variant | change vs its base | singles eff / false | verdict |
|---|---|---|---|
| v1_flash | G-FLASH THRESHOLD only | - | superseded |
| v2_elim | + PSS/LIQ AREA/AMP 1..60, PSS amp thr 50 | - | superseded |
| v3_shapes | + 551 ns WAL and LIQ templates | - | LIQ part rejected |
| v4_walshapes | v2 + WALL templates only | 95.2 / 0.6 | superseded by v8 |
| v5_liqshort | v4 + 81 ns LIQ templates | - | rejected |
| v6_lowthr | v4 + PSS/LIQ amp thr 25 | 95.2 / 0.6 | **no effect on the matcher** |
| v7_step | v4 + STEP SIZE WAL 5/5, PSS 2/3, LIQ 2/3 | 95.1 / 0.6 | neutral/slightly worse |
| v8_pssfit | v4 + PSS shape fitting, 101 ns templates | 96.4 / 0.6 | the big win; superseded by v12 |
| v9_liqaug | v4 + LIQ shipped pair + a measured third | 95.2 / 0.6 | neutral |
| v10_pssfit_step | v8 + PSS STEP SIZE 2/3 | 96.4 / 0.6 | equal to v8 |
| v11_pssfit_width | v8 + PSS SIGNAL WIDTH LOW 4 ns | 96.4 / 0.6 | superseded by v12 |
| **v12_liqpileup** | v11 + LIQ STEP SIZE 1/3, fast/slow boundary | **96.4 / 0.6** | **production, shipped to n_TOF** |
| v13_liqexpand | v12 + LIQ EXPAND PULSES 1, 150 ns width | - | rejected, -17..-28 % liquid hits |

### The liquids, settled

`liq_study/FINDINGS_liquids.md` has the detail. In short:

- **not templates**: every template we built was measured on the isolated
  minority of pulses and none transferred. Note the "8-24 % isolated" figure
  uses a 200 ns window, so it measures TAIL overlap -- the fast components are
  mostly resolvable (24-30 ns median gap vs 6 ns FWHM)
- **photon statistics floor**: fit residual scales as sqrt(A), flat at
  0.61-0.67 over a factor 25 in amplitude, so no template basis can absorb it.
  **07-29: true of LIQA/LIQC/LIQD, NOT of LIQB** (residual/sqrt(A) 0.62 -> 1.59;
  an amplitude-binned basis cuts it 24 % held-out). And the "saturation breaks
  the scaling at the rail" line was measuring the ADC wrap -- with wrapped
  pulses dropped it does not break
- **not two pulse classes**: tail/total is one band at 0.21 above 3000 ADC
- **v12 works**: LIQ `STEP SIZE` 2/4 -> 1/3 gives **+14 to +21 % yield**, chi2
  neutral-to-better, pileup flag +50 %, walls and plastics bit-identical
- **PSD is not obtainable from the PSA**: `afast`/`aslow` are 0 % filled;
  setting the boundary fills `afast` but leaves `aslow` at zero because the
  slow component lies outside the reconstructed pulse boundary, and expansion
  to reach it costs more than it gains. **The reported liquid `area` has
  therefore always been missing its slow component** -- in the official
  processing too.
- **raw waveforms would NOT help**: an iterative deconvolution on the raw data
  finds 0.67x the PSA's hits, not more (`deconv_vs_psa.py`). And 67-76 % of
  pulses have a neighbour inside their own 150 ns tail, so a custom fitter
  faces the same overlap. This is a rate limitation, not a software one.

### The processing has hit its floor at 96.4 %

**Note (07-29):** every figure in this section is from the original 100-bunch
grading, which is the sample all the variants were compared on and so remains
the right basis for comparing them *to each other*. On the larger 252-bunch
sample the absolute numbers are v12 96.3 % / 0.5 % against v4 95.3 % / 0.5 % --
same gap, slightly lower false rate. Quote the 252-bunch numbers outwardly.

v8, v10 and v11 give **identical** matcher efficiency (96.4 % / 0.6 %) and
identical leg breakdowns, despite reconstructing the plastics very differently:

```
              PSS chi2 p50   PSS amp p50   PSS hits   hits with amp>2000
  v8            baseline       baseline     baseline        2834 / 4757
  v10_step      +1.6..+4.0%    +20..+44%    -13..-17%       2822 / 4746
  v11_width     -2.6..-13.2%   -16..-25%    +28..+34%       2845 / 4767
```

The last column is why: **above ~2000 ADC the three are identical to <1 %**.
Once shape fitting is on, the trigger-relevant plastic population is fixed and
the remaining knobs only shuffle small pulses, which the discriminator never
sees. Three very different reconstructions giving the same efficiency is strong
evidence the limit is no longer in the PSA.

So the remaining 2.5 % plastic-leg loss is **not a pulse-recognition problem**.
If it is worth chasing, it is analysis-side: the per-arm discriminator
threshold model (`thr['plastic']`), the D_PMTS channel selection, plastic dead
time, or genuine detector inefficiency. Do not spend more UserInput variants
on it.

**v11 chosen over v8** on the tiebreakers: same efficiency, same timing and
amplitude quality (every metric within 0.8 %), same large-pulse yield, but
3-13 % better plastic fit chi2 and 28-34 % more plastic hits available to
non-trigger analyses. **v12 then adds the liquid fix on top of v11 and is what
ships**; the plastic and wall configuration is identical between them.

### What the second sweep established

- **v8_pssfit wins on the headline**: 95.2 -> 96.4 % efficient at the same
  0.6 % false, 93.4 -> 95.5 % in the hardest 1-3 ms bin.
- **and it does so with FEWER plastic hits** -- 0.72-0.99 of v4 at every
  amplitude cut -- while producing MORE valid candidates (103,816 vs 101,809).
  The gain is plastic TIMING in pileup, not plastic yield. Hit count and
  quality point opposite ways here; that is what the scorecard is for.
- **v6_lowthr changes nothing for the matcher** despite +15-25 % more plastic
  and +29-47 % more liquid hits: the trigger emulation thresholds the plastic
  leg in mV at the discriminator, so 25-50 ADC hits never enter it. Those hits
  are real and remain available for non-trigger analyses, but they do not
  belong in production.
- **The plastic leg is still what limits the AND.** Wall-only efficiency is
  98.9 % and flat in time; the AND is 96.4 %. The plastic costs 2.5 % overall
  and 3.4-3.7 % at 1-10 ms -- a pileup signature. v10/v11 target exactly that.
- The liquids have now resisted three template treatments. Stop there.

## Scorecard, and the v4 baseline to beat

Efficiency is the headline; timing and amplitude are the guards. Accept a gain
only if the guards do not degrade.

```
efficiency  dream_regression.py   singles matcher   95.2 % eff / 0.6 % false
                                  (1-3 ms bin)      93.4 % / 2.6 %
timing      quality_metrics.py    T1 wall top<->bot sigma      6.65 ns
                                  T2 wall<->plastic sigma      6.46 ns
                                  T2 centre (wall vs plastic)  8.75 ns
                                  T3 walk over amp deciles     1.38 ns
amplitude                         A1 MIP peak                  1081 ADC
                                  A1 FWHM/peak                 1.22
                                  A2 log(top/bot) resid        0.362
                                  A2 sqrt(top*bot) flatness   +9.1 %
content     grade_candidate.py    flash-id bad bunches         0.0 %
                                  per-arm offsets      +1.5/+2.0/+1.0/-2.0 ns
            compare_fits.py       WAL chi2 p50           0.85-1.06
```

Two things to remember when reading these:

- **Accidental subtraction is not optional.** Both trees are high-rate; without
  an off-time sideband subtracted, T2 reads 38.8 ns instead of 6.46 and the MIP
  peak does not exist. Any coincidence width quoted elsewhere in this project
  without subtraction is inflated.
- **`match_window`'s efficiency is not evidence at early times** -- its own
  false-match probability is ~100 % at 1-3 ms. Quote the singles matcher.
- **These sigmas come from a background-subtracted second moment.** An earlier
  FWHM/2.355 estimator with 2.5 ns bins reported 3.18 ns for every variant and
  could not discriminate at all -- `tof` is quantised to 1 ns. If you find
  3.18 ns quoted anywhere, it is the stale estimator.

## The loop

```bash
# 1. generate + stage + submit
python ntof_processing/make_variants.py vX_name
./ntof_processing/deploy_userinput.sh vX_name <local> /afs/cern.ch/work/d/dneff/x17_reproc/userinputs
rsync -a -e "ssh -K" <local>/ dneff@lxplus:/afs/cern.ch/work/d/dneff/x17_reproc/userinputs/
ssh -K dneff@lxplus  # then RunProcessing.sh -y 2026 -a EAR2 -c X17_measurement -r 224572 \
                     #   -p .../userinputs/vX/UserInput.h -o /eos/user/d/dneff/x17/reproc/vX

# 2. grade -- partials land ~25 min in, no need to wait for all 16
xrdcp partials 0001,0002 down       # bunches 1-397; the DREAM pair needs 146-245
python ntof_processing/dream_regression.py <dir>     # efficiency
python ntof_processing/quality_metrics.py  v=<files> # timing + amplitude
python ntof_processing/grade_candidate.py  v=<files> # flash id + offsets + counts
python ntof_processing/compare_fits.py     a=.. b=.. # fit chi2

# 3. move off the user quota when a variant is kept
#    (verify per-file sizes BEFORE deleting the source)
```

## Which runs to reprocess, and what is actually still on EOS

Measured 2026-07-28 (`scratchpad/eos_inventory.txt`, regenerate with the loop in
the git log of this file): **156 of the 329 run directories still carry
stream1**, spanning **2026-07-02 to 07-28**. The missing 181 are scattered
through the range, not a clean age cutoff -- so retention is NOT the simple
"~2 weeks" the earlier handoff assumed, and it cannot be predicted per run.
Check before planning, do not extrapolate.

The two DREAM runs staged locally, and the n_TOF runs that cover them by
wall-clock:

| DREAM run | window | n_TOF runs | on EOS? |
|---|---|---|---|
| run_79 | 07-26 18:07 -> 07-27 10:00 | 224572 (done), 224573-224579, 224580 | all present |
| run_55 | 07-18 19:11 -> 23:53 | 224498, 224499 | both present, 165 / 156 files |

So **9 runs remain to reprocess** for the two DREAM runs we hold, and all of
their raw data is still on disk. Several are short (224575 has 17 raw files,
224579 has 1, 224578 has 71) -- fine, just quick.

## Production status

run_79's n_TOF coverage is **reprocessed and complete**. Verified that partial
count equals job-list count for every run, so nothing failed silently:

| run | raw files | job lists | partials |
|---|---|---|---|
| 224572 | 152 | 16 | 16 (the reference, many variants) |
| 224573 | 156 | 16 | 16 |
| 224574 | 152 | 16 | 16 |
| 224575 | 17 | 2 | 2 |
| 224576 | 150 | 15 | 15 |
| 224577 | 166 | 17 | 17 |
| 224578 | 71 | 8 | 8 |
| 224579 | 1 | 1 | 1 |

`RunProcessing.sh` splits by ~10 raw files per job, which is why the short runs
have few partials -- not a failure. The merge node aborts on every long run
(the 1024 MB condor transfer cap); that is expected and bypassed by reading
partials. **run_55 was dropped deliberately** -- n_TOF will reprocess the
campaign from our UserInput instead.

## Next

0. ~~Run `archive/PRE_SHIP_TESTS.md`~~ -- **done 07-29**, results in
   `FINDINGS_2026-07-29_pre_ship_tests.md`. T1 green on 2.5x the sample
   (v12 96.3 % vs v4 95.3 %, gap preserved), T3 green, T5 says keep the
   boundary but document it hard, T6 holds except on LIQB. **T4 is not closed**
   -- the per-hit raw classification could not be made trustworthy.
1. Send `ntof_handoff/` to n_TOF (UserInput v12, 26 templates, README, report),
   **with three additions to the README** that came out of the tests:
   - `satuflag` (**rewritten 2026-07-29 evening**): it is reliable on the
     liquids -- verified against the raw waveforms on 119 of 123 clipped runs,
     including every physics-time clip -- and is **never set on the walls**,
     because wall saturation is an undershoot outside the detected pulse
     window. A flagged hit must be **cut, not used**: its `amp` is a fit
     extrapolation (66 k-832 k against a physical ceiling of 63 800). The old
     advice to cut on `amp` above ~31 000 was based on a decoding error and
     would throw away ordinary half-scale pulses.
   - `aslow` is always zero and `(area - afast)/area` is **not** an n/gamma
     discriminant: its per-pulse spread is 4-9x the physical band and it drifts
     a factor two with amplitude. Usable in aggregate only.
   - the liquid `area` is missing its slow component (already known).
2. Raise with n_TOF separately, because these are PSA/DAQ properties and affect
   the official processing too:
   - the liquid `area` / slow-component issue;
   - ~~ADC under-range wrap-around~~ **withdrawn: there is no wrap** (our
     decoding error, not a DAQ property);
   - `satuflag` not being set for the walls at all -- still true, and now
     understood: wall saturation is a negative undershoot, opposite to the
     pulse direction, and `AnalyseSaturation` only scans inside found pulses.
3. **Optional, if the liquid yield claim has to be airtight:** T4's per-hit
   question needed the raw-vs-reconstructed time alignment, and that is now
   **solved** (2026-07-29 evening): `tof = start + j - 259` for zero-suppressed
   blocks, constant to +-0.6 ns over 220 isolated pulses, because the ACQC
   block `start` is the zero-suppression trigger sample and the payload begins
   259 pre-samples earlier. The flash block starts at 0 and needs no offset.
   Nothing to ask n_TOF; T4 can simply be redone.
3. If liquid PSD is wanted, request stream1 raw for the runs of interest
   (~2.7 GB per 70 s chunk, ~150 files per run); reader and extraction tooling
   already exist in this repo.
