# n_TOF reprocessing: current state

**Keep this file current.** It is the resume point if a session drops. Detail
lives in `FINDINGS_2026-07-28_psa_optimization.md` (what was measured),
`FLASH_TIME_BASE.md` (the divert and the flash), `userinputs/README.md` (how to
run one) and `flash_timing/README.md` (the PKUP-referenced calibration).

Last updated: 2026-07-29. **The pre-ship tests have been run and two of them
changed the answer** -- see `FINDINGS_2026-07-29_pre_ship_tests.md`. The
headline is confirmed on a larger sample; the liquid fast/slow boundary should
be dropped, which needs one more variant; and two output-integrity problems
(ADC wrap-around, unusable `satuflag`) belong in the handoff regardless.

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
  0.61-0.67 over a factor 25 in amplitude, so no template basis can absorb it
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

0. ~~Run `PRE_SHIP_TESTS.md`~~ -- **done 07-29**, results in
   `FINDINGS_2026-07-29_pre_ship_tests.md`. T1 green on 2.5x the sample
   (v12 96.3 % vs v4 95.3 %, gap preserved), T3 green, T5 says keep the
   boundary but document it hard, T6 holds except on LIQB. **T4 is not closed**
   -- the per-hit raw classification could not be made trustworthy.
1. Send `ntof_handoff/` to n_TOF (UserInput v12, 26 templates, README, report),
   **with three additions to the README** that came out of the tests:
   - `satuflag` is unreliable -- never set on any wall, and it catches only a
     third to a half of the over-rail liquid hits. Cut on `amp` above the
     per-channel baseline instead (~31 000 liquids and plastics, ~34 100-34 500
     walls). Roughly 0.006-0.06 % of hits, but their `amp` reaches 3.2e8.
   - `aslow` is always zero and `(area - afast)/area` is **not** an n/gamma
     discriminant: its per-pulse spread is 4-9x the physical band and it drifts
     a factor two with amplitude. Usable in aggregate only.
   - the liquid `area` is missing its slow component (already known).
2. Raise with n_TOF separately, because these are PSA/DAQ properties and affect
   the official processing too:
   - the liquid `area` / slow-component issue;
   - **ADC under-range wrap-around**: pulses larger than the baseline wrap to
     ~65 535 instead of clipping, corrupting shape and amplitude. Sub-percent,
     but silent;
   - `satuflag` not being set for the walls at all.
3. **Optional, if the liquid yield claim has to be airtight:** T4's per-hit
   question needs the PSA `tof` <-> raw sample alignment understood first (see
   the findings file). Ask n_TOF what `tof` marks on a fitted pulse -- that one
   answer probably unblocks it.
3. If liquid PSD is wanted, request stream1 raw for the runs of interest
   (~2.7 GB per 70 s chunk, ~150 files per run); reader and extraction tooling
   already exist in this repo.
