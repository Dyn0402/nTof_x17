# n_TOF reprocessing: current state

**Keep this file current.** It is the resume point if a session drops. Detail
lives in `FINDINGS_2026-07-28_psa_optimization.md` (what was measured),
`FLASH_TIME_BASE.md` (the divert and the flash), `userinputs/README.md` (how to
run one) and `flash_timing/README.md` (the PKUP-referenced calibration).

Last updated: 2026-07-28, after submitting the second variant sweep.

---

## Where things are

| | |
|---|---|
| processed output | `/eos/experiment/ntof/data/x17/reproc/<variant>/completed/224572/` |
| processing scratch (must be /afs) | `/afs/cern.ch/work/d/dneff/x17_reproc/` |
| UserInputs, staged | `/afs/cern.ch/work/d/dneff/x17_reproc/userinputs/<variant>/` |
| UserInputs, source | `ntof_processing/userinputs/<variant>/` |

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
| **v8_pssfit** | v4 + PSS shape fitting, 101 ns templates | **96.4 / 0.6** | **production** |
| v9_liqaug | v4 + LIQ shipped pair + a measured third | 95.2 / 0.6 | neutral |
| v10_pssfit_step | v8 + PSS STEP SIZE 2/3 | - | submitted |
| v11_pssfit_width | v8 + PSS SIGNAL WIDTH LOW 4 ns | - | submitted |

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
timing      quality_metrics.py    T1 wall top<->bot sigma      3.18 ns
                                  T2 wall<->plastic sigma      3.18 ns
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
  an off-time sideband subtracted, T2 reads 38.8 ns instead of 3.18 and the MIP
  peak does not exist. Any coincidence width quoted elsewhere in this project
  without subtraction is inflated.
- **`match_window`'s efficiency is not evidence at early times** -- its own
  false-match probability is ~100 % at 1-3 ms. Quote the singles matcher.

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

## Next

1. Grade v10/v11 against v8 (needs partial 0002 for the DREAM bunch range).
2. Reprocess those 9 runs with the winner.
3. Liquids have resisted three template treatments. Stop there.
