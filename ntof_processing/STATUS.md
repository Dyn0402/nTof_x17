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

| variant | change vs its base | status |
|---|---|---|
| v1_flash | G-FLASH THRESHOLD only | done, superseded |
| v2_elim | + PSS/LIQ AREA/AMP 1..60, PSS amp thr 50 | done, superseded |
| v3_shapes | + 551 ns WAL and LIQ templates | done, LIQ part rejected |
| **v4_walshapes** | v2 + WALL templates only | **done -- current production** |
| v5_liqshort | v4 + 81 ns LIQ templates | done, rejected |
| v6_lowthr | v4 + PSS/LIQ amp thr 25, PSS AREA/AMP low 0.2 | submitted |
| v7_step | v4 + STEP SIZE WAL 5/5, PSS 2/3, LIQ 2/3 | submitted |
| v8_pssfit | v4 + PSS shape fitting, 101 ns templates | submitted |
| v9_liqaug | v4 + LIQ shipped pair PLUS a measured third shape | submitted |

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

## Next

1. Grade v6-v9 on the scorecard; build a combined variant from what wins.
2. Reprocess the other DREAM-paired runs with the winner. Runs before ~224520
   are past the 2-week EOS window and need CTA staging -- that is the clock.
3. Liquids are not template-limited. If v9 is neutral, stop spending variants
   on the templates and look elsewhere.
