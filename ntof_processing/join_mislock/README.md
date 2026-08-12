# join_mislock — the 2026-08-11/12 matching-failure investigation

**The n_TOF↔DREAM matching failures (25.7 % of attempted beam, 107 of 291
segments) are `pulse_match` silently taking a supercycle-shifted wall-clock
lock on a count tie. The data is fine and recoverable — demonstrated at
95.47 % efficiency on a formerly total-loss hour.**

Narrative and full evidence chain: the campaign report
[`../SLIM_CAMPAIGN_2026-08-12.md`](../SLIM_CAMPAIGN_2026-08-12.md) and the
published note <https://dylan-neff.web.cern.ch/notes/ntof-dream-join-mislock.html>.
This directory holds the scripts that established it, in the order they ran,
plus the key result files. Everything else (inventory, shift table, margin
CSV, all diag outputs) is on lxplus in `/afs/cern.ch/work/d/dneff/x17slim/`.

## The mechanism, one paragraph

`pulse_match.match_subrun` aligns each DREAM sub-run to the beam record by
scanning a ±120 s offset and keeping the one that matches the most trigger
clusters. Both the pulse timing and the parasitic/dedicated intensity schedule
repeat with the accelerator supercycle (39.6 s or 43.2 s depending on the
hour), so locks whole cycles apart match ~100 % of clusters with ~5 ms rms —
the objective cannot tell them apart. On a tie, `n > best_n` keeps the first
(most negative) offset scanned; ±120 s admits exactly ±3 cycles, so corrupt
hours sit at −3 supercycles (measured: bunch shifts +26 and +20 = 118.8 s at
two different spacings; +41 = 129.6 s = 3×43.2 on the other schedule). Which
hour fails is Poisson noise on which lock catches one more cluster — the
margin study found 35 of 41 failures at count-margin exactly 0, and **14
accepted segments at margin ≤ 2** (healthy by coin flip; listed for
verification). Boundary slivers are the same count degeneracy in
`bunch_join`'s ±3 s δ-scan against a truncated pulse list (minority side lost
26 of 26 straddling pairs). A mis-locked segment looks healthy in every join
statistic and simply writes no file — the QA never sees it.

## Scripts (chronological)

| script | question it answered |
|---|---|
| `marginal_closure_test.py` | is the failure a statistics floor? (no — 14-min truncations of correctly-joined data fit easily; also proved the −0.98 ms wide-scan feature sits under healthy fits) |
| `flashref_subpop_test.py` | is the −0.98 ms structure a mis-tagged-flash subpopulation? (no — carried by ordinary matched events; it is the 0.9927 ms post-flash hold-off edge) |
| `dream_forensics.py` | is anything wrong with DREAM in the corrupt hours? (no — flash leads 100 % of bursts, multiplicity and hold-off identical to clean hours) |
| `perbunch_lag.py` | does the coincidence exist per bunch in a failed segment? (yes, everywhere — though the per-bunch lags it reports for a mis-joined segment are artifact+noise; see the kill-list) |
| `recovery_shift.py` | does the standard chain recover a mis-joined hour once the bunch shift is applied? (**yes: 95.47 % / cv 95.43 %, S/N 1319, accidental 0.065 %** — `recovery_shift_run_96_stat090_0001_224597.json`) |
| `margin_study.py` | what separates failed from fitted sub-runs? (the pulse_match count margin: failures 35/41 at 0, all ≤ 8; fitted median 23; **14 fitted at ≤ 2**; run_102/0003's dead tie −69.3 s r=0.508 chosen vs +60.3 s r=0.925 true validated the intensity-fluctuation discriminator in the wild) |

The wide bunch-shift scans that found the mechanism used the existing
`slim_pipeline/segment_diagnose.py` with `--span 200` (condor wrapper
`diag3_*` on lxplus). Eight of eight scanned failures show exactly one sharp
peak (S/N 541–1822).

Scripts here run standalone next to the repo packages (they `sys.path` their
own directory; run them from a checkout root or a condor sandbox that
transfers `ntof_processing`, `ntof_dream_merge`, `ntof_july_analysis`,
`common`, `mx_july_beam_qa`). `X17_BEAM_JULY` must point at a beam_july tree.

## The fix (design agreed 08-12 ~03:00, NOT yet implemented — Dylan's call)

1. `pulse_match`: score locks by count **+ intensity-fluctuation correlation**
   (the schedule repeats, the fluctuations don't: r 0.925 vs 0.508 picked the
   true lock outright) against the per-window grid measured from the pulse
   timestamps; check offset continuity across a DREAM run; **fail loudly on
   an ambiguous winner** — the silent tie-break is the actual bug.
2. `clock_qa` absolute check 'pulse_match margin adequate': proposed
   WARN < 10, FAIL ≤ 2 unless scan-verified; test set must include a
   margin-0-but-correct segment.
3. Provenance before any bulk re-slim: `calibration.json` gains `join_shift`,
   lock margin, r-score.

## Recovery recipe

Whole-hour class (41 segments): per segment, wide shift scan → apply the
found shift at the join → record provenance → standard chain. Table
(`shift_predictions.txt`) as cross-check only — the 43.2 s rows under-count
(harmonic). ~one condor evening. Sliver class (66): needs the reformulated
join fit, or δ transferred from the same sub-run's fitted majority side.

## Traps recorded on the way

- Consecutive `BunchNumber`s are n_TOF-**recorded** bunches, not PS pulses —
  bunch-index and wall-clock arithmetic diverge; only direct timestamp
  measurement is trustworthy (this trap fired twice in one hour).
- `PSTime` in official merged files can be corrupt (denormals/NaN in
  run224607); use `Time`.
- A nohup on lxplus does not survive the ssh session's 2FA expiry (systemd
  session cleanup); the failure mode is a 0-byte log. Use condor.
- The full kill-list (12 confident wrong models, each killed by a minutes-long
  measurement) is in the campaign report — it is the argument for the
  loud-failure requirement.
