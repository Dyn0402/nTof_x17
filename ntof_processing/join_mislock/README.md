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
| `arbitration_floor.py` | is there a segment length below which arbitration cannot work? (yes, and it is structural: the count margin grows ~linearly with clusters when schedules differ — run_79/0001 wins by count at 25 clusters — but the intensity discriminant is a STEP function: it is carried by sparse schedule-break events ~one per 10–40 min, so run_96/0001 has r-separation 0.001 through 150 clusters then 4.2σ at 200, and run_86/0001 stays ambiguous to 400. A 5-minute sub-run typically contains no discriminating event at all; the scan route is the standard path for short segments, and the AmbiguousLock message now says so below 200 clusters) |
| `sliver_census.py` | which "sliver" failures are really sliver-class? (36 of 78 have no fitted sibling and are whole-sub-run mislocks in disguise — recoverable; the mystery class is the 42 with a fitted sibling, all ≤ 402 bunches; the run-START orientation hypothesis died 33/33) |
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

Whole-hour class (41 segments): with the 2026-08-12 pulse_match fix
(below), **just re-run the slim** — the join locks correctly on its own and
the standard chain runs unmodified. Acceptance test: run_96/0001×224597
end-to-end through the fixed chain with zero manual input →
**eff 95.4773 %, accidental 0.065 %, full product written**, provenance
`chosen_by: intensity, margin 0, r_sig 3.2` in calibration.json. Sub-runs
the fix declares AmbiguousLock need one shift scan
(`segment_diagnose --span 200`), then re-run with
`match_subrun(..., accept_offset_s=<scanned offset>)`.

## ⚠ The sliver class is NOT recovered, and it is NOT a join error

Tested 2026-08-12 morning: run_79/stat090_0002 × 224573 re-slimmed with the
majority side's δ transferred (`delta_hint_s = 0.829`, the value its own
×224572 segment fitted at margin 507; the sliver's δ-scan accepts it
cleanly) — **and the clock fit still finds no sharp coincidence**. Since the
same sub-run's pulse_match offset is proven right by the ×224572 fit, the δ
is right, and the bunch mapping follows from both, the sliver failure
survives a fully correct join. Corollary: the per-bunch scan's "random
−1…−24 ms lags" on this segment were measured on CORRECT bunch pairings.

**Census (2026-08-12, `sliver_census.py` on the campaign inventory) — the
78 non-OK "sliver" segments are two populations:**

- **42 with a fitted sibling** (the same DREAM sub-run fitted on its other
  side, pinning the shared pulse_match offset): the true mystery class.
  All small — ≤ 402 joined bunches, median 158, overlap median 0.26. The
  pulse_match fix does nothing for these.
- **36 with no fitted sibling** (on 24 sub-runs): the offset was never
  verified, and these include every *large* "sliver" failure (up to 1101
  bunches / 55.6 min) plus all four dark DREAM runs appearing among
  slivers (run_126/128/150/156). Cached v1 offsets betray supercycle
  mislocks: run_132/0007 locked −68.08 s where its run's OK sub-run locked
  +48.32 (−68.08 + 3×39.6 = +50.7); run_139/0007 locked −48.88 vs 0003's
  +67.5 (118.8 apart) — both with the trademark 100 % match / ~4 ms rms.
  **These are whole-sub-run mislocks that happened to straddle a boundary
  and got filed as slivers. They belong in the plain re-run recovery with
  the whole-hour 41.**

The "failed slivers contain n_TOF run STARTS" observation did not survive
the census: mystery-class failures split run_END vs run_START almost
exactly evenly (and OK slivers exist in both orientations). Orientation
carries no information; withdrawn.

Two probes on the mystery class: (a) the same sliver against an
independent processing of 224573 (`reproc/prod_v11/224573` — the local
re-slim already used the official-done merge, which is the v12 lineage) —
if it fits, the anomaly is in the processing, not the raw data; (b)
per-bunch lag + per-bunch refit at the now-known-correct join
(`perbunch_lag.py --delta-hint`) — if each bunch has a findable sharp
offset, sliver recovery is per-bunch, n_TOF-side or not.

## The fix — IMPLEMENTED 2026-08-12 (this commit)

- `ntof_july_analysis/pulse_match.py`: `select_lock()` enumerates every
  candidate lock, arbitrates near-ties (count margin < 10) by the
  intensity-fluctuation correlation (Fisher-z ≥ 3σ), and raises
  `AmbiguousLock`/`NoLock` instead of silently returning a winner. Caches
  are versioned (v2); pre-fix caches rebuild automatically.
  `accept_offset_s=` is the scan-verified override (recorded as
  `chosen_by: 'verified'`; refuses to override a confident contradicting
  selection). Tests: `ntof_july_analysis/test_pulse_match_locks.py`
  (5 synthetic cases incl. the margin-0-but-correct case, which must RAISE,
  never coin-flip).
- `ntof_dream_merge/bunch_join.py`: `delta_hint_s=` (majority-side δ
  transfer within a sub-run), δ-scan ambiguity guard (margin < 3 without a
  hint raises), join provenance in `DataFrame.attrs`.
- `slim_pipeline/slim.py`: `Segment.delta_hint_s`; `calibration.json` gains
  the `join` block (pulse_match offset / margin / chosen_by / r_sig, δ,
  δ-margin, hint) — recovered and originally-clean segments are permanently
  distinguishable.
- Validation: healthy sub-runs bit-identical (run_79/0001 offset +51.918,
  margin 222, by count); both corrupt hours lock correctly (run_96/0001
  +50.722 by intensity 3.2σ; run_86/0001 +42.32 at 12.1σ = the scan's +20
  pulses); margin-5 run_86/0000 correctly raises pending verification;
  `tests/test_clock_qa.py` 29/29 unchanged.
- Still with the matching-QA session: the `clock_qa` absolute check
  'pulse_match margin adequate' (WARN < 10 / FAIL ≤ 2 unless verified),
  reading the new `calibration.json:join` block.

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
