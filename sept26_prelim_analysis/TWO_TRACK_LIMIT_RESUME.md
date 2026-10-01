# Two-track limit — where it stands and how to resume (2026-09-30 evening)

The full record, with every measurement, is `TWO_TRACK_FIT_LOG.md` (the
2026-09-29 and 2026-09-30 entries). The task is `HANDOFF_TWO_TRACK_LIMIT.md`.
The report is `~/x17/sept26_prelim/two_track_limit/report/report.html`
(`python -m sept26_prelim_analysis.make_two_track_limit_report`).
Nothing is committed. Every production change is opt-in, and with the switches
off, output is production's.

## Status 2026-10-01 evening — validation COMPLETE

All 147 jobs of cluster 4334051 returned and are merged; **A and C both pass
the split-ab contract on all seven tags** (C fixed: 0.47 % clean singles split,
no event loses a track, 466 → 234 production tracks not recovered). Report
regenerated. Step 2 under "Next" (fixed + profc bench) was already done
(log, 2026-09-30). Step 1, the F rescan on split-ab, is SUBMITTED. See the
next section.

## F rescan on real triggers — submitted 2026-10-01

One pass of split-ab of the fixed chain, both chambers, all 7 tags × 8 shards
(112 jobs), with `TWO_TRACK_F_LADDER` replaying every F in
300…4800 exactly (`wft.reco.two_track_ladder`; log entry 2026-10-01).
Package `~/x17/two_track_ladder_condor` (`make_two_track_package.py --ladder`),
staged at `lxplus:~/two_track_ladder/`, same EOS inputs and results dir.
When done:

    .venv/bin/python sept26_prelim_analysis/condor/two_track/merge_two_track.py --pkg ~/x17/two_track_ladder_condor
    cat ~/x17/sept26_prelim/intra_bench/split_ab_ladder_{A,C}_7tags/summary_ladder.csv

Read it against the bench ROC (`two_track_limit/report/figures/r3_roc.csv`,
fixed_f0): the lowest F whose real-trigger clean-single split rate is ≤ 0.66 %
with no event losing a track gives the operating point. Its entry at the
matched F must equal `split_ab_fixed_<arm>_7tags`.

## Status 2026-10-01 08:45 (superseded)

Condor cluster 4334051: everything done except 36 `split-ab fixed_C` shards
(C events are busier, ~14 h per shard). **Chamber A passes the contract on all
seven tags** (clean singles split 0.50 % against the ≤ 0.66 % limit, no event
losing a track, 80 more production tracks kept); C passes on the 20 shards in.
Report and log are updated. When C finishes:

    .venv/bin/python sept26_prelim_analysis/condor/two_track/merge_two_track.py
    .venv/bin/python sept26_prelim_analysis/two_track_scratch/contract.py
    .venv/bin/python -m sept26_prelim_analysis.make_two_track_limit_report

## Picking this up elsewhere (written 2026-10-01 before an OS switch)

- **Code and write-up are committed** on branch `two-track-joint-fit`. That
  covers the log, this note, the report generator, the condor package and the
  scratch scripts.
- **Data products are not in git.** They live on this machine's data disk
  (`/media/dylan/data/x17/sept26_prelim/`: `intra_bench/` benches and merged
  split-ab, `two_track_limit/` ladder outputs and report). On another machine,
  point `X17_ROOT` (or `X17_SEPT26_OUT`) at a copy; see
  `sept26_prelim_analysis/paths.py`.
- **Condor does not depend on any local session.** Cluster 4334051's results
  land on EOS (`/eos/user/d/dneff/x17/two_track_limit/results/`), and
  `merge_two_track.py` pulls them from wherever you run it, given ssh to
  lxplus. The package it needs (`jobs.txt`) is in `~/x17/two_track_condor/`
  here, and also at `lxplus:~/two_track_limit/`.
- The local C watcher stops with this session; nothing else was running
  locally.
- The scratch scripts that read session pickles (`profile_pairing.py`'s
  `*.pkl`, `repair_debug*`) must be rerun to regenerate their inputs.

## Answer so far

- **Physical limit** (perfect model, ideal fit): ~one strip pitch. 100 % of pairs
  resolved at every separation ≥ 1.5 mm.
- **Real-track limit** (ideal fit on real overlays): ~2–3 mm, set by how
  imperfectly the forward model describes a real track. The mismatch grows with
  charge.
- **Production** reaches 33–58 % where the ideal fit reaches 94–100 %. The losses
  are the trigger, the fstat scale, local minima in the search, and x/y pairing.
- **Fixed chain** (`WFT_TWO_TRACK_SCALE=two`, `WFT_TWO_TRACK_SEARCH=grid`, no
  trigger, every candidate), at a threshold matched to production's false-split
  rate on real singles (A F = 1200, C F = 2400). Overlay bench, chamber A, event
  level: < 12 mm 19 → 57 %, 12–24 mm 48 → 75 % (coincident); ≥ 24 mm unchanged.
- **x/y pairing by constrained depth profile** (`xy_pairing_*_profc.json`,
  calibrated on stat090_0001): A ≥ 24 mm swaps 12.2 → 7.2 %; C barely moves.
- **Dropped**: a fractional model error ε (no gain at a fixed false-split rate);
  loosening the 1.2 mm guard (it blocks 42 % of false splits on real singles).
- **Flagged for Dylan, outside this task**: q_sum > 10⁶ on 12–33 % of stage-3
  tracks, from unconstrained depth bins. Every charge-derived quantity of
  those tracks is meaningless.

## Validation on lxplus condor — SUBMITTED 2026-09-30 18:30, cluster 4334051

147 jobs, all seven file tags of run_145/stat090_0000:
- bench: `fixed_C_replace`, `fixed_{A,C}_replace_profc` (one job per tag);
- `split-ab`: `fixed_{A,C}` (8 event shards per tag, flavour `tomorrow`) and
  `current_{A,C}` (one job per tag).

Package: `sept26_prelim_analysis/condor/two_track/` (make / run / wrapper /
sub / merge), built into `~/x17/two_track_condor/` and staged at
`lxplus:~/two_track_limit/`. Inputs are on EOS
(`/eos/user/d/dneff/x17/two_track_limit/inputs.tar.gz`, sha256 in
PROVENANCE.txt); results land in `.../two_track_limit/results/`. Smoke-tested
interactively on lxplus under LCG_105 before submission. Code = working tree
at b5657ff + uncommitted changes (listed in PROVENANCE.txt).

    ssh lxplus 'condor_q 4334051 -totals; condor_q 4334051 -hold -af HoldReason'
    .venv/bin/python sept26_prelim_analysis/condor/two_track/merge_two_track.py
    python sept26_prelim_analysis/two_track_scratch/headline.py fixed_C_replace fixed_A_replace_profc fixed_C_replace_profc
    cat ~/x17/sept26_prelim/intra_bench/split_ab_{fixed,current}_{A,C}_7tags/summary.csv

The merge flags missing shards and merges what is there. To resubmit only the
failed ones: `condor_release` for held jobs, or a jobs file of the missing lines
with `condor_submit two_track.sub jobfile=<file>`.

The local run of `split-ab fixed A`, one tag (tag 000, started 14:08), was left
to finish as an exact cross-check of condor's tag-000 shards; the local chain's
later steps were stopped (condor covers them).

## Local validation chain — superseded by condor, kept for reference

    sept26_prelim_analysis/two_track_validate.sh 12 > validate.log 2>&1

Each step is skipped when its output exists under
`~/x17/sept26_prelim/intra_bench/`. A step killed midway restarts from its
beginning. Steps, per chamber: `fixed_<arm>_replace/` (bench),
`split_ab_fixed_<arm>/` and `split_ab_current_<arm>/` (real-trigger contract,
one file tag).

State at writing: A bench done; A `split-ab` fixed running (started 14:08).

Scoring:
- bench — `python sept26_prelim_analysis/two_track_scratch/headline.py fixed_A_replace`
  (compare `pairing_rescue16_two_final_replace`);
- split-ab — `summary.csv` in each directory; the contract is clean singles split
  ≤ 0.66 %, events losing a track ≈ 1 in 3 250, unsplit events bit-identical.

## Scratch tools

`two_track_scratch/` holds the one-off scripts the log cites (`headline.py`,
`selfnoise.py`, `search_test.py`, `profile_pairing.py`, `repair_debug*.py`,
`toy_asimov.py`). They write next to themselves and some read pickles made in
the session scratchpad, so rerun them rather than expect their inputs to be
there.

## Next

1. Finish the validation. If `split-ab` shows the fixed chain splitting clean
   singles more often than production, the matched thresholds are wrong for real
   triggers (they were set on overlay donors): rescan F on `split-ab`.
2. Combine the fixed chain with `profc` pairing and rerun the bench.
3. ~~Move the long runs to lxplus condor~~ — done, see above.

## lxplus condor — notes from the feasibility check

Checked 2026-09-30: ssh to lxplus works, the queue is idle, and EOS holds
everything:
- waveforms, hits and run config under `/eos/experiment/ntof/data/x17/july_beam/runs/`
  (`ntof_tracking/condor/run_beam_job.py` already fetches them per (arm, tag)
  with xrdcp and redirects `BASE_PATH`);
- stage-2 full-pass products, 12 929 per-(run, sub-run, arm, tag) tarballs, in
  `/eos/user/d/dneff/x17/sept26_fullpass/`.

The shape: one job per (arm, file tag) for `split-ab`, and per (arm, tag) shards
for the bench and `two_track_limit real`. Ship the code with `git archive`, plus
bundles, stage-3 tracks and pairing JSONs (a few MB), following
`make_beam_package.py` / `stage2_fullpass.sub`. Their lessons carry over:
2 CPUs per slot, xrdcp output to EOS from the job, sharded log dirs.
`split-ab` could then run on all tags and sub-runs rather than one tag.
