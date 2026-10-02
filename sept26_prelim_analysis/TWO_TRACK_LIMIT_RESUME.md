# Two-track limit — where it stands and how to resume

**Updated 2026-10-02.** This is the handoff. The task is
`HANDOFF_TWO_TRACK_LIMIT.md`. The full record, with every measurement, is
`TWO_TRACK_FIT_LOG.md` (entries 2026-09-29 → 2026-10-01). The report is
`~/x17/sept26_prelim/two_track_limit/report/report.html`, built by
`python -m sept26_prelim_analysis.make_two_track_limit_report`.

Everything is committed and pushed on branch `two-track-joint-fit`. Every
production change is **opt-in**. With the switches off, output is production's,
bit for bit.

## In one paragraph

The physical limit for two tracks in one plane is about one strip pitch. The
real-track limit is about 2–3 mm, set by forward-model mismatch. Production was
far from both. The **fixed chain** (`WFT_TWO_TRACK_SCALE=two`,
`WFT_TWO_TRACK_SEARCH=grid`, no trigger, every candidate) with **profile x/y
pairing** closes much of the gap. On the full overlay bench, coincident pairs:
A < 12 mm goes 19 → 58 % and 12–24 mm goes 48 → 79 %; C goes 36 → 49 % and
57 → 69 %. It **passes the split-ab contract on real triggers in both chambers,
all seven tags**. One question is still open: is C's threshold (F = 2400) too
strict? A one-pass rescan of F on real triggers is running on condor to answer
it.

## Where it stands

| step | state |
|---|---|
| Physical / real-track limit (R1–R4) | done — report sections R1–R4 |
| Fixed chain at matched F, overlay bench, 7 tags | done — cluster 4334051 |
| Fixed + profc pairing, overlay bench, 7 tags | done — cluster 4334051 |
| split-ab contract, fixed vs current, 7 tags | **done, A and C pass** — cluster 4334051 |
| F rescan on real triggers (split-ab ladder) | **running** — cluster 4348153, submitted 2026-10-01 22:15 |
| Choose operating F per chamber | waiting on the rescan |
| Ship to production (bundles, condor env, re-pass) | **Dylan's decision**, not started |

### The contract result (cluster 4334051, merged 2026-10-01)

Real triggers are re-reconstructed by both chains on the **same triggers** and
matched to the frozen full pass. The contract: clean single muons split at most
0.66 %, no event losing a track.

| | A current | A fixed | C current | C fixed |
|---|---:|---:|---:|---:|
| triggers | 22 434 | 22 434 | 22 788 | 22 788 |
| clean singles split | 0.33 % | **0.50 %** | 0.66 % | **0.47 %** |
| events losing a track | 0 | **0** | 0 | **0** |
| production tracks not recovered | 485 | **405** | 466 | **234** |
| events split / gaining a track | 272 / 85 | 346 / 70 | 913 / 300 | 207 / 46 |

At F = 2400, C's fixed chain splits far fewer real events than current
production does. It is inside the contract with room to spare. The bench says a
lower F buys pairs: C fixed resolves 58 % at F = 2400, 69 % at 1200 and 70 % at
1000. The rescan measures how low F can go on real triggers.

### The F rescan (cluster 4348153, running)

This is one pass of split-ab of the fixed chain: A and C, 7 tags × 8 event
shards, 112 jobs. C shards take about 14 h. Every F in
300, 400, 600, 800, 1000, 1200, 1600, 2000, 2400, 3200, 4800 is replayed
**exactly** from the same attempts.

How it can be one pass: neither the attempts nor their `fstat` depend on the
threshold; only acceptance does. So `wft.reco.two_track_ladder` (worker option
`TWO_TRACK_F_LADDER`) keeps both children of every attempt. For each F it
rebuilds the candidate lists, applies the worker's lost-track revert, and runs
the selector. The primary F stays the matched one (A 1200, C 2400), so the
primary output reproduces cluster 4334051.

The replay was verified exact before submission on 52 C triggers: against a
primary run at 2400, and against one at 300 that had 3 accepted splits. It
matched event by event and track by track (log, 2026-10-01).

## Resume here

1. **Check the cluster.**

       ssh lxplus 'condor_q 4348153 -totals; condor_q 4348153 -hold -af HoldReason'

   Held jobs: `condor_release`. To resubmit only some jobs: write a jobs file of
   the missing lines from `~/x17/two_track_ladder_condor/jobs.txt` and run
   `condor_submit two_track.sub jobfile=<file>` in `lxplus:~/two_track_ladder/`.
   The merge lists any missing shards.

2. **Merge and read.**

       .venv/bin/python sept26_prelim_analysis/condor/two_track/merge_two_track.py --pkg ~/x17/two_track_ladder_condor
       cat ~/x17/sept26_prelim/intra_bench/split_ab_ladder_{A,C}_7tags/summary_ladder.csv
       .venv/bin/python -m sept26_prelim_analysis.make_two_track_limit_report

   The report's **Operating point** section reads the result:
   - it puts the real-trigger ladder next to the bench (`r3_scan`, fixed_f0, on
     the same F grid);
   - it picks the lowest F meeting the contract, by point estimate, with the
     90 % Clopper–Pearson upper limit shown;
   - it flags a pick at the bottom of the ladder.

3. **Check before believing it.** Each chamber's ladder row at its matched F
   (A 1200, C 2400) must equal `split_ab_fixed_<arm>_7tags/summary.csv`:
   events split, gaining a track, not recovered. If it does not, the replay is
   wrong; stop.

4. **Write it up.** Add a log entry with the table, update this note and the
   report verdict. If the pick moves C's F, the bench
   (`fixed_C_replace_profc`) at the new F is already in `r3_scan`. A full
   event-level bench at the new F is a single `bench` job set; see
   `make_two_track_package.py` `job_list`.

## Decisions for Dylan

- **The operating F per chamber**, from the rescan. The contract sample is
  small: 0.66 % of C's 1 057 clean singles is 7 events, so neighbouring F
  values are not statistically distinct. The pick is a threshold choice, not a
  measurement.
- **Whether to ship.** That means bundles carrying the profc `xy_pairing`, the
  fixed-chain switches plus the chosen F in the condor environment, the rescue
  floor (`WFT_SIG_FLOOR_LOCAL_MM=16`, mode `rescue`), and a full re-pass.
  Compute is not a constraint.
- **Outside this task, flagged:** q_sum > 10⁶ on 12–33 % of stage-3 tracks,
  from unconstrained depth bins. Every charge-derived quantity of those tracks
  is meaningless.

## What is not established

- **One sub-run** (run_145 stat090_0000), chambers A and C. B and D are
  untested. Everything is on the post-23-July (noisy) configuration.
- **Real triggers have no truth.** split-ab bounds the cost of a threshold. The
  gain comes only from overlays of clean donors, and real pairs are busier.
- **Bench donors split more readily than clean singles on real triggers** (C at
  2400: 1.2 % vs 0.47 %). This is why the matched thresholds may be too strict,
  and why the real-trigger rescan, not the bench, sets F.
- **Parallel co-located tracks are degenerate at any Δt.** They stay lost and
  must be carried as inefficiency.
- **Dropped, with reasons in the log:** a fractional model error ε (no gain at a
  fixed false-split rate) and a looser 1.2 mm guard (it blocks 42 % of false
  splits on real singles).

## Reference

**Data products are not in git.** They are on the data disk
(`/media/dylan/data/x17/sept26_prelim/`, linked as `~/x17/sept26_prelim`):
- `intra_bench/`: benches, merged split-ab runs `split_ab_*_7tags/`, and
  `contract_fixed_vs_current.csv`;
- `two_track_limit/`: R1–R4 outputs, `r3_scan.parquet` and the report.

On another machine, point `X17_ROOT` (or `X17_SEPT26_OUT`) at a copy; see
`paths.py`.

**Condor.** Package code: `sept26_prelim_analysis/condor/two_track/`.

| | built into | staged at |
|---|---|---|
| validation (`make_two_track_package.py`) | `~/x17/two_track_condor/` | `lxplus:~/two_track_limit/` |
| F rescan (`--ladder`) | `~/x17/two_track_ladder_condor/` | `lxplus:~/two_track_ladder/` |

- Both read `/eos/user/d/dneff/x17/two_track_limit/inputs.tar.gz` (identical
  content for both builds, checked).
- Both write to `.../two_track_limit/results/`.
- `merge_two_track.py --pkg <package>` rsyncs the results and merges only that
  package's jobs.
- Jobs run under LCG_105 with 2 CPUs and 6 GB each, and push to EOS from the job.
- No local session is needed while they run. Non-interactive `ssh lxplus` works
  once there is a valid Kerberos ticket.

**Commands and checks.**
- `python sept26_prelim_analysis/two_track_scratch/headline.py <variant>`:
  event-level bench headline.
- `two_track_scratch/contract.py`: fixed vs current on the same triggers.
- `python -m sept26_prelim_analysis.two_track_limit scan`: re-applies any
  threshold offline to the R3 threshold-0 runs.
- The other scratch scripts read session pickles, so rerun them rather than
  expect their inputs to exist.

**Superseded:** the local validation chain `two_track_validate.sh` is kept for
reference; condor replaced it.
