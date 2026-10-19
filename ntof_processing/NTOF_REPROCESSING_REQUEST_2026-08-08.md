# Request to n_TOF: 41 X17 runs still to process

> # ⛔ SUPERSEDED 2026-08-10 — DO NOT SEND THIS VERSION
>
> **The central claim below — "these runs have no processed output at all" — is
> wrong for 28 of the 41.** Their reconstruction finished; a complete,
> contiguous, v12 partial set is sitting in
> `/eos/experiment/ntof/processing/official/completed/<run>/`, verified
> byte-identical in `history` to `done/run224572.root`. Only the MERGE is
> missing, which is why `done/` looks empty for them.
>
> That is 90.1 of the 116.6 blocked beam hours. **Only 13 runs genuinely need
> processing**: 224688-224698 (the pass stopped 08-07 19:56) and 224649/224650
> (tape). The right ask of n_TOF is *re-run the merge*, plus a note that
> `done/run224405.root` and `done/run224667.root` are zero-byte failed merges.
>
> Evidence, audit scripts and the corrected picture:
> [`skip_diagnosis/README.md`](skip_diagnosis/README.md).
> §"A clue: only large runs were skipped" below is still a valid *association*,
> but it is an association about the merge step, not the reconstruction.

**From the X17 / DREAM group (Dylan Neff), 2026-08-08.** Contact: dneff@cern.ch.

Thank you — the pass you ran on 2026-08-05 to 08-07 is **exactly right**, and we
have verified it: every X17 file in
`/eos/experiment/ntof/processing/official/done/` carries the UserInput we
proposed, parameter for parameter and template for template. We diffed the
`history` string in `run224572.root` against our own copy: identical on all 14
detector rows and all 26 pulse-shape filenames.

**325 runs are done. 41 are still missing, and we need them.** They
block **117 hours** of X17 beam time — 46% of our
campaign, none of which has any processed output at all.

---

## What happened, as far as we can see

The pass ran from **5 August** and the last file landed **7 August at 19:56**.
Nothing has been written since — about a day as we write this. That stop cleanly
explains the tail: **11 runs (224688–224698) are simply after the
point where `done/` ends**, and they cover our last two days of data taking
(DREAM runs 150–156).

What it does **not** explain is the rest. There are **66 runs missing
from inside 224300–224687**, scattered through the range rather than
clustered at either end, and **30** of those overlap X17 beam time.
We cannot see a reason for them from the outside. One partial correlation: 
35 of the 66 in-range gaps no longer have their stream1
staged on the EOS disk, which would explain a skip if the pass reads from disk —
but the other 31 do still have it and were skipped anyway.

**If you know why those were passed over, we would like to hear it** — it is the
one piece we cannot reconstruct, and it would tell us whether re-running them is
straightforward or whether something about them is broken.

## A clue: only large runs were skipped

We looked for anything that distinguishes them. Directory structure is identical
on both sets — `stream0` + `stream1`, every file `.finished`, no stragglers. An
output-size cap does not fit: it would have to sit below 21 GB, and 42 processed
runs exceed that. Position in the run range says nothing; the gaps are scattered.

**Raw size fits, and not subtly.** Of the 135 in-range runs whose stream1 is
still staged:

| raw TB | runs | skipped | rate |
|---|---|---|---|
| 0.00–0.05 | 26 | 0 | 0% |
| 0.05–0.15 | 14 | 0 | 0% |
| 0.15–0.25 | 16 | 0 | 0% |
| 0.25–0.35 | 7 | 0 | 0% |
| 0.35–0.45 | 13 | 5 | 38% |
| 0.45–0.55 | 36 | 12 | 33% |
| 0.55–0.65 | 14 | 9 | 64% |
| 0.65–0.75 | 6 | 3 | 50% |
| 0.75–1.00 | 3 | 1 | 33% |

**Below 0.35 TB nothing was ever skipped — 0 of 63.** At or above it
30 of 72 were, and the rate keeps climbing with size. That is
the shape of a resource limit a large job sometimes misses and sometimes makes —
wall clock, memory or scratch — rather than a rule that rejects a run outright.
If it were deterministic the big runs would all have failed; they did not.

The control: of the 11 runs missing from *after* 224687, three
(224689, 224694, 224698) are below 0.35 TB, a band in which the pass never skipped
anything. So those are missing because the pass stopped, not for this reason.
Two mechanisms, cleanly separated.

We cannot see your job configuration, so this is an association, not a
diagnosis — but if there is a per-job limit worth raising for the re-run, that
size distribution is where we would look.

## What we need

| | |
|---|---|
| runs | the 41 listed below |
| UserInput | the same one already used — `UserInput_2026_EAR2_X17_v4.h` |
| output | the same place, `/eos/experiment/ntof/processing/official/done/` |
| raw | 39 still have stream1 staged on disk under `/eos/experiment/ntof/DAQ/2026/EAR2/X17_measurement/<run>/stream1/`; 2 (224649, 224650) will need a recall from tape |
| order | **whatever suits your queue** — we want all of them |

### These runs have no processed output at all

Not just no v12 — **nothing**. There is no file for any of them anywhere under
`/eos/experiment/ntof/processing/`, and none under the earlier `v2` processing
either. `done/` keeps older output (files back to April 2025, including 141 from
July 2026), but in 224300–224687 every one of the 325 files present is
dated 5–7 August. A run processed under v2 and then skipped by this pass would
still be sitting there with its old timestamp; none is.

### The one naming point that will confuse anyone checking

The file is called `UserInput_2026_EAR2_X17_v4.h` and its content is what our
group tracks internally as **v12_liqpileup**. Both names refer to the same
thing; the header comment inside the file says so. We mention it only because
our own repository also has a *different* file called `v4`, and we do not want
anyone to reconcile the two by filename.

---

## The 41 runs

`beam h blocked` is how much X17 DREAM beam time depends on that n_TOF run.
A `bracketed` window means the run has neither a processed file nor staged
stream1, so we placed it between its nearest measurable neighbours by run
number — coarse, but enough to show it overlaps beam.

| n_TOF run | start (UTC) | hours | window | stream1 | raw files | raw TB | DREAM runs affected | beam h blocked |
|---|---|---|---|---|---|---|---|---|
| **224573** | 2026-07-26 18:54 | 2.9 | measured | yes | 156 | 0.46 | run_79 | 2.9 |
| **224576** | 2026-07-27 01:07 | 2.6 | measured | yes | 150 | 0.43 | run_79 | 2.6 |
| **224577** | 2026-07-27 03:46 | 2.9 | measured | yes | 166 | 0.45 | run_79 | 2.9 |
| **224597** | 2026-07-28 11:38 | 3.9 | measured | yes | 161 | 0.55 | run_96 | 3.7 |
| **224606** | 2026-07-29 08:30 | 2.9 | measured | yes | 164 | 0.54 | run_102 | 2.9 |
| **224614** | 2026-07-29 17:26 | 3.4 | measured | yes | 188 | 0.60 | run_104 | 3.4 |
| **224617** | 2026-07-30 03:18 | 3.5 | measured | yes | 202 | 0.54 | run_104 | 3.5 |
| **224618** | 2026-07-30 06:52 | 2.7 | measured | yes | 148 | 0.48 | run_104 | 2.7 |
| **224624** | 2026-07-31 01:08 | 3.2 | measured | yes | 184 | 0.50 | run_108 run_110 | 3.2 |
| **224625** | 2026-07-31 04:24 | 2.7 | measured | yes | 162 | 0.47 | run_110 | 2.7 |
| **224628** | 2026-07-31 08:04 | 3.1 | measured | yes | 167 | 0.55 | run_112 run_114 | 3.1 |
| **224629** | 2026-07-31 11:13 | 3.0 | measured | yes | 156 | 0.59 | run_114 | 3.0 |
| **224632** | 2026-07-31 15:42 | 4.3 | measured | yes | 250 | 0.88 | run_116 | 4.3 |
| **224635** | 2026-08-01 01:58 | 3.1 | measured | yes | 178 | 0.55 | run_116 | 3.1 |
| **224637** | 2026-08-01 08:41 | 3.4 | measured | yes | 196 | 0.63 | run_116 | 3.4 |
| **224638** | 2026-08-01 12:04 | 3.2 | measured | yes | 168 | 0.64 | run_116 | 3.2 |
| **224639** | 2026-08-01 15:17 | 3.1 | measured | yes | 178 | 0.69 | run_116 | 3.1 |
| **224640** | 2026-08-01 18:23 | 1.9 | measured | yes | 118 | 0.38 | run_116 | 1.9 |
| **224649** | 2026-08-02 16:59 | 2.5 | bracketed | no -- recall from tape | — | 0.00 | run_118 run_120 | 0.3 |
| **224650** | 2026-08-02 16:59 | 2.5 | bracketed | no -- recall from tape | — | 0.00 | run_118 run_120 | 0.3 |
| **224652** | 2026-08-02 19:37 | 3.7 | measured | yes | 211 | 0.70 | run_122 run_124 | 3.5 |
| **224653** | 2026-08-02 23:22 | 3.0 | measured | yes | 198 | 0.61 | run_124 | 3.0 |
| **224654** | 2026-08-03 02:26 | 2.9 | measured | yes | 178 | 0.54 | run_124 | 2.9 |
| **224655** | 2026-08-03 05:23 | 2.8 | measured | yes | 173 | 0.60 | run_124 | 2.8 |
| **224660** | 2026-08-03 15:31 | 5.3 | measured | yes | 176 | 0.67 | run_128 run_130 run_132 | 5.0 |
| **224661** | 2026-08-03 20:49 | 3.5 | measured | yes | 156 | 0.55 | run_132 | 3.5 |
| **224666** | 2026-08-04 19:56 | 3.7 | measured | yes | 163 | 0.54 | run_139 | 3.6 |
| **224667** | 2026-08-04 23:39 | 3.8 | measured | yes | 163 | 0.53 | run_139 | 3.8 |
| **224671** | 2026-08-05 15:33 | 3.2 | measured | yes | 154 | 0.64 | run_147 | 3.2 |
| **224673** | 2026-08-05 21:44 | 3.0 | measured | yes | 135 | 0.49 | run_147 | 3.0 |
| **224688** | 2026-08-07 11:02 | 3.3 | measured | yes | 173 | 0.76 | run_150 | 3.3 |
| **224689** | 2026-08-07 14:20 | 0.3 | measured | yes | 19 | 0.08 | run_150 | 0.3 |
| **224690** | 2026-08-07 16:20 | 2.9 | measured | yes | 157 | 0.68 | run_152 | 2.9 |
| **224691** | 2026-08-07 19:15 | 2.7 | measured | yes | 150 | 0.61 | run_152 | 2.7 |
| **224692** | 2026-08-07 22:27 | 3.3 | measured | yes | 153 | 0.61 | run_154 | 3.1 |
| **224693** | 2026-08-08 01:49 | 3.2 | measured | yes | 144 | 0.52 | run_154 | 3.2 |
| **224694** | 2026-08-08 05:02 | 1.0 | measured | yes | 42 | 0.16 | run_154 | 1.0 |
| **224695** | 2026-08-08 06:26 | 3.2 | measured | yes | 140 | 0.56 | run_156 | 3.2 |
| **224696** | 2026-08-08 09:38 | 2.9 | measured | yes | 141 | 0.60 | run_156 | 2.9 |
| **224697** | 2026-08-08 12:35 | 3.0 | measured | yes | 154 | 0.67 | run_156 | 3.0 |
| **224698** | 2026-08-08 15:35 | 0.3 | measured | yes | 47 | 0.20 | run_156 | 0.3 |
| | | | | | **6119** | **21.2** | | **117** |

Machine-readable: `missing_runs_2026-08-08.csv`.

## Where the campaign stands

| | |
|---|---|
| processed and verified | 325 runs, 224300–224687 |
| still needed | 30 inside that range, 11 after its end |
| X17 campaign | DREAM runs 77–156, 2026-07-26 to 08-08, 282 beam sub-runs |
| processed today | 133 h of 253 h (52%) |
| not processed | 120 h (48%) |
| after these 41 | ~253 h (essentially all of it) |

## The 36 in-range gaps we are NOT asking about

Of the 66 runs missing from 224300–224687, we are asking for the
30 that overlap X17 beam time. The other 36 were live
while DREAM was not, so they block nothing for us:

```
224340 224378 224395 224396 224397 224405 224451 224452 224453 224454
224461 224462 224465 224475 224481 224488 224497 224499 224500 224502
224508 224510 224513 224519 224525 224535 224541 224543 224546 224547
224549 224557 224558 224563 224564 224565
```

We mention them only because they are part of the same unexplained set — if they
were skipped for a reason that also applies to the ones we are asking for, that
would be worth knowing.

## Why it matters to us

We key every n_TOF hit to a DREAM trigger through a time calibration that is
fitted **per (DREAM run, n_TOF processing) pair** and does not transfer between
processings. Mixing a v12 run with an older processing inside one DREAM run is
not an option: the plastic flash identification alone differs by 37–85 % of
bunches, and our own v11 differs from v12 by 14–21 % in liquid hit yield. So a
DREAM run is either fully covered by this processing or it waits.
