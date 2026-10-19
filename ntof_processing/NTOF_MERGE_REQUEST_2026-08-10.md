# Correction and a much smaller request: 55 X17 runs need only the merge

**From the X17 / DREAM group (Dylan Neff), 2026-08-10.** Contact: dneff@cern.ch.
**Supersedes `NTOF_REPROCESSING_REQUEST_2026-08-08.md`.** *Draft — not sent.*

---

> ## Status, 2026-08-11 evening: ask 1 is essentially done, ask 2 is not
>
> Whether or not this draft went out, n_TOF worked the backlog on 08-10/08-11.
> Of the **55 unmerged runs** listed at the bottom:
>
> | | runs | |
> |---|---|---|
> | now **MERGED** | **27** | done |
> | **being reprocessed right now** | **23** | `completed/` emptied and refilling |
> | still PARTIALS_ONLY | 3 | 224451, 224452, 224453 |
> | still zero-byte in `done/` | 2 | 224405, 224667 |
>
> They chose to **re-run the reconstruction** rather than only the merge, and
> one run that was already merged — **224560** — was wiped and re-queued with
> it, so its 31.8 GB merged file is gone for now.
>
> **Ask 2 (224688-224718) has not been started and does not look like it will
> be.** All 31 are still `RAW_ONLY`; the pass spent 08-11 below 224688 and then
> moved on to 224719+, a different experiment. We have 30 of the 31 processed
> and on the ntof disk, and they check out — see
> [`FINDINGS_2026-08-11_official_ledger.md`](FINDINGS_2026-08-11_official_ledger.md).
> If this draft is ever sent, ask 2 should become an offer to hand over our
> output rather than a request.
>
> Current state for all 445 runs:
> [`campaign_qa/results/ledger_2026-08-11.csv`](campaign_qa/results/ledger_2026-08-11.csv).

---

## First, a correction to what we sent you on 8 August

We asked you to reprocess 41 runs, and said they had "no processed output at
all". **That was wrong for 28 of them, and the error was ours.** We checked
`processing/official/done/` and concluded from its absence that the runs had not
been processed. We had not looked in `processing/official/completed/`.

They are all there. The 5-7 August pass reconstructed them successfully; what is
missing is only the **merge**. We are sorry for the noise — it would have cost
you about 117 hours of queue time to redo work that was already done.

## What we verified before writing this

For each of the 28 runs listed below, in `official/completed/<run>/`:

| check | result |
|---|---|
| partials contiguous `run<run>_0001 … _NNNN`, no gaps | pass, all 28 |
| count equals `ceil(raw stream1 files / 4)` | pass (224667 predates the 07-08 split change and matches `ceil(raw/10)`) |
| `history` string identical to `done/run224572.root` | pass — md5 `e51a3ef3dc0b32c1803e59ad18639a7c`, 10 696 bytes, all 28 |
| partials open, 16 trees, sane entry counts | pass on first/middle/last of each |

So they are the same processing as the 325 that merged, not a degraded variant.
Total 1 184 partials, 899 GB.

## Two zero-byte files you will want to know about

```
/eos/experiment/ntof/processing/official/done/run224405.root   0 bytes  2026-08-05
/eos/experiment/ntof/processing/official/done/run224667.root   0 bytes  2026-08-05
```

Both have a complete 17-partial set in `completed/`. A failed merge leaving an
empty file is worth knowing about generally, because anything that tests for
existence rather than size will treat those runs as done. We hit exactly that in
our own code and now check the size.

## The whole campaign, since we have now looked properly

We inventoried all 445 X17 run directories rather than just the 41 we had asked
about:

| state | runs | |
|---|---|---|
| merged in `done/` | 359 | fine |
| **processed, not merged** | **53** | in `completed/`, usable |
| **zero-byte `done/` file** | **2** | 224405, 224667 |
| raw staged, nothing processed | 31 | 224688-224718 |

In beam time, over the 289 h our DREAM runs cover (data taking ended 08-10):
**223 h (77 %) is available today**, 61 h (21 %) needs processing, 5 h had no
n_TOF run live. The 61 h is entirely DREAM run_150 onward — i.e. everything
after the pass stopped.

## What we are actually asking for

**1. Re-run the merge for the 55 unmerged runs.** Not the reconstruction.

**2. Process 224688-224718** — 31 runs, 12.76 TB of raw, all still staged. These
are after the point where the pass stopped (last output 08-07 19:56) and cover
our final three days, DREAM runs 150-162. **We have started doing these
ourselves** with the same `UserInput_2026_EAR2_X17_v4.h`, so if it is easier for
you to skip them, say so and we will finish and hand you the output.

**3. Please drop 224649 and 224650** from our 8 August list. We asked you to
recall them from tape; they have no DAQ directory at all, and the window we gave
them was a guess between neighbouring run numbers. Our mistake.

## One observation, offered rather than asserted

We cannot see your job configuration, so this is an association only. Over the
413 runs that were processed at all, merge success depends on **output size**:

| output (sum of partials) | runs | merged | rate |
|---|---|---|---|
| < 20 GB | 275 | 275 | **100 %** |
| 20-25 GB | 17 | 13 | 76 % |
| 25-30 GB | 42 | 31 | 74 % |
| 30-35 GB | 65 | 37 | 57 % |
| 35-40 GB | 8 | 2 | 25 % |
| > 40 GB | 6 | 1 | 17 % |

Below 20 GB nothing ever failed. Above it the rate falls with size, but not
sharply enough to be a rule: merged runs run to 42.0 GB and unmerged start at
20.3 GB, so the two overlap almost entirely. That looks like a limit being
approached rather than a threshold being enforced — and the two zero-byte files
say at least some merges started and died.

We also ran one of the missing runs ourselves (224688, 44 partials, 34 GB) with
`RunProcessing.sh`. The 44 processing jobs all succeeded; the merge node died in
**five minutes**:

```
max total download bytes exceeded (max=1024 MB, this file=764 MB)   [49 files]
Job removed by SYSTEM_PERIODIC_REMOVE due to disk usage exceeded allowed max
Disk (KB): 58332564 used, 3000 requested
```

That is condor's file-transfer cap and the merge job's disk request, not its
one-hour wall clock. We do not claim this is what happened on your side — you
merged plenty of runs whose partials also exceed 1 GB, so your merge must be
configured differently from what the script generates for us. But if the
transfer budget or `request_disk` on the merge node is the binding constraint,
raising it may be all the re-run needs.

**If instead the large runs are deliberately left unmerged, that is completely
fine by us** — we read the partials directly and they work (see above). In that
case all we would ask is that it be recorded somewhere we can see, because from
outside `done/` an unmerged run is indistinguishable from an unprocessed one,
and that is what cost us the wrong request on 8 August.

## The runs

The 28 from our 8 August list that are merely unmerged:

```
224573 224576 224577 224597 224606 224614 224617 224618 224624 224625
224628 224629 224632 224635 224637 224638 224639 224640 224652 224653
224654 224655 224660 224661 224666 224667 224671 224673
```

A further 27 unmerged runs fall outside that list (they were live while DREAM
was not, so they block nothing for us, but they are in the same state):

```
224405 224451 224452 224453 224454 224461 224462 224481 224488 224499
224500 224502 224508 224510 224513 224519 224525 224541 224543 224546
224547 224549 224557 224558 224563 224564 224565
```

Machine-readable state for all 445 runs:
`skip_diagnosis/inputs/inventory_2026-08-10.csv`.

## Why it matters to us, unchanged

We key every n_TOF hit to a DREAM trigger through a time calibration fitted per
(DREAM run, n_TOF processing) pair, which does not transfer between processings.
A DREAM run is therefore either fully covered by this processing or it waits.
With the unmerged runs readable we recover **90 of the 117 hours** we wrote to
you about, and the remaining gap is the 224688-224718 tail.
