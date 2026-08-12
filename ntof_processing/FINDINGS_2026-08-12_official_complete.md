# n_TOF official processing — the campaign closed overnight, one run lost

Refresh of [`FINDINGS_2026-08-11_official_ledger.md`](FINDINGS_2026-08-11_official_ledger.md)
after n_TOF's manual resubmission of the tail block on the evening of 08-11.
Scans: `campaign_qa/settled_runs.py`, `campaign_qa/completed_ledger.py`,
`campaign_qa/compare_identity.py`, all 2026-08-12 morning.

## Verdict

**444 of 445 runs now have a complete official product.** The tail block
224688-224718 that n_TOF had never processed was submitted manually on the
evening of 08-11 and finished by 20:26; nothing has moved since. 224576, the
run we could only hold off-recipe, finished too.

**Run 224526's official product covers 13.3 % of its beam bunches.** The run
itself is *not* lost — the complete raw is on CTA tape and it is recoverable.
See §4.

> **Correction (same day).** This document first called 224526 "permanently
> lost", on the strength of `eos fileinfo` reporting layout `replica … d2::t0`
> for the DAQ copy. That was the wrong test: it describes the layout of *that
> disk copy*, not whether an archive copy exists. Per
> [`docs/Lxplus _ NTOF _ TWiki.pdf`](docs/Lxplus%20_%20NTOF%20_%20TWiki.pdf)
> §3.1, n_TOF raw is written to **two** locations — EOS `DAQ/`, which holds it
> for **two weeks only**, and CTA tape at `/eos/ctapublicdisk/archive/ntof/`,
> which is permanent. Verified: all 445 runs of the campaign are on tape,
> complete and contiguous.

Consequences for us: **the handoff scripts are no longer needed**, `prod_v11`
can be retired in full, and our stalled 224709 job can be abandoned.

## 1. The tail block completed

All 31 runs of 224688-224718 plus 224576 — 32 runs — check out on every test:

| test | result |
|---|---|
| partials present, contiguous `0001..N` | 32 / 32 |
| `N == ceil(raw stream1 files / 4)` | 32 / 32 (224576's raw is gone, so exempt) |
| `history_<run>.root` present | 32 / 32 |
| bunch coverage — last partial reaches the run's last bunch | 32 / 32 |
| recipe fingerprint | 32 / 32 identical |

The recipe is `UserInput_2026_EAR2_X17_v4.h`, fingerprint `e737ed0da496`, which
is **the same fingerprint as the pre-existing official runs (224297, 224572,
224632) and as our own `prod_v12`**. So the block was processed with the
campaign recipe, not something new.

My first read of the partial counts looked alarming — 224689 with 5 partials,
224710 with 3 — but those are simply short runs (19 and 10 raw files). Every
count matches `ceil(raw/4)` exactly.

## 2. Two independent processings of the same raw agree hit for hit

The block now exists in both processings, so the equivalence argument that had
to stand in for a diff can be replaced by the diff itself. Spot-checked on two
runs (`identity_224705_2026-08-12.json`, `identity_224710_2026-08-12.json`):

| tree | 224705 | 224710 |
|---|---|---|
| WAL A-D | IDENTICAL | IDENTICAL |
| PSS A-D | IDENTICAL | IDENTICAL |
| SILI, PKUP | IDENTICAL | IDENTICAL |
| LIQ A-D | same hit count, `afast` differs on 3-6 hits (0.00-0.02 %) | same hit count, `afast` differs on 1-8 hits (0.00-0.02 %) |

Identical means all 22 per-hit columns exact on every hit of the compared
bunches. This is the **same signature as run 224572** on 08-11 — the residual is
confined to `afast` on a handful of LIQ hits and the outliers are enormous
(1.6e9), which reads as a fit that occasionally diverges rather than a
configuration difference. Worth a look someday; it does not affect hit finding.

## 3. Bookkeeping

`completed_ledger.py`, coverage from the files themselves, merge ignored:

| | official | ours |
|---|---|---|
| COVERED | 442 | 30 |
| SHORT | 2 | 1 (224709) |
| MERGED_ONLY | 1 | 0 |
| OFF_RECIPE | 0 | 7 (`prod_v11`, 224573-224579) |
| ABSENT | 0 | 407 |

The three official exceptions resolve on inspection:

* **224566** — partials on disk are `0003 0005 0007 0008`, non-contiguous, so the
  coverage test fails. The merged file is complete: all 1768 beam bunches carry
  hits. The partials were **cleaned up after the merge**.
* **224569** — no partials at all, same reason. Merged file covers all 3053
  bunches.
* **224526** — genuinely short. §4.

So the coverage test's SHORT/MERGED_ONLY states have a benign cause we had not
seen before: **post-merge partial cleanup**. A run with a healthy merged file and
gapped partials is finished, not broken. The test needs the merged file as a
fallback, which it does not currently do.

## 4. Run 224526's official product is 87 % short — and it is recoverable

`official/done/run224526.root` is 4.0 GB where its neighbours are 27-33 GB. Of
3313 beam bunches, **440 carry hits — 13.3 %**.

The cause is visible in the raw directory. 224526 is one of only two runs whose
surviving raw is **non-contiguous**:

```
run224526_0_s1.raw.finished     run224526_103   run224526_127   run224526_155
run224526_19                    run224526_105   run224526_135   run224526_161
run224526_44                    run224526_109   run224526_136
run224526_60  _65  _78  _82  _89   _111 _112 _113 _116 _118   _138
```

22 files with a maximum index of 161 — so ~162 were written and **140 have aged
off disk**. n_TOF then reprocessed the run on 08-07 15:35 from what was left,
and the surviving fraction matches exactly: 22/162 = 13.6 % of files,
440/3313 = 13.3 % of bunches.

### The input is intact on tape

```
xrdfs root://eosctapublicdisk.cern.ch/ ls \
  /eos/ctapublicdisk/archive/ntof/2026/EAR2/X17_measurement/224526/stream1
```

**167 files, indices 0..166, no gaps, 313.8 GB.** Every one reports
`on_tape: true, online: false, path_exists: true` — the normal dormant state,
which `xrdfs prepare -s -f` clears.

That is not special to this run. Sweeping the whole archive:

| | |
|---|---|
| runs in `DAQ/` (disk) | 445 |
| runs on CTA | 445 |
| in `DAQ/` but not on tape | 0 |
| tape copies with a contiguous `0..N-1` index sequence | **445 / 445** |

**Nothing in the campaign is lost.** The EOS `DAQ/` area is a two-week staging
buffer, not the archive.

### Why the official product came out short

`RunProcessing.sh` builds its file list from **whatever is on EOS**. On 08-07
that was 22 files, so it produced 6 partials and stopped — it did not fall back
to CTA for the other 145. The arithmetic matches exactly: 22/162 of the files
that were then on disk, 440/3313 of the bunches.

So the failure is not "the data expired", it is **reprocessing a run whose input
had partly expired, without staging it back first**. Any run reprocessed more
than two weeks after acquisition is exposed to this, and the campaign's last
runs passed the two-week mark during the 08-11 resubmission.

The sweep for a gapped raw sequence over all 445 runs found only one other
candidate, **224531** (12 files, max index 25), and it is **fine** — 652 of 652
beam bunches carry hits, so its processing predates the aging. 224526 is alone.

### Recovery

[`recover_224526/recover_224526.sh`](recover_224526/recover_224526.sh),
five sub-commands: `stage` → `check` → `filelists` → `process` → `verify`.

It never reads the EOS remnant. It lists from CTA and processes with
`ProcessFileList.sh -c 1` ("Use CTA (1) or EOS (0)"), so the short-file-list trap
cannot recur. Built and verified as far as it can go without the recall:
**42 job lists, 167 distinct CTA paths, correct idx pairing**, at the v12
UserInput that produced everything else of ours.

The recall is the slow step — the wiki quotes up to 72 h, usually hours.
Reprocessing is then ~42 jobs, comparable to any other run of this size.

## 5. What this changes for us

| | |
|---|---|
| `handoff_publish/publish_x17_block.sh`, `publish_224709.sh` | **not needed** — n_TOF processed all 31 runs themselves |
| `prod_v11` (224573-224579, incl. 224576) | **retirable** — every one is now COVERED officially at v4 |
| `prod_v12/224709` (SHORT, 18 of 86 partials) | **abandon** — official has all 86 |
| `prod_v12`, the other 30 block runs | duplicates of official; keep or drop as storage dictates |

Nothing has been deleted. `prod_v11/224576` was the only complete product of
that run in existence until last night; it no longer is.

## 6. What this does not rule out

* **The identity check is two runs, not thirty.** 224705 and 224710 are exact;
  the other 28 are inferred from an identical recipe fingerprint. The full diff
  is cheap to run and has not been run.
* **The 224526 recall has not been fired.** Everything up to it is verified; the
  staging, processing and coverage recovery are predicted, not measured.
* **224526's 440 bunches in the official product have not been checked for
  quality** — only that they exist.
* **Whether other runs need restaging is untested.** The tape archive is
  complete, but any *future* reprocessing of a run whose EOS copy has expired
  will silently truncate unless it goes through CTA. That is a property of
  `RunProcessing.sh`, and it is not guarded anywhere.
* **The `afast` LIQ divergence is unexplained.** It appeared on 224572 and on
  both block runs, always a handful of hits, always with absurd magnitudes.
* **Coverage says nothing about correctness.** A run can be complete and still
  have been processed against the wrong templates; the fingerprint check is what
  covers that, and it passed.
* **224405 and 224667 still carry zero-byte merged files** and have not moved
  since 08-05. Their partial sets are COVERED, so they are usable unmerged, but
  `exists()` will happily return true for that empty file.
