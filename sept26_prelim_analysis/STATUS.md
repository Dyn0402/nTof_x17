# Preliminary analysis — status

**Keep this file current. It is the resume point if a session drops.**
Plan of record: [`PLAN.md`](PLAN.md). Board:
<https://dylan-neff.web.cern.ch/x17/analysis.html>.

**One line:** **stages 0, 1 and 2 all run end to end on run_145.** Sample frozen
at 36 runs / 293 sub-runs / 25.60 M triggers; the candidate filter classifies
every trigger at **5.8 core-hours per million**; the stage-2 allowlist cuts the
reconstruction to **3.0 % of a full pass — ~610 core-hours campaign-wide, not
the ~1 900 the plan budgeted** — and reproduces the August full pass row for
row. Next: launch stage 2 on condor, and stage 3's schema.

Last updated **2026-09-07**.

---

## Resume here

Written on Windows, executed on the **Ubuntu laptop** on 2026-09-07.
**N0–N5 are done** and kept below for the record, each marked with what it
found. Stages 0, 1 and 2 all run.

**Next, in order:**

1. **Fix the `BUSY` prescale or retire its acceptance claim** (30 min) — 44
   seeded arm-events is not a measurement. See
   [seed efficiency](#what-the-allowlist-costs-at-the-seeder--the-number-that-must-not-be-lost).
2. **Launch stage 2 on condor** for run_145's three censused sub-runs:
   `allowlist.py --subrun all`, then
   `make_beam_package.py --allow <json> --arms A,B,C,D`, rsync, submit. ~4.5
   core-hours; the point is to prove the packaged path end to end before the
   campaign, not to get the numbers.
3. **Stage 3's schema and `build_tracks.py`** — `PLAN.md` §3 stage 3 has the
   column list. This is the artefact the week has to leave behind.
4. Then the campaign stage-1 pass (~135 core-hours) and stage 2 (~610).

### N0 · Land on the machine — ✅ done

Kerberos ticket live (`dneff@CERN.CH`, renewable to 12 Sep), `ssh lxplus`
works, both repos clean and current.

```bash
cd ~/PycharmProjects/nTof_x17      && git status && git pull
cd ~/PycharmProjects/dylan-cern-site && git status && git pull
```

Check `git status` **before** pulling on each: the Ubuntu clones may carry
their own unpushed commits (this has bitten before — `../CLAUDE.md` warns about
it for the DAQ clone). Then:

```bash
kinit dneff@CERN.CH
ssh lxplus true && echo "lxplus OK"
```

The two Claude skills (`publish-note`, `x17-board`) live in `~/.claude/skills/`,
which is machine-local and does **not** travel with the repositories. Canonical
copies are committed in the site repo, so install them once on the new machine:

```bash
cp -r ~/PycharmProjects/dylan-cern-site/skills/publish-note \
      ~/PycharmProjects/dylan-cern-site/skills/x17-board ~/.claude/skills/
```

### N1 · Publish what is already written — ✅ done

Both pages return `200`: the board and the plan note are live.

```bash
cd ~/PycharmProjects/dylan-cern-site && ./scripts/deploy-eos.sh
```

That puts the restructured board and the plan note live. Confirm:

```bash
curl -sI https://dylan-neff.web.cern.ch/x17/analysis.html | head -1
curl -sI https://dylan-neff.web.cern.ch/notes/x17-prelim-plan.html | head -1
```

If `/eos` comes back "Permission denied", the forwarded ticket is stale behind a
live `ControlPersist` master — `ssh -O exit lxplus`, then retry.

### N2 · Verify the five CERN assumptions — ✅ done

**All five answered; the answers are in
[Verified at CERN](#verified-at-cern).** Two of them change the plan and are
written up under [What the verification changed](#what-the-verification-changed).
The original five questions follow, for the record.

1. **Is the re-slim complete?** `../ntof_processing/SLIM_CAMPAIGN_2026-08-12.md`
   reports 170 fitted segments, 107 failures, and a recovery campaign started
   and stopped at 20 of 83 jobs. The site's pulse ledger (frozen 2026-08-16,
   from `slim_recovery_2026-08-13`) says 99.45 % of beam pulses are matched,
   which implies the recovery finished. **Confirm from the files on EOS, not
   from either document**, and produce the list of sub-runs that actually have a
   slim product. This is the single most important check: stage 1 reads the
   slim files, and a sub-run without one cannot enter the sample.
2. **Which runs already have a wft reconstruction**, with which bundle and at
   which code commit? run_79 and run_145 are known; anything else is not.
   Check `CODE_COMMIT.txt` beside each product — a mismatch there has bitten
   this analysis before (`../ntof_tracking/RUN145_R06_2026-08-19.md` §4, where
   45 jobs ran on a stale `code.tar.gz`).
3. **run_79's products still carry the two geometry defects** — the mirrored
   in-plane sign and the wrong pointing lever
   (`../ntof_tracking/RUN145_ALIGNMENT_2026-08-20.md` §6). They were fixed in
   the *analysis*, not in the parquet. Anything reading run_79's imaging output
   must apply the fix or be rebuilt. Decide which, and note it.
4. **Measure the link speed to CERN from the Ubuntu laptop.** 310 kB/s was
   measured in August; if it still holds, the run_145 pull below is an overnight
   job and the download order in `PLAN.md` §4 matters.
5. **Condor throughput and quota** — what the `workday` flavour delivers now,
   and whether `/afs/cern.ch/work/d/dneff` has room for the campaign outputs.

### N3 · Stage the run_145 development bundle — ✅ partly done

Staged under `/media/dylan/data/x17/beam_july/analysis/`:

| what | where | size |
|---|---|---|
| the 60 reco tarballs + `CODE_COMMIT.txt` + `jobs.txt` | `wft_beam145/results/` | 129 MB |
| the same, unpacked to `out/mx17_{A,B,C,D}/` | `wft_beam145/extracted/` | 143 MB |
| run_145 `stat090_0000` slim + all sidecars | `slim_dev/run_145_stat090_0000/` | 52 MB |

**Not pulled, and now a disk question rather than a link question:** the
`combined_hits_root` (515 MB per sub-run — the stage-1 input) and any
`decoded_root` (6.0–7.0 GB per sub-run). `/media/dylan/data` has **6.8 GB free
of 477 GB**. At 36 MB/s one sub-run's waveforms is ~3 minutes to transfer and
does not fit. **Free disk before pulling waveforms** — that is the real N3
blocker, and it did not exist when the plan was written.

### N4 · Scaffold the package — ✅ done

`figstyle.py` and `paths.py`, both with a runnable `__main__`.

- **`paths.py`** — every root resolves through it (`X17_ROOT` and four
  per-tree overrides), raises with the variable name that would fix it rather
  than returning a path that does not exist, and carries the CERN-side paths as
  strings so scripts spell EOS the same way. `python paths.py` prints what
  resolves and what exists.
- **`figstyle.py`** — one 16:9 canvas, 18 pt base, the Okabe-Ito four-chamber
  palette **re-validated for this package** (ALL CHECKS PASS; the two warnings
  are discharged by `det_style` always returning a marker with its colour, and
  by direct labelling). `save()` **refuses to write a PNG without its CSV** —
  pass `data=NO_DATA` for a schematic and expect to justify it. `python
  figstyle.py` renders the smoke test.
  - `end_labels()` de-collides direct labels with leader lines. It exists
    because the first smoke test collided chambers B and C at the right-hand
    edge, which is exactly what converging efficiency curves will do.

### N5 · Start the chain — **stage 0 done, stage 1 next**

Two small modules, before any analysis code, because everything else imports
them:

- `figstyle.py` — the shared presentation matplotlib style. 16:9 at a fixed
  figure size, base font ≥ 18 pt at final size, a `preliminary(ax)` badge
  helper, and a `save(fig, path)` that writes the PNG **and** the numbers as
  CSV beside it. Every figure in the deck comes through this.
- `paths.py` — machine-aware data roots, so no script hard-codes
  `/media/dylan/data/x17/`. Resolve from an environment variable with the
  laptop path as the default, and fail loudly rather than silently returning a
  path that does not exist.

### N5 · Then start the chain

**Stage 0 is written and run** — `freeze_sample.py`. See
[Stage 0, frozen](#stage-0-frozen) below for the numbers it produced.

**Stage 1 is the next code to write.** Its input is staged:
`analysis/run145_hits/stat090_0000/` — 7 file tags, 515 MB, a flat hit table
(`eventId, feu, channel, amplitude, time, time_of_max, integral, significance,
saturated, …`, ~1.9 M hits per tag). `PLAN.md` §3 stage 1 has the
specification; `../ntof_tracking/reco/noise.py` and `reco/segments.py` already
do the de-noising and the cluster taxonomy.

Two things to settle first, both raised by today's staging:

1. **Check the cluster taxonomy per arm before believing any census** — see
   [C3](#c3--arm-d-seeds-twice-as-often-as-a-b-and-c).
2. The n_TOF arm flags come from the slim file, staged for `stat090_0000` at
   `analysis/slim_dev/run_145_stat090_0000/`.

---

## Stage board

| stage | state | what exists | next |
|---|---|---|---|
| 0 · sample | **DONE** | the cuts have been *applied* to the frozen registry (2026-09-07): `mode=beam & phys & run≥79` gives 40 runs / 329 sub-runs / **25.87 M triggers** / 8.41 TB, and every one of those 40 is already ³He + Ar/Iso 90/10 + `st=complete` + 8 FEUs, so those cuts cost nothing. Dropping run_82 (watermark × IPD scan) and run_161 (detector-A resist × drift scan) leaves the core sample: **38 runs / 296 sub-runs / 25.62 M triggers / 8.32 TB** | — (see [Stage 0, frozen](#stage-0-frozen)) |
| 1 · time base | **todo** | the flash calibration and clock QA are done and published | pick the veto window; write `t_since_flash` and E_n per trigger |
| 2 · candidate filter | **RUNS** | `../ntof_tracking/reco/noise.py` and `reco/segments.py` already do the clustering and the taxonomy; the slim files carry the n_TOF arm flags | census on a full sub-run; then the stage-2 event-id allowlist |
| 3 · reco | **RUNS (filtered)** | `wft_beam.py` takes `--allow`; `allowlist.py` builds it from stage 1; `make_beam_package.py --allow` ships it and drops empty (arm, tag) jobs. Benchmarked and validated on run_145 tag 000, all four arms | launch on condor for run_145's three censused sub-runs, then the campaign |
| 4 · database | **todo** | `microtpc_lib.pair_planes`, `reco/geometry.py` (corrected sign + pinwheel), `reco/pairing.py` | fix the schema, write `build_tracks.py` |
| 5 · scint positions | **todo** | `../ntof_processing/quality_metrics.py` A1/A2 has both estimators and their caveats | recalibrate λ and the Δt scale against MM tracks |
| 6 · pairs & spectrum | **todo** | nothing | after 4 |
| — · figures | **scaffolded** | `figstyle.py` (validated palette, 16:9, PNG+CSV enforced) and `paths.py` | build them as each stage lands |

The board carries the same eleven stages with their full descriptions; this
table is the short form. Keep the two in step by moving the board stage
(`x17_board.py stage <slug> --status …`) rather than editing its numbers.

---

## Blockers

**B1 · lxplus access — resolved by switching machines, 2026-09-07.**
On the Windows box `ssh lxplus` fails with
`Permission denied (publickey,gssapi-with-mic,keyboard-interactive)`: no
Kerberos ticket (`klist` → 0 cached), no `kinit` installed, and `~/.ssh/id_rsa`
(2019) is not accepted. The `Host lxplus` block in `~/.ssh/config` is correct
and requests GSSAPI delegation — there was simply no ticket to delegate. The
same gap blocks `deploy-eos.sh`, which is why the board and the note were built
but not live.

**The fix is not to repair Windows: the analysis moves to the Ubuntu laptop**,
which has a working `kinit`, the data under `/media/dylan/data/x17/`, and every
path the repo's documentation already assumes. Nothing else was blocked by B1,
and nothing needs redoing after the move — everything written today is in git.

**Confirmed closed on Ubuntu the same evening:** ticket live, `ssh lxplus` OK,
`deploy-eos.sh` run, and both pages return `200`.

**B2 · Local disk is full — open.** `/media/dylan/data` has 6.8 GB free of
477 GB and `/home` 9.9 GB of 115 GB. The parquet and slim products fit; a
sub-run of `decoded_root` (6–7 GB) does not. This blocks nothing until N3's
waveform step, and it is the only thing in this file that has to be fixed on
the laptop rather than at CERN. See [C1](#c1--the-link-is-no-longer-the-bottleneck-so-the-disk-is).

---

## Verified at CERN

Filled in by N2 on 2026-09-07 — **all five checked against the files on EOS and
lxplus, not against the documents.**

| # | question | answer | when |
|---|---|---|---|
| 1 | is the re-slim complete, and which sub-runs have a slim product? | **Yes — 326 of 329 beam+phys sub-runs in run_79–162, 99.1 %.** 450 slim segments on EOS over 49 runs, 26.0 GB (24.3 GB from run_79 on). Three gaps, all trivial: run_120/`stat090_0000` (22 476 ev), run_137/`stat090_0000` (44 ev), run_145/`stat090_0003` (a 3.3 MB stub sub-run). The 46 runs in 79–162 with no slim at all are the **beam-off cosmic and pulser runs** — no n_TOF join by design, not a gap. | 2026-09-07 |
| 2 | which runs have a wft reco, with which bundle and code commit? | **Only run_145, and only 2 of its 4 sub-runs.** `lxplus:~/wft_beam145/` holds 60 tarballs = 4 arms × 15 file tags, 129 MB, `CODE_COMMIT` `5f1ee4a9`. Tags cover `stat090_0000` (7) and `stat090_0001` (8); `stat090_0002` (9 tags, 7.0 GB) is **not** reconstructed. Bundles verified kernel-ordered: A/B/D `calib_bundle_r06` with `c2_over_c1 = 0.6`, C `calib_bundle_lp` with a stored c2 = 0.0528 < c1 = 0.0642 (0.82), physical as intended. **No other beam reco exists at CERN.** | 2026-09-07 |
| 3 | do run_79's products get rebuilt, or patched at read time? | **Moot — there is nothing to patch.** No run_79 wft product exists on the laptop or on lxplus. The 2026-07-30 prelim was arm A only, tags 000–002 of 13, on `calib_bundle_prelim` (c2/c1 = 1.14 — the inverted kernel retired 2026-08-21), so it would have to be re-run regardless. **Rebuild.** | 2026-09-07 |
| 4 | link speed to CERN from the laptop | **17.5 MB/s from AFS, 36 MB/s from EOS** — 56–115× the August figure of 310 kB/s. Measured by `rsync` over ssh: 129 MB of tarballs in 7.7 s; a 54 MB slim file in 1.5 s. **The link is no longer the bottleneck; the local disk is** (`/media/dylan/data` 6.8 GB free of 477 GB, `/home` 9.9 GB of 115 GB). One measurement, one time of day — re-measure before committing to a multi-hundred-GB pull. | 2026-09-07 |
| 5 | condor throughput and AFS/EOS quota | **Queue is clear and fast** — an 8-job × 8-core `workday` probe was fully started in 78 s and done in 131 s. Cluster-wide 285 running / 263 idle, 0 for dneff. **Storage: use EOS, not AFS.** `/afs/.../work/d/dneff` is 74 % of 100 GB → ~26 GB free; AFS `~` is 62 % of 10 GB. `/eos/user/d/dneff` is 41 % filled of a **2 TB** quota — ample for the campaign outputs. | 2026-09-07 |

---

## Stage 2 — the event-id allowlist

Built 2026-09-07. `allowlist.py` turns the stage-1 class table into the list of
triggers stage 2 fits; `wft_beam.py --allow` consumes it; the condor packager
ships it and builds the job list from it.

### The unit of selection is the (event, arm) pair, not the event

This is where most of the saving comes from and it is worth stating plainly: an
`INTER` event needs its **two lit arms** fitted. The other two chambers have
nothing in them, and fitting them buys nothing but CPU.

| class | arms reconstructed | why |
|---|---|---|
| `INTER` | the 2 lit arms | the pair |
| `INTRA` | the 1 lit arm | both tracks are in it |
| `IMPLIED` | the lit arm **plus** every arm with an n_TOF coincidence and no track | forced — "was there a hint the hit-level finder missed?" is the only question `IMPLIED` exists to ask |
| `SINGLE` / `BUSY` / `NONE` | **all four**, prescaled | a control that only fitted the arms stage 1 already liked could not measure what stage 1 missed |

Control prescales are **per class**, not one global rate: `SINGLE` 5 %,
`BUSY` 10 %, `NONE` 1 %. Those classes differ by two orders of magnitude in
population, so a single rate would either bankrupt us on `NONE` or leave `BUSY`
in single digits. The draw is `blake2b(salt|run|subrun|event) < rate` — no RNG
state, reproducible on any machine, and adding a run does not repartition the
runs already drawn. **Changing `PRESCALE_SALT` redraws the whole control**, so
a redrawn control's efficiency is not comparable to the one before it; say so
here if it is ever bumped.

### The cost, measured

run_145/`stat090_0000`, and the three censused sub-runs agree to 2 %:

| class | triggers | selected | (arm, event) fits | of a full reco |
|---|---:|---:|---:|---:|
| `INTER` | 643 | 643 (100 %) | 1 286 | 0.56 % |
| `INTRA` | 1 430 | 1 430 (100 %) | 1 430 | 0.62 % |
| `IMPLIED` | 181 | 181 (100 %) | 398 | 0.17 % |
| `SINGLE` | 9 629 | 475 (4.9 %) | 1 900 | 0.82 % |
| `BUSY` | 687 | 64 (9.3 %) | 256 | 0.11 % |
| `NONE` | 45 172 | 439 (1.0 %) | 1 756 | 0.76 % |
| **total** | **57 742** | **3 232 (5.6 %)** | **7 026** | **3.04 %** |

Signal is 3 114 of those fits, the control 3 912 — **the control is 56 % of the
budget.** That is the price of having an acceptance at all; it is the first
number to cut if the budget moves, and cutting it means saying what the
spectrum's acceptance is then based on.

Timing, tag `260805_14H06_000`, all four arms, 8 workers on the laptop
(`stage2/bench/bench_run145_*.csv`):

| arm | allowed | seeded | wall | core-s | core-s/fit |
|---|---:|---:|---:|---:|---:|
| A | 223 | 145 | 20.7 s | 103 | 0.71 |
| B | 256 | 167 | 41.0 s | 236 | 1.42 |
| C | 276 | 189 | 36.2 s | 224 | 1.19 |
| D | 206 | 176 | 30.1 s | 156 | 0.88 |

719 core-s for 677 fits over 8 353 triggers = **0.086 core-s per trigger across
all four arms**, so:

| | triggers | core-hours |
|---|---:|---:|
| one sub-run | 57 742 | **1.4** |
| run_145 (24 tags) | 189 724 | **4.5** |
| **the campaign** | 25 598 064 | **≈ 610** |

**PLAN.md §5 budgeted ~1 900 core-hours** from the 3.80 % event-level selection.
The measured number is a third of that, for two reasons the plan did not model:
arm scoping (2.17 arms per event, not 4) and the seeder firing on only 70 % of
allowlisted arm-events. Includes interpreter start-up and I/O in every figure,
so it is an upper bound. **CPU is not the constraint** — 8 200 jobs each pulling
~290 MB from EOS is, and the current one-job-per-(arm, tag) design fetches the
same `combined_hits` file four times. Left alone for now; noted as the thing to
fix if the condor pass is I/O-bound.

### It reproduces the August full pass, row for row

run_145 `stat090_0000`/tag 000/arm A already has a **full, unfiltered** table
from the 2026-08-19 condor pass. The filtered run against it:

- all 145 event ids are a **subset** of the full table's 3 233 — the allowlist
  removes events, it does not invent them;
- **144 of 145 agree on all 52 columns exactly**;
- the one difference (event 1048) is a *different candidate cluster* chosen —
  31 vs 34 strips, χ² 17 386 vs 18 800, both far past `quality_ok`. Re-running
  the filtered job reproduces itself bit-for-bit, so this is not
  non-determinism: `wft/` moved between the August pass and HEAD (`d044073`,
  `b9c7856`, `6247750` — the kernel-inversion work). **Not an allowlist effect.**

### What the allowlist costs at the seeder — the number that must not be lost

An allowlisted event the beam seeder finds no cluster for produces **no row**,
and a missing row is indistinguishable downstream from an event that was never
selected. Unmeasured, that seeding loss silently becomes a reconstruction
inefficiency attributed to physics. So it is measured, per tag and per arm, and
written into every `.meta.json` as `allowlist.n_allowed/n_seeded/n_missing`.
Against the full August pass (`allowlist.py --seed-eff`):

| class | reason | n | seeded | frac |
|---|---|---:|---:|---:|
| `INTER` | lit | 1 286 | 1 267 | **0.985** |
| `INTRA` | lit | 1 430 | 1 404 | **0.982** |
| `IMPLIED` | lit | 181 | 176 | 0.972 |
| `IMPLIED` | forced_silent | 217 | 173 | **0.797** |
| `SINGLE` | control | 1 900 | 994 | 0.523 |
| `NONE` | control | 1 756 | 843 | 0.480 |
| `BUSY` | control | 256 | 44 | **0.172** |
| | **all** | 7 026 | 4 901 | 0.698 |

Three things to read off it:

1. **The signal path is safe.** 98.3 % of `lit` (arm, event)s seed — 97.3–99.8 %
   across arms. Stage 1 → stage 2 loses 1.7 % of what stage 1 identified.
2. **`forced_silent` seeds at 80 %.** In four cases out of five where n_TOF says
   a particle crossed a chamber and stage 1 found no track, the *waveform*
   seeder does find a clusterable deposit. `IMPLIED` is not an empty class, and
   the forced fit has something to work on.
3. **`BUSY` seeds at 17 %, and the two "busy" definitions disagree.** Stage 1
   calls an event `BUSY` at > 120 strips in an arm; the seeder vetoes a *plane*
   at > 150 hits (`BUSY_PLANE_HITS`). In a `BUSY` event most planes are over the
   seeder's veto, so they are dropped. At a 10 % prescale that leaves **44
   seeded arm-events** — not enough to measure anything. Either raise the `BUSY`
   prescale or state that `BUSY` has no acceptance measurement; do not leave it
   looking like it has one. **Open.**

Per-arm seeding runs A 0.64, B 0.64, C 0.69, **D 0.84** — D seeds most, as its
higher raw track rate in stage 1 already said.

### Files

```
sept26_prelim_analysis/allowlist.py          build, cost, seed efficiency
ntof_tracking/wft_beam.py                    --allow, load_allowlist(), sidecar
ntof_tracking/condor/run_beam_job.py         --allow, fails loud if missing
ntof_tracking/condor/make_beam_package.py    --allow: ships it, builds jobs from it
ntof_tracking/condor/beam_reco.sub           $(allowfile) in transfer_input_files
<out>/stage2/allowlist_<run>_<subrun>.json     what the worker reads
<out>/stage2/allowlist_<run>_<subrun>.parquet  the join key stage 3 needs
<out>/stage2/allowlist_cost_*.csv, seed_efficiency_*.csv, bench/
```

A job that runs **without** its allowlist reconstructs the whole tag and looks
perfectly fine doing it, so both the packager and the worker fail loudly rather
than fall back: `run_beam_job.py` exits if `--allow` names a file condor did not
transfer, and `load_allowlist` raises on an arm the document does not mention.

---

## Stage 1 — the full run_145 census, and what it says about `INTER`

`candidate_filter.py` over **all three sub-runs of run_145: 24 file tags,
189 724 triggers**, ~59 min wall at 52–53 triggers/s.

### The census is stable; the selection fraction is 3.80 %

| class | n | fraction | `0000` | `0001` | `0002` |
|---|---|---|---|---|---|
| `INTER` | 1 935 | **1.020 %** | 1.114 % | 1.077 % | 0.885 % |
| `INTRA` | 4 713 | **2.484 %** | 2.477 % | 2.583 % | 2.396 % |
| `IMPLIED` | 566 | **0.298 %** | 0.314 % | 0.285 % | 0.298 % |
| `SINGLE` | 30 703 | 16.183 % | 16.68 % | 16.10 % | 15.84 % |
| `BUSY` | 2 222 | 1.171 % | 1.19 % | 1.17 % | 1.16 % |
| `NONE` | 149 585 | 78.843 % | 78.23 % | 78.79 % | 79.42 % |

`INTER + INTRA + IMPLIED` = **3.80 %** → stage 2 ≈ **1 900 core-hours** at ~2 arms
per selected trigger. Stage 1 itself: **5.2–5.3 core-hours per 10⁶**, so
~135 core-hours for the campaign.

**The hot mask is stable across sub-runs** — 102, 103, 105 channels, each
measured independently from a different file tag. That is the evidence for
treating it as a per-run-condition calibration rather than something that
drifts file to file.

### `INTER` is the one class that is not Poisson — and it is not understood

χ² against a constant rate over the 23 full tags:

| class | mean | min | max | χ²/dof |
|---|---|---|---|---|
| **`INTER`** | 1.022 % | 0.685 % | 1.351 % | **2.85 — over-dispersed** |
| `INTRA` | 2.479 % | 2.083 % | 2.975 % | 1.53 |
| `IMPLIED` | 0.297 % | 0.192 % | 0.396 % | 0.67 |
| `SINGLE` | 16.18 % | 15.32 % | 17.55 % | 1.67 |
| `BUSY` | 1.172 % | 1.095 % | 1.298 % | 0.17 |
| `NONE` | 78.85 % | 76.86 % | 80.08 % | 0.74 |

**`INTER` should be Poisson.** The trigger is a wall+plastic coincidence in any
one arm, so every event contains one guaranteed particle; the second track is
the pair partner from the same interaction, i.e. pair-creation physics, whose
probability per triggered event is a constant. It scales with neither the
per-arm rate squared nor the beam intensity. So over-dispersion is an artefact
to be found, not a feature to be explained.

And it is not scatter. **Five *consecutive* tags spanning the sub-run boundary**
(`14H06_004,005,006` → `15H07_000,001`, roughly 14:40–15:25 on 5 August) sit at
**1.333 %** against **0.935 %** for the other eighteen — a 43 % elevation
lasting ~40 minutes.

**What has been excluded, each by measurement:**

| candidate cause | verdict |
|---|---|
| High voltage | **no** — mesh stable to 0.003–0.09 V, drift to 0.16 V over the whole run |
| Mesh current (≈ rate) | **no** — currents step up 13–35 % at 14:30 and *stay* up to 17:00, while `INTER` rises at 14:40 and falls back at 15:24 |
| Hot-mask scoping | **no** — see below |
| Beam intensity | ruled out for the pair component on physics grounds. *Not* excluded for an accidental second-track component, which would scale with rate |

**The mask does drift, and it does not matter.** Measuring the mask
independently on each of the 7 tags of `stat090_0000`, its membership moves
steadily away from the tag-000 mask that `run_tags` applies to all of them —
Jaccard 1.000 → 0.931 → 0.885 → 0.905 → 0.824 → 0.814 → 0.843, with arm A's
masked count growing 18 → 26 and arm C acquiring channels it did not have
early on. So the detector state genuinely evolves within an hour.

But re-running the first elevated tag (`14H06_004`) with **its own** mask
(98 channels) instead of the shared one (102) gives **`INTER` = 111 events
either way — identical to the event**. The classification is insensitive to the
~16 % mask churn, so the elevation is real data, not a processing artefact, and
the "measure once per sub-run" scoping stands.

**Status: deferred to October, by Dylan, 2026-09-07.** A ~40 % time-dependent
excursion on `INTER`, cause unknown, in the one class the analysis is built
around. Dylan's read: most likely one of the marginal chambers going noisy for
a while, or any of several similar things, and **it is not diagnosable at
stage 1** — telling a noisy chamber from a real pair excess needs the
reconstructed tracks and their match to the n_TOF scintillators, which is
stage 2 and stage 4. Do not spend more stage-1 effort on it.

It does not move the 3.80 % selection fraction much and it does not change what
stage 2 does. **Re-open it once the track database exists**: the same 24-tag
split, but cutting on `quality_ok`, on the two arms' `t0` agreement, and on
whether both tracks point at the target. If it is noise, the excess events fail
those cuts; if it survives them, it is something else. Whoever picks this up
wants the tag table in `stage1/census_per_tag_run_145_*.csv` and the window
14:40–15:25 on 5 August 2026.

### What stage 1 cannot do, and two things I got wrong on the way

**Stage 1 has no inter-arm timing.** A real pair is simultaneous; two unrelated
tracks are spread over the ~1.2 µs drift window. Nothing in this stage
distinguishes them — the per-arm waveform `t0` from stage 2 is that
discriminant. **No statement about the opening angle, or about how much of
`INTER` is signal, can be made from the stage-1 ledger alone.**

Two errors made and corrected while getting here, recorded because both are
easy to repeat:

1. **Treating the four arms as independent.** An "accidental" model of the form
   P(2 arms) = Σᵢⱼ pᵢpⱼ, and the symmetric event mixing built on it, are both
   **wrong for a triggered sample**. The trigger guarantees one particle, so the
   arms are anti-correlated by construction and the second track is not p². The
   29 % "deficit against accidentals" this produced was the trigger constraint,
   not a physics result. `event_mixing_background()` now raises rather than run.

   **Fixing it did not rescue the method.** The corrected estimator
   (`accidental_second_track()`) identifies the trigger arm from its
   wall+plastic coincidence, holds it fixed and mixes only the others — and
   returns observed 6.2335 % against "accidental" 6.3154 ± 0.0082 %, a ratio
   of 0.99. That is an identity, not a measurement: **permuting an arm's flag
   among the events where it is not the trigger arm conserves that arm's total
   count exactly**, so mixing only reassigns which event each second track
   lands in. Measured directly — 11 459 non-trigger lit arms over 11 077
   events, and the whole 1.3 % gap is clumping (372 events with ≥ 2), not
   content.

   **The limit is structural.** The stage-1 ledger holds only counts, and no
   permutation of counts can separate a pair partner from an unrelated track:
   what distinguishes them is that the pair is simultaneous and points at the
   target, i.e. timing and geometry. `PLAN.md` §5 puts event mixing at stage 5
   for exactly this reason; at stage 1 it is vacuous. The function is kept,
   renamed to the quantity it does measure (second-arm clumping), and says so
   in its own docstring.
2. **Concluding `IMPLIED` was broken.** From the same p² error: "if the
   coincidence flags were independent P(≥2 arms) would be 24.9 %, and it is
   1.65 %". The trigger requires *one* arm's coincidence and does not suppress a
   second, so the 1.65 % is the genuine second-particle population plus
   accidentals — exactly the parent `IMPLIED` draws from. The class is sound.

What the n_TOF data does show, and this stands: per arm,
P(coincidence | arm lit) against P(coincidence | arm not lit) is 68 %/16 % on A,
73 %/23 % on C, 63 %/23 % on D and 43 %/27 % on B — lifts of 4.3, 3.2, 2.8 and
1.6. The track finding is real, and B is the weakest, consistent with
everything else known about B.

---

## Stage 1, earlier — and the arm-D anomaly is solved

`candidate_filter.py` → `/media/dylan/data/x17/sept26_prelim/stage1/`
(`candidates_*.parquet`, `census_*.csv`, `arm_rates_*.csv`, `hot_mask_*.csv`,
`bench_*.json`). Numbers below are the **full file tag**,
run_145/stat090_0000/260805_14H06_000, 8 353 triggers.

### The class census

| class | n | fraction | before the `INTRA` fix |
|---|---|---|---|
| `INTER` | 84 | **1.01 %** | 1.01 % |
| `INTRA` | 189 | **2.26 %** | 2.66 % |
| `IMPLIED` | 26 | **0.31 %** | 0.31 % |
| `SINGLE` | 1 371 | 16.4 % | 16.0 % |
| `BUSY` | 106 | 1.27 % | 1.27 % |
| `NONE` | 6 577 | 78.7 % | 78.7 % |

**`INTER + INTRA + IMPLIED` is 3.6 % of triggers**, not the ~1 % `PLAN.md` §5
hoped for. At ~2 arms reconstructed per selected trigger that is
**≈ 1 800 core-hours** of stage 2 across the campaign — a day or two on condor
at this morning's throughput, so it is affordable, but it is *the* number that
decides whether the filter tightens, and one file tag of one sub-run of one run
is not yet the campaign. A sweep over all three sub-runs is running.

### Two bugs found by reading the code, not by watching it run

Both were caught before they reached a result that mattered; the second one
moved a headline number.

**The tag path loaded the whole sub-run first.** Asking for one file tag read
all ~13 M hits of the sub-run, discarded them, then read the one file — about
**8x the necessary I/O** across a `run_tags` sweep, with the 54 MB slim
re-parsed once per tag on top. Fixed, and `arm_flags` is now memoised per
sub-run: **load per tag 18.3 s → 1.5 s**. Some of the slowness earlier
attributed to rsync contention was actually this.

**`INTRA` was counting other chambers' doubles.** `n_sep_best` took the maximum
separated-cluster count over **all four arms**, but it is only consulted where
exactly one arm is lit — so two resolvable clusters in a *different, unlit*
chamber classified the event `INTRA`, booking one chamber's pair against
another. It now counts inside the lit arm only, with the all-arms value kept as
`n_sep_any_arm` so the size of what is excluded stays measurable.

It cost **15 % of `INTRA`** (222 → 189 on the reference tag; the 33 events moved
to `SINGLE`) and nothing else — `INTER`, `IMPLIED`, `BUSY` and `NONE` are
unchanged to the event, which is exactly what the fix predicts, since the bug
lived only in the one-lit-arm branch. The selection fraction went 4.0 % → 3.6 %
and the stage-2 estimate down by ~10 %.

### C3 resolved: arm D was half hot channels

`PLAN.md` §3 stage 1 step 1 calls for a dead/hot channel mask; none existed
(D10 is deferred). Measured per channel on run_145: **every arm's per-plane
median occupancy is 1.1–2.5 %**, so the arms are alike in the bulk. The tail is
not — **arm D has 26 channels above 20 % occupancy on each plane against 0–4
for A, B and C**, clustered at connector boundaries (x 92–127, x 448–450). An
excess concentrated on individual channels while the plane median is
unremarkable is instrumental; physics spreads across a plane.

`candidate_filter.hot_channel_mask` masks a channel firing in **> 8 % of
triggers *and* > 8× its own plane's median** — both, not either. What it does:

Both sides measured on the full 8 353-trigger tag:

| | A | B | C | D |
|---|---|---|---|---|
| channels masked (of 1 024) | 19 | 1 | **0** | **82** |
| strips/event removed | 2.7 | 0.2 | **0.0** | **23.2** |
| track rate (both planes), **unmasked** | 8.00 % | 3.69 % | 8.84 % | **6.17 %** |
| track rate, **masked** | 7.99 % | 3.71 % | 8.84 % | **4.91 %** |
| median clean strips, unmasked | 3 | 3 | 0 | **39** |
| median clean strips, masked | 0 | 3 | 0 | **11** |

Figure: `stage1/figures/arm_rates_mask_stat090_0000_tag000.png`.

Its effect on the census is to remove candidates, not create them —
`INTER` 1.22 % → 1.01 %, `INTRA` 2.93 % → 2.66 %, `NONE` 77.7 % → 78.7 %. About
1 % of triggers were being promoted out of `NONE` by D's hot channels alone.

**B and C are unchanged to the digit and only D moves** — the signature of a
targeted instrumental fix rather than a cut that flatters the data. After it
the arms order the way the calibration says they should: A and C, the good
pair, ahead of B and D. About 1.3 % of triggers were being promoted out of
`NONE` by D's hot channels alone.

Two honest caveats, both in the docstring: the mask is **measured on the same
data it is applied to** (a dedicated pedestal-run map is D10), and it is **per
run condition** — it must be re-measured, never carried across the 23-July
boundary. It is written out beside the candidates so a later D10 map can be
diffed against it.

### Benchmark

| what | value |
|---|---|
| throughput | **51 triggers/s** single core (noise + classify), 8 353 triggers |
| campaign cost, stage 1 | **5.4 core-hours per 10⁶ triggers → ≈ 140 core-hours for 25.6 M** |
| hot channels masked | 102 of 4 096 (run_145, both planes, four arms) |

`PLAN.md` §5 budgeted ~10³ core-hours for stage 1; it is ~7× cheaper than that,
so stage 1 over the whole campaign is an overnight condor job and not a
constraint on the week.

⚠ **Do not benchmark on this laptop while rsync is running.** The analysis and
both transfers share one `fuseblk` mount; with two rsyncs active the classifier
ran at ~23 % duty cycle (99 % CPU but 2 min of CPU per 8 min of wall). The
51 ev/s above is from a run that overlapped the transfers and is therefore a floor, not a ceiling.

---

## Stage 0, frozen

`freeze_sample.py` → `/media/dylan/data/x17/sept26_prelim/stage0/`
(`sample.csv` 2 270 rows, `cut_ledger.csv`, `sample_summary.json`,
`figures/sample_timeline.png` + `.csv`). Re-runnable; nothing here is typed in
by hand.

| | |
|---|---|
| **runs** | 36 |
| **sub-runs** | 293 |
| **triggers** | 25 598 064 |
| **beam hours** | 238.2 of 498 in the campaign |
| **matched pulses** | 239 675 of 239 679 delivered |
| **slim volume** | 20.2 GB |
| **EOS footprint** | 8.31 TB |
| **carries the arm-A x mask** | run_79 only (§2.2) |

### The cut ledger

Each sub-run is charged to the **first** cut it failed, so the column sums to
the campaign and nothing is double-counted.

| cut | sub-runs | runs | beam h |
|---|---|---|---|
| before the production trigger (run_79) | 1 822 | 71 | 239.9 |
| **kept** | **293** | **36** | **238.2** |
| not beam physics (cosmics) | 111 | 47 | 16.7 |
| detector-A resist × drift HV scan on beam (run_161) | 25 | 1 | 2.4 |
| not beam physics (pulser) | 9 | 3 | 0.4 |
| watermark × inter-packet-delay DAQ scan (run_82) | 8 | 1 | 0.6 |
| not joined to n_TOF | 2 | 2 | 0.1 |

**The production-trigger cut is the whole story**: it costs 240 of the 498 beam
hours, and after it almost nothing else is removed. Every one of the 36
surviving runs was already ³He + Ar/Iso 90/10 + `st=complete` + 8 FEUs, so
those four cuts cost *nothing* — they are worth stating precisely because they
turned out to be free.

### The honest denominator

The sample is 293 sub-runs; **296 exist**. `freeze_sample.unenumerated()`
reports the difference rather than letting it vanish:

- **run_120** and **run_137** (44 events) are in the ledger but never joined to
  n_TOF — they appear as cut rows.
- **run_145 `stat090_0003`** has no pulse-ledger record *and* no slim product,
  so neither input names it and it would otherwise be invisible. It is a 3.3 MB
  stub sub-run of the development run.

Sub-run names differ across all three inputs (`stat090_0000` in the slim
inventory, `0000` in the pulse ledger, unnamed in the run survey).
`freeze_sample._short()` is the single place that reconciles them and
`_check_join()` **raises** if the reconciliation ever stops covering the
sample — a silent join failure would look exactly like a missing slim product
and would quietly shrink every downstream denominator.

---

## What the verification changed

Three things N2 found that the plan did not assume. Each is a *change to the
plan*, not a detail — recorded here so §5 of `PLAN.md` is read against them.

### C1 · The link is no longer the bottleneck, so the disk is

`PLAN.md` §4 and the run_145 handoff both build on **310 kB/s**, measured in
August, and that number is what makes "run the reco at CERN and bring back the
parquet" the only route. Measured today from this laptop: **17.5 MB/s from AFS
and 36 MB/s from EOS** — 56–115× faster. run_145's entire `decoded_root`
(19.7 GB over its three real sub-runs) is **~10 minutes**, not 60 hours.

This does *not* overturn the condor design — 8.3 TB of campaign waveforms is
still work that belongs where the data is, and one measurement at one time of
day is not a guarantee. What it overturns is the **development** loop: the
waveform path can now be debugged locally on real data, which `PLAN.md` §4
explicitly gave up on.

**The new constraint is local disk.** `/media/dylan/data` is 99 % full
(6.8 GB free of 477 GB) and `/home` has 9.9 GB. One sub-run of waveforms does
not fit. Freeing disk is now a prerequisite for N3's step 4, and it is the one
thing here that cannot be done from the CERN side.

### C2 · Campaign outputs go to EOS, not AFS

`/afs/cern.ch/work/d/dneff` is at 74 % of its 100 GB — **~26 GB free**, and AFS
`~` has ~4 GB. The candidate ledger alone is "a few GB" by `PLAN.md` §3, before
any reco output. `/eos/user/d/dneff` is at 41 % of a **2 TB** quota. Every
campaign product from stage 1 on should be written there; the condor package
directory can stay on AFS because it is small.

### C3 · Arm D seeds twice as often as A, B and C

Across all 15 reconstructed run_145 tags: **A 47 546, B 44 896, C 48 470,
D 97 537** events. Per tag that is ~80 % of triggers seeded on D against
37–39 % on the other three, and it is not duplication — every `event_id` is
unique and all four arms span the same [1, ~8 350] range per tag.

This matters directly for stage 1. `INTER` is defined as "track-like clusters
in exactly 2 arms", so an arm that fires on twice as many triggers will
dominate the class census and push real one-arm events into `INTER`. **The
cluster taxonomy has to be checked per arm before the census is believed**, and
D's seeding threshold is the first thing to look at. Whether this is occupancy,
noise or a threshold difference is not yet known — it is a new open question,
not a known one.

---

## Decisions taken

| date | decision | why |
|---|---|---|
| 2026-09-07 | **run_145 is the local development run** | post-23-July and post-run_83 so neither campaign-wide condition bites; small (62 GB, 4 sub-runs); already reconstructed on all four arms with `calib_bundle_r06`; its slim join exists; its pointing result is a ready-made sanity check |
| 2026-09-07 | **intra-chamber = both tracks in one chamber; inter-chamber = one track in each of two** | fixes the naming used in the original brief. The X17 opening angle is ≥ 109°, which cannot fit in one chamber's ±40°, so **the signal is the inter-chamber topology** and the intra-chamber one is the low-angle control |
| 2026-09-07 | **filter on hits before reconstructing** | reconstructing the campaign blind is ~10⁵ core-hours. The filter is what makes the week possible, and its own efficiency is the price (D13) |
| 2026-09-07 | **B and D are tagging chambers this week** | their in-situ k (1.99, 1.70) is unphysical and B truncates half its columns. They say *whether* there was a track; their angles carry a flag and are never quoted alone |
| 2026-09-07 | **run_67 and run_68 are out of the sample** | run_67 is the quiet noise configuration and run_68 is inside the pedestal bracket and unplaced. Neither can be pooled with the production period (D14) |
| 2026-09-07 | **no invariant mass** | needs scintillator calorimetry, which needs a calibration we do not have (D6). The opening angle is the deliverable |
| 2026-09-07 | **the analysis runs on the Ubuntu laptop** | it is the only machine here with lxplus, and every path in the repo's documentation already assumes it |

---

## Benchmarks

Filled in as they are measured. Empty is honest; a guess is not.

| what | value | measured on |
|---|---|---|
| wft reco throughput | — | |
| stage-2 filter throughput | — | |
| class census (fraction per class) | — | |
| campaign reco cost estimate | — | |
| link speed to CERN | **17.5 MB/s** (AFS), **36 MB/s** (EOS), rsync over ssh | 2026-09-07, Ubuntu laptop, 14:10 CEST |
| per-arm seeding rate, run_145 | A 39 %, B 37 %, C 39 %, **D 80 %** of triggers | 2026-09-07, 15 tags, `wft_beam145` |
| condor queue latency, `workday` 8-core | **78 s** to first start, 101 s for the last of 8; every job got its 8 CPUs | 2026-09-07, 8-job probe, cluster 4139571 |
| condor worker spread | ±15 % on a fixed sha256 loop across 6 distinct workers | 2026-09-07, same probe |

The only number in hand is the historical one: run_79, 12 534 events on
6 cores in ~1 h 50 m ≈ **1 100 events per core-hour per chamber**, on
20-sample windows with `WFT_MODEL_FRAC=0.03 WFT_PRESCAN=1`. Treat it as an
order of magnitude until re-measured.

---

## Log

**2026-09-07 (latest, Ubuntu)** — Stage 2's allowlist: built, benchmarked,
validated against the August full pass, and a third the budgeted cost.

*The `INTER` excursion is parked, on Dylan's call.* It is not diagnosable at
stage 1 — separating a chamber going noisy from a real pair excess needs
reconstructed tracks matched to the n_TOF scintillators. October, with the
track database in hand; the re-open recipe is written into the section above.

*The allowlist.* `allowlist.py` selects **(event, arm) pairs**, not events, and
that scoping is most of the saving: 2.17 arms per selected event rather than 4.
`INTER`/`INTRA`/`IMPLIED` in full, plus `IMPLIED`'s n_TOF-coincident silent arms
forced, plus a per-class-prescaled control over all four arms. The draw is a
blake2b hash, not an RNG, so it is reproducible with no stored state and a rate
change moves only the events near the old threshold.

*It reproduces the August pass.* On tag 000/arm A, 144 of 145 events agree with
the unfiltered table on all 52 columns; the one difference is a different
candidate cluster in a χ² ≈ 18 000 event, and re-running reproduces itself
bit-for-bit — `wft/` moved between 2026-08-19 and HEAD, the allowlist did not
cause it.

*The cost is a third of the plan's.* 0.086 core-s per trigger across four arms
→ **≈ 610 core-hours campaign-wide** against PLAN.md §5's ~1 900. Arm scoping
and a 70 % seeder hit rate account for the difference. The constraint is not
CPU: it is 8 200 condor jobs pulling ~290 MB each from EOS, with `combined_hits`
fetched four times over. Left as-is, flagged.

*One number worth having found.* Of the events the allowlist selects, the beam
seeder produces a cluster for 98.3 % of the `lit` ones — the signal path is
safe — but only **17 % of `BUSY`**, because stage 1's "busy" (> 120 strips in an
arm) and the seeder's (> 150 hits in a plane) are different tests and the
seeder throws the planes away. At a 10 % prescale that is 44 seeded arm-events:
not a measurement. Either the `BUSY` prescale goes up or `BUSY` is stated to
have no acceptance. Open, and cheap to fix before the campaign pass.

**2026-09-07 (late, Ubuntu)** — Stage 1 written and running; the arm-D anomaly
turned out to be instrumental.

*The upload was never slow.* The "5 MB/s" in the previous entry was a
measurement error of mine: every window it was averaged over also contained a
download, a 3-route upload benchmark, or an EOS tree walk. Two clean windows
with nothing else running give **23 and 25 MB/s**, and a single 512 MB file
reaches EOS at 13 MB/s *while contending*. Nothing needed fixing.

*There is a real contention effect, on disk not network.* The analysis and both
rsyncs share one `fuseblk` mount; with two transfers active the classifier ran
at ~23 % duty cycle (99 % CPU, ~2 min of CPU per 8 min of wall). Benchmark on a
quiet disk or treat the number as a floor.

*Stage 1.* `candidate_filter.py` classifies every trigger into the six-class
partition from `combined_hits` + the slim, no fitting and no geometry.
`reco.search.sift_events` was deliberately **not** reused — it ranks rather than
partitions, and it does 3D pairing per event, which is both the wrong basis
(`../RECONSTRUCTION_BASIS.md`) and far too slow at campaign scale. `io`, `noise`
and `segments` are reused as-is, plus one backward-compatible `measure=False`
on `find_segments` to skip the anchored fit that candidate-finding never uses.

*The finding.* C3 is resolved and it was not physics. Per-channel occupancy
shows all four arms have the same **1.1–2.5 % per-plane median**; what differs
is the tail, and arm D's tail is 26 channels above 20 % occupancy *per plane*
against 0–4 elsewhere, sitting at connector boundaries. The mask built from
that (`> 8 %` **and** `> 8x` the plane median) touches **82 channels on D, 19 on
A, 1 on B and none at all on C** — B and C's track rates come back identical to
the digit while D's falls by a third. A cut that only moves the arm that was
anomalous, and leaves the two good chambers untouched, is the shape an
instrumental fix should have.

*What it costs.* 51 triggers/s, **5.4 core-hours per 10⁶** → ~140 core-hours for
the campaign, ~7x cheaper than `PLAN.md` §5 budgeted. But
`INTER+INTRA+IMPLIED` is **4.0 %**, not the ~1 % the plan hoped, which puts
stage 2 at ≈ 2 000 core-hours. Affordable, and still one file tag of one
sub-run — it needs re-measuring across sub-runs before anything is launched.

**2026-09-07 (night, Ubuntu)** — Disk reclaimed, package scaffolded, stage 0 done.

*Disk.* `sps_run53_det4_check` was 220 GB of the 477 GB volume. Checked it
against EOS file by file: **all 294 `.fdf` and all 175 `combined_hits` ROOT
files matched an EOS file of identical size**, so those 140 GB went
immediately — the volume is at 73 % instead of 99 %. The other 92 GB is
`dec_*`/`hits_*` ROOT that was decoded *locally* and is on EOS nowhere; it is
being uploaded to `/eos/user/d/dneff/x17/sps_run53_det4_check/` before being
deleted here. The 3.5 GB of `.npz`/`.json`/`.csv`/figures — the actual SPS
analysis products — were mirrored to EOS **and kept on disk**.
`sps_beam_test_26/RESTORE_LOCAL_STAGING.md` is the record;
`DELETED_MANIFEST_2026-09-07.json` beside the data lists all 1 824 files.

⚠ **EOS's own `decoded_root` is not a substitute for ours** — it holds FEU `_01`
(the P2 detector) only, decoded with different settings. Our det4 side is FEU
`_03`, which EOS never decoded.

*What was already local.* `beam_july` holds 111 GB of n_TOF processed ROOT for
the **July commissioning** runs (224466, 224476–79, 224489, 224503) plus a 50 GB
`hitcache`, and 15 GB of MM `run_55`. **None of it is in the September sample**,
which starts at n_TOF run 224572. Two things worth knowing about it: the local
`run224*.root` are *a different processing vintage* from EOS — consistently
81–85 % of the EOS size, and both open cleanly, so this is not truncation — and
nothing else from the production period was ever staged here.

*Scaffolding and stage 0.* `paths.py` and `figstyle.py` (N4), then
`freeze_sample.py`, which produced the frozen sample above. The sample is
**36 runs / 293 sub-runs / 238 beam hours**, and the cut ledger says the whole
cost is the production-trigger cut — the four condition cuts (target, gas,
status, FEUs) turned out to remove nothing at all.

*Measured on the way.* Link holds ~16 MB/s **in both directions at once** (a
515 MB `combined_hits` pull ran at 15.8 MB/s while the 92 GB upload was
saturating the other direction). Upload to EOS is slower per byte than
download, ~5 MB/s across many mid-size files.

**2026-09-07 (evening, Ubuntu)** — Landed on the laptop and ran N0–N3. Kerberos,
lxplus and the deploy all work; the board and the plan note are live. All five
CERN assumptions checked against the files rather than the documents:

- **The re-slim is complete** — 326 of 329 beam+phys sub-runs from run_79 on,
  99.1 %. The 46 runs in that range with no slim at all turned out to be the
  beam-off cosmic and pulser runs, which have no n_TOF join by design; the
  alternating gap in the run numbers is that alternation, not a failure. Three
  real gaps remain and all are negligible (run_120, run_137's 44 events, and
  run_145's stub 4th sub-run).
- **run_145's reconstruction covers 2 of its 4 sub-runs**, not the whole run —
  `stat090_0002` (9 tags, 7.0 GB) was never reconstructed. All four arms'
  bundles verified kernel-ordered on the way in.
- **run_79 has no surviving reconstruction anywhere**, so the "rebuild or
  patch?" question answers itself: there is nothing to patch, and the 7-30
  product was arm A only on the retired inverted kernel regardless.
- **The link is 56–115× faster than the August measurement** and the local disk
  is now the constraint instead (C1).
- **AFS work is nearly full and EOS user has 2 TB** (C2).

One genuinely new open question came out of staging the products: **arm D seeds
twice as many triggers as A, B and C** (C3), which stage 1's class census
cannot be believed without explaining.

Nothing was run beyond verification and staging — the package scaffolding (N4)
is still the next code to write.

**2026-09-07 (later)** — Moving to the Ubuntu laptop; lxplus is unreachable from
the Windows box and repairing it there is not worth the time when the machine
that already works is a reboot away. Everything written today is committed and
pushed: the plan, this status file, the package README, the note generator, and
on the site side the restructured board plus the plan note, both **built but not
deployed** — `./scripts/deploy-eos.sh` on Ubuntu is the one command that has
been waiting. Next steps written up as N0–N5 above.

**2026-09-07** — Plan written. Took stock of the git history and the inherited
state: the DREAM ↔ n_TOF match is effectively done (99.45 % of beam pulses
since run_79), n_TOF reprocessing is complete (445/445 runs), and the
waveform-first reconstruction is proven on run_79 and run_145 with the
corrected sharing kernel and the corrected geometry. Established that the X17
opening angle of ≥ 109° cannot fit inside one chamber's ±40° acceptance, which
makes the two-chamber topology the signal region and puts chambers B and D —
the two that are not calibrated — on the critical path. Chose run_145 as the
local development run.
