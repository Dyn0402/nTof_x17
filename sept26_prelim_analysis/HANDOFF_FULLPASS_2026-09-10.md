# HANDOFF — the campaign full pass is done, and the angle scale is now the blocker

**Written 2026-09-10, covering the session of 2026-09-09 evening → 2026-09-10 morning.**

Companions: [`STATUS.md`](STATUS.md), [`PLAN.md`](PLAN.md),
[`DATASET.md`](DATASET.md), `condor/`, `overnight_fullpass_2026-09-09.sh`.

---

## 0 · Read this first

The session started from one question — *are we ready to make opening-angle
distributions?* — and ended somewhere else. In order:

1. **We were not ready, and the reason was not the one anyone expected.** The
   stage-1 filter keeps **12.8 %** of the triggers that actually reconstruct
   into a two-track event (§1). That is why the campaign gave 8.2× run_145's
   A–C pairs and not the 50× the plan projected.
2. **The scintillators say most of what the filter discarded is not real**
   (§2), so a full pass looked roughly break-even. **Dylan overrode that**:
   *"not worth missing anything."* Recorded, not re-argued.
3. **The full pass ran and succeeded** — 12 929 of 12 932 jobs, **9.5× more
   reconstructed events** (§3, §4).
4. **Then the run_86 calibration test falsified the borrowed angle scale**
   (§5). **This now blocks the opening angle, and it is a bigger problem than
   the statistics the full pass just bought.**

**If you do one thing: §5, then §6(a).** Do not build an opening-angle
spectrum on the current `k` for arms C or D.

---

## 1 · Why the full pass happened — the filter keeps an eighth of the pairs

`run_145` is the only run with a **blind full pass**, so it is the only place
the stage-1 filter can be scored against ground truth. Of the **595** triggers
its full pass turns into two gated, angle-calibrated tracks within 30 mm of the
axis, stage 1 put **76 (12.8 %)** into the classes `allowlist.py` reconstructs
in full.

| where the 595 two-track triggers were classified | n |
|---|---:|
| `NONE` (drawn at 1 %) | 355 |
| `SINGLE` (drawn at 5 %) | 161 |
| `INTRA` | 39 |
| `INTER` | 35 |
| `BUSY` / `IMPLIED` | 5 |

**No cheap retune of stage 1 closes this.** Scored on the same truth:

| stage-1 rule | triggers selected | two-track events captured |
|---|---:|---:|
| current (`INTER`/`INTRA`/`IMPLIED`) | 3.80 % | **12.8 %** |
| `n_arms_loose >= 2` | 13.77 % | 47.7 % |
| `n_arms_loose >= 1` | 57.77 % | 89.2 % |

89 % capture costs 58 % of all triggers — a full pass in all but name.

**The earlier 98.3 % seeding number is not in tension and does not answer
this.** It measures whether stage 2 can *see* what stage 1 chose. It says
nothing about what stage 1 discarded. Two different questions; only the second
sets the pair yield.

**Stage 2 was never at fault.** Every campaign run converted its `INTER`
triggers into two-track events at 2.1–3.9 %, and run_145's own `INTER`
triggers converted at 1.8 %. The difference was entirely *which events stage 1
offered*.

---

## 2 · The scintillator check — it argued AGAINST the full pass

New: `accidental_timing.fit_by_stage1_class` (+ `stage1_class`), which runs
from the existing CLI and writes
`<out>/accidental_timing/fit_by_stage1_class_run_145.csv`.

The two-arm-tagged sample is small (76 pairs, 65 of them "missed"), so **the
non-parametric comparison carries the result** — no KDE, no template fit:

| population | n | median \|Δt\| | within 30 ns |
|---|---:|---:|---:|
| prompt template (both arms born together) | — | 15.2 ns | 0.82 |
| **stage-1 selected** | 11 | 18.8 ns | **0.82** |
| **stage-1 missed** | 65 | 33.0 ns | **0.46** |
| accidental template | — | — | 0.37 |

The pairs stage 1 keeps are **prompt**. The ones it misses are not: 46 % within
30 ns is inconsistent with pure prompt at **p = 8 × 10⁻¹¹** and only marginally
above pure accidental (**p = 0.09**). The two populations differ at **p = 0.022**
(Mann–Whitney) / **p = 0.029** (Fisher). Implied true-coincidence fraction of
the missed population **0.20**, independently reproduced by the MLE at
**f = 0.19 [0.07, 0.31]** against the pooled **f = 0.29 [0.17, 0.40]**.

**The projection that argued against the pass.** run_145 splits 61 selected
against 405 missed inter-chamber pairs. Folding both fractions through:

| | now | after a full pass |
|---|---:|---:|
| true pairs | ~49 | ~126 (2.3–3.1×) |
| purity | ~0.80 | ~0.27 |
| S/√(S+B) | 6.25 | 5.83 (**0.93×**, range 0.57–1.29) |

Roughly break-even, for ~20× the compute. **The weak link is n = 11 on the
selected side** — everything above rests on those 11 being genuinely prompt.

**This is now cheaply fixable and NOT done.** The campaign drew `NONE` at 1 %
and `SINGLE` at 5 % in **all four arms** and holds 151× run_145's `INTER`, so
repeating this split campaign-wide needs no new reconstruction — only pointing
the slim read at the exported parquet instead of `scintillators.read_slim`'s
ROOT (which is run_145-only). That would put hundreds of pairs on the selected
side. **Do it before quoting f anywhere.**

One number worth revisiting while you are there: the accidental template sits
at **0.37 within 30 ns** because the production accept window is (−100, +60) ns.
The narrower window this module already recommends would sharpen every row.

---

## 3 · What was built, and how to re-run it

| file | what |
|---|---|
| `condor/make_stage2_campaign.py` | **`--full-pass`** added. Tags come from the stage-1 candidate tables (one row per trigger ⇒ their distinct tags ARE the sub-run's tags), not from the allowlist, which only lists tags the filter selected. Emits `logdir` as a 4th job column. |
| `condor/stage2_fullpass.sub` | new. No `allowlists.tar.gz`; `workday` (8 h); stderr sharded by run. |
| `condor/run_stage2_fullpass_wrapper.sh` | new. No allowlist unpack; writes to `sept26_fullpass` on EOS. |
| `overnight_fullpass_2026-09-09.sh` | new. Smoke-gate → submit → watch → release → clear → pull/unpack. Resumable. |

```bash
.venv/bin/python sept26_prelim_analysis/condor/make_stage2_campaign.py \
    --full-pass --dest /home/dylan/x17/sept26_fullpass
rsync -a --exclude 'allow/' --exclude 'allowlists.tar.gz' --exclude 'bundles/' \
    /home/dylan/x17/sept26_fullpass/ lxplus:sept26_fullpass/
bash sept26_prelim_analysis/overnight_fullpass_2026-09-09.sh   # detached
```

**The allowlist path is untouched**, so the earlier pass stays reproducible.

### Three deviations from the allowlist pass, each for a measured reason

- **`workday` (8 h), not `longlunch` (2 h).** These jobs are ~20× longer:
  measured median **32 min**, p90 **50 min**, max **53 min** (D's smoke job
  76 min). A job killed on the flavour limit is held and retried into the same
  limit.
- **stderr sharded by run** into ~36 directories. AFS caps entries per
  directory and a flat 22k-file log dir already degraded the shared schedd once.
- **Tags from stage 1, not the allowlist** — see the table above.

### The smoke gate, and why it is not optional

Four jobs (one per arm) on `run_145/stat090_0000` tag `260805_14H06_000`, whose
August blind pass is known. The gate requires each arm within 2 % of it, and it
came back **exact**:

| arm | full pass | August blind pass |
|---|---:|---:|
| A | 3 233 | 3 233 |
| B | 3 098 | 3 098 |
| C | 3 256 | 3 256 |
| D | 6 690 | 6 690 |

That is the one check that catches the way this could silently go wrong — an
allowlist still being applied somewhere in the path. `run_beam_job.py`'s own
FATAL text says as much: without `--allow` it "would silently reconstruct every
trigger of the tag", which here is the *intent*, so nothing else would complain.

---

## 4 · What the full pass produced

**22:27 → 06:31 on condor (8 h), pull+unpack to 07:19.** `12 929 / 12 932`
outputs, **0 problems on unpack**, all **293** sub-runs present.

| arm | full pass | allowlist pass | ratio |
|---|---:|---:|---:|
| A | 9 458 115 | 1 096 449 | 8.6× |
| B | 9 377 791 | 1 175 912 | 8.0× |
| C | 9 573 426 | 1 362 793 | 7.0× |
| D | 19 356 728 | 1 406 033 | 13.8× |
| **all** | **47 766 060** | **5 041 187** | **9.5×** |

| | |
|---|---|
| local reco | `<out>/reco_fullpass` — **28 GB**, `<run>/<subrun>/mx17_<arm>/events_<tag>[.candidates].parquet` |
| EOS | `/eos/user/d/dneff/x17/sept26_fullpass` |
| tarballs | `/home/dylan/x17/sept26_fullpass/tarballs` — **25 GB, redundant once unpacked, reclaimable** |
| disk left | 113 GB on `/media/dylan/data` |

> **`<out>/fullpass` is NOT this.** Despite its name it holds the **allowlist**
> pass's reco. The full pass deliberately went to a new tree so the comparison
> between them survives. `campaign_tracks` takes `--fullpass <path>`; pass it
> explicitly or you will rebuild from the old sample.

### Two corrections this pass forced

**The corrupt file is three files, not one.** `STATUS.md` recorded
`run_104/stat090_0016` tag `260730_11H29_000` **FEU 03** as the campaign's only
data loss. Reading every arm of every tag found **FEUs 02, 03 and 07 of that
tag all empty** — full size on EOS (75–85 MB), zero ROOT keys, identical
signature. Arms A, C, D held; **arm B ran fine**. The allowlist pass saw only
FEU 03 because it never asked the other two arms for that tag. Still one tag of
one sub-run; three quarters of it rather than one quarter. The three jobs were
released once, failed identically, and were retired as deterministic.

**A bug in the driver's own watch loop.** Held jobs count as queued, so "wait
until the queue is empty" never exits once a deterministic failure holds — the
pass would have finished at 06:31 and the driver would have spun until morning
without pulling anything. Fixed to wait on running-plus-idle only, then release,
then record-and-clear. The submit step now also refuses to run while anything
is queued: a **running** job has no EOS tarball yet, so the done-list alone
would have submitted a second copy of everything in flight.

---

## 5 · THE BLOCKER — the run_86 test falsifies the borrowed angle scale

### 5.1 The test never ran on its own, and nearly destroyed its own control

`unattended_2026-09-09.sh` called `k_arm.py --out <dir>`. **That flag does not
exist**, so after a 14-hour local full pass of run_86 the calibration step
died on `unrecognized arguments` and the comparison printed a blank row.

**`k_arm` has no `--out` at all** — it always writes
`paths.out('kcal')/k_arm_<run>.json`. So the obvious "fix" of dropping the flag
would have **overwritten run_86's calibration-pass file**: exactly the trap that
deleted run_145's published calibration on 2026-09-09. Re-run by hand with the
control copied aside first. Both now coexist:

```
kcal/k_arm_run_86.fullpass.json    the new full-pass measurement
kcal/k_arm_run_86.calibpass.json   the calibration-pass control
kcal/k_arm_run_86.json             RESTORED to the calib-pass content
```

**If you re-run `k_arm` for any run, copy the existing JSON aside first.**

### 5.2 The result

| arm | r86 FULL | r86 calib | full/calib | r145 FULL | **r86 / r145** |
|---|---:|---:|---:|---:|---:|
| A | 1.2294 | 1.1840 | +3.8 % | 1.2662 | **−2.9 %** |
| B | 1.5664 | — | — | 2.1421 | −26.9 % (never certifies) |
| C | 1.1560 | 1.3500 | **−14.4 %** | 1.6163 | **−28.5 %** |
| D | 1.3413 | — | — | 1.7667 | **−24.1 %** |

**The method hypothesis fails on its own terms.** It predicted the full pass
would sit ~6 % **above** the calibration pass on both arms. A moves **+3.8 %**,
C moves **−14.4 %** — opposite direction, larger size. There is no single
method offset.

**With the method now MATCHED — both are full passes — `k` still differs
between the two runs by 24–29 % on C and D**, while A agrees to 2.9 %. run_86
and run_145 are on the **same side** of the 27 July access. This is `PLAN.md`'s
own stated falsifier arriving: *if `k` varies strongly run to run, one bundle is
wrong and stage 2 must be re-cut by condition.*

**Applying run_145's `k` campaign-wide is not supported for C or D.** C is half
of the A–C opposing pair, which is the signal topology.

### 5.3 Three caveats, none of which rescue it

- Every run_86 verdict is **PROVISIONAL**, focus scan flat over **30–37 %** —
  the estimator is weak.
- Within-run estimator spread is **8–15 %**: below the 24–29 % gap, not
  negligible.
- **run_86 has no hot-strata table, so its hot-channel cut was NOT applied.**
  On run_145 that cut moved D's `k` by 3.65 % and removed 22.4 % of D's
  coincident sample. **D's comparison is partly confounded; C's is not.**

### 5.4 Also corrected

The "run_86 calibration-pass k (A 1.184, C 1.350)" quoted in `STATUS.md` was
**never certified** — that file's `apply` is `{}` and both arms read
`NOT CALIBRATED`. They are raw fit values, not measurements.

---

## 6 · What to do, in order

**(a) Per-run `k` on the new full pass. — THE BLOCKER.** The full pass just
made this affordable for all 36 runs and §5 says it is also necessary. Guard
the existing JSONs first (§5.1). Expect B to keep failing everywhere; the
question is whether A/C/D certify per run and how much they move. **If C and D
scatter at the 25 % level, the single-bundle assumption behind stage 2 is what
has to be revisited, not just `k`** — and that is a re-cut by condition, not a
downstream correction.

**(b) Rebuild the track database.** `campaign_tracks --fullpass
<out>/reco_fullpass`, with `--k-from` only if (a) says one `k` is defensible.
It stamps `k_source` on every row, so a borrowed scale can never read back as a
per-run measurement — but 83 % of the *current* table's calibrated angles rest
on a scale §5 does not support for two of three usable arms.

**(c) The campaign-wide timing split** (§2). No new reconstruction; it retires
the n = 11 weakness that the whole full-pass case turned on.

**(d) Only then, the opening angle.** Four things are still missing, all
identified before the full pass and none fixed by it:

1. **`opening_angle.py` is single-run.** It takes one `--run`, has no `src`
   argument, and reads `<out>/stage3_fullpass`. It needs the campaign entry
   point `tight_coincidence.py` already grew (`--all-runs`, `--campaign`).
2. **The expectation is wired to the wrong physics.** `expectations()` folds
   `pair_physics.VARIANTS`. `CLAUDE.md` and `PLAN.md` §S4 both say it must be
   `ipc_channels.thermal_spectrum()` (Born multipole, thermal M1/E0 mix).
   Nothing in the angle chain imports it. Flagged in the plan as
   *"still to do: wire it in"*.
3. **Acceptance is run_145-only** and applies efficiency independently of
   incidence — a real θ-dependent bias, median head-on tracking ratio 0.80.
4. **No condition switch.** `tight_coincidence` excludes run_79/81 by default;
   `opening_angle` has no such flag. 18 183 of 229 950 selected tracks are
   pre-access.

---

## 7 · Traps

- **`<out>/fullpass` is the allowlist pass.** The full pass is
  `<out>/reco_fullpass`. Pass `--fullpass` explicitly.
- **`k_arm` has no `--out`** and overwrites `k_arm_<run>.json`. Copy aside first.
- **`scintillators.read_slim` reads ROOT and is run_145-only** on this machine.
  Campaign-wide, use the parquet export (`slim_export.read_export`), the way
  `tight_coincidence._slim` does.
- **`angle_calibrated` is a filter in disguise** (`DATASET.md` §3): a run whose
  `k_arm` never certified contributes *nothing*, silently. B never certifies.
- **The tight-coincidence cut enriches a single-particle background.** Use
  `tight_pair`, never `tight`, for anything about X17.
- **`tight_coincidence_campaign.meta.json` mislabels itself** as `run_145` with
  three sub-runs — it stamps `a.run` regardless of `--all-runs`. The data is
  campaign-wide (1 000 tagged pairs vs run_145's 76); only the header is wrong.
  Its event-mixed null **is** run_145-only, because the histogram call still
  passes the single-run arguments. Not fixed.
- **25 GB of tarballs** in `/home/dylan/x17/sept26_fullpass/tarballs` are
  redundant once unpacked.
