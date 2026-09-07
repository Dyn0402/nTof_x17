# Preliminary analysis — status

**Keep this file current. It is the resume point if a session drops.**
Plan of record: [`PLAN.md`](PLAN.md). Board:
<https://dylan-neff.web.cern.ch/x17/analysis.html>.

**One line:** the plan is written and the board matches it; nothing has been
run. The work moves to the Ubuntu laptop, because lxplus is unreachable from
the Windows box — start at [Resume here](#resume-here).

Last updated **2026-09-07**.

---

## Resume here

Written on Windows, to be executed on the **Ubuntu laptop**. Do these in order;
each is a few minutes except N2.

### N0 · Land on the machine

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

### N1 · Publish what is already written — one command, and it has been waiting

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

### N2 · Verify the five CERN assumptions, and write the answers down

Every one of these is something the plan rests on and none has been checked
since August. Record each answer in [Verified at CERN](#verified-at-cern) below
as it lands — an unanswered row is more useful than a forgotten one.

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

### N3 · Stage the run_145 development bundle

`PLAN.md` §4 fixes run_145 as the local development run and gives the pull
order. Pull in that order, because it is cheapest-and-most-useful first:
the existing parquet (MB) unblocks stages 3–5 immediately, and the waveforms
(~1 GB for one file tag) are only needed to prove the reconstruction path.

### N4 · Scaffold the package

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

Stage 0 (`freeze_sample.py` → `sample.csv` + the timeline figure), then stage 1
(the candidate filter on run_145). `PLAN.md` §3 has the specification for each;
`PLAN.md` §6 has the day-by-day target.

---

## Stage board

| stage | state | what exists | next |
|---|---|---|---|
| 0 · sample | **todo** | the frozen registries (`x17-runs.json`, `x17-match.json`) are on the site and hold every input the cuts need | write `freeze_sample.py`, produce `sample.csv` + the timeline figure |
| 1 · time base | **todo** | the flash calibration and clock QA are done and published | pick the veto window; write `t_since_flash` and E_n per trigger |
| 2 · candidate filter | **todo** | `../ntof_tracking/reco/noise.py` and `reco/segments.py` already do the clustering and the taxonomy; the slim files carry the n_TOF arm flags | write the per-trigger classifier, run it on run_145 |
| 3 · reco | **todo** | `wft_beam.py` + `../ntof_tracking/condor/` proven on run_145 (60 jobs, 4 arms, `calib_bundle_r06`) | add the event-id allowlist; benchmark on one run_145 tag |
| 4 · database | **todo** | `microtpc_lib.pair_planes`, `reco/geometry.py` (corrected sign + pinwheel), `reco/pairing.py` | fix the schema, write `build_tracks.py` |
| 5 · scint positions | **todo** | `../ntof_processing/quality_metrics.py` A1/A2 has both estimators and their caveats | recalibrate λ and the Δt scale against MM tracks |
| 6 · pairs & spectrum | **todo** | nothing | after 4 |
| — · figures | **todo** | nothing | `figstyle.py` first — N4 above |

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
same gap blocks `deploy-eos.sh`, which is why the board and the note are built
but not live.

**The fix is not to repair Windows: the analysis moves to the Ubuntu laptop**,
which has a working `kinit`, the data under `/media/dylan/data/x17/`, and every
path the repo's documentation already assumes. Nothing else was blocked by B1,
and nothing needs redoing after the move — everything written today is in git.

---

## Verified at CERN

Filled in by N2. Empty means not yet checked, not "fine".

| # | question | answer | when |
|---|---|---|---|
| 1 | is the re-slim complete, and which sub-runs have a slim product? | — | |
| 2 | which runs have a wft reco, with which bundle and code commit? | — | |
| 3 | do run_79's products get rebuilt, or patched at read time? | — | |
| 4 | link speed to CERN from the laptop | — | |
| 5 | condor throughput and AFS/EOS quota | — | |

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
| link speed to CERN | — | |

The only number in hand is the historical one: run_79, 12 534 events on
6 cores in ~1 h 50 m ≈ **1 100 events per core-hour per chamber**, on
20-sample windows with `WFT_MODEL_FRAC=0.03 WFT_PRESCAN=1`. Treat it as an
order of magnitude until re-measured.

---

## Log

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
