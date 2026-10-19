# Overnight campaign pass — 2026-09-08/09

**Autonomous session. Dylan asleep. Decisions below were taken by Dylan before
bed (AskUserQuestion, 2026-09-08 ~23:30) and are binding — do not re-litigate
them on resume.**

Loop: `/loop` dynamic mode is armed; each turn re-arms a ScheduleWakeup so the
work resumes automatically after a 5-hour usage cutoff. **Re-arm every turn.**

## The four decisions

1. **Tight coincidence = both arms prompt AND mutually prompt.** Each arm's
   scintillator hit within **±30 ns** of its own DREAM trigger (the measured
   peak core, HANDOFF_ACCIDENTAL_TIMING §2.1) **and** |t_arm1 − t_arm2| ≤ 20 ns.
   Loose wall-OR-plastic tag (not strict AND — only 8/464 pairs pass strict).
   Ship a **window scan** (opening angle vs window width) as the systematic.
2. **Cut downstream, never in stage-1.** `DT_WINDOW` in `candidate_filter.py`,
   `efficiency.py`, `scintillators.py` **stays at the production (−100,+60)**.
   The slim keeps the full ±1000 ns `dt_ns` and `is_control`. The tight cut is
   a per-pair boolean column so the window is re-tunable offline all week
   without another campaign pass.
3. **Scope: stage-1 over all 293 sub-runs, then stage-2 on the allowlist.**
   ~4.1 % of a full pass, ~760 core-hours, condor at CERN. Then stage-3
   tracks + n_TOF attach.
4. **Products on `/eos/user/d/dneff/`; download budget ~50 GB** to
   `/media/dylan/data/x17/`. AFS home has only ~4 GB free — never stage there.

## Also in force

- `no_hotstrip` is the production hot-channel cut (HANDOFF_HOT_WILDCARD §6).
  `hot_seed_strata` must be rebuilt **per run** and run **before** `k_arm`.
- Hot/dead classification and every calibration is **per run condition**
  (CLAUDE.md). run_67/68 excluded; run_79 needs the A-x 448–511 mask.
- Geometry comes from waveforms, never from `combined_hits` times
  (RECONSTRUCTION_BASIS.md).

## Progress log — append, never rewrite

- 2026-09-08 23:3x — decisions recorded, wakeup armed. Starting.
- 2026-09-09 08:1x — **the overnight run did NOT happen.** The loop was armed
  but re-arms only if `ScheduleWakeup` is the last action of each turn; on the
  23:27 tick it was not called, so the loop died after one iteration. Stage 1
  was still 11/293 and no condor job had ever been submitted. Cause recorded
  here because the failure mode is invisible from the products.
- 2026-09-09 08:2x — **stage 1 moved to condor** rather than re-run locally:
  `campaign_census.sh` is 17-25 h on the desktop (its own header says
  "multi-night"), the data is already at CERN, and `paths.py` already resolves
  every root through an environment variable, so the unmodified CLI runs on a
  worker. New: `condor/{make_stage1_package.py,stage1.sub,run_stage1_wrapper.sh,
  fetch_stage1.sh}`. 282 jobs submitted (clusters 4141478, 4141479), all
  running. Two path bugs found by smoke test first: xrootd needs `root://host//eos`
  (double slash), and `ntof_tracking.reco.io` resolves through
  `common/beam_july_paths.py::X17_BEAM_JULY`, which wants the PARENT of `runs/`.
  Measured 29.3 ev/s on a worker -> ~1.2 h per sub-run.
- 2026-09-09 09:0x — **tight coincidence built** (`tight_coincidence.py`,
  `make_tight_figures.py`), downstream only, production `DT_WINDOW` untouched
  as decided. On run_145: 76 loose-tagged inter-chamber pairs, 20 pass
  |t_arm| <= 30 ns and |t1-t2| <= 20 ns.
  **Finding not anticipated by the plan: the tight cut ENRICHES a
  single-particle background.** 5 of the 20 survivors are opposing-chamber
  (A-C) pairs above 170 deg -- one particle crossing the target and punching
  through both opposite chambers, which is perfectly time-coincident because it
  is one particle. 8 of the 11 tight opposing pairs are above 150 deg against
  25 % of the loose sample. This is `HANDOFF_ACCIDENTAL_TIMING.md` sec 5's D12
  caveat showing up in the data. Flagged as `back_to_back`/`tight_pair`, never
  silently dropped. **Any X17 statement from a timing-coincident sample needs
  this veto first.**
- 2026-09-09 10:0x — **n_TOF slim COMPLETE: 293/293 sub-runs, 9.6 GB local.**
  Two condor lessons paid for on the way, both written up in
  `condor/EOS_WRITE_TEST.md`:
  * **`transfer_output_remaps` to EOS does not work.** The ACCESS POINT (the
    schedd) performs the transfer and has no EOS mount, so the job runs to
    completion and is then HELD (`errno 2`). Measured, cluster 4141483. What
    works is the job pushing its own output with `xrdcp` and handing condor
    back only a small marker. `transfer_output_files` must then be named
    explicitly or HTCondor returns the whole scratch directory.
  * **AFS home is 10 GB and I filled it to 100 %**, holding 3 stage-1 jobs
    mid-transfer. Large per-job output on AFS needs a DRAIN; a monitor now
    pulls and deletes every 4 minutes and warns at 85 %.
- 2026-09-09 10:1x — **`k_arm` now reads the parquet slim**, so campaign
  calibration needs no ROOT locally. `TI.pointing_coincidence` gained an
  optional pre-loaded-frame argument (default None = unchanged ROOT path).
  **Verified bit-identical on run_145: A 1.2661593, C 1.6162916, D 1.7666667.**
- 2026-09-09 10:2x — **the shipped tracks table keeps the FULL stage-3 schema.**
  An earlier version projected to 83 columns and float32 to hit a ~50 GB
  budget; that budget was my own invention, not a requirement (Dylan), and the
  fit diagnostics are what a resolution or gate-efficiency study needs. ~10 GB
  campaign-wide against 214 GB free.
- 2026-09-09 10:2x — **`source_imaging._track_table` filters on
  `angle_calibrated`**, so a run whose k is computed AFTER `build_tracks`
  contributes nothing to the pair analysis, silently. `campaign_chain.sh`
  therefore runs merge -> k_arm -> tracks in that order. The track source is
  now a parameter (`src=`), defaulting to the run_145 `stage3_fullpass` store
  so no published product is repointed or overwritten.

## FINDING — a third campaign condition boundary, at the 27 July access

**Found 2026-09-09 from the campaign stage-1 census (152 sub-runs in), which is
the first time stage 1 has ever run across the whole campaign.**

Median CLEAN strips per trigger (after noise flagging and the hot mask), by run:

| run | A | B | C | D |
|---|---:|---:|---:|---:|
| **79** | **10.4** | 5.8 | **1.6** | **18.2** |
| **81** | **11.0** | 3.0 | **2.0** | **20.0** |
| 84 | 0.0 | 4.0 | 0.0 | 13.0 |
| 86 | 0.0 | 3.8 | 0.0 | 13.8 |
| ... | 0.0 | ~1-2 | 0.0 | 11-14 |

**run_79 and run_81 sit on one side of a step; run_84 onward on the other, and
the step is the 27 July access** -- the same access `CLAUDE.md` records as
fixing chamber A's x-view connector 8. Arm A goes from a median of ~10 strips
per trigger to **zero**, and C from ~2 to zero. B and D drop too but less.

**The consequence, and it is the reason this matters:** run_79 and run_81 carry
**about twice** every other run's INTER and INTRA fraction --

| | INTER | INTRA | NONE |
|---|---:|---:|---:|
| run_79 | 0.0192 | 0.0529 | 0.7357 |
| run_81 | 0.0207 | 0.0504 | 0.7279 |
| every other run | ~0.010 | ~0.025 | ~0.785 |

INTER is the signal topology. Pooling these two runs with the rest would put a
detector-condition artefact straight into the campaign pair rate.

**It is NOT the beam and NOT the trigger.** The n_TOF arm coincidence fraction
is the same on both sides to within a percent (A 0.156 vs 0.167, B 0.232 vs
0.235, C 0.228 vs 0.229, D 0.214 vs 0.214) -- the scintillator side did not
move, only the Micromegas occupancy did. **It is also not the hot mask**: run_79
and run_81 have FEWER masked channels (59 per sub-run against 93), so the mask
is removing less there, not more.

**Most likely a threshold/pedestal reload at the access**, with A's connector
repair on top -- a disconnected input floats and can register noise, so fixing
connector 8 plausibly REMOVES A hits rather than adding them, which is the
direction actually seen. Not established here; what IS established is the step
and its size.

**What to do:** treat 27 July as a condition boundary alongside the 23 July
noise change. run_79 and run_81 are 9 of the 293 sub-runs and ~0.9 M triggers;
quote them separately or exclude them, and never pool an INTER/INTRA rate
across the boundary without saying so. `sample.csv` already flags run_79's 16
sub-runs via `flag_a_x_mask` -- that flag now marks a broader condition than
its name suggests.

## DATA LOSS — one corrupt decoded_root file, 2026-09-09

`run_104/stat090_0016`, tag `260730_11H29_000`, FEU **03**:
`Mx17_stat090_0016_datrun_260730_11H29_000_03.root`

**It is 84,838,747 bytes on EOS -- the full expected size -- and contains no
ROOT keys at all.** `uproot.open(...).keys()` returns `[]`; there is no `nt`
tree. Copied fresh with `xrdcp` the copy is byte-complete and equally empty, so
this is corruption AT SOURCE, not a staging failure.

**A size check does not detect it, and that is how it fooled the first
diagnosis here** -- the file was declared healthy from its `ls` size, which is
exactly the wrong evidence. Only opening it shows the fault.

Consequence: arms A, C and D cannot reconstruct that ONE tag (B does not read
FEU 03). Those three condor jobs fail deterministically and were removed rather
than released -- an auto-release loop would retry them forever. One tag of
~2,900 in the campaign; the sub-run's other 9 tags are unaffected.

**Not yet checked: whether other decoded files share the fault.** A campaign
scan for *zero-size* files found none, but this file is full size, so that scan
was blind to it. Detecting it needs each file opened.

## MEASURED — does the angle scale k drift across the campaign? 2026-09-09

The falsifier stated when every arm was pinned to its run_145 bundle was: *if k
varies strongly run to run, one bundle is wrong.* Measured, on the dedicated
calibration pass (SINGLE prescale 0.25 against the main pass's 0.05, 23
sub-runs over six runs spanning 27 Jul - 10 Aug).

| arm | n runs | min | max | mean | spread |
|---|---:|---:|---:|---:|---:|
| A | 6 | 1.141 | 1.184 | 1.167 | **3.7 %** |
| C | 6 | 1.350 | 1.550 | 1.410 | 14.2 % |
| D | 3 | 1.550 | 1.650 | 1.584 | 6.3 % |

B yields no measurable sample in any run, as expected (`PLAN.md` §2.3).

**Arm A -- the only arm whose k was ever credible -- is stable to 3.7 % across
two weeks.** So the run-to-run drift is small.

### The anchor is what makes this interpretable, and it fired

run_145 was included precisely because its k is already published from a FULL
local pass. The same run, measured from the calibration pass, does NOT
reproduce it:

| arm | full pass (published) | calibration pass | offset |
|---|---:|---:|---:|
| A | 1.2662 | 1.1840 | **−6.5 %** |
| C | 1.6163 | 1.5500 | −4.1 % |

**The method-to-method offset (6.5 %) is LARGER than the run-to-run spread
(3.7 %).** And every run's A sits below the published 1.2662 (1.141-1.184),
which is what a systematic offset predicts and what genuine drift would not --
drift would scatter about the full-pass value, not sit uniformly under it.

**Reading: k does not meaningfully drift; the estimator is sample-dependent.**
Pinning every arm to run_145's bundle is therefore defensible, and the price is
a systematic on the angle scale of roughly **4-7 %**, which is larger than the
campaign drift it was feared to hide.

**Caveat, stated because it weakens the anchor:** run_145's calibration-pass
values did NOT certify (A reproducibility 10.5 %, C's band/track estimators
nan), so the offset is suggestive rather than established. What is solid is the
A spread across six runs; what is inferred is that the offset is method rather
than time.

v_insitu for A rises gently over the campaign (36.0 → 37.3 µm/ns), the
direction a drying-gas trend predicts, and consistent in size with the 3.7 %.
