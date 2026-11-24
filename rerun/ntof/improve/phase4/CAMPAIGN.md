# Running the phase-4 test campaign: handoff (written 2026-11-24)

This file is for the session that runs the campaign. It says what the campaign is, what is already prepared, the
checks to make before submitting, the commands, how to turn the outputs into numbers, what decides each switch, and
what comes after. Read it with `README.md` here (the switches) and `../reaudit/report.html` (why the campaign looks the
way it does). `../reaudit/OPEN_QUESTIONS.md` lists what we do not understand.

**Dylan decides two things, not the session: whether a second pass is run after the test, and with which switches.**
The test campaign itself is agreed (Dylan, 2026-11-24: "run with the spline map and make the fixes for the
re-processing"). If you find the text below ambiguous about "the campaign", it means this test campaign of nine
configurations. Section 8 is the outline of the second pass that would follow.

## 1. What the campaign is

We re-reconstruct 15 sub-runs with each reconstruction switch on alone, with all on together, and with all off, and
compare each with the all-off reference.

- **Sub-runs:** the five beam sub-runs of the ringing study (run_145 stat090_0000 to 0002, run_104 stat090_0003,
  run_139 stat090_0003) and ten beam-off cosmic sub-runs of run_149 (cosbounce_cos_0000 to 0009). The cosmic sub-runs
  give about 1,400 A–C cosmic lines, the only angle truth.
- **Chambers:** A, C and D. B is not reconstructed (positions only; the pass-1 B files are not needed by the chain).
- **Configurations** (`make_jobs.py` `CFGS`). The depth profiles are stored in every configuration.

| cfg | what is on | chambers |
|---|---|---|
| `ref` | nothing (pass 1 on this code) | A, C, D |
| `floor30` | local seeding floor, 30 mm (4.12) | A, C, D |
| `t0prior` | t0 prior on split children (4.3) | A, C, D |
| `seedrank` | seeds ranked by charge, cap after the wide-band veto (4.4) | A, C, D |
| `headon` | head-on rescue (4.5) | A, C, D |
| `ringing` | ringing correction (4.10) | A, C |
| `tctol` | n_TOF x–y time tolerance, A 151, C 296, D 535 ns | A, C, D |
| `osc` | D's oscillating channels in the hot list (4.9) | D |
| `all` | all of the above except `osc` | A, C, D |

- **Jobs:** one per acquisition and chamber. Per configuration 154 A/C jobs and 77 D jobs (47 beam + 30 cosmic per
  chamber). In total 1,848 jobs.
- **Cost:** about 10,000 to 12,000 core-hours (measured on this morning's clusters: a beam acquisition takes about
  2 h on A, 3.6 to 5 h on C and 5.4 to 8.3 h on D, on 4 cores; with the ringing correction A and C take a third of
  that; cosmic acquisitions are short). Our limits are 1,000 cores at once and 300,000 core-hours per 30 days.

## 2. What is already prepared

| item | where | state |
|---|---|---|
| Code with the fixes | repo `main`, commit named in `HANDOFF.md` | committed and pushed |
| Baseline view with the spline map | `$WORK_DATA/ntof_pass1/stage3_r3_final/` (desktop) | see `HANDOFF.md` for whether the build finished |
| Spline map | `rerun/ntof/stage3/angle_map/angle_map_r3.json` | in the repo |
| Job-list generator | `make_jobs.py` here | tested locally |
| Submit file | `phase4.sub` here | unchanged; flavour and `maxmat` are submit macros |
| Chain and comparisons | `stage3_chain.sh`, `compare.py`, `compare_lines.py` here | tested (chain on the `ref` outputs of the morning; `compare_lines.py` on pass-1 cosmic tables) |
| Deployment on lxplus, job lists there, the validation cluster | see `HANDOFF.md` ("In progress") | done by the writing session if it says so; otherwise section 3 |

The switches are all off by default, so the code reproduces pass 1. Checked on 2026-11-24 on 400 events of run_145
(chamber A): 1,094 of 1,095 candidates are identical to pass 1, and the one that differs is a degenerate 78-strip
candidate of rank 3.

## 3. Before submitting

Use the `condor-campaign` skill for the submission itself. Read `~/.claude/CLAUDE.md`, section "lxplus storage".

1. **AFS space.** The quota was at 90 % on 2026-11-24 (9.39 of 10 GB), and `~/x17rerun/ntof/pass1/log_pass1_2026-11`
   used 93 % of its name slots. A full quota holds every job. Run `ssh lxplus lxstore status`. Free space with
   `lxstore offload` (it copies to EOS and leaves a link). Do not delete anything without asking Dylan. There are 58
   `~/x17rerun/code_*` directories; a full code extract is 725 MB, so ask Dylan which old ones can go.
2. **Deploy the code slim.** A full `git archive` is 750 MB. The reconstruction runs from an extract without the
   presentation and analysis folders (193 MB; checked locally on 2026-11-24 with every switch on):

       cd ~/PycharmProjects/nTof_x17 && H=$(git rev-parse --short=8 HEAD)
       EXC="ntof_athens_26 ntof_july_analysis garfield_sim mpgd26 ntof_italy_26 papers infn_in2p3_26 \
            weekend_summary_slides sps_beam_test_26 mx_july_beam_qa ntof_processing"
       git archive HEAD -- . $(for e in $EXC; do printf ':(exclude)%s ' $e; done) | \
           ssh lxplus "mkdir -p ~/x17rerun/code_$H && tar -x -C ~/x17rerun/code_$H"

   The code directory stays on AFS (never on EOS: `rerun/condor/README.md`).
3. **Oscillating-channel bundles on EOS** for configuration `osc` (the cosmic one, `D_E2_cos`, was not uploaded before):

       tar -C $WORK_DATA/ntof_pass1/bundles/pf_osc -cf - D_E2_beam_L1 D_E2_cos | ssh lxplus \
         "mkdir -p /eos/experiment/ntof/data/x17/july_beam/Analysis_pass1/tests/phase4_$H/bundles_osc && \
          tar -C /eos/experiment/ntof/data/x17/july_beam/Analysis_pass1/tests/phase4_$H/bundles_osc -xf -"

   Check that the pin of `bundles_osc/D_E2_beam_L1/bundle.json` is the L1 pin (`t0_abs` x "0": −17.8, y "0": −11.2).
   This morning's sizing cluster used `D_E2_beam` (pin +18.3 / +20.0), which shifted every D t0 by +36 ns.
4. **Job lists:**

       .venv/bin/python rerun/ntof/improve/phase4/make_jobs.py --code $H --outdir /tmp/p4jobs_$H
       scp /tmp/p4jobs_$H/* rerun/ntof/improve/phase4/phase4.sub lxplus:x17rerun/ntof/pass1/p4jobs_$H/

   (make the directory first). `ntof_stage.sh` is `~/x17rerun/ntof/pass1/ntof_stage.sh` (unchanged).
5. **A validation cluster first** (the budget rules require a test cluster before more than 50 jobs). Take from
   `jobs_ref_AC.txt`, `jobs_ref_D.txt`, `jobs_all_AC.txt` and `jobs_all_D.txt` the rows of four A/C acquisitions of
   run_145 and run_139, the short D acquisition `260805_16H07_008`, one run_104 D acquisition and two cosmic
   acquisitions. Submit with `flavour=tomorrow`. Then:
   - `condor_budget size <log>` for the requests;
   - `ref` must reproduce the baseline: build it with `stage3_chain.sh` and run `compare.py` against
     `stage3_r3_final`. Expect every gated track matched and differences at the 10⁻⁶ mm level. A real difference stops
     the campaign.
   - `all` must run without error and its tables must carry `x_n_sig5`, `x_rescued` and (in the candidates)
     `q_profile`.

## 4. Submitting

In `~/x17rerun/ntof/pass1/p4jobs_$H` on lxplus, one new log directory per submission:

    mkdir log_<cfg>_AC && condor_submit phase4.sub code=$HOME/x17rerun/code_$H jobfile=jobs_<cfg>_AC.txt \
        logdir=log_<cfg>_AC diag="$(cat diag_<cfg>.txt)" cpus=4 mem=4000 flavour=workday maxmat=<n>
    mkdir log_<cfg>_D  && condor_submit phase4.sub code=$HOME/x17rerun/code_$H jobfile=jobs_<cfg>_D.txt \
        logdir=log_<cfg>_D  diag="$(cat diag_<cfg>.txt)" cpus=4 mem=4000 flavour=tomorrow maxmat=<n>

- A and C fit `workday` (8 h). D needs `tomorrow`: its large acquisitions ran 6.9 to 7.6 h this morning.
- The peak memory was 957 MB. `mem=4000` with 4 CPUs costs nothing extra (1 CPU = 3 GB).
- **Stay under 1,000 cores in total**: the sum of `maxmat` over the running submissions, times 4, must stay below
  1,000 (for example `maxmat=25` on ten submissions). Submit `ref` first, then the others as slots free up. The
  `condor_submit` shim checks the budget; do not override it without Dylan.
- Outputs go to `/eos/experiment/ntof/data/x17/july_beam/Analysis_pass1/tests/phase4_$H/out/<cfg>/`. Chamber D of
  run_104 goes under `out/<cfg>/run104_D_L1/` (the canon directory; stage 3 looks there).

## 5. While it runs

- After each cluster compare job starts with completions in the user log (events `001` and `005`), and read the hold
  and abort reasons (`012`, `009`, `condor_q -hold -af HoldReason`). A failure rate is a bug to explain.
- `ssh lxplus lxstore status` now and then.
- Run nothing heavy on the desktop in parallel with the post-processing of section 6 (it uses 5 processes).

## 6. From outputs to numbers

On the desktop, per configuration (the chain fetches over `ssh lxplus tar`, about 100 MB per configuration):

    PHASE4_CODE=$H bash rerun/ntof/improve/phase4/stage3_chain.sh <cfg>
    D=$WORK_DATA/ntof_pass1/phase4
    .venv/bin/python rerun/ntof/improve/phase4/compare.py       --cfg $D/<cfg>/final --ref $D/ref/final --out $D/compare --name <cfg>
    .venv/bin/python rerun/ntof/improve/phase4/compare_lines.py --cfg $D/<cfg>/final --ref $D/ref/final --out $D/compare --name <cfg>

- Build `ref` first and compare it with the baseline: `compare.py --cfg $D/ref/final --ref $WORK_DATA/ntof_pass1/stage3_r3_final`.
- The chain holds the spline map and the quality references of `stage3_r3_final` fixed (`PHASE4_MAP`, `PHASE4_REFS`).
- `compare.py`: gated tracks per time-of-flight bin and per trigger, matched and unmatched tracks, differences of
  position, angle and t0, railing (also of split children), the same-chamber pair fraction, the pointing at the
  source, and the wall fraction of tracks only one set has. `compare_lines.py`: the cosmic-line residual width of
  each view in both sets on the same lines, with a paired bootstrap error, also for near-normal and steep lines.
- For `osc` judge D on `gated_uncensored` and `gated_precut` as well as `gated`: more hot channels raise the censored
  fraction by construction.
- For the ringing rule of 2026-11-23, apply `quality_join.py apply --view <dir>/final --low-charge r28` to a copy of
  the `ringing` and `all` views and compare the early purity (wall fraction before 12 ms) with `--low-charge q600`
  on `ref`.

**Local truth tests, no condor** (each about 25 min per chamber on 7 cores; not more than one at a time):
- The injection test for 4.3 and 4.12: `rerun/ntof/improve/t0_study/inject.py` (README there; `--floor-local-mm 30`,
  and `WFT_DIAG=reco.TWO_TRACK_T0_PRIOR=1`). The five sub-runs hold only 158 (A) and 103 (C) two-track events from
  12 ms on, so the data cannot judge these two switches.
- The Geant4 digitisation, only after the two items of `../reaudit/OPEN_QUESTIONS.md` item 7 (a simulation bundle;
  `X17_G4_W_MESH=35.1` for the EAR2-beam library). The data and Geant4 must go through the same seeder switches.

## 7. What decides each switch

Written before the campaign runs, so that the measure is not chosen after seeing the result.

| switch | adopt if | reject if |
|---|---|---|
| `ref` | it reproduces `stage3_r3_final` within tolerance | any real difference: stop |
| `headon` | the added tracks have the wall fraction of the matched tracks, and the cosmic-line width at \|slope\| < 0.1 does not grow | the added tracks point worse or the near-normal width grows |
| `ringing` | tracks from 20 ms on are unchanged against `ref`, and the early wall fraction rises | late tracks change |
| `t0prior` | fewer split children at the t0 limit, and injected pairs keep their opening angle | |
| `floor30` | injected pairs are both found more often, and the added single tracks are wall-confirmed | more fakes on D (wall fraction of added D tracks below that of matched ones) |
| `seedrank` | no wall-confirmed late track is lost on any chamber, D included | real tracks disappear |
| `tctol` | the pair fraction and the wall fraction of late tracks do not fall, and early mis-paired tracks fall (`xy_balance_ok`) | real late tracks are lost |
| `osc` | D's purity at 20 to 30 ms rises with no loss of wall-confirmed tracks in `gated_uncensored` | real tracks are lost |
| `all` | it equals the sum of the adopted single switches within errors | an interaction appears: run the pair of switches involved |

Then write the comparison as a report (a `make_report.py` next to the results, as `../reaudit/make_report.py`), and
give Dylan the table. He decides the second pass.

## 8. If Dylan decides a second pass

- **Configuration:** the adopted switches. `reco.STORE_PROFILES=1` in any case.
- **Size:** pass 1 was 15,550 jobs (3,937 acquisitions on four chambers) plus about 2,550 re-runs of D. Measure the
  pass-1 core-hours from its user logs before submitting (`condor_budget size` on the logs under
  `~/x17rerun/ntof/pass1/log_pass1_2026-11`); do not extrapolate from run_145, whose acquisitions are the largest.
  The machinery is pass 1's: `rerun/ntof/NTOF_PASS_PLAN_2026-11-02.md` (P6), `rerun/ntof/condor/`, the ledger
  (`ntof_ledger.py`, `ledger_canon.json`).
- **Bundles:** the canon bundles. Every D beam row on the L1 pin (`D_E2_beam_L1`).
- **After the reconstruction**, in this order:
  1. stage 3 with the context of `context_r2` (`ntof_stage3.py --stage1-dir/--trigtime-dir`);
  2. v(t) view, then the spline map refitted on the new cosmic lines and the view rebuilt:
     `rerun/ntof/improve/run_r3_chain.sh` with the new roots (its header gives the variables);
  3. the quality flags with the decided rules: `quality_join.py apply --low-charge r28 --b-box-cut`;
  4. the switch on EOS and in the ledger (roadmap 2.11), then `rerun/ntof/PASS1_DOWNSTREAM.sh` (about 6 h);
  5. the Geant4 digitisation with the same switches (after `OPEN_QUESTIONS.md` item 7).

## 9. Gotchas

- Compare with a tolerance, never by bytes: outputs differ in the last digits between worker nodes.
- `x_quality_ok` / `y_quality_ok` take their reference median per sub-run over the acquisitions present. They are not
  part of `gated`.
- Stage 3 reads the pass-1 bundles for the bow patch and the pins. This is right for `osc` only because its bundles
  differ in the hot lists alone.
- The cosmic sub-runs have no flash: their context columns are null and `t_since_flash_ns` is null. Select beam-off by
  registry class, never by "time is null".
- The `ringing` configuration is not run on D (the correction is for A and C).
- `wft/tests/test_two_track.py::test_accepted_split_replaces_its_parent` can fail under load and pass alone;
  `test_profile_is_retired` fails on main.
- The desktop has no xrootd binaries: the chain streams `tar` over `ssh lxplus`. Reach lxplus only as `lxplus`.
