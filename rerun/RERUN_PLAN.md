# The from-scratch re-run of the cosmic-bench and SPS analysis (plan, 2026-10-23)

**Goal (Dylan, 2026-10-23):** one complete, consistent pass of every June cosmic-bench and SPS det4 analysis, on
lxplus condor, with the physical detector model, producing every number and figure of the two detector papers.
n_TOF is out of scope for these papers. The pass should run **once**, so everything that would force it to run again
is settled first. This file is the plan; `CHECKS_2026-10-23.md` is the record of the checks behind it.

## 0 · Decisions taken

| | decision (2026-10-23) |
|---|---|
| Kernel | **The physical RC kernel** (`wft.model.build_matrix_rc`): prompt footprint σ0 per view, RC spread along the resistive strips (D_rc, Y only), drift diffusion; no fitted copy amplitudes. Supersedes the 10-23 morning note "C2 = freeze r06". |
| X asymmetry | Not modelled; `wft.xasym` is a ± systematic (XASYM_CONTINGENCY.md). Toy cost: −0.05° overall, ∓0.19° head-on. |
| Inputs | Keep EOS `decoded_root`, `combined_hits_root` and M3 v2 rays where a fresh decode of a sample reproduces them; re-decode what does not. |
| Where | lxplus condor only. Outputs on EOS; AFS holds code and logs only. |
| Scope | All June bench rows (golden long runs, HV scans, det3 drift scan), all SPS det4 runs, timing on waveforms. The hits-vs-waveforms comparison (Paper II R1–R5) is a side project at the **end**, on comparison datasets only. |
| Datasets | Golden runs set the calibrations and the quoted numbers; the other runs are consistency checks, read skeptically where a run is known to be bad (§5). |

## 1 · What forces a re-run, and what does not

Only **upstream** choices force the full pass to be repeated: the inputs (decode, hits for seeding, M3/uRWELL
reference), every calibration constant in a bundle, the reconstruction settings (waveform preparation, depth grid,
solver, seeding) and the alignment. Everything **downstream** of the reconstructed tables — stacks, gas fits, D_T,
efficiency definitions, timing, maps, figures — can be iterated after the pass for the cost of a script. The pre-run
work below is therefore about the upstream list; the downstream questions (§6) are worked during and after the pass.

## 2 · What the checks found (CHECKS_2026-10-23.md)

| | finding | consequence |
|---|---|---|
| C3 | **Five June nights decode differently** with the current decoder (fa80c06, RAW eventID split): det1/det2 6-17, the det2/det3 6-22 HV scan, det3/det4 6-23, **det4 6-24 (= g_det4)**, the det6/det7 6-26 HV scan. 1–54 extra events per file; 0.1–0.4 % of events in those files are two triggers fused. Identical: det3 6-27 (golden + drift scan), det6/det7 6-26 overnight (golden), det2/det3 6-22 long runs | re-decode (W9), and re-make the hits of those nights |
| C5 | The depth grid (18 × 60 ns = 1080 ns) is shorter than the drift column on the slow nights; det6's column overruns it and the window | depth grid per dataset (W2) |
| C6 | The unregularised depth-charge solve runs away on 0.2–2 % of bench fits; the column guard (opt-in, written) removes it on all five chambers with 98–99.5 % of fits bit-identical and σθ within 0.008° | guard on by default (W2) |
| C7/C8 | **The calibration's reference position comes from the hits alignment** and biases the fitted v by 3–15 %. With the position profiled, the RC calibration's v comes within 2–5 % of the independent timing v (gap/T) on det2/3/4/7 (det4 needs K = 24); a +3–5 % residual is left for W1 | calibrate with the position profiled, reference rebuilt from the waveform alignment (W1, W12) |
| C9 | **The reconstruction's common mode moves with wide clusters**: baseline error up to ~100 ADC on every strip of a 64-channel block, 10–25 % of a neighbour's tail | signal-masked common mode (opt-in, written), gated before the pass (W11) |
| C10 | **Common mode** is removed only in software (all June rows RAW); FEU-wide plus 9–20 ADC per chip, X/Y anti-correlated; the plain median biases the far neighbours' charge low (−1…−25 % at ±5) | masked CM (W11) |
| C11 | **Decoder audit**: fa80c06 = an independent decoder on all 1278 June files; ≤ 0.23 % incomplete events; no M3 ray lacks its detector event | June re-decoded (W9) |
| C1/C2/C4 | Code and venv run on lxplus (tests pass); the decoder builds; all 38 registered SPS sub-runs have det4 raw on EOS (uRWELL hits missing for run_61 m100V and run_70) | — |

det6 is slow by every v estimator (timing 25.9, RC 21.7–27.7 against det7's 36–37 on the same night and field): a
property of det6 (wetter, or a weaker effective field), not of the kernel. It gets its own v and is quoted with that.

## 3 · Pre-run work packages (all close before the code freeze)

| id | item | state | closes when |
|---|---|---|---|
| W1 | **RC calibration protocol**: planned loop does not converge (C16); waveform χ² flat along v (C17). Candidate (arm D): v from timing, σ0_x, σ0_y, D_rc,y fitted with p0 profiled ±2 mm about the waveform-aligned reference; stable under a rebuild (resolution 0.03°, σ0 5 %, D_rc ≤ 19 %) | **done (2026-10-25 night): the physical RC model `phys`** (C21, §10.1); bundles `<golden>/wft/calib_bundle_phys`, w0/kw measured under them; promoted to `calib_bundle` by `rerun/promote_phys.py` (§11) | protocol chosen |
| W12 | **Reference geometry**: waveform alignment built (01_alignment on RC reco); differs from the hits alignment by 1–3 mm in z only; cannot pin the plane finely enough for a pinned v (C16) | done (finding) | — |
| W2 | **Depth grid and solver**: K = 24 (30 on drift ≤ 500 V) in the registry and bundles; column guard on in every pass job (WFT_COL_GUARD=0.01) | done | — |
| W11 | **Signal-masked common mode**: investigated (C10), gated on the fit (C15): **not adopted** — no resolution gain, χ² not better; plain median in the fit and the hits; masked−plain constant shifts (D_rc ≤ 19 %, v ≤ 6 %) carried as a systematic | **done (off)** | — |
| W3 | **Channel layer**: final gain, dead and hot masks per chamber (`strip_gain_audit.bad_set`, incl. 10-19's inclined-track holes). C20: no bench bundle ever carried a mask or gain; `seedfit/channel_census.py` tables in `rerun/channel_layer/` (noisy, shorts, dead, weak, map check); det7 X shorts censored recover ~40 % of its lost angles | **done, CLOSED 2026-10-26**: census dead + shorts censored in the bundle of all five chambers (Dylan: one treatment everywhere; review in `channel_layer/README.md`: masks verified on waveforms, the det4/det6 gate shifts were noise); `WFT_DEAD_SEED` on by default in wft and the beam seeder (production A/B null) | — |
| W4 | **Template**: one X-derived electronics template for both views (`rc_tmpl='x'`); saturated samples censored | decided | charge results use unsaturated events (the template reads saturated charge 5–12 % low, §41) |
| W5 | **t0 prior**: one policy for all chambers under the RC kernel | **closed (2026-10-26): no prior.** C18: the trigger t0 prior has no effect on w (paired Δ 0.00 removed or moved 20 ns); the phys bundles carry `t0_prior.sigma = 0` | gate on the golden keys |
| W6 | **Angle constants w0/kw**: `rerun/w12/set_w0kw.py` measures them under the final bundle (finalize/vrefit DAG nodes) | done (tooling, tested) | — |
| W7 | **Per-condition protocol**: `rerun/pass_registry.csv` — HV scans inherit the group bundle; other nights / fields refit v (14 groups); under arm D those need their own timing v (port headon_cache + unsat_width) | **phys policy (2026-10-26):** `make_pass_dag vrefit --keep-v` — no waveform v fit (C17); each v group keeps its golden bundle (+ its registry K) and gets its own w0/kw; depth v = timing v downstream (S4) | — |
| W8 | **Run-quality registry**: classes in `rerun/pass_registry.csv` (5 golden calib, 2 golden quoted, 107 consistency, 45 skeptical, 11 excluded); 01H29 false start excluded in `rerun/inputs.py` | done (for review) | — |
| W9 | **Inputs**: every June row re-decoded with fa80c06 into `decoded_root_fa80c06/` and audited against an independent decoder (C11: 1278/1278 identical; M3 rays consistent, no re-tracking); hits regenerated from them (**`*_rr1_cm0` adopted**, plain CM, C12/C13; `*_rr1_cm5` deleted 10-23); still to do: decode SPS FEU 03; matched lists on lxplus with the lock venv; incomplete events (audit npz) out of every efficiency denominator | decode + re-hit done; **SPS FEU 03 decoded + audited: 280/280 non-empty files** (run_60 overnight_10 file 005 is 0 bytes at source); matched lists built in-job from the M3 rays (lock venv, NClus OK); incomplete events out of every denominator via the stager | every row has inputs with one decoder provenance |
| W10 | **Software**: path layer done (C14): `rerun/inputs.py` (input generations, false-start exclusions, EOS stager with INPUTS.json + incomplete_events.json), `qa_config` X17_BENCH_ROOT/X17_ANALYSIS_ROOT, `rerun/stage_job.py/.sh/.sub` (one stage, one row, any stage unedited), `rerun/RUNBOOK.md`; regression test portable (69/69 on lxplus) | **done** (dry run of every stage on sat_det3) | a dry run of every stage on one row |

## 4 · Stages of the pass

| stage | what | gate before the next stage |
|---|---|---|
| S0 | Re-decode + re-hit the June rows; decode SPS FEU 03; matched lists for every row; inventory | every row has decoded, hits and reference; counts logged against the old ones |
| S1 | Calibrate each chamber on its golden calibration run (W1–W6, W11, W12), with the iteration | gate table on the golden evaluation run: resolution, bias, slope, implied-v flatness, efficiency, χ²; v against timing and Magboltz |
| S2 | Reconstruct every row with its per-condition bundle (W7), one job per (row, file) | completion ledger, no row missing |
| S3 | Alignment → efficiency → angles → maps → digest, per row | the digest reproduces S1's gate on the golden rows |
| S4 | Detector characterization: v(E) and gas, gap, D_T, D_rc, σ0, attachment, gain and HV curves, sparks, charge balance, timing on waveforms (D3), edge tilt (D4), X asymmetry; SPS: kernel, spatial resolution on the waveform fit (centroid as the cross-check), known-angle test at 25.6° (R7) | each result reproduces its pre-run value or the difference is explained in writing |
| S5 | Consistency checks on the non-golden runs | a table per chamber |
| S6 | Paper figures and numbers from `papers/*/figures.toml`; reports | every `\requote` replaced |
| S7 | Side project: hits vs waveforms (R1–R5) on the comparison datasets | — |

Rough cost: W1–W12 a few working days (W1/W12 and W9/W10 are the large ones); S0–S3 one to two days of condor;
S4–S6 about a week.

## 5 · Run-quality registry (draft — for Dylan to confirm)

| class | rows | why |
|---|---|---|
| golden, calibration | det3 sat_det3 · det2 o22_long_det2 · det4 g_det4 · det6 g_det6_long · det7 g_det7_long | the 8-12 calibration runs |
| golden, quoted | det3 g_det3_wknd (22 k rays) · det2 g_det2 (19 k) · det4/6/7: as above, held out by event | the largest clean runs, a different run from the calibration where one exists |
| consistency | HV scans, short/longer runs, det3 drift scan, det7 longer run (4σ different v from its long run) | trend-grade |
| skeptical | det2 6-17 overnight (half-dead X; fused events in its decode), det3 6-23 (degraded bottom-slot night; fused events), det3 6-22 Monday (unusable micro-TPC angle) | known problems; re-judge 6-17 and 6-23 after the re-decode |
| asterisk | det7 g_det7_long / g_det7_longer (6-26 night): M1 cards badly seated, pogo-field shorts on X card 1 (2026-10-25) | quote det7 X from ch 256–511 (connectors 5–8) only; `rerun/det7_pogo/README.md`, `rerun/pogo_check/` |
| excluded | det1 (no calibration), pre-June, ArIso, stale duplicate trees, telescope-off rows 90/91 | 8-12 manifest reasons |

## 6 · Downstream questions (do not block the pass; worked in S4)

- det3's X diffusion excess (+160…+260 µm/√cm) — re-measure with the masked common mode and a full-column selection.
- Water per chamber and night; det6 slower than det7 on the same night.
- Three-body O₂ attachment scale (beam).
- det4's X under the RC kernel (+0.06…+0.16° against production).
- The X asymmetry's mechanism (closed unexplained).

## 7 · Stated limitations (no new data)

The detectors are gone. Anything not answerable from the existing data, simulation or the literature is written down
as a limitation: the asymmetry mechanism, the resistive-strip mask, the τ and c2 tails beyond the SPS window, a
dry-gas bench run, Stephan's unanswered geometry questions, det6's window-limited deep charge.

## 8 · Decisions on the open questions (Dylan, 2026-10-23 evening)

1. **Re-decode every June row** with one decoder, after an audit that the decoder is now correct; audit the M3
   decoding/tracking chain the same way.
2. **Analysis first.** Quote from the largest-statistics runs if they show no problem; fall back to other runs if they do.
3. **Diffusion: per-chamber measured Dp** (Magboltz as the comparison), but its consistency across chambers and runs
   is a result to be checked; if the measurements disagree, write it up as a limit of our understanding with an
   error bar, not hide it.
4. **det6: full analysis**, every number produced; what to quote is decided at writing time (det6 is known poor).
5. C8: redo the calibration fully with the reference position from the waveforms. C9: investigate the common mode
   thoroughly and build the correction into the analysis.

## 9 · Settled on the night of 2026-10-23 (C13)

- **Hits common mode: plain median** (`*_rr1_cm0`). The masked version lost 0.5–30 points of track matching (sparks
  and saturated chips on det6/det7); the masked CM stays in wft only, with the coherent-event guard.
- **M3: production v2 rays stand.** Rebuild reproduces them; outlier rescue and masked CM judged against the
  detectors and rejected / not needed.
- **Run registry (W8)**: exclude det3 6-27's 01H29 false start (still in EOS `raw_daq_data/`; new products moved to
  `_false_start_01H29/`).

## 10 · Decisions (Dylan, 2026-10-25 late evening)

1. **Kernel for the pass: `phys`** (C21 recipe, profiled σ0) on all five chambers. timeD is retired as a pass option.
2. **Efficiency loss budget**: where every M3-crossing muon goes, on the waveform reco of the golden rows, per view and
   X+Y, at the 5 mm match circle (MPGD26 comparability) and at 3 mm.
3. **Channel masks (W3) go into the pass bundles** (all five, decided 2026-10-26 after the review) after an overnight gate (masked vs unmasked phys, held out: s68,
   far, χ²); adopted per chamber only where they do not hurt.
4. **Fiducial**: headline numbers on the interior (≥ 30 mm from the own-coordinate edge where unpassivated); the edge
   bands are a separate characterization (C22 follow-ups). Downstream, does not block the pass.
5. Edge bending is expected (parallel-plate fringe field). Degrador rings: only det B's removal before n_TOF is documented; the bench ring state per run was never audited (memory drift-cage-degrador-rings) — a stated limitation, not a fact to argue from.
6. Two-track refit: off for production (C22 follow-ups).

## 11 · Ready to run (2026-10-26 night): the morning procedure

> **Superseded (2026-10-28): the procedure for the pass is `PASS_PLAN_2026-10-28.md`.** What below is stale: the kernel
> is the `fe` recipe rebuilt from scratch (not `phys` + `promote_phys.py`); `stage_job.sh` sets no `WFT_*` (every
> setting is built into wft); vrefit is ~117 own-v rows (468 nodes), the pass ~758 nodes; the code archive is a new
> `code_<hash>`; two-track is on (reco.TWO_TRACK, 10-27). Kept as the record of the 10-26 machinery.

> **On hold (2026-10-27, §12):** this procedure runs today's `wft`. Dylan's 10-26 call (one rewritten reconstruction)
> changes seeding and reconstruction settings, which §1 says force a re-run. Do not submit it until §12 steps 1–5 close.

State: kernel `phys` (§10.1); channel masks on **all five chambers** (W3 closed 2026-10-26, `channel_layer/README.md`;
the seeder bridges dead strips, `WFT_DEAD_SEED` default on); t0 prior off (W5); v groups by `vrefit --keep-v` (W7: golden bundle, the group's own K, w0/kw measured and
the mean kw folded into v = the row's ladder v); `stage_job.sh` sets `WFT_COL_GUARD=0.01` and `WFT_PIVOT_START=1`;
inputs rr1 (W9, SPS FEU 03 decoded). Code: **`/eos/user/d/dneff/x17rerun/code_d93c687`** (or a later archive of main; it must
include d93c687 for the dead-strip seeding default).
Dry runs (`/eos/user/d/dneff/x17rerun/Analysis_dryrun`, `~/x17rerun/dryrun_1026{,b}`): vrefit --keep-v on det3 6-25
(500 V) and the pass on sat_det3 + that row + an inheriting row, --limit 300: every node rc 0, every stage wrote.
On `code_469cbb1`, vrefit --keep-v on the 500 V drift-scan row wrote K = 30 and folded v 35.8 → 15.8 µm/ns (kw 1.00 /
1.00); the timing v there is 12.5–14.5 (unsat_width), so the folded (ladder) v is the angle scale only: depth takes the timing v.

```bash
# on lxplus, from a clone of main at the pass commit (promote_phys.py reads only EOS + its own file)
C=/afs/cern.ch/user/d/dneff/x17rerun/code_469cbb1
# 1. S1 close: promote the golden bundles -- DONE 2026-10-26 ~02:30 (PROMOTED.json in each calib_bundle):
#    det2/det3/det7 <- calib_bundle_physmask (finalize arm under the mask; det7 X kw 1.034 -> 1.005), det4/det6 <- calib_bundle_phys
#    python3 $C/rerun/promote_phys.py --mask det2,det3,det7 --apply
#    RE-PROMOTED 2026-10-26 17:48: all five <- calib_bundle_physmask (det4/det6 finalized under the mask on code_d93c687,
#    ~/x17rerun/finalize_physmask_det4det6; w0/kw moved <= 0.005)
#    python3 $C/rerun/promote_phys.py --mask det2,det3,det4,det6,det7 --apply --force
# 2. S1b: the 14 v groups (56 nodes, ~1 h)
mkdir -p ~/x17rerun/pass_main/log && cd ~/x17rerun/pass_main && cp ../pass/stage1.sub .
python3 $C/rerun/make_pass_dag.py vrefit $C --keep-v && condor_submit_dag vrefit.dag
# 3. S2+S3, after vrefit exits 0: every non-excluded row (159 rows, 655 nodes)
python3 $C/rerun/make_pass_dag.py pass $C && condor_submit_dag pass.dag
```

Then the downstream (§4 S3–S6, iterate freely): `rerun/effbudget/budget.py` on the pass tables (the budget of
2026-10-26 is on the 8000-event finalize tables), the timing v per v group (S4, `cloud_basics/unsat_width.py` per row),
det7 X quoted from ch 256–511.

## 12 · Before the pass: what still blocks it (status review, 2026-10-27)

> **2026-10-28: the live checklist is `REPROCESSING_TODO.md`.** B4 (model audit) is closed: FA–FE and round 3 (GA–GF)
> in `model_audit/FOLLOWUPS.md`. The `fe` recipe is the reconstruction, and every calibration is redone from scratch.

**Not submitted.** As of 10-27 there is no `~/x17rerun/pass_main` on lxplus and no condor jobs. The §11 machinery is
ready and dry-run tested. Running it now would mean running it twice, because two upstream items are still open (§1).

### Blockers (upstream: each would force the pass to re-run)

| id | item | closes when |
|---|---|---|
| B1 | **One rewritten reconstruction** (Dylan, 10-26). Fold in the validated prototypes as one algorithm for every detector, not as `WFT_*` switches: the band veto (`DROP_FLAT_HITS`), the head-on rule and rescue (`HEADON_SIG`/`HEADON_RESCUE`), x↔y charge pairing, the joint two-track fit (`TWO_TRACK_FIT`), per-channel and bottom rails with multi-start (satblob), and `PIVOT_START` as built-in behaviour rather than a `stage_job.sh` setting. The spec is the measured behaviour in HANDOFF (satstudy section). Added 10-28 (Dylan): the LM variable-projection outer solver with t0±60 starts and covariance errors in place of grid + Nelder–Mead — spec in HANDOFF "Fitter: LM variable-projection outer solver", review `model_audit/algo_review/REPORT.md`; ~5× fewer solves, same angles (gate it), not a resolution lever. | the code is on main, the tests pass, and the switches are retired |
| B2 | **Fleet gate on the rewrite.** Every prototype was measured on det3 only (plus injection for head-ons). Gate it on all five golden rows the way W3 was gated: s68, far fraction, χ², efficiency, held out. If w0/kw move, re-run the finalize and promote steps (`promote_phys.py`). | a gate table on all five chambers; no chamber gets worse, or the cost is accepted in writing |
| B3 | **SPS det4 in the pass.** The §0 scope includes every SPS det4 run, but `pass_registry.csv` has 170 bench rows and no SPS rows. Nothing in `rerun/` stages SPS (only the W9 FEU 03 decode is done). Needed: registry rows, a stager (uRWELL reference; run_61 m100V and run_70 lack uRWELL hits), and the bundle policy for SPS. | SPS rows dry-run through `stage_job` like sat_det3 did |
| B4 | **Model audit** (a separate session, night of 10-27): an audit of the whole detector/reco model before the freeze. Its findings feed B1/B2. | the audit's findings are triaged: each one either fixed before the freeze or recorded as a systematic or limitation |


### B4 · Model audit, night of 2026-10-27 — triage (details and numbers: `rerun/model_audit/FINDINGS.md`, report `~/x17/cosmic_bench/model_audit/report.html`)

Audited: the phys kernel and the pass's fitter on the five golden rows (finalize tables + 1320 held-out events per chamber
through the pass's own fitter, event-paired arms on condor) and SPS det4 (run_63 25.64° drift ladder, self-seeded, bench
det4 bundle).  Verdict: the fleet tells one consistent story (ladder on timing v within 2–5 %, resolution 1.0–1.9°,
positions 0.50–0.76 mm, efficiency budget reproduced), with **one meaningful improvement left in the model — the angle
response compresses with angle** — and a set of bookkeeping defects, now fixed or written down.

| # | finding | status |
|---|---|---|
| F4/F18/F21 | **Angle response nonlinear on the bench**: implied v falls 4–15 % from \|tan\| 0.1 to 0.4 on every chamber; steep tracks read 0.25–1.5° shallow (det3 2–5 %, det4/det6 10–20 %); the model, not the optimiser, prefers the shallow angle (93–96 % on det3, 83–91 % det4/det7). Mechanism: the width–tilt trade — every unit of modelled lateral width (σ0 profile ~1 %, Dp ±30 % → ∓1.5 %, rc_D_y ×1.5 → −1…−2.6 %) lowers the slope; the data want less width at depth on steep tracks. | **B1 input**: a narrower deep footprint for inclined tracks (or per-event Dp) and σ0 chosen from a measured quantity, not by χ²; gate on the steep bins |
| F16 | **SPS 25.64°**: fitted slope = the fit's own column duration to +4 % at 235 V/cm (model self-consistent); 11 % low against the span-route v because the span anchor / 28.8 mm gap sit 14 % above the column v at every field; ≤ 200 V/cm the column overruns the depth grid (K margin now +15). | **LIMIT** for the beam v (needs an independent gap); not a model defect |
| F7 | Column plausibility window hard-coded for v = 36.6: 23–60 % of det4/6/7 Y tracks "implausible" | **fixed** on main: `u_window(cal)` = (0.30, 1.60) × 30 mm / v |
| F5/F11 | dt_xy stale by one 60 ns bin (det4 empty), x–y gate fails on 15–44 % of single tracks, n_tracks = 0 on 40–80 % | **fixed (tooling)**: `set_w0kw.py --table` re-measures dt_xy + `xy_pairing`; add to the finalize/vrefit W0KW nodes (`--table {out}/wft/.../events.parquet`) and widen the gate to the measured width |
| F2 | RC kernel is a 5 ns staircase in t0; t0_err NaN on 27–31 % of det6/det7 fits | **fixed (opt-in)** `WFT_RC_T0_INTERP=1`; arm `t0i` |
| F1/F6 | t0 prior on for det2/3/4, off for det6/7; inert as written; stored `t0_abs` is another kernel's (pinning moves the ladder +2–5 %) | **policy**: sigma 0 everywhere, or re-measure `t0_abs` under phys and scale the prior by χ²/dof |
| F3 | per-event `tan_err`/`p0_err` are constant floors (1°, 0.33 mm) | **systematic**: downstream weights use per-chamber resolution |
| F8/F9 | X–Y ladder split (det3 2.2 %), w0_y −0.16…−0.40 µm/ns everywhere, field-dependent | **systematic** where kw/w0 cannot be measured (beam) |
| F12 | SPS: bench dead list is in bench channel numbers; ZS zero samples are not measurements | **fixed (tooling)**: `sps_reco.py` drops the list; `WFT_ZS_CENSOR=1` |
| F13/F17/F19 | K = 30, MODEL_FRAC 0.03/0.05 (far +1…+12 pp: reject), SAT_SOLVE (far −0.1…−0.4 pp), t0 interpolation (angles unchanged, t0 continuous), two-sided start (det4 X far −1.2 pp), X asymmetric copy (≤ 0.05° on det7 X, needs per-chamber amplitude) | **null / small** (closed; adopt t0 interpolation for correctness) |
| F22 | **Fleet gate (B2)**: band veto +0.7 / +0.8 / 0.0 / +0.35 / +1.1 pt detected (det2/3/4/6/7), head-on rescue +0.4 / +0.3 / +3.2 / +1.4 / +3.7 pt, both at no resolution cost to the existing tracks; two-track fit detection-neutral but det6 accepts 417 splits of 8000 | **pass** band + head-on; two-track needs det6's false-split rate |
| F15 | ln(qx/qy) width 0.058 (det3) … 0.28 (det7): charge pairing is chamber-specific | **systematic** |
| B3 | SPS rows: `rerun/sps_registry.csv` (71 plateaus, 13 datasets, from datasets.py), stage `rerun/model_audit/sps_reco.py` + `sps_stage.sh` (dry run on condor) | **done** (wiring); bundle policy = bench det4 bundle + v(E) per plateau + K; the 10 % angle-scale deficit (F16) is the open physics |

### Decisions for Dylan (part of B1)

- **The trigger-track rule** (80 % held out vs production's 54 %, cosmic-only: it relies on vertical muons) — put it in
  the reco, or keep it as a downstream selection? If downstream, it does not block the pass.
- **`PIVOT_START`**: built in (recommended), or kept as a setting.

### Not blocking (downstream of the pass, or limitations)

- Pairing sim: it fails the σ ln(qx/qy) gate (0.101 vs 0.055). That is simulation for the X17 pairs, not the bench pass.
- Paper I A1 fractions and `effbudget/budget.py` are re-derived from the pass tables. The Overleaf build of A1 is not yet confirmed.
- The §6 questions; timing v per v group (S4); det7 X quoted from ch 256–511.
- The noise-band mechanism and the det2/4/6 whole-plane bands are stated limitations (§7). The det6/det7 Y-card
  correlation and a pogo beam reader are optional.
- **Unreviewed risk:** `rerun/bands`, `multiparticle`, `pairing_sim` and `satstudy/satblob*` were run but never
  reviewed, and some overlay arms ran with `WFT_DEAD_SEED=0`. They are B1's spec, so read them before porting them.

### Order

1. Finish the det3 misses (Dylan, 10-27).
2. B4: the model audit, tonight in a separate session.
3. B1: write the unified reco, folding in the audit's findings.
4. B2: run the fleet gate. Do B3 (SPS wiring) while it runs.
5. Archive the code (`code_<hash>`). Re-finalize and promote if w0/kw moved. Then run §11: vrefit --keep-v, then pass,
   now with the SPS rows included.
