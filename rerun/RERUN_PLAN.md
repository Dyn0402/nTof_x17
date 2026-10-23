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
| W1 | **RC calibration protocol**: σ0_x, σ0_y, D_rc,y and v per chamber, fitted together on the golden calibration run with the reference position profiled; Dp from the head-on fit; a bundle format for the RC kernel (templates, gains, masks, t0 table, w0/kw, K) | tooling exists (`recal.py`, v now free) | v stable across the iteration below and within the timing v's errors; det6/det7 explained; paired bench not worse than §12 on any plane |
| W12 | **Reference geometry**: rebuild `ref_mesh` (calibration caches, matched lists) from the waveform alignment (`mx_june_wft/01_alignment.py`) instead of `alignment_tpc_veto50`; iterate calibrate → reco → align → calibrate | found | v changes < 0.5 % between iterations |
| W2 | **Depth grid and solver**: K per dataset from the measured column (gap / v + margin, capped by the window); column guard on | guard written + tested | early fits bit-identical, runaway class gone, no χ² loss on long columns (det4, det6) |
| W11 | **Signal-masked common mode**: investigated (C10), gated on the fit (C15): **not adopted** — no resolution gain, χ² not better; plain median in the fit and the hits; masked−plain constant shifts (D_rc ≤ 19 %, v ≤ 6 %) carried as a systematic | **done (off)** | — |
| W3 | **Channel layer**: final gain, dead and hot masks per chamber (`strip_gain_audit.bad_set`, incl. 10-19's inclined-track holes) | partly in bundles | one committed table per chamber, carried by every bundle |
| W4 | **Template**: one X-derived electronics template for both views (`rc_tmpl='x'`); saturated samples censored | decided | charge results use unsaturated events (the template reads saturated charge 5–12 % low, §41) |
| W5 | **t0 prior**: one policy for all chambers under the RC kernel | gate to run | gate on the golden keys |
| W6 | **Angle constants w0/kw**: measured on the calibration run, applied to the others | in reco | ≈ 0 / 1 under W1; a large kw is a calibration failure, not a constant |
| W7 | **Per-condition protocol**: what each row inherits and what it fits — HV scans: gain only; det3 drift scan: v and Dp per field, K per field, most columns overrun the window; SPS: beam gas (O₂, water), det4's beam D_rc 2.2e-4, attachment absorbed by the free charge profile | to write | a committed table |
| W8 | **Run-quality registry** (§5) | draft | committed table read by every stage |
| W9 | **Inputs**: every June row re-decoded with fa80c06 into `decoded_root_fa80c06/` and audited against an independent decoder (C11: 1278/1278 identical; M3 rays consistent, no re-tracking); hits regenerated from them (**`*_rr1_cm0` adopted**, plain CM, C12/C13; `*_rr1_cm5` deleted 10-23); still to do: decode SPS FEU 03; matched lists on lxplus with the lock venv; incomplete events (audit npz) out of every efficiency denominator | decode + re-hit done | every row has inputs with one decoder provenance |
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
