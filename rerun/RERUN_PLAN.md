# The from-scratch re-run of the cosmic-bench and SPS analysis (plan, 2026-10-23)

**Goal (Dylan, 2026-10-23):** one complete, consistent pass of every June cosmic-bench and SPS det4 analysis, on
lxplus condor, with the physical detector model, producing every number and figure of the two detector papers.
n_TOF is out of scope for these papers. The pass should run **once**: everything that would force it to run again
is settled first. This file is the plan; `CHECKS_2026-10-23.md` is the record of the checks run before it.

## 0 · Decisions already taken (2026-10-23)

| | decision |
|---|---|
| Kernel | **The physical RC kernel** (`wft.model.build_matrix_rc`): prompt footprint σ0 per view, RC spread along the resistive strips (D_rc, Y only), drift diffusion; no fitted copy amplitudes. Supersedes the 10-23 morning note "C2 = freeze r06". |
| X asymmetry | Not modelled; `wft.xasym` is a ± systematic (XASYM_CONTINGENCY.md). Toy cost: −0.05° overall, ∓0.19° head-on. |
| Inputs | Keep EOS `decoded_root`, `combined_hits_root` and M3 v2 rays where a fresh decode of a sample reproduces them. Re-decode what does not (see §2). |
| Where | lxplus condor only. Outputs to EOS (`/eos/experiment/ntof/data/x17/cosmic_bench/rerun_<tag>/`, SPS under `p2_sps_july/rerun_<tag>/` or user space); AFS holds code and logs only. |
| Scope | All June bench rows (golden long runs, HV scans, det3 drift scan), all SPS det4 runs, timing on waveforms. The hits-vs-waveforms comparison (Paper II R1–R5) is a side project at the **end**, on comparison datasets only. |
| Datasets | Golden runs set calibrations and quoted numbers; the other runs are consistency checks, read skeptically where a run is known to be bad (§5). |

## 1 · What forces a re-run, and what does not

Only **upstream** choices force the full pass to be repeated: the inputs (decode, hits for seeding, M3/uRWELL
reference), every calibration constant in a bundle, the reconstruction settings, and the alignment. Everything
**downstream** of the reconstructed tables (stacks, gas fits, D_T, efficiency definitions, timing, maps, figures)
can be iterated after the pass for the cost of an analysis script. So the pre-run work below is about the upstream
list only; the downstream questions (§6) stay open and are worked during and after the pass.

## 2 · Pre-run work packages (all must close before the code freeze)

| id | upstream item | status 10-23 | what closes it |
|---|---|---|---|
| W1 | **RC calibration protocol**: σ0_x, σ0_y, D_rc,y, v per chamber, fitted together on the golden calibration run; Dp from the head-on fit (Magboltz dry equivalent within errors, §12). **The reference position must not come from the hits alignment** (CHECKS C8: pinning p0 there moves det3's v from 37.0 to 39.6–42.1): calibrate with p0 profiled, reconstruct, align on waveform positions, rebuild the reference, iterate until v is stable | C7/C8 running | v agrees with Magboltz + gap within the gas uncertainty; v stable under the iteration; det6 and det7 (same night, same gas, same field) consistent; paired bench vs production not worse on any plane beyond the §12 numbers |
| W2 | **Depth grid and window**: K fixed at 18 × 60 ns = 1080 ns covers a 30 mm column only for v ≥ 28 µm/ns; the slow bench nights (det4 600 V, det6/det7 700 V) are at 197–238 V/cm, where 0.5–1 % water puts the column at 0.8–1.7 µs. det6's deep charge already runs past the window | measured (§3 of CHECKS) | K per dataset from the measured column (gap / fitted v + margin); depth bins whose pulse does not peak inside the window censored (the NNLS-runaway guard of qsum_runaway / OCTOBER O10), validated: early fits bit-identical, no χ² loss on late fits |
| W3 | **Channel layer**: per-channel gain, dead and hot masks per chamber (`strip_gain_audit.bad_set`, incl. the inclined-track holes found 10-19: det7 +20 X channels) | partly in bundles | every bundle carries the final masks; one table per chamber in the repo |
| W4 | **Electronics template**: one X-derived template drives both views (`rc_tmpl='x'`: Y's measured template carries the RC undershoot); the template's amplitude dependence reads saturated-event charge 5–12 % low (§41) | known | decision: keep the single template, censor saturated samples (geometry unaffected, §41), quote charge from unsaturated events only. An amplitude-dependent template would be a pre-run item only if a charge result of the papers needs saturated events |
| W5 | **t0 prior**: today adopted on det2/det4 only (8-12 gate) | mixed | one policy for all chambers under the RC kernel, decided by a gate on the golden keys |
| W6 | **Angle constants w0/kw**: measured post hoc against M3, applied in reco | applied | measured on the calibration run, applied to the others; under the RC kernel with v fitted they should come out ≈ 0 / 1, and a large kw is a calibration failure, not a constant to absorb |
| W7 | **Per-condition protocol**: HV scans (gain only: same constants), det3 drift scan (v and Dp per field; most fields' columns overrun the window), SPS (beam gas with O₂, det4's beam D_rc 2.2e-4 measured, attachment absorbed by the free charge profile) | to write | a table: for each row, which constants are inherited and which are fitted, and on which events |
| W8 | **Run-quality registry** (§5): golden / consistency / excluded per row, with the reason | to write | committed table read by every stage |
| W9 | **Inputs**: decoded spot check (139 subruns submitted, condor 4538802); RAW-mode nights must be re-decoded with the eventID-split decoder; hits spot check; M3 v2 coverage; SPS FEU 03 decode (never decoded on EOS) | running | every row has inputs that reproduce, or a re-decode job |
| W10 | **Software**: code-freeze commit; EOS lock venv (`venvs/ntof-lock-cvmfs`, Python 3.13, numpy 2.2.6) for every job; decoder pinned at `mm_strip_reconstruction` fa80c06; matched lists made on lxplus with that venv (the LCG_105 NClus bug is gone with it); every wrapper in the repo; the model-regression test made portable (its reference data path is hard-coded) | tests pass on lxplus (63) | a dry run of every stage on one row |

## 3 · Stages of the pass

| stage | what | where | gate before the next stage |
|---|---|---|---|
| S0 | Inputs: re-decode RAW nights + SPS FEU 03; make matched lists for every row; inventory | condor | every row has decoded, hits, reference; counts logged |
| S1 | Calibrate each chamber on its golden calibration run (W1–W6); write RC bundles | condor | gate table on the golden evaluation run (resolution, bias, slope, implied-v flatness, efficiency) |
| S2 | Reconstruct every row with its per-condition bundle (W7) | condor, one job per (row, file) | completion ledger; no row missing |
| S3 | Alignment → efficiency → angles → maps → digest, per row | condor or login node | digest reproduces S1's gate numbers on the golden rows |
| S4 | Detector characterization: v(E) and gas, gap, D_T, D_rc, σ0, attachment, gain and HV curves, sparks, charge balance, timing on waveforms (D3), edge tilt (D4), X asymmetry; SPS: kernel, spatial resolution on the waveform fit (centroid as cross-check), known-angle test at 25.6° (R7) | condor | each result reproduces its pre-run value or the difference is explained |
| S5 | Consistency checks on the non-golden runs (§5) | condor | a table per chamber |
| S6 | Paper figures and numbers from `papers/*/figures.toml` producers; reports | login node | every `\requote` replaced |
| S7 | Side project: hits vs waveforms (R1–R5) on the comparison datasets | condor | — |

## 4 · Checks run before the plan (details: CHECKS_2026-10-23.md)

Filled in as results arrive.

## 5 · Run-quality registry (draft)

| class | rows | why |
|---|---|---|
| golden, calibration | det3 sat_det3 · det2 o22_long_det2 · det4 g_det4 · det6 g_det6_long · det7 g_det7_long | the 8-12 calibration runs |
| golden, quoted | det3 g_det3_wknd (22 k rays) · det2 g_det2 (19 k) · det4/6/7 as above (held out by event) | largest clean runs |
| consistency | HV scans, short/longer runs, det3 drift scan | trend-grade rows |
| skeptical | det2 6-17 overnight (RAW decode, half-dead X), det3 6-23 and det2 6-17 (degraded bottom-slot nights), det3 6-22 Monday (unusable micro-TPC angle) | known problems |
| excluded | det1 (no calibration), pre-June, ArIso, stale duplicate trees, telescope-off rows 90/91 | 8-12 manifest reasons |

## 6 · Downstream questions (do not block the pass; worked during S4)

- det3's X diffusion excess (+160…+260 µm/√cm), and whether partial-drift-column pulses in the head-on selection make it.
- The water per chamber and night (the serial gas chain A→B→C→D could make the downstream chambers wetter).
- Three-body O₂ attachment scale (beam).
- The X asymmetry's mechanism (closed unexplained).
- det4's X under the RC kernel (+0.06…+0.16° against production).

## 7 · Stated limitations (no new data)

The detectors are gone. Anything not answerable from the existing data, simulation or the literature is written down
as a limitation: the asymmetry mechanism, the resistive-strip mask, the τ and c2 tails beyond the SPS window, a
dry-gas bench run, Stephan's unanswered geometry questions.
