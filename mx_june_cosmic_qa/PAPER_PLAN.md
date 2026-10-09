# MX17 papers — the two-paper plan (2026-10-08)

**Decision (2026-10-08):** the detector paper is split into two companion papers.

- **Paper I, detector & performance.** The MX17 resistive-strip Micromegas
  TPCs: design, operation, gas, timing, efficiency, and tracking performance
  on the June cosmic bench and det4 in the SPS H4 beam.
- **Paper II, reconstruction.** Why per-strip hit times fail on resistive
  strips, the waveform forward model that fixes it, and how much a hit-based
  readout loses. The comparison is made with hits extracted from the same DREAM
  waveforms, with and without a sharing correction. A full VMM emulation is
  left open for later; `ntof_cosmics/g4_digi` is where it would plug in.

The dependency runs one way: **I cites II** for every position, angle and
v_drift number, and for the sharing kernel. II needs only a short detector
description and cites I for the rest. Post both to arXiv on the same day,
submit them to the same journal as companions, and keep II at least level
with I.

**Scope:** the cosmic bench (det2, 3, 4, 6, 7) and SPS H4 (det4) in both.
The n_TOF campaign is a separate physics paper. The ongoing n_TOF reconstruction
work feeds II only through the tripwires below.

Slide-note version, with the evidence:
<https://dylan-neff.web.cern.ch/notes/mx17-detector-paper-status.html>, built by
`make_paper_plan_deck.py`, which **parses the tables in this file**. Edit the
tables here, then rebuild and republish:

    python mx_june_cosmic_qa/make_paper_plan_deck.py
    python ~/PycharmProjects/dylan-cern-site/scripts/add-note.py ~/x17/paper_status/mx17-detector-paper-status.html --slug mx17-detector-paper-status --force --deploy

Section and item states: `ready` (analysis done, write it), `requote`
(values exist, must be reproduced before quoting), `partial`, `open`,
`decision` (Dylan's call), `deferred`.

## Paper I — outline

| § | section | state | backed by |
|---|---|---|---|
| 1 | Chambers & resistive design, X/Y charge balance | ready | PLAN_38: f = 0.487 / 0.531 (det3/det2), σ68 0.07, flat in position and angle |
| 2 | Setups: cosmic bench (M3) and SPS H4 (uRWELL telescope) | ready | MICROTPC_RUNBOOK.md, sps_beam_test_26/analysis/README.md |
| 3 | Operation: HV, gain, sparks | ready | topic 6: optima 480/480/440 V, sparks non-propagating, no post-spark dead time (PLAN_39), spark waveforms |
| 4 | Efficiency, maps, edge turn-on | partial | efficiency stands (hits detection); the −3° edge tilt is a hits angle → D4 |
| 5 | Gas: v(E), attachment, gap topography | partial | 10-10 cloud_basics §16–25: compositions from waveforms, one physics model, no free loss rate. Bench det3 drift scan (6 fields): 0.95 % water, no air (≲ 10 ppm O2). Beam run_71: 1.51 % water + 163 ppm O2 (attachment, measured in X and Y); run_63 230→147 ppm overnight. Report: notes/mx17-beam-attachment. r06 (D2) still to tie in |
| 6 | Timing | open | PLAN_42's 33 ns is from hit times; waveform port (D3) |
| 7 | Tracking performance on cosmics (with II's reco) | requote | FLEET_DIGEST r06: σ_θ 1.15–2.51°, within-5 mm 93/92/42/75/57 %; needs C1 |
| 8 | Intrinsic resolution: det4 at SPS, and the bench decomposed | ready | spatial_resolution: 176 ± 10 µm = 0.30 × pitch; bench = M3 ⊕ scattering (arithmetic, D6 optional) |
| 9 | det4: the non-amplifying stripes | ready | det4_sps_assessment: 62 % of the area does not amplify |

## Paper II — outline

| § | section | state | backed by |
|---|---|---|---|
| 1 | Signal formation on resistive strips: the sharing kernel, measured head-on | ready | SPS sharing_kernel + angled_kernel: c2/c1 0.42–0.47, ±1 delay ~49 ns, gain-invariant 1–3 % (run_66); c2 < c1 is required. **10-09: the 0.45 is Y's; X ±2/±1 ≈ 0.15–0.20 on all five chambers vs Y 0.36–0.54 (model-free, plane_ratio)** |
| 2 | Why hit times fail: the estimator-independent ladder compression | partial | WAVEFORM_FIRST_THREADING §3 (det3): every estimator compresses 20–30 %; fleet extension R1 |
| 3 | The forward model and its calibration | ready | WAVEFORM_FIRST_THREADING §4, §9, §15; wft/ package; calibration protocol R9 |
| 4 | The physics floor: diffusion, not electronics | partial | toy closure §12: 0.02° electronics, ~1° from 0.3 mm centroid jitter; re-run on the production model R6 |
| 5 | Hits against waveforms, same events | open | the core new work: R1–R5; July det3 prototype and FLEET_DIGEST cover pieces |
| 6 | What a hit readout must record | open | threshold and ±2-neighbour study R4; VMM emulation deferred |
| 7 | Known-angle test in the beam | open | det4 at 25.6° in SPS (R7): a reference-free angle truth |
| 8 | Fleet performance, cost, outlook | partial | FLEET_DIGEST; cost table R10; outlook: VMM, n_TOF in-situ calibration, two-track |

## Work list

| id | short | paper | item | state | size | needs | notes |
|---|---|---|---|---|---|---|---|
| C1 | reproduce r06 | both | Reproduce the r06 golden numbers: pull lxplus ~/wft_campaign_r06, collect_results --promote, re-run mx_june_wft 01–04 on the five golden keys, check digest.py reproduces FLEET_DIGEST | ready | S | – | Done 10-09: every waveform-fit cell of FLEET_DIGEST reproduces exactly (efficiency, position, σ_θ, bias, implied-v, v) on all five keys. Needed the 07-24 reprocessed combined_hits from EOS for det2/det6/det7 (local June originals parked as combined_hits_root_june_orig) and 02_efficiency --max-dropped -1 for the headline. Not reproduced: the hits-chain comparison table on det2/6/7 (09-07 hits caches predate the significance floor) → rebuilt in R1. |
| C2 | freeze calibration | both | Freeze the calibration: r06 as it is, or build the per-plane c2/c1 ratio first; delay vs lp kernel form | decision | – | – | Studied 10-09 overnight (plane_ratio/FINDINGS.md; report /home/dylan/x17/cosmic_bench/plane_ratio/report.html). r06 hypers + c2_over_c1_y = 0.95 (X 0.6) improves Y in the golden-key gate: det3 −0.040 (head-on −0.101), det2 −0.063 (−0.118), det7 −0.082 (−0.133) deg, 5–8σ; X/efficiency/position unchanged; det4/det6 smaller and entangled with their stale w0/kw. Refitting the ratio fails (bench chi2 model-error dominated) — pinned. Re-pass staged (condor_campaign_pvy95, det2/3/7 only), NOT submitted. Open: submit?; c2<c1 gate on hyper vs observable (Y improves to 1.6); det4/det6 w0/kw refresh. |
| C3 | n_TOF tripwires | both | n_TOF tripwire check before submission | partial | S | – | See the tripwire table. None so far touches the model itself. |
| C4 | journal & sign-off | both | Journal, author list, collaboration sign-off, companion-submission letter | decision | – | – | NIM A or JINST, both accept companion submissions. |
| R1 | hit tiers H0/H1 | II | Hit tiers from the DREAM waveforms on r06, all five chambers: H0 production combined_hits, H1 best per-strip estimator (CFD, leading edge, matched filter, peak time) | partial | M | C1 | July: det3 only, R&D model (scripts 04/05). |
| R2 | hit corrections H2a/b | II | Hit-level corrections on r06: H2a slope remap (script 16) and H2b unsharing + angle-calibrated hybrid (scripts 26–34, 36) | partial | M | R1 | Both exist for det3 (July). The hybrid's constants were fitted on hits: refit on r06. |
| R3 | forward fit on hits H2c | II | H2c, new: the forward model fitted to hit lists (one time + amplitude per strip) instead of waveforms | open | L | R1 | The key new result: how much of the gap a sharing-aware fit recovers without waveforms. |
| R4 | threshold & neighbours | II | What a hit readout must record: hit threshold 3–7σ, ±1/±2 neighbour readout, amplitude on/off | open | M | R1 | Cheap once R1 exists; answers the VMM question in general terms. |
| R5 | benchmark, all tiers | II | The benchmark on r06, every tier, same events: σ_θ vs angle, bias, implied-v flatness, threading < 1 mm, position, efficiency, near-vertical | partial | M | R1 R2 R3 | FLEET_DIGEST has H0 vs W on four metrics; the July det3 table has five tiers on the R&D model. |
| R6 | physics-floor toy | II | Toy closure (physics floor) on the production wft model and r06 kernel | partial | S | C1 | July script 18 used the R&D model v2 (c2/c1 0.17). |
| R7 | SPS known angle | II | Known-angle test: det4 rotated 25.6° in SPS, hits vs waveforms against the stage angle | open | M | – | Reference-free; independent of M3. |
| R8 | kernel section | II | Kernel section from the SPS measurements | ready | S | – | τ and c2 are bounds (the tail outlasts the 3.84 µs window); state it, it cannot be fixed. |
| R9 | calibration protocol | II | Calibration protocol and its traps: template, kernel + v hyperfit, per detector and condition, check_kernel_ordering, never v from a prior | partial | S | – | In pieces across WAVEFORM_FIRST_THREADING §15, RETIRE_C2GTC1, insitu findings. |
| R10 | cost table | II | Cost table: time per plane, per tier | open | S | – | |
| D1 | re-quote cosmics | I | Re-quote cosmic performance from the reproduced r06 products | requote | S | C1 | Per-row bundle labels (det4 lp_t0p, det6 lp). |
| D2 | v(E) on r06 | I | Drift velocity v(E) and gas fit on r06: six det3 drift-scan rows, locally | partial | M | C1 | 10-10: the six det3 drift-scan rows are fitted from waveforms (cloud_basics/driftscan_fit.py, trigger-placed stacks, no M3): one composition, v 2.5–37.8 µm/ns over 35–382 V/cm. Still open: the same rows reconstructed with r06 for the paper's v(E) points. |
| D3 | timing on waveforms | I | Time resolution ported to waveforms (PLAN_42) | open | M | – | Hits gave σ_t 33 ns. |
| D4 | edge angle tilt | I | Edge / fringe-field angle tilt re-measured with wft angles | open | S | C1 | The −3° is a hits angle. |
| D5 | gas skeptic tests | I | Gas skeptic tests (PLAN_40): water sets v, O₂ sets λ; gap topography | partial | M | D2 | 10-10 cloud_basics §16–25: water from v, O2 from the loss (per time, not depth; no undershoot; X=Y; flat in rate/gain); gap spread and field profile fitted (bench det3 1.5 mm, k≈0.1). Open: three-body O2 attachment scale (ppm model-dependent). |
| D6 | bench resolution split | I | Bench resolution decomposed: M3 kink (2.6 mrad measured) or a scattering sim of the bench stack | open | M | – | Optional; turns the arithmetic into a result. |
| D7 | write ready sections | I | Write the ready sections: design, setups, HV / sparks, charge balance, SPS 176 µm, det4 stripes | ready | M | – | Start from the MPGD26 figure scripts. |
| V1 | VMM emulation | II | Full VMM emulation (shaper, peak/time, threshold, neighbour logic, dead time) in g4_digi | deferred | L | R4 | Left open for a later version or a follow-up. |

## n_TOF tripwires

What the ongoing n_TOF reconstruction work has found, and whether it changes Paper II.

| finding | source | effect on II |
|---|---|---|
| Beam seeder needed 3 strips, not 5, for near-normal tracks | insitu findings 10-07 | none: a beam setting; the bench already uses 3 |
| Bundles took v from the Magboltz prior, not the kernel fit | insitu findings 10-07 | none on the model; goes into R9 as a calibration trap |
| TAN_MAX and other cuts are in raw-tan units, so they move with v | repass readiness 10-08 | none on the model; R9 implementation note |
| Beam reads 20–33 % shallower than cosmics at the walls | g4_angle 10-07 | none: electron scattering in Geant4 with an ideal reco; not the reconstruction |
| Cosmic raw/true falls with angle on A and C with borrowed bench bundles | angle response 10-06 | **watch**: the in-situ A bundle is flat (y 1.00, x ±3 %), so on A it was calibration transfer. If C stays non-linear on its own bundle, it is a model-linearity question II must answer |
| Chamber C non-linear with heavy tails, not kernel-driven | insitu findings 10-07 | **watch**: same question for the lp-kernel chamber |
| Two-track joint fit | two-track limit | outlook only |
