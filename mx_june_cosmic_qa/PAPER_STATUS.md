# June cosmic det3 micro-TPC paper — readiness audit (2026-07-10)

*Full audit of the 10 planned paper topics against everything on disk (scripts 01–36,
`DET3_WEEKEND_ANALYSIS.md`, `MICROTPC_RUNBOOK.md`, `JUNE_RESULTS_SUMMARY.md`, the spark /
reco_far LaTeX notes, and the Analysis output tree). Execution plans for the missing
pieces live in `paper_plans/`. Data campaign is FROZEN — everything below is final on
existing statistics.*

**Headline verdict: 9 of 10 topics are analysis-complete with figures on disk** (X/Y charge
balance done 7-11 → PLAN_38; time resolution done 7-11 → PLAN_42 + LaTeX report).
One still needs new (cheap) analysis: the M3 pointing deconvolution (every quoted spatial σ
is detector⊕telescope). Time resolution is now fully measured — the earlier "not feasible"
call was wrong: the bench has a scintillator-trigger absolute reference and the DREAM fine
timestamp is applied, giving detector σ_t ≈ 33 ns (≈1 mm drift) and an absolute event-time
σ68 = 37.7 ns (detector-dominated). The
polished write-up `report_det3_weekend/main.pdf` (rev 4, 7-04) is two analysis
generations stale: it predates M3-v2, the file-003 recovery, and scripts 31–36 (which
hold most of the paper's best material). The markdown digests are the source of truth.

## Scoreboard

| # | Paper topic | Status | Headline result | Main gap → plan |
|---|---|---|---|---|
| 1 | Head-on charge spreading X vs Y | ✅ measured | c1/c2: x 0.45/0.05, y 0.52/0.15; extent floor y 4.7 vs x 3.1 mm | dedicated X-vs-Y figure → PLAN_41 |
| 2 | Unsharing correction | ✅ done + benchmarked | time-fit v converges to v_geom; final 0.19° bias / 1.9° | persist before/after table → PLAN_41 |
| 3 | Hybrid tracking | ✅ done + validated | ~1.8° at ALL angles, 97–99 % cov; det2 transfer OK | systematics + pub figures → PLAN_41 |
| 4 | X/Y charge balance (pixel layer) | ✅ measured (7-11) | f=q_X/(q_X+q_Y) med 0.487(det3)/0.531(det2), σ68 0.07, flat in pos+angle; 3 charge proxies agree | done → PLAN_38 results |
| 5 | Fringe field / edge | ✅ done (det3) | reach ≲25–40 mm; −3° tilt; eff 0→96 % over 0–25 mm; position robust, angle not | standalone turn-on fig → PLAN_41 |
| 6 | HV scans + sparks | ✅ done (richest) | optima 480/480/440 V; sparks Poisson, muon-induced 4–6×, edge-dominated; **no post-spark dead time** (PLAN_39) | ~~dead-time per spark~~ **DONE (PLAN_39)** |
| 7 | Drift gap / moisture | ⚠️ solid, closure-based | 30 mm mech vs ~23 mm recorded; λ≈15–18 mm; water sets v, O₂ sets λ | skeptic hardening → **PLAN_40** |
| 8 | Drift velocity scan | ✅ done | 4 clean points 700–1100 V + M3-free hybrid cross-check | 900 V x/y split unexplained |
| 9 | Spatial resolution | ⚠️ convolved only | sub-pitch: 0.61/0.72 mm @ θ<5° (early-charge centroid) vs 0.78 mm pitch | M3 deconvolution → **PLAN_37** |
| 10 | Time resolution | ✅ measured (7-11) | detector σ_t 33 ns (29 walk-corr, ≈1 mm drift); absolute t0 37.7 ns (ftst-corrected, detector-dominated); bias ≈0 | done → PLAN_42 + report |

## Per-topic status

### 1. Charge spreading per direction — do the resistive strips matter? YES, measured
Sharing constants from vertical tracks (26): **c1 = 0.449 (x) / 0.516 (y), c2 = 0.052 (x)
/ 0.151 (y)**, +69 ns neighbour delay — 3× larger ±2-strip reach along the resistive-strip
direction (strips run along y). Non-geometric extent floor **4.7 mm (y) vs 3.1 mm (x)**.
Confirmed on det2 (0.43/0.52) → design property. Head-on tagger bonus result (33): timing
symmetry carries NO head-on information (AUC 0.41); charge topology carries it all
(Fisher LDA AUC 0.92, 81 % purity @ 20 % eff).
Figures: `bias_study/unsharing_sharing.png`, `headon/headon_benchmark.png` (paths below).
Gap: no single figure isolating the X-vs-Y spreading contrast.

### 2. Unsharing — where needed, where not, justified by benchmarks
Needed for ANY time-based angle/velocity at all angles (bias is physical, in the ladder —
waveform re-timing alone does not fix it; 4 independent estimators shown). After
unsharing: v_cfd 30.8→33.8 (x) / 29.0→32.9 (y), converging with v_geom 33.9; toy closure
27.4→33.1 @ v_true=34; angle bias halved; after tan calibration (28):
**plateau bias 0.19°, σ 1.9°**. NOT needed (counterproductive) for low-angle position —
raw early-charge centroid beats unshared (sharing = free interpolator, script 36).
NOT a fixed scale factor (det2's pre-correction bias differs) — that justifies the
complication over simpler global corrections. Benchmarks vs simpler algorithms live in
the hybrid table (34): production 14.9° → unshared 5.3 (51 % cov) → hybrid 1.75° (97 %).
Gaps: before/after numbers are stdout-only (persist to CSV); offline-only (fine if framed
as offline correction).

### 3. Hybrid tracking — done, validated three ways
θ<5° band: production 14.9° σ68 → **hybrid 1.75° @ 97 % coverage**; plateau 1.86° @
98 % coverage → uniform ~1.8° everywhere, exclusion gap gone. Validated: odd-eid holdout; det2
transfer with FROZEN det3 model (2.85° vs 2.47° self-trained, weights near-identical →
not overfit); drift scan (35) reproduces v_geom(E) within +1–3 % at 700–1100 V from
hit-level features only → M3-free in-situ gas monitor claim for n_TOF.
Gaps: all errors statistical-only (no systematics statement); figures are 6-panel QA
dashboards; needs clean σ68-vs-θ standalone, method schematic, consolidated
4-estimator × 2-detector table.

### 4. X/Y charge balance via the pixelated top layer — MEASURED (7-11, PLAN_38)
`38_xy_charge_balance.py`. f = q_X/(q_X+q_Y) is narrow (σ68 ≈ 0.07) and essentially
position- and angle-independent: the pixel top layer routes the avalanche charge evenly
to both strip layers. Median f **0.487 (det3, 490 V) / 0.531 (det2, 525 V)** — close to
0.5 with a ~0.04 per-chamber offset (assembly effect, not a universal number; the
*constancy* is the design metric). r(q_X,q_Y) 0.88/0.85. f-map inner std 0.020 vs 0.009
expected from statistics → real position variation only ~2 %. Cross-checked three ways
(hits saturation-corrected amplitude, hits `integral` on the unsaturated subset, clustered
segment `amp_sum`) — medians agree to ≤0.008 (det3). Key subtlety: combined_hits
`amplitude` is saturation-CORRECTED, so saturated-peak events are KEPT (not dropped as the
plan assumed) and the `saturated` flag is used only as a systematic split (Δf(sat−clean)
−0.007 det3 / −0.037 det2). Figures: `charge_balance/xy_charge_balance.{png,csv}`.
**Standalone write-up: `report_charge_balance/main.pdf` (6 pp, 7-12)** — physics schematics
of the routing + metric definitions, then paper-ready figures (`38b_charge_balance_report_figs.py`).

### 5. Fringe field — measured, quantified, two-sided answer
Distortion confined to **≲25–40 mm from edge**; inward apparent-angle tilt δ≈−0.06 tan
(≈−3°), same sign both sides of both planes = radial fringe-field signature; drift slows
(T_sat 678→745 / 702→834 ns x/y); efficiency turn-on **0→96 % over 0–25 mm** (degrader
band holds most of the naive full-area inefficiency). Answer: we lose ~25 mm of
ANGLE-fiducial area, but **hit-mode position is robust to the edge** (residuals flat —
earliest-strip anchor sits at the mesh). Not correctable per-event; cleanly
fiducializable. Unifying paper point (implied across notes, never stated in one place):
the degrader edge is simultaneously the efficiency turn-on band, the angle-distortion
band, and the spark-initiation band.
Gaps: turn-on curve is a subpanel (needs standalone); no field simulation (optional);
det3-only.

### 6. HV scans + sparks — most complete topic
Scans: turn-on→plateau→falloff for det6/7, plateaus det2/3; optima 480/480/440 V
(det2/3/6 @ 480, det7 @ 440). `any_hit` flat ~100 % while efficiency rolls off = high-HV
loss is spark-induced reco failure, not silence; spark fraction 2.6 %→51 % (det3
450→525 V) exactly where efficiency dies. Spark characterization publication-grade
(det3 + det7 LaTeX notes): Poisson in time (0.33 / 1.61 Hz), muon-induced (4.1×/5.9×
enhancement, ~75–83 % muon-seeded; shower ruled out), edge-dominated, non-propagating;
DAQ cross-talk quantified (det3 spark → +54 % junk clusters in shared M3 FEU; det6↔det7
odds ratio 4.3). Crossing-based spark fractions: det3 4.4 %, det2 5.5 %, det6 24.0 %,
det7 31.9 %.
**Dead time — DONE (PLAN_39, `39_spark_deadtime.py`):** there is NO measurable
post-spark dead time. The DAQ next-event interval is unchanged after a spark (det3
192 vs 186 ms, det7 168 vs 167 ms; median eventId skip 1 either way), and both the
reconstruction efficiency and the pad gain are FLAT vs time-since-spark (efficiency
χ²/ndf ≈ 1; transient deficit at Δt≈64 ms +0.6 ± 1.3 % det3 / −5.2 ± 1.8 % det7,
95 % UL ≤3 pts; gain first-bin sag −3 %). So the only spark-induced loss is the
in-spark crossing coincidence itself (f_inspark 4.4 % det3 / 35.7 % det7), which
drops the operational efficiency to 88.8 % / 40.7 % and is already inside the
measured numbers. Discharges are localised, non-propagating, and impose no
irreducible recovery-time ceiling beyond the coincidence rate — the null IS the
result (the exp-recovery τ is unmeasurably small, not merely unfit). Figures:
`spark_deadtime.png` under each det's `alignment_tpc_veto50/spark_deadtime/`.

**Waveform anatomy — DONE (7-12, `40_spark_waveforms.py sat_det3`; see
`SPARK_WAVEFORM_FINDINGS.md`):** raw DREAM waveforms (decoded_root, 512 ch × 32
samp × 60 ns, CM intact) of 2878 sat_det3 sparks vs a normal-muon control. A spark
is a **fast GLOBAL common-mode step** — the whole FEU baseline rises together, on
31 %(X)/23 %(Y) of sparks to full saturation, then decays back within the 1.92 µs
window (94 % recovered). It is NOT a propagating streamer: the high-ADC onset is
flat across all 512 channels to ~1–2 samples (103 ns spread; a 40 cm front would
sweep over µs). The "50+ strips" is inflated by the common mode — genuine
(CNS) localised charge is ~half (raw 93 → genuine 40/56 strips) and sits at ONE
EDGE (edge-seeded), with only weak position–onset order (|corr| 0.21, spread ≈
drift window = drift not propagation). This is the resistive-Micromegas
spark-protection signature (self-quenching, non-propagating, edge-seeded) shown
directly in the raw data, and it is the waveform-level CAUSE of the no-dead-time
result. NB the headline g_det3_wknd spark run has no local decoded waveforms over
its spark region → ran on sat_det3 (same detector/gas/operating point). Figures:
`spark_waveforms_{gallery,analysis}.png` in `sat_det3/…/spark_waveforms/`.

### 7. Drift gap / moisture — the skeptical audit
Two separable claims the analysis keeps distinct: (a) recorded column ~23 mm < 30 mm
mechanical (v·T_sat, field-independent ≥700 V — a geometric scale); (b) cause =
attachment, not geometry.

**Strong evidence:** the money plot (19) — amplitude vs drift TIME disagrees across HV,
vs drift DISTANCE collapses to one exponential (λ≈15–18 mm); loss-per-length =
attachment, loss-per-time would be electronics. Model-independent bound v(233 V/cm) ≤
30 mm/992 ns = 30.2 µm/ns excludes clean gas regardless of estimator. Magboltz closes
twice: v(E) shape → ~1 % H₂O (RMS 0.84 over 5 fields); λ → ~1 % air (0.2 % O₂). Fleet
survey (30) shows det3 DRYING >3 %→1 % over the week — time-dependence geometry can't
produce.

**Where a referee pushes (in order):** (a) no gas-probe ground truth — composition
inferred from the data it explains; campaign closed; (b) absolute λ and the O₂ inference
inherit v_geom's absolute scale (the collapse itself only needs relative v(HV));
(c) gain/attachment degeneracy dismissed via "resist turn-on matches clean gas", not a
measurement — can diffusion+threshold alone fake the decay?; (d) det2 λ≈40 mm vs det3
17 mm — must be framed up front as per-detector O₂ budget or it reads as failed
replication. **Correct framing: water sets v (shared gas line, confirmed on det2), O₂
sets attachment (per-detector).** Cheap hardening on frozen data → **PLAN_40**
(diffusion-only toy bound, angle-binned decay, det2/det3 same-v-different-λ overlay).

### 8. Drift velocity scan — ready once topic 7 is framed
Four clean points (700–1100 V): 23.3 / 29.7 / 34.3 / 35.5 µm/ns — rising curve
flattening toward the Magboltz peak (the shape IS the gas-ID lever; not strictly linear).
Three estimators per point: v_geom (physics), v_ridge (documented-bias cautionary tale),
v_sig (hybrid regression, telescope-free, +1–3 %). Low-field limitation exactly as
suspected and quantified: 500 V T_sat≈1357 ns vs 1.92 µs window minus jitter (v_sig
+19 %); ≤300 V no lever arm at all; stated fix = longer readout window (future).
Loose ends: 900 V point has ~20 % x/y plane split (unexplained); script 23 writes CSV
but no figure.

### 9. Spatial resolution — sub-pitch achieved, but convolved
**780 µm pitch is NOT the floor**: early-charge centroid (first 2 samples, RAW — resistive
sharing acts as built-in interpolator) reaches **0.61/0.72 mm σ68 @ θ<5°** vs 0.73/0.94
production earliest-strip; COMBO keeps the gain at parity elsewhere, 99 % coverage.
Degradation with angle (~1.0 mm @ θ>15°, slant smearing) is generic micro-TPC-mode.
**Every quoted σ is detector⊕M3** — M3 pointing resolution at the detector plane never
computed, nothing deconvolved. Detector sits INSIDE the telescope span (interpolation,
not extrapolation) so the reference error is small but not negligible at 0.6 mm.
→ **PLAN_37** (highest-leverage remaining measurement — changes every headline σ).

### 10. Time resolution — MEASURED, absolute AND intrinsic (7-11, PLAN_42 + report)
`42_time_resolution.py`. **The original "not feasible" verdict was WRONG and is
corrected.** The bench HAS an absolute reference: acquisition is triggered by a top+bottom
scintillator-paddle coincidence (~5 ns), and the reconstruction APPLIES the 3-bit DREAM
fine timestamp `ftst` (10 ns steps) that re-references the free-running 60 ns sampling clock
to the trigger (`WaveformAnalyzer.cpp:390-392`; verified in data — hit `time` flat vs ftst,
slope −0.2 ns/step, vs −10.2 uncorrected). So per-strip `time` is an absolute leading-edge
time (30 % constant-fraction, sub-sample interpolated) vs the phase-corrected trigger.

Three measured numbers (det3 sat run, 28.4k dual-plane events; det2 replicates within ~10 %):
**single-strip σ_t = 38.9 ns (1.33 mm drift)**; **detector σ_t = 33.1 ns (1.13 mm)** from the
two independent orthogonal readout layers timing the same drift electrons (telescope- AND
geometry-free; walk-corrected floor **29.0 ns = 1.00 mm**); **absolute event-time σ68 = 37.7 ns**
(UL, ftst-corrected), whose budget = detector 33 ⊕ scint 5 ⊕ ftst-quant 2.9 ⊕ geometry 17 ns
→ **detector-dominated**. **Inter-plane bias −1.3 ns (≈0)** confirms the "one drift gap, two
orthogonal readouts of the same arrival times" picture. ≈1 mm drift precision matches the
transverse spatial resolution (topic 9) — not timing-limited relative to geometry. Per-strip
time-walk is negligible (+3 ns/rel-amp; the 30 % constant fraction already removes it), so
**unsharing is the dominant TPC-timing correction, not walk**; the residual inter-plane walk
(~−100 ns/asym) is a small S/N effect (33→29 ns). Figures + LaTeX write-up:
`alignment_tpc_veto50/time_resolution/{time_resolution.png,figs/}` and
`mx_june_cosmic_qa/report_time_resolution/main.pdf`.

## Key figure inventory (verified on disk)

Root: `~/x17/cosmic_bench/Analysis/mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3/alignment_tpc_veto50/`
(det2 twin: `~/x17/cosmic_bench/Analysis/mx17_det2_det3_overnight_6-22-26/longer_run/mx17_2/alignment_tpc_veto50/`)

- `hybrid/hybrid_tracking.png` + `hybrid_summary.csv` (+ det2 `hybrid_tracking_transfer.png`, `hybrid_summary_transfer.csv`, `hybrid_model.json`)
- `microtpc_metrics/microtpc_metrics.png`, `microtpc_direction_explainer.png`, `microtpc_metrics_summary.csv`, `microtpc_segments.csv`
- `headon/headon_benchmark.png`, `headon_features.csv`
- `position/position_benchmark.png`, `position_summary.csv`, `position_estimates.csv`
- `edge_fringe/edge_fringe_field.png` (9-panel), `edge_zone_table.csv`
- `bias_study/unsharing_{sharing,v,ladders,angles,resolution}.png`, `pulse_width_gap.png`, `amplitude_vs_drifttime.png`, `wf_shapes.png`, `wf_timing_ridge.png`, `strip_timing_displays.png`, `skirt_timing_and_estimators.png`
- `Analysis/mx17_det3_saturday_scan_6-27-26/drift_velocity/mx17_3/`: `geometry_vdrift_scan.{png,csv}`, `drift_velocity_scan.{png,csv}`, `drift_velocity_vs_magboltz.png`, `amplitude_attachment.png`, `attachment_vs_magboltz.png`, `gap_attachment_test.{png,csv}`, `hybrid_vdrift_scan.{png,csv}`, `core_geometry_vdrift.csv` (no png)
- HV: `~/x17/cosmic_bench/Analysis/june_hv_scans.pdf` + per-run `hv_scan/mx17_N/{efficiency_vs_hv,resolution_vs_hv}.{png,csv}`
- Sparks (in-repo): `det3_spark_analysis/`, `det7_spark_analysis/`, `det3_recofar_analysis/`, `det6_det7_crosstalk/` (each with figs + main.pdf + meta json)
- Spark dead time (PLAN_39): `alignment_tpc_veto50/spark_deadtime/spark_deadtime.{png,csv,json}` under det3 (`mx17_det3_p2_det1_overnight_6-27-26/long_run_p2_det1_sanity_check/mx17_3/`) and det7 (`mx17_det6_det7_overnight_6-26-26/long_run/mx17_7/`)
- Time resolution (PLAN_42): `alignment_tpc_veto50/time_resolution/time_resolution.{png,csv,json}` under det3 (`sat_det3`) and det2 (`o22_long_det2`)
- Overview: `~/x17/cosmic_bench/Analysis/june_detectors_overview.pdf`, `fleet_gas_survey.{png,csv}`

⚠ `*_prev2_backup/` twins are pre-M3-v2 — never use for the paper.
⚠ `angle_calibration.png` (28) currently exists ONLY in `_prev2_backup/bias_study/` — must be regenerated into the live tree (PLAN_41).

## Order of work

1. **PLAN_37** — M3 pointing budget + deconvolution (changes every quoted σ).
2. ~~**PLAN_38** — X/Y charge balance~~ ✅ DONE 7-11 (see PLAN_38 results section).
3. ~~**PLAN_39** — spark dead-time → efficiency ceiling~~ ✅ DONE 7-11: null result (no post-spark dead time); completes topic 6.
4. **PLAN_40** — drift-gap skeptic tests (hardens topic 7).
5. **PLAN_41** — publication figures + housekeeping (after 37–40 so their outputs are included).
6. Writing: rev the report / start the paper skeleton from `DET3_WEEKEND_ANALYSIS.md` + `MICROTPC_RUNBOOK.md` §0. Narrative arc: *sharing measured → breaks time-based tracking → unsharing + geometry estimator fix it → hybrid makes it uniform → sharing repaid as sub-pitch position*, plus the gas/attachment leg and the operations leg (HV, sparks, edge).
