# Detector 3 weekend deep-dive — micro-TPC angle correlation & drift velocity

*2026-07-01 analysis of the 6-27/6-28 weekend det3 runs; rev 7-04 (FINAL:
geometry estimator supersedes all time-based fits). Scripts:
`13_tpc_angle_bias.py` … `25_signal_formation_toy.py`; run key `sat_det3`.*

> **7-06 UPDATE — re-verified on M3 tracking v2.** The full chain was re-run
> on the v2-reprocessed reference rays (recipe chi2<5 & NClus≥3; alignment
> identical, match quality 85.6→95.6 % within 10 mm, σ_x/σ_y 0.83/0.92 →
> 0.76/0.83 mm). **Every physics conclusion below holds**: v_geom(1000 V)
> = 33.90 ± 0.25 µm/ns, recorded column 23.4 mm, T_sat 691 ns, λ_att
> 15.4 mm, gas ranking unchanged, sharing constants identical, unshared
> time-fit converges (34.2/32.9 x/y). Only the 700 V drift point moved
> (21.6 → 23.3 µm/ns, ~2σ). Comparison table: `MICROTPC_RUNBOOK.md` §0b.
> New in this pass: `31_microtpc_metrics.py` (micro-TPC-mode performance
> scoreboard) and `32_edge_fringe_field.py` (degrader fringe-field study)
> — results in §7–8 below.

> **Full LaTeX write-up: `report_det3_weekend/main.pdf`** (rev 7-04, 22 pp).
>
> **Headline (7-04): v_drift(1000 V) = 34 ± 1.5 µm/ns.** Every TIME-based
> track fit — the production anchored/amp-weighted fit AND four independent
> waveform-level estimators (threshold/CFD/rise-fit/derivative) — is biased
> LOW 10–20 % by **prompt charge sharing** between resistive strips: each
> strip's pulse start is contaminated by its shallower neighbour's shared
> charge, so the time–position ladder is S-shaped (flat at both cluster
> ends; seen directly in waveform displays). The **geometry estimator**
> (cluster-extent slope [mm/unit-tan] ÷ time-span plateau) has no time-fit
> in it and is validated unbiased (±3 %) by a signal-formation toy MC that
> reproduces the full data pattern (extent slope, T_sat, floors, bias
> hierarchy) with 35 %/12 % sharing.

## Datasets

`mx17_det3_saturday_scan_6-27-26` (det3 in TOP slot, FEU 7=X / 8=Y):

| subrun | what | stats |
|---|---|---|
| `long_run_resist_490V_drift_1000V` | operating point, **141 min** (stopped for p2 run) | 29k events, 47k M3 tracks; decoded_root waveforms pulled from lxplus AFS |
| `drift_scan_..._drift_<100..1100>V` | 6 pts × 15 min | ~5k M3 tracks/pt |
| `hv_scan(2)_resist_<425..525>V` | resist scans | ~2k rays/pt |

Alignment: z = 714 mm, θ = 89.45°, stable across the weekend. Headline:
eff ≈ 80 %, σ_x/y = 0.83/0.92 mm. Resist scans: plateau 455–490 V (best
83.2 % @470 V), spark-driven rolloff >505 V → recommend 470–480 V.

## 1. Off-diagonal angle correlation — mechanism (unchanged)

Cluster TIME span is fixed (≈z_rec/v for every track) while SPATIAL width
has a ~2.2 mm non-geometric floor (spreading; w = |intercept|·T_sat is
velocity-independent) → tanθ_det ≈ tanθ_ref ± w/z_rec, sign-following,
angle-independent → parallel outward ridge, |θ_det|≳5° exclusion, horizontal
arms. NOT rotation. Universal because design + gas are shared.
Resistive strips run along y (extent floors: y 4.7 mm vs x 3.1 mm).

## 2. Drift velocity — the estimator saga (13/14 → 20–25)

- **Ridge fit** (slope of S vs tanθ_ref): assumes angle-independent
  spreading. It is violated: the ridge is CONVEX (inner window 26, outer
  32 µm/ns) — w shrinks with angle, and slope trades 1:1 with w′.
- **Geometry estimator** (`21_geometry_vdrift_scan.py`,
  `23_core_geometry_vdrift.py`): extent-vs-tanθ slope (no times!) ÷ T_sat.
  Core-fraction stable (20/30/40 % → 35.0±0.6 @1000 V), both planes equal.
- **Waveforms** (`24_waveform_investigation.py`, decoded_root from lxplus:
  512 ch × 32 samples): hit times faithful to waveforms (4 estimators agree
  28–30.5) ⇒ ladder itself distorted; rise scale 140 ns all amplitudes;
  skirt late-tails +200–400 ns.
- **Toy MC closure** (`25_signal_formation_toy.py`): clusters + attachment
  + diffusion + prompt sharing (±1: 35 %, ±2: 12 %) + CR-RC²(140 ns) +
  60 ns sampling + thresholds. Reproduces extent slope 23.6 mm, T_sat
  695 ns, floor, AND the bias hierarchy. Geometry estimator recovers
  v_true ±3 % everywhere; time fits low 15–25 %. T_sat=690 ns reproduced
  only at v_true ≈ 34.

**v(E), E = HV/3 cm** (geometry; ridge kept to document bias):

| HV | E [V/cm] | v_geom [µm/ns] | v_ridge (biased) | T_sat [ns] |
|---:|---:|---:|---:|---:|
| 500 | 167 | 12.37 ± 0.39 | 12.77 | (1361, window) |
| 700 | 233 | 21.61 ± 0.53 | 19.45 | 992 |
| 900 | 300 | 30.03 ± 0.64 | 26.17 | 754 |
| **1000** | **333** | **33.91 ± 0.25** | 28.12 | 690 |
| 1100 | 367 | 35.13 ± 0.93 | 29.44 | 661 |

Model-independent bound: v(233 V/cm) ≤ 30 mm/992 ns = 30.2 µm/ns → clean gas
(41) excluded regardless of estimator.

## 3. Drift gap: 30 mm mechanical, ~23 mm recorded

Recorded column = extent slope = 21.5–23.5 mm, constant ≥700 V
(= v_geom·T_sat). Amplitude vs DEPTH collapses to one exponential across
fields, λ ≈ 16–18 mm (attachment; `amplitude_attachment.png` — vs TIME the
curves disagree ⇒ not electronics). Arrival-time tail endpoints straddle
30 mm but mix attenuated deep charge with late shared/RC signals —
consistency check, not a precision gap measurement. ~1/4 of track charge
lost in transit.

## 4. Gas: Ar/iso 95/5 + 1.0–1.1 % H2O + ~1 % air

RMS vs v_geom(E) (fine water grid run on **lxplus**,
`mm_water_grid_lxplus.py`; LCG_107 + `ROOT.gSystem.Load('libGarfield')`):

| mixture | RMS [µm/ns] |
|---|---:|
| **Ar/iso + 1 % H2O + 1 % N2** | **0.84** |
| Ar/iso + 1 % H2O | 1.12 |
| Ar/iso + 1 % H2O + 2 % air | 1.35 |
| 1.2 % / 0.8 % H2O | 2.8 / 4.7 |
| Ar/CO2 90/10 | 5.4 (also excluded: +50 V iso-gain, zero attachment) |
| clean / iso-rich / dry-air-only | 9–19 |

Attachment closes on O2: measured λ 16–18 mm ↔ Magboltz 1 % air
(0.21 % O2); water ENHANCES O2 attachment (1 % H2O + 0.42 % O2 → λ=4.9 mm)
so true O2 ≈ 0.15–0.25 %. One air/moisture ingress explains velocity AND
attachment; gain untouched (resist turn-on matches clean gas).
**No re-scan possible (campaign closed).** Identification rests on the
internal closure (v(E) shape RMS 0.84 over 5 fields + attachment + gain +
all alternatives excluded); out-of-sample check = other detectors' runs
(same gas line).

## 5. Tracking fix: waveform unsharing — PROTOTYPED AND WORKING (7-04)

Production fit (`_fit_single_axis`): amplitude-weighted, ANCHORED at the
earliest hit, on pulse-start `time` — worst of the tested variants.
Waveform-level re-timing alone does NOT fix it (bias is physical, in the
ladder). **Fix = unsharing** (`26_unsharing_analysis.py`): solve
(1 + c1·E±1 + c2·E±2)·w′ = w per time sample (banded), BEFORE timing.
- Sharing measured from vertical tracks (amplitude ratios, no velocity
  input): **c1 = 0.45 (x) / 0.52 (y), c2 = 0.05 / 0.15**, neighbour delay
  +69 ns. Half of each strip's charge sits on its neighbours! More ±2 reach
  in y = along-resistive-strip RC.
- Toy closure: unsharing restores v_cfd 27.4 → 33.1 at v_true=34.
- **Data: v_cfd_core before → after = 30.8 → 33.8 (x), 29.0 → 32.9 (y),
  agreeing with v_geom = 33.9.** Three independent methods now converge.
- Note: after unsharing, clusters shrink toward the direct footprint
  (min_strips must drop 4→3; extent-based observables change meaning).
**Kernel refinement + angle payoff (`27_unsharing_refinement.py`):**
mixed prompt/delayed kernel (α scan, honoring the +69 ns delay) — v robust
to ±0.5 µm/ns across α (α=0.5 most consistent: 33.5/33.5 x/y). Angles
(each frame at its own ridge v): **bias halved** (±4–4.6° → ±2–2.6°),
**exclusion gap** |θ_det|≳5–7° → ≳3–4°, transition-region σ_θ (3–8°)
2.5–4.4° → 2.1–2.8°; plateau σ_θ stays 1.5–2°. Residual ±2° offset =
incompletely-nulled diffusion floor (post-unshare neighbour ratio
0.2–0.34) — removed by **additive tan-space calibration**
(`28_angle_calibration.py`): b_x = +0.033, b_y = +0.029 (measured on the
plateau of the unshared angles). **FINAL: plateau |bias| = 0.19°,
σ_θ = 1.9° (68 %), usable to ~3–4°** — vs ±4–5° bias in the production
analysis. Per-event fits lose skirt strips (clusters → direct footprint).

**Notes for later (data frozen — no re-runs, NO dry-gas re-scan possible):**
- (2) mm_processor integration: one linear pre-filter per event/FEU in the
  decoded→hits step (banded solve, α=0.5 mixed kernel, causal delayed
  part); (c1, c2, α) = per-detector constants from any run's vertical
  tracks; downstream unchanged except min_strips 4→3.
- (3) Cluster template fit (strip×sample matrix, slope + per-strip charges,
  sharing kernel fixed): max-information estimator; prototype/closure-test
  on the toy (25). Expected gains moderate — plateau now diffusion/
  sampling-limited (~1.9°); mainly helps small angles.
- Validation path without new data: run geometry estimator + unsharing on
  det2/6/7 June runs (same gas line ⇒ same v(E), λ_att, sharing constants
  predicted; decoded_root on lxplus).

## 6. OUT-OF-SAMPLE VALIDATION on det2 — done (7-04, `29_det2_validation.py`)

det2 = mx17_2 (FEU 6/8, top slot), 6-22 overnight run, chain re-run from
scratch (11.9k matched events, decoded waveforms pulled from lxplus):

| observable | det3 (6-27) | det2 (6-22) |
|---|---|---|
| sharing c1 (x/y) | 0.45 / 0.52 | **0.43 / 0.52** ✓ |
| sharing c2 (x/y) | 0.05 / 0.15 | 0.06 / 0.20 ✓ |
| neighbour Δt | +69 ns | +74–85 ns ✓ |
| v_geom | 33.9 | 35.0 / 30.8 (x/y) |
| **v unshared** | 33.8 / 32.9 | **35.0 / 35.3** ✓ converges |
| λ_att | 16–18 mm | **≈40 mm** ✗ differs! |
| recorded column | ~23 mm | ~25–26 mm (deeper, consistent w/ λ) |

### 6b. FLEET & TIME SURVEY (7-04, `30_fleet_gas_survey.py`) — det3 DRIED OUT

Drift HV per run decoded from hv_monitor.csv (channel map calibrated on the
sat run: 0:7=top drift, 0:6=bottom drift, 3:4/3:3=resists, 0:8-11=M3@500).
Implied H2O from v_geom at the TRUE field via the Magboltz grid:

| date | det | drift | v_geom | H2O | λ [mm] |
|---|---|---:|---:|---|---:|
| 6-22 | det3 (test AND overnight, 2 runs!) | ~1000 | 8.7–9.2 | **>3 % (saturated!)** | 16–37 |
| 6-22 | det2 | 1000 | 32.9 | 1.05 % | 42 |
| 6-23 | det3 | 600 | 1.1 | broken pt (excluded) | — |
| 6-23/24 | det4 | 600 | 21.8–23.6 | 0.76–0.83 % | 25–29 |
| 6-25 | det3 | 500(ambig) | 18.8 | 0.75–1.2 % | 8.7 |
| 6-26 | det6/det7 | 700 | 19.1/29.2 | 1.19/0.73 % | 8.5/8.9 |
| 6-27+27n | det3 | 1000 | 32.8/34.3 | **1.05/0.97 %** | 23/22 |

**Findings:** (1) det3 was WATER-SATURATED (>3 %) on 6-22 — two independent
runs agree — and dried to the fleet band by 6-25, stable ~1 % on 6-27:
flushing worked, took days. The original "2–2.5 % water-saturated"
conclusion was accidentally right for EARLIER runs. (2) Fleet equilibrium
= 0.7–1.2 % H2O (system property). (3) O2 highest in newest installs
(det6/7 ≈2 % air-equiv, det3 ≈1 %, det2/4 ≈0.5 %) — air flushes out too.
(4) Same-day det3 pairs agree internally (method solid); det6 x/y split
40 % (user's distrust justified), det7 12 %. NB the June 12_scan
efficiency work ("det3 needs drift ≥700 V") happened while det3 was WET —
its low-drift behaviour partly reflects the saturated gas of that week.

**det2 vs det3 (6-22, same day, same run):**
sharing constants = design property (transfer ✓);
v(1000 V) ≈ 34–35 µm/ns confirmed on a second detector (H2O in the shared
gas ✓); but ATTACHMENT is detector-specific: det2's O2 is ~2.5× lower
(λ 40 vs 17 mm) — water sets v, O2 sets attachment, and the O2 budget
differs per detector volume / evolved over the 5 days between runs. Water
without O2 slows but doesn't capture — exactly det2's behaviour. Future gas
accounting needs per-detector λ from each detector's own
amplitude-vs-depth curve. Also NB det2's pre-correction bias differs from
det3's (before: 33.5/30.3 vs 30.8/29.0) — the time-fit bias is
condition-dependent (det2 ran resist 525 V), so the unsharing correction
(not a fixed scale factor) is the right production fix.

## 7. Micro-TPC-mode performance metrics (7-06, `31_microtpc_metrics.py`)

The hit-mode-analogous scoreboard for the DIRECTIONAL measurement, on the
final chain (unsharing α=0.5 + tan calibration, M3 v2 rays). Denominator =
good single v2 rays traversing the fiducial active area (5 mm margin),
spark-vetoed events removed. Numbers below = FULL statistics after the
file-003 recovery (16.3k rays, 22.1k segments); the pre-recovery pass
(13.4k rays) gave the same values within 0.1–0.2.

| metric | value |
|---|---|
| hit-mode efficiency (X+Y cluster, r<10 mm) | **92.9 ± 0.2 %** |
| micro-TPC segment efficiency (X+Y unshared fit, ≥3 strips) | 50.7 ± 0.4 % (54.6 % given hit) |
| direction-agreement efficiency (both planes \|Δθ\|<5°) | 37.8 ± 0.4 % |
| plateau (\|θ\|>8°) angular bias / σ68 | **−0.17° / 1.75°** |
| fraction of segments \|Δθ\| < 3° / 5° | 76 % / 84 % |
| Pearson r(tanθ_det, tanθ_ref) | 0.70 all / 0.89 plateau (\|θ\|>8°) |
| 3D opening angle ψ median / 68 % | **2.4° / 3.9°** |
| fraction ψ < 5° / 10° | 72 % / 81 % |

The 93 → 51 % step is the strict waveform-segment requirement (unshared
THR 150 ADC, ≥3 core strips with valid CFD): near-vertical tracks shrink to
2–3 direct strips after unsharing. The directional measurement is premium —
available for half the tracks, and when available it points to the ray
within 5° for ~84 % (per plane) / ψ<5° for 73 % (3D).

**DEAD-TAIL FINDING — RESOLVED (7-06 evening):** FEU 8 (y-plane) had no
*decoded* data for datrun file 003 (eid > 38,926). The original diagnosis
("file never written") was wrong: the raw `..._003_08.fdf` exists on EOS
with a byte-identical size to its FEU 1/7 siblings, and the DAQ log shows
47,452 events in all 3 FEUs — the *online decode* was interrupted after
003_07 when the run ended (last decode 01:15, run end 01:10). File 003 was
re-decoded and re-combined with both FEUs via `process_run.py` (p2 file-000
recipe) and synced back to EOS. Re-running `03 --refit` + `31 --rebuild` +
`32` on complete statistics reproduces the guarded numbers exactly:
hit-mode 92.9 ± 0.2 % over the full eid range (denominator 16.3k rays,
live-range guard now a no-op), segments 18,164 → 22,081, all angular and
edge-zone metrics unchanged, and the long-run drift point moves
v_geom(1000 V) 33.90 ± 0.25 → **34.30 ± 0.28 µm/ns** (15.2k quality
events; within the 34 ± 1.5 headline). Head-on tagger on full stats:
LDA AUC 0.918 (holdout 0.920), 20 % eff → 82 % purity — identical
conclusions. The per-FEU live-range guard in `31/32` stays
as a defensive measure. NOTE: after any such recovery the
`cache/event_results_veto50.pkl` per-event cache must be rebuilt
(`03 --refit`), or the recovered range contributes denominator rays but no
matches.

### 7b. Reading the direction metrics (`microtpc_direction_explainer.png`)

- **|Δθ|<5° per plane** compares each plane's 2D projected angle to the
  ray's projection; the "direction agrees" tier requires BOTH planes.
- **ψ** combines tanθx+tanθy into one 3D unit vector and takes the opening
  angle to the ray — a single "how well does it point" number.
- Both look low ONLY because cosmics peak near θ=0 where a drift TPC has no
  time-position lever arm: restricted to θ>10°, the ladder reads 94 % hit /
  60 % segment / 49 % agree, and agreement GIVEN a segment reaches 80–95 %.
  Median ψ falls from ~11° (θ<5°) to ~2° (θ>10°): at θ_ref = 2° even a
  perfect 2°-resolution device gives ψ ≈ 2–3° (relative error is O(1)).

## 8. Edge / fringe-field study (7-06, `32_edge_fringe_field.py`)

det3's drift gap is graded by a 3-step copper-ring degrader at the
perimeter; distortion is still expected near the edges. Method: whole-
detector (core-only) median response baseline(tanθ_ref) subtracted per
plane (removes the sharing bias, which is angle-sign-following), then the
residual δ profiled vs distance-to-edge with an OUTWARD sign convention;
positions = ray in raw detector frame. Zone systematics floor ±0.007 tan
(cosmic position–angle acceptance correlation).

| zone (dist to edge) | v_geom x/y [µm/ns] | T_sat x/y [ns] | outward δtanθ x/y |
|---|---|---|---|
| edge 0–25 mm | 22.1 / 28.7 | 745 / 834 | **−0.055** / −0.002 |
| mid 25–60 mm | 34.2 / 27.7 | 670 / 723 | +0.003 / +0.007 |
| core > 60 mm | 34.5 / 30.4 | 678 / 702 | +0.006 / +0.006 |

(δ shown for the unshared angle source; the production-fit source gives the
same picture — edge δx −0.056 — so the finding is estimator-independent.
The x-plane profile crosses zero at ~30–35 mm; the y-plane zone median is
diluted by its larger extent floor, but its per-side profile shows the same
−0.06 inward dip below 25 mm.)

Findings:
1. **Distortion is confined to ≲25–40 mm from the edge.** Beyond 40 mm the
   detector is uniform (angle maps flat, T_sat flat, v_geom stable).
2. **Inward apparent-angle tilt at the edge** (δ ≈ −0.06 tan ≈ −3° at
   <25 mm), same sign on BOTH sides of BOTH planes = radial signature —
   drift lines bow inward near the rings, exactly a fringe-field shape (a
   rotation or shear would be antisymmetric).
3. **Drift slows near the edge**: T_sat rises 678→745 ns (x) / 702→834 ns
   (y) while the recorded column proxy rises — consistent with tilted,
   longer drift paths in a weaker/distorted edge field.
4. **Efficiency turn-on over the first ~25 mm** (0→80 % over 0–25 mm from
   the edge, hit-mode and micro-TPC alike) — the edge band, not the core,
   holds much of the naive full-area inefficiency.
5. **Position residuals stay flat to the edge** — the cluster anchor
   (earliest strip, mesh end) is where the drift lines are least distorted,
   so hit-mode POSITION is robust even where angles distort.

Practical: fiducialise micro-TPC ANGLE analyses to >25 mm (ideally 40 mm)
from the edge; hit-mode positions need no extra fiducial.

## 9. Head-on track tagging (7-06, `33_headon_tracks.py`)

Truth = v2 ray θ<5° (prevalence 11.9 %). The production time-fit angle is
ANTI-correlated with head-on-ness (AUC 0.457): charge sharing pushes truly
vertical tracks to ±5–7° measured (the exclusion-gap mechanism), so a
"small measured angle" sample peaks at θ_ref = 10–15° (10 % purity). The
waveform signature carries the real information:

| classifier | AUC | coverage |
|---|---|---|
| production \|tanθ\| (current algo) | 0.457 | 100 % |
| unshared+calibrated \|tanθ\| | 0.512 | 55 % |
| neighbour time symmetry t_asym | 0.412 | 100 % |
| lead-charge fraction q_frac | 0.803 | 100 % |
| lead time-over-threshold tot_lead | 0.866 | 100 % |
| n_strips X+Y (cheap baseline) | 0.841 | 100 % |
| **signature Fisher LDA (6 features)** | **0.918** | 100 % (holdout identical) |

Physics: a head-on track drops its whole 23 mm column on 1–2 direct strips
→ long lead ToT + high lead-charge fraction + narrow footprint. The
LEFT/RIGHT TIME symmetry hypothesis does NOT discriminate: RC-fed
neighbour pulses are small (≈0.45×lead) and their CFD times are
noise-dominated at 60 ns sampling — amplitude/charge topology beats
timing. Working points (LDA): eff 80 % → purity 48 %, eff 50 % → 68 %,
eff 20 % → 81 % (7× enrichment). Use for selecting head-on samples;
conversely the tag can VETO tracks where the micro-TPC angle is
untrustworthy.

## 10. Hybrid tracking — angles at ALL angles (7-07, `34_hybrid_tracking.py`)

The head-on signature features are CONTINUOUS in |θ| (footprint, lead ToT,
lead-charge fraction = a per-event geometry estimator), so instead of a
binary tag we REGRESS |tanθ| from them (per plane; linear fit on
standardized features + monotonic binned-median calibration; trained on
even eids, everything below = odd-eid holdout). Sign at low angle from the
signed L/R asymmetries (shallow-side neighbour has more charge, fires
earlier): 92 % correct at 3°, ≥94 % beyond 5° (74 % at 1°, where sign is
irrelevant). Hybrid rule per plane: if the regressor says inclined
(|tan_reg| > 0.09 ≈ 5°) and a segment exists → unshared+calibrated time
fit; otherwise → signed regression. NB the switch must be decided by the
REGRESSOR — switching on the segment angle mis-assigns true low-angle
tracks whose time fit fluctuated high (1.75° → 2.34° in the <5° band).

| \|θ_ref\| < 5° band | coverage | bias | σ68 |
|---|---:|---:|---:|
| production time-fit (current algo) | 94 % | −3.4° | **14.9°** |
| unshared+cal time-fit (track-only) | 51 % | −0.2° | 5.3° |
| signature regression only | 97 % | −0.1° | 1.81° |
| **HYBRID** | **97 %** | **−0.1°** | **1.75°** |

Plateau (|θ|>8°): hybrid 1.86° at 98 % coverage (track-only 1.72° at
85 %). → **uniform ~1.8° angular resolution at ALL angles with 97–99 %
coverage**; the correlation plot is a continuous diagonal through zero
(exclusion gap gone). Surprise: regression ALONE gives 2.1° on the
plateau — the features carry nearly the full angular information; the
time fit only refines the plateau by ~0.3°.

### 10b. Overfitting check — det2 transfer validation (7-07)

Full chain run on det2 (`o22_long_det2`, v2 rays pulled, alignment refit:
θ=89.20°, σ 0.77/0.79 mm, 10.7k matched). det2 differs in gain (resist
525 V), attachment (λ≈40 vs 17 mm), FEUs (6/8) — if the det3 regression
had learned amplitude artifacts it would fail here. |θ|<5° band, odd-eid
holdout:

| hybrid variant | coverage | σ68 |
|---|---:|---:|
| det3, self-trained | 97 % | 1.75° |
| det2, self-trained | 96 % | 2.47° |
| **det2, FROZEN det3 model (zero det2 training)** | **96 %** | **2.85°** |
| det2 track-only (unshared time fit) | 57 % | 7.91° |
| det2 production time-fit | 94 % | 14.6° |

Verdict: **not overfit** — (a) within-run holdout identical; (b) the
learned weights transfer nearly unchanged (x: det3 [−.036,…,+.048] vs
det2 [−.033,…,+.041] — a design property, like the sharing constants);
(c) the frozen model costs only ~15 % resolution on a different
detector, attributable to real physics (det2's weaker attachment
changes the ToT/amplitude-vs-depth mapping). det2 is intrinsically a bit
worse at low angle even self-trained (2.47° vs 1.75°; its head-on
signature is softer: LDA AUC 0.881 vs 0.918, λ_att 40 mm → less
depth-contrast). Recommended production scheme: per-detector training
where reference tracks exist, det3 constants as the fleet default.

Other caveats: (1) below ~1.5° the response flattens (|θ|-folding +
feature floor). (2) Regression-only drifts at |θ|>15° (feature
saturation) — the hybrid switch handles it. (3) Features need decoded
waveforms; a production integration would compute them in mm_processor
alongside the unsharing pre-filter.

det2 side-results from this pass (31/33 on `o22_long_det2`): scoreboard
hit-mode 90.6 %, segment 60.9 % (67 % given hit — more strips survive
with λ=40 mm), plateau bias −0.36° / σ68 2.01°, ψ median 3.7°; head-on
LDA AUC 0.881 (holdout 0.883), production-angle AUC 0.51 (again ~random).

## Open follow-ups
1. Implement + validate the unsharing time estimator; wire geometry v and
   corrected micro-TPC into `03_alignment_and_tpc.py`.
2. Gas: long flush + re-scan (one hour decides); humidity/O2 probe; flow &
   leak check. Top hardware action.
3. Mechanics: confirm 30 mm drawing; confirm DREAM 60 ns sampling.
4. For more det3 stats use the p2 run (53k rays); Saturday run complete at
   141 min.
5. Longer readout window for future low-drift-field points.
