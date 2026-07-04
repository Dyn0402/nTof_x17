# Detector 3 weekend deep-dive — micro-TPC angle correlation & drift velocity

*2026-07-01 analysis of the 6-27/6-28 weekend det3 runs; rev 7-04 (FINAL:
geometry estimator supersedes all time-based fits). Scripts:
`13_tpc_angle_bias.py` … `25_signal_formation_toy.py`; run key `sat_det3`.*

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

## Open follow-ups
1. Implement + validate the unsharing time estimator; wire geometry v and
   corrected micro-TPC into `03_alignment_and_tpc.py`.
2. Gas: long flush + re-scan (one hour decides); humidity/O2 probe; flow &
   leak check. Top hardware action.
3. Mechanics: confirm 30 mm drawing; confirm DREAM 60 ns sampling.
4. For more det3 stats use the p2 run (53k rays); Saturday run complete at
   141 min.
5. Longer readout window for future low-drift-field points.
