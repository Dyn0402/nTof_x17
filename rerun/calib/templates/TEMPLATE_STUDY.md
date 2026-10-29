# Pulse templates for the from-scratch pass: investigation (started 2026-10-28, laptop)

Part of the calibration rebuild (`rerun/PASS_PLAN_2026-10-28.md` §2, E1 / CAL1 "TMPL"). The question is whether the
template that `wft.calibrate.measure_templates` produces is the right object for the forward model, and what the pass
should use. Scripts, results and figures are in this directory. Run them with `../../../.venv/bin/python -I <script>`.

## Morning summary (2026-10-29) — read this first

**Answer for the pass:** the calibration basis holds. Rebuild it as planned with **one change to the DRC step** and
no change to the templates or σ0. Nothing found overnight moves angle resolution by more than ~0.06° (1–2σ), but one
planned step (the D_rc,y fit on the raw objective) would have cost 0.07° in Y.

| piece | finding | for E1/E3 |
|---|---|---|
| templates | carry the strip's drift segment, 4–7 % too wide (T1). Segment-free h_eff: X −0.05°, Y +0.05° (T5/T6). Y on its own template: +0.10° worse (T14) | keep `measure_templates`, X template for both views |
| template zero / t0 | χ² wants δ = −9…−76 ns per chamber and view (T7/T8); resolution-neutral, shifts the angle scale | don't adopt δ; keep t0 = C − 10·ftst; ladder shift goes into the angle-scale systematic |
| D_rc,y | raw objective steered by saturated planes (88 % of χ²) → 4.3–5.4e-4 on det3, −0.07° Y (T6). Pinned + unsaturated → 3.5–3.8e-4 = held-out optimum | **DRC: `recal.py --t0-pin --unsat-only`**, disjoint training sample |
| σ0 | SPS 0.30 independent of the template (T10: 0.278/0.282 X, 0.323/0.324 Y) and of every other issue; free bench σ0 runs to 0.058 mm (not a measurement) | keep 0.30 as a physical input (Dylan: same build, same σ0) |
| fleet | recalibrated D_rc,y + δ vs plan: resolution-neutral on all five chambers (T8) | — |
| model shape | centre strip under-filled 5–11 % on every chamber (det6 Y 24 %), ±1 over-filled (T9 fleet). Any Gaussian width change trades centre/±1 vs ±2; better-shaped models improve χ² but not angles (T13) | stated limitation; the χ² flag is relative (5 × median). Y RC branch cannot carry footprint tails (code) |

**Open questions for Dylan:**
1. δ: adopt or not? (Recommended: not. It is resolution-neutral and unexplained, and kw covers the angle scale.)
2. Model shape: worth a footprint/RC-form change before the pass (e.g. Voigt tails that survive the Y RC growth, a
   two-component RC), or carry it as a limitation? Overnight evidence: the angles don't care; χ² and charge do.
3. det6 Y: the plan's D_rc,y (5.9e-4) was furthest off (corrected 4.9e-4), and its residual mismatch is the largest.
   It deserves a look at its RC model before its numbers go into papers.

**Caveats:** 1500-event rr1 caches (400 held-out events per comparison, s68 errors ±0.02–0.07°). Fitter: LM from T6
on (T5 ran NM, internally consistent). D_rc,y fits used 500–920 training events; the pass's CAL nodes see ~8000.

## The calibration pieces the pass rebuilds (for reference)

| # | piece | where it comes from | status in this study |
|---|---|---|---|
| 1 | reference geometry: calibration cache along the M3 corridor, then waveform alignment, iterated | CAL0–CAL2 | — |
| 2 | **templates** | `measure_templates` (TMPL0/TMPL1) | **this note** |
| 3 | trigger latency C → pinned t0 (σ 0.05) | `measure_latency` (model-free) | its zero convention couples to #2 (T5) |
| 4 | RC kernel D_rc,y (det4 2.2e-4 from H4) | `recal.py --free rc_D_y` | Y template contamination is why it exists |
| 5 | physics inputs: σ0 0.30 mm, Dp, gaps, timing v | `physics_inputs.json` (E1) | stated, re-checked in S4 |
| 6 | channel masks (+ the 14 GA channels, F7) | census + shorts | — |
| 7 | angle response w0/kw, dt_xy, pairing width, tan_floor | `set_w0kw --table` | — |
| 8 | bow block (all five chambers) | `set_bow` (E2) | — |
| 9 | per-row v (vrefit), K from the registry | vrefit `--keep-v` | — |
| 10 | SPS det4 bundle (no bench bow/v_map/w0/t0; ch 372) | E4 | — |

## What the production template is

The median, t50-aligned, unit-peak pulse of the two brightest strips of tracks with M3 |tan| > 0.22 (amplitude
500–3550 ADC, peak away from the window edges). The X template drives both views (`rc_tmpl='x'`), because the Y
template carries RC undershoot (cloud_basics §8). In the bundles on disk the template is a hand-down from early bundles
(r06 → lp2 → …). The pass re-measures it in TMPL0/TMPL1.

## T1 · The template depends on the track angle: it carries the strip's own drift segment

`t1_tan_amp.py`, `t1_extrap.py`. Inputs: the rr1 calibration caches `model_audit/ws_t0/caches/<det>.pkl` (1500
events per chamber, waveform alignment). Prediction registered before running: if the template is the electronics
response alone, all |tan| bins agree. If it carries the segment, the width falls with |tan|.

- The FWHM falls monotonically with |tan| on all five chambers and both views, with no plateau at the production
  cut. det3 X: 489 / 379 / 328 / 312 / 300 ns at |tan| 0–.08 / .08–.15 / .15–.22 / .22–.30 / .30–.40 (bootstrap
  ±2–11 ns). The production template (≥ 0.22) sits in that slope.
- Amplitude bins (500 → 3500 ADC, inclined tracks) agree within errors on every chamber. **The template has no
  measurable amplitude dependence**, so an amplitude-dependent template (the "next" of cloud_basics §41) is not
  supported by this test.
- Extrapolating with FWHM² = A + B/tan² to tan → ∞ gives 290–309 ns (X) and 279–296 ns (Y). The production template
  is **11–21 ns (4–7 %) wider**. The equivalent lateral segment the brightest strip integrates is 1.0–1.5 mm (pitch
  0.78 mm plus footprint and sharing). det7 Y has too few bins for this and is not quoted.
- The forward model already builds that segment explicitly (depth voxels × σ0 footprint × sharing). **The production
  template therefore counts it twice.**

## T2/T3/T4 · Against the simulation (MX17_Geant v4c)

- **T2 (`t2_shaper_model.py`)**: the DREAM nominal filter alone (PZC + ζ 0.75 SK, 166–180 ns) has FWHM 270–300 ns.
  With the *analytic* 340 ns ion rectangle at f_ion 0.908 it is 385–410 ns, far wider than any data template. I first
  read this as "the data leave no room for ions". **Withdrawn by T3**: production v4c uses the *reconstructed* ion
  current (ions.reconstructed_longitudinal, f_ion 0.900 but 50/90/99 % of the ion charge by 107/191/222 ns), not the
  rectangle.
- **T3 (`t3_sim_template.py`)**: the same template algorithm on the v4c sim waveforms at known angles gives almost the
  data's template. In X at tan 0.18 / 0.27 / 0.36 / 0.58 the sim has FWHM 334 / 309 / 300 / 294 ns, rise 210 / 190 /
  180 / 180, peak 160 / 150 / 150 / 140. Data det3 X at tan .15–.22 / .22–.30 / .30–.40 has FWHM 328 / 312 / 300,
  rise 190 / 170 / 160, peak 150 / 140 / 140. They differ in two places: the sim X undershoot is −0.10 against −0.02 in
  data, and the vertical X width is 586 against 489 ns (the known rise mismatch, FRESH_LOOK §1).
- **T4 (`t4_closure.py`), closure on the sim.** The sim's true electronics + ion response (DREAM 166 ns, β 0.75, v4c
  ions) has FWHM 315 ns. Yet the sim's measured template at tan 0.58 is *narrower* (294 ns) with a deeper undershoot.
  That is impossible for h ⊗ box unless the strip's own response includes more than electronics + ions: in the sim,
  the resistive layer takes charge away from the struck strip (a high-pass). So **the template measures the strip
  self-response (electronics + ions + resistive self-term)**, not the electronics. That is the right object for wft's X
  view, which has no RC term (rc_D_x = 0). It is not something to compare with the shaper datasheet.
- **The physical parameters are not identifiable from templates.** A joint fit of h = DREAM(t_peak, β) ⊗ [(1−f)δ +
  f·ion(stretch)] ⊗ box(L/(v tan)) over angle bins reproduces the sim's templates at rms 0.016, but with t_peak 134 ns
  and β 1.0 against a truth of 166 / 0.75. Only the segment-free *shape* it returns is meaningful.
- Data, segment removed (same fit): h_eff FWHM 292–302 ns in X (det2 295, det3 292, det4 302, det6 315, det7 302),
  rise 123–150 ns. The joint-fit rms is 0.007–0.015 of the peak. L = 0.9–1.6 mm, consistent with T1b.

## T5 · Does a segment-free template reconstruct better? (det3 held-out, in progress)

`t5_make_arms.py` builds the fe_nobow det3 bundle with tmpl_x = the current template or h_eff, each shifted by
−40…+40 ns. A shift is equivalent to moving the pinned t0, because t0 is pinned from the model-free latency while the
template's zero is its t50, so the two conventions only meet approximately. `heldout_arm.py` runs 400 held-out events
with `--t0-fix`, WFT_COL_GUARD 0.01 and WFT_RC_T0_INTERP 1. `t5_compare.py` makes the event-paired comparison.

Zero shift, h_eff vs current: X s68 −0.053 ± 0.029°, Y +0.040 ± 0.023°, far % and χ²/dof unchanged (48.8 / 56.7),
ladder 35.6 vs 36.1 (timing v 35.8), fitted charge +4 % (unit-peak convention). No net gain at zero shift.

Full scan (`results/t5_det3.json`, per-arm summaries in `results/t5_heldout/`). Each cell is s68 [°] (Δ vs cur_d+0
± paired bootstrap) and χ²/dof:

| shift | X cur | X h_eff | Y cur | Y h_eff |
|---|---|---|---|---|
| −40 | 1.097 (−0.006 ± 0.033) · 48.7 | 1.040 (−0.064 ± 0.039) · 47.7 | 0.921 (+0.027 ± 0.027) · 54.9 | 0.952 (+0.058 ± 0.029) · 54.1 |
| −20 | 1.097 (−0.007 ± 0.028) · 48.3 | 1.040 (−0.064 ± 0.037) · 48.0 | 0.886 (−0.008 ± 0.025) · 53.5 | 0.936 (+0.042 ± 0.024) · 52.5 |
| 0 | 1.104 · 48.9 | 1.051 (−0.053 ± 0.031) · 48.8 | 0.894 · 56.6 | 0.934 (+0.040 ± 0.024) · 56.7 |
| +20 | 1.113 (+0.010 ± 0.037) · 51.5 | 1.051 (−0.053 ± 0.035) · 51.8 | 0.906 (+0.013 ± 0.025) · 60.5 | 0.953 (+0.059 ± 0.026) · 62.4 |
| +40 | 1.110 (+0.006 ± 0.049) · 57.3 | 1.064 (−0.040 ± 0.048) · 58.5 | 0.942 (+0.049 ± 0.030) · 69.4 | 0.969 (+0.076 ± 0.032) · 70.7 |

- **Template zero vs latency.** χ² is lowest at −20…−40 ns for both templates and rises steeply for positive shifts.
  s68 is flat within errors over −40…+20 ns. The ladder moves ~0.7–0.8 µm/ns per 20 ns: from 35.0 to 37.3 for cur
  over −40…+40. kw absorbs this, but it is an angle-scale coupling. **The pass should set the template zero explicitly
  (the χ² minimum against the pinned latency) rather than inherit "t50 = 0".** That is one more convergence quantity
  for CAL1/CAL2.
- **Segment-free template.** X is better by 0.04–0.06° at every shift (each ~1.5–2σ, all consistent). Y is worse by
  0.04–0.08°. χ² and far % do not move. The Y arm is not self-consistent: Y is modelled as the X template ⊗ the RC
  kernel, and D_rc,y (3.44e-4) was fitted with the old, wider template, whose extra width partly stood in for RC
  spreading. The fair Y test is h_eff with D_rc,y refitted (DRC step), below.
- **χ²/dof ≈ 50 is not the template.** A 4–7 % width change and ±20 ns shifts move it by only a few percent. The
  dominant model error is elsewhere, and the template is not the lever for it.

**D_rc,y refit (the PASS_PLAN DRC step as specified).** `recal.py --free rc_D_y --objective cold --p0-profile
--p0-window 2`, σ0/Dp/v fixed, 180 training events (held-out events start at 180). Outputs:
`results/t5_drc_{cur,heff}.json`.

| | D_rc,y | training χ² | held-out X s68 (Δ vs cur_d+0) · χ²/dof | held-out Y s68 (Δ) · χ²/dof |
|---|---|---|---|---|
| fe bundle as is | 3.44e-4 | 1.0330e8 | 1.104 · 48.9 | 0.894 · 56.6 |
| cur, refitted | 4.91e-4 | 1.0224e8 (−1.0 %) | 1.109 (+0.006 ± 0.016) · 49.1 | **0.964 (+0.070 ± 0.028) · 65.2** |
| h_eff, refitted | 4.78e-4 | 1.0327e8 | 1.054 (−0.050 ± 0.029) · 48.8 | 0.988 (+0.095 ± 0.035) · 62.5 |

- **The refit lowers the training χ² and raises the held-out χ² (Y 56.6 → 65.2), costing 0.07° in Y.** The two
  objectives treat t0 differently. `recal.py`'s cold objective profiles t0 freely on a 20 ns grid, while the pass
  reconstructs with t0 pinned to the latency (σ 0.05). D_rc,y and the mesh time trade against each other (a slower
  Y spread vs a later start), so a D_rc,y fitted with t0 free is wrong for a pinned-t0 reconstruction.
  **Consequence for E3: every CAL fit (DRC, and anything else fitted by recal.py) must use the reconstruction's own
  t0 treatment (pinned to the latency), or the calibration and the pass optimise different models.** Not yet
  checked: whether the 10-26 phys/fe bundles' D_rc,y (3.44e-4) was itself fitted with t0 free; the held-out
  numbers say it is closer to right for pinned t0 than the refit.
- h_eff vs the current template, both on refitted D_rc,y: X −0.055°, Y +0.024°. The same pattern as without the
  refit. The template comparison cannot be closed until the DRC fit has a pinned-t0 objective.

> Fitter note: every T5 arm ran with the NM fitter (the LM merge reached this checkout afterwards). They are
> consistent with each other. From T6 on, every arm runs on main's LM fitter. LM reproduces the NM baseline
> (det3 X s68 1.109 vs 1.104, χ²/dof 49.1 vs 48.9).

## T6 · Why the D_rc,y refit went wrong: saturated planes steer the raw-χ² objective (2026-10-29)

`recal.py` has two new options. `--t0-pin` takes t0 per plane from the bundle's `t0_abs[ftst]`, as the pass does.
`--unsat-only` builds the objective from unsaturated planes only. `--train-start` picks the training sample.
Results are in `results/t6/`.

**1. Pinning t0 alone does not fix it.** With t0 pinned, the refit wants an even larger D_rc,y: 5.43e-4 (cur) and
5.23e-4 (h_eff) on events 0–179. On the held-out events (180–579) the same objective wants 4.30e-4, so the raw
objective also depends on the sample.

**2. What the reconstruction itself prefers** (held-out scan, LM, pinned t0, det3 Y; s68 Δ vs cur at 3.44e-4):

| D_rc,y | cur: s68 (Δ) · χ²/dof · ladder | h_eff: s68 (Δ) · χ²/dof |
|---|---|---|
| 1.50e-4 | 1.008 (+0.119 ± 0.044) · 79.9 · 36.96 | — |
| 2.50e-4 | 0.903 (+0.014 ± 0.029) · 60.7 · 36.50 | 0.903 (+0.014 ± 0.025) · 59.4 |
| **3.44e-4** | **0.889 · 56.7 · 36.41** | 0.939 (+0.050 ± 0.024) · 56.7 |
| 4.30e-4 | 0.930 (+0.041 ± 0.021) · 59.2 · 35.90 | 0.974 (+0.085 ± 0.030) · 58.8 |
| 5.40e-4 | 0.976 (+0.087 ± 0.033) · 69.2 · 35.74 | 1.022 (+0.133 ± 0.042) · 69.2 |

X does not depend on D_rc,y (identical rows), as expected with rc_D_x = 0. The reconstruction's optimum is 2.5–3.4e-4.
The ladder moves 1.2 µm/ns (3 %) over the scan, so D_rc,y is also an angle-scale lever in Y.

**3. The mechanism.** Split the held-out Y χ² by event class. **Saturated planes carry 88 % of the χ² sum** (64 of
72 M). The unsaturated sum has a clear minimum at 3.44e-4 (8.74 / 8.28 / 8.58 M at 2.5 / 3.44 / 4.3); the saturated
sum is flat or leans to 4.3. A raw-χ² calibration objective is therefore steered by saturated pulses and drags
D_rc,y up.

**4. The fix.** Pinned t0 + unsaturated planes only: D_rc,y = **3.57e-4** (events 0–179) and **3.82e-4**
(events 580–1499). That is stable across samples and sits at the reconstruction's optimum. h_eff: 3.60e-4.

**Consequences for the pass (E3, DRC node):** run `recal.py --free rc_D_y --t0-pin --unsat-only` on a training sample
disjoint from any evaluation sample. The same rule applies to any other χ²-sum calibration fit.
**Template, at its own calibrated D_rc,y (~3.6e-4):** h_eff vs cur is X −0.056 ± 0.03°, Y +0.05 ± 0.025°: no net
gain. The h_eff decision moves to the fleet comparison (T8) and is not urgent: the effect is ≤ 0.06° either way.

## T7 · The template zero vs the latency: δ ≈ −40 ns on det3 (2026-10-29)

The pinned, unsaturated-only objective (T6) over events 580–1499, at D_rc,y 3.7e-4, as a function of a template
shift (`results/t7/z_<tmpl>_<shift>.json`, χ² in units of 10⁶):

| shift [ns] | −80 | −60 | −40 | −20 | 0 | +20 | +40 | parabola minimum |
|---|---|---|---|---|---|---|---|---|
| cur | 58.64 | 57.29 | 56.66 | 57.09 | 58.65 | 62.86 | 69.12 | **−39.6 ns** (−3.4 % vs 0) |
| h_eff | 58.77 | 57.55 | 56.72 | 57.06 | 58.52 | 62.62 | 68.90 | **−37.9 ns** (−3.0 %) |

- Both templates want their t50 placed ~40 ns *before* the pinned t0. A negative shift is equivalent to a pinned t0
  later by 40 ns relative to the template. The latency C (`measure_latency`) is the 30 % crossing of the earliest
  strip on near-vertical tracks. That strip integrates the whole column, so C comes after the mesh charge first
  arrives. t0 = C − 10·ftst therefore leaves out an offset δ that depends on how the pulse builds up.
  (I first guessed "30 % vs 50 % crossing". It has the wrong sign and is withdrawn.)
- Held-out reconstruction (T5, NM) agrees on direction but is less decisive. s68 is flat over −40…0. The ladder
  reads 36.1/36.3 at 0, 35.4/35.9 at −20 and 35.0/35.5 at −40 against a timing v of 35.8, so the angle scale is
  nearest the timing v around −20. kw absorbs the rest.
- **For the pass:** keep the template at t50 = 0 and calibrate one offset δ per chamber and view in t0_abs (the
  minimum of the pinned, unsaturated objective over a global t0 shift). Record δ in the bundle and its convergence
  (±5 ns) in CAL1/CAL2. det3: δ = −40 ± 10 ns (the ±10 covers the held-out ladder's −20).

## T9 · The "bright under-fill" is a constant fractional shape mismatch, lateral + temporal (2026-10-29)

`t9_peak_residual.py`: held-out unsaturated planes (det3, cache events ≥ 180), fitted from the reference start with t0
pinned (main's LM). The residual (data − model) / model peak is taken on the brightest strip, its ±1 and ±2
neighbours, at sample offsets k from the peak, and stacked by amplitude. Medians are in `results/t9/`.

| σ0 [mm] | view | centre k = −1 / 0 / +1 / +2 | ±1 at k=0 | ±2 at k=0 | χ²/dof (800–1500 / 2500–3549 ADC) |
|---|---|---|---|---|---|
| 0.30 | X | +0.04 / +0.05 / +0.02 / −0.00 (k=+3 −0.02) | −0.03 | +0.01 | 14.7 / 57.4 |
| 0.30 | Y | +0.05 / +0.06 / +0.04 / +0.03 | −0.06 | +0.01 | 14.9 / 61.7 |
| 0.24 | X | +0.035 / +0.04 / +0.015 / −0.01 | −0.02 | +0.02 | 14.9 / 55.5 |
| 0.24 | Y | +0.04 / +0.045 / +0.03 / +0.02 | −0.04 | +0.02 | 14.8 / 58.6 |
| 0.20 | X | +0.03 / +0.03 / +0.01 / −0.01 | −0.015 | +0.02 | 14.9 / 56.0 |
| 0.20 | Y | +0.04 / +0.04 / +0.025 / +0.015 | −0.03 | +0.02 | 14.1 / 57.8 |

(The rows are the same within ±0.01 in every amplitude bin; the table shows their typical values.)

- **It does not depend on amplitude.** The centre-strip excess is the same fraction of the peak (+0.05–0.06 at
  σ0 0.30) at 800–1500 and at 2500–3550 ADC, in both views. χ²/dof grows with amplitude (15 → 60) only because a
  fixed fractional residual scales as amplitude². This is what cloud_basics §37/§41 called "the template
  under-filling bright pulses", and what MODEL_FRAC and the per-event σ0 profile were patching. Combined with T1 (the
  template shape does not depend on amplitude), **the bright-pulse χ² excess is a shape error that every event
  carries; only the brightest show it.**
- **The lateral part:** the centre strip is above the model and the ±1 neighbours below it, so the model spreads the
  charge too widely. Narrowing σ0 from 0.30 to 0.20 removes about half of it. That matches FC's unexplained "bench σ0
  0.19–0.27 below SPS 0.30", and FC's own physics: σ0 is mostly resistive spreading within the first sample, set by
  *each chamber's* D, so it need not be one fleet number.
- **The temporal part:** in X the centre is above the model at the peak and below it on the late side (k = +2…+3),
  at every σ0. The data pulse is narrower in time than the model, which is the signature of the too-wide template
  (T1).
- Held-out with σ0 0.24 (cur template): χ²/dof −5 % (X), −3 % (Y); s68 −0.011 ± 0.028 / −0.006 ± 0.019 (neutral);
  far +0.5 pt in X (n.s.); ladder +1.3 % (kw absorbs, as FC found).

- **h_eff combined with σ0** (det3, D_rc,y 3.6e-4; held-out Δs68 vs cur/σ0 0.30): X −0.056 / −0.040 / −0.019 at σ0
  0.30 / 0.24 / 0.20; Y +0.047 / +0.041 / +0.047. In X, h_eff removes part of the late-side negative residual (k=+3:
  −0.02 → −0.01). In Y it adds a positive late tail (+0.02 at k=+3). Y wants more temporal spread than X template ⊗
  RC kernel provides, and the old, wider template happened to supply it. The Y penalty is the same at every σ0, so
  it is a real Y-view model deficit, not noise.

## σ0: physical constant, not a per-chamber calibration (2026-10-29, Dylan's question)

**Question (Dylan):** the same detector build at the same HV should have the same σ0. Does the recalibration change
FC's SPS σ0, and did that measurement depend on the old calibration?

**Did the SPS measurement depend on the old calibration?** FC's `fc_headon_grid.py` (det4 run_71 head-on RAW,
three fields) uses only unsaturated events (lead strip 250–3200 ADC) and leaves t0 free per event (no latency
pin). It profiles D_rc,y and a bounded D_x per event jointly with σ0 and Dp. So it is immune to T6 (saturation),
T7 (δ) and the stale D_rc,y. Its one inherited input is the det4 bench template. T10 (`t10_sps_sigma0.py`) re-runs
FC's grid with h_eff to measure that dependence.

**T10 result (raw700, 243 V/cm, 300 views per view, reduced grid: σ0 0.14–0.46, Dp 0.005–0.010, D 1.0–3.5e-4,
summed χ² profiled):** current det4 template X **0.278 ± 0.016**, Y **0.323 ± 0.024** mm. Segment-free h_eff X
**0.282 ± 0.016**, Y **0.324 ± 0.023** mm. **The SPS σ0 does not depend on the template.** The current-template
run reproduces FC's full-grid 0.31 ± 0.01 within errors. So FC's σ0 = 0.30 depends on none of the calibration
issues found here and stands as a physical input. Outputs: `results/t10/`.

**Why bench fits cannot measure σ0:**
- Free σ0 in the joint bench fit (det2, pinned + unsaturated objective, `results/t8/DIAG_det2_free_sigma0.json`)
  runs to **σ0_x 0.058 mm, σ0_y 0.20 mm**, not converged after 126 evaluations. 0.058 mm is below the static
  induction floor (~0.07 mm, FC stack-up), so the fit is using σ0 to absorb other errors.
- The model's lateral width is σ0 ⊕ Dp·√u, with u the drift time in ns. On the bench, Dp ≈ 0.023 mm/√ns gives
  ~0.65 mm at full depth. In the SPS gas, Dp 0.005–0.01 leaves σ0 dominant. σ0 and Dp trade (FC: σ0_X ↔ Dp_X), and
  the bench X width excess (cloud_basics §29–32) and the template's temporal error (T1, T9) feed into the same
  freedom.
- T12 (`t12_neighbour_time.py`, model-free, head-on unsaturated): the ±1 / centre amplitude ratio is 0.35–0.42 (X)
  at the leading edge and the peak, and grows to ~1 by k = +8 (deep charge lands wider). Y shows the RC signature:
  centre undershoot −0.10…−0.13, ±2 rising until k ≈ 6–8. In the T9 residuals at σ0 0.30, the model's ±1 is above
  the data at every k (X −0.008…−0.037, Y −0.013…−0.060). The model over-spreads at all times, which σ0 alone
  cannot fix without breaking ±2 (T9).
- T11 (`t11_footprint.py`, time-integrated R1/R2) is **invalid as a σ0 test** (kept for the record). Integrated
  charge includes the deep, wide-landing charge, and its toy used Dp in mm/√mm instead of mm/√ns.

**Decision basis:** σ0 = 0.30 mm stays a physical constant from SPS, where Dp is small and σ0 is cleanly
separated (pending T10's template check). Bench preferences for a smaller σ0 are compensation for the other
width terms (Dp / the X width excess, the template width, the footprint shape). Those terms are where the
remaining model error lives.

## T8 · Fleet test of the recalibration: resolution-neutral on all five chambers (2026-10-29 night)

`t8_fleet.py` (`T8_ARMS=recal`), `t8_summary.py` → `results/t8/summary.json`. Per chamber:
- **A** = the pass as planned: fe_nobow bundle, σ0 0.30, the bundle's D_rc,y, no t0 offset.
- **B** = σ0 0.30 fixed (a physical input). D_rc,y (det4 kept at 2.2e-4) and δ_x, δ_y are fitted jointly by
  `recal.py --t0-pin --unsat-only` on cache events 580–1079.

Held-out events 180–579, LM, `--t0-fix`, event-paired:

| chamber | δ_x [ns] | δ_y [ns] | D_rc,y fe → B | X Δs68 (χ²/dof A→B; ladder A→B) | Y Δs68 (χ²/dof; ladder) | timing v |
|---|---|---|---|---|---|---|
| det2 | −12.4 | −46.2 | 4.67 → 4.78e-4 | +0.011 ± 0.025 (103.5→102.0; 38.44→38.25) | +0.006 ± 0.041 (100.3→104.9; 39.98→38.77) | 39.3 |
| det3 | −48.2 | −50.2 | 3.44 → 3.54e-4 | −0.031 ± 0.035 (49.1→49.1; 36.20→34.82) | +0.030 ± 0.033 (56.6→55.6; 36.40→35.19) | 35.8 |
| det4 | −76.3 | −43.4 | 2.2e-4 (fixed) | −0.056 ± 0.074 (18.0→15.6; 33.76→32.46) | +0.039 ± 0.054 (13.1→12.0; 36.09→34.63) | 33.0 |
| det6 | −43.3 | −27.8 | 5.90 → 4.89e-4 | +0.022 ± 0.042 (237.4→240.2; 24.01→23.83) | −0.010 ± 0.050 (409.6→400.0; 23.45→23.45) | 24.2 |
| det7 | −28.3 | −9.5 | 6.26 → 6.75e-4 | +0.058 ± 0.067 (135.9→126.7; 36.71→35.85) | −0.019 ± 0.042 (292.0→283.1; 37.48→37.31) | 36.8 |

- **Resolution:** every Δs68 is inside its error and the fleet means are ≈ 0 (X +0.001, Y +0.009). The recalibrated
  pieces are not a resolution lever. What matters is *not* using the raw objective, which cost 0.07° in det3 Y (T6).
- **D_rc,y:** the corrected objective moves the fe values by +2…+8 % on det2/3/7 and −17 % on det6.
- **δ:** negative on every chamber and view, but spread from −9 to −76 ns, with X and Y differing by up to 37 ns
  (det2, det4). A single latency-definition offset would be common to all, so δ also absorbs chamber timing
  (e.g. how well C was measured, gain). Its held-out effect is the angle scale: ladders fall 0.2–1.5 µm/ns, toward
  the timing v on det2 Y / det4 Y / det7 X and past it on det3 X/Y and det4 X. kw absorbs it either way.
- **Recommendation for E1/E3:** (i) DRC with `--t0-pin --unsat-only`, a must, since it is the correct objective.
  (ii) δ: do *not* adopt it as a calibrated quantity. It buys no resolution, its scatter is unexplained, and its
  only effect is an angle scale that kw already measures. Keep t0 = C − 10·ftst and quote δ's ladder shift as part
  of the angle-scale systematic. (Dylan's call; the data allow either.)

## T13 · Drift-diffusion width vs footprint shape (det3, 2026-10-29 night)

`t13_width_shape.py`. Base = fe det3 at D_rc,y 3.44e-4, σ0 0.30, main's LM. Arms: Dp × 0.8 / 1.2; pseudo-Voigt
prompt footprint (η 0.2 γ 0.6 mm; η 0.3 γ 0.8 mm). Residuals use `t9_peak_residual.py` → `results/t13/`
(table: `t13_table.py`, amplitude-weighted medians ≥ 800 ADC). Held-out as T5.

| arm | X residual: centre k=0 / ±1 k=0 / ±2 k=0 | Y residual: centre / ±1 / ±2 | X Δs68 · χ²/dof (base 49.1) | Y Δs68 · χ²/dof (base 56.7) |
|---|---|---|---|---|
| base | +0.050 / −0.032 / +0.010 | +0.057 / −0.058 / +0.010 | — | — |
| Dp × 0.8 | +0.024 / −0.012 / +0.020 | +0.033 / −0.038 / +0.022 | +0.033 ± 0.035 · 45.8 | +0.001 ± 0.028 · 54.3 |
| Dp × 1.2 | +0.081 / −0.052 / −0.005 | +0.086 / −0.080 / −0.006 | **−0.049 ± 0.032** · 63.2 | +0.060 ± 0.025 · 69.8 |
| Voigt .2/.6 | +0.054 / −0.037 / −0.006 | = base | −0.001 ± 0.025 · 49.8 | +0.001 ± 0.013 · 57.2 |
| Voigt .3/.8 | +0.067 / −0.046 / −0.023 | = base | **−0.061 ± 0.030** · 58.2 | +0.013 ± 0.017 · 57.9 |

- **Dp is a stronger lever on the shape than σ0, but it is the same trade.** Narrowing either Gaussian width cuts
  the centre/±1 mismatch and raises ±2. A single Gaussian cannot represent the lateral profile: the data have a
  narrower core and wider tails. Tails alone (Voigt) fix ±2 but add width to the core.
- **Held-out angles disagree with χ² and the residuals.** In X, the wider or tailier arms (Dp × 1.2, Voigt 0.3/0.8)
  improve s68 by 0.05–0.06° (1.5–2σ) while χ² rises 18–29 %. The narrower one (Dp × 0.8) has better χ² and
  residuals but s68 +0.03 and far +1 pt. This is FINDINGS F18 again: a wider footprint makes the fit tolerant of
  δ-rays and noisy strips. **Shape and width choices must be judged on held-out angles (and charge), not on χ² or
  residual pattern.** Every effect here is ≤ 0.06°.
- **Model limitation found:** in Y's RC branch (`build_matrix_rc`, D > 0) the first RC step replaces the prompt
  footprint F0 by a pure Gaussian of variance σ0² + Dp²u + 2Ds. The increments telescope, so Voigt tails only move
  charge between the first two time tensors and vanish from the first RC edge on. Y can never carry footprint tails
  (Y rows identical to base). Harmless in production (tails off), but a Y-tail test needs a model change
  (convolve the RC growth with the Voigt instead of rebuilding a Gaussian). Not done: Dylan's call.
- Fitted charge moves with the footprint (Voigt 0.3/0.8: +11 %; Dp × 0.8: −2 %), so footprint choices are also a
  charge-scale systematic.
- **Narrower core + tails (X only; queue 3):** Dp × 0.8 + Voigt η 0.2 γ 0.6 gives the best X residual shape of any
  arm (centre +0.031, ±1 −0.021, ±2 +0.002) and the best χ² (44.6, −9 %), but s68 +0.027 ± 0.030. η 0.3 γ 0.8: s68
  +0.001, χ² 48.3, charge +9 %. **The best waveform model is not the best angle model.**
- **T14, Y on its own measured template (`rc_tmpl` 0):** Y s68 **+0.100 ± 0.033°**, χ² 56.7 → 76.5. Confirms W4 (X
  template drives both views). Y's missing late spread is in the RC model's form, not the template.

## T9 on all five chambers: the residual pattern is universal (2026-10-29 night)

Plan bundles (T8 A: fe_nobow, σ0 0.30), `t9_peak_residual.py` → `results/t9/<det>_plan.json`, `t13_table.py`
(amplitude-weighted medians, unsaturated planes ≥ 800 ADC):

| | X: centre k=−1/0/+1/+2/+3 | X ±1 k=0 / ±2 k=0 | Y: centre k=−1/0/+1/+2/+3 | Y ±1 / ±2 |
|---|---|---|---|---|
| det2 | +.048 +.057 +.044 +.024 +.007 | −.056 / +.003 | +.076 +.090 +.072 +.046 +.017 | −.080 / +.007 |
| det3 | +.042 +.050 +.025 −.002 −.021 | −.032 / +.010 | +.054 +.057 +.043 +.028 +.010 | −.058 / +.010 |
| det4 | +.055 +.083 +.044 −.005 −.039 | −.013 / +.014 | +.051 +.074 +.041 +.003 −.030 | −.019 / +.015 |
| det6 | +.068 +.114 +.086 +.033 +.002 | −.105 / −.000 | +.191 +.239 +.188 +.123 +.058 | −.147 / −.002 |
| det7 | +.036 +.051 +.026 −.003 −.021 | −.018 / −.001 | +.062 +.076 +.050 +.018 −.007 | −.045 / +.009 |

- **The model under-fills the centre strip on every chamber and view** (+0.05…+0.11, det6 Y +0.24), with ±1
  below the model. It is a fleet-wide model-shape error, not a det3 or calibration quirk.
- **det4 (and det3/det7 X) are mostly temporal.** Centre high at the peak, low late (k = +3: −0.02…−0.04) and a
  small ±1 deficit: the data pulse is narrower in time than the model. That is the template-width signature (T1).
- **det2/det6 are mostly lateral.** ±1 deficit about as large as the centre excess and no late sign change. det6,
  the slow-gas chamber (v 24 µm/ns, so the longest drift times and the largest Dp·√u and 2Ds widths), is extreme in
  Y. Its plan D_rc,y 5.9e-4 is the value the corrected objective moved most (→ 4.9e-4, T8). The Y deficit there is
  the RC growth more than σ0.
- Read together with T13: the shape error is real and universal, but changing widths to fix it does not improve
  angles (and can cost them). It shows mainly in χ² and charge. For the pass it is a stated model limitation (the
  χ² quality flag is relative, 5 × median, so it adapts), unless a footprint/RC shape change is made and validated
  on held-out angles.

## Interim verdict (2026-10-28 evening) — SUPERSEDED by the morning summary at the top

1. The production template is not pure response: it is 4–7 % too wide because it carries the strip's drift segment,
   and the forward model counts that segment twice. Measured on all five chambers, both views.
2. A segment-free template (h_eff) helps X by ~0.05° (consistent over every arm, ~2σ each) and costs Y
   0.02–0.08°. It changes neither χ² nor the far fraction. **Not ready to adopt.** Decide it after the pinned-t0 DRC
   fit, on all five chambers, with 1320 events.
3. The template zero must be set explicitly against the latency (χ² minimum 20–40 ns earlier than t50 = 0). It moves
   the angle scale by ~2 % per 20 ns.
4. No amplitude-dependent template: the shape does not change from 500 to 3500 ADC.
5. ~~The recal.py objective (t0 free) and the pass (t0 pinned) disagree on D_rc,y.~~ **Corrected by T6:** the main
   cause is that saturated planes dominate the raw χ² sum (88 %). The fix is pinned t0 + unsaturated planes, which
   gives 3.6–3.8e-4 on both samples, at the reconstruction's optimum.
6. The template is the strip self-response. Its physical decomposition (peaking time, β, ion fraction) is not
   identifiable from templates (sim closure). Don't quote shaper parameters from it.

## Next steps (2026-10-29 morning)

1. Dylan's calls: δ (recommended not to adopt), model-shape change vs stated limitation, det6 Y RC look.
2. E1/E3 code: the DRC node calls `recal.py --free rc_D_y --t0-pin --unsat-only --train-start <disjoint>`. Add the
   D_rc,y convergence criterion (±5 %) on that objective. TMPL unchanged.
3. If a shape change is wanted: fix `build_matrix_rc` so the Y RC growth convolves the prompt footprint instead
   of rebuilding a Gaussian (Voigt tails survive in Y). Then judge on held-out angles and charge across the fleet,
   not on χ².
4. Optional, decides nothing for the pass: the h_eff fleet arm (C), sim-side follow-up on the resistive self-term
   (sim X undershoot −0.10 vs data −0.02).

## Not ruled out / open

- These caches are 1500 events per chamber: 40–240 inclined pulses per bin. The pass's TMPL nodes see 8000 matched
  events.
- The extrapolation assumes the segment adds in quadrature (T1b), or as a box (T4). Both give the same h_eff width
  within ~10 ns.
- The Y template's RC content is not separated here (rc_tmpl='x' sidesteps it).
- The resistive self-term the sim shows has not been compared with data directly. The data X undershoot (−0.02) is
  much shallower than the sim's (−0.10), so the sim's resistive self-term in X may be too strong. That is an input for
  MX17_Geant, not for the pass.
