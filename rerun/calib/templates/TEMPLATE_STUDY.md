# Pulse templates for the from-scratch pass: investigation (started 2026-10-28, laptop)

Part of the calibration rebuild (`rerun/PASS_PLAN_2026-10-28.md` §2, E1 / CAL1 "TMPL"). The question is whether the
template that `wft.calibrate.measure_templates` produces is the right object for the forward model, and what the pass
should use. Scripts, results and figures are in this directory. Run them with `../../../.venv/bin/python -I <script>`.

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

## Verdict so far (2026-10-28 evening)

1. The production template is not pure response: it is 4–7 % too wide because it carries the strip's drift segment,
   and the forward model counts that segment twice. Measured on all five chambers, both views.
2. A segment-free template (h_eff) helps X by ~0.05° (consistent over every arm, ~2σ each) and costs Y
   0.02–0.08°. It changes neither χ² nor the far fraction. **Not ready to adopt.** Decide it after the pinned-t0 DRC
   fit, on all five chambers, with 1320 events.
3. The template zero must be set explicitly against the latency (χ² minimum 20–40 ns earlier than t50 = 0). It moves
   the angle scale by ~2 % per 20 ns.
4. No amplitude-dependent template: the shape does not change from 500 to 3500 ADC.
5. **The recal.py objective (t0 free) and the pass (t0 pinned) disagree on D_rc,y.** That is a blocker for E3's DRC
   step as written in PASS_PLAN §2.
6. The template is the strip self-response. Its physical decomposition (peaking time, β, ion fraction) is not
   identifiable from templates (sim closure). Don't quote shaper parameters from it.

## Next steps

1. Give `recal.py` a pinned-t0 objective (t0 = the bundle's t0_abs per ftst, as `heldout_arm --t0-fix`). Redo the
   D_rc,y refit for cur and h_eff on det3, and check that the refit then improves the held-out Y.
2. Add the template zero as a scanned/fitted quantity in TMPL (χ² vs shift at pinned t0) and as a convergence item.
3. If (1) leaves h_eff ahead or level: run cur vs h_eff (each with its own pinned-t0 D_rc,y) on all five chambers,
   1320 events, on condor. Adopt by the fleet result.
4. For TMPL in the pass: build h_eff from the angle bins (t4's fit, shape only) as a candidate output beside the
   plain median, so the decision does not need another cache pass.

## Not ruled out / open

- These caches are 1500 events per chamber: 40–240 inclined pulses per bin. The pass's TMPL nodes see 8000 matched
  events.
- The extrapolation assumes the segment adds in quadrature (T1b), or as a box (T4). Both give the same h_eff width
  within ~10 ns.
- The Y template's RC content is not separated here (rc_tmpl='x' sidesteps it).
- The resistive self-term the sim shows has not been compared with data directly. The data X undershoot (−0.02) is
  much shallower than the sim's (−0.10), so the sim's resistive self-term in X may be too strong. That is an input for
  MX17_Geant, not for the pass.
