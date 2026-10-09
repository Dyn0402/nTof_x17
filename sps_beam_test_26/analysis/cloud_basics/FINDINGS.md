# Model basics: prompt cloud, drift diffusion, resistive spreading, timing (2026-10-09)

Dylan's questions after the per-view ratio study (`../plane_ratio/FINDINGS.md`):

1. The beam fits give a prompt cloud `sigma_p0` ~0.15 mm, the bench 0.4 mm. Is that
   real? Gas (even wet) should not change it much; the amplification cloud should
   be one number across detectors and conditions. Make it consistent first, then
   deal with the sharing.
2. Where does the model's copy timing come from, and why are Y's copies too early
   on the beam?

Everything here is measured on head-on tracks (the view's |tan| < 0.03), where every
depth lands at the same strip, so the neighbour signal is pure spread and the drift
ladder is absent. Bench: the five golden big caches. Beam: det4 run_71 RAW at three
drift plateaus (`../plane_ratio/sps_cache.py`).

## 1 · The two fitted sigma_p0 values are a parameter split, not a physical difference

The model has three ways to put charge on a neighbour: the prompt Gaussian
`sigma_p0`, drift diffusion `Dp`, and the delayed copies (`c1`, `c2`, `kY`, `tau_s`,
`sigma_s`). The bench calibration puts `c1` on its floor (0.05) and loads the
neighbour charge into `sigma_p0` (0.39–0.43 mm). The beam fit, with det4's `lp`
kernel, loads it into copies that peak almost with the centre strip, leaving
`sigma_p0` at 0.15 mm. Same data features, different bookkeeping.

Model-free (`neighbour_vs_time.py`), ±1/centre at the leading edge (first charge):

| | X | Y |
|---|---|---|
| bench det2/3/4/7 | 0.27–0.34 | 0.36–0.49 |
| bench det6 | 0.14 | 0.44 |
| beam det4, 243 / 150 / 92 V/cm | 0.26 / 0.26 / 0.25 | 0.28 / 0.29 / 0.26 |

The beam number does not move with the drift field. The prompt neighbour fraction
is a detector property.

## 2 · Prompt + delayed decomposition with the centre strip as its own basis

`decompose.py`: W(±1) = a·W0 + b·LP_tau(W0). No template, no drift profile.

Beam det4 (all three fields, fit to 1.3 µs and to 2.0–2.4 µs):

| | prompt a1 | delayed b1 | tau | ±2 prompt a2 | ±2 delayed b2 |
|---|---|---|---|---|---|
| X | 0.32–0.38 | 0.00–0.04 | — | 0.03 | 0.00–0.03 |
| Y | 0.27–0.29 | 0.21–0.27 | 530–710 ns | 0.05–0.07 | 0.19–0.28 |

- **X has no delayed component on the beam.** All of X's neighbour charge is prompt.
- **Y's slow component reaches ±2 as strongly as ±1** (b2 ≈ b1). That is not a
  chain of discrete copies; see §3.
- On the bench the same decomposition is unphysical for X (negative a, tiny tau),
  because drift diffusion makes the neighbour/centre ratio grow through the pulse
  (later charge comes from deeper and lands wider). On the beam the charge is
  front-loaded (§5), so that growth is absent and the decomposition is clean.

## 3 · Y's slow component is RC diffusion along the resistive strip

The board (`mpgd26/scenes_chamber.py`, from the gerbers): resistive film in strips
550 µm wide on a 0.80 mm pitch, **running along y**, 150 µm below the mesh; pads,
then Y and X readout strips below. Charge on the film can move along y (an RC
line: sigma_y^2 grows as 2·D_rc·t) but not across the strip in x.

`rc_diffusion.py` fits each view's head-on (offset, time) stack with
W_o = S ⊗ dF_o/ds, S = all-strip sum (electronics × arrival, from the data),
F_o the fraction over strip o of a Gaussian with sigma^2 = sigma_0^2 + 2·D_rc·s.

| | X: D_rc [mm²/ns] | Y: D_rc [mm²/ns] | Y spread after 1 µs | Y residual, prompt-only → +RC |
|---|---|---|---|---|
| beam det4 (3 fields) | 1.2–2.1e-5 (≈ 0) | 2.0–2.3e-4 | 0.63–0.68 mm | 0.018–0.020 → 0.0055 |
| bench det4 | 1.6e-5 | 3.3e-4 | 0.81 mm | 0.022 → 0.0049 |
| bench det3 | 1.9e-5 | 4.9e-4 | 0.98 mm | 0.026 → 0.0060 |
| bench det2 | 2.5e-5 | 6.4e-4 | 1.13 mm | 0.024 → 0.0064 |
| bench det7 | 1.6e-5 | 7.4e-4 | 1.22 mm | 0.022 → 0.0056 |
| bench det6 | 2.4e-5 | 5.4e-4 | 1.04 mm | 0.020 → 0.0107 |

One parameter per chamber fits Y about 4× better than prompt sharing alone; X
shows none, as the geometry requires. D_rc = 1/(R_s · c): 3e-4 mm²/ns with a
~50 µm insulator is a surface resistivity of a few MΩ/□, i.e. a construction
constant that differs chamber to chamber (det4 lowest, det7 highest). det4 bench
vs beam (3.3 vs 2.0–2.3e-4) is the same chamber in different conditions; the
bench value still carries some drift-diffusion leakage.

**Consequence for the c2 < c1 question:** under continuous RC spreading there are
no discrete copies, so there is no c2/c1. ±2 is reached through ±1 automatically,
and late in the pulse ±2 can approach ±1, which is what the data show.

## 4 · Where the model's timing comes from

- `tau_s`, `sigma_s` (and `c1`, `c2`, `kY`) are **fitted on bench cosmics** by
  `wft.calibrate.fit_hypers`; they do not come from the Garfield/MX17_Geant
  simulation. Production values: tau_s 105–166 ns, one value for both views.
- The template ("impulse response") is **measured from the brightest strip of
  inclined bench tracks** (`measure_templates`). It therefore contains the
  strip's own depth segment and the copies it receives from its neighbours. Beam
  bundles reuse the bench template (`template='seed'`).
- Direct waveform measurements give much slower Y spreading: 410 ns on the det3
  bench (X 230 ns, HANDOFF_2026-07-30), 300–700 ns on the beam (here and
  `M70V_FLAT_ANALYSIS.md`). The fitted 105–166 ns is a compromise the bench
  χ² finds when the copy form cannot represent a √t spread; the "Y copies
  13–15σ too early on the beam" and "Y ±2 delay too short on the bench" closures
  are the same defect seen twice.
- DREAM shaping is the same on bench and beam (peaking 180 ns,
  `M70V_FLAT_ANALYSIS.md`); the electronics are not the difference.

## 5 · ~~The beam gas lost the deep charge~~ — RETRACTED (§10: stacking artefact)

Head-on pulses (`pulse_shapes.py`, figures/pulse_shapes.png): the bench pulse is a
flat-topped box of ~800 ns (gap / v, uniform ionisation). The beam pulse is a
~300 ns burst followed by a ~0.05–0.1 tail out to ~2 µs (the measured v is 12–14
µm/ns at 240 V/cm, `EXTRACTION_2026-08-05b.md`). About two thirds of the charge comes from the
first few mm of drift: the wet run_71 gas attached most electrons from deeper.
That is why the beam measures the prompt footprint almost free of drift diffusion,
and why beam and bench fits weight diffusion so differently.

## 6 · Bench: prompt footprint vs drift diffusion are degenerate on head-on data

`bench_joint.py` (both views, shared sigma_0 and drift diffusion Dd, Y's D_rc,
electronics recovered as the step response of the box-shaped head-on pulse):
it describes the data (rms 0.005–0.012) but sigma_0 trades against Dd and the
box length T (det3 sigma_0 0.36 ↔ 0.48 as Dd moves ×5). Head-on bench data alone
cannot separate them; Dd must come from Magboltz + the measured v and T from
gap / v. Magboltz runs (dry and wet, both gases, drift and amplification
fields): condor 4410527 / 4410529, `magboltz_drift.py`.

Magboltz, dry (from the earlier tables): Ar/iso 95/5 D_T ≈ 475–500 µm/√cm at
80–400 V/cm; Ar/CF4/iso 88/10/2 D_T ≈ 215–250 µm/√cm at 150–250 V/cm.
The production bench Dp (0.014–0.015 mm/√ns, i.e. ~240 µm/√cm at v = 37 µm/ns)
is half the dry Magboltz value; the missing diffusion is in sigma_p0.

## 7 · Magboltz, dry and wet (condor 4410527 / 4410529, `results/magboltz_*.json`)

| gas, ~200 V/cm | v [µm/ns] | D_T [µm/√cm] |
|---|---|---|
| bench Ar/iso 95/5, dry | 41 | 488 |
| + 0.5 % H2O | 31 | 341 |
| + 1 % H2O | 18 | 248 |
| + 2 % H2O | 9 | 183 |
| beam Ar/CF4/iso 88/10/2, dry | 67 | 230 |
| + 0.5 % / 1.7 % / 3 % H2O | 29 / 10 / 6 | 200 / 175 / 160 |

(The JSON's `v_um_ns` field is Garfield cm/ns × 1e3, i.e. units of 10 µm/ns.)

Amplification field (25–40 kV/cm, bench gas): D_T 180–225 µm/√cm, so over the
150 µm gap the avalanche spreads **~25 µm**. The measured prompt footprint is
350–450 µm. Gas, wet or dry, cannot make or change it: it is the readout
(550 µm resistive strips, induction onto the pads, inter-strip coupling).
Drift diffusion is the gas-dependent term, and water changes it by up to ×2.5.

The bench chambers' measured v (34–40 µm/ns; det6 26.7) put them at ~0.1–0.6 %
water (det6 ~0.9 %) at 250 V/cm, i.e. D_T 470–340 (det6 290) µm/√cm.

## 8 · ~~The bench also loses deep charge~~ (RETRACTED, §10), and Y's template is contaminated

The X template (bundle `tmpl_x`) is clean electronics: unipolar, −2 % undershoot.
A uniform head-on track through it would give a flat-topped pulse to ~600 ns;
every bench chamber's pulse instead sags steadily (det4: 0.93 → 0.52 between 200
and 600 ns where a box gives 0.90 → 0.99). The arrival current falls with depth:
**attachment on the bench too**, with lengths 13–35 mm (beam: a few mm).

The Y template has a −7…−9 % undershoot X lacks; DREAM is the same for both views,
so that undershoot is charge leaving along the resistive strip. Y's "impulse
response" in every bundle carries part of the RC spreading.

## 9 · ~~Bench fit with attachment~~ — SUPERSEDED by §10 (fitted the stacking artefact)

h = X template (both views), I(u) = exp(−u/λ) over T = 30 mm / v, footprint
σ0² + 2·Dd·u (+ 2·D_rc·s on Y), per-view amplitude and time offset.

| chamber | σ0, Dd = Magboltz | σ0, Dd free | Dd free / Magboltz | λ | Y D_rc | rms (Magboltz / free / Dd = 0) |
|---|---|---|---|---|---|---|
| det2 | 0.353 | 0.423 | 0.62 | 35 mm | 4.5e-4 | 0.0081 / 0.0076 / 0.0090 |
| det3 | 0.405 | 0.407 | **0.99** | 19 mm | 3.4e-4 | 0.0091 / 0.0091 / 0.0102 |
| det4 | 0.404 | 0.449 | 0.39 | 13 mm | 2.9e-4 | 0.0073 / 0.0070 / 0.0071 |
| det6 | 0.287 | 0.359 | 0.27 | 29 mm | 8.2e-4 | 0.0131 / 0.0125 / 0.0126 |
| det7 | 0.516 | 0.504 | **1.11** | 30 mm | 6.2e-4 | 0.0087 / 0.0087 / 0.0094 |
| beam det4 (`rc_diffusion.py`) | X 0.40–0.43, Y 0.37–0.39 | | | ~4 mm | 2.0–2.3e-4 | |

- **σ0 ≈ 0.40 mm, and det4 reads the same on bench (0.40) and beam (0.40–0.43)**
  — different gas, field, water and electronics settings. Chamber to chamber
  0.29 (det6) to 0.52 (det7).
- Free drift diffusion lands on Magboltz for det3/det7, below it for det2/4/6
  (the water estimate from v, the box length and attachment all enter;
  det4 has the strongest attachment, so the least deep charge to measure Dd on).
- None of these fits uses the copy kernel; the production `c1/c2/kY/tau_s/sigma_s`
  description is replaced by σ0, Dd (Magboltz) and one D_rc per chamber.

## 10 · Correction: there is no attachment; the sag was my stacking

Dylan: we had excluded attachment (`mx_june_wft/GAP_STUDY_2026-07-30.md`: X
NNLS charge profiles of contained tracks flat to the cathode, tau_att 22 us /
infinite; memory `gas-water-synthesis`: η ≈ 0 on waveform evidence). Correct.

- **Template-free test** (`charge_vs_depth.py`): inclined X tracks
  (|tan| 0.15–0.35), each strip's integrated charge against its depth segment.
  Flat within ±5–10 % from 2 to ~22 mm on all five chambers; the fall in the
  last few mm is the gap end (det3's known 27.9 mm column). No attachment.
- **The cause of the "sag":** §1–§9 stacks normalised each event to its own
  maximum and aligned it on 50 % of that maximum. Ionisation along a track is
  clustered (Landau), so the maximum sits on the largest cluster and, on
  average, less charge follows it: a uniform track stacks into a burst + decay.
  Stronger on the beam, where slow drift puts fewer clusters per sample.
- **Unbiased stacking** (`pulse_unbiased.py`; now the default in
  `rc_diffusion.stack`): no normalisation, aligned on a fixed 60 ADC threshold,
  plain mean. Bench head-on pulses are flat-topped and match box(30 mm / v) ⊗
  X template (det7 to ±0.02 everywhere; det3 short = its 27.9 mm gap). The beam
  pulse has a long plateau across the ~2 µs drift but declines 0.98 → 0.56 over
  1.2 µs; whether that is charge loss in the wet beam gas or spread beyond the
  5-strip sum needs the template-free test on the 25.6° beam data. Open.
- **What survives:** the RC fits (§3) and the decomposition (§2) are linear
  relations that hold event by event, so stacking cannot fake them. Re-run on
  the unbiased stacks: beam Y D_rc 2.0–2.15e-4 (unchanged), bench Y D_rc
  5–9e-4, X D_rc ≈ 0 everywhere.
- **What does not:** every σ0 / Dd number in §6, §9 (attenuation λ fitted the
  artefact).

Redone on unbiased stacks, uniform charge, X template as the electronics
(`bench_attach.py`, `results/bench_uniform.json`):

| chamber | σ0 (Dd free) | Dd free → D_T [µm/√cm] | Magboltz Ar/iso dry | Y D_rc |
|---|---|---|---|---|
| det2 | 0.42 | 499 | 488–499 | 4.2e-4 |
| det3 | 0.41 | 463 | | 2.8e-4 |
| det4 | 0.44 | 505 | | 2.8e-4 |
| det7 | 0.56 | 507 | | 5.7e-4 |
| det6 | 0.35 | 356 (det6's slow v: ~0.5 % water) | | 7.9e-4 |

**The fitted drift diffusion matches dry Magboltz on four of five chambers with
nothing tuned.** The water estimate from the bundle v in §7 (0.1–0.6 %) was too
high (the bundle v is itself kernel-dependent, cf. GAP_STUDY's v_geom).

Leading-edge prompt footprint (first charge, unbiased stack, r1 → equivalent σ):
bench X det2 0.43, det3 0.47, det4 0.50, det6 0.28, det7 0.41–0.46; beam det4 X
0.42–0.51, Y 0.36–0.38. det4 bench ≈ beam. The beam forward fit with the bench
template (`beam_fit.py`) does NOT close (rms 2–3× the bench; parameters jump
between plateaus; v at the two lower fields is window-floored), so beam σ0 is
quoted from the template-free leading edge only.

The beam's X > Y prompt width (0.42–0.51 vs 0.36–0.38) is consistent with the
board: Y strips are L5, X strips L6, one layer deeper below the pads, so X's
induced footprint should be wider. Per-view σ0 is therefore physical, but
expected to be a fixed geometry ratio, not a free per-chamber knob.

## 11 · The beam decline is common to X and Y (Dylan: attachment would hit both views)

`beam_xy_pulse.py`, run_71 head-on in BOTH views, unbiased stacks, t from the X
threshold, strip sums ±0/±1/±2/±4. Wide (±4) sums, each / its value at 300 ns:

| field | view | 600 | 900 | 1200 | 1500 | 1800 | 2100 ns |
|---|---|---|---|---|---|---|---|
| 243 V/cm | X | 0.945 | 0.967 | 0.926 | 0.779 | 0.765 | 0.453 |
| | Y | 0.920 | 0.962 | 0.906 | 0.736 | 0.730 | 0.445 |
| 150 V/cm | X | 0.910 | 0.876 | 0.785 | 0.730 | 0.711 | 0.654 |
| | Y | 0.909 | 0.855 | 0.760 | 0.748 | 0.681 | 0.659 |
| 92 V/cm | X | 0.797 | 0.801 | 0.756 | 0.769 | 0.678 | 0.571 |
| | Y | 0.805 | 0.792 | 0.730 | 0.763 | 0.653 | 0.587 |

- Same in both views: in the arriving charge, not a readout effect. The
  earlier "X decline" was X alone on a narrower sum.
- Y's narrow sums fall faster than its wide ones (the RC spread carrying charge
  outward); X's do not. Spreading changes the narrow sums, never the totals.
- Field-dependent at fixed time (by 600 ns: −5 % at 243, −9 % at 150, −20 % at
  92 V/cm), so not electronics (the electronics see a flat current until the
  drift ends, identically at every field). Loss faster at low field = O2
  attachment, enhanced by the run_71 water. Magboltz check (1.7 % H2O +
  0.05–0.2 % O2): condor 4410533.
- Leading spike growing at low field (5 → 20 %): hypothesis = primary
  ionisation inside the amplification gap, relatively larger as v (and the
  drift current) falls. Untested.
- Bench control (`beam_xy_pulse.py det3 det4 det7`, both views head-on, n 40–85):
  X and Y wide sums agree and stay flat to the drift end. Bench: no attachment.
- Still to do: template-free charge per strip against depth on the 25.6° beam
  runs (raw files deleted locally, ZS; needs a re-pull and censoring care).

## 12 · Physical kernel, held-out bench (no refit; constants from §10)

`make_rc_arms.py` → `plane_bench.py` (2000 held-out events) → `compare_bench.py`
(paired, SCALE-CORRECTED s68 = after dividing each arm's tan by its own slope;
negative = better). rcm: Dp from the head-on fit; rcmd: Dp from dry Magboltz.
Blind start:

| chamber | Y all | Y head-on | X all | X head-on | slope X / Y (prod → rcm) |
|---|---|---|---|---|---|
| det3 | **−0.090 ± 0.018** | **−0.136 ± 0.031** | +0.005 ± 0.018 | +0.010 ± 0.026 | 0.99→0.96 / 1.00→0.97 |
| det2 | **−0.154 ± 0.034** | **−0.362 ± 0.050** | +0.004 ± 0.023 | −0.026 ± 0.030 | 0.97→0.94 / 1.00→0.95 |
| det7 | **−0.117 ± 0.046** | **−0.278 ± 0.066** | −0.025 ± 0.055 | **−0.161 ± 0.054** | 0.98→0.95 / 1.01→0.97 |
| det4 | **−0.145 ± 0.047** | **−0.204 ± 0.059** | **+0.164 ± 0.049** | +0.008 ± 0.060 | 0.94→0.86 / 0.98→0.91 |
| det6 | **−0.546 ± 0.065** | **−0.897 ± 0.091** | **−0.369 ± 0.049** | **−0.500 ± 0.072** | 0.92→0.89 / 0.92→0.87 |

rcmd (pure Magboltz diffusion) is within errors of rcm everywhere. The ref start
agrees (det2 Y −0.10/−0.30, det6 Y −0.27/−0.70, det4 X +0.09).

- Y improves on every chamber with no fitted kernel constant.
- X: neutral on det2/3, better on det6/7, WORSE on det4 overall; X χ²/dof is
  worse than production everywhere (det3 61 → 84, det2 110 → 189). The model's X
  is missing something production's small X copies were absorbing.
- Every rc arm reads angles 3–8 % flat (production kw would absorb it); the
  bundle v was calibrated under the old kernel.
- Refits of the physical constants only (per-view σ0, D_rc_y; + D_rc_x; + Dp):
  condor 4410535, `recal.py --free ... --seed-json arm_<det>_rcm.json`.

## 13 · ~~Magboltz attachment does not reproduce the beam decline~~ — CORRECTED by §14 (threshold-aligned stacks)

> The Magboltz numbers below stand. The data side does not: the §11 stacks are
> threshold-aligned, and that alignment creates a field-ordered spike and early drop
> (§14). In trigger-aligned stacks the decline is field-independent in time, and one
> O2 level of ≈ 130–240 ppm fits it.

`att_compare.py` → `results/att_compare.json`. Gas Ar/CF4/iso + 1.7 % H2O at
720.8 Torr, with 0 / 0.05 / 0.1 / 0.2 % O2.

- **Water alone attaches nothing:** η = 0 at every field. The "enhanced by the
  run_71 water" in §11 has no Magboltz support. Any loss must come from O2, or
  from something that is not attachment.
- **O2's rate per unit time hardly depends on field.** η·v per 0.1 % O2 is
  13.8 / 11.6 / 11.4 ×10⁻⁴ ns⁻¹ at 243 / 150 / 92 V/cm, so at fixed time it is
  slightly *faster* at high field. The data's early drop is the other way round
  (−5 / −9 / −20 % by 600 ns), and in size and field trend it matches the leading
  spike of §11 (5 → 20 %). Taken at 300 ns the reference sits on the spike's
  tail, so the "drop by 600 ns" is mostly the spike decaying, not loss.
- **After the spike (660–1300 ns)** the log-slopes (X/Y wide sums) need
  **34–43 ppm O2 at 243 V/cm** but **143–216 ppm at 150 and 92 V/cm**. All three
  plateaus are one 30-min block of run_71 (05:22–05:52, `RAW_RUN71_STATUS.md`,
  one gas fill), so a single O2 level cannot produce both. Attachment can be at
  most a small part of the decline, bounded by the 243 V/cm plateau (≲ 40 ppm,
  ≈ 5 % per µs).
- The decline is still common to X and Y, so it is in the arriving charge (§11),
  but its mechanism is open. The leading spike (§11 hypothesis: primary
  ionisation inside the amplification gap) is now the first thing to test,
  because it alone explains the field ordering of the early drop. The 243 V/cm
  step at ≈ 1.4 µs (0.93 → 0.76) is not exponential either.
- The bench result stands: no attachment (§10).

## 14 · The leading spike is the alignment; the decline is field-flat in time (≈ 150–200 ppm O2)

`spike_test.py` → `results/spike_test.json`. run_71 RAW, 20 000 events per plateau,
9-strip X and Y sums centred on the **uRWELL prediction** (not on the pulse). Two
stacks: **trig**, where time is the DREAM sample index (the window opens on the
beam trigger, which is asynchronous to the 60 ns clock, so this smears by one
sample and cannot create structure), and **thrN**, the §11 method (aligned where
X crosses N ADC).

Spike = stack peak / level 360 ns after the peak:

| field | trig | thr30 | thr60 (§11) | thr120 |
|---|---|---|---|---|
| 243 V/cm | 1.110 | 1.150 | 1.057 | 1.440 |
| 150 V/cm | 1.063 | 1.160 | 1.092 | 1.605 |
|  92 V/cm | 1.099 | 1.193 | 1.251 | 1.932 |

- **The field-ordered spike is made by the alignment.** It grows with the threshold
  (up to ×1.9 at 92 V/cm) and is absent from the trigger stacks, which show the same
  ≈ 1.06–1.11 peak at every field. At low v the drift current per ns is small, so a
  fixed threshold is crossed when a large ionisation cluster arrives; the
  alignment then puts that cluster at t = 0 in every event. This is the same class
  of error as the §10 stacking sag. The "primary ionisation in the amplification
  gap" hypothesis is no longer needed. It was also too small: a ~150 µm gap holds
  ≈ 0.5 % of a 30 mm column's primaries, at ~1/ln G of the gain, so ≈ 0.05 %.
- **Trigger-aligned, the X stacks are near-identical at all three fields** from the
  peak to 2.6 µs (0.99 → 0.71), with the same rise. So the decline depends on time,
  not depth — which is what Magboltz predicts for O2 (η·v is field-flat, §13).
- Exponential slopes (baseline = first 5 samples), converted with the 0.1 % O2 rate:

| window [ns] | 243 X / Y [ppm] | 150 X / Y | 92 X / Y |
|---|---|---|---|
| 1260–2580 | 143 / 167 | 188 / 229 | 180 / 225 |
| 1500–2580 | 141 / 161 | 209 / 233 | 216 / 242 |
| 1260–2100 | 125 / 156 | 155 / 239 | 132 / 210 |

  One O2 level of **≈ 150–200 ppm** (with 1.7 % H2O, which itself attaches nothing)
  describes all three fields within ±25 %, against the 5× spread of §13. Residual
  tension: Magboltz expects 243 V/cm to decline ≈ 20 % *faster* than 92 V/cm per
  ns; the data have it 5–20 % slower.
- **Not the electronics:** a high-pass that removes 30 % of a long current pulse
  leaves an undershoot of ≈ −0.3 of the peak when the current stops. At 243 V/cm the
  drift ends inside the window and the X stack settles at **−0.02** of the peak.
- **Y declines 15–25 % faster than X** in every window, which fits Y's RC spread
  carrying charge out of the ±4 sum at late times (§11). The X rate is the cleaner
  attachment measure.
- **Open:** the bench "no attachment" control (§10, §11) used threshold-aligned
  stacks too. It needs redoing trigger-aligned, with the scintillator t0 and ftst,
  on the five golden big caches (on the desktop data drive, not on the laptop). For
  the model: a per-run attachment length on beam. For 150–200 ppm, λ = 1/η ≈ 45–65 mm
  at 243 V/cm (35–50 % of the cathode-end charge lost) and ≈ 19–26 mm at 92 V/cm.

_(sections appended as results land)_
