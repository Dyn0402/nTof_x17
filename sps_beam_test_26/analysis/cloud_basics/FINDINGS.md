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

> Direction confirmed by §16, numbers superseded: `spike_test.py` filled the FEU's dropped RAW samples
> with zeros (~6 % fake late loss) and used the block-median common mode. Use §16's rates.

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

## 15 · Bench control redone on the trigger: still no attachment (X), Y reads the RC spread

> Overstated (§16): with ±7-strip sums the bench X fits move by up to ±2e-4/ns. An 800 ns bench drift
> cannot null a 2e-4/ns loss on a head-on forward fit. The template-free bench ladder is the better
> control, and it is only modestly significant.

`bench_trig.py` → `results/bench_trig.json`. The five golden big caches (EOS
`plane_ratio/inputs`), each view used where it is head-on itself (|tan| < 0.03),
9-strip sum centred on the M3 prediction, events placed by the trigger
(t = sample·60 ns − the bundle's `t0_abs[view][ftst]`). The bench drift (~800 ns)
is too short against the shaping to read a slope, so the stack is fitted forward:
A·[X template ⊗ box(0, T)·e^(−r t)](t − t0), with A, t0, T and r free.

| chamber | n (x/y) | X: r [1e-4/ns] → O2 [ppm] | Y: r → O2 | rms X / Y |
|---|---|---|---|---|
| det2 | 268/409 | −1.00 ± 0.31 → −83 ± 26 | +0.21 ± 0.58 → +18 ± 48 | 0.020 / 0.039 |
| det3 | 431/577 | −0.36 ± 0.31 → −25 ± 22 | +4.14 ± 0.81 → +286 ± 56 | 0.018 / 0.055 |
| det4 | 383/545 | −0.48 ± 0.31 → −31 ± 20 | +1.26 ± 0.54 → +83 ± 35 | 0.025 / 0.047 |
| det7 | 217/293 | −0.35 ± 0.18 → −24 ± 12 | +2.36 ± 0.87 → +163 ± 60 | 0.011 / 0.061 |

(det6 skipped: its `calib_bundle_lp` has no `t0_abs`.)

- **X, the clean view, has no decline on any chamber** (r ≤ 0, fits at 1–2.5 % of
  peak): O2-equivalent ≲ 20 ppm, against **125–215 ppm in the beam X view (§14)**.
  The §10 conclusion survives the alignment change; the beam gas carries roughly
  10× more attaching contamination than the bench gas. The small negative r is a
  template/end-of-drift systematic of order −30 ppm.
- **Y reads positive with 2–5× worse residuals.** This is the same Y-faster-than-X
  pattern as on the beam: Y's RC spread carries late charge out of the ±4 sum, and
  the X template has no RC term. It is not attachment, which would hit both views.
- Side observation: fitted T is 1–14 % longer than 30 mm / v_bundle (det4 996 vs
  878 ns). The box carries no longitudinal diffusion, so some of that is the
  end-of-drift smearing. Not chased here.

## 16 · The late-charge observable, rebuilt from the raw samples (report: `make_report.py`)

Dylan (10-09 evening): four reversals already; be sure the observable is there in every dataset claimed,
in BOTH views, and show it. Report: `/home/dylan/x17/cosmic_bench/cloud_basics/attachment/report.html`
(`make_figures.py` then `make_report.py`).

**Two artefacts in every earlier beam stack.** (1) RAW run_71 loses 20–25 % of samples in ~5-sample
packets (present fraction 0.80 early → 0.75 late). Zero-filling them adds ~6 % fake late loss.
`headon_stack.py` keeps them NaN and averages each strip-sample over the events that have it.
(2) The cached run_71 waveforms used a block-median common mode at ±4 strips. Re-extracted with
`extract_det4_only.py --cm masked --keep 12` (new `--keep` option; stripes JSON fallback). The block CM
turns out to bias R by < 1 %.

**run_71 RAW (clean):** X (±2) and Y (±8, where its RC spread is contained) agree. R =
level(2.4–2.7 µs)/level(1.08–1.26 µs) = 0.769 / 0.757 / 0.755 in X at 243 / 150 / 92 V/cm. Forward fit
(box·e^(−rt) ⊗ the det3 bench template, `template_det3_x.json`): r = 1.81 ± 0.09, 1.79 ± 0.04,
1.87 ± 0.04 e-4/ns, the same in time while depth differs ~3× (a per-depth loss is excluded at > 9σ at
92 V/cm). Undershoot after the 243 V/cm drift end: −0.002 (X), +0.004 (Y), against −0.30 for a readout
high-pass with the same sag.

**Discriminators** (`headon_split.py`, `gain_vs_loss.py`, `split_toy.py`). Flat in beam rate and in
spill phase. Flat against local gain over ×1.4–2.8 (det4 stripes): charging would give a slope of
≈ −0.23, the fits give −0.004 … +0.13. Bright events lose more, which is a selection effect attachment
itself creates; the toy at r = 1.9e-4/ns matches the mid and bright terciles and overall R (0.768). The
faint tercile loses less than the toy (≈ 3.5σ at 243 V/cm): open.

**Other datasets.** `ladder_profile.py`: the run_63 25.64° Y ladder is template-free and censoring-proof
(quantiles with an unfired strip = 0). It falls monotonically with depth at 142/108/75 V/cm
(0.043 / 0.054 / 0.045 per mm; per ns 3.3 / 3.2 / 1.8e-4, which reads ~30 % steep from ZS and peak
estimators). `zs_timestack.py`: every ZS arm loses late charge in both views (run_63 flat and rotated,
run_56 CO2). R is selection-dependent at ±0.05 (all events vs the middle 60 %), and ZS Y reads low from
censoring. `zs_emulate.py`: RAW run_71 emulated at ZS 4σ gives X ±1 0.736, matching run_63 flat's real
ZS 0.734. CO2 loses more (R_X 0.49–0.58).

**Bench.** The ladder (`bench_ladder.py`) is flat to ±5 % over 6–20 mm in both views (det4 X excluded:
its stripes run across X). That gives no beam-like loss at modest significance only (see the §15 note).

**O2 equivalent** (Magboltz, 1.7 % water): 131 / 154 / 164 ppm at 243 / 150 / 92 V/cm. Air grid
(N2/O2/Ar, water 0–3 %, three gases): condor 4410646.

## 17 · RC refits (condor 4410535): refitting does not close X

`recal.py` arms seeded from rcm, benched paired against production (`run_bench_rc.sh`, outputs
`~/x17/cosmic_bench/cloud_basics/bench_rc/`, `compare_bench.py --start blind` →
`results/compare_bench_rc.json`).

- **rc_D_x fits to zero** (0–1.2e-5 against rc_D_y 3.6–5.9e-4 mm²/ns), so rcf4x ≡ rcf3. X has no
  resistive spread, as the board requires.
- **Freeing Dp (rcfD) brings back the §6 degeneracy.** Dp drops to 0.013–0.022 (Magboltz ~0.03) and
  σ0 inflates to 0.45–0.59 (X) / 0.47–0.75 (Y), for a 1–2 % χ² gain. The fit is using the footprint to
  absorb a model deficiency.
- **Resolution is the same as rcm (no refit).** Y is better on all five chambers. X is neutral on
  det2/3 and better on det6/7, and det4 X is still worse (rcf3 +0.16 ± 0.05°, rcfD +0.07 ± 0.04°).
- **Next for X:** HANDOFF step 3 (charge snapping to the 550 µm resistive-strip centres, which a
  Gaussian footprint cannot represent).

## 18 · Air model: 1.55 % water + 0.07 % air; the water cannot come from the same leak

Magboltz grid (condor 4410646, `results/air/`): water 0–3 % (bench 0–1 %) × air 0–2 % (N2/O2/Ar
78.08/20.95/0.93, replacing Ar) for the CF4, CO2 and bench gases. `air_fit.py` fits v(E) and r(E)
together. v comes from the run_63 ladder slopes (7.77 / 5.83 / 4.04 µm/ns at 142 / 108 / 75 V/cm) and
the run_71 drift end (14.50 at 243 V/cm, 30 mm gap); r comes from §16. Errors: v 3 %, r 10 %.

- **CF4 period: water 1.55 % [1.53, 1.58], air 0.070 % [0.065, 0.075] = 147 ppm O2**, χ² 12.6 / 5.
  v fits all four fields to 1 %; air does not move v at these levels.
- **The loss rate's field shape is not established.** The data are flat (1.87 / 1.79 / 1.81e-4/ns)
  and the model rises (1.29 / 1.75 / 2.18), but the Magboltz η jitters ±15–20 % between neighbouring
  fields at `GenerateGasTable(5)`. Rerun at higher statistics before reading anything into it.
- **CO2 period** (run_56 ZS; drift field assumed 243 V/cm; v = 12.33 from run_57): water ≈ 1.60 %, the
  same, and r ≈ 3.4–5.2e-4/ns, i.e. 0.13–0.18 % air (270–385 ppm O2), about twice the CF4 period's O2.
- **Water/O2 ≈ 106 in the gas against ≈ 0.05 in room air.** A bulk leak large enough to supply the
  water would bring ~6 % O2. A 0.07 % air leak can supply the O2, but the water needs another path.
  Permeation through polymer tubing fits both: H2O permeates far faster than O2, and both scale as
  1/flow. To check: tubing material and length and the flow on our H4 line. To keep in mind: the
  water-from-v figure assumes the nominal drift field and a 30 mm gap.
- Model caveat (unchanged): three-body O2 attachment with H2O or isobutane as the third body, so the
  ppm scale is uncertain by a factor of a few.

## 19 · The faint-tercile excess was a missing-sample selection artefact (`faint_test.py`)

The §16 per-event charge split classified events by Σ over PRESENT samples. Events that lost
packets on their high-signal part were classed faint: the faint class had 73–74 % of samples
present, the bright class 79–80 %. Among faint events that still had their 1.08–1.26 µs reference,
the reference was biased low and R high. The class also tracked the event's own baseline noise
(pre-level +3…+8 ADC faint, −2…−9 bright). Classified by the per-event mean over present samples,
with a baseline taken from separate pre-trigger samples and normalised by the local (position-cell)
gain, sample presence no longer tracks class. At 243 V/cm the data give 0.837 ± 0.018 /
0.790 ± 0.014 / 0.757 ± 0.014 against the toy (r = 1.9e-4/ns, no gain scatter) 0.808 / 0.799 /
0.732: +1.6σ, −0.5σ, +1.8σ. The faint-to-bright spread is 0.080 (data) against 0.076 (toy).
Resolved: what remains is the attachment selection effect.

## 20 · The ~0.7 µs ripple is a trigger-locked coherent pickup (`ripple_test.py`)

RAW run_71, strips 9–12 away from the track (no signal), averaged at fixed sample index:
- **Y:** a reproducible pattern with a 690 ns period at 150 and 92 V/cm, ±0.3–0.6 ADC per strip
  (half-vs-half correlation 0.49–0.69). It is absent at 700 V drift.
- **X:** a smaller trigger-locked pattern (rms 0.2 ADC, correlation 0.87–0.93) with a sharp feature
  at samples 32–34 (the 2.04 µs bump in wide X sums).

The pickup is locked to the trigger (it survives averaging), so it is excited by the event or the
trigger itself. It appears at the lower drift voltages, the same as in the rotated run_63 blocks
(425/325 V). Hypothesis: the drift HV network ringing. It oscillates around zero, so it cannot
make a monotonic loss; it inflates the Y χ².

## 21 · Bench drift scan (det3, 6-27, 35–382 V/cm): no attachment, no field gradient (`bench_driftscan.py`)

Six sub-runs (100–1100 V drift, resist 490 V, 15 min each) pulled from EOS
(`june_tests/Run/mx17_det3_saturday_scan_6-27-26`). Straight from decoded_root with a masked
common mode; one cluster ≤ 12 mm; all strips in the cluster ±3 mm summed; events placed by the det3
bundle's `t0_abs[ftst]`. 1700–2500 events per field and view; X and Y agree.
- **35 and 104 V/cm: the plateau after the peak is flat to ±2 % out to 2 µs**, i.e. 1.2 µs of drift.
  A beam-like 1.9e-4/ns would remove ~14 % over that span.
- **No field gradient.** A gradient acts through dv/dE, which is largest at low field in Ar/iso, so
  the k ≈ −0.3 that single-field bench fits wanted would tilt the 35/104 V/cm plateaus strongly. That
  k was the plateau degeneracy (undershoot ↔ gradient ↔ air), not geometry.
- **Drift ends inside the window** at 243/312/382 V/cm: T ≈ 1.2 / 0.9 / 0.7 µs, v ≈ 23 / 31 / 40 µm/ns
  over 27.9 mm. At 174 V/cm a slow fall begins by 2 µs, plausibly the shortest (dished-cathode)
  columns ending first.
- Low-field bench Magboltz for the ppm bound: condor 4410787 (35/60/104/174 V/cm).
- The bench gain scan (resist 425–525 V at 1000 V drift) is in the same run and can test
  gain-dependent effects on the bench.

## 22 · Bench gas from single-field stacks: water fits, air is degenerate there (`bench_stack.py`, `bench_fit.py`)

All-strip, trigger-placed stacks (any |tan| ≤ 0.2 with the ladder contained; det4 X head-on only,
because its amplification stripes run across X): 1200–2800 events per view. With gap spread
profiled, det2 and det3 fit well (χ² 38 and 36 / 86) at 0.90 and 1.00 % water and no air. That
water agrees with RECONSTRUCTION_BASIS ("v 36.6 matches Ar/iso + 0.8 % H2O"). det4 and det7 show a
plateau RISING by ~10 %. With a field gradient, the electronics undershoot and air all free, the
single-field bench fits are degenerate (they push k and air to their limits). The drift scan (§21)
removes the gradient. The air bound comes from §21 and the bench ladder, not from these fits.

## 23 · X model: no snapping; the footprint tails are real but do not limit resolution

- `snap_test.py` (bench big caches, |tan| < 0.05). X centroids show no 0.80 mm (resistive-pitch)
  periodicity: Rayleigh power 0.2–1.7, where ~1 is noise. Y shows the 0.78 mm readout DNL at 12–40.
  Snapping is < ~10 %.
- `footprint_test.py`: head-on X profiles have tails beyond the depth mixture (σ0² + D²z) on all
  five chambers. Half of the tail is in the MEDIAN event (det3 at ±3 strips: 0.0053 against a mixture of
  0.0018), so it is not only δ-electrons. A pseudo-Voigt (η 0.10–0.36, γ 0.53–0.91 mm) improves χ²
  by 17–37 %.
- `wft/model.py`: opt-in `lor_frac_<plane>`, `lor_gamma_<plane>` on the physical kernel's prompt
  landing; bit-identical when absent (47 tests). Benches rcm / rcmlor / rcmlor2 (condor 4410788,
  `results/compare_bench_lor.json`): X scaled s68 is within errors on every chamber (det2 −0.021 ± 0.023,
  det3 0.000 ± 0.020, det6 −0.010 ± 0.048, det7 −0.04 ± 0.05, det4 +0.18 ± 0.05); Y is unchanged.
  The tails are real but they are not the X deficit.
- What remains is det4's X: its amplification stripes run across X, which a uniform-gain model
  cannot represent. Next hypothesis, untested.

## 24 · High-statistics Magboltz: η needs smoothing in E; then the beam fits at all three fields

Grid: condor 4410759 (697) + 4410787 (120), 3e8 collisions per (mixture, field), `results/air_hs/`.
Even at 3e8 collisions the attachment estimate scatters ±10–15 % between neighbouring fields (η·v at
75/92/108 V/cm = 1.36/1.66/1.42e-4/ns), so `gasmodel.GasGrid` smooths η, D_L and D_T per mixture
(quadratic in log E); v is left as computed. The coarse grid's v was also off by ~4 % from interpolating
across its 1.0 → 1.7 % water step. Smoothed, η·v(E) is flat from 75 to 150 V/cm and rises at 243.

## 25 · Gas compositions as model curves (`beam_comp_fit.py`, `driftscan_fit.py`, `bench_fit.py`, `make_comp_figures.py`)

One physics model for both setups: `predict.current_field` (uniform ionisation, Magboltz v/η/D_L,
optional linear field profile k and gap spread), a parametric shaper, trigger jitter. No free loss rate.
Shared per dataset: composition, geometry, shaper. Free per stack: amplitude and t0.

| dataset | water | air | O2 | χ² |
|---|---|---|---|---|
| run_63 25.6°, 142 V/cm, Aug 3 00:22 (ZS) | (1.4–1.7) | 0.110 % | 230 ppm | 69/55 |
| run_63 25.6°, 108 V/cm, 00:30 (ZS) | (1.4–1.7) | 0.095 % | 199 ppm | 41/55 |
| run_63 flat, 243 V/cm, 01:00–01:54 (ZS) | 1.59 % | 0.070 % | 147 ppm | 57/55 |
| run_71 RAW, 3 fields × X/Y + ladder v, 05:22 | 1.51 % | 0.078 % | 163 ppm | 560/330 (92 V/cm 76, 73) |
| run_56 CO2 gas, Aug 1 (ZS 5σ, rough) | 1.56 % | 0.139 % | 291 (257–355) ppm | — |
| bench det3, 6 fields 35–382 V/cm | 0.95 % | 0 | ≲ 2 (stat.) / ≲ 10 ppm | 579/546 |
| bench det2 / det4 / det7 (one field) | 0.95 / 0.60 / 0.70 % | ≤ 0.005 / 0.025 / 0.005 % | | 50 / 55 / 191 per 86 |

- The bench needs no air at any field; the same gas with 0.04 % air (half the beam's) is rejected at
  low field (χ² ×2.5–13).
- The beam O2 fell through the night of Aug 2–3 (230 → 147–163 ppm) and was ~2× higher on Aug 1 in the
  CO2 gas. The water is ~1.5–1.6 % throughout. The run_71 composition does not describe run_63 (χ² 136–1196).
- H2O/O2 ≈ 100 on the beam and ≥ 1000 on the bench, against 0.05 for room air: the water does not come
  with the O2. The O2 is specific to the beam line and varies over hours.
- ZS arms use an ADDITIVE ZS distortion (zero suppression removes negative samples, so a ratio fails
  past the drift end), from RAW run_71 at the nearest field, with a 1 % systematic.
- Effective shapers: fitted per setup (they include the gas-dependent ion tail); the DREAM settings
  are identical (register 1 0x081F 0xD023, RdClk_Div 6 on both).
- det7 (bench) fits poorly (rising plateau); bench single-field gradients are limited to |k| ≤ 0.1, the
  size the det3 drift scan supports.

## 26 · Report

`make_figures.py` → `make_comp_figures.py` → `make_report.py --inline …` →
`~/x17/cosmic_bench/cloud_basics/attachment/report.html`, published as
notes/mx17-beam-attachment.

## 27 · Does the beam attachment contaminate the SPS model constants? (2026-10-11)

Dylan accepted §16–26 (beam gas: O2 attachment, ~150–230 ppm, per run; bench gas: moisture only) and
asked whether the attachment spoils the det4 model constants measured at SPS, so that they no longer
apply on the (attachment-free) cosmic bench. Slide note: notes/mx17-sps-to-bench (`make_transfer_deck.py`).

**Answer: no, for every constant the bench model takes; yes, for the prompt/delayed split nobody uses
any more.**

- **Why, in principle.** Head-on, every strip sees the same drive C(t) (arrival current x electronics)
  through its own lateral transfer n_d. Attachment only changes C. The cross-relation
  n_0 (x) W_d = n_d (x) W_0 (sharing_kernel) and the RC fit W_o = S (x) dF_o/ds with S the measured
  strip sum (§3) both cancel C. The leading-edge widths are taken before any loss acts. The exception
  is a depth-dependent footprint (drift diffusion), which lets the changed depth weights back in.
- **Injection test** (`attach_transfer.py`, `results/attach_transfer.json`): the SPS measurement code
  itself (rc_diffusion stack + fit, neighbour_vs_time, decompose, fit_kernel cascade/delay) on toy
  head-on events with known σ0 (X 0.45, Y 0.37 mm) and D_rc (2.1e-4), beam gas from Magboltz at the
  run_71 composition, attachment on/off with common random numbers, 4 x 3000 events per case. Toy vs
  run_71 at 243 V/cm: X σ0 0.52 vs 0.54, Y D_rc 2.4 vs 2.1e-4, delay c2/c1 0.43 vs 0.47, cascade c
  0.62 vs 0.66, b1 0.27 vs 0.22. Shifts at the run_71 level (η·v 2.2e-4/ns) / at 2x:

| constant | shift, run_71 O2 | 2x |
|---|---|---|
| X leading-edge width | −2 ± 2 % | −4 % |
| X σ0 (RC fit) | +2 % | +4 % |
| Y leading-edge width | −4 % | −10 % |
| Y σ0 (RC fit) | +7 % | +11 % |
| Y D_rc | −6 % | −12 % |
| Y c2/c1 (delay form) | 0 % | +1 % |
| Y cascade c | −6 % | −10 % |
| Y delayed fraction b1 (decompose) | −27 % | −51 % |
| Y delay τ (decompose) | +11 ± 5 % | +36 % |

  (4 seeds, paired; errors ≤ 1 % unless given.)
  Halving D_T still leaves D_rc at −4 %, so drift diffusion is not the only path; Y's RC spread carrying late
  charge outside the ±4-strip source sum is the other. **The toy is not realistic below 243 V/cm** (its
  widths grow with falling field, run_71's shrink); the per-field gain match made no difference (the
  9-strip peak is set by the largest cluster, ~1000 ADC at every field in both). The shifts are quoted
  at 243 V/cm; the loss per unit time is the same at all three fields.
- **Beam D_rc with drift diffusion in the fit** (`rc_drift.py`, Magboltz D_T and v at the run_71
  composition, arrival origin t_a fitted): det4 Y D_rc 2.08 / 2.08 / 1.94e-4 at 243 / 150 / 92 V/cm
  (2.23 / 2.26 / 2.11 without the D_T term); σ0 Y 0.39–0.43, X 0.52–0.53 mm. t_a runs to its bound: the
  data want less depth growth than Magboltz. Loss-corrected (÷ 0.94): **det4 D_rc ≈ 2.2e-4 mm²/ns ±10 %**.
- **Bench, same fit, three D_T arms** (none / 275 ≈ 1 % water from v(E) / 480 dry), Y D_rc [1e-4]:
  det2 6.9 / 6.0 / 4.4, det3 5.3 / 4.4 / 3.6, det4 5.5 / 4.6 / 3.0, det6 6.4 / 5.3 / 3.5, det7 9.0 /
  8.0 / 6.2, with σ0_y moving 0.07–0.92 mm across the arms. **On the bench D_rc and σ0 are degenerate
  with the drift diffusion assumed**; the beam (2–3x less drift spread) is where they are measured
  cleanly. X rms prefers the dry arm on det2/3/4/6/7, Y is flat between wet and dry: the bench D_T
  question (v(E) says ~1 % water, widths look dry) is open and now matters for the bench model.
  `width_tan_split.py` (model-free width growth split by reference |tan|, to test an inclination leak)
  was too noisy to decide (det2/det3 ~240 µm/√cm, det4/det6 negative).
- **det4 bench vs beam D_rc: 1.5–2.7x**, under every D_T assumption. Not attachment (none on the bench;
  the correction goes the other way) and not drift diffusion within its range (toy bench-gas leak
  +3…+17 %). Open: resistive-layer conditions (temperature, humidity, five weeks apart), the bench Y
  selection, or a term both fits lack.

**What the bench model takes from SPS:** the structure (Y continuous RC spread, X none, no discrete
copies, c2 < c1 automatic; the prompt footprint is readout, one σ0 per view per chamber; X wider than Y
by board geometry; sharing shape gain-invariant 1–3 %), and det4's numbers (D_rc ≈ 2.2e-4 ±10 %,
σ0 leading edge X 0.42–0.51 / Y 0.36–0.38 mm). **Not:** the H4 gas (v, D_T, η), the beam bundles (old
copy kernel, H4 v/Dp), the decompose b1/τ, or the 176 µm resolution as a bench number (a beam-condition
result; the loss removes ~18 % of the charge at 243 V/cm, so ≲ 10 % pessimistic if statistics-limited).

Next: a bench det4 arm with D_rc pinned at the beam value (paired compare_bench vs the bench-fitted
rcm); settle the bench D_T; then freeze the physical kernel.

## 28 · det4 takes the beam D_rc on the bench; the joint X+Y fit is not degenerate (2026-10-12)

The §27 next step: a bench det4 arm with D_rc pinned at the beam 2.2e-4. Done with the model the
production-candidate arm (rcm, §12) came from — `bench_attach.py`'s unbiased head-on fit over X and Y
together (X-template electronics, uniform arrival over T = gap/v, σ² = σ0² + 2·Dd·u + 2·D_rc·s on Y) —
in `bench_pinrc.py`, and benched with `plane_bench.py` (2000 held-out events, paired).

**det4: the beam D_rc costs nothing, in the stack or in reconstruction.**

| det4 arm | σ0 | Dd [1e-4] | D_rc [1e-4] | stack rms |
|---|---|---|---|---|
| rcf (all free; = rcm refitted) | 0.464 | 4.04 | 2.57 | 0.01635 |
| rcp (beam D_rc, Dd free) | 0.457 | 4.28 | **2.2** | 0.01639 |
| rcpd (beam D_rc, dry Dd) | 0.458 | 4.25 | 2.2 | 0.01639 |
| rcpw (beam D_rc, wet Dd) | 0.591 | 1.95 | 2.2 | 0.01725 |

Paired against rcf (scale-corrected s68, blind / ref start): rcp Y +0.027 ± 0.027 / +0.008 ± 0.019°,
X −0.008 ± 0.026 / 0.000 ± 0.020°; rcpd likewise (Y +0.039 ± 0.027 / +0.007 ± 0.020). Every RC arm is
−0.12…−0.20° on Y and +0.12…+0.19° on X against production, as in §12 (results
`compare_bench_pinrc_det4_*.json`). The reconstruction cannot tell 2.2 from 2.6e-4.

**Why §27 saw 1.5–2.7×.** `rc_drift.py` fits Y alone with D_T assumed. It is not biased: on toy stacks
from this model (known D_rc, dry Dd) it returns D_rc within 2–8 % *when given the true D_T*. The
spread came from the D_T assumptions (none / wet). The joint fit settles D_T itself because X has no RC
term, so X's width growth is drift diffusion alone: profiling D_rc over 1–7.5e-4 (Dd refitted) the
rms has a single minimum on every chamber and rises 6–23 % at 1e-4 (8–37 % at 7.5e-4, except det6 whose minimum is there) (`bench_pinrc.py --profile`,
`results/bench_pinrc_profile_<det>.json`). With dry Dd, det4 bench is 2.57 (joint) / 3.04 (rc_drift)
against 2.2 ± 0.22 beam: **1.2–1.4×, and the stack rms is flat between 2.2 and 3.0e-4.** det4 can carry
its beam D_rc.

**The other chambers do differ from det4** (profile, D_rc free → cost of pinning 2.2e-4):
det2 4.2e-4 (+6.5 % rms), det3 2.8e-4 (+1.5 %), det6 7.9e-4 (+13 %), det7 5.7e-4 (+9 %). D_rc is per
chamber; only det4 has a beam measurement.

**Bench drift diffusion.** Free Dd per chamber against Magboltz at each run's real field
(`bench_water_e.py`): det2 4.97 (wet 4.11 / dry 4.67), det3 3.93 (2.43 / 4.27), det4 4.04 (2.07 / 3.96),
det6 1.69 (1.03 / 3.09), det7 4.69 (2.26 / 4.24) e-4 mm²/ns. Dry fits det2/3/4/7; det6 sits between,
nearer wet. Two candidate explanations tested:

- *Field.* `bench_pinned.py` inverted v → water at 250 V/cm for all chambers. The runs were at drift
  1000 V (det2, det3) = 347 V/cm, 700 V (det6, det7) = 243 V/cm and 600 V (det4) = 208 V/cm. At the real
  field (air_hs grid) det2 goes to ~0.1 % water (its v sits at the dry value), det4 0.60 → 0.43 %,
  det3 0.48 → 0.59 %, det6/det7 unchanged. A real correction, but it does not reconcile det3/4/7.
  `bench_pinned.json` is left as it was: its own fits predate the unbiased stacks and re-running them
  now puts the box model at its bounds.
- *Inclination leak.* Tightening the head-on cut from 0.05 to 0.01 (`bench_tanscan.py`) leaves Dd flat
  (det3 3.95 → 3.54, det4 3.98 → 4.30, det7 4.28 → 5.72 e-4) and D_rc flat. The leak goes into **σ0**
  instead: det4 0.497 → 0.365, det3 0.442 → 0.372 mm, i.e. down to det4's beam Y leading-edge width
  (0.36–0.38). So the footprint transfers too once the inclination is cut out, and the σ0 in the arms
  (|tan| < 0.03) carries ~0.1 mm of track walk.

So the bench D_T tension is now: v(E) (bundle v, and det3's six-field scan, §25) says 0.4–0.6 %
water, while the X width growth says Magboltz dry, on 4 of 5 chambers. Left: Magboltz's D_T for
small water admixtures, or an X width term that grows with drift time and is not diffusion. The beam
(§27: t_a at its bound, *less* depth growth than wet Magboltz) leans the other way, which argues
against a missing Magboltz D_T term common to both.

**For the physical kernel:** det4 can take D_rc = 2.2e-4 from the beam; the other chambers keep their
bench D_rc; Dd per chamber from the joint fit (≈ dry), not from v; σ0 should be measured on the
tight-|tan| stack. None of these choices moves det4's held-out resolution by more than ±0.04°.

## 29 · The X width excess: every dataset, not the gas, everywhere on the chamber (2026-10-12)

> **Status: UNSOLVED, needs follow-up.** The mechanism is not identified. The follow-up list (det3 Y,
> a Magboltz cross-check, a beam template, other contaminants) is in HANDOFF.md, section "X width excess".
> **No new data can be taken (the detectors are gone, 2026-10-16): the question must be settled with the data in hand.**

Dylan asked whether the X widening that Magboltz cannot explain (§28) holds on every dataset, what it
could be, and whether it depends on position. Report: `~/x17/cosmic_bench/cloud_basics/xwidth/report.html`
(`make_xwidth_report.py`).

**Data.** `headon_cache.py` builds every head-on reference event of a run (|tan| < 0.06 in either
view, no 3000-event cap). It has a ref-only mode (`--subrun`) that takes the M3 rays through another
sub-run's alignment, and on the det3 long run it reproduces the hits-chain reference exactly. Caches
are in `~/x17/cosmic_bench/cloud_basics/xwidth/caches/`:
- 10 bench runs on 5 chambers;
- det3's 6-27 drift scan (6 fields) and two amplification scans (18 sub-runs at 347 V/cm), pulled
  from EOS `cosmic_bench/june_tests/Run/`.

The fit is `x_width.py`, X only, so there is no RC term and the width growth is all depth: bench_attach's
model, with σ0, Dd, T, amplitude and offset free and bootstrap errors. It is vectorised (`predict_x`, 90×
faster), and its last slice is fractional so that T can be fitted; `bench_attach.predict` rounds T to
whole 30 ns slices. D_T = σ_drift(gap)/√gap, which barely depends on v, so v enters only through the
composition inferred from it.

- **Every bench dataset, the same excess.** X D_T is 400–520 µm/√cm on det2/3/4/6/7, on every run
  (det3 5 runs, det7 2, det2 2). Magboltz Ar/iso 95/5 with the water that each run's v needs
  (0.6–1.0 %) gives 270–325. In quadrature the extra is **D_x ≈ 300–430 µm/√cm on every chamber**
  (det6, the wettest, the lowest). det6 and det7 ran in the same gas at the same field with v 35 %
  apart: D_T moves the way water says (399 vs 475), but both sit ~+300 in quadrature above it.
  Dropped: g_det6_longer (σ0 at its bound) and det3's 6-22 runs (T at its bound).
- **Not gain.** det3 at 347 V/cm, mesh 425–505 V: D_T flat at 404–496 (mean ≈ 450). σ0 rises 0.3 →
  0.5 mm with gain. Above 505 V the stacks saturate (σ0 0.6–0.9, D_T meaningless).
- **Not the gas at any composition tried** (`iso_fit.py`; Magboltz grids condor 4440949 and 4440956:
  Ar/iso 0.5–8 % × water 0–1 % (to 1.5 % at iso ≤ 3 %) × 7 fields, 384 points; 5 of the dry
  near-pure-argon points had not finished and their columns are dropped). Less isobutane is the one
  knob that makes a gas both slow and hot. On a single-field chamber an iso-poor gas (≈1 % iso,
  0.7 % water) matches both numbers: det4 0.2/0, det7 4.0/2, but det6 still 9.1/0. det3's five fields
  cannot be matched: v alone χ² 24/17, D_T alone 12/17, **both 80/36** (best at iso 1.5 %, water
  0.94 %, which misses v at 243 V/cm). So isobutane depletion is not excluded chamber by chamber,
  but it is excluded where the field scan exists.
- **The beam has it too.** `x_width_beam.py` uses rc_drift's template-free X model, since no beam
  template is on disk; on the bench it reads 5–12 % below the template fit. det4 run_71: D_T 202 / 240 /
  372 (± 20) at 243 / 150 / 92 V/cm, against Magboltz 163 / 193 / 237. That is +118 / +143 / +286 in
  quadrature. At 92 V/cm it is above the thermal floor (234), which no beam-gas composition can exceed
  by much. (§27's "t_a at its bound, less growth than Magboltz" was the Y fit with D_T assumed.) Caveat:
  at 92 V/cm the drift time is 3× the window, and t_a and dd2 trade off.
- **Position** (`x_width_map.py`; det3 pooled long run + gain scan ≤ 505 V, 3574 head-on events; det4
  683). Model-free second moment of the stack, early (0–240 ns) vs late (600–900 ns):
  - The growth is positive in every quintile across and along the strips (det3 0.10–0.41 mm², pooled
    0.26 ± 0.03). Its scatter is ~2σ, with no hot spot, and det4 agrees within its errors.
  - **What depends on position is the early width.** On det3 it is 0.43–0.52 mm² in the middle and
    0.75–0.95 mm² in the outer quintiles, in both directions. The free fit puts this in σ0. At the
    edges the growth reads low, partly because the ±4-strip window truncates a wide cluster.
- **Per depth, not per time.** Across chambers with v from 25 to 37 µm/ns, D_x is about constant per
  unit depth. A constant per unit time would give det6 1.16× det7's D_x; the data give 0.86 ± 0.11
(det6 295 ± 24, det7 345 ± 30), ~2.7σ statistical only. Bench vs beam per time: 4e-4 vs ~2e-5 mm²/ns,
so not one per-time term either.

So the excess is depth-correlated, lands at the anode (no X undershoot, rc_D_x → 0, §17), is
independent of gain and position, and appears on two gases and two particle sources. It is not drift
diffusion at the composition v implies. Two readings remain, and the data in hand cannot separate them:
1. **Magboltz's water model.** Water slows the drift as Magboltz says but cools the transverse energy
   less than it says. Both setups are wet.
2. **A non-gas term linear in depth** that nothing in the current model has.

No new data can be taken (the detectors are gone, 2026-10-16), so a dry-gas measurement is not an option.
The checks left all use existing data or simulation: the bench drift scan's Y view, where Dd is shared with the RC
term; a second Magboltz cross-section set and published Ar/iso/H2O D_T data; other contaminants in the grid; and the
beam X with a proper template. If those cannot separate the two readings, it stays a stated limitation.

**For the kernel:** keep Dd from the X stack (≈ dry), not from v (§28's choice stands). Do not
carry a v-derived water fraction into D_T.


## 30 · The X width excess is mostly ADC saturation (2026-10-16)

> **Supersedes most of §29.** No new data can be taken (detectors gone); everything here is from the existing caches.
> **Its 347 V/cm Magboltz column is wrong (water-branch bug, no gap correction): corrected in §31.**

At operating HV 30–85 % of the head-on bench events reach ~3500+ ADC on the leading strip (det3 long run 42 %,
det4 30 %, det2/6/7 70–85 %). The det3 amplification scan at 347 V/cm showed the stack-fitted v flat at ~39 µm/ns
(GAP-30 frame) from 425 to 485 V and falling once saturation sets in (36.5 at 495 V, 24 at 525 V). `unsat_width.py`
refits every dataset with x_width.py's model on all events and on the events whose X window stays below 3500 ADC
(`results/unsat_width.json`; `XW_MIN_EV=50` for det2/6/7).

| dataset | n all → unsat | v all → unsat | X D_T all → unsat | Magboltz D_T at unsat v |
|---|---|---|---|---|
| det4 g_det4 (208 V/cm) | 683 → 476 | 29.6 → 31.8 | 448 → **339 ± 20** | 332 (0.51 % water) |
| det2 o22_long (347) | 269 → 86 | 35.5 → 37.9 | 479 → **410 ± 48** | 425 (0.25 %) |
| det6 g_det6_long (243) | 353 → 64 | 24.6 → 25.9 | 399 → **279 ± 88** | 272 (0.92 %) |
| det6 g_det6_longer (243) | 120 → 87 | 28.1 → 28.1 | 290 → 271 ± 21 | 286 (0.82 %) |
| det3 700 V (243) | 177 → 128 | 26.6 → 27.3 | 322 → **304 ± 36** | 281 (0.86 %) |
| det3 long run (347) | 1133 → 654 | 36.7 → 38.3 | 435 → **373 ± 10** | 450 (0.14 %) / ~320 (0.85 % branch) |
| det3 sat_det3 (347) | 968 → 539 | 37.5 → 38.4 | 423 → 372 ± 12 | as above |

- **Saturation biases v low and D_T and σ0 high.** With it removed, X D_T agrees with Magboltz at the v-implied
  water on det4, det2 and det6, and on det3 at 243 V/cm. The "excess on every chamber" of §29 was mostly this.
- **det3 at 347 V/cm is ambiguous**: v is non-monotonic in water there (dry 38.8, 0.5 % 40.5, 0.95 % 35.8), so
  v = 38.3 means ~0.1 % or ~0.85 %. On the 0.85 % branch (which det3's 700 V point supports) a residual of
  ~+190 µm/√cm in quadrature remains. The 27.9 mm gap correction is not yet applied to these numbers.
- det7: too few unsaturated events (79 % saturated) to fit; untested.
- Ruled out on the way: a drift-field deficit (dry gas at a scaled or offset field cannot reproduce det3's steep
  v(E) and would put D_T ≈ 490 at every field; χ² ≈ 44/5).

**X/Y isotropy (`xy_isotropy.py`, `results/xy_isotropy.json`, unsaturated events, X and Y < 3500 ADC).** det3 drift
scan 500–1100 V, D_rc shared across fields and profiled, per-field Dd_x, Dd_y. Profile minimum D_rc ≈ 3.6e-4; there
y/x = 0.77–0.91 (0.89 ± 0.07 at 347 V/cm). The arms Dd_y = Dd_x (cost 0.4924), free (0.4894) and Dd_y = Magboltz
(0.4949) are indistinguishable: **inconclusive**, Y's Dd stays degenerate with D_rc even across fields (y/x swings
1.14 → 0.68 between D_rc 2.8 and 4.5e-4).

**Consequence:** the saturated events must be cut (or the clipping modelled) wherever widths or v are fitted from
stacks: bench_pinrc / bench_attach arms, driftscan_fit's 0.95 % composition (drift scan at 490 V, 28–41 % saturated),
and the bundle v. The bench water fractions quoted so far are all from saturated samples.

## 31 · After the saturation cut the excess is gone at 208–243 V/cm but stays at 347 V/cm (2026-10-16, pm)

> **Corrects §30's 347 V/cm rows. Qualified by §32:** det4/det7's agreement at 3500 ADC depends on the cut.
> Same fits (`results/unsat_width.json`), no refit; only the Magboltz comparison changed
> (`unsat_width.py --recompare`). No new data (detectors gone).

**A bug in the comparison.** `bench_water_e.invert` sorted Magboltz v and interpolated water against it. At 347 V/cm
v(water) is not monotonic: 38.8 dry → 41.1 at 0.4 % → 35.1 at 1 %. A v below the dry value has only the wet solution,
but the sort returned a low-water one with a dry-like D_T, so §30's "Magboltz D_T at unsat v" column was wrong at
347 V/cm (det2 0.25 % / 425, really 0.81 % / 328; det3 long run 0.14 % / 450, really 0.78 % / 340). The fit at
208/243 V/cm was unaffected (v monotonic there). `invert` now walks a 0.005 % water grid and returns every branch
(`branches`, `ambiguous`). The same bug put det3's bundle v (36.6) at 0.59 % water in `bench_water_e.json`; it is
0.90 %, the drift-scan composition (§25). det2's bundle v 39.94 is genuinely two-valued (0.12 % or 0.60 %).

**The gap correction, now applied** (`gapcorr` in the JSON): v × gap/30, D_T × √(30/gap), gap from bench_fit.py
(det2 30.6, det3 27.9, det7 27.5 mm; det4, det6 30). For det3 it takes v below dry, so the two-branch question of §30
goes away: only the wet branch remains, at 0.94–0.97 % water, the same as the drift-scan composition fit.

| dataset (unsaturated) | E V/cm | v (own gap) | X D_T | Magboltz at that v | excess in quadrature |
|---|---|---|---|---|---|
| det4 g_det4 | 208 | 31.8 ± 0.5 | 339 ± 20 | 332 (0.51 %) | +68 (0.3 sd) |
| det6 g_det6_long / _longer | 243 | 25.9 / 28.1 | 279 ± 88 / 271 ± 21 | 272 / 286 | +61 / −92 |
| det3 drift scan 700 V | 243 | 25.4 ± 0.7 | 316 ± 37 | 269 (0.94 %) | +164 (1.3 sd) |
| det3 long run | 347 | 35.6 ± 0.3 | 387 ± 10 | 303 (0.97 %) | **+241 (8.2 sd)** |
| det3 sat_det3 (overlaps long run) | 347 | 35.7 ± 0.4 | 386 ± 13 | 304 (0.96 %) | +238 (6.6 sd) |
| det3 hv scan 445 / 465 V | 347 | 35.5 / 37.6 | 404 ± 35 / 388 ± 26 | 302 / 325 | +268 / +213 (2.9 / 2.4 sd) |
| det2 o22_long | 347 | 38.6 ± 1.1 | 406 ± 48 | 337 (0.76 %) | +227 (1.5 sd) |

- **The saturation cut removes the excess at 208 and 243 V/cm** (det4, det6; det3 at 243 is 1.3 sd).
- **At 347 V/cm an excess of ~+230 µm/√cm in quadrature survives** on both chambers measured there: det3 at 8 sd,
  det2 at 1.5 sd. So the excess left is field-dependent, which is what the data now has to explain. 347 V/cm is
  where v(water) turns over, i.e. where the predicted D_T depends most on the composition.
- Sensitivity: putting det3's field at V/2.79 cm (its own gap) instead of V/2.88 moves v(dry) to 35.8–41.0 and leaves
  the det3 347 V/cm unsat v (35.6) just below Magboltz's range at 1 % water. The excess does not shrink.
- Not yet fitted on unsaturated events: det3 drift scan 900 / 1100 V (312 / 382 V/cm; only saturated fits exist, below
  MIN_EV), which would test whether the excess grows with field; det7 (79 % saturated). These need the head-on caches,
  which live on the desktop (`~/x17/cosmic_bench/cloud_basics/xwidth/caches/`):
  `XW_MIN_EV=50 python unsat_width.py --keys sat_det3__drift_scan_resist_490V_drift_900V,sat_det3__drift_scan_resist_490V_drift_1100V,g_det7_long`.
- §29's `x_width.json` "MB wet" column used the same `invert`; its 347 V/cm entries are affected the same way.

## 32 · det3's excess is robust; det4 and det7 widths depend on the event amplitude (2026-10-16, evening, laptop)

> **Qualifies §30 and §31. Qualified by §33:** the estimator has no amplitude bias (toy), so the det4/det7 cut
> dependence below is in the data; the 'charge-max centring self-selects' guess is wrong.
> Same model and stack as x_width.py (charge-centred unless stated), unsaturated events,
> gap-corrected. New laptop caches: det3 drift scan 900 / 1100 V, det3 long run, det4, det7 long_run, longer_run and the
> six det7 HV-scan sub-runs (480–505 V, pooled as `g_det7_long__hvpool_…`; ref-only via headon_cache.py --subrun).
> Results in `results/unsat_width.json` under `<key>[@sat<N>][@ref][@tan<T>]`. No new data.

**det3 at five drift fields.** Excess of D_T² over Magboltz at the v-implied water, in 10⁴ µm²/cm:
243 V/cm 2.7 ± 2.3, 312 V/cm 5.5 ± 3.0 (n 69), 347 V/cm 5.8 ± 0.8, 382 V/cm 7.6 ± 2.9 (n 75). The water implied by
v is 0.89–1.00 % at every field: the gas reading is self-consistent, and the D_T is ~+200–280 µm/√cm in quadrature
above Magboltz at each one (500 V: v below the 1 %-water grid, not compared).

**Cut stability: det3 holds, det4 and det7 do not.** D_T (µm/√cm) against the saturation cut on the window maximum:

| cut (ADC) | 3500 | 3000 | 2500 | 2000 | 1500 | Magboltz |
|---|---|---|---|---|---|---|
| det3 long run (347 V/cm) | 387 ± 10 | 374 ± 16 | 391 ± 14 | 373 ± 21 | 359 ± 48 | ~300 |
| det4 g_det4 (208) | 351 ± 21 | 329 ± 21 | 307 ± 21 | 270 ± 25 | 264 ± 26 | ~337 |
| det7 longer_run (243) | 389 ± 28 | 361 ± 36 | 338 ± 41 | 306 ± 47 | 278 ± 55 | ~300 |
| det7 long_run (243) | 383 ± 34 | 389 ± 45 | 404 ± 41 | 406 ± 80 | 307 ± 113 | ~335 |
| det7 HV pool 480–505 V (243) | 320 ± 40 | 249 ± 47 | 237 ± 58 | 208 ± 66 | 141 ± 72 | ~325 |

v is stable against the cut on every set. With |tan| < 0.01 instead of 0.03 the pattern is the same: det3 359 ± 24
(3500) / 355 ± 32 (2000); det4 323 ± 22 / 239 ± 31. So:

- **det3's excess is not a selection effect**: ~+200 µm/√cm in quadrature at every cut, both tan cuts and all fields
  243–382 V/cm.
- **On det4 and det7 the charge-centred D_T falls steadily as the amplitude cut tightens** and ends 2–3 sd *below*
  Magboltz. Their §30/§31 "agreement with Magboltz" at 3500 ADC is where a falling curve happens to cross it, not a
  measurement. The width estimator has an amplitude-dependent bias of order ±100 µm/√cm on these chambers.
  Saturated events read wide; small ones read narrow. A plausible cause is the stack's centre strip, chosen per event as
  its own charge maximum: on few-electron events that self-selects concentrated charge. Not proven.
- **det7 gas drifts over the night:** v-implied water 0.72 % (longer_run, early) → 0.57 % (HV scan) → 0.56 %
  (long_run, last 24 h), at the same HV.

**Reference-centred stack (CB_CENTRE=ref, opt-in in rc_diffusion.stack): not usable.** Centring on the strip nearest the
M3 track inflates D_T on every chamber (550–620 µm/√cm at |tan| < 0.03; det3 427, det4 560 at |tan| < 0.01). The
reference error grows with depth (inclination, angle resolution, alignment), so it does not stay in σ0 as assumed.

**Where this leaves the X width:** saturation is the dominant bias (it moves det7 D_T 546 → 383 and v 29 → 34). Beyond it,
det3 shows a real, field-independent-within-errors excess of ~+200 µm/√cm. On det4/det7 the estimator's own amplitude
dependence (~±100) is as large as any residual, so "Magboltz-like" and "det3-like" cannot be told apart there. For the
kernel: take Dd per chamber from the unsaturated stack, and carry ±100 µm/√cm as a stated systematic on det4/det6/det7.

## 33 · Closure toy: the estimator has no amplitude bias; det4/det7's trend is in the data (2026-10-16, night, lxplus)

> **Qualifies §32. Followed by §34** (tilt, amplitude terciles, footprint vs amplitude). `xw_sim.py` (condor 4474928, 180 jobs, 100 replicas × 9 arms × 4 datasets, each replica the
> data's own head-on statistics) → `results/xw_sim.json`; raw replicas on EOS `/eos/user/d/dneff/cloud_basics/xw_sim/`.
> Simulation only, no new data.

**The toy.** Synthetic head-on events on the real events' layouts (strip positions, channel noise, reference position and
tan_x of every cache event): Fischle Ar clusters (2.8 /mm), Magboltz D_T and D_L at the v-implied water, Polya gain
(θ = 0.5), the bundle's own X template, 60 ns sampling with random phase, noise with the data's lag-1 autocorrelation
(0.71–0.85) and common-mode fraction (0.10–0.17), per-channel ADC ceilings drawn from the data's saturated maxima. The
per-event gain spread is tuned so that the window-maximum quantiles and saturated fraction match the data (det3, det4:
lognormal; det7: lognormal + 30 % faint events, since its spectrum is bimodal; det7R's faintest decile is still 2× too
bright in the toy). Truth: v and σ0 = the data's unsat fit, D_T = Magboltz in the GAP-30 frame. Same `stack` + `fit_x`,
same cuts.

| D_T (µm/√cm), mean of 100 | truth | all events | 3500 | 2500 | 1500 | data 3500 → 1500 |
|---|---|---|---|---|---|---|
| det3 long run, base | 292 | 319 | 301 | 299 | 297 | 373 → 347 |
| det4, base | 336 | 406 | 353 | 354 | 352 | **351 → 264** |
| det7 longer_run, base | 289 | 328 | 298 | 294 | 290 | **373 → 267** |
| det7 HV pool, base | 317 | 345 | 320 | 320 | 316 | **306 → 135** |

Replica rms 7–24, so one data-sized sample scatters by about the data's bootstrap error. Arms (each against base, unsat cuts):
no clipping, flat gain (no per-event spread), δ-ray extent (Katz-Penfold range), D_T truth ×0.75 / ×1.25: **all flat
in the cut to ≤ 8 µm/√cm**, response linear (bias −1…+24 at every truth).

- **The charge-centred stack + fit_x has no amplitude-dependent bias** for any ingredient modelled here. Closure on
  unsaturated events: D_T +3…+17 high, v 0.0…−1.0 µm/ns, σ0 +0.0…+0.05 mm. The ~100 µm/√cm fall of det4/det7 with the
  cut (§32) is therefore **a property of their data, not of the estimator**, and the §32 guess (charge-max centring
  self-selects on faint events) is wrong. What in the data does it is not identified: not visible in v (flat with the
  cut on all four sets, so not shallow partial tracks), and not in anything the toy contains.
- **Plain clipping underpredicts the data's saturation bias by far:** all events vs unsat, toy +18…+53 D_T and −0.6…−1.9 v;
  data +62 (det3) … +325 (det7H) and v down to 26–28 on det7. Real DREAM saturation does more than clip the sample
  (recovery, baseline or neighbour effects; not modelled). The unsat cut remains necessary; the toy cannot be used to
  correct saturated events.
- **Neighbour copies read as drift diffusion.** The bundle lp crosstalk (c1 ≈ 0.05, c2 = 0.6 c1, τ 130–170 ns, with the
  bundle σ_p0) raises the fitted D_T by +75…+90 µm/√cm, flat in the cut (≈ +240 in quadrature on det3). That is the size
  of det3's excess: the 'share' arm gives det3 378 / 374 at 3500 / 1500 against the data's 373 / 347. But it does not
  single out det3: all bundles have c1 ≈ 0.05, and the same arm overshoots det4 (426 vs 351) and det7H (399 vs 306) while matching det7R (372 vs 373),
  and the arm's σ0 does not match det3's (0.445 vs 0.348). So it is a candidate mechanism with the right size,
  not an explanation.
- **Centring on the true track inflates D_T by +90…+135** (refc arm, perfect reference). Most of the §32
  reference-centred inflation (430–620) is that, not the M3 reference error. Charge-max centring is the right choice.

**Where this leaves the X width:** det3's ~+200 µm/√cm (quadrature) excess is not an estimator artefact (toy closure
+9). det4/det7's cut dependence is in the data and unexplained, so their D_T stays uncertain by ±100 µm/√cm. That is
now a stated limitation of the data, not a bias that can be corrected.

## 34 · det3 is not a tilted field; det4/det7's cut dependence is an amplitude-dependent footprint (2026-10-17, laptop)

> Follows §33. `xw_tan.py` → `results/xw_tan.json`, `xw_amp.py` → `results/xw_amp.json`, `xw_compress.py` →
> `results/xw_compress.json`. Existing caches and the §33 toy (same truth, same tuned amplitudes). No new data.

**1. Field tilt / dished cathode (det3).** det3's charge column ends at 25.7–29.2 mm across the chamber (GAP_STUDY
2026-07-30). A non-flat cathode tilts the drift field. In the X view that is an extra angle δ the M3 reference does not
see, so the head-on width should be smallest at tan_x = −δ. Fit D_T² = a + b (tan + δ)² to D_T in six signed tan_x
bins (−0.06…0.06, unsat):

| | toy (vertical field) | data, b free | data, b = toy |
|---|---|---|---|
| det3 long run | δ −0.002 ± 0.004, floor 298 (truth 292) | δ +0.013 ± 0.010, floor 347 | δ +0.019 ± 0.013, floor 351 |
| det4 | δ +0.005 ± 0.002, floor 341 (truth 336) | δ +0.003 ± 0.007, floor 331 | δ +0.003 ± 0.009, floor 337 |
| det7 longer / HV pool | δ ≈ 0, floor 293 / 311 | no constraint (±60–130 per bin) | δ +0.014 ± 0.030 / +0.005 ± 0.017 |

det3 by position (b = toy): δ = −0.004 ± 0.013 / +0.030 ± 0.026 / +0.038 ± 0.022 in x terciles,
+0.019 ± 0.029 / +0.026 ± 0.018 / +0.019 ± 0.016 in y terciles. That is one common δ ≈ +0.02, with no sign change
across the chamber, and every tercile's floor above the toy's.

- **A tilt does not remove det3's excess:** at the parabola's minimum the width is still ~350 against the toy's 298.
- **The dish is too shallow to cause it.** Raising the floor from 298 to 350 by a spread of local tilts needs
  σ_δ ≈ √((351² − 298²) / b) ≈ 0.05 rad. The GAP_STUDY map's steepest step is ~2 mm per ~80 mm bin (0.026 at the
  cathode, about half that averaged over the drift), and the measured δ does not vary across the chamber. The dish
  is ruled out at the level needed. The common δ ≈ +0.02 (1.3–1.5σ) could equally be a residual alignment angle.

**2. Faint against bright (exclusive terciles of window max, unsat head-on).** D_T (free fit) against the toy:

| | faint | middle | bright |
|---|---|---|---|
| det3 (edges 1813 / 2558) | 351 ± 29 / toy 294 | 380 ± 15 / 296 | 376 ± 18 / 300 |
| det4 (460 / 1293) | 361 ± 30 / 350 | **250 ± 30 / 360** | 380 ± 25 / 355 |
| det7 longer (180 / 1270) | empty events | 265 ± 33 / 288 | **406 ± 44 / 297** |
| det7 HV pool (363 / 1714) | empty events | 196 ± 71 / 331 | 347 ± 30 / 320 |

- det3's excess is the same in every tercile: +57…+84 over the toy, and the model-free growth (late − early second
  moment) is larger too (0.20 / 0.15 against 0.11 mm² in the upper two terciles). It is amplitude-independent and in
  the growth, not in σ0.
- det4/det7 are not 'faint reads narrow'. det7's faint tercile is 70–74 % within 20 mm of the hit-area edge, with a
  median charge of 60–360 ADC: reference tracks at or past the chamber edge, which carry no width. det7's bright
  unsaturated events read wide (+109, 2.5σ on longer_run). det4's middle tercile reads narrow (−110, 3.5σ) with a
  larger early width (m2 0.57 against the toy's 0.37 mm²; free σ0 0.42 against 0.35). With σ0 pinned it reads 315 ± 17.
  So the cumulative cut moves D_T because the classes differ, in both directions.
- Not a residual baseline: the strips' pre-onset ADC is the same in data and toy (−3.6…+0.6 on det4/det3), with no
  positive offset on the outer strips.

**3. The footprint itself depends on amplitude on det4/det7 (model-free).** At the leading strip's peak sample,
r = (W[c−1] + W[c+1]) / 2W[c] against the peak ADC:

| peak ADC | 200–500 | 500–1000 | 1000–1500 | 1500–2000 | 2000–2500 | 2500–3000 | 3000–3500 | ≥ 3500 |
|---|---|---|---|---|---|---|---|---|
| det3 data / toy | – | .43 / .43 | .41 / .42 | .41 / .43 | .39 / .42 | .41 / .41 | .34 / .41 | .50 / .44 |
| det4 data / toy | .35 / .44 | .34 / .45 | .37 / .45 | .37 / .45 | .42 / .44 | .45 / .47 | .44 / .42 | .52 / .48 |
| det7 HV data / toy | .02 / .34 | .11 / .38 | .17 / .36 | .17 / .37 | .35 / .40 | .33 / .39 | .26 / .40 | .51 / .42 |

(errors: det3 ±0.02, det4 ±0.02–0.03, det7 ±0.06–0.13; det7 longer_run the same as HV pool)

- **There is no compression below the ceiling:** r does not rise above the toy approaching 3500, so the bright-wide
  det7 events are not a front-end non-linearity of this kind.
- **On det4 and det7 the cluster core is narrower than a linear response at low amplitude, and reaches the toy only
  at 2000–3000 ADC.** The ±1 neighbours of small signals are suppressed relative to the leading strip. Only the
  chambers that show this have a cut-dependent D_T. det3's r follows the toy at every amplitude below 3000 ADC
  (0.34 against 0.41 at 3000–3500, just under the cut), and det3's D_T is cut-stable. Likely candidates: an amplitude-dependent prompt footprint, or the neighbour-copy kernel not
  scaling linearly. Which one cannot be decided from the head-on X data.
- Above the ceiling every chamber reads r ≈ 0.50–0.52 against the toy's 0.42–0.48: real saturation widens more than
  clipping, as in §33.

**Where this leaves the X width (final with the data we have):**
- **det3:** excess ~+60–80 µm/√cm in the frame (+200–250 in quadrature), present at every field, cut, amplitude
  tercile and position. It is not the estimator (§33), not a field tilt or the dish (above), and not amplitude. It is
  the same size as the bundle's neighbour copies would give (§33 'share' arm), but every chamber has those.
  **Unexplained; a stated limitation.**
- **det4/det7:** the response is non-linear at low amplitude, so a single σ0 + Dd describes them only for a given
  amplitude mix. Their D_T depends on the selection by ±100 µm/√cm. That is a limitation of these chambers' response,
  not an estimator bias, and cannot be corrected without a model of the footprint against amplitude.
- **For the kernel:** take Dd per chamber from the unsaturated stack at the 3500 cut (the production amplitude mix),
  with +100 µm/√cm (det3, unexplained excess) and ±100 µm/√cm (det4/det6/det7, amplitude dependence) as stated
  systematics.

## 35 · Dead strips and gain structure under the stacks; det4's amplitude dependence is its dead/weak strips (2026-10-19, desktop)

> Qualifies §32–34 for det4. `strip_gain_audit.py` → `results/strip_gain_audit.json` (+ per-event flags
> `gainflags_<key>.pkl` next to the head-on caches); `xw_clean.py` → `results/xw_clean.json`. Existing data only.

Until now no cloud_basics fit masked anything: the stacks take the 9 strips around the charge maximum whatever
they are, and the toy assumes uniform gain. The audit, per chamber:

- **Gain map at strip scale across X** (wft events, Q = √(q_x q_y)/path at (x_p0, y_p0), kernel 2 mm across ×
  15 mm along the X strips). Median |Δ ln g| over 3 mm at a head-on event: det2/det3 0.05–0.06, det6/det7
  0.06–0.11, **det4 0.30** (q90 0.69; X-profile p90/p10 = 20 against 1.7–3.4 elsewhere).
- **Strip health** (median peak ratio at ±1 from the cluster maximum, and the ±1 firing rate above 5σ, which
  also works on saturated events): det3 one dead X strip (307), 8 % of head-on events have a dead/weak X strip
  within ±4; det2 none; **det4 11 dead + 25 weak X strips, 55 % of events affected**; det6 14 dead X channels
  in alternating pairs (52–60 even, 208/210, 222/224, 388–392: a connector pattern), 12 % affected; det7 4 dead
  X (74, 76, 127, 269), 7 %. Y channels 7 and 9 are dead on both det6 and det7. det6/det7 show 60–80 X channels
  with noise > 2× the chamber median (not acted on).

**det4.** On det4 the amplitude cut is a position cut: head-on events with window max < 1000 ADC sit at local
gain e^−0.66 with a bad strip in ±4 for 71 %; events above 3500 sit at e^+1.1 (48 %). With the ±2 neighbours
healthy, the model-free r = (W[c−1]+W[c+1])/2W[c] is **0.43–0.48 at every amplitude**, the linear toy's
0.44–0.47; with a dead/weak strip in ±2 it is 0.27–0.37. The §34 'non-linear footprint at low amplitude' is the
dead/weak strips. With σ0 fixed the clean-window D_T is flat in the cut (σ0 = 0.35: 413 / 409 / 396 / 420 at
3500 / 2500 / 2000 / 1500 ADC; all events fall 347 → 307). Free fits on clean events trade σ0 (0.43 → 0.51)
against D_T (354 ± 32 → 280 ± 56), within errors.

**det3** is unaffected: clean events give the same D_T (377 ± 12 against 373 ± 10 at 3500). **det7**: no
unsaturated sample big enough on this machine (its HV-pool caches are on the laptop); its §34 faint tercile was
already edge events. det7's dead strips now flagged; not refitted.

**Consequences.** (1) det4's ±100 µm/√cm systematic of §34 is its dead strips, not its response: use clean
windows (no dead/weak strip in ±4) for det4 widths. (2) Every stack and the toy should mask the flagged
channels; det6/det7's were never masked. (3) The SPS det4 analyses (run_71: headon_stack, beam_xy_pulse,
x_width_beam) do not mask either, and the 65–88 mm beam spot covers several of det4's 12 mm live bands and
their edges; the beam widths/σ0 are open to the same bias (late/early charge ratios over wide sums are not).

## 36 · The drift gap without timing: det3 28.1, det2 31.5, det4 30.7, det7 29.9 mm (2026-10-19, desktop)

> `gap_extent.py` (caches `~/x17/cosmic_bench/cloud_basics/gap/incl_<key>.pkl`, every reference track with
> 0.12 ≤ |tan| ≤ 0.8) → `results/gap_extent.json`; closure `gap_extent_toy.py`. The one gap table is now
> `gaps.py`, used by unsat_width / bench_fit / driftscan_fit / iso_fit / xy_isotropy / make_xwidth_report.

Every earlier gap was a time (T_end × v_geom), with a ±1 mm kernel systematic and a strong amplitude dependence
(det4 +9 mm). Here no time enters: the X strip at x collects the ionisation from depth z = (x − x_mesh)/tan_ref
over pitch/|tan|. Per-strip window integrals, spread over their z interval, stacked against z (dead/weak strips
out of charge and exposure; events with any X sample ≥ 3500 dropped; each event must keep ≥ 85 % of its charge
in z ∈ (−6, 36) and is normalised to its total; |tan| ≥ 0.2). The gap is the distance between the profile's
half-height crossings: any symmetric blur (footprint, diffusion, reference error, strip interval, and a mixture
of them over the tan range) leaves a step's half-height point in place, and an x_mesh offset moves both edges.

**Closure** (toy: Fischle clusters, Polya gain, Magboltz-size diffusion, the bundle template timed to the data's
peak sample, 60 ns sampling, noise, reference smear): G = 30 → 30.0–30.1 at v = 26–38 µm/ns, G = 27.9 → 27.75.
The peak-sample version reads +0.6 to +2.3 mm high (blur-dependent): not used. Normalising events to their
own plateau instead of their total let a few mis-referenced events dominate (a first pass read det3 30.2 —
an artefact, superseded).

| chamber (run) | gap [mm] | +tan / −tan | by \|tan\| 0.2–0.3 / 0.3–0.45 / ≥ 0.45 | timing column (K = 22) |
|---|---|---|---|---|
| det3 (6-27 sat) | **28.10 ± 0.23** (Y 28.87 ± 0.35) | 28.54 / 27.64 | 27.94 / 28.52 / – | 27.89 |
| det2 (6-22 longer) | **31.45 ± 0.38** | 30.20 / 32.56 | 30.83 / 31.36 | 30.62 |
| det4 (6-24) | **30.67 ± 0.41** (Y 30.14 ± 0.50) | 30.87 / 30.55 | 30.45 / 30.48 / 31.38 | 25.55 (amplitude) |
| det7 (6-26 long / longer) | **30.08 ± 1.33 / 29.56 ± 1.51** | – | – | 27.51 |
| det6 | not measurable (≥ 27.6) | | | 27.85 |

- **det3's short gap is real** and now independent of any kernel: 28.1 ± 0.2 against 27.9 from timing. det2 −
  det3 = +3.4 mm, as the timing contrast (+2.8 to +3.4). All measured chambers are within ~2 mm of the
  mechanical 30.
- det2's ±tan halves differ by 2.4 mm, a drift-field/reference tilt of ≈ 0.012 (the mean is first-order free
  of it). det3's halves differ by 0.9 mm (δ ≈ 0.005).
- **det6's deep charge runs past the 32-sample window** (deep strips' last sample at 40–86 % of their peak), so
  its column is a lower limit. det3 6-22 (drift 8.9 µm/ns that night) likewise: excluded.
- Not included: the M3 slope scale (gap ∝ 1/scale). Statistical errors only.
- **Field invariance (det3 drift scan, weak):** the scan sub-runs have ~200 usable tracks each (waveforms pulled to
  `/media/dylan/data/x17/cosmic_bench/det3_scan/`, `gap_extent.py build sat_det3 --subrun … --base …`). 1100 V
  (382 V/cm): 28.04 ± 0.71 (half-height); 900 V: 27.5 ± 1.1 (erf; the half-height finder is unstable at this n);
  700 V unusable (deep charge at the window end, as det6). Consistent with 28.1, not a precise test.

**Effect on the width comparison** (`unsat_width.py --recompare`, no refit): det3 unchanged (+230 at 347 V/cm,
7.8 sd). det7 (27.5 → 29.85) now sits on Magboltz at 0.43 % water (+88, 0.3 sd). det2 (30.6 → 31.45) becomes
two-valued again: 0.09 % (−228) or 0.65 % (+188), both ~1 sd. det4 (30 → 30.67) +62 (0.3 sd).
Water implied by v: det2 0.09/0.65, det3 0.93–0.97, det4 0.46, det7 0.43–0.61 %.

## 37 · Saturation: censoring is adequate per event; the stacks are what cannot use saturated events (2026-10-19)

> `chi2_split.py` (per-event χ² of plane_bench's held-out ref-start fits, summaries `results/chi2_split_*.json`,
> per-event files `~/x17/cosmic_bench/cloud_basics/chi2_split/`), `sat_residual.py` → `results/sat_residual_det3.json`.

- Production (and every wft arm) censors samples ≥ 3550 ADC with a one-sided penalty. On det3, saturated
  events carry 4–6× the χ²/dof of unsaturated ones in every kernel (X: production 186 vs 32; Y 374 vs 68).
- **That is brightness, not clipping.** Residuals (data − model, in units of the event's model peak, rcms25 arm)
  stacked against the sample offset k from the end of the clipped run: the saturated strip reads +0.073 just
  before the clip, +0.059 / +0.019 at k = +1 / +2 and −0.014 at k = +5. Bright *unsaturated* peak strips
  (peak > 1500 ADC), around their peak, read +0.061 / +0.055 / +0.022 and −0.029: the same template mismatch at
  high amplitude. The ±1 neighbours of a saturated strip fit no worse (−0.013…+0.011) than those of bright
  unsaturated events (−0.05…+0.009). No recovery tail, baseline shift or neighbour pick-up shows after the clip.
- So the per-event forward model treats saturation as well as the data allow; what is left is a fractional
  template error that scales with amplitude (MODEL_FRAC in wft.model is the existing handle).
- **The stack studies (§30–34) are a different matter**: a stack cannot censor, so they drop saturated events,
  70–85 % of det2/det6/det7, and on a chamber with gain structure that cut selects low-gain regions (§35). The
  way to use every event is a per-event censored width fit (the wft forward model, with σ0/Dd as the fitted
  hypers) rather than stacks. Not done.
- §33's 'data saturation bias 2–6× the toy's' is a property of the stacks (clipped samples averaged in), not of
  the electronics' recovery: not resolved further.

## 38 · The RC model's X χ² deficit is its footprint width (2026-10-19)

§12/§17: the physical kernel's X χ²/dof is worse than production on every chamber (det3 58 → 79) while
resolution is neutral. `chi2_split.py`, det3, 1000 held-out events, production / rcm / variants:

- The deficit is uniform: ×1.2–1.4 on unsaturated, saturated, head-on, inclined and faint events alike. Not
  saturation.
- **Residual by strip offset (head-on, unsaturated)**: rcm under-predicts the centre strip (+4.1 σ per sample)
  and over-predicts ±1 (−3.3 / −4.4): the model's footprint is too wide. Pseudo-Voigt tails (§23's rcmlor,
  rcmlor2) make χ² worse (80 → 85 / 89).
- **Narrowing σ0_x** (rcm keeps everything else): χ²/dof 80 (σ0_x 0.41) → 68 (0.35) → 60 (0.30) → 56 (0.25)
  → 54 (0.20), against production 57; on saturated events 157 at 0.25 against production's 186. With 0.25 the
  core residuals drop to +1.4 / −1.2 / −1.4; what is left is an **asymmetry** (−2: +1.2, +2: −2.0; production
  shows the same sign), the internal X sharing asymmetry production carries as `c1_asym`. A symmetric footprint
  cannot follow it.
- **Resolution** (`run_bench_s0.sh` → `~/x17/cosmic_bench/cloud_basics/bench_s0/`, 2000 held-out, scale-corrected,
  blind start, vs production): rcm X +0.005 ± 0.019 (head-on +0.010 ± 0.028); σ0_x 0.30 / 0.25 / 0.20:
  +0.015 / +0.013 / +0.013 ± 0.019 overall, head-on +0.045 / +0.041 / +0.071 ± 0.03. Slope 0.963 (rcm) →
  0.974–0.978, v-flatness spread 0.021 → 0.012–0.017. Y unchanged (−0.090 ± 0.019, head-on −0.136).
- So on det3 a narrower X footprint fixes the fit quality and the angle scale at no cost overall and a small
  (1.5–2.4σ) cost head-on. The per-event fit wants σ0_x ≈ 0.20–0.25 where the head-on stacks measure
  0.35–0.43: one of the two is reading the asymmetry/neighbour structure as width. det2/det4/det6/det7 benches
  are running (same script).

**§35 addendum — masking the flagged strips in the stacks** (`rc_diffusion.annotate_bad`, `unsat_width.py --mask`,
keys `…@mask`): the flagged strips' samples are NaN in the stack (cells average over events that have them).
det3 unchanged (386 ± 10, +235 at 347 V/cm). det6_long 268 ± 99 (unchanged within errors). **det4 widens**,
347 → 378 ± 23 at 3500 ADC, and is still cut-dependent: 378 / 334 / 306 / 304 at 3500 / 2500 / 2000 / 1500
(unmasked 351 / 307 / 270 / 264, GAP-30 frame). Masking removes only the flagged strips; the gain falls over
the band edges around them, so det4 widths need the clean-window selection (no flagged strip in ±4), which with
σ0 fixed was flat. det7_long produced no masked fit (too few unsaturated events after masking).
