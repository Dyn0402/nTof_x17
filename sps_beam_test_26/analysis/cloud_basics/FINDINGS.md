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

## 5 · The beam gas lost the deep charge

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

## 8 · The bench also loses deep charge, and Y's template is contaminated

The X template (bundle `tmpl_x`) is clean electronics: unipolar, −2 % undershoot.
A uniform head-on track through it would give a flat-topped pulse to ~600 ns;
every bench chamber's pulse instead sags steadily (det4: 0.93 → 0.52 between 200
and 600 ns where a box gives 0.90 → 0.99). The arrival current falls with depth:
**attachment on the bench too**, with lengths 13–35 mm (beam: a few mm).

The Y template has a −7…−9 % undershoot X lacks; DREAM is the same for both views,
so that undershoot is charge leaving along the resistive strip. Y's "impulse
response" in every bundle carries part of the RC spreading.

## 9 · Bench fit with the physics written out (`bench_attach.py`)

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

_(sections appended as results land)_
