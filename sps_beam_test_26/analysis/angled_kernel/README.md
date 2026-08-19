> **SUPERSEDED in part, 2026-08-18.** The deconvolution section
> (`deconv_kernel.py`) ran on the pre-made `d4_kernel_fit_raw*.npz`, which are
> dated 08-03 and predate the clean selection — 12.4 % of peak sits before the
> pulse in them, against 1.7 % clean. Its conclusion (the shipped kernel form
> cannot make the measured tail) stands; its numbers do not, and the tail τ
> turns out to be window-dependent rather than a constant. Redone properly,
> without any deconvolution, in
> `sps_beam_test_26/analysis/sharing_kernel/`. The angled/flat comparison and
> the drift-invariance work below are unaffected.

# angled_kernel — does det4's spreading kernel survive a 25.64° rotation?

Answers three questions asked on 2026-08-18, using run_63's own flat / 25.64°
A/B (both mounts, 90 minutes apart, either side of one H4 TAX access — same run,
same gas, resist held at 769.8 V, ZS 4σ throughout):

1. **Do the flat-mount calibration numbers hold at an angle?** Yes, to 5–12 %.
2. **Does the angled run give a second drift lever?** Yes — 142 / 108 / 75 V/cm
   at fixed resist, though only `d425` has an untruncated ladder.
3. **Is the measured kernel really a transverse-diffusion convolution?** No.
   The lateral width is field-independent where diffusion would grow as E^(-1/2).

## The structural fact everything rests on

**The X view is at normal incidence in BOTH mounts.** det4 is rolled +90.2°, so
the tilt lands entirely in the Y view. Verified from the data, not assumed — the
signed median of (peak time vs strip position) per event:

| arm | X view | Y view |
|---|---|---|
| flat700 | −4.4 ns/mm | −0.4 ns/mm |
| 25.64° d425 | −0.5 ns/mm | **−209 ns/mm** |

So X isolates the kernel from geometry, and Y measures the geometry.
−209 ns/mm ⇒ v ≈ 10.0 µm/ns at 142 V/cm, right for this wet CF₄ mix.

## Headline numbers

| | flat700 (233 V/cm) | 25.64° d425 (142 V/cm) |
|---|---|---|
| X ±1 peak / centre | 0.2884 | 0.2733 |
| X ±1 delay | +49 ns | +49 ns |
| Y ±2/±1 peak | 0.062 | **0.298** |
| Y ±1 delay asymmetry | −8 ns (symmetric = RC) | **−104 ns (antisymmetric = ladder)** |
| lateral rms about the telescope | 0.759 mm | 0.846 mm |

Diffusion bound: < 0.16 mm rms, under 5 % of σ².

## Traps

- **Mount and field are confounded** — one flat arm, and it is the only one at
  233 V/cm. The mount test is an extrapolation of the rotated arms' field trend.
- **v-from-ladder is a truncation diagnostic.** It *rises* with falling field
  (10.0 → 11.3 → 33.3 µm/ns) because the ladder overruns the 3.84 µs window.
  d325 is partly truncated, d225 badly. Only d425 carries weight.
- **ZS 4σ everywhere.** Absolute values are not comparable to the RAW arms
  (TWOGAS_HEADON F3: ~6 % low on peak, ~18 % low on area). Read differences only.
- **Median kills ±2.** With ~33 % detection and absent strips entered as zeros,
  the per-event median is exactly zero. Stacks are 20 %-trimmed means, matching
  the campaign convention.
- **Pile-up.** 7.6 % of events carry a second separated cluster; a charge-weighted
  centroid lands between them and blows the telescope residual up to 5–9 mm.
  Single-cluster + leading strip gives 0.57 mm MAD and slope 1.001 vs pX.

## Added 2026-08-18 (second round)

**`deconv_kernel.py` — the sharing kernel MEASURED, not assumed.** At normal
incidence every strip sees the same column, so `W_d(f)/W_0(f) = G_d(f)` with the
column *and* the electronics cancelling exactly. `g_d(t)` is then the sharing
kernel with no template, no drift model, no v_drift, no functional form. Only
works head-on — at an angle each strip sees a different depth slice.

Measured (RAW run_71, Y view, symmetrised, two fields):

| d | area | peak | centroid | tail τ |
|---|---|---|---|---|
| ±1 | 0.49–0.53 | +0 ns | +263…+320 ns | 250–400 ns |
| ±2 | 0.24–0.27 | +60…+120 ns | +733 ns | long (plateau) |

**wft assumes `g_1(t) = c1·Gauss(t − 145 ns, σ=12 ns)` — a symmetric bump with
no prompt term and no tail. The truth is a prompt spike plus a one-sided ~280 ns
exponential.** Different objects. This is how the kernels should be obtained.
(The *impulse response* is already measured, by `wft.calibrate.measure_templates`;
it is only the *sharing* kernel that is assumed.)

Systematics: area ±10 % over λ = 0.005–0.10; tail τ quote as 250–400 ns, not to
three figures; acceptance falls to ~0.37 at the window edges; **X does not
deconvolve cleanly** (+d/−d disagree 30 %, 18 % of weight at t<0) — Y only.

## Two corrections to the first round

- **±2 flat-vs-angled is 2.8×, not 4.8×.** Both arms are ZS 4σ but their ±2
  *detection* differs (0.87 vs 0.98), so censoring was unmatched for ±2. Re-gate
  the flat arm to 900–3000 ADC → detection 0.996, and ±2/±1 is 0.218 vs 0.618.
  The flat matched value agrees with the RAW 0.249 to 12 %, the ZS–RAW offset
  TWOGAS predicts.
- **The cosmic bench is RAW, not zero-suppressed** (37 % of strips peak below
  20 ADC against a ~7 ADC noise σ; samples reach −411). The bench drift trend is
  **window truncation**: last-sample/peak is +0.130 at 300 V and −0.02 at 700 V+,
  and the window is 1.92 µs against a 1.87 µs column at 300 V. On the untruncated
  points at fixed central amplitude the Y ±1 ratio moves only +5 % over 1.57× in
  field. X still walks +33 % — open.

## Run

    ../../../.venv/bin/python measure.py        # -> results.json
    ../../../.venv/bin/python make_figures.py   # -> figures/
    ../../../.venv/bin/python deconv_kernel.py  # -> deconv_kernel.json + figure
    ../../../.venv/bin/python make_report.py    # -> report.html
