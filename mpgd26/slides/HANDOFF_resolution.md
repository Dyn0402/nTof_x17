# Slide 13 resolution tile — is "0.6–0.7 mm" reference-limited?

**Answer: no.** The caveat in the HTML comment above slide 13 (`index.html:378–389`)
and in `NOTES.md:88–93` rests on a wrong premise. It says the M3 reference "only has
~500 µm resolution", which is the resolution of **one M3 plane**. The quantity that
matters is the **four-plane fit's pointing error interpolated to the DUT plane**, and
that was measured on this very run: **0.21 mm (X) / 0.24 mm (Y)**. It is ~11 % of the
residual *variance*, not a floor. The quadrature subtraction is therefore **stable and
small**: 0.61/0.73 mm → **0.57/0.69 mm**.

The real problem with the tile is a different one, and it is bigger than the
deconvolution: the figure and the number are **hits-basis**, from a reco generation
superseded on 2026-07-25, on a run that was never reprocessed afterwards. See §5.

---

## 1. The measured widths, and where they come from

The tile (`index.html:407`) and the figure next to it
(`assets/img/spatial_residuals.png` ← `mx_june_cosmic_qa/engineer_package/figures/10-det3A-spatial-residuals.pdf`,
committed `1f3fa65`, 2026-07-16) carry these numbers:

| quantity | value | source |
|---|---|---|
| det3 X residual, Gaussian core fit | **0.61 ± 0.00 mm** | annotation inside the tracked PDF itself (panel 3, top row) — reproduced by `pdftocairo`; also `engineer_package/figures/FIGURE_GUIDE.md:137` |
| det3 Y residual, Gaussian core fit | **0.73 ± 0.00 mm** | same |
| run | `mx17_det3_p2_det1_overnight_6-27-26 / long_run_p2_det1_sanity_check`, det3 top slot, FEU 7/8 (`qa_config.py:233`, key `g_det3_wknd`) | `FIGURE_GUIDE.md:141` |
| recipe | M3 χ² < 1.0 & NClus = 4 | `FIGURE_GUIDE.md:136` |
| estimator | iterative Gaussian fit in a ±2.5 σ window, `fit_residual_peak()` / `plot_residuals()` | `cosmic_bench_analysis/cosmic_micro_tpc_analysis.py:2521`, `:2656`, saved at `:2778` |
| basis | **hits** (`combined_hits` production earliest-strip position) | `RECONSTRUCTION_BASIS.md:104` migration table |

I re-derived per-axis widths myself rather than trusting prose, through the same
accounting `mx_june_wft/02_efficiency.py` uses (same M3 rays, χ²<1 & NClus=4, spark
veto, 0.5–99.5 % active box, `|r| < 15 mm`), on **det3 `sat_det3`** — the one det3 run
that *does* carry the post-fix reco — for both bases:

| basis (sat_det3, z_mean = 714 mm) | core σ X / Y | σ68 X / Y | 3σ-clipped rstd X / Y | radial rstd \|r\| |
|---|---|---|---|---|
| hits chain | 0.558 / 0.605 mm | 0.639 / 0.685 | 0.644 / 0.683 | **0.4477** |
| waveform-first (`wft`) | 0.536 / 0.533 mm | 0.633 / 0.620 | 0.639 / 0.636 | **0.4597** |

Validation of that pipeline: the radial column reproduces the published
`mx_june_wft/state/det3/efficiency__efficiency_breakdown*.json` `core_sigma_mm` to all
printed digits (0.4597 wft / 0.4477 hits), and my per-axis rstd for `wft` (0.639/0.636)
reproduces the independently quoted 0.63/0.61 mm in `mx_june_wft/RECO_BENCH_2026-07-29.md:80`.
Script: `mx_june_cosmic_qa/m3_self_resolution/peraxis_deconvolve.py` (written for this
note; run it with the repo venv, no arguments).

**Estimator spread is ±0.08 mm** on the same events — core-Gaussian < σ68 ≈ rstd. Any
quoted number has to name its estimator.

## 2. The reference contribution

Measured, on the same detector and (for `sat_det3`) literally the same run:
`mx_june_cosmic_qa/m3_self_resolution/` — `M3_SELF_RESOLUTION.md`, `results.json`,
`uncertainty.json`.

| quantity | value | source |
|---|---|---|
| M3 = 8 MGv2 planes in 4 stations at z = 24, 144, 1185, 1302 mm → each coordinate is an independent 4-point line fit; DUT is **interpolated** at z ≈ 714 | — | `M3_SELF_RESOLUTION.md:20–31` |
| per-plane intrinsic σ (Gaussian core) | X 0.409–0.415 mm (uniform); Y 0.445–0.511 mm | `M3_SELF_RESOLUTION.md:87–96`; `results.json` |
| **reference pointing at the DUT plane, P(702)** | **X 0.206 / Y 0.242 mm** | `results.json` → `pointing`; `M3_SELF_RESOLUTION.md:125–128` |
| **P at the run's own fitted z_mean = 714 mm** | **X 0.2066 / Y 0.2434 mm** | recomputed from the measured σ_k with `analyze.py:73` `hat_coeffs`, by `peraxis_deconvolve.py` |
| statistical error on P | ±0.002 mm (2000-replica bootstrap) | `uncertainty.json` → `pointing_um` |
| multiple scattering | enters as residual **tails**, not core broadening; core P is MS-free by construction | `M3_SELF_RESOLUTION.md:105–119` |

**So the "~500 µm reference" in the slide comment is the per-plane number.** The
four-plane fit interpolating mid-gap does 2× better than any single plane — z̄ = 664 mm
is ~50 mm from the DUT slot, which is where pointing is *best*
(`M3_SELF_RESOLUTION.md:123`, and the crossover table `:141–150`: inside the whole M3
volume the reference out-points the DUT).

### Two errors in the repo to be aware of (do not re-quote them)

1. **"M3 pointing 0.40 mm"** in `mx_june_wft/ANALYSIS_STATE_2026-07-31.md:340` (row S14),
   `mx_june_wft/RECO_BENCH_2026-07-29.md:80` and `RECONSTRUCTION_BASIS.md:44`
   ("the ~0.4 mm M3 pointing floor") is **not** the pointing. 0.40 mm is the
   *deconvolved DUT* value from the same study; the pointing is 0.21–0.24 mm. Those
   three lines inflate the reference term by ~2× and are the origin of the
   "reference-limited" language.
2. **The published σ_DUT ≈ 0.40 mm is a units mismatch.** `analyze.py:38` feeds
   `DUT_RESID_CORE = 0.47 mm` into the subtraction, but that 0.47 is the **radial**
   width — `rstd` of `|r| = hypot(dx, dy)` (`det3_recofar_analysis/m3_cut_tradeoff.py:201, :264`;
   the same statistic as `mx_june_wft/02_efficiency.py:171`) — while P is **per axis**.
   Radial 0.45–0.47 mm and per-axis 0.54–0.68 mm are different numbers for the same
   data (see the table in §1). So `σ_DUT ≈ 0.40 mm` is not a per-axis detector
   resolution and must not be compared with the slide's 0.6–0.7 mm.

## 3. Deconvolution, done per axis

σ_DUT = √(σ_meas² − P²), per axis, P at z = 714 mm:

| input | σ_meas X / Y | reference share of variance | **σ_DUT X / Y** |
|---|---|---|---|
| **the slide's own figure** (hits, `g_det3_wknd`, core fit) | 0.61 / 0.73 | 11.5 % / 11.1 % | **0.57 / 0.69 mm** |
| hits, `sat_det3`, core fit | 0.558 / 0.605 | 13.7 % / 16.2 % | 0.52 / 0.55 mm |
| **wft, `sat_det3`, core fit** | 0.536 / 0.533 | 14.8 % / 20.9 % | **0.50 / 0.47 mm** |
| wft, `sat_det3`, σ68 | 0.633 / 0.620 | 10.7 % / 15.4 % | 0.60 / 0.57 mm |
| wft, `sat_det3`, rstd | 0.639 / 0.636 | 10.4 % / 14.7 % | 0.61 / 0.59 mm |

**Is the subtraction stable? Yes.** The reference term is 10–21 % of the variance, so
the correction shrinks the width by only 4–11 %, and P itself is known to ±1 % stat
(±10 % if you want to be generous about systematics — unweighted producer fit, one run,
core-only σ_k). Propagating ±10 % on P moves σ_DUT by **< 0.01 mm**. This is the
comfortable regime for a quadrature subtraction, not the pathological one.

**What is *not* stable is the input width.** Choice of estimator moves it ±0.08 mm and
reco generation moves it much more than that (§5). That is the honest uncertainty to
carry, and it is why I would not print a third significant figure.

Angular resolution is untouched by any of this: σ_slope = σ_st/1166 mm ≈ 0.02–0.04°
against 1.7° (`paper_plans/PLAN_37_m3_pointing_deconvolution.md:95–97`).

## 4. Which basis the number came from

The tile's spatial number is **hits-basis**, which `CLAUDE.md` / `RECONSTRUCTION_BASIS.md`
forbid as a geometry basis. Position at the mesh is the one place where that rule is
soft — `RECONSTRUCTION_BASIS.md:84–87` says positions at the mesh are "much less
affected", and the two chains agree here (per-axis core 0.558/0.605 hits vs 0.536/0.533
wft, radial 0.448 vs 0.460). So the number is not *wrong*, but the deck's own caption
advertises the waveform-first fit for angles while the spatial tile is hits-era — and
the wft basis is slightly **better** in Y, so switching costs nothing.

## 5. The bigger problem: the figure is from a superseded reco

- The tracked figure and its 0.61/0.73 mm are the 2026-07-16 state of the chain.
- The same run's `residuals.png` on disk today
  (`~/x17/cosmic_bench/Analysis/mx17_det3_p2_det1_overnight_6-27-26/long_run_p2_det1_sanity_check/mx17_3/alignment_tpc_veto50/residuals.png`,
  mtime 2026-07-25 02:12) reads **σ_X = 0.62, σ_Y = 0.89 mm**, and my own per-axis pass
  on its cache gives core 0.588 / 0.827 mm, radial 0.667 mm.
- That degradation is the known **2026-07-25 significance-floor regression**
  (`mx_june_cosmic_qa/DET3_RECO_FIX_2026-07-25.md`). The fix was re-run for `sat_det3`
  only (`DET3_FULL_CHAIN_2026-07-25.md`); `g_det3_wknd` has no `*_prefloor_*` backup
  dir, i.e. **it was never reprocessed**. Its on-disk numbers are the regressed ones —
  documented as 93.0 % / 0.473 mm radial, now 0.667 mm.
- Consequence: **the run behind the slide's figure has no currently-valid output.**
  Quote `sat_det3` instead. Its post-fix hits σ68 is 0.63/0.68 mm
  (`DET3_FULL_CHAIN_2026-07-25.md:63`, and the log
  `~/x17/cosmic_bench/Analysis/_grand_logs/det3_full_20260725_151347.log:308`), and its
  wft numbers are in §1. (Note: the `position_summary.csv` on disk for `sat_det3` is
  *not* those numbers — it was overwritten at 16:01 by the reverted charge-sharing
  experiment described in `DET3_FULL_CHAIN_2026-07-25.md:76–110`. Trust the log.)
- One nuance for a possible question from the floor: resolution is angle-dependent,
  0.53 → 0.94 mm from flat to steep tracks, as diffusion predicts
  (`RECO_BENCH_2026-07-29.md:78–83`).

---

## 6. Two options for the tile — exact markup

Current line, for reference (`index.html:407`, **do not edit — this file is owned by
another agent**):

```html
<div class="stat"><div class="num">0.6&ndash;0.7&nbsp;mm</div><div class="lbl">spatial core resolution</div></div>
```

### Option (a) — deconvolved number *(recommended)*

Arithmetic on the numbers the displayed figure itself is annotated with, so the tile and
the figure cannot contradict each other; one clause of caveat.

```html
<div class="stat"><div class="num">0.57 / 0.69&nbsp;mm</div><div class="lbl">spatial resolution X / Y &mdash; M3 reference pointing (0.21 / 0.24&nbsp;mm at the DUT plane) deconvolved</div></div>
```

If you prefer one number over a pair, this is the same statement rounded:

```html
<div class="stat"><div class="num">0.6&ndash;0.7&nbsp;mm</div><div class="lbl">spatial resolution, reference-deconvolved (M3 points to the DUT plane to 0.22&nbsp;mm)</div></div>
```

### Option (b) — keep the measured width, flag the convolution

Use this only if you would rather not do arithmetic on stage.

```html
<div class="stat"><div class="num">0.6&ndash;0.7&nbsp;mm</div><div class="lbl">spatial core resolution &mdash; det3 &oplus; the M3 reference, whose 0.22&nbsp;mm pointing is &lt;12% of the variance</div></div>
```

### Recommendation

**Ship (a), the two-number form.** Reasons:

1. It is the defensible statement: the reference term is measured on the same run, it is
   small, and the subtraction is stable to < 0.01 mm.
2. It costs no slide space — same tile, one clause in the label — and it pre-empts the
   obvious question ("what's your telescope resolution?") instead of inviting it.
3. Option (b) leaves a convolved number labelled as a resolution, which is exactly the
   thing the comment in the deck was worried about.

Do **not** describe the measurement as reference-limited on stage; it is not
(reference ≈ 11 % of variance). If asked how well the telescope points: *0.22 mm at the
DUT plane, from a four-plane fit whose planes are individually 0.41–0.51 mm — the DUT
sits near the telescope's centroid, where interpolation is best.*

### One thing worth doing before 3 September, if there is time

Regenerate the residual figure from **`sat_det3` on the `wft` basis**. It is the only
det3 run with a valid post-fix reco, it matches the basis the rest of the deck
advertises, and the resulting deconvolved number is **≈ 0.50 / 0.47 mm** — better than
what the slide currently claims. Path:
`mx_june_wft/04_maps.py` / `02_efficiency.py` already produce the inputs;
`peraxis_deconvolve.py` prints the widths.

---

*Written 2026-08-10. Numbers in §1 and §3 recomputed from data; everything else cited
to file:line. Nothing in `index.html` was modified.*
