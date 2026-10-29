# Chamber bow: one correction per chamber that holds across its datasets (2026-10-29)

## FINAL (2026-10-29 evening) — read this first

**Rule (Dylan):** one bow parameter set per detector, extracted on its golden row and applied to every other dataset of
that detector with no feedback into the bow. No slot- or run-dependent bow (the bench slot is not a physical change).

**Result: it works on every dataset of every chamber, through the production code.** `bow_finalize.py` builds each
chamber's block from its golden table, installs it with the pass's own W0KW step (`rerun/w12/set_w0kw.py --bow-json`)
and maps every target event with `wft.reco.bow_UW` / `tan_bow` (what `apply_bow` does in the reconstruction). On the
target only the rigid tilt w0 and the angle scale kw are re-measured (2 numbers per view, which every row gets with or
without a bow); the bow shape never sees the target.
- **Angles:** better on all 24 non-golden view-datasets (12 datasets × 2 views), none worse. det3: X −0.07…−0.17°,
  Y −0.07…−0.21° (6-28 night −0.09/−0.14, 500 V −0.13/−0.07, 700 V −0.07/−0.19, 6-25 −0.17/−0.21). The far
  fraction does not rise. det4 X, which the table-level test said does not transfer, gains −0.10 on 6-23 here.
- **Positions:** the golden per-view mesh height (p0 − tan·h_view, alignment terms refit per row as the row's own
  waveform alignment does) improves 33/34 view-datasets: det2/det3 2–5 %, det6 8–13 %, det7 3–5 %
  (exception det3 500 V Y +0.007 mm on 679 events). The reconstruction stored the cross-view MEAN height, which made
  positions worse; it now stores each view's own (`x_bow_h`, `y_bow_h`; `wft.reco.bow_h(..., plane)`).
- **Why w0/kw must stay per row (the "strict" column):** with the golden w0/kw unchanged, rows at another drift field
  are off by +1…+3.6° (kw carries v, which changes with the field: 500 V row v ≈ 0.44× golden), and det4 6-23 / det6
  "longer" X by +0.4 (a rigid-tilt change between runs). That is the angle scale, not the bow.
- **Stability:** a bootstrap of each golden row moves the predicted tan offset by 0.03–0.07° rms, against a 1.7–4.4°
  span across the chamber; the shape is measured to ~2 % of itself.

| chamber | dataset | n X / Y | no bow s68 X / Y [°] | golden bow, row w0/kw: Δ X / Y [°] | golden bow + golden w0/kw (strict): Δ X / Y | position s68 X / Y [mm]: no h → golden h |
|---|---|---|---|---|---|---|
| det2 | det2 (golden, in-sample) — 6-22 longer, 495 V mesh, 1000 V drift | 3325 / 3432 | 1.13 / 1.41 | **-0.083 ± 0.018 / -0.315 ± 0.030** | -0.08 / -0.32 | 0.541 → 0.527 / 0.536 → 0.524 |
| det2 | det2_o22long — 6-22 long (same night, other subrun) | 10808 / 11186 | 1.11 / 1.36 | **-0.093 ± 0.013 / -0.269 ± 0.016** | -0.10 / -0.27 | 0.583 → 0.568 / 0.568 → 0.548 |
| det3 | det3 (golden, in-sample) — 6-27 sat long, 490 / 1000 V | 6615 / 6646 | 1.07 / 1.02 | **-0.101 ± 0.010 / -0.135 ± 0.011** | -0.10 / -0.13 | 0.598 → 0.580 / 0.569 → 0.562 |
| det3 | det3_day625 — 6-25 day, BOTTOM slot, 495 / 500 V | 7458 / 7485 | 1.16 / 1.35 | **-0.166 ± 0.010 / -0.210 ± 0.012** | +2.08 / +1.38 | 0.489 → 0.464 / 0.459 → 0.455 |
| det3 | det3_d700 — 6-27 drift scan 700 V (small) | 727 / 732 | 1.09 / 1.08 | **-0.069 ± 0.032 / -0.186 ± 0.042** | +1.91 / +1.06 | 0.575 → 0.556 / 0.548 → 0.523 |
| det3 | det3_d500 — 6-27 drift scan 500 V (small) | 650 / 679 | 1.16 / 1.11 | **-0.129 ± 0.034 / -0.073 ± 0.035** | +3.64 / +2.69 | 0.609 → 0.585 / 0.546 → 0.553 |
| det3 | det3_p2 — 6-28 night (same mount) | 11080 / 11195 | 1.07 / 1.04 | **-0.091 ± 0.006 / -0.135 ± 0.009** | -0.08 / -0.12 | 0.606 → 0.588 / 0.560 → 0.552 |
| det4 | det4 (golden, in-sample) — 6-24 day long, 495 / 600 V | 5042 / 5150 | 1.71 / 1.74 | **-0.119 ± 0.022 / -0.378 ± 0.029** | -0.12 / -0.38 | 0.742 → 0.718 / 0.610 → 0.599 |
| det4 | det4_o23 — 6-23 night | 3534 / 3682 | 1.82 / 1.93 | **-0.096 ± 0.027 / -0.487 ± 0.034** | +0.45 / +0.09 | 0.730 → 0.716 / 0.598 → 0.575 |
| det6 | det6 (golden, in-sample) — 6-26 night long, 495 / 700 V | 6667 / 6855 | 1.83 / 1.94 | **-0.551 ± 0.019 / -0.537 ± 0.021** | -0.55 / -0.54 | 0.538 → 0.491 / 0.527 → 0.499 |
| det6 | det6_longer — 6-26 longer (same night) | 3753 / 3832 | 1.83 / 2.15 | **-0.385 ± 0.028 / -0.479 ± 0.029** | +0.38 / -0.14 | 0.529 → 0.475 / 0.530 → 0.512 |
| det6 | det6_hv505 — mesh 505 V | 556 / 587 | 1.92 / 1.84 | **-0.507 ± 0.069 / -0.463 ± 0.072** | -0.42 / -0.52 | 0.552 → 0.481 / 0.593 → 0.536 |
| det6 | det6_hv530 — mesh 530 V | 204 / 231 | 2.05 / 2.30 | **-0.558 ± 0.135 / -0.450 ± 0.176** | -0.22 / -0.33 | 0.613 → 0.566 / 0.634 → 0.586 |
| det7 | det7 (golden, in-sample) — 6-26 night long, 495 / 700 V | 5315 / 6000 | 1.63 / 1.59 | **-0.084 ± 0.017 / -0.306 ± 0.017** | -0.08 / -0.31 | 0.788 → 0.760 / 0.690 → 0.663 |
| det7 | det7_longer — 6-26 longer (same night) | 2893 / 3301 | 1.61 / 1.57 | **-0.084 ± 0.026 / -0.317 ± 0.026** | -0.01 / -0.29 | 0.804 → 0.775 / 0.700 → 0.670 |
| det7 | det7_hv480 — mesh 480 V | 480 / 528 | 1.59 / 1.44 | **-0.138 ± 0.063 / -0.352 ± 0.060** | -0.15 / -0.35 | 0.794 → 0.771 / 0.698 → 0.674 |
| det7 | det7_hv505 — mesh 505 V | 417 / 457 | 1.75 / 1.63 | **-0.072 ± 0.078 / -0.288 ± 0.064** | -0.09 / -0.25 | 0.843 → 0.804 / 0.740 → 0.722 |

Golden-shape stability (30 bootstrap resamples of the golden row): the tan offset it predicts across ±160 mm spans det2 X 2.54° (rms 0.060°), Y 3.98° (0.061°); det3 X 1.71° (rms 0.038°), Y 1.94° (0.034°); det4 X 1.69° (rms 0.058°), Y 4.00° (0.056°); det6 X 3.82° (rms 0.045°), Y 4.41° (0.039°); det7 X 2.07° (rms 0.074°), Y 2.80° (0.049°).


**For the pass (wired as specified; E3 still to write):** `rerun/calib/bow_node.py` (the BOW node) builds the block
from the golden row's CAL3 table, alignment, M3 rays and bundle on EOS; verified to reproduce the study block exactly
on det3. Every row inherits it; vrefit's W0KW re-measures w0/kw with `--bow-json`. After the pass reco,
`bow_finalize.py --manifest <pass datasets>` is the **transfer gate**: the chamber block must not worsen any row's
angle or position s68 beyond its error, rows ≥ 2000 events. No per-mount block.

**Changes on main:** tan-form drift tilt `a_x/a_y` (`wft.reco.tan_bow`, `set_w0kw.py`, `set_bow.py`); per-view mesh
height in the reconstruction; `set_bow.block_from_table`, `fbcommon.load_path`; `rerun/calib/bow_node.py`;
`rerun/calib/bow/bow_finalize.py`. Old bundles (no `a_`, no `h_<view>`) are bit-identical in angle.

**Limits:** FB's tables are finalize-era (older σ0 handling, promoted bundles); the gate re-runs on the pass tables.
Golden numbers are in-sample (2-fold values in the table below: det3 −0.08/−0.13). Small rows (200–700 events) carry
±0.03–0.18°. det7 X still includes the faulty half (x < 200 mm) in these tables.

---

## Earlier today (transfer study, superseded where it disagrees)

**Question (Dylan):** the bow correction was planned; make sure it is fixed per detector and carries between that
detector's datasets, not fitted to one run. det3 first.

## Answer
- **Yes on det3, in tan form, per mount.** A single det3 shape, never fitted on the target, gives the target's own-map
  gain on every top-slot det3 dataset (6-27 golden, 6-28 night, 700 V and 500 V drift points): X −0.02…−0.09°,
  Y −0.13…−0.16° on s68 ≈ 1.0–1.1°.
- **Store the tilt as a geometric tan offset, not in µm/ns.** The tilt is v·r_f·dh/du, so its µm/ns size scales with
  the drift velocity. The block as built (FB, µm/ns) fails on det3's 500 V row (Y +0.01 ± 0.06°, the row's own map
  −0.12); in tan form the same golden shape gives −0.10 ± 0.04 and the pooled other-dataset shape −0.14 ± 0.04.
  Implemented: bow block keys `a_x`, `a_y` (`wft.reco.tan_bow`, `set_w0kw.py`, `set_bow.py`); blocks without them
  are bit-identical.
- **The mount may matter, but it cannot be validated.** On det3's bottom-slot run (6-25) the top-slot shape gives
  70–80 % of the run's own map (X −0.14 vs −0.17, Y −0.22 vs −0.32). Every other det3 bottom-slot row is already
  classed skeptical in the registry (6-22: micro-TPC angle unusable, r ≈ 0.62; 6-23: degraded night), so a bottom-slot
  shape could only ever be fitted on 6-25 itself and never checked on a second dataset. **Rule: one bow shape per
  chamber (its golden mount); a separate shape for another mount only if a second usable row of that mount confirms
  it.** For det3 6-25 that means the chamber shape (a validated 70–80 % of the gain). A per-run amplitude does not
  rescue it: it closes part of the gap (Y −0.28) but overfits small rows (500 V X −0.09 → −0.03).
- Other chambers: the shape transfers between same-mount datasets of det2, det6 and det7 (pooled ≈ golden ≈ 85–100 %
  of own on Y). Two exceptions: **det4 X** (no transferable shape: pooled +0.02/−0.00 vs own −0.13/−0.09; FB also found
  det4 X gains nothing), and **det6 X** (pooled −0.39 vs own −0.55; det6's map is partly a slope-scale reco bias, skeptic
  review). det7 X: the faulty half is not masked in these tables.

## What was compared
`bow_transfer.py` on FB's per-event tables (finalize-era reco, promoted bundles; det3_o23 and det2_o17 excluded:
broken angles). For each chamber and each of its datasets as the target, only the rigid tilt c0 and slope B (= kw·v)
are refit on the target, 2-fold by event parity, and scored on the other half. Arms: `const` (no bow, today's per-run
w0/kw); `gold_um` (golden shape in µm/ns, as stored); `gold_tan` (golden shape as tan offset); `loo_tan` (tan shape
pooled over all **other** datasets of the chamber, never the target); `loo_amp` (loo_tan + a per-run amplitude);
`own` (the target's own 2-D map, 2-fold: the ceiling). Δs68 [°] vs const ± paired bootstrap. B/B_gold is the
target's slope relative to the golden row (the drift-field change).

| target dataset | view | n | B/B_gold | const s68 | gold_um Δ | gold_tan Δ | loo_tan Δ | loo_amp Δ | own Δ |
|---|---|---|---|---|---|---|---|---|---|
| det2 (golden) — 6-22 longer, 495 V mesh, 1000 V drift | X | 3325 | 1.01 | 1.133 | -0.085 ± 0.020 | -0.083 ± 0.019 | -0.090 ± 0.018 | -0.084 ± 0.021 | -0.091 ± 0.020 |
| det2 (golden) — 6-22 longer, 495 V mesh, 1000 V drift | Y | 3432 | 1.02 | 1.412 | -0.309 ± 0.029 | -0.310 ± 0.031 | -0.327 ± 0.034 | -0.319 ± 0.030 | -0.313 ± 0.031 |
| det2_o22long — 6-22 long (same night, other subrun) | X | 10808 | 1.00 | 1.109 | -0.080 ± 0.012 | -0.080 ± 0.011 | -0.080 ± 0.011 | -0.085 ± 0.011 | -0.094 ± 0.011 |
| det2_o22long — 6-22 long (same night, other subrun) | Y | 11186 | 1.01 | 1.348 | -0.260 ± 0.016 | -0.258 ± 0.016 | -0.258 ± 0.016 | -0.268 ± 0.017 | -0.273 ± 0.015 |
| det3 (golden) — 6-27 sat long, 490 / 1000 V | X | 6615 | 1.00 | 1.074 | -0.076 ± 0.010 | -0.076 ± 0.010 | -0.082 ± 0.010 | -0.075 ± 0.009 | -0.102 ± 0.010 |
| det3 (golden) — 6-27 sat long, 490 / 1000 V | Y | 6646 | 1.00 | 1.019 | -0.129 ± 0.012 | -0.129 ± 0.011 | -0.131 ± 0.012 | -0.128 ± 0.010 | -0.132 ± 0.011 |
| det3_day625 — 6-25 day, BOTTOM slot, 495 / 500 V | X | 7458 | 0.61 | 1.159 | -0.169 ± 0.013 | -0.139 ± 0.009 | -0.138 ± 0.009 | -0.140 ± 0.009 | -0.172 ± 0.010 |
| det3_day625 — 6-25 day, BOTTOM slot, 495 / 500 V | Y | 7485 | 0.61 | 1.367 | -0.299 ± 0.017 | -0.224 ± 0.012 | -0.219 ± 0.012 | -0.277 ± 0.016 | -0.317 ± 0.017 |
| det3_d700 — 6-27 drift scan 700 V (small) | X | 727 | 0.69 | 1.093 | -0.039 ± 0.032 | -0.022 ± 0.029 | -0.022 ± 0.026 | -0.015 ± 0.020 | -0.042 ± 0.034 |
| det3_d700 — 6-27 drift scan 700 V (small) | Y | 732 | 0.69 | 1.081 | -0.154 ± 0.047 | -0.156 ± 0.040 | -0.141 ± 0.048 | -0.158 ± 0.040 | -0.162 ± 0.039 |
| det3_d500 — 6-27 drift scan 500 V (small) | X | 650 | 0.44 | 1.141 | -0.080 ± 0.046 | -0.095 ± 0.037 | -0.088 ± 0.037 | -0.025 ± 0.018 | -0.060 ± 0.030 |
| det3_d500 — 6-27 drift scan 500 V (small) | Y | 679 | 0.44 | 1.131 | +0.010 ± 0.062 | -0.103 ± 0.038 | -0.135 ± 0.038 | -0.084 ± 0.037 | -0.124 ± 0.035 |
| det3_p2 — 6-28 night (same mount) | X | 11080 | 1.01 | 1.068 | -0.082 ± 0.007 | -0.082 ± 0.007 | -0.084 ± 0.007 | -0.085 ± 0.008 | -0.093 ± 0.008 |
| det3_p2 — 6-28 night (same mount) | Y | 11195 | 1.01 | 1.029 | -0.127 ± 0.008 | -0.127 ± 0.009 | -0.130 ± 0.011 | -0.125 ± 0.010 | -0.129 ± 0.009 |
| det4 (golden) — 6-24 day long, 495 / 600 V | X | 5042 | 1.00 | 1.726 | -0.034 ± 0.019 | -0.033 ± 0.018 | +0.015 ± 0.014 | +0.015 ± 0.013 | -0.125 ± 0.022 |
| det4 (golden) — 6-24 day long, 495 / 600 V | Y | 5150 | 1.00 | 1.730 | -0.367 ± 0.025 | -0.367 ± 0.028 | -0.323 ± 0.027 | -0.323 ± 0.028 | -0.389 ± 0.029 |
| det4_o23 — 6-23 night | X | 3534 | 1.13 | 1.839 | -0.002 ± 0.018 | -0.003 ± 0.020 | -0.003 ± 0.019 | +0.000 ± 0.019 | -0.089 ± 0.026 |
| det4_o23 — 6-23 night | Y | 3682 | 1.16 | 1.923 | -0.423 ± 0.031 | -0.430 ± 0.030 | -0.430 ± 0.033 | -0.444 ± 0.034 | -0.493 ± 0.033 |
| det6 (golden) — 6-26 night long, 495 / 700 V | X | 6667 | 0.94 | 1.833 | -0.455 ± 0.020 | -0.451 ± 0.022 | -0.386 ± 0.018 | -0.425 ± 0.020 | -0.554 ± 0.023 |
| det6 (golden) — 6-26 night long, 495 / 700 V | Y | 6855 | 0.94 | 1.931 | -0.489 ± 0.020 | -0.488 ± 0.019 | -0.492 ± 0.021 | -0.494 ± 0.021 | -0.535 ± 0.019 |
| det6_longer — 6-26 longer (same night) | X | 3753 | 0.73 | 1.841 | -0.291 ± 0.028 | -0.299 ± 0.022 | -0.298 ± 0.025 | -0.273 ± 0.022 | -0.373 ± 0.021 |
| det6_longer — 6-26 longer (same night) | Y | 3832 | 0.73 | 2.104 | -0.365 ± 0.037 | -0.418 ± 0.031 | -0.415 ± 0.033 | -0.407 ± 0.030 | -0.457 ± 0.027 |
| det6_hv505 — mesh 505 V | X | 556 | 0.84 | 1.940 | -0.442 ± 0.074 | -0.447 ± 0.067 | -0.369 ± 0.062 | -0.404 ± 0.066 | -0.438 ± 0.073 |
| det6_hv505 — mesh 505 V | Y | 587 | 0.87 | 1.904 | -0.513 ± 0.076 | -0.485 ± 0.068 | -0.485 ± 0.068 | -0.518 ± 0.062 | -0.503 ± 0.065 |
| det6_hv530 — mesh 530 V | X | 204 | 0.77 | 2.068 | -0.585 ± 0.166 | -0.489 ± 0.122 | -0.465 ± 0.116 | -0.436 ± 0.119 | -0.489 ± 0.143 |
| det6_hv530 — mesh 530 V | Y | 231 | 0.73 | 2.340 | -0.395 ± 0.193 | -0.295 ± 0.173 | -0.282 ± 0.195 | -0.281 ± 0.179 | -0.371 ± 0.195 |
| det7 (golden) — 6-26 night long, 495 / 700 V | X | 5315 | 1.00 | 1.634 | -0.054 ± 0.016 | -0.054 ± 0.018 | -0.060 ± 0.017 | -0.058 ± 0.018 | -0.065 ± 0.018 |
| det7 (golden) — 6-26 night long, 495 / 700 V | Y | 6000 | 1.00 | 1.600 | -0.288 ± 0.020 | -0.290 ± 0.019 | -0.292 ± 0.018 | -0.292 ± 0.018 | -0.296 ± 0.020 |
| det7_longer — 6-26 longer (same night) | X | 2893 | 0.92 | 1.623 | -0.054 ± 0.029 | -0.053 ± 0.029 | -0.052 ± 0.029 | -0.054 ± 0.033 | -0.111 ± 0.033 |
| det7_longer — 6-26 longer (same night) | Y | 3301 | 0.91 | 1.581 | -0.284 ± 0.028 | -0.284 ± 0.022 | -0.285 ± 0.026 | -0.282 ± 0.024 | -0.330 ± 0.026 |
| det7_hv480 — mesh 480 V | X | 480 | 0.99 | 1.589 | -0.074 ± 0.053 | -0.075 ± 0.059 | -0.069 ± 0.060 | -0.036 ± 0.057 | -0.131 ± 0.059 |
| det7_hv480 — mesh 480 V | Y | 528 | 0.99 | 1.442 | -0.327 ± 0.061 | -0.319 ± 0.061 | -0.310 ± 0.060 | -0.305 ± 0.063 | -0.289 ± 0.064 |
| det7_hv505 — mesh 505 V | X | 417 | 0.92 | 1.730 | +0.076 ± 0.110 | +0.079 ± 0.101 | +0.073 ± 0.104 | +0.033 ± 0.109 | +0.066 ± 0.112 |
| det7_hv505 — mesh 505 V | Y | 457 | 0.93 | 1.703 | -0.241 ± 0.080 | -0.266 ± 0.067 | -0.247 ± 0.071 | -0.243 ± 0.066 | -0.200 ± 0.071 |

## For the pass (PASS_PLAN E2)
1. BOW node writes the **tan form** (`a_x`, `a_y`), measured on the chamber's golden row, **one block per chamber**;
   every row of the chamber inherits it (vrefit `--keep-v`). A per-mount block only with a second usable row of
   that mount to validate it (none exists today: all usable rows except det3 6-25 are on their chamber's golden mount).
2. **Transfer gate** (new, table level, cheap): after the pass reco, run `bow_transfer.py` on every row of each
   chamber with the pass tables. A mount's shape is accepted if, on every other row of that mount with ≥ 2000 events,
   loo_tan reaches ≥ 70 % of the row's own map and never worsens s68. Report det4 X / det6 X separately.
3. If the re-decode rescues det3 6-22/6-23 (registry: "re-judge on the re-decode"), the gate can test a bottom-slot
   shape across those nights and 6-25; until then 6-25 takes the chamber shape.

## Not ruled out / limits
- Tables are finalize-era (old σ0 handling, promoted bundles), not the planned pass chain; the gate re-does this on
  the pass tables.
- Small rows (det3 d500/d700 ~700 events, det6/det7 HV rows 200–600) carry ±0.03–0.2° errors.
- The 6-25 gap may be the mount or that day's conditions; with one usable bottom-slot table they cannot be separated.
