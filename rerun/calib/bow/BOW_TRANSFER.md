# Chamber bow: one correction per chamber that holds across its datasets (2026-10-29)

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
- **The mount matters.** On det3's bottom-slot run (6-25) the top-slot shape gives 70–80 % of the run's own map
  (X −0.14 vs −0.17, Y −0.22 vs −0.32). det3 has 35 bottom-slot rows in the pass (6-22, 6-23 overnights, 6-25), so
  the pass needs **one bow shape per chamber per mount**. A per-run amplitude instead does not work: it closes part
  of that gap (Y −0.28) but overfits small rows (500 V X −0.09 → −0.03).
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
1. BOW node writes the **tan form** (`a_x`, `a_y`), measured on the chamber's golden row, **one block per mount**
   (slot × mounting epoch; det2 remounted after 6-17). Rows inherit the block of their own mount.
2. **Transfer gate** (new, table level, cheap): after the pass reco, run `bow_transfer.py` on every row of each
   chamber with the pass tables. A mount's shape is accepted if, on every other row of that mount with ≥ 2000 events,
   loo_tan reaches ≥ 70 % of the row's own map and never worsens s68. Report det4 X / det6 X separately.
3. The bottom-slot det3 shape can only be checked across datasets once the pass tables of 6-22/6-23 exist (today the
   only usable bottom-slot table is 6-25). That is the first thing the gate must show.

## Not ruled out / limits
- Tables are finalize-era (old σ0 handling, promoted bundles), not the planned pass chain; the gate re-does this on
  the pass tables.
- Small rows (det3 d500/d700 ~700 events, det6/det7 HV rows 200–600) carry ±0.03–0.2° errors.
- The 6-25 gap may be the mount or that day's conditions; with one usable bottom-slot table they cannot be separated.
