# GB — Is the M3 reference limiting the bench angle resolution? What does SPS show? (round 3, item 2, 2026-10-28)

## Verdict
**No.** M3 + multiple scattering at the chamber is **about 0.03–0.05°**, and the data bound it at **≤ 0.11° (X) / 0.14° (Y)**.
That is 0.4–2 % of the unexplained variance on det2/det3 and < 1 % on det4/det6/det7.

- **The earlier "≤ 0.15°" was generous by about 3×, not an underestimate.** Scattering is small for three reasons: the
  production cut (χ² < 1, NClus 4) removes large kinks; at z = 702 the 4-point fit already absorbs about half of any local
  kink; the median muon momentum through the stack is about 2.3 GeV/c.
- **det6 and det7 do share an error, but it is not M3.** A common term of 0.5° (X) / 0.9° (Y) is position-dependent,
  shared by both chambers, and disappears once each chamber's own 2D map is removed. The M3 slope is pinned by the two
  chambers' positions to 0.03° cell to cell. So the common term is **the bow** (FB), and this is a reference-free
  confirmation of it.
- **SPS (reference = the beam, 0.03°):**
  - det4 head-on reads like the bench: 1.3–1.6° against 1.53°.
  - At 25.6° SPS reads 1.0–1.2° (core), better than the bench, in a different gas.

## 1. M3 error at the chamber (`ms_mc.py`)
- **Stack:** 4 M3 stations at z = 1302 / 1185 / 144 / 24; chambers at z = 702 (det2/3/4/7) and z = 232 (det6). Per-plane
  σ is X 0.41 mm and Y 0.45–0.51 mm.
- **Noise only:** slope error 0.35 mrad; at the chamber s68 0.022° (X) / 0.026° (Y).
- **Scattering:** a kink k at height z_s leaves a reference error c·k at the chamber:

| kink at | c, chamber at z = 702 | c, chamber at z = 232 |
|---|---|---|
| just above the chamber | −0.45 | −0.85 |
| just below the chamber | +0.52 | +0.12 |
| air at z = 467 | +0.33 | −0.67 |
| M3 station 1185 / 144 | −0.05 / +0.06 | −0.05 / +0.06 |

  The chamber's own cathode scatters the track before the gap, so it is counted in the local angle.
- **Monte Carlo:** Highland scattering, sea-level spectrum (Reyna–Bugaev), the production cut, per-slot material scanned
  over 0–16 % X0.
  - 2–8 % X0: s68 0.035–0.05°, rms 0.05–0.09°.
  - 16 % X0 (extreme): s68 0.06–0.09°.
  - In quadrature on a 1° chamber: ≤ 0.1° even in the extreme case.

## 2. Data tests (no new data)
- **(a) Resolution vs M3 χ² quintile** (`chi2_split.py`, production range only):
  - det2, det3, det7 and det4 Y are flat to ±0.05°;
  - det4 X (1.51 → 1.74°) and det6 X (1.68 → 1.90°) rise;
  - det7 X uses the same M3 coordinate on the same tracks as det6 X and is flat (1.59 → 1.59°), so the det6 rise is
    not the reference.
- **(b) Kink between the two M3 doublets as a momentum proxy** (`kink_det3.py`; sat det3 run, 6850 events, the only run
  with per-plane M3 positions):
  - det3 X and Y show no rise toward soft muons;
  - regression of the residual on the kink explains ≤ 0.055°;
  - sensitivity check: replacing the reference by the single-doublet slope (known noise 0.28–0.35°) worsens the
    resolution by exactly that, so a term of that size cannot hide.
- **(c) Two chambers, same track — the key test** (`pair_cov.py`, `pos_pin.py`). det6 (z = 232) + det7 (z = 702), 6-26:

| covariance of residuals, deg² | X | Y |
|---|---|---|
| raw | 0.25 ± 0.06 (common 0.50°) | 0.80 ± 0.05 (common 0.90°) |
| after removing each chamber's angle trend | 0.23 | 0.81 |
| after also removing each chamber's own 40 mm map | −0.01 ± 0.05 | 0.07–0.11 ± 0.04 (≤ 0.42° at 95 %) |
| split by M3 χ² tercile (after angle trend) | flat | flat |

  - The cell medians of the two chambers correlate at r = 0.57 (X) and 0.83 (Y).
  - **Position pin:** r6 − r7 = 470 mm × (M3 slope error) + position noise, so the measured spread bounds the M3 slope
    error at ≤ 0.107° (X) / 0.140° (Y) per event. Its excess over position noise is ≤ 0.03° at 95 %.
  - The cell map of the slope error spreads by 0.03° and does not correlate with the common angle map.
  - Side result, position: there is a common position term of about 0.42 mm, which supports "bench *position* resolution
    is reference-limited". Angles are not.
  - det3 + det2 (6-22, a poor run) agree: common term ≤ 0.23° (X) / ≤ 0.38° (Y) at 95 %.

## 3. SPS (det4 at H4, 120 GeV π, `sps_compare.py`)
- **No per-event reference is needed.** Beam divergence is 0.53 mrad = 0.03° and scattering is about 0.02 mrad, so the
  spread about the median is det4's own resolution, over a ±30–36 mm spot.
- Data: run_63, Ar/CF4/iso, zero-suppressed, h4 mask, 1000 events per plateau. Mount-scale values are in brackets.

| | SPS 235 V/cm | SPS 200 V/cm | bench det4 (vs M3) |
|---|---|---|---|
| head-on X, s68 | 1.44° (1.56°) | 1.31° (1.45°) | 0–5°: 1.53° |
| inclined Y at 25.64°, s68 | 0.99° (1.04°) | 0.96° (1.04°) | 15–30°: Y 1.42°, X 1.93° |
| fraction beyond 10° | 20 % | 20 % | 12–17 % |

- **165 V/cm is excluded:** its column overruns the readout window.
- **Head-on, the bench is not reference-limited:** SPS, with a 0.03° reference, reads the same.
- **At 25.6° SPS is better by 1.0–1.6° in quadrature.** M3 is bounded at 0.11°, so it can't be the reference. The
  candidates are the slow gas (2.3× more samples across the column) and the small area sampled (less bow).
- No number transfers to the bench: different gas, zero suppression, one chamber.

## 4. Consequences
| chamber | M3 + MS, best estimate | measured bound | share of FD unexplained variance |
|---|---|---|---|
| det2 | 0.05° | ≤ 0.23–0.38° (weak pair) | 0.4 % (2.0 % at 0.11°) |
| det3 | 0.05° | ≤ 0.055° (kink regression) | 0.5 % (2.2 %) |
| det4 | 0.05° | Monte Carlo + χ² flatness only | 0.1 % (0.6 %) |
| det6 | 0.05–0.07° | ≤ 0.11 / 0.14° (position pin) | 0.2 % (0.8 %) |
| det7 | 0.05° | ≤ 0.11 / 0.14° (position pin) | 0.1 % (0.6 %) |

- **FD budget:** replace the M3 row (0.15°) with **0.05° (≤ 0.11° measured)**. The unexplained terms become
  0.78 / 0.74 / 1.38 / 1.26 / 1.37° (det2/3/4/6/7). They are chamber-side.
- **Quote bench "vs M3" angle resolutions as the chamber's own** (no deconvolution), with "reference contribution < 0.1°
  (measured)".

## Files
`rerun/model_audit/followups/gb_m3_angle/`:
- code: `common.py`, `ms_mc.py`, `kink_det3.py`, `pair_cov.py`, `pos_pin.py`, `chi2_split.py`, `sps_compare.py`;
- results: `results/*.json`. No figures or report.html yet.

## Limitations
- **Kink regression:** det3 only (per-plane M3 positions exist only in the sat run).
- **χ² test:** covers only the production range (χ² < 1). Large-kink events are cut, not measured.
- **Material is unrecorded**, hence the 0–16 % X0 scan. Single-scattering tails beyond Highland matter only for the
  > 5° tail.
- **det4 has no partner chamber,** so its bound rests on the Monte Carlo alone.
- **6-22 pair:** the M3 → strip-frame rotation was inferred by regression, without the alignment file.
- **SPS:** one chamber in another gas, zero-suppressed, angle scale 8 % low (both scales shown), 20 % far tail.
- **det6/det7 covariance precision ±0.04–0.05 deg²:** this excludes the unexplained term by > 30σ but not a 0.1° reference
  term; the position pin bounds that.
