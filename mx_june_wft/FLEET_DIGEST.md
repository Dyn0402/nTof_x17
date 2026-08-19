# Waveform-first vs hits-chain digest

| quantity | sat_det3 | o22_long_det2 | g_det4 | g_det6_long | g_det7_long |
|---|---|---|---|---|---|
| rays | 7054  (was 7130) | 3675  (was 3728) | 12259  (was 12318) | 9628  (was 10383) | 9428  (was 9557) |
| has_any % | 100.0  (was 100.0, worse) | 100.0  (was 100.0) | 95.8  (was 69.6, better) | 100.0  (was 95.8, better) | 100.0  (was 96.1, better) |
| within 5 mm % | 93.3  (was 93.4, worse) | 92.0  (was 91.1, better) | 41.6  (was 20.7, better) | 74.9  (was 57.8, better) | 57.1  (was 43.1, better) |
| reco-at-all % | 97.3  (was 96.4, better) | 96.3  (was 95.4, better) | 49.3  (was 23.1, better) | 78.8  (was 60.9, better) | 66.3  (was 50.6, better) |
| reco_far % | 4.0  (was 3.0, worse) | 4.3  (was 4.2, worse) | 7.7  (was 2.4, worse) | 3.9  (was 3.1, worse) | 9.3  (was 7.5, worse) |
| core sigma r mm | 0.45  (was 0.48, better) | 0.44  (was 0.44, better) | 0.56  (was 0.67, better) | 0.46  (was 0.45, worse) | 0.63  (was 0.59, worse) |
| median r mm | 0.72  (was 0.80, better) | 0.70  (was 0.79, better) | 0.97  (was 1.06, better) | 0.80  (was 0.81, better) | 1.01  (was 1.07, better) |
| spark_frac % | 8.2  (was 9.1) | 9.7  (was 6.5) | 9.8  (was 6.0) | 22.3  (was 27.3) | 37.4  (was 38.9) |
| sigma_theta X | 1.15  (was 2.42, better) | 1.21  (was 2.47, better) | 2.09  (was 2.60, better) | 2.24  (was 3.42, better) | 1.94  (was 2.82, better) |
| sigma_theta Y | 1.18  (was 2.60, better) | 1.57  (was 2.04, better) | 2.28  (was 2.50, better) | 2.51  (was 2.58, better) | 1.87  (was 2.54, better) |
| bias X deg | +0.00 | -0.02 | +0.08 | -0.01 | +0.00 |
| bias Y deg | -0.00 | -0.00 | +0.04 | +0.20 | +0.00 |
| implied-v spread X | 1.11 | 1.86 | 3.85 | 6.31 | 2.17 |
| implied-v spread Y | 1.41 | 5.17 | 4.08 | 7.82 | 3.31 |
| v_drift um/ns | 36.6  (was 34.3) | 39.9  (was 34.3) | 34.2  (was 34.3) | 26.7  (was 34.3) | 36.6  (was 34.3) |

### The hits chain today, through this same accounting

| quantity | sat_det3 | o22_long_det2 | g_det4 | g_det6_long | g_det7_long |
|---|---|---|---|---|---|
| within 5 mm % | 93.13 | 92.06 | 40.67 | 71.19 | 52.73 |
| reco-at-all % | 97.02 | 96.11 | 48.66 | 74.60 | 62.87 |
| core sigma r mm | 0.45 | 0.43 | 0.66 | 0.38 | 0.52 |
| median r mm | 0.76 | 0.74 | 1.14 | 0.69 | 0.98 |

(det2/det6/det7 hits caches predate the 2026-07-25 significance floor unless rebuilt — check for a `cache/event_results.meta.json` before trusting their position rows.)

### Which calibration each column is

| key | bundle | c2/c1 |
|---|---|---|
| sat_det3 | `calib_bundle_r06` | 0.60 |
| o22_long_det2 | `calib_bundle_r06` | 0.60 |
| g_det4 | `calib_bundle_lp_t0p` | 0.67 |
| g_det6_long | `calib_bundle_lp` | 0.82 |
| g_det7_long | `calib_bundle_r06` | 0.60 |

A bundle is per detector **and** per run condition; columns built on different bundles are not interchangeable.

GATE: all thresholds met
