# Waveform-first vs hits-chain digest

| quantity | sat_det3 | o22_long_det2 | g_det4 | g_det6_long | g_det7_long |
|---|---|---|---|---|---|
| rays | 7055  (was 7130) | 3669  (was 3728) | 12259  (was 12318) | 9626  (was 10383) | 9429  (was 9557) |
| has_any % | 100.0  (was 100.0) | 100.0  (was 100.0) | 95.8  (was 69.6, better) | 100.0  (was 95.8, better) | 100.0  (was 96.1, better) |
| within 5 mm % | 93.5  (was 93.4, better) | 91.9  (was 91.1, better) | 41.9  (was 20.7, better) | 75.4  (was 57.8, better) | 56.9  (was 43.1, better) |
| reco-at-all % | 97.3  (was 96.4, better) | 96.3  (was 95.4, better) | 49.3  (was 23.1, better) | 78.8  (was 60.9, better) | 66.3  (was 50.6, better) |
| reco_far % | 3.7  (was 3.0, worse) | 4.4  (was 4.2, worse) | 7.4  (was 2.4, worse) | 3.4  (was 3.1, worse) | 9.4  (was 7.5, worse) |
| core sigma r mm | 0.46  (was 0.48, better) | 0.44  (was 0.44, better) | 0.67  (was 0.67, better) | 0.43  (was 0.45, better) | 0.63  (was 0.59, worse) |
| median r mm | 0.71  (was 0.80, better) | 0.69  (was 0.79, better) | 1.11  (was 1.06, worse) | 0.75  (was 0.81, better) | 1.02  (was 1.07, better) |
| spark_frac % | 8.2  (was 9.1) | 9.7  (was 6.5) | 9.8  (was 6.0) | 22.3  (was 27.3) | 37.4  (was 38.9) |
| sigma_theta X | 1.08  (was 2.42, better) | 1.14  (was 2.47, better) | 2.36  (was 2.60, better) | 2.28  (was 3.42, better) | 1.98  (was 2.82, better) |
| sigma_theta Y | 1.11  (was 2.60, better) | 1.63  (was 2.04, better) | 2.86  (was 2.50, worse) | 2.52  (was 2.58, better) | 2.09  (was 2.54, better) |
| bias X deg | -0.03 | -0.07 | -0.01 | +0.03 | -0.02 |
| bias Y deg | -0.01 | -0.05 | -0.05 | -0.95 | +0.31 |
| implied-v spread X | 2.12 | 1.92 | 4.13 | 4.82 | 3.67 |
| implied-v spread Y | 3.40 | 6.43 | 6.03 | 6.36 | 7.38 |
| v_drift um/ns | 36.6  (was 34.3) | 39.9  (was 34.3) | 34.2  (was 34.3) | 26.7  (was 34.3) | 36.6  (was 34.3) |

### The hits chain today, through this same accounting

| quantity | sat_det3 | o22_long_det2 | g_det4 | g_det6_long | g_det7_long |
|---|---|---|---|---|---|
| within 5 mm % | 93.13 | 92.06 | 40.67 | 71.19 | 52.73 |
| reco-at-all % | 97.02 | 96.11 | 48.66 | 74.60 | 62.87 |
| core sigma r mm | 0.45 | 0.43 | 0.66 | 0.38 | 0.52 |
| median r mm | 0.76 | 0.74 | 1.14 | 0.69 | 0.98 |

(det2/det6/det7 hits caches predate the 2026-07-25 significance floor unless rebuilt — check for a `cache/event_results.meta.json` before trusting their position rows.)

GATE: all thresholds met
