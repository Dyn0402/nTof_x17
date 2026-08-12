# Waveform-first vs hits-chain digest

| quantity | sat_det3 | o22_long_det2 | g_det4 | g_det6_long | g_det7_long |
|---|---|---|---|---|---|
| rays | 7049  (was 7130) | 3678  (was 3728) | 12258  (was 12318) | 9628  (was 10383) | 9479  (was 9557) |
| has_any % | 100.0  (was 100.0) | 100.0  (was 100.0) | 95.8  (was 69.6, better) | 100.0  (was 95.8, better) | 100.0  (was 96.1, better) |
| within 5 mm % | 93.3  (was 93.4, worse) | 92.0  (was 91.1, better) | 41.6  (was 20.7, better) | 74.9  (was 57.8, better) | 56.9  (was 43.1, better) |
| reco-at-all % | 97.3  (was 96.4, better) | 96.3  (was 95.4, better) | 49.3  (was 23.1, better) | 78.8  (was 60.9, better) | 66.1  (was 50.6, better) |
| reco_far % | 4.0  (was 3.0, worse) | 4.3  (was 4.2, worse) | 7.7  (was 2.4, worse) | 3.9  (was 3.1, worse) | 9.2  (was 7.5, worse) |
| core sigma r mm | 0.45  (was 0.48, better) | 0.43  (was 0.44, better) | 0.56  (was 0.67, better) | 0.46  (was 0.45, worse) | 0.62  (was 0.59, worse) |
| median r mm | 0.72  (was 0.80, better) | 0.70  (was 0.79, better) | 0.97  (was 1.06, better) | 0.80  (was 0.81, better) | 1.02  (was 1.07, better) |
| spark_frac % | 8.2  (was 9.1) | 9.7  (was 6.5) | 9.8  (was 6.0) | 22.3  (was 27.3) | 37.4  (was 38.9) |
| sigma_theta X | 1.16  (was 2.42, better) | 1.29  (was 2.47, better) | 2.49  (was 2.60, better) | 2.43  (was 3.42, better) | 2.07  (was 2.82, better) |
| sigma_theta Y | 1.12  (was 2.60, better) | 1.50  (was 2.04, better) | 2.45  (was 2.50, better) | 2.82  (was 2.58, worse) | 1.75  (was 2.54, better) |
| bias X deg | -0.07 | -0.36 | -0.16 | +0.03 | -0.09 |
| bias Y deg | -0.28 | -0.39 | -0.19 | -1.04 | -0.32 |
| implied-v spread X | 0.78 | 1.20 | 4.42 | 5.95 | 1.60 |
| implied-v spread Y | 1.15 | 5.07 | 4.13 | 7.51 | 2.74 |
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
