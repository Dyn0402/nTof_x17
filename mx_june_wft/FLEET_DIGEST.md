# Waveform-first vs hits-chain digest

| quantity | sat_det3 | o22_long_det2 | g_det4 | g_det6_long | g_det7_long |
|---|---|---|---|---|---|
| rays | 7053  (was 7130) | 3668  (was 3728) | — | 9626  (was 10383) | 9411  (was 9557) |
| has_any % | 100.0  (was 100.0) | 100.0  (was 100.0) | — | 100.0  (was 95.8, better) | 100.0  (was 96.1, better) |
| within 5 mm % | 92.1  (was 93.4, worse) | 91.2  (was 91.1, better) | — | 75.4  (was 57.8, better) | 57.3  (was 43.1, better) |
| reco-at-all % | 97.3  (was 96.4, better) | 96.3  (was 95.4, better) | — | 78.8  (was 60.9, better) | 66.4  (was 50.6, better) |
| reco_far % | 5.1  (was 3.0, worse) | 5.1  (was 4.2, worse) | — | 3.4  (was 3.1, worse) | 9.1  (was 7.5, worse) |
| core sigma r mm | 0.47  (was 0.48, better) | 0.48  (was 0.44, worse) | — | 0.43  (was 0.45, better) | 0.62  (was 0.59, worse) |
| median r mm | 0.78  (was 0.80, better) | 0.81  (was 0.79, worse) | — | 0.75  (was 0.81, better) | 1.01  (was 1.07, better) |
| spark_frac % | 8.2  (was 9.1) | 9.7  (was 6.5) | — | 22.3  (was 27.3) | 37.4  (was 38.9) |
| sigma_theta X | 1.20  (was 2.42, better) | 1.31  (was 2.47, better) | — | 2.28  (was 3.42, better) | 1.96  (was 2.82, better) |
| sigma_theta Y | 1.14  (was 2.60, better) | 1.56  (was 2.04, better) | — | 2.52  (was 2.58, better) | 1.71  (was 2.54, better) |
| bias X deg | -0.04 | -0.38 | — | +0.03 | -0.10 |
| bias Y deg | -0.29 | -0.38 | — | -0.95 | -0.31 |
| implied-v spread X | 2.31 | 3.94 | — | 4.82 | 1.34 |
| implied-v spread Y | 2.39 | 5.33 | — | 6.36 | 2.65 |
| v_drift um/ns | 36.6  (was 34.3) | 39.9  (was 34.3) | — | 26.7  (was 34.3) | 36.7  (was 34.3) |

### The hits chain today, through this same accounting

| quantity | sat_det3 | o22_long_det2 | g_det4 | g_det6_long | g_det7_long |
|---|---|---|---|---|---|
| within 5 mm % | 93.13 | 80.86 | — | 44.43 | 18.17 |
| reco-at-all % | 97.02 | 95.60 | — | 76.40 | 64.81 |
| core sigma r mm | 0.45 | 0.61 | — | 0.66 | 2.71 |
| median r mm | 0.76 | 1.07 | — | 2.13 | 48.45 |

(det2/det6/det7 hits caches predate the 2026-07-25 significance floor unless rebuilt — check for a `cache/event_results.meta.json` before trusting their position rows.)

GATE: FAILED
- sat_det3: within_R = 92.145 fails >= 93.0
