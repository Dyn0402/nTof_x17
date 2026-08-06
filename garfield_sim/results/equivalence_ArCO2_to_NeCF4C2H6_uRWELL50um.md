# Equal-gain mesh voltage: Ar/CO₂ 70/30 uRW50 → Ne/CF₄/C₂H₆ 80/10/10 uRW50

Gap 50 µm, T = 293.15 K. Each row is one gain: read the Ar/CO₂ 70/30 uRW50 voltage on the left and the Ne/CF₄/C₂H₆ 80/10/10 uRW50 voltage that reaches the same simulated gain on the right. `*` marks a voltage outside that mixture's simulated span.

**Penning.** `Ar_CO2_70_30_uRW50` uses Garfield++'s built-in parameterisation (auto); `Ne_CF4_C2H6_80_10_10_uRW50_rP040` is hand-set — Garfield++ has no curve for it; `Ne_CF4_C2H6_80_10_10_uRW50_rP050` is hand-set — Garfield++ has no curve for it; `Ne_CF4_C2H6_80_10_10_uRW50_rP060` is hand-set — Garfield++ has no curve for it.

The two sides are therefore not on equal footing: one is a measurement Garfield++ ships, the other is a choice. Where a bracket is shown it is an assumption, not an uncertainty propagated from data.

Across the full bracket that assumption is worth at most **±8 V** on this map (largest half-spread in the tables below). Judge it against the other error terms before deciding it is the one that matters.

## CERN_450m

Simulated spans: Ar/CO₂ 70/30 uRW50 280–520 V (1000 events/point); Ne/CF₄/C₂H₆ 80/10/10 uRW50 r=0.40 220–480 V (400 events/point); Ne/CF₄/C₂H₆ 80/10/10 uRW50 r=0.50 220–480 V (1000 events/point); Ne/CF₄/C₂H₆ 80/10/10 uRW50 r=0.60 220–480 V (400 events/point).

| V(ref) | gain | Ne/CF₄/C₂H₆ 80/10/10 uRW50 r=0.40 | Ne/CF₄/C₂H₆ 80/10/10 uRW50 r=0.50 | Ne/CF₄/C₂H₆ 80/10/10 uRW50 r=0.60 | ΔV central | bracket |
|---|---|---|---|---|---|---|
| 296 | 72 | 225 | 221 | 220* | -75 V | ±3 V |
| 312 | 109 | 242 | 239 | 237 | -74 V | ±3 V |
| 329 | 166 | 260 | 256 | 253 | -72 V | ±3 V |
| 345 | 252 | 277 | 274 | 271 | -71 V | ±3 V |
| 362 | 382 | 296 | 292 | 288 | -69 V | ±4 V |
| 378 | 579 | 314 | 311 | 306 | -67 V | ±4 V |
| 394 | 879 | 333 | 330 | 324 | -65 V | ±4 V |
| 411 | 1,333 | 353 | 349 | 343 | -62 V | ±5 V |
| 427 | 2,022 | 372 | 368 | 362 | -59 V | ±5 V |
| 444 | 3,068 | 393 | 388 | 381 | -56 V | ±6 V |
| 460 | 4,654 | 414 | 408 | 401 | -53 V | ±6 V |
| 477 | 7,060 | 435 | 428 | 422 | -49 V | ±7 V |
| 494 | 10,710 | 457 | 449 | 443 | -44 V | ±7 V |
| 510 | 16,247 | 480 | 471 | 465 | -39 V | ±8 V |

Same table in **field**, which is the form that travels to a detector with a different amplification gap. Equal gain means equal effective Townsend coefficient, and that condition has no gap in it — so divide out the 50 µm gap these numbers were simulated at, and multiply back in by whatever gap the other detector has. See `mm_gap_scaling.py` for how far that actually holds — it was checked around the 150 µm Micromegas case, where rebuilding the map at 128 vs 150 µm moved it only a few volts, and it degraded outside that range. It has NOT been checked around this geometry.

| E(ref) kV/cm | gain | Ne/CF₄/C₂H₆ 80/10/10 uRW50 r=0.40 kV/cm | Ne/CF₄/C₂H₆ 80/10/10 uRW50 r=0.50 kV/cm | Ne/CF₄/C₂H₆ 80/10/10 uRW50 r=0.60 kV/cm |
|---|---|---|---|---|
| 59.21 | 72 | 45.02 | 44.28 | 44.00 |
| 62.48 | 109 | 48.45 | 47.75 | 47.32 |
| 65.76 | 166 | 51.94 | 51.27 | 50.70 |
| 69.03 | 252 | 55.50 | 54.84 | 54.14 |
| 72.32 | 382 | 59.13 | 58.47 | 57.64 |
| 75.60 | 579 | 62.84 | 62.15 | 61.21 |
| 78.89 | 879 | 66.62 | 65.90 | 64.86 |
| 82.19 | 1,333 | 70.50 | 69.72 | 68.59 |
| 85.49 | 2,022 | 74.47 | 73.60 | 72.40 |
| 88.79 | 3,068 | 78.54 | 77.55 | 76.30 |
| 92.09 | 4,654 | 82.72 | 81.58 | 80.30 |
| 95.41 | 7,060 | 87.01 | 85.69 | 84.40 |
| 98.72 | 10,710 | 91.44 | 89.89 | 88.61 |
| 102.04 | 16,247 | 96.00 | 94.18 | 92.94 |

Closed-form linear map, `V_target = m·V_ref + c` (from G = A·e^(B·V) on each curve):

| variant | m | c (V) | max resid vs table |
|---|---|---|---|
| Ne/CF₄/C₂H₆ 80/10/10 uRW50 r=0.40 | 1.1870 | -131.8 | 6.2 V |
| Ne/CF₄/C₂H₆ 80/10/10 uRW50 r=0.50 | 1.1696 | -129.5 | 4.6 V |
| Ne/CF₄/C₂H₆ 80/10/10 uRW50 r=0.60 | 1.1535 | -127.7 | 6.2 V |

## Saclay_160m

Simulated spans: Ar/CO₂ 70/30 uRW50 280–520 V (1000 events/point); Ne/CF₄/C₂H₆ 80/10/10 uRW50 r=0.40 220–480 V (400 events/point); Ne/CF₄/C₂H₆ 80/10/10 uRW50 r=0.50 220–480 V (1000 events/point); Ne/CF₄/C₂H₆ 80/10/10 uRW50 r=0.60 220–480 V (400 events/point).

| V(ref) | gain | Ne/CF₄/C₂H₆ 80/10/10 uRW50 r=0.40 | Ne/CF₄/C₂H₆ 80/10/10 uRW50 r=0.50 | Ne/CF₄/C₂H₆ 80/10/10 uRW50 r=0.60 | ΔV central | bracket |
|---|---|---|---|---|---|---|
| 301 | 73 | 226 | 224 | 220 | -77 V | ±3 V |
| 317 | 111 | 244 | 241 | 237 | -77 V | ±3 V |
| 334 | 169 | 261 | 258 | 254 | -76 V | ±4 V |
| 351 | 257 | 279 | 276 | 272 | -75 V | ±4 V |
| 367 | 391 | 298 | 294 | 290 | -73 V | ±4 V |
| 384 | 596 | 317 | 312 | 308 | -72 V | ±4 V |
| 400 | 907 | 336 | 331 | 326 | -69 V | ±5 V |
| 417 | 1,382 | 355 | 350 | 345 | -67 V | ±5 V |
| 433 | 2,104 | 375 | 369 | 364 | -64 V | ±5 V |
| 450 | 3,205 | 395 | 389 | 384 | -61 V | ±6 V |
| 466 | 4,880 | 415 | 409 | 404 | -57 V | ±6 V |
| 483 | 7,432 | 436 | 430 | 424 | -53 V | ±6 V |
| 499 | 11,318 | 458 | 451 | 445 | -48 V | ±7 V |
| 515 | 17,237 | 480 | 472 | 466 | -43 V | ±7 V |

Same table in **field**, which is the form that travels to a detector with a different amplification gap. Equal gain means equal effective Townsend coefficient, and that condition has no gap in it — so divide out the 50 µm gap these numbers were simulated at, and multiply back in by whatever gap the other detector has. See `mm_gap_scaling.py` for how far that actually holds — it was checked around the 150 µm Micromegas case, where rebuilding the map at 128 vs 150 µm moved it only a few volts, and it degraded outside that range. It has NOT been checked around this geometry.

| E(ref) kV/cm | gain | Ne/CF₄/C₂H₆ 80/10/10 uRW50 r=0.40 kV/cm | Ne/CF₄/C₂H₆ 80/10/10 uRW50 r=0.50 kV/cm | Ne/CF₄/C₂H₆ 80/10/10 uRW50 r=0.60 kV/cm |
|---|---|---|---|---|
| 60.17 | 73 | 45.24 | 44.72 | 44.00 |
| 63.50 | 111 | 48.74 | 48.15 | 47.40 |
| 66.82 | 169 | 52.29 | 51.64 | 50.85 |
| 70.14 | 257 | 55.90 | 55.18 | 54.36 |
| 73.45 | 391 | 59.57 | 58.78 | 57.92 |
| 76.76 | 596 | 63.30 | 62.45 | 61.54 |
| 80.07 | 907 | 67.10 | 66.18 | 65.23 |
| 83.37 | 1,382 | 70.98 | 69.97 | 68.99 |
| 86.67 | 2,104 | 74.93 | 73.84 | 72.82 |
| 89.96 | 3,205 | 78.96 | 77.79 | 76.73 |
| 93.25 | 4,880 | 83.08 | 81.82 | 80.71 |
| 96.53 | 7,432 | 87.28 | 85.93 | 84.78 |
| 99.81 | 11,318 | 91.59 | 90.14 | 88.95 |
| 103.09 | 17,237 | 96.00 | 94.45 | 93.21 |

Closed-form linear map, `V_target = m·V_ref + c` (from G = A·e^(B·V) on each curve):

| variant | m | c (V) | max resid vs table |
|---|---|---|---|
| Ne/CF₄/C₂H₆ 80/10/10 uRW50 r=0.40 | 1.1782 | -133.0 | 5.7 V |
| Ne/CF₄/C₂H₆ 80/10/10 uRW50 r=0.50 | 1.1604 | -130.6 | 5.1 V |
| Ne/CF₄/C₂H₆ 80/10/10 uRW50 r=0.60 | 1.1538 | -132.8 | 5.6 V |

