# Equal-gain mesh voltage: Ar/CO₂ 70/30 → Ne/CF₄/C₂H₆ 80/10/10

Gap 150 µm, T = 293.15 K. Each row is one gain: read the Ar/CO₂ 70/30 voltage on the left and the Ne/CF₄/C₂H₆ 80/10/10 voltage that reaches the same simulated gain on the right. `*` marks a voltage outside that mixture's simulated span.

**Penning.** `Ar_CO2_70_30` uses Garfield++'s built-in parameterisation (auto); `Ne_CF4_C2H6_80_10_10_rP040` is hand-set — Garfield++ has no curve for it; `Ne_CF4_C2H6_80_10_10_rP050` is hand-set — Garfield++ has no curve for it; `Ne_CF4_C2H6_80_10_10_rP060` is hand-set — Garfield++ has no curve for it.

The two sides are therefore not on equal footing: one is a measurement Garfield++ ships, the other is a choice. Where a bracket is shown it is an assumption, not an uncertainty propagated from data.

Across the full bracket that assumption is worth at most **±6 V** on this map (largest half-spread in the tables below). Judge it against the other error terms before deciding it is the one that matters.

## CERN_450m

Simulated spans: Ar/CO₂ 70/30 400–740 V (10000 events/point); Ne/CF₄/C₂H₆ 80/10/10 r=0.40 240–540 V (400 events/point); Ne/CF₄/C₂H₆ 80/10/10 r=0.50 240–540 V (1000 events/point); Ne/CF₄/C₂H₆ 80/10/10 r=0.60 240–540 V (400 events/point).

| V(ref) | gain | Ne/CF₄/C₂H₆ 80/10/10 r=0.40 | Ne/CF₄/C₂H₆ 80/10/10 r=0.50 | Ne/CF₄/C₂H₆ 80/10/10 r=0.60 | ΔV central | bracket |
|---|---|---|---|---|---|---|
| 403 | 10 | 241 | 240 | 240* | -163 V | ±0 V |
| 434 | 18 | 262 | 262 | 261 | -172 V | ±1 V |
| 464 | 31 | 284 | 283 | 282 | -181 V | ±1 V |
| 492 | 55 | 305 | 304 | 302 | -188 V | ±1 V |
| 520 | 96 | 327 | 325 | 323 | -195 V | ±2 V |
| 547 | 167 | 348 | 346 | 344 | -201 V | ±2 V |
| 573 | 293 | 370 | 367 | 364 | -206 V | ±3 V |
| 599 | 513 | 391 | 389 | 385 | -210 V | ±3 V |
| 624 | 898 | 413 | 410 | 406 | -214 V | ±4 V |
| 648 | 1,571 | 435 | 430 | 426 | -218 V | ±4 V |
| 672 | 2,750 | 456 | 451 | 447 | -220 V | ±5 V |
| 695 | 4,813 | 478 | 472 | 467 | -223 V | ±5 V |
| 718 | 8,424 | 499 | 493 | 488 | -225 V | ±6 V |
| 740 | 14,744 | 521 | 514 | 508 | -226 V | ±6 V |

Same table in **field**, which is the form that travels to a detector with a different amplification gap. Equal gain means equal effective Townsend coefficient, and that condition has no gap in it — so divide out the 150 µm gap these numbers were simulated at, and multiply back in by whatever gap the other detector has. See `mm_gap_scaling.py` for how far that actually holds (a few volts between 128 and 150 µm; it degrades outside that).

| E(ref) kV/cm | gain | Ne/CF₄/C₂H₆ 80/10/10 r=0.40 kV/cm | Ne/CF₄/C₂H₆ 80/10/10 r=0.50 kV/cm | Ne/CF₄/C₂H₆ 80/10/10 r=0.60 kV/cm |
|---|---|---|---|---|
| 26.88 | 10 | 16.03 | 16.03 | 16.00 |
| 28.93 | 18 | 17.47 | 17.44 | 17.39 |
| 30.91 | 31 | 18.91 | 18.86 | 18.77 |
| 32.82 | 55 | 20.35 | 20.28 | 20.16 |
| 34.67 | 96 | 21.79 | 21.69 | 21.54 |
| 36.47 | 167 | 23.23 | 23.09 | 22.92 |
| 38.22 | 293 | 24.66 | 24.50 | 24.29 |
| 39.92 | 513 | 26.10 | 25.90 | 25.67 |
| 41.58 | 898 | 27.54 | 27.30 | 27.04 |
| 43.20 | 1,571 | 28.97 | 28.70 | 28.42 |
| 44.78 | 2,750 | 30.41 | 30.09 | 29.78 |
| 46.33 | 4,813 | 31.84 | 31.48 | 31.15 |
| 47.85 | 8,424 | 33.28 | 32.87 | 32.52 |
| 49.33 | 14,744 | 34.71 | 34.26 | 33.88 |

Closed-form linear map, `V_target = m·V_ref + c` (from G = A·e^(B·V) on each curve):

| variant | m | c (V) | max resid vs table |
|---|---|---|---|
| Ne/CF₄/C₂H₆ 80/10/10 r=0.40 | 0.8302 | -101.7 | 8.0 V |
| Ne/CF₄/C₂H₆ 80/10/10 r=0.50 | 0.8096 | -92.7 | 7.5 V |
| Ne/CF₄/C₂H₆ 80/10/10 r=0.60 | 0.7941 | -86.9 | 7.5 V |

## Saclay_160m

Simulated spans: Ar/CO₂ 70/30 400–740 V (10000 events/point); Ne/CF₄/C₂H₆ 80/10/10 r=0.40 240–540 V (400 events/point); Ne/CF₄/C₂H₆ 80/10/10 r=0.50 240–540 V (1000 events/point); Ne/CF₄/C₂H₆ 80/10/10 r=0.60 240–540 V (400 events/point).

| V(ref) | gain | Ne/CF₄/C₂H₆ 80/10/10 r=0.40 | Ne/CF₄/C₂H₆ 80/10/10 r=0.50 | Ne/CF₄/C₂H₆ 80/10/10 r=0.60 | ΔV central | bracket |
|---|---|---|---|---|---|---|
| 406 | 9 | 242 | 241 | 240 | -165 V | ±1 V |
| 437 | 16 | 263 | 262 | 261 | -175 V | ±1 V |
| 466 | 27 | 284 | 283 | 281 | -184 V | ±2 V |
| 495 | 47 | 305 | 304 | 302 | -191 V | ±2 V |
| 522 | 82 | 327 | 324 | 322 | -198 V | ±2 V |
| 549 | 141 | 347 | 345 | 343 | -204 V | ±2 V |
| 575 | 244 | 368 | 366 | 363 | -209 V | ±3 V |
| 600 | 420 | 389 | 386 | 383 | -214 V | ±3 V |
| 625 | 725 | 410 | 407 | 403 | -218 V | ±3 V |
| 649 | 1,251 | 431 | 427 | 424 | -222 V | ±4 V |
| 673 | 2,158 | 452 | 447 | 443 | -225 V | ±4 V |
| 695 | 3,723 | 472 | 468 | 463 | -228 V | ±4 V |
| 718 | 6,424 | 493 | 488 | 483 | -230 V | ±5 V |
| 740* | 11,084 | 513 | 508 | 503 | -232 V | ±5 V |

Same table in **field**, which is the form that travels to a detector with a different amplification gap. Equal gain means equal effective Townsend coefficient, and that condition has no gap in it — so divide out the 150 µm gap these numbers were simulated at, and multiply back in by whatever gap the other detector has. See `mm_gap_scaling.py` for how far that actually holds (a few volts between 128 and 150 µm; it degrades outside that).

| E(ref) kV/cm | gain | Ne/CF₄/C₂H₆ 80/10/10 r=0.40 kV/cm | Ne/CF₄/C₂H₆ 80/10/10 r=0.50 kV/cm | Ne/CF₄/C₂H₆ 80/10/10 r=0.60 kV/cm |
|---|---|---|---|---|
| 27.10 | 9 | 16.14 | 16.08 | 16.00 |
| 29.14 | 16 | 17.55 | 17.47 | 17.38 |
| 31.10 | 27 | 18.96 | 18.86 | 18.76 |
| 32.99 | 47 | 20.37 | 20.25 | 20.13 |
| 34.83 | 82 | 21.77 | 21.63 | 21.50 |
| 36.61 | 141 | 23.17 | 23.01 | 22.86 |
| 38.34 | 244 | 24.56 | 24.38 | 24.21 |
| 40.03 | 420 | 25.95 | 25.75 | 25.56 |
| 41.67 | 725 | 27.34 | 27.11 | 26.90 |
| 43.27 | 1,251 | 28.73 | 28.48 | 28.23 |
| 44.84 | 2,158 | 30.11 | 29.83 | 29.57 |
| 46.37 | 3,723 | 31.49 | 31.19 | 30.89 |
| 47.86 | 6,424 | 32.86 | 32.54 | 32.21 |
| 49.33 | 11,084 | 34.23 | 33.88 | 33.52 |

Closed-form linear map, `V_target = m·V_ref + c` (from G = A·e^(B·V) on each curve):

| variant | m | c (V) | max resid vs table |
|---|---|---|---|
| Ne/CF₄/C₂H₆ 80/10/10 r=0.40 | 0.8097 | -93.3 | 7.6 V |
| Ne/CF₄/C₂H₆ 80/10/10 r=0.50 | 0.7963 | -88.5 | 7.5 V |
| Ne/CF₄/C₂H₆ 80/10/10 r=0.60 | 0.7824 | -83.4 | 7.3 V |

