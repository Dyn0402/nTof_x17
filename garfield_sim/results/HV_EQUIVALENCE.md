# Ar/iC₄H₁₀ HV equivalence — matching 95/5 gas gain

Maps the mesh voltage of each Ar/isobutane mixture to the voltage of **Ar/iC₄H₁₀ 95/5** that gives the **same simulated gas gain** (Garfield++/Magboltz). Use it to put HV scans in different mixtures on a common footing.

Reference 95/5 voltage is swept over its simulated span **400–490 V**. Mixtures whose match falls outside their own simulated range are flagged `*` (extrapolated — larger uncertainty; this happens for the high-isobutane mixtures, which need much higher HV than was simulated).

## Analytic map (closed form)

Each gain curve is ≈ exponential, `G = A·exp(B·V)`, so equal gain gives a **linear** voltage map

```
V_equiv = m · V(95/5) + c
```

with `m = B_ref/B_mix` and `c = ln(A_ref/A_mix)/B_mix`. Coefficients per mixture and pressure (`resid` = max deviation of this linear form from the accurate quadratic-fit lookup over the reference range):

### Saclay_160m

| Mixture | iC₄H₁₀ % | m (slope) | c (V) | max resid (V) |
|---|---|---|---|---|
| 98/2 | 2 | 1.0279 | -69.9 | 3.5 |
| 95/5 | 5 | 1.0000 | +0.0 | 0.0 |
| 90/10 | 10 | 1.0135 | +69.0 | 0.5 |
| 85/15 | 15 | 1.0799 | +104.3 | 5.2 |
| 80/20 | 20 | 1.1879 | +119.5 | 21.2 |
| 75/25 | 25 | 1.3365 | +121.0 | 47.0 |
| 93/5/2 r=0.30 | 2 | 1.1406 | -16.8 | 0.6 |
| 93/5/2 r=0.40 | 2 | 1.0388 | -17.7 | 0.3 |
| 93/5/2 r=0.50 | 2 | 0.9506 | -15.2 | 1.3 |

### CERN_450m

| Mixture | iC₄H₁₀ % | m (slope) | c (V) | max resid (V) |
|---|---|---|---|---|
| 98/2 | 2 | 1.0218 | -64.8 | 2.4 |
| 95/5 | 5 | 1.0000 | +0.0 | 0.0 |
| 90/10 | 10 | 1.0027 | +71.2 | 0.3 |
| 85/15 | 15 | 1.0618 | +107.3 | 4.8 |
| 80/20 | 20 | 1.1639 | +123.4 | 18.0 |
| 75/25 | 25 | 1.2868 | +131.2 | 41.9 |
| 93/5/2 r=0.30 | 2 | 1.1175 | -6.1 | 0.6 |
| 93/5/2 r=0.40 | 2 | 1.0266 | -11.2 | 0.6 |
| 93/5/2 r=0.50 | 2 | 0.9542 | -16.4 | 2.7 |

## Lookup table (accurate, quadratic-fit gain match)

Equivalent mesh voltage (V) to reach the same gain as 95/5 at the given V(95/5). `*` = extrapolated beyond the mixture's simulated voltage range.

### Saclay_160m

| V(95/5) | G(95/5) | 98/2 | 95/5 | 90/10 | 85/15 | 80/20 | 75/25 | 93/5/2 r=0.30 | 93/5/2 r=0.40 | 93/5/2 r=0.50 |
|---|---|---|---|---|---|---|---|---|---|---|
| 400 | 2,617 | 345* | 400 | 475 | 537 | 591 | 640* | 440 | 398* | 366* |
| 410 | 3,595 | 354* | 410 | 485 | 548 | 602* | 651* | 451 | 408* | 376* |
| 420 | 4,939 | 364* | 420 | 495 | 558 | 612* | 661* | 462 | 418* | 385* |
| 430 | 6,787 | 374* | 430 | 505 | 568 | 622* | 671* | 474 | 429 | 394* |
| 440 | 9,330 | 383* | 440 | 515 | 578 | 632* | 681* | 485 | 439 | 403* |
| 450 | 12,827 | 393* | 450 | 525 | 589 | 642* | 691* | 496 | 450 | 413* |
| 460 | 17,639 | 403 | 460 | 535 | 599 | 652* | 701* | 508 | 460 | 422 |
| 470 | 24,262 | 413 | 470 | 545 | 609* | 661* | 710* | 519 | 471 | 432 |
| 480 | 33,379 | 423 | 480 | 555 | 618* | 671* | 720* | 531* | 481 | 441 |
| 490 | 45,931 | 434 | 490 | 565 | 628* | 680* | 729* | 543* | 491 | 451 |

### CERN_450m

| V(95/5) | G(95/5) | 98/2 | 95/5 | 90/10 | 85/15 | 80/20 | 75/25 | 93/5/2 r=0.30 | 93/5/2 r=0.40 | 93/5/2 r=0.50 |
|---|---|---|---|---|---|---|---|---|---|---|
| 400 | 3,052 | 346* | 400 | 473 | 533 | 586 | 633* | 441 | 399* | 363* |
| 410 | 4,194 | 356* | 410 | 483 | 543 | 597 | 643* | 452 | 409* | 373* |
| 420 | 5,763 | 366* | 420 | 493 | 554 | 607* | 653* | 463 | 420 | 383* |
| 430 | 7,918 | 376* | 430 | 503 | 564 | 617* | 663* | 474 | 430 | 393* |
| 440 | 10,878 | 385* | 440 | 513 | 574 | 627* | 673* | 485 | 441 | 403* |
| 450 | 14,943 | 395* | 450 | 523 | 584 | 637* | 682* | 497 | 451 | 413* |
| 460 | 20,525 | 405 | 460 | 533 | 594 | 647* | 692* | 508 | 461 | 422 |
| 470 | 28,189 | 415 | 470 | 543 | 603* | 657* | 701* | 519 | 471 | 432 |
| 480 | 38,711 | 426 | 480 | 552 | 613* | 666* | 711* | 531 | 482 | 442 |
| 490 | 53,155 | 436 | 490 | 562 | 623* | 676* | 720* | 542* | 492 | 451 |

## Ar/CO₂/iC₄H₁₀ 93/5/2 — the operating mixture

Garfield++ has **no built-in Penning parameterisation** for this ternary: `EnablePenningTransfer()` returns *false* and would leave the mixture with **zero** Penning transfer while the 95/5 reference runs at r = 0.40. It was therefore simulated at three hand-set transfer probabilities — r = 0.30, 0.40 (central) and 0.50 — and the spread between them is quoted below as the Penning systematic. The central value follows Garfield's own binary parameterisations at this quencher content (Ar/CO₂ gives 0.376 at 7% CO₂, Ar/iC₄H₁₀ 0.400 flat).

### Saclay_160m

| V(95/5) | G(95/5) | V(93/5/2) r=0.30 | V(93/5/2) r=0.40 | V(93/5/2) r=0.50 | ΔV central | Penning spread |
|---|---|---|---|---|---|---|
| 400 | 2,617 | 440 | 398* | 366* | -2 | ±37 V |
| 410 | 3,595 | 451 | 408* | 376* | -2 | ±38 V |
| 420 | 4,939 | 462 | 418* | 385* | -2 | ±39 V |
| 430 | 6,787 | 474 | 429 | 394* | -1 | ±40 V |
| 440 | 9,330 | 485 | 439 | 403* | -1 | ±41 V |
| 450 | 12,827 | 496 | 450 | 413* | -0 | ±42 V |
| 460 | 17,639 | 508 | 460 | 422 | +0 | ±43 V |
| 470 | 24,262 | 519 | 471 | 432 | +1 | ±44 V |
| 480 | 33,379 | 531* | 481 | 441 | +1 | ±45 V |
| 490 | 45,931 | 543* | 491 | 451 | +1 | ±46 V |

Closed form at the central Penning value: `V(93/5/2) = 1.0388 · V(95/5) -17.7` V (max deviation from the table above: 0.3 V).

### CERN_450m

| V(95/5) | G(95/5) | V(93/5/2) r=0.30 | V(93/5/2) r=0.40 | V(93/5/2) r=0.50 | ΔV central | Penning spread |
|---|---|---|---|---|---|---|
| 400 | 3,052 | 441 | 399* | 363* | -1 | ±39 V |
| 410 | 4,194 | 452 | 409* | 373* | -1 | ±40 V |
| 420 | 5,763 | 463 | 420 | 383* | -0 | ±40 V |
| 430 | 7,918 | 474 | 430 | 393* | +0 | ±41 V |
| 440 | 10,878 | 485 | 441 | 403* | +1 | ±41 V |
| 450 | 14,943 | 497 | 451 | 413* | +1 | ±42 V |
| 460 | 20,525 | 508 | 461 | 422 | +1 | ±43 V |
| 470 | 28,189 | 519 | 471 | 432 | +1 | ±44 V |
| 480 | 38,711 | 531 | 482 | 442 | +2 | ±44 V |
| 490 | 53,155 | 542* | 492 | 451 | +2 | ±45 V |

Closed form at the central Penning value: `V(93/5/2) = 1.0266 · V(95/5) -11.2` V (max deviation from the table above: 0.6 V).

`*` = the match falls outside the voltage range actually simulated for the ternary and is an extrapolation of its fit.

## Notes

- Gain model is per-mixture `ln G = a + b·V + c₂·V²` (R² ≥ 0.997); the closed-form linear map above uses the single-exponential fit and agrees with the table to within the listed residual inside the reference range.
- The reference itself is simulated to 490 V at Saclay but only to 480 V at CERN, so the CERN 490 V row is a 10 V extrapolation of the 95/5 fit (small compared with its 80 V fitted span, but it is not measured).
- 95/5 is only simulated to 490 V, so the reference does not extrapolate; the equivalents for 80/20 and 75/25 (and the low-voltage end of 98/2) *do* extrapolate and should be treated as indicative.
- Two pressure conditions are reported (Saclay 160 m ≈ 746 Torr, CERN 450 m ≈ 721 Torr); pick the one matching the operating site.
- Regenerate with `python3 mm_hv_equivalence.py` after refreshing the Ar/iC4H10 quencher-scan JSONs in `results/`.
