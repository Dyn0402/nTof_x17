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

### CERN_450m

| Mixture | iC₄H₁₀ % | m (slope) | c (V) | max resid (V) |
|---|---|---|---|---|
| 98/2 | 2 | 1.0218 | -64.8 | 2.4 |
| 95/5 | 5 | 1.0000 | +0.0 | 0.0 |
| 90/10 | 10 | 1.0027 | +71.2 | 0.3 |
| 85/15 | 15 | 1.0618 | +107.3 | 4.8 |
| 80/20 | 20 | 1.1639 | +123.4 | 18.0 |
| 75/25 | 25 | 1.2868 | +131.2 | 41.9 |

## Lookup table (accurate, quadratic-fit gain match)

Equivalent mesh voltage (V) to reach the same gain as 95/5 at the given V(95/5). `*` = extrapolated beyond the mixture's simulated voltage range.

### Saclay_160m

| V(95/5) | G(95/5) | 98/2 | 95/5 | 90/10 | 85/15 | 80/20 | 75/25 |
|---|---|---|---|---|---|---|---|
| 400 | 2,617 | 345* | 400 | 475 | 537 | 591 | 640* |
| 410 | 3,595 | 354* | 410 | 485 | 548 | 602* | 651* |
| 420 | 4,939 | 364* | 420 | 495 | 558 | 612* | 661* |
| 430 | 6,787 | 374* | 430 | 505 | 568 | 622* | 671* |
| 440 | 9,330 | 383* | 440 | 515 | 578 | 632* | 681* |
| 450 | 12,827 | 393* | 450 | 525 | 589 | 642* | 691* |
| 460 | 17,639 | 403 | 460 | 535 | 599 | 652* | 701* |
| 470 | 24,262 | 413 | 470 | 545 | 609* | 661* | 710* |
| 480 | 33,379 | 423 | 480 | 555 | 618* | 671* | 720* |
| 490 | 45,931 | 434 | 490 | 565 | 628* | 680* | 729* |

### CERN_450m

| V(95/5) | G(95/5) | 98/2 | 95/5 | 90/10 | 85/15 | 80/20 | 75/25 |
|---|---|---|---|---|---|---|---|
| 400 | 3,052 | 346* | 400 | 473 | 533 | 586 | 633* |
| 410 | 4,194 | 356* | 410 | 483 | 543 | 597 | 643* |
| 420 | 5,763 | 366* | 420 | 493 | 554 | 607* | 653* |
| 430 | 7,918 | 376* | 430 | 503 | 564 | 617* | 663* |
| 440 | 10,878 | 385* | 440 | 513 | 574 | 627* | 673* |
| 450 | 14,943 | 395* | 450 | 523 | 584 | 637* | 682* |
| 460 | 20,525 | 405 | 460 | 533 | 594 | 647* | 692* |
| 470 | 28,189 | 415 | 470 | 543 | 603* | 657* | 701* |
| 480 | 38,711 | 426 | 480 | 552 | 613* | 666* | 711* |
| 490 | 53,155 | 436 | 490 | 562 | 623* | 676* | 720* |

## Notes

- Gain model is per-mixture `ln G = a + b·V + c₂·V²` (R² ≥ 0.997); the closed-form linear map above uses the single-exponential fit and agrees with the table to within the listed residual inside the reference range.
- 95/5 is only simulated to 490 V, so the reference does not extrapolate; the equivalents for 80/20 and 75/25 (and the low-voltage end of 98/2) *do* extrapolate and should be treated as indicative.
- Two pressure conditions are reported (Saclay 160 m ≈ 746 Torr, CERN 450 m ≈ 721 Torr); pick the one matching the operating site.
- Regenerate with `python3 mm_hv_equivalence.py` after refreshing the Ar/iC4H10 quencher-scan JSONs in `results/`.
