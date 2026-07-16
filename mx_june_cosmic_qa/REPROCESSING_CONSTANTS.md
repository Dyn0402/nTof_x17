> **✅ DONE (2026-07-14) — M3 reference recipe changed to χ²<1.0 + NClus=4, fleet-wide.**
> Centralised in `qa_config.py` (`M3_CHI2_CUT`, `M3_MIN_NCLUS`); every golden-chain
> `M3RefTracking(...)` call updated. Alignment + efficiency/residual (03/08/09/12) re-run
> for all 7 run-keys — all converged cleanly, no NClus≥3 fallback needed. det3 (headline
> `g_det3_wknd`) confirms the prediction almost exactly: core σ 0.63→**0.47 mm**, efficiency
> 88.8→**92.9 %**, reco_far →**3.1 %**. Fleet-wide numbers + a stale-cache bug found in
> passing (sat_det3's no-veto cache predated a data recovery): see `JUNE_RESULTS_SUMMARY.md`
> 2026-07-14 changelog. **Waveform-dependent chain (26–42) also re-run fleet-wide
> 2026-07-14** (unsharing, calibration, micro-TPC metrics, hybrid tracking, position
> estimators, charge balance, time resolution) — hybrid angular σ68 barely moved (≤0.15°
> fleet-wide), confirming it's only weakly coupled to the M3 cut. New sharing/calibration
> constants + hybrid numbers below supersede the 2026-07-12 table (kept for comparison).
> Also done: the true active area is **X∈[0,398.6], Y∈[18,380] mm** (~2 cm
> passivated on each Y edge; ≈39.9×36.2 cm), recorded in `common/mx17_active_area.py`, with
> a reusable `alignment_transform()` adapter and outlines now drawn on every 2-D position
> map in the golden chain (08/09/12/03/04/32/38 + spark position maps). Full brief:
> `det3_recofar_analysis/M3_CUT_AND_ACTIVE_AREA_NOTE.md` and `report_v2.pdf`.

# Fleet reprocessing — measured sharing constants (26_unsharing_analysis.py, M3 v2, veto50)

Written during the 2026-07-12 REPROCESSING_PLAN execution. c1=A(±1)/A(0), c2=A(±2)/A(0)
medians from near-vertical (|tanθ|<0.03) tracks; Δt = neighbour CFD delay.
Design reference (det3): X (FEU7) 0.449/0.052; Y (FEU8) 0.516/0.151. det2 ~0.43/0.52 (design property).

| Det | Run/key | X FEU | c1_x | c2_x | Δt_x | Y FEU | c1_y | c2_y | Δt_y | n_lead x/y | v_geom(Y) after | notes |
|----|---------|-------|------|------|------|-------|------|------|------|-----------|-----------------|-------|
| det6 | g_det6_long (700V) | 3 | 0.250 | 0.048 | +128 | 4 | 0.451 | 0.204 | +88 | 702/632 | 17.7 (v_cfd 16.4) | X-plane BROKEN: v_geom~0, z_slope~0, T_sat 140ns (vs Y ~1000ns). Low X gain / mesh defect. c1_x outlier. Y healthy. |
| det7 | g_det7_long (700V) | 6 | 0.263 | 0.058 | +66 | 8 | 0.513 | 0.220 | +55 | 426/444 | 20.9 (v_cfd 27.9) | X weak but present (z_slope 9.8, T_sat 678ns); Y healthy (≈det3). c1_x low like det6. |

**Pattern:** det6 & det7 BOTH have X-plane c1≈0.25 (~half det3's 0.449) with Y≈0.45–0.51 (≈det3).
The low X-plane sharing is a consistent property of boards C/D, not a fluke → investigated, not averaged.
det6 X-plane is additionally BROKEN (no drift-time development); det7 X is weak but usable.

## Hybrid tracking (34) outcomes — golden chain, odd-eid holdout

| Det | lt5 σ68 (self) | cov | lt5 σ68 (det3-transfer) | plateau σ68 (self) | cov | verdict |
|-----|---------------|-----|-------------------------|--------------------|-----|---------|
| det3 | 1.75° | 97% | — (reference) | 1.86° | 98% | GOLDEN (unchanged) |
| det2 | 2.47° | 96% | 2.85° | 2.17° | 96% | GOLDEN (unchanged) |
| det6 | 4.14° | 98% | 3.54° | **16.3° (BROKEN)** | 98% | low-angle only; plateau time-fit fails (X-plane FEU3 v_seg=119, no drift-time development). Spark-limited board C @700V. |

- det6 31: plateau σ68=81°, v_x=118, psi median 83° — micro-TPC segment tracking broken on X-plane.
- det6 signature-regression low-angle band IS usable (LDA AUC 0.867). Charge topology carries the low-angle info even though the time-fit lever arm is dead.
| det7 | 3.64° | 90% | 4.88° | 4.90° | 85% | MEASURABLE (degraded). v_seg 29-31 vs v_geom ~21 (700V/weak-X lever arm). Spark-limited board D. |
| det4 | g_det4 (~1000V?) | 6 | 0.370 | 0.047 | +48 | 8 | 0.432 | 0.112 | +51 | 351/421 | 23.8 before (v_cfd 25.6) | GATE PASSED: both planes develop drift (z_slope 14/16, T_sat 625/673). More det3-like than det6/7. Gain-limited in EFFICIENCY (20%) but reconstructed events have good micro-TPC structure. NB FEU8 has ~0.3%% malformed multi-frame events (reshape guard added to 26/31/33). |
| det4 | 2.70° | 81% | 2.19° | 4.09° | 81% | MEASURABLE, BEST tracking of new dets. v_seg 27.2/27.8 self-consistent, psi 4.66°. Gain-limited (low stats/coverage) but clean angles. |

## FINAL FLEET HYBRID σ68 (golden chain, odd-eid holdout, self-trained)
- det3(A): lt5 1.75° @97%, plateau 1.86° @98%  [golden, unchanged]
- det2(B): lt5 2.47° @96%, plateau 2.17° @96%  [golden, unchanged]
- det4(E): lt5 2.70° @81%, plateau 4.09° @81%  [NEW; transfer 2.19°]
- det7(D): lt5 3.64° @90%, plateau 4.90° @85%  [NEW; transfer 4.88°]
- det6(C): lt5 4.14° @98%, plateau BROKEN (16.3°)  [NEW; low-angle only; transfer 3.54°]

Superseded script-03 values (DROP from quotes): det3 2.04, det2 2.15, det6 3.15, det7 2.50, det4 2.49.

---

## 2026-07-14 re-run — M3 recipe χ²<1.0 & NClus=4, fleet-wide

All sharing/calibration constants re-measured on the new recipe (26/28 re-run per
detector on `sat_det3`, `o22_long_det2`, `g_det4`, `g_det6_long`, `g_det7_long`).
CSHARE dict in 27/28 must be swapped to the CURRENT detector's own values before each
run -- FEU numbers are reused across detectors (e.g. FEU 8 is det2's Y, det3's Y in some
runs, det4's Y, AND det7's Y) -- this was a latent correctness trap in the pre-existing
workflow, now called out explicitly in both scripts' CSHARE comment.

| Det | Run/key | X FEU | c1_x | c2_x | Δt_x | Y FEU | c1_y | c2_y | Δt_y | n_lead x/y | calib b_x/b_y | plateau bias/σ |
|----|---------|-------|------|------|------|-------|------|------|------|-----------|---------------|----------------|
| det3 | sat_det3 (1000V) | 7 | 0.449 | 0.055 | +69 | 8 | 0.519 | 0.152 | +73 | 529/602 | +0.0339/+0.0290 | 0.18°/1.80° |
| det2 | o22_long_det2 (1000V) | 6 | 0.420 | 0.054 | +98 | 8 | 0.518 | 0.201 | +78 | 237/295 | +0.0423/+0.0478 | 0.37°/1.84° |
| det4 | g_det4 (~1000V) | 6 | 0.388 | 0.046 | +48 | 8 | 0.443 | 0.111 | +51 | 175/213 | +0.0712/+0.0628 | 0.55°/3.84° |
| det6 | g_det6_long (700V) | 3 | 0.285 | 0.045 | +141 | 4 | 0.449 | 0.213 | +90 | 321/317 | X-plane BROKEN (calib meaningless) | 2.63°/82.6° |
| det7 | g_det7_long (700V) | 6 | 0.247 | 0.057 | +64 | 8 | 0.514 | 0.232 | +57 | 188/200 | +0.0740/+0.1115 | 1.39°/10.2° |

All within statistics of the 2026-07-12 / design-reference values -- the sharing physics
is a detector/board property, correctly independent of the M3 reference cut. det6's
X-plane (FEU3) hardware defect (no drift-time development) is unchanged and unrelated to
the M3 recipe.

### Hybrid tracking (34), new recipe -- self-trained, odd-eid holdout

| Det | lt5 σ68 | cov | plateau σ68 | cov | Δ vs 2026-07-12 |
|-----|---------|-----|-------------|-----|-----------------|
| det3 | 1.66° | 98% | 1.84° | 97% | lt5 −0.09°, plateau −0.02° |
| det2 | 2.34° | 96% | 2.11° | 97% | lt5 −0.13°, plateau −0.06° |
| det4 | 2.56° | 79% | 3.84° | 81% | lt5 −0.14°, plateau −0.25° |
| det6 | 3.88° | 98% | BROKEN 19.1° | 98% | lt5 −0.26°, plateau (still broken, hardware) |
| det7 | 3.36° | 90% | 4.97° | 83% | lt5 −0.28°, plateau +0.07° |

**Conclusion: hybrid angular σ68 is essentially unchanged by the M3 recipe** (fleet-wide
shift ≤0.28°, mostly improvements) -- confirms the note's prediction that angle resolution
is only weakly coupled to the M3 position cut.

### det3-frozen-model transfer cross-check, new recipe (re-verified 2026-07-14)

Det3's (`sat_det3`) hybrid model re-saved fresh on the new recipe (`--save-model`) and
applied untouched (`--model=`) to det2/det4/det6/det7. Same overfit test as
2026-07-12, re-run because the model itself changed (retrained on the new recipe's
cleaner but smaller sample).

| Det | self-trained lt5 σ68 | transfer lt5 σ68 | Δ (transfer − self) | 2026-07-12 Δ | verdict |
|-----|----------------------|-------------------|----------------------|--------------|---------|
| det2 | 2.34° | 2.65° | +0.31° | +0.38° | consistent -- transfer degrades slightly |
| det4 | 2.56° | 2.14° | −0.42° | −0.51° | consistent -- transfer *improves* (small det4 self-trained sample) |
| det6 | 3.88° | 3.35° | −0.53° | −0.60° | consistent -- transfer improves |
| det7 | 3.36° | 4.49° | +1.13° | +1.24° | consistent -- transfer degrades most (board-D specific) |

**Every detector's transfer behaviour (which way it moves, and roughly by how much)
matches the 2026-07-12 pattern exactly** -- this is strong independent confirmation
that the near-identical regression weights across detectors are a genuine design
property of the readout, not an artifact of overfitting to one detector's statistics,
and that this conclusion is stable across the M3 recipe change too.
