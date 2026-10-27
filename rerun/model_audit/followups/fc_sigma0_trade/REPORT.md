# FC — FOLLOWUPS items 5 (σ0 X vs Y) and 6 (width–tilt after all corrections), 2026-10-27

(Written by the coordinating session from the workstream's final message; the workstream could not write this file.
Commit cef45df.)

## Item 5 — one σ0 for both views: 0.30 ± 0.02 mm
"Y 0.32 vs X ≤ 0.29" was not a view difference: σ0_X trades with Dp_X (Dp_X forced to Y's value → σ0_X 0.38; Dp_X free
→ 0.29).

SPS det4 head-on RAW run_71, 800 clean views/view/field, Y RC free, X with and without a bounded across-strip RC term
(errors: event bootstrap; χ²/dof 8–9 is model-limited, so Δχ²/dof scaling would overstate):

| field | σ0 X (prod X model) | σ0 Y | Y − X (with X term) | common σ0 | cost of one σ0, % χ² (prod X / X term) |
|---|---|---|---|---|---|
| 243 V/cm | 0.332 ± 0.023 | 0.293 ± 0.011 | −0.026 ± 0.024 | 0.295 | 0.10 / 0.001 |
| 156 V/cm | 0.285 ± 0.022 | 0.288 ± 0.011 | −0.015 ± 0.024 | 0.287 | 0.0 / 0.05 |
| 96 V/cm | 0.331 ± 0.018 | 0.299 ± 0.006 | −0.020 ± 0.019 | 0.297 | 0.12 / 0.002 |

The X across-strip term improves X χ² ~1 % at every field and brings Dp_X − Dp_Y to +0.001…+0.003 (most of the "X widens
at 2× Magboltz" anomaly). Adding the one-sided copy moves σ0_X −0.03…−0.05 (remaining X ambiguity). A single shared
sheet D for both views costs 0.6–1.1 % χ² (disfavoured): across-strip spread 1e-4 vs along 3e-4 mm²/ns.

Bench head-on (|tan| < 0.05, unsaturated) Y − X: det2 +0.06 ± 0.03, det3 +0.018 ± 0.006, det4 +0.07 ± 0.06, det6 −0.10 ±
0.06, det7 0.00 ± 0.04 → mixed signs; one σ0 costs 0.05–0.6 % χ². Bench σ0 0.19–0.27 on small samples (49–289 views) —
below SPS, unexplained.

**Physics** (stack-up in `docs/detector_spec/spec.py`, `stackup_induction.py`): both views read the same L4 pads (0.68
mm on 0.78 mm pitch, ~70 µm under the resistive strips), routed by vias to X (L6) / Y (L5) strips → readout depth has no
lever. Static induction (grounded pads, 70.5 µm kapton, 150 µm gas): 0.07 mm Gaussian-equivalent; avalanche ~0.025 mm →
truly prompt ≤ 0.1 mm. **The fitted 0.3 mm is mostly resistive-sheet spreading within the first 60 ns sample**
(2Ds = 0.026–0.075 mm² → σ 0.16–0.27 mm with each chamber's D). That is the only view-dependent part: along the strips (Y)
it keeps growing (RC kernel); across (X) it stops at the strip edge (w²/12 = 0.025 mm² for 550 µm, reached in 20–57 ns).
Production has no X term, so σ0_X and Dp_X shared it. The physical X ≠ Y is bounded (σ0 0.30 → 0.34) and would make X
*wider*; with it modelled, data show none. Pitch quantisation (800 µm strips over 780 µm pads) is the same bounded box;
its 31.2 mm beat is not seen (cloud_basics §42).

## Item 6 — the trade survives at the ~1 % level
1320 held-out events/chamber, t0 pinned at the latency, per-event σ0 off; fixed σ0 arms 0.24/0.31/0.38/0.45 (both views)
and X 0.27/Y 0.33; bow map measured on half (eid parity) applied to the other; 20 ns arm K = 72. "Before" = audit
production arm, same events. C = steep scale (|tan| 0.22–0.45) / shallow (0.08–0.16) − 1; B = brightest / faintest third − 1.

| view | C before → after % | B before → after % | s68 before → after ° |
|---|---|---|---|
| det2 X / Y | −0.2 → +0.1 / +1.5 → −1.1 | −0.9 → 0.0 / +3.0 → +5.7 | 1.10 → 1.01 / 1.33 → 1.07 |
| det3 X / Y | 0.0 → +0.1 / −0.1 → −0.5 | −0.6 → +0.6 / +1.2 → +1.7 | 1.09 → 1.03 / 1.04 → 0.88 |
| det4 X / Y | −2.0 → −1.4 / −2.4 → −4.1 | −5.4 → +2.0 / −1.6 → +1.6 | 1.53 → 1.61 / 1.58 → 1.35 |
| det6 X / Y | −7.7 → −2.2 / −0.9 → −1.5 | −8.0 → −3.2 / −7.5 → +0.1 | 1.70 → 1.45 / 1.89 → 1.41 |
| det7 X / Y | −2.3 → −1.7 / −2.5 → −4.5 | −6.5 → −5.6 / −12.2 → −4.0 | 1.52 → 1.48 / 1.66 → 1.36 |

(±1–3 % on C, B.) Residual compression common −1.2 ± 0.5 % (χ² 7.7/9 about the mean); det4 Y −4.1 ± 2.1 and det7 Y −4.5 ±
2.1; det4 Y moved with nothing (its "7 %" reads 4 ± 2 here); the bow map removed det6 X's −8 %. Brightness: the fixed σ0
removed most of it (det7 Y −12 → −4, det6 Y −7.5 → 0, det4 X −5.4 → +2); t0 pin alone did nothing; left: det6 X −3.2 ± 2.3,
det7 X/Y −5.6/−4.0 (pooled ~−4 ± 1.3 % on the two chambers saturating 70–85 % of events); det2 Y opposite +5.7 ± 2.0.
The trade itself: σ0 0.24 → 0.45 lowers the global scale 3.5–4 % (det4 X 7.5 %; kw absorbs on the bench) and raises C by
2–9 points; C = 0 at σ0 0.31–0.38 on det2/det3/det4 X, ≥ 0.45 or never on det4 Y/det6/det7 Y → no single σ0 zeroes it
everywhere. Two σ0 vs one: angles equal (X 0.02–0.04° worse with 0.27 on det3/det4). 20 ns: s68 −1…−3 % det2/3/4,
neutral det6/7, C/B unchanged, χ²/dof −3…−10 %. Pinned vs free t0: angles, C, B unchanged. Fixed σ0 vs production + bow
map: 7/10 views better or equal (det6 Y 1.48 → 1.41°); det3 X, det4 X −0.05…−0.06° worse, det2 Y −0.02°.

## Spec for FE
1. `sigma_p0_x = sigma_p0_y = 0.30` in every bench and n_TOF bundle; drop `sigma_p0_prof_x/y`; quote ±0.02 (SPS) and the
   bench 0.19–0.26 as systematic (~1.1 % angle scale per 0.07 mm, absorbed by kw/v on the bench).
2. Gate pinned t0 + FB's bow `angle_map` together on full finalize tables (caches hold ~20 tracks/view above |tan| 0.28).
3. 20 ns: `model.DT = 20`, K × 3 after `use_calibration` (`fc_arm.py`); ~3× cost; gate on n_TOF and SPS 25.6° tracks.
4. X across-strip term (`rcb_D_x`, `fc_headon_grid.py`) study override only (strip width / insulator thickness "open").
5. Systematics: −1.2 % compression, −4 % det6/det7 bright tracks, −4 % det4 Y.

## Risks
SPS = det4 only, other gas; bench σ0 below SPS unexplained (small, unsaturated-selected samples); 550 µm / 70.5 µm geometry
"open" in the spec; 1320 events/chamber → single-view residuals < ~4 % not significant; bow map from the same arm's
events (half cross-validation); "before" arms ran older code with the per-event choice on.

## Files
Scripts: fc_headon_grid.py, analyse_fc_headon.py, common_sigma0.py, stackup_induction.py, fc_arm.py, make_fc_dag.py,
trade_analysis.py, make_fc_figures.py. Results: results/{fc_headon,common_sigma0,stackup_induction,fc_trade}.json.
Figures (force-added; captions in figures/captions.txt): fc_sigma0_xy, fc_sigma0_profiles, stackup_mechanisms,
fc_trade_response, fc_trade_sigma, fc_trade_arms. EOS `/eos/user/d/dneff/x17rerun/followups/fc/{trade,headon}/`,
mirror `~/x17/cosmic_bench/model_audit/fc/`.
