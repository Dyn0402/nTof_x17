# GA — Is the gain map important for reconstruction? (round 3, item 1, 2026-10-28)

## Verdict: no — on any chamber, and not needed before reprocessing
- **The variation in the map gallery is real:** track charge varies ±20–40 % across every chamber, and by up to ×34 on
  det4 X (p90/p10).
- **The forward fit is blind to that scale.** Anything common to a cluster (slower than about ±2 strips) only rescales the
  free per-depth-bin charge profile q.
- **Only two parts can bias the fit:** a footprint-scale gradient, Δtan = (d ln g/dx)·Dp²/(kw·v), which is χ²-blind; and
  strip-to-strip gain inside a cluster. Both are small on data:
  - strip-to-strip gain in the bulk of det3 and det4 is 0.02–0.04 in ln g, consistent with zero given map noise of
    0.05–0.08;
  - a gradient that matters exists only on det4 X (the amplification stripes).
- **Measured maps, cross-validated and paired by event, change angle s68 by zero within errors** on det4 and det3
  (−0.02…+0.04 ± 0.03°). Y χ²/dof gets worse by up to 16 %.
- **FD's ranking "strip gain spread 0.55° on det4" is withdrawn.** FD modelled white strip noise with σ = 0.31. That 0.31
  was the 3 mm change of a 2 mm-smoothed charge map, i.e. the stripes, not strip noise.
- **Recommendation:** leave the gain slot at 1.0 in the unified reconstruction. Put the few reproducible weak or hot
  channels (det4 X 11, det3 3) into the channel-layer lists instead.

## 1. How much the gain varies, and at what scale
**Charge maps** (`ga_chargemap.py`): median ln(q cos θ) for tracks with |tan_ref| < 0.15, in 1 mm bins of M3 position.
Events are split into two independent halves, so signal = √cov(A, B). Signal rms in ln g (noise per half in brackets):

| | > 8 mm | 1.5–8 mm | < 1.5 mm | d ln g/dx at 1.5 mm [1/mm] | p90/p10 |
|---|---|---|---|---|---|
| det2 X / Y | 0.21 / 0.33 | ~0 | ~0 | 0.014 / 0.014 | 2.4 / 2.7 |
| det3 X / Y | 0.16 / 0.15 | 0.03 / 0.05 | ~0 | ~0 | 2.2 / 1.8 |
| **det4 X** | **0.68** | **0.57** | 0.12 (0.26) | **0.22 (0.06)** | **34** |
| det4 Y | 0.32 | 0.13 (0.22) | 0.28 (0.40) | 0.07 (0.10) | 7.5 |
| det6 X / Y | 0.15 / 0.14 | 0.05 | 0.05 | 0.014 / 0.013 | 1.9 / 1.8 |
| det7 X / Y | 0.35 / 0.38 | 0.26 / 0.07 | ~0 | 0.08 / ~0 | 15 / 5.0 |

- **det4 X:** the stripes are constant along the strips (lower-half vs upper-half correlation 0.87).
- **det7 X, x < 200 mm:** this is the known M1 readout fault, not gain.

**In-fit per-channel map** (`ga_gainfit.py`, `ga_robust.py`): the ratio data/model inside production `fit_plane`, on 1500
M3 events split by parity, two iterations. Bulk channels:

| | det3 X | det3 Y | det4 X | det4 Y |
|---|---|---|---|---|
| signal | 0.022 | 0.017 | 0.024 | 0.039 |
| noise | 0.047 | 0.053 | 0.068 | 0.076 |
| A–B correlation | 0.10 | 0.05 | 0.06 | 0.12 |

- Connector blocks are equal to ±1–3 % (agrees with F15).
- Reproducible outliers (|ln g| > 0.5 in both halves):
  - det4 X: ch 94, 125, 196, 376, 400, 435, 439, 455, 472, 498 weak (g 0.27–0.52); ch 274 hot (g 2.55);
  - det3: X ch 62 hot (2.8), Y ch 41 weak (0.31), Y ch 510 dead-like (≈ 0.01).
- The charge maps are lower bounds on the large-scale variation, not a calibration. They are compressed by efficiency,
  threshold and saturation, and a censored dead strip has extrapolated q.

## 2. Mechanism, and the toy (`ga_toy.py`: FD generator + production fit, X view)
- **Gradient:** a = 0.22/mm on det4 gives about 0.15° rms (0.4° at stripe edges); on det2/3/6, a ≤ 0.015/mm gives
  ≤ 0.01°.
- **Toy, det4:** quadrature excess over no gain pattern (±0.05–0.1°), at track angles of

| arm | 0° | 3° | 9° | 16° | 25° |
|---|---|---|---|---|---|
| white σ 0.31 per strip (FD L4) | 0.20 | 0.62 | 0.60 | 0.45 | 0.65 |
| measured stripes | 0.20 | 0.14 | 0.24 | 0.32 | 0.42 |
| 8 mm-smooth part only | 0.10 | 0.09 | 0.14 | 0.16 | 0.17 |
| fit knows the exact map (oracle) | 0.13 | 0.07 | 0.16 | 0.23 | 0.44 |

  - **Even the oracle map recovers only 0.05–0.1° below 16°, and nothing at 25°.** The cost is signal lost in the
    non-amplifying bands, which no map can restore.
  - det2/3/6/7: every arm is within noise.

## 3. Data (no event is corrected with a map made from it)
- **(a) Gradient against M3** (`ga_gradient_test.py`):
  - det4 X: slope of data on prediction k = 0.34 ± 0.12. Applying the correction moves s68 from 1.773° to 1.774°.
  - Other views: predicted rms 0.03–0.10°.
- **(b) In-fit map, cross-validated** (map from half A applied to half B and vice versa; paired bootstrap):

| | det4 X | det4 Y | det3 X | det3 Y |
|---|---|---|---|---|
| full map Δs68 | −0.017 ± 0.032 | +0.042 ± 0.035 | +0.028 ± 0.017 | +0.004 ± 0.017 |
| 4 mm-smooth part only | −0.008 ± 0.014 | +0.005 ± 0.012 | +0.004 ± 0.006 | +0.007 ± 0.006 |
| χ²/dof median | 14.6 → 14.4 | 14.0 → 16.2 | 40.2 → 42.3 | 55.7 → 60.5 |

- **(c) Charge-profile map in the bundle slot** (built from disjoint finalize events, `heldout_arm.py` on condor):
  - det4 X/Y: +0.015/+0.012 ± 0.03°;
  - det3 X/Y: −0.005/−0.009 ± 0.012°.
- **Earlier result agrees:** the Oct-24 `rerun/det4` map gave det4 X −0.06 ± 0.09 and det3 X +0.06 ± 0.04.

## 4. Decision
- **Unified reconstruction:** gain slot stays at 1.0 on all five bench chambers.
- **Weak and hot channels:** add the reproducible ones to the channel-layer lists (weak/dead → `dead`; det4 X ch 274 and
  det3 X ch 62 → `hot`). Gate them with the flank test like the other masks. **Not done.**
- **det4 stripes:** they are part of det4's chamber floor (toy 0.2–0.4°, data ~0.05°). A map cannot recover this; state
  it as a limitation.
- **If a map is ever wanted** (charge or dE/dx, not angles): derive it per chamber × run condition, from the charge
  profile (large scale) plus the in-fit ratio (strip scale), with split-half reproducibility. Never carry a bench map to
  n_TOF.

## What this does not rule out
- **Only det4 and det3 were tested on data** (1500-event caches). det2/6/7 rest on the charge maps and the toy; det7 X is
  confounded by its readout fault.
- **Strip-to-strip structure below ~4 % is not excluded.** The toy says σ = 0.06 costs ≤ 0.1°.
- **The data arms used the promoted physmask bundle, not `fe`.** The gain term is independent of the pin and the bow, but
  this was not re-run on `fe`.
- **The toy covers the X view only**, one mechanism at a time.
- **Not measured:** a full 8000-muon gate with a map, SPS det4, n_TOF.

## Files
- **Scripts** (`rerun/model_audit/followups/ga_gainmap/`): `ga_chargemap.py`, `ga_gainfit.py`, `ga_robust.py`,
  `ga_analyse.py`, `ga_gradient_test.py`, `ga_chargegain.py`, `ga_qgain_compare.py`, `ga_heldout.py`, `ga_toy.py`,
  `ga_figures.py`, `condor/`.
- **Results:** `results/*`.
- **Figures:** `figures/{chargemap_profiles,infit_map_det4,infit_map_det3_condor,gradient_test,toy_s68}.png`.
- **Study bundles** (not production): `~/x17/cosmic_bench/model_audit/ga/<det>/calib_bundle_{qgain,base}`.
- **EOS:** `/eos/user/d/dneff/x17rerun/followups/ga/`.
