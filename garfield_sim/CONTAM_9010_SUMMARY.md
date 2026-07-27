# Ar/iso 90/10 contaminated-gas Magboltz suite — July beam drift-velocity deficit

**Date:** 2026-07-20. **Motivation:** `ntof_july_analysis/run58_scan/analyze_drift.py`
measures, on the clean Det A (nominal 30 mm gap), v_drift ≈ 35.6–35.8 µm/ns at
E = 200–233 V/cm — **~12–16 % below pure Ar/iso 90/10 Magboltz** (40.5 / 42.6) —
and the curve plateaus where the pure mix keeps climbing. run58 only had a bench
*95/5* curve baked in and explicitly flagged the need for a 90/10 curve. This is
the 90/10 analogue of the June 95/5 contamination trio.

## What was run
`mm_one_mixture.py` × 15 as independent HTCondor jobs (cluster 11814512) on
lxplus under LCG_107, **Ar/iso 90/10 base, CERN 720.8 Torr, 293 K, ncoll=5**,
v(E) + attachment η(E) + diffusion over E = 40–500 V/cm. Merged to
`results/drift_9010_contam_cern.json`; plots `drift_9010_contam_cern.png`
(v vs E) and `drift_9010_contam_attachment.png` (charge surviving 30 mm).
(An earlier single 15-way `multiprocessing.Pool` job deadlocked when one O2
worker died — the all-or-nothing design was replaced by one-job-per-mixture.)

## Result — matching the ~12 % velocity deficit (v@200 = 35.6 vs pure 40.5)

| candidate | v@200 (µm/ns) | RMS dev | η@200 (cm⁻¹) | charge surviving 30 mm |
|---|---|---|---|---|
| pure 90/10 | 40.5 | 5.9 | 0 | 100 % |
| **+0.16 % H₂O** (interp) | 35.6 | ~4 | 0 | **100 %** |
| +0.3 % H₂O | 31.6 | 4.5 | 0 | 100 % |
| +0.5 % H₂O | 25.7 | 9.8 | 0 | 100 % |
| **+3 % N₂** (interp) | 35.6 | ~2.1 | 0 | **100 %** |
| +5 % N₂ | 32.9 | 2.1 | 0 | 100 % |
| +1 % air | 38.8 | 4.3 | 0.90 | 6.7 % |
| +2 % air | 37.3 | 2.9 | 1.88 | 0.35 % |
| +3 % air | 35.9 | 1.8 | 3.03 | **0.011 %** |
| +0.5 % O₂ | 39.9 | 5.3 | 2.09 | 0.19 % |
| +1.0 % O₂ | 39.3 | 4.8 | 4.21 | 0.0000 % |

## Conclusions

1. **The deficit is real but small.** Confirmed against a proper 90/10 CERN
   Magboltz curve (didn't exist before). ~12 % low at 200 V/cm.

2. **Water is a very potent suppressant at 90/10** — only **~0.2 % H₂O** explains
   the deficit; ≥0.5 % overshoots wildly (v halves). So if it *is* water the
   detector is nearly dry — the long flush worked. (Contrast June det3, which
   fit ~1 % H₂O; that much water here would give v ≈ 16, not 35.6.)

3. **Air-in-line and O₂ are EXCLUDED by attachment.** Any air (≥2 %) or O₂ (≥0.5 %)
   fraction large enough to bend v down to the measured value attaches at
   η ≈ 2–5 cm⁻¹, so **essentially no charge survives the drift from the cathode
   half of the gap** (0.01–0.4 %). That is inconsistent with the observed
   full-depth micro-TPC tracks. The good v-fit of +3 % air (RMS 1.8) is a
   coincidence its own attachment kills. → **not an air leak, not O₂.**

4. **Surviving explanations (zero attachment, charge preserved):** trace water
   (~0.2 %) or a few-% inert N₂. Physically, slow **water outgassing/permeation**
   from detector materials after long flushing is the textbook culprit and needs
   only ~0.2 %; free N₂ has no natural source (air ingress brings O₂ too, which
   is excluded). **Best answer: residual water outgassing at the few-tenths-%
   level — exactly the worry, but far milder than June.**

## The decisive test — DONE on the run58 data (`attachment_run58.py`)
η is the discriminator. Measured **mean clean-hit amplitude vs drift depth** on
run58 Det A from the cached `driftspec` (sum_amp/n_clean per 20 ns bin,
time→depth via the measured v_drift), at drift 700/600/500 V. Result:

| drift | E (V/cm) | data λ | **A(cathode)/A(anode)** | air/O₂ would give |
|---|---|---|---|---|
| 700 | 233 | −64 mm (flat/rising) | **1.95** | +1% air → 0.07 ; +0.5% O₂ → 0.007 |
| 600 | 200 | −92 mm | **2.38** | +1% air → 0.07 ; +0.5% O₂ → 0.002 |
| 500 | 167 | −126 mm | **1.81** | +1% air → 0.04 ; +0.5% O₂ → 0.001 |

**The data amplitude is FLAT (actually rises ~2× to mid-gap from micro-TPC
geometry, then flat) across the full 30 mm — zero attenuation toward the
cathode.** Air/O₂ at the level needed to bend v down would attenuate the
cathode-side charge to 0.1–7 % of the anode; the data shows ~200 %. Plot
`attachment_run58.png`: the points sit on the green η=0 (dry) line and reject
the red (+1% air) and orange (+0.5% O₂) curves by 1–3 orders of magnitude at
depth. → **air-in-line and O₂ are experimentally excluded; the velocity deficit
is a no-attachment slow-down = trace water (~0.2%, outgassing) or inert.**
Confirms the sim's discriminator on the real detector. (Caveat: v_measured
assumes exactly 30 mm; a smaller effective gap would shrink the apparent
deficit and the required water further.)
