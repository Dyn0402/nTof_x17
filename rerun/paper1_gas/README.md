# Bench gas against time and field (Paper I §6) — pass 1, 2026-10-31

**Verdict: the bench drift velocity is the water.**
- **Field:** one water describes det3's whole 6-27 drift scan.
- **Time:** the in-row v(t) drifts are the water drying out overnight.
- **Chambers:** each chamber carries its own water.
- **det6's ladder-vs-column disagreement is attachment.** For the first ~4 h of 6-26, an electronegative contaminant truncates
  det6's drift column. The M3 ladder does not see it; the timing estimators (stack, column duration) do.

Report: `report.html` (figures in `figures/`; a copy with figures is on EOS
`/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/paper1_gas/report/`). Numbers: `results/summary.json`, `results/gas.json`.

## Numbers

**Water per chamber**: timing v = gap / T of the head-on stack. S4 re-check recipe: pass template, bundle dead list ∪
strip_gain_audit flags, M3 reference ≥ 30 mm inside, every unsaturated amplitude. Magboltz bench Ar/iso 95/5 + water, fine
grid `air_merged`. Errors: bootstrap T ⊕ 1.5 %.

| chamber | row | E (V/cm) | v (µm/ns) | water (%) | gap ± err band |
|---|---|---|---|---|---|
| det2 | o22_long_det2 (6-22) | 321 | 38.25 ± 0.83 | 0.69 [0.63, 0.75] | 0.64–0.74 |
| det3 | 6-27 drift scan, 500–1100 V | 177–390 | 4 fields | **0.97 [0.95, 0.99]**, χ² 2.5/3 | gap profile 29 [28, 30] mm |
| det4 | g_det4 (6-24) | 197 | 32.70 ± 0.67 | 0.42 [0.40, 0.45] | 0.39–0.45 |
| det6 | g_det6_long (6-26 06–10 h) | 250 | 24.78 ± 0.86 | 1.00 [0.96, 1.04] | 0.91–1.09 |
| det7 | g_det7_long (6-26 06–10 h) | 238 | 35.90 ± 0.64 | 0.48 [0.45, 0.51] | 0.38–0.58 |

The pre-pass det3 value (FINDINGS §40, 0.955 %) is reproduced on pass 1. The angle-scale v (v_drift × kw) of every drift-scan
field, 33–367 V/cm, sits on the same Magboltz curve (`figures/gas_vE_timing.png`).

**6-26, time-resolved.**
- **The anchor:** timing stack / ladder on the attachment-free rows (≥ 06:00 and the afternoon HV scan):
  - det6 k = 0.970 ± 0.013 (χ² 6.2/7);
  - det7 k = 0.984 ± 0.011 (χ² 7.4/7).
- **Water from the anchored ladder:** det6 **1.47 → 0.76 %**, det7 **0.68 → 0.35 %** over 01:00 → 15:00.
- **One time shape:** the two excesses over their afternoon plateaus correlate at r = 0.975. det6's is 2.17× det7's.
  - The ×2.8 in v (vt_within_rows) is partly Magboltz nonlinearity: v is steeper in water at det6's wetter point.
  - The plateaus differ (0.76 vs 0.35 %), so the two chambers do not share one water.

**6-26 attachment (det6):** the visible fraction of the drift column, T / T_true with T_true = gap / (k v_ladder):

| hours on 6-26 | 1.2 | 1.5 | 1.8 | 2.3 | 3.0 | 3.7 | ≥ 6 |
|---|---|---|---|---|---|---|---|
| T / T_true | 0.44 | 0.45–0.53 | 0.61 | 0.72 | 0.81–0.84 | 0.93 | 0.92–1.03 (one 51-event HV row 0.88) |
| τ (ns), T / ln 2 | 1070 | 1100–1240 | 1400 | 1580 | 1700–1750 | — | — |
| O₂-equivalent (ppm) | 600 | 520–590 | 460 | 410 | 370–380 | — | — |

det7 is truncated only in its first row (0.86 at 01:30). This is what made the column duration and the stack disagree with the
ladder on det6 (rerun/vt_all/CROSSCHECK_2026-10-31.md): early in the night the column reads short (fast), and later it lengthens as
the attachment clears. The ladder is the right v estimator there. The attachment clears by ~4 h, while the water relaxes over ~10 h:
two different species. This matches the early-night gain ×25 recovery seen before (vt_within_rows).

**Other nights** (stack thirds; `figures/nights.png`):
- **det4:** 0.70 → 0.21 % through 6-23 (14 h); then 0.41–0.44 % on 6-24. This is the +13 % T step the cross-check saw at
  the det3+det4 → det4-alone change.
- **det2:** 0.73 → 0.57 % on 6-22. Its water is weakly determined: 321 V/cm is near the velocity maximum.

## Method
- **Stacks:** `condor/pg_stack.sh` (60 rows, S4 `s4_stacks.py`, code `code_75287c0`). Rows without their own stage-1 products take
  the align_key row's alignment and the calib_key row's bundle (`condor/make_jobs.py`).
- **Fit:** `pg_specs.py` + `rerun/calib/s4/s4f_fit.py` (`condor/pg_fit.sh`, code `code_c0ce26e4`). Arms: interior (quoted), all, and
  the interior on event-id thirds of the long rows (event id rises with time within a row). Fits that rail at their starting T
  (det3 300 V: the column does not fit in the window; one det7 third) are dropped.
- **Analysis:** `pg_gas.py <fit dir>`, then `pg_figs.py`, then `make_report.py`.
- **Inversion:** `gasinv.py` turns v at E into water on the fine grid. Near the velocity maximum two waters give one v; the wet
  branch is taken and flagged.
- **Field:** E = V_drift / gap with each chamber's timing-free gap (gaps.GAP_GEOM; det3 28.19 from the S4 re-check).
- **M3 v2 rays:** the golden rows' files were deleted for a while on 10-31 (the EOS move). Every stack here either finished
  before that or was re-run after the restore: a missing file fails the stage, it does not fall back.

## Limitations (no new data can be taken)
- **No bench gas record:** flow, fill, line order and materials are unknown. The cause of the water (outgassing after opening,
  permeation) and of the contaminant is not identified.
- **The attachment τ is model-dependent:** a box-end ≈ half-height assumption, ×2. The stack model has no attachment term, and
  "O₂-equivalent" is a rate, not an identification.
- **The gap is the leading systematic on each absolute water** (the band column). The drift scan alone gives 28–30 mm.
- **g_det4's −9 % in-row drift** (ladder and column) is not resolved by the stack thirds (33.1 / 32.2 / 33.2 ± 1.2–2.9 µm/ns).
- **Short HV-scan rows** (~15 min) fall below 50 interior head-on events and are not fitted. Only 5–6 per chamber are.
- **Bench vs SPS** (item 3): the pass-1 SPS gas fit (rerun/sps_gasfit, another session) had not landed when this was
  written. The comparison below uses the pre-pass SPS compositions (cloud_basics FINDINGS §25).

## Bench vs SPS (one story, pending the pass-1 SPS fit)

| setup | gas | water | O₂ |
|---|---|---|---|
| bench, steady state | Ar/iso 95/5 | 0.35–1.0 % per chamber, drying overnight, stepping up between runs | ≲ 10 ppm (det3 drift scan, FINDINGS §25) |
| bench det6, first ~4 h of 6-26 | same | 1.47 % | an attaching contaminant, 370–600 ppm O₂-eq, cleared by ~4 h |
| SPS CF₄ (run_63/71, pre-pass) | Ar/CF₄/iso 88/10/2 | 1.51–1.59 % | 147–230 ppm, falling overnight |

So both setups carry water at the 0.4–1.6 % level. The bench is mostly free of attachment once flushed. The SPS line carries
O₂ that varies over hours.

## Open questions for Dylan
1. **Which v(E) figure for §6?** Recommendation: `gas_vE_timing` (timing v + angle-scale v + Magboltz). The old `gas_vE`
   (angle-scale only) is kept.
2. **Quote the det6 early-night attachment in §6?** It is the cleanest attachment measurement on the bench, but the τ → ppm
   step carries a ×2 model factor.
3. **Do you remember** whether det6 had just been opened or refilled before 6-26 (the 1.47 % water and the contaminant)?
