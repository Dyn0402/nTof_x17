# Channel layer (W3)

Per chamber and night: `channel_census.py` (rerun/seedfit) tables `<det>_<key>.json` (noisy, shorts + groups, bands,
dead, weak, map check). C20 in CHECKS_2026-10-23.md.

## Mask gate (2026-10-26, `mask_gate.sh` locally / `condor/` on lxplus, cluster 4547063 + 4547064)

Mask = census **dead + shorts** as bundle `dead` (censored). Noisy channels not masked (C20: worse on det7), weak not
masked, unit gains. Held-out events [180, 1500) of each golden row (laptop vladder caches), phys and physwide arms with
and without the mask, per-event σ0 profile, `rerun/det4/compare.py` → `results_mask_gate.json`:

| | mask X / Y ch | X far (pp) | X s68 (°) | X χ²/dof | X ladder | Y | verdict |
|---|---|---|---|---|---|---|---|
| det3 | 3 / 0 | −0.2 ± 0.1 | −0.002 ± 0.008 | 42.6 → 41.0 | 36.38 → 36.38 | identical | **adopt** |
| det2 | 7 / 11 | −0.1 ± 0.1 | +0.003 ± 0.006 | 72.0 → 72.0 | 38.57 → 38.56 | neutral | **adopt** |
| det7 | 114 / 11 | **−4.6 ± 0.8** | +0.013 ± 0.031 | **173 → 111** | 37.28 → 37.01 | neutral | **adopt** (w0/kw re-measured) |
| det4 | 77 / 12 | +0.3 ± 0.3 | −0.003 ± 0.027 | 13.7 → 14.7 | 32.63 → 32.52 | s68 +0.012 ± 0.010 | not adopted |
| det6 | 24 / 24 | −0.1 ± 0.2 | +0.010 ± 0.017 | 226 → 231 | 23.68 → 23.68 | χ² 454 → 458 | not adopted |

Adopted chambers get a finalize arm under the mask (`w12/it1/phys_mask/bundle` → `make_pass_dag finalize
--bundle-sub phys_mask/bundle --out-tag physmask` → `wft/calib_bundle_physmask`, w0/kw measured under the mask), and
`rerun/promote_phys.py --mask det2,det3,det7` promotes those; det4/det6 promote `calib_bundle_phys`.
