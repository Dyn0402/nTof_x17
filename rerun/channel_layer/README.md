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

## Is the mask implemented right? (2026-10-26 review)

Dead channels are already the wildcard: `wft.model.prep_plane` censors their samples (out of the chi2 and the dof,
noise 1e9 so the saturation penalty cannot pull either). The review, scripts in the session scratch:

- **The census is right.** Fitted without the masked channel, the model's prediction for it against what it read,
  r = Σ W·M / Σ M² over on-track events: det2 X 6/6 channels r < 0.25, det4 X 22/25 (median 0.10), det6 19/19 (0.01),
  det7 X shorts 94/102 (0.00; the groups are up to ~20 wide, so each channel carries ≤ 1/n), det3 ch 307 0.01. The
  two-channel shorts read r ≈ 0.5 (det3 398/400 0.46, det2 Y 0.51) = half the pair's sum, as a 2-way short should.
- **The det4/det6 "not adopted" verdicts are noise.** Restricted to windows that contain a masked channel (where
  anything can change), every det4/det6 shift is < 1.5σ and changes sign between the phys and profile arms (e.g. det4 X
  s68 +0.042 ± 0.037 vs −0.007 ± 0.041). The χ²/dof rise is bookkeeping: the censored samples carried less χ² per sample
  than the event mean (det4 X: 1.3 M over 67 k samples ≈ 20, against ~220), so dropping them raises the ratio with the
  fit unchanged.
- **The wildcard interpolates without bias.** Injection on det3 (600 events/view): censor a live interior strip, refit.
  Data/prediction on it: median 1.00 (X) / 0.95 (Y), IQR 0.78–1.29 (real strip-to-strip charge fluctuation), no
  runaway (p95 2.2). One censored interior strip moves the angle 0.44° (p68) and p0 0.09 mm (median), which is the
  information the strip carried. A smoothness prior on q could only recover the part of that scatter that is not
  physics; not pursued.
- **The gate never tested the production path.** `wft.calibrate.build_cache` cuts windows along the M3 corridor
  (±5 mm), not from seeds. In production a dead strip cannot fire, so a cluster tends to END at a dead run and the
  3-strip pad lands on censored strips: on the M3 windows already 9 % (det7 X) / 6 % (det4 X) have one side's outer 3
  strips all masked. Dead runs reach 19 strips (det7 X) and 12 (det4 X), longer than the 12 mm seed gap.
  `WFT_DEAD_SEED=1` (opt-in, default off = bit-identical; `wft/tests/test_dead_seed.py`): the seed gap does not count
  dead strips and the window pad counts live strips.
- **The production-path A/B is null; the flag stays off.** lxplus DAG `~/x17rerun/deadseed_ab/` (code
  `/eos/user/d/dneff/x17rerun/code_3f6db25`, outputs `/eos/user/d/dneff/x17rerun/Analysis_deadseed/<row>/wft/ds{0,1}/`):
  `wft.cli reco --matched-only --limit 8000` on the golden rows with the masked bundle (det7 promoted physmask; det4
  promoted phys + census mask), flag 0 vs 1, effbudget on the production alignment, compared ray by ray
  (`deadseed_ab/pair.py`). The flag acts (det7 X windows changed 2 105/8 000, fits moved > 1 mm 245; det4 X 2 078 and
  137) but the outcome does not move:

  | active, X+Y, 5 mm | detected | far | rays changing category |
  |---|---|---|---|
  | det7 off → on | 51.08 → 51.06 % | 1.35 → 1.33 % | 79 / 12 214, all but 6 into/out of hw_known (card 1) |
  | det4 off → on | 37.65 → 37.64 % | 1.62 → 1.63 % | 10 / 14 786, mostly into/out of hw_known (dead bands) |

  The windows that change are the ones on known-bad hardware, which the budget excludes anyway; elsewhere a track
  next to a dead run is already found and fitted without the bridge.

## Decision — CLOSED 2026-10-26 (Dylan): one treatment everywhere

- **Census dead + shorts are censored (`dead`) in the bundle of all five chambers.** det4/det6 got the finalize arm under
  the mask (`~/x17rerun/finalize_physmask_det4det6`, code `code_d93c687`; w0/kw moved ≤ 0.005, v unchanged), and
  `promote_phys.py --mask det2,det3,det4,det6,det7 --apply --force` (17:48) put `calib_bundle_physmask` in every
  golden row's `calib_bundle`. Mask sizes X/Y: det2 7/11, det3 3/0, det4 77/12, det6 24/24, det7 114/11.
- **`WFT_DEAD_SEED` is on by default** (d93c687) in `wft.reco` and the beam seeder `ntof_tracking.wft_beam`: dead strips
  do not count toward the seed gap and the window pad counts live strips. A bundle without a `dead` list is unaffected;
  `WFT_DEAD_SEED=0` restores the old seeding. Tables record it in `meta.selection.dead_seed`. The det2/3/7 physmask
  w0/kw were measured before the default changed; the A/B above shows that does not move the result.
- Research scripts that call `seed_candidates` / `extract_window` directly (sept26_prelim_analysis, event displays) get
  the old behaviour unless they pass `dead_pos` / `dead`.
