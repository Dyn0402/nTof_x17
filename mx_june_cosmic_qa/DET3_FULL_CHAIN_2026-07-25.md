# det3 through the full detailed chain — results and improvement notes

> **2026-07-28: hits-chain results.** The clean separation below between
> "depends on the hits tree" and "reads waveforms directly" is exactly the split
> the waveform-first rebuild acts on — the first column is what moves to `wft/`.
> See `../RECONSTRUCTION_BASIS.md`.

**2026-07-25.** `sat_det3` on the significance-floor reco (`DET3_RECO_FIX_2026-07-25.md`),
run through the complete per-detector chain plus the det3-specific scans.

- Driver: `rerun_det3_full.sh` (same steps and flags as `rerun_june_analysis.sh`'s
  `process_key()`, so numbers are directly comparable to `RERUN_RESULTS_20260725_011307.md`)
- Log: `~/x17/cosmic_bench/Analysis/_grand_logs/det3_full_20260725_151347.log`
- **29 steps OK / 0 WARN** (the overnight fleet rerun had 5 WARNs; the two known
  breakages, `39_spark_deadtime` and `44_final_vdrift_plot`, both pass for det3)

---

## 1. What actually changed, and what didn't

The single most useful result of this run is a clean separation of concerns.

| depends on the hits tree (moved) | reads waveforms directly (did not move) |
|---|---|
| `03` alignment, σ_θ, v_drift | `31` micro-TPC metrics |
| `08`/`09`/`12` efficiency | `33` head-on tagger |
| `36` `prod` estimator | `34` hybrid tracking |
| — | `42` time resolution |

`31`/`33`/`34` reproduce their pre-fix numbers to 3–4 significant digits on the
new reco. That is expected, not a bug: they rebuild clusters from `decoded_root`
waveforms with their own pedestal + CNS, so the hit-level significance floor
cannot reach them. `42` likewise reads `combined_hits` directly and never touches
the reco cache — its bit-identical output is correct behaviour, and it also
explains why the handoff's "physics results that did not move" didn't move.

**Correction to `DET3_RECO_FIX_2026-07-25.md` §5:** the "estimator coverage drops
~7 points in `34`" caveat was wrong. It came from an ad-hoc `34` run that omitted
`--veto=50`. With the driver's flags, coverage is *identical* to baseline
(lt5 0.9781, gt8 0.9722) and σ68 matches to the 4th digit. There is no coverage
cost, and the `--sigrel=0.08` fallback recommended on that basis is unnecessary.

## 2. Headline numbers (final state)

```
active-area clean M3 rays: 7119
  reco_near   :  6630  ( 93.1%)      has_any      = 100.0%
  reco_far    :   277  (  3.9%)      within 5 mm  =  93.1%
  spark       :   190  (  2.7%)      reco-at-all  =  97.0%
  hit_no_reco :    21  (  0.3%)      core sigma   =  0.45 mm
  no_hit      :     1  (  0.0%)      median |r|   =  0.76 mm
  spark_frac  =  8.2%
```

σ_θ 1.99 / 1.87° (was 2.42 / 2.60), corr 0.857 / 0.898 (was 0.808 / 0.826).

## 3. `36` position estimators — the clearest win

σ68 X / Y [mm], all bands:

| estimator | before | after |
|---|---|---|
| `prod` production anchor | 0.755 / 0.937 | **0.626 / 0.681** |
| `lead_u` earliest unshared strip | 0.673 / 0.802 | 0.643 / 0.705 |
| `cog_raw` cluster centroid | 1.948 / 2.422 | 1.846 / 2.172 |
| `early_raw` early-charge centroid | 0.689 / 0.862 | 0.617 / 0.674 |
| `fit_t0` track-fit impact point | 0.690 / 0.811 | 0.653 / 0.683 |
| **`combo`** (best) | 0.670 / 0.793 | **0.608 / 0.652** |

Two things worth noting. **`prod` now beats `lead_u`** (0.626/0.681 vs
0.643/0.705), reversing the ordering that motivated writing `36` in the first
place — the production anchor is no longer the weak link. And the waveform-based
estimators improved too, which the floor cannot explain directly: better reco →
better alignment (z 713 → 714/715, offsets refit) → tighter residuals for
*everything* measured against the reference. The alignment is a shared upstream
dependency, so a reco improvement propagates further than the reco itself.

## 4. Charge sharing: `c1` is a tunable, not a measured constant

This was the one genuine surprise, and it cost a false alarm before it was
understood — worth reading before touching `26`'s outputs.

`26` measures det3 FEU 8 (Y plane) at `c1 = 0.519`. `31`/`33` hardcode
`0.432` (labelled "det4 Y measured"). Feeding the *measured* value into the
unsharing kernel makes angle reconstruction visibly **worse**. Scanning it
against downstream angle quality (`cshare_scan.sh`, 6 points × ~3 min):

| c1 | c2 | plateau σ68 | psi68 | pearson | frac dθ<3 | hyb lt5 | hyb gt8 |
|---|---|---|---|---|---|---|---|
| 0.432 | 0.112 *(current default)* | 1.6312 | 3.5392 | 0.9177 | 0.7834 | 1.6308 | 1.7362 |
| **0.432** | **0.152** | **1.6154** | **3.3643** | **0.9202** | **0.7935** | **1.6069** | **1.7013** |
| 0.470 | 0.130 | 1.6549 | 3.6820 | 0.9126 | 0.7773 | 1.6279 | 1.7543 |
| 0.519 | 0.112 | 1.7574 | 5.0880 | 0.9007 | 0.7270 | 1.7396 | 1.8614 |
| 0.519 | 0.152 *(26's measured)* | 1.6900 | 3.9529 | 0.9023 | 0.7630 | 1.6636 | 1.7823 |
| 0.400 | 0.112 | 1.6152 | 3.4248 | 0.9193 | 0.7903 | 1.6283 | 1.7064 |

`c1 = 0.432, c2 = 0.152` wins on **every** metric — better than both the current
default and the measured pair. So `26`'s measured **c2** is an improvement and its
**c1** is not: c1 wants to sit near 0.40–0.43, and degrades monotonically above it.

Why they differ is not established. Plausibly `26`'s near-vertical-lead median
estimates a different quantity than the kernel's effective nearest-neighbour
coupling (it is measured on the raw cluster and the kernel is iterative with
`ALPHA = 0.5` time-sharing), so the two are not interchangeable. **Do not wire
`cache/cshare.json` into the unsharing scripts on the assumption that measured is
better.** I tried exactly that, and it degraded `31`'s psi68 by 33 % and `34`'s
HYBRID gt8 from 1.72 → 1.90 before the A/B isolated it. Reverted; `31`/`33`/`36`
are back at HEAD.

## 5. Improvement notes, ranked

**1 — The raw-multiplicity veto is duplicated in ~11 places and now over-cuts by 11 %.**
The biggest remaining item, and the same root cause as the reco regression.
`03` and `09` were fixed to count strips after the significance floor; every
other script still derives `hits_per_event <= VETO` from raw hits independently:
`14`, `17`, `19`, `22`, `23`, `24`, `29`, `30`, `42`, `hv_scan.py`, plus `10`,
`39`, `40` on `SPARK_THRESH`. Measured on det3:

```
veto50 on RAW      : keeps 27,132 / 32,793 (82.7%)
veto50 on FILTERED : keeps 30,097 / 32,793 (91.8%)   -> +2,965 events (+10.9%)
```

`42` shows the symptom directly: `n_events_dualplane` fell 28,404 → 26,230
(−7.7 %) across the analyzer rework and has not recovered, because the fix never
reached it. Five of these scripts ran in this chain (`14`, `17`, `19`, `23`, `42`),
so their statistics are ~11 % smaller than they should be. **Recommendation:**
centralise into one `cm.load_det_hits(cfg, veto=…, sigrel=…)` helper and have all
of them call it, rather than patching eleven copies.

**2 — Adopt `c2 = 0.152` for det3's Y plane, and scan `c1` per detector.**
Free ~1.5 % on the headline hybrid σ68 (1.6308 → 1.6069) and 4.5 % on psi68.
But the `{feu: (c1, c2)}` keying is not valid fleet-wide — an FEU number does not
identify a detector (FEU 8 is det3's Y plane in this run and det4's in another),
so this needs a per-key table, not another dict edit. `36` is the worst case: it
applies det3's constants to FEU 3/4/6, where det6's measured X is 0.231 vs the
0.449 assumed.

**3 — Retune `36`'s `N_SWITCH = 9`.** The combo estimator switches from
early-charge centroid to the production anchor above 9 strips. That threshold was
chosen when `prod` was the degraded estimator; `prod` is now 17–27 % better, so
the optimal crossover has almost certainly moved. Cheap to scan (`36` is 70 s).

**4 — Consider tightening the spark threshold from 50 to ~25 surviving strips.**
Measured collapse point on filtered multiplicity: within-5mm is 99.2 % up to 20
strips, 82.2 % at 20–25, then 43.7 % at 25–30. The current threshold of 50 admits
a population that reconstructs at <50 %. Tightening cleans the residual tail at
some cost in the `within5` accounting (vetoed crossings stay in the denominator).
Judgement call — flagged, not made.

**5 — `31`'s `eff_seg` is only 0.495.** Barely half of matched rays yield a
usable micro-TPC segment, and this did not improve with the reco fix (it is
waveform-side). Given `36` shows the position side is now in good shape, segment
efficiency is the natural next target for angle performance.

**6 — Re-run the other five detectors.** det2/det6/det7 are still on the broken
reco; the fleet-level steps (`30`, `46`, `46b`, `46c`, `47`, `47b`) and
`build_final_pdf` were deliberately skipped here because they aggregate across
detectors and would mix hit generations.

## 6. State on disk

`cache/cshare.json` holds `26`'s measurement record (`{7: [0.456, 0.056],
8: [0.519, 0.152]}`) — it is the measurement, and `27`/`28` read it as before.
`31`/`33`/`36` are at HEAD and use their hardcoded dicts, so all outputs are
consistent with the committed code. Pre-fix outputs and the new-cshare variants
are preserved in the session scratchpad (`backup_sat_det3/`, `newcshare/`),
along with `cshare_scan.sh` / `cshare_scan.tsv`. Nothing committed or pushed.
