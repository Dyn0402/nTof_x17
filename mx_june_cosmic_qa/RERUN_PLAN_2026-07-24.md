# Full June-cosmic analysis rerun on the a1cce79 (matched-filter) hits

**Status: PLANNED, not started.** Execute overnight on the user's signal via
`rerun_june_analysis.sh`. Scope agreed 2026-07-24: **everything** (full golden
chain + side analyses) on the **primary golden runs + HV scans**, run **blind**
(no per-detector hand-tuning of results; measured constants ARE wired through
automatically — see §4).

## 0. Why a plain re-run is not enough (the two traps)

1. **Stale caches.** Script 03 caches per-event reco to
   `Analysis/<run>/<sub>/<det>/cache/event_results{,_veto50}.pkl`; 08/09/12/31/
   33/34/36/38/42 all read those caches. The waveform scripts additionally cache
   `microtpc_segments.csv`, `headon_features.csv`, amp/spark pickles. Every one
   of these is derived from the hits we just replaced. **The driver deletes each
   target run's `cache/` pkls first and passes every rebuild flag**
   (`03 --refit`, `26 --refit`, `31/33/36/40 --rebuild`, `39 --rebuild-amp`,
   `14 --refit`) so nothing survives from the old hits.

2. **Per-detector sharing constants.** `27_unsharing_refinement.py` and
   `28_angle_calibration.py` hardcode `CSHARE = {6:(0.247,0.057),
   8:(0.514,0.232)}` — det7's numbers. Blindly running them applies det7 sharing
   to det2/3/4/6. Script **26 measures c1/c2 per FEU** for the detector at hand.
   The driver captures 26's measured constants and writes them to
   `cache/cshare.json`; 27/28 read that file when present (one-line hook, falls
   back to the hardcoded dict). This keeps the run *blind* (no human tuning) yet
   *per-detector correct*.

## 1. Run scope

**Primary golden runs (full waveform chain):**

| key | run / subrun | det | FEU X,Y | drift |
|---|---|---|---|---|
| `o22_long_det2` | det2_det3 6-22 / longer_run | det2 | 6,8 | 1000 V |
| `sat_det3` | det3 saturday 6-27 / long_run_490/1000 | det3 (ref) | 7,8 | 1000 V |
| `g_det3_wknd` | det3 p2 6-27 / sanity_check | det3 | 7,8 | 1000 V |
| `g_det4` | det4_day 6-24 / long_run | det4 | 6,8 | 900 V |
| `g_det6_long` | det6_det7 6-26 / long_run | det6 | 3,4 | 700 V |
| `g_det7_long` | det6_det7 6-26 / long_run | det7 | 6,8 | 700 V |

**HV / drift scans (efficiency via 10; vdrift via 14/21/23):**
- det3 saturday 6-27 drift scan (100–1100 V) → 14/21/23 drift velocity + 10 eff
- det3 saturday 6-27 resist hv_scan/hv_scan2 → 10 (seed from sat_det3)
- det6/det7 6-26 hv scan (`g_det6_hv`,`g_det7_hv`) → 10 (seed from g_det{6,7}_long)
- det2/det3 6-22 resist scan → 10 (seed from o22_long)
- det3/det4 6-23 resist scan → 10 (seed from o23_long_det{3,4})

## 2. Per-detector chain (each primary key, in order)

Continue-on-error; per-step timeout; all output to one timestamped log.

```
wipe   cache/event_results*.pkl, microtpc_segments.csv, headon_features.csv
01     raw_detector_qa                      (quick sanity)
02     m3_reference_qa
04     detector_deep_qa
03     --refit --full          -> veto50 event cache + alignment + maps
03     --refit --no-veto       -> no-veto event cache (needed by 08/09)
08     efficiency_maps
09     efficiency_breakdown
12     efficiency_map_sliding --kernel=25 --grid=120
26     --refit                 -> MEASURE c1/c2, write cache/cshare.json
27     unsharing_refinement    (reads cache/cshare.json)
28     angle_calibration       (reads cache/cshare.json)
31     --rebuild               -> microtpc_segments.csv
33     --rebuild               -> headon_features.csv
34     --dump-events --save-model     (self-trained hybrid)
34     --model=<det3 frozen>          (transfer cross-check; det3 model first)
36     --rebuild               position estimators
42     time_resolution
38 ; 38b                       charge balance + figs
39 --rebuild-amp ; 40 --rebuild  sparks
```

**det3 ordering:** run `sat_det3` first so its saved hybrid model exists before
the other detectors' transfer cross-check (step 34-transfer).

## 3. Fleet / scan stages (once, after per-detector)

```
30     fleet_gas_survey                     (all detectors)
14 --refit ; 21 ; 23                        drift velocity (sat_det3 drift scan)
15 ; 17 ; 18 ; 19                           gas / attachment vs Magboltz
              (Magboltz tables already in garfield_sim/results/ — plotting only)
44 ; 45 ; 46/46b/46c                        vdrift summary + reference-scan
10  (per HV/drift scan run, with --seed)    HV-scan efficiency curves
43 ; 47 ; 47b                               window-truncation / Y-slow-rise (already fresh, re-confirm)
build_final_pdf.py <all keys>               per-detector overview PDF
engineer_package rebuild (optional)         report + slide deck
```

## 4. The cshare.json hook (only code change; reversible)

Add to the top of `27` and `28`, right after the hardcoded `CSHARE = {...}`:

```python
import json, os
_cj = os.path.join(CFG.out_dir('cache'), 'cshare.json')
if os.path.exists(_cj):
    CSHARE = {int(k): tuple(v) for k, v in json.load(open(_cj)).items()}
    print(f'CSHARE from {_cj}: {CSHARE}')
```

The driver writes `cache/cshare.json` from 26's measured medians. Absent the
file (e.g. re-running 27 by hand), behaviour is unchanged. **This is the only
edit to analysis scripts; it is backward-compatible and easy to revert.**

## 5. Known failure modes (expected; continue-on-error handles them)

- **det4 gain-limited:** 26 may not find enough near-vertical leads (clusters
  rarely ≥3 strips at the June operating point). If 26 can't measure, it writes
  no cshare.json and 27/28 fall back / skip — recorded as "det4 hybrid not
  measurable", NOT silently filled with det3 constants. The a1cce79 matched
  filter recovered ~5× more det4 hits, so this may now succeed — a key thing to
  check in the morning.
- **Magboltz scripts (15/17/18):** use existing tables in
  `garfield_sim/results/`; if a script tries to *regenerate* a table it may need
  the Garfield env and will time out — acceptable, the plots use cached tables.
- **HV-scan seeds:** 10 needs `--seed=<key>` for scan runs lacking a long_run
  subrun; seeds encoded in the driver per the runbook.
- **det3 gas eras:** never pool sat_det3 (post-dry) with earlier det3; the
  6-23 bottom-slot det3 point stays excluded (v_geom≈1). Driver does not pool.
- **Runtime:** waveform scripts 5–15 min each × 6 detectors × ~8 scripts, plus
  scans → estimate **6–10 h**. That is why it runs overnight.

## 6. Outputs & how we review

- Everything under `~/x17/cosmic_bench/Analysis/<run>/<sub>/<det>/...` (maps,
  ladders, CSVs, PNGs) — same tree as before, now from a1cce79 hits.
- Master log: `Analysis/_grand_logs/rerun_<stamp>.log`; per-step OK/WARN lines.
- A `RERUN_RESULTS_<stamp>.md` digest the driver appends: per-detector
  efficiency (08/09), σ_res (03), c1/c2 (26), hybrid σ68 + coverage (34),
  time-res (42) — the headline numbers, old-vs-this-run side by side where the
  previous value is known, so regressions are obvious at a glance.
- Nothing is committed or pushed; nothing goes to lxplus. Pure local regen.

## 7. What is deliberately NOT done (blind run)

- No per-detector re-tuning of fit windows, veto, MIN_STRIPS, λ-fit ranges.
- No doc/paper/quote propagation (JUNE_RESULTS_SUMMARY etc. stay as-is until we
  review the numbers).
- No archive/purge of superseded outputs (that is REPROCESSING_PLAN §5, later).
- CSHARE is measured & applied per-detector, but not hand-vetted per FEU.
