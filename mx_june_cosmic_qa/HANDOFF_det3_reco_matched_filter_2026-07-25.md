# Handoff: recover det3 reconstruction quality on the matched-filter (a1cce79) hits

**Written 2026-07-25 after the full June re-analysis (`RERUN_RESULTS_20260725_011307.md`).**
**Audience: someone picking this up cold. You do not need to have run anything before.**

---

## 1. The one-paragraph problem

The waveform analyzer was reworked on 7-24 to trigger on low-amplitude pulses
(matched-filter boxcar gate, commit `a1cce79`), and the whole local cosmic bench
was reprocessed onto it (+40 % hits). A full blind re-analysis then ran overnight
7-25 (6 h 41 m, 144 steps OK / 5 WARN). Raw yield went up as intended, and every
detector's **hybrid** angular resolution improved — but **position accuracy got
worse on all six detectors**, and det3's headline efficiency fell from
**93.4 % → 84.1 %** within 5 mm, with core residual σ **0.48 → 0.64 mm**.

Your job: get det3 back to ~93 % / ~0.48 mm on the *new* hits, without throwing
away the low-amplitude sensitivity that the rework bought (det4 gained 15 points
of efficiency from it).

---

## 2. What is already established — do not re-derive these

### 2.1 It is NOT the spark threshold
`09_efficiency_breakdown.py` classifies events with > 50 firing strips as
"spark" and excludes them. Spark fraction roughly doubled (det3 9.1 → 17.3 %),
which looks like an obvious culprit. It isn't. Sweeping the threshold
non-destructively:

| key | `--spark=50` | `100` | `200` | `400` |
|---|---|---|---|---|
| sat_det3 within5 | 84.1 | 84.6 | 84.8 | — |
| g_det6_long within5 | 42.8 | 44.0 | 44.3 | 44.5 |

Raising it just migrates events from `spark` into `reco_far`. They **do**
reconstruct — they land > 5 mm from the M3 reference ray. The loss is position
accuracy, not event classification.

### 2.2 It is NOT stale data anywhere
- caches were wiped and rebuilt (`cache/event_results*.pkl`, mtimes during the run)
- all `combined_hits_root` files carry the `significance` branch (a1cce79), mtime 7-24 18:0x
- the driver passed every rebuild flag (`03 --refit`, `26 --refit`, `31/33/36/40 --rebuild`, …)

### 2.3 The mechanism (this is the important part)

Reconstructed DUT position is **the strip position of the single earliest-time
hit in the largest spatial cluster** — not a centroid, not a fit.

```
cosmic_micro_tpc_analysis.py:615  _fit_single_axis()
  ...
  df_axis['_cluster'] = df_axis[pos_col].diff().gt(gap_threshold).cumsum()   # GAP_THRESHOLD_MM = 12.0
  largest_cluster_id = df_axis['_cluster'].value_counts().idxmax()           # biggest cluster wins
  earliest_idx = df_cluster['time'].idxmin()                                 # <-- THE ANCHOR
  pos_anchor  = df_cluster.loc[earliest_idx, pos_col]
  ...
  mesh_position_mm = float(pos_anchor)          # line 693

cosmic_micro_tpc_analysis.py:405/410
  det_x_mm -> x_fit.mesh_position_mm            # this is what 08/09 score against M3
```

`MIN_STRIPS = 3`, `GAP_THRESHOLD_MM = 12.0` (lines 74–75). There is **no quality
requirement on the anchor strip**. Any hit that is timed early and survives the
5σ gate can become the reported position, no matter how marginal.

Measured on `sat_det3`, one combined file, veto50 sample (163,965 hits /
7,451 events / 14,641 fitted planes):

| quantity | value |
|---|---|
| anchor significance, percentiles (10/25/50/75/90) | 7.6 / 19.6 / 97.7 / 277.5 / 631.9 |
| cluster **median** significance (10/25/50/75/90) | 53.2 / 76.8 / 112.0 / 159.0 / 217.1 |
| anchor is the **lowest-significance strip** in its own cluster | **14.5 % of planes** |
| anchor significance < 10 | 15.9 % of planes |
| anchor significance < 0.3 × cluster max | 50.9 % of planes |
| \|anchor − amplitude-centroid\| median / p90 / >5 mm | 2.22 mm / 4.58 mm / 7.1 % |
| …when anchor significance < 10 | **3.52 mm** / — / **18.8 %** |
| …when anchor significance ≥ 10 | 1.96 mm / — / 4.9 % |

So in ~16 % of fitted planes the reported position is defined by a marginal
strip, and those are displaced a median 3.5 mm — with 18.8 % beyond the 5 mm
efficiency cut. Two planes per event must both be good. That is the right order
of magnitude for a 9-point efficiency drop.

This also explains why **hybrid σ68 improved on 6/6 detectors** while classical
metrics degraded: the trained regression/segment estimators never use the naive
earliest-hit anchor.

The physics intent of the anchor is right — earliest arriving charge = the track
crossing at the mesh plane. The failure is that a noise hit anywhere inside a
12 mm-gap cluster can steal it.

---

## 3. Suggested line of attack (cheapest first)

Nothing here has been tried. Prototype **offline** first (see §4) — a full
`03 --refit` is ~10 min for `sat_det3`, but a standalone script over one file
runs in seconds, and the anchor is computed inside 03, so you cannot iterate
from the cache.

1. **Significance floor at load time.** `significance` is already in the hits
   tree and already loaded (`_load_hits` uses `uproot.concatenate(..., library='pd')`
   with no `expressions=`, so every branch is in the DataFrame). One line in
   `03_alignment_and_tpc.py:_load_hits()`. Scan S ∈ {5 (current), 8, 10, 15, 20}.
   Cheapest possible fix; test it first because if it works, everything
   downstream is unchanged.
2. **Robust anchor selection** in `_fit_single_axis`. Candidates:
   - require anchor significance ≥ max(10, 0.3 × cluster max)
   - earliest hit *among the top-N amplitude strips* in the cluster
   - evaluate the fitted time-vs-position line at the earliest robust time
     instead of taking a raw strip position
   This is more surgical than (1): it keeps marginal strips in the time fit
   (where `sigma = 1/sqrt(amp)` already down-weights them) but stops them from
   defining the position.
3. **Tighten `GAP_THRESHOLD_MM`** (12 mm is generous). With more hits, noise
   bridges what used to be separate clusters, so the "largest cluster" can now
   swallow junk far from the track. Test 4–8 mm.
4. **Cluster selection** — largest-by-count is fragile once noise is dense.
   Consider largest-by-charge.

Expect (1) and (2) to interact; scan them together rather than one after the other.

---

## 4. How to work efficiently

**Prototype offline.** This reproduces 03's exact hit preparation and takes
seconds — modify the anchor rule here before touching any pipeline script:

```python
import glob, numpy as np, uproot, qa_config
from qa_config import setup_paths; setup_paths()
import cosmic_micro_tpc_analysis as cm
from common.Mx17StripMap import RunConfig

c  = qa_config.get_config('sat_det3')
rc = RunConfig(c.run_config_path, c.MAP_CSV_PATH); det = rc.get_detector(c.DET_NAME)
f  = sorted(glob.glob(c.combined_hits_dir + '*.root'))[0]
df = uproot.open(f)['hits'].arrays(library='pd')
df = df[df['feu'].isin(c.MX17_FEUS)].copy()
df = df[df.groupby('eventId')['channel'].transform('size') <= 50].copy()   # veto50, as 03 does
df = cm._map_strip_positions(df, det)
# ... now cluster per plane (gap 12 mm, >=3 strips) and try anchor rules
```

The full script that produced the §2.3 table is worth rebuilding from that stub;
it took ~1 min to run.

**Then validate properly** on `sat_det3` only:

```bash
cd mx_june_cosmic_qa
rm -f <OUT_BASE>/cache/event_results*.pkl
../.venv/bin/python 03_alignment_and_tpc.py sat_det3 --refit --full
../.venv/bin/python 03_alignment_and_tpc.py sat_det3 --refit --no-veto
../.venv/bin/python 08_efficiency_maps.py sat_det3
../.venv/bin/python 09_efficiency_breakdown.py sat_det3
../.venv/bin/python 12_efficiency_map_sliding.py sat_det3 --kernel=25 --grid=120
```

`OUT_BASE` for `sat_det3` is
`~/x17/cosmic_bench/Analysis/mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3`
(get any key's path with
`python -c "import qa_config; print(qa_config.get_config('sat_det3').OUT_BASE)"`).

**Score yourself** against the stored pre-rework baseline:

```bash
../.venv/bin/python -c "
import json, rerun_digest as R
o=json.load(open('rerun_baseline.json'))['keys']['sat_det3']; n=R.harvest('sat_det3')
for f in ['within5','reco_at_all','core_sigma_mm','median_r_mm','sigma_theta_x_deg','sliding_within']:
    print(f, o.get(f), '->', n.get(f))"
```

---

## 5. Acceptance criteria

Recover det3 **without** giving back the low-amplitude gain.

| metric | pre-rework | now | target |
|---|---|---|---|
| sat_det3 within 5 mm | 93.4 % | 84.1 % | ≥ 93 % |
| sat_det3 core σ\|r\| | 0.48 mm | 0.64 mm | ≤ 0.50 mm |
| sat_det3 median \|r\| | 0.80 mm | 1.01 mm | ≤ 0.85 mm |
| sat_det3 σ_θ X / Y | 2.42 / 2.60° | 2.42 / 2.60° | no worse |
| sat_det3 sliding-map within | 94.4 % | 85.9 % | ≥ 93 % |

**Guard rails — a fix that breaks these is not a fix:**

- `g_det4` must keep its gain: `has_any` ≥ 95 % (was 69.6 % pre-rework, 95.6 % now)
  and within-5mm ≥ 35 % (was 20.7 %, now 35.3 %). If a significance floor claws
  det3 back by killing det4, it is the wrong knob.
- hybrid σ68 must not regress from the current run (sat_det3 lt5 1.63°, gt8 1.64°;
  det6 gt8 4.90°; det7 lt5 2.48°).
- `26`'s measured charge sharing must still be measurable on all six detectors
  (see §6.3).

Check `g_det4` and `g_det6_long` before declaring victory — det6 is the second
worst hit (57.8 → 42.8 %) and det7 the worst (43.1 → 16.7 %), so a real fix
should lift them too.

---

## 6. Things that will bite you

### 6.1 Baseline vintage is not uniform
`rerun_baseline.json` was snapshotted at 01:10 on 7-25 from whatever was on disk,
and those files are of mixed age (`rerun_baseline_vintage.json`):

| key | baseline files | comparable? |
|---|---|---|
| sat_det3 | 7-17 / 7-18 | yes |
| g_det3_wknd | 6-29 / 7-06 | **NO** — predates the 7-13 M3 recipe change (χ²<1 + NClus≥4) |
| o22_long_det2, g_det4, g_det6_long, g_det7_long | bulk 7-14 | yes |

`g_det3_wknd`'s σ_θ 2.12 → 2.65 is an artifact of that recipe change, not a
measured degradation. **Use `sat_det3` as your reference detector**, not the
weekend run.

### 6.2 Open anomaly: sat_det3 σ_θ is *bit-identical* across generations
2.417884172795585 with n_events 6306, both before and after — while efficiency
and time-resolution from the same cache clearly moved. Ruled out: stale file
(recomputing from the current cache in a scratch dir reproduces the JSON field
for field), stale cache (rebuilt 01:18; 42's dualplane count moved
28404 → 26230), old-generation hits (`significance` present). Every other
detector's σ_θ moved by +9 % to +30 %. **Unexplained.** If your work touches the
angle path, be aware this may be masking something.

### 6.3 Charge-sharing constants are per-detector and must stay that way
`27_unsharing_refinement.py` and `28_angle_calibration.py` historically hardcoded
`CSHARE = {6:(0.247,0.057), 8:(0.514,0.232)}` — det7's numbers, silently applied
to every detector. As of this run they read `cache/cshare.json` when present
(written from `26`'s measured medians), falling back to the hardcoded dict
otherwise. Measured values differ by 2× across the fleet:

| det | drift | X-plane c1 | Y-plane c1 |
|---|---|---|---|
| det6 | 700 V | 0.231 | 0.445 |
| det7 | 700 V | 0.231 | 0.545 |
| det4 | 900 V | 0.354 | 0.422 |
| det2 | 1000 V | 0.418 | 0.518 |
| det3 (both runs) | 1000 V | 0.450 / 0.451 | 0.522 / 0.518 |

det7's fresh measurement reproduces its own 7-14 hardcoded values to ~6 % on a
different analyzer generation, so `26` is reproducible and the spread is real
detector-to-detector variation. If you re-run `27`/`28`, make sure `26` ran
first for that detector.

### 6.4 Known-broken steps (pre-existing, not your problem)
- `39_spark_deadtime.py` fails for det2/det4/det6 — `SPARK_DIR` at line 83 only
  has `mx17_3` and `mx17_7`, and only `det3_spark_analysis/` and
  `det7_spark_analysis/` exist on disk.
- `40_spark_waveforms.py` needs re-running for `g_det3_wknd`; it crashed on a
  malformed event before the guard was added (see §7).
- `44_final_vdrift_plot.py` **must** be given a key. With no argument
  `config_from_argv()` falls back to `DEFAULT_RUN` (the 6-16 ArIso det1 run) and
  dies on a missing CSV. Use `44_final_vdrift_plot.py sat_det3`.
- `10_hv_scan_efficiency.py --seed=` takes a **path to an alignment.json**, not
  a config key. Passing a key silently falls back to a hardcoded `SEED_DEFAULT`
  (z=243, offsets "from memory"). The 6-23 run's alignment lives under its
  `long_run` subrun, which is already 10's default — do not override it there.

### 6.5 Physics results that did NOT move (leave them alone)
- v_unshared(1000 V) = **33.70 ± 0.82 µm/ns** vs the established 34 ± 1.5
- gas ranking unchanged: `Ar94_iso5_H2O1` best (RMS 0.59 vs 1.81 for next)
- single-strip σ_t stable within a few percent on every detector

Watch: `v_geom_y` is NaN at 500/700/1100 V and low (14–18 vs ~29 for X) where it
resolves. Possibly the PLAN_47 Y slow-rise surfacing in the geometry estimator;
there is no pre-rerun value to compare against.

---

## 7. Code changed on 7-25 (all reversible, none of it tunes a result)

| file | change |
|---|---|
| `27_unsharing_refinement.py`, `28_angle_calibration.py` | read `cache/cshare.json` when present (plan §4 hook); fall back to the hardcoded dict |
| `40_spark_waveforms.py` | skip events whose amplitude array is not 32×512 and report the count. A malformed entry (size 31744 = two merged events minus a row) killed the step on `g_det3_wknd`; the guard then caught 5 such events on `g_det4` |
| `rerun_june_analysis.sh` | `10 --seed` gets a real alignment.json path; `44` gets `sat_det3`; added det3 6-27 scan to `10`, plus 46/46b/46c and 43/47/47b per plan §3 |
| `rerun_digest.py` (new) | harvests headline numbers from the scripts' own outputs; `--snapshot` mode freezes a baseline |
| `rerun_baseline.json`, `rerun_baseline_vintage.json` (new) | pre-rerun values and their file vintages |

Nothing was committed, nothing was pushed, nothing went to lxplus/EOS.

---

## 8. Reading order for context

1. `RERUN_RESULTS_20260725_011307.md` — the full results digest and session notes
2. `REPROCESSING_2026-07-24.md` — what the analyzer rework changed and why
3. `RERUN_PLAN_2026-07-24.md` — the plan this run executed
4. `MICROTPC_RUNBOOK.md` — how the micro-TPC chain fits together
5. Log: `~/x17/cosmic_bench/Analysis/_grand_logs/rerun_20260725_011307.log`
