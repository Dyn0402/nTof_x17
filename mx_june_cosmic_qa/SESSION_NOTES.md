# MX17 June cosmic-bench QA — session handoff

Saclay cosmic-bench QA for the MX17 micro-TPC Micromegas detectors. Pull a run from
the DAQ PC, QA the detector + M3 reference tracker, align to the reference, build
efficiency maps. Code here in `mx_june_cosmic_qa/` **reuses** (does not fork)
`../cosmic_bench_analysis/` (`cosmic_micro_tpc_analysis.py`, `detector_qa.py`,
`M3RefTracking.py`) and `../common/Mx17StripMap.py`.

**Python env:** `../venv/bin/python` (has uproot/awkward). e.g.
`../venv/bin/python 03_alignment_and_tpc.py short_det1 --full`.

---

## ▶ Analysis flow for a NEW detector/run (quickstart)

1. **Pull the data** (combined_hits + m3_tracking + run_config only; not the raw .fdf):
   ```
   ./pull_run.sh <run_name> <sub_run> [area=det1_det2] [host=rays_daplxa]
   # e.g. ./pull_run.sh mx17_det1_det2_short_6-18-26 short_run det1_det2
   ```
   Lands data in `~/x17/cosmic_bench/<area>/<run>/<sub_run>/`. `~/x17` →
   `/media/dylan/data` (477G).

   **Data source(s).** The June runs live in TWO places, so try whichever is up:
   - **`rays_daplxa`** (the DAQ PC, sedipcaa28 via the `daplxa` jump) under
     `/mnt/cosmic_data/MX17/Run/<run>/<sub_run>/` — `pull_run.sh`'s default. May be
     wiped/repurposed between beam periods (e.g. the overnight 6-17-26 run is no longer
     there).
   - **`lxplus`** (Kerberos; `kinit dneff@CERN.CH` first if `ssh lxplus` prompts) under
     `/afs/cern.ch/user/d/dneff/x17/cosmic_bench/june_tests/<run>/<sub_run>/`. The user
     copied the June runs here for safekeeping; as of 6-19-26 it has all of them
     (`mx17_det1_det2_overnight_6-17-26`, `..._short_6-18-26`, `mx17_det3_ArIso_Test_6-16-26`,
     `zs_compression_scan_4_6-6-26`, etc.). Same `<sub_run>/combined_hits_root` +
     `m3_tracking_root` layout. Pull with a direct `rsync -a --update -e "ssh -o BatchMode=yes"
     lxplus:<remote> <local>` loop (pull_run.sh's remote path is the rays layout, not AFS). Check the new run_config: `zero_suppress` must be
   **false** (ZS kills the M3 reference, §1), and `det_orientation.z` must be **90**
   for the mx17 detectors (§2). If the field is missing/0, set it in run_config.json.

2. **Register in `qa_config.py`** — add a `_Config` per detector to the `RUNS` dict:
   ```python
   'short_det1': _Config('short_det1', '<run>', '<sub_run>',
                         feus=[X_feu, Y_feu], det_z=<det_center_z>, det_name='mx17_1',
                         zero_suppressed=False, base_path='.../cosmic_bench/det1_det2/'),
   ```
   `feus` = the X then Y FEU number (first index of `dream_feus` per axis in
   run_config). `det_z` = `det_center_coords.z` from run_config. One key PER DETECTOR
   (a 2-det run needs det1 + det2 keys, different FEUs/z/det_name). Always include the
   right `sub_run` — output/cache is keyed on it (subruns of one run otherwise collide).

3. **Run the QA pipeline** (every script takes the key as first arg):
   ```
   01_raw_detector_qa.py <key>            # occupancy, rates, amplitude maps
   02_m3_reference_qa.py <key>            # M3 tracker health (want ~54-57% clean tracks)
   04_detector_deep_qa.py <key>          # pathology: per-strip firing, spark/multiplicity tail
   03_alignment_and_tpc.py <key> --full  # alignment + micro-TPC + efficiency/resolution maps
   03_alignment_and_tpc.py <key> --no-veto   # builds the no-veto cache 08/09 need
   08_efficiency_maps.py <key>           # ray-based efficiency (primary deliverable)
   09_efficiency_breakdown.py <key>      # breakdown table + spatial plots
   plot_amplitude_vs_strip.py <key>      # pulse-height (local_max) vs strip position, per plane
   ```
   Big runs (≳300 MB combined_hits) are slow on `03`/`08` — run in background. Run
   detectors SEQUENTIALLY (two concurrent 300 MB loads can OOM).

4. **Read the result** (see §4/§5 for what "good" looks like):
   - `03`: alignment should converge to **sub-mm residual** (σ≈0.8 mm) at θ≈90°, z near
     `det_z`. If σ≈15 mm and the z-scan **rails to its window edge**, the z-range is
     wrong — widen it (`Z_LO=.. Z_HI=.. ../venv/bin/python 03_...`), don't blame the
     detector. The Y-residual Gaussian often diverges on an outlier shoulder; the robust
     core (rotation-scan σ) is the real number.
   - `09`: efficiency breakdown (reco_near = the efficiency; hit_no_reco / no_hit = the
     loss channels). `04` + `plot_amplitude_vs_strip` diagnose *why* (gain, dead strips,
     sparking).

**Output** goes to `~/x17/cosmic_bench/Analysis/<run>/<sub_run>/<det_name>/` (a single
top-level `Analysis/` tree, kept separate from the data). NOT in the git repo. Plots +
caches regenerate by rerunning; only the inputs under `cosmic_bench/<area>/` are precious.

---

## Run registry (`qa_config.py`)
`RUNS` dict keyed by short name. Current entries (all non-ZS, det_orientation.z=90,
Ar/Iso 95/5 unless noted):

| key | run / subrun | det | FEUs (X,Y) | z [mm] |
|---|---|---|---|---|
| `day_det1` | mx17_det1_daytime_run_1-28-26 / overnight_run | mx17_1 | 4,6 | 251* |
| `ovn_det1` | mx17_det1_det2_overnight_6-17-26 / longer_run | mx17_1 | 3,4 | 232 |
| `ovn_det2` | mx17_det1_det2_overnight_6-17-26 / longer_run | mx17_2 | 7,8 | 702 |
| `long_det1` | mx17_det1_det2_overnight_6-17-26 / long_run | mx17_1 | 3,4 | 232 |
| `long_det2` | mx17_det1_det2_overnight_6-17-26 / long_run | mx17_2 | 7,8 | 702 |
| `short_det1` | mx17_det1_det2_short_6-18-26 / short_run | mx17_1 | 3,4 | 232 |
| `short_det2` | mx17_det1_det2_short_6-18-26 / short_run | mx17_2 | 7,8 | 702 |
| `wknd_det2` | mx17_det2_det3_weekend_6-19-26 / short_run | mx17_2 | 3,4 | 232 |
| `wknd_det3` | mx17_det2_det3_weekend_6-19-26 / short_run | mx17_3 | 7,8 | 702 |

NB the weekend run renames the slots: **mx17_2** = the FEU 3/4 bottom (z=232) detector,
**mx17_3** = the FEU 7/8 top (z=702) detector (different physical dets than det1/det2).
Pulled from rays_daplxa. Also has a fine `resist_<NNN>V_drift_1000V` HV scan (450–525 V,
5 V steps) — feed `wknd_det2` to `10_hv_scan_efficiency.py` if a gain curve is wanted.

\* `day_det1` run_config nominal z is 411, but the M3-frame alignment optimum is ~251
(this bench's M3 stations sit at different z); det_z is set to 251 so the default ±60
z-scan brackets it. **Lesson: if the z-scan rails its edge, the nominal z is off — set
det_z to the optimum or widen with Z_LO/Z_HI.** (Older det_3 runs `det3_ariso`,
`zs_initial`, `long_run` were removed from the table; data not local.)

## Scripts
- `pull_run.sh <run> <sub_run> [area] [host]` — rsync the inputs from the DAQ PC.
- `01_raw_detector_qa.py` — occupancy/channel, hits-vs-position, rate-vs-time,
  amplitude-vs-time, hit scatter, amplitude maps. (Own loader: `detector_qa.load_hits`
  hardcodes `mx17_1`, wrong for det2.)
- `02_m3_reference_qa.py` — M3 tracker standalone: chi2, multiplicity, angles, beam
  profile, station hit maps.
- `03_alignment_and_tpc.py <key> [--full] [--no-veto] [--refit] [--rot0=D] [--veto=N]` —
  per-event micro-TPC analysis (cached), iterative z/θ/translation alignment, residual/
  correlation/angle plots; `--full` adds efficiency/resolution maps. **Default pipeline.**
- `04_detector_deep_qa.py` — noise/pathology: surface hitmap, per-strip firing fraction
  (always-on strips), event multiplicity (spark tail), multiplicity-vs-time.
- `05/06/07` — specialised one-offs (correlation-outlier translation, cluster-quality
  scan, clean z-refit). NOT part of the standard set; their old `--flipy` flag is retired.
- `08_efficiency_maps.py <key> [--r=5]` — **ray-based efficiency** (primary deliverable).
- `09_efficiency_breakdown.py <key> [--r=5]` — breakdown table + spatial plots (incl.
  `nonreco_ray_positions_scatter.png`).
- `plot_amplitude_vs_strip.py <key> [--field=F] [--channel] [--ymax=N]` — 2D hist of
  pulse height vs strip. Default field `local_max` (baseline-subtracted peak ADC,
  0–4095). NB `max_sample` is a near-constant small quantity (not pulse height);
  `amplitude` is the integral.

---

## Key findings

### 1. Zero suppression destroys M3 reference tracking
ZS (`zs_type: tpc`) suppresses the M3 reference Micromegas (read out via DREAM FEU 1) →
~4% clean tracks, broad chi2. Non-ZS gives **~54–57%** clean single tracks, chi2 peaked
at 0. **Always run non-ZS (or `zs_type: tracker`) for cosmic alignment.**

### 2. Detector↔M3 frame = a pure 90° rotation (standardised convention)
The mx17 detector frame is a proper **+90° rotation** of the M3 frame (det(R)=+1, no
flip; raw correlations detX↔refY ≈ +0.95, detY↔refX ≈ −0.93, diagonals ~0). This is now
captured cleanly:
- **`run_config.json` `det_orientation.z` = 90** for every mx17 detector. `03` reads it
  as the base rotation; the fine ±2° scan absorbs the small real misalignment (lands at
  θ≈88.75–90.5°). Set this field on any new run.
- **`AlignmentParams.ref_x_sign`** carries the M3-X handedness: `+1` = clean raw-M3
  convention (the June pipeline); default `-1` reproduces the legacy det_4/n_TOF
  `x_ref = -x_ref`, so those runs are byte-for-byte unchanged.
- The old per-run `--flipy` + `rot0=90` + hardcoded `x_ref=-x_ref` recipe (two
  reflections cancelling into a rotation) is **retired**. Result: sub-mm residual with
  **no flags**. (Fixed a bug where the z/rot/translation scan helpers dropped
  `ref_x_sign` from returned params → reverted mid-iteration and railed.)
- **Angle correlation is rotation-aware** (`plot_angle_correlation` takes `params`):
  it rotates the detector (tanθ_x,tanθ_y) vector into the M3 frame before pairing and the
  v_drift scan. Without it, det-X-angle was paired with ref-X-angle (uncorrelated for a
  90° detector) → washed-out plot and a v_drift that railed to the edge. With it: clean
  y=x, sensible v_drift (~35 µm/ns for day_det1; run-dependent).

### 3. Default alignment recipe (baked into `03`)
1. **Spark veto** `--veto 50` (drop events > N raw hits) — biggest cleaner; also lets the
   z-scan find the right z (sparks distort its variance landscape).
2. **Cluster-quality cut** `--maxdrop 2` for alignment determination only (competing-cluster
   strips, `n_dropped`, bias the z-scan).
3. Base rotation from `det_orientation.z`, raw M3 ref (`ref_x_sign=+1`).
4. **Iterative z/θ/translation** on the clean subset (3 iters, z ±60 mm @1 mm, θ ±2° @0.05°).
Good detector → z near det_z, θ≈90°, **σ ≈ 0.8 mm both axes**. Cache keyed by veto tag
only (no y-flip applied). 08/09 use the **no-veto** cache (`event_results.pkl`) so sparks
stay in the efficiency denominator; build it with `03 <key> --no-veto`.

### 4. Efficiency — ray-based method (`08`, `09`)
DAQ saves only events with a valid hit, so **an M3 track with no DREAM event is a genuine
MISS** (keep it in the denominator — do NOT match-filter). Per clean M3 single track:
project to the aligned plane → (x,y); record `within_range` (det reco within R mm) and
`has_any` (det fired any strip). Breakdown categories: reco_near (the efficiency),
reco_far (reco but >R), hit_no_reco (fired, no valid X+Y), no_hit (silent). Active-area
box is taken empirically from reco positions (`get_active_det_bounds` is degenerate in
the aligned frame).

### 5. What we've seen on the detectors
- **day_det1** (good detector, FEU 4/6): 97.5% both-XY, alignment σ≈0.82 mm, efficiency
  **65.6%**, reco-at-all 76.5%. This is what a healthy mx17 looks like.
- **det1 (ovn/long/short, FEU 3/4): rough — global low gain / low multiplicity.** Tracks
  sub-mm WHEN it reconstructs, but only ~9–16% of muons form a valid X+Y point. Cause:
  each plane silent ~40% per muon (no_hit ~31–56%); when it fires, only **~1.5–2
  strips/muon** vs the ≥3 needed to cluster → can't reconstruct (hit_no_reco dominates).
  Spatially uniform (no dead regions in the efficiency scatter) → a gain/HV/threshold
  problem, plus ~4–12% spark events. STABLE across subruns. `plot_amplitude_vs_strip`
  shows a **persistent dead/low-response gap on the Y plane (FEU 4) at ~175–275 mm** —
  source of the Y-residual outlier shoulder.
- **det2 (FEU 7/8): half-dead + state-dependent.** FEU 7 (X) connectors 5–8 dead → X only
  ~0–200 mm. Behaviour varies wildly by subrun/HV: longer_run ~dead (8% occ, 63 tracks),
  long_run half-functional (98% occ, FEU 8/Y tracks at corr 0.92 but FEU 7/X weak → ~0%
  clean 2D points), short_run pure massive sparking (median 503 strips/event). Records
  one coordinate at best; not a working 2D tracker.

### Implementation gotchas
- **sub_run isolation is mandatory** in OUT_BASE. A run with multiple subruns (longer_run
  vs long_run) will share caches if the path omits sub_run → the 2nd run silently reuses
  the 1st's per-event fits matched against the wrong M3 rays → fake decorrelation
  (σ~160 mm). Tell-tale: byte-identical numbers across two runs.
- z-scan **railing** its window edge ⇒ wrong z-range, not a bad detector (widen Z_LO/Z_HI
  or fix det_z). σ≈15 mm with a railed z is the signature.
- `detector_qa.load_hits` hardcodes `mx17_1` → wrong for det2 (`01` works around it).
- Big NON-ZS runs are memory-heavy (single uproot.concatenate of the combined_hits). Run
  in background, one detector at a time.

---

## Status / where to pick up
**Done:** standardised rotation convention (det_orientation.z) + rotation-aware angle
correlation; `pull_run.sh`; full QA on day_det1, ovn det1/det2, long det1/det2, short
det1/det2; det1 root-cause dissection; det2 characterisation; output consolidated to
`Analysis/`.

**Open / next:**
1. **det1 is gain-limited** — the physics ask is HV/threshold/gas to raise gain so
   clusters reach ≥3 strips. QA side: a per-axis (|Δx|,|Δy|) efficiency variant of `08`
   (currently radial only) would help.
2. **det2** needs the FEU 7 dead connectors (5–8) fixed before it can be a 2D tracker.
3. HV-scan subruns (`resist_<NNN>V_drift_1000V`) — det1 mesh-HV scan inside the overnight
   run (FEU 3/4 stepped 450→520 V in 10 V steps, drift fixed 1000 V, 30 min each; det2 HV
   unchanged). Pulled from lxplus on 6-19-26. Analysed by `10_hv_scan_efficiency.py <key>`
   (reuses the cosmic_micro_tpc building blocks with the **June convention** `ref_x_sign=+1`
   and θ from the established det1 alignment — NOT the legacy `cosmic_bench_analysis/
   hv_scan_efficiency.py`, which hardcodes the old det_4 θ≈0/`ref_x_sign=-1` recipe and
   would mirror the per-event match → bogus-low efficiencies on the 90°-rotated det1).
4. Upstream `get_active_det_bounds` still degenerate in the aligned frame (worked around).

Auto-memory (`~/.claude/.../memory/`): `june-cosmic-qa-day-det1`,
`june-cosmic-qa-overnight-det1-det2`, `june-cosmic-qa-output-location` (+ older
`june-cosmic-qa-procedure`, `-alignment-axis-fix`, `-det3-alignment-fail`).
