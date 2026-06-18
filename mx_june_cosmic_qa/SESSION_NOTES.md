# MX17 June cosmic-bench QA — session handoff

Saclay cosmic-bench QA for the MX17 micro-TPC Micromegas detectors. Pull a run from
the DAQ PC, QA the detector + M3 reference tracker, align to the reference, and build
efficiency maps. Code lives here in `mx_june_cosmic_qa/` and **reuses** (does not fork)
`../cosmic_bench_analysis/` (`cosmic_micro_tpc_analysis.py`, `detector_qa.py`,
`M3RefTracking.py`) and `../common/Mx17StripMap.py`.

Python env: `../.venv` (has uproot/awkward). Run e.g. `../.venv/bin/python 03_alignment_and_tpc.py ovn_det1 --full`.

---

## Data pull
- Runs live on `ssh rays_daplxa` under `/mnt/cosmic_data/MX17/Run/<run>/`.
- Copy to `/home/dylan/x17/cosmic_bench/det_<N>/` (or `det1_det2/` for the 2-det run).
- Check remote `du -sh` and local `df -h /home/dylan` first. Remote rsync is old (3.0.9):
  no `--info`, use `--stats`.
- Only `combined_hits_root/` and `m3_tracking_root/` are needed for analysis; the raw
  `.fdf` (bulk of a run) are not. Default `rsync -a --exclude='*.fdf'`, or filter to just
  those two subdirs for a minimal pull.

## Run registry (`qa_config.py`)
`RUNS` dict keyed by short name; every script takes the key as first CLI arg. Output is
keyed `output/<run>/<det_name>/` so multi-detector runs don't collide.

| key | run / subrun | det | FEUs (X,Y) | z [mm] | ZS? |
|---|---|---|---|---|---|
| `det3_ariso` | mx17_det3_ArIso_Test_6-16-26 / run | mx17_1 | 7,8 | 702 | **YES (tpc)** |
| `zs_initial` | zs_compression_scan_4_6-6-26 / initial_run | mx17_1 | 3,4 | 232 | no |
| `long_run` | mx17_det3_long_run_5-6-26 / long_run | mx17_1 | 3,4 | 232 | no |
| `ovn_det1` | mx17_det1_det2_overnight_6-17-26 / longer_run | mx17_1 | 3,4 | 232 | no |
| `ovn_det2` | mx17_det1_det2_overnight_6-17-26 / longer_run | mx17_2 | 7,8 | 702 | no |

`feu` column = first index of `dream_feus` in run_config.json.

## Scripts
- `01_raw_detector_qa.py <key>` — occupancy/channel, hits-vs-position, rate-vs-time,
  amplitude-vs-time, hit scatter, amplitude maps. (Has its own loader using `CFG.DET_NAME`
  because `detector_qa.load_hits` hardcodes `mx17_1`.)
- `02_m3_reference_qa.py <key>` — M3 tracker standalone: chi2, multiplicity, angles,
  beam profile, station hit maps.
- `03_alignment_and_tpc.py <key> [--full]` — per-event micro-TPC analysis (cached),
  iterative z/θ/translation alignment, residual/correlation/angle plots; `--full` adds the
  upstream efficiency/resolution maps. **This is the default alignment pipeline** (see below).
- `04_detector_deep_qa.py <key>` — noise/pathology: surface hitmap, per-strip firing
  fraction (always-on strips), event multiplicity (spark tail), multiplicity-vs-time.
- `05_align_correlation_outliers.py <key> --flipy` — robust correlation-line translation +
  outlier characterisation.
- `06_cluster_quality_scan.py <key> --flipy` — trade-off scan of cluster-quality cuts.
- `07_refit_z_clean.py <key> --flipy` — clean z re-fit demonstration.
- `08_efficiency_maps.py <key> [--r=5]` — **ray-based efficiency** (primary deliverable).
- `09_efficiency_breakdown.py <key> [--r=5]` — the efficiency breakdown table + spatial plots.

---

## Key findings (chronological)

### 1. Zero suppression destroys the M3 reference tracking
The first run (`det3_ariso`) had `zero_suppress: true, zs_type: tpc`. The M3 reference
Micromegas are read out through DREAM FEU 1, so TPC-ZS suppresses their signal → terrible
tracking: only **4.3%** clean single tracks, broad (absolute, not reduced) chi2.
Non-ZS runs give **~54–57%** clean tracks with chi2 sharply peaked at 0. **Always run
non-zero-suppressed (or `zs_type: tracker`) for cosmic alignment.**

### 2. 90° axis swap + handedness — why alignment "totally failed"
The mx17 strip-map frame is rotated **~90°** vs the M3 frame: detector-X measures M3-**Y**
(corr 0.95), detector-Y measures M3-**X** (corr −0.93); the X↔X / Y↔Y combos are ~0. The
upstream alignment only scans ±2°, so it never absorbed the 90° → residual σ ≈ detector
width, looking like total failure. (Hidden at first because the ZS run's tracking was random.)
Second subtlety: the code hardcodes `x_ref = -x_ref` in 3 places, which turns the proper +90°
rotation into a rotation+reflection (improper) → one axis comes out mirrored (anti-diagonal).
**This is NOT a strip-map / connector-inversion bug** — `Mx17StripMap.map_hit` correctly
applies `dream_feu_orientation` ("inverted"); verified (forcing it off worsens the result).
It is a detector↔M3 *frame* convention (Saclay bench axes ~90° rotated), which is the
alignment's job, not the strip map's.

Fix (in `03`, default-on for this bench): `rot0=90` base rotation + `--flipy` (negate
`y_position_mm`) to make the rotation proper so BOTH axes align.

### 3. The two detectors (overnight run)
- **det1 (mx17_1, z=232, FEU 3/4): noisy.** Fires in 96% of events, ~46 hits/event, flat
  occupancy. The noise is **spark/discharge events**: bimodal multiplicity (median 4
  strips/event, tail up to all 1024 strips). Spark events (>50 raw hits) are ~14% of events
  but carry ~89% of all hits.
- **det2 (mx17_2, z=702, FEU 7/8): inefficient + half-dead.** Fires in only ~8% of events.
  FEU 7 (X) reads out only channels 0–255 → **connectors 5–8 dead**, X instrumented only
  over ~0–200 mm. (det2 correlation with reference is weaker, ~0.46.)

### 4. Default alignment recipe (now baked into `03`)
In impact order:
1. **Spark veto** `--veto 50` (drop events > N raw hits). Biggest single cleaner AND lets the
   z-scan find the right z (sparks distort its variance landscape).
2. **Cluster-quality cut** `--maxdrop 2` for *alignment determination only* (events with strips
   in competing clusters, `n_dropped`, bias the z-scan).
3. `rot0=90` + `--flipy` (frame fix, section 2).
4. **Iterative z/θ scan on the clean subset** (3 iterations, z ±60 mm @1 mm, θ ±2° @0.05°).

Result for det1: converges stably z_x=243, z_y=244, θ=90.5°, **residual σ ≈ 0.78 mm both
axes** (sub-mm), drift velocity X=48.0 / Y=48.5 µm/ns (consistent). NB: `residuals.png` Y
Gaussian auto-fit diverges on the outlier shoulder — the robust core σ_y is 0.78 mm.

Defaults: `VETO=50` (`--no-veto`/`--veto=N`), `MAXDROP=2` (`--no-clustercut`/`--maxdrop=N`),
`ROT0=90` (`--rot0=`), `FLIPY=on` (`--no-flipy`). Per-event cache keyed by veto+flipy tag.

### 5. Efficiency — ray-based method (`08`, `09`)
The micromegas DAQ saves only events with a valid hit, so **an M3 track with no DREAM event
is a genuine MISS** (keep it in the denominator — do NOT match-filter). Per clean M3 single
track: project to the aligned plane → (x,y), record `within_range` (det reco hit within R mm)
and `has_any` (det fired any FEU 3/4 strip). Build a ray hit/miss list → scatter (green/red),
binned map (self-normalises spatially → ~0 outside active area), integrate inside the active
area. Sparks need no special handling (garbage position fails `within_range`).

**det1, longer_run, R=5 mm, active area** (`09` breakdown table — enshrined):

| category | meaning | % crossings |
|---|---|---|
| reco_near | reconstructed, \|r\|≤5 mm (the efficiency) | **14.8%** |
| reco_far | reconstructed but \|r\|>5 mm (competing cluster) | 4.0% |
| hit_no_reco | fired strips, no valid X+Y reco | **50.2%** |
| no_hit | detector silent (true miss) | 31.0% |
| has_any | detector responded (any strip) | **69.0%** |
| reco-at-all | formed a valid X+Y point | 18.8% |

Of reconstructed hits: **78.9% within 5 mm, core σ(\|r\|)=0.63 mm, median 1.22 mm** — i.e.
when it reconstructs, the residual is tight (like the alignment residual). The low map
efficiency is a **reconstruction-efficiency** effect (~19% reco-at-all), NOT displaced hits:
`reco_far` is only 4%. The dominant loss is `hit_no_reco` (50%) — fired strips but never
formed a reconstructable ≥3-strip X+Y cluster.

Spatial: well-reconstructed detector positions concentrate on the **left half (x≲0)**;
non-reconstructed muons are spread fairly uniformly.

### Implementation gotchas (bench-specific assumptions in upstream code)
- `detector_qa.load_hits` hardcodes `get_detector('mx17_1')` → wrong mapping for det2.
  (`01` works around it.)
- `get_active_det_bounds` returns degenerate bounds in the flipped frame → define the
  active-area box empirically from reconstructed hit positions (`08`/`09` do this).
- The pipeline's efficiency `plot_efficiency_map` denominator uses ALL M3 tracks — fine here
  (DREAM⊂M3, missing = miss), but worth verifying for other setups.
- Hardcoded `x_ref = -x_ref` is the handedness culprit (section 2).
- Big NON-ZS runs OOM the per-event analysis (single uproot.concatenate of ~1 GB
  combined_hits). Run in background; chunk if needed. (`mx17_det3_long_run_5-6-26` died this
  way and still needs a chunked re-run.)

---

## Status / where to pick up
**Done:** data pulls (det3_ariso, zs_initial=initial_run, ovn det1/det2; long_run partial);
full QA + sub-mm alignment + efficiency maps for **det1** (ovn_det1). Default pipeline set.

**Open / next steps:**
1. **Dissect `hit_no_reco` (50%)** — the next obvious question. Split into: fired one plane
   only (X xor Y) / fired both but <3 strips (`MIN_STRIPS`) / enough strips but fit failed.
   Tells us if reconstruction is limited by a loosenable threshold, a dead plane, or signal
   quality. (Have per-event fits + raw hits already.)
2. **`long_run` subrun** of the overnight run: combined_hits present but `m3_tracking_root`
   was still being produced (empty as of 6-18 ~15:15). Re-pull when the rays files land, then
   rerun `03 ovn-longrun --full`, `08`, `09` for proper efficiency-map statistics
   (longer_run is only ~2 h → 973/1600 bins masked).
3. **Apply the efficiency chain to det2** (`ovn_det2`) — expect poor (half-dead X).
4. **Per-axis (|Δx|,|Δy|) efficiency** variant of `08` (currently radial only).
5. **Upstream cleanups** (after confirming with the old n_TOF/det4 runs they don't regress):
   wire `det_orientation.z` from run_config into the alignment as a base rotation; reconcile
   the hardcoded `x_ref` sign; fix `get_active_det_bounds` for arbitrary alignment.

Auto-memory (outside repo, `~/.claude/.../memory/`): `june-cosmic-qa-procedure`,
`june-cosmic-qa-alignment-axis-fix`, `june-cosmic-qa-overnight-det1-det2`,
`june-cosmic-qa-det3-alignment-fail`.

Note: `output/` (plots + caches, ~76 MB) is gitignored and local-only; regenerate by
rerunning the scripts (needs the data under `/home/dylan/x17/cosmic_bench/`).
