# HANDOFF: reco-far events (det3/det2) & M3 reference-track audit

*Written 7-04 after the micro-TPC/gas/unsharing campaign (see
`DET3_WEEKEND_ANALYSIS.md` and `report_det3_weekend/main.pdf` rev 4 for
the full state of knowledge). Data is FROZEN — no re-runs possible; all
work is on existing files.*

## Thread A — the reco-far tail and spark accounting

### The question
Saturday det3 long run: has_any 97.6 %, reco 80.2 %, within-5 mm 65.8 %
→ a **14.4 % reco_far tail** (reconstructed X+Y but >5 mm from the M3
ray). The p2 overnight run (same detector, same day, 45k rays) shows a
much smaller tail (within-5mm 79.7 % ≈ reco). Suspicion: sparks. The user
assumed spark events are "set aside" by the efficiency counting.

### KEY CODE FINDING (verified 7-04, start here)
**They are not.** `09_efficiency_breakdown.py`:
- loads `cache/event_results.pkl` — the **UN-vetoed** cache (not
  `event_results_veto50.pkl`!). A spark (>50-strip discharge) still yields
  cluster centroids → `has_both` → enters `reco{}` → is categorised
  **reco_near or reco_far by wherever its centroid lands**.
- `spark_frac` (multiplicity > `--spark=50` on raw hits) is computed and
  *printed alongside* but **never excludes events from the categories**.
- So the working hypothesis is: reco_far ≈ (sparks whose centroid is far)
  + (genuine mis-reco) + (M3 ray mismatch). At resist 490 V the Saturday
  spark fraction is ~9 % — same order as the 14.4 % tail.

### Concrete audit checklist
1. **Tag reco_far events with their raw multiplicity** (join the category
   loop of `09_*.py` with `_mult = det_raw.groupby('eventId').size()`).
   Distribution of multiplicity for reco_near vs reco_far vs hit_no_reco:
   if reco_far is dominated by >50 (or 30–50 "sub-veto sparks"), the tail
   is a counting artifact, not detector physics.
2. **Recompute the breakdown with sparks as their own category**
   (no_hit / hit_no_reco / spark / reco_far / reco_near). Then compare
   Saturday vs p2 again — does the Saturday/p2 within-5mm gap close?
3. **Check the veto threshold physics**: veto is total rows/event across
   BOTH FEUs (X+Y). A "half spark" (one plane discharging, ~30-50 strips)
   passes. Look at the multiplicity spectrum (it is bimodal? where is the
   valley?) — the 50 cut was ad hoc.
4. **Time correlation**: sparks cluster in time (HV recovering). Check
   reco_far event timestamps (`trigger_timestamp_ns`) vs spark event
   timestamps — are reco_far events within ~seconds after a spark
   (sagging field / baseline recovery)?
5. **Look at actual reco_far events** at waveform level (decoded_root is
   local for the Saturday run, det3, and for det2 o22): the event-display
   machinery from `22_strip_timing_and_estimators.py` /
   `24_waveform_investigation.py` plots any eventId. Classify by eye:
   discharge / two clusters (delta ray, second track) / noise cluster
   (FEU 6/8 common-mode — CNS is OFF in production hits!) / plausible
   cluster at genuinely wrong position.
6. **Ray-side mismatch**: the category loop iterates ALL clean rays in the
   active box (`for e, x, y in zip(evn, px, py)`) — if an event has 2 rays
   (0.2 % have >1 good track; more below chi2 ambiguity), the same det
   reco point is compared to each ray; a muon pair where the detector saw
   the OTHER ray → automatic reco_far. Quantify: reco_far rate for
   single-ray vs multi-ray events.
7. **Amplitude/quality of reco_far clusters**: from the veto50 results
   objects (n_strips, amplitudes, chi2 of strip fits) — genuine muon
   clusters look different from noise/spark residue.
8. Cross-check on **det2 o22 run** (decoded data already local,
   `29_det2_validation.py` has the loaders): det2 ran at resist 525 V —
   spark-rich. If the spark-accounting hypothesis is right, det2's
   reco_far should be even larger and even more multiplicity-correlated.

### Data / tools inventory for thread A
- Efficiency code: `09_efficiency_breakdown.py` (categories + spark_frac),
  `08_efficiency_maps.py`, `10_hv_scan_efficiency.py`,
  `12_efficiency_map_sliding.py`.
- Caches: `Analysis/<run>/<subrun>/<det>/cache/event_results.pkl`
  (UN-vetoed) and `event_results_veto50.pkl` (≤50 rows/event).
- Waveforms local: det3 Saturday long run + det2 o22 longer_run
  (`<base>/<run>/<subrun>/decoded_root/`, 512 ch × 32 samples,
  sample-major; eventIds continuous across files). Other runs: lxplus
  `~dneff/x17/cosmic_bench/june_tests/<run>/<subrun>/decoded_root/`.
- Sharing/unsharing chain: `26/27/28_*.py` (c1≈0.45-0.52, c2≈0.05-0.15
  measured per plane; min_strips 4→3 after unsharing).

## Thread B — audit of the inherited M3 reference tracking

### Why
Every efficiency number, alignment, and the micro-TPC angle reference
lean on `M3RefTracking` (inherited, never audited by us). Known oddities
collected during the campaign, each worth an explanation:

1. **z discrepancy**: alignment converges to z = 714 mm (det3 sat run,
   both weekend runs, ±10 µm repeatability) vs the nominal detector
   z = 702 mm. 12 mm is huge. Either the bench survey, the M3 internal
   z-scale, or a systematic in `get_xy_positions(ray_data, z)`
   propagation. An M3 z-scale error rescales REFERENCE ANGLES
   (tanθ_ref ∝ 1/z-lever) → would bias the geometry velocity estimator
   scale! (A +1.7 % z error → −1.7 % on v_geom. Our v systematic quotes
   ±5 %; check.)
2. **"Cutting on detector size: ~65 % of tracks remain"** — printed by
   M3RefTracking on load. Which cut is this, is it per-track or per-event,
   and does it bias the angular distribution (edge tracks removed →
   fine for ridge fits, but check).
3. **chi2_cut = 20** used everywhere — chi2 of what, how many dof, and
   what fraction of kept tracks are fake? ~35 % of events have 0 good
   tracks pre-chi2; 0.02–0.2 % have >1. The multi-track and near-cut
   populations feed thread A item 6.
4. **Angle conventions**: `ref_x_sign` flip and the θ_align ≈ 89.45°
   (not 90.00°) rotation. Is 0.55° a real detector rotation or partly an
   M3 axis convention? (The off-diagonal saga made us sensitive: check
   whether M3's own x/y axes are exactly orthogonal in the code.)
5. **eventId gaps**: hits eventIds skip (e.g. 25954 → 25957 across file
   boundaries in the 6-22 run). Confirm the M3 rays use the same eventId
   stream (they are built from the same DAQ files on the DAQ PC) — an
   off-by-N ray↔det mismatch would produce... a reco_far tail. A strong
   test: shift eventId by ±1 and watch within-5mm efficiency collapse
   (if it does NOT collapse, matching is loose somewhere).
6. **Ray angular resolution**: we assumed σ(θ_ref) ≈ 0.05° (negligible
   regression dilution in ridge fits). Verify from M3 geometry (lever
   arm, cluster resolution) or from track-fit residuals in the code.

### Where the code is
- Local analysis side: `cosmic_bench_analysis/M3RefTracking.py`
  (`M3RefTracking`, `get_xy_angles`, `get_xy_positions`) — the consumer.
- Ray PRODUCTION is remote/inherited: runs on the DAQ PC (rays_daplxa,
  frequently unreachable; needs `root_6_30_02` + `devtoolset-9` env —
  see memory note "Local reco from raw fdf"). Producer outputs
  `m3_tracking_root/*_rays.root` per datrun file. The producer source
  lives on the DAQ machine — pull a copy into the repo when the DAQ is
  reachable, so it can be audited offline (it currently is not in git!).
- Ray files: `<base>/<run>/<subrun>/m3_tracking_root/` (local for all
  analyzed runs; also on lxplus archive).

### Suggested order
1. Thread A items 1–2 (one afternoon: re-categorise with multiplicity;
   likely resolves the Saturday-vs-p2 puzzle and fixes the June PDF
   efficiency numbers fleet-wide).
2. Thread B item 5 (eventId shift test — cheap, high value) and item 1
   (z-scale — feeds the v_geom systematic).
3. Deeper M3 code audit once the producer source is retrievable.

## Cross-cutting gotchas (hard-won, do not rediscover)
- Repo venv is `.venv` (NOT `../venv`). Garfield runs on system python3
  (env global) or lxplus (LCG_107 + `ROOT.gSystem.Load('libGarfield')`).
- Production hits have NO common-noise subtraction; FEU 6/8 have huge
  common mode (raw σ~115). Waveform-level work must do pedestal + per-chip
  CNS (see `24_*.py`).
- Charge sharing is ~50 % to first neighbours: hits-level times are
  biased (see report Sec. "timing"); never fit velocities from raw strip
  times without unsharing.
- Drift HV varies per run AND per slot: decode from `hv_monitor.csv`
  (channel map calibrated on the sat run: `0:7` = top drift, `0:6` =
  bottom drift, `3:4`/`3:3` = top/bottom resist, `0:8-0:11` = M3 @ 500 V).
  Several early runs were NOT at 1000 V (6-23: 600 V, 6-25: 500 V?,
  6-26: 700 V).
- det3 was water-saturated (>3 % H2O) until ~6-24 and at ~1 % from 6-25
  (`30_fleet_gas_survey.py`) — efficiency/gain numbers from early-week
  det3 runs reflect wet gas.
- The Saturday "7 h" long run is complete at 141 min (4 file pairs).
- mx17_det3_det4_overnight_6-23-26 det3 point is broken (v_geom ≈ 1);
  don't build on it without understanding why.
