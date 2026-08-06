# PLAN 08 — waveform-first (`wft/`) reconstruction on run_79, and the merge with the n_TOF scintillators

**Written 2026-07-30. Planning only — nothing here has been executed.** Facts
marked **[verified today]** were read out of the data or the config in the
session that wrote this; **[from X]** cites an existing document; everything
else is **[inferred]** and is flagged where it matters.

**Mission.** Reconstruct MM tracks in DREAM `run_79` from the *waveforms*
(`wft/`, the basis mandated by `../RECONSTRUCTION_BASIS.md`) instead of the
hits-chain micro-TPC of PLAN_02/03, and join the resulting per-event track
table to our own reprocessed n_TOF scintillator stream (`v12_liqpileup`,
run 224572) through the machinery in `ntof_dream_merge/`.

This plan supersedes, for run_79, the angle/depth parts of PLAN_02/03 and the
frozen `models/mx17_*_hits6.json` regressions — see the 2026-07-28 correction
at the top of `README.md`.

Reading order before executing: `../RECONSTRUCTION_BASIS.md`,
`../mx_june_wft/HANDOFF_2026-07-30.md` (state + procedures of the reco),
`../mx_june_wft/RECO_BENCH_2026-07-29.md` (what was tried and what won),
`../ntof_dream_merge/DREAM_NTOF_CALIBRATION.md` (the merge, and the time
calibration it rests on).

---

## 1. What run_79 actually is  [verified today, from `run_config.json`]

| | |
|---|---|
| DREAM data | `/media/dylan/data/x17/beam_july/runs/run_79/` |
| sub-runs with data | `stat090_0000` (07-26 18:07) and `stat090_0001` (19:07) only; `_0002`…`_0015` are stubs |
| per sub-run | 13 file-tags × 8 FEUs, `decoded_root` **12 GB**, `combined_hits_root` 1.2 GB, ~106 k / 109 k events |
| readout | `sample_period` **60 ns**, `n_samples_per_waveform` **20** (= 1200 ns), `latency` 27, IPD 5, **`zero_suppress` False (full 512-ch raw)** |
| gas | **Ar/iC₄H₁₀ 90/10**, single line daisy-chained A→B→C→D |
| HV | drift **700 V on all four**; resist A 540 / B 540 / C 525 / D 520 V |
| trigger | PS + SINGLES (wall M1 0.5 MIP **AND** plastic M2 0.90 MIP), Hwm 2 / Lwm 1 |
| beam | neutrons, ³He target, no Pb |
| n_TOF partner | run **224572** covers *both* sub-runs (bunches 146–1157 and 1165–2213) [from the merge handoff] |

Detectors and FEUs (`det_labels.py`, and the run config's `dream_feus`):

| beam name | bench detector | FEU x | FEU y | bench wft state |
|---|---|---|---|---|
| mx17_A | **det3** | 3 | 4 | full v3 chain, `calib_bundle_lp2`, gap map |
| mx17_B | det2 | 5 | 6 | `calib_bundle_lp` (v pinned 39.94), gap map |
| mx17_C | det6 | 7 | 8 | old-kernel bundle only |
| mx17_D | det7 | 1 | 2 | `calib_bundle_v36` (old kernel), fast-gas anomaly |

Two pieces of plumbing already work unchanged **[verified today]**:

* `common.Mx17StripMap.RunConfig(<beam run_config>, mx17_m1_map.csv)` loads the
  beam config and `get_detector('mx17_A').map_hit(3, ch)` returns X positions
  for all 512 channels, 0–398.58 mm — i.e. `wft.io.strip_position_map` works
  on beam configs as-is.
* `decoded_root` file names end `_NN.root` with NN = FEU, so
  `wft.io.subrun_files` globs correctly; each event carries 10 240 = 20 × 512
  amplitudes plus `ftst`. No `_pedestals_` files are present in the tree.
* `combined_hits` carries `significance`, `trunc_left`, `trunc_right` — this
  run was processed with the post-7-24 matched-filter analyzer, so
  `wft.seed`'s significance floor is meaningful here.

---

## 2. The answer to the calibration question, up front

**No, we cannot reconstruct run_79 with the bench bundles as they stand — and
no, we do not have to recalibrate everything either.** The bundle splits
cleanly into hardware constants that transfer and gas/DAQ constants that do
not. What changed between the bench bundle (`sat_det3`: resist 490 V, drift
1000 V = 333 V/cm, **Ar/iso 95/5 + ~0.8–1 % H₂O**, 32 × 60 ns) and run_79
(resist 520–540 V, drift 700 V = 233 V/cm, **Ar/iso 90/10**, CERN 720.8 Torr,
20 × 60 ns) is the *gas and the field*, not the detector.

| bundle field | transfers? | why / what to do |
|---|---|---|
| `tmpl['x'/'y']` impulse response | **probably** — verify | set by DREAM shaping + gap signal formation; both chambers and (assumed) the DREAM `.cfg` are the same. **Re-measure in situ (cheap) and compare shapes**; only freeze the bench one if they agree. The July `.cfg` (`Tcm_Mx17_July.cfg`) vs the bench `CosmicTb_MX17.cfg` are on the DAQ boxes, not the laptop — diff them, an unequal peaking time invalidates the transfer outright. |
| `c1`, `c2`, `kY` (sharing amplitudes) | **yes** — measured | ±25 % costs < 2 % of σθ (`mx_june_wft/WINDOW_ABLATION_2026-07-30.md` §3). Carry them. |
| `tau_s`, `kTauY` (sharing timescale) | **verify, don't assume** | RC of the resistive strip layer — hardware, so it *should* transfer, but ±25 % costs ±9 % of σθ Y, so it has to be checked with the reference-free neighbour-copy measurement (`bench/rc_line_step3.py` port). |
| the kernel **as a whole** | **positions yes, angles NO** | Measured (§4 of the ablation doc): transferring a kernel leaves within-5 mm and core σ unchanged but costs **21–22 % of σθ across two runs of the same detector** and **43–53 % across detectors**. A bench bundle can reconstruct run_79 *positions* today; angles need the in-situ verification first. |
| `gain` | yes (unit) | unmeasured everywhere but det3, ablated as sub-noise. |
| `v_drift` | **NO** | different gas, different field, and a per-detector water gradient. Bench 36.6 µm/ns vs a clean-Magboltz prediction of **42.6** at 233 V/cm CERN P [from `mx_july_beam_qa/DRIFT_WINDOW_ANALYSIS.md` §1] — and B/C/D breathed ~0.8 % H₂O on 07-19, which cost C/D ~20 % (33.0 / 34.2 µm/ns at 800 V). **Must be measured per detector, in situ, on this run.** |
| `sigma_p0`, `Dp` (cloud size, transverse diffusion) | **NO** | gas- and field-dependent. Magboltz gives the ratio; a short in-situ refit gives the value. |
| `w0`, `kw` (angle mapping) | as a **prior** | measured against M3 on the bench; small (w0 ≈ −0.2 µm/ns ≈ 0.3°, kw ≈ 1 under the lp kernel). No M3 here — validate against target pointing (§7). |
| `sample_ns`, `n_depth_bins`, `sat_adc` | **NO** | DAQ/run properties: 20 samples not 32, and the beam ADC ceiling is not the bench's 3550. |
| drift gap (used to turn a column time into v) | **assumed yes** | bench-measured **det3 27.9 mm dished, det2 30.5 mm flat**; det6/det7 never mapped [from `mx_june_wft/GAP_STUDY_2026-07-30.md`]. Assumes the cathode travelled with the chamber — state it, don't hide it. |

The CLAUDE.md rule ("a bundle used outside its conditions is a silent error")
therefore applies with force: **run_79 gets its own bundles**, four of them,
seeded from the bench kernels, named e.g. `calib_bundle_run79_A`.

---

## 3. The three things that make run_79 harder than the bench

### 3.1 The window is 20 samples, and the deep end of the column may not be in it

The bench fit uses K = 18 charge bins × 60 ns = 1080 ns of drift, inside a
1920 ns window. run_79 gives 1200 ns *total*, with the drift onset placed at
sample ~2 by the run_78 latency scan and the run config claiming "20 samples
holds 95 % of the drift charge plus margin".

Full-gap drift at 233 V/cm: ~700 ns (≈ 12 samples) if the gas is as dry as
det A was on 07-19, ~950–1000 ns (≈ 16–17 samples) at 0.8 % H₂O
**[inferred, scaled from `DRIFT_WINDOW_ANALYSIS.md` §1c]**. A single pulse is
~11 samples wide (rise onset at peak−4, back to baseline at peak+7)
[from the same doc §2b]. So on the wet chambers the *last* primary's pulse
plausibly runs off the end of the window, and 31 % of all hits already carry
`trunc_right` and 23 % `trunc_left` **[verified today]** (mostly small hits,
but the number is a warning, not a footnote).

Charge loss is not the question — **reconstruction loss** is. The fit reads the
slope off the whole ladder; cutting the deep end removes the long lever arm and
lets NNLS push the missing deep charge into bins that are still visible. That
has to be quantified, and it can be, on the bench, against M3 truth (§4.1).

Consequences already visible in the design: `K` must come from the window
(≈ 15–16 bins, not 18), `q_uend`'s plausibility gate (`U_MIN_NS` 250 /
`U_MAX_NS` 1100 in `wft/reco.py`) must be re-tuned per run condition, and a
`column_truncated` flag belongs in the output table.

### 3.2 There is no reference telescope, and calibration is currently ref-pinned

`wft.calibrate` is explicitly built around M3: `build_cache` refuses to run
without the hits-chain alignment and the M3 ray file, `fit_hypers` pins
(p0, w) to the reference, `measure_w0` needs `tan_ref`. **None of that exists
at n_TOF, and there are no cosmics there** [README correction 2026-07-12].

What is reference-free already, or can be made so:

* `measure_templates` — uses the reference only to *select* inclined tracks.
  Inclination can be selected from the hits (a legitimate hits use: which
  events, not what geometry).
* `measure_dt_xy` — a median t₀ difference; works off free fits.
* the sharing-kernel shape measurement (`bench/rc_line_step3.py`) — needs
  near-vertical tracks, selectable geometrically at beam (see §6.4).
* `bench/gap_study.py`'s stacked NNLS charge-arrival profile + erfc endpoint —
  needs *no* per-event truth at all; with the gap known it gives **v**.

What is genuinely lost: the ref-pinned 8-parameter hyper fit, and the absolute
angle scale (`w0`/`kw`). §6 replaces the first with a frozen-kernel 3-parameter
free-geometry fit, and §7 replaces the second with target pointing and the
scintillator arm.

### 3.3 The occupancy is nothing like a cosmic bench  [verified today]

One hits file of `stat090_0000` (8 426 events, 2.45 M hits):

| FEU | 1 (D_x) | 2 (D_y) | 3 (A_x) | 4 (A_y) | 5 (B_x) | 6 (B_y) | 7 (C_x) | 8 (C_y) |
|---|---|---|---|---|---|---|---|---|
| median hits/event/plane | 44 | 38 | 19 | 24 | 7 | 10 | 3 | 27 |
| p99 | 464 | 467 | 412 | 462 | 353 | 512 | 305 | 513 |
| median amplitude [ADC] | 293 | 226 | 138 | 96 | 100 | 101 | 115 | 75 |
| saturated fraction | 1.5 % | 2.1 % | 0.2 % | 0.2 % | 2.0 % | 1.0 % | 1.8 % | 1.7 % |

A bench cosmic gives ~6 strips in one cluster. Here a median event has tens of
hits per plane and the tail reaches full-plane. `wft.seed`'s production
seeder — significance floor, 12 mm gap clustering, largest-3 clusters, spark
veto at **50 hits** — will veto a large fraction of run_79 events outright and
will mis-seed many of the rest. **The seeder is the single biggest piece of new
code this plan needs.** The good news is it already exists in another form:
`reco/noise.py` (coherent time-band finder, isolated-hit removal, hot-channel
mask) and `reco/segments.py` (spatio-temporal clustering + cluster taxonomy
`{track, point, band_fragment, blob}`) were written for exactly this data, and
`mx_july_beam_qa/24_drift_time_edges.py` already carries the monster-event cut.

An open, scope-setting question follows from this: **how many run_79 events
contain a gap-crossing column at all?** At run_30 only ~3 % of compact clusters
were inclined, most n_TOF deposits being point-like and isochronous [README].
The forward model describes a column crossing the gap; a point deposit is not
that. §5.3 measures this before anything expensive is built on top of it.

---

## 4. Phase 0 — answer the calibration question on the bench, before touching run_79

Both of these run on bench data with M3 truth, on the existing benchmark
harness (`mx_june_wft/bench/build_cache.py` + `run_bench.py`, which already
supports named variants, `--patch`, `k_bins` and bundle swaps). Neither needs
new data. **This phase is the actual deliverable the user asked for** — it
converts "do we need in-situ calibration?" from an opinion into a measurement.

### 4.1 The window ablation — what does a 20-sample window cost?

Add a `trunc20` variant that crops each cached det3 window from 32 samples to
the 20 that run_79's framing would have recorded (and a `trunc20_k15` that also
shortens the charge basis), then score against M3 exactly as the det3 gate did.

```bash
.venv/bin/python mx_june_wft/bench/run_bench.py sat_det3 \
    --bundle <det3>/wft/calib_bundle_lp2 --variant trunc20 --subset 1500 --jobs 5
```

Read out: within-5 mm, core σ, σ_θ per plane, angle bias, and the implied-v
spread across angle bins (the compression signature). Baseline to beat:
93.54 % / 0.460 mm / 1.08–1.11° [from `HANDOFF_2026-07-30.md`].

Do it twice — once at the det3 1000 V run (column ~800 ns, fits in 20 samples)
and once at the **drift-scan 700 V subrun** (`drift_scan_resist_490V_drift_700V`,
v = 21.6 µm/ns wet, column ~1.4 µs, genuinely truncated). The pair brackets the
two regimes run_79's dry-A and wet-B/C/D fall into, and tells us whether the
degradation is graceful or a cliff.

**Decision it feeds:** whether run_79 angles are quotable at all on the wet
chambers, and what `K` and the `q_uend` gates should be.

### 4.2 The transfer ablation — is a kernel + new v enough?

Take the det3 bundle calibrated at 1000 V and reconstruct the 900 V and 700 V
drift-scan subruns with **only `v_drift` changed** to that point's measured
value, against the same subrun reconstructed with its own full calibration.
The drift scan changes v by a factor ~1.7 across 700→1100 V — a bigger swing
than 95/5→90/10 at fixed field — so if angle bias and σ_θ survive a v-only
transfer there, the run_79 plan (freeze kernel, re-measure v and diffusion) is
justified with evidence rather than by assertion.

Then repeat the same v-only transfer across a **resist**-HV step (`hv_scan`
subruns), which changes gain but not the gas: that isolates whether the
amplification-field change (490 → 520–540 V at beam) touches the response at
all.

### 4.3 Which hypers actually matter

A one-at-a-time sensitivity scan on the bench cache: perturb `c1`, `c2`, `kY`,
`tau_s`, `sigma_p0`, `Dp`, `v` by ±10, ±25 % and record the induced angle bias,
σ_θ and position shift. Cheap (it is the same objective the calibration already
evaluates) and it gives the in-situ calibration a priority order: anything whose
±25 % perturbation moves σ_θ by less than the statistical error does not need to
be re-measured at n_TOF at all.

**Exit criterion for Phase 0:** a table of "bench numbers under the run_79
window and under a v-only transfer", plus the hyper priority order.

> **DONE, 2026-07-30 — `mx_june_wft/WINDOW_ABLATION_2026-07-30.md`.** All three
> questions are answered and the answers change this plan:
> 1. **The 20-sample window costs angles, not positions.** within-5 mm and core
>    σ are flat from 32 down to 14 samples; σθ Y goes 1.11 → 1.22° at the run_79
>    framing, with a compression bias of −0.41°. Per chamber, by measured tail
>    margin: A ≈ n 20, D ≈ 18, C ≈ 16–17, B off the end of the scan.
> 2. **The frame is ~2–3 samples too early.** Moving the signal later at fixed
>    n = 20 halves the compression bias for free (latency 27 → 29–30). The
>    leading edge constrains t0 and is worth more than the same number of tail
>    samples. Shrinking K does *not* help.
> 3. **Sensitivity**: `c1`, `c2`, `kY`, `sigma_s` transfer (< 2 % of σθ at
>    ±25 %); `sigma_p0`, `Dp`, `tau_s`, `kTauY` must be measured or verified in
>    situ. **Transfer**: a foreign kernel keeps positions but costs 21–53 % of
>    σθ. **v_drift never enters the fit** (position bit-identical at v ±10 %) —
>    it is a pure post-hoc angle scale, though a 10 % error still inflates σθ by
>    12–57 %, so it must be right to ~1 % before angles are quoted.
>
> Net effect on the plan: **Phase 1–3 can produce run_79 positions with a bench
> bundle immediately**; angles wait on the in-situ diffusion + timescale
> measurement (§6.4–6.5). Update the `RECONSTRUCTION_BASIS.md` migration table
> when the first beam positions land.
>
> Detector status from the same campaign
> (`mx_july_beam_qa/HANDOFF_2026-07-30_readout_window_and_detB.md`): **A and C
> are healthy; B's drift field is not set by its supply** (flat column length
> across 700→200 V on run_58, zero bleeder current at all nine voltages — its
> degrador divider is absent); **D is unexplained** (column length flat at
> 19 samples across the whole drift sweep). **Build the beam chain on A, then
> C.**

---

## 5. Phase 1 — plumbing (no physics)

### 5.1 A beam config adapter — `ntof_tracking/wft_beam.py`

`wft` never imports `qa_config` in its hot path; `reco.reconstruct_run`,
`io.strip_position_map` and `_load_hits` only touch duck-typed attributes.
A ~60-line `BeamConfig` supplies them:

```python
KEY, BASE_PATH='/media/dylan/data/x17/beam_july/runs/', RUN='run_79',
SUB_RUN='stat090_0000', DET_NAME='mx17_A', MX17_FEU_X/Y, MX17_FEUS,
MAP_CSV_PATH, run_config_path, combined_hits_dir (trailing '/'),
OUT_BASE=<analysis>/run_79/<subrun>/<det>, out_dir(*parts)
```

plus a `conditions` dict (gas, drift HV, resist HV, sample_ns, n_samples) read
from the run config, so `CalibrationBundle.check_conditions` has something real
to check. Output tree under
`/media/dylan/data/x17/beam_july/analysis/wft/run_79/<subrun>/<det>/`, **not**
the repo.

### 5.2 A beam seeder — `ntof_tracking/wft_seed_beam.py`

Same contract as `wft.seed.seeds_from_hits` (`{eventId: {'x': [Seed], 'y':
[Seed], 'n_hits', 'spark'}}`, channels only, never times-as-geometry), but:

* per-file-tag, streaming — the whole-run `uproot.concatenate` in
  `wft.reco._load_hits` would pull ~32 M hits into one DataFrame per sub-run;
  the hits files are already one per tag, so seed tag-by-tag and hand the
  seeds to `_stream_windows` for the matching decoded pair (this needs a small
  `seed_fn`/per-tag hook in `reconstruct_run` — keep the bench path
  bit-identical, `wft/tests/test_model_regression.py` must stay green);
* coherent-band and hot-channel rejection from `reco/noise.py`;
* cluster taxonomy from `reco/segments.py`, keeping `track`-class clusters and
  demoting `point`/`band_fragment`/`blob`;
* a beam-appropriate busy/flash veto (`BUSY_DET_STRIPS=120` clean strips in
  `reco/search.py`, and the monster cut of `24_drift_time_edges.py`) instead of
  the bench's `SPARK_VETO_HITS = 50`;
* `N_CANDIDATES` raised (3 → 5) because the occupancy makes multi-cluster
  events the norm, with `WFT_PAIR_SELECT=1` (the X/Y t₀-coincidence rule in
  `reco.select_pair`) switched **on** — it exists, was never needed on the
  bench, and this is the regime it was written for.

### 5.3 The scope measurement (do this before anything else in Phase 2)

Run the seeder alone over both sub-runs and report, per detector:
events with ≥1 track-class cluster per plane, with both planes, and the
distribution of cluster strip-count and hit-time span. That number — plausibly
a few percent, i.e. 5–20 k events over the two sub-runs — sets the whole cost
model, decides whether all four chambers are worth doing, and is a legitimate
hits-level QA answer under `RECONSTRUCTION_BASIS.md`.

### 5.4 The framing measurement

Run `mx_july_beam_qa/24_drift_time_edges.py` (per-connector common mode,
robust thresholds, monster cut) on run_79 per detector and plane: prompt onset
sample, deep-edge percentiles, ceiling occupancy at sample 19. This gives the
t₀ prior for the fit, `K`, the `q_uend` gates, and the first read on §3.1's
truncation risk — and it is the same estimator that produced the numbers this
plan quotes from 07-19, so the comparison is apples to apples.

---

## 6. Phase 2 — the in-situ calibration, per detector

Order matters; each step feeds the next. Do **A (det3) first**: it is first in
the gas chain and was dry on 07-19, it has the best bench bundle, and it has
the mapped drift gap.

1. **Seed bundle.** Copy the bench lp bundle (`calib_bundle_lp2` for A,
   `calib_bundle_lp` for B; C/D have no lp bundle — either run the fleet lp
   rollout on the bench first, or accept the old kernel and say so), then
   overwrite `sample_ns`, `n_depth_bins` (from §5.4), `sat_adc` (from the
   decoded amplitude ceiling per FEU), and the `conditions` dict.

2. **Template, measured in situ.** Port `measure_templates` to take an
   externally supplied "inclined, bright" event list instead of `tan_ref`:
   clusters of ≥6 strips whose hit `max_sample` ladder spans ≥5 samples
   monotonically (hits selecting events, not defining geometry). Compare
   rise₁₀₋₉₀, peak position and undershoot to the bench template. **Agreement
   within a few percent ⇒ freeze the bench template** (it is better measured);
   disagreement ⇒ use the in-situ one and find out why (DREAM `.cfg` diff).

3. **v_drift — two independent handles, and they must agree.**
   * *Column endpoint*: port `bench/gap_study.py` — stacked NNLS charge-arrival
     profiles over many tracks, erfc endpoint → T_full; v = gap / T_full with
     the bench gap (A 27.9 mm, B 30.5 mm; C/D 30 mm nominal ± the unmapped
     dish). Reference-free.
   * *Magboltz*: `garfield_sim` Ar/iso 90/10 at 233 V/cm, CERN 720.8 Torr →
     42.6 µm/ns clean, with the water grid giving the wet branch. For **A**
     (dry) this is a prediction to test; for **B/C/D** invert it to read the
     water content, which is a gas-line diagnostic worth having anyway.
   * **Do not** use the fit's own χ²(v) valley: it is kernel-degenerate and
     runs to ~40 µm/ns on both kernels [from `GAP_STUDY_2026-07-30.md`].
   * Repeat per sub-run (and per hour if the two disagree): the 16 h run can
     dry out or wet up, and v drifting under us would masquerade as an angle
     scale error.

4. **Sharing kernel — frozen, but verified.** Port `bench/rc_line_step3.py`'s
   neighbour-copy shape measurement onto near-vertical beam tracks (selectable
   as clusters of ≤4 strips near the chamber centre, where a target-pointing
   track is normal to the plane). If the ±1 copy's low-pass time constant
   matches the bench 230 ns X / 410 ns Y within ~20 %, freeze `c1`, `c2`, `kY`,
   `tau_s`, `kTauY`. If it does not, the transfer assumption is dead and
   Phase 2 becomes a full free-geometry hyper fit (much more expensive — call
   it out rather than absorbing it).

5. **Diffusion — a small free-geometry refit.** With the kernel and template
   frozen, fit only `{sigma_p0, Dp}` (and optionally v as a cross-check on
   step 3) by replacing `fit_hypers`' ref-pinned inner loop with the production
   free fit (`_global_start` + `fit_plane_raw`), on ~150 clean events. Three
   parameters against a frozen 5-parameter kernel is a very different
   degeneracy problem from the bench's 8; seed from Magboltz-scaled bench
   values so a failed fit is visible as "it didn't move".

6. **`dt_xy`.** Free fits, median (t₀x − t₀y) per `ftst` difference. No truth
   needed. Feeds `select_pair`, which §5.2 turns on.

7. **`w0` / `kw`.** Carry the bench values as the prior; refine or falsify in
   §7.2. Under the lp kernel kw ≈ 1 and w0 is a ~0.3° effect, so this is a
   systematic to bound, not a blocker.

8. **Re-run** the reco with the final bundle and record everything in a
   `RUN79_CALIB_<det>.md` with the same provenance discipline as the bench
   bundles (the `conditions` field is what stops this bundle being used on
   run_55 by accident).

---

## 7. Phase 3 — validating a reconstruction with no telescope

Internal consistency alone is *blind* to a bias that hits both planes the same
way — the README's own warning. So the validation has to reach outside the
chamber. Three handles, in increasing strength:

### 7.1 Internal closure
Per event, (t₀x − t₀y) must equal the measured `dt_xy` — a per-event closure
test the fit does not impose unless `PAIR_SELECT` chose the pair on it (report
it on single-candidate events, where it is free). Plus χ²/dof distributions,
`q_uend` vs the window ceiling, and the X/Y charge balance already used in
`reco/pairing.py`.

### 7.2 Target pointing — the substitute for the M3 corridor
The tracks come from a point-ish source: the ³He target at the origin, with the
chamber 234.6 mm away and 400 mm across. So for a track crossing chamber A at
local position u, the *expected* incidence is tan θ ≈ u / 234.6 — a strong,
purely geometric position–angle correlation spanning ±40°. Fit the measured
(p₀, tan θ) relation: **the intercept is `w0`, the slope is `kw`**, and the
scatter around it bounds σ_θ (inflated by the target size, scattering and
non-target background, so it is an upper limit, not a measurement).

This is the closest thing to truth available and it is cheap. Caveat to state
loudly: it assumes the strip-axis signs/offsets and the global frame are right
(PLAN_01/04's open in-plane calibration), so a failure here is ambiguous
between reconstruction and alignment until §7.3 breaks the tie.

### 7.3 Scintillator pointing — the killer plot, and the reason for the merge
Every run_79 DREAM trigger is a wall **AND** plastic coincidence in one arm.
Through the merge we know *which* n_TOF channels fired: the SiPM wall is
**16 bars of 25 mm** at z = 332 mm for arm A (89 mm downstream of the chamber
plane), the plastics are two 20×30 cm bars at z = 425 mm, the liquid at 483 mm.
So:

* extrapolate the reconstructed 3-D segment to the wall plane and compare the
  predicted bar with the bar that actually fired — **25 mm granular external
  position truth, with a ~90 mm lever arm**, on every matched event;
* the same for the plastic L/R bar (a 200 mm-scale binary check that also
  re-tests the "missing plastic" acceptance story of the merge handoff §5);
* the wall's top/bottom timing gives a coarse coordinate *along* the bar,
  i.e. the second transverse direction.

Two independent things get validated at once: the tracking (does the track
point where the light was?) and the matcher (are we joining the right DREAM
event to the right n_TOF pulse?). A mismatch that is flat in time is
tracking/alignment; one that grows at 1–3 ms is the matcher's known false-match
region. **This is the plot to aim the whole plan at.**

### 7.4 Bench mirror
Finally, run the *beam-configured* pipeline (beam seeder, truncated window,
in-situ-style calibration) on a bench run with M3 available, and check it
reproduces the Phase-0 ablation expectations. Any surprise here is a bug in the
new code, not physics.

---

## 8. Phase 4 — the merged record

Output of Phase 3: one parquet per (sub-run, detector),
`events.parquet` + `events.meta.json` in the `wft` schema (`event_id`,
per-plane `p0/w/t0/tan_theta/theta_deg`, errors, χ², charge-profile summary,
quality flags) plus the beam additions: `column_truncated`, `n_candidates`,
global-frame position/direction from `reco/geometry.py`, and the arm.

Join, reusing `ntof_dream_merge` exactly as `ntof_processing/dream_regression.py`
does it (order matters — build the bunch join first, *then* repoint the reader,
sandbox `CACHE_DIR`, and turn `repair_tflash` **off** against v12):

```
event_id → bunch_join.dream_event_to_bunch(run_79, <subrun>, 224572)
        → intra_burst_align (k ≈ 108.9 ppm, t0 ≈ −198 ns, fitted per sub-run)
        → dream_trigger wall-SINGLES matcher (96.3 %/0.5 % on v12)
        → per-arm wall / plastic / liquid amplitudes, times, satuflag-and-ceiling
          cuts (`ntof_io.saturated`, WAL amp > 34 600)
        → t_since_flash (flash_timing: t_flash = tof_PKUP + C) → E_n over 19.5 m
```

Final table: one row per DREAM event with a reconstructed track, carrying the
track, the matched arm's scintillator record, the pulse intensity, the match
confidence flag, and E_n. Deliverable figures: §7.3's pointing plot, and
PLAN Phase 5's track rate + scint-tagged track rate vs `t_since_flash` with the
E_n bands overlaid (the mid-window turn-off and the ³He capture flood must land
where the July QA says they do).

---

## 9. Order of work and rough cost

| step | what | cost |
|---|---|---|
| 0 | Phase 0 bench ablations (§4.1–4.3) | ~1 session, no new data, CPU-light (1 500-event variants) |
| 1 | `wft_beam.py` + beam seeder + per-tag streaming (§5.1–5.2) | ~1 session of code |
| 2 | scope + framing measurements (§5.3–5.4) | ~1 h CPU per sub-run |
| 3 | detector A calibration (§6) | ~2 h CPU + judgement |
| 4 | detector A reco, both sub-runs | scale from det3's 13 min/run at 8 jobs by the seeded fraction |
| 5 | validation §7, then the merge §8 | 1–2 sessions |
| 6 | B/C/D repeat (C/D need a bench lp calibration first, or an explicit old-kernel caveat) | ×3 |

Nothing in steps 0–2 needs the cosmic reprocessing to be finished; step 0 needs
only the existing bench caches and bundles, which are all on the laptop.

## 10. Risks, in the order they can kill the plan

1. **Too few gap-crossing tracks.** If §5.3 returns a few hundred usable events
   per detector, the merge physics is statistics-limited and the whole plan
   should be re-scoped toward "reconstruct the best N events well" rather than
   a production pass. Measure first.
2. **Truncation on the wet chambers.** If §4.1 shows a cliff and §5.4 shows
   B/C/D riding the sample-19 ceiling, their angles are not quotable; positions
   at the mesh probably still are (they are far less affected — the bench gate
   showed position at parity while angles halved). Say which is which.
3. **The template does not transfer** (different DREAM shaping). Recoverable —
   measure in situ — but it invalidates any bundle built before the check.
4. **The kernel does not transfer** (temperature/resistivity). Expensive:
   a free-geometry hyper fit with no reference. Test it early (§6.4).
5. **Angle scale unverifiable.** §7.2 and §7.3 bound it; if they disagree, the
   in-plane strip-axis signs/offsets (PLAN_01/04's open item) are the prime
   suspect, not the fit.
6. **v drifting through the 16 h run.** Monitor per sub-run; a moving v is a
   moving angle scale.
7. **Merge-side traps** are all documented and all have bitten someone before —
   cache sandboxing, join-before-repoint, `repair_tflash=False`, never `hadd`
   a run, don't trust `match_window` efficiency at early times.

## 11. Open questions to settle before or during execution

* Is `Tcm_Mx17_July.cfg` the same shaping as the bench `CosmicTb_MX17.cfg`?
  (DAQ boxes; a one-line diff that decides §6.2.)
* Did the chambers keep their cathodes between the June bench and the July
  installation? Decides whether the det3 27.9 mm dished gap applies at n_TOF —
  and hence whether the column-endpoint v is biased ~7 % on A.
* Do we run the bench lp rollout for det6/det7 first (giving C/D a modern
  kernel) or accept old-kernel bundles at beam with a stated caveat?
  (`mx_june_wft/HANDOFF_2026-07-30.md` open item 1 — user approval was det3-only.)
* Which sub-run(s): both, or `stat090_0000` first? Both are needed for the full
  2 061-bunch merge, but one is enough to validate.
