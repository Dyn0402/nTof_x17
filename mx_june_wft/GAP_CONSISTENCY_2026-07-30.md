# Does the drift-gap map reproduce? — the consistency campaign, 2026-07-30

Follow-up to `GAP_STUDY_2026-07-30.md`, which concluded from one det3 dataset
and one det2 control that det3's cathode sits ~2 mm short. The question asked
here: **does each detector give the same gap map in independent runs?** A real
chamber geometry must; a reconstruction artefact need not.

Short answer: **the chamber-to-chamber contrast is real and reproduces (det2 −
det3 = +2.8 to +3.4 mm under every calibration tried), but the ABSOLUTE column
carries a ±1 mm calibration systematic that the original report does not quote.**
"det3 = 27.9 mm" is really 27.4–29.8 mm depending on which equally-valid
calibration bundle you use, against a 30 mm mechanical gap. Every differential
statement survives; the headline absolute number needs the error bar.

## What was run

| dataset | detector, slot | what it tests |
|---|---|---|
| `sat_det3` 6-27 | det3, top (FEU 7/8) | the reference map |
| `g_det3_wknd` 6-28 P2 | det3, top, mount untouched | run-to-run repeat |
| `g_det3` 6-22 long | det3, **bottom** (FEU 3/4) | independent mount — **unusable, see below** |
| `o22_long_det2` 6-22 | det2, top | the control chamber |
| `g_det2` 6-22 long | det2, top, 8x stats subrun | control repeat |
| det3 drift scan 700/900/1000/1100 V | det3, top | drift-field invariance |
| det4 / det6 / det7 | first maps | fleet (in flight) |

Each dataset got its own RC-ladder calibration, reconstruction, alignment,
bench cache, w0/kw retrofit and gap study — the approved rollout recipe of
`HANDOFF_2026-07-30.md`, driven by `bench/gap_consistency.sh`.

## 1. The data x bundle matrix — the central result

Every dataset fitted with every dataset's calibration bundle (200 condor jobs,
`bench/gap_matrix.py`). Charge-visible column [mm], X plane:

| data \ bundle | det3 sat | det3 P2 | det3 6-22 | det2 longer | det2 long |
|---|---|---|---|---|---|
| **det3 6-27 sat** | 27.80 | 29.48 | 28.53 | 27.41 | 29.32 |
| **det3 6-28 P2** | 27.97 | 29.78 | 28.83 | 27.52 | 29.47 |
| det3 6-22 bot | 11.59 | 11.70 | 26.66 | 11.51 | 11.57 |
| **det2 6-22 longer** | 31.10 | 32.34 | 31.37 | 30.52 | 31.95 |
| **det2 6-22 long** | 31.52 | 32.60 | 31.76 | 30.69 | 32.42 |

Read it two ways:

- **Down a column (fixed calibration, different data) — the physics.** The two
  det3 runs agree to **0.2-0.3 mm** in every column; the two det2 subruns to
  ~0.3 mm. The chambers separate cleanly and consistently.
- **Across a row (fixed data, different calibration) — the systematic.** Each
  dataset spans **1.8-2.3 mm** (rms ~0.9 mm) across five legitimate bundles.

Chamber contrast, computed per bundle so the systematic cancels:

| bundle | det3 mean | det2 mean | contrast |
|---|---|---|---|
| det3 6-27 sat | 27.88 | 31.31 | **+3.43** |
| det3 6-28 P2 | 29.63 | 32.47 | **+2.84** |
| det3 6-22 bot | 28.68 | 31.57 | **+2.89** |
| det2 6-22 longer | 27.46 | 30.61 | **+3.14** |
| det2 6-22 long | 29.39 | 32.19 | **+2.79** |

**det2 reads 2.8-3.4 mm longer than det3 no matter whose calibration is used.**
That is the `GAP_STUDY` conclusion, confirmed and if anything strengthened
(2.6 mm was quoted). It is not a calibration artefact, not a gain artefact and
not a reference artefact.

## 2. How the calibration systematic was found

The det3 repeat looked like a **failure** at first: 27.81 mm on 6-27 vs
29.76 mm on 6-28, same chamber, alignment identical to 0.02 mm, one day apart.
Ruled out one by one:

- **gain / signal size** — the endpoint does depend on amplitude (+1.5 mm across
  det3's charge range, +2.2 mm on det2: weak signals fade below the fit's
  sensitivity earlier), but the sat-vs-P2 offset is ~2.1 mm in *every* charge
  quintile, so it is not that;
- **assumed drift speed** — scanning the model's v from 34 to 40 um/ns moves the
  column by **<0.01 mm** (`bench/gap_vpin_test.py`); the pin is irrelevant;
- **M3 reference angle scale** — the two runs' |tan| distributions agree to
  0.2 % (median 0.1036 vs 0.1034);
- **DAQ time base** — cancels exactly in the product `T_end x w / tan`.

The cross-test settled it: swapping the *bundle* on fixed data moves the column
by 1.7 mm, swapping the *data* at fixed bundle by 0.06-0.4 mm. The column
follows the calibration. The two bundles differ in template and hyper
(`tau_s` 146 vs 132 ns, `sigma_s` 12 vs 40 ns, `Dp` 0.0134 vs 0.0032) — both
are legitimate ref-pinned fits of the same detector at the same conditions.
This is the known kernel degeneracy showing up in a new place: it barely moves
positions and angles, but it shifts the arrival-time endpoint and the slope
scale together, and the column is their product.

**Consequence for quoting numbers.** Absolute columns must carry ~±1 mm
(systematic, calibration) — det3 = 27.9 ± 0.1 (stat) ± 1.0 (calib). Differences
computed at a common bundle do not: the systematic is a near-uniform scale
factor, so contrasts, run-to-run repeats and the (x, y) topography are clean.

## 3. Drift-field invariance (`bench/gap_vs_drift.py`)

Same chamber and bundle, drift field varied — the strongest test that the
column is a length and not a time:

| drift | v_geom (geometric) | T_end | column |
|---|---|---|---|
| 700 V | 25.31 | 1052 ± 45 ns | 26.63 ± 1.13 **[truncated]** |
| 900 V | 33.38 | 817 ± 10 ns | **27.26 ± 0.32** |
| 1000 V | 36.31 | 760 ± 3 ns | **27.59 ± 0.10** |

Drift time changes by 8 % while the column stays put to 0.3 mm. Below ~28 um/ns
the 30 mm column runs past the 1080 ns model basis and the endpoint becomes a
lower limit — the 700 V point is flagged, and this is also why **no det3 run
outside 6-27 can measure the gap** (see below).

## 4. det3 has no usable independent-mount dataset

The 6-22 bottom-slot run was meant to be the decisive test (different day,
different slot, different FEUs). It is unusable: the fitted drift speed is
13.5 um/ns, the endpoint fits are railed (chi2 ~9500, Y at its bound), and the
column reads 11.5-26.7 mm depending on bundle. This is **not** a wft artefact —
the independent hits-based fleet survey measured v ~ 8.9 um/ns for det3 that
night against 25.1 for det2 in the *same* run at the same nominal 1000 V. det3
was in a pathological drift state. Its other runs are no better for this
purpose: 6-23 at 600 V, 6-25 at 500 V, both far too slow for the column to fit
inside the readout window.

So the det3 dish is established from the 6-27 pair (same mount) plus the det2
control, **not** from a remount. Confirming it against a remount needs a new
run at drift >= 900 V with det3 in the bottom slot. Worth doing before the
result is quoted as chamber geometry in a paper.

## 5. Statistical quality of the maps

Split-half (even/odd events of the same dataset) sets the estimator's own noise
floor: det3 map rms 0.64 mm, correlation 0.83, against a 25.7 -> 29.2 mm
topography — the structure is ~3 sigma per grid cell. det2: rms 0.84, corr 0.63,
over a flatter 28.6 -> 31.8 mm range, consistent with "flat within errors".

## 6. Where the work runs now

The study outgrew this laptop (a systematics sweep is ~25 full refits). The
pipeline was split (`mx_june_wft/condor/`, see its README):

- **local, unavoidable**: reco / `build_cache` — they read the decoded
  waveforms and seed from the 7-24 reprocessed `combined_hits`;
- **condor**: the fits (`bench/gap_fit.py`, sharded 8 ways, ~50 MB input each)
  and the calibration hyper fit (`bench/calib_hyper.py`, 2 MB input). 200-job
  sweep + 3 calibrations ran in ~40 min wall clock with zero failures.

`bench/gap_merge.py` reassembles shards into exactly the `gap_study.json` +
`event_profiles.parquet` the local chain writes, so `gap_compare.py`,
`gap_map_hires.py` and `gap_charge_check.py` consume grid output unchanged.

## 7. Bug fixed on the way

`wft/calibrate.py::measure_dt_xy` never told the model the window length, so a
run mixing 32- and 37-sample waveforms died on a shape mismatch (`(448,18)` vs
`(518,1)` = 14x32 vs 14x37). It blocked det4's calibration entirely. Fixed (set
the sample count per event, skip un-fittable events);
`wft/tests/test_model_regression.py` still passes.

## What should change in GAP_STUDY_2026-07-30.md

1. Quote the calibration systematic on every absolute column (~±1 mm).
2. Keep the det3-vs-det2 contrast as the headline — it is bundle-independent.
3. Drop / footnote the claim that the 27.9 mm is confirmed against remounting;
   it is confirmed against a same-mount repeat, a drift-field scan and a control
   chamber. A bottom-slot det3 run at >= 900 V drift is the missing test.
4. Add the amplitude systematic (+1.5-2.2 mm across a run's charge range) —
   maps and contrasts should be quoted at matched charge.

## Open items

**All three were closed on 2026-07-31 — results in
`ANALYSIS_STATE_2026-07-31.md` §3.5.**

- ~~det4 / det6 / det7 first maps: 128 condor jobs in flight~~ **done** — the
  jobs had in fact completed on lxplus and were merged on 07-31. Fleet columns
  (own bundle, X): det2 30.53, det3 27.81, det7 27.75, det6 27.04 (endpoint fit
  **railed**), det4 25.46 (**dominated by the amplitude systematic**: its charge
  quartiles run 24.4 → 33.5 mm). Only det2 and det3 are fleet-grade.
- ~~Map-shape reproducibility~~ **done** — det3 6-27 vs 6-28: rms difference
  0.64 mm against a 0.42 mm split-half floor, **correlation 0.90** as-is versus
  0.30 mirrored. The topography is a property of the chamber.
- ~~`gap_charge_check.py` not yet run~~ — the charge-split null test now runs
  inside `gap_compare.py` and gives the per-chamber amplitude systematic quoted
  above; the dedicated q ∝ L test remains worth doing on det3/det2 only.
