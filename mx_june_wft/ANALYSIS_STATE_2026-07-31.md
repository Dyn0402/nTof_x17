# State of the cosmic-bench track reconstruction — 2026-07-31

**This is the entry point for anyone auditing the June cosmic-bench tracking
analysis.** It states what the analysis currently claims, which basis produced
each number, what is validated and what is provisional, the systematics that are
known and quantified, and the specific places where an auditor should push.

Everything below is about the **June 2026 cosmic bench** (five MX17 chambers on
the M3 telescope). The July n_TOF beam analysis consumes these results but is
documented separately (`ntof_tracking/`, `ntof_processing/`).

Companion documents, in reading order:

| # | document | what it is |
|---|---|---|
| 1 | `../RECONSTRUCTION_BASIS.md` | the canonical rule: geometry comes from waveforms, not hits |
| 2 | `RECO_BENCH_2026-07-29.md` | the fit benchmark campaign — every variant tried, what won, what was falsified |
| 3 | `DET3_GATE_2026-07-29.md` | the accept/reject gate against the old chain, both chains through one accounting |
| 4 | `FLEET_2026-07-29.md` | all five chambers, both chains |
| 5 | `GAP_STUDY_2026-07-30.md` + `GAP_CONSISTENCY_2026-07-30.md` | the drift-gap result and its reproducibility campaign |
| 6 | `WINDOW_ABLATION_2026-07-30.md` | what a short readout window and a foreign calibration cost (bench→beam transfer) |
| 7 | `HANDOFF_2026-07-30.md` | operational state and procedures |
| 8 | this file | current state, systematics register, audit guide |

---

## 1. The claim, in one page

1. **A per-strip hit time on these resistive-strip detectors is not a drift-time
   measurement.** It aggregates the strip's own charge with delayed, dispersed
   copies of its neighbours'. Reconstructing geometry from it compresses the
   drift-time ladder by 20–30 %, reads inclined tracks ~4° too steep, and walks
   the reconstructed cluster away from the true track with depth. This is
   estimator-independent (rising edge, CFD, matched filter all show it).
2. **The replacement is a forward fit of the whole (strip × sample) picture**
   (`wft/`): charge arriving in each 60 ns slice of drift, folded through the
   measured per-plane impulse response and a measured sharing kernel, fitted per
   event. The neighbours' copies become part of the model instead of
   contamination.
3. **Angles improve ~2× fleet-wide and become unbiased and angle-independent.**
   det3: σ_θ 2.42/2.60° → 1.08/1.11°, bias ~0, and the implied-v spread across
   angle bins — the direct compression signature — falls from ~17 to 1–3 µm/ns.
4. **Position is at parity with the old hits chain**, not better: within 5 mm of
   the M3 reference is parity to +4 points across the fleet, and the hits chain
   retains a marginally better core σ on several chambers. The waveform chain's
   advantage is angles and depth, which is what it was built for.
5. **Detection is untouched.** `has_any` and `spark_frac` are identical between
   chains by construction: detection is decided by the analyzer's trigger on the
   hits tree, not by the fit. If they ever move, the seeding is broken.
6. **Physics that came out of the rebuild**: det3's drift column is ~2.7 mm
   short of the mechanical gap with a reproducible dished topography, while the
   det2 control reads the full gap — chamber geometry, not gas (but see §3.5:
   three other chambers also read short, each with a systematic big enough to
   explain it, so det3 is not established as unique); the ±1-strip sharing copy is a
   low-passed template (an RC ladder), not a delayed-and-smeared one; the Y
   template carries a uniform resistive leak with τ_g = 5.3–7.3 µs per chamber;
   v_drift is a pure angle scale that never enters the fit.

---

## 2. The reconstruction as it stands

### 2.1 Model

Per plane, per event: a non-negative charge ladder q(u) on K = 18 depth slices of
60 ns, a transverse position p0 and a transverse speed w, convolved with

* the **direct-strip template** — measured per plane from the data;
* the **sharing kernel** — the ±1 neighbour copy is the template passed through a
  single-stage RC low-pass (`share_lp`), the ±2 copy two cascaded stages. The
  *shape* was measured directly on near-vertical tracks (the dim neighbour of a
  bright strip, n ≈ 1000/plane): τ ≈ 230 ns X / 410 ns Y, fitting the observed
  copy 14× (X) / 54× (Y) better than the old delay-and-smear form. The bundle's
  *fitted* values (det3: `tau_s` = 146 ns, `kTauY` = 1.78, i.e. 259 ns on Y) come
  from the ref-pinned hyper fit and are the ones actually used;
* transverse spread: an initial cloud size `sigma_p0` plus diffusion `Dp`.

The angle is **not** fitted directly: `tan θ = (w·10³ − w0) / (kw · v_drift)`,
with per-plane constants `w0` (transverse-speed offset of reference-vertical
tracks) and `kw` (slope scale). Under the RC-ladder kernel `kw` comes out ≈ 1.0,
i.e. the −3 % Y slope compression that `kw` used to patch empirically is
explained and removed by the copy shape.

### 2.2 Production configuration

```
WFT_MODEL_FRAC=0.03   3 % fractional model-error term in the chi2 weights
WFT_PRESCAN=1         coarse K=9 x 120 ns global pre-scan
WFT_CHI2DOF_BAD=250   quality flag, re-derived for the new chi2 scale
bundle: calib_bundle_lp / calib_bundle_lp2   (share_lp = 1 in the hyper)
```

Cost: det3's 7,093 events reconstruct in 13 min on 8 jobs (0.32 s per plane fit).

**All four settings are config-relative and must move together.** χ²/dof means
different things per configuration (unweighted ~110/180, mf3 ~20/60), and a
bundle is only valid with the kernel it was fitted under.

### 2.3 Calibration procedure (per detector, per run condition)

1. `python -m wft.calibrate <key> --jobs 8 --share-lp --fix-v <v>` — hypers
   fitted ref-pinned on a ~180-event corridor; v **pinned** to the chamber's
   independently measured drift speed (the fit cannot identify v on its own —
   see §5.2). Low-gain chambers need `--tmpl-tan-min 0.10 --tmpl-min-amp 250`.
2. Reconstruct with the production env above.
3. `bench/set_w0.py <key> --bundle <b> --write` — the **production** w0/kw
   retrofit, then re-reconstruct. The corridor values from step 1 are biased by
   ~0.06 µm/ns / 1.5 %; skipping this pass leaves a real angle bias.
4. Analysis chain `01_alignment` → `02_efficiency` → `03_angles` → `04_maps` →
   `digest`. These consume `events.parquet` only, so they are bundle-agnostic.

`mx_june_wft/rollout_lp.sh <key>` runs steps 2–4 with the backups.

---

## 3. Validated results

### 3.1 det3 — the golden run (`sat_det3`, 7,093 M3-matched events)

Both chains through the identical accounting of `02_efficiency.py`
(same M3 rays, same active box, same spark tagging), which reproduces the old
chain's published 93.4 % / 0.48 mm at 93.13 % / 0.448 mm — that is what
validates the accounting.

| metric | hits chain | wft v2 (old kernel + w0/kw) | **wft v3 (RC-ladder, production)** |
|---|---|---|---|
| within 5 mm | 93.13 % | **93.71 %** | 93.54 % |
| core σ \|r\| | **0.448 mm** | 0.467 | 0.460 |
| median \|r\| | 0.764 mm | 0.739 | **0.708** |
| σ_θ X / Y | 2.42 / 2.60° | 1.10 / 1.07° | 1.08 / 1.11° |
| σ_θ Y, reference-selected 0.08–0.45 | — | 1.05° | **1.02°** |
| σ_θ Y, near-vertical (<0.08) | — | 2.67° | **1.50°** |
| angle bias X / Y | — | −0.01 / −0.00° | −0.03 / −0.01° |
| implied-v spread | ~17 µm/ns | 2.5 / 1.2 | ~2 |
| has_any / spark_frac | 99.99 / 8.22 % | 100.0 / 8.22 % | 100.0 / 8.22 % |
| reco wall time (8 jobs) | — | 19 min | **13 min** |

σ_θ ≈ 1.05–1.1° is the **physics floor** set by diffusion and charge
granularity, established by toy closure (`WAVEFORM_FIRST_THREADING.md` §12), not
a fit limitation.

### 3.2 The depth test — the measurement the rule rests on

Charge-weighted median |cluster − M3 reference line| against drift depth, 600
det3 muons, from a line-free 2-D deconvolution (never from the fitted track, so
the comparison is not circular):

| depth [mm] | 3 | 9 | 15 | 21 | 26.5 |
|---|---|---|---|---|---|
| hits, common frame | 0.45 | 0.50 | 0.61 | 0.67 | 0.63 |
| hits, own t0 and v | 0.43 | 0.63 | 0.88 | 1.07 | 1.17 |
| **waveform-first** | **0.39** | **0.44** | **0.49** | **0.50** | **0.55** |

Both agree at the mesh, at the ~0.4 mm M3 pointing floor; only the hits cluster
walks away with depth.

### 3.3 Fleet (07-29 generation, old kernel + w0 — being superseded, see §4)

| detector | within 5 mm, hits → wft | σ_θ X/Y, hits → wft |
|---|---|---|
| det3 | 93.13 → 93.47 | 2.42/2.60 → 1.20/1.14° |
| det2 | 92.06 → 92.07 | 2.47/2.04 → 1.31/1.56° |
| det4 | 40.67 → 41.65 | 3.38/3.18 → 2.73/2.58° |
| det6 | 71.19 → 75.41 | 3.42/2.58 → 2.28/2.52° |
| det7 | 52.73 → 56.95 | *not usable — see §4* |

All five hits caches carry the 2026-07-25 significance floor, so this table is
apples to apples. The det2/det6/det7 "+10 to +31 point" gaps quoted earlier in
the campaign were an artefact of comparing against pre-floor caches and are
**withdrawn**; the corrected statement is parity to +4 points.

### 3.4 Drift gap — chamber geometry

Stacked NNLS charge-arrival profiles of M3-**contained** tracks (the reference
guarantees the track crosses the whole gap inside the active area), erfc
endpoint, converted to mm with the *geometric* drift speed v_geom = kw·v.

* **det3 27.9 mm, dished/tilted** (hi-res map 25.5 → 29.5 mm, worst along
  y > 300 mm); **det2 30.5 mm, flat** = the full mechanical gap. Same gas, same
  reconstruction, same estimator, same selection.
* **Gas attachment excluded** — both X profiles are flat to the endpoint
  (τ_att runs to 22 µs / ∞).
* **Edge fringe excluded for the bulk** — a field perturbation at the boundary
  of a parallel-plate gap decays as exp(−πs/d) (~3·10⁻⁵ at s = 100 mm), so a
  kernel-free rebinning by distance-to-edge is decisive: interior s > 100 mm
  reads det3 28.02 ± 0.11 vs det2 31.26 ± 0.16 mm, the full deficit at ~15σ.
* **Reproduces**: the two det3 runs agree to 0.2–0.3 mm at fixed calibration;
  drift field 900 → 1000 V changes the drift *time* by 8 % and the *column* by
  0.3 mm; the chamber contrast is +2.8 to +3.4 mm under **every** bundle tried.
* **But the absolute column carries a ±1 mm calibration systematic** (§5.1).

### 3.5 New, 2026-07-31: the map *shape* reproduces, and the fleet's first maps

Two things that were open on 07-30 are now measured (`bench/gap_compare.py`,
figures `gap_consistency_{maps,repeat}.png`, `gap_vs_charge.png`).

**(a) The det3 topography reproduces between independent runs.** Comparing the
6-27 and 6-28 maps cell by cell (same mount, one day apart, each with its own
calibration):

| pair | rms difference | split-half noise floor | correlation | correlation under flip-x / flip-y |
|---|---|---|---|---|
| det3 6-27 vs det3 6-28 | 0.64 mm | 0.42 mm | **0.90** | 0.30 / 0.30 |
| det2 6-22 longer vs long | 0.83 mm | 0.53 mm | 0.62 | 0.24 / 0.35 |
| det3 6-27 vs det3 6-22 (bottom slot) | 1.13 mm | 0.39 mm | −0.09 | — |

A correlation of 0.90 between independent runs, against 0.30 for the same map
mirrored, says the (x, y) structure is a property of the chamber and not of the
estimator. det2's weaker 0.62 is what a genuinely flat chamber should give —
there is little structure to correlate. The 6-22 bottom-slot det3 run correlates
with nothing (−0.09), which is the third independent sign that that dataset is
broken rather than merely different.

**(b) The fleet's first maps, and a serious amplitude caveat.** All five
chambers now have a column measured with their own RC-ladder bundle
(128 condor fits, merged 07-31):

| chamber | column X [mm] | v_geom | split-half map rms | fit quality |
|---|---|---|---|---|
| det2 | **30.62 ± 0.11** | 39.0 | 0.84 | clean, flat = the full mechanical gap |
| det3 | **27.89 ± 0.08** | 36.8 | 0.64 | clean, dished |
| det6 | **27.85 ± 0.13** | 27.3 | 0.83 | clean *after* the deep-basis fix (§10.4) |
| det7 | 27.51 ± 0.15 | 38.3 | 1.35 | soft edge (σ_e 177 ns), noisy map |
| det4 | 25.55 ± 0.39 | 32.2 | 2.15 | **dominated by the amplitude systematic — see below** |

(All at the K = 22 charge basis of §10.4, each with its own RC-ladder bundle;
add the ±1 mm calibration systematic S2 to every absolute number.)

The null tests split the same events in half by charge and by \|tan\|, which
bounds the systematics that do not cancel:

| chamber | column, low vs high charge | Δ | column, low vs high \|tan\| | Δ |
|---|---|---|---|---|
| det6 | 27.09 / 27.10 | **−0.01** | 25.50 / 27.84 | −2.34 |
| det3 | 27.48 / 28.15 | −0.67 | 27.55 / 28.13 | −0.59 |
| det2 | 29.88 / 31.16 | −1.28 | 30.64 / 30.51 | +0.13 |
| det7 | 26.76 / 29.09 | −2.33 | 27.15 / 28.33 | −1.19 |
| det4 | 20.87 / 30.17 | **−9.30** | 26.70 / 24.94 | +1.76 |

**det4's column is not a measurement of anything geometric**: its two charge
halves read 20.9 and 30.2 mm, i.e. the number you get is the number your gain
gives you. (det7's charge *quartiles* run 26.7 → 33.5 mm, the same disease
milder.) det4 is the gain-limited chamber, so this is the S3 amplitude
systematic in its extreme form, and det4's 25.5 mm must not be quoted. det7's
−2.3 mm keeps it provisional too. **det2, det3 and det6 have fleet-grade
columns** (det6 only after §10.4); det7 is marginal and det4 is not usable.

**What this does to the headline claim.** It survives, but its scope narrows —
and in an interesting direction. At the common K = 22 basis, **three chambers
agree closely (det3 27.89, det6 27.85, det7 27.51) and det2 is the one that
reads the full mechanical 30.62 mm.** So the fleet does not say "det3 is
uniquely dished"; it says det2 is the chamber that reads full, and three others
read 2.1–2.5 mm short of the mechanical gap. Two readings survive that:
(i) several chambers really are assembled a couple of mm short, or (ii) the
estimator loses the last ~2 mm of a faint column and det2's higher gain hides
it. The det3-specific part of the claim — the **spatially resolved, run-to-run
reproducible (corr 0.90) dish** — is unaffected either way, because a
topography cannot be produced by a global scale error. **The global "27.9 mm"
should be quoted as a chamber-to-chamber comparison, not as an absolute cathode
distance.**

---

## 4. Fleet state — which numbers on disk are trustworthy

Generated by `mx_june_wft/fleet_state.py` (which also flags any analysis output
that predates the table it describes). `events.meta.json` in each chamber's
`wft/` is the authority: it records the bundle, the reconstruction configuration
and the angle constants that produced the table.

| chamber | reco generation | kernel | v [µm/ns] | within 5 mm (hits) | core σ [mm] | σ_θ X / Y [°] | bias X / Y [°] | column X [mm] |
|---|---|---|---|---|---|---|---|---|
| det3 (`sat_det3`) | 2026-07-30 01:00 | RC-ladder (share_lp) | 36.60 | 93.54 (93.13) | 0.460 | 1.08 / 1.11 | -0.03 / -0.01 | 27.89 |
| det2 (`o22_long_det2`) | 2026-07-31 14:24 | RC-ladder (share_lp) | 39.94 | 91.91 (92.06) | 0.436 | 1.14 / 1.63 | -0.07 / -0.05 | 30.62 |
| det4 (`g_det4`) | 2026-07-31 13:17 | RC-ladder (share_lp) | 34.16 | 41.89 (40.67) | 0.667 | 2.36 / 2.86 | -0.01 / -0.05 | 25.55 |
| det6 (`g_det6_long`) | 2026-07-29 02:00 | legacy | 26.74 | 75.41 (71.19) | 0.429 | 2.28 / 2.52 | +0.03 / -0.95 | 27.85 |
| det7 (`g_det7_long`) | 2026-07-31 12:38 | RC-ladder (share_lp) | 36.60 | 56.89 (52.73) | 0.635 | 1.98 / 2.09 | -0.02 / +0.31 | 27.51 |

(`within 5 mm` shows the waveform chain with the hits chain in brackets, both
through the same `02_efficiency.py` accounting. Columns are the K = 22 basis of
§10.4 and each carries the ±1 mm calibration systematic S2.)

**What changed on 2026-07-31.** Four of the five chambers were taken through the
approved RC-ladder rollout (`rollout_lp.sh`: reco → w0/kw production retrofit →
re-reco → analysis chain). Against the 07-29 generation:

| chamber | σ_θ X / Y before → after | angle bias before → after | verdict |
|---|---|---|---|
| det7 | **7.05 / 4.47 → 1.98 / 2.09°** | +0.07 / −0.69 → −0.02 / +0.31 | **fixed** — the stored chain had been produced by a calibration the campaign had already rejected |
| det2 | 1.31 / 1.56 → **1.14** / 1.63° | **−0.38 / −0.38 → −0.07 / −0.05** | improved; the bias is gone and core σ 0.483 → 0.436 mm |
| det4 | 2.73 / 2.58 → **2.36** / 2.86° | −0.21 / −0.30 → **−0.01 / −0.05** | mixed: X and bias better, Y worse |
| det6 | 2.28 / 2.52 → 2.62 / 3.43° | +0.03 / −0.95 → −0.10 / +0.48 | **regressed — rolled back** (§10.6) |
| det3 | already on it | — | unchanged |

**det6 is deliberately still on the legacy kernel.** Its RC-ladder calibration is
degenerate (`sigma_p0` = 0.039 mm and `Dp` = 0.0016 both sitting on their lower
guards, against 0.41/0.013 on every other chamber) and it costs 0.3–0.9° of
angle resolution. The lp generation is parked in
`<det6>/wft/lp_attempt_20260731/` and the pre-rollout generation restored, so
det6's numbers above are the legacy-kernel ones — **the only chamber whose
angles come from a different kernel than the rest, which matters for any
fleet-wide statement.** Recalibrating it (seeded from det3's kernel, as det7's
`v36` bundle was) is the top fleet follow-up.

**Detection numbers are chain-independent** and unchanged throughout: `has_any`
100 % on det2/3/6/7 (95.8 % det4), `spark_frac` 8.2 % (det3), 9.7 (det2),
9.8 (det4), 22.3 (det6), 37.4 % (det7).

---

---

## 5. Systematics register

Every known systematic, its size, and its status. An auditor should be able to
find each one's evidence from this table alone.

### 5.1 Reconstruction and calibration

| # | effect | size | what it touches | status | evidence |
|---|---|---|---|---|---|
| S1 | **Aggregate hit-time compression** — the defect that motivated the rebuild | ladder compressed 20–30 %, angles ~4° too steep, depth-dependent walk-off | everything geometric on the hits basis | **resolved** by the waveform basis; hits chain must not be extended | `RECONSTRUCTION_BASIS.md`; `WAVEFORM_FIRST_THREADING.md` §3 (estimator independence) |
| S2 | **Kernel ↔ calibration degeneracy** — several legitimate ref-pinned bundles of the same chamber fit the same data | absolute drift column spans **1.8–2.3 mm** (rms ~0.9) across five bundles; positions and angles barely move | absolute column / depth scale | **open, must be quoted**: det3 = 27.9 ± 0.1 (stat) ± 1.0 (calib). Differentials taken at a common bundle are immune | `GAP_CONSISTENCY_2026-07-30.md` §1–2 |
| S3 | **Endpoint depends on signal amplitude** — weak signals fade below the fit's sensitivity earlier | +1.5 mm across det3's charge range, +2.2 mm on det2 | gap maps, chamber-to-chamber comparison at unequal gain | known; quote maps at matched charge. **Not yet applied to the det4/6/7 first maps** | `GAP_CONSISTENCY_2026-07-30.md` §2 |
| S4 | **Y-response tilt** — both chambers' Y planes show a ~0.8–0.9 µs apparent attachment tilt that X does not | shifts the Y endpoint; gas would tilt X identically, so it is a response artefact | Y-side column and profile physics | **open**; always quote the **X** plane for column physics. Harmless for angles/positions (kw ≈ 1) | `GAP_STUDY_2026-07-30.md` |
| S5 | **Calibration transfer** — a bundle used on data it was not fitted on | σ_θ **+21–22 %** same detector five days apart (upper bound: that bundle is itself suspect, kw 0.37/0.40), **+43–53 %** across detectors. **Position transfers perfectly** (within-5 mm and core σ unchanged) | any use of a foreign bundle — the central constraint on bench→beam transfer | measured | `WINDOW_ABLATION_2026-07-30.md` §4 |
| S6 | **v_drift scale** — v never enters the fit, only `tan = (w·10³ − w0)/(kw·v)` | position **bit-identical** at v ±10 %; σ_θ degrades 12–57 % | angles only; correctable after the fact without re-reconstructing | measured | `WINDOW_ABLATION_2026-07-30.md` §4a |
| S7 | **v is not identifiable from the fit alone** — the sharing kernel and v trade off along a valley while `tan θ = w/v` stays determined | det7's free fit: c1 = 0.004 (impossible on resistive strips) with v = 36.7 where the drift scan gives 26.4, yet unbiased angles | any fitted-v number | **procedure**: always pin v to an independent measurement; treat a fitted v as a nuisance parameter | `FLEET_2026-07-29.md`; `WAVEFORM_FIRST_THREADING.md` §17.2 |
| S8 | **χ²(v) valley in the gap fit** — fixed-hyper timing χ²(v) scans minimise at ~40 µm/ns under **both** kernels | would inflate every column by ~10 % if used | drift-gap conversion | **procedure**: convert with the *geometric* v_geom = kw·v (anchored by the unbiased M3 angle validation), never with timing χ²(v) | `GAP_STUDY_2026-07-30.md`; `bench/vscan_lp.py` |
| S9 | **w0/kw corridor bias** — constants measured on the 150-event calibration corridor differ from production-run values | ~0.06 µm/ns on w0, ~1.5 % on kw → a real residual angle bias | angles | **procedure**: the production `set_w0.py` retrofit + re-reco is mandatory (pass 2) | `RECO_BENCH_2026-07-29.md` |
| S10 | **Metric selection bias** — `slope_reliable` selects on the *fitted* \|tan\| ≥ 0.08 | up to ±0.5–0.8° in the central angle bins; it manufactured a fake "Y regression" for the RC-ladder kernel | reported σ_θ and bias | **known trap**: quote the reference-selected per-\|tan_ref\|-bin numbers (`comp14`/`s14`) as headline | `RECO_BENCH_2026-07-29.md` §5, and the 07-30 Y-knob scan |
| S11 | **Reference-free seeding** — production cannot seed at the M3 position without making alignment and efficiency circular | core σ +0.02 mm (4 % X, 17 % Y in a controlled same-window test); **zero** on the angle | position resolution | accepted, measured price. Part of the R&D's quoted mesh resolution was optimism | `DET3_GATE_2026-07-29.md` |
| S12 | **Giant-charge tail** — delta rays / small showers the model cannot describe; the χ² fit slides off | the residual reco_far population (1.2–1.7 % of rays), median fitted charge ~160× MIP | position tail only | mf3 weighting recovered ~40 %; the rest is a model limitation, and the hits chain's crude centroid is more robust there (+1.1 pt on common events) | `RECO_BENCH_2026-07-29.md` §2 |
| S13 | **Multi-cluster seeding** — "largest cluster wins" picked the wrong charge in ~5 % of events | was −1.0 pt of within-5 mm; reference sat a median 37 mm outside the fit window on those events | position | **resolved** by the candidate-cluster seeder (of 155 remaining far events, *zero* have more than one candidate) | `DET3_GATE_2026-07-29.md`, `RECO_BENCH_2026-07-29.md` §1 |

### 5.2 Reference and inputs

| # | effect | size | status |
|---|---|---|---|
| S14 | **M3 pointing floor** — the reference's own resolution at the DUT plane | 0.40 mm core; per-plane σ_X ≈ 0.41, σ_Y ≈ 0.45–0.51 mm | measured and closed (`m3-self-resolution` study). Position resolution quotes are reference-limited: per-axis 0.63/0.61 mm = 0.40 (M3) ⊕ ~0.45–0.49 (detector) |
| S15 | **M3-vs-detector attribution of the common Y w0** — four chambers share w0_Y ≈ −0.2 µm/ns, and all four strip maps are ~90° rotated against M3 | a single **+0.3° M3 tan_X systematic** would produce the common part; detector-specific parts (det6-Y, det2-X) sit on top | **open, not separable with this data**. Operationally harmless (every angle is scored against M3) but it means the *physical* chamber tilts are not established |
| S16 | **Hits-cache significance floor** (07-25 fix) | worth ~10 pt on det2, ~27 on det6, ~35 on det7 of within-5 mm | **closed** — all five caches rebuilt; the earlier "+10 to +31 point" waveform-vs-hits gaps are withdrawn (they compared against pre-floor caches) |
| S17 | **Mixed 32/37-sample readout windows** | crashed det4's calibration outright (shape mismatch) | **fixed** 07-30 in `wft/calibrate.py::measure_dt_xy`; regression test passes |
| S18 | **`gap_merge.py` shard glob was too loose** — `profiles_<label>_*` also matched the cross-bundle labels `profiles_<label>__with__<other>_*` | it would merge fits made with *different calibrations* into one result, i.e. fold the 1.7 mm bundle systematic into the answer | **found and fixed 07-31**. The published numbers are unaffected: the own-bundle shards sort first and `drop_duplicates(['eid','plane'])` discarded the foreign rows, verified by re-merging det6 (26.97 mm on 8 shards vs 27.04 on the accidental 32 — the 0.07 mm is the endpoint-bound change of §10.4, not the glob) |

---

## 6. Open items, ranked

**Physics / analysis**

0. **Recalibrate det6 under the RC-ladder kernel, seeded from det3's** — it is
   the one chamber still on the legacy kernel because its own lp fit railed
   (§10.6). Until it is done, no fleet-wide statement rests on a single kernel.
1. **A bottom-slot det3 run at ≥ 900 V drift** — the one missing test that would
   turn "det3's cathode is dished" from a same-mount result into a remount-
   confirmed one (§8.1C). Needs beam time on the bench, not analysis.
2. **Re-measure the fleet columns at matched charge** — det4's column moves
   9.3 mm across its own charge range and det7's 2.3 mm; until the maps are
   quoted in a common amplitude band, three of five chambers have no usable
   column (§3.5).
3. ~~det6's endpoint fit is railed~~ — **closed 07-31** (§10.4): the charge
   basis was shallower than det6's column. Re-fitted at K = 22 the width comes
   off its bound and det6 reads 27.85 ± 0.13 mm.
4. **v(E) / gas fit under the RC-ladder kernel** — the 300–1100 V drift scan is
   still analysed on the old kernel and with the timing-χ²(v) method that S8
   shows to be degenerate. Redo with the geometric v.
5. **Y-response tilt** (S4) — the last Y model imperfection; caps Y-side column
   physics, harmless for angles.
6. **Time resolution (paper topic 10)** — the last analysis still on a mixed
   basis; the RC-ladder t0 should be cleaner.
7. **M3-vs-detector attribution of the common Y w0** (S15) — needs a second
   reference or cross-detector coincidence data.

**Bookkeeping that an audit will trip over**

8. `GAP_STUDY_2026-07-30.md` still quotes absolute columns without the ±1 mm
   calibration systematic and still implies the remount confirmation; the four
   corrections are listed at the end of `GAP_CONSISTENCY_2026-07-30.md`.
9. `03_angles.py`'s headline σ is the selection-biased `slope_reliable` one
   (S10); it should report the reference-selected per-bin numbers.
10. The whole 07-29/30/31 arc is **uncommitted** — the documents, `bench/`,
    `condor/`, `rollout_lp.sh`, `fleet_state.py`, `state/` and the `wft/`
    changes (see §9).
11. `FLEET_2026-07-29.md` and `DET3_GATE_2026-07-29.md` predate the rollout;
    their per-chamber numbers are superseded by §4. They are kept because their
    *method* sections (how both chains were put through one accounting) are the
    reference for that procedure.

---

## 7. Where everything lives

| what | path |
|---|---|
| reconstruction library | `wft/` (model, calibrate, reco, seed, cli) |
| bench analysis chain | `mx_june_wft/01_alignment.py` … `04_maps.py`, `digest.py` |
| benchmark + study tools | `mx_june_wft/bench/` |
| grid packaging | `mx_june_wft/condor/` — `make_package.py` (gap fits, calibrations), `make_bench_package.py` (benchmark scans, residual audit) |
| state snapshot for review | `mx_june_wft/state/` — the source json of every number quoted here, exported by `fleet_state.py --export` |
| per-detector outputs | `<Analysis>/<run>/<subrun>/<det>/wft/` |
| bench waveforms | `/media/dylan/data/x17/cosmic_bench/<det>/<run>/<subrun>/decoded_root` |
| grid staging | `/home/dylan/x17/cosmic_bench/condor_wft`, `…/condor_bench` |
| run registry | `mx_june_cosmic_qa/qa_config.py` (run keys used throughout) |

Reproduce a chamber end to end:

```bash
mx_june_wft/rollout_lp.sh <run_key> --jobs 8      # reco -> w0 retrofit -> re-reco -> chain
.venv/bin/python mx_june_wft/digest.py <run_key>
.venv/bin/python mx_june_wft/fleet_state.py       # what is on disk, per chamber

# the drift column (grid path: 8 shards, ~5 min wall clock)
.venv/bin/python mx_june_wft/condor/make_package.py --k-bins 22 --datasets <keys>
rsync -a /home/dylan/x17/cosmic_bench/condor_wft/ lxplus:~/wft_gap/
ssh lxplus 'cd ~/wft_gap && condor_submit gap_fit.sub'
.venv/bin/python mx_june_wft/bench/gap_merge.py --dir <shards> --label <key>_k22 --bundle <b> --out <d>

# the benchmark scan and the residual audit
.venv/bin/python mx_june_wft/condor/make_bench_package.py --key <key> --residual 8
ssh lxplus 'cd ~/wft_bench && condor_submit bench_scan.sub && condor_submit residual.sub'
.venv/bin/python mx_june_wft/bench/summarize_scans.py --dir <out> --ref base
.venv/bin/python mx_june_wft/bench/residual_merge.py --dir <out> --png residual_audit.png
```

---

## 8. For the auditor

### 8.1 What to attack first

These are the load-bearing claims, ordered by how much rests on them and how
much room there is to be wrong. For each: the claim, what would falsify it, and
what has already been done to try.

**A. "Hit times cannot carry geometry."** This justifies discarding an entire
analysis chain. It rests on (i) the measured neighbour-copy amplitude
(~29 %/40 % at ±1 strip, directly measured on near-vertical tracks), (ii) the
depth-resolved displacement table in §3.2, (iii) estimator independence across
rising edge, 20 % leading edge, CFD and matched filter. *Falsify it by* showing
an estimator that reads the ladder without compression, or by showing the §3.2
displacement is an artefact of the deconvolution used to draw the cluster (it is
line-free and never sees the fitted track — check that in
`37_threading_displays.py`).

**B. "Angles are 2× better and unbiased."** Everything is scored against M3, so
this is a statement about *agreement with M3*, not about truth. The internal
cross-check that does not depend on M3's absolute scale is the **implied-v
flatness across angle bins** (a geometrically honest reconstruction must give
the same w/tan in every bin): ~17 µm/ns of spread on the hits ladder → 1–3 on
the waveform fit. *Falsify it by* finding an M3 systematic that mimics both
(S15 shows one candidate, +0.3° in tan_X, that survives), or by showing the
reference-selected metric (`comp14`, unbiased by construction) disagrees with
the headline σ_θ.

**C. "The det3 drift column is short because the cathode is dished."** The
control chamber reads the full mechanical gap with the same estimator, gas
attachment is excluded by flat X profiles, and fringe fields are excluded by an
exp(−πs/d) argument plus a kernel-free interior measurement. *The weak point is
that it has never been confirmed against a remount*: det3's only other runs are
either at too low a drift field for the column to fit in the readout window, or
(6-22, bottom slot) taken with the chamber in a pathological drift state
(v ≈ 8.9 µm/ns by an independent hits-based survey, against det2's 25.1 in the
same run). A bottom-slot det3 run at ≥ 900 V is the missing test, and until it
exists the result should be quoted as "same-mount repeat + field scan + control
chamber", not "confirmed geometry".

**D. "The RC-ladder kernel is the right sharing model."** Its evidence is a
direct shape measurement (14×/54× better fit to the observed neighbour copy)
and the fact that the empirical slope patch `kw` goes to 1.0 by itself once the
shape is right — a genuine out-of-sample prediction. *The uncomfortable part*:
under this kernel the hyper fit drives the discrete sharing amplitude `c1` down
to its 0.05 floor and puts the charge spread into `sigma_p0` ≈ 0.4 mm instead.
Two independent seeds land on the same optimum, so it is not a seed artefact,
but "a fitted parameter sitting on its bound" is exactly where an auditor should
push. §10.1 records a dedicated scan of it, and the answer is that the floor is
harmless: raising `c1` monotonically destroys the angles, halving it does
nothing that survives validation.

**E. "Position is at parity."** This is the honest, unglamorous claim and it is
the one most likely to be *understated* elsewhere in the repo's history — an
earlier version of the fleet table claimed +10 to +31 points and was withdrawn
when the hits caches were rebuilt with the significance floor (S16). Any
position comparison must use `02_efficiency.py --source hits` (one accounting,
both chains) and a floor-corrected cache. Check `.meta.json` sidecars exist.

### 8.2 Known soft spots, stated plainly

* **Absolute depth/column numbers carry ±1 mm of calibration systematic** (S2).
  `GAP_STUDY_2026-07-30.md` did not carry it; it now opens with an amendment box
  stating that, the missing remount test, the amplitude systematic and the
  K = 22 basis change. Differential statements are unaffected.
* **det4's gap column is not usable** (S3): its two charge halves read 20.9 and
  30.2 mm. det7's (−2.3 mm across charge) is marginal. Only det2, det3 and det6
  have fleet-grade columns.
* **At a common basis three chambers agree near 27.5–27.9 mm and det2 is the one
  reading the full 30.6** (§3.5, §10.4). The det3 *topography* is solid; the
  claim that det3 is uniquely short is not.
* **det7 is a chamber-level anomaly, not a reconstruction result**: its data
  insist on v ≈ 36.6 µm/ns at a field where its neighbour in the same run
  calibrates to 26.7, and it sparks at 37 %. Do not quote det7 absolute v or
  depth. Its angles are usable only under the v = 36.6 bundle.
* **Y-side column physics is capped by S4** and X should be quoted everywhere.
* **The physical chamber tilts are not separated from a possible M3 tan_X
  systematic** (S15).
* **The `wft` fit is a local optimizer**: a Nelder-Mead restart test (`nmx`)
  found it converged, so the risk is model error, not optimisation. χ²/dof is
  ~20–60 under the production weighting — the model is *not* a statistically
  adequate description of the data, and every uncertainty derived from the χ²
  curvature would be optimistic. The angle errors quoted here are empirical
  (spread against M3), not curvature-derived, which is why they are trustworthy.

### 8.3 Numbers an auditor can regenerate cheaply

| check | command | expected |
|---|---|---|
| what is on disk right now | `.venv/bin/python mx_june_wft/fleet_state.py` | per chamber: reco date, kernel, bundle, w0/kw, position, angles, column |
| model/production regression | `.venv/bin/python -m pytest wft/tests -q` | 4 passed; production path bit-identical with all new flags off |
| det3 headline, both chains | `mx_june_wft/02_efficiency.py sat_det3 --max-dropped -1` and `--source hits` | 93.5 % vs 93.13 % within 5 mm |
| old-chain accounting validation | as above, `--source hits` | reproduces the published 93.4 % / 0.48 mm at 93.13 % / 0.448 mm |
| angle metric without selection bias | `bench/run_bench.py sat_det3 --variant prod --bundle <lp2>` → `comp14_*`, `sig14_*` | X clean, Y ≈ +0.03° under the RC-ladder kernel |
| gap contrast at a common bundle | `bench/gap_matrix.py` | det2 − det3 = +2.8 to +3.4 mm under every bundle |
| the fleet columns | `bench/gap_merge.py --label <key>_k22` | det2 30.62, det3 27.89, det6 27.85, det7 27.51, det4 25.55 mm (X plane) |
| the model's residual structure | `bench/residual_merge.py --dir <out>` | mean \|residual\| ≈ 1 % of peak, coherent: peak under-modelled, tail over-modelled |
| which bundle produced a table | `cat <det>/wft/events.meta.json` | `hyper.share_lp = 1` for the RC-ladder generation |

---

## 9. Provenance and repository state

**Code version.** The reconstruction library `wft/` is at repo commit `1292d91`
plus uncommitted working changes (`calib.py`, `calibrate.py`, `model.py`,
`reco.py`; ~360 lines) that carry the RC-ladder kernel, the model-error
weighting, the per-plane angle constants, the coarse pre-scan and the
mixed-window fix. `wft/tests/test_model_regression.py` passes, which is the
guarantee that the production path is bit-identical when the new switches are
off.

**Uncommitted at the time of writing** — an auditor reading only `git log` will
not see this arc:

```
 M wft/{calib,calibrate,model,reco}.py
 M mx_june_wft/bench/{run_bench,summarize_scans}.py
?? mx_june_wft/{RECO_BENCH,GAP_STUDY,GAP_CONSISTENCY,HANDOFF,WINDOW_ABLATION,ANALYSIS_STATE}_*.md
?? mx_june_wft/{bench,condor}/          the benchmark, gap, framing and grid tooling
?? mx_june_wft/{rollout_lp.sh,fleet_state.py}
```

**Data products.** Each chamber's `wft/` directory keeps its superseded
generations (`prev_<date>_<what>/`), so every number in the older documents can
still be reproduced from the tree it was written against.

**Grid runs behind the current numbers.**

| what | cluster | jobs | outcome |
|---|---|---|---|
| gap fits, det2/det3 datasets × 5 bundles | (07-30) | 200 | merged, `GAP_CONSISTENCY` §1 |
| RC-ladder calibrations, det4/6/7 | (07-30) | 3 | bundles installed as `calib_bundle_lp` |
| gap fits, det4/6/7 × own and det3 bundles | 11907585 | 128 | completed 07-30, **merged 07-31** (§3.5) |
| det3 angle-optimality scan (split 0:2) | 13324193 | 48 | §10.1 |
| det3 residual / goodness-of-fit audit | 13324196 | 8 | §10.3 (re-run as 13324294 after the per-cell normalisation fix) |
| det3 scan validation (disjoint split 1:2) | 13324197 | 11 | §10.1 |
| fleet gap fits at the K = 22 charge basis | 13324198 | 40 | §10.4 |
| fleet residual audit | 13324285–8, re-run 13324294–8 | 72 | §10.5 |
| det6 calibration scan | 13324299 | 12 | §10.6 |

---

## 10. New diagnostics run for this review (2026-07-31)

Three things were run on lxplus condor to close questions an auditor would
otherwise have to raise. Tooling: `condor/make_bench_package.py` +
`bench_scan.sub` / `residual.sub`, all new today.

### 10.1 Is the production bundle at the angle optimum? (48 jobs, cluster 13324193)

`WINDOW_ABLATION` §3 noticed that several ±25 % perturbations of the det3
bundle *improve* σ_θ, and suggested a targeted refit could buy ~10 % on Y. This
scan settles it: 48 configurations — 1-D arms on every model constant, 2-D
corners on the two known degeneracies, and the untested readout-window corner —
each scored on the **same** 1,963 events (split 0:2 of `sat_det3`), production
configuration, RC-ladder bundle. Headline metric is `s14Y`, the
reference-selected angle spread at \|tan_ref\| ≥ 0.14, which carries no
selection bias (S10) and, being a MAD about the median, no bias sensitivity.

| configuration | σ_θ Y | s14 Y | vs base | compression cmp14 Y | core σ [mm] |
|---|---|---|---|---|---|
| **base** (`calib_bundle_lp2`) | 1.129° | 0.982° | — | −0.173° | 0.483 |
| `sigma_p0` × 1.25 | 1.066 | 0.905 | **−7.9 %** | −0.290 | 0.479 |
| `sigma_p0` × 1.4 | 1.064 | 0.884 | **−10.0 %** | −0.323 | 0.500 |
| `sigma_p0` × 1.6 | 1.077 | 0.871 | **−11.3 %** | −0.418 | 0.508 |
| `sigma_p0` × 1.25, `Dp` × 0.5 | 1.062 | 0.940 | −4.3 % | −0.239 | 0.482 |
| `tau_s` × 0.85, `kTauY` × 0.85 | 1.054 | 0.952 | −3.1 % | −0.354 | 0.478 |
| `c1` × 0.5 | 1.094 | 0.952 | −3.1 % | −0.224 | 0.484 |
| `c1` × 4 | 1.378 | 0.959 | −2.4 % | +0.004 | 0.485 |
| `Dp` × 3 | 1.366 | 1.002 | +2.0 % | −0.238 | 0.593 |

**Validated on the disjoint half** (split 1:2, 1,954 events never used in the
scan — because picking the best of 48 configurations on one sample and quoting
its number is exactly how a fluctuation becomes a result):

| configuration | s14 Y, scan half | s14 Y, validation half | Δ vs base (validated) | Δ s14 X | cmp14 Y | core σ |
|---|---|---|---|---|---|---|
| base | 0.982 | 0.942 | — | — | −0.187 | 0.475 |
| **`sigma_p0` × 1.25, `Dp` × 0.5** | 0.940 | 0.896 | **−4.9 %** | **−5.0 %** | −0.195 | 0.480 |
| `sigma_p0` × 1.4 | 0.884 | 0.898 | −4.7 % | −10.1 % | −0.259 | 0.490 |
| `sigma_p0` × 1.25 | 0.905 | 0.900 | −4.4 % | −5.0 % | −0.236 | 0.482 |
| `kTauY` × 0.85 | 0.958 | 0.931 | −1.2 % | +0.0 % | −0.231 | 0.471 |
| `c1` × 0.5 | 0.952 | 0.959 | +1.8 % | +0.5 % | −0.211 | 0.482 |

**Four conclusions.**

1. **The χ²-optimal bundle is not the resolution-optimal one, but the gap is
   ~5 %, not the ~10 % the single-split scan suggested** — the difference is
   the winner's curse, and it is why the validation pass exists. The
   best-balanced point (`sigma_p0` × 1.25 with `Dp` × 0.5) buys **~5 % on both
   planes with no bias penalty** (cmp14 Y −0.195 against the base's −0.187) and
   0.005 mm of core σ. Worth a refit; not a headline.
1b. The timescale knobs (`tau_s`, `kTauY`) do **not** survive validation
   (−1 % against a ±2.5 % statistical error): leave them alone.
2. **The gain is bought against slope compression.** Every configuration that
   improves the spread also drives `cmp14_Y` more negative (−0.17 → −0.32° at
   × 1.4): a model with more assumed transverse smearing pulls fitted slopes
   inward. That bias is exactly what the per-plane `kw` constant absorbs, and
   `s14` is bias-insensitive, so the resolution gain is genuine — but it means
   any refit must re-measure `kw` afterwards, and it raises a physics question
   rather than settling one: **why does the fit want ~40 % more transverse
   spread than the χ² fit gives it?** Candidates: an unmodelled depth
   dependence of the cloud size, or an M3 contribution absorbed into the
   detector term. Not something to tune away silently.
3. **`c1` sitting on its 0.05 floor is not an optimizer artefact.** Raising it
   monotonically destroys the angles (× 2 → σ_θ Y 1.146, × 4 → 1.378 with the
   implied-v spread doubling); halving it changes nothing that survives
   validation (+1.8 %). Under the RC-ladder copy shape the data genuinely want
   very little discrete sharing, and the floor is not binding on anything that
   matters — audit point 8.1D is answered, in favour of the kernel.

### 10.2 The readout-window corner — a concrete DAQ recommendation

`WINDOW_ABLATION` §2d left one corner unmeasured: raising the DREAM latency
*and* the sample count together. It is now measured (same events, same bundle;
`crop start:n` emulates the beam frame, start 6 / n 20 = run_79 as it ran):

| framing | σ_θ Y | s14 Y vs full window | compression cmp14 Y |
|---|---|---|---|
| full 32 samples (bench) | 1.129° | — | −0.173° |
| **start 3, n = 26** | 1.153 | **+0.3 %** | −0.187 |
| **start 4, n = 26** | 1.128 | **+1.1 %** | −0.214 |
| start 4, n = 24 | 1.173 | +1.4 % | −0.205 |
| start 4, n = 20 | 1.221 | +2.4 % | −0.295 |
| start 5, n = 20 | 1.246 | +7.4 % | −0.305 |
| **start 6, n = 20 (run_79 as it ran)** | 1.333 | **+12.4 %** | **−0.492** |

**Moving the DREAM latency by ~2–3 units and the window to 26 samples recovers
essentially the full-window angle performance** — s14Y within ~1 % of the
32-sample bench, and the compression bias more than halved. That is the answer
to the open question, and it costs two DAQ settings, not hardware.

### 10.3 Goodness of fit — where the model actually fails (8 jobs, cluster 13324196)

`bench/residual_audit.py` + `bench/residual_merge.py` (both new) stack
(data − model) in the fit's own frame — rows = strips relative to the fitted
mesh position, columns = samples — over **7,058 X and 7,084 Y plane fits** of
`sat_det3` under the production configuration. Figure:
`residual_audit.png`; numbers: `state/residual_audit.json`.

| plane | χ²/dof p5 / p50 / p95 | mean \|residual\| / peak | worst cell | mean \|pull\| | worst \|pull\| |
|---|---|---|---|---|---|
| X | 7.3 / 22.6 / 116 | 1.27 % | 15.0 % | 0.97 | 5.7 |
| Y | 15.9 / 49.6 / 181 | 1.19 % | 7.7 % | 1.20 | 10.3 |

*(Normalised per cell, not per event: a strip 4 away from the track appears in
only ~25 % of the events a central strip does — 1,600 against 7,000 — so a
per-event mean would dilute exactly the outer region the diagnostic is asking
about. The first pass did that and under-reported the mismatch by ~30 %.)*

**The mismatch is coherent, not noise, and it has a specific shape.** Averaged
over thousands of events the residual does not average away — which is the
direct statement that χ²/dof ≈ 20–60 is model error:

* **on the track strip the model's peak is too low and its tail is too heavy**:
  a strong positive residual at the rise (samples ~5–13, +6–7 % of peak) followed
  by a systematically negative pull along the whole late tail (pull −3 X, −6 Y);
* **the neighbours' late signal is under-modelled**, most clearly on Y: positive
  lobes at ±2–4 strips around samples 15–22 (+4–5 % of peak). The RC-ladder
  copies are not carrying enough amplitude that far out, that late.

**This connects to §10.1.** The scan independently found that the fit wants
~25 % more transverse spread (`sigma_p0`) than the χ² calibration gives it. Both
diagnostics point at the same deficiency: *not enough charge at ±2–4 strips.*
A larger cloud size is the crude fix the scan found; the residual image says the
physical fix is in the lateral spread of the late (deep-drift) charge — i.e. the
sharing model's range, not the cloud's initial size. That is the concrete next
model improvement, and it is now specified rather than guessed.

Per-chamber results and what they imply: §10.5.

**For the auditor**: this is the honest answer to "your fit is not statistically
adequate — how much does that cost you?" The residual is ~1 % of peak amplitude
on average, its structure is understood, and the empirical (M3-scored) angle and
position resolutions quoted throughout this document already contain whatever it
costs. What it forbids is quoting **curvature-derived** per-event uncertainties
as if they were calibrated — they are not, and nothing here does.

### 10.4 The charge basis was too shallow for the slow chambers (40 jobs, cluster 13324198)

det6's endpoint fit was **railed**: the erfc width sat exactly on its 400 ns
bound. The cause is structural — the charge basis is K = 18 slices of 60 ns
= 1080 ns, and at det6's v ≈ 27 µm/ns a 30 mm column takes 1110 ns, so the end
of the column falls off the end of the model. `gap_fit.py` gained `--k-bins`,
`gap_merge.py` now reads the basis depth from the data instead of assuming 18
bins (and scales the fit window and bounds with it), and the whole fleet was
re-fitted at **K = 22 (1320 ns)**:

| chamber | column K = 18 | column K = 22 | σ_e (edge width) 18 → 22 | fit χ² 18 → 22 |
|---|---|---|---|---|
| det3 | 27.80 | 27.89 | 86 → 86 | 1346 → 1394 |
| det2 | 30.52 | 30.62 | 77 → 78 | 1206 → 1230 |
| det7 | 27.75 | 27.51 | 138 → 177 | 1266 → 1418 |
| det4 | 24.38 | 25.55 | 372 → 356 | 323 → 336 |
| **det6** | **27.04** | **27.85** | **400 (railed) → 253** | **520 → 224** |

**det6's column is now a measurement** — the width comes off its bound, the χ²
halves, the endpoint error halves (±10 → ±5 ns), and the column moves +0.8 mm.
The fast chambers move by ≤ 0.1 mm, which is the control that says the deeper
basis is not simply inflating everything. det4 moves +1.2 mm but its edge is
still 356 ns wide — that is the amplitude systematic, not the basis.

All five chambers' `gap_study/` now hold the K = 22 result; the K = 18
generation is preserved beside it as `gap_study_k18/`. **Open item 3 is closed**;
the fleet columns in §3.5 are the K = 22 ones.

### 10.5 The residual audit across the fleet (40 jobs, clusters 13324294–8)

The same goodness-of-fit stack on every chamber, 1,200 events × 8 shards each,
each with its own RC-ladder bundle. `mean |pull|` is the noise-normalised
mismatch (1.0 would be "model error comparable to noise"); `mean |residual|` is
in units of the event's peak model amplitude.

| chamber | χ²/dof median X / Y | mean \|residual\| X / Y | mean \|pull\| X / Y | σ_θ X / Y for comparison |
|---|---|---|---|---|
| det3 | 22.6 / 49.6 | 1.27 / 1.19 % | 0.97 / 1.20 | 1.08 / 1.11° |
| det2 | 34.5 / 68.0 | 2.06 / 1.12 % | 1.14 / 1.54 | 1.14 / 1.63° |
| det6 | 56.7 / 113.5 | 2.25 / 2.89 % | 1.13 / 2.08 | 2.28 / 2.52° |
| det7 | 52.7 / 94.0 | 5.44 / 3.16 % | 1.38 / 2.38 | 1.98 / 2.09° |
| det4 | 5.7 / 10.7 | 0.92 / 8.96 % | **0.47 / 0.84** | 2.36 / 2.86° |

**The model-mismatch ordering predicts the angular-resolution ordering** — det3
best, then det2, then det6/det7 at 2–3× the pull — *without using the M3
reference at all*. That makes the residual audit a per-chamber model-quality
figure of merit that can be computed at the beam, where no reference exists.

**det4 is the informative exception**: its pull is the *lowest* in the fleet
(0.47/0.84) while its angles are the worst. det4 is the gain-limited chamber, so
its fits are **noise-dominated, not model-dominated** — the model describes what
little signal there is perfectly well, and the resolution is lost to photon
statistics. Two different failure modes that a χ²/dof number alone would
conflate: det6/det7 are *mis-modelled*, det4 is *starved*.

Per-chamber figures: `residual_audit.png` (det3) and
`residual_audit_<key>.png`; numbers in `state/residual_*.json`.

### 10.6 Why det6 was rolled back (12 jobs, cluster 13324299)

det6's rollout made its angles worse (σ_θ 2.28/2.52 → 2.62/3.43°, implied-v
spread 4.8/6.4 → 8.2/10.8 µm/ns). Its RC-ladder calibration is the only one in
the fleet that railed:

| chamber | `sigma_p0` [mm] | `Dp` |
|---|---|---|
| det7 | 0.458 | 0.0135 |
| det2 | 0.423 | 0.0135 |
| det3 | 0.409 | 0.0134 |
| det4 | 0.261 | 0.0163 |
| **det6** | **0.039** | **0.0016** |

Both constants sit on their lower guards — the same failure mode that made
det7's first v-pinned bundle unusable. A dedicated scan (`sigma_p0` × 2…16,
`Dp`, `c1`, `kY` on 1,500 det6 events) confirms the diagnosis but also shows it
is **not a one-knob fix**: raising `sigma_p0` recovers the raw spread
(σ_θ Y 3.47 → 2.74°, X 2.59 → 2.26°) but the reference-selected s14 Y barely
moves (1.855 → 1.82°) and the slope compression doubles (cmp14 Y −0.75 →
−1.16°). The whole calibration has to be redone, seeded from a good kernel —
exactly the recipe that rescued det7. Until then det6 stays on the legacy
kernel, and that is recorded in §4.
