# Waveform-first threading study — det3 June cosmics, from zero

**Question:** why don't the reconstructed micro-TPC hit clusters thread the M3
reference track in depth, and can a reconstruction built directly from the raw
waveforms — with no inherited assumptions — recover tracks that match the
reference line within its uncertainty?

**Data:** `sat_det3` Saturday long run (`mx17_det3_saturday_scan_6-27-26 /
long_run_resist_490V_drift_1000V`, 490 V resistive / 1000 V drift), decoded
non-ZS waveforms (512 ch × 32 samples @ 60 ns, FEU 7 = X, FEU 8 = Y), M3
tracking v2, recipe χ²<1 & NClus=4, alignment `alignment_tpc_veto50`
(θ=89.45°, z=714/715). 6,317 matched muons (both planes, radial < 10 mm).

**Session:** 2026-07-25. Scripts in this directory (see §8); figures and
caches in `<Analysis>/mx17_3/waveform_first/`.

---

## 0. TL;DR

1. **The depth divergence is an artefact of per-strip *aggregate* timing, and
   it is estimator-independent.** Each strip's waveform is a mixture of its
   own charge and *delayed, dispersed* copies of its neighbours' charge
   (resistive sharing). Any time you extract from that aggregate — production
   rising-edge, CFD, leading-edge 20 %, full matched-filter — is pulled toward
   the cluster-mean arrival time. Measured on core strips: **+40 ns too late
   at the mesh → −130 ns too early at the cathode** (S-shaped compression;
   skirt strips ±(110–210) ns). A compressed time ladder read with the
   physical v looks like a track that is too steep in space — exactly the
   reported "+4° too steep / reference fans away with depth".
2. **A forward model fixes this per event.** Fit the raw waveform matrix
   directly: track line `pos(u) = p0 + w·u` (w = tan·v, so no v assumption
   enters), free t0, non-negative per-depth charge profile (NNLS), empirical
   impulse template, geometric spread + diffusion, and a delayed smeared
   sharing kernel. Sharing is *modelled forward*, never inverted — this is
   well-conditioned where the hit-level deconvolution ("unsharing") is not.
3. **Result (2,000-event disjoint test set, |tan₃| 0.10–0.40):** per-event
   angle residual vs M3 **σ = 1.03°/1.17°** (x/y, med −0.06°/−0.23°) — vs
   **1.58°/1.53°** for the shipped SOTA hybrid and **4.97°/5.74°** for the
   production raw ladder *on the same events*. Full-depth line deviation
   < 1 mm over the 29 mm gap for **74 %/68 %** of events (production:
   24 %/16 %); < 0.5 mm for 36 %/32 %. Mesh-position residual σ 0.58 mm both
   planes (production earliest-strip: 0.67/0.76 mm).
4. **The waveforms actively prefer the reference-sloped track.** Pinning the
   line to the production ladder costs a median Δχ²/dof of +97/+132 vs the
   free fit; pinning to the reference line costs only +16/+25. Calibrating
   *all* sharing hyperparameters while pinned to the production ladders
   plateaus at a total χ² ≈ 50 % worse than the reference-pinned calibration —
   no sharing model can make the waveforms look like the steep ladders.
5. **Physicality check passed:** the fitted transverse speed w divided by
   tan_ref is *flat* vs angle at ≈ 36.5 µm/ns, where the production ladder's
   implied v falls 56 → 39 µm/ns over the same angle range (the 46-series
   "not a physical velocity" signature). The calibration prefers
   **v = 36.7 µm/ns**, ~5–8 % above the geometry/unsharing estimate 34–35 —
   a real residual tension (§6).
6. **Honest bottom line on threading:** with the mixing modelled, the typical
   event now misses the reference line by ~0.7 mm max over the full 29 mm
   depth (unbiased, med angle offset < 0.25°). Threading *within the M3
   pointing uncertainty* (~0.2 mm) per event is still out of reach — that
   would need σ(angle) ≲ 0.4°, and the remaining per-event noise is ~1°. But
   the reconstruction is no longer systematically diverging: the miss is
   symmetric scatter, 3–5× smaller than any prior per-event reconstruction.

---

## 1. From-zero rebuild — what was taken and what was not

Taken as ground truth (validated elsewhere): the M3 rays + recipe, the
alignment transform (mesh anchor + rotated tangents; centred to < 10 µm), the
M3 pointing σ (0.206/0.242 mm aligned frame). Everything else — pedestals,
common noise, pulse shape, timing, sharing, velocity — was re-derived from the
decoded waveforms in this study.

Cache (`01_build_cache.py` → `wfcache.pkl`): for every matched muon, the
pedestal/CNS-subtracted waveforms of all strips within ±5 mm of the reference
corridor (both planes), per-channel noise σ, ftst, reference geometry
(raw-frame mesh anchor + rotated tangents), and the production per-event fits
(post-`a1cce79` matched-filter hits) for comparison.

Incidental: FEU8−FEU7 fine-timestamp offset is a constant −1 unit (mod 6) —
per-plane free t0 absorbs it.

## 2. Empirical impulse template (`03_template.py`)

Median of 1,172 bright unsaturated strips on |tan| > 0.25 tracks (direct
deposit ≤ ~1.5 samples): 10–90 rise ≈ 200 ns, FWHM ≈ 350 ns, few-% undershoot
out to ≥ 1.2 µs. Self-check: the near-vertical median pulse shape is
reproduced by template ⊗ gap-long boxcar with strong charge attenuation along
drift — independently consistent with the known det3 attachment.

## 3. The estimator-independent compression (`04_mf_ladder.py`, `05_estimator_compare.py`)

Per-strip times vs the reference-implied ladder (t0 floated per event), core
strips, |tan| > 0.08:

| u since mesh [ns] | 50 | 250 | 450 | 650 | 850 | 950 |
|---|---|---|---|---|---|---|
| matched-filter t50 residual (x) [ns] | +38 | +9 | 0 | −25 | −90 | −127 |
| leading-edge 20 % (x) [ns] | +52 | +13 | 0 | −26 | −71 | −104 |

Same shape in y (slightly larger). Ladder-slope consequences (implied
v = 1/(s·tan_ref), medians): matched filter **42/44 µm/ns** (x/y), production
`a1cce79` hits **47/50 µm/ns** — every aggregate-time ladder is compressed
20–30 %, i.e. spatially ~40–50 % too steep at v = 34. This is the entire
"reference fans away with depth" effect.

(Historical note: the June-era "time-fit v = 28–30" numbers came from
duration/pulse-width-style estimators — a *different* distortion, stretch from
the 350 ns shaper width and sharing tails. The 46-series scan minima 41–43 on
raw hits match the compression measured here.)

## 4. The forward model (`forward_model.py`)

Per plane: parameters (p0 [mm], w [mm/ns], t0 [ns]) + K=18 non-negative
charge amplitudes q_k in 60 ns arrival bins. Strip i, sample t:

```
model_i(t) = Σ_k q_k [ F_ik h(t−t0−u_k) + c1 (F_{i−1,k}+F_{i+1,k}) h_s(t−t0−u_k−τ_s)
                       + c2 (F_{i±2,k}) h_s(t−t0−u_k−2τ_s) ]
```

F_ik = geometric fraction of bin-k charge on strip i (segment boxcar ⊗
Gaussian σ_p(u) = √(σ_p0² + D_p²u)); h = empirical template; h_s = template
smeared by σ_s (the shared copy is dispersed, not just delayed). Saturated
samples (> 3550 ADC-ped) masked. NNLS for q given (p0,w,t0); Nelder-Mead
outside. ~0.4 s/plane.

**Calibration** (`06_calibrate.py`, 180 medium-angle training events, lines
pinned to the reference, v a free hyper):

| c1 | c2 | τ_s [ns] | σ_s [ns] | σ_p0 [mm] | D_p [mm/√ns] | v [µm/ns] |
|---|---|---|---|---|---|---|
| 0.306 | 0.057 | 47 | 87 | 0.098 | 0.0114 | 36.65 |

Training χ² 7.34e7 → 4.98e7. The sharing that matters is a ~31 %-per-neighbour
*broadly dispersed* delayed copy (47 ± 87 ns) — consistent in spirit with the
hit-level kernel (0.45–0.52 aggregate amplitude ratio, +69 ns) but measured
here as a waveform-level transfer function.

## 5. Results on the disjoint test set (`07_freefit.py` → `09_three_way.py`)

2,000 events, |tan₃| 0.10–0.40, production & SOTA numbers computed on the
same events:

| per-event angle residual vs M3 | x med / σ | y med / σ |
|---|---|---|
| production raw ladder (v=v_cal) | +0.08° / 4.97° | −0.22° / 5.74° |
| SOTA hybrid (unshared + calibrated, script 34) | −0.09° / 1.58° | −0.32° / 1.53° |
| **waveform forward fit** | **−0.06° / 1.03°** | **−0.23° / 1.17°** |

| full-depth max line deviation over 29 mm | < 0.5 mm | < 1.0 mm | < 1.5 mm |
|---|---|---|---|
| production raw ladder (x / y) | 6 % / 4 % | 24 % / 16 % | 43 % / 32 % |
| **forward fit (x / y)** | **36 % / 32 %** | **74 % / 68 %** | **90 % / 86 %** |

Mesh position: med +0.045/−0.011 mm, σ 0.583/0.588 mm (x/y).
Implied v flat vs angle (36.3–36.7 x; 38→35.7 y) where production falls 56→39.
Model comparison per event: median Δχ²/dof (constrained − free) = +16/+25 for
the reference line vs +97/+132 for the production ladder; the reference line
fits better than the production ladder in 79 % of plane-fits.

Anti-circularity control (`hyper_prod.json`, `freefit_prodcal.pkl`):
recalibrating *all* hypers with lines pinned to the production ladders
converges 49 % worse in total χ² (7.44e7 vs 4.98e7) and picks contorted
values (τ_s = 176 ns, σ_s = 164 ns, D_p ≈ 0, v = 31) — no sharing model makes
the waveforms look like the steep ladders. And free fits run with those
prod-taught hypers *still* agree better with the reference (σ 3.4°/3.9°,
med ≤ 0.35°) than with production's own slopes (σ 4.5°/4.7°), and do not
reproduce the angle-dependent steepening. The data pull toward the reference
under either calibration.

Figures: `three_way_comparison.png` (flagship), `ff_w_vs_tan.png`,
`ff_v_vs_angle.png`, `ff_mesh_res.png`, `ff_dchi2.png`, `ff_threading.png`,
`forward_fit_calibrated.png` (event displays),
`estimator_residual_curves.png`, `mf_residual_vs_depth.png`, `template.png`.

## 6. Open items / caveats

1. **v = 36.7 vs 34 ± 1.5 (geometry) / 34–35 (unshared).** The forward fit
   prefers a 5–8 % higher velocity. The y-plane implied-v shows a small
   residual angle trend (38 → 35.7), so part of this is residual model
   imperfection (template/gain systematics leaking into w). Not resolved.
2. **Remaining per-event noise ~1°** is model-systematics dominated
   (per-strip gain spread is not modelled; template is global; saturation is
   masked, not modelled; δ-rays are real). Obvious next steps: per-strip gain
   nuisances, per-FEU templates, joint two-plane fit sharing one charge
   profile, saturation recovery model.
3. **Near-vertical planes (|tan| ≲ 0.08)** carry no slope information in
   timing (charge sits on 1–2 strips) — the free slope can wander. Position
   is fine. Any production use should constrain or downweight slope there.
4. Test set is medium-angle (0.10–0.40); steepest and near-vertical
   populations not yet batch-characterised.
5. Calibration pins lines to the M3 reference — legitimate (like alignment),
   and the anti-circularity control + physicality tests guard it, but a
   fully-independent calibration (e.g. per-strip gains from vertical tracks,
   template per FEU, v from an independent measurement) would be cleaner.

## 7. Comparison with the current analysis (REFERENCE_TRACK_THREADING_REPORT.md)

- Confirms: no misalignment; divergence-with-depth real on aggregate hits;
  hit-level unsharing direction (ensemble ~34 µm/ns) — all reproduced from
  zero.
- Sharpens: the divergence is *not* specifically a rising-edge/CFD artefact —
  it is intrinsic to any single-strip aggregate time; the mixing must be
  modelled (forward) or removed (deconvolution). Forward modelling is
  well-conditioned per event, deconvolution is not — which is why unsharing
  helped only in ensemble.
- Supersedes (for per-event work): "there is currently no per-event
  reconstruction that threads a single muon through the whole gap" — the
  forward fit now threads ~70 % of medium-angle events to < 1 mm over the
  full depth with unbiased angles, beating the SOTA ensemble resolution
  per event.

## 8. Reproduction (phase 1)

```bash
cd mx_june_cosmic_qa/waveform_first_threading
../../.venv/bin/python 01_build_cache.py        # ~10 min, 23 MB cache
../../.venv/bin/python 03_template.py
../../.venv/bin/python 04_mf_ladder.py
../../.venv/bin/python 06_calibrate.py --mode ref   # ~30 min, 14 cores
../../.venv/bin/python 07_freefit.py --hyper hyper_ref.json --n 2000
../../.venv/bin/python 08_threading_metrics.py freefit.pkl
../../.venv/bin/python 09_three_way.py
```

Everything reads/writes `<Analysis>/mx17_3/waveform_first/`. Scripts were
developed in a session scratchpad and copied here verbatim; paths to the
Analysis tree are absolute.

---

# Phase 2 (same day, second pass): reconstruction R&D — model v2, benchmarks, the physics floor, and v(E)

Mandate: find the best possible reconstruction for this detector. Scripts
`11`–`19` + `forward_model2.py`; all outputs in `waveform_first/`.

## 9. Model v2

Diagnostics first (scripts 11/12):

- **Per-plane templates**: X and Y impulse responses have identical rise
  (10–90 = 180 ns) and FWHM (~290 ns), but **Y's undershoot is 4× deeper
  (−8.5 % vs −2.3 %)** — deep hits ride on the undershoot of earlier shared
  copies, which was Y's excess scatter (see §11). (Connects to PLAN_47's
  "Y slow rise" — at 60 ns sampling the difference is in the recovery, not
  the rise.)
- **Per-channel gains**: ensemble flat-field from the v1 fits gives a spread
  of only **1.4–1.5 %** (452/446 channels measured) — strip gain is NOT a
  significant residual term.
- **FEU7→FEU8 timing**: t0x − t0y = −18.8 ns constant, +60 ns when the ftst
  difference wraps (−5 ≡ +1 mod 6; ftst unit = 10 ns). Stored in
  `dt_xy.json`; makes the joint two-plane fit well-defined.

v2 = v1 + per-plane templates + gain correction + saturation *censoring*
(one-sided penalty for clipped samples) + per-plane sharing scale kY + optional
joint two-plane fit (shared charge profile, tied t0). Recalibrated (8 hypers,
same 180-event ref-pinned protocol): c1 = 0.288, c2 = 0.048, **kY = 1.375**
(Y sharing 38 % stronger — matches the hit-level kernel asymmetry
0.516/0.449), τ_s = 47 ns, σ_s = 90 ns, σ_p0 = 0.087 mm, Dp = 0.008,
**v = 36.60** (unchanged from v1).

## 10. Benchmark (2,000-event test set, |tan₃| 0.10–0.40)

| variant | ang σ x/y [deg] | mesh σ x/y [mm] | <1 mm x/y [%] | v-flat x/y | notes |
|---|---|---|---|---|---|
| production raw ladder | 4.97 / 5.74 | 0.67 / 0.76* | 24 / 16 | 17 / 17† | *report §2 |
| SOTA hybrid (unshared+cal) | 1.58 / 1.53 | — | — | — | same events |
| mf ladder + slope remap (cheap) | 1.51 / 1.86 | 0.79 / 0.73 | 54 / 52 | 9.4 / 9.7 | script 16 |
| forward v1 | 1.03 / 1.17 | 0.58 / 0.59 | 74 / 68 | 0.45 / 2.28 | |
| **forward v2** | **1.06 / 1.10** | 0.57 / 0.59 | 74 / 70 | **0.57 / 0.99** | |
| forward v2 joint | 1.04 / 1.09 | 0.63 / 0.56 | 73 / 72 | 1.19 / 1.98 | n=1000 |

(† prod falls 56→39 µm/ns over the angle bins — v-flat is that spread.)

- v2's win over v1 is **Y**: σ 1.17→1.10 and v-flatness 2.28→0.99 (per-plane
  template + kY). X was already at its floor.
- The joint fit changes nothing at medium angles (t0 was not limiting); its
  value is for near-vertical slope stabilisation (§13).
- The **cheap path** (matched-filter re-time + one trained slope remap,
  α ≈ 0.86–0.89 = the decompression) already reaches SOTA-hybrid σ at
  trivial cost — but cannot fix the angle-dependent bias (v-flat ~9) and
  threads 20 pp worse than the forward fit. Good candidate for a fast
  production pre-pass; not a replacement.

## 11. Ablations (script 17; 800 events, no recalibration)

Every single-component ablation (no gain map, one shared template, no
censoring, c2 = 0, σ_s = 0, Dp = 0, kY = 1) stays within statistical noise
(±0.05°) of full v2 on **per-event angle σ**. The components matter only for
*ensemble physicality*: dropping the per-plane template (y v-flat 0.99→2.06),
kY (→1.86), or c2 (→1.71) re-introduces an angle-dependent implied-v.
**Implication: the essential ingredients are the sharing kernel (c1, τ_s) and
the per-plane template; everything else is a refinement.** A production
implementation can be simple.

## 12. The resolution floor is physics, not reconstruction (script 18)

Toy closure: waveforms generated from the calibrated v2 model with known
tracks (real event geometries, real fitted charge profiles, real noise,
clipping), then fit back:

| per-60ns-bin transverse charge-centroid jitter | angle σ x/y [deg] | p0 σ [µm] |
|---|---|---|
| 0 (electronics noise only) | **0.02 / 0.02** | 7 / 6 |
| 0.15 mm | 0.55 / 0.54 | 219 / 172 |
| **0.30 mm** | **1.05 / 1.04** | 354 / 295 |
| 0.50 mm | 1.77 / 1.78 | 473 / 484 |

Electronic noise contributes essentially nothing (0.02°). A **0.30 mm RMS
per-depth-bin centroid jitter reproduces the observed ~1.05° exactly** — the
scale expected from transverse diffusion (~0.3–0.4 mm at mid-drift) sampled
by few avalanche-weighted effective electrons per 2 mm of track, plus
δ-rays. **The ~1° per-event scatter is the physical information limit of
this detector geometry; v2 is extracting essentially all of it.** Gains
beyond this come only from ensemble averaging, not better per-event fitting.

## 13. All-angle behaviour and near-vertical handling

freefit2_all.pkl: 4,200 events, all angles, v2 single-plane fits:

| plane | \|tan\| bin | <0.04 | 0.04–0.08 | 0.08–0.15 | 0.15–0.25 | 0.25–0.45 |
|---|---|---|---|---|---|---|
| x | angle σ [deg] | 1.42 | 1.16 | 1.10 | 1.06 | 1.03 |
| x | mesh σ [mm] | 0.53 | 0.49 | 0.55 | 0.59 | 0.65 |
| y | angle σ [deg] | 2.35 | 1.07 | 1.03 | 1.02 | 1.07 |
| y | mesh σ [mm] | 0.47 | 0.50 | 0.58 | 0.72 | 1.01 |

Graceful, not catastrophic: slope quality is flat at ~1.0–1.1° everywhere
above |tan| > 0.04 and degrades to 1.4° (x) / 2.4° (y) robust σ below it
(with occasional wild-slope outliers — single-event slope should still not be
*trusted* there; use joint fit or report position-only). Mesh position is
actually best at near-vertical (charge concentrated on few strips). The
per-event angle bias stays < 0.1° (x) / ~−0.3° (y, the §16.2 residual).

## 14. Drift velocity vs field — the model generalises and the v tension resolves (scripts 19/20)

Ref-pinned χ²(v) scans per drift-scan subrun (v2 hypers frozen, only v
scanned; 250 events each):

| drift HV | 300 | 500 | 700 | 900 | 1000 (long) | 1100 |
|---|---|---|---|---|---|---|
| forward fit v [µm/ns] | 12.0 | 20.6 | 26.4 | 35.5 | 36.6 | 38.8 |
| unshared ladder (46-series) | 3.4 (broken) | 14.2 | 23.3 | 30.6 | 34.3 | 35.3 |

Figure `v_vs_E_forward.png`:

- The forward-fit series is smooth and monotonic, and lies **between the
  Magboltz Ar/iso 95/5 + 1 % H₂O and + 0.3 % H₂O curves** (RMS 3.6 µm/ns vs
  +1 % H₂O at ≥700 V; dry 95/5 and 90/10 are far off). The Saturday run was
  the driest of the week (3 %→1 % drying trajectory), so a ~0.5–0.7 %
  effective humidity is entirely plausible.
- **The v = 36.6-vs-34 tension is a humidity ambiguity, not a method
  contradiction**: the unshared ladder matched the +1 % H₂O curve, the
  forward fit a slightly drier one. |v| carries a ~5 % gas-composition
  systematic either way; the *shape* of v(E) is reproduced by both.
- At low field, where window truncation breaks gap-based estimators (500 V:
  geometry gives 12.2 with only ~1.35 µs of a needed ~2.4 µs visible), the
  forward fit stays on-curve — it never needs the full gap.

## 15. Recommended reconstruction (the deliverable)

1. **Per-event tracking (angles + positions): forward model v2**, single-plane
   fits; joint fit (or position-only) for |tan| < 0.08 planes. Expected:
   σ(angle) ≈ 1.0–1.1° (at the physics floor), mesh σ ≈ 0.58 mm,
   unbiased (< 0.3°), ~70 % of events threading < 1 mm over the full gap.
   Cost ~0.4–1 s/plane (unoptimised NumPy; NNLS dominated — vectorisation
   and warm starts can plausibly reach ~50 ms/plane).
2. **Fast pre-pass / trigger-level: matched-filter ladder + slope remap**
   (script 16): SOTA-level σ at negligible cost, biased at the ±0.3° level,
   fine for monitoring and selection, not for physics angles.
3. **Ensemble drift-velocity / gas monitoring: ref-pinned χ²(v) scan**
   (script 19) — works at every HV including truncated low-field runs.
4. Calibration protocol per detector/run: impulse template per plane
   (bright inclined strips), sharing kernel + v by ref-pinned hyperfit on
   ~200 medium-angle muons, optional gain map. All automated in scripts
   03/11/12/13.

## 16. Phase-2 open items

1. Absolute |v| to better than ~5 % needs an independent humidity/gas handle
   (or a gap-independent length calibration); currently degenerate with H₂O
   fraction.
2. The residual +0.6–1.2 µm/ns v-flat slope (y) hints at percent-level
   residual model imperfection; harmless at the current floor.
3. Speed: production-grade implementation (vectorised NNLS, warm starts,
   C++/numba) not yet done.
4. Generality: apply the calibration protocol to det2/det4 bench runs (and
   eventually the July beam runs of the same chambers) to confirm
   portability.
5. Near-vertical joint-fit benchmark on a dedicated |tan|<0.08 sample.

---

# Phase 3 (2026-07-26): the velocity tension resolved, the gas pinned, and the production stack completed

## 17. The v = 36.6-vs-34 tension — clean re-examination (scripts 22-25)

Four independent attacks; all four favour the forward-fit velocity.

**17.1 Dissecting the geometry estimator (34.3).** Scripts 21/23 compute
`v_core = (core cluster-extent slope vs |tanθ|) / (core time-span plateau)`.
The *time* part agrees with this study: the 46-series T_sat (676 ns at
1000 V) matches the forward fit's deconvolved gap-crossing duration
U₅₀ = 674 ± 4 ns. The discrepancy is entirely in the *extent column*: the
46-series' own time-free measurement gives 24.5 mm (→ 24.5/0.676 = 36.2
µm/ns), while v_geom = 34.3 corresponds to an effective column of 23.2 mm.
The 1.3 mm difference is the treatment of the ~2 mm resistive-spread floor
in the extent-vs-angle slope. **The entire tension is a floor-subtraction
ambiguity in the geometry method** — which the forward model handles
explicitly (σ_p0, D_p, sharing kernel).

**17.2 Toy calibration-bias bound (script 24).** Toys generated with a
*deliberately different* sharing truth (hit-level kernel: per-plane
c1 = 0.449/0.516, c2 = 0.052/0.151, τ = 69 ns, **no dispersion smear**,
σ_p0 = 0.30 mm) at **v_true = 34.0**, then run through the standard 8-hyper
calibration: recovered v = **33.11** (−0.9), with the truth kernel found
(c1 = 0.456, τ = 64, σ_s = 2 ≈ 0, σ_p0 = 0.25). Under strong model mismatch
the calibration *deflates* v by ~2.6 %; it cannot inflate 34 → 36.6.
Model systematic on v: ~±1 µm/ns.

**17.3 Profile likelihood (script 25).** χ²(v) with τ_s, σ_s, σ_p0 re-fit
at every v (training set): sharp minimum at **v = 36.7**; v = 34.5 costs
+4.4×10⁵ (0.8 % of total χ²) even with the sharing re-optimised; the
sharing-parameter drift along the valley is mild (τ 48→56 ns over the whole
33–39 range) — the v↔sharing degeneracy is weak.

**17.4 The charge-visible column (scripts 22/23) — new physics input.**
The deconvolved charge-arrival profile (NNLS q_k, template and sharing
removed) is **flat with a sharp edge** (edge width 50–80 ns):
- U₅₀ = 674 ± 4 ns at 1000 V; angle-independent; unchanged for early-t0
  events (rules out window truncation).
- Implied charge-visible column v·U = **24.7 ± 0.1 mm** at 1000 V, and
  24.0 ± 6.0 / 24.8 ± 3.1 mm at 900/1100 V — **constant vs HV**, and
  consistent with the independent time-free extent column (24.5 mm).
- The flatness also *dissolves an old paradox*: the amplitude-vs-depth
  decline (runs 17–19, "attachment") is evidently mostly **diffusion
  spreading + per-strip threshold** — per-strip amplitudes drop with depth
  but the summed collected charge does not.
- **Open hardware question:** the visible column (~24.5–25 mm) is 4–5 mm
  short of the nominal 29–30 mm gap, sharply, at all HV. Either the true
  drift gap is smaller than nominal, or the last ~4 mm below the cathode do
  not deliver charge (dead region / field distortion). This is now the
  single dominant unknown in the gap-based velocity conversation and needs
  a mechanical/HV-mapping answer.

**Verdict: v(1000 V) = 36.7 ± 0.3 (fit) ± 0.9 (model) µm/ns.** The
geometry-estimator 34.3 is identified as floor-subtraction-biased; the
old gap-filling 42 and duration-based 28–30 numbers were already understood.
`45_slope_reference_vdrift_scan`-era conclusions that unsharing "converges
to 34" inherit the same extent/floor systematics at the 1-2 µm/ns level.

## 18. Magboltz gas fit to the forward v(E) (scripts 21 + garfield_sim water grids)

New grids: 11 local mixtures (`mm_water_grid2_local.py`, ~2.6 h each on
14 cores) + 30 lxplus-condor jobs (`mm_water2d_one.py`, cluster 11848941):
fine H₂O 0.4–1.1 % at 95/5, iso 3–8 % × H₂O grid, N₂/air contamination
variants. Reproducibility: the same mixture run locally and on condor
agrees to 0.01 µm/ns.

- Best fit at operating fields (≥700 V): **Ar/iso 95/5 + 0.80 % H₂O**,
  RMS 0.63–0.67 µm/ns; the water parabola (0.70 → 2.2, 0.75 → 1.4,
  0.80 → 0.67, 0.85 → 0.89, 0.90+ worse) interpolates to
  **0.81 % (gap-30 E-convention) / 0.86 % (gap-29)**.
- Iso fraction: 5 % clearly preferred at ≥700 V (4 % marginally better on
  all-points only because of the least-reliable low-HV points).
- Trace N₂ (0.25–1 %) is degenerate at the 0.05-RMS level — not resolvable.
- Consistent with the known drying trajectory (bottle humidity 3 %→1 %
  during the week; Saturday = driest).

Figures: `v_vs_E_forward_refined.png`, `rms_vs_h2o.png`;
`garfield_sim/results/water_grid2.json`, `water2d.json`.

## 19. Production stack completion

- **forward_model3.py** — vectorized fitter (cached time tensors, vectorized
  strip fractions, coarse-grid + short-NM search): **~5× faster** (0.5 s →
  ~0.1 s/plane uncontended) and finds a *lower* χ² than the v2 fitter in
  ~20 % of events (the coarse grid escapes local minima; 11/12 disagreements
  resolved in fm3's favour). Same model, same hypers.
- **wft_reco.py** — the production API: `WFTReco(calib_dir)` loads the
  calibration bundle (per-plane templates, gain map, hypers+v, FEU t0
  offsets); `fit_plane()/fit_event()` return position, angle, charge
  profile, χ², **errors = statistical curvature ⊕ physics floor**
  (FLOOR_TAN = 0.018, FLOOR_P0 = 0.33 mm from §12), `slope_reliable`
  (|tan| ≥ 0.08) and `quality_ok` (χ²/dof < 300) flags, optional joint
  refit when one plane is near-vertical.
- **det4 portability** (script 26 + standalone freefit): the full protocol
  (cache → per-plane templates → 8-hyper calibration → free-fit benchmark)
  ran end-to-end on the g_det4 day run, absorbing a different DAQ window
  (mixed 32/37 samples — `fm2.set_nsamp()` added), and found a genuinely
  different detector: c1 = 0.24, **kY = 2.36** (det4's Y shares far more),
  σ_s = 234 ns, **v = 34.2 µm/ns** (det4's own drift point/gas).
  Benchmark (800 events, |tan₃| 0.10–0.40): **angle σ 2.00°/1.97°**
  (bias ≤ 0.3°), mesh σ 0.93/0.71 mm, full-depth < 1 mm for 48/51 % —
  ~2× det3's floor, as expected for the gain-limited detector, and a mild
  residual implied-v angle slope (33 → 31) indicating det4 would profit
  from its own refinement pass (per-plane c1, template quality). Freefit of
  800 events took 42 s on 14 cores with fm3.

## 20. Updated open items

1. **Hardware: the ~4–5 mm missing drift column** (§17.4) — is the true
   gap ~25–26 mm, or is there a dead/distorted region below the cathode?
   Check mechanical drawings / assembly records; a Garfield field map of
   the real geometry would settle it.
2. |v| to better than ±1 µm/ns needs an independent humidity measurement
   (gas analyzer) — the Magboltz water fraction (0.8 %) is otherwise the
   best constraint.
3. det4 numbers to be added; det2 after.
4. Speed: fm3 at ~0.1 s/plane; a numba/C++ port of build_matrix+NNLS is the
   next 10×.

---

# Phase 4 (overnight 2026-07-26/27): the "short gap" dissolves into cathode topography

Adversarial re-examination of §17.4's 24.7 mm charge-visible column before
any hardware action (scripts 29–35 in the session scratchpad; figures
`endpoint_robust.png`, `K_scan.png`, `gap_map.png`, `tail_and_dets.png`).

## 21. The 24.7 mm number was biased; the honest column is ~28 mm with ±2–3 mm topography

**21.1 Two stacked estimator biases found.**
1. *Median over sparse NNLS profiles*: adjacent 60 ns charge bins are
   degenerate under the 350 ns template, so per-event NNLS solutions are
   sparse (per-bin occupancy ~50 %); a per-bin median then truncates the
   profile tail. Mean / trimmed-mean / rebinned-median estimators read
   ~+40 ns longer.
2. *Free-fit vs ref-pinned profiles*: the §17.4 profiles came from free fits
   (p0, w, t0 all floating), which trade slope/t0 against profile support.
   Ref-pinned (M3 line) profiles — the correct configuration for measuring
   the physical column — read another ~+45 ns longer.

With both fixed: **U50 = 762 ns, stable for basis K = 18…27** (junk
accumulates only at K = 30), i.e. **column = 27.9 mm** at v = 36.6, with
u95 = 810 ns (29.6 mm).

**21.2 Closure-validated.** Toys with known step-columns (674/720/793 ns),
real geometries/noise, fit + estimator chain as in data: recovered to ≤5 ns,
*including* toys generated with a 2×-deeper template undershoot and fit with
the standard template. The chain cannot shrink a 29 mm column to 762 ns —
and the data edge is *soft* (roll-off ~700→810 ns), unlike the recovered
step-toys: the endpoint varies event to event.

**21.3 It varies with position: gap topography.** U50 vs track position
(7,240 plane-profiles): det3's column runs **26.5–29.5 mm** across the
active area — a monotonic **~2.7 mm tilt along y** (29.3 → 26.6 mm) plus a
slight bow in x (edges high). det4 shows **its own, different ±3 mm
pattern** (overall 28.8 ± 0.3 mm at det4's less-certain v). The soft
ensemble edge is exactly the stack of these varying endpoints.

**21.4 Mild HV dependence on top.** det3 mean column 28.4 → 27.9 → 27.8 mm
at 900/1000/1100 V (~2σ trend): a small **electrostatic pull** (∝E²)
superposed on the fixed mechanical tilt; the y-tilt persists at every HV.
At 500 V (extended basis K=26) the column reads 29.1 mm — consistent with
less sag at low field.

**Verdict: there is no 25 mm gap.** The drift gap is nominal (~29 mm) at
the frame; the cathode plane is **tilted/bowed by ~2–3 mm** (chamber-specific
pattern, det3 tilt mostly along y), with a small additional electrostatic
deflection at operating fields. §17.4's "single dominant unknown" is
resolved as detector-plane flatness — measurable in situ by this method
(per-chamber gap maps), no disassembly required to see the pattern, though a
mechanical check of cathode flatness/standoffs would confirm the origin.

Consequences:
- All *gap-based* velocity estimators inherit a −3…−7 % bias from using
  29 mm where the local column is 26.5–29.5 mm — (another) reason v_geom
  read low. The forward-fit v (angle-based, gap-free) is unaffected.
- The per-event maximum drift depth varies with position; depth-dependent
  analyses (attachment/efficiency vs z) should use the local column, not a
  global gap.
- det2 (detector B) run in progress for the third-chamber cross-check.

## 22. Third-chamber control: det2 (detector B) is FLAT at the full gap

Full protocol on the 6-22 overnight long run (13,556 candidates; script
`26b_det2_pipeline.py` + `wf36`): calibration lands on a kernel nearly
identical to det3's (c1 = 0.29, kY = 1.42, τ_s = 49 ns, σ_s = 91 ns — same
production batch), v = 39.9 µm/ns at det2's run conditions. Reconstruction:
angle σ 1.33°/1.54°, mesh σ 0.76/0.56 mm — between det3 and det4, healthy.

**det2 column: 30.5 mm overall, and the 3×3 map is flat (29.9–31.5 mm).**
The method reads the full mechanical gap on a flat chamber. Chamber summary:

| chamber | column avg | topography |
|---|---|---|
| det2 (B) | 30.5 mm | flat (±0.8 mm) |
| det3 (A) | 27.9 mm | ~2.7 mm tilt along y + slight x-bow (26.5–29.5) |
| det4 (E) | 28.8 mm | ±3 mm bow, own pattern |

(Absolute scales carry each chamber's v calibration ±3–5 %; the *relative*
topography is robust.)

Cross-check note: a crude NNLS-free estimate (cluster-summed near-vertical
pulses vs template⊗boxcar) reads 24.9/26.4 mm on det3 — same ballpark,
biased ~1–2 mm low by its peak-normalized/no-sharing approximations; the
closure-validated NNLS chain and the flat det2 control are the decisive
measurements. (A single-strip version of the same check reads 20–21 mm —
even near-vertical tracks cross ~1 strip pitch over the gap, so single-strip
durations under-cover the column; do not use.)

**Final answer to "is the gap really 25 mm?": No. Do not disassemble on
this evidence.** The gap is nominal at the frame on all three chambers;
det3(A) has a ~2.7 mm cathode-plane tilt (plus ~0.5 mm electrostatic pull at
operating field) and det4(E) a ±3 mm bow — worth a mechanical flatness check
whenever a chamber is next open, but the in-situ U50(x, y) map (script 34)
measures it non-invasively per chamber, and should simply become part of the
per-detector calibration bundle.

---

# Phase 5 (2026-07-28): the displays — does the M3 track thread the cluster?

Script `37_threading_displays.py`, write-up `THREADING_DISPLAYS_2026-07-28.md`.

The question §0 opened with, answered in picture form and then measured. The
cluster in the displays is **not** taken from the forward fit (that would be
circular): it comes from a line-free 2-D charge deconvolution (strip x depth,
NNLS + Tikhonov) and from a sub-pitch **free ladder** — one free transverse
position and charge per 60 ns depth bin, no relation imposed between bins. Only
`t0` is borrowed from the fit, and it shifts both clusters together.

Over 600 events, charge-weighted median |cluster - M3 line|: production
0.49/0.55 mm (x/y) vs waveform-first **0.44/0.39 mm**, and above 15 mm depth
0.64/0.70 vs **0.52/0.51 mm**. The depth profile is the point: production walks
from 0.45 to 0.66 mm across the gap (0.43 -> 1.17 mm in its own t0/v frame,
which is what the existing 3-D displays draw), the waveform-first cluster stays
0.39 -> 0.55 mm. The divergence with depth is real and is largely removed; both
methods agree at the mesh at the M3 pointing floor (~0.4 mm).
