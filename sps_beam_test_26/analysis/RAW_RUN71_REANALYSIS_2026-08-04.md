# run_71 reanalysis — the ">3.8 µs tail" was three artefacts, and the drift is slow but fine

Ground-up reanalysis of the Sunday-night flat data (run_63 flat, run_71 RAW),
prompted by the question: *is the drift time really more than 4 µs? that can't
be true unless the drift is broken.* 2026-08-04.

Answer up front: **the drift is not broken.** At the 700 V operating point the
full drift ladder completes in ≈ 2.1 µs, inside the window. The gas is
~4× slower than dry-gas Magboltz says it should be (open item, water is the
prime suspect), so the *deliberately low* drift points of the scan
(450 / 275 V) do run their ladders off the end of the 3.84 µs window — but
that is slow drift at low field, not a detector fault. The
"44–52 % of the signal still present at the last sample" number that drove
the window-wall conclusion is dominated by three measurement artefacts, all
identified and removed below.

Scripts + figures: `staging/run_71/reanalysis_2026-08-04/` on the data disk.
The 700 V RAW block (groups 002–007) was pulled from EOS for this — the
previous pass never had a high-field RAW point.

---

## 1. Three artefacts in the mean waveform

The window-wall claim rested on **peak-aligned mean waveforms with no
per-event baseline inspection**. In absolute window time the mean central
strip sits at **22–31 % of its peak *before the pulse starts*** — before the
triggering particle's charge exists. Three causes, in decreasing order:

1. **Two oscillating channels, ch 510 and ch 372 (both Y view).** They swing
   ±400–900 ADC quasi-periodically (~800 ns) with quiet neighbours, and they
   pass the 400–3000 ADC leading-strip selection: **ch 510 alone was the
   "central strip" of 22 % of all selected events** (689/3171 at 450 V).
   Their flat-in-time swings pad the mean's pre-window and tail equally.
2. **Beam pile-up.** Individual events show additional pulses at random times
   in the 3.84 µs window (H4 spill, det4 sees the full beam, not just the
   triggered track). The leading-strip selection preferentially *chooses* the
   pile-up pulse when it is bigger than the trigger pulse. ~15–20 % of
   events carry a pile-up structure in the 540 ns pre-window alone.
   The median event is clean: pre-level p50 = 3 ADC, but p95 = 481 ADC.
3. **No per-event baseline subtraction** in the RAW chain (`per_strip` has no
   local baseline, contrary to the comment in `extract_det4_only.py`), so 1+2
   went straight into every mean.

Fix: exclude ch 510/372 as central strips, require |pre-window mean| < 15 ADC
(keeps 60–64 %), subtract the per-event pre-level, and use per-sample
median / 20 %-trimmed mean. (`robust_waveforms.py`, `kernel_refit_clean.py`)

## 2. The clean central response is CONTAINED — the window wall retracts

Per-sample median, clean events, Y view:

| plateau | peak | returns < 5 % of peak | level at last 4 samples |
|---|---:|---:|---:|
| raw450 | 960 ns | **1980 ns** | **−3.8 %** (undershoot) |
| raw275 | 1080 ns | **1800 ns** | **−6.1 %** (undershoot) |

The central strip's own response is over ~2 µs into the window and ends
*negative* — a small AC/shaper undershoot. **"44–52 % of peak at the last
sample" is retracted**; with artefacts removed the number is −4 to −6 %.
`tau_s` and `c2` are therefore *measurable* in this data after all — but see
§4 for what actually limits them now.

What stays true from the old pass: the far neighbours' dispersed tails ARE
long. ±2 peaks very late (~2–2.8 µs) at ~6 % of the central peak and holds to
the window end; ±3 is still rising at 3.84 µs. That is slow transverse RC
spreading on the resistive surface — a few-percent effect on far strips, not
a 44 % effect on the central one.

## 3. Drift: the ladder is visible, it scales, and it is SLOW

The three RAW plateaus have **sample-for-sample identical** central pulse
cores (on50 = 803 ns, peak = 1020 ns, off50 ≈ 1600 ns at 700, 450 and
275 V): the leading edge is first-arriving charge ⊗ shaper and cannot depend
on drift field. The field dependence lives in the tail, exactly where a
drift ladder seen through a quasi-differentiating (AC) front end should be:

* **700 V: a deep end-of-ladder negative lobe** (−0.30 of peak, minimum
  ~3.1–3.2 µs, also on ±1) — the current *stops* inside the window.
  Ladder start ~650 ns ⇒ **T_drift(233 V/cm) ≈ 2.0–2.3 µs ⇒
  v ≈ 13–15 µm/ns.**
* **450 V: only a shallow late negative** — the ladder end
  (2.1 µs × 233/150 ≈ 3.3 µs after start) sits at the window edge.
* **275 V: no end lobe, positive-lobe tail thinnest, last samples most
  negative-sagged** — the ladder (≈ 5.3 µs scaled) runs off the window.
  This is the kernel of truth in the old ">3.8 µs" number.
* Independent cross-check: µTPC-style dt/dx on the 25.64° run_63 ladder
  gives an apparent **v ≈ 16 µm/ns at 142 V/cm** (154 clusters; aggregate-
  time sharing bias makes this an over-estimate per the June work) —
  consistent with the end-lobe number and v ∝ E (constant mobility) across
  92–233 V/cm.

Consequences: at the operating point (700 V) everything completes by
~2.8 µs — no reconstruction problem. The 450/275 V diffusion-lever points
are *window-truncated by drift*, so any tail-integral quantity compared
across drift voltages must model that; shape-invariance tests should use the
first ~1.8 µs only, where all three plateaus are complete-enough and, in
fact, identical.

**Open item — why is the gas ~5× slower than dry Magboltz?** Magboltz
(this reanalysis, CERN pressure 720.8 Torr, ncoll 5,
`garfield_sim/results/drift_velocity_beamtest_cf4_CERN.json`), dry
Ar/CF4/iC4H10 88/10/2:

| E [V/cm] | 50 | 70 | 91.7 | 120 | 150 | 190 | **233** | 300 | 400 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v [µm/ns] | 18.3 | 26.0 | 34.0 | 43.9 | 53.4 | 64.6 | **74.7** | 86.9 | 98.7 |

Dry prediction at the operating field: **74.7 µm/ns (T_drift 0.40 µs)**
against the measured **13–15 µm/ns (≈2.1 µs)** — a factor ~5.
Also note the measured v is closer to **v ∝ E** (constant mobility) while
the dry curve is strongly sub-linear; heavy vibrational cooling by a polar
contaminant (water) produces exactly that low-energy behaviour.

**Water closes the gap quantitatively.** Same mixture + 1 % H₂O
(`vdrift_wet.py`, ncoll 4):

| E [V/cm] | dry | +1 % H₂O | +2 % H₂O | measured |
|---|---:|---:|---:|---:|
| 91.7 | 34.0 | 6.9 | 3.8 | — |
| 150 | 53.4 | 11.9 | 6.4 | ~10–12 (µTPC at 142 V/cm, bias-corrected) |
| 233 | 74.7 | 20.1 | 10.4 | **13–15** (end lobe) |

The measured 13–15 µm/ns at 233 V/cm sits between the 1 % (20.1) and 2 %
(10.4) curves: **water content ≈ 1.3–1.7 %** at run_71 time
(`drift_velocity_beamtest_cf4_wet{1,2}_CERN.json`). That is exactly the June det3 story — a freshly-plumbed chamber
carrying percent-level water and drying over days. The dry CO₂-mixture curve
(`drift_velocity_beamtest_co2_CERN.json`: 39.4 µm/ns at 233 V/cm, 0.76 µs)
says even the *first-night* gas was never the problem. Consequences:
run-by-run v is unusable from tables for this campaign (the water content
was drying with time); any depth-resolved analysis must take v from the data
itself (end-lobe or µTPC per run). Candidates, with the evidence in hand:

1. **Water.** The chamber travelled and the H4 line is new; bench det3
   carried >3 % water for a week in June. A wet-mixture Magboltz scan is the
   discriminator (started; `drift_velocity` wet points).
2. **The CF₄ exchange may not have completed.** run_60 was "taken while the
   gas was still changing"; nobody measured the flush constant. But even the
   *old* gas, dry, is ~3× faster than measured — this cannot be the whole story.
3. **The drift line's load is non-ohmic** (hv_run71: 131 nA @ 700 V,
   6 nA @ 450 V, 0 @ 275 V; an intact ~GΩ/ring degrador draws I ∝ V).
   det4's field cage deserves the same audit detector B got. For central
   tracks the cathode still sets ≈ V/gap, so this is unlikely to explain 4×
   by itself — but it is a real hardware finding of its own.

## 4. The kernel: invariance PASSES on clean data; the cascade model is the wrong container

Refit of `W_d = α_d W_0 + β_d (W_0 ⊛ K_τ^{|d|})` on clean trimmed-mean
waveforms (`kernel_refit_clean.py`):

| | raw450 | raw275 |
|---|---:|---:|
| τ_s | 1308 ns | 1316 ns |
| c1 | 0.635 | 0.623 |
| c2 | 0.621 | 0.659 |
| α(±1) prompt | 0.168 | 0.178 |

**The drift-invariance test now passes** (τ ±0.6 %, c1 ±2 %, c2 ±6 %) — the
old "test fails" verdict was the artefacts, not the physics. The premise
that the kernel belongs to the resistive layer holds.

But do not quote these β as charge fractions: β(±2) ≈ β(±1) and Σβ ≫ 1 —
the one-pole-cascade decomposition against a basis that itself loses charge
sideways (and undershoots) is not a physical charge accounting. The
model-independent charge budget over the window (trim20, clean, /central):

| d | 0 | ±1 | ±2 | ±3 |
|---|---:|---:|---:|---:|
| window-integral area | 1.00 | 0.71–0.77 | 0.40–0.48 | 0.15–0.18 |
| peak amplitude | 1.00 | 0.16–0.19 | 0.06–0.08 | 0.03 |

**The central strip holds only ~27 % of the 7-strip integrated charge, but
~65 % of the peak-amplitude sum.** That is the sharpest measurement yet of
why aggregate (time/area) estimators fan out and compress — the
quantitative basis for `RECONSTRUCTION_BASIS.md` — and why peak-amplitude /
early-time weighting localizes.

Previous kernel numbers across the campaign (c1 0.23–0.35, τ_s 215–565 ns)
were each contaminated differently (ZS censoring in run_56/63, pile-up +
ch 510 in run_71's first pass); their spread was the systematics moving, not
the layer. The stable observables to carry forward:

* the ±1 peak-time shift — but note it is **estimator-dependent**: the
  event-wise median shift on clean events is **+54–61 ns** (stable across
  700/450/275 V; closer still to the bench's τ ≈ 47 ns), while the
  historical +29–36 ns was the peak of the *mean trace* under contaminated /
  censored processing. The clean *median trace* of ±1 peaks ~300 ns late
  because the dispersed hump dominates the average once tails are included.
  Quote the event-wise number,
* the **response library itself**: clean median W_d(t), d = 0…±3, three
  drift fields — archived in `reanalysis_2026-08-04/` npz/py,
* the **undershoot**: −4 to −6 % late negative on the central strip,
* c1/c2/τ **only via a physical 2-D RC-sheet model fitted to the library**
  (the `share_lp` structure is right; the |d|-cascade compression of it is
  what broke).

## 5. What this buys the fleet (det2/3/6/7 reconstruction)

1. **Validation of the wft basis choice.** Peak/early-time weighting (mf3)
   vs area/TOT: 65 % vs 27 % concentration, measured, at normal incidence,
   RAW. Use it as the standing justification.
2. **A measured single-track response library** for the Dream chain at
   64 × 60 ns: shaper rise (on50→peak ≈ 220 ns), undershoot amplitude, and
   the dispersed-copy shapes per |d|. `wft/model.py`'s `share_lp` branch can
   be fitted against these shapes directly instead of inferring the kernel
   through track fits. The kernel is per-detector, but the *functional form
   and the fitting recipe* transfer.
3. **Gain- and drift-invariance of the sharing** are now both demonstrated
   (run_56 590→625 V; run_71 450 vs 275 V clean). A bench kernel measured at
   one operating point is safe across that detector's conditions — this was
   the premise `wft` calibration bundles needed.
4. **Pile-up / pathological-channel hygiene**: leading-strip selections in
   beam data must mask oscillating channels and check the pre-window.
   ch 510 / ch 372 are det4-specific, but the *check* belongs in every
   beam-data chain (n_TOF included). Cosmic-bench data is immune (no beam
   pile-up), so June calibrations are unaffected.
5. **Undershoot in the model.** −5 % × µs-scale recovery will bias any
   late-window baseline or slow-tail fit that ignores it (it partially
   cancels the ±2/±3 dispersed tails on aggregate estimators).
6. **Drift-velocity caution transfers as a method, not a number**: v here is
   4× below dry Magboltz. For run_79 (n_TOF beam, 90/10) the same
   possibility must be entertained before trusting any v from tables —
   measure in situ (the ladder end-lobe trick works whenever a run at high
   drift field exists).

## 6. Corrections to earlier documents (originals left in place)

* `RAW_RUN71_PHYSICS.md` §3 "THE BLOCKER" and the t10/t50/t90 table:
  **superseded** — the numbers are artefact-dominated (this doc §1–2).
* `RAW_RUN71_PHYSICS.md` §3 "the dispersed tail is longer than 3.8 µs":
  **retracted for d=0**; survives only as the few-percent ±2/±3 surface tail.
* `RAW_RUN71_PHYSICS.md` §3 "the drift-invariance test does not survive":
  **reversed** — it passes on clean data.
* `RAW_RUN71_PHYSICS.md` §3 c1 → 0.32–0.35 in RAW ("retracts" the 0.23–0.28
  stability): that retraction was itself artefact-driven; c1 as a
  cascade-model β is not a robust observable in any of the passes. Use §4's
  library + charge budget instead.
* `README.md` "The window wall" section: **stands only for the 450/275 V
  drift points, and the cause is drift, not the resistive tail.** 128-sample
  windows remain a good idea for any future *low-drift-field* running.
* `datasets.py`: run_71 subruns now include groups 002–007 (700 V block,
  pulled from EOS 2026-08-04).
* The det4 ~0.4° tilt measurements used mean waveforms too; the sign of the
  effect is robust (three independent runs agree) but the magnitude should
  be redone with the clean selection before being carried as a constant.
