# PLAN 42 — Time resolution (paper topic 10)

*Status: DONE (2026-07-11). Originally written up in `PAPER_STATUS.md` as "❌ not
feasible". **That verdict was WRONG and is now corrected** (thanks to Dylan's
pointers + reading `mm_strip_reconstruction`): the bench DOES have an absolute
time reference and the fine-timestamp phase correction IS applied. Absolute AND
intrinsic time resolution are both measured. A full write-up with figures and
algorithm diagrams is in `report_time_resolution/main.pdf`.*

Script: `42_time_resolution.py`  ·  run from `mx_june_cosmic_qa/`
`../.venv/bin/python 42_time_resolution.py sat_det3 --veto=50` (+ `o22_long_det2`)

---

## The corrected picture — absolute timing IS available

Two facts, both verified against the reconstruction source and the data:

1. **Scintillator trigger = absolute t≈0.** Acquisition is triggered by a top+bottom
   scintillator-paddle coincidence (PMTs, ~5 ns each). The DREAM readout window
   opens on that trigger, so the muon crossing time IS the reference — there IS an
   external clock. (The DAQ records no scintillator *waveform*, only that a trigger
   fired, so the ~5 ns paddle term can't be measured from these data, but it's
   provably subdominant.)
2. **Fine-timestamp phase correction is applied.** The 60 ns DREAM sampling clock is
   free-running w.r.t. the trigger; the 3-bit `ftst` (0–7, 10 ns steps, 100 MHz)
   records the trigger-to-clock phase, and `WaveformAnalyzer.cpp:390-392` ADDS
   `ftst*10 ns` to every strip time. Verified in data: hit `time` is flat vs `ftst`
   (slope −0.2 ns/step); undoing it restores −10.2 ns/step. So per-strip `time` is an
   absolute leading-edge time (30 %-of-peak constant-fraction crossing, sub-sample
   interpolated) referenced to the phase-corrected trigger.

**The earlier "unknown common phase, unrecoverable" claim was the error** — the phase
is recorded in `ftst` and removed. What I first found (continuous `time`) was only
pulse-shape sub-sample interpolation; the `ftst` phase correction is the separate,
critical piece.

## Three measured numbers

- **A. Single-strip σ_t** — scatter of the micro-TPC strip times about the
  fitted line `time(position)` on inclined tracks (per-event, (N−2)-normalised).
- **B. Plane-to-plane DETECTOR σ_t** (headline; telescope- AND geometry-free). The X
  and Y strip layers sit under ONE drift gap and

- **A. Single-strip σ_t** — scatter of the micro-TPC strip times about the
  fitted line `time(position)` on inclined tracks (per-event, (N−2)-normalised).
  Folds in S/N, the sub-sample interpolation, and longitudinal diffusion.
  collect the SAME drifting electrons, so the shortest-drift (leading-edge)
  charge yields the SAME leading time in both layers. The two layers are two
  INDEPENDENT measurements of one event timestamp → σ(t_X−t_Y)/√2 is the
  single-plane DETECTOR resolution; the median of the difference is the residual
  inter-plane time-walk (a bias). Differencing cancels the trigger, the `ftst`
  phase, AND the drift geometry → isolates the detector. **No M3 telescope used.**
- **C. Absolute event-time σ68** — the earliest leading-edge time vs the
  (phase-corrected) trigger. Upper limit (still folds the event-to-event minimum
  drift geometry). Budget: detector (B) ⊕ scintillator (~5 ns) ⊕ ftst-quant
  (10/√12 = 2.9 ns) ⊕ geometry — detector-dominated. The ftst correction is
  demonstrated by comparing the absolute spread with/without it.

All converted to equivalent drift distance via σ_z = v_drift·σ_t
(v_drift = 34.30 µm/ns, sat_det3 anchor), linking timing to the spatial-resolution
story (topic 9).

## Method (as implemented)

`combined_hits` → keep detector FEUs → multiplicity veto (≤50 strips/event) →
map strip positions → keep hits with `time ∈ (0, 2000) ns` and `amplitude > 100`.
Per event & plane: take the largest spatial cluster (gap threshold 2 mm, ≥3
strips). `t_lead = min(time)`, `t_med = median(time)`. For A, fit `time(pos)` on
core strips (amp ≥ 30% of cluster max, ≥4 strips, position span ≥ 2 mm) and take
the (N−2)-normalised residual RMS. For B, difference the per-plane summaries over
events that have a cluster in **both** planes. The inter-plane leading-edge walk
is fit against the plane charge asymmetry (q_X−q_Y)/(q_X+q_Y); subtracting it
gives the walk-corrected (intrinsic-floor) resolution.

---

## Results

### det3 — `sat_det3` (490 V resist / 1000 V drift, FEU 7=X/8=Y), 28,404 dual-plane events

| quantity | value | equiv. drift dist |
|---|---|---|
| single-strip σ_t (line-fit residual) | **38.9 ns** (x 39.2 / y 38.6) | 1.33 mm |
| detector σ_t, leading edge (plane-to-plane /√2) | **33.1 ns** | 1.13 mm |
| detector σ_t, walk-corrected (intrinsic floor) | **29.0 ns** | 1.00 mm |
| inter-plane leading-edge bias | −1.3 ns (≈0) | — |
| inter-plane time-walk vs charge asymmetry | −105 ns / unit asym | — |
| per-strip time-walk (vs rel-amp) | +3 ns / rel-amp (negligible) | — |
| **absolute event-time σ68** (leading edge vs trigger, UL) | **37.7 ns** | 1.29 mm |
| absolute σ68 without ftst correction | 41.3 ns | — |
| ftst slope corrected / uncorrected | −0.2 / −10.2 ns/step | — |
| median-time estimator (cross-check, worse) | 70.3 ns single-plane | — |
| abs budget: det 33 ⊕ scint 5 ⊕ ftst-q 2.9 ⊕ geom 17 ns | detector-dominated | — |

### det2 — `o22_long_det2` (525 V, FEU 6=X/8=Y), 23,894 dual-plane events — replication

| quantity | value | equiv. drift dist |
|---|---|---|
| single-strip σ_t | 40.5 ns | 1.39 mm |
| detector σ_t, leading edge | 36.8 ns | 1.26 mm |
| detector σ_t, walk-corrected | 34.0 ns | 1.17 mm |
| inter-plane bias | −2.2 ns (≈0) | — |
| walk vs charge asymmetry | −96 ns / unit asym | — |
| absolute event-time σ68 (UL) | 37.6 ns | 1.29 mm |
| ftst slope corrected / uncorrected | +0.4 / −9.6 ns/step | — |

Two independent detectors, different HV and FEU pairs, agree to ~10 % on every
number → this is a design/gas property, not a per-chamber accident.

### Reading for the paper

- The two strip layers timestamp the same cosmic muon to **≈30 ns**, i.e. **≈1 mm
  of drift** — matching the transverse spatial resolution (topic 9, 0.6–0.8 mm).
  The detector's longitudinal (drift-time) precision is consistent with its
  transverse precision; the micro-TPC is not timing-limited relative to its
  geometry.
- The inter-plane bias is consistent with zero (−1 to −2 ns), directly confirming
  the "one drift gap, two orthogonal readouts of the same arrival times" picture
  used throughout the tracking analysis.
- **Absolute timing IS available** and the ftst phase correction works (proven in
  data). The absolute event-time resolution is 37.7 ns (UL), detector-dominated;
  the honest caveat is only that the ~5 ns scintillator term is external to the
  DAQ data (no PMT waveform recorded) so it can't be measured here — but it's
  provably subdominant. **Do not repeat the old "not feasible" sentence.**
- **Time-walk & unsharing (answers the design question):** per-strip time-walk is
  *negligible* (+3 ns/rel-amp) because reconstruction times on a 30 % constant
  fraction — so a per-strip walk calibration would barely help the TPC slope, and
  **unsharing (RC deconvolution) is the dominant timing correction**, not walk.
  The residual *inter-plane* walk (~−100 ns/asym) is a small S/N effect (brighter
  plane crosses slightly earlier); removing it takes the detector resolution 33→29.
  Correct ordering if ever calibrating walk: unshare first (changes amplitudes),
  time each cleaned strip, then walk-correct on the *unshared* amplitude.
- The inter-plane bias is consistent with zero (−1 to −2 ns), directly confirming
  the "one drift gap, two orthogonal readouts of the same arrival times" picture.
- The leading edge is the sharpest estimator (33 ns) vs the median strip time
  (70 ns) — as a TOF device you'd time on the first electrons.
- σ_t grows steeply with drift depth (28 ns at ~1 mm → 150+ ns for tracks whose
  earliest charge is already deep) from longitudinal diffusion plus attachment
  S/N loss (λ≈15 mm) — the same attachment that sets the drift-gap story (topic 7).

## Outputs

Per run/det: `<run>/<subrun>/<det>/alignment_tpc_veto50/time_resolution/`
- `time_resolution.png` (overview), `figs/*.png` (report figures: algorithm +
  geometry schematics, ftst correction, single-strip, plane-to-plane, drift depth)
- `time_resolution.csv`, `time_resolution.json` (all numbers)

Report: `mx_june_cosmic_qa/report_time_resolution/main.pdf` (7 pp, LaTeX source +
figs; explains the timing chain/algorithm, method, results, det2 replication).

## Caveats / where a referee pushes

- σ_t (single-strip) is an *upper bound* on the electronic timing — it includes
  diffusion and any track non-straightness. Framed as the effective per-strip
  timing used in tracking, that's the right number.
- The plane-to-plane number assumes the two layers' leading strips see the same
  physical first-arrival. Charge sharing only *delays* neighbours, so the min-time
  estimator is the sharing-robust choice; the ≈0 bias is the evidence it works.
- The absolute number (C) is an upper limit because it still folds the event-to-event
  minimum-drift geometry (~17 ns term). A fully model-independent absolute number
  would need a per-event drift prediction (e.g. M3) — deliberately avoided to keep
  the measurement self-contained. The clean detector figure of merit is (B).
- Reconstruction refs: `mm_strip_reconstruction/waveform_analysis/src/WaveformAnalyzer.cpp`
  timing lines 848–867, ftst application lines 390–392.
