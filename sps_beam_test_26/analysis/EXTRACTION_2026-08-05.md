# Squeezing the rest of the SPS test — 2026-08-05 extraction pass

**Status: in progress — numbers land as the pipelines finish; sections marked
(pending) fill in below.**

The 2026-08-05 sweep through everything the campaign had not yet extracted:
the `share_lp` port into the fleet reconstruction, the gas-flush transient
(run_60), the last CO₂ dataset (run_59), the 25.64° drift-ladder forward fit
(run_63 rot25), the unanalyzed run_61 sub-runs, and the flat-data statistics
doubling (operating_02 tail). Context documents: `GAS_FLUSH_TIMELINE.md`
(new), `BEAM_CONSTRAINED_CALIB_2026-08-05.md` (../..../mx_june_wft/).

## 1. share_lp is in the fleet model (`wft/model.py`)

The RC-dispersed sharing kernel the beam measured is now a first-class
citizen of the forward model: `share_mode: 'lp'` on the CalibrationBundle
replaces the delayed copy with the impulse response convolved with a one-pole
RC of constant `tau_s` (cascaded for ±2), area-preserving so `c1`/`c2` keep
their meaning. Delay mode is bit-compatible with the old behaviour;
`wft/tests/test_share_modes.py` (new, self-contained — the old regression
test's R&D fixtures died with the campaign machine) checks both branches and
round-trips fits in each.

**Bench recalibration arms** (June det4 cosmics, reprocessed hits, all with
c1/c2 pinned to the beam values 0.25/0.10 and v = 34):

| arm | mode | pinned | fitted | chi2 (train) | σ_θ X/Y | verdict |
|---|---|---|---|---|---|---|
| canonical | delay | c1,c2,τ=60,σ_p0,v | kY,σ_s,Dp | 172.0e6 | 2.63°/2.44° | previous best |
| A | lp | c1,c2,σ_p0,v | **τ**,kY,σ_s,Dp | **171.4e6** | **2.60°/2.46°** | equal-best; τ_RC = 108 ns |
| B | lp | c1=0.53,c2=0.43,τ=800 (RAW full-window), σ_p0,v | kY,σ_s,Dp | 198.4e6 | — | **rejected**: full-window RAW betas are not transplantable amplitudes (kY collapsed to 0.59) |
| C | lp | c1,c2,v | τ,kY,σ_s,Dp,**σ_p0** | **170.1e6** | **2.53°/2.22°** | **NEW BEST** — see below |

Arm A's τ_RC = 108 ns is the within-window RC constant, and it is consistent
with the measured bench ±1 peak shift (+60 ns): an RC rise of ~100 ns peaks
the copy ~60–80 ns after the central strip. The 300–800 ns constants from
the beam fits are *representation-dependent* (they include the far tail that
the bench window never sees); transplanting them naively is exactly what arm
B disproves.

**Conclusion:** the share_lp port delivers the predicted structural gain —
just not where expected. With the kernel shape correct, σ_p0 no longer needs
to be pinned: arm C (τ_RC = 141 ns, σ_p0 = 0.404 mm free, kY 2.10) beats
every delay-mode configuration on BOTH views —

| | σ_θ X | σ_θ Y | s68 X/Y | pos res X/Y | bias | implied-v spread |
|---|---|---|---|---|---|---|
| lost legacy (delay) | 2.73° | 2.58° | — | — | — | — |
| canonical delay (yesterday's best) | 2.63° | 2.44° | 3.17/— | 0.65/0.89 | −0.20/−0.25° | 4.1/4.5 |
| **lp, σ_p0 free (arm C)** | **2.53°** | **2.22°** | **2.92/2.78** | 0.64/0.89 | −0.16/−0.29° | 4.0/4.4 |

Under the *delay* kernel a free σ_p0 was a pathology (it absorbed the wrong
copy shape and reconstruction degraded to 4.1°/3.4°); under the *lp* kernel
the same freedom improves every angle metric at unchanged position
resolution. Note σ_p0 = 0.40 mm should now be read as the genuine effective
width of (initial cloud ⊕ whatever short-range spread the single-τ RC does
not carry), not necessarily a literal primary-cloud size — det3's 0.098 mm
came from the delay parameterisation and is not directly comparable.
**Arm C is PROMOTED to the canonical det4 bundle** (same evening): the old
delay-mode products are archived beside it
(`calib_bundle_delay60_archived20260805`, `events_delay60.parquet`,
`alignment_delay60/`, `angles_delay60/`, `efficiency_delay60/`).
Full-chain validation on the promoted bundle: within-5 mm efficiency
41.93 % (delay: 41.96 % — parity), core |r| 0.678 mm (0.700), σ_θ
2.53°/2.22° (2.63°/2.44°), digest updated. The reproduce line becomes:

```bash
python -m wft.calibrate g_det4 --jobs 12 --share-mode lp \
    --fix-hyper "c1=0.25,c2=0.10" --fix-v 34.0
```

## 2. The run_71 library in the wft representation

`mx_june_wft/06_share_lp_library_fit.py` (new): the clean RAW library's
central trace is reconstructed with the bench impulse template (NNLS charge
ladder — the same algebra as a wft track fit), neighbours fitted as
`alpha·W0 + beta·RC^|d|(W0)`:

| view | tau [ns] | c1 | c2 | drift-invariance |
|---|---|---|---|---|
| Y | 716 / 853 / 774 (275/450/700 V) | 0.51–0.56 | 0.41–0.47 | τ ±9 %, c1 ±5 % |
| X | 381 / 597 / 679 | **0.240–0.249** | 0.02–0.04 | c1 ±1.7 %; τ ±27 % (tilt-contaminated, as expected) |

Two closures fall out:

- **kY ≈ 2.2 from the beam itself.** The library's Y/X amplitude ratio
  (0.53/0.245) reproduces the bench-fitted kY = 2.1–2.2 — independently
  confirming that the ZS-era beam kY = 1.12 was tilt-biased, exactly as
  `BEAM_CONSTRAINED_CALIB` §2.3 suspected.
- **The X-view ±1 asymmetry is quantified — and it is NOT the tilt.**
  Measured: β(−1)/β(+1) asymmetry −0.25 → −0.49 growing with drift field
  (Y view: +0.01–0.03). Forward-modelling the measured mount walk
  (tan θ_X = −0.0157 at wet v = 14 µm/ns) through the lp model predicts a
  ±1 area asymmetry of only **0.025** — an order of magnitude short. The
  sharing asymmetry is therefore **detector-internal** (the X strips'
  coupling under the resistive stack), distinct from the drift-invariant
  centroid walk (which stays geometric). Three consequences: quote-Y-only
  is structural, not situational; the walk and the sharing asymmetry must
  stop being conflated in the record; and det3's same-sign X asymmetry
  (BEAM_CONSTRAINED §4) says this is a design/stack property, not a det4
  defect.

## 3. The gas story (see GAS_FLUSH_TIMELINE.md for the model)

Flow was ~2 ln/h into ~4.6 L → mixture exchange τ ≈ 2.3–4.6 h, and the
percent-level water floor is a *flow* equilibrium that flushing never
removes. TAX-dating the accesses closed two logbook questions on the way
(run_59's truncation = the 20:24:07 gas access; run_60's `overnight_15`
collapse = SPS FTARGET outage 04:50–08:30, spill record).

**run_60 IS the flush measurement — and it is now measured**
(`flush_run60.py`, 13 beam-on sub-runs at fixed HV drift 700.5 / resist
649.75 V, anchored by run_59's CO₂ span and run_63's fully-exchanged span):

| | measured |
|---|---|
| transport lag (line volume at 2 ln/h) | **1.72 ± 0.23 h** ≈ 3.4 L of line upstream |
| exchange constant τ | **3.49 ± 0.57 h** = 1.5 × ideal V/Q (4.6 L / 2 l/h) |
| 95 % exchanged | +12 h after the switch |
| v_drift step | span 2340 → 1996 ns → **v(CF₄-mix)/v(CO₂-mix) = 1.17** at 243 V/cm |
| gain step at fixed HV | −19 % (Y), −18 % (X) — the CF₄ quencher load |
| hits/event | flat (null check) |

Watch-outs baked into the script: a plain exponential on a lagged transient
fits 12–15 h taus (wrong); the gain estimator is not comparable across runs
(each run's ZS σ differs) so only the span crosses run boundaries; sub-runs
06–08's statistics dip is the 00:30–01:30 SPS spill dip, and run_59's HV
was resist 669.8 V (not 649.75) — both verified against the records.
Consequence: residual old-mix was ~2 % by run_61's 15° half and negligible
from the 25° half on — **the kernel gas-transfer claims stand**, and the
run_61 15-vs-25° efficiency gap keeps only a ≲2 % gas term.

## 4. The 25.64° drift ladder — the σ_p0/Dp lever

Two routes, both new:

**Estimator route (`ladder_span.py`)** — the amp≥150 hit-time span per HV
plateau, anchored to the run_71 end-lobe (233 V/cm : 14 µm/ns):

| field [V/cm] | 235 | 217 | 200 | 182 | 165 | 148 | 113 |
|---|---|---|---|---|---|---|---|
| v (CF₄-mix) [µm/ns] | 14.0 | 13.2 | 12.4 | 11.8 | (11.6) | (11.6) | (11.5) |

The bracketed low-field points are window-truncated (t90 rails at the
3.84 µs edge) — lower bounds on the span, upper bounds on nothing. The
first four points are the wet-CF₄ v(E) curve. run_55 (flat CO₂ ladder)
shows a **non-monotonic** span (2188 → 2564 → 2338 → 2363 ns over 243→139
V/cm) — the Ar/CO₂ mobility-peak shape — but its plateau windows are still
approximate (refine against `detE_scan.log` before quoting).

**Forward route (`wft_beam_fit.py`)** — ref-pinned wft fit (kernel pinned
lp, ZS-censored samples, warm-started t0, span-bounded v) on the rot25
ladder. Status: machinery works (sign auto-detection is decisive, v lands
near the span values, e.g. 10.7 µm/ns at 113 V/cm where the span estimator
is truncation-biased high), BUT within a single plateau σ_p0 and Dp are
degenerate (σ_p0 0.05/Dp 0.063 and σ_p0 0.57/Dp 0.067 fit equally well),
and chi2/dof ~125 says the noise model and the thin rot25 alignment
(det(A) 1.07 vs the true 1.11; the 454k-track run61_op25 alignment does
NOT transplant — ZS centroids are condition-dependent) are the current
floor. **The σ_p0/Dp separation needs the joint fit with σ_p0 shared
across plateaus** — the machinery is committed; that global fit is the
identified next step, now cheap.

## 5. New datasets brought into the record

- **run_59 `detE_long_00`** — decoded (19 groups). The last CO₂ dataset
  (22 min of beam, 64 samples, 3σ — confirmed by rate arithmetic: 13 MB/s
  sits between the measured 2σ and 5σ rates). Its HV was **resist 669.8 V**,
  not run_60's 649.75 — checked from its own `hv_monitor.csv`, which is why
  its gain is not the flush anchor (its span is; §3).
- **run_63 `operating_02` tail** — staged, decoded, paired:
  `wf_run63_flat.npz` now holds **42.6k events** (was 27k). The kernel
  refit reproduces the documented numbers within errors (Y: c1 0.239 vs
  0.233, τ 213 vs 215 ns, α 0.206 vs 0.209; ±1 sides agree to 2 %), and the
  tilt walk repeats (X −0.238 µm/ns, Y −0.027). The flat-CF₄ kernel
  measurement is now statistics-stable — closed.
- **run_61 m70V–m100V** — decoded. NOT scan points: hv_monitor shows
  drift 700.2 / resist 769.8 V held for ~80 min → this is the
  high-statistics **operating-point block at 25.64°** (same HV as
  run_63/71), the 4th ladder plateau. m100V is a 67 % point (SPS linac,
  `M100V_PARTIAL.md`).
- **run_61 m20V/m30V** — staged (the 15° drift-scan sub-runs; m30V holds
  TWO same-named passes, 13H46 = 15° and 16H08 = 25° — distinguish by the
  datrun stamp).
- **run_55** — staged (8.4 GB). The *flat* CO₂ drift scan (700→400 V),
  previously believed not to exist ("no drift lever in the flat data" —
  true only of the CF₄ era).

## 6. Data/infrastructure state

- The TAX + spill CSVs are staged locally under
  `records/beam/backfill_nxcals/` (pulled from mx17-daq 2026-08-05).
- `flat_align_eff.py` now corrects RAW waveforms (pedestal + signal-masked
  block CM) instead of refusing — closes RAW_RUN71_PHYSICS §3b's remaining
  cross-check item.
- `decode_dataset.py`/`pair_dataset.py` handle the EOS directory layout
  (`raw_daq_data/`, `combined_hits_root/`) and per-sub-run fdf stems.
- **Saturday CO₂ 25° ladder (run_57 gap350V/400V + run_58 operating_00–02,
  ~100 GB) is NOT pulled** — needs a fresh Kerberos ticket and ~100 GB of
  disk; the CF₄ ladder (§4) covers the same physics. Listed as optional
  follow-up.
