# Detector 3 weekend deep-dive — micro-TPC angle correlation & drift velocity

*2026-07-01 analysis of the 6-27/6-28 weekend det3 runs, revised 7-03/7-04
(drift gap & gas). Scripts: `13_tpc_angle_bias.py` … `19_amplitude_attachment_plot.py`;
run key `sat_det3`.*

> **Full LaTeX write-up: `report_det3_weekend/main.pdf`** (compile:
> `pdflatex main.tex` in that dir; figures collected by `mk_diagrams.py` + copies).
>
> **Drift gap (REVISED 7-03):** the gap is **30 mm mechanical** (user was
> right); the 19.4 ± 0.2 mm from v·T_sat is the **visible drift column** —
> the depth at which drifting charge falls below threshold. Proof
> (`17_gap_attachment_test.py`): (a) the strip arrival-time distribution has
> an exponential tail *beyond* T_sat terminating at ~28–29 mm (the 30 mm
> drift time) at every HV where the window allows; (b) the median strip
> amplitude vs drift **depth** collapses onto one universal decay curve
> (λ ≈ 13–15 mm) for all fields — signal loss depends on distance drifted =
> **electron attachment**, not geometry. The earlier "hard cliff" reading
> was a threshold crossing. Field scale is therefore **E = HV/3.0 cm**.

## Datasets

`mx17_det3_saturday_scan_6-27-26` (det3 in TOP slot, FEU 7=X / 8=Y, z=702 nominal):

| subrun | what | decoded stats |
|---|---|---|
| `long_run_resist_490V_drift_1000V` | operating point; configured 720 min, **stopped after 141 min** (p2 overnight started 01:33) | complete: 4 file-pairs, 29k det events, 47k M3 tracks |
| `drift_scan_resist_490V_drift_<100..1100>V` | 6 pts × 15 min, resist 490 V | ~5k M3 tracks / pt |
| `hv_scan_resist_<425..525>V` + `hv_scan2_<460..520>V` | resist scans, drift 1000 V | ~2k clean rays / pt |

plus `mx17_det3_p2_det1_overnight_6-27-26 / long_run_p2_det1_sanity_check`
(`g_det3_wknd`, 53k rays) — the current headline det3 in the June PDF.

Both weekend runs align to the **same geometry**: z = 714 mm, θ = 89.45°,
offsets agree to 10 µm — the detector did not move; the long-run alignment is
valid for all scan subruns (translation refit per point shifts < 0.1 mm).

Headline numbers (Saturday long run, 4 decoded files): efficiency ≈ 80 %
(matches g_det3_wknd 79.7 %), core resolution σ_x = 0.83 mm, σ_y = 0.92 mm.
Resist scans: turn-on at 425 V, plateau 455–490 V at 80–83 % (best 83.2 % at
470 V), roll-off > 505 V driven by sparking (spark fraction 4 % → 56 % from
470 → 525 V). 490 V operating point is on-plateau.

## 1. Why the angle correlation is ALWAYS off-diagonal (all detectors)

**Observation.** In `angle_correlation_corrected.png` the dense ridge runs
*parallel* to the y = x diagonal but displaced outward — |θ_det| exceeds
|θ_ref| by ~5° — in both planes, both signs, and identically in det2, det3,
det6, det7. There are essentially no events with |θ_det| ≲ 5°, and events with
θ_ref ≈ 0 form horizontal arms at large θ_det.

**It is NOT a rotation/alignment effect**: a residual in-plane rotation is
already absorbed by the alignment (θ scan converges to 0.05°); an out-of-plane
tilt would displace the correlation in ONE direction, whereas the observed
offset flips sign with the track direction (outward in both quadrants).

**Mechanism (verified quantitatively on the 2.4 h run, `13_tpc_angle_bias.py`):**

Every cosmic crosses the full drift gap, so the time span of the recorded
strip cluster is a constant ~T_sat = z_vis/v_drift regardless of angle
(measured: median span saturates at ≈ 690 ns for |θ| > 10° at 1000 V; z_vis
= 19.4 mm is the visible column, see gap note). The *spatial* width of the
cluster, however, is NOT purely geometric (z_vis·tanθ): it has a
**non-geometric floor of w ≈ 2 mm** (≈ 3 strips at 0.78 mm pitch) from
resistive-layer charge spreading + transverse diffusion + capacitive coupling
— measured directly: median cluster extent at θ_ref ≈ 0 is 3–5 mm where
geometry predicts ~0. The micro-TPC fit slope is therefore inflated:

    dx/dt ≈ (z_vis·tanθ + w) / T_sat  =  v·tanθ ± v·(w/z_vis)

    →  tanθ_det ≈ tanθ_ref ± w/z_vis        (sign follows the track sign)

An **additive, angle-independent excess in tan-space**. Measured
w/z_vis = 0.115 (constant for |θ_ref| ≳ 6°, and constant vs drift HV ≥ 500 V),
i.e. arctan(0.115) ≈ 6.5° — precisely the observed ridge displacement, the
|θ_det| floor, and (with the min-strips ≥ 4 selection at small angles) the
horizontal arms. Wider clusters carry a larger excess, as expected.
Overlaying `tanθ_det = tanθ_ref ± w/z_vis` on the density plot tracks the
ridge in both planes (`bias_study/correlation_with_prediction.png`).

**Consequences**
- The off-diagonal offset is a *detector-physics constant* (spreading width /
  visible column), not an analysis bug — universal across detectors because
  they share the same resistive design and the same (contaminated) gas supply.
- The production diagonal-projection σ-scan v_drift (30.5–33.5 µm/ns) is
  biased HIGH by this offset. Don't use it for physics.
- A first-order angle correction exists: subtract sign(θ)·w/z_vis in
  tan-space. Note z_vis will grow toward 30 mm when the gas is cleaned —
  the correction constant is gas-quality-dependent.

## 2. Drift velocity measurement (`14_drift_velocity_scan.py`)

**Bias-free estimator**: per plane and per track-sign, straight-line fit of
the rotated strip-fit slope S [µm/ns] vs tanθ_ref over 0.06 < |tanθ_ref| < 0.55:

    S = v_drift · tanθ_ref ± v_drift · (w/z_vis)

The SLOPE is v_drift — the additive spreading term goes entirely into the
intercept, and the fitted slope is a *local* dx/dt, unaffected by the
attachment truncation. Four independent fits (X/Y × ±) are combined; their
spread is the systematic. M3 angle resolution (~0.05°) is negligible.

**Result (resist 490 V, nominal Ar/iso 95/5, ~745.8 Torr, E = HV/3 cm):**

| drift HV | E [V/cm] | v_ridge [µm/ns] | T_sat [ns] | note |
|---:|---:|---:|---:|---|
| 100 | 33 | — | — | invalid: drift time ≫ DREAM window |
| 300 | 100 | 5.70 ± 0.54 | (1198) | window truncation marginal |
| 500 | 167 | 12.77 ± 0.43 | (1361) | T_sat truncated, ridge OK |
| 700 | 233 | 19.45 ± 0.53 | 992 | |
| 900 | 300 | 26.17 ± 0.84 | 754 | |
| **1000 (2.4 h)** | **333** | **28.12 ± 0.65** | 690 | long run, on-curve |
| 1100 | 367 | 29.44 ± 0.47 | 661 | |

- **15-minute scan points give 1–3 % statistical precision** — statistics are
  NOT a limitation; the ridge fit is far more efficient than the σ-scan.
- v·T_sat = 19.4 ± 0.2 mm at every valid HV = the **visible column** (fixed
  depth ⇒ product constant); clock-invariant (v ∝ 1/k, T ∝ k).
- Absolute-scale systematic ~5 % (fit-window choice); the *shape* v(E) is
  much tighter.
- Efficiency vs drift HV: 44 % @100 V → 83 % @700 V plateau → ~81 % @1100 V.
  Low-drift loss = charge outside the readout window (+ stronger attachment).

## 3. The drift gap: 30 mm mechanical, 19.4 mm visible (`17_gap_attachment_test.py`)

Per drift point, strip arrival times are converted to depth z = v_ridge·t
(figure `gap_attachment_test.png`, table `.csv`, under
`Analysis/<run>/drift_velocity/mx17_3/`):

| drift HV | v [µm/ns] | z(T_sat) [mm] | tail endpoint [mm] | λ_amp mid [mm] |
|---:|---:|---:|---:|---:|
| 500 | 12.8 | 17.4 | (19.9, window-limited) | 26.8 |
| 700 | 19.5 | 19.3 | (25.7, window-limited) | 14.8 |
| 900 | 26.2 | 19.7 | 25.1 | 14.1 |
| 1000 | 28.1 | 19.4 | 28.7 | 12.9 |
| 1100 | 29.4 | 19.5 | 28.3 | 14.0 |

- Arrival-depth distributions collapse across HV: flat plateau to 19.4 mm,
  then a common exponential shoulder terminating at ~28–29 mm ≈ the 30 mm
  mechanical gap (endpoint is threshold-soft, so slightly below 30).
- Amplitude vs depth is ONE curve for all fields, decaying ×~10 from mid-gap
  to 30 mm: loss depends on drift *distance* ⇒ attachment (+ some diffusion
  dilution per strip). Money plot: `amplitude_attachment.png`
  (`19_amplitude_attachment_plot.py`) — vs TIME the decay timescale changes
  with HV (not electronics); vs DISTANCE all curves collapse onto
  exp(−z/λ), λ ≈ 10 mm (8–24 mm window) / 13–15 mm (in-column), matching
  Magboltz 1–2 % air.
- The 690 ns "cliff" at 1000 V is the occupancy crossing the hit threshold —
  the earlier geometric-edge reading was wrong.

## 4. Gas: air/moisture contamination (velocity + attachment + gain closure)

With E = HV/3.0 cm (`15_drift_velocity_vs_magboltz.py --gap=3.0`, RMS vs
measured):

| hypothesis | RMS [µm/ns] | verdict |
|---|---:|---|
| Ar/CO2 90/10 (wrong bottle) | 1.8 | best v(E) — **excluded** by gain & attachment |
| **Ar/iso 95/5 + 1 % H2O** | **4.4** | consistent; best fit ≈ 1–1.5 % H2O |
| Ar/iso 95/5 + 2 % H2O | 8.5 | too slow |
| + 2.5–3 % H2O | 12–14 | excluded (old best fit under 19.4 mm scale) |
| + 1–2 % dry air | 17–19 | no v suppression — air alone can't do it |
| clean nominal / iso-rich | 12–20 | excluded |

Discriminators:
- **Gain** (`mm_townsend_compare.py`, 150 µm gap): Ar/CO2 90/10 needs ~+50 V
  at iso-gain (gain 100: 528 V vs 475 V). Observed turn-on 425 V/plateau
  455 V matches Ar/iso ⇒ wrong bottle excluded.
- **Attachment** (`mm_attachment_{table,air_candidates}.py`, ncoll=5):
  Magboltz η = 0 for clean Ar/CO2 AND for Ar/iso + H2O only (water doesn't
  attach at drift energies). O2 does: 1 % dry air (0.21 % O2) → λ = 16–21 mm,
  2 % air (0.42 % O2) → λ ≈ 10 mm at 230–370 V/cm. Measured λ ≈ 13–15 mm ⇒
  **~0.2–0.4 % O2** (`18_attachment_vs_magboltz.py`,
  `attachment_vs_magboltz.png`).

**Diagnosis: air + moisture in the drift gas** — ~1–1.5 % H2O (40–70 % RH
equivalent; slows the electrons) + ~0.2–0.4 % O2 (~1–2 % air-equivalent;
eats the far third of the drift column). H2O ≫ air-implied share is normal:
water permeates/outgasses much faster than O2 enters. Amplification is
insensitive at these levels — consistent with clean-gas gain behaviour.

**Decisive test: fresh-gas re-scan after a long flush** — v(1000 V) should
rise 28 → ~39 µm/ns AND the visible column should extend 19.4 → ~30 mm with
the amplitude decay flattening. Plus humidity + O2 probes on the exhaust.

Output: `Analysis/mx17_det3_saturday_scan_6-27-26/drift_velocity/mx17_3/`
(`drift_velocity_scan.png/.csv`, `gap_attachment_test.png/.csv`,
`drift_velocity_vs_magboltz.png`, `attachment_vs_magboltz.png`); bias study
under `.../long_run_resist_490V_drift_1000V/mx17_3/alignment_tpc_veto50/bias_study/`.
Magboltz tables in `garfield_sim/results/` (attachment_*.json,
townsend_compare.json).

## Open follow-ups
1. ~~Decode the remaining file-pairs of the Saturday long run~~ **CORRECTED
   7-02: the run was stopped after 141 min** — the 4 file-pairs are the
   complete dataset. For more det3 stats use the p2 run (53k rays).
2. Gas system: long flush + re-scan (one hour decides), humidity/O2 probe,
   flow & leak check. This is now the top action item — it costs velocity,
   ~1/3 of the signal charge, and micro-TPC quality.
3. Mechanics: confirm 30 mm on the drawing (data endpoint: 28–29 mm);
   confirm DREAM 60 ns sampling period.
4. Angle-corrected micro-TPC estimator (subtract w/z_vis term; better
   per-strip timing) → real angular-resolution number for the TDR. Note the
   correction constant depends on gas quality via z_vis.
5. The 100/300 V drift points need a longer readout window to be usable.
6. Wire the ridge-fit v_drift into `03_alignment_and_tpc.py` in place of (or
   alongside) the σ-scan.
