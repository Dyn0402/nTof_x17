# run_55 micro-TPC tracking — stringent true-track selection + source-hypothesis alignment (2026-07-19)

**Goal: take the run_55 data and (1) build a STRINGENT selection that keeps only
realistic tracks — a particle that crossed the full 30 mm drift gap, matched in
X and Y — throwing out the ³He-capture blobs / gamma-flash junk / coherent
ringing that "any x/y cluster = track" was picking up; then (2) use the
source hypothesis (all tracks come from the He-3 target) to check and improve
the reconstruction, alignment and drift calibration.**

Data: the 24 surviving run_55 subruns (`~/x17/beam_july/runs/run_55/`, resist
scan r560→r520, drift 800 B/C/D + 600 A, Ar/iso 90/10, scint-doubles trigger,
30 ms gate, ³He target — same runs as `HV_SCAN_RUN55_ANALYSIS.md`). Both
`combined_hits_root` (per-strip hits) and `decoded_root` (32-sample × 60 ns
waveforms, no ZS) were used. Physics is read only in the two sampled/recovered
in-gate windows **b1 (8–12 ms)** and **b2 (16–28 ms)** — the 3 ms target and
0–8 ms are unsampled/dead (the DAQ comb, see the HV-scan doc).

Scripts (all `venv/bin/python mx_july_beam_qa/…`):
- `27_track_extract.py` — per-strip candidate-cluster cache → `cache/27_run55/`
- `trackcache.py` — loader + derived quantities + the realism **GATE**
- `27b_track_gate.py` — gate + validation → `figures/27_tracks/01-03`, `calib/27_tracks.npz`
- `27c_source_align.py` — source calibration/alignment/distortion → `figures/27_tracks/04-06`, `calib/27_align.json`
- `27w_headon_waveforms.py` — decoded-waveform head-on purification → `figures/27_tracks/07-08`, `calib/27_headon_wf.json`

## The micro-TPC picture (what a real track looks like)

Each chamber is a single micro-TPC: strips read (u, v) = (in-plane transverse,
beam axis), electrons drift the 30 mm gap along the radial normal w. A particle
from the source deposits along its whole path; the charge from drift-depth w
arrives at time t(w). Two limiting signatures, BOTH spanning the full gap:

- **inclined track** → a MONOTONIC micro-TPC trail: strip position vs peak
  drift-sample is an ordered line; the peak-time span (`dur` = smphi−smplo)
  covers a good fraction of the ~11–12-sample (60 ns) full-gap crossing
  (garfield v: A 40.5, B/C/D 44.1 µm/ns ⇒ T_gap 680–740 ns).
- **head-on track** (normal incidence, ~radial) → one strip, a LONG single
  waveform hump spanning the full-gap drift time (the cosmics "hybrid" case).

**Key discriminator geometry.** For a full-gap track the drift-time span is
angle-INDEPENDENT (= gap/v), while the position spread is gap·tanθ, so the
anchored time-fit slope `S = dt/du = 1/(v·tanθ)` and the reconstructed angle is
`tanθ = 1000/(S·v)` — NOT `S·v` (that is the cotangent; getting this backwards
is the easy mistake). Source tracks are near-NORMAL incidence (small θ, small
position extent, steep dt/du).

## The realism gate (`trackcache.GATE`)

Per chamber, per plane, on gap-clustered (12 mm) hits above 400 ADC:

| requirement | inclined | head-on |
|---|---|---|
| strips n | 4–20 | ≤3 |
| position extent | ≤25 mm (compact) | — |
| peak-time trail span `dur` | ≥6 samples | ≤3 (peaks bunched) |
| trail monotonicity \|corr(pos,sample)\| | ≥0.8 | — |
| single-strip pulse width `wfmax` | — | ≥9 samples, envelope ≥9 |

then **X↔Y consistency**: the x and y track-like clusters of an event must
overlap in absolute drift-time window (IoU ≥ 0.3), match in time-length
(\|Δt_occ\| ≤ 6) and balance charge (\|f−0.49\|/0.10 ≤ 3).

**Critical lesson (cost me a wrong first pass):** use the PEAK-TIME span (`dur`,
from `max_sample`) for "full-gap", NOT the pulse-ENVELOPE occupancy
`t_occ = max(right_sample)−min(left_sample)`. At high gain (C/D) single pulses
get wide, so `t_occ` is large even for PROMPT (flat-trail, tanθ≈0) blobs — using
it admitted a big tan≈0 contamination and broke the source-pointing for B/C/D.
`dur` + monotonicity is the honest full-gap-trail measure.

### Gate validation — the tracks are real

- **Survival cascade (b1/b2, all subruns).** Candidates → compact → full-gap →
  track-like: **A** 14206→8823→698→1787 (562 inc, 1225 head-on); **B**
  15212→6895→1283→1376 (648, 728); **C** 12292→5746→1582→2285 (1137, 1148);
  **D** 59268→28281→12613→628 (354, 274). D's 59 k candidates (the capture
  flood) collapse to 354 inclined — the gate does exactly the contamination
  rejection we wanted. D's compact clusters peak at monotonicity **0** (isotropic
  capture blobs), A/C peak at **±1** (real trails).
- **762 clean X/Y 3-D segments** (A 308, B 153, C 287, D 14).
- **Charge balance = the independent real-track proof.** Matched pairs sit at
  the June BENCH charge-balance value f = qX/(qX+qY) ≈ 0.48–0.50 and are much
  TIGHTER than the event-shuffled accidental null: A 0.496±0.081 vs null
  0.502±0.147; B/C 0.48±0.11 vs ±0.15; D 0.45±0.13 vs ±0.16. Random combinatorics
  cannot reproduce the bench value with that width.
- Track-like multiplicity is **1.04 per plane** (90 % of two-plane events are a
  unique 1×1 pairing) — the pairing is essentially unambiguous. (The raw match
  COUNT is a bad purity metric: the trigger fixes t0 so every track shares the
  same absolute drift window and a shuffled Y always finds an overlapping
  partner — the discriminant is the charge-balance WIDTH, not the count.)
- Clean-track yield rises with resist HV for A/B/C (gain turn-on; A,C reach
  ~40–45 in b1/b2 across all subruns at 555–560 V), flat and low for D — the
  same story as the 25b efficiency proxy, now on VERIFIED tracks.

## Source hypothesis → alignment, drift calibration, distortion

Using the clean **inclined** single-plane clusters (the workhorse; head-on carry
tanθ≈0 and don't constrain the slope). The source model per plane is
`tanθ = (u − u0)/R`, R ≈ 234 mm. Fit tanθ_meas vs strip position u in a
fiducial window (70–330 mm, edges cut):

| plane | u0 (mm) | mech. exp. | **align resid** | s·(−R) [1=ideal] | v_true impl. | σθ (raw) |
|---|---|---|---|---|---|---|
| Ax | 190 | 183 | **+7.5 mm** | 0.65 | 63 | 12.4° |
| Bx | 192 | 184 | **+8.4 mm** | 0.35 | 126 | 10.4° |
| Cx | 172 | 182 | **−10.2 mm** | 0.68 | 65 | 9.9° |
| Dx | 184 | 184 | **+0.4 mm** | 0.62 | 71 | 11.1° |
| Ay | 172 | 199 | **−27 mm** | 0.51 | 80 | 15.6° |
| By | 159 | 199 | **−41 mm** | 0.46 | 96 | 11.7° |
| Cy | 161 | 199 | **−38 mm** | 0.57 | 78 | 10.9° |

Three concrete results:

1. **Tracks point back at the source** — measured tanθ vs position is monotonic
   with the source sign in all 8 planes (this alone confirms the inclined sample
   is real and the alignment is right to first order).
2. **Transverse (x) alignment is good to ~1 cm** (Ax +7.5, Bx +8.4, Cx −10.2,
   Dx +0.4 mm) — matching the "confident to a cm" prior. But the **beam-axis (y)
   planes are consistently ~3 cm LOW** (Ay −27, By −41, Cy −38 mm, all same
   sign). That is a real, systematic finding: either the chamber stack sits
   ~3 cm off in y, the beam spot is ~3 cm off centre in y, or the y-plane angle
   reconstruction carries a common bias. **Worth a hardware/beam-position check.**
3. **Drift calibration / angle scale.** The fitted slope is **0.62 ± of the
   ideal 1/R, consistent across A/C/D** (B is the outlier at 0.35 — its known
   wide-cluster/ringing pathology). Because A (600 V) and C/D (800 V) would need
   DIFFERENT v_drift but share the SAME 0.62, this is NOT a v_drift error — it is
   the ~40 % charge-sharing angle COMPRESSION of the raw anchored time-fit (the
   exact bias the bench regression estimator removes; bench raw fit 6.9° vs
   regression 1.8°). So: garfield v_drift is consistent with the data once the
   angle-scale bias is separated; the raw-fit angular resolution here is ~11°
   (inflated by the extended source's own ±0.085 tan ≈ 5° spread).

4. **The "bending as a function of position" = fringe-field EDGE distortion.**
   The residual (tanθ_meas − source fit) is flat (±0.05) through the fiducial and
   turns sharply UP (+0.15–0.20 tan ≈ +8–11°) at the high strip-position edge
   (>330 mm) in EVERY plane — the same shape in all four chambers ⇒ a real drift
   fringe field near the cage edge, not per-chamber noise. This is the bending
   seen when an hour of tracks was overlaid. It is mapped per plane in
   `calib/27_align.json` (`dist_pos`/`dist_dtan`); fiducializing to 70–330 mm
   removes it, and the map can be applied to recover edge tracks.

## Head-on tracks need the waveform (and are rare)

The combined_hits head-on proxy (`wfmax` = right−left sample) is ~80 %
contaminated: pulling the decoded LEADING-strip waveform (script 27w), only
**~20 %** of head-on-tagged clusters show a genuine full-gap single hump
(width ≥7 samples, single-peaked), vs **2 %** of wide-blob lead strips (the
junk control — cleanly rejected) and 36 % of inclined lead strips. Real head-on
humps are ~500 ns wide and peak ~1–2 k ADC (they do NOT saturate at run_55
gains, so pulse WIDTH not the saturation plateau is the discriminator). Blob
lead strips are flat ±300 ADC noise. **Conclusion: select head-on with
waveforms only; even then it is a minor sample. The inclined monotonic-trail
tracks are the reliable basis for physics and alignment.**

## Follow-up part 2 — regression, full waveforms, drift velocity, source location (2026-07-19)

Scripts: `27x_features.py` (bench hits6 features from combined_hits at 100 ADC —
the low-threshold neighbours the 400-ADC gate cache dropped), `27y_regression_align.py`
(apply the frozen bench regression restandardized), `27z_waveform_reco.py`
(decoded-waveform CFD reconstruction of all 2701 inclined tracks). Outputs
`calib/27_{features,regression,waveform}.npz/json`, `figures/27_tracks/09-13`.

**1. The frozen bench regression does NOT transfer to run_55 (negative result).**
Restandardized onto the run_55 inclined features, the bench |tan| regressor
(mx17_3/2/6/7 hits6, bench holdout σ 1.74–1.76°, sign-acc 0.95) collapses:
source-pointing slope → ~0 (scale 0.03–0.10 vs the raw 0.5–0.6). Cause is the
README-flagged caveat, now confirmed: the models were trained on COSMICS
spanning a broad angle range; run_55 source tracks are near-NORMAL incidence
(small, narrow tan), so restandardization mis-maps the feature scale and the
monotonic calibration squashes everything to one angle. **Frozen_rs transfer is
not usable for point-source beam tracks — the angle model must be RE-TRAINED
in situ (source-truth or an in-beam telescope).**

**2. Full-waveform CFD reconstruction (2701 tracks) — the ~0.6 compression is
REAL, not a timing artifact.** Reconstructing each strip's 30 %-CFD sub-sample
leading edge and refitting the trail gives the SAME source-pointing scale as the
60 ns max_sample fit (CFD 0.53–0.65 vs raw 0.62; B 0.36) and the SAME angular
resolution (σθ ≈ 11°, unchanged). So the angle compression is genuine
charge-sharing, and the ~11° single-track resolution is NOT limited by timing
quantisation — it is set by (a) the extended source's own ~5° spread, (b) the
small lever arm of near-normal tracks (little position extent), and (c) coherent
baseline noise corrupting the CFD of low-amplitude trail strips (the same CM
noise the ZS work flagged — [[july-zs-optimization]]). Sub-sample timing does
not help here; better S/N (ZS/CM correction) and in-situ angle training would.

**3. Drift velocity, telescope-free and angle-independent.** For a full-gap
track the drift-time SPAN (latest−earliest arrival) = gap/v regardless of angle.
From the CFD span (upper-half median, full 24 subruns):

| det | CFD span | v_cfd | garfield | note |
|---|---|---|---|---|
| A | 622 ns | **48** | 40.5 | dry gas, 600 V drift |
| B | 1005 ns | **30** | 44.1 | wide-cluster/ringing pathology — untrusted |
| C | 710 ns | **42** | 44.1 | **matches garfield** |
| D | 594 ns | **50** | 44.1 | high |

C sits on garfield (42 vs 44); A/D read ~10 % high and B low. Per-chamber
scatter is physical (A dry vs B/C/D ~0.8 % H2O, B's pathology, residual
span-truncation where the far-edge charge dips below threshold) rather than
method noise. **Net: v_drift ≈ 44 µm/ns (garfield) is consistent for the trusted
chamber C; a clean per-chamber v needs the in-situ-trained angle to fully
separate v from the charge-sharing scale.**

**4. Source LOCATION triangulated (and the y offset is PHYSICS).** The fitted
u0 per plane, minus strip centre, is the effective source position in that
coordinate, and the chambers agree:
- transverse: A −9, B −7, C −26, D −12 mm → ~1 cm off-axis, consistent (alignment).
- **beam/vertical (y): A −21, B −29, C −48 mm → the source is ~3 cm LOW, same in
  every chamber.** At n_TOF EAR2 the beam is vertical and neutrons enter the ³He
  capsule from below, so captures pile up at the BOTTOM of the D20×L40 capsule —
  the effective source really is low in y. The chambers independently
  triangulating the same ~3 cm offset is a genuine source-position measurement,
  NOT a y-misalignment to correct. (This resolves the "y-planes 3 cm low" puzzle
  from part 1.)

## Numbers to keep

- 24 subruns, 76 214 triggers → 110 115 candidate clusters, 1.06 M strips.
- 762 verified X/Y 3-D segments (b1/b2); charge balance 0.48–0.50 at bench width.
- Transverse alignment ≤1 cm (x); **beam-axis y offset ~3 cm, all chambers**.
- Raw angle compression ~0.62 (charge sharing), consistent A/C/D ⇒ garfield
  v_drift OK; B anomalous.
- Edge fringe-field distortion ≈ +10° beyond ~330 mm strip position, all planes.

## Caveats

- Everything is b1/b2 only (the 3 ms goal is unsampled — see HV-scan doc).
- The absolute angle SCALE and v_drift stay partly degenerate: the frozen bench
  regression that would have separated them does NOT transfer to near-normal
  point-source tracks (part 2 §1), so v is pinned via the angle-independent CFD
  drift-span instead (part 2 §3), trustworthy for C.
- The extended source (D20×L40 mm) adds an irreducible ~5° angular spread that
  inflates the single-track resolution; the ~3 cm-low y source position it
  implies is the physics of bottom-of-capsule capture (part 2 §4), not a bias.
- B is not trustworthy for the angle scale OR v_drift (wide-cluster/ringing
  pathology, flagged already in the HV-scan doc).
- B is not trustworthy for the angle scale (wide-cluster/ringing pathology,
  flagged already in the HV-scan doc).

## Follow-ups

1. **RE-TRAIN the angle regressor in situ** — the frozen bench transfer failed
   (part 2 §1). Train `microtpc_lib.train_tan_regression` on run_55 tracks with
   SOURCE-truth labels (tanθ = (u−u0)/R), holding out for the residual σθ; that
   both removes the charge-sharing compression and cleanly separates v_drift.
   (Partly circular for pointing, but the holdout σ and v are honest.)
2. **Improve S/N before precision** — the ~11° single-track resolution is
   coherent-CM-noise-limited on low-amp trail strips ([[july-zs-optimization]]);
   re-processing with CM correction / ZS should sharpen the trail and v_drift.
3. The ~3 cm-low y source position is bottom-of-capsule capture PHYSICS —
   cross-check against the plastic/SiPM imaging (wall geometry) which also sees
   y, and use it as the true source point in vertexing (don't "correct" it).
4. Apply the fringe-field distortion map (or fiducialize 70–330 mm) in any
   track-based alignment / e⁺e⁻ vertexing; edges bias angles ~10°.
5. Feed the 762-segment table (`calib/27_tracks.npz`) into inter-chamber linking
   / vertexing (ntof_tracking PLAN_04) — the (low-y) source is the truth anchor.
6. B wide-cluster/ringing diagnosis at the waveform level (also open from HV scan).

## Figures (`figures/27_tracks/`)

- `01_discriminators.png` — dur / monotonicity / head-on population per det (gate)
- `02_validation.png` — X/Y IoU & charge-balance real-vs-null, yield vs HV (KEY)
- `03_cascade.png` — survival cascade + source-pointing preview
- `04_align.png` — tanθ vs position + source fit, 8 planes (KEY)
- `05_distortion.png` — fringe-field residual map, 8 planes (the "bending")
- `06_summary.png` — angle-scale / v_true / alignment across planes (KEY)
- `07_headon_waveforms.png` — head-on vs inclined vs blob lead-strip waveforms
- `08_headon_features.png` — plateau/width; proxy vs true
- `09_regression.png` — frozen bench regression vs raw angle (transfer FAILS)
- `10_source_loc.png` — regression scale/σθ/v per plane (part 2 §1)
- `11_wf_examples.png` — waveform CFD micro-TPC trails + strip waveforms
- `12_wf_vdrift.png` — CFD drift-span → v_drift per chamber vs garfield (KEY)
